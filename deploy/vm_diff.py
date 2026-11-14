#!/usr/bin/env python3
"""Classify files for a safe VM deploy.

Inputs are manifests: text files with lines "<sha256>  <relative/path>" (the
output of `sha256sum`).  Three manifests take part:

  build  the tree we are about to deploy (git archive of the ref)
  vm     the tree on the VM right now
  last   the tree as it was at the last deploy (optional; missing = unknown)

Categories (see classify()):
  same              build == vm, nothing to do
  build_only        changed in the build only since last deploy  -> safe, deployed
  new_in_build      file exists only in the build                -> safe, deployed
  removed_in_build  removed in the build, VM copy untouched      -> safe, left in place
  vm_only_changed   VM copy edited since last deploy             -> DANGER
  conflict          changed on both sides since last deploy      -> DANGER
  vm_only_new       file exists only on the VM (not deployed)    -> DANGER
  vm_deleted        file deployed last time, deleted on the VM   -> DANGER
  unknown_differs   differs and there is no last manifest        -> DANGER

Files listed in `accepted` are explicitly acknowledged by the operator: a
DANGER file that is accepted is reported as `accepted` and does not block.
"""
import argparse
import json
import sys

SAME = "same"
BUILD_ONLY = "build_only"
NEW_IN_BUILD = "new_in_build"
REMOVED_IN_BUILD = "removed_in_build"
VM_ONLY_CHANGED = "vm_only_changed"
CONFLICT = "conflict"
VM_ONLY_NEW = "vm_only_new"
VM_DELETED = "vm_deleted"
UNKNOWN_DIFFERS = "unknown_differs"
ACCEPTED = "accepted"

DANGER = {VM_ONLY_CHANGED, CONFLICT, VM_ONLY_NEW, VM_DELETED, UNKNOWN_DIFFERS}

EXPLAIN = {
    VM_ONLY_CHANGED: "edited on the VM since the last deploy (a deploy would overwrite it)",
    CONFLICT: "changed on the VM AND in the build since the last deploy",
    VM_ONLY_NEW: "exists only on the VM, not in the build",
    VM_DELETED: "deployed last time but deleted on the VM",
    UNKNOWN_DIFFERS: "differs from the build and there is no record of the last deploy",
}


def parse_manifest(text):
    """Parse `sha256sum` output into {path: sha}. Blank lines ignored."""
    out = {}
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        sha, _, path = line.partition("  ")
        if not path:
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError("bad manifest line: %r" % line)
            sha, path = parts
        if path.startswith("./"):
            path = path[2:]
        out[path] = sha.strip()
    return out


def classify(build, vm, last=None, accepted=()):
    """Return {path: category}. `last` None means the last deploy is unknown."""
    accepted = set(accepted)
    result = {}
    for path in set(build) | set(vm) | set(last or {}):
        b, v = build.get(path), vm.get(path)
        l = None if last is None else last.get(path)
        if b is not None and v is not None:
            if b == v:
                cat = SAME
            elif last is None or l is None:
                cat = UNKNOWN_DIFFERS
            elif l == v:
                cat = BUILD_ONLY
            elif l == b:
                cat = VM_ONLY_CHANGED
            else:
                cat = CONFLICT
        elif b is not None:  # in build, not on VM
            if last is not None and l is not None:
                cat = VM_DELETED
            else:
                cat = NEW_IN_BUILD
        elif v is not None:  # on VM, not in build
            if last is not None and l is not None and l == v:
                cat = REMOVED_IN_BUILD
            elif last is not None and l is not None:
                cat = CONFLICT  # we dropped it, someone edited it on the VM
            else:
                cat = VM_ONLY_NEW
        else:
            continue  # only in last: gone on both sides
        if cat in DANGER and path in accepted:
            cat = ACCEPTED
        # an accepted file that is merely identical needs no acceptance
        result[path] = cat
    return result


def danger_paths(result):
    return sorted(p for p, c in result.items() if c in DANGER)


def summarize(result):
    counts = {}
    for c in result.values():
        counts[c] = counts.get(c, 0) + 1
    return counts


def pull_help(paths, last_ref, vm_host="USER@VM", app="/opt/salesgod/app"):
    """Ready-to-run shell text to pull the VM copies of `paths` into a branch."""
    q = " ".join("'%s'" % p for p in paths)
    lines = [
        "# 1. copy the VM versions here:",
        "mkdir -p vm-pull && ssh %s \"sudo tar czf - -C %s %s\" | tar xzf - -C vm-pull"
        % (vm_host, app, q),
        "# 2. put them on a branch (from the last deployed ref) so they are saved in git:",
        "git switch -c vm-work-$(date +%%Y%%m%%d) %s" % (last_ref or "<last-deployed-ref>"),
        "for f in %s; do mkdir -p \"$(dirname \"$f\")\"; cp \"vm-pull/$f\" \"$f\"; done"
        % q,
        "git add %s && git commit -m 'Capture work done directly on the VM'" % q,
        "# 3. then merge it into the branch you are deploying (git merge, or per file:",
        "#    git merge-file <ours> <base-at-last-deploy> vm-pull/<file>) and re-run the deploy.",
    ]
    return "\n".join(lines)


def _read(path):
    with open(path) as f:
        return parse_manifest(f.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--vm", required=True)
    ap.add_argument("--last", help="manifest of the last deployed ref (omit if unknown)")
    ap.add_argument("--accept", action="append", default=[], help="acknowledged path (repeatable)")
    ap.add_argument("--last-ref", default="")
    ap.add_argument("--vm-host", default="USER@VM")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-lists", help="dir to write build_only.txt, new_in_build.txt, accepted.txt")
    a = ap.parse_args(argv)

    build, vm = _read(a.build), _read(a.vm)
    last = _read(a.last) if a.last else None
    res = classify(build, vm, last, a.accept)
    bad = danger_paths(res)
    counts = summarize(res)

    if a.write_lists:
        import os
        os.makedirs(a.write_lists, exist_ok=True)
        for name, cats in (("changed.txt", {BUILD_ONLY, NEW_IN_BUILD}),
                           ("added.txt", {NEW_IN_BUILD}),
                           ("accepted.txt", {ACCEPTED}),
                           ("danger.txt", DANGER)):
            with open(os.path.join(a.write_lists, name), "w") as f:
                f.write("".join(p + "\n" for p, c in sorted(res.items()) if c in cats))

    if a.json:
        print(json.dumps({"counts": counts, "danger": bad,
                          "files": {p: c for p, c in res.items() if c != SAME}}, indent=1))
    else:
        print("last deploy: %s" % ("known (" + (a.last_ref or "manifest") + ")" if last is not None
                                   else "UNKNOWN (no DEPLOYED_MANIFEST on the VM yet)"))
        print("summary: " + (", ".join("%s=%d" % kv for kv in sorted(counts.items())) or "no files"))
        for cat in (BUILD_ONLY, NEW_IN_BUILD, REMOVED_IN_BUILD, ACCEPTED):
            ps = sorted(p for p, c in res.items() if c == cat)
            if ps:
                print("\n%s (%d):" % (cat, len(ps)))
                for p in ps:
                    print("  " + p)
        if bad:
            print("\nDANGER (%d) - deploy blocked:" % len(bad))
            for p in bad:
                print("  %-18s %s   <- %s" % (res[p], p, EXPLAIN[res[p]]))
            print("\nTo save the VM's versions into git first:")
            print(pull_help(bad, a.last_ref, a.vm_host))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
