# Deploying SalesGod to the production VM

Always deploy with `deploy/deploy_vm.sh`. Never `tar x` or `scp` over `/opt/salesgod/app` by hand:
on 14 Nov a hand deploy overwrote a console UI that existed only on the VM.

```
deploy/deploy_vm.sh <git-ref> --dry-run     # look first; changes nothing
deploy/deploy_vm.sh <git-ref>               # real deploy
deploy/deploy_vm.sh --status                # deployed ref + VM-only drift right now
deploy/deploy_vm.sh <git-ref> --seed        # record DEPLOYED_REF only (VM must already equal the ref)
```
Options: `--accept PATH` (repeatable; you confirm this VM-only file may be discarded/ignored),
`--force-overwrite-vm-only` (override every DANGER; the VM copies still go into the backup).
Env: `DEPLOY_HOST` (default `lakshay@34.143.141.218`), `LIVE_TIMEOUT` (600 s), `READY_TIMEOUT` (300 s).

## What it does
1. `git archive <ref>` (without `data/`), computes a manifest (sha256 per file).
2. Reads the VM tree manifest (`sudo find` + `sha256sum` under `/opt/salesgod/app`, skipping `data/`,
   `__pycache__`, `*.pyc`, `*.egg-info`, `.pytest_cache`) and the record of the last deploy:
   `/opt/salesgod/DEPLOYED_REF` and `/opt/salesgod/DEPLOYED_MANIFEST`.
3. `deploy/vm_diff.py` classifies every file:

| class | meaning | effect |
|---|---|---|
| same | build == VM | nothing |
| build_only / new_in_build | changed or added in the build only | deployed |
| removed_in_build | dropped from the build, VM copy untouched | left in place |
| vm_only_changed | VM copy edited since the last deploy | DANGER |
| conflict | changed on both sides | DANGER |
| vm_only_new | file exists only on the VM | DANGER |
| vm_deleted | deployed before, deleted on the VM | DANGER |
| unknown_differs | differs and there is no last-deploy record | DANGER |

   Any DANGER aborts with exit 2, the list, and a ready `ssh ... tar` command to pull those files into a
   git branch (then `git merge`, or `git merge-file <ours> <base> <vm-copy>` per file) before you retry.
4. Waits until no call is live: the last `WebSocket /plivo/stream ... [accepted]` in the journal must be
   followed by `POST /plivo/hangup` (an accept older than 30 min with no hangup is treated as dead), and no
   `/api/sim` request in the last 60 s. Timeout aborts with nothing changed. It re-checks just before restart.
5. Backs up the app (without `data/`) to `/opt/salesgod/backup-app-<UTC>-<shortref>.tgz`, uploads the
   archive, extracts it as user `salesgod`. `data/` is never in the archive, so it is never touched.
6. Restarts `salesgod-voice`, waits for the `ready:` log line, checks `/` is 200 with a
   `SalesGod Console` title and `/classic` is 200, and greps the journal for `Traceback` since the restart.
   Any failure restores the backup (deleting files the deploy added), restarts, and exits non-zero.
7. On success writes `DEPLOYED_REF`, `DEPLOYED_MANIFEST` (the VM's actual tree after the deploy) and appends
   to `DEPLOYED_LOG`, then prints a summary. Secrets (`/opt/salesgod/.env`) are never read.

## First run (no DEPLOYED_REF yet)
With no record, the tool treats the VM as unknown: every file that differs from the ref, and every VM-only
file, blocks the deploy unless identical or `--accept`ed. If `--dry-run` reports only `same`, the VM already
equals the ref; either `--seed` it (records the ref, changes no files, no restart) or just deploy, which
seeds it on success. From then on drift is measured against the last deploy.

## When it aborts
Pull the listed files (command is printed), commit them on a branch off the last deployed ref, merge into
the branch you deploy, re-run `--dry-run`. Use `--accept` only for junk (e.g. a stray empty file).

## Manual rollback
`sudo -u salesgod tar xzf /opt/salesgod/backup-app-<...>.tgz -C /opt/salesgod && sudo systemctl restart salesgod-voice`
