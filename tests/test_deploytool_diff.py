import importlib.util
import pathlib
import subprocess
import sys

P = pathlib.Path(__file__).resolve().parent.parent / "deploy" / "vm_diff.py"
spec = importlib.util.spec_from_file_location("vm_diff", P)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_parse_manifest():
    m = d.parse_manifest("aa  ./x/y.py\nbb  z.txt\n\ncc  with space.txt\n")
    assert m == {"x/y.py": "aa", "z.txt": "bb", "with space.txt": "cc"}


def test_known_last_all_categories():
    last = {"same": "1", "bo": "1", "vo": "1", "both": "1", "gone": "1", "vdel": "1", "rem": "1", "remedit": "1"}
    build = {"same": "1", "bo": "2", "vo": "1", "both": "3", "new": "9", "vdel": "1"}
    vm = {"same": "1", "bo": "1", "vo": "5", "both": "4", "vonly": "7", "rem": "1", "remedit": "8"}
    r = d.classify(build, vm, last)
    assert r["same"] == d.SAME
    assert r["bo"] == d.BUILD_ONLY
    assert r["vo"] == d.VM_ONLY_CHANGED
    assert r["both"] == d.CONFLICT
    assert r["new"] == d.NEW_IN_BUILD
    assert r["vonly"] == d.VM_ONLY_NEW
    assert r["vdel"] == d.VM_DELETED
    assert r["rem"] == d.REMOVED_IN_BUILD
    assert r["remedit"] == d.CONFLICT
    assert "gone" not in r
    assert d.danger_paths(r) == ["both", "remedit", "vdel", "vo", "vonly"]


def test_unknown_last_requires_identical_or_accepted():
    build = {"a": "1", "b": "2", "c": "3"}
    vm = {"a": "1", "b": "9", "stray": "x"}
    r = d.classify(build, vm, None)
    assert r["a"] == d.SAME
    assert r["b"] == d.UNKNOWN_DIFFERS
    assert r["c"] == d.NEW_IN_BUILD
    assert r["stray"] == d.VM_ONLY_NEW
    assert d.danger_paths(r) == ["b", "stray"]
    r2 = d.classify(build, vm, None, accepted=["b", "stray"])
    assert d.danger_paths(r2) == []
    assert r2["b"] == d.ACCEPTED


def test_identical_trees_clean():
    t = {"a": "1", "b": "2"}
    r = d.classify(t, dict(t), None)
    assert set(r.values()) == {d.SAME}
    assert d.danger_paths(r) == []


def test_accept_does_not_hide_other_danger():
    r = d.classify({"a": "1"}, {"a": "2", "b": "3"}, {"a": "1"}, accepted=["b"])
    assert r["a"] == d.VM_ONLY_CHANGED and r["b"] == d.ACCEPTED
    assert d.danger_paths(r) == ["a"]


def test_cli_exit_codes(tmp_path):
    (tmp_path / "b").write_text("1  a\n")
    (tmp_path / "v").write_text("2  a\n")
    cmd = [sys.executable, str(P), "--build", str(tmp_path / "b"), "--vm", str(tmp_path / "v")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 2 and "DANGER" in p.stdout and "tar czf" in p.stdout
    p = subprocess.run(cmd + ["--accept", "a"], capture_output=True, text=True)
    assert p.returncode == 0


def test_deploy_script_syntax():
    sh = P.parent / "deploy_vm.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    p = subprocess.run([str(sh)], capture_output=True, text=True)
    assert p.returncode == 64 and "usage" in p.stderr
