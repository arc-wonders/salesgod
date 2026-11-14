"""deploy/data_sync.py: deploys merge data/ add-only and refuse when a product would lose its negotiation range.

15 Nov: the deploy of 3e6c1f9 shipped code only; the live products.json kept the old FitZone without economics, so
live haggling was dead ("fixed at ₹2,999").
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
import data_sync  # noqa: E402

ECO = {"currency": "₹", "unit": "3 months", "list_price": 2999, "cost": 2000, "min_margin_pct": 20, "ladder": []}


def _app(tmp: Path, live: dict, seeds: bool = False) -> tuple[Path, Path]:
    app = tmp / "app"
    shutil.copytree(ROOT / "acie", app / "acie")
    (app / "deploy").mkdir(parents=True)
    shutil.copy2(ROOT / "deploy" / "merge_products.py", app / "deploy" / "merge_products.py")
    (app / "data" / "voice").mkdir(parents=True)
    (app / "data" / "voice" / "products.json").write_text(json.dumps(live), encoding="utf-8")
    bd = tmp / "build" / "data" / "voice"
    bd.mkdir(parents=True)
    if seeds:
        (bd / "knowledge" / "default").mkdir(parents=True)
        (bd / "knowledge" / "default" / "gym.json").write_text("{}", encoding="utf-8")
    return app, tmp / "build" / "data"


def _gym(**kw):
    d = {"id": "fitzone-gym", "product_name": "FitZone membership", "seller_name": "FitZone",
         "offer_price": "₹2,999 for 3 months", "original_price": "₹6,000 for 3 months"}
    d.update(kw)
    return d


def test_adds_missing_economics_and_keeps_console_edits(tmp_path):
    live = {"fitzone-gym": _gym(tone="console-edited tone")}
    app, bdata = _app(tmp_path, live, seeds=True)
    (bdata / "voice" / "products.json").write_text(json.dumps({"fitzone-gym": _gym(tone="build tone", economics=ECO, tone_level="mid")}))
    assert data_sync.run(app, bdata, dry=False) == 0
    out = json.loads((app / "data" / "voice" / "products.json").read_text())["fitzone-gym"]
    assert out["tone"] == "console-edited tone"                 # console edit kept
    assert out["economics"]["list_price"] == 2999 and out["tone_level"] == "mid"   # missing keys added
    assert (app / "data" / "voice" / "knowledge" / "default" / "gym.json").exists()  # seed copied
    backups = list((app / "data" / "voice").glob(".sync-backup-*"))
    assert backups and (backups[0] / "products.json").exists()
    assert data_sync.restore(app, backups[0]) == 0
    restored = json.loads((app / "data" / "voice" / "products.json").read_text())["fitzone-gym"]
    assert "economics" not in restored and not (app / "data" / "voice" / "knowledge" / "default" / "gym.json").exists()


def test_never_overwrites_existing_seed_files(tmp_path):
    app, bdata = _app(tmp_path, {"fitzone-gym": _gym(economics=ECO)}, seeds=True)
    live_kb = app / "data" / "voice" / "knowledge" / "default" / "gym.json"
    live_kb.parent.mkdir(parents=True)
    live_kb.write_text('{"live": true}')
    (bdata / "voice" / "products.json").write_text(json.dumps({"fitzone-gym": _gym(economics=ECO)}))
    assert data_sync.run(app, bdata, dry=False) == 0
    assert json.loads(live_kb.read_text()) == {"live": True}


def test_fails_when_live_economics_is_unusable(tmp_path):
    bad = dict(ECO, list_price=0)  # a console edit broke the range: the build has one, live would not -> refuse
    app, bdata = _app(tmp_path, {"fitzone-gym": _gym(economics=bad)})
    (bdata / "voice" / "products.json").write_text(json.dumps({"fitzone-gym": _gym(economics=ECO)}))
    before = (app / "data" / "voice" / "products.json").read_text()
    assert data_sync.run(app, bdata, dry=False) == 2
    assert (app / "data" / "voice" / "products.json").read_text() == before  # nothing written


def test_dry_run_writes_nothing(tmp_path):
    app, bdata = _app(tmp_path, {"fitzone-gym": _gym()})
    (bdata / "voice" / "products.json").write_text(json.dumps({"fitzone-gym": _gym(economics=ECO)}))
    before = (app / "data" / "voice" / "products.json").read_text()
    assert data_sync.run(app, bdata, dry=True) == 0
    assert (app / "data" / "voice" / "products.json").read_text() == before


def test_deploy_script_syntax_and_wiring():
    sh = (ROOT / "deploy" / "deploy_vm.sh").read_text()
    assert "data_sync.py" in sh and "--restore" in sh and "data.tar" in sh
    assert subprocess.run(["bash", "-n", str(ROOT / "deploy" / "deploy_vm.sh")]).returncode == 0
