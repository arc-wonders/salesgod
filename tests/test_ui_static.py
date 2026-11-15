"""Console UI: static files parse, every API path the new UI calls is in the platform spec (or an existing endpoint),
and (when headless Chromium exists) every page renders against a stub server, also with the new APIs missing."""
import html
import importlib.util
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "voice" / "static"
_spec = importlib.util.spec_from_file_location("ui_stub", Path(__file__).with_name("ui_stub.py"))
ui_stub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ui_stub)

# PLATFORM_SPEC.md "Backend API" + endpoints the console already used before this work
SPEC = ["/api/projects", "/api/projects/*", "/api/projects/*/fields", "/api/projects/*/fields/*", "/api/calltypes", "/api/calltypes/*", "/api/calltypes/*/duplicate",
        "/api/customers", "/api/customers/*", "/api/customers/import", "/api/leads/webhook/*", "/api/customers/*/timeline",
        "/api/campaigns", "/api/campaigns/*", "/api/campaigns/*/start", "/api/campaigns/*/pause", "/api/campaigns/*/queue", "/api/campaigns/*/call/*", "/api/customers/*/call",
        "/api/followups", "/api/ai/assist"]
EXISTING = ["/api/tenants", "/api/tenants/*", "/api/products", "/api/agents", "/api/agents/*", "/api/agents-default", "/api/agents-default/*", "/api/calls", "/api/sim/start", "/api/sim/*", "/api/progress", "/productgen"]


def used_paths(text: str) -> set[str]:
    out = set()
    for m in re.findall(r"[`'\"](/api/[^`'\"\s]*)", text):
        p = re.sub(r"\$\{[^}]*\}", "*", m.split("?")[0]).rstrip("/")
        out.add(p)
    return out


def test_platform_js_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    assert subprocess.run([node, "--check", str(STATIC / "platform.js")], capture_output=True, text=True).returncode == 0
    scripts = re.findall(r"<script>(.*?)</script>", (STATIC / "console.html").read_text(encoding="utf-8"), re.S)
    tmp = ROOT / "data" / "_ui_check.js"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(scripts[0], encoding="utf-8")
    try:
        r = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        tmp.unlink(missing_ok=True)


def test_html_parses_and_loads_platform():
    tags = ("script", "style", "div", "body", "head", "html")

    class P(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack = []; self.srcs = []

        def handle_starttag(self, tag, attrs):
            if tag in tags:
                self.stack.append(tag)
            if tag == "script":
                self.srcs.append(dict(attrs).get("src"))

        def handle_endtag(self, tag):
            if tag in tags:
                assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}>"
                self.stack.pop()
    for name in ("console.html", "productgen.html"):
        p = P(); p.feed((STATIC / name).read_text(encoding="utf-8")); assert not p.stack, name
        if name == "console.html":
            assert "/ui/platform.js" in p.srcs


def test_every_api_path_is_in_the_spec():
    pats = [re.compile("^" + re.escape(s).replace(r"\*", "[^/]+") + "$") for s in SPEC + EXISTING]
    paths = used_paths((STATIC / "platform.js").read_text(encoding="utf-8"))
    assert {"/api/ai/assist", "/api/customers/import", "/api/followups", "/api/campaigns"} <= paths
    bad = sorted(p for p in paths if not any(r.match(p) for r in pats))
    assert not bad, f"API paths not in the spec: {bad}"
    spec_file = ROOT.parent / "PLATFORM_SPEC.md"
    if spec_file.exists():
        t = spec_file.read_text(encoding="utf-8")
        for needle in ("/api/ai/assist", "/api/customers/import", "/api/followups", "/api/campaigns", "/api/calltypes", "/api/projects"):
            assert needle in t


def test_tone_dropdown_exposed():
    js = (STATIC / "platform.js").read_text(encoding="utf-8"); html = (STATIC / "console.html").read_text(encoding="utf-8")
    for s in ("professional", "casual", "Friendly (mid)", "Casual (non-professional)", "tone_level", "Sir, yeh hamari sabse achhi offer hai.", "Bhai, 2,799 kar deta hoon, aaj hi le lo?"):
        assert s in js, s
    assert "tone_level" in html and "toneField" in html


def test_server_route_serves_platform_js():
    assert '"/ui/{name}"' in (ROOT / "voice" / "server.py").read_text(encoding="utf-8")


CHROME = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
PAGES = {"dashboard": "Dashboard", "products": "Products", "calltypes": "Call types", "calltypes/mine": "Gym trial booking", "customers": "Customers & Leads", "customers/c1": "Asha Rao",
         "campaigns": "Campaigns", "campaigns/new": "New campaign", "followups": "Follow-ups", "settings": "Settings", "onboarding": "Set up a project"}


def dump(port, route):
    r = subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=5000", "--dump-dom", f"http://127.0.0.1:{port}/#/{route}"], capture_output=True, text=True, timeout=60)
    return html.unescape(re.sub(r"<script.*?</script>|<style.*?</style>", "", r.stdout, flags=re.S))


@pytest.mark.skipif(not CHROME, reason="no headless Chromium")
@pytest.mark.parametrize("missing", [False, True])
def test_pages_render_against_stub(missing):
    srv = ui_stub.serve(0, missing)
    try:
        port = srv.server_address[1]
        for route, title in PAGES.items():
            if missing and route in ("calltypes/mine", "customers/c1", "campaigns/new"):
                continue
            dom = dump(port, route)
            assert "interface module did not load" not in dom, route
            assert title in dom, f"{route}: '{title}' not rendered"
            if missing and route in ("customers", "campaigns", "followups"):
                assert "not available yet" in dom, route
    finally:
        srv.shutdown()
