"""Tiny stub server for the console UI: serves voice/static and fake JSON for every endpoint the UI uses.
Run: python tests/ui_stub.py [port] [--missing]   (--missing makes the new CRM/ai endpoints answer 404)"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).resolve().parents[1] / "voice" / "static"
NEW = ("/api/projects", "/api/calltypes", "/api/customers", "/api/campaigns", "/api/followups", "/api/ai/assist", "/api/leads")

PROJECTS = [{"id": "fitzone", "name": "FitZone Gym", "color": "#4f7cff", "languages": ["en", "hinglish"], "timezone": "Asia/Kolkata", "calling_hours": {"start": "10:00", "end": "19:00"}}]
PRODUCTS = [{"id": "fitzone-gym", "tenant": "fitzone", "project_id": "fitzone", "product_name": "FitZone Gym", "seller_name": "FitZone", "agent_name": "Arjun", "offer_price": "2,799", "discount": "30% off", "original_price": "3,999", "allowed_numbers": [], "tone_level": "mid"}]
CTS = [{"id": "sales_pitch", "name": "Sales pitch", "builtin": True, "objective": "Sell the product", "steps": [{"id": "s1", "text": {"en": "Hello"}}], "selling_allowed": True},
       {"id": "follow_up", "name": "Follow-up", "builtin": True, "objective": "Continue the last call", "steps": []},
       {"id": "mine", "name": "Gym trial booking", "objective": "Book a trial", "opening": {"en": "Hi, this is Arjun."}, "steps": [{"id": "q1", "text": {"en": "What is your goal?", "hi": "Aapka goal?"}, "answer_type": "text", "required": True}, {"id": "q2", "text": {"en": "When can you come?"}, "answer_type": "date"}], "allowed_actions": ["book_demo"], "outcomes": ["demo_booked"]}]
CUST = [{"id": "c1", "name": "Asha Rao", "phone": "+919876543210", "stage": "interested", "score": 72, "tags": ["gym", "pune"], "language": "hinglish", "city": "Pune", "next_action": {"type": "follow_up", "due_at": "2026-11-16T10:00:00", "call_type": "follow_up"}, "history": {"agreed_price": "2,799", "objections": ["price"]}, "custom": {"budget": "3000"}},
        {"id": "c2", "name": "Rahul Mehta", "phone": "+919999911111", "stage": "new", "score": 20, "tags": [], "language": "en"},
        {"id": "c3", "name": "Sana Khan", "phone": "+918888822222", "stage": "negotiating", "score": 55, "tags": ["gym"], "dnc": False}]
CAMPS = [{"id": "launch", "name": "Launch week", "product_id": "fitzone-gym", "call_type_id": "mine", "status": "draft", "dial_mode": "manual", "language": "hinglish", "stats": {"called": 3, "connected": 1, "queued": 12}, "audience": {"stages": ["new"]}, "schedule": {"start": "10:00", "end": "19:00", "days": ["Mon", "Tue"]}}]
FOLLOW = [{"customer": CUST[0], "due_at": "2026-11-16T10:00:00", "reason": "Asked for a callback", "next_action": {"type": "follow_up", "call_type": "follow_up"}}]
CALLS = [{"call_id": "k1", "to": "+919876543210", "customer_name": "Asha Rao", "campaign": "fitzone-gym", "project_id": "fitzone", "status": "ended", "created_at": 1763200000, "turns": []}]


def route(method, path, q, body, missing):
    if missing and path.startswith(NEW):
        return 404, {"detail": "Not Found"}
    if path == "/api/me": return 200, {"authed": True, "from_number": "+910000000000"}
    if path == "/api/tenants": return 200, {"tenants": [{"id": "fitzone", "name": "FitZone Gym", "color": "#4f7cff", "products": ["fitzone-gym"]}]}
    if path == "/api/projects": return 200, ({"projects": PROJECTS} if method == "GET" else body)
    if path.startswith("/api/projects/") and path.endswith("/fields"): return 200, {"fields": [{"key": "budget", "label": "Budget", "type": "number"}]}
    if path.startswith("/api/projects/"): return 200, body or PROJECTS[0]
    if path == "/api/products": return 200, {"products": PRODUCTS}
    if path == "/api/calls": return 200, {"calls": CALLS} if method == "GET" else {"calls": [{"call_id": "k2"}]}
    if path == "/api/tree/gates": return 200, {"gates": [{"id": "safety", "title": "Safety", "rule": "r", "intents": []}]}
    if path == "/api/calltypes": return 200, {"calltypes": CTS}
    if path.startswith("/api/calltypes/"): return 200, {"calltype": CTS[2], "id": "copy"}
    if path == "/api/customers": return 200, {"customers": CUST, "total": len(CUST), "page": 1}
    if path == "/api/customers/import": return 200, {"valid": 2, "duplicates": 1, "dnc": 0, "invalid": 0, "imported": 2}
    if path.endswith("/timeline"): return 200, {"timeline": [{"type": "call", "call_id": "k1", "at": "2026-11-14T10:00:00", "outcome": "callback_requested", "call_type": "sales_pitch", "summary": "Asked to call back Monday"}, {"type": "note", "text": "Prefers evenings", "at": "2026-11-14T11:00:00"}]}
    if path.startswith("/api/customers/"): return 200, CUST[0]
    if path == "/api/campaigns": return 200, {"campaigns": CAMPS}
    if path.endswith("/queue"): return 200, {"queue": [{"customer_id": "c2", "name": "Rahul Mehta", "phone": "+919999911111", "reason": "New lead, never called", "position": 1}]}
    if path.startswith("/api/campaigns/"): return 200, {"ok": True}
    if path == "/api/followups": return 200, {"followups": FOLLOW}
    if path == "/api/ai/assist": return 200, {"draft": {"name": "Reminder call", "objective": "Remind kindly", "selling_allowed": False, "outcomes": ["paid"]}}
    if path == "/api/progress": return 200, {"roadmap": {"tests": 1, "updated": "x", "overall_pct": 50, "areas": [], "log": []}, "stats": {"calls": 1}}
    if path == "/api/sim/start": return 200, {"sim_id": "s1", "agent": "Hello, this is Arjun from FitZone."}
    if path.startswith("/api/sim/"): return 200, {"agent": "Sure, 2,799 for the year.", "intent": "price", "trace": {"gate": "safety"}}
    return 404, {"detail": "Not Found"}


def make_handler(missing):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _do(self, method):
            u = urlparse(self.path); n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if method == "GET" and u.path in ("/", "/classic"):
                return self._send(200, (STATIC / "console.html").read_bytes(), "text/html")
            if method == "GET" and u.path.startswith("/ui/"):
                f = STATIC / u.path[4:]
                return self._send(200, f.read_bytes(), "text/javascript") if f.is_file() else self._send(404, b"{}", "application/json")
            if method == "GET" and u.path == "/productgen":
                return self._send(200, (STATIC / "productgen.html").read_bytes(), "text/html")
            if u.path == "/api/events":
                return self._send(200, b": hi\n\n", "text/event-stream")
            code, data = route(method, u.path, parse_qs(u.query), body, missing)
            self._send(code, json.dumps(data).encode(), "application/json")

        def _send(self, code, b, ct):
            self.send_response(code); self.send_header("content-type", ct); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self): self._do("GET")
        def do_POST(self): self._do("POST")
        def do_PUT(self): self._do("PUT")
        def do_DELETE(self): self._do("DELETE")
    return H


def serve(port=0, missing=False):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(missing))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8765
    srv = serve(port, "--missing" in sys.argv); print("stub on", port); threading.Event().wait()
