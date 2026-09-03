"""
Tiny save endpoint for stance/report.html review controls.

The report is opened from file:// so its videos seek properly; this server only
receives the review state and writes it to stance/review.json.

    python stance_review_server.py          # listens on 127.0.0.1:8767

GET  /review.json  -> current review state
POST /review.json  -> replace review state with the posted JSON object
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REVIEW = Path(__file__).parent / "stance" / "review.json"
PORT = 8767


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        body = REVIEW.read_bytes() if REVIEW.exists() else b"{}"
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            assert isinstance(data, dict)
        except Exception:
            self.send_response(400); self._cors(); self.end_headers(); return
        REVIEW.parent.mkdir(exist_ok=True)
        REVIEW.write_text(json.dumps(data, indent=1))
        self.send_response(200); self._cors(); self.end_headers(); self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        if self.command == "POST":
            print(f"saved {REVIEW} ({args[0]})")


if __name__ == "__main__":
    print(f"review server on http://127.0.0.1:{PORT}  ->  {REVIEW}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
