"""
gateway/webui_server.py — Static File Server for Web UI

Serves the NeuralClaw Web UI static files (HTML, CSS, JS) using Python's
built-in http.server. Runs alongside the WebSocket gateway on a separate port.

Zero external dependencies — uses stdlib only.
"""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from functools import partial

from observability.logger import get_logger

log = get_logger(__name__)

# Ensure JS files get the right MIME type
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class WebUIHandler(BaseHTTPRequestHandler):
    """Simple static file HTTP handler for the Web UI."""

    webui_dir: Path = Path(__file__).parent.parent / "webui"

    def do_GET(self):
        # Serve index.html for root
        path = self.path.split("?")[0].strip("/")
        if not path:
            path = "index.html"

        file_path = self.webui_dir / path

        # Security: prevent directory traversal
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(self.webui_dir.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
        except (ValueError, OSError):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        if not file_path.is_file():
            # Serve index.html for SPA routing
            file_path = self.webui_dir / "index.html"

        try:
            content = file_path.read_bytes()
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            # CORS headers for WebSocket connection from different port
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        # Suppress default HTTP logging — use structlog instead
        pass


async def start_webui_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    webui_dir: Path | None = None,
) -> asyncio.AbstractServer:
    """
    Start the Web UI static file server.

    Returns the asyncio server so it can be properly closed.
    """
    if webui_dir:
        WebUIHandler.webui_dir = Path(webui_dir)

    handler_factory = partial(WebUIHandler)

    loop = asyncio.get_event_loop()
    server = await loop.create_server(
        lambda: asyncio.Protocol(),  # placeholder — replaced below
        host,
        port,
    )
    # Close the placeholder and use the threaded approach
    server.close()
    await server.wait_closed()

    # Use a simple threaded HTTP server since http.server is synchronous
    import threading
    from http.server import HTTPServer

    httpd = HTTPServer((host, port), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    log.info("webui.http_started", host=host, port=port)
    return httpd
