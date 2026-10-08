"""fetch_url_optimized: HTML extraction and the SSRF guard (no internet needed)."""

import http.server
import socketserver
import threading

import pytest

from latent_gate.web_fetch import fetch_text, html_to_text


def test_html_to_text_keeps_visible_text_only():
    title, text = html_to_text(
        "<html><head><title> My  Page </title><style>x{}</style></head>"
        "<body><p>Hello <b>world</b> &amp; co</p><script>evil()</script><li>a</li></body></html>"
    )
    assert title == "My Page"
    assert text == "Hello world & co\na"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/", "gopher://x/", "nohost"])
def test_rejects_non_http_schemes(url):
    with pytest.raises(ValueError):
        fetch_text(url)


def test_refuses_private_addresses_even_when_reachable():
    srv = socketserver.TCPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        for url in (f"http://127.0.0.1:{port}/", f"http://localhost:{port}/"):
            with pytest.raises(PermissionError, match="non-public"):
                fetch_text(url)
    finally:
        srv.shutdown()
        srv.server_close()
