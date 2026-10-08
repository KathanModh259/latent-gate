"""
Fetch a public URL as text, safely enough to run on a shared server.

SSRF guard: only http(s), and the address we actually connected to must be a
public (globally routable) IP. The check runs on the live socket after connect,
before any request is sent, so DNS tricks that resolve to 127.0.0.1, 10.x or
169.254.169.254 (cloud metadata) are refused even if DNS changes between lookups.
Redirects are followed by hand so every hop gets the same check.
"""

import http.client
import ipaddress
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
TIMEOUT_S = 10
_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "+json", "+xml")


def _check_peer(sock) -> None:
    ip = ipaddress.ip_address(sock.getpeername()[0].split("%")[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if not ip.is_global:
        sock.close()
        raise PermissionError(f"Refusing to fetch a non-public address ({ip})")


class _GuardedHTTP(http.client.HTTPConnection):
    def connect(self):
        super().connect()
        _check_peer(self.sock)


class _GuardedHTTPS(http.client.HTTPSConnection):
    def connect(self):
        super().connect()
        _check_peer(self.sock)


class _TextExtractor(HTMLParser):
    """Visible text of an HTML page; one line per block element."""

    _SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "pre", "table", "ul", "ol", "header", "footer", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, visible text) of an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.text()


def fetch_text(url: str) -> dict:
    """GET a public URL; return {url, content_type, title, text}. Raises ValueError/PermissionError."""
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("Only http:// and https:// URLs can be fetched")
        cls = _GuardedHTTPS if parts.scheme == "https" else _GuardedHTTP
        conn = cls(parts.hostname, parts.port, timeout=TIMEOUT_S)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            conn.request("GET", target, headers={
                "User-Agent": "LatentGate (+https://pypi.org/project/latent-gate/)",
                "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Encoding": "identity",
            })
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308) and resp.getheader("Location"):
                url = urljoin(url, resp.getheader("Location"))
                continue
            if resp.status >= 400:
                raise ValueError(f"HTTP {resp.status} {resp.reason} from {url}")
            ctype = resp.getheader("Content-Type", "")
            if ctype and not any(t in ctype.lower() for t in _TEXT_TYPES):
                raise ValueError(f"Not a text response ({ctype.split(';')[0]})")
            body = resp.read(MAX_FETCH_BYTES + 1)
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, PermissionError):
                raise
            raise ValueError(f"Could not fetch {url}: {e}") from e
        finally:
            conn.close()
        if len(body) > MAX_FETCH_BYTES:
            raise ValueError(f"Response is over {MAX_FETCH_BYTES // 1048576}MB")
        charset = resp.headers.get_content_charset() or "utf-8"
        text = body.decode(charset, errors="replace")
        title = ""
        if "html" in ctype.lower() or text.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
            title, text = html_to_text(text)
        return {"url": url, "content_type": ctype.split(";")[0], "title": title, "text": text}
    raise ValueError(f"Too many redirects (>{MAX_REDIRECTS})")

