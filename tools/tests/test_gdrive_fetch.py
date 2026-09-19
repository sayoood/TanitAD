"""Tests for `tools/gdrive_fetch.py` — the Drive domain allowlist, enforced.

Why these tests exist
---------------------
The allowlist is the whole point of the module. The input to `gdrive_fetch` is a
share URL, and share URLs arrive pasted out of docs, chat and other agents'
reports — so "follow whatever `Location` comes back" would turn a pasted string
into arbitrary outbound egress. `test_a_redirect_off_the_allowlist_is_refused`
is the test that makes that a property of the code instead of a comment.

The second thing under test is the *shape of the negative*. A probe that answers
a different question than the one asked is worse than no probe (`CLAUDE.md`, the
`df` trap and its three costumes). Here that means: an HTTP 404 for a bogus file
id is a **healthy** answer — the host served us — and only a proxy policy denial
counts as blocked. `test_404_is_reachable_not_blocked` pins that down, because
reading a 404 as "Drive is blocked" is exactly the mistake available.

Everything here runs offline against a fake opener. No test touches the network.
"""

from __future__ import annotations

import email.message
import hashlib
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdrive_fetch as gf                                        # noqa: E402

# A realistically shaped Drive id. Using a short stub here would quietly dodge
# the length guard in `file_id_from`, so the fetch tests would stop exercising
# the parse path they are supposed to go through.
FID = "1AbC_dEfG-hIjKlMnOpQrStUvWxYz"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

def _headers(**kw) -> email.message.Message:
    m = email.message.Message()
    for k, v in kw.items():
        m[k.replace("_", "-")] = v
    return m


class FakeResponse:
    def __init__(self, body: bytes = b"", *, content_type: str = "application/octet-stream",
                 status: int = 200, url: str = "https://drive.usercontent.google.com/x"):
        self._buf = body
        self.headers = _headers(Content_Type=content_type)
        self.status = status
        self._url = url

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            out, self._buf = self._buf, b""
            return out
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def geturl(self) -> str:
        return self._url


class FakeOpener:
    """Maps URL -> FakeResponse | HTTPError | Exception, and records the walk."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.seen: list[str] = []

    def open(self, url, timeout=None):                            # noqa: D102
        self.seen.append(url)
        try:
            r = self.routes[url]
        except KeyError:                                          # pragma: no cover
            raise AssertionError(f"unexpected URL requested: {url}") from None
        if isinstance(r, BaseException):
            raise r
        return r


def _redirect(to: str, code: int = 303) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://drive.google.com/uc", code, "See Other",
                                  _headers(Location=to), None)


# --------------------------------------------------------------------------
# host matching — the dot boundary is the bug that writes itself
# --------------------------------------------------------------------------

def test_the_three_documented_hosts_are_the_allowlist():
    assert gf.ALLOWED_HOSTS == ("drive.google.com",
                                "drive.usercontent.google.com",
                                "*.googleusercontent.com")


@pytest.mark.parametrize("host", [
    "drive.google.com",
    "drive.usercontent.google.com",
    "lh3.googleusercontent.com",
    "drive-thirdparty.googleusercontent.com",
    "doc-0g-88-docs.googleusercontent.com",
    "a.b.googleusercontent.com",                 # wildcard is any depth, by design
])
def test_allowlisted_hosts_pass(host):
    assert gf.host_allowed(host)


@pytest.mark.parametrize("host", [
    "notgoogleusercontent.com",                  # the classic bare-endswith bug
    "evilgoogleusercontent.com",
    "googleusercontent.com.attacker.test",
    "googleusercontent.com",                     # the bare apex is NOT in the set
    "google.com",
    "docs.google.com",                           # a real Google host, still not ours
    "drive.google.com.attacker.test",
    "",
    None,
])
def test_non_allowlisted_hosts_are_refused(host):
    assert not gf.host_allowed(host)


@pytest.mark.parametrize("host", [
    "Drive.Google.COM", "drive.google.com:443", "drive.google.com.",
    "DRIVE.GOOGLE.COM:443.",
])
def test_case_port_and_trailing_dot_are_all_the_same_host(host):
    """Two spellings of one host must not be two different answers."""
    assert gf.host_allowed(host)


def test_plain_http_is_refused_even_on_an_allowlisted_host():
    """An http:// hop leaves the proxy's TLS path; Drive never needs it."""
    assert gf.url_allowed("https://drive.google.com/uc?id=x")
    assert not gf.url_allowed("http://drive.google.com/uc?id=x")
    assert not gf.url_allowed("ftp://drive.google.com/x")


# --------------------------------------------------------------------------
# URL shapes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://drive.google.com/file/d/1AbC_dEfG-hIj/view?usp=sharing", "1AbC_dEfG-hIj"),
    ("https://drive.google.com/open?id=1AbC_dEfG-hIj", "1AbC_dEfG-hIj"),
    ("https://drive.google.com/uc?export=download&id=1AbC_dEfG-hIj", "1AbC_dEfG-hIj"),
    ("https://drive.usercontent.google.com/download?id=1AbC_dEfG-hIj&export=download",
     "1AbC_dEfG-hIj"),
    ("1AbC_dEfG-hIj", "1AbC_dEfG-hIj"),
])
def test_every_share_link_shape_yields_the_file_id(url, expected):
    assert gf.file_id_from(url) == expected


def test_a_url_on_another_host_is_refused_before_any_request():
    """The allowlist check happens at parse time, not after a request goes out."""
    with pytest.raises(gf.BlockedHost):
        gf.file_id_from("https://evil.test/file/d/1AbC_dEfG-hIj/view")


def test_a_drive_url_with_no_id_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        gf.file_id_from("https://drive.google.com/drive/my-drive")
    with pytest.raises(ValueError):
        gf.file_id_from("")


@pytest.mark.parametrize("junk", ["ep.pt", "id", "a b c", "../etc/passwd", "1AbC/d"])
def test_a_bare_string_that_is_not_id_shaped_is_rejected(junk):
    """Real Drive ids are long and [A-Za-z0-9_-] only.

    Without this guard any stray argument -- a filename, a half-pasted path --
    becomes a request for a nonexistent file, and the 404 that comes back reads
    like "the file is gone" rather than "you passed the wrong thing".
    """
    with pytest.raises(ValueError):
        gf.file_id_from(junk)


def test_download_url_targets_the_bytes_host_not_the_entry_point():
    """MEASURED: drive.google.com/uc answers 303 and serves no bytes."""
    u = gf.download_url("1AbC_dEfG-hIj")
    assert u.startswith("https://drive.usercontent.google.com/download?")
    assert "id=1AbC_dEfG-hIj" in u and "export=download" in u


# --------------------------------------------------------------------------
# THE SECURITY PROPERTY
# --------------------------------------------------------------------------

def test_a_redirect_that_stays_on_the_allowlist_is_followed():
    start = gf.download_url("ID")
    hop = "https://lh3.googleusercontent.com/blob"
    op = FakeOpener({start: _redirect(hop), hop: FakeResponse(b"bytes")})
    assert gf.open_checked(start, opener=op).read() == b"bytes"
    assert op.seen == [start, hop]


def test_a_redirect_off_the_allowlist_is_refused():
    """A pasted share URL must not become arbitrary egress."""
    start = gf.download_url("ID")
    op = FakeOpener({start: _redirect("https://evil.test/payload")})
    with pytest.raises(gf.BlockedHost, match="NOT followed"):
        gf.open_checked(start, opener=op)
    assert op.seen == [start]                     # the second hop never went out


def test_a_sign_in_redirect_says_the_file_is_private_not_that_the_allowlist_is_wrong():
    """MEASURED 2026-08-08: unauthenticated Drive answers 302 -> accounts.google.com.

    The refusal is correct, but the *message* decides what happens next. Read as
    "allowlist too narrow" someone widens the allowlist; read as "not shared"
    they fix the sharing. Only the second is right.
    """
    start = gf.download_url(FID)
    op = FakeOpener({start: _redirect(
        "https://accounts.google.com/ServiceLogin?service=wise", 302)})
    with pytest.raises(gf.BlockedHost, match="not shared\npublicly|not shared publicly"):
        gf.open_checked(start, opener=op)


def test_a_redirect_downgraded_to_http_is_refused():
    start = gf.download_url("ID")
    op = FakeOpener({start: _redirect("http://drive.usercontent.google.com/download")})
    with pytest.raises(gf.BlockedHost):
        gf.open_checked(start, opener=op)


def test_a_redirect_loop_terminates():
    a = gf.download_url("ID")
    op = FakeOpener({a: _redirect(a)})
    with pytest.raises(RuntimeError, match="too many redirects"):
        gf.open_checked(a, opener=op, max_redirects=3)


def test_a_non_redirect_http_error_propagates_unchanged():
    start = gf.download_url("ID")
    err = urllib.error.HTTPError(start, 404, "Not Found", _headers(), None)
    op = FakeOpener({start: err})
    with pytest.raises(urllib.error.HTTPError):
        gf.open_checked(start, opener=op)


# --------------------------------------------------------------------------
# the large-file interstitial
# --------------------------------------------------------------------------

def test_the_virus_scan_form_is_rebuilt_into_a_download_url():
    html = ('<form id="download-form" action="https://drive.usercontent.google.com/download" '
            'method="get"><input type="hidden" name="id" value="1AbC_dEfG-hIjKlMnOpQrStUvWxYz">'
            '<input type="hidden" name="export" value="download">'
            '<input type="hidden" name="confirm" value="t">'
            '<input type="hidden" name="uuid" value="abc-123"></form>')
    u = gf.confirm_url_from_html(html)
    assert u.startswith("https://drive.usercontent.google.com/download?")
    assert "confirm=t" in u and "uuid=abc-123" in u and f"id={FID}" in u
    assert gf.url_allowed(u)


def test_an_html_page_with_no_form_returns_none_so_the_caller_can_say_why():
    """No form means it is an error page (private / quota), not an interstitial.

    Returning None here is what lets `fetch` refuse instead of writing Google's
    error HTML to disk under the artifact's name.
    """
    assert gf.confirm_url_from_html("<html><body>Access denied</body></html>") is None


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def test_fetch_writes_the_file_and_returns_a_receipt_with_digests(tmp_path):
    body = b"tanitad-checkpoint-bytes" * 100
    op = FakeOpener({gf.download_url(FID): FakeResponse(body)})
    dest = tmp_path / "sub" / "ep.pt"
    r = gf.fetch(FID, dest, opener=op)
    assert dest.read_bytes() == body
    assert r["bytes"] == len(body)
    assert r["md5"] == hashlib.md5(body).hexdigest()
    assert r["sha256"] == hashlib.sha256(body).hexdigest()
    assert r["file_id"] == FID


def test_fetch_follows_the_confirm_form_for_a_large_file(tmp_path):
    body = b"x" * 4096
    html = ('<form action="https://drive.usercontent.google.com/download">'
            '<input name="id" value="1AbC_dEfG-hIjKlMnOpQrStUvWxYz"><input name="confirm" value="t"></form>')
    first = gf.download_url(FID)
    second = gf.confirm_url_from_html(html)
    op = FakeOpener({first: FakeResponse(html.encode(), content_type="text/html; charset=utf-8"),
                     second: FakeResponse(body)})
    dest = tmp_path / "big.bin"
    assert gf.fetch(FID, dest, opener=op)["bytes"] == len(body)
    assert dest.read_bytes() == body


def test_an_md5_mismatch_fails_and_leaves_nothing_behind(tmp_path):
    """A corrupt pull must never be mistakable for a complete artifact."""
    op = FakeOpener({gf.download_url(FID): FakeResponse(b"wrong bytes")})
    dest = tmp_path / "ep.pt"
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        gf.fetch(FID, dest, opener=op, expect_md5="0" * 32)
    assert not dest.exists()
    assert not dest.with_name("ep.pt.part").exists()


def test_a_matching_md5_passes(tmp_path):
    body = b"good bytes"
    op = FakeOpener({gf.download_url(FID): FakeResponse(body)})
    dest = tmp_path / "ep.pt"
    gf.fetch(FID, dest, opener=op, expect_md5=hashlib.md5(body).hexdigest().upper())
    assert dest.read_bytes() == body


def test_a_private_or_quota_limited_file_is_reported_not_written(tmp_path):
    """Google's error HTML must not land on disk wearing the artifact's name."""
    op = FakeOpener({gf.download_url(FID): FakeResponse(
        b"<html><body>You need access</body></html>", content_type="text/html")})
    dest = tmp_path / "ep.pt"
    with pytest.raises(RuntimeError, match="HTML page, not a file"):
        gf.fetch(FID, dest, opener=op)
    assert not dest.exists()


def test_an_empty_download_is_a_failure_not_an_empty_artifact(tmp_path):
    op = FakeOpener({gf.download_url(FID): FakeResponse(b"")})
    dest = tmp_path / "ep.pt"
    with pytest.raises(RuntimeError, match="0 bytes"):
        gf.fetch(FID, dest, opener=op)
    assert not dest.exists()


# --------------------------------------------------------------------------
# the preflight, and the shape of its negative
# --------------------------------------------------------------------------

def test_check_probes_a_concrete_host_for_the_wildcard():
    """`*.googleusercontent.com` is not a hostname; the probe must use a real one."""
    routes = {"https://drive.google.com/": FakeResponse(status=302),
              "https://drive.usercontent.google.com/": FakeResponse(status=404),
              "https://lh3.googleusercontent.com/": FakeResponse(status=400)}
    recs = gf.check_hosts(opener=FakeOpener(routes))
    wild = [r for r in recs if r["entry"] == "*.googleusercontent.com"][0]
    assert wild["probed_host"] == "lh3.googleusercontent.com"
    assert wild["wildcard"] is True
    assert gf.check_exit_code(recs) == 0


def test_404_is_reachable_not_blocked():
    """The host answered. Reading that as 'Drive is blocked' is the mistake here."""
    routes = {f"https://{h}/": urllib.error.HTTPError(
        f"https://{h}/", 404, "Not Found", _headers(), None)
        for h in ("drive.google.com", "drive.usercontent.google.com",
                  "lh3.googleusercontent.com")}
    recs = gf.check_hosts(opener=FakeOpener(routes))
    assert {r["status"] for r in recs} == {gf.REACHABLE}
    assert gf.check_exit_code(recs) == 0


@pytest.mark.parametrize("code", [403, 407])
def test_a_proxy_policy_denial_is_reported_as_blocked(code):
    """Per /root/.ccr/README.md: 403/407 from the proxy is an egress denial."""
    routes = {f"https://{h}/": urllib.error.HTTPError(
        f"https://{h}/", code, "Denied", _headers(), None)
        for h in ("drive.google.com", "drive.usercontent.google.com",
                  "lh3.googleusercontent.com")}
    recs = gf.check_hosts(opener=FakeOpener(routes))
    assert {r["status"] for r in recs} == {gf.POLICY_DENIED}
    assert gf.check_exit_code(recs) == 1


def test_a_failed_tunnel_is_a_policy_denial_not_a_dead_host():
    routes = {f"https://{h}/": OSError("Tunnel connection failed: 403 Forbidden")
              for h in ("drive.google.com", "drive.usercontent.google.com",
                        "lh3.googleusercontent.com")}
    recs = gf.check_hosts(opener=FakeOpener(routes))
    assert {r["status"] for r in recs} == {gf.POLICY_DENIED}


def test_a_genuinely_dead_host_is_unreachable_not_denied():
    routes = {f"https://{h}/": urllib.error.URLError("timed out")
              for h in ("drive.google.com", "drive.usercontent.google.com",
                        "lh3.googleusercontent.com")}
    recs = gf.check_hosts(opener=FakeOpener(routes))
    assert {r["status"] for r in recs} == {gf.UNREACHABLE}
    assert gf.check_exit_code(recs) == 1


def test_one_blocked_host_fails_the_preflight():
    """Two of three is not 'Drive works' — the chain needs every hop."""
    routes = {"https://drive.google.com/": FakeResponse(status=302),
              "https://drive.usercontent.google.com/": urllib.error.HTTPError(
                  "https://drive.usercontent.google.com/", 403, "Denied", _headers(), None),
              "https://lh3.googleusercontent.com/": FakeResponse(status=400)}
    recs = gf.check_hosts(opener=FakeOpener(routes))
    assert gf.check_exit_code(recs) == 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_requires_a_destination(capsys):
    with pytest.raises(SystemExit):
        gf.main(["some-file-id"])


def test_cli_reports_a_blocked_host_instead_of_raising(tmp_path, capsys):
    rc = gf.main(["https://evil.test/file/d/1AbC_dEfG/view", "-o", str(tmp_path / "x")])
    assert rc == 1
    assert "BlockedHost" in capsys.readouterr().err
