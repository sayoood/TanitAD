#!/usr/bin/env python3
"""gdrive_fetch — pull a Google Drive artifact, over an ENFORCED domain allowlist.

Why this exists
---------------
The programme's raw material lives on Google Drive (the Windows dev box works
out of ``G:\\Meine Ablage\\SayBouBase\\raw\\Projects\\TanitAD``). A web/pod
session has no such mount: it gets a fresh clone and an egress proxy, so the
only way to reach that material is over HTTPS, and only if the hosts Drive
actually serves from are permitted by the session's egress policy.

Drive needs **three** hosts, not one, and that is the fact this module exists to
pin down:

===============================  ==========================================
``drive.google.com``             the entry point. ``/uc?export=download`` does
                                 NOT serve bytes -- it answers **303** and
                                 redirects.
``drive.usercontent.google.com`` where the bytes are actually served. Allow
                                 only the entry point and every download dies
                                 one hop from the file.
``*.googleusercontent.com``      the legacy/adjacent content hosts
                                 (``lh3.``, ``drive-thirdparty.``, the older
                                 ``doc-XX-XX-docs.`` shards).
===============================  ==========================================

MEASURED 2026-08-08 from a Claude-Code-on-the-web session (evidence:
``TanitAD Research Hub/Tools&DevEnv/Implementation/incoming/2026-08-08-google-drive-domains/``):
all three resolve and answer through the agent proxy with ``ssl_verify_result=0``
and no 403/407 policy denial, and the chain
``drive.google.com/uc?export=download&id=..`` -> **303** ->
``drive.usercontent.google.com/download?..`` is intact. So the egress policy
already permits Drive; what was missing was a checked, repeatable way to use it.

What the allowlist is FOR (it is not decoration)
------------------------------------------------
Every hop is re-checked against ``ALLOWED_HOSTS`` before it is followed. A Drive
link that redirects off the allowlist is **refused**, not followed. That matters
because the input to this tool is a share URL -- frequently pasted from a doc, a
chat message or another agent's report -- and "follow whatever Location comes
back" turns a pasted string into arbitrary outbound egress. The allowlist is the
thing that stops that, so it is enforced in code and covered by tests rather
than written in a comment.

The same check is why ``--check`` exists: the preflight probes the hosts and
distinguishes *"the proxy denied this by policy"* (403/407, per
``/root/.ccr/README.md``) from *"the host answered"*. An HTTP **404** for a
nonexistent id is a **healthy** answer -- the host served us. Reading that 404
as "blocked" is the same class of error as the ``df`` trap in ``CLAUDE.md``: a
probe answering a different question than the one asked.

Usage
-----
    python tools/gdrive_fetch.py --check                      # preflight, table
    python tools/gdrive_fetch.py --check --json probes.json

    python tools/gdrive_fetch.py <share-url-or-file-id> -o out.bin
    python tools/gdrive_fetch.py <id> -o ep.pt --expect-md5 <hex>

Exit codes: ``0`` ok, ``1`` failure (blocked host, refused redirect, md5
mismatch, Drive returned HTML instead of bytes).

Notes
-----
* **stdlib only.** ``stack/pyproject.toml`` declares ``torch`` and ``numpy``;
  ``requests`` is only pulled in by the ``real`` extra. A repo-level tool must
  not need an optional extra to run. ``urllib.request`` honours ``HTTPS_PROXY``
  and the CA bundle already configured for the session.
* The download lands on ``<dest>.part`` and is renamed only after the digest
  check passes, so an interrupted or corrupt pull can never be mistaken for a
  complete artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# THE ALLOWLIST — the three hosts Google Drive actually serves from.
# ---------------------------------------------------------------------------
# A leading "*." means "this domain's subdomains, at any depth" -- the egress
# allowlist convention, NOT the TLS-certificate convention (which is one label).
# It deliberately does NOT match the bare apex: `googleusercontent.com` itself
# serves nothing and is not in the set.
ALLOWED_HOSTS: tuple[str, ...] = (
    "drive.google.com",
    "drive.usercontent.google.com",
    "*.googleusercontent.com",
)

# Concrete stand-ins for the wildcard, so `--check` probes a real name.
WILDCARD_PROBE_HOSTS: dict[str, str] = {
    "*.googleusercontent.com": "lh3.googleusercontent.com",
}

MAX_REDIRECTS = 8
CHUNK = 1 << 20

# Statuses reported by `--check`.
REACHABLE = "REACHABLE"
POLICY_DENIED = "POLICY_DENIED"
UNREACHABLE = "UNREACHABLE"


# ---------------------------------------------------------------------------
# allowlist matching
# ---------------------------------------------------------------------------

def normalise_host(host: str | None) -> str:
    """Lowercase, drop the port and any trailing root dot.

    ``Drive.Google.COM:443.`` and ``drive.google.com`` must not be two different
    answers -- that difference is exactly how a check gets bypassed.
    """
    if not host:
        return ""
    h = host.strip().lower()
    if h.startswith("[") and "]" in h:              # IPv6 literal
        h = h[: h.index("]") + 1]
    elif ":" in h:
        h = h.split(":", 1)[0]
    return h.rstrip(".")


def host_allowed(host: str | None, allowed: tuple[str, ...] = ALLOWED_HOSTS) -> bool:
    """True iff ``host`` is covered by the allowlist.

    The dot boundary is load-bearing: ``notgoogleusercontent.com`` must NOT be
    accepted by ``*.googleusercontent.com``. A bare ``endswith`` on the suffix
    is the classic form of this bug.
    """
    h = normalise_host(host)
    if not h:
        return False
    for entry in allowed:
        e = entry.strip().lower().rstrip(".")
        if e.startswith("*."):
            base = e[2:]
            if h.endswith("." + base):              # dot boundary, any depth
                return True
        elif h == e:
            return True
    return False


def url_allowed(url: str, allowed: tuple[str, ...] = ALLOWED_HOSTS) -> bool:
    """https:// only, and the host must be on the allowlist.

    Scheme is checked too: an http:// hop would leave the proxy's TLS path and
    is never something Drive legitimately asks for.
    """
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if p.scheme.lower() != "https":
        return False
    return host_allowed(p.hostname, allowed)


class BlockedHost(RuntimeError):
    """A hop left the allowlist. Raised instead of following it."""


# ---------------------------------------------------------------------------
# Drive URL shapes
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_PATH_ID_RE = re.compile(r"/(?:file|folders)/d/([A-Za-z0-9_-]+)")
_D_RE = re.compile(r"/d/([A-Za-z0-9_-]+)")


def file_id_from(url_or_id: str) -> str:
    """Extract the file id from any of the share-link shapes, or a bare id.

    Drive hands out at least four shapes and they all appear in our docs:
    ``/file/d/<id>/view``, ``/open?id=``, ``/uc?id=``/``?export=download&id=``,
    and the bare id copied out of a URL bar.
    """
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("empty file id/URL")

    if "://" not in s:
        if _ID_RE.match(s):
            return s
        raise ValueError(f"not a Drive file id or URL: {s!r}")

    p = urllib.parse.urlsplit(s)
    if not host_allowed(p.hostname):
        raise BlockedHost(
            f"{p.hostname!r} is not a Google Drive host "
            f"(allowlist: {', '.join(ALLOWED_HOSTS)})")

    qs = urllib.parse.parse_qs(p.query)
    if qs.get("id"):
        return qs["id"][0]
    m = _PATH_ID_RE.search(p.path) or _D_RE.search(p.path)
    if m:
        return m.group(1)
    raise ValueError(f"no file id found in URL: {s!r}")


def download_url(file_id: str, confirm: str = "t") -> str:
    """The bytes host, addressed directly.

    Going straight to ``drive.usercontent.google.com`` skips the 303 from the
    entry point. ``confirm`` clears the large-file interstitial for public files.
    """
    q = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": confirm})
    return f"https://drive.usercontent.google.com/download?{q}"


# ---------------------------------------------------------------------------
# HTTP, with every hop re-checked
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects as HTTPError so we can vet the Location ourselves."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):   # noqa: D102
        return None


def build_opener() -> urllib.request.OpenerDirector:
    """An opener that does not auto-follow. Proxy + CA come from the env."""
    return urllib.request.build_opener(_NoRedirect, urllib.request.HTTPCookieProcessor())


def _classify(exc: BaseException) -> str:
    """Tell a policy denial apart from a host that is simply not answering.

    The distinction this function exists for: **an HTTPError is an ANSWER.** A
    404 for a bare path, or a 400 for a request Google does not like, means the
    host served us -- that is ``REACHABLE``, and reporting it as "blocked" is
    the same class of error as reading ``df`` on a pod. Only the proxy's own
    refusal codes are a denial.

    ``/root/.ccr/README.md``: 403/407 from the proxy means the destination is
    not permitted by the org's egress policy -- report it, never route around
    it. A tunnel failure shows up as an OSError carrying that text.

    The one ambiguity, stated rather than hidden: a 403 *originating at Google*
    would be reported as a denial too. For these three hosts on a bare path that
    is not what they do -- MEASURED 2026-08-08: 302 / 404 / 400 -- so a 403 here
    is the proxy. If it ever matters, the proxy's own status endpoint records
    the reason and settles it.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return POLICY_DENIED if exc.code in (403, 407) else REACHABLE
    text = str(getattr(exc, "reason", exc))
    if "tunnel connection failed" in text.lower():
        return POLICY_DENIED
    return UNREACHABLE


def _auth_hint(url: str) -> str:
    """Turn the one refusal that has a specific meaning into a specific message.

    MEASURED 2026-08-08: an unauthenticated ``https://drive.google.com/`` answers
    **302 -> accounts.google.com/ServiceLogin**. So a sign-in redirect is not a
    misconfigured allowlist -- it is Drive saying *this file is not publicly
    shared*. Without this hint the failure reads as "our allowlist is too
    narrow", and the next person widens it instead of fixing the sharing.
    """
    if normalise_host(urllib.parse.urlsplit(url).hostname) == "accounts.google.com":
        return ("\n  -> this is a SIGN-IN redirect: the file is not shared "
                "publicly ('Anyone with the link'). These three hosts cover "
                "anonymous downloads only; authenticated Drive would need "
                "accounts.google.com + the API hosts, which are deliberately "
                "not on the allowlist.")
    return ""


def open_checked(url: str, opener=None, max_redirects: int = MAX_REDIRECTS,
                 timeout: float = 60.0):
    """GET ``url``, following redirects only while they stay on the allowlist.

    Returns the final response. Raises :class:`BlockedHost` the moment a hop
    would leave the allowlist -- the redirect is reported, not followed.
    """
    opener = opener or build_opener()
    if not url_allowed(url):
        raise BlockedHost(f"refusing to fetch off-allowlist URL: {url}")

    seen = [url]
    for _ in range(max_redirects + 1):
        try:
            return opener.open(seen[-1], timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            loc = e.headers.get("Location") or ""
            nxt = urllib.parse.urljoin(seen[-1], loc)
            if not url_allowed(nxt):
                raise BlockedHost(
                    f"redirect left the allowlist and was NOT followed: "
                    f"{seen[-1]} -> {nxt}{_auth_hint(nxt)}") from e
            seen.append(nxt)
    raise RuntimeError(f"too many redirects ({max_redirects}): {' -> '.join(seen)}")


# ---------------------------------------------------------------------------
# the large-file interstitial
# ---------------------------------------------------------------------------

_FORM_RE = re.compile(r"<form[^>]*\baction=\"([^\"]+)\"[^>]*>(.*?)</form>",
                      re.IGNORECASE | re.DOTALL)
_INPUT_RE = re.compile(r"<input[^>]*\bname=\"([^\"]+)\"[^>]*\bvalue=\"([^\"]*)\"[^>]*>",
                       re.IGNORECASE)


def confirm_url_from_html(html: str) -> str | None:
    """Rebuild the download URL from Drive's "can't scan for viruses" form.

    Above ~100 MB Drive answers the download URL with an HTML page carrying a
    form whose hidden inputs (``id``, ``export``, ``confirm``, ``uuid``) must be
    replayed. Returns ``None`` when the page carries no such form -- which is
    the *interesting* case, because it means the HTML is an error page (private
    file, quota exceeded) and the caller should say so rather than write it out.
    """
    m = _FORM_RE.search(html)
    if not m:
        return None
    action, body = m.group(1), m.group(2)
    action = action.replace("&amp;", "&")
    params = {k: v.replace("&amp;", "&") for k, v in _INPUT_RE.findall(body)}
    if not action:
        return None
    if params:
        sep = "&" if "?" in action else "?"
        action = f"{action}{sep}{urllib.parse.urlencode(params)}"
    return action


def _is_html(resp) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return "text/html" in ctype


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def fetch(url_or_id: str, dest: str | os.PathLike, *, opener=None,
          expect_md5: str | None = None, timeout: float = 60.0) -> dict:
    """Download a Drive file to ``dest``. Returns a receipt dict.

    The bytes land on ``<dest>.part`` and are renamed only once the digest check
    passes: a failed pull must never leave something that looks like a complete
    artifact behind.
    """
    opener = opener or build_opener()
    fid = file_id_from(url_or_id)
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")

    resp = open_checked(download_url(fid), opener=opener, timeout=timeout)
    if _is_html(resp):
        html = resp.read(1 << 20).decode("utf-8", "replace")
        nxt = confirm_url_from_html(html)
        if nxt is None:
            raise RuntimeError(
                f"Drive returned an HTML page, not a file, for id {fid!r} — the "
                f"file is most likely private, deleted, or over its download "
                f"quota. First 300 chars: {html[:300]!r}")
        if not url_allowed(nxt):
            raise BlockedHost(f"confirm form pointed off-allowlist: {nxt}")
        resp = open_checked(nxt, opener=opener, timeout=timeout)
        if _is_html(resp):
            raise RuntimeError(
                f"Drive still returned HTML after the confirm step for id {fid!r}")

    md5, sha256, n = hashlib.md5(), hashlib.sha256(), 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "wb") as fh:
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            fh.write(buf)
            md5.update(buf)
            sha256.update(buf)
            n += len(buf)

    receipt = {
        "file_id": fid,
        "dest": str(dest),
        "bytes": n,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "final_url": resp.geturl(),
    }

    if expect_md5 and receipt["md5"].lower() != expect_md5.strip().lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"md5 mismatch for id {fid!r}: got {receipt['md5']}, "
            f"expected {expect_md5} — partial file removed")
    if n == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded 0 bytes for id {fid!r}")

    part.replace(dest)
    return receipt


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def check_hosts(allowed: tuple[str, ...] = ALLOWED_HOSTS, *, opener=None,
                timeout: float = 25.0) -> list[dict]:
    """Probe each allowlist entry and say whether egress permits it.

    Any HTTP answer -- including 400/404 for a bare path or a bogus id -- means
    REACHABLE: the host served us. Only a proxy policy denial or a dead
    connection is a real negative.
    """
    opener = opener or build_opener()
    out = []
    for entry in allowed:
        probe = WILDCARD_PROBE_HOSTS.get(entry, entry)
        rec = {"entry": entry, "probed_host": probe, "wildcard": entry != probe}
        try:
            resp = opener.open(f"https://{probe}/", timeout=timeout)
            rec.update(status=REACHABLE, http=resp.status)
        except urllib.error.HTTPError as e:
            rec.update(status=_classify(e), http=e.code)
        except Exception as e:                        # noqa: BLE001
            rec.update(status=_classify(e), http=None, error=f"{type(e).__name__}: {e}")
        out.append(rec)
    return out


def check_exit_code(records: list[dict]) -> int:
    return 0 if all(r["status"] == REACHABLE for r in records) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gdrive_fetch",
        description="Fetch a Google Drive artifact over an enforced domain allowlist.")
    ap.add_argument("target", nargs="?", help="Drive share URL or bare file id")
    ap.add_argument("-o", "--out", help="destination path")
    ap.add_argument("--expect-md5", help="fail (and delete the partial) on mismatch")
    ap.add_argument("--check", action="store_true",
                    help="probe the allowlisted hosts and exit")
    ap.add_argument("--json", help="also write the result as JSON to this path")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    if args.check:
        recs = check_hosts(timeout=min(args.timeout, 25.0))
        width = max(len(r["entry"]) for r in recs)
        for r in recs:
            note = f"  (probed {r['probed_host']})" if r["wildcard"] else ""
            print(f"{r['entry']:<{width}}  {r['status']:<14} "
                  f"http={r.get('http')}{note}")
        if args.json:
            Path(args.json).write_text(json.dumps(recs, indent=2), encoding="utf-8")
        rc = check_exit_code(recs)
        print("\nallowlist:", ", ".join(ALLOWED_HOSTS))
        if rc:
            print("At least one host is not permitted by this session's egress "
                  "policy. Report the blocked host — do not route around it.",
                  file=sys.stderr)
        return rc

    if not args.target or not args.out:
        ap.error("a target and -o/--out are required unless --check is given")

    try:
        receipt = fetch(args.target, args.out,
                        expect_md5=args.expect_md5, timeout=args.timeout)
    except (BlockedHost, RuntimeError, ValueError, urllib.error.URLError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
