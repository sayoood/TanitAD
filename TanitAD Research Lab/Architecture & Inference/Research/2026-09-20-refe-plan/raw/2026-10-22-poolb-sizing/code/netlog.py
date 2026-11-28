"""Metadata-only network helpers for POOLB-SIZING. Every request is appended to raw/requests_log.jsonl (url, method, status,
bytes of body read). Methods allowed: GET of a listing / API JSON / S3 XML listing, and HEAD (never follows a redirect into a GET).
No object body is ever read. `head()` uses http.client directly so a redirect can never turn a HEAD into a GET."""
import json, time, os, urllib.request, urllib.parse, http.client

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "raw", "requests_log.jsonl")
UA = "tanitad-poolb-sizing (metadata listing only)"


def _log(rec):
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def get_listing(url, kind="listing"):
    """GET a metadata endpoint (HF tree/API JSON, S3 XML listing). Returns (bytes, headers)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read(); st = r.status; hdr = dict(r.headers)
    _log({"url": url, "method": "GET(" + kind + ")", "status": st, "bytes": len(body)})
    return body, hdr


def head(url, max_hops=3, extra_headers=None):
    """HEAD without following redirects automatically; follows Location manually with HEAD only. Returns list of hops."""
    hops = []
    for _ in range(max_hops + 1):
        u = urllib.parse.urlsplit(url)
        Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        c = Conn(u.netloc, timeout=60)
        path = u.path + (("?" + u.query) if u.query else "")
        h = {"User-Agent": UA}
        if extra_headers: h.update(extra_headers)
        c.request("HEAD", path, headers=h)
        r = c.getresponse(); r.read()
        hdr = {k.lower(): v for k, v in r.getheaders()}
        hops.append({"url": url, "status": r.status, "headers": hdr})
        _log({"url": url, "method": "HEAD", "status": r.status, "bytes": 0})
        c.close()
        if r.status in (301, 302, 303, 307, 308) and "location" in hdr:
            url = urllib.parse.urljoin(url, hdr["location"])
            continue
        break
    return hops
