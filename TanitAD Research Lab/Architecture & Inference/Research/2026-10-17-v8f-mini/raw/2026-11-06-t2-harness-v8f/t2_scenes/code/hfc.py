"""HF access helpers for the T2 scene screen. The token is read in place from Keys.txt and never printed, logged or written."""
import json, os, re, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
import requests

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
KEYS = "E:/Projects/TanitAD/Keys.txt"
_TOKS = None

def tokens():
    global _TOKS
    if _TOKS is None:
        txt = open(KEYS, encoding="utf-8", errors="ignore").read()
        seen = []
        for m in re.findall(r"hf_[A-Za-z0-9]+", txt):
            if m not in seen:
                seen.append(m)
        _TOKS = seen
    return _TOKS

_ACTIVE = None
def active_token():
    """First token that authenticates (whoami-v2 == 200); cached. Returns the token (never print it)."""
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    for t in tokens():
        r = requests.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": "Bearer " + t}, timeout=60)
        if r.status_code == 200:
            _ACTIVE = t
            return t
    raise RuntimeError("no hf_ token in Keys.txt authenticates")

def auth():
    return {"Authorization": "Bearer " + active_token(), "User-Agent": "tanitad-t2-scenes/1"}

def get(url, headers=None, tries=5, **kw):
    """GET with the bearer token on the huggingface.co hop ONLY; redirects are followed by hand without the token."""
    last = None
    for i in range(tries):
        try:
            h = dict(headers or {})
            host = url.split("/")[2]
            if host.endswith("huggingface.co"):
                h.update(auth())
            r = requests.get(url, headers=h, allow_redirects=False, timeout=120, **kw)
            hops = 0
            while r.status_code in (301, 302, 303, 307, 308) and hops < 6:
                loc = r.headers["Location"]
                if loc.startswith("/"):
                    loc = "https://huggingface.co" + loc
                h2 = {k: v for k, v in (headers or {}).items()}
                if loc.split("/")[2].endswith("huggingface.co"):
                    h2.update(auth())
                r = requests.get(loc, headers=h2, allow_redirects=False, timeout=120, **kw)
                hops += 1
            return r
        except (requests.RequestException, OSError) as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last
