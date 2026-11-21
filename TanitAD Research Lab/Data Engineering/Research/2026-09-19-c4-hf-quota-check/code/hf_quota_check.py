"""E18 / C4 — the PI's HF storage, read-only, against the corpus-rebuild push.

⛔ CHECK ONLY. This script performs GET requests and metadata reads. It uploads,
creates, deletes and modifies nothing. The HF quota is a HARD CEILING (PI
2026-08-22); nothing here spends against it.

Token: read in place from the git-ignored Keys.txt; never printed, never on argv.
The whoami `auth` block is never printed.

Outputs per repo: type, visibility, usedStorage (the Hub's own per-repo storage
figure, which includes history), plus account-level probes for any numeric
ceiling. Visibility is split because the Hub meters PRIVATE storage against the
plan; the corpus repo is PRIVATE.
"""
import json
import re
import sys
from pathlib import Path

import truststore
truststore.inject_into_ssl()
import requests  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hf_quota_check.json")
tok = re.search(r"hf_[A-Za-z0-9]+",
                Path("D:/Projects/TanitAD/Keys.txt").read_text(encoding="utf-8",
                                                               errors="replace")).group(0)
api = HfApi(token=tok)
H = {"Authorization": "Bearer " + tok}

who = api.whoami()
user = who["name"]
who_pub = {k: v for k, v in who.items() if k not in ("auth", "orgs", "email", "emailVerified")}
print("user=%s type=%s isPro=%s periodEnd=%s canPay=%s"
      % (user, who.get("type"), who.get("isPro"), who.get("periodEnd"), who.get("canPay")))


def find_keys(o, pat=re.compile(r"storage|quota|limit|usage|capacity", re.I), path=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            p = path + "/" + str(k)
            if pat.search(str(k)) and not isinstance(v, (dict, list)):
                hits.append((p, v))
            hits += find_keys(v, pat, p)
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            hits += find_keys(v, pat, "%s[%d]" % (path, i))
    return hits


# ── account-level probes: does ANY endpoint state a numeric ceiling? ─────────
probes = {}
for name, url in [
    ("whoami-v2", "https://huggingface.co/api/whoami-v2"),
    ("users/overview", "https://huggingface.co/api/users/%s/overview" % user),
    ("settings/billing/usage", "https://huggingface.co/api/settings/billing/usage"),
    ("settings/storage", "https://huggingface.co/api/settings/storage"),
]:
    try:
        r = requests.get(url, headers=H, timeout=30)
        body = r.json() if "json" in r.headers.get("content-type", "") else None
        if isinstance(body, dict):
            body.pop("auth", None)
        hits = find_keys(body) if body is not None else []
        probes[name] = {"status": r.status_code,
                        "top_keys": sorted(body.keys())[:40] if isinstance(body, dict) else None,
                        "storage_like_fields": [[p, v] for p, v in hits][:40]}
    except Exception as e:  # noqa: BLE001
        probes[name] = {"error": type(e).__name__ + ": " + str(e)[:160]}
    print("probe %-24s %s" % (name, {k: (v if k != "top_keys" else len(v or []))
                                      for k, v in probes[name].items()}))

# ── every repo on the account, with the Hub's own usedStorage ───────────────
rows = []
for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets),
                     ("space", api.list_spaces)):
    for r in lister(author=user):
        rid = r.id
        try:
            info = api.repo_info(rid, repo_type=kind, expand=["usedStorage", "private"])
            used = getattr(info, "used_storage", None)
            priv = getattr(info, "private", None)
        except Exception as e:  # noqa: BLE001
            used, priv = None, getattr(r, "private", None)
            print("  expand failed on %s: %s" % (rid, type(e).__name__))
        rows.append({"type": kind, "repo": rid, "private": priv, "used_bytes": used})
rows.sort(key=lambda x: -(x["used_bytes"] or 0))


def tot(sel):
    return sum(x["used_bytes"] or 0 for x in rows if sel(x))


res = {
    "user": user, "isPro": who.get("isPro"), "periodEnd": who.get("periodEnd"),
    "canPay": who.get("canPay"), "whoami_public_keys": sorted(who_pub.keys()),
    "probes": probes,
    "n_repos": len(rows),
    "n_repos_missing_used_storage": sum(1 for x in rows if x["used_bytes"] is None),
    "used_bytes_total": tot(lambda x: True),
    "used_bytes_private": tot(lambda x: x["private"] is True),
    "used_bytes_public": tot(lambda x: x["private"] is False),
    "used_bytes_unknown_visibility": tot(lambda x: x["private"] is None),
    "repos": rows,
}
OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
GB = 1e9
print("\nrepos: %d  (usedStorage missing on %d)" % (len(rows), res["n_repos_missing_used_storage"]))
print("used total   %9.3f GB" % (res["used_bytes_total"] / GB))
print("used PRIVATE %9.3f GB" % (res["used_bytes_private"] / GB))
print("used PUBLIC  %9.3f GB" % (res["used_bytes_public"] / GB))
print("\nlargest 12:")
for x in rows[:12]:
    print("  %-8s %-7s %-52s %10.3f GB" % (x["type"], "PRIVATE" if x["private"] else "public",
                                          x["repo"][:52], (x["used_bytes"] or 0) / GB))
