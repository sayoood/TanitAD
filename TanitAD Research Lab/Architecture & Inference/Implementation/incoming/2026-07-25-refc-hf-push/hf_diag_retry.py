"""Diagnose the ckpt.pt upload failure with FULL error detail, then retry once.
Token from stdin. Never prints the token."""
import os
import sys
import time
import traceback

TOK = sys.stdin.readline().strip().lstrip("﻿")
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

api = HfApi(token=TOK)

print("=== account / storage ===", flush=True)
try:
    import requests
    r = requests.get("https://huggingface.co/api/whoami-v2",
                     headers={"Authorization": f"Bearer {TOK}"}, timeout=30)
    d = r.json()
    safe = {k: v for k, v in d.items() if k not in ("auth",)}
    print("  whoami-v2:", str(safe)[:1500], flush=True)
except Exception as e:
    print("  whoami-v2 failed:", type(e).__name__, str(e)[:200], flush=True)

for path in ("https://huggingface.co/api/users/Sayood/overview",):
    try:
        r = requests.get(path, headers={"Authorization": f"Bearer {TOK}"}, timeout=30)
        print(f"  {path} -> {r.status_code} {r.text[:800]}", flush=True)
    except Exception as e:
        print(f"  {path} failed: {type(e).__name__} {str(e)[:200]}", flush=True)

JOBS = [
    ("Sayood/tanitad-refc-xl", "/root/models/refc-xl-30k/ckpt.pt"),
    ("Sayood/tanitad-refc-base", "/root/models/refc-base-30k/ckpt.pt"),
]

for repo, ck in JOBS:
    print(f"\n=== retry {repo} :: {ck} ({os.path.getsize(ck)} B) ===", flush=True)
    t0 = time.time()
    try:
        api.upload_file(path_or_fileobj=ck, path_in_repo="ckpt.pt", repo_id=repo,
                        repo_type="model",
                        commit_message="add ckpt.pt (REF-C step 29999 FINAL, registry-verified)")
        print(f"[ok] uploaded in {time.time()-t0:.0f}s", flush=True)
    except HfHubHTTPError as e:
        print(f"[FAIL] HfHubHTTPError after {time.time()-t0:.0f}s", flush=True)
        resp = getattr(e, "response", None)
        if resp is not None:
            print("  status_code:", resp.status_code, flush=True)
            print("  reason     :", getattr(resp, "reason", None), flush=True)
            print("  url        :", resp.url, flush=True)
            print("  body       :", (resp.text or "")[:2000], flush=True)
            hdrs = {k: v for k, v in resp.headers.items()
                    if k.lower().startswith("x-") or k.lower() in ("content-type",)}
            print("  headers    :", str(hdrs)[:800], flush=True)
        print("  server_message:", getattr(e, "server_message", None), flush=True)
        print("  str(e)[:1500]:", str(e)[:1500], flush=True)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__} after {time.time()-t0:.0f}s", flush=True)
        traceback.print_exc()
        print("  str(e)[:1500]:", str(e)[:1500], flush=True)

print("\nDIAG_DONE", flush=True)
