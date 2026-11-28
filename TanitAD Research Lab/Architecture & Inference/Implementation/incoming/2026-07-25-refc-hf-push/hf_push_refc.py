"""Push REF-C XL + base checkpoints to private HF repos under Sayood/.

Mirrors the visibility of the direct sibling `Sayood/tanitad-refc-small-evalonly`
(private=True, gated=False) — the least-open sibling, per the PhysicalAI-AV gating rule.

Token is read from STDIN (never in argv, never printed, never written to a file).
md5 is verified against the MODEL_REGISTRY value BEFORE any upload; a mismatch aborts.
"""
import hashlib
import os
import sys
import time

TOK = sys.stdin.readline().strip().lstrip("﻿")
assert TOK.startswith("hf_") and len(TOK) > 20, "no valid token on stdin"

from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=TOK)

JOBS = [
    {
        "repo": "Sayood/tanitad-refc-xl",
        "dir": "/root/models/refc-xl-30k",
        "md5": "966d4eff1ea5ddf86efba01b8344e198",   # MODEL_REGISTRY.md 4.1
        "readme": "/root/refc_hf/README_xl.md",
    },
    {
        "repo": "Sayood/tanitad-refc-base",
        "dir": "/root/models/refc-base-30k",
        "md5": "8f10d6f934f4199e11ddc7352e074939",   # MODEL_REGISTRY.md 4.3
        "readme": "/root/refc_hf/README_base.md",
    },
]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- GATE 1: md5 every checkpoint BEFORE touching HF ----------------------
for j in JOBS:
    ck = os.path.join(j["dir"], "ckpt.pt")
    t0 = time.time()
    got = md5_of(ck)
    ok = got == j["md5"]
    print(f"[md5] {ck} -> {got} expected {j['md5']} :: {'MATCH' if ok else 'MISMATCH'} "
          f"({time.time()-t0:.0f}s, {os.path.getsize(ck)} B)", flush=True)
    if not ok:
        print("ABORT: checkpoint drifted from the registry md5 — refusing to publish.", flush=True)
        sys.exit(3)
    j["size"] = os.path.getsize(ck)

# ---- upload ---------------------------------------------------------------
for j in JOBS:
    repo = j["repo"]
    print(f"\n=== {repo} ===", flush=True)
    try:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        print("[repo] created/exists, private=True (mirrors tanitad-refc-small-evalonly)", flush=True)
    except Exception as e:
        print(f"[repo] CREATE FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        continue

    uploads = [
        (j["readme"], "README.md"),
        (os.path.join(j["dir"], "config.json"), "config.json"),
        (os.path.join(j["dir"], "metrics.json"), "metrics.json"),
        (os.path.join(j["dir"], "ckpt.pt"), "ckpt.pt"),
    ]
    for local, name in uploads:
        if not os.path.exists(local):
            print(f"[skip] missing {local}", flush=True)
            continue
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=local, path_in_repo=name, repo_id=repo,
                repo_type="model",
                commit_message=f"add {name} (REF-C step 29999 FINAL, registry-verified)",
            )
            dt = time.time() - t0
            sz = os.path.getsize(local)
            rate = sz / dt / 1e6 if dt > 0 else 0
            print(f"[up] {name} {sz} B in {dt:.1f}s ({rate:.1f} MB/s)", flush=True)
        except Exception as e:
            print(f"[up] FAILED {name}: {type(e).__name__}: {str(e)[:600]}", flush=True)

    # verify
    try:
        info = api.model_info(repo)
        print(f"[verify] private={info.private} gated={info.gated}", flush=True)
        for f in api.list_repo_tree(repo, recursive=True):
            print(f"[verify]   {f.path}  {getattr(f, 'size', None)}", flush=True)
    except Exception as e:
        print(f"[verify] FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)

print("\nPUSH_DONE", flush=True)
