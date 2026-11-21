"""Option A, executed in the mandated order so the weights are NEVER world-downloadable:

  1. set gated="manual"   (while the repo is still PRIVATE)
  2. flip private=False
  3. VERIFY via the API: private is False AND gated == "manual"   <-- hard gate
  4. re-verify ckpt md5 against MODEL_REGISTRY
  5. only then upload ckpt.pt
  6. verify final file list + sizes

Any failure at step 3 or 4 aborts THAT model without uploading weights.
Token from stdin; never printed, never written to disk, never in argv.
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
    {"repo": "Sayood/tanitad-refc-xl",
     "ck": "/root/models/refc-xl-30k/ckpt.pt",
     "md5": "966d4eff1ea5ddf86efba01b8344e198"},     # MODEL_REGISTRY 4.1
    {"repo": "Sayood/tanitad-refc-base",
     "ck": "/root/models/refc-base-30k/ckpt.pt",
     "md5": "8f10d6f934f4199e11ddc7352e074939"},     # MODEL_REGISTRY 4.3
]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_gated(g):
    """HF returns 'manual' / 'auto' / False / None."""
    return g if isinstance(g, str) else ("false" if g in (False, None) else str(g))


results = []
for j in JOBS:
    repo = j["repo"]
    print(f"\n================ {repo} ================", flush=True)

    before = api.model_info(repo)
    print(f"[0] BEFORE: private={before.private} gated={norm_gated(before.gated)}", flush=True)

    # --- STEP 1: gate FIRST, while still private -------------------------
    try:
        api.update_repo_settings(repo_id=repo, repo_type="model", gated="manual")
        print("[1] gated=manual requested (repo still private)", flush=True)
    except Exception as e:
        print(f"[1] GATING FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        print("[ABORT] refusing to publish or upload weights without gating.", flush=True)
        results.append((repo, "ABORT-gating-failed"))
        continue

    mid = api.model_info(repo)
    print(f"[1] readback: private={mid.private} gated={norm_gated(mid.gated)}", flush=True)
    if norm_gated(mid.gated) != "manual":
        print("[ABORT] gating did not stick while private — not flipping to public.", flush=True)
        results.append((repo, "ABORT-gating-not-confirmed"))
        continue

    # --- STEP 2: publish --------------------------------------------------
    try:
        api.update_repo_settings(repo_id=repo, repo_type="model", private=False)
        print("[2] private=False requested", flush=True)
    except Exception as e:
        print(f"[2] PUBLISH FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        results.append((repo, "ABORT-publish-failed"))
        continue

    # --- STEP 3: HARD GATE — must be public AND gated=manual ---------------
    time.sleep(2)
    info = api.model_info(repo)
    is_public = (info.private is False)
    is_gated = (norm_gated(info.gated) == "manual")
    print(f"[3] VERIFY: private={info.private} gated={norm_gated(info.gated)} "
          f"-> public={is_public} gated_manual={is_gated}", flush=True)
    if not (is_public and is_gated):
        print("[ABORT] repo is not (public AND gated=manual). NO WEIGHTS UPLOADED.", flush=True)
        results.append((repo, "ABORT-state-not-public-gated"))
        continue

    # --- STEP 4: md5 re-gate ----------------------------------------------
    got = md5_of(j["ck"])
    ok = got == j["md5"]
    print(f"[4] md5 {got} vs registry {j['md5']} :: {'MATCH' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        print("[ABORT] checkpoint drifted from the registry md5 — refusing to publish it.", flush=True)
        results.append((repo, "ABORT-md5-mismatch"))
        continue

    # --- STEP 5: upload ----------------------------------------------------
    sz = os.path.getsize(j["ck"])
    print(f"[5] uploading ckpt.pt ({sz} B) ...", flush=True)
    t0 = time.time()
    try:
        api.upload_file(path_or_fileobj=j["ck"], path_in_repo="ckpt.pt", repo_id=repo,
                        repo_type="model",
                        commit_message="add ckpt.pt (REF-C step 29999 FINAL, md5-verified vs MODEL_REGISTRY)")
        dt = time.time() - t0
        print(f"[5] uploaded in {dt:.0f}s ({sz/dt/1e6:.1f} MB/s)", flush=True)
        results.append((repo, "OK"))
    except Exception as e:
        print(f"[5] UPLOAD FAILED {type(e).__name__}", flush=True)
        resp = getattr(e, "response", None)
        if resp is not None:
            print(f"    status={resp.status_code} body={(resp.text or '')[:600]}", flush=True)
        print(f"    str(e)[:600]: {str(e)[:600]}", flush=True)
        results.append((repo, "FAIL-upload"))

    # --- STEP 6: final verification ---------------------------------------
    fin = api.model_info(repo)
    print(f"[6] FINAL: private={fin.private} gated={norm_gated(fin.gated)}", flush=True)
    for f in api.list_repo_tree(repo, recursive=True):
        print(f"[6]   {f.path}  {getattr(f, 'size', None)} B", flush=True)

print("\n=== SUMMARY ===", flush=True)
for repo, status in results:
    print(f"  {repo}: {status}", flush=True)
print("GATE_PUBLISH_PUSH_DONE", flush=True)
