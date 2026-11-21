"""Back up the E1b FT checkpoint to HuggingFace — it is the last single-disk
artifact of the E1b/E1c line (527 MB, pod3 only; a volume event costs 3.5 h of
A40 time to rebuild).

Executed in the program's mandated order so the weights are NEVER
world-downloadable (identical sequencing to
`…/2026-07-25-refc-hf-push/hf_gate_publish_push.py`, which the PI approved as
"Option A" after the free-tier private-storage 403):

  1. create the repo PRIVATE
  2. set gated="manual" WHILE STILL PRIVATE
  3. flip private=False
  4. VERIFY via the API: private is False AND gated == "manual"   <-- hard gate
  5. only then upload ckpt.pt (+ card, config, metrics, train log)
  6. verify the final file list + sizes

Any failure at step 4 aborts WITHOUT uploading weights.
Token is read from STDIN; never printed, never written to disk, never in argv.
Pushing from the pod runs at ~118-246 MB/s; the dev-box relay is ~1 MB/s.
"""
import hashlib
import os
import sys
import time

TOK = sys.stdin.readline().strip().lstrip("﻿")
assert TOK.startswith("hf_") and len(TOK) > 20, "no valid token on stdin"

from huggingface_hub import HfApi  # noqa: E402

REPO = "Sayood/tanitad-refc-base-e1b-clsft"
SRC = "/workspace/e1b/refc-base-e1b-clsft"
CARD = """---
license: other
license_name: physicalai-av-derived-research-only
tags:
- tanitad
- autonomous-driving
- anchored-diffusion
- planner
- closed-loop
- fine-tune
extra_gated_prompt: >-
  These weights are trained on NVIDIA PhysicalAI-AV data (TanitAD research
  program). Access is granted per request for research/evaluation use only;
  you agree not to redistribute.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
---

# TanitAD REF-C base — E1b failure-gated closed-loop SFT (research artifact)

**This checkpoint is a pre-registered `BOUND` result, not a deployed model.**
It is published for reproducibility and because it was the last single-disk
artifact of the E1b/E1c experiment line.

- **Base:** `Sayood/tanitad-refc-base` (REF-C anchored-diffusion planner,
  104,191,577 params, 128 anchors, 2 denoise steps), step 29999.
- **Fine-tune:** 4000 steps, lr 2e-5, cosine, warmup 100, encoder **frozen**
  (13,732,945 trainable / 90,458,632 frozen). Two interleaved objectives:
  R2LPL-shaped anchor-score/traj supervision toward a logged-corridor *recovery*
  demonstration at 3,537 mined recoverable pre-failure states, plus an
  open-loop replay branch on the parity-train corpus.
- **Mining/replay corpus:** `physicalai-train-e438721ae894` (2376 episodes).
  **Evaluation:** a byte-level-disjoint 44-episode held-out set.

## Measured (paired episode-cluster bootstrap, B=2000, over the held-out episodes)

| metric (K=185, 18.5 s closed loop) | base | this ckpt | paired delta |
|---|---|---|---|
| corridor-departure, overall | 0.5877 | 0.1603 | −0.4274 [−0.5161, −0.3378] **separated** |
| corridor-departure, junction | 0.8414 | 0.4144 | −0.4270 [−0.6838, −0.1648] **separated** |
| peak abs XTE (m) | 38.94 | 3.04 | −35.90 [−49.33, −24.12] **separated** |
| OOD peak ratio (in-band check) | 1.2664 | 1.1339 | −0.1325 **separated (favourable)** |

| open-loop guardrail (held-out) | base | this ckpt | paired delta |
|---|---|---|---|
| ADE@2s (m) | 0.4747 | 0.6693 | **+0.1947 [+0.1415, +0.2522] separated WORSE** |
| anchor accuracy | 0.6815 | 0.6163 | **−0.0651 separated WORSE** |
| anchor traj L1 | 0.1775 | 0.2399 | **+0.0624 separated WORSE** |

**Verdict: `BOUND`.** The pre-registration committed that a closed-loop win
bought with a CI-separated open-loop regression is *not* a success. The
diagnosed cause: the forgetting guard was monitored on the corpus it replays,
so it could only ever report success. E1c re-runs this configuration with the
guard on held-out data and reports the full frontier.

## Files
`ckpt.pt` (model + optimizer, step 3999) · `config.json` · `metrics.json` ·
`train_log.jsonl`

**Not a safety claim.** The closed loop here is map/agent-free and measures
corridor-keeping/drift, not collision or off-road safety.
"""


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_gated(g):
    return g if isinstance(g, str) else ("false" if g in (False, None) else str(g))


api = HfApi(token=TOK)
who = api.whoami()
print(f"[0] auth: user={who.get('name')} type={who.get('type')}", flush=True)

# --- STEP 1: create PRIVATE -------------------------------------------------
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
info = api.model_info(REPO)
print(f"[1] created/exists: private={info.private} gated={norm_gated(info.gated)}",
      flush=True)

# --- STEP 2: gate FIRST, while still private --------------------------------
api.update_repo_settings(repo_id=REPO, repo_type="model", gated="manual")
mid = api.model_info(REPO)
print(f"[2] readback: private={mid.private} gated={norm_gated(mid.gated)}",
      flush=True)
if norm_gated(mid.gated) != "manual":
    print("[ABORT] gating did not stick while private — not flipping to public.",
          flush=True)
    print("HF_PUSH_ABORT", flush=True)
    sys.exit(2)

# --- STEP 3: publish (gated) ------------------------------------------------
api.update_repo_settings(repo_id=REPO, repo_type="model", private=False)
time.sleep(2)
fin = api.model_info(REPO)
ok = (fin.private is False) and (norm_gated(fin.gated) == "manual")
print(f"[3] VERIFY: private={fin.private} gated={norm_gated(fin.gated)} -> ok={ok}",
      flush=True)
if not ok:
    print("[ABORT] repo is not (public AND gated=manual). NO WEIGHTS UPLOADED.",
          flush=True)
    print("HF_PUSH_ABORT", flush=True)
    sys.exit(3)

# --- STEP 4: small files ----------------------------------------------------
open("/tmp/_e1b_card.md", "w").write(CARD)
api.upload_file(path_or_fileobj="/tmp/_e1b_card.md", path_in_repo="README.md",
                repo_id=REPO, repo_type="model",
                commit_message="model card: E1b failure-gated CL-SFT (BOUND)")
for f in ("config.json", "metrics.json", "train_log.jsonl"):
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO,
                        repo_type="model", commit_message=f"add {f}")
        print(f"[4] uploaded {f}", flush=True)

# --- STEP 5: the weights ----------------------------------------------------
ck = os.path.join(SRC, "ckpt.pt")
sz = os.path.getsize(ck)
m = md5_of(ck)
print(f"[5] ckpt.pt {sz} B md5 {m}", flush=True)
t0 = time.time()
api.upload_file(path_or_fileobj=ck, path_in_repo="ckpt.pt", repo_id=REPO,
                repo_type="model",
                commit_message=f"add ckpt.pt (E1b CL-SFT step 3999, md5 {m})")
dt = time.time() - t0
print(f"[5] uploaded in {dt:.0f}s ({sz / max(dt, 1e-9) / 1e6:.1f} MB/s)", flush=True)

# --- STEP 6: verify ---------------------------------------------------------
fin = api.model_info(REPO)
print(f"[6] FINAL: private={fin.private} gated={norm_gated(fin.gated)}", flush=True)
for f in api.list_repo_tree(REPO, recursive=True):
    print(f"[6]   {f.path}  {getattr(f, 'size', None)} B", flush=True)
print(f"[6] URL https://huggingface.co/{REPO}", flush=True)
print(f"[6] LOCAL_MD5 {m}", flush=True)
print("HF_PUSH_DONE", flush=True)
