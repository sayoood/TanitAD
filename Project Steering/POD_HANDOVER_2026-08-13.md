# Pod handover — can we continue on fresh pods? YES, with these preconditions

**PI plan (2026-08-13):** *"I will let the pods run maximally the next two days, afterwards
I will start fresh ones… Will we be able to continue the training and the pipeline from the
new pods?"*

**Answer: yes for both, provided the four artifacts below reach HF before the pods die.**
Code is already safe (GitHub, pushed continuously). Everything else lives only on pod disk.

---

## 1. What must leave the pods

| # | artifact | where now | size | needed to resume |
|---|---|---|---|---|
| **1** | **`v6F-SW-30k/ckpt.pt`** + `config.json` + `stage_gate.json` | pod5 | ~1.3 GB/ckpt | **the training run.** `--resume auto` + `--init-from` read exactly this. |
| **2** | **`ph0_prod4/`** — v2 JSON + sam3 JSON (+ bridged mp4/ego) | pod4 | ~0.9 GB | **the pipeline.** The labels are the product; the mp4/ego are re-derivable from the corpus. |
| **3** | **`ph0_prod/`** — the 2400-clip bridged corpus (mp4 + ego) | pod5 | **5.46 GB** | saves ~2 h of re-bridging; NOT strictly required (re-derivable from the v2 cache). |
| **4** | **P-battery gate JSONs** | pod5 | KB | the S-W verdict; tiny, push with the checkpoint. |

⚠️ **The v2 caches themselves (80 GB train + 20 GB val) are NOT in this list.** They are
rebuilt by `v2_compressed.py` from the HF PhysicalAI source on any new pod — hours of
download, but zero risk of loss, and pushing 100 GB to HF would be slower than rebuilding.
**Parity is preserved by the skip-hash + episode-uid check, not by the bytes.**

## 2. Resume semantics — what actually restores

- **Training.** `load_resume` does a **strict** state-dict load, so a checkpoint resumes
  only into the *same architecture*. The new pod must run **the same flags**, which is why
  `config.json` is banked beside `ckpt.pt` — it records every one. `--resume auto` then
  continues from the exact step with the optimiser state.
- ⛔ **A config drift silently invalidates the resume.** Same architecture ⇒ strict load
  passes; different `--pred-modern`/`--vit5-encoder`/dims ⇒ it *refuses* (by design, tested).
  So the risk is not silent corruption, it is a refused restart — recoverable, but only if
  `config.json` travelled with the checkpoint.
- **Pipeline.** `ph0_v2.py --resume` skips clips already all-valid and writes incrementally
  after every clip, so a pod dying mid-run costs at most one clip. `ph0_sam3.py` re-runs
  from the v2 JSON. **Nothing needs the old pod.**

## 3. What a fresh pod needs, in order

1. `git clone` the branch → **all code**, including the chunked shipper.
2. Restore the venv: `torch==2.8.0+cu128`/`torchvision==0.23.0` **from the pinned index,
   installed LAST** (the twice-measured trap: `uv pip install <anything>` drags torch
   forward to a wheel the driver cannot run). Then `--no-deps` for the rest, and verify
   with a real **`conv2d` on CUDA**, not `import torch`.
3. `pip install` the pipeline extras with `--no-deps`: `sam3`, `open_clip_torch` (the CLIP
   BPE vocab the sam3 wheel does not ship), `opencv-python-headless`, `imageio`/
   `imageio-ffmpeg`, `lm-format-enforcer`, `qwen_vl_utils`.
4. Rebuild the v2 caches (`v2_compressed.py build …`) — **hours**, unattended.
5. `hf download` artifacts 1–4.
6. Relaunch with the flags from the banked `config.json`.

⚠️ **`facebook/sam3` is a gated repo.** Access is granted to the *account*, not the pod, so
the token carries it — but the new pod needs the HF token present at
`/root/.cache/huggingface/token`. Verified working today (`whoami` → Sayood).

## 4. Push discipline while the pods live

Checkpoints every **250** steps. A push loop mirrors the newest `ckpt.pt` and the pipeline
JSONs to HF on a fixed cadence, so the worst case is losing the delta since the last push
rather than the run. **Code goes to GitHub on every commit** (already the practice).

## 5. Honest risk list

- **Biggest:** S-W at 30 k needs far longer than two days. **The run WILL be interrupted** —
  that is expected, not a failure, and the resume path is exactly what makes it fine.
  What matters is that the last checkpoint is on HF, not that the run finishes.
- **Second:** the 5.46 GB bridged corpus has never been transferred off pod5. If the pod
  dies first we re-bridge (~2 h). Annoying, not fatal.
- **Third:** the venv rebuild is the least-automated step and historically the most
  error-prone (two CUDA breakages measured). It deserves a scripted, verified recipe rather
  than a doc — a work item, not a blocker.
