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

## 4b. ⛔ THE BACKUP WAS SILENTLY 100 % BROKEN — found and fixed 2026-08-14

**MEASURED: `v6F-SW-30k/ckpt.pt` had NEVER pushed. Zero successes, all day.**
Every attempt aborted at ~1 % (23.9 MB of 3.53 GB). The push loop was alive and
reported `pushed 6 [...]` every cycle — but only the small JSONs were landing, and
**the loop's log TRUNCATED the error to `"Bad request for commit endp"`**, so a
100 %-failing upload rendered as a healthy heartbeat. Re-running the upload with the
traceback captured gave the real message:

> `Bad request for commit endpoint: Private repository storage limit reached,
> please upgrade your plan to increase your private storage limit`

HF **private** storage was full (`canPay=False`). Public repos were irrelevant to it;
the private consumers were **~53 GB belonging to a different project**
(`final_technical_dataset` 18.23 GB, `final_user_friendly_dataset` 18.23 GB,
`final_technical_model_Qwe25_7B` 16.58 GB) — left untouched, deletion is the PI's call.

**The fix, on PI direction (2026-08-14): `Sayood/tanitad-v6` is now PUBLIC + GATED with
manual access review.** Applied in the safe order — `gated="manual"` set **first while
still private**, verified, and only then `private=False` — so there was never a window
in which the weights were openly downloadable. Verified state: `gated=manual`,
`private=False`.

**Verified from the FAR SIDE (repo listing, not the push log):**

| file | size on HF |
|---|---:|
| `v6F-SW-30k/ckpt.pt` | **3 528.178 MB** ✅ |
| `v6F-SW-30k/weights_fp16.pt` | 673.305 MB ✅ |
| `v6F-SW-30k/config.json` · `metrics.json` | ✅ |

⚠️ **The durable lesson is the truncated log, not the quota.** A backup is not verified
by its writer's log — the summary line counted files *attempted*. **Verify from the far
side.** `stack/ops/ckpt_fp16_snapshot.py` now carries a `verify_remote()` that lists the
repo and checks the size, and that is the check every future push loop must end with.
*(Same family as the monitor-echo trap in `CLAUDE.md`: a status line that reports on its
own intent rather than on the observed result.)*

**New tool: `stack/ops/ckpt_fp16_snapshot.py`.** Weights-only fp16 snapshot,
**3.53 GB → 0.67 GB (5.3×), MEASURED at 336 559 305 params** — matching config E exactly.
Enough for the P-battery, any eval, and `--init-from`; **NOT** enough for `--resume auto`
with optimiser state. It carries `config` on purpose, because `load_resume` is a strict
load and a snapshot without its config is a refused restart on the new pod.

## 4c. Measured training state — the run will NOT finish before the pods are replaced

| | MEASURED 2026-08-14 |
|---|---|
| step | **2 000 / 30 000** (6.7 %) |
| rate | **17.32 s/step** (the trainer's `step_s_note` confirms real per-step, not the ÷`--log-every` trap) |
| loss | 450 → 3.995 · 950 → 2.058 · 1 800 → 1.451 (descending) |
| `o5_k` | **60** → the 6 s rollout contract is active |
| O-battery | o1/o3/o5/o6 all logging (`o1_factual_ade` 0.312, `o3_visible_err` 0.440, `o5_growth` 0.753, `o6_sigreg` 7.88) |
| ⚠️ `gnorm` | **85.07** — high; worth a watch, not yet an alarm |

⇒ **30 k steps = 144 h = 6.0 days at this rate. The PI's pods have ~2 days left.**
48 h buys ~+9 980 steps, so the run reaches roughly **step 12 000 of 30 000 (~40 %)**
before replacement. **The resume path is therefore the CERTAIN path, not a contingency**,
and the fresh pods must carry ~4 more days of S-W training. That is the single most
decision-relevant number in this document.

**pod4 / PH0 production, MEASURED same time:** VLM (v2) stage complete for **600 clips**;
SAM3 stage running (PID 843758), emitting detections per clip. GPU 100 %, 4.6 GB.

## 5. Honest risk list

- ~~**Biggest:** S-W at 30 k needs far longer than two days.~~ **CONFIRMED AND QUANTIFIED
  2026-08-14 (§4c): 17.32 s/step ⇒ 6.0 days for 30 k, so the run reaches ~step 12 000
  (~40 %) before the pods are replaced.** The interruption is certain, not a risk. It is
  survivable **only because §4b's backup failure was caught** — until 2026-08-14 the
  checkpoint had never once reached HF, so this risk was silently a total-loss risk.
- **Second:** the 5.46 GB bridged corpus has never been transferred off pod5. If the pod
  dies first we re-bridge (~2 h). Annoying, not fatal.
- **Third:** the venv rebuild is the least-automated step and historically the most
  error-prone (two CUDA breakages measured). It deserves a scripted, verified recipe rather
  than a doc — a work item, not a blocker.
