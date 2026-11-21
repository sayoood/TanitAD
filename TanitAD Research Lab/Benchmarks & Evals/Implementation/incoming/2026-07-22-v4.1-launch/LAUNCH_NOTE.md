# flagship v4.1 — 30k run LAUNCHED and VERIFIED TRAINING (the lr_trunk fix)

`2026-07-22 18:35:51 local (16:35:51 UTC)` · pod `tanitad-pod2` (NVIDIA A40) · Sayed-authorized
v4.1 restart (relayed via coordinator). Evidence class: **MEASURED** unless marked otherwise.

## Why v4.1 — v4's WM canary degraded, cause = trunk LP-FT (MEASURED)

v4 (`lr_trunk=3e-4`) WM-integrity canary, from `flagship-v4-30k/train.log`:

| step | v4 canary_ade@2s | vs base | v4 controller_action |
|---|---|---|---|
| 0 (base) | 0.42148 | — | — |
| 500 | 0.599 | +0.177 | controller_halved_lambda |
| 1000 | 0.814 | +0.393 | hard_breach_1/3 |
| 1500 | 1.305 | +0.883 | hard_breach_2/3 |
| 2000 | 0.943 | +0.521 | **alarm_lambda_to_zero** (`lam_mult=0`) |
| 2500–4000 | 0.88–1.48 | — | still `alarm_lambda_to_zero` |

The canary kept climbing **after** the controller zeroed the planner gradient at step 2000
(`lam_mult=0` from 2000 on) — so the damage is the **trunk fine-tuning**, not the planner. v4 was
stopped at ~step 4000 (PID 75844 killed by explicit PID, GPU confirmed freed).

## The fix (MEASURED applied)

- **`--lr-trunk 3e-5`** — 10× lower than v4's 3e-4. Confirmed live in the run: lr_trunk at step 0 =
  2e-08 (v4 was 1.5e-07), step 500 = 7.52e-06 (v4 was 7.52e-05) — exactly 10× down.
- **No phase-A trunk-freeze knob exists** in the trainer (`grep` of `train_flagship_v4.py`: the trunk
  is a single AdamW param group at cosine-scheduled `lr_trunk`, trained from step 0; no
  `freeze`/`unfreeze`/`requires_grad` toggle). So the global lr cut is the only no-code-change lever;
  I did **not** add a freeze mechanism (out of scope, would need a re-test). NOTED for a future rev.
- Warm-start: the **v1 trunk `flagship4b-speedjerk-30k`** (NOT the degraded v4 ckpt). Fresh
  `--out flagship-v4.1-30k` ⇒ no resume ⇒ clean warm-start from `--trunk`.

## STATUS: 🟢 LIVE AND TRAINING (MEASURED)

| fact | value | source |
|---|---|---|
| PID | **79542** (`/usr/bin/python3`) | `nvidia-smi --query-compute-apps` + `kill -0` |
| out dir | `pod2:/workspace/experiments/flagship-v4.1-30k` | — |
| GPU lock | `v4.1-train` tied to `--pid 79542` | `gpu_lock.sh` |
| step-0 canary baseline | 0.42148 (n=881) | step-0 `canary_baseline` row |
| step-0 loss (finite) | total 1202.75 | step-0 row |
| loss trajectory | 1202.75(s0) → 904.13(s50) → 122.36(s100) → 23.37(s150) → 10.31(s500) | train rows |
| phase-A rate | ~1.55–1.6 s/step (step 500 @ elapsed_s 799.4) | train rows |
| GPU | 100% util, ~33 GB, ~277 W | `nvidia-smi` |

⚠️ Phase-A rate only (`lambda_plan=0`); planner (phase B) + long-horizon-k=50 (phase C) raise s/step.
Gate at step 10000 — keep GATE_PROTOCOL ~10.9 s/step (~30 h) as the conservative bound.

## ✅ THE CANARY VERDICT (the whole point) — matched-step, v4.1 vs v4

| step | v4 canary (breaching) | **v4.1 canary** | v4.1 controller |
|---|---|---|---|
| 0 (base) | 0.42148 | 0.42148 | — |
| **500** | 0.599 (halved_lambda) | **0.45112 (+0.030)** | **ok** |
| **1000** | 0.814 (hard_breach_1/3) | **0.45994 (+0.038)** | **ok** |

The fix is **CONFIRMED working** at both checkpoints: v4.1's canary stays **flat near baseline**
(0.421 → 0.451 @ 500 → 0.460 @ 1000, controller **"ok"** throughout), while v4 was already
**breaching** at the same steps (0.599 halved-λ @ 500, 0.814 hard-breach @ 1000, en route to 1.30 @
1500). v4.1's world model is NOT degrading — `lr_trunk 3e-5` fixed it. (The in-loop `val ade@2s`
improves 3.27 @ 500 → 1.17 @ 1000 as the planner warms; ≈ or better than v4 at matched steps, and not
the metric the fix targets.)

## HF push of the flagship v1.6 ckpt — ⛔ BLOCKED, ESCALATED (needs Sayed)

Coordinator step 1 (unblock the sibling AlpaSim task): push the flagship v1.5/v1.6 head+ckpt to
`Sayood/`. Selected **v16-ab-ft** (best: val ade@2s **0.442** vs v15-abc 0.534, and self-contained —
`ckpt_best.pt` carries `model`+`head`+`grounding`, 1.23 GB, no separate trunk).
- Repo `Sayood/flagship-v16-ab-ft` **created (private, EMPTY)**; token valid (write OK).
- **Upload failed: HTTP 403 "Private repository storage limit reached."** The 1.23 GB LFS file exceeds
  the account's free **private** storage quota.
- **The fix requires a publishing decision I will not make unilaterally**: make the repo **public**
  (the established 2026-07-16 workflow used public-gated `Sayood/` model repos) OR upgrade private
  storage. Making content public is gated on **Sayed's explicit word** (my safety rules + the repo's
  own `LOOP_STATE` "AWAITING SAYED" for HF pushes) — an agent relay does not authorize it.
- **Turnkey once approved**: the uploader is staged at `pod2:/workspace/hf_upload_v16.py` (reads the
  token from a tmpfs file `/dev/shm/hf_tok`, deletes it on read; set `private=False` or flip the repo
  public, re-supply the token, re-run). Files it pushes: `ckpt_best.pt`, `anchors256.pt`, `probes8.pt`,
  `config.json`, `eval_v16.json`, `metrics.json`, `README.md`. Token handling: read in place from
  `Keys.txt`, never argv, never persistent disk; the tmpfs copy is confirmed deleted (`TOKEN_ABSENT_OK`).

## Deliverable manifest

| artifact | where |
|---|---|
| this note | `repo: …/incoming/2026-07-22-v4.1-launch/LAUNCH_NOTE.md` (STAGED) |
| v4.1 launch/preflight/smoke script | `repo: …/2026-07-22-v4.1-launch/run_v4.1_launch.sh` (STAGED) + `pod2:/workspace/run_v4.1_launch.sh` |
| v4.1 training process | `pod2` PID 79542 (detached nohup) — run, not a file |
| v4.1 log + pid | `pod2:/workspace/experiments/flagship-v4.1-30k/{train.log,train.pid}` |
| HF uploader (ready, blocked) | `pod2:/workspace/hf_upload_v16.py` |
| generic monitor | `pod2:/workspace/monitor_generic.sh` |

## Next / escalation

1. 🔴 **Sayed decision on the HF push** (make `Sayood/flagship-v16-ab-ft` public, or upgrade private
   storage) — the ONLY blocker for the AlpaSim sibling. Turnkey recipe above.
2. v4.1 canary CONFIRMED flat through step 1000 (0.451/0.460, "ok"). Keep watching past step 1500+
   (v4 peaked 1.30 @ 1500, then rode `lam_mult=0`); if v4.1 ever climbs, escalate before the gate.
3. **G1 gate at step 10000** — run `run_gate.py` on the gate ckpt (`flagship-v4.card.json`).
4. Do NOT eval on pod2 while it trains; do NOT recycle `flagship-v4.1-30k/`.
