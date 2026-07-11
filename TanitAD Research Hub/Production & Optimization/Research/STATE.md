# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-11 (run #4, Saturday agent) — branch `agent/prod-opt-20260711`
  (worktree `.claude/worktrees/prod-opt-20260711`, D-026 isolation; **branched from `1a5754d`** so the
  07-10 run #3 work is a linear ancestor → orchestrator merges ONE branch to get both). **Run #3
  (`1a5754d`, INT8 + windowing) is committed but UNMERGED into main** — see HANDOFF for the merge/
  residue note.
- **QUALITY:** full (one measured experiment with decision-grade numbers — the imagination-NLL
  `exp(-logvar)` overflow thresholds + reachability + fix parity, CPU/$0 — and one compliance review
  shipped as a 10-test standalone-green intake package; G-H, G-P1, G-P2 met. GPU was contended (100 %
  util) so the *measured* experiment was deliberately GPU-independent; the week's optimization
  accuracy-vs-speed experiment of record is the 07-10 INT8 curve now stacked in this branch — not a skip.)
- **Phase:** 0

## Where this stream stands (one paragraph)

Run #4. **Found + fixed a live silent-NaN path in the imagination loss.** `imagination_nll`
(`imagination.py:135`) runs `torch.exp(-logvar)` on an **unbounded** logvar head; it overflows to
`+inf` at logvar < **−88.72** (fp32) / **−11.09** (fp16, the deployment precision) — measured ==
theory `−ln(finfo.max)`. It is wired into the live trainer (`train_worldmodel.py:338`) with **no
nan/inf guard** before `backward`/`opt.step`, so one bad cell NaN-corrupts every weight and the atomic
save persists it. **Reachability is measured:** the NLL's own optimum `logvar*=ln(err2)` is past the
fp16 boundary for any cell with err2 < 1.53e-5, and plain SGD reaches non-finite loss in **45 steps**
(fp16) / 356 (fp32); the gradient explodes to 2.4e8@−20 / 2.8e34@−80 even before overflow. Fix
(intake `2026-07-11-imagination-nll-logvar-clamp`): clamp logvar to [−8, 8] — **finite in fp32 & fp16,
in-band parity vs the stack = 0.0 exactly** (also fixes `d9_rows`' unseeded shuffle → reproducible D9
floor); 10 standalone tests. This protects the H15/H2/LOPS uncertainty signal, not just training
stability. **Prior (run #3, `1a5754d`, unmerged): INT8 weight-quant localizes sensitivity to the
READOUT, not the ViT.**
Per-output-channel symmetric int8 weight-only fake-quant, scored in decision space on the same 64
windows (step-6500, 4060): ViT-encoder INT8 **98.4 %** agreement (1 flip, 1.6 cm), predictor INT8
**95.3 %** (3 flips, 10.9 cm) → both SAFE; but heads/whole-model INT8 collapse to **48.4 %** (33
flips, **1.67 m** wp-shift), and `int8_all == int8_heads` while encoder- and predictor-alone are
safe → the damage is entirely in **{readout, inv_dyn, imagination}**, tell = encoder-state rel-err
7e-4→1.67e-2 (only the readout produces state). **Deploy rule: INT8 the ViT encoder + predictor
weights (−552 MB of 825 MB at 4×), keep the readout ≥fp16.** This **refines the KB "ViT INT8 trap"**
— that trap is an *activation*-quant phenomenon; ViT *weights* round cleanly. Also **closed P1.4b**:
clean idle-GPU fp32 tick **15.76 ms / 63.5 Hz** (≈ the 15.07 ms baseline → the 33.5 ms 07-09 read
was contention, falsifier confirmed), fp16 **13.40 ms / 74.6 Hz / 1.18×**. Compliance **review #3**
(data-window index) shipped a fail-loud intake: silent short-episode drop across `_contract.py:120`
/ `toy_driving.py:131` / `comma2k19.py:278` → count + warn + `ValueError` on empty, parity preserved,
10 standalone tests. INT8 *latency* + the joint fp16-act × int8-weight bar still need the TRT engine
(P1.4a).

## Next actions (checkboxes)

- [ ] **Next run — P1.8 (top): add the missing `loss` nan/inf guard in `train_worldmodel.py`** before
      `backward`/`opt.step` (defence-in-depth beyond the clamp: any future non-finite loss source
      should fail loud / skip the step, never silently corrupt the checkpoint) — its own small intake.
- [ ] **P1.4a:** build the mixed engine once the TRT/ModelOpt toolchain is in (dev-box
      `tensorrt`+`onnxruntime-gpu`, or an idle pod): INT8 weights on encoder+predictor, readout ≥fp16,
      fp16 activations. Verify the **pre-registered joint bar** — agreement ≥95 % / wp-shift ≤~4 cm on
      the 64 windows; if the two ~5 % error terms compound below it, drop the predictor to fp16 and
      INT8 only the encoder (98.4 %, −228 MB, the safest single win).
- [ ] **Clean per-precision VRAM:** re-measure fp16/int8 with a single model resident (07-10's
      fp16 1.65 GB double-counts the fp32 reference) → publish the true fp16-vs-fp32 VRAM delta.
      Needs an **exclusive** 4060 (this run's was 100 % contended).
- [x] ~~P1.7 `imagination_nll` `exp(-logvar)` clamp~~ **DONE 2026-07-11** (intake, 10 tests). The
      other half (`tactical_pred` fail-fast) is **subsumed** — `tactical_pred` is an
      `OperativePredictor` (`fourbrain.py:49`), already covered by the pending review-#2 intake.
- [ ] Compliance **review #4** (`stack/scripts/` + training loop): resume/atomic-write/log hygiene,
      cgroup awareness (F-5/F-6/F-7) — fold in the `epcache` DONE-marker-unused finding.

## Standing facts / gotchas (this stream)

- RTX 4060 is the declared Orin latency proxy (I8). Clean decision-tick baseline: **15.07 ms p50 /
  1.08 GB VRAM fp32** (2026-07-08 MVP loop, exclusive GPU). **Absolute latency needs an exclusive,
  clock-pinned GPU** — the 2026-07-09 run was CarlaUE4-contended (99 % util) and read 33.5 ms; only
  the accuracy deltas and speedup *ratios* survive contention, not absolute Hz or per-precision VRAM.
- **Precision policy: fp16 on the decision path, never bf16** (measured 2026-07-09). TRT-fp16
  acceptance bar pre-registered (≥95 % agreement, ≤~4 cm wp-shift on 64 windows).
- **INT8 weight-quant policy (measured 2026-07-10): INT8-safe = ViT encoder (98.4 %) + predictor
  (95.3 %); INT8-RED-LINE = the readout / state-projection** (heads INT8 → 48.4 %, 1.67 m). The
  "keep ViT FP16" heuristic is an *activation*-quant rule and does NOT apply to weight-quant — ViT
  weights round cleanly. Clean idle-GPU latency baseline: fp32 **15.76 ms/63.5 Hz**, fp16
  **13.40 ms/74.6 Hz** (per-precision VRAM still needs a single-model-resident run).
- **Imagination-NLL silent-NaN (found 2026-07-11, fixed via intake):** `imagination_nll` runs
  `exp(-logvar)` on an unbounded head → non-finite at logvar < −88.72 (fp32) / −11.09 (fp16); no
  nan/inf guard before `opt.step` → silent whole-model + checkpoint corruption. Clamp logvar to [−8,8].
  **General lesson:** any `exp`/`1/x`/`log` on an unbounded head is a deploy-time NaN risk — grep for
  them in review #4; and the trainer still needs a `loss` finiteness guard (P1.8).
- **TensorRT not installed on the dev box** (`import tensorrt`→missing; ORT CPU EP only). The ONNX
  IR is exported + parity-clean, so only the toolchain/engine step remains.
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

**For the orchestrator — merge instructions (both prod-opt runs land in ONE branch):**
- Run #4 branch `agent/prod-opt-20260711` was **branched from run #3's commit `1a5754d`**, so merging
  `agent/prod-opt-20260711` brings in BOTH the 07-10 INT8/windowing work AND the 07-11 imagination-NLL
  work as a linear history. Do **not** separately merge `prod-opt-20260710` — it is an ancestor.
- **Merge hazard:** the `main` working tree still holds **untracked byte-identical residue** of the
  07-10 work (`Implementation/int8_quant/`, `Implementation/incoming/2026-07-10-contract-windowing-failloud/`).
  `diff` vs `1a5754d` = 0 (verified this run: the script, `windowing.py`, and the test file are
  identical). `git merge`/checkout will refuse to overwrite untracked files, so **`git clean -fd` those
  two paths on main first** (safe — they are already committed at `1a5754d`), then merge. The 07-10
  run intended to remove them but the run was interrupted before cleanup persisted.
- Run #4 itself wrote nothing to the main tree; all its files are inside the worktree/branch.

Run #4 completed cleanly. No stack/ writes (D-011). GPU 100 %-contended all run → the measured
experiment was chosen GPU-independent (CPU, $0).
