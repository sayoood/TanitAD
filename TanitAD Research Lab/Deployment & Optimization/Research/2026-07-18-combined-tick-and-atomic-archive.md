# Combined deploy-tick (fp16 encoder + CUDA-graph predictor) + atomic milestone archive

- **Agent / date:** Production & Optimization (Saturday), 2026-07-18 (**run #5**, follows run #4 09:17)
- **Phase:** 0 · **Backlog:** A3 (measure the combined tick) + P1.4c (clean VRAM) + review #3 (ops-fragility)
- **Hardware / cost:** local **RTX 4060 8 GB** (declared Orin proxy, I8), exclusive; ~20 min; **$0**.
  GPU idle (1.87 GB, 6 % util, no CarlaUE4) before the run.
- **Resource declaration (G-I):** 4060 only. **Why not the eval-pod A40:** this is a **batch-1
  single-stream latency** measurement — the Orin deploy target is single-stream, so the 4060 *is*
  the right instrument (the A40 answers server/throughput, a different question). The A40 job is
  still the TRT-fp16 engine build (P0 #1), which stays **toolchain-blocked** on the dev box
  (`import tensorrt` → ModuleNotFoundError; ORT CPU EP only) and all three pods are training
  (pod2 flagship no-touch; pod1 REF-B; pod3 REF-A) → no idle pod this window; carried as run #4's job card.

> **Context (no duplication):** run #4 (09:17 today, `9c36c95`) already executed the predictor
> CUDA-graph attack (2.57×) and the numerics-safety sweep. This run advances run #4's **explicit
> A3 next-step** — replace the *additive projection* of the combined tick with a *measured* one —
> and covers the still-open review #3 (`stack/scripts/` + training loop), which run #4 did not touch.

---

## 1. Experiment — the combined deploy tick, measured (G-H / G-P2)

**Question.** Run #4 measured the two orthogonal latency levers **separately** and *projected* the
deployed tick additively: fp16-encode (compute-bound lever) + CUDA-graph-select (launch-bound lever)
→ ~9.1 ms / 109 Hz. A projection is not a measurement. This run runs **both levers together in one
decision tick** and times it end-to-end on real comma2k19 windows, with the end-to-end decision cost
(agreement + waypoint shift) beside the speed (G-P2). Tick = `encode(1 frame) + select_K9` (identical
to run #3/#4). fp32, tf32 **off**, 200 reps p50/p95 after 20-rep warmup, step-6500 ckpt, 64 real
windows, one fixed fp64 RidgeProbe (graph/precision-invariant). Script:
`Implementation/combined_tick/combined_tick_harness.py` → `combined_tick_20260718.json`.

| variant (tick = encode + select_K9) | encode ms | select ms | **tick ms** | **Hz** | ×fp32 | agreement | wp-shift (mean/max) |
|---|---|---|---|---|---|---|---|
| fp32 eager (reference) | 9.34 | 8.41 | **17.75** | 56.3 | 1.00 | 1.00 | 0 |
| fp16 encode + fp16 eager select | 6.83 | 8.24 | **15.07** | 66.3 | 1.18× | 0.969 | 1.1 cm / 5.6 cm |
| **fp16 encode + CUDA-graph select** (DEPLOY) | 6.78 | **4.37** | **11.16** | **89.6** | **1.59×** | **0.969** | **0.7 cm / 1.9 cm** |

**Findings**

1. **The combined tick is MEASURED at 11.16 ms / 89.6 Hz — and it matches run #4's additive
   projection (11.21 ms) to 0.4 %.** The two levers **compose with no interference**: the measured
   combined tick equals the sum of the separately-optimized stages. Run #4's projection is confirmed;
   the deployable recipe is real, not a paper estimate.

2. **Both levers are necessary and orthogonal, exactly as run #4 argued.** fp16 alone gets 56.3 → 66.3
   Hz (encode 9.34 → 6.83, the compute-bound win); the graph alone (on the launch-bound select) gets
   the rest, 66.3 → 89.6 Hz (select 8.41 → 4.37). fp16 barely moves *select* (8.41 → 8.24) — precision
   cannot help a launch-bound pass — and the graph barely helps *encode* — confirming the run #3/#4
   split (encoder = compute-bound → precision; predictor = launch-bound → CUDA graph).

3. **G-P2 — the whole decision cost is decision-SAFE and it is entirely the fp16 encoder, not the
   graph.** Combined agreement **96.9 %** (2 flips / 64), decoded-waypoint shift **0.7 cm mean /
   1.9 cm max** — above the ≥95 % deploy bar. The **same 2 flips** appear in fp16-eager (no graph),
   and the graph's own delta was 0.00 m in run #4 (it replays identical fp32 kernels). So the graph is
   a **zero-accuracy-cost** latency lever; the only accuracy budget spent is the known ~5 % fp16-encoder
   flip rate (run #3: 95.3 %). Net: deploy fp16-encoder + graph, keep the ViT tower ≥ fp16.

4. **The CUDA graph also makes the tick clock-ROBUST (a bonus over the mean speedup).** The graphed
   select is **1.92×** here (8.41 → 4.37) vs run #4's 1.33× (5.94 → 4.45) — *because the graphed replay
   time is near-fixed (~4.4 ms both sessions) while the eager time scales with GPU clock*. This box's
   clocks are not admin-pinnable (run #3), so eager latency drifts with thermal/clock state (fp32 tick
   17.75 ms here vs 14.79 ms run #3); the graph collapses per-kernel launch variance into one fixed
   replay → **lower latency variance, not just lower mean.** Absolute Hz is therefore clock-dependent
   (89.6 Hz this session; ~109 Hz at run #3 clocks), but all points sit 3–6× above the 10–20 Hz
   operative requirement, and the graphed tick is the least clock-sensitive.

## 2. P1.4c — clean one-process-per-precision VRAM (co-residency artifact CLOSED)

Run #3/#4 flagged their fp16/bf16 VRAM rows (1.65 GB) as co-resident-inflated — the accuracy harness
keeps the fp32 reference model alive (261 M × 2 B ≈ the 0.52 GB delta). Measured each precision in its
**own process** (`--mode vram`, exactly one model resident):

| precision | weights (theory) | **peak alloc** | peak reserved | vs fp32 |
|---|---|---|---|---|
| fp32 standalone | 1.051 GB | **1.078 GB** | 1.181 GB | 1.00× |
| fp16 standalone | 0.526 GB | **0.560 GB** | 0.583 GB | **1.93× smaller** |

fp32 reproduces run #3's 1.10 GB (harness validated); the true fp16 footprint is **0.56 GB**, not the
1.65 GB co-resident number — **never quote the co-resident figure**. The deployed mixed recipe (fp16
encoder + fp32 predictor for the graph) sits between the two whole-model bounds; the fp16-whole 0.56 GB
is the achievable floor if the predictor is also cast (safe per the fp16 policy, and it's launch-bound
so no latency cost either way). Either way the Orin VRAM envelope is comfortable.

## 3. Compliance review #3 — `stack/scripts/` + training loop (ops-fragility F-5/6/7), G-P1

**Live bug found: non-atomic milestone-checkpoint archive.** All three pod trainers preserve gate
milestones with a **non-atomic** copy to the final path — `train_flagship4b.py:337` (pod2),
`refb_train.py:358` (pod1), `refa_train_plus.py:540` (pod3) — guarded by
`if step >= m and not arch.exists(): shutil.copy2(ckpt, arch)`. A kill **during** `copy2` (the
documented pod2 self-kill / eval-OOM 2026-07-16 / Errno122-quota-full history) leaves
`ckpt_step{m}.pt` **truncated but existing**; the next save sees `arch.exists()` → **never re-archives**,
so the corrupt milestone silently stands, and the gate protocol later `torch.load`s it for D1/D2/D3 →
crash or garbage metrics. It is the **same silent-corrupt class** the atomic *resume* write
(`tmp.replace(ckpt)`, already present in every trainer) guards — the archive path was simply missed.

**The resume-write path is clean everywhere** (`train_worldmodel.py:354`, `train_flagship4b.py:326`,
`refc_train.py:136`, `refb_train.py:346`, `refa_train4b.py:303` all do `tmp → .replace`). Only the
newer milestone archive (added 2026-07-18 per Sayed) is unguarded.

**Fix + intake:** `Implementation/incoming/2026-07-18-atomic-milestone-archive/` — a shared
`ckpt_io.atomic_archive` (copy to `.partial` → `os.replace`) + `archive_milestones()` drop-in, with
**4 tests, all green (1.58 s)** incl. genuine failing-then-passing witnesses: test #1 reproduces the
live bug (a half-written milestone is unloadable and the current guard adopts it forever); tests #2–4
prove the atomic path leaves no corrupt final file on an `OSError(122)` mid-copy and self-heals on the
next save. Behaviour-identical on the happy path; three one-line call-site swaps. G-P1 satisfied
(file:line + failing-then-passing).

## 4. Actionable recommendations

- **A1 (deploy, confirmed):** the operative decision tick deploys as **fp16 encoder + CUDA-graph
  predictor/select** — **measured 11.16 ms / 89.6 Hz, 1.59× over fp32, agreement 96.9 %, wp-shift
  ≤1.9 cm**. Both levers required; the graph is a zero-accuracy-cost, clock-robust win. This replaces
  run #4's additive projection with a measured number.
- **A2 (integrate, timely):** ship the atomic-archive fix — **3 live pod trainers** are one kill away
  from a silently-corrupt gate milestone, and the whole 3-arm bake-off verdict rides on those D1/D2/D3
  milestone evals. Small, mergeable, test-covered.
- **A3 (next run):** with the combined tick measured, the remaining latency item is the **TRT-fp16
  engine** (P0 #1) on an idle pod / TRT box — the projection→measurement pattern now applies to the
  engine vs the CUDA-graph baseline. P1.6 quant stays a VRAM/energy play (run #4).
- **Handoff → Benchmarks & Eval (efficiency ledger / CNCE):** add the combined-tick row — *deploy tick
  fp16+graph 17.75 → 11.16 ms (1.59×), 89.6 Hz, agreement 96.9 %, wp-shift 0.7 cm; fp16 standalone
  VRAM 0.56 GB (1.93× < fp32); $0, 4060*. Feeds the CNCE latency term (lower tick = higher CNCE at
  fixed params). Established handoff channel (run #2/#3/#4 precedent).

---

**Gate self-check:** G-A (every claim → JSON/repo path). G-B (A1–A3 actionable). G-C (KB updated).
G-D (no hypothesis *status* change; efficiency evidence → Benchmarks CNCE per precedent). G-E (4-test
intake, green). G-H (measured combined-tick + VRAM; hardware/wall-clock/projection cross-check). G-P1
(file:line + failing-then-passing witness). G-P2 (accuracy delta beside every speed delta). G-I
(resource declared; real-compute 4060 run; TRT job card carried for the blocked A40 item).
**QUALITY: full.**
