# PREREG L7-SCORER, AMENDMENT 1 (PILOT-S) -- **REGISTERED**

**Status.** REGISTERED by the Master Mind at 2026-10-22 20:19 Berlin (clock read), BEFORE any L7-S training step; drafted 2026-10-22 by the l7s_prep agent, reviewed and sealed unchanged except this status line. sha256 in `PREREG_L7_SCORER.AMENDMENT-1.sha256.txt` beside it. No L7-S number exists. `PREREG_L7_SCORER.md` (sealed 2026-10-21 23:33) is unchanged; this amends sec. 2 PILOT only.
**Why.** Sec. 2 PILOT admits E / F only; `pilot/RESULT_PILOT.md` (c3471b9): both pick worse than refcv8's own pick on held-out TRAIN episodes. L7-S (scene tokens) is a registered
arm (sec. 1) whose trainer did not exist; `train_l7.py --mode S` now does (dry run + zero-step preflight on the real train139 banks PASS, `l7s/PREFLIGHT_L7S.*`). **L7-S nests on
L7-E, not on L7-F**: it reads no model-score feature, only geometry + v0 + the detached scene tokens.

## A1. PILOT-S (adds to sec. 2 PILOT; nothing else changes)

L7-S at **training seeds 0 and 1** on the SAME train139 bank as the E / F PILOT (s0 + s1: 8,730 train / 980 validation windows; validation = 14 held-out TRAIN episodes x 2
inference seeds, split seed 20261021, validation-row sha12 `5af23a842e87`). Every sec. 1 setting unchanged (d_model 128, 2 layers, 4 heads, no cross-candidate self-attention, AdamW
3e-4, wd 0.01, batch 32, <= 30 epochs, patience 3 on validation loss, T 0.5 m, Huber 0.1, e9 masks); scene sources `bev [480x96]` + `agent [300x384]`, 480,904 parameters.
**Read only the training-split validation slice, at each run's min-validation-loss epoch, as `RESULT_PILOT.md` read E / F. No eval139 read of any kind** (the PILOT's one permitted eval
read stays unspent). **Never deciding**: no sec. 3 bar is scored. It answers training variance only (S ts0 vs ts1) -- not "another draw of episodes" (14 episodes, no interval) nor inference variance.
References (MEASURED, `pilot/*/train_log.json`; pick cost = target `cost`, ADE vs GT on the 8-slot grid, m): own pick **0.99895**; oracle-in-fan 0.35874; L7-E **1.5892** (ts0) / **1.5921** (ts1);
L7-F 1.0714 / 1.1627. Training-seed spread on this rig: E 0.0029, **F 0.0913** (largest on record).

## A2. Controls (each must read its known value, or that seed's read is WITHHELD)

* **C-S1 scene-shuffle** (new; evaluation only, never touches training): the same validation windows scored with the scene tokens of a DIFFERENT episode (`cross_episode_map`,
  deterministic), logged per epoch as `val_scene_shuffled`, read at the best epoch. Known value if S reads the scene: shuffled pick cost is **worse** than the true-scene one.
  (Without it "S beats E" cannot be told from "a decoder with more capacity beats an MLP".)
* **K7** `comf` held-out AUC >= 0.95 at the best epoch, each seed; finite loss every epoch. **K6** `assemble_inputs` admits only `bev/agent` tokens + pads (tested).

## A3. Committed reading, fixed before any S data

Per seed s: `own` = 0.99895; `E_s` as above; `nu` = |S_ts0 - S_ts1|; `g_own` = own - S_s; `g_E` = E_s - S_s; `h` = S_shuffled_s - S_s; `M_own` = max(nu, 0.0913); `M_E` = max(nu, 0.0029).
First matching row; every condition must hold at BOTH seeds, with K7 holding at both.

| outcome | condition | what follows |
|---|---|---|
| **O1 BEATS-OWN** | `g_own > M_own` and `h > nu` | The full-bank read trains L7-S first; **P-L7S is its headline arm** (E, F still trained as the PV controls). Sec. 3 bars and the sec. 6 navtest rule unchanged. |
| **O2 READS-THE-SCENE** | not O1, but `g_E > M_E` and `h > nu` | Scene signal beyond geometry + v0, not enough to beat the shipped pick at this bank size. The sec. 4 "P1 fails" order stands (trunk-gradient scorer first); S stays the PV candidate; E, F, S as registered. |
| **O3 NO-SCENE-SIGNAL** | otherwise (S does not beat E by `M_E`, or `h <= nu` at either seed) | Reported as "a detached-token scorer at 8,730 windows does not read the scene". S goes last in the full-bank queue (still a registered arm); sec. 4 levers (1) trunk-gradient and (2) the F feature route go first. |
| **WITHHELD** | K7 miss, non-finite loss, or C-S1 unreadable at a seed | That seed is TRAINING-SUSPECT, no row drawn. ONE re-run from scratch, identical settings, training seed 2, both reported; **no hyper-parameter change** (F ts1's K7 0.926 is the precedent). |

S vs F is reported, never decisional. Report S, S-shuffled, own and oracle together, with per-head AUC (comf / nc / ttc / dac) per seed.

## A4. Cost, and two facts that do not depend on the outcome

CPU only (`OMP_NUM_THREADS=4`), two runs sequential. MEASURED, 4 preflight repeats (`l7s/PREFLIGHT_L7S.log`, box shared with a launch gate): 0.37-0.39 s per batch of 32, **126-133 s per
epoch** incl. both validation reads and the shuffled read, so **<= 63-66 min per run at 30 epochs** (patience 3 normally stops earlier; E / F's best epochs were 2-8); peak working set 1.09 GB;
the optimizer step is not in that number. Scorer training needs the PI's go. MEASURED on train139: the scene members of s0 and s1 are **bit-identical** (equal zip CRC32 + size), i.e. scene
tokens do not depend on the inference seed; `r7_agent_pad` is all-False (0 of 4,855 x 300 slots padded) and `r7_bev_pad` is absent.

**Not changed:** sec. 1 arms, sec. 3 bars, sec. 4 outcomes, sec. 5 (PILOT-S is a training-seed pair), sec. 6, sec. 7, sec. 8 (nothing is fitted, tuned or early-stopped on eval139).
