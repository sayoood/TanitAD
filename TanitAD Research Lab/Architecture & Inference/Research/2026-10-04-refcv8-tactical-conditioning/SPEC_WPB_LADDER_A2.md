# SPEC_WPB_LADDER — AMENDMENT A2 (Master Mind, 2026-10-11 ~18:30 Europe/Berlin)

Registered before any gradient-share reading exists: the ladder's first launch (13:44:42Z, tree `235156d`) died in V0
chunk 1 inside the first reading, at row 250, with no checkpoint and no `gs_*` row written. No arm has a scored or an
instrument number. Every clause of `SPEC_WPB_LADDER.md` (sha256 `e1b6aff7…`) and A1 (`cbd64fa2…`) applies unchanged —
the arms, the levers, the seeds, the floor F, the bars, the reading order — except the three EXECUTION clauses below.

## Why (MEASURED, Thor, 2026-10-11)
* V0 trains at **3.15 s/step**; setup ≈ 348 s.
* The first reading under a bf16 trunk is the fp32 replay (`grad_share.fp32_replay`, TF32 off, eager): forward ≈ **228 s**,
  then **≈ 181 s per retained `autograd.grad` pass** (one for the total, one per term, one for `rest`), from the probe
  timeline (`tasks/bahl8rgac.output`): forward start 16:56:34 → pass total 17:00:23 → traj 17:03:24 → agent 17:06:26.
* With every term separated, one reading is 7 passes ≈ **1,496 s ≈ 25 min**. At the base SPEC's `--grad-share-every 250`
  that is 12 readings ≈ **5 h per 3,000-step arm — more than the arm's training (≈ 2.6 h)** — and the first one alone
  pushed V0 past the 2,400 s chunk cap with no checkpoint.
* bf16 in-graph readings are not an option: they fail the 1e-4 linearity control (MEASURED 5.9e-3, `RESULT_L5_TF32_REPLAY.md`).

## Validation smoke (MEASURED 2026-10-11 ~18:41-18:55 CEST, Thor, AFTER the text above was written; a debug tree, NOT a ladder arm)
V0's exact argv + `--grad-share-terms tac_v6,map_hires --grad-share-at 50 --steps 60` (`gsdbg/k2`). The first attempt
(`gsdbg/k`) was REFUSED before step 1 by G-DVB: the two new flags had no declared-vs-built entry. They are now registered
as `runtime` levers. The CPU tests had not reached that guard; the smoke did. The reading at row 50 then:
* 4 passes;
* fp32 forward 228 s, measure 560 s (map_hires 18 s, tac_v6 181 s, rest 181 s), replay total **788 s**, inside `GS_S` = 1,100 s;
* `gs_trunk_lin_rel_err` **0.0** and `gs_bev025_lin_rel_err` **5e-6** (bar 1e-4);
* every key ladder_score / ladder_i0 read is present.

Its values are a V0 debug reading at row 50: no consumer reads them and no ladder number is derived from them. One
is recorded here so its existence is not hidden: `gs_trunk_proj_tac_v6` = -2.9e-5.

## The amendment
1. **`--grad-share-terms tac_v6,map_hires` on every arm** (new trainer flag, `stack/tanitad/train/grad_share.parse_keep`).
   Only these two terms are separated; every other term folds into `gs_*_rest`. The kept terms' `proj` / `norm`, and every
   `norm_total` / `lin_rel_err`, are **identical** to the full reading. This is pinned analytically (closed form, keep one
   of three terms) and on the real agent rig (rel 1e-5) by `stack/tests/test_refcv8_grad_share_keep.py`, with two
   deliberate regressions that go RED. Cost: 4 passes ≈ 953 s.
   **What is lost:** per-term shares of traj / agent / box3d / the other terms on ladder arms. **No ladder consumer reads
   them:**
   * `ladder_score` reads `gs_trunk_proj_tac_v6`, `gs_trunk_norm_tac_v6`, `gs_trunk_norm_total` and `gs_bev025_proj_map_hires`;
   * `ladder_i0` I0-c reads `gs_*_lin_rel_err`.
2. **`--grad-share-at gs_at(N)` on every arm** (new trainer flag, `grad_share.parse_at` / `due`):
   * ONE reading row inside §4.2's sizing window (0, 0.2 N];
   * TWO inside the L2 / L3 premise window (0.5 N, N];
   * at N = 3,000 that is rows **500, 2000, 2500**, pinned by `stack/tests/test_refcv8_grad_share_at.py`, against both
     windows' literal bounds.

   Cost ≈ 3 × 953 s ≈ **48 min per arm** instead of ≈ 5 h.
   **What is lost — disclosed, not hidden:**
   * §4.2's k is sized on the median of **1** reading (was 2: rows 250, 500);
   * the L2 / L3 premise medians are over **2** readings (was 6).

   Both remain instrument outputs, not scored outcomes, and both keep their bars as written (L2: median
   `gs_trunk_proj_tac_v6` ≥ 0.01; L3: median `gs_bev025_proj_map_hires` > 0). A premise read on n = 2 is reported with
   its n, and a reading that misses its bar by less than the spread of its two values is reported as **UNDERPOWERED**, not
   as a FAIL.
3. **Jobs ≤ 60 min** (§1 said ≤ 40 min): `ladder_chain.sh` `CAP_S` 2400 → 3600. The chunk predictor now adds
   `GS_S` = 1,100 s for any save interval that holds a `--grad-share-at` row, which it reads from the arm's own argv, so a
   reading interval is never cut mid-way. The trainer's implicit resume from `<out>/ckpt.pt` is unchanged.
4. **`ladder_eval.py` stores one more per-window array, `fan_min_vmax`** (ADDITIVE; no registered reading changes):
   the fan's smallest planned max speed, captured inside the existing `SpeedCeilingFilter` spy with the filter's own
   `planned_max_speed`. With `v_lim` already stored, it gives SPEC_REFCV8's X3 **exempt-row rate** (rows where no
   candidate complies, so the argmax filter keeps the whole fan). Verified against the real filter on 20 random fans:
   732 exempt rows = its own `speed_rows_empty` count, while a `max` mutant reads 882.

## Not changed
The one-variable audit (`ladder_arms.audit`) passes on every arm. The new tokens are common to all arms, so no rung can
read them as a second variable — the same treatment as `--pose-sync-sidecar` (X10). Seeds, N, eval rows, sampler seeds,
floors, bars and reading order are unchanged.
