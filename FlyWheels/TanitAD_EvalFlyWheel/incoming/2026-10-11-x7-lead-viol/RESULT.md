# X7 — LEAD-VIOL instrument + 6 s lead block (EvalFlyWheel, 2026-10-11)

The Master Mind wrote this from the implementing agent's hand-back, because the harness refused the agent's own report
file. Code and tests are in this commit; the raw control numbers are in `raw/real_block_controls.json`.

## 1. LEAD-VIOL (MEASURED: tests run)

**Definition used** (`Project Steering/SPEC_REFCV8_DRAFT.md` ~l.768): the share of LEAD windows in which the plan's
minimum time gap is **< 1.0 s** while the human's is **≥ 1.0 s**.
* The plan side is strict `<` and the human side is inclusive `>=`.
* A NaN time gap is not a violation.

**⚠ Denominator — the draft does not define it.** Two readings are reported side by side:
* **headline:** over ALL lead windows, including those where the human is already under 1 s;
* `share_of_at_risk`: the conditional reading, emitted beside it.

**→ SPEC §8.7(a) must pin one before registration.**

**Code and tests:**
* Code is in `taniteval/taniteval/lead_metrics.py`: `lead_viol`, `paired_lead_viol` and `lead_viol_vs_human`, with an
  episode-cluster bootstrap and a paired A−B delta.
* It is wired into `four_families.longitudinal` as `distance_keeping.lead_viol`.
* `test_lead_metrics.py` needed a one-key change, because it pins the exact `_per_window` key set.
* `taniteval/tests/test_lead_viol.py`: **31 passed**. The hand fixture has 1 violation in 5 lead windows = **0.2**.
* The tests also cover:
  * boundary values;
  * a human-as-plan control that must read **0**;
  * a STOP control;
  * a deranged-lead regression (0.2 → 0.0);
  * 7 mutants that go RED;
  * a real-source mutation (12 RED, then restored).

**Comparison against the tip:** the Master Mind compared the full `taniteval/tests` suite on the pure tip with and
without this change (plus X3/X10). The failure sets were **identical** (60 failed / 17 errors, all missing-data in an
archive tree), and the change added **68 passing** tests.

## 2. The 6 s lead block (MEASURED: built on the dev box in 14 s, no GPU)

The 147-clip build covers eval139 completely. The artifacts stay **out of git**: they rebuild in 14 s, and they are kept
with their `.report.json` files on the dev box at `C:/Users/Admin/qland/landed_hold_x7/artifacts/`.

| block | md5 | grid |
|---|---|---|
| `b1_eval_lead_block_k12_dt0p5.npz` (**recommended**) | `c346dff3b3d356f55ef2786f327c6c91` | shares every instant of both refcv3 grids |
| `b1_eval_lead_block_k30_dt0p2.npz` | `ff112bb3f4af373bad6a34019820acf4` | 6 s grid fully; only {1, 2} s of the 2 s grid |

**Control on the real block (hold-v0, 6 s, dt 0.5):**

| block | LEAD-VIOL |
|---|---|
| 6 s | **0.096 [0.056, 0.138]** |
| banked 2 s | **0.024** |

⇒ **The 2 s block is nearly uninformative for X7. Bar X7 on the 6 s block.**

## 3. Findings the SPEC must absorb

* **STOP reads exactly 0, so LEAD-VIOL can be won by braking.** Pair the X7 bar with ego-progress, or with the hold-v0
  control.
* **A plan that leaves the lane has an undefined time gap and is not a violation.** This is MEASURED: lost-lead counts
  are 117–211. These windows are counted separately and must be reported beside the share.

## 4. The refcv7 @ 50,400 baseline row (NOT run)

* **Dumps:** `D:/refcv7_eval_kit/battery/step50400_a6text/dump_s0|s1` (139 episodes, 4,754 windows). They keep only the
  first 4 model slots, i.e. the 2 s grid.
* **A 6 s row needs a GPU re-roll** with `refcv3_arm --grid 6s`; the battery hard-codes `--grid 2s`. **This is a PI
  compute decision.**
* **A CPU-only interim row** (2 s horizon, using the k12 block) is possible now. Its exact flags are UNVERIFIED.
* **UNVERIFIED:** whether `refcv3_arm._distance_keeping` forwards `state` / `lead_valid` / `eid`. If it does not,
  LEAD-VIOL silently falls back to a flagged human-corridor denominator.
