# PREREG_V8F_FLAT: does v8F-mini's hierarchy beat its own capacity-matched FLAT control? (written 2026-10-25, before any instrument or any FLAT data)

**Status: queued after `PREREG_V8F_MIXO` on box 2.** It runs under the PI's go of 2026-10-25: *"Go, act autonmously until positive results are achieved … Use only tge resozrces I allocated for you … our goal is to validate the v8f by real dominant driving results."* Training starts only after a launch-gate PASS.

## Why this arm, and why now (evidence class: MEASURED, cited)

The programme's thesis is that the hierarchy drives better than a flat planner. **v8F-mini has never been read against a flat control.**

* **`H-MINI-K-1` PASSED against P10, a kinematic prior, not a learned flat planner.** M2b − P10 ADE6 reads −0.3117 [−0.4635, −0.1656] / −0.4324 [−0.5654, −0.2961] at s0 / s1, separated (`RESULT_FULL.md`).
* **TAC-USED showed that O DEPENDS on T's code, not that the code helps.** A donor episode's code costs 4.4–6.4 m ADE6 (`RESULT_V8F_TAC.md`). A planner trained to lean on an input is hurt when that input is corrupted, whether or not the input beats having no input at all.
* **The oracle headroom (own 2.175 / 2.055 → full GT code 0.971 / 1.004 m, `raw/2026-10-25-kind/run/kind_read.json`) does not show the hierarchy's value either.** A GT code is a function of the logged future, so a causal model cannot reach it.
* `2604.03208` Table 16 is the published form of this control: flat and hierarchical at matched capacity. It is the comparison that turns *"we built a hierarchy"* into *"the hierarchy is worth X"*.

⇒ The cheapest discriminating experiment for *"validate v8F"* is a flat O that is matched in every other respect.

## The arm (ONE lever: whether O reads the hierarchy above it)

* **FLAT = M2b + `--o-interface cut`.** O reads a CONSTANT in place of both of its hierarchical inputs:
  * T's tactical code is replaced by zeros, after every override, so the oracle and donor evals cannot reach O either;
  * S's checkpoints are replaced by the nominal `ck_nom`, with `ok = 0`.
* **Everything else is B's, so the control is capacity- and multi-task-matched:**
  * O keeps every parameter and every other conditioning token: ego, nav, limit, age and the scene;
  * S and T are still built and trained on their own losses, and T still reads S, so the shared adapter receives the same auxiliary gradients;
  * data, steps (19,164), p_gt schedule, WM auxiliary, windows and eval are unchanged.
* **The default `--o-interface full` is B,** and the launch gate must prove that bit for bit.
* ⚠️ **What FLAT still has: the route.** O reads nav in both arms (nav is an input at training and inference, PI ruling). The read therefore asks what S and T add *beyond* a planner that already knows the route. That is the honest question, and a harder bar than flat-without-nav.

## Proofs before any run (launch gate, `SPEC_REFCV7.md` §2; each with a deliberate-regression arm that must FAIL)

1. **G-DVB:** `--o-interface` has an entry, and the built model's value equals the argv.
2. **Identity:** `full` reproduces B's built model and a 200-step loss trace exactly.
3. **Structural zero:**
   * under `cut`, O's `traj60` is bit-for-bit invariant to S's checkpoints and T's code. Perturb both (a donor code, GT code, random code; shifted checkpoints) and require max|Δ traj60| = **0.0**;
   * under `full`, the same perturbation must move `traj60`. This is the control that proves the probe can see a change.
4. **Gradient reachability:** under `cut`, `e_code.weight` receives a gradient of **exactly 0.0**; under `full` it is non-zero. Under both, S's and T's heads receive non-zero gradients (they are still trained).
5. **Mutant:** delete the cut line. Checks 3 and 4 must go RED.

## The reading (tier T1-class open loop, eval-141, 141 episodes / 2,399 windows; paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m, 2F = 0.2415 m)

Both seeds are trained. The verdict needs both, because a hierarchy claim on one seed is exactly what `H-ESTIM-SEED-1` forbids.

| read | definition | role |
|---|---|---|
| **PRIMARY** | own-code ADE6, Δ_s = M2b_s − FLAT_s (negative = the hierarchy is better), at s = 0 and s = 1 | **≤ −2F and separated, at BOTH seeds** |
| families | longitudinal (incl. distance keeping), lateral, tactical (T is still trained in FLAT, so its decision metrics are read on both), strategic (NOT PAUSED for v8F-mini) | beside, per family, never pooled |
| strata | Δ_s within JUNCTION / CURVE / NO-TURN (`raw/2026-10-25-kind/window_strata_eval141.npz`; strata under 10 episodes get a point estimate only) | descriptive; no bar |
| identity | FLAT's GT-code oracle minus own ADE6 must read **exactly 0** at both seeds | a structural control; a nonzero value voids the run |

| outcome | condition | what it means | next (committed) |
|---|---|---|---|
| **HIERARCHY BEATS FLAT** | Δ_0 ≤ −2F and Δ_1 ≤ −2F, both separated | S and T add at least 2F over a route-aware planner of the same capacity | register `C-V8F-HIER-1`; the interface-quality arm (MIXO's outcome, then T-side) is built on it |
| **NOT ESTABLISHED** | either seed misses −2F or is not separated, and neither seed reads ≥ +2F | the hierarchy, as trained, has not shown 2F of value over flat. Reported to the PI plainly: **v8F is not validated against flat at 2F** | the interface is the defect. Run the lever MIXO names, then repeat this read on that arm |
| **FLAT BETTER** | Δ_0 ≥ +2F and Δ_1 ≥ +2F, both separated | at own codes the interface HURTS: O leans on a code that is worse than nothing | an O-robustness arm (code dropout or MIXO-type mixing toward an empty code) under its own PREREG |
| **SEEDS DISAGREE** | one seed ≤ −2F separated, the other ≥ +2F separated | the seed floor dominates | seed 2 of BOTH arms (B has no seed 2 banked), 2 × B's wall-clock on box 2 |

⛔ No goalpost moves after data. A "better at JUNCTION only" pattern stays descriptive unless a new PREREG commits it first.

## Compute and order

* **Box 2:** about 2 × B's banked wall-clock per seed (`summary.json` `seconds`, about 4.4 h; ESTIMATED), plus 2 × the eval set (own, oracle identity, families).
* **Order:** after MIXO's chain ends.
  * It proceeds after MIXO's ZZV8FMINI-DONE.
  * It also proceeds after a ZZV8FMINI-STOP that is a verdict-level or run-level stop.
  * It does NOT proceed after a guard, thermal, reboot or unreachable stop: those are box-level, and the box is not healthy.
* Box 2 runs one job at a time, in SAFE MODE, under the temperature guard.
