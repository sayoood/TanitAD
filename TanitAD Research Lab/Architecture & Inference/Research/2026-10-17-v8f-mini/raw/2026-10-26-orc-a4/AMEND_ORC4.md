
## Amendment 4 (`A4-2026-10-26`), before any run: two instrument defects found on real inputs, fixed before data

**When.** Written 2026-10-26 ~12:40.
* The box-2 chain's first launch gate (ORC s0, 12:20) REFUSED, and the chain stopped before any training.
* The dev-box T1 chain was built and smoked on a stand-in, the banked M2b; every output of that smoke is labelled SMOKE.
* No O-RECOVER fine-tune, checkpoint, rollout or eval exists.

### 1. The ego probe refused a correct rebuild on a row AT REST

**MEASURED on box 2's own gate batch** (`raw/2026-10-26-orc-a4/diag_egoprobe_s0.json`). The trainer ran unchanged, on the gate's argv, with
a wrapper that dumps the selected rows when the probe refuses.
* 1 of 29 selected rows (row 25) had logged v0 = 0.0 and a0 = 0.0, and |base| = 0.0.
* The perturbation moved its a0 to −0.528 m/s² and its κ0 to −0.0099.
* At zero speed, a non-positive acceleration keeps the kinematic base at rest, so its base stayed identically 0.
* The base of every one of the other 28 selected rows changed.

**Fix (`v8f_mini_train.py`, `ego_perturb_probe`):**
* The base clause exempts exactly the selected rows AT REST: logged v0 = 0, perturbed a0 ≤ 0, and base identically 0 before and after.
* It counts them (`base_rows_at_rest_exempt`). Every other selected row must still change, and at least one such row must exist.
* `p10f`, `kin3` and `p10_t` must still change on EVERY selected row.

**Evidence.**
* Three tests in `test_v8f_ego_perturb.py`:
  * the unmodified batch exempts none, and a row put at rest is exempted and counted;
  * a moving row whose perturbed a0 ≤ 0 is still refused when its base is not rebuilt;
  * an all-at-rest batch is refused.
* Three mutants in `mutate_orecover.py`, one per clause.

The lever, its law and every other gate check are unchanged.

### 2. K4's comparator differed from tick 0 by construction

K4 (§3, Amendment 2) requires tick 0 of the `--limit logged` rollout to equal the arm's own T0 eval to 1e-3. Tick 0 runs at fresh ages
(0, 0), while box 2's standard `evalorc` runs `--ages cycle`.

**MEASURED** with the reader's own K4 on a banked checkpoint and its banked logged rollout (`raw/2026-10-26-orc-a4/k4_demo.out`, from the dev-box T1 chain's build):
* Against a fresh-age T0 eval, K4 passes: max |Δpos| 4.4e-5 m, lat / lon F1 Δ 0.0 / 0.0, 2,399 / 2,399 argmax equal.
* Against the cycle-age eval, K4 fails: Δpos 6.05 / 5.72 / 5.32 m.
* T1-HEADS' K4, which §3 names, compared against the fresh-age eval.

**Fix:**
* K4's comparator is the arm's own T0 eval at fresh ages (`v8f_mini_eval.py run --ages fresh`, the SFRESH argv). It runs on the dev-box CPU,
  once per run, about 21 s each.
* R-T0, T0-CKPT and the four families stay on box 2's standard eval, as written.
* The reader takes the two from separate manifest keys, so K4 can never read the standard eval and R-T0 can never read the fresh one.

Nothing else moves: the arms, the law, the rows, F, the bars, the outcome table, the committed next steps, and Amendments 1 to 3.
