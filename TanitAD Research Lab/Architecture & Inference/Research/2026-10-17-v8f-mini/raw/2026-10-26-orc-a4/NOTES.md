# raw/2026-10-26-orc-a4: PREREG_V8F_ORECOVER Amendment 4 (before any run)

## 1. The gate refusal and its diagnosis
* **12:20.** The O-RECOVER chain's first launch gate (ORC s0) REFUSED on box 2: `ego probe: base is unchanged on 1 of 29 selected rows`.
  The chain stopped; nothing trained.
* **Diagnosis.** `diag_egoprobe.py` runs the tip's trainer UNCHANGED on the gate's own argv (`diag_gate_line.txt`, `--out` moved), on
  box 2, as a SAFE MODE job (`diag_egoprobe_s0.cmd`). It wraps `ego_perturb_probe` so that a refusal dumps every selected row.
  * `diag_egoprobe_s0.json` gives the refusal, reproduced. Row 25 has v0 = 0.0, a0 = 0.0, |base| = 0.0 before and after. Its perturbed a0 is
    −0.528 and κ0 −0.0099. The other 28 rows' bases all changed.
  * `out/diag_egoprobe_s0.log` is box 2's job log.
* ⇒ The rebuild is correct. A car at rest with a non-positive acceleration cannot move, so its kinematic base stays 0. The probe's clause
  was over-strict.

## 2. The fix (Amendment 4, item 1)
* **Trainer.** `ego_perturb_probe` exempts exactly the selected rows at rest and counts them (`base_rows_at_rest_exempt`). At rest means
  logged v0 = 0, perturbed a0 ≤ 0, and base identically 0 before and after. Every other selected row must still change, and at least one
  such row must exist. The change is 12 lines (`out/v8f_mini_train.py.snap.vs7f06ff00.diff`, the snapshot against the tip).
* **Tests** (`test_v8f_ego_perturb.py`, +3):
  * `exempts_exactly_the_selected_rows_at_rest`: the unmodified batch exempts 0, and a row put at rest exempts 1;
  * `does_not_hide_a_moving_row_whose_base_is_not_rebuilt`;
  * `refuses_a_batch_whose_selected_rows_are_all_at_rest`.
* **Mutants** (`mutate_orecover.py`, +3): the exemption dropped; the exemption ignoring speed and base; the all-at-rest refusal turned off.

## 3. The K4 comparator (Amendment 4, item 2)
`k4_demo.out` is from the dev-box T1 chain's build (`C:/Users/Admin/orct1/`, landing with that chain). Measured with the reader's own K4:
* against a fresh-age T0 eval, it passes (4.4e-5 m);
* against box 2's cycle-age eval, it fails (6.05 m).

The reader's separate K4 comparator key lands with the T1 chain, before any ORC rollout is read.

## 4. A mutation-anchor regression from the O-RECOVER landing (034ad7f1)
`mutate_oacc.py --check` failed on the tip. `evalorc` repeats `evaloacc`'s refusal and skip lines, so anchors #193 and #194 matched twice.
* Both anchors now carry `evaloacc`'s own TG and E0 lines, and the mutant edits are unchanged.
* `out/mutate_oacc_check.out` is the failure. `out/mutate_oacc_check_respelled.out` reads 309 mutants, each anchor found once.
  `out/mutate_oacc_193_194.out` is the two re-run.

## 5. Evidence (CPU; a fresh archive of 7f06ff00 that no other job read or mutated)
* `out/pytest_ego_perturb.out`, `out/pytest_orc_chain_oacc.out`, `out/mutate_mixo_check.out` (the MIXO kill version's 162 anchors against
  the edited trainer), and the full O-RECOVER proof
  (`out/mutate_orecover.out`). The proof and the two re-run mutate_oacc anchors were still running at landing (box 2 was idle), so they are
  banked by a follow-up commit.
* **Provenance after `R77`.** The landed files were SNAPSHOTTED (`snap_SHA256.txt`) while only read-only tests ran, before any mutation
  run, and the hold is built from the snapshot. After the proof's byte-identical restore, the tree is compared with the snapshot
  (`out/snapshot_vs_restored.out`).
