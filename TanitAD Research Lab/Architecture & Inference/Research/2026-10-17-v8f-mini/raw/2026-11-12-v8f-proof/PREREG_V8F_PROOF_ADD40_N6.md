# PREREG_V8F_PROOF_ADD40_N6 : the read of source `cand-imagkin` repeated on the RL bases of the new seeds, and the response control on the dose arms

Note to card `PREREG_V8F_PROOF_ADD40.md` after notes N2 to N5 and row H-V8F-PROOF-81. Frozen on 2026-11-18 while RL_s2 and RL_s3 are still training: no checkpoint of a fine-tune at base 2 or 3 exists, no tool run on a dose arm (weights 0.001, 0.01) exists, and the dose arm RLA3_s0 is still in training. Tier T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react) for the tool, T0 for the reader it is matched against; never driving performance. STRATEGIC: NOT COMPUTED (the arms carry no strategic layer and the readers have no strategic instrument; n = 0), as row H-V8F-PROOF-83 states.

## 1. Why

Row 81 names `IMAG-LEVEL-CUTS-UNSAFE` with `IMAG-LEVEL-EQUALS-EXTRAPOLATION` on the control arm RLFT at bases 0 and 1: one fine-tune run per base. A separated interval from one run per base answers "would another draw of episodes say this" and not "would another training run say this". Two more bases (the RL bases of seeds 2 and 3 of card ADD37 note N4, chains `imag_b2`, `imag_b3`, then `actcons_e2`, `actcons_e3`) are being trained for another purpose; the same tool read on them is the replicate.

## 2. Runs (tool folder, family, set, bars, knots and logs of notes N3 to N5, unchanged)

* RLFT_s2, RLFT_s3 (the control arm, the arm that names) and RLAm_s2, RLAm_s3 (beside): `IPo IPi IPk IPkid1 IPkid3 IPks IPc IPl` with suffix x, then control p3 in the subset form (first 600 windows; `IPk`, `IPkn` with suffix m), compared with the compare tool of the sequences (`pod_cmp_subset.sh`).
* Dose arms at bases 0 and 1 (RLA3_s0, RLA2_s0, RLA3_s1, RLA2_s1; card ADD37 note N5): `IPo IPi` with suffix x (the run with the p4/p5 log), for control p5b only.
* Order on the pod: the pilot of card ADD41 takes a free GPU slot first; a tool run starts only with fewer than three GPU jobs and under the memory guard. Every run waits until its arm is marked done in its chain (no checkpoint in training is read).

## 3. Reads and names

* Controls first, per base, as in row 81: p1 first part, p1b, p3 (subset form), k1, k2, k3, p7, p5r (against the reader's dump of the same checkpoint), p8, p2, p6. A name is read only if every control passed at both new bases.
* On RLFT at bases 2 and 3, by the rules of card section 4 with `cand-imagkin` in the place of `cand-imag`: `IMAG-LEVEL-CUTS-UNSAFE-CONFIRMED` if the unsafe-headway rate of `cand-imagkin` minus off is below 0 at both new bases with the interval excluding 0; else `IMAG-LEVEL-CUT-NOT-CONFIRMED`. The extrapolation name of the new bases is reported as it reads (`-BEATS-`, `-EQUALS-`, `-WORSE-THAN-EXTRAPOLATION` at bases 2 and 3), with the oracle and price reads beside it.
* Beside, naming nothing: the four-base read (bases 0 to 3 in one two-level bootstrap) of the same differences; RLAm at bases 2 and 3 with the same reads.
* Control p5b (source `cand-imag`, decel family, bars unchanged: at least 0.5 for an arm that the reader names `IMAG-RESPONDS`, below 0.25 for one it names `IMAG-DEAF`): read on RLFT and RLAm at bases 2 and 3, and on the dose arms at bases 0 and 1. For a dose weight w: `DOSE-RESPONDS-ON-OWN-PLANS-<w>` if p5b is at least 0.5 at both bases of that weight and the reader of card ADD36 names the arm `IMAG-RESPONDS` with the level kept (the definition of card ADD37 note N5 section 3). It is a control read of this card; which arm becomes arm X stays the rule of card ADD37 notes N4 and N5.

## 4. What follows (committed before the data)

* `IMAG-LEVEL-CUTS-UNSAFE-CONFIRMED`: the cut of row 81 holds on four bases; the reading of row 81 stands on four training runs. With `-EQUALS-EXTRAPOLATION` again: card section 6, second bullet, unchanged.
* `IMAG-LEVEL-CUT-NOT-CONFIRMED`: reported as written; the cut of row 81 is then a result of two bases, and the next step is the four-base read with the per-base values, not a new lever.
* p5b met on a dose weight or on RLAm at the new bases: source `cand-imag` becomes evaluable on that arm under the card as written (p5 of the card), own row. p5b missed again everywhere: the mean-form candidate actions of note N5 section 4, own note before any run.

## 5. Limits

One fine-tune run per base and arm; a lead that does not react; one rule and one candidate family; the full form of p3 (2,399 windows) is not repeated on the new bases.
