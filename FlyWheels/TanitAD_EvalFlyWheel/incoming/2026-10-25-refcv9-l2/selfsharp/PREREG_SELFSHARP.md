# PREREG_SELFSHARP — make refcv8's lon head CONFIDENT in its own choice (Master Mind, 2026-10-26 ~00:10 CET)

Sealed before the roll. 0 training, one GPU roll (seed 0, 4,754 windows, os-only, per-frame nav, EMA-48,852 md5 b4364029…).

**Why (MEASURED, descriptive, `../oracle_attr/ORACLE_ATTR.json`, identity on unforced windows exactly 0.0):** the oracle arm's
missed-decel gain vs BASE-os (0.4695 → 0.3412, S6 compliant, stage on, seed 0) splits by what the force changed: group flips −45
misses (+13 false decels), within-group −4, and **windows where the oracle forced the SAME class the head already chose (pure
sharpening) −33 misses and −13 false decels**. The probe-gate (group flips at probe quality) FAILED (P1). So the hypothesis: refcv8's
lon head is right but not CONFIDENT, and the soft posterior costs caught decelerations.

* **Arm SELFSHARP:** one-hot ±20 on the head's OWN argmax over the six LON6 v7 columns (0,1,3,4,5,7), every window, via the existing
  seam (`l2_roll.py --arm oracle --labels-npz labels_selfsharp_s0.npz`). Labels = argmax of `raw_lon_logits` in the banked oracle
  trace (the head's logits are deterministic: z_tac is bit-identical across inference seeds, L1 K6). Deployable: uses no label.
* **Reference arms (banked, not re-rolled):** BASE-os = L1's os-only bank `traj_own` (identity to the oracle trace's unforced
  windows exactly 0.0); ORACLE = `trace_oracle_s0.npz`.
* **Population / estimator:** S6 compliant (GT-decel n 639, GT-cruise n 1,856, 136 episodes), the shipped stage applied as SHIP-R8
  does; paired episode-cluster bootstrap over episodes, B 2000, seed 0 -- "another draw of EPISODES" only; one training seed;
  one inference seed (0).
* **Controls:** K1 the SELFSHARP labels equal argmax(raw) recomputed from the trace (exact); K2 windows where the roll's applied
  argmax already equalled the raw argmax with margin ≥ 20 are unchanged vs BASE-os (diagnostic, stated); K3 BASE-os MD/FD reproduce
  0.4695 / 0.1395 exactly.
* **Bars:** P1 SELFSHARP missed-decel separated BELOW BASE-os. P2 SELFSHARP false-decel NOT separated above BASE-os.
  G-strict: no S6 four-family row (`l2_common.arm_metrics`, ADE / LON / LAT / TAC) separated WORSE than BASE-os.
  G-channel (reported beside): rows worse under SELFSHARP that are ALSO worse under ORACLE (the forcing channel's own cost).
  S2 rows are NOT computed in this run (stated PENDING).
* **Outcomes:** PASS = P1 + P2 + G-strict. PASS-WITH-CHANNEL-COST = P1 + P2 and every worse row is also worse under ORACLE.
  FAIL otherwise (name the clause). Ceiling share (MD_base − MD_ss)/(MD_base − MD_oracle) beside. Prediction stated before data:
  MD falls by up to ~0.05 (the sharpen-type windows) minus what sharpening a WRONG head choice costs.
* **Next lever, committed:** PASS → refcv9 trains the lon head toward confident posteriors (or ships a temperature) and SHIP-R8 re-reads
  the arm; FAIL on P1 → the sharpen gain needs the head to be right, so the lever is head accuracy (train toward the z_tac probe).
