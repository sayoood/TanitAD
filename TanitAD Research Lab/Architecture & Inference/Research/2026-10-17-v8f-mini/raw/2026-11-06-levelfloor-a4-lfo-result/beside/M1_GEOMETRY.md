# M1 tick geometry of the LFO075 rollouts (descriptive, computed by the Master Mind AFTER the A4 read, 2026-11-06 04:4x)

`m1_geom.py` reads the LFO075 tick dumps (F:/TanitAD-scratch/motion_c/lfo_read/out/) with lf_read.py's own logged_speed and m1_ticks.
It explains CEILING-SHORT and chooses nothing: kappa 1 was already the A4 card's committed next arm (section 5).

* share of M1 ticks where 0.75 x v_log < 5.2 m/s (a kappa 0.75 floor cannot lift the target above the M1 threshold): s0 0.848, s1 0.838
* share of M1 ticks where 1.00 x v_log < 5.2 m/s: 0.0 at both seeds
* share of M1 ticks where the 1.0 m/s^2 ramp cannot reach 5.2 m/s within the executed 2 s: s0 0.122, s1 0.133
* M1 ticks: v0 median 4.484 / 4.478 m/s, v_log median 5.968 / 6.023 m/s
* bind rate in M1 ticks 0.536 / 0.621; where 0.75 x v_log >= 5.2: 0.861 / 0.872
