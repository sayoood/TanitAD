# refcv8 W-PB ladder: the score (2026-10-14)

**Registered result: I-0 FAIL, nothing scored.** The chain finished all 11 arms (ZZALLDONEZZ 04:10:50Z) and
`ladder_score.py` stopped at its I-0 gate (`raw/I0_registered.json`, `raw/LADDER_SCORE_registered.json`, both
md5-identical to Thor's `W/`, see `raw/thor.md5`). That stands as written.

**Amendment A1 (POST-HOC, needs the PI to accept it): I-0 PASS, and the ladder scores
L1 NOT QUOTABLE, L2 PREMISE-UNMET, L3 FAILS.** Shadow run: `i0_a1/W_A1` on Thor (symlinks to the same `W/arms`,
`W/eval`, `W/s0`; the registered `W/` files are untouched), same `ladder_score.py` (md5 7c85d214), same
`--v9-train` (md5 f63ece41). Files: `raw/I0_A1.json`, `raw/LADDER_SCORE_A1.json`, `code/ladder_i0_a1.py`.

## Why I-0 failed: three readers, one real (harmless) non-identity

| check | registered reader | what the trainer writes | A1 reader | A1 |
|---|---|---|---|---|
| I0-d agent term in the conflict detector | a `cd_*` KEY containing `agent` | the detector SUMS its aux terms into one gradient; train.log states `aux terms = agent, box3d, map_hires, tac_v6` on every arm | every `conflict detector ON` line lists `agent`, and `cd_*` rows exist | PASS 11/11 |
| I0-e 100 % coverage | `n_known == n_windows`, `mode == 'n2'` (pre-2026-10-11 schema); refcv8 half read `cfg.refcv8.v9_join`, never written, so it was INERT | `n_windows_fed == n_windows`, `window_n_unknown 0`, `source 'label'` (PI Q-B); v9 census at `cfg.refcv8.v9.<split>.join` | those fields, both splits; v9 join `n_joined == n_windows` on refcv8 arms | PASS 11/11 (train 746,946/746,946, eval 23,772/23,772) |
| MM-2 V0 vs V-R8 fed-speed CRC | equal on every logged step | unequal on 10 of 60: steps 300-500 and 2300-2500 | the speed mask's generator restarts from its seed at every RESUME and every 500-step boundary; mismatch PREDICTED iff the two arms' last reset differs; PASS iff observed == predicted EXACTLY | PASS, and OUT OF SAMPLE on V0r / V-R8r (same 10 steps predicted and observed) |

MM-2's mismatch is real: V-R8 resumed at steps 250 and 2250, V0 did not, so for 2 x 250 of 3,000 steps the two arms drew
different (identically distributed) speed-unknown masks. That is training noise of the kind the replicate arms already
measure, not a systematic difference. `ladder_i0_a1.py --self-test`: 8 mutation fixtures RED, 3 green (dev box and Thor).

## The score under A1 (paired episode-cluster bootstrap, eval139, 1,112 windows, T0 open-loop, both sampler seeds)

* **L1 (H-R8-RECIPE-1): NOT QUOTABLE.** B-REG: V-R8d's turn-direction gain over V0 <= F holds (0.037 / 0.047), but
  **V-R8 vs V-R8d is not separated (R -0.17 / -0.11)**: the trained V-R8 picks turn direction no better than the arm fed a
  DERANGED nav command. B-RECIPE also fails (dir gain R 0.06 / 0.16; head15 R -2.3 / -2.5; a family worse beyond F;
  LEGAL row not within +0.05 m ADE of V0). Consistent with the L1 diagnosis (`emit_off_diag_2026-10-12/`): the
  listwise-trained selector. **V-R8-noLW is the registered next arm and started on Thor at 04:12Z.**
* **L2 (H-R8-BUDGET-1): PREMISE-UNMET.** `gs_trunk_proj_tac_v6` median 0.0042 < 0.01 (n = 2 readings). Its bars
  would fail anyway: turn direction separated WORSE (-0.084 / -0.103), box3d AP drop 0.0092 > F 0.0038. B-REG holds.
* **L3 (H-R8-MAPW-1): FAILS.** Premise met (0.833, n = 2). Headline = the DUMP 0.25 m map IoU: drivable +0.0386,
  **R 2.12 (clears)**; lane +0.0463 but **R 0.61** (F 0.0756 once V-MAP4r enters it); box3d / agent AP drop beyond F
  (0.0043 > 0.0038; 0.0008 > 0.00055). Route no worse. B-REG holds (V-MAP4-roll collapses the map).
  ⚠ My 2026-10-13 descriptive read used the IN-RUN 10 cm IoU (drivable missed, lane cleared); the scorer's headline is
  the dump metric. Same verdict, different reason.
* L4 / L4b / L5 INCOMPLETE by the PI's 11-run cap (2026-10-11).

## What it means for the launch argv (if the PI accepts A1)

`--w-tac-v6` stays at its registered 1.0 (L2 PREMISE-UNMET); `--w-map-hires` stays 1.0 (L3 FAILS); the R8-RECIPE flags
wait on V-R8-noLW (pre-registered P1 / P2, `PREREG_VR8_NOLW.md` + A1), whose readout runs automatically on the dev box.
