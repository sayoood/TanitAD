# N7 wider candidate generator for the DAC = 0 residual: navtrain development, 2026-11-13

Engineering agent for the REFe Master Mind. Navtrain only; no navtest token was built, scored or read. CPU only, 2 processes, BLAS threads 1.
**Evidence class for every number: MEASURED** (builder scorer = W-TRANSP nuPlan-DB rebuild with the LQR tracker, tier T1 builder-side). Artifacts are in `raw/` (candidate plans + metrics: `raw/candidates_navtrain.tar.xz`).

## Headline
- **On the 17 navtrain residual tokens the TF-N6 generator (REPAIR-D + L1, 12 candidates, tolerance certificate) certifies 2; the proposed cascade certifies 13 with lateral and mild-slow-down levers, and 16 with a deep slow-down tier. One token (`229a59137a705430`) is rescued by nothing.**
- **0 regressions on the 73 repaired tokens, by construction and measured:** all 73 are certified by the TF-N6 tier alone (T0), the cascade never reaches a later tier for them, and the REPAIR-D plan equals a recorded iterate bit for bit (73 of 73).
- **The 16-of-17 figure rests on a tier that edits the speed of the label by 40 to 50 % (s = 0.5 to 0.6).** It is disclosed as its own tier; the 13-of-17 figure does not need it. The decision whether T4 is admissible as "teacher output" is the MM's / PI's (see "Open decisions").

## 1. Baseline and control (priority 1)
| | n | DAC = 1 iterate exists | tolerance-certified | full-certified (TTC, C not worse too) |
|---|---|---|---|---|
| REPAIR-D at A1 settings (cap 5, margin 0.05) | 17 | 1 | **0** | **0** |
| L1 at TF-N5 settings (cap 5, margin 0.05) | 17 | 4 | **2** | 2 |

- **Control reproduced:** REPAIR-D at A1 settings has **0** full-certified iterates on the 17, and the derive rows record N4D `applied = False` on all 17. The base label metrics recomputed here equal the derive rows' `base` on 17 of 17 (and 73 of 73).
- The 2 tolerance-certified tokens are rescued by L1 (`69b54b1998f05c13`, `e8a4675c22a354b4`). On the 17, REPAIR-D has one iterate with DAC = 1 that fails the tolerance certificate (which clause: not checked).
- The tolerance certificate is the TF-N6 N6-T one, copied from `n6_nt_build.CARD6["N6T"]` (DAC = 1; NC and DDC not worse than the token's base label; builder PDMS strictly better; TTC and C free). `test_certificate_equals_n6_card` pins the copy and its clause grid.

## 2. Levers (priority 2)
All levers are pure functions of (scene, target, label, base metrics, parameters); parameters are in one frozen dict, `n7_lib.LEVERS`. Candidates are every iterate / grid point scored (the TF-N6 recording-proxy idea, `Rec`).

| lever | what | grid |
|---|---|---|
| L2-D | REPAIR-D target iteration | cap {5, 10, 20} x margin {0.05, 0.10, 0.20} m |
| L2-L1 | L1 two-sided repair | the same 9 |
| STR_label (L4) | time-stretch of the P5-R label: pose at t := plan pose at s t (cubic through the origin), scored directly | s in {0.95, 0.90, 0.85, 0.80} |
| STR_target (L4) | the same stretch on the TARGET, label rebuilt by the pipeline (Tc20s + P5-R), scored | the same 4 |
| OFF_label | lateral offset of the label, ramped over t_ramp, scored directly | dy in +-{0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0} m x ramp {0.2, 1.0, 2.0} s |
| STR_OFF | mild stretch then offset | s in {0.9, 0.8} x the OFF grid |
| STR_deep (T4) | stretch of the label, larger edit | s in {0.7, 0.6, 0.5, 0.4, 0.3, 0.2} |

Cap 5 and 10 are read as a PREFIX of the cap 20 run (k <= cap); `test_real_cap_prefix_property` pins this on real tokens for D and L1.

### Tokens gaining a tolerance-certified candidate on the 17 (any candidate of the lever; `raw/lever_table17.json`)
| lever | cap 5 | cap 10 | cap 20 |
|---|---|---|---|
| D margin 0.05 | 0 | 2 | 2 |
| D margin 0.10 | 1 | 3 | 4 |
| **D margin 0.20** | 5 | 7 | **7** |
| L1 margin 0.05 | 2 | 3 | 3 |
| L1 margin 0.10 | 3 | 3 | 4 |
| L1 margin 0.20 | 3 | 3 | 3 |

| lever | tolerance-certified (of 17) | full-certified |
|---|---|---|
| STR_label | 4 | 4 |
| STR_target | 3 | 3 |
| OFF_label | 11 | 11 |
| STR_OFF | 12 | 12 |
| STR_deep | 16 | 16 |

- The margin matters more than the cap: margin 0.20 reaches 5 of 17 at cap 5 and 7 at caps 10 and 20; margin 0.05 stays at 0 to 2. L1 does not improve with a larger margin (reason not investigated; UNVERIFIED whether its corridor turns INFEASIBLE).
- STR_label and the offset grid rescue tokens D and L1 cannot: the union D(m 0.20, cap 20) + L1(m 0.05, cap 20) is 7; adding STR_label gives 9; adding OFF_label and STR_OFF gives 12 or 13 (13 with all T0 to T3 candidates).

## 3. The picked generator: a CASCADE, smallest change first (priority 3)
Tiers (`n7_select.py`, frozen): **T0** TF-N6 (REPAIR-D + L1 at A1 / N5 settings, cap 5, margin 0.05) -> **T1** L2 (D and L1, caps to 20, margins 0.05 / 0.10 / 0.20) -> **T2** stretch (STR_label, STR_target) -> **T3** offsets (OFF_label, STR_OFF) -> **T4** deep stretch (STR_deep, floor s >= 0.5). In the first tier with a tolerance-certified candidate, the TF-N6 rule picks the maximum builder PDMS (ties to the earliest).

| tier that certifies | tokens (of 17) |
|---|---|
| T0 (= TF-N6 as is) | 2 |
| T1 L2 | 7 |
| T2 stretch | 2 |
| T3 offsets | 2 |
| T4 deep stretch (s >= 0.5) | 3 |
| nothing | **1** |

With T0 to T3 only: **13 of 17**. With T4: **16 of 17**. Floor on T4: s >= 0.5 loses nothing on the 17 (a floor of 0.6 loses 1, 0.8 loses 3).

### Per token (`raw/sel17_floor05.json`; builder sub-scores of the chosen plan; NC = 1.0 and DDC = 1.0 on all 16 chosen)
| token | tier, lever, parameter | TTC | builder PDMS |
|---|---|---|---|
| 0e8ccba41b17541e | T1 D cap 20 m 0.20, iterate 8 | 0.0 | 0.555 |
| 0edf786bcfbd5820 | T1 L1 m 0.05, iterate 8 | 1.0 | 0.992 |
| 1dc616ca2c5353f8 | T1 D m 0.20, iterate 3 | 1.0 | 1.000 |
| 437e7aa96ae951c9 | T1 D m 0.05, iterate 7 | 1.0 | 1.000 |
| 7d4286e68aee5c8b | T1 D m 0.20, iterate 2 | 1.0 | 1.000 |
| c582b4959ae65d55 | T1 L1 m 0.10, iterate 3 | 1.0 | 1.000 |
| faf94859c4095b79 | T1 D m 0.10, iterate 15 | 1.0 | 0.967 |
| 36c4507970805f49 | T2 STR_label s 0.85 | 1.0 | 1.000 |
| fc61cadc28715436 | T2 STR_label s 0.90 | 1.0 | 1.000 |
| 24bcf45bcc6f5dfb | T3 OFF_label dy 0.5 m, ramp 1 s | 1.0 | 0.976 |
| 85e35e87946d502d | T3 OFF_label dy -1.0 m, ramp 1 s | 1.0 | 1.000 |
| 8f0f005a5f6b58e6 | T4 STR_deep s 0.6 | 1.0 | 1.000 |
| ed589b058ca85557 | T4 STR_deep s 0.6 | 0.0 | 0.388 |
| ff81f481d2ad5270 | T4 STR_deep s 0.5 | 1.0 | 0.764 |
| 69b54b1998f05c13 | T0 L1 iterate 1 | 1.0 | 1.000 |
| e8a4675c22a354b4 | T0 L1 iterate 5 | 1.0 | 0.987 |
| **229a59137a705430** | **nothing** | | |

### The class nothing rescues, and the other hard classes
- **`229a59137a705430` (1 token): no DAC = 1 candidate among 290.** The label starts at 0.14 m/s and is outside the drivable union from step 9 to 39 (20 steps, max 0.116 m). The lateral grid to 1 m and the stretch to s = 0.2 do not reach DAC = 1 (s = 0.2 leaves 0.005 m outside). HYPOTHESIS: the start pose is already within the footprint's margin of the boundary, so any plan that moves reads DAC = 0 and only standing still would not; not tested (s = 0, a stationary label, is a degenerate label and was not added).
- **`8f0f005a5f6b58e6`: DAC = 1 candidates exist in T1 but all have NC = 0** (an agent is in the way); only slowing down (s <= 0.6) clears both. **`ed589b058ca85557`, `ff81f481d2ad5270`: base DDC is 0.5, and every lateral DAC = 1 candidate has DDC 0.0, which the certificate refuses;** only the slow-down candidates read DDC 1.0. They are blocked by the NC / DDC clauses, not by geometry. Lifting the DDC clause would rescue the last two laterally; that is a certificate change and is NOT proposed (PI-approved N6 certificate unchanged).
- **Offset tier sizes:** the T3 rescues use 0.5 m and 1.0 m lateral offsets, i.e. large edits of a label. Reported, not hidden.

## 4. No harm on the 73 (priority 2, second half)
`raw/sel73_floor05.json`, `raw/lever_table73.json` (levers run: D and L1 cap 20 margin 0.05, STR_label, STR_target, OFF_label, STR_OFF, STR_deep; the margin 0.10 / 0.20 L2 levers were NOT run on the 73: unneeded for the cascade's no-harm property, about 70 s per token, stopped after 5 tokens).
- **73 of 73 are tolerance-certified AND full-certified by T0 (the TF-N6 candidate set);** the cascade selects T0 for all 73, so the chosen plan is the TF-N6 plan. **0 changed, 0 harmed.**
- Informational (is a lever also able to certify the 73): D 73, L1 73, STR_label 35, STR_target 36, OFF_label 63, STR_OFF 66, STR_deep 69 of 73. Using any of these in place of T0 would change a plan that is already certified, which the cascade does not do.
- The no-harm property is therefore structural (superset + cascade), not an empirical hope: a later tier cannot alter a token whose earlier tier certifies.

## 5. Tests (priority 4; `test_run.log`, 20 passed, `CUDA_VISIBLE_DEVICES=-1`)
- **Identity:** zero lever (s = 1, dy = 0) returns the plan bit-identical, also through the real scorer (scores equal the base metrics).
- **Determinism:** pure transforms; and two real runs of L1, STR_target, OFF_label are bit-identical (plans and metrics).
- **Certificate:** accepts a good candidate with TTC and C free; refuses NC 0.0 and 0.5 (also against a base NC of 0.5: equal passes, 0.0 refused), DAC 0, DDC drop, PDMS not better; equals the N6 card over a clause grid.
- **Deliberate-regression arms (MEASURED, and not as clean as hoped):**
  - STR_label inverted (faster, s -> 2 - s): rescues **1** of the 5 test tokens against 4 for the true lever; D margin 0.20 cap 5 inverted: 1 against 2; L1 inverted: **1 against 1** (reported, not asserted). The faster plan certifies on `85e35e87946d502d` (it outruns the problem: a speed edit is NOT monotone in its effect on DAC). The test asserts "fewer than the true lever", not "zero".
  - On the two tokens the true slow-down rescues (`36c4`, `fc61`), faster plans s = 1.05, 1.1, 1.2 are refused: **0 rescues**.
  - **OFF_label's grid is two-sided, so its sign flip scores the same plans (vacuous, asserted as a fact).** Its regression arm is the zero-offset grid, which rescues 0.
- Reachability: every lever name in `LEVERS` has an executing branch in `run_lever` (an unknown kind raises); the cap / margin values read back equal the dict (test_levers_frozen_dict_complete); caps change behaviour (cap-5 candidates are a strict prefix of cap-20's).

## 6. Caveats (Rule: fail loud)
- **n = 17 tokens on 7 logs.** The tier order and the s floor were set on these tokens; there is no held-out navtrain residual (the other 877 logs are being rolled and would give one). A log-cluster interval is not meaningful at 7 logs; none is quoted.
- The builder scorer is the nuPlan-DB rebuild; DAC agrees with the harness cache on navtest (WTRANSP_DIAG, INHERITED). The N6 run read DAC = 1 on all certified navtest tokens, so a builder DAC = 1 is trusted at the DAC level; not re-verified here.
- The cascade's value on navtest is ESTIMATED: navtest residual anatomy (infeasible corridor, ping-pong) is not the same distribution as the navtrain 17.
- T4 labels are slow plans: builder PDMS of the three T4 picks is 1.000, 0.388 and 0.764 (EP 1.0, 0.53, 0.43 for the last two). Students trained on them learn slower driving at these scenes; the tracker-aware check the N6 card asks for applies.
- Parameters were set once on the 17 and not tuned further. The offsets of 0.5 and 1.0 m, and the slow-down floor 0.5, are the largest edits and the ones to look at first.

## Open decisions for the MM / PI
1. **Is T4 (deep slow-down, s 0.5 to 0.6) admissible as "teacher output"?** Without it: 13 of 17 on navtrain; with it 16 of 17. The draft card runs both (N7-T with T4 as the primary, N7-T3 without as an attribution arm).
2. DDC clause: unchanged; two tokens are blocked by it laterally.
3. 229a-class (stationary, boundary-hugging): no lever; a stand-still label is degenerate.

## Reproduce
`C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -B n7_run.py --set r17 --levers <names> --out f.jsonl` (needs `derive_rows.tar.xz` extracted and `N7_ROWS` set for `--set rep73`), then `python n7_select.py out.json f.jsonl... --floor 0.5`; tests: `python -B -m pytest -q test_n7_gen.py`. Pinned tree `F:/tanitad_scratch/n4nt/run_a1` and the n5_l1 hold are read only and imported, not copied. Code sha256 prefixes: n7_lib `dfd5c7c4`, n7_run `d1c79b4b`, n7_select `bf2cbfa4`, n7_analyze `326cf4fc`, n7_env `4f32d58b`, test `db6d8e3e`.
