# MM verdict: K-INPUT read (KI_s0, KI_s1), 2026-11-23

**Outcome: KINPUT-REPAIRS by the card's letter, with MECH4-NO.** MECH4 does not support the reading the card attaches to the name.

* **Card:** `PREREG_V8F_KINPUT.md`, sha256 `05e786a8…`, frozen 2026-11-23T08:57:56Z, before any KI step or number.
* **PI go:** "Yes do it", in the MM chat on 2026-11-23, about 13:05Z.
* **What the name says:** G-T0 holds at both seeds, STRATEGIC is BETTER at both seeds, and no family is WORSE.
* **What it does not say:** that the logged `base` lowers the T0 cost against RH. MECH4 (KI − RH at T0) is not separated at either seed.
* **What the read adds at T1:** keeping the logged `base` costs RH's gains. KI − RH is WORSE on LATERAL and TACTICAL under both forms. Against B0, KI keeps only STRATEGIC BETTER, where RK, RL, RA and RH also had LATERAL and TACTICAL BETTER.

Evidence class MEASURED (ours). The arms trained on the LoRA pod; the read ran on the dev box with zero training.

## Arms and gates
| item | value |
|---|---|
| arms | KI_sK = RH_sK's argv with exactly the two changes of the card's section 2: `--out runs/KI_sK/train` and `--dagger-base logged`. Code 4097a89e, RH banks, readout causal_g1x, p 0.1, 3,000 steps of stage B. 64 tokens; argv sha256 s0 `baf33625…`, s1 `2ad1ccae…` |
| GPU lock | 13:09:14Z to 13:40:21Z = **0.52 pod-h**. The card ESTIMATED 0.5-0.6; its return line is 0.9 |
| pre-lock | tree_ki 3,425 blobs verified; G-SUITE at 4097a89e: 0 regressions, 0 new failures, 0 missing; dry mint and dry walk PASS (`pod_ki/HANDOFF_POD_KI.md`) |
| section 4 records | selected rows of the probe batch: s0 **46 / 64**, s1 **51 / 64**. Share of selected rows whose own `base` differs from the logged one: **1.0** at both seeds |
| gates | gate record MINTED and verify OK (both); zerograd PASS 252/252; post 252/252 tensors with Adam state and moved; summary at step 3000 |
| lever acts (section 3) | `levers.dagger.dagger_base` = **logged**; declared_vs_built PASS; `dagger_probe` pass with `rows_changed.base` **0** on the selected rows; selected rows > 0 at both seeds; train log `dagger_rows_sum` **191,664** (s0) / **191,687** (s1) over 60 records |
| checkpoints | sha256 s0 `d9e93f63…`, s1 `c12c989f…`, the same in the pod's summary and in the read's provenance |
| records | `pod_ki_run/` (29 files, every one equal to the pod's sha256: `SHA256_POD.txt`) |

## Tiers and estimator
* P1 and ATTR are read at **T1-kin/hb0.2**, the executor of record: a kinematic rollout on the model's own actions, with a heading bound of 0.2 1/m.
* G-T0 and MECH4 are read at **T0**. T0 is a world-model diagnostic, never driving performance; the card makes it binding.
* Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 (`taniteval/ci.py`), over 141 episodes and 2,399 T0 windows. It answers "another draw of EPISODES"; the two seeds answer "another TRAINING run". Inference is deterministic.

## G-T0 (decides): KI − B0 at T0 ADE6
The bar is an upper bound ≤ +0.0705 m per seed. F_T0 is INHERITED from one seed pair, so it is a weak floor.

| seed | point | interval | verdict |
|---|---|---|---|
| s0 | +0.0312 m | [+0.0049, +0.0583] | **HOLDS** (0.0122 m under the bar) |
| s1 | +0.0251 m | [−0.0111, +0.0604] | **HOLDS** (0.0101 m under the bar) |

* Beside, at the same windows:
  * RH − B0: s0 +0.0503 [+0.0060, +0.0929] fails; s1 +0.0217 [−0.0122, +0.0596] holds.
  * RK − B0: s0 +0.0364 [+0.0004, +0.0746] fails; s1 +0.0070 [−0.0304, +0.0456] holds.
* KI is the first arm on this line to hold G-T0 at both seeds.
* Mean T0 ADE6 (m):

| arm | s0 | s1 |
|---|---|---|
| KI | 2.1214 | 2.0740 |
| RH | 2.1405 | 2.0707 |
| RK | 2.1265 | 2.0560 |
| B0 | 2.0901 | 2.0490 |

## MECH4 (decides the next lever only): KI − RH at T0 ADE6
| seed | point | interval |
|---|---|---|
| s0 | −0.0191 m | [−0.0628, +0.0258] |
| s1 | +0.0034 m | [−0.0420, +0.0430] |

* Neither interval sits below 0, so this is **MECH4-NO**. Instrument: `mech_ki.py`, output `MECH4_KINPUT.json`.
* Beside: KI − RK s0 −0.0052 [−0.0450, +0.0337], s1 +0.0181 [−0.0277, +0.0604].
* ⇒ KI's T0 ADE6 is not separated from RH's at either seed; at s1 KI's point is the higher of the two. The difference between RH failing G-T0 at s0 and KI holding it is a difference in where each interval sits against the bar, not a separated effect of the lever.

## P1: KI − B0 under C* (R0C10XHb20), four families
Each cell gives the two-seed D, its interval, the per-seed values, and the floor F (two-seed F_c).

**LONGITUDINAL: NOT-SEPARATED**
* speed −0.085 m/s [−0.178, −0.008] (−0.043 / −0.126), F 0.126: NOT-SEPARATED. It straddles the floor; its sign-resolved verdict is BETTER.
* along −0.055 m [−0.157, +0.026] (−0.013 / −0.096), F 0.122: NOT-SEPARATED.
* headway unsafe rate U +0.0062 [−0.0034, +0.0175] (+0.0053 / +0.0071), F 0.0231: NOT-SEPARATED.

**LATERAL: MIXED**
* heading +1.140° [+0.749, +1.635] (+0.889 / +1.391), F 1.136: **WORSE**, by 0.005° beyond the floor.
* yaw rate +0.34 °/s [−0.94, +1.39] (+0.83 / −0.15), F 3.06: NOT-SEPARATED.

**TACTICAL: NOT-SEPARATED**
* ade_tac −0.057 m [−0.216, +0.074] (+0.017 / −0.131), F 0.164.

**STRATEGIC: BETTER** (in full; the pause was lifted on 2026-11-16)
* Position at the 30 / 50 / 80 m marks: −1.76 [−2.26, −1.32] / −3.51 [−4.48, −2.65] / −5.40 [−7.09, −3.91] m, F 0.71 / 1.77 / 2.47. Every per-seed value is beyond its floor.

No family is WORSE, and the reader lists no seed-disagree cell. The one floor-straddling cell is the speed cell.

## ATTR (beside, no bar)
**KI − RH under C\*:**
* LONGITUDINAL MIXED: speed +0.180 m/s WORSE, along +0.170 m WORSE, headway U −0.012 NOT-SEPARATED (straddles, BETTER by sign).
* LATERAL WORSE: heading +1.97° [+1.46, +2.56], yaw rate +5.45 °/s [+4.26, +6.65].
* TACTICAL WORSE: ade_tac +0.306 m [+0.148, +0.454].
* STRATEGIC MIXED: 30 m −0.15 m BETTER, 50 and 80 m NOT-SEPARATED.

**KI − RH under R0Hb20:**
* LONGITUDINAL MIXED.
* LATERAL WORSE: heading +1.71°, yaw rate +5.85 °/s.
* TACTICAL WORSE: +0.142 m.
* STRATEGIC NOT-SEPARATED.

⇒ **Keeping the logged `base` costs RH's T1 gains on LATERAL and TACTICAL.** The own `base` is load-bearing for those gains, which is the card's HARMS reading, even though the HARMS row does not fire: no family reads WORSE against B0.

**The other after-checks, all failing:**
* **P2 fails:** KI is not NOT-WORSE on every family under ACC / D1 / D2. The B0 attribution control reads WORSE on LATERAL and TACTICAL.
* **P3 fails:** DEGRADES under R0Hb20 and R0C10XHb20; the B0 control reads DEGRADES as it must.
* **P4 fails:** KI under C\* against B0 under K reads LONGITUDINAL MIXED, LATERAL WORSE, TACTICAL WORSE, STRATEGIC NOT-SEPARATED.

## Name by the card's precedence
BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order.

1. **Not BLOCKED:** every gate passed, the section 4 records are in, and every lever-acts item reads its known value.
2. **Not HARMS:** G-T0 holds at both seeds, and no P1 family is WORSE. LATERAL is MIXED, not WORSE: the ROLLHIST card's family rule (section 3) calls a family WORSE only when every named cell is WORSE.
3. **Not SEED-DISAGREE:**
   * No cell has its seeds opposite beyond the floor.
   * The one straddling cell (speed) is not decisive. Read by its sign it is BETTER, which turns LONGITUDINAL MIXED; that adds no BETTER family and no WORSE family, so the name is the same either way. The reader's sign-resolved outcome is unchanged too.
   * The heading cell, 0.005° beyond its floor, is not decisive either: NOT-SEPARATED there would turn LATERAL NOT-SEPARATED, and the name would stay the same.
4. **REPAIRS:** G-T0 holds at both seeds; in P1, STRATEGIC is BETTER at both seeds (each cell's interval excludes 0, both seed means agree and |D| > F_c) and no family is WORSE.

⚠ **The reader prints `ROLLHIST-PARTIAL` and maps it to "KINPUT-PARTIAL".** That comes from the ROLLHIST card's own table, where P2 and P3 fail. The read extension says so at `rh_read.py:649`: section 5 is the MM's table, and the reader prints the closest name. The KINPUT card uses P2, P3 and P4 as ATTR with no bar, so the name here is the MM's, by section 5.

**The reading.** The card's REPAIRS reading, "the own `base` input carried the T0 cost", is **not supported**:
* MECH4 is NO at both seeds, with points −0.019 and +0.003 m.
* The four-seed stage-A spread at T0 (0.117 m, beside) is larger than every G-T0 point on this line.

What the read does show, at the primary tier: the own `base` carries RH's T1 gains on LATERAL and TACTICAL. Four levers have now been read against their parent at T0: RL, RA, RK and KI. None separates from it, and only KI loses T1 gains.

## Next lever
**What the card commits for REPAIRS:** a deployable form under a new card, a prior fed a `base` that is consistent at train and test.
* RH is already consistent: own `p10f` with own `base` in a swapped sample, own at T1, logged at T0.
* So a deployable form different from RH has to change how `base` is built for the prior, for example from the pose history `p10f` itself. That is a change to the model's input, an architecture decision for the PI.

**MM recommendation instead:** the third seed pair the card names (B0_s2, RH_s2, KI_s2 on UE_s2.A, banks generated on UE_s2.A), under a new card that adopts the four-seed F_T0 before its data.
* **Why:** every G-T0 failure on this line is at s0, on one B0_s0. The T0 differences between arms are not separated, and the guard's floor rests on one seed pair. The third pair asks directly whether RH, the arm with the T1 gains, fails the T0 guard only at seed 0.
* **Cost, ESTIMATED about 2.7 pod-h:**
  * a bank on UE_s2.A, about 1.9 pod-h, from the MEASURED RHK bank at 1 h 52 min per seed;
  * three stage-B arms at about 0.26 pod-h each, from the MEASURED RK 0.55 and KI 0.52 for two arms.
* Both choices are PI decisions; neither runs on this card's go.

## F_T0 at four seeds (beside, printed, never used here)
* From the KINLAB read: stage-A means 1.9671 / 2.0376 / 1.9202 / 1.9806 m, largest pairwise difference 0.1174 m (s1-s2).
* Under that floor, every G-T0 upper bound on this line would hold, the largest being RH − B0 s0 at 0.0929.
* This card keeps 0.0705 and its name does not change.

## Instrument notes
* `GT0_ROLLHIST_KI.json` / the result's `G_T0.holds_at` prints `[]` while `per_seed` reads HOLDS at both seeds and `holds` is true. The MM checked each upper bound against 0.0705 by hand (0.0583, 0.0604).
* The per-seed key `mean_ade6_RH` carries KI's mean, and the beside keys keep `BESIDE_RH_minus_B0_*` names for KI − B0. These are legacy labels for the treated arm and change no value.

## Limits
* Two training seeds.
* F_T0 used by G-T0 rests on one seed pair.
* T1-kin only, with no re-perception (T2 is not provisioned).
* T0 is a world-model diagnostic.
* ATTR has no bar.
* KI is a diagnostic arm, not a recipe: the prior reads an (own `p10f`, logged `base`) pair that never occurs at inference.

## Files
* `readki/MM_VERDICT_KINPUT.md`
* `read_ki/` = `out/read_ki/`: RESULT_ROLLHIST_KI.json, GT0_ROLLHIST_KI.json, MECH4_KINPUT.json, READ_STRAT_ROLLHIST.json, hw/, q3/
* `read_ki/logs/`: rh_read_ki_chain.log, rh_read_ki_retry.log
* `pod_ki_run/`: 29 pod records with SHA256_POD.txt; `pod_ki_run/rh_ki_chain.log`
