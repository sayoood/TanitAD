# P5-C readiness: what stands between us and launching it (2026-11-11)

P5-C = the REFe student trained on the P5-S tracker-aware labels (Tc20s, cubic) instead of the raw teacher `traj`, on the pod after P7 (~15 Nov).
Evidence classes: MEASURED (own, path given), PUBLISHED, INHERITED, ESTIMATED, UNVERIFIED. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. File:line cites are to the mirror tip `66681a60` (E: lags it by 756 commits; the P5 eval scripts and navsim_dac / p4_tracker_comp / navtrain_scenarios are blob-identical on E:, checked by `git hash-object`).

## Verdict
Nothing is launchable today. **Seven blockers**, none of them compute on the pod:
1. **No P5-C pre-registration exists.** The card holds one line (`PREREG_P5...md:183`, "fine-tune the WTA on Tc20 labels, P5-D as control") plus "the label P5-C pre-registers" (:279, :297). `RESULT_P5.md:181` lists the prereg as the next step. No arms, bars or controls are committed. This is a new file, not an amendment.
2. **0 of 150,603 training tuples have a Tc20s label** (Q1).
3. **`train.py` has no label-table input** (Q3). Nothing reaches the loss.
4. **PI decision (a) is open** (card §8: Tc20 is "a softer form of test-time compensation").
5. **Tc20s repair on rank-1 twins is unmeasured** (P5-S used rank-0 step-0 only).
6. **The P7 bank's rows must be on the dev box** (they sit on the pod / HF; UNVERIFIED that HF holds them).
7. **PI go and pod time.** Until P7 closes, any land that touches `train.py` voids bound records (memory: land-closure rule), so wiring is staged, not landed.

## 1. Label coverage
- **Existing labels:** `PK/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz` holds `labels20_t20_tc20_tw` [1018, 3, 20, 3], rank-0 step-0 held-out navtrain, 11 logs. MEASURED. These logs are held out of training, so they cover **0 of the 150,603** P7 tuples (92,896 rank-0 + 57,707 rank-1; `launch_gate.py:361`, `SPEC_P7.md:144`).
- **20 Hz is not a dependency.** The labeller needs the row's own `traj` (20 poses at 5 Hz, `build_targets.py` STRIDE=2 from a 10 Hz rollout) and the nuPlan ego state at its token. The legacy 20 Hz rows (18,122 dev box, 4,265 pod; `RESULT_E2E1.md` E1) were superseded by the 10 Hz D9 bank (`READINESS.md:27`, C17 :49). Probes run on this box, scanning every rank-0 and aug row of `E:/Projects/TanitAD/data/refe_navtrain10` (123,642 lines, shard duplicates included): all carry `sim_hz 10`, 0 legacy. Not counted on the P7 bank itself: UNVERIFIED. One-line probe once its jsonl is local: count rows with `sim_hz != 10`.
- **DBs:** all 408 logs in the dev-box bank resolve to a nuPlan DB under `splits/` (top-level 326 + trainval 1,381 + test 1,349 files). MEASURED. The P7 bank spans ~1,076 train logs; the full list is not local, so UNVERIFIED. The pod has `$DATA/navtrain_dbs`.
- **Table key:** (log, token, rank, sha256 of the row's own `traj` bytes). Rank-1 twins share the rank-0 token, so a token key would collide. This also makes the label provably the one computed from the row it replaces.

## 2. Label cost (CPU, dev box)
- **MEASURED** (`p5c/evidence/labonly.py`, `CUDA_VISIBLE_DEVICES=-1`, OMP 2, navsim-crun venv): Tc20s only, cubic, `ITERS=3`, 60 rows from the P5-S set.
  - **Bit-identical to the banked P5-S labels: 60/60, max abs diff 0.0.**
  - `iterate20` takes 0.115 s/row.
  - Scenario build: 2.8 s for the first log (map load), 0.03 s for 42 rows of another.
- P5-S's full pipeline (all 7 refs, DAC, lane, margin) ran 270 s / 1,018 rows = 0.265 s/row (`p5s_cubic.json "seconds"`). MEASURED.
- **ESTIMATED:**
  - Tc20s for the full bank: 150,603 × ~0.135 s = **~5.6 process-h**, ~3 h wall on the 2 processes allowed. Upper bound at the full-pipeline rate: 11.1 process-h.
  - If P5-D (drop-if-sim8) or the Tsel selection is wanted: the sim8 flags need the full pipeline, so up to 11.1 process-h more.
- **Script:** the P5-S code unchanged (`eval/p5_target_labels.py` `iterate20` and `interp20`, blob 294644df). A new thin driver is needed that reads bank rows instead of the 1,018-token set and writes the table (Q3).
- **Run command that worked:** `NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0`. `navtrain_scenarios.py:75` defaults to `D:/...`, which does not exist, and fails every row with "does not exist" (MEASURED; the p5 script lists it as an error, not a crash).
- **Regenerating 20 Hz rows:** not needed.

## 3. Trainer wiring (plan, not implemented)
The trainer edit is ~35 lines, but the table tool, gate entries and G-LIVE probes are several hundred. Over the ~50-line limit, so nothing was written to `hold/`. The existing yaw table is the template: `ego_yaw_table.py` plus the `--ego-yaw-table` hops in train.py (blob 0e115c0d).

**Flag:** `--traj-label-table <gz json>`, default None (OFF).

**Hops, argparse to loss (tip train.py):**
| hop | where | change |
|---|---|---|
| argparse | after `--ego-yaw-table`, :2239 | add the flag |
| train bank ctor call | :1412 | pass `traj_label_table=a.traj_label_table or None` |
| rebuilt-bank ctor call | :1672 | pass it too (the `--grow` path; forbidden by P6/P7, but a dropped kwarg is the failure class) |
| `TargetBank.__init__` | :651 signature; new block mirroring :826-849 | load the table, REFUSE unless every final row's key is covered (100 %) and each label is [20,3], finite, with \|Δyaw vs `traj`\| < π; store the sha256 |
| `__getitem__` | :969-971, `torch.tensor(r["traj"])` | replace with table label or `r["traj"]` |
| loop and loss | unpack :1728, `wta_loss` :1763, `traj_L1` | unchanged |
| identity and meta | :1135-1138, :1547-1552 | add `traj_label_sha256` |

- Held-out and eval banks do NOT get the table.
- `r["traj"]` is also read at :316 and :446 (scorer / on-policy banks). Those are not used under `refe_p6` / `refe_p7` (forbidden flags). State this in the G-DVB entry.

**Launch gate:**
- New profile `refe_p5c`, cloned from `refe_p7` (`launch_gate.py:379`).
- Data flag `--traj-label-table`, required in the argv; a gate_checks `DVB_KINDS["traj_label_table"]` entry like `ego_yaw_table` (:370).
- Registered pins `P5C_LABEL_TABLE_SHA256` and rows = `P7_BANK_TUPLES` (:361-363).
- Argv = the P7 argv plus the one flag, so P5-C minus P7 is a single-lever contrast.
- Pin edits move 6 profile fingerprints pinned in 3 tests (memory): run all launch_gate-reading tests before landing.

**G-LIVE** (the mechanism is a data substitution, so the probe is not a parameter gradient):
- (a) Through the real dataloader, `item[3]` equals the table label bit-for-bit on N rows and differs from raw `traj` on at least the fraction measured at table build. Control: flag-off `item[3]` equals raw bit-for-bit.
- (b) Fixed-weights forward: `wta_loss(table)` minus `wta_loss(raw)` is non-zero with the flag and **exactly 0.0** without it.
- (c) An identity table (label := `traj`) reproduces flag-off bit-for-bit.
- (d) Deliberate-regression arms that must FAIL: `label_table_not_passed` (drop the kwarg at :1412) and `label_lookup_removed` (revert :971).
- (e) Table `check-bank`: 150,603 / 150,603 keys.

## 4. Pre-registration: what exists and what to decide
**Committed by the card today** (all that exists):
- P5-C is a fine-tune of the WTA on Tc20 labels, P5-D (drop-if-sim8) its control (:183).
- P5-S bar met for the label: sim8 −7.07 pp [−13.26, −1.83]; learnability 1.69× / 1.76× against the 2× bar.
- §8's four limits: no PDMS claim, no student-lever claim, PI decision (a) not ruled, own navtest read needed.
- Teacher-side evidence for the label (`D-REFE-TF1-1`): +7.64 [+2.29, +13.60] tracked PDMS on 1,018 held-out targets.

**To decide in the new `PREREG_P5C.md`** (listed, not written):
1. **Arm form.** The card says fine-tune; the brief says retrain at the P6/P7 recipe. The cost differs about 5× (Q5).
2. **Base and control.** P7 (yaw + aux head + reserve bank) as base. P7 itself is the free control only if the argv, seed, bank and hardware match. The card's P5-D control needs its own arm and the Q2 sim8 pass; keep or drop it explicitly.
3. **Label population.** Rank-1 twins are 38 % of the bank and P5-S never measured them. Either measure the repair on a rank-1 sample first (CPU, ~0.3 s/row) or label rank-0 only. Mixed conventions in WTA are not attributable.
4. **Effect bar.** The teacher gain does not transfer one-for-one: only 29 % of the student's DAC failures are tracker-repairable (P4, INHERITED). Register an ADOPT threshold against the noise floor F = 0.33 PDMS (`RESULT_P2REP`, one draw). A separated CI does not clear it (`H-ESTIM-SEED-1`); a replicate or a margin of several F is needed.
5. **PI decision (a).** Rule it before launch. Note `D-REFE-ZEROTAIL-4` closed the inference-side pre-compensation (net reach about 0); P5-C is training-side and is not covered by that closure.
6. **Tier and families.** T1 NAVSIM v1.1 under the standard tracker, no harness change. Four families per the binding rule; STRATEGIC reported PAUSED. Add the student-side learnability readout (train-label ADE, heading-rate error against Tc20s).
7. **K ledger:** one navtest read, K+1.
8. **Provenance pins:** table sha256, 150,603-row coverage, all rows `sim_hz` 10, yaw-wrap branch check, P7 bank sha `92af8d66...`.

## 5. Pod cost
**ESTIMATED**, from P6's measured 57.0 s/step (A40; `SPEC_P6` §0.2) with 0.863 as the L40 factor (memory, ESTIMATED from the P6 rate):
| arm | steps | A40-h | **L40-h** |
|---|---|---|---|
| P5-C at the P7 recipe (full retrain, incl. egress / stall / head, `SPEC_P7.md:74-85`) | 9,075 | 148-151 | **~128-130** |
| same, P6 recipe (no head, no reserve) | 10,075 | ~163 | ~141 |
| P5-C as a fine-tune (card wording), 2,000 steps | 2,000 | ~32 | **~27** |

- A replicate costs the same again.
- Label building is CPU only (Q2).
- Label table build and the gate run (CPU, 0 L40-h) must finish before the pod frees up.

## Order of work (0 L40-h until step 5)
1. MM and PI: rule decision (a), arm form and rank-1 policy.
2. Pull the P7 bank's `targets_rank*.jsonl` (rows only, no images) to the dev box, then run the `sim_hz` probe and the DB-coverage probe.
3. Build the table (~3 h wall on 2 processes) and run `check-bank`.
4. Stage the train.py, table tool, gate profile and test hold, with G-LIVE (a)-(e). The MM lands it after P7's last arm trains.
5. Write `PREREG_P5C.md`, hash it, then run the gate on CPU. A PASS token is bound to the exact commit, argv and data digests.

## Deliverable manifest
| artifact | where |
|---|---|
| this file | repo-external: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/P5C_READINESS.md` (single copy) |
| timing / identity probe | `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/evidence/{labonly.py,p5copy.py,ZZLAB.txt}` (single copy) |
| train.py hold | none (plan only) |
| Deletions | scratch files in `F:/tanitad_scratch/p5c/t20/` and `C:/Users/Admin/AppData/Local/Temp/` (lg.py, spec6.md, spec7.md, train.py), each logged in the hand-back; no git writes |
