# DESIGN_NOTES: PREREG_V8F_DAGGER (draft), 2026-10-26

Evidence classes: **MEASURED** (ours, artifact path) · **PUBLISHED** · **INHERITED** (another agent or document, not re-verified here) ·
**ESTIMATED** · **HYPOTHESIS**.
Code references are to the mirror at `e6a9a13c` (`git -C C:/Users/Admin/tanitad-push show e6a9a13c:<path>`).
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`.

⛔ No ORC or ORC-0 output was read: nothing under `E:/Projects/TanitAD-artifacts/v8f-orecover*`, `C:/Users/Admin/box2_ship/logs/`, or
`C:/Users/Admin/orct1/logs/`. From `orct1/` only `NOTES.md` and `HANDOFF.md` were read.

## 1. Code facts the design rests on (MEASURED by reading source at e6a9a13c)

| fact | where |
|---|---|
| ORC's injection replaces exactly `p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav` and keeps `feats, frame_ok, limit, limit_ok, a_t, a_s, pose_now`. `act` stays as the world model's record | `stack/tanitad/data/v8f_mini_data.py:584`, `:587`, `:590` |
| `EgoPerturb.apply`: the off path returns the same object with no draw; a private (seed, step) generator; the ONE rebuild line | `v8f_mini_data.py:669`, `:700` |
| A training item reads every label at ROWS: `lab` at t, `lab_t` at t − a_t, `lab_s` at t − a_s. `gt` is the logged future in the logged frame at t; `pose_now` = the logged pose at t | `v8f_mini_data.py:494`, `:512`, `:516`, `:518`, `:429` |
| T1-kin's `WindowState` holds a ClipStore-shaped dict whose ego rows after t0 are own-executed. `build_item` calls the SAME `ds.item` on it, so a T1 input is a training-shaped item | `stack/scripts/v8f_t1_eval.py:154`, `:201` (`_finish`), `:237`, `:239` |
| T1-kin's per-tick GT is time-matched ("rejoin the human's path"): the logged rows j+1..j+30 in the executed frame | `v8f_t1_eval.py:312` |
| The decisive T reader scores labels at r′ = the nearest logged pose among rows ≥ t0 (first minimum), with a 3 m off-route cut | `PK/raw/2026-10-26-t1hdread/t1hd_read.py:34`, `:161`, `:172` |
| T reads its own state through ONE ego token at T's frame (`_ego(p10f[last_t])`), plus nav and S's checkpoints moved by `pose_s → pose_t` | `stack/tanitad/models/v8f_mini.py:686`, `:745` |
| The S loss compares S's output (in S's tick frame `pose_s`) with `lab_s.rc` (in the label row's frame) | `v8f_mini.py:851` |
| The world-model term reads `act` only | `v8f_mini.py:884` |
| The trainer refuses any parser flag without a `FLAG_CHECKS` entry; ORC's swap point is `b = ego.apply(b, step)` on the collated batch in the main loop | `stack/scripts/v8f_mini_train.py:407`, `:1279`; G-LIVE template `ego_perturb_probe` `:765` |
| M2b's config: hist 4, **hmax 16**, max_age_t 2, max_age_s 6, kin_mode yaw, consistency on. The s0 checkpoint sha256 begins `601cb435267317862539` (matches ORC §1) | MEASURED, read from `sfresh/inputs/M2b_s0/ckpt_last.pt` (`ck["cfg"]`) |
| M2b's TRAIN set: 4,512 clips, 306,566 windows, `--exclude 540:600`; box 2's `cache_full` is not the parity corpus | `Project Steering/MODEL_REGISTRY.md` §17 (registry, primary); `PK/raw/full/runs/M2b_s0/argv.json`, `summary.json` (`n_windows` 306566) |

## 2. Design decisions, with the reason for each

1. **Route-paired, not time-paired.** The sample is the logged item at (clip, r′) with its ego tensors swapped. That is ORC's own invariant set,
   unchanged, so DAG and ORC differ only in where the replacement state comes from.
   * The time-paired alternative (clock-row cameras at j, labels at r′) teaches the T1-kin rig's camera/position mismatch. That mismatch does not
     exist at deployment.
   * It also couples labels at r′ with a scene at j: the light can change colour between the two rows.
   * MEASURED descriptive fact that makes route-pairing sufficient: T's lat F1 drop survives at zero row offset, 0.606 / 0.647 against 0.828 / 0.857
     (`RESULT_V8F_T1_HEADS.md`, "Is it the T head, or the replayed cameras?").
2. **The time-nearest tie rule and the polyline projection.**
   * The first-minimum point rule pairs a stopped own car with the FIRST row of the human's wait. That is the opposite of the launch lesson `R76`
     needs.
   * The point distance carries up to half a row of along-track spacing into the 3 m cut.
   * The probe's synthetic case `stationary_tie_differs_from_first_min` is True: the rules really differ (MEASURED, `probe/probe_out*.json`).
   * Scoring keeps the reader's rule, unchanged, so the decisive rows are ORC's rows.
3. **Re-anchoring at r′'s logged pose.** The model cannot see a lateral or heading offset at T1-kin: no map, logged images, and only relative odometry
   between held frames. The rigid re-anchor therefore removes nothing the model reads.
   * MEASURED in the probe: `pose_now` re-anchored = logged, max 0.0 m; the held-frame relative pose is preserved to 1.35e-5 m (float32 world
     coordinates).
   * Consequence, stated in the PREREG: no lateral-recovery lesson.
4. **`lab_s.rc` must be re-expressed.** S's output frame in a swapped sample is the own `pose_s`, not the logged pose at r′ − a_s (`v8f_mini.py:851`).
   Without the frame change, S's target, and T's teacher-forced S input (`:745`, p_gt 0.2), would be displaced by the difference between the own and
   the logged motion over a_s ≤ 6 rows. It is the identity at a_s = 0.
5. **`nav` follows ORC's rule, anchored at r′.** Token and validity are the route at r′; the progress history comes from the own arc.
   * Disclosed difference from T1-kin: the T1 tool holds the token from t0 (dead reckoning). The difference is identical for DAG and DAG-0.
6. **`limit` is kept as the clip's logged bucket**, as M2b trained and as ORC kept it. Training on the causal bucket would be a second lever.
7. **One pass, frozen M2b, ticks 1–2 of m = 10 rollouts from every TRAIN window.** One pass is the single-lever form. Iteration 2 is committed only as
   a next step (ROBUST / PARTIAL).
8. **The p mapping (0.3 / 0.1) is fixed by ORC's outcome row, never by its numbers.** Eligibility uses no speed cut: the turn-class loss is not a
   low-speed phenomenon. ⇒ DAG's dose is not ORC's, so DAG − ORC is beside only.
9. **The replicate floor comes free.** DAG-0's argv = ORC-0's + `--dagger-p 0`, so ORC-0 is a same-argv replicate. It answers "would another training
   run say this?" (`H-ESTIM-SEED-1`) without a fifth run. If the two runs are bit-identical, RF = 0 and the bar is ORC's 2F.
10. **Next steps are chosen by the cheapest discriminating experiment.**
    * NO EFFECT → T-EGOCUT: zero new data, and it decides whether T should read its own ego token at all.
    * Coverage failure → the rejoin-target form: the only way to supervise states more than 3 m off the route.

## 3. The probe (MEASURED; `C:/Users/Admin/dagger/probe/`)

**Script and constants.**
* `probe_dagger_cost.py`, sha256 `ee1136d8…`.
* Constants fixed beforehand: `THRESHOLDS_FIXED.txt`, sha256 `438366d5e5ce4077…`, printed by every run.
* Code: a `git archive` of `e6a9a13c` in the session scratchpad (`dtree/`). The pinned shared tool trees were not touched.
* Run on the dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`, 4 threads, one process, M2b_s0.
* ⚠️ The box was contended: load about 43 %, 4.4 GB free RAM, two other python jobs at about 0.7 GB each.

**Data scope.** Eval-141 windows were used for **timing only**: the TRAIN cache is not on the dev box. No own state was kept, and no off-route
count, lag or label statistic was computed or recorded.

**Synthetic matcher cases, all PASS (K-B4's seed):**
* straight;
* stationary tie, time-nearest;
* the tie rule differs from first-minimum;
* lateral cut;
* heading cut;
* end edge;
* loop.

**Identities (both banked runs):**
* the tick-0 own ego tensors equal the logged item bit for bit (`tick0_ego_bitexact` True, K-B0's seed);
* re-anchor `pose_now` max 0.0 m;
* the relative `pose_t` is preserved to 1.354e-05 m.

| run | file | windows | generation, ms per window (2 forwards) | of which forward | match, ms per own state | item build, ms per own state |
|---|---|---|---|---|---|---|
| spread (1 window per clip) | `probe_out.json`, `probe_run.log` | 128 | 156.4 | 20.3 | 18.56 (a clip reload per window) | 3.55 |
| **contiguous (clip-ordered, the TRAIN-pass pattern)** | `probe_out_contiguous.json`, `probe_run_contiguous.log` | 256 | **88.1** (excluding the 3.0 ms tick-0 control) | **35.7** | **0.23** | **3.52** |

* The first, unbanked run (spread, 128 windows) printed 111.4 ms per window before the forward timer existed. Its JSON was overwritten by the second
  run, so that number exists only in this note.
* `probe_out.json` was written by the revision before `--contiguous` was added. That edit added the flag and a tick-0-control timer, nothing else.

## 4. Cost arithmetic

| item | arithmetic | class |
|---|---|---|
| own states per seed, before cuts | 2 ticks × 306,566 windows = 613,132 | arithmetic on the registry count |
| dev-box CPU bank, per seed | 306,566 × 88.1 ms = 27,008 s = **7.5 h** (4 threads); 15.0 h for both seeds | ESTIMATED from MEASURED |
| ⚠️ but the dev box has no TRAIN cache | eval cache 2.8 GB for 141 clips (`du`), so 19.9 MB per clip × 4,512 ≈ **90 GB** | ESTIMATED |
| ⇒ the bank runs on **box 2** (i9-11900KF, RTX 4060, 32 GB; INHERITED, memory `box2-second-computer`) | CPU share ≈ 88.1 − 35.7 = 52.4 ms per window; 306,566 × 52.4 ms = 4.46 h single-process, so **≈ 1.1–1.5 h per seed with 4 clip-sharded processes**, the forwards on the GPU | ESTIMATED |
| the same, all on box 2's CPU | 88.1 ms × 306,566 / 4 processes ≈ 1.9 h per seed | ESTIMATED |
| bank size | per own state ≈ p10f 16×8 f32 (512 B) + pose 16×3 f64 (384 B) + arc 16 f64 (128 B) + kin3 (24 B) + ids and match (≈ 40 B) ≈ 1.1 KB; × 613,132 ≈ **0.67 GB per seed** before cuts | ESTIMATED |
| swaps used by one run | 3,000 steps × 64 = 192,000 samples; × p 0.3 × eligible e. At e ≈ 0.8, about 46,000 swaps from about 400,000 kept own states: no reuse to overfit | ESTIMATED (e is unknown until the bank exists) |
| training overhead | ≤ 3.5 ms per swapped row × (0.3 × 0.8 × 64 ≈ 15 rows per step) ≈ 54 ms per step, against M2b's 4.4 h / 19,164 steps = 0.826 s per step: **≤ +6.5 %** | ESTIMATED |
| reached-training 3σ | √(0.3 · 0.7 / 153,600) × 3 = 0.0035 (p 0.3); √(0.1 · 0.9 / 153,600) × 3 = 0.0023 (p 0.1); against a ±0.02 bar | ESTIMATED |
| box-2 total | bank 2.2–3 h + 4 runs × ≈ 0.7 h (the box-2 chain's own estimate, INHERITED from `orct1/NOTES.md` §10) + gates and T0 evals ≈ **5–6.5 h** | ESTIMATED |
| dev-box T1 reads | as ORC: 8 rollouts (181–360 s each, INHERITED `orct1/NOTES.md`) + 4 fresh T0 evals (21 s each, INHERITED) + the reader ≈ 1 h of CPU, 2 lanes | INHERITED / ESTIMATED |

## 5. What must be built (nothing exists yet; ESTIMATED lines)

| # | file | content | lines |
|---|---|---|---|
| 1 | `stack/tanitad/data/v8f_dagger.py` | `match_route`, `reanchor`, `DaggerBank` (load shards, sha256, inverse map (ci, r′) → records, header checks), `DaggerSwap` (`apply(batch, step, p=None)` with EgoPerturb's contract, `eligible`, `built_levers`, `problems`, the `lab_s.rc` frame change) | ~300 |
| 2 | `stack/scripts/v8f_dagger_bank.py` | the generator. It imports `v8f_t1_eval`'s `make_states / build_item / model_planner / WindowState` (no tool change), rebuilds M2b's TRAIN list via the trainer's `build_data`, shards by clip, writes npz shards and a header | ~220 |
| 3 | `stack/scripts/v8f_mini_train.py` | 4 flags, `FLAG_CHECKS` entries, `make_dagger`, `bad_dagger`, `dagger_probe` (G-LIVE), the loop line `b = dag.apply(b, step)` beside ORC's, run totals | ~200 |
| 4 | `stack/tests/test_v8f_dagger.py` + `mutate_dagger.py` | on `make_synthetic_world`: K-B0, K-B3's refusal, K-B4, K-B5, swap partition, the p = 0 bit-exact 3-step trace, G-DVB refusals, the at-rest / identical-row exemption; mutants for the swap line, the loop line, the rc frame change and each probe clause | ~350 + ~150 |
| 5 | `PK/raw/…-dagread/dag_read.py` | from `orc_read.py`: arm names, the pair rule, `--key-built-p dagger.dagger_p`, the RF row and the bar max(2F, 2·RF), BLOCKED, the reached-training keys | ~120 changed + tests |
| 6 | `dag_bank_qa.py` | K-B0 to K-B5 and COVERAGE on the real bank; lag and cut tables | ~120 |
| 7 | `code/box2_ops/chain_v8f_dag.sh` | stages `bankdag / gatedag / traindag / evaldag` (from `chain_v8f_orc.sh`). The predecessor predicate reads ORC's read outcome | ~150 + test |
| 8 | the dev-box T1 chain | orct1, re-parametrised for DAG / DAG-0 | ~60 |

Total ≈ 1,700 lines with tests. Agent-days: ESTIMATED 1–1.5 days of build, all CPU-only, and none of it touches box 2 until the bank stage.

## 6. Open questions (need the PI or the Master Mind; NOT settled here)

1. **PI: box-2 time** for the bank (≈ 2–3 h) and the four runs (≈ 3 h), as O-RECOVER needed. May the bank run speculatively, before ORC's read?
2. **MM: route-paired versus time-paired.** I chose route-paired (§2.1). If the PI reads "own T1-kin states" as "exactly the T1-kin input", the
   time-paired form is the alternative. It is a different arm, and the two should not be merged.
3. **MM: a zero-training diagnostic BEFORE this arm, T-EGO-SWAP.**
   * At T1-kin, feed ONLY T's ego token the logged p10f at r′_T.
   * If lat / lon F1 at ticks ≥ 1 recovers most of the 0.23 / 0.13 drop, T's ego token is confirmed as the channel this arm targets. If it does not,
     the drop runs through nav or S's moved checkpoints, and DAgger's prior falls.
   * About 4 CPU rollouts. It is independent of ORC, so it can run now.
4. **MM: the coverage floor of 0.5 and the 30° heading cut are ESTIMATED, not derived.** They are fixed before data. Say now if they should differ;
   after the bank exists they cannot move.
5. **MM: should DAG-0 reuse ORC-0** (save about 1.4 h of box 2) instead of being re-run? I chose to re-run: the control is trained by the same build
   as its arm, and the pair gives the replicate floor.
6. **Whether ORC Amendments 1–4 carry over verbatim** (K4's fresh-age comparator, K7 = the rebuilt track, the pair-rule form, the reader's adopted
   interpretations). I assumed yes. They are named in the PREREG §6.

## 7. Not verified (UNVERIFIED)

* That box 2's TRAIN cache and `join_train.npz` serve `ds.item` for every TRAIN window with `kin_mode yaw`: the bank generator was not run on TRAIN.
* That the S label `rc` is expressed in the label row's own logged frame. It is inferred from `t1hd_read.py`'s `src = PGr(rS)` and from the loss's
  comparison in S's tick frame, not from the label builder's source.
* The eligible fraction e, the cut counts and the lag distribution: unknown until the bank exists, by design.
* That ORC-0's training is bit-reproducible on box 2. If it is not, RF > 0, and the bar rises by design.
