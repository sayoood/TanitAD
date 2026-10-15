# S1 end-to-end read driver (`s1_read.py`), 2026-10-15

Tier of everything it reads: **T0-like** (held-out selection and perception on teacher-forced banked sets; `eval/S1_POD_RUNBOOK.md` l.5), never driving performance.
Estimator everywhere: paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927 (`aux_s1_readout`'s). It answers "another draw of LOGS" only; training-run variance is the N-vs-N seed floor.
Evidence classes: MEASURED (artifact named) / UNVERIFIED.

## 1. Headline: what the Master Mind must do

1. **Land `s1_read.py` and `test_s1_read.py` into `eval/`** (copies are in this folder; `ZZTEST_PASS 94`, log in `ZZTEST_s1_read.log`). Nothing is staged (brief's exception).
2. **A real instrument defect was found by the dry run: `rl_val_eval.load_scores` cannot load a real S1 dump.** MEASURED: `aux_s1_readout.py selection` on the three banked N arms died with `numpy ArrayMemoryError` after 38 s (`dryrun/` history, reproduced twice).
   - Cause (`eval/rl_val_eval.py:140-142`): the loop reads `z["scores"][i]` per row. `NpzFile.__getitem__` re-reads and re-decompresses the whole array on every call (10.7 MiB = 3137 x 64 x 7 float64), and `np.asarray` of the row keeps a **view**, so 3,137 rows pin 3,137 copies = **33.5 GB per arm** on a 31.8 GB box.
   - It was invisible to the harness tests (fixtures of 12 sets). The driver runs `selection` through a shim (`load_scores_copy`, `_harness` subcommand) that reads each array once and copies each row; every run first proves the shim equals the original on a 48-set file (`check_loader_equivalence`), and `chk_loader_shim` + 3 RED mutations pin it. Result with the shim: 20 s for three arms.
   - Proposed fix to `rl_val_eval.py` (not applied; the driver works without it, and the shim stays harmless if you land it): in the `.npz` branch read each array once (`a = {k: z[k] for k in z.files}`), index `a[...]`, and pass `np.array(a["scores"][i], np.float64)` to `put`.
3. **Four prereg wordings need a ruling before the full read** (the driver implements the literal reading, reports the alternative's outcome beside it, and never resolves them silently). AMB-1, AMB-2, AMB-3, AMB-4 below; the others are recorded in the output.
4. **The completeness rule is A5, not the brief's convenience.** The driver reads only when all NINE arms (N, H1, H2 x s0, s1, s2) are present and clean (`PREREG_AUX_DRIVABLE_S1.md` A5 l.298, A3.3 l.233-237: the H1-s1 / H1-s2 drop rule is not exercised). Reading H2 alone after six arms would need a dated amendment.

## 2. Files in this folder

| file | what |
|---|---|
| `s1_read.py` | the driver (copy destined for `eval/`) |
| `test_s1_read.py` | synthetic-arm tests, real-data checks on the banked N arms, 60 RED mutations |
| `ZZTEST_s1_read.log` | the test output, `ZZTEST_PASS 94 (29 checks + 4 real-data + 60 RED mutations + coverage)` |
| `dryrun/S1_READ.json`, `S1_READ.md` | the driver's output on N-s0, N-s1, N-s2: **INCOMPLETE** |
| `dryrun/fetch_manifest.json` | 27 files, 44,664,585 B, path / size / sha256 / what it was verified against |
| `dryrun/instrument/selection_N_only.json` | the harness's N-vs-N selection read (a rig property, see 4) |
| `dryrun/swap_demo/` | real N-s1 evidence filed as N-s2, end to end: **REFUSED** |
| `dryrun/headpath_standin/` | the head path on the untrained seed-0 head (plumbing control) + `CONTROL.md` |
| `HANDOFF.md` | the next run |

## 3. Read-to-clause map (`s1_read.py` line, prereg clause, harness function)

| read | driver | prereg clause | instrument |
|---|---|---|---|
| completeness (nine arms) | `decide:786`, `reads_table:152` | A5 l.298; A3.3 l.233-237 | - |
| arm identity (label vs argv / token / manifest / run_check / sft) | `check_arm_evidence:365` | S1 2 l.51 (token per arm), proposal 3 (argv); PI rule "swapped label" | - |
| cross-arm parity (argv core, distinct dumps, H2 acts) | `check_cross_arm:457` | S1 1 l.28-31, proposal 4 (same data order within a seed) | - |
| fetch verification | `verify_file:198`, `fetch_verified:218` | brief; `S1_POD_RUNBOOK` 4.6 (LFS sha256 / git blob id) | - |
| seed floor (3 N, largest abs delta) | `components.seed_floor` (N arms only) | S1 4.3 l.108, l.126 | `aux_s1_readout selection` |
| pair H-sK minus N-sK, PDMS and DAC classification | `classify:501` (re-derived, compared with the harness) | S1 4.3 l.108, 2 l.48 | `selection` |
| K2 (IoU [0,40) < 0.80), navtest governs | `k2_from:512` | draft K2; S1 4.1 l.93, l.98 | `perception` (navtest, then P1-HO) |
| P3 + A2 floor | `p3_decide:485` (re-derived, compared with the harness) | S1 3 R2, R3; A2.1 l.154-156 | `aux_navtest_forward` + `perception` with `--pred-init` |
| K6 | `read_k6:535` | draft K6 | `sft.jsonl`, G12 `bce_declared` |
| K3 | `read_k3:553` | draft K3; proposal 2 | `sft.jsonl`, `PREFLIGHT_PASS.json` |
| K5 (exact controls) | `k5_problems:446` | draft K5 | G12 raw fields |
| H1 identity | `read_h1_identity:573` | S1 4.3 l.124 | `selection` pair H1-sK minus N-sK |
| H2 promotion | `promote_h2:582` | S1 4.3 l.110-118 | - |
| H1 promotion | `promote_h1:607` | S1 4.3 l.120-122 | - |
| four families | `families_block:764` | draft "Four families"; S1 4.2 l.100-104 | `selection` summaries |
| estimator pin | `check_harness_json:616`, `check_pins_against_harness` | S1 2 "harness commit" row | - |

**Families (binding, per family, never pooled).** LONGITUDINAL: EP and TTC of the pick (paired CI) measured; speed / along-track / headway not computed. LATERAL: DAC of the pick measured; heading / cross-track / curvature / yaw-rate not computed. TACTICAL: pick / random / best per arm measured; decision accuracy / kappa / command adherence not computed. All "not computed" need the navtest planner seam, which does not apply H2's feed (S1 4.2). **STRATEGIC: PAUSED** (PI 2026-09-19, item 25): printed as PAUSED with the ruling, not a pass, not omitted.

## 4. Dry run on the real N arms (MEASURED, `dryrun/`)

- Command: `s1_read.py read --arms N-s0 N-s1 N-s2 --cache D:/Projects/TanitAD-artifacts/s1_read_cache --out dryrun`, clean cache, 2026-10-15.
- Output: `ZZS1READ_INCOMPLETE ... instrument_failures none`. Status INCOMPLETE; missing H1-s0, H1-s1, H1-s2, H2-s0, H2-s1, H2-s2; every read names its missing arms; **no outcome label** (checked by `chk_incomplete_gate`). No H-vs-N conclusion exists anywhere in it.
- Fetch: 27 files, 44,664,585 B, 27 downloaded / 0 cached, **all verified** (size + LFS sha256 for the dumps and the tar, git blob id for the small files; the PASS token's file sha256 equals the manifest row's `token_sha256`).
- Wall time per step: listing 0.6 s, fetch 47.0 s (0.95 MB/s), untar of `s1_heldout_pdm.tar` 1.0 s, selection (3 arms, 3 pairs) 21.6 s, **total 71.7 s**.
- Not fetched, with the reason: `model_sft_B.pt` (1.29 GB per arm: no read needs it, the selection read uses the banked score dump, the head read uses `aux_drv.pt` on the cached u400 ctx because the trunk is frozen, S1 4.1 and A2.3), `scorer_states_last.pt` (34 MB), `scores_update400_A/_base`, `scores_step0_*` (the common init is compared by listing sha256 only).
- **N-only component (a rig property, stamped `not_an_outcome`):** the seed floor over N-s0/N-s1/N-s2, MEASURED (`instrument/selection_N_only.json`, held-out 3,137 sets, 24 logs, rule navsim_v1): largest |paired mean delta| PDMS **0.352**, DAC 0.255, EP 0.546, NC 0.223, TTC 0.223 (x100 units). Under the harness's `classify`, an H2-sK pair counts "above the seed floor" only if its PDMS CI lower bound exceeds 0.352.
- **Swap demo (`dryrun/swap_demo/`):** the real N-s1 evidence filed under the label N-s2 -> `REFUSED`, seven independent codes: `ID_MANIFEST_NO_ROW`, `ID_RUNCHECK`, `ID_ARGV_SEED`, `ID_ARGV_OUT`, `ID_ARGV_DUMP`, `ID_SFT_DUMP_PATH` (x2), `DUP_SCORES`. Read-through: no outcome, the arm is excluded from the seed floor.
- **Head-path control (`dryrun/headpath_standin/CONTROL.md`):** the driver's own commands on the untrained seed-0 head: forward 325 s on CPU, arrays **bit-identical** to the banked A2.4 floor, perception 10 s, trained == init gives the paired difference exactly 0.0 [0.0, 0.0].

## 5. Tests (`test_s1_read.py`, `ZZTEST_PASS 94`)

29 checks, 4 real-data checks (the banked N arms pass under their own labels, the cross-arm check is clean, the real evidence of one arm is refused under another label), **60 RED mutations**, each snippet occurring once in the source. By class:
- pins (6): `N_BOOT` 1000, seed changed, rule `navsim_v1_lane`, K2 bar 0.70, K6 factor 0.8, the selection command without `--n-boot`.
- estimator (2): the harness JSON's resamples/seed unchecked, the harness tool name unchecked.
- gate (4): INCOMPLETE gate off, REFUSED gate off, SEL read without the three N, K3 read without the same-seed N.
- identity (9): seed / class / out / manifest row / last manifest row / run_check (the six that catch a swapped label), argv sha, updates, K5 fields unchecked.
- parity (3): duplicate dump, H2 equal to its N, argv core drift.
- fetch (7): sha256 / size / git blob unchecked, cache trusted without re-verifying, unverifiable listing accepted, tar traversal allowed, partial push counted as an arm.
- rules (26): the A2 floor condition dropped (flips the outcome at lower bound <= 0), floor `>=` 0, FLOOR_UNREAD treated as a pass, margin dropped, kill boundary; classify ignoring the floor; K2 boundary / navtest not governing / label unchecked; K6 strict, wrong row; K3 three limbs; H2 promotion: provisional needs 3, floor unmeasured, adverse ignored, K2/K3/K6 ignored, majority-only P3; H1: identity branch off, kill ignored, defect never; re-derivation off (P3, classification); STRATEGIC not paused; plumbing with the wrong seed's floor.
- shim (3): row is a view, off-by-one row, duplicate key allowed.
Pins and the A2 rule are also cross-checked against the REAL harness on a grid (`chk_p3_matches_harness`: 192 cases incl. NaN and None; `chk_classify_matches_harness`).

## 6. Prereg ambiguities (not resolved silently; full table in `S1_READ.json` -> `ambiguities`)

| id | text | used | alternative recomputed? |
|---|---|---|---|
| AMB-1 | cond 1 names "the P3 decision" but there are 3 H2 seeds | all 3 seeds in {EVALUATED, PROVISIONAL}; PROVISIONAL regime if any is | yes (`sensitivity_AMB-1`: >= 2 of 3) |
| AMB-2 | "K2 / K6 on H1's head" with 3 H1 seeds | any seed fires => not promoted | yes (`sensitivity_AMB-2`: H1-s0 only) |
| AMB-3 | K6 "0.9 x step-0 after 10 % of the steps": which number and row | G12 `bce_declared`; first train row with update >= 40, i.e. update 50 (a 25-update mean) | yes (`alt_fires`, `sensitive_to_ambiguity`) |
| AMB-4 | K3 "+3 % step time or 1 GB at micro-batch 4"; S1 uses batch 8 | memory: `head_retained_gb` > 1 or preflight `gpu_mem_gb` delta vs same-seed N > 1; time: run-mean `s_per_update` at update 400 vs same-seed N > 3 % | limbs reported separately |
| AMB-5 | K5 is a check on an exact test | G12 raw fields; a failure REFUSES the arm | - |
| AMB-6 | held-out rule not named | `navsim_v1` (draft; the harness passes L=None) | - |
| AMB-7 | "investigated before any read" | blocks the H1 outcome, stamps the H2 outcome | - |
| AMB-8 | P3 decision names are H2_* | decided for H2 arms only; H1's P3 reported | - |
| AMB-9 | "above the seed floor" | the harness's `classify` (re-derived) | - |

## 7. UNVERIFIED

- **No H arm was seen.** Field names for head arms come from `refe/scorer_finetune.py` at the tip (`aux_eval` / `aux_bce` rows, G12 raw fields incl. `bce_declared`, `head_retained_gb`, `dump/head_pred_{step0,update400}.npz`, `arms/<ARM>/aux_drv.pt` per runbook 4.6). A different name makes the arm PARTIAL (named paths) or REFUSED (named code), never a wrong read.
- The COMPLETE path of `decide()` is tested on synthetic harness JSON shaped from the harness source, not on a real nine-arm run. The P1-HO call on a real dump and the nine-arm selection run are untested.
- Full-read cost is ESTIMATED: fetch about 250 MB (about 4-5 min at 0.95 MB/s), forward 6 heads about 14 min (one head 325 s MEASURED, about 100 s per extra head from the 3-head floor run), perception about 1 min, selection about 1 min: **about 25 min, run it in the background**.
- A trained head is read through the planner's fp32 ctx (A2.3) although it trained under bf16 (1-1.5 % relative L2, MEASURED in the forward tool's smoke); not re-measured.
- The head pass runs on the `driverl-eval` venv (torch 2.7.1, CPU), the venv that produced the A2.4 floors; the control shows it reproduces them bit for bit (MEASURED, untrained head).
- K3's time limb uses the pod's run-mean `s_per_update`; a pod with other load would bias it.
- D: is a `subst` alias of E: (`subst D: E:\` if missing); the ctx cache (8.4 GB), the floor heads and the held-out logs file live under `D:/Projects/TanitAD-artifacts/` (dev box only; the held-out logs file's sha256 `b814998d...` equals the PASS token's binding).

## 8. The command for the full read (all nine arms on HF)

```
cd "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1)" PYTHONIOENCODING=utf-8 \
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/s1_read.py read \
  --cache D:/Projects/TanitAD-artifacts/s1_read_cache --out raw/<date>-s1-read
```
Background it (about 25 min ESTIMATED). Output: `S1_READ.json`, `S1_READ.md`, `fetch_manifest.json`, `instrument/*`. Marker: `ZZS1READ_READ` (exit 0), `ZZS1READ_INCOMPLETE` (0), `ZZS1READ_REFUSED` (3), instrument failure (4).
Plumbing check of one head arm (values go to raw files only): add `--plumbing-check H2-s0`.

## 9. Deliverable manifest

| artifact | where | copies |
|---|---|---|
| `s1_read.py`, `test_s1_read.py`, `ZZTEST_s1_read.log`, `NOTES.md`, `HANDOFF.md`, `dryrun/**` (238 KB, no file over 1 MB) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-15-s1-read-driver/` (E: working tree, unstaged by instruction) | **1 (E: only)** |
| working copy of the two scripts for running | `C:/Users/Admin/AppData/Local/Temp/s1rd/.../eval/` (temp extraction of tip 066ca5a) | disposable |
| fetch cache: 3 N arms, labels, `swap_repo`, `swap_cache`, `headpath_standin/heads/head_pred_H2-s0.npz` (11 MB) | dev box only: `D:/Projects/TanitAD-artifacts/s1_read_cache/` | regenerable (HF, deterministic) |
