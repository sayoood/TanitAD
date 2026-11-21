# P6-ADHERENCE: the command-adherence instrument for P6's pre-registered navtest read (2026-10-22, 0 GPU)

**Headline.** The instrument is built exactly as PREREG_P6 Amendment 2026-10-22 (b) / `MM_RULING_P6_READ_GAPS.md` E3 defines it, wired into the read runbook, and validated on the banked P2 (C2) and u400 (C1) navtest plans. `ZZADH PASS 24/24` (every check has a RED arm); six source-level mutations each turn the suite RED. **No P6 checkpoint output on navtest was read or produced.** No parameter was missing, so there is no STOP question (one disclosure for the record: section 3).
⛔ **INTEGRATION ESCALATION (Master Mind): land 2 NEW files + 1 runbook edit.** Nothing else changes: no existing file is edited, so no pin moves (`stage2_sft.py` md5 `3a30ab4c...` and `STAGE2_SFT_MD5` untouched). Then close BACKLOG row `P6-ADHERENCE`. Base verified: `eval/`, `raw/2026-10-22-p6-read-gaps/`, `taniteval/`, `stack/tanitad/refs/` are byte-unchanged from `a9bc000` to the current mirror tip `a8f2c89` (only `refe/` trainer files moved).

Evidence classes: **MEASURED** = ours, with the artifact named. **INHERITED** = another doc, not re-verified. Tier of every adherence number: **T1** (a NAVSIM plan, self-action open loop; the command is an INPUT of the agent).

## 1. What was built

| file | role |
|---|---|
| `PK/eval/command_adherence.py` (NEW) | the instrument: `read` (P6 read-day command) and `control` (banked-arm known-value read) |
| `PK/eval/test_command_adherence.py` (NEW) | 24 checks, each with a deliberate-regression arm |
| `PK/raw/2026-10-22-p6-read-gaps/RUNBOOK_P6_READ.md` (EDIT, diff in `code/RUNBOOK_P6_READ.diff`) | preflight line, the step-5 command replacing the "NOT IMPLEMENTED" bullet, traps |

`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

Read-day command (0 GPU, about 1 min, needs only the step-3 seam and the dump; it does not wait for the 4-6 h selection-quality columns):
`python eval/command_adherence.py read --name p6_full --dump <seams/stage2_p6_full.proposals.npz> --base u400 --also p2`

It follows how `selection_quality.py` was wired: a standalone CLI plus the runbook plus a test file (grep of the tip: `stage3_full.py` never imports `selection_quality.py`), so `stage3_full.py` is not edited. Output JSON, per arm (P6, C1, C2): accuracy and Cohen's kappa with CI and n, on all tokens and on the turn-active subset; the 3x3 confusion (command rows x plan columns); per-command recall; the random-proposal floor with `actual_minus_floor`; the paired P6 - C1 and P6 - C2 differences (accuracy and kappa, separately); the raw-proposal descriptive row; gates; controls; the human-future reference (descriptive, not part of the definition). Authority is stamped **NONE**; STRATEGIC is stamped PAUSED.

## 2. The estimator, found and quoted
The registered read uses **`proxy_eval.boot`** (`eval/proxy_eval.py:313-327`), called from `stage3_full.py:216` (`PE.boot(d, tl, n=n, seed=seed)`) with `N_BOOT, SEED = 10000, 20260927` (`stage3_full.py:77`; imported at `:60`). `selection_quality.py` uses the same one (`:38,:45,:164,:186`). `taniteval/ci.py::episode_cluster_bootstrap` (`ci.py:225`, draws by `rng.choice` at `ci.py:158-165`, `n_boot` 2000 in the families blocks) is the families-block estimator and was **not** used.
`proxy_eval.boot` is defined only for a per-token mean. Accuracy is a mean, so it runs through the registered stream. Kappa is a function of the 3x3 confusion, so the instrument uses the **same resample stream** (`default_rng(seed)`, one `rng.integers(0, G, G)` per resample, logs sorted) on per-log confusion matrices. This is the identical resample, not a second estimator: the accuracy intervals are **bit-equal to `proxy_eval.boot`** on the same tokens, levels and paired differences, gated on every read (`G_PE_BOOT_EQUAL`, 6 of 6 equal on the control) and tested (`C3`, `C4`).
It answers one question: *would another draw of LOGS say this?* (MEASURED design property.) Training variance (one seed per arm, H-ESTIM-SEED-1) is unmeasured; inference is deterministic. Turn-active intervals resample the 113 logs that contain turn-active tokens (an n_logs per cell is reported).

## 3. Applying the labeller to REFe's plan format: no free parameter, one disclosure
* Plan = the seam's `poses` [N,8,3]: 8 poses at 0.5 s, ego frame (x forward, y left). Fixed by `refe_navtest_seam.py:41` (`NAVSIM_DT, NAVSIM_N = 0.5, 8`, `to_navsim`) and by the banked call `families6.py:76` (`scenes_to_win(..., frame="ego", origin_included=False, dt_s=0.5)`).
* Labeller call = the one `tactical_from_trajectory` makes: `four_families.maneuver_kinematics(pred_xy, 0.5)` then `refc_tactical.factor_from_kinematics`, LATERAL factor (`lane_keep / turn_left / turn_right` = STRAIGHT / LEFT / RIGHT). `command_adherence.py` assigns **none** of the labeller's thresholds (test `B3`), and patching the canonical `YAW_TURN_RAD` changes its output (test `B2`: the instrument really reads the canonical labeller).
* Command order (LEFT, STRAIGHT, RIGHT, UNKNOWN) = `refe/planner.py:184-186`.
* **Proof that this is the banked call, not a lookalike (MEASURED):** applied to (executed plan, human future) it reproduces the banked families blocks' lateral confusion **exactly**, for both arms, and their heading-held / never-moved counts (P2 995/39 plan, u400 1,051/59, human 1,276/41): P2 `[[6971,296,247],[185,2626,2],[196,1,1622]]`, u400 `[[7003,253,258],[193,2615,5],[199,2,1618]]`. Artifacts: `raw/probes/probe1.out`; banked files `raw/2026-10-05-sft-stage2/p2_full/p2_full_families.json`, `raw/p2_r2_instrument/baseline/u400_navtest_full_families.json` (md5-pinned in `stage3_full.BASELINES`). Test `B1` also checks it live against `tactical_from_trajectory` itself (`B4`). The E: tree's `four_families.py` differs from the mirror tip only at lines 483, 694-695, 768-799 and 1834 (longitudinal lead-viol and the per-window export), outside `_seq_geometry`, `_kappa_k`, `_class_report`, `maneuver_kinematics` and `tactical_from_trajectory`, and `refc_tactical.py` is identical (LF-normalised); the control gives identical numbers under both trees (MEASURED, `raw/control_p2_vs_u400.Etree.json` vs `raw/control_p2_vs_u400.json`).
* **Disclosure (not a blocker, no choice made):** the v1 lateral gate is `|dyaw| > 0.15 rad` where `dyaw` is the path tangent at the END of the supplied plan, i.e. at 4.0 s for REFe's plan, while the labeller's own `LABEL_HORIZON` constant is 20 steps = 2 s at 10 Hz. The banked families path already applies it unchanged to the 4 s plan, and the definition says "as used by `tactical_from_trajectory`", so this instrument does the same. If the MM wanted a 2 s class (the first 4 poses) that would be a new parameter and needs a dated amendment BEFORE the read. Question for the record: *keep the full 4 s plan (as built), or amend to 2 s?*
* **A7 / "raw proposal descriptive":** A7 rewrites only `poses[..., -1, 2]` (`planner.py:159-167`) and the labeller reads only xy, so the raw and executed classes are **identical by construction**. The pre-A7 plan is not banked (seam and dump both hold executed forms). Reported as `IDENTICAL_BY_CONSTRUCTION` with the measured 0 class differences when the whole heading column is zeroed (P2: 0 of 12,146).

## 4. Known-value control: P2 (C2) vs u400 (C1), BANKED navtest plans (MEASURED; not a P6 look)
Inputs: P2 plans `stage2_p2_full.npz` (md5 `49a1f6ec...`, sha256 `a0401350...`), u400 plans `stage2_sft4_B_u400_navtest_full.npz` (md5 `b009e941...`, sha256 `efbaa194...`), commands and logs from the W3 export `navtest_inputs.json.gz` (sha256 `b0dc05f3...`), P2's banked 64-proposal table `p2_full_table/table_p2_full.npz` (sha256 `33441082...`, for the floor). Output `raw/control_p2_vs_u400.json` (INVALID none, INCOMPLETE none; all 7 gates PASS; 15 s).
**Command counts reproduce the ruling exactly:** LEFT 2,501 / STRAIGHT 8,070 / RIGHT 1,575 / UNKNOWN 0; turn-active 4,076; 12,146 tokens, 136 logs.

| arm (T1, banked) | subset | n (logs) | accuracy [95 % CI] | kappa [95 % CI] |
|---|---|---|---|---|
| P2 (C2) | **turn-active (headline)** | 4,076 (113) | **0.8827** [0.8586, 0.9053] | 0.7786 [0.7338, 0.8188] |
| P2 (C2) | all tokens | 12,146 (136) | 0.8638 [0.8423, 0.8847] | 0.7422 [0.7079, 0.7747] |
| u400 (C1) | **turn-active (headline)** | 4,076 (113) | **0.8832** [0.8591, 0.9052] | 0.7794 [0.7359, 0.8191] |
| u400 (C1) | all tokens | 12,146 (136) | 0.8682 [0.8460, 0.8895] | 0.7498 [0.7140, 0.7837] |

3x3 (command rows LEFT / STRAIGHT / RIGHT, plan columns): P2 `[[2278,210,13],[638,6894,538],[7,248,1320]]`; u400 `[[2271,211,19],[592,6945,533],[7,239,1329]]`.

**Paired C2 - C1 (P2 - u400, identical tokens, identical resamples):**

| subset | accuracy delta [95 % CI] | kappa delta [95 % CI] |
|---|---|---|
| turn-active | -0.0005 [-0.0050, +0.0044], not separated | -0.0008 [-0.0084, +0.0075], not separated |
| all tokens | -0.0044 [-0.0083, -0.0002], separated | -0.0075 [-0.0145, -0.0002], separated |

Read it as: the two banked arms adhere equally on the headline cell; on all tokens P2 is lower by under half a point (the CI edge is -0.0002, and this is a single-seed pair, so it is NOT a lever claim). Descriptive only; no authority.

**Random-proposal floor (P2's 64 proposals, MEASURED):** turn-active floor accuracy 0.8920 [0.8724, 0.9092], kappa 0.7945 [0.7591, 0.8251]; all-tokens floor 0.8611 [0.8411, 0.8805], kappa 0.7395. `actual - floor`: turn-active -0.0092 [-0.0189, -0.0003] (kappa -0.0159 [-0.0322, -0.0008]), all tokens +0.0027 [-0.0041, +0.0098]. So for P2 the selection does not add adherence beyond what the 64 proposals already have (they are all goal-conditioned); on the turn-active subset it is slightly below the floor, which the instrument flags (`floor_ge_actual_point`) and does not treat as an instrument failure. The floor at P6 may read very differently (P6 is command-conditioned).

**Descriptive reference, not part of the definition:** the same labeller on the logged human future vs the same command reads turn-active 0.8810 [0.8557, 0.9026], all tokens 0.8758. The 0.88 plateau is the labeller-vs-command agreement ceiling of the human path itself (the command comes from the route, the class from a 4 s path tangent), so a P6 reading near 0.88 is not "poor adherence".

## 5. Controls that must read their known values (all MEASURED, all in `raw/control_p2_vs_u400.json` and the tests)
* **RED arm, LEFT<->RIGHT swapped commands:** P2 turn-active accuracy falls from 0.8827 to **0.0049** [0.0023, 0.0080], kappa to **-0.7077**; all-tokens accuracy 0.5692.
* **Zero arm, constant-STRAIGHT plan:** kappa **= 0.0 exactly** (all tokens and turn-active; test `D3` uses `==`), accuracy 8,070/12,146 = 0.6644 on all tokens and 0.0 turn-active. A perfect plan reads 1.0 (RED). Any constant plan reads kappa 0.
* **Floor identity:** the floor equals the brute-force mean over the 64 columns (1e-12); `proposals[pick]` equals the seam bit for bit after joining on `token` (a shuffled `pick` or a short token set is INVALID).
* **kappa and confusion equal the programme's own** `four_families._kappa_k` / `_class_report` (1e-12), also on random vectors; an undefined kappa is NaN/None, never 1.0.
* **Pairing:** C2 - C1 equals acc(P2) - acc(u400) exactly; C1 - C2 mirrors it; the paired interval is far narrower than an unpaired one (the RED arm "paired == unpaired" fails).

**Mutation (source level, in the overlay copy only; `raw/mutation_*.log`, `raw/mutation_run.py`):** each turns the suite RED.

| mutation | RED checks |
|---|---|
| M1 class mapping rotated (`LAT_TO_CMD` = STRAIGHT, RIGHT, LEFT) | B1, B1-mutation, B4, D1, D2 (19/24) |
| M2 lane_keep mapped to LEFT | B1, B1-mutation, B2, B4, D1, D2 (18/24) |
| M3 labeller re-implemented with a smuggled 0.15 threshold (behaviourally identical) | B2 (the reachability test is the only one that can see it) (23/24) |
| M4 bootstrap stream seed shifted | C3, C4, D1 (21/24) |
| M5 dt = 0.1 instead of 0.5 | B1, B1-mutation, D1 (21/24) |
| M6 STRAIGHT/UNKNOWN swapped in the command decode | A1, A2, D1, D3, D4 x2, D5, D6 (16/24) |

## 6. Findings that are not about adherence (report, do not bury)
* ⚠️ **Name collision: `D:/Projects/TanitAD-artifacts/c2_navtest_2026-10-22.tar` is NOT P2's executed plans.** It is the **NAVPIPE "C2" read** (`final/RESULT_NAVPIPE.json`: arm "P2HEAD (no head substitution)", outcome `C2_FAIL_LOWER_CI`, dPDMS -0.155 [-0.350, +0.037]); the brief's "C2" there is a different workstream's label. PREREG_P6's C2 = P2, whose banked per-token executed plans are `stage2_p2_full.npz` (md5-pinned in `stage3_full.BASELINES`). The tar was not used. (GNU `tar -tf D:/...` also fails: "Cannot connect to D:"; use `--force-local` or python `tarfile`.)
* The proposal table's token order is chunk order, not the seam's: joining by position makes `proposals[pick]` differ from the seam by up to 50.7 m and gives a nonsense floor (0.46 turn-active in my first probe, which I discarded); the instrument joins on `token`.
* Not wired into `stage3_full.py` (by design, as selection quality). If the MM wants one combined JSON that is a later, pin-moving change.

## 7. How to re-run (the MM's method: overlay of the mirror tip, package files on top)
`bash raw/make_overlay.sh && bash raw/sync_overlay.sh`, then from the overlay `PK`: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I eval/test_command_adherence.py` -> `ZZADH PASS 24/24` (about 50 s), and `... eval/command_adherence.py control --out <json>`. In the repo itself the tests run in place (`REPO` is derived from the file's location).

## DELIVERABLE MANIFEST (nothing staged, committed or pushed: the brief's local override forbids touching a git tree. **Everything below exists in ONE place: `C:/Users/Admin/par18/p6adh/` on the dev box, single copy until the MM lands it.**)
Paths under `code/` mirror the repo; `PK` as above. Hashes: `MANIFEST.sha256`.

| # | artifact | lands at (repo) | kind | lives now |
|---|---|---|---|---|
| 1 | `command_adherence.py` | repo:`PK/eval/command_adherence.py` | NEW | dev box `p6adh/code/...` only (single copy) |
| 2 | `test_command_adherence.py` (24 checks) | repo:`PK/eval/test_command_adherence.py` | NEW | single copy |
| 3 | `RUNBOOK_P6_READ.md` (edited) + `RUNBOOK_P6_READ.diff` | repo:`PK/raw/2026-10-22-p6-read-gaps/RUNBOOK_P6_READ.md` | MODIFIED (base blob `aac838f6...` = unchanged `a9bc000` -> `a8f2c89`) | single copy; pass `base=<blob>` when landing |
| 4 | `RESULT.md` (this file) | suggested repo:`PK/raw/2026-10-22-p6-adherence/RESULT.md` | NEW | single copy |
| 5 | `raw/`: `control_p2_vs_u400.json` (+ `.out`, `.Etree.json`), `test_command_adherence.out`, `mutation_*.log` + `mutation_summary.txt` + `mutation_run.py`, `probes/` (2 probes + outputs), `make_overlay.sh`, `sync_overlay.sh` | suggested repo:`PK/raw/2026-10-22-p6-adherence/raw/` | evidence | single copy |

Not part of the package (scratch, deletable): `work/` (the mirror-tip overlay, 40 MB, rebuilt by `raw/make_overlay.sh`; `work/probe/probe2.py` is the discarded misaligned probe). Nothing was written to `E:/Projects/TanitAD`, the data dir, the pod, or any git tree; no GPU was used (`CUDA_VISIBLE_DEVICES=-1`).
