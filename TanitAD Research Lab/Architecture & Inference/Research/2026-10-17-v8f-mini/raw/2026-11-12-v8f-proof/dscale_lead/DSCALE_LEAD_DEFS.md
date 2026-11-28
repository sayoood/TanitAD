# DSCALE_LEAD_DEFS: what the kit of card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD) with note N1 contains

Tier: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance. Folder: `F:/TanitAD-scratch/v8proof/dscale_lead/`.
Nothing here was executed by its builders except `bash -n` and the `DRY=1` runs of the kit script and `ast.parse` syntax checks (see the last section).

## 1. Files

| file | what it is |
|---|---|
| `make_tok_probe_u.py` | generator: 12 asserted text edits of `../finegrid_r/tok_probe_gr.py` (each old text exactly once) -> `tok_probe_u.py`; the source is never edited |
| `tok_probe_u.py` | the ported probe tool (generated, never hand edited; `test_tok_probe_u.py` compares it with the generator output byte for byte) |
| `dscale_precheck.py` | numpy only: counts, d1 (+ `--plant`), d2, d2j, d2n and d2g (new clips, added by the lead), sha256 of every input -> `PRECHECK.json` |
| `pod_dscale_lead_run.sh` | the kit: stages precheck, smoke, run, extract, checklog; `DRY=1` prints and runs nothing |
| `TREE_PINS.txt` | sha256 pins of the three tree files and `tok_probe_u.py` (placeholders until the lead fills them) |
| `dscale_lead_read.py` | the reader: controls first, names only if every control passes |
| `test_tok_probe_u.py` | torch, CPU, `--tree`: the tool's levers on a synthetic world (pod) |
| `test_dscale_precheck.py` | numpy only: the precheck on tiny npz files |
| `test_dscale_lead_read.py` | numpy only (needs the pilot reader chain's paths): the reader and the kit script under `DRY=1` |
| `MUTATIONS_U.json`, `run_mutations_u.py` | the list of 20 mutations and the runner; result file `MUTATIONS_U_RESULT.json` |

## 2. Edits to the source tool (all default-off; without the flags the training path is the source tool's, tested bit for bit)

* `--clip-list FILE`: keep only the training clips whose id is in FILE; windows rebuilt with the trainer's own `D.train_windows`; refuses a listed id that was not built, an empty or duplicated list, and a missing `--steps`. Meta records `n_listed`, `n_kept` and the sha256 of the file.
* `--eval-guard FILE` (control d1 at training time): refuses when an id of FILE is among the clips the trainer built; meta records `n_eval_ids`, `n_shared`.
* `--w-clos 1.0|0.0`: 0.0 sets the closing label of every batch to NaN, so the closing part of the lead term is exactly 0.0 with exactly 0.0 gradient (note N1: the union lead file has no closing speed). Other values refused.
* `--shuffle-gap-labels`: permute the finite gap labels among the rows that have one (fixed rng 12345); presence labels and counts stay; needs `--w-clos 0.0`; excludes `--shuffle-closing-labels`.
* `precheck ...`: delegates to `dscale_precheck.py`. `check_log`: a gap-permuted arm (header key `shuffle_gap`) is read on the presence part only.

## 3. Arm map and matched compute

| card arm | folder / arm | steps | flags beyond the common ones |
|---|---|---|---|
| `P16C_sK` K = 0..3 | `runC/TK16_sK` | 19,164 | `--clip-list BSET` |
| `P16U_sK` K = 0..3 | `runU/TK16_sK` | 19,164 | none |
| `P16U2_s0` (beside, names nothing) | `runU2/TK16_s0` | 38,328 | none |
| `P16CSH_s0` (control, d5) | `runCSH/TK16_s0` | 19,164 | `--clip-list BSET --shuffle-gap-labels` |
| `P16USH_s0` (control, d5) | `runUSH/TK16_s0` | 19,164 | `--shuffle-gap-labels` |

Common: grid pool4x10, width 16, `--w-clos 0.0`, `--eval-guard`, `--expect-steps` equal to `--steps`. The tool's own step count is `ceil(n_windows_all_clips / batch) x steps-mult`, which depends on the clip set; the kit therefore passes
`--steps` (and `--expect-steps`) to every folder, so total steps and `LRSchedule(lr, warmup 200, total)` are identical for corpus and union arms. No `--exclude` in the trainer argv: the union already lacks the 60 holdout clips and `--exclude 540:600` would drop 60 other union clips.

## 4. Stages of the kit (`bash pod_dscale_lead_run.sh <stage>`; markers in `$OUT/markers/`; STOP and DONE files; flock on `/root/gpu_queue.lock`; refuses under 20 GB free container memory, cgroup v1)

1. `precheck` (CPU): pins, input sha256 prefixes, `dscale_precheck.py`, then the planted-id run which must exit 2 -> `$OUT/precheck/PRECHECK.json`, `PRECHECK_PLANTED.json`.
2. `smoke`: 200 steps of the four arms of runC on the union; projects the GPU hours of all five folders; exit 4 above `MAX_H` 6 h.
3. `run`: the five folders in sequence, one GPU process each.
4. `extract`: evaluation split only (`--eval-range 0:141`), per folder -> `$OUT/states/<folder>/`.
5. `checklog`: the train-log check per folder, then `DSCALE_LEAD_DONE.txt`.
Reader call (after extract and checklog): `python dscale_lead_read.py --in-dir $OUT/states --run-dir $OUT --out-dir <NEW> [--done-marker $OUT/DSCALE_LEAD_DONE.txt]` from the pod mirror folder where the pilot reader chain's fixed paths resolve. Exit 0 names, 2 a control failed, 3 an input is missing.

## 5. Controls and where each lives

| control | where it is computed | gate |
|---|---|---|
| d1 no leak (ids) | `dscale_precheck.py` (`d1_no_leak`, and `PRECHECK_PLANTED.json` must show it FAILED); `--eval-guard` in training (meta `eval_guard.n_shared` = 0, read by reader `p5_u`) | reader `precheck_controls`, `d4_train_log` |
| d2 labels on one clock | precheck `d2_labels_one_clock` (200 corpus clips, seed 45, tol 1e-4, flag and gap, rows in one file only = 0); the same-clock result of the tree owner session is recorded by sha256 (`--clock-check-json`), not recomputed | reader `d2_labels_one_clock_precheck` |
| d2j join = lead file | precheck `d2j_join_matches_lead_file` (k = 2 j) | `d2j_join_labels_precheck` |
| new clips carry labels in the join | precheck `d2n_join_matches_lead_file_new_clips` (at least 200 new clips) and `d2g_join_labels_per_group` (presence and gap-label frames above 0 in both groups) | `d2n_new_clips_join_matches_lead_file`, `d2g_new_clips_carry_gap_labels`; absent record = FAIL |
| d3 corpus arm is a live gap probe | reader: `P16C_s0`, `P16C_s1`: gap R2 interval lower end above 0 and presence AUROC at least 0.85; row 97 range 0.513 to 0.791 printed beside, gates nothing | `d3_corpus_arm_live\|...` |
| d4 training is live | reader `p5_u` on every arm: last 1,000 steps below first 1,000 (control arms on the presence part), gap frames above 0 at every step, run complete, header: `total` = expected, `w_clos` 0.0, `shuffle_gap` iff permuted arm, eval guard recorded with 0 shared, clip list kept count 4,512 for corpus arms and absent for union arms | `d4_train_log\|...` |
| d5 negative control | reader: both permuted-gap arms: gap R2 interval lower end at most 0 and presence AUROC at least 0.80 | `d5_permuted_gap_dead\|...` |
| d6 same order | stated as a LIMIT, not a pass: the windows of corpus and union arms differ; both `r1_batches.json` sampler chunks printed; gates nothing | `limits.d6` |
| d7 reader controls | `TKR.paired_read_controls` (self reads exactly 0, swap reads the negative, planted shift read back) and `control_d7` (two-level: gain at every seed separated, one of four not, identical pairs exactly 0 with [0, 0], swapped order the negative) | `d7_...`, `paired_*` |
| records, pack | probe form of every eval state record (grid, 40 tokens, width 16, ln, step, arm), pack counts, head forward and no-future-input controls of the pilot reader | `records_probe_form\|...`, `a1_*`, `k2_*`, `k4_*` |

Every control gates except `k3_*` (closing controls: no closing read) and d6. The names (card section 4 with note N1): `LEAD-READ-SCALES-GAP` (two-level interval of S = U minus C on M2 above 0, at least 3 of 4 seeds above 0, mean gap error of U at most 0.90 x that of C),
`LEAD-READ-WORSE-AT-SCALE` (interval below 0), else `LEAD-READ-FLAT-AT-SCALE`. No closing name. `P16U2_s0` is printed beside and names nothing. Every number carries the tier line T0; the verdict prints the four family lines (LONGITUDINAL, LATERAL, TACTICAL not measured; STRATEGIC NOT COMPUTED, n = 0).

## 6. Fill-in list for the lead (nothing below is known to the kit)

1. `TREE` (the shipped study tree, folder holding `stack/`), `OUT` (a NEW folder), `P10` (p10 prior `.pt` for the union clips), `TRACKS` (native tracks `.pt` that cover the 10,000 new clips).
2. `EVAL_IDS` (optional: the 48 evaluation episodes' clip ids; the eval cache manifest is always used) and `CLOCK_CHECK_JSON` (result file of the tree owner session's same-clock check).
3. `TREE_PINS.txt`: the sha256 of the three tree files and of `tok_probe_u.py` (the pins file refuses any placeholder; at least 4 pins). Regenerate `tok_probe_u.py` only through `make_tok_probe_u.py`.
4. Cache layout: the trainer reads `<CACHE>/train/manifest.jsonl` and `<CACHE>/train/clips/`; the brief says `manifest.jsonl` and `clips/` at the cache root: confirm which holds on the pod.
5. Eval label source: `eval_pack.npz` of card ADD43 at the reader's `--pack-dir` (default `TKR.READ`), and the pilot reader chain's fixed paths (`FINEGRID_BASE_DIR`, the tokprobe folder) on the pod mirror.
6. Check that the precheck's d2n has at least 200 new clips with rows in the union lead file, else the control fails by design.
7. Gate profile and token of the launch gate for the new tool, if the programme's launch-gate rule applies to a probe-only run (not decided here).

## 7. Never executed (by the builders)

`tok_probe_u.py` itself (only `ast.parse`); `test_tok_probe_u.py`, `test_dscale_precheck.py`, `test_dscale_lead_read.py` (not run, not collected); `run_mutations_u.py` and every one of the 20 mutations (the old texts were counted by `grep -c` for four of them only; the runner itself checks all); `dscale_lead_read.py` end to end
(no pack, no states, no real or synthetic run); `dscale_precheck.py` on any file; the kit stages other than as `DRY=1` text; the smoke, the run, the extract; the generator re-run (the generated file was checked once by its author with `ast.parse`).
Facts taken from source reading and not executed: the key names `lead_sup_clos` of `lead_sup_terms`, `pres`/`gap`/`clos` of `lead_sup_labels`, the header keys of `train_log.jsonl` (the meta dict is passed as the log header), the `r2_ci95` key of `R["M2"]["head"]`, and that `step` and `arm` sit in the eval state record.

## 8. Additions by the lead with note N2 (2026-11-19; `HANDOVER_KIT.md` is the current run description)

* Kit stage `cpucheck` (CPU only, no GPU lock) between precheck and smoke; the smoke requires it.
* Cache layout: `resolve_cache` uses `<CACHE>/train` if it holds a manifest, else builds `$OUT/cache_view` with two links (train, eval); nothing is copied.
* Control d2c: `dscale_clockcheck.py` (ego speed of the lead file against the ego speed of the token cache at shifted pairings; planted shift must exit 2); reader gate `d2c_lead_rows_on_the_token_clock`, `d2c_planted_shift_fails`.
* Control d2e and report d3t: `dscale_tokens_same.py` with `STUDY_EVAL_MD5.tsv`, `STUDY_TRAIN_MD5.tsv` (planted swap must exit 2); reader gate `d2e_eval_set_is_the_study_set`, `d2e_planted_swap_fails`; d3t printed under limits.
* `TREE_PINS.txt` filled: three tree files and six kit files (the kit asks for 9 pins). `EXPECT_UNION`, `EXPECT_CORPUS` are passed to the precheck (defaults 14,512 and 4,512).
* Tests: `test_dscale_lead_read.py` has two more tests (24 in all); `MUTATIONS_U.json` has two more mutations (22 in all). `pytest_shim/` is a stand-in runner for a pod without pytest; it never deletes a folder and says in its last line that it is not pytest.
* Executed since section 7 was written (pod of this fork, CPU only, stand-in data): the three test files, the mutation list, the precheck, the clock check, the evaluation-set check, the kit stages precheck and cpucheck, 20 training steps from the shipped tree. Still not executed: the reader end to end, any GPU stage, anything on the union files.
