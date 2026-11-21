# SPEEDSUB package: results of the build (no real-data number exists)

Package: `E:/Projects/TanitAD-artifacts/dagland/speedsub/`. Drafted 2026-10-28 from commit `01f3d34e`. Nothing is staged in git, committed or pushed. **The lander integrates `stage/` and promotes `PREREG_V8F_SPEEDSUB.md`.**

## 1. What was built

| item | where | state |
|---|---|---|
| `PREREG_V8F_SPEEDSUB.md` (draft, both outcomes committed) | package root | written; banned-word scan clean; instrument sha256 `a95f4fb1…` filled in |
| instrument change: modes `ego` and `egoshuf`, subcommands `subevals` / `subread`, `decide_sub`, `controls_sub` | `stage/stack/scripts/v8f_mini_eval_speedread.py` | sha256 `a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca`; 1,027 lines against the base's 787; `MODES`, `evals`, `read`, `decide` untouched |
| tests | `stage/stack/tests/test_v8f_speedsub.py` (39 new) plus the unchanged `test_v8f_speedread.py` (27) | **66 passed** (`logs/pytest_staged_green.txt`) |
| golden reference | `stage/stack/tests/golden/` (pinned original `fd5ce0be…` and `golden_speedread_modes.json`; generator `make_golden_speedread.py`) | golden made from the original, not from the instrument under test |
| families runner | `stage/stack/scripts/families_longlat_sub.py` | parametrised copy of SPEEDREAD's; smoked on SPEEDREAD's dumps |
| run scripts | `run/run_speedsub.sh`, `run/make_tree.sh` | `bash -n` clean; `make_tree.sh` executed (built `F:/TanitAD-scratch/speedsub_tmp/tree`, 477 MB); `run_speedsub.sh` NOT executed |
| mutation harness and result | `tools/mutate_speedsub.py`, `logs/mutation_results.json`, `logs/mutation_run.txt` | 28 mutants |

## 2. Test evidence (all MEASURED in this session, CPU only, `CUDA_VISIBLE_DEVICES=-1`, at most 2 concurrent pytest runs)

| requirement from the brief | evidence | artifact |
|---|---|---|
| `ego` RED on the unmodified instrument, GREEN on the new one | with `SPEEDREAD_MODULE` = the pinned original, **31 of 39 new tests fail**, 8 pass; with the staged instrument 39 of 39 pass. Two of the 8 that pass deserve a name: the hook-level `ego` test (the original's hook accepts any mode string when handed a read-out) and the no-read-out refusal. The driver-level test `test_ego_is_RED_on_the_unmodified_instrument_and_GREEN_on_this_one` is the in-suite form: the original's `cmd_evals` writes seven `speedread` dumps and none for `ego`; the new `cmd_subevals` writes five `speedsub` dumps including `ego` and `egoshuf`. | `logs/pytest_RED_on_original.txt`, `logs/pytest_staged_green.txt` |
| every existing mode bit-identical | (a) same-process differential against the pinned original: hook output for all 7 modes, hook stats, cells, `controls`, `decide`, `read_rows`, `cell_table`, `fit_all` read-outs, `save_readouts` / `load_readouts`, and the four SPEEDREAD subparsers, all equal; (b) stored golden from the original, matched to 1e-6 (exact for stats and strings) | same log |
| mutation pass of at least 8 mutants, all killed | **28 / 28 killed**, 0 survived, 0 not applied, 0 errors; baseline green before and after on the same instrument sha | `logs/mutation_results.json` |

The first mutation run reported 27 of 28 "killed" for the wrong reason (a relative-path usage error, pytest rc 4, counted as a kill). The harness now counts only a test failure (rc 1) as a kill and refuses to start if the unmutated instrument is not green. The numbers above are from the repaired harness.

Other measured checks:
* `logs/real_speedavg_base_check.txt`: the staged hook subclass on the REAL `v8f_mini_eval_speedavg.SpeedAvgHook` (from the SPEEDREAD run tree, via the overlay tree) hands O the `mlp_ego` cells to 4.4e-7, `nonspeed_max_abs_dev` 0.0, `tick_mirror_ok` true.
* The banked `readout.pt` (sha256 `a205609a…`, equal to `fit.json`) loads under the staged instrument; `mlp_ego` is ego-only with a 22-wide `xm`; its output equals the original instrument's scene-path output bit for bit; `pred_mlp_ego` in the banked identity dump is (2399, 5); the registered permutation has 0 fixed points at n = 2,399.
* `logs/smoke/smoke_vs_banked.txt`: `families_longlat_sub.py` run on SPEEDREAD's banked s0 dumps (arm `read`) reproduces the banked `longlat_s0.json` `read − own` rows: 36 numbers, max abs difference 0, controls LL-1..3 true.

## 3. What is infeasible or different from the brief (flagged, none hidden)

1. **"Measured ego-motion history only" is not what `mlp_ego` reads.** It reads, at the tick frame only, the 8 P10 kinematic features, the 5 `nav` channels, the 4-wide speed-limit one-hot with its validity (a sidecar label file, not ego state) and the 4 `frame_ok` flags: 22 dims. No history beyond what the P10 features encode. The PREREG says so (§2) and flags `limit` as UNVERIFIED with the PI.
2. **Nav is optimistic on PhysicalAI** (the ego's own future path supplies it). A PASS could be partly route supply. No ablation can be fitted under "no re-fit"; the PREREG names it as the first step of the PASS follow-up (§6).
3. **F/2 is 0.06035, not 0.0604.** The instrument compares against `F_M / 2.0`.
4. **The separation condition will probably bind before the point bar** (SPEEDREAD's CI half-widths 0.075 and 0.055 against a 0.06035 bar). Stated in §1.
5. **Distance keeping cannot be computed through the harness path** (UNAVAILABLE, n = 0, in all 14 SPEEDREAD dumps). `v8f_dk_read.read_dk` takes arbitrary arms but its CLI is hard-wired to `M1` / `M2b`; an adapter of about 60 lines is a **work item, not built**. The PREREG reports DK as not computed with the reason.
6. **STRATEGIC is not paused** and will read exactly zero by construction (the substitution acts after S and T). It is reported, with that explanation.
7. **No real-data smoke**: the harness has no window cap and may not be changed in this brief. Real-window reachability is covered by controls S7 / S7b / S10 and a stop after seed 0 (`preflight_s0` in the run script).
8. **The T1-kin follow-up cannot be specified yet** (DAG-OACC / DAG-ORC checkpoints and the rollout command were not read). The PREREG's §6 says so.
9. **The arm was chosen after the eval-141 beside table was read.** Disclosed in §0; the selection rule is re-derived from TRAIN fold-0 numbers only and is a control (S13). It is the third read-out examined on these windows.
10. **Control tolerances differ from SPEEDREAD's C5.** SPEEDREAD required `own` to equal the banked value exactly (0.0). Here `own` is compared with SPEEDREAD's banked `own` at 1e-5 m because the thread count may differ (4 against 8). S1 (`identity` equals `own` inside one run) stays exactly 0.0.

## 4. Cost (ESTIMATED from SPEEDREAD's MEASURED stage times)

Decision path 20 to 40 min wall at 4 threads, **about 1.3 to 2.7 CPU-h**; families 25 to 45 min wall, about 1.7 to 3 CPU-h; **total about 3.0 to 5.7 CPU-h**. Derivation in PREREG §8.

## 5. Hand-off (integration, not a request in a README)

**Integration to escalate:** land `stage/` (6 files: the instrument, `families_longlat_sub.py`, `test_v8f_speedsub.py`, `make_golden_speedread.py`, `golden/…` x2; `test_v8f_speedread.py` is a byte-identical copy of the repo blob and is a no-op) and promote `PREREG_V8F_SPEEDSUB.md`; open claim row `H-V8F-SPEEDSUB-1`. Then `bash run/make_tree.sh` (or reuse `F:/TanitAD-scratch/speedsub_tmp/tree`) and launch `run/run_speedsub.sh` DETACHED with `INSTR_SHA` set to the landed instrument's sha256. Re-run the 66 tests on the landed tree first (`CUDA_VISIBLE_DEVICES=-1`).

**Single-location items:** SPEEDREAD's banked `eval_*-speedread-{own,gt}.npz`, `speedread_cells_*-identity.npz` and `fit/readout.pt` exist only on `F:/TanitAD-scratch/speedread/` (not in the repo). If F: is lost, controls S5, S6, S7b and the whole run cannot be reproduced.

**Master Mind, 2026-10-28 (landing):** `families_longlat_sub.py` moved from `stage/stack/scripts/` to `families/` in this package, as SPEEDREAD banked its `families_longlat.py` under `raw/2026-10-28-speedread/families/`. It parses `sys.argv` at import, so the lander's import probe refused it as a `stack/` module. `run/make_tree.sh` copies it from `families/` into the run tree, unchanged (sha256 `f981ea7a…`).
