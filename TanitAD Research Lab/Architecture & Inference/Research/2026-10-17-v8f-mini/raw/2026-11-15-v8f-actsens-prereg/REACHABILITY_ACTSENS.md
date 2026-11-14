# REACHABILITY_ACTSENS: is each lever reachable from the ladder trainer, and what code makes it so (agent, 2026-11-15)

All line numbers refer to the ladder tree `F:/TanitAD-scratch/ladder_tree`. Its trainer has sha256 `510b1e61…`, which equals `trainer_sha256` in `ladder_R35_s0/gate_token.json` (commit `1faa1a06`). Nothing under `E:/Projects/TanitAD/stack` and nothing in `launch_gate.py` was edited. Probes ran on dev-box CPU with `CUDA_VISIBLE_DEVICES=-1`, `tanitad.__file__` = the ladder tree.

## Verdict
| lever | reachable today? | evidence (MEASURED) | what has to change |
|---|---|---|---|
| IDM weight per level | **NO from the caller; YES at model level** | `--wm-idm-gamma` / `--wm-gamma`: 0 hits in the trainer (control `"--wm-strides"`: 1). The only route is `--wm a\|b`, which gives gamma 0 or 100. At model level, `cfg.wm_gamma` reaches the loss: the IDM-head gradient at 300 / at 100 = 3.0000 at all three levels, gamma 0 reads exactly 0.0, and the predictor gradient is bit-identical across gamma (`PROBE_ACTSENS_RESULT.json` partB) | §1 |
| DIVRS sampler | **NO** | `wm_sampler`: 0 hits in the ladder trainer (control, levers2 `comb` tree: 26). `DiversitySampler`: 0 in the ladder data module (comb: 1). `levers2/diffs/2_divrs.diff` dry-run on a copy: data module and test apply cleanly; **trainer: 3 of 7 hunks FAIL** (at 120 FLAG_CHECKS, 312 argparse, 3111 levers record); 4 apply with offset or fuzz | §2 |

## 1 IDM weight: `--wm-idm-gamma G1 G2 G3`

### Trace of the path the arm will use (flag → loss → backward)
1. argparse, new, after `--wm-strides` (`v8f_mini_train.py:318`): `ap.add_argument("--wm-idm-gamma", dest="wm_idm_gamma", type=float, nargs=3, default=None, metavar=("G1","G2","G3"))`. Absent means (cfg.wm_gamma,) x 3, which is every earlier arm bit for bit. Refused with `--wm a` or `--wm off`, and refused for any G <= 0.
2. `FLAG_CHECKS` (`:70`, beside `"wm_strides": "lever:wm_strides"` at `:123`): add `"wm_idm_gamma": "lever:wm_idm_gamma"`. Without this entry the trainer's own gate refuses the flag (`:1427-1430`: `flags without a declared-vs-built entry (the gate refuses them)`). This in-trainer check is the G-DVB that box-2 WM-only arms run under. `launch_gate.py`'s `v8f` profile is the Thor-side gate for the full trainer.
3. `make_config` (`:402-406`): one line, `wm_gamma_levels=declared_wm_idm_gamma(a)`. It is the single line mutation M1 deletes.
4. `V8FMiniConfig` (`v8f_mini.py:209-210`): a new field `wm_gamma_levels: tuple = ()`, where () means (wm_gamma,) x 3. `wm_gamma` stays 100, so `INIT_CKPT_MATCH` (`v8f_mini_train.py:748`) still matches M2b. The new key is not in `INIT_CKPT_MATCH`, so M2b, which lacks it, loads unchanged (`:832` compares only keys the checkpoint has).
5. `WMAux.__init__` (`v8f_mini.py:687-692`): `mk` takes the gamma per level, and the three `LevelConfig` are built with `gamma=gl[0..2]`, where `gl = tuple(cfg.wm_gamma_levels) or (gamma,)*3`. `self.gamma` keeps the record value, because `built_levers()["wm"]` reads it as b.
6. Loss: `V8FMini.wm_only_loss` (`v8f_mini.py:1182`) → `WMAux.forward_mean` (`:1188`) → `HJEPAStack.loss`, where `hjepa.py:749` computes `level_loss = pred + c.beta*sg + c.gamma*idm` with `c = self.cfg[lv-1]`. This is the trainer loop's call (`v8f_mini_train.py:3372`), followed by `T["total"].backward()`.
7. Read-back: a new `bad_wm_idm_gamma(a, cfg, model)` added to `declared_vs_built` (`:1414-1426`). It reads `[r["gamma"] for r in model.wm.stack.built_levers().values()]` (`hjepa.py:573-583`), which is the LevelConfig the loss multiplies by. It does **not** read `WMAux.gamma`. When the flag is named, `levers["wm_idm_gamma_readback"]` is written, following the pattern of `:3196` (a flag-off selfcheck keeps its key set).

### Why the read-back must read the stack (MEASURED, mutation in `probe_actsens.py` partB)
The mutation: cfg declares 300 while the loss's LevelConfigs keep 100.
* The present trainer check (`v8f_mini_train.py:1500`, `built["wm_idm_gamma"]` = `WMAux.gamma`) **passes**.
* `stack.built_levers()` reads [100, 100, 100], so the proposed reader goes **red**.
* The IDM-head gradient ratio against gamma 100 reads 1.0 instead of 3.0, so the gradient probe goes **red**.

### Gradient proof plan (non-zero against an exact-0.0 control)
* **Model level, run (synthetic batch, R35_s0 config, seed 0):**
  * IDM-head grad abs sum at gamma 100: [6177.09, 2669.62, 2116.82]; at 300: [18531.28, 8008.86, 6350.46]; ratio 2.99999992 / 2.99999982 / 3.00000023.
  * wm a (gamma 0): **exactly 0.0** at all three levels.
  * The predictor gradient is bit-identical at 100, 300 and 0. So the IDM weight never reaches the predictor directly; it acts through the representation (`wm.proj` and the upper encoders). That is the mechanism the card's MECH read tests.
* **Per level, to run once the code exists** (`test_v8f_actsens.py`):
  * `--wm-idm-gamma 1000 100 100`, compared with the record on one batch: the level-1 IDM-head gradient is 10.0x, and the level-2 and level-3 IDM-head gradients are bit-equal.
  * The predictor gradient is bit-equal.
  * `--wm a` reads 0.0.
* **On real data at the gate** (card C3): the unweighted terms equal R35_s{k}'s `selfcheck.json` `live.terms`, and the level/L1 difference equals 900 x idm/L1.
* **Post-training** (card C2): `gamma_from_log.py` recovers the BUILT weight from every logged step through the loss identity. On R35_s0 and R35_s1 it reads 100.0000 (min 99.99998, max 100.00002, 384 steps, all levels). The IDM arm must read 1000 / 100 / 100.

### Mutations that must go red (`test_v8f_actsens.py`, run one at a time on a scratch copy, `--basetemp` on F:)
| id | mutation | must turn red |
|---|---|---|
| M1 | the `make_config` line deleted | `bad_wm_idm_gamma` (declared 1000, built 100) |
| M2 | `WMAux` ignores `wm_gamma_levels` | the stack reader (`WMAux.gamma` alone stays green: MEASURED) |
| M3 | the `FLAG_CHECKS` entry deleted | the trainer gate (`:1428`) |
| M4 | the read-back reads `WMAux.gamma` | the test that runs M2's builder |
| M5 | the default changed (None → (1000, 100, 100)) | C1 PARITY (flag-absent build bit-equal to the ladder tree: state_dict and `live.terms`) |
| M6 | named with `--wm a` | refusal |
| M7 | gamma applied to levels 2/3 too | the per-level gradient test (L2/L3 grads must be bit-equal to the record) |

## 2 DIVRS: port `levers2/diffs/2_divrs.diff` to the ladder tree
* **Applies cleanly** (MEASURED, `git apply --check` / `patch --dry-run` on copies under `F:/TanitAD-scratch/actsens/divrs_apply/`):
  * `stack/tanitad/data/v8f_mini_data.py`: `ClipStore.get_act`, `window_act_means`, `DiversitySampler`, `check_cell_shares`.
  * `stack/tests/test_v8f_divrs.py`: 18 passed on the levers2 tree (`levers2/logs/pytest_divrs_final.log`).
* **Hand-port** (failed on context: the ladder added FLAG_CHECKS, argparse and levers lines since):
  * the `"wm_sampler": "data:wm_sampler"` FLAG_CHECKS entry (`:120` hunk);
  * the `--wm-sampler {uniform,divrs}` argparse (`:312` hunk);
  * the `levers["wm_sampler"]` record (`:3111` hunk).
* **Apply with offset:** the sampler line at the ladder's `:3081` (`D.ChunkedSampler(...)` → `make_sampler(...)`), the `bad_wm_sampler` clause in `declared_vs_built`, and the control-1 probe.
* **Ladder-specific checks, UNVERIFIED until the port runs:**
  * (a) `window_act_means(store, J, wins, cfg.hmax)` at hmax 91 on the `--wm-windows wmonly --wm-pooled on --wm-zcache` path: `store` must be the `ClipStore` holding `get_act`.
  * (b) `bad_wm_sampler` compares the record's `n_windows` with `data_info["n_windows"]`. On the ladder the count also lives under `wm_ladder.windows.n_windows` (40,358 for R35_s0), so the port must name the key it reads.
  * (c) the uniform default must reproduce R35's epoch order bit for bit. This is covered by card C1 and by the diff's own uniform-identity test.
* **Mutations:** the card's 7 (`levers2/mutations_divrs.json`, D1-D7) run **to the end** on the ported tree. Only the baseline and part of D1 ran before (`HANDOFF_LEVERS2.md`: stopped at 0.8 GB free RAM). Rule: free RAM >= 3 GB, one at a time.
* **Gradient / G-LIVE:** a sampler changes which windows feed the loss, not the loss itself, so the proof is distributional. Card control 1 checks the realised cell shares over the first 2,000 batches against the 1/count target. Its deliberate regression (flag set, uniform draw) must FAIL. The existing WM-only G-LIVE (`wm_only_live_check`, `:1018`: every WM group gets a non-zero gradient) runs unchanged.

## 3 Launch-gate needs (no edit made here)
* **Trainer G-DVB:** each new flag needs (1) a `FLAG_CHECKS` entry, (2) a `declared_vs_built` clause that reads what was BUILT, and (3) a selfcheck record when the flag is named. Both levers' defaults reproduce R35 bit for bit (C1).
* **Package gate:** `theory/actsens/box2/actsens_check.py`, from `egowm/box2/egowm_check.py`.
  * NEEDLES add `'"--wm-idm-gamma"'`, `'"--wm-sampler"'`, `"def bad_wm_idm_gamma"`, `"def bad_wm_sampler"`, `"lever:wm_idm_gamma"` and `"data:wm_sampler"`.
  * Selfcheck asserts: `levers.wm_idm_gamma_readback.built == [1000, 100, 100]` for IDM arms and absent otherwise; `levers.wm_sampler.kind == "divrs"` and `wm_sampler_probe.pass` for DIV arms; the frozen argv sha per tag.
  * Post-check: `gamma_from_log.py` and `probe_zerograd3.py`.
* **G-SUITE:** if the dev-box pinned run lists test files, the new `test_v8f_actsens.py` and `test_v8f_divrs.py` enter the pinned list (UNVERIFIED). A launch_gate pin or sha edit moves profile fingerprints, so every launch_gate-reading test runs before landing. That is the Master Mind's lander work.

## 4 Read tools must run on the ACTSENS tree (G-EVAL)
* The new checkpoints carry `cfg["wm_gamma_levels"]`, and arms that name the flag also carry `wm_sampler`. Config dataclasses refuse undeclared attributes (G-HYG), so the old ladder tree may refuse to build them.
* Gamma changes only the loss, so the inference model is the same on either tree. The tools must still run with the ACTSENS tree first on `PYTHONPATH`, asserting `tanitad.__file__`: `run_hwm.py` (via `hwm_data`), `run_ladder_read.py` / `lr_lib.py`, `decode_check.py`, and `collect_latents_train_R35.py` (`WMACT_CODE_ROOT`).
* `probe_actsens.py` filters config keys and does not depend on this.

## Instruments beside this file
* `probe_actsens.py` (sha256 `7e1425da…`): Part A is the dose input and the card's beside mechanism instrument; Part B is the reachability and mutation probe.
* `PROBE_ACTSENS_RESULT.json`: its output. The `levels_built` rows were dropped; the full output is sha256 `b828da7b…`.
* `gamma_from_log.py` (sha256 `ce2468ce…`): the post-training weight identity.
