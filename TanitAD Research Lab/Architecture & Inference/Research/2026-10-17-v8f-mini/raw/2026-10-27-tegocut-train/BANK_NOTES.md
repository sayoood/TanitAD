# T-EGOCUT trainer lever: built, proved, PARKED (2026-10-27)

**Status: PARKED, not launched, not merged into `stack/`.** The draft (`raw/2026-10-26-tegocut-draft/PREREG_V8F_TEGOCUT.md`) maps the
DAgger read's outcomes onto this arm. It runs only under DAgger NO EFFECT or HARMS. `PREREG_V8F_DAGGER` read **ROBUST** on 2026-10-27
(`raw/2026-10-27-dagger-run/read/DAG_READ.json`), so the arm is parked.

The build is banked here as a patch against `b43c7a9a`, with its proofs, so that it can be applied without being rebuilt.

**What it is.** `--t-ego {full,const}` (default `full`).
* Under `const`, T's ego token (`v8f_mini.py` `run_t`, the one swap line) is replaced by a constant c. Here c is the init checkpoint's
  `e_ego.bias`, held as the non-trainable buffer `t_ego_c`.
* The trainer checks c bit for bit against the init file (`bad_t_ego`), and refuses `const` without `--init-ckpt`.
* A gate probe (`t_ego_probe`) runs both branches.

**Proofs, on the final bytes (MEASURED, logs in `logs/`):**

| suite | result |
|---|---|
| new `test_v8f_tegocut.py` | 56 passed |
| `test_v8f_dagger_train.py` / `test_v8f_mini.py` / `test_v8f_ego_perturb.py` | 97 / 29 / 82 passed |
| default-off identity against the pre-change trainer and model (`pre_tegocut_*.ref`); same-stream test | pass |
| four widened additions-only pins (`test_v8f_flat`, `_mixo`, `_oacc`, `_window_start`): their saved-pre-change and 200-step tests, plus 8 `an_old_checkpoint` tests | 22 + 8 passed |
| mutation proof (`mutate_tegocut.py`, 55 mutants: swap line deleted, flag defaulted on, c trainable, c from the live model, the resume check lost, …) | **55 of 55 killed** |
| R77: the delivered files' sha256 before the proof (`R77_SHA256_before.txt`) re-checked after it | *"delivered files changed during the run: none"* |

**Not done, and required before any launch:**
* the four widened suites run whole;
* K-BUILD on the real gate batch, K-DATA and K-TOOL;
* the reader and the chain stages;
* a launch-gate G-DVB entry for `--t-ego`, as the 2026-09-26 rule requires.

**The builder's reported ambiguities** (`HANDOFF.md` §"Draft ambiguities"), still open:
1. Only a const-built model creates the buffer, so a checkpoint whose config lost `t_ego` refuses loudly.
2. The probe's flipped copy carries the buffer, unused.
3. The probe adds gate time that has not been measured.

**Files here:**
* `PATCH_tegocut_vs_b43c7a9a.diff`;
* `banked_tree/` (the changed and new `stack/` files, final bytes);
* `mutate_tegocut.py` and `pre_tegocut_*.ref`;
* `run_mut.cmd`, `logs/`, `R77_SHA256_before.txt`, `HANDOFF.md`.
The mutation scratch (773 MB) stays on E: (`E:/Projects/TanitAD-artifacts/egocut_build/mut/`).
