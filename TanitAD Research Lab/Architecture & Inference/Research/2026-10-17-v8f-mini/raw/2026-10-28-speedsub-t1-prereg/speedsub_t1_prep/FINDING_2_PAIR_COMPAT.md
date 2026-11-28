# FINDING 2: DAG-base pair-compatibility of the banked read-out (PREREG_V8F_SPEEDSUB section 6, infeasible-as-written item 2)

Agent: Sonnet, 2026-10-28. CPU only. Evidence: `F:/TanitAD-scratch/speedsub_t1_prep/compat.json` (sha256 `8e6d3f17…`), script `compat_check.py` (`4206b2fd…`). Reference = the fit's X_s (M2b-mixo s0 / s1, shas `c6a62876…` / `68a0f335…`, both re-hashed and equal to the pins). Candidates = DAG-OACC s0 / s1 (the CTRL arm of DAG-ORC) and, for completeness, DAG-ORC A and REP s0 / s1. Every number below is MEASURED from the checkpoints and run records themselves; line references are to the tip tree (`ea5e3d3b`).

## 0. Verdict

**On the code-space criteria the pair is compatible: every one passes, bitwise where a bitwise test exists. On the letter of SPEEDAVG's `pair_violations` it FAILS on one item (three training-only config keys).** Three further, non-code-space differences decide whether a DAG result means anything (section 3). None is a blocker to writing the PREREG; the second and third change what it must say.

## 1. Where the bases are, and which ones (second location probed)

| arm | checkpoint | sha256 | confirmed at |
|---|---|---|---|
| DAG-OACC s0 (= DAG-ORC CTRL s0) | `E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt` (86,971,029 B) | `67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f` | the dir's `SHA256SUMS.txt`; **second location** `C:/Users/Admin/box2_ship/logs/v8f_dagoacc_r2/ckpt_sha256.json` (`M2b-dagoacc_s0_trained` and `_after_evals`); `dagorc_t1/PINS.txt` `ctrl=` line; `v8f-dagoacc-t1/done_DAGOACC_s0_m10_causal_heads.txt` (`ckpt_sha256_before == _after`); `RESULT_V8F_DAG_ORC.md` §1 ("CTRL `67265ec1…`") |
| DAG-OACC s1 | `…/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt` | `90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333` | same five |
| DAG-ORC A s0 / s1 | `…/v8f-dag-orc-oacc/M2b_s{0,1}_dagoaccorc/ckpt_last.pt` | `380b230d…` / `870dded7…` | `SHA256SUMS.txt`, `RESULT_V8F_DAG_ORC.md` §1 |
| DAG-ORC REP s0 / s1 | `…/v8f-dag-orc-oacc/M2b_s{0,1}_dagoaccorcrep/ckpt_last.pt` | `b03fd6ef…` / `77853e56…` | same |

* ⚠️ **Do not use** `F:/TanitAD-scratch/aside_from_E/v8f-dag-orc-oacc_r2{a_marker,b_queue_edit}_stop/M2b_s{0,1}_dagoaccorc/ckpt_last.pt` (shas `6ea4d6ba…`, `b245c8e7…`): those are the REFUSED r2a / r2b runs, not the final A. The box-2 mirror `box2_ship/logs/v8f_dagoacc/runs/M2b_s*` holds only `summary.json`, no checkpoint.
* The MM's state note names "DAG-OACC CTRL + REP": REP here is the DAG-ORC replicate (seeds 10 / 11), p = 0. Both are covered above.

## 2. The checks (reference X_s vs each candidate; all 6 candidates give identical results unless a row says otherwise)

| check (what SPEEDSUB section 6 / SPEEDAVG asks) | result | evidence (file:key or file:line) |
|---|---|---|
| **T's `--kin-mode`** | **PASS** (`yaw` everywhere) | ckpt `cfg.kin_mode` = `yaw` on all 8 checkpoints (`compat.json` `ref.<s>.kin_mode`, `cand.<n>.kin_mode`); run record `summary.json levers.kin_mode` = `yaw`; `argv.json` `--kin-mode yaw`; eval and T1 both force yaw: `v8f_mini_eval_speedavg.py:221`, `v8f_t1_eval.py` `cmd_run` `:508` ("REFUSED … a `--kin-mode steer` checkpoint") |
| **cell standardisation `cons_mu` / `cons_sd`** | **PASS, bitwise** | `ck["model"]["cons_mu"]` and `["cons_sd"]` (26 wide) `torch.equal` to X_s0's AND X_s1's on all 6 candidates; max abs deviation 0.0 (`compat.json` `cons_*_bitwise_equal_ref_s0/_s1`, `*_max_abs_dev_vs_ref_same_seed`). The five speed cells sit at cons index 12, 22, 23, 24, 25: mu 11.0331, 10.8181, 11.2585, 9.5276, 12.1352; sd 7.8149, 7.7219, 7.7265, 7.9767, 7.4464. ⚠️ These are NOT `fit.json`'s `const` (10.71, 10.58, 10.83, 9.24, 11.99, the train mean the skill is measured against): two different constants, easy to confuse. |
| …and it would not matter if they differed | note | the hook standardises the read-out's physical m/s with the CHECKPOINT it is evaluating: `v8f_mini_eval_speedread.py:484` (`mu, sd = model.cons_mu[self.ci]…`), `:445` `phys_to_std`, `:464` |
| **O's input layout for the five cells** | **PASS** | code = 22 goal + 5 lat + 7 lon + 26 cons = 60 (`v8f_mini.py:44-45`); cons cells enter `code` at `34 + CI` = **[46, 56, 57, 58, 59]** (`speed_idx` at `v8f_mini_eval_speedavg.py:71-80` returns it for the tip module, cross-checked against `CODE_KINDS['speed']` and `code_group_index('speed')`); O embeds the whole 60-vector with `e_code = nn.Linear(CODE_DIM, d)` (`v8f_mini.py:619`, used at `:704`). `v8f_mini.py` is **byte-identical** (sha256 `8ec5e190…`) at the tip, at `aeb54468` (DAG-OACC's training commit), at `8c24823f` (DAG-ORC's) and in the pinned T1 tree, so the indices cannot differ. State-dict keys and shapes equal X_s's on all candidates. |
| `pool4x10` | **PASS** | `ck["grid"]` = `pool4x10` on all 8; `hist` = 4 on all (the eval refuses otherwise, `v8f_mini_eval_speedread.py` `cmd_evals` "REFUSED: X_s reads hist"). ⚠️ `pool4x10` is the scene-token grid, not a layout of the five cells; the ego-only read-out never reads the scene, so it is not a constraint on `mlp_ego`. |
| **corpus parity** (fit used X_s's 4,512 train clips; eval-141 must stay outside) | **PASS** | every candidate's `summary.json`: `n_clips` 4512, `n_windows` 306566, `excluded_sha256` `f6a1dee0…`, `train_sha256` `37f4f9ee…`, `exclude` `540:600`, all equal to X_s's and to `fit.json` `extract.parity`; `fit.json` `eval_train_disjoint` true |
| ages | **PASS** | `periodic` schedule at m = 10 gives ticks at steps 0, 10, 20: `a_t = step % 3` = 0, 1, 2, `a_s` = 0, 4, 2 (`v8f_t1_eval.py:136-143`); the read-out was fitted on `a_t = i % 3` (`fit.json extract.ages`), so the tick ages are inside its support. `mlp_ego` does not read `a_s`. |
| **SPEEDAVG `pair_violations(X_s, DAG-OACC_s)` (literal)** | **FAIL, 1 item** | returns `["the two checkpoints' configs differ in ['code_mix', 'p_gt_start', 'w_acc']"]` on all 6 candidates; every other test passes (grid, state-dict layout, `cons_mu`/`cons_sd` bitwise, weights differ). `built_cfg` diff: `code_mix` kind -> whole, `p_gt_start` 1.0 -> 0.2, `w_acc` 0.0 -> 49.6969. |
| …are those three keys inference-relevant? | **NO** | `code_mix`: used only under `use_labels and p_gt > 0` (`v8f_mini.py:751-752`; `forward_infer` calls `_forward(g, 0.0, False)`, `:773-781`). `w_acc`: loss term only (`:888`). `p_gt_start`: the training schedule (`:173`, `:927`). The sanity row: `pair_violations(X_s0, X_s1)` = `[]` (`compat.json ref_pair_violations`), so the function itself passes on its design case. |

Read plainly: `pair_violations` was written for two seeds of ONE recipe; it correctly says these are different recipes. It does not say the code spaces differ, and they do not.

## 3. What is NOT compatible (these are not in `pair_violations`; the follow-up PREREG must address them)

1. **The DAG bases are not M2b-mixo.** DAG-OACC_s is a 3,000-step fine-tune (lr 1e-4) of **M2b_s** (not M2b-mixo), `--init-ckpt …\M2b_s0\ckpt_last.pt` sha `601cb435…` (s1: `6104fb15…`), with a DAgger swap `--dagger-p 0.1`, `--p-gt-start/-end 0.2`, `--w-acc 49.6969` (`argv.json`, `summary.json levers.init_ckpt`). T's speed cells, the thing being replaced, and O's response to them are those of a different, rollout-trained model. SPEEDSUB's PASS (M2b-mixo, T0 grid) does not transfer by itself; the follow-up needs its own `own`/`gt`/`ego`/`egoshuf` rows on the DAG base.
2. **The ego block the read-out sees changes inside a T1-kin rollout.** At T0 the read-out reads the LOGGED measured ego state. At T1-kin `p10f` is rebuilt from the model's OWN executed track (`WindowState._finish`, `v8f_t1_eval.py:201-221`, via `fk.kin_features`), `nav` is dead-reckoned (route progress from own arc, token/valid held from t0, `:216-220`), and **`limit` is the causal bucket** (`--limit causal`, `build_item` `:237-258`, `causal_bucket` `:261-272`). The read-out was fitted on logged values of all three. See finding 1 section 3: `limit` is its strongest channel and, in the logged lane, a future-speed oracle; in the causal lane it is a backward-looking function of the ego's own speed.
3. **The follow-up PREREG must state which `limit` lane.** `--limit logged` (the K4 lane, `done_*_logged_heads.txt`; exists for every DAG-ORC arm) keeps the read-out in the regime it was fitted on, with the oracle open; `--limit causal` is the decisive lane of the DAG-OACC / DAG-ORC chains (the registered T1-kin) and puts the read-out out of distribution on its strongest input. Neither is wrong; choosing silently would be.

## 4. Not checked, named

* That SPEEDSUB's T0 evals and the T1 rollouts score the same 2,399 windows in the same order (both report 2,399 / 141 episodes; the T1 grid file md5 is `34ed8ac56365eeff72e42581bde2f5c4`; I did not diff the window lists).
* `p10_prior.pt`: `p10f` is `kin_features` of the track (no prior in it), but the T0 eval passes `--p10` for the floor; not compared.
* Behavioural compatibility (does O on a DAG base respond to the cells as M2b-mixo's O does) cannot be read without running evals, which was out of scope.
