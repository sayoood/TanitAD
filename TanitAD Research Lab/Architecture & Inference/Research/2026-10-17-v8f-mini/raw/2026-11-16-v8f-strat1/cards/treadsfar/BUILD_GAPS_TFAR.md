# BUILD_GAPS_TFAR: what a Sonnet build agent must build for PREREG_V8F_TFAR

All line numbers are at **`aedf8716`** (branch `agent/v8f-ckpt120-20261117`, CKPT120 fix3, the commit that trained H2_s0 / H2_s1 / R2). They come from `git show aedf8716:<path>` in the C: mirror; check them again on the branch tip before patching. Nothing here is built. It is staged for a later agent, and the build starts on the PI's go (Q1-Q3).

## 1. Branch: build from `aedf8716`, not from `4dedac45` or `f95de67a`

| commit | what it is | CKPT120 levers present? |
|---|---|---|
| `aedf8716` | CKPT120 fix3 on `a4357e79` (20 files vs `a4357e79`) | yes: `grep -c s_far_out stack/tanitad/models/v8f_mini.py` = 15 |
| `4dedac45` | STRAT1 S1d checker kind on `a4357e79` | no (0). It touches only `stack/scripts/strat1_check.py` and `stack/tests/test_strat1_s1d.py`, neither of which `aedf8716` touches, so a merge into the build branch is conflict-free (M, `git diff --stat`). Merge it only if the DGOAL D0 rung needs the S1d kind. |
| `f95de67a` | NAVRB2 roundabout channel on `4dedac45` (14 files) | no (0). It overlaps CKPT120 in `v8f_mini.py` (31 lines), `v8f_mini_train.py` (142) and `v8f_mini_data.py` (31). **Do not merge** for TFAR; a NAVRB2 rebase is its own card (TFAR rebase rule (b)). |

New branch `agent/v8f-tfar-<date>` from `aedf8716`. Never commit to `main`; stage only, and the MM commits.

## 2. Model: `stack/tanitad/models/v8f_mini.py`

| piece | attach at | what |
|---|---|---|
| mode tuples, key prefixes | after `:177-184` (`S_ADAPTER_DETACH_MODES`, `S_FAR_KEY_PREFIXES`, `NAV_S_KEY_PREFIXES`, `NEW_LEVER_KEY_PREFIXES` :184) | `T_FAR_MODES`, `T_FAR_INPUTS = ("pred", "zero")`, `T_FAR_VARIANTS = ("real", "roll")`, `T_FAR_TF = ("same", "off")`, `NAV_STRAT_T_MODES`; `T_FAR_KEY_PREFIXES = ("e_ck_far.",)`, `NAV_S_T_KEY_PREFIXES = ("e_nav_s_t.",)`, both joined to `NEW_LEVER_KEY_PREFIXES` |
| config fields | after `:284` (`s_adapter_detach`), validation in the loop at `:326` | `t_far="off"`, `t_far_input="pred"`, `t_far_variant="real"`, `t_far_tf="same"`, `nav_strat_t="off"` (G-HYG: declared and validated) |
| modules | after `:934-938` (`e_nav_s`), built LAST | `e_ck_far = nn.Linear(4, d)` and `e_nav_s_t = nn.Linear(5, d)`, both `zeros_` on weight and bias (no RNG draw, so the global stream is H2's) |
| T's token | `run_t` `:1046-1063`; the checkpoint token at `:1051` | new kwargs `far=None, far_ok=None`: `ck_tok = self._ck_cond(ck_t, ck_ok)` (+ `self.e_ck_far(cat([far[..., :2] / 50, far[..., 2:3], far_ok[..., None]], -1))` when built). The nav token at `:1049`: + `self.e_nav_s_t(self._at(b["nav_s"], i))` when `nav_strat_t` on. ⛔ `run_o` (`:1064`, its `_ck_cond` at `:1077`) is NOT touched |
| teacher forcing, one draw | `_forward` `:1137-1141` | refactor `use = (torch.rand(B, …) < p_gt)[:, None] & …` into `u = torch.rand(B, …) < p_gt` (ONE draw), then `use = u[:, None] & rc_ok & has`, and `use_far = u & lab_s["rc120_ok"] & lab_s["has"]` (only when `t_far_tf == "same"`). `far_src = where(use_far, rc120, ckpt_far)`. Detach under `detach_upper` (`:1140`) |
| frame | after `:1142` (`ck_for_t = transform_ckpt(…)`; `transform_ckpt` `:368` takes `[B, M, 3]`) | `far_for_t = transform_ckpt(far_src[:, None], pose_s, pose_t)[:, 0]` |
| variants | same place | `zero`: features and ok are 0. `roll` (train only, `use_labels` True): `torch.roll(far_for_t, 1, 0)` and ok likewise. An eval-only override for donor / zero / ORACLE as a MODEL ATTRIBUTE (e.g. `self._t_far_eval`), never a batch key (K-KEYS) |
| call | `:1144-1148` | pass `far=far_for_t, far_ok=…` to both `run_t` calls |
| param groups | `:1413-1416` (`g["SFAR"]`, `g["NAVS"]`) | `g["TFAR"] = [self.e_ck_far]`, `g["NAVST"] = [self.e_nav_s_t]` |
| built levers | `built_levers` `:1419`, CKPT120 entries `:1471-1482` | `lv["t_far"]`, `t_far_input`, `t_far_variant`, `t_far_tf`, `t_far_zero_init`, `lv["nav_strat_t"]` |
| guards | `assert_no_situation_classifier_input` `:1543`; the CKPT120 `nav_s` key guard (T / O never read `nav_s`) | unchanged read set (K-KEYS). Relax the T-side `nav_s` guard ONLY when `nav_strat_t` is on; O stays guarded |

## 3. Trainer: `stack/scripts/v8f_mini_train.py`

| piece | attach at | what |
|---|---|---|
| flags | after `:220-236` (`--s-far` … `--s-adapter-detach`) | `--t-far`, `--t-far-lr-mult`, `--t-far-input`, `--t-far-variant`, `--t-far-tf`, `--nav-strat-t`, `--nav-strat-t-lr-mult` |
| G-DVB map | the lever map, CKPT120 entries at `:122-125` | one entry per new flag (`lever:` / `opt:`); a flag without an entry is refused by the gate itself |
| config | `make_config` `:469` | copy the five fields |
| fresh keys | `:857` (`fresh_ok`), `:877` (CKPT120 flag list) | the new prefixes load fresh from an M2b init |
| refusals | new `bad_tfar(a, model, opt)` beside `bad_sfar` (`:1776`), called at `:1891` | `--t-far on` with `--s-far off`; without `--detach-upper`; a non-default input / variant / tf with the layer off; lr-mult not what the group holds; `--nav-strat-t on` without `--nav-strat on` |
| optimiser groups | beside `:3881-3882` (`sfar` / `sfar_nd`); `want_lrs` `:1974` | `tfar` / `tfar_nd` and `navst` / `navst_nd` at `lr x mult` |
| G-LIVE probe | new `tfar_probe` beside `ckpt120_probe` (`:2490`), called near `:3999`, recorded at `:4055` | step-0 grad non-zero on `e_ck_far.weight` / `e_nav_s_t.weight`; **exactly 0.0** on `e_ck_far.weight` under input zero (bias non-zero); **exactly 0.0** on `s_far_out.*`, `head_s.*`, `read_s.*`, `e_nav_s.*` from T's loss terms alone (T-ISO-UP); step-0 outputs bit-identical to the layer-off model (K-REACH-a2); one `torch.rand(B)` draw (K-RNG) |
| DAgger probe | `dagger_probe` `:3223` | no new batch key, so expect no change; prove it with the fix3 test pack (`test_v8f_ckpt120_dagger_nav_s.py`) on a TF argv |

## 4. Gate, reach and replays

| piece | file | what |
|---|---|---|
| study checker + token | new `stack/scripts/tfar_check.py`, modelled on `ckpt120_check.py` (`ARMS` / `SPEC` `:49-52`, `render` `:72`, subcommands `:482-489`) | kinds TF, R3, TZ, TN (+ TFc, H2c, TFnt, TFa, TFd for branches); seeds s0, s1, rep, s10, s11; base argv = the H2 render; mints `tfar-gate-token-v1` bound to commit, argv sha256 and data sha256s (RC-A120 `0f122570…`, nav_s `0f7431a1…`); `post` = moved identity (e_ck_far moved, S groups moved as in H2) |
| reach | `stack/scripts/ckpt120_reach.py`, `PREFIXES` `:37` | add `"t_far": "e_ck_far."`, `"nav_s_t": "e_nav_s_t."`, with the planted HEAD-DEAD controls (delete the Adam state of `e_ck_far.weight`, which must read HEAD-DEAD) |
| replays | `g_box_overfit._build_model`, `refcv7_loader` | mirror the new attachment, or G-EVAL / box binds refuse (memory: replays must mirror the trainer build) |
| launch_gate.py | unchanged | the study-token route needs no pin edit. If any pin moves, run all ~69 launch_gate-reading tests (memory) |

## 5. Evaluators and reader

| piece | file | what |
|---|---|---|
| T0 | new wrapper `stack/scripts/v8f_mini_eval_tfar.py` on `v8f_mini_eval_far.py` | four far-token variants: real / donor (`donor_indices` seed 0, CKPT120's map) / zero / ORACLE (RC-A120 in T's frame where valid, flagged) via the model attribute; dumps T's lon / lat logits per variant |
| T1-kin | new wrapper `stack/scripts/v8f_t1_eval_tfar.py` on `v8f_t1_eval_far.py` (the far point is already held and transformed per tick, `:149-150`) | records `t_far_in` (what T read per tick) and `nav_s` dead-reckoning for TN. ⛔ Do not edit the shared `v8f_t1_eval.py`: it is sha-pinned (memory); add flags in the wrapper |
| reader | new `tfar_read.py` beside the landed CKPT120 reader (`F:/TanitAD-scratch/strat1/read/ckpt120/c120_read.py`, landed `6cfcdf6b`) | the FB stratum from labels (K-LABEL 202 / 48), D-LON-1/2 and D-TAC-1/2, T-READS, ORACLE rows, R3 / TZ must-fail rules, floors (TF_rep and the banked H1_rep), the 24-cell table, the STRATEGIC cells reused from c120, the echo share, the 4-seed rule |

## 6. Tests (about 25, CPU, `CUDA_VISIBLE_DEVICES=-1`)

* **Bit identity.** K-REACH-a (layer off = `aedf8716` H2, keys and outputs) and K-REACH-a2 (step 0 with the layer on). Put a new snapshot beside `v8f_mini_tip_snapshot_ckpt120.txt`.
* **Gradients.** K-REACH-b (non-zero, plus the exact-zero control); T-ISO-UP (exact 0.0 on S from T's loss); T-ISO-O (O's checkpoint token unchanged under a far perturbation).
* **Read set and timing.** K-KEYS (no new batch key; `rc120` never read in `forward_infer`); G-CLOCK-TFAR (planted poses; the skipped-transform mutation FAILS); K-RNG (one draw; a second draw FAILS).
* **Variants.** Roll, zero, and the eval override, which is never a batch key.
* **Gate refusals.** One per refusal in `bad_tfar`; one tfar_check render per kind; reach with planted HEAD-DEAD.
* **K-REACH-c mutations.** Delete the `+ e_ck_far(...)` term; feed RC-A80 in place of the far point; skip the transform; draw a second Bernoulli. Each must turn its named test RED.
* **DAgger.** The fix3 pack, on a TF argv.
* Run serially, with TMP on a per-test directory that is deleted afterwards (memory: REFe tests leak temp dirs). Run `test_text_encoding_is_explicit` before landing (memory).

## 7. Pod kit (after the go)

* Pod tree = the `aedf8716` tree + the changed files, verified by sha256 list.
* Run the G-SUITE in two waves with the memory guard (container cap 50 GB).
* Chain under `flock /root/gpu_queue.lock`, in this order: H2_s10, H2_s11 (if not run), TF_s0, TF_s1, TF_rep, R3_s0, TZ_s0, TN_s0.
* Per arm: STOP check → GPU idle → mint → run → post → reach → `ZZ ARM <tag> DONE ZZ`.
* Bank each arm to HF private, with hub sha256 verified, before the read.
