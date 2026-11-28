# REACHABILITY_ACTRSP (agent 2026-11-16): how `--wm-rsp-w` must reach the loss, and how a build proves it does

Line numbers refer to the ACTSENS commit `81cee6c1760068cf76c6f0465d1fffd9bc1d7215` (tip of `origin/agent/v8f-actsens-20261115`, read from `C:/Users/Admin/tanitad-push` with `git show`), the commit every ACTSENS arm trained on. Files: `stack/scripts/v8f_mini_train.py` (sha256 `f20e345f…`, 3,589 lines) = **TR**, `stack/tanitad/models/v8f_mini.py` = **VM**, `stack/tanitad/models/hjepa.py` = **HJ**. Nothing was edited; no code exists yet. This is the plan the build follows.

⚠️ **Not the `agent/arch-inf-20260803` tip.** At `8a716bb9` the same trainer path has 0 hits for `wm-only` (same-breath control: 92 hits for `add_argument`), so the WM-only recipe does not exist there (MEASURED, `git grep -c`).

## 1 The hop table (argv → namespace → config → built module → loss term → backward)

| hop | where today | what the build adds | mutation that must go RED |
|---|---|---|---|
| argv | TR:332 (`--wm-idm-gamma`, the pattern) | `--wm-rsp-w` float, **default 0.0**, help names PREREG_V8F_ACTRSP and "needs --wm b, --kin-mode yaw, --wm-only" | MR2: default 0.0 → 1.0 (C1 parity red) |
| FLAG_CHECKS | TR:70, entry pattern TR:126 | `"wm_rsp_w": "lever:wm_rsp_w"`. Without it TR:1540 refuses the launch (`flags without a declared-vs-built entry`) | (the existing TR:1540 check) |
| config | TR:424 (`make_config`, the ONE line per lever) | `wm_rsp_w=float(a.wm_rsp_w)` | MR1: delete the line (G-DVB red: declared 1.0, built 0.0) |
| config field | VM:211 (`wm_gamma_levels`, the pattern) | `wm_rsp_w: float = 0.0`; `__post_init__` (VM:243, validation pattern VM:281-291): finite and ≥ 0; > 0 needs `wm == "b"` and `kin_mode == "yaw"` | a config with `wm a` and 1.0 must raise (unit test) |
| built module | VM:700 (level 1's `LevelConfig` in `WMAux.__init__`) | pass `rsp_w=float(cfg.wm_rsp_w)` and `rsp_delta=(0.42, 0.24)` to level 1 only | MR8: drop the kwarg (G-DVB red) |
| LevelConfig | HJ:464 (`rollout_w`, last field) | `rsp_w: float = 0.0`, `rsp_delta: tuple = (0.42, 0.24)`; constants `RSP_K = 10`, `RSP_STRIDE = 10`, `RSP_CTX = 1`, `RSP_S_MIN = 0.1` in HJ (no flag) | — |
| loss term | HJ:755-758 (the rollout block), HJ:759 (`level/L{lv}`) | after HJ:758: `if lv == 1 and c.rsp_w > 0.0: r = self.response_loss(z, enc.idm_target[0], c)`; `out["rsp/L1"] = r["loss"]`; the six detached diagnostics; `level_loss = level_loss + c.rsp_w * r["loss"]`. With `rsp_w == 0.0` the block does not exist: no key, no op, the earlier total bit for bit | MR3: compute but do not add (P3 identity red) |
| the method | new, after `rollout_loss` (HJ:799-829) | `response_loss(z, a, c)`: card section 2.1 steps 2-9; contrasts batched into ONE `self.rollout(1, z0x4, a4, context=RSP_CTX)` call; readout via `torch.func.functional_call(self.idm_heads[0], {k: v.detach() …}, (prev, nxt))`; slope from `self.idm_heads[0]` on `z[:, :-1].detach(), z[:, 1:].detach()` under `torch.no_grad()`, float32 | MR4 (readout not detached), MR5 (`z0` not detached), MR6 (one contrast sign flipped), MR7 (one random draw) |
| WM-only loss | VM:1189-1203 (`wm_only_loss` → `forward_mean` VM:771 → `stack.loss`) | nothing: the term rides inside `stack.loss`, and `T.update({f"wm/{k}": v …})` (VM:1201) carries it into the terms | — |
| loop and backward | TR:3500 (`model.wm_only_loss(b)`), TR:3505 (`T["total"].backward()`), log TR:3516-3517 (every term) | nothing | — |
| built reader | HJ:573-594 (`built_levers`; the rollout row HJ:591) | `if c.rsp_w > 0: row.update(rsp_w=…, rsp_delta=…, rsp_k=…, rsp_stride=…, rsp_ctx=…)` (present only when moved, so a flag-off stack keeps its key set) | — |
| measured reader | HJ:839-886 (`built_rollout`, the pattern) | `built_rsp(n_frames)`: run the BUILT loss once on noise with the RNG forked, a hook on `rollout` counting the term's calls; returns `w_in_loss_rsp` = (level/L1 − every other L1 term) / rsp/L1, the anchors, K, the contrast count and the context actually passed | MR3 reads 0.0 here |

### ⛔ Two integration hazards found in source (MEASURED by reading; both are build blockers, not footnotes)
1. **`built_rollout` would misread the ladder lever.** HJ:880-882 computes level 1's `w_in_loss` as (level − pred − β sigreg − γ idm) / rollout. With the new term on, that reads `1 + λ·rsp/rollout`, not 1.0, and `bad_wm_ladder` (TR:1023) refuses `--wm-rollout-w 1.0`. The build subtracts `c.rsp_w * out["rsp/L1"]` in HJ:881's `other`. Mutation **MR9** restores the old `other` and must turn the ladder G-DVB red on an RSP argv.
2. **`gamma_from_log.py` (ACTSENS control C2) would misread gamma on an RSP log**: its identity (`actsens/gamma_from_log.py`, 16 lines) has no `rsp` term. ACTRSP's C2 uses a new `rsp_w_from_log.py` that subtracts both, and recovers λ (RSP arms) or reads no `rsp/L1` key (NULL arms).

### Other places the new field touches
* `INIT_CKPT_MATCH` (TR:766): the new key must NOT be added. The M2b init checkpoints have no `wm_rsp_w` in their `cfg`, and the reader shape does not depend on it.
* Readers: an ACTRSP checkpoint's `cfg` carries `wm_rsp_w`, so it loads only under a tree that declares the field (G-HYG refuses undeclared attributes). R35 and NULL checkpoints load under the ACTRSP tree through the default. The read runs under the ACTRSP tree (`READ_PLAN.md`).
* `built_rollout` and `built_rsp` call `self.loss` under `torch.no_grad()` on noise: the term runs there with the slope clamped at 0.1, which is finite.

## 2 Gradient probes (dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`, a real G-LIVE batch where the tree has one, else the synthetic batch of `test_v8f_actsens.py`)
* **P1, the term alone.** On a deep copy with every parameter thawed, backward of `c.rsp_w * out["rsp/L1"]` only. **Non-zero** grad abs sum on `wm.stack.predictors.0.*` and `wm.stack.act_embed.*`. **Exactly 0.0** on `wm.stack.idm_heads.0.*` (the readout is detached), `wm.proj.*` (anchor and slope detached), `wm.stack.predictors.1/2`, `wm.stack.encoders.*`, `wm.stack.macro_encoders.*`, `wm.stack.idm_heads.1/2`, and every non-WM parameter. The exact zeros are the controls that prove the probe can read a zero.
* **P2, flag absent.** No `rsp/L1` key; `total` and the state_dict equal the ACTSENS build's bit for bit (card C1).
* **P3, dose identity.** Same init and batch: level/L1(λ) − level/L1(off) = λ × rsp/L1 to 1e-5 relative at λ 1.0 AND λ 0.3 (two points: the term enters linearly and at the declared weight). The gate form of this is card C3 at step 0 (`live_first`, AMEND_ACTSENS_1).
* **P4, analytic identity** (card C9, float64): the linear predictor / linear readout cases give G exactly; `K₂B = diag(s)` gives loss and gradient exactly 0.0; `B = 0` gives exactly 0.5. The slope clamp: a readout with zero slope is used at exactly 0.1.
* **P5, pairing** (card C4): flag on vs off at seeds 0 and 2: equal WM init sha256 and equal (ci, t) over the first 50 batches.

## 3 Mutations (each must turn a named check RED; runner = `F:/TanitAD-scratch/actsens/mutate_actsens.py` pattern, results `mutation_results_rsp_ALL.json`)
| id | mutation | check that goes RED |
|---|---|---|
| MR1 | delete TR:424's new kwarg | G-DVB (`bad_wm_rsp`: declared 1.0, built 0.0); G-LIVE missing `wm/rsp/L1` |
| MR2 | argparse default 0.0 → 1.0 | C1 parity (`live.terms` gain a key, total moves) |
| MR3 | the term computed, not added | P3 identity; `built_rsp.w_in_loss_rsp` = 0.0 ≠ 1.0 |
| MR4 | readout parameters not detached | P1: `idm_heads.0` grad ≠ 0 |
| MR5 | anchor `z0` not detached | P1: `proj` grad ≠ 0 |
| MR6 | `latm` contrast sign flipped | P4: the `K₂B = diag(s)` case reads a loss above 0.0 |
| MR7 | one `torch.rand(1)` inside the term | P5: the first-50-batch (ci, t) lists differ |
| MR8 | `rsp_w` not passed to level 1's LevelConfig | G-DVB built 0.0 |
| MR9 | HJ:881's `other` without the rsp term | the ladder G-DVB (`bad_wm_ladder`, `w_in_loss` L1) on an RSP argv |

## 4 The default that disables the term, and the argv rule
`--wm-rsp-w` default **0.0** disables it: the HJ block does not exist, no key, no op. Every ACTRSP lever argv NAMES `--wm-rsp-w 1.0`; NULL argvs leave it absent (C1 makes absent = the ACTSENS build). A selfcheck record `levers.wm_rsp_readback` (TR:3316-3321 pattern) is written only when the argv names the flag, so a flag-off `selfcheck.json` keeps its key set.

## 5 G-DVB entry (what the gate reads; `launch_gate.py` is NOT edited)
* **Trainer side (the G-DVB WM-only box-2 arms run under):** `FLAG_CHECKS["wm_rsp_w"] = "lever:wm_rsp_w"`; `bad_wm_rsp(a, cfg, model)` beside `bad_wm_idm_gamma` (TR:1503-1521), called from `declared_vs_built` beside TR:1534. It refuses: `--wm-rsp-w` non-finite or < 0; > 0 without `--wm b`, `--kin-mode yaw` or `--wm-only`; the declared λ not equal to (a) `HJEPAStack.built_levers()[1]["rsp_w"]` and (b) `WMAux.built_rsp(hmax)["w_in_loss_rsp"]` to 1e-6; a measured anchor count ≠ 9, K ≠ 10, contrasts ≠ 4 or context ≠ 1 at `--hmax 91`. G-LIVE (TR:1098; its `expect` set TR:1128) adds `wm/rsp/L1` to the expected terms when `cfg.wm_rsp_w > 0`.
* **Package side:** `actrsp_check.py gate` (copy of `F:/TanitAD-scratch/actsens/box2_build/actsens_check.py`) issues no token unless the selfcheck's `wm_rsp_readback.built.w_in_loss_rsp` equals the argv's λ to 1e-6 and C3's identity holds at step 0.
* **Why not `launch_gate.py`:** at `81cee6c1`, `launch_gate.py` has 0 hits for `wm-only`, `wm_idm` or `wm-sampler` (MEASURED, grep); its `v8f` profile (`launch_gate.py:979-1044`) is the refcv8-derived Thor gate of another trainer path (`_R8` with the P10 prior). ACTSENS's freeze note F1 placed the WM-only G-DVB in the package checker and the trainer selfcheck. ⚠️ If the Master Mind wants a `launch_gate.py` profile for WM-only arms, that is a separate decision; a pin or sha edit there moves six profile fingerprints pinned in three tests (memory note), so it is the lander's work, not this card's.

## 6 Tests and suites before landing
* New `stack/tests/test_v8f_actrsp.py`: P1-P5, the config refusals, MR9's reader fix. Run with `test_v8f_actsens.py`, `test_v8f_divrs.py` and the ladder tests (the ACTSENS commit message records 67 pass for actsens + divrs + ladder), plus `test_text_encoding_is_explicit`, on dev-box CPU with `CUDA_VISIBLE_DEVICES=-1`, serially, `--basetemp` on F: two levels deep.
* The code lands as a diff under `raw/` on a branch from `81cee6c1` (ACTSENS freeze note F2), never into `stack/` of `agent/arch-inf-20260803`.
