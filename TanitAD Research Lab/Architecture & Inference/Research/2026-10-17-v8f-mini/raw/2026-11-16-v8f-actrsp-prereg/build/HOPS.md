# HOPS: how `--wm-rsp-w` reaches the loss (ACTRSP build on ACTSENS commit 81cee6c1)

Line numbers refer to `F:/TanitAD-scratch/actrsp/build/tree/` as built (evidence class MEASURED by reading and by running; each hop has a mutation in `mutation_results_rsp_ALL.json` that turns it RED).
TR = `stack/scripts/v8f_mini_train.py`, VM = `stack/tanitad/models/v8f_mini.py`, HJ = `stack/tanitad/models/hjepa.py`.

| # | hop | file:line | what happens | mutation |
|---|---|---|---|---|
| 1 | argv | TR:338 | `--wm-rsp-w` float, `default=0.0` (off), `dest=wm_rsp_w` | MR2 default 1.0 |
| 2 | FLAG_CHECKS | TR:129 | `"wm_rsp_w": "lever:wm_rsp_w"` (else the TR:1617 check `flags without a declared-vs-built entry` refuses every launch) | existing check |
| 3 | namespace to config | TR:431 (`make_config`) | `wm_rsp_w=float(a.wm_rsp_w)` | MR1 delete the line |
| 4 | config field and validation | VM:213 (field), VM:288-294 (`__post_init__`) | `wm_rsp_w: float = 0.0`; finite and >= 0; > 0 needs `wm == "b"` and `kin_mode == "yaw"` | `test_config_refusals` |
| 5 | built module | VM:709 (`WMAux.__init__`, level 1's `mk(...)` call) | `rsp_w=float(cfg.wm_rsp_w), rsp_delta=RSP_DELTA` passed to level 1 only, and only when > 0 (VM:167 `RSP_DELTA = (0.42, 0.24)`) | MR8 drop the kwarg |
| 6 | LevelConfig | HJ:470-471 | `rsp_w: float = 0.0`, `rsp_delta: tuple = (0.42, 0.24)`; constants HJ:260 `RSP_K, RSP_STRIDE, RSP_CTX, RSP_S_MIN = 10, 10, 1, 0.1` | |
| 7 | loss term | HJ:768-772 (`HJEPAStack.loss`) | `if lv == 1 and c.rsp_w > 0.0:` calls `response_loss` (HJ:845), writes `rsp/L1` and the six detached diagnostics `rsp_g_aa/L1 ... rsp_s_w/L1`, then `level_loss += c.rsp_w * rs["loss"]` (HJ:772). `rsp_w == 0.0`: the block does not exist (no key, no op) | MR3 not added |
| 8 | term body | HJ:845-880 | anchors `range(0, T-10, 10)` (9 at T = 91); `z0 = z[:, idx].detach()` (MR5); four contrasts `e[..] = 1,-1,1,-1` (MR6); ONE batched `self.rollout(1, z0x4, a4, context=1)` (HJ:865); readout `functional_call(idm_heads[0], detached params)` (MR4); slope from real transitions under `no_grad`, float32 then cast, floor 0.1; `G = diff / (2 delta s)`; loss `mean((G - I)^2)`. No `torch.rand` (MR7) | MR4 MR5 MR6 MR7 |
| 9 | WM-only loss | VM:781 `forward_mean` -> `stack.loss`; VM:1204 `wm_only_loss`, VM:1216 `T.update({f"wm/{k}": v ...})` | the term rides inside `stack.loss`; `wm/rsp/L1` and the diagnostics reach `T` | |
| 10 | loop and backward | TR:3577 `model.wm_only_loss(b)`, TR:3582 `T["total"].backward()`, log of every term | no change | |
| 11 | built reader (config echo) | HJ:600-601 (`built_levers`) | row gains `rsp_w, rsp_delta, rsp_k, rsp_stride, rsp_ctx` only when moved | |
| 12 | built reader (measured) | HJ:945 `built_rsp`, VM:785 `WMAux.built_rsp` | runs the BUILT loss once on noise on a float64 deep copy, a wrapper on `rollout` records the term's call (5-D actions): `calls, contrasts, anchors, k, ctx_passed`, and `w_in_loss_rsp = (level/L1 - pred - beta sigreg - gamma idm - rollout_w rollout) / rsp/L1` | MR3 reads 0.0 |
| 13 | G-DVB | TR:1559 `bad_wm_rsp`, called at TR:1609 | refuses non-finite or < 0; > 0 without `--wm b`, `--kin-mode yaw`, `--wm-only`; declared != built config; declared != `built_levers()[1]["rsp_w"]`; declared != `built_rsp.w_in_loss_rsp` (1e-6); geometry != (1 call, 4 contrasts, K 10, context 1, `len(range(0, hmax-10, 10))` anchors); flag absent but a built `rsp_w` key | MR1 MR8 MR3 |
| 14 | selfcheck record | TR:3396 `levers.wm_rsp_readback` (only when the argv names the flag); TR:3413 `live_first` also written for the flag | flag-off `selfcheck.json` keeps its key set | `test_gate_flag_absent_keeps_the_key_set` |
| 15 | G-LIVE | TR:1118-1153 (`wm_only_live_check`) | `wm/rsp/L1` joins `expect`; with the flag on, step 0 is compared with a deep copy of the SAME weights and batch at `rsp_w = 0` (`live.c3`): unweighted terms equal, levels 2-3 equal, `level/L1` difference = rsp_w x `rsp/L1` | MR3 MR7 |
| 16 | ladder reader fixes (hazards found in source) | HJ:938-939 (`other` includes `c.rsp_w * rsp/L1`), HJ:`if actions.dim() == 5` skip in `built_rollout` | without them `bad_wm_ladder` reads `w_in_loss` 1 + rsp_w x rsp / rollout (REACHABILITY blocker 1) and counts the term's 10 steps as ladder free-run steps (a third hazard, `k` 14 instead of 4) | MR9 MR10 |

Not wired by design: `INIT_CKPT_MATCH` (TR:766 area) gets no `wm_rsp_w` key (REACHABILITY section 1); `gamma_from_log.py` is not reused: `actrsp_check.py` computes the rsp identity of card C2 (`rsp_from_rows`).

Note on hop 16: HJ line numbers for the `built_rollout` skip are those of the `if actions.dim() == 5:` line inside `built_rollout` (HJ:893 onward); `built_rsp` carries its own recorder at HJ:945 onward.
