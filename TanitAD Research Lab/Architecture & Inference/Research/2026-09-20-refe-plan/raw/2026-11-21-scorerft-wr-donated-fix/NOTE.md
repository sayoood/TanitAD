# The SCORER-FT weight rule crashed on the pod: donated buffers vs `retain_graph` (MM, 2026-11-21)

**What happened.** `after_f.sh` ran `refe/weight_rule_rank.py` from package A (`code_c98e6114`) at 16:42:48Z, on T's argv (`ARGS_WR.txt`, which carries `--compile`).
It exited 1 at 16:43:52Z, in `measure()` at line 188, with:
`RuntimeError: This backward function was compiled with non-empty donated buffers which requires create_graph=False and retain_graph=False`
(`weight_rule_rank_CRASH_pod_20261121T1643Z.log`). The model built and loaded C1's final checkpoint (`f09949e2`), and the online labeller started. So the crash came at the first gradient.
This is a CODE defect: no K2 / K2b outcome was produced, and none may be read from it. The chain logged `ZZAF_SFT_SKIP` and moved on to the field replicates, as designed.

**Mechanism (read from the pod's torch 2.7.1 source).** `measure()` takes three `torch.autograd.grad(..., retain_graph=True)` on one graph: l_score, L_rank and l_traj.
Under `--compile`, AOTAutograd's compiled backward donates saved-activation buffers, and it then refuses any backward that retains the graph.
The refusal is gated on `torch._functorch.config.donated_buffer` (`_aot_autograd/runtime_wrappers.py:2200-2203`). So is the donation itself, in the buffer-index collection (`jit_compile_runtime_wrappers.py:878`).
The trainer takes ONE backward per step, so the trainer is not affected; only this measurement tool is.

**Fix.** `build_instrumented()` sets `torch._functorch.config.donated_buffer = False` before `train.run` builds and compiles the model. The result JSON records `donated_buffer`.
Donation only reuses memory, so no gradient value changes, and the rule itself (card section 2.5: candidates, r(w), rho(w), K2, K2b) is unchanged.

**Tests** (`test_rank_loss.py`, on an export of tip `4cd217ca` with the fix; `rank_fast.log`, `rank_full.log`):
- `--fast` passes 145 checks, which is 144 + the new assertion (`chk_weight_rule_smoke`: `res["donated_buffer"] is False`).
- Full passes 145 checks with 32 RED mutations, including the new line mutation "donated buffers left on": the line becomes `pass` and `weight_rule_smoke` goes RED.

**What could not be reproduced off the pod.**
- `repro_donated.py` is a tiny compiled trunk with three retained-graph gradients.
- On the dev box, `aot_eager` donates nothing, and `inductor` needs a C++ compiler the box lacks.
- On the pod's CPU, inductor (`nice -n 19`, CPU only, the GPU was training) donated nothing on that tiny graph either. Both modes ran clean, and the compiled-vs-eager gradient difference was 1.9e-9 (`repro_pod_cpu.txt`).
- So the decisive test is the real weight-rule run on the pod. If it fails again, the next lever is three separate forward passes (one backward each, no retained graph), at three times the cost.
