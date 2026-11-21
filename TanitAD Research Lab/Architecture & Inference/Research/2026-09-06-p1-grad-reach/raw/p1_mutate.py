"""P1 MUTATION PROOFS -- show every diagnostic's FAILURE BRANCH is reachable.

A diagnostic that reads the same on a healthy and a broken model has measured
nothing. Each block below REINTRODUCES the condition and requires the readout
to FLIP.

  M1  stage freeze        : S-W -> S-J must flip A_STAGE_FREEZE -> live
  M2  guarded-off loss    : O1/O3 weights 0 -> 1 must flip grad None -> present
  M3  AdamW semantics     : the DECISIVE control for M71's inference --
                            in-optimizer + wd>0 + grad None  => BIT-EXACT
                            in-optimizer + wd>0 + grad ZEROS => DECAYS
  M4  init-std artifact   : is `step_readout_op.net.{1,3}` "trained", or is the
                            reported std just nn.Linear's own init scale?

ASCII-only. CPU only, zero GPU.
"""
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stack"))
sys.path.insert(0, str(HERE / "stack" / "scripts"))

from tanitad.models.v6 import V6Stack, apply_stage_freeze
from tanitad.models.metric_dynamics import StepDisplacementReadout
from train_v6_staged import V6LossWeights, synthetic_train_batch, v6_loss_step
from p1_audit2 import build_cfg, build_weights

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "C:/Users/Admin/tanitad-caches/mm-e19-assets-20260901/"
           "v7tiny_postrain30k")
CFG = json.loads((RUN / "config.json").read_text())
LR = float(CFG["args"]["lr"]); WD = float(CFG["args"]["wd"])
STEPS = int(CFG["args"]["steps"])
O5_K = int(CFG["args"].get("o5_k", 1))
TARGETS_B = ["step_readout_op.net.0.weight",
             "masked_cells.blocks.layers.0.norm1.weight"]
TARGETS_A = ["predictor_tac.blocks.0.0.weight", "vocab_tac.norm.weight"]
FAILS = []


def check(label, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", label, detail))
    if not cond:
        FAILS.append(label)


def build(stage):
    torch.manual_seed(0)
    s = V6Stack(build_cfg(CFG))
    apply_stage_freeze(s, stage)
    return s


def backward_once(stack, weights, stage, o1_k=10, seed=0):
    stack.zero_grad(set_to_none=True)
    b = synthetic_train_batch(stack, batch=2, k=max(o1_k, O5_K, 2), seed=seed)
    b["gt_wp"] = torch.randn(2, o1_k, 2,
                             generator=torch.Generator().manual_seed(seed))
    out = v6_loss_step(stack, b, stage=stage, weights=weights, o1_k=o1_k,
                       o5_k=O5_K, o5_form=CFG["args"].get("o5_form", "l1"),
                       generator=torch.Generator().manual_seed(seed),
                       sigreg_generator=torch.Generator().manual_seed(seed))
    out["loss"].backward()
    return out


def status(stack, names):
    d = dict(stack.named_parameters())
    return {n: (bool(d[n].requires_grad), d[n].grad is None) for n in names}


print("=" * 78)
print("M1 -- STAGE-FREEZE MUTATION: does the 'A' readout flip when the stage "
      "changes?")
print("=" * 78)
s_sw = build("S-W")
s_sj = build("S-J")
st_sw = status(s_sw, TARGETS_A)
st_sj = status(s_sj, TARGETS_A)
for n in TARGETS_A:
    print("  %-40s S-W requires_grad=%-5s | S-J requires_grad=%s"
          % (n, st_sw[n][0], st_sj[n][0]))
check("M1 baseline: frozen at S-W",
      all(not st_sw[n][0] for n in TARGETS_A))
check("M1 MUTATION: live at S-J (failure branch REACHABLE)",
      all(st_sj[n][0] for n in TARGETS_A))
tr_sw = [p for p in s_sw.parameters() if p.requires_grad]
tr_sj = [p for p in s_sj.parameters() if p.requires_grad]
check("M1 optimizer membership changes", len(tr_sj) > len(tr_sw),
      "(S-W %d tensors -> S-J %d tensors)" % (len(tr_sw), len(tr_sj)))

print("")
print("=" * 78)
print("M2 -- GUARDED-OFF-LOSS MUTATION: does 'grad is None' flip when O1/O3 "
      "are switched ON?")
print("=" * 78)
w_off = build_weights(CFG)                       # the run's own: O1=O3=0
w_on = dataclasses.replace(w_off, o1_ctrl=1.0, o1_fact=1.0, o1_scene=1.0,
                           o3_masked=1.0)
s_off = build("S-W"); backward_once(s_off, w_off, "S-W")
s_on = build("S-W"); backward_once(s_on, w_on, "S-W")
a_off = status(s_off, TARGETS_B); a_on = status(s_on, TARGETS_B)
for n in TARGETS_B:
    print("  %-46s O1/O3=0 grad_is_None=%-5s | O1/O3=1 grad_is_None=%s"
          % (n, a_off[n][1], a_on[n][1]))
check("M2 baseline: grad is None with the run's own weights",
      all(a_off[n][1] for n in TARGETS_B))
check("M2 MUTATION: grad APPEARS when the objective is on "
      "(failure branch REACHABLE)", all(not a_on[n][1] for n in TARGETS_B))
check("M2 requires_grad was True THROUGHOUT (so 'frozen' is NOT the cause)",
      all(a_off[n][0] and a_on[n][0] for n in TARGETS_B))

print("")
print("=" * 78)
print("M3 -- ADAMW SEMANTICS: is 'bit-exact at 1.0' evidence of being OUTSIDE "
      "the optimizer?")
print("=" * 78)
print("  torch %s | lr=%g wd=%g steps=%d" % (torch.__version__, LR, WD, STEPS))
ln_none = nn.LayerNorm(64)
ln_zero = nn.LayerNorm(64)
opt = torch.optim.AdamW(list(ln_none.parameters()) + list(ln_zero.parameters()),
                        lr=LR, weight_decay=WD)
w0 = ln_none.weight.detach().clone()
N = 2000
for _ in range(N):
    ln_none.weight.grad = None                     # never touched by any loss
    ln_zero.weight.grad = torch.zeros_like(ln_zero.weight)   # touched, zero
    opt.step()
d_none = float((ln_none.weight.detach() - w0).abs().max())
d_zero = float((ln_zero.weight.detach() - w0).abs().max())
pred = (1.0 - LR * WD) ** N
got = float(ln_zero.weight.detach()[0])
# the EXACT mechanism, replicated in the same dtype: decoupled decay is a
# repeated float32 multiply, so it carries float32 accumulation error. A
# float64 closed form is NOT the right control for it -- this is.
sim = torch.ones(1, dtype=torch.float32)
f = torch.tensor(1.0 - LR * WD, dtype=torch.float32)
for _ in range(N):
    sim = sim * f
sim = float(sim[0])
print("  grad=None : max|dw| after %d steps = %.17g  (value %.8f)"
      % (N, d_none, float(ln_none.weight.detach()[0])))
print("  grad=ZEROS: max|dw| after %d steps = %.17g  (value %.8f)"
      % (N, d_zero, got))
print("             float64 closed form (1-lr*wd)^n = %.8f (delta %.3g)"
      % (pred, abs(got - pred)))
print("             float32 REPLICATION of the same multiply = %.8f "
      "(delta %.3g)" % (sim, abs(got - sim)))
check("M3a grad=None param in the optimizer stays BIT-EXACT", d_none == 0.0)
check("M3b grad=ZEROS param in the SAME optimizer DECAYS "
      "(failure branch REACHABLE)", d_zero > 0.0)
check("M3c the decay IS decoupled weight decay and nothing else "
      "(float32 replication)", abs(got - sim) < 1e-7,
      "-- the float64 gap %.3g is float32 accumulation, not a second effect"
      % abs(got - pred))
full = (1.0 - LR * WD) ** STEPS
print("  => at the run's real %d steps a STEPPED norm weight would read "
      "%.6f, not 1.0" % (STEPS, full))
check("M3d CONCLUSION: 'bit-exact' does NOT imply 'outside the optimizer'",
      d_none == 0.0 and d_zero > 0.0)

print("")
print("=" * 78)
print("M4 -- IS step_readout_op.net.{1,3} 'TRAINED', OR IS THAT ITS INIT "
      "SCALE?")
print("=" * 78)
d_op = CFG["v6_config"]["_derived"]["d_op"]
torch.manual_seed(1234)
fresh = StepDisplacementReadout(d_op)
ck = torch.load(RUN / "ckpt.pt", map_location="cpu", weights_only=False)
sd = ck.get("stack") or ck.get("model") or ck
for key, mod in (("net.1", fresh.net[1]), ("net.3", fresh.net[3])):
    fan_in = mod.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    init_std = bound / math.sqrt(3.0)
    t = sd["step_readout_op.%s.weight" % key].float()
    print("  %s  fan_in=%-5d  U(-a,a) a=%.8f  analytic init std=%.8f"
          % (key, fan_in, bound, init_std))
    print("       fresh-init std=%.8f | CHECKPOINT std=%.8f | ckpt max|w|="
          "%.8f" % (float(mod.weight.std()), float(t.std()), float(t.abs().max())))
    check("M4 %s ckpt max|w| is INSIDE the init bound a=%.6f "
          "(=> never stepped)" % (key, bound), float(t.abs().max()) <= bound)
    check("M4 %s ckpt std matches the ANALYTIC INIT std to 3 dp" % key,
          abs(float(t.std()) - init_std) < 5e-4)
b0 = sd["step_readout_op.net.0.bias"].float()
print("  net.0.bias (LayerNorm, init EXACTLY 0): max|b| = %.17g"
      % float(b0.abs().max()))
check("M4 net.0.bias is bit-exactly 0 -- a second at-init witness",
      float(b0.abs().max()) == 0.0)

print("")
print("=" * 78)
print("RESULT: %s" % ("ALL MUTATION PROOFS PASSED" if not FAILS
                      else "FAILURES: %s" % FAILS))
print("=" * 78)
sys.exit(1 if FAILS else 0)
