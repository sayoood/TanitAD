"""P2b - THE MUTATION PROOF, on the REAL banked checkpoints.

Not a synthetic module: the same `.pt` files whose numbers are in the register.

  A) NEGATIVE (must refuse): the 9 banked v7-tiny `step_readout_op`.
  B) POSITIVE (must NOT refuse): the same readout after ONE AdamW step at the
     arm's own `lr 1e-4, wd 0.05` -- the mutation that reverses the defect.
  C) END-TO-END: `t1_eval.py`'s own decode expression, `grounding.step["op"]`,
     on a real banked checkpoint, with the flag OFF and ON.
  D) CONTROL that the guard is not just "refuse everything": a REAL trained
     readout from a flagship v4 checkpoint (`grounding['step.op.*']`, banked at
     step=1) passes.
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-mdhazard\stack")
sys.path.insert(0, r"C:\Users\Admin\tanitad-mdhazard\stack\scripts")

import tanitad                                                   # noqa: E402
from tanitad.models.metric_dynamics import StepDisplacementReadout  # noqa: E402
from tanitad.models.v6 import (UntrainedMetricReadout,           # noqa: E402
                               assert_metric_readout_trained,
                               metric_readout_status)

assert "tanitad-mdhazard" in tanitad.__file__, tanitad.__file__
print("[import]", tanitad.__file__, flush=True)

CKROOT = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
OUT = r"C:\Users\Admin\tanitad-mdhazard\_work\p2_mutation.json"
RES = {"import_origin": tanitad.__file__}


def load_readout(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["stack"]
    args = ((ck.get("config") or {}).get("args")
            if isinstance(ck.get("config"), dict) else None)
    ro = {k[len("step_readout_op."):]: v for k, v in sd.items()
          if k.startswith("step_readout_op.")}
    d = ro["net.1.weight"].shape[1] // 2
    m = StepDisplacementReadout(d)
    m.load_state_dict(ro)
    del ck, sd
    return m, args, d


# --------------------------------------------------------------------------- #
# A / B  -- refuse on the banked file, pass after ONE real optimizer step      #
# --------------------------------------------------------------------------- #
arms = {}
for a in sorted(os.listdir(CKROOT)):
    p = os.path.join(CKROOT, a, "ckpt.pt")
    if not (a.startswith("v7tiny_") and os.path.isfile(p)):
        continue
    m, args, d = load_readout(p)
    row = {"state_dim": d}
    try:
        assert_metric_readout_trained(m, where="A/banked", run_args=args)
        row["A_banked"] = "DID NOT REFUSE (defect)"
    except UntrainedMetricReadout as e:
        row["A_banked"] = "REFUSED"
        row["A_verdict"] = metric_readout_status(m)["verdict"]
        row["A_msg_head"] = str(e)[:90]
    # MUTATION: one real AdamW step at the arm's own hyper-parameters
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
    g = torch.Generator().manual_seed(0)
    opt.zero_grad()
    z = torch.randn(4, d, generator=g)
    m(z, torch.randn(4, d, generator=g)).pow(2).mean().backward()
    opt.step()
    # the arm's OWN record still says O1 == 0, so asking WITH it must read
    # CONTRADICTION (a real state, not a false positive); asking the way a
    # fresh arm trained at O1 > 0 would present must read TRAINED.
    st_rec = metric_readout_status(m, run_args=args, refresh=True)
    st_norec = metric_readout_status(m, run_args=None, refresh=True)
    row["B_with_the_arms_own_O1_zero_record"] = st_rec["verdict"]
    row["B_as_an_O1_positive_arm_would_present"] = st_norec["verdict"]
    row["B_max_ratio"] = round(st_norec["linear"]["net.1.weight"]["max_ratio"], 8)
    try:
        assert_metric_readout_trained(m, where="B/mutated", run_args=None)
        row["B_mutated"] = "PASSED"
    except UntrainedMetricReadout:
        row["B_mutated"] = "STILL REFUSED (guard too blunt)"
    arms[a] = row
    print(f"  [{a:28s}] A={row['A_banked']:9s} ({row.get('A_verdict')})  "
          f"B={row['B_mutated']:9s} verdict={row['B_as_an_O1_positive_arm_would_present']:8s} "
          f"with_record={row['B_with_the_arms_own_O1_zero_record']:14s} "
          f"max_ratio={row['B_max_ratio']}", flush=True)
RES["A_negative_and_B_mutation"] = arms

# ⚠️ note: B's verdict is CONTRADICTION rather than TRAINED for these arms,
# because their own record says O1 == 0 while the weights just moved. That is
# the guard doing its job -- so the "did the refusal lift" question is asked
# WITHOUT the record, exactly as a fresh arm trained at O1 > 0 would present.

# --------------------------------------------------------------------------- #
# D -- the guard is NOT "refuse everything": a REAL trained readout passes     #
# --------------------------------------------------------------------------- #
pos = {}
for p in sorted(glob.glob(
        r"C:\Users\Admin\AppData\Local\Temp\v4smoke_*\ckpt.pt"))[:20]:
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        pos[p] = f"UNREADABLE {e}"
        continue
    g = ck.get("grounding")
    if not isinstance(g, dict) or "step.op.net.1.weight" not in g:
        del ck
        continue
    ro = {k[len("step.op."):]: v for k, v in g.items()
          if k.startswith("step.op.")}
    d = ro["net.1.weight"].shape[1] // 2
    hidden = ro["net.1.weight"].shape[0]
    m = StepDisplacementReadout(d, hidden=hidden)
    m.load_state_dict(ro)
    st = metric_readout_status(m)
    try:
        assert_metric_readout_trained(m, where="D/real trained flagship readout")
        verdict = "PASSED"
    except UntrainedMetricReadout:
        verdict = "REFUSED (false positive)"
    pos[os.path.basename(os.path.dirname(p))] = {
        "banked_train_steps": ck.get("step"), "status": st["verdict"],
        "guard": verdict,
        "max_ratio": round(st["linear"]["net.1.weight"]["max_ratio"], 8)}
    del ck
    if len(pos) >= 4:
        break
RES["D_real_trained_readout"] = pos
print("\n[D] real trained flagship readouts:")
for k, v in pos.items():
    print(f"  {k}: {v}", flush=True)

# --------------------------------------------------------------------------- #
# C -- END-TO-END through t1_eval's own decode expression                      #
# --------------------------------------------------------------------------- #
from tanitad.eval.v6_probe_trunk import V6Grounding                # noqa: E402


class _StackShim:
    """Only what V6Grounding touches -- so the end-to-end check does not need
    to rebuild a 19 M-param stack from a config the box may not carry."""

    def __init__(self, ro):
        self.step_readout_op = ro


m, args, d = load_readout(os.path.join(CKROOT, "v7tiny_emao14_30k", "ckpt.pt"))
e2e = {}
g_off = V6Grounding(_StackShim(m), run_args=args)
try:
    g_off.step["op"]                                # t1_eval.py's own line
    e2e["flag_off"] = "DID NOT REFUSE (defect)"
except UntrainedMetricReadout as e:
    e2e["flag_off"] = "REFUSED"
    e2e["flag_off_msg"] = str(e)[:140]
g_on = V6Grounding(_StackShim(m), allow_untrained_readout=True, run_args=args)
e2e["flag_on"] = ("PASSED and recorded: "
                  + json.dumps(g_on.readout_status["verdict"]))
e2e["artifact_stamp_verdict"] = g_on.readout_status["verdict"]
e2e["artifact_stamp_evidence"] = g_on.readout_status["evidence"]
RES["C_end_to_end_t1_eval_decode_expression"] = e2e
print("\n[C] t1_eval's decode expression on v7tiny_emao14_30k:")
print(" ", json.dumps(e2e, indent=1))

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(RES, fh, indent=1)
print("[wrote]", OUT)
