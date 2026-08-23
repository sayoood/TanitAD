"""E-DEPLOY-4b - does TF32 move the LEARNED tensors, or only leave the
(degenerate) plan output alone?

WHY THIS IS NOT OPTIONAL. E-DEPLOY-4 measured TF32+graph as 22 % faster with
ZERO deviation on `plan.waypoints`, which invites the headline "a free 2x,
bit-identical". That headline would be a false positive: on this stage-S-W
checkpoint the waypoints are `unicycle_rollout(0, 0, v0)` — provably insensitive
to matmul precision — while the TF32 switch was proven LIVE (a 2048^2 fp32
matmul changed by 7.3e-02). A tensor that cannot move is not evidence that
nothing moved.

⇒ Measure the deviation on the tensors that ARE learned: z_op, z_tac, z_str,
plan.feat. Whatever this finds decides how the 22 % may be described:

  * trunk deviation ZERO           -> TF32 genuinely is a free lever here
  * trunk deviation NONZERO        -> TF32 is a LOSSY lever whose accuracy cost
                                      is unmeasured; the 22 % ships only behind
                                      the four-family gate, exactly like fp16

CONTROL: strict-fp32 repeated twice must be bit-identical at every stage, or a
TF32 delta is noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

STACK = os.path.expanduser("~/TanitAD/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

import torch  # noqa: E402

R = {"spec": "E-DEPLOY-4b", "controls": {}}
STAGES = ["z_op", "z_tac", "z_str"]


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)
    torch.set_float32_matmul_precision("high" if on else "highest")


def grab(o):
    d = {k: o[k].float().clone() for k in STAGES}
    d["plan.feat"] = o["plan"]["feat"].float().clone()
    d["plan.waypoints"] = o["plan"]["waypoints"].float().clone()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/v7tiny/champ30k"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dv = torch.device("cuda")

    with open(os.path.join(a.ckpt, "config.json")) as fh:
        cfg = json.load(fh)
    from train_v6_staged import build_stack_from_args, synthetic_train_batch
    stack = build_stack_from_args(argparse.Namespace(**cfg["args"]))
    ck = torch.load(os.path.join(a.ckpt, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    res = stack.load_state_dict(ck["stack"], strict=False)
    assert not res.missing_keys and not res.unexpected_keys
    del ck
    stack = stack.to(dv).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    b = synthetic_train_batch(stack, batch=1, k=4, seed=0, device=dv)
    frames, acts, v0 = b["frames"], b["actions2"], b["v0"]
    A = stack.cfg.predictor.action_dim
    if acts.shape[-1] != A:
        acts = torch.cat([acts, torch.zeros(acts.shape[0], acts.shape[1],
                                            A - acts.shape[-1], device=dv,
                                            dtype=acts.dtype)], dim=-1)

    def fwd():
        with torch.no_grad():
            return stack(frames, acts, v0)

    set_tf32(False)
    ref = grab(fwd())
    rep = grab(fwd())
    ctrl = {k: float((rep[k] - ref[k]).abs().max()) for k in ref}
    R["controls"]["fp32_repeat_bit_identical"] = {
        "expect": "0.0 at every stage", "measured": ctrl,
        "pass": all(v == 0.0 for v in ctrl.values())}
    print("CONTROL fp32 repeat:",
          R["controls"]["fp32_repeat_bit_identical"]["pass"], flush=True)

    set_tf32(True)
    tf = grab(fwd())
    R["tf32_deviation"] = {}
    for k in ref:
        d = (tf[k] - ref[k]).abs()
        R["tf32_deviation"][k] = {
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "ref_absmax": float(ref[k].abs().max()),
            "rel_to_scale": (float(d.max()) / float(ref[k].abs().max()))
            if float(ref[k].abs().max()) > 0 else None,
            "moved": bool(float(d.max()) > 0.0)}
        v = R["tf32_deviation"][k]
        print("  %-16s max_abs %.4e  (scale %.4g)  rel %.3e  moved=%s"
              % (k, v["max_abs"], v["ref_absmax"],
                 v["rel_to_scale"] if v["rel_to_scale"] is not None else -1,
                 v["moved"]), flush=True)

    moved = [k for k, v in R["tf32_deviation"].items() if v["moved"]]
    R["moved_stages"] = moved
    R["verdict"] = (
        "TF32 is a LOSSY lever on this stack — it moves %s. The 22 %% "
        "TF32+graph speedup must go through the four-family gate exactly like "
        "fp16; it may NOT be described as free or bit-identical." % moved
        if moved else
        "TF32 left every measured tensor bit-identical — the 22 %% speedup is "
        "free on this stack.")
    print(R["verdict"], flush=True)
    R["_evidence_class"] = "MEASURED (ours; Thor, in-process)"
    json.dump(R, open(a.out, "w"), indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
