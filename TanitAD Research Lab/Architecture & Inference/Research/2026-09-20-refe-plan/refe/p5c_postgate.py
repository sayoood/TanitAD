#!/usr/bin/env python3
"""P5-C post-training gate G-ADAM (PREREG_P5C.md 10.4): run on each arm's run directory BEFORE the arm is read.  CPU only, nothing is trained.

Pattern of `probe_zerograd3.py` / `vdac_postgate.py`: Adam allocates its state LAZILY, so a registered parameter with NO entry in `opt['state']` never received a gradient, and one whose first moment is exactly zero
received only zero ones.  Reads <out>/ckpt_last.pt (the trainer's `opt.state_dict()` + `model_partial`) and <out>/model_final.pt (FULL), and the base file the arm warm-started from.
  1. every REGISTERED optimiser tensor holds state, with step == --steps, a finite NON-ZERO exp_avg_sq (card) AND exp_avg (brief), and the state covers EXACTLY --n-params elements (the trainable count
     the trainer printed / wrote to config.json `params.trainable`; the frozen trunk has no state, so a trunk in the optimiser would overshoot).  Meaningful only because the optimiser started FRESH (card 3).
  1b. config.json's `warm_start.sha256` == the base file's sha256 (the run started from THIS base).
  2. weights: the trainable head/LoRA tensors differ from the base (||dW|| > 0) in this arm; the frozen trunk is bit-identical to the base; with --other, this arm's traj_head differs from the other arm's.
  3. the probe's own CONTROLS must read what they must (a zero-gradient parameter reads EXACTLY 0.0 first moment; a real one reads > 0), else the probe is broken and the gate FAILS.
  usage: p5c_postgate.py <out dir> --base <init-from file> --steps 2000 (--n-params N | --n-params-from <config.json>) [--other <other arm out dir>]
  prints ZZP5C_POSTGATE_OK ... (rc 0) or ZZP5C_POSTGATE_FAIL ... (rc 1).  Library: judge_adam, judge_weights, judge_pair, controls (each returns failure strings, [] = pass).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _lora(name: str) -> bool:
    return name.split(".")[0] == "backbone" and name.rsplit(".", 1)[-1] in ("A", "B")


def is_frozen(name: str) -> bool:
    """the frozen trunk: `backbone.*` except the LoRA A / B tensors, and the goal frequencies (gate_checks.FROZEN_GROUPS)."""
    top = name.split(".")[0]
    return (top == "backbone" and not _lora(name)) or top == "goal_freqs"


def adam_numel(opt) -> int:
    return sum(int(opt["state"][p]["exp_avg"].numel()) for g in opt["param_groups"] for p in g["params"] if p in opt["state"] and "exp_avg" in opt["state"][p])


def judge_adam(opt, steps: int, n_params: int) -> list:
    import torch
    if not isinstance(opt, dict) or "state" not in opt or "param_groups" not in opt:
        return ["the checkpoint carries no optimiser state"]
    registered = [p for g in opt["param_groups"] for p in g["params"]]
    state = opt["state"]
    with_state = [p for p in registered if p in state]
    bad = []
    if not registered:
        bad.append("the optimiser registers no parameter")
    if len(with_state) != len(registered):
        bad.append(f"{len(registered) - len(with_state)} of {len(registered)} registered parameters have NO Adam state (they never received a gradient)")
    zero_m, zero_v, nonfinite, wrong_step, no_m, numel = [], [], [], [], [], 0
    for p in with_state:
        st = state[p]
        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
        if m is None or v is None:
            no_m.append(p)
            continue
        numel += int(m.numel())
        if not (bool(torch.isfinite(m).all()) and bool(torch.isfinite(v).all())):
            nonfinite.append(p)
        if float(m.abs().sum()) == 0.0:
            zero_m.append(p)
        if float(v.abs().sum()) == 0.0:
            zero_v.append(p)
        if float(st.get("step", -1)) != float(steps):
            wrong_step.append((p, float(st.get("step", -1))))
    if no_m:
        bad.append(f"{len(no_m)} parameters have Adam state without exp_avg / exp_avg_sq: {no_m[:6]}")
    if zero_m:
        bad.append(f"{len(zero_m)} parameters have an EXACTLY ZERO first moment (only zero gradients were applied): {zero_m[:8]}")
    if zero_v:
        bad.append(f"{len(zero_v)} parameters have an EXACTLY ZERO exp_avg_sq: {zero_v[:8]}")
    if nonfinite:
        bad.append(f"{len(nonfinite)} parameters have a non-finite Adam state: {nonfinite[:8]}")
    if wrong_step:
        bad.append(f"{len(wrong_step)} parameters took a step count other than {steps}: {wrong_step[:4]}")
    if numel != n_params:
        bad.append(f"the Adam state covers {numel} parameters, the built model has {n_params} trainable")
    return bad


def _state(sd):
    return sd["model"] if isinstance(sd, dict) and "model" in sd else sd


def judge_weights(final, base) -> list:
    """`final` / `base`: FULL state dicts (name -> tensor).  Frozen trunk bit-identical to the base; some trainable tensor moved; every float tensor finite."""
    import torch
    bad = []
    if set(final) != set(base):
        return [f"the final model's tensor names differ from the base's: only final {sorted(set(final) - set(base))[:4]}, only base {sorted(set(base) - set(final))[:4]}"]
    moved_frozen = [k for k in final if is_frozen(k) and not torch.equal(final[k], base[k])]
    if moved_frozen:
        bad.append(f"{len(moved_frozen)} FROZEN-trunk tensors differ from the base (they must be bit-identical): {moved_frozen[:4]}")
    if not any(is_frozen(k) for k in final):
        bad.append("no frozen-trunk tensor found: the probe cannot tell the trunk from the heads")
    head_moved = [k for k in final if not is_frozen(k) and final[k].dtype.is_floating_point and not torch.equal(final[k], base[k])]
    if not head_moved:
        bad.append("no trainable tensor differs from the base (||dW|| == 0): the arm trained nothing")
    th = [k for k in final if k.startswith("traj_head.") and final[k].dtype.is_floating_point]
    if not th:
        bad.append("the final model has no traj_head tensors")
    elif all(torch.equal(final[k], base[k]) for k in th):
        bad.append("the traj_head is bit-identical to the base (||dW|| == 0): the head did not train")
    nf = [k for k, v in final.items() if v.dtype.is_floating_point and not bool(torch.isfinite(v).all())]
    if nf:
        bad.append(f"non-finite tensors in the final model: {nf[:4]}")
    return bad


def judge_pair(final_a, final_b) -> list:
    """the two arms really differ in the traj head (C1 vs C0)."""
    import torch
    th = [k for k in final_a if k.startswith("traj_head.") and k in final_b]
    if not th:
        return ["no traj_head tensors to compare between the arms"]
    if all(torch.equal(final_a[k], final_b[k]) for k in th):
        return ["the two arms' traj_head weights are bit-identical: the arms do not differ"]
    return []


def controls() -> list:
    """the probe reads zero where it must and non-zero where it must: a zero-gradient parameter has an EXACTLY 0.0 first moment, a real one does not, and a parameter without a gradient has no state."""
    import torch
    p0, p1, p2 = (torch.nn.Parameter(torch.ones(3)) for _ in range(3))
    opt = torch.optim.AdamW([p0, p1, p2], lr=1e-2)
    p0.grad, p1.grad = torch.zeros(3), torch.ones(3)
    opt.step()
    sd = opt.state_dict()
    bad = []
    if float(sd["state"][0]["exp_avg"].abs().sum()) != 0.0:
        bad.append("control: a zero-gradient parameter's first moment does not read exactly 0.0")
    if not float(sd["state"][1]["exp_avg"].abs().sum()) > 0.0:
        bad.append("control: a real-gradient parameter's first moment reads 0")
    if 2 in sd["state"]:
        bad.append("control: a parameter that never received a gradient has Adam state")
    if not judge_adam(sd, 1, 6):                         # the pair above MUST be judged RED (p0 zero moment, p2 no state)
        bad.append("control: the judge passed an optimiser holding a zero-moment and a stateless parameter")
    return bad


def PGfile_sha256(path, chunk: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--n-params", type=int)
    ap.add_argument("--n-params-from", help="a run's config.json (P7's): params.trainable is the count")
    ap.add_argument("--other", help="the other arm's run dir: this arm's traj_head must differ from its model_final.pt")
    a = ap.parse_args()
    import torch
    n = a.n_params
    if a.n_params_from:
        n2 = json.load(open(a.n_params_from, encoding="utf-8"))["params"]["trainable"]
        if n is not None and n != n2:
            print(f"ZZP5C_POSTGATE_FAIL --n-params {n} != {a.n_params_from}'s params.trainable {n2}")
            return 1
        n = n2
    if n is None:
        print("ZZP5C_POSTGATE_FAIL give --n-params or --n-params-from: the exact trainable count is never defaulted")
        return 1
    bad = controls()
    ck_p, fin_p = os.path.join(a.out, "ckpt_last.pt"), os.path.join(a.out, "model_final.pt")
    for p in (ck_p, fin_p, a.base):
        if not os.path.isfile(p):
            bad.append(f"{p} does not exist")
    if not bad:
        ck = torch.load(ck_p, map_location="cpu", weights_only=False)
        bad += judge_adam(ck.get("opt"), a.steps, n)
        cfg = os.path.join(a.out, "config.json")
        if os.path.isfile(cfg):
            cj = json.load(open(cfg, encoding="utf-8"))
            own = cj.get("params", {}).get("trainable")
            if own != n:
                bad.append(f"the arm's own config.json says {own} trainable, the gate expects {n}")
            ws_ = (cj.get("warm_start") or {}).get("sha256")          # train.py records the base it loaded (card 10.3 asks the checkpoint meta for this; the trainer writes it to config.json)
            if ws_ != PGfile_sha256(a.base):
                bad.append(f"the arm's config.json warm_start.sha256 {str(ws_)[:12]} is not the sha256 of the base file {a.base} ({PGfile_sha256(a.base)[:12]}): it did not start from this base")
        del ck
        final, base = _state(torch.load(fin_p, map_location="cpu", weights_only=False)), _state(torch.load(a.base, map_location="cpu", weights_only=False))
        bad += judge_weights(final, base)
        if a.other:
            op = os.path.join(a.other, "model_final.pt")
            if not os.path.isfile(op):
                bad.append(f"{op} does not exist")
            else:
                bad += judge_pair(final, _state(torch.load(op, map_location="cpu", weights_only=False)))
    if bad:
        print("ZZP5C_POSTGATE_FAIL " + "; ".join(bad))
        return 1
    print(f"ZZP5C_POSTGATE_OK {n} trainable parameters, every one with Adam state (step {a.steps}, non-zero first and second moment), frozen trunk bit-identical to the base, controls exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
