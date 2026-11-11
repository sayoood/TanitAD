#!/usr/bin/env python3
"""VDAC post-training gate (PREREG_VDAC_P6 section 7, row "post-training"): every head parameter has Adam state, and the state is not the state of a head that never received a gradient.
Pattern of the zero-grad probe (`probe_zerograd3.py`): Adam allocates its state LAZILY, so a registered parameter with no entry in `opt['state']` never received a gradient, and a parameter whose first moment is exactly zero received only zero ones.
Reads <arm out dir>/aux_drv.pt (written by scorer_finetune.save_aux: head state_dict + the head optimiser's state_dict + update + meta).  CPU only, nothing is trained.
  usage: vdac_postgate.py <out dir> --updates N [--n-params 2440450]      prints ZZVDAC_POSTGATE_OK ... (rc 0) or ZZVDAC_POSTGATE_FAIL ... (rc 1)
  library: judge(ck, updates, n_params) -> list of failure strings ([] = pass).  The deliberate regressions (a state entry dropped, a first moment zeroed, a wrong step) are exercised by test_vdac_gate.py and must each make judge() non-empty.
"""
import argparse
import math
import os
import sys

N_PARAMS_CARD = 2440450          # PREREG_VDAC_P6 section 7: aux_drv declared = head built (n_params 2,440,450, S1c run.json)


def judge(ck, updates, n_params=N_PARAMS_CARD) -> list:
    import torch
    bad = []
    if not isinstance(ck, dict) or ck.get("format") != "aux_drv/1":
        return [f"not an aux_drv/1 checkpoint (format {ck.get('format') if isinstance(ck, dict) else type(ck).__name__})"]
    opt = ck.get("opt")
    if not isinstance(opt, dict) or "state" not in opt or "param_groups" not in opt:
        return ["the checkpoint carries no head optimiser state"]
    if ck.get("update") != updates:
        bad.append(f"checkpoint update {ck.get('update')} != the arm's {updates}")
    registered = [p for g in opt["param_groups"] for p in g["params"]]
    state = opt["state"]
    with_state = [p for p in registered if p in state]
    if not registered:
        bad.append("the head optimiser registers no parameter")
    if len(with_state) != len(registered):
        bad.append(f"{len(registered) - len(with_state)} of {len(registered)} registered head parameters have NO Adam state (they never received a gradient)")
    numel, zero_m, nonfinite, wrong_step = 0, [], [], []
    for p in with_state:
        st = state[p]
        m = st.get("exp_avg")
        if m is None:
            bad.append(f"parameter {p}: Adam state without exp_avg")
            continue
        numel += int(m.numel())
        if not bool(torch.isfinite(m).all()) or not bool(torch.isfinite(st.get("exp_avg_sq", m)).all()):
            nonfinite.append(p)
        if float(m.abs().sum()) == 0.0:
            zero_m.append(p)
        step = float(st.get("step", -1))
        if step != float(updates):
            wrong_step.append((p, step))
    if zero_m:
        bad.append(f"{len(zero_m)} head parameters have an EXACTLY ZERO first moment (only zero gradients were applied): {zero_m[:8]}")
    if nonfinite:
        bad.append(f"{len(nonfinite)} head parameters have a non-finite Adam state: {nonfinite[:8]}")
    if wrong_step:
        bad.append(f"{len(wrong_step)} head parameters took a step count other than {updates}: {wrong_step[:4]}")
    if numel != n_params:
        bad.append(f"the Adam state covers {numel} head parameters, the card's built head has {n_params}")
    head = ck.get("head")
    if not isinstance(head, dict) or not head:
        bad.append("the checkpoint carries no head state_dict")
    else:
        for k, v in head.items():
            if hasattr(v, "dtype") and v.dtype.is_floating_point and not bool(torch.isfinite(v).all()):
                bad.append(f"head tensor {k} is non-finite")
                break
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--n-params", type=int, default=N_PARAMS_CARD)
    a = ap.parse_args()
    import torch
    p = os.path.join(a.out, "aux_drv.pt")
    if not os.path.isfile(p):
        print(f"ZZVDAC_POSTGATE_FAIL {p} does not exist")
        return 1
    bad = judge(torch.load(p, map_location="cpu", weights_only=False), a.updates, a.n_params)
    if bad:
        print("ZZVDAC_POSTGATE_FAIL " + "; ".join(bad))
        return 1
    print(f"ZZVDAC_POSTGATE_OK {a.n_params} head parameters, every one with Adam state and a non-zero first moment, {a.updates} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
