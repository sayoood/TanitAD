#!/usr/bin/env python3
"""The committed weight rule for L_field (eval/PREREG_AUX_FIELD_LOSS_DRAFT.md, "Weight rule (committed now)"), as a tool against the PATCHED refe/train.py.

    On 8 real micro-batches of the arm's own bank at the host's init (seeded draw, seed 20261012), measure
        r(w) = || grad_{traj_head} (w * L_field) ||  /  || grad_{traj_head} l_traj ||
    Candidates w in {1/4, 1/2, 1, 2, 4, 8, 16, 32, 64}.  Take the SMALLEST w with r(w) >= 0.25.
    K2  : r(64)  <  0.25  -> the arm is NOT launched (the host's fans are too clean for the loss to matter: report the dirty-fan share instead).
    K2b : r(1/4) >  1.0   -> the arm is NOT launched (the term would dominate the trajectory loss at the LOWEST weight: check the label sign and extent).

    python weight_rule.py --args "<the arm's train.py argv, as the chain's $ARGS>" [--refe <dir of the patched refe>] [--out weight_rule.json]
    exit 0  a weight was selected (written to --out; the value goes into the prereg's blanks and then into the arm's argv as --w-field)
    exit 5  K2 or K2b: the arm is not launched          exit 4  the tool could not measure (a refusal, never a guess)

HOW IT MEASURES.  It runs the REAL `train.run(a)` on the arm's argv under instrumentation until the optimiser is built (the same technique as the launch gate's BUILD run), capturing the
MODEL and the TargetBank the trainer itself built -- warm start from --init-from, nav mode, cameras, amp, label store included -- so the tool cannot drift from the trainer.  Nothing is
trained and nothing is written (the stop is before the run directory is created; a --resume is ignored on purpose: the rule is defined AT THE HOST'S INIT).  It then draws 8 micro-batches
of --batch rows from the bank's rows with `torch.Generator().manual_seed(20261012)` + `randperm` (without replacement), forwards each exactly as the trainer does (autocast per --amp,
`traj.float()`), and takes the gradient of l_traj (`wta_loss`, the argv's --yaw-wrap) and of L_field (`field_loss` with the argv's margin / interp, the loader's labels and valid mask)
with respect to the parameters of `traj_head`.

THE DECISION STATISTIC (an interpretation the prereg does not spell out -- FLAGGED for the Master Mind): the gradients are POOLED over the 8 micro-batches (their mean, what an accumulation
window hands the optimiser) before the norms are taken; the per-micro-batch ratios are reported as diagnostics (min / median / max) and do not decide.  Both L_field and l_traj enter the
trainer as `loss = (l_traj + l_score + w * L_field) / accum`, so r is EXACTLY LINEAR in w: r(w) = w * r(1).  The candidates are all powers of two, for which w * g is bit-identical to the
gradient of (w * L_field) (the tests prove it); the tool reports r at every candidate from that.

WHAT IT CANNOT SAY: it needs the arm's real bank, images and label store, and the host's init checkpoint (P2's final checkpoint does not exist yet).  On the dev box it is exercised on a
self-made tiny bank only (test_field_wiring.py part e); `--test-pin` pins such a store and stamps the result NOT A DECISION.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261012
N_MICRO = 8
CANDIDATES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
R_TARGET, R_MAX_LOW = 0.25, 1.0


class Stop(Exception):
    pass


def decide(r_by_w: dict) -> dict:
    """The committed rule on {w: r(w)}.  Pure: no I/O, no model.  K2b is judged on r(1/4), K2 on r(64); otherwise the smallest candidate with r >= 0.25."""
    ws = sorted(r_by_w)
    if tuple(ws) != tuple(sorted(CANDIDATES)):
        raise ValueError(f"the candidates must be exactly {CANDIDATES}, got {ws}")
    if any(not math.isfinite(r_by_w[w]) or r_by_w[w] < 0.0 for w in ws):
        return {"status": "ERROR", "w_field": None, "why": f"a non-finite or negative r(w): {r_by_w}"}
    if r_by_w[ws[0]] > R_MAX_LOW:
        return {"status": "K2b", "w_field": None, "why": f"r(1/4) = {r_by_w[ws[0]]:.6g} > {R_MAX_LOW}: the field term would dominate the trajectory loss at the lowest weight; check the label sign and extent"}
    if r_by_w[ws[-1]] < R_TARGET:
        return {"status": "K2", "w_field": None, "why": f"r(64) = {r_by_w[ws[-1]]:.6g} < {R_TARGET}: the host's fans are too clean for the loss to matter; report the dirty-fan share instead"}
    w = next(w for w in ws if r_by_w[w] >= R_TARGET)
    return {"status": "SELECT", "w_field": w, "why": f"the smallest candidate with r(w) >= {R_TARGET}: r({w:g}) = {r_by_w[w]:.6g}"}


def build_instrumented(refe: Path, argv: list, test_pin: str | None):
    """Run the real train.run(a) until the optimiser is built; return (T, a, model, ds, dev)."""
    sys.path.insert(0, str(refe))
    import torch
    import train as T
    if test_pin:
        import aux_drivable_head as AUX
        sha, n = test_pin.split(":")
        AUX.PINNED_STORE_SHA256, AUX.PINNED_STORE_KEYS = sha, int(n)
    ap = T.build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "field_labels", None):
        raise SystemExit("weight_rule: the argv has no --field-labels: there is no label to measure L_field against")
    if not a.out:
        a.out = str(Path(tempfile.gettempdir()) / "weight_rule_unused_run_dir")          # run() refuses a long run without --out; it is never created (we stop before it)
    cap = {"model": None, "ds": None}
    Orig, Bank, AdamW0 = T.REFe, T.TargetBank, torch.optim.AdamW

    class Rec(Orig):
        def __init__(self, cfg):
            super().__init__(cfg)
            cap["model"] = self

    class RecBank(Bank):
        def __init__(self, *a_, **k_):
            super().__init__(*a_, **k_)
            cap["ds"] = self

    class StopAdam(AdamW0):
        def zero_grad(self, *a_, **k_):
            raise Stop()

    patches = [(T, "REFe", Orig), (T, "TargetBank", Bank), (torch.optim, "AdamW", AdamW0)]
    T.REFe, T.TargetBank, torch.optim.AdamW = Rec, RecBank, StopAdam
    stub = None
    if a.labeller == "online":                        # the rule needs no labels; do not start a worker pool that nothing will close
        import online_labeller as OLm
        stub = OLm.OnlineLabeller

        class NoLab:
            def __init__(self, *a_, **k_):
                pass
        OLm.OnlineLabeller = NoLab
    rc = None
    try:
        rc = T.run(a)
    except Stop:
        rc = "STOPPED_AT_BUILD"
    finally:
        for obj, name, orig in patches:
            setattr(obj, name, orig)
        if stub is not None:
            OLm.OnlineLabeller = stub
    if rc != "STOPPED_AT_BUILD" or cap["model"] is None or cap["ds"] is None:
        raise SystemExit(f"weight_rule: the trainer did not reach the optimiser build (rc={rc!r}); nothing was measured")
    if cap["ds"].field is None:
        raise SystemExit("weight_rule: the trainer built its TargetBank without a label source")
    dev = next(cap["model"].parameters()).device
    return T, a, cap["model"], cap["ds"], dev


def measure(T, a, model, ds, dev) -> dict:
    """Per micro-batch the gradients of l_traj and of L_field (weight 1) on traj_head; pooled norms; r at every candidate."""
    import torch
    from torch.utils.data import default_collate
    head = [p for p in model.traj_head.parameters() if p.requires_grad]
    if not head:
        raise SystemExit("weight_rule: traj_head has no trainable parameter")
    n_rows = len(ds)
    need = N_MICRO * a.batch
    if n_rows < need:
        raise SystemExit(f"weight_rule: the bank has {n_rows} rows, fewer than the {need} the rule draws ({N_MICRO} micro-batches x --batch {a.batch})")
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_rows, generator=g).tolist()[:need]
    batches = [perm[k * a.batch:(k + 1) * a.batch] for k in range(N_MICRO)]
    ids = [[(ds.rows[i].get("log_name"), ds.rows[i].get("token"), int(ds.rows[i].get("rank", 0))) for i in b] for b in batches]
    sum_t = sum_f = None
    per = []
    labelled = total = 0
    inext, viol = [], []
    for b in batches:
        batch = default_collate([ds[i] for i in b])
        img, ego, goal, tgt, cand_xy, cand_tg, cand_m, cand_id, cal = batch[:9]
        sdf, valid = batch[-2].to(dev), batch[-1].to(dev)
        img, ego, goal, tgt = (x.to(dev) for x in (img, ego, goal, tgt))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.amp == "bf16" and dev.type == "cuda")):
            traj, _ = model(img, ego, goal, calib=cal if cal.numel() else None)[:2]
        traj = traj.float()
        l_traj = T.wta_loss(traj, tgt, yaw_wrap=(a.yaw_wrap == "on"))[0]
        l_field, parts = T.field_loss(traj, sdf, margin=a.field_margin, valid=valid, interp=a.field_interp, return_parts=True)
        gt = torch.autograd.grad(l_traj, head, retain_graph=True, allow_unused=True)
        gf = torch.autograd.grad(l_field, head, retain_graph=True, allow_unused=True)
        vt = torch.cat([(x if x is not None else torch.zeros_like(p)).detach().double().flatten() for x, p in zip(gt, head)])
        vf = torch.cat([(x if x is not None else torch.zeros_like(p)).detach().double().flatten() for x, p in zip(gf, head)])
        sum_t = vt if sum_t is None else sum_t + vt
        sum_f = vf if sum_f is None else sum_f + vf
        nt, nf = float(vt.norm()), float(vf.norm())
        per.append({"l_traj": float(l_traj), "l_field": float(l_field), "norm_traj": nt, "norm_field": nf, "r1": (nf / nt) if nt > 0 else None})
        labelled += int(valid.sum())
        total += int(valid.numel())
        inext.append(float(parts["in_extent_frac"]))
        viol.append(float(parts["violating_frac"]))
        model.zero_grad(set_to_none=True)
    Gt, Gf = sum_t / N_MICRO, sum_f / N_MICRO
    nT, nF = float(Gt.norm()), float(Gf.norm())
    if not nT > 0.0:
        raise SystemExit("weight_rule: ||grad_traj_head l_traj|| is zero on the pooled batches: r is undefined")
    r1 = nF / nT
    rs = [p["r1"] for p in per if p["r1"] is not None]
    srt = sorted(rs)
    return {"seed": SEED, "n_micro": N_MICRO, "batch": a.batch, "micro_batch_ids": ids, "head_params": int(sum(p.numel() for p in head)),
            "pooled": {"norm_grad_traj_loss": nT, "norm_grad_field_loss_w1": nF, "r1": r1},
            "per_micro_batch": per, "r1_per_micro_min_median_max": [srt[0], srt[len(srt) // 2], srt[-1]] if srt else None,
            "r_by_w": {repr(w): w * r1 for w in CANDIDATES}, "label_coverage_of_the_8_batches": labelled / max(total, 1),
            "field_inext_mean": sum(inext) / len(inext), "field_viol_mean": sum(viol) / len(viol)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--args", required=True, help="the arm's train.py argv as one string (the chain's $ARGS); must carry --field-labels")
    ap.add_argument("--refe", default=str(HERE), help="the directory holding the PATCHED train.py (default: this file's directory)")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--test-pin", default=None, help="TEST ONLY: '<sha256>:<keys>' of a self-made label store; the result is then stamped NOT A DECISION")
    a = ap.parse_args(argv)
    targv = a.args.split()                              # the shell's word splitting of an unquoted $ARGS (default IFS) -- exactly gate_token.argv_from_string, which the chain and the gate use
    T, ns, model, ds, dev = build_instrumented(Path(a.refe).resolve(), targv, a.test_pin)
    res = measure(T, ns, model, ds, dev)
    rule = decide({w: res["r_by_w"][repr(w)] for w in CANDIDATES})
    res.update({"rule": rule, "argv_sha256": hashlib.sha256("\x00".join(targv).encode()).hexdigest(), "init_from": ns.init_from, "field_margin": ns.field_margin,
                "field_interp": ns.field_interp, "label_store": ns.field_labels, "device": str(dev),
                "NOT_A_DECISION": bool(a.test_pin), "ignored_resume": bool(ns.resume)})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"weight_rule: pooled ||grad l_traj|| = {res['pooled']['norm_grad_traj_loss']:.6g}   ||grad L_field|| (w=1) = {res['pooled']['norm_grad_field_loss_w1']:.6g}   r(1) = {res['pooled']['r1']:.6g}")
    print("  r(w): " + "  ".join(f"{w:g}:{res['r_by_w'][repr(w)]:.4g}" for w in CANDIDATES))
    print(f"  label coverage of the 8 batches {res['label_coverage_of_the_8_batches']:.3f}   in-extent {res['field_inext_mean']:.3f}   violating {res['field_viol_mean']:.3f}")
    print(f"  RULE: {rule['status']}  w_field = {rule['w_field']}  -- {rule['why']}" + ("   [TEST PIN: NOT A DECISION]" if a.test_pin else ""))
    return {"SELECT": 0, "K2": 5, "K2b": 5}.get(rule["status"], 4)


if __name__ == "__main__":
    sys.exit(main())
