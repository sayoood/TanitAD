#!/usr/bin/env python
"""Q1 — is the refcv6 SPEC §2 recipe actually implemented in the trainer?

SPEC_REFCV6_V2.md §2 (still binding under §10.2's trunk swap):
    "optimiser | **AdamW**, weight decay 1e-4, **encoder lr x0.5** of the
     heads, warm-up then cosine"

⛔ THE METHOD, and why it is not a restatement of the code under test
(advisory class F shape 6 — "the regression arm restates the code it guards"):

  * the optimiser is built by the REAL ``refc_v3_train.build_optimizer``;
  * the schedule lambda and the per-step learning-rate application are NOT
    retyped here. They are recovered from the trainer's SOURCE by AST
    (``ast.get_source_segment`` over the ``train`` function body) and
    ``exec``-ed verbatim, so this probe measures the code that will run and
    drifts to RED — not to a silent restatement — if the trainer changes.
  * the EXPECTED values are LITERALS taken from the SPEC prose, never
    expressions over the trainer.

Run:
  PYTHONPATH=D:/Projects/TanitAD/stack python probe_recipe.py --json <out.json>
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math          # noqa: F401 — the exec'd sched lambda needs it in scope
import pathlib
import sys

import torch

TRAINER = pathlib.Path(r"D:/Projects/TanitAD/stack/scripts/refc_v3_train.py")

# ---- LITERALS FROM THE SPEC, not from the code -------------------------- #
SPEC = {
    "optimiser": "AdamW",
    "weight_decay": 1e-4,
    "encoder_lr_mult": 0.5,
    "schedule": "warmup then cosine",
}


def _load_trainer():
    spec = importlib.util.spec_from_file_location("refc_v3_train", TRAINER)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train"] = m
    spec.loader.exec_module(m)
    return m


def recover_schedule_source() -> dict:
    """Pull the LIVE sched lambda + lr-apply loop out of ``train`` by AST.

    Returns their exact source text and line numbers. An instrument that
    retyped these would go green over a trainer that had changed them.
    """
    tree = ast.parse(TRAINER.read_text(encoding="utf-8", errors="replace"))
    src = TRAINER.read_text(encoding="utf-8", errors="replace")
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    sched_src = lr_src = None
    sched_ln = lr_ln = None
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and sched_src is None
                and any(getattr(t, "id", None) == "sched" for t in node.targets)
                and isinstance(node.value, ast.Lambda)):
            sched_src, sched_ln = ast.get_source_segment(src, node), node.lineno
        if (isinstance(node, ast.For) and lr_src is None
                and isinstance(node.target, ast.Name) and node.target.id == "g"
                and "param_groups" in (ast.get_source_segment(src, node.iter) or "")):
            lr_src, lr_ln = ast.get_source_segment(src, node), node.lineno
    if sched_src is None or lr_src is None:
        raise SystemExit("INVALID: could not recover sched/lr-apply from source "
                         f"(sched={sched_src is not None} lr={lr_src is not None})")
    return {"sched_src": sched_src, "sched_line": sched_ln,
            "lr_apply_src": lr_src, "lr_apply_line": lr_ln}


def build_model(m, extra_argv: list[str]):
    from tanitad.refs import refc_v3 as v3
    p = m.build_parser()
    argv = ["--smoke", "--arm", "hier", "--out", "unused"] + extra_argv
    args = p.parse_args(argv)
    cfg = m._pin_trainer_cfg(v3.refc_v3_smoke_config(args.arm == "hier"), args)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)
    return args, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--opt", default="dd")
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--warmup", default="2000")
    ap.add_argument("--steps", default="30000")
    a = ap.parse_args()

    m = _load_trainer()
    rec = recover_schedule_source()
    out: dict = {"spec_literals": SPEC, "recovered": rec, "arms": {}}
    print("== LIVE SOURCE, recovered by AST (not retyped) ==")
    print(f"  refc_v3_train.py:{rec['sched_line']}  {rec['sched_src']}")
    print(f"  refc_v3_train.py:{rec['lr_apply_line']}  {rec['lr_apply_src']}")

    for opt_kind in ("adam", "dd"):
        argv = ["--opt", opt_kind, "--lr", a.lr,
                "--warmup", a.warmup, "--steps", a.steps]
        if opt_kind == "dd":
            # refcv6 §2/§10.2: the timm ImageNet trunk is what the 0.5x is for.
            argv += ["--trunk", "timm", "--trunk-name", "resnet34",
                     "--trunk-in-channels", "3",
                     "--no-trunk-pretrained"]
        args, model = build_model(m, argv)
        opt = m.build_optimizer(model, args)
        groups = [{"name": g.get("name", f"g{i}"),
                   "lr": float(g["lr"]),
                   "weight_decay": float(g.get("weight_decay", 0.0)),
                   "n_tensors": len(g["params"]),
                   "n_params": int(sum(p.numel() for p in g["params"]))}
                  for i, g in enumerate(opt.param_groups)]
        arm = {"opt_class": type(opt).__name__,
               "args_lr": float(args.lr),
               "groups_at_build": groups}

        # ---- replay the LIVE schedule code, verbatim ------------------- #
        ns = {"args": args, "opt": opt, "math": math, "max": max}
        exec(rec["sched_src"], ns)                 # noqa: S102 — live source
        traj = []
        for step in (0, 1, int(args.warmup) - 1, int(args.warmup),
                     int(args.warmup) + 1, int(args.steps) // 2,
                     int(args.steps) - 1):
            ns["step"] = step
            exec(rec["lr_apply_src"], ns)          # noqa: S102 — live source
            traj.append({"step": step,
                         "sched": float(ns["sched"](step)),
                         "lrs": [float(g["lr"]) for g in opt.param_groups]})
        arm["after_live_schedule"] = traj
        out["arms"][opt_kind] = arm

        print(f"\n== --opt {opt_kind} ==")
        print(f"  class = {arm['opt_class']}   args.lr = {arm['args_lr']}")
        for g in groups:
            print(f"    group {g['name']:<8} lr={g['lr']:.6e} "
                  f"wd={g['weight_decay']} tensors={g['n_tensors']} "
                  f"params={g['n_params']:,}")
        if len(groups) == 2:
            r = groups[0]["lr"] / groups[1]["lr"] if groups[1]["lr"] else float("nan")
            arm["ratio_at_build"] = r
            print(f"    AT BUILD  encoder/head lr ratio = {r:.4f} "
                  f"(SPEC literal {SPEC['encoder_lr_mult']})")
        print("  after the LIVE per-step lr application:")
        for t in traj:
            rr = (t["lrs"][0] / t["lrs"][1]) if len(t["lrs"]) == 2 and t["lrs"][1] else None
            arm.setdefault("ratio_after", []).append(rr)
            print(f"    step {t['step']:>6}  sched={t['sched']:.6f}  "
                  f"lrs={['%.6e' % v for v in t['lrs']]}"
                  + (f"  ratio={rr:.4f}" if rr is not None else ""))

    # ---- VERDICTS, against SPEC LITERALS --------------------------------- #
    dd = out["arms"]["dd"]
    v = {}
    v["AdamW"] = (dd["opt_class"] == SPEC["optimiser"])
    v["weight_decay_1e-4"] = all(
        abs(g["weight_decay"] - SPEC["weight_decay"]) < 1e-12
        for g in dd["groups_at_build"])
    v["encoder_lr_x0.5_AT_BUILD"] = (
        abs(dd.get("ratio_at_build", float("nan")) - SPEC["encoder_lr_mult"]) < 1e-9)
    ratios = [r for r in dd["ratio_after"] if r is not None]
    v["encoder_lr_x0.5_SURVIVES_THE_SCHEDULE"] = bool(ratios) and all(
        abs(r - SPEC["encoder_lr_mult"]) < 1e-9 for r in ratios)
    # warm-up: sched must RISE from ~1/warmup to 1.0 across the warm-up window
    s = {t["step"]: t["sched"] for t in dd["after_live_schedule"]}
    v["warmup_present"] = (s[0] < s[1] < s[int(a.warmup) - 1] <= 1.0 + 1e-9)
    # cosine: after warm-up the multiplier must fall to ~0 at the last step
    v["cosine_present"] = (s[int(a.warmup)] > s[int(a.steps) // 2]
                           > s[int(a.steps) - 1] >= -1e-9)
    out["verdicts"] = v
    print("\n== VERDICT vs SPEC LITERALS (--opt dd) ==")
    for k, ok in v.items():
        print(f"  {'PASS' if ok else '⛔ FAIL'}  {k}")
    print(f"\n  ratios after the live schedule: {ratios}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2),
                                        encoding="utf-8")
        print(f"\n[artifact] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
