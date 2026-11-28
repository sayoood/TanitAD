#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze numerics A/B on the REAL trainer (restart item 2): does freezing the ten admitted dead
groups leave every OTHER parameter's trajectory bit-identical?

Runs `refc_v3_train.main` (the real trainer, real dev-box data) in-process on the canonical refcv7
launch argv, path-mapped exactly as the step-cost harness maps it
(`…/2026-09-27-refcv7-step-cost/code/rc7_argv.py`, eval139 cache as the train set -- a TIMING/
NUMERICS rig, never a model result), with the dev-box departures stated in `run_spec.json`:
b1, CPU only, workers 0, no eval, a smaller trunk (`--trunk-name`, default resnet34), no bf16,
`--trunk-chunk-ckpt 1` (the step-cost CPU recipe).

Nothing in the tree is edited. The instrument is a wrapper on `build_optimizer` (it receives the
model) and on the optimiser INSTANCE's `step`:
  * BEFORE every `opt.step`: every parameter's gradient state -- None / exact-zero / non-zero --
    and a sha256 of the gradient bytes (so "the other parameters' gradients are identical" is
    read, not inferred);
  * AFTER every `opt.step`: a sha256 of every parameter's bytes;
  * at the end: a sha256 of every AdamW state tensor, keyed by parameter NAME.
It writes `digests.json` (per step, per parameter name) and the run's own `metrics.jsonl`.

Usage: numerics_ab.py --tree <tree> --out <dir> --steps N [--resume-ckpt <ckpt.pt>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPCOST = HERE.parents[2] / "2026-09-27-refcv7-step-cost" / "code"
PATHMAP = STEPCOST / "pathmap_devbox_eval139_as_train.txt"
TEN = ("core.strategic.gru", "core.strategic.proj", "nav_to_str", "str_goal_head", "gstr_embed",
       "gstr_film", "core.decoder.ctx_to_cond", "core.route_head", "core.decoder.lat_to_anchor",
       "core.decoder.lon_to_anchor")


def in_ten(n: str) -> str | None:
    for p in TEN:
        if n == p or n.startswith(p + "."):
            return p
    return None


def sha(t) -> str:
    import torch
    t = t.detach()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:24]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000000)
    ap.add_argument("--trunk-name", default="resnet34.a1_in1k")
    ap.add_argument("--resume-ckpt", default=None,
                    help="copy this ckpt.pt into the run dir first: the trainer RESUMES from it")
    ap.add_argument("--conflict-every", default=None)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--map-hires-grad-ckpt", default=None, choices=(None, "on", "off"))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.resume_ckpt:
        shutil.copyfile(a.resume_ckpt, run_dir / "ckpt.pt")
    sys.path[:0] = [str(Path(a.tree) / "stack"), str(Path(a.tree) / "stack" / "scripts"),
                    str(Path(a.tree) / "taniteval"), str(STEPCOST)]
    import rc7_argv
    extra = [("set", "--device", ["cpu"]), ("set", "--trunk-name", [a.trunk_name]),
             ("drop", "--trunk-bf16"), ("set", "--trunk-chunk-ckpt", ["1"]),
             ("set", "--allow-eval-clips-in-train", []),
             ("set", "--label-clock-max-unverified", ["0.05"]), ("set", "--episodes", ["12"]),
             ("drop", "--eval-cache"), ("drop", "--eval-labels"),
             ("drop", "--speed-max-sidecar-v6-eval")]
    if a.conflict_every:
        extra.append(("set", "--conflict-every", [str(a.conflict_every)]))
    if a.map_hires_grad_ckpt:
        extra.append(("set", "--map-hires-grad-ckpt", [a.map_hires_grad_ckpt]))
    spec = rc7_argv.build(a.tree, "R7", str(PATHMAP), batch=1, steps=a.steps,
                          out=str(run_dir).replace("\\", "/"), workers=0, log_every=a.log_every,
                          eval_every=1000000, save_every=a.save_every, extra_edits=extra)
    import torch
    if torch.cuda.device_count() != 0:
        raise SystemExit("CUDA visible -- REFUSED (CPU-only rig)")
    import tanitad
    tf = Path(tanitad.__file__).resolve()
    if not str(tf).lower().startswith(str(Path(a.tree).resolve()).lower()):
        raise SystemExit(f"tanitad imported from {tf}, not from {a.tree}")
    import refc_v3_train as T
    rec = {"tree": a.tree, "tanitad_file": str(tf), "torch": torch.__version__,
           "host": platform.node(), "python": sys.version.split()[0],
           "omp_threads": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
           "argv": spec["argv"], "devbox_departures": spec["devbox_departures"], "edits": extra,
           "resume_ckpt": a.resume_ckpt, "t_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "run_spec.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    S = {"steps": [], "model": None, "opt": None}
    orig_bo = T.build_optimizer

    def build_optimizer(model, args):
        opt = orig_bo(model, args)
        S["model"], S["opt"] = model, opt
        names = {id(p): n for n, p in model.named_parameters()}
        S["names"] = names
        S["requires_grad"] = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
        S["init"] = {n: sha(p) for n, p in model.named_parameters()}
        S["in_opt"] = [names[id(p)] for g in opt.param_groups for p in g["params"]]
        orig_step = opt.step

        def step(*sa, **sk):
            pre = {}
            for n, p in model.named_parameters():
                g = p.grad
                if g is None:
                    pre[n] = "none"
                elif not bool((g != 0).any()):
                    pre[n] = "zero:" + sha(g)
                else:
                    pre[n] = "nz:" + sha(g)
            r = orig_step(*sa, **sk)
            post = {n: sha(p) for n, p in model.named_parameters()}
            S["steps"].append({"grad": pre, "param": post})
            return r
        opt.step = step
        return opt
    T.build_optimizer = build_optimizer
    t0 = time.time()
    err = None
    try:
        T.main(spec["argv"])
    except SystemExit as e:
        if e.code not in (0, None):
            err = f"SystemExit: {e.code}"
    except BaseException as e:                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
    opt = S["opt"]
    opt_state = {}
    if opt is not None:
        for g in opt.param_groups:
            for p in g["params"]:
                st = opt.state.get(p) or {}
                opt_state[S["names"][id(p)]] = {k: sha(v) for k, v in st.items()
                                               if torch.is_tensor(v)}
    res = {"error": err, "wall_s": round(time.time() - t0, 1), "n_opt_steps": len(S["steps"]),
           "requires_grad": S.get("requires_grad"), "in_opt": S.get("in_opt"),
           "init": S.get("init"), "steps": S["steps"], "opt_state": opt_state}
    (out / "digests.json").write_text(json.dumps(res), encoding="utf-8")
    ten = {n: [st["grad"][n] for st in S["steps"]] for n in (S.get("init") or {}) if in_ten(n)}
    print(json.dumps({"error": err, "wall_s": res["wall_s"], "n_opt_steps": len(S["steps"]),
                      "n_params": len(S.get("init") or {}),
                      "n_in_opt": len(S.get("in_opt") or []),
                      "ten_grad_states": sorted({s.split(':')[0] for v in ten.values() for s in v})},
                     indent=1), flush=True)
    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
