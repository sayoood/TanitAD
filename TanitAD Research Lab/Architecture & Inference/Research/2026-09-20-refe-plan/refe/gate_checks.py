#!/usr/bin/env python3
"""The REFe launch-gate CHECKS.  Route and the full contract: eval/REFE_LAUNCH_GATE.md.  CLI and profiles: refe/launch_gate.py.

Every check inspects THE THING (the built model, the live gradient, the loaded checkpoint), never the declaration.  The trainer is run
IN-PROCESS under instrumentation (`instrumented_run`): the real `train.run(a)` executes unchanged; the harness only RECORDS what it
builds (model, optimiser, schedule, banks, loaders) and what flows through it (per-group gradients before every optimiser step, the
first real batch).  Two runs per gate: BUILD (the real levers, stopped before the first step: the schedule and every constructor
argument are observed) and SMOKE (the same levers, `--steps N`: the loss terms and the gradients are observed).

  check   what must hold                                                          regression arm (TANITAD_GATE_ARM, must FAIL)
  G-HYG   REFeConfig refuses an undeclared attribute; the trainer reads no        no_cfg_guard, undeclared_namespace_read
          namespace attribute the parser does not declare; no foreign attribute
  G-DVB   every argv lever equals the BUILT value (table `DVB_KINDS`); every      lever_ignored, unknown_flag, stale_registry
          parser flag is in the table (an unknown flag FAILS)
  G-LIVE  every declared loss term present and finite on every logged row; every  zero_score, frozen_group
          declared trainable group has a non-zero finite gradient; the declared-
          frozen groups read EXACTLY 0 (n_grad == 0); a zero-loss control reads
          exactly 0.0 on every parameter and a real loss does not
  G-EVAL  the planner's load of model_final.pt builds the identical model:        planner_cfg_differs, loader_drops_tensor
          config fields equal (allow-list: training-only), tensors and forward
          outputs on the first REAL batch bit-identical
  G-CKPT  halt + resume == uninterrupted: model, optimiser state, data position   resume_drops_opt
  G-SUITE the listed test scripts exit 0 and print their PASS token               suite_failing_test
  G-CLOCK (P6) the realised schedule equals the registered one                    clock_wrong_epochs

Evidence is written to disk (`evidence_<check>.json`); the token is minted from the files, never from a return value.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARM_ENV = "TANITAD_GATE_ARM"

# the model's parameter groups.  A parameter in NO group is a FINDING: a module was added without a declaration.
TRAINABLE_GROUPS = ("backbone.lora", "registers", "queries", "pos3d_mlp", "reg_compress", "scene_proj", "ego_enc", "dec", "traj_head",
                    "score_dec", "score_q_mlp", "score_head")
FROZEN_GROUPS = ("backbone", "goal_freqs")
SCORER_GROUPS = ("score_dec", "score_q_mlp", "score_head")


def arm_is(name: str, arm: str | None) -> bool:
    return (arm if arm is not None else os.environ.get(ARM_ENV)) == name


def group_of(pname: str) -> str:
    top = pname.split(".")[0]
    if top == "backbone" and pname.rsplit(".", 1)[-1] in ("A", "B"):
        return "backbone.lora"
    return top


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


class Stop(Exception):
    pass


# ============================================================================================ the instrumented run
def instrumented_run(argv: list, *, mode: str, work, arm: str | None = None) -> SimpleNamespace:
    """Run the REAL `train.run(a)` in this process on `argv`, recording what it builds.  mode='build' stops at the first
    `opt.zero_grad` (everything the levers decide is built; nothing has trained); mode='smoke' runs to the end."""
    import torch
    import train as T
    import load_dinov3 as LD
    optim_mod = importlib.import_module("torch.optim.optimizer")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cap = SimpleNamespace(mode=mode, model=None, cfg=None, opt=None, sched=None, steps=None, banks={}, bank_calls={}, loaders=[],
                          samplers=[], grads=[], first_batch=None, fwd_dtype=None, warm=None, trunk_dir=None, trunk_loaded=None,
                          rc=None, error=None, args=None, parser_unknown=[], seed=None, tf32=None, compiled=None, device=None,
                          metrics=[], summary=None, log=str(work / f"{mode}.log"), run_dir=None, wta_yaw_wrap=None, bank_len={}, nav_modules={})
    ap = T.build_parser()
    a, unknown = ap.parse_known_args(list(argv))
    cap.parser_unknown, cap.args = unknown, a
    if unknown:
        cap.rc, cap.error = "UNRECOGNISED_ARGUMENTS", f"the trainer's parser does not declare: {unknown}"
        return cap
    cap.run_dir = a.out
    patches = []

    def patch(obj, name, new):
        patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)

    Orig = T.REFe

    class Rec(Orig):
        def __init__(self, cfg):
            if arm_is("lever_ignored", arm):          # the model quietly ignores the declared --scorer-sees-trunk
                cfg.detach_scorer_context = True
            super().__init__(cfg)
            if cap.model is None:
                cap.model, cap.cfg = self, self.cfg
                cap.nav_modules = {"cmd_embed": hasattr(self, "cmd_embed"), "goal_freqs": hasattr(self, "goal_freqs")}     # read before the BUILD model is freed
                if arm_is("frozen_group", arm):       # a group that should train is frozen after the fact
                    for p in self.traj_head.parameters():
                        p.requires_grad_(False)

                def pre(_m, args, kwargs):
                    if cap.first_batch is None:
                        cal = kwargs.get("calib")
                        cap.first_batch = tuple((x.detach().cpu().clone() if hasattr(x, "detach") else x) for x in args[:3]) + (
                            cal.detach().cpu().clone() if hasattr(cal, "detach") else cal,)
                    return None
                self.register_forward_pre_hook(pre, with_kwargs=True)
                self.backbone.register_forward_hook(lambda _m, _i, o: setattr(cap, "fwd_dtype", str(o.dtype)))

    Opt0 = torch.optim.AdamW

    class RecAdamW(Opt0):
        def __init__(self, params, *a_, **k_):
            super().__init__(params, *a_, **k_)
            cap.opt = {"lr": self.defaults["lr"], "weight_decay": self.defaults["weight_decay"], "fused": self.defaults.get("fused"),
                       "n_tensors": sum(len(g["params"]) for g in self.param_groups)}

        def zero_grad(self, *a_, **k_):
            r = super().zero_grad(*a_, **k_)
            if mode == "build" and cap.steps is None:
                cap.steps = a.steps
                cap.seed = torch.initial_seed()
                cap.tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
                cap.device = next(cap.model.parameters()).device.type
                cap.compiled = getattr(cap.model.backbone, "_compiled_call_impl", None) is not None
                raise Stop()
            return r

    Sched0 = torch.optim.lr_scheduler.CosineAnnealingLR

    class RecSched(Sched0):
        def __init__(self, optimizer, T_max, *a_, **k_):
            super().__init__(optimizer, T_max, *a_, **k_)
            cap.sched = {"cls": "CosineAnnealingLR", "T_max": T_max, "eta_min": self.eta_min}

    def plain(v):
        return v if isinstance(v, (str, int, float, type(None), bool)) else type(v).__name__

    def wrap_bank(name):
        base = getattr(T, name)

        class W(base):
            def __init__(self, *a_, **k_):
                cap.banks.setdefault(name, []).append({"args": [plain(x) for x in a_], "kwargs": {k: plain(v) for k, v in k_.items()}})
                super().__init__(*a_, **k_)
                if hasattr(self, "scene_groups"):                       # the TargetBank: what the schedule is sized on (G-CLOCK)
                    cap.bank_len[name] = {"n_tuples": len(self.rows), "n_scenes": len(self.scene_groups)}
        W.__name__ = name
        return W

    def wrap_sampler(name):
        base = getattr(T, name)

        class W(base):
            def __init__(self, *a_, **k_):
                super().__init__(*a_, **k_)
                cap.samplers.append({"cls": name, "n": int(self.n)})
        W.__name__ = name
        return W

    DL0 = T.DataLoader

    class RecDL(DL0):
        def __init__(self, ds, *a_, **k_):
            cap.loaders.append({"batch_size": k_.get("batch_size"), "num_workers": k_.get("num_workers")})
            super().__init__(ds, *a_, **k_)

    ws0, snap0, lstate0, mib0 = T.warm_start, T.bank_snapshot, LD.load_state, LD.map_into_backbone
    wta0 = T.wta_loss

    def rec_wta(traj, target, *a_, **k_):
        if cap.wta_yaw_wrap is None:                                    # P6 / D5: the yaw_wrap the loss ACTUALLY received (G-DVB's observed value)
            cap.wta_yaw_wrap = bool(k_.get("yaw_wrap", True))
        return wta0(traj, target, *a_, **k_)

    def rec_warm(model, path):
        cap.warm = path
        return ws0(model, path)

    def rec_snap(*a_, **k_):
        cap.bank_calls["bank_snapshot"] = cap.bank_calls.get("bank_snapshot", 0) + 1
        return snap0(*a_, **k_)

    def rec_state(p):
        cap.trunk_dir = str(p)
        return lstate0(p)

    def rec_map(bb, sd):
        r = mib0(bb, sd)
        cap.trunk_loaded = [len(r[0]), len(r[1]), len(r[2])]
        return r

    def step_pre(optimizer, _a, _k):
        if cap.model is None or not isinstance(optimizer, Opt0):
            return
        rec: dict = {}
        for n, p in cap.model.named_parameters():
            r = rec.setdefault(group_of(n), {"abs_sum": 0.0, "finite": True, "n": 0, "n_grad": 0, "n_zero": 0, "numel": 0, "n_req": 0})
            r["n"] += 1
            r["numel"] += p.numel()
            r["n_req"] += int(p.requires_grad)
            if p.grad is not None:
                s = float(p.grad.detach().double().abs().sum())
                r["n_grad"] += 1
                r["abs_sum"] += s
                r["n_zero"] += int(s == 0.0)
                r["finite"] = r["finite"] and bool(torch.isfinite(p.grad).all())
        cap.grads.append(rec)

    handle = None
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        patch(T, "REFe", Rec)
        patch(torch.optim, "AdamW", RecAdamW)
        patch(torch.optim.lr_scheduler, "CosineAnnealingLR", RecSched)
        for nm in ("TargetBank", "OnPolicyBank", "ScorerBank"):
            patch(T, nm, wrap_bank(nm))
        for nm in ("SceneEpochSampler", "EpochSampler"):
            patch(T, nm, wrap_sampler(nm))
        patch(T, "DataLoader", RecDL)
        patch(T, "wta_loss", rec_wta)
        import online_labeller as OLm
        OL0 = OLm.OnlineLabeller

        class RecLab(OL0):                                              # P6 / D1: record what the labeller was BUILT with
            def __init__(self, *a_, **k_):
                cap.banks.setdefault("OnlineLabeller", []).append({"args": [plain(x) for x in a_], "kwargs": {k: plain(v) for k, v in k_.items()}})
                super().__init__(*a_, **k_)
        RecLab.__name__ = "OnlineLabeller"
        patch(OLm, "OnlineLabeller", RecLab)
        patch(T, "warm_start", rec_warm)
        patch(T, "bank_snapshot", rec_snap)
        patch(LD, "load_state", rec_state)
        patch(LD, "map_into_backbone", rec_map)
        if arm_is("zero_score", arm):                 # the score term silently contributes nothing
            patch(torch.nn.functional, "binary_cross_entropy_with_logits",
                  lambda x, *a_, **k_: (x * 0.0).sum() if k_.get("reduction") != "none" else x * 0.0)
        handle = optim_mod.register_optimizer_step_pre_hook(step_pre)
        with open(cap.log, "w", encoding="utf-8") as lg, contextlib.redirect_stdout(lg), contextlib.redirect_stderr(lg):
            try:
                cap.rc = T.run(a)
            except Stop:
                cap.rc = "STOPPED_AT_BUILD"
            except Exception:                                                # noqa: BLE001 -- recorded, judged by the check
                cap.rc, cap.error = "EXCEPTION", traceback.format_exc()[-1800:]
    finally:
        if handle is not None:
            handle.remove()
        for obj, name, orig in reversed(patches):
            setattr(obj, name, orig)
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
    if a.out:
        mp = Path(a.out) / "metrics.jsonl"
        if mp.exists():
            for ln in mp.read_text(encoding="utf-8").splitlines():
                try:
                    cap.metrics.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        sp = Path(a.out) / "summary.json"
        if sp.exists():
            cap.summary = json.loads(sp.read_text(encoding="utf-8"))
    return cap


# ============================================================================================ evidence
def evidence(check: str, status: str, reasons: list, details: dict, *, arm=None, rehearsal=False, inputs_read=None) -> dict:
    return {"check": check, "status": status, "reasons": reasons, "details": details, "arm": arm, "rehearsal": bool(rehearsal),
            "inputs_read": inputs_read or {}, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def verdict(check, reasons, details, *, arm=None, **kw):
    return evidence(check, "FAIL" if reasons else "PASS", reasons, details, arm=arm or os.environ.get(ARM_ENV), **kw)


# ============================================================================================ G-HYG
def _train_run_source() -> str:
    return (HERE / "train.py").read_text(encoding="utf-8")


NAMESPACE_FUNCS = ("run", "run_identity", "_g_probe")        # the trainer functions whose parameter `a` IS the argparse namespace


def namespace_reads(src: str) -> set:
    """Every `a.<attr>` the trainer's namespace-taking functions read (AST, nested closures included), i.e. every namespace attribute the
    trainer depends on.  Scoped to those functions: elsewhere `a` is an ordinary local (e.g. `a.shape`)."""
    tree = ast.parse(src)
    out = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in NAMESPACE_FUNCS:
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "a":
                    out.add(node.attr)
    return out


def check_hyg(arm=None, rehearsal=False) -> dict:
    import train as T
    from model import REFeConfig
    reasons, det = [], {}
    cfg = REFeConfig.for_backbone("vits16")
    removed = arm_is("no_cfg_guard", arm)
    orig = REFeConfig.__dict__.get("__setattr__")
    if removed:                                       # the D0 guard removed (a mutation of model.py)
        REFeConfig.__setattr__ = object.__setattr__
    try:
        try:
            cfg._g_hyg_probe_undeclared = 1
            reasons.append("REFeConfig ACCEPTED an undeclared attribute (the D0 guard is absent)")
        except AttributeError:
            det["cfg_refuses_undeclared"] = True
        det["foreign_in_default_cfg"] = sorted(set(vars(REFeConfig.for_backbone("vits16"))) - set(REFeConfig.__dataclass_fields__))
        if det["foreign_in_default_cfg"]:
            reasons.append(f"config carries non-field attributes {det['foreign_in_default_cfg']}")
    finally:
        if removed:
            REFeConfig.__setattr__ = orig
    src = _train_run_source()
    if arm_is("undeclared_namespace_read", arm):
        src += "\n\ndef _g_probe(a):\n    return a.undeclared_lever_g_hyg\n"
    dests = {x.dest for x in T.build_parser()._actions if x.dest != "help"}
    reads = namespace_reads(src)
    undeclared = sorted(reads - dests)
    det["namespace_reads"] = len(reads)
    det["undeclared_namespace_reads"] = undeclared
    if undeclared:
        reasons.append(f"the trainer reads namespace attributes its parser does not declare: {undeclared}")
    return verdict("G-HYG", reasons, det, arm=arm, rehearsal=rehearsal)


# ============================================================================================ G-DVB
# kind: "built" (the observed value is compared), "consumed" (static: the trainer reads it; observed only through its effect), "run" (run
# management; behaviour pinned by G-CKPT), "control" (an ablation/test switch: FORBIDDEN in a real launch argv, see the profile)
DVB_KINDS = {
    "backbone": "built", "weights": "built", "targets": "built", "scorer_targets": "built", "images": "built", "calib": "built",
    "max_missing_images": "built", "grow": "built", "grow_scenes": "built", "synthetic": "built", "overfit": "control",
    "overfit_n": "control", "overfit_thresh": "control", "spread": "control", "break_optimizer": "control", "steps": "built",
    "batch": "built", "accum": "built", "n_cameras": "built", "undistort": "built", "detach_scorer_context": "built", "lr": "built",
    "weight_decay": "built", "cosine": "built", "epoch_unit": "built", "epochs": "built", "score_w": "consumed", "log_every": "run",
    "seed": "built", "cpu": "built", "amp": "built", "tf32": "built", "fused_adam": "built", "compile": "built", "workers": "built",
    "out": "run", "resume": "run", "ckpt_every_min": "run", "ckpt_every_steps": "run", "halt_after_steps": "control",
    "scorer_mode": "built", "onpolicy_targets": "built", "declare_change": "control", "init_from": "built", "pdm_labels": "built",
    "pdm_only": "built", "preflight": "control",
    # P6 (SPEC_P6 section 2): each new lever has its observed row in `dvb_rows`
    "yaw_wrap": "built", "nav_mode": "built", "labeller": "built", "label_backend": "built", "label_workers": "built", "label_timeout_s": "built",
    "label_python": "built", "metric_cache": "built",
}
DVB_REASON = {"score_w": "weights the score BCE inside run(); observed through the live `score` term (G-LIVE), statically traced here",
              "log_every": "logging cadence", "out": "output directory", "resume": "supervised relaunch",
              "ckpt_every_min": "checkpoint cadence (resume behaviour pinned by G-CKPT)", "ckpt_every_steps": "checkpoint cadence (pinned by G-CKPT)",
              "grow_scenes": "enters the schedule only under --grow (re-derived in the `steps` row)",
              "accum": "enters the schedule (re-derived in the `steps` row)", "epochs": "enters the schedule (the `steps` row)"}


def _close(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) or isinstance(y, float):
        return math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=0.0)
    return x == y


def dvb_rows(a, b, s) -> list:
    """{flag, declared, built, how, ok} for every observable lever.  `b` = BUILD capture (real levers), `s` = SMOKE capture or None."""
    from model import BACKBONES, REFeConfig
    from load_dinov3 import BACKBONE_ROOT
    rows = []

    import train as _T
    opt_of = {x.dest: x.option_strings[0] for x in _T.build_parser()._actions if x.option_strings}

    def add(dest, declared, built, how):
        rows.append({"dest": dest, "flag": opt_of.get(dest, "--" + dest.replace("_", "-")), "declared": declared, "built": built, "how": how,
                     "ok": _close(declared, built)})
    cfg = b.cfg
    tb = (b.banks.get("TargetBank") or [{"args": [], "kwargs": {}}])[0]
    ta, tk = tb["args"], tb["kwargs"]
    ob = (b.banks.get("OnPolicyBank") or [None])[0]
    sb = (b.banks.get("ScorerBank") or [{"args": [None]}])[0]
    bb = BACKBONES[a.backbone]
    add("backbone", a.backbone, cfg.backbone, "cfg.backbone")
    add("backbone", (bb["width"], bb["depth"], bb["heads"]), (cfg.width, cfg.depth, cfg.heads), "cfg width/depth/heads == BACKBONES[...]")
    add("weights", None if a.weights == "none" else (a.weights or os.path.join(BACKBONE_ROOT, bb["weights"])), b.trunk_dir,
        "load_dinov3.load_state(path)")
    add("targets", a.targets, ta[0] if ta else None, "TargetBank(targets)")
    add("images", a.images, ta[1] if len(ta) > 1 else None, "TargetBank(images)")
    add("calib", a.calib, tk.get("calib"), "TargetBank(calib=)")
    add("max_missing_images", a.max_missing_images, tk.get("max_missing_frac"), "TargetBank(max_missing_frac=)")
    add("synthetic", bool(a.synthetic), bool(tk.get("synthetic")), "TargetBank(synthetic=)")
    add("scorer_targets", a.scorer_targets if (a.scorer_mode != "onpolicy" and a.labeller != "online") else None, sb["args"][0], "ScorerBank(path)")
    # ---- P6 levers: the BUILT value is read off the built objects, never off the namespace
    lb = (b.banks.get("OnlineLabeller") or [None])[0]
    add("labeller", a.labeller, "online" if lb is not None else "bank", "an OnlineLabeller is built iff --labeller online")
    if tb.get("kwargs") is not None:
        add("labeller", a.labeller == "online", bool(tk.get("online")), "TargetBank(online=)")
    if lb is not None:
        lk = lb["kwargs"]
        add("label_backend", a.label_backend, lk.get("backend"), "OnlineLabeller(backend=)")
        add("label_workers", a.label_workers, lk.get("workers"), "OnlineLabeller(workers=)")
        add("label_timeout_s", a.label_timeout_s, lk.get("timeout_s"), "OnlineLabeller(timeout_s=)")
        add("label_python", a.label_python or None, lk.get("python"), "OnlineLabeller(python=)")
        add("metric_cache", a.metric_cache or None, lk.get("metric_cache"), "OnlineLabeller(metric_cache=)")
    has_cmd, has_goal = bool(b.nav_modules.get("cmd_embed")), bool(b.nav_modules.get("goal_freqs"))
    add("nav_mode", a.nav_mode, cfg.nav_mode, "cfg.nav_mode")
    add("nav_mode", a.nav_mode, "command" if (has_cmd and not has_goal) else "goal" if (has_goal and not has_cmd) else "BOTH-OR-NEITHER",
        "which navigation encoder the model BUILT (cmd_embed xor goal_freqs)")
    if s is not None and s.wta_yaw_wrap is not None:
        add("yaw_wrap", a.yaw_wrap == "on", bool(s.wta_yaw_wrap), "the yaw_wrap kwarg the loss function actually received")
    add("scorer_mode", a.scorer_mode, "onpolicy" if ob is not None else "fixed", "an OnPolicyBank is built iff onpolicy")
    if a.scorer_mode == "onpolicy":
        add("onpolicy_targets", a.onpolicy_targets, ob["args"][0] if ob else None, "OnPolicyBank(path)")
        add("pdm_labels", a.pdm_labels or None, ob["kwargs"].get("pdm_labels") if ob else None, "OnPolicyBank(pdm_labels=)")
        add("pdm_only", bool(a.pdm_only), bool(ob["kwargs"].get("pdm_only")) if ob else None, "OnPolicyBank(pdm_only=)")
    add("init_from", a.init_from, b.warm, "warm_start(path)")
    add("grow", bool(a.grow), b.bank_calls.get("bank_snapshot", 0) > 0, "bank_snapshot() is called iff --grow")
    add("n_cameras", a.n_cameras if a.n_cameras is not None else REFeConfig.for_backbone(a.backbone).n_cameras, cfg.n_cameras, "cfg.n_cameras")
    add("undistort", bool(a.undistort), bool(cfg.undistort), "cfg.undistort")
    add("detach_scorer_context", bool(a.detach_scorer_context), bool(cfg.detach_scorer_context), "cfg.detach_scorer_context")
    add("lr", a.lr, b.opt["lr"], "AdamW lr")
    add("weight_decay", a.weight_decay, b.opt["weight_decay"], "AdamW weight_decay")
    add("fused_adam", bool(a.fused_adam and b.device == "cuda"), bool(b.opt["fused"]), "AdamW fused (cuda only)")
    add("cosine", bool(a.cosine), b.sched is not None, "a CosineAnnealingLR is built iff cosine")
    add("epoch_unit", "SceneEpochSampler" if a.epoch_unit == "scenes" else "EpochSampler",
        b.samplers[0]["cls"] if b.samplers else None, "sampler class")
    add("batch", a.batch, b.loaders[0]["batch_size"] if b.loaders else None, "DataLoader batch_size")
    add("workers", a.workers, b.loaders[0]["num_workers"] if b.loaders else None, "DataLoader num_workers")
    # the schedule: re-derived here, independently of run()
    n = b.samplers[0]["n"] if b.samplers else 0
    eff = max(a.batch, 1) * max(a.accum, 1)
    sched_len = a.grow_scenes if a.grow else n
    want_steps = a.epochs * max(math.ceil(sched_len / eff), 1) if a.epochs else a.steps
    add("steps", want_steps, b.steps, "steps re-derived as epochs x ceil(epoch_len / (batch x accum))")
    if a.cosine:
        add("steps", want_steps, b.sched["T_max"] if b.sched else None, "cosine T_max spans the whole run")
        add("cosine", 0.0, b.sched["eta_min"] if b.sched else None, "cosine decays to exactly zero")
    add("seed", a.seed, b.seed, "torch.initial_seed()")
    add("tf32", bool(a.tf32), bool(b.tf32), "torch.backends.cuda.matmul.allow_tf32")
    add("cpu", "cpu" if (a.cpu or not _cuda()) else "cuda", b.device, "device of the model")
    add("compile", bool(a.compile and b.device == "cuda"), bool(b.compiled), "backbone compiled (cuda only)")
    if s is not None and s.fwd_dtype is not None:
        add("amp", "torch.bfloat16" if (a.amp == "bf16" and not a.cpu and _cuda()) else "torch.float32", s.fwd_dtype,
            "dtype of the backbone's output under autocast")
    return rows


def static_consumed(src: str, dest: str) -> bool:
    return dest in namespace_reads(src)


def argv_flag_dest(parser, token: str):
    for act in parser._actions:
        if token in act.option_strings:
            return act.dest
    return None


def check_dvb(argv: list, b, s, profile: dict, arm=None, rehearsal=False) -> dict:
    """b = BUILD capture on the launch argv; s = SMOKE capture (amp observation) or None."""
    import train as T
    reasons, det = [], {}
    parser = T.build_parser()
    dests = {x.dest for x in parser._actions if x.dest != "help"}
    table = dict(DVB_KINDS)
    if arm_is("unknown_flag", arm):
        dests = dests | {"g_probe_new_flag"}
    if arm_is("stale_registry", arm):
        table["g_probe_removed_flag"] = "built"
    unknown = sorted(dests - set(table))
    stale = sorted(set(table) - dests)
    det["parser_flags"], det["registry_flags"] = len(dests), len(table)
    det["unknown_flags"], det["stale_registry"] = unknown, stale
    if unknown:
        reasons.append(f"trainer flags with NO declared-vs-built entry (a new lever needs its check): {unknown}")
    if stale:
        reasons.append(f"registry entries for flags the parser no longer has: {stale}")
    if b.parser_unknown:
        reasons.append(f"argv declares lever(s) the trainer does not build (its parser rejects them): {b.parser_unknown}")
        det["rejected_by_parser"] = b.parser_unknown
        return verdict("G-DVB", reasons, det, arm=arm, rehearsal=rehearsal)
    passed = [(t, argv_flag_dest(parser, t)) for t in argv if t.startswith("--")]
    det["passed_flags"] = [t for t, _ in passed]
    for t, d in passed:
        if d is None or d not in table:
            reasons.append(f"argv flag {t} is not covered by the registry")
        elif t in profile.get("forbidden_flags", ()):
            reasons.append(f"argv flag {t} is FORBIDDEN in a launch ({profile['name']}): "
                           f"{'a test/ablation control' if table[d] == 'control' else 'profile rule'}")
    if b.rc != "STOPPED_AT_BUILD":
        reasons.append(f"the BUILD run did not reach the optimiser: rc={b.rc} {str(b.error or '')[-300:]}")
        det["build_log"] = b.log
        return verdict("G-DVB", reasons, det, arm=arm, rehearsal=rehearsal)
    rows = dvb_rows(b.args, b, s)
    det["rows"] = rows
    for r in rows:
        if not r["ok"]:
            reasons.append(f"{r['flag']}: declared {r['declared']!r} != built {r['built']!r} ({r['how']})")
    src = _train_run_source()
    for d, k in table.items():
        if k == "consumed" and d in dests and not static_consumed(src, d):
            reasons.append(f"--{d.replace('_', '-')} is never read by the trainer (a dead lever)")
    det["kinds"] = {k: sum(1 for v in table.values() if v == k) for k in ("built", "consumed", "run", "control")}
    det["observed_rows"] = len(rows)
    det["not_observed_reasons"] = DVB_REASON
    return verdict("G-DVB", reasons, det, arm=arm, rehearsal=rehearsal)


# ============================================================================================ G-LIVE
def zero_loss_control(cap) -> dict:
    """The probe's own control: on the first REAL batch, a zero loss leaves EVERY parameter's gradient exactly 0.0 (or None), and a real
    loss leaves a non-zero one -- so a zero reading anywhere else is a property of the model, not of a broken probe."""
    m = cap.model
    if m is None or cap.first_batch is None:
        return {"ran": False}
    dev = next(m.parameters()).device
    img, ego, goal, calib = cap.first_batch
    img, ego, goal = img[:1].to(dev), ego[:1].to(dev), goal[:1].to(dev)
    cal = calib[:1].to(dev) if calib is not None and getattr(calib, "numel", lambda: 0)() else None
    m.train()
    out = {}
    for tag, scale in (("zero", 0.0), ("real", 1.0)):
        m.zero_grad(set_to_none=True)
        traj, score = m(img, ego, goal, calib=cal)[:2]
        ((traj.float().sum() + score.float().sum()) * scale).backward()
        out[tag] = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.zero_grad(set_to_none=True)
    out["ran"] = True
    return out


def check_live(argv: list, cap, profile: dict, arm=None, rehearsal=False, control: bool = True) -> dict:
    reasons, det = [], {}
    if cap.rc != 0:
        reasons.append(f"the SMOKE trainer run did not finish: rc={cap.rc} {str(cap.error or '')[-300:]}")
        det["smoke_log"] = cap.log
        return verdict("G-LIVE", reasons, det, arm=arm, rehearsal=rehearsal)
    steps = [r for r in cap.metrics if "traj_L1" in r]
    det["logged_step_rows"] = len(steps)
    if not steps:
        reasons.append("no logged step rows")
    terms = list(profile["live_terms"])
    for t in terms:
        miss = [r.get("step") for r in steps if t not in r]
        bad = [r.get("step") for r in steps if t in r and not _finite(r[t])]
        if miss:
            reasons.append(f"declared loss term {t!r} ABSENT from the trainer's metrics on steps {miss[:4]}")
        if bad:
            reasons.append(f"declared loss term {t!r} not finite on steps {bad[:4]}")
    det["terms_checked"] = terms
    if not cap.grads:
        reasons.append("no optimiser step was observed")
    else:
        agg: dict = {}
        for rec in cap.grads:
            for g, r in rec.items():
                x = agg.setdefault(g, {"max_abs_sum": 0.0, "finite": True, "n": r["n"], "numel": r["numel"], "n_req": r["n_req"], "grad_seen": 0})
                x["max_abs_sum"] = max(x["max_abs_sum"], r["abs_sum"])
                x["finite"] = x["finite"] and r["finite"]
                x["grad_seen"] += r["n_grad"]
        det["groups"] = agg
        trainable = TRAINABLE_GROUPS + (("cmd_embed",) if getattr(cap.cfg, "nav_mode", "goal") == "command" else ())
        for g in sorted(set(agg) - set(trainable) - set(FROZEN_GROUPS)):
            reasons.append(f"parameter group {g!r} is neither declared trainable nor declared frozen (a module was added without a declaration)")
        for g in trainable:
            x = agg.get(g)
            if x is None or x["numel"] == 0:
                reasons.append(f"declared trainable group {g!r} is absent from the model")
            elif x["n_req"] != x["n"]:
                reasons.append(f"declared trainable group {g!r} is FROZEN ({x['n_req']}/{x['n']} tensors require grad)")
            elif not x["finite"]:
                reasons.append(f"group {g!r}: non-finite gradient")
            elif not x["max_abs_sum"] > 0.0:
                reasons.append(f"group {g!r}: gradient is EXACTLY zero on every observed optimiser step ({x['numel']:,} parameters never trained)")
        for g in FROZEN_GROUPS:
            x = agg.get(g)
            if x is not None and (x["grad_seen"] != 0 or x["n_req"] != 0):
                reasons.append(f"declared-frozen group {g!r} received gradient / requires grad (its control must read exactly 0)")
        det["frozen_control_exact_zero"] = all(agg.get(g, {"grad_seen": 0})["grad_seen"] == 0 for g in FROZEN_GROUPS)
    if cap.args.scorer_mode == "onpolicy" or getattr(cap.args, "labeller", "bank") == "online":
        cov = steps[-1].get("scorer_cov") if steps else None
        det["scorer_cov_last"] = cov
        if not (cov and cov > 0):
            reasons.append("on-policy mode: no labelled set ever reached the loss (scorer_cov == 0): the scorer groups were not exercised")
    if control and not reasons:
        try:
            c = zero_loss_control(cap)
            det["zero_loss_control"] = c
            if c.get("ran") and not (c["zero"] == 0.0 and c["real"] > 0.0):
                reasons.append(f"probe control failed: zero loss read {c['zero']} (must be exactly 0.0), real loss read {c['real']} (must be > 0)")
        except Exception as e:                                                        # noqa: BLE001
            reasons.append(f"probe control could not run: {type(e).__name__}: {e}")
    return verdict("G-LIVE", reasons, det, arm=arm, rehearsal=rehearsal)


# ============================================================================================ G-EVAL
PLANNER_ALLOWED_CFG_DIFF = ("detach_scorer_context",)      # training-only: it changes gradients, never a forward value (proved below)


def check_eval(argv: list, cap, profile: dict, arm=None, rehearsal=False) -> dict:
    """The planner's own load sequence (planner.py: REFeConfig.for_backbone -> ckpt_io.config_for_checkpoint -> REFe -> ckpt_io.load_for_inference)
    on the SMOKE run's model_final.pt: same config, same tensors as the trained model, same forward outputs as the TRAINER's own load."""
    import torch
    import ckpt_io
    import train as T
    from model import REFe, REFeConfig
    reasons, det = [], {}
    fin = Path(cap.run_dir or ".") / "model_final.pt"
    if not fin.exists() or cap.model is None or cap.first_batch is None:
        return verdict("G-EVAL", [f"no model_final.pt / trained model / first batch to compare (rc={cap.rc})"], det, arm=arm, rehearsal=rehearsal)
    src = (HERE / "planner.py").read_text(encoding="utf-8")
    for needle in ("REFeConfig.for_backbone(backbone)", "ckpt_io.config_for_checkpoint(self.cfg, checkpoint)", "REFe(self.cfg)",
                   "ckpt_io.load_for_inference(self.model, checkpoint"):
        if needle not in src:
            reasons.append(f"planner.py no longer contains its load step {needle!r}: this gate emulates a loader that has changed")
    pcfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(cap.args.backbone), str(fin))
    allowed = tuple(PLANNER_ALLOWED_CFG_DIFF) + tuple(profile.get("eval_cfg_allowed", ()))
    for f in profile.get("eval_cfg_allowed", ()):       # REHEARSAL ONLY: the planner cannot build a reduced rig, so the rehearsal gives it the trained one
        setattr(pcfg, f, getattr(cap.cfg, f))
    if arm_is("planner_cfg_differs", arm):
        pcfg.undistort = not pcfg.undistort            # the planner builds a different forward than the trainer did
    tcfg = cap.cfg
    diff = sorted(f for f in REFeConfig.__dataclass_fields__ if getattr(tcfg, f) != getattr(pcfg, f))
    det["cfg_fields_differing"] = diff
    bad = [f for f in diff if f not in allowed]
    if bad:
        reasons.append("the planner builds a different config from the trainer's on " + ", ".join(
            f"{f}: trained {getattr(tcfg, f)!r} / planner {getattr(pcfg, f)!r}" for f in bad[:4]))
    pm = REFe(pcfg)
    ckpt_io.load_for_inference(pm, str(fin), map_location="cpu", backbone=cap.args.backbone, meta_out={})
    if arm_is("loader_drops_tensor", arm):             # a loader that quietly skips one tensor
        with torch.no_grad():
            next(p for n, p in pm.named_parameters() if group_of(n) == "dec").add_(1.0)
    ts = {k: v.detach().cpu() for k, v in cap.model.state_dict().items()}
    ps = {k: v.detach().cpu() for k, v in pm.state_dict().items()}
    det["state_keys"] = {"trainer": len(ts), "planner": len(ps)}
    if set(ts) != set(ps):
        reasons.append(f"state-dict keys differ: trainer-only {sorted(set(ts) - set(ps))[:3]}, planner-only {sorted(set(ps) - set(ts))[:3]}")
    neq = sorted(k for k in set(ts) & set(ps) if not torch.equal(ts[k], ps[k]))
    det["tensors_not_bit_identical"] = len(neq)
    if neq:
        reasons.append(f"{len(neq)} tensor(s) differ between the trained model and the planner's load, e.g. {neq[:3]}")
    tm = REFe(copy.deepcopy(tcfg))
    T.warm_start(tm, str(fin))                          # the TRAINER's own load of the same file (p2_gate G1's pairing; uncompiled)
    img, ego, goal, calib = cap.first_batch
    dev = "cuda" if _cuda() else "cpu"
    cal = calib.to(dev) if calib is not None and getattr(calib, "numel", lambda: 0)() else None
    outs = []
    for m in (tm, pm):
        m.to(dev).eval()
        with torch.no_grad():
            r = m(img.to(dev), ego.to(dev), goal.to(dev), calib=cal)
        outs.append([x.float().cpu() for x in r])
        m.cpu()
    dmax = [float((x - y).abs().max()) for x, y in zip(*outs)]
    det["forward_max_abs_diff"] = dmax
    if any(d != 0.0 for d in dmax):
        reasons.append(f"forward outputs on the first real batch are not bit-identical (max |diff| {dmax})")
    return verdict("G-EVAL", reasons, det, arm=arm, rehearsal=rehearsal)


# ============================================================================================ G-CKPT
def _set(argv, flag, value):
    out, i, done = [], 0, False
    while i < len(argv):
        if argv[i] == flag:
            out += [flag, value]
            i += 2
            done = True
        else:
            out.append(argv[i])
            i += 1
    return out if done else out + [flag, value]


def _drop(argv, flag, nvalues: int = 0):
    """argv without `flag` and its `nvalues` following tokens."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1 + nvalues
        else:
            out.append(argv[i])
            i += 1
    return out


def _train_sub(args: list, log: Path) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    env.pop(ARM_ENV, None)
    r = subprocess.run([sys.executable, str(HERE / "train.py")] + args, cwd=str(HERE), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log.write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode


def _digest_sd(sd) -> dict:
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k, v in sd.items()}


def _opt_digest(o: dict) -> str:
    h = hashlib.sha256()
    for pid in sorted(o.get("state", {})):
        for k, v in sorted(o["state"][pid].items()):
            h.update(f"{pid}:{k}:".encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes() if hasattr(v, "detach") else repr(v).encode())
    h.update(json.dumps(o.get("param_groups"), sort_keys=True, default=str).encode())
    return h.hexdigest()


def check_ckpt(argv: list, work, profile: dict, arm=None, rehearsal=False) -> dict:
    """U: n steps uninterrupted.  H: halt at the checkpoint after step n/2, then --resume to n.  Model, optimiser state, data position
    and the logged loss trajectory must be IDENTICAL.  `argv` is the SMOKE argv."""
    import torch
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    reasons, det = [], {}
    n = int(profile["ckpt_steps"])
    h = n // 2

    def with_out(tag, extra):
        # --compile is dropped: the resume contract is independent of it, and three compiles would cost ~6 min on the pod
        return _drop(_drop(_set(_set(argv, "--out", str(work / tag)), "--steps", str(n)), "--resume"), "--compile") + extra
    rc_u = _train_sub(with_out("U", []), work / "U.log")
    rc_h1 = _train_sub(with_out("H", ["--ckpt-every-steps", str(h), "--halt-after-steps", str(h)]), work / "H1.log")
    ck = work / "H" / "ckpt_last.pt"
    det["rc"] = {"U": rc_u, "H1": rc_h1}
    if rc_u != 0 or not ck.exists():
        reasons.append(f"setup failed: U rc {rc_u}, H1 rc {rc_h1}, checkpoint exists {ck.exists()}")
        return verdict("G-CKPT", reasons, det, arm=arm, rehearsal=rehearsal)
    if arm_is("resume_drops_opt", arm):               # the resume starts Adam from zero moments
        st = torch.load(ck, map_location="cpu", weights_only=False)
        for k in st["opt"]["state"]:
            for vv in st["opt"]["state"][k].values():
                if hasattr(vv, "zero_"):
                    vv.zero_()
        torch.save(st, ck)
    rc_h2 = _train_sub(with_out("H", ["--resume"]), work / "H2.log")
    det["rc"]["H2"] = rc_h2
    if rc_h2 != 0:
        reasons.append(f"the resumed run failed (rc {rc_h2}); see H2.log")
        return verdict("G-CKPT", reasons, det, arm=arm, rehearsal=rehearsal)
    du = _digest_sd(torch.load(work / "U" / "model_final.pt", map_location="cpu", weights_only=False)["model"])
    dh = _digest_sd(torch.load(work / "H" / "model_final.pt", map_location="cpu", weights_only=False)["model"])
    bad = sorted(k for k in set(du) | set(dh) if du.get(k) != dh.get(k))
    det["model_tensors"], det["model_tensors_differing"] = len(du), len(bad)
    if bad:
        reasons.append(f"resumed model != uninterrupted model on {len(bad)}/{len(du)} tensors, e.g. {bad[:3]}")
    cu = torch.load(work / "U" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    chh = torch.load(work / "H" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    det["optimizer_state_equal"] = _opt_digest(cu["opt"]) == _opt_digest(chh["opt"])
    if not det["optimizer_state_equal"]:
        reasons.append("optimiser state after the resumed run != the uninterrupted run's")
    pos_u = {k: cu["state"][k] for k in ("step", "epoch", "pos", "micro")}
    pos_h = {k: chh["state"][k] for k in ("step", "epoch", "pos", "micro")}
    det["data_position"] = {"U": pos_u, "H": pos_h}
    if pos_u != pos_h:
        reasons.append(f"data position differs: uninterrupted {pos_u} vs resumed {pos_h}")

    def losses(tag):
        return [r.get("traj_L1") for r in map(json.loads, (work / tag / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) if "traj_L1" in r]
    lu, lh = losses("U"), losses("H")
    det["loss_trajectory_equal"] = lu == lh
    if lu != lh:
        reasons.append(f"logged traj_L1 trajectories differ: {lu[:4]} vs {lh[:4]}")
    return verdict("G-CKPT", reasons, det, arm=arm, rehearsal=rehearsal)


# ============================================================================================ G-SUITE, G-CLOCK
def check_suite(profile: dict, arm=None, rehearsal=False) -> dict:
    reasons, det = [], {"scripts": []}
    entries = [dict(e) for e in profile.get("suite", ())]
    if arm_is("suite_failing_test", arm):
        entries.append({"raw": [sys.executable, "-c", "import sys; print('ZZTEST_FAIL'); sys.exit(1)"], "token": "ZZTEST_PASS"})
    for e in entries:
        argv = e["raw"] if "raw" in e else [sys.executable, str(HERE / e["script"]), *e.get("args", [])]
        token = e["token"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", REFE_GATE_NESTED="1")
        env.pop(ARM_ENV, None)
        r = subprocess.run(argv, cwd=str(HERE), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = r.returncode == 0 and token in r.stdout
        skipped = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZTEST_SKIP")]
        name = e.get("script", "raw")
        det["scripts"].append({"script": name, "rc": r.returncode, "token": token, "token_seen": token in r.stdout, "skips": skipped[:4]})
        if not ok:
            reasons.append(f"{name}: rc {r.returncode}, {token!r} {'seen' if token in r.stdout else 'NOT seen'}")
        if skipped and profile.get("suite_skips_fail", True):
            reasons.append(f"{name}: {len(skipped)} check(s) SKIPPED -- a skip is not a pass ({skipped[0][:100]})")
    return verdict("G-SUITE", reasons, det, arm=arm, rehearsal=rehearsal)


def check_clock(argv: list, b, profile: dict, arm=None, rehearsal=False) -> dict:
    exp = profile.get("expect_schedule")
    if exp is None:
        return evidence("G-CLOCK", "NA", [profile.get("clock_na", "no label clock in this profile")], {}, arm=arm, rehearsal=rehearsal)
    reasons, det = [], {}
    if b.rc != "STOPPED_AT_BUILD":
        reasons.append(f"BUILD run did not reach the optimiser (rc {b.rc})")
    else:
        a = b.args
        epochs = exp["epochs"] + (1 if arm_is("clock_wrong_epochs", arm) else 0)
        det.update({"steps": b.steps, "epochs_flag": a.epochs, "sampler_n": b.samplers[0]["n"] if b.samplers else None})
        if a.epochs != epochs:
            reasons.append(f"realised epochs {a.epochs} != registered {epochs}")
        if b.steps != exp["steps"]:
            reasons.append(f"realised steps {b.steps} != registered {exp['steps']}")
        if b.samplers and b.samplers[0]["n"] != exp["epoch_len"]:
            reasons.append(f"epoch length {b.samplers[0]['n']} != registered {exp['epoch_len']}")
        if "n_tuples" in exp:                         # P6 / D8: the FULL bank from step 0 (no --grow), not a growing one
            bl = (b.bank_len or {}).get("TargetBank") or {}
            det["bank"] = bl
            if bl.get("n_tuples") != exp["n_tuples"]:
                reasons.append(f"step-0 bank has {bl.get('n_tuples')} tuples != registered {exp['n_tuples']}")
            if bl.get("n_scenes") != exp["epoch_len"]:
                reasons.append(f"step-0 bank has {bl.get('n_scenes')} scenes != registered {exp['epoch_len']}")
    return verdict("G-CLOCK", reasons, det, arm=arm, rehearsal=rehearsal)


# ============================================================================================ P6: G-LIVE-LAB, G-LIVE-MEM, G-EGR (POD-RUN: they consume their instruments' output)
# These test components that need the pod (the NAVSIM stack on a D9 cache, a CUDA device, the network). The INSTRUMENTS write raw measurements --
# refe/diag_p6_labeller_identity.py (LAB), the trainer's own metrics.jsonl from a 200-micro-batch run (MEM), refe/hf_egress.py gate (EGR) -- and the gate JUDGES them
# (thresholds live HERE, not in the instrument), binds their sha256 into the token, and never greps. Each judge has a fixture-driven regression arm in test_p6_levers.py.
LAB_NEED_SETS_S, LAB_MARGIN, LAB_STALL_MAX, MEM_MAX_GB, MEM_MIN_MICRO = 4.49, 1.5, 0.05, 44.0, 200


def judge_lab(lab: dict, smoke_metrics: list, workers_launch: int, arm=None) -> list:
    """lab = diag_p6_labeller_identity.py's JSON; smoke_metrics = the SMOKE trainer's metrics rows. Returns the failure reasons."""
    r = []
    if arm_is("lab_two_workers", arm):
        lab = dict(lab, workers=2, throughput_sets_s=lab.get("throughput_sets_s", 0.0) * 2.0 / max(lab.get("workers", 2), 1))
    if not lab.get("identity_ok") or lab.get("n_sets", 0) < 64 or lab.get("identity_equal_sets", 0) < 64:
        r.append(f"D1(a) identity: {lab.get('identity_equal_sets')}/{lab.get('n_sets')} sets bit-identical to the offline labeller (need all, over >= 64 sets)")
    if not lab.get("stale_refused"):
        r.append("D1(c): a one-step-stale proposal was NOT refused")
    bar = LAB_MARGIN * LAB_NEED_SETS_S
    if not lab.get("throughput_sets_s", 0.0) >= bar:
        r.append(f"throughput {lab.get('throughput_sets_s')} sets/s < {bar:.3f} (1.5 x {LAB_NEED_SETS_S})")
    if lab.get("workers") != workers_launch:
        r.append(f"throughput measured at {lab.get('workers')} workers but the launch argv says --label-workers {workers_launch}")
    ev = [m for m in smoke_metrics if m.get("event") == "labeller"]
    if not ev:
        r.append("the SMOKE run logged no `labeller` event (the online labeller did not run)")
    else:
        if any(set(e.get("lag_hist", {})) - {"0"} for e in ev):
            r.append(f"label lag != 0 in the smoke run: {[e['lag_hist'] for e in ev][:2]}")
        sets, lab_n = sum(e["sets"] for e in ev), sum(e["labelled"] for e in ev)
        if sets == 0 or lab_n / sets <= 0.0:
            r.append(f"coverage {lab_n}/{sets}: no set was labelled")
        stall = max(e["stall_frac"] for e in (ev[1:] or ev))        # the first window includes pool start-up; steady state is judged
        if stall > LAB_STALL_MAX:
            r.append(f"GPU stall {100 * stall:.1f} % of a window > {100 * LAB_STALL_MAX:.0f} %")
    return r


def judge_mem(rows: list, arm=None) -> list:
    """rows = metrics.jsonl step rows of a >= 200-micro-batch run on the real card. Peak must be <= 44 GB (SPEC_P6 3.2: two live graphs on a 48 GB A40)."""
    r = []
    sr = [m for m in rows if "cuda_max_mem_gb" in m and "traj_L1" in m]
    if not sr or any(m["cuda_max_mem_gb"] is None for m in sr):
        return ["no CUDA memory reading (cuda_max_mem_gb is null): G-LIVE-MEM needs a real A40, it cannot be judged on CPU"]
    peak = max(m["cuda_max_mem_gb"] for m in sr) + (20.0 if arm_is("mem_over_budget", arm) else 0.0)
    n_micro = max(m.get("samples", 0) for m in sr) / max(sr[-1].get("batch", 1), 1)
    if n_micro < MEM_MIN_MICRO:
        r.append(f"only {n_micro:.0f} micro-batches were observed (< {MEM_MIN_MICRO})")
    if peak > MEM_MAX_GB:
        r.append(f"peak {peak:.2f} GB > {MEM_MAX_GB} GB: drop to micro-batch 2 x accum 128 (SPEC_P6 3.2)")
    return r


def judge_egr(res: dict, local_size=None, remote_size=None, arm=None) -> list:
    """res = hf_egress.py gate's g_egr_result.json; optionally the first ckpt_last.pt push's local vs LISTING size (the spec's check)."""
    r = []
    if res.get("pass") is not True:
        r.append("hf_egress.py gate did not PASS")
    if not (res.get("main") or {}).get("ok"):
        r.append(f"the gate JSON push failed: {(res.get('main') or {}).get('reason')}")
    if not (res.get("regression") or {}).get("refused_by_server"):
        r.append("the deliberate-regression push was not refused BY THE SERVER (our own pre-check proves nothing about the token)")
    if local_size is not None:
        want = local_size + (1 if arm_is("egr_wrong_size", arm) else 0)
        if remote_size != want:
            r.append(f"ckpt_last.pt: remote listing size {remote_size} != expected {want}")
    return r


def _consume(path):
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_live_lab(lab_json, smoke_cap, argv: list, arm=None, rehearsal=False) -> dict:
    import gate_token as GT_
    lab = _consume(lab_json)
    if lab is None:
        return verdict("G-LIVE-LAB", [f"labeller instrument output missing: {lab_json} (run refe/diag_p6_labeller_identity.py on the pod)"], {}, arm=arm,
                       rehearsal=rehearsal)
    w = int((GT_.flag_values(argv, "--label-workers") or ["12"])[0])
    reasons = judge_lab(lab, smoke_cap.metrics if smoke_cap is not None else [], w, arm=arm)
    return verdict("G-LIVE-LAB", reasons, {"lab": lab, "file_sha256": GT_.sha256_file(lab_json)}, arm=arm, rehearsal=rehearsal)


def check_live_mem(mem_metrics, arm=None, rehearsal=False) -> dict:
    import gate_token as GT_
    if not mem_metrics or not Path(mem_metrics).is_file():
        return verdict("G-LIVE-MEM", [f"memory-run metrics missing: {mem_metrics}"], {}, arm=arm, rehearsal=rehearsal)
    rows = [json.loads(ln) for ln in Path(mem_metrics).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return verdict("G-LIVE-MEM", judge_mem(rows, arm=arm), {"rows": len(rows), "file_sha256": GT_.sha256_file(mem_metrics)}, arm=arm, rehearsal=rehearsal)


EGR_CKPT_MIN_BYTES = 50_000_000      # MEASURED ckpt_last.pt of this model class: 228,234,301 B (SPEC_P6 section 4); a probe far below it proves nothing about the real push


def check_egr(egr_json, ckpt_sizes=None, arm=None, rehearsal=False, require_ckpt=True) -> dict:
    """G-EGR. The first-checkpoint size comes from `ckpt_sizes=(local, listing)` if given, else from the `ckpt` record `hf_egress.py gate --ckpt <ckpt_last.pt>` wrote into the
    SAME json (so its sha256, bound into the token, covers the sizes too). `require_ckpt`: a P6 token needs that size check; a result without it FAILS."""
    import gate_token as GT_
    res = _consume(egr_json)
    if res is None:
        return verdict("G-EGR", [f"hf_egress.py gate output missing: {egr_json}"], {}, arm=arm, rehearsal=rehearsal)
    ls, rs = (ckpt_sizes or (None, None))
    ck = res.get("ckpt") or {}
    if ckpt_sizes is None and ck:
        ls, rs = ck.get("local_size"), ck.get("remote_size")
    reasons = judge_egr(res, ls, rs, arm=arm)
    if require_ckpt:
        if ls is None:
            reasons.append("no first-checkpoint push recorded (hf_egress.py gate --ckpt <ckpt_last.pt>): SPEC_P6 section 5's size check was not made")
        elif ls < EGR_CKPT_MIN_BYTES:
            reasons.append(f"the pushed checkpoint is {ls} B < {EGR_CKPT_MIN_BYTES} B: not a real ckpt_last.pt of this model")
        if ck and ck.get("push_ok") is not True:
            reasons.append(f"the first-checkpoint push failed: {ck.get('reason')}")
    return verdict("G-EGR", reasons, {"result": res, "file_sha256": GT_.sha256_file(egr_json)}, arm=arm, rehearsal=rehearsal)
