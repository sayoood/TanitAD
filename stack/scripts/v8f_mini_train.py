#!/usr/bin/env python3
"""v8F-mini training (track M): S -> T -> O on the cached frozen DINOv3-L pooled features. CPU-testable, 8 GB GPU-sized.

Seeded, resumable, writes ``summary.json {"done": true}`` ONLY at the very end (a supervisor that restarts a run whose
done-marker is missing will resurrect it; one that finds the marker leaves it alone).

LAUNCH-GATE DISCIPLINE (the project rule of 2026-09-26): every argparse flag has an entry in :data:`FLAG_CHECKS`; a flag
without one makes ``self_check`` REFUSE the launch (and ``tests/test_v8f_mini.py`` fail). ``declared_vs_built``
compares what argv DECLARES against what the BUILT model/optimiser/data actually are; ``live_check`` runs three
optimiser steps on a real batch and requires every declared loss term finite and every trainable parameter group to
receive a non-zero gradient, against a control (the frozen P10 prior) that must read EXACTLY 0.0.

Arms (PREREG_V8F_MINI.md): M1, M1-navonly, M1-nocons, M2a, M2b are trained arms. ``M1-navshuf`` is an EVALUATION control
(nav taken from another episode): it trains exactly M1 and is accepted here as an alias so the supervisor can name it.
H-DATA-SCALE-1: ``--data-frac {0.25,0.5,1.0}`` and ``--steps-mult {1,2,4}``; S = one epoch of windows at 100 % of the
TRAIN clips at the chosen batch size, the optimiser-step budget is ``S * steps-mult`` regardless of the data fraction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import queue
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")

ARM_SPECS = {
    "M1": dict(navonly=False, consistency=True, wm="off"),
    "M1-navshuf": dict(navonly=False, consistency=True, wm="off"),      # trained as M1; the control is at EVAL time
    "M1-navonly": dict(navonly=True, consistency=True, wm="off"),
    "M1-nocons": dict(navonly=False, consistency=False, wm="off"),
    "M2a": dict(navonly=False, consistency=True, wm="a"),
    "M2b": dict(navonly=False, consistency=True, wm="b"),
}

#: dest -> how it is verified. "cfg:<field>" = equals the BUILT config; "lever:<k>" = equals built_levers()[k];
#: "opt:<k>" = read back from the optimiser / scheduler; "data:<k>" = read back from the built data; "run" = a
#: run-control value recorded in summary.json; "path" = an input/output location asserted to exist (or be created).
FLAG_CHECKS = {
    "arm": "lever:arm", "navonly": "lever:navonly", "no_consistency": "lever:consistency", "wm": "lever:wm",
    "cache_dir": "path", "out": "path", "v9_train": "path", "v9_train_md5": "data:v9_md5", "speed_train": "path",
    "lead_train": "path", "join_train": "path", "p10": "path", "grid": "cfg:grid",
    "data_frac": "data:n_clips", "steps_mult": "run", "steps": "run", "batch": "data:batch", "seed": "run",
    "lr": "opt:lr", "wd": "opt:wd", "warmup": "opt:warmup", "device": "run", "amp": "run", "lru": "data:lru",
    "ckpt_every": "run", "log_every": "run", "resume": "run", "no_resume": "run", "max_clips": "data:n_clips",
    "hist": "cfg:hist", "hmax": "cfg:hmax", "d": "cfg:d", "n_q": "cfg:n_q", "nb": "cfg:nb",
    "p_gt_start": "cfg:p_gt_start", "p_gt_end": "cfg:p_gt_end", "w_ckpt": "cfg:w_ckpt", "w_tac": "cfg:w_tac",
    "w_cons": "cfg:w_cons", "w_traj": "cfg:w_traj", "w_pen": "cfg:w_pen", "w_pen_self": "cfg:w_pen_self",
    "w_wm": "cfg:w_wm", "detach_upper": "cfg:detach_upper", "synthetic": "run", "selfcheck_only": "run",
    "fit_join": "run", "workers": "run", "clip_grad": "opt:clip_grad", "threads": "run",
}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARM_SPECS), default="M1")
    ap.add_argument("--navonly", action="store_true", default=None, help="override: S and T lose the scene")
    ap.add_argument("--no-consistency", action="store_true", default=None, help="override: penalties at 0")
    ap.add_argument("--wm", choices=("off", "a", "b"), default=None, help="override: H-JEPA auxiliary (a: no IDM, b: IDM)")
    ap.add_argument("--cache-dir", default="C:/Users/Admin/refav1_hj_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v9-train", default="C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_train.npz")
    ap.add_argument("--v9-train-md5", default="f63ece410b725febb8a5242cf2b01d3c")
    ap.add_argument("--speed-train", default="C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_speed_max_v8_train.jsonl")
    ap.add_argument("--lead-train", default="D:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz")
    ap.add_argument("--join-train", default=None, help="a prebuilt join npz (else built from the v9 release)")
    ap.add_argument("--fit-join", action="store_true", help="(re)build the join even if --join-train exists")
    ap.add_argument("--p10", default=os.path.join(PKG, "raw", "p10", "p10_prior.pt"))
    ap.add_argument("--grid", choices=("pool4x10", "pool8x20"), default="pool4x10")
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--max-clips", type=int, default=0, help="smoke: cap the TRAIN clips (0 = all)")
    ap.add_argument("--steps-mult", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="override the step budget (smoke)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip-grad", dest="clip_grad", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (UNVERIFIED on GPU at this commit)")
    ap.add_argument("--lru", type=int, default=0, help="clips held in RAM (0 = auto from the grid size)")
    ap.add_argument("--workers", type=int, default=1, help="1 = a single prefetch thread")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--hmax", type=int, default=0, help="0 = auto: 10 (M1) or 16 (M2)")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-q", dest="n_q", type=int, default=4)
    ap.add_argument("--nb", type=int, default=12)
    ap.add_argument("--p-gt-start", type=float, default=1.0)
    ap.add_argument("--p-gt-end", type=float, default=0.2)
    ap.add_argument("--w-ckpt", type=float, default=1.0)
    ap.add_argument("--w-tac", type=float, default=1.0)
    ap.add_argument("--w-cons", type=float, default=0.5)
    ap.add_argument("--w-traj", type=float, default=1.0)
    ap.add_argument("--w-pen", type=float, default=0.5)
    ap.add_argument("--w-pen-self", type=float, default=0.25)
    ap.add_argument("--w-wm", type=float, default=0.1)
    ap.add_argument("--detach-upper", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="run on a tiny synthetic world (smoke / CI)")
    ap.add_argument("--selfcheck-only", action="store_true")
    return ap


# ------------------------------------------------------------------------------------------------ declared levers
def declared_levers(a) -> dict:
    spec = dict(ARM_SPECS[a.arm])
    if a.navonly is not None:
        spec["navonly"] = bool(a.navonly)
    if a.no_consistency is not None:
        spec["consistency"] = not a.no_consistency
    if a.wm is not None:
        spec["wm"] = a.wm
    return spec


def make_config(a) -> M.V8FMiniConfig:
    lv = declared_levers(a)
    hmax = a.hmax or (16 if lv["wm"] != "off" else 10)
    return M.V8FMiniConfig(d=a.d, ff=2 * a.d, hist=a.hist, hmax=hmax, n_q=a.n_q, nb=a.nb, navonly=lv["navonly"],
                           consistency=lv["consistency"], wm=lv["wm"], n_tok=40 if a.grid == "pool4x10" else 160,
                           p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end, w_ckpt=a.w_ckpt, w_tac=a.w_tac,
                           w_cons=a.w_cons, w_traj=a.w_traj, w_pen=a.w_pen, w_pen_self=a.w_pen_self, w_wm=a.w_wm,
                           detach_upper=a.detach_upper)


def declared_vs_built(a, cfg, model, opt, data_info) -> list:
    """Return the list of mismatches (empty = pass). Each flag is checked against what was BUILT, by FLAG_CHECKS kind."""
    bad = []
    parser_dests = {act.dest for act in build_parser()._actions if act.dest != "help"}
    missing = sorted(parser_dests - set(FLAG_CHECKS))
    if missing:
        bad.append(f"flags without a declared-vs-built entry (the gate refuses them): {missing}")
    lv = declared_levers(a)
    built = model.built_levers()
    for k, want in (("navonly", lv["navonly"]), ("consistency", lv["consistency"]), ("wm", lv["wm"])):
        if built[k] != want:
            bad.append(f"lever {k}: declared {want} but built {built[k]}")
    if not built["prior_frozen"]:
        bad.append("the P10 prior is not frozen")
    if model.cfg.n_tok != (40 if a.grid == "pool4x10" else 160):
        bad.append(f"grid {a.grid} declared but n_tok built {model.cfg.n_tok}")
    for f in ("hist", "hmax", "d", "n_q", "nb", "p_gt_start", "p_gt_end", "w_ckpt", "w_tac", "w_cons", "w_traj", "w_pen",
              "w_pen_self", "w_wm", "detach_upper"):
        want = {"hist": a.hist, "hmax": cfg.hmax, "d": a.d, "n_q": a.n_q, "nb": a.nb}.get(f, getattr(a, f, None))
        got = getattr(model.cfg, f)
        if want is not None and not (got == want or (isinstance(want, float) and abs(got - want) < 1e-12)):
            bad.append(f"cfg.{f}: declared {want} but built {got}")
    # weights that the loss will actually use must be the declared ones (nocons -> 0)
    if not lv["consistency"] and (built["w"]["w_pen"] != 0.0 or built["w"]["w_pen_self"] != 0.0):
        bad.append("M1-nocons declared but penalty weights are non-zero in the built loss")
    if lv["consistency"] and not built["w"]["w_pen"] > 0.0:
        bad.append("consistency penalties declared ON but w_pen is 0 -- a lever that is declared and does nothing")
    if lv["wm"] != "off" and not built["w"]["w_wm"] > 0.0:
        bad.append("WM auxiliary declared but w_wm is 0 -- a lever that is declared and does nothing")
    if lv["wm"] == "off" and model.wm is not None:
        bad.append("wm declared off but a WM module was built")
    if lv["wm"] != "off" and (model.wm is None or built["wm_idm_gamma"] != (0.0 if lv["wm"] == "a" else cfg.wm_gamma)):
        bad.append(f"wm {lv['wm']} declared but built gamma {built['wm_idm_gamma']}")
    # optimiser read-back
    lrs = {g["lr"] for g in opt.param_groups}
    if lrs != {a.lr}:
        bad.append(f"lr: declared {a.lr} but optimiser holds {lrs}")
    if {g["weight_decay"] for g in opt.param_groups if g["weight_decay"] > 0} not in ({a.wd}, set()):
        bad.append("weight decay not as declared")
    # data read-back
    if data_info["n_clips"] != data_info["n_clips_expected"]:
        bad.append(f"data: {data_info['n_clips']} clips built, {data_info['n_clips_expected']} declared "
                   f"(data-frac {a.data_frac}, max-clips {a.max_clips})")
    if data_info["batch"] != a.batch:
        bad.append("batch size not as declared")
    if data_info.get("v9_md5") and a.v9_train_md5 and data_info["v9_md5"] != a.v9_train_md5 and not data_info.get("md5_overridden"):
        bad.append(f"v9 release md5 {data_info['v9_md5']} != declared {a.v9_train_md5}")
    return bad


def live_check(model, batch, cfg, steps: int = 3, lr: float = 1e-3) -> dict:
    """G-LIVE: three optimiser steps on a real batch (on a COPY). Every declared loss term finite; every trainable
    parameter receives a non-zero gradient at some step, except the declared-dead set; the frozen prior reads exactly 0."""
    m = copy.deepcopy(model)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    acc = {n: 0.0 for n, p in m.named_parameters() if p.requires_grad}
    first = None
    terms_seen = None
    for s in range(steps):
        opt.zero_grad(set_to_none=False)
        out = m.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(s))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"G-LIVE: loss term {k} is not finite at step {s}: {float(v)}")
        terms_seen = {k: float(v.detach()) for k, v in T.items()}
        T["total"].backward()
        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                acc[n] += float(p.grad.abs().sum())
        if s == 0:
            first = {n: float(p.grad.abs().sum()) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
        opt.step()
    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "total"}
    if cfg.consistency:
        expect |= {f"pen_{r}" for r in M.RULES} | {f"pens_{r}" for r in M.RULES}
    if cfg.wm != "off":
        expect.add("wm")
    missing = sorted(expect - set(terms_seen))
    extra_pen = sorted(k for k in terms_seen if k.startswith("pen") and not cfg.consistency)
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    dead = sorted(n for n, g in acc.items() if g == 0.0)
    unexpected_dead = sorted(set(dead) - set(declared_dead))
    declared_dead_alive = sorted(n for n in declared_dead if acc[n] != 0.0)
    ctrl = [float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in m.prior.parameters()]
    head_first = sum(v for n, v in first.items() if n.startswith("head_o."))
    res = {"terms": terms_seen, "missing_terms": missing, "unexpected_terms": extra_pen, "dead_params": dead,
           "declared_dead": declared_dead, "unexpected_dead": unexpected_dead, "declared_dead_but_alive": declared_dead_alive,
           "control_prior_grad_sum": float(sum(ctrl)), "head_o_grad_step0": head_first,
           "n_trainable": len(acc), "groups_grad": {g: float(sum(v for n, v in acc.items() if n.startswith(pre)))
                                                   for g, pre in (("adapter", "adapter"), ("S", "read_s"), ("T", "read_t"),
                                                                  ("O", "read_o"), ("head_s", "head_s"), ("head_o", "head_o"),
                                                                  ("wm", "wm."))}}
    ok = (not missing and not extra_pen and not unexpected_dead and not declared_dead_alive
          and res["control_prior_grad_sum"] == 0.0 and head_first > 0.0)
    res["pass"] = bool(ok)
    if not ok:
        raise AssertionError("G-LIVE failed: " + json.dumps({k: v for k, v in res.items() if k != "groups_grad"}, default=str)[:1500])
    return res


# ------------------------------------------------------------------------------------------------ data
def build_data(a, cfg):
    """-> (store, J, train_windows, clips, info). Joins the v9 release onto the cache grid unless a join npz exists."""
    info = {"batch": a.batch, "md5_overridden": False}
    if a.synthetic:
        root = os.path.join(a.out, "_synthetic")
        w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=a.seed)
        J = D.load_join(os.path.join(root, "join_train.npz"))
        cache = root
        clips_all = w["names"]["train"]
    else:
        cache = a.cache_dir
        man = D.read_manifest(cache, "train")
        clips_all = [c for c, _ in man]
        if a.join_train and os.path.exists(a.join_train) and not a.fit_join:
            J = D.load_join(a.join_train)
        else:
            from tanitad.data.v9_labels import load_v9_release
            rel = load_v9_release(a.v9_train, expect_md5=a.v9_train_md5 or None)
            info["v9_md5"] = rel.md5
            J = D.build_label_join(man, rel, lead_npz=a.lead_train if os.path.exists(a.lead_train) else None,
                                   speed=D.load_speed_sidecar(a.speed_train))
            if a.join_train:
                D.save_join(J, a.join_train)
    clips = D.select_train_clips(clips_all, a.data_frac)
    if a.max_clips:
        clips = clips[:a.max_clips]
    expected = (max(1, int(round(a.data_frac * len(clips_all)))) if a.data_frac < 1.0 else len(clips_all))
    if a.max_clips:
        expected = min(a.max_clips, expected)
    info.update({"n_clips": len(clips), "n_clips_expected": expected, "n_clips_all": len(clips_all)})
    lru = a.lru or max(8, int(1.5e9 // (J["n"].max() * cfg.n_tok * 1024 * 2)))
    info["lru"] = lru
    store = D.ClipStore(cache, "train", lru=lru, grid=a.grid)
    allw = D.train_windows(store, J, clips_all, hist=cfg.hist)
    wins = D.train_windows(store, J, clips, hist=cfg.hist)
    info.update({"n_windows_all_clips": len(allw), "n_windows": len(wins)})
    return store, J, wins, clips, info


class Prefetch:
    def __init__(self, gen, fn, depth=3):
        self.q, self.fn = queue.Queue(depth), fn
        self.t = threading.Thread(target=self._run, args=(gen,), daemon=True)
        self.t.start()

    def _run(self, gen):
        try:
            for x in gen:
                self.q.put(self.fn(x))
        except Exception as e:                                  # surface in the consumer
            self.q.put(e)
        self.q.put(None)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            if isinstance(x, Exception):
                raise x
            yield x


def to_dev(b, dev):
    return {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev)) for k, v in b.items()}


def argv_sha(argv) -> str:
    return hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = make_config(a)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    store, J, wins, clips, info = build_data(a, cfg)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    prior = M.load_p10(a.p10) if not a.synthetic else M.P10Prior.synthetic(a.seed)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, prior, mu, sd).to(dev)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}], lr=a.lr)
    S = math.ceil(info["n_windows_all_clips"] / a.batch)
    total = a.steps or S * a.steps_mult
    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)),
                               max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    # ---- the launch gate
    bad = declared_vs_built(a, cfg, model, opt, info)
    if bad:
        raise SystemExit("ZZDVB-REFUSEDZZ declared-vs-built: " + "; ".join(bad))
    probe = D.collate([ds.item(*w) for w in next(sampler.epoch(0))])
    live = live_check(model.cpu(), probe, cfg)
    model.to(dev)
    adm = M.assert_no_situation_classifier_input(model.cpu().eval(), probe)
    model.to(dev).train()
    sc = {"declared_vs_built": "PASS", "live": live, "admissibility": adm, "levers": model.built_levers(),
          "steps_total": total, "S_steps_per_epoch_100pct": S, "info": info, "argv_sha": argv_sha(argv)}
    json.dump(sc, open(os.path.join(a.out, "selfcheck.json"), "w"), indent=1, default=str)
    if a.selfcheck_only:
        print("SELFCHECK PASS", json.dumps({k: sc[k] for k in ("steps_total", "info")}, default=str))
        return 0
    json.dump({"argv": argv, "sha": argv_sha(argv)}, open(os.path.join(a.out, "argv.json"), "w"))

    # ---- resume
    ck_path = os.path.join(a.out, "ckpt_last.pt")
    step, ep, bi = 0, 0, 0
    if os.path.exists(ck_path) and not a.no_resume:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["argv_sha"] != argv_sha(argv):
            raise SystemExit("ZZRESUME-REFUSEDZZ the checkpoint was written by a different argv")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, ep, bi = ck["step"], ck["epoch"], ck["batch_in_epoch"]
        print(f"[resume] step {step} epoch {ep} batch {bi}", flush=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - a.warmup) / max(total - a.warmup, 1), 1.0))))

    def save(final=False):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep, "batch_in_epoch": bi,
                    "argv_sha": argv_sha(argv), "cfg": cfg.__dict__, "p10_path": os.path.abspath(a.p10) if not a.synthetic else "",
                    "grid": a.grid, "arm": a.arm, "seed": a.seed, "cons_mu": mu, "cons_sd": sd,
                    "cfg_tuple_fields": {"wm_strides": list(cfg.wm_strides)}}, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    log = open(os.path.join(a.out, "train_log.jsonl"), "a", encoding="utf-8")
    amp = a.amp and dev.type == "cuda"
    t0 = time.time()
    done = step >= total
    while not done:
        gen_batches = sampler.epoch(ep)
        for _ in range(bi):                                    # fast-forward within the epoch (indices only)
            next(gen_batches, None)
        pf = Prefetch(gen_batches, lambda idx: D.collate([ds.item(*w) for w in idx]), depth=3)
        for b in pf:
            b = to_dev(b, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            p_gt = M.p_gt_schedule(step, total, cfg.p_gt_start, cfg.p_gt_end)
            torch.manual_seed(a.seed * 1_000_003 + step)               # per-step RNG: a resumed run is the same run
            if os.environ.get("V8F_TRAIN_CRASH_AT") == str(step):      # test hook (resume test); never set in a launch
                raise RuntimeError("V8F_TRAIN_CRASH_AT")
            model.train()
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                out = model.loss(b, p_gt=p_gt, gen=None)
            T = out["terms"]
            if not torch.isfinite(T["total"]):
                raise SystemExit(f"ZZNONFINITE-LOSSZZ at step {step}: {({k: float(v) for k, v in T.items()})}")
            opt.zero_grad(set_to_none=True)
            T["total"].backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip_grad)
            opt.step()
            step += 1
            bi += 1
            if step % a.log_every == 0 or step == total:
                rec = {"step": step, "epoch": ep, "p_gt": p_gt, "lr": lr_at(step - 1), "grad_norm": float(gn),
                       "sec": time.time() - t0, **{k: float(v) for k, v in T.items()}}
                if "viol" in out:
                    rec["viol_any_batch"] = float(torch.stack(list(out["viol"].values())).any(0).float().mean())
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[{step}/{total}] ade {rec['ade']:.3f} total {rec['total']:.3f} p_gt {p_gt:.2f}", flush=True)
            if step % a.ckpt_every == 0:
                save()
            if step >= total:
                done = True
                break
        else:
            ep += 1
            bi = 0
    save(final=True)
    summary = {"done": True, "steps": step, "arm": a.arm, "seed": a.seed, "data_frac": a.data_frac,
               "steps_mult": a.steps_mult, "S": S, "n_clips": info["n_clips"], "n_windows": info["n_windows"],
               "seconds": time.time() - t0, "argv_sha": argv_sha(argv), "levers": model.built_levers(),
               "selfcheck": "selfcheck.json"}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)      # LAST, atomically meaningful
    print("DONE", json.dumps({k: summary[k] for k in ("steps", "arm", "seed", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
