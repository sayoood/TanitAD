"""SPEC_WPB (REGISTERED sha256 c952d4b4..., 2026-10-04T12:33:09Z) -- the WP-B arms on R1's frozen-trunk harness.

Reuses R1's cache, windows, dense labels, announced-nav table and head path (`r1_arms.py` / `r1_heads.py`, imported --
never copied), on the WP-B FIX TREE (tip + code/fix overlay on PYTHONPATH). The refcv8 seams are attached POST HOC to
the strictly loaded refcv7-50,400 model (`RefCV3Model.enable_refcv8`) -- the warm start itself.

Phases (each its own Thor GPU job under thor_gpu.lock, <= ~40 min):
  --phase iw      identity gate I-W on the 1,112 grid windows (seed 0): seams attached + no training == T0-untrained
                  bit for bit; allocation + prior-free attached with emission off: plan / base fan bit-identical and
                  base scores within 1e-5. -> arms/I_W.json. Fail => nothing else runs.
  --phase timing  30 head-only training steps for T0, T1, T2, X2b, nothing saved or scored -> arms/timing_wpb.json
                  (+ the projected refcv8 s/step, SPEC sec. 4)
  --phase train   one arm -> arms/<arm>/heads.pt + train_log.jsonl + train_record.json
  --phase eval    one arm on the 4,634 scored + 1,112 grid windows, sampler seeds 0 and 1, batch 1, with the forced
                  passes (--ctrl) -> arms/<arm>/eval_s<seed>.pt
sha12 only. Nothing here edits a tree file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC_SHA256 = "c952d4b45a55dfd422c920264f900057e4876780d81d01b859b6695c2d60e04f"
R1_CODE_DEFAULT = "/home/nvidia/refcv8_r1/code"
REFCV7_STEP_S = 9.9                    # refcv7-r101-s0 MEASURED step time (PLAN_REFCV8 sec. 6); the B-COST anchor
REFCV7_BATCH = 16
COST_CEILING_S = 10.5

_R8_COMMON = dict(w_listwise=1.0, w_cons=0.05, cond_dropout=0.15)
ARMS = {
    "T0": dict(seams=False, seed=0),
    "T0r": dict(seams=False, seed=1),
    "T1": dict(seams=True, r8={}),
    "T1d": dict(seams=True, r8={}, derange=True),
    "T2": dict(seams=True, r8=dict(n_alloc=32, alloc_emit=True)),
    "T2d": dict(seams=True, r8=dict(n_alloc=32, alloc_emit=True), derange=True),
    "X1h": dict(seams=True, r8=dict(w_subscore=0.5)),
    "T2s": dict(seams=True, r8=dict(n_alloc=32, alloc_emit=True, w_sat=0.1)),
    "X2a": dict(seams=True, r8=dict(lat_prior_dropout=0.3)),
    "X2b": dict(seams=True, r8=dict(prior_free_group=True, prior_free_emit=True)),
}
STOP_D = (10.0, 20.0)
STOP_V0_MIN, STOP_DECEL_MAX = 3.0, 4.0


def _r1(a):
    sys.path.insert(0, a.r1_code)
    import r1_arms as A     # noqa: E402  (R1's harness: cache, labels, nav, feed/batch builders, the model build)
    import r1_heads as H    # noqa: E402
    import r1_lib as L1     # noqa: E402
    A.DEV = a.device
    return A, H, L1


def r8cfg(arm: str):
    from tanitad.refs import refcv8_conditioning as C
    spec = ARMS[arm]
    kw = dict(_R8_COMMON)
    kw.update(spec.get("r8", {}))
    return C.R8Config(**kw)


def build_arm(a, A, arm: str):
    """R1's strict build of refcv7-50,400 on THIS tree (its inert H5 hooks included), then the refcv8 seams post hoc."""
    model, cfg, args, tr, H, feed, TAC, OUT = A.build(a)
    spec = ARMS[arm]
    if spec["seams"]:
        n = model.enable_refcv8(r8cfg(arm))
        model.to(a.device)
        model._r8_derange_feed = bool(spec.get("derange", False))
        print(f"[wpb] {arm}: refcv8 seams attached post hoc, {n} new parameters", flush=True)
    return model, cfg, args, tr, H, feed, TAC, OUT


def _sync(dev):
    import torch
    if dev == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------------------- #
# I-W                                                                                                             #
# ------------------------------------------------------------------------------------------------------------- #
def phase_iw(a):
    import torch
    A, H, L1 = _r1(a)
    D, _sel = A.load_split(a.cache, "eval")
    grid = [j for j, f in enumerate(D["_flags"]) if "grid" in f]
    if not grid and "smoke" in str(a.out):
        grid = list(range(len(D["_flags"])))          # a SMOKE cache carries no grid windows
    if a.max_eval_windows:
        grid = grid[: a.max_eval_windows]
    table = json.loads((Path(a.r1_code).parent / "raw" / "r1_nav_ann_table.json").read_text())
    nav, _ = A.nav_ann(D, table)

    def run(model, OUT, feed, tr, args):
        outs = []
        for j in grid:
            ix = np.array([j])
            feed.cur = A.make_feed(D, ix, a.device)
            b = A.make_batch(D, ix, None, None, nav)
            OUT.clear()
            torch.manual_seed(0)
            with torch.no_grad():
                H.forward_like_trainer(model, b, a.device, args.mode, tr)
            o = OUT["out"]
            outs.append({k: o[k].detach().float().cpu() for k in ("traj", "sel_idx", "anchor_traj", "sel_score_v3")})
            if "r8_n_base" in o:
                outs[-1]["nb"] = int(o["r8_n_base"])
        return outs

    m0, _c, args, tr, _H, feed, _T, OUT = build_arm(a, A, "T0")
    m0.eval()
    base = run(m0, OUT, feed, tr, args)
    del m0
    rep = {"spec_sha256": SPEC_SHA256, "n_windows": len(grid)}
    for name, extra in (("seams_no_extension", {}), ("alloc32_prior_free_emit_off",
                                                     dict(n_alloc=32, prior_free_group=True))):
        from tanitad.refs import refcv8_conditioning as C
        m, _c, args, tr, _H, feed, _T, OUT = A.build(a)
        m.enable_refcv8(C.R8Config(**dict(_R8_COMMON, **extra)))
        m.to(a.device).eval()
        o = run(m, OUT, feed, tr, args)
        nd = {"traj": 0, "sel_idx": 0, "anchor_traj_base": 0}
        dmax = 0.0
        for x, y in zip(base, o):
            nb = y.get("nb", x["anchor_traj"].shape[1])
            nd["traj"] += int(not torch.equal(x["traj"], y["traj"]))
            nd["sel_idx"] += int(not torch.equal(x["sel_idx"], y["sel_idx"]))
            nd["anchor_traj_base"] += int(not torch.equal(x["anchor_traj"], y["anchor_traj"][:, :nb]))
            dmax = max(dmax, float((x["sel_score_v3"] - y["sel_score_v3"][:, :nb]).abs().max()))
        exact = name == "seams_no_extension"
        ok = all(v == 0 for v in nd.values()) and (dmax == 0.0 if exact else dmax <= 1e-5)
        rep[name] = {"windows_differing": nd, "max_abs_dscore_base": dmax, "PASS": bool(ok),
                     "criterion": "bit-identical" if exact else "plan + base fan bit-identical, base scores <= 1e-5"}
        del m
    rep["PASS"] = bool(rep["seams_no_extension"]["PASS"] and rep["alloc32_prior_free_emit_off"]["PASS"])
    out = Path(a.out) / "I_W.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1), flush=True)
    return 0 if rep["PASS"] else 3


# ------------------------------------------------------------------------------------------------------------- #
# train                                                                                                           #
# ------------------------------------------------------------------------------------------------------------- #
def train_arm(a, arm: str, steps: int, timing_only: bool = False) -> dict:
    import torch
    A, H, L1 = _r1(a)
    from tanitad.train import refcv8_train as RT
    spec = ARMS[arm]
    D, _sel = A.load_split(a.cache, "train")
    n = len(D["win_sha12"])
    lat, lon = A.dense_labels(D)
    table = json.loads((Path(a.r1_code).parent / "raw" / "r1_nav_ann_table.json").read_text())
    nav, _ = A.nav_ann(D, table)
    model, cfg, args, tr, H_, feed, TAC, OUT = build_arm(a, A, arm)
    seams = bool(spec["seams"])
    tr.SEL_V3_WEIGHT = 0.0 if not seams else tr.SEL_V3_WEIGHT     # T0: R1-H4's way; seams: the r8 path zeroes it
    params = [p for _n, p in H.trainable_head_params(model)]
    model._r1_tac_cond.weight.requires_grad_(False)                 # R1's H5 port: not a WP-B lever
    model._r1_tac_cond.bias.requires_grad_(False)
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = 100

    def lr_at(s):
        return a.lr * (s + 1) / warm if s < warm else \
            a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    seed = int(spec.get("seed", 0))
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = []
    while len(order) < steps * a.batch:
        order += g.permutation(n).tolist()
    model.train()
    if seams:
        model._r8_rc_dropout = RT.RC_DROPOUT_MIN
        model._r8_nav_args_dropout = 0.5
    out_dir = Path(a.out) / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / ("timing_log.jsonl" if timing_only else "train_log.jsonl"), "w", encoding="utf-8")
    t0, st, n_bad = time.time(), [], 0
    for s in range(steps):
        ix = np.array(sorted(order[s * a.batch:(s + 1) * a.batch]))
        _sync(a.device)
        ts = time.time()
        feed.cur = A.make_feed(D, ix, a.device)
        b = A.make_batch(D, ix, lat, lon, nav)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s)
        if seams:
            model._r8_tf_ratio = RT.tf_ratio(s, steps, model.cfg.refcv8.tf_start, model.cfg.refcv8.tf_end)
        OUT.clear()
        losses = tr.compute_losses_v3(model, b, a.device, mode=args.mode)
        loss = losses["loss"]
        if not seams:
            o = OUT["out"]
            ll = A.listwise_loss(o["sel_score_v3"], o["anchor_traj"].detach(), D["gt"][ix].to(a.device),
                                 D["gt_valid"][ix].to(a.device), o["reach_keep"].bool())
            loss = loss + ll
        if not torch.isfinite(loss):
            n_bad += 1
            opt.zero_grad(set_to_none=True)
            if n_bad > 5:
                raise SystemExit("[wpb] > 5 non-finite losses -- stopping")
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _sync(a.device)
        st.append(time.time() - ts)
        if s % 25 == 0 or s == steps - 1:
            row = {"step": s, "loss": float(loss), "lr": lr_at(s), "step_s": st[-1], "elapsed_s": time.time() - t0}
            for k, v in losses.items():
                if (k.startswith("r8_") or k in ("traj", "cls", "sel_v3", "tac_v6", "cascade")) and torch.is_tensor(v):
                    row[k] = float(v)
            if seams:
                d = model.core.decoder
                row.update(beta_lat=float(d.r8_sel.beta_lat), beta_lon=float(d.r8_sel.beta_lon),
                           gamma_prog=float(d.r8_sel.gamma_prog), gamma_head=float(d.r8_sel.gamma_head),
                           mod_wnorm=float(sum(p.weight.norm() for p in d.r8_mod.proj)))
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[wpb] {arm} step {s}/{steps} loss {float(loss):.4f} step_s {st[-1]:.3f}", flush=True)
    log.close()
    rec = {"arm": arm, "steps": steps, "batch": a.batch, "lr": a.lr, "train_seed": seed, "n_train_windows": n,
           "n_trainable_params": int(sum(p.numel() for p in params)), "r8": (r8cfg(arm).to_dict() if seams else None),
           "derange": bool(spec.get("derange", False)), "n_nonfinite_skipped": n_bad,
           "step_s_mean_after_5": float(np.mean(st[5:])) if len(st) > 5 else float(np.mean(st)),
           "wall_s": round(time.time() - t0, 1), "spec_sha256": SPEC_SHA256,
           "cuda_max_gib": (torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else None)}
    if timing_only:
        return rec
    keep = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith(H.FROZEN_PREFIXES)}
    torch.save(keep, out_dir / "heads.pt")
    rec["heads_md5"] = hashlib.md5((out_dir / "heads.pt").read_bytes()).hexdigest()
    (out_dir / "train_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ------------------------------------------------------------------------------------------------------------- #
# eval                                                                                                            #
# ------------------------------------------------------------------------------------------------------------- #
def eval_arm(a, arm: str) -> dict:
    import torch
    A, H, L1 = _r1(a)
    from tanitad.refs import refcv8_conditioning as C
    spec = ARMS[arm]
    D, _sel = A.load_split(a.cache, "eval")
    n = len(D["win_sha12"]) if not a.max_eval_windows else min(a.max_eval_windows, len(D["win_sha12"]))
    table = json.loads((Path(a.r1_code).parent / "raw" / "r1_nav_ann_table.json").read_text())
    nav, nav_reason = A.nav_ann(D, table)
    model, cfg, args, tr, H_, feed, TAC, OUT = build_arm(a, A, arm)
    model._r8_derange_feed = False                    # eval feeds every model its OWN tactical output (H5c's rule)
    hp = Path(a.out) / arm / "heads.pt"
    sd = torch.load(hp, map_location=a.device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(H.FROZEN_PREFIXES)]
    if bad or unexpected:
        raise SystemExit(f"[wpb] heads load: missing {bad[:3]} unexpected {unexpected[:3]}")
    model.eval()
    seams = bool(spec["seams"])
    out_dir = Path(a.out) / arm
    res = {}
    t0 = time.time()
    for seed in [int(x) for x in a.eval_seeds.split(",")]:
        fp = out_dir / f"eval_s{seed}.pt"
        if fp.exists():
            continue
        R = {k: [] for k in ("traj", "sel", "s_e9", "p_lat", "p_lon", "fan16", "reach", "cons_lat", "cons_lon",
                             "kind", "lat3", "hyp", "ctrl_dir", "ctrl_alloc_share", "ctrl_alloc_share_lat3", "stop_d_err",
                             "stop_alloc_ok")}
        for j in range(n):
            ix = np.array([j])
            feed.cur = A.make_feed(D, ix, a.device)
            b = A.make_batch(D, ix, None, None, nav)
            TAC["force"], TAC["mode"] = None, None
            OUT.clear()
            torch.manual_seed(seed)
            with torch.no_grad():
                H.forward_like_trainer(model, b, a.device, args.mode, tr)
            o = OUT["out"]
            R["traj"].append(o["traj"].float().cpu())
            R["sel"].append(o["sel_idx"].cpu())
            R["s_e9"].append(o["sel_score_v3"].float().cpu())
            R["p_lat"].append(o["tacv6_lat_logits"].float().softmax(-1).cpu())
            R["p_lon"].append(o["tacv6_lon_logits"].float().softmax(-1).cpu())
            R["fan16"].append(o["anchor_traj"].half().cpu())
            R["reach"].append(o["reach_keep"].bool().cpu())
            if seams:
                R["cons_lat"].append(o["r8_cons_lat"].float().cpu())
                R["cons_lon"].append(o["r8_cons_lon"].float().cpu())
                R["kind"].append(o["r8_kind"][None].cpu())
                R["lat3"].append(o["r8_lat3"].cpu())
                if "r8_hyp_alloc" in o:
                    R["hyp"].append(o["r8_hyp_alloc"].cpu())
            # ---- forced passes (SPEC sec. 4) ------------------------------------------------------------------
            scored = ("turn" in D["_flags"][j]) or ("sample" in D["_flags"][j])
            if a.ctrl and scored and D["_cls"][j] in ("turnL", "turnR", "straight", "gentle"):
                dirs, shares, shares3 = [], [], []
                for lat_name in ("TURN_L", "TURN_R", "LANE_KEEP"):
                    OUT.clear()
                    torch.manual_seed(seed)
                    if seams:
                        model.set_r8_force(lat3=C.LAT3.index(lat_name))
                    else:                              # T0: R1's mechanism (the tactical posterior only)
                        TAC["force"] = L1.LAT_V7.index(lat_name)
                    with torch.no_grad():
                        H.forward_like_trainer(model, b, a.device, args.mode, tr)
                    TAC["force"] = None
                    oc = OUT["out"]
                    dirs.append(int(L1.dir_class(L1.terminal_heading(oc["traj"][0].float().cpu().numpy()))))
                    if seams and "r8_kind" in oc and bool((oc["r8_kind"] == 1).any()):
                        fa = oc["anchor_traj"][0][oc["r8_kind"] == 1].float().cpu().numpy()
                        want = {"TURN_L": 1, "TURN_R": -1, "LANE_KEEP": 0}[lat_name]
                        shares.append(float((L1.dir_class(L1.terminal_heading(fa)) == want).mean()))
                        # the second reading, the lat3 (30 deg excursion) class the tags use -- reported beside it
                        want3 = {"TURN_L": 6, "TURN_R": 7, "LANE_KEEP": 0}[lat_name]
                        shares3.append(float((L1.dense_lat(fa, np.ones(fa.shape[:2], bool)) == want3).mean()))
                    else:
                        shares.append(float("nan"))
                        shares3.append(float("nan"))
                R["ctrl_dir"].append(torch.tensor([dirs]))
                R["ctrl_alloc_share"].append(torch.tensor([shares]))
                R["ctrl_alloc_share_lat3"].append(torch.tensor([shares3]))
                v0 = float(D["b_pose_last"][j, 3])
                errs, aok = [], []
                for d in STOP_D:
                    if not seams or v0 < STOP_V0_MIN or v0 * v0 / (2 * d) > STOP_DECEL_MAX:
                        errs.append(float("nan"))
                        aok.append(float("nan"))
                        continue
                    cons = torch.tensor([[float("nan"), float("nan"),
                                          math.log1p(d) / C.PROG_LOG_SCALE, 0.0]], device=a.device)
                    model.set_r8_force(lon6=C.LON6.index("BRAKE_TO"), cons=cons)
                    OUT.clear()
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        H.forward_like_trainer(model, b, a.device, args.mode, tr)
                    oc = OUT["out"]
                    errs.append(_stop_err(oc["traj"][0].float().cpu().numpy(), v0, d))
                    if "r8_kind" in oc and bool((oc["r8_kind"] == 1).any()):
                        fa = oc["anchor_traj"][0][oc["r8_kind"] == 1].float().cpu().numpy()
                        aok.append(float(np.mean([abs(_stop_err(p, v0, d)) <= max(2.0, 0.1 * d)
                                                  if np.isfinite(_stop_err(p, v0, d)) else 0.0 for p in fa])))
                    else:
                        aok.append(float("nan"))
                R["stop_d_err"].append(torch.tensor([errs]))
                R["stop_alloc_ok"].append(torch.tensor([aok]))
            elif a.ctrl:
                R["ctrl_dir"].append(torch.full((1, 3), -9))
                R["ctrl_alloc_share"].append(torch.full((1, 3), float("nan")))
                R["ctrl_alloc_share_lat3"].append(torch.full((1, 3), float("nan")))
                R["stop_d_err"].append(torch.full((1, len(STOP_D)), float("nan")))
                R["stop_alloc_ok"].append(torch.full((1, len(STOP_D)), float("nan")))
            if j % 500 == 0:
                print(f"[wpb] {arm} eval seed {seed} ZZ{j}-{n}ZZ {time.time() - t0:.0f}s", flush=True)
            if (time.time() - t0) > a.budget_s:
                raise SystemExit(f"[wpb] eval budget exceeded at window {j} -- nothing written for seed {seed}")
        out = {}
        for k, v in R.items():
            if not v:
                continue
            out[k] = (torch.cat(v, 0) if k not in ("fan16", "reach", "s_e9", "lat3", "kind") else v)
        out.update(win_sha12=D["win_sha12"][:n], b_r1_i=D["b_r1_i"][:n], nav_cmd_used=torch.as_tensor(nav[:n]),
                   arm=arm, seed=seed, spec_sha256=SPEC_SHA256)
        torch.save(out, str(fp) + ".tmp")
        os.replace(str(fp) + ".tmp", fp)
        res[seed] = {"n": n, "wall_s": round(time.time() - t0, 1)}
    return res


def _stop_err(path, v0, d):
    """the PLAN's own stop distance minus d (m): arc to the first slot segment at <= 0.5 m/s; NaN if it never stops."""
    slot_t = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    q = np.concatenate([np.zeros((1, 2)), np.asarray(path, np.float64)], 0)
    seg = np.linalg.norm(np.diff(q, axis=0), axis=-1)
    v = seg / np.diff(np.concatenate([[0.0], slot_t]))
    s = np.cumsum(seg)
    hit = np.nonzero(v <= 0.5)[0]
    if hit.size == 0:
        return float("nan")
    return float((s[hit[0] - 1] if hit[0] > 0 else 0.0) - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["iw", "timing", "train", "eval"])
    ap.add_argument("--arm", default="T0", choices=sorted(ARMS))
    ap.add_argument("--r1-code", default=R1_CODE_DEFAULT)
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--out", default="/home/nvidia/refcv8_wpb/arms")
    ap.add_argument("--ckpt", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt")
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--timing-steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-seeds", default="0,1")
    ap.add_argument("--ctrl", action="store_true")
    ap.add_argument("--budget-s", type=float, default=2400.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-eval-windows", type=int, default=0, help="smoke only")
    a = ap.parse_args()
    spec = HERE.parent / "SPEC_WPB.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest() != SPEC_SHA256:
        raise SystemExit("[wpb] SPEC_WPB.md on disk differs from the REGISTERED one -- refusing")
    if a.device != "cuda" and "smoke" not in str(a.out):
        raise SystemExit("[wpb] --device cpu only with an --out path containing 'smoke'")
    import tanitad
    print(f"[wpb] tanitad from {tanitad.__file__}", flush=True)
    if a.phase == "iw":
        return phase_iw(a)
    if a.phase == "timing":
        rows = {arm: train_arm(a, arm, a.timing_steps, timing_only=True) for arm in ("T0", "T1", "T2", "X2b")}
        base = rows["T0"]["step_s_mean_after_5"]
        rep = {"spec_sha256": SPEC_SHA256, "nothing_saved_or_scored": True, "rows": rows,
               "projection": {arm: {"head_step_s": r["step_s_mean_after_5"],
                                    "delta_head_s": r["step_s_mean_after_5"] - base,
                                    "projected_refcv8_step_s": REFCV7_STEP_S
                                    + (r["step_s_mean_after_5"] - base) * REFCV7_BATCH / a.batch,
                                    "B_COST_PASS": (REFCV7_STEP_S + (r["step_s_mean_after_5"] - base)
                                                    * REFCV7_BATCH / a.batch) <= COST_CEILING_S}
                              for arm, r in rows.items()},
               "estimator": "ESTIMATED: 9.9 s (refcv7 MEASURED) + (head-only step delta MEASURED) x 16/32"}
        p = Path(a.out) / "timing_wpb.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep["projection"], indent=1), flush=True)
        return 0
    iw = Path(a.out) / "I_W.json"
    if not iw.exists() or not json.loads(iw.read_text())["PASS"]:
        raise SystemExit("[wpb] I-W has not PASSED -- no arm runs (SPEC sec. 2)")
    if a.phase == "train":
        steps = a.steps
        if steps <= 0:
            t = json.loads((Path(a.cache).parent / "arms" / "timing_smoke.json").read_text())
            steps = int(t["steps_per_arm"])                # R1's registered step rule, the same for every arm
        print(json.dumps(train_arm(a, a.arm, steps), indent=1), flush=True)
        return 0
    print(json.dumps(eval_arm(a, a.arm)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
