"""refcv8 R1 arms -- train the planner-side heads of refcv7-r101-s0 @ 50,400 from the frozen-trunk cache, then evaluate.

SPEC: `SPEC_R1.md` (REGISTERED by the Master Mind, sha256 e9873e12..., 2026-10-04T11:15:41Z); the script refuses to run
if the shipped copy differs. Harness: DESIGN.md. Phases (each one Thor GPU job under thor_gpu.lock, <= ~40 min):
  --phase timing  : a few H1 steps, NOTHING saved or scored -> raw/timing_smoke.json (+ the step count by the
                    registration rule: 1,320 if a step <= 1.35 s, else floor(1,800 / step_s))
  --phase train   : one arm, fixed steps, heads saved (trainable state only) -> arms/<arm>/heads.pt + train_log.jsonl
  --phase eval    : one arm on the 4,634 scored + 1,112 grid eval windows, sampler seeds 0 and 1, batch 1
                    (+ --ctrl: forced TURN_L / TURN_R / LANE_KEEP on classified windows) -> arms/<arm>/eval_s<seed>.pt
Nothing here edits a tree file; the stubs are r1_heads.py's. ⛔ sha12 only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_lib as L1  # noqa: E402

SPEC_SHA256 = "e9873e1274edc0247240e6ed5ce9cff1ef331d3c63885ae5735a00c1d123b741"
SPEC_A2_SHA256 = "23c407b6e89111aa9f5acd335807c8321a644211e8b8b127d8e0369cce0e05f2"   # SPEC_R1_A2, 2026-10-05T04:53:28Z
EXPECT_STEP = 50400
ARMS = {
    # arm: (labels, nav, selector, cond)
    "H0": (None, "clip", None, None),
    "H0n": (None, "ann", None, None),
    "H1": ("old", "clip", "ce", None),
    "H2": ("dense", "clip", "ce", None),
    "H2s": ("dense_shuf", "clip", "ce", None),
    "H3": ("dense", "ann", "ce", None),
    "H4": ("dense", "ann", "listwise", None),
    "H5": ("dense", "ann", "listwise", "tac"),
    "H5c": ("dense", "ann", "listwise", "tac_derange"),
    # SPEC_R1_A2 (registered separately; post-hoc relative to SPEC_R1)
    "H1n": ("old", "clip", "ce", None),            # A2-3: H1 on the natural turn share
}
SUBSAMPLE = {"H1n": "natural_turn_share"}          # A2-3: all non-turn train windows + a seeded 585 GT-turn ones
A2_N_TURN = 585
LIST_T = 1.0                       # SPEC_R1 sec. 3: p*_k ∝ exp(-cost_k / 1.0)
LIST_SPEED_SCALE_M = 1.0           # cost: mean |s_k - s_GT| / 1.0 m
LIST_DIR_SCALE_RAD = math.radians(15.0)
FORCE_LOGIT = 20.0                 # a forced tactical posterior: one-hot at +20 logits (eval only, no loss)
DEV = "cuda"                       # "cpu" only for a code-path SMOKE (--device cpu), never for an arm


def _sync():
    import torch
    if DEV == "cuda":
        torch.cuda.synchronize()


def _maxmem():
    import torch
    return torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else float("nan")


# --------------------------------------------------------------------------------------------- #
# cache                                                                                           #
# --------------------------------------------------------------------------------------------- #
def load_split(cache, split):
    import torch
    shards = sorted(glob.glob(str(Path(cache) / split / "shard_*.pt")))       # trunk-carrying shards only
    if not shards:
        raise SystemExit(f"[r1-arms] no shards under {cache}/{split}")
    parts = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
    D = {}
    for k in parts[0]:
        v = [p[k] for p in parts]
        D[k] = torch.cat(v, 0) if torch.is_tensor(v[0]) else [x for xs in v for x in xs]
    sel = json.loads((Path(cache) / split / "selection.json").read_text())
    winfo = {w["i"]: w for w in sel["windows"]}
    D["_cls"] = [winfo[int(i)]["cls"] for i in D["b_r1_i"]]
    D["_flags"] = [winfo[int(i)]["flags"] for i in D["b_r1_i"]]
    D["_t"] = [winfo[int(i)]["t"] for i in D["b_r1_i"]]
    return D, sel


def dense_labels(D):
    lat = L1.dense_lat(D["gt"].numpy(), D["gt_valid"].numpy())
    lon = L1.dense_lon(D["b_pose_last"][:, 3].numpy(), D["b_future_poses_ext"][:, :, 3].numpy(),
                       D["b_future_valid_ext"].numpy())
    return lat, lon


def nav_ann(D, table):
    side = np.zeros(len(D["win_sha12"]), np.int64)
    reason = []
    for j, s in enumerate(D["win_sha12"]):
        r = table.get(s)
        if r is None:
            reason.append("no_record")
            continue
        t_rel = float(D["b_r1_t_now"][j]) - float(r["t0_s"])
        side[j], rs = L1.nav_ann_state(r["entries"], r["announced"], t_rel, L1.NAV_H_S)
        reason.append(rs)
    return np.array([L1.SIDE_TO_NAV[int(x)] for x in side], np.int64), reason


def cum_arclen(P):
    import torch
    Q = torch.cat([torch.zeros_like(P[..., :1, :]), P], -2)
    return torch.linalg.vector_norm(Q[..., 1:, :] - Q[..., :-1, :], dim=-1).cumsum(-1)


def term_heading(P):
    import torch
    d = P[..., -1, :] - P[..., -2, :]
    th = torch.atan2(d[..., 1], d[..., 0])
    return torch.where(torch.linalg.vector_norm(d, dim=-1) < L1.STALL_M, torch.zeros_like(th), th)


def listwise_loss(score, fan, gt, gv, reach):
    """SPEC_R1 sec. 3: cost_k = mean_valid |s_k - s_GT| / 1 m + |wrap(th_k - th_GT)| / 15 deg; p* ∝ exp(-cost / 1);
    loss = -sum p* log softmax(score) over the reach-kept candidates. A row with no valid slot is excluded; a row
    whose slot 60 is invalid keeps the speed term only (no terminal heading exists)."""
    import torch
    with torch.no_grad():
        v = gv.float()
        n = v.sum(-1)
        s_k = cum_arclen(fan.float())                              # [B, N, 8]
        s_g = cum_arclen(gt.float())[:, None]                      # [B, 1, 8]
        e_sp = ((s_k - s_g).abs() * v[:, None]).sum(-1) / n.clamp_min(1)[:, None] / LIST_SPEED_SCALE_M
        dth = term_heading(fan.float()) - term_heading(gt.float())[:, None]
        dth = torch.remainder(dth + math.pi, 2 * math.pi) - math.pi
        e_dir = dth.abs() / LIST_DIR_SCALE_RAD * gv[:, -1:].float()
        cost = e_sp + e_dir
        cost = cost.masked_fill(~reach, float("inf"))
        tgt = torch.softmax(-cost / LIST_T, dim=-1)
        ok = n > 0
    logp = torch.log_softmax(score.float().masked_fill(~reach, float("-inf")), dim=-1)
    l_row = -(tgt * logp.masked_fill(~reach, 0.0)).sum(-1)
    return l_row[ok].mean() if bool(ok.any()) else score.sum() * 0.0


# --------------------------------------------------------------------------------------------- #
# model                                                                                           #
# --------------------------------------------------------------------------------------------- #
def build(a):
    import torch
    from torch import nn
    from tanitad.eval import refcv7_loader as L
    tr = L.trainer()
    import r1_heads as H
    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device=DEV, strict=True)
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != EXPECT_STEP or sd["missing"] or sd["unexpected"]:
        raise SystemExit("[r1-arms] bad load")
    feed = H.Feed()
    H.install(model, feed)
    # head-only loss surface (DESIGN.md sec. 2): the frozen modules' terms are switched off
    model._w_agent = 0.0
    model._w_box3d = 0.0
    model._w_map = 0.0
    model._w_map_hires = 0.0
    model._map_hires = None
    tr.LAW_WEIGHT = 0.0
    d_pool = int(model.core.strategic.gru.weight_ih_l0.shape[1])
    model.core.encode_pooled = lambda x: torch.zeros(x.shape[0], d_pool, device=x.device)
    # H5 tactical condition (SPEC_R1 sec. 3): zero-init Linear(16 -> d_cond) into the decoder's `cond`
    TAC = {"mode": None, "vec": None, "pred": None, "force": None, "w": float(getattr(a, "guidance_w", 1.0))}
    d_cond = int(model.core.decoder.cond_proj.weight.shape[0])
    model._r1_tac_cond = nn.Linear(16, d_cond).to(DEV)
    nn.init.zeros_(model._r1_tac_cond.weight)
    nn.init.zeros_(model._r1_tac_cond.bias)

    def cond_hook(_m, _i, o):
        if TAC["mode"] is None:
            return o
        if TAC["mode"] == "vec":
            v = TAC["vec"]
        else:                                     # "pred": the same forward's tactical posteriors (detached)
            if TAC["pred"] is None:
                raise RuntimeError("[r1-arms] the tactical decoder did not run before the decoder's cond")
            v = TAC["pred"]
        return o + float(TAC.get("w", 1.0)) * model._r1_tac_cond(v.to(o.dtype))      # A2-2: guidance scale w

    def tac_hook(_m, _i, o):
        if TAC["force"] is not None:              # eval only: force the tactical LATERAL decision
            o = dict(o)
            f = torch.full_like(o["lat_logits"], 0.0)
            f[:, TAC["force"]] = FORCE_LOGIT
            o["lat_logits"] = f
        TAC["pred"] = torch.cat([o["lat_logits"].float().softmax(-1), o["lon_logits"].float().softmax(-1)],
                                -1).detach()
        return o
    model.core.decoder.cond_proj.register_forward_hook(cond_hook)
    model.tac_decoder_v6.register_forward_hook(tac_hook)
    OUT = {}

    def out_hook(_m, _i, o):
        if "out" not in OUT:
            OUT["out"] = o
    model.register_forward_hook(out_hook)
    return model, cfg, args, tr, H, feed, TAC, OUT


def make_feed(D, ix, dev):
    import torch
    q = D["bev_q"][ix].to(dev)
    return {"kv": D["kv"][ix].to(dev).float(), "pooled": D["pooled"][ix].to(dev).float(),
            "bev": L1.dequant_u8(q, D["bev_lo"][ix].to(dev), D["bev_scale"][ix].to(dev)),
            "slots": {k[5:]: (D[k][ix].to(dev).float() if D[k].is_floating_point() else D[k][ix].to(dev))
                      for k in D if k.startswith("slot_")},
            "bev_pad": None}


def make_batch(D, ix, lat=None, lon=None, nav=None):
    import torch
    b = {k[2:]: D[k][ix] for k in D if k.startswith("b_")}
    n = len(ix)
    b["frames"] = torch.zeros(n, 8, 1, 1, 1, dtype=torch.uint8)          # never read on the stubbed path
    b["future_frames"] = torch.zeros(n, 5, 1, 1, 1, dtype=torch.uint8)   # LAW target, weight 0
    if lat is not None:
        b["lat_v7"] = torch.as_tensor(lat[ix], dtype=torch.long)
        b["lon_v7"] = torch.as_tensor(lon[ix], dtype=torch.long)
    if nav is not None:
        b["nav_cmd"] = torch.as_tensor(nav[ix], dtype=torch.long)
    return b


def onehot16(lat, lon):
    import torch
    v = torch.zeros(len(lat), 16)
    for j, (a_, o_) in enumerate(zip(lat, lon)):
        if int(a_) != L1.IGNORE_ID:
            v[j, int(a_)] = 1.0
        if int(o_) != L1.IGNORE_ID:
            v[j, 8 + int(o_)] = 1.0
    return v


# --------------------------------------------------------------------------------------------- #
# train                                                                                           #
# --------------------------------------------------------------------------------------------- #
def train(a, steps, timing_only=False):
    import torch
    labels, navk, selk, condk = ARMS[a.arm]
    if labels is None:
        raise SystemExit(f"[r1-arms] {a.arm} is not trained")
    D, _sel = load_split(a.cache, "train")
    if SUBSAMPLE.get(a.arm) == "natural_turn_share":
        cls_ = np.array(D["_cls"])
        is_t = np.isin(cls_, ["turnL", "turnR"])
        t_ix = np.nonzero(is_t)[0]
        keep = np.sort(np.concatenate([np.nonzero(~is_t)[0],
                                       t_ix[np.sort(np.random.default_rng(0).choice(len(t_ix), A2_N_TURN,
                                                                                     replace=False))]]))
        for k in list(D):
            if hasattr(D[k], "shape") and not isinstance(D[k], list):
                D[k] = D[k][torch.as_tensor(keep)]
            elif isinstance(D[k], list):
                D[k] = [D[k][int(j)] for j in keep]
        print(f"[r1-arms] {a.arm}: natural turn share -- {len(keep)} windows, {int(is_t[keep].sum())} GT-turn",
              flush=True)
    n = len(D["win_sha12"])
    lat = lon = None
    lab_rec = {}
    if labels in ("dense", "dense_shuf"):
        lat, lon = dense_labels(D)
        lab_rec = {"lat_counts": {L1.LAT_V7[i] if i >= 0 else "IGNORE": int((lat == i).sum()) for i in sorted(set(lat))},
                   "lon_counts": {L1.LON_V7[i] if i >= 0 else "IGNORE": int((lon == i).sum()) for i in sorted(set(lon))}}
        if labels == "dense_shuf":
            perm, frac = L1.shuffle_within_clip(np.arange(n), D["win_sha12"], seed=0)
            lat, lon = lat[perm], lon[perm]
            lab_rec["shuffle_frac_windows_moved"] = frac
            lab_rec["shuffle_frac_lat_changed"] = float((lat != dense_labels(D)[0]).mean())
    nav = None
    if navk == "ann":
        table = json.loads((HERE.parent / "raw" / "r1_nav_ann_table.json").read_text())
        nav, _r = nav_ann(D, table)
        lab_rec["nav_ann_counts"] = {str(k): int((nav == k).sum()) for k in (0, 1, 2)}
    model, cfg, args, tr, H, feed, TAC, OUT = build(a)
    if selk == "listwise":
        tr.SEL_V3_WEIGHT = 0.0                     # the E9 CE is REPLACED by the listwise term
    params = [p for _n, p in H.trainable_head_params(model)]
    if condk is None:
        model._r1_tac_cond.weight.requires_grad_(False)
        model._r1_tac_cond.bias.requires_grad_(False)
        params = [p for p in params if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = 100

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    g = np.random.default_rng(a.train_seed)
    torch.manual_seed(a.train_seed)
    order = []
    while len(order) < steps * a.batch:
        order += g.permutation(n).tolist()
    model.train()
    out_dir = Path(a.out) / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / ("timing_log.jsonl" if timing_only else "train_log.jsonl"), "w", encoding="utf-8")
    t0 = time.time()
    st = []
    n_nonfinite = 0
    for s in range(steps):
        ix = np.array(sorted(order[s * a.batch:(s + 1) * a.batch]))
        _sync()
        ts = time.time()
        feed.cur = make_feed(D, ix, DEV)
        b = make_batch(D, ix, lat, lon, nav)
        TAC["mode"] = None
        if condk == "tac":
            TAC["mode"], TAC["vec"] = "vec", onehot16(lat[ix], lon[ix]).to(DEV)
        elif condk == "tac_derange":
            TAC["mode"], TAC["vec"] = "vec", onehot16(np.roll(lat[ix], 1), np.roll(lon[ix], 1)).to(DEV)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s)
        OUT.clear()
        losses = tr.compute_losses_v3(model, b, DEV, mode=args.mode)
        loss = losses["loss"]
        if selk == "listwise":
            o = OUT["out"]
            ll = listwise_loss(o["sel_score_v3"], o["anchor_traj"].detach(), D["gt"][ix].to(DEV),
                               D["gt_valid"][ix].to(DEV), o["reach_keep"].bool())
            loss = loss + ll
        if not torch.isfinite(loss):
            n_nonfinite += 1
            opt.zero_grad(set_to_none=True)
            if n_nonfinite > 5:
                raise SystemExit("[r1-arms] > 5 non-finite losses -- stopping")
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _sync()
        st.append(time.time() - ts)
        if s % 25 == 0 or s == steps - 1:
            row = {"step": s, "loss": float(loss), "lr": lr_at(s), "step_s": st[-1],
                   "elapsed_s": time.time() - t0, "cuda_max_gib": _maxmem()}
            for k in ("traj", "cls", "sel_v3", "tac_v6", "cascade", "goal_tac"):
                if k in losses and torch.is_tensor(losses[k]):
                    row[k] = float(losses[k])
            if selk == "listwise":
                row["listwise"] = float(ll)
            if condk is not None:
                row["tac_cond_wnorm"] = float(model._r1_tac_cond.weight.norm())
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[r1-arms] {a.arm} step {s}/{steps} loss {float(loss):.4f} step_s {st[-1]:.3f}", flush=True)
    log.close()
    rec = {"arm": a.arm, "steps": steps, "batch": a.batch, "lr": a.lr, "train_seed": a.train_seed,
           "n_train_windows": n, "n_trainable_params": int(n_par), "labels": labels, "nav": navk, "selector": selk,
           "cond": condk, "label_record": lab_rec, "n_nonfinite_skipped": n_nonfinite,
           "step_s_mean_after_5": float(np.mean(st[5:])) if len(st) > 5 else float(np.mean(st)),
           "wall_s": round(time.time() - t0, 1), "cuda_max_gib": _maxmem(),
           "spec_sha256": SPEC_SHA256}
    if timing_only:
        return rec
    keep = {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith(H.FROZEN_PREFIXES)}
    torch.save(keep, out_dir / "heads.pt")
    rec["heads_md5"] = hashlib.md5((out_dir / "heads.pt").read_bytes()).hexdigest()
    (out_dir / "train_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# --------------------------------------------------------------------------------------------- #
# eval                                                                                            #
# --------------------------------------------------------------------------------------------- #
def evaluate(a):
    import torch
    labels, navk, selk, condk = ARMS[a.arm]
    D, sel = load_split(a.cache, a.eval_split)
    tag = "eval" if a.eval_split == "eval" else "trainfit"
    if a.eval_split == "train":
        # SPEC_R1 sec. 7 TRAIN-FIT diagnostic: 1,000 of the arm's OWN training windows (seed 0), never a bar
        keep = np.sort(np.random.default_rng(0).choice(len(D["win_sha12"]), min(1000, len(D["win_sha12"])),
                                                       replace=False))
        for k in list(D):
            if torch.is_tensor(D[k]):
                D[k] = D[k][torch.as_tensor(keep)]
            elif isinstance(D[k], list):
                D[k] = [D[k][int(j)] for j in keep]
    n = len(D["win_sha12"]) if not a.max_eval_windows else min(a.max_eval_windows, len(D["win_sha12"]))
    nav = None
    nav_reason = None
    if navk == "ann":
        table = json.loads((HERE.parent / "raw" / "r1_nav_ann_table.json").read_text())
        nav, nav_reason = nav_ann(D, table)
    model, cfg, args, tr, H, feed, TAC, OUT = build(a)
    if labels is not None:
        hp = (Path(a.heads_dir) / a.arm / "heads.pt") if a.heads_dir else (Path(a.out) / a.arm / "heads.pt")
        sd_h = torch.load(hp, map_location=DEV, weights_only=False)
        missing, unexpected = model.load_state_dict(sd_h, strict=False)
        bad = [k for k in missing if not k.startswith(H.FROZEN_PREFIXES)]
        if bad or unexpected:
            raise SystemExit(f"[r1-arms] heads load: missing {bad[:3]} unexpected {unexpected[:3]}")
    model.eval()
    out_dir = Path(a.out) / (a.tag or a.arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.eval_seeds.split(",")]
    t0 = time.time()
    res_all = {}
    for seed in seeds:
        fp = out_dir / f"{tag}_s{seed}.pt"
        if fp.exists():
            print(f"[r1-arms] {a.arm} eval seed {seed} banked -- skip", flush=True)
            continue
        R = {k: [] for k in ("traj", "sel", "selb", "s_e9", "p_lat", "p_lon", "lat_tac", "lon_tac", "fan16",
                             "reach", "ctrl_sel_dir", "ctrl_fan_dir_share")}
        ctrl_on = bool(a.ctrl)
        for j in range(n):
            ix = np.array([j])
            feed.cur = make_feed(D, ix, DEV)
            b = make_batch(D, ix, None, None, nav)
            TAC["force"] = None
            TAC["mode"] = "pred" if condk is not None else None
            OUT.clear()
            torch.manual_seed(seed)
            with torch.no_grad():
                H.forward_like_trainer(model, b, DEV, args.mode, tr)
            o = OUT["out"]
            R["traj"].append(o["traj"].float().cpu())
            R["sel"].append(o["sel_idx"].cpu())
            R["selb"].append(o["sel_idx_base"].cpu())
            R["s_e9"].append(o["sel_score_v3"].float().cpu())
            R["p_lat"].append(o["tacv6_lat_logits"].float().softmax(-1).cpu())
            R["p_lon"].append(o["tacv6_lon_logits"].float().softmax(-1).cpu())
            R["lat_tac"].append(o["lat_logits_tac"].float().softmax(-1).cpu())
            R["lon_tac"].append(o["lon_logits_tac"].float().softmax(-1).cpu())
            R["fan16"].append(o["anchor_traj"].half().cpu())
            R["reach"].append(o["reach_keep"].bool().cpu())
            scored_j = (a.eval_split == "train") or ("turn" in D["_flags"][j]) or ("sample" in D["_flags"][j])
            if ctrl_on and scored_j and D["_cls"][j] in ("turnL", "turnR", "straight", "gentle"):
                sd_, sh_ = [], []
                for f in (L1.LAT_V7.index("TURN_L"), L1.LAT_V7.index("TURN_R"), L1.LAT_V7.index("LANE_KEEP")):
                    TAC["force"] = f
                    OUT.clear()
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        H.forward_like_trainer(model, b, DEV, args.mode, tr)
                    oc = OUT["out"]
                    want = {L1.LAT_V7.index("TURN_L"): 1, L1.LAT_V7.index("TURN_R"): -1}.get(f, 0)
                    pick_dir = int(L1.dir_class(L1.terminal_heading(oc["traj"][0].float().cpu().numpy())))
                    fan_dir = L1.dir_class(L1.terminal_heading(oc["anchor_traj"][0].float().cpu().numpy()))
                    sd_.append(pick_dir)
                    sh_.append(float((fan_dir == want).mean()))
                TAC["force"] = None
                R["ctrl_sel_dir"].append(torch.tensor([sd_]))
                R["ctrl_fan_dir_share"].append(torch.tensor([sh_]))
            elif ctrl_on:
                R["ctrl_sel_dir"].append(torch.full((1, 3), -9))
                R["ctrl_fan_dir_share"].append(torch.full((1, 3), float("nan")))
            if j % 500 == 0:
                print(f"[r1-arms] {a.arm} eval seed {seed} ZZ{j}-{n}ZZ {time.time() - t0:.0f}s", flush=True)
            if (time.time() - t0) > a.budget_s:
                raise SystemExit(f"[r1-arms] eval budget exceeded at window {j} -- nothing written for seed {seed}")
        out = {k: torch.cat(v, 0) for k, v in R.items() if v}
        out["win_sha12"] = D["win_sha12"]
        out["b_r1_i"] = D["b_r1_i"]
        out["nav_cmd_used"] = (torch.as_tensor(nav) if nav is not None else D["b_nav_cmd"])
        if nav_reason is not None:
            out["nav_reason"] = nav_reason
        out["arm"], out["seed"], out["spec_sha256"] = (a.tag or a.arm), seed, SPEC_SHA256
        out["heads_arm"], out["guidance_w"] = a.arm, float(a.guidance_w)
        torch.save(out, str(fp) + ".tmp")
        os.replace(str(fp) + ".tmp", fp)
        res_all[seed] = {"n": n, "wall_s": round(time.time() - t0, 1)}
        print(f"[r1-arms] {a.arm} eval seed {seed} DONE {time.time() - t0:.0f}s", flush=True)
    return res_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["timing", "train", "eval"])
    ap.add_argument("--arm", default="H1", choices=sorted(ARMS))
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--out", default="/home/nvidia/refcv8_r1/arms")
    ap.add_argument("--ckpt", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt")
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--timing-steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--eval-seeds", default="0,1")
    ap.add_argument("--ctrl", action="store_true")
    ap.add_argument("--budget-s", type=float, default=2400.0)
    ap.add_argument("--device", default="cuda", help="cpu: a code-path smoke only; writes under --out, never an arm")
    ap.add_argument("--max-eval-windows", type=int, default=0, help="smoke only")
    ap.add_argument("--guidance-w", type=float, default=1.0, help="SPEC_R1_A2-2: scale of the tactical-condition term")
    ap.add_argument("--tag", default="", help="output arm name (A2-2: e.g. H5g_w3 evaluates H5's heads at w = 3)")
    ap.add_argument("--heads-dir", default="", help="load <heads-dir>/<arm>/heads.pt (A2-1 stages heads in /dev/shm)")
    ap.add_argument("--eval-split", default="eval", choices=["eval", "train"],
                    help="train = SPEC_R1 sec. 7's TRAIN-FIT diagnostic (1,000 own training windows) -> trainfit_s<seed>.pt")
    a = ap.parse_args()
    global DEV
    DEV = a.device
    if DEV != "cuda" and "smoke" not in str(a.out):
        raise SystemExit("[r1-arms] --device cpu only with an --out path containing 'smoke'")
    spec = HERE / "SPEC_R1.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest() != SPEC_SHA256:
        raise SystemExit("[r1-arms] the shipped SPEC_R1.md differs from the REGISTERED one")
    if a.arm == "H1n" or float(a.guidance_w) != 1.0 or a.eval_split == "train":
        s2 = HERE / "SPEC_R1_A2.md"
        if not s2.exists() or hashlib.sha256(s2.read_bytes()).hexdigest() != SPEC_A2_SHA256:
            raise SystemExit("[r1-arms] an A2 arm needs the REGISTERED SPEC_R1_A2.md beside this script")
    if a.phase == "timing":
        a.arm = "H1"
        rec = train(a, a.timing_steps, timing_only=True)
        st = rec["step_s_mean_after_5"]
        steps = 1320 if st <= 1.35 else int(math.floor(1800.0 / st))
        rec.update({"rule": "1,320 if step <= 1.35 s else floor(1,800 / step_s) (SPEC_R1 registration note)",
                    "measured_step_s": st, "steps_per_arm": steps, "nothing_saved_or_scored": True})
        p = Path(a.out) / "timing_smoke.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps(rec, indent=1), flush=True)
        return 0
    if a.phase == "train":
        if a.steps <= 0:
            t = json.loads((Path(a.out) / "timing_smoke.json").read_text())
            a.steps = int(t["steps_per_arm"])
        rec = train(a, a.steps)
        print(json.dumps(rec, indent=1), flush=True)
        return 0
    res = evaluate(a)
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
