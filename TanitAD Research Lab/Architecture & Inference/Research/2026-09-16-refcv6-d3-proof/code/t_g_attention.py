#!/usr/bin/env python3
"""T-G -- ATTENTION ATTRIBUTION. Where does the SELECTED anchor look?

    python t_g_attention.py --max-windows 600 --out raw/t_g.json

The operative decoder's image cross-attention is `refc.CrossAttnLayer.cross`
(`nn.MultiheadAttention`, `refc.py:1289`), called at `:1315` with
`need_weights=False`. This script wraps that ONE call per layer so it is made with
`need_weights=True` and the averaged head weights are captured; nothing else changes
(the returned attention output is the module's own, so the forward is unaltered).

CONCENTRATION. For the SELECTED anchor's attention row over the 8 x 20 = 160 image
tokens:

    concentration = (mass on the lead's token COLUMNS) / (those columns' area share)

Area share = n_lead_columns / 20. ⭐ A UNIFORM row reads EXACTLY 1.0 -- that is the
registered control, and it is computed by the SAME function on a uniform vector, not
asserted.

The lead's token columns come from the same projection T-B masks with
(`leadmask.Projector`), so the two tests address the same pixels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

REPO = os.environ.get("D3_REPO", r"C:\Users\Admin\refcv5cmp\repo")
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), REPO,
          os.path.join(REPO, "stack", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leadmask as LM  # noqa: E402

CKPT = r"C:\Users\Admin\refcv5v2_final\ckpt.pt"
CONFIG = r"C:\Users\Admin\refcv5v2_final\config.json"
EPS = r"C:\Users\Admin\refcv5cmp\data\eval"
LABELS = r"C:\Users\Admin\refcv5cmp\data\s2_labels_v7.2_eval.jsonl.gz"
LEAD_BLOCK = r"C:\Users\Admin\refcv5cmp\data\b1_eval_lead_block.npz"
AGENT_JOIN = r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz"
EXTRINSICS = r"C:\Users\Admin\refcv5v2_final\extrinsics141.json"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def concentration(row: np.ndarray, cols: set, n_col: int) -> float:
    """row [n_tokens] attention mass (sums to 1) over an (n_row x n_col) token grid."""
    g = row.reshape(-1, n_col)
    share = len(cols) / float(n_col)
    if share <= 0:
        return float("nan")
    return float(g[:, sorted(cols)].sum() / share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=600)
    ap.add_argument("--gap-max-m", type=float, default=30.0)
    ap.add_argument("--infer-seed", type=int, default=0)
    ap.add_argument("--window-stride", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arm = _mod("refcv3_arm_real",
               os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
    tr = arm.trainer()
    torch.manual_seed(a.infer_seed)
    np.random.seed(a.infer_seed)
    dev = "cuda"
    model, cfg, targs, prov = arm.load_model(CKPT, CONFIG, dev, False)
    model.eval()
    steps = int(prov["decoder_steps"])
    W = int(cfg.core.window)

    ns = types.SimpleNamespace(
        episodes=EPS, labels=LABELS, lru=8, episodes_n=0, nav_source="v72")
    eps, files, clip_ids, ds, lman, join, nav_src, raw_off = \
        arm.build_corpus(ns, cfg, prov)

    # -- wrap the image cross-attention of every decoder layer ---------------- #
    CAP = {}
    layers = []
    dec = model.core.decoder
    for name, m in dec.named_modules():
        if m.__class__.__name__ == "CrossAttnLayer":
            layers.append((name, m))
    if not layers:
        raise SystemExit("no CrossAttnLayer found on the decoder")
    for li, (nm, lyr) in enumerate(layers):
        orig = lyr.cross.forward

        def wrapped(q, k, v, *args, _orig=orig, _li=li, **kw):
            kw = dict(kw)
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            o, w = _orig(q, k, v, *args, **kw)
            CAP[_li] = w.detach().float().cpu().numpy()
            return o, None
        lyr.cross.forward = wrapped
    print(f"[t-g] {len(layers)} CrossAttnLayer(s): {[n for n, _ in layers]}",
          flush=True)

    # -- lead geometry, from T-B's own projector ------------------------------ #
    keep = set(clip_ids)
    agents = LM.load_agents(AGENT_JOIN, keep)
    agents.pop("_stats")
    leads = LM.load_leads(LEAD_BLOCK, keep)
    proj = LM.Projector(EXTRINSICS)
    FRW = proj.FR.width

    stride = max(1, int(a.window_stride))
    sel = [(wi, e_i, t) for wi, (e_i, t) in enumerate(ds.index) if wi % stride == 0]
    feed_ego = bool(getattr(cfg, "ego_state_inject", False))
    from tanitad.refs import refc_v3 as v3mod

    rows, n_seen = [], 0
    rng = np.random.default_rng(a.infer_seed)
    grid_rows = grid_cols = None
    for (wi, e_i, t) in sel:
        if n_seen >= a.max_windows:
            break
        cid = clip_ids[e_i]
        t0 = t + W - 1
        raw = t0 + raw_off
        ld = leads.get((cid, raw))
        if ld is None or not ld[0] or not ld[1]:
            continue
        gap = ld[2]
        if not (gap == gap and gap <= a.gap_max_m):
            continue
        agrows = agents.get((cid, raw)) or []
        tgt = [r for r in agrows if r[0] == ld[1]]
        if not tgt:
            continue
        _, cx, cy, yaw, l, w_, cls = tgt[0]
        bb = proj.bbox(cid, cx, cy, yaw, l, w_, LM.CLASS_H_M.get(cls, LM.DEFAULT_H_M))
        if bb is None:
            continue
        item = ds[wi]
        fr = tr.frames_to_device(item["frames"][None], dev)
        nid = ds._nav_by_sid.get(int(ds.episodes[e_i].episode_id))
        nav_t = torch.tensor([0 if nid is None else int(nid)], dtype=torch.long,
                             device=dev)
        v0 = float(item["pose_last"].float()[3])
        v0_t = torch.full((1,), v0, dtype=torch.float32, device=dev)
        ego = (v3mod.ego_state_from_batch(
            {"pose_last": item["pose_last"].float()[None],
             "actions": item["actions"].float()[None]}, device=dev)
            if feed_ego else None)
        CAP.clear()
        with torch.no_grad():
            out = model(fr, nav_cmd=nav_t, v0=v0_t, steps=steps, ego_state=ego)
        si = int(out["sel_idx"][0])
        if not CAP:
            raise SystemExit("the attention wrapper captured nothing")
        if grid_cols is None:
            n_tok = CAP[max(CAP)].shape[-1]
            gh, gw = cfg.core.encoder.grid_hw() if hasattr(
                cfg.core.encoder, "grid_hw") else (n_tok // 20, 20)
            grid_rows, grid_cols = int(n_tok // int(gw)), int(gw)
            print(f"[t-g] token grid {grid_rows} x {grid_cols} = {n_tok}", flush=True)
        c0, c1 = bb[0], bb[1]
        tc = set(range(max(c0 * grid_cols // FRW, 0),
                       min(c1 * grid_cols // FRW + 1, grid_cols)))
        if not tc:
            continue
        # area-matched random column band, away from the lead's columns
        width = len(tc)
        cand = [s for s in range(0, grid_cols - width + 1)
                if not (set(range(s, s + width)) & tc)]
        _rs = int(rng.choice(cand)) if cand else -1
        rc = set(range(_rs, _rs + width)) if _rs >= 0 else set()
        rec = {"sha12": LM.sha12(cid), "raw_frame": int(raw), "gap_m": round(gap, 3),
               "sel_idx": si, "n_lead_cols": width,
               "area_share": round(width / grid_cols, 4)}
        for li in sorted(CAP):
            wgt = CAP[li]                       # [B, n_anchors, n_tokens]
            row = wgt[0, si]
            rec[f"L{li}_conc_lead"] = concentration(row, tc, grid_cols)
            if rc:
                rec[f"L{li}_conc_rand"] = concentration(row, rc, grid_cols)
            rec[f"L{li}_conc_uniform_control"] = concentration(
                np.full_like(row, 1.0 / row.size), tc, grid_cols)
            rec[f"L{li}_row_sum"] = float(row.sum())
        rows.append(rec)
        n_seen += 1
        if n_seen % 100 == 0:
            print(f"  {n_seen} windows", flush=True)

    if not rows:
        raise SystemExit("no scoreable window")
    lyr_keys = sorted({int(k[1:k.index("_")]) for r in rows for k in r
                       if k.startswith("L") and "_conc_lead" in k})
    eidv = [r["sha12"] for r in rows]
    from taniteval.ci import (episode_cluster_bootstrap,
                              paired_episode_cluster_bootstrap)
    summary = {}
    for li in lyr_keys:
        cl = np.array([r[f"L{li}_conc_lead"] for r in rows])
        cr = np.array([r.get(f"L{li}_conc_rand", np.nan) for r in rows])
        cu = np.array([r[f"L{li}_conc_uniform_control"] for r in rows])
        m = np.isfinite(cl) & np.isfinite(cr)
        summary[f"layer{li}"] = {
            "lead_concentration_mean": round(float(cl.mean()), 4),
            "lead_concentration_ci": episode_cluster_bootstrap(cl, eidv, n_boot=2000, seed=0),
            "random_band_concentration_mean": round(float(np.nanmean(cr)), 4),
            "uniform_control": {"mean": float(cu.mean()), "min": float(cu.min()),
                                "max": float(cu.max()),
                                "known_value": 1.0,
                                "max_abs_dev_from_1": float(np.abs(cu - 1.0).max())},
            "lead_minus_random_paired": paired_episode_cluster_bootstrap(
                cl[m], cr[m], [eidv[i] for i in np.nonzero(m)[0]],
                n_boot=2000, seed=0),
            "attention_row_sum_mean": round(
                float(np.mean([r[f"L{li}_row_sum"] for r in rows])), 6)}
    out = {"_what": "T-G attention attribution on the SELECTED anchor",
           "tier": "T1 (self-action OPEN loop)",
           "n_windows": len(rows), "n_clips": len(set(eidv)),
           "window_stride": stride, "gap_max_m": a.gap_max_m,
           "infer_seed": a.infer_seed,
           "token_grid": [grid_rows, grid_cols],
           "definition": ("mass on the lead's token COLUMNS / (n_lead_cols / n_cols); "
                          "a uniform row reads EXACTLY 1.0"),
           "note": ("batch = 1 here (one nav conditioning), so this forward's ddim "
                    "noise -- and therefore its sel_idx -- is this script's own, "
                    "not the banked roll's. The attribution is self-consistent: "
                    "the row read is the row the anchor this forward SELECTED used."),
           "layers": summary, "per_window": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str)[:2500])
    print("[out]", a.out)


if __name__ == "__main__":
    main()
