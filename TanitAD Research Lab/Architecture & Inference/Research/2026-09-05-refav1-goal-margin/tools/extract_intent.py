#!/usr/bin/env python
"""Bank `intent` — the goal head's INPUT — so the head can be re-fit at zero GPU.

`stack/tanitad/refs/refa_v1.py:1218`:
    lat_head = nn.Sequential(LayerNorm(d_int), Linear(d_int, n_lat))
applied at `:1956` to `intent`, which `plan()` builds at `:2156` via
    brains = self._run_brains(pooled_win, nav_cmd)          # <-- ego=None
⇒ THE GOAL HEAD IS ONE LINEAR MAP OVER A 1024-d VECTOR, and that vector is a
function of VISION and NAV ONLY. Bank the vector and the whole Step-2 re-fit is
a CPU linear fit — the trunk and the world model are not merely "frozen", they
are NOT IN THE OPTIMISATION AT ALL.

⛔ P3 ADMISSIBILITY, MEASURED NOT ASSUMED. `plan()` passes `ego=None`, so no ego
channel reaches the goal decode; `nav_cmd` is the only non-vision input. Because
nav on PhysicalAI is ultimately supplied from the ego's own future path, this
script banks `intent` TWICE per window -- once under true nav and once under
SHUFFLED nav (`refav1_arm.shuffle_nav`, the D-REFAV1-NAV-DEPTH control) -- so a
re-fit can be scored under both and a fit that leans on the route signal is
visible rather than assumed away.

Also banked: `gt_kappa` per window, computed from the episode POSES by the same
expression `gt_kappa.py` uses (`ff._seq_geometry(g, 0.2)`; yaw_rate.mean /
speed.mean.clamp_min(0.5)), so a dense grid gets a geometric target without a
banked dump. Labels may use ego (PI 2026-08-03); inference may not, and does not.

⛔ CONTROLS (a failure is a refusal, not a warning):
  C1  on the stride-40 sub-grid the emitted `lat_logits` must reproduce the
      banked `logits_stride40.npz` to < 1e-4  -- proves this is the same
      forward pass that produced every banked number.
  C2  the emitted `gt_kappa` must reproduce `gt_kappa_282.npz` to < 1e-4 on the
      same windows -- proves the geometric target is the banked definition.
  C3  a SAME-BREATH non-degenerate check: the intent matrix must have non-zero
      variance on > 90 % of its columns, and its row norms must not be constant.
      (A pre-allocated array that never got written reads as a clean zero bank;
      CLAUDE.md's memmap-of-zeros trap.)
  C4  nav shuffle must actually change some rows; `n_changed` is printed.

Run:
  python extract_intent.py --ckpt ... --config ... --cache ... --episodes ...
      --labels ... --nav ... --armtool-dir <taniteval/tools> --window-stride 2
      --assert-logits <logits_stride40.npz> --assert-gt-kappa <gt_kappa_282.npz>
      --out <out.npz>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _p(*a):
    print(*a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--nav", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-stride", type=int, default=2)
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--horizon-k", type=int, default=10)
    ap.add_argument("--wm-k", type=int, default=30)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--nav-shuffle-seed", type=int, default=0)
    ap.add_argument("--allow-nonstrict", action="store_true")
    ap.add_argument("--armtool-dir", required=True)
    ap.add_argument("--assert-logits", default=None)
    ap.add_argument("--assert-gt-kappa", default=None)
    ap.add_argument("--gt-dt", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.abspath(a.armtool_dir))
    from refav1_arm import (load_model, build_loader, episode_names,
                            shuffle_nav, gt_waypoints)
    from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
    from taniteval import four_families as ff

    import tanitad
    _p(f"[tree] tanitad imported from {tanitad.__file__}")

    t0 = time.time()
    dev = a.device
    model, cfg, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
    k = int(a.horizon_k)
    k_wm = int(a.wm_k) or int(cfg.op_steps)
    k_loader = max(k, k_wm)
    names = episode_names(a.cache)
    if a.episodes_n:
        names = names[:int(a.episodes_n)]
    ld = build_loader(a, cfg, k_loader, names)
    vv = getattr(cfg, "tac_vocab_version", "v6.0")
    lat_names = list(tactical_lat_actions(vv))
    lon_names = list(tactical_lon_actions_v(vv))
    _p(f"[model] step={prov['step']} params={prov['trainable_parameters']:,} "
       f"vocab={vv} d_state={cfg.d_state} device={dev}")

    stride = max(1, int(a.window_stride))
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % stride == 0]
    if not sel:
        raise SystemExit("stride selected zero windows")
    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)
            nav_valid[i] = nid is not None
    nav_shuf, nav_stats = shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)
    _p(f"[grid] stride={stride} windows={len(sel)} episodes="
       f"{len(set(e for _, e, _ in sel))}  nav_shuffle={nav_stats}")

    by_ep: dict[int, list] = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    keys = ("lat_logits", "lon_logits", "route_logits", "lat_label", "lon_label",
            "route_label", "ws", "v0", "eid", "clip_index", "nav_cmd",
            "nav_shuf", "nav_valid", "gt_kappa")
    out: dict[str, list] = {kk: [] for kk in keys}
    intent_l, intent_s_l = [], []
    clip_names: list[str] = []

    ld._order, ld._cursor = [0], 0
    for fi, ei in enumerate(sorted(by_ep)):
        nm = ld.names[ei]
        o = torch.load(ld.episode_dir / f"{nm}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        poses = o["poses"].float()
        F_ep, v_ep, kap_ep = ld._episode(nm)
        clip_names.append(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            nav_t = (torch.tensor([int(nav_true[i])], device=dev)
                     if ld._nav_on else None)
            nav_s = (torch.tensor([int(nav_shuf[i])], device=dev)
                     if ld._nav_on else None)
            with torch.no_grad():
                field = model.encode(feats)
                pooled = field.mean(dim=-2)
                # ⭐ EXACTLY plan()'s call at refa_v1.py:2156 — ego=None.
                br = model._run_brains(pooled, nav_t)
                if br is None:
                    raise SystemExit("no brains — checkpoint has no hierarchy")
                lat = model.lat_head(br["intent"])[0]
                lon = model.lon_head(br["intent"])[0]
                rou = br["route_logits"][0]
                brs = model._run_brains(pooled, nav_s)
                # -- the geometric target, from POSES only (no model) --------
                g = gt_waypoints(poses, t, k)                    # [1,k,2]
                Gg = ff._seq_geometry(g.float().cpu(), a.gt_dt)
                gk = float((Gg["yaw_rate"].mean(1)
                            / Gg["speed"].mean(1).clamp_min(0.5))[0])
            intent_l.append(br["intent"][0].float().cpu().numpy())
            intent_s_l.append(brs["intent"][0].float().cpu().numpy())
            out["lat_logits"].append(lat.float().cpu().numpy())
            out["lon_logits"].append(lon.float().cpu().numpy())
            out["route_logits"].append(rou.float().cpu().numpy())
            for src in ("lat_label", "lon_label", "route_label"):
                v = b.get(src)
                out[src].append(-100 if v is None else int(v[0]))
            out["ws"].append(int(t))
            out["v0"].append(float(v_ep[2 * t]))
            out["eid"].append(fi)
            out["clip_index"].append(ei)
            out["nav_cmd"].append(int(nav_true[i]))
            out["nav_shuf"].append(int(nav_shuf[i]))
            out["nav_valid"].append(bool(nav_valid[i]))
            out["gt_kappa"].append(gk)
        if (fi + 1) % 10 == 0:
            _p(f"  [{fi + 1}/{len(by_ep)}] {len(out['ws'])} windows "
               f"{time.time() - t0:.0f}s")

    arr = {k2: np.asarray(v) for k2, v in out.items()}
    I = np.asarray(intent_l, dtype=np.float32)
    Is = np.asarray(intent_s_l, dtype=np.float32)
    n = len(arr["ws"])
    _p(f"[done] {n} windows, intent {I.shape}, {time.time() - t0:.0f}s")

    # ------------------------------ CONTROLS ------------------------------ #
    ctrl: dict = {}

    # C3 — the bank is not a field of zeros
    colvar = I.var(0)
    rown = np.linalg.norm(I, axis=1)
    ctrl["C3_intent_nondegenerate"] = {
        "frac_columns_with_nonzero_var": float((colvar > 0).mean()),
        "row_norm_min": float(rown.min()), "row_norm_max": float(rown.max()),
        "row_norm_is_constant": bool(np.allclose(rown, rown[0])),
        "abs_mean": float(np.abs(I).mean()),
        "passes": bool((colvar > 0).mean() > 0.90 and not
                       np.allclose(rown, rown[0]) and np.abs(I).mean() > 0),
    }
    # C4 — the shuffle changed rows
    ch = int((arr["nav_cmd"] != arr["nav_shuf"]).sum())
    ctrl["C4_nav_shuffle_changed_rows"] = {
        "n_changed": ch, "n": n, "frac": ch / max(n, 1),
        "intent_differs": bool(not np.allclose(I, Is)),
        "passes": bool(ch > 0 and not np.allclose(I, Is))}

    key = {(int(c), int(w)): i for i, (c, w) in
           enumerate(zip(arr["clip_index"], arr["ws"]))}

    # C1 — reproduce the banked logits
    if a.assert_logits:
        Z = np.load(a.assert_logits, allow_pickle=False)
        idx, ref = [], []
        for j, (c, w) in enumerate(zip(Z["clip_index"], Z["ws"])):
            i = key.get((int(c), int(w)))
            if i is not None:
                idx.append(i); ref.append(j)
        idx, ref = np.array(idx), np.array(ref)
        if len(idx) == 0:
            raise SystemExit("C1 CONTROL: zero overlap with the banked logits "
                             "grid -- refusing (an empty check is not a pass)")
        d = float(np.abs(arr["lat_logits"][idx]
                         - Z["lat_logits"][ref].astype(np.float64)).max())
        am_here = arr["lat_logits"][idx].argmax(-1)
        am_ref = Z["lat_logits"][ref].argmax(-1)
        ctrl["C1_reproduces_banked_lat_logits"] = {
            "n_overlap": int(len(idx)), "max_abs_diff": d,
            "argmax_agreement": f"{int((am_here == am_ref).sum())}/{len(idx)}",
            "passes": bool(d < 1e-4 and (am_here == am_ref).all())}
        _p(f"[C1] n={len(idx)} max|dlogit|={d:.3e} "
           f"argmax {int((am_here == am_ref).sum())}/{len(idx)}")

    # C2 — reproduce the banked gt_kappa
    if a.assert_gt_kappa:
        G = np.load(a.assert_gt_kappa, allow_pickle=False)
        idx, ref = [], []
        for j, (c, w) in enumerate(zip(G["clip_index"], G["ws"])):
            i = key.get((int(c), int(w)))
            if i is not None:
                idx.append(i); ref.append(j)
        idx, ref = np.array(idx), np.array(ref)
        if len(idx) == 0:
            raise SystemExit("C2 CONTROL: zero overlap with the banked "
                             "gt_kappa grid -- refusing")
        d = float(np.abs(arr["gt_kappa"][idx]
                         - G["gt_kappa"][ref].astype(np.float64)).max())
        yh = (np.abs(arr["gt_kappa"][idx]) > 1e-3)
        yr = (np.abs(G["gt_kappa"][ref]) > 1e-3)
        ctrl["C2_reproduces_banked_gt_kappa"] = {
            "n_overlap": int(len(idx)), "max_abs_diff": d,
            "turn_label_agreement": f"{int((yh == yr).sum())}/{len(idx)}",
            "passes": bool(d < 1e-4 and (yh == yr).all())}
        _p(f"[C2] n={len(idx)} max|dkappa|={d:.3e} "
           f"turnlab {int((yh == yr).sum())}/{len(idx)}")

    failed = [k2 for k2, v in ctrl.items() if not v.get("passes")]
    if failed:
        raise SystemExit(f"CONTROL FAILED: {failed} -- refusing to bank")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **arr, intent=I, intent_navshuf=Is,
                        lat_names=np.array(lat_names),
                        lon_names=np.array(lon_names),
                        clip_names=np.array(clip_names))
    with open(a.out.replace(".npz", "_prov.json"), "w", encoding="utf-8") as fh:
        json.dump({"ckpt": a.ckpt, "step": prov.get("step"), "cache": a.cache,
                   "episodes": a.episodes, "window_stride": stride,
                   "n_windows": int(n), "n_episodes": len(by_ep),
                   "device": dev, "d_intent": int(I.shape[1]),
                   "lat_names": lat_names, "lon_names": lon_names,
                   "brains_call": "model._run_brains(pooled, nav) with ego=None "
                                  "-- exactly plan() at refa_v1.py:2156",
                   "nav_shuffle_seed": a.nav_shuffle_seed,
                   "nav_shuffle_stats": {k3: (int(v3) if isinstance(v3, (int, np.integer))
                                              else v3)
                                         for k3, v3 in dict(nav_stats).items()}
                   if isinstance(nav_stats, dict) else str(nav_stats),
                   "controls": ctrl,
                   "wallclock_s": time.time() - t0}, fh, indent=1)
    _p(f"[wrote] {a.out}")
    _p("[CONTROLS] " + json.dumps({k2: v.get("passes") for k2, v in ctrl.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
