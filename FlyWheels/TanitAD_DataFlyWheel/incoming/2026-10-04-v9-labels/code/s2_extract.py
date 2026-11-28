"""WP-A Stage 2 (SPEC §9.E + SPEC_ADDENDUM_S2A1): per-window features for the route-checkpoint / nav echo-leak study.

Populations: eval139 = every trainer window (D1 truth table, row == ds index); trainS = a seeded 600-clip TRAIN sample
(seed 20261004) — every window of those clips. The trainer's NOW (D1 `t_now_s`, reproduced bit-exactly by D1) is used.
RC / nav come from the builder (`stack/scripts/build_v9_labels.py`, imported from the 50efa52 extract); everything else
here (GT, future speed, the oracle control, the D2-style excursion) is computed in this file.
CPU only; one clip resident at a time.

usage: python s2_extract.py <eval139|trainS> <out.npz>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\r8_wpa\stack\scripts")
import build_v9_labels as B  # noqa: E402

EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
AUD = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit"
SRC = {
    "eval139": {"table": f"{AUD}/D1_label_census/tables/label_truth_eval139.npz",
                "manifest": "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
                "labels": "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz",
                "labels_md5": "eefc38d1453bd1c73802d44d45affced"},
    "trainS": {"table": f"{AUD}/D1_label_census/tables/label_truth_train.npz",
               "manifest": "D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt",
               "labels": "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz",
               "labels_md5": "b45377a1f25263b5c0f3d318c126b1ac"},
}
# S2-A2 (E2'/E3'): ALL train clips, one window per clip per 2 s (dataset index t mod 20 == 0)
SRC["trainE2p"] = dict(SRC["trainS"])
SLOTS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
TAUS = np.arange(1, 7, dtype=np.float64)
LADDER_KMH = np.array([20, 30, 50, 70, 80, 100, 120, 130], np.float64)
VARS = ("A30", "A50", "A80", "B")
SAMPLE_SEED, N_TRAIN_CLIPS = 20261004, 600
STRAIGHT_SUPPORT_DEG = 10.0          # SPEC_ADDENDUM_S2A1
HALF_SUPPORT_M = 3.0 * 25.0          # the heavy reference's kernel half-width


def d2_excursion(x, y, psi):
    """D2's lat_features_arrays re-typed: peak |lateral residual| after removing the constant-curvature arc through
    the window's net heading change; net heading change (deg)."""
    dpsi = np.degrees(psi - psi[0])
    c, s = math.cos(psi[0]), math.sin(psi[0])
    dx, dy = x - x[0], y - y[0]
    lat = -s * dx + c * dy
    arcs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    L = float(arcs[-1])
    net = math.radians(float(dpsi[-1]))
    kap = net / L if L > 1e-6 else 0.0
    lat_arc = (1.0 - np.cos(kap * arcs)) / kap if abs(kap) > 1e-9 else np.zeros_like(arcs)
    res = lat - lat_arc
    j = int(np.argmax(np.abs(res)))
    return float(abs(res[j])), float(np.sign(res[j])), float(dpsi[-1]), L


def main(split, out):
    import torch
    cfg = SRC[split]
    assert B.file_md5(cfg["labels"]) == cfg["labels_md5"], "label blob md5 mismatch"
    z = np.load(cfg["table"], allow_pickle=True)
    sha_w = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in z["clip_sha12"]])
    t_w = z["t"].astype(np.int64)
    now_w = z["t_now_s"].astype(np.float64)
    widx = z["window_index"].astype(np.int64)
    m = torch.load(cfg["manifest"], map_location="cpu", weights_only=False)
    cid_by_sha = {B.sha12(c): c for c in m["clip_id"]}
    sup = {}
    with gzip.open(cfg["labels"], "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sup[B.sha12(r["clip_id"])] = r.get("turn_suppression")
    clips = sorted(set(sha_w.tolist()))
    if split == "trainS":
        clips = sorted(np.random.default_rng(SAMPLE_SEED).choice(np.array(clips), N_TRAIN_CLIPS, replace=False).tolist())
    keep = np.isin(sha_w, clips)
    if split == "trainE2p":
        keep &= (t_w % 20) == 0
    sha_w, t_w, now_w, widx = sha_w[keep], t_w[keep], now_w[keep], widx[keep]
    N = len(now_w)
    cols: dict = {}

    def put(name, i, val, dtype=np.float64, shape=()):
        if name not in cols:
            cols[name] = np.full((N,) + shape, np.nan if np.issubdtype(dtype, np.floating) else -1, dtype=dtype)
        cols[name][i] = val

    t0 = time.time()
    n_sup_clips = 0
    for ci, s12 in enumerate(clips):
        idx = np.nonzero(sha_w == s12)[0]
        log = B.load_log(EGO.format(cid_by_sha[s12]))
        sp = sup.get(s12)
        n_sup_clips += int(bool(sp) and bool(sp.get("applied")))
        G = B.clip_geo(log, sp)
        p8, p25 = G.paths[B.LITERALS["rc_sigma_m"]], G.paths[B.LITERALS["rc_sigma_heavy_m"]]
        for i in idx:
            now = float(now_w[i])
            x0, y0, psi0, v0 = (float(a) for a in B.sample(G.log, now))
            put("v0", i, v0)
            if now >= 0.5:
                put("a0", i, (v0 - float(B.sample(G.log, now - 0.5)[3])) / 0.5)
            # ---- route checkpoint + nav (the builder) -------------------------------------------------------- #
            rc = B.rc_fields(G.tr, G.log, G.paths, G.turns, now, G.s_nat)
            for k, val in rc.items():
                if k == "rc_b_kind":
                    put("rc_b_kind", i, ("turn_end", "clamped_max", "clamped_min", "no_turn").index(val), np.int8)
                else:
                    put(k, i, float(val))
            s_now = rc["s_now"]
            nav = B.nav_fields(G.turns, s_now, now, v0, G.s_max)
            for k, val in nav.items():
                put(k, i, float(val))
            # ---- E3: delta vs the heavy route at the same arc; straight-support flag ------------------------- #
            for vname in VARS:
                L = rc["rc_b_L_m"] if vname == "B" else float(vname[1:])
                st = s_now + L
                if rc[f"rc_{vname}_valid"] and st <= p25.s_hi_valid:
                    x8, y8 = np.interp(st, p8.s, p8.x), np.interp(st, p8.s, p8.y)
                    xh, yh, ph = np.interp(st, p25.s, p25.x), np.interp(st, p25.s, p25.y), np.interp(st, p25.s, p25.psi)
                    put(f"delta_{vname}", i, (x8 - xh) * -math.sin(ph) + (y8 - yh) * math.cos(ph))
                    lo, hi = max(s_now - HALF_SUPPORT_M, 0.0), min(st + HALF_SUPPORT_M, p25.s[-1])
                    sel = (p25.s >= lo) & (p25.s <= hi)
                    rng = float(np.degrees(p25.psi[sel].max() - p25.psi[sel].min())) if sel.any() else np.nan
                    put(f"support_range_deg_{vname}", i, rng)
            # ---- GT (log) at the 8 slots, NOW frame ------------------------------------------------------------ #
            ts = now + SLOTS
            gx, gy, _, _ = B.sample(G.log, ts)
            lx, ly = B.to_now_frame(gx, gy, x0, y0, psi0)
            put("gt", i, np.stack([lx, ly], 1), shape=(8, 2))
            gv = np.array([(tt <= G.log.t_end) and B.max_gap(G.log, now, tt) <= B.LITERALS["g_max_s"] for tt in ts])
            put("gt_valid", i, gv.astype(np.int8), np.int8, shape=(8,))
            # ---- E2 targets + the oracle control --------------------------------------------------------------- #
            ok6 = (now + 6.0 <= G.log.t_end) and B.max_gap(G.log, now, now + 6.0) <= B.LITERALS["g_max_s"]
            if ok6:
                put("vfut", i, B.sample(G.log, now + TAUS)[3], shape=(6,))
                vmax = float(B.sample(G.log, now + np.arange(0, 6.0001, 0.1))[3].max())
                kmh = vmax * 3.6
                step = LADDER_KMH[np.searchsorted(LADDER_KMH, kmh - 1e-9)] if kmh <= LADDER_KMH[-1] else LADDER_KMH[-1]
                put("o1_ms", i, step / 3.6)
                # E3 secondary: the ego's offset at NOW+6 from the heavy route at the same arc
                s6 = float(B.s_at(G.tr, now + 6.0))
                if s6 <= p25.s_hi_valid:
                    p6x, p6y = (float(a) for a in B.sample(G.log, now + 6.0)[:2])
                    xh, yh, ph = np.interp(s6, p25.s, p25.x), np.interp(s6, p25.s, p25.y), np.interp(s6, p25.s, p25.psi)
                    put("dev6", i, (p6x - xh) * -math.sin(ph) + (p6y - yh) * math.cos(ph))
            # ---- lane-change-scale excursion over [NOW, NOW+8] (E3 split) --------------------------------------- #
            if (now + 8.0 <= G.log.t_end) and B.max_gap(G.log, now, now + 8.0) <= B.LITERALS["g_max_s"]:
                ex, ey, epsi, _ = B.sample(G.log, now + np.arange(0, 8.0001, 0.1))
                eps, sgn, net, Lp = d2_excursion(ex, ey, epsi)
                put("exc_eps_m", i, eps)
                put("exc_net_deg", i, net)
                put("exc_path_m", i, Lp)
        if ci % 50 == 0:
            print(f"[s2_extract {split}] {ci + 1}/{len(clips)} clips, {time.time() - t0:.0f}s", flush=True)
    B.assert_no_g_imports()
    np.savez_compressed(out, clip_sha12=sha_w, t=t_w, t_now_s=now_w, window_index=widx,
                        meta=json.dumps({"split": split, "n_windows": int(N), "n_clips": len(clips),
                                         "n_suppressed_clips": n_sup_clips,
                                         "builder_md5": B.file_md5(B.__file__),
                                         "s2_geom_emit_v7_md5": B.file_md5(os.path.join(os.path.dirname(B.__file__), "s2_geom_emit_v7.py")),
                                         "labels_md5": cfg["labels_md5"], "literals": {k: list(v) if isinstance(v, tuple) else v for k, v in B.LITERALS.items()},
                                         "sample_seed": SAMPLE_SEED if split == "trainS" else None,
                                         "wall_s": round(time.time() - t0, 1)}),
                        **cols)
    print(f"[s2_extract {split}] done: {N} windows, {len(clips)} clips, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
