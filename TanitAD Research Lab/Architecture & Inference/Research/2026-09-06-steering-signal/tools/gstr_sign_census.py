"""D-GSTR-1 P1 -- the sign histogram of g_str's TARGET and its OUTPUT, side by side.

ZERO GPU. Reads the BANKED 141-episode dump (which carries `ep_poses`, `ws`,
`pose_last` and `gstr_nav_true`) and re-derives the TRAINING LABEL for
`str_goal_head` -- `refc.RefCModel.goal_targets` over
`tanitad.data.lan.lan_window_features` -- on EXACTLY the same windows.

⛔ Controls that must read a KNOWN value, or the census is inadmissible:
  C1 INDEX     `ep_poses[idx]` must equal the dumped `pose_last` bit-for-bit on
               every window. A wrong index would silently census a different
               window than the one g_str was measured on.
  C2 MIRROR    negating the world y-axis (y -> -y, yaw -> -yaw) must FLIP the
               target's lateral sign on every window with a non-zero one. A probe
               that reads something other than lateral cannot pass this.
  C3 STRAIGHT  a synthetic straight-ahead path must give sin == 0 exactly.

ASCII-only output (cp1252-safe).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/nvidia/navpred/stack")
from tanitad.data.lan import (LAN_FEATS_PER_ANCHOR, LanConfig,  # noqa: E402
                              lan_window_features)

DUMP = os.environ.get("GSTR_DUMP",
                      "/home/nvidia/navpred/navflip_dump/decisions")
TOP = os.path.dirname(DUMP)
TAU = float(os.environ.get("TAU_G", "0.10"))     # contingency.py's deadband
OUT = os.environ.get("GSTR_OUT", "/home/nvidia/navroute/GSTR_SIGN_CENSUS.json")

L, S, R = -1, 0, 1
NAMES = {L: "LEFT", S: "STRAIGHT", R: "RIGHT"}


def cls(sin_vals, tau):
    """contingency.py's `dir_from_y`, on the lateral component."""
    out = np.zeros(len(sin_vals), dtype=np.int64)
    out[sin_vals > tau] = L          # +y is LEFT (dump ORIENTATION_CONTROL)
    out[sin_vals < -tau] = R
    return out


def goal_target_from_feats(f, k):
    """numpy mirror of `refc.RefCModel.goal_targets` for ONE window."""
    f = np.asarray(f, dtype=np.float64).reshape(k, LAN_FEATS_PER_ANCHOR)
    valid = f[:, 3] > 0.5
    any_valid = bool(valid.any())
    first = int(np.argmax(valid.astype(np.float64)))
    picked = f[first]
    n = np.linalg.norm(picked[:2])
    bearing = picked[:2] / max(n, 1e-6)
    span = max(k - 1, 1)
    dist_pref = first / span * 2.0 - 1.0
    return bearing, dist_pref, any_valid, first


def census(mirror=False):
    cfg = LanConfig()
    k = cfg.k
    files = sorted(glob.glob(os.path.join(DUMP, "*.npz")))
    tgt_sin, tgt_cos, out_sin, valids, firsts = [], [], [], [], []
    gt_y, eps, dist_t = [], [], []
    idx_ok = idx_bad = 0
    offsets = {}
    for f in files:
        base = os.path.basename(f)
        top = os.path.join(TOP, base)
        if not os.path.exists(top):
            continue
        t = np.load(top, allow_pickle=True)
        z = np.load(f, allow_pickle=True)
        # `ep_poses` lives in the DECISIONS file (refcv3_arm.py:1912); the
        # top-level file carries `clip_index`, the join key.
        if "ep_poses" not in z.files or "ws" not in z.files:
            continue
        if "gstr_nav_true" not in z.files or "pose_last" not in z.files:
            continue
        poses = np.asarray(z["ep_poses"], dtype=np.float64)          # [T, 4]
        ws = np.asarray(z["ws"]).astype(np.int64).reshape(-1)
        ci = (int(np.asarray(t["clip_index"]).reshape(-1)[0])
              if "clip_index" in t.files else -1)
        g = np.asarray(z["gstr_nav_true"], dtype=np.float64)         # [N, 3]
        pl = np.asarray(z["pose_last"], dtype=np.float64).reshape(len(ws), 4)
        gfe = (np.asarray(z["gt_future_ext"], dtype=np.float64)
               if "gt_future_ext" in z.files else None)

        # ---- C1: resolve the index convention by POSITIVE ASSERTION ----
        off = None
        for cand in range(0, 32):
            j = ws + cand
            if j.max() >= poses.shape[0]:
                continue
            if np.allclose(poses[j], pl, atol=1e-5, rtol=0):
                off = cand
                break
        if off is None:
            idx_bad += len(ws)
            continue
        offsets[off] = offsets.get(off, 0) + len(ws)
        idx_ok += len(ws)

        p = poses.copy()
        if mirror:
            p[:, 1] = -p[:, 1]
            p[:, 2] = -p[:, 2]
        for i, w in enumerate(ws.tolist()):
            j = w + off
            feats = lan_window_features(p, j, cfg)
            b, dp, av, fi = goal_target_from_feats(feats, k)
            tgt_cos.append(b[0])
            tgt_sin.append(b[1])
            dist_t.append(dp)
            valids.append(av)
            firsts.append(fi)
            out_sin.append(g[i, 1])
            eps.append(ci)
            if gfe is not None:
                gy = float(gfe[i, -1, 1]) if gfe.ndim == 3 else np.nan
            else:
                gy = np.nan
            gt_y.append(-gy if mirror else gy)
    return {"tgt_sin": np.array(tgt_sin), "tgt_cos": np.array(tgt_cos),
            "out_sin": np.array(out_sin), "valid": np.array(valids),
            "first": np.array(firsts), "gt_y": np.array(gt_y),
            "ep": np.array(eps), "dist_t": np.array(dist_t),
            "idx_ok": idx_ok, "idx_bad": idx_bad, "offsets": offsets}


def hist(sin_vals, tau):
    c = cls(sin_vals, tau)
    return {NAMES[q]: int((c == q).sum()) for q in (L, S, R)}


def pct(x):
    if x.size == 0:
        return {}
    q = np.percentile(x, [0, 1, 25, 50, 75, 99, 100])
    return {"min": round(float(q[0]), 4), "p1": round(float(q[1]), 4),
            "p25": round(float(q[2]), 4), "median": round(float(q[3]), 4),
            "p75": round(float(q[4]), 4), "p99": round(float(q[5]), 4),
            "max": round(float(q[6]), 4),
            "frac_negative": round(float((x < 0).mean()), 4),
            "frac_positive": round(float((x > 0).mean()), 4)}


def main():
    A = census(mirror=False)
    n = A["tgt_sin"].size
    res = {
        "tool": "D-GSTR-1 P1 sign census: g_str TARGET vs OUTPUT",
        "dump": DUMP,
        "tier": "T1 (self-action OPEN loop, 2026-09-02 ruling) for the OUTPUT; "
                "the TARGET is a LABEL, no tier applies",
        "evidence_class": "MEASURED (ours)",
        "tau_g": TAU,
        "n_windows": int(n),
        "n_episodes": int(np.unique(A["ep"]).size),
        "label_fn": "tanitad.data.lan.lan_window_features -> "
                    "refc.RefCModel.goal_targets (numpy mirror)",
        "CONTROLS": {},
    }
    # ---- C1 ----
    res["CONTROLS"]["C1_index_convention"] = {
        "rule": "ep_poses[ws + off] must equal the dumped pose_last on EVERY "
                "window; a wrong index censuses the wrong window",
        "windows_resolved": int(A["idx_ok"]),
        "windows_unresolved": int(A["idx_bad"]),
        "offsets_used": {str(k2): int(v) for k2, v in A["offsets"].items()},
        "PASS": bool(A["idx_bad"] == 0 and A["idx_ok"] > 0),
    }
    # ---- C3 straight-ahead ----
    straight = np.zeros((400, 4))
    straight[:, 0] = np.arange(400) * 1.0
    straight[:, 3] = 10.0
    fs = lan_window_features(straight, 0, LanConfig())
    b3, _, av3, _ = goal_target_from_feats(fs, LanConfig().k)
    res["CONTROLS"]["C3_straight_path"] = {
        "rule": "a straight path must give lateral EXACTLY 0.0 and be valid",
        "lateral": float(b3[1]), "cos": float(b3[0]), "any_valid": bool(av3),
        "PASS": bool(abs(float(b3[1])) < 1e-9 and av3),
    }
    # ---- C2 mirror ----
    B = census(mirror=True)
    nz = np.abs(A["tgt_sin"]) > 1e-9
    flipped = np.allclose(B["tgt_sin"][nz], -A["tgt_sin"][nz], atol=1e-6)
    res["CONTROLS"]["C2_mirror_world"] = {
        "rule": "negating world y (and yaw) must flip the TARGET's lateral sign "
                "on every window; a probe reading a non-lateral slot cannot pass",
        "n_nonzero": int(nz.sum()),
        "mirrored_hist": hist(B["tgt_sin"], TAU),
        "PASS": bool(flipped),
    }

    v = A["valid"]
    res["LABEL_VALIDITY"] = {
        "any_valid_frac": round(float(v.mean()), 4),
        "n_valid": int(v.sum()), "n_invalid": int(v.size - v.sum()),
        "first_anchor_hist": {str(i): int((A["first"][v] == i).sum())
                              for i in range(LanConfig().k)},
        "arclengths_m": list(LanConfig().arclengths_m),
        "note": "goal_targets picks the FIRST leak-guard-admissible anchor; "
                "strategic_goal_loss is MASKED by any_valid",
    }

    # ---- THE TABLE ----
    res["SIGN_HISTOGRAM"] = {
        "_convention": "+lateral is LEFT (dump ORIENTATION_CONTROL: mean GT "
                       "terminal y +2.8608 under nav=LEFT, -4.1898 under "
                       "nav=RIGHT)",
        "TARGET_all_windows": hist(A["tgt_sin"], TAU),
        "TARGET_valid_only": hist(A["tgt_sin"][v], TAU),
        "OUTPUT_g_str_all_windows": hist(A["out_sin"], TAU),
        "OUTPUT_g_str_valid_only": hist(A["out_sin"][v], TAU),
    }
    res["DISTRIBUTIONS"] = {
        "TARGET_lateral_valid": pct(A["tgt_sin"][v]),
        "TARGET_lateral_all": pct(A["tgt_sin"]),
        "OUTPUT_lateral": pct(A["out_sin"]),
        "TARGET_cos_valid": pct(A["tgt_cos"][v]),
        "OUTPUT_dist_pref": pct(np.array([])),
        "TARGET_dist_pref_valid": pct(A["dist_t"][v]),
    }
    if np.isfinite(A["gt_y"]).any():
        gy = A["gt_y"]
        m = np.isfinite(gy)
        res["GT_CROSSCHECK"] = {
            "rule": "the LABEL's lateral sign must agree with the GT future's "
                    "own terminal lateral offset far more often than chance",
            "n": int(m.sum()),
            "gt_terminal_y_hist": hist(gy[m], 1.0),
            "agree_sign_frac": round(float(
                (np.sign(gy[m & v]) == np.sign(A["tgt_sin"][m & v])).mean()), 4),
        }
    # correlation between target and output
    if v.sum() > 10:
        a, b = A["tgt_sin"][v], A["out_sin"][v]
        res["TARGET_VS_OUTPUT"] = {
            "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "cos_loss_1_minus_cos": round(float(
                (1.0 - (A["tgt_cos"][v] * np.cos(np.arcsin(np.clip(b, -1, 1)))
                        + a * b)).mean()), 4),
            "_note": "r is over the VALID-label windows only",
        }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
