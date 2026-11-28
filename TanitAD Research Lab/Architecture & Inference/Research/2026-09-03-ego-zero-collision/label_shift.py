"""E-ARCH-EGOZERO-1 (b) — the LABEL-DISTRIBUTION SHIFT the ego-zero collision injects.

Structural fact established at source (`tanitad/refs/refc_tactical.py:202-242`,
md5 a0b28c1296949f2f64ec61d9e2beaf33 == the pod's):
    lat = f(dyaw)                    <- does NOT read v0
    lon = f(dv, v0, v1), dv = v1-v0  <- READS v0, the dropped channel
        brake_stop <= dv < -1.0  OR  (v1 < 0.3 AND v0 >= 1.0)

So on the 50 % of samples where `ego_dropout` zeroes v0, the longitudinal head
must predict a label that is an explicit function of the value it was denied,
while the lateral head loses an input its label never used.

This script measures, over the EXACT trainer windows, the conditional label
distributions that share the input token "v = 0.0":
  * TRAIN, input==0 : a MIXTURE — 95.7 % withheld (label ~ overall marginal)
                                +  4.3 % genuinely stationary
  * EVAL,  input==0 : dropout is off, so it is 100 % genuinely stationary
and reports the divergence between them, per head, with the lateral head as the
comparison arm.

Reads the same manifest the trainer's provider hands the dataset.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack")
from tanitad.refs import refc_tactical as tac  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, MAX_HORIZON, HORIZON = 8, 20, tac.LABEL_HORIZON   # 8, 20, 20
P = 0.5                                                    # refc.py:429
CHUNK = 200


def labels_for(man):
    """(lat, lon, v0) over every trainer window, in index order."""
    lat_l, lon_l, v0_l = [], [], []
    poses = man["poses"]
    for i0 in range(0, len(poses), CHUNK):
        pl, fp = [], []
        for p in poses[i0:i0 + CHUNK]:
            a = torch.as_tensor(np.asarray(p, dtype=np.float32))
            T = a.shape[0]
            n = T - WINDOW - MAX_HORIZON
            for t in range(n):
                pl.append(a[t + WINDOW - 1])
                fp.append(a[t + WINDOW:t + WINDOW + HORIZON])
        pl_t = torch.stack(pl)
        fp_t = torch.stack(fp)
        lat, lon = tac.window_factored_labels(pl_t, fp_t, horizon=HORIZON)
        lat_l.append(lat.numpy())
        lon_l.append(lon.numpy())
        v0_l.append(pl_t[:, 3].numpy())
    return (np.concatenate(lat_l), np.concatenate(lon_l),
            np.concatenate(v0_l).astype(np.float64))


def dist(x, k):
    c = np.bincount(x, minlength=k).astype(np.float64)
    return c / c.sum(), c.astype(int)


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / np.clip(q[m], 1e-12, None))).sum())


def analyse(man, split):
    lat, lon, v0 = labels_for(man)
    n = lat.size
    Z = v0 == 0.0
    q = float(Z.mean())
    out = {"split": split, "n_windows": int(n), "n_clips": len(man["poses"]),
           "P_v0_exactly_0": q,
           "chance_CE_3class": float(np.log(3))}
    for name, y, k in (("lat", lat, tac.N_LAT), ("lon", lon, tac.N_LON)):
        classes = (tac.LAT_CLASSES if name == "lat" else tac.LON_CLASSES)
        p_all, c_all = dist(y, k)
        p_z, c_z = dist(y[Z], k)                 # genuinely stationary
        p_nz, _ = dist(y[~Z], k)
        # what the TRAIN model sees behind the token "input == 0":
        #   withheld (any v0, so the OVERALL marginal) with weight P/(P+(1-P)q)
        #   genuine zero                             with weight (1-P)q/(...)
        denom = P + (1 - P) * q
        w_withheld = P / denom
        p_train_at0 = w_withheld * p_all + (1 - w_withheld) * p_z
        out[name] = {
            "classes": list(classes),
            "marginal": {"probs": p_all.tolist(), "counts": c_all.tolist(),
                         "entropy_nats": ent(p_all)},
            "given_v0_exactly_0": {"probs": p_z.tolist(),
                                   "counts": c_z.tolist(), "n": int(Z.sum()),
                                   "entropy_nats": ent(p_z)},
            "given_v0_nonzero": {"probs": p_nz.tolist()},
            # the two meanings of the SAME input token
            "train_at_input0": {"probs": p_train_at0.tolist(),
                                "share_withheld": w_withheld},
            "eval_at_input0": {"probs": p_z.tolist(),
                               "share_withheld": 0.0},
            "shift_train_vs_eval_at_input0": {
                "total_variation": tv(p_train_at0, p_z),
                "kl_eval_from_train_nats": kl(p_z, p_train_at0)},
            # how far the stationary population is from the corpus marginal
            "shift_stationary_vs_marginal": {
                "total_variation": tv(p_z, p_all),
                "kl_nats": kl(p_z, p_all)},
        }
    # the joint fact that makes the longitudinal case structural, not
    # correlational: the brake_stop STOP-BRANCH is gated on v0 >= MOVING_V_MS
    v1 = None
    out["stop_branch"] = {
        "rule": "brake_stop <= dv < -1.0 OR (v1 < 0.3 AND v0 >= 1.0)",
        "MOVING_V_MS": tac.MOVING_V_MS, "STOP_V_MS": tac.STOP_V_MS,
        "DV_BRAKE_MS": tac.DV_BRAKE_MS, "DV_ACCEL_MS": tac.DV_ACCEL_MS,
        "note": "a window with v0 == 0 can NEVER take the stop branch "
                "(it needs v0 >= 1.0), and dv = v1 - 0 >= 0 so it can never "
                "take the dv branch either => brake_stop is UNREACHABLE at "
                "v0 == 0, while it has full support in the withheld mixture.",
        "brake_count_at_v0_zero": int(
            (lon[Z] == tac.LON_BRAKE_STOP).sum()),
    }
    out["lateral_label_reads_v0"] = False
    out["longitudinal_label_reads_v0"] = True
    return out


def main():
    res = {}
    for split, fn in (("train", "train_manifest.pt"), ("eval", "eval_manifest.pt")):
        man = torch.load(os.path.join(HERE, fn), map_location="cpu",
                         weights_only=False)
        res[split] = analyse(man, split)
        r = res[split]
        print(f"[{split}] n={r['n_windows']} q(v0==0)={r['P_v0_exactly_0']:.4%}")
        for h in ("lat", "lon"):
            d = r[h]
            print(f"  {h}: marginal {np.round(d['marginal']['probs'],4).tolist()} "
                  f"H={d['marginal']['entropy_nats']:.4f} | given v0==0 "
                  f"{np.round(d['given_v0_exactly_0']['probs'],4).tolist()} "
                  f"| TV(train@0, eval@0)={d['shift_train_vs_eval_at_input0']['total_variation']:.4f}")
        print(f"  brake_stop windows at v0==0: "
              f"{r['stop_branch']['brake_count_at_v0_zero']}")
    with open(os.path.join(HERE, "label_shift.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote label_shift.json")


if __name__ == "__main__":
    main()
