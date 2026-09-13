"""Temporal-consistency filter for single-frame Qwen-Drive detections (0 GPU).

A detection at frame j is KEPT only if a detection exists in frame j-1 or j+1 (0.2 s away) whose centre,
propagated by ITS OWN predicted velocity over the frame gap and moved into frame j's rig frame through the
ego poses, lies within RADIUS of it. Hallucinations are transient; real objects persist.

⛔ Every knob (score floor, radius, support count) is SELECTED on the FIT clips only and REPORTED on the
SCORED clips, beside the unfiltered upstream 0.25 baseline on the same frames -- a rule tuned on the
frames it is scored on manufactures its own gain. First and last frames have one neighbour; they are
kept in the scored set (not dropped), so the filter is charged for its edge behaviour.
Matching as ours_metrics.py: class-agnostic BEV centre <= 2 m, GT and predictions <= 50 m.
"""
import itertools, json
from pathlib import Path
import numpy as np

Q = Path("/home/nvidia/qwendrive/v2")
FIT = ["seq_4fbd97b6a4b7", "seq_73495082f98b"]
SCORED = ["seq_0d90d20036a3", "seq_6924358fafe0"]
FLOORS = [0.10, 0.15, 0.20, 0.25]
RADII = [1.5, 2.5, 4.0]
SUPPORT = [1, 2]            # neighbours (of up to 2) that must confirm


def match_count(pxy, ps, gxy):
    used = np.zeros(len(gxy), bool)
    for i in np.argsort(-ps):
        if not len(gxy):
            break
        d = np.hypot(*(gxy - pxy[i]).T); d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= 2.0:
            used[j] = True
    return int(used.sum())


def load_seq(s):
    sd = Q / s
    poses = json.loads((sd / "poses.json").read_text())
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    frames = []
    for tok in toks:
        pp = Q / f"out_{s}" / f"{tok}.npz"
        if not pp.exists():
            return None
        r, g = np.load(pp), np.load(sd / tok / "gt.npz")
        gb = g["boxes"]; gb = gb[np.hypot(gb[:, 0], gb[:, 1]) <= 50.0]
        t = json.loads((sd / tok / "meta.json").read_text())["t_ref_us"] * 1e-6
        frames.append({"b": r["boxes"], "s": r["scores"], "gt": gb, "T": np.array(poses[tok]["T_world_rig"]), "t": t})
    return frames


def to_frame(b, src, dst):
    """Box centres of frame `src`, propagated by their own velocity to dst's time, in dst's rig frame."""
    dt = dst["t"] - src["t"]
    xy = b[:, :2] + b[:, 7:9] * dt
    p = np.c_[xy, b[:, 2], np.ones(len(b))]
    w = p @ src["T"].T
    return (w @ np.linalg.inv(dst["T"]).T)[:, :2]


def evaluate(seqs, floor=None, radius=None, support=None):
    tp = npred = ngt = 0
    for s in seqs:
        F = load_seq(s)
        if F is None:
            return None
        for j, fr in enumerate(F):
            inr = np.hypot(fr["b"][:, 0], fr["b"][:, 1]) <= 50.0
            if floor is None:                                  # upstream baseline
                keep = inr & (fr["s"] >= 0.25)
            else:
                cand = inr & (fr["s"] >= floor)
                votes = np.zeros(len(fr["b"]), int)
                for k in (j - 1, j + 1):
                    if 0 <= k < len(F):
                        nb = F[k]; nm = nb["s"] >= floor
                        if nm.any():
                            q = to_frame(nb["b"][nm], nb, fr)
                            d = np.hypot(fr["b"][:, None, 0] - q[None, :, 0], fr["b"][:, None, 1] - q[None, :, 1])
                            votes += (d.min(axis=1) <= radius).astype(int)
                need = np.full(len(fr["b"]), support)
                if j == 0 or j == len(F) - 1:
                    need = np.minimum(need, 1)
                keep = cand & (votes >= need)
            tp += match_count(fr["b"][keep, :2], fr["s"][keep], fr["gt"][:, :2]); npred += int(keep.sum()); ngt += len(fr["gt"])
    P, R = tp / max(npred, 1), tp / max(ngt, 1)
    return {"tp": tp, "n_pred": npred, "n_gt": ngt, "precision": round(P, 3), "recall": round(R, 3),
            "f1": round(2 * P * R / max(P + R, 1e-9), 3)}


res = {"fit_sets": FIT, "scored_sets": SCORED}
res["fit_baseline_0.25"] = evaluate(FIT)
grid = []
for fl, ra, su in itertools.product(FLOORS, RADII, SUPPORT):
    e = evaluate(FIT, fl, ra, su); e.update(floor=fl, radius=ra, support=su); grid.append(e)
res["fit_grid"] = grid
best = max(grid, key=lambda e: e["f1"])
res["selected_on_fit"] = {k: best[k] for k in ("floor", "radius", "support", "f1")}
res["scored_baseline_0.25"] = evaluate(SCORED)
res["scored_selected_rule"] = evaluate(SCORED, best["floor"], best["radius"], best["support"])
(Q / "temporal_filter.json").write_text(json.dumps(res, indent=1))
print("FIT baseline 0.25:", res["fit_baseline_0.25"])
print("SELECTED on fit:", res["selected_on_fit"])
print("SCORED baseline 0.25:", res["scored_baseline_0.25"])
print("SCORED selected rule:", res["scored_selected_rule"])
print("ZZTF-DONEZZ")
