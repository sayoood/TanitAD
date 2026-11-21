"""E-DATA-HORIZON-1 — the future-action horizon, the window budget, and the
minimum decorrelating time-shift for an O11 same-clip negative.

⛔ WHY THIS EXISTS. `Project Steering/V7_LAUNCH_GATE.md` names one measurement as
the gate on a queued 8.6 h arm's REQUIRED anti-shortcut control:

    "time-shifted negatives from the window's OWN clip ... needs
     `future_actions2` **longer than `o5_k`** -- at k=60 that is >=60+shift steps
     per window. ⚠️ **Measure the dataset's available future-action horizon
     first**; if it equals `o5_k`, this is not free either"

and, one paragraph later, "the dataset's future-action horizon is the first
thing to measure."

This script measures three things off banked artifacts, no GPU:

  A. **T** -- the real per-episode frame count of the `-w120-256x640cyl` v2ep
     cache family. The trainer's own reachability docstring
     (`train_v6_staged.py:2864-2890`) builds a published limit table on T=120,
     flagging that figure INHERITED and "consistent with the `-w120-` cache
     name". A cache name is not a measurement.

  B. **The window budget** -- `t_max = T - window - max_horizon`
     (`tanitad/data/_contract.py:120`), i.e. what raising `max_horizon` costs in
     training windows, and where the corpus is actually exhausted.

  C. **The action autocorrelation r(lag)** -- which sets the MINIMUM time-shift
     at which a same-clip time-shifted "counterfactual" action is actually a
     counterfactual. ⛔ This is the half the gate did not price: a shift that is
     affordable but too small makes the negative nearly the TRUE action, pulls
     the O11 loss toward its floor, and reads as action-blindness that is not
     there -- the exact defect `train_v6_staged.py:3255-3260` already guards
     against for `randperm` fixed points.

Inputs are all banked and local:
  * v2ep episodes  : the `physicalai-val-w120-256x640cyl` build (24 episodes)
  * arm configs    : the four v7-tiny 30k arm `config.json` files

⚠️ SCOPE. The episodes measured are the VAL build of this cache family. The
TRAIN build (`physicalai-train-e438721ae894-w120-256x640cyl`) lives on Thor and
is deliberately NOT touched. The train build's T is corroborated INDEPENDENTLY
inside `RESULT.md` from a figure the gate itself publishes.

Usage:
    python measure_horizon_budget.py --out ../raw/horizon_budget.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

WINDOW = 6  # cfg.predictor.window for every v7-tiny 30k arm (banked configs)

DEFAULT_EPDIR = (r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
                 r"\sp2\cache\physicalai-val-w120-256x640cyl")
DEFAULT_ARMDIR = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
ARMS = ("v7tiny_postrain30k", "v7tiny_k60clip05p30k",
        "v7tiny_k8clip05p30k", "v7tiny_rdw8p30k")


def read_arm_configs(armdir: str) -> dict:
    """The DERIVED `max_horizon` next to the `args.max_horizon` an args-diff sees.

    ⛔ The point of reading both: `args.max_horizon` is `None` on every arm, so a
    diff over `args` reports them IDENTICAL, while the value the windowing
    actually uses differs 20 vs 60.
    """
    out = {}
    for a in ARMS:
        p = os.path.join(armdir, a, "config.json")
        if not os.path.exists(p):
            out[a] = {"_absent": p}
            continue
        raw = open(p, "rb").read()
        j = json.loads(raw.decode("utf-8"))
        ar = j.get("args", {})
        out[a] = {
            "config_md5": hashlib.md5(raw).hexdigest(),
            "stage": j.get("stage"),
            "max_horizon_DERIVED": j.get("max_horizon"),
            "args_max_horizon_AS_DIFFED": ar.get("max_horizon"),
            "o5_k": ar.get("o5_k"), "o1_k": ar.get("o1_k"),
            "o11_k": ar.get("o11_k"), "o11_negs": ar.get("o11_negs"),
            "w_o11_cf": ar.get("w_o11_cf"),
            "window": ar.get("window"), "clip": ar.get("clip"),
            "init_from": ar.get("init_from"), "seed": ar.get("seed"),
            "steps": ar.get("steps"), "batch": ar.get("batch"),
            "v2_cache": ar.get("v2_cache"),
        }
    return out


def load_actions(epdir: str):
    """Return (clip_ids, T array, list of [T,2] action arrays). Actions only --
    the PNG buffer is never decoded, so this is seconds and needs no GPU."""
    fs = sorted(glob.glob(os.path.join(epdir, "*.v2ep.pt")))
    if not fs:
        raise SystemExit(f"no *.v2ep.pt under {epdir}")
    ids, Ts, acts, meta = [], [], [], None
    for f in fs:
        d = torch.load(f, map_location="cpu", weights_only=False)
        a = d["actions"].numpy().astype(np.float64)
        t = int(d["jpeg_len"].shape[0])
        # ⛔ assert on CONTENT, not on presence: a silently truncated or
        # zero-filled action array would otherwise flow straight into r(lag).
        if a.shape != (t, 2):
            raise SystemExit(f"{f}: actions {a.shape} != ({t}, 2)")
        if not np.isfinite(a).all():
            raise SystemExit(f"{f}: non-finite actions")
        if float(np.abs(a).max()) == 0.0:
            raise SystemExit(f"{f}: all-zero actions -- poisoned bank")
        ids.append(d["clip_id"]); Ts.append(t); acts.append(a)
        if meta is None:
            meta = {"codec": d.get("codec"), "image_h": d.get("image_h"),
                    "image_w": d.get("image_w"), "n_stack": d.get("n_stack"),
                    "frame": d.get("frame")}
    return ids, np.asarray(Ts), acts, meta


def window_budget(Ts: np.ndarray, horizons) -> dict:
    """`t_max = T - window - max_horizon`, clamped at 0 (tanitad/data/_contract.py:120)."""
    rows = {}
    for mh in horizons:
        tm = np.maximum(Ts - WINDOW - mh, 0)
        rows[str(mh)] = {
            "windows_per_episode_min": int(tm.min()),
            "windows_per_episode_max": int(tm.max()),
            "windows_per_episode_mean": float(tm.mean()),
            "episodes_contributing_zero_windows": int((tm == 0).sum()),
            "n_episodes": int(len(tm)),
        }
    return rows


def autocorr(acts, lags) -> dict:
    """Pooled Pearson r between a[t] and a[t+lag], per action channel.

    Pooled over episodes with per-episode pairs concatenated first, then centred
    once -- a lag whose pairs come from different episodes is never formed.
    """
    out = {}
    for lag in lags:
        if lag == 0:
            out[str(lag)] = {"steer_road_rad": 1.0, "accel_mps2": 1.0,
                             "n_pairs": int(sum(a.shape[0] for a in acts))}
            continue
        chans = {}
        n_pairs = 0
        for ch, name in ((0, "steer_road_rad"), (1, "accel_mps2")):
            xs, ys = [], []
            for a in acts:
                if a.shape[0] <= lag:
                    continue
                xs.append(a[:-lag, ch]); ys.append(a[lag:, ch])
            x = np.concatenate(xs); y = np.concatenate(ys)
            n_pairs = int(x.size)
            x = x - x.mean(); y = y - y.mean()
            den = np.sqrt((x * x).sum() * (y * y).sum())
            chans[name] = float((x * y).sum() / den) if den > 0 else float("nan")
        chans["n_pairs"] = n_pairs
        out[str(lag)] = chans
    return out


def min_shift_for(auto: dict, channel: str, threshold: float):
    """Smallest lag whose |r| <= threshold, by linear interpolation between the
    measured lags. Returns None if the grid never reaches the threshold."""
    lags = sorted(int(k) for k in auto)
    prev_l, prev_r = None, None
    for l in lags:
        r = abs(auto[str(l)][channel])
        if r <= threshold:
            if prev_l is None:
                return float(l)
            # interpolate between (prev_l, prev_r) and (l, r)
            if prev_r == r:
                return float(l)
            return float(prev_l + (prev_r - threshold) * (l - prev_l) / (prev_r - r))
        prev_l, prev_r = l, r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epdir", default=DEFAULT_EPDIR)
    ap.add_argument("--armdir", default=DEFAULT_ARMDIR)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids, Ts, acts, meta = load_actions(a.epdir)
    lags = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35,
            40, 45, 50, 55, 60, 70, 80]
    auto = autocorr(acts, lags)

    res = {
        "_experiment": "E-DATA-HORIZON-1",
        "_date": "2026-09-02",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_evidence_class": "MEASURED (ours; banked v2ep episodes + banked arm configs)",
        "_tier": "N/A -- a corpus/windowing measurement, not a model read",
        "_scope": ("VAL build of the -w120-256x640cyl v2ep cache family; the "
                   "TRAIN build lives on Thor and was NOT touched. The train "
                   "build's T is corroborated independently in RESULT.md."),
        "episodes": {
            "n": len(ids),
            "dir": a.epdir,
            "clip_ids": ids,
            "T_min": int(Ts.min()), "T_max": int(Ts.max()),
            "T_mean": float(Ts.mean()),
            "T_unique": sorted(set(int(t) for t in Ts)),
            "meta_first_episode": meta,
        },
        "window": WINDOW,
        "window_budget_t_max": window_budget(
            Ts, [20, 30, 40, 45, 60, 80, 100, 120, 140, 160, 180, 190, 195, 200]),
        "action_autocorrelation_r_by_lag": auto,
        "min_decorrelating_shift_steps": {
            f"{ch}@|r|<={thr}": min_shift_for(auto, ch, thr)
            for ch in ("steer_road_rad", "accel_mps2")
            for thr in (0.75, 0.50, 0.25, 0.10)
        },
        "arm_configs": read_arm_configs(a.armdir),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"[E-DATA-HORIZON-1] wrote {a.out}")
    print(f"  T unique          : {res['episodes']['T_unique']}")
    print(f"  windows/ep @mh=20 : {res['window_budget_t_max']['20']['windows_per_episode_mean']:.2f}")
    print(f"  windows/ep @mh=60 : {res['window_budget_t_max']['60']['windows_per_episode_mean']:.2f}")
    for k, v in res["min_decorrelating_shift_steps"].items():
        print(f"  min shift {k:32s}: {v}")


if __name__ == "__main__":
    main()
