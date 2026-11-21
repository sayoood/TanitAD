"""E-WC2 preconditions, verified ON THE REAL DUMP — not on the unit-test fixtures.

Two properties decide whether the σ* number means anything, and both have a KNOWN
failure mode that returns a plausible wrong answer rather than an error:

  1. ⛔ **σ IS PER-AXIS, NOT RADIAL.** §3.1 injects ``N(0, s)`` per axis
     (``sel_winners_curse_law.py:221``), so ``s`` is the per-axis SD of an isotropic
     2-D Gaussian. Reading the RADIAL RMS against the same threshold inflates σ by
     **√2 = 1.414** — enough to move a verdict on arithmetic alone. Pinned here by
     reproducing §3.1's OWN published ratios from its own published σ* and fan
     numbers: 0.8 / 0.4714 = 1.70 and 0.8 / 0.1639 = 4.88.

  2. ⛔ **LOEO MUST BE EPISODE-DISJOINT, NOT WINDOW-DISJOINT.** On a stride-8 grid a
     window's neighbours are near-duplicates; a window-disjoint split puts them in
     train and reports a memorisation artefact. This re-runs the ACTUAL σ estimate
     under both schemes on the ACTUAL latents and prints the ratio, so the
     understatement is a measured number on this dump rather than a fixture claim.
     (The fixture measurement was 40×; the real-data figure is emitted below.)

Run (0 GPU, CPU only):
    PYTHONPATH=<repo>/stack;<repo>/stack/scripts;<repo>/taniteval \
    python verify_ewc2_preconditions.py <dump.pt> [<dump2.pt> ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import e_wc2_sigma_star as E


def loeo_disjointness(eid) -> dict:
    """Measured, on this dump's own 881-row episode vector."""
    uids = np.asarray([int(x) for x in eid])
    folds = E.loeo_folds(eid)
    per_fold = {int(f): np.unique(uids[folds == f]).tolist()
                for f in np.unique(folds)}
    sizes = {f: len(v) for f, v in per_fold.items()}
    # every episode appears in exactly one fold, and every fold holds exactly one
    ep_to_folds = {}
    for f, eps in per_fold.items():
        for e in eps:
            ep_to_folds.setdefault(int(e), set()).add(f)
    straddlers = {e: sorted(fs) for e, fs in ep_to_folds.items() if len(fs) > 1}
    return {
        "n_windows": int(uids.size),
        "n_episodes": int(np.unique(uids).size),
        "n_folds": int(np.unique(folds).size),
        "episodes_per_fold_max": int(max(sizes.values())),
        "episodes_per_fold_min": int(min(sizes.values())),
        "one_episode_per_fold": all(v == 1 for v in sizes.values()),
        "episodes_straddling_folds": straddlers,
        "episode_disjoint": not straddlers and all(v == 1 for v in sizes.values()),
        "windows_per_fold_min": int(min((folds == f).sum()
                                        for f in np.unique(folds))),
        "windows_per_fold_max": int(max((folds == f).sum()
                                        for f in np.unique(folds))),
    }


def window_disjoint_folds(n: int, k: int, seed: int = 0) -> np.ndarray:
    """The LEAKY scheme: k random window folds, blind to episode boundaries."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def sigma_under(X, y, folds) -> float:
    r = E.ridge_oof_predict(X, y[:, 0], folds)
    s = E.ridge_oof_predict(X, y[:, 1], folds)
    res = np.stack([y[:, 0] - r["yhat"], y[:, 1] - s["yhat"]], axis=1)
    return E.sigma_from_residuals(res)["sigma_perax_m"]


def leak_control(dump, horizon_idx: int = 0) -> dict:
    """σ under LOEO vs under window-disjoint folds, same X, same y, same dump."""
    X = np.concatenate([np.asarray(dump[b], dtype=np.float64)
                        for b in ("pooled", "ctx")], axis=1)
    y = np.asarray(dump["gt_endpoint"][:, horizon_idx], dtype=np.float64)
    keep = np.asarray(dump["endpoint_valid"][:, horizon_idx], dtype=bool)
    keep &= np.isfinite(y).all(axis=1)
    X, y = X[keep], y[keep]
    eid = [int(x) for i, x in enumerate(dump["eid"]) if keep[i]]
    s_loeo = sigma_under(X, y, E.loeo_folds(eid))
    n_ep = len(set(eid))
    s_win = sigma_under(X, y, window_disjoint_folds(len(eid), n_ep))
    return {"horizon_index": horizon_idx, "n_windows": int(X.shape[0]),
            "n_episodes": n_ep,
            "sigma_perax_LOEO_m": round(s_loeo, 6),
            "sigma_perax_window_disjoint_m": round(s_win, 6),
            "understatement_x": round(s_loeo / s_win, 4) if s_win else None,
            "leaky_scheme_understates": bool(s_win < s_loeo),
            "_why": ("window-disjoint folds put a window's stride-8 NEIGHBOURS in "
                     "train; the σ they report is a memorisation artefact and is "
                     "biased DOWN on exactly the number E-WC2 exists to produce")}


def perax_unit_check() -> dict:
    """§3.1's own published numbers must reproduce §3.1's own published ratios."""
    ref = E.PREREG
    s, ade, orc = (ref["reference_sigma_star_m"], 0.4714, 0.1639)
    per_ax = {"sigma_star_m": s, "sel_ade": ade, "oracle_ade": orc,
              "sigma_over_ade": round(s / ade, 4),
              "sigma_over_oracle": round(s / orc, 4),
              "published_ratio_vs_ade": ref["reference_ratio_vs_ade"],
              "published_ratio_vs_oracle": ref["reference_ratio_vs_oracle"]}
    # ⚠️ Compared AT THE PUBLISHED PRECISION. §3.1 quotes 1.7 and 4.9 to one
    # decimal; 0.8/0.1639 = 4.8810 rounds to 4.9 and an absolute 0.01 tolerance
    # would fail it for being *more* precise than the number it is checked
    # against. The discriminating comparison is against the radial misreading,
    # which lands at 2.4 and 6.9 — nowhere near either published figure.
    per_ax["reproduces_published"] = (
        round(per_ax["sigma_over_ade"], 1) == ref["reference_ratio_vs_ade"]
        and round(per_ax["sigma_over_oracle"], 1)
        == ref["reference_ratio_vs_oracle"])
    radial = s * np.sqrt(2.0)
    per_ax["radial_misreading"] = {
        "sigma_radial_rms_m": round(float(radial), 4),
        "sigma_over_ade_if_radial": round(float(radial / ade), 4),
        "sigma_over_oracle_if_radial": round(float(radial / orc), 4),
        "inflation_x": round(float(np.sqrt(2.0)), 4),
        "_why": ("a radial-RMS reading inflates σ by 1.414 and does NOT reproduce "
                 "§3.1's published 1.70 — that is what pins the unit")}
    return per_ax


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rep = {"_what": "E-WC2 precondition verification, measured on the REAL dumps",
           "perax_unit_check": perax_unit_check(), "dumps": {}}
    for p in a.dumps:
        d = torch.load(p, map_location="cpu", weights_only=False)
        rep["dumps"][Path(p).name] = {
            "loeo": loeo_disjointness(d["eid"]),
            "leak_control_2s": leak_control(d, 0),
            "leak_control_6s": leak_control(d, 1),
        }
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = (rep["perax_unit_check"]["reproduces_published"]
          and all(v["loeo"]["episode_disjoint"] for v in rep["dumps"].values()))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
