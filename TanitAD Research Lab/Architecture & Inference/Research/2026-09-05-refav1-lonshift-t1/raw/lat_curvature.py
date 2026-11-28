"""LATERAL **curvature** MAE as a PAIRED, CI'd delta — the one form in which the
binding four-families rule's "curvature error" is currently missing.

⚠️ SCOPED PRECISELY, because the loose version of this claim is false.
`four_families` ALREADY reports `curvature_mae_1pm` / `curvature_bias_1pm` in each
arm's ABSOLUTE panel (banked `wk15.cl` reads 0.0310 / -0.0071 over
`n_steps_curvature` 271, `excluded_below_min_ds` 96). What does NOT exist is
curvature in the **paired** panel: `refav1_arm._FAMILY_OF` lists only
`LAT_cross_mae_m`, `LAT_heading_mae_deg`, `LAT_yaw_rate_mae_radps`, so curvature
gets **no paired delta and no interval**, and every cross-arm LATERAL comparison
has therefore been made without it. ⇒ the gap is the ESTIMATOR, not the metric.

⭐ It is computed POST HOC from the banked dumps — `four_families._seq_geometry`
already returns `curvature` (`dh/ds`), it was simply never reduced to a per-window
MAE. So this adds the metric with **zero GPU** and **without touching the instrument
that produced the arms**, which keeps every arm bit-comparable to the banked panel.

⚠️ MASKING IS LOAD-BEARING AND IS REPORTED. `four_families` masks heading / yaw /
curvature where a step's displacement is below `MIN_DS_MPS` (curvature is undefined
as ds -> 0). This reduces over `Pg.valid & Gg.valid` exactly as the heading metric
does, and PRINTS the valid fraction per arm — its own docstring warns that "a
curvature number computed over a silently-shrinking subset is not comparable".

Usage:  lat_curvature.py <stack> <taniteval> NAME=DUMPDIR [NAME=DUMPDIR ...] -- B-A [B-A ...]
"""
import json
import math
import os
import sys

import numpy as np


def main(argv):
    stack, tev = argv[0], argv[1]
    sys.path.insert(0, stack)
    sys.path.insert(0, tev)
    sys.path.insert(0, os.path.join(tev, "tools"))
    rest = argv[2:]
    specs = [x for x in rest if "=" in x and not x.startswith("-")]
    pairs = [x for x in rest if "-" in x and "=" not in x]

    import torch
    from taniteval import four_families as ff
    from taniteval import ci as CI
    from refav1_paired_delta import load_dump, cat

    names, dumps = [], {}
    for s in specs:
        nm, _, path = s.partition("=")
        names.append(nm)
        dumps[nm] = load_dump(path)

    ref = names[0]
    fr, Er, mr = dumps[ref]
    G = cat(Er, "g")
    dt = float(mr.get("grid", {}).get("dt_s", 0.2))

    # ---- the windows must be THE SAME, asserted, never assumed --------------
    for nm in names[1:]:
        _, E, _ = dumps[nm]
        for k in ("ws", "v0", "g", "clip_index"):
            o, v = cat(E, k), cat(Er, k)
            if o.shape != v.shape or not np.array_equal(o, v):
                raise SystemExit(f"REFUSED: {nm} is not on {ref}'s windows ({k} differs)")
    eid = np.concatenate([[e["_eid"]] * len(e["ws"]) for e in Er])

    arms = {}
    for nm in names:
        arms[f"{nm}.cl"] = cat(dumps[nm][1], "cl")
    for fl in ("ha", "ha0", "ha0_ext", "ol"):
        if all(fl in e for e in Er):
            arms[fl] = cat(Er, fl)

    gt = torch.as_tensor(G).float()
    Gg = ff._seq_geometry(gt, dt)

    comps, validfrac = {}, {}
    for k, v in arms.items():
        pt = torch.as_tensor(v).float()
        Pg = ff._seq_geometry(pt, dt)
        # ⛔ `curvature` is a PAIR quantity of shape [n, H-1] and its mask is
        # `pair_valid` (= valid[:,1:] & valid[:,:-1]), NOT the per-step `valid`
        # of shape [n, H]. MEASURED: valid is (3,11) where curvature is (3,10),
        # so masking with `valid` mis-aligns or raises rather than being merely
        # imprecise.
        both = Pg["pair_valid"] & Gg["pair_valid"]
        d = (Pg["curvature"] - Gg["curvature"]).abs()
        assert d.shape == both.shape, (d.shape, both.shape)
        nv = both.sum(1)
        mae = torch.where(nv > 0, (d * both).sum(1) / nv.clamp_min(1),
                          torch.full_like(nv, float("nan"), dtype=d.dtype))
        comps[k] = mae.numpy().astype(np.float64)
        validfrac[k] = float((nv > 0).float().mean())

    out = {"metric": "LAT_curvature_mae_invm", "family": "lateral",
           "units": "1/m", "dt_s": dt,
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "n_boot": 2000,
           "note": ("post hoc from the banked dumps; four_families._seq_geometry's own "
                    "curvature (dh/ds), masked by Pg.valid & Gg.valid exactly as the "
                    "heading metric is"),
           "valid_window_fraction": validfrac,
           "means": {k: float(np.nanmean(v)) for k, v in comps.items()},
           "n_windows": int(len(eid)), "paired": {}}

    todo = []
    for p in pairs:
        b, _, x = p.partition("-")
        todo.append((f"{x}.cl" if x in names else x, f"{b}.cl" if b in names else b))
    for nm in names:                       # every arm against every shared floor
        for fl in ("ha", "ha0", "ha0_ext"):
            if fl in arms:
                todo.append((fl, f"{nm}.cl"))
    todo.append((f"{ref}.cl", f"{ref}.cl"))          # KNOWN-VALUE CONTROL

    seen = set()
    for x, y in todo:
        if (x, y) in seen or x not in comps or y not in comps:
            continue
        seen.add((x, y))
        a_v, b_v = comps[x], comps[y]
        keep = np.isfinite(a_v) & np.isfinite(b_v)
        if keep.sum() == 0:
            out["paired"][f"{y} - {x}"] = {"status": "REFUSED",
                                           "reason": "no window finite for both arms"}
            continue
        e = [t for t, kp in zip(eid, keep) if kp]
        r = CI.paired_episode_cluster_bootstrap(b_v[keep], a_v[keep], e,
                                                n_boot=2000, seed=0)
        r["n"] = int(keep.sum())
        r["n_dropped_nonfinite"] = int((~keep).sum())
        out["paired"][f"{y} - {x}"] = r

    ctl = out["paired"].get(f"{ref}.cl - {ref}.cl", {})
    out["known_value_control"] = {
        "pair": f"{ref}.cl - {ref}.cl",
        "must_be": "+0.0000 with a zero-width interval",
        "observed": ctl,
        "passes": bool(abs(ctl.get("delta", 1.0)) < 1e-12
                       and abs(ctl.get("lo", 1.0)) < 1e-12
                       and abs(ctl.get("hi", 1.0)) < 1e-12)}

    print("### LATERAL curvature MAE (1/m) — paired episode-cluster bootstrap, "
          f"n = {out['n_windows']} windows")
    print()
    print("| arm | mean curvature MAE (1/m) | valid-window frac |")
    print("|---|---|---|")
    for k in sorted(out["means"]):
        print(f"| `{k}` | {out['means'][k]:.6f} | {validfrac[k]:.4f} |")
    print()
    print("| pair | delta (1/m) | 95% CI | n | separated |")
    print("|---|---|---|---|---|")
    for pr, d in out["paired"].items():
        if d.get("status") == "REFUSED":
            print(f"| `{pr}` | REFUSED | {d['reason']} | - | - |")
            continue
        lo, hi = d.get("lo"), d.get("hi")
        # use the estimator's OWN `separated` flag rather than re-deriving it
        sep = "**yes**" if d.get("separated") else "no"
        print(f"| `{pr}` | {d.get('delta'):+.6f} | [{lo:+.6f}, {hi:+.6f}] "
              f"| {d.get('n')} | {sep} |")
    print()
    print(f"KNOWN-VALUE CONTROL passes: {out['known_value_control']['passes']}")

    dst = os.environ.get("LAT_CURV_OUT")
    if dst:
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"[out] {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
