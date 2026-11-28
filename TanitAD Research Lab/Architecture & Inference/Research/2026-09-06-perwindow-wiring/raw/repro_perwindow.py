"""P0/P1 evidence: reproduce the 7x6 per-window tactical vocabulary directly
from episode poses -- the input V3Dataset already holds -- and prove the
on-the-fly path equals the cached path.

ASCII-only output (cp1252 console).
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter

import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
sys.path.insert(0, WT + "/stack")
sys.path.insert(0, WT + "/stack/scripts")

import refb_labels                                    # noqa: E402
import v4_labels as V4                                # noqa: E402
from tanitad.train.v4_curriculum import IGNORE_INDEX  # noqa: E402

CACHE = sys.argv[1] if len(sys.argv) > 1 else (
    "C:/Users/Admin/tanitad-data/physicalai/_epcache/"
    "physicalai-train-14231cd29c74")
N_EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400

LAT = list(V4.LAT_TOKENS)
LON = list(V4.LON_TOKENS)


def main() -> int:
    fs = sorted(glob.glob(CACHE + "/*.pt"))[:N_EPS]
    print("CACHE      :", CACHE)
    print("EPISODES   :", len(fs), "(control: must be non-zero)")
    if not fs:
        print("INCONCLUSIVE: no episode files readable")
        return 2

    lat_all: list[int] = []
    lon_all: list[int] = []
    n_win = 0
    n_ep_used = 0
    t_derive = 0.0
    read_fail = 0

    for p in fs:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            poses = d["poses"]
        except Exception:
            read_fail += 1
            continue
        T = poses.shape[0]
        n = T - V4.WINDOW - V4.MAX_HORIZON
        if n <= 0:
            continue
        n_ep_used += 1
        t0 = time.perf_counter()
        for i in range(n):
            L = V4.WINDOW - 1 + i
            lat_all.append(V4.lat_target(poses, L))
            lon_all.append(V4.lon_target(poses, L))
        t_derive += time.perf_counter() - t0
        n_win += n

    print("UNREADABLE :", read_fail, "(must be 0, else INCONCLUSIVE not ABSENT)")
    print("EPISODES USED:", n_ep_used)
    print("WINDOWS n  :", n_win)
    print("d (features): poses[T,4] only -- no cache file, no label artifact")
    print("DERIVE COST: %.3f ms/window  (%.1f s for all %d)"
          % (1000.0 * t_derive / max(n_win, 1), t_derive, n_win))
    print()

    lat = np.array(lat_all)
    lon = np.array(lon_all)

    def report(name, arr, toks):
        cov = float((arr != IGNORE_INDEX).mean())
        print("--- %s : K=%d  coverage=%.4f  (IGNORE=%d)"
              % (name, len(toks), cov, int((arr == IGNORE_INDEX).sum())))
        c = Counter(arr.tolist())
        for i, t in enumerate(toks):
            k = int(c.get(i, 0))
            print("    %-14s idx %d  n=%7d  %6.3f%%"
                  % (t, i, k, 100.0 * k / max(len(arr), 1)))
        # MAJORITY-CLASS CONTROL: what a constant predictor reads.
        present = [i for i in range(len(toks)) if c.get(i, 0) > 0]
        maj = max(present, key=lambda i: c[i]) if present else -1
        valid = arr[arr != IGNORE_INDEX]
        pooled = float((valid == maj).mean()) if len(valid) else float("nan")
        k_present = len(present)
        macro = 1.0 / k_present if k_present else float("nan")
        print("    CONTROL constant-predictor(%s): pooled=%.4f  "
              "macro-recall=%.4f  (== 1/%d = %.4f -- MUST match)"
              % (toks[maj] if maj >= 0 else "none", pooled, macro,
                 k_present, 1.0 / k_present if k_present else float("nan")))
        return cov, {toks[i]: int(c.get(i, 0)) for i in range(len(toks))}, maj

    cov_lat, hist_lat, maj_lat = report("lat_target", lat, LAT)
    print()
    cov_lon, hist_lon, maj_lon = report("lon_target", lon, LON)
    print()

    # ---- EQUIVALENCE CONTROL: on-the-fly path vs the cached (mint_episode) row
    d = torch.load(fs[0], map_location="cpu", weights_only=False)
    poses = d["poses"]
    ep = V4.mint_episode(poses)
    n = poses.shape[0] - V4.WINDOW - V4.MAX_HORIZON
    direct_lat = torch.tensor([V4.lat_target(poses, V4.WINDOW - 1 + i)
                               for i in range(n)], dtype=torch.long)
    direct_lon = torch.tensor([V4.lon_target(poses, V4.WINDOW - 1 + i)
                               for i in range(n)], dtype=torch.long)
    eq_lat = bool(torch.equal(direct_lat, ep["lat_target"]))
    eq_lon = bool(torch.equal(direct_lon, ep["lon_target"]))
    print("EQUIVALENCE (on-the-fly vs mint_episode, ep0, n=%d):" % n)
    print("   lat bitwise-equal:", eq_lat, "  lon bitwise-equal:", eq_lon)
    # same-breath NEGATIVE control: a deliberately shifted derivation MUST differ
    shifted = torch.tensor([V4.lat_target(poses, V4.WINDOW - 1 + i)
                            for i in range(1, n + 1)], dtype=torch.long)
    neg = bool(torch.equal(shifted, ep["lat_target"]))
    print("   NEGATIVE control (anchor shifted by 1) equal:", neg,
          "-- MUST be False, else the comparison is vacuous")

    out = {
        "cache": CACHE, "episodes": n_ep_used, "windows": n_win,
        "unreadable": read_fail,
        "derive_ms_per_window": 1000.0 * t_derive / max(n_win, 1),
        "lat": {"K": len(LAT), "coverage": cov_lat, "hist": hist_lat},
        "lon": {"K": len(LON), "coverage": cov_lon, "hist": hist_lon},
        "equivalence": {"lat": eq_lat, "lon": eq_lon, "negative_control": neg},
    }
    with open("C:/Users/Admin/tanitad-pwwire/repro_perwindow.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("WROTE repro_perwindow.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
