"""Re-issue every published single-arm interval under the CORRECT estimator.

360-review W1/P2. The registry's ``± ci95`` numbers came from 8 OVERLAPPING
random 20 % holdouts divided by sqrt(8) ("8-split episode-disjoint jackknife").
This script reads the surviving per-window artifacts
(``results/windows_<key>.pt`` = pred/gt/cv/eid, n=881 over 40 val episodes) and
prints, side by side:

  BEFORE  the deprecated ``overlapping_holdout_se`` (reproduced from the raw
          artifact and CHECKED against the published ``results/<key>.json`` — if
          the reproduction misses, the row is flagged, not quietly printed)
  AFTER   the episode-cluster bootstrap (B=2000 over the 40 episodes)

then runs the PAIRED episode-clustered tests for the head-to-head claims, which
are the comparisons the registry leaderboard actually asserts.

CPU only. No checkpoint, no GPU, no pod.

Usage:
  python taniteval/recompute_ci.py --results <dir with windows_*.pt and *.json>
  python taniteval/recompute_ci.py --results ./results --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/root/taniteval")

from taniteval import ci as C  # noqa: E402

# key -> (label, registry citation for the published number)
ARMS = [
    ("flagship-30k", "flagship v1 (speed+jerk) FINAL", "MODEL_REGISTRY.md:151/:737"),
    ("refc-xl-30k", "REF-C-XL anchored diffusion FINAL", "MODEL_REGISTRY.md:545/:738"),
    ("refb-v2-30k", "REF-B v2 (arch-v2) FINAL", "MODEL_REGISTRY.md:415/:476/:742"),
    ("flagship-speed", "flagship v1 19k relay", "MODEL_REGISTRY.md:180"),
    ("refa-dynin-30k", "REF-A dyn-in 30k (H4 closure)", "MODEL_REGISTRY.md:360"),
    ("refb-v2-20k", "REF-B v2 @20k", "MODEL_REGISTRY.md:415"),
    ("refc-xl", "REF-C-XL @16k", "MODEL_REGISTRY.md:545"),
    ("flagship-nospeed", "flagship pre-speed", "MODEL_REGISTRY.md §1"),
    ("flagship-v2-6k", "flagship v2 @6k (killed arm)", "flagshipv2-6k-diagnostic"),
    ("refc-v12", "REF-C v1.2 learned rescorer", "refc-v12 note §4.5"),
]

# head-to-head claims the registry asserts, as (A, B, claim)
PAIRS = [
    ("flagship-30k", "refc-xl-30k",
     "leaderboard rows 1 vs 2 — 'REF-C-XL finishes 0.006 m behind flagship v1'"),
    ("flagship-30k", "refb-v2-30k", "H1: flagship vs REF-B v2"),
    ("flagship-30k", "refa-dynin-30k", "D-A5 / H4: flagship vs REF-A (frozen encoder)"),
    ("refc-xl-30k", "refb-v2-30k", "REF-C-XL vs REF-B v2"),
    ("flagship-30k", "flagship-speed", "v1 30k vs its own 19k relay"),
]

N_BOOT = 2000


def ade_0_2s_per_window(pred, gt):
    """Per-window ADE over the 4 waypoint horizons — the ``ade_0_2s`` component."""
    de = torch.linalg.norm(pred - gt, dim=-1).numpy().astype(np.float64)
    return de.mean(axis=1)


def naive_published(win, val_frac=0.2, n_splits=8, seed=0):
    """Reproduce the DEPRECATED published interval from the raw artifact.

    Uses the program's own ``split_by_episode`` so the reproduction is the real
    protocol, not a lookalike. Returns (mean, ci95) or None if the split
    primitive is unavailable (pod-path import).
    """
    try:
        from tanitad.eval.gates import split_by_episode
    except Exception:                                            # noqa: BLE001
        return None
    pred, gt, eid = win["pred"], win["gt"], win["eid"]
    means = []
    for s in range(seed, seed + n_splits):
        _tr, va = split_by_episode(eid, val_frac, s)
        va = torch.tensor(va)
        de = torch.linalg.norm(pred[va] - gt[va], dim=-1)
        means.append(float(de.mean()))
    v = np.asarray(means)
    return float(np.nanmean(v)), C.overlapping_holdout_se(v)


def main():
    ap = argparse.ArgumentParser("recompute_ci")
    ap.add_argument("--results", default="results",
                    help="dir holding windows_<key>.pt and <key>.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--json", default=None, help="write the table as JSON")
    a = ap.parse_args()
    R = Path(a.results)

    rows, loaded = [], {}
    for key, label, cite in ARMS:
        wp = R / f"windows_{key}.pt"
        if not wp.exists():
            print(f"[skip] {key}: no {wp.name}")
            continue
        win = torch.load(wp, map_location="cpu", weights_only=False)
        loaded[key] = win
        eid = win["eid"]
        v = ade_0_2s_per_window(win["pred"], win["gt"])

        pub = None
        jp = R / f"{key}.json"
        if jp.exists():
            d = json.loads(jp.read_text())
            pub = d["heldout"]["model"]["ade_0_2s"]

        rep = naive_published(win)
        boot = C.episode_cluster_bootstrap(v, eid, n_boot=a.n_boot, seed=0)

        row = {"key": key, "label": label, "cite": cite,
               "n_windows": int(len(v)), "n_episodes": boot["n_episodes"],
               "published_mean": pub["mean"] if pub else None,
               "published_ci95": pub["ci95"] if pub else None,
               "repro_mean": round(rep[0], 4) if rep else None,
               "repro_ci95": round(rep[1], 4) if rep else None,
               "full_set_mean": round(float(v.mean()), 4),
               "boot_ci95": boot["ci95"], "boot_lo": boot["lo"],
               "boot_hi": boot["hi"], "boot_se": boot["se"]}
        if pub and rep:
            row["repro_ok"] = (abs(rep[0] - pub["mean"]) < 5e-4
                               and abs(rep[1] - pub["ci95"]) < 5e-4)
        row["widen_x"] = (round(boot["ci95"] / pub["ci95"], 2)
                          if pub and pub["ci95"] else None)
        rows.append(row)

    print("\n=== SINGLE-ARM INTERVALS: BEFORE (overlapping_holdout_se) vs "
          f"AFTER (episode_cluster_bootstrap, B={a.n_boot}) ===\n")
    hdr = (f"{'key':<18}{'pub mean':>9}{'pub ±':>9}{'repro':>7}"
           f"{'full-set':>10}{'boot ±':>9}{'boot 95% CI':>20}{'widen':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ok = "ok" if r.get("repro_ok") else ("MISS" if r.get("repro_ok") is False else "-")
        ci = f"[{r['boot_lo']:.4f}, {r['boot_hi']:.4f}]"
        print(f"{r['key']:<18}{r['published_mean'] or float('nan'):>9.4f}"
              f"{r['published_ci95'] or float('nan'):>9.4f}{ok:>7}"
              f"{r['full_set_mean']:>10.4f}{r['boot_ci95']:>9.4f}{ci:>20}"
              f"{(r['widen_x'] or float('nan')):>7.2f}")

    print("\n=== HEAD-TO-HEAD: PAIRED episode-clustered bootstrap "
          "(same 881 windows, so pair them) ===\n")
    pairs_out = []
    for ka, kb, claim in PAIRS:
        if ka not in loaded or kb not in loaded:
            continue
        wa, wb = loaded[ka], loaded[kb]
        assert wa["eid"] == wb["eid"], f"{ka}/{kb} not aligned on the same windows"
        va = ade_0_2s_per_window(wa["pred"], wa["gt"])
        vb = ade_0_2s_per_window(wb["pred"], wb["gt"])
        p = C.paired_episode_cluster_bootstrap(vb, va, wa["eid"],
                                               n_boot=a.n_boot, seed=0)
        ca = C.episode_cluster_bootstrap(va, wa["eid"], n_boot=a.n_boot, seed=0)
        cb = C.episode_cluster_bootstrap(vb, wb["eid"], n_boot=a.n_boot, seed=0)
        quad = float(np.hypot(ca["ci95"], cb["ci95"]))
        corr = float(np.corrcoef(va, vb)[0, 1])
        rec = {"a": ka, "b": kb, "claim": claim, "delta_b_minus_a": p["delta"],
               "lo": p["lo"], "hi": p["hi"], "ci95": p["ci95"],
               "separated": p["separated"], "p_gt0": p["p_delta_gt0"],
               "unpaired_quadrature_ci95": round(quad, 4),
               "per_window_corr": round(corr, 3),
               "power_gain_x": round(quad / p["ci95"], 2) if p["ci95"] else None}
        pairs_out.append(rec)
        sep = "SEPARATED" if p["separated"] else "NOT separated"
        print(f"{kb} - {ka}: delta {p['delta']:+.4f} m  CI [{p['lo']:+.4f}, "
              f"{p['hi']:+.4f}]  {sep}  (p>0 {p['p_delta_gt0']:.3f}; "
              f"corr {corr:.3f}; unpaired-quadrature ± {quad:.4f} -> paired ± "
              f"{p['ci95']:.4f}, {rec['power_gain_x']}x tighter)")
        print(f"    claim: {claim}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"estimator": "episode_cluster_bootstrap", "n_boot": a.n_boot,
             "arms": rows, "pairs": pairs_out}, indent=2))
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
