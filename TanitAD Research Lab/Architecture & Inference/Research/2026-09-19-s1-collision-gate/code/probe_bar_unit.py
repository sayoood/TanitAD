"""What is the INDEPENDENT UNIT behind the prereg's 6.06 m near-forward bar?

`raw/box_axis_probe.json` reports **n = 129 near-forward pairs** (623 all-matched) from
**24 windows**. A pair-level interval on 129 would be PSEUDO-REPLICATION: several pairs come from
the same window and are not independent draws. The Master Mind flagged the window-vs-pair error;
this measures the level BELOW it, which decides whether 24 windows are themselves independent:

* if the 24 windows were CONSECUTIVE, adjacent ones would share ``window - 1`` of their frames and
  the effective n would collapse toward the number of CLIPS;
* ``s1_pass.trainer_windows`` is a ``torch.randperm(seed=12345)``, i.e. a scattered draw, so they
  should land in near-distinct episodes — but "should" is not a measurement.

⛔ This reproduces the probe's OWN selection (same corpus, same seed, same eligibility filter,
same ``[:24]``) and reports the episode spread. It reads no model and touches no GPU.

Usage: python probe_bar_unit.py <out.json> [--n 24]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "taniteval" / "tools",
          ROOT / "stack" / "scripts"):
    sys.path.insert(0, str(p))

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"


# ⛔ A CHECKER MUST NOT DIE ON ITS OWN OUTPUT. MEASURED 2026-09-20: three separate
# readouts crashed with a cp1252 `UnicodeEncodeError` on this box mid-print -- one of
# them after reporting "lines lost = 1" but BEFORE naming the line, i.e. it had verified
# nothing while looking like it had. Relying on the caller to export PYTHONIOENCODING is
# a habit; this is a guard. `errors="replace"` means the print degrades instead of
# raising even if the stream cannot take utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 -- a stream that cannot be reconfigured is not fatal
    pass


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    out_path = a[0]
    n_win = int(a[2]) if len(a) > 2 and a[1] == "--n" else 24
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import s1_pass as SP

    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     None)
    # ⛔ byte-for-byte the probe's own selection line
    # ⭐ SWEEP. `torch.randperm(N, seed)` is the same permutation whatever the slice, so a pool
    # of 4,000 has the first 1,000 IDENTICAL to the probe's own call -- the n=24 row therefore
    # reproduces the banked selection exactly and serves as this sweep's CONTROL.
    pool = [w for w in SP.trainer_windows(corp.ds, 4000) if corp.eligibility(w) is None]
    sweep = {}
    for n in (24, 50, 100, 200, 400):
        if n > len(pool):
            sweep[str(n)] = {"note": "pool exhausted (%d eligible)" % len(pool)}
            continue
        ws = pool[:n]
        e = [int(corp.ds.index[w][0]) for w in ws]
        t = [int(corp.ds.index[w][1]) for w in ws]
        cc = Counter(e)
        gaps = []
        for ep in {x for x in e if cc[x] > 1}:
            tt = sorted(tv for ev, tv in zip(e, t) if ev == ep)
            gaps += [b - a_ for a_, b in zip(tt, tt[1:])]
        W0 = int(corp.W)
        sweep[str(n)] = {
            "n_windows": n, "n_distinct_episodes": len(cc),
            "episodes_per_window": round(len(cc) / n, 4),
            "n_overlapping_pairs": sum(1 for g in gaps if g < W0),
            "min_same_episode_gap": (min(gaps) if gaps else None),
            "max_windows_in_one_episode": max(cc.values()),
        }
    wis = pool[:n_win]
    eps = [int(corp.ds.index[w][0]) for w in wis]
    ts = [int(corp.ds.index[w][1]) for w in wis]
    c = Counter(eps)
    # the frame distance between windows that SHARE an episode — the overlap question
    same_ep_gaps = []
    for e in {e for e in eps if c[e] > 1}:
        tt = sorted(t for e2, t in zip(eps, ts) if e2 == e)
        same_ep_gaps += [b - a_ for a_, b in zip(tt, tt[1:])]
    W = int(corp.W)
    rep = {
        "_what": "the INDEPENDENT UNIT behind the 6.06 m near-forward bar",
        "_evidence_class": "MEASURED (ours)",
        "n_windows": len(wis),
        "n_distinct_episodes": len(c),
        "windows_per_episode": dict(sorted(Counter(c.values()).items())),
        "episodes_with_more_than_one_window": sum(1 for v in c.values() if v > 1),
        "window_len_frames": W,
        "same_episode_frame_gaps": sorted(same_ep_gaps),
        "min_same_episode_gap": (min(same_ep_gaps) if same_ep_gaps else None),
        "any_overlapping_pair": bool(same_ep_gaps and min(same_ep_gaps) < W),
        "total_grid_windows": len(corp.ds.index),
        "sweep_over_n": sweep,
        # ⚠️ DENOMINATOR: this is eligible-out-of-the-4,000-WINDOW POOL, NOT out of the
        # grid. I misread my own field once and published "2,882 of 10,600 = 27.2 %";
        # the true grid figure is 71.22 % (raw/eligibility_census.json). The field now
        # carries its own denominator so the number cannot be read without it.
        "n_eligible_in_pool": len(pool),
        "pool_size_drawn": 4000,
        "frac_eligible_OF_POOL": round(len(pool) / 4000.0, 4),
        "selection": "s1_pass.trainer_windows(ds, 1000) randperm seed 12345, "
                     "eligibility-filtered, first %d — the probe's own line" % n_win,
    }
    # ---- CLUSTER SIZES. An episode-clustered bootstrap resamples clusters as EQUAL draws, so a
    # clip contributing 10 windows carries the same weight as one contributing 132. "21 episodes"
    # and "60 episodes" therefore OVERSTATE the effective n whenever the sizes are skewed.
    # ⛔ The right statistic is the EFFECTIVE CLUSTER COUNT (inverse-Simpson / Kish):
    #        n_eff = (sum n_i)^2 / sum n_i^2 , which equals the cluster count only when EQUAL.
    # Measured on two populations: the BAR's own 24-window draw (near-forward targets per episode,
    # which is what governs the landed 5.9539 [4.89, 7.11]) and the full eligible pool.
    import numpy as _np
    _bank = ("C:/Users/Admin/tanitad-caches/p3-prebuild-20260920/"
             "p3_targets_v2ep-eval124clean-416x1024cyl-halfB.npz")
    n_gate_all = _np.load(_bank)["n_gate"]

    def _neff(sizes):
        a = _np.asarray([s for s in sizes if s > 0], dtype=float)
        if a.size == 0:
            return {"n_clusters": 0, "n_eff": 0.0}
        ne = float(a.sum() ** 2 / (a ** 2).sum())
        return {"n_clusters": int(a.size), "total": int(a.sum()),
                "min": int(a.min()), "median": float(_np.median(a)), "max": int(a.max()),
                "n_eff": round(ne, 2), "n_eff_over_n_clusters": round(ne / a.size, 3)}

    bar_pairs, elig = {}, {}
    for w in wis:
        e = int(corp.ds.index[w][0])
        bar_pairs[e] = bar_pairs.get(e, 0) + int(n_gate_all[w])
    for w in pool:
        e = int(corp.ds.index[w][0])
        elig[e] = elig.get(e, 0) + 1
    rep["cluster_sizes"] = {
        "_why": "an episode-clustered bootstrap weights every cluster EQUALLY; unequal sizes make "
                "the effective n smaller than the cluster count",
        "bar_draw_near_forward_targets_per_episode": _neff(bar_pairs.values()),
        "bar_draw_total_cross_check_vs_banked_129": int(sum(bar_pairs.values())),
        "eligible_pool_windows_per_clip": _neff(elig.values()),
    }
    rep["verdict"] = (
        "the 24 windows land in %d distinct episodes; %s"
        % (len(c),
           ("NO two share a window's worth of frames, so they are approximately independent "
            "and the WINDOW is the right bootstrap unit"
            if not rep["any_overlapping_pair"] else
            "at least one pair OVERLAPS in frames, so the window is NOT independent and the "
            "unit must be the EPISODE")))
    Path(out_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print("ZZBARUNIT-DONE ZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
