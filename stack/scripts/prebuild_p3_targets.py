#!/usr/bin/env python3
"""prebuild_p3_targets.py — P3's gate-relevant supervision targets, built on CPU, banked.

`PREREG_PERCEPTION_BOX_QUALITY.md`'s **P3-SUPERVISION** swaps the target population from the
32-nearest **360°** set to the **gate-relevant** one (``x ∈ [0, 60] m, |y| ≤ 16 m`` — the decode's
own range and the population :mod:`box_quality` gates on). Everything that swap needs is a pure
function of the agent join and the window grid, so it costs **zero GPU** and can be banked before
the arm starts: the arm then spends its card time training instead of indexing.

⛔ THE GRID MUST BE THE TRAINER'S OWN, NOT A RE-DERIVATION. The window index is built through
``refcv3_arm.build_corpus`` — the same call the trainer-faithful eval path uses — and then
CROSS-CHECKED against literals a real run wrote into its own ``config.json``
(``agent_join_stats.train``: ``n_windows``, ``n_windows_labelled``, ``n_target_boxes_prefilter``).
Re-running a producer's own derivation and finding agreement measures determinism, not
correctness; these three literals come from a different process on a different day, so they are an
independent reference. **A mismatch is reported INCONCLUSIVE and nothing is banked** — a target
bank built on a grid that is not the consumer's grid is the artifact-scope trap with the object
swapped.

⭐ WHAT THIS MEASURES BEYOND THE CENSUS. The A7/P configuration is ``--agent-pad 32`` while the
join carries up to ``reader_max_agents_per_frame`` boxes per frame (153 on halfA, 395 on halfB).
Two budgets could bite BEFORE any loss is computed, so the census COUNTS both per window rather
than assuming either:

* ``pad`` (32) — the dataset keeps the nearest 32 and COUNTS the rest as truncated;
* ``queries`` — the set loss matches ``min(n_target, n_query)`` rows, so on a window with more
  valid targets than queries the surplus is unmatched and unsupervised.

⚠️ ``--queries`` DEFAULTS TO THE **box3d** HEAD'S 100, NOT to ``--agent-queries``. MEASURED
2026-09-20, after I had written the opposite into a draft finding: the refcv6 box decoder is built
with ``n_queries=100`` (``box3d_head.N_QUERIES_DEFAULT``, and a real run stamps
``refcv6_perception.n_queries: 100`` into its own ``config.json``), while ``--agent-queries 16``
governs the SEPARATE 2-D agents head. Reading 16 here produces a query-budget "confound" in P3
that **does not exist** — 100 queries is above the pad in every window, so the query budget never
binds on the head this pre-build serves. Pass ``--queries`` explicitly for any other head.

Clip ids never leave this file in the clear: sha12 only.

Usage
-----
    python stack/scripts/prebuild_p3_targets.py --config <a refcv6 run's config.json> \
        --v2-cache <halfA|halfB> --agent-join <join.jsonl.xz> --labels <v8 labels> \
        --out <dir> [--expect-windows N --expect-labelled N --expect-boxes N] [--bank]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "taniteval" / "tools",
          ROOT / "stack" / "scripts"):
    sys.path.insert(0, str(p))

#: ⛔ The gate-relevant population, IDENTICAL to `box_quality.population_mask("near_forward")`.
#: One spelling, or the arm optimises a set the scorer does not read.
X_FWD_M, Y_HALF_M = 60.0, 16.0


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def gate_mask(cx, cy) -> np.ndarray:
    """``x ∈ [0, 60] ∧ |y| ≤ 16``, inclusive at both edges (`box_quality`'s own boundaries)."""
    cx, cy = np.asarray(cx, dtype=np.float64), np.asarray(cy, dtype=np.float64)
    return (cx >= 0.0) & (cx <= X_FWD_M) & (np.abs(cy) <= Y_HALF_M)


def nearest_n(cx, cy, n: int) -> np.ndarray:
    """The dataset's own truncation rule: ``argsort(hypot(x, y))[:n]`` (``_agent_item``)."""
    return np.argsort(np.hypot(np.asarray(cx, dtype=np.float64),
                               np.asarray(cy, dtype=np.float64)))[:int(n)]


def census(ds, eps, reader, window: int, *, pad: int, queries: int) -> dict:
    """One pass over the window grid. Returns the census plus the per-window selection."""
    n_raw, n_gate, n_kept_raw, n_kept_gate, lab = [], [], [], [], []
    sel_off, sel_idx = [0], []
    t0 = time.time()
    for e_i, t in ds.index:
        ep = eps[e_i]
        ag = reader.lookup(int(ep.episode_id), int(t) + int(window) - 1)
        if ag is None:                                   # NO_LABEL ≠ labelled-clear
            lab.append(False)
            n_raw.append(0); n_gate.append(0); n_kept_raw.append(0); n_kept_gate.append(0)
            sel_off.append(sel_off[-1])
            continue
        lab.append(True)
        cx, cy = np.asarray(ag[:, 0], float), np.asarray(ag[:, 1], float)
        n_raw.append(int(cx.shape[0]))
        g = np.nonzero(gate_mask(cx, cy))[0]
        n_gate.append(int(g.size))
        # what each population actually DELIVERS to the head after the pad truncation
        n_kept_raw.append(min(int(cx.shape[0]), pad))
        keep_g = g[nearest_n(cx[g], cy[g], pad)] if g.size > pad else g
        n_kept_gate.append(int(keep_g.size))
        sel_idx.append(keep_g.astype(np.int16))
        sel_off.append(sel_off[-1] + int(keep_g.size))
    a = {k: np.asarray(v) for k, v in
         (("n_raw", n_raw), ("n_gate", n_gate), ("n_kept_raw", n_kept_raw),
          ("n_kept_gate", n_kept_gate))}
    m = np.asarray(lab, dtype=bool)
    def _d(x, mask):
        x = x[mask]
        return {"n_windows": int(x.size), "total": int(x.sum()),
                "mean": round(float(x.mean()), 4) if x.size else None,
                "median": float(np.median(x)) if x.size else None,
                "p95": float(np.percentile(x, 95)) if x.size else None,
                "max": int(x.max()) if x.size else None,
                "frac_windows_empty": round(float((x == 0).mean()), 4) if x.size else None}
    rep = {
        "n_windows": int(m.size), "n_windows_labelled": int(m.sum()),
        "n_target_boxes_prefilter": int(a["n_raw"].sum()),
        "pad": int(pad), "queries": int(queries),
        "population_360": _d(a["n_raw"], m),
        "population_gate_relevant": _d(a["n_gate"], m),
        "delivered_360_after_pad": _d(a["n_kept_raw"], m),
        "delivered_gate_after_pad": _d(a["n_kept_gate"], m),
        # ⭐ the two budgets, counted rather than assumed
        "frac_labelled_windows_over_pad_360":
            round(float((a["n_raw"][m] > pad).mean()), 4) if m.any() else None,
        "frac_labelled_windows_over_queries_360":
            round(float((a["n_kept_raw"][m] > queries).mean()), 4) if m.any() else None,
        "frac_labelled_windows_over_queries_gate":
            round(float((a["n_kept_gate"][m] > queries).mean()), 4) if m.any() else None,
        "gate_share_of_boxes":
            round(float(a["n_gate"].sum() / max(a["n_raw"].sum(), 1)), 4),
        "wall_s": round(time.time() - t0, 1),
    }
    return {"report": rep, "labelled": m, "arrays": a,
            "sel_off": np.asarray(sel_off, dtype=np.int64),
            "sel_idx": (np.concatenate(sel_idx) if sel_idx
                        else np.zeros(0, dtype=np.int16))}


def grid_verdict(rep: dict, expect: dict) -> tuple[dict, list, str]:
    """⛔ The refusal, as a PURE function so it can be tested and broken on purpose.

    ``expect`` maps a report key to the literal a real run wrote. A key whose literal is ``None``
    is ``NOT_CHECKED`` and can never make the verdict OK-by-omission: a bank is admissible only
    when at least one literal was actually compared AND every compared literal matched.
    """
    checks, bad, n_checked = {}, [], 0
    for key, want in expect.items():
        got = rep.get(key)
        if want is None:
            checks[key] = "NOT_CHECKED"
            continue
        n_checked += 1
        checks[key] = ("MATCH %s" % got if got == want
                       else "MISMATCH got %s want %s" % (got, want))
        if got != want:
            bad.append(key)
    if bad:
        return checks, bad, "INCONCLUSIVE — grid does not match the run's own literals"
    if n_checked == 0:
        return checks, bad, "UNVERIFIED — no literal was compared; census only, do NOT bank"
    return checks, bad, "OK"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a refcv6 run's config.json (the grid's shape)")
    ap.add_argument("--v2-cache", required=True)
    ap.add_argument("--agent-join", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lru", type=int, default=2)
    ap.add_argument("--pad", type=int, default=32)
    ap.add_argument("--queries", type=int, default=100,
                    help="the SCORED head's query count (box3d = 100), never --agent-queries")
    ap.add_argument("--expect-windows", type=int, default=None)
    ap.add_argument("--expect-labelled", type=int, default=None)
    ap.add_argument("--expect-boxes", type=int, default=None)
    ap.add_argument("--bank", action="store_true", help="write the per-window selection npz")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")        # ⛔ never touch the card
    import types
    import refcv3_arm as ARM
    from train_p8_occupancy import JoinFileReader

    # ⭐ PHASE TIMERS. The premise behind pre-building is that the arm spends card time
    # INDEXING; these three numbers are what decides whether that premise is true, so they
    # are measured rather than assumed, and they are banked with the census.
    cfg_json = json.loads(Path(a.config).read_text(encoding="utf-8"))
    cfg, _targs, _src = ARM.rebuild_config(cfg_json)
    _t = time.time()
    ns = types.SimpleNamespace(episodes=a.v2_cache, lru=a.lru, episodes_n=0,
                               labels=a.labels, nav_source="none")
    eps, _f, clip_ids, ds, _x, _j, _y, _raw_off = ARM.build_corpus(ns, cfg, {})
    t_corpus = time.time() - _t
    window = int(cfg.core.window)
    _t = time.time()
    reader = JoinFileReader(a.agent_join,
                            episode_ids=[int(e.episode_id) for e in eps])
    t_join = time.time() - _t
    out = census(ds, eps, reader, window, pad=a.pad, queries=a.queries)
    out["report"]["phase_s"] = {"corpus_build": round(t_corpus, 1),
                                "join_load": round(t_join, 1),
                                "window_scan": out["report"]["wall_s"]}
    rep = out["report"]
    rep.update({"window": window, "v2_cache": Path(a.v2_cache).name,
                "join_sha12": sha12(Path(a.agent_join).name),
                "n_episodes": len(eps), "clips_sha12": [sha12(c) for c in clip_ids][:3],
                "config_src": Path(a.config).name,
                "reader_max_agents_per_frame": int(reader.max_agents_per_frame),
                "reader_n_records": int(reader.n_records)})

    # ---- the independent cross-check, BEFORE anything is banked -------------------------
    checks, bad, status = grid_verdict(rep, {"n_windows": a.expect_windows,
                                             "n_windows_labelled": a.expect_labelled,
                                             "n_target_boxes_prefilter": a.expect_boxes})
    rep["grid_cross_check"], rep["status"] = checks, status

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"p3_census_{Path(a.v2_cache).name}.json").write_text(
        json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("status", "n_windows", "n_windows_labelled",
                       "n_target_boxes_prefilter", "gate_share_of_boxes",
                       "grid_cross_check", "wall_s")}, indent=1))
    print("360°  per labelled window: %s" % rep["population_360"])
    print("gate  per labelled window: %s" % rep["population_gate_relevant"])
    if bad:
        print("ZZP3-CENSUS-INCONCLUSIVE %s ZZ" % bad, flush=True)
        return 3
    if a.bank and status != "OK":
        # ⛔ An UNVERIFIED grid is not a verified one. The census is still written (it is
        # readable as a census); the BANK is refused, because a target bank on an unproven
        # grid is worse than no bank — it looks exactly like a verified one.
        print("ZZP3-BANK-REFUSED %s ZZ" % status, flush=True)
        return 5
    if a.bank:
        np.savez_compressed(d / f"p3_targets_{Path(a.v2_cache).name}.npz",
                            sel_off=out["sel_off"], sel_idx=out["sel_idx"],
                            labelled=out["labelled"], n_raw=out["arrays"]["n_raw"],
                            n_gate=out["arrays"]["n_gate"],
                            window=np.int64(window), pad=np.int64(a.pad))
        print("ZZP3-BANKED %s ZZ" % (d / f"p3_targets_{Path(a.v2_cache).name}.npz"), flush=True)
    print("ZZP3-CENSUS-OK ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
