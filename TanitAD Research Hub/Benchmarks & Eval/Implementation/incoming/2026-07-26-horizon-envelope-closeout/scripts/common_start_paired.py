#!/usr/bin/env python3
"""common_start_paired.py — the K-vs-K contrast on IDENTICAL windows, offline.

The sweep was stopped after K=70 (T3 is a deliverable and pod2 runs one job at a
time), so the driver's own end-of-run paired block never executed. It does not
need a GPU: every per-window tensor is on disk, and the paired episode-cluster
bootstrap is arithmetic. This reproduces that block from
``artifacts/perwindow_K*.pt``.

DESIGN. ``starts = range(0, T - W - K, stride)`` shrinks with K, so the start set
at the LARGEST K is a subset of every smaller K's. Intersecting on
``(episode_index, t0)`` therefore yields the K=185 start set — the same
common-start design ``e1a_horizon`` and the committed gate driver use. Deltas are
``paired_episode_cluster_bootstrap`` (taniteval/ci.py, B=2000), oriented
``CDR(K) - CDR(K_ref)``; POSITIVE = the longer horizon departs more.

⚠️ STATED, NOT HIDDEN: this is paired on window IDENTITY, not on trajectory. Each
K is its own rollout and ``corridor_rollout``'s nearest-reference search runs over
``K+1`` future poses, so the same window is fed a different frame at different K
(MEASURED: max |dlat| difference 0.149 m, max |dyaw| 9.24 deg over the first 20
steps of K=185 vs K=20 on the shared windows). The pairing removes the
window-COMPOSITION confound; it does not make the two rollouts nested.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPO = HERE.parents[5]   # …/<repo>/TanitAD Research Hub/…/scripts
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval import ci as _ci          # noqa: E402
from taniteval import corridor as _corr  # noqa: E402
from taniteval import ood as _ood        # noqa: E402

PRIMARY = _corr.CORRIDOR_HALFWIDTH_M
B = 2000


def main():
    dumps = {}
    for p in sorted(ART.glob("perwindow_K*.pt")):
        K = int(p.stem.split("K")[-1])
        dumps[K] = torch.load(str(p), weights_only=False)
    Ks = sorted(dumps)
    key = {K: [(int(a), int(b)) for a, b in zip(d["epi"], d["t0"])]
           for K, d in dumps.items()}
    common = sorted(set.intersection(*[set(v) for v in key.values()]))
    sel = {K: np.array([{c: i for i, c in enumerate(key[K])}[c] for c in common])
           for K in Ks}
    ref = min(Ks)
    eid = [str(dumps[ref]["eid"][i]) for i in sel[ref]]
    hd = dumps[ref]["hd2s"].numpy()[sel[ref]]
    spd = dumps[ref]["speed"].numpy()[sel[ref]]
    strata = _corr.strata(hd, spd, _corr.JUNCTION_DEG)

    out = {"_experiment": "common-start PAIRED K contrast, closed loop",
           "_evidence_class": "MEASURED (ours; artifact = this JSON)",
           "_estimator": ("paired_episode_cluster_bootstrap (taniteval/ci.py) "
                          f"B={B}, unit = val EPISODE. overlapping_holdout_se "
                          "used NOWHERE."),
           "_caveat": ("paired on window IDENTITY, not on trajectory — each K is "
                       "its own rollout and the nearest-reference search range "
                       "is K+1, so the same window is fed a different frame at "
                       "different K. Removes the window-composition confound, "
                       "not the reference-range one."),
           "reference_K": ref,
           "n_common_windows": len(common),
           "n_common_episodes": len(set(eid)),
           "n_by_stratum": {k: int(v.sum()) for k, v in strata.items()},
           "horizons": Ks, "per_K": {}, "paired_delta_vs_K%d" % ref: {}}

    for K in Ks:
        d, s = dumps[K], sel[K]
        lat = d["lat"].numpy()[s]
        yaw = d["yaw"].numpy()[s]
        node = {}
        for name, m in strata.items():
            ix = np.flatnonzero(m)
            if len(ix) < 2 or len({eid[i] for i in ix}) < 2:
                node[name] = None
                continue
            blk = _corr.corridor_block(lat[ix], [eid[i] for i in ix],
                                       yaw_abs_deg=yaw[ix], n_boot=B,
                                       surface="closed_loop")
            fr = _ood.envelope_fractions(lat[ix], yaw[ix])
            node[name] = {
                "n_windows": blk["n_windows"], "n_episodes": blk["n_episodes"],
                "corridor_departure_rate": blk["corridor_departure_rate"],
                "peak_xte_m": blk["peak_xte_m"],
                "frac_steps_any_out_of_envelope": fr["frac_steps_any"],
                "frac_windows_any_step_out_of_envelope":
                    fr["frac_windows_any_step_out_of_envelope"],
                "EXTRAPOLATION_VERDICT": _ood._verdict_string(
                    False, fr["frac_windows_any_step_out_of_envelope"],
                    fr["frac_steps_any"])}
        out["per_K"][str(K)] = node

    base = dumps[ref]["lat"].numpy()[sel[ref]]
    for K in Ks:
        if K == ref:
            continue
        lat = dumps[K]["lat"].numpy()[sel[K]]
        row = {}
        for name, m in strata.items():
            ix = np.flatnonzero(m)
            if len(ix) < 2 or len({eid[i] for i in ix}) < 2:
                row[name] = None
                continue
            dd = _ci.paired_episode_cluster_bootstrap(
                _corr.corridor_departure(lat[ix], PRIMARY),
                _corr.corridor_departure(base[ix], PRIMARY),
                [eid[i] for i in ix], n_boot=B)
            dd["_orientation"] = (f"CDR(K={K}) - CDR(K={ref}) on the SAME "
                                  f"windows; POSITIVE = longer horizon departs more")
            row[name] = dd
        out["paired_delta_vs_K%d" % ref][str(K)] = row

    (ART / "common_start_paired.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    lines = []
    lines.append(f"common-start: {len(common)} windows / {len(set(eid))} episodes"
                 f" | strata {out['n_by_stratum']} | ref K={ref}")
    lines.append(f"{'K':>4} | " + " | ".join(f"{n:>34}" for n in strata))
    for K in Ks:
        cells = []
        for n in strata:
            x = out["per_K"][str(K)][n]
            cells.append("n/a".rjust(34) if not x else
                         (f"CDR {x['corridor_departure_rate']['mean']:.4f} "
                          f"winOUT {x['frac_windows_any_step_out_of_envelope']:.4f} "
                          f"n={x['n_windows']}/{x['n_episodes']}").rjust(34))
        lines.append(f"{K:>4} | " + " | ".join(cells))
    lines.append("")
    lines.append("PAIRED deltas vs K=%d:" % ref)
    for K in Ks:
        if K == ref:
            continue
        r = out["paired_delta_vs_K%d" % ref][str(K)]
        for n in strata:
            x = r[n]
            if not x:
                lines.append(f"  K={K:>3} {n:>13}: n too small")
                continue
            lines.append(f"  K={K:>3} {n:>13}: {x['delta']:+.4f} "
                         f"[{x['lo']:+.4f},{x['hi']:+.4f}] "
                         f"{'SEPARATED' if x.get('separated') else 'not separated'}"
                         f"  p={x.get('p_delta_gt0')}")
    txt = "\n".join(lines)
    (ART / "common_start_paired.txt").write_text(txt, encoding="utf-8")
    sys.stdout.buffer.write((txt + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
