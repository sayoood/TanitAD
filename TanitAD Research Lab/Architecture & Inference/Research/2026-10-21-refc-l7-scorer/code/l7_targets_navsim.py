"""l7_targets_navsim.py -- per-candidate NAVSIM PDM targets for the L7 scorer on refc's fan (scorer9).

NO PDM IS RE-IMPLEMENTED HERE.  Two existing pieces are imported unchanged and only glued:
  * the plan refc's bridge SUBMITS for a candidate: ``taniteval.bench.navsim.bridge.knots_to_navsim`` -- the E2 C2
    not-a-knot cubic through (0,0) and refc's 8 knots (0.5 ... 6 s), evaluated at NAVSIM's 0.5 ... 4.0 s, heading = the
    spline tangent (``refcv8_bridge.py:18-20`` routes the emitted plan through exactly this function);
  * REFe's ``navsim_pdm_targets.score_plans(parts, plans8)`` (REFe repo ``refe/navsim_pdm_targets.py:91-132``): NAVSIM
    v1.1's own PDMSimulator (LQR tracker) + PDMScorer on [PDM-Closed, candidates], EP reproduced pairwise -- REFe's
    scorer targets, bit for bit.  ``parts`` come from REFe's P6/D9 cache (``build_metric_cache.py``:
    ``<root>/<log>/<token>.pkl.xz`` = lzma(pickle({"format": "tanitad.p6parts/1", "parts": ...}))), or are rebuilt from
    the nuPlan scenario with ``metric_cache_parts`` when no cache exists.

ENVIRONMENT: the NAVSIM v1.1 venv (navsim @ 3e8291b + nuplan-devkit), exactly REFe's labeller environment.  The dev box's
installed devkit is NAVSIM v2 (``WeightedMetricIndex`` has LANE_KEEPING / HISTORY_COMFORT, no COMFORTABLE), so this
script REFUSES to run there rather than score with the wrong rules (``--allow-v2`` does not exist on purpose).

INPUT  --fan-npz: the NAVSIM capture roll's dump: ``token [W]``, ``log_name [W]``, ``anchor_traj [W,N,8,2]`` (refc ego
       frame, refc slots), ``sel_idx [W]``, optional ``reach_keep / ceil_keep / r8_emit_keep``.
OUTPUT --out-npz: ``token, log_name, nc, dac, ep, ttc, c, ddc, pdms [W,N]`` (float32), ``ok [W]`` (parts found), and a
       json manifest with CPU-s per token and per candidate.

    python l7_targets_navsim.py --fan-npz F --refe-dir <repo>/.../2026-09-20-refe-plan/refe --parts-root <D9 cache> \
        --out-npz T.npz --out-json T.json [--limit 5] [--workers 16]

CONTROLS -- ⚠️ NOT YET IMPLEMENTED IN THIS FILE (UNVERIFIED; they need the capture's harness CSV and the human plans,
neither of which exists yet).  The capture chain must run them before any target is used:
  N_SEL   the candidate at ``sel_idx``, scored here, equals the harness's per-token sub-scores for the emitted R8_LEGAL plan
          on the same tokens: NC / DAC exact, EP / TTC / C / PDMS within 1e-6.
  N_HUMAN the human 8-pose plan scores DAC = 1 on every token (REFe's gate "human DAC = 1 on 1224/1224").
  NOTE: the harness pairs EP against PDM-Closed per proposal SET; ``score_plans`` reproduces it pairwise per candidate
  (navsim_pdm_targets.py:115-121), which is why N_SEL is expected to match exactly.
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np

PARTS_FORMAT = "tanitad.p6parts/1"
SUBS = ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")


def require_navsim_v11():
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex as W
    names = {m.name for m in W}
    if "COMFORTABLE" not in names or "DRIVING_DIRECTION" not in names:
        raise SystemExit(f"[l7ns] REFUSING: this navsim is not v1.1 (WeightedMetricIndex = {sorted(names)}); REFe's "
                         f"score_plans reads v1.1's COMFORTABLE / DRIVING_DIRECTION")


def load_parts(root: str, log: str, token: str):
    p = os.path.join(root, log, token + ".pkl.xz")
    if not os.path.exists(p):
        return None
    d = pickle.loads(lzma.decompress(open(p, "rb").read()))
    if d.get("format") != PARTS_FORMAT:
        raise SystemExit(f"[l7ns] {p}: parts format {d.get('format')!r} != {PARTS_FORMAT!r}")
    return tuple(d["parts"])


def to_plans8(knots: np.ndarray) -> np.ndarray:
    """[N,8,2] refc knots -> [N,8,3] NAVSIM poses through the bridge's own conversion."""
    from taniteval.bench.navsim.bridge import knots_to_navsim
    return np.stack([knots_to_navsim(np.asarray(k, np.float64)) for k in knots]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", required=True)
    ap.add_argument("--refe-dir", required=True)
    ap.add_argument("--parts-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    require_navsim_v11()
    sys.path.insert(0, a.refe_dir)
    import navsim_pdm_targets as PT  # noqa: E402  (REFe, unchanged)
    z = np.load(a.fan_npz, allow_pickle=True)
    tok = np.asarray(z["token"]).astype(str)
    log = np.asarray(z["log_name"]).astype(str)
    W = len(tok) if not a.limit else min(len(tok), a.limit)
    N = int(z["anchor_traj"].shape[1])
    out = {k: np.full((W, N), np.nan, np.float32) for k in SUBS}
    ok = np.zeros(W, bool)
    t_parts = t_conv = t_score = 0.0
    for i in range(W):
        t0 = time.process_time()
        parts = load_parts(a.parts_root, log[i], tok[i])
        t_parts += time.process_time() - t0
        if parts is None:
            continue
        t0 = time.process_time()
        plans = to_plans8(z["anchor_traj"][i])
        t_conv += time.process_time() - t0
        t0 = time.process_time()
        r = PT.score_plans(parts, plans)
        t_score += time.process_time() - t0
        for k in SUBS:
            out[k][i] = np.asarray(r[k], np.float32)
        ok[i] = True
    n = int(ok.sum())
    np.savez_compressed(a.out_npz, token=tok[:W], log_name=log[:W], ok=ok, **out)
    man = {"producer": "l7_targets_navsim.py (scorer9)", "fan_npz": a.fan_npz, "parts_root": a.parts_root,
           "n_tokens": int(W), "n_scored": n, "n_missing_parts": int(W - n), "candidates_per_token": N,
           "cpu_s": {"parts_load": round(t_parts, 3), "knots_to_navsim": round(t_conv, 3), "score_plans": round(t_score, 3)},
           "cpu_s_per_token": round((t_parts + t_conv + t_score) / max(n, 1), 4),
           "cpu_ms_per_candidate": round(1e3 * (t_conv + t_score) / max(n * N, 1), 3)}
    json.dump(man, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print("ZZL7NS " + json.dumps({k: man[k] for k in ("n_scored", "n_missing_parts", "cpu_s_per_token",
                                                      "cpu_ms_per_candidate")}))
    return 0 if n == W else 1


if __name__ == "__main__":
    raise SystemExit(main())
