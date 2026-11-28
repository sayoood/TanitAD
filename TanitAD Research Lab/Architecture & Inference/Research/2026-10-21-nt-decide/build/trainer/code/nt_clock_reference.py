#!/usr/bin/env python3
"""nt_clock_reference -- G-CLOCK's INDEPENDENT clock reference for the NavSim token-episodes (H-NT-DECIDE-1).

G-CLOCK (``launch_gate.job_clock``) checks, on every window, ``|trainer NOW - (g_ref + (t + w - 1 + n_stack - 1) *
dt_ref)| <= tol`` against a per-clip clock MEASURED by an instrument that is not the sidecar the trainer reads. The
PhysicalAI references are q4c fits of the 100 Hz egomotion log. For a NavSim token-episode the measurement is the
LOG's own 2 Hz timestamps: E1 places the 2 Hz frame k (k = -1 .. +12 around the token) at raw row ``9 + 5k``
(INTERFACE.md sec. 3), so the pairs ``(row 9 + 5k, 0.9 + (t_k - t_0))`` -- actual microsecond timestamps -- are
fitted by least squares to ``time = g + row * dt``. A perfectly nominal log gives ``(0.0, 0.1)`` exactly; nuPlan's
measured spacing (0.49933 - 0.50055 s) gives sub-millisecond residuals, which is the claim being checked.

Output: the q4c schema ``load_clock_reference`` reads: ``{"splits": {"nt_train": {"per_clip": [{"clip": sha12,
"grid_start_on_label_timeline_s", "dt_s", "fit_resid_max_ms", "n_knots"}]}}, "controls": {...}}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nt_common as C  # noqa: E402

K_RANGE = range(-1, 13)            # 2 Hz frames t0 - 0.5 s .. t0 + 6.0 s (raw rows 4 .. 69)
SPACING = (0.45, 0.55)             # INTERFACE.md sec. 3: a dropped frame ends the chain


def fit_token(ts_us: np.ndarray, i: int, t_raw: int) -> dict | None:
    t0 = int(ts_us[i])
    rows, times = [], []
    for k in K_RANGE:
        j = i + k
        if j < 0 or j >= len(ts_us):
            continue
        row = C.NOW_RAW_ROW + 5 * k
        if row >= int(t_raw):
            break
        # the chain from the token outwards must be unbroken (a gap ends it, as E1's writer)
        lo, hi = sorted((i, j))
        d = np.diff(ts_us[lo:hi + 1]) / 1e6
        if len(d) and not np.all((d >= SPACING[0]) & (d <= SPACING[1])):
            continue
        rows.append(row)
        times.append(C.episode_now_s() + (int(ts_us[j]) - t0) / 1e6)
    if len(rows) < 3:
        return None
    A = np.stack([np.ones(len(rows)), np.asarray(rows, np.float64)], 1)
    (g, dt), *_ = np.linalg.lstsq(A, np.asarray(times, np.float64), rcond=None)
    resid = np.asarray(times) - (g + dt * np.asarray(rows))
    return {"grid_start_on_label_timeline_s": float(g), "dt_s": float(dt),
            "fit_resid_max_ms": float(np.abs(resid).max() * 1e3), "n_knots": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--token-table", required=True, help="nt_labels.py token_table_nt.json")
    ap.add_argument("--e1-manifest", default=None, help="E1's MANIFEST (its T_raw per token wins over ours)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = json.load(open(a.token_table, encoding="utf-8"))["rows"]
    traw = {}
    if a.e1_manifest:
        traw = {str(r["token"]): int(r["T_raw"]) for r in
                (json.loads(l) for l in open(a.e1_manifest, encoding="utf-8") if l.strip())}
    by_log: dict = {}
    for r in rows:
        by_log.setdefault(r["log_name"], []).append(r)
    per, worst_g, worst_dt, worst_res, n_none = [], 0.0, 0.0, 0.0, 0
    for ln, rr in sorted(by_log.items()):
        fr = C.load_log(Path(a.logs_dir) / f"{ln}.pkl")
        ts = np.array([int(f["timestamp"]) for f in fr], dtype=np.int64)
        for r in rr:
            f = fit_token(ts, int(r["frame_idx"]), traw.get(r["token"], int(r["T_raw"])))
            if f is None:
                n_none += 1
                continue
            per.append({"clip": C.sha12(r["clip_id"]), "clip_id": r["clip_id"], **f})
            worst_g = max(worst_g, abs(f["grid_start_on_label_timeline_s"]))
            worst_dt = max(worst_dt, abs(f["dt_s"] - C.DT_S))
            worst_res = max(worst_res, f["fit_resid_max_ms"])
    out = {"schema": "q4c-style clock reference (nt)", "what": "H-NT-DECIDE-1 NavSim token-episode clocks MEASURED "
           "from the log's 2 Hz microsecond timestamps (an instrument independent of clip_clock_nt.jsonl)",
           "splits": {"nt_train": {"per_clip": per}},
           "controls": {"n_clips": len(per), "n_unfittable": n_none, "worst_abs_grid_start_s": worst_g,
                        "worst_abs_dt_minus_0p1_s": worst_dt, "worst_fit_resid_ms": worst_res,
                        "legacy_clock_offset_s": round(C.episode_now_s() - 0.7, 9),
                        "note": "the legacy (t + w - 1) * 0.1 clock reads NOW 0.7 s vs 0.9 s: G-CLOCK's positive "
                                "control must flag it on every window"},
           "tool": {"nt_clock_reference.py": C.md5_file(__file__)}}
    C.json_dump(out, a.out)
    print(json.dumps(out["controls"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
