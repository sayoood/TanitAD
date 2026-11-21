"""read_pilot_s.py -- applies PREREG_L7_SCORER.AMENDMENT-1 sec. A3 (DRAFT until sealed) to the two PILOT-S ``train_log.json`` files, mechanically.

    python read_pilot_s.py <L7S_ts0>/train_log.json <L7S_ts1>/train_log.json

The committed reading lives here as code so that nobody re-derives it after seeing the numbers.  Constants are the amendment's literals (MEASURED in the E / F PILOT,
``pilot/*/train_log.json`` at c3471b9).  It reads the validation slice ONLY and refuses a log that is not the E / F slice (980 windows, own pick 0.99895 m, oracle 0.35874 m,
validation-row sha12 5af23a842e87).  It never reads eval139.
"""
from __future__ import annotations

import json
import math
import sys

OWN = 0.99895
ORACLE = 0.35874
E_PICK = {0: 1.5892, 1: 1.5921}
SPREAD_F = 0.0913            # |L7-F ts0 - ts1| = |1.0714 - 1.1627|: the largest training-seed spread on this rig
SPREAD_E = 0.0029            # |L7-E ts0 - ts1| = |1.5892 - 1.5921|
K7_MIN = 0.95
VAL_SHA12 = "5af23a842e87"


def verdict(s_pick, s_shuf, comf_auc, finite) -> dict:
    """``s_pick`` / ``s_shuf`` / ``comf_auc`` / ``finite`` are 2-lists indexed by training seed (0, 1).  First matching row of A3."""
    nu = abs(s_pick[0] - s_pick[1])
    m_own, m_e = max(nu, SPREAD_F), max(nu, SPREAD_E)
    g_own = [OWN - s_pick[s] for s in (0, 1)]
    g_e = [E_PICK[s] - s_pick[s] for s in (0, 1)]
    h = [s_shuf[s] - s_pick[s] for s in (0, 1)]
    detail = {"nu": round(nu, 5), "M_own": round(m_own, 5), "M_E": round(m_e, 5), "g_own": [round(x, 5) for x in g_own],
              "g_E": [round(x, 5) for x in g_e], "h": [round(x, 5) for x in h]}
    suspect = [s for s in (0, 1) if not (finite[s] and comf_auc[s] is not None and comf_auc[s] >= K7_MIN and math.isfinite(s_shuf[s]))]
    if suspect:
        return {"outcome": "WITHHELD", "suspect_seeds": suspect, **detail}
    if all(g_own[s] > m_own and h[s] > nu for s in (0, 1)):
        return {"outcome": "O1 BEATS-OWN", **detail}
    if all(g_e[s] > m_e and h[s] > nu for s in (0, 1)):
        return {"outcome": "O2 READS-THE-SCENE", **detail}
    return {"outcome": "O3 NO-SCENE-SIGNAL", **detail}


def best_row(log: dict) -> dict:
    """The epoch with the minimum validation LOSS (sec. 1 / the way RESULT_PILOT.md read E / F), plus the run's finiteness."""
    eps = log["epochs"]
    if not eps:
        raise SystemExit("[read_pilot_s] a train_log with no epochs")
    ep = min(eps, key=lambda e: e["val"]["loss"])
    v = ep["val"]
    sh = ep.get("val_scene_shuffled")
    if (log.get("n_val"), v.get("n"), v.get("own_pick_cost_m"), v.get("oracle_cost_m")) != (980, 980, OWN, ORACLE):
        raise SystemExit(f"[read_pilot_s] not the E / F validation slice: n_val {log.get('n_val')}, own {v.get('own_pick_cost_m')}, oracle {v.get('oracle_cost_m')}")
    if log.get("val_rows_sha12") not in (VAL_SHA12,):
        raise SystemExit(f"[read_pilot_s] validation-row sha12 {log.get('val_rows_sha12')} != {VAL_SHA12}")
    finite = all(math.isfinite(e["val"]["loss"]) and all(math.isfinite(x) for x in e["train_loss"].values()) for e in eps)
    return {"epoch": ep["epoch"], "pick": v["pick_cost_m"], "loss": v["loss"], "comf_auc": v["head_auc"].get("comf"),
            "shuf": None if sh is None else sh["pick_cost_m"], "finite": finite, "head_auc": v["head_auc"],
            "n_epochs": len(eps), "mode": log["args"]["mode"], "seed": log["args"]["seed"]}


def read(path0: str, path1: str) -> dict:
    rows = []
    for s, p in enumerate((path0, path1)):
        r = best_row(json.load(open(p, encoding="utf-8")))
        if r["mode"] != "S" or r["seed"] != s:
            raise SystemExit(f"[read_pilot_s] {p} is mode {r['mode']} seed {r['seed']}, expected S seed {s}")
        rows.append(r)
    if any(r["shuf"] is None for r in rows):
        raise SystemExit("[read_pilot_s] a log without val_scene_shuffled: control C-S1 is unreadable")
    v = verdict([r["pick"] for r in rows], [r["shuf"] for r in rows], [r["comf_auc"] for r in rows], [r["finite"] for r in rows])
    return {"verdict": v, "per_seed": rows, "own": OWN, "oracle": ORACLE, "E_pick": E_PICK}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(read(sys.argv[1], sys.argv[2]), indent=1))
