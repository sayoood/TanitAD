"""PREREG_V8F_LISTEN -- the reading. Per-cell paired deltas, n ratios, the "listens" flags, the two outcome tables and Q-NAV, APPLIED VERBATIM.

Estimator everywhere: paired episode-cluster bootstrap, B = 2,000, seed 0 (``taniteval/ci.py`` through the eval's own ``paired`` and
macro-F1 reducer). Tier: T1-class open loop, DIAGNOSTIC (donor inputs are interventions, not driving). It answers "would another draw of
EPISODES say this?", never "would another TRAINING RUN say this?"; the two training seeds ride side by side for that.

READINGS OF THE PREREG TEXT THAT THE TEXT LEAVES OPEN (all fixed here, in code, BEFORE any LISTEN number was computed):
  R1  "largest n": argmax over the row's INPUT columns of the UNCLIPPED n = delta(L,X) / delta(L,all) (clipping to [0,1] is for colour only).
      n is defined only where delta(L,all) > 0 and separated; where it is not, n is NA and the row cannot support a hypothesis at that seed.
  R2  a hypothesis HOLDS at a seed iff the largest-n cell of EVERY named row lies in the named set AND that cell "listens" at that seed
      (delta > 0, separated, |delta| above the donor-seed difference). A cell LISTENS iff that holds at BOTH seeds (the PREREG definition).
  R3  control 4 (the donor-seed floor) exists only for s0 (donor seeds 0 and 1). It is applied to BOTH seeds as the cell's floor:
      |delta| must exceed |delta(s0, donor 0) - delta(s0, donor 1)| of the SAME cell.
  R4  headline precedence of the outcome table: SHOWN if P-UPPER and P-LOWER both hold at both seeds; else PARTIAL if exactly one holds at
      both seeds; else NOT SHOWN if neither holds at any seed; else SEEDS DISAGREE (a hypothesis holds at one seed only and none at both).
      A hypothesis that holds at exactly one seed is ALWAYS also reported as SEEDS DISAGREE for that hypothesis, whatever the headline.
  R5  Q-NAV: D = ADE6(nav at T only) - ADE6(nav at O only), paired. THROUGH T iff D > 0 and separated at both seeds; AROUND T iff D < 0 and
      separated at both seeds; else BOTH / NOT ESTABLISHED. "Nav at X only" injects the donor's nav token at X and lets the change flow
      downstream (S only: S -> T -> O; T only: T -> O; O only: the O x nav cell).
"""
from __future__ import annotations

import json

import numpy as np

import listen as LS
import v8f_mini_eval as EV
from taniteval import v8f_consistency as C
from taniteval.ci import paired_episode_cluster_bootstrap

B_BOOT = 2000
ROWS = ("S", "T-lat", "T-lon", "O")
LAYER_OF = {"S": "S", "T-lat": "T", "T-lon": "T", "O": "O"}
COLS = LS.INPUTS
COL_LABEL = {"scene": "scene", "ego": "ego", "nav": "nav", "limit": "limit", "ck": "S checkpoints", "code": "T code", "age": "tick ages"}
UNIT = {"S": "m", "T-lat": "F1", "T-lon": "F1", "O": "m"}
METRIC = {"S": "checkpoint position error at 50 m (donor - clean)", "T-lat": "lateral macro-F1 (clean - donor)",
          "T-lon": "longitudinal macro-F1 (clean - donor)", "O": "ADE6 (donor - clean)"}
HYP = {"P-UPPER": {"rows": ("S", "T-lat"), "set": ("scene", "nav")}, "P-LOWER": {"rows": ("O",), "set": ("ck", "code")}}
QNAV_ARMS = ("nav@S", "nav@T", "nav@O", "nav@STO")


# ------------------------------------------------------------------------------------------------ run files
def save_run(res: dict, path, meta: dict):
    flat = {f"{n}|{k}": v for n, r in res.items() for k, v in r.items()}
    flat["__meta__|json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **flat)


def load_run(path) -> tuple:
    z = np.load(path, allow_pickle=False)
    res, meta = {}, {}
    for k in z.files:
        name, key = k.split("|")
        if name == "__meta__":
            meta = json.loads(str(z[k]))
        else:
            res.setdefault(name, {})[key] = z[k]
    return res, meta


# ------------------------------------------------------------------------------------------------ per-window metric vectors
def _enc(pred, X, kind):
    n, col = (5, "lat_cls") if kind == "lat" else (7, "lon_cls")
    true = np.where(X["has"] & (X[col] >= 0), X[col], -1)
    return EV._f1_encoded(true, np.asarray(pred), n)


def f1_reducer(n: int):
    """The eval's own macro-F1 reducer (``v8f_mini_eval._f1_reducer``) behind an integer cast. The bootstrap hands a reducer float64 arrays, and
    ``C.macro_f1`` then indexes its confusion matrix with floats (IndexError on this numpy). The encoded values are exact integers, so the cast
    changes no number; it is the only departure from the eval's reducer (reported in LISTEN_RESULT.md)."""
    base = EV._f1_reducer(n)

    def red(v):
        return base(np.rint(np.asarray(v, np.float64)).astype(np.int64))
    return red


def vec(row: str, r: dict, X: dict) -> np.ndarray:
    """The per-window metric vector of one stored cell (S: position error at 50 m, NaN off the label mask; T: F1-encoded predictions; O: ADE6)."""
    if row == "S":
        return C.checkpoint_errors(r["ck_now"], X["rc"], X["rc_ok"])["pos_err_m"][:, 1]
    if row == "T-lat":
        return _enc(r["lat_pred"], X, "lat")
    if row == "T-lon":
        return _enc(r["lon_pred"], X, "lon")
    if row == "O":
        return r["ade6"]
    if row == "O-ade2":
        return r["ade2"]
    raise ValueError(row)


def cell_stats(row: str, cell: np.ndarray, clean: np.ndarray, X: dict, B: int = B_BOOT) -> dict:
    """Delta (sign convention of the PREREG: S, O = donor - clean; T = clean - donor) with its paired episode-cluster CI."""
    eid = np.asarray(X["eid"])
    if row in ("S", "O", "O-ade2"):
        ok = X["rc_ok"][:, 1] if row == "S" else np.ones(len(eid), bool)
        a, b, e = cell[ok], clean[ok], eid[ok]
        pt = float(np.mean(a) - np.mean(b))
        r = EV.paired(a, b, e, B)
        return {"delta": pt, "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "n_windows": int(ok.sum()),
                "n_episodes": int(len(np.unique(e)))}
    red = f1_reducer(5 if row == "T-lat" else 7)
    pt = float(red(clean) - red(cell))
    r = paired_episode_cluster_bootstrap(clean, cell, eid, n_boot=B, seed=0, reduce=red)
    return {"delta": pt, "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "n_windows": int((clean >= 0).sum()),
            "n_episodes": int(len(np.unique(eid[clean >= 0])))}


def level(row: str, clean: np.ndarray, X: dict) -> float:
    if row == "S":
        return float(np.mean(clean[X["rc_ok"][:, 1]]))
    if row in ("T-lat", "T-lon"):
        return float(f1_reducer(5 if row == "T-lat" else 7)(clean))
    return float(np.mean(clean))


def run_record(res: dict, X: dict, B: int = B_BOOT) -> dict:
    """Every cell of one (checkpoint, donor-seed) run: rows S / T-lat / T-lon / O (+ ADE2 beside), the row references, and Q-NAV."""
    rec = {"rows": {}, "beside": {}, "qnav": {}}
    for row in ROWS + ("O-ade2",):
        L = LAYER_OF.get(row, "O")
        clean = vec(row, res[f"clean/{L}"], X)
        ent = {"unit": UNIT.get(row, "m"), "clean_level": level(row, clean, X), "cells": {}}
        for col in COLS + ("all",):
            nm = f"{L}/{col}"
            ent["cells"][col] = cell_stats(row, vec(row, res[nm], X), clean, X, B) if nm in res else None
        (rec["beside"] if row == "O-ade2" else rec["rows"])[row] = ent
    cleano = vec("O", res["clean/O"], X)
    for arm in QNAV_ARMS:
        if arm in res:
            rec["qnav"][arm] = cell_stats("O", vec("O", res[arm], X), cleano, X, B)
    if "nav@T" in res and "nav@O" in res:
        a, b = vec("O", res["nav@T"], X), vec("O", res["nav@O"], X)
        r = EV.paired(a, b, np.asarray(X["eid"]), B)
        rec["qnav"]["D_T_minus_O"] = {"delta": float(np.mean(a) - np.mean(b)), "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]),
                                      "n_windows": int(len(a)), "n_episodes": int(len(np.unique(X["eid"])))}
    return rec


# ------------------------------------------------------------------------------------------------ n, flags, outcomes
def add_n(rec: dict):
    """n(L,X) = delta(L,X) / delta(L,all) (R1: NA unless delta(L,all) > 0 and separated); ``n_clipped`` in [0,1] is the colour."""
    for row, ent in rec["rows"].items():
        al = ent["cells"]["all"]
        ok = al is not None and al["delta"] > 0 and al["separated"]
        ent["all_defined"] = bool(ok)
        for col in COLS:
            c = ent["cells"][col]
            if c is None:
                continue
            c["n"] = float(c["delta"] / al["delta"]) if ok else None
            c["n_clipped"] = float(min(max(c["n"], 0.0), 1.0)) if ok else None


def donor_floor(d0: dict, d1: dict) -> dict:
    """Control 4 (R3): per cell |delta(s0, donor 0) - delta(s0, donor 1)|."""
    out = {}
    for row in ROWS:
        out[row] = {}
        for col in COLS + ("all",):
            a, b = d0["rows"][row]["cells"][col], d1["rows"][row]["cells"][col]
            out[row][col] = None if a is None else float(abs(a["delta"] - b["delta"]))
    return out


def seed_flags(rec: dict, floor: dict) -> dict:
    """Per seed: delta > 0, separated, and |delta| above the donor-seed floor."""
    fl = {}
    for row in ROWS:
        fl[row] = {}
        for col in COLS:
            c = rec["rows"][row]["cells"][col]
            fl[row][col] = None if c is None else bool(c["delta"] > 0 and c["separated"] and abs(c["delta"]) > floor[row][col])
    return fl


def argmax_col(rec: dict, row: str):
    best, arg = -np.inf, None
    for col in COLS:
        c = rec["rows"][row]["cells"][col]
        if c is not None and c.get("n") is not None and c["n"] > best:
            best, arg = c["n"], col
    return arg


def holds(rec: dict, flags: dict, hyp: str) -> dict:
    spec, per_row, ok = HYP[hyp], {}, True
    for row in spec["rows"]:
        col = argmax_col(rec, row)
        good = col is not None and col in spec["set"] and bool(flags[row][col])
        per_row[row] = {"argmax": col, "in_set": col in spec["set"] if col else False, "listens_at_seed": bool(flags[row][col]) if col else False,
                        "holds": good}
        ok &= good
    return {"holds": ok, "rows": per_row}


def outcome(per_seed_hold: dict) -> dict:
    """R4. ``per_seed_hold[hyp][seed] -> bool``."""
    both = {h: all(per_seed_hold[h].values()) for h in HYP}
    one = {h: (sum(per_seed_hold[h].values()) == 1) for h in HYP}
    if all(both.values()):
        head = "DIVISION OF LABOUR SHOWN"
    elif any(both.values()):
        head = "PARTIAL"
    elif not any(any(v.values()) for v in per_seed_hold.values()):
        head = "NOT SHOWN"
    else:
        head = "SEEDS DISAGREE"
    return {"headline": head, "holds_at_both_seeds": both, "holds_at_exactly_one_seed": one,
            "seeds_disagree_on": [h for h in HYP if one[h]]}


def qnav_outcome(per_seed: dict) -> dict:
    ds = {s: q["D_T_minus_O"] for s, q in per_seed.items()}
    through = all(d["delta"] > 0 and d["separated"] for d in ds.values())
    around = all(d["delta"] < 0 and d["separated"] for d in ds.values())
    return {"headline": "THROUGH T" if through else ("AROUND T" if around else "BOTH / NOT ESTABLISHED"), "D": ds}


NEXT_STEPS = {
    "DIVISION OF LABOUR SHOWN": "the grid becomes the report's primary evidence for 'different abstractions'; the attention video is kept as illustration only",
    "PARTIAL": "the report's claim is narrowed to the cells that hold, stated per cell. If O's largest cell is the scene, O re-derives the plan from the image instead of following the hierarchy: that is FLAT's question, and PREREG_V8F_FLAT decides it",
    "NOT SHOWN": "the report states that the levels' input dependence does not differ as the attention suggested, and the abstraction claim rests on the oracle and donor reads alone",
    "SEEDS DISAGREE": "reported as such; seeds come before windows (H-ESTIM-SEED-1)",
}
NEXT_QNAV = {
    "THROUGH T": "the answer to the PI's question is yes: the trajectory follows nav mainly through T's decision",
    "AROUND T": "O follows nav directly, and T's nav reaction is not what conditions the trajectory. Next: pre-register an arm with nav REMOVED from O's inputs. That is an architecture change, so it needs the PI's go",
    "BOTH / NOT ESTABLISHED": "reported with both cells and their intervals",
}


def analyse(recs: dict, X: dict | None = None) -> dict:
    """``recs``: {'s0': record (donor 0), 's1': record (donor 0), 's0_d1': record (s0, donor 1)} -> the full READ (no controls)."""
    for r in recs.values():
        add_n(r)
    floor = donor_floor(recs["s0"], recs["s0_d1"])
    flags = {s: seed_flags(recs[s], floor) for s in ("s0", "s1")}
    listens = {row: {col: (None if flags["s0"][row][col] is None else bool(flags["s0"][row][col] and flags["s1"][row][col])) for col in COLS}
               for row in ROWS}
    hold = {h: {s: holds(recs[s], flags[s], h) for s in ("s0", "s1")} for h in HYP}
    out = outcome({h: {s: hold[h][s]["holds"] for s in hold[h]} for h in HYP})
    qn = qnav_outcome({s: recs[s]["qnav"] for s in ("s0", "s1")})
    return {"floor_donor_seed": floor, "seed_flags": flags, "listens": listens, "hypotheses": hold, "outcome": out, "qnav": qn,
            "next_committed": NEXT_STEPS[out["headline"]], "qnav_next_committed": NEXT_QNAV[qn["headline"]]}
