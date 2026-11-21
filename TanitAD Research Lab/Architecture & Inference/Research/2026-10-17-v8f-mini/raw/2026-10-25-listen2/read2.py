"""PREREG_V8F_LISTEN2 -- the reading. The three committed tables, APPLIED VERBATIM (readings M1 to M7 are fixed in ``listen2.py``'s docstring).

Estimator everywhere: paired episode-cluster bootstrap, B = 2,000, seed 0 (``taniteval/ci.py`` through the eval's own ``paired`` and macro-F1 reducer;
LISTEN's ``listen_read`` functions are REUSED, not re-implemented). Tier: T1-class open loop, DIAGNOSTIC. It answers "would another draw of EPISODES say
this?", not "would another TRAINING RUN say this?" (the two training seeds ride side by side) and not "another inference run" (the planner is deterministic).

E1  T-lat F1(clean) - F1(donor nav at S only), at BOTH seeds: > 0, separated, above the donor-seed floor (donor seeds 0 and 1 on s0)
    ROUTE FLOWS DOWN | PARALLEL PATH (E1 not separated at either seed) | SEEDS DISAGREE (anything else)
E2  per row the argmax of the mean-replacement effect equals the grid's largest-n input, at both seeds: ROBUST (4 rows) | PARTLY ROBUST (2-3) | NOT ROBUST (<2)
E3  ADE6(joint triple) - max over the three single swaps: < 0 and separated at both seeds => INCOHERENCE, else NOT SHOWN
"""
from __future__ import annotations

import numpy as np

import listen as LS
import listen2 as L2
import listen_read as LR
import v8f_mini_eval as EV

B_BOOT = LR.B_BOOT
ROWS = LR.ROWS                                                      # S, T-lat, T-lon, O
#: the grid's largest-n input per row, as the PREREG states it (re-derived from LISTEN's READ.json and ASSERTED equal in ``grid_argmax``)
PREREG_GRID_ARGMAX = {"S": "nav", "T-lat": "ck", "T-lon": "ego", "O": "ego"}


# ------------------------------------------------------------------------------------------------ E1 / E3 (donor runs)
def _row_clean(row, res, X):
    L = LR.LAYER_OF.get(row, "O")
    return LR.vec(row, res[f"clean/{L}"], X)


def e1_block(res: dict, X: dict, B: int = B_BOOT) -> dict:
    """E1 and its beside-reads for ONE (checkpoint, donor-seed) donor run: the whole-chain arm 'donor nav at S only' read in every row, plus LISTEN's
    reference cells (T/ck, T/nav, S/nav, O singles) recomputed in the same run."""
    out = {"rows": {}, "ref": {}}
    for row in ("S", "T-lat", "T-lon", "O", "O-ade2"):
        clean = _row_clean(row, res, X)
        out["rows"][row] = LR.cell_stats(row, LR.vec(row, res[L2.E1_NAME], X), clean, X, B)
        out["rows"][row]["clean_level"] = LR.level(row, clean, X)
    for nm, row in (("S/nav", "S"), ("T/ck", "T-lat"), ("T/ck", "T-lon"), ("T/nav", "T-lat"), ("T/nav", "T-lon"), ("O/ego", "O"), ("O/ck", "O"), ("O/code", "O")):
        out["ref"][f"{nm}@{row}"] = LR.cell_stats(row, LR.vec(row, res[nm], X), _row_clean(row, res, X), X, B)
    ck = out["ref"]["T/ck@T-lat"]["delta"]
    out["fraction_E1_over_Tlat_x_ck"] = float(out["rows"]["T-lat"]["delta"] / ck) if ck > 0 else None
    out["fraction_E1_over_Tlon_x_ck"] = float(out["rows"]["T-lon"]["delta"] / out["ref"]["T/ck@T-lon"]["delta"]) if out["ref"]["T/ck@T-lon"]["delta"] > 0 else None
    return out


def e3_block(res: dict, X: dict, B: int = B_BOOT) -> dict:
    """E3 for ONE donor run. M5: the comparator is the single swap with the largest MEAN ADE6 on the full set."""
    eid = np.asarray(X["eid"])
    clean = res["clean/O"]["ade6"]
    joint = res[L2.E3_NAME]["ade6"]
    singles = {x: res[f"O/{x}"]["ade6"] for x in L2.E3_SINGLES}
    means = {x: float(v.mean()) for x, v in singles.items()}
    comp = max(means, key=means.get)
    r = EV.paired(joint, singles[comp], eid, B)
    out = {"comparator": comp, "mean_ade6": {"clean": float(clean.mean()), "joint": float(joint.mean()), **means},
           "E3": {"delta": float(joint.mean() - singles[comp].mean()), "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]),
                  "n_windows": int(len(joint)), "n_episodes": int(len(np.unique(eid)))},
           "joint_vs_clean": LR.cell_stats("O", joint, clean, X, B), "single_vs_clean": {x: LR.cell_stats("O", v, clean, X, B) for x, v in singles.items()},
           "joint_minus_each_single": {}}
    for x, v in singles.items():
        rr = EV.paired(joint, v, eid, B)
        out["joint_minus_each_single"][x] = {"delta": float(joint.mean() - v.mean()), "lo": rr["lo"], "hi": rr["hi"], "separated": bool(rr["separated"])}
    pwmax = np.max(np.stack(list(singles.values())), 0)                                  # DESCRIPTIVE only (M5)
    rr = EV.paired(joint, pwmax, eid, B)
    out["joint_minus_perwindow_max_DESCRIPTIVE"] = {"delta": float(joint.mean() - pwmax.mean()), "lo": rr["lo"], "hi": rr["hi"], "separated": bool(rr["separated"])}
    return out


def donor_floor_e1(b0: dict, b1: dict) -> dict:
    """M6: |E1(s0, donor 0) - E1(s0, donor 1)| per row, from two donor runs on the SAME checkpoint."""
    return {row: float(abs(b0["rows"][row]["delta"] - b1["rows"][row]["delta"])) for row in b0["rows"]}


def e1_seed_flags(block: dict, floor: dict) -> dict:
    d = block["rows"]["T-lat"]
    return {"delta": d["delta"], "lo": d["lo"], "hi": d["hi"], "separated": d["separated"], "floor": floor["T-lat"],
            "positive": bool(d["delta"] > 0), "above_floor": bool(abs(d["delta"]) > floor["T-lat"]),
            "bar_met": bool(d["delta"] > 0 and d["separated"] and abs(d["delta"]) > floor["T-lat"])}


def e1_outcome(flags: dict) -> dict:
    """M7. ``flags[seed]`` from ``e1_seed_flags``."""
    if all(f["bar_met"] for f in flags.values()):
        head = "ROUTE FLOWS DOWN"
    elif not any(f["separated"] for f in flags.values()):
        head = "PARALLEL PATH"
    else:
        head = "SEEDS DISAGREE"
    return {"headline": head, "bar_met": {s: f["bar_met"] for s, f in flags.items()}, "separated": {s: f["separated"] for s, f in flags.items()}}


E1_MEANING = {
    "ROUTE FLOWS DOWN": "T's lateral decision follows the route S derived from nav: nav -> S -> T is a working chain",
    "PARALLEL PATH": "T-lateral reads S's checkpoints for something nav does not change; nav reaches T only directly. The hierarchy claim is narrowed to "
                     "\"S -> O\" and \"nav -> T\"",
    "SEEDS DISAGREE": "reported; seeds before windows",
}


def e3_outcome(blocks: dict) -> dict:
    ok = all(b["E3"]["delta"] < 0 and b["E3"]["separated"] for b in blocks.values())
    return {"headline": "INCOHERENCE" if ok else "NOT SHOWN",
            "meaning": ("O reacts to the CONTRADICTION between its inputs more than to a coherent foreign state" if ok else
                        "the joint triple is not shown to hurt less than the worst single swap at both seeds")}


# ------------------------------------------------------------------------------------------------ E2 (mean-replacement runs)
def grid_argmax(read_json: dict) -> dict:
    """The grid's largest-n input per row and seed, RE-DERIVED from LISTEN's READ.json with LISTEN's own ``argmax_col`` (R1), and asserted equal to the
    PREREG's statement."""
    out = {}
    for s in ("s0", "s1"):
        rec = read_json["grid"][s]
        out[s] = {row: LR.argmax_col(rec, row) for row in ROWS}
        assert out[s] == PREREG_GRID_ARGMAX, (s, out[s], PREREG_GRID_ARGMAX)
    return out


def e2_argmax(rec: dict, row: str, key: str = "delta", absolute: bool = False):
    """M3: the input with the largest signed effect (or, descriptively, the largest |effect|)."""
    best, arg = -np.inf, None
    for col in LR.COLS:
        c = rec["rows"][row]["cells"][col]
        if c is None:
            continue
        v = abs(c[key]) if absolute else c[key]
        if v > best:
            best, arg = v, col
    return arg


def e2_verdict(mean_recs: dict, grid_arg: dict) -> dict:
    """M4. ``mean_recs[seed]`` = ``LR.run_record`` of that seed's mean-replacement run."""
    per = {}
    for s, rec in mean_recs.items():
        per[s] = {}
        for row in ROWS:
            a = e2_argmax(rec, row)
            runner = sorted(((c["delta"], col) for col, c in rec["rows"][row]["cells"].items() if c is not None and col != "all"), reverse=True)
            top = rec["rows"][row]["cells"][a]
            second = next(((v, col) for v, col in runner if col != a), None)
            sec_cell = rec["rows"][row]["cells"][second[1]] if second else None
            per[s][row] = {"argmax": a, "grid_argmax": grid_arg[s][row], "agrees": a == grid_arg[s][row],
                           "argmax_by_abs_DESCRIPTIVE": e2_argmax(rec, row, absolute=True), "delta": top["delta"], "lo": top["lo"], "hi": top["hi"],
                           "separated": top["separated"], "runner_up": second[1] if second else None,
                           "runner_up_delta": sec_cell["delta"] if sec_cell else None,
                           "intervals_overlap": bool(sec_cell is not None and not (top["lo"] > sec_cell["hi"] or sec_cell["lo"] > top["hi"])),
                           "grid_cell_delta_under_mean": rec["rows"][row]["cells"][grid_arg[s][row]]["delta"]}
    rows_agree = {row: all(per[s][row]["agrees"] for s in per) for row in ROWS}
    k = sum(rows_agree.values())
    head = "ROBUST" if k == 4 else ("PARTLY ROBUST" if k >= 2 else "NOT ROBUST")
    return {"headline": head, "k_rows_agreeing_at_both_seeds": k, "rows_agree_at_both_seeds": rows_agree,
            "k_per_seed": {s: sum(per[s][row]["agrees"] for row in ROWS) for s in per}, "per_seed": per}


E2_MEANING = {
    "ROBUST": "the grid's ordering is not an artefact of donor swapping: mean replacement picks the same input in all four rows at both seeds",
    "PARTLY ROBUST": "the ordering survives in part; the rows that disagree are reported with their cells",
    "NOT ROBUST": "the grid's ordering is largely an artefact of donor swapping (cross-input conflict), and the grid's per-row ranking should not be quoted as a hierarchy statement",
}
