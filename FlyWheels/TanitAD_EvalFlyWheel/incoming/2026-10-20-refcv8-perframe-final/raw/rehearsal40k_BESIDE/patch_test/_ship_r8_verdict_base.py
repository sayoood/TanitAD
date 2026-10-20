"""ship_r8_verdict.py -- SHIP-R8 clauses (1)-(7), applied MECHANICALLY on RETIME_DEPLOY8 (D-R8-SHIP-RETIME + AMENDMENT 1 + AMENDMENT 2, Project Steering/GOALS_AND_CLAIMS.md), CPU only.

NEW module; pure json in, json + md out; imports no stack code (the verdict must not share code with the instruments it judges).

    python ship_r8_verdict.py --tree step48852ema_unc     # the condition's tree
    python ship_r8_verdict.py --tree step30000ema_unc     # ORIENTATION ONLY (the condition's tree is step48852ema_unc)
    -> raw/ship_r8/SHIP_R8_<tree>.json + SHIP_R8_<tree>.md

SCORED ARM (AMENDMENT 2): `RETIME_DEPLOY8` = ONE re-timing call per window on all 8 model slots (0.5 ... 6 s) with Amendment 1's table, prior ha0_ext_pose at the same 8 slots; each surface is read
at its own instants.  The per-surface form `RETIME_DEPLOY` (Amendment 1's), `RETIME_TRAINFIT` and the flat 0.5 arm are printed BESIDE, never scored.

Inputs (each missing / unusable input makes the clauses that need it INCOMPLETE, never PASS):
  raw/retime_trainfit/FORWARD_TRAINFIT_<tree>.json    clause (1) (the register's wording) + the beside arm
  raw/retime_deploy/FORWARD_DEPLOY8_<tree>.json       clauses (1)-(4), arm RETIME_DEPLOY8 (sibling build: deploy8_forward_r8.py)
  raw/retime_deploy/TAC_<tree>.json                   clause (5), its RETIME_DEPLOY8 rows (tac_retime_r8.py)
  raw/retime_deploy/PRIOR_PARITY.json                 clause (7) (sibling build: prior_parity_r8.py)
  raw/retime_deploy/FORWARD_DEPLOY_<tree>.json        the per-surface form, BESIDE only
  C:/Users/Admin/refcv8_eval_archive/battery/<tree>/PAIRED_READ_<tree>.json   clause (6) (strategic block; informational: the condition pins it not applicable)

OVERALL: FAIL if any clause FAILS (the register: "Any failed clause = the stage does NOT ship"); else INCOMPLETE if any clause is INCOMPLETE; else PASS.  Clause (6) is N/A by the
condition's own text and counts as satisfied when the battery's strategic block agrees (status NOT APPLICABLE, n 0); if it disagrees it is INCOMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TF_DIR = PKG / "raw" / "retime_trainfit"
DEP_DIR = PKG / "raw" / "retime_deploy"
OUT_DIR = PKG / "raw" / "ship_r8"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
CONDITION_TREE = "step48852ema_unc"
FROZEN_SHA256 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
SEEDS = ("0", "1")
ARM = "RETIME_DEPLOY8"
PS_ARM = "RETIME_DEPLOY"            # the per-surface form of Amendment 1: BESIDE
PARITY_TOL_M = 1e-3
PARITY_MIN_WINDOWS = 200
N_SLOTS = 8

#: the register's own text (GOALS_AND_CLAIMS.md: D-R8-SHIP-RETIME, SHIP-R8 AMENDMENT 1, AMENDMENT 2), verbatim
CONDITION = ("SHIP-R8 holds iff, on tree `step48852ema_unc` (EMA weights), ceiling-compliant windows, arm `RETIME_TRAINFIT` with the frozen gains UNCHANGED (sha256 b208c9c1e4236075…), "
             "both inference seeds 0 and 1, paired episode-cluster bootstrap B 2000: ... Any failed clause = the stage does NOT ship and refcv8 is reported failing that bar.")
AMENDMENT = ("AMENDMENT 1: SHIP-R8 clauses (1)–(6) are evaluated on `RETIME_DEPLOY`; `RETIME_TRAINFIT` (per-surface) is printed beside. If `RETIME_DEPLOY` cannot be computed for 48,852, SHIP-R8 reads INCOMPLETE.  "
             "AMENDMENT 2: `RETIME_DEPLOY` = ONE call per window on all 8 model slots with Amendment 1's table, prior = `ha0_ext_pose` at the same 8 slots; each surface is scored on that call's outputs at its own "
             "instants; the per-surface form is printed beside.  New clause (7) PRIOR PARITY.")
CLAUSE_TEXT = {
    1: "(1) FORWARD_TRAINFIT controls ALL_PASS",
    2: "(2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds",
    3: "(3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds",
    4: "(4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT)",
    5: ("(5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds "
        "(instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS)"),
    6: "(6) STRATEGIC not applicable, n = 0, `--no-strategic`",
    7: ("(7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via "
        "`kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE"),
}
TAC_KEYS = ("TAC_traj_lat_correct", "TAC_traj_lon_correct")


def _clause(i: int, status: str, reason: str, evidence: dict | None = None) -> dict:
    return {"id": i, "text": CLAUSE_TEXT[i], "status": status, "reason": reason, "evidence": evidence or {}}


def _sep_below_zero(c) -> bool | None:
    """ERROR-style cell arm - reference: 'separated below 0' <=> delta < 0 AND the CI upper bound < 0.  None if the cell carries no interval."""
    if not isinstance(c, dict) or c.get("delta") is None or c.get("hi") is None:
        return None
    return bool(c["delta"] < 0 and c["hi"] < 0)


def _arm_rows(fd: dict | None, seed: str, surf: str, arm: str = ARM) -> dict | None:
    try:
        return fd["seeds"][seed]["surfaces"][surf]["sets"]["compliant"]["arms"][arm]
    except (KeyError, TypeError):
        return None


def _boot_ok(f: dict) -> bool | None:
    """B 2000: the explicit `n_boot` field when the file has one, else the estimator text's 'B 2000'; None when the file says neither."""
    if f.get("n_boot") is not None:
        return bool(f["n_boot"] == 2000)
    est = f.get("estimator")
    return None if not isinstance(est, str) else bool("B 2000" in est)


def _recorded_gain_hashes(f: dict) -> list:
    c = f.get("controls") or {}
    cand = [f.get("frozen_gains_sha256"), (c.get("pins") or {}).get("train_sha256"), f.get("source_sha256"), f.get("train_sha256"), f.get("deploy_gains_sha256")]
    return [x for x in cand if isinstance(x, str)]


# ------------------------------------------------------------------------------------------ clauses (pure)
def clause1(fd: dict | None, tf: dict | None, deploy_sha: str | None = None) -> dict:
    ev = {"forward_trainfit_present": tf is not None, "forward_deploy8_present": fd is not None}
    if tf is None:
        return _clause(1, "INCOMPLETE", "FORWARD_TRAINFIT_<tree>.json does not exist", ev)
    if fd is None:
        return _clause(1, "INCOMPLETE", "FORWARD_DEPLOY8_<tree>.json does not exist: RETIME_DEPLOY8 cannot be computed (AMENDMENT 1/2: INCOMPLETE)", ev)
    allowed = {FROZEN_SHA256} | ({deploy_sha} if deploy_sha else set())
    rec = _recorded_gain_hashes(fd)
    ev.update({"trainfit_controls_ALL_PASS": (tf.get("controls") or {}).get("ALL_PASS"), "deploy8_controls_ALL_PASS": (fd.get("controls") or {}).get("ALL_PASS"),
               "trainfit_frozen_gains_sha256": tf.get("frozen_gains_sha256"), "deploy8_recorded_gain_hashes": rec, "deploy8_n_boot_ok": _boot_ok(fd), "trainfit_n_boot_ok": _boot_ok(tf),
               "trainfit_seeds": sorted((tf.get("seeds") or {}).keys()), "deploy8_seeds": sorted((fd.get("seeds") or {}).keys())})
    bad, unk = [], []
    if ev["trainfit_controls_ALL_PASS"] is not True:
        bad.append("FORWARD_TRAINFIT controls.ALL_PASS is not true")
    if ev["deploy8_controls_ALL_PASS"] is not True:
        bad.append("FORWARD_DEPLOY8 controls.ALL_PASS is not true")
    if ev["trainfit_frozen_gains_sha256"] != FROZEN_SHA256:
        bad.append(f"the frozen gains sha256 in FORWARD_TRAINFIT is {ev['trainfit_frozen_gains_sha256']}, not the pinned {FROZEN_SHA256}")
    if not rec:
        unk.append("FORWARD_DEPLOY8 records no gains hash")
    elif not any(h in allowed for h in rec):
        bad.append(f"FORWARD_DEPLOY8 records gain hashes {rec}, none is the pinned frozen-train hash (or the deployed table's)")
    if ev["trainfit_seeds"] != list(SEEDS) or ev["deploy8_seeds"] != list(SEEDS):
        bad.append("both inference seeds 0 and 1 are not present in both files")
    for nm in ("deploy8_n_boot_ok", "trainfit_n_boot_ok"):
        if ev[nm] is False:
            bad.append(f"{nm[:-9]} bootstrap B is not 2000")
        elif ev[nm] is None:
            unk.append(f"{nm[:-9]} file states no bootstrap B")
    if bad:
        return _clause(1, "FAIL", "; ".join(bad), ev)
    if unk:
        return _clause(1, "INCOMPLETE", "; ".join(unk), ev)
    return _clause(1, "PASS", "FORWARD_TRAINFIT and FORWARD_DEPLOY8 controls ALL_PASS; the frozen gains hash is the pinned one; seeds 0 and 1; B 2000", ev)


def clause_e(i: int, surf: str, fd: dict | None, key: str = "minus_ha0_ext_pose", arm: str = ARM) -> dict:
    name = "E1" if surf == "S2" else "E3"
    if fd is None:
        return _clause(i, "INCOMPLETE", "FORWARD_DEPLOY8_<tree>.json does not exist")
    cells, flags = {}, {}
    for s in SEEDS:
        a = _arm_rows(fd, s, surf, arm)
        c = None if a is None else a.get(key)
        cells[f"s{s}"] = c
        flags[f"s{s}"] = _sep_below_zero(c)
    ev = {"surface": surf, "reference": "ha0_ext_pose", "arm": arm, "cells": cells, "separated_below_0": flags}
    if any(v is None for v in flags.values()):
        return _clause(i, "INCOMPLETE", f"{name}: a ({', '.join(s for s, v in flags.items() if v is None)}) cell is absent or carries no interval", ev)
    if all(flags.values()):
        return _clause(i, "PASS", f"{name}: {arm} - ha0_ext_pose ADE on {surf} is separated below 0 at both inference seeds", ev)
    return _clause(i, "FAIL", f"{name}: not separated below 0 at seed(s) {', '.join(s for s, v in flags.items() if not v)}", ev)


def clause4(fd: dict | None) -> dict:
    if fd is None:
        return _clause(4, "INCOMPLETE", "FORWARD_DEPLOY8_<tree>.json does not exist")
    b = (fd.get("bars") or {}).get(f"compliant|{ARM}")
    if not b:
        return _clause(4, "INCOMPLETE", f"no bars['compliant|{ARM}'] block in FORWARD_DEPLOY8")
    fams = (b.get("FZR-NR") or {}).get("per_family") or {}
    lat = b.get("FZR-LAT") or {}
    ev = {"FZR-NR_per_family": fams, "FZR-LAT_HOLDS": lat.get("HOLDS"), "FZR-LAT_separated_worse_at_any_seed": lat.get("separated_worse_at_any_seed")}
    need = ("ADE", "LONGITUDINAL", "LATERAL")
    missing = [f for f in need if f not in fams]
    fail = [f"{f}: {fams[f]}" for f in need if f in fams and str(fams[f]).startswith("FAIL")]
    unev = [f for f in need if f in fams and not str(fams[f]).startswith("FAIL") and fams[f] != "PASS"]
    if lat.get("HOLDS") is False:
        fail.append(f"FZR-LAT does not hold (separated worse: {lat.get('separated_worse_at_any_seed')})")
    if fail:
        return _clause(4, "FAIL", "; ".join(fail), ev)
    if missing or unev or lat.get("HOLDS") is None:
        return _clause(4, "INCOMPLETE", f"missing families {missing}, not evaluable {unev}, FZR-LAT HOLDS {lat.get('HOLDS')}", ev)
    return _clause(4, "PASS", "FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS", ev)


def clause5(tac: dict | None) -> dict:
    """Re-derives the verdict from the stored cells with its OWN rule (ACCURACY rows, arm - refcv7: worse AND separated <=> CI upper < 0); the TAC file's own status is cross-checked, never trusted alone.
    Only the file's RETIME_DEPLOY8 block counts: a file scored on any other arm (e.g. an older per-surface TAC file) is INCOMPLETE."""
    if tac is None:
        return _clause(5, "INCOMPLETE", "TAC_<tree>.json does not exist: the tactical instrument has not run (INCOMPLETE, not PASS)")
    if (tac.get("controls") or {}).get("ALL_PASS") is not True:
        return _clause(5, "INCOMPLETE", "the TAC instrument's controls are not ALL_PASS: its numbers are not admissible", {"controls_ALL_PASS": (tac.get("controls") or {}).get("ALL_PASS")})
    c5 = tac.get("clause5") or {}
    if c5.get("arm") != ARM:
        return _clause(5, "INCOMPLETE", f"the TAC file's clause-5 arm is {c5.get('arm')!r}, not {ARM!r} (AMENDMENT 2): re-run tac_retime_r8.py", {"tac_file_arm": c5.get("arm")})
    cells = c5.get("cells") or {}
    need = [f"{s}|{k}|s{sd}" for s in ("S2", "S6") for k in TAC_KEYS for sd in (0, 1)]
    miss = [n for n in need if n not in cells or not isinstance(cells[n].get("cell"), dict) or cells[n]["cell"].get("hi") is None]
    worse = [n for n in need if n not in miss and cells[n]["cell"]["hi"] < 0]
    ev = {"n_cells": len(need), "separated_worse_cells": worse, "missing": miss, "tac_file_status": c5.get("status"), "deploy8_load_status": c5.get("deploy8_load_status"),
          "cells": {n: cells[n]["cell"] for n in need if n in cells}}
    if worse:
        return _clause(5, "FAIL", "separated worse than refcv7-50,400: " + ", ".join(worse), ev)
    if miss:
        return _clause(5, "INCOMPLETE", "cells absent or without an interval: " + ", ".join(miss), ev)
    if ev["tac_file_status"] != "PASS":
        return _clause(5, "INCOMPLETE", f"the cells say PASS but the TAC file's own status is {ev['tac_file_status']!r}: the two disagree, read before quoting", ev)
    return _clause(5, "PASS", "no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of RETIME_DEPLOY8 is separated worse than refcv7-50,400", ev)


def clause6(pr: dict | None) -> dict:
    if pr is None:
        return _clause(6, "N/A", "UNVERIFIED against a battery file (PAIRED_READ not found); the condition itself pins clause (6) as not applicable, n = 0", {"paired_read": None})
    st = pr.get("strategic") or {}
    ev = {"strategic": {"status": st.get("status"), "n": st.get("n")}}
    if st.get("status") == "NOT APPLICABLE" and st.get("n") == 0:
        return _clause(6, "N/A", "the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states", ev)
    return _clause(6, "INCOMPLETE", f"the battery's strategic block reads {st.get('status')!r}, n {st.get('n')!r}: the condition assumed not applicable, n = 0", ev)


def _find_numbers(obj, name_pred, want_len=None, path=""):
    """First (path, value) in a nested json where the KEY satisfies name_pred and the value is a list of `want_len` numbers (or, if want_len is None, a number)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            ok = name_pred(str(k).lower())
            if ok and ((want_len is None and isinstance(v, (int, float)) and not isinstance(v, bool)) or
                       (want_len is not None and isinstance(v, list) and len(v) == want_len and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))):
                return f"{path}/{k}", v
        for k, v in obj.items():
            r = _find_numbers(v, name_pred, want_len, f"{path}/{k}")
            if r:
                return r
    return None


def _panels_from_sibling_schema(pp: dict):
    """prior_parity_r8.py's schema: `parity` = list of {tree, inference_seed, n_windows, model_path_vs_rebuilt: {per_slot_max_m [8], ...}}.  Returns [(label, n, per_slot_max)] or None."""
    par = pp.get("parity")
    if not isinstance(par, list) or not par:
        return None
    out = []
    for q in par:
        try:
            ps = q["model_path_vs_rebuilt"]["per_slot_max_m"]
            n = q["n_windows"]
            label = f"{q.get('tree')}|s{q.get('inference_seed')}"
        except (KeyError, TypeError):
            return None
        if not (isinstance(ps, list) and len(ps) == N_SLOTS and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in ps) and isinstance(n, int)):
            return None
        out.append((label, n, [float(x) for x in ps]))
    return out


def clause7(pp: dict | None) -> dict:
    """PRIOR PARITY from the sibling's PRIOR_PARITY.json.  The verdict RE-DERIVES the rule from the NUMBERS (per-slot max |delta| over every panel, the smallest panel's window count), never from the
    file's own reading; a file it cannot parse is INCOMPLETE; failed instrument controls (known values / source trace) make the numbers inadmissible -> INCOMPLETE."""
    if pp is None:
        return _clause(7, "INCOMPLETE", "PRIOR_PARITY.json does not exist: prior parity cannot be shown (INCOMPLETE, not PASS)")
    panels = _panels_from_sibling_schema(pp)
    if panels is None:                                    # generic fallback: one per-slot list of 8 numbers + one window count anywhere in the file
        slot = _find_numbers(pp, lambda k: "max" in k and ("slot" in k or "abs" in k or "diff" in k or "delta" in k), N_SLOTS)
        nwin = _find_numbers(pp, lambda k: k in ("n_windows", "n_win", "windows", "n"), None)
        panels = None if (slot is None or nwin is None) else [("file", int(nwin[1]), [float(x) for x in slot[1]])]
    ev = {"panels": None if panels is None else [{"panel": a, "n_windows": n, "per_slot_max_m": ps} for a, n, ps in panels],
          "file_reading": (pp.get("clause7") or {}).get("READING") or pp.get("verdict") or pp.get("PASS"),
          "known_values_ok": (pp.get("clause7") or {}).get("known_values_ok"), "source_trace_all_verified": pp.get("source_trace_all_verified")}
    if panels is None:
        return _clause(7, "INCOMPLETE", f"PRIOR_PARITY.json schema not recognised (need panels with a per-slot max |delta| list of {N_SLOTS} numbers and a window count): read the file by hand", ev)
    if ev["known_values_ok"] is False or ev["source_trace_all_verified"] is False:
        return _clause(7, "INCOMPLETE", "the parity instrument's own controls did not pass (known values / source trace): its numbers are not admissible", ev)
    nmin = min(n for _a, n, _p in panels)
    slot_max = [max(ps[i] for _a, _n, ps in panels) for i in range(N_SLOTS)]
    ev.update({"min_windows_per_panel": nmin, "slot_max_abs_m": {"values": slot_max}})
    if nmin < PARITY_MIN_WINDOWS:
        return _clause(7, "INCOMPLETE", f"parity shown on {nmin} windows in the smallest panel < {PARITY_MIN_WINDOWS}", ev)
    if any(x != x for x in slot_max):
        return _clause(7, "INCOMPLETE", "a per-slot value is NaN", ev)
    over = [i for i, x in enumerate(slot_max) if abs(x) > PARITY_TOL_M]
    if over:
        return _clause(7, "FAIL", f"max |delta| exceeds {PARITY_TOL_M} m at slot(s) {over}: {[slot_max[i] for i in over]}", ev)
    return _clause(7, "PASS", f"max |delta| <= {PARITY_TOL_M} m at all {N_SLOTS} slots on >= {PARITY_MIN_WINDOWS} windows (smallest panel {nmin}; {len(panels)} panel(s))", ev)


def overall(clauses: list) -> str:
    st = [c["status"] for c in clauses]
    if "FAIL" in st:
        return "FAIL"
    if "INCOMPLETE" in st:
        return "INCOMPLETE"
    return "PASS"


def beside(fd8: dict | None, tf: dict | None, tac: dict | None, fd_ps: dict | None, raw_fd8: dict | None, raw_fd_ps: dict | None = None) -> dict:
    """Everything printed beside the condition (outside it): the per-surface form of Amendment 1, RETIME_TRAINFIT, the flat-0.5 arm, the pre-ruling ha0_ext reads, the raw weights, TAC of the beside arms."""
    def e(f, arm, surf, key):
        out = {}
        for s in SEEDS:
            a = _arm_rows(f, s, surf, arm) if f else None
            out[f"s{s}"] = None if a is None else a.get(key)
        return out
    b = {"pre_ruling_ha0_ext": {"E1_S2": e(fd8, ARM, "S2", "minus_ha0_ext"), "E3_S6": e(fd8, ARM, "S6", "minus_ha0_ext")},
         "flat_0p5_CONST_G05": {"E1_S2_vs_pose": e(fd_ps or fd8, "CONST_G05", "S2", "minus_ha0_ext_pose"), "E3_S6_vs_pose": e(fd_ps or fd8, "CONST_G05", "S6", "minus_ha0_ext_pose")},   # the per-surface file carries it
         "per_surface_RETIME_DEPLOY_amendment1": {"E1_S2_vs_pose": e(fd_ps, PS_ARM, "S2", "minus_ha0_ext_pose"), "E3_S6_vs_pose": e(fd_ps, PS_ARM, "S6", "minus_ha0_ext_pose"),
                                                  "clause2_on_this_arm": clause_e(2, "S2", fd_ps, "minus_ha0_ext_pose", PS_ARM)["status"] if fd_ps else None,
                                                  "clause3_on_this_arm": clause_e(3, "S6", fd_ps, "minus_ha0_ext_pose", PS_ARM)["status"] if fd_ps else None},
         "RETIME_TRAINFIT_per_surface": {"E1_S2_vs_pose": e(tf, "RETIME_TRAINFIT", "S2", "minus_ha0_ext_pose"), "E3_S6_vs_pose": e(tf, "RETIME_TRAINFIT", "S6", "minus_ha0_ext_pose"),
                                         "clause2_on_this_arm": clause_e(2, "S2", tf, "minus_ha0_ext_pose", "RETIME_TRAINFIT")["status"] if tf else None,
                                         "clause3_on_this_arm": clause_e(3, "S6", tf, "minus_ha0_ext_pose", "RETIME_TRAINFIT")["status"] if tf else None},
         "raw_weights_tree": None}
    if raw_fd8:
        b["raw_weights_tree"] = {"tree": raw_fd8.get("tree_name") or raw_fd8.get("tree"), "form": ARM, "E1_S2_vs_pose": e(raw_fd8, ARM, "S2", "minus_ha0_ext_pose"), "E3_S6_vs_pose": e(raw_fd8, ARM, "S6", "minus_ha0_ext_pose")}
    elif raw_fd_ps:
        b["raw_weights_tree"] = {"tree": raw_fd_ps.get("tree_name") or raw_fd_ps.get("tree"), "form": PS_ARM + " (per-surface form; no FORWARD_DEPLOY8 file for the raw tree)",
                                 "E1_S2_vs_pose": e(raw_fd_ps, PS_ARM, "S2", "minus_ha0_ext_pose"), "E3_S6_vs_pose": e(raw_fd_ps, PS_ARM, "S6", "minus_ha0_ext_pose")}
    if tac:
        b["tactical_per_surface_form"] = {k: v for k, v in (tac.get("clause5_beside_per_surface_RETIME_DEPLOY") or {}).items() if k in ("status", "separated_worse_cells", "arm")}
        t = {}
        for surf in ("S2", "S6"):
            for sd in (0, 1):
                try:
                    c = tac["surfaces"][surf][f"s{sd}"]["compliant"]["cells"]
                    t[f"{surf}|s{sd}"] = {k: {"os_minus_refcv7": c["os_minus_refcv7_50400"][k], "TRAINFIT_minus_refcv7": c["RETIME_TRAINFIT_minus_refcv7_50400"][k],
                                              "per_surface_DEPLOY_minus_refcv7": c["RETIME_DEPLOY_minus_refcv7_50400"][k], "DEPLOY8_minus_os": c.get(f"{ARM}_minus_os", {}).get(k)} for k in TAC_KEYS}
                except (KeyError, TypeError):
                    t[f"{surf}|s{sd}"] = None
        b["tactical_beside"] = t
    return b


def evaluate(tree: str, fd8, tf, tac, pr, pp=None, fd_ps=None, raw_fd8=None, deploy_sha: str | None = None, raw_fd_ps=None) -> dict:
    clauses = [clause1(fd8, tf, deploy_sha), clause_e(2, "S2", fd8), clause_e(3, "S6", fd8), clause4(fd8), clause5(tac), clause6(pr), clause7(pp)]
    ov = overall(clauses)
    orient = tree != CONDITION_TREE
    return {"tool": "ship_r8_verdict.py", "generated": time.strftime("%FT%T"), "tree": tree, "condition_tree": CONDITION_TREE, "orientation_only": orient,
            "label": (f"SHIP-R8 on {tree}: {ov} -- ORIENTATION ONLY (the condition's tree is {CONDITION_TREE}; this is not the verdict)" if orient else
                      f"SHIP-R8 VERDICT on the condition's tree {tree}: {ov}"),
            "overall": ov, "clauses": clauses, "arm": ARM, "condition_text": CONDITION, "amendment_text": AMENDMENT,
            "per_clause": {str(c["id"]): c["status"] for c in clauses},
            "caveats": ["single training seed", "T1 open loop (not driving performance)", "the stage was found post-hoc", "the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF)",
                        "the train gains under-correct on eval (H-R8-GAINTRAJ-1)", "the bootstrap answers 'another draw of EPISODES' only (not another training or inference run)"],
            "beside": beside(fd8, tf, tac, fd_ps, raw_fd8, raw_fd_ps)}


# ------------------------------------------------------------------------------------------ IO
def _load(p):
    if not p:
        return None
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f(c) -> str:
    if not isinstance(c, dict) or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")


def render(v: dict) -> str:
    L = [f"# SHIP_R8_{v['tree']}", "", f"**{v['label']}**", "", f"scored arm: `{v['arm']}` (one call on all 8 model slots, AMENDMENT 2)", "", f"condition: {v['condition_text']}", "", f"amendments: {v['amendment_text']}", "",
         "| clause | status | reading |", "|---|---|---|"]
    for c in v["clauses"]:
        L.append(f"| {c['text']} | **{c['status']}** | {c['reason']} |")
    L += ["", "## evidence", ""]
    e2, e3 = v["clauses"][1]["evidence"], v["clauses"][2]["evidence"]
    for nm, e in ((f"E1 (S2) {v['arm']} - ha0_ext_pose", e2), (f"E3 (S6) {v['arm']} - ha0_ext_pose", e3)):
        L.append(f"* {nm}: " + " | ".join(f"{s}: {_f(c)}" for s, c in (e.get("cells") or {}).items()))
    e4 = v["clauses"][3]["evidence"]
    L.append(f"* FZR-NR per family: {e4.get('FZR-NR_per_family')}; FZR-LAT HOLDS {e4.get('FZR-LAT_HOLDS')}")
    e5 = v["clauses"][4]["evidence"]
    if e5.get("cells"):
        L += ["", f"TACTICAL cells ({v['arm']} - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):", "", "| cell | delta [lo, hi] |", "|---|---|"]
        for n, c in e5["cells"].items():
            L.append(f"| {n} | {_f(c)} |")
    e7 = v["clauses"][6]["evidence"]
    if e7.get("slot_max_abs_m"):
        L.append(f"* PRIOR PARITY max |delta| by slot (m, worst over panels): {e7['slot_max_abs_m']['values']}; smallest panel {e7.get('min_windows_per_panel')} windows")
    b = v["beside"]
    ps = b["per_surface_RETIME_DEPLOY_amendment1"]
    L += ["", "## printed beside (outside the condition)", "",
          "* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): " + " ; ".join(f"{k}: " + " | ".join(f"{s}: {_f(c)}" for s, c in d.items()) for k, d in b["pre_ruling_ha0_ext"].items()),
          "* flat 0.5 (CONST_G05) - ha0_ext_pose: " + " ; ".join(f"{k}: " + " | ".join(f"{s}: {_f(c)}" for s, c in d.items()) for k, d in b["flat_0p5_CONST_G05"].items()),
          "* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: " + " ; ".join(f"{k}: " + " | ".join(f"{s}: {_f(c)}" for s, c in d.items()) for k, d in ps.items() if isinstance(d, dict)),
          "* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: " + " ; ".join(f"{k}: " + " | ".join(f"{s}: {_f(c)}" for s, c in d.items())
                                                                                   for k, d in b["RETIME_TRAINFIT_per_surface"].items() if isinstance(d, dict))]
    if b.get("raw_weights_tree"):
        r = b["raw_weights_tree"]
        L.append(f"* raw weights ({r['tree']}, {r.get('form')}): " + " ; ".join(f"{k}: " + " | ".join(f"{s}: {_f(c)}" for s, c in d.items()) for k, d in r.items() if isinstance(d, dict)))
    else:
        L.append("* raw weights: no FORWARD_DEPLOY8 / FORWARD_DEPLOY file for the raw-weights tree")
    L += ["", "caveats: " + "; ".join(v["caveats"]), ""]
    return "\n".join(L)


def paths(tree: str, root: Path | None = None) -> dict:
    tf_dir, dep_dir = ((root / "retime_trainfit"), (root / "retime_deploy")) if root else (TF_DIR, DEP_DIR)
    raw_tree = tree.replace("ema_unc", "_unc") if "ema_unc" in tree else None
    return {"tf": tf_dir / f"FORWARD_TRAINFIT_{tree}.json", "fd8": dep_dir / f"FORWARD_DEPLOY8_{tree}.json", "tac": dep_dir / f"TAC_{tree}.json", "pp": dep_dir / "PRIOR_PARITY.json",
            "fd_ps": dep_dir / f"FORWARD_DEPLOY_{tree}.json", "pr": (root / "paired_read" / f"PAIRED_READ_{tree}.json") if root else Path(f"{ARCHIVE}/{tree}/PAIRED_READ_{tree}.json"),
            "raw_fd8": (dep_dir / f"FORWARD_DEPLOY8_{raw_tree}.json") if raw_tree else None,
            "raw_fd_ps": (dep_dir / f"FORWARD_DEPLOY_{raw_tree}.json") if raw_tree else None, "deploy_sha": dep_dir / "FROZEN_GAINS_RETIME_deploy.sha256"}


def run(tree: str, root: Path | None = None, out_dir: Path | None = None) -> dict:
    p = paths(tree, root)
    dsha = p["deploy_sha"].read_text().strip() if p["deploy_sha"].exists() else None
    v = evaluate(tree, _load(p["fd8"]), _load(p["tf"]), _load(p["tac"]), _load(p["pr"]), _load(p["pp"]), _load(p["fd_ps"]), _load(p["raw_fd8"]), dsha, _load(p["raw_fd_ps"]))
    v["inputs"] = {k: (str(x) if x else None) for k, x in p.items()}
    v["inputs_sha256"] = {k: (hashlib.sha256(Path(x).read_bytes()).hexdigest() if x and Path(x).exists() else None) for k, x in p.items()}
    v["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    od = Path(out_dir) if out_dir else OUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    jp = od / f"SHIP_R8_{tree}.json"
    jp.write_text(json.dumps(v, indent=1, default=str) + "\n", encoding="utf-8")
    (od / f"SHIP_R8_{tree}.md").write_text(render(v), encoding="utf-8")
    chk = json.loads(jp.read_text(encoding="utf-8"))              # assert on the ARTIFACT
    assert chk["overall"] in ("PASS", "FAIL", "INCOMPLETE") and len(chk["clauses"]) == 7
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    v = run(ap.parse_args().tree)
    print(v["label"])
    for c in v["clauses"]:
        print(f"  {c['text'][:70]:<70} {c['status']:<10} {c['reason'][:120]}")
    print(f"ZZSHIPVERDICT_{v['overall']}ZZ {v['tree']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
