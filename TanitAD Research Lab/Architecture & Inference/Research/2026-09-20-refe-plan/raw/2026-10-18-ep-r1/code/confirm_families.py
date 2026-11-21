#!/usr/bin/env python3
"""FOUR METRIC FAMILIES on the confirm seams (MM_DECISIONS_EPR1B D8, second bullet; CLAUDE.md "EVERY EVAL REPORTS FOUR METRIC FAMILIES").  CPU only.  NEW file; it wires the EXISTING tool.

THE EXISTING TOOL, unchanged.  `families6.py` (FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/) is called exactly as `stage3_full.compute_families` calls it (the call
FREEPOINT's `fp_read.py` and RL-1's `nt_read.py` use): `python families6.py --seam S --inputs <navtest_inputs.json.gz> --stage 1 --label L --out O --n-boot 2000`.  Its per-reading blocks are compared to
P2's with `proxy_eval.fam_adverse` (unpaired episode-cluster interval non-overlap on the LONGITUDINAL / LATERAL components, direction-aware; conservative); that function is PORTED here as `fam_adverse` because
importing `proxy_eval` drags in the model stack, and an equivalence test extracts the original from its source and requires identical output on every pair of banked families JSONs.

WHAT THIS ADDS.  `stage_read` reports `families6: PENDING`.  Here every reading of the look (P2, EPR1_s*, SRTTC_s*, JOINT_s*) gets a report with all four families:
  LONGITUDINAL  target-speed accuracy (three bands), speed MAE / bias, along-track MAE, acceleration MAE with episode-cluster intervals; DISTANCE-KEEPING: UNAVAILABLE with the tool's own reason and n
                (no lead-agent track reaches families6.py: PREREG_EPR1.md:110, PREREG_SELRANK_ARM.md:94) -- a work item, never a pass
  LATERAL       heading, curvature, yaw-rate and cross-track error with intervals
  TACTICAL      lateral / longitudinal / 5-way manoeuvre decision (accuracy, kappa) and tactical goal-setting, as families6 reports them
  STRATEGIC     PAUSED (PI 2026-09-19, PI_DECISION_QUEUE item 25): printed as PAUSED with the ruling, never a pass, never omitted.  families6's own refusal (NAVSIM scores no strategic decision) is kept beside it.
EVERY required item is COMPUTED, UNAVAILABLE with a reason and an n, or MISSING.  MISSING means the tool's block lacks the item and gives no reason: that reading is INCOMPLETE, and an incomplete reading
makes the whole report incomplete (a missing metric is a work item, not an excuse).  families6 never ran for a reading => NOT COMPUTED with the reason, never a pass.
THE REGISTERED CLAUSE (PREREG_EPR1.md:110, PREREG_SELRANK_ARM.md:94): "no longitudinal or lateral family component separated adverse (families6, unpaired, conservative)" = `vs_p2.clears`: adverse == []
AND at least one component compared (zero compared means the clause was NOT evaluated; a zero is a claim about the probe).

POST-LEDGER ONLY.  families6 scores the seams against the logged human future, which is outcome-bearing; like the landed `ready_for_score.json` it waits for the look.  `run_families` refuses, before it
reads anything else, when the ledger is absent.  Nothing here opens `pdms` / `sub`.

    confirm_families.py run --out-dir <the read's out dir> --look-ledger <ledger> [--families6 F] [--inputs J] [--python P] [--n-boot 2000] [--runner subprocess|inprocess]
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import refe_head_confirm as C   # noqa: E402  (LANDED, frozen)
import epr1_dev as D            # noqa: E402  (LANDED, frozen)

SCHEMA = "confirm-families-v1"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EXPORT_CANDIDATES = (os.environ.get("REFE_NAVTEST_INPUTS", ""), "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
_F6 = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/families6.py"
FAMILIES6_CANDIDATES = (
    os.environ.get("REFE_FAMILIES6", ""),
    os.path.normpath(os.path.join(HERE, *([".."] * 7), _F6)),           # in the repo: REFE/raw/<pkg>/code -> the repo root is seven levels up
    "D:/Projects/TanitAD/" + _F6,
    "E:/Projects/TanitAD/" + _F6,
)
MODEL_KEY = "refcv6"                                                      # the row name families6.py hard-codes for the model under test
FAMILIES_DIR = "families"
REPORT_NAME = "confirm_families_report.json"
STRATEGIC_RULING = "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE item 25): the strategic layer is switched off temporarily; reported PAUSED, never a pass, never omitted"
N_BOOT_DEFAULT = 2000                                                     # stage3_full.compute_families
FAMILIES6_SHA256 = "5dfd0cf154b1ea29a488ca4a888817d1bdade70e2bfbe6661bd08e5a8a1f8551"      # MEASURED 2026-10-19: identical at the tip (FETCH_HEAD), on D: and on E:; a different tool is DISCLOSED, not refused

# the required items per family (CLAUDE.md binding table).  (item, [components of families6's ci.components]) for the interval-bearing families
LONG_ITEMS = (("target_speed_accuracy", ("target_speed_acc.within_0.5_mps", "target_speed_acc.within_1.0_mps", "target_speed_acc.within_2.0_mps")),
              ("speed_mae_mps", ("speed_mae_mps",)), ("speed_bias_mps", ("speed_bias_mps",)), ("along_mae_m", ("along_mae_m",)), ("accel_mae_mps2", ("accel_mae_mps2",)))
LAT_ITEMS = (("heading_mae_deg", ("heading_mae_deg",)), ("curvature_mae_1pm", ("curvature_mae_1pm",)), ("yaw_rate_mae_degps", ("yaw_rate_mae_degps",)),
             ("cross_mae_m", ("cross_mae_m",)), ("cross_final_mae_m", ("cross_final_mae_m",)))
TACTICAL_ITEMS = ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed", "goal_setting")


class FamiliesRefusal(C.ConfirmRefusal):
    """A precondition of the families step is not met (the tool, its inputs, the ledger, a seam)."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the ONLY file-system / process functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
_io_exists, _io_sha256, _io_read_bytes, _io_makedirs, _io_write_bytes = C._io_exists, C._io_sha256, C._io_read_bytes, C._io_makedirs, C._io_write_bytes


def _io_path_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _io_file_sha256(path: str) -> str | None:
    """sha256 of a TOOL file (families6.py), which lives outside the confirm guard's dirs; None when unreadable."""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _io_run_subprocess(argv: list, log_path: str, cwd: str) -> int:
    """The production runner: the SAME call `stage3_full.compute_families` makes (the tanitad venv's python, families6's own directory as cwd, UTF-8 output, stdout+stderr to a log)."""
    with open(log_path, "w", encoding="utf-8") as lg:
        return int(subprocess.call(argv, cwd=cwd, env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=lg, stderr=subprocess.STDOUT))


def _io_run_inprocess(argv: list, log_path: str, cwd: str) -> int:
    """The test runner: families6.main(argv[2:]) in THIS process (one Python process), same argv, same tool file."""
    path = argv[1]
    spec = importlib.util.spec_from_file_location("families6_under_confirm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["families6_under_confirm"] = mod
    buf = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    old_cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
        os.chdir(cwd)
        with contextlib.redirect_stdout(buf):
            rc = int(mod.main(argv[2:]) or 0)
    except SystemExit as ex:
        rc = int(ex.code or 0)
    except Exception as ex:                                                             # noqa: BLE001
        buf.write(f"\nRAISED {type(ex).__name__}: {ex}\n")
        rc = 1
    finally:
        os.chdir(old_cwd)
        sys.modules.pop("families6_under_confirm", None)
    buf.flush()
    with open(log_path, "wb") as lg:
        lg.write(buf.buffer.getvalue())
    return rc


RUNNERS = {"subprocess": _io_run_subprocess, "inprocess": _io_run_inprocess}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# configuration and tool resolution
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FamiliesConfig:
    families6: str | None = None
    inputs: str | None = None
    python: str = PY_T
    n_boot: int = N_BOOT_DEFAULT
    runner: str = "subprocess"


def resolve_tool(cfg: FamiliesConfig) -> dict:
    """-> {"families6", "inputs", "python"} or raises FamiliesRefusal naming what is missing.  Used by the wrapper's PREFLIGHT, i.e. before the look is spent: a read that spends the look and then cannot
    run its families would have wasted the one look."""
    f6 = cfg.families6 or next((c for c in FAMILIES6_CANDIDATES if c and _io_path_exists(c)), None)
    inp = cfg.inputs or next((c for c in EXPORT_CANDIDATES if c and _io_path_exists(c)), None)
    miss = []
    if not f6 or not _io_path_exists(f6):
        miss.append(f"families6.py (tried {[c for c in FAMILIES6_CANDIDATES if c] if not cfg.families6 else [cfg.families6]})")
    if not inp or not _io_path_exists(inp):
        miss.append(f"the navtest inputs export (tried {[c for c in EXPORT_CANDIDATES if c] if not cfg.inputs else [cfg.inputs]})")
    if cfg.runner == "subprocess" and not _io_path_exists(cfg.python):
        miss.append(f"the python interpreter {cfg.python}")
    if cfg.runner not in RUNNERS:
        miss.append(f"runner {cfg.runner!r} (one of {sorted(RUNNERS)})")
    if miss:
        raise FamiliesRefusal("the families6 tool is not runnable here: missing " + "; ".join(miss) + ". The look must not be spent until it is.")
    return {"families6": os.path.abspath(f6), "inputs": os.path.abspath(inp), "python": cfg.python}


def families6_argv(tool: dict, cfg: FamiliesConfig, seam: str, label: str, out: str) -> list:
    """The argv `stage3_full.compute_families` builds (checked against its source by a test)."""
    return [tool["python"], tool["families6"], "--seam", seam, "--inputs", tool["inputs"], "--stage", "1", "--label", label, "--out", out, "--n-boot", str(cfg.n_boot)]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# fam_adverse, ported from proxy_eval.py (equivalence-tested against its source)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
FAM_SKIP = ("gt_progress_mean_m", "t0_axis_gt_self_ratio")          # properties of the human future, not of the arm


def fam_direction(metric: str) -> str:
    m = metric.lower()
    if "within_" in m or "accuracy" in m or "kappa" in m:
        return "higher"
    if "bias" in m:
        return "magnitude"
    if "progress_ratio" in m:
        return "closer_to_1"
    return "lower"                                                    # mae / rmse / error / rate


def fam_adverse(fw: dict, fp: dict, arm_key: str = MODEL_KEY) -> dict:
    """families6 blocks of the BASELINE (fw) and the arm (fp): every LONGITUDINAL / LATERAL component whose arm interval lies entirely on the WORSE side of the baseline's.  Unpaired interval
    non-overlap, a conservative reading.  Tactical carries accuracy / kappa beside, untested."""
    adverse, compared = [], 0
    for fam in ("longitudinal", "lateral"):
        cw = ((fw.get("families", {}).get(arm_key, {}).get(fam) or {}).get("ci") or {}).get("components", {})
        cp = ((fp.get("families", {}).get(arm_key, {}).get(fam) or {}).get("ci") or {}).get("components", {})
        for m in sorted(set(cw) & set(cp)):
            if m.rsplit(".", 1)[-1] in FAM_SKIP or not all(
                    isinstance(x.get(k), (int, float)) for x in (cw[m], cp[m]) for k in ("lo", "hi")):
                continue
            compared += 1
            (wl, wh), (pl, ph) = (cw[m]["lo"], cw[m]["hi"]), (cp[m]["lo"], cp[m]["hi"])
            d = fam_direction(m)
            if d == "higher":
                bad = ph < wl
            elif d == "lower":
                bad = pl > wh
            elif d == "magnitude":
                bad = (pl > wh >= 0 and wl >= 0) or (ph < wl <= 0 and wh <= 0)
            else:
                bad = (pl > wh >= 1.0 and wl >= 1.0) or (ph < wl <= 1.0 and wh <= 1.0)
            if bad:
                adverse.append({"family": fam, "metric": m, "direction": d, "W_ci": [wl, wh], "P_ci": [pl, ph]})
    tac = {}
    for k in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        tw = ((fw.get("families", {}).get(arm_key, {}).get("tactical") or {}).get(k) or {})
        tp = ((fp.get("families", {}).get(arm_key, {}).get("tactical") or {}).get(k) or {})
        tac[k] = {"W": {q: tw.get(q) for q in ("accuracy", "kappa")}, "P": {q: tp.get(q) for q in ("accuracy", "kappa")}}
    return {"adverse": adverse, "components_compared": compared, "tactical_no_interval": tac,
            "strategic": "UNAVAILABLE in NAVSIM (no strategic decision scored; the arm has no strategic layer)"}


def vs_p2(base: dict | None, arm: dict | None) -> dict:
    """The registered families6 clause for ONE reading against P2: clears iff nothing separated adverse AND something was compared."""
    if not base or not arm:
        return {"evaluated": False, "clears": False, "reason": "a families6 JSON is absent: NOT COMPUTED (never a pass)"}
    r = fam_adverse(base, arm)
    r["evaluated"] = bool(r["components_compared"] > 0)
    r["clears"] = bool(r["evaluated"] and not r["adverse"])                                    # MUTATION-TARGET-FAM-CLEARS
    r["note"] = "REPORTED against the registered clause (PREREG_EPR1.md:110); unpaired interval non-overlap, a conservative reading; zero components compared = NOT evaluated"
    return r


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the four-family report of ONE reading
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _component_item(ci: dict, keys: tuple) -> dict:
    comps = (ci or {}).get("components") or {}
    got, miss = {}, []
    for k in keys:
        c = comps.get(k)
        if isinstance(c, dict) and all(_num(c.get(q)) for q in ("mean", "lo", "hi")):
            got[k] = {"value": c["mean"], "ci95": [c["lo"], c["hi"]], "n_windows": c.get("n_windows"), "n_episodes": c.get("n_episodes")}
        else:
            miss.append(k)
    if not miss:
        return {"status": "COMPUTED", "components": got}
    un = (ci or {}).get("unavailable")
    listed = [k for k in miss if un and k in json.dumps(un, default=str)]
    if len(miss) == len(listed) and not got:
        return {"status": "UNAVAILABLE", "reason": "listed in the block's ci.unavailable: " + json.dumps(un, default=str)[:300], "n": (ci or {}).get("n_windows")}
    return {"status": "MISSING", "reason": f"components {miss} are absent from the block's ci.components and are not listed as unavailable", "components_present": sorted(got)}      # MUTATION-TARGET-FAM-MISSING


def _decision_item(blk: dict | None) -> dict:
    if not isinstance(blk, dict):
        return {"status": "MISSING", "reason": "the block is absent"}
    if blk.get("status") != "OK":
        return {"status": "UNAVAILABLE", "reason": str(blk.get("reason") or blk.get("status")), "n": blk.get("n")}
    kappa_ok = _num(blk.get("kappa")) or (blk.get("kappa") is None and bool(blk.get("kappa_undefined_reason")))          # a single-class window set has no kappa; the tool says why
    if not (_num(blk.get("accuracy")) and kappa_ok):
        return {"status": "MISSING", "reason": "status OK but accuracy is not a number, or kappa is neither a number nor undefined-with-a-reason", "n": blk.get("n")}
    return {"status": "COMPUTED", "accuracy": blk["accuracy"], "kappa": blk.get("kappa"), "kappa_undefined_reason": blk.get("kappa_undefined_reason"), "n": blk.get("n"),
            "classes": blk.get("classes"), "ci": blk.get("ci")}


def _goal_item(blk: dict | None) -> dict:
    if not isinstance(blk, dict):
        return {"status": "MISSING", "reason": "the block is absent"}
    if blk.get("status") != "OK":
        return {"status": "UNAVAILABLE", "reason": str(blk.get("reason") or blk.get("status")), "n": blk.get("n")}
    keys = ("goal_point_error_m", "goal_bearing_mae_deg", "goal_range_ratio")
    if not all(k in blk for k in keys[:2]):
        return {"status": "MISSING", "reason": f"status OK but {[k for k in keys[:2] if k not in blk]} are absent", "n": blk.get("n")}
    return {"status": "COMPUTED", **{k: blk.get(k) for k in keys + ("goal_long_bias_m", "goal_lat_bias_m")}, "n": blk.get("n"), "ci": blk.get("ci")}


def _complete(items: dict) -> bool:
    return bool(items) and all(v["status"] in ("COMPUTED", "UNAVAILABLE") for v in items.values()) and all(("reason" in v and v.get("n") is not None) for v in items.values() if v["status"] == "UNAVAILABLE")


def four_family_report(reading: str, fam_json: dict | None, base_json: dict | None, n_tokens_expected: int | None = None, baseline: bool = False) -> dict:
    """ONE reading, all four families.  `fam_json` is families6's output file (None => NOT COMPUTED); `base_json` is P2's; `baseline` marks P2 itself (nothing to compare it to)."""
    rep = {"reading": reading, "strategic": {"status": "PAUSED", "ruling": STRATEGIC_RULING, "is_a_pass": False}}      # MUTATION-TARGET-FAM-STRATEGIC
    blk = ((fam_json or {}).get("families") or {}).get(MODEL_KEY)
    if not isinstance(blk, dict) or blk.get("status") == "RAISED" or "longitudinal" not in blk:
        why = ("families6 produced no output for this reading" if not fam_json else
               f"families6 block unusable: {json.dumps(blk, default=str)[:300] if blk is not None else 'no ' + MODEL_KEY + ' row'}; file status {fam_json.get('status')!r} {fam_json.get('reason')!r}")
        rep.update({"status": "NOT_COMPUTED", "reason": why, "longitudinal": {"complete": False}, "lateral": {"complete": False}, "tactical": {"complete": False}, "complete": False,
                    "vs_p2": {"baseline": True} if baseline else {"evaluated": False, "clears": False, "reason": why}})
        return rep
    lg, lt, tc = blk.get("longitudinal") or {}, blk.get("lateral") or {}, blk.get("tactical") or {}
    long_items = {name: _component_item(lg.get("ci"), keys) for name, keys in LONG_ITEMS}
    dk = lg.get("distance_keeping")
    if isinstance(dk, dict) and dk.get("status") == "OK":
        long_items["distance_keeping"] = {"status": "COMPUTED", "value": {k: v for k, v in dk.items() if k not in ("status",)}, "n": dk.get("n")}
    elif isinstance(dk, dict) and dk.get("status") == "UNAVAILABLE":
        long_items["distance_keeping"] = {"status": "UNAVAILABLE", "reason": str(dk.get("reason")), "n": dk.get("n")}      # MUTATION-TARGET-FAM-DK
    else:
        long_items["distance_keeping"] = {"status": "MISSING", "reason": "the longitudinal block carries no distance_keeping entry"}
    lat_items = {name: _component_item(lt.get("ci"), keys) for name, keys in LAT_ITEMS}
    tac_items = {k: (_goal_item(tc.get(k)) if k == "goal_setting" else _decision_item(tc.get(k))) for k in TACTICAL_ITEMS}
    if tc.get("status") not in (None, "OK") and not any(v["status"] == "COMPUTED" for v in tac_items.values()):
        for k in tac_items:
            tac_items[k] = {"status": "UNAVAILABLE", "reason": str(tc.get("reason") or tc.get("status")), "n": tc.get("n")}
    rep["longitudinal"] = {"items": long_items, "complete": _complete(long_items), "n_windows": lg.get("n_windows")}
    rep["lateral"] = {"items": lat_items, "complete": _complete(lat_items), "n_windows": lt.get("n_windows")}
    rep["tactical"] = {"items": tac_items, "complete": _complete(tac_items), "n": tc.get("n"), "note": "accuracy / kappa as families6 reports them; the registered bars do not test this family"}
    rep["strategic"]["families6_own_status"] = (blk.get("strategic") or {}).get("status")
    rep["strategic"]["families6_own_reason"] = str((blk.get("strategic") or {}).get("reason") or "")[:300]
    rep["n_scored"], rep["n_logs"] = (fam_json or {}).get("n"), (fam_json or {}).get("n_logs")
    rep["n_tokens_expected"] = n_tokens_expected
    rep["all_tokens_scored"] = bool(n_tokens_expected is None or rep["n_scored"] == n_tokens_expected)
    rep["status"] = "COMPUTED"
    rep["complete"] = bool(rep["longitudinal"]["complete"] and rep["lateral"]["complete"] and rep["tactical"]["complete"])             # MUTATION-TARGET-FAM-COMPLETE
    rep["vs_p2"] = {"baseline": True} if baseline else vs_p2(base_json, fam_json)
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the run (post-ledger)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_one(guard, tool: dict, cfg: FamiliesConfig, seam: str, label: str, out_dir: str) -> dict | None:
    out = os.path.join(out_dir, f"{label}_families.json")
    log = os.path.join(out_dir, f"{label}_families.log")
    argv = families6_argv(tool, cfg, seam, label, out)
    guard.check_write(out)
    guard.check_write(log)
    rc = RUNNERS[cfg.runner](argv, log, os.path.dirname(tool["families6"]))
    if not _io_exists(guard, out):
        return {"_not_computed": True, "rc": rc, "log": log}
    doc, _sha = D.load_json(guard, out)
    doc["_rc"] = rc
    return doc


def run_families(opts: C.ConfirmOptions, cfg: FamiliesConfig, read_out_dir: str | None = None, out_dir: str | None = None) -> dict:
    t0 = time.time()
    guard = opts.guard()
    if not opts.look_ledger:
        raise FamiliesRefusal("families need --look-ledger")
    if not _io_exists(guard, opts.look_ledger):                                              # MUTATION-TARGET-FAM-LEDGER
        raise FamiliesRefusal(f"{opts.look_ledger} does not exist: the navtest look is not spent, so families6 (which reads the logged human future) does not run (the landed seams wait for the read)")
    tool = resolve_tool(cfg)
    read_out = read_out_dir or opts.out_dir
    res_doc, res_sha = D.load_json(guard, os.path.join(read_out, "confirm_read_result.json"))
    led, led_sha = D.load_json(guard, opts.look_ledger)
    names = sorted(led.get("picks_sha256") or {}, key=lambda n: (n != "P2", n))                  # P2 first: it is the paired baseline of every clause
    seams = res_doc.get("seams") or {}
    missing = [n for n in names if n not in seams]
    if missing:                                                                              # MUTATION-TARGET-FAM-COMPLETENESS
        raise FamiliesRefusal(f"the read result lists no seam for reading(s) {missing}: every reading of the look gets a families report, so this cannot start")
    fdir = out_dir or os.path.join(read_out, FAMILIES_DIR)
    _io_makedirs(guard, fdir)
    n_expected = (res_doc.get("estimator") or {}).get("n_tokens")
    docs, reports, runs = {}, {}, {}
    for nm in names:
        seam = seams[nm]["path"]
        if _io_sha256(guard, seam) != seams[nm]["sha256"]:                                   # MUTATION-TARGET-FAM-SEAMSHA
            raise FamiliesRefusal(f"{nm}: seam {seam} sha256 differs from the one the read recorded (edited or replaced after the look)")
        docs[nm] = run_one(guard, tool, cfg, seam, f"confirm_{nm}", fdir)
        runs[nm] = {"rc": docs[nm].get("_rc") if docs[nm] and not docs[nm].get("_not_computed") else (docs[nm] or {}).get("rc"), "produced_output": bool(docs[nm] and not docs[nm].get("_not_computed")),
                    "seam_sha256": seams[nm]["sha256"], "label": f"confirm_{nm}"}
        if docs[nm] is not None and docs[nm].get("_not_computed"):
            runs[nm]["log"] = docs[nm].get("log")
            docs[nm] = None
    base = docs.get("P2")
    for nm in names:
        reports[nm] = four_family_report(nm, docs[nm], base, n_expected or None, baseline=(nm == "P2"))
    arms = {}
    for arm, fam in (("EP-R1", "EPR1"), ("SR-TTC", "SRTTC")):
        rs = {sd: reports.get(f"{fam}_s{sd}") for sd in C.SEED_KEYS}
        if all(v is not None for v in rs.values()):
            arms[arm] = {"per_seed_clears": {sd: bool(rs[sd]["vs_p2"].get("clears")) for sd in C.SEED_KEYS},
                         "both_seeds_clear_families6_clause": bool(all(rs[sd]["vs_p2"].get("clears") for sd in C.SEED_KEYS)),                       # MUTATION-TARGET-FAM-BOTH
                         "adverse": {sd: rs[sd]["vs_p2"].get("adverse") for sd in C.SEED_KEYS}, "components_compared": {sd: rs[sd]["vs_p2"].get("components_compared") for sd in C.SEED_KEYS},
                         "clause": "no longitudinal or lateral family component separated adverse (families6, unpaired, conservative); PREREG_EPR1.md:110, PREREG_SELRANK_ARM.md:94"}
    rep = {"schema": SCHEMA, "stage": "FAMILIES", "tier": "T1 NAVSIM v1.1 navtest, self-action open loop (EVAL_DOCTRINE)", "look": {"ledger": opts.look_ledger, "ledger_sha256": led_sha},
           "read_result_sha256": res_sha, "tool": {**tool, "families6_sha256": _io_file_sha256(tool["families6"]), "families6_sha256_pinned": FAMILIES6_SHA256,
                                                    "tool_is_the_pinned_one": bool(_io_file_sha256(tool["families6"]) == FAMILIES6_SHA256), "n_boot": cfg.n_boot, "runner": cfg.runner},
           "tool_runs": runs, "readings": reports, "readings_expected": names,
           "all_readings_reported": bool(all(n in reports for n in names)), "all_complete": bool(all(r["complete"] for r in reports.values())),
           "not_computed": sorted(n for n, r in reports.items() if r["status"] != "COMPUTED"), "arms": arms,
           "strategic": {"status": "PAUSED", "ruling": STRATEGIC_RULING},
           "estimator": "episode-cluster bootstrap over the navtest logs, per reading (families6 --n-boot %d); comparisons against P2 are UNPAIRED interval non-overlap (conservative), as the registered clause states" % cfg.n_boot,
           "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    _io_write_bytes(guard, os.path.join(fdir, REPORT_NAME), C.canon(rep), exclusive=False)
    print(f"[{time.strftime('%H:%M:%S')}] families: {len(reports)} readings, all_complete={rep['all_complete']}, not_computed={rep['not_computed']}", flush=True)
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Four-family report (families6) for every reading of the navtest look. Post-ledger only.")
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--out-dir", required=True, help="the landed `read` stage's out dir (confirm_read_result.json and seams/)")
    ap.add_argument("--look-ledger", required=True)
    ap.add_argument("--families6", default=None)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--python", default=PY_T)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--runner", choices=sorted(RUNNERS), default="subprocess")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = C.ConfirmOptions(out_dir=a.out_dir, bank_dir=os.path.join(a.out_dir, "no_bank"), look_ledger=a.look_ledger)
    cfg = FamiliesConfig(families6=a.families6, inputs=a.inputs, python=a.python, n_boot=a.n_boot, runner=a.runner)
    try:
        run_families(opts, cfg)
    except (C.ConfirmRefusal, D.DevRefusal, D.DevGuardError) as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
