#!/usr/bin/env python3
"""strategic_paired.py -- the PAIRED STRATEGIC rows (S-GOAL) for the refcv9 E3 read, REPORT-ONLY beside FAM-1.  CPU only, no model run, no new estimator.

    python strategic_paired.py table --dump <battery dump dir> --labels <v7.2 labels jsonl.gz> --arm E3_K2DC02 --out table_e3.json [--tau RAD]
    python strategic_paired.py table --dump <refcv8 battery dump>  --labels <same>               --arm refcv8     --out table_r8.json [--tau RAD]
    python strategic_paired.py pair  --a table_e3.json --b table_r8.json --out strategic_e3_vs_r8.json

WHAT S-GOAL IS (MM design 2026-11-23, PI 2026-11-16: strategic = decision + goal/route-setting quality).  E3 (arm K2DC02) trains with ``--no-strategic`` (argv file
``stack/ops/runs.d/r9e3_K2DC02.argv.json``), so it has NO strategic head: S-DEC is NOT-COMPUTABLE by design (n = 0).  Goal / route-setting quality is read BEHAVIOURALLY:
nav compliance of the EMITTED PLAN -- the terminal heading of the model's emitted path (``plan_full_<cond>``) must have the sign of the COMMANDED turn and magnitude
>= tau, on INFORMATIVE windows only (``taniteval.nav_compliance``: true command left/right, token valid, the commanded manoeuvre overlaps [t_now, t_now + 6 s] by >= 1 s).

THE TRACE (file:line at the e3s_tree tip; all primary):
  * the dump's ``decisions/ep*.npz`` sidecar already holds, per window and per conditioning, ``plan_full_{nav_true,nav_shuffled,nav_zero}`` --
    ``taniteval/tools/refcv3_arm.py:2646-2672`` (``_navcomp_row``; ``conds_fed`` :2231, ``do_navzero`` :2246) -- plus ``nav_cmd`` / ``nav_cmd_shuf`` / ``nav_valid`` / ``ws``
    (:2628) and the GT (``gt_future_ext``, ``pose_last``).  So NO new model run and no dump change is needed: the per-window number is derived post hoc from the existing dump.
  * the compliance number: ``taniteval/taniteval/nav_compliance.py:387`` ``complies(signal, side, tau)`` = 1.0 where sign(signal) == the commanded side AND |signal| >= tau; per window,
    scored on ``compliance_arm`` (:432) ``comp[c][inf]``; the signal is ``terminal_heading`` (:250) of the plan.  ``from_refcv3_dump`` (:1227) reduces it to a pooled rate + a paired
    true-minus-zero / true-minus-shuffled delta ONLY -- it never exposes the per-window vector, which is what an A-vs-B pairing needs.  THIS tool re-derives it per window with the SAME
    library functions (load_labels, time_base_control, join_windows_to_labels, informative_mask, derive_tolerance, terminal_heading, complies) and is HELD EQUAL to
    ``from_refcv3_dump`` by test (rate, n, tau).
  * the threshold: ``nav_compliance_report`` (:806+) derives tau by Youden-J over the GT's terminal heading change on informative vs follow windows (GT + label only; fallback 15 deg).
    It is NOT the training ``--nav-compliance-tau-rad`` (0.1806 in launch_gate); it is a property of the corpus windows, so it is the SAME for two arms read on the same windows --
    ``pair`` REFUSES if the two tables disagree on tau, on the informative set, or on the window set.
  * route label compared against: the COMMANDED side of the TRUE nav token ``nav_cmd`` (``commanded_side``, :160), informative-gated by the v7.2 labels' manoeuvre timing.

TIER: every row is stamped ``T1*`` -- the battery arms ``os`` / ``os_navshuf`` / ``os_navzero`` are ONE forward at the window origin on the OBSERVED frames
(``refcv3_arm.py`` ``ARM_TIERS`` :247; "T1 (RULING OPEN)" :364-372).  The ruling on that star is open (EVAL_DOCTRINE); it is reported as stamped, never promoted to a closed-loop T1.
T0 is not used anywhere here.

ECHO CONTROL (``openloop_suite.py`` ``_strategic_block`` docstring: a row without its echo control is INADMISSIBLE).  Under nav_zero (nav withheld) the same plan readout is scored against the
SAME true command; the within-arm nav-true minus nav-zero effect is the metric.  ``pair`` REFUSES S-GOAL (status NOT-COMPUTABLE, no rows) when either table lacks nav_zero.  nav_shuffled
is added when BOTH tables carry it, and its absence is listed.

OUTPUT SHAPE.  ``{"STRATEGIC": {...}}``: ``status`` "REPORT-ONLY"; flat dict-valued rows -- ``r9_e3_verdict.strategic_report`` reads ONLY dict-valued entries of the STRATEGIC dict as rows (a
nested ``S-GOAL: {row: ...}`` dict would itself be read as ONE unreadable row).  Rows:
    S-GOAL.plan.nav_true               compliance, nav-true, A - B          better higher
    S-GOAL.plan.ECHO.nav_zero          the echo control: compliance with the true command when nav is WITHHELD, A - B   better lower (conventional: less scene-coincidence)
    S-GOAL.plan.ECHO.nav_shuffled      same under the permuted token (when both arms carry it)                          better lower
    S-GOAL.plan.EFFECT.true_minus_zero      within-arm (true - zero) effect, A's minus B's   better higher (the metric: how much the command moves the plan)
    S-GOAL.plan.EFFECT.true_minus_shuffled  same against the shuffle                         better higher
  ``S-DEC``: a NOT-COMPUTABLE stamp (no delta, no per-window arrays): the judge carries it VERBATIM under ``not_computable`` with its reason, n = 0 and tier
  (``stack/scripts/r9_e3_verdict.py`` ``strategic_report``, MM fix 2026-11-23) -- never read as a row, never a pass, never silently omitted.
Each row: delta, lo, hi, separated, better, n, n_episodes, estimator, tier (+ role, readout, tau).  Estimator: ``taniteval.ci.paired_episode_cluster_bootstrap`` B 2000 seed 0, windows paired
by ``(clip_id, ws)`` through ``ci.align_windows`` (never by ep### file index), episode = clip.  ``separated`` answers "another draw of EPISODES" only -- not another training run (H-ESTIM-SEED-1)
and not another inference run; this is one seed per arm.

TABLE SCHEMA ``r9-e3-strategic-table/1`` (JSON): {schema, arm, tier, readout "plan", dump_dir, labels, tau, tau_source, time_base, n_windows, n_informative, n_dropped_nonfinite,
  conds [..], clip_id [N], ws [N], informative [N] (bool), comply {cond: [N] floats 0/1}}.  ``informative`` already excludes windows whose plan is non-finite under ANY cond (counted).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)                       # <repo>/taniteval
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)

SCHEMA = "r9-e3-strategic-table/1"
TIER = "T1*"
READOUT = "plan"
CONDS = ("nav_true", "nav_shuffled", "nav_zero")
N_BOOT, SEED = 2000, 0
S_DEC_REASON = ("NOT-COMPUTABLE by design: the E3 recipe (arm K2DC02, argv file stack/ops/runs.d/r9e3_K2DC02.argv.json) trains with --no-strategic, so the model has no strategic "
                "decision head to read; n = 0")
STRATEGIC_REPORT_ONLY_NOTE = ("REPORT-ONLY beside FAM-1 (PI 2026-11-16): S-GOAL = nav compliance of the emitted plan, paired E3 vs the comparator, with its echo control; never a bar")


class Refused(Exception):
    pass


# --------------------------------------------------------------------------- #
# the producer: one dump -> one per-window table                                #
# --------------------------------------------------------------------------- #
def table_from_dump(dump_dir: str, labels_path: str, arm: str, *, tau: float | None = None) -> dict:
    from taniteval import nav_compliance as nc

    files = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    if not files:
        raise Refused(f"no decisions/ep*.npz under {dump_dir}")
    with open(os.path.join(dump_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    labels_path = nc.resolve_labels_path(labels_path, manifest.get("corpus") or {})
    if not labels_path or not os.path.exists(labels_path):
        raise Refused(f"labels blob not found at {labels_path!r}; pass --labels")
    labels = nc.load_labels(labels_path)
    ep_man = {e["file_index"]: e for e in manifest.get("episodes", [])}
    raw_off = int((((manifest.get("corpus") or {}).get("frames") or {}).get("provider_to_raw_frame_offset", 0)))

    eps = []
    for f in files:
        with np.load(f) as d:
            eps.append({k: d[k] for k in d.files} | {"_file": os.path.basename(f)})
    need = ("plan_full_nav_true", "plan_full_nav_zero", "gt_future_ext", "pose_last", "ep_poses", "nav_cmd", "nav_valid", "ws")
    for k in need:
        bad = [e["_file"] for e in eps if k not in e or np.ndim(e[k]) < 1 or (k != "ep_poses" and e[k].shape[0] != e["ws"].shape[0])]
        if bad:
            raise Refused(f"sidecar key {k!r} is missing or not one-row-per-window in {len(bad)} episode(s) (e.g. {bad[0]}): "
                          f"the dump predates the nav-compliance keys or was written without the nav-zero forward -- re-roll with os_navzero; nothing here can be computed post hoc")
    conds = [c for c in CONDS if all(f"plan_full_{c}" in e and e[f"plan_full_{c}"].shape[0] == e["ws"].shape[0] for e in eps)]
    if "nav_true" not in conds or "nav_zero" not in conds:
        raise Refused("the dump carries no complete nav_true / nav_zero plan: the echo control is REQUIRED")

    clip, ws_l, eid, ep_yaw = [], [], [], {}
    for e in eps:
        fi = int(str(e["_file"]).replace("ep", "").replace(".npz", ""))
        cid = str(ep_man.get(fi, {}).get("clip_id", f"file{fi}"))
        n = int(e["ws"].shape[0])
        clip += [cid] * n
        ws_l += [int(x) for x in e["ws"]]
        eid += [cid] * n
        ep_yaw[cid] = np.asarray(e["ep_poses"])[:, 2]
    N = len(clip)
    if len(set(zip(clip, ws_l))) != N:
        raise Refused("duplicate (clip_id, ws) keys in the dump: pairing on them would be arbitrary")
    cat = lambda k: np.concatenate([np.asarray(e[k]) for e in eps])     # noqa: E731
    nav_true = cat("nav_cmd").astype(np.int64)
    nav_valid = cat("nav_valid").astype(bool)
    ws = np.asarray(ws_l, dtype=np.float64)
    t_now = (ws + raw_off) * nc.DT

    tb = nc.time_base_control(ep_yaw, labels, dt=nc.DT, raw_offset_frames=raw_off)
    if tb["status"] != "OK":
        raise Refused(f"label time base not established on the GT ({tb['status']}): no informative mask exists (this says nothing about the arm)")
    join = nc.join_windows_to_labels(clip, t_now, labels, time_base=tb["chosen"])
    inf = nc.informative_mask(nav_true, nav_valid, t_now, join["t_start_s"], join["t_end_s"], horizon_s=nc.PLAN_HORIZON_S)

    # tau: derived on the GT exactly as nav_compliance_report does (GT + label only), unless given
    gt = cat("gt_future_ext").astype(np.float64)
    gtv = cat("gt_future_valid_ext").astype(bool) if all("gt_future_valid_ext" in e for e in eps) else np.ones(gt.shape[:2], bool)
    pl = cat("pose_last").astype(np.float64)
    gt_dpsi = nc.wrap_pi(gt[:, -1, 2] - pl[:, 2])
    gt_dpsi[~gtv[:, -1]] = np.nan
    follow = (nav_true == nc.NAV_FOLLOW) & nav_valid
    if tau is None:
        d = nc.derive_tolerance(gt_dpsi[inf], gt_dpsi[follow])
        if d.get("tau") is not None:
            tau, tau_src = float(d["tau"]), "derived: Youden-J on the GT heading change (nav_compliance_report rule)"
        else:
            tau, tau_src = math.radians(15.0), "fallback 15 deg (GT tolerance underivable)"
    else:
        tau_src = "given (--tau)"

    side = nc.commanded_side(nav_true)
    comply, fin = {}, np.ones(N, dtype=bool)
    for c in conds:
        sig = nc.terminal_heading(cat(f"plan_full_{c}").astype(np.float64))
        fin &= np.isfinite(sig)
        comply[c] = nc.complies(sig, side, tau)
    n_drop = int((inf & ~fin).sum())
    inf = inf & fin                                   # a NaN plan is not a non-compliant plan (complies() maps NaN to 0.0): drop it, count it
    return {"schema": SCHEMA, "arm": arm, "tier": TIER, "readout": READOUT, "dump_dir": dump_dir, "labels": labels_path,
            "tau": float(tau), "tau_source": tau_src, "time_base": tb["chosen"], "n_windows": int(N), "n_informative": int(inf.sum()),
            "n_dropped_nonfinite": n_drop, "conds": list(conds), "clip_id": clip, "ws": ws_l, "informative": [bool(x) for x in inf],
            "comply": {c: [float(x) for x in comply[c]] for c in conds}}


# --------------------------------------------------------------------------- #
# the pairing                                                                  #
# --------------------------------------------------------------------------- #
def _check_table(t, name):
    if not isinstance(t, dict) or t.get("schema") != SCHEMA:
        raise Refused(f"{name}: not a {SCHEMA} table")
    n = len(t.get("clip_id", []))
    if not (n == len(t.get("ws", [])) == len(t.get("informative", [])) and n > 0):
        raise Refused(f"{name}: clip_id / ws / informative lengths disagree or are empty")
    for c, v in (t.get("comply") or {}).items():
        if len(v) != n:
            raise Refused(f"{name}: comply[{c}] has {len(v)} entries for {n} windows")


def _row(a, b, eid, *, better, role, tau, n_boot=N_BOOT, seed=SEED, extra=None):
    from taniteval import ci
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, float), np.asarray(b, float), list(eid), n_boot=n_boot, seed=seed)
    row = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "better": better,
           "n": r["n_windows"], "n_episodes": r["n_episodes"], "estimator": r["estimator"], "tier": TIER, "role": role,
           "readout": READOUT, "tau": tau}
    for k in ("degenerate", "undefined"):
        if r.get(k):
            row[k] = r[k]
    row.update(extra or {})
    return row


def pair_tables(ta: dict, tb: dict, *, a_name: str = "A", b_name: str = "B", n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    from taniteval import ci
    _check_table(ta, "a")
    _check_table(tb, "b")
    base = {"status": "REPORT-ONLY", "reason": STRATEGIC_REPORT_ONLY_NOTE,
            "S-DEC": {"status": "NOT-COMPUTABLE", "reason": S_DEC_REASON, "n": 0, "tier": TIER}}
    meta = {"A": a_name, "B": b_name, "delta_is": "A - B", "tier": TIER,
            "pairing": "(clip_id, ws) via taniteval.ci.align_windows; episode = clip; B 2000 seed 0"}   # a SIBLING of STRATEGIC: any dict inside it would be read as a row
    missing = [f"{nm}: {c}" for nm, t in ((a_name, ta), (b_name, tb)) for c in ("nav_true", "nav_zero") if c not in (t.get("comply") or {})]
    if missing:
        base["S-GOAL_status"] = "NOT-COMPUTABLE"
        base["S-GOAL_reason"] = (f"echo control missing ({'; '.join(missing)}): a nav-compliance row without its nav-zero control is INADMISSIBLE "
                                 f"(openloop_suite._strategic_block) -- no S-GOAL row is emitted")
        base["S-GOAL_rows"] = []
        base["S-GOAL"] = {"status": "NOT-COMPUTABLE", "reason": base["S-GOAL_reason"], "n": 0, "tier": TIER}   # a stamp the judge carries (not a row)
        return {"STRATEGIC": base, "_meta": meta}
    if abs(float(ta["tau"]) - float(tb["tau"])) > 1e-9:
        raise Refused(f"tau differs between the arms ({ta['tau']} vs {tb['tau']}): compliance at different thresholds is not paired; re-run `table` with a common --tau")
    keys_a, keys_b = list(zip(ta["clip_id"], ta["ws"])), list(zip(tb["clip_id"], tb["ws"]))
    idx_a, idx_b, eid, rep = ci.align_windows(keys_a, list(range(len(keys_a))), keys_b, list(range(len(keys_b))))
    if rep["n_only_a"] or rep["n_only_b"]:
        raise Refused(f"the two arms' window sets differ ({rep}): no silent intersection -- re-dump on the same window set")
    ia, ib = np.asarray(idx_a), np.asarray(idx_b)
    inf_a = np.asarray(ta["informative"], bool)[ia]
    inf_b = np.asarray(tb["informative"], bool)[ib]
    if not np.array_equal(inf_a, inf_b):
        raise Refused(f"the informative mask differs on {int((inf_a != inf_b).sum())} paired windows (labels / nav / non-finite plans differ between the arms): not paired")
    m = inf_a
    if not m.any():
        base["S-GOAL_status"] = "NOT-COMPUTABLE"
        base["S-GOAL_reason"] = "no informative window"
        base["S-GOAL_rows"] = []
        base["S-GOAL"] = {"status": "NOT-COMPUTABLE", "reason": base["S-GOAL_reason"], "n": 0, "tier": TIER}   # a stamp the judge carries (not a row)
        return {"STRATEGIC": base, "_meta": meta}
    eidm = [e for e, k in zip(eid, m) if k]
    get = lambda t, i, c: np.asarray(t["comply"][c], float)[i][m]     # noqa: E731
    tau = float(ta["tau"])
    rows = {}
    kw = dict(n_boot=n_boot, seed=seed, tau=tau)
    rows["S-GOAL.plan.nav_true"] = _row(get(ta, ia, "nav_true"), get(tb, ib, "nav_true"), eidm, better="higher", role="compliance-nav-true", **kw)
    rows["S-GOAL.plan.ECHO.nav_zero"] = _row(get(ta, ia, "nav_zero"), get(tb, ib, "nav_zero"), eidm, better="lower", role="echo-control", **kw)
    rows["S-GOAL.plan.EFFECT.true_minus_zero"] = _row(get(ta, ia, "nav_true") - get(ta, ia, "nav_zero"), get(tb, ib, "nav_true") - get(tb, ib, "nav_zero"), eidm,
                                                      better="higher", role="echo-control-effect", **kw)
    if "nav_shuffled" in ta["comply"] and "nav_shuffled" in tb["comply"]:
        rows["S-GOAL.plan.ECHO.nav_shuffled"] = _row(get(ta, ia, "nav_shuffled"), get(tb, ib, "nav_shuffled"), eidm, better="lower", role="echo-control", **kw)
        rows["S-GOAL.plan.EFFECT.true_minus_shuffled"] = _row(get(ta, ia, "nav_true") - get(ta, ia, "nav_shuffled"), get(tb, ib, "nav_true") - get(tb, ib, "nav_shuffled"),
                                                              eidm, better="higher", role="echo-control-effect", **kw)
    else:
        base["nav_shuffled_status"] = "not carried by both tables: shuffle rows omitted (nav_zero echo control present)"
    base.update(rows)
    base["S-GOAL_rows"] = list(rows)
    meta["pairing_detail"] = {"n_windows_common": int(len(eid)), "n_informative": int(m.sum()), "n_episodes": int(len(set(eidm))), "tau": tau,
                        "tau_source": {a_name: ta.get("tau_source"), b_name: tb.get("tau_source")}}
    return {"STRATEGIC": base, "_meta": meta}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--dump", required=True)
    t.add_argument("--labels", default=None)
    t.add_argument("--arm", required=True)
    t.add_argument("--tau", type=float, default=None)
    t.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--a-name", default="E3")
    p.add_argument("--b-name", default="refcv8")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "table":
            out = table_from_dump(a.dump, a.labels, a.arm, tau=a.tau)
        else:
            out = pair_tables(json.load(open(a.a, encoding="utf-8")), json.load(open(a.b, encoding="utf-8")), a_name=a.a_name, b_name=a.b_name)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
