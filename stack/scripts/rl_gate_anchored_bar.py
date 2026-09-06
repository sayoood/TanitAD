#!/usr/bin/env python3
"""DERIVE an ANCHORED bar on the gate statistic's OWN scale, then score against it.

Executes the Master Mind ruling of 2026-09-06:

    RULING 2. THE 0.30 BAR IS VOID ON THE NEW STATISTIC. ... Derive a bar on
    `S6_wmr`'s OWN scale, pre-register it, and only then score. Anchor it to
    something measured, not chosen. ... Do not reuse 0.30, and do not pick the
    bar after seeing where the arms land.

    RULING 3. Authorised: map `progress_lead_cap True -> "lead"` and evaluate it
    as a pre-registered arm on a FRESH grid, with both outcomes committed in
    advance.

THE ANCHOR (pre-registered as a PROCEDURE, never as a number):

    the one-sided alpha critical value of the EPISODE-CLUSTERED SIGN-FLIP
    PERMUTATION NULL of the statistic, computed on the rows being scored.

        for b in 1..B:  s_e ~ U{-1,+1} per EPISODE;  null_b = S(s_e * g)
        crit = percentile(null, 100*alpha)
        PASS iff  S_observed <= crit    (equivalently perm p <= alpha)

WHY A PROCEDURE AND NOT A NUMBER. A literal bar is exactly the object that can be
transposed across a change of statistic -- which is the defect the ruling
convicts. A quantile OF THE STATISTIC ITSELF is unrepresentable in another
statistic's units, and it recomputes on whatever rows it meets, so it cannot
ride across a change of split, grid or episode count either. THIS TOOL THEREFORE
REFUSES A PREREG THAT CARRIES A LITERAL `PREREG_BAR:` LINE.

WHAT THE BAR IS NOT: a FLOOR, necessary and not sufficient. Clearing a
no-preference null says the reward's preference for the human is DETECTABLE, not
that it is large enough to train on. The magnitude is printed beside every
verdict.

WHICH VARIANCE: EPISODES. Not training (H-ESTIM-SEED-1), not inference. No arm
is trained and no planner samples here; the cells are deterministic arithmetic
over one fixed geometry.

Reads the rows banked by `rl_statistic_mutation_rank.py`, so the geometry cannot
differ from the ranking by anything except the grid.

ASCII-ONLY OUTPUT: every print() here is 7-bit. A cp1252 console makes a
non-ASCII print a fatal error, and a tool that dies at its last line has thrown
away the compute that produced the answer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(HERE))

_spec = importlib.util.spec_from_file_location(
    "rl_statistic_mutation_rank_for_anchor",
    os.path.join(REPO, "stack", "scripts", "rl_statistic_mutation_rank.py"))
_m = importlib.util.module_from_spec(_spec)
sys.modules["rl_statistic_mutation_rank_for_anchor"] = _m
_spec.loader.exec_module(_m)
np = _m.np
CANDIDATES = _m.CANDIDATES
LADDER = _m.LADDER
WEIGHTS = _m.WEIGHTS
gaps = _m.gaps
comp = _m.comp
boot_stat = _m.boot_stat

#: the banked G-REWARD all-window RATE on the pre-repair reward, fit120 grid.
#: CONTROL 4 (CROSS-GRID TOOL IDENTITY): re-scoring the OLD rows through THIS tool
#: must reproduce it to 0.000e+00, which is what makes a difference between the two
#: grids a difference in EPISODES rather than in code.
BANKED_FIT120_RATE = 0.441780259484316

RE_ID = re.compile(r"^[ \t]*PREREG_HYPOTHESIS_ID:\s*(\S+)\s*$", re.M)
RE_STAT = re.compile(r"^[ \t]*PREREG_STATISTIC:\s*(\S+)\s*$", re.M)
RE_ANCHOR = re.compile(r"^[ \t]*PREREG_ANCHOR:\s*(\S+)\s*$", re.M)
RE_ALPHA = re.compile(r"^[ \t]*PREREG_ALPHA:\s*([-+0-9.eE]+)\s*$", re.M)
RE_DIR = re.compile(r"^[ \t]*PREREG_DIRECTION:\s*(<=|>=)\s*$", re.M)
RE_RUNG = re.compile(r"^[ \t]*PREREG_RUNG:\s*(\S+)\s*$", re.M)
RE_CELL = re.compile(r"^[ \t]*PREREG_CELL:\s*(\S+)\s*$", re.M)
RE_GRID = re.compile(r"^[ \t]*PREREG_GRID:\s*(\S+)\s*$", re.M)
RE_PAIRED = re.compile(r"^[ \t]*PREREG_PAIRED:\s*(\S+)\s*-\s*(\S+)\s*$", re.M)
RE_LITERAL_BAR = re.compile(r"^[ \t]*PREREG_BAR:", re.M)

SUPPORTED_ANCHORS = ("signflip_episode_null",)


def _blocks(txt):
    """Split the prereg into per-hypothesis blocks, keyed by PREREG_HYPOTHESIS_ID."""
    out, cur, cur_id = {}, [], None
    for line in txt.splitlines():
        m = RE_ID.match(line)
        if m:
            if cur_id:
                out[cur_id] = "\n".join(cur)
            cur_id, cur = m.group(1), [line]
        elif cur_id:
            cur.append(line)
    if cur_id:
        out[cur_id] = "\n".join(cur)
    return out


def read_prereg(path, hyp_id):
    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    if RE_LITERAL_BAR.search(txt):
        raise SystemExit(
            "[anchor] the pre-registration %s carries a LITERAL `PREREG_BAR:` line -- "
            "refusing. An anchored bar is a PROCEDURE, never a number; a literal is "
            "exactly the object that can be transposed across a change of statistic."
            % path)
    blocks = _blocks(txt)
    if hyp_id not in blocks:
        raise SystemExit("[anchor] %s declares no hypothesis %r (has: %s)"
                         % (path, hyp_id, ", ".join(sorted(blocks)) or "none"))
    b = blocks[hyp_id]
    paired = RE_PAIRED.search(b)
    got = {"hypothesis_id": hyp_id,
           "statistic": RE_STAT.search(b), "rung": RE_RUNG.search(b),
           "grid": RE_GRID.search(b)}
    need = ["statistic", "rung", "grid"]
    if paired is None:
        got.update({"anchor": RE_ANCHOR.search(b), "alpha": RE_ALPHA.search(b),
                    "direction": RE_DIR.search(b), "cell": RE_CELL.search(b)})
        need += ["anchor", "alpha", "direction", "cell"]
    missing = [k for k in need if got.get(k) is None]
    if missing:
        raise SystemExit("[anchor] hypothesis %s does not declare %s -- refusing"
                         % (hyp_id, ", ".join("PREREG_" + k.upper() for k in missing)))
    pre = {"hypothesis_id": hyp_id,
           "statistic": got["statistic"].group(1),
           "rung": got["rung"].group(1), "grid": got["grid"].group(1)}
    if pre["statistic"] not in CANDIDATES:
        raise SystemExit("[anchor] statistic %r is not a candidate" % pre["statistic"])
    if paired is not None:
        pre["paired"] = [paired.group(1), paired.group(2)]
    else:
        pre.update({"anchor": got["anchor"].group(1),
                    "alpha": float(got["alpha"].group(1)),
                    "direction": got["direction"].group(1),
                    "cell": got["cell"].group(1)})
        if pre["anchor"] not in SUPPORTED_ANCHORS:
            raise SystemExit("[anchor] anchor %r is not implemented (have: %s)"
                             % (pre["anchor"], ", ".join(SUPPORTED_ANCHORS)))
    return pre


# --------------------------------------------------------------------------- #
# THE ANCHOR                                                                    #
# --------------------------------------------------------------------------- #
def signflip_null(fn, g, eids, n_perm=4000, seed=0):
    """Episode-clustered sign-flip permutation null of an ARBITRARY statistic.

    Each episode's windows are flipped TOGETHER, so the |g| magnitudes are
    preserved EXACTLY and only the side the mass is paid to is randomised. The
    resulting distribution is the statistic under a reward with NO systematic
    preference between the two paths.
    """
    uniq = np.unique(eids)
    pos = {e: np.flatnonzero(eids == e) for e in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        s = rng.integers(0, 2, size=len(uniq)) * 2 - 1
        gp = g.copy()
        for e, si in zip(uniq, s):
            if si < 0:
                gp[pos[e]] = -gp[pos[e]]
        draws[b] = fn(gp)
    return draws


def anchored_bar(fn, g, eids, alpha, n_perm, seed):
    null = signflip_null(fn, g, eids, n_perm, seed)
    obs = fn(g)
    crit = float(np.percentile(null, 100.0 * alpha))
    p = float((null <= obs).mean())
    return {"crit": crit, "observed": obs, "perm_p": p,
            "null_median": float(np.median(null)),
            "null_q05": crit, "null_q95": float(np.percentile(null, 95.0)),
            "null_sd": float(null.std(ddof=1)), "n_perm": int(n_perm),
            "alpha": float(alpha)}, null


def paired_delta(fn, ga, gb, eids, n_boot=4000, seed=0, alpha=0.05):
    """Paired episode-cluster bootstrap of S(ga) - S(gb) on the SAME windows."""
    uniq = np.unique(eids)
    pos = {e: np.flatnonzero(eids == e) for e in uniq}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([pos[e] for e in pick])
        d[b] = fn(ga[sel]) - fn(gb[sel])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": float(fn(ga) - fn(gb)), "ci": [float(lo), float(hi)],
            "sd_boot": float(d.std(ddof=1)),
            "separated": bool(lo > 0.0 or hi < 0.0)}


def frozen_gap(rows, cell):
    fr = np.array([sum(w * float(r["frozen__" + cell].get(k, 0.0))
                       for k, w in WEIGHTS.items()) for r in rows])
    hu = np.array([sum(w * float(r["human__" + cell].get(k, 0.0))
                       for k, w in WEIGHTS.items()) for r in rows])
    return fr - hu, float((fr >= hu - 1e-12).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="rows banked by the ranking tool")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--hypothesis", action="append", required=True,
                    help="PREREG_HYPOTHESIS_ID to score; repeatable")
    ap.add_argument("--old-rows", default=None,
                    help="the fit120 rows, for CONTROL 4 (cross-grid tool identity)")
    ap.add_argument("--old-clips", default=None, help="fit120 clip list, for CONTROL 5")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=4000)
    ap.add_argument("--n-selftest", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with open(a.rows, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    eids = np.array([r["eid"] for r in rows])
    tg = np.array([float(r.get("gt_time_gap_min_s", np.nan)) for r in rows])
    clips = sorted({r["clip"] for r in rows})
    cells = sorted({k.split("__", 1)[1] for k in rows[0] if k.startswith("human__")})

    print("== GRID  %d windows / %d episodes / %d clips-with-lead"
          % (len(rows), len(np.unique(eids)), len(clips)))
    print("   cells: %s" % ", ".join(cells))

    # ------------------------------------------------------------------ #
    # CONTROLS -- computed BEFORE any verdict                              #
    # ------------------------------------------------------------------ #
    ctrl = {}
    print("\n== CONTROLS (each must read a KNOWN value)")

    # C1 NULL: the clone-and-add path at d=0 must reproduce `nopert` EXACTLY
    if "null" in cells and "nopert" in cells:
        w = 0.0
        for side in ("human", "hold_v0", "frozen"):
            for k in ("progress", "collision", "headway", "comfort", "feasibility"):
                w = max(w, float(np.abs(comp(rows, side, "null", k)
                                        - comp(rows, side, "nopert", k)).max()))
        ctrl["C1_NULL"] = {"max_abs_component_delta": w, "pass": bool(w == 0.0)}
        print("   C1 NULL                 max|dcomponent| %.3e  pass=%s"
              % (w, ctrl["C1_NULL"]["pass"]))

    # C2/C3 CAP INERTNESS and its DISCRIMINATING control
    if "p5_cap" in cells and "nopert_cap" in cells:
        w = 0.0
        for side in ("human", "hold_v0", "frozen"):
            w = max(w, float(np.abs(comp(rows, side, "p5_cap", "progress")
                                    - comp(rows, side, "nopert_cap", "progress")).max()))
        ctrl["C2_CAP_INERTNESS"] = {"max_abs_dprogress": w, "pass": bool(w == 0.0)}
        print("   C2 CAP INERTNESS        max|dprogress| %.3e  pass=%s"
              % (w, ctrl["C2_CAP_INERTNESS"]["pass"]))
    if "p5end_cap" in cells and "nopert_cap" in cells:
        d = np.abs(comp(rows, "hold_v0", "p5end_cap", "progress")
                   - comp(rows, "hold_v0", "nopert_cap", "progress"))
        ctrl["C3_CAP_DISCRIMINATING"] = {"max_abs_dprogress": float(d.max()),
                                         "moved_frac": float((d > 0).mean()),
                                         "pass": bool(d.max() > 0.0)}
        print("   C3 CAP DISCRIMINATING   max|dprogress| %.6f  moved %.4f  pass=%s"
              % (d.max(), (d > 0).mean(), ctrl["C3_CAP_DISCRIMINATING"]["pass"]))

    # C4 CROSS-GRID TOOL IDENTITY -- the OLD rows through THIS tool
    if a.old_rows:
        with open(a.old_rows, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        r_old = CANDIDATES["S1_rate"][0](gaps(old, "nopert_min"))
        err = abs(r_old - BANKED_FIT120_RATE)
        ctrl["C4_CROSS_GRID_TOOL_IDENTITY"] = {
            "fit120_all_window_rate": r_old, "banked": BANKED_FIT120_RATE,
            "abs_err": err, "n_windows_old": len(old), "pass": bool(err <= 1e-12)}
        print("   C4 CROSS-GRID IDENTITY  fit120 S1_rate %.15f  banked %.15f  "
              "err %.3e  pass=%s"
              % (r_old, BANKED_FIT120_RATE, err, err <= 1e-12))

    # C5 GRID DISJOINTNESS, with a NON-ZERO control
    if a.old_clips:
        with open(a.old_clips, "r", encoding="utf-8") as fh:
            oc = {ln.strip() for ln in fh if ln.strip()}
        inter = len(set(clips) & oc)
        ctrl["C5_GRID_DISJOINT"] = {
            "fresh_clips_with_lead": len(clips), "old_clips": len(oc),
            "intersection": inter, "nonzero_control_old_vs_old": len(oc & oc),
            "pass": bool(inter == 0 and len(oc & oc) > 0)}
        print("   C5 GRID DISJOINT        fresh(with lead) %d  old %d  "
              "intersection %d   NON-ZERO CONTROL old^old %d  pass=%s"
              % (len(clips), len(oc), inter, len(oc & oc),
                 ctrl["C5_GRID_DISJOINT"]["pass"]))

    out_hyps = {}
    for hyp_id in a.hypothesis:
        pre = read_prereg(a.prereg, hyp_id)
        fn = CANDIDATES[pre["statistic"]][0]

        # ---------------- PAIRED arm (H-RL-CAP-1) ----------------
        if "paired" in pre:
            ca, cb = pre["paired"]
            for c in (ca, cb):
                if c not in cells:
                    raise SystemExit("[anchor] cell %r absent from the rows" % c)
            ga, gb = gaps(rows, ca), gaps(rows, cb)
            res = paired_delta(fn, ga, gb, eids, a.n_boot, a.seed)
            attrib = {}
            for c in sorted({ca, cb, "nopert_min"} & set(cells)):
                attrib[c] = {k: float((comp(rows, "hold_v0", c, k)
                                       - comp(rows, "human", c, k)).mean() * WEIGHTS[k])
                             for k in ("progress", "headway", "collision")}
            same_mech = None
            if ca in attrib and cb in attrib:
                same_mech = bool(
                    attrib[ca]["headway"] == attrib[cb]["headway"]
                    and attrib[ca]["collision"] == attrib[cb]["collision"])
            if res["separated"] and res["delta"] > 0:
                verdict = "A_REPLICATES"
            elif res["separated"] and res["delta"] < 0:
                verdict = "C_REVERSES"
            else:
                verdict = "B_DOES_NOT_REPLICATE"
            out_hyps[hyp_id] = {"prereg": pre, "paired": res, "attribution": attrib,
                                "headway_and_collision_identical": same_mech,
                                "OUTCOME": verdict}
            print("\n== %s   PAIRED  %s - %s   (rung %s, grid %s)"
                  % (hyp_id, ca, cb, pre["rung"], pre["grid"]))
            print("   S(%s) %.6f   S(%s) %.6f" % (ca, fn(ga), cb, fn(gb)))
            print("   delta %+.6f  CI [%+.6f, %+.6f]  separated=%s"
                  % (res["delta"], res["ci"][0], res["ci"][1], res["separated"]))
            print("   COMMITTED OUTCOME: %s" % verdict)
            print("   mechanism: headway and collision identical across the pair = %s"
                  % same_mech)
            continue

        # ---------------- ANCHORED bar (H-RL-GATE-STAT-3a/3b) ----------------
        if pre["cell"] not in cells:
            raise SystemExit("[anchor] cell %r absent from the rows" % pre["cell"])
        g = gaps(rows, pre["cell"])
        ladder = {}
        for thr in LADDER:
            m = ((np.isfinite(tg) & (tg <= thr)) if np.isfinite(thr)
                 else np.ones(len(rows), bool))
            label = "all" if not np.isfinite(thr) else "<=%.1fs" % thr
            if int(m.sum()) == 0:
                continue
            lo, hi, sd = boot_stat(fn, g[m], eids[m], a.n_boot, a.seed)
            ladder[label] = {"n_windows": int(m.sum()),
                             "n_episodes": int(len(np.unique(eids[m]))),
                             "value": fn(g[m]), "ci": [lo, hi], "sd_boot": sd}
        if pre["rung"] not in ladder:
            raise SystemExit("[anchor] rung %r not in the ladder" % pre["rung"])
        cellres = ladder[pre["rung"]]
        mrung = ((np.isfinite(tg) & (tg <= float(pre["rung"].strip("<=s"))))
                 if pre["rung"] != "all" else np.ones(len(rows), bool))

        bar, null = anchored_bar(fn, g[mrung], eids[mrung], pre["alpha"],
                                 a.n_perm, a.seed)
        # C6 PERMUTATION NULL SANITY: the null must be centred at neutral
        c6 = {"null_median": bar["null_median"],
              "pass": bool(abs(bar["null_median"] - 0.5) <= 0.02)}
        # C7 ANCHOR SELF-TEST: a NEUTRAL-BY-CONSTRUCTION gap must FAIL ~ (1-alpha)
        rng = np.random.default_rng(a.seed + 991)
        uniq = np.unique(eids[mrung])
        posn = {e: np.flatnonzero(eids[mrung] == e) for e in uniq}
        gm = g[mrung]
        npass = 0
        for _ in range(a.n_selftest):
            s = rng.integers(0, 2, size=len(uniq)) * 2 - 1
            gp = gm.copy()
            for e, si in zip(uniq, s):
                if si < 0:
                    gp[posn[e]] = -gp[posn[e]]
            if fn(gp) <= bar["crit"]:
                npass += 1
        c7 = {"neutral_pass_frac": npass / float(a.n_selftest),
              "expected": pre["alpha"], "n": a.n_selftest,
              "pass": bool(abs(npass / float(a.n_selftest) - pre["alpha"]) <= 0.05)}

        # the OTHER two anchors the ruling named -- REPORTED, never a verdict
        gfz, fz_frac = frozen_gap([r for r, k in zip(rows, mrung) if k], pre["cell"])
        anchor_b = fn(gfz)
        anchor_c = 0.5 - 2.0 * cellres["sd_boot"]

        # VACUITY CLASSIFICATION, pre-registered
        vac = "OK"
        if (0.5 - bar["crit"]) < cellres["sd_boot"]:
            vac = "WEAK"
        if bar["crit"] < anchor_b:
            vac = "UNREACHABLE"

        passed_point = bool(cellres["value"] <= bar["crit"])
        passed_strict = bool(cellres["ci"][1] <= bar["crit"])
        out_hyps[hyp_id] = {
            "prereg": pre, "ladder": ladder, "anchor": bar,
            "anchor_B_frozen_trivial_path": anchor_b,
            "anchor_C_neutral_minus_2sd": anchor_c,
            "frozen_ge_human_frac": fz_frac,
            "vacuity": vac,
            "controls": {"C6_NULL_SANITY": c6, "C7_ANCHOR_SELFTEST": c7},
            "VERDICT": {"rung": pre["rung"], "cell": pre["cell"],
                        "statistic": pre["statistic"],
                        "value": cellres["value"], "ci": cellres["ci"],
                        "bar_derived": bar["crit"], "perm_p": bar["perm_p"],
                        "PASS_primary_point_vs_crit": passed_point,
                        "PASS_strict_whole_interval": passed_strict}}
        print("\n== %s   ANCHORED  %s on cell %s (rung %s, grid %s)"
              % (hyp_id, pre["statistic"], pre["cell"], pre["rung"], pre["grid"]))
        print("   C6 NULL SANITY          null median %.6f (must be 0.5 +- 0.02)  pass=%s"
              % (c6["null_median"], c6["pass"]))
        print("   C7 ANCHOR SELF-TEST     a NEUTRAL gap clears the bar %.4f of the time "
              "(expected %.2f, n=%d)  pass=%s"
              % (c7["neutral_pass_frac"], pre["alpha"], c7["n"], c7["pass"]))
        print("   DERIVED BAR             crit %.6f   null median %.6f  null sd %.6f  "
              "null q95 %.6f  (n_perm %d, alpha %.2f)"
              % (bar["crit"], bar["null_median"], bar["null_sd"], bar["null_q95"],
                 bar["n_perm"], pre["alpha"]))
        print("   anchor B (frozen trivial-path reference) %.6f   "
              "anchor C (0.5 - 2*SD_boot) %.6f   [REPORTED, never the verdict]"
              % (anchor_b, anchor_c))
        print("   VACUITY                 %s" % vac)
        print("   %-9s %6s %5s %11s %-26s" % ("rung", "nwin", "neps", "value", "CI"))
        for label, c in ladder.items():
            print("   %-9s %6d %5d %11.6f [%9.6f, %9.6f]"
                  % (label, c["n_windows"], c["n_episodes"], c["value"],
                     c["ci"][0], c["ci"][1]))
        print("   VERDICT at rung %s: value %.6f  vs DERIVED bar %.6f   perm p %.4f"
              % (pre["rung"], cellres["value"], bar["crit"], bar["perm_p"]))
        print("   PRIMARY (point vs crit): %s      STRICT (whole interval): %s"
              % ("PASS" if passed_point else "FAIL",
                 "PASS" if passed_strict else "FAIL"))

    out = {
        "_what": "ANCHORED bar derived on the statistic's OWN scale, and the "
                 "pre-registered arms scored against it",
        "_prereg": os.path.basename(a.prereg),
        "_not_a_gate": "G-REWARD remains FAILED on its committed statistic (the RATE). "
                       "Nothing here re-scores it.",
        "_anchor": "one-sided alpha critical value of the EPISODE-CLUSTERED SIGN-FLIP "
                   "permutation null of the statistic, computed on the scored rows. "
                   "A FLOOR: necessary, not sufficient.",
        "_evidence_class": "MEASURED (ours)",
        "_tier": "T0 instrument probe, NON-PARITY RL-fit windows",
        "_estimator": "episode-cluster bootstrap (n_boot %d) and episode-clustered "
                      "sign-flip permutation (n_perm %d) -- both answer ONLY 'would "
                      "another draw of EPISODES say this?'. Not training variance "
                      "(H-ESTIM-SEED-1), not inference variance: no arm is trained and "
                      "no planner samples here." % (a.n_boot, a.n_perm),
        "grid": {"rows": os.path.basename(a.rows), "n_windows": len(rows),
                 "n_episodes": int(len(np.unique(eids))),
                 "n_clips_with_lead": len(clips), "cells": cells},
        "controls": ctrl, "hypotheses": out_hyps,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n[anchor] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
