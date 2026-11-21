#!/usr/bin/env python3
"""Field-loss F-vs-C read: per-token arrays -> shared-draw paired log-cluster bootstrap -> the card's verdict.

Implements PREREG_AUX_FIELD_LOSS_P7.md (FINAL 2026-11-17, Amendments A and B), section "Estimator, tier, endpoints, adverse checks, rule",
"Kill conditions" (K4), "Reports: four families" and "Outcomes committed in advance". It decides nothing the card did not decide.
Built on eval/p5c/p5c_read.py (Boot, per_token, contrast, entry_adverse, fam_direction): same draws, same seed, same polarity rule
EXCEPT (v3, 2026-11-20) the three lead-agent safety margins min_ttc_s / headway_min_m / time_gap_min_s, which are higher-is-safer: the inherited
fam_direction defaults them to lower-is-better, so a RISE read as ADVERSE. The correction is applied HERE (see entry_adverse_pol), so it holds even when
FIELD_READ_P5C_DIR points at an old harness export; every cell the correction changes is printed with both readings and the reason.

INPUT   transp_score rows (p5c_read_prep.py convert + score on the F and C seams; arms.<name>.{tracked,transparent}.{nc,dac,ddc,ttc,ep,pdms,c}),
        the two seams + proposals dumps (picked plan, 64-proposal fan), the harness CSVs, a seam ledger, the 64-proposal tables (M1, selection
        quality), fam_ci JSON of F vs C (families), strategic_family.py JSON, p5c_tactical.py JSON. Scores are printed x 100.
ESTIMATOR  paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, ONE draw matrix for every arm and contrast.
        It answers "would another draw of LOGS say this?" ONLY. Another TRAINING RUN is the replicate pair (Fs, Cs; H-ESTIM-SEED-1).
        Another INFERENCE run: none, the planner is deterministic (asserted by the P5-C determinism control on the same scorer).
TIER    T1: NAVSIM v1 PDMS, open-loop pseudo-simulation of the executed plan (LQR-tracked). Not closed loop. Not T0.
RULE    REFUTED (E1 hi < 0 = K4, or E2 DAC-zero-share lo > 0) -> ADOPT-shaped (E1 lo > 0, E2 DAC-zero-share hi < 0, no A1-A3 adverse, no
        other longitudinal/lateral component separated adverse) -> TRADE -> NOT PROVEN. Without BOTH replicates an ADOPT-shaped result
        prints "separated, single seed, not yet a lever claim", never ADOPT.
CONTROLS CTRL-SET, CTRL-ID, CTRL-CSV, CTRL-SEAM must each PASS or the read is INVALID (no verdict).
Run with --dry-run "<text>" on stand-in arms: every output is stamped DRY RUN - NOT A FIELD VERDICT and carries no field claim.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.environ.get("FIELD_READ_P5C_DIR", ""), os.path.join(HERE, "..", "p5c")):
    if _d and os.path.exists(os.path.join(_d, "p5c_read.py")):
        sys.path.insert(0, os.path.abspath(_d))
        break
import p5c_read as PR                      # noqa: E402  (Boot, per_token, contrast, entry_adverse, fam_direction, METRICS, SUBS)
from p5c_read import Boot, Refused, SEED, NBOOT, SUBS, METRICS, SCORERS   # noqa: E402
sys.path.insert(0, HERE)
import field_rawdac as RD              # noqa: E402  (ctrl_sim, ctrl_pick, load_rows: numpy only, no NAVSIM import)

DRY_STAMP = "DRY RUN — NOT A FIELD VERDICT"
TIER = "T1 (NAVSIM v1 PDMS, open-loop pseudo-simulation of the executed LQR-tracked plan; not closed loop, not T0)"
ANSWERS = ("another draw of LOGS only; another TRAINING run = the replicate pair (Fs - Cs floor, Fs - Cs sign reproduction; H-ESTIM-SEED-1); "
           "another INFERENCE run = none (deterministic planner, asserted by the P5-C determinism control)")
SINGLE_SEED = "separated, single seed, not yet a lever claim"
N_NAVTEST, L_NAVTEST = 12146, 136
A1_POINT, A2_BAR, A3_BAR = -1.0, 0.97, 0.90      # card: EP delta <= -1.0 ; picked 4 s along-track F/C hi < 0.97 ; fan spread F/C hi < 0.90
LARGE_E1, LARGE_CUT = 1.0, 0.30
CSV_BAR = 1e-9
T4 = 7                                            # pose index of t = 4.0 s (8 poses at dt 0.5 s: seam `sampling` = [8, 0.5])
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


# ------------------------------------------------------------------------------------------------------ io
def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_slim(paths, arms):
    """rows jsonl -> ({token: {log, arms: {a: {tracked, transparent}}}}, [error tokens not later scored]); keeps only what the read needs."""
    rows, errs = {}, {}
    for fp in paths:
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if "error" in r:
                    errs[r.get("token")] = str(r["error"])[:120]
                    continue
                keep = {a: {s: {m: r["arms"][a][s][m] for m in METRICS} for s in SCORERS} for a in arms if a in r.get("arms", {})}
                rows[r["token"]] = {"token": r["token"], "log": r["log"], "arms": keep}
                errs.pop(r["token"], None)
    return rows, sorted(errs)


def seam_geometry(seam_path, dump_path, tokens):
    """-> picked plan's 4 s along-track, fan spread (mean pairwise 4 s endpoint distance over the 64 proposals), identity + sha256s.
    `tokens` fixes the order. Refuses on a missing token, a non-finite value or a dump whose pick is not the seam's plan."""
    z = np.load(seam_path, allow_pickle=False)
    st = [str(t) for t in z["token"]]
    poses = np.asarray(z["poses"], float)
    if poses.ndim != 3 or poses.shape[1:] != (8, 3) or not np.isfinite(poses).all():
        raise Refused(f"seam {seam_path}: bad poses {poses.shape} or non-finite")
    smp = [float(x) for x in z["sampling"]]
    if smp != [8.0, 0.5]:
        raise Refused(f"seam {seam_path}: sampling {smp} != [8, 0.5]: pose index {T4} is not t = 4.0 s")
    si = {t: i for i, t in enumerate(st)}
    miss = [t for t in tokens if t not in si]
    if miss:
        raise Refused(f"seam {seam_path} lacks {len(miss)} tokens (first {miss[0]})")
    r = np.array([si[t] for t in tokens])
    out = {"seam": os.path.abspath(seam_path), "seam_sha256": sha256_file(seam_path), "n": len(tokens), "sampling": smp,
           "along4": poses[r, T4, 0], "seam_tokens": st}
    d = np.load(dump_path, allow_pickle=False)
    dt = [str(t) for t in d["token"]]
    di = {t: i for i, t in enumerate(dt)}
    miss = [t for t in tokens if t not in di]
    if miss:
        raise Refused(f"dump {dump_path} lacks {len(miss)} tokens (first {miss[0]})")
    rd = np.array([di[t] for t in tokens])
    prop = np.asarray(d["proposals"], float)[rd]
    pick = np.asarray(d["pick"])[rd]
    if prop.shape[1:] != (64, 8, 3) or not np.isfinite(prop).all():
        raise Refused(f"dump {dump_path}: bad proposals {prop.shape} or non-finite")
    out["pick_is_seam_plan"] = bool(np.array_equal(prop[np.arange(len(tokens)), pick], np.asarray(z["poses"], float)[r]))
    out["dump"] = os.path.abspath(dump_path)
    out["dump_sha256"] = sha256_file(dump_path)
    out["dump_tokens"] = dt
    e = prop[:, :, T4, :2]
    spread = np.empty(len(tokens))
    for a in range(0, len(tokens), 256):
        x = e[a:a + 256]
        dd = np.sqrt(((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1))
        spread[a:a + 256] = dd.sum((1, 2)) / (64 * 63)
    out["spread4"] = spread
    return out


# ------------------------------------------------------------------------------------------------------ estimands
def ratio_ci(bt: Boot, vf, vc) -> dict:
    """F/C ratio of the logs-resampled means (same draws as every other estimand)."""
    mf, mc = bt.draws(vf), bt.draws(vc)
    r = mf / mc
    pt = float(np.mean(vf) / np.mean(vc))
    return {"point": pt, "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def zero_share(P, arm, scorer, metric):
    return 100.0 * (P[arm][scorer][metric] == 0).astype(float)


def core(P, logs, F, C, geo, n_boot=NBOOT) -> dict:
    """E1/E2/E3, the six sub-scores, M2, A1-A3 for F - C on the per-token arrays P (x100). Pure: no files."""
    bt = Boot(logs, n_boot)
    con = PR.contrast(bt, P, F, C)
    out = {"n_tokens": len(logs), "n_logs": bt.L, "contrast_pdms_and_subscores": con}
    out["E1"] = {"endpoint": "PDMS F - C (tracked, decision endpoint)", "tracked": con["pdms"]["tracked"],
                 "transparent": con["pdms"]["transparent"], "transparent_label": PR.NO_TRACKER}
    z = {k: {s: zero_share(P, a, s, "dac") for s in SCORERS} for k, a in (("F", F), ("C", C))}
    dz = {s: bt.ci(z["F"][s] - z["C"][s]) for s in SCORERS}
    lev = {k: {s: bt.ci(z[k][s]) for s in SCORERS} for k in ("F", "C")}
    cut = None
    if lev["C"]["tracked"]["point"] > 0:
        cut = float((lev["F"]["tracked"]["point"] - lev["C"]["tracked"]["point"]) / lev["C"]["tracked"]["point"])
    out["E2"] = {"endpoint": "DAC sub-score F - C and DAC-zero share F - C (percentage points; the mechanism)", "dac": con["dac"]["tracked"],
                 "dac_zero_share_diff_pp": dz["tracked"], "dac_zero_share_levels_pct": lev, "dac_zero_share_transparent_diff_pp": dz["transparent"],
                 "relative_cut_of_dac_zero_share": cut}
    pz = {k: zero_share(P, a, "tracked", "pdms") for k, a in (("F", F), ("C", C))}
    out["E3"] = {"endpoint": "zeroed-token share (PDMS == 0) F - C, pp", "diff_pp": bt.ci(pz["F"] - pz["C"]),
                 "levels_pct": {k: bt.ci(v) for k, v in pz.items()}}
    out["six_subscores_tracked_and_transparent"] = {m: {"tracked": con[m]["tracked"], "transparent": con[m]["transparent"]} for m in SUBS}
    # M2: raw -> tracked transfer. "raw" = the plan executed as given (transparent scorer), see card_questions Q-F2.
    d_tr, d_tp = con["dac"]["tracked"], con["dac"]["transparent"]
    zt, zp = dz["tracked"], dz["transparent"]
    raw_up = d_tp["lo"] > 0
    tracked_up = d_tr["lo"] > 0
    out["M2"] = {"dDAC_tracked": d_tr, "dDAC_raw_transparent": d_tp, "dDAC_tracked_minus_raw": con["dac"]["TD"],
                 "dDACzero_share_tracked_pp": zt, "dDACzero_share_raw_pp": zp,
                 "raw_improves_tracked_does_not": bool(raw_up and not tracked_up),
                 "routing": ("tracker-aware compensation, not a new loss" if (raw_up and not tracked_up) else "no raw-vs-tracked split (raw improves and tracked does not: false)")}
    # adverse checks
    ep = con["ep"]["tracked"]
    a1 = {"quantity": "EP delta F - C (x100), tracked", "ci": ep, "bar": f"hi < 0 AND point <= {A1_POINT}", "adverse": bool(ep["hi"] < 0 and ep["point"] <= A1_POINT)}
    r2 = ratio_ci(bt, geo[F]["along4"], geo[C]["along4"])
    a2 = {"quantity": "picked plan 4 s along-track, mean F / mean C", "ci": r2, "bar": f"hi < {A2_BAR}", "adverse": bool(r2["hi"] < A2_BAR)}
    r3 = ratio_ci(bt, geo[F]["spread4"], geo[C]["spread4"])
    a3 = {"quantity": "fan spread (mean pairwise 4 s endpoint distance over 64 proposals), mean F / mean C", "ci": r3, "bar": f"hi < {A3_BAR}",
          "adverse": bool(r3["hi"] < A3_BAR)}
    out["A1"], out["A2"], out["A3"] = a1, a2, a3
    out["K4"] = {"condition": "E1 upper bound < 0", "fired": bool(con["pdms"]["tracked"]["hi"] < 0)}
    out["_boot"] = bt
    return out


#: v3 POLARITY CORRECTION, applied here and not only in p5c_read so it survives an old harness export. ONE rule: same tuple + matcher as p5c_read / proxy_eval
#: (test_field_read asserts they agree when the imported p5c_read carries them).
SAFETY_MARGIN_HIGHER = ("min_ttc_s", "headway_min_m", "time_gap_min_s")
POLARITY_REASON = "lead-agent safety margin (bigger = safer); the inherited fam_direction() has no rule for it and defaults to lower-is-better"


def is_safety_margin(metric: str) -> bool:
    seg = metric.lower().rsplit(".", 1)[-1]
    return seg in SAFETY_MARGIN_HIGHER or (seg.startswith("mean_") and seg[5:] in SAFETY_MARGIN_HIGHER)


def fam_direction_pol(k: str) -> str:
    return "higher" if is_safety_margin(k) else PR.fam_direction(k)


def entry_adverse_pol(k: str, e: dict):
    """is F separated WORSE than C on this component (diff = B - A = F - C)? p5c_read.entry_adverse EXCEPT the safety margins, which are higher-is-better."""
    if is_safety_margin(k):
        if e.get("status") != "OK":
            return None
        return bool(e["diff_B_minus_A"]["hi"] < 0)
    return PR.entry_adverse(k, e)


def polarity_changes(j) -> list:
    """every executed_tracked cell whose adverse flag the correction CHANGES vs the inherited (imported) rule, with both readings."""
    out = []
    fr = ((j or {}).get("frames") or {}).get("executed_tracked") or {}
    for fam in ("longitudinal", "lateral"):
        for k, e in ((fr.get("families") or {}).get(fam) or {}).items():
            if not isinstance(e, dict) or "status" not in e:
                continue
            old, new = PR.entry_adverse(k, e), entry_adverse_pol(k, e)
            if old != new:
                out.append({"family": fam, "component": k, "diff_F_minus_C": e.get("diff_B_minus_A"), "inherited_reading": {"direction": PR.fam_direction(k), "adverse": old},
                            "corrected_reading": {"direction": fam_direction_pol(k), "adverse": new}, "reason": POLARITY_REASON})
    return out


def family_guard(j, label_a="C", label_b="F") -> dict:
    """no LONGITUDINAL / LATERAL component of the TRACKED-EXECUTED frame separated adverse (fam_ci: diff = B - A = F - C)."""
    if j is None:
        return {"status": "NOT-EVALUATED", "reason": "no fam_ci JSON (F vs C) supplied", "adverse": [], "components_compared": 0}
    fr = (j.get("frames") or {}).get("executed_tracked")
    if not fr:
        return {"status": "NOT-EVALUATED", "reason": "fam_ci JSON has no executed_tracked frame", "adverse": [], "components_compared": 0}
    if j.get("label_b") not in (None, label_b) or j.get("label_a") not in (None, label_a):
        return {"status": "NOT-EVALUATED", "reason": f"fam_ci labels {j.get('label_a')!r}/{j.get('label_b')!r} are not {label_a!r}/{label_b!r}: the diff direction is unproven",
                "adverse": [], "components_compared": 0}
    adverse, comp, und = [], 0, []
    for fam in ("longitudinal", "lateral"):
        for k, e in (fr["families"].get(fam) or {}).items():
            if k.split(".")[-1] in ("gt_progress_mean_m", "t0_axis_gt_self_ratio"):
                continue
            r = entry_adverse_pol(k, e)
            if r is None:
                und.append(f"{fam}.{k}:{e.get('status')}")
                continue
            comp += 1
            if r:
                adverse.append({"family": fam, "component": k, "direction": fam_direction_pol(k), "diff_F_minus_C": e["diff_B_minus_A"]})
    if comp == 0:
        return {"status": "NOT-EVALUATED", "reason": "0 comparable LONGITUDINAL / LATERAL components", "adverse": [], "components_compared": 0, "components_undecided": und}
    changes = polarity_changes(j)
    inh = [f"{c['family']}.{c['component']}" for c in changes if c["inherited_reading"]["adverse"]]       # flagged by the inherited rule only
    still = [f"{a['family']}.{a['component']}" for a in adverse]
    inherited_status = "FAIL" if (adverse or inh) else "PASS"
    return {"status": "FAIL" if adverse else "PASS", "adverse": adverse, "components_compared": comp, "components_undecided": und,
            "frame": "executed_tracked", "estimator": j.get("estimator"),
            "polarity_corrections": changes,
            "guard_status_under_inherited_rule": {"status": inherited_status, "flipped_by_correction": inherited_status != ("FAIL" if adverse else "PASS"),
                                                  "adverse_inherited_only": inh, "adverse_both_rules": still}}


# ------------------------------------------------------------------------------------------------------ the card's rule
def decide(c: dict, fam: dict, controls_ok, rep=None, dry: bool = False) -> dict:
    """card: REFUTED, then ADOPT, TRADE, NOT PROVEN, in that order. `rep` = {"fs_minus_cs_e1": ci, "fs_minus_cs_dz": ci, "floor": |E1(Cs - C)|} or None."""
    e1, dz = c["E1"]["tracked"], c["E2"]["dac_zero_share_diff_pp"]
    cl = {"E1_lo_gt_0": bool(e1["lo"] > 0), "E2_dacZeroShare_hi_lt_0": bool(dz["hi"] < 0), "A1_adverse": c["A1"]["adverse"], "A2_adverse": c["A2"]["adverse"],
          "A3_adverse": c["A3"]["adverse"], "family_guard": fam["status"], "family_adverse": [a["component"] for a in fam.get("adverse", [])],
          "K4_E1_hi_lt_0": c["K4"]["fired"], "E2_dacZeroShare_lo_gt_0": bool(dz["lo"] > 0)}
    if controls_ok is not True:
        return {"verdict": "INVALID", "wording": "a control failed or is missing: no verdict; the Deltas are descriptive only", "clauses": cl}
    if c["K4"]["fired"]:
        return {"verdict": "REFUTED", "wording": "K4: E1's upper bound < 0; stop", "clauses": cl}
    if cl["E2_dacZeroShare_lo_gt_0"]:
        return {"verdict": "REFUTED", "wording": "the DAC-zero share rose (E2 lower bound > 0)", "clauses": cl}
    one_two = cl["E1_lo_gt_0"] and cl["E2_dacZeroShare_hi_lt_0"]
    adv123 = cl["A1_adverse"] or cl["A2_adverse"] or cl["A3_adverse"]
    if one_two and adv123:
        return {"verdict": "TRADE", "wording": "E1 and E2 hold, and at least one of A1-A3 is adverse", "clauses": cl}
    if one_two and fam["status"] == "FAIL":
        return {"verdict": "NOT PROVEN", "wording": "E1 and E2 hold, A1-A3 clear, but a longitudinal/lateral family component is separated adverse (ADOPT blocked; the card names no TRADE for this case, see Q-F4)",
                "clauses": cl, "blocked_by": cl["family_adverse"]}
    if one_two:
        if fam["status"] != "PASS":
            return {"verdict": "INCOMPLETE", "wording": "ADOPT-shaped on E1/E2/A1-A3 but the family guard was not evaluated (" + fam.get("reason", "") + "): no verdict", "clauses": cl}
        size = "LARGE" if (c["E1"]["tracked"]["point"] >= LARGE_E1 and (c["E2"]["relative_cut_of_dac_zero_share"] is not None
                                                                          and c["E2"]["relative_cut_of_dac_zero_share"] <= -LARGE_CUT)) else "SMALL"
        if rep is None:
            return {"verdict": "ADOPT-SHAPED", "wording": SINGLE_SEED, "size_label_if_adopted": size, "clauses": cl,
                    "note": "never ADOPT without the replicate pair (Fs, Cs): the seed floor is unmeasured"}
        both = bool(rep["fs_minus_cs_e1"]["point"] > 0 and rep["fs_minus_cs_dz"]["point"] < 0)
        above = bool(e1["lo"] > rep["floor"])
        cl.update(replicate_reproduces_both_signs=both, E1_lo_gt_seed_floor=above, seed_floor=rep["floor"])
        if both and above:
            return {"verdict": "ADOPT", "wording": f"ADOPT, {size}", "size_label": size, "clauses": cl}
        return {"verdict": "NOT PROVEN", "wording": "ADOPT-shaped, but the replicate pair does not reproduce both signs and/or E1's lower bound is not above the measured seed floor",
                "clauses": cl}
    return {"verdict": "NOT PROVEN", "wording": "NOT PROVEN", "clauses": cl}


# ------------------------------------------------------------------------------------------------------ controls
def ctrl_set(tokens_by_arm: dict, n_expected=None, logs=None, l_expected=None) -> dict:
    """identical token set AND order for every arm (rows, seams, dumps); the full-navtest population when n_expected is given."""
    ref_name = next(iter(tokens_by_arm))
    ref = tokens_by_arm[ref_name]
    bad = {k: ("set differs" if set(v) != set(ref) else "same set, different order") for k, v in tokens_by_arm.items() if list(v) != list(ref)}
    pop = []
    if n_expected is not None and len(ref) != n_expected:
        pop.append(f"n tokens {len(ref)} != {n_expected}")
    if l_expected is not None and logs is not None and len(set(logs)) != l_expected:
        pop.append(f"n logs {len(set(logs))} != {l_expected}")
    ok = not bad and not pop and len(ref) > 0 and len(set(ref)) == len(ref)
    return {"control": "CTRL-SET", "pass": bool(ok), "n": len(ref), "unequal": bad, "population_problems": pop, "arms": list(tokens_by_arm)}


def ctrl_csv(rows_raw: dict, arm: str, csv_path: str, expected_tokens, bar: float = CSV_BAR) -> dict:
    """per-token tracked sub-scores and PDMS of `arm` vs the harness CSV, bar 1e-9 (p5c_read_prep.check_identity, tightened)."""
    import p5c_read_prep as RP
    banked = RP.read_harness_csv(csv_path)
    res = RP.check_identity(rows_raw, [], arm, banked, expected_tokens, bar)
    res.update(control="CTRL-CSV", csv=os.path.abspath(csv_path), csv_sha256=sha256_file(csv_path))
    return res


def ctrl_seam(geo_by_arm: dict, ledger_path=None, convert_report=None, names=None) -> dict:
    """seam sha256 recorded (ledger) and re-checked against the file at read time; dump pick == seam plan; rows' convert report agrees."""
    fails, rec = [], {}
    for arm, g in geo_by_arm.items():
        again = sha256_file(g["seam"])
        again_d = sha256_file(g["dump"])
        rec[arm] = {"seam": g["seam"], "seam_sha256": g["seam_sha256"], "dump": g["dump"], "dump_sha256": g["dump_sha256"]}
        if again != g["seam_sha256"] or again_d != g["dump_sha256"]:
            fails.append(f"{arm}: file changed during the read")
        if not g["pick_is_seam_plan"]:
            fails.append(f"{arm}: dump proposals[pick] is not the seam's plan bit for bit")
    ledger_state = "none supplied"
    if ledger_path:
        if os.path.exists(ledger_path):
            old = json.load(open(ledger_path, encoding="utf-8"))
            for arm, r in rec.items():
                o = (old.get("arms") or {}).get(arm)
                if o is None:
                    fails.append(f"{arm}: not in the ledger {ledger_path}")
                elif o["seam_sha256"] != r["seam_sha256"] or o["dump_sha256"] != r["dump_sha256"]:
                    fails.append(f"{arm}: ledger sha256 != file sha256 (seam or dump changed since it was recorded)")
            ledger_state = "re-checked"
        else:
            with open(ledger_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"arms": rec}, f, indent=1)
            ledger_state = "recorded now (first sight)"
    cr = "none supplied"
    if convert_report:
        j = json.load(open(convert_report, encoding="utf-8"))
        cr = "checked"
        for arm, g in geo_by_arm.items():
            nm = (names or {}).get(arm, arm)
            a = (j.get("arms") or {}).get(nm)
            if a is None:
                fails.append(f"{arm}: absent from convert_report ({nm})")
            elif a.get("seam_sha256") != g["seam_sha256"]:
                fails.append(f"{arm}: rows were scored from a seam with sha256 {a.get('seam_sha256')} != {g['seam_sha256']}")
    return {"control": "CTRL-SEAM", "pass": not fails, "fails": fails, "arms": rec, "ledger": ledger_state, "convert_report": cr}


def zero_exact(o) -> list:
    """every number reachable from o that is not exactly 0.0, with its path (CTRL-ID)."""
    bad = []

    def rec(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, p + [str(k)])
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            if x != 0.0:
                bad.append(".".join(p))
    rec(o, [])
    return bad


def ctrl_id(P, logs, arm, geo, n_boot=NBOOT) -> dict:
    """the read of an arm against ITSELF: every Delta exactly 0.0, every ratio exactly 1.0, verdict NOT PROVEN."""
    c = core(P, logs, arm, arm, geo, n_boot)
    deltas = {"E1": c["E1"]["tracked"], "E1t": c["E1"]["transparent"], "E2dac": c["E2"]["dac"], "E2dz": c["E2"]["dac_zero_share_diff_pp"], "E3": c["E3"]["diff_pp"],
              "subs": {m: {s: v[s] for s in SCORERS} for m, v in c["six_subscores_tracked_and_transparent"].items()}, "M2": {k: c["M2"][k] for k in ("dDAC_tracked", "dDAC_raw_transparent")}}
    ratios = {"A2": c["A2"]["ci"], "A3": c["A3"]["ci"]}
    nonzero = zero_exact(deltas)
    nonone = [k for k, v in ratios.items() if not (v["point"] == 1.0 and v["lo"] == 1.0 and v["hi"] == 1.0)]
    fam = {"status": "PASS", "adverse": []}
    v = decide(c, fam, True, None)
    ok = not nonzero and not nonone and v["verdict"] == "NOT PROVEN"
    return {"control": "CTRL-ID", "pass": bool(ok), "arm": arm, "nonzero_deltas": nonzero, "ratios_not_one": nonone, "verdict": v["verdict"]}


# ------------------------------------------------------------------------------------------------------ M1 and tactical selection quality
def load_table(paths):
    t = {}
    for fp in paths:
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    if "error" not in r:
                        t[r["token"]] = r
    return t


def table_arrays(tab: dict, tokens, tracked_pdms_x100, pick):
    """per-token arrays from a 64-proposal table, after CTRL-T (tracked[pick] == the rows' tracked PDMS to 1e-9) and the pick check."""
    miss = [t for t in tokens if t not in tab]
    if miss:
        raise Refused(f"proposal table lacks {len(miss)} tokens (first {miss[0]})")
    d_tr = np.array([tab[t]["tr"]["dac"] for t in tokens], float)
    d_tp = np.array([tab[t]["tp"]["dac"] for t in tokens], float)
    pd = 100.0 * np.array([tab[t]["tr"]["pdms"] for t in tokens], float)
    tpick = np.array([tab[t]["pick"] for t in tokens])
    ctrl = {"pick_equals_dump_pick": bool(np.array_equal(tpick, pick)),
            "tracked_pick_pdms_equals_rows_max_abs": float(np.abs(pd[np.arange(len(tokens)), tpick] - tracked_pdms_x100).max())}
    ctrl["pass"] = bool(ctrl["pick_equals_dump_pick"] and ctrl["tracked_pick_pdms_equals_rows_max_abs"] <= 1e-7)    # x100 scale of the 1e-9 raw bar
    return {"dac_tracked": d_tr, "dac_raw": d_tp, "pdms": pd, "pick": tpick, "ctrl_t": ctrl}


def rawdac_arrays(rows: dict, tokens, c3_tokens, TX, pick, arm: str) -> dict:
    """The C3 subset of one arm's field_rawdac rows, aligned to the read population's token order (c3 tokens only), after CTRL-SIM and
    CTRL-PICK. Refuses loudly on a missing C3 token, a bad shape, or a failed control. TX = the arm's `table_arrays` (its tracked DAC is the
    CTRL-SIM reference); pick = the dump's pick over the read population."""
    idx = np.array([i for i, t in enumerate(tokens) if t in c3_tokens])
    if len(idx) != len(c3_tokens):
        raise Refused(f"M1: {len(c3_tokens) - len(idx)} C3 tokens are absent from the read population")
    c3t = [tokens[i] for i in idx]
    miss = [t for t in c3t if t not in rows]
    if miss:
        raise Refused(f"rawdac rows of {arm} lack {len(miss)} of the {len(c3t)} C3 tokens (first {miss[0]})")
    bad = [t for t in c3t if len(rows[t].get("raw", [])) != 64 or len(rows[t].get("sim", [])) != 64]
    if bad:
        raise Refused(f"rawdac rows of {arm}: {len(bad)} rows without 64 raw/sim flags (first {bad[0]})")
    table = {t: {"tr": {"dac": TX["dac_tracked"][i]}} for t, i in zip(c3t, idx)}
    try:
        cs = RD.ctrl_sim(rows, table, c3t)
        cp = RD.ctrl_pick(rows, {t: int(pick[i]) for t, i in zip(c3t, idx)}, c3t)
    except ValueError as e:
        raise Refused(f"rawdac {arm}: {e}")
    if not (cs["pass"] and cp["pass"]):
        raise Refused(f"rawdac rows of {arm} fail their controls: CTRL-SIM {cs}, CTRL-PICK {cp}")
    return {"raw": np.array([rows[t]["raw"] for t in c3t], float), "sim": np.array([rows[t]["sim"] for t in c3t], float),
            "pick": np.array([rows[t]["pick"] for t in c3t], int), "idx": idx, "ctrl_sim": cs, "ctrl_pick": cp}


def m1_block(tokens, logs, c3_tokens, TF, TC, n_boot=NBOOT, RF=None, RC=None) -> dict:
    """M1: share of the 64 proposals NAVSIM-DAC-compliant, F - C on the C3 tokens, in every form that exists, each labelled:
    `dac_tracked` (NAVSIM's LQR-tracked table), `raw_navsim_dac` (the CARD's form: navsim_dac_violation(mode='raw'), from field_rawdac rows RF/RC),
    `simulated_navsim_dac` (same function, mode='simulated'), `raw_transparent` (the proposal executed as given, table `tp`, transp_score dt 0.5;
    also under its v1 key `dac_raw`). RF/RC = rawdac_arrays() of F and C."""
    idx = np.array([i for i, t in enumerate(tokens) if t in c3_tokens])
    if len(idx) != len(c3_tokens):
        raise Refused(f"M1: {len(c3_tokens) - len(idx)} C3 tokens are absent from the read population")
    lg = [logs[i] for i in idx]
    bt = Boot(lg, n_boot)
    out = {"population": f"C3 tokens, n = {len(idx)}, logs = {bt.L}: failure-enriched, so these are NOT rates", "estimator": "paired log-cluster bootstrap on the C3 subset's logs (own draw matrix, seed 20260927)"}
    for k in ("dac_tracked", "dac_raw"):
        f = 100.0 * (TF[k][idx] == 1.0).mean(1)
        c = 100.0 * (TC[k][idx] == 1.0).mean(1)
        out[k] = {"F": bt.ci(f), "C": bt.ci(c), "F_minus_C_pp": bt.ci(f - c)}
    out["raw_transparent"] = out["dac_raw"]
    out["raw_transparent_mapping"] = "raw_transparent = the proposal executed as given (table `tp`, transp_score transparent states at dt 0.5): the v1 stand-in for the card's form, kept as a labelled second column"
    if (RF is None) != (RC is None):
        raise Refused("M1: rawdac rows must be supplied for both arms or neither")
    if RF is not None:
        for r in (RF, RC):
            if not np.array_equal(r["idx"], idx):
                raise Refused("M1: rawdac arrays are not aligned to the C3 tokens of this population")
        for key, ak in (("raw_navsim_dac", "raw"), ("simulated_navsim_dac", "sim")):
            f = 100.0 * (RF[ak] == 0.0).mean(1)                      # 1.0 = off the drivable area, so compliant = 0.0
            c = 100.0 * (RC[ak] == 0.0).mean(1)
            out[key] = {"F": bt.ci(f), "C": bt.ci(c), "F_minus_C_pp": bt.ci(f - c)}
        out["raw_navsim_dac_mapping"] = ("raw_navsim_dac = refe.navsim_dac.navsim_dac_violation(mode='raw') per proposal (grid navsim8, ego_state and map_api rebuilt from the navtest DB), "
                                         "share compliant of the 64; the card's own form (Q-F2, resolved). simulated_navsim_dac = the same function, mode='simulated'.")
        # M2 cross-check on the C3 subset: the PICK's DAC (x100, compliant = 100) F - C, in each form
        ar = np.arange(len(idx))
        pk_f, pk_c = RF["pick"], RC["pick"]

        def pk(arm_pick, flags):
            return 100.0 * (flags[ar, arm_pick] == 0.0)
        cc = {"population": out["population"], "note": "CROSS-CHECK on the C3 subset only (failure-enriched, not a rate); M2 itself is the whole population on the transparent scorer, unchanged",
              "dDAC_pick_raw_navsim_dac": bt.ci(pk(pk_f, RF["raw"]) - pk(pk_c, RC["raw"])),
              "dDAC_pick_simulated_navsim_dac": bt.ci(pk(pk_f, RF["sim"]) - pk(pk_c, RC["sim"])),
              "dDAC_pick_tracked_table": bt.ci(100.0 * (TF["dac_tracked"][idx, pk_f] == 1.0) - 100.0 * (TC["dac_tracked"][idx, pk_c] == 1.0)),
              "dDAC_pick_raw_transparent": bt.ci(100.0 * (TF["dac_raw"][idx, pk_f] == 1.0) - 100.0 * (TC["dac_raw"][idx, pk_c] == 1.0))}
        out["M2_c3_pick_crosscheck"] = cc
        out["rawdac_controls"] = {"F": {"CTRL-SIM": RF["ctrl_sim"], "CTRL-PICK": RF["ctrl_pick"]}, "C": {"CTRL-SIM": RC["ctrl_sim"], "CTRL-PICK": RC["ctrl_pick"]}}
    return out


def load_rawdac(globs, dump_path: str, arm: str) -> dict:
    """rows of one arm's field_rawdac files (globs), bound to the arm's proposals dump: every file's `.header.s*.json` must exist, be a non-RED run,
    and carry the dump's sha256."""
    files = [p for g in globs for p in sorted(glob.glob(g))]
    if not files:
        raise Refused(f"rawdac {arm}: no files match {globs}")
    want = sha256_file(dump_path)
    for fp in files:
        hs = sorted(glob.glob(glob.escape(fp) + ".header.s*.json"))
        if not hs:
            raise Refused(f"rawdac {arm}: {fp} has no .header.s*.json (the seam sha256 binding)")
        for h in hs:
            hd = json.load(open(h, encoding="utf-8"))
            if hd.get("red"):
                raise Refused(f"rawdac {arm}: {h} is a RED (rolled) run")
            if hd.get("proposals_sha256") != want:
                raise Refused(f"rawdac {arm}: {h} proposals sha256 {str(hd.get('proposals_sha256'))[:12]} != the dump's {want[:12]}")
    return RD.load_rows(files)


def selection_block(TF, TC, n_boot, logs) -> dict:
    """TACTICAL selection quality from the 64-proposal tables: actual (picked) / random (mean of 64) / best-of-64 tracked PDMS, F - C paired."""
    bt = Boot(logs, n_boot)
    n = len(logs)
    ar = np.arange(n)

    def three(T):
        pd = T["pdms"]
        return {"actual": pd[ar, T["pick"]], "random": pd.mean(1), "best_of_64": pd.max(1)}
    f, c = three(TF), three(TC)
    return {"definition": "tracked PDMS x100 from the 64-proposal table; random = mean over the 64 proposals (mandatory random control); never pooled with another family",
            "levels": {"F": {k: bt.ci(v) for k, v in f.items()}, "C": {k: bt.ci(v) for k, v in c.items()}},
            "F_minus_C": {k: bt.ci(f[k] - c[k]) for k in f}, "actual_minus_random": {"F": bt.ci(f["actual"] - f["random"]), "C": bt.ci(c["actual"] - c["random"])}}


def relabel_tactical(t: dict) -> dict:
    """p5c_tactical.py keys its arms C1 / C0 / base; this read's arms are F / C. Rename, and drop its STRATEGIC stub (STRATEGIC is reported in full)."""
    m = {"C1": "F", "C0": "C"}

    def rn(x):
        if isinstance(x, dict):
            return {m.get(k, k).replace("C1-", "F-").replace("C0-", "C-"): rn(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rn(v) for v in x]
        return x
    r = rn({k: v for k, v in t.items() if k != "strategic"})
    ca = r.get("command_adherence") or {}
    r["labelling"] = "arm names relabelled by field_read.relabel_tactical: C1 -> F, C0 -> C (p5c_tactical.py hard-codes C1/C0)"
    r["pooled"] = False
    if "paired_C1_minus_x" in ca:
        ca["paired_F_minus_x"] = ca.pop("paired_C1_minus_x")
    return r


def strategic_block(j, n) -> dict:
    if j is None:
        return {"status": "NOT SUPPLIED", "reason": "no strategic_family.py JSON (F vs C) supplied", "n": n}
    txt = json.dumps(j)
    if "PAUSED" in txt:
        raise Refused("strategic_family JSON contains PAUSED: STRATEGIC is reported in full since the PI's 2026-11-16 ruling")
    return {"status": "REPORTED IN FULL (not gating; PI ruling 2026-11-16)", "instrument": j.get("instrument"), "arms": j.get("arms"),
            "paired_vs_base": j.get("paired_vs_base"), "gates": j.get("gates"), "controls": j.get("controls"), "estimator": j.get("estimator"), "tier": j.get("tier"),
            "evidence_class": j.get("evidence_class")}


def families_block(fam_json, tact, strat, sel, fam_guard) -> dict:
    out = {"pooled": False, "rule": "each family stands alone; no composite"}
    if fam_json is None:
        for k in ("LONGITUDINAL", "LATERAL"):
            out[k] = {"status": "NOT SUPPLIED", "reason": "no fam_ci JSON (F vs C)"}
        out["TACTICAL_fam_ci_components"] = {"status": "NOT SUPPLIED", "reason": "no fam_ci JSON (F vs C)"}
    else:
        fr = fam_json["frames"]["executed_tracked"]["families"]
        for fam, key in (("LONGITUDINAL", "longitudinal"), ("LATERAL", "lateral"), ("TACTICAL_fam_ci_components", "tactical")):
            comp = {}
            for k, e in (fr.get(key) or {}).items():
                if not isinstance(e, dict) or "status" not in e:
                    continue
                comp[k] = {"status": e["status"], "n_tokens": e.get("n_tokens"), "F": e.get("B"), "C": e.get("A"), "diff_F_minus_C": e.get("diff_B_minus_A"),
                           "adverse": (entry_adverse_pol(k, e) if key != "tactical" else None)}
            out[fam] = comp
        out["frame"] = "executed_tracked (fam_ci diff = B - A = F - C)"
    out["TACTICAL_command_adherence"] = tact if tact is not None else {"status": "NOT SUPPLIED", "reason": "no p5c_tactical.py JSON for F vs C"}
    out["TACTICAL_selection_quality"] = sel if sel is not None else {"status": "NOT COMPUTED", "reason": "no 64-proposal tables supplied"}
    out["STRATEGIC"] = strat
    out["guard_longitudinal_lateral"] = fam_guard
    return out


# ------------------------------------------------------------------------------------------------------ the read
def run_read(a) -> dict:
    arms = [x for x in (a.f, a.c, a.fs, a.cs) if x]
    uniq = list(dict.fromkeys(arms))
    paths = [p for g in a.rows for p in sorted(glob.glob(g))]
    rows, errs = load_slim(paths, uniq)
    if errs:
        raise Refused(f"{len(errs)} tokens have only error rows (first {errs[0]})")
    P = PR.per_token(rows, uniq)
    tokens, logs = P["tokens"], P["logs"]
    dry = bool(a.dry_run)
    if not dry:
        need = {"--csv-f": a.csv_f, "--csv-c": a.csv_c, "--seam-ledger": a.seam_ledger, "--fam-ci": a.fam_ci, "--strategic": a.strategic, "--tactical": a.tactical,
                "--c3-tokens": a.c3_tokens, "--proptable-f": a.proptable_f, "--proptable-c": a.proptable_c,
                "--rawdac-f": a.rawdac_f, "--rawdac-c": a.rawdac_c}
        miss = [k for k, v in need.items() if not v]
        if miss:
            raise Refused("a real read refuses without: " + ", ".join(miss))
    geo = {}
    for nm, sp, dp in ((a.f, a.seam_f, a.dump_f), (a.c, a.seam_c, a.dump_c), (a.fs, a.seam_fs, a.dump_fs), (a.cs, a.seam_cs, a.dump_cs)):
        if nm and nm not in geo:
            if not (sp and dp):
                raise Refused(f"arm {nm}: --seam-/--dump- paths are required")
            geo[nm] = seam_geometry(sp, dp, tokens)
    toks_by = {f"rows:{x}": tokens for x in uniq}
    for nm, g in geo.items():
        toks_by[f"dump_ordered:{nm}"] = [t for t in tokens if t in set(g["dump_tokens"])]
    seam_sets = {f"seam:{nm}": g["seam_tokens"] for nm, g in geo.items()}
    set_ctl = ctrl_set(toks_by, None if dry else N_NAVTEST, logs, None if dry else L_NAVTEST)
    for k, st in seam_sets.items():        # the seam must hold exactly the read population (not merely contain it) in a real read
        if (not dry) and set(st) != set(tokens):
            set_ctl["pass"] = False
            set_ctl["population_problems"].append(f"{k}: seam token set != rows token set")
    csv_ctl = []
    for nm, cp in ((a.f, a.csv_f), (a.c, a.csv_c), (a.fs, a.csv_fs), (a.cs, a.csv_cs)):
        if nm and cp:
            csv_ctl.append(dict(ctrl_csv(rows, nm, cp, tokens), arm_name=nm))
        elif nm and not dry:
            raise Refused(f"arm {nm}: --csv is required in a real read")
    seam_ctl = ctrl_seam(geo, a.seam_ledger, a.convert_report)
    idc = ctrl_id(P, logs, a.f, geo, a.n_boot)
    idc2 = ctrl_id(P, logs, a.c, geo, a.n_boot) if a.c != a.f else None
    controls = [set_ctl] + csv_ctl + [seam_ctl, idc] + ([idc2] if idc2 else [])
    missing_ctrl = []
    if not csv_ctl:
        missing_ctrl.append("CTRL-CSV not supplied")
    if not a.seam_ledger:
        missing_ctrl.append("CTRL-SEAM ledger not supplied (only the in-read re-hash ran)")
    controls_ok = (False if any(not c["pass"] for c in controls) else (None if (missing_ctrl and not dry) else True))
    c = core(P, logs, a.f, a.c, geo, a.n_boot)
    bt = c.pop("_boot")
    self_read = a.f == a.c
    fam_j = json.load(open(a.fam_ci, encoding="utf-8")) if a.fam_ci else None
    guard = family_guard(fam_j, *(a.fam_labels or ("C", "F")))
    rep = None
    rep_note = "no replicate pair supplied (--fs and --cs): the seed floor is unmeasured"
    if a.fs and a.cs:
        fs_cs_e1 = bt.ci(P[a.fs]["tracked"]["pdms"] - P[a.cs]["tracked"]["pdms"])
        fs_cs_dz = bt.ci(zero_share(P, a.fs, "tracked", "dac") - zero_share(P, a.cs, "tracked", "dac"))
        floor = abs(float(np.mean(P[a.cs]["tracked"]["pdms"]) - np.mean(P[a.c]["tracked"]["pdms"])))
        rep = {"fs_minus_cs_e1": fs_cs_e1, "fs_minus_cs_dz": fs_cs_dz, "floor": floor}
        rep_note = "replicate pair supplied"
    elif a.fs or a.cs:
        raise Refused("--fs and --cs must be supplied together")
    v = decide(c, guard, controls_ok, rep, dry)
    if self_read and (v["verdict"] != "NOT PROVEN" or not idc["pass"]):
        raise Refused(f"self-read (F == C == {a.f}) did not give NOT PROVEN with every Delta exactly 0.0: {v['verdict']}, {idc}")
    # M1 / selection quality
    m1 = sel = None
    m1_why = None
    if a.proptable_f and a.proptable_c:
        TF = table_arrays(load_table([p for g in a.proptable_f for p in sorted(glob.glob(g))]), tokens, P[a.f]["tracked"]["pdms"],
                          np.asarray(_pick(geo[a.f], tokens)))
        TC = table_arrays(load_table([p for g in a.proptable_c for p in sorted(glob.glob(g))]), tokens, P[a.c]["tracked"]["pdms"],
                          np.asarray(_pick(geo[a.c], tokens)))
        for nm, T in (("F", TF), ("C", TC)):
            if not T["ctrl_t"]["pass"]:
                raise Refused(f"proposal table of {nm} fails CTRL-T: {T['ctrl_t']}")
        sel = selection_block(TF, TC, a.n_boot, logs)
        if a.c3_tokens:
            c3 = set(json.load(open(a.c3_tokens, encoding="utf-8"))["tokens"])
            RF = RC = None
            if a.rawdac_f or a.rawdac_c:
                if not (a.rawdac_f and a.rawdac_c):
                    raise Refused("--rawdac-f and --rawdac-c must be supplied together")
                RF = rawdac_arrays(load_rawdac(a.rawdac_f, a.dump_f, "F"), tokens, c3, TF, np.asarray(_pick(geo[a.f], tokens)), "F")
                RC = rawdac_arrays(load_rawdac(a.rawdac_c, a.dump_c, "C"), tokens, c3, TC, np.asarray(_pick(geo[a.c], tokens)), "C")
            m1 = m1_block(tokens, logs, c3, TF, TC, a.n_boot, RF, RC)
            m1["c3_tokens_sha256"] = sha256_file(a.c3_tokens)
            if RF is None:
                m1["raw_navsim_dac"] = {"status": "NOT COMPUTED", "reason": "no --rawdac-f/--rawdac-c (field_rawdac.py rows); only the transparent stand-in `raw_transparent` exists"}
        else:
            m1_why = "no --c3-tokens"
        m1_ctrl = {"F": TF["ctrl_t"], "C": TC["ctrl_t"]}
    else:
        m1_why = "no 64-proposal tables (--proptable-f/--proptable-c): M1's per-proposal DAC and the selection quality need them"
        m1_ctrl = None
    tact = relabel_tactical(json.load(open(a.tactical, encoding="utf-8"))) if a.tactical else None
    strat = strategic_block(json.load(open(a.strategic, encoding="utf-8")) if a.strategic else None, len(tokens))
    fams = families_block(fam_j, tact, strat, sel, guard)
    res = {"instrument": "field_read.py", "card": "PREREG_AUX_FIELD_LOSS_P7.md (FINAL 2026-11-17, Amendments A, B)",
           "evidence": "MEASURED on the rows supplied; " + TIER, "tier": TIER,
           "estimator": {"kind": "paired log-cluster bootstrap", "draws": a.n_boot, "seed": SEED, "interval": "95 % percentile",
                         "statistic": "ratio of sums over drawn logs (rates, shares and F/C ratios too)", "shared": "ONE draw matrix for every arm and contrast (the p5c_read / fam_ci draws)",
                         "answers": ANSWERS},
           "n_tokens": len(tokens), "n_logs": bt.L, "units": "PDMS, sub-scores x 100; shares in percent; ratios unitless",
           "arms": {"F": a.f, "C": a.c, "Fs": a.fs, "Cs": a.cs}, "self_read": self_read,
           "controls": controls, "controls_all_pass": controls_ok, "controls_missing": missing_ctrl,
           "endpoints": {k: c[k] for k in ("E1", "E2", "E3", "M2")},
           "six_subscores": c["six_subscores_tracked_and_transparent"],
           "adverse": {k: c[k] for k in ("A1", "A2", "A3")}, "K4": c["K4"],
           "M1": m1 if m1 is not None else {"status": "NOT COMPUTED", "reason": m1_why, "n": len(tokens)}, "M1_table_ctrl_t": m1_ctrl,
           "replicate": {"note": rep_note, **({k: v for k, v in rep.items()} if rep else {})},
           "families": fams, "verdict": v}
    res["card_questions"] = [
        "Q-F1 A1-A3 bars are the card's proposed thresholds (EP delta <= -1.0 x100 with hi < 0; F/C upper bounds 0.97 and 0.90); a ratio is mean F / mean C over the same tokens, bootstrapped on the shared log draws.",
        "Q-F2 M1's `raw` is now computed in the card's own form, `raw_navsim_dac` = refe.navsim_dac.navsim_dac_violation(mode='raw') per proposal (eval/field/field_rawdac.py rows, CTRL-SIM + CTRL-PICK enforced here), alongside `simulated_navsim_dac` (mode='simulated'). The earlier stand-in, the proposal executed as given on the transparent NAVSIM scorer (table `tp`, dt 0.5), stays as the labelled second column `raw_transparent`; the two are different instruments and are never merged. M2 is unchanged (whole population, transparent); its C3-subset mode='raw' pick cross-check is `M1.M2_c3_pick_crosscheck`.",
        "Q-F3 DAC-zero share and zeroed-token share are the share of tokens whose tracked DAC / PDMS equals exactly 0; the 'cut' for the LARGE label is relative to C's share.",
        "Q-F4 E1 and E2 hold, A1-A3 clear, but a longitudinal/lateral family component is separated adverse: the card forbids ADOPT but names no TRADE; this read prints NOT PROVEN with the component named.",
        "Q-F5 the replicate pair must be given as arms in the same rows; floor = |mean PDMS(Cs) - mean PDMS(C)| as the card writes it (a point, not an interval).",
        "Q-F6 CTRL-CSV bar is 1e-9 on the raw per-token values (all 7 metrics); the P5-C read used 1e-6.",
        "Q-F7 POLARITY (v3, 2026-11-20): min_ttc_s / headway_min_m / time_gap_min_s are higher-is-safer; the inherited fam_direction defaulted them to lower-is-better (a rise read adverse). The correction is applied in this file (entry_adverse_pol), not only in p5c_read, and every cell it changes is listed under families.guard_longitudinal_lateral.polarity_corrections with both readings.",
    ]
    if dry:
        res["STAMP"] = DRY_STAMP
        res["DRY_RUN"] = {"label": a.dry_run, "claim": "none: the arms are stand-ins, the verdict string is a pipeline exercise",
                          "verdict_as_computed": v["verdict"]}
        res["verdict"] = dict(v, verdict="DRY RUN:" + v["verdict"])
    return res


def _pick(g, tokens):
    d = np.load(g["dump"], allow_pickle=False)
    di = {str(t): i for i, t in enumerate(d["token"])}
    return np.asarray(d["pick"])[np.array([di[t] for t in tokens])]


def ci_s(x):
    return "%.4f [%.4f, %.4f]" % (x["point"], x["lo"], x["hi"])


def text_report(r: dict) -> str:
    L = []
    if "STAMP" in r:
        L += [r["STAMP"], "(" + r["DRY_RUN"]["label"] + ")", ""]
    v = r["verdict"]
    L.append(f"VERDICT: {v['verdict']} | {v['wording']}")
    L.append(f"tier: {r['tier']}")
    L.append(f"estimator: {r['estimator']['kind']}, {r['estimator']['draws']} draws, seed {r['estimator']['seed']}; answers: {r['estimator']['answers']}")
    L.append(f"n tokens {r['n_tokens']}, logs {r['n_logs']}; arms {r['arms']}")
    L.append("controls: " + ", ".join(f"{c['control']}={'PASS' if c['pass'] else 'FAIL'}" for c in r["controls"]) + f"  (all pass: {r['controls_all_pass']})")
    e = r["endpoints"]
    L.append("E1 PDMS F-C (tracked, decision): " + ci_s(e["E1"]["tracked"]))
    L.append("E2 DAC F-C: " + ci_s(e["E2"]["dac"]) + " | DAC-zero share F-C pp: " + ci_s(e["E2"]["dac_zero_share_diff_pp"]))
    L.append("E3 zeroed-token share F-C pp: " + ci_s(e["E3"]["diff_pp"]))
    for m, d in r["six_subscores"].items():
        L.append(f"  sub {m}: tracked {ci_s(d['tracked'])} | transparent ({PR.NO_TRACKER}) {ci_s(d['transparent'])}")
    for k in ("A1", "A2", "A3"):
        a = r["adverse"][k]
        L.append(f"{k} {a['quantity']}: {ci_s(a['ci'])} bar {a['bar']} -> {'ADVERSE' if a['adverse'] else 'clear'}")
    L.append(f"K4: {'FIRED' if r['K4']['fired'] else 'not fired'}")
    m1 = r["M1"]
    L.append("M1: " + (f"{m1['population']}; DAC-compliant share of 64, F-C pp: tracked {ci_s(m1['dac_tracked']['F_minus_C_pp'])}" if "dac_raw" in m1 else f"NOT COMPUTED: {m1.get('reason')}"))
    if "dac_raw" in m1:
        if "F_minus_C_pp" in m1.get("raw_navsim_dac", {}):
            L.append(f"  raw, navsim_dac_violation(mode='raw') [the card's form]: {ci_s(m1['raw_navsim_dac']['F_minus_C_pp'])}; mode='simulated': {ci_s(m1['simulated_navsim_dac']['F_minus_C_pp'])}")
        else:
            L.append("  raw, navsim_dac_violation(mode='raw') [the card's form]: NOT COMPUTED (no rawdac rows)")
        L.append(f"  raw, transparent NAVSIM scorer, dt 0.5 [second column]: {ci_s(m1['raw_transparent']['F_minus_C_pp'])}")
        cc = m1.get("M2_c3_pick_crosscheck")
        if cc:
            L.append(f"  M2 cross-check, C3 subset, dDAC of the PICK (x100): raw_navsim_dac {ci_s(cc['dDAC_pick_raw_navsim_dac'])}, simulated {ci_s(cc['dDAC_pick_simulated_navsim_dac'])}, tracked table {ci_s(cc['dDAC_pick_tracked_table'])}, raw_transparent {ci_s(cc['dDAC_pick_raw_transparent'])}")
        rc = m1.get("rawdac_controls")
        if rc:
            for k in ("F", "C"):
                s_ = rc[k]["CTRL-SIM"]
                L.append(f"  rawdac CTRL-SIM {k}: n={s_['n_tokens']} tokens, per-proposal {100 * s_['per_proposal_agreement']:.2f} %, per-token {100 * s_['per_token_agreement']:.2f} %; CTRL-PICK {k}: {rc[k]['CTRL-PICK']['n_mismatch']} mismatches")
    m2 = e["M2"]
    L.append(f"M2: dDAC tracked {ci_s(m2['dDAC_tracked'])}, raw {ci_s(m2['dDAC_raw_transparent'])} -> {m2['routing']}")
    f = r["families"]
    L.append("families (each stands alone, never pooled):")
    for k in ("LONGITUDINAL", "LATERAL"):
        b = f[k]
        if "status" in b:
            L.append(f"  {k}: {b['status']} ({b['reason']})")
        else:
            adv = [n for n, e in b.items() if e.get("adverse")]
            L.append(f"  {k}: {len(b)} components, separated adverse (F vs C, tracked frame): {adv if adv else 'none'}")
    ca = f["TACTICAL_command_adherence"]
    sq = f["TACTICAL_selection_quality"]
    L.append(f"  TACTICAL: command adherence {ca.get('command_adherence', {}).get('status', ca.get('status'))}; selection quality "
             + (("actual F-C " + ci_s(sq["F_minus_C"]["actual"]) + ", random F-C " + ci_s(sq["F_minus_C"]["random"])) if "F_minus_C" in sq else "NOT COMPUTED"))
    L.append("  STRATEGIC: " + str(f["STRATEGIC"].get("status")))
    g = f["guard_longitudinal_lateral"]
    L.append("  guard (longitudinal/lateral separated adverse): " + g["status"] + (f" ({g['reason']})" if g.get("reason") else ""))
    for pc in g.get("polarity_corrections") or []:
        d = pc["diff_F_minus_C"]
        L.append(f"  POLARITY CORRECTED {pc['family']}.{pc['component']}: diff F-C {d['point']:+.4g} [{d['lo']:+.4g}, {d['hi']:+.4g}]; inherited rule ({pc['inherited_reading']['direction']}) -> "
                 f"adverse={pc['inherited_reading']['adverse']}; corrected ({pc['corrected_reading']['direction']}) -> adverse={pc['corrected_reading']['adverse']}; {pc['reason']}")
    gi = g.get("guard_status_under_inherited_rule")
    if gi:
        L.append(f"  guard under the inherited rule: {gi['status']} (flipped by the correction: {gi['flipped_by_correction']})")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--f", required=True, help="arm name of F in the rows")
    ap.add_argument("--c", required=True, help="arm name of C in the rows")
    ap.add_argument("--fs", default=None)
    ap.add_argument("--cs", default=None)
    for k in ("f", "c", "fs", "cs"):
        ap.add_argument(f"--seam-{k}", default=None)
        ap.add_argument(f"--dump-{k}", default=None)
        ap.add_argument(f"--csv-{k}", default=None)
    ap.add_argument("--seam-ledger", default=None, help="JSON of seam/dump sha256 per arm: written on first sight, re-checked after (CTRL-SEAM)")
    ap.add_argument("--convert-report", default=None, help="p5c_read_prep convert_report.json: its seam_sha256 must equal the seam's")
    ap.add_argument("--proptable-f", nargs="*", default=None, help="64-proposal table jsonl glob(s) of F (p5c_prop_score.py)")
    ap.add_argument("--proptable-c", nargs="*", default=None)
    ap.add_argument("--rawdac-f", nargs="*", default=None, help="field_rawdac.py rows (jsonl glob(s)) of F on the C3 tokens: M1's card form navsim_dac_violation(mode='raw')")
    ap.add_argument("--rawdac-c", nargs="*", default=None, help="same for C")
    ap.add_argument("--c3-tokens", default=None, help="PK/raw/2026-10-11-c3-selection-headroom/tokens.json")
    ap.add_argument("--fam-ci", default=None, help="fam_ci.py JSON with --a C --b F")
    ap.add_argument("--fam-labels", nargs=2, default=None, metavar=("A", "B"), help="labels fam_ci carries for the two arms (default C F); a dry run on stand-ins passes C0 C1")
    ap.add_argument("--tactical", default=None, help="p5c_tactical.py JSON run with --seam C1=<F seam> C0=<C seam> base=<P7 seam>")
    ap.add_argument("--strategic", default=None, help="strategic_family.py read JSON (--arm F=... --base C=...)")
    ap.add_argument("--dry-run", default=None, help="label; stamps every output DRY RUN - NOT A FIELD VERDICT")
    ap.add_argument("--n-boot", type=int, default=NBOOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        res = run_read(a)
    except Refused as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
    txt = text_report(res)
    if a.text:
        with open(a.text, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    e1 = res["endpoints"]["E1"]["tracked"]
    print(("ZZFIELD_READ " + (res["STAMP"] + " " if "STAMP" in res else "")) + res["verdict"]["verdict"], "n", res["n_tokens"], "logs", res["n_logs"],
          "E1 %.4f [%.4f, %.4f]" % (e1["point"], e1["lo"], e1["hi"]), "controls", res["controls_all_pass"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
