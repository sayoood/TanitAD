#!/usr/bin/env python3
"""P5-C read instrument B7: per-token arrays -> shared-draw paired log-cluster bootstrap -> the card's verdict table and qualifier.

Implements PREREG_P5C.md (frozen 2026-11-12, sha256 in PREREG_P5C.sha256) section 5 (estimator), 6 (bars, verdicts, control, replicate),
7 (qualifier), 8.1 (sub-scores, zero-pick rates, mechanism counts) and 6.5 (deploy clause, goal). It decides nothing the card did not decide;
card questions are listed under `card_questions` in the output.

INPUT  `transp_score.py` rows (--rows, jsonl, one row per token: arms.<name>.{tracked,transparent}.{nc,dac,ddc,ttc,ep,c,pdms}).
       Every arm must carry every token, no error rows: otherwise REFUSED (exit 2). Scores are printed x 100.
ESTIMATOR  paired log-cluster bootstrap over the logs present, 10,000 draws, seed 20260927, 95 % percentile; ONE draw matrix shared by every
       arm, every scorer and every contrast (the SAME draws `fam_ci.py` uses: rng.integers(0, L, (n_boot, L)) over the sorted unique logs).
       statistic = sum of the drawn logs' per-token sums / sum of their token counts (the ratio estimator; rates use it too).
       It answers "would another draw of LOGS say this?" only. Training-run variance: the floors and C0r (card 6.3). Inference: asserted.
SCORERS Every number is computed twice, tracked (NAVSIM LQR) and transparent (same pick, no tracker), from the same tokens and draws.
       TD = d - d_T is computed PER DRAW (paired), never as a difference of two intervals.
Run with --rehearsal "<text>" to run on non-P5-C data: the verdict is then printed as REHEARSAL:<verdict> with no P5-C claim.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys

import numpy as np

SEED, NBOOT = 20260927, 10000
# card section 6.1 inputs: B = max(t * s_p * sqrt(2), 2F) = max(2.776 * 0.1791 * 1.4142, 2 * 0.33) = 0.7031 (arithmetic on card inputs M)
T_CRIT, S_P, F_SEED = 2.776, 0.1791, 0.33
B_BAR = round(max(T_CRIT * S_P * math.sqrt(2.0), 2 * F_SEED), 4)
F3_2X = 0.759                  # 2*F3, card 6.1: reported beside B, decides nothing
GOAL_LEVEL = 94.55             # DriveZero Table A13 [P] (card 6.5)
SUBS = ("nc", "dac", "ddc", "ttc", "ep", "c")
G1_SUBS = ("nc", "dac", "ttc", "ep", "c")        # card 6.2 G1 list: NC, DAC, TTC, EP, C (DDC is reported, not a G1 term)
METRICS = SUBS + ("pdms",)
SCORERS = ("tracked", "transparent")
STRATEGIC = {"status": "PAUSED", "reason": "PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25); not a pass, not omitted"}
NO_TRACKER = "no tracker in this number"


class Refused(Exception):
    pass


# ------------------------------------------------------------------------------------------------------ data
def load_rows(paths):
    rows, errs = {}, []
    for fp in paths:
        for ln in open(fp, encoding="utf-8"):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "error" in r:
                errs.append(r.get("token"))
            else:
                rows[r["token"]] = r
    bad = [t for t in errs if t not in rows]
    return rows, bad


def per_token(rows: dict, arms) -> dict:
    """-> {"tokens", "logs", arm: {scorer: {metric: array x100}}}. Refuses on a missing arm / token / non-finite value."""
    toks = sorted(rows)
    out = {"tokens": toks, "logs": [rows[t]["log"] for t in toks]}
    for a in arms:
        out[a] = {s: {m: np.empty(len(toks)) for m in METRICS} for s in SCORERS}
        for i, t in enumerate(toks):
            arm = rows[t].get("arms", {}).get(a)
            if arm is None:
                raise Refused(f"arm {a} missing on token {t}")
            for s in SCORERS:
                for m in METRICS:
                    v = arm[s][m]
                    if v is None or not math.isfinite(float(v)):
                        raise Refused(f"non-finite {a}.{s}.{m} on token {t}")
                    out[a][s][m][i] = 100.0 * float(v)
    return out


class Boot:
    """ONE draw matrix, shared. The same construction as fam_ci.cluster_boot / _draws, so family CIs and these CIs share draws."""

    def __init__(self, logs, n_boot: int = NBOOT, seed: int = SEED):
        ul = sorted(set(logs))
        pos = {l: i for i, l in enumerate(ul)}
        self.li = np.array([pos[l] for l in logs])
        self.L = len(ul)
        self.n = np.bincount(self.li, minlength=self.L).astype(float)
        self.D = np.random.default_rng(seed).integers(0, self.L, (n_boot, self.L))
        self.nD = self.n[self.D].sum(1)
        self.n_boot, self.seed = n_boot, seed

    def draws(self, v) -> np.ndarray:
        s = np.bincount(self.li, weights=np.asarray(v, float), minlength=self.L)
        return s[self.D].sum(1) / self.nD

    def ci(self, v) -> dict:
        m = self.draws(v)
        return {"point": float(np.mean(v)), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def ci_pair(bt: Boot, vt, vp) -> dict:
    """number printed twice: tracked, transparent, and TD = tracked - transparent on the SAME draws (paired per draw)."""
    mt, mp = bt.draws(vt), bt.draws(vp)
    td = mt - mp
    f = lambda m, pt: {"point": float(pt), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}   # noqa: E731
    return {"tracked": f(mt, np.mean(vt)), "transparent": f(mp, np.mean(vp)), "TD": f(td, np.mean(vt) - np.mean(vp))}


def level(bt, P, arm, metric):
    return ci_pair(bt, P[arm]["tracked"][metric], P[arm]["transparent"][metric])


def contrast(bt, P, x, y):
    """x - y, tracked + transparent + TD, for pdms and all six sub-scores."""
    return {m: ci_pair(bt, P[x]["tracked"][m] - P[y]["tracked"][m], P[x]["transparent"][m] - P[y]["transparent"][m]) for m in METRICS}


def rate_arrays(P, arm):
    """pick rates and mechanism indicators per token (ratio estimator = mean of the 0/1 indicator on the drawn tokens)."""
    t, p = P[arm]["tracked"], P[arm]["transparent"]
    ind = {}
    for nm, k in (("dac_zero", "dac"), ("pdms_zero", "pdms"), ("nc_zero", "nc")):
        ind[nm] = (t[k] == 0).astype(float), (p[k] == 0).astype(float)
    made = ((t["dac"] == 0) & (p["dac"] == 100.0)).astype(float)      # tracker-made: tracked 0, transparent 1
    saved = ((t["dac"] == 100.0) & (p["dac"] == 0)).astype(float)     # tracker-saved: tracked 1, transparent 0
    return ind, made, saved


def rates_block(bt, P, arms, c1, c0, base):
    out = {"definition": "pick rate = share of tokens whose pick's own component equals 0 (ratio estimator), printed x 100 (percent)", "arms": {}, "contrasts": {}}
    ind = {a: rate_arrays(P, a) for a in arms}
    for a in arms:
        i, made, saved = ind[a]
        blk = {k: ci_pair(bt, 100 * v[0], 100 * v[1]) for k, v in i.items()}
        blk["tracker_made_dac_zero"] = {"mechanism": True, "ci": bt.ci(100 * made), "definition": "tracked DAC 0 and transparent DAC 1, % of tokens (card 8.1); both scorers are inside the definition"}
        blk["tracker_saved_dac_zero"] = {"mechanism": True, "ci": bt.ci(100 * saved), "definition": "tracked DAC 1 and transparent DAC 0, % of tokens (card 8.1); both scorers are inside the definition"}
        out["arms"][a] = blk
    for x, y in ((c1, c0), (c0, base), (c1, base)):
        ix, mx, sx = ind[x]
        iy, my, sy = ind[y]
        blk = {k: ci_pair(bt, 100 * (ix[k][0] - iy[k][0]), 100 * (ix[k][1] - iy[k][1])) for k in ix}
        blk["tracker_made_dac_zero"] = {"mechanism": True, "ci": bt.ci(100 * (mx - my)), "definition": "difference of the tracker-made rate, pp; both scorers are inside the definition"}
        blk["tracker_saved_dac_zero"] = {"mechanism": True, "ci": bt.ci(100 * (sx - sy)), "definition": "difference of the tracker-saved rate, pp; both scorers are inside the definition"}
        out["contrasts"][f"{x}-{y}"] = blk
    return out


# ------------------------------------------------------------------------------------------------------ G2 (families, tracked-executed frame only)
def fam_direction(metric: str) -> str:
    """VENDORED from eval/proxy_eval.py::fam_direction (the polarity rule behind P6/P7's ADOPT guard; 2026-09-28 word-match fix included)."""
    m = metric.lower()
    if "within_" in m or "accuracy" in m or "kappa" in m:
        return "higher"
    if "bias" in m:
        return "magnitude"
    if "progress_ratio" in m:
        return "closer_to_1"
    return "lower"


def entry_adverse(name: str, e: dict) -> bool | None:
    """is C1 (= B in fam_ci) separated WORSE than C0 (= A) on this component? None when not decidable. Directional metrics use the PAIRED
    difference B-A; bias / ratio metrics use P6's unpaired interval rule on the two arms' own intervals (conservative), as proxy_eval does."""
    if e.get("status") != "OK":
        return None
    d = e["diff_B_minus_A"]
    k = fam_direction(name)      # the FULL dotted name, as proxy_eval does: "target_speed_acc.within_1.0_mps" must not be split at its decimal point
    if k == "higher":
        return bool(d["hi"] < 0)
    if k == "lower":
        return bool(d["lo"] > 0)
    A, Bv = e.get("A"), e.get("B")
    if not (A and Bv):
        return None
    wl, wh, pl, ph = A["lo"], A["hi"], Bv["lo"], Bv["hi"]
    c = 0.0 if k == "magnitude" else 1.0
    if k == "magnitude":
        return bool((pl > wh >= 0 and wl >= 0) or (ph < wl <= 0 and wh <= 0))
    return bool((pl > wh >= c and wl >= c) or (ph < wl <= c and wh <= c))


def g2_from_famci(j: dict | None) -> dict:
    """G2: no LONGITUDINAL or LATERAL component of the TRACKED-EXECUTED frame is separated adverse. NOT-EVALUATED without the fam_ci JSON."""
    if j is None:
        return {"status": "NOT-EVALUATED", "reason": "no fam_ci JSON (C1 vs C0) supplied", "adverse": []}
    fr = (j.get("frames") or {}).get("executed_tracked")
    if not fr:
        return {"status": "NOT-EVALUATED", "reason": "fam_ci JSON has no executed_tracked frame", "adverse": []}
    adverse, compared, undecided = [], 0, []
    for fam in ("longitudinal", "lateral"):
        for k, e in (fr["families"].get(fam) or {}).items():
            if k.split(".")[-1] in ("gt_progress_mean_m", "t0_axis_gt_self_ratio"):
                continue
            r = entry_adverse(k, e)
            if r is None:
                undecided.append(f"{fam}.{k}:{e.get('status')}")
                continue
            compared += 1
            if r:
                adverse.append({"family": fam, "component": k, "direction": fam_direction(k),
                                "diff_C1_minus_C0": e["diff_B_minus_A"]})
    if compared == 0 and not adverse:     # an evaluation over zero components is not an evaluation (found 2026-11-13: a RAISED families block gave an empty frame and a PASS)
        return {"status": "NOT-EVALUATED", "reason": "fam_ci frame holds no comparable LONGITUDINAL / LATERAL component (0 compared): G2 was not evaluated", "adverse": [],
                "components_compared": 0, "components_undecided": undecided}
    return {"status": "FAIL" if adverse else "PASS", "adverse": adverse, "components_compared": compared, "components_undecided": undecided,
            "frame": "executed_tracked (the transparent frame is never a guard, card 8.2)", "strategic": STRATEGIC,
            "source_estimator": j.get("estimator")}


# ------------------------------------------------------------------------------------------------------ the card's clauses
def qualifier(td: dict, dt: dict) -> dict:
    """card section 7. TM+C is checked before TM (its condition implies TM's): UNVERIFIED reading, listed in card_questions."""
    if td["lo"] > 0 and dt["hi"] < 0:
        q, txt = "TM+C", "the gain exists only through NAVSIM's tracker; C1's plans are worse as paths by d_T [CI]"
    elif td["lo"] > 0 and dt["lo"] <= 0:
        q, txt = "TM", "the gain exists only through NAVSIM's tracker"
    elif dt["lo"] > 0:
        q, txt = "TI", "the plans also score higher as paths on the PDMS formula"
    else:
        q, txt = "TU", "undetermined: printed with both CIs"
    return {"code": q, "text": txt}


def verdict_table(d: dict, B: float, g1_fail: list, g2: dict, ctrl1_valid: bool, controls_ok, p1c: dict | None, missing_kinds=()) -> dict:
    """card 6.2. d = tracked C1 - C0 {point, lo, hi}. Order: INVALID, REFUTED, NULL, SWAMPED, GUARD-FAIL, then the LEVER ladder."""
    p1a, p1b = d["lo"] > 0, d["point"] > B
    base = {"P1-a": bool(p1a), "P1-b": bool(p1b), "G1": "FAIL" if g1_fail else "PASS", "G2": g2["status"], "CTRL-1": bool(ctrl1_valid),
            "controls_all_pass": controls_ok}
    if controls_ok is False or not ctrl1_valid:
        return {"verdict": "INVALID", "wording": "no verdict; C1 - C0 is printed as descriptive only", "clauses": base}
    if d["hi"] < 0:
        return {"verdict": "REFUTED", "wording": "C1's plans survive the tracker worse", "clauses": base}
    if not p1a:
        return {"verdict": "NULL", "wording": "NOT PROVEN", "clauses": base}
    if not p1b:
        return {"verdict": "SWAMPED", "wording": "NOT PROVEN", "clauses": base}
    if g1_fail or g2["status"] == "FAIL":
        return {"verdict": "GUARD-FAIL", "wording": "NOT ADOPTED; the component is named", "named": {"G1": g1_fail, "G2": [a["component"] for a in g2["adverse"]]},
                "clauses": base}
    if g2["status"] != "PASS" or controls_ok is None:
        return {"verdict": "INCOMPLETE", "wording": "a required input is missing (" + ("G2 not evaluated" if g2["status"] != "PASS" else ("required controls missing: " + ", ".join(missing_kinds)) if missing_kinds else "no controls supplied") + "): no verdict",
                "clauses": base}
    if p1c is None:
        return {"verdict": "SEPARATED-SINGLE", "wording": "separated, single seed, not yet a lever claim", "clauses": base}
    base["P1-c"] = p1c["pass"]
    if p1c["pass"]:
        return {"verdict": "LEVER", "wording": "C1's plans survive the benchmark's tracker better than C0's", "clauses": base}
    return {"verdict": "SWAMPED-BY-REPLICATE", "wording": "NOT PROVEN: inside this rig's own run-to-run noise", "clauses": base}


def p1c_clause(bt, P, c1, c0, c0r, d_point, B):
    """card 6.3: d_r = C1 - C0r; F_p5c = |C0r - C0|; pass iff CI-lo(d_r) > 0 AND min(d, d_r) > max(B, 2 F_p5c)."""
    dr = bt.ci(P[c1]["tracked"]["pdms"] - P[c0r]["tracked"]["pdms"])
    f = abs(float(np.mean(P[c0r]["tracked"]["pdms"])) - float(np.mean(P[c0]["tracked"]["pdms"])))
    thr = max(B, 2 * f)
    return {"d_r": dr, "F_p5c": f, "threshold_max_B_2F": thr, "min_d_dr": min(d_point, dr["point"]),
            "pass": bool(dr["lo"] > 0 and min(d_point, dr["point"]) > thr),
            "reported_beside": {"F": F_SEED, "2F3": F3_2X, "note": "no floor is dropped after it is seen (card 6.3)"}}


def goal_status(lv: dict) -> dict:
    """card 6.5 'as in PREREG_P6': MET iff point >= 94.55 AND ADOPT vs C1; NOT EXCLUDED iff point < 94.55 <= upper; BELOW iff upper < 94.55.
    The 'ADOPT vs C1' half has no counterpart on this card (UNVERIFIED reading): only the LEVEL half is computed."""
    p, hi = lv["point"], lv["hi"]
    s = "BELOW GOAL" if hi < GOAL_LEVEL else ("LEVEL AT/ABOVE GOAL (level half of GOAL MET only)" if p >= GOAL_LEVEL else "GOAL NOT EXCLUDED")
    return {"goal": GOAL_LEVEL, "status": s, "level_tracked": lv, "harness_gap": "<= ~0.25 reported, never subtracted (card 6.5)"}


def run(rows: dict, c1, c0, base, c0r=None, human=None, fam_ci=None, controls=(), n_boot=NBOOT, rehearsal=None, tactical=None, sealed=None, require_kinds=(), rows_paths=None) -> dict:
    arms = [a for a in (c1, c0, base, c0r, human) if a]
    P = per_token(rows, arms)
    bt = Boot(P["logs"], n_boot)
    res = {"instrument": "p5c_read.py B7", "card": "PREREG_P5C.md (frozen 2026-11-12)", "evidence": "MEASURED on the rows supplied; T1 executed",
           "estimator": {"kind": "paired log-cluster bootstrap", "draws": n_boot, "seed": SEED, "interval": "95 % percentile",
                         "statistic": "ratio of sums over drawn logs (rates too)", "shared": "ONE draw matrix for every arm, scorer and contrast; the fam_ci draws",
                         "answers": "another draw of LOGS only; training-run variance is NOT answered (floors + C0r, H-ESTIM-SEED-1); inference determinism is asserted (B10)"},
           "n_tokens": len(P["tokens"]), "n_logs": bt.L, "arms": {"C1": c1, "C0": c0, "base": base, "C0r": c0r, "human": human},
           "strategic_family": STRATEGIC, "units": "PDMS and sub-scores x 100"}
    res["levels"] = {a: {m: level(bt, P, a, m) for m in METRICS} for a in arms}
    cons = {f"{c1}-{c0}": contrast(bt, P, c1, c0), f"{c0}-{base}": contrast(bt, P, c0, base), f"{c1}-{base}": contrast(bt, P, c1, base)}
    if c0r:
        cons[f"{c1}-{c0r}"] = contrast(bt, P, c1, c0r)
        cons[f"{c0r}-{c0}"] = contrast(bt, P, c0r, c0)
    res["contrasts"] = cons
    prim = cons[f"{c1}-{c0}"]
    d, dT, TD = prim["pdms"]["tracked"], prim["pdms"]["transparent"], prim["pdms"]["TD"]
    res["B"] = {"value": B_BAR, "arithmetic": f"max({T_CRIT} * {S_P} * sqrt(2), 2 * {F_SEED}) = max({T_CRIT * S_P * math.sqrt(2):.4f}, {2 * F_SEED:.2f})", "2F3_beside": F3_2X}
    g1_fail = [m for m in G1_SUBS if prim[m]["tracked"]["hi"] < 0]
    g2 = g2_from_famci(fam_ci)
    c01 = cons[f"{c0}-{base}"]["pdms"]
    ctrl1 = {"c0_minus_base": c01, "valid": not (c01["tracked"]["hi"] < 0 and c01["tracked"]["point"] < -B_BAR),
             "rule": "INVALID iff CI-hi(C0 - base) < 0 AND (C0 - base) < -B (card 6.4); tracked scorer",
             "finding": ("the fine-tune recipe moves the base by itself" if (c01["tracked"]["lo"] > 0 and c01["tracked"]["point"] > B_BAR) else None)}
    p1c = p1c_clause(bt, P, c1, c0, c0r, d["point"], B_BAR) if c0r else None
    cs = []
    for fp in controls:
        j = json.load(open(fp, encoding="utf-8"))
        cs.append({"file": fp, "kind": j.get("control"), "pass": bool(j.get("pass", j.get("verdict") == "PASS"))})
    sealed_res = None
    if sealed is not None:                      # B11: SEALED.json is verified HERE, so the read cannot consume a tampered or late seal
        import p5c_seal
        sealed_res = p5c_seal.verify_sealed(sealed["doc"], sealed.get("card"), bool(sealed.get("recheck")), rows_paths)
        cs.append({"file": sealed.get("file"), "kind": p5c_seal.KIND_SEALED, "pass": sealed_res["pass"]})
    kinds = {c["kind"] for c in cs if c["kind"]}
    missing_kinds = [k for k in require_kinds if k not in kinds]
    controls_ok = (False if any(not c["pass"] for c in cs) else (None if (not cs or missing_kinds) else True))
    v = verdict_table(d, B_BAR, g1_fail, g2, ctrl1["valid"], controls_ok, p1c, missing_kinds)
    q = qualifier(TD, dT)
    dep_ci = cons[f"{c1}-{base}"]
    adv_base = [m for m in G1_SUBS + ("ddc",) if dep_ci[m]["tracked"]["hi"] < 0]
    lever = v["verdict"] == "LEVER"
    deploy = {"rule": "LEVER AND CI-lo(C1 - base) > 0 tracked AND no tracked sub-score of C1 - base separated adverse (card 6.5)",
              "c1_minus_base": dep_ci["pdms"], "adverse_subscores": adv_base,
              "outcome": ("TRUNK CHANGE" if (lever and dep_ci["pdms"]["tracked"]["lo"] > 0 and not adv_base) else "label lever shown, trunk unchanged")}
    res.update(g1={"subscores": list(G1_SUBS), "adverse": g1_fail, "status": "FAIL" if g1_fail else "PASS", "rule": "adverse = CI-hi < 0 of tracked C1 - C0"},
               g2=g2, ctrl1=ctrl1, p1c=p1c, controls={"supplied": cs, "all_pass": controls_ok, "required_kinds": list(require_kinds), "missing_kinds": missing_kinds}, sealed=sealed_res, verdict=v, qualifier=q, deploy=deploy,
               goal=goal_status(res["levels"][c1]["pdms"]["tracked"]), rates=rates_block(bt, P, arms, c1, c0, base),
               d_headline={"d_tracked": d, "d_T_transparent": dT, "TD": TD})
    res["no_tracker_numbers"] = {"note": f"numbers with no tracker in them carry the label '{NO_TRACKER}'"}
    res["card_questions"] = [
        "Q-1 qualifier order: TM+C's condition implies TM's; this instrument prints TM+C when both match (card 7 does not say).",
        "Q-2 GOAL MET: PREREG_P6's 'AND ADOPT vs C1' has no counterpart on the P5-C card; only the level half is computed and labelled so.",
        "Q-3 G1 lists NC, DAC, TTC, EP, C (no DDC); DDC is reported but is not a G1 term.",
        "Q-4 G2 polarity: directional components use the PAIRED C1-C0 interval; bias / ratio components use P6's unpaired interval rule (proxy_eval.fam_adverse), conservative.",
        "Q-5 with no controls supplied, no positive verdict is issued (INCOMPLETE): the card makes any failed read control INVALID but is silent on absent ones.",
        "Q-7 B10: the card names per-token PDMS; the instrument requires EVERY sub-score identical too (stricter), and the subset rule is k = ceil(0.10 n) via default_rng(20261112).choice (the card gives n and seed, not the RNG).",
        "Q-8 B11: the card gives the ledger entry shape, not the seal's schema; SEALED.json carries both sha256s, card and base sha256, UTC time and a self-hash; a real read refuses without it and checks the score rows are not older than the seal.",
        "Q-10 G2 over ZERO comparable components (an empty fam_ci frame, e.g. a RAISED families block) is NOT-EVALUATED, not PASS; the card does not say. B9 also refuses a RAISED families block.",
        "Q-9 B12: the card calls the TACTICAL row computable 'as in P6's read'; selection quality (64 harness columns, ~4 to 6.4 h CPU) is reported NOT COMPUTED with reason and n unless a table is supplied. Required controls for a real read default to B10_determinism + B11_sealed (W-TRANSP ii-iv carry no kind name yet).",
    ]
    import p5c_tactical
    if tactical is None:                        # B12: never a silent omission
        tactical = p5c_tactical.not_computed("no tactical.json (p5c_tactical.py) was supplied to this read", len(P["tokens"]))
    bad_t = p5c_tactical.validate_block(tactical)
    if bad_t:
        raise Refused("tactical block invalid: " + "; ".join(bad_t))
    res["tactical"] = tactical
    res["tactical_authority"] = "NONE: a reported family; it never decides the verdict (card 8.2)"
    if rehearsal:
        res["REHEARSAL"] = {"label": rehearsal, "claim": "none: no P5-C arm has trained; arms here are stand-ins, the verdict string is a pipeline exercise"}
        res["verdict"] = dict(v, verdict="REHEARSAL:" + v["verdict"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--c0", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--c0r", default=None)
    ap.add_argument("--human", default=None)
    ap.add_argument("--fam-ci", default=None, help="fam_ci.py JSON of C1 vs C0 (--a C0 --b C1); B9 writes it")
    ap.add_argument("--controls", nargs="*", default=[], help="JSON results with a boolean `pass` (B6 identity, W-TRANSP controls, B10 determinism)")
    ap.add_argument("--rehearsal", default=None, help="label; REQUIRED when no --controls are supplied")
    ap.add_argument("--tactical", default=None, help="tactical.json from p5c_tactical.py (B12); absent = the block says NOT COMPUTED")
    ap.add_argument("--sealed", default=None, help="SEALED.json (B11, card section 11); REQUIRED unless --rehearsal; verified by the read itself")
    ap.add_argument("--card", default=None, help="PREREG_P5C.md: with --sealed, its sha256 must equal the sealed card_sha256")
    ap.add_argument("--recheck-files", action="store_true", help="with --sealed: re-hash the sealed model files")
    ap.add_argument("--require-kinds", nargs="*", default=None, help="control kinds that must be supplied; default for a real read: B10_determinism B11_sealed")
    ap.add_argument("--n-boot", type=int, default=NBOOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.controls and not a.rehearsal:
        print("REFUSED: no --controls and no --rehearsal label: a real read needs its controls", file=sys.stderr)
        return 2
    if not a.sealed and not a.rehearsal:
        print("REFUSED: a real read needs --sealed SEALED.json (card section 11: both arms' model_final.pt sha256 sealed before either pass)", file=sys.stderr)
        return 2
    paths = [p for g in a.rows for p in sorted(glob.glob(g))]
    rows, bad = load_rows(paths)
    if bad:
        print(f"REFUSED: {len(bad)} tokens have only error rows (first {bad[0]})", file=sys.stderr)
        return 2
    famj = json.load(open(a.fam_ci, encoding="utf-8")) if a.fam_ci else None
    try:
        sealed = {"doc": json.load(open(a.sealed, encoding="utf-8")), "file": a.sealed, "card": a.card, "recheck": a.recheck_files} if a.sealed else None
        tact = json.load(open(a.tactical, encoding="utf-8")) if a.tactical else None
        req = tuple(a.require_kinds) if a.require_kinds is not None else (() if a.rehearsal else ("B10_determinism", "B11_sealed"))
        res = run(rows, a.c1, a.c0, a.base, a.c0r, a.human, famj, a.controls, a.n_boot, a.rehearsal, tact, sealed, req, paths)
    except Refused as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    dd = res["d_headline"]
    print("ZZP5CR_READ", res["verdict"]["verdict"], "qualifier", res["qualifier"]["code"], "n", res["n_tokens"], "logs", res["n_logs"],
          "d %.4f [%.4f, %.4f]" % (dd["d_tracked"]["point"], dd["d_tracked"]["lo"], dd["d_tracked"]["hi"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
