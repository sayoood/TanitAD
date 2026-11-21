#!/usr/bin/env python3
"""P2-s1 read: RR1 / RR2 / RR3 of eval/PREREG_P2_REPLICATE.md (sections 2 and 3), mapped to EXACTLY one outcome row.

    python p2rep_read.py --name p2s1_full --r2-result <stage3 r2_result.json of the arm> [--out-dir DIR]
    python p2rep_read.py --check-inputs                 # resolve + pin-check every banked input, read nothing of P2-s1

WHAT IT IS. A pure READER of three per-token tables of the SAME 12,146 navtest tokens (136 logs):
    P2-s1  the replicate's stage-2 table   score/refe_stage2_<name>/refe_stage2_<name>.csv   (harness output)
    P2-s0  P2 of record, banked            raw/p2_readout/p2_full/arm/refe_stage2_p2_full.csv      md5 bed5687f...
    u400   SFT-4 arm B u400, banked        raw/p2_r2_instrument/baseline/u400_navtest_full.csv     md5 b17c4fbac...
No inference. It reuses the program's OWN functions from eval/stage3_full.py (`load_table`, `read_pair`, `decide`,
`families_vs`) which in turn call proxy_eval.boot / read_csv / fam_adverse: nothing of the estimator is reimplemented here.

    RR1  P2-s1 minus u400, the IDENTICAL rule of PREREG_P2_AMENDMENT_1.md section 4 (stage3_full.decide): lo > 0 AND none of
         NC, DAC, EP, TTC, C, DDC adverse (paired interval entirely below 0).
    RR2  Delta_seed = PDMS(P2-s1) - PDMS(P2-s0), paired per token; F = |mean Delta_seed| WITH its interval (one draw of the
         run-to-run difference, never a variance).
    RR3  the four families: P2-s1 vs u400 and P2-s1 vs P2-s0, as R2 reports them (STRATEGIC: PAUSED, never a pass).
    Outcome (PREREG_P2_REPLICATE section 3, verbatim logic):
         any gate failed                               -> INVALID
         RR1 fails                                     -> NOT REPLICATED
         RR1 passes and F <  min(D_s0, D_s1)           -> REPLICATED
         RR1 passes and F >= min(D_s0, D_s1)           -> REPLICATED, FLOOR-BOUND
         (D_s0, D_s1 = the two arms' mean Delta vs u400, x100; the comparison is on the POINT ESTIMATES, as registered)
Estimator: paired log-cluster bootstrap over the 136 logs, 10,000 resamples, 95 %, seed 20260927. It answers 'another draw of
LOGS'. The replicate itself (seed 1 vs seed 0) is what supplies the training-variance draw; F is ONE such draw.
Tier: T1 = NAVSIM v1.1 PDMS, self-action open loop (not closed loop).

Gates (a failed gate makes the read INVALID, never a pass; all are in the output):
    stage3:*            every gate of the arm's own r2_result.json (harness PASS 12146/12146, 0 seam misses, rule, frame control,
                        checkpoint sha256 ...) plus: not a dry run, same table md5, the checkpoint sha was CHECKED (not merely recorded)
    pin:*               the P2-s0 table, the u400 table and the token file are the pinned bytes (md5, or md5 after CRLF->LF)
    estimator_pinned    10,000 / 20260927 here and in stage3_full
    p2s0_vs_u400_...    P2-s0 vs u400 recomputed here equals the banked R2 of P2 BIT FOR BIT (the identity control)
    rr1_equals_stage3   this script's RR1 equals stage3_full's own R2 block for the arm BIT FOR BIT (cross-instrument control)
    rows / finite       all three tables carry exactly the 12,146 tokens, finite numbers
    s1_not_s0 / s1_not_u400   the replicate's table is not token-for-token one of the other two (reading P2-s0 as P2-s1 gives
                        F = 0 and 'REPLICATED' -- the worst false pass this read can make)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK_SUFFIX = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
MIRRORS = ("E:/Projects/TanitAD/", "D:/Projects/TanitAD/")


def _find_eval_dir() -> Path:
    cands = [os.environ.get("REFE_EVAL_DIR")]
    try:
        cands.append(HERE.parents[1] / "eval")
    except IndexError:
        pass
    cands += [Path(m + PK_SUFFIX + "/eval") for m in MIRRORS]
    for c in cands:
        if c and (Path(c) / "stage3_full.py").is_file():
            return Path(c)
    raise SystemExit("p2rep_read: eval/stage3_full.py not found (set REFE_EVAL_DIR or run from the package)")


EVAL = _find_eval_dir()
PKG = EVAL.parent
sys.path.insert(0, str(EVAL))
import stage3_full as S3  # noqa: E402  (the program's own R2 instrument: load_table / read_pair / decide / families_vs)

PE = S3.PE
N_BOOT, SEED = 10000, 20260927
SUBS = list(S3.SUBS)

ESTIMATOR = ("paired log-cluster bootstrap over the 136 navtest logs (cluster unit log_name), 10,000 resamples, 95 %, seed 20260927 "
             "(proxy_eval.boot via stage3_full.read_pair). It answers 'another draw of LOGS'. Delta_seed is ONE draw of the run-to-run "
             "difference of two training seeds: F is a one-draw estimate, never a variance (PREREG_P2_REPLICATE section 2).")
TIER = S3.TIER
STRATEGIC = S3.STRATEGIC
EVIDENCE = "MEASURED (ours): T1 NAVSIM v1.1 PDMS on the 12,146 pinned navtest tokens; tables named in `inputs`"

# banked, content-pinned inputs.  `md5` = the bytes the harness wrote (CRLF); `md5_lf` = the same bytes with CRLF -> LF, which is what
# `git archive` yields on this repo (MEASURED 2026-10-16: both forms read the same rows).
PINS = {
    "p2_s0_csv": {"md5": "bed5687fb75e884d45cf991c2ff391c9", "md5_lf": "1dc7d7a7e0ed0ec8950311471a914758"},
    "u400_csv": {"md5": "b17c4fbac26d064f17ce790dca4d9d08", "md5_lf": "4ade3a8b849b15e9bd59d0b39f40543e"},
    "tokens": {"md5": "3a8b3653fd70891e50a230a37eb76ae8", "md5_lf": "3a8b3653fd70891e50a230a37eb76ae8", "n": 12146, "logs": 136},
}
BANK = PKG / "raw" / "p2_r2_instrument" / "baseline"
S0_ARM = PKG / "raw" / "p2_readout" / "p2_full" / "arm"


def _mirror(rel: str):
    return [Path(m + PK_SUFFIX + "/" + rel) for m in MIRRORS]


CANDS = {
    "p2_s0_csv": [*_mirror("raw/p2_readout/p2_full/arm/refe_stage2_p2_full.csv"), S0_ARM / "refe_stage2_p2_full.csv",
                  S3.DATA / "score/refe_stage2_p2_full/refe_stage2_p2_full.csv"],
    "u400_csv": [*S3.BASELINES["u400"]["csv"], *_mirror("raw/p2_r2_instrument/baseline/u400_navtest_full.csv"), BANK / "u400_navtest_full.csv"],
    "tokens": [S3.DATA / "navtest_full_tokens.json", *_mirror("raw/p2_r2_instrument/baseline/navtest_full_tokens.json"), BANK / "navtest_full_tokens.json"],
    "p2_s0_r2_result": [*_mirror("raw/p2_readout/p2_full/r2_result.json"), PKG / "raw" / "p2_readout" / "p2_full" / "r2_result.json"],
    "p2_s0_families": [*_mirror("raw/2026-10-05-sft-stage2/p2_full/p2_full_families.json"),
                       S3.S2_RAW / "p2_full/p2_full_families.json", PKG / "raw/2026-10-05-sft-stage2/p2_full/p2_full_families.json"],
    "u400_families": [*S3.BASELINES["u400"]["families"], *_mirror("raw/p2_r2_instrument/baseline/u400_navtest_full_families.json")],
}


# ---------------------------------------------------------------- small utilities
def md5_pair(path) -> tuple:
    """(md5 of the bytes, md5 with CRLF -> LF)"""
    raw = Path(path).read_bytes()
    return hashlib.md5(raw).hexdigest(), hashlib.md5(raw.replace(b"\r\n", b"\n")).hexdigest()


def pin_match(path, pin: dict) -> tuple:
    """(ok, form) -- ok iff the file is the pinned content, as written (CRLF) or as `git archive` yields it (LF)"""
    raw_md5, lf_md5 = md5_pair(path)
    if raw_md5 == pin["md5"]:
        return True, "as written"
    if lf_md5 == pin["md5_lf"]:
        return True, "line endings normalised (CRLF->LF)"
    return False, f"md5 {raw_md5} / lf {lf_md5} differ from the pin"


def resolve(key: str, pins: dict = PINS, cands: dict = CANDS) -> dict:
    """the first candidate whose CONTENT matches the pin (the name proves nothing)"""
    why = []
    for c in cands[key]:
        c = Path(c)
        if not c.is_file():
            why.append(f"{c}: absent")
            continue
        ok, form = pin_match(c, pins[key])
        if ok:
            return {"ok": True, "path": str(c), "form": form, "md5": md5_pair(c)[0], "tried": why}
        why.append(f"{c}: {form}")
    return {"ok": False, "path": None, "form": None, "md5": None, "tried": why}


def first_existing(key: str, cands: dict = CANDS):
    for c in cands[key]:
        if Path(c).is_file():
            return Path(c)
    return None


def load_json(p):
    return json.load(open(p, encoding="utf-8")) if p and Path(p).is_file() else None


def dump(obj, path) -> None:
    S3.dump(obj, Path(path))


def load_token_file(path, pin: dict) -> dict:
    """tokens + the cluster map the FILE carries (W3 format), gated on its pin: md5, n, logs, every token mapped, unique"""
    raw = Path(path).read_bytes()
    d = json.loads(raw.decode("utf-8"))
    toks = sorted(d["tokens"] if isinstance(d, dict) else d)
    tl = dict(d.get("token_log") or {}) if isinstance(d, dict) else {}
    md5s = (hashlib.md5(raw).hexdigest(), hashlib.md5(raw.replace(b"\r\n", b"\n")).hexdigest())
    g = {"path": str(path), "md5": md5s[0], "n": len(toks), "n_logs": len({tl.get(t) for t in toks}),
         "unique": len(set(toks)) == len(toks), "all_mapped": all(t in tl for t in toks),
         "md5_ok": pin["md5"] in md5s or pin["md5_lf"] in md5s, "n_ok": len(toks) == pin["n"], "logs_ok": len({tl.get(t) for t in toks}) == pin["logs"]}
    g["ok"] = bool(g["unique"] and g["all_mapped"] and g["md5_ok"] and g["n_ok"] and g["logs_ok"])
    return {"tokens": toks, "token_log": {t: tl[t] for t in toks if t in tl}, "gate": g}


# ---------------------------------------------------------------- the registered outcome mapping
OUTCOMES = {
    "REPLICATED": {
        "claim": "supported for 'P2's recipe raises PDMS over u400', with the confounds of amendment section 6 (labels, un-detached ctx and joint LoRA are not separated)",
        "next": "the lever is recorded at the mean of the two seeds' Delta; SFT-4R remains the arm that separates the confounds"},
    "REPLICATED, FLOOR-BOUND": {
        "claim": "NOT supported as a magnitude; the direction holds on both seeds",
        "next": "report both Deltas and F; any P2-derived lever is quoted as 'direction replicated, size within run-to-run noise'"},
    "NOT REPLICATED": {
        "claim": "NOT supported: P2's +0.57 lies within run-to-run variance",
        "next": "the next lever is the NOT PROVEN row of amendment section 4: SFT-4R (scorer-only on complete labels), then a seed replicate of the better arm (Rule Zero: never a stopping point)"},
    "INVALID": {
        "claim": "none -- an R2 gate failed; no outcome is recorded",
        "next": "fix the instrument and re-read; the S1 no-head seeds are the fallback floor only if this arm cannot produce a valid read"},
}


def map_outcome(rr1_pass: bool, d_s0: float, d_s1: float, d_seed: float, gates_ok: bool) -> dict:
    """PREREG_P2_REPLICATE section 3, verbatim: one row, INVALID first"""
    F = abs(d_seed)
    floor = min(d_s0, d_s1)
    if not gates_ok:
        name = "INVALID"
    elif not rr1_pass:
        name = "NOT REPLICATED"
    elif F < floor:
        name = "REPLICATED"
    else:
        name = "REPLICATED, FLOOR-BOUND"
    return {"outcome": name, "F": F, "min_delta_vs_u400": floor, "F_lt_min_delta": bool(F < floor), **OUTCOMES[name]}


def _same_read(a: dict, b: dict) -> bool:
    return a["mean"] == b["mean"] and a["ci95"] == b["ci95"]


def rr_read(s1: dict, s0: dict, u400: dict, toks: list, tl: dict, gates: dict | None = None, n_boot: int = N_BOOT,
            seed: int = SEED, banked_s0: dict | None = None, stage3_rr1: dict | None = None, require_controls: bool = False) -> dict:
    """RR1 + RR2 + the outcome row.  s1 / s0 / u400: token -> {PDMS, NC, DAC, EP, TTC, C, DDC} x100 (stage3_full.load_table).
    gates: the arm's stage-3 gate record ({name: {ok: bool, ...}}); None = no record = INVALID (fail closed)."""
    tk = set(toks)
    own = {}
    own["rows"] = {"ok": all(set(t) == tk for t in (s1, s0, u400)), "tokens": len(tk), "s1": len(s1), "p2_s0": len(s0), "u400": len(u400)}
    own["finite"] = {"ok": all(np.isfinite(v) for tbl in (s1, s0, u400) for row in tbl.values() for v in row.values())}
    sel = [t for t in toks if t in s1 and t in s0 and t in u400]
    rr1 = S3.read_pair(s1, u400, sel, tl, n_boot, seed)                     # RR1: P2-s1 - u400
    s0u = S3.read_pair(s0, u400, sel, tl, n_boot, seed)                     # P2-s0 - u400 (the banked R2 of P2)
    rr2 = S3.read_pair(s1, s0, sel, tl, n_boot, seed)                       # RR2: Delta_seed = P2-s1 - P2-s0
    own["s1_not_s0"] = {"ok": not (rr2["primary"]["tied"] == rr2["primary"]["n"]),
                        "why": "the replicate's table is token-for-token identical to P2-s0's PDMS (Delta_seed identically 0)"}
    own["s1_not_u400"] = {"ok": not (rr1["primary"]["tied"] == rr1["primary"]["n"]),
                          "why": "the replicate's table is token-for-token identical to u400's PDMS"}
    if banked_s0 is None and require_controls:
        own["p2s0_vs_u400_reproduces_banked_R2"] = {"ok": False, "why": "P2's banked r2_result.json not found: the identity control cannot run"}
    if stage3_rr1 is None and require_controls:
        own["rr1_equals_stage3_R2"] = {"ok": False, "why": "the arm's stage-3 R2 block is absent: the cross-instrument control cannot run"}
    if banked_s0 is not None:                                               # the identity control, in production
        eq = _same_read(s0u["primary"], banked_s0["primary"]) and all(_same_read(s0u["subscores"][k], banked_s0["subscores"][k]) for k in SUBS)
        own["p2s0_vs_u400_reproduces_banked_R2"] = {"ok": bool(eq), "recomputed": [s0u["primary"]["mean"], s0u["primary"]["ci95"]],
                                                    "banked": [banked_s0["primary"]["mean"], banked_s0["primary"]["ci95"]]}
    if stage3_rr1 is not None:                                              # cross-instrument control
        own["rr1_equals_stage3_R2"] = {"ok": bool(_same_read(rr1["primary"], stage3_rr1)), "here": [rr1["primary"]["mean"], rr1["primary"]["ci95"]],
                                       "stage3": [stage3_rr1.get("mean"), stage3_rr1.get("ci95")]}
    allg = dict(gates) if gates else {"stage3_gate_record": {"ok": False, "why": "no stage-3 gate record supplied: cannot establish the read is valid"}}
    allg.update(own)
    gates_ok = all(isinstance(g, dict) and g.get("ok") is True for g in allg.values())

    dec = S3.decide(rr1["primary"]["ci95"][0], rr1["primary"]["ci95"][1], rr1["adverse_subscores"], True)
    rr1_pass = dec["outcome"] == "GO"
    d_s0, d_s1, d_seed = s0u["primary"]["mean"], rr1["primary"]["mean"], rr2["primary"]["mean"]
    out = map_outcome(rr1_pass, d_s0, d_s1, d_seed, gates_ok)
    failed = sorted(k for k, g in allg.items() if not (isinstance(g, dict) and g.get("ok") is True))
    res = {
        "read": "P2-s1 replicate: RR1 + RR2 (+ RR3 families), PREREG_P2_REPLICATE sections 2-3",
        "estimator": ESTIMATOR, "tier": TIER, "evidence": EVIDENCE, "n_boot": n_boot, "seed": seed, "n_tokens": len(sel),
        "n_logs": len({tl[t] for t in sel}),
        "outcome": out["outcome"], "result_recorded": out["outcome"] != "INVALID",
        "outcome_detail": out, "gates_ok": gates_ok, "failed_gates": failed,
        "RR1": {"vs": "SFT-4 arm B u400 (88.25 PDMS)", "rule": "PREREG_P2_AMENDMENT_1 section 4 (stage3_full.decide): lo > 0 AND no sub-score adverse",
                "passes": rr1_pass, "amendment_4_row": dec["outcome"], "why": dec["why"], **rr1},
        "RR2": {"definition": "Delta_seed = PDMS(P2-s1) - PDMS(P2-s0), paired per token; F = |mean Delta_seed|, with its interval",
                "delta_seed_mean": d_seed, "delta_seed_ci95": rr2["primary"]["ci95"], "F": abs(d_seed),
                "F_upper_if_interval_endpoint": max(abs(rr2["primary"]["ci95"][0]), abs(rr2["primary"]["ci95"][1])),
                "note": "F = |point estimate| is the registered floor; the interval shows how loosely one pair of seeds pins it. One draw, never a variance.",
                **rr2},
        "P2_s0_vs_u400": {"note": "P2 of record against u400, recomputed here from the banked tables (the banked R2: +0.57 [+0.02, +1.18])", **s0u},
        "D_s0": d_s0, "D_s1": d_s1, "mean_of_two_seeds_delta": (d_s0 + d_s1) / 2.0,
        "arm_pdms_means_x100": {"u400": rr1["primary"]["base_mean_x100"], "p2_s0": s0u["primary"]["arm_mean_x100"], "p2_s1": rr1["primary"]["arm_mean_x100"]},
        "gates": allg,
    }
    return res


def families_block(fam_s1, fam_s0, fam_u400) -> dict:
    """RR3 exactly as R2 reports the families (stage3_full.families_vs = proxy_eval.fam_adverse): never pooled, never gating"""
    return {"s1_vs_u400": S3.families_vs(fam_u400, fam_s1), "s1_vs_p2_s0": S3.families_vs(fam_s0, fam_s1),
            "strategic": STRATEGIC, "gating": False,
            "note": "RR3 is reported, not tested (PREREG_P2_REPLICATE section 2: 'as R2'); adverse list = unpaired interval non-overlap, a conservative reading"}


def stage3_gates(r2: dict | None, name: str, s1_csv_md5: str | None) -> tuple:
    """(gates, stage3 R2 primary or None) from the arm's own stage-3 record"""
    if not r2:
        return {"stage3_record": {"ok": False, "why": "r2_result.json absent or unreadable"}}, None
    g = {"stage3:gates_ok": {"ok": r2.get("gates_ok") is True}}
    for k, v in (r2.get("gates") or {}).items():
        g[f"stage3:{k}"] = {"ok": isinstance(v, dict) and v.get("ok") is True}
    g["stage3:not_dry_run"] = {"ok": r2.get("dry_run") is False}
    g["stage3:name"] = {"ok": r2.get("name") == name, "got": r2.get("name"), "want": name}
    g["stage3:same_table"] = {"ok": bool(s1_csv_md5) and (r2.get("arm") or {}).get("csv_md5") == s1_csv_md5, "stage3": (r2.get("arm") or {}).get("csv_md5"), "read_here": s1_csv_md5}
    cks = ((r2.get("gates") or {}).get("checkpoint") or {}).get("sha256_matches_expected")
    g["stage3:checkpoint_sha256_was_checked"] = {"ok": cks is True, "sha256_matches_expected": cks}
    return g, ((r2.get("R2") or {}).get("primary"))


# ---------------------------------------------------------------- rendering
def _pm(x):
    return f"{x:+.2f}"


def line(res: dict, name: str) -> str:
    p, d = res["RR1"]["primary"], res["RR2"]
    return (f"ZZP2REP_READ {name} {res['outcome']} RR1 {_pm(p['mean'])} [{_pm(p['ci95'][0])}, {_pm(p['ci95'][1])}] "
            f"adverse={','.join(res['RR1']['adverse_subscores']) or 'none'} | F={d['F']:.2f} dseed {_pm(d['delta_seed_mean'])} "
            f"[{_pm(d['delta_seed_ci95'][0])}, {_pm(d['delta_seed_ci95'][1])}] D_s0={_pm(res['D_s0'])} D_s1={_pm(res['D_s1'])} gates_ok={res['gates_ok']}")


def render_md(res: dict, name: str, inputs: dict) -> str:
    o = res["outcome_detail"]
    L = [f"# P2-s1 read: {name}  ->  **{res['outcome']}**", "",
         f"> {line(res, name)}", "", f"Tier: {res['tier']}.", f"Estimator: {res['estimator']}", "",
         f"Claim on P2's lever: {o['claim']}.  Next: {o['next']}.", ""]
    if not res["result_recorded"]:
        L += [f"**INVALID. No outcome is recorded.** Failed gates: {', '.join(res['failed_gates'])}.", ""]
    L += ["## Six sub-scores (x100, paired, 95 % log-cluster CI)", "",
          "| sub-score | RR1: P2-s1 - u400 | adverse | RR2: P2-s1 - P2-s0 | P2-s0 - u400 (banked) |", "|---|---|---|---|---|"]
    for k in ["PDMS"] + SUBS:
        a = res["RR1"]["primary"] if k == "PDMS" else res["RR1"]["subscores"][k]
        b = res["RR2"]["primary"] if k == "PDMS" else res["RR2"]["subscores"][k]
        c = res["P2_s0_vs_u400"]["primary"] if k == "PDMS" else res["P2_s0_vs_u400"]["subscores"][k]
        f = lambda r: f"{_pm(r['mean'])} [{_pm(r['ci95'][0])}, {_pm(r['ci95'][1])}]"
        L.append(f"| {k} | {f(a)} | {'ADVERSE' if (k != 'PDMS' and a.get('adverse')) else ''} | {f(b)} | {f(c)} |")
    L += ["", f"PDMS means x100: u400 {res['arm_pdms_means_x100']['u400']:.4f}, P2-s0 {res['arm_pdms_means_x100']['p2_s0']:.4f}, P2-s1 {res['arm_pdms_means_x100']['p2_s1']:.4f}.",
          f"RR2: F = |{_pm(res['RR2']['delta_seed_mean'])}| = {res['RR2']['F']:.4f}; registered comparison F < min(D_s0, D_s1) = {o['min_delta_vs_u400']:.4f}: {o['F_lt_min_delta']}.",
          f"Mean of the two seeds' Delta vs u400: {res['mean_of_two_seeds_delta']:+.4f}.", "",
          "## Families (RR3; reported, not gating)", ""]
    fb = res.get("families")
    if fb:
        for key in ("s1_vs_u400", "s1_vs_p2_s0"):
            r = fb[key]
            if r.get("missing"):
                L.append(f"- {key}: {r['note']}")
            else:
                L.append(f"- {key}: {r['components_compared']} longitudinal/lateral components compared; adverse: "
                         f"{', '.join(a['family'] + '.' + a['metric'] for a in r['adverse']) or 'none'}. Tactical rows (no interval): see JSON.")
    else:
        L.append("- families not computed (a families JSON is absent): NOT COMPUTED, never a pass")
    L += [f"- STRATEGIC: {STRATEGIC}.", "", "## Gates", ""]
    for k, g in res["gates"].items():
        L.append(f"- {'PASS' if g.get('ok') is True else 'FAIL'}  {k}")
    L += ["", "## Inputs", ""] + [f"- {k}: {v}" for k, v in inputs.items()]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_inputs(a, pins=PINS, cands=CANDS) -> dict:
    """every input of the read, resolved and pin-checked"""
    inp = {"p2_s0_csv": resolve("p2_s0_csv", pins, cands), "u400_csv": resolve("u400_csv", pins, cands)}
    tp = a.tokens or None
    if tp is None:
        for c in cands["tokens"]:
            if Path(c).is_file() and pin_match(c, pins["tokens"])[0]:
                tp = c
                break
    inp["tokens_path"] = tp
    inp["p2_s0_r2_result"] = str(first_existing("p2_s0_r2_result", cands) or "")
    inp["p2_s0_families"] = str(first_existing("p2_s0_families", cands) or "")
    inp["u400_families"] = str(first_existing("u400_families", cands) or "")
    return inp


def main(argv=None, _pins=None, _cands=None) -> int:
    pins, cands = _pins or PINS, _cands or CANDS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="p2s1_full", help="the stage-2 run name of the replicate (stage2_sft.py --name)")
    ap.add_argument("--s1-csv", default=None, help="default: score/refe_stage2_<name>/refe_stage2_<name>.csv under the navtest data dir")
    ap.add_argument("--s0-csv", default=None), ap.add_argument("--u400-csv", default=None), ap.add_argument("--tokens", default=None)
    ap.add_argument("--r2-result", default=None, help="the replicate's stage3_full r2_result.json (its gates decide INVALID); default raw/p2_readout/<name>/r2_result.json")
    ap.add_argument("--s0-r2-result", default=None, help="P2's banked r2_result.json (the identity control)")
    ap.add_argument("--s1-families", default=None), ap.add_argument("--s0-families", default=None), ap.add_argument("--u400-families", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--check-inputs", action="store_true", help="resolve and pin-check the banked inputs and exit (nothing of P2-s1 is read)")
    a = ap.parse_args(argv)

    inp = build_inputs(a, pins, cands)
    s0_path = Path(a.s0_csv) if a.s0_csv else (Path(inp["p2_s0_csv"]["path"]) if inp["p2_s0_csv"]["ok"] else None)
    u4_path = Path(a.u400_csv) if a.u400_csv else (Path(inp["u400_csv"]["path"]) if inp["u400_csv"]["ok"] else None)
    pin_gates = {"pin:p2_s0_csv": {"ok": bool(s0_path and s0_path.is_file() and pin_match(s0_path, pins["p2_s0_csv"])[0]), "path": str(s0_path)},
                 "pin:u400_csv": {"ok": bool(u4_path and u4_path.is_file() and pin_match(u4_path, pins["u400_csv"])[0]), "path": str(u4_path)}}
    tok = load_token_file(inp["tokens_path"], pins["tokens"]) if inp["tokens_path"] and Path(inp["tokens_path"]).is_file() else None
    pin_gates["pin:tokens"] = tok["gate"] if tok else {"ok": False, "why": "token file not found"}
    if a.check_inputs:
        print(json.dumps({"inputs": inp, "gates": pin_gates, "eval_dir": str(EVAL), "estimator_pinned": [S3.N_BOOT, S3.SEED] == [N_BOOT, SEED]}, indent=1, default=str))
        ok = all(g["ok"] for g in pin_gates.values()) and bool(inp["p2_s0_r2_result"]) and [S3.N_BOOT, S3.SEED] == [N_BOOT, SEED]
        print("ZZP2REP_INPUTS_" + ("OK" if ok else "FAIL"))
        return 0 if ok else 3

    name = a.name
    out_dir = Path(a.out_dir) if a.out_dir else HERE / "out" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_csv = Path(a.s1_csv) if a.s1_csv else S3.DATA / "score" / f"refe_stage2_{name}" / f"refe_stage2_{name}.csv"
    r2_path = Path(a.r2_result) if a.r2_result else PKG / "raw" / "p2_readout" / name / "r2_result.json"
    miss = [str(p) for p in (s1_csv, s0_path, u4_path) if not (p and Path(p).is_file())]
    if miss or tok is None:
        print("ZZP2REP_READ_ERROR missing input(s):", miss or ["tokens"], flush=True)
        return 3
    toks, tl = tok["tokens"], tok["token_log"]
    s1, s0, u4 = (S3.load_table(p) for p in (s1_csv, s0_path, u4_path))
    s1_md5 = md5_pair(s1_csv)[0]
    r2 = load_json(r2_path)
    g3, s3_rr1 = stage3_gates(r2, name, s1_md5)
    gates = {**pin_gates, **g3, "estimator_pinned": {"ok": [S3.N_BOOT, S3.SEED] == [N_BOOT, SEED], "stage3": [S3.N_BOOT, S3.SEED], "here": [N_BOOT, SEED]}}
    banked = load_json(a.s0_r2_result or first_existing("p2_s0_r2_result", cands))
    res = rr_read(s1, s0, u4, toks, tl, gates=gates, banked_s0=(banked or {}).get("R2"), stage3_rr1=s3_rr1, require_controls=True)
    fams = [load_json(p) for p in (a.s1_families or S3.S2_RAW / name / f"{name}_families.json", a.s0_families or first_existing("p2_s0_families", cands),
                                   a.u400_families or first_existing("u400_families", cands))]
    res["families"] = families_block(fams[0], fams[1], fams[2])
    res.update({"name": name, "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": "p2rep_read.py", "eval_dir": str(EVAL),
                "inputs": {"s1_csv": str(s1_csv), "s1_csv_md5": s1_md5, "p2_s0_csv": str(s0_path), "u400_csv": str(u4_path), "tokens": inp["tokens_path"],
                           "r2_result": str(r2_path), "families": [str(p) for p in (a.s1_families, a.s0_families, a.u400_families)]}})
    dump(res, out_dir / "p2rep_result.json")
    (out_dir / "P2REP_READ.md").write_text(render_md(res, name, res["inputs"]), encoding="utf-8")
    print(line(res, name), flush=True)
    return 0 if res["result_recorded"] else 2


if __name__ == "__main__":
    sys.exit(main())
