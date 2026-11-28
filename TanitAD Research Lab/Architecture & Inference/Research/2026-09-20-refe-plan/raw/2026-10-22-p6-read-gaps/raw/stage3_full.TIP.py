#!/usr/bin/env python3
"""Stage 3 / read R2 of eval/PREREG_P2_AMENDMENT_1.md: P2 on ALL 12,146 navtest tokens (136 logs) against SFT-4 arm B u400
(88.25 PDMS), which decides P6 GO / NO-GO (section 4 of the amendment). R2b (the same tokens against the deployed system,
84.46 PDMS) rides along, reported and deciding nothing.

WHY A SIBLING OF stage2_sft.py AND NOT A FLAG ON IT. stage2_sft.py pairs against deployed-run constants (BASE_CSV / BASE_SEAM)
and is R1's instrument, "unchanged" by the amendment; this file leaves it BYTE-IDENTICAL (its md5 is recorded by --selfcheck)
and reuses its estimator, not its main(): proxy_eval.boot (paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927),
proxy_eval.read_csv and proxy_eval.fam_adverse are the SAME functions stage 2 calls.

WHAT IT DOES NOT DO: inference. The seam and the harness are run by the unchanged stage2_sft.py (it already takes the
whole-navtest path when --tokens is the full set); this script READS what that run wrote and applies the registered rule.
Run order, expected times and the exact commands: eval/P2_READOUT_RUNBOOK.md.

    python eval/stage3_full.py --name p2_full [--ckpt <model_final.pt>] [--expect-rule navsim_v1]
    python eval/stage3_full.py --selfcheck          # the instrument precondition of amendment section 3 (banks raw/p2_r2_instrument/)

Reads (all under E:/Projects/TanitAD/data/refe_navtest unless a flag says otherwise), for --name N:
    score/refe_stage2_N/refe_stage2_N.csv      the harness's per-token table of the arm
    seams/stage2_N.npz  (+ .report.json)       the arm's executed poses, and the seam's own report
    <stage-2 raw dir>/N/score_status.json      the harness's OWN status (a CSV existing is not evidence)
    <stage-2 raw dir>/N/N_families.json        families6 on the arm (computed here if absent)
Baselines are PINNED by md5 (u400, deployed): a wrong table is refused, not paired. A copy of each lives in
raw/p2_r2_instrument/baseline/ so R2 does not depend on one disk.

Outcome (the amendment's section-4 table, applied verbatim):
    delta = PDMS(arm) - PDMS(u400), x100, lo / hi = the 95 % interval
    lo > 0 and no PDMS sub-score adverse      -> GO
    lo > 0 and some sub-score adverse         -> NO-GO (names the component)
    lo <= 0 <= hi                             -> NO-GO (NOT PROVEN)
    hi < 0                                    -> NO-GO (P2 below u400)
    an instrument gate failed                 -> INVALID (no read; never a GO)
A sub-score is ADVERSE iff the paired interval of (arm - u400) lies entirely below 0 (NC, DAC, EP, TTC, C, DDC).
The 21 longitudinal / lateral family components are REPORTED against u400 and against deployed and do NOT gate.
The rows "R1 REFUTED" and "P2 does not finish / launch gate failed" of the table are states, not data: the runbook handles them.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import proxy_eval as PE  # noqa: E402  (the SAME estimator stage 2 uses)

DATA = Path(os.environ.get("REFE_NAVTEST_DIR", "E:/Projects/TanitAD/data/refe_navtest"))
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
# stage2_sft.py writes under ITS OWN hard-coded package root (the E: mirror of this package, byte-identical to it)
S2_RAW = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2")
BANK = PKG / "raw" / "p2_r2_instrument" / "baseline"          # the in-repo copy of every baseline table
OUT_ROOT = PKG / "raw" / "p2_readout"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
STAGE2_SFT = HERE / "stage2_sft.py"
STAGE2_SFT_MD5 = "831a8d76bb5b8be2016288b02cd29ee5"             # R1's instrument; this file must never change it
FAMILIES6_CANDIDATES = (
    os.environ.get("REFE_FAMILIES6", ""),
    "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/families6.py",
    "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad/ev6e/families6.py",
)

N_BOOT, SEED = 10000, 20260927
ESTIMATOR = ("paired log-cluster bootstrap over the navtest logs (cluster unit log_name), 10,000 resamples, 95 %, seed 20260927 "
             "(proxy_eval.boot, the estimator of RESULT_SFT4.md). It answers 'another draw of LOGS' and nothing else; "
             "training variance (one seed per arm, H-ESTIM-SEED-1) is unmeasured; inference is deterministic.")
TIER = "T1 = NAVSIM v1.1 PDMS, an ego pseudo-simulation of an open-loop plan against logged agents (self-action open loop, not closed loop)"
STRATEGIC = ("PAUSED by the PI (2026-09-19, PI_DECISION_QUEUE item 25) and UNAVAILABLE in NAVSIM (no strategic decision is scored); "
             "reported as such, never as a pass")

SUBS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
        "TTC": "time_to_collision_within_bound", "C": "comfort", "DDC": "driving_direction_compliance"}
FULL_PINS = {"md5": "3a8b3653fd70891e50a230a37eb76ae8", "n": 12146, "logs": 136}
C1_PINS = {"md5": "bc813be8300b69f9795e190238806329", "n": 12110, "logs": 136}   # navtest minus the 36 C1 tokens / 4 logs
# content pins, MEASURED 2026-10-12 on the files that produced RESULT_SFT4.md's full-navtest numbers (the pairing of these two
# tables reproduces its result.json to the bit -- that, not the file name, is what ties the pins to the banked result)
BASELINES = {
    "u400": {
        "label": "SFT-4 arm B u400 (rule navsim_v1_lane)", "pdms_x100": 88.2469,
        "csv": [DATA / "score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv", BANK / "u400_navtest_full.csv"],
        "csv_md5": "b17c4fbac26d064f17ce790dca4d9d08",
        "seam": [DATA / "seams/stage2_sft4_B_u400_navtest_full.npz", BANK / "u400_navtest_full_seam.npz"],
        "seam_md5": "b009e941620b6cbbbe63e1dd8a94a506",
        "report": [DATA / "seams/stage2_sft4_B_u400_navtest_full.report.json", BANK / "u400_navtest_full_seam.report.json"],
        "families": [S2_RAW / "sft4_B_u400_navtest_full/sft4_B_u400_navtest_full_families.json", BANK / "u400_navtest_full_families.json"],
        "families_md5": "ed816ea98dd0e3fd2df2178fb3366714",
    },
    "deployed": {
        "label": "deployed route-fix system (rule navsim_v1)", "pdms_x100": 84.4561,
        "csv": [DATA / "score/refe_navtest_final_routefix/refe_navtest_final_routefix.csv", BANK / "deployed_navtest_full.csv"],
        "csv_md5": "2101bc7f2084195b2536e5ee09c6be32",
        "seam": [DATA / "seams/refe_navtest_final_routefix.npz", BANK / "deployed_navtest_full_seam.npz"],
        "seam_md5": "1ddc2c63b538bf49fcc71c6047260696",
        "report": [DATA / "seams/refe_navtest_final_routefix.report.json", BANK / "deployed_navtest_full_seam.report.json"],
        "families": [S2_RAW / "sft4_B_u400_navtest_full/deployed_families.json", BANK / "deployed_navtest_full_families.json"],
        "families_md5": "4b4d776cca3838edd607d3b59ffba5d7",
    },
}


# ---------------------------------------------------------------- small utilities
def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def first_matching(cands, md5: str | None):
    """the first existing candidate whose CONTENT hash equals md5 (the name proves nothing); (path, why-not list)"""
    why = []
    for c in cands:
        c = Path(c)
        if not c.exists():
            why.append(f"{c}: absent")
            continue
        if md5 is None or md5_file(c) == md5:
            return c, why
        why.append(f"{c}: md5 differs from the pin")
    return None, why


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---------------------------------------------------------------- tokens
def load_tokens(path: Path, pins: dict | None, export_logs: dict) -> dict:
    """tokens + their gate. pins=None -> no pin (dry run / custom set). Gates: md5, n, logs, every token in the export, and
    the file's own token_log (if it carries one) equal to the export's."""
    raw = Path(path).read_bytes()
    d = json.loads(raw.decode("utf-8"))
    toks = sorted(d["tokens"] if isinstance(d, dict) else d)
    tl = {t: export_logs[t] for t in toks if t in export_logs}
    g = {"path": str(path), "md5": md5_bytes(raw), "n": len(toks), "n_logs": len(set(tl.values())),
         "all_in_export": len(tl) == len(toks), "unique": len(set(toks)) == len(toks)}
    own = d.get("token_log") if isinstance(d, dict) else None
    g["token_log_matches_export"] = (None if own is None else all(own.get(t) == tl.get(t) for t in toks))
    ok = g["all_in_export"] and g["unique"] and g["token_log_matches_export"] in (True, None)
    if pins is not None:
        g["pinned"] = pins
        g["md5_ok"] = g["md5"] == pins["md5"]
        g["n_ok"] = g["n"] == pins["n"]
        g["logs_ok"] = g["n_logs"] == pins["logs"]
        ok = ok and g["md5_ok"] and g["n_ok"] and g["logs_ok"]
    g["ok"] = bool(ok)
    return {"tokens": toks, "token_log": tl, "gate": g}


# ---------------------------------------------------------------- tables and the paired read
def load_table(csv_path) -> dict:
    """token -> {PDMS, NC, DAC, EP, TTC, C, DDC} all x100 (stage2_sft.py's own `100.0 * float(r["score"])`)"""
    out = {}
    for t, r in PE.read_csv(csv_path).items():
        row = {"PDMS": 100.0 * float(r["score"])}
        for k, c in SUBS.items():
            row[k] = 100.0 * float(r[c])
        out[t] = row
    return out


def read_one(arm: dict, base: dict, sel: list, tl: dict, key: str, n: int = N_BOOT, seed: int = SEED) -> dict:
    d = {t: arm[t][key] - base[t][key] for t in sel}
    mu, lo, hi = PE.boot(d, tl, n=n, seed=seed)
    return {"mean": mu, "ci95": [lo, hi], "base_mean_x100": float(np.mean([base[t][key] for t in sel])),
            "arm_mean_x100": float(np.mean([arm[t][key] for t in sel])),
            "better": sum(v > 0 for v in d.values()), "worse": sum(v < 0 for v in d.values()),
            "tied": sum(v == 0 for v in d.values()), "n": len(sel), "n_logs": len({tl[t] for t in sel})}


def read_pair(arm: dict, base: dict, toks: list, tl: dict, n: int = N_BOOT, seed: int = SEED) -> dict:
    """PDMS and the six sub-scores, arm minus base, one estimator. adverse <=> the interval lies entirely below 0."""
    sel = [t for t in toks if t in arm and t in base]
    prim = read_one(arm, base, sel, tl, "PDMS", n, seed)
    prim["zero_base"] = sum(base[t]["PDMS"] == 0 for t in sel)
    prim["zero_arm"] = sum(arm[t]["PDMS"] == 0 for t in sel)
    subs = {}
    for k in SUBS:
        r = read_one(arm, base, sel, tl, k, n, seed)
        r["adverse"] = bool(r["ci95"][1] < 0)
        subs[k] = r
    return {"primary": prim, "subscores": subs, "adverse_subscores": [k for k, v in subs.items() if v["adverse"]],
            "n_rows": len(sel)}


def decide(lo: float, hi: float, adverse: list, gates_ok: bool) -> dict:
    """amendment section 4, verbatim"""
    if not gates_ok:
        return {"outcome": "INVALID", "p6": "NO READ", "why": "an instrument gate failed; this is not a result and never a GO",
                "next": "fix the gate; the read is void until it passes"}
    if lo > 0 and not adverse:
        return {"outcome": "GO", "p6": "GO",
                "why": "lower bound > 0 and no PDMS sub-score adverse",
                "next": "finalise and land PREREG_P6.md (+ the PI's P6-T/C/N/Y decisions); the SPEND is the PI's call, this only supplies the confidence"}
    if lo > 0:
        return {"outcome": "NO-GO (adverse sub-score)", "p6": "NO-GO",
                "why": f"lower bound > 0 but adverse sub-score(s): {', '.join(adverse)}",
                "next": "scorer-side fix aimed at the named component, run as the cheaper SFT-4R-style arm first"}
    if hi < 0:
        return {"outcome": "NO-GO (P2 below u400)", "p6": "NO-GO",
                "why": "upper bound < 0: P2 reads below u400",
                "next": "SFT-4R (PREREG_SFT4R.md) plus P2's training log (score loss, LoRA movement, held-out ranking) to say which half failed"}
    return {"outcome": "NO-GO (NOT PROVEN)", "p6": "NO-GO",
            "why": "interval contains 0",
            "next": "SFT-4R (scorer-only on COMPLETE labels, PREREG_SFT4R.md), then a training-seed replicate of the better arm"}


# ---------------------------------------------------------------- gates
def consistency(arm_seam: Path, base_seam: Path, arm: dict, base: dict, sel: list) -> dict:
    """every token whose executed poses are bit-equal in the two seams must score identically (stage 2's control)"""
    za, zb = np.load(arm_seam), np.load(base_seam)
    ap = {str(t): i for i, t in enumerate(za["token"])}
    bp = {str(t): i for i, t in enumerate(zb["token"])}
    same = [t for t in sel if t in ap and t in bp and np.array_equal(za["poses"][ap[t]], zb["poses"][bp[t]])]
    mism = [t for t in same if arm[t]["PDMS"] != base[t]["PDMS"]]
    other = [t for t in same if any(arm[t][k] != base[t][k] for k in SUBS)]
    return {"ok": not mism, "same_pose_tokens": len(same), "same_pose_different_score": len(mism),
            "same_pose_any_column_differs": len(other), "seam_rows_arm": len(ap), "seam_rows_base": len(bp)}


def resolve_families6():
    for c in FAMILIES6_CANDIDATES:
        if c and Path(c).exists():
            return Path(c)
    return None


def compute_families(seam_npz: Path, label: str, out_dir: Path, toks: set) -> dict | None:
    """families6 on the seam (the call stage2_sft.families makes), with the first families6.py that exists"""
    f6 = resolve_families6()
    if f6 is None:
        return None
    z = np.load(seam_npz)
    tt = z["token"]
    idx = [i for i, t in enumerate(tt) if str(t) in toks]
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = out_dir / f"{label}_seam.npz"
    np.savez(sub, token=tt[idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx], sampling=z["sampling"], arm=z["arm"])
    out = out_dir / f"{label}_families.json"
    subprocess.call([PY_T, str(f6), "--seam", str(sub), "--inputs", EXPORT, "--stage", "1", "--label", label, "--out", str(out),
                     "--n-boot", "2000"], cwd=str(f6.parent), env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    stdout=open(out_dir / f"{label}_families.log", "w"), stderr=subprocess.STDOUT)
    return json.load(open(out, encoding="utf-8")) if out.exists() else None


def families_vs(base_fam: dict | None, arm_fam: dict | None) -> dict:
    if not base_fam or not arm_fam:
        return {"missing": True, "note": "a families JSON is absent: NOT COMPUTED (never a pass)"}
    r = PE.fam_adverse(base_fam, arm_fam)             # (W = baseline, P = arm), exactly stage 2's call
    r["note"] = "REPORTED, NOT GATING (amendment section 5); unpaired interval non-overlap, a conservative reading"
    return r


def tree_diff(a, b, path="") -> list:
    """every path at which two JSON-like trees differ (exact equality)"""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: present in one only")
            else:
                bad += tree_diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            bad += tree_diff(x, y, f"{path}[{i}]")
    elif a != b:
        bad.append(f"{path}: {str(a)[:60]} != {str(b)[:60]}")
    return bad


# ---------------------------------------------------------------- the R2 pipeline
def pick_baseline(name: str) -> dict:
    b = BASELINES[name]
    out = {"name": name, "label": b["label"]}
    for k in ("csv", "seam", "report", "families"):
        md5 = b.get(f"{k}_md5") if k != "report" else None
        p, why = first_matching(b[k], md5)
        out[k] = p
        if p is None:
            out[f"{k}_missing"] = why
    out["csv_md5"] = md5_file(out["csv"]) if out["csv"] else None
    out["seam_md5"] = md5_file(out["seam"]) if out["seam"] else None
    out["pinned_csv_md5"], out["pinned_seam_md5"] = b["csv_md5"], b["seam_md5"]
    out["pin_ok"] = bool(out["csv"] and out["seam"] and out["csv_md5"] == b["csv_md5"] and out["seam_md5"] == b["seam_md5"])
    return out


def arm_sources(a) -> dict:
    """the arm's files: --arm-from <baseline> (self-check), explicit flags, or the stage-2 naming convention for --name"""
    if a.arm_from:
        s = pick_baseline(a.arm_from)
        s["status"] = None
        s["named_baseline"] = True
        return s
    n = a.name
    s2 = S2_RAW / n
    return {"name": n, "label": f"arm {n}", "named_baseline": False,
            "csv": Path(a.arm_csv) if a.arm_csv else DATA / "score" / f"refe_stage2_{n}" / f"refe_stage2_{n}.csv",
            "seam": Path(a.arm_seam) if a.arm_seam else DATA / "seams" / f"stage2_{n}.npz",
            "report": Path(a.arm_report) if a.arm_report else DATA / "seams" / f"stage2_{n}.report.json",
            "families": Path(a.arm_families) if a.arm_families else s2 / f"{n}_families.json",
            "status": Path(a.arm_status) if a.arm_status else s2 / "score_status.json",
            "stage2_result": s2 / "result.json"}


def run_r2(a) -> dict:
    t_start = time.time()
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    export_logs = {t: v["log_name"] for t, v in exp.items()}
    out_dir = Path(a.out_dir) if a.out_dir else OUT_ROOT / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokens
    tok_path = Path(a.tokens) if a.tokens else first_matching([DATA / "navtest_full_tokens.json", BANK / "navtest_full_tokens.json"], FULL_PINS["md5"])[0]
    T = load_tokens(tok_path, None if a.dry_run else FULL_PINS, export_logs)
    toks, tl = T["tokens"], T["token_log"]
    c1_path = first_matching([DATA / "navtest_full_minus_c1_tokens.json", BANK / "navtest_full_minus_c1_tokens.json"], C1_PINS["md5"])[0]
    C1 = load_tokens(c1_path, C1_PINS, export_logs) if (c1_path and not a.dry_run) else None

    arm_s = arm_sources(a)
    base_s = pick_baseline(a.base)
    also_s = pick_baseline(a.also) if a.also else None
    res = {"name": a.name, "read": "R2 (+R2b)", "dry_run": bool(a.dry_run), "estimator": ESTIMATOR, "tier": TIER,
           "strategic_family": STRATEGIC, "script": "eval/stage3_full.py", "stage2_sft_md5": md5_file(STAGE2_SFT),
           "stage2_sft_md5_expected": STAGE2_SFT_MD5, "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    gates = {"tokens": T["gate"]}
    gates["tokens_minus_c1"] = (C1["gate"] if C1 else {"ok": True, "note": "not applicable (dry run / file absent)"})
    res["stage2_sft_unmodified"] = res["stage2_sft_md5"] == STAGE2_SFT_MD5     # recorded, NOT a gate: an unrelated landing must not void R2
    gates["base_pinned"] = {"ok": base_s["pin_ok"], "base": a.base, "csv_md5": base_s["csv_md5"], "pinned": base_s["pinned_csv_md5"],
                            "seam_md5": base_s["seam_md5"], "pinned_seam": base_s["pinned_seam_md5"]}
    if also_s:
        gates["also_pinned"] = {"ok": also_s["pin_ok"], "also": a.also, "csv_md5": also_s["csv_md5"], "seam_md5": also_s["seam_md5"]}
    res["arm"] = {"name": arm_s["name"], "csv": str(arm_s["csv"]), "seam": str(arm_s["seam"])}
    if not arm_s["csv"].exists() or not arm_s["seam"].exists():
        gates["arm_files"] = {"ok": False, "csv_exists": arm_s["csv"].exists(), "seam_exists": arm_s["seam"].exists()}
        res["gates"], res["gates_ok"] = gates, False
        res["R2"] = decide(0.0, 0.0, [], False)
        dump(res, out_dir / "r2_result.json")
        print("ZZSTAGE3", a.name, "INVALID (arm files absent)", flush=True)
        return res
    res["arm"]["csv_md5"], res["arm"]["seam_md5"] = md5_file(arm_s["csv"]), md5_file(arm_s["seam"])
    s2r = arm_s.get("stage2_result")
    if s2r is not None and Path(s2r).exists():
        j = json.load(open(s2r, encoding="utf-8"))
        res["arm"]["stage2_run"] = {k: j.get(k) for k in ("ckpt", "rule", "tokens", "n_tokens", "scored_as", "exploratory", "verdict")}
        res["arm"]["stage2_run"]["WARNING"] = ("that verdict is R1/R2b-style (arm vs DEPLOYED). It is NOT the P6 verdict; "
                                                "the P6 GO / NO-GO is this file's R2 block only")

    # the harness's OWN status, and the seam's own report
    if arm_s.get("status") is not None or not arm_s.get("named_baseline"):
        st = json.load(open(arm_s["status"], encoding="utf-8")) if arm_s["status"] and Path(arm_s["status"]).exists() else {}
        gates["harness_status"] = {"ok": (st.get("status") == "PASS" and st.get("log_failed") == 0
                                          and st.get("log_successful") == len(toks) == st.get("csv_valid_rows")
                                          and st.get("C1_max_abs_delta") == 0.0) or (a.dry_run and st.get("status") == "PASS"),
                                   "status": st.get("status"), "log_successful": st.get("log_successful"),
                                   "log_failed": st.get("log_failed"), "csv_valid_rows": st.get("csv_valid_rows"),
                                   "C1_max_abs_delta": st.get("C1_max_abs_delta"), "summary_x100_4dp": st.get("summary_x100_4dp")}
    else:
        st = {}
        gates["harness_status"] = {"ok": True, "note": "named baseline: its status is banked in baseline/*_score_status.json / RESULT_SFT4.md"}
    rep = json.load(open(arm_s["report"], encoding="utf-8")) if arm_s.get("report") and Path(arm_s["report"]).exists() else None
    brep = json.load(open(base_s["report"], encoding="utf-8")) if base_s.get("report") else None
    if rep is not None:
        sg = {"rule": rep.get("rule"), "ckpt": rep.get("ckpt"), "rows": rep.get("rows"), "misses": rep.get("misses"),
              "tokens_asked": rep.get("tokens_asked"), "seconds": rep.get("seconds"),
              "repair_last_heading": rep.get("repair_last_heading"), "sanitize_goal": rep.get("sanitize_goal"),
              "goal_fix": rep.get("goal_fix"), "select": rep.get("select"), "frame_control": rep.get("frame_control")}
        okrule = (a.expect_rule is None) or (rep.get("rule") == a.expect_rule)
        same_cfg, diffs = True, []
        if brep is not None:
            for k in ("select", "repair_last_heading", "sanitize_goal", "goal_fix"):
                if k in rep and k in brep and rep[k] != brep[k]:
                    same_cfg = False
                    diffs.append({k: [rep[k], brep[k]]})
        sg.update({"rule_ok": okrule, "expected_rule": a.expect_rule, "seam_settings_equal_to_base": same_cfg, "seam_setting_diffs": diffs,
                   "frame_control_equal_to_base": (None if not (brep and brep.get("frame_control") and rep.get("frame_control"))
                                                   else brep["frame_control"] == rep["frame_control"])})
        # the frame control compares the LOG's own future with the export's -- no planner is involved -- so on the full set it must
        # equal u400's bit for bit: the declared C1 set (36 tokens / 4 logs, max 6.18063877792441 m, RESULT_SFT4.md Deviation 1) and
        # nothing else. A different value means the seam environment or the frames moved, and the read is void.
        fc_ok = bool(a.dry_run or sg["frame_control_equal_to_base"] in (True, None))
        sg["frame_control_gate_ok"] = fc_ok
        sg["ok"] = bool(rep.get("misses") == 0 and rep.get("rows") == len(toks) and okrule and same_cfg and fc_ok)
        gates["seam_report"] = sg
    else:
        gates["seam_report"] = {"ok": bool(arm_s.get("named_baseline")), "note": "no seam report found" if not arm_s.get("named_baseline") else "named baseline"}

    # the checkpoint the seam ran must be the one whose sha256 was checked against P2's summary.json
    if a.ckpt:
        c = {"path": a.ckpt, "bytes": Path(a.ckpt).stat().st_size, "sha256": sha256_file(a.ckpt)}
        norm = lambda x: str(x).replace("\\", "/").lower()
        c["seam_report_ckpt"] = rep.get("ckpt") if rep else None
        c["seam_ckpt_path_matches"] = bool(rep and norm(rep.get("ckpt")) == norm(a.ckpt))
        c["sha256_matches_expected"] = (None if not a.expect_ckpt_sha256 else c["sha256"] == a.expect_ckpt_sha256.lower())
        c["ok"] = bool(c["seam_ckpt_path_matches"] and c["sha256_matches_expected"] in (True, None))
        gates["checkpoint"] = c
        res["checkpoint"] = c
    # tables and the pairing
    arm, base = load_table(arm_s["csv"]), load_table(base_s["csv"])
    sel = [t for t in toks if t in arm and t in base]
    gates["rows"] = {"ok": len(sel) == len(toks) and set(arm) == set(toks) and set(toks) <= set(base), "scored": len(sel), "asked": len(toks),
                     "arm_rows": len(arm), "base_rows": len(base)}
    gates["consistency_vs_base"] = consistency(arm_s["seam"], base_s["seam"], arm, base, sel)
    also = r2b = None
    if also_s:
        also = load_table(also_s["csv"])
        r2b = read_pair(arm, also, sel, tl)
        # cross-instrument control: stage2_sft.py (R1's instrument, run on this same arm) pairs the arm with the deployed run
        # through the same estimator; where its result.json exists, this script's R2b must equal it BIT FOR BIT
        s2r = arm_s.get("stage2_result")
        if a.also == "deployed" and s2r is not None and Path(s2r).exists():
            pr = json.load(open(s2r, encoding="utf-8")).get("primary", {})
            gates["r2b_equals_stage2_primary"] = {"ok": bool(pr and pr.get("mean") == r2b["primary"]["mean"] and pr.get("ci95") == r2b["primary"]["ci95"]),
                                                  "stage2": [pr.get("mean"), pr.get("ci95")], "stage3": [r2b["primary"]["mean"], r2b["primary"]["ci95"]]}
    gates_ok = all(g.get("ok", False) for g in gates.values())
    res["gates"], res["gates_ok"] = gates, bool(gates_ok)

    # the harness summary must be reproduced by the CSV columns (the instrument reads the right columns)
    summ = st.get("summary_x100_4dp") if st else None
    if summ:
        mine = {k: round(float(np.mean([arm[t][k] for t in sel])), 4) for k in ("PDMS", *SUBS)}
        res["csv_vs_harness_summary"] = {"harness": summ, "csv_means": mine,
                                          "equal_4dp": all(abs(mine[k] - summ[k]) < 5e-5 + 1e-9 for k in mine)}

    r2 = read_pair(arm, base, sel, tl)
    dec = decide(r2["primary"]["ci95"][0], r2["primary"]["ci95"][1], r2["adverse_subscores"], gates_ok and not a.dry_run)
    if a.dry_run:
        dec["outcome"] = "DRYRUN (not a read): would read " + decide(r2["primary"]["ci95"][0], r2["primary"]["ci95"][1],
                                                                    r2["adverse_subscores"], True)["outcome"]
        dec["p6"] = "NONE (dry run)"
    # families
    fam_arm = None
    if arm_s.get("families") and Path(arm_s["families"]).exists():
        fam_arm = json.load(open(arm_s["families"], encoding="utf-8"))
    elif not a.no_families:
        fam_arm = compute_families(arm_s["seam"], f"{a.name}", out_dir, set(sel))
    fam_base = json.load(open(base_s["families"], encoding="utf-8")) if base_s.get("families") else None
    r2["families_vs_base"] = families_vs(fam_base, fam_arm)
    res["R2"] = {"vs": base_s["label"], **r2, **dec}
    if a.base == "u400" and base_s["pin_ok"]:
        res["R2"]["sft4_B_u400_pdms_x100_banked"] = BASELINES["u400"]["pdms_x100"]
    # sensitivity: minus the 36 C1 tokens (RESULT_SFT4.md Deviation 1; reported, not gated)
    if C1:
        selc = [t for t in C1["tokens"] if t in arm and t in base]
        res["R2_minus_C1"] = {"note": "reported, not gated (RESULT_SFT4.md Deviation 1)", "tokens_md5": C1["gate"]["md5"],
                              **read_pair(arm, base, selc, C1["token_log"])}
    # R2b
    if also_s:
        fam_also = json.load(open(also_s["families"], encoding="utf-8")) if also_s.get("families") else None
        r2b["families_vs_also"] = families_vs(fam_also, fam_arm)
        r2b["consistency_vs_also"] = consistency(arm_s["seam"], also_s["seam"], arm, also, sel)
        res["R2b"] = {"vs": also_s["label"], "status": "REPORTED, decides nothing (amendment section 3)", **r2b}
    # the tactical rows are reported beside the intervals (no interval exists for them)
    res["tactical_vs_base"] = res["R2"]["families_vs_base"].get("tactical_no_interval")
    res["seconds"] = round(time.time() - t_start, 1)
    dump(res, out_dir / "r2_result.json")
    p = res["R2"]["primary"]
    print("ZZSTAGE3", a.name, res["R2"]["outcome"], f"{p['mean']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]",
          "adverse=" + (",".join(res["R2"]["adverse_subscores"]) or "none"), "gates_ok=" + str(gates_ok), flush=True)
    return res


# ---------------------------------------------------------------- the instrument precondition (amendment section 3)
SWAP_TOL = 1e-12


def check_selfpair(rd: dict) -> dict:
    p = rd["primary"]
    zeros = all(v["mean"] == 0.0 and v["ci95"] == [0.0, 0.0] for v in rd["subscores"].values())
    ok = (p["mean"] == 0.0 and p["ci95"] == [0.0, 0.0] and p["better"] == 0 and p["worse"] == 0 and p["tied"] == p["n"] and zeros)
    return {"ok": bool(ok), "mean": p["mean"], "ci95": p["ci95"], "better_worse_tied": [p["better"], p["worse"], p["tied"]],
            "n": p["n"], "n_logs": p["n_logs"], "subscores_all_exactly_zero": bool(zeros)}


def check_swap(fwd: dict, rev: dict) -> dict:
    """arm<->base swapped: sign flips; point estimate bit-exact, CI ends exact to SWAP_TOL (see the note: np.percentile's
    interpolation is not bitwise antisymmetric, MEASURED -- a 1-ulp difference on ONE CI end of the real data)"""
    def one(f, r):
        pt = r["mean"] == -f["mean"]
        d_lo, d_hi = abs(r["ci95"][0] + f["ci95"][1]), abs(r["ci95"][1] + f["ci95"][0])
        return {"point_bitwise": bool(pt), "ci_bitwise": bool(d_lo == 0.0 and d_hi == 0.0), "ci_max_abs_dev": max(d_lo, d_hi),
                "counts_swapped": bool(r["better"] == f["worse"] and r["worse"] == f["better"] and r["tied"] == f["tied"]),
                "ok": bool(pt and max(d_lo, d_hi) <= SWAP_TOL and r["better"] == f["worse"] and r["worse"] == f["better"])}
    per = {"PDMS": one(fwd["primary"], rev["primary"])}
    for k in SUBS:
        per[k] = one(fwd["subscores"][k], rev["subscores"][k])
    return {"ok": all(v["ok"] for v in per.values()), "tolerance": SWAP_TOL, "per_quantity": per,
            "forward_primary": [fwd["primary"]["mean"], *fwd["primary"]["ci95"]],
            "reverse_primary": [rev["primary"]["mean"], *rev["primary"]["ci95"]]}


def check_tokens_bytes(raw: bytes, pins: dict, export_logs: dict) -> dict:
    d = json.loads(raw.decode("utf-8"))
    toks = d["tokens"]
    g = {"md5": md5_bytes(raw), "n": len(toks), "n_logs": len({export_logs[t] for t in toks if t in export_logs})}
    g["md5_ok"], g["n_ok"], g["logs_ok"] = g["md5"] == pins["md5"], g["n"] == pins["n"], g["n_logs"] == pins["logs"]
    g["ok"] = bool(g["md5_ok"] and g["n_ok"] and g["logs_ok"])
    return g


def shuffled(tbl: dict, seed: int = 7) -> dict:
    """the deliberate regression: every token receives ANOTHER token's whole row (the token -> score mapping is broken, the
    multiset of scores is not)"""
    toks = sorted(tbl)
    perm = np.random.default_rng(seed).permutation(len(toks))
    return {toks[i]: copy.deepcopy(tbl[toks[int(perm[i])]]) for i in range(len(toks))}


def selfcheck(a) -> int:
    out = PKG / "raw" / "p2_r2_instrument"
    out.mkdir(parents=True, exist_ok=True)
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    export_logs = {t: v["log_name"] for t, v in exp.items()}
    u, d = pick_baseline("u400"), pick_baseline("deployed")
    R = {"what": "instrument precondition for R2 (PREREG_P2_AMENDMENT_1.md section 3), run on the real 12,146-token tables",
         "estimator": ESTIMATOR, "tier": TIER, "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "stage2_sft_md5": md5_file(STAGE2_SFT), "stage2_sft_unmodified": md5_file(STAGE2_SFT) == STAGE2_SFT_MD5}
    tok_path = first_matching([DATA / "navtest_full_tokens.json", BANK / "navtest_full_tokens.json"], FULL_PINS["md5"])[0]
    T = load_tokens(tok_path, FULL_PINS, export_logs)
    toks, tl = T["tokens"], T["token_log"]
    U, Dp = load_table(u["csv"]), load_table(d["csv"])
    checks, regress = {}, {}

    # --- check 3 (provenance): token file md5, and u400's per-token table md5 RECORDED
    raw = Path(tok_path).read_bytes()
    checks["token_file_md5"] = {**check_tokens_bytes(raw, FULL_PINS, export_logs), "expected_prefix_in_RESULT_SFT4": "3a8b3653",
                                "path": str(tok_path), "gate_all_in_export_and_unique": T["gate"]["ok"]}
    # R2's inference run (stage2_sft.py --tokens <this file>) takes the WHOLE-FILTER scoring path iff the token set equals the export's:
    # that omits --tokens (a 230,812-char Hydra override dies on Windows' 32,767 limit, RESULT_SFT4.md Deviation 2) and makes the harness's
    # C2 guard demand that all 12,146 tokens score. Same predicate as stage2_sft.py line ~144.
    checks["r2_takes_whole_filter_path"] = {"ok": bool(set(toks) == set(export_logs)), "n_tokens": len(toks), "n_export": len(export_logs)}
    checks["u400_table_recorded"] = {"csv_md5": u["csv_md5"], "csv_sha256": sha256_file(u["csv"]), "csv_rows": len(U),
                                     "csv_path": str(u["csv"]), "seam_md5": u["seam_md5"], "seam_sha256": sha256_file(u["seam"]),
                                     "tokens_equal_token_file": set(U) == set(toks), "pin_ok": u["pin_ok"],
                                     "ok": bool(u["pin_ok"] and len(U) == FULL_PINS["n"] and set(U) == set(toks))}
    # the pinned table IS the one behind the banked result: its means reproduce the harness summary, and its pairing reproduces result.json
    status = json.load(open(BANK / "u400_navtest_full_score_status.json", encoding="utf-8"))
    mine = {k: round(float(np.mean([U[t][k] for t in toks])), 4) for k in ("PDMS", *SUBS)}
    checks["u400_csv_reproduces_harness_summary"] = {"harness": status["summary_x100_4dp"], "csv_means": mine,
                                                     "ok": all(abs(mine[k] - status["summary_x100_4dp"][k]) < 5e-5 + 1e-9 for k in mine)}

    # --- the families6 fallback (compute_families, repo copy): reproduces u400's BANKED families block exactly. Stage 2 normally supplies the
    # arm's families from its own hard-coded scratch copy of families6; this proves the repo copy is interchangeable if that copy is gone.
    import tempfile
    f6 = resolve_families6()
    if f6 is None:
        checks["families6_reproduces_banked_u400"] = {"ok": False, "note": "no families6.py found"}
    else:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            fam = compute_families(u["seam"], "u400_repro", Path(td), set(toks))
        banked_fam = json.load(open(u["families"], encoding="utf-8"))
        diffs = tree_diff(fam["families"], banked_fam["families"]) if fam else ["no output"]
        checks["families6_reproduces_banked_u400"] = {"ok": not diffs, "families6": str(f6), "families6_md5": md5_file(f6),
                                                      "differences_in_families_block": len(diffs), "first_differences": diffs[:5],
                                                      "seconds": round(time.time() - t0)}

    # --- check 1 (self-pair), through the SAME read_pair AND the full pipeline
    selfpair = read_pair(U, load_table(u["csv"]), toks, tl)          # two independent loads of the same file
    checks["self_pair_u400"] = check_selfpair(selfpair)

    # --- check 2 (swap) + the known value: u400 - deployed must reproduce RESULT_SFT4.md's result.json to the bit
    fwd = read_pair(U, Dp, toks, tl)
    rev = read_pair(Dp, U, toks, tl)
    banked = {"mean": 3.7908187056340563, "ci95": [3.0470289517787528, 4.542726430876108], "better": 5167, "worse": 3188, "tied": 3791,
              "zero_deployed": 837, "zero_arm": 529, "pdms_deployed": 84.45612239059116, "pdms_arm": 88.24694109622521,
              "source": "raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/result.json (primary)"}
    fp = fwd["primary"]
    checks["known_value_u400_vs_deployed"] = {
        "banked": banked, "recomputed": {"mean": fp["mean"], "ci95": fp["ci95"], "better": fp["better"], "worse": fp["worse"], "tied": fp["tied"],
                                         "zero_deployed": fp["zero_base"], "zero_arm": fp["zero_arm"], "pdms_deployed": fp["base_mean_x100"],
                                         "pdms_arm": fp["arm_mean_x100"]},
        "ok": bool(fp["mean"] == banked["mean"] and fp["ci95"] == banked["ci95"] and [fp["better"], fp["worse"], fp["tied"]] == [5167, 3188, 3791]
                   and fp["zero_base"] == 837 and fp["zero_arm"] == 529 and fp["base_mean_x100"] == banked["pdms_deployed"]
                   and fp["arm_mean_x100"] == banked["pdms_arm"])}
    checks["swap_flips_sign"] = check_swap(fwd, rev)

    # --- the decision table on the real tables: three of its four rows without a synthetic input
    d_fwd = decide(fp["ci95"][0], fp["ci95"][1], fwd["adverse_subscores"], True)
    d_rev = decide(rev["primary"]["ci95"][0], rev["primary"]["ci95"][1], rev["adverse_subscores"], True)
    d_self = decide(selfpair["primary"]["ci95"][0], selfpair["primary"]["ci95"][1], selfpair["adverse_subscores"], True)
    hyb = copy.deepcopy(U)                                            # u400's PDMS and sub-scores, EXCEPT ego progress made worse than deployed's
    for t in hyb:
        hyb[t]["EP"] = max(Dp[t]["EP"] - 10.0, 0.0)
    rd_h = read_pair(hyb, Dp, toks, tl)
    d_h = decide(rd_h["primary"]["ci95"][0], rd_h["primary"]["ci95"][1], rd_h["adverse_subscores"], True)
    d_inv = decide(fp["ci95"][0], fp["ci95"][1], [], False)
    rows = {"u400_vs_deployed": {"outcome": d_fwd["outcome"], "expect": "GO", "adverse": fwd["adverse_subscores"]},
            "deployed_vs_u400": {"outcome": d_rev["outcome"], "expect": "NO-GO (P2 below u400)", "adverse": rev["adverse_subscores"]},
            "u400_vs_u400": {"outcome": d_self["outcome"], "expect": "NO-GO (NOT PROVEN)", "adverse": selfpair["adverse_subscores"]},
            "u400_with_worse_EP_vs_deployed": {"outcome": d_h["outcome"], "expect": "NO-GO (adverse sub-score)", "adverse": rd_h["adverse_subscores"],
                                               "primary": [rd_h["primary"]["mean"], *rd_h["primary"]["ci95"]]},
            "gate_failed": {"outcome": d_inv["outcome"], "expect": "INVALID"}}
    checks["decision_table_rows"] = {"rows": rows, "ok": all(v["outcome"] == v["expect"] for v in rows.values())
                                     and rows["u400_with_worse_EP_vs_deployed"]["adverse"] == ["EP"]}

    # --- DELIBERATE REGRESSIONS: each must make its check FAIL, else the check proves nothing
    sh = shuffled(U)
    sp = check_selfpair(read_pair(sh, U, toks, tl))
    regress["shuffled_arm_self_pair"] = {"what": "shuffle the arm's token->score mapping, then run the self-pair check", "check_result": sp,
                                         "required_to_fail": True, "failed_as_required": not sp["ok"]}
    badrev = read_pair(Dp, U, toks, tl, seed=1)                        # a reverse read with a different resampling seed
    sw = check_swap(fwd, badrev)
    regress["swap_with_wrong_seed"] = {"what": "reverse read with seed 1 instead of 20260927", "check_result": {"ok": sw["ok"], "per_PDMS": sw["per_quantity"]["PDMS"]},
                                       "required_to_fail": True, "failed_as_required": not sw["ok"]}
    cut = json.dumps({"tokens": toks[:-1]}).encode("utf-8")
    tg = check_tokens_bytes(cut, FULL_PINS, export_logs)
    regress["token_file_one_token_short"] = {"what": "token file with the last token dropped", "check_result": tg, "required_to_fail": True,
                                             "failed_as_required": not tg["ok"]}
    wrong = {"csv_md5": d["csv_md5"], "pinned": BASELINES["u400"]["csv_md5"]}
    regress["deployed_table_passed_as_u400"] = {"what": "the deployed CSV where the u400 pin is checked", "check_result": wrong,
                                                "required_to_fail": True, "failed_as_required": wrong["csv_md5"] != wrong["pinned"]}
    regress["_note"] = "a failed check here is a PASS of the instrument: the check can fail"

    # --- the full pipeline, twice (exercises argparse, gates, families, JSON): u400 vs u400 and deployed vs u400
    pipe = {}
    for tag, arm_from, base, also, rule in (("selfpair_u400", "u400", "u400", "deployed", "navsim_v1_lane"),
                                            ("swapped_deployed_vs_u400", "deployed", "u400", "u400", "navsim_v1")):
        ns = argparse.Namespace(name=tag, arm_from=arm_from, base=base, also=also, expect_rule=rule, tokens=None, dry_run=False,
                                no_families=False, out_dir=str(out / tag), arm_csv=None, arm_seam=None, arm_report=None,
                                arm_families=None, arm_status=None, ckpt=None, expect_ckpt_sha256=None)
        pipe[tag] = run_r2(ns)
    sp_pipe = pipe["selfpair_u400"]
    rv_pipe = pipe["swapped_deployed_vs_u400"]
    checks["pipeline_self_pair"] = {"gates_ok": sp_pipe["gates_ok"], "R2": check_selfpair(sp_pipe["R2"]), "outcome": sp_pipe["R2"]["outcome"],
                                    "families_adverse": sp_pipe["R2"]["families_vs_base"].get("adverse"),
                                    "R2b_is_the_known_value": [sp_pipe["R2b"]["primary"]["mean"], *sp_pipe["R2b"]["primary"]["ci95"]],
                                    "R2b_equals_banked": bool(sp_pipe["R2b"]["primary"]["mean"] == banked["mean"] and sp_pipe["R2b"]["primary"]["ci95"] == banked["ci95"])}
    checks["pipeline_self_pair"]["ok"] = bool(checks["pipeline_self_pair"]["gates_ok"] and checks["pipeline_self_pair"]["R2"]["ok"]
                                              and checks["pipeline_self_pair"]["R2b_equals_banked"]
                                              and checks["pipeline_self_pair"]["families_adverse"] == [])
    pw = check_swap(sp_pipe["R2b"], rv_pipe["R2"])
    checks["pipeline_swap"] = {"gates_ok": rv_pipe["gates_ok"], "swap": {k: pw[k] for k in ("ok", "tolerance", "forward_primary", "reverse_primary")},
                               "ok": bool(rv_pipe["gates_ok"] and pw["ok"])}

    # --- the pipeline REFUSES a defective arm: the same plumbing with one row of the arm's table dropped must read INVALID, never an outcome
    with tempfile.TemporaryDirectory() as td:
        raw_csv = Path(d["csv"]).read_bytes()
        short = Path(td) / "arm_short.csv"
        # the arm's table minus its last TOKEN row. ⚠️ the CSV's physical last line is the harness's `average` row, which read_csv skips:
        # dropping THAT line (this regression's first draft did) removes no token, the gate rightly stays green, and the regression is void
        lines = raw_csv.splitlines(keepends=True)
        last_tok = max(i for i, ln in enumerate(lines) if i > 0 and ln.split(b",")[1] != b"average")
        short.write_bytes(b"".join(ln for i, ln in enumerate(lines) if i != last_tok))
        common = dict(arm_from=None, base="u400", also=None, expect_rule="navsim_v1", tokens=None, dry_run=False, no_families=True,
                      arm_seam=str(d["seam"]), arm_report=str(d["report"]), arm_families=None,
                      arm_status=str(BANK / "u400_navtest_full_score_status.json"), ckpt=None, expect_ckpt_sha256=None)
        ctl = run_r2(argparse.Namespace(name="control_whole_arm", out_dir=str(Path(td) / "ok"), arm_csv=str(d["csv"]), **common))
        bad = run_r2(argparse.Namespace(name="regress_arm_one_row_short", out_dir=str(Path(td) / "bad"), arm_csv=str(short), **common))
    regress["arm_table_one_row_short_through_the_pipeline"] = {
        "what": "the pipeline with the arm's CSV minus its last row (control: the same plumbing with the whole CSV)",
        "check_result": {"control_outcome": ctl["R2"]["outcome"], "control_gates_ok": ctl["gates_ok"], "defective_outcome": bad["R2"]["outcome"],
                         "defective_gates_ok": bad["gates_ok"], "defective_failed_gates": [k for k, v in bad["gates"].items() if not v.get("ok")]},
        "required_to_fail": True,
        "failed_as_required": bool(ctl["gates_ok"] and ctl["R2"]["outcome"] != "INVALID" and not bad["gates_ok"] and bad["R2"]["outcome"] == "INVALID")}

    R["checks"], R["regressions"] = checks, regress
    R["all_checks_ok"] = all(v["ok"] for v in checks.values())
    R["all_regressions_failed_as_required"] = all(v["failed_as_required"] for k, v in regress.items() if not k.startswith("_"))
    R["R2_PRECONDITION_MET"] = bool(R["all_checks_ok"] and R["all_regressions_failed_as_required"] and R["stage2_sft_unmodified"])
    dump(R, out / "selfcheck.json")
    print("ZZSELFCHECK", "checks_ok=" + str(R["all_checks_ok"]), "regressions_failed_as_required=" + str(R["all_regressions_failed_as_required"]),
          "R2_PRECONDITION_MET=" + str(R["R2_PRECONDITION_MET"]), flush=True)
    return 0 if R["R2_PRECONDITION_MET"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true", help="the instrument precondition (amendment section 3); banks raw/p2_r2_instrument/")
    ap.add_argument("--name", default=None, help="the stage-2 run name of the arm (stage2_sft.py --name)")
    ap.add_argument("--base", default="u400", choices=sorted(BASELINES), help="R2's comparator; default u400, md5-pinned")
    ap.add_argument("--also", default="deployed", choices=sorted(BASELINES), help="R2b's comparator (reported, deciding nothing)")
    ap.add_argument("--no-also", action="store_true")
    ap.add_argument("--expect-rule", default="navsim_v1", help="the selection rule the arm's seam must record (P2: navsim_v1)")
    ap.add_argument("--tokens", default=None, help="default: navtest_full_tokens.json (md5 3a8b3653...)")
    ap.add_argument("--ckpt", default=None, help="the arm's checkpoint: its sha256 is recorded")
    ap.add_argument("--expect-ckpt-sha256", default=None)
    ap.add_argument("--dry-run", action="store_true", help="an entry-point check on a small token set: pins relaxed, never a verdict")
    ap.add_argument("--no-families", action="store_true")
    ap.add_argument("--arm-from", default=None, choices=sorted(BASELINES), help=argparse.SUPPRESS)
    ap.add_argument("--arm-csv"), ap.add_argument("--arm-seam"), ap.add_argument("--arm-report")
    ap.add_argument("--arm-families"), ap.add_argument("--arm-status")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck(a)
    if not a.name:
        ap.error("--name is required (or --selfcheck)")
    if a.no_also:
        a.also = None
    res = run_r2(a)
    return 0 if res["gates_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
