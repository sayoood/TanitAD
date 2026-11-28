#!/usr/bin/env python3
"""navtest_E: IMPL_PLAN step 9 of PREREG_SAFEOBJ -- the navtest T1 read of the four u1000 scorer states (R0d-A, R0d-B, R1d-A, R1d-B), written as the `navtest_E.json` that
`outcome/safeobj_outcome.py` bar E consumes (OUTCOME_BUILD.md section 7), plus `navtest_families.json` (the four metric families per pair).  CPU only; no pod contact from this file.

    python navtest_E.py plan   [--arms-root /workspace/data/rls1/arms] [--work W] [--remote /workspace/ops/sobjE]      -> prints the exact pod + dev-box command sequence (PLAN_E.md)
    python navtest_E.py expected-states --state R0d_A=<scorer_states_u1000.pt>:A --state R0d_B=...:B --state R1d_A=...:A --state R1d_B=...:B --out W/EXPECTED_STATES_E.json
    python navtest_E.py seams    --work W --chunks W/out [--limit N --present-only]         -> W/seam_<STATE>.npz (+ seam_BASE), W/picks_report_E.json     (the free-point picker, state-generic)
    python navtest_E.py families --work W                                                    -> W/families/sobj_<STATE>_families.json   (families6, the free-point call)
    python navtest_E.py build    --work W --irep-json F --out-dir O
                                                                                             -> O/navtest_E.json  O/navtest_E.provenance.json  O/navtest_families.json
    python navtest_E.py irep     --a csvA --b csvB --out irep.json                           -> {"irep_abs_delta", "unit": "PDMS x100"} from two harness tables of ONE state read twice
    python navtest_E.py selftest                                                             -> the known-value anchors (RESULT_SFT4 to 1e-12), no files written

WHAT THE FILE IS (all values x100, B - A per pair, T1 = NAVSIM v1.1 PDMS on the full navtest, 12,146 tokens / 136 logs):
  pairs.<R0d|R1d>.<PDMS|NC|DAC|EP|TTC|C|DDC> = {"mean", "ci95"}   paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927  (stage3_full.read_pair -> proxy_eval.boot, the estimator of
                                                                   every banked navtest read: it is IMPORTED, not re-implemented)
  pooled.PDMS = the mean of the two pairs' per-token PDMS contrasts, its interval from the SAME estimator on that mean series; pooled.resample = "joint".  Because proxy_eval.boot re-seeds
                identically per call and resamples LOGS, bootstrapping the per-token mean of the two contrasts IS the joint resample of PREREG addendum 1 R4 (one index set per draw, applied to
                both pairs): the mean over a resampled token multiset of (d0 + d1) / 2 equals (mean d0 + mean d1) / 2 over that SAME multiset.  The test pins this against an explicit
                one-index-set-per-draw loop and against the independent-draws alternative (which it must NOT equal).
  n_tokens / n_logs = 12,146 / 136 (the pinned navtest token set, the same for both pairs).
  floors.irep_abs_delta = an EXPLICIT input (`--irep-json`); there is NO default, because a defaulted 0 silently weakens E clause 3.
  There is NO seed floor in the file: since the 14:15 revision of OUTCOME_BUILD section 7 the READER derives it (RL-val |R0d-R1d| and |R0-R1o| from bar L's inputs, and the navtest
  |R0d-R1d|) and flags a `floors.seed_floor_abs` as IGNORED.  The navtest part is recorded in the sidecar for cross-checking.

THE CONTRACT IS EXACT: navtest_E.json carries ONLY tier / estimator / n_tokens / n_logs / pairs / pooled / floors.  Everything else (input hashes, gates, levels, counts) is in
`navtest_E.provenance.json`.
FAIL CLOSED: a failed hard gate, a token set that is not the pinned 12,146 / 136, a x1-scale table, a sub-score missing, a CI out of order, an unreadable floor -> exit 2 and NO navtest_E.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True                      # the free-point tree C:/fp/PK is READ-ONLY for this tool: no .pyc is ever written into it
HERE = os.path.dirname(os.path.abspath(__file__))
FPDIR = os.path.join(HERE, "fp")
if FPDIR not in sys.path:
    sys.path.insert(0, FPDIR)
import fp_read as FPR  # noqa: E402  (COPY of the free-point reader: load_s3, KNOWN, DATA, EXPORT, FAM_KEYS, fam_summary)

TOOL = "navtest_E/1"
TIER = "T1"
N_BOOT, SEED = 10000, 20260927
ESTIMATOR = f"paired log-cluster bootstrap, {N_BOOT} resamples, 95 %, seed {SEED}"          # OUTCOME_BUILD.md section 7, character for character
PAIRS = ("R0d", "R1d")
POOLED_RESAMPLE = "joint"                                                      # OUTCOME_BUILD section 7: ONE index set of the 136 logs per draw, applied to BOTH pairs
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
KEYS = ("PDMS",) + SUBS
PAIR_STATES = {"R0d": ("R0d_A", "R0d_B"), "R1d": ("R1d_A", "R1d_B")}          # (control A, treated B): the contrast is ALWAYS B - A
STATES = tuple(s for p in PAIRS for s in PAIR_STATES[p])
LABEL_PREFIX = "refe_sobj_"                                                   # the harness refuses any label that does not start with "refe"
EXPECT_UPDATE = 1000
UNIT = "PDMS x100"
PK = FPR.PK
DATA = FPR.DATA
EXPORT = FPR.EXPORT
TOKENS_FILE = f"{DATA}/navtest_full_tokens.json"
FULL_PINS = {"md5": "3a8b3653fd70891e50a230a37eb76ae8", "n": 12146, "logs": 136}
INSTRUMENT_MD5 = {"stage3_full.py": "a501ef4eac66f58b9e6115f61ef48ba5", "proxy_eval.py": "d177f0c9ab493390cb2b5bb8811c7cd3"}      # MEASURED 2026-10-17 on C:/fp/PK (git archive 19785c8)
LIVE_REMOTE = "/workspace/ops/freept"                                         # the free-point run's pod directory: this tool must never use it
DEFAULT_REMOTE = "/workspace/ops/sobjE"
KAPPA_FALL_MAX = 0.03                                                          # PREREG_RL_S1 section 4: tactical decision kappa may not fall by > 0.03 vs the control
ARM_KEY = "refcv6"                                                             # families6's arm key (proxy_eval.fam_adverse default)
STRATEGIC = {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE item 25 (the strategic layer is switched off temporarily)",
             "note": "reported as PAUSED: never a pass, never omitted; also UNAVAILABLE in NAVSIM (no strategic decision is scored)"}

_S3 = None


class Refused(Exception):
    """a hard refusal: nothing is written"""


def get_s3():
    """stage3_full via the free-point loader (read-only import of C:/fp/PK).  The two instrument files are pinned by md5."""
    global _S3
    if _S3 is None:
        _S3 = FPR.load_s3()
    return _S3


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def instrument_pins() -> dict:
    got = {n: md5_file(f"{PK}/eval/{n}") for n in INSTRUMENT_MD5}
    return {"pinned": INSTRUMENT_MD5, "got": got, "ok": got == INSTRUMENT_MD5}


# ============================================================ tables
def label_of(state: str) -> str:
    return LABEL_PREFIX + state


def table_path(state: str, root: str | None = None) -> str:
    lab = label_of(state)
    return f"{root or DATA + '/score'}/{lab}/{lab}.csv"


def load_tables(S3, paths: dict) -> dict:
    """state -> {token: {PDMS, NC, DAC, EP, TTC, C, DDC}} all x100 (stage3_full.load_table)."""
    miss = [s for s in STATES if s not in paths]
    if miss:
        raise Refused(f"tables missing for state(s) {miss}: the read needs exactly {list(STATES)}")
    out = {}
    for s in STATES:
        if not os.path.exists(paths[s]):
            raise Refused(f"{s}: harness table not found: {paths[s]}")
        out[s] = S3.load_table(paths[s])
    return out


def unit_guard(name: str, tab: dict) -> dict:
    """x1 vs x100 slip guard.  A NAVSIM score is in [0, 1]; the programme's tables are x100.  A table whose PDMS never exceeds 1 is on the wrong scale (every contrast would be 100x
    too small and bar E would read it as noise), and one above 100 is on a scale too large."""
    v = np.array([r["PDMS"] for r in tab.values()], dtype=np.float64)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise Refused(f"{name}: an empty table or a non-finite PDMS")
    lo, hi = float(v.min()), float(v.max())
    if hi <= 1.0 + 1e-9:
        raise Refused(f"{name}: UNIT SLIP -- the largest PDMS is {hi:.6g} <= 1: this table is on the x1 scale, the contract is x100")
    if hi > 100.0 + 1e-6 or lo < -1e-9:
        raise Refused(f"{name}: PDMS outside [0, 100] (min {lo:.6g}, max {hi:.6g}): not a x100 NAVSIM score")
    for k in SUBS:
        w = np.array([r[k] for r in tab.values()], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.min() < -1e-9 or w.max() > 100.0 + 1e-6:
            raise Refused(f"{name}: sub-score {k} outside [0, 100] or non-finite")
    return {"pdms_mean_x100": float(v.mean()), "pdms_max": hi, "n": int(v.size)}


def token_gate(S3, tokens_file: str | None, pins: dict | None, export: dict | None):
    """-> (toks, tl, gate).  pins=None is a DRY RUN / synthetic-fixture path and is stamped as such."""
    if tokens_file is None:
        raise Refused("no token file")
    t = S3.load_tokens(__import__("pathlib").Path(tokens_file), pins, export)
    if not t["gate"]["ok"]:
        raise Refused(f"the token set is not the pinned navtest set: {json.dumps(t['gate'])[:300]}")
    return t["tokens"], t["token_log"], t["gate"]


def same_tokens(tabs: dict, toks: list) -> None:
    for s, t in tabs.items():
        miss = [x for x in toks if x not in t]
        if miss:
            raise Refused(f"{s}: {len(miss)} of the {len(toks)} navtest tokens have no row (first {miss[:2]})")


def nondegenerate(tabs: dict, toks: list) -> dict:
    """a pair whose B table is token-for-token its A table reads +0.00 everywhere (the worst false pass: one arm read twice); two pairs with identical contrasts are one pair counted twice"""
    out = {}
    for p in PAIRS:
        a, b = PAIR_STATES[p]
        same = all(tabs[a][t][k] == tabs[b][t][k] for t in toks for k in KEYS)
        if same:
            raise Refused(f"pair {p}: table {b} is token-for-token table {a}: the contrast is identically 0 (one arm read twice)")
        out[p] = {"A_equals_B": False}
    d0 = [tabs[PAIR_STATES["R0d"][1]][t]["PDMS"] - tabs[PAIR_STATES["R0d"][0]][t]["PDMS"] for t in toks]
    d1 = [tabs[PAIR_STATES["R1d"][1]][t]["PDMS"] - tabs[PAIR_STATES["R1d"][0]][t]["PDMS"] for t in toks]
    if d0 == d1:
        raise Refused("R0d and R1d have identical per-token PDMS contrasts: one pair counted twice")
    shared = sorted({(a, b) for a in STATES for b in STATES if a < b and all(tabs[a][t][k] == tabs[b][t][k] for t in toks for k in KEYS)})
    out["tables_identical_across_pairs"] = [list(x) for x in shared]          # disclosed, not refused: two seeds may legitimately share a control table in a fixture
    return out


# ============================================================ the estimator (imported, not re-implemented)
def ci_obj(r: dict) -> dict:
    return {"mean": float(r["mean"]), "ci95": [float(r["ci95"][0]), float(r["ci95"][1])]}


def contrasts(S3, tabs: dict, toks: list, tl: dict, n: int, seed: int) -> dict:
    """per pair: stage3_full.read_pair(B, A) -- PDMS and the six sub-scores, B - A, paired log-cluster bootstrap"""
    out = {}
    for p in PAIRS:
        a, b = PAIR_STATES[p]
        out[p] = S3.read_pair(tabs[b], tabs[a], toks, tl, n, seed)
    return out


def pair_object(rp: dict) -> dict:
    o = {"PDMS": ci_obj(rp["primary"])}
    for k in SUBS:
        o[k] = ci_obj(rp["subscores"][k])
    return o


def pair_delta(tabs: dict, pair: str, key: str, toks: list) -> dict:
    a, b = PAIR_STATES[pair]
    return {t: tabs[b][t][key] - tabs[a][t][key] for t in toks}


def pooled_pdms(S3, tabs: dict, toks: list, tl: dict, n: int, seed: int) -> dict:
    """the pooled PDMS contrast: mean of the two pairs' per-token contrasts, ONE bootstrap over the 136 logs (joint resample, addendum 1 R4)"""
    d0, d1 = pair_delta(tabs, "R0d", "PDMS", toks), pair_delta(tabs, "R1d", "PDMS", toks)
    series = {t: (d0[t] + d1[t]) / 2.0 for t in toks}
    mu, lo, hi = S3.PE.boot(series, tl, n=n, seed=seed)
    return {"mean": float(mu), "ci95": [float(lo), float(hi)]}


# ============================================================ floors (the file carries only the I-rep: the READER derives the seed floors)
def read_irep(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise Refused("floors.irep_abs_delta has NO default: pass --irep-json (a defaulted 0 would silently weaken E clause 3)")
    try:
        d = json.load(open(path, encoding="utf-8"))
        v = d["irep_abs_delta"]
        unit = d["unit"]
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"{path}: not an irep file ({type(e).__name__}: {str(e)[:80]})")
    if unit != UNIT:
        raise Refused(f"{path}: unit {unit!r} is not {UNIT!r}: the contract is x100")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise Refused(f"{path}: irep_abs_delta must be a finite number >= 0, got {v!r}")
    return {"value": float(v), "source": os.path.abspath(path), "sha256": sha256_file(path), "kind": d.get("kind")}


def floors_object(irep: dict) -> dict:
    return {"irep_abs_delta": float(irep["value"])}


# ============================================================ the contract (strict) and its validator
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _ci_ok(x) -> bool:
    return (isinstance(x, dict) and set(x) == {"mean", "ci95"} and _num(x["mean"]) and isinstance(x["ci95"], list) and len(x["ci95"]) == 2
            and all(_num(v) for v in x["ci95"]) and x["ci95"][0] <= x["ci95"][1])


def validate_contract(c, full: bool = True) -> None:
    """navtest_E.json EXACTLY as OUTCOME_BUILD.md section 7 (stricter than the reader: no extra keys).  full=False is for a --dry-run-subset file (n_tokens / n_logs of the subset)."""
    pb = []
    if not isinstance(c, dict) or set(c) != {"tier", "estimator", "n_tokens", "n_logs", "pairs", "pooled", "floors"}:
        raise Refused(f"navtest_E.json must carry exactly tier / estimator / n_tokens / n_logs / pairs / pooled / floors, got {sorted(c) if isinstance(c, dict) else type(c).__name__}")
    if c["tier"] != TIER:
        pb.append(f"tier must be {TIER!r}")
    if c["estimator"] != ESTIMATOR:
        pb.append("estimator is not the registered string")
    for k, want in (("n_tokens", FULL_PINS["n"]), ("n_logs", FULL_PINS["logs"])):
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0 or (full and v != want):
            pb.append(f"{k} must be {want if full else 'a positive int'}, got {v!r}")
    if not isinstance(c["pairs"], dict) or set(c["pairs"]) != set(PAIRS):
        pb.append(f"pairs must be exactly {list(PAIRS)}")
    else:
        for p in PAIRS:
            o = c["pairs"][p]
            if not isinstance(o, dict) or set(o) != set(KEYS):
                pb.append(f"pair {p}: must carry exactly {list(KEYS)} (every sub-score, EP included), got {sorted(o) if isinstance(o, dict) else type(o).__name__}")
                continue
            for k in KEYS:
                if not _ci_ok(o[k]):
                    pb.append(f"pair {p}.{k}: not a {{mean, ci95:[lo<=hi]}} object of finite numbers")
    po = c["pooled"]
    if not isinstance(po, dict) or set(po) != {"PDMS", "resample"} or not _ci_ok(po.get("PDMS")) or po.get("resample") != POOLED_RESAMPLE:
        pb.append(f"pooled must be exactly {{PDMS: {{mean, ci95}}, resample: {POOLED_RESAMPLE!r}}}")
    elif not pb and abs(po["PDMS"]["mean"] - (c["pairs"]["R0d"]["PDMS"]["mean"] + c["pairs"]["R1d"]["PDMS"]["mean"]) / 2.0) > 1e-9:
        pb.append("pooled.PDMS.mean is not the mean of the two pairs' PDMS means")
    fl = c["floors"]
    if not isinstance(fl, dict) or set(fl) != {"irep_abs_delta"} or not (_num(fl.get("irep_abs_delta")) and fl["irep_abs_delta"] >= 0):
        pb.append("floors must be exactly {irep_abs_delta >= 0} (the reader derives the seed floors; a seed_floor_abs in the file is ignored by it)")
    if pb:
        raise Refused("; ".join(pb))


# ============================================================ gates that come from the run (pod + dev-box stages)
def read_run_gates(work: str, expected_states_path: str | None = None, rep_path: str | None = None) -> dict:
    """hard gates -> Refused if any False; soft gates -> status IDENTITY-FAILED.  Reads W/picks_report_E.json (the `seams` stage), W/EXPECTED_STATES_E.json and the harness status files."""
    rep_path = rep_path or f"{work}/picks_report_E.json"
    expected_states_path = expected_states_path or f"{work}/EXPECTED_STATES_E.json"
    hard, soft, notes = {}, {}, {}
    for s in STATES:
        stp = f"{work}/score_logs/{label_of(s)}.status.json"
        try:
            st = json.load(open(stp, encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        hard[f"harness_PASS_{s}"] = bool(st.get("status") == "PASS")
    try:
        rep = json.load(open(rep_path, encoding="utf-8"))
    except (OSError, ValueError):
        rep = None
    try:
        exp = json.load(open(expected_states_path, encoding="utf-8"))
    except (OSError, ValueError):
        exp = None
    if rep is None:
        hard["picks_report_present"] = False
    else:
        hard["picks_report_present"] = True
        hard["IC2_replicated_scorer_path_bit_equal"] = bool(rep.get("IC2_max_abs") == 0.0)
        hard["n_tokens_12146"] = bool(rep.get("n_tokens") == FULL_PINS["n"])
        got = rep.get("state_info") or {}
        if exp is None:
            hard["states_bound_to_expected_hashes"] = False
            notes["states"] = "EXPECTED_STATES_E.json missing: the state -> A/B binding is unverified"
        else:
            ok = all(s in got and s in exp and got[s].get("sha256_tensors") == exp[s].get("sha256_tensors") and got[s].get("update") == EXPECT_UPDATE == exp[s].get("update")
                     and got[s].get("key") == exp[s].get("key") for s in STATES)
            hard["states_bound_to_expected_hashes"] = bool(ok)
            hs = [exp[s]["sha256_tensors"] for s in STATES if s in exp]
            hard["four_states_tensor_distinct"] = bool(len(hs) == len(STATES) and len(set(hs)) == len(hs))
            hard["keys_A_B_as_named"] = bool(all(exp.get(s, {}).get("key") == s[-1] for s in STATES))
        ic3 = rep.get("IC3_BASE_vs_banked_u400") or {}
        soft["IC3_BASE_vs_banked_u400"] = bool(ic3.get("pass"))
        ic5 = rep.get("ic5_forward") or {}
        soft["IC5_forward_bit_identical"] = bool(ic5.get("bit_identical"))
    status = "VALID" if all(hard.values()) and all(soft.values()) else ("INVALID" if not all(hard.values()) else "IDENTITY-FAILED")
    return {"hard": hard, "soft": soft, "status": status, "notes": notes, "fixture": bool((rep or {}).get("FIXTURE"))}


# ============================================================ the build
def build_E(tables: dict, *, tokens_file: str, export_logs: dict | None = None, pins: dict | None = FULL_PINS, irep_path: str,
            run_gates: dict | None = None, n_boot: int = N_BOOT, seed: int = SEED, check_instruments: bool = True) -> tuple:
    """-> (contract, provenance, extras).  Raises Refused; writes nothing."""
    S3 = get_s3()
    if check_instruments:
        ip = instrument_pins()
        if not ip["ok"]:
            raise Refused(f"the estimator instrument changed under us: {ip}")
    if n_boot != N_BOOT or seed != SEED:
        if pins is not None:
            raise Refused("the registered estimator is 10,000 resamples, seed 20260927: a different n/seed is only allowed on a pins=None (dry-run / fixture) path")
    if run_gates is not None and run_gates["status"] == "INVALID":
        raise Refused("a hard gate failed: " + ", ".join(k for k, v in run_gates["hard"].items() if not v))
    exp_logs = export_logs
    if exp_logs is None:
        exp_logs = {t: v["log_name"] for t, v in json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"].items()}
    toks, tl, tgate = token_gate(S3, tokens_file, pins, exp_logs)
    tabs = load_tables(S3, tables)
    same_tokens(tabs, toks)
    levels = {s: unit_guard(s, tabs[s]) for s in STATES}
    nd = nondegenerate(tabs, toks)
    rp = contrasts(S3, tabs, toks, tl, n_boot, seed)
    pairs = {p: pair_object(rp[p]) for p in PAIRS}
    pooled = {"PDMS": pooled_pdms(S3, tabs, toks, tl, n_boot, seed), "resample": POOLED_RESAMPLE}
    irep = read_irep(irep_path)
    full = pins is not None
    contract = {"tier": TIER, "estimator": ESTIMATOR, "n_tokens": len(toks), "n_logs": len(set(tl[t] for t in toks)), "pairs": pairs, "pooled": pooled, "floors": floors_object(irep)}
    validate_contract(contract, full=full)
    fcomp = {"navtest_seed_floor_abs_R0d_vs_R1d": abs(pairs["R0d"]["PDMS"]["mean"] - pairs["R1d"]["PDMS"]["mean"]), "irep_abs_delta": irep["value"],
             "note": "the file carries only the I-rep: the READER derives the RL-val training-seed floor (bar L's inputs) and the navtest seed floor, and takes the max of all three"}
    prov = {"tool": TOOL, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            "status": (run_gates or {}).get("status", "UNGATED (library call: no run gates evaluated)"), "run_gates": run_gates,
            "FIXTURE": ("the gate inputs are stamped FIXTURE: this is NOT a read of the SAFEOBJ states" if (run_gates or {}).get("fixture") else None),
            "pinned_tokens": pins is not None, "token_gate": tgate, "n_boot": n_boot, "seed": seed, "estimator": ESTIMATOR, "tier_text": S3.TIER,
            "instruments": instrument_pins() if check_instruments else None,
            "tables": {s: {"path": os.path.abspath(tables[s]), "sha256": sha256_file(tables[s]), "level_x100": levels[s]} for s in STATES},
            "nondegenerate": nd, "floors_components": fcomp, "irep": irep,
            "pairs_counts": {p: {"better": rp[p]["primary"]["better"], "worse": rp[p]["primary"]["worse"], "tied": rp[p]["primary"]["tied"], "n": rp[p]["primary"]["n"],
                                 "n_logs": rp[p]["primary"]["n_logs"], "zero_pdms_A": rp[p]["primary"]["zero_base"], "zero_pdms_B": rp[p]["primary"]["zero_arm"],
                                 "A_level_x100": rp[p]["primary"]["base_mean_x100"], "B_level_x100": rp[p]["primary"]["arm_mean_x100"]} for p in PAIRS},
            "pooled_structure": "mean of the two pairs' per-token PDMS contrasts; one log-cluster bootstrap on that series (== one index set per draw applied to both pairs)",
            "python": sys.version.split()[0], "numpy": np.__version__}
    return contract, prov, {"read_pair": rp, "tabs": tabs, "toks": toks, "tl": tl, "full": full}


# ============================================================ families (the four families per pair)
def _comp(ci: dict, k: str):
    c = ci.get(k)
    return {"mean": c["mean"], "lo": c["lo"], "hi": c["hi"]} if isinstance(c, dict) and all(isinstance(c.get(q), (int, float)) for q in ("mean", "lo", "hi")) else None


def _fam_block(fam: dict | None) -> dict:
    return ((fam or {}).get("families") or {}).get(ARM_KEY) or {}


def family_components(S3, fa: dict, fb: dict, fam: str, adverse_set: set) -> dict:
    ca = ((_fam_block(fa).get(fam) or {}).get("ci") or {}).get("components") or {}
    cb = ((_fam_block(fb).get(fam) or {}).get("ci") or {}).get("components") or {}
    rows = {}
    for m in sorted(set(ca) & set(cb)):
        if m.rsplit(".", 1)[-1] in S3.PE.FAM_SKIP:
            continue                                                       # properties of the human future, not of the arm
        a, b = _comp(ca, m), _comp(cb, m)
        if a is None or b is None:
            continue
        rows[m] = {"A": a, "B": b, "delta_B_minus_A": b["mean"] - a["mean"], "direction": S3.PE.fam_direction(m), "adverse": (fam, m) in adverse_set}
    return rows


def tactical_block(fa: dict, fb: dict) -> dict:
    ta, tb = _fam_block(fa).get("tactical") or {}, _fam_block(fb).get("tactical") or {}
    out = {"status": "OK" if ta and tb else "NOT_COMPUTED", "estimator": "rows WITHOUT a paired interval (families6 tactical: accuracy / kappa per arm); reported, only the kappa rule below is a rule"}
    for k in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        a, b = ta.get(k) or {}, tb.get(k) or {}
        out[k] = {"A": {q: a.get(q) for q in ("accuracy", "kappa")}, "B": {q: b.get(q) for q in ("accuracy", "kappa")}}
        if isinstance(a.get("kappa"), (int, float)) and isinstance(b.get("kappa"), (int, float)):
            out[k]["delta_kappa_B_minus_A"] = b["kappa"] - a["kappa"]
    ld = out.get("lateral_decision", {})
    if "delta_kappa_B_minus_A" in ld:
        ld["kappa_fell_by_more_than_0.03"] = bool(ld["delta_kappa_B_minus_A"] < -KAPPA_FALL_MAX)
        ld["rule"] = "PREREG_RL_S1 section 4: the lateral-decision kappa may not fall by > 0.03 vs the control"
    ga, gb = ((ta.get("goal_setting") or {}).get("ci") or {}), ((tb.get("goal_setting") or {}).get("ci") or {})
    gs = {}
    for k in sorted(set(ga) & set(gb)):
        a, b = _comp(ga, k), _comp(gb, k)
        if a and b:
            gs[k] = {"A": a, "B": b, "delta_B_minus_A": b["mean"] - a["mean"]}
    out["goal_setting"] = gs
    return out


def build_families(S3, read_pair_by_pair: dict, fams: dict, *, n_tokens: int, n_logs: int) -> dict:
    """navtest_families.json: per pair, LONGITUDINAL / LATERAL (PDM members with the paired estimator + families6 components), TACTICAL, STRATEGIC=PAUSED.  fams: state -> families6 JSON or None"""
    out = {"tool": TOOL, "tier": S3.TIER, "n_tokens": n_tokens, "n_logs": n_logs, "strategic_pause": STRATEGIC,
           "estimators": {"pdm_members": ESTIMATOR + " (B - A, x100; stage3_full.read_pair)",
                          "families6_components": "families6's own: episode-cluster bootstrap, 2,000 resamples, PER ARM and UNPAIRED; a component is 'adverse' when B's interval lies entirely on the worse side "
                                                  "of A's (proxy_eval.fam_adverse via stage3_full.families_vs).  REPORTED, NOT GATING (bar E clause 4 is about the PDMS sub-scores).",
                          "note": "the families6 components are NOT on the paired estimator: families6 emits per-arm intervals, not per-window values, so a paired B - A interval needs a families6 change (open point)"},
           "pdm_family_membership": {"LONGITUDINAL": ["EP", "TTC", "C"], "LATERAL": ["DAC", "DDC"], "unassigned_in_this_file": ["NC (a PDMS sub-score: bar E clause 4 reads it in navtest_E.json)"],
                                     "source": "PREREG_FREEPOINT.md families rows"},
           "pairs": {}}
    any_fam = False
    for p in PAIRS:
        a, b = PAIR_STATES[p]
        fa, fb = fams.get(a), fams.get(b)
        rp = read_pair_by_pair[p]
        have = fa is not None and fb is not None
        any_fam = any_fam or have
        vs = S3.families_vs(fa, fb) if have else {"missing": True, "note": "a families JSON is absent: NOT COMPUTED (never a pass)"}
        adv = {(x["family"], x["metric"]) for x in (vs.get("adverse") or [])} if have else set()
        cov = {}
        for who, f in (("A", fa), ("B", fb)):
            cc = (_fam_block(f).get("_ci_coverage") or {}) if f else {}
            cov[who] = {fam: (cc.get(fam) or {}).get("unavailable") for fam in ("longitudinal", "lateral")}
        blk = {}
        for fam, up, members in (("longitudinal", "LONGITUDINAL", ("EP", "TTC", "C")), ("lateral", "LATERAL", ("DAC", "DDC"))):
            pdm = {k: dict(ci_obj(rp["subscores"][k]), adverse=bool(rp["subscores"][k]["adverse"])) for k in members}
            f6 = ({"status": "OK", "components": family_components(S3, fa, fb, fam, adv), "unavailable_at_navtest": cov, "n_components_compared": vs.get("components_compared")}
                  if have else {"status": "NOT_COMPUTED", "reason": "families6 JSON of A and/or B absent: run `navtest_E.py families` (never read as a pass)"})
            blk[up] = {"pdm_members": pdm, "families6": f6}
        blk["TACTICAL"] = tactical_block(fa, fb) if have else {"status": "NOT_COMPUTED", "reason": "families6 JSON of A and/or B absent (never read as a pass)"}
        blk["STRATEGIC"] = dict(STRATEGIC)
        blk["families6_adverse_components"] = sorted(f"{f}.{m}" for f, m in adv) if have else None
        out["pairs"][p] = blk
    out["status"] = "COMPUTED" if any_fam else "PDM_MEMBERS_ONLY (families6 not supplied)"
    return out


# ============================================================ writing
def write_outputs(out_dir: str, contract: dict, prov: dict, families: dict | None, *, dry_run: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    validate_contract(contract, full=not dry_run)
    name = "navtest_E.DRYRUN.json" if dry_run else "navtest_E.json"
    paths = {"contract": os.path.join(out_dir, name), "provenance": os.path.join(out_dir, name.replace(".json", ".provenance.json"))}
    if dry_run:
        prov = dict(prov, DRY_RUN="NOT A READ: a subset / unpinned token set; the reader must never be given this file")
    with open(paths["contract"], "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=1)
    with open(paths["provenance"], "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1, default=str)
    if families is not None:
        paths["families"] = os.path.join(out_dir, "navtest_families.DRYRUN.json" if dry_run else "navtest_families.json")
        with open(paths["families"], "w", encoding="utf-8") as f:
            json.dump(families, f, indent=1, default=str)
    return paths


# ============================================================ seams: the free-point picker, state-generic
def build_seams(chunks: str, work: str, *, names=STATES, limit: int = 0, present_only: bool = False, order: list | None = None) -> dict:
    """merge the pod's chunk outputs, pick by the trainer's sel_score per state (and BASE), write one W3 seam per state.  Every function is the free-point module's own (fp_picks copy)."""
    import fp_picks as FPP
    AT = FPP.load_attrib()
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    zb = np.load(FPP.BANKED_U400_SEAM, allow_pickle=False)
    order = order or [str(t) for t in zb["token"]]
    tok, traj, lg, ic2, meta = FPP.merge_chunks(chunks, order, limit, present_only)
    need = ["BASE"] + list(names)
    miss = [k for k in need if k not in lg]
    if miss:
        raise Refused(f"the chunks carry no logits for {miss} (they carry {sorted(lg)}): was the worker started with the four --state arguments?")
    if ic2.max() != 0.0:
        raise Refused(f"IC2: the replicated scorer path differs from the model's own score (max {ic2.max()})")
    pk = {k: np.asarray(AT.picks(np.asarray(lg[k], np.float64)), np.int64) for k in need}
    seams = {}
    os.makedirs(work, exist_ok=True)
    for k in need:
        seams[k] = FPP.executed_poses(traj, pk[k])
        FPP.write_seam(f"{work}/seam_{k}.npz", tok, seams[k], exp, f"SOBJE_{k}")
    changes = {}
    for p in PAIRS:
        a, b = PAIR_STATES[p]
        if a not in pk or b not in pk:
            continue                                                          # a partial name list (the free-point dry-run chunk carries C0A / R0B only)
        changes[p] = {"pick_changes_B_vs_A": int((pk[b] != pk[a]).sum()), "poses_identical_B_vs_A": int(sum(np.array_equal(seams[b][i], seams[a][i]) for i in range(len(tok))))}
    rep = {"n_tokens": len(tok), "IC2_max_abs": float(ic2.max()), "IC3_BASE_vs_banked_u400": FPP.ic3(seams["BASE"], tok, traj, np.asarray(lg["BASE"], np.float64), AT),
           "pairs": changes, "pick_changes_vs_BASE": {k: int((pk[k] != pk["BASE"]).sum()) for k in names}, "chunks": [m.get("chunk") for m in meta], "state_info": meta[0]["states"],
           "ic5_forward": meta[0].get("ic5_forward"), "device": meta[0].get("device"), "gpu": meta[0].get("gpu"), "torch": meta[0].get("torch"), "ckpt_md5": meta[0].get("ckpt_md5")}
    np.savez(f"{work}/points_E.npz", token=np.array(tok), **{f"pick_{k}": pk[k] for k in need})
    json.dump(rep, open(f"{work}/picks_report_E.json", "w"), indent=1)
    return rep


def cmd_seams(a):
    try:
        rep = build_seams(a.chunks, a.work, limit=a.limit, present_only=a.present_only)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    print(json.dumps({k: v for k, v in rep.items() if k != "state_info"}, indent=1)[:2500])
    print("ZZSEAMS_E_OK")
    return 0


# ============================================================ expected states, irep, families stage
def cmd_expected(a):
    import torch
    import fp_forward as FF
    out = {}
    for spec in a.state:
        name, _, rest = spec.partition("=")
        path, _, key = rest.rpartition(":")
        if name not in STATES or key not in ("A", "B") or key != name[-1]:
            print(f"REFUSED: {spec!r}: the state {name!r} must be one of {list(STATES)} and its key must be its own suffix")
            return 2
        ss = torch.load(path, map_location="cpu", weights_only=False)
        if int(ss.get("update", -1)) != EXPECT_UPDATE:
            print(f"REFUSED: {path} is update {ss.get('update')}, expected {EXPECT_UPDATE}")
            return 2
        out[name] = {"file": os.path.basename(path), "key": key, "update": int(ss["update"]), "sha256_tensors": FF.sha256_tensors(ss[key]), "file_md5": md5_file(path)}
    if sorted(out) != sorted(STATES):
        print(f"REFUSED: need exactly {list(STATES)}, got {sorted(out)}")
        return 2
    if len({v["sha256_tensors"] for v in out.values()}) != len(out):
        print("REFUSED: two of the four states are tensor-identical")
        return 2
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v["sha256_tensors"][:16] for k, v in out.items()}))
    print("ZZEXPECTED_E_OK")
    return 0


def irep_from_tables(path_a: str, path_b: str) -> dict:
    """|mean PDMS x100 (run 1) - mean PDMS x100 (run 2)| of ONE state read twice, on the tokens both tables carry"""
    S3 = get_s3()
    ta, tb = S3.load_table(path_a), S3.load_table(path_b)
    toks = sorted(set(ta) & set(tb))
    if not toks:
        raise Refused("the two tables share no token")
    unit_guard("irep.a", ta)
    unit_guard("irep.b", tb)
    ma, mb = float(np.mean([ta[t]["PDMS"] for t in toks])), float(np.mean([tb[t]["PDMS"] for t in toks]))
    return {"irep_abs_delta": abs(ma - mb), "unit": UNIT, "kind": "navtest inference replicate: one state, two harness reads", "n_tokens": len(toks), "mean_a": ma, "mean_b": mb,
            "inputs": {"a": {"path": os.path.abspath(path_a), "sha256": sha256_file(path_a)}, "b": {"path": os.path.abspath(path_b), "sha256": sha256_file(path_b)}}}


def cmd_irep(a):
    try:
        r = irep_from_tables(a.a, a.b)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("irep_abs_delta", "unit", "n_tokens")}))
    return 0


def cmd_families(a):
    from pathlib import Path
    S3 = get_s3()
    toks = sorted(json.load(open(a.tokens, encoding="utf-8"))["tokens"])
    famdir = Path(a.work) / "families"
    for s in STATES:
        f = S3.compute_families(Path(f"{a.work}/seam_{s}.npz"), f"sobj_{s}", famdir, set(toks))
        print(f"families6 {s}: {'OK' if f else 'MISSING'}", flush=True)
        if not f:
            return 2
    print("ZZFAMILIES_E_OK")
    return 0


# ============================================================ build (CLI)
def cmd_build(a):
    work = a.work
    S3 = get_s3()
    try:
        tables = {s: table_path(s, a.table_root) for s in STATES}
        for spec in a.table:
            n, _, p = spec.partition("=")
            tables[n] = p
        pins = FULL_PINS
        tokens_file = a.tokens or TOKENS_FILE
        if a.dry_run_subset:
            pins = None
        elif a.tokens and os.path.abspath(a.tokens) != os.path.abspath(TOKENS_FILE):
            raise Refused("a token file other than the pinned navtest_full_tokens.json needs --dry-run-subset (and then writes only *.DRYRUN.json)")
        gates = read_run_gates(work)
        contract, prov, ex = build_E(tables, tokens_file=tokens_file, pins=pins, irep_path=a.irep_json, run_gates=gates)
        fams = {}
        for s in STATES:
            fp_ = f"{work}/families/sobj_{s}_families.json"
            fams[s] = json.load(open(fp_, encoding="utf-8")) if os.path.exists(fp_) else None
        families = build_families(S3, ex["read_pair"], fams, n_tokens=len(ex["toks"]), n_logs=len(set(ex["tl"].values())))
        paths = write_outputs(a.out_dir, contract, prov, families, dry_run=bool(a.dry_run_subset))
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    print(json.dumps({"status": prov["status"], "n_tokens": contract["n_tokens"], "n_logs": contract["n_logs"], "pairs_PDMS": {p: contract["pairs"][p]["PDMS"] for p in PAIRS},
                      "pooled": contract["pooled"], "floors": contract["floors"], "files": paths}, indent=1))
    print(f"ZZNAVTEST_E_{'DRYRUN' if a.dry_run_subset else 'OK'} {prov['status']}")
    return 0


def cmd_selftest(a):
    """known-value anchors through THIS tool's own functions: u400 - deployed must equal RESULT_SFT4 to 1e-12 (the free-point selftest's gate), the unit guard must refuse a x1 table"""
    S3 = get_s3()
    u, d = S3.pick_baseline("u400"), S3.pick_baseline("deployed")
    tu, td = S3.load_table(u["csv"]), S3.load_table(d["csv"])
    full = json.load(open(TOKENS_FILE, encoding="utf-8"))
    toks = sorted(full["tokens"])
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: exp[t]["log_name"] for t in toks}
    r = S3.read_pair(tu, td, toks, tl, N_BOOT, SEED)["primary"]
    dev = max(abs(r["mean"] - FPR.KNOWN["mean"]), abs(r["ci95"][0] - FPR.KNOWN["lo"]), abs(r["ci95"][1] - FPR.KNOWN["hi"]))
    x1 = {t: {k: v / 100.0 for k, v in row.items()} for t, row in tu.items()}
    try:
        unit_guard("x1", x1)
        refused = False
    except Refused:
        refused = True
    good = dev <= 1e-12 and refused and instrument_pins()["ok"]
    print({"u400_minus_deployed": [r["mean"], r["ci95"]], "max_abs_dev_from_RESULT_SFT4": dev, "unit_guard_refuses_x1": refused, "instruments_pinned": instrument_pins()["ok"]})
    print("ZZNAVTEST_E_SELFTEST_" + ("PASS" if good else "FAIL"))
    return 0 if good else 1


# ============================================================ plan (the exact command sequence)
def state_specs(arms_root: str) -> list:
    """the four --state arguments for the pod worker.  scorer_states_last.pt of an arm that ran to u1000 IS the u1000 snapshot (update == 1000: the worker refuses anything else)"""
    out = []
    for p in PAIRS:
        for s in PAIR_STATES[p]:
            out.append(f"--state {s}={arms_root}/{p}/scorer_states_last.pt:{s[-1]}")
    return out


def plan_text(arms_root: str, work: str, remote: str) -> str:
    if remote.rstrip("/") == LIVE_REMOTE:
        raise Refused(f"{LIVE_REMOTE} is the free-point run's directory: use another --remote")
    me = "navtest_E.py"
    fp = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -X utf8"
    worker = (f"cd {remote} && OMP_NUM_THREADS=2 REFE_BACKBONE_ROOT=/workspace/data/backbones PYTHONPATH=/workspace/refe-plan-rls1/refe nice -n 10 setsid nohup /workspace/venv-teacher/bin/python "
              f"fp_forward.py serve --work-dir {remote} --inputs inputs_full.json --calib calib.npz --ckpt /workspace/data/rls1/base/model_sft4_B_u400.pt " + " ".join(state_specs(arms_root))
              + " --device cuda --repeat-first 128 > serve.log 2>&1 < /dev/null & echo $! > worker.pid")
    lines = [
        "# 0. GATES (do not start before ALL hold): R0d, R1d, R1o finished or killed-and-decided (the A40 is free: never add load to a training pod); the P6 re-roll gate of fp_forward serve is honoured or",
        "#    explicitly overridden by the Master Mind (--no-gate); NO other process owns " + remote + ".",
        f"{fp} {me} expected-states --state R0d_A=<dev>/rls1-r0d/scorer_states/scorer_states_u1000.pt:A --state R0d_B=<dev>/rls1-r0d/scorer_states/scorer_states_u1000.pt:B "
        f"--state R1d_A=<dev>/rls1-r1d/scorer_states/scorer_states_u1000.pt:A --state R1d_B=<dev>/rls1-r1d/scorer_states/scorer_states_u1000.pt:B --out {work}/EXPECTED_STATES_E.json",
        "# 1. dev box: the model inputs are the free-point run's finished artifacts (copy, do not re-capture)",
        f"cp C:/Users/Admin/freept/work/inputs_full.json C:/Users/Admin/freept/work/calib.npz C:/Users/Admin/freept/work/plan.json C:/Users/Admin/freept/work/ship_plan.json {work}/   # md5 recorded in picks_report_E",
        "# 2. pod prep (once; ships fp_forward.py + inputs + rig; NO states: they are already on the pod).  sobj_ship overrides REMOTE/STATES of the fp_ship COPY",
        f"{fp} sobj_ship.py prep-pod --work {work} --remote {remote}",
        "# 3. pod worker (one command; gated by fp_forward on /workspace/ops/bank3c/full/started_epoch unless --no-gate is added by the Master Mind)",
        f"ssh <pod> '{worker}'",
        "#    stop only by pid: ssh <pod> 'kill $(cat " + remote + "/worker.pid)'   (never pgrep -f / pkill -f)",
        "# 4. uploader + pull, detached (WMI Create, CreateFlags 0x08000200); ends with ZZSHIP_ALL_PULLED",
        f"{fp} sobj_ship.py run --work {work} --remote {remote} > {work}/ship.log 2>&1",
        "# 5. dev-box tail (CPU; resumable): seams -> harness x4 -> families6 x4 -> build",
        f"{fp} {me} seams --work {work} --chunks {work}/out",
        f"{fp} fp/fp_score.py score --work {work} --points {','.join(STATES)} --label-prefix {LABEL_PREFIX} --parallel 1",
        f"{fp} {me} families --work {work} --tokens {TOKENS_FILE}",
        f"{fp} {me} build --work {work} --out-dir {work}/result --irep-json <irep.json>",
        "# 6. the outcome reader",
        f"python eval/safeobj_outcome.py --arm R0d=$R/arms/R0d --arm R1d=$R/arms/R1d --arm R1o=$R/arms/R1o --arm R0=<R0 arm dir> --navtest-e {work}/result/navtest_E.json "
        "--dose raw/2026-10-14-rls1b-attribution/dose.json --prereg raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md "
        "--addendum raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.addendum1.md --out ...",
    ]
    return "\n".join(lines)


def cmd_plan(a):
    try:
        print(plan_text(a.arms_root, a.work, a.remote))
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--arms-root", default="/workspace/data/rls1/arms")
    p.add_argument("--work", default="C:/Users/Admin/sobjE/work")
    p.add_argument("--remote", default=DEFAULT_REMOTE)
    p = sp.add_parser("expected-states")
    p.add_argument("--state", action="append", default=[])
    p.add_argument("--out", required=True)
    p = sp.add_parser("seams")
    p.add_argument("--work", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--present-only", action="store_true")
    p = sp.add_parser("families")
    p.add_argument("--work", required=True)
    p.add_argument("--tokens", default=TOKENS_FILE)
    p = sp.add_parser("build")
    p.add_argument("--work", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--irep-json", default=None)
    p.add_argument("--table-root", default=None)
    p.add_argument("--table", action="append", default=[], help="STATE=csv (overrides the harness location)")
    p.add_argument("--tokens", default=None)
    p.add_argument("--dry-run-subset", action="store_true", help="DRY RUNS ONLY: unpinned token set, writes *.DRYRUN.json")
    p = sp.add_parser("irep")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    sp.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "expected-states": cmd_expected, "seams": cmd_seams, "families": cmd_families, "build": cmd_build, "irep": cmd_irep, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
