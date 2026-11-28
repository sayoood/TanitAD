#!/usr/bin/env python3
"""select_ckpt_r8.py -- the D-R8-CKPT-RULE decision script (numpy + stdlib only; any python).

Implements, EXACTLY, the rule registered in ``Project Steering/GOALS_AND_CLAIMS.md`` row
``D-R8-CKPT-RULE`` (2026-10-20) + ``AMENDMENT 1`` (2026-10-21), which chooses refcv8's deliverable checkpoint on the NavSim
**navtrain** sample.  Nothing in this file is a judgement call; every number that decides is a LITERAL below.

THE RULE (registered text, condensed to what this file decides)
    arms        ``R8_RT`` (EMA weights + the registered per-frame RETIME table ``2aacc9ff...``) at EMA-40k and EMA-48,852,
                inference seeds 0 and 1, the SAME tokens.
    metric      NAVSIM v1 PDMS per token; PAIRED  40k - 48,852  on the same tokens; log-cluster bootstrap (cluster = ``log_name``),
                B = 2000, per inference seed, through ``taniteval/ci.py::paired_episode_cluster_bootstrap`` (log_name = its "episode").
    decision    choose **EMA-40k** iff, at BOTH inference seeds, delta > 0 with the 95 % interval excluding 0, AND the point margin
                exceeds 2x the inference floor |delta_s0 - delta_s1|.  Otherwise **EMA-48,852** (the registered default).
    refusal     fewer than 1,000 tokens, any token missing in any arm, any stand-in row, token sets that differ  ->  verdict
                ``INCOMPLETE`` and the default (EMA-48,852) holds.  (Rule text: "< 1,000 tokens ... UNDERPOWERED and the default
                holds"; "Any deviation ... or the selection reads INCOMPLETE and the default holds".)
    controls    48,852 - 48,852 reads EXACTLY 0 [0, 0]; a token-SHUFFLED pairing widens the interval; STOP / CV / PRIOR rows and the
                sub-scores (NC, DAC, EP, TTC, C, DDC) are printed BESIDE and never decide.

TWO READINGS THE REGISTERED TEXT LEAVES OPEN (both fixed here BEFORE any navtrain score exists; neither changes a bar)
    * "the point margin" = the SMALLER of the two seeds' deltas, ``min(delta_s0, delta_s1)``.  The rule says "at BOTH inference
      seeds", and ``navsim_bars8.py`` (N1/N4) applies "margin > 2x floor" to each seed separately; the mean of the seeds is a weaker
      reading and is printed beside (``margin_mean_beside``), never used.
    * a cluster bootstrap needs >= 2 clusters (``MIN_CLUSTERS``); with one log every resample is identical and the "interval" is a
      point.  This is a structural impossibility guard, not a new bar; it refuses toward the default like every other refusal.

WHAT IT CHECKS BEFORE IT COMPUTES (any failure -> INCOMPLETE; the numbers are then WITHHELD, so an incomplete run cannot be read as a
result)
    sample FIXED and its sorted-token sha256 re-derives; per arm: the scorer CSV (every token valid, finite, PDMS in [0, 1], the PDMS
    identity), the scorer ``counts.json`` (status PASS, n_expected == the sample), the scorer ``binding.json`` (the CSV was produced from
    THIS seam file, by sha256), the seam npz (every row ``precomputed``: no CV stand-in), the seam manifests (partial false, zero
    stand-ins, split navtrain, retime table = the registered deploy table, ``derived_from`` = the right rows file) and the forward
    manifest of the same bridge dir (checkpoint md5 = the registered literal for that EMA, device cuda, seed, KPR PASS) -- so a swapped
    or mislabelled arm cannot be read as the other checkpoint.

Evidence class of every output number: MEASURED (the input artifacts named in ``inputs``).  Tier: NavSim open-loop PDM scorer
(T1-family; NOT closed loop).  Scope: a choice between two EMA snapshots of ONE training run; it says nothing about another seed.

    python select_ckpt_r8.py --spec spec.json --out DECISION.json [--md DECISION.md] [--ci-path .../taniteval/ci.py]
    python select_ckpt_r8.py --layout-root <root> --sample <NAVTRAIN_SAMPLE_TOKENS.json> --out DECISION.json

``R8SEL_MUTATION`` (tests only, see ``MUTATIONS``) reintroduces one real defect class so ``tests/test_mutation_select_ckpt_r8.py`` can
prove the literal tests go RED on it; ``main`` REFUSES to run while it is set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np

MUTATION = os.environ.get("R8SEL_MUTATION", "")
MUTATIONS = ("sign_flip", "seed0_only", "drop_floor_gate", "floor_1x", "margin_mean", "positional_join", "token_cluster",
             "no_separation_needed", "standin_allowed", "min_tokens_off", "missing_token_ignored", "identity_unchecked",
             "shuffle_control_identity", "controls_not_vetoing", "alpha_90", "n_boot_200", "incomplete_chooses_40k")

# --------------------------------------------------------------------------------------------------------------------------------
# THE REGISTERED CONSTANTS -- LITERALS (D-R8-CKPT-RULE 2026-10-20 + AMENDMENT 1).  Tests pin each one as a literal too.
# --------------------------------------------------------------------------------------------------------------------------------
N_BOOT = 200 if MUTATION == "n_boot_200" else 2000       # (3) "B 2000"
ALPHA = 0.10 if MUTATION == "alpha_90" else 0.05          # (3)/(4) "95 % interval"
BOOT_SEED = 0                                             # the ci.py default; the SAME resampled clusters at both inference seeds
SHUFFLE_SEED = 1                                          # (6) the token-shuffled pairing control
MIN_TOKENS = 1000                                         # (1) "< 1,000 tokens ... UNDERPOWERED"
MIN_CLUSTERS = 2                                          # structural guard (see module docstring)
FLOOR_MULT = 1.0 if MUTATION == "floor_1x" else 2.0       # (4) "2x the inference floor"
CHALLENGER = "ema40000"
DEFAULT = "ema48852"
SEEDS = (0, 1)
CKPT_STEP = {"ema40000": 40000, "ema48852": 48852}
CKPT_MD5 = {"ema40000": "57807456bd7169800be159ff95871d22",       # ckpt_ema_40000.pt (GROUNDWORK section 1, md5-verified)
            "ema48852": "b4364029b68f29a33eeb4aebbbbd002b"}       # ckpt_ema_48852.pt (rule text: md5 b4364029...)
RT_DEPLOY_SHA256 = "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"   # refcv8_retime_deploy_gains_pf.json
RT_FROZEN_SHA256 = "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80"    # the registered per-frame table
PDMS_IDENTITY_TOL = 1e-9
SUBSCORES = ("NC", "DAC", "EP", "TTC", "C", "DDC")
CSV_COLS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
            "TTC": "time_to_collision_within_bound", "C": "comfort", "DDC": "driving_direction_compliance", "PDMS": "score"}
LABEL_PREFIX = "r8sel"                                    # W3's score driver requires labels starting with r8
SEAM_ARM = {0: "R8_RT", 1: "R8_RT_s1"}
SRC_ARM = {0: "R8_LEGAL", 1: "R8_LEGAL_s1"}
TIER = ("NavSim open-loop PDM scorer (T1-family; NOT closed loop). Choice between two EMA snapshots of ONE training run; says nothing "
        "about another training seed. Backend: rolls on Thor CUDA, scoring on the Thor CPU backend (D-R8-CKPT-RULE AMENDMENT 1c).")


def arm_label(ckpt: str, seed: int) -> str:
    return f"{LABEL_PREFIX}_{ckpt}_s{seed}"


# --------------------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tokens(tokens) -> str:
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


def locate_ci(explicit: str = "") -> str:
    """Find ``taniteval/ci.py`` BY PATH.  ``import taniteval.ci`` is deliberately not used: when the repo root is on ``sys.path`` the
    outer ``taniteval/`` directory binds as a NAMESPACE package and ``taniteval.ci`` raises ModuleNotFoundError (memory:
    taniteval-namespace-package-shadow); loading the file directly cannot be shadowed."""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("TANITEVAL_CI"):
        cands.append(os.environ["TANITEVAL_CI"])
    repo = os.environ.get("TANITAD_REPO", "")
    if repo:
        cands.append(f"{repo}/taniteval/taniteval/ci.py")
    for p in sys.path:
        if not p:
            continue
        pn = p.replace("\\", "/").rstrip("/")
        cands += [f"{pn}/taniteval/ci.py", f"{pn}/taniteval/taniteval/ci.py", f"{pn}/ci.py" if pn.endswith("/taniteval") else ""]
    cands.append("C:/lgs/Lfb47a58376/taniteval/taniteval/ci.py")
    for c in cands:
        if c and os.path.isfile(c) and os.path.basename(c) == "ci.py":
            return os.path.abspath(c)
    raise FileNotFoundError("taniteval/ci.py not found: pass --ci-path or set TANITEVAL_CI")


def load_ci(explicit: str = ""):
    path = locate_ci(explicit)
    spec = importlib.util.spec_from_file_location("_r8sel_taniteval_ci", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__r8sel_path__ = path
    mod.__r8sel_sha256__ = sha256_file(path)
    return mod


# --------------------------------------------------------------------------------------------------------------------------------
# loading ONE arm from the scorer's artifacts (every check here only APPENDS a problem; deciding happens in decide())
# --------------------------------------------------------------------------------------------------------------------------------
def read_scorer_csv(path: str):
    """``(table, problems)``; ``table[token] = {valid, PDMS, NC, ..}``.  The devkit's trailing ``average`` row is skipped."""
    table, problems = {}, []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        need = ["token", "valid"] + list(CSV_COLS.values())
        miss = [c for c in need if c not in (rd.fieldnames or [])]
        if miss:
            return {}, [f"CSV_COLUMNS_MISSING:{miss}"]
        for r in rd:
            tok = r["token"]
            if tok == "average":
                continue
            if tok in table:
                problems.append(f"DUPLICATE_TOKEN:{tok}")
                continue
            valid = str(r["valid"]).strip() in ("True", "true", "1")
            try:
                vals = {k: float(r[c]) for k, c in CSV_COLS.items()}
            except (TypeError, ValueError):
                vals = {k: float("nan") for k in CSV_COLS}
            table[tok] = dict(vals, valid=valid)
    return table, problems


def _json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_arm(spec: dict, *, ckpt: str, seed: int, sample_sha: str, n_sample: int, kind: str = "deliverable"):
    """-> ``(table, info, problems)`` for one arm spec ``{csv, counts, binding, seam, bridge_dir}``."""
    name = f"{ckpt}_s{seed}" if kind == "deliverable" else str(ckpt)
    info = {"csv": spec.get("csv"), "seam": spec.get("seam"), "binding": spec.get("binding"), "counts": spec.get("counts")}
    problems = []
    try:
        table, p = read_scorer_csv(spec["csv"])
        problems += [f"{name}:{x}" for x in p]
        info["csv_sha256"] = sha256_file(spec["csv"])
    except (OSError, KeyError) as e:
        return {}, info, [f"{name}:CSV_UNREADABLE:{type(e).__name__}"]
    # ---- counts.json (the scorer's own guards) ----
    if spec.get("counts"):
        try:
            c = _json(spec["counts"])
            if c.get("status") != "PASS":
                problems.append(f"{name}:SCORER_COUNTS_STATUS_{c.get('status')}")
            if c.get("failures"):
                problems.append(f"{name}:SCORER_COUNTS_FAILURES")
            if c.get("n_expected") != n_sample:
                problems.append(f"{name}:SCORER_N_EXPECTED_{c.get('n_expected')}_NE_{n_sample}")
            if c.get("C3_missing") not in (0, None) or c.get("C3_unexpected") not in (0, None):
                problems.append(f"{name}:SCORER_TOKEN_SET_C3")
            info["counts_status"] = c.get("status")
        except (OSError, ValueError) as e:
            problems.append(f"{name}:COUNTS_UNREADABLE:{type(e).__name__}")
    elif kind == "deliverable":
        problems.append(f"{name}:COUNTS_ABSENT")
    # ---- seam npz: the stand-in flag per token ----
    if spec.get("seam"):
        try:
            z = np.load(spec["seam"], allow_pickle=False)
            stoks, ssrc = [str(x) for x in z["token"]], [str(x) for x in z["source"]]
            info["seam_sha256"] = sha256_file(spec["seam"])
            if len(set(stoks)) != len(stoks):
                problems.append(f"{name}:SEAM_DUPLICATE_TOKENS")
            if set(stoks) != set(table):
                problems.append(f"{name}:SEAM_CSV_TOKEN_SETS_DIFFER")
            flag = {t: (s != "precomputed") for t, s in zip(stoks, ssrc)}
            for t in table:
                table[t]["standin"] = flag.get(t)
            if kind == "deliverable" and "arm" in z.files and str(z["arm"]) != SEAM_ARM[seed]:
                problems.append(f"{name}:SEAM_ARM_LABEL_{z['arm']}_NE_{SEAM_ARM[seed]}")
        except (OSError, KeyError, ValueError) as e:
            problems.append(f"{name}:SEAM_UNREADABLE:{type(e).__name__}")
            for t in table:
                table[t]["standin"] = None
    else:
        for t in table:
            table[t]["standin"] = None if kind == "deliverable" else False
        if kind == "deliverable":
            problems.append(f"{name}:SEAM_ABSENT")
    # ---- binding.json: the CSV was produced from THIS seam, for THIS sample ----
    if spec.get("binding"):
        try:
            b = _json(spec["binding"])
            if spec.get("seam") and b.get("seam_sha256") != info.get("seam_sha256"):
                problems.append(f"{name}:BINDING_SEAM_SHA_MISMATCH")
            if b.get("csv_sha256") != info.get("csv_sha256"):
                problems.append(f"{name}:BINDING_CSV_SHA_MISMATCH")
            if b.get("sample_sha256_sorted_tokens") != sample_sha:
                problems.append(f"{name}:BINDING_SAMPLE_SHA_MISMATCH")
            if b.get("split") != "navtrain":
                problems.append(f"{name}:BINDING_SPLIT_{b.get('split')}")
        except (OSError, ValueError) as e:
            problems.append(f"{name}:BINDING_UNREADABLE:{type(e).__name__}")
    elif kind == "deliverable":
        problems.append(f"{name}:BINDING_ABSENT")
    # ---- identity: which checkpoint / seed / table this arm really is (read from the bridge's own manifests) ----
    if kind == "deliverable" and MUTATION != "identity_unchecked":
        bdir = spec.get("bridge_dir") or (os.path.dirname(spec["seam"]) if spec.get("seam") else "")
        try:
            fm = _json(os.path.join(bdir, f"seam_{SRC_ARM[seed]}.manifest.json"))
            rm = _json(os.path.join(bdir, f"seam_{SEAM_ARM[seed]}.manifest.json"))
            model = fm.get("model") or {}
            ident = {"ckpt_md5": model.get("ckpt_md5"), "step": model.get("step"), "device": fm.get("device"),
                     "precision": fm.get("precision"), "seed": (fm.get("spec") or {}).get("seed"), "split": fm.get("split"),
                     "KPR": (fm.get("KPR") or {}).get("verdict"), "rt_derived_from": rm.get("derived_from"),
                     "rt_table_sha256": (rm.get("retime") or {}).get("table_sha256"), "rt_partial": rm.get("partial"),
                     "rt_n_cv_standin_rows": rm.get("n_cv_standin_rows"), "fwd_n_cv_standin_rows": fm.get("n_cv_standin_rows")}
            info["identity"] = ident
            if ident["ckpt_md5"] != CKPT_MD5[ckpt] or ident["step"] != CKPT_STEP[ckpt]:
                problems.append(f"{name}:ARM_IDENTITY_MISMATCH:ckpt_md5={ident['ckpt_md5']}:step={ident['step']}")
            if ident["device"] != ["cuda"]:
                problems.append(f"{name}:ARM_NOT_ON_THOR_CUDA:{ident['device']}")
            if ident["seed"] != seed:
                problems.append(f"{name}:ARM_SEED_MISMATCH:{ident['seed']}")
            if ident["split"] != "navtrain" or rm.get("split") != "navtrain":
                problems.append(f"{name}:ARM_SPLIT_NOT_NAVTRAIN")
            if ident["KPR"] != "PASS":
                problems.append(f"{name}:ARM_KPR_{ident['KPR']}")
            if ident["rt_derived_from"] != f"rows_{SRC_ARM[seed]}.jsonl":
                problems.append(f"{name}:RT_DERIVED_FROM_{ident['rt_derived_from']}")
            if ident["rt_table_sha256"] != RT_DEPLOY_SHA256:
                problems.append(f"{name}:RT_TABLE_NOT_THE_REGISTERED_DEPLOY_TABLE")
            if ident["rt_partial"] is not False or ident["fwd_n_cv_standin_rows"] != 0 or ident["rt_n_cv_standin_rows"] != 0:
                problems.append(f"{name}:MANIFEST_PARTIAL_OR_STANDIN")
        except (OSError, ValueError, KeyError) as e:
            problems.append(f"{name}:IDENTITY_MANIFESTS_UNREADABLE:{type(e).__name__}")
    return table, info, problems


def load_sample(path: str):
    s = _json(path)
    problems = []
    toks = list(s.get("tokens", []))
    if s.get("status") != "FIXED":
        problems.append(f"SAMPLE_STATUS_{s.get('status')}")
    if len(set(toks)) != len(toks):
        problems.append("SAMPLE_DUPLICATE_TOKENS")
    if s.get("count") != len(toks):
        problems.append("SAMPLE_COUNT_FIELD_NE_LEN")
    sha = sha256_tokens(toks)
    if sha != s.get("sha256_sorted_tokens"):
        problems.append("SAMPLE_SHA256_DOES_NOT_REDERIVE")
    return s, sha, problems


def load_spec(spec_path: str):
    """-> ``(sample, tables, references, problems, provenance)``"""
    spec = _json(spec_path)
    sample, sha, problems = load_sample(spec["sample"])
    n = len(sample.get("tokens", []))
    tables, infos = {}, {}
    for key, aspec in spec.get("arms", {}).items():
        ckpt, s = key.rsplit("_s", 1)
        seed = int(s)
        tab, info, p = load_arm(aspec, ckpt=ckpt, seed=seed, sample_sha=sha, n_sample=n)
        tables[(ckpt, seed)], infos[key] = tab, info
        problems += p
    refs, rinfos = {}, {}
    for rname, rspec in spec.get("references", {}).items():
        try:
            tab, info, p = load_arm(rspec, ckpt=rname, seed=0, sample_sha=sha, n_sample=n, kind="reference")
            refs[rname], rinfos[rname] = tab, info
            if p:
                rinfos[rname]["unusable_because"] = p
                refs[rname] = {}
        except Exception as e:                                           # noqa: BLE001  (a BESIDE row never blocks the decision)
            rinfos[rname] = {"unusable_because": [f"{type(e).__name__}:{e}"]}
            refs[rname] = {}
    return sample, tables, refs, problems, {"spec": os.path.abspath(spec_path), "sample": os.path.abspath(spec["sample"]),
                                              "sample_sha256_sorted_tokens": sha, "arms": infos, "references": rinfos}


# --------------------------------------------------------------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------------------------------------------------------------
def _vec(table: dict, toks: list, metric: str) -> np.ndarray:
    """A metric vector aligned to ``toks``.  Pairing is BY TOKEN (a dict lookup), never by row position."""
    if MUTATION == "positional_join":
        vals = [r[metric] for r in table.values()]
        return np.asarray(vals[:len(toks)], dtype=np.float64)
    return np.asarray([table[t][metric] for t in toks], dtype=np.float64)


def _paired(ci, a, b, eid, n_boot):
    return ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=BOOT_SEED, alpha=ALPHA)


def _positive(delta: float, r: dict) -> bool:
    """delta > 0 with the interval EXCLUDING 0 (i.e. lo > 0).  ``separated`` is decided by ci.py on the UNROUNDED bounds."""
    if MUTATION == "no_separation_needed":
        return delta > 0
    return bool(delta > 0 and r["separated"] and r["lo"] > 0 and not r.get("degenerate", False))


def _x100(v):
    return None if v is None else round(100.0 * float(v), 4)


def _result_skeleton(sample: dict, prov: dict, ci) -> dict:
    return {"schema": "r8-ckpt-select/1", "rule": "D-R8-CKPT-RULE 2026-10-20 + AMENDMENT 1 (GOALS_AND_CLAIMS.md)",
            "tier": TIER, "evidence_class": "MEASURED (input artifacts in `inputs`)",
            "constants": {"N_BOOT": N_BOOT, "ALPHA": ALPHA, "BOOT_SEED": BOOT_SEED, "SHUFFLE_SEED": SHUFFLE_SEED,
                          "MIN_TOKENS": MIN_TOKENS, "MIN_CLUSTERS": MIN_CLUSTERS, "FLOOR_MULT": FLOOR_MULT,
                          "CHALLENGER": CHALLENGER, "DEFAULT": DEFAULT, "SEEDS": list(SEEDS), "CKPT_MD5": CKPT_MD5},
            "estimator": "paired_episode_cluster_bootstrap (cluster = log_name) via taniteval/ci.py",
            "ci_py": {"path": getattr(ci, "__r8sel_path__", None), "sha256": getattr(ci, "__r8sel_sha256__", None)},
            "inputs": prov, "sample_status": sample.get("status"), "n_tokens": len(sample.get("tokens", []))}


def decide(tables: dict, sample: dict, log_of: dict, ci, *, pre_problems=(), references=None, prov=None,
           beside: bool = True, n_boot: int | None = None) -> dict:
    """The rule.  ``tables[(ckpt, seed)][token] = {PDMS, NC.., valid, standin}``; ``log_of[token] = log_name``."""
    n_boot = N_BOOT if n_boot is None else n_boot
    out = _result_skeleton(sample, prov or {}, ci)
    problems = list(pre_problems)
    toks = sorted(sample.get("tokens", []))
    tokset = set(toks)
    if len(toks) < MIN_TOKENS and MUTATION != "min_tokens_off":
        problems.append("UNDERPOWERED_LT_1000_TOKENS")
        out["underpowered"] = True
    for ck in (CHALLENGER, DEFAULT):
        for s in SEEDS:
            if (ck, s) not in tables:
                problems.append(f"ARM_ABSENT:{ck}_s{s}")
    use = list(toks)
    for (ck, s), tab in sorted(tables.items()):
        missing = [t for t in toks if t not in tab]
        extra = [t for t in tab if t not in tokset]
        if missing and MUTATION != "missing_token_ignored":
            problems.append(f"TOKEN_MISSING_IN_ARM:{ck}_s{s}:{len(missing)}")
        if missing and MUTATION == "missing_token_ignored":
            use = [t for t in use if t in tab]
        if extra:
            problems.append(f"TOKEN_SET_DIFFERS:{ck}_s{s}:extra{len(extra)}")
        for t in toks:
            r = tab.get(t)
            if r is None:
                continue
            if not r.get("valid", False) or not all(math.isfinite(r.get(k, float("nan"))) for k in ["PDMS"] + list(SUBSCORES)):
                problems.append(f"ROW_INVALID_OR_NONFINITE:{ck}_s{s}")
                break
        for t in toks:
            r = tab.get(t)
            if r is not None and not (-1e-9 <= r.get("PDMS", float("nan")) <= 1 + 1e-9):
                problems.append(f"PDMS_OUT_OF_RANGE:{ck}_s{s}")
                break
        if MUTATION != "standin_allowed":
            n_stand = sum(1 for t in toks if t in tab and tab[t].get("standin") is True)
            n_unk = sum(1 for t in toks if t in tab and tab[t].get("standin") is None)
            if n_stand:
                problems.append(f"CV_STANDIN_ROWS:{ck}_s{s}:{n_stand}")
            if n_unk:
                problems.append(f"STANDIN_STATUS_UNVERIFIED:{ck}_s{s}:{n_unk}")
        bad = [t for t in toks if t in tab and abs(tab[t]["NC"] * tab[t]["DAC"] * (5 * tab[t]["EP"] + 5 * tab[t]["TTC"] + 2 * tab[t]["C"]) / 12.0
                                                    - tab[t]["PDMS"]) > PDMS_IDENTITY_TOL]
        if bad:
            problems.append(f"PDMS_IDENTITY_VIOLATED:{ck}_s{s}:{len(bad)}")
    logs_missing = [t for t in toks if t not in log_of]
    if logs_missing:
        problems.append(f"TOKEN_WITHOUT_LOG_NAME:{len(logs_missing)}")
    n_clusters = len({log_of[t] for t in toks if t in log_of})
    out["n_clusters"] = n_clusters
    if n_clusters < MIN_CLUSTERS:
        problems.append(f"FEWER_THAN_{MIN_CLUSTERS}_CLUSTERS:{n_clusters}")
    problems = sorted(set(problems))
    if problems:
        return _finish_incomplete(out, problems)

    toks = use
    out["n_tokens_used"] = len(toks)
    eid = [log_of[t] if MUTATION != "token_cluster" else t for t in toks]
    a_ck, b_ck = (DEFAULT, CHALLENGER) if MUTATION == "sign_flip" else (CHALLENGER, DEFAULT)
    per_seed, zero_ctrl, shuf_ctrl = {}, {}, {}
    for s in SEEDS:
        a, b = _vec(tables[(a_ck, s)], toks, "PDMS"), _vec(tables[(b_ck, s)], toks, "PDMS")
        r = _paired(ci, a, b, eid, n_boot)
        delta = float(np.nanmean(a) - np.nanmean(b))
        per_seed[s] = {"delta": delta, "delta_x100": _x100(delta), "lo": r["lo"], "hi": r["hi"], "ci95": r["ci95"],
                       "separated": r["separated"], "p_delta_gt0": r["p_delta_gt0"], "n_windows": r["n_windows"],
                       "n_episodes_logs": r["n_episodes"], "n_boot": r["n_boot"], "estimator": r["estimator"],
                       "pdms_x100": {CHALLENGER: _x100(np.nanmean(_vec(tables[(CHALLENGER, s)], toks, "PDMS"))),
                                     DEFAULT: _x100(np.nanmean(_vec(tables[(DEFAULT, s)], toks, "PDMS")))},
                       "positive_and_excludes_zero": _positive(delta, r)}
        if r.get("degenerate"):
            per_seed[s]["degenerate"] = True
        # control 1: 48,852 - 48,852 reads EXACTLY 0 [0, 0]
        bb = _vec(tables[(DEFAULT, s)], toks, "PDMS")
        z = _paired(ci, bb, bb.copy(), eid, n_boot)
        zero_ctrl[s] = {"delta": z["delta"], "lo": z["lo"], "hi": z["hi"], "separated": z["separated"],
                        "reads_exactly_zero": bool(z["delta"] == 0.0 and z["lo"] == 0.0 and z["hi"] == 0.0 and not z["separated"])}
        # control 2: a token-SHUFFLED pairing leaves the point estimate alone and WIDENS the interval
        perm = np.arange(len(toks)) if MUTATION == "shuffle_control_identity" else np.random.default_rng(SHUFFLE_SEED + s).permutation(len(toks))
        sh = _paired(ci, a, b[perm], eid, n_boot)
        w_true, w_shuf = r["hi"] - r["lo"], sh["hi"] - sh["lo"]
        shuf_ctrl[s] = {"delta_shuffled": sh["delta"], "lo": sh["lo"], "hi": sh["hi"], "width_true": round(w_true, 6),
                        "width_shuffled": round(w_shuf, 6), "point_unchanged": bool(abs(sh["delta"] - r["delta"]) <= 5e-5),
                        "widens": bool(w_shuf > w_true)}
    d0, d1 = per_seed[0]["delta"], per_seed[1]["delta"]
    floor = abs(d0 - d1)
    margin_min, margin_mean = min(d0, d1), 0.5 * (d0 + d1)
    margin = margin_mean if MUTATION == "margin_mean" else margin_min
    margin_ok = bool(True if MUTATION == "drop_floor_gate" else margin > FLOOR_MULT * floor)
    seeds_checked = (0,) if MUTATION == "seed0_only" else SEEDS
    both_positive = all(per_seed[s]["positive_and_excludes_zero"] for s in seeds_checked)
    controls_ok = bool(all(zero_ctrl[s]["reads_exactly_zero"] for s in SEEDS)
                       and all(shuf_ctrl[s]["widens"] and shuf_ctrl[s]["point_unchanged"] for s in SEEDS))
    out["controls"] = {"zero_48852_minus_48852": {str(s): zero_ctrl[s] for s in SEEDS},
                       "token_shuffled_pairing": {str(s): shuf_ctrl[s] for s in SEEDS}, "all_read_known_values": controls_ok}
    out["per_seed"] = {str(s): per_seed[s] for s in SEEDS}
    out["inference_floor_abs_delta_s0_minus_s1"] = floor
    out["inference_floor_x100"] = _x100(floor)
    out["margin"] = {"definition": "min(delta_s0, delta_s1)", "value": margin, "value_x100": _x100(margin),
                     "needed_gt": FLOOR_MULT * floor, "needed_gt_x100": _x100(FLOOR_MULT * floor), "exceeds": margin_ok}
    out["margin_mean_beside"] = {"value_x100": _x100(margin_mean), "exceeds_2x_floor": bool(margin_mean > 2.0 * floor)}
    if not controls_ok and MUTATION != "controls_not_vetoing":
        out["beside"] = _beside(tables, references, toks, eid, ci, n_boot) if beside else {}
        return _finish_incomplete(out, ["CONTROL_DID_NOT_READ_ITS_KNOWN_VALUE"], keep_numbers=True)
    choose = bool(both_positive and margin_ok)
    out["decision_terms"] = {"both_seeds_positive_and_interval_excludes_zero": both_positive, "margin_exceeds_2x_floor": margin_ok,
                             "controls_read_known_values": controls_ok}
    out["verdict"] = "CHOOSE_EMA_40000" if choose else "DEFAULT_EMA_48852"
    out["choice"] = CHALLENGER if choose else DEFAULT
    out["reasons"] = [] if choose else [r for r, bad in (
        ("A_SEED_IS_NOT_POSITIVE_WITH_AN_INTERVAL_EXCLUDING_ZERO", not both_positive),
        ("MARGIN_NOT_ABOVE_2X_INFERENCE_FLOOR", not margin_ok)) if bad]
    out["beside"] = _beside(tables, references, toks, eid, ci, n_boot) if beside else {}
    out["beside_note"] = "sub-scores, seed floors and reference rows are printed beside and NEVER decide (rule item 3 / 6)"
    return out


def _finish_incomplete(out: dict, problems: list, keep_numbers: bool = False) -> dict:
    out["verdict"] = "INCOMPLETE_DEFAULT_EMA_48852"
    out["choice"] = CHALLENGER if MUTATION == "incomplete_chooses_40k" else DEFAULT
    out["reasons"] = list(problems)
    out["numbers"] = "kept (a control failed)" if keep_numbers else "WITHHELD: an incomplete run is not a result"
    if not keep_numbers:
        for k in ("per_seed", "controls", "margin", "beside"):
            out.pop(k, None)
    out["note"] = "the registered default (EMA-48,852) holds; no deliverable-checkpoint choice may be read from an INCOMPLETE run"
    return out


def _beside(tables, references, toks, eid, ci, n_boot) -> dict:
    """Everything printed beside, never deciding: sub-score deltas, per-arm means, the inference floor per checkpoint, references."""
    res = {"subscore_paired_delta_40k_minus_48852": {}, "arm_means_x100": {}, "inference_floor_per_checkpoint_x100": {}, "references": {}}
    for s in SEEDS:
        res["subscore_paired_delta_40k_minus_48852"][str(s)] = {}
        for m in SUBSCORES:
            a, b = _vec(tables[(CHALLENGER, s)], toks, m), _vec(tables[(DEFAULT, s)], toks, m)
            r = _paired(ci, a, b, eid, n_boot)
            res["subscore_paired_delta_40k_minus_48852"][str(s)][m] = {"delta_x100": _x100(np.mean(a) - np.mean(b)), "lo_x100": _x100(r["lo"]),
                                                                       "hi_x100": _x100(r["hi"]), "deciding": False}
        for ck in (CHALLENGER, DEFAULT):
            res["arm_means_x100"][f"{ck}_s{s}"] = {m: _x100(np.mean(_vec(tables[(ck, s)], toks, m))) for m in ("PDMS",) + SUBSCORES}
    for ck in (CHALLENGER, DEFAULT):
        res["inference_floor_per_checkpoint_x100"][ck] = _x100(abs(np.mean(_vec(tables[(ck, 0)], toks, "PDMS"))
                                                                    - np.mean(_vec(tables[(ck, 1)], toks, "PDMS"))))
    for rname, tab in (references or {}).items():
        if not tab or any(t not in tab for t in toks) or set(tab) - set(toks):
            res["references"][rname] = {"usable": False, "why": "empty, unreadable or token set differs from the sample"}
            continue
        row = {"usable": True}
        for m in ("PDMS",) + SUBSCORES:
            v = _vec(tab, toks, m)
            r = ci.episode_cluster_bootstrap(v, eid, "mean", n_boot=n_boot, seed=BOOT_SEED, alpha=ALPHA)
            row[m] = {"mean_x100": _x100(r["mean"]), "lo_x100": _x100(r["lo"]), "hi_x100": _x100(r["hi"])}
        res["references"][rname] = row
    return res


# --------------------------------------------------------------------------------------------------------------------------------
# layout helper + report + CLI
# --------------------------------------------------------------------------------------------------------------------------------
def spec_from_layout(root: str, sample_path: str) -> dict:
    """The directory convention ``RUNBOOK_NAVTRAIN_SELECT.md`` produces:
    ``<root>/<ckpt>/bridge/seam_R8_RT[_s1].npz`` + manifests, ``<root>/scores/<label>/<label>.{csv,counts.json,binding.json}``."""
    root = root.replace("\\", "/").rstrip("/")
    arms = {}
    for ck in (CHALLENGER, DEFAULT):
        for s in SEEDS:
            lab = arm_label(ck, s)
            sd = f"{root}/scores/{lab}"
            arms[f"{ck}_s{s}"] = {"csv": f"{sd}/{lab}.csv", "counts": f"{sd}/{lab}.counts.json", "binding": f"{sd}/{lab}.binding.json",
                                  "seam": f"{root}/{ck}/bridge/seam_{SEAM_ARM[s]}.npz", "bridge_dir": f"{root}/{ck}/bridge"}
    refs = {}
    for r in ("STOP", "CV"):
        lab = f"{LABEL_PREFIX}_ref_{r}"
        sd = f"{root}/scores/{lab}"
        refs[r] = {"csv": f"{sd}/{lab}.csv", "counts": f"{sd}/{lab}.counts.json", "binding": f"{sd}/{lab}.binding.json"}
    lab = f"{LABEL_PREFIX}_ref_PRIOR_ha0p"
    sd = f"{root}/scores/{lab}"
    refs["PRIOR_ha0p"] = {"csv": f"{sd}/{lab}.csv", "counts": f"{sd}/{lab}.counts.json", "binding": f"{sd}/{lab}.binding.json",
                          "seam": f"{root}/{DEFAULT}/bridge/seam_PRIOR_ha0p.npz"}
    return {"sample": sample_path, "arms": arms, "references": refs}


def render_md(res: dict) -> str:
    L = [f"# D-R8-CKPT-RULE decision -- {res['verdict']}", "",
         f"**Choice: {res['choice']}** ({res['n_tokens']} tokens, {res.get('n_clusters')} log clusters). Tier: {res['tier']}", ""]
    if res["verdict"].startswith("INCOMPLETE") and "per_seed" not in res:
        L += ["Reasons (the registered default EMA-48,852 holds; numbers withheld):", ""] + [f"* `{r}`" for r in res["reasons"]]
        return "\n".join(L) + "\n"
    L += ["| seed | 40k - 48,852 PDMS x100 | 95 % log-cluster interval | positive and excludes 0 |", "|---|---|---|---|"]
    for s, r in res["per_seed"].items():
        L.append(f"| {s} | {_x100(r['delta'])} | [{_x100(r['lo'])}, {_x100(r['hi'])}] | {r['positive_and_excludes_zero']} |")
    m = res["margin"]
    L += ["", f"inference floor |d_s0 - d_s1| = {res['inference_floor_x100']} ; margin = {m['definition']} = {m['value_x100']} ; "
          f"needed > {m['needed_gt_x100']} ; exceeds: {m['exceeds']}", "",
          f"controls read their known values: {res['controls']['all_read_known_values']}", ""]
    L += ["Reasons:"] + [f"* `{r}`" for r in res.get("reasons", [])] if res.get("reasons") else []
    return "\n".join(L) + "\n"


def run(spec_path: str, out_path: str, *, ci_path: str = "", expect_ci_sha256: str = "", md_path: str = "") -> dict:
    ci = load_ci(ci_path)
    if expect_ci_sha256 and ci.__r8sel_sha256__ != expect_ci_sha256:
        raise SystemExit(f"[select_ckpt_r8] ci.py sha256 {ci.__r8sel_sha256__} != expected {expect_ci_sha256}")
    sample, tables, refs, problems, prov = load_spec(spec_path)
    log_of = dict(sample.get("token_log", {}))
    res = decide(tables, sample, log_of, ci, pre_problems=problems, references=refs, prov=prov)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    if md_path:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_md(res))
    return res


def main(argv=None) -> int:
    if MUTATION:
        sys.exit("[select_ckpt_r8] R8SEL_MUTATION is set -- a mutation-test environment never produces a decision")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--layout-root", default="")
    ap.add_argument("--sample", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--ci-path", default="")
    ap.add_argument("--expect-ci-sha256", default="")
    a = ap.parse_args(argv)
    spec_path = a.spec
    if not spec_path:
        if not (a.layout_root and a.sample):
            sys.exit("[select_ckpt_r8] pass --spec, or --layout-root and --sample")
        spec_path = os.path.splitext(a.out)[0] + ".spec.json"
        with open(spec_path, "w", encoding="utf-8") as fh:
            json.dump(spec_from_layout(a.layout_root, a.sample), fh, indent=1)
    res = run(spec_path, a.out, ci_path=a.ci_path, expect_ci_sha256=a.expect_ci_sha256, md_path=a.md)
    print(f"ZZSELECT verdict={res['verdict']} choice={res['choice']} reasons={len(res.get('reasons', []))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
