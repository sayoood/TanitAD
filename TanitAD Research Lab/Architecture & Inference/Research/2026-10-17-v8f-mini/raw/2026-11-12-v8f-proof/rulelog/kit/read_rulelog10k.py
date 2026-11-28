#!/usr/bin/env python
"""read_rulelog10k.py -- read tool of card PREREG_V8F_PROOF_ADD137 (RULELOG10K): in the unsafe windows of the label form, does the rule's forecast of the lead or the rollout's execution lose the gap?
Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py, heading bound 0.2 1/m); not driving performance. CPU only, banked runs and banked rule logs, no new rollout.

Re-reads the dumps of the label form RLc (the subject) and, beside it, GEc at the model seeds 0 to 3, and their banked rule logs. The unsafe windows, the model arc, the model gap and the speeds come from the tool of
card ADD135 (read_labelunsafe10k.py, imported unchanged by path) and from the engine it binds; this file holds no definition of the engine. Per unsafe window the gap lost against the rule's target (S) is split into
the forecast part e_L and the execution part e_X (section 2 of the card); measures R1 to R4 with a cluster bootstrap over the evaluation episodes (2,000 draws, seed 137, 95 % interval); names of section 4;
controls w0 to w5 of section 5.

  python read_rulelog10k.py controls [--t1-root D] [--out-dir D] --tests-out DEV_OUT --tests-out POD_OUT
  python read_rulelog10k.py names    [--t1-root D] [--out-dir D]

Stage controls prints only `ctl w<k> PASS|FAIL <reason>` lines (a reason holds no value) and `ZZ READ-RULELOG10K controls all_pass True|False`; it writes RULELOG10K_CONTROLS.json.
Stage names is refused (rc 3) unless the controls file has all six controls passing and every input file hashes as the controls stage hashed it; it writes RULELOG10K_RESULT.json and RULELOG10K_VERDICT.md and prints
`ZZ READ-RULELOG10K names done`. An output file that exists is never overwritten. Exit codes: 0 stage done, 3 refused, 1 error.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD137"
PRE = "RULELOG10K"
TIER_SHORT = "T1-kin/hb0.2"
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m; not driving performance)"
TAGS = {"RLc": "I10RLcKF5Hb20", "GEc": "I10RQ50KF5Hb20gGEc"}
SUBJECT = "RLc"
POD_V = "/workspace/v8proof"
DEV_V = "F:/TanitAD-scratch/v8proof"
LU_POD = POD_V + "/labelunsafe/kit/read_labelunsafe10k.py"
LU_DEV = DEV_V + "/labelunsafe/kit/read_labelunsafe10k.py"
LU_SHA16 = "46be1fad5df31db3"
CFG = {
    "tags": TAGS, "subject": SUBJECT, "seeds": (0, 1, 2, 3), "expect_files": 141, "expect_unsafe": [22, 19, 22, 20],
    "expected_rates": {"RLc": {"per_seed": [0.0391, 0.0338, 0.0391, 0.0356], "pooled": 0.0369}, "GEc": {"pooled": 0.0480}},
    "every": 5, "tol_join": 1e-3, "tol_id": 1e-6, "n_boot": 2000, "boot_seed": 137, "lu_sha16": LU_SHA16,
}
DT, D_M, TAU_S = 0.2, 3.0, 1.2
NAME_BAR_SPLIT, NAME_BAR_RULE, NAME_BAR_BRAKE, SPEED_GAP = 0.5, 0.9, 0.5, 0.5
CONTROLS = tuple("w%d" % i for i in range(6))
CONTROLS_NAME = "RULELOG10K_CONTROLS.json"
RESULT_NAME = "RULELOG10K_RESULT.json"
VERDICT_NAME = "RULELOG10K_VERDICT.md"
N_FORECAST = PRE + "-K0-LEAD-FORECAST-CARRIES"
N_EXEC = PRE + "-K0-EXECUTION-CARRIES"
N_MIXED = PRE + "-K0-MIXED"
N_RULE_ON = PRE + "-K0-RULE-ON"
N_RULE_OFF = PRE + "-K0-RULE-OFF-IN-PART"
N_BRAKE = PRE + "-K0-BRAKING-LIMIT-BINDS"
N_NOBRAKE = PRE + "-K0-BRAKING-LIMIT-DOES-NOT-BIND"
REQ = ("ci", "t0", "k", "active", "present", "sat_exec", "g0", "v_L", "a_m")
NOT_READ = "not read by this card (the family reads of these banked runs are in rows 193 and 202); n = 0 here"
FAMILIES = [
    "LONGITUDINAL: read here (distance keeping: the gap lost against the rule's target split into the lead's forecast and the rollout's execution), %s, not driving performance" % TIER_SHORT,
    "LATERAL: " + NOT_READ, "TACTICAL: " + NOT_READ, "STRATEGIC: " + NOT_READ,
]
LIMITS = ("%s only, no re-perception, the lead moves as logged; windows of one line over four model seeds (about 83, 31 distinct), so the intervals are wide; the cap is written with the realised speed of the step before, "
          "the rule itself uses the planned one; the split reads the last tick before the first unsafe step alone, a loss built up over earlier ticks sits in g_m(k_t); the label form is a reference with label values and not "
          "a candidate. The interval answers what another draw of episodes would say, not what another training run would say." % TIER_SHORT)

LU = None


class LoadError(Exception):
    pass


# ============================================================================================ small helpers
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lf_sha(p):
    """sha256 of a file with every carriage return removed (the hash of the tool of card ADD135 is given in that form)"""
    with open(p, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r", b"")).hexdigest()


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def write_new(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def dump_aggregate(folder):
    h, n = hashlib.sha256(), 0
    if os.path.isdir(folder):
        for f in sorted(x for x in os.listdir(folder) if re.match(r"ep.*\.npz$", x)):
            h.update(("%s %s\n" % (f, sha_file(os.path.join(folder, f)))).encode("utf-8"))
            n += 1
    return {"path": os.path.abspath(folder), "sha256": h.hexdigest(), "n_files": n}


def inputs_changed(inputs):
    fl, rd = (inputs or {}).get("files"), (inputs or {}).get("run_dumps")
    if not isinstance(fl, dict) or not fl or not isinstance(rd, dict) or not rd:
        return ["the controls file records no input hashes"]
    bad = []
    for lab, v in sorted(fl.items()):
        p, want = (v or {}).get("path"), (v or {}).get("sha256")
        got = sha_file(p) if p and os.path.isfile(p) else None
        if got != want:
            bad.append("%s differs from the controls stage" % lab)
    for lab, v in sorted(rd.items()):
        now = dump_aggregate((v or {}).get("path", ""))
        if now["sha256"] != (v or {}).get("sha256") or now["n_files"] != (v or {}).get("n_files"):
            bad.append("dump files of %s differ from the controls stage" % lab)
    return bad


# ============================================================================================ the tool of card ADD135 (imported by path, unchanged)
def default_lu_path():
    if os.environ.get("LABELUNSAFE_TOOL"):
        return os.environ["LABELUNSAFE_TOOL"]
    return LU_POD if os.path.isfile(LU_POD) else LU_DEV


def bind_lu(path):
    """imports read_labelunsafe10k.py by path (the module object is reused when the same file is already loaded) and keeps it as LU"""
    global LU
    p = os.path.abspath(path or default_lu_path())
    if not os.path.isfile(p):
        raise LoadError("the tool of card ADD135 cannot be found")
    d = os.path.dirname(p)
    if d not in sys.path:
        sys.path.insert(0, d)
    old = sys.modules.get("read_labelunsafe10k")
    if old is not None and norm(getattr(old, "__file__", "") or "") == norm(p):
        LU = old
    else:
        spec = importlib.util.spec_from_file_location("read_labelunsafe10k", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["read_labelunsafe10k"] = mod
        spec.loader.exec_module(mod)
        LU = mod
    return LU


# ============================================================================================ rule logs
def log_paths(root, tag, s):
    d = os.path.join(LU.run_folder(root, tag, s), "ticks")
    return [os.path.join(d, "rule_log_%s_s%d.npz" % (tag, s)), os.path.join(d, "rule_log_mid_%s_s%d.npz" % (tag, s))]


def read_logs(root, tag, s):
    """the tick log (rows at the steps 0, 10, 20) and the later-look log (rows at 5, 15, 25) of a run, concatenated, the fields of REQ"""
    cols = {k: [] for k in REQ}
    for p in log_paths(root, tag, s):
        if not os.path.isfile(p):
            raise LoadError("a rule log file is missing")
        with np.load(p, allow_pickle=False) as z:
            for k in REQ:
                if k not in z.files:
                    raise LoadError("a rule log lacks a field")
                cols[k].append(np.asarray(z[k]))
    return {k: np.concatenate(v) for k, v in cols.items()}


def _same_row(L, a, b):
    for k in REQ:
        x, y = float(L[k][a]), float(L[k][b])
        if not (x == y or (math.isnan(x) and math.isnan(y))):
            return False
    return True


def index_rows(L, keyset):
    """{(ci, t0, k): row} for the rows of the windows in keyset ((ci, t0) pairs of the grid); rows of other windows are cut. Several rows with one key and all fields equal count as one row; with differing fields
    the key maps to -1 (ambiguous)"""
    ci, t0, k = np.asarray(L["ci"], np.int64), np.asarray(L["t0"], np.int64), np.asarray(L["k"], np.int64)
    out = {}
    for r in range(len(ci)):
        w = (int(ci[r]), int(t0[r]))
        if w not in keyset:
            continue
        key = (w[0], w[1], int(k[r]))
        if key not in out:
            out[key] = r
        elif out[key] != -1 and not _same_row(L, out[key], r):
            out[key] = -1
    return out


def tick_of(kstar, every):
    """the last rule tick strictly before the first unsafe step kstar (steps from the window origin; ticks at 0, every, 2 every, ...)"""
    return int(every) * ((int(kstar) - 1) // int(every))


# ============================================================================================ per-window table and the split
KEYS = ("ep", "ks", "kt", "i", "am_kt", "am_ks", "gm_kt", "gm_ks", "vmp", "g0", "vL", "on", "sat")


def run_table(E, w, key, L, cfg, check_g0=True):
    """-> (table of arrays over the unsafe windows of the run, set of problems (texts without values)). Rows of the log are joined to windows by (ci, t0); the engine gives a_m, g_m and the speeds."""
    D, A, S = w.D[key], w.A[key], w.S[key]
    every, tol = int(cfg["every"]), float(cfg["tol_join"])
    G = w.L["G"]
    ci_all, ws_all = np.asarray(D["clip_index"], np.int64), np.asarray(D["ws"], np.int64)
    idx = index_rows(L, set(zip(ci_all.tolist(), ws_all.tolist())))
    am_eng = E.H.progress(D["g"], D["cl"])
    problems, recs = set(), []
    for i in np.flatnonzero(S["sel"]):
        ks = int(S["first"][i])
        kt = tick_of(ks, every)
        ci, t0 = int(ci_all[i]), int(ws_all[i])
        rows, ok = {}, True
        for k in range(0, kt + 1, every):
            r = idx.get((ci, t0, k))
            if r is None:
                problems.add("a rule tick of an unsafe window is missing from the log")
                ok = False
                continue
            if r == -1:
                problems.add("a rule tick of an unsafe window has several differing rows")
                ok = False
                continue
            rows[k] = r
            am_e = 0.0 if k == 0 else float(am_eng[i, k - 1])
            gm_e = float(G[i, 0]) if k == 0 else float(A["gm"][i, k - 1])
            if not abs(float(L["a_m"][r]) - am_e) <= tol:
                problems.add("the logged arc of the model differs from the engine's at a tick")
            g0 = float(L["g0"][r])
            if check_g0 and math.isfinite(g0) and not abs(g0 - gm_e) <= tol:
                problems.add("the logged gap g0 differs from the engine's model gap at a tick")
        if not ok:
            continue
        r = rows[kt]
        on = bool(L["active"][r]) and bool(L["present"][r])
        g0, vL = float(L["g0"][r]), float(L["v_L"][r])
        if on and not (math.isfinite(g0) and math.isfinite(vL)):
            problems.add("a window with the rule on has a non finite g0 or v_L")
            continue
        recs.append({
            "ep": int(w.inv[i]), "ks": ks, "kt": kt, "i": ks - kt,
            "am_kt": 0.0 if kt == 0 else float(am_eng[i, kt - 1]), "am_ks": float(am_eng[i, ks - 1]),
            "gm_kt": float(G[i, 0]) if kt == 0 else float(A["gm"][i, kt - 1]), "gm_ks": float(A["gm"][i, ks - 1]),
            "vmp": float(D["v0"][i]) if ks == 1 else float(A["vm"][i, ks - 2]),
            "g0": g0, "vL": vL, "on": on, "sat": int(L["sat_exec"][r])})
    T = {k: np.array([x[k] for x in recs], dtype=(bool if k == "on" else (np.int64 if k in ("ep", "ks", "kt", "i", "sat") else np.float64))) for k in KEYS}
    return T, problems


def parts(T, vL=None):
    """the split of section 2 of the card: S, e_L and e_X, and (beside) e_G = g0 minus the model gap at the tick; vL (array) replaces the logged lead speed the rule assumed.
    P(k) = a_m(k) + g_m(k); S = max(3, 1.2 v_m(k* - 1)) - g_m(k*); e_L = P(k_t) + v_L 0.2 i - P(k*); e_X = [a_m(k*) - a_m(k_t)] - [g0 + v_L 0.2 i - cap]"""
    v = T["vL"] if vL is None else np.asarray(vL, np.float64)
    cap = np.maximum(D_M, TAU_S * T["vmp"])
    adv = v * DT * T["i"]
    p_kt, p_ks = T["am_kt"] + T["gm_kt"], T["am_ks"] + T["gm_ks"]
    return {"S": cap - T["gm_ks"], "eL": p_kt + adv - p_ks, "eX": (T["am_ks"] - T["am_kt"]) - (T["g0"] + adv - cap), "eG": T["g0"] - T["gm_kt"], "P_kt": p_kt, "P_ks": p_ks}


def realised_speed(T):
    """the lead's mean speed over the interval from the tick to the first unsafe step, along the logged path"""
    d = parts(T)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (d["P_ks"] - d["P_kt"]) / (DT * T["i"])


def concat(tabs):
    if not tabs:
        return {k: np.zeros(0) for k in KEYS}
    return {k: np.concatenate([t[k] for t in tabs]) for k in KEYS}


# ============================================================================================ measures and bootstrap
def med(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def share(x):
    x = np.asarray(x)
    return float(np.mean(x)) if len(x) else float("nan")


def measures_of(T, n_ep, idx):
    """R1 to R4 with the cluster bootstrap over the evaluation episodes (the draws idx are shared by every measure). R1 and R2 are over the unsafe windows, R3 and R4 over those with the rule on at the tick"""
    ep, on = T["ep"], np.asarray(T["on"], bool)

    def sums(x):
        return np.bincount(ep, weights=np.asarray(x, np.float64), minlength=n_ep) if len(ep) else np.zeros(n_ep)
    d = parts(T) if len(ep) else {"S": np.zeros(0), "eL": np.zeros(0), "eX": np.zeros(0)}
    one = np.ones(len(ep))
    n_all, n_on = int(len(ep)), int(on.sum())
    out = {}
    out["R1"] = _read(sums(on), sums(one), n_all, idx)
    out["R2"] = _read(sums(T["sat"] > 0), sums(one), n_all, idx)
    out["R3"] = _read(sums(np.where(on, d["eL"], 0.0)), sums(np.where(on, d["S"], 0.0)), n_on, idx)
    out["R4"] = _read(sums(on & (d["eL"] > d["eX"])), sums(on), n_on, idx)
    return out


def _read(numE, denE, n, idx):
    v, iv, _ = LU.share_read(numE, denE, idx)
    return {"value": v, "interval": iv, "n": int(n)}


def plain_values(T):
    """R1 to R4 as plain values (no interval), for the per-seed lines"""
    on = np.asarray(T["on"], bool)
    if not len(on):
        return {k: float("nan") for k in ("R1", "R2", "R3", "R4")}
    d = parts(T)
    sd = float(d["S"][on].sum())
    return {"R1": share(on), "R2": share(T["sat"] > 0), "R3": (float(d["eL"][on].sum()) / sd) if on.any() and sd > 0 else float("nan"),
            "R4": share((d["eL"] > d["eX"])[on]) if on.any() else float("nan")}


def beside_of(T, n_ep, idx, gec=False):
    on = np.asarray(T["on"], bool)
    d = parts(T) if len(on) else {"S": np.zeros(0), "eL": np.zeros(0), "eX": np.zeros(0), "eG": np.zeros(0)}
    v_real = realised_speed(T) if len(on) else np.zeros(0)
    Ton = {k: T[k][on] for k in KEYS}
    out = {"tier": TIER_SHORT, "n_unsafe_windows": int(len(on)), "n_rule_on_windows": int(on.sum()),
           "medians_rule_on": {"S": med(d["S"][on]), "e_L": med(d["eL"][on]), "e_X": med(d["eX"][on]), "i": med(Ton["i"])},
           "medians_all_unsafe": {"S": med(d["S"]), "i": med(T["i"])},
           "share_lead_mean_speed_more_than_0.5_below_v_L": share((T["vL"] - v_real > SPEED_GAP)[on]) if on.any() else float("nan"),
           "share_negative_e_X": share((d["eX"] < 0)[on]) if on.any() else float("nan"),
           "share_tick_at_step_0": share(T["kt"] == 0)}
    if gec:
        out["third_part_g0_minus_label_gap_at_the_tick"] = {"median": med(d["eG"][on]), "share_positive": share((d["eG"] > 0)[on]) if on.any() else float("nan")}
    return out


def read_all(cfg, w, tabs):
    n_ep = w.n_ep
    idx = LU.make_draws(n_ep, cfg["n_boot"], cfg["boot_seed"])
    seeds = list(cfg["seeds"])
    res = {}
    for arm in cfg["tags"]:
        per = [tabs[(arm, s)] for s in seeds]
        T = concat(per)
        res[arm] = {"T": T, "m": measures_of(T, n_ep, idx), "per_seed": {"R1": [], "R2": [], "R3": [], "R4": []}, "per_seed_n": [int(len(t["on"])) for t in per]}
        for t in per:
            pv = plain_values(t)
            for k in pv:
                res[arm]["per_seed"][k].append(pv[k])
    sub = cfg["subject"]
    beside = {"tier": TIER_SHORT, "n_evaluation_episodes": int(n_ep), "subject": dict(beside_of(res[sub]["T"], n_ep, idx), per_seed_R1_to_R4=res[sub]["per_seed"], n_unsafe_windows_per_seed=res[sub]["per_seed_n"])}
    for arm in cfg["tags"]:
        if arm != sub:
            beside[arm] = dict(beside_of(res[arm]["T"], n_ep, idx, gec=True), R1_to_R4=res[arm]["m"], per_seed_R1_to_R4=res[arm]["per_seed"], n_unsafe_windows_per_seed=res[arm]["per_seed_n"])
    return res[sub]["m"], beside


# ============================================================================================ names (two routines that must agree)
def _iv(m, k):
    lo, hi = m[k]["interval"]
    return float(lo), float(hi)


def split_name(m):
    lo, hi = _iv(m, "R3")
    if lo >= NAME_BAR_SPLIT:
        return N_FORECAST
    if hi <= NAME_BAR_SPLIT:
        return N_EXEC
    return N_MIXED


def split_name_b(m):
    lo, hi = _iv(m, "R3")
    a = math.isfinite(lo) and (lo - NAME_BAR_SPLIT >= 0.0)
    b = math.isfinite(hi) and (NAME_BAR_SPLIT - hi >= 0.0)
    return [N_MIXED, N_FORECAST, N_EXEC, N_EXEC][int(a) + 2 * int((not a) and b)]


def rule_name(m):
    return N_RULE_ON if _iv(m, "R1")[0] >= NAME_BAR_RULE else N_RULE_OFF


def rule_name_b(m):
    lo = _iv(m, "R1")[0]
    return (N_RULE_OFF, N_RULE_ON)[int(math.isfinite(lo) and lo - NAME_BAR_RULE >= 0.0)]


def brake_name(m):
    return N_BRAKE if _iv(m, "R2")[0] >= NAME_BAR_BRAKE else N_NOBRAKE


def brake_name_b(m):
    lo = _iv(m, "R2")[0]
    return (N_NOBRAKE, N_BRAKE)[int(math.isfinite(lo) and lo - NAME_BAR_BRAKE >= 0.0)]


def derive_names(m):
    a = [split_name(m), rule_name(m), brake_name(m)]
    b = [split_name_b(m), rule_name_b(m), brake_name_b(m)]
    if a != b:
        raise SystemExit("REFUSED: the two routines of the names disagree")
    return a


# ============================================================================================ controls
def cfg_lu(cfg):
    c = dict(LU.CFG)
    c.update({"tags": dict(cfg["tags"]), "seeds": tuple(cfg["seeds"]), "expect_files": cfg["expect_files"], "expected": cfg["expected_rates"]})
    return c


def ctl_w0(E, cfg, w, lu_path):
    """windows: the tool of card ADD135 hashes as recorded; its own identity control on the rates of the two arms reads PASS on these runs; the unsafe windows and first unsafe steps of this tool are the
    tool's (per run, with its second routine agreeing on the first unsafe step); the subject's counts per model seed are the card's"""
    if lf_sha(lu_path)[:16] != cfg["lu_sha16"]:
        return False, "the tool of card ADD135 does not hash as recorded"
    if not w.complete:
        return False, "the runs could not be read"
    ok, why = LU.ctl_u0(cfg_lu(cfg), w)
    if not ok:
        return False, "the tool of card ADD135 reads its identity control as failed on these runs"
    sub = cfg["subject"]
    for s, want in zip(cfg["seeds"], cfg["expect_unsafe"]):
        if int(w.S[(sub, s)]["sel"].sum()) != int(want):
            return False, "the unsafe windows of %s at model seed %d are not those of card ADD135" % (sub, s)
    for (arm, s), D in w.D.items():
        S2 = LU.second_routine(E, D["g"], D["cl"], w.L)
        sel = w.S[(arm, s)]["sel"]
        if not np.array_equal(np.asarray(S2["first"])[sel], np.asarray(w.S[(arm, s)]["first"])[sel]) or not bool((np.asarray(w.S[(arm, s)]["first"])[sel] >= 1).all()):
            return False, "a first unsafe step differs from the tool's second routine (%s_s%d)" % (arm, s)
    return True, ""


def ctl_w1(tabs, problems, cfg):
    """log join: every unsafe window has its rule ticks (steps 0, every, ... up to the last tick before the first unsafe step) once in the log of its run; at each tick the logged arc of the model equals the
    engine's within 1e-3 m, and for the label form the logged g0 equals the engine's model gap within 1e-3 m"""
    for key, pr in problems.items():
        if pr:
            return False, "%s_s%d: %s" % (key[0], key[1], sorted(pr)[0])
    for key, T in tabs.items():
        if key[0] == cfg["subject"] and not len(T["on"]):
            return False, "no unsafe window of the subject at model seed %d" % key[1]
    return True, ""


def ctl_w2(tabs, cfg):
    """identity: S = e_L + e_X within 1e-6 m in every window with the rule on at the tick (subject); for the candidate S = e_L + e_X + e_G, e_G = g0 minus the model gap at the tick"""
    n = 0
    for key, T in tabs.items():
        on = np.asarray(T["on"], bool)
        if not on.any():
            continue
        d = parts(T)
        res = d["S"] - d["eL"] - d["eX"] - (0.0 if key[0] == cfg["subject"] else d["eG"])
        if not bool(np.all(np.abs(res[on]) <= cfg["tol_id"])):
            return False, "S differs from the sum of its parts in a window (%s_s%d)" % key
        n += int(on.sum())
    if n == 0:
        return False, "no window with the rule on to read"
    return True, ""


def ctl_w3(tabs, cfg):
    """the forecast part reads the forecast alone: with v_L replaced by the lead's realised mean speed over the interval e_L is 0 within 1e-6 m and e_X equals S (for the candidate e_X + e_G equals S); with v_L
    raised by 1.0 m/s e_L rises by 0.2 m times i, in every window with the rule on at the tick"""
    tol, n = cfg["tol_id"], 0
    for key, T in tabs.items():
        on = np.asarray(T["on"], bool)
        if not on.any():
            continue
        vr = realised_speed(T)
        d = parts(T, vr)
        extra = 0.0 if key[0] == cfg["subject"] else d["eG"]
        if not bool(np.all(np.abs(d["eL"][on]) <= tol)) or not bool(np.all(np.abs((d["eX"] + extra - d["S"])[on]) <= tol)):
            return False, "with the realised lead speed the forecast part is not 0 or the execution part is not S (%s_s%d)" % key
        d0, d1 = parts(T), parts(T, T["vL"] + 1.0)
        if not bool(np.all(np.abs((d1["eL"] - d0["eL"] - DT * T["i"])[on]) <= tol)):
            return False, "raising the lead speed by 1.0 m/s does not raise the forecast part by 0.2 m times i (%s_s%d)" % key
        n += int(on.sum())
    if n == 0:
        return False, "no window with the rule on to read"
    return True, ""


def ctl_w4(paths):
    """tests: each of the two outputs (dev box, pod) of this tool's test file has a last non-empty line `N passed` (N > 0) with no failed or error count, names test_read_rulelog10k, and its path has no
    component `kit` (and is not inside the tool's folder)"""
    if len(paths) != 2:
        return False, "two test outputs are needed (dev box and pod)"
    here = norm(os.path.dirname(os.path.abspath(__file__)))
    for p in paths:
        if not os.path.isfile(p):
            return False, "a test output is missing"
        parts_ = [x.lower() for x in re.split(r"[\\/]", os.path.abspath(p))]
        if "kit" in parts_ or norm(p).startswith(here + os.sep):
            return False, "a test output lies inside the kit folder"
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not LU.pytest_ok(text) or "test_read_rulelog10k" not in text:
            return False, "a test output does not read N passed with no failure"
    return True, ""


def ctl_w5(cfg, w, root):
    """counts: 141 dump files, the done marker and the rule log files (tick log and later-look log) for each of the eight runs"""
    for arm, s in LU.run_keys(cfg):
        key = "%s_s%d" % (arm, s)
        if not w.done.get(key):
            return False, "done marker missing for %s" % key
        if w.n_files.get(key) != cfg["expect_files"]:
            return False, "dump file count of %s is not the card's" % key
        for p in log_paths(root, cfg["tags"][arm], s):
            if not os.path.isfile(p):
                return False, "rule log file missing for %s" % key
    if w.problems or not w.complete:
        return False, "windows of the runs are missing or differ"
    return True, ""


def load_tables(E, cfg, w, root):
    tabs, problems = {}, {}
    for arm, s in LU.run_keys(cfg):
        L = read_logs(root, cfg["tags"][arm], s)
        tabs[(arm, s)], problems[(arm, s)] = run_table(E, w, (arm, s), L, cfg, check_g0=(arm == cfg["subject"]))
    return tabs, problems


def run_controls(E, cfg, root, paths, tests, lu_path):
    """-> (dict w<k> -> (ok, reason), world or None, tables or None)"""
    res, w, tabs = {}, None, None
    try:
        w = LU.load_world(E, cfg_lu(cfg), root, paths["join"], paths["grid"])
    except Exception as ex:                                                        # noqa: BLE001
        why = "inputs could not be read (%s)" % type(ex).__name__
        for c in ("w0", "w1", "w2", "w3", "w5"):
            res[c] = (False, why)
    if w is not None:
        try:
            tabs, problems = load_tables(E, cfg, w, root)
        except Exception as ex:                                                    # noqa: BLE001
            tabs, problems = None, None
            why = "the rule logs could not be read (%s)" % type(ex).__name__
            for c in ("w1", "w2", "w3"):
                res[c] = (False, why)
        jobs = [("w0", lambda: ctl_w0(E, cfg, w, lu_path)), ("w5", lambda: ctl_w5(cfg, w, root))]
        if tabs is not None:
            jobs += [("w1", lambda: ctl_w1(tabs, problems, cfg)), ("w2", lambda: ctl_w2(tabs, cfg)), ("w3", lambda: ctl_w3(tabs, cfg))]
        for c, fn in jobs:
            try:
                res[c] = fn()
            except Exception as ex:                                                # noqa: BLE001
                res[c] = (False, "the control raised %s" % type(ex).__name__)
    try:
        res["w4"] = ctl_w4(tests)
    except Exception as ex:                                                        # noqa: BLE001
        res["w4"] = (False, "the control raised %s" % type(ex).__name__)
    return res, w, tabs


# ============================================================================================ inputs (hashes)
def collect_inputs(E, cfg, root, paths, tests, lu_path):
    files = {"this_tool": os.path.abspath(__file__), "labelunsafe_tool": lu_path, "join": paths["join"], "grid": paths["grid"]}
    files.update(E.files)
    for i, t in enumerate(tests):
        files["tests_out_%d" % i] = t
    for arm, s in LU.run_keys(cfg):
        files["done_%s_s%d" % (arm, s)] = os.path.join(LU.run_folder(root, cfg["tags"][arm], s), "T1_DONE")
        for j, p in enumerate(log_paths(root, cfg["tags"][arm], s)):
            files["rulelog%d_%s_s%d" % (j, arm, s)] = p
    return {"files": {lab: {"path": os.path.abspath(p), "sha256": sha_file(p) if os.path.isfile(p) else None} for lab, p in files.items() if p},
            "run_dumps": {"%s_s%d" % (arm, s): dump_aggregate(LU.dump_folder(root, cfg["tags"][arm], s)) for arm, s in LU.run_keys(cfg)}}


# ============================================================================================ files
def result_json(cfg, m, beside, controls_sha):
    return {"card": CARD, "tier": TIER, "names": derive_names(m), "measures": m, "beside": beside, "families": FAMILIES,
            "estimator": "cluster bootstrap over the evaluation episodes, %d draws, seed %d, 95 percent interval; the interval answers what another draw of episodes would say, not what another training run would say" % (cfg["n_boot"], cfg["boot_seed"]),
            "controls_sha256": controls_sha}


def _f(x, nd=4):
    return "nan" if x is None or not math.isfinite(x) else ("%." + str(nd) + "f") % x


def verdict_md(res):
    m, b = res["measures"], res["beside"]
    L = ["# VERDICT RULELOG10K (card %s)" % CARD, "",
         "Tier: %s. Not driving performance. Controls w0 to w5 passed before this file was written (controls file sha256 %s)." % (res["tier"], str(res["controls_sha256"])[:16]), "",
         "## Names", "", "* Split: **%s**" % res["names"][0], "* Rule state: **%s**" % res["names"][1], "* Braking limit: **%s**" % res["names"][2], "",
         "## Measures on the unsafe windows of RLc (pooled over the four model seeds; %s)" % TIER_SHORT, "", "| measure | value | 95 % interval | n |", "|---|---|---|---|"]
    txt = {"R1": "R1 share with the rule on at the tick (unsafe windows)", "R2": "R2 share with sat_exec above 0 at the tick (unsafe windows)",
           "R3": "R3 pooled share of the loss carried by the forecast (rule-on windows)", "R4": "R4 share of rule-on windows with e_L larger than e_X"}
    for k in ("R1", "R2", "R3", "R4"):
        L.append("| %s | %s | [%s, %s] | %d |" % (txt[k], _f(m[k]["value"]), _f(m[k]["interval"][0]), _f(m[k]["interval"][1]), m[k]["n"]))
    s = b["subject"]
    L += ["", "## Beside (no name)", "",
          "* Medians over the rule-on windows: S %s m, e_L %s m, e_X %s m, i %s steps; over all unsafe windows: S %s m, i %s steps." % (
              _f(s["medians_rule_on"]["S"], 3), _f(s["medians_rule_on"]["e_L"], 3), _f(s["medians_rule_on"]["e_X"], 3), _f(s["medians_rule_on"]["i"], 1), _f(s["medians_all_unsafe"]["S"], 3), _f(s["medians_all_unsafe"]["i"], 1)),
          "* Share of rule-on windows in which the lead's mean speed over the interval lies more than 0.5 m/s below v_L: %s; share with a negative e_X: %s; share with the tick at step 0: %s." % (
              _f(s["share_lead_mean_speed_more_than_0.5_below_v_L"]), _f(s["share_negative_e_X"]), _f(s["share_tick_at_step_0"])),
          "* Unsafe windows per model seed: %s; rule-on windows: %d of %d." % (s["n_unsafe_windows_per_seed"], s["n_rule_on_windows"], s["n_unsafe_windows"]),
          "* R1 to R4 per model seed (values): %s." % {k: [_f(x) for x in v] for k, v in s["per_seed_R1_to_R4"].items()}]
    for arm, v in b.items():
        if arm in ("tier", "n_evaluation_episodes", "subject"):
            continue
        L.append("* %s (g0 is the camera gap, P is read with the label gap): " % arm + "; ".join("%s %s [%s, %s] n %d" % (k, _f(v["R1_to_R4"][k]["value"]), _f(v["R1_to_R4"][k]["interval"][0]), _f(v["R1_to_R4"][k]["interval"][1]), v["R1_to_R4"][k]["n"]) for k in ("R1", "R2", "R3", "R4")))
        L.append("* %s third part (g0 minus the label gap at the tick): median %s m, share positive %s; medians over rule-on windows: S %s, e_L %s, e_X %s." % (
            arm, _f(v["third_part_g0_minus_label_gap_at_the_tick"]["median"], 3), _f(v["third_part_g0_minus_label_gap_at_the_tick"]["share_positive"]), _f(v["medians_rule_on"]["S"], 3), _f(v["medians_rule_on"]["e_L"], 3), _f(v["medians_rule_on"]["e_X"], 3)))
    L += ["", "## Families", ""] + ["* " + x for x in res["families"]] + ["", "## Limits", "", LIMITS, ""]
    return "\n".join(L)


# ============================================================================================ stages
def refuse(why):
    print("REFUSED: %s" % why)
    return 3


def paths_of(a):
    d = LU.default_paths()
    for k in ("headway", "q3", "stageb", "tev", "longroll", "join", "grid"):
        v = getattr(a, k, None)
        if v:
            d[k] = v
    return d


def stage_controls(a, cfg):
    out = os.path.join(a.out_dir, CONTROLS_NAME)
    if os.path.exists(out):
        return refuse("%s exists" % out)
    try:
        lu_path = os.path.abspath(a.lu_tool or default_lu_path())
        bind_lu(lu_path)
    except LoadError as ex:
        return refuse(str(ex))
    os.makedirs(a.out_dir, exist_ok=True)
    paths = paths_of(a)
    tests = list(a.tests_out or [])
    E = LU.bind_engine(paths)
    res, _, _ = run_controls(E, cfg, a.t1_root, paths, tests, lu_path)
    rec = {"card": CARD}
    for c in CONTROLS:
        rec["pass_" + c] = bool(res[c][0])
        print("ctl %s %s%s" % (c, "PASS" if res[c][0] else "FAIL", "" if res[c][0] else " " + res[c][1]))
    rec["all_pass"] = bool(all(res[c][0] for c in CONTROLS))
    rec["reasons"] = {c: res[c][1] for c in CONTROLS if not res[c][0]}
    rec["inputs"] = collect_inputs(E, cfg, a.t1_root, paths, tests, lu_path)
    write_new(out, json.dumps(clean(rec), indent=1))
    print("ZZ READ-RULELOG10K controls all_pass %s" % rec["all_pass"])
    return 0


def stage_names(a, cfg):
    cpath = os.path.join(a.out_dir, CONTROLS_NAME)
    if not os.path.isfile(cpath):
        return refuse("no controls file")
    try:
        with open(cpath, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except ValueError:
        return refuse("the controls file cannot be read")
    if not isinstance(rec, dict) or rec.get("card") != CARD:
        return refuse("the controls file is not that of the card")
    for c in CONTROLS:
        if rec.get("pass_" + c) is not True:
            return refuse("control %s is not recorded as passed" % c)
    if rec.get("all_pass") is not True:
        return refuse("all_pass is not True")
    bad = inputs_changed(rec.get("inputs"))
    if bad:
        return refuse("an input differs from the controls stage: " + "; ".join(bad[:4]))
    rpath, vpath = os.path.join(a.out_dir, RESULT_NAME), os.path.join(a.out_dir, VERDICT_NAME)
    if os.path.exists(rpath) or os.path.exists(vpath):
        return refuse("an output file exists")
    fl = rec["inputs"]["files"]
    try:
        lu_path = os.path.abspath(a.lu_tool or default_lu_path())
        bind_lu(lu_path)
    except LoadError as ex:
        return refuse(str(ex))
    if norm(__file__) != norm((fl.get("this_tool") or {}).get("path") or ""):
        return refuse("this tool is not the one hashed by the controls stage")
    if norm(lu_path) != norm((fl.get("labelunsafe_tool") or {}).get("path") or ""):
        return refuse("the tool of card ADD135 is not the one hashed by the controls stage")
    if lf_sha(lu_path)[:16] != cfg["lu_sha16"]:
        return refuse("the tool of card ADD135 does not hash as recorded")
    paths = paths_of(a)
    E = LU.bind_engine(paths)
    for lab, p in E.files.items():
        if norm(p) != norm((fl.get(lab) or {}).get("path") or ""):
            return refuse("the engine file %s is not the one hashed by the controls stage" % lab)
    w = LU.load_world(E, cfg_lu(cfg), a.t1_root, paths["join"], paths["grid"])
    if not w.complete or w.problems:
        return refuse("the runs are incomplete")
    tabs, problems = load_tables(E, cfg, w, a.t1_root)
    if any(problems.values()):
        return refuse("a rule log does not join to its windows")
    m, beside = read_all(cfg, w, tabs)
    res = result_json(cfg, m, beside, sha_file(cpath))
    write_new(rpath, json.dumps(clean(res), indent=1))
    write_new(vpath, verdict_md(clean(res)))
    print("names: %s | %s | %s" % tuple(res["names"]))
    print("ZZ READ-RULELOG10K names done")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description="read tool of card %s" % CARD)
    ap.add_argument("stage", choices=("controls", "names"))
    ap.add_argument("--t1-root", default=POD_V + "/integ10k/t1")
    ap.add_argument("--out-dir", default=POD_V + "/rulelog/read_out")
    ap.add_argument("--tests-out", action="append", default=None, help="pytest output of the dev box, then of the pod (outside the kit folder)")
    ap.add_argument("--lu-tool", default=None, help="read_labelunsafe10k.py of card ADD135 (default: env LABELUNSAFE_TOOL, else the pod path, else the dev path)")
    for k in ("headway", "q3", "stageb", "tev", "longroll", "join", "grid"):
        ap.add_argument("--" + k, default=None)
    return ap


def main(argv=None, cfg=None):
    a = build_parser().parse_args(argv)
    cfg = cfg or CFG
    try:
        return stage_controls(a, cfg) if a.stage == "controls" else stage_names(a, cfg)
    except LoadError as ex:
        return refuse(str(ex))


if __name__ == "__main__":
    sys.exit(main())
