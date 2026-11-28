"""hc_common.py -- card PREREG_V8F_PROOF_ADD152 (HEADCLOSE10K): the card's own controls (t1 to t3 from the box-2 files, t4 run line, t5 fed values) and the combined
judgement of card section 5. Pure functions of files; tested in test_read_headclose.py on synthetic data. Tier T1-kin/hb0.2; never driving performance."""
import glob
import io
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(HERE, "table")
STRIDE = 1 << 20
TOL_EST = 1e-9
TOL_SRC = 1e-5            # the tick-level closing passes through float32 in the reader feed; the later looks are float64
TOL_VL = 1e-5
FAMS4 = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


def jl(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception:                                                                   # noqa: BLE001
        return None


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_table(path):
    with np.load(path, allow_pickle=False) as z:
        key = z["clip_index"].astype(np.int64) * STRIDE + z["frame"].astype(np.int64)
        v = z["est"].astype(np.float64)
    o = np.argsort(key, kind="stable")
    return key[o], v[o]


def table_at(T, ci, fr):
    key, val = T
    k = np.asarray(ci, np.int64) * STRIDE + np.asarray(fr, np.int64)
    pos = np.searchsorted(key, k)
    posc = np.minimum(pos, len(key) - 1)
    hit = (pos < len(key)) & (key[posc] == k) & (np.asarray(fr) >= 0)
    return np.where(hit, val[posc], np.nan)


def c_tables(pull, table_dir=TABLE_DIR, zero_runs=("H_RQ50KVz_s0", "H_RQ50KVz_P30_s0"), head_sha=None, zero_sha=None):
    """t1 (TABLE.json of the box-2 table job: all controls passed; the pulled tables' sha256 equal its record and the queue's pins) and t2 / t3 (CONTROL_ZERO.json
    written on box 2 by the queue in each zero-table run folder)."""
    t = jl(os.path.join(table_dir, "TABLE.json")) or {}
    rec = t.get("tables") or {}
    got = {k: sha(os.path.join(table_dir, f)) if os.path.isfile(os.path.join(table_dir, f)) else None for k, f in (("head", "HEAD_TABLE.npz"), ("zero", "ZERO_TABLE.npz"))}
    t1 = {"controls_all_pass": t.get("controls_all_pass") is True, "eval141": t.get("eval141"), "sha_equal_record": got == {"head": rec.get("head"), "zero": rec.get("zero")},
          "sha_equal_queue_pins": (head_sha is None or got["head"] == head_sha) and (zero_sha is None or got["zero"] == zero_sha)}
    t1["PASS"] = bool(t1["controls_all_pass"] and t1["sha_equal_record"] and t1["sha_equal_queue_pins"])
    zc = {}
    for r in zero_runs:
        c = jl(os.path.join(pull, r, "CONTROL_ZERO.json")) or {}
        zc[r] = {"PASS": c.get("PASS") is True, "dump_files": c.get("dump_files"), "dump_numeric_arrays": c.get("dump_numeric_arrays"),
                 "logs": {k: v.get("PASS") for k, v in (c.get("logs") or {}).items()}}
    out = {"t1_table": t1, "t2_t3_zero_tables": zc, "PASS": bool(t1["PASS"] and zc and all(v["PASS"] for v in zc.values()))}
    return out


def fed_one(run_dir, key, T, head_sha):
    """t4 run line and t5 fed values for one run folder."""
    td = os.path.join(run_dir, "ticks")
    res, ok = {}, True
    n_present, n_present_ok = 0, 0
    for kind, has_q in (("rule_log_%s.npz", False), ("rule_log_mid_%s.npz", True), ("rule_log_pre_%s.npz", False)):
        p = os.path.join(td, kind % key)
        if not os.path.isfile(p):
            if kind.startswith("rule_log_pre"):
                continue                                                        # no pre-roll in a grid141 run
            res[kind % "X"] = {"PASS": False, "why": "missing"}
            ok = False
            continue
        with np.load(p, allow_pickle=True) as z:
            need = ("ci", "j", "c_est", "est_ok", "c_src", "v_l", "v_L", "present") + (("q",) if has_q else ())
            miss = [k for k in need if k not in z.files]
            if miss:
                res[kind % "X"] = {"PASS": False, "why": "keys missing: %s" % miss}
                ok = False
                continue
            ci, j = z["ci"].astype(np.int64), z["j"].astype(np.int64)
            q = z["q"].astype(np.int64) if has_q else np.zeros(len(ci), np.int64)
            kk = z["k"].astype(np.int64) if "k" in z.files else None
            ce, eo = z["c_est"].astype(np.float64), z["est_ok"].astype(bool)
            cs, vl, vL, pr = z["c_src"].astype(np.float64), z["v_l"].astype(np.float64), z["v_L"].astype(np.float64), z["present"].astype(bool)
        # j of every log is the frame of the look itself (measured 2026-11-28 on the ADD152 runs: in the later-look log j = t0 + k with k counting the q executed
        # steps after the tick, so the look's tick frame is j - q). The look's step k - q must sit on a replan tick (a multiple of 10) in the tick and later-look logs.
        e0 = int(((kk - q) % 10 != 0).sum()) if (kk is not None and not kind.startswith("rule_log_pre")) else 0
        tv = table_at(T, ci, j)
        fin = np.isfinite(tv)
        e1 = int((eo != fin).sum())
        e2 = int((np.abs(ce[eo] - tv[eo]) > TOL_EST).sum()) if eo.any() else 0
        e3 = int(np.isfinite(ce[~eo]).sum())
        fed = np.where(eo, ce, 0.0)
        e4 = int((np.abs(cs[pr] - fed[pr]) > TOL_SRC).sum()) if pr.any() else 0
        with np.errstate(invalid="ignore"):
            e5 = int((np.abs(vL[pr] - np.maximum(0.0, vl[pr] + cs[pr])) > TOL_VL).sum()) if pr.any() else 0
        n_present += int(pr.sum())
        n_present_ok += int((pr & eo).sum())
        good = bool(len(ci) > 0 and e0 == 0 and e1 == 0 and e2 == 0 and e3 == 0 and e4 == 0 and e5 == 0 and eo.any())
        res[kind % "X"] = {"rows": int(len(ci)), "look_step_not_tick_plus_q": e0, "est_ok_rows": int(eo.sum()), "est_ok_not_where_table_finite": e1, "c_est_differs": e2, "c_est_finite_without_est_ok": e3,
                           "c_src_not_fed_value_on_present": e4, "v_L_not_max0_vl_plus_c": e5, "nonzero_fed_on_present": int((pr & (fed != 0)).sum()), "PASS": good}
        ok &= good
    lines = []
    for f in glob.glob(os.path.join(run_dir, "**", "run_*.txt"), recursive=True):
        lines += [ln.strip() for ln in io.open(f, encoding="utf-8", errors="replace") if ln.startswith("est looks")]
    m = [re.search(r"file ([0-9a-f]{64})", ln) for ln in lines]
    line_ok = bool(lines) and all(x is not None and x.group(1) == head_sha for x in m)
    res["t4_run_line"] = {"lines": lines[:3], "PASS": line_ok}
    res["present_looks"] = n_present
    res["present_looks_with_est_ok_share"] = round(n_present_ok / n_present, 4) if n_present else None
    res["PASS"] = bool(ok and line_ok)
    return res


def c_fed(pull, seeds, arms, table_path, head_sha):
    T = load_table(table_path)
    per = {}
    for arm in arms:
        for m in seeds:
            key = "%s_s%d" % (arm, m)
            per[key] = fed_one(os.path.join(pull, key), key, T, head_sha)
    return {"per_run": per, "PASS": bool(per and all(v["PASS"] for v in per.values()))}


def u_word(cell):
    lo, hi = cell["interval"]
    return "LOWER" if hi < 0 else ("HIGHER" if lo > 0 else "NOT-SEPARATED")


def combine(v1, p30, ceiling_free_gb=None):
    """Card section 5: names and the committed consequence from the two pair blocks (each: headway U, C, F cells; families dict of the four families)."""
    u, c = u_word(v1["headway"]["U"]), u_word(v1["headway"]["C"])
    pu, pc = u_word(p30["headway"]["U"]), u_word(p30["headway"]["C"])
    worse = sorted(set(f for f in FAMS4 if v1["families"].get(f) == "WORSE") | set(f for f in FAMS4 if p30["families"].get(f) == "WORSE"))
    name, pname = "HEADCLOSE10K-U-%s" % u, "HEADCLOSE10K-P30-U-%s" % pu
    ceiling = False
    if u == "LOWER" and not worse and c != "HIGHER" and pc != "HIGHER" and pu != "HIGHER":
        cons = "ADOPT: the head closing (mean of the two seeds) is the closing input of the candidate's reader form (specification section 1, the rule line, via the other session)"
    elif u == "LOWER":
        cons = "TO-THE-PI: with its cost (families WORSE: %s; C %s, P30 C %s, P30 U %s)" % (", ".join(worse) or "none", c, pc, pu)
    else:
        ceiling = True
        cons = "KEEP-ZERO: the reader form keeps closing zero; the conditional label-closing ceiling H_RQ50KC1 runs (card section 3)"
        if ceiling_free_gb is not None and ceiling_free_gb < 1.75:
            cons += "; BLOCKED on D: space of box 2 (%.2f GB free, card needs 1.75)" % ceiling_free_gb
            ceiling = False
    return {"name": name, "name_p30": pname, "V1": {"U": u, "C": c}, "V1P30": {"U": pu, "C": pc}, "families_WORSE_either": worse, "consequence": cons, "run_ceiling": ceiling}


def ceiling_verdict(c1, v1c1):
    """Card section 5, after the ceiling arm: c1 = H_RQ50KC1 - H_RQ50KE1, v1c1 = H_RQ50KV1 - H_RQ50KC1 (pair blocks)."""
    cu = u_word(c1["headway"]["U"])
    if cu == "LOWER":
        return {"ceiling_U": cu, "lever": "HEAD-ACCURACY: the label closing lowers U in the reader form and the head's closing does not; next, a longer-history or wider head on the same 3,984 clips"}
    return {"ceiling_U": cu, "lever": "PRESENCE-AND-GAP: no closing input lowers U at this presence and gap; the reader's gap and presence are the next lever"}
