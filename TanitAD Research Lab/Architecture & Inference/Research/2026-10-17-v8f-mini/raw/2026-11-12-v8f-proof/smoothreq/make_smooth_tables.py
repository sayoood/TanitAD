"""make_smooth_tables.py -- pod tool of card PREREG_V8F_PROOF_ADD85 (SMOOTHREQ10K): the closing-input tables of the arms S50, S90, S7B, W7B, Z00, STKM, TKD (section 2 of the card).

Pod, CPU only. Imports functions of make_tab_tables.py (card ADD84) and, through it, make_surrogate.py (card ADD83): found in --tool-dir, $SMOOTH_TOOL_DIR, this folder, ../tabclose (and ../shufclose).
Inputs (defaults on the pod):
  --tab-dir /workspace/v8proof/closeread/est_tab     tables tab_L100/TK/STK/S70/W70_s<k>.npz and TAB_TABLES_RECORD.json of card ADD84: REFUSED unless the record says PASS for controls a0 and a1 and the
                                                     sha256 of each of these files equals the record's entry
  --dump-dir /workspace/v8proof/closeread/dump/both  reader_s<k>_score.npz (rows of the scored frames of card ADD78, by the parent's u4 code path)
  --est-dir /workspace/v8proof/closeread/est         est_score_N10_s<k>.npz, est_score_L25_s<k>.npz (arrays clip_index, frame, est)
  --fit-dir /workspace/v8proof/closeread/tools       folder of closeread_fit.py
  --run-dir /workspace/v8proof/integ10k/t1           banked runs I10RQ50eTKKF5P30Hb20_s<k> and I10RQ50eSTKKF5P30Hb20_s<k>: <run-dir>/<run>_s<k>/ticks/rule_log_<run>_s<k>.npz and rule_log_mid_<run>_s<k>.npz
                                                     (columns ci, j, present, est_ok, c_est)
  --seeds 0,1   --out-dir /workspace/v8proof/closeread/est_smooth (REFUSES an existing folder)   --tool-dir DIR
Writes tab_<ARM>_s<k>.npz (arrays clip_index int64, frame int64, est float64, NaN = no value; row order of the L100 table) for ARM in S50, S90, S7B, W7B, Z00, STKM, TKD and SMOOTH_TABLES_RECORD.json.
Random numbers: default_rng(85000 + 10 a + k), a = 0..4 for S50, S90, S7B, W7B, Z00; one standard normal per row in row order.
Controls: b0 (planted quality of S50, S90, S7B, W7B and the second draws), b1 (Z00), b2 (STKM and TKD). Last line: ZZ SMOOTHTABLES PASS   or   ZZ SMOOTHTABLES FAIL <controls>.
--selftest: synthetic world, every control and refusal shown to turn under a planted failure; last line ZZ SELFTEST PASS / FAIL."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRIDE = 1 << 20
SEEDS_DEFAULT = (0, 1)
RNG_BASE = 85000
ARM_INDEX = {"S50": 0, "S90": 1, "S7B": 2, "W7B": 3, "Z00": 4}
R_TARGET = {"S50": 0.50, "S90": 0.90, "S7B": 0.70, "W7B": 0.70}
SMOOTH_ARMS = ("S50", "S90", "S7B")
OUT_ARMS = ("S50", "S90", "S7B", "W7B", "Z00", "STKM", "TKD")
BANKED_USED = ("L100", "TK", "STK", "S70", "W70")
BANKED_RUNS = {"TK": "I10RQ50eTKKF5P30Hb20", "STK": "I10RQ50eSTKKF5P30Hb20"}
SECOND_DRAW_BAND = (-0.10, 0.10)
B0_R_TOL, B0_SD_BAND, B0_MEAN_TOL = 0.02, (0.95, 1.05), 0.05
B0_AC_SMOOTH, B0_AC_WHITE = (0.85, 0.95), (-0.05, 0.05)
B1_MEAN_TOL, B1_SD_BAND, B1_AC, B1_R = 0.01, (0.95, 1.05), (0.85, 0.95), 0.15
B2_CONST_TOL = 1e-12
U4_H = 25


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tt(tool_dir=""):
    for c in (tool_dir, os.environ.get("SMOOTH_TOOL_DIR", ""), HERE, os.path.join(HERE, "..", "tabclose")):
        if c and os.path.isfile(os.path.join(c, "make_tab_tables.py")):
            p = os.path.normpath(os.path.join(c, "make_tab_tables.py"))
            spec = importlib.util.spec_from_file_location("make_tab_tables_sm", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["make_tab_tables_sm"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: make_tab_tables.py not found (give --tool-dir)")


# ====================================================================================================== the tables
def make_arm(TT, arm, l100, clip_index, frame, k):
    """-> est [N] of arm S50, S90, S7B or W7B of seed k: m + r^2 (L - m) + s sqrt(r^2 (1 - r^2)) e on the frames where L100 has a value, e planted as in card ADD84 (mean 0, exactly uncorrelated with L, sd 1); the
    smooth arms draw e by the autoregression 0.9 along each clip, the white arm per frame. default_rng(85000 + 10 a + k), one standard normal per row."""
    rng = np.random.default_rng(RNG_BASE + 10 * ARM_INDEX[arm] + k)
    z = rng.standard_normal(len(l100))
    V = np.isfinite(l100)
    L = l100[V]
    m, s = float(L.mean()), float(L.std())
    if arm in SMOOTH_ARMS:
        z = TT.ar_along_clips(z, clip_index, frame)
    e = TT.plant_noise(z[V], L, uncorrelated=True)
    r = R_TARGET[arm]
    out = np.full(len(l100), np.nan)
    out[V] = m + r * r * (L - m) + s * np.sqrt(r * r * (1.0 - r * r)) * e
    return out


def make_z00(TT, tk_est, clip_index, frame, k):
    """Z00: on EVERY row z * e, e = the autoregression 0.9 over all frames of each clip made exactly mean 0 and unit sd (ddof 0) over all rows; z = sd (ddof 0) of the finite values of the TK table."""
    rng = np.random.default_rng(RNG_BASE + 10 * ARM_INDEX["Z00"] + k)
    e = TT.ar_along_clips(rng.standard_normal(len(tk_est)), clip_index, frame)
    e = e - e.mean()
    e = e / e.std()
    fin = np.asarray(tk_est, np.float64)
    z = float(fin[np.isfinite(fin)].std())
    return z * e, z


def look_mean(logs):
    """mean of c_est over the rows with present AND est_ok true, over all given logs of one run together"""
    v = [np.asarray(g["c_est"], np.float64)[np.asarray(g["present"], bool) & np.asarray(g["est_ok"], bool)] for g in logs]
    v = np.concatenate(v) if v else np.zeros(0)
    if len(v) == 0:
        raise SystemExit("REFUSED: a banked run has no look with a lead read present and an estimate")
    return float(v.mean()), int(len(v))


def look_mean_b(logs):
    """second, independent routine: a row loop with an exact sum (math.fsum)"""
    tot, n = [], 0
    for g in logs:
        pr, eo, ce = g["present"], g["est_ok"], g["c_est"]
        for i in range(len(ce)):
            if bool(pr[i]) and bool(eo[i]):
                tot.append(float(ce[i]))
                n += 1
    return (math.fsum(tot) / n if n else float("nan")), n


def lookup(tab, ci, fr):
    key = np.asarray(tab[0], np.int64) * STRIDE + np.asarray(tab[1], np.int64)
    o = np.argsort(key, kind="stable")
    ks, vs = key[o], np.asarray(tab[2], np.float64)[o]
    t = np.asarray(ci, np.int64) * STRIDE + np.asarray(fr, np.int64)
    pos = np.minimum(np.searchsorted(ks, t), len(ks) - 1)
    return np.where(ks[pos] == t, vs[pos], np.nan)


def err_lag1(est, l100, ci, fr):
    """lag-1 autocorrelation of (table - L) over pairs of consecutive frames of one clip where L100 has a value and the table is finite at both -> (r, pairs)"""
    ok = np.isfinite(est) & np.isfinite(l100)
    err = np.where(ok, est - l100, np.nan)
    i = np.nonzero(ok[:-1] & ok[1:] & (ci[:-1] == ci[1:]) & (fr[1:] == fr[:-1] + 1))[0]
    if len(i) < 3 or err[i].std() == 0 or err[i + 1].std() == 0:
        return float("nan"), int(len(i))
    return float(np.corrcoef(err[i], err[i + 1])[0, 1]), int(len(i))


def noise_of(v, L, m, s, r):
    return (v - m - r * r * (L - m)) / (s * np.sqrt(r * r * (1.0 - r * r)))


# ====================================================================================================== controls
def check_b0(TT, tabs, banked, seeds):
    """tabs {(arm, seed): (ci, fr, est)} of S50, S90, S7B, W7B and L100 as read back; banked {(arm, seed): (ci, fr, est)} of S70 and W70. Per arm and seed as control a1 of card ADD84: r within 0.02 of the target, sd within
    0.95 to 1.05 times r s, |mean - m| <= 0.05, lag-1 autocorrelation of the noise recovered from the table in [0.85, 0.95] (smooth) or [-0.05, +0.05] (W7B), valued frames those of L100 exactly. Also the r between
    the noise of S7B and that of the banked S70, and of W7B and W70 (a second draw, not a copy), inside [-0.10, +0.10]."""
    out, ok = {}, True
    for k in seeds:
        Lc, Lf, Lv = tabs[("L100", k)]
        V = np.isfinite(Lv)
        L = Lv[V]
        m, s = float(L.mean()), float(L.std())
        noises = {}
        for arm in ("S50", "S90", "S7B", "W7B"):
            nm = "%s_s%d" % (arm, k)
            c, f, v = tabs[(arm, k)]
            same = bool(np.array_equal(c, Lc) and np.array_equal(f, Lf) and np.array_equal(np.isfinite(v), V))
            if not same:
                out[nm], ok = {"ok": False, "valued_frames_equal_L100": False}, False
                continue
            r_t = R_TARGET[arm]
            x = v[V]
            r_obs = float(np.corrcoef(x, L)[0, 1])
            sd_obs, mean_obs = float(x.std()), float(x.mean())
            noise = noise_of(v, Lv, m, s, r_t)
            noises[arm] = noise
            ac, n_pairs = TT._lag1(noise, V, c, f)
            lo, hi = B0_AC_WHITE if arm == "W7B" else B0_AC_SMOOTH
            r_ok = bool(abs(r_obs - r_t) <= B0_R_TOL)
            sd_ok = bool(B0_SD_BAND[0] <= sd_obs / (r_t * s) <= B0_SD_BAND[1])
            mean_ok = bool(abs(mean_obs - m) <= B0_MEAN_TOL)
            ac_ok = bool(lo <= ac <= hi)
            good = bool(r_ok and sd_ok and mean_ok and ac_ok)
            out[nm] = {"ok": good, "r": r_obs, "r_target": r_t, "sd": sd_obs, "sd_target": r_t * s, "mean": mean_obs, "m": m, "s": s, "lag1_autocorr": ac, "lag1_pairs": n_pairs, "valued_frames": int(V.sum()),
                       "r_ok": r_ok, "sd_ok": sd_ok, "mean_ok": mean_ok, "autocorr_ok": ac_ok, "valued_frames_equal_L100": True}
            ok = ok and good
        for arm, bk in (("S7B", "S70"), ("W7B", "W70")):
            nm = "%s_vs_%s_s%d" % (arm, bk, k)
            bc, bf, bv = banked[(bk, k)]
            if arm not in noises or not np.array_equal(np.isfinite(bv), V):
                out[nm], ok = {"ok": False, "why": "arm or banked table not comparable"}, False
                continue
            nb = noise_of(bv, Lv, m, s, 0.70)
            rr = float(np.corrcoef(noises[arm][V], nb[V])[0, 1])
            good = bool(SECOND_DRAW_BAND[0] <= rr <= SECOND_DRAW_BAND[1])
            out[nm] = {"ok": good, "r_noise": rr, "band": list(SECOND_DRAW_BAND)}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def check_b1(TT, z00, tk, l100, seeds):
    """z00 {seed: (ci, fr, est)}, tk {seed: (ci, fr, est)}, l100 {seed: (ci, fr, est)}. Value on every row; |mean| <= 0.01; sd within 0.95 to 1.05 times that of the finite TK values; lag-1 autocorrelation inside
    a clip over consecutive frames in [0.85, 0.95]; r against L on the frames where L100 has a value inside [-0.15, +0.15]."""
    out, ok = {}, True
    for k in seeds:
        c, f, v = z00[k]
        Lc, Lf, Lv = l100[k]
        fin = bool(np.isfinite(v).all()) and bool(np.array_equal(c, Lc) and np.array_equal(f, Lf))
        tkv = tk[k][2][np.isfinite(tk[k][2])]
        if not fin or len(tkv) < 3:
            out["s%d" % k], ok = {"ok": False, "finite_on_every_row": fin}, False
            continue
        mean, sd = float(v.mean()), float(v.std())
        ac, n_pairs = TT._lag1(v, np.ones(len(v), bool), c, f)
        V = np.isfinite(Lv)
        rl = float(np.corrcoef(v[V], Lv[V])[0, 1])
        c1, c2 = bool(abs(mean) <= B1_MEAN_TOL), bool(B1_SD_BAND[0] <= sd / float(tkv.std()) <= B1_SD_BAND[1])
        c3, c4 = bool(B1_AC[0] <= ac <= B1_AC[1]), bool(abs(rl) <= B1_R)
        good = bool(c1 and c2 and c3 and c4)
        out["s%d" % k] = {"ok": good, "finite_on_every_row": True, "mean": mean, "sd": sd, "sd_tk_finite": float(tkv.std()), "lag1_autocorr": ac, "lag1_pairs": n_pairs, "r_against_label": rl,
                          "mean_ok": c1, "sd_ok": c2, "autocorr_ok": c3, "r_ok": c4}
        ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def check_b2(SU, CF, D, tabs, consts_rec, logs, seeds):
    """tabs {(arm, seed): (ci, fr, est)} of TK, STK, STKM, TKD read back; consts_rec {seed: (c_k, d_k)} as used; logs {(run key 'TK'|'STK', seed): [log dicts]}; D {seed: dump}. STKM equals STK - d_k and TKD equals
    TK - c_k exactly; c_k and d_k recomputed by a second independent routine agree to 1e-12; r of STKM against the label on the scored frames of card ADD78 for H 25 inside [-0.15, +0.15] pooled over the seeds."""
    out, ok, rows = {}, True, []
    for k in seeds:
        c_k, d_k = consts_rec[k]
        mt, _ = look_mean_b(logs[("TK", k)])
        ms, _ = look_mean_b(logs[("STK", k)])
        c2, d2 = mt, ms - mt
        stk, tk = tabs[("STK", k)][2], tabs[("TK", k)][2]
        stkm, tkd = tabs[("STKM", k)][2], tabs[("TKD", k)][2]
        e1 = bool(np.array_equal(stkm, stk - d_k, equal_nan=True))
        e2 = bool(np.array_equal(tkd, tk - c_k, equal_nan=True))
        e3 = bool(abs(c_k - c2) <= B2_CONST_TOL and abs(d_k - d2) <= B2_CONST_TOL)
        good = bool(e1 and e2 and e3)
        out["s%d" % k] = {"ok": good, "STKM_equals_STK_minus_d": e1, "TKD_equals_TK_minus_c": e2, "constants_agree_second_routine": e3, "c_k": c_k, "d_k": d_k, "c_k_second": c2, "d_k_second": d2}
        ok = ok and good
        c, f, v = tabs[("STKM", k)]
        ident = {"clip_index": tabs[("TK", k)][0], "frame": tabs[("TK", k)][1], "est": tabs[("TK", k)][2]}
        rows.append(SU.u4_rows(CF, D[k], U4_H, {"clip_index": c, "frame": f, "est": v}, ident))
    ok4, d4 = SU.check_u4(rows)
    out["r_STKM_label_H25"] = dict(d4, ok=bool(ok4), H=U4_H)
    return {"PASS": bool(ok and ok4), "runs": out}


# ====================================================================================================== the tool
def admit_banked(TT, tab_dir, seeds):
    """REFUSE unless the record of card ADD84 says PASS for controls a0 and a1 and the sha256 of each used table equals the record's entry. -> (record, {(arm, seed): arrays}, paths)"""
    rp = os.path.join(tab_dir, "TAB_TABLES_RECORD.json")
    if not os.path.isfile(rp):
        raise SystemExit("REFUSED: no TAB_TABLES_RECORD.json in %s" % tab_dir)
    rec = json.load(open(rp, encoding="utf-8"))
    for c in ("a0", "a1"):
        if not (isinstance(rec.get(c), dict) and rec[c].get("PASS") is True):
            raise SystemExit("REFUSED: the record of card ADD84 does not say PASS for control %s" % c)
    tabs, paths = {}, {}
    for arm in BANKED_USED:
        for k in seeds:
            fn = "tab_%s_s%d.npz" % (arm, k)
            p = os.path.join(tab_dir, fn)
            if not os.path.isfile(p):
                raise SystemExit("REFUSED: %s not found" % p)
            if (rec.get("tables") or {}).get(fn) != sha256_file(p):
                raise SystemExit("REFUSED: the sha256 of %s differs from the record" % fn)
            tabs[(arm, k)], paths[(arm, k)] = TT.load_tab(p), p
    return rec, tabs, paths


def find_run_logs(run_dir, run, k):
    key = "%s_s%d" % (run, k)
    out = []
    for fn in ("rule_log_%s.npz" % key, "rule_log_mid_%s.npz" % key):
        p = os.path.join(run_dir, key, "ticks", fn)
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: rule log %s not found" % p)
        z = np.load(p, allow_pickle=True)
        for a in ("ci", "j", "present", "est_ok", "c_est"):
            if a not in z.files:
                raise SystemExit("REFUSED: rule log %s lacks %s" % (p, a))
        out.append((p, {a: np.asarray(z[a]) for a in ("ci", "j", "present", "est_ok", "c_est")}))
    return out


def run(dump_dir, tab_dir, est_dir, fit_dir, run_dir, out_dir, seeds, tool_dir="", tamper=None):
    """tamper: selftest hook, called as tamper(('tab', ARM, k) or ('const', k), arrays dict) before a file is written / constants are used. -> (record, failed names)."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    TT = load_tt(tool_dir)
    SU = TT.load_surrogate_tool("")
    CF = SU.load_fit(fit_dir)
    rec_in, btabs, bpaths = admit_banked(TT, tab_dir, seeds)
    D, dpath = {}, {}
    for k in seeds:
        D[k], dpath[k] = TT.load_dump_file(CF, dump_dir, k)
    logs, lpaths = {}, {}
    for k in seeds:
        for key, run_name in BANKED_RUNS.items():
            ll = find_run_logs(run_dir, run_name, k)
            logs[(key, k)] = [g for _, g in ll]
            lpaths[(key, k)] = [p for p, _ in ll]
    est_tabs, epaths = {}, {}
    for e in ("N10", "L25"):
        for k in seeds:
            p = os.path.join(est_dir, "est_score_%s_s%d.npz" % (e, k))
            if not os.path.isfile(p):
                raise SystemExit("REFUSED: %s not found" % p)
            est_tabs[(e, k)], epaths[(e, k)] = TT.load_tab(p), p
    os.makedirs(out_dir)
    rec = {"card": "PREREG_V8F_PROOF_ADD85", "seeds": list(seeds), "inputs": {}, "tables": {}, "constants": {}, "rng": {}, "error_autocorr": {}}
    for (arm, k), p in bpaths.items():
        rec["inputs"]["banked_%s_s%d" % (arm, k)] = {"path": p, "sha256": sha256_file(p)}
    for k in seeds:
        rec["inputs"]["dump_s%d" % k] = {"path": dpath[k], "sha256": sha256_file(dpath[k])}
    for (key, k), ps in lpaths.items():
        for i, p in enumerate(ps):
            rec["inputs"]["runlog_%s_s%d_%d" % (key, k, i)] = {"path": p, "sha256": sha256_file(p)}
    for (e, k), p in epaths.items():
        rec["inputs"]["est_%s_s%d" % (e, k)] = {"path": p, "sha256": sha256_file(p)}
    names = {}
    consts = {}
    for k in seeds:
        ci, fr, l100 = btabs[("L100", k)]
        tk = btabs[("TK", k)][2]
        stk = btabs[("STK", k)][2]
        c_k, n_c = look_mean(logs[("TK", k)])
        m_stk, n_s = look_mean(logs[("STK", k)])
        d_k = m_stk - c_k
        const = {"c_k": c_k, "d_k": d_k}
        if tamper is not None:
            tamper(("const", k), const)
        c_k, d_k = const["c_k"], const["d_k"]
        arms = {a: make_arm(TT, a, l100, ci, fr, k) for a in ("S50", "S90", "S7B", "W7B")}
        arms["Z00"], z = make_z00(TT, tk, ci, fr, k)
        arms["STKM"] = stk - d_k
        arms["TKD"] = tk - c_k
        consts[k] = (c_k, d_k)
        rec["constants"]["s%d" % k] = {"c_k": c_k, "d_k": d_k, "z": z, "looks_TK": n_c, "looks_STK": n_s}
        for a in ARM_INDEX:
            rec["rng"]["%s_s%d" % (a, k)] = RNG_BASE + 10 * ARM_INDEX[a] + k
        for arm in OUT_ARMS:
            tab = {"clip_index": ci.copy(), "frame": fr.copy(), "est": np.asarray(arms[arm], np.float64)}
            if tamper is not None:
                tamper(("tab", arm, k), tab)
            nm = "tab_%s_s%d.npz" % (arm, k)
            names[(arm, k)] = nm
            np.savez(os.path.join(out_dir, nm), clip_index=np.asarray(tab["clip_index"], np.int64), frame=np.asarray(tab["frame"], np.int64), est=np.asarray(tab["est"], np.float64))
        ea = {}
        for tag, tb in (("TK", btabs[("TK", k)]), ("N10", est_tabs[("N10", k)]), ("L25", est_tabs[("L25", k)]), ("S70", btabs[("S70", k)]), ("W70", btabs[("W70", k)])):
            ev = tb[2] if tag in ("TK", "S70", "W70") else lookup(tb, ci, fr)
            r_, n_ = err_lag1(ev, l100, ci, fr)
            ea[tag] = {"lag1_autocorr_of_error": r_, "pairs": n_}
        rec["error_autocorr"]["s%d" % k] = ea
    back = {(arm, k): TT.load_tab(os.path.join(out_dir, names[(arm, k)])) for arm in OUT_ARMS for k in seeds}
    allt = dict(back)
    for (arm, k), v in btabs.items():
        allt[(arm, k)] = v
    b0 = check_b0(TT, {(a, k): allt[(a, k)] for a in ("S50", "S90", "S7B", "W7B", "L100") for k in seeds}, {(a, k): btabs[(a, k)] for a in ("S70", "W70") for k in seeds}, seeds)
    rec["b0"] = b0
    print("%s b0 %s" % ("PASS" if b0["PASS"] else "FAIL", json.dumps({n: {x: (round(y, 5) if isinstance(y, float) else y) for x, y in d.items() if x in ("ok", "r", "sd", "lag1_autocorr", "r_noise")} for n, d in b0["runs"].items()})), flush=True)
    b1 = check_b1(TT, {k: back[("Z00", k)] for k in seeds}, {k: btabs[("TK", k)] for k in seeds}, {k: btabs[("L100", k)] for k in seeds}, seeds)
    rec["b1"] = b1
    print("%s b1 %s" % ("PASS" if b1["PASS"] else "FAIL", json.dumps({n: {x: (round(y, 5) if isinstance(y, float) else y) for x, y in d.items() if x in ("ok", "mean", "sd", "lag1_autocorr", "r_against_label")} for n, d in b1["runs"].items()})), flush=True)
    b2 = check_b2(SU, CF, D, {(a, k): allt[(a, k)] for a in ("TK", "STK", "STKM", "TKD") for k in seeds}, consts, logs, seeds)
    rec["b2"] = b2
    print("%s b2 %s" % ("PASS" if b2["PASS"] else "FAIL", json.dumps({n: {x: (round(y, 5) if isinstance(y, float) else y) for x, y in d.items() if x in ("ok", "c_k", "d_k", "r_surrogate")} for n, d in b2["runs"].items()})), flush=True)
    failed = [n for n, c in (("b0", b0), ("b1", b1), ("b2", b2)) if not c["PASS"]]
    for k in seeds:
        for arm in OUT_ARMS:
            nm = names[(arm, k)]
            rec["tables"][nm] = sha256_file(os.path.join(out_dir, nm))
            print("TABLE %s rows %d valued %d sha256 %s" % (nm, len(back[(arm, k)][2]), int(np.isfinite(back[(arm, k)][2]).sum()), rec["tables"][nm]), flush=True)
    rec["pass"] = not failed
    rec["failed"] = failed
    with open(os.path.join(out_dir, "SMOOTH_TABLES_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec, failed


# ====================================================================================================== selftest
def make_world(root, TT, n_clips=60, seeds=SEEDS_DEFAULT):
    """synthetic world: dump, tk, ADD84 run logs (via the parent's synth), the ADD84 tables (the parent's tool), est tables N10, L25 and rule logs of the banked TK and STK runs"""
    world = TT.synth(root, n_clips=n_clips)
    rec, failed = TT.run(os.path.join(root, "dump"), os.path.join(root, "tk"), "", os.path.join(root, "runs"), "I10RQ50cKF5P30Hb20", os.path.join(root, "est_tab"), list(seeds))
    assert not failed, failed
    os.makedirs(os.path.join(root, "est"))
    os.makedirs(os.path.join(root, "t1"))
    rg = np.random.default_rng(3)
    for k in seeds:
        ci, fr, l100 = TT.load_tab(os.path.join(root, "est_tab", "tab_L100_s%d.npz" % k))
        for e in ("N10", "L25"):
            est = 0.5 * np.nan_to_num(world["lab_closing"]) + rg.normal(0, 0.8, len(ci))
            np.savez(os.path.join(root, "est", "est_score_%s_s%d.npz" % (e, k)), clip_index=ci, frame=fr, est=est)
        for key, run_name in BANKED_RUNS.items():
            tab = TT.load_tab(os.path.join(root, "est_tab", "tab_%s_s%d.npz" % (key, k)))
            td = os.path.join(root, "t1", "%s_s%d" % (run_name, k), "ticks")
            os.makedirs(td)
            for fn, off in (("rule_log_%s_s%d.npz" % (run_name, k), 0), ("rule_log_mid_%s_s%d.npz" % (run_name, k), 5)):
                sel = rg.choice(len(ci), 4000, replace=False)
                c_est = tab[2][sel]
                np.savez(os.path.join(td, fn), ci=ci[sel], j=fr[sel], present=rg.uniform(size=4000) < 0.6, est_ok=np.isfinite(c_est), c_est=c_est)
    return world


def selftest():
    res = []

    def need(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        res.append(bool(cond))
    TT = load_tt(os.environ.get("SMOOTH_TOOL_DIR", ""))
    root = tempfile.mkdtemp(prefix="tmp_selfsm_", dir=HERE)
    seeds = SEEDS_DEFAULT
    world = make_world(root, TT)
    base = dict(dump_dir=os.path.join(root, "dump"), tab_dir=os.path.join(root, "est_tab"), est_dir=os.path.join(root, "est"), fit_dir="", run_dir=os.path.join(root, "t1"), seeds=list(seeds))
    cnt = [0]

    def go(tamper=None, **over):
        cnt[0] += 1
        a = dict(base, **over)
        a.setdefault("out_dir", os.path.join(root, "out_%d" % cnt[0]))
        return run(tamper=tamper, **a)
    rec, failed = go()
    out = os.path.join(root, "out_1")
    need("clean case: b0, b1, b2 pass and the record says pass", rec["pass"] is True and not failed and all(rec[c]["PASS"] for c in ("b0", "b1", "b2")))
    need("fourteen tables and the record exist; record holds a sha256 per table", all(os.path.isfile(os.path.join(out, "tab_%s_s%d.npz" % (a, k))) for a in OUT_ARMS for k in seeds)
         and len(rec["tables"]) == 14 and all(len(v) == 64 for v in rec["tables"].values()))
    lit = {"S50_s0": 85000, "S90_s0": 85010, "S7B_s0": 85020, "W7B_s0": 85030, "Z00_s0": 85040, "S50_s1": 85001, "S90_s1": 85011, "S7B_s1": 85021, "W7B_s1": 85031, "Z00_s1": 85041}
    need("generator seeds are default_rng(85000 + 10 a + k) with a = 0..4 for S50, S90, S7B, W7B, Z00", rec["rng"] == lit)
    Lc0, Lf0, Lv0 = TT.load_tab(os.path.join(root, "est_tab", "tab_L100_s0.npz"))
    V0 = np.isfinite(Lv0)
    m0, s0 = float(Lv0[V0].mean()), float(Lv0[V0].std())
    nz = noise_of(TT.load_tab(os.path.join(out, "tab_S7B_s0.npz"))[2], Lv0, m0, s0, 0.70)[V0]
    ar_of = lambda seed: TT.ar_along_clips(np.random.default_rng(seed).standard_normal(len(Lv0)), Lc0, Lf0)[V0]       # noqa: E731
    need("S7B is the smooth draw of seed 85020 (noise correlates >0.95 with its autoregression, not with those of 85000 or 85030)",
         np.corrcoef(nz, ar_of(85020))[0, 1] > 0.95 and abs(np.corrcoef(nz, ar_of(85000))[0, 1]) < 0.3 and abs(np.corrcoef(nz, ar_of(85030))[0, 1]) < 0.3)
    txt = open(os.path.join(out, "SMOOTH_TABLES_RECORD.json"), encoding="utf-8").read()
    need("the record line starts with one space and reads \"pass\": true", any(x.startswith(' "pass": true') for x in txt.split("\n")))
    ci, fr, l0 = TT.load_tab(os.path.join(root, "est_tab", "tab_L100_s0.npz"))
    V = np.isfinite(l0)
    s7 = TT.load_tab(os.path.join(out, "tab_S7B_s0.npz"))[2]
    need("S7B: values exactly where L100 has one; same rows", np.array_equal(np.isfinite(s7), V) and np.array_equal(TT.load_tab(os.path.join(out, "tab_S7B_s0.npz"))[0], ci))
    z0 = TT.load_tab(os.path.join(out, "tab_Z00_s0.npz"))[2]
    tk0 = TT.load_tab(os.path.join(root, "est_tab", "tab_TK_s0.npz"))[2]
    need("Z00: finite on every row, mean 0, sd = z (sd of the finite TK values)", np.isfinite(z0).all() and abs(z0.mean()) < 1e-9 and abs(z0.std() - tk0[np.isfinite(tk0)].std()) < 1e-9)
    # constants by hand
    lg = []
    for fn in ("rule_log_I10RQ50eTKKF5P30Hb20_s0.npz", "rule_log_mid_I10RQ50eTKKF5P30Hb20_s0.npz"):
        zz = np.load(os.path.join(root, "t1", "I10RQ50eTKKF5P30Hb20_s0", "ticks", fn))
        lg.append(zz["c_est"][zz["present"] & zz["est_ok"]])
    c0 = float(np.concatenate(lg).mean())
    need("c_k is the mean of c_est over present and est_ok rows of BOTH logs of the TK run (by hand)", abs(rec["constants"]["s0"]["c_k"] - c0) < 1e-12)
    ls = []
    for fn in ("rule_log_I10RQ50eSTKKF5P30Hb20_s0.npz", "rule_log_mid_I10RQ50eSTKKF5P30Hb20_s0.npz"):
        zz = np.load(os.path.join(root, "t1", "I10RQ50eSTKKF5P30Hb20_s0", "ticks", fn))
        ls.append(zz["c_est"][zz["present"] & zz["est_ok"]])
    need("d_k is the STK look mean minus c_k (by hand)", abs(rec["constants"]["s0"]["d_k"] - (float(np.concatenate(ls).mean()) - c0)) < 1e-12)
    need("error_autocorr is reported for TK, N10, L25, S70, W70 per seed with pair counts", all(set(rec["error_autocorr"]["s%d" % k]) == {"TK", "N10", "L25", "S70", "W70"} and rec["error_autocorr"]["s%d" % k]["S70"]["pairs"] > 100 for k in seeds))
    need("the smooth banked error has a larger lag-1 autocorrelation than the white one", rec["error_autocorr"]["s0"]["S70"]["lag1_autocorr_of_error"] > rec["error_autocorr"]["s0"]["W70"]["lag1_autocorr_of_error"] + 0.3)
    # refusals
    def refused(label, fn, frag="REFUSED"):
        try:
            fn()
            need("refusal: " + label, False)
        except SystemExit as e:
            need("refusal: " + label, frag in str(e))
    refused("an existing out folder", lambda: go(out_dir=out))
    import shutil
    d2 = os.path.join(root, "est_tab_badrec")
    shutil.copytree(os.path.join(root, "est_tab"), d2)
    r2 = json.load(open(os.path.join(d2, "TAB_TABLES_RECORD.json"), encoding="utf-8"))
    r2["a1"]["PASS"] = False
    json.dump(r2, open(os.path.join(d2, "TAB_TABLES_RECORD.json"), "w", encoding="utf-8"))
    refused("the record of card ADD84 says a1 failed", lambda: go(tab_dir=d2))
    d3 = os.path.join(root, "est_tab_badhash")
    shutil.copytree(os.path.join(root, "est_tab"), d3)
    z = TT.load_tab(os.path.join(d3, "tab_TK_s1.npz"))
    np.savez(os.path.join(d3, "tab_TK_s1.npz"), clip_index=z[0], frame=z[1], est=z[2] + 1e-9)
    refused("a banked table whose sha256 differs from the record", lambda: go(tab_dir=d3))
    d4 = os.path.join(root, "est_tab_missing")
    shutil.copytree(os.path.join(root, "est_tab"), d4)
    os.remove(os.path.join(d4, "tab_S70_s0.npz"))
    refused("a banked table missing", lambda: go(tab_dir=d4))
    d5 = os.path.join(root, "t1_missing")
    shutil.copytree(os.path.join(root, "t1"), d5)
    os.remove(os.path.join(d5, "I10RQ50eSTKKF5P30Hb20_s1", "ticks", "rule_log_mid_I10RQ50eSTKKF5P30Hb20_s1.npz"))
    refused("a rule log of a banked run missing", lambda: go(run_dir=d5))
    d6 = os.path.join(root, "est_missing")
    os.makedirs(d6)
    refused("an est table missing", lambda: go(est_dir=d6))
    # planted failures, one control each
    def planted(label, tamper, control):
        r, f = go(tamper)
        ok = (control in f) and r["pass"] is False and all(os.path.isfile(os.path.join(os.path.join(root, "out_%d" % cnt[0]), "tab_%s_s0.npz" % a)) for a in OUT_ARMS)
        need("planted %s fails %s (tables still written, record pass false)" % (label, control), ok)
        return r

    def tab_t(arm, k, fn):
        def t(kind, d):
            if kind[0] == "tab" and kind[1] == arm and kind[2] == k:
                d["est"] = fn(d["est"])
        return t
    planted("S50 with 0.15 of L added", tab_t("S50", 0, lambda e: np.where(np.isfinite(e), e + 0.15 * (l0 - np.nanmean(l0)), e)), "b0")
    planted("S90 sd x 1.2", tab_t("S90", 1, lambda e: np.where(np.isfinite(e), np.nanmean(e) + 1.2 * (e - np.nanmean(e)), e)), "b0")
    planted("S7B + 0.2", tab_t("S7B", 0, lambda e: e + 0.2), "b0")
    planted("W7B replaced by a smooth table", tab_t("W7B", 1, lambda e: make_arm(TT, "S7B", l0, ci, fr, 1)), "b0")
    planted("S7B replaced by a white table", tab_t("S7B", 0, lambda e: make_arm(TT, "W7B", l0, ci, fr, 0)), "b0")
    s70 = TT.load_tab(os.path.join(root, "est_tab", "tab_S70_s0.npz"))[2]
    planted("S7B a copy of the banked S70 (not a second draw)", tab_t("S7B", 0, lambda e: s70.copy()), "b0")
    w70 = TT.load_tab(os.path.join(root, "est_tab", "tab_W70_s1.npz"))[2]
    planted("W7B a copy of the banked W70", tab_t("W7B", 1, lambda e: w70.copy()), "b0")
    planted("a valued frame of S90 dropped", tab_t("S90", 0, lambda e: np.where(np.arange(len(e)) == int(np.nonzero(np.isfinite(e))[0][4]), np.nan, e)), "b0")
    planted("a value of W7B on a frame where L100 has none", tab_t("W7B", 0, lambda e: np.where(np.arange(len(e)) == int(np.nonzero(~np.isfinite(e))[0][4]), 0.5, e)), "b0")
    planted("Z00 with a NaN row", tab_t("Z00", 0, lambda e: np.where(np.arange(len(e)) == 7, np.nan, e)), "b1")
    planted("Z00 + 0.5 (mean)", tab_t("Z00", 1, lambda e: e + 0.5), "b1")
    planted("Z00 sd x 1.5", tab_t("Z00", 0, lambda e: e * 1.5), "b1")
    planted("Z00 with white noise (no autocorrelation)", tab_t("Z00", 1, lambda e: np.random.default_rng(1).standard_normal(len(e)) * e.std()), "b1")
    planted("Z00 related to the label (+ 1.0 L)", tab_t("Z00", 0, lambda e: e + 1.0 * np.nan_to_num(l0 - np.nanmean(l0))), "b1")
    planted("STKM off by 1e-6", tab_t("STKM", 1, lambda e: e + 1e-6), "b2")
    planted("TKD off by 1e-6", tab_t("TKD", 0, lambda e: e + 1e-6), "b2")
    stk0 = TT.load_tab(os.path.join(root, "est_tab", "tab_STK_s0.npz"))[2]

    def t_const(kind, d):
        if kind[0] == "const" and kind[1] == 1:
            d["d_k"] = d["d_k"] + 1e-9
    planted("a constant d_k off by 1e-9 against the second routine", t_const, "b2")
    planted("STKM equal to the token-head table (label related, r above the band)", tab_t("STKM", 0, lambda e: tk0.copy() - 0.0), "b2")
    planted("TKD with a NaN pattern changed", tab_t("TKD", 1, lambda e: np.where(np.arange(len(e)) == int(np.nonzero(np.isfinite(e))[0][2]), np.nan, e)), "b2")
    ok = all(res)
    print("ZZ SELFTEST " + ("PASS" if ok else "FAIL") + " checks %d failed %d tmp %s" % (len(res), len(res) - sum(res), os.path.basename(root)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--tab-dir", default="/workspace/v8proof/closeread/est_tab")
    ap.add_argument("--dump-dir", default="/workspace/v8proof/closeread/dump/both")
    ap.add_argument("--est-dir", default="/workspace/v8proof/closeread/est")
    ap.add_argument("--fit-dir", default="/workspace/v8proof/closeread/tools")
    ap.add_argument("--run-dir", default="/workspace/v8proof/integ10k/t1")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out-dir", default="/workspace/v8proof/closeread/est_smooth")
    ap.add_argument("--tool-dir", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    rec, failed = run(a.dump_dir, a.tab_dir, a.est_dir, a.fit_dir, a.run_dir, a.out_dir, seeds, a.tool_dir)
    print("ZZ SMOOTHTABLES " + ("PASS" if not failed else "FAIL " + "; ".join(failed)))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
