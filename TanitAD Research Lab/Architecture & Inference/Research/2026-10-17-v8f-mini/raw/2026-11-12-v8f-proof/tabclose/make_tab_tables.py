"""make_tab_tables.py -- pod tool of card PREREG_V8F_PROOF_ADD84 (TABCLOSE10K): the closing-input tables of the arms TK, STK, L100, W00, W50, W70, W90, S70 (sections 2 to 4 of the card).

Pod, CPU only. Inputs (defaults on the pod):
  --dump-dir /workspace/v8proof/closeread/dump/both   reader_s<k>_score.npz (a subfolder both/ is tried when the file is not in the folder itself)
  --tk-dir /workspace/v8proof/tokclose/fits           tokclose_T25_s<k>_table.npz (arrays clip_index, frame, est; other arrays are ignored)
  --fit-dir /workspace/v8proof/closeread              folder of closeread_fit.py (rows of the scored frames of card ADD78)
  --rrc-run-dir DIR                                   folder of the banked runs of form RRc: DIR/<run>_s<k>/ticks/rule_log_<run>_s<k>.npz and rule_log_mid_<run>_s<k>.npz
  --rrc-run-name I10RQ50cKF5P30Hb20   --seeds 0,1
  --surrogate-tool-dir                                folder of make_surrogate.py (default: beside this file, then ../shufclose)
  --out-dir /workspace/v8proof/closeread/est_tab      REFUSES an existing folder
Writes in the out folder (contract with the reader): tab_<ARM>_s<k>.npz for ARM in TK, STK, L100, W00, W50, W70, W90, S70 (arrays clip_index int64, frame int64, est float64, NaN = no value; rows in the row
order of the dump of seed k), map_STK_s<k>.npz (the map of the surrogate, read by control u1) and TAB_TABLES_RECORD.json ("pass", "tables" {file: sha256}, controls a0, a1, a4_u1, a4_u4 each with "PASS").
Valued frames of L100 (what form RRc feeds as a closing input, read from the wrapper): the frames where the label reads a lead and the trailing run is at least RUN_MIN = 3 frames (the label closing is
then a least-squares slope; with a shorter run or no lead the label source returns 0.0, which the table writes as no value). The table depends on the label alone, not on the reader's presence read.
Planted noise (build detail): the draw of default_rng(84000 + 10 a + k) is made exactly mean 0 and (W50, W70, W90, S70) exactly uncorrelated with L on the valued frames, then exactly unit standard deviation.
Controls: a0 (L100 against the rule logs of the banked RRc runs), a1 (planted quality per arm and seed), a4_u1 and a4_u4 (the surrogate of TK, as u1 and u4 of card ADD83 with H = 25).
Last line: ZZ TABTABLES PASS   or   ZZ TABTABLES FAIL <controls>.   --selftest: synthetic world, every control shown to turn FAIL under a planted failure; last line ZZ SELFTEST PASS / FAIL."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRIDE = 1 << 20                                        # key of a table = clip index * STRIDE + frame (the wrapper's TABLE_STRIDE)
RUN_MIN = 3                                             # v8f_t1_eval_rule.py:39
SEEDS_DEFAULT = (0, 1)
ARM_INDEX = {"W00": 0, "W50": 1, "W70": 2, "W90": 3, "S70": 4}   # a of default_rng(84000 + 10 a + k)
R_TARGET = {"W50": 0.50, "W70": 0.70, "W90": 0.90, "S70": 0.70}
AR_COEF = 0.9
W00_SD = 0.5
A0_TOL = 1e-6
A0_SHARE = 0.999
A1_R_TOL = 0.02
A1_R_TOL_W00 = 0.03
A1_SD_BAND = (0.95, 1.05)
A1_MEAN_TOL = 0.05
A1_AC_S70 = (0.85, 0.95)
A1_AC_W70 = (-0.05, 0.05)
TABLE_ARMS = ("TK", "STK", "L100", "W00", "W50", "W70", "W90", "S70")
U4_H = 25


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_surrogate_tool(tool_dir=""):
    for c in (tool_dir, os.environ.get("SURROGATE_TOOL_DIR", ""), HERE, os.path.join(HERE, "..", "shufclose")):
        if c and os.path.isfile(os.path.join(c, "make_surrogate.py")):
            p = os.path.normpath(os.path.join(c, "make_surrogate.py"))
            spec = importlib.util.spec_from_file_location("make_surrogate_tt", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["make_surrogate_tt"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: make_surrogate.py not found (give --surrogate-tool-dir)")


# ====================================================================================================== the tables
def valued_mask(D):
    """frames on which the label has a closing speed that form RRc can feed: a lead read present and a trailing run of at least RUN_MIN frames (the dump's lab_run, capped at RUN_MAX)."""
    return (np.asarray(D["lab_present"], bool) & (np.asarray(D["lab_run"], np.int64) >= RUN_MIN) & np.isfinite(np.asarray(D["lab_closing"], np.float64)))


def l100_est(D):
    est = np.full(len(D["frame"]), np.nan)
    v = valued_mask(D)
    est[v] = np.asarray(D["lab_closing"], np.float64)[v]
    return est


def ar_along_clips(z, clip_index, frame, coef=AR_COEF):
    """e_t = coef e_(t-1) + sqrt(1 - coef^2) z_t along the frames of a clip in frame order; e at the first frame of a clip = z; restarted at every clip. Rows are taken in (clip, frame) order and returned in the
    row order of the table."""
    z = np.asarray(z, np.float64)
    order = np.lexsort((np.asarray(frame, np.int64), np.asarray(clip_index, np.int64)))
    ci = np.asarray(clip_index, np.int64)[order]
    zo = z[order]
    e = np.empty_like(zo)
    w = np.sqrt(1.0 - coef * coef)
    for i in range(len(zo)):
        e[i] = zo[i] if (i == 0 or ci[i] != ci[i - 1]) else coef * e[i - 1] + w * zo[i]
    out = np.empty_like(e)
    out[order] = e
    return out


def plant_noise(e, L, uncorrelated):
    """e [n] on the valued frames, L [n] the label values there. -> noise with mean exactly 0, (uncorrelated: exactly uncorrelated with L, by subtracting the least-squares projection on [1, L]) and standard
    deviation exactly 1 (ddof 0). Refuses a degenerate case."""
    e = np.asarray(e, np.float64)
    L = np.asarray(L, np.float64)
    if len(e) < 3 or L.std() == 0.0:
        raise SystemExit("REFUSED: fewer than three valued frames or a constant label closing speed")
    if uncorrelated:
        Xm = np.stack([np.ones(len(L)), L], 1)
        beta = np.linalg.lstsq(Xm, e, rcond=None)[0]
        e = e - Xm @ beta
    else:
        e = e - e.mean()
    sd = e.std()
    if not sd > 0.0:
        raise SystemExit("REFUSED: the planted noise has no spread")
    return e / sd


def make_arm(arm, l100, clip_index, frame, k):
    """-> est [N] of arm W00, W50, W70, W90 or S70 of seed k, drawn from default_rng(84000 + 10 a + k): one standard normal per row of the table, used on the valued frames."""
    a = ARM_INDEX[arm]
    rng = np.random.default_rng(84000 + 10 * a + k)
    n = len(l100)
    z = rng.standard_normal(n)
    V = np.isfinite(l100)
    L = l100[V]
    m, s = float(L.mean()), float(L.std())
    if arm == "S70":
        z = ar_along_clips(z, clip_index, frame)
    e = plant_noise(z[V], L, uncorrelated=(arm != "W00"))
    out = np.full(n, np.nan)
    if arm == "W00":
        out[V] = m + W00_SD * s * e
    else:
        r = R_TARGET[arm]
        out[V] = m + r * r * (L - m) + s * np.sqrt(r * r * (1.0 - r * r)) * e
    return out


# ====================================================================================================== loading
def load_dump_file(CF, dump_dir, k):
    p = os.path.join(dump_dir, "reader_s%d_score.npz" % k)
    if not os.path.isfile(p):
        p = os.path.join(dump_dir, "both", "reader_s%d_score.npz" % k)
    if not os.path.isfile(p):
        raise SystemExit("REFUSED: no reader_s%d_score.npz in %s" % (k, dump_dir))
    return CF.load_dump(p), p


def load_tab(path):
    with np.load(path, allow_pickle=False) as z:
        for a in ("clip_index", "frame", "est"):
            if a not in z.files:
                raise SystemExit("REFUSED: %s lacks %s" % (path, a))
        return np.asarray(z["clip_index"], np.int64), np.asarray(z["frame"], np.int64), np.asarray(z["est"], np.float64)


def load_npz_all(path):
    with np.load(path, allow_pickle=False) as z:
        return {a: np.asarray(z[a]) for a in z.files}


def find_rrc_logs(run_dir, run_name, k):
    """-> (replan log path, mid-look log path) or None entries. Folder layout of the ADD83 reader: <run_dir>/<run>_s<k>/ticks/rule_log_<run>_s<k>.npz and rule_log_mid_<run>_s<k>.npz."""
    key = "%s_s%d" % (run_name, k)
    out = []
    for fn in ("rule_log_%s.npz" % key, "rule_log_mid_%s.npz" % key):
        got = None
        for d in (os.path.join(run_dir, key, "ticks"), os.path.join(run_dir, key), os.path.join(run_dir, "ticks")):
            if os.path.isfile(os.path.join(d, fn)):
                got = os.path.join(d, fn)
                break
        out.append(got)
    return tuple(out)


def load_log(path):
    if path is None:
        return None
    z = np.load(path, allow_pickle=True)
    need = ("ci", "j", "c_src", "present")
    for a in need:
        if a not in z.files:
            raise SystemExit("REFUSED: rule log %s lacks %s" % (path, a))
    return {a: np.asarray(z[a]) for a in need}


# ====================================================================================================== controls
def check_a0(tables, logs, tol=A0_TOL, share_min=A0_SHARE):
    """tables {seed: (clip_index, frame, est)} of L100 as read back from the files; logs {(seed, 'rule'|'mid'): dict(ci, j, c_src, present) or None}.
    A look agrees when: the log closing input is finite and not 0 and the table has a finite value within tol at (ci, j); or the log closing input is exactly 0 and the table has no value or exactly 0 there.
    Judged: every (seed, log) pair has looks and the share of agreeing looks over ALL looks is at least share_min; also required: some looks with a nonzero log value (else the control reads nothing).
    Reported beside: the same share over the looks with the log column present true, the number of looks where the log is nonzero and the table differs, and where the log is 0 and the table has a nonzero value."""
    out, ok = {}, True
    for (k, tag), lg in sorted(logs.items()):
        nm = "s%d_%s_log" % (k, tag)
        if lg is None or k not in tables:
            out[nm], ok = {"ok": False, "why": "log or table missing"}, False
            continue
        ci, fr, ev = tables[k]
        look = {int(a) * STRIDE + int(b): float(c) for a, b, c in zip(ci, fr, ev)}
        lci, lj, lc = np.asarray(lg["ci"], np.int64), np.asarray(lg["j"], np.int64), np.asarray(lg["c_src"], np.float64)
        pr = np.asarray(lg["present"], bool)
        n = len(lc)
        agree = np.zeros(n, bool)
        nz_differs = zero_nonzero = n_nz = 0
        for i in range(n):
            tv = look.get(int(lci[i]) * STRIDE + int(lj[i]), float("nan"))
            c = float(lc[i])
            if np.isfinite(c) and c != 0.0:
                n_nz += 1
                agree[i] = bool(np.isfinite(tv) and abs(tv - c) <= tol)
                if not agree[i]:
                    nz_differs += 1
            elif c == 0.0:
                agree[i] = bool((not np.isfinite(tv)) or tv == 0.0)
                if not agree[i]:
                    zero_nonzero += 1
        share_all = float(agree.mean()) if n else float("nan")
        share_pr = float(agree[pr].mean()) if pr.any() else float("nan")
        good = bool(n > 0 and n_nz > 0 and share_all >= share_min)
        out[nm] = {"ok": good, "looks": int(n), "looks_present": int(pr.sum()), "looks_log_nonzero": int(n_nz), "share_agree_all_looks": share_all, "share_agree_present_looks": share_pr,
                   "log_nonzero_table_differs": int(nz_differs), "log_zero_table_nonzero": int(zero_nonzero), "share_min": share_min, "tol": tol}
        ok = ok and good
    if not logs:
        ok = False
    return {"PASS": bool(ok), "runs": out}


def _lag1(noise, V, ci, fr):
    i = np.nonzero(V[:-1] & V[1:] & (ci[:-1] == ci[1:]) & (fr[1:] == fr[:-1] + 1))[0]
    if len(i) < 3:
        return float("nan"), int(len(i))
    a, b = noise[i], noise[i + 1]
    if a.std() == 0 or b.std() == 0:
        return float("nan"), int(len(i))
    return float(np.corrcoef(a, b)[0, 1]), int(len(i))


def check_a1(tabs, seeds):
    """tabs {(arm, seed): (clip_index, frame, est)} read back from the files, arms L100, W00, W50, W70, W90, S70. Per arm and seed on the frames with a value: r against L within 0.02 of the target (W00: inside
    [-0.03, +0.03]); sd within 0.95 to 1.05 times r s (W00: 0.5 s); |mean - m| <= 0.05; lag-1 autocorrelation of the planted noise (recovered from the table) in [0.85, 0.95] for S70 and [-0.05, +0.05] for W70
    (reported for the others); the frames with a value equal those of L100."""
    out, ok = {}, True
    for k in seeds:
        Lc, Lf, Lv = tabs[("L100", k)]
        V = np.isfinite(Lv)
        L = Lv[V]
        m, s = float(L.mean()), float(L.std())
        for arm in ("W00", "W50", "W70", "W90", "S70"):
            nm = "%s_s%d" % (arm, k)
            c, f, v = tabs[(arm, k)]
            rows_same = bool(np.array_equal(c, Lc) and np.array_equal(f, Lf))
            set_same = bool(rows_same and np.array_equal(np.isfinite(v), V))
            if not set_same:
                out[nm], ok = {"ok": False, "rows_equal_L100": rows_same, "valued_frames_equal_L100": False}, False
                continue
            x = v[V]
            r_obs = float(np.corrcoef(x, L)[0, 1])
            sd_obs, mean_obs = float(x.std()), float(x.mean())
            if arm == "W00":
                r_t, sd_t = 0.0, W00_SD * s
                noise = (v - m) / (W00_SD * s)
                r_ok = abs(r_obs) <= A1_R_TOL_W00
            else:
                r_t = R_TARGET[arm]
                sd_t = r_t * s
                noise = (v - m - r_t * r_t * (Lv - m)) / (s * np.sqrt(r_t * r_t * (1.0 - r_t * r_t)))
                r_ok = abs(r_obs - r_t) <= A1_R_TOL
            ac, n_pairs = _lag1(noise, V, c, f)
            sd_ok = bool(A1_SD_BAND[0] <= sd_obs / sd_t <= A1_SD_BAND[1])
            mean_ok = bool(abs(mean_obs - m) <= A1_MEAN_TOL)
            ac_ok = True
            if arm == "S70":
                ac_ok = bool(A1_AC_S70[0] <= ac <= A1_AC_S70[1])
            if arm == "W70":
                ac_ok = bool(A1_AC_W70[0] <= ac <= A1_AC_W70[1])
            good = bool(r_ok and sd_ok and mean_ok and ac_ok)
            out[nm] = {"ok": good, "r": r_obs, "r_target": r_t, "sd": sd_obs, "sd_target": sd_t, "mean": mean_obs, "m": m, "s": s, "lag1_autocorr": ac, "lag1_pairs": n_pairs, "valued_frames": int(V.sum()),
                       "r_ok": bool(r_ok), "sd_ok": sd_ok, "mean_ok": mean_ok, "autocorr_ok": ac_ok, "valued_frames_equal_L100": True}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


# ====================================================================================================== the tool
def run(dump_dir, tk_dir, fit_dir, rrc_run_dir, rrc_run_name, out_dir, seeds, surrogate_tool_dir="", tamper=None):
    """tamper: selftest hook, called as tamper(('tab', ARM, k) or ('map', 'STK', k), arrays dict) before a file is written. -> (record, failed names)."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    SU = load_surrogate_tool(surrogate_tool_dir)
    CF = SU.load_fit(fit_dir)
    if int(CF.RUN_MIN) != RUN_MIN:
        raise SystemExit("REFUSED: RUN_MIN of closeread_fit (%s) is not %d" % (CF.RUN_MIN, RUN_MIN))
    D, dpath, tpath = {}, {}, {}
    for k in seeds:
        D[k], dpath[k] = load_dump_file(CF, dump_dir, k)
    for k in seeds[1:]:                                    # the label columns are one object: equal between the seeds' dumps
        for col in ("clip_index", "frame", "lab_present", "lab_gap", "lab_closing", "lab_run"):
            if not np.array_equal(D[seeds[0]][col], D[k][col], equal_nan=(D[k][col].dtype.kind == "f")):
                raise SystemExit("REFUSED: the label column %s of the dump of seed %d differs from seed %d" % (col, k, seeds[0]))
    TK = {}
    for k in seeds:
        tpath[k] = os.path.join(tk_dir, "tokclose_T25_s%d_table.npz" % k)
        if not os.path.isfile(tpath[k]):
            raise SystemExit("REFUSED: %s not found" % tpath[k])
        with np.load(tpath[k], allow_pickle=False) as z:
            for a in ("clip_index", "frame", "est"):
                if a not in z.files:
                    raise SystemExit("REFUSED: %s lacks %s" % (tpath[k], a))
            TK[k] = {a: np.asarray(z[a]) for a in ("clip_index", "frame", "est")}
        if not (np.array_equal(TK[k]["clip_index"].astype(np.int64), np.asarray(D[k]["clip_index"], np.int64)) and np.array_equal(TK[k]["frame"].astype(np.int64), np.asarray(D[k]["frame"], np.int64))):
            raise SystemExit("REFUSED: the rows of %s are not the rows of the dump of seed %d" % (tpath[k], k))
    os.makedirs(out_dir)
    rec = {"card": "PREREG_V8F_PROOF_ADD84", "seeds": list(seeds), "inputs": {}, "tables": {}, "rng": {}, "m_s": {}, "counts": {}, "a0": {}, "a1": {}, "a4_u1": {}, "a4_u4": {}}
    failed = []
    names = {}
    for k in seeds:
        ci = np.asarray(D[k]["clip_index"], np.int64)
        fr = np.asarray(D[k]["frame"], np.int64)
        rec["inputs"]["dump_s%d" % k] = {"path": dpath[k], "sha256": sha256_file(dpath[k])}
        rec["inputs"]["tk_s%d" % k] = {"path": tpath[k], "sha256": sha256_file(tpath[k])}
        l100 = l100_est(D[k])
        V = np.isfinite(l100)
        rec["m_s"]["s%d" % k] = {"m": float(l100[V].mean()), "s": float(l100[V].std()), "rows": int(len(l100)), "valued": int(V.sum())}
        arms = {"L100": l100}
        for arm in ("W00", "W50", "W70", "W90", "S70"):
            arms[arm] = make_arm(arm, l100, ci, fr, k)
            rec["rng"]["%s_s%d" % (arm, k)] = 84000 + 10 * ARM_INDEX[arm] + k
        ids, share, counts, totals = SU.share_of(D[k])
        pi = SU.compute_map(ids, share)
        tk_est = np.asarray(TK[k]["est"], np.float64)
        stk = SU.surrogate_values(TK[k]["clip_index"], TK[k]["frame"], tk_est, ids, pi)
        arms["TK"] = tk_est.copy()
        arms["STK"] = stk
        rec["maps"] = rec.get("maps", {})
        rec["maps"]["s%d" % k] = {"clips": ids.tolist(), "map": pi.tolist(), "share": share.tolist()}
        for arm in TABLE_ARMS:
            tab = {"clip_index": ci.copy(), "frame": fr.copy(), "est": np.asarray(arms[arm], np.float64)}
            if tamper is not None:
                tamper(("tab", arm, k), tab)
            nm = "tab_%s_s%d.npz" % (arm, k)
            names[(arm, k)] = nm
            np.savez(os.path.join(out_dir, nm), clip_index=np.asarray(tab["clip_index"], np.int64), frame=np.asarray(tab["frame"], np.int64), est=np.asarray(tab["est"], np.float64))
        mp = {"map": pi.copy(), "map_clips": ids.copy()}
        if tamper is not None:
            tamper(("map", "STK", k), mp)
        np.savez(os.path.join(out_dir, "map_STK_s%d.npz" % k), map=np.asarray(mp["map"], np.int64), map_clips=np.asarray(mp["map_clips"], np.int64))
    # ---- the controls read the files back from disk
    back = {(arm, k): load_tab(os.path.join(out_dir, names[(arm, k)])) for arm in TABLE_ARMS for k in seeds}
    logs, log_paths = {}, {}
    for k in seeds:
        pr, pm = find_rrc_logs(rrc_run_dir, rrc_run_name, k)
        log_paths[k] = (pr, pm)
        logs[(k, "rule")] = load_log(pr)
        logs[(k, "mid")] = load_log(pm)
        for tag, p in (("rule", pr), ("mid", pm)):
            if p:
                rec["inputs"]["rrc_%s_log_s%d" % (tag, k)] = {"path": p, "sha256": sha256_file(p)}
    a0 = check_a0({k: back[("L100", k)] for k in seeds}, logs)
    rec["a0"] = a0
    print("%s a0 %s" % ("PASS" if a0["PASS"] else "FAIL", json.dumps({n: {x: y for x, y in d.items() if x != "tol"} for n, d in a0["runs"].items()})), flush=True)
    a1 = check_a1(back, seeds)
    rec["a1"] = a1
    for n, d in sorted(a1["runs"].items()):
        print("%s a1 %s %s" % ("PASS" if d["ok"] else "FAIL", n, json.dumps({x: (round(y, 5) if isinstance(y, float) else y) for x, y in d.items() if x != "ok"})), flush=True)
    u1, u4rows = {}, []
    u1_ok = True
    for k in seeds:
        mapf = load_npz_all(os.path.join(out_dir, "map_STK_s%d.npz" % k))
        c, f, v = back[("STK", k)]
        surr = {"clip_index": c, "frame": f, "est": v, "map": mapf["map"], "map_clips": mapf["map_clips"]}
        ids, share, counts, totals = SU.share_of(D[k])
        ok1, d1 = SU.check_u1(surr, back[("TK", k)], ids, counts, totals)
        u1["s%d" % k] = dict(d1, passed=bool(ok1))
        u1_ok = u1_ok and ok1
        print("%s a4_u1 s%d %s" % ("PASS" if ok1 else "FAIL", k, json.dumps(d1)), flush=True)
        ident = {"clip_index": back[("TK", k)][0], "frame": back[("TK", k)][1], "est": back[("TK", k)][2]}
        u4rows.append(SU.u4_rows(CF, D[k], U4_H, {"clip_index": c, "frame": f, "est": v}, ident))
    rec["a4_u1"] = {"PASS": bool(u1_ok), "seeds": u1}
    ok4, d4 = SU.check_u4(u4rows)
    rec["a4_u4"] = dict(d4, PASS=bool(ok4), H=U4_H)
    print("%s a4_u4 H %d r_surrogate %.4f r_estimate %.4f rows %d/%d (band +-%.2f)" % ("PASS" if ok4 else "FAIL", U4_H, d4["r_surrogate"], d4["r_estimate"], d4["rows_with_surrogate"], d4["rows_scored"], SU.R_BAND), flush=True)
    for nm, good in (("a0", a0["PASS"]), ("a1", a1["PASS"]), ("a4_u1", u1_ok), ("a4_u4", ok4)):
        if not good:
            failed.append(nm)
    for k in seeds:
        for arm in TABLE_ARMS:
            nm = names[(arm, k)]
            c, f, v = back[(arm, k)]
            rec["tables"][nm] = sha256_file(os.path.join(out_dir, nm))
            rec["counts"][nm] = {"rows": int(len(v)), "valued": int(np.isfinite(v).sum())}
            print("TABLE %s rows %d valued %d sha256 %s" % (nm, len(v), int(np.isfinite(v).sum()), rec["tables"][nm]), flush=True)
    rec["pass"] = not failed
    rec["failed"] = failed
    with open(os.path.join(out_dir, "TAB_TABLES_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec, failed


# ====================================================================================================== selftest
def synth(root, n_clips=21, seeds=SEEDS_DEFAULT, seed=11, run_name="I10RQ50cKF5P30Hb20"):
    """a synthetic world in new folders root/dump, root/tk, root/runs: a dump per seed (label columns identical between the seeds, p different), the T25 tables (closing estimate correlated with the label
    on the clip itself), and two rule logs per seed of banked RRc runs consistent with the L100 definition (looks at the replan ticks and at the later looks, the logged closing input = the label closing
    at (clip, frame) at EVERY look, 0.0 beyond the clip). Returns the label arrays for the tests."""
    rg = np.random.default_rng(seed)
    for d in ("dump", "tk", "runs"):
        os.makedirs(os.path.join(root, d))
    cid, ci, fr, pres, run, clo = [], [], [], [], [], []
    lens = {}
    for c in range(n_clips):
        T = int(rg.integers(240, 260))
        lens[c] = T
        pr = np.zeros(T, bool)
        j = 0
        while j < T:                                       # alternating segments without and with a lead
            j += int(rg.integers(3, 25))
            L = int(rg.integers(8, 60))
            pr[j:j + L] = True
            j += L
        rn = np.zeros(T, np.int64)
        for t in range(T):
            rn[t] = min(5, (int(rn[t - 1]) if t > 0 else 0) + 1) if pr[t] else 0
        cl = np.zeros(T)
        ar = 0.0
        for t in range(T):
            ar = 0.9 * ar + rg.normal(0, 1.0)
            if pr[t] and rn[t] >= RUN_MIN:
                cl[t] = 0.0 if rg.uniform() < 0.1 else 1.5 * ar
        cid += [c] * T
        ci += [c] * T
        fr += list(range(T))
        pres.append(pr)
        run.append(rn)
        clo.append(cl)
    cid, ci, fr = np.array(cid, np.int64), np.array(ci, np.int64), np.array(fr, np.int64)
    pres, run, clo = np.concatenate(pres), np.concatenate(run), np.concatenate(clo)
    n = len(cid)
    lab_gap = np.where(pres, rg.uniform(5, 50, n), np.nan)
    clips = np.array(["clip%03d" % c for c in range(n_clips)])
    for k in seeds:
        share = rg.uniform(0.2, 0.9, n_clips)
        p = np.where(rg.uniform(size=n) < share[cid], 0.9, 0.1)
        np.savez(os.path.join(root, "dump", "reader_s%d_score.npz" % k), cid=cid, clip_index=ci, frame=fr, p=p, gap=rg.uniform(5, 50, n), clos=rg.normal(0, 0.1, n), valid=np.ones(n, bool), lab_present=pres,
                 lab_gap=lab_gap, lab_closing=clo, lab_run=run, clips=clips, dt_s=np.float64(0.2), seed=np.int64(k))
        est = 0.6 * clo + rg.normal(0, 0.7, n)
        est[fr < 24] = np.nan
        np.savez(os.path.join(root, "tk", "tokclose_T25_s%d_table.npz" % k), clip_index=ci, frame=fr, est=est, extra_array=np.arange(5))
        key = "%s_s%d" % (run_name, k)
        td = os.path.join(root, "runs", key, "ticks")
        os.makedirs(td)
        lookup = {(int(a), int(b)): float(c) for a, b, c in zip(ci, fr, clo)}
        rule = {"ci": [], "j": [], "c_src": [], "present": []}
        mid = {"ci": [], "j": [], "c_src": [], "present": []}
        for c in range(n_clips):
            for j in range(0, lens[c] + 6, 10):           # replan ticks (a tick may sit beyond the dump's last frame: the label source returns 0.0)
                rule["ci"].append(c)
                rule["j"].append(j)
                rule["c_src"].append(lookup.get((c, j), 0.0))
                rule["present"].append(bool(rg.uniform() < 0.5))
                for q in (5,):
                    mid["ci"].append(c)
                    mid["j"].append(j + q)
                    mid["c_src"].append(lookup.get((c, j + q), 0.0))
                    mid["present"].append(bool(rg.uniform() < 0.5))
        np.savez(os.path.join(td, "rule_log_%s.npz" % key), ci=np.array(rule["ci"], np.int64), j=np.array(rule["j"], np.int64), c_src=np.array(rule["c_src"]), present=np.array(rule["present"], bool))
        np.savez(os.path.join(td, "rule_log_mid_%s.npz" % key), ci=np.array(mid["ci"], np.int64), j=np.array(mid["j"], np.int64), c_src=np.array(mid["c_src"]), present=np.array(mid["present"], bool))
    return {"ci": ci, "frame": fr, "lab_present": pres, "lab_run": run, "lab_closing": clo, "n_clips": n_clips}


def selftest():
    res = []

    def need(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        res.append(bool(cond))

    root = tempfile.mkdtemp(prefix="tmp_selftt_", dir=HERE)
    seeds = SEEDS_DEFAULT
    world = synth(root)
    args = dict(dump_dir=os.path.join(root, "dump"), tk_dir=os.path.join(root, "tk"), fit_dir="", rrc_run_dir=os.path.join(root, "runs"), rrc_run_name="I10RQ50cKF5P30Hb20", seeds=list(seeds))
    cnt = [0]

    def go(tamper=None, **over):
        cnt[0] += 1
        a = dict(args, **over)
        a.setdefault("out_dir", os.path.join(root, "out_%d" % cnt[0]))
        return run(tamper=tamper, **a)

    rec, failed = go()
    need("clean case: every control passes and the record says pass", rec["pass"] is True and not failed and all(rec[c]["PASS"] for c in ("a0", "a1", "a4_u1", "a4_u4")))
    out = os.path.join(root, "out_1")
    need("the sixteen table files and the record exist", all(os.path.isfile(os.path.join(out, "tab_%s_s%d.npz" % (a, k))) for a in TABLE_ARMS for k in seeds) and os.path.isfile(os.path.join(out, "TAB_TABLES_RECORD.json")))
    need("record 'tables' holds a sha256 per file", len(rec["tables"]) == 16 and all(len(v) == 64 for v in rec["tables"].values()))
    tk0 = load_tab(os.path.join(out, "tab_TK_s0.npz"))
    tkin = load_tab(os.path.join(root, "tk", "tokclose_T25_s0_table.npz"))
    need("TK equals the input table exactly", all(np.array_equal(x, y, equal_nan=True) for x, y in zip(tk0, tkin)))
    l0 = load_tab(os.path.join(out, "tab_L100_s0.npz"))
    want = np.where(world["lab_present"] & (world["lab_run"] >= RUN_MIN), world["lab_closing"], np.nan)
    need("L100 equals the label closing speed where the label has a slope (present, run >= 3), NaN elsewhere; a valued zero stays a value", np.array_equal(l0[2], want, equal_nan=True) and int((l0[2] == 0.0).sum()) > 0)
    # refusal of an existing folder and of differing label columns
    try:
        go(out_dir=out)
        need("an existing out folder is refused", False)
    except SystemExit as e:
        need("an existing out folder is refused", "REFUSED" in str(e))
    p1 = os.path.join(root, "dump", "reader_s1_score.npz")
    z = load_npz_all(p1)
    z2 = dict(z)
    z2["lab_closing"] = z["lab_closing"].copy()
    z2["lab_closing"][5] += 1.0
    d2 = os.path.join(root, "dump_bad")
    os.makedirs(d2)
    np.savez(os.path.join(d2, "reader_s1_score.npz"), **z2)
    z0 = load_npz_all(os.path.join(root, "dump", "reader_s0_score.npz"))
    np.savez(os.path.join(d2, "reader_s0_score.npz"), **z0)
    try:
        go(dump_dir=d2)
        need("label columns that differ between the two seeds' dumps are refused", False)
    except SystemExit as e:
        need("label columns that differ between the two seeds' dumps are refused", "label column" in str(e))
    # ---- planted failures, one control each
    def planted(label, tamper, control, **over):
        r, f = go(tamper, **over)
        need("planted %s fails %s" % (label, control), (control in f) and r["pass"] is False)
        return r

    def t_wrong_r(kind, tab):
        if kind[0] == "tab" and kind[1] == "W70" and kind[2] == 0:
            e = tab["est"].copy()
            V = np.isfinite(e)
            Lv = l0[2]
            e[V] = e[V] + 0.15 * (Lv[V] - np.nanmean(Lv))
            tab["est"] = e
    planted("a wrong r (W70 with 0.15 of L added)", t_wrong_r, "a1")

    def t_wrong_sd(kind, tab):
        if kind[0] == "tab" and kind[1] == "W90" and kind[2] == 1:
            e = tab["est"].copy()
            V = np.isfinite(e)
            e[V] = e[V].mean() + 1.2 * (e[V] - e[V].mean())
            tab["est"] = e
    planted("a wrong spread (W90 sd x 1.2)", t_wrong_sd, "a1")

    def t_wrong_mean(kind, tab):
        if kind[0] == "tab" and kind[1] == "W50" and kind[2] == 0:
            tab["est"] = tab["est"] + 0.2
    planted("a wrong mean (W50 + 0.2)", t_wrong_mean, "a1")

    def t_ac_s70(kind, tab):                                          # S70 replaced by a W70 table: noise independent in time
        if kind[0] == "tab" and kind[1] == "S70" and kind[2] == 0:
            tab["est"] = make_arm("W70", l0[2], l0[0], l0[1], 0)
    planted("an S70 table with white noise (lag-1 autocorrelation 0)", t_ac_s70, "a1")

    def t_ac_w70(kind, tab):                                          # W70 replaced by an S70 table
        if kind[0] == "tab" and kind[1] == "W70" and kind[2] == 1:
            tab["est"] = make_arm("S70", l0[2], l0[0], l0[1], 0)
    planted("a W70 table with smooth noise", t_ac_w70, "a1")

    def t_valued(kind, tab):
        if kind[0] == "tab" and kind[1] == "W50" and kind[2] == 1:
            e = tab["est"].copy()
            e[int(np.nonzero(np.isfinite(e))[0][3])] = np.nan
            tab["est"] = e
    planted("a valued frame of W50 dropped", t_valued, "a1")

    def t_valued2(kind, tab):
        if kind[0] == "tab" and kind[1] == "W90" and kind[2] == 0:
            e = tab["est"].copy()
            e[int(np.nonzero(~np.isfinite(e))[0][3])] = 0.5
            tab["est"] = e
    planted("a value on a frame that L100 has none", t_valued2, "a1")

    def t_l100_shift(kind, tab):
        if kind[0] == "tab" and kind[1] == "L100" and kind[2] == 0:
            e = tab["est"].copy()
            V = np.isfinite(e) & (e != 0.0)
            e[V] = e[V] + 1e-3
            tab["est"] = e
    planted("a shifted label in L100 (+1e-3 on every nonzero value)", t_l100_shift, "a0")

    def t_l100_frame(kind, tab):
        if kind[0] == "tab" and kind[1] == "L100" and kind[2] == 1:
            tab["frame"] = tab["frame"] + 1
    planted("a wrong frame key in L100 (frame + 1)", t_l100_frame, "a0")

    def t_l100_zero_nonzero(kind, tab):                                # a table that has a value where the logs hold 0: every NaN row set to 3.0
        if kind[0] == "tab" and kind[1] == "L100" and kind[2] == 0:
            e = tab["est"].copy()
            e[~np.isfinite(e)] = 3.0
            tab["est"] = e
    planted("a table with values where the log reads 0", t_l100_zero_nonzero, "a0")

    def t_fixed_point(kind, d):
        if kind[0] == "map":
            d["map"] = d["map"].copy()
            d["map"][0] = d["map_clips"][0]
    planted("a fixed point in the map", t_fixed_point, "a4_u1")

    def t_wrong_value(kind, tab):
        if kind[0] == "tab" and kind[1] == "STK" and kind[2] == 0:
            e = tab["est"].copy()
            j = int(np.nonzero(np.isfinite(e))[0][50])
            e[j] += 0.001
            tab["est"] = e
    planted("a wrong value in STK", t_wrong_value, "a4_u1")

    def t_same(kind, tab):
        if kind[0] == "tab" and kind[1] == "STK":
            tab["est"] = tk0[2].copy() if kind[2] == 0 else load_tab(os.path.join(out, "tab_TK_s1.npz"))[2].copy()
    r = planted("a surrogate equal to the estimate", t_same, "a4_u4")
    need("the surrogate equal to the estimate reads r above the band", r["a4_u4"]["r_surrogate"] > 0.15)
    # a0 on a log with a missing file
    import shutil
    bad_runs = os.path.join(root, "runs_missing")
    os.makedirs(bad_runs)
    shutil.copytree(os.path.join(root, "runs", "I10RQ50cKF5P30Hb20_s0"), os.path.join(bad_runs, "I10RQ50cKF5P30Hb20_s0"))
    r, f = go(rrc_run_dir=bad_runs)
    need("a missing rule log of seed 1 fails a0", "a0" in f)
    need("the autoregression restarts at every clip and has the stated lag-1 coefficient", abs(_lag1_all() - 0.9) < 0.03)
    ok = all(res)
    print("ZZ SELFTEST " + ("PASS" if ok else "FAIL") + " checks %d failed %d tmp %s" % (len(res), len(res) - sum(res), os.path.basename(root)))
    return 0 if ok else 1


def _lag1_all():
    rg = np.random.default_rng(5)
    n_c, T = 40, 400
    ci = np.repeat(np.arange(n_c), T)
    fr = np.tile(np.arange(T), n_c)
    e = ar_along_clips(rg.standard_normal(n_c * T), ci, fr)
    same = ci[1:] == ci[:-1]
    return float(np.corrcoef(e[:-1][same], e[1:][same])[0, 1])


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dump-dir", default="/workspace/v8proof/closeread/dump/both")
    ap.add_argument("--tk-dir", default="/workspace/v8proof/tokclose/fits")
    ap.add_argument("--fit-dir", default="/workspace/v8proof/closeread")
    ap.add_argument("--rrc-run-dir", default="/workspace/v8proof/integ10k/pull")
    ap.add_argument("--rrc-run-name", default="I10RQ50cKF5P30Hb20")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out-dir", default="/workspace/v8proof/closeread/est_tab")
    ap.add_argument("--surrogate-tool-dir", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    rec, failed = run(a.dump_dir, a.tk_dir, a.fit_dir, a.rrc_run_dir, a.rrc_run_name, a.out_dir, seeds, a.surrogate_tool_dir)
    print("ZZ TABTABLES " + ("PASS" if not failed else "FAIL " + "; ".join(failed)))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
