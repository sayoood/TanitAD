"""fcread10k.py -- card PREREG_V8F_PROOF_ADD146 (FCREAD10K): can the reader carry the lead forecast of card ADD144? A T0-style design read on logged frames.

  python fcread10k.py --pull PULL --join join_eval141.npz --fc-ref <H_RLcE1FC_s0 fc_log npz> --out OUT.json [--runs H_RQ50KE1_s0,H_RQ50KE1_s1,H_RQ50KE1_s2]

Reader rows come from the tick and later-look logs of the runs (with their _rd side files): one entry per (clip, frame), deduplicated over windows.
Reader lead at x: p >= 0.5, history ok, finite used gap. v_L,r(f) = max(0, v_l(f) + (g(f) - g(f-5)) / 1.0); a_L,r(f) = (v_L,r(f) - v_L,r(f-5)) / 1.0.
Label side: LabelSource of the rule wrapper on the join (rows at or before the frame); v_L = max(0, v_l + label closing), a_L as rzgfhfc.lead_accel.
S = Pearson r of the reader closing against the label closing at f; A = Pearson r of a_L,r against the label a_L. Bars 0.30. Groups: reader seeds (runs s0 and s2
share TK16_s0, run s1 uses TK16_s1). Controls k1 to k5 (k5 by amendment 1 of the card) run before any value is written; values are written only when all pass.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, "..", ".."))
KB = os.path.join(V8, "rhacc10k", "kit_build")
sys.path.insert(0, KB)
import v8f_t1_eval_rule as W  # noqa: E402

DT, K, P_MIN, BAR = 0.2, 5, 0.5, 0.30
READER_OF = {0: "TK16_s0", 1: "TK16_s1", 2: "TK16_s0"}


def load_join(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- pure parts (tested on synthetic data)
def reader_v(R, ci, x):
    """(ok, v_L,r at frame x) from the reader map R[(ci, frame)] = (v_l, p, g, ok)."""
    a, b = R.get((ci, x)), R.get((ci, x - K))
    if a is None or b is None:
        return False, float("nan")
    for r in (a, b):
        if not (r[1] >= P_MIN and r[3] and np.isfinite(r[2]) and np.isfinite(r[0])):
            return False, float("nan")
    return True, max(0.0, a[0] + (a[2] - b[2]) / (K * DT))


def reader_row(R, ci, f):
    """(ok, reader closing at f, a_L,r at f); reads frames f, f-5, f-10 only."""
    ok1, v1 = reader_v(R, ci, f)
    ok0, v0 = reader_v(R, ci, f - K)
    if not (ok1 and ok0):
        return False, float("nan"), float("nan")
    a, b = R[(ci, f)], R[(ci, f - K)]
    return True, (a[2] - b[2]) / (K * DT), (v1 - v0) / (K * DT)


def label_v(label, V, ci, x):
    v = V.get((ci, x))
    if v is None:
        return False, float("nan"), float("nan")
    pres, g, c = label.read(int(ci), int(x))
    if not pres or not (np.isfinite(g) and np.isfinite(c) and np.isfinite(v)):
        return False, float("nan"), float("nan")
    return True, max(0.0, v + float(c)), float(c)


def label_row(label, V, ci, f):
    """(ok, label closing at f, label a_L at f) exactly as rzgfhfc.lead_accel where both frames have a lead."""
    ok1, v1, c1 = label_v(label, V, ci, f)
    ok0, v0, _ = label_v(label, V, ci, f - K)
    if not (ok1 and ok0):
        return False, float("nan"), float("nan")
    return True, c1, (v1 - v0) / (K * DT)


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return pearson(rx, ry)


# ---------------------------------------------------------------- loading
def run_maps(pull, run):
    """-> R {(ci, frame): (v_l, p, g_used, ok)}, V {(ci, frame): v_l}, alignment record (k5)."""
    d = os.path.join(pull, run, "ticks")
    R, V = {}, {}
    al = {"rows": 0, "present_rows": 0, "max_abs_gsrc_minus_gused": 0.0, "len_equal": True, "conflicts": 0}
    for log, rd in (("rule_log_%s.npz" % run, "rule_log_%s_rd.npz" % run), ("rule_log_mid_%s.npz" % run, "rule_log_mid_%s_rd.npz" % run)):
        with np.load(os.path.join(d, log), allow_pickle=True) as z, np.load(os.path.join(d, rd), allow_pickle=True) as r:
            ci, j, vl, gs, pres = z["ci"], z["j"], z["v_l"], z["g_src"], z["present"]
            p, g, ok = r["p"], r["g_used"], r["ok"]
            if not (len(ci) == len(p) == len(g) == len(ok)):
                al["len_equal"] = False
                continue
            al["rows"] += int(len(ci))
            m = pres & np.isfinite(gs) & np.isfinite(g)
            al["present_rows"] += int(m.sum())
            if m.any():
                al["max_abs_gsrc_minus_gused"] = max(al["max_abs_gsrc_minus_gused"], float(np.max(np.abs(gs[m] - np.asarray(g, np.float64)[m]))))
            for i in range(len(ci)):
                key = (int(ci[i]), int(j[i]))
                rec = (float(vl[i]), float(p[i]), float(g[i]), bool(ok[i]))
                old = R.get(key)
                if old is not None and not (np.allclose(old[:3], rec[:3], equal_nan=True) and old[3] == rec[3]):
                    al["conflicts"] += 1
                R[key] = rec
                V[key] = float(vl[i])
    return R, V, al


def merge(maps):
    R, V, n_conf = {}, {}, 0
    for Rm, Vm in maps:
        for k, v in Rm.items():
            if k in R and not (np.allclose(R[k][:3], v[:3], equal_nan=True) and R[k][3] == v[3]):
                n_conf += 1
            R[k] = v
        V.update(Vm)
    return R, V, n_conf


def rows_for(R, V, label):
    out = []
    for (ci, f) in sorted(R):
        okr, cr, ar = reader_row(R, ci, f)
        if not okr:
            continue
        okl, cl, al = label_row(label, V, ci, f)
        if not okl:
            continue
        out.append((ci, f, cr, ar, cl, al))
    return out


# ---------------------------------------------------------------- controls
def k1(label, V, fc_ref):
    with np.load(fc_ref) as z:
        ci, f, aL, okp, pres = z["ci"], z["f"], z["a_L"], z["ok_prev"], z["present"]
    n, bad, seen = 0, 0, set()
    for i in range(len(ci)):
        key = (int(ci[i]), int(f[i]))
        if key in seen or not (pres[i] and okp[i]):
            continue
        seen.add(key)
        ok, _, a = label_row(label, V, key[0], key[1])
        if not ok:
            continue
        n += 1
        if a != float(aL[i]):
            bad += 1
    return {"shared_rows": n, "differing": bad, "PASS": bool(n > 0 and bad == 0)}


class _Lab:
    def __init__(self, rows):
        self.rows = rows

    def read(self, ci, j):
        return self.rows.get((ci, j), (False, float("nan"), 0.0))


def k2_k3():
    R = {(0, x): (10.0, 0.9, 30.0 - 2.0 * x * DT, True) for x in range(0, 40)}
    ok, c, a = reader_row(R, 0, 20)
    k2a = bool(ok and abs(c + 2.0) < 1e-12 and abs(a) < 1e-12)
    R2 = {(0, x): (10.0, 0.9, 30.0 - 2.0 * x * DT - 0.5 * 1.0 * (x * DT) ** 2, True) for x in range(0, 40)}
    ok2, c2, a2 = reader_row(R2, 0, 20)
    k2b = bool(ok2 and abs(a2 + 1.0) < 1e-9)                     # gap with second derivative -1 m/s2: the closing falls by 1 m/s per second
    R3 = dict(R2)
    for x in range(21, 40):
        R3[(0, x)] = (99.0, 0.9, -50.0, True)
    k3 = bool(reader_row(R3, 0, 20)[1:] == reader_row(R2, 0, 20)[1:])
    lab = _Lab({(0, 20): (True, 10.0, -1.0), (0, 15): (True, 11.0, 0.5)})
    lab2 = _Lab({(0, 20): (True, 10.0, -1.0), (0, 15): (True, 11.0, 0.5), (0, 21): (True, 1.0, -9.0)})
    V = {(0, x): 5.0 for x in range(40)}
    k3b = bool(label_row(lab, V, 0, 20) == label_row(lab2, V, 0, 20))
    return {"k2_constant_closing": k2a, "k2_planted_ramp": k2b, "PASS": bool(k2a and k2b)}, {"k3_reader_future": k3, "k3_label_future": k3b, "PASS": bool(k3 and k3b)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--fc-ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="H_RQ50KE1_s0,H_RQ50KE1_s1,H_RQ50KE1_s2")
    a = ap.parse_args(argv)
    label = W.LabelSource(load_join(a.join))
    runs = [x for x in a.runs.split(",") if x]
    loaded = {r: run_maps(a.pull, r) for r in runs}
    groups = {}
    for r in runs:
        m = int(r.rsplit("_s", 1)[1])
        groups.setdefault(READER_OF[m], []).append(r)
    allV = {}
    for r in runs:
        allV.update(loaded[r][1])
    C = {"k1_label_aL_equals_fc_log": k1(label, allV, a.fc_ref)}
    C["k2_planted"], C["k3_planted_future"] = k2_k3()
    rows, n_conf = {}, {}
    for gname, rr in groups.items():
        R, V, nc = merge([(loaded[r][0], loaded[r][1]) for r in rr])
        rows[gname] = rows_for(R, V, label)
        n_conf[gname] = nc
    C["k4_rows"] = {"rows": {g: len(v) for g, v in rows.items()}, "PASS": all(len(v) >= 200 for v in rows.values())}
    al = {r: loaded[r][2] for r in runs}
    C["k5_alignment"] = {"per_run": al, "cross_run_conflicts": n_conf,
                         "PASS": all(x["len_equal"] and x["present_rows"] > 0 and x["max_abs_gsrc_minus_gused"] <= 1e-4 and x["conflicts"] == 0 for x in al.values())
                         and all(v == 0 for v in n_conf.values())}
    ok = all(v["PASS"] for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD146", "tier": "T0-style design read on logged frames (reader reads of logged camera tokens; nothing rolled out); never driving performance",
           "controls": C, "controls_all_pass": bool(ok), "groups": groups, "values": None, "name": None}
    if ok:
        vals = {}
        for gname, rw in rows.items():
            X = np.array(rw, np.float64)
            cr, ar, cl, alab = X[:, 2], X[:, 3], X[:, 4], X[:, 5]
            vals[gname] = {"n": int(len(X)), "clips": int(len(set(int(c) for c in X[:, 0]))), "S_pearson": round(pearson(cr, cl), 4), "A_pearson": round(pearson(ar, alab), 4),
                           "S_spearman": round(spearman(cr, cl), 4), "A_spearman": round(spearman(ar, alab), 4),
                           "rms_aL_reader_minus_label": round(float(np.sqrt(np.mean((ar - alab) ** 2))), 4), "rms_label_aL": round(float(np.sqrt(np.mean(alab ** 2))), 4),
                           "rms_closing_reader_minus_label": round(float(np.sqrt(np.mean((cr - cl) ** 2))), 4)}
        A_ok = all(v["A_pearson"] >= BAR for v in vals.values())
        S_ok = all(v["S_pearson"] >= BAR for v in vals.values())
        name = "FCREAD10K-CARRIED" if A_ok else ("FCREAD10K-SPEED-ONLY" if S_ok else "FCREAD10K-NOT-CARRIED")
        res.update(values=vals, name=name)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"controls": {k: v["PASS"] for k, v in C.items()}, "name": res["name"], "values": res["values"]}, indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
