"""nv_select.py -- per-log selection for the NavSim navtest reels (CPU only).  Rule is stated in README.md BEFORE rendering.

Reads the three per-token score CSVs (navsim_bars8.read_scores_csv) + navtest_inputs.json.gz, orders each log's tokens by the CURRENT frame's
timestamp, splits into contiguous RUNS (gap <= GAP_S between consecutive scenes' current frames), and ranks logs by mean(PDMS theta_0.5 - PDMS EMA)
over the tokens of the log's longest run (>= 40 tokens).  Writes selection.json.
"""
import gzip, json, sys, importlib.util
import numpy as np, pandas as pd

BARS = "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/navsim_bars8.py"
CSV = {"W": "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/stage2/scores_navtest/r8nt_w0p50_R8_LEGAL/r8nt_w0p50_R8_LEGAL.csv",
       "E": "C:/Users/Admin/refcv8_pf/navtest_read/raw/full/scores_navtest/r8nt_s48852e_R8_RT/r8nt_s48852e_R8_RT.csv",
       "P": "C:/Users/Admin/refcv8_pf/navtest_read/raw/full/scores_navtest/r8nt_s48852e_PRIOR_ha0p/r8nt_s48852e_PRIOR_ha0p.csv"}
INP = "C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/inputs/navtest_inputs.json.gz"
MIN_TOK = 40


def load_bars():
    sys.path.insert(0, BARS.rsplit("/", 1)[0])
    spec = importlib.util.spec_from_file_location("navsim_bars8_v", BARS)
    m = importlib.util.module_from_spec(spec); sys.modules["navsim_bars8_v"] = m; spec.loader.exec_module(m)
    return m


def load_all():
    B = load_bars()
    sc = {k: B.read_scores_csv(p) for k, p in CSV.items()}
    inp = json.load(gzip.open(INP, "rt", encoding="utf-8"))["tokens"]
    return sc, inp


def runs_by_log(inp):
    """{log: [ [tok,...] runs ]} -- tokens of a log ordered by t0 timestamp; a run breaks where the t0 gap exceeds 3x the log's median gap."""
    by = {}
    for t, r in inp.items():
        by.setdefault(r["log_name"], []).append((r["timestamps_us"][-1], t))
    out = {}
    for ln, lst in by.items():
        lst.sort()
        ts = np.array([x[0] for x in lst], np.float64) / 1e6
        gaps = np.diff(ts)
        med = float(np.median(gaps)) if len(gaps) else 0.0
        runs, cur = [], [lst[0][1]]
        for i, g in enumerate(gaps):
            if g > max(3 * med, 0.0) and med > 0:
                runs.append(cur); cur = []
            cur.append(lst[i + 1][1])
        runs.append(cur)
        out[ln] = {"runs": runs, "median_gap_s": med, "gaps_s": gaps}
    return out


if __name__ == "__main__":
    sc, inp = load_all()
    for k, d in sc.items():
        print(k, len(d), "mean PDMS x100 = %.3f" % (100 * d["PDMS"].mean()), "n_invalid_dropped", d.attrs.get("n_invalid_dropped"))
    R = runs_by_log(inp)
    gs = np.concatenate([v["gaps_s"] for v in R.values() if len(v["gaps_s"])])
    print("t0 gap between consecutive tokens (s): min %.3f p5 %.3f median %.3f p95 %.3f max %.3f" % (gs.min(), np.percentile(gs, 5), np.median(gs), np.percentile(gs, 95), gs.max()))
    vals, cnts = np.unique(np.round(gs, 1), return_counts=True)
    print("rounded gap histogram:", dict(zip(vals.tolist(), cnts.tolist())) if len(vals) < 30 else "many")
    print("logs:", len(R), " logs with a run >= %d: %d" % (MIN_TOK, sum(1 for v in R.values() if max(len(r) for r in v['runs']) >= MIN_TOK)))


# ---------------------------------------------------------------- selection (rule in ../README.md section 1) --------------------------- #
SEAMS = {"W": "C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/seams_w050/seam_R8_LEGAL.npz",
         "E_LEGAL": "C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/seams/seam_R8_LEGAL.npz",
         "E_RT": "C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/seams/seam_R8_RT.npz",
         "P": "C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/seams/seam_PRIOR_ha0p.npz"}
CSV_EMA = {"LEGAL": "C:/Users/Admin/refcv8_pf/navtest_read/raw/full/scores_navtest/r8nt_s48852e_R8_LEGAL/r8nt_s48852e_R8_LEGAL.csv",
           "RT": CSV["E"]}
GAP_MAX_S = 0.6


def contiguous_runs(inp):
    by = {}
    for t, r in inp.items():
        by.setdefault(r["log_name"], []).append((r["timestamps_us"][-1], t))
    out = {}
    for ln, lst in by.items():
        lst.sort()
        runs, cur = [], [lst[0]]
        for a, b in zip(lst[:-1], lst[1:]):
            if (b[0] - a[0]) / 1e6 <= GAP_MAX_S:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        out[ln] = max(runs, key=len)
    return out


def select(ema_arm):
    B = load_bars()
    inp = json.load(gzip.open(INP, "rt", encoding="utf-8"))["tokens"]
    W = B.read_scores_csv(CSV["W"]); E = B.read_scores_csv(CSV_EMA[ema_arm])
    best = contiguous_runs(inp)
    rows = []
    for ln, run in best.items():
        if len(run) < MIN_TOK:
            continue
        toks = [t for _, t in run]
        d = (W.loc[toks, "PDMS"] - E.loc[toks, "PDMS"])
        dall = [t for t, r in inp.items() if r["log_name"] == ln]
        rows.append({"log": ln, "n_tokens": len(toks), "t0_first_us": run[0][0], "t0_last_us": run[-1][0],
                     "span_s": (run[-1][0] - run[0][0]) / 1e6, "mean_delta_pdms_x100": 100 * float(d.mean()),
                     "mean_pdms_W_x100": 100 * float(W.loc[toks, "PDMS"].mean()), "mean_pdms_E_x100": 100 * float(E.loc[toks, "PDMS"].mean()),
                     "mean_pdms_P_x100": 100 * float(P_SC.loc[toks, "PDMS"].mean()),
                     "whole_log_n": len(dall), "whole_log_mean_delta_x100": 100 * float((W.loc[dall, "PDMS"] - E.loc[dall, "PDMS"]).mean()),
                     "tokens": toks})
    rows.sort(key=lambda r: -r["mean_delta_pdms_x100"])
    n = len(rows)
    med = (n - 1) / 2.0
    order_mid = sorted(range(n), key=lambda i: abs(i - med))
    top, bot, mid = list(range(min(2, n))), list(range(max(0, n - 2), n)), [i for i in order_mid if i not in range(min(2, n)) and i not in range(max(0, n - 2), n)][:2]
    pick = sorted(set(top + mid + bot))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["tier"] = "HIGH" if i in top else ("LOW" if i in bot else ("MID" if i in mid else "--"))
        r["picked"] = i in pick
    return rows, n


P_SC = None

if __name__ == "__main__" and "--select" in sys.argv:
    B = load_bars(); P_SC = B.read_scores_csv(CSV["P"])
    out = {}
    for arm in ("LEGAL", "RT"):
        rows, n = select(arm)
        out[arm] = {"n_eligible": n, "rows": rows}
        print("EMA arm", arm, "eligible logs:", n)
        for r in rows:
            print("  rank %d %-4s %s n=%d span=%.1fs  delta=%+.2f  W=%.2f E=%.2f P=%.2f  (whole log n=%d delta=%+.2f) picked=%s" % (
                r["rank"], r["tier"], r["log"], r["n_tokens"], r["span_s"], r["mean_delta_pdms_x100"], r["mean_pdms_W_x100"], r["mean_pdms_E_x100"],
                r["mean_pdms_P_x100"], r["whole_log_n"], r["whole_log_mean_delta_x100"], r["picked"]))
    json.dump(out, open("../selection.json", "w", encoding="utf-8"), indent=1)
