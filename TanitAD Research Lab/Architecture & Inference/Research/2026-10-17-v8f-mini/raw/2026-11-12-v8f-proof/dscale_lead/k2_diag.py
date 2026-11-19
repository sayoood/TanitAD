"""k2_diag: what control k2 compares (reader-side head output vs the pod's written output), per arm, with the statistics that say whether the difference is float32 rounding.
Reuses dscale_lead_read (ARM_MAP, _files) and the reader's own TKR/N functions (load_head, load_states, _history_idx, row_outputs, head_out_units). Reads no label and no outcome.
  python k2_diag.py --in-dir half1/out/states --out-dir half1/k2diag
The float64 line is a supplementary forward written here with the same weights (the reader's row_outputs is float32 only)."""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dscale_lead_read as D  # noqa: E402

N, TKR = D.N, D.TKR
PAIRS = (("P16C_s0", "P16C_s1"), ("P16U_s0", "P16U_s1"), ("P16CSH_s0", "P16C_s0"), ("P16USH_s0", "P16U_s0"))


def forward64(T, Z, idx5):
    import torch
    import torch.nn.functional as F
    ok = (idx5 >= 0).all(1)
    out = np.full((len(Z), 3), np.nan, np.float64)
    d = torch.float64
    with torch.no_grad():
        W, b = torch.from_numpy(T["frame_w"]).to(d), torch.from_numpy(T["frame_b"]).to(d)
        E = F.linear(torch.from_numpy(np.asarray(Z, np.float32)).to(d), W, b).numpy()
        rows = np.flatnonzero(ok)
        for i in range(0, len(rows), 20000):
            r = rows[i:i + 20000]
            w = torch.from_numpy(E[idx5[r]].reshape(len(r), -1))
            h = F.gelu(F.linear(w, torch.from_numpy(T["mlp0_w"]).to(d), torch.from_numpy(T["mlp0_b"]).to(d)))
            out[r] = F.linear(h, torch.from_numpy(T["mlp2_w"]).to(d), torch.from_numpy(T["mlp2_b"]).to(d)).numpy()
    return out, ok


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def stats(dev, pod):
    d = np.abs(dev - pod)
    i = int(np.argmax(d))
    return {"largest_abs_diff": float(d[i]), "pod_value_at_max": float(pod[i]), "reader_value_at_max": float(dev[i]),
            "p50": float(np.quantile(d, .5)), "p99": float(np.quantile(d, .99)), "p999": float(np.quantile(d, .999)),
            "largest_abs_pod_value": float(np.abs(pod).max()), "largest_rel_diff": float((d / np.maximum(np.abs(pod), 1.0)).max()),
            "pearson_abs_pod_vs_abs_diff": corr(np.abs(pod), d), "spearman_abs_pod_vs_abs_diff": corr(rank(np.abs(pod)), rank(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res, side, L = {"arms": {}, "planted": {}}, {}, []
    for label, (folder, arm, steps, sh, corpus) in D.ARM_MAP.items():
        files = D._files(a.in_dir, folder, arm)
        T = N.load_head(files["eval_lead"] + ".head.npz")
        st = N.load_states(files["eval_states"])
        idx, _ = N._history_idx(st, T["hist"])
        ze = np.load(files["eval_lead"], allow_pickle=False)
        pod = {k: ze[k] for k in ("gap_m", "closing_ms", "has_history")}
        raw, ok = N.row_outputs(T, st["Z"], idx)
        dev = N.head_out_units(np.where(np.isfinite(raw), raw, np.nan))
        both = ok & np.isfinite(pod["gap_m"]) & np.isfinite(pod["closing_ms"])
        r = {"rows_compared": int(both.sum()), "dtype_reader_gap": str(dev["gap_m"].dtype), "dtype_reader_raw": str(raw.dtype),
             "dtype_pod_gap": str(pod["gap_m"].dtype), "dtype_pod_closing": str(pod["closing_ms"].dtype),
             "gap": stats(dev["gap_m"][both], np.asarray(pod["gap_m"], np.float64)[both]),
             "closing": stats(dev["closing_ms"][both], np.asarray(pod["closing_ms"], np.float64)[both])}
        raw64, _ = forward64(T, st["Z"], idx)
        d64 = N.head_out_units(np.where(np.isfinite(raw64), raw64, np.nan))
        r["float64_reader_vs_pod"] = {"gap": float(np.abs(d64["gap_m"][both] - pod["gap_m"][both]).max()),
                                      "closing": float(np.abs(d64["closing_ms"][both] - pod["closing_ms"][both]).max()),
                                      "float64_vs_float32_reader_gap": float(np.abs(d64["gap_m"][both] - dev["gap_m"][both]).max())}
        res["arms"][label] = r
        side[label] = (dev, ok, pod)
        g, c = r["gap"], r["closing"]
        L.append("%s n=%d dt(reader,pod)=%s,%s gapMax=%.3e pod@max=%.4f rd@max=%.4f maxAbsPod=%.2f relMax=%.3e p999=%.3e r_pear=%.3f r_spear=%.3f | clsMax=%.3e relMax=%.3e p999=%.3e r_pear=%.3f | f64 gap=%.3e cls=%.3e (f64-f32 reader gap %.3e)"
                 % (label, r["rows_compared"], r["dtype_reader_gap"], r["dtype_pod_gap"], g["largest_abs_diff"], g["pod_value_at_max"], g["reader_value_at_max"], g["largest_abs_pod_value"],
                    g["largest_rel_diff"], g["p999"], g["pearson_abs_pod_vs_abs_diff"], g["spearman_abs_pod_vs_abs_diff"], c["largest_abs_diff"], c["largest_rel_diff"], c["p999"],
                    c["pearson_abs_pod_vs_abs_diff"], r["float64_reader_vs_pod"]["gap"], r["float64_reader_vs_pod"]["closing"], r["float64_reader_vs_pod"]["float64_vs_float32_reader_gap"]))
    L.append("PLANTED (reader-side of A vs pod-side of B, same rows):")
    for A, B in PAIRS:
        devA, okA, _ = side[A]
        podB = side[B][2]
        if len(devA["gap_m"]) != len(podB["gap_m"]):
            res["planted"]["%s|%s" % (A, B)] = {"skipped": "row counts differ"}
            L.append("%s vs %s: row counts differ" % (A, B))
            continue
        both = okA & np.isfinite(podB["gap_m"]) & np.isfinite(podB["closing_ms"])
        dg = float(np.abs(devA["gap_m"][both] - podB["gap_m"][both]).max())
        dc = float(np.abs(devA["closing_ms"][both] - podB["closing_ms"][both]).max())
        res["planted"]["%s|%s" % (A, B)] = {"rows": int(both.sum()), "max_abs_gap_m": dg, "max_abs_closing_ms": dc}
        L.append("%s reader vs %s pod: rows=%d gapMax=%.4e closingMax=%.4e" % (A, B, int(both.sum()), dg, dc))
    json.dump(res, open(a.out_dir + "/K2_DIAG.json", "w", encoding="utf-8"), indent=1)
    L.append("ZZ K2-DIAG DONE")
    open(a.out_dir + "/K2_DIAG.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
