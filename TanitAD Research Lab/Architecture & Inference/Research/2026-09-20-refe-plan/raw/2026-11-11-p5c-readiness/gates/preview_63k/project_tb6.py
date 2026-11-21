"""TB-6 full-bank PROJECTION (ESTIMATED, diagnostic only, never the gate).  MM 2026-11-12.

The dev box labelled 63,495 of 150,603 bank rows (logs with a local nuPlan DB).  The rest are labelled only on the pod after P7.  TB-6 is
a ratio of medians (P5-S's own rule, p5_target_labels.py:388), so the full-bank verdict depends on the unlabelled rows' raw `traj` too,
and that half needs no labels.  This script:
  1. reports the raw-traj medians (max |acc|, max |heading rate|) on labelled vs unlabelled rows, per rank (MEASURED);
  2. projects the full-bank LABEL median by reweighting the labelled rows to the unlabelled rows' raw distribution in 40 quantile bins of
     the same metric (covariate-shift assumption: label | raw-metric bin is the same across logs; UNVERIFIED), and divides by the
     full-bank raw median (ESTIMATED).
usage: python project_tb6.py <bank_dir with targets_rank{0,1}.jsonl> <labels_s0of2.jsonl> <labels_s1of2.jsonl>
"""
import json
import sys

import numpy as np

DT = 0.2


def learn(q20):
    q = np.concatenate([np.zeros((1, 3)), np.asarray(q20, np.float64)], 0)
    acc = np.linalg.norm(np.diff(q[:, :2], n=2, axis=0), axis=1) / DT ** 2
    hr = np.abs(np.diff(np.unwrap(q[:, 2]))) / DT
    return acc.max(), np.degrees(hr.max())


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def main():
    bank, l0, l1 = sys.argv[1:4]
    lab = {}
    for fp in (l0, l1):
        for ln in open(fp, encoding="utf-8"):
            r = json.loads(ln)
            if "tc20s" in r:
                lab[(r["log"], r["token"], r["rank"])] = r["tc20s"]
    out = {}
    for rk in (0, 1):
        L_t, L_l, U_t = [], [], []
        for ln in open(f"{bank}/targets_rank{rk}.jsonl", encoding="utf-8"):
            r = json.loads(ln)
            k = (r["log_name"], r["token"], int(r["rank"]))
            if k in lab:
                L_t.append(learn(r["traj"]))
                L_l.append(learn(lab[k]))
            else:
                U_t.append(learn(r["traj"]))
        L_t, L_l, U_t = map(np.array, (L_t, L_l, U_t))
        allT = np.concatenate([L_t, U_t])
        o = {"n_labelled": len(L_l), "n_unlabelled": len(U_t)}
        for m, name in ((1, "heading_rate_dps"), (0, "acc_mps2")):
            edges = np.quantile(allT[:, m], np.linspace(0, 1, 41))
            edges[0] -= 1
            edges[-1] += 1
            bl = np.clip(np.searchsorted(edges, L_t[:, m]) - 1, 0, 39)
            bu = np.clip(np.searchsorted(edges, U_t[:, m]) - 1, 0, 39)
            nl = np.bincount(bl, minlength=40).astype(float)
            nu = np.bincount(bu, minlength=40).astype(float)
            w_u = np.where(nl[bl] > 0, nu[bl] / np.maximum(nl[bl], 1), 0.0)
            x = np.concatenate([L_l[:, m], L_l[:, m]])
            w = np.concatenate([np.ones(len(L_l)), w_u])
            proj = float(wmedian(x, w))
            tm = float(np.median(allT[:, m]))
            o[name] = {"raw_median_labelled": float(np.median(L_t[:, m])), "raw_median_unlabelled": float(np.median(U_t[:, m])),
                       "raw_median_full_bank": tm, "label_median_labelled": float(np.median(L_l[:, m])),
                       "ratio_labelled_rows_MEASURED": float(np.median(L_l[:, m]) / np.median(L_t[:, m])),
                       "label_median_full_bank_PROJECTED": proj, "ratio_full_bank_PROJECTED": proj / tm}
        out[f"rank{rk}"] = o
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
