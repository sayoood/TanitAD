"""Task 2 report (parse coverage) and task 4 (agreement of the trainer's nav token with the parsed nav text).  CPU, numpy only.
Writes navtok_report.json, agreement.json, disagreements.csv.  Inputs: raw/*.npz, navtok_table.npz (build_table.py)."""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navtok as N
import build_table as B

COARSE_NAMES = ["FOLLOW", "TURN_L", "TURN_R", "UTURN", "UNPARSED"]


def parse_report():
    texts = json.load(open(os.path.join(HERE, "raw", "texts.json"), encoding="utf-8"))
    P = [N.parse_full(t) for t in texts]
    un = [p.raw for p in P if p.cls == "UNPARSED"]
    byc = collections.Counter(p.cls for p in P)
    nodist = sum(p.dist_m is None for p in P)
    rep = {"n_distinct_texts": len(texts), "n_unparsed": len(un), "unparsed_fraction": len(un) / len(texts), "unparsed_texts": un,
           "n_classes_used": len(byc), "n_classes_in_vocab": len(N.VOCAB), "distinct_texts_per_class": dict(sorted(byc.items())),
           "distinct_texts_without_distance": nodist,
           "classes_in_vocab_never_seen": [c for c in N.VOCAB if c != "UNPARSED" and c not in byc]}
    json.dump(rep, open(os.path.join(HERE, "navtok_report.json"), "w"), indent=1)
    return rep


def coarse_of(cls_id, vocab):
    lut = np.array([N.coarse(c) for c in vocab], np.int8)
    out = np.full(cls_id.shape, -9, np.int8)
    ok = cls_id >= 0
    out[ok] = lut[cls_id[ok]]
    return out


def agree_split(split, J, T, vocab, t0_us=B.T0_US, nav=None):
    tv = B.trainer_view(J)
    if nav is not None:
        T = B.build(J, nav, t0_us)
    sel = tv["has"] & (T["flag"] == 0)
    tok = tv["tok"][sel].astype(int)
    cid = T["cls_id"][sel].astype(int)
    d = T["dist_m"][sel]
    co = coarse_of(cid, vocab).astype(int)             # -1 UTURN, 0 FOLLOW, 1 L, 2 R
    col = np.where(co == -1, 3, co)                     # 0 FOLLOW 1 L 2 R 3 UTURN
    return sel, tok, cid, d, col, T, tv


def confusion(tok, col, ncol=4):
    M = np.zeros((3, ncol), np.int64)
    np.add.at(M, (tok, col), 1)
    return M


def main():
    rep = {"parse": parse_report()}
    nav = B.load_nav()
    Z = np.load(os.path.join(HERE, "navtok_table.npz"))
    vocab = [str(v) for v in Z["vocab"]]
    out = {"t0_us": B.T0_US}
    rows_csv = []
    for split, jf in (("train", "join_train.npz"), ("eval", "join_eval141.npz")):
        J = dict(np.load(os.path.join(HERE, "raw", jf)))
        T = {k: Z[f"{split}_{k}"] for k in ("cls_id", "dist_m", "flag", "text_i", "age_s")}
        sel, tok, cid, d, col, T, tv = agree_split(split, J, T, vocab)
        n = int(sel.sum())
        M = confusion(tok, col)
        o = {"n_windows": n, "rows_trainer_token": ["FOLLOW", "TURN_L", "TURN_R"], "cols_text_coarse": ["FOLLOW", "TURN_L", "TURN_R", "UTURN"],
             "confusion_n": M.tolist(), "trainer_token_counts": np.bincount(tok, minlength=3).tolist(),
             "text_coarse_counts": np.bincount(col, minlength=4).tolist()}
        # full-class confusion: trainer token x text class (only non-zero columns)
        F = np.zeros((3, len(vocab)), np.int64)
        np.add.at(F, (tok, cid), 1)
        o["full_class_confusion_n"] = {vocab[c]: F[:, c].tolist() for c in range(len(vocab)) if F[:, c].sum()}
        # text turn within X m (turn classes only, any degree) vs trainer token
        o["text_turn_within_X"] = {}
        for X in (30, 60, 100, 150, 250):
            tcol = np.where((col == 1) & (d <= X), 1, np.where((col == 2) & (d <= X), 2, 0))
            Mx = confusion(tok, tcol, 3)
            o["text_turn_within_X"][str(X)] = {"confusion_n": Mx.tolist(), "agree": float(np.trace(Mx) / Mx.sum()),
                                               "trainer_turn_text_same_side": float(((Mx[1, 1] + Mx[2, 2]) / max(Mx[1:].sum(), 1)))}
        # distance distribution where the trainer says turn
        q = lambda a: {str(p): (None if not len(a) else float(np.nanpercentile(a, p))) for p in (5, 25, 50, 75, 95)}
        for name, m in (("trainer_turn_L", tok == 1), ("trainer_turn_R", tok == 2)):
            side = 1 if name.endswith("L") else 2
            mm = m & (col == side)
            o[f"dist_text_where_{name}_and_text_same_side_turn"] = {"n": int(mm.sum()), "quantiles_m": q(d[mm])}
            o[f"text_class_where_{name}"] = dict(collections.Counter(vocab[c] for c in cid[m]).most_common(12))
            o[f"text_dist_where_{name}_text_not_turn_same_side"] = {"n": int((m & ~(col == side)).sum()),
                                                                  "n_dist_none": int((m & ~(col == side) & np.isnan(d)).sum())}
        # trainer's own distance vs text distance, same-side turns, both valid
        td = tv["d"][sel]
        for nm, side in (("L", 1), ("R", 2)):
            mm = (tok == side) & (col == side) & np.isfinite(td) & np.isfinite(d)
            diff = (d[mm] - td[mm])
            o[f"text_minus_trainer_dist_m_same_side_{nm}"] = {"n": int(mm.sum()), "quantiles_m": q(diff),
                                                              "frac_abs_le_30": float((np.abs(diff) <= 30).mean()) if mm.any() else None}
        # trainer turn-side vs text opposite side
        o["opposite_side"] = {"trainer_L_text_R": int(M[1, 2]), "trainer_R_text_L": int(M[2, 1])}
        # trainer FOLLOW but text announces a turn within 30 m / trainer turn but the text says FOLLOW class
        o["trainer_follow_text_turn_le30m"] = int(((tok == 0) & np.isin(col, (1, 2)) & (d <= 30)).sum())
        o["trainer_turn_text_follow"] = int(((tok > 0) & (col == 0)).sum())
        # which text FOLLOW classes accompany a trainer turn
        out[split] = o
        # disagreement examples
        rng = np.random.default_rng(0)
        idx = np.flatnonzero(sel)
        steps_clip = np.repeat(np.arange(len(J["n"])), J["n"])
        cats = {"trainer_turn__text_follow": (tok > 0) & (col == 0), "trainer_turn__text_opposite": ((tok == 1) & (col == 2)) | ((tok == 2) & (col == 1)),
                "trainer_follow__text_turn_le30m": (tok == 0) & np.isin(col, (1, 2)) & (d <= 30)}
        for cat, m in cats.items():
            w = np.flatnonzero(m)
            for i in rng.choice(w, size=min(60, len(w)), replace=False) if len(w) else []:
                g = idx[i]
                c = steps_clip[g]
                rows_csv.append([split, cat, str(J["clip"][c]), int(g - J["off"][c]), round(float(J["now_s"][g]), 3), ["FOLLOW", "TURN_L", "TURN_R"][tok[i]],
                                 None if np.isnan(tv["d"][g]) else round(float(tv["d"][g]), 1), nav["texts"][int(T["text_i"][g])],
                                 round(float(T["age_s"][g]), 3)])
        # T0 scan: agreement (trainer turn side == text turn side within 60 m, vs everything) for T0 in [-2, +1] s
        if split == "train":
            scan = {}
            for t0 in range(-2000000, 1000001, 100000):
                s2, tok2, cid2, d2, col2, _, _ = agree_split(split, J, None, vocab, t0, nav)
                tc = np.where((col2 == 1) & (d2 <= 60), 1, np.where((col2 == 2) & (d2 <= 60), 2, 0))
                Mx = confusion(tok2, tc, 3)
                scan[str(t0)] = {"agree": round(float(np.trace(Mx) / Mx.sum()), 5), "turn_turn_same_side": int(Mx[1, 1] + Mx[2, 2]),
                                 "opposite": int(Mx[1, 2] + Mx[2, 1])}
            out["t0_scan_train_text_turn_within_60m"] = scan
            best = max(scan.items(), key=lambda kv: kv[1]["turn_turn_same_side"])
            out["t0_scan_best"] = {"t0_us": int(best[0]), **best[1]}
    json.dump(out, open(os.path.join(HERE, "agreement.json"), "w"), indent=1)
    with open(os.path.join(HERE, "disagreements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "category", "clip", "step_j", "now_s", "trainer_token", "trainer_d_next_m", "nav_text", "text_row_age_s"])
        w.writerows(rows_csv)
    print(json.dumps(out["train"], indent=0)[:3500])
    print(json.dumps(out.get("t0_scan_best")))
    print(rep["parse"]["n_unparsed"], rep["parse"]["classes_in_vocab_never_seen"])


if __name__ == "__main__":
    main()
