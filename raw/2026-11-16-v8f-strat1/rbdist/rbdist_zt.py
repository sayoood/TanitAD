"""rbdist_zt.py -- does the route text carry the roundabout exit SIDE?  Zero-training, CPU.  Card: PREREG_RBDIST_ZT.md (written before any number).

Tier T0, nav-only decodability (never driving performance).  Features are a pure function of the table arrays and (r, o): rows <= r of the clip.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
sys.path.insert(0, HERE + "/src")
import navtok_src as NT  # noqa: E402  (frozen grammar, sha256 ed7e290d...)

IGNORE = -100
MARGIN = 0.0383
NB, SEED = 2000, 0
HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, class_weight=None, random_state=0, early_stopping=False)
PIN_TABLE = "2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992"
FEATS = {"S": [0], "ED": [0, 1, 2, 3, 4], "E": [0, 1, 2], "D": [0, 3, 4]}   # columns of the feature matrix
RB_PREFIX = ("RB_", "ROTARY")
EST = ("taniteval.ci.paired_episode_cluster_bootstrap on per-window correctness indicators, B=2000, seed 0, episodes of the RB windows; "
       "answers 'another draw of EPISODES' only")
TIER = "T0 nav-only decodability (NOT driving performance); route supplied along the driven path"


# ------------------------------------------------------------------ pure feature rule (table rows <= r only; no label argument)
def rb_state(cls_is_rb, flag, r):
    return bool(flag[r] == 0 and cls_is_rb[r])


def exit_last(ex, flag, r, o, _shift=0):
    """latest row at or before r (+_shift: ONLY for the mutant test) in the clip with flag 0 and an exit number; 0 if none"""
    for k in range(r + _shift, o - 1, -1):
        if k < len(ex) and flag[k] == 0 and ex[k] >= 0:
            return int(ex[k])
    return 0


def features(cls_is_rb, flag, ex, dist, r, o, _shift=0):
    """-> [state, exit_cur, exit_last, dist/100 clipped [0,3], dist_valid] at row r of the clip starting at o.  _shift is for the mutant test only."""
    q = r + _shift
    q = min(q, len(flag) - 1)
    st = float(rb_state(cls_is_rb, flag, q))
    ec = float(ex[q]) if (flag[q] == 0 and ex[q] >= 0) else 0.0
    el = float(exit_last(ex, flag, r, o, _shift))
    d = dist[q]
    ok = bool(np.isfinite(d))
    return [st, ec, el, float(np.clip(d / 100.0, 0.0, 3.0)) if ok else 0.0, float(ok)]


def feature_matrix(cls_is_rb, flag, ex, dist, rows, offs, _shift=0):
    return np.array([features(cls_is_rb, flag, ex, dist, int(r), int(o), _shift) for r, o in zip(rows, offs)], np.float64)


# ------------------------------------------------------------------ rules (pure, mutable)
def label_rule(has, lat_cls):
    has = np.asarray(has).astype(bool)
    lat = np.asarray(lat_cls).astype(np.int64)
    return np.where(has & (lat >= 0), lat, IGNORE)


def verdict_rule(gain, separated, margin=MARGIN):
    if gain >= margin and separated:
        return "TEXT-CARRIES-SIDE"
    return "TEXT-SILENT"


def suggestive_unseparated(gain, separated, margin=MARGIN):
    return bool(gain >= margin and not separated)


def make_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**HGB)


def fit_predict(Xtr, ytr, Xte, cols):
    clf = make_clf().fit(Xtr[:, cols], ytr)
    return clf.classes_[clf.predict_proba(Xte[:, cols]).argmax(1)]


def macro_f1(y, p, K=5):
    f = []
    for k in range(K):
        tp = float(((y == k) & (p == k)).sum()); fp = float(((y != k) & (p == k)).sum()); fn = float(((y == k) & (p != k)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def paired(CI, a, b, eid, nb=NB):
    r = CI.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid), n_boot=nb, seed=SEED)
    return {"delta": float(np.mean(a) - np.mean(b)), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(r["n_windows"]), "n_episodes": int(r["n_episodes"])}


def level(CI, a, eid, nb=NB):
    r = CI.episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(eid), reduce="mean", n_boot=nb, seed=SEED)
    return {"point": float(np.mean(a)), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(r["n_windows"]), "n_episodes": int(r["n_episodes"])}


# ------------------------------------------------------------------ data
def text_arrays(texts_path):
    texts = json.load(open(texts_path, encoding="utf-8"))
    P = [NT.parse_full(t) for t in texts]
    return np.array([-1 if p.exit_n is None else p.exit_n for p in P], np.int32)


def split_arrays(tab, split, ex_t):
    vocab = [str(s) for s in tab["vocab"]]
    isrb_v = np.array([c.startswith(RB_PREFIX) for c in vocab])
    cid, ti = np.asarray(tab[f"{split}_cls_id"]), np.asarray(tab[f"{split}_text_i"])
    is_rb = np.where(cid >= 0, isrb_v[np.clip(cid, 0, None)], False)
    ex = np.where(ti >= 0, ex_t[np.clip(ti, 0, None)], -1).astype(np.int32)
    return is_rb, np.asarray(tab[f"{split}_flag"]), ex, np.asarray(tab[f"{split}_dist_m"], np.float64)


def run(out_dir, nb=NB, log=print):
    sys.path.insert(0, S1 + "/navrb2/read/tree/taniteval")
    from taniteval import ci as CI
    t0 = time.time()
    tpath, jt, je = S1 + "/read/inputs/navtok_table.npz", S1 + "/ckpt120build/in/join_train.npz", S1 + "/read/inputs/join_eval141.npz"
    assert sha(tpath) == PIN_TABLE, "navtok_table sha mismatch"
    tab = np.load(tpath, allow_pickle=False)
    JT, JE = dict(np.load(jt, allow_pickle=False)), dict(np.load(je, allow_pickle=False))
    for k in ("clip", "off", "n"):
        assert np.array_equal(tab["train_" + k], JT[k]) and np.array_equal(tab["eval_" + k], JE[k]), f"table/join mismatch on {k}"
    tr_clips, ev_clips = set(map(str, JT["clip"])), set(map(str, JE["clip"]))
    assert not (tr_clips & ev_clips), "train and eval clips overlap"
    ex_t = text_arrays(HERE + "/src/texts.json")
    # ---- train RB windows
    rb_t, fl_t, ex_tr, di_tr = split_arrays(tab, "train", ex_t)
    off_t = np.repeat(np.asarray(JT["off"]), np.asarray(JT["n"]))
    ytr_all = label_rule(JT["has"], JT["lat_cls"])
    rows_t = np.where((fl_t == 0) & rb_t & (ytr_all != IGNORE))[0]
    Xtr = feature_matrix(rb_t, fl_t, ex_tr, di_tr, rows_t, off_t[rows_t])
    ytr = ytr_all[rows_t]
    assert (Xtr[:, 0] == 1).all()
    ep_t = np.repeat(np.arange(len(JT["n"])), np.asarray(JT["n"]))[rows_t]
    # ---- eval windows
    d = S1 + "/read/t1/dump_STRAT1-C0_s0_m10_causal_heads"
    zs = [np.load(os.path.join(d, f"ep{i:04d}.npz")) for i in range(141)]
    eid = np.concatenate([z["eid"] for z in zs]); cidx = np.concatenate([z["clip_index"] for z in zs]); ws = np.concatenate([z["ws"] for z in zs])
    rows_e = np.asarray(JE["off"], np.int64)[cidx] + ws.astype(np.int64)
    rb_e, fl_e, ex_e, di_e = split_arrays(tab, "eval", ex_t)
    off_e = np.asarray(JE["off"], np.int64)[cidx]
    state_e = np.array([rb_state(rb_e, fl_e, int(r)) for r in rows_e])
    yall = label_rule(JE["has"][rows_e], JE["lat_cls"][rows_e])
    assert int(state_e.sum()) == 235 and len(set(eid[state_e])) == 16, f"RB windows {int(state_e.sum())} / {len(set(eid[state_e]))} episodes, expected 235 / 16"
    sel = state_e & (yall != IGNORE)
    Xe = feature_matrix(rb_e, fl_e, ex_e, di_e, rows_e[sel], off_e[sel])
    ye, ee = yall[sel], eid[sel]
    res = {"tool": "rbdist_zt.py", "card": "PREREG_RBDIST_ZT.md", "evidence_class": "MEASURED (ours; pinned navtok table + frozen navtok grammar, zero training, CPU)",
           "tier": TIER, "estimator": EST, "margin": MARGIN, "hgb": HGB,
           "inputs_sha256": {"navtok_table.npz": PIN_TABLE, "join_train.npz": sha(jt), "join_eval141.npz": sha(je), "texts.json": sha(HERE + "/src/texts.json"),
                             "navtok_src.py": sha(HERE + "/src/navtok_src.py"), "PREREG_RBDIST_ZT.md": sha(HERE + "/PREREG_RBDIST_ZT.md")},
           "train": {"rb_windows_labelled": int(len(ytr)), "episodes": int(len(set(ep_t))), "label_counts": np.bincount(ytr, minlength=5).tolist(),
                     "exit_cur_counts": {str(k): int((Xtr[:, 1] == k).sum()) for k in sorted(set(Xtr[:, 1]))}},
           "eval": {"rb_windows": int(state_e.sum()), "rb_episodes": int(len(set(eid[state_e]))), "scored_windows": int(len(ye)), "scored_episodes": int(len(set(ee))),
                    "label_counts": np.bincount(ye, minlength=5).tolist(), "exit_cur_counts": {str(k): int((Xe[:, 1] == k).sum()) for k in sorted(set(Xe[:, 1]))},
                    "exit_last_counts": {str(k): int((Xe[:, 2] == k).sum()) for k in sorted(set(Xe[:, 2]))}},
           "clips_disjoint": True}
    # ---- arms
    pred = {n: fit_predict(Xtr, ytr, Xe, c) for n, c in FEATS.items()}
    corr = {n: (p == ye).astype(np.float64) for n, p in pred.items()}
    maj = int(np.bincount(ytr, minlength=5).argmax())
    probs = []
    # controls
    ctrl = {"K-CONST": {"train_majority_class": maj, "S_pred_unique": sorted(set(pred["S"].tolist())), "S_accuracy": float(corr["S"].mean()),
                        "eval_share_of_majority_class": float((ye == maj).mean())}}
    ctrl["K-CONST"]["pass"] = bool(set(pred["S"].tolist()) == {maj} and ctrl["K-CONST"]["S_accuracy"] == ctrl["K-CONST"]["eval_share_of_majority_class"])
    oh = lambda y: np.eye(5)[np.clip(y, 0, 4)]  # noqa: E731
    pr = fit_predict(oh(ytr), ytr, oh(ye), [0, 1, 2, 3, 4])
    ctrl["K-REACH"] = {"label_as_input_accuracy": float((pr == ye).mean()), "pass": bool((pr == ye).mean() == 1.0)}
    # arms table
    arms = {}
    for n in FEATS:
        arms[n] = {"accuracy": level(CI, corr[n], ee, nb), "macro_f1": macro_f1(ye, pred[n]), "pred_counts": np.bincount(pred[n], minlength=5).tolist()}
        if n != "S":
            arms[n]["gain_over_S_paired"] = paired(CI, corr[n], corr["S"], ee, nb)
    turn = np.isin(ye, (1, 2))
    for n in FEATS:
        t = turn & np.isin(pred[n], (1, 2))
        arms[n]["turn_windows"] = {"n": int(turn.sum()), "episodes": int(len(set(ee[turn]))), "turn_recall": float(np.isin(pred[n][turn], (1, 2)).mean()) if turn.any() else None,
                                   "side_correct_of_turn_windows": float((pred[n][turn] == ye[turn]).mean()) if turn.any() else None}
    # crosstab (descriptive): eval label by exit_cur
    res["eval_label_by_exit_cur"] = {str(int(k)): np.bincount(ye[Xe[:, 1] == k], minlength=5).tolist() for k in sorted(set(Xe[:, 1]))}
    res["train_label_by_exit_cur"] = {str(int(k)): np.bincount(ytr[Xtr[:, 1] == k], minlength=5).tolist() for k in sorted(set(Xtr[:, 1]))}
    # regression arms
    cols_f = [1, 2, 3, 4]
    reg_eval = []
    for s in range(10):
        Xs = Xe.copy(); Xs[:, cols_f] = Xe[np.random.default_rng(s).permutation(len(Xe))][:, cols_f]
        p = fit_predict(Xtr, ytr, Xs, FEATS["ED"])
        g = paired(CI, (p == ye).astype(float), corr["S"], ee, nb)
        reg_eval.append({"seed": s, "accuracy": float((p == ye).mean()), "gain_over_S": g["delta"], "lo": g["lo"], "hi": g["hi"], "separated": g["separated"],
                         "would_satisfy_rule": verdict_rule(g["delta"], g["separated"]) == "TEXT-CARRIES-SIDE"})
    reg_train = []
    for s in range(5):
        Xs = Xtr.copy(); Xs[:, cols_f] = Xtr[np.random.default_rng(s).permutation(len(Xtr))][:, cols_f]
        p = fit_predict(Xs, ytr, Xe, FEATS["ED"])
        g = paired(CI, (p == ye).astype(float), corr["S"], ee, nb)
        reg_train.append({"seed": s, "accuracy": float((p == ye).mean()), "gain_over_S": g["delta"], "lo": g["lo"], "hi": g["hi"], "separated": g["separated"],
                          "would_satisfy_rule": verdict_rule(g["delta"], g["separated"]) == "TEXT-CARRIES-SIDE"})
    me, mt = float(np.mean([r["gain_over_S"] for r in reg_eval])), float(np.mean([r["gain_over_S"] for r in reg_train]))
    ctrl["R-SHUF-EVAL"] = {"runs": reg_eval, "mean_gain_over_S": me, "n_would_satisfy_rule": int(sum(r["would_satisfy_rule"] for r in reg_eval)), "pass": bool(me < MARGIN)}
    ctrl["R-SHUF-TRAIN"] = {"runs": reg_train, "mean_gain_over_S": mt, "n_would_satisfy_rule": int(sum(r["would_satisfy_rule"] for r in reg_train)), "pass": bool(mt < MARGIN)}
    for k in ("K-CONST", "K-REACH", "R-SHUF-EVAL", "R-SHUF-TRAIN"):
        if not ctrl[k]["pass"]:
            probs.append(f"{k} did not read as committed")
    g = arms["ED"]["gain_over_S_paired"]
    verdict = verdict_rule(g["delta"], g["separated"])
    res.update({"arms": arms, "controls": ctrl, "problems": probs,
                "reading": {"rule": f"TEXT-CARRIES-SIDE iff accuracy(ED) - accuracy(S) >= {MARGIN} and CI separated (lo > 0); else TEXT-SILENT", "gain": g,
                            "verdict": "INVALID (control problem)" if probs else verdict, "rule_verdict_if_controls_held": verdict,
                            "suggestive_unseparated": suggestive_unseparated(g["delta"], g["separated"])},
                "elapsed_s": round(time.time() - t0, 1)})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RESULT_RBDIST_ZT.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    r = run(a.out, a.n_boot)
    print(json.dumps({"verdict": r["reading"]["verdict"], "problems": r["problems"], "gain": r["reading"]["gain"], "elapsed_s": r["elapsed_s"]}))
