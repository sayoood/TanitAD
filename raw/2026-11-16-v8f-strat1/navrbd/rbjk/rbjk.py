"""rbjk.py -- RB-JOINT-K (card PREREG_V8F_NAVRBD.md sec 0a items 4-5, row RBJK, sec 7 RBJK reading).
Copy of the HGB half of rbjoint.py (windows, folds, tree, feature sets, paired bootstrap unchanged); ONLY the target changes:
the 3-class kinematic lat label (read_otband.labels via obd_geom.build().labels) of the GT 6 s track of each window.
GT track builder = D.gt_future(cache pose_grid)[t]  (the ClipStore's d["gt"][t0] = dump "g").
Stages: python rbjk.py proof | cv.   CPU only.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
for p in (S1 + "/rbdist", S1 + "/rbdist/src", S1 + "/rbvis", S1 + "/read", S1 + "/read/theory/obd/read", S1 + "/read/theory/otband/read",
          S1 + "/read/tree/stack", S1 + "/navrb2/read/tree/taniteval"):
    sys.path.insert(0, p)
import rbdist_zt as RD  # noqa: E402
import rbvis as RV  # noqa: E402
import strat1_read as S  # noqa: E402
import obd_geom as G  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import ci as CI  # noqa: E402

MARGIN = 0.0383
NFOLD = 5
RJ = S1 + "/rbjoint/out"
CL = ("lane_keep", "turn_left", "turn_right")


def fold_of(clip_id, salt="rbjoint-fold", k=NFOLD):
    return int(hashlib.sha256((salt + "|" + str(clip_id)).encode()).hexdigest(), 16) % k


def shuffle_labels(y, seed):
    return np.asarray(y)[np.random.RandomState(seed).permutation(len(y))]


def pose_grid(cache, clip):
    import torch
    o = torch.load(RV.clip_path(cache, clip), map_location="cpu", weights_only=True, mmap=True)
    return o["pose_grid"].numpy().astype(np.float64)


def gt_track(pg, t):
    """GT ego-frame 6 s track [30,2] of cache step t, and ok flag -- exactly ClipStore.get(clip)['gt'][t], ['gt_ok'][t]"""
    gt, ok = D.gt_future(pg)
    return gt[t], ok[t]


def proof(out):
    geom = G.build()
    d = S1 + "/read/t1/dump_STRAT1-C0_s0_m10_causal_heads"
    dg = S.load_dump_n(d, 141)
    clips = np.load(S1 + "/read/inputs/join_eval141.npz", allow_pickle=False)["clip"].astype(str)
    n = len(dg["eid"])
    gl_dump, _ = geom.labels(dg["g"])
    mine = np.zeros((n, 30, 2), np.float64)
    oks = np.zeros(n, bool)
    cache = {}
    for i in range(n):
        c = str(clips[dg["clip_index"][i]])
        if c not in cache:
            cache[c] = D.gt_future(pose_grid(RV.EVAL_CACHE, c))
        mine[i], oks[i] = cache[c][0][int(dg["ws"][i])], cache[c][1][int(dg["ws"][i])]
    gl_mine, _ = geom.labels(mine)
    maxdiff = float(np.abs(mine - np.asarray(dg["g"], np.float64)).max())
    res = {"n_windows": int(n), "n_episodes": int(len(np.unique(dg["eid"]))), "all_gt_ok": bool(oks.all()),
           "label_agree": int((gl_mine == gl_dump).sum()), "g_max_abs_diff_m": maxdiff, "g_exact_equal": bool(np.array_equal(mine.astype(np.float32), np.asarray(dg["g"], np.float32))),
           "class_counts_dump": np.bincount(gl_dump, minlength=3).tolist(), "tanitad_file": D.__file__, "dump": d,
           "pass": bool((gl_mine == gl_dump).all() and oks.all())}
    json.dump(res, open(out + "/PROOF_RBJK.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def cv(out):
    t0 = time.time()
    geom = G.build()
    M = np.load(RJ + "/feat/meta.npz", allow_pickle=False)
    clip, sidx, t_all = M["tr_clip"], M["sidx"], M["tr_t"]
    y5, X_all, ids = M["tr_y"], M["tr_X"], M["tr_clip_ids"]
    N_all = len(clip)
    # labels of the GT track at every window
    glab = np.full(N_all, -1, np.int64)
    okm = np.zeros(N_all, bool)
    for c in np.unique(clip):
        rows = np.where(clip == c)[0]
        pg = pose_grid(RV.TRAIN_CACHE, str(ids[c]))
        gt, ok = D.gt_future(pg)
        tt = t_all[rows]
        good = rows[ok[tt]]
        okm[good] = True
        if len(good):
            glab[good] = geom.labels(gt[t_all[good]].astype(np.float64))[0]
    print("labels built", f"{time.time() - t0:.0f}s", flush=True)
    N_v = len(sidx)
    in_s = np.zeros(N_all, bool)
    in_s[sidx] = True
    # RB-JOINT's own population for reference
    pop = {"all_windows_t_ge3": int(N_all), "strided_windows": int(N_v), "strided_gt_ok": int(okm[sidx].sum()), "strided_dropped_no_6s_future": int((~okm[sidx]).sum()),
           "all_gt_ok": int(okm.sum())}
    fold_all = np.array([fold_of(ids[c]) for c in clip])
    oof = {n: np.full(N_all, -1, np.int64) for n in ("S", "ED", "SHUF_ED")}
    S_const = []
    for k in range(NFOLD):
        trm = (fold_all != k) & okm
        tem = (fold_all == k) & in_s & okm
        ytr = glab[trm]
        for n, cols, yy in (("S", RD.FEATS["S"], ytr), ("ED", RD.FEATS["ED"], ytr), ("SHUF_ED", RD.FEATS["ED"], shuffle_labels(ytr, k))):
            oof[n][tem] = RD.fit_predict(X_all[trm], yy, X_all[tem], cols)
        S_const.append({"fold": k, "train_majority": int(np.bincount(ytr, minlength=3).argmax()), "S_pred_unique": sorted(set(oof["S"][tem].tolist())), "n_fit": int(trm.sum()), "n_test": int(tem.sum())})
        print("fold", k, S_const[-1], f"{time.time() - t0:.0f}s", flush=True)
    ev = in_s & okm
    yv, cv_ = glab[ev], clip[ev]
    pred = {n: oof[n][ev] for n in oof}
    assert all((v >= 0).all() for v in pred.values())
    corr = {n: (p == yv).astype(np.float64) for n, p in pred.items()}
    arms = {n: {"accuracy": RD.level(CI, corr[n], cv_), "pred_counts": np.bincount(pred[n], minlength=3).tolist()} for n in pred}
    P = {"ED-S": RD.paired(CI, corr["ED"], corr["S"], cv_), "SHUF_ED-S": RD.paired(CI, corr["SHUF_ED"], corr["S"], cv_)}
    shares = np.bincount(yv, minlength=3) / len(yv)
    maj = int(shares.argmax())
    d = P["ED-S"]
    reading = ("RBJK-CARRIES" if (d["separated"] and d["delta"] >= MARGIN) else "RBJK-WEAK" if (d["separated"] and d["delta"] > 0 and d["lo"] > 0) else "RBJK-SILENT")
    ctrl = bool(P["SHUF_ED-S"]["delta"] < MARGIN)
    rj = json.load(open(RJ + "/RESULT_RBJOINT.json"))
    res = {"tool": "rbjk.py (copy of rbjoint.py HGB half; target changed only)", "card": "PREREG_V8F_NAVRBD.md sha256 881d7f3d, row RBJK, sec 7",
           "evidence_class": "MEASURED (ours; cached features table, HGB tree, 5-fold clip-grouped CV, CPU, zero model training)",
           "tier": "T0 decodability (NOT driving performance)",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap on per-window correctness, B=2000, seed 0, episodes = train clips; answers 'another draw of EPISODES' only",
           "target": "3-class kinematic lat label (read_otband.labels via obd_geom) of D.gt_future(train cache pose_grid)[t], the 6 s GT track",
           "margin": MARGIN, "population": pop, "n_windows": int(ev.sum()), "n_episodes": int(len(set(cv_))),
           "class_shares": {c: float(s) for c, s in zip(CL, shares)}, "majority_class": CL[maj], "majority_rate": float(shares[maj]),
           "arms": arms, "paired": P, "reading": reading, "control_SHUF_ED_minus_S_below_margin": "PASS" if ctrl else "FAIL", "folds": S_const,
           "rbjoint_lat_cls_reference": {"ED-S": rj["paired"]["ED-S"], "S_acc": rj["arms"]["S"]["accuracy"]["point"], "ED_acc": rj["arms"]["ED"]["accuracy"]["point"],
                                         "SHUF_ED-S_delta": rj["validity"]["R-SHUF-TRAIN"]["SHUF_ED"]["delta"], "majority_share_5way": rj["validity"]["K-CONST"]["oof_share_of_majority"], "n_windows": rj["n_windows_oof"]},
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(res, open(out + "/RESULT_RBJK.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("n_windows", "n_episodes", "class_shares", "majority_rate", "paired", "reading", "control_SHUF_ED_minus_S_below_margin", "population")}, indent=1))
    return res


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    st = sys.argv[1]
    if st == "proof":
        proof(HERE)
    else:
        try:
            cv(HERE)
            open(HERE + "/RBJK.done", "w").write("DONE\n")
        except BaseException:
            import traceback
            open(HERE + "/RBJK.done", "w").write("FAIL\n" + traceback.format_exc())
            raise
