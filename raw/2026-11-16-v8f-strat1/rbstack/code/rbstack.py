"""rbstack.py -- RBSTACK: the stacking-combiner check (card PREREG_RBSTACK.md, sha256 bf4d5395, frozen before any number).
Does the RB-JOINT RAW8 vision probe's NESTED out-of-fold class probabilities, appended to the RBJK ED tree's features, decode the 3-class
kinematic lat label better than the ED tree alone?  Tier T0 decodability (NOT driving performance).  CPU only, zero training of the driving model.
Reuses rbjk.py (population, label, folds, HGB, bootstrap) and rbjoint.py (RAW8 probe: make_net / fit_arm).  Neither is edited.
Run:  python rbstack.py --out DIR [--smoke]     sentinel = last line of <out>/rbstack_log.txt:  'ZZ RBSTACK DONE ZZ' or 'ZZ RBSTACK FAIL ZZ <reason>'
Deviation notes (see HANDOFF): probe features exist only for RB-JOINT's strided windows, so the EDV / EDV_SHUF / ED_STR trees fit on strided
labelled fold-train windows; the ED / S reference arms stay dense (RBJK, K-REPRO).  The 10 unordered inner fits serve both ordered pairs (j,k),(k,j).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
sys.path.insert(0, S1 + "/navrbd/rbjk")
sys.path.insert(0, S1 + "/rbjoint")
import rbjk as K  # noqa: E402  (also sets the rbdist / rbvis / tanitad / taniteval import paths)
import rbjoint as J  # noqa: E402

RD, RV, CI, D, G = K.RD, K.RV, K.CI, K.D, K.G
CARD = HERE + "/PREREG_RBSTACK.md"
CARD_SHA = "bf4d5395bb481cbeaca552bd66a987071e772655b11f8a1d1a8bda2e22c72901"
MARGIN = 0.0383
NFOLD = 5
MAJ_RATE = 0.4075357297531399
RJ_META = S1 + "/rbjoint/out/feat/meta.npz"
RBJK_RESULT = S1 + "/navrbd/rbjk/RESULT_RBJK.json"
SHUF_SEED = 0
FREE_F_GB = 15.0
FREE_COMMIT_GB = 5.5


class NestingError(AssertionError):
    pass


# ------------------------------------------------------------------ pure rules (each a mutation target of the tests)
def plan_fits(nfold=NFOLD):
    """The nested fit plan.  outer k: fit on every fold but k, predict fold k.  inner {j,k}: fit on the folds other than j and k, predict j and k.
    (The ordered pairs (j,k) and (k,j) name the SAME fit set, so one fit serves both: 5 + 10 fits per seed.)"""
    fits = []
    for k in range(nfold):
        fits.append({"name": f"outer{k}", "kind": "outer", "fit_folds": tuple(f for f in range(nfold) if f != k), "pred_folds": (k,)})
    for j in range(nfold):
        for k in range(j + 1, nfold):
            fits.append({"name": f"inner{j}{k}", "kind": "inner", "fit_folds": tuple(f for f in range(nfold) if f not in (j, k)), "pred_folds": (j, k)})
    return fits


def plan_plain_oof(nfold=NFOLD):
    """MUTANT: plain out-of-fold probabilities (one probe per fold, fit on all the others) used for every stack row."""
    return [{"name": f"plain{j}", "kind": "plain", "fit_folds": tuple(f for f in range(nfold) if f != j), "pred_folds": (j,)} for j in range(nfold)]


def provenance(plan, nfold=NFOLD):
    """prov[k][j] = (fit_folds actually used, name) of the probe whose probabilities fill rows of fold j in the stack for outer fold k"""
    prov = {}
    for k in range(nfold):
        prov[k] = {}
        for j in range(nfold):
            if j == k:
                c = [p for p in plan if p["kind"] == "outer" and p["pred_folds"] == (k,)]
            else:
                c = [p for p in plan if p["kind"] == "inner" and j in p["pred_folds"] and k in p["pred_folds"]]
            if not c:  # a plan without nested fits (the plain out-of-fold mutant) falls back to whichever probe predicts fold j
                c = [p for p in plan if j in p["pred_folds"]]
            prov[k][j] = (tuple(c[0]["fit_folds"]), c[0]["name"])
    return prov


def assert_nested(prov, nfold=NFOLD):
    """K-NEST.  prov[k][j] = folds the probe saw.  Test fold k: probe saw no row of fold k.  Training fold j != k: the probe saw neither j nor k."""
    for k in range(nfold):
        for j in range(nfold):
            seen = set(prov[k][j][0])
            if j == k:
                if k in seen:
                    raise NestingError(f"outer {k}: test-row probe saw fold {k}")
            elif j in seen or k in seen:
                raise NestingError(f"outer {k}: train fold {j} probabilities from a probe that saw {sorted(seen & {j, k})}")
    return True


def softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def shuffle_rows(P, seed):
    return np.asarray(P)[np.random.RandomState(seed).permutation(len(P))]


def repro_ok(value, ref, tol=1e-9):
    return bool(abs(float(value) - float(ref)) <= tol)


def reading_of(d, margin=MARGIN):
    """one seed's section-3 reading from the paired EDV - ED dict"""
    if d["separated"] and d["lo"] > 0 and d["delta"] >= margin:
        return "STACK-CARRIES"
    if d["separated"] and d["lo"] > 0 and d["delta"] > 0:
        return "STACK-WEAK"
    return "STACK-SILENT"


RANK = {"STACK-CARRIES": 2, "STACK-WEAK": 1, "STACK-SILENT": 0}


def combine_readings(rs):
    """at both seeds; the lower reading holds"""
    return min(rs, key=lambda r: RANK[r])


# ------------------------------------------------------------------ data
def free_gb(path):
    return shutil.disk_usage(path).free / 2 ** 30


def commit_free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def load_meta(out, smoke):
    """RB-JOINT's meta (copied into out/feat so a cleanup of rbjoint/out cannot remove it).  smoke: keep the first 3 clips (with strided rows) per fold."""
    os.makedirs(out + "/feat", exist_ok=True)
    dst = out + "/feat/meta.npz"
    if not os.path.exists(dst):
        shutil.copyfile(RJ_META, dst)
    M = dict(np.load(dst, allow_pickle=False))
    if smoke:
        ids = M["tr_clip_ids"]
        sclip = np.unique(M["tr_clip"][M["sidx"]])
        keep = []
        cnt = {k: 0 for k in range(NFOLD)}
        for c in sclip:
            f = K.fold_of(ids[c])
            if cnt[f] < 3:
                cnt[f] += 1
                keep.append(int(c))
        mask = np.isin(M["tr_clip"], keep)
        newpos = np.cumsum(mask) - 1
        sm = mask[M["sidx"]]
        M = {"tr_clip": M["tr_clip"][mask], "tr_t": M["tr_t"][mask], "tr_y": M["tr_y"][mask], "tr_X": M["tr_X"][mask],
             "sidx": newpos[M["sidx"][sm]], "tr_clip_ids": M["tr_clip_ids"]}
    return M


def build_labels(M, out, smoke):
    """kinematic 3-class label of the GT 6 s track at every window + gt_ok, exactly rbjk.cv"""
    p = out + "/labels.npz"
    if os.path.exists(p):
        z = np.load(p)
        return z["glab"], z["okm"]
    geom = G.build()
    clip, t_all, ids = M["tr_clip"], M["tr_t"], M["tr_clip_ids"]
    glab = np.full(len(clip), -1, np.int64)
    okm = np.zeros(len(clip), bool)
    for c in np.unique(clip):
        rows = np.where(clip == c)[0]
        pg = K.pose_grid(RV.TRAIN_CACHE, str(ids[c]))
        gt, ok = D.gt_future(pg)
        good = rows[ok[t_all[rows]]]
        okm[good] = True
        if len(good):
            glab[good] = geom.labels(gt[t_all[good]].astype(np.float64))[0]
    np.savez(p, glab=glab, okm=okm)
    return glab, okm


def extract_x8(M, out, log):
    """RAW8 input = pool8x20 frames {t-3, t} of each strided window, fp8 bytes (exactly rbjoint.extract's X8; X4 not needed)"""
    import torch
    p, ok = out + "/feat/X8.npy", out + "/feat/X8.ok"
    sidx, N = M["sidx"], len(M["sidx"])
    shp = (N, 2, 160, 1024)
    if os.path.exists(ok) and os.path.exists(p):
        return
    gb = N * 2 * 160 * 1024 / 2 ** 30
    assert free_gb(out) - gb >= FREE_F_GB, f"F: free {free_gb(out):.1f} GB minus {gb:.1f} GB below the {FREE_F_GB} GB floor"
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=shp)
    ids = M["tr_clip_ids"]
    by = {}
    for k, i in enumerate(sidx):
        by.setdefault(int(M["tr_clip"][i]), []).append(k)
    t0 = time.time()
    for n, (c, ks) in enumerate(by.items()):
        z = torch.load(RV.clip_path(RV.TRAIN_CACHE, str(ids[c])), map_location="cpu", mmap=True, weights_only=False)
        for k in ks:
            mm[k] = J.read_window(z["pool8x20"], int(M["tr_t"][sidx[k]]), "X8").view(torch.uint8).numpy()
        if n % 50 == 0:
            log(f"extract X8 clip {n}/{len(by)} {time.time() - t0:.0f}s")
    mm.flush()
    del mm
    open(ok, "w").write("OK\n")
    log(f"extract X8 done {N} windows {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ probe fits (nested)
class Probe:
    """real RAW8 probe: fit_arm of rbjoint.py (6 epochs, its own epoch selection on the inner clip holdout), softmax of the chosen epoch's logits"""

    def __init__(self, F, y, ncls, seed, log):
        self.F, self.y, self.ncls, self.seed, self.log = F, y, ncls, seed, log

    def __call__(self, fit_idx, hold_idx, pred_idx):
        r = J.fit_arm("X8", self.F, self.y, fit_idx, hold_idx, {"p": (self.F, None, pred_idx)}, self.ncls, None, self.seed, self.log)
        return softmax(r["logits"]["p"]), {"E": int(r["E"]), "hold": {str(e): v for e, v in r["hold"].items()}}


def run_fits(plan, fit_fn, y_v, ok_v, fold_v, clip_v, ids, ncls, cache_dir, tag, log):
    """execute every fit of the plan.  fit rows = strided windows of the fit folds with a label; the epoch-selection holdout is the rbjoint inner-clip rule.
    -> fits[name] = {probs [N,ncls] (NaN outside pred folds), fit_folds_actual, E, hold, n_fit}"""
    N = len(y_v)
    inner_hold = np.array([K.fold_of(ids[c], salt="rbjoint-inner", k=5) == 0 for c in clip_v])
    fits = {}
    os.makedirs(cache_dir, exist_ok=True)
    for p in plan:
        cp = f"{cache_dir}/{tag}_{p['name']}.npz"
        fit_rows = np.where(np.isin(fold_v, p["fit_folds"]) & ok_v)[0]
        hold_i, fit_i = fit_rows[inner_hold[fit_rows]], fit_rows[~inner_hold[fit_rows]]
        pred_i = np.where(np.isin(fold_v, p["pred_folds"]))[0]
        if os.path.exists(cp):
            z = np.load(cp, allow_pickle=False)
            probs, meta = z["probs"], json.loads(str(z["meta"]))
        else:
            t0 = time.time()
            log(f"  fit {tag} {p['name']}: fit {len(fit_i)} hold {len(hold_i)} pred {len(pred_i)}")
            pr, meta = fit_fn(fit_i, hold_i, pred_i)
            probs = np.full((N, ncls), np.nan, np.float32)
            probs[pred_i] = pr
            meta["n_fit"], meta["n_hold"], meta["n_pred"] = int(len(fit_i)), int(len(hold_i)), int(len(pred_i))
            np.savez(cp, probs=probs, meta=json.dumps(meta))
            log(f"  fit {tag} {p['name']}: E {meta.get('E')} hold {meta.get('hold')} {time.time() - t0:.0f}s")
        # the folds the probe REALLY saw come from the row indices it was given, not from the plan
        seen = tuple(sorted(set(fold_v[fit_i].tolist()) | set(fold_v[hold_i].tolist())))
        fits[p["name"]] = {"probs": probs, "seen_folds": seen, "meta": meta, "kind": p["kind"], "pred_folds": p["pred_folds"], "fit_folds_plan": p["fit_folds"]}
    return fits


def real_provenance(plan, fits, nfold=NFOLD):
    prov = provenance(plan, nfold)
    return {k: {j: (fits[prov[k][j][1]]["seen_folds"], prov[k][j][1]) for j in range(nfold)} for k in range(nfold)}


def stack_probs(prov, fits, k, nfold=NFOLD, fold_v=None):
    """probability matrix for outer fold k: every row takes the probabilities of the probe named in prov[k][fold of the row]"""
    N, C = next(iter(fits.values()))["probs"].shape
    P = np.full((N, C), np.nan, np.float32)
    for j in range(nfold):
        rows = np.where(fold_v == j)[0]
        P[rows] = fits[prov[k][j][1]]["probs"][rows]
    return P


# ------------------------------------------------------------------ stacking trees
def stack_arms(plan, fits, X_v, y_v, ok_v, fold_v, ncls, log):
    """-> preds {EDV, EDV_SHUF, ED_STR, V} over strided rows (-1 where not scored), plus the nesting provenance (asserted)"""
    prov = real_provenance(plan, fits)
    assert_nested(prov)
    N = len(y_v)
    pred = {n: np.full(N, -1, np.int64) for n in ("EDV", "EDV_SHUF", "ED_STR", "V")}
    nc = X_v.shape[1]
    ed_cols = list(RD.FEATS["ED"])
    for k in range(NFOLD):
        P = stack_probs(prov, fits, k, fold_v=fold_v)
        trm = (fold_v != k) & ok_v
        tem = (fold_v == k) & ok_v
        assert not np.isnan(P[trm | tem]).any()
        Xa = np.hstack([X_v, P]).astype(np.float64)
        cols = ed_cols + list(range(nc, nc + ncls))
        pred["EDV"][tem] = RD.fit_predict(Xa[trm], y_v[trm], Xa[tem], cols)
        Ptr, Pte = shuffle_rows(P[trm], SHUF_SEED), shuffle_rows(P[tem], SHUF_SEED)
        Xs_tr, Xs_te = np.hstack([X_v[trm], Ptr]), np.hstack([X_v[tem], Pte])
        pred["EDV_SHUF"][tem] = RD.fit_predict(Xs_tr, y_v[trm], Xs_te, cols)
        pred["ED_STR"][tem] = RD.fit_predict(X_v[trm].astype(np.float64), y_v[trm], X_v[tem].astype(np.float64), ed_cols)
        pred["V"][tem] = P[tem].argmax(1)
        log(f"  stack fold {k}: n_fit {int(trm.sum())} n_test {int(tem.sum())}")
    return pred, prov


def dense_refs(X_all, y_all, ok_all, fold_all, in_s):
    """ED and S dense reference arms exactly as rbjk.cv (fit on every labelled fold-train window, score the strided held-out windows)"""
    oof = {n: np.full(len(y_all), -1, np.int64) for n in ("S", "ED")}
    for k in range(NFOLD):
        trm = (fold_all != k) & ok_all
        tem = (fold_all == k) & in_s & ok_all
        for n in oof:
            oof[n][tem] = RD.fit_predict(X_all[trm], y_all[trm], X_all[tem], RD.FEATS[n])
    return oof


def score(pred_v, ref_v, y_v, ok_v, clip_v, ncls, n_boot=None):
    """per-arm accuracy and the paired contrasts on the strided labelled windows"""
    ev = ok_v
    yv, cv = y_v[ev], clip_v[ev]
    allp = dict(pred_v)
    allp.update(ref_v)
    for n, p in allp.items():
        assert (p[ev] >= 0).all(), n
    corr = {n: (p[ev] == yv).astype(np.float64) for n, p in allp.items()}
    arms = {n: {"accuracy": RD.level(CI, corr[n], cv), "pred_counts": np.bincount(allp[n][ev], minlength=ncls).tolist()} for n in allp}
    pairs = {"EDV-ED": ("EDV", "ED"), "EDV-ED_STR": ("EDV", "ED_STR"), "EDV_SHUF-ED": ("EDV_SHUF", "ED"), "EDV_SHUF-ED_STR": ("EDV_SHUF", "ED_STR"),
             "V-S": ("V", "S"), "EDV-V": ("EDV", "V"), "ED-S": ("ED", "S")}
    P = {n: RD.paired(CI, corr[a], corr[b], cv) for n, (a, b) in pairs.items()}
    shares = np.bincount(yv, minlength=ncls) / len(yv)
    return {"arms": arms, "paired": P, "class_shares": shares.tolist(), "majority_rate": float(shares.max()), "n_windows": int(ev.sum()), "n_episodes": int(len(set(cv.tolist())))}


def controls(sc, prov, refs_check, ref_x=None):
    """K-NEST, K-SHUF, K-V for one seed (K-REPRO is seed independent and computed once)"""
    kn = True
    try:
        assert_nested(prov)
    except NestingError:
        kn = False
    ks = bool(sc["paired"]["EDV_SHUF-ED"]["delta"] < MARGIN)
    v = sc["paired"]["V-S"]
    kv = bool(sc["arms"]["V"]["accuracy"]["point"] > sc["majority_rate"] and v["separated"] and v["delta"] > 0)
    return {"K-NEST": "PASS" if kn else "FAIL", "K-SHUF": "PASS" if ks else "FAIL", "K-V": "PASS" if kv else "FAIL"}


# ------------------------------------------------------------------ driver
def run_target(tag, seeds, ncls, y_all, ok_all, M, F, out, log, fit_factory=None, refs=None, write_stage=True, plan=None):
    sidx = M["sidx"]
    ids, clip_all = M["tr_clip_ids"], M["tr_clip"]
    fold_all = np.array([K.fold_of(ids[c]) for c in clip_all])
    in_s = np.zeros(len(clip_all), bool)
    in_s[sidx] = True
    y_v, ok_v, fold_v, clip_v, X_v = y_all[sidx], ok_all[sidx], fold_all[sidx], clip_all[sidx], M["tr_X"][sidx]
    plan = plan or plan_fits()
    if refs is None:
        refs = dense_refs(M["tr_X"], y_all, ok_all, fold_all, in_s)
    ref_v = {n: refs[n][sidx] for n in refs}
    results = {}
    for seed in seeds:
        t0 = time.time()
        fit_fn = (fit_factory or (lambda s: Probe(F, np.where(ok_v, y_v, 0).astype(np.int64), ncls, s, log)))(seed)
        fits = run_fits(plan, fit_fn, np.where(ok_v, y_v, 0).astype(np.int64), ok_v, fold_v, clip_v, ids, ncls, out + "/cache", f"{tag}_s{seed}", log)
        pred, prov = stack_arms(plan, fits, X_v, y_v, ok_v, fold_v, ncls, log)
        sc = score(pred, ref_v, y_v, ok_v, clip_v, ncls)
        res = {"tag": tag, "probe_seed": seed, **sc, "reading_this_seed": reading_of(sc["paired"]["EDV-ED"]),
               "controls_seed": controls(sc, prov, None), "provenance_seen_folds": {str(k): {str(j): list(v[0]) for j, v in d.items()} for k, d in prov.items()},
               "probe_fits": {n: {"seen_folds": list(f["seen_folds"]), "pred_folds": list(f["pred_folds"]), **f["meta"]} for n, f in fits.items()},
               "elapsed_s": round(time.time() - t0, 1), "evidence_class": "MEASURED (ours; cached DINOv3 pool8x20 features, RAW8 probe + HGB, CPU)", "tier": "T0 decodability (NOT driving performance)"}
        results[seed] = res
        if write_stage:
            with open(f"{out}/stage_{tag}_s{seed}.json" if tag != "kin3" else f"{out}/stage_s{seed}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        log(f"STAGE {tag} seed {seed} done: EDV-ED {sc['paired']['EDV-ED']['delta']:+.4f} [{sc['paired']['EDV-ED']['lo']:+.4f},{sc['paired']['EDV-ED']['hi']:+.4f}] {res['reading_this_seed']} {res['elapsed_s']:.0f}s")
    return results, refs, fits


def k_repro(refs, sidx, y_all, ok_all, clip_all, smoke):
    """ED / S dense arms against RESULT_RBJK.json (|diff| <= 1e-9)"""
    if smoke:
        return {"status": "SKIPPED-SMOKE"}
    ref = json.load(open(RBJK_RESULT))
    in_s = np.zeros(len(y_all), bool)
    in_s[sidx] = True
    ev = in_s & ok_all
    out = {}
    for n in ("ED", "S"):
        acc = float((refs[n][ev] == y_all[ev]).mean())
        out[n] = {"mine": acc, "rbjk": ref["arms"][n]["accuracy"]["point"], "pass": repro_ok(acc, ref["arms"][n]["accuracy"]["point"])}
    out["n_windows"] = {"mine": int(ev.sum()), "rbjk": ref["n_windows"]}
    out["pass"] = bool(out["ED"]["pass"] and out["S"]["pass"] and out["n_windows"]["mine"] == out["n_windows"]["rbjk"])
    return out


def write_result(out, res3, krepro, beside, stageb):
    seeds = sorted(res3)
    rd = combine_readings([res3[s]["reading_this_seed"] for s in seeds]) if len(seeds) == 2 else None
    ctrl = {"K-REPRO": "PASS" if krepro.get("pass") else ("SKIPPED" if krepro.get("status") else "FAIL")}
    for n in ("K-NEST", "K-SHUF", "K-V"):
        ctrl[n] = "PASS" if all(res3[s]["controls_seed"][n] == "PASS" for s in seeds) else "FAIL"
    held = all(v == "PASS" for v in ctrl.values())
    final = (rd if held else "NOT-READ") if rd else "INCOMPLETE"
    R = {"tool": "rbstack.py", "card": "PREREG_RBSTACK.md sha256 " + CARD_SHA, "tier": "T0 decodability (NOT driving performance)",
         "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, episodes = train clips; answers 'another draw of EPISODES' only (probe training variance: two probe seeds)",
         "margin": MARGIN, "reading": final, "reading_by_seed": {str(s): res3[s]["reading_this_seed"] for s in seeds}, "controls": ctrl,
         "not_read_reason": None if held else [k for k, v in ctrl.items() if v != "PASS"], "K_REPRO": krepro,
         "primary": {str(s): {k: res3[s][k] for k in ("arms", "paired", "n_windows", "n_episodes", "majority_rate", "class_shares", "controls_seed")} for s in seeds},
         "beside_lat5_seed0_no_bar": beside, "stage_B": stageb}
    with open(out + "/RESULT_RBSTACK.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    L = ["# RESULT_RBSTACK", "", f"Card PREREG_RBSTACK.md sha256 {CARD_SHA}. Tier T0 decodability, NOT driving performance. Evidence class MEASURED (ours).",
         "Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, on per-window correctness; answers another draw of episodes only. Probe variance: seeds 0 and 1.", "",
         f"## Reading: **{final}**  (by seed: {R['reading_by_seed']})", "", f"Controls: {ctrl}" + ("" if held else f"  not read because {R['not_read_reason']}"), "",
         "| seed | arm | accuracy [95% CI] |", "|---|---|---|"]
    for s in seeds:
        for n, a in res3[s]["arms"].items():
            x = a["accuracy"]
            L.append(f"| {s} | {n} | {x['point']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}] |")
    L += ["", "| seed | contrast | delta | CI | separated |", "|---|---|---|---|---|"]
    for s in seeds:
        for n, d in res3[s]["paired"].items():
            L.append(f"| {s} | {n} | {d['delta']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] | {d['separated']} |")
    L += ["", f"n windows {res3[seeds[0]]['n_windows']} / episodes {res3[seeds[0]]['n_episodes']}; majority rate {res3[seeds[0]]['majority_rate']:.4f}.",
          "EDV, EDV_SHUF, ED_STR fit on strided labelled fold-train windows (probe features exist for those only); ED and S are the dense RBJK arms.",
          "", f"K-REPRO: {json.dumps(krepro)}", "", "## Beside: 5-way lat_cls, probe seed 0, no bar", "", "```", json.dumps(beside, indent=1, default=str)[:6000], "```", "",
          "## Stage B (held-out D1-HO windows)", "", stageb]
    open(out + "/RESULT_RBSTACK.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    return R


STAGE_B = ("NOT-YET: the NAVRBD D1-HO windows are the 10k data-scale RB grid windows (5,736 / 497 under the card, 7,594 / 545 in the NAVRBD count). Their pooled 8x20 DINOv3 "
           "features and join/ED rows are not on this machine (strat1/navrbd/in is absent; the cached pooled grids here cover only the train4572 and eval-141 clips), so stage B needs "
           "the pod-side cache, and the card forbids pulling from the pod and re-selecting episodes. It never changes the section 3 reading.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    LOG = open(out + "/rbstack_log.txt", "a", encoding="utf-8")
    sentinel = out + "/RBSTACK.done"

    def log(*x):
        s = time.strftime("%H:%M:%S ") + " ".join(str(v) for v in x)
        assert "ZZ RBSTACK" not in s
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()

    def finish(line):
        LOG.write(line + "\n")
        LOG.flush()
        open(sentinel, "w").write(line + "\n")
    try:
        import torch
        torch.set_num_threads(8)
        torch.set_num_threads(8)
        if not a.smoke:
            assert K.RD.sha(CARD) == CARD_SHA, "card changed after hashing"
            cf = commit_free_gb()
            assert cf >= FREE_COMMIT_GB, f"commit free {cf:.1f} GB below {FREE_COMMIT_GB} GB"
            assert free_gb(out) >= FREE_F_GB, f"F: free {free_gb(out):.1f} GB below {FREE_F_GB} GB"
        else:
            J.EPOCHS, J.CKPTS = 1, (1,)
        log("start", "smoke" if a.smoke else "full", "commit free GB", round(commit_free_gb(), 1), "F free GB", round(free_gb(out), 1))
        M = load_meta(out, a.smoke)
        glab, okm = build_labels(M, out, a.smoke)
        log("labels", int(okm.sum()), "gt_ok windows;", int(okm[M["sidx"]].sum()), "strided gt_ok")
        extract_x8(M, out, log)
        F = torch.from_numpy(np.load(out + "/feat/X8.npy", mmap_mode="r"))
        # ---- 1,2: primary target, probe seeds 0 and 1
        res3, refs, _ = run_target("kin3", (0, 1), 3, glab, okm, M, F, out, log)
        kr = k_repro(refs, M["sidx"], glab, okm, M["tr_clip"], a.smoke)
        R = write_result(out, res3, kr, {"status": "PENDING"}, STAGE_B)
        log("RESULT written (primary):", R["reading"], R["controls"])
        # ---- 3: beside, 5-way lat_cls at probe seed 0 (RB-JOINT population), no bar
        y5 = M["tr_y"].astype(np.int64)
        ok5 = np.ones(len(y5), bool)
        res5, refs5, fits5 = run_target("lat5", (0,), 5, y5, ok5, M, F, out, log)
        rj = json.load(open(S1 + "/rbjoint/out/RESULT_RBJOINT.json"))
        agree = None
        if not a.smoke:  # the outer fits of the 5-way run replay RB-JOINT's RAW8 folds: same fit rows, same inner holdout, same seed
            ag = []
            for k in range(NFOLD):
                z = np.load(S1 + f"/rbjoint/out/cv/RAW8_f{k}.npz")
                te = np.where(np.array([K.fold_of(M["tr_clip_ids"][c]) for c in M["tr_clip"][M["sidx"]]]) == k)[0]
                ag.append(float((fits5[f"outer{k}"]["probs"][te].argmax(1) == z["pred"]).mean()))
            agree = {"per_fold_agreement_with_rbjoint_RAW8": ag, "note": "diagnostic of the re-extracted features; drift from thread counts can flip a few windows"}
        beside = {"population": {"windows": res5[0]["n_windows"], "episodes": res5[0]["n_episodes"]}, "paired": res5[0]["paired"], "arms": res5[0]["arms"],
                  "reading_this_seed_for_reference_only": res5[0]["reading_this_seed"], "rbjoint_reference": {"ED_acc": rj["arms"]["ED"]["accuracy"]["point"], "RAW8_acc": rj["arms"]["RAW8"]["accuracy"]["point"]},
                  "feature_replay_check": agree, "controls_seed": res5[0]["controls_seed"]}
        R = write_result(out, res3, kr, beside, STAGE_B)
        # cleanup of this run's own temp: the re-created fp8 feature file (re-creatable by extract_x8)
        del F
        p = out + "/feat/X8.npy"
        if os.path.exists(p) and not a.smoke:
            sz = os.path.getsize(p)
            os.remove(p)
            os.remove(out + "/feat/X8.ok")
            open(out + "/deletions.txt", "a").write(f"{p} {sz} bytes (re-creatable: rbstack.py extract_x8)\n")
        log("all stages done; reading", R["reading"])
        finish("ZZ RBSTACK DONE ZZ")
    except BaseException as e:
        tb = traceback.format_exc()
        open(out + "/traceback.txt", "a", encoding="utf-8").write(tb + "\n")
        print(tb, flush=True)
        reason = (type(e).__name__ + ": " + str(e)).replace("\n", " ")[:300].replace("ZZ RBSTACK", "")
        finish("ZZ RBSTACK FAIL ZZ " + reason)
        raise


if __name__ == "__main__":
    main()
