"""S0e read (PREREG_V8F_PROOF_ADD12 note N7, stage 0e: the reads of stage 0b and stage 0d on the TRAIN corpus; choices in S0E_DEFS.md, rules E1..E20).
Tier T0: a world-model diagnostic, never driving performance. CPU only. Reuses spatial/s0/s0_read.py and spatial/s0c/s0c_read.py by import.
Subcommands:
    run                     full read on the true labels -> RESULT_S0E.json, VERDICT_S0E.md (resumable per fold, stages/); exit code 2 = a control missed
    smoke [--clips N] [--folds K]
                            label-blind: whole-clip label tracks are swapped among the probe clips (seed 0) before anything is computed; prints shapes,
                            counts, timings, e2 (label free) and e4 only; nothing computed on the swapped labels is stored or quoted.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import hashlib
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = "F:/TanitAD-scratch/v8proof"
for _p in (f"{ROOT}/spatial/s0", f"{ROOT}/spatial/s0c"):
    if _p not in sys.path:
        sys.path.append(_p)
import s0_read as S0  # noqa: E402
import s0c_read as S0C  # noqa: E402
RP = S0.RP

DATA = f"{HERE}/data"
JOIN = "C:/Users/Admin/lorapod/ship2/join_train.npz"
SEED = S0.SEED
B_BOOT = S0.B_BOOT
NCELL, CELLD, TOKD = S0.NCELL, S0.CELLD, 64
PROBE_CAP = 20000                      # E4: at most this many training rows / pairs per probe fit
SUB_N = 30000                          # E10: size of the fixed random subsets on which the label-free controls e2 run
GAP_LO, GAP_HI = 0.0, 60.0             # E3: the lead gap range kept (metres)
ARMS = S0C.ARMS
TIER, STATEMENT, FAMILIES = S0.TIER, S0.STATEMENT, S0.FAMILIES
E3_REF = {"grid_gap_r2": 0.3981, "grid_tol": 0.03, "mean_gap_r2": 0.2577, "mean_tol": 0.05}
E1_BAR = 0.02
SUFFIX = "stage 0e (train corpus)"
EGO_NAMES = ("v", "a", "yaw_rate")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ===================================================================================== labels (E3)
def derive_labels(lead_valid, lead_gap):
    """Presence = lead_valid; gap = lead_gap where lead_valid is 1, the gap is finite and 0 <= gap <= 60 m, else NaN."""
    v = np.asarray(lead_valid).astype(np.int64)
    g = np.asarray(lead_gap, np.float64)
    ok = (v == 1) & np.isfinite(g) & (g >= GAP_LO) & (g <= GAP_HI)
    return np.where(ok, g, np.nan), (v == 1).astype(np.float64)


def sub_rows(idx, cap: int = PROBE_CAP, seed: int = 0):
    """E4: a fixed random subsample (seed 0) of at most `cap` indices, in ascending order."""
    idx = np.asarray(idx)
    if len(idx) <= cap:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))


# ===================================================================================== data (E1, E2)
def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_index(data_dir=DATA):
    with open(f"{data_dir}/INDEX.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_join(path=JOIN):
    z = np.load(path)
    return {"clip": z["clip"].astype(str), "off": z["off"].astype(np.int64), "n": z["n"].astype(np.int64),
            "lead_valid": z["lead_valid"], "lead_gap": z["lead_gap"], "now_s": z["now_s"] if "now_s" in z.files else None}


def load_train(n_clips: int | None = None, data_dir: str = DATA, join: dict | None = None, verify_sha: bool = True):
    """Projected tokens, kinematics and join labels of the probe clips. n_clips: the first n clips of INDEX (smoke) or all (None).
    Every shard that is read is checked against the sha256 in INDEX.json."""
    I = load_index(data_dir)
    J = join if join is not None else load_join()
    clips = list(I["clips"])
    take = clips if n_clips is None else clips[:n_clips]
    shard_of = I["shard_of"]
    need = sorted({shard_of[i] for i in range(len(take))})
    Xs, Ks, Cs, Ts, shas = [], [], [], [], {}
    for sh in need:
        p = f"{data_dir}/{sh}"
        if verify_sha:
            shas[sh] = sha256_file(p)
            assert shas[sh] == I["shards"][sh], f"sha256 mismatch {sh}"
        z = np.load(p)
        ci = z["clip_idx"].astype(np.int64)
        m = ci < len(take) if n_clips is not None else np.ones(len(ci), bool)
        Xs.append(z["X"][m]); Ks.append(z["kin"][m]); Cs.append(ci[m]); Ts.append(z["t"][m].astype(np.int64))
        del z
    X = np.concatenate(Xs) if len(Xs) > 1 else Xs[0]
    kin = np.concatenate(Ks).astype(np.float64); ci = np.concatenate(Cs); t = np.concatenate(Ts)
    del Xs, Ks, Cs, Ts
    o = np.lexsort((t, ci))
    if not np.array_equal(o, np.arange(len(o))):
        X, kin, ci, t = X[o], kin[o], ci[o], t[o]
    assert X.shape[1:] == (NCELL, TOKD) and X.dtype == np.float16, (X.shape, X.dtype)
    uep = take
    frames = np.asarray(I["frames"], np.int64)[:len(take)]
    cnt_rows = np.bincount(ci, minlength=len(take))
    jc = {c: i for i, c in enumerate(J["clip"])}
    ji = np.array([jc.get(c, -1) for c in uep])
    assert (ji >= 0).all(), "a probe clip is missing from the join"
    jidx = J["off"][ji[ci]] + t
    gap_l, pres_l = derive_labels(J["lead_valid"][jidx], J["lead_gap"][jidx])
    beid = np.array(uep)[ci]
    D = {"X": X, "kin": kin, "beid": beid, "bg": t, "epi": ci, "uep": uep, "gap": gap_l.copy(), "pres": pres_l.copy(), "gap_loaded": gap_l, "pres_loaded": pres_l,
         "frames_index": frames, "rows_per_clip": cnt_rows, "join_idx": ji, "shas": shas, "index": I}
    return D


def drop_nonfinite_kin(D):
    ok = np.isfinite(D["kin"]).all(1)
    n_bad = int((~ok).sum())
    if n_bad:
        for k in ("X", "kin", "beid", "bg", "epi", "gap", "pres", "gap_loaded", "pres_loaded"):
            D[k] = D[k][ok]
    D["n_dropped_nonfinite_kin"] = n_bad
    return D


def block_swap_tracks(D, seed: int = 0):
    """Smoke only (label blind): the (gap, presence) track of every clip is replaced by the track of another clip (a fixed random permutation of the
    clips, seed 0), cycled to the clip length. Within-clip continuity is kept so the pair set and the timings are representative; the alignment of
    labels with tokens is destroyed."""
    uep = D["uep"]; n = len(uep)
    perm = np.random.default_rng(seed).permutation(n)
    if n > 1 and (perm == np.arange(n)).any():
        perm = np.roll(np.arange(n), 1)
    by = [np.flatnonzero(D["epi"] == c) for c in range(n)]
    gap, pres = D["gap"].copy(), D["pres"].copy()
    for c in range(n):
        src = by[perm[c]]; dst = by[c]
        if len(src) == 0 or len(dst) == 0:
            continue
        sel = np.arange(len(dst)) % len(src)
        gap[dst] = D["gap_loaded"][src][sel]; pres[dst] = D["pres_loaded"][src][sel]
    D["gap"], D["pres"] = gap, pres
    return D


# ===================================================================================== pairs (E7)
def assemble(D):
    assert S0C.PAIR_RULE == "continuous", S0C.PAIR_RULE
    ri, rm, rk, pc = S0C.build_pairs(D["beid"], D["bg"], D["gap"])
    d = D["gap"][rk] - D["gap"][ri]
    Eb = S0C.ego_block(D["kin"], ri, rm, rk)
    folds_b = RP.make_folds(D["beid"])
    return {"ri": ri, "rm": rm, "rk": rk, "Eb": Eb, "ep": D["epi"][ri], "d": d, "fold_p": folds_b[ri], "folds_b": folds_b, "counts": pc}


# ===================================================================================== guards and controls: e4
def guard_fold_leak(X, tr, n_probe: int = 5000, seed: int = 0):
    """The states of the training rows must not change when the tokens of the held-out rows change: rows outside `tr` get large random noise and the
    grid and mean states of the training rows are compared bitwise. LEAK_PCA_FIT = True (the mutation) turns it red."""
    rng = np.random.default_rng(seed)
    n = len(X)
    pick = np.sort(rng.choice(n, min(n, n_probe), replace=False))
    Xs = X[pick]
    trs = np.flatnonzero(np.isin(pick, tr))
    te = np.setdiff1d(np.arange(len(pick)), trs)
    if len(trs) < 20 or len(te) < 20:
        return {"pass": False, "reason": "too few rows", "n_train": int(len(trs)), "n_test": int(len(te))}
    Xp = Xs.copy()
    Xp[te] = (rng.standard_normal(Xp[te].shape) * 5.0).astype(np.float16)
    Sa, Sb = S0.grid_state(Xs, trs)[trs], S0.grid_state(Xp, trs)[trs]
    Ma, Mb = S0.mean_token(Xs), S0.mean_token(Xp)
    Ta, Tb = S0.mean_state(Ma, trs, TOKD)[trs], S0.mean_state(Mb, trs, TOKD)[trs]
    ok = bool(np.array_equal(Sa, Sb) and np.array_equal(Ta, Tb))
    return {"pass": ok, "n_train": int(len(trs)), "n_test": int(len(te)), "bar": "states of training rows identical when held-out tokens change"}


def _eq_nan(a, b):
    return bool((a == b) or (np.isnan(a) and np.isnan(b)))


def check_e4(D, J, n_rand: int = 30, seed: int = 0, eval_clips=(), kin_ref: dict | None = None, require_kin_ref: bool = False, expect_clips: int | None = None):
    """e4 alignment. Labels of 30 random (clip, frame) rows are re-derived from the join by clip NAME and frame, independently of the row index used to
    load them, and compared exactly. Also: join n equals the INDEX frame count for every clip, frames contiguous 0..n-1, clip count, no eval clip,
    kinematics against an optional reference of cache values (not available on this box unless KIN_REF.npz is supplied)."""
    uep = D["uep"]
    jc = {c: i for i, c in enumerate(J["clip"])}
    res = {}
    res["clip_count"] = {"n": int(len(uep)), "expected": expect_clips, "pass": bool(expect_clips is None or len(uep) == expect_clips)}
    ev = set(eval_clips)
    res["eval_clips_in_probe_set"] = {"count": int(sum(1 for c in uep if c in ev)), "n_eval_clips_listed": int(len(ev)), "pass": bool(not any(c in ev for c in uep))}
    bad_n = [c for c, fr in zip(uep, D["frames_index"]) if J["n"][jc[c]] != fr]
    bad_rows = [uep[c] for c in range(len(uep)) if D["rows_per_clip"][c] != D["frames_index"][c]]
    contig = True
    for c in range(len(uep)):
        tt = D["bg"][D["epi"] == c] if len(uep) <= 200 else None
        if tt is not None and not np.array_equal(np.sort(tt), np.arange(len(tt))):
            contig = False
    if len(uep) > 200:                                    # vectorised form: frame index equals position inside its clip
        starts = np.concatenate([[0], np.cumsum(D["rows_per_clip"])[:-1]])
        contig = bool(np.array_equal(D["bg"], np.arange(len(D["bg"])) - np.repeat(starts, D["rows_per_clip"])))
    res["join_n_equals_index_frames"] = {"bad_clips": int(len(bad_n)), "pass": bool(len(bad_n) == 0)}
    res["rows_per_clip_equals_index_frames"] = {"bad_clips": int(len(bad_rows)), "pass": bool(len(bad_rows) == 0)}
    res["frames_contiguous"] = {"pass": bool(contig)}
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(D["bg"]), min(n_rand, len(D["bg"])), replace=False)
    bad_lab = 0
    for r in pick:
        c, t = str(D["beid"][r]), int(D["bg"][r])
        ji = jc.get(c, -1)
        if ji < 0 or not (0 <= t < J["n"][ji]):
            bad_lab += 1
            continue
        g, p = derive_labels(np.array([J["lead_valid"][J["off"][ji] + t]]), np.array([J["lead_gap"][J["off"][ji] + t]]))
        if not (_eq_nan(g[0], D["gap_loaded"][r]) and p[0] == D["pres_loaded"][r]):
            bad_lab += 1
    res["labels_vs_join"] = {"n_checked": int(len(pick)), "bad": int(bad_lab), "pass": bool(bad_lab == 0)}
    if kin_ref is None:
        res["kinematics_vs_cache"] = {"status": "NOT CHECKED: no cache reference on this box (supply data/KIN_REF.npz)", "pass": not require_kin_ref}
    else:
        pos = {(str(c), int(t)): i for i, (c, t) in enumerate(zip(D["beid"], D["bg"]))}
        bad, n = 0, 0
        for (c, t), v in kin_ref.items():
            i = pos.get((c, t), -1)
            if i < 0:
                continue
            n += 1
            bad += int(not np.array_equal(np.asarray(v, np.float64), D["kin"][i]))
        res["kinematics_vs_cache"] = {"n_checked": n, "bad": bad, "pass": bool(n > 0 and bad == 0)}
    if D.get("shas") is not None:
        res["shard_sha256"] = {"n_shards": int(len(D["shas"])), "pass": True}
    res["pass"] = bool(all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v))
    return res


def load_kin_ref(data_dir=DATA):
    p = f"{data_dir}/KIN_REF.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {(str(c), int(t)): v for c, t, v in zip(z["clip"].astype(str), z["t"], z["kin"])}


# ===================================================================================== e2 glue (planted position on 64-wide cells)
def e2_position_fold(f, Xs, Ms, folds_s, epi_s, pl, side_s):
    """S0 R18 on the subset rows: the planted vector sits in the cell or its mirror cell (per row, `side_s`); the grid cells are refitted on the training
    rows with the planted tokens, the mean token gets the same shift vec / 40 for both sides. Returns OOF scores {'g': , 'm': } for the test rows."""
    tr = np.flatnonzero(folds_s != f); te = np.flatnonzero(folds_s == f)
    trs = sub_rows(tr)
    pa, pb, vec = pl["cell"], pl["cell_b"], pl["vec_f5"]
    Sg = S0.grid_state(Xs, tr)
    Sg[:, pa * CELLD:(pa + 1) * CELLD] = S0.cell_block(Xs, tr, pa, CELLD, add=(1.0 - side_s)[:, None] * vec[None, :])
    Sg[:, pb * CELLD:(pb + 1) * CELLD] = S0.cell_block(Xs, tr, pb, CELLD, add=side_s[:, None] * vec[None, :])
    Sm = S0.mean_state(Ms + vec[None, :] / NCELL, tr, TOKD)
    out = {}
    for s, Sx in (("g", Sg), ("m", Sm)):
        o = np.full(len(Xs), np.nan)
        o[te] = S0.fit_probe(Sx[trs], side_s[trs].astype(np.float64), epi_s[trs], SEED + f)(Sx[te])
        out[s] = o
    return out


# ===================================================================================== r-a and change read, one fold (E5, E6, E8)
def ra_fold(f, St, ego, y_gap, y_pres, epi, folds_b, out, lab):
    """Lead gap (finite rows) and presence (all rows) from grid state, mean state and the ego vector; probe = S0.fit_probe on at most PROBE_CAP training
    rows (seed 0); the ego vector is z-scored on the training rows of the fold."""
    tr = np.flatnonzero(folds_b != f); te = np.flatnonzero(folds_b == f)
    mu, sd = ego[tr].mean(0), ego[tr].std(0) + 1e-9
    feats = {"g": St["g"], "m": St["m"], "ego": (ego - mu) / sd}
    info = {}
    for s, F in feats.items():
        for tg, y in (("gap", y_gap), ("pres", y_pres)):
            key = f"ra|{lab}|{s}|{tg}"
            if key not in out:
                out[key] = np.full(len(y), np.nan)
            vtr_all = tr[np.isfinite(y[tr])]
            vtr = sub_rows(vtr_all); vte = te[np.isfinite(y[te])]
            info[f"{tg}|train_rows_available"] = int(len(vtr_all)); info[f"{tg}|train_rows_used"] = int(len(vtr))
            if len(vtr) >= 12 and len(vte):
                out[key][vte] = S0.fit_probe(F[vtr], y[vtr], epi[vtr], SEED + f)(F[vte])
    return info


def change_fold(f, St, Pr, dsets, out, folds_b):
    ri, rm = Pr["ri"], Pr["rm"]
    fold_p = Pr["fold_p"]
    trp_all, tep = np.flatnonzero(fold_p != f), np.flatnonzero(fold_p == f)
    trp = sub_rows(trp_all)
    for lab, d in dsets.items():
        for arm in ARMS:
            F = S0C.arm_features(arm, St, Pr["Eb"], ri, rm, trp)
            key = f"arm|{lab}|{arm}"
            out[key] = np.full(len(ri), np.nan)
            if len(tep):
                out[key][tep] = S0C.arm_oof_fold(F, d, Pr["ep"], trp, tep, f)
    return {"pairs_train_available": int(len(trp_all)), "pairs_train_used": int(len(trp)), "pairs_test": int(len(tep))}


def process_fold(f, D, M, Pr, labsets, K2, E2, log_=log):
    """All fits of fold f. labsets = {'main': {'gap','pres','d'}, 'e1': {...}}. Returns (dict of arrays, dict of step seconds, info)."""
    t0 = time.time(); T = {}
    X = D["X"]
    folds_b = Pr["folds_b"]
    tr = np.flatnonzero(folds_b != f)
    St = {"g": S0.grid_state(X, tr), "m": S0.mean_state(M, tr, TOKD)}
    T["states"] = time.time() - t0
    log_(f"fold {f}: states g {St['g'].shape} m {St['m'].shape} ({T['states']:.0f}s)")
    out, info = {}, {}
    t1 = time.time()
    for lab, L in labsets.items():
        info[f"ra|{lab}"] = ra_fold(f, St, D["kin"], L["gap"], L["pres"], D["epi"], folds_b, out, lab)
    T["r_a"] = time.time() - t1
    log_(f"fold {f}: r-a done ({T['r_a']:.0f}s)")
    t1 = time.time()
    info["change"] = change_fold(f, St, Pr, {lab: L["d"] for lab, L in labsets.items()}, out, folds_b)
    T["change"] = time.time() - t1
    log_(f"fold {f}: change read done ({T['change']:.0f}s)")
    t1 = time.time()
    k = K2
    k_fold = Pr["fold_p"][k["sel"]]
    trp_k_all, tep_k = np.flatnonzero(k_fold != f), np.flatnonzero(k_fold == f)
    trp_k = sub_rows(trp_k_all)
    k2 = S0C.k2_fold(D["X"], M, St, trp_k, tep_k, Pr["ri"][k["sel"]], Pr["rm"][k["sel"]], k["spec"], k["side"], Pr["ep"][k["sel"]], f)
    for kk, v in k2.items():
        out[f"k2|{kk}"] = np.full(len(k["sel"]), np.nan); out[f"k2|{kk}"][tep_k] = v
    T["e2_motion"] = time.time() - t1
    t1 = time.time()
    e2 = e2_position_fold(f, E2["X"], E2["M"], E2["folds"], E2["epi"], E2["pl"], E2["side"])
    for s, v in e2.items():
        out[f"e2p|{s}"] = v
    T["e2_position"] = time.time() - t1
    log_(f"fold {f}: e2 done (motion {T['e2_motion']:.0f}s, position {T['e2_position']:.0f}s); fold total {time.time() - t0:.0f}s")
    return out, T, info


def merge_oof(parts):
    out = {}
    for d in parts:
        for k, v in d.items():
            if k not in out:
                out[k] = v.copy()
            else:
                m = np.isfinite(v)
                out[k][m] = v[m]
    return out


# ===================================================================================== analysis (E9 to E14)
def boot_auc_chunked(ep, y, s, C, chunk: int = 50):
    return np.concatenate([RP.boot_auc(ep, y, s, C[i:i + chunk]) for i in range(0, len(C), chunk)])


def single_auc_c(ep, y, p, C):
    m = np.isfinite(y) & np.isfinite(p)
    return {"auc": RP.auc_pooled(y, p), "ci95": RP.ci(boot_auc_chunked(ep, y, p, C)), "n": int(m.sum()), "n_clips": int(len(np.unique(ep[m])))}


def paired_auc_c(ep, y, pa, pb, C):
    return RP.diff_row(boot_auc_chunked(ep, y, pa, C), boot_auc_chunked(ep, y, pb, C), RP.auc_pooled(y, pa), RP.auc_pooled(y, pb))


def single_r2c(ep, y, p, C, n_ep):
    r = S0.single_r2(ep, y, p, C, n_ep)
    r["n_clips"] = int(len(np.unique(ep[np.isfinite(y) & np.isfinite(p)])))
    return r


def ra_outcomes(diffs_gap):
    o = {}
    for nm, k in (("grid state", "grid minus ego"), ("mean state", "mean minus ego")):
        o[nm] = "S0-LEAD-KEPT" if (diffs_gap[k]["diff"] > 0 and diffs_gap[k]["separated_pos"]) else "S0-LEAD-LOST"
    o["pair"] = "GRID-ADDS" if (diffs_gap["grid minus mean"]["diff"] > 0 and diffs_gap["grid minus mean"]["separated_pos"]) else "GRID-EQUAL"
    return o


def analyse_main(D, Pr, oof, C, n_ep):
    epi, yg, yp = D["epi"], D["gap"], D["pres"]
    ra = {}
    for s in ("g", "m", "ego"):
        ra[f"gap|{s}"] = single_r2c(epi, yg, oof[f"ra|main|{s}|gap"], C, n_ep)
        ra[f"pres|{s}"] = single_auc_c(epi, yp, oof[f"ra|main|{s}|pres"], C)
    P = lambda s, tg: oof[f"ra|main|{s}|{tg}"]
    dg = {"grid minus ego": S0.paired(epi, yg, P("g", "gap"), P("ego", "gap"), C, n_ep), "mean minus ego": S0.paired(epi, yg, P("m", "gap"), P("ego", "gap"), C, n_ep),
          "grid minus mean": S0.paired(epi, yg, P("g", "gap"), P("m", "gap"), C, n_ep)}
    dp = {"grid minus ego": paired_auc_c(epi, yp, P("g", "pres"), P("ego", "pres"), C), "mean minus ego": paired_auc_c(epi, yp, P("m", "pres"), P("ego", "pres"), C),
          "grid minus mean": paired_auc_c(epi, yp, P("g", "pres"), P("m", "pres"), C)}
    ch = S0C.analyse_main(Pr["d"], Pr["ep"], oof, C, n_ep)
    return {"r_a": {"estimates": ra, "diffs_gap_r2": dg, "diffs_presence_auc": dp, "outcomes": ra_outcomes(dg)}, "change": ch}


def e1_values(D, Pr, oof):
    gap_r2 = {f"ra gap R2 {s}": RP.r2_pooled(D["gap"], oof[f"ra|e1|{s}|gap"]) for s in ("g", "m", "ego")}
    sk = {f"change skill {a}": S0C.skill(Pr["d_e1"], oof[f"arm|e1|{a}"]) for a in ARMS}
    auc = {f"ra presence AUROC {s} (printed, no bar)": RP.auc_pooled(D["pres_e1"], oof[f"ra|e1|{s}|pres"]) for s in ("g", "m", "ego")}
    vals = {**gap_r2, **sk}
    return {"pass": bool(all(v <= E1_BAR for v in vals.values())), "max_value": float(max(vals.values())), "bar": E1_BAR, "values": vals, "presence_auc_printed": auc,
            "n_gap_rows": int(np.isfinite(D["gap"]).sum()), "n_pairs": int(len(Pr["d"])), "n_clips": int(len(D["uep"]))}


# ===================================================================================== e3 (eval bank, same projection)
def e3_read(proj, smoke: bool, log_=log):
    """Stage-0 r-a read of the projected eval-bank tokens: grid state (16 directions per cell) and mean state (all 64), lead gap pooled R2."""
    t0 = time.time()
    Dv = S0.load_data(smoke)
    X = Dv["X"]; N = len(X)
    Xp = np.empty((N, NCELL, TOKD), np.float16)
    for a in range(0, N, 256):
        Xp[a:a + 256] = (X[a:a + 256].astype(np.float32) @ proj.astype(np.float32)).astype(np.float16)
    gap = Dv["gap"].copy()
    if smoke:
        gap = S0.permute_across(gap, 11)
    del X, Dv["X"]
    Mp = S0.mean_token(Xp)
    folds = RP.make_folds(Dv["beid"])
    og, om = np.full(N, np.nan), np.full(N, np.nan)
    for f in range(int(folds.max()) + 1):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        Sg, Sm = S0.grid_state(Xp, tr), S0.mean_state(Mp, tr, TOKD)
        vtr, vte = tr[np.isfinite(gap[tr])], te[np.isfinite(gap[te])]
        if len(vtr) >= 12 and len(vte):
            og[vte] = S0.fit_probe(Sg[vtr], gap[vtr], Dv["epi"][vtr], S0.SEED + f)(Sg[vte])
            om[vte] = S0.fit_probe(Sm[vtr], gap[vtr], Dv["epi"][vtr], S0.SEED + f)(Sm[vte])
    rg, rm_ = RP.r2_pooled(gap, og), RP.r2_pooled(gap, om)
    ok = bool(abs(rg - E3_REF["grid_gap_r2"]) <= E3_REF["grid_tol"] and abs(rm_ - E3_REF["mean_gap_r2"]) <= E3_REF["mean_tol"])
    res = {"pass": ok, "grid_gap_r2": float(rg), "mean_gap_r2": float(rm_), "reference": E3_REF, "n_gap_rows": int(np.isfinite(gap).sum()), "n_rows": int(N),
           "n_clips": int(len(Dv["uep"])), "seconds": time.time() - t0, "smoke_labels_permuted": bool(smoke)}
    return res


# ===================================================================================== verdict
def ci_s(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def write_verdict(path, res, meta):
    ctr = res["controls"]
    L = [f"# VERDICT_S0E: PREREG_V8F_PROOF_ADD12 note N7, {SUFFIX}"]
    L.append(f"Tier {TIER}: {STATEMENT}. Estimator: paired episode-cluster bootstrap with clips as clusters ({meta['n_clips']} clips), B {meta['B']}, seed 0, one draw shared by every number. {FAMILIES}")
    L.append("A negative of this read is a negative about a LINEAR / kernel read of untrained pooled tokens projected to 64 numbers per cell; stage 1 still runs.")
    L.append("")
    L.append("## CONTROLS (first)")
    for k in ("e1", "e2_position", "e2_motion", "e3", "e4"):
        c = ctr[k]
        L.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} :: " + json.dumps({a: b for a, b in c.items() if a != 'pass'}, default=float))
    L.append(f"- controls_pass: {res['controls_pass']}")
    L.append("")
    if not res["controls_pass"]:
        L.append("## OUTCOME NAMES WITHHELD: a control did not read its known value. The numbers are in RESULT_S0E.json and are not quotable.")
    else:
        m = res["main"]
        ra, ch = m["r_a"], m["change"]
        n = meta["counts"]
        L.append(f"## Reads (a): lead gap and presence from one frame ({TIER}; {STATEMENT})")
        L.append(f"Frames {n['bank_rows']}, frames with a lead gap in 0 to 60 m {n['gap_rows']}, clips {meta['n_clips']}; probe fit subsample at most {PROBE_CAP} training rows per fold.")
        for s, nm in (("g", "grid state"), ("m", "mean state"), ("ego", "ego vector")):
            a, b = ra["estimates"][f"gap|{s}"], ra["estimates"][f"pres|{s}"]
            L.append(f"- {nm}: gap R2 {a['r2']:+.3f} {ci_s(a['ci95'])} (n {a['n']} frames, {a['n_clips']} clips); presence AUROC {b['auc']:.3f} {ci_s(b['ci95'])} (n {b['n']} frames, {b['n_clips']} clips)")
        for k, v in ra["diffs_gap_r2"].items():
            w = ra["diffs_presence_auc"][k]
            L.append(f"- {k}: gap R2 {v['diff']:+.3f} {ci_s(v['ci95'])}; presence AUROC {w['diff']:+.3f} {ci_s(w['ci95'])}")
        L.append(f"## Reads (b): change of the lead gap over 2.0 s, skill over persistence ({TIER}; {STATEMENT})")
        ds = ch["d_stats"]
        L.append(f"Pairs {ds['n_pairs']} of {meta['n_pair_clips']} clips; pairs removed by the continuity rule {n['pairs_removed_by_the_continuity_rule']} (step above {n['step_max_m']} m or a gap missing inside the 2 s); std(d) {ds['std_d_m']:.3f} m, mean {ds['mean_d_m']:+.3f} m, share |d| above 2 m {ds['share_abs_d_above_2m']:.3f}; fit subsample at most {PROBE_CAP} training pairs per fold.")
        for a in ARMS:
            r = ch["arms"][a]
            L.append(f"- {a}: skill {r['skill']:+.3f} {ci_s(r['ci95'])}; plain R2 {r['r2']:+.3f} {ci_s(r['r2_ci95'])}; n pairs {r['n_pairs']}, clips {r['n_episodes']}")
        for k, v in ch["diffs"].items():
            L.append(f"- {k}: {v['diff']:+.3f} {ci_s(v['ci95'])}")
        L.append("## OUTCOME NAMES (" + SUFFIX + ")")
        o = ra["outcomes"]
        L.append(f"- r-a grid state against ego vector: {o['grid state']}, {SUFFIX}")
        L.append(f"- r-a mean state against ego vector: {o['mean state']}, {SUFFIX}")
        L.append(f"- r-a grid against mean: {o['pair']}, {SUFFIX}")
        oc = ch["outcomes"]
        L.append(f"- change, grid state (H-g against E): {oc['state_g']}, {SUFFIX}")
        L.append(f"- change, mean state (H-m against E): {oc['state_m']}, {SUFFIX}")
        L.append(f"- change, grid against mean (H-g against H-m): {oc['pair']}, {SUFFIX}")
        L.append(f"- change, history (H-g against N-g): {oc['history']}, {SUFFIX}")
    L.append("")
    L.append("Counts: " + json.dumps(meta["counts"]))
    L.append("Subsamples: " + json.dumps(meta["subsamples"]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ===================================================================================== driver
def peak_ram_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 1e9
    except Exception:
        return float("nan")


def cmd_run(smoke: bool, n_clips: int | None, n_folds: int | None):
    t_all = time.time()
    assert not S0.LEAK_PCA_FIT
    out_dir = f"{HERE}/smoke" if smoke else HERE
    stage_dir = f"{out_dir}/stages"
    os.makedirs(stage_dir, exist_ok=True)
    B = 200 if smoke else B_BOOT
    proj = np.load(f"{DATA}/proj.npy")
    J = load_join()
    D = load_train(n_clips if smoke else None, join=J)
    if smoke:
        D = block_swap_tracks(D, 0)
    t_load = time.time() - t_all
    t1 = time.time()
    kin_ref = load_kin_ref()
    eval_clips = [os.path.splitext(x)[0] for x in os.listdir(S0.CLIPS)] if os.path.isdir(S0.CLIPS) else []
    ctr_e4 = check_e4(D, J, eval_clips=eval_clips, kin_ref=kin_ref, expect_clips=None if smoke else 1200)
    ctr_e4["fold_leak_guard"] = guard_fold_leak(D["X"], np.flatnonzero(RP.make_folds(D["beid"]) != 0))
    ctr_e4["pass"] = bool(ctr_e4["pass"] and ctr_e4["fold_leak_guard"]["pass"])
    T_e4 = time.time() - t1
    n_rows_loaded = len(D["X"])
    D = drop_nonfinite_kin(D)
    ctr_e4["rows_dropped_nonfinite_kin"] = int(D["n_dropped_nonfinite_kin"])
    N = len(D["X"])
    Pr = assemble(D)
    ri = Pr["ri"]
    folds_b = Pr["folds_b"]
    M = S0.mean_token(D["X"])
    d, fold_p = Pr["d"], Pr["fold_p"]
    Pr["d_e1"] = S0.permute_within_fold(d, fold_p, 1)
    D["gap_e1"] = S0.permute_within_fold(D["gap"], folds_b, 1)
    D["pres_e1"] = S0.permute_within_fold(D["pres"], folds_b, 2)
    labsets = {"main": {"gap": D["gap"], "pres": D["pres"], "d": d}, "e1": {"gap": D["gap_e1"], "pres": D["pres_e1"], "d": Pr["d_e1"]}}
    # fixed subsets for the label-free controls (E10)
    rsel = np.sort(np.random.default_rng(0).choice(N, min(N, SUB_N), replace=False))
    ksel = np.sort(np.random.default_rng(1).choice(len(ri), min(len(ri), SUB_N), replace=False))
    spec = S0C.k2_setup(D)
    K2 = {"sel": ksel, "spec": spec, "side": S0C.k2_side(len(ksel), spec["seed_side"])}
    pl = S0.planted_setup(D)
    E2 = {"X": D["X"][rsel], "M": M[rsel], "folds": folds_b[rsel], "epi": D["epi"][rsel], "pl": pl, "side": (np.random.default_rng(6).random(len(rsel)) < 0.5).astype(np.float64)}
    n_ep = len(D["uep"])
    log(f"rows {N} (loaded {n_rows_loaded}), clips {n_ep}, gap rows {int(np.isfinite(D['gap']).sum())}, pairs {len(ri)} (three rows {Pr['counts']['pairs_with_three_rows']}, "
        f"removed by continuity {Pr['counts']['pairs_removed_by_the_continuity_rule']}), e2 rows {len(rsel)}, e2 pairs {len(ksel)}, k2 cell {spec['cell']}")
    sig = {"N": N, "P": int(len(ri)), "smoke": smoke, "clips": n_ep, "cap": PROBE_CAP, "sub": SUB_N}
    nf = 5 if n_folds is None else n_folds
    parts, times, infos = [], {}, {}
    for f in range(nf):
        sp = f"{stage_dir}/fold{f}.npz"
        if not smoke and os.path.exists(sp):
            z = np.load(sp, allow_pickle=True)
            if json.loads(str(z["sig"])) == sig:
                parts.append({k: z[k] for k in z.files if k not in ("sig", "times", "info")}); log(f"fold {f}: resumed"); continue
        o, T, info = process_fold(f, D, M, Pr, labsets, K2, E2)
        times[f"fold{f}"] = T; infos[f"fold{f}"] = info
        if not smoke:
            np.savez(sp, sig=json.dumps(sig), times=json.dumps(T), info=json.dumps(info), **o)
        parts.append(o)
    ctr = {"e4": ctr_e4}
    if smoke:
        o = merge_oof(parts)
        pos_ok = RP.auc_pooled(E2["side"], o["e2p|g"]), RP.auc_pooled(E2["side"], o["e2p|m"])
        ctr["e2_position"] = S0.check_s2(*pos_ok)
        ctr["e2_position"].update({"n_rows": int(len(rsel)), "n_clips": int(len(np.unique(E2["epi"]))), "cell": pl["cell"], "mirror_cell": pl["cell_b"], "folds_run": nf})
        ctr["e2_motion"] = S0C.check_k2(RP.auc_pooled(K2["side"], o["k2|grid2"]), RP.auc_pooled(K2["side"], o["k2|mean2"]), RP.auc_pooled(K2["side"], o["k2|grid1"]))
        ctr["e2_motion"].update({"n_pairs": int(len(ksel)), "cell": spec["cell"]})
        t_e3 = time.time()
        e3 = e3_read(proj, True)
        t_e3 = time.time() - t_e3
        sec = {k: float(sum(t[k] for t in times.values()) / max(len(times), 1)) for k in next(iter(times.values()))}
        keep = {"shapes": {"X": list(D["X"].shape), "kin": list(D["kin"].shape), "S0g": [N, NCELL * CELLD], "S0m": [N, TOKD], "H-g": [len(ri), 2 * NCELL * CELLD + 9], "H-m": [len(ri), 2 * TOKD + 9],
                           "N-g": [len(ri), NCELL * CELLD + 9], "E": [len(ri), 9]},
                "counts": {**Pr["counts"], "rows_loaded": n_rows_loaded, "rows_used": N, "gap_rows": int(np.isfinite(D["gap"]).sum()), "clips": n_ep,
                           "pairs_per_fold": [int((fold_p == f).sum()) for f in range(5)]},
                "e2_position": ctr["e2_position"], "e2_motion": ctr["e2_motion"], "e4": ctr["e4"],
                "e3_ran_on_permuted_eval_labels": {"seconds": e3["seconds"], "n_rows": e3["n_rows"], "n_clips": e3["n_clips"]},
                "mean_seconds_per_fold_by_step": sec, "folds_run": nf, "load_seconds": t_load, "e4_seconds": T_e4, "e3_seconds": t_e3,
                "peak_ram_gb": peak_ram_gb(), "total_seconds": time.time() - t_all}
        with open(f"{out_dir}/RESULT_S0E_SMOKE.json", "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=1, default=float)
        log("SMOKE", json.dumps(keep, default=float))
        return 0
    o = merge_oof(parts)
    ctr["e2_position"] = S0.check_s2(RP.auc_pooled(E2["side"], o["e2p|g"]), RP.auc_pooled(E2["side"], o["e2p|m"]))
    ctr["e2_position"].update({"n_rows": int(len(rsel)), "n_clips": int(len(np.unique(E2["epi"]))), "cell": pl["cell"], "mirror_cell": pl["cell_b"]})
    ctr["e2_motion"] = S0C.check_k2(RP.auc_pooled(K2["side"], o["k2|grid2"]), RP.auc_pooled(K2["side"], o["k2|mean2"]), RP.auc_pooled(K2["side"], o["k2|grid1"]))
    ctr["e2_motion"].update({"n_pairs": int(len(ksel)), "n_clips": int(len(np.unique(Pr["ep"][ksel]))), "cell": spec["cell"]})
    ctr["e1"] = e1_values(D, Pr, o)
    log("loading the eval bank for e3 ...")
    ctr["e3"] = e3_read(proj, False)
    C = RP.boot_counts(n_ep, B, SEED)
    meta = {"note": "PREREG_V8F_PROOF_ADD12_N7.md", "defs": "S0E_DEFS.md", "tier": TIER, "statement": STATEMENT, "B": B, "n_clips": n_ep, "n_pair_clips": int(len(np.unique(Pr["ep"]))),
            "estimator": "paired episode-cluster bootstrap, clips as clusters",
            "counts": {**Pr["counts"], "bank_rows": N, "gap_rows": int(np.isfinite(D["gap"]).sum()), "pairs_per_fold": [int((fold_p == f).sum()) for f in range(5)]},
            "subsamples": {"probe_cap": PROBE_CAP, "e2_rows": int(len(rsel)), "e2_pairs": int(len(ksel)), "per_fold": infos, "pca_fit_rows": "all training-fold rows"},
            "stage_seconds": times, "load_seconds": t_load, "peak_ram_gb": peak_ram_gb()}
    res = {"controls": ctr, "meta": meta}
    res["controls_pass"] = bool(all(ctr[k]["pass"] for k in ("e1", "e2_position", "e2_motion", "e3", "e4")))
    res["main"] = analyse_main(D, Pr, o, C, n_ep)
    res["main"]["change"]["d_stats"]["n_pairs"] = int(len(d))
    if not res["controls_pass"]:
        res["main"]["r_a"].pop("outcomes"); res["main"]["change"].pop("outcomes")
    with open(f"{out_dir}/RESULT_S0E.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_verdict(f"{out_dir}/VERDICT_S0E.md", res, meta)
    log(f"done {time.time() - t_all:.0f}s; controls_pass {res['controls_pass']}; peak RAM {peak_ram_gb():.2f} GB")
    return 0 if res["controls_pass"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "smoke"])
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    sys.exit(cmd_run(a.cmd == "smoke", a.clips, a.folds if a.cmd == "smoke" else None))
