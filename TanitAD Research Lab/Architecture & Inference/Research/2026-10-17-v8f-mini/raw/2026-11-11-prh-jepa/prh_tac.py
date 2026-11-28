"""PRH-TAC (PREREAD_PRH_TAC.md, sha256 647c8ed5...): is the seed-shared adapter subspace task-shaped?  CPU only, zero training.

  python prh_tac.py --out RESULT_PRH_TAC.json --md RESULT_PRH_TAC.md

Reuses prh_seed.py (extraction of nm / TR0 / TR1 / RI0 / RI1, fit_cca, ci95, is_sep_pos, SLOW fit, beside helpers) and prh_seed_v2.py (_ctx = clip-parity split + bootstrap counts seed 0,
metric 'CC32', permuted).  The only new objects are the TAC targets, the TAC ridge, the label-shuffle control, the R2 recovery row and the outcome rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("PRH_SRC_DIR", os.path.dirname(os.path.abspath(__file__))))
import prh_seed as P  # noqa: E402
import prh_seed_v2 as V  # noqa: E402

CARD_SHA = "647c8ed5a10aa575ef01e081ce37b94a6ae21c983c5b9aa16b9fd785f2bd6fbd"
JOIN = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CALIB_PRH_SEED_V2.json")
N_GOAL, N_LAT, N_LON = 22, 5, 7
SHUF_SEED_MAIN, SHUF_SEEDS_EXTRA = 11, (12, 13, 14, 15)

CHOICES = {
    "label_source": "join_eval141.npz built by tanitad/data/v8f_mini_data.py:build_label_join (lines 165-240); the eval harness loads it at stack/scripts/v8f_t1_eval.py:515 (D.load_join) and the trainer packs one row into the "
                    "tactical targets at v8f_mini_data.py:_lab_pack (lines 607-615).",
    "frame_to_window_rule": "Labels are PER FRAME, no window anchor needed: join row off[c] + j is cache step j of clip c (v8f_mini_data.py:162-163), the same step index t that prh_seed.extract stores (asserted n[c] == frames of clip c, "
                            "clip order identical). Frames used = ok(v2) AND has AND lat_allowed / lon_allowed single-bit; every representation (TR, RI, TAC, SLOW, PCA) is restricted to those frames for every S, fit half and score half alike.",
    "goal_bits": "goal_y bit b of the uint32 for b in 0..21, imputed 0.5 where goal_w bit is 0 (the trainer's own target rule, v8f_mini.py:_code_gt line 895-896).",
    "lat_lon": "one-hot of lat_cls (5) and lon_cls (7); single-class = popcount(lat_allowed) == 1 and popcount(lon_allowed) == 1 (cls equals that bit, checked).",
    "constraints": "the 26 'cons' cells are not used (card).",
    "degenerate": "a target column is dropped if its standard deviation is < 1e-9 on the fit half OR on the score half of the TAC frames.",
    "TAC_rank": "the 31 non-constant outputs are linearly dependent (lat one-hot of 3 present classes and lon one-hot sum to 1, plus goal-bit dependencies): TAC is rotated onto the eigen-directions of its fit-half covariance with eigenvalue > 1e-8 x max "
                "(rank_reduce), d_TAC = that rank. Added AFTER a first run at d=31 failed C-SELF (S 0.834, see README); the first run's numbers are kept there.",
    "TAC_ridge": "W = (C + lam I)^-1 Cov(nm, Y) with C = covariance of nm on the fit-half TAC frames, lam = 1e-3 x mean eigenvalue of C; centred by fit-half means; TAC = (nm - mu_fit) @ W for every frame.",
    "metric": "CC32 of prh_seed_v2.metric: mean of the first min(32, d_TAC) canonical correlations (fit half directions, score half correlations; ridge 1e-6), B 2000 clip bootstrap, counts rng seed 0 over the 70 score clips.",
    "SLOW_PCA": "built exactly as prh_seed_v2.step_b (SLOW from (t, t+5) pairs over ALL ok fit-half frames, PCA of nm over the same ok fit-half frames) and then evaluated on the TAC frames only.",
    "C-SHUF": "prh_seed_v2.permuted (rng seed 5) applied to TAC: frames permuted within the fit half and within the score half (clip membership broken); pass if S(TR0, TAC_shuf) and S(TR1, TAC_shuf) < the v2 FLOOR "
              "(CALIB_PRH_SEED_V2.json CC32 FLOOR point).",
    "C-LABEL-SHUF": "targets re-assigned across FIT-HALF CLIPS: the fit clips' target blocks are cyclically re-dealt by a random permutation of the clips (rng seed 11; a block is tiled/truncated to the receiving clip's frame count), "
                    "TAC refit on the scrambled targets, same ridge rule. Pass = Q1 NOT separated-positive at both seeds (the card's Q1 condition); the two-sided flag is also reported. Four more shuffle seeds (12-15) are reported as points only.",
    "R2_recovery": "ridge readout of the real targets (fit half -> score half; lam = 1e-3 x mean eigenvalue of the source's own fit-half covariance, centred by fit means) from nm, TR, RI; R2 = mean over target columns of 1 - SSE/SST on the score half (SST about the fit-half mean); "
                   "fraction = R2(source) / R2(nm). Bootstrap over score clips, same counts.",
    "outcome_rule": "controls first: any control fails -> UNUSABLE (not interpreted). Else TAC-LEAN iff Q1 separated positive at both seeds, else NO-TAC-LEAN.",
}


# ------------------------------------------------------------------------------ targets
def popcount(a):
    a = np.asarray(a).astype(np.int64)
    return np.array([bin(int(x)).count("1") for x in a])


def build_targets(J):
    """Per join row: Y [N, 34] = 22 goal bits (0.5 where unlabelled) + lat one-hot 5 + lon one-hot 7, and the mask of rows that carry single-class labels."""
    N = len(J["goal_y"])
    gy, gw = J["goal_y"].astype(np.int64), J["goal_w"].astype(np.int64)
    bits = np.arange(N_GOAL)
    y, w = ((gy[:, None] >> bits) & 1).astype(np.float64), ((gw[:, None] >> bits) & 1).astype(np.float64)
    goal = y * w + 0.5 * (1 - w)
    la, lo = J["lat_allowed"].astype(np.int64), J["lon_allowed"].astype(np.int64)
    single = J["has"].astype(bool) & (popcount(la) == 1) & (popcount(lo) == 1)
    lat = np.zeros((N, N_LAT))
    lon = np.zeros((N, N_LON))
    ii = np.where(single)[0]
    lat[ii, J["lat_cls"][ii].astype(int)] = 1.0
    lon[ii, J["lon_cls"][ii].astype(int)] = 1.0
    assert ((1 << J["lat_cls"][ii].astype(int)) == la[ii]).all() and ((1 << J["lon_cls"][ii].astype(int)) == lo[ii]).all()
    return np.concatenate([goal, lat, lon], 1), single


def nondegenerate_cols(Y, fit_idx, score_idx, tol=1e-9):
    return np.where((Y[fit_idx].std(0) > tol) & (Y[score_idx].std(0) > tol))[0]


# ------------------------------------------------------------------------------ TAC
def ridge_W(X, Y, lam_rule=1e-3):
    """Ridge weights of Y on X (rows), centred; returns (W [dx, dy], mu_x, mu_y)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    n = len(X)
    C = Xc.T @ Xc / n
    lam = lam_rule * float(np.linalg.eigvalsh(C).mean())
    W = np.linalg.solve(C + lam * np.eye(C.shape[0]), Xc.T @ Yc / n)
    return W, mx, my


RANK_TOL = 1e-8


def rank_reduce(T, fit_idx, tol=RANK_TOL):
    """Rotate T onto the eigen-directions of its fit-half covariance whose eigenvalue exceeds tol x the largest one. The target set is linearly dependent (one-hot blocks sum to 1, so the ridge outputs
    inherit exact null directions); a canonical correlation on a null direction is noise, and C-SELF cannot read 1 with them present. An orthogonal change of basis inside the span: CCA with a full-rank partner is unchanged."""
    m = T[fit_idx].mean(0)
    w, Vv = np.linalg.eigh(np.cov((T[fit_idx] - m).T, bias=True))
    keep = w > tol * w.max()
    return (T - m) @ Vv[:, keep], int(keep.sum())


def fit_tac(nm, Y, fit_idx):
    """TAC representation (rank-reduced) and its dimension."""
    W, mx, _ = ridge_W(nm[fit_idx], Y[fit_idx])
    return rank_reduce((nm - mx) @ W, fit_idx)


def label_shuffle(Y, cid, fit_idx, seed):
    """Targets of the fit-half rows re-dealt across fit-half CLIPS (block of clip j goes to clip i = perm(j), tiled/truncated to len(i)); rows outside the fit half are left as they are (never read by the fit)."""
    rng = np.random.default_rng(seed)
    out = Y.copy()
    fit_clips = np.unique(cid[fit_idx])
    perm = rng.permutation(len(fit_clips))
    rows = {c: fit_idx[cid[fit_idx] == c] for c in fit_clips}
    for k, c in enumerate(fit_clips):
        src = rows[fit_clips[perm[k]]]
        out[rows[c]] = Y[src[np.arange(len(rows[c])) % len(src)]]
    return out


# ------------------------------------------------------------------------------ rules (mutation-checked)
def q1_separated(lo, hi):
    """Q1 bar: the 95 % interval of the paired difference lies above zero."""
    return bool(P.is_sep_pos(lo, hi))


def decide(q1_flags, controls_pass):
    """q1_flags: [bool, bool] (seed 0, seed 1); controls_pass: bool -> outcome string."""
    if not controls_pass:
        return "UNUSABLE (a control failed; not interpreted)"
    return "TAC-LEAN" if all(q1_flags) else "NO-TAC-LEAN"


def q1_block(tr, ri, tac, ctx):
    """Paired S(TR,TAC) - S(RI,TAC) with the shared bootstrap counts."""
    a, b = V.metric("CC32", tr, tac, ctx), V.metric("CC32", ri, tac, ctx)
    d = a[1] - b[1]
    lo, hi = P.ci95(d)
    return {"S_TR": a[0], "S_TR_ci95": P.ci95(a[1]), "S_RI": b[0], "S_RI_ci95": P.ci95(b[1]), "diff": a[0] - b[0], "diff_ci95": [lo, hi], "separated_positive": q1_separated(lo, hi), "separated_any": bool(P.is_sep_any(lo, hi))}


def r2_cols_stats(pred, Y, fit_mean, cs, nC):
    """Per-clip SSE and SST per column on the score rows -> (sse [nC, d], sst [nC, d])."""
    sse, sst = np.zeros((nC, Y.shape[1])), np.zeros((nC, Y.shape[1]))
    np.add.at(sse, cs, (Y - pred) ** 2)
    np.add.at(sst, cs, (Y - fit_mean) ** 2)
    return sse, sst


def r2_readout(src, Y, ctx):
    """Ridge readout of Y from src (fit half -> score half); returns (point, boot[B]) of the mean per-column R2."""
    fi, si = ctx["fit_idx"], ctx["score_idx"]
    W, mx, my = ridge_W(src[fi], Y[fi])
    pred = (src[si] - mx) @ W + my
    sse, sst = r2_cols_stats(pred, Y[si], my, ctx["cid_score"], ctx["nC"])
    pt = float((1 - sse.sum(0) / sst.sum(0)).mean())
    c = ctx["counts"]
    boot = (1 - (c @ sse) / (c @ sst)).mean(1)
    return pt, boot


# ------------------------------------------------------------------------------ main
def build_slow_pca(X, ctx_all):
    nm = X["nm"].astype(np.float64)
    fi, ok, cid, is_fit = ctx_all["fit_idx"], ctx_all["ok"], ctx_all["cid"], ctx_all["is_fit"]
    n_all = len(nm)
    nxt = np.arange(n_all) + P.LAG
    valid = nxt < n_all
    valid[valid] &= (cid[nxt[valid]] == cid[valid.nonzero()[0]]) & ok[nxt[valid]]
    pair_idx = np.where(valid & ok & is_fit)[0]
    fs = P.fit_cca(nm[pair_idx], nm[pair_idx + P.LAG], P.RIDGE_B, k=P.K)
    slow = (nm - fs["ma"]) @ fs["Pa"]
    mpca = nm[fi].mean(0)
    w, Vv = np.linalg.eigh(np.cov((nm[fi] - mpca).T, bias=True))
    return slow, (nm - mpca) @ Vv[:, ::-1][:, :P.K], int(len(pair_idx))


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(X, J, B=P.B_BOOT):
    nm = X["nm"].astype(np.float64)
    reps = {k: X[k].astype(np.float64) for k in ("tr0", "tr1", "ri0", "ri1")}
    clips = [str(c) for c in X["clips"]]
    cid, t = X["cid"], X["t"]
    # label join alignment
    assert [str(c) for c in J["clip"]] == clips
    cnt = np.bincount(cid, minlength=len(clips))
    assert (cnt == J["n"]).all(), "frame counts per clip differ between extract and join"
    row = J["off"][cid] + t
    Yall, single = build_targets(J)
    Yf, sf = Yall[row], single[row]
    ctx_all = V._ctx(X, B)
    X_t = dict(X)
    X_t["ok"] = (X["ok"].astype(bool) & sf)
    ctx = V._ctx(X_t, B)
    fi, si = ctx["fit_idx"], ctx["score_idx"]
    keep = nondegenerate_cols(Yf, fi, si)
    Y = Yf[:, keep]
    d_tac = int(len(keep))
    dropped = [int(i) for i in range(Yall.shape[1]) if i not in set(keep.tolist())]
    TAC, d_rank = fit_tac(nm, Y, fi)
    slow, pca, n_pairs = build_slow_pca(X, ctx_all)

    out = {"n": {"frames_total": int(len(nm)), "frames_ok_v2": int(ctx_all["ok"].sum()), "frames_TAC": int(ctx["ok"].sum()), "fit_frames": int(len(fi)), "score_frames": int(len(si)),
                 "fit_clips_with_frames": int(len(np.unique(cid[fi]))), "score_clips_with_frames": int(len(np.unique(cid[si]))), "score_clips_bootstrapped": ctx["nC"], "B": B, "slow_pairs": n_pairs},
           "d_TAC": d_rank, "d_TAC_columns_after_constant_drop": d_tac, "dropped_columns_of_34": dropped, "cc_dims_used": min(32, d_rank),
           "lat_classes_present": sorted(set(int(c) for c in J["lat_cls"][single])), "lon_classes_present": sorted(set(int(c) for c in J["lon_cls"][single]))}

    q1 = {f"s{s}": q1_block(reps[f"tr{s}"], reps[f"ri{s}"], TAC, ctx) for s in (0, 1)}
    out["Q1"] = q1

    # beside rows
    beside = {}
    for s in (0, 1):
        a, b = V.metric("CC32", reps[f"tr{s}"], TAC, ctx), V.metric("CC32", reps[f"tr{s}"], slow, ctx)
        d = a[1] - b[1]
        beside[f"S(TR{s},TAC)-S(TR{s},SLOW)"] = {"point": a[0] - b[0], "ci95": P.ci95(d), "S(TR,SLOW)": b[0], "S(TR,SLOW)_ci95": P.ci95(b[1])}
    sp = V.metric("CC32", pca, TAC, ctx)
    beside["S(PCA,TAC)"] = {"point": sp[0], "ci95": P.ci95(sp[1])}
    r2 = {}
    r2["nm"] = r2_readout(nm, Y, ctx)
    for k in ("tr0", "tr1", "ri0", "ri1"):
        r2[k] = r2_readout(reps[k], Y, ctx)
    rec = {}
    for k in ("tr0", "tr1", "ri0", "ri1"):
        fr = r2[k][1] / r2["nm"][1]
        rec[k] = {"R2": r2[k][0], "R2_ci95": P.ci95(r2[k][1]), "fraction_of_nm": r2[k][0] / r2["nm"][0], "fraction_ci95": P.ci95(fr)}
    for s in (0, 1):
        d = (r2[f"tr{s}"][1] - r2[f"ri{s}"][1]) / r2["nm"][1]
        rec[f"fraction_TR{s}_minus_RI{s}"] = {"point": (r2[f"tr{s}"][0] - r2[f"ri{s}"][0]) / r2["nm"][0], "ci95": P.ci95(d)}
    beside["R2_recovery"] = {"R2_nm": r2["nm"][0], "R2_nm_ci95": P.ci95(r2["nm"][1]), **rec}
    out["beside"] = beside

    # controls
    ctl = {}
    s_self = V.metric("CC32", TAC, TAC, ctx)[0]
    ctl["C-SELF"] = {"S": s_self, "pass": bool(abs(s_self - 1) <= 1e-6)}
    floor = json.load(open(CALIB, encoding="utf-8"))["metrics"]["CC32"]["values"]["FLOOR"]["point"]
    tac_sh = V.permuted(TAC, ctx)
    sh = {f"S(TR{s},TAC_shuf)": V.metric("CC32", reps[f"tr{s}"], tac_sh, ctx) for s in (0, 1)}
    ctl["C-SHUF"] = {"v2_FLOOR_CC32": floor, **{k: {"S": v[0], "ci95": P.ci95(v[1])} for k, v in sh.items()}, "pass": bool(all(v[0] < floor for v in sh.values()))}
    Ysh = label_shuffle(Y, cid, fi, SHUF_SEED_MAIN)
    TAC_l, d_l = fit_tac(nm, Ysh, fi)
    ql = {f"s{s}": q1_block(reps[f"tr{s}"], reps[f"ri{s}"], TAC_l, ctx) for s in (0, 1)}
    extra = {}
    for sd in SHUF_SEEDS_EXTRA:
        T2, _ = fit_tac(nm, label_shuffle(Y, cid, fi, sd), fi)
        extra[f"seed{sd}"] = {f"s{s}": V.metric("CC32", reps[f"tr{s}"], T2, ctx)[0] - V.metric("CC32", reps[f"ri{s}"], T2, ctx)[0] for s in (0, 1)}
    ctl["C-LABEL-SHUF"] = {"seed": SHUF_SEED_MAIN, "d_shuffled_TAC": d_l, "Q1_shuffled": ql, "extra_seed_points_only": extra, "pass": bool(not any(v["separated_positive"] for v in ql.values())),
                           "two_sided_non_separated": bool(not any(v["separated_any"] for v in ql.values()))}
    out["controls"] = ctl
    out["controls_all_pass"] = bool(all(v["pass"] for v in ctl.values()))
    flags = [q1[f"s{s}"]["separated_positive"] for s in (0, 1)]
    out["outcome"] = decide(flags, out["controls_all_pass"])
    return out


def write_md(R, path):
    L = [f"# RESULT_PRH_TAC ({R['meta']['utc']} UTC)", "", f"**Outcome: {R['outcome']}** (controls all pass: {R['controls_all_pass']})", ""]
    if not R["controls_all_pass"]:
        L += ["**A CONTROL FAILED: the read is unusable (instrument failure); do not interpret the outcome.**", ""]
    n = R["n"]
    L += ["Evidence class: MEASURED, ours; artifact `RESULT_PRH_TAC.json` beside this file. Representation probe, no driving tier.", "",
          f"d_TAC = {R['d_TAC']} (31-ish columns after constant drop: {R['d_TAC_columns_after_constant_drop']}; CC dims used {R['cc_dims_used']}); frames: fit {n['fit_frames']}, score {n['score_frames']} (of {n['frames_ok_v2']} ok frames); score clips with frames {n['score_clips_with_frames']} / bootstrapped {n['score_clips_bootstrapped']}; B {n['B']}.", "",
          "## Q1 (CC32, 95 % episode-cluster bootstrap, paired)", "", "| seed | S(TR,TAC) | S(RI,TAC) | diff | CI95 | separated, positive |", "|---|---|---|---|---|---|"]
    for k, v in R["Q1"].items():
        L.append(f"| {k} | {v['S_TR']:.4f} | {v['S_RI']:.4f} | {v['diff']:.4f} | [{v['diff_ci95'][0]:.4f}, {v['diff_ci95'][1]:.4f}] | {v['separated_positive']} |")
    L += ["", "## Beside rows (no bar)", "", "```json", json.dumps(R["beside"], indent=1), "```", "", "## Controls", "", "```json", json.dumps(R["controls"], indent=1), "```", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="F:/TanitAD-scratch/otband_read/tree")
    ap.add_argument("--root", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/cache")
    ap.add_argument("--tracks", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt")
    ap.add_argument("--ckpt0", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt")
    ap.add_argument("--ckpt1", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s1/ckpt_last.pt")
    ap.add_argument("--work", default="F:/TanitAD-scratch/prh_seed/work")
    ap.add_argument("--join", default=JOIN)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    X = P.extract(a)
    t1 = time.time()
    with np.load(a.join, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    R = run(X, J)
    R["choices"] = CHOICES
    R["meta"] = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "card_sha256": CARD_SHA, "wall_s": time.time() - t0, "analysis_wall_s": time.time() - t1,
                 "join_sha256": sha_file(a.join), "calib_sha256": sha_file(CALIB), "ckpt_sha256": [P.sha_file(a.ckpt0), P.sha_file(a.ckpt1)]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    write_md(R, a.md)
    print(R["outcome"], "controls_all_pass", R["controls_all_pass"], "d_TAC", R["d_TAC"])


if __name__ == "__main__":
    main()
