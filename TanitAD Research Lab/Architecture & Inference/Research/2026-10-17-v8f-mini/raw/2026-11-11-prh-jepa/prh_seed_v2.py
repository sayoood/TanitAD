"""PRH_SEED v2 (PREREG_PRH_SEED_V2.md, sha256 861f0a05...): Step A calibrates the instrument on controls only, Step B reads the arms once with the chosen metric.

  python prh_seed_v2.py calib --calib CALIB_PRH_SEED_V2.json      # Step A: touches only nm, TR0 and fresh random affine maps (R_a seed 200, R_b seed 201)
  python prh_seed_v2.py read  --calib CALIB_PRH_SEED_V2.json --out RESULT_PRH_SEED_V2.json --md RESULT_PRH_SEED_V2.md   # Step B

Reuses v1 (prh_seed.py): extraction, whitener, fit_cca, split, bootstrap counts (rng seed 0), SLOW/PCA construction, beside rows.
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

CHOICES_V2 = {
    "random_affine": "R_x = Linear(1024,256) weights/bias from the adapter's own init (torch.nn.Linear default, after nn.LayerNorm(1024) built first, as in V8FMini.adapter) under torch.manual_seed(seed), applied to nm via the module's own pooled formula lin(nm * ln.weight + ln.bias); then scaled so its total fit-half variance equals TR0's (so MIX mixes comparable scales).",
    "PROC_whitening_ridge": "1e-3 x mean eigenvalue (as v1 beside rows); R fit on fit half; R^2 scored on the score half, centred by the fit-half mean.",
    "CC_ridge": "1e-6 x mean eigenvalue (as v1 M); CCk = mean of the first k canonical correlations in fit-half order, evaluated on the score half.",
    "CKA": "linear CKA on the score half, un-whitened reps, centred by score-half means (inside each bootstrap resample).",
    "SHUF": "second element permuted (rng seed 5) across clips within fit half and within score half; all fits redone on the permuted data.",
    "separated": "95 % percentile CI of the paired difference (same resample counts) with lower bound > 0 (admissibility rule 3); D-MIX2 separated = CI excludes 0 in either sign.",
    "choice_ratio": "(MIX(0.5) - FLOOR) point / half-width of the paired-difference CI of that difference; tie = within 10 % of the best ratio -> PROC if admissible and tied.",
    "C-ZC": "re-run against 3 clips per seed re-pulled by sftp (read only) in this run, same code path as v1.",
}
METRICS = ("CC8", "CC32", "PROC", "CKA")


# ------------------------------------------------------------------------------ metric stats + bootstrap
def _ctx(X, B=P.B_BOOT):
    C = len(X["clips"])
    ok = X["ok"].astype(bool)
    cid, tt = X["cid"], X["t"]
    fit_clip = np.arange(C) % 2 == 0
    is_fit = fit_clip[cid]
    fit_idx = np.where(ok & is_fit)[0]
    score_idx = np.where(ok & ~is_fit)[0]
    score_clips = np.where(~fit_clip)[0]
    remap = -np.ones(C, int)
    remap[score_clips] = np.arange(len(score_clips))
    nC = len(score_clips)
    counts = np.random.default_rng(0).multinomial(nC, np.ones(nC) / nC, size=B).astype(np.float64)
    return {"C": C, "ok": ok, "cid": cid, "t": tt, "is_fit": is_fit, "fit_idx": fit_idx, "score_idx": score_idx, "cid_score": remap[cid[score_idx]], "nC": nC, "counts": counts,
            "sub_idx": score_idx[(tt[score_idx] % 5) == 0], "B": B}


def metric(name, A, Bm, ctx):
    """(point, boot[B]) of metric ``name`` for the pair (A, Bm) [n, 256] over the score half, fit on the fit half."""
    fi, si, cs, nC, counts = ctx["fit_idx"], ctx["score_idx"], ctx["cid_score"], ctx["nC"], ctx["counts"]
    ones = np.ones((1, nC))
    if name in ("CC8", "CC32"):
        k = 8 if name == "CC8" else 32
        f = P.fit_cca(A[fi], Bm[fi], P.RIDGE_M)
        a, b = (A[si] - f["ma"]) @ f["Pa"][:, :k], (Bm[si] - f["mb"]) @ f["Pb"][:, :k]
        st = P.clip_stats(a, b, cs, nC)
        return float(P.corr_from_stats(st, ones)[0].mean()), P.corr_from_stats(st, counts).mean(1)
    if name == "PROC":
        ma, mb = A[fi].mean(0), Bm[fi].mean(0)
        Wa = P.whitener(np.cov((A[fi] - ma).T, bias=True), P.RIDGE_B)
        Wb = P.whitener(np.cov((Bm[fi] - mb).T, bias=True), P.RIDGE_B)
        U, _, Vt = np.linalg.svd(((Bm[fi] - mb) @ Wb).T @ ((A[fi] - ma) @ Wa))
        R = U @ Vt
        Za, Zb = (A[si] - ma) @ Wa, (Bm[si] - mb) @ Wb @ R
        num, den = np.zeros(nC), np.zeros(nC)
        np.add.at(num, cs, ((Za - Zb) ** 2).sum(1))
        np.add.at(den, cs, (Za ** 2).sum(1))
        return float(1 - num.sum() / den.sum()), 1 - (counts @ num) / (counts @ den)
    if name == "CKA":
        x, y = A[si], Bm[si]
        d1, d2 = x.shape[1], y.shape[1]
        n = np.bincount(cs, minlength=nC).astype(float)
        Sx, Sy = np.zeros((nC, d1)), np.zeros((nC, d2))
        Sxx, Syy, Sxy = np.zeros((nC, d1 * d1)), np.zeros((nC, d2 * d2)), np.zeros((nC, d1 * d2))
        for c in range(nC):
            m = cs == c
            xc, yc = x[m], y[m]
            Sx[c], Sy[c] = xc.sum(0), yc.sum(0)
            Sxx[c], Syy[c], Sxy[c] = (xc.T @ xc).ravel(), (yc.T @ yc).ravel(), (xc.T @ yc).ravel()

        def cka(cnt):
            out = np.empty(len(cnt))
            for s in range(0, len(cnt), 250):
                c = cnt[s:s + 250]
                nn = c @ n
                mx, my = (c @ Sx) / nn[:, None], (c @ Sy) / nn[:, None]
                Cxx = ((c @ Sxx) / nn[:, None]).reshape(-1, d1, d1) - mx[:, :, None] * mx[:, None, :]
                Cyy = ((c @ Syy) / nn[:, None]).reshape(-1, d2, d2) - my[:, :, None] * my[:, None, :]
                Cxy = ((c @ Sxy) / nn[:, None]).reshape(-1, d1, d2) - mx[:, :, None] * my[:, None, :]
                out[s:s + 250] = (Cxy ** 2).sum((1, 2)) / np.sqrt((Cxx ** 2).sum((1, 2)) * (Cyy ** 2).sum((1, 2)))
            return out
        return float(cka(ones)[0]), cka(counts)
    raise KeyError(name)


# ------------------------------------------------------------------------------ rules (mutation-checked)
def admissible(v):
    """v: dict with SELF, FLOOR, MIX25, MIX50, MIX75, SHUF (points) and 'diff50_ci' = [lo, hi] of MIX50 - FLOOR."""
    r = {"1_self": abs(v["SELF"] - 1) <= 1e-6,
         "2_mix_monotone": v["MIX25"] > v["MIX50"] > v["MIX75"],
         "3_mix50_minus_floor_separated": bool(P.is_sep_pos(*v["diff50_ci"])),
         "4_shuf_below_floor": v["SHUF"] < v["FLOOR"]}
    return r, all(r.values())


def choose(table):
    """table: {metric: {'admissible': bool, 'ratio': float}} -> chosen metric or None (largest ratio; ties within 10 % go to PROC)."""
    adm = {k: v["ratio"] for k, v in table.items() if v["admissible"]}
    if not adm:
        return None
    best = max(adm.values())
    if "PROC" in adm and adm["PROC"] >= 0.9 * best:
        return "PROC"
    return max(adm, key=adm.get)


# ------------------------------------------------------------------------------ random affine maps
def rand_affine(nm, seed, ref, fit_idx):
    import torch
    torch.manual_seed(seed)
    ln, lin = torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 256)
    with torch.no_grad():
        r = lin(torch.as_tensor(nm, dtype=torch.float32) * ln.weight + ln.bias).numpy().astype(np.float64)
    return r * np.sqrt(ref[fit_idx].var(0).sum() / r[fit_idx].var(0).sum())


def permuted(Rm, ctx, seed=5):
    out = Rm.copy()
    pr = np.random.default_rng(seed)
    for idxs in (ctx["fit_idx"], ctx["score_idx"]):
        out[idxs] = Rm[pr.permutation(idxs)]
    return out


def ci95_diff(x, y):
    return P.ci95(x - y)


# ------------------------------------------------------------------------------ Step A
def step_a(X, ctx):
    nm = X["nm"].astype(np.float64)
    tr0 = X["tr0"].astype(np.float64)
    Ra = rand_affine(nm, 200, tr0, ctx["fit_idx"])
    Rb = rand_affine(nm, 201, tr0, ctx["fit_idx"])
    pairs = {"SELF": (tr0, tr0), "FLOOR": (Ra, Rb), "MIX25": (tr0, 0.75 * tr0 + 0.25 * Ra), "MIX50": (tr0, 0.5 * tr0 + 0.5 * Ra), "MIX75": (tr0, 0.25 * tr0 + 0.75 * Ra),
             "SHUF": (tr0, permuted(Ra, ctx))}
    table, tab_out = {}, {}
    for m in METRICS:
        res = {k: metric(m, A, B, ctx) for k, (A, B) in pairs.items()}
        d_pt = res["MIX50"][0] - res["FLOOR"][0]
        lo, hi = ci95_diff(res["MIX50"][1], res["FLOOR"][1])
        vals = {k: res[k][0] for k in pairs}
        vals["diff50_ci"] = [lo, hi]
        rules, adm = admissible(vals)
        ratio = d_pt / ((hi - lo) / 2)
        table[m] = {"admissible": bool(adm), "ratio": float(ratio)}
        tab_out[m] = {"values": {k: {"point": res[k][0], "ci95": P.ci95(res[k][1])} for k in pairs}, "MIX50_minus_FLOOR": {"point": d_pt, "ci95": [lo, hi]},
                      "rules": {k: bool(v) for k, v in rules.items()}, "admissible": bool(adm), "ratio": float(ratio)}
    ch = choose(table)
    return {"card": "PREREG_PRH_SEED_V2.md", "card_sha256": "861f0a05e19518903e367da64b73e6ec1333592b6d51f49a9798f01f1a03814d", "metrics": tab_out, "chosen": ch,
            "status": "INSTRUMENT-NOT-FOUND" if ch is None else "CHOSEN", "choices": CHOICES_V2, "calib_representations": "R_a seed 200, R_b seed 201 (R_c seed 300 reserved for Step B)"}


# ------------------------------------------------------------------------------ Step B
def step_b(X, ctx, S, a_cli):
    nm = X["nm"].astype(np.float64)
    reps = {k: X[k].astype(np.float64) for k in ("tr0", "tr1", "ri0", "ri1")}
    fi, si = ctx["fit_idx"], ctx["score_idx"]
    cid, ok, is_fit, n_all = ctx["cid"], ctx["ok"], ctx["is_fit"], len(nm)
    nxt = np.arange(n_all) + P.LAG
    valid = nxt < n_all
    valid[valid] &= (cid[nxt[valid]] == cid[valid.nonzero()[0]]) & ok[nxt[valid]]
    pair_idx = np.where(valid & ok & is_fit)[0]
    fs = P.fit_cca(nm[pair_idx], nm[pair_idx + P.LAG], P.RIDGE_B, k=P.K)
    SLOW = (nm - fs["ma"]) @ fs["Pa"]
    mpca = nm[fi].mean(0)
    w, V = np.linalg.eigh(np.cov((nm[fi] - mpca).T, bias=True))
    PCA = (nm - mpca) @ V[:, ::-1][:, :P.K]
    Rc = rand_affine(nm, 300, reps["tr0"], fi)
    DMIX2 = 0.5 * reps["tr0"] + 0.5 * Rc

    def mt(A, B_):
        return metric(S, A, B_, ctx)
    Mv = {"S(TR0,TR1)": mt(reps["tr0"], reps["tr1"]), "S(RI0,RI1)": mt(reps["ri0"], reps["ri1"]),
          "S(TR0,SLOW)": mt(reps["tr0"], SLOW), "S(TR1,SLOW)": mt(reps["tr1"], SLOW), "S(RI0,SLOW)": mt(reps["ri0"], SLOW), "S(RI1,SLOW)": mt(reps["ri1"], SLOW),
          "S(TR0,PCA)": mt(reps["tr0"], PCA), "S(TR1,PCA)": mt(reps["tr1"], PCA), "S(TR0,DMIX2)": mt(reps["tr0"], DMIX2), "S(TR0,TR0)": mt(reps["tr0"], reps["tr0"])}
    out = {"metric": S, "M": {k: {"point": v[0], "ci95": P.ci95(v[1])} for k, v in Mv.items()}}

    def diff(x, y):
        d = Mv[x][1] - Mv[y][1]
        lo, hi = P.ci95(d)
        return {"point": Mv[x][0] - Mv[y][0], "ci95": [lo, hi], "separated_positive": bool(P.is_sep_pos(lo, hi))}
    D_ = {"P1: S(TR0,TR1)-S(RI0,RI1)": diff("S(TR0,TR1)", "S(RI0,RI1)")}
    for s in (0, 1):
        D_[f"P2a_s{s}: S(TR{s},SLOW)-S(RI{s},SLOW)"] = diff(f"S(TR{s},SLOW)", f"S(RI{s},SLOW)")
        D_[f"P2b_s{s}: S(TR{s},SLOW)-S(TR{s},PCA)"] = diff(f"S(TR{s},SLOW)", f"S(TR{s},PCA)")
    out["differences"] = D_
    p1 = D_["P1: S(TR0,TR1)-S(RI0,RI1)"]["separated_positive"]
    p2 = all(v["separated_positive"] for k, v in D_.items() if k.startswith("P2"))
    out["P1"], out["P2"], out["outcome"] = bool(p1), bool(p2), P.outcome(p1, p2)

    ctl = {}
    ctl["C-SELF"] = {"S": Mv["S(TR0,TR0)"][0], "pass": bool(abs(Mv["S(TR0,TR0)"][0] - 1) <= 1e-6)}
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((P.K, P.K)))
    s_rot = metric(S, reps["tr0"], reps["tr1"] @ Q, ctx)[0]
    ctl["C-ROT"] = {"S_rotated": s_rot, "S_plain": Mv["S(TR0,TR1)"][0], "absdiff": abs(s_rot - Mv["S(TR0,TR1)"][0]), "pass": bool(abs(s_rot - Mv["S(TR0,TR1)"][0]) <= 1e-6)}
    perm = permuted(reps["tr1"], ctx)
    sh_pt, sh_boot = mt(reps["tr0"], perm)
    sh_b = P.beside(reps["tr0"], perm, fi, si, ctx["sub_idx"], SLOW)
    ctl["C-SHUF"] = {"S": sh_pt, "ci95": P.ci95(sh_boot), "S(RI0,RI1)": Mv["S(RI0,RI1)"][0], "mutual_10NN": sh_b["mutual_10NN"], "chance": sh_b["chance_10NN"],
                     "pass": bool(sh_pt < Mv["S(RI0,RI1)"][0] and sh_b["mutual_10NN"] <= 2 * sh_b["chance_10NN"])}
    d = Mv["S(TR0,DMIX2)"][1] - Mv["S(TR0,TR1)"][1]
    lo, hi = P.ci95(d)
    ctl["D-MIX2"] = {"S(TR0,DMIX2)": Mv["S(TR0,DMIX2)"][0], "S(TR0,TR1)": Mv["S(TR0,TR1)"][0], "delta": Mv["S(TR0,DMIX2)"][0] - Mv["S(TR0,TR1)"][0], "ci95": [lo, hi], "pass": bool(P.is_sep_any(lo, hi))}
    ctl["C-ZC"] = P.c_zc(X, a_cli)
    out["controls"] = ctl
    out["controls_all_pass"] = bool(all(v["pass"] for v in ctl.values()))
    out["beside"] = {"TR0_vs_TR1": P.beside(reps["tr0"], reps["tr1"], fi, si, ctx["sub_idx"], SLOW), "RI0_vs_RI1": P.beside(reps["ri0"], reps["ri1"], fi, si, ctx["sub_idx"], SLOW)}
    out["n"] = {"fit_frames": int(len(fi)), "score_frames": int(len(si)), "score_clips": ctx["nC"], "slow_pairs": int(len(pair_idx)), "B": ctx["B"]}
    return out


def write_md(R, path):
    L = [f"# RESULT_PRH_SEED_V2 ({R['meta']['utc']} UTC)", "", f"**Outcome: {R['outcome']}** with metric {R['metric']} (P1={R['P1']}, P2={R['P2']}; controls all pass: {R['controls_all_pass']})", ""]
    if not R["controls_all_pass"]:
        L += ["**A CONTROL FAILED: the read is unusable (instrument failure); do not interpret the outcome.**", ""]
    L += ["Evidence class: MEASURED, ours; artifact `RESULT_PRH_SEED_V2.json`. Calibration: `CALIB_PRH_SEED_V2.json` sha256 " + R["meta"]["calib_sha256"], "",
          "| S | point | CI95 |", "|---|---|---|"]
    L += [f"| {k} | {v['point']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |" for k, v in R["M"].items()]
    L += ["", "| diff | point | CI95 | separated, positive |", "|---|---|---|---|"]
    L += [f"| {k} | {v['point']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['separated_positive']} |" for k, v in R["differences"].items()]
    L += ["", "## Controls", "", "```json", json.dumps(R["controls"], indent=1), "```", "", "## Beside (no bar)", ""]
    for k, v in R["beside"].items():
        p = np.array(v["profile_along_SLOW"])
        L.append(f"* {k}: Procrustes R2 {v['procrustes_R2']:.3f}, mutual 10-NN {v['mutual_10NN']:.3f} (chance {v['chance_10NN']:.4f}); profile along SLOW mean dirs 1-16 {p[:16].mean():.3f}, 17-64 {p[16:64].mean():.3f}, 65-256 {p[64:].mean():.3f} (256 values in the JSON)")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "read"])
    ap.add_argument("--tree", default="F:/TanitAD-scratch/otband_read/tree")
    ap.add_argument("--root", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/cache")
    ap.add_argument("--tracks", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt")
    ap.add_argument("--ckpt0", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt")
    ap.add_argument("--ckpt1", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s1/ckpt_last.pt")
    ap.add_argument("--work", default="F:/TanitAD-scratch/prh_seed/work")
    ap.add_argument("--zc-dir", default="F:/TanitAD-scratch/prh_seed/work/zc")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out")
    ap.add_argument("--md")
    a = ap.parse_args()
    t0 = time.time()
    X = P.extract(a)
    ctx = _ctx(X)
    if a.mode == "calib":
        # Step A: only nm, tr0, ok/cid/t/clips are read below; no TR1 / RI / SLOW / PCA statistic exists in this code path.
        Xa = {k: X[k] for k in ("nm", "tr0", "ok", "cid", "t", "clips")}
        R = step_a(Xa, _ctx(Xa))
        R["wall_s"] = time.time() - t0
        with open(a.calib, "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=1)
        print("CALIB_SHA256", sha_file(a.calib))
        print("STATUS", R["status"], "CHOSEN", R["chosen"])
        for m, v in R["metrics"].items():
            print(m, "admissible", v["admissible"], "ratio", round(v["ratio"], 2), {k: round(x["point"], 4) for k, x in v["values"].items()}, "rules", v["rules"])
        return
    cal = json.load(open(a.calib, encoding="utf-8"))
    if cal["chosen"] is None:
        print("INSTRUMENT-NOT-FOUND")
        return
    R = step_b(X, ctx, cal["chosen"], a)
    R["meta"] = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "calib_sha256": sha_file(a.calib), "card_sha256": cal["card_sha256"], "wall_s": time.time() - t0,
                 "ckpt_sha256": [P.sha_file(a.ckpt0), P.sha_file(a.ckpt1)]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    write_md(R, a.md)
    print(R["outcome"], "controls_all_pass", R["controls_all_pass"], "metric", R["metric"])


if __name__ == "__main__":
    main()
