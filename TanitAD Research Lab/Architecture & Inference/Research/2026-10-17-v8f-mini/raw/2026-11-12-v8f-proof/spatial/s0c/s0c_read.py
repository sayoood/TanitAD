"""S0c read (PREREG_V8F_PROOF_ADD12 note N5, stage 0c: the read of the CHANGE of the lead gap; choices in S0C_DEFS.md, rules C1..C16).
Tier T0: a world-model diagnostic, never driving performance. CPU only. Reuses spatial/s0/s0_read.py by import (loader, states, probe, folds, bootstrap helpers).
Subcommands:
    run     full read on the true labels -> RESULT_S0C.json, VERDICT_S0C.md (resumable per fold, stages/); exit code 2 = a control missed
    smoke   12 episodes, lead-gap labels permuted across rows BEFORE pairs are formed -> smoke/ (shapes, counts, timings, k2 and k3 only)
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
S0_DIR = "F:/TanitAD-scratch/v8proof/spatial/s0"
if S0_DIR not in sys.path:
    sys.path.append(S0_DIR)
import s0_read as S0  # noqa: E402
RP = S0.RP

FWD, BACK = 10, 5                     # pair geometry: (g, g+10) with a bank row at g-5 (cache steps of 0.2 s)
SEED = S0.SEED
B_BOOT = S0.B_BOOT
ARMS = ("H-g", "H-m", "N-g", "E")
TIER, STATEMENT, FAMILIES = S0.TIER, S0.STATEMENT, S0.FAMILIES
K4_REF = {"grid_gap_r2": 0.3981, "ego_gap_r2": 0.1327}
K1_BAR = 0.02
PAIR_RULE = "continuous"         # note N6: "continuous" = stage 0d; "all" = stage 0c
STEP_MAX_M = 3.0
K2_BAR_HI, K2_BAR_LO = 0.95, 0.60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ===================================================================================== pairs (C1)
def build_pairs(beid, bg, gap, fwd: int = FWD, back: int = BACK):
    """Pair index over the bank rows: row i = (clip, g); rk = row at (clip, g+fwd); rm = row at (clip, g-back). A pair needs all three rows
    and a finite lead gap at i and at rk. Returns int arrays (ri, rm, rk) in row order plus a counts dict."""
    pos = {(str(e), int(g)): i for i, (e, g) in enumerate(zip(beid, bg))}
    ri, rm, rk, raw, broken = [], [], [], 0, 0
    for i, (e, g) in enumerate(zip(beid, bg)):
        j = pos.get((str(e), int(g) + fwd), -1)
        m = pos.get((str(e), int(g) - back), -1)
        if j < 0 or m < 0:
            continue
        raw += 1
        if np.isfinite(gap[i]) and np.isfinite(gap[j]):
            if PAIR_RULE == "continuous":            # note N6 (stage 0d): a finite lead gap on every frame g .. g+fwd and no 0.2 s step above STEP_MAX_M
                seq = [pos.get((str(e), int(g) + q), -1) for q in range(fwd + 1)]
                gs = np.array([gap[q] if q >= 0 else np.nan for q in seq], np.float64)
                if not np.all(np.isfinite(gs)) or float(np.abs(np.diff(gs)).max()) > STEP_MAX_M:
                    broken += 1
                    continue
            ri.append(i); rm.append(m); rk.append(j)
    cnt = {"pairs_with_three_rows": raw, "pairs_used": len(ri), "pair_rule": PAIR_RULE, "pairs_removed_by_the_continuity_rule": broken, "step_max_m": STEP_MAX_M}
    return np.array(ri, np.int64), np.array(rm, np.int64), np.array(rk, np.int64), cnt


def ego_block(E, ri, rm, rk):
    """The ego vector at g, at g-5 and at g+10, 24 numbers per pair."""
    return np.concatenate([E[ri], E[rm], E[rk]], 1)


# ===================================================================================== skill and its bootstrap (C6, C7)
def skill(d, p):
    """1 - sum (d - p)^2 / sum d^2 over finite pairs; persistence (p = 0) gives exactly 0."""
    m = np.isfinite(d) & np.isfinite(p)
    return float(1.0 - ((d[m] - p[m]) ** 2).sum() / (d[m] ** 2).sum())


def ep_stats_skill(ep, d, p, n_ep):
    """Per-episode sufficient statistics [n_ep, 3]: n, sum d^2, SSE (finite pairs only)."""
    m = np.isfinite(d) & np.isfinite(p)
    S = np.zeros((n_ep, 3))
    e, dd, pp = ep[m], d[m], p[m]
    np.add.at(S[:, 0], e, 1.0); np.add.at(S[:, 1], e, dd * dd); np.add.at(S[:, 2], e, (dd - pp) ** 2)
    return S


def boot_skill(S, C):
    t = C @ S
    return 1.0 - t[:, 2] / t[:, 1]


def single_skill(ep, d, p, C, n_ep):
    S = ep_stats_skill(ep, d, p, n_ep)
    r2 = S0.single_r2(ep, d, p, C, n_ep)
    return {"skill": skill(d, p), "ci95": RP.ci(boot_skill(S, C)), "r2": r2["r2"], "r2_ci95": r2["ci95"], "n_pairs": int((np.isfinite(d) & np.isfinite(p)).sum()),
            "n_episodes": int(len(np.unique(ep[np.isfinite(d) & np.isfinite(p)])))}


def paired_skill(ep, d, pa, pb, C, n_ep):
    Sa, Sb = ep_stats_skill(ep, d, pa, n_ep), ep_stats_skill(ep, d, pb, n_ep)
    return RP.diff_row(boot_skill(Sa, C), boot_skill(Sb, C), skill(d, pa), skill(d, pb))


# ===================================================================================== arms (C4, C5)
def arm_features(arm, St, Eb, ri, rm, tr_pairs):
    """Feature matrix [P, *] of an arm. St = {'g': [N,640], 'm': [N,128]} states of ALL bank rows of this fold (PCA fitted on training-fold rows).
    The ego block is z-scored on the training pairs; when a state is present it is multiplied by the square root of the mean column variance of that
    state on the training pairs (the S0 R6 scaling), so that no block dominates the global scale of fit_probe."""
    mu, sd = Eb[tr_pairs].mean(0), Eb[tr_pairs].std(0) + 1e-9
    z = (Eb - mu) / sd
    if arm == "E":
        return z
    s = arm.split("-")[1]
    S = St[s]
    sc = float(np.sqrt(S[ri[tr_pairs]].var(0).mean()) + 1e-12)
    parts = [S[ri]]
    if arm.startswith("H"):
        parts.append(S[ri] - S[rm])
    parts.append(z * sc)
    return np.concatenate(parts, 1)


def arm_oof_fold(F, d, ep, trp, tep, f):
    """Fit the stage-0 probe on the training-fold pairs, predict the test-fold pairs. Returns predictions [len(tep)]."""
    fit = S0.fit_probe(F[trp], d[trp], ep[trp], SEED + f)
    return fit(F[tep])


# ===================================================================================== k2: planted moving object (C12)
def k2_setup(D, seed_cell: int = 8, seed_side: int = 9):
    """Label free. Cell: a row of the 4x10 grid and an interior column 1..8 (never a border cell), so both neighbours exist in the same grid row.
    Vector: the stage-0 f5 vector (unit direction of seed 5 times the median token norm of planted_setup's field median_token_norm).
    side[p] = 0 (g-5 object in the left neighbour c-1) or 1 (right neighbour c+1), a random half each."""
    pl = S0.planted_setup(D)
    rng = np.random.default_rng(seed_cell)
    ncell = D["X"].shape[1]
    rows = max(ncell // 10, 1)
    r = int(rng.integers(0, rows)); col = int(rng.integers(1, 9))
    return {"cell": r * 10 + col, "vec": pl["vec_f5"].astype(np.float32), "median_token_norm": pl["median_token_norm"], "seed_side": seed_side,
            "grid_row": r, "grid_col": col}


def k2_side(n_pairs: int, seed_side: int = 9):
    return (np.random.default_rng(seed_side).random(n_pairs) < 0.5).astype(np.float64)


def k2_fold(X, M, St, trp, tep, ri, rm, spec, side, ep, f):
    """Planted moving object, one fold. In every pair the vector sits in cell c at frame g; at frame g-5 it sits in c-1 (side 0) or c+1 (side 1).
    PCA of the three touched cells (on the stacked frames g and g-5 of the training pairs, planted tokens included) and of the mean token (same
    stack, the mean token carrying vec / ncell at both frames) are refitted on the training pairs only. Returns OOF scores of three probes."""
    c, vec = spec["cell"], spec["vec"]
    P, ncell = len(ri), X.shape[1]
    fsel = np.arange(P) if S0.LEAK_PCA_FIT else trp
    Sg_g, Sg_m = St["g"][ri].copy(), St["g"][rm].copy()
    for t in (c - 1, c, c + 1):
        add_g = vec[None, :] * (1.0 if t == c else 0.0)
        flag_m = (side == 0) if t == c - 1 else ((side == 1) if t == c + 1 else np.zeros(P, bool))
        add_m = flag_m[:, None].astype(np.float32) * vec[None, :]
        xg = X[ri, t].astype(np.float32) + add_g
        xm = X[rm, t].astype(np.float32) + add_m
        mu, V = S0.pca_basis(np.concatenate([xg[fsel], xm[fsel]]), S0.CELLD)
        mu32, V32 = mu.astype(np.float32), V.astype(np.float32)
        Sg_g[:, t * S0.CELLD:(t + 1) * S0.CELLD] = (xg - mu32) @ V32
        Sg_m[:, t * S0.CELLD:(t + 1) * S0.CELLD] = (xm - mu32) @ V32
    Mg, Mm = M[ri] + vec[None, :] / ncell, M[rm] + vec[None, :] / ncell
    mu, V = S0.pca_basis(np.concatenate([Mg[fsel], Mm[fsel]]), S0.MEAN_D)
    mu32, V32 = mu.astype(np.float32), V.astype(np.float32)
    Mg_s, Mm_s = (Mg - mu32) @ V32, (Mm - mu32) @ V32
    feats = {"grid2": np.concatenate([Sg_g, Sg_g - Sg_m], 1), "mean2": np.concatenate([Mg_s, Mg_s - Mm_s], 1), "grid1": Sg_g}
    out = {}
    for k, F in feats.items():
        out[k] = arm_oof_fold(F, side, ep, trp, tep, f)
    return out


# ===================================================================================== k3: alignment (C13)
def check_k3(raw_beid, raw_bg, raw_E, raw_gap, beid, bg, ri, rm, rk, Eb, gap_i, gap_k, compare_gap: bool = True):
    """Every pair is re-derived from the RAW banked arrays by (clip, frame) key, independently of the row indices used to build it. Exact equality."""
    key = {(str(e), int(g)): j for j, (e, g) in enumerate(zip(raw_beid, raw_bg))}
    bad_idx = bad_ego = bad_gap = 0
    for p in range(len(ri)):
        e, g = str(beid[ri[p]]), int(bg[ri[p]])
        if not (str(beid[rm[p]]) == e and str(beid[rk[p]]) == e and int(bg[rk[p]]) == g + FWD and int(bg[rm[p]]) == g - BACK):
            bad_idx += 1
            continue
        j0, jm, jk = key.get((e, g), -1), key.get((e, g - BACK), -1), key.get((e, g + FWD), -1)
        if min(j0, jm, jk) < 0:
            bad_idx += 1
            continue
        if not np.array_equal(Eb[p], np.concatenate([raw_E[j0], raw_E[jm], raw_E[jk]])):
            bad_ego += 1
        if compare_gap and not (gap_i[p] == raw_gap[j0] and gap_k[p] == raw_gap[jk]):
            bad_gap += 1
    return {"pass": bool(len(ri) > 0 and bad_idx == 0 and bad_ego == 0 and bad_gap == 0), "n_pairs": int(len(ri)), "bad_index": bad_idx, "bad_ego": bad_ego,
            "bad_gap": bad_gap, "gaps_compared": bool(compare_gap), "bar": "all counts exactly 0; arrays compared with array_equal"}


def check_tokens(D, ri, n: int = 30, seed: int = 3):
    """k3, token side: for up to n random pairs the stored tokens of the row at g equal the cache row g of the clip file (fp8 -> float16, exact)."""
    import torch
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ri), min(n, len(ri)), replace=False)
    bad = 0
    for p in pick:
        i = int(ri[p]); c, g = str(D["beid"][i]), int(D["bg"][i])
        o = torch.load(f"{S0.CLIPS}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        ref = o["pool4x10"][g].to(torch.float32).numpy().astype(np.float16)
        if not np.array_equal(ref, D["X"][i]):
            bad += 1
        del o
    return {"n_checked": int(len(pick)), "bad": bad}


# ===================================================================================== controls
def check_k1(skills: dict, bar: float = K1_BAR):
    mx = max(skills.values())
    return {"pass": bool(all(v <= bar for v in skills.values())), "max_skill": float(mx), "bar": bar, "values": skills}


def check_k2(auc_grid2: float, auc_mean2: float, auc_grid1: float):
    ok = bool(auc_grid2 > K2_BAR_HI and auc_mean2 <= K2_BAR_LO and auc_grid1 <= K2_BAR_LO)
    return {"pass": ok, "auc_grid_two_frames": float(auc_grid2), "auc_mean_two_frames": float(auc_mean2), "auc_grid_one_frame": float(auc_grid1),
            "bars": f"grid two frames > {K2_BAR_HI}; mean two frames <= {K2_BAR_LO}; grid one frame <= {K2_BAR_LO}"}


def check_k4(r2_grid: float, r2_ego: float):
    ok = bool(round(r2_grid, 4) == K4_REF["grid_gap_r2"] and round(r2_ego, 4) == K4_REF["ego_gap_r2"])
    return {"pass": ok, "grid_gap_r2": float(r2_grid), "ego_gap_r2": float(r2_ego), "reference": K4_REF, "bar": "equal to the reference at 4 decimals"}


# ===================================================================================== fold loop (C3)
def process_fold(f, D, M, Pr, dsets, folds_b, spec, side, do_k4, log_=log):
    t0 = time.time()
    X, ri, rm = D["X"], Pr["ri"], Pr["rm"]
    N, P = len(X), len(ri)
    fold_p = folds_b[ri]
    tr = np.flatnonzero(folds_b != f)
    trp, tep = np.flatnonzero(fold_p != f), np.flatnonzero(fold_p == f)
    St = {"g": S0.grid_state(X, tr), "m": S0.mean_state(M, tr)}
    log_(f"fold {f}: states g {St['g'].shape} m {St['m'].shape}; pairs train {len(trp)} test {len(tep)} ({time.time() - t0:.0f}s)")
    out = {}
    ep = Pr["ep"]
    for lab, d in dsets.items():
        for arm in ARMS:
            F = arm_features(arm, St, Pr["Eb"], ri, rm, trp)
            key = f"arm|{lab}|{arm}"
            out[key] = np.full(P, np.nan)
            if len(tep):
                out[key][tep] = arm_oof_fold(F, d, ep, trp, tep, f)
            log_(f"fold {f}: {lab} {arm} dim {F.shape[1]} ({time.time() - t0:.0f}s)")
    k2 = k2_fold(X, M, St, trp, tep, ri, rm, spec, side, ep, f)
    for k, v in k2.items():
        out[f"k2|{k}"] = np.full(P, np.nan)
        out[f"k2|{k}"][tep] = v
    log_(f"fold {f}: k2 done ({time.time() - t0:.0f}s)")
    if do_k4:                      # stage-0 r-a gap probe on S0g, exactly as S0.process_fold (training-fold bank rows with a finite gap)
        y = D["gap"]
        te = np.flatnonzero(folds_b == f)
        vtr, vte = tr[np.isfinite(y[tr])], te[np.isfinite(y[te])]
        out["k4|ra_g"] = np.full(N, np.nan)
        if len(vtr) >= 12 and len(vte):
            out["k4|ra_g"][vte] = S0.fit_probe(St["g"][vtr], y[vtr], D["epi"][vtr], SEED + f)(St["g"][vte])
        log_(f"fold {f}: k4 r-a probe done ({time.time() - t0:.0f}s)")
    return out


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


def compute_oof(D, M, Pr, dsets, folds_b, spec, side, do_k4, stage_dir=None, sig=None, log_=log):
    parts, times = [], {}
    for f in range(int(folds_b.max()) + 1):
        sp = f"{stage_dir}/fold{f}.npz" if stage_dir else None
        if sp and os.path.exists(sp):
            z = np.load(sp, allow_pickle=True)
            if json.loads(str(z["sig"])) == sig:
                parts.append({k: z[k] for k in z.files if k != "sig"}); log_(f"fold {f}: resumed"); continue
        t0 = time.time()
        o = process_fold(f, D, M, Pr, dsets, folds_b, spec, side, do_k4, log_)
        times[f"fold{f}"] = time.time() - t0
        if sp:
            np.savez(sp, sig=json.dumps(sig), **o)
        parts.append(o)
    return merge_oof(parts), times


# ===================================================================================== analysis (C6 to C10)
def outcome_names(skills: dict, diffs: dict):
    out = {}
    e_ok = skills["E"]["skill"] > 0 and skills["E"]["ci95"][0] > 0
    for s in ("g", "m"):
        dd = diffs[f"H-{s} minus E"]
        out[f"state_{s}"] = "S0-CHANGE-READ" if (dd["diff"] > 0 and dd["separated_pos"]) else ("S0-CHANGE-EGO" if e_ok else "S0-CHANGE-FLAT")
    out["pair"] = "GRID-ADDS-CHANGE" if (diffs["H-g minus H-m"]["diff"] > 0 and diffs["H-g minus H-m"]["separated_pos"]) else "GRID-EQUAL-CHANGE"
    out["history"] = "HISTORY-ADDS" if (diffs["H-g minus N-g"]["diff"] > 0 and diffs["H-g minus N-g"]["separated_pos"]) else "HISTORY-EQUAL"
    return out


def analyse_main(d, ep, oof, C, n_ep):
    sk = {a: single_skill(ep, d, oof[f"arm|main|{a}"], C, n_ep) for a in ARMS}
    pr = lambda a: oof[f"arm|main|{a}"]
    diffs = {"H-g minus E": paired_skill(ep, d, pr("H-g"), pr("E"), C, n_ep), "H-m minus E": paired_skill(ep, d, pr("H-m"), pr("E"), C, n_ep),
             "H-g minus H-m": paired_skill(ep, d, pr("H-g"), pr("H-m"), C, n_ep), "H-g minus N-g": paired_skill(ep, d, pr("H-g"), pr("N-g"), C, n_ep)}
    dstat = {"n_pairs": int(len(d)), "std_d_m": float(np.std(d)), "mean_d_m": float(np.mean(d)), "share_abs_d_above_2m": float((np.abs(d) > 2.0).mean())}
    return {"arms": sk, "diffs": diffs, "d_stats": dstat, "outcomes": outcome_names(sk, diffs)}


def ci_s(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def write_verdict(path, res, meta, smoke: bool):
    L = []
    ctr = res["controls"]
    L.append(f"# VERDICT_S0C{' (SMOKE: permuted labels)' if smoke else ''}: PREREG_V8F_PROOF_ADD12 note N5, stage 0c (the change of the lead gap, +2.0 s)")
    L.append(f"Tier {TIER}: {STATEMENT}. Estimator: paired episode-cluster bootstrap over {meta['n_episodes']} episodes, B {meta['B']}, seed 0, one draw shared by all arms. {FAMILIES}")
    L.append("A negative of stage 0c is a negative about a LINEAR read of two pooled frames on an untrained state; stage 1 still runs.")
    L.append("")
    L.append("## CONTROLS (first)")
    keys = ("k2", "k3") if smoke else [k for k in ("k1", "k2", "k3", "k4") if k in ctr]
    for k in keys:
        c = ctr[k]
        L.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} :: " + json.dumps({a: b for a, b in c.items() if a != 'pass'}, default=float))
    if not smoke:
        L.append(f"- controls_pass: {res['controls_pass']}")
    L.append("")
    if smoke:
        L.append("SMOKE: lead-gap labels permuted across rows before the pairs were formed; no outcome name and no skill is quoted. Shapes, counts and timings are in RESULT_S0C_SMOKE.json.")
    elif not res["controls_pass"]:
        L.append("## OUTCOME NAMES WITHHELD: a control did not read its known value. Numbers are in RESULT_S0C.json, not quotable.")
    else:
        m = res["main"]
        ds = m["d_stats"]
        L.append(f"## Target d = gap(g+10) - gap(g), metres: n pairs {ds['n_pairs']}, episodes {meta['n_episodes_with_pairs']}; std(d) {ds['std_d_m']:.3f} m, mean {ds['mean_d_m']:+.3f} m; share of pairs with |d| above 2 m {ds['share_abs_d_above_2m']:.3f} ({TIER}; {STATEMENT})")
        L.append("## Skill over persistence (1 - SSE / sum d^2; persistence = 0) and plain R2 beside")
        for a in ARMS:
            r = m["arms"][a]
            L.append(f"- {a}: skill {r['skill']:+.3f} {ci_s(r['ci95'])}; plain R2 {r['r2']:+.3f} {ci_s(r['r2_ci95'])}; n pairs {r['n_pairs']}, episodes {r['n_episodes']}")
        L.append("## Paired differences of skill (paired episode-cluster bootstrap)")
        for k, v in m["diffs"].items():
            L.append(f"- {k}: {v['diff']:+.3f} {ci_s(v['ci95'])}")
        L.append("## OUTCOME NAMES")
        o = m["outcomes"]
        L.append(f"- grid state (H-g against E): {o['state_g']}")
        L.append(f"- mean state (H-m against E): {o['state_m']}")
        L.append(f"- pair (H-g against H-m): {o['pair']}")
        L.append(f"- history (H-g against N-g): {o['history']}")
    L.append("")
    L.append("Counts: " + json.dumps(meta["counts"]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ===================================================================================== driver
def cmd_run(smoke: bool):
    t_all = time.time()
    out_dir = f"{HERE}/smoke" if smoke else HERE
    stage_dir = f"{out_dir}/stages"
    os.makedirs(stage_dir, exist_ok=True)
    B = 200 if smoke else B_BOOT
    D = S0.load_data(smoke)
    N = len(D["X"])
    t_load = time.time() - t_all
    Zr = np.load(S0.BANK)
    raw = {"beid": Zr["beid"].astype(str), "bg": Zr["bg"].astype(np.int64), "E": Zr["E"].astype(np.float64), "gap": Zr["lead_gap"].astype(np.float64)}
    D["gap_banked"] = D["gap"].copy()
    if smoke:
        D["gap"] = S0.permute_across(D["gap"], 11)           # BEFORE pairs are formed into targets
    folds_b = RP.make_folds(D["beid"])
    ri, rm, rk, pc = build_pairs(D["beid"], D["bg"], D["gap"])
    d = D["gap"][rk] - D["gap"][ri]
    Eb = ego_block(D["E"], ri, rm, rk)
    ep = D["epi"][ri]
    n_ep = len(D["uep"])
    fold_p = folds_b[ri]
    Pr = {"ri": ri, "rm": rm, "rk": rk, "Eb": Eb, "ep": ep}
    dsets = {"main": d, "k1": S0.permute_within_fold(d, fold_p, 1)}
    M = S0.mean_token(D["X"])
    spec = k2_setup(D)
    side = k2_side(len(ri), spec["seed_side"])
    log(f"rows {N}, pairs {len(ri)} (three rows: {pc['pairs_with_three_rows']}), episodes {n_ep}, with pairs {len(np.unique(ep))}; k2 cell {spec['cell']} norm {spec['median_token_norm']:.2f}")
    sig = {"N": N, "P": int(len(ri)), "smoke": smoke}
    oof, times = compute_oof(D, M, Pr, dsets, folds_b, spec, side, not smoke, None if smoke else stage_dir, sig)
    C = RP.boot_counts(n_ep, B, SEED)
    # controls
    ctr = {}
    ctr["k3"] = check_k3(raw["beid"], raw["bg"], raw["E"], raw["gap"], D["beid"], D["bg"], ri, rm, rk, Eb, D["gap_banked"][ri], D["gap_banked"][rk], compare_gap=True)
    ctr["k3"]["tokens"] = check_tokens(D, ri)
    ctr["k3"]["pass"] = bool(ctr["k3"]["pass"] and ctr["k3"]["tokens"]["bad"] == 0)
    ctr["k2"] = check_k2(RP.auc_pooled(side, oof["k2|grid2"]), RP.auc_pooled(side, oof["k2|mean2"]), RP.auc_pooled(side, oof["k2|grid1"]))
    ctr["k2"]["n_pairs"] = int(len(ri)); ctr["k2"]["n_episodes"] = int(len(np.unique(ep)))
    ctr["k2"]["cell"] = spec["cell"]; ctr["k2"]["vector_norm"] = float(spec["median_token_norm"])
    meta = {"note": "PREREG_V8F_PROOF_ADD12_N5.md", "defs": "S0C_DEFS.md", "tier": TIER, "statement": STATEMENT, "B": B, "n_episodes": n_ep,
            "n_episodes_with_pairs": int(len(np.unique(ep))), "smoke": smoke, "estimator": "paired episode-cluster bootstrap",
            "counts": {**pc, "bank_rows": N, "pairs_per_fold": [int((fold_p == f).sum()) for f in range(5)]},
            "shapes": {"X": list(D["X"].shape), "S0g": [N, S0.NCELL * S0.CELLD], "S0m": [N, S0.MEAN_D], "H-g": [len(ri), 2 * S0.NCELL * S0.CELLD + 24],
                       "H-m": [len(ri), 2 * S0.MEAN_D + 24], "N-g": [len(ri), S0.NCELL * S0.CELLD + 24], "E": [len(ri), 24]},
            "stage_seconds": times, "load_seconds": t_load}
    res = {"controls": ctr, "meta": meta}
    if not smoke:
        yk = oof["k4|ra_g"]; g = D["gap"]
        from_floor, det = S0.e_floor_ra(D["E"], g, np.isfinite(g), D["epi"], folds_b, False)
        ctr["k4"] = check_k4(RP.r2_pooled(g, yk), RP.r2_pooled(g, from_floor))
        ctr["k4"]["ego_floor_choice"] = det
        sk1 = {a: skill(dsets["k1"], oof[f"arm|k1|{a}"]) for a in ARMS}
        ctr["k1"] = check_k1(sk1)
        ctr["k1"]["n_pairs"] = int(len(ri)); ctr["k1"]["n_episodes"] = int(len(np.unique(ep)))
    res["controls_pass"] = bool(all(v["pass"] for v in ctr.values()))
    res["main"] = analyse_main(d, ep, oof, C, n_ep)
    if not res["controls_pass"]:
        res["main"].pop("outcomes")                      # a failed control: numbers are kept, no outcome name is written anywhere
    json.dumps(res, default=float)
    chk = f"{out_dir}/_format_check.tmp"
    write_verdict(chk, {**res, "controls_pass": True}, meta, False) if "outcomes" in res["main"] else None
    if os.path.exists(chk):
        os.remove(chk)
    if smoke:     # never store a skill of permuted labels: keep k2, k3, shapes, counts, timings
        keep = {"meta": meta, "controls": {k: ctr[k] for k in ("k2", "k3")}, "total_seconds": time.time() - t_all}
        with open(f"{out_dir}/RESULT_S0C_SMOKE.json", "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=1, default=float)
        write_verdict(f"{out_dir}/VERDICT_S0C_SMOKE.md", res, meta, True)
        log("SMOKE shapes", meta["shapes"], "counts", meta["counts"])
        log("SMOKE k2 (label free):", ctr["k2"]); log("SMOKE k3:", ctr["k3"])
        log(f"SMOKE total {time.time() - t_all:.0f}s, load {t_load:.0f}s, stage seconds {times}")
        return 0
    with open(f"{out_dir}/RESULT_S0C.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_verdict(f"{out_dir}/VERDICT_S0C.md", res, meta, False)
    log(f"done {time.time() - t_all:.0f}s; controls_pass {res['controls_pass']}")
    return 0 if res["controls_pass"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "smoke"])
    a = ap.parse_args()
    sys.exit(cmd_run(a.cmd == "smoke"))
