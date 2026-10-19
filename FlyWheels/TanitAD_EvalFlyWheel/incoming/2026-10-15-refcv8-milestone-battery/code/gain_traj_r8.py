"""gain_traj_r8.py -- does refcv8's best arc-length RE-TIMING gain move toward 1 across the banked checkpoints?   (CPU only, zero GPU, zero training)

NEW module (PREREG_GAINTRAJ.md, sha256 pinned below).  `a0k0_headroom_retime_r8` (RT) and `compliant_reread_r8` (CR) are imported and NOT edited.

    s_x(t_k)  = cumulative chord arc length of the polyline through (0,0) and path x's points at the surface's S-grid instants  (= RT.polyline_geometry(x)["L"][:, 1:])
    g*        = argmin_g  sum_{compliant windows, instants} (s_prior + g (s_os - s_prior) - s_human)^2  =  sum(d*h) / sum(d*d),  d = s_os - s_prior, h = s_human - s_prior
    CI        = episode-cluster bootstrap, B 2000, default_rng(0), identical draws for every tree of a (surface, seed); paired Delta on the same draws
                ("another draw of EPISODES" only: NOT another training run, NOT another inference run)

    python gain_traj_r8.py run       # controls -> (WITHHELD.json + exit) | GAINTRAJ.json / GAINTRAJ.md
    python gain_traj_r8.py train     # optional: the same g* on the 139 TRAIN-clip rolls of EMA-30k -> GAINTRAJ_TRAIN.json (separate process: it re-points the archive root)

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -X utf8 gain_traj_r8.py run
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_retime_r8 as RT  # noqa: E402

PKG = HERE.parent
OUT = PKG / "raw" / "gain_traj"
PREREG = OUT / "PREREG_GAINTRAJ.md"
PREREG_SHA256 = "4245baceb53d75fded31602fa996ad7b95ea5306e69c2056f0ed78ad80a6ddcc"
FROZEN_30K = PKG / "raw" / "a0k0_retime" / "FROZEN_GAINS_RETIME_ema30k.json"

EMA_TREES = ("step5000ema_unc", "step10500ema_unc", "step15000ema_unc", "step20000ema_unc", "step30000ema_unc")
RAW_TREES = ("step5000_unc", "step10500_unc", "step15000_unc", "step20000_unc", "step30000_unc")
STEP = {t: int(re.match(r"step(\d+)", t).group(1)) for t in EMA_TREES + RAW_TREES}
CTRL_TREE = "step30000ema_unc"
SURFS = ("S2", "S6")
SEEDS = (0, 1)
N_BOOT = 2000
BOOT_SEED = 0
MASK_SHA = "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129"
N_COMPLIANT_30K = {"S2": 4159, "S6": 3251}        # the brief's literals
TOL_CTRL = 1e-9
TOL_C4 = 1e-6
LABEL_UPPER = 0.8
CAVEATS = ("single training seed (the trees are checkpoints of ONE run); T1 open loop on logged eval139 frames; per-instant fits are in-sample per tree; eval episodes only; "
           "the interval answers 'another draw of EPISODES' only (not another training run, not another inference run); no extrapolation beyond step 30,000 (final = 48,852)")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_prereg() -> None:
    if sha256_file(PREREG) != PREREG_SHA256:
        raise SystemExit(f"[gain_traj] {PREREG} does not hash to the registered sha256 {PREREG_SHA256}: the numbers must be produced under the registered text")


# ------------------------------------------------------------------------------------------ pure estimators
def arc(x) -> np.ndarray:
    """Cumulative chord arc length at the grid instants [n, T] (RT's own definition), float64."""
    return RT.polyline_geometry(np.asarray(x, np.float64))["L"][:, 1:]


def flat_gain(s_prior, s_os, s_human) -> float:
    """PRIMARY estimand: sum(d*h)/sum(d*d), d = s_os - s_prior, h = s_human - s_prior, pooled over all windows and instants.  Refuses sum(d*d) == 0."""
    d = np.asarray(s_os, np.float64) - np.asarray(s_prior, np.float64)
    h = np.asarray(s_human, np.float64) - np.asarray(s_prior, np.float64)
    den = float((d * d).sum())
    if not np.isfinite(den) or den <= 0.0:
        raise SystemExit("[gain_traj] sum(d*d) is zero / non-finite: the gain is undefined")
    return float((d * h).sum()) / den


def flat_gain_loop(s_prior, s_os, s_human) -> float:
    """INDEPENDENT plain-python re-derivation of `flat_gain` (control C5): per-window, per-instant python floats, no shared helper."""
    a = b = 0.0
    for i in range(len(s_os)):
        for k in range(len(s_os[i])):
            d = float(s_os[i][k]) - float(s_prior[i][k])
            h = float(s_human[i][k]) - float(s_prior[i][k])
            a += d * h
            b += d * d
    return a / b


def flat_gain_swapped(s_prior, s_os, s_human) -> float:
    """DELIBERATE REGRESSION (C-MUT): prior and os swapped inside the estimator."""
    return flat_gain(s_os, s_prior, s_human)


def perinstant_ls(s_prior, s_os, s_human) -> list:
    d = np.asarray(s_os, np.float64) - np.asarray(s_prior, np.float64)
    h = np.asarray(s_human, np.float64) - np.asarray(s_prior, np.float64)
    den = (d * d).sum(0)
    return [float((d[:, k] * h[:, k]).sum() / den[k]) if den[k] > 0 else None for k in range(d.shape[1])]


def episode_stats(s_prior, s_os, s_human, eid):
    """(unique episodes, A_e = sum d*h, B_e = sum d*d) per episode."""
    d = np.asarray(s_os, np.float64) - np.asarray(s_prior, np.float64)
    h = np.asarray(s_human, np.float64) - np.asarray(s_prior, np.float64)
    uniq, inv = np.unique(np.asarray([str(x) for x in eid]), return_inverse=True)
    A = np.bincount(inv, weights=(d * h).sum(1), minlength=len(uniq))
    B = np.bincount(inv, weights=(d * d).sum(1), minlength=len(uniq))
    return uniq, A, B


def draw_counts(n_ep: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Multiplicity matrix [n_boot, n_ep]: row b = how often each episode is drawn in replicate b (episodes resampled with replacement)."""
    picks = np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))
    C = np.zeros((n_boot, n_ep))
    for b in range(n_boot):
        C[b] = np.bincount(picks[b], minlength=n_ep)
    return C


def boot_gain(A, B, C) -> np.ndarray:
    return (C @ A) / (C @ B)


def interval(v, alpha=0.05) -> tuple:
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def gain_cell(A, B, C) -> dict:
    g = float(A.sum() / B.sum())
    lo, hi = interval(boot_gain(A, B, C))
    return {"g_star": g, "lo": lo, "hi": hi, "n_episodes": int(len(A))}


def paired_cell(A1, B1, A0, B0, C) -> dict:
    """g*(tree1) - g*(tree0) on the SAME resampled episodes."""
    pt = float(A1.sum() / B1.sum() - A0.sum() / B0.sum())
    d = boot_gain(A1, B1, C) - boot_gain(A0, B0, C)
    lo, hi = interval(d)
    return {"delta": pt, "lo": lo, "hi": hi, "separated_gt0": bool(lo > 0), "separated_lt0": bool(hi < 0), "p_delta_gt0": float((d > 0).mean())}


def outcome_label(cells: dict) -> dict:
    """PREREG section 3.  cells[seed] = {'g': {tree: cell}, 'delta_30k_5k': paired cell}; EMA trees in EMA_TREES order; S2, compliant."""
    per = {}
    for s, c in cells.items():
        g = [c["g"][t]["g_star"] for t in EMA_TREES]
        mono = all(g[i + 1] >= g[i] for i in range(len(g) - 1))
        per[s] = {"g_star_by_step": dict(zip([STEP[t] for t in EMA_TREES], g)), "non_decreasing": bool(mono),
                  "consecutive_diffs": [g[i + 1] - g[i] for i in range(len(g) - 1)],
                  "delta_30k_minus_5k_separated_gt0": bool(c["delta_30k_5k"]["separated_gt0"]),
                  "g30k_hi": float(c["g"]["step30000ema_unc"]["hi"]), "g30k_hi_lt_0p8": bool(c["g"]["step30000ema_unc"]["hi"] < LABEL_UPPER)}
    converging = all(p["delta_30k_minus_5k_separated_gt0"] and p["non_decreasing"] for p in per.values())
    structural = (not converging) and all(p["g30k_hi_lt_0p8"] for p in per.values())
    return {"label": "CONVERGING" if converging else ("STRUCTURAL" if structural else "MIXED"), "per_seed": per}


# ------------------------------------------------------------------------------------------ data
def load_tree(tree: str, tools: dict, CR) -> dict:
    """Compact compliant arrays of every (surface, seed) of one tree.  Model-free references / GT / window order asserted identical across the two inference seeds (RT.surface_inputs)."""
    first = {}
    res = {s: {} for s in SURFS}
    for seed in SEEDS:
        ctx = CR.prepare_seed(tree, seed, tools, tools["rc"])
        if ctx["sha"] != MASK_SHA:
            raise SystemExit(f"[gain_traj] {tree} s{seed}: mask sha {ctx['sha']} != the registered {MASK_SHA}")
        for surf in SURFS:
            S = RT.surface_inputs(tools, ctx, surf, first, tree)
            if not (S["dec"]["chk"]["PASS"] and S["V"]["PASS"]):
                raise SystemExit(f"[gain_traj] {tree} s{seed} {surf}: decisions / reference validation failed")
            sel = np.asarray(S["mask"], bool)
            X = S["X"]
            O, HP, G = (np.asarray(X[k][sel]) for k in ("O", "HP", "G"))
            if not (np.isfinite(O).all() and np.isfinite(HP).all() and np.isfinite(G).all()):
                raise SystemExit(f"[gain_traj] {tree} s{seed} {surf}: non-finite plan / prior / GT")
            res[surf][seed] = {"O": O, "HP": HP, "G": G, "eid": np.asarray([str(x) for x in np.asarray(S["eid"])[sel]]),
                               "key": np.stack([S["clip"][sel], S["ws"][sel]]), "n_all": int(len(sel)), "n_comp": int(sel.sum()), "hz": list(S["hz"]),
                               "sp": arc(HP), "so": arc(O), "sh": arc(G), "input_sha": RT._sha_arrays(O, HP, G)}
        del ctx
    return res


def assert_consistent(trees: dict) -> dict:
    """Across trees: identical windows / order / episodes, identical (model-free) prior and GT; counts equal and equal to the brief's 30k literals."""
    ref = trees[CTRL_TREE]
    rep = {}
    for t, R in trees.items():
        for surf in SURFS:
            for seed in SEEDS:
                a, b = R[surf][seed], ref[surf][seed]
                ok = {"key_equal": bool(np.array_equal(a["key"], b["key"])), "eid_equal": bool(np.array_equal(a["eid"], b["eid"])),
                      "HP_bit_equal": bool(np.array_equal(a["HP"], b["HP"])), "G_bit_equal": bool(np.array_equal(a["G"], b["G"])), "hz_equal": a["hz"] == b["hz"],
                      "n_comp": a["n_comp"], "n_comp_is_brief_literal": a["n_comp"] == N_COMPLIANT_30K[surf]}
                ok["PASS"] = bool(all(v for k, v in ok.items() if k != "n_comp"))
                rep[f"{t}|{surf}|s{seed}"] = ok
                if not ok["PASS"]:
                    raise SystemExit(f"[gain_traj] cross-tree consistency failed: {t} {surf} s{seed}: {ok}")
    return rep


# ------------------------------------------------------------------------------------------ controls
def run_controls(ctrl: dict, frozen: dict) -> dict:
    out = {"tree": CTRL_TREE, "rows": {}, "PASS": False}
    ok_all = True
    # C1-C3 + C-MUT + C5 on the compliant S2 (and S6 beside) arrays of EMA-30k, both seeds, through the production estimator
    for surf in SURFS:
        for seed in SEEDS:
            R = ctrl[surf][seed]
            sp, so, sh, eid = R["sp"], R["so"], R["sh"], R["eid"]
            row = {}
            for name, s_os, want in (("C1_os_eq_human", sh, 1.0), ("C2_os_eq_prior_plus_2x", sp + 2.0 * (sh - sp), 0.5), ("C3_os_eq_prior_plus_4x", sp + 4.0 * (sh - sp), 0.25)):
                g = flat_gain(sp, s_os, sh)
                uniq, A, B = episode_stats(sp, s_os, sh, eid)
                bs = boot_gain(A, B, draw_counts(len(uniq), 200))
                row[name] = {"expected": want, "got": g, "abs_err": abs(g - want), "boot_min": float(bs.min()), "boot_max": float(bs.max()),
                             "PASS": bool(abs(g - want) <= TOL_CTRL and abs(bs.min() - want) <= TOL_CTRL and abs(bs.max() - want) <= TOL_CTRL)}
            gm1, gm3 = flat_gain_swapped(sp, sh, sh), flat_gain_swapped(sp, sp + 4.0 * (sh - sp), sh)
            row["C_MUT_swapped_estimator"] = {"C1_reading": gm1, "C1_expected_correct": 1.0, "C1_is_RED": bool(abs(gm1 - 1.0) > 1e-3),
                                              "C3_reading": gm3, "C3_expected_correct": 0.25, "C3_is_RED": bool(abs(gm3 - 0.25) > 1e-3)}
            row["C_MUT_swapped_estimator"]["PASS"] = bool(row["C_MUT_swapped_estimator"]["C1_is_RED"] and row["C_MUT_swapped_estimator"]["C3_is_RED"])
            g_vec, g_loop = flat_gain(sp, so, sh), flat_gain_loop(sp, so, sh)
            row["C5_vectorised_vs_plain_loop"] = {"vectorised": g_vec, "loop": g_loop, "abs_err": abs(g_vec - g_loop), "PASS": bool(abs(g_vec - g_loop) <= TOL_CTRL)}
            row["PASS"] = bool(all(v["PASS"] for v in row.values() if isinstance(v, dict)))
            out["rows"][f"{surf}|s{seed}"] = row
            ok_all &= row["PASS"]
    # C4: the data path reproduces the banked frozen EMA-30k compliant re-timing gains (both seeds pooled)
    c4 = {}
    for surf in SURFS:
        O2 = np.concatenate([ctrl[surf][s]["O"] for s in SEEDS])
        B2 = np.concatenate([ctrl[surf][s]["HP"] for s in SEEDS])
        G2 = np.concatenate([ctrl[surf][s]["G"] for s in SEEDS])
        got = RT.retime_fit(O2, B2, G2)
        want = np.asarray(frozen["surfaces"][surf]["compliant_info_only"]["RETIME"], np.float64)
        n_fit = int(frozen["surfaces"][surf]["compliant_info_only"]["n_fit_rows_pooled"])
        c4[surf] = {"got": [float(x) for x in got], "banked": [float(x) for x in want], "max_abs_diff": float(np.abs(got - want).max()),
                    "n_pooled_rows": int(len(O2)), "banked_n_fit_rows_pooled": n_fit,
                    "PASS": bool(np.abs(got - want).max() <= TOL_C4 and len(O2) == n_fit and ctrl[surf][0]["n_comp"] == N_COMPLIANT_30K[surf])}
        ok_all &= c4[surf]["PASS"]
    out["C4_data_path_vs_frozen_gains"] = c4
    out["PASS"] = bool(ok_all)
    return out


# ------------------------------------------------------------------------------------------ the run
def per_tree_cells(trees: dict) -> dict:
    """cells[tree][surf][seed] = primary cell + per-instant diagnostics; stats[(surf, seed)] = (uniq, C, {tree: (A, B)})."""
    cells, stats = {}, {}
    for surf in SURFS:
        for seed in SEEDS:
            uniq0 = None
            AB = {}
            for t, R in trees.items():
                r = R[surf][seed]
                uniq, A, B = episode_stats(r["sp"], r["so"], r["sh"], r["eid"])
                if uniq0 is None:
                    uniq0 = uniq
                elif not np.array_equal(uniq0, uniq):
                    raise SystemExit(f"[gain_traj] {t} {surf} s{seed}: episode set differs from the first tree")
                AB[t] = (A, B)
            C = draw_counts(len(uniq0))
            stats[(surf, seed)] = (uniq0, C, AB)
            for t, R in trees.items():
                r = R[surf][seed]
                A, B = AB[t]
                c = gain_cell(A, B, C)
                c["g_star_check_vs_flat_gain"] = abs(c["g_star"] - flat_gain(r["sp"], r["so"], r["sh"]))
                c.update({"n_windows": int(len(r["eid"])), "n_instants": int(r["sp"].shape[1]), "instants_s": [h / 10 for h in r["hz"]],
                          "per_instant_ls_gain": perinstant_ls(r["sp"], r["so"], r["sh"]),
                          "per_instant_retime_fit_gain": [float(x) for x in RT.retime_fit(r["O"], r["HP"], r["G"])],
                          "mean_abs_s_os_minus_s_prior_m": float(np.abs(r["so"] - r["sp"]).mean()), "mean_abs_s_human_minus_s_prior_m": float(np.abs(r["sh"] - r["sp"]).mean()),
                          "input_sha256": r["input_sha"]})
                cells.setdefault(t, {}).setdefault(surf, {})[seed] = c
    return cells, stats


def paired_all(trees: dict, stats: dict) -> dict:
    out = {}
    for surf in SURFS:
        for seed in SEEDS:
            _u, C, AB = stats[(surf, seed)]
            for series, names in (("EMA", EMA_TREES), ("RAW", RAW_TREES)):
                d = {}
                lo, hi = names[0], names[-1]
                d[f"{STEP[hi]}_minus_{STEP[lo]}"] = paired_cell(*AB[hi], *AB[lo], C)
                for i in range(len(names) - 1):
                    a, b = names[i], names[i + 1]
                    d[f"{STEP[b]}_minus_{STEP[a]}"] = paired_cell(*AB[b], *AB[a], C)
                out[f"{series}|{surf}|s{seed}"] = d
    return out


def fmt(c: dict) -> str:
    return f"{c['g_star']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def render_md(res: dict) -> str:
    L = []
    cells, paired, lab = res["cells"], res["paired"], res["label"]
    L += ["# GAINTRAJ -- does the best arc-length re-timing gain move toward 1 across refcv8's banked checkpoints?", "",
          f"Generated {res['generated']}. Registered text: `raw/gain_traj/PREREG_GAINTRAJ.md` (sha256 `{res['prereg_sha256']}`). Script `code/gain_traj_r8.py` sha256 `{res['script_sha256']}`.",
          "Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Compliant windows (ceiling-compliant mask `4ebc6b86...`; S2 4,159 / S6 3,251 windows).", "",
          f"## LABEL (EMA, S2, both inference seeds): **{lab['label']}**", ""]
    for s, p in lab["per_seed"].items():
        L.append(f"* seed {s}: g* by step {', '.join(f'{k}: {v:.3f}' for k, v in p['g_star_by_step'].items())}; non-decreasing = {p['non_decreasing']}; "
                 f"Delta(30k - 5k) separated > 0 = {p['delta_30k_minus_5k_separated_gt0']}; 30k upper bound {p['g30k_hi']:.3f} (< 0.8: {p['g30k_hi_lt_0p8']})")
    L += ["", "Rule (registered): CONVERGING = Delta(30k-5k) separated > 0 at both seeds AND g* non-decreasing across the five EMA trees at both seeds; STRUCTURAL = not CONVERGING AND 30k upper bound < 0.8 at both seeds; MIXED otherwise. "
          "The label cannot tell a flat g* from a falling one; read the Delta table. **No extrapolation to the final step 48,852.**", ""]
    for series, names, title in (("EMA", EMA_TREES, "EMA trees (the label's trees)"), ("RAW", RAW_TREES, "RAW trees (beside, not in the label)")):
        for surf in SURFS:
            L += [f"## g* [95 % episode-cluster CI], {title}, {surf}", "", "| step | seed 0 | seed 1 | seed0 - seed1 | n windows | n episodes |", "|---|---|---|---|---|---|"]
            for t in names:
                c0, c1 = cells[t][surf]["0"], cells[t][surf]["1"]
                L.append(f"| {STEP[t]:,} | {fmt(c0)} | {fmt(c1)} | {c0['g_star'] - c1['g_star']:+.4f} | {c0['n_windows']:,} | {c0['n_episodes']} |")
            L.append("")
            L += [f"Paired Delta g* ({series}, {surf}; same episode draws):", "", "| pair | seed 0 | seed 1 |", "|---|---|---|"]
            keys = list(paired[f"{series}|{surf}|s0"].keys())
            for k in keys:
                a, b = paired[f"{series}|{surf}|s0"][k], paired[f"{series}|{surf}|s1"][k]
                f2 = lambda c: f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" + (" sep>0" if c["separated_gt0"] else (" sep<0" if c["separated_lt0"] else ""))  # noqa: E731
                L.append(f"| {k.replace('_minus_', ' - ')} | {f2(a)} | {f2(b)} |")
            L.append("")
    L += ["## Per-instant gains, EMA S2 (diagnostics, in-sample, no CI; (a) = RT.retime_fit (position loss along the os path), (b) = arc-length LS per instant)", "",
          "| step | seed | instants (s) | (a) retime_fit | (b) LS per instant |", "|---|---|---|---|---|"]
    for t in EMA_TREES:
        for s in ("0", "1"):
            c = cells[t]["S2"][s]
            L.append(f"| {STEP[t]:,} | {s} | {c['instants_s']} | {[round(x, 3) for x in c['per_instant_retime_fit_gain']]} | {[None if x is None else round(x, 3) for x in c['per_instant_ls_gain']]} |")
    L += ["", "## Per-instant gains, EMA S6", "", "| step | seed | instants (s) | (a) retime_fit | (b) LS per instant |", "|---|---|---|---|---|"]
    for t in EMA_TREES:
        for s in ("0", "1"):
            c = cells[t]["S6"][s]
            L.append(f"| {STEP[t]:,} | {s} | {c['instants_s']} | {[round(x, 3) for x in c['per_instant_retime_fit_gain']]} | {[None if x is None else round(x, 3) for x in c['per_instant_ls_gain']]} |")
    L += ["", "## Controls (all PASS, else the run is WITHHELD)", ""]
    ctl = res["controls"]
    L.append(f"Overall: **{'PASS' if ctl['PASS'] else 'FAIL'}** on {ctl['tree']} (compliant arrays, both seeds, both surfaces). Detail in GAINTRAJ.json `controls`.")
    r = ctl["rows"]["S2|s0"]
    L.append(f"* C1 (s_os = s_human) g* = {r['C1_os_eq_human']['got']!r} (want 1); C2 (prior + 2x) = {r['C2_os_eq_prior_plus_2x']['got']!r} (want 0.5); C3 (prior + 4x) = {r['C3_os_eq_prior_plus_4x']['got']!r} (want 0.25) [S2 seed 0]")
    L.append(f"* C-MUT swapped estimator reads C1 = {r['C_MUT_swapped_estimator']['C1_reading']:.4f} (RED vs 1) and C3 = {r['C_MUT_swapped_estimator']['C3_reading']:.4f} (RED vs 0.25); C2 is swap-invariant (0.5 <-> 0.5), which is why C3 was added")
    L.append(f"* C4 data path vs `FROZEN_GAINS_RETIME_ema30k.json` compliant gains: max abs diff S2 {ctl['C4_data_path_vs_frozen_gains']['S2']['max_abs_diff']:.2e}, S6 {ctl['C4_data_path_vs_frozen_gains']['S6']['max_abs_diff']:.2e}")
    L.append(f"* C5 vectorised vs plain-loop g* (S2 s0): abs err {r['C5_vectorised_vs_plain_loop']['abs_err']:.2e}")
    L += ["", "## Caveats (binding)", "", CAVEATS + ".", ""]
    return "\n".join(L)


def cmd_run() -> int:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[gain_traj] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    check_prereg()
    import compliant_reread_r8 as CR
    tools = RT.tools_load()
    frozen = json.loads(FROZEN_30K.read_text(encoding="utf-8"))
    trees = {}
    log(f"loading {CTRL_TREE} (controls)")
    trees[CTRL_TREE] = load_tree(CTRL_TREE, tools, CR)
    ctl = run_controls(trees[CTRL_TREE], frozen)
    log(f"controls PASS={ctl['PASS']}")
    if not ctl["PASS"]:
        RT.jdump({"WITHHELD": True, "reason": "a known-value control did not read its known value", "controls": ctl, "generated": time.strftime("%FT%T")}, OUT / "WITHHELD.json")
        print("WITHHELD: control failed; see raw/gain_traj/WITHHELD.json")
        return 2
    for t in EMA_TREES + RAW_TREES:
        if t not in trees:
            log(f"loading {t}")
            trees[t] = load_tree(t, tools, CR)
    cons = assert_consistent(trees)
    log("cross-tree consistency PASS; computing cells")
    cells, stats = per_tree_cells(trees)
    paired = paired_all(trees, stats)
    cells_s2 = {s: {"g": {t: cells[t]["S2"][s] for t in EMA_TREES}, "delta_30k_5k": paired[f"EMA|S2|s{s}"]["30000_minus_5000"]} for s in SEEDS}
    lab = outcome_label(cells_s2)
    jc = {t: {surf: {str(s): v for s, v in cells[t][surf].items()} for surf in SURFS} for t in cells}
    res = {"tool": "gain_traj_r8.py run", "generated": time.strftime("%FT%T"), "prereg": "raw/gain_traj/PREREG_GAINTRAJ.md", "prereg_sha256": PREREG_SHA256,
           "script_sha256": sha256_file(__file__), "tier": RT.TIER, "caveats": CAVEATS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "mask_sha256": MASK_SHA,
           "estimator": "episode-cluster bootstrap of the ratio sum(d*h)/sum(d*d) (episodes resampled with replacement, identical draws for every tree of a (surface, seed)); "
                        "answers 'another draw of EPISODES' only",
           "label": lab, "cells": jc, "paired": paired, "controls": ctl, "cross_tree_consistency": cons,
           "episode_sums_ema30k": {f"{surf}|s{seed}": {"episodes": stats[(surf, seed)][0].tolist(), "A": stats[(surf, seed)][2][CTRL_TREE][0].tolist(),
                                                       "B": stats[(surf, seed)][2][CTRL_TREE][1].tolist()} for surf in SURFS for seed in SEEDS},
           "trees": {t: STEP[t] for t in EMA_TREES + RAW_TREES}, "wall_s": round(time.time() - t0, 1)}
    RT.jdump(res, OUT / "GAINTRAJ.json")
    (OUT / "GAINTRAJ.md").write_text(render_md(res), encoding="utf-8")
    print(f"WROTE {OUT / 'GAINTRAJ.json'} label={lab['label']} wall {res['wall_s']} s")
    print("ZZGTDONEZZ")
    return 0


# ------------------------------------------------------------------------------------------ optional: train-clip g*
def cmd_train() -> int:
    t0 = time.time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[gain_traj] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    check_prereg()
    import retime_trainfit_r8 as TF
    ev_path = OUT / "GAINTRAJ.json"
    if not ev_path.exists():
        raise SystemExit("[gain_traj train] run `run` first (the eval EMA-30k cells are the comparison)")
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    tools = RT.tools_load()
    per, ctxs = TF._load_train(tools, SEEDS)
    out = {"tool": "gain_traj_r8.py train", "generated": time.strftime("%FT%T"), "prereg_sha256": PREREG_SHA256, "script_sha256": sha256_file(__file__),
           "tree": f"{TF.TRAIN_ROOT}/{TF.TRAIN_TREE}", "mask_sha_by_seed": ctxs, "cells": {}, "vs_eval_ema30k": {},
           "estimator": "same ratio estimator and episode-cluster bootstrap on 139 TRAIN clips; train-vs-eval difference = UNPAIRED (two different clip sets): independent episode draws within each set",
           "caveats": CAVEATS + "; train clips are in the training set of this run (the model has seen them)"}
    for surf in SURFS:
        for seed in SEEDS:
            S = per[surf][seed]
            X = S["X"]
            for set_name in ("compliant", "all"):
                sel = np.asarray(S["mask"], bool) if set_name == "compliant" else np.ones(len(S["eid"]), bool)
                sp, so, sh = arc(X["HP"][sel]), arc(X["O"][sel]), arc(X["G"][sel])
                uniq, A, B = episode_stats(sp, so, sh, np.asarray(S["eid"])[sel])
                C = draw_counts(len(uniq))
                c = gain_cell(A, B, C)
                c.update({"n_windows": int(sel.sum()), "per_instant_ls_gain": perinstant_ls(sp, so, sh),
                          "per_instant_retime_fit_gain": [float(x) for x in RT.retime_fit(X["O"][sel], X["HP"][sel], X["G"][sel])]})
                out["cells"][f"{surf}|{set_name}|s{seed}"] = c
                if set_name == "compliant":
                    es = ev["episode_sums_ema30k"][f"{surf}|s{seed}"]
                    Ae, Be = np.asarray(es["A"], np.float64), np.asarray(es["B"], np.float64)
                    ge = float(Ae.sum() / Be.sum())
                    rng = np.random.default_rng(BOOT_SEED)
                    pt, pe = rng.integers(0, len(A), size=(N_BOOT, len(A))), rng.integers(0, len(Ae), size=(N_BOOT, len(Ae)))
                    gt_b = np.array([A[i].sum() / B[i].sum() for i in pt])
                    ge_b = np.array([Ae[i].sum() / Be[i].sum() for i in pe])
                    lo, hi = interval(gt_b - ge_b)
                    out["vs_eval_ema30k"][f"{surf}|s{seed}"] = {"train_g": c["g_star"], "train_ci": [c["lo"], c["hi"]], "n_train_episodes": int(len(A)), "eval_g": ge,
                                                                "eval_ci": ev["cells"]["step30000ema_unc"][surf][str(seed)] and [ev["cells"]["step30000ema_unc"][surf][str(seed)]["lo"], ev["cells"]["step30000ema_unc"][surf][str(seed)]["hi"]],
                                                                "n_eval_episodes": int(len(Ae)), "diff_train_minus_eval": c["g_star"] - ge, "diff_lo": lo, "diff_hi": hi,
                                                                "separated": bool(lo > 0 or hi < 0)}
    RT.jdump(out, OUT / "GAINTRAJ_TRAIN.json")
    print(f"WROTE {OUT / 'GAINTRAJ_TRAIN.json'} wall {round(time.time() - t0, 1)} s")
    print("ZZGTTRAINDONEZZ")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    sys.exit({"run": cmd_run, "train": cmd_train}[cmd]())
