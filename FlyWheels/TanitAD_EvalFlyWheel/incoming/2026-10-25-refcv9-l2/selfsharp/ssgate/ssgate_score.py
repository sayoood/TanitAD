"""ssgate_score.py -- PREREG_SSGATE.md: margin-gated SELFSHARP, composed per window from the banked rolls.  CPU only.

    python ssgate_score.py --out <dir> [--bank-ss F:/refcv9_selfsharp]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SS = HERE.parent
L2 = SS.parent
sys.path.insert(0, str(L2 / "code"))
sys.path.insert(0, str(L2 / "probegate"))
import l2_common as C  # noqa: E402
import l2_score as S  # noqa: E402
from pg_score import fixed_broken, ESTIMATOR  # noqa: E402

LON6 = np.array((0, 1, 3, 4, 5, 7))
QS = tuple(round(0.1 * i, 1) for i in range(1, 10))
K_ALL, K_NONE = (0.38967, 0.15032), (0.46948, 0.13955)      # literals from SS_RESULT.json (prereg section 4)
NEXT = {"PASS": "confirm with a GPU roll of the gated labels at the full-data tau (seed 0) AND an inference-seed-1 roll; then refcv9 confidence-gated "
                "sharpen / temperature in the lon head, and the PI for SHIP-R8.",
        "FAIL_P1": "margin does not separate right from wrong choices: the lever is head ACCURACY (train the lon head toward the z_tac probe on the clean L1C split).",
        "FAIL_G": "the lateral cost rides on the sharpened windows whatever the margin: the lever is a SOFTER sharpen (temperature on the LON6 logits) with a "
                  "lateral guard, one GPU roll, pre-registered first.",
        "FAIL_P2": "the gated sharpen converts cruise into false decels: read the per-decile FD split before any roll.",
        "INVALID": "a control failed: fix the composer, re-score (0 GPU)."}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sub(cells, m):
    return {k: v[m] for k, v in cells.items()}


def g_worse(PR, eid, A, B, rows, keys):
    cmp = S.compare(PR, {"eid": eid}, A, B, rows)
    worse = sorted(k for k in list(keys) + ["ade_m"] if k in cmp and cmp[k].get("worse") is True)
    return cmp, worse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-ss", default="F:/refcv9_selfsharp")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = sha(HERE / "PREREG_SSGATE.md")
    if pre != (HERE / "PREREG_SSGATE.sha256").read_text().split()[0]:
        raise SystemExit("[ssgate] PREREG_SSGATE.md does not match its seal: refusing")
    st = C.stack()
    PR = st["PR"]
    rows = S.metric_rows(PR)
    keys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    E = C.load_eval_pf(0)
    b1 = S.load_l1_base("C:/Users/Admin/refcv8_pf/refcv9_l1/bank", 0)
    tss = S.load_trace(Path(a.bank_ss) / "trace_selfsharp_s0.npz")
    torc = S.load_trace(L2 / "bank" / "trace_oracle_s0.npz")
    p_base, p_ss = S.join_plan8(b1, E), S.join_plan8(tss, E)
    raw = S.join_plan8({"idx": tss["idx"], "plan8": tss["raw"]}, E)               # raw lon logits per S6 window (same join)
    srt = np.sort(raw[:, LON6], 1)
    m = (srt[:, -1] - srt[:, -2]).astype(np.float64)
    eid = np.asarray(E["eid"])
    comp = np.asarray(E["compliant"], bool)
    ctl, res = {}, {"tool": "ssgate_score.py", "prereg_sha256": pre, "script_sha256": sha(__file__), "estimator": ESTIMATOR,
                    "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
                    "families_not_computed": {"LONGITUDINAL distance-keeping (headway/TTC)": "no lead-agent track in the banked refcv8 dumps; n = 0",
                                              "STRATEGIC": "refcv8 runs --no-strategic; n = 0", "S2 rows": "PENDING (as SELFSHARP)"}}
    # K-IND: unforced ORACLE windows are bit-equal to BASE-os (per-window independence of the roll)
    lab = np.load(L2 / "bank" / "oracle_labels.npz")
    forced = {(int(c), int(w)): (int(v) in LON6) for c, w, v in zip(lab["clip_index"], lab["ws"], lab["lon_v7"])}
    ju = [(k, i) for k, i in torc["idx"].items() if not forced.get(k, False)]
    eq = [np.array_equal(torc["plan8"][i], b1["plan8"][b1["idx"][k]]) for k, i in ju]
    ctl["K-IND"] = {"pass": bool(len(eq) > 0 and all(eq)), "n_unforced": len(eq), "n_equal": int(sum(eq))}

    def cells_for(gate):
        return S.arm_cells(E, np.where(gate[:, None, None], p_ss, p_base))["stage"]
    c_base, c_ss = cells_for(np.zeros(len(m), bool)), cells_for(np.ones(len(m), bool))
    r = lambda c: (round(S.rate(c, "missed_decel"), 5), round(S.rate(c, "false_decel"), 5))  # noqa: E731
    ctl["K-ALL"] = {"pass": r(c_ss) == K_ALL, "got": r(c_ss)}
    ctl["K-NONE"] = {"pass": r(c_base) == K_NONE, "got": r(c_base)}
    ssr = json.loads((SS / "result" / "SS_RESULT.json").read_text())["comparisons"]["SELFSHARP|stage|vs BASE-os"]["missed_decel"]
    mine = S.compare(PR, E, c_ss, c_base, rows)["missed_decel"]
    ctl["K-SS"] = {"pass": all(mine[f] == ssr[f] for f in ("delta", "lo", "hi", "separated")), "got": {f: mine[f] for f in ("delta", "lo", "hi")}}
    # ---- cross-fit tau ----------------------------------------------------------------------------------------------------------------
    ue = np.unique(eid)
    perm = np.random.default_rng(0).permutation(len(ue))
    fold_of = {int(e): int(r_ % 5) for e, r_ in zip(ue[perm], range(len(ue)))}
    fold = np.array([fold_of[int(e)] for e in eid])
    gate_cf = np.zeros(len(m), bool)
    folds = []
    cache = {}
    for k in range(5):
        tr = fold != k
        cands = [np.inf, -np.inf] + [float(np.quantile(m[tr & comp], q)) for q in QS]
        best = None
        cinfo = []
        for tau in cands:
            ck = ("tau", tau)
            if ck not in cache:
                cache[ck] = cells_for(m >= tau)
            A = sub(cache[ck], tr)
            B = sub(c_base, tr)
            cmp, worse = g_worse(PR, eid[tr], A, B, rows, keys)
            fd_up = bool(cmp["false_decel"]["lo"] > 0) if "lo" in cmp["false_decel"] else False
            md = S.rate(A, "missed_decel")
            adm = (not worse) and (not fd_up)
            cinfo.append({"tau": tau, "md_train": md, "admissible": adm, "worse": worse, "fd_up": fd_up, "n_sharp_train": int((m[tr] >= tau).sum())})
            if adm and (best is None or md < best[1] - 1e-12 or (abs(md - best[1]) <= 1e-12 and tau > best[0])):
                best = (tau, md)
        gate_cf[fold == k] = m[fold == k] >= best[0]
        folds.append({"fold": k, "tau": best[0], "md_train": best[1], "n_test_windows": int((fold == k).sum()), "candidates": cinfo})
        print(f"[ssgate] fold {k} tau={best[0]:.3f}", flush=True)
    res["folds"] = folds
    c_cf = cells_for(gate_cf)
    cmp, worse = g_worse(PR, eid, c_cf, c_base, rows, keys)
    cmp_ss, worse_ss = g_worse(PR, eid, c_ss, c_base, rows, keys)
    bars = {"P1": bool(cmp["missed_decel"]["hi"] < 0), "P2": bool(not (cmp["false_decel"]["lo"] > 0)), "G_rows_separated_worse": worse, "G_strict": not worse,
            "P1_cell": cmp["missed_decel"], "P2_cell": cmp["false_decel"], "n_sharpened": int(gate_cf.sum()), "n_windows": int(len(m))}
    res.update(controls=ctl, bars=bars, comparisons={"SSGATE-CF|stage|vs BASE-os": cmp},
               rates={nm: {"missed_decel": S.rate(c, "missed_decel"), "false_decel": S.rate(c, "false_decel")} for nm, c in (("BASE-os", c_base), ("SELFSHARP", c_ss), ("SSGATE-CF", c_cf))})
    # ---- descriptive: per margin decile -----------------------------------------------------------------------------------------------
    edges = np.quantile(m[comp], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, 9)
    G6 = ["LAT_cross_abs_final_m", "LAT_cross_mae_m", "LAT_curvature_mae_1pm_valid", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps_valid", "TAC_traj_lat_correct"]
    desc = []
    for d in range(10):
        w = dec == d
        row = {"decile": d, "margin_lo": float(edges[d]), "margin_hi": float(edges[d + 1]),
               "md": fixed_broken(c_base["missed_decel"][w], c_ss["missed_decel"][w]), "fd": fixed_broken(c_base["false_decel"][w], c_ss["false_decel"][w])}
        for g in G6:
            if g in c_ss:
                dd = c_ss[g][w] - c_base[g][w]
                row[g] = float(np.nanmean(dd)) if np.isfinite(dd).any() else None
        desc.append(row)
    full = cells_for(m >= np.inf)
    res["descriptive"] = {"per_margin_decile_SELFSHARP_minus_BASE": desc, "SELFSHARP_worse_rows_recomputed": worse_ss}
    # full-data tau (the deployable value a confirmation roll would use): same rule on all S6 windows
    best = None
    for tau in [np.inf, -np.inf] + [float(np.quantile(m[comp], q)) for q in QS]:
        A = cells_for(m >= tau)
        c2, w2 = g_worse(PR, eid, A, c_base, rows, keys)
        adm = (not w2) and not (c2["false_decel"].get("lo", 0) > 0)
        md = S.rate(A, "missed_decel")
        if adm and (best is None or md < best[1] - 1e-12 or (abs(md - best[1]) <= 1e-12 and tau > best[0])):
            best = (tau, md)
    res["descriptive"]["full_data_tau"] = {"tau": best[0], "md": best[1], "n_sharpened": int((m >= best[0]).sum())}
    del full
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"])
    if ctl_fail:
        word, nl = f"INVALID (control failed: {ctl_fail})", NEXT["INVALID"]
    elif bars["P1"] and bars["P2"] and bars["G_strict"]:
        word, nl = "PASS", NEXT["PASS"]
    else:
        failing = [k for k in ("P1", "P2", "G_strict") if not bars[k]]
        word = "FAIL (" + ", ".join(failing) + ")"
        nl = NEXT["FAIL_P1"] if not bars["P1"] else NEXT["FAIL_P2"] if not bars["P2"] else NEXT["FAIL_G"]
    res.update(verdict=word, next_lever=nl, wall_s=round(time.time() - t0, 1))
    C.jdump(res, out / "SSGATE_RESULT.json")
    fmt = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    R = res["rates"]
    L = [word, "", f"# SSGATE RESULT (mechanical; PREREG_SSGATE sha256 {pre[:12]})", "", f"tier: {res['tier']}", f"estimator: {ESTIMATOR}", "",
         f"MD (S6 compliant, stage on): BASE-os {R['BASE-os']['missed_decel']:.4f} -> SSGATE-CF {R['SSGATE-CF']['missed_decel']:.4f} (SELFSHARP {R['SELFSHARP']['missed_decel']:.4f})",
         f"FD: BASE-os {R['BASE-os']['false_decel']:.4f} -> SSGATE-CF {R['SSGATE-CF']['false_decel']:.4f} (SELFSHARP {R['SELFSHARP']['false_decel']:.4f})",
         f"sharpened windows {bars['n_sharpened']} / {bars['n_windows']}; tau per fold {[round(f['tau'], 3) for f in folds]}", "",
         "| bar | value | cell |", "|---|---|---|",
         f"| P1 MD separated below BASE-os | {bars['P1']} | {fmt(bars['P1_cell'])} |", f"| P2 FD not separated above | {bars['P2']} | {fmt(bars['P2_cell'])} |",
         f"| G-strict | {bars['G_strict']} | {', '.join(worse) or 'none'} |", "",
         "| S6 row (SSGATE-CF - BASE-os) | delta [95% CI] | worse |", "|---|---|---|"] + \
        [f"| {k} | {fmt(c)} | {c.get('worse')} |" for k, c in cmp.items()] + \
        ["", f"controls: {json.dumps({k: v['pass'] for k, v in ctl.items()})}", f"full-data tau: {json.dumps(res['descriptive']['full_data_tau'])}", "",
         f"NEXT: {nl}", "", "ZZSSGATEENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    print(word, {k: v["pass"] for k, v in ctl.items()}, bars["n_sharpened"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
