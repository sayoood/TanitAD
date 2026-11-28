"""ssguard_score.py -- PREREG_SSGUARD.md: native path x SELFSHARP progress, composed per window.  CPU only.

    python ssguard_score.py --out <dir> [--bank-ss F:/refcv9_selfsharp]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l1/code")
import ssgate_score as G  # noqa: E402  (helpers + constants; its main() is not run)
import l1_fit_score as F  # noqa: E402

C, S, L2, SS = G.C, G.S, G.L2, G.SS
AL, PH = F.AL, F.PH
NEXT = {"PASS": "the lat/lon-FACTORED decode is the lever: roll SELFSHARP at inference seed 1 and re-read at seed 1, then refcv9's decode design "
                "(lon class drives the speed profile only) and the PI for SHIP-R8.",
        "FAIL_P1": "the gain lived in the path, not the speed profile: the temperature roll (softer sharpen) is next, pre-registered first.",
        "FAIL_G": "the cost is not the path change either: the temperature roll is next, pre-registered first.",
        "FAIL_P2": "the guarded sharpen raises false decels: read the per-decile FD split, then the temperature roll.",
        "INVALID": "a control failed: fix the composer, re-score (0 GPU)."}


def guard(P_path, P_prog):
    return PH.at_progress(P_path, AL.arc(P_prog))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-ss", default="F:/refcv9_selfsharp")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = G.sha(HERE / "PREREG_SSGUARD.md")
    if pre != (HERE / "PREREG_SSGUARD.sha256").read_text().split()[0]:
        raise SystemExit("[ssguard] PREREG_SSGUARD.md does not match its seal: refusing")
    st = C.stack()
    PR = st["PR"]
    rows = S.metric_rows(PR)
    keys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    E = C.load_eval_pf(0)
    b1 = S.load_l1_base("C:/Users/Admin/refcv8_pf/refcv9_l1/bank", 0)
    tss = S.load_trace(Path(a.bank_ss) / "trace_selfsharp_s0.npz")
    torc = S.load_trace(L2 / "bank" / "trace_oracle_s0.npz")
    p_base, p_ss = S.join_plan8(b1, E), S.join_plan8(tss, E)
    raw = S.join_plan8({"idx": tss["idx"], "plan8": tss["raw"]}, E)
    srt = np.sort(raw[:, G.LON6], 1)
    m = (srt[:, -1] - srt[:, -2]).astype(np.float64)
    eid = np.asarray(E["eid"])
    comp = np.asarray(E["compliant"], bool)
    p_g = guard(p_base, p_ss).astype(p_base.dtype)
    cells = lambda P: S.arm_cells(E, P)["stage"]  # noqa: E731
    r = lambda c: (round(S.rate(c, "missed_decel"), 5), round(S.rate(c, "false_decel"), 5))  # noqa: E731
    c_base, c_ss, c_g = cells(p_base), cells(p_ss), cells(p_g)
    ctl = {"K-ID": {"got": r(cells(guard(p_base, p_base).astype(p_base.dtype)))}, "K-SSID": {"got": r(cells(guard(p_ss, p_ss).astype(p_ss.dtype)))},
           "K-ALL": {"got": r(c_ss)}, "K-NONE": {"got": r(c_base)}}
    ctl["K-ID"]["pass"] = ctl["K-ID"]["got"] == G.K_NONE
    ctl["K-SSID"]["pass"] = ctl["K-SSID"]["got"] == G.K_ALL
    ctl["K-ALL"]["pass"] = ctl["K-ALL"]["got"] == G.K_ALL
    ctl["K-NONE"]["pass"] = ctl["K-NONE"]["got"] == G.K_NONE
    lab = np.load(L2 / "bank" / "oracle_labels.npz")
    forced = {(int(c), int(w)): (int(v) in G.LON6) for c, w, v in zip(lab["clip_index"], lab["ws"], lab["lon_v7"])}
    eq = [np.array_equal(torc["plan8"][i], b1["plan8"][b1["idx"][k]]) for k, i in torc["idx"].items() if not forced.get(k, False)]
    ctl["K-IND"] = {"pass": bool(len(eq) > 0 and all(eq)), "n_unforced": len(eq)}
    res = {"tool": "ssguard_score.py", "prereg_sha256": pre, "script_sha256": G.sha(__file__), "estimator": G.ESTIMATOR,
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)", "controls": ctl,
           "families_not_computed": {"LONGITUDINAL distance-keeping (headway/TTC)": "no lead-agent track in the banked refcv8 dumps; n = 0",
                                     "STRATEGIC": "refcv8 runs --no-strategic; n = 0", "S2 rows": "PENDING"},
           "degenerate_base_path_windows": int((AL.arc(p_base)[:, -1] <= 1e-6).sum())}

    def bars_of(cA):
        cmp, worse = G.g_worse(PR, eid, cA, c_base, rows, keys)
        return cmp, {"P1": bool(cmp["missed_decel"]["hi"] < 0), "P2": bool(not (cmp["false_decel"]["lo"] > 0)), "G_rows_separated_worse": worse, "G_strict": not worse,
                     "P1_cell": cmp["missed_decel"], "P2_cell": cmp["false_decel"]}
    cmp_all, b_all = bars_of(c_g)
    res["GUARD-ALL"] = {"bars": b_all, "comparison_vs_BASE": cmp_all, "rates": {"missed_decel": S.rate(c_g, "missed_decel"), "false_decel": S.rate(c_g, "false_decel")}}
    res["descriptive"] = {"GUARD-ALL_vs_SELFSHARP": S.compare(PR, E, c_g, c_ss, rows)}
    passed = lambda b: b["P1"] and b["P2"] and b["G_strict"]  # noqa: E731
    final, arm = b_all, "GUARD-ALL"
    if not passed(b_all):
        ue = np.unique(eid)
        perm = np.random.default_rng(0).permutation(len(ue))
        fold_of = {int(e): int(r_ % 5) for e, r_ in zip(ue[perm], range(len(ue)))}
        fold = np.array([fold_of[int(e)] for e in eid])
        gate = np.zeros(len(m), bool)
        folds, cache = [], {}
        for k in range(5):
            tr = fold != k
            best = None
            for tau in [np.inf, -np.inf] + [float(np.quantile(m[tr & comp], q)) for q in G.QS]:
                if tau not in cache:
                    cache[tau] = cells(np.where((m >= tau)[:, None, None], p_g, p_base))
                A = G.sub(cache[tau], tr)
                cmp, worse = G.g_worse(PR, eid[tr], A, G.sub(c_base, tr), rows, keys)
                adm = (not worse) and not (cmp["false_decel"].get("lo", 0) > 0)
                md = S.rate(A, "missed_decel")
                if adm and (best is None or md < best[1] - 1e-12 or (abs(md - best[1]) <= 1e-12 and tau > best[0])):
                    best = (tau, md)
            gate[fold == k] = m[fold == k] >= best[0]
            folds.append({"fold": k, "tau": best[0], "md_train": best[1]})
            print(f"[ssguard] CF fold {k} tau={best[0]:.3f}", flush=True)
        c_cf = cells(np.where(gate[:, None, None], p_g, p_base))
        cmp_cf, b_cf = bars_of(c_cf)
        res["GUARD-CF"] = {"bars": b_cf, "comparison_vs_BASE": cmp_cf, "folds": folds, "n_sharpened": int(gate.sum()),
                           "rates": {"missed_decel": S.rate(c_cf, "missed_decel"), "false_decel": S.rate(c_cf, "false_decel")}}
        final, arm = b_cf, "GUARD-CF"
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"])
    if ctl_fail:
        word, nl = f"INVALID (control failed: {ctl_fail})", NEXT["INVALID"]
    elif passed(final):
        word, nl = f"PASS ({arm})", NEXT["PASS"]
    else:
        failing = [k for k in ("P1", "P2", "G_strict") if not final[k]]
        word = f"FAIL ({arm}: " + ", ".join(failing) + ")"
        nl = NEXT["FAIL_P1"] if not final["P1"] else NEXT["FAIL_P2"] if not final["P2"] else NEXT["FAIL_G"]
    res.update(verdict=word, next_lever=nl, wall_s=round(time.time() - t0, 1))
    C.jdump(res, out / "SSGUARD_RESULT.json")
    fmt = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    Lx = [word, "", f"# SSGUARD RESULT (mechanical; PREREG_SSGUARD sha256 {pre[:12]})", "", f"tier: {res['tier']}", f"estimator: {G.ESTIMATOR}", "",
          f"MD: BASE-os {S.rate(c_base, 'missed_decel'):.4f} -> GUARD-ALL {S.rate(c_g, 'missed_decel'):.4f} (SELFSHARP {S.rate(c_ss, 'missed_decel'):.4f})",
          f"FD: BASE-os {S.rate(c_base, 'false_decel'):.4f} -> GUARD-ALL {S.rate(c_g, 'false_decel'):.4f} (SELFSHARP {S.rate(c_ss, 'false_decel'):.4f})", ""]
    for nm in ("GUARD-ALL", "GUARD-CF"):
        if nm not in res:
            continue
        b, cmp = res[nm]["bars"], res[nm]["comparison_vs_BASE"]
        Lx += [f"## {nm} vs BASE-os", "", f"P1 {b['P1']} {fmt(b['P1_cell'])}; P2 {b['P2']} {fmt(b['P2_cell'])}; G-strict {b['G_strict']} ({', '.join(b['G_rows_separated_worse']) or 'none'})", "",
               "| S6 row | delta [95% CI] | worse |", "|---|---|---|"] + [f"| {k} | {fmt(c)} | {c.get('worse')} |" for k, c in cmp.items()] + [""]
    Lx += [f"controls: {json.dumps({k: v['pass'] for k, v in ctl.items()})}; degenerate base paths: {res['degenerate_base_path_windows']}", "", f"NEXT: {nl}", "", "ZZSSGUARDENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(Lx), encoding="utf-8")
    print(word, {k: v["pass"] for k, v in ctl.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
