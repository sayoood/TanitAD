"""H-DD-FITGAP, M2b only (PARTIAL, deliverable 3). Frozen rules: FROZEN_CARDS.md F1-F5 + FROZEN_AMENDMENT_1.md A2/A3.
TRAIN side: M2b's own rollout states (bank_s0 / bank_s1) over TRAIN GT-launch windows.  EVAL side: banked T1-kin dump of plain M2b ("B", m = 10, eval-141).
Primary statistic = R2 (executed-chord speed over the last 0.2 s before the tick) on BOTH sides; R1 on the TRAIN side beside.
Usage: python -I fitgap_m2b.py   ->  ../fitgap_m2b.json, ../fitgap_m2b.md      (numpy only; dumps must be extracted under raw/eval_dumps/)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import stats as S  # noqa: E402

ART = Path("E:/Projects/TanitAD-artifacts")
DT = 0.2
EVAL_B = {"s0": C.RAW / "eval_dumps" / "dump_B_s0_m10", "s1": C.RAW / "eval_dumps" / "dump_B_s1_m10"}
CTRL_DAGOACC = {"s0": ART / "v8f-dagoacc-t1" / "dump_DAGOACC_s0_m10_causal_heads", "s1": ART / "v8f-dagoacc-t1" / "dump_DAGOACC_s1_m10_causal_heads"}
# the M-LAUNCH instrument's own R2 means for DAG-OACC (launch_intent_mlaunch/MLAUNCH.json sensitivity.cells.main_s*.y.mean, MEASURED, INHERITED here as a control value)
MLAUNCH_R2_DAGOACC = {"s0": -0.7991361833219048, "s1": -0.9808361793597838}
DAG_LIMIT = {sd: {lim: ART / "v8f-dagger-t1" / f"dump_DAG_{sd}_m10_{lim}_heads" for lim in ("causal", "logged")} for sd in ("s0", "s1")}


def load_dump(d):
    files = sorted(Path(d).glob("ep*.npz"))
    assert len(files) == 141, (d, len(files))
    zs = [np.load(f) for f in files]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "v0", "eid", "ws")}


def eval_w(dump):
    """Per-window R2 statistic w = mean over ticks 1,2 of (v_own - v_GT) with chords over the last 0.2 s; launch mask (hmc_read form)."""
    g, cl, v0 = dump["g"], dump["cl"], dump["v0"]
    v6 = np.hypot(*(g[:, 29] - g[:, 28]).T) / DT
    launch = (v0 < 6.0) & (v6 >= v0 + 1.0)
    diffs = []
    for n in (1, 2):
        vo = np.hypot(*(cl[:, 10 * n - 1] - cl[:, 10 * n - 2]).T) / DT
        vg = np.hypot(*(g[:, 10 * n - 1] - g[:, 10 * n - 2]).T) / DT
        diffs.append(vo - vg)
    return np.mean(diffs, axis=0), launch


def boot_eval(dump):
    w, launch = eval_w(dump)
    eid = dump["eid"]
    ue = np.unique(eid[launch])
    pos = np.searchsorted(ue, eid[launch])
    num = np.bincount(pos, weights=w[launch], minlength=len(ue))
    den = np.bincount(pos, minlength=len(ue)).astype(float)
    b = S.ratio_boot(num, den)
    return b, {"n_launch_windows": int(launch.sum()), "n_launch_episodes": int(len(ue)), "n_windows": int(len(w)), "mean": float(w[launch].mean())}


def train_side(seed):
    tab = C.attach_logged(C.load_table(seed), C.load_logged())
    lg = C.load_logged()
    G, V = lg["G"], lg["V"]
    order = np.lexsort((tab["tick"], tab["t0"], tab["clip_g"]))
    g = lambda k: tab[k][order]
    tick, clip, t0 = g("tick"), g("clip_g").astype(np.int64), g("t0").astype(np.int64)
    a, b = slice(0, None, 2), slice(1, None, 2)
    assert (tick[a] == 1).all() and (tick[b] == 2).all() and (clip[a] == clip[b]).all() and (t0[a] == t0[b]).all()
    cidx, T = g("cidx")[a], g("T")[a]
    off = lg["offs"][cidx]
    t0w, clipw = t0[a], clip[a]
    ok_f = (t0w >= 1) & (t0w + 30 <= T - 1)
    i0 = np.where(ok_f, off + t0w, 0)
    i29 = np.where(ok_f, off + t0w + 29, 0)
    i30 = np.where(ok_f, off + t0w + 30, 0)
    v0 = V[i0]
    v6 = np.hypot(*(G[i30, :2] - G[i29, :2]).T) / DT
    launch = ok_f & np.isfinite(v0) & np.isfinite(v6) & (v0 < 6.0) & (v6 >= v0 + 1.0)
    res_w = {}
    w_r2, w_r1 = [], []
    for sl, n in ((a, 1), (b, 2)):
        j = t0w + 10 * n
        ij = np.where(ok_f, off + j, 0)
        px, py = g("px")[sl], g("py")[sl]
        vo2 = np.hypot(g("ox")[sl] - px, g("oy")[sl] - py) / DT
        vg2 = np.hypot(*(G[ij, :2] - G[np.where(ok_f, ij - 1, 0), :2]).T) / DT
        w_r2.append(vo2 - vg2)
        vo1 = g("v_own")[sl]
        vg1 = np.hypot(*(G[np.where(ok_f, ij + 1, 0), :2] - G[np.where(ok_f, ij - 1, 0), :2]).T) / (2 * DT)
        w_r1.append(vo1 - vg1)
    w_r2, w_r1 = np.mean(w_r2, axis=0), np.mean(w_r1, axis=0)
    reason = g("reason")
    both_ok = (reason[a] == C.OK) & (reason[b] == C.OK)
    out = {"n_windows": int(len(t0w)), "n_launch_windows": int(launch.sum())}

    def boot(w, mask):
        uc = np.unique(clipw[mask])
        pos = np.searchsorted(uc, clipw[mask])
        num = np.bincount(pos, weights=w[mask], minlength=len(uc))
        den = np.bincount(pos, minlength=len(uc)).astype(float)
        bb = S.ratio_boot(num, den)
        return bb, {"n_windows": int(mask.sum()), "n_clips": int(len(uc)), "mean": float(w[mask].mean())}

    b2, i2 = boot(w_r2, launch)
    out["R2_all_records"] = {k: v for k, v in b2.items() if k != "boots"} | i2
    out["_boots_R2"] = b2["boots"]
    b1, i1 = boot(w_r1, launch)
    out["R1_all_records_beside"] = {k: v for k, v in b1.items() if k != "boots"} | i1
    bo, io = boot(w_r2, launch & both_ok)
    out["R2_both_ticks_ok_beside"] = {k: v for k, v in bo.items() if k != "boots"} | io
    # raw stats for the r CI
    uc = np.unique(clipw[launch])
    pos = np.searchsorted(uc, clipw[launch])
    out["_num"] = np.bincount(pos, weights=w_r2[launch], minlength=len(uc))
    out["_den"] = np.bincount(pos, minlength=len(uc)).astype(float)
    out["train_launch_windows_all_clips"] = int(launch.sum())
    out["train_launch_clips"] = int(len(uc))
    return out


def classify(r):
    if r is None or not np.isfinite(r):
        return "UNDEFINED"
    return "FIT-LIMITED" if r >= 0.5 else ("GEN-GAP" if r <= 0.25 else "MIXED")


def main():
    out = {"deliverable": "H-DD-FITGAP (M2b only, PARTIAL)", "frozen_cards_sha256": C.FROZEN_SHA, "frozen_amendment_1_sha256": C.AMEND_SHA,
           "tier": "TRAIN side: T1-kin rollout states of M2b on TRAIN GT-launch windows (the DAgger bank); EVAL side: T1-kin dump of plain M2b on eval-141 (reference only)",
           "evidence_class": "MEASURED", "statistic": "R2 executed-chord speed, ticks 1-2 mean, v_own - v_GT (m/s); D = -C", "seeds": {}, "controls": {}}
    # controls: my eval-side R2 code must reproduce the M-LAUNCH instrument's own R2 mean for DAG-OACC (launch set 327 / 49)
    for sd in ("s0", "s1"):
        bb, inf = boot_eval(load_dump(CTRL_DAGOACC[sd]))
        out["controls"][f"R2_DAGOACC_{sd}_vs_MLAUNCH.json"] = {"mine": bb["point"], "banked": MLAUNCH_R2_DAGOACC[sd], "abs_diff": abs(bb["point"] - MLAUNCH_R2_DAGOACC[sd]),
                                                              "n_launch_windows": inf["n_launch_windows"], "n_launch_episodes": inf["n_launch_episodes"],
                                                              "PASS": bool(abs(bb["point"] - MLAUNCH_R2_DAGOACC[sd]) < 1e-9 and inf["n_launch_windows"] == 327 and inf["n_launch_episodes"] == 49)}
    out["controls"]["PASS"] = all(v["PASS"] for k, v in out["controls"].items() if k != "PASS")
    assert out["controls"]["PASS"], out["controls"]
    # limit setting sensitivity (DAG arm, both forms exist): eval-side C under limit logged vs causal
    out["limit_sensitivity_DAG_arm_beside"] = {}
    for sd in ("s0", "s1"):
        cc = boot_eval(load_dump(DAG_LIMIT[sd]["causal"]))[0]["point"]
        ll = boot_eval(load_dump(DAG_LIMIT[sd]["logged"]))[0]["point"]
        out["limit_sensitivity_DAG_arm_beside"][sd] = {"C_eval_causal": cc, "C_eval_logged": ll, "logged_minus_causal": ll - cc}
    classes = []
    for sd in ("s0", "s1"):
        tr = train_side(sd)
        ev_b, ev_info = boot_eval(load_dump(EVAL_B[sd]))
        num_tr, den_tr = tr.pop("_num"), tr.pop("_den")
        tr.pop("_boots_R2")
        Ctr, Cev = tr["R2_all_records"]["point"], ev_b["point"]
        Dtr, Dev = -Ctr, -Cev
        r_info = None
        r = Dtr / Dev if Dev > 0 else None
        if Dev > 0:
            # D = -C: resample sums with a sign flip; r = (-num_tr/den_tr) / (-num_ev/den_ev)
            w_e, la_e = eval_w(load_dump(EVAL_B[sd]))
            eid = load_dump(EVAL_B[sd])["eid"]
            ue = np.unique(eid[la_e])
            pe = np.searchsorted(ue, eid[la_e])
            num_ev = np.bincount(pe, weights=w_e[la_e], minlength=len(ue))
            den_ev = np.bincount(pe, minlength=len(ue)).astype(float)
            r_info = S.two_sample_ratio_boot(num_tr, den_tr, num_ev, den_ev)
            r = r_info["point"]
        cls = classify(r)
        classes.append(cls)
        out["seeds"][sd] = {"C_train_R2": tr["R2_all_records"], "C_train_R1_beside": tr["R1_all_records_beside"], "C_train_R2_both_ticks_ok_beside": tr["R2_both_ticks_ok_beside"],
                            "train_launch_windows_all_clips": tr["train_launch_windows_all_clips"], "train_launch_clips": tr["train_launch_clips"],
                            "C_eval_B_R2": {k: v for k, v in ev_b.items() if k != "boots"} | ev_info,
                            "D_train": Dtr, "D_eval": Dev, "r": r, "r_CI95": ([r_info["lo"], r_info["hi"]] if r_info else None), "r_frac_boot_den_nonpositive": (r_info or {}).get("frac_boot_den_nonpositive"),
                            "class": cls}
        shift = out["limit_sensitivity_DAG_arm_beside"][sd]["logged_minus_causal"]      # C(causal) = C(logged) - d, D = -C  =>  D(causal) = D(logged) + d, d = C(logged) - C(causal) of the DAG arm (a different arm: ESTIMATED)
        out["seeds"][sd]["r_causal_adjusted_beside"] = {"D_eval_adjusted": Dev + shift, "r_adjusted": (Dtr / (Dev + shift) if Dev + shift > 0 else None),
                                                        "class_adjusted": classify(Dtr / (Dev + shift) if Dev + shift > 0 else None),
                                                        "basis": "DAG arm logged-vs-causal shift applied to plain M2b; ESTIMATED; enters no verdict"}
    # PARTIAL aggregate (F5)
    k_fit, k_gen = classes.count("FIT-LIMITED"), classes.count("GEN-GAP")
    rem = 6 - len(classes)
    out["partial"] = {"checkpoints_observed": len(classes), "of": 6, "classes": classes,
                      "FIT-LIMITED_needs_of_remaining": (max(0, 5 - k_fit) if k_fit + rem >= 5 else "UNREACHABLE"),
                      "GEN-GAP_needs_of_remaining": (max(0, 5 - k_gen) if k_gen + rem >= 5 else "UNREACHABLE"),
                      "remaining_checkpoints": ["DAG-OACC s0", "DAG-OACC s1", "REP s0", "REP s1"], "note": "the brief says 1 of 6; M2b s0 and s1 are two of the card's six"}
    n_launch = out["seeds"]["s0"]["train_launch_windows_all_clips"]
    out["other_checkpoints_cost_ESTIMATE"] = {
        "rate_basis": "RUN_SUMMARY hacc-lat: B_s0_m10 (2,399 eval windows, 3 ticks) 290 s on the dev-box CPU, i.e. ~8.3 windows/s (ESTIMATED, INHERITED log)",
        "train_launch_windows_all_clips": n_launch, "launch_windows_only_all_clips_hours_per_ckpt": n_launch / 8.3 / 3600,
        "10pct_clip_subset_launch_windows_only_hours_per_ckpt": 0.1 * n_launch / 8.3 / 3600,
        "10pct_clip_subset_ALL_windows_hours_per_ckpt": 0.1 * 306566 / 8.3 / 3600,
        "checkpoints": 4,
        "inputs_banked": "eval-side dumps: DAG-OACC s0/s1 E:/.../v8f-dagoacc-t1; REP s0/s1 F:/TanitAD-scratch/dagorc_t1_out (MLAUNCH.json runs); checkpoints: v8f-dagoacc/M2b_s*_dagoacc/ckpt_last.pt (REP ckpts: UNVERIFIED location)"}
    (C.PKG / "fitgap_m2b.json").write_text(json.dumps(out, indent=1, default=float))
    write_md(out, C.PKG / "fitgap_m2b.md")
    print(json.dumps({"seeds": {s: {k: out["seeds"][s][k] for k in ("D_train", "D_eval", "r", "r_CI95", "class")} for s in out["seeds"]}, "partial": out["partial"]}, indent=1, default=float))


def write_md(o, path):
    L = ["# H-DD-FITGAP, M2b only (PARTIAL; deliverable 3)",
         f"\nFrozen cards sha256 `{o['frozen_cards_sha256']}`; amendment 1 sha256 `{o['frozen_amendment_1_sha256']}`.",
         "Evidence MEASURED (ours; `fitgap_m2b.json`). TRAIN side = M2b's own rollout states (the DAgger bank) on TRAIN GT-launch windows; EVAL side = the banked T1-kin dump of plain M2b ('B', m = 10, eval-141, reference only, never a selector). "
         "Statistic = R2 (executed-chord speed over the last 0.2 s before the tick), ticks 1-2 mean of v_own - v_GT, per launch window; D = -C is the speed deficit (m/s). Estimator: cluster bootstrap over launch-bearing clips / episodes, B = 2000, seed 0 (answers 'another draw of clips / episodes' only; blind to training and inference variance).",
         f"\nControls PASS: {o['controls']['PASS']} (my eval-side R2 reproduces the M-LAUNCH instrument's own DAG-OACC R2 mean to < 1e-9 on 327 windows / 49 episodes).\n"]
    rows = []
    for s, d in o["seeds"].items():
        t, e = d["C_train_R2"], d["C_eval_B_R2"]
        rows.append([f"M2b {s}", f"{t['point']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}] ({t['n_windows']} win / {t['n_clips']} clips)",
                     f"{e['point']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] ({e['n_launch_windows']} win / {e['n_launch_episodes']} ep)",
                     (f"{d['r']:.3f} [{d['r_CI95'][0]:.3f}, {d['r_CI95'][1]:.3f}]" if d["r"] is not None else "undefined"), d["class"]])
    L.append(C.md_table(rows, ["checkpoint", "C_train (m/s) [95 % CI]", "C_eval, plain M2b dump (m/s) [95 % CI]", "r = D_train / D_eval [95 % CI]", "class (frozen: >= 0.5 FIT, <= 0.25 GEN)"]))
    p = o["partial"]
    L.append(f"\n**PARTIAL verdict:** {p['checkpoints_observed']} of 6 checkpoints observed ({p['classes']}); the card's >= 5-of-6 rule is not evaluable. "
             f"FIT-LIMITED needs {p['FIT-LIMITED_needs_of_remaining']} of the remaining 4; GEN-GAP needs {p['GEN-GAP_needs_of_remaining']}. Remaining: {', '.join(p['remaining_checkpoints'])}.")
    ls = o["limit_sensitivity_DAG_arm_beside"]
    L.append("\nConfig mismatch (disclosed): the plain-M2b dumps ran `--limit logged`, the bank rollouts `--limit causal`. Beside, on the DAG arm that exists in both forms, C_eval(logged) - C_eval(causal) = "
             + ", ".join(f"{s}: {v['logged_minus_causal']:+.4f} m/s" for s, v in ls.items()) + " (enters no verdict).")
    L.append("Beside (ESTIMATED, no verdict): shifting the plain-M2b eval deficit by the DAG arm's logged-to-causal change gives r = "
             + ", ".join(f"{s}: {d['r_causal_adjusted_beside']['r_adjusted']:.3f} ({d['r_causal_adjusted_beside']['class_adjusted']})" for s, d in o["seeds"].items())
             + ". R1 (state speed) on the TRAIN side, beside: " + ", ".join(f"{s}: {d['C_train_R1_beside']['point']:+.4f}" for s, d in o["seeds"].items()) + " m/s.")
    c = o["other_checkpoints_cost_ESTIMATE"]
    L.append(f"\nCost of the other 4 checkpoints (ESTIMATED from a 290 s / 2,399-window dev-box CPU log): TRAIN launch windows on all clips = {c['train_launch_windows_all_clips']}; "
             f"launch-windows-only roll-outs ~{c['launch_windows_only_all_clips_hours_per_ckpt']:.1f} h per checkpoint (all clips), ~{c['10pct_clip_subset_launch_windows_only_hours_per_ckpt']:.2f} h on a 10 % clip subset; "
             f"all windows of a 10 % clip subset ~{c['10pct_clip_subset_ALL_windows_hours_per_ckpt']:.1f} h per checkpoint.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
