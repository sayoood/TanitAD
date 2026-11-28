"""Markdown tables for RESULT.md, read straight from raw/route_analysis.json and raw/box_nms.json (no number is
typed by hand). Usage: python make_tables.py <raw dir>"""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ci(c, nd=3):
    if not c or c[0] is None:
        return ""
    return f" [{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def chain(R, key="chain_eval_s0"):
    C = R[key]
    links = [("N_nav", "N nav token side == GT"), ("N_nav_informative_only", "N nav (left/right clips only)"),
             ("T_tac_side", "T tactical lat side"), ("T_tac_strict", "T tactical strict TURN_x"),
             ("T_label_lat_v7_side", "label control: lat_v7 side"), ("G_goal_side", "G goal tokens TURN_L/R"),
             ("C_fan_all117", "C fan contains (117)"), ("C_fan_reach", "C fan contains (reach_keep)"),
             ("C_fan_reach_ceil", "C fan contains (reach ∧ ceiling)"), ("C_oracle_cand_correct", "C oracle candidate"),
             ("K_core_pick", "K decoder pick"), ("E_e9_pick", "E E9 emitted pick"),
             ("E_e9_pick_26p1fix", "E E9 pick, §26.1 ceiling fix"), ("R_random_reach", "R random reach candidate"),
             ("C_fan_reach_headagree", "C contains (|Δθ| ≤ 15°)"), ("C_oracle_headagree", "C oracle (|Δθ| ≤ 15°)"),
             ("K_core_pick_headagree", "K decoder pick (|Δθ| ≤ 15°)"), ("E_e9_pick_headagree", "E E9 pick (|Δθ| ≤ 15°)"),
             ("R_random_reach_headagree", "R random (|Δθ| ≤ 15°)")]
    cols = ["turnL", "turnR", "turn", "straight", "gentle"]
    out = ["| link | " + " | ".join(f"{c} (n={C[c]['n_windows']}, {C[c]['n_episodes']} ep)" for c in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for k, nm in links:
        cells = []
        for c in cols:
            v = C[c].get(k, {})
            cells.append("–" if v.get("rate") is None else f"{v['rate']:.3f}{ci(v['ci95'], 2)}" +
                         (f" (n {v['n']})" if v.get("n") != C[c]["n_windows"] else ""))
        out.append(f"| {nm} | " + " | ".join(cells) + " |")
    for nm, k in (("ADE oracle-117 (m)", "ade_oracle117"), ("ADE decoder pick (m)", "ade_core"),
                  ("ADE E9 pick (m)", "ade_e9")):
        out.append(f"| {nm} | " + " | ".join(f"{C[c][k]['mean']:.2f}{ci(C[c][k]['ci95'], 2)}"
                                              if C[c][k]["mean"] is not None else "–" for c in cols) + " |")
    return "\n".join(out)


def decision(R, key="decision_eval_s0"):
    D = R[key]
    out = ["| drop (GT-turn windows) | point | 95 % CI (paired episode-cluster bootstrap) | CI excludes 0 |",
           "|---|---|---|---|"]
    for k, v in D["drops"].items():
        out.append(f"| {k} | {v['drop']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['excludes_0']} |")
    out.append(f"\nrates on GT-turn windows: {json.dumps(D['rates_turn'])}  →  **break: {D['break']}**")
    return "\n".join(out)


def confusion(R, key="confusion_eval_s0"):
    C = R[key]
    out = ["| link | GT L → L / S / R | GT S → L / S / R | GT R → L / S / R |", "|---|---|---|---|"]
    for k, t in C.items():
        cells = [f"{t[g]['L']} / {t[g]['S']} / {t[g]['R']}" + (f" (n/a {t[g]['n/a']})" if t[g]["n/a"] else "")
                 for g in ("GT_L", "GT_S", "GT_R")]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out)


METRICS = [("ade", "ADE"), ("fde", "FDE"), ("along_signed_6s", "along 6 s (signed)"), ("along_abs_2s", "|along| 2 s"),
           ("along_abs_6s", "|along| 6 s"), ("speed_mae_0_2s", "speed MAE 0–2 s"), ("cross_abs_2s", "|cross| 2 s"),
           ("cross_abs_6s", "|cross| 6 s"), ("heading_mae_0_2s_deg", "heading MAE 0–2 s (°)"),
           ("curv_mae_0_2s", "curv MAE 0–2 s"), ("term_heading_err_deg", "terminal heading err (°)"),
           ("dir_correct", "turn/straight dir correct"), ("nav_complies", "nav compliance"),
           ("turn_ratio_posthoc", "turn ratio θ/θ_gt (post-hoc)")]


def levers(L, cls, keys=None, metrics=METRICS):
    keys = keys or list(L)
    out = ["| arm | " + " | ".join(m[1] for m in metrics) + " |", "|---|" + "---|" * len(metrics)]
    for k in keys:
        r = L[k]
        cells = []
        for mk, _ in metrics:
            e = r[cls].get(mk)
            if e is None or e["mean"] is None:
                cells.append("–")
                continue
            s = f"{e['mean']:.3f}"
            if "delta_vs_V0" in e and e["delta_vs_V0"] is not None:
                s += f" (Δ {e['delta_vs_V0']:+.3f}{ci(e['delta_ci95'])})"
            cells.append(s)
        out.append(f"| {k} (changed {r['pick_changed_frac']:.2f}) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def fan(R, key="fan_eval_s0"):
    F = R[key]
    ks = [("e9_top8_circstd_deg", "top-8 (E9) circ. std θ (°)"), ("e9_top8_range_deg", "top-8 (E9) range θ (°)"),
          ("e9_top8_LR", "top-8 (E9) holds L AND R"), ("core_top8_LR", "top-8 (decoder) holds L AND R"),
          ("top8_has_correct", "top-8 (E9) holds a dir-correct cand."),
          ("rho_e9", "Spearman(E9 score, −ADE)"), ("rho_core", "Spearman(decoder score, −ADE)"),
          ("rho_sampler_conf", "Spearman(sampler conf, −ADE)"), ("rho_random", "CONTROL Spearman(random, −ADE)")]
    cols = ["turn", "straight", "gentle", "all"]
    out = ["| statistic | " + " | ".join(f"{c} (n={F[c]['n']})" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for k, nm in ks:
        out.append(f"| {nm} | " + " | ".join(f"{F[c][k]['mean']:.3f}{ci(F[c][k]['ci95'], 3)}"
                                              if F[c][k]["mean"] is not None else "–" for c in cols) + " |")
    return "\n".join(out)


def main():
    raw = Path(sys.argv[1])
    R = json.loads((raw / "route_analysis.json").read_text(encoding="utf-8"))
    print("## chain eval_s0\n" + chain(R))
    print("\n## decision eval_s0\n" + decision(R))
    if "chain_eval_s1" in R:
        print("\n## chain eval_s1\n" + chain(R, "chain_eval_s1"))
        print("\n## decision eval_s1\n" + decision(R, "decision_eval_s1"))
    print("\n## confusion eval_s0\n" + confusion(R))
    for cls in ("turn", "straight", "gentle", "all"):
        print(f"\n## levers eval_s0 — {cls}\n" + levers(R["levers_eval_s0"], cls))
    if "A1" in R:
        for cls in ("turn", "straight", "gentle", "all"):
            print(f"\n## A1 levers eval_s0 — {cls}\n" + levers(R["A1"]["levers_eval_s0"], cls))
    print("\n## fan eval_s0\n" + fan(R))
    if "chain_reel_s0" in R:
        print("\n## chain reel_s0\n" + chain(R, "chain_reel_s0"))
        print("\n## decision reel_s0\n" + decision(R, "decision_reel_s0"))


if __name__ == "__main__" and len(sys.argv) == 2:
    main()


# ---------------------------------------------------------------------------------------------------------------- #
# compact tables for RESULT.md (all levers / bounds in one place, with the seed-1 replicate)                       #
# ---------------------------------------------------------------------------------------------------------------- #
def _d(L, k, cls, m="ade"):
    e = L.get(k, {}).get(cls, {}).get(m)
    if not e or e.get("delta_vs_V0") is None:
        return "–"
    c = e["delta_ci95"]
    return f"{e['delta_vs_V0']:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"


def lever_summary(rows):
    """rows: [(label, L_s0, L_s1, key, bar)]"""
    out = ["| arm (fit on TRAIN) | picks changed | ΔADE all [CI] | ΔADE GT-turn [CI] | Δ turn dir-correct [CI] | "
           "ΔADE GT-straight [CI] | seed 1: ΔADE all | seed 1: ΔADE turn | bar |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, L0, L1, k, bar in rows:
        r = L0[k]
        out.append(f"| {lab} | {r['pick_changed_frac']:.2f} | {_d(L0, k, 'all')} | {_d(L0, k, 'turn')} | "
                   f"{_d(L0, k, 'turn', 'dir_correct')} | {_d(L0, k, 'straight')} | "
                   f"{_d(L1, k, 'all') if L1 else '–'} | {_d(L1, k, 'turn') if L1 else '–'} | "
                   f"{'CLEARS' if bar and bar.get('CLEARS') else 'FAILED' if bar else 'bound'} |")
    return "\n".join(out)


FF = [("ade", "ADE"), ("fde", "FDE"), ("along_abs_6s", "LON |along| 6 s"), ("along_signed_6s", "LON along 6 s signed"),
      ("speed_mae_0_2s", "LON speed MAE 0–2 s"), ("cross_abs_6s", "LAT |cross| 6 s"),
      ("heading_mae_0_2s_deg", "LAT heading MAE 0–2 s °"), ("curv_mae_0_2s", "LAT curv MAE 0–2 s"),
      ("term_heading_err_deg", "LAT term. heading err °"), ("dir_correct", "TAC dir correct"),
      ("nav_complies", "STR nav compliance")]


def four_family(L, keys, cls):
    out = ["| arm | " + " | ".join(n for _, n in FF) + " |", "|---|" + "---|" * len(FF)]
    for lab, k in keys:
        r = L[k][cls]
        out.append(f"| {lab} | " + " | ".join("–" if r.get(m) is None or r[m]["mean"] is None else f"{r[m]['mean']:.3f}"
                                              for m, _ in FF) + " |")
    return "\n".join(out)


def compact(raw):
    R = json.loads((raw / "route_analysis.json").read_text(encoding="utf-8"))
    a2 = json.loads((raw / "a2_rescorer.json").read_text(encoding="utf-8"))
    a3 = json.loads((raw / "a3_rescorer.json").read_text(encoding="utf-8"))
    L0, L1 = R["levers_eval_s0"], R["levers_eval_s1"]
    A0, A1_ = R["A1"]["levers_eval_s0"], R["A1"].get("levers_eval_s1")
    bs, ba = R["bar_all_levers"], R["A1"]["bar_all"]
    rows = []
    for k in L0:
        if k in ("V0|None", "ORACLE|None"):
            continue
        rows.append((f"SPEC §5 {k}", L0, L1, k, bs.get(k)))
    for k in A0:
        if k.startswith("W"):
            rows.append((f"A1 {k}", A0, A1_, k, ba.get(k)))
    for k in ("X1", "X2_no_nav", "X3_no_goalhead", "CTRL_shuffled_target"):
        rows.append((f"A2 {k}", a2["levers_eval_s0g"], a2["levers_eval_s1"], k, a2["bar_eval_s0g"].get(k)))
    for k in ("Y1", "Y2_no_nav"):
        rows.append((f"A3 {k}", a3["levers_eval_s0g"], a3["levers_eval_s1"], k, a3["bar_eval_s0g"].get(k)))
    B = R["A4_bounds"]["eval_s0"]
    for k in ("B1", "B1t", "B2", "B3", "B4", "ORACLE"):
        rows.append((f"A4 bound {k} (label-side)", B, None, k, None))
    print("## lever summary\n" + lever_summary(rows))
    keys = [("V0 shipped E9", "V0|None"), ("V0c §26.1 ceiling", "V0c|None"), ("V5 decoder pick", "V5|None"),
            ("V2 nav filter", "V2|0.05"), ("V3 navc ×10", "V3|10"), ("ORACLE-117", "ORACLE|None")]
    for cls in ("turn", "straight", "all"):
        print(f"\n## four families eval_s0 — {cls}\n" + four_family(L0, keys, cls))
    print("\n## four families A1/A4 — turn\n" + four_family({**A0, **B}, [("W3 E9 graft off", "W3|0"),
                                                                        ("B1t timed-direction bound", "B1t"),
                                                                        ("B3 speed bound", "B3")], "turn"))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "compact":
    compact(Path(sys.argv[1]))
