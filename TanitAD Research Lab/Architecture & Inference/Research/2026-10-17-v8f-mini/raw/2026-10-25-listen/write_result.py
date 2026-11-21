"""write_result.py -- renders the TABLES of LISTEN_RESULT.md from READ.json (no number is typed by hand).

    python write_result.py [--read READ.json] [--out LISTEN_RESULT_tables.md]

The narrative (reading of the numbers) is written separately and prepended; this file only formats what READ.json holds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = ("S", "T-lat", "T-lon", "O")
COLS = ("scene", "ego", "nav", "limit", "ck", "code", "age")
COL_LABEL = {"scene": "scene", "ego": "ego", "nav": "nav", "limit": "limit", "ck": "S checkpoints", "code": "T code", "age": "tick ages", "all": "ALL (row reference)"}
UNIT = {"S": "m", "T-lat": "F1", "T-lon": "F1", "O": "m", "O-ade2": "m"}
ROW_METRIC = {"S": "S: checkpoint position error @ 50 m, donor - clean", "T-lat": "T-lat: lateral macro-F1, clean - donor",
              "T-lon": "T-lon: longitudinal macro-F1, clean - donor", "O": "O: ADE6, donor - clean"}


def f(v, row, signed=True):
    if v is None:
        return "NA"
    d = 3 if UNIT.get(row, "m") == "m" else 4
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def ci(c, row):
    return f"[{f(c['lo'], row)}, {f(c['hi'], row)}]"


def grid_table(R, seed):
    g, A = R["grid"][seed], R["analysis"]
    out = [f"| row (metric) | input | delta ({'m or F1'}) | 95 % CI | separated | n windows / episodes | ratio n = d(L,X)/d(L,all) | donor-seed floor | per-seed flag ({seed}) | LISTENS (both seeds) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        ent = g["rows"][row]
        for col in COLS + ("all",):
            c = ent["cells"][col]
            if c is None:
                out.append(f"| {row} | {COL_LABEL[col]} | not an input | | | | | | | |")
                continue
            if col == "all":
                out.append(f"| {row} | **{COL_LABEL[col]}** | **{f(c['delta'], row)} {UNIT[row]}** | {ci(c, row)} | {'yes' if c['separated'] else 'no'} | {c['n_windows']} / {c['n_episodes']} | 1 by definition ({'defined' if ent['all_defined'] else 'UNDEFINED: delta(L,all) not > 0 and separated'}) | {f(A['floor_donor_seed'][row]['all'], row, False)} | | |")
                continue
            n = c.get("n")
            fl = A["seed_flags"][seed][row][col]
            ls = A["listens"][row][col]
            out.append(f"| {row} | {COL_LABEL[col]} | {f(c['delta'], row)} {UNIT[row]} | {ci(c, row)} | {'yes' if c['separated'] else 'no'} | {c['n_windows']} / {c['n_episodes']} | "
                       f"{'NA' if n is None else f'{n:.3f}'} | {f(A['floor_donor_seed'][row][col], row, False)} | {'listens' if fl else 'no'} | {'**LISTENS**' if ls else 'no'} |")
    # ADE2 beside
    ent = g["beside"].get("O-ade2")
    if ent:
        out.append("")
        out.append("O, ADE2 beside (donor - clean, m):")
        out.append("")
        out.append("| input | delta (m) | 95 % CI | separated |")
        out.append("|---|---|---|---|")
        for col in COLS + ("all",):
            c = ent["cells"][col]
            if c is not None:
                out.append(f"| {COL_LABEL[col]} | {f(c['delta'], 'O-ade2')} | {ci(c, 'O-ade2')} | {'yes' if c['separated'] else 'no'} |")
    return "\n".join(out)


def clean_levels(R):
    out = ["| row | clean level, seed 0 | clean level, seed 1 |", "|---|---|---|"]
    for row in ROWS:
        out.append(f"| {ROW_METRIC[row]} | {f(R['grid']['s0']['rows'][row]['clean_level'], row, False)} {UNIT[row]} | {f(R['grid']['s1']['rows'][row]['clean_level'], row, False)} {UNIT[row]} |")
    return "\n".join(out)


def largest_table(R):
    A = R["analysis"]
    out = ["| seed | row | largest-n cell (R1: argmax of unclipped n) | delta [95 % CI] | n | listens at this seed (delta > 0, separated, above donor floor) | in the hypothesis set |",
           "|---|---|---|---|---|---|---|"]
    setof = {"S": "scene or nav (P-UPPER)", "T-lat": "scene or nav (P-UPPER)", "O": "S checkpoints or T code (P-LOWER)"}
    for s in ("s0", "s1"):
        for row in ROWS:
            ent = R["grid"][s]["rows"][row]
            hyp = "P-UPPER" if row in ("S", "T-lat") else ("P-LOWER" if row == "O" else None)
            if hyp:
                pr = A["hypotheses"][hyp][s]["rows"][row]
                col = pr["argmax"]
            else:
                col = None
                best = -9e9
                for cc in COLS:
                    c = ent["cells"][cc]
                    if c is not None and c.get("n") is not None and c["n"] > best:
                        best, col = c["n"], cc
                pr = None
            if col is None:
                out.append(f"| {s} | {row} | none (n undefined: delta(L,all) not > 0 and separated) | | | | |")
                continue
            c = ent["cells"][col]
            listens = A["seed_flags"][s][row][col]
            insets = (f"{'yes' if pr['in_set'] else 'no'} ({setof[row]})") if pr else "(no hypothesis on this row)"
            out.append(f"| {s} | {row} | **{COL_LABEL[col]}** | {f(c['delta'], row)} {UNIT[row]} {ci(c, row)} | {c['n']:.3f} | {'yes' if listens else 'no'} | {insets} |")
    return "\n".join(out)


def hyp_table(R):
    A = R["analysis"]
    out = ["| hypothesis | seed | row | largest-n cell | in set | listens at seed | row holds | hypothesis holds at seed |", "|---|---|---|---|---|---|---|---|"]
    for h in ("P-UPPER", "P-LOWER"):
        for s in ("s0", "s1"):
            hh = A["hypotheses"][h][s]
            for row, pr in hh["rows"].items():
                out.append(f"| {h} | {s} | {row} | {COL_LABEL.get(pr['argmax'], 'NA') if pr['argmax'] else 'NA'} | {pr['in_set']} | {pr['listens_at_seed']} | {pr['holds']} | {hh['holds']} |")
    return "\n".join(out)


def qnav_table(R):
    out = ["| seed | arm (donor nav injected at) | ADE6 change vs clean (m) | 95 % CI | separated |", "|---|---|---|---|---|"]
    lab = {"nav@S": "S only (S -> T -> O)", "nav@T": "T only (T -> O)", "nav@O": "O only (the O x nav cell)", "nav@STO": "S + T + O"}
    for s in ("s0", "s1"):
        q = R["grid"][s]["qnav"]
        for a in ("nav@S", "nav@T", "nav@O", "nav@STO"):
            c = q[a]
            out.append(f"| {s} | {lab[a]} | {f(c['delta'], 'O')} | {ci(c, 'O')} | {'yes' if c['separated'] else 'no'} |")
        d = q["D_T_minus_O"]
        out.append(f"| {s} | **D = ADE6(nav at T only) - ADE6(nav at O only), paired** | **{f(d['delta'], 'O')}** | {ci(d, 'O')} | {'yes' if d['separated'] else 'no'} |")
    return "\n".join(out)


def controls_md(R):
    C = R["controls"]
    L = []
    L.append("**C2. Reproduction** (the clean pass vs the banked own-code eval of the same checkpoint; stop threshold 1e-3). MEASURED: `repro_s0.json`, `READ.json -> controls.reproduction`.")
    L.append("")
    L.append("| seed | reference | max abs metric diff (S@50, ADE6, T-lat F1, T-lon F1) | max per-window diff (traj30 / ADE6 / goal_prob / ckpt) | argmax flips lat / lon | pass |")
    L.append("|---|---|---|---|---|---|")
    for s in ("s0", "s1"):
        for lab, key in (("banked box-2 GPU eval", "banked_box2"), ("SFRESH CPU re-run", "cpu_rerun_sfresh")):
            d = C["reproduction"][s][key]
            m = d["max_abs_diff"]
            L.append(f"| {s} | {lab} | {d['max_metric_diff']:.3e} | {m['traj30']:.2e} / {m['ade6']:.2e} / {m['goal_prob']:.2e} / {m['ckpt_tick_frame']:.2e} | {d['argmax_lat_flips']} / {d['argmax_lon_flips']} | {d['pass_1e-3']} |")
    L.append("")
    L.append("Metrics, mine vs banked (box-2) :")
    L.append("")
    L.append("| seed | metric | mine | banked | abs diff |")
    L.append("|---|---|---|---|---|")
    for s in ("s0", "s1"):
        for k, v in C["reproduction"][s]["banked_box2"]["metrics_mine_vs_banked"].items():
            L.append(f"| {s} | {k} | {v['mine']:.6f} | {v['banked']:.6f} | {v['abs_diff']:.2e} |")
    L.append("")
    L.append("**C1. Identity** (donor = self => every per-window array bit-identical to the clean pass, so every delta is exactly 0.0). MEASURED: `runs/s*_id.npz`.")
    L.append("")
    L.append("| seed | cells compared | max abs diff vs clean | cells not identical | pass |")
    L.append("|---|---|---|---|---|")
    for s, d in C["identity"].items():
        L.append(f"| {s} | {d['n_cells']} | {d['max_abs_diff_vs_clean']} | {len(d['cells_not_identical'])} | {d['pass']} |")
    L.append("")
    L.append("**C3. Structural zero** (deliberate regression: a copy whose O/T/S nav token is a constant must read the nav cells as exactly 0.0; a copy whose T scene path is constant must read T x scene as exactly 0.0; every unprobed cell of the same copy must stay alive, > 0). MEASURED: `runs/s*_navc.npz`, `runs/s*_tsoff.npz`.")
    L.append("")
    L.append("| copy | probed cells -> max abs diff vs the copy's own clean | unprobed (alive) cells -> max abs diff | pass |")
    L.append("|---|---|---|---|")
    for k, d in C["structural_zero"].items():
        L.append(f"| {k} | {', '.join(f'{a} = {v}' for a, v in d['probed'].items())} | {', '.join(f'{a} = {v:.3e}' for a, v in d['alive'].items())} | {d['pass']} |")
    L.append("")
    b = C["banked_shuffle_consistency"]
    L.append("**C5. Consistency with the banked shuffles** (descriptive; plus the implementation check that O x T code at donor seed 1 IS `--tac shuffle`). MEASURED: `banked/`, `runs/s0_d1.npz`.")
    L.append("")
    L.append(f"* s0, donor seed 1: max per-window absolute difference between my O x code ADE6 and the banked `--tac shuffle` ADE6 = **{b['max_abs_diff_per_window_ade6_vs_banked_tacshuf']:.3e}** m; mine O x code ADE6 mean {b['mine_O_x_code_ade6_mean']:.4f} m, banked tac-shuffle {b['banked_tacshuf_ade6_mean']:.4f} m, banked own-code {b['banked_tacpred_ade6_mean']:.4f} m; banked delta {b['banked_delta_m']:+.4f} m, mine {b['mine_delta_m']:+.4f} m.")
    for s, d in C["banked_shuffle_consistency_descriptive"].items():
        L.append(f"* {s}, donor seed 0: O x T code delta = {d['mine_O_x_code_delta_m_donor0']:+.4f} m (the report's cited range for the banked shuffles: +4.4 to +6.4 m ADE6).")
    L.append("")
    L.append("**C4. Donor-seed noise** (s0, donor seeds 0 and 1; a cell counts as listening only if |delta| exceeds this): per cell, |delta(donor 0) - delta(donor 1)|.")
    L.append("")
    L.append("| row | " + " | ".join(COL_LABEL[c] for c in COLS) + " | ALL |")
    L.append("|---|" + "---|" * (len(COLS) + 1))
    fl = R["analysis"]["floor_donor_seed"]
    for row in ROWS:
        L.append(f"| {row} ({UNIT[row]}) | " + " | ".join(("n/a" if fl[row][c] is None else f"{fl[row][c]:.4f}") for c in COLS + ("all",)) + " |")
    L.append("")
    L.append(f"**All controls pass: {C['all_pass']}.**")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read", default=str(here / "READ.json"))
    ap.add_argument("--out", default=str(here / "LISTEN_RESULT_tables.md"))
    a = ap.parse_args(argv)
    R = json.load(open(a.read, encoding="utf-8"))
    A = R["analysis"]
    parts = {
        "OUTCOME": f"**{A['outcome']['headline']}**; holds at both seeds: {A['outcome']['holds_at_both_seeds']}; at exactly one seed: {A['outcome']['holds_at_exactly_one_seed']}; seeds disagree on: {A['outcome']['seeds_disagree_on']}",
        "QNAV": f"**{A['qnav']['headline']}**",
        "NEXT": A["next_committed"], "QNEXT": A["qnav_next_committed"],
        "HYP": hyp_table(R), "LARGEST": largest_table(R), "CLEAN": clean_levels(R),
        "GRID_S0": grid_table(R, "s0"), "GRID_S1": grid_table(R, "s1"), "QNAV_T": qnav_table(R), "CONTROLS": controls_md(R),
    }
    json.dump(parts, open(Path(a.out).with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    body = "\n\n".join(f"<!-- {k} -->\n{v}" for k, v in parts.items())
    Path(a.out).write_text(body, encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
