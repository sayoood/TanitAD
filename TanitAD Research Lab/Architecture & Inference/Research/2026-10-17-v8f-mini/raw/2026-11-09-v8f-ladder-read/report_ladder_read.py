"""Builds RESULT_LADDER_READ.md / .json from results/arm_*.json: the card's outcome labels per arm (both seeds required) for the arms present, the per-level reach tables, the R35b floor comparison
(R35b pending until its result exists) and the beside rows. Labels only: no verdict (the Master Mind writes it). Tier T0 on every number. Usage: python report_ladder_read.py [--smoke]"""
import glob, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lr_lib as L

ORDER = ["C0", "S35", "R35", "R35H", "R35b"]
LV = ("1", "2", "3")


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def iv(r, d=3):
    return "n/a" if not r or r.get("point") is None else f"{r['point']:.{d}f} [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"


def hkey(h):
    return f"{h:g}"


def load(smoke):
    pat = "smoke_arm_*.json" if smoke else "arm_*.json"
    runs = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", pat))):
        r = json.load(open(p, encoding="utf-8"))
        runs[r["name"]] = r
    return runs


def main(smoke=False):
    runs = load(smoke)
    arms = sorted({r["arm"] for r in runs.values()}, key=lambda a: ORDER.index(a) if a in ORDER else 99)
    cp = os.path.join(HERE, "CONTROL_M2B.json")
    ctrl = json.load(open(cp)) if os.path.exists(cp) else None
    out = {"tier": "T0", "card": "theory/PREREG_V8F_LADDER.md (Read, Committed outcomes)", "smoke": smoke, "arms": {}, "controls": {}, "r35b": "present" if "R35b" in arms else "PENDING"}
    out["controls"]["m2b_reproduction"] = {"pass": ctrl["pass"] if ctrl else None, "tolerances": ctrl["tolerances"] if ctrl else None}
    out["controls"]["copy_vs_itself_max_abs"] = {n: r["copy_vs_itself_max_abs"] for n, r in runs.items()}
    out["controls"]["built_vs_argv"] = {n: r["argv_checked"] for n, r in runs.items()}
    lines = ["# RESULT_LADDER_READ" + (" (SMOKE: 19 reference draws, not the deciding run)" if smoke else ""), "",
             "**Tier T0**: a world-model diagnostic (latent-space prediction of the encoded future of 141 eval clips). It is not driving performance, and no number below is.",
             "Card: `theory/PREREG_V8F_LADDER.md` (Read, Committed outcomes). Implementation choices: `IMPL_CHOICES_LADDER_READ.md`. This file carries labels and tables only; the verdict is the Master Mind's.", ""]
    lines += ["## Status and controls", "",
              f"* Arms read: {', '.join(sorted(runs)) or 'none'}. **R35b (the run-to-run floor): {'read' if 'R35b' in arms else 'PENDING (still training on box 2; add `R35b_s0` to the arm list and rerun this report)'}.**",
              f"* Control 1, M2b reproduction through this code path (six MRR values, chance, RankMe on IMAG1B's windows): {'PASS' if ctrl and ctrl['pass'] else 'FAIL' if ctrl else 'not run'} (`CONTROL_M2B.json`)."]
    worst = max([r["copy_vs_itself_max_abs"] for r in runs.values()] or [0.0])
    lines += [f"* Control 2, Copy-Last against itself, every level and reach of every arm: max |point, lo, hi| = {worst} (must be exactly 0.0).",
              "* Control 3, BUILT model equals argv (strides three ways, trained k, free-run anchors, context, history K, dilation, causal pad): " + ("all arms passed (a mismatch raises and the arm is not read)." if runs else "no arm yet."),
              "* Control 4, deliberate regressions: `test_ladder_read.py` (`TEST_LADDER_READ.log`).", ""]
    lines += ["## Families (T0)", "",
              "LONGITUDINAL and LATERAL appear only through the IA-2 targets (speed, lead gap, yaw rate) and the IA-3 gains below. TACTICAL: n = 0, no tactical target is read in this deliverable. **STRATEGIC: PAUSED** (PI item 25).", ""]
    lines += ["## Outcome per arm (the card's labels; both seeds required)", "", "| arm | label | levels beating Copy-Last at the band end | levels not beating | HEALTHY L1 / L2 / L3 (all seeds) | note |", "|---|---|---|---|---|---|"]
    for a in arms:
        ps = {r["seed"]: r["passes"] for r in runs.values() if r["arm"] == a}
        oc = L.band_outcome(ps)
        out["arms"][a] = {"seeds": sorted(ps), "outcome": oc, "per_seed": {str(s): {"outcome": L.band_outcome({s: p}), "levels": p} for s, p in ps.items()}}
        seeds = "both seeds" if len(ps) == 2 else f"seed {sorted(ps)} only"
        lines.append(f"| {a} ({seeds}) | **{oc['label']}** | {', '.join('L' + x for x in oc['levels_beating']) or 'none'} | {', '.join('L' + x for x in oc['levels_not_beating']) or 'none'} | "
                     f"{' / '.join('yes' if oc['healthy'][x] else 'NO' for x in LV)} | {oc['note']} |")
    lines += ["", "Per seed:", "", "| run | label | L1 2.0 s skill [95 % CI] | L2 6.0 s | L3 12.0 s | healthy L1/L2/L3 |", "|---|---|---|---|---|---|"]
    for n, r in runs.items():
        oc = L.band_outcome({r["seed"]: r["passes"]})
        c = []
        for lv in LV:
            p = r["passes"][lv]
            tag = "BEATS" if p["beats"] else "WORSE" if p["worse"] else "no" if p["beats"] is not None else "unreadable"
            c.append(f"{f(p['skill'])} [{f(p['lo'])}, {f(p['hi'])}] {tag}")
        lines.append(f"| {n} | {oc['label']} | {c[0]} | {c[1]} | {c[2]} | {' / '.join('yes' if r['passes'][x]['healthy'] else 'NO' for x in LV)} |")
    lines += ["", "Skill = 1 - sum|pred - oracle|^2 / sum|copy - oracle|^2 in the level's latent (0 = Copy-Last). Interval: episode-cluster bootstrap (B 2,000, seed 0). BEATS = lower bound above 0 AND point above the 97.5th percentile of the same statistic over 199 draws with the answer deranged inside the clip.", ""]
    lines += ["## Reach tables (IA-0 skill against Copy-Last; deciding rows marked *)", "",
              "| run | level | requested s | actual s (k) | on grid | n anchors | clips excluded (<2 anchors) / masked at clip end | skill [95 % CI] | ref 97.5th | above | below |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        for lv in LV:
            for h, x in r["ia0"][lv]["rows"].items():
                star = "*" if x.get("band_end") else ""
                if not x.get("readable"):
                    lines.append(f"| {n} | L{lv} | {h}{star} | {f(x.get('actual_s'), 1)} ({x.get('k')}) | {x.get('on_grid')} | {x.get('n', 0)} | {x.get('n_excluded_lt2_per_clip')} / {x.get('n_masked_end_of_clip')} | not readable (fewer than 2 anchors per clip) | | | |")
                    continue
                lines.append(f"| {n} | L{lv} | {h}{star} | {f(x['actual_s'], 1)} ({x['k']}) | {x['on_grid']} | {x['n']} | {x['n_excluded_lt2_per_clip']} / {x['n_masked_end_of_clip']} | {iv(x)} | {f(x['thr_hi'])} | {x['above']} | {x['below']} |")
    lines += ["", "n anchors = (clip, state) pairs scored; 'masked at clip end' = anchors whose target state lies past the end of the clip, counted and not scored (summed over clips).", ""]
    lines += ["## IA-0 health (k = 1: MRR of the encoded future among the clip's anchors, predicted latent as query)", "", "| run | level | MRR | chance | excess [95 % CI] | ref 97.5th | separated (two-part) | RankMe pred / oracle |", "|---|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        for lv in LV:
            h = r["ia0"][lv]["health"]
            lines.append(f"| {n} | L{lv} | {f(h['mrr'])} | {f(h['chance'])} | {iv(h)} | {f(h['thr97'])} | {h['separated_two_part']} | {f(h['rankme']['pred_k1'], 1)} / {f(h['rankme']['oracle_k1'], 1)} |")
    lines.append("")
    lines += ["## Lever effects against the R35b run-to-run floor (band-end skill)", ""]
    fl = {}
    if "R35b_s0" in runs and "R35_s0" in runs:
        for lv in LV:
            a, b = runs["R35_s0"]["passes"][lv]["skill"], runs["R35b_s0"]["passes"][lv]["skill"]
            fl[lv] = abs(a - b) if a is not None and b is not None else None
        lines += ["Floor per level = |skill(R35 s0) - skill(R35b s0)| (same flags, same seed): " + ", ".join(f"L{lv} {f(fl[lv], 4)}" for lv in LV) + ". A difference counts as a lever effect only if it exceeds the floor of its level.", ""]
    else:
        lines += ["**R35b is pending, so no difference between arms is called a lever effect yet.** The differences are printed for reference only.", ""]
    out["floor"] = fl or "R35b pending"
    lines += ["| comparison (same seed) | level | skill A | skill B | A - B | exceeds floor |", "|---|---|---|---|---|---|"]
    for (A, B) in (("S35", "C0"), ("R35", "S35"), ("R35H", "R35"), ("R35", "C0")):
        for s in (0, 1):
            a, b = runs.get(f"{A}_s{s}"), runs.get(f"{B}_s{s}")
            if not a or not b:
                continue
            for lv in LV:
                x, y = a["passes"][lv]["skill"], b["passes"][lv]["skill"]
                if x is None or y is None:
                    continue
                ex = "R35b pending" if not fl else ("yes" if fl.get(lv) is not None and abs(x - y) > fl[lv] else "no")
                lines.append(f"| {A} - {B} s{s} | L{lv} | {f(x)} | {f(y)} | {f(x - y)} | {ex} |")
    lines.append("")
    lines += ["## Beside: planning-form context (context 1) band-end skill (decides nothing)", "", "| run | L1 2.0 s | L2 6.0 s | L3 12.0 s |", "|---|---|---|---|"]
    for n, r in runs.items():
        c = []
        for lv in LV:
            x = [v for v in r["ia0_planning_context1"][lv]["rows"].values() if v.get("band_end")]
            ok = bool(x) and x[0].get("readable")
            c.append(iv(x[0]) + (" above" if x[0].get("above") else " below" if x[0].get("below") else "") if ok else "n/a")
        lines.append(f"| {n} | {c[0]} | {c[1]} | {c[2]} |")
    lines.append("")
    lines += ["## Beside (no bar; ridge / logistic probes, bootstrap intervals only, no permutation reference)", "",
              "### IA-1 capture (score = 1 - NMSE, 0 = the marginal mean) on the three IA-2 targets", "", "| run | level | speed | lead gap | yaw rate |", "|---|---|---|---|---|"]
    for n, r in runs.items():
        for lv in LV:
            c = [r["beside"]["ia1"].get(f"{lv}|{q}", {}) for q in L.IA2_TARGETS]
            lines.append(f"| {n} | L{lv} | " + " | ".join(f"{iv(x, 2)} n={x.get('n')}" if x.get("point") is not None else "n/a" for x in c) + " |")
    lines += ["", "### IA-2 probe transfer at the band end: predicted minus Copy-Last score (positive = predicted decodes better)", "", "| run | level | target | pred - copy [95 % CI] | oracle score | n |", "|---|---|---|---|---|---|"]
    for n, r in runs.items():
        for lv in LV:
            h = hkey(L.BAND_END[int(lv)])
            for q in L.IA2_TARGETS:
                x = r["beside"]["ia2"].get(f"{lv}|{h}|{q}", {})
                if x.get("pred_minus_copy"):
                    lines.append(f"| {n} | L{lv} ({h} s) | {q} | {iv(x['pred_minus_copy'])} | {f((x.get('ora') or {}).get('point'))} | {x.get('n', 0)} |")
                else:
                    lines.append(f"| {n} | L{lv} ({h} s) | {q} | not readable | | {x.get('n', 0)} |")
    lines += ["", "### IA-3 action audit at the band end (gain = decoded effect / physical effect, 1.0 = right size; cosine = imagined vs physical 2 x 2 Jacobian)", "", "| run | level | k (s) | gain lon [CI] | gain lat [CI] | cosine [CI] | n |", "|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        for lv in LV:
            pl = [p for p in r["horizon_plan"][lv] if abs(p[0] - L.BAND_END[int(lv)]) < 1e-9][0]
            x = r["beside"]["ia3"].get(f"{lv}|{pl[1]}")
            if x:
                lines.append(f"| {n} | L{lv} | {pl[1]} ({f(x['h_s'], 1)} s) | {iv(x['gain_lon'])} | {iv(x['gain_lat'])} | {iv(x['cos'])} | {x['n']} |")
    lines += ["", "### TSI (mean of 1 - score over the fast targets minus over the slow ones, from IA-1) and cross-level slope", "", "| run | TSI L1 | TSI L2 | TSI L3 | slope L1 vs L2 | slope L2 vs L3 |", "|---|---|---|---|---|---|"]
    for n, r in runs.items():
        t = r["beside"]["tsi"]
        s = r["beside"]["slope"]
        lines.append(f"| {n} | {f(t['1']['tsi'])} | {f(t['2']['tsi'])} | {f(t['3']['tsi'])} | {iv(s.get('L1vsL2'))} ({(s.get('L1vsL2') or {}).get('horizon_s', '')} s) | {iv(s.get('L2vsL3'))} ({(s.get('L2vsL3') or {}).get('horizon_s', '')} s) |")
    lines += ["", "TSI uses ten IA-1 targets; labels exist at the 9,596 bank frames only (lead-gap and lead-closing rows have fewer, n per row in the json). Slope: decoded speed change of the coarser level on the finer level over the same real horizon.", "",
              "## Provenance", "", "| run | argv sha256 (first 12) | adapter sha256 (first 12) | ckpt step | free GB at start | wall s |", "|---|---|---|---|---|---|"]
    for n, r in runs.items():
        lines.append(f"| {n} | {r['argv_sha256'][:12]} | {r['adapter_sha256'][:12]} | {r['ckpt_step']} | {r['free_gb_at_start']} | {r['wall_s']} |")
    lines.append("")
    name = "RESULT_LADDER_READ_SMOKE" if smoke else "RESULT_LADDER_READ"
    open(os.path.join(HERE, name + ".md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump({**out, "runs": runs}, open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8"))
    print("wrote", name, "arms", arms)


if __name__ == "__main__":
    main("--smoke" in sys.argv)
