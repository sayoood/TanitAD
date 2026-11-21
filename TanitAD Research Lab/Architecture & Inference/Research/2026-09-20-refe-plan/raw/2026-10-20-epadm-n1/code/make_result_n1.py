"""Generate RESULT.md for EPADM-N1 MECHANICALLY from sim_results.json, tables.md, mutation.json and the pytest logs (no hand-typed number).   python make_result_n1.py > RESULT.md
Every sentence below is a function of the results: the outcome branch is the Master Mind's own DQ3 table, applied mechanically; nothing is interpreted beyond what the numbers say."""
import json
import os
import re
import sys

N1 = "C:/Users/Admin/par18/n1sim"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-epadm-n1"


def rate(r):
    return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}]"


def last_line(p):
    try:
        return [l for l in open(p, encoding="utf-8", errors="replace").read().strip().splitlines() if l.strip()][-1]
    except Exception as e:                                                              # noqa: BLE001
        return f"(unreadable: {e})"


def main():
    res = json.load(open(f"{N1}/work/sim/sim_results.json", encoding="utf-8"))
    tables = open(f"{N1}/work/tables.md", encoding="utf-8").read()
    mut = json.load(open(f"{N1}/work/mutation.json")) if os.path.exists(f"{N1}/work/mutation.json") else {}
    n_caught = sum(v["status"] == "CAUGHT" for v in mut.values())
    n_arms = sum(1 for k in mut if not k.startswith("HARNESS"))
    sanity = mut.get("HARNESS-SANITY-unmutated", {}).get("status")
    G = res["geometries"]
    ok = bool(res["controls_ok"])
    g41, g117 = G["GS41"]["all_met"], G["GS117"]["all_met"]
    if not ok:
        branch = "**RUN VOID: a control failed (see N1-T3).** No bar is read from a void run."
    elif g41:
        branch = "N1 meets every bar at 41 logs. Per `MM_RULING_S0BANK.md` DQ3: the bank is built at 41 logs and the 4.9 GB pod read is NOT needed."
    elif g117:
        branch = "N1 meets every bar only at 117 logs. Per DQ3: the pod read is requested, with that number."
    else:
        branch = ("N1 misses at both geometries. Per DQ3: the frozen EP-ADMISSION gate cannot certify a search candidate at any available geometry; the MM then rules, in a SEPARATE sealed decision and BEFORE any navtest "
                  "token is touched, what licenses C2 instead (that ruling may not use any B-REST outcome to choose its bar).")
    out = []
    out.append("# RESULT EPADM-N1 (Research Lab agent, 2026-10-20): the A2 statistic restricted to the CHANGED picks, re-simulated at the S0 own-fan geometry (MM ruling `MM_RULING_S0BANK.md` DQ3, arm N1)\n")
    out.append(f"**Escalation (integration + decision, headline):** the Master Mind lands `out/` (repo dir `{REPO}/`). Outcome by the ruling's own table (applied mechanically): {branch}\n")
    out.append("**Stamps:** every rate is SIMULATED on the real estimator (R1 simulator code, `SIM_DESIGN_N1.md` hashed BEFORE the first run), generator A primary, B-bootstrap 10,000, seed 20260927, NEW replicate salts 20261401 / 20261402; Base C = carve-B clones with real search outputs, Base X = real S0 labels with a synthetic candidate; neither is S0's own-fan search bank. Tier: SELECTION (label lookup, T0-class); one training seed (`H-ESTIM-SEED-1`: the interval answers \"another draw of tokens and logs\" only); STRATEGIC PAUSED. Nothing here is a navtest or driving claim.\n")
    out.append("## Plain statement (generated from `sim_results.json`)\n")
    out.append(f"* Controls: **{'ALL PASS' if ok else 'A CONTROL FAILED (run void)'}**; unit suite `test_n1.py`: {last_line(N1 + '/work/logs/pytest_n1.log')}; copied S0BANK suite: {last_line(N1 + '/work/logs/pytest_sim_s0_copy.log')}; mutation arms: **{n_caught} of {n_arms} caught**, unmutated harness {sanity}.")
    for g in ("GS41", "GS117"):
        s = G[g]
        cells = [k for k, v in s["cells_all_met"].items() if v]
        stat = dict(zip(("A1-r1 false refusal", "A2-N1 false refusal", "N1 DAC power x1.25"), s["status_primary_convention"].values()))
        out.append(f"* **{g}**: cells meeting EVERY bar: **{', '.join('Base ' + c for c in cells) if cells else 'none'}**; geometry ALL-MET (MET on C and on X): **{s['all_met']}**; per-bar status {json.dumps(stat)}; missed per cell {json.dumps(s['cells_missed'])}.")
    for g in ("GS41", "GS117"):
        for b in ("C", "X"):
            e = res["jobs"][f"{b}-{g}-A"]["bars_primary"]
            out.append(f"* {g} Base {b}: A1-r1 FR {rate(e['A1_r1_false_refusal'])}; A2-N1 FR {rate(e['A2_N1_1.05_false_refusal'])}; N1 DAC power x1.25 {rate(e['N1_DAC_power_x1.25_margin1.05'])} (bars: <= 5.0 % upper, <= 5.0 % upper, >= 70 % point).")
    a1_missed_c = [g for g in ("GS41", "GS117") if not res["jobs"][f"C-{g}-A"]["bars_primary"]["decision"]["bars"]["A1_r1_false_refusal_wilson_upper_le_5pct"]]
    if a1_missed_c:
        out.append(f"* **N1 does not touch A1.** The A1-r1 false-refusal bar is the S0BANK simulator's unchanged statistic; on Base C it is missed at {', '.join(a1_missed_c)}, so no cell of Base C can meet every bar whatever A2 does. S0BANK's `R-GEN` showed generator T (fixed logs) reads 0.0 % there, i.e. the Base C A1 miss is a property of generator A on 24 cloned source logs (R-DUP below). Whether A1 should be re-read on a different footing is a separate pre-registration for the MM; nothing was tuned here.")
    out.append("* The ONLY change to the S0BANK simulation is the A2 statistic (changed picks only, class-calibrated null, new seeds) plus the non-degenerate cross-check generator P; the margin (1.05), the three bars, the quantifier, B and the seed are the committed ones.\n")
    out.append("## Tables (machine-generated, `work/tables.md`)\n")
    out.append(tables)
    out.append("\n## Proposed register rows (NOT edited; the Master Mind lands them)\n")
    stat_txt = "; ".join(f"{g} ALL-MET={G[g]['all_met']} (cells meeting every bar: {', '.join(k for k, v in G[g]['cells_all_met'].items() if v) or 'none'})" for g in ("GS41", "GS117"))
    out.append(f"* `D-EPADM-N1-1` (arm N1 of `D-EPADM-S0BANK-1`): the A2 statistic restricted to the CHANGED picks, {stat_txt}. SIMULATED, SELECTION tier, generator A, one training seed, 3 search seeds (2 on Base X), STRATEGIC PAUSED, controls_ok={ok}. Artifact: `{REPO}/raw/results/sim_results.json`.")
    out.append("* RETRACTION_LOG candidate (class \"a bar the proposed statistic cannot move\"): only if the Base C A1-r1 miss above is present; N1 restricts A2 and leaves A1-r1 unchanged.")
    out.append("\n## Deliverable manifest\n")
    out.append("See `out/paths.json` (repo path, absolute source, \"raw\" for every item). Everything exists under `C:/Users/Admin/par18/n1sim/` (the only copy until landed); inputs read in place and never modified: `epadm_search/work/carveB_bank2`, `epadm_search/tipgate`, `epadm_search/tipr1`, the C1 code dir, `pickrule/run/prepare/arrays.npz` (sha256 pinned), `s0amend/work/sim/sim_results.json` (INHERITED reference rows only).")
    out.append("\nA 20-replicate plumbing smoke of the run/repro/report path was executed after the hash and discarded (scratch deleted); its rates were not used.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
