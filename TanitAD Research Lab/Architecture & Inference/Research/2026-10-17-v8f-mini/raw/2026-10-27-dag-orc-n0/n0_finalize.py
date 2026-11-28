#!/usr/bin/env python3
"""n0_finalize.py -- composes N0_READ.json (core read + tick-level beside rows + the dag2_read --repro-only cross-check + the PREREG section 8 routing) and N0_compact.txt.
Routing is coded from PREREG_V8F_DAG_ORC section 8 (the N0 table) as registered; nothing here moves a bar or a row."""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
N0 = "E:/Projects/TanitAD-artifacts/dag-orc-n0"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


core = json.load(open(f"{N0}/N0_READ_core.json", encoding="utf-8"))
tk = json.load(open(f"{N0}/N0_ticks_beside.json", encoding="utf-8"))
rc2 = json.load(open(f"{N0}/controls/RC_repro_dag2read.json", encoding="utf-8"))

# ---- PREREG section 8, the N0 table, coded cell by cell.  DAG-OACC's read: status read from disk (no OACC_READ file / no DAG-OACC checkpoint exists).
oacc_status = {"exists": False, "evidence": "E:/Projects/TanitAD-artifacts/v8f-dagoacc/ holds only M2b_s0_dagoacc/gate/ and wacc/ (offline w_acc probes); no checkpoint, no rollout, no read; none on the mirror's Project Steering tree either",
               "separated_better_at_both_seeds_on_C": None}


def route(outcome, oacc):
    if outcome == "REFUSED":
        return "REFUSED", "fix and re-read; no row is reported"
    if outcome == "NAVMOVE":
        return "ROUTE-EXTENT", ("a PREREG for the dead-reckoner fix, which advances the nav row through the logged route's manoeuvre sequence by the car's own arc. Zero training, dev-box CPU, control --nav deadreckon. "
                                "If it reproduces N0's gain the defect is route extent; if not, N0's gain was the timing leak and LAUNCH-INTENT follows. DAG-ORC is HELD (precondition 3).")
    if oacc["exists"] and oacc["separated_better_at_both_seeds_on_C"]:
        return "COMPOSE", "DAG-OACC's acceleration term on the base, read on Read A under its own PREREG"
    return "LAUNCH-INTENT", ("O's along-track loss over the executed horizon (knots 1-12 at m = 10), up-weighted on GT-launch TRAIN samples (v0 < 6 m/s and GT v(6 s) >= v0 + 1 m/s). One flag with its G-DVB entry, DAG init, "
                             "3,000 steps, under its own PREREG. DAG-ORC is NOT held (precondition 3: it launches as written once its other preconditions and the PI's box-2 go are met).")


nxt, nxt_text = route(core["OUTCOME"], oacc_status)
cell = {"N0": core["OUTCOME"], "DAG-OACC read": "not yet read (no such read exists)", "table_row": "not NAVMOVE | not, or not yet read",
        "conditional": "If DAG-OACC's read later shows DAG-OACC - DAG separated-better at both seeds on every row of C, the same N0 outcome maps to COMPOSE instead (second not-NAVMOVE row)."}

# ---- share of DAG's measured excess over P10 (recomputed here, K-RC green) that nav-logged moves, per crawl row -- context only
dp = core["beside"]["read_a_DAGdr_minus_P10_recomputed"]
nb = core["beside"]["read_a_N0_minus_DAGdr"]
shares = {}
for s in ("s0", "s1"):
    shares[s] = {}
    for lab in core["decisive"][s]["C_s"]:
        ex, mv = dp[s][lab]["delta"], nb[s][lab]["delta"]
        shares[s][lab] = {"DAG_minus_P10": ex, "N0_minus_DAGdr": mv, "share_of_excess_moved_pct": (100.0 * mv / ex) if ex else None}

out = dict(core)
out["OUTCOME_FULL"] = {"verdict": core["OUTCOME"], "NAVMOVE": core.get("NAVMOVE"), "definition": "NAVMOVE = every row of C_s separated-better (N0 - DAG deadreckon, same checkpoint, paired, B = 2000) at BOTH seeds; C_0 = {curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share}, C_1 = C_0 + {yaw-rate MAE}",
                       "per_seed": {s: {"C_s": core["decisive"][s]["C_s"], "separated_better_rows": [l for l, r in core["decisive"][s]["rows"].items() if r["separated_better"]],
                                        "not_separated_better_rows": [l for l, r in core["decisive"][s]["rows"].items() if not r["separated_better"]]} for s in ("s0", "s1")}}
out["routing"] = {"PREREG_section_8_cell": cell, "committed_next": nxt, "committed_next_text": nxt_text, "DAG_OACC_read": oacc_status,
                  "DAG_ORC_launch": "NOT HELD by N0 (precondition 3: held only if N0 reads NAVMOVE). Still gated on P-0 (RF0 frozen), the reader with K-RC + mutation proof, and the PI's go for box-2 time (PREREG section 10)."}
out["context_share_of_DAG_excess_over_P10_moved_by_nav_logged"] = {"note": "DAG - P10 recomputed here from DAG's banked dumps (K-RC green, 0 differences vs banked READ_A.json); N0 - DAG from this read. Context only; decides nothing.", "rows": shares}
out["beside"]["tick_level"] = tk
out["beside"]["repro_cross_check"] = {"tool": "dag2t1/dag2_read.py --repro-only (the same invocation dag2t1/repro/ used), run again today", "file": f"{N0}/controls/RC_repro_dag2read.json", "sha256": sha(f"{N0}/controls/RC_repro_dag2read.json"),
                                      "RC-DAG": rc2["RC-DAG"], "controls_failed": [k for k, v in rc2["controls_PASS"].items() if not v]}
out["provenance"]["core_json"] = {"file": f"{N0}/N0_READ_core.json", "sha256": sha(f"{N0}/N0_READ_core.json")}
out["provenance"]["ticks_json_sha256"] = sha(f"{N0}/N0_ticks_beside.json")
out["provenance"]["finalizer_sha256"] = sha(os.path.abspath(__file__))
out["evidence_class"] = "MEASURED (ours; artifact paths under dag-orc-n0/ and v8f-dagger-t1/). Tier T1-kin with nav = LOGGED as an OPEN channel (PREREG section 8 warning). Not closed loop."
json.dump(out, open(f"{N0}/N0_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------------------------------------------------------------------------------------------------------- compact text
L = []
L.append("N0 -- the nav probe (PREREG_V8F_DAG_ORC section 8 @ 85d0b125).  DAG, T1-kin, nav LOGGED vs nav DEADRECKON, same checkpoint, paired episode-cluster bootstrap B=2000, 141 ep / 2,399 windows.")
L.append("X = nav-logged, Y = nav-deadreckon, delta = X - Y.  BETTER/WORSE = separated AND on that side of the row's direction.  [C] = crawl-set row (decisive).  Evidence: MEASURED.  Tier: T1-kin (self-action OPEN loop), nav-logged = OPEN channel.")
L.append(f"OUTCOME: {core['OUTCOME']}   ->  committed next: {nxt}   (DAG-OACC read: not yet read)")
L.append("controls: " + ", ".join(f"{k}={'PASS' if v['PASS'] else 'FAIL'}" for k, v in core["controls"].items() if not k.startswith("K-PINS")))
L.append("")


def fmt_tab(title, tab, s, mark=None):
    rows = [f"--- {title}, seed {s[1]}", f"{'family':13s} {'row':30s} {'delta':>10s} {'lo':>10s} {'hi':>10s}  verdict"]
    for lab, r in tab[s].items():
        if r.get("status"):
            rows.append(f"{r.get('family', ''):13s} {lab:30s} {'NOT COMPUTED':>32s}")
            continue
        v = "BETTER" if r["separated_better"] else ("WORSE" if r["separated_worse"] else "ns")
        c = " [C]" if (mark and lab in mark) else ""
        rows.append(f"{r['family']:13s} {lab:30s} {r['delta']:+10.5f} {r['lo']:+10.5f} {r['hi']:+10.5f}  {v}{c}")
    return rows


for s in ("s0", "s1"):
    L += fmt_tab("N0 - DAG(deadreckon): nav-logged minus nav-deadreckon [DECISIVE rows marked [C]; the rest beside]", core["beside"]["read_a_N0_minus_DAGdr"], s, core["decisive"][s]["C_s"])
    L.append(f"    all rows of C_{s[1]} separated-better: {core['decisive'][s]['all_C_separated_better']}")
    L.append("")
for s in ("s0", "s1"):
    L += fmt_tab("BESIDE: N0 (nav-logged) - P10", core["beside"]["read_a_N0_minus_P10"], s, core["decisive"][s]["C_s"]) + [""]
    L += fmt_tab("BESIDE: DAG (nav-deadreckon) - P10, recomputed from the banked dumps (== banked READ_A.json, 0 differences)", core["beside"]["read_a_DAGdr_minus_P10_recomputed"], s, core["decisive"][s]["C_s"]) + [""]
L.append("--- BESIDE: share of DAG's excess over P10 that nav-logged moves (context only; negative = nav-logged REDUCES the excess; the S-POS rows of the DAG - P10 tables above are DAG_READ.json's banked values)")
for s in ("s0", "s1"):
    for lab, v in shares[s].items():
        L.append(f"  {s} {lab:30s} DAG-P10 {v['DAG_minus_P10']:+.5f}   N0-DAG {v['N0_minus_DAGdr']:+.5f}   = {v['share_of_excess_moved_pct']:+.1f} % of the excess")
L.append("")
L.append("--- BESIDE: STRATEGIC S-POS vs CTRV (within run, unpaired; delta = arm - CTRV position error, negative = closer to the logged path), from the tick dumps")
for s in ("s0", "s1"):
    sp = tk["seeds"][s]["S-POS_vs_CTRV_within_run"]
    for k in sp["N0_nav_logged"]:
        a, b = sp["N0_nav_logged"][k], sp["DAG_nav_deadreckon_rederived"][k]
        L.append(f"  {s} {k}: N0 {a['delta']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]   DAG-dr {b['delta']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]")
    L.append(f"  {s} (DAG-dr re-derived from its banked ticks vs DAG_READ.json: max abs diff {sp['rederived_vs_banked_max_abs_diff']})")
L.append("--- BESIDE: TACTICAL T-head macro-F1, ticks >= 1, matched cells, N0 - DAG-dr (higher better; the paired tool reads tick dumps)")
for s in ("s0", "s1"):
    for hd in ("lat", "lon"):
        r = tk["seeds"][s]["T_head_macroF1_N0_minus_DAGdr"][f"{hd}_matched"]
        L.append(f"  {s} T-{hd}: {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] separated={r['separated']}  (F1 N0 {r['f1_a']:.4f} vs DAG-dr {r['f1_b']:.4f}, n_cells {r['n_cells_matched_pairs']})")
L.append("--- BESIDE: reachability of the lever (informational control K-REACH): windows whose executed cl track differs from the banked deadreckon dump")
for s in (0, 1):
    r = core["beside"]["reach"][str(s)] if str(s) in core["beside"]["reach"] else core["beside"]["reach"][s]
    L.append(f"  s{s}: {r['windows_with_changed_cl_track']} of {r['of']} windows, max |delta cl| {r['max_abs_cl_diff_m']:.2f} m")
open(f"{N0}/N0_compact.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
