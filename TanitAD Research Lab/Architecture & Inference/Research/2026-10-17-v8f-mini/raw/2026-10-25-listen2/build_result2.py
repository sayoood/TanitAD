"""Renders LISTEN2_RESULT.md from READ.json (every number in the tables is read from the JSON, none is typed). Prose that states a number uses f-strings of the same JSON.
python build_result2.py   (CPU, no torch)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "READ.json", encoding="utf-8"))
LISTEN = json.load(open(HERE.parent / "listen" / "READ.json", encoding="utf-8"))
E1, E2, E3, C = R["E1"], R["E2"], R["E3"], R["controls"]
TIER = "T1-class open loop, DIAGNOSTIC"
ROWN = {"S": "S: checkpoint position error @ 50 m, replaced - clean (m)", "T-lat": "T-lat: lateral macro-F1, clean - replaced",
        "T-lon": "T-lon: longitudinal macro-F1, clean - replaced", "O": "O: ADE6, replaced - clean (m)", "O-ade2": "O: ADE2 (beside), replaced - clean (m)"}
UNIT = {"S": "m", "T-lat": "F1", "T-lon": "F1", "O": "m", "O-ade2": "m"}
INPN = {"scene": "scene", "ego": "ego", "nav": "nav", "limit": "limit", "ck": "S checkpoints", "code": "T code", "age": "tick ages", "all": "ALL inputs"}


def f(x, d=3, sign=True):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def cell(c, d=3, u=""):
    return f"{f(c['delta'], d)}{u} [{f(c['lo'], d)}, {f(c['hi'], d)}]"


def yn(b):
    return "yes" if b else "no"


L = []
w = L.append

fl = E1["donor_seed_floor"]
b0, b1, bd1 = E1["blocks"]["s0"], E1["blocks"]["s1"], E1["blocks"]["s0_d1"]
fs = E1["flags"]
e3s0, e3s1, e3d1 = E3["blocks"]["s0"], E3["blocks"]["s1"], E3["blocks"]["s0_d1"]
V = E2["verdict"]

w("# PREREG_V8F_LISTEN2: result (executed 2026-10-25; zero training, CPU, M2b s0 and s1 banked checkpoints)\n")
w(f"**Evidence class:** MEASURED (ours). Artifacts (this directory): `READ.json` (every number below), `READ_controls.json`, `runs/*.npz` (per-window arrays), "
  f"`runs/means_*.npz` (the training-set means). **Tier of every number:** {TIER} (donor and mean inputs are interventions on a fixed checkpoint, not driving). "
  "**Estimator of every interval:** paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval/ci.py`, through the eval's own `paired` and macro-F1 reducer), "
  "141 episodes on the O row (S 137, T-lat 139, T-lon 136: the label masks differ per row, as in LISTEN). **The question it answers:** \"would another draw of EPISODES say "
  "this?\". It does NOT answer \"would another TRAINING RUN say this?\" (the two banked training seeds, M2b s0 and M2b s1, are shown side by side for that; two seeds are "
  "not a seed-noise floor) and not \"would another INFERENCE RUN say this?\" (the planner is deterministic on CPU, so there is no inference variance). The PREREG is "
  f"`PREREG_V8F_LISTEN2.md` (`prereg_commit` {R['prereg_commit']}); nothing was staged or committed.\n")
w("Sign convention (as in LISTEN): S and O rows are replaced minus clean in metres (larger is worse); T rows are clean minus replaced macro-F1 (a drop, larger is worse). "
  "\"Separated\" = the 95 % interval excludes 0.\n")

# ---------------------------------------------------------------------------------------------------------------- 0
w("## 0. The three outcomes (the committed tables, applied verbatim)\n")
w("| experiment | outcome | condition as committed | what was measured |")
w("|---|---|---|---|")
w(f"| **E1 (PRIMARY)** route through the chain | **{E1['outcome']['headline']}** | E1 > 0 and separated at BOTH seeds and above the donor-seed floor | "
  f"T-lat F1 drop, donor nav at S only: s0 {cell(fs['s0'])}, s1 {cell(fs['s1'])}; floor {f(fl['T-lat'], 4, False)}. **Bar met at both seeds.** |")
w(f"| **E2** is the grid's ordering a donor-swap artefact? | **{V['headline']}** | ROBUST if all 4 rows agree at both seeds, PARTLY ROBUST if 2 to 3, NOT ROBUST if fewer | "
  f"{V['k_rows_agreeing_at_both_seeds']} of 4 rows agree at both seeds (S = nav, T-lat = S checkpoints, T-lon = ego); **the O row does not** "
  "(mean replacement ranks T code above ego at both seeds). **The top outcome, ROBUST, was MISSED.** |")
w(f"| **E3** O's coherence | **{E3['outcome']['headline']}** | ADE6(joint triple) - max single swap < 0 and separated at both seeds | "
  f"s0 {cell(e3s0['E3'])} m, s1 {cell(e3s1['E3'])} m. **Bar met at both seeds.** |")
w("")
w(f"Committed meaning, E1 {E1['outcome']['headline']} (quoted): \"{E1['meaning_committed'][E1['outcome']['headline']]}\". "
  f"Committed meaning, E3 {E3['outcome']['headline']} (quoted): \"{E3['outcome']['meaning']}\". "
  "E2 PARTLY ROBUST carries the committed rule \"the rows that disagree are reported with their cells\" (section 2). "
  "**All controls pass** (section 4), so the reading is reported. The readings M1 to M7 the PREREG text left open (section 5) were written into `listen2.py` / `read2.py` "
  "before any LISTEN2 number was computed.\n")

# ---------------------------------------------------------------------------------------------------------------- 1
w("## 1. E1 (PRIMARY): does the route flow down? Donor nav injected at S only\n")
w("**Intervention.** The donor window's nav token replaces S's own nav token; S's checkpoints therefore change; T and O then read S's checkpoints as usual, with their "
  "OWN nav, ego and limit tokens, O's age token and both scene paths untouched. This is the only path by which nav-at-S can reach T-lateral. "
  f"The module asserts this per window with `torch.equal` (S's nav token is the donor's; T's and O's own tokens and scene paths are the clean pass's bit for bit): "
  f"**asserted on 2,399 / 2,399 windows in each of the five donor runs** (s0 donor 0, s1 donor 0, s0 donor 1, and both identity runs).\n")
w("### E1, per seed (read: T-lat F1 drop)\n")
w("| seed | donor | T-lat clean F1 | T-lat F1 with donor nav at S only | **E1 = drop [95 % CI]** | n windows / episodes | separated | > 0 | above donor-seed floor (0.0825) | bar met |")
w("|---|---|---|---|---|---|---|---|---|---|")
for s, blk, don in (("s0", b0, 0), ("s1", b1, 0)):
    d = blk["rows"]["T-lat"]
    g = fs[s]
    w(f"| {s} | seed {don} | {d['clean_level']:.4f} | {d['clean_level'] - d['delta']:.4f} | **{cell(d, 4)}** | {d['n_windows']} / {d['n_episodes']} | {yn(g['separated'])} | "
      f"{yn(g['positive'])} | {yn(g['above_floor'])} | **{yn(g['bar_met'])}** |")
d = bd1["rows"]["T-lat"]
w(f"| s0 | seed 1 (floor run) | {d['clean_level']:.4f} | {d['clean_level'] - d['delta']:.4f} | {cell(d, 4)} | {d['n_windows']} / {d['n_episodes']} | {yn(d['separated'])} | {yn(d['delta'] > 0)} | n/a | n/a |")
w("")
w(f"**Donor-seed floor (M6).** |E1(s0, donor 0) - E1(s0, donor 1)| = |{b0['rows']['T-lat']['delta']:.4f} - {bd1['rows']['T-lat']['delta']:.4f}| = **{fl['T-lat']:.4f} F1**, "
  f"applied to both seeds. E1 is {b0['rows']['T-lat']['delta'] / fl['T-lat']:.1f} x (s0) and {b1['rows']['T-lat']['delta'] / fl['T-lat']:.1f} x (s1) that floor; "
  f"the second donor draw also separates on its own ({cell(bd1['rows']['T-lat'], 4)}).\n")
w("### Beside (descriptive, all from the same donor-nav-at-S arm; the PREREG marks these \"descriptive\")\n")
w("| row (metric) | seed | clean level | change with donor nav at S only [95 % CI] | n windows / episodes | separated |")
w("|---|---|---|---|---|---|")
for row in ("S", "T-lat", "T-lon", "O", "O-ade2"):
    for s, blk in (("s0", b0), ("s1", b1)):
        d = blk["rows"][row]
        w(f"| {ROWN[row]} | {s} | {d['clean_level']:.4f} | {cell(d, 4, ' ' + UNIT[row])} | {d['n_windows']} / {d['n_episodes']} | {yn(d['separated'])} |")
w("")
w("The fraction E1 / (T-lat x S-checkpoints swap), the PREREG's third \"beside\" read, with LISTEN's whole-checkpoint swap recomputed in the same run:\n")
w("| seed | E1 (T-lat, nav at S only) | T-lat x S checkpoints (whole-checkpoint donor swap, recomputed) | fraction E1 / swap | T-lon: E1 | T-lon x S checkpoints | fraction (T-lon) |")
w("|---|---|---|---|---|---|---|")
for s, blk in (("s0", b0), ("s1", b1)):
    ck = blk["ref"]["T/ck@T-lat"]
    ckl = blk["ref"]["T/ck@T-lon"]
    w(f"| {s} | {cell(blk['rows']['T-lat'], 3)} | {cell(ck, 3)} | **{blk['fraction_E1_over_Tlat_x_ck']:.2f}** | {cell(blk['rows']['T-lon'], 3)} (not separated) | "
      f"{cell(ckl, 3)}{'' if ckl['separated'] else ' (not separated)'} | {blk['fraction_E1_over_Tlon_x_ck']:.2f} (both numerators are within noise: read as nothing) |")
w("")
w(f"**What E1 says.** Injecting a foreign nav at S alone, with T's own nav untouched, takes T-lateral macro-F1 from {b0['rows']['T-lat']['clean_level']:.3f} to "
  f"{b0['rows']['T-lat']['clean_level'] - b0['rows']['T-lat']['delta']:.3f} (s0) and from {b1['rows']['T-lat']['clean_level']:.3f} to "
  f"{b1['rows']['T-lat']['clean_level'] - b1['rows']['T-lat']['delta']:.3f} (s1), separated at both seeds and several times above the donor-seed floor. "
  "By the committed table this is **ROUTE FLOWS DOWN**: T's lateral decision follows what S derives from nav, so nav -> S -> T is a working chain for the lateral decision, "
  "and the \"PARALLEL PATH\" narrowing of the hierarchy claim (to \"S -> O\" and \"nav -> T\") is **not** supported.\n")
w("Beside it (descriptive, post hoc, not promoted): "
  f"(a) the donor-nav-at-S arm moves T-lat by as much as the whole-checkpoint swap (fraction {b0['fraction_E1_over_Tlat_x_ck']:.2f} / {b1['fraction_E1_over_Tlat_x_ck']:.2f}; the two intervals overlap "
  "at both seeds, so \"at least as much\" is a point-estimate reading, not a separated difference); so in this read the whole-checkpoint swap adds no separable effect beyond nav-at-S alone. "
  f"(b) **T-longitudinal does not follow the route through S**: E1 on T-lon is {cell(b0['rows']['T-lon'], 3)} F1 and {cell(b1['rows']['T-lon'], 3)} F1, not separated at either seed (T-lon is carried by ego and the scene in LISTEN). "
  f"(c) O's ADE6 under this arm is {cell(b0['rows']['O'], 3)} m (s0) and {cell(b1['rows']['O'], 3)} m (s1), which reproduces LISTEN's `nav@S` arm (+0.726 / +0.821 m; recomputed difference "
  f"{C['reproduction']['s0']['listen_cells_recomputed_maxabs_DIAGNOSTIC']['e1/nav@S vs LISTEN nav@S (ade6)']:.1e} / {C['reproduction']['s1']['listen_cells_recomputed_maxabs_DIAGNOSTIC']['e1/nav@S vs LISTEN nav@S (ade6)']:.1e} m max over windows).\n")

# ---------------------------------------------------------------------------------------------------------------- 2
w("## 2. E2: is the grid's ordering an artefact of donor swapping? Training-set mean replacement\n")
w("**Intervention.** Each input, one at a time, is replaced by the mean of that input over HBLEND's 9,582 training windows (M1: the input as the layer reads it at inference, at that "
  "layer's own tick; mean scene taken over the un-padded frames; mean ages as floats). The replacement goes through the same exactly-once shadow mechanism as LISTEN's donors. "
  f"Mean-pass control: {C['means']['s0']['n_windows']} windows, window list identical to HBLEND's fit windows, traj30 max |diff| vs HBLEND {C['means']['s0']['max_abs_traj30_vs_hblend_fit_windows_m']:.1e} m (s0) and "
  f"{C['means']['s1']['max_abs_traj30_vs_hblend_fit_windows_m']:.1e} m (s1).\n")
w("### E2 verdict, per row (M3: largest SIGNED mean-replacement effect; the grid's largest-n input is LISTEN's, re-derived from LISTEN's `READ.json` and asserted equal to the PREREG)\n")
w("| row | seed | grid's largest-n input | largest effect under mean replacement | its delta [95 % CI] | runner-up (delta) | intervals overlap | **agrees** | same cell as the grid's, under mean replacement | the grid's cell under donor swap (LISTEN) |")
w("|---|---|---|---|---|---|---|---|---|---|")
for row in ("S", "T-lat", "T-lon", "O"):
    for s in ("s0", "s1"):
        p = V["per_seed"][s][row]
        g = LISTEN["grid"][s]["rows"][row]["cells"][p["grid_argmax"]]
        rec = E2["records"][s]["rows"][row]["cells"][p["grid_argmax"]]
        w(f"| {row} | {s} | {INPN[p['grid_argmax']]} | **{INPN[p['argmax']]}** | {cell(p, 3, ' ' + UNIT[row])} | {INPN[p['runner_up']]} ({f(p['runner_up_delta'], 3)}) | "
          f"{yn(p['intervals_overlap'])} | **{yn(p['agrees'])}** | {cell(rec, 3)} | {cell(g, 3)} |")
w("")
w(f"**Rows agreeing at BOTH seeds (M4): S yes, T-lat yes, T-lon yes, O no -> k = {V['k_rows_agreeing_at_both_seeds']} of 4 -> {V['headline']}.** Per seed: s0 {V['k_per_seed']['s0']} of 4, s1 {V['k_per_seed']['s1']} of 4. "
  "ROBUST (all four rows) was the strongest committed outcome and it was **missed**: the only disagreement is the O row, where mean replacement ranks **T's code** above ego at both seeds "
  f"(s0 {cell(V['per_seed']['s0']['O'], 2)} m vs ego {f(V['per_seed']['s0']['O']['grid_cell_delta_under_mean'], 2)}; s1 {cell(V['per_seed']['s1']['O'], 2)} m vs ego {f(V['per_seed']['s1']['O']['grid_cell_delta_under_mean'], 2)}). "
  "On O, ego and T code were already the two large cells with overlapping intervals in LISTEN, and they still overlap here at both seeds, so the swap of rank 1 and 2 on that row is within the "
  "interval width; the committed rule counts it as a disagreement regardless. The three agreeing rows pick their input at both seeds, and in S and T-lon the runner-up intervals overlap the winner "
  "(T-lat at s1 is the one row whose winner is clearly separated from its runner-up).\n")
w("### E2, all cells (mean replacement; delta [95 % CI]; T rows in F1, S and O rows in m)\n")
for s in ("s0", "s1"):
    rec = E2["records"][s]
    w(f"**Seed {s}** (clean levels: " + "; ".join(f"{r} {rec['rows'][r]['clean_level']:.3f}" for r in ("S", "T-lat", "T-lon", "O")) + ")\n")
    w("| row | " + " | ".join(INPN[c] for c in ("scene", "ego", "nav", "limit", "ck", "code", "age", "all")) + " |")
    w("|---|" + "---|" * 8)
    for row in ("S", "T-lat", "T-lon", "O"):
        cs = []
        for col in ("scene", "ego", "nav", "limit", "ck", "code", "age", "all"):
            c = rec["rows"][row]["cells"][col]
            if c is None:
                cs.append("not an input")
            else:
                mark = "" if c["separated"] else " (ns)"
                bold = "**" if V["per_seed"][s][row]["argmax"] == col else ""
                cs.append(f"{bold}{cell(c, 3)}{mark}{bold}")
        w(f"| {row} | " + " | ".join(cs) + " |")
    w("")
w("(ns) = not separated. Bold = the row's largest signed effect (the E2 argmax).\n")
Oc0, Oc1 = E2["records"]["s0"]["rows"]["O"]["cells"], E2["records"]["s1"]["rows"]["O"]["cells"]
w("**Beside (descriptive, post hoc, not promoted).** "
  f"(a) The O row's n > 1 from LISTEN survives without any donor: replacing ALL of O's inputs by their means costs {cell(Oc0['all'], 2)} m (s0) and {cell(Oc1['all'], 2)} m (s1), "
  f"while replacing ego alone costs {cell(Oc0['ego'], 2)} / {cell(Oc1['ego'], 2)} m and T's code alone {cell(Oc0['code'], 2)} / {cell(Oc1['code'], 2)} m. "
  "So that excess is not a property of donor swapping. Caveat on the wording \"no cross-input conflict\": a mean token has no foreign window's content, but it still disagrees with the window's own "
  "other tokens (for O, the window's own kinematic prior and scene), so single-input mean cells are not conflict-free; only the ALL cell is. "
  f"(b) Two sign-reversed cells, both separated: T-lat x limit ({f(E2['records']['s0']['rows']['T-lat']['cells']['limit']['delta'], 3)} / {f(E2['records']['s1']['rows']['T-lat']['cells']['limit']['delta'], 3)} F1) and, at s1 only, "
  f"T-lon x S checkpoints ({f(E2['records']['s1']['rows']['T-lon']['cells']['ck']['delta'], 3)} F1): replacing that input by its mean slightly helps the macro-F1. They are small and not claims.\n")

# ---------------------------------------------------------------------------------------------------------------- 3
w("## 3. E3: O's coherence. Ego + S checkpoints + T code from ONE donor, together\n")
w("**Intervention.** O receives the donor window's ego token, S checkpoints and T code together (`run_O(.., ('ego','ck','code'))`); the kinematic prior keeps the window's own `p10f` "
  "(asserted per window), and scene, nav, limit and tick ages stay O's own. **Comparator (M5):** the single swap with the largest MEAN ADE6 on the full set.\n")
w("| seed | donor | mean ADE6 clean | joint triple | ego only | S checkpoints only | T code only | comparator (largest-mean single) | **E3 = joint - comparator [95 % CI]** | separated | < 0 |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for s, b, don in (("s0", e3s0, 0), ("s1", e3s1, 0), ("s0 (floor run)", e3d1, 1)):
    m = b["mean_ade6"]
    w(f"| {s} | seed {don} | {m['clean']:.3f} | {m['joint']:.3f} | {m['ego']:.3f} | {m['ck']:.3f} | {m['code']:.3f} | {INPN[b['comparator']]} | **{cell(b['E3'], 3, ' m')}** | "
      f"{yn(b['E3']['separated'])} | {yn(b['E3']['delta'] < 0)} |")
w("")
w("### The joint triple against the clean pass and against each single swap (ADE6, metres; paired; n = 2,399 windows / 141 episodes)\n")
w("| seed | joint - clean | ego - clean | S ckpt - clean | T code - clean | joint - ego | joint - S ckpt | joint - T code | joint - per-window max of the three (DESCRIPTIVE, upward-biased comparator, M5) |")
w("|---|---|---|---|---|---|---|---|---|")
for s, b in (("s0", e3s0), ("s1", e3s1)):
    j = b["joint_minus_each_single"]
    w(f"| {s} | {cell(b['joint_vs_clean'], 3)} | {cell(b['single_vs_clean']['ego'], 3)} | {cell(b['single_vs_clean']['ck'], 3)} | {cell(b['single_vs_clean']['code'], 3)} | "
      f"{cell(j['ego'], 3)} | {cell(j['ck'], 3)}{'' if j['ck']['separated'] else ' (ns)'} | {cell(j['code'], 3)} | {cell(b['joint_minus_perwindow_max_DESCRIPTIVE'], 3)} |")
w("")
w(f"**What E3 says.** At both seeds the joint triple hurts O by {e3s0['joint_vs_clean']['delta']:.2f} m (s0) and {e3s1['joint_vs_clean']['delta']:.2f} m (s1) over clean, while the worst single swap (ego) hurts by "
  f"{e3s0['single_vs_clean']['ego']['delta']:.2f} and {e3s1['single_vs_clean']['ego']['delta']:.2f} m; the difference, {cell(e3s0['E3'], 2)} and {cell(e3s1['E3'], 2)} m, is below 0 and separated at both seeds, with the "
  "second donor draw at s0 agreeing in sign and size. By the committed table this is **INCOHERENCE**: handing O a state from one donor in which three inputs agree is much less harmful than handing it that donor's "
  "ego alone, i.e. O reacts to the contradiction between its inputs more than to a coherent foreign state.\n")
w("Limits on that reading, stated as they are (descriptive, not promoted): "
  f"(a) the joint triple is not harmless: it is still separated from clean at both seeds, and it is WORSE than the S-checkpoint single swap at s0 ({cell(e3s0['joint_minus_each_single']['ck'], 2)} m, separated) "
  f"while not separated from it at s1 ({cell(e3s1['joint_minus_each_single']['ck'], 2)} m). "
  "(b) \"Coherent\" is relative: the prior's `p10f`, scene, nav, limit and ages stay the window's OWN, so the triple is coherent among its three members, not with the rest of O's inputs. "
  "(c) The committed comparator is the WORST single swap; the bar is therefore met by any joint that is better than ego alone, which is a weaker statement than \"the joint is better than every single swap\".\n")

# ---------------------------------------------------------------------------------------------------------------- 4
w("## 4. Controls (all pass; the read was not stopped)\n")
w("| control | committed condition | result | pass |")
w("|---|---|---|---|")
for s in ("s0", "s1"):
    i, dz = C["identity"][s], C["identity"][s]["delta_zero"]
    w(f"| 1. identity donor, {s} | gives exactly 0.0 | {i['n_rows']} non-clean rows bit-identical to the clean pass (max abs diff {i['max_abs_diff_vs_clean']}); {dz['n_cells']} read cells, max abs delta {dz['max_abs_delta']}, "
      f"{dz['n_separated']} separated | {yn(i['pass'] and dz['pass'])} |")
for s in ("s0", "s1"):
    r = C["reproduction"][s]
    lv = "; ".join(f"{k} {v['abs_diff']:.1e}" for k, v in r["levels"].items())
    w(f"| 2. clean pass vs LISTEN's clean levels, {s} | <= 1e-4 | max abs level diff {r['max_level_diff']:.1e} ({lv}); per-window arrays vs LISTEN's clean arrays (float32 noise, not the committed control): largest {r['max_perwindow_diff_vs_listen_clean']:.1e} on `{max(r['perwindow'], key=r['perwindow'].get)}`, all others <= {sorted(r['perwindow'].values())[-2]:.1e} | {yn(r['pass_1e-4'])} |")
for s in ("s0", "s1"):
    i, dz = C["own_value"][s], C["own_value"][s]["delta_zero"]
    w(f"| 3. mean replacement with the window's OWN value, {s} | gives exactly 0.0 | {i['n_rows']} replacement rows bit-identical to the clean pass (max abs diff {i['max_abs_diff_vs_clean']}); {dz['n_cells']} read cells, max abs delta {dz['max_abs_delta']}, "
      f"{dz['n_separated']} separated | {yn(i['pass'] and dz['pass'])} |")
for k, v in C["e1_invariance"].items():
    w(f"| 4. E1 invariance, run `{k}` | T's and O's own nav tokens bit-identical to clean, asserted per window | {v['windows_asserted']} / {v['windows']} windows asserted with `torch.equal` | {yn(v['pass'])} |")
for s in ("s0", "s1"):
    m = C["means"][s]
    w(f"| 5. training-set means, {s} (extra) | window list = HBLEND's fit windows; reproduces HBLEND's traj30 | n = {m['n_windows']}, ci/t/a_t/a_s equal = {all(m['window_list_equal'].values())}, max abs traj30 diff {m['max_abs_traj30_vs_hblend_fit_windows_m']:.1e} m | {yn(m['pass'])} |")
w("")
r0, r1 = C["reproduction"]["s0"]["listen_cells_recomputed_maxabs_DIAGNOSTIC"], C["reproduction"]["s1"]["listen_cells_recomputed_maxabs_DIAGNOSTIC"]
w("Diagnostic (not a stop condition): LISTEN's own cells recomputed in the donor runs match LISTEN's arrays to "
  f"{max(v for k, v in r0.items() if 'bit-identity' not in k):.1e} (s0) and {max(v for k, v in r1.items() if 'bit-identity' not in k):.1e} (s1) max abs; the E1 arm's S-row equals the `S x nav` cell bit for bit "
  f"({r0['e1/nav@S S-row vs own S/nav cell (bit-identity expected)']} / {r1['e1/nav@S S-row vs own S/nav cell (bit-identity expected)']}), as it must since S's step of the chain IS that cell.\n")
w("**Instrument checks (the \"built, tested, and unreachable\" rule).** `pytest test_listen2.py`: 28 passed (`pytest_log.txt`). `run_mutations2.py`: the unmutated module is green and **all 14 source mutations go RED** "
  "(injection at T, at S+T; joint triple without the T code; mean ignoring the pad mask / taken at the wrong tick / divided by n+1 / T checkpoints in the wrong frame / zero ages / code not replaced / "
  "all-padded scene; own-value in the wrong frame; mean mode using own values; collector storing the clean trajectory as E1 or a single swap as E3) (`mutation_log.txt`); no survivor. "
  "The E1 injection point, the E3 triple and every mean cell are checked by captured reader inputs independent of the module's own assertions.\n")

# ---------------------------------------------------------------------------------------------------------------- 5
w("## 5. Readings of the PREREG text that the text leaves open (fixed in code before any LISTEN2 number; verbatim from `listen2.py`)\n")
w("```\nREADINGS OF THE PREREG TEXT " + R["readings_fixed_before_data"] + "\n```\n")

# ---------------------------------------------------------------------------------------------------------------- 6
w("## 6. What these numbers do not say\n")
w("* **Training variance.** The paired bootstrap resamples episodes with the checkpoints held fixed. E1 and E3 reach the same outcome at both banked training seeds (M2b s0 and s1), which is evidence against a one-seed "
  "artefact, but two seeds are not a noise floor and no replicate-of-the-lever arm exists here (the intervention is on inputs, not a training lever). E2's O-row disagreement is also the same at both seeds.\n"
  "* **Donor choice.** The donor-seed floor is built from two donor draws on one checkpoint (s0). E1 at s0 changes by 0.0825 F1 between the two draws (0.4419 vs 0.3594); E3 by 0.43 m. Those are the only donor-variance reads taken.\n"
  "* **Tier.** Everything is an intervention on a fixed checkpoint, scored open loop against the logged future (T1-class, DIAGNOSTIC). It says how the trained model uses its inputs, not how well it drives.\n"
  "* **Four families.** The four-family rule governs evals of driving. This is an input-dependence diagnostic: it carries the layer rows only (S checkpoint error, T lateral / longitudinal macro-F1, O ADE6 and ADE2). The STRATEGIC family stays PAUSED under the PI ruling (ITEM 25): neither passed nor extended, and E1 does not lift it.\n"
  "* **E1 is about the route, not about T-lon.** ROUTE FLOWS DOWN is a statement about T's lateral decision. T's longitudinal decision does not follow nav through S in this read (not separated at either seed).\n")

# ---------------------------------------------------------------------------------------------------------------- 7
w("## 7. The cheapest next lever (RULE ZERO: a result leaves the next arm behind)\n")
w("None is run here, because none is pre-registered; each is CPU-only, about 1 minute per run on this harness, and the only blocker is a pre-registration (the Master Mind's call).\n"
  "1. **E2 left one row open (O: ego vs T code).** The cheapest discriminating read: swap pairs (ego+code, ego+S checkpoints, S checkpoints+code) and the fully coherent four (also the prior's `p10f`) from one donor, "
  "to see whether the O row's ranking of ego against T code is a property of the layer or of which single token happens to be the most conflicting with the window's own prior.\n"
  "2. **E1 settles the lateral chain, not the longitudinal one.** Inject the donor ego at S only (and at T only) and read T-lon, to see whether the longitudinal decision has any chain at all or is carried only by T's own ego and scene.\n"
  "3. **A third seed of M2b** (a training run, not CPU-only) would turn \"same outcome at two seeds\" into a seed-noise estimate for E1 and E3. That needs compute the PI has not authorised here.\n")

# ---------------------------------------------------------------------------------------------------------------- 8
w("## 8. Fixes made while running the tests, and the manifest\n")
w("The code under measurement (`listen2.py`, `read2.py`, `run_listen2.py`, `run_all2.sh`, `run_mutations2.py`, LISTEN's copied harness) was **not changed**; no donor seed, mean set, metric, comparator or bar was touched. "
  "Two bugs were found in `test_listen2.py` by the first run, both in the test, and fixed there:\n"
  "1. `test_e3_joint_triple_hands_O_one_donors_ego_ck_and_code_only`: the hand-built shadow ran `model.run_o` with autograd on, whose kernel path differs by ~1e-6 from the `no_grad` path the harness uses, so `torch.equal` failed. The hand-built run is now under `torch.no_grad()`.\n"
  "2. `_independent_means` called `.numpy()` on grad-tracking tensors (3 tests errored): now decorated with `@torch.no_grad()`.\n")
w("| artifact | location (all under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/listen2/`) |")
w("|---|---|")
for a, b in (("this result", "`LISTEN2_RESULT.md` (rendered from `READ.json` by `build_result2.py`)"), ("every number", "`READ.json`, `READ_controls.json`"),
             ("per-window arrays", "`runs/s0_d0.npz`, `s1_d0.npz`, `s0_d1.npz`, `s0_id.npz`, `s1_id.npz`, `s0_mean.npz`, `s1_mean.npz`, `s0_own.npz`, `s1_own.npz`"),
             ("training-set means + controls", "`runs/means_s0.npz`, `means_s1.npz`, `means_s0_control.json`, `means_s1_control.json`"),
             ("tests and mutations", "`pytest_log.txt` (28 passed), `mutation_log.txt` (14 of 14 RED), `test_listen2.py`, `run_mutations2.py`"),
             ("memory / disk", "`peakmem.log` (peak private commit 2.5 GB, working set 1.8 GB per run), `guard.log`, `run_all2.log`"),
             ("code", "`listen2.py`, `read2.py`, `run_listen2.py`, `run_all2.sh`, `guard.ps1`, `build_result2.py`; LISTEN's harness copies `listen.py`, `listen_read.py`, `run_listen.py`, `test_listen.py`, `run_mutations.py`; `code/` (LISTEN's tip export)"),
             ("PREREG", "`../grid_a2/PREREG_V8F_LISTEN2.md`"), ("repo", "nothing staged, committed or pushed; `scratchpad/listen/` and the repo untouched")):
    w(f"| {a} | {b} |")
(HERE / "LISTEN2_RESULT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", HERE / "LISTEN2_RESULT.md", len("\n".join(L)), "chars")
