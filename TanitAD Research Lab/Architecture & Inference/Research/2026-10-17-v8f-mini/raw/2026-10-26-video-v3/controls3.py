"""controls3.py -- CONTROLS_V3.json: HACC formula + banked plan, every on-screen number vs its source, v2's controls re-checked.   numpy + json only, ONE python process, read-only on every
input.  Run AFTER build_video3.py (it reads data/numreg.json and data/onscreen_text.json, which the build writes) and controls_v2_rerun.py (CONTROLS_V3_part2.json).
  python controls3.py"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
from v3_lib import SC, OUT, V1, V2, HACC, HBLEND, HSPEC, SOURCES, KEYS, WINDOWS, SEEDS, getp  # noqa: E402

TOL = 1e-5
C = {"spec": "viz3 controls: HACC as drawn vs P + w (tau - P) and the banked per-window plan; every on-screen number vs its JSON source (binding registry + RESULT md cross-check + "
             "existence scan of every drawn numeric token); v2's controls re-read and re-executed. Every check below carries its tolerance; PASS is computed here, not copied."}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
jl = lambda p: json.load(open(p))
SD = {"M2b_s0": "s0", "M2b_s1": "s1"}

# ====================================================================================================================== control H1: HACC formula + banked plan (+ a deliberate-regression arm)
pk = np.load(f"{OUT}/data/plans_pack3.npz")
h1 = {"per_seed": {}, "per_window": {}}
mut_axes, mut_w1 = 0.0, 0.0
for seed in SEEDS:
    hb, hc = np.load(f"{HBLEND}/score_{seed}_perwindow.npz"), np.load(f"{HACC}/score_{seed}_perwindow.npz")
    Wj = jl(f"{HACC}/weights_HACC_{seed}.json")
    w = np.stack([Wj["w_along_by_step"], Wj["w_cross_by_step"]], 1)
    P, tau = hb["P"].astype(np.float64), hb["tau"].astype(np.float64)
    formula = P + w * (tau - P)
    h1["per_seed"][seed] = {"n_windows": int(len(P)), "banked_HACC_vs_formula_maxabs_m": float(np.abs(hc["HACC"] - formula).max()),
                            "weights_file": f"hacc/weights_HACC_{seed}.json", "weights_shape": list(w.shape), "weights_sha256": sha(f"{HACC}/weights_HACC_{seed}.json")}
    # deliberate regressions: the check must be able to FAIL (axes swapped; w = 1 i.e. B itself)
    m1 = float(np.abs(hc["HACC"] - (P + w[:, ::-1] * (tau - P))).max())
    m2 = float(np.abs(hc["HACC"] - tau).max())
    mut_axes, mut_w1 = max(mut_axes, m1), max(mut_w1, m2)
    for name in KEYS:
        wi = WINDOWS[name]["window"]
        drawn = pk[f"{seed}/{name}/HACC"]
        h1["per_window"][f"{seed}/{name}"] = {"drawn_vs_formula_P_plus_w_tau_minus_P_m": float(np.abs(drawn - formula[wi]).max()), "drawn_vs_banked_perwindow_array_m": float(np.abs(drawn - hc["HACC"][wi]).max()),
                                              "drawn_P10_vs_hblend_P_m": float(np.abs(pk[f"{seed}/{name}/P10"] - hb["P"][wi]).max()), "drawn_B_vs_hblend_tau_m": float(np.abs(pk[f"{seed}/{name}/B"] - hb["tau"][wi]).max())}
h1["tolerance_m"] = TOL
vals = [v for d in h1["per_window"].values() for v in d.values()] + [d["banked_HACC_vs_formula_maxabs_m"] for d in h1["per_seed"].values()]
h1["max_abs_m"] = max(vals)
h1["deliberate_regression_arms"] = {"axes_swapped_maxabs_m": mut_axes, "w_equals_1_maxabs_m": mut_w1, "must_exceed_m": 1e-3, "both_detected": bool(mut_axes > 1e-3 and mut_w1 > 1e-3)}
h1["PASS"] = bool(h1["max_abs_m"] <= TOL and h1["deliberate_regression_arms"]["both_detected"])
h1["note"] = ("HACC as drawn (the arrays the renderer reads, data/plans_pack3.npz) equals P + w (.) (tau - P) recomputed here from hblend's P and tau and the banked weights (30 x 2, axis 0 along / axis 1 cross), "
              "and equals the banked per-window plan hacc/score_<seed>_perwindow.npz, over ALL 2,399 windows (eval-wide) and at the 4 drawn windows. P as drawn for P10 is hblend's `P` (the P10 FILE differs from it by "
              "<= 1.53e-5 m, RESULT_V8F_HACC control 3).")
C["control_H1_HACC_as_drawn_equals_P_plus_w_tau_minus_P_and_the_banked_plan"] = h1

# ====================================================================================================================== control H2: ADE6 recomputed, tables vs sources
h2 = {"per_window": {}, "eval_wide": {}}
for seed in SEEDS:
    hb, hs, hc = np.load(f"{HBLEND}/score_{seed}_perwindow.npz"), np.load(f"{HSPEC}/score_hs2_{seed}_perwindow.npz"), np.load(f"{HACC}/score_{seed}_perwindow.npz")
    for name in KEYS:
        wi = WINDOWS[name]["window"]
        gt = pk[f"{seed}/{name}/GT"].astype(np.float64)
        d = {}
        for arm, src, key in (("B", hb, "B__ade6"), ("H", hb, "H__ade6"), ("P10", hb, "P10__ade6"), ("HS", hs, "HS__ade6"), ("HACC", hc, "HACC__ade6")):
            a6 = float(np.linalg.norm(pk[f"{seed}/{name}/{arm}"].astype(np.float64) - gt, axis=-1).mean())
            d[f"ade6_recomputed_vs_banked_{arm}"] = abs(a6 - float(src[key][wi]))
            d[f"pack_ade6_{arm}_vs_recomputed"] = abs(float(pk[f"{seed}/{name}/ade6_{arm}"]) - a6)
        h2["per_window"][f"{seed}/{name}"] = d
    sj, hj = jl(f"{HACC}/score_{seed}.json"), jl(f"{HBLEND}/score_{seed}.json")
    e = {}
    for h in range(1, 7):
        e[f"ADE{h}_mean_of_banked_perwindow_vs_json_HACC"] = abs(float(hc[f"HACC__ade{h}"].mean()) - sj["single"]["HACC"][f"ade{h}"]["mean"])
    e["json_rounding_allowance"] = 5.1e-5
    e["B_H_P10_single_equal_to_hblend_json_ade1_6_mean_lo_hi_exact"] = all(sj["single"][a][f"ade{h}"][k] == hj["single"][a][f"ade{h}"][k] for a in ("B", "H", "P10") for h in range(1, 7) for k in ("mean", "lo", "hi"))
    t2 = jl(f"{OUT}/data/tables.json")["hblend_hspec"][seed]["ade"]
    e["B_H_P10_single_equal_to_tables_v2_exact"] = all(sj["single"][a][f"ade{h}"][k] == t2[a][f"ade{h}"][k] for a in ("B", "H", "P10") for h in range(1, 7) for k in ("mean", "lo", "hi"))
    hsj = jl(f"{HSPEC}/score_hs2_{seed}.json")
    e["HS_single_in_tables_v2_equals_hspec_json_exact"] = all(hsj["single"]["HS"][f"ade{h}"][k] == t2["HS"][f"ade{h}"][k] for h in range(1, 7) for k in ("mean", "lo", "hi"))
    h2["eval_wide"][seed] = e
bsc = {sd: jl(f"{HACC}/score_{sd}.json")["single"]["B"]["ade6"]["mean"] for sd in SEEDS}
Fexact = jl(f"{HACC}/READ.json")["F_exact_m"]
h2["F_check"] = {"F_exact_m_in_READ_json": Fexact, "abs_ADE6_B_s0_minus_s1_from_score_json_means": abs(bsc[SEEDS[0]] - bsc[SEEDS[1]]), "allowance": 1e-4}
ok = all(v <= TOL for d in h2["per_window"].values() for v in d.values())
ok &= all(d[k] <= d["json_rounding_allowance"] for d in h2["eval_wide"].values() for k in d if k.startswith("ADE"))
ok &= all(d[k] is True for d in h2["eval_wide"].values() for k in d if isinstance(d[k], bool))
ok &= abs(h2["F_check"]["F_exact_m_in_READ_json"] - h2["F_check"]["abs_ADE6_B_s0_minus_s1_from_score_json_means"]) <= 1e-4
h2["tolerance_m"] = TOL
h2["PASS"] = bool(ok)
C["control_H2_ade6_recomputed_from_drawn_plans_and_tables_equal_sources"] = h2

# ====================================================================================================================== control N1: the binding registry (every number registered by the scenes)
num = jl(f"{OUT}/data/numreg.json")
txt = jl(f"{OUT}/data/onscreen_text.json")
norm = lambda z: " ".join(z.split())
text_of = {t["scene"]: norm("\n".join(t["texts"])) for t in txt}
fresh = {}


def src_json(key):
    if key not in fresh:
        fresh[key] = json.load(open(SOURCES[key]))      # re-read from disk in THIS process
    return fresh[key]


bad, n_ok, n_not_drawn = [], 0, 0
for r in num:
    v = getp(src_json(r["src"]), r["ptr"])
    if "ptr2" in r:
        v = v - getp(src_json(r["src"]), r["ptr2"])
    v = v * r.get("scale", 1.0)
    exp = r["spec"].format(v)
    drawn = norm(r["shown"]) in text_of.get(r["scene"], "")
    if exp != r["shown"]:
        bad.append({"scene": r["scene"], "ptr": r["ptr"], "shown": r["shown"], "expected_from_source": exp})
    elif not drawn:
        n_not_drawn += 1
        bad.append({"scene": r["scene"], "ptr": r["ptr"], "shown": r["shown"], "problem": "registered string not found in the scene's drawn text"})
    else:
        n_ok += 1
n1 = {"n_registered_numbers": len(num), "n_bound_and_drawn": n_ok, "n_mismatch_or_not_drawn": len(bad), "first_failures": bad[:10],
      "n_distinct_sources": len({r['src'] for r in num}), "sources": {k: {"path": v.replace(SC, "<scratchpad>"), "sha256": sha(v)} for k, v in SOURCES.items() if k in {r['src'] for r in num}},
      "PASS": bool(len(bad) == 0 and len(num) > 300),
      "note": "Each registered number was produced by N() / ND() in the scene code from a JSON pointer; here the pointer is re-resolved from the file re-read in this process, re-formatted with the same spec, "
              "compared to the string the scene drew, and the string must occur in that frame's drawn text."}
C["control_N1_every_registered_on_screen_number_equals_its_json_source"] = n1

# ====================================================================================================================== control N2: dominance table / accel bands / KEEP vs RESULT_V8F_HACC.md (a second, independent source)
md = open(f"{HACC}/RESULT_V8F_HACC.md", encoding="utf-8").read()
sec4 = md[md.index("## 4. PRIMARY-DOM"):md.index("## 5.")]
ROWMAP = {"ADE2": "ADE2", "ADE6": "ADE6", "accel MAE": "accel MAE", "curvature MAE": "curvature MAE", "speed MAE": "speed MAE", "along MAE": "along MAE", "cross-track MAE": "cross-track MAE",
          "heading MAE": "heading MAE", "yaw-rate MAE": "yaw-rate MAE", "plan-derived lateral decision agreement": "plan-derived lateral decision agreement",
          "plan-derived longitudinal decision agreement": "plan-derived longitudinal decision agreement", "CONSISTENCY(a) 'any'": "CONSISTENCY(a) 'any'", "DK headway min (m)": "DK headway min (m)",
          "DK time-gap min (s)": "DK time-gap min (s)", "DK LEAD-VIOL at-risk share": "DK LEAD-VIOL at-risk share"}
cell = re.compile(r"([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]")
dom_text = text_of["dom_table(1.0,)"]
n2, bad2, nrow = {}, [], 0
for line in sec4.splitlines():
    if not line.startswith("| ") or line.startswith("| metric") or line.startswith("|---"):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if cells[0] not in ROWMAP:
        continue
    m = [cell.search(cells[i]) for i in (2, 4)]
    for sd, mm in zip(SEEDS, m):
        shown = f"{mm.group(1)} [{mm.group(2)}, {mm.group(3)}]"
        nrow += 1
        if norm(shown) not in dom_text:
            bad2.append({"row": cells[0], "seed": sd, "md": shown})
# accel bands (md section 6 vs the drawn band tables)
sec6 = md[md.index("## 6. The per-step accel profile"):md.index("## 7.")]
band_ok, band_n = 0, 0
for sdname, part in zip(SEEDS, re.split(r"\*\*M2b_s\d\*\*", sec6)[1:3]):
    for line in part.splitlines():
        mm = re.match(r"\| (0\.8-1\.6 s|2\.2-2\.6 s|2\.8-5\.0 s|5\.2-5\.8 s|all 29 cells)(?: \(cells [\d-]+\))? \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \|", line)
        if mm:
            band_n += 1
            lab = mm.group(1)
            exp = f"{lab:<14s}  {mm.group(2)}  {mm.group(3)}  {mm.group(4)}   {mm.group(6)}    {mm.group(7)}"
            if norm(exp) in text_of["dom_accel(1.0,)"]:
                band_ok += 1
            else:
                bad2.append({"band": lab, "seed": sdname, "expected_row": exp})
keep_md = {"s0": re.search(r"PRIMARY-KEEP M2b_s0: ADE6 HACC - H = (\+[\d.]+ \[[+-][\d.]+, \+[\d.]+\])", md).group(1), "s1": re.search(r"PRIMARY-KEEP M2b_s1: ADE6 HACC - H = (\+[\d.]+ \[[+-][\d.]+, \+[\d.]+\])", md).group(1)}
cav = text_of["dom_caveats(,)"] if "dom_caveats(,)" in text_of else text_of[[k for k in text_of if k.startswith("dom_caveats")][0]]
keep_ok = {k: (norm(v) in cav) for k, v in keep_md.items()}
if not all(keep_ok.values()):
    bad2.append({"keep_md": keep_md, "found": keep_ok})
n2 = {"dominance_rows_x_seeds_compared_with_RESULT_md_section_4": nrow, "accel_band_rows_compared_with_RESULT_md_section_6": band_n, "accel_band_rows_found_on_screen": band_ok,
      "KEEP_strings_from_md": keep_md, "KEEP_found_on_screen": keep_ok, "mismatches": bad2, "PASS": bool(nrow == 30 and band_n == 10 and band_ok == 10 and not bad2),
      "note": "An independent second source: the PI-facing RESULT_V8F_HACC.md tables (generated by hacc/build_result.py from the same JSON) parsed as text and required to appear verbatim in the drawn text."}
C["control_N2_dominance_table_accel_bands_and_KEEP_equal_RESULT_md"] = n2

# ====================================================================================================================== control N3: derived aggregates recomputed independently
rd = jl(f"{HACC}/READ.json")
rows = rd["per_seed"]["M2b_s0"]["PRIMARY_DOM"]["rows"]
nsep = {sd: sum(1 for k in rows if rd["per_seed"][sd]["PRIMARY_DOM"]["rows"][k]["separated"] and rd["per_seed"][sd]["PRIMARY_DOM"]["rows"][k]["better"]) for sd in SEEDS}
nworse = sum(1 for sd in SEEDS for k in rows if rd["per_seed"][sd]["PRIMARY_DOM"]["rows"][k]["worse"])
sj = {sd: jl(f"{HACC}/score_{sd}.json") for sd in SEEDS}
mx15 = max(abs(sj[sd]["paired"]["HACC_minus_H"][f"ade{h}"]["mean"]) for sd in SEEDS for h in range(1, 6))
margin = rd["KEEP_bar_m"] - rd["per_seed"]["M2b_s1"]["PRIMARY_KEEP"]["hi"]
nonmono = {sd: {ax: int(np.sum(np.diff(np.array(jl(f"{HACC}/weights_HACC_{sd}.json")[f"w_{ax}_by_step"])) < -1e-9)) for ax in ("along", "cross")} for sd in SEEDS}
d3 = {"separated_better_rows_of_15": nsep, "separated_worse_rows": nworse, "max_abs_ADE1_5_HACC_minus_H_m": mx15, "KEEP_margin_s1_m": margin, "non_monotone_steps": nonmono}
tdom = text_of["dom_table(1.0,)"]
checks3 = {"caption_counts_13_12": norm(f"on {nsep['M2b_s0']} of 15 rows at seed 0, {nsep['M2b_s1']} of 15 at seed 1") in tdom, "caption_worse_rows": norm(f"Separated-worse rows: {nworse}.") in tdom,
           "within_X_of_H": norm(f"within {mx15:.3f} m of H at 1-5 s") in text_of["plans_table_card()"],
           "keep_margin": norm(f"passes the bar by only {margin:.4f} m") in cav,
           "non_monotone_counts": norm(f"along {nonmono['M2b_s0']['along']} / {nonmono['M2b_s1']['along']}, cross {nonmono['M2b_s0']['cross']} / {nonmono['M2b_s1']['cross']}") in text_of["dom_mech()"]}
d3["checks_found_on_screen"] = checks3
d3["PASS"] = bool(all(checks3.values()) and nsep == {"M2b_s0": 13, "M2b_s1": 12} and nworse == 0)
C["control_N3_derived_aggregates_recomputed_independently_and_found_on_screen"] = d3

# ====================================================================================================================== control N4: existence scan of EVERY numeric token drawn (all frames)
corpus = set()


CF = []


def add_num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return
    CF.append(abs(float(x)))
    for sc in (1.0, 100.0):
        for d in range(0, 6):
            corpus.add(f"{abs(x * sc):.{d}f}")
        corpus.add(f"{abs(x * sc):,.0f}")


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    else:
        add_num(o)


for k in ("hacc_read", "hacc_score_s0", "hacc_score_s1", "hacc_accel", "hacc_dk", "hacc_w_s0", "hacc_w_s1", "hblend_w_s0", "hblend_w_s1", "hspec_s0", "hspec_s1", "sfresh", "sfresh_extra", "tables_v2", "cf_evalwide", "aux3"):
    walk(src_json(k))
for extra in (f"{SC}/listen/READ.json", f"{SC}/listen/READ_supplement.json", f"{SC}/listen/LISTEN_RESULT_tables.json", f"{SC}/hacc/fit_M2b_s0.json", f"{SC}/hacc/fit_M2b_s1.json"):
    if os.path.exists(extra):
        walk(jl(extra))
for f in ("hblend_w", "hacc_w"):
    pass
cfz = np.load(f"{OUT}/data/cf_windows.npz")
for k in cfz.files:
    a = cfz[k]
    if a.size <= 64 and a.dtype.kind in "fiu":
        for x in np.ravel(a).astype(float):
            add_num(float(x))
for k in pk.files:
    a = pk[k]
    if a.size <= 64 and a.dtype.kind in "fiu":
        for x in np.ravel(a).astype(float):
            add_num(float(x))
# declared constants: counts, ids, labels, units, protocol numbers that are not results (each is a name or a fixed protocol value)
DECL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "20", "24", "25", "29", "30", "50", "60", "75", "80", "95", "100", "120", "141", "256", "640", "1131", "654", "763", "2158",
        "2,399", "2,000", "2,319", "9,582", "0.2", "0.1", "300", "9a6551d5", "583cfc02", "1.5", "0.8", "1.6", "2.2", "2.6", "2.8", "5.0", "5.2", "5.8", "10-14", "26", "28", "29", "27",
        "2026", "0,1", "2.0", "0.5", "1e-5", "2e-5", "1.5e-5", "20260926", "17", "18", "21", "22", "23", "19", "31", "35", "59", "7", "11", "40", "45", "70"}
tok = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*(?:e-\d+)?)(?![\w])")
unmatched = {}
n_tokens = 0
for t in txt:
    for s in t["texts"]:
        for m_ in tok.finditer(s):
            n_tokens += 1
            raw = m_.group(1).rstrip(".")
            nraw = raw
            if nraw in DECL or nraw.replace(",", "") in {d.replace(",", "") for d in DECL}:
                continue
            if raw in corpus or raw.replace(",", "") in corpus:
                continue
            if "e-" in raw:
                v_ = float(raw)
                if any(abs(c_ - v_) <= 0.06 * v_ for c_ in CF if c_ > 0):
                    continue
            unmatched.setdefault(raw, []).append(t["scene"])
n4 = {"n_numeric_tokens_scanned": n_tokens, "n_distinct_unmatched": len(unmatched), "unmatched_tokens_with_first_scene": {k: v[0] for k, v in list(unmatched.items())[:60]},
      "corpus_size": len(corpus), "declared_constants": sorted(DECL), "PASS": len(unmatched) == 0,
      "note": ("EXISTENCE check, weaker than N1 (it cannot bind a number to its row): every numeric token drawn on ANY frame must be a declared constant (counts, ids, protocol values, band labels) or equal a value "
               "(at 0-5 decimals, abs, also x100) present in the source corpus (all JSON sources of the video + the LISTEN tables + small arrays of the banked npz). Anything else is a failure.")}
C["control_N4_every_numeric_token_drawn_is_a_declared_constant_or_a_source_value"] = n4

# ====================================================================================================================== control B1: the four primary dominance intervals re-derived with an independent numpy bootstrap
def _boot(diff, eid, B=2000, seed=1):
    eids = np.unique(eid)
    idx = {e: np.where(eid == e)[0] for e in eids}
    sums = np.array([diff[idx[e]].sum() for e in eids])
    cnt = np.array([len(idx[e]) for e in eids])
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        k = rng.integers(0, len(eids), len(eids))
        out[b] = sums[k].sum() / cnt[k].sum()
    return float(diff.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(len(eids))


b1 = {"per_seed_row": {}}
rd_ = jl(f"{HACC}/READ.json")
ROWS_B1 = (("ADE2", "ade2"), ("ADE6", "ade6"), ("accel MAE", "fw_lon_accel_mae"), ("curvature MAE", "fw_lat_curvature_mae_1pm"))
okb = True
for seed in SEEDS:
    hb_, hc_ = np.load(f"{HBLEND}/score_{seed}_perwindow.npz"), np.load(f"{HACC}/score_{seed}_perwindow.npz")
    for lab, key in ROWS_B1:
        a, b_ = hc_[f"HACC__{key}"].astype(np.float64), hb_[f"P10__{key}"].astype(np.float64)
        ok_ = np.isfinite(a) & np.isfinite(b_)
        m, lo, hi, ne = _boot((a - b_)[ok_], hb_["eid"][ok_])
        ref = rd_["per_seed"][seed]["PRIMARY_DOM"]["rows"][lab]
        wdt = ref["hi"] - ref["lo"]
        tolb = max(0.03 * wdt, 5e-4)
        r = {"own_mean_lo_hi": [m, lo, hi], "read_json_mean_lo_hi": [ref["mean"], ref["lo"], ref["hi"]], "n_windows": int(ok_.sum()), "n_episodes": ne, "tolerance": tolb,
             "ok": bool(abs(m - ref["mean"]) <= max(5.1e-5, 1e-4 * abs(ref["mean"])) + 5e-5 and abs(lo - ref["lo"]) <= tolb and abs(hi - ref["hi"]) <= tolb and hi < 0)}
        b1["per_seed_row"][f"{seed}/{lab}"] = r
        okb &= r["ok"]
b1["PASS"] = bool(okb)
b1["note"] = ("Independent numpy re-derivation (own seed, B = 2,000, resampling the 141 episodes, pooled window means) of the four primary rows of HACC - P10 from the banked per-window arrays; the READ.json intervals "
              "(taniteval/ci.py bootstrap, seed 0) must agree within 3 % of the interval width (the two bootstraps use different random streams) and exclude 0 on the better side. This is a cross-check, not a replacement of the registered estimator.")
C["control_B1_primary_row_intervals_reproduced_by_an_independent_bootstrap"] = b1

# ====================================================================================================================== control V2: v2's controls still pass
v2c = jl(f"{V2}/CONTROLS_V2.json")
v2 = {"CONTROLS_V2.json_sha256": sha(f"{V2}/CONTROLS_V2.json"), "ALL_PASS_as_recorded": bool(v2c["ALL_PASS"])}
c1 = v2c["control_1_own_nav_all_layers_reproduces_B_own_code_plan"]
c2 = v2c["control_2_drawn_plans_equal_banked_perwindow_arrays"]
c3 = v2c["control_3_frame_alignment_as_v1"]
c4 = v2c["control_4_zeroed_e_nav_structural_zero"]
v2["rederived"] = {"control_1_all_zero": all(c1[k] == 0.0 for k in c1 if k.startswith(("chain_", "own_nav"))),
                   "control_2_all_le_1e-5": all(c2[k] <= 1e-5 for k in c2 if k.endswith(("_source", "pred", "json")) or k.startswith("ade6_")),
                   "control_3_all_self_and_psnr_ge_48": bool(c3["total_argmax_is_self"] == c3["total_steps"] and c3["psnr_min_db"] >= 48.0),
                   "control_4_structural_zero_recorded": bool(c4["PASS"] and c4["max_abs"] == 0.0)}
# re-run on the data v3 draws: structural zero + D6 recomputation + v2 pack arrays unchanged
zk = [k for k in cfz.files if "/cfzero/" in k and k.endswith(("/D6", "/D6_TURN_L", "/D6_TURN_R", "/lat_sep_LR", "/LR_dist"))]
d6_err = 0.0
for sd in SEEDS:
    for name in KEYS:
        for cfg in ("a_all", "b_O", "c_T", "d_S"):
            f, l, r = (cfz[f"{sd}/{name}/cf/{cfg}/{c}/traj30"][29] for c in ("FOLLOW", "TURN_L", "TURN_R"))
            d6 = 0.5 * (np.linalg.norm(l - f) + np.linalg.norm(r - f))
            d6_err = max(d6_err, abs(float(d6) - float(cfz[f"{sd}/{name}/cf/{cfg}/D6"])), abs(float(abs(l[1] - r[1])) - float(cfz[f"{sd}/{name}/cf/{cfg}/lat_sep_LR"])))
v2pk = np.load(f"{V2}/data/plans_pack.npz")
pack_same = max(float(np.abs(pk[k] - v2pk[k]).max()) for k in v2pk.files)
copies = {f: sha(f"{OUT}/data/{f}") == sha(f"{V2}/data/{f}") for f in ("cf_windows.npz", "cf_evalwide.json", "tables.json")}
sel_sha = sha(f"{V2}/SELECTION_V2.md")
v2["rerun_on_v3_data"] = {"structural_zero_cells": len(zk), "structural_zero_max_abs": float(max(abs(float(cfz[k])) for k in zk)), "D6_and_LR_recomputed_from_drawn_plans_max_abs_m": d6_err,
                          "v3_pack_keys_of_v2_equal_v2_plans_pack_max_abs_m": pack_same, "data_copies_byte_identical_to_viz2": copies,
                          "SELECTION_V2.md_sha256": sel_sha, "SELECTION_V2.md_sha256_recorded_by_v2": v2c["pre_declaration"]["sha256"],
                          "selection_unchanged": sel_sha == v2c["pre_declaration"]["sha256"],
                          "the_4_windows_are_those_of_SELECTION": {k: {"window": WINDOWS[k]["window"], "clip8": WINDOWS[k]["clip"][:8], "t": WINDOWS[k]["t"]} for k in KEYS}}
p2 = jl(f"{OUT}/CONTROLS_V3_part2.json") if os.path.exists(f"{OUT}/CONTROLS_V3_part2.json") else None
v2["model_level_rerun_CONTROLS_V3_part2"] = ({k: p2[k] for k in ("control_1_v2_own_nav_all_layers_reproduces_forward", "regeneration_of_the_drawn_counterfactual_arrays", "frame_accessor_pixel_identical_to_v2", "ALL_PASS", "seconds")}
                                              if p2 else "MISSING: controls_v2_rerun.py has not been run")
v2["PASS"] = bool(v2["ALL_PASS_as_recorded"] and all(v2["rederived"].values()) and v2["rerun_on_v3_data"]["structural_zero_max_abs"] == 0.0 and d6_err <= 1e-5 and pack_same == 0.0
                  and all(copies.values()) and v2["rerun_on_v3_data"]["selection_unchanged"] and p2 is not None and p2["ALL_PASS"])
C["control_V2_v2_controls_still_pass"] = v2

# ====================================================================================================================== control N5: more independent cross-checks of what is drawn
n5 = {}
# (a) plans_table_card ADE means vs RESULT md section 5 (4 dp text): pointer correctness of the 40 registered cells per seed
sec5 = md[md.index("## 5. The four families"):md.index("## 6. The per-step")]
parts5 = re.split(r"\*\*M2b_s(\d)\*\*", sec5)
mdade = {}
for i_ in range(1, len(parts5), 2):
    sdn = "s" + parts5[i_]
    for mm in re.finditer(r"\| ADE/FDE \| ADE(\d) \(m\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", parts5[i_ + 1]):
        for arm, v in zip(("B", "H", "HACC", "P10"), mm.groups()[1:]):
            mdade[(sdn, arm, int(mm.group(1)))] = float(v)
bad_a, n_a = [], 0
for r in num:
    mm = re.match(r"single/(B|H|HACC|P10)/ade(\d)/mean$", r["ptr"])
    if r["scene"] == "plans_table_card()" and mm and r["src"].startswith("hacc_score_"):
        key = (r["src"][-2:], mm.group(1), int(mm.group(2)))
        n_a += 1
        if key not in mdade or abs(float(r["shown"]) - mdade[key]) > 6e-4:
            bad_a.append({"cell": key, "shown": r["shown"], "md": mdade.get(key)})
n5["ade_table_cells_vs_RESULT_md_section_5"] = {"n_cells": n_a, "n_md_values_parsed": len(mdade), "mismatches": bad_a, "tolerance": 6e-4, "PASS": bool(n_a == 48 and not bad_a and len(mdade) == 48)}
# (b) gap card rows vs RESULT md section 5 verbatim
rowspec = [("LONGITUDINAL | accel MAE (m/s^2)", {"HACC - H": 7, "HACC - P10": 9}), ("LATERAL | curvature MAE (1/m)", {"HACC - H": 7, "HACC - P10": 9}), ("LATERAL | cross-track MAE (m)", {"HACC - H": 7}),
           ("ADE/FDE | ADE6 (m)", {"HACC - H": 7}), ("ADE/FDE | FDE6 (m)", {"HACC - H": 7}), ("CONSISTENCY | viol_any", {"HACC - B": 8, "HACC - P10": 9})]
gtxt = text_of["plans_gap_card()"]
bad_b, n_b = [], 0
for i_ in range(1, len(parts5), 2):
    for lab, cols in rowspec:
        line = next((l for l in parts5[i_ + 1].splitlines() if l.startswith("| " + lab.split(" | ")[0] + " | " + lab.split(" | ")[1].split(" (")[0])), None)
        if line is None:
            bad_b.append({"row": lab, "missing_in_md": True})
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for cname, ci in cols.items():
            mm = cell.search(cells[ci - 1] if ci - 1 < len(cells) else "")
            if not mm:
                bad_b.append({"row": lab, "col": cname, "unparsed": cells[ci - 1:ci]})
                continue
            exp = f"{mm.group(1)} [{mm.group(2)}, {mm.group(3)}]"
            n_b += 1
            if norm(exp) not in gtxt:
                bad_b.append({"row": lab, "col": cname, "seed": parts5[i_], "expected": exp})
n5["gap_card_rows_vs_RESULT_md_section_5_verbatim"] = {"n_cells_compared": n_b, "mismatches": bad_b, "PASS": bool(n_b == 18 and not bad_b)}
# (c) HACC weights quoted on the mechanism card vs RESULT md section 3 (w* by step, i = 10 and 14)
sec3 = md[md.index("## 3. The fit"):md.index("## 4. PRIMARY-DOM")]
bad_c, n_c = [], 0
for sdn, part in zip(("s0", "s1"), re.split(r"\*\*M2b_s\d\.\*\*", sec3)[1:3]):
    for i_, ident in ((10, 9), (14, 13)):
        mm = re.search(rf"i={i_}: ([\d.]+) / ([\d.]+)", part)
        n_c += 1
        wv = json.load(open(f"{HACC}/weights_HACC_M2b_{sdn}.json"))["w_along_by_step"][ident]
        if abs(float(mm.group(1)) - wv) > 5e-4:
            bad_c.append({"seed": sdn, "i": i_, "md": mm.group(1), "json": wv})
        if f"{wv:.2f}" not in text_of["dom_mech()"]:
            bad_c.append({"seed": sdn, "i": i_, "not_on_screen": f"{wv:.2f}"})
n5["mechanism_card_weights_vs_RESULT_md_section_3"] = {"n_values": n_c, "mismatches": bad_c, "PASS": bool(n_c == 4 and not bad_c)}
# (d) the per-window ADE6 table of every plans frame vs the banked per-window arrays (independent of the pack)
bad_d, n_d = [], 0
for name in KEYS:
    wi = WINDOWS[name]["window"]
    vals = {}
    for sd in SEEDS:
        hb_, hs_, hc_ = np.load(f"{HBLEND}/score_{sd}_perwindow.npz"), np.load(f"{HSPEC}/score_hs2_{sd}_perwindow.npz"), np.load(f"{HACC}/score_{sd}_perwindow.npz")
        vals[sd] = {"P10": hb_["P10__ade6"][wi], "B": hb_["B__ade6"][wi], "H": hb_["H__ade6"][wi], "HS": hs_["HS__ade6"][wi], "HACC": hc_["HACC__ade6"][wi]}
    key_ = [k for k in text_of if k.startswith(f"plans_frame('{name}', 1.0")]
    for arm in ("P10", "B", "H", "HS", "HACC"):
        n_d += 1
        exp = f"{arm:<10s}{vals[SEEDS[0]][arm]:>8.2f}   {vals[SEEDS[1]][arm]:>8.2f}"
        if not key_ or norm(exp) not in text_of[key_[0]]:
            bad_d.append({"window": name, "arm": arm, "expected": exp})
n5["plans_frames_ade6_tables_vs_banked_perwindow_arrays"] = {"n_rows": n_d, "mismatches": bad_d, "PASS": bool(n_d == 20 and not bad_d)}
# (e) nav-counterfactual per-window numbers: D6 and left-right spread recomputed from the drawn plans, found in the frames that show them
bad_e, n_e = [], 0
for name in ("CURVE", "FAILURE(worst)"):
    key_ = [k for k in text_of if k.startswith(f"cf_four('{name}', 1.0")]
    for cfg in ("a_all", "b_O", "c_T", "d_S"):
        f_, l_, r_ = (cfz[f"M2b_s0/{name}/cf/{cfg}/{c}/traj30"][29] for c in ("FOLLOW", "TURN_L", "TURN_R"))
        d6 = 0.5 * (np.linalg.norm(l_ - f_) + np.linalg.norm(r_ - f_))
        exp = f"D6 = {d6:.1f} m   L-R spread = {abs(l_[1] - r_[1]):.1f} m"
        n_e += 1
        if not key_ or norm(exp) not in text_of[key_[0]]:
            bad_e.append({"window": name, "cfg": cfg, "expected": exp})
for name in ("JUNCTION", "NO-TURN"):
    key_ = [k for k in text_of if k.startswith(f"cf_bcd('{name}', 1.0")]
    for cfg in ("b_O", "c_T", "d_S"):
        f_, l_, r_ = (cfz[f"M2b_s0/{name}/cf/{cfg}/{c}/traj30"][29] for c in ("FOLLOW", "TURN_L", "TURN_R"))
        d6 = 0.5 * (np.linalg.norm(l_ - f_) + np.linalg.norm(r_ - f_))
        exp = f"D6 = {d6:.1f} m L-R lateral spread = {abs(l_[1] - r_[1]):.1f} m"
        n_e += 1
        if not key_ or norm(exp) not in text_of[key_[0]]:
            bad_e.append({"window": name, "cfg": cfg, "expected": exp})
n5["nav_counterfactual_D6_and_spread_recomputed_from_drawn_plans_found_on_screen"] = {"n_cells": n_e, "mismatches": bad_e, "PASS": bool(n_e == 14 and not bad_e)}
# (f) deliberate-regression arms: the string checks must be able to FAIL
mut = {}
num_mut = [dict(r) for r in num[:50]]
num_mut[7]["shown"] = num_mut[7]["shown"][:-1] + ("1" if num_mut[7]["shown"][-1] != "1" else "2")
bad_m = 0
for r in num_mut:
    v = getp(src_json(r["src"]), r["ptr"])
    if "ptr2" in r:
        v = v - getp(src_json(r["src"]), r["ptr2"])
    bad_m += int(r["spec"].format(v * r.get("scale", 1.0)) != r["shown"])
mut["N1_one_digit_flipped_is_detected"] = bad_m >= 1
mut["N2_one_md_cell_perturbed_is_not_found_on_screen"] = norm("-0.0107 [-0.0150, -0.0056]") not in dom_text
mut["N4_an_invented_number_is_flagged"] = (lambda tk: (tk not in corpus and tk not in DECL))("7.1234")
mut["N5d_wrong_arm_value_is_not_found"] = norm(f"{'HACC':<10s}{9.99:>8.2f}   {9.99:>8.2f}") not in text_of[[k for k in text_of if k.startswith("plans_frame('JUNCTION', 1.0")][0]]
n5["deliberate_regression_arms_all_detected"] = mut
n5["PASS"] = bool(all(v["PASS"] for v in n5.values() if isinstance(v, dict) and "PASS" in v) and all(mut.values()))
C["control_N5_independent_cross_checks_of_drawn_numbers_and_their_deliberate_regressions"] = n5

# ====================================================================================================================== wording guard: words the brief forbids must not be drawn (except the mandated negations)
alltxt = "\n".join(s for t in txt for s in t["texts"])
bad_words = {}
for pat in (r"\bdrives\b", r"\bSOTA\b", r"closed[- ]loop"):
    ms = [m.start() for m in re.finditer(pat, alltxt, flags=re.I)]
    ctxs = [alltxt[max(0, i - 14):i + 22].replace("\n", " ") for i in ms]
    bad_words[pat] = ctxs
allowed_neg = lambda c: ("NOT closed loop" in c) or ("not closed loop" in c)
viol = {k: [c for c in v if not allowed_neg(c)] for k, v in bad_words.items()}
C["control_W_forbidden_words"] = {"occurrences_with_context": {k: v[:12] for k, v in bad_words.items()}, "violations_after_removing_the_mandated_negation": viol,
                                  "PASS": all(len(v) == 0 for v in viol.values())}
stale = [s for t in txt for s in t["texts"] if re.search(r"\bpaused\b|un-paused|PAUSED", s)]
C["control_S_strategic_caption_not_stale"] = {"strings_mentioning_pause": stale[:5],
                                              "strategic_in_force_strings_found": sum(1 for t in txt for s in t["texts"] if "IN FORCE" in s), "PASS": len(stale) == 0 and any("IN FORCE" in s for t in txt for s in t["texts"])}

C["ALL_PASS"] = all(v.get("PASS", True) for v in C.values() if isinstance(v, dict))
json.dump(C, open(f"{OUT}/CONTROLS_V3.json", "w"), indent=1, default=float)
print(json.dumps({k: (v.get("PASS") if isinstance(v, dict) else None) for k, v in C.items()}, indent=1))
print("ALL_PASS", C["ALL_PASS"])
if n4["unmatched_tokens_with_first_scene"]:
    print("UNMATCHED", json.dumps(n4["unmatched_tokens_with_first_scene"], indent=0)[:3000])
if n1["first_failures"]:
    print("N1 FAILURES", json.dumps(n1["first_failures"], indent=0)[:2000])
if n2["mismatches"]:
    print("N2 MISMATCH", json.dumps(n2["mismatches"], indent=0)[:2000])
