#!/usr/bin/env python3
"""Seal PREREG_L2_FRESH_AMENDMENT2.md: fills PREREG_L2_FRESH_AMENDMENT2.src.md from the design calculation's JSON, the scoping probe's JSON, the MM ruling (VERBATIM) and the pins as the files on disk say they
are; writes the sealed file (LF bytes) + .sha256 + SEALED_AT_AMENDMENT2.txt (which records the B run dir listing, the chain log's last line and the read's seal-guard dry run on the PARENT before and after).
Refuses to overwrite a sealed file and refuses any unresolved placeholder.  Reads NO label and NO score.

    python seal_amendment2.py
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amendment2_check as AC  # noqa: E402

W = AC.W
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
B_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint"
SRC = f"{W}/PREREG_L2_FRESH_AMENDMENT2.src.md"
OUT = f"{W}/PREREG_L2_FRESH_AMENDMENT2.md"
RULING = f"{W}/MM_RULING_L2_FRESH_D.md"
CALC_JSON, CALC_CODE = f"{W}/raw/design_calc_ball.json", f"{W}/code/design_calc_ball.py"
SCOPE_JSON, SCOPE_CODE = f"{W}/raw/scope_unseen_stratum.json", f"{W}/code/scope_unseen_stratum.py"
PARENT = f"{W}/PREREG_L2_FRESH.md"


def f4(x):
    return f"{x:.4f}"


def e3(x):
    return f"{x:.3e}"


def seal_check_parent() -> str:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([PY, f"{W}/code/analyse_fresh.py", "--seal-check-only", "--prereg", PARENT, "--prereg-sha256", AC.sha_lf(PARENT), "--sample", f"{W}/raw/fresh_sample_token_disjoint.json"],
                       capture_output=True, text=True, env=env, timeout=300)
    return f"rc={r.returncode} " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:])


def values() -> dict:
    c = json.load(open(CALC_JSON, encoding="utf-8"))
    sc = json.load(open(SCOPE_JSON, encoding="utf-8"))
    assert c["identity"]["pass"]
    A = {s: c["arms"][s] for s in ("H_s0", "H_s1", "P2")}
    P = {s: A[s]["projection"] for s in A}
    al = c["allocation"]
    v = {}
    for tag, s in (("S0", "H_s0"), ("S1", "H_s1"), ("P2", "P2")):
        r = c["identity"]["arms"][s]["recomputed"]
        v[f"A1_{tag}"], v[f"A1_{tag}_LO"], v[f"A1_{tag}_HI"] = f"{r['point']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}"
    for tag, s in (("S0", "H_s0"), ("S1", "H_s1")):
        p, a, o = P[s], A[s], P[s]["outer_bootstrap_over_logs"]
        v[f"HW_OBS_{tag}"] = f4(p["hw_obs_f1"])
        v[f"HW_{tag}"] = f4(p["proportional_form"]["HW"])
        v[f"HW_EXACT_{tag}"] = f4(p["allocation_exact_form"]["HW"])
        v[f"HW_SIM_{tag}"] = f4(p["simulated_sealed_estimator"]["B_ALL_allocation"]["median"])
        v[f"HW_UP_{tag}"] = f4(p["plugin_resample_UPPER_bracket"]["median"])
        v[f"HW_FLOOR_{tag}"] = f4(p["floor_n_to_infinity_(b=0)"]["HW"])
        k2 = [k for k in p if k.startswith("at_2x")][0]
        v[f"HW_2X_{tag}"] = f4(p[k2]["HW"])
        lo, md, hi = o["HW_proportional_2.5_50_97.5"]
        v[f"OUT_{tag}"] = f"{f4(lo)} / {f4(md)} / {f4(hi)}"
        v[f"OUT_{tag}_LO"], v[f"OUT_MED_{tag}"] = f4(lo), f4(md)
        v[f"SHARE_OUT_{tag}"] = f"{100 * o['share_of_outer_draws_with_HW_proportional_<=0.025']:.1f} %"
        cf = a["anova_crosscheck"]["floor_closed_form_B_ALL"]
        v[f"CF_BAND_{tag}"] = f"{f4(cf['HW_floor_95pct_normal_theory'][0])} to {f4(cf['HW_floor_95pct_normal_theory'][1])}"
        v[f"CF_LO_{tag}"] = f4(cf["HW_floor_95pct_normal_theory"][0])
        v[f"A_{tag}"], v[f"B_{tag}"] = e3(a["fit"]["a"]), e3(a["fit"]["b"])
        v[f"R2_MEANS_{tag}"] = f"{a['fit']['R2_level_means']:.3f}"
        v[f"R2_ALL_{tag}"] = f"{a['fit']['all_draws']['R2']:.3f}"
    s0 = A["H_s0"]
    nA, nB = al["n_per_log_stepA"], al["n_per_log_B_ALL"]
    v["SHARE_STEPA_S0"] = f"{100 * s0['fit']['a'] / (s0['fit']['a'] + s0['fit']['b'] / nA):.0f} %"
    v["SHARE_B_S0"] = f"{100 * s0['fit']['a'] / (s0['fit']['a'] + s0['fit']['b'] / nB):.0f} %"
    an = s0["anova_crosscheck"]
    v["SB2_S0"], v["SW2_S0"], v["ICC_S0"] = f"{an['sigma_b2']:.2f}", f"{an['sigma_w2']:.1f}", f"{an['ICC_of_e']:.3f}"
    v["HW_FLOOR_CF_S0"] = f4(an["floor_closed_form_B_ALL"]["HW_floor_point"])
    v["DF"] = f"{an['floor_closed_form_B_ALL']['df']:.1f}"
    v["GEFF_A"], v["GEFF_B"] = f"{al['G_eff_stepA']:.1f}", f"{al['G_eff_B_ALL']:.1f}"
    v["CORR_ALLOC"] = f"{al['corr_per_log_counts']:.4f}"
    v["RATIO_MIN"], v["RATIO_MAX"] = f"{al['ratio_B_over_A_per_log_min_max'][0]:.2f}", f"{al['ratio_B_over_A_per_log_min_max'][1]:.2f}"
    v["WIN_LO"], v["WIN_HI"] = f"{s0['fit_window_n_tokens_per_log'][0]:.1f}", f"{s0['fit_window_n_tokens_per_log'][1]:.1f}"
    v["N_DRAWS"], v["R_REPS"] = str(s0["fit"]["all_draws"]["n_draws"]), str(c["args"]["reps"])
    v["N_OUTER"] = str(s0["projection"]["outer_bootstrap_over_logs"]["n_outer"])
    v["NB"], v["XF"] = f"{nB:.1f}", f"{al['extrapolation_factor_in_n']:.2f}"
    v["N2X"] = f"{2 * nA:.1f}"
    v["DBAR"] = f"{s0['dbar_pairs_per_token']:.2f}"
    b0 = c["context_banked_A4_A5_s0"]
    v["A4_S0"], v["A5_S0"] = f"{b0['A4']['point']:.4f} [{b0['A4']['lo']:.4f}, {b0['A4']['hi']:.4f}]", f"{b0['A5']['point']:.4f}"
    hw0, hw1 = P["H_s0"]["proportional_form"]["HW"], P["H_s1"]["proportional_form"]["HW"]
    v["HW_MAX"] = f4(max(hw0, hw1))
    v["PCT_GAIN"] = f"{100 * (1 - hw0 / P['H_s0']['hw_obs_f1']):.0f}"
    v["PT_NEEDED"] = f"{0.80 + hw0:.3f}"
    v["BRANCH"] = c["branch"]["selected"]
    # sizing table (balanced new logs; both seeds)
    rows = ["| tokens per log | logs needed (seed 0 / seed 1) | tokens in all (seed 0 / seed 1) | jpgs at 4 cameras (seed 0 / seed 1) |", "|---|---|---|---|"]
    sz = {s: A[s]["projection"]["sizing"] for s in ("H_s0", "H_s1")}
    for key in sz["H_s0"]:
        n = key.split("=")[1]
        g0, g1 = sz["H_s0"][key]["logs_needed_for_HW<=0.025"], sz["H_s1"][key]["logs_needed_for_HW<=0.025"]
        if float(n) > 1e8:
            rows.append(f"| infinity | {g0:.0f} / {g1:.0f} | - | - |")
        else:
            nn = float(n)
            rows.append(f"| {nn:g} | {g0:.0f} / {g1:.0f} | {g0 * nn:.0f} / {g1 * nn:.0f} | {4 * g0 * nn:.0f} / {4 * g1 * nn:.0f} |")
    v["SIZING_TABLE"] = "\n".join(rows)
    g4 = [max(sz["H_s0"]["n=4"]["logs_needed_for_HW<=0.025"], sz["H_s1"]["n=4"]["logs_needed_for_HW<=0.025"])]
    v["SZ4_LOGS"], v["SZ4_TOK"], v["SZ4_JPG"] = f"{g4[0]:.0f}", f"{4 * g4[0]:.0f}", f"{16 * g4[0]:.0f}"
    # scoping
    o, n_ = sc["overlaps"], sc["navtrain_token_table"]
    v["N_TV"], v["N_TV_IN_NT"], v["N_NT"] = str(sc["db_name_sets"]["splits/trainval"]), str(o["navtrain_named_in_trainval"]), str(n_["named_logs"])
    v["N_U"] = str(sc["candidate_sets"]["U_trainval_minus_navtrain (A1.5(ii) by name)"])
    v["N_ROOTS"] = str(len(sc["location_1_local_frame_roots"]))
    # pins
    v["PARENT_SHA"], v["RULING_SHA"] = AC.sha_lf(PARENT), AC.sha_lf(RULING)
    v["CALC_CODE_SHA"], v["CALC_JSON_SHA"] = AC.sha_lf(CALC_CODE), AC.sha_raw(CALC_JSON)
    v["SCOPE_CODE_SHA"], v["SCOPE_JSON_SHA"] = AC.sha_lf(SCOPE_CODE), AC.sha_raw(SCOPE_JSON)
    v["MM_RULING"] = open(RULING, "rb").read().replace(b"\r\n", b"\n").decode("utf-8").rstrip("\n")
    return v


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"refusing to overwrite the sealed file {OUT}")
    before = seal_check_parent()
    v = values()
    txt = open(SRC, encoding="utf-8").read()
    for k, val in v.items():
        txt = txt.replace("{{" + k + "}}", val)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", txt)
    if left:
        raise SystemExit(f"unresolved placeholders: {sorted(set(left))}")
    if "Status: DRAFT" in txt or "Status: SEALED" not in txt:
        raise SystemExit("the sealed text must carry the SEALED marker and no draft marker")
    if v["MM_RULING"] not in txt:
        raise SystemExit("the ruling is not carried verbatim")
    b_files = sorted(os.listdir(B_DIR)) if os.path.isdir(B_DIR) else []
    forbidden = [f for f in b_files if f in ("bundle", "perts.npz", "labels.npz", "scores.npz", "feats", "fresh_ep_scores.json", "FRESH_RESULT.json")]
    if forbidden:
        raise SystemExit(f"a label / score artifact of the B run exists already: {forbidden}; this amendment must be sealed BEFORE the read")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sha = AC.sha_lf(OUT)
    open(f"{W}/PREREG_L2_FRESH_AMENDMENT2.sha256", "w", newline="\n").write(f"{sha}  PREREG_L2_FRESH_AMENDMENT2.md\n")
    after = seal_check_parent()
    chain = open(f"{W}/chain_B.log", encoding="utf-8", errors="replace").read().splitlines()
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [f"SEALED_AT {now}", "file PREREG_L2_FRESH_AMENDMENT2.md", f"sha256 (LF bytes) {sha}", "", "state of the pre-launched B chain at the seal (no fresh label or score existed):",
             f"  chain_B.log last line: {chain[-1] if chain else '(empty)'}", f"  B_token_disjoint/ holds: {b_files}", "",
             "the read's seal guard on the PARENT (analyse_fresh.py --seal-check-only; reads no label, no score), before and after this file was written:", f"  before: {before}", f"  after:  {after}",
             f"  identical: {before == after}", "", "pins named in the sealed text:", f"  parent PREREG_L2_FRESH.md: {v['PARENT_SHA']}", f"  ruling MM_RULING_L2_FRESH_D.md: {v['RULING_SHA']}",
             f"  design_calc_ball.py: {v['CALC_CODE_SHA']}", f"  design_calc_ball.json: {v['CALC_JSON_SHA']}", f"  scope_unseen_stratum.py: {v['SCOPE_CODE_SHA']}", f"  scope_unseen_stratum.json: {v['SCOPE_JSON_SHA']}"]
    open(f"{W}/SEALED_AT_AMENDMENT2.txt", "w", newline="\n").write("\n".join(lines) + "\n")
    info = AC.check_amendment(OUT, sha, PARENT, RULING, CALC_JSON, CALC_CODE)
    print("ZZSEALED_AMENDMENT2", sha, json.dumps(info), "guard before/after identical:", before == after, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
