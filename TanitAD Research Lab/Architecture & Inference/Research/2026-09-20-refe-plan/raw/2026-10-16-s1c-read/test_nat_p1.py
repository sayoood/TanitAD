#!/usr/bin/env python3
"""Tests for nat_p1.py.  Prints `ZZTEST_PASS <n>` (exit 0) or `ZZTEST_FAIL ...` (exit 1).  Run in the driverl-eval venv:  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_nat_p1.py

Synthetic checks (the maths of the weights, the estimator, the guards), then the REAL-DATA IDENTITY (S1's H1-s0 / H1-s1 / H1-s2 navtest predictions reproduce SHIFT_LOCATE's banked natural reads
0.5608520369346311 / 0.5768301739064818 / 0.5539321832326178 and ESS 575.3679 through nat_p1.run, i.e. through the harness's own frame_stats), then deliberate regressions: the SOURCE is edited one
line at a time and the identity must go RED.  The real-data part is SKIPPED (and counted as skipped, not passed) when the regenerated predictions are absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nat_p1 as N  # noqa: E402

SRC = (HERE / "nat_p1.py").read_text(encoding="utf-8")
SCRATCH = Path(os.environ.get("SHF_SCRATCH", "C:/Users/Admin/AppData/Local/Temp/shf_scratch"))
N_OK, FAILS, SKIPPED = 0, [], []


def check(name, cond, detail=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(f"{name} {detail}".strip())
        print("  FAIL", name, detail, flush=True)


def mutated(old, new):
    assert SRC.count(old) == 1, f"mutation anchor not unique: {old!r} x{SRC.count(old)}"
    m = types.ModuleType("nat_p1_mut")
    m.__file__ = str(HERE / "nat_p1.py")
    exec(compile(SRC.replace(old, new), str(HERE / "nat_p1.py"), "exec"), m.__dict__)
    return m


def synth(n=420, seed=3):
    rng = np.random.default_rng(seed)
    maps = rng.choice(np.array(N.MAPS), n, p=[0.15, 0.35, 0.3, 0.2])
    strata = np.where(rng.random(n) < 0.5, "zero", "random")
    return maps, strata


SHARE_MAP = {"LV": 0.33, "BOS": 0.31, "PIT": 0.2, "SG": 0.16}
SHARE_ST = {"zero": 0.05, "random": 0.95}
SPEC = {"share_map": SHARE_MAP, "share_stratum": SHARE_ST}


def main():
    # ---------------------------------------------------------------- the weights
    maps, strata = synth()
    w = N.natural_weights(maps, strata, SPEC)
    check("weights_mean_one", abs(w.mean() - 1.0) < 1e-12)
    check("weights_positive", bool((w > 0).all()))
    check("map_margin_satisfied", all(abs(w[maps == m].sum() / w.sum() - SHARE_MAP[m]) < 1e-9 for m in N.MAPS))
    check("stratum_margin_satisfied", all(abs(w[strata == s].sum() / w.sum() - SHARE_ST[s]) < 1e-9 for s in SHARE_ST))
    check("ess_of_equal_weights_is_n", abs(N.ess(np.ones(50)) - 50) < 1e-9)
    check("ess_below_n_for_unequal", N.ess(w) < len(w))
    # a sample that already has the population's composition gets weight 1 everywhere
    pm = np.array(sum(([m] * int(round(SHARE_MAP[m] * 1000)) for m in N.MAPS), []))
    ps = np.array(["zero" if i % 20 == 0 else "random" for i in range(len(pm))])
    spec2 = {"share_map": {m: float((pm == m).mean()) for m in N.MAPS}, "share_stratum": {s: float((ps == s).mean()) for s in ("zero", "random")}}
    w2 = N.natural_weights(pm, ps, spec2)
    check("weights_one_when_sample_equals_population", float(np.abs(w2 - 1).max()) < 1e-9)
    # a population map that is absent from the sample is skipped (b1's `if s > 0`), the rest still fit
    keep = maps != "SG"
    w3 = N.natural_weights(maps[keep], strata[keep], SPEC)
    check("absent_map_is_skipped_not_crashed", bool(np.isfinite(w3).all()) and abs(w3.mean() - 1) < 1e-12)
    # ipf_rows == rake on a row of ones; margins satisfied on resample rows
    mg = np.asarray([N.MAPS.index(m) for m in maps])
    sg = np.asarray([0 if s == "zero" else 1 for s in strata])
    groupings = [(mg, np.asarray([SHARE_MAP[m] for m in N.MAPS])), (sg, np.asarray([SHARE_ST["zero"], SHARE_ST["random"]]))]
    Mf, resid = N.ipf_rows(np.ones((1, len(maps))), groupings, 200)
    check("ipf_rows_equals_rake_on_ones", float(np.abs(Mf[0] / Mf[0].mean() - w).max()) < 1e-9)
    rng = np.random.default_rng(5)
    Mr = rng.multinomial(len(maps), np.ones(len(maps)) / len(maps), size=64).astype(float)
    Mrf, resid2 = N.ipf_rows(Mr, groupings, 60)
    check("ipf_rows_margins_satisfied_per_resample", resid2 < 1e-9, f"resid {resid2}")

    # ---------------------------------------------------------------- the estimator: a hand-rolled bootstrap against the module's
    n = 300
    rng = np.random.default_rng(11)
    keys = [(f"log{i // 5:03d}", f"tok{i:04d}") for i in range(n)]
    union = rng.integers(40, 200, n).astype(float)
    inter_a = np.floor(union * rng.uniform(0.3, 0.9, n))
    inter_b = np.floor(union * rng.uniform(0.2, 0.8, n))
    st_a, st_b = {"inter": inter_a, "union": union}, {"inter": inter_b, "union": union}
    mp, sr = synth(n, 9)
    wn = N.natural_weights(mp, sr, SPEC)
    ul = sorted(set(k[0] for k in keys))
    W = N.RO.boot_weights(len(ul), 2000, 20260927)
    r_unw = N.read_population(keys, st_a, None, W)
    r_w = N.read_population(keys, st_a, wn, W)
    lix = np.asarray([ul.index(k[0]) for k in keys])
    est_u = inter_a.sum() / union.sum()
    est_w = (wn * inter_a).sum() / (wn * union).sum()
    check("unweighted_point_is_ratio_of_sums", abs(r_unw["est"] - est_u) < 1e-12)
    check("weighted_point_is_weighted_ratio_of_sums", abs(r_w["est"] - est_w) < 1e-12)
    rngb = np.random.default_rng(20260927)
    reps = []
    for _ in range(2000):
        mult = np.bincount(rngb.integers(0, len(ul), len(ul)), minlength=len(ul))[lix]
        reps.append((mult * wn * inter_a).sum() / (mult * wn * union).sum())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    check("fixed_weight_interval_equals_hand_bootstrap", abs(lo - r_w["ci95"][0]) < 1e-12 and abs(hi - r_w["ci95"][1]) < 1e-12, f"{lo} {hi} vs {r_w['ci95']}")
    check("interval_brackets_point", r_w["ci95"][0] < r_w["est"] < r_w["ci95"][1])
    d_self = N.read_contrast(keys, st_a, st_a, wn, W)
    check("contrast_with_itself_is_exactly_zero", d_self["est"] == 0.0 and d_self["ci95"] == [0.0, 0.0])
    d_ab = N.read_contrast(keys, st_a, st_b, wn, W)
    check("contrast_est_is_difference_of_points", abs(d_ab["est"] - (est_w - (wn * inter_b).sum() / (wn * union).sum())) < 1e-12)
    d_ba = N.read_contrast(keys, st_b, st_a, wn, W)
    check("contrast_antisymmetric", abs(d_ab["est"] + d_ba["est"]) < 1e-12 and abs(d_ab["ci95"][0] + d_ba["ci95"][1]) < 1e-12)
    # sensitivity: with W = every log exactly once (no resampling) the re-fitted weights equal the point-estimate weights
    W1 = np.ones((1, len(ul)))
    reps1, resid3, absent = N.reraked_reps(keys, mp, sr, SPEC, W1, [st_a])
    check("reraked_with_no_resampling_equals_point_estimate", abs(reps1[0][0] - est_w) < 1e-9, f"{reps1[0][0]} vs {est_w}")
    check("reraked_margins_satisfied", resid3 < 1e-9 and absent == 0)
    repsB, residB, _ = N.reraked_reps(keys, mp, sr, SPEC, W, [st_a, st_b])
    check("reraked_reps_finite_and_paired", bool(np.isfinite(repsB[0]).all()) and repsB[0].shape == (2000,) and residB < 1e-9)

    # ---------------------------------------------------------------- guards
    tokj = {"tokens": ["a", "b"], "stratum": {"a": "zero", "b": "random"}, "provenance": {"n_csv_tokens": 12146, "n_zero_NCxDAC": 529}}
    pop = {"log_map": {"L1": "LV", "L2": "BOS"}, "count_by_map": dict(N.FROZEN_POP), "n": 12146, "n_shards": 136}
    sp = N.build_spec(tokj, pop)
    check("spec_equals_frozen_shift_locate_targets", all(abs(sp["share_map"][m] - N.FROZEN_SHARE_MAP[m]) < 1e-15 for m in N.MAPS) and abs(sp["share_stratum"]["zero"] - N.FROZEN_SHARE_STRATUM["zero"]) < 1e-15)
    for label, bad_pop in (("population_total_differs_from_provenance", dict(pop, n=12145, count_by_map=dict(pop["count_by_map"], LV=4063))),
                           ("population_drifted_from_frozen_targets", dict(pop, count_by_map=dict(pop["count_by_map"], LV=4063, BOS=3732)))):
        try:
            N.build_spec(tokj, bad_pop)
            check(label + "_REFUSED", False)
        except N.NatRefused:
            check(label + "_REFUSED", True)
    try:
        N.build_spec(tokj, dict(pop, count_by_map=dict(pop["count_by_map"], LV=4063, BOS=3732)), frozen_check=False)
        check("frozen_check_can_be_turned_off_for_maths_tests", True)
    except N.NatRefused:
        check("frozen_check_can_be_turned_off_for_maths_tests", False)
    for label, kj, lm in (("frame_without_stratum", {"stratum": {}}, {"L1": "LV"}), ("frame_without_city", {"stratum": {"t": "zero"}}, {}), ("frame_with_alien_stratum", {"stratum": {"t": "other"}}, {"L1": "LV"})):
        try:
            N.frame_labels([("L1", "t")], kj, lm)
            check(label + "_REFUSED", False)
        except N.NatRefused:
            check(label + "_REFUSED", True)
    mm, ss = N.frame_labels([("L1", "a"), ("L2", "b")], tokj, pop["log_map"])
    check("frame_labels_ok", list(mm) == ["LV", "BOS"] and list(ss) == ["zero", "random"])
    check("identity_block_none_for_unbanked_arm", N.identity_block("H1u10k-s0", 0.5, 0.6, 575.0) is None)
    b = N.identity_block("H1-s0", N.BANKED_C3["H1-s0"], N.BANKED_NATURAL["H1-s0"], N.BANKED_ESS)
    check("identity_block_ok_on_banked", b["ok"] and b["abs_diff_natural"] == 0.0)
    check("identity_block_RED_on_1e8_natural_shift", not N.identity_block("H1-s0", N.BANKED_C3["H1-s0"], N.BANKED_NATURAL["H1-s0"] + 1e-8, N.BANKED_ESS)["ok"])
    check("identity_block_RED_on_wrong_ess", not N.identity_block("H1-s0", N.BANKED_C3["H1-s0"], N.BANKED_NATURAL["H1-s0"], N.BANKED_ESS + 1.0)["ok"])
    check("band_index_is_the_k2_band", N.K2_BAND_INDEX == 3 and N.RO.K2_BAND == (0.0, 40.0))
    check("pins_equal_the_harness", N.N_BOOT == 10000 and N.SEED == 20260927 and N.N_BOOT == N.RO.N_BOOT and N.SEED == N.RO.SEED)

    # ---------------------------------------------------------------- real data: the identity
    preds = {a: SCRATCH / f"navtest_head_pred_{a}.npz" for a in ("H1-s0", "H1-s1", "H1-s2")}
    if not all(p.is_file() for p in preds.values()) or not (N.C3_DIR / "tokens.json").is_file():
        SKIPPED.append("real-data identity (the regenerated S1 navtest predictions are absent)")
    else:
        tmp = tempfile.mkdtemp(prefix="natp1_")
        out = os.path.join(tmp, "nat.json")
        res = N.run({a: str(p) for a, p in preds.items()}, [("H1-s1", "H1-s0"), ("H1-s0", "H1-s0")], N.C3_DIR / "tokens.json", out, n_boot=10000, sensitivity=True)
        for a in ("H1-s0", "H1-s1", "H1-s2"):
            nat_est = res["arms"][a]["natural"]["est"]
            check(f"REAL_natural_read_{a}_equals_SHIFT_LOCATE", abs(nat_est - N.BANKED_NATURAL[a]) <= 1e-12, f"{nat_est} vs {N.BANKED_NATURAL[a]}")
            check(f"REAL_c3_read_{a}_equals_banked_navtest_P1", abs(res["arms"][a]["c3"]["est"] - N.BANKED_C3[a]) <= 1e-12)
            check(f"REAL_rounded_{a}_is_0.561_0.577_0.554", round(nat_est, 3) == {"H1-s0": 0.561, "H1-s1": 0.577, "H1-s2": 0.554}[a])
        check("REAL_ess_575.3679", abs(res["population"]["ess"] - 575.3679059904473) < 1e-9)
        check("REAL_identity_ok_flag", res["identity_ok"] is True)
        banked = json.load(open(N.PK / "raw/2026-10-15-s1-read/instrument/perception_navtest.json", encoding="utf-8"))["arms"]["H1-s0"]["p1"]["all"]["0-40"]["ci95"]
        check("REAL_c3_interval_equals_harness_interval", all(abs(x - y) < 1e-12 for x, y in zip(res["arms"]["H1-s0"]["c3"]["ci95"], banked)), f"{res['arms']['H1-s0']['c3']['ci95']} vs {banked}")
        c = res["contrasts"]["H1-s1-minus-H1-s0"]
        check("REAL_contrast_c3_is_difference_of_banked_points", abs(c["c3"]["est"] - (N.BANKED_C3["H1-s1"] - N.BANKED_C3["H1-s0"])) < 1e-12)
        check("REAL_contrast_nat_is_difference_of_banked_points", abs(c["natural"]["est"] - (N.BANKED_NATURAL["H1-s1"] - N.BANKED_NATURAL["H1-s0"])) < 1e-12)
        cs = res["contrasts"]["H1-s0-minus-H1-s0"]
        check("REAL_zero_control_self_contrast_exact", cs["c3"]["est"] == 0.0 and cs["natural"]["est"] == 0.0 and cs["natural"]["ci95"] == [0.0, 0.0] and cs["c3"]["ci95"] == [0.0, 0.0])
        check("REAL_population_cells_n", res["population"]["n_by_cell"] == {"BOS|random": 164, "BOS|zero": 146, "LV|random": 171, "LV|zero": 62, "PIT|random": 114, "PIT|zero": 162, "SG|random": 80, "SG|zero": 159})
        check("REAL_per_city_has_four_cities", sorted(res["arms"]["H1-s0"]["per_city"]) == sorted(N.MAPS))
        check("REAL_sensitivity_residual_tiny", res["sensitivity"]["max_margin_residual"] < 1e-9 and res["sensitivity"]["gating"] is False)
        js = json.load(open(out, encoding="utf-8"))
        check("REAL_json_written_and_stamped", js["tool"] == N.TOOL and "LOGS" in js["estimator"] and "T0-like" in js["tier"])
        # the CLI: exit 0 on the identity, 6 when a banked constant is made wrong in a COPY of the module
        r = subprocess.run([sys.executable, "-I", str(HERE / "nat_p1.py"), "--pred"] + [f"{a}={p}" for a, p in preds.items()] + ["--no-sensitivity", "--n-boot", "200", "--out", os.path.join(tmp, "cli.json")],
                           capture_output=True, text=True, encoding="utf-8")
        check("REAL_cli_exit_0_and_marker", r.returncode == 0 and "ZZNATP1_OK" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
        bad = mutated("BANKED_NATURAL = {\"H1-s0\": 0.5608520369346311", "BANKED_NATURAL = {\"H1-s0\": 0.5608520469346311")
        rb = bad.run({a: str(p) for a, p in preds.items()}, [], N.C3_DIR / "tokens.json", None, n_boot=200, sensitivity=False)
        check("REAL_identity_RED_when_the_banked_constant_is_off_by_1e_8", rb["identity_ok"] is False)
        # ---- deliberate regressions of the weights: edit the SOURCE, the natural read must leave the banked value
        from_cache = {}
        pr = N.RO.load_head_pred(str(preds["H1-s0"]))
        src_lab = N.RO.open_labels("navtest")
        tj = json.load(open(N.C3_DIR / "tokens.json", encoding="utf-8"))
        pop_r = N.navtest_population()
        spec_r = N.build_spec(tj, pop_r)
        maps_r, strata_r = N.frame_labels(pr["keys"], tj, pop_r["log_map"])
        stats_r = N.arm_stats(pr, src_lab)
        from_cache.update(maps=maps_r, strata=strata_r, stats=stats_r, spec=spec_r)

        def nat_est_with(mod):
            ww = mod.natural_weights(from_cache["maps"], from_cache["strata"], from_cache["spec"])
            return mod.pooled(from_cache["stats"]["inter"], from_cache["stats"]["union"], ww)
        check("REAL_unmutated_weights_give_the_banked_natural", abs(nat_est_with(N) - N.BANKED_NATURAL["H1-s0"]) <= 1e-12)
        muts = [
            ("W1 the stratum margin dropped from the fit", "    return rake(maps, strata, spec[\"share_map\"], spec[\"share_stratum\"], IPF_ITERS)", "    return rake(maps, strata, spec[\"share_map\"], None, IPF_ITERS)"),
            ("W2 the map margin dropped from the fit", "    return rake(maps, strata, spec[\"share_map\"], spec[\"share_stratum\"], IPF_ITERS)", "    return rake(maps, strata, None, spec[\"share_stratum\"], IPF_ITERS)"),
            ("W3 the stratum shares swapped", "    return rake(maps, strata, spec[\"share_map\"], spec[\"share_stratum\"], IPF_ITERS)",
             "    return rake(maps, strata, spec[\"share_map\"], {\"zero\": spec[\"share_stratum\"][\"random\"], \"random\": spec[\"share_stratum\"][\"zero\"]}, IPF_ITERS)"),
            ("W4 a single IPF pass", "IPF_ITERS, IPF_ITERS_BOOT = 200, 60", "IPF_ITERS, IPF_ITERS_BOOT = 1, 60"),
            ("W5 the weights do not enter the numerator", "    return float((w[m] * inter[m]).sum() / d) if d > 0 else float(\"nan\")", "    return float((inter[m]).sum() / d) if d > 0 else float(\"nan\")"),
        ]
        for label, old, new in muts:
            mod = mutated(old, new)
            try:
                red = abs(nat_est_with(mod) - N.BANKED_NATURAL["H1-s0"]) > 1e-9
            except Exception:
                red = True
            check(f"mutation_RED: {label}", red)
        # spec-level mutation (the frozen check must catch it) and the wrong-band mutation
        mod = mutated("    share_map = {m: pop[\"count_by_map\"][m] / pop[\"n\"] for m in MAPS}", "    share_map = {m: 0.25 for m in MAPS}")
        try:
            mod.build_spec(tj, pop_r)
            check("mutation_RED: population shares not from the store -> the frozen check refuses", False)
        except mod.NatRefused:
            check("mutation_RED: population shares not from the store -> the frozen check refuses", True)
        mod = mutated("K2_BAND_INDEX = 3 ", "K2_BAND_INDEX = 2 ")
        try:
            wrong = mod.arm_stats(pr, src_lab)
            red = abs(mod.pooled(wrong["inter"], wrong["union"]) - N.BANKED_C3["H1-s0"]) > 1e-6
        except Exception:
            red = True
        check("mutation_RED: wrong band index (the 40-72 m band instead of [0, 40))", red)
    if SKIPPED:
        print("SKIPPED:", "; ".join(SKIPPED), flush=True)
    if FAILS:
        print(f"ZZTEST_FAIL {len(FAILS)} failed, {N_OK} passed: " + " | ".join(FAILS))
        return 1
    print(f"ZZTEST_PASS {N_OK}" + (f" (skipped: {len(SKIPPED)})" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
