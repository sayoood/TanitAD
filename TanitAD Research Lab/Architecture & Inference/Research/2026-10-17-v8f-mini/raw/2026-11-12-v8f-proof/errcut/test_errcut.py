"""test_errcut.py -- tests of the kit of card ADD96 (ERRCUT10K): mixed joins, queue, reader controls and names, with the planted faults of control e4. Synthetic data only; nothing runs on the pod.
Run: PYCAP_MB=3000 bash ops/pycap.sh -m pytest errcut/test_errcut.py -q -p no:cacheprovider --basetemp errcut/tmp_pt/<new name>"""
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_mix_joins as MMJ          # noqa: E402
import read_errcut10k as RE            # noqa: E402

QUEUE = HERE / "pod_errcut10k_queue.sh"
MODEL_QUEUE = HERE.parent / "gaprule" / "pod_gaprule10k_queue.sh"


# ------------------------------------------------------------------------------------------------ mixed joins: e0 and e1 and their planted faults
@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    base = tmp_path_factory.mktemp("mj")
    J, dumps, qmaps = MMJ.synthetic_all()
    rec = MMJ.build(J, dumps, qmaps, str(base / "clean"), meta=MMJ.synthetic_meta())
    return {"J": J, "dumps": dumps, "qmaps": qmaps, "clean": rec, "base": base}


def fault_build(mix, fault):
    return MMJ.build(mix["J"], mix["dumps"], mix["qmaps"], str(mix["base"] / ("f_" + fault)), faults=(fault,), meta=MMJ.synthetic_meta())


def test_clean_build_passes_e0_and_e1(mix):
    r = mix["clean"]
    assert r["e0"]["PASS"], r["e0"]
    assert r["e1"]["PASS"], r["e1"]
    assert len(r["arms"]) == 8 and sorted(r["arms"]) == sorted("%s_s%d" % (a, s) for a, _k in MMJ.ARMS for s in (0, 1))
    for nm, a in r["arms"].items():
        assert a["max_abs_dev_mixed_minus_label_from_k_times_gap_minus_label"] <= 1e-4 and a["other_keys_all_equal"] is True and a["n_changed_outside_mixed_frames"] == 0
        assert 0.0 <= a["fraction_without_reader_read"] <= 0.05 and a["n_with_reader_read"] > 0 and a["n_mixed_above_60"] >= 0 and a["n_mixed_below_0"] >= 0


def test_mixed_values_by_hand(mix):
    J, D, qm = mix["J"], mix["dumps"][1], mix["qmaps"][1]
    with np.load(str(mix["base"] / "clean" / "K25_s1.npz"), allow_pickle=False) as z:
        g_saved = z["lead_gap"]
        keys = set(z.files)
    assert keys == set(k for k in J if isinstance(J[k], np.ndarray)) and g_saved.dtype == np.asarray(J["lead_gap"]).dtype
    names = [str(c) for c in J["clip"]]
    # three frames by hand: dump row -> join row by clip NAME; M = L + k (min(r, Q(r)) - L)
    rows = np.flatnonzero(MMJ.reader_read_mask(D, MMJ.frame_table(J, D)) & np.isfinite(D["lab_gap"]))[[0, 50, 300]]
    for i in rows:
        ji = names.index(str(D["clips"][int(D["cid"][i])]))
        jr = int(J["off"][ji]) + int(D["frame"][i])
        L = float(np.float32(J["lead_gap"][jr]))
        r = float(D["gap"][i])
        xs, ys = qm["xs"], qm["ys"]
        q = ys[0] * r / xs[0] if r < xs[0] else (r - (xs[-1] - ys[-1]) if r > xs[-1] else float(np.interp(r, xs, ys)))
        g = min(r, q)
        assert abs(float(g_saved[jr]) - (L + 0.25 * (g - L))) < 1e-4
    vm = MMJ.valid_mask(J)
    notmixed = ~np.isin(np.arange(len(vm)), MMJ.frame_table(J, D)["row"][MMJ.reader_read_mask(D, MMJ.frame_table(J, D))])
    assert np.array_equal(g_saved[notmixed], np.asarray(J["lead_gap"])[notmixed], equal_nan=True)


def test_k_scales_the_error_linearly(mix):
    a = mix["clean"]["arms"]
    for s in (0, 1):
        e100 = a["K100_s%d" % s]["error_statistics_mixed_minus_label"]
        for nm, k in (("K50", 0.5), ("K25", 0.25), ("K12", 0.125)):
            e = a["%s_s%d" % (nm, s)]["error_statistics_mixed_minus_label"]
            for key in ("mean", "sd", "rms_about_clip_mean", "step_rms", "sd_of_clip_means"):
                assert abs(e[key] - k * e100[key]) < 1e-4 * max(1.0, abs(e100[key])), (nm, key)


def test_error_statistics_by_hand():
    ji = np.array([0] * 6 + [1] * 5 + [2] * 3)
    row = np.array([10, 11, 12, 13, 15, 16, 30, 31, 32, 33, 34, 50, 51, 52])
    err = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    s = MMJ.error_stats(err, ji, row)
    assert s["n_frames"] == 14 and abs(s["mean"] - err.mean()) < 1e-12 and s["n_frames_in_clips_ge5"] == 11          # clip 2 has 3 frames
    cm0 = 3.5
    exp_fast = np.sqrt((np.sum((err[:6] - cm0) ** 2) + 0.0) / 11)
    assert abs(s["rms_about_clip_mean"] - exp_fast) < 1e-12
    de = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])          # pairs (10,11) (11,12) (12,13) (15,16), and 30..34 four zero pairs; 13 -> 15 is not neighbouring
    assert s["n_step_pairs"] == 8 and abs(s["step_rms"] - np.sqrt(np.mean(de ** 2))) < 1e-12 and abs(s["step_sd"] - np.std(de)) < 1e-12


def test_wrong_k_turns_e1_red(mix):
    r = fault_build(mix, "wrong_k")
    assert not r["e1"]["PASS"] and any("deviation" in x for x in r["e1"]["failures"]) and r["e0"]["PASS"]


def test_raw_gap_in_place_of_mapped_gap_turns_e1_red(mix):
    r = fault_build(mix, "raw_gap")
    assert not r["e1"]["PASS"] and any("deviation" in x for x in r["e1"]["failures"])


def test_reader_seed_of_the_other_model_seed_turns_e1_red(mix):
    r = fault_build(mix, "other_seed")
    assert not r["e1"]["PASS"] and any("deviation" in x for x in r["e1"]["failures"])


def test_join_with_another_field_changed_turns_e1_red(mix):
    r = fault_build(mix, "other_field")
    assert not r["e1"]["PASS"] and any("another field" in x for x in r["e1"]["failures"])


def test_dump_label_gap_not_equal_to_the_join_turns_e0_red(mix):
    bad = dict(mix["dumps"][0])
    lg = np.array(bad["lab_gap"], copy=True)
    lg[np.flatnonzero(np.isfinite(lg))[7]] += 0.01
    bad["lab_gap"] = lg
    r = MMJ.build(mix["J"], {0: bad, 1: mix["dumps"][1]}, mix["qmaps"], str(mix["base"] / "f_label"), meta=MMJ.synthetic_meta())
    assert not r["e0"]["PASS"] and any("label gap difference" in x for x in r["e0"]["failures"])


def test_e0_red_when_the_dump_is_of_the_other_reader_seed_or_not_on_the_join(mix):
    d1 = dict(mix["dumps"][1])
    d1["seed"] = np.int64(0)
    r = MMJ.build(mix["J"], {0: mix["dumps"][0], 1: d1}, mix["qmaps"], str(mix["base"] / "f_dumpseed"), meta=MMJ.synthetic_meta())
    assert not r["e0"]["PASS"] and any("records seed" in x for x in r["e0"]["failures"])
    d0 = dict(mix["dumps"][0])
    d0["clip_index"] = (np.asarray(d0["clip_index"]) + 1) % len(mix["J"]["n"])
    r = MMJ.build(mix["J"], {0: d0, 1: mix["dumps"][1]}, mix["qmaps"], str(mix["base"] / "f_clipidx"), meta=MMJ.synthetic_meta())
    assert not r["e0"]["PASS"] and any("n_clip_index_mismatch" in x for x in r["e0"]["failures"])


def test_e1_limits_and_record_fields(mix):
    c = mix["clean"]

    def with_(nm, **kw):
        r = copy.deepcopy(c)
        r["arms"][nm].update(kw)
        return r
    assert MMJ.e1_check(with_("K50_s0", fraction_without_reader_read=0.05))["PASS"] is True
    assert MMJ.e1_check(with_("K50_s0", fraction_without_reader_read=0.0501))["PASS"] is False
    assert MMJ.e1_check(with_("K25_s1", min_mixed=-0.01, n_mixed_below_0=3))["PASS"] is True          # note N1: a value below 0 is counted, no bar
    assert MMJ.e1_check(with_("K25_s1", n_mixed_below_0=None))["PASS"] is False
    assert MMJ.e1_check(with_("K12_s0", n_changed_outside_mixed_frames=1))["PASS"] is False
    assert MMJ.e1_check(with_("K100_s1", k=0.9))["PASS"] is False
    r = copy.deepcopy(c)
    r["seeds"]["1"]["qmap_file"] = "QMAP_LR16_s0_q50.json"
    assert MMJ.e1_check(r)["PASS"] is False
    assert MMJ.e1_check(c, expect_qmap_dir="/workspace/v8proof/integ10k/qmap_sweep")["PASS"] is True
    assert MMJ.e1_check(c, expect_qmap_dir="/somewhere/else")["PASS"] is False
    r = copy.deepcopy(c)
    r["arms"].pop("K12_s1")
    assert MMJ.e1_check(r)["PASS"] is False


def test_e0_record_fields(mix):
    r = copy.deepcopy(mix["clean"])
    r["seeds"]["0"]["e0"]["max_abs_diff"] = 1.1e-4
    assert MMJ.e0_check(r)["PASS"] is False
    r["seeds"]["0"]["e0"]["max_abs_diff"] = 1e-4
    assert MMJ.e0_check(r)["PASS"] is True
    r["seeds"]["0"]["e0"]["n_duplicate_keys"] = 1
    assert MMJ.e0_check(r)["PASS"] is False


def test_out_dir_refused_and_arguments_required(mix):
    with pytest.raises(SystemExit) as ex:
        MMJ.build(mix["J"], mix["dumps"], mix["qmaps"], str(mix["base"] / "clean"))
    assert "REFUSED" in str(ex.value)
    with pytest.raises(SystemExit) as ex:
        MMJ.main(["--eval-join", "x.npz", "--out-dir", str(mix["base"] / "o")])
    assert "REFUSED" in str(ex.value)


def test_qmap_apply_equals_closeread_fit_on_random_input():
    sys.path.insert(0, str(HERE.parent / "closeread"))
    import closeread_fit as CF            # noqa: E402  (imported here only)
    rng = np.random.default_rng(7)
    for t in range(5):
        xs = np.sort(rng.uniform(1.0, 60.0, 6 + t))
        ys = np.sort(rng.uniform(0.5, 55.0, 6 + t))
        g = rng.uniform(-5.0, 90.0, 4000)
        g[rng.random(4000) < 0.05] = np.nan
        a, b = MMJ.qmap_apply(g, xs, ys), CF.qmap_apply(g, xs, ys)
        assert np.array_equal(a, b, equal_nan=True)


def test_dump_field_semantics_cid_indexes_the_dumps_clips_and_clip_index_the_join(mix):
    D, J = mix["dumps"][0], mix["J"]
    assert [str(c) for c in D["clips"]] != [str(c) for c in J["clip"]]          # the synthetic dump order differs from the join order
    names = [str(c) for c in J["clip"]]
    ok = all(names.index(str(D["clips"][int(c)])) == int(ci) for c, ci in zip(D["cid"][::97], D["clip_index"][::97]))
    assert ok


def test_main_through_files_and_selftest(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    dd, qd = tmp_path / "dump", tmp_path / "qmap"
    dd.mkdir()
    qd.mkdir()
    np.savez(str(tmp_path / "join.npz"), **J)
    for s in (0, 1):
        np.savez(str(dd / ("reader_s%d_score.npz" % s)), **dumps[s])
        (qd / ("QMAP_LR16_s%d_q50.json" % s)).write_text(json.dumps({"xs": qmaps[s]["xs"].tolist(), "ys": qmaps[s]["ys"].tolist()}), encoding="utf-8")
    assert MMJ.main(["--eval-join", str(tmp_path / "join.npz"), "--dump-dir", str(dd), "--qmap-dir", str(qd), "--out-dir", str(tmp_path / "out")]) == 0
    rec = json.loads((tmp_path / "out" / "MIX_JOINS_RECORD.json").read_text(encoding="utf-8"))
    assert rec["e0"]["PASS"] and rec["e1"]["PASS"] and rec["eval_join_sha256"] == MMJ.sha_file(str(tmp_path / "join.npz"))
    assert rec["seeds"]["0"]["qmap_sha256"] == MMJ.sha_file(str(qd / "QMAP_LR16_s0_q50.json")) and rec["seeds"]["1"]["dump_sha256"] == MMJ.sha_file(str(dd / "reader_s1_score.npz"))
    ok = RE.ctl_e1(rec, str(tmp_path / "out"), rec["eval_join_sha256"])
    assert ok["PASS"], ok
    assert RE.ctl_e0(rec, rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_e1(rec, str(tmp_path / "out"), "0" * 64)["PASS"]
    assert not RE.ctl_e1(rec, str(tmp_path / "nowhere"), rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_e0(rec, "0" * 64)["PASS"]


# ------------------------------------------------------------------------------------------------ queue (text checks; DRY=1 needs pod paths for the file checks it skips, so the function text is compared too)
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def queue_static_facts(t):
    """static facts of the queue text: the scoring join is the evaluation join, unchanged; a rule join appears only after --rule-join; -> list of (fact, ok)"""
    f = fn_text(t)
    outside = t.replace(f, "")
    facts = [("scoring gets --eval-join \"$JOINE\" (and the DRY print --eval-join $JOINE)", f.count('--eval-join "$JOINE"') == 1 and f.count("--eval-join $JOINE") == 1),
             ("no other --eval-join anywhere", t.count("--eval-join") == 2 + len(re.findall(r"#.*--eval-join", t))),
             ("JOINE is assigned once, to the pinned evaluation join", len(re.findall(r"^JOINE=", t, re.M)) == 1 and "JOINE=$V/hierrep/pins/join_eval141.npz" in t),
             ("--rule-join once, as extra+=(--rule-join \"$RJOIN\")", f.count("--rule-join") == 1 and 'extra+=(--rule-join "$RJOIN")' in f),
             ("the rule join is named outside the rollout function only in comments or as RJOIN= before a rollout call",
              all(x.lstrip().startswith("#") or ("RJOIN=" in x and " rollout " in x) or "$PJ" not in x or "PJ=" in x or "admit" in x or "NO-MIX" in x for x in outside.split("\n") if "RJOIN" in x or "$PJ" in x)),
             ("no variable of the rule join in the scoring arguments", "$RJOIN" not in f.split("--eval-join")[1].split("\n")[0] and "$PJ" not in f)]
    return facts


def test_rollout_function_is_byte_identical_to_the_model_queue():
    a, b = fn_text(QUEUE.read_text(encoding="utf-8")), fn_text(MODEL_QUEUE.read_text(encoding="utf-8"))
    assert a == b and len(a) > 3000


def test_queue_edits_and_names():
    t = QUEUE.read_text(encoding="utf-8")
    assert "GP=${GR_ROOT:-/workspace/v8proof/errcut10k}" in t and "CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}" in t
    assert "START_ERRCUT10K$PS.txt" in t and "ERRCUT10K_DONE$PS.txt" in t and "ERRCUT10K_STOP$PS.txt" in t and "queue_errcut$PS.log" in t and 'PS="_s$SEEDSW"' in t
    assert "cbf7063d867c9f4e" in t and t.count("cbf7063d867c9f4e") >= 3 and "MIX_JOINS_RECORD.json" in t
    assert "GAPRULE" not in t and "planted" not in t.lower()
    for arm in ("K100", "K50", "K25", "K12"):
        assert ('RJOIN="$PJ/%s_s$k.npz" rollout I10RLcKF5P30Hb20m%s RLc "$k" "$k"' % (arm, arm)) in t
    assert t.count("rollout I10RLcKF5P30Hb20m") == 4
    adm = t[t.index("admit() {"):t.index("\n}\n", t.index("admit() {"))]
    assert 'rec[c].get("PASS") is not True' in adm and "eval_join_sha256" in adm and "sha256" in adm and '"e0", "e1"' in adm
    assert "28" in t and " rm " not in t and " mv " not in t and " kill " not in t


def test_queue_static_facts_hold_and_each_planted_fault_turns_one_red():
    t = QUEUE.read_text(encoding="utf-8")
    assert all(ok for _f, ok in queue_static_facts(t)), [f for f, ok in queue_static_facts(t) if not ok]
    bad1 = t.replace('--eval-join "$JOINE"', '--eval-join "${RJOIN:-$JOINE}"')
    bad2 = t.replace("JOINE=$V/hierrep/pins/join_eval141.npz", "JOINE=${RJOIN:-$V/hierrep/pins/join_eval141.npz}")
    bad3 = t.replace('extra+=(--rule-join "$RJOIN")', 'extra+=(--rule-join "$RJOIN" --eval-join "$RJOIN")')
    for b in (bad1, bad2, bad3):
        assert b != t and any(not ok for _f, ok in queue_static_facts(b))


def bash_exe():
    return next((c for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe") if os.path.isfile(c)), "bash")


def dry_commands(q):
    out = subprocess.run([bash_exe(), str(q).replace("\\", "/")], env=dict(os.environ, DRY="1"), capture_output=True, text=True, encoding="utf-8").stdout
    return [x for x in out.splitlines() if x.startswith("CMD:")]


def test_queue_dry_run_has_eight_commands_equal_to_the_model_dry_after_arm_and_join():
    q = str(QUEUE).replace("\\", "/")
    assert subprocess.run([bash_exe(), "-n", q]).returncode == 0
    mine, model = dry_commands(QUEUE), dry_commands(MODEL_QUEUE)

    def norm(c):
        c = re.sub(r"--rule-join \S+ ", "", c)
        return re.sub(r"I10RLcKF5P30Hb20[jm](G0|K100|K50|K25|K12)", "RUN", c)
    assert len(mine) == 8 and len(model) == 12
    assert all("--rule-join" in c and "rzgfhqetj.py" in c and "--form RLc" in c and "--kappa-max 0.2" in c and "--preroll-steps 30" in c for c in mine)
    assert all("--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c for c in mine)
    g0 = [norm(c) for c in model if "jG0" in c]
    assert len(g0) == 2
    for arm in ("K100", "K50", "K25", "K12"):
        for s in (0, 1):
            cs = [c for c in mine if ("m%s_s%d " % (arm, s)) in c or ("m%s_s%d/" % (arm, s)) in c]
            assert cs and all(("/%s_s%d.npz" % (arm, s)) in c for c in cs)
            assert all(norm(c) in g0 for c in cs)


# ------------------------------------------------------------------------------------------------ reader: names, controls, binding
def iv(lo, hi):
    return [lo, hi]


def test_name_rules_on_planted_intervals():
    n = RE.names_of("K100", iv(-0.05, -0.01), iv(0.0, 0.004), iv(-0.02, 0.03), 0.0954, 0.0)         # H = 0.0477
    assert n["K100_CONTACT"].endswith("ERRCUT10K-K100-CUTS-CONTACT") and n["K100_HALF"].endswith("-K100-WITHIN-HALF-THE-GAP") and n["K100_PRICE"].endswith("-K100-FALLING-BEHIND-NOT-SEPARATED")
    assert RE.names_of("K50", iv(-0.01, 0.0), iv(0.02, 0.0477), iv(0.0, 0.1), 0.0954, 0.0)["K50_HALF"].endswith("WITHIN-HALF-THE-GAP")      # at H: within
    assert RE.names_of("K50", iv(-0.01, 0.0), iv(0.0478, 0.06), iv(0.0, 0.1), 0.0954, 0.0)["K50_HALF"].endswith("OUTSIDE-HALF-THE-GAP")
    assert RE.names_of("K25", iv(-0.01, 0.0), iv(0.02, 0.05), iv(0.0, 0.1), 0.0954, 0.0)["K25_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED")
    assert RE.names_of("K12", iv(0.0, 0.01), iv(0.02, 0.05), iv(-0.1, -0.001), 0.0954, 0.0)["K12_CONTACT"].endswith("CONTACT-NOT-SHOWN")
    assert RE.names_of("K12", iv(0.0, 0.01), iv(0.02, 0.05), iv(-0.1, -0.001), 0.0954, 0.0)["K12_PRICE"].endswith("FALLS-BEHIND-LESS")
    assert RE.names_of("K12", iv(0.0, 0.01), iv(0.02, 0.05), iv(0.001, 0.1), 0.0954, 0.0)["K12_PRICE"].endswith("FALLS-BEHIND-MORE")


def halves(*flags):
    return {a + "_HALF": RE.PRE + "-" + a + ("-WITHIN-HALF-THE-GAP" if f else "-OUTSIDE-HALF-THE-GAP") for a, f in zip(RE.ARMS, flags)}


def test_ladder_names_three_cases_and_not_monotone():
    assert RE.ladder_name(halves(1, 1, 1, 1)) == "ERRCUT10K-CUT-BY-1-SUFFICES"
    assert RE.ladder_name(halves(0, 1, 1, 1)) == "ERRCUT10K-CUT-BY-2-SUFFICES"
    assert RE.ladder_name(halves(0, 0, 1, 1)) == "ERRCUT10K-CUT-BY-4-SUFFICES"
    assert RE.ladder_name(halves(0, 0, 0, 1)) == "ERRCUT10K-CUT-BY-8-SUFFICES"
    assert RE.ladder_name(halves(0, 0, 0, 0)) == "ERRCUT10K-CUT-BY-8-NOT-SHOWN-TO-SUFFICE"
    for fl in ((1, 0, 1, 1), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 1), (0, 0, 1, 0), (1, 1, 1, 0)):
        assert RE.ladder_name(halves(*fl)) == "ERRCUT10K-LADDER-NOT-MONOTONE", fl
    for fl in __import__("itertools").product((0, 1), repeat=4):
        assert RE.ladder_check(halves(*fl), RE.ladder_name(halves(*fl)))["PASS"]


def test_a_ladder_name_from_a_ladder_that_is_not_monotone_turns_the_check_red():
    h = halves(1, 0, 1, 1)
    assert RE.ladder_check(h, "ERRCUT10K-CUT-BY-1-SUFFICES")["PASS"] is False
    assert RE.ladder_check(h, "ERRCUT10K-CUT-BY-4-SUFFICES")["PASS"] is False
    assert RE.ladder_check(h, "ERRCUT10K-CUT-BY-8-NOT-SHOWN-TO-SUFFICE")["PASS"] is False
    assert RE.ladder_check(h, "ERRCUT10K-LADDER-NOT-MONOTONE")["PASS"] is True


def payload(rr=0.0954, rlc=0.0, within_arms=("K100", "K50", "K25", "K12")):
    pr = {}
    for a in RE.ARMS:
        hi = 0.004 if a in within_arms else 0.06
        pr[a + "-VS-RR"] = {k: {"interval": iv(-0.05, -0.01), "pooled": -0.03} for k in ("C", "U", "F")}
        pr[a + "-VS-RLc"] = {k: {"interval": iv(0.0 if a in within_arms else 0.055, hi), "pooled": 0.002} for k in ("C", "U", "F")}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}}, "pairs": pr, "families": {}, "strategic": {"pairs": {}}}


def test_thirteen_names_and_the_ladder_in_them():
    n = RE.all_names(payload())
    assert len(n) == 13 and n["LADDER"] == "ERRCUT10K-CUT-BY-1-SUFFICES"
    assert RE.all_names(payload(within_arms=("K25", "K12")))["LADDER"] == "ERRCUT10K-CUT-BY-4-SUFFICES"
    assert RE.all_names(payload(within_arms=("K100", "K12")))["LADDER"] == "ERRCUT10K-LADDER-NOT-MONOTONE"
    assert RE.all_names(payload(within_arms=()))["LADDER"] == "ERRCUT10K-CUT-BY-8-NOT-SHOWN-TO-SUFFICE"


def test_controls_e2_e3_e4_e5():
    assert RE.ctl_e2("x\nZZ CHECK-WRAPPER-J PASS\n", "cbf7063d867c9f4e a.py", "cbf7063d867c9f4e")["PASS"]
    assert not RE.ctl_e2("ZZ CHECK-WRAPPER-J FAIL J2A\n", "cbf7063d867c9f4e", "cbf7063d867c9f4e")["PASS"]
    assert not RE.ctl_e2("ZZ CHECK-WRAPPER-J PASS\n", "0123456789abcdef", "cbf7063d867c9f4e")["PASS"]
    assert RE.ctl_e3(0.09541, 0.0)["PASS"] and RE.ctl_e3(0.09536, 0.00004)["PASS"]
    assert not RE.ctl_e3(0.0960, 0.0)["PASS"] and not RE.ctl_e3(0.0954, 0.0138)["PASS"] and not RE.ctl_e3(None, 0.0)["PASS"]
    assert RE.ctl_e4("25 passed in 3s")["PASS"] and not RE.ctl_e4("2 failed, 3 passed")["PASS"] and not RE.ctl_e4("")["PASS"]
    assert RE.ctl_e5(True, True, True)["PASS"] and not RE.ctl_e5(False, True, True)["PASS"] and not RE.ctl_e5(True, False, True)["PASS"] and not RE.ctl_e5(True, True, False)["PASS"]


ALL_PASS = {c: {"PASS": True} for c in ("e0", "e1", "e2", "e3", "e4", "e5")}


def test_name_written_with_a_control_failed_turns_red(tmp_path):
    ctl = RE.assemble_controls(dict(ALL_PASS, e1={"PASS": False}), payload())
    assert ctl["controls_all_pass"] is False and "payload" not in ctl and "withheld" in ctl
    f = tmp_path / "controls.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "result.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 3 and not out.exists()
    f2 = tmp_path / "forged.json"
    f2.write_text(json.dumps(dict(ctl, controls_all_pass=True, payload=payload())), encoding="utf-8")        # an edited flag does not open the names: the controls themselves are re-checked
    assert RE.main(["--stage", "names", "--controls", str(f2), "--out", str(tmp_path / "r2.json")]) == 3
    f3 = tmp_path / "missing.json"
    short = {k: v for k, v in ALL_PASS.items() if k != "e5"}                                                   # a controls file without one control does not open the names
    f3.write_text(json.dumps(dict(RE.assemble_controls(short, payload()))), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(f3), "--out", str(tmp_path / "r3.json")]) == 3 and not (tmp_path / "r3.json").exists()


def test_names_stage_binds_the_controls_file(tmp_path):
    ctl = RE.assemble_controls(ALL_PASS, payload())
    f = tmp_path / "c.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "r.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["controls_sha256"] == RE.sha_file(str(f)) and len(r["names"]) == 13 and r["card"] == "PREREG_V8F_PROOF_ADD96"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 2             # refuses an existing output


def test_reader_source_carries_the_strategic_family_and_the_engine_bootstrap_setting():
    t = (HERE / "read_errcut10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and "R.strategic_grid(" in t and "S.pair_read(" in t and "R.family_of(" in t
    assert 'CARD, PRE = "PREREG_V8F_PROOF_ADD96", "ERRCUT10K"' in t and '"RLcKF5P30Hb20m" + a' in t and 'ARMS = ("K100", "K50", "K25", "K12")' in t
    assert 'controls = {"e0"' in t and '"e5": ctl_e5(' in t


def test_reader_selftest_runs():
    assert RE.selftest() == 0


def test_podread_text():
    t = (HERE / "podread_errcut10k.sh").read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=-1" in t and "GP=$V/errcut10k" in t and "OUT=$GP/read" in t and "CHK=$V/gaprule10kb/check_j2" in t
    assert "ERRCUT10K_CONTROLS.json" in t and "ERRCUT10K_RESULT.json" in t and "ERRCUT10K_DONE_s0.txt" in t and "ERRCUT10K_DONE_s1.txt" in t and "lead_pytest.out" in t and "MIX_JOINS_RECORD.json" in t
    assert "I10RLcKF5P30Hb20m${x}_s$s" in t and "read_errcut10k.py" in t
    assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_errcut10k.sh").replace("\\", "/")]).returncode == 0


def test_texts_hold_no_banned_words():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    n = 0
    for f in ("make_mix_joins.py", "make_errcut_queue.py", "pod_errcut10k_queue.sh", "read_errcut10k.py", "podread_errcut10k.sh", "test_errcut.py", "ERRCUT_DEFS.md"):
        p = HERE / f
        if p.exists():
            n += 1
            t = p.read_text(encoding="utf-8").lower()
            hit = [w for w in bad if w in t]
            assert not hit, (f, hit)
    assert n >= 6


def test_note_n1_a_negative_reader_gap_is_counted_and_left(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    meta = MMJ.synthetic_meta()
    d0 = dict(dumps[0])
    g = np.array(d0["gap"], np.float64)
    ok = np.flatnonzero(np.asarray(d0["valid"], bool) & np.isfinite(g) & np.asarray(d0["lab_present"], bool))
    g[ok[:7]] = -4.0
    d0["gap"] = g
    rec = MMJ.build(J, {0: d0, 1: dumps[1]}, qmaps, str(tmp_path / "joins"), meta=meta)
    a = rec["arms"]["K100_s0"]
    assert a["n_mixed_below_0"] >= 7 and a["min_mixed"] <= -4.0 + 1e-6
    assert rec["e1"]["PASS"] is True
