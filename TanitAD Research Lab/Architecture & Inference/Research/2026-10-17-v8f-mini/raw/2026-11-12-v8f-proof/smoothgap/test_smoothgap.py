"""test_smoothgap.py -- tests of the kit of card ADD97 (SMOOTHGAP10K): processed joins, queue, reader controls and names, with the planted faults of control p4. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest smoothgap/test_smoothgap.py -q -p no:cacheprovider --basetemp smoothgap/tmp_pt/<new name>"""
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
import make_smooth_joins as MMJ          # noqa: E402
import read_smoothgap10k as RE            # noqa: E402

QUEUE = HERE / "pod_smoothgap10k_queue.sh"
ERR_QUEUE = HERE.parent / "errcut" / "pod_errcut10k_queue.sh"
MODEL_QUEUE = HERE.parent / "gaprule" / "pod_gaprule10k_queue.sh"
ARMS = ("E", "S50", "S25", "ES25")


# ------------------------------------------------------------------------------------------------ processed joins: p0 and p1 and their planted faults
@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    base = tmp_path_factory.mktemp("sg")
    J, dumps, qmaps = MMJ.synthetic_all()
    rec = MMJ.build(J, dumps, qmaps, str(base / "clean"), meta=MMJ.synthetic_meta())
    return {"J": J, "dumps": dumps, "qmaps": qmaps, "clean": rec, "base": base}


def fault_build(mix, fault):
    cache = mix.setdefault("fault_cache", {})
    if fault not in cache:
        cache[fault] = _fault_build(mix, fault)
    return cache[fault]


def _fault_build(mix, fault):
    return MMJ.build(mix["J"], mix["dumps"], mix["qmaps"], str(mix["base"] / ("f_" + fault)), faults=(fault,), meta=MMJ.synthetic_meta())


def test_clean_build_passes_p0_and_p1(mix):
    r = mix["clean"]
    assert r["p0"]["PASS"], r["p0"]
    assert r["p1"]["PASS"], r["p1"]
    assert len(r["arms"]) == 8 and sorted(r["arms"]) == sorted("%s_s%d" % (a, s) for a in ARMS for s in (0, 1)) and r["card"] == "PREREG_V8F_PROOF_ADD97"
    for nm, a in r["arms"].items():
        assert a["max_abs_dev_saved_minus_processed_second_routine"] <= 1e-4 and a["other_keys_all_equal"] is True and a["n_changed_outside_mixed_frames"] == 0
        assert 0.0 <= a["fraction_without_reader_read"] <= 0.05 and a["n_with_reader_read"] > 0 and a["n_mixed_above_60"] >= 0 and a["n_mixed_below_0"] >= 0
        assert a["causal_max_abs_change_earlier_frames"] == 0.0 and a["causal_n_earlier_frames_checked"] > 0 and a["n_label_frames_without_read_not_label_gap"] == 0
        assert a["error_statistics_processed_minus_label"]["n_frames"] == a["n_with_reader_read"]


def test_E_and_ES25_of_the_two_model_seeds_are_equal_in_every_array(mix):
    for arm in ("E", "ES25"):
        with np.load(str(mix["base"] / "clean" / (arm + "_s0.npz")), allow_pickle=False) as z0, np.load(str(mix["base"] / "clean" / (arm + "_s1.npz")), allow_pickle=False) as z1:
            assert sorted(z0.files) == sorted(z1.files)
            assert all(MMJ.equal_key(z0[k], z1[k]) for k in z0.files)
        assert mix["clean"]["pairs_equal"][arm] is True and mix["clean"]["arms"][arm + "_s0"]["sha256"] == mix["clean"]["arms"][arm + "_s1"]["sha256"]
    with np.load(str(mix["base"] / "clean" / "S50_s0.npz"), allow_pickle=False) as a, np.load(str(mix["base"] / "clean" / "S50_s1.npz"), allow_pickle=False) as b:
        assert not MMJ.equal_key(a["lead_gap"], b["lead_gap"])          # the arms with a reader seed of their own differ between model seeds
    assert len({mix["clean"]["arms"][n]["sha256"] for n in mix["clean"]["arms"]}) == 6


def dump_values(mix, s):
    """g of every (clip name, frame) with a reader read, by hand: min(r, Q(r)) with the plain formula of the map"""
    D, qm = mix["dumps"][s], mix["qmaps"][s]
    xs, ys = qm["xs"], qm["ys"]
    out = {}
    for i in range(len(D["frame"])):
        r = float(D["gap"][i])
        if not (bool(D["valid"][i]) and np.isfinite(r)):
            continue
        q = ys[0] * r / xs[0] if r < xs[0] else (r - (xs[-1] - ys[-1]) if r > xs[-1] else float(np.interp(r, xs, ys)))
        out[(str(D["clips"][int(D["cid"][i])]), int(D["frame"][i]))] = min(r, q)
    return out


def first_run(mix, seeds, min_len=4):
    """(clip name, [frames]) of a run of neighbouring mix frames of at least min_len frames, found by hand"""
    J = mix["J"]
    vals = [dump_values(mix, s) for s in seeds]
    vm = MMJ.valid_mask(J)
    for ci, c in enumerate(str(x) for x in J["clip"]):
        run = []
        for f in range(int(J["n"][ci])):
            if vm[int(J["off"][ci]) + f] and all((c, f) in v for v in vals):
                run.append(f)
            else:
                if len(run) >= min_len:
                    return c, run
                run = []
        if len(run) >= min_len:
            return c, run
    raise AssertionError("no run")


def saved_gap(mix, name):
    with np.load(str(mix["base"] / "clean" / (name + ".npz")), allow_pickle=False) as z:
        return z["lead_gap"]


def row_of(J, c, f):
    names = [str(x) for x in J["clip"]]
    return int(J["off"][names.index(c)]) + f


def test_smoothed_values_by_hand(mix):
    J = mix["J"]
    c, fr = first_run(mix, (1,))
    v1 = dump_values(mix, 1)
    for nm, a in (("S50_s1", 0.5), ("S25_s1", 0.25)):
        g = saved_gap(mix, nm)
        P = None
        for f in fr[:4]:
            x = v1[(c, f)]
            P = x if P is None else a * x + (1 - a) * P
            assert abs(float(g[row_of(J, c, f)]) - P) < 1e-4, (nm, f)
    c, fr = first_run(mix, (0, 1))
    v0, v1 = dump_values(mix, 0), dump_values(mix, 1)
    gE, gES = saved_gap(mix, "E_s0"), saved_gap(mix, "ES25_s1")
    P = None
    for f in fr[:4]:
        m = 0.5 * (v0[(c, f)] + v1[(c, f)])
        P = m if P is None else 0.25 * m + 0.75 * P
        assert abs(float(gE[row_of(J, c, f)]) - m) < 1e-4
        assert abs(float(gES[row_of(J, c, f)]) - P) < 1e-4


def test_first_frame_of_a_run_carries_the_gap_itself_and_a_run_ends_at_a_frame_with_no_read(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    tab = MMJ.frame_table(J, dumps[1])
    g, rd = MMJ.reader_rows(J, dumps[1], tab, qmaps[1])
    vm, clip = MMJ.valid_mask(J), MMJ.clip_of_row(J)
    mixm = rd & vm
    idx, start, length = MMJ.run_structure(mixm, clip)
    k = int(np.flatnonzero(length >= 6)[0])
    r0, ln = int(idx[start[k]]), int(length[k])
    cut = r0 + 3                                                    # remove the read at the 4th frame of that run
    d1 = dict(dumps[1])
    ok = np.array(d1["valid"], copy=True)
    sel = np.flatnonzero((tab["row"] == cut))
    assert len(sel) == 1
    ok[sel] = False
    d1["valid"] = ok
    rec = MMJ.build(J, {0: dumps[0], 1: d1}, qmaps, str(tmp_path / "j"), meta=MMJ.synthetic_meta())
    assert rec["p1"]["PASS"], rec["p1"]
    with np.load(str(tmp_path / "j" / "S25_s1.npz"), allow_pickle=False) as z:
        sv = np.asarray(z["lead_gap"], np.float64)
    g1 = MMJ.reader_rows(J, d1, MMJ.frame_table(J, d1), qmaps[1])[0]
    assert abs(sv[r0] - g1[r0]) < 1e-4
    assert abs(sv[cut] - float(np.float32(J["lead_gap"][cut]))) < 1e-9                       # no read: the label gap
    if ln > 4 and rd[cut + 1] and vm[cut + 1]:
        assert abs(sv[cut + 1] - g1[cut + 1]) < 1e-4                                           # restart: the first frame of the new run is the gap itself
        assert abs(sv[cut + 1] - (0.25 * g1[cut + 1] + 0.75 * sv[cut - 1])) > 1e-9


def test_smoothing_function_by_hand_on_three_runs():
    g = np.array([10.0, 20.0, 30.0, 99.0, 5.0, 7.0, 1.0, 2.0, 3.0, 4.0])
    mixm = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], bool)
    clip = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])               # rows 5 and 6 are neighbours but lie in two clips
    P = MMJ.smooth_runs(g, mixm, clip, 0.5)
    exp = [10.0, 15.0, 22.5, np.nan, 5.0, 6.0, 1.0, 1.5, 2.25, 3.125]
    assert np.allclose(P, exp, equal_nan=True)
    P2 = MMJ.smooth_runs(g, mixm, clip, 0.25)
    assert abs(P2[2] - (0.25 * 30 + 0.75 * (0.25 * 20 + 0.75 * 10))) < 1e-12 and P2[4] == 5.0 and P2[6] == 1.0
    assert not np.allclose(MMJ.smooth_runs(g, mixm, clip, 0.5, across=True), exp, equal_nan=True)       # the fault: running on across the gap


def test_each_planted_fault_of_control_p4_turns_p1_red(mix):
    for fault, word in (("centred", "changes by"), ("wrong_a", "second time"), ("mean_twice", "second time"), ("run_across", None), ("other_field", "another field"), ("other_seed", "second time")):
        r = fault_build(mix, fault)
        assert not r["p1"]["PASS"], fault
        if word:
            assert any(word in x for x in r["p1"]["failures"]), (fault, r["p1"]["failures"][:2])
        assert r["p0"]["PASS"]
    r = fault_build(mix, "run_across")
    assert any("second time" in x for x in r["p1"]["failures"])
    r = fault_build(mix, "centred")
    assert any("changes by" in x for x in r["p1"]["failures"]) and any("second time" in x for x in r["p1"]["failures"])


def test_a_smoothing_that_uses_a_later_frame_is_seen_by_the_causality_probe_alone(mix):
    """the probe is a check of its own: it is red for the centred average even when the value check is looked at separately"""
    r = fault_build(mix, "centred")
    for nm, a in r["arms"].items():
        if a["a"] is not None:
            assert a["causal_max_abs_change_earlier_frames"] > 0.0, nm
    assert all(r["arms"][n]["causal_max_abs_change_earlier_frames"] == 0.0 for n in ("E_s0", "E_s1"))


def test_dump_label_gap_not_equal_to_the_join_turns_p0_red(mix):
    bad = dict(mix["dumps"][0])
    lg = np.array(bad["lab_gap"], copy=True)
    lg[np.flatnonzero(np.isfinite(lg))[7]] += 0.01
    bad["lab_gap"] = lg
    r = MMJ.build(mix["J"], {0: bad, 1: mix["dumps"][1]}, mix["qmaps"], str(mix["base"] / "f_label"), meta=MMJ.synthetic_meta())
    assert not r["p0"]["PASS"] and any("label gap difference" in x for x in r["p0"]["failures"])


def test_p0_red_when_the_dump_is_of_the_other_reader_seed_or_not_on_the_join(mix):
    d1 = dict(mix["dumps"][1])
    d1["seed"] = np.int64(0)
    r = MMJ.build(mix["J"], {0: mix["dumps"][0], 1: d1}, mix["qmaps"], str(mix["base"] / "f_dumpseed"), meta=MMJ.synthetic_meta())
    assert not r["p0"]["PASS"] and any("records seed" in x for x in r["p0"]["failures"])
    d0 = dict(mix["dumps"][0])
    d0["clip_index"] = (np.asarray(d0["clip_index"]) + 1) % len(mix["J"]["n"])
    r = MMJ.build(mix["J"], {0: d0, 1: mix["dumps"][1]}, mix["qmaps"], str(mix["base"] / "f_clipidx"), meta=MMJ.synthetic_meta())
    assert not r["p0"]["PASS"] and any("n_clip_index_mismatch" in x for x in r["p0"]["failures"])


def test_p1_limits_and_record_fields(mix):
    c = mix["clean"]

    def with_(nm, **kw):
        r = copy.deepcopy(c)
        r["arms"][nm].update(kw)
        return r
    assert MMJ.p1_check(with_("S50_s0", fraction_without_reader_read=0.05))["PASS"] is True
    assert MMJ.p1_check(with_("S50_s0", fraction_without_reader_read=0.0501))["PASS"] is False
    assert MMJ.p1_check(with_("E_s1", fraction_without_reader_read=0.0501))["PASS"] is False
    assert MMJ.p1_check(with_("S25_s1", min_mixed=-0.01, n_mixed_below_0=3))["PASS"] is True          # counted, no bar
    assert MMJ.p1_check(with_("S25_s1", n_mixed_below_0=None))["PASS"] is False
    assert MMJ.p1_check(with_("S25_s1", n_mixed_above_60=None))["PASS"] is False
    assert MMJ.p1_check(with_("ES25_s0", n_changed_outside_mixed_frames=1))["PASS"] is False
    assert MMJ.p1_check(with_("ES25_s0", n_label_frames_without_read_not_label_gap=2))["PASS"] is False
    assert MMJ.p1_check(with_("S50_s1", a=0.4))["PASS"] is False
    assert MMJ.p1_check(with_("S50_s1", max_abs_dev_saved_minus_processed_second_routine=1.1e-4))["PASS"] is False
    assert MMJ.p1_check(with_("S50_s1", max_abs_dev_saved_minus_processed_second_routine=1e-4))["PASS"] is True
    assert MMJ.p1_check(with_("S50_s1", causal_max_abs_change_earlier_frames=1e-12))["PASS"] is False
    assert MMJ.p1_check(with_("S50_s1", causal_n_runs=0))["PASS"] is False
    r = copy.deepcopy(c)
    r["seeds"]["1"]["qmap_file"] = "QMAP_LR16_s0_q50.json"
    assert MMJ.p1_check(r)["PASS"] is False
    assert MMJ.p1_check(c, expect_qmap_dir="/workspace/v8proof/integ10k/qmap_sweep")["PASS"] is True
    assert MMJ.p1_check(c, expect_qmap_dir="/somewhere/else")["PASS"] is False
    r = copy.deepcopy(c)
    r["arms"].pop("ES25_s1")
    assert MMJ.p1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["pairs_equal"]["E"] = False
    assert MMJ.p1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["card"] = "PREREG_V8F_PROOF_ADD96"
    assert MMJ.p1_check(r)["PASS"] is False


def test_p0_record_fields(mix):
    r = copy.deepcopy(mix["clean"])
    r["seeds"]["0"]["e0"]["max_abs_diff"] = 1.1e-4
    assert MMJ.p0_check(r)["PASS"] is False
    r["seeds"]["0"]["e0"]["max_abs_diff"] = 1e-4
    assert MMJ.p0_check(r)["PASS"] is True
    r["seeds"]["0"]["e0"]["n_duplicate_keys"] = 1
    assert MMJ.p0_check(r)["PASS"] is False


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
        for v in g[:50]:
            if np.isfinite(v):
                assert abs(MMJ.qmap_scalar(float(v), list(xs), list(ys)) - MMJ.qmap_apply(np.array([v]), xs, ys)[0]) < 1e-12


def test_main_through_files(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    dd, qd = tmp_path / "dump", tmp_path / "qmap"
    dd.mkdir()
    qd.mkdir()
    np.savez(str(tmp_path / "join.npz"), **J)
    for s in (0, 1):
        np.savez(str(dd / ("reader_s%d_score.npz" % s)), **dumps[s])
        (qd / ("QMAP_LR16_s%d_q50.json" % s)).write_text(json.dumps({"xs": qmaps[s]["xs"].tolist(), "ys": qmaps[s]["ys"].tolist()}), encoding="utf-8")
    assert MMJ.main(["--eval-join", str(tmp_path / "join.npz"), "--dump-dir", str(dd), "--qmap-dir", str(qd), "--out-dir", str(tmp_path / "out")]) == 0
    rec = json.loads((tmp_path / "out" / "SMOOTH_JOINS_RECORD.json").read_text(encoding="utf-8"))
    assert rec["p0"]["PASS"] and rec["p1"]["PASS"] and rec["eval_join_sha256"] == MMJ.sha_file(str(tmp_path / "join.npz"))
    assert rec["seeds"]["0"]["qmap_sha256"] == MMJ.sha_file(str(qd / "QMAP_LR16_s0_q50.json")) and rec["seeds"]["1"]["dump_sha256"] == MMJ.sha_file(str(dd / "reader_s1_score.npz"))
    ok = RE.ctl_p1(rec, str(tmp_path / "out"), rec["eval_join_sha256"])
    assert ok["PASS"], ok
    assert RE.ctl_p0(rec, rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_p1(rec, str(tmp_path / "out"), "0" * 64)["PASS"]
    assert not RE.ctl_p1(rec, str(tmp_path / "nowhere"), rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_p0(rec, "0" * 64)["PASS"]


def test_values_below_0_and_above_60_are_counted_and_left_as_they_are(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    d0 = dict(dumps[0])
    g = np.array(d0["gap"], np.float64)
    ok = np.flatnonzero(np.asarray(d0["valid"], bool) & np.isfinite(g) & np.asarray(d0["lab_present"], bool))
    g[ok[:7]] = -4.0
    g[ok[10:14]] = 300.0
    d0["gap"] = g
    rec = MMJ.build(J, {0: d0, 1: dumps[1]}, qmaps, str(tmp_path / "joins"), meta=MMJ.synthetic_meta())
    for nm in ("S50_s0", "S25_s0", "E_s0", "ES25_s0"):
        a = rec["arms"][nm]
        assert a["n_mixed_above_60"] >= 1 and a["max_mixed"] > 60.0, nm
        if nm.startswith("S"):
            assert a["n_mixed_below_0"] >= 1 and a["min_mixed"] < 0.0, nm
    assert rec["arms"]["S50_s0"]["n_mixed_below_0"] >= 1 and rec["p1"]["PASS"] is True and rec["p0"]["PASS"] is True
    assert rec["arms"]["S25_s0"]["min_mixed"] <= -1.0                      # the value is stored as it is, not cut at 0


# ------------------------------------------------------------------------------------------------ queue (text checks)
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def queue_static_facts(t):
    """static facts of the queue text: the scoring join is the evaluation join, unchanged; a rule join appears only after --rule-join; -> list of (fact, ok)"""
    f = fn_text(t)
    outside = t.replace(f, "")
    return [("scoring gets --eval-join \"$JOINE\" (and the DRY print --eval-join $JOINE)", f.count('--eval-join "$JOINE"') == 1 and f.count("--eval-join $JOINE") == 1),
            ("no other --eval-join anywhere", t.count("--eval-join") == 2 + len(re.findall(r"#.*--eval-join", t))),
            ("JOINE is assigned once, to the pinned evaluation join", len(re.findall(r"^JOINE=", t, re.M)) == 1 and "JOINE=$V/hierrep/pins/join_eval141.npz" in t),
            ("--rule-join once, as extra+=(--rule-join \"$RJOIN\")", f.count("--rule-join") == 1 and 'extra+=(--rule-join "$RJOIN")' in f),
            ("the rule join is named outside the rollout function only in comments or as RJOIN= before a rollout call",
             all(x.lstrip().startswith("#") or ("RJOIN=" in x and " rollout " in x) or "$PJ" not in x or "PJ=" in x or "admit" in x or "NO-SMOOTH" in x for x in outside.split("\n") if "RJOIN" in x or "$PJ" in x)),
            ("no variable of the rule join in the scoring arguments", "$RJOIN" not in f.split("--eval-join")[1].split("\n")[0] and "$PJ" not in f)]


def test_rollout_function_is_byte_identical_to_the_model_queues():
    a = fn_text(QUEUE.read_text(encoding="utf-8"))
    assert a == fn_text(MODEL_QUEUE.read_text(encoding="utf-8")) == fn_text(ERR_QUEUE.read_text(encoding="utf-8")) and len(a) > 3000


def test_queue_edits_and_names():
    t = QUEUE.read_text(encoding="utf-8")
    assert "GP=${GR_ROOT:-/workspace/v8proof/smoothgap10k}" in t and "CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}" in t
    assert "START_SMOOTHGAP10K$PS.txt" in t and "SMOOTHGAP10K_DONE$PS.txt" in t and "SMOOTHGAP10K_STOP$PS.txt" in t and "queue_smoothgap$PS.log" in t and 'PS="_s$SEEDSW"' in t
    assert "cbf7063d867c9f4e" in t and t.count("cbf7063d867c9f4e") >= 3 and "SMOOTH_JOINS_RECORD.json" in t and "MIX_JOINS" not in t
    assert "GAPRULE" not in t and "planted" not in t.lower() and "ERRCUT" not in t and "errcut" not in t
    for arm in ARMS:
        assert ('RJOIN="$PJ/%s_s$k.npz" rollout I10RLcKF5P30Hb20p%s RLc "$k" "$k"' % (arm, arm)) in t
    assert t.count("rollout I10RLcKF5P30Hb20p") == 4 and "rollout I10RLcKF5P30Hb20m" not in t
    adm = t[t.index("admit() {"):t.index("\n}\n", t.index("admit() {"))]
    assert 'rec[c].get("PASS") is not True' in adm and "eval_join_sha256" in adm and "sha256" in adm and '"p0", "p1"' in adm and '"e0"' not in adm
    assert 'admit "$PJ/SMOOTH_JOINS_RECORD.json" "$PJ" "$JOINE" "E_s0,E_s1,S50_s0,S50_s1,S25_s0,S25_s1,ES25_s0,ES25_s1"' in t
    assert 'I10RLcKF5P30Hb20mK100_s$k/T1_DONE" ] || stop "NO-BANKED-K100 s$k"' in t
    assert 'NAMES="I10RLcKF5P30Hb20pE I10RLcKF5P30Hb20pS50 I10RLcKF5P30Hb20pS25 I10RLcKF5P30Hb20pES25"' in t
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
        return re.sub(r"I10RLcKF5P30Hb20[jmp](G0|ES25|E|S50|S25|K100|K50|K25|K12)", "RUN", c)
    assert len(mine) == 8 and len(model) == 12
    assert all("--rule-join" in c and "rzgfhqetj.py" in c and "--form RLc" in c and "--kappa-max 0.2" in c and "--preroll-steps 30" in c for c in mine)
    assert all("--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c for c in mine)
    g0 = [norm(c) for c in model if "jG0" in c]
    assert len(g0) == 2
    for arm in ARMS:
        for s in (0, 1):
            cs = [c for c in mine if ("p%s_s%d " % (arm, s)) in c or ("p%s_s%d/" % (arm, s)) in c]
            assert cs and all(("/%s_s%d.npz" % (arm, s)) in c for c in cs)
            assert all(norm(c) in g0 for c in cs)


# ------------------------------------------------------------------------------------------------ reader: names, controls, binding
def iv(lo, hi):
    return [lo, hi]


def test_name_rules_on_planted_intervals():
    n = RE.names_of("E", iv(-0.05, -0.01), iv(0.0, 0.004), iv(-0.02, 0.03), iv(-0.03, -0.001), 0.0954, 0.0)         # H = 0.0477
    assert n["E_CONTACT"].endswith("SMOOTHGAP10K-E-CUTS-CONTACT") and n["E_HALF"].endswith("-E-WITHIN-HALF-THE-GAP") and n["E_PRICE"].endswith("-E-FALLING-BEHIND-NOT-SEPARATED")
    assert n["E_K100"].endswith("-E-LESS-CONTACT-THAN-K100")
    k = lambda lo, hi: RE.names_of("S50", iv(-0.01, 0.0), iv(0.02, 0.05), iv(0.0, 0.1), iv(lo, hi), 0.0954, 0.0)["S50_K100"]      # noqa: E731
    assert k(-0.02, -0.0001).endswith("LESS-CONTACT-THAN-K100") and k(-0.02, 0.0).endswith("CONTACT-NOT-SEPARATED-FROM-K100") and k(0.0, 0.02).endswith("CONTACT-NOT-SEPARATED-FROM-K100")
    assert k(0.0001, 0.02).endswith("MORE-CONTACT-THAN-K100") and k(-0.02, 0.02).endswith("CONTACT-NOT-SEPARATED-FROM-K100")
    assert RE.names_of("S50", iv(-0.01, 0.0), iv(0.02, 0.0477), iv(0.0, 0.1), iv(0, 0), 0.0954, 0.0)["S50_HALF"].endswith("WITHIN-HALF-THE-GAP")      # at H: within
    assert RE.names_of("S50", iv(-0.01, 0.0), iv(0.0478, 0.06), iv(0.0, 0.1), iv(0, 0), 0.0954, 0.0)["S50_HALF"].endswith("OUTSIDE-HALF-THE-GAP")
    assert RE.names_of("S25", iv(-0.01, 0.0), iv(0.02, 0.05), iv(0.0, 0.1), iv(0, 0), 0.0954, 0.0)["S25_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED")
    assert RE.names_of("ES25", iv(0.0, 0.01), iv(0.02, 0.05), iv(-0.1, -0.001), iv(0, 0), 0.0954, 0.0)["ES25_CONTACT"].endswith("CONTACT-NOT-SHOWN")
    assert RE.names_of("ES25", iv(0.0, 0.01), iv(0.02, 0.05), iv(-0.1, -0.001), iv(0, 0), 0.0954, 0.0)["ES25_PRICE"].endswith("FALLS-BEHIND-LESS")
    assert RE.names_of("ES25", iv(0.0, 0.01), iv(0.02, 0.05), iv(0.001, 0.1), iv(0, 0), 0.0954, 0.0)["ES25_PRICE"].endswith("FALLS-BEHIND-MORE")


def payload(rr=0.0954, rlc=0.0, within_arms=("E", "S50", "S25", "ES25"), more_arms=()):
    pr = {}
    for a in RE.ARMS:
        hi = 0.004 if a in within_arms else 0.06
        fl = iv(0.01, 0.05) if a in more_arms else iv(-0.02, 0.03)
        pr[a + "-VS-RR"] = {"C": {"interval": iv(-0.05, -0.01), "pooled": -0.03}, "U": {"interval": iv(-0.05, -0.01), "pooled": -0.03}, "F": {"interval": fl, "pooled": 0.0}}
        pr[a + "-VS-RLc"] = {k: {"interval": iv(0.0 if a in within_arms else 0.055, hi), "pooled": 0.002} for k in ("C", "U", "F")}
        pr[a + "-VS-K100"] = {k: {"interval": iv(-0.02, 0.01), "pooled": -0.005} for k in ("C", "U", "F")}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}, "K100": {"C": 0.0477}}, "pairs": pr, "families": {}, "strategic": {"pairs": {}}}


def test_card_name_rule_cases():
    assert RE.all_names(payload())["CARD"] == "SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES"
    assert RE.all_names(payload(within_arms=("S25",)))["CARD"] == "SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES"
    assert RE.all_names(payload(within_arms=()))["CARD"] == "SMOOTHGAP10K-NO-PROCESSED-GAP-SUFFICES"
    assert RE.all_names(payload(within_arms=("E",), more_arms=("E",)))["CARD"] == "SMOOTHGAP10K-NO-PROCESSED-GAP-SUFFICES"          # within half the gap but falls behind more
    assert RE.all_names(payload(within_arms=("E", "S50"), more_arms=("E",)))["CARD"] == "SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES"      # another arm is within and does not fall behind more
    assert RE.all_names(payload(within_arms=("S25",), more_arms=("ES25",)))["CARD"] == "SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES"
    n = RE.all_names(payload())
    assert len(n) == 17 and set(k.split("_")[0] for k in n) == set(RE.ARMS) | {"CARD"}


def test_a_card_name_that_the_intervals_do_not_give_turns_the_check_red():
    p = payload(within_arms=("E",), more_arms=("E",))
    assert RE.card_check(p, "SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES")["PASS"] is False
    assert RE.card_check(p, "SMOOTHGAP10K-NO-PROCESSED-GAP-SUFFICES")["PASS"] is True
    p = payload(within_arms=())
    assert RE.card_check(p, "SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES")["PASS"] is False
    p = payload()
    assert RE.card_check(p, RE.all_names(p)["CARD"])["PASS"] is True and RE.card_check(p, "SMOOTHGAP10K-NO-PROCESSED-GAP-SUFFICES")["PASS"] is False


def test_controls_p2_p3_p4_p5():
    assert RE.ctl_p2("x\nZZ CHECK-WRAPPER-J PASS\n", "cbf7063d867c9f4e a.py", "cbf7063d867c9f4e")["PASS"]
    assert not RE.ctl_p2("ZZ CHECK-WRAPPER-J FAIL J2A\n", "cbf7063d867c9f4e", "cbf7063d867c9f4e")["PASS"]
    assert not RE.ctl_p2("ZZ CHECK-WRAPPER-J PASS\n", "0123456789abcdef", "cbf7063d867c9f4e")["PASS"]
    assert RE.ctl_p3(0.09541, 0.0, 0.04769)["PASS"] and RE.ctl_p3(0.09536, 0.00004, 0.04774)["PASS"]
    assert not RE.ctl_p3(0.0960, 0.0, 0.0477)["PASS"] and not RE.ctl_p3(0.0954, 0.0138, 0.0477)["PASS"] and not RE.ctl_p3(None, 0.0, 0.0477)["PASS"]
    assert not RE.ctl_p3(0.0954, 0.0, 0.0462)["PASS"] and not RE.ctl_p3(0.0954, 0.0, None)["PASS"] and not RE.ctl_p3(0.0954, 0.0, 0.0954)["PASS"]
    assert RE.ctl_p4("25 passed in 3s")["PASS"] and not RE.ctl_p4("2 failed, 3 passed")["PASS"] and not RE.ctl_p4("")["PASS"]
    assert RE.ctl_p5(True, True, True)["PASS"] and not RE.ctl_p5(False, True, True)["PASS"] and not RE.ctl_p5(True, False, True)["PASS"] and not RE.ctl_p5(True, True, False)["PASS"]


ALL_PASS = {c: {"PASS": True} for c in RE.CONTROLS}


def test_name_written_with_a_control_failed_turns_red(tmp_path):
    ctl = RE.assemble_controls(dict(ALL_PASS, p1={"PASS": False}), payload())
    assert ctl["controls_all_pass"] is False and "payload" not in ctl and "withheld" in ctl
    f = tmp_path / "controls.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "result.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 3 and not out.exists()
    f2 = tmp_path / "forged.json"
    f2.write_text(json.dumps(dict(ctl, controls_all_pass=True, payload=payload())), encoding="utf-8")        # an edited flag does not open the names: the controls themselves are re-checked
    assert RE.main(["--stage", "names", "--controls", str(f2), "--out", str(tmp_path / "r2.json")]) == 3
    f3 = tmp_path / "missing.json"
    short = {k: v for k, v in ALL_PASS.items() if k != "p5"}                                                   # a controls file without one control does not open the names
    f3.write_text(json.dumps(dict(RE.assemble_controls(short, payload()))), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(f3), "--out", str(tmp_path / "r3.json")]) == 3 and not (tmp_path / "r3.json").exists()
    for c in RE.CONTROLS:                                                                                      # each one control failed opens nothing
        f4 = tmp_path / ("one_%s.json" % c)
        f4.write_text(json.dumps(RE.assemble_controls(dict(ALL_PASS, **{c: {"PASS": False}}), payload())), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(f4), "--out", str(tmp_path / ("o_%s.json" % c))]) == 3 and not (tmp_path / ("o_%s.json" % c)).exists()


def test_names_stage_binds_the_controls_file(tmp_path):
    ctl = RE.assemble_controls(ALL_PASS, payload())
    f = tmp_path / "c.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "r.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["controls_sha256"] == RE.sha_file(str(f)) and len(r["names"]) == 17 and r["card"] == "PREREG_V8F_PROOF_ADD97"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 2             # refuses an existing output


def test_share_of_the_contact_gap_and_the_u_f_differences_against_k100():
    p = payload()
    s = RE.share_rows(p["levels"], p["pairs"])
    assert abs(s["E"]["share"] - 0.03 / 0.0954) < 1e-12 and set(s) == set(RE.ARMS)
    assert RE.print_arm_sections(p) and all("K100" in x for x in RE.print_arm_sections(p))


def test_reader_source_carries_the_strategic_family_the_k100_reference_and_the_engine_bootstrap_setting():
    t = (HERE / "read_smoothgap10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and "R.strategic_grid(" in t and "S.pair_read(" in t and "R.family_of(" in t
    assert 'CARD, PRE = "PREREG_V8F_PROOF_ADD97", "SMOOTHGAP10K"' in t and '"RLcKF5P30Hb20p" + a' in t and 'ARMS = ("E", "S50", "S25", "ES25")' in t
    assert 'GAINS = {"K100": "RLcKF5P30Hb20mK100"}' in t and "runs.update(GAINS)" in t and 'pairs_def[x + "-VS-K100"]' in t and "R.ALL_RUNS.setdefault(r," in t
    assert 'controls = {"p0"' in t and '"p5": ctl_p5(' in t and '"p3": ctl_p3(' in t and "differences_against_K100" in t and "ladder" not in t.lower()


def test_reader_selftest_runs():
    assert RE.selftest() == 0


def test_podread_text():
    t = (HERE / "podread_smoothgap10k.sh").read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=-1" in t and "GP=$V/smoothgap10k" in t and "OUT=$GP/read" in t and "CHK=$V/gaprule10kb/check_j2" in t
    assert "SMOOTHGAP10K_CONTROLS.json" in t and "SMOOTHGAP10K_RESULT.json" in t and "SMOOTHGAP10K_DONE_s0.txt" in t and "SMOOTHGAP10K_DONE_s1.txt" in t and "lead_pytest.out" in t
    assert "joins/SMOOTH_JOINS_RECORD.json" in t and "I10RLcKF5P30Hb20p${x}_s$s" in t and "read_smoothgap10k.py" in t and "p[0-5]" in t
    assert 'for d in "$W/t1/I10RLcKF5P30Hb20mK100_s0" "$W/t1/I10RLcKF5P30Hb20mK100_s1"; do' in t and '[ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"' in t
    assert "ERRCUT" not in t and "errcut" not in t
    assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_smoothgap10k.sh").replace("\\", "/")]).returncode == 0


def test_texts_hold_no_banned_words():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    n = 0
    for f in ("make_smooth_joins.py", "make_smoothgap_queue.py", "pod_smoothgap10k_queue.sh", "read_smoothgap10k.py", "podread_smoothgap10k.sh", "test_smoothgap.py", "SMOOTHGAP_DEFS.md"):
        p = HERE / f
        if p.exists():
            n += 1
            t = p.read_text(encoding="utf-8").lower()
            hit = [w for w in bad if w in t]
            assert not hit, (f, hit)
    assert n >= 6
