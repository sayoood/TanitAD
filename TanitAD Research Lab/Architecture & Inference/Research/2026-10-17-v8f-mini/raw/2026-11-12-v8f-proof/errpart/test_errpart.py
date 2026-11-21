"""test_errpart.py -- tests of the kit of card ADD98 (ERRPART10K): part joins, queue, reader controls and names, with the planted faults of control q4. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest errpart/test_errpart.py -q -p no:cacheprovider --basetemp errpart/tmp_pt/<new name>"""
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
import make_part_joins as MMJ          # noqa: E402
import read_errpart10k as RE            # noqa: E402

QUEUE = HERE / "pod_errpart10k_queue.sh"
GEN = HERE / "make_errpart_queue.py"
ERR_QUEUE = HERE.parent / "errcut" / "pod_errcut10k_queue.sh"
PREV_QUEUE = HERE.parent / "smoothgap" / "pod_smoothgap10k_queue.sh"
MODEL_QUEUE = HERE.parent / "gaprule" / "pod_gaprule10k_queue.sh"
ARMS = ("M", "W", "S", "F")


# ------------------------------------------------------------------------------------------------ part joins: q0 and q1 and their planted faults
@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    base = tmp_path_factory.mktemp("ep")
    J, dumps, qmaps = MMJ.synthetic_all()
    rec = MMJ.build(J, dumps, qmaps, str(base / "clean"), meta=MMJ.synthetic_meta())
    return {"J": J, "dumps": dumps, "qmaps": qmaps, "clean": rec, "base": base}


def fault_build(mix, fault):
    cache = mix.setdefault("fault_cache", {})
    if fault not in cache:
        cache[fault] = MMJ.build(mix["J"], mix["dumps"], mix["qmaps"], str(mix["base"] / ("f_" + fault)), faults=(fault,), meta=MMJ.synthetic_meta())
    return cache[fault]


def test_clean_build_passes_q0_and_q1(mix):
    r = mix["clean"]
    assert r["q0"]["PASS"], r["q0"]
    assert r["q1"]["PASS"], r["q1"]
    assert len(r["arms"]) == 8 and sorted(r["arms"]) == sorted("%s_s%d" % (a, s) for a in ARMS for s in (0, 1)) and r["card"] == "PREREG_V8F_PROOF_ADD98"
    assert r["window_half_frames"] == 5 and r["run_jump_max_m"] == 3.0
    for nm, a in r["arms"].items():
        assert a["max_abs_dev_saved_minus_processed_second_routine"] <= 1e-4 and a["other_keys_all_equal"] is True and a["n_changed_outside_mixed_frames"] == 0
        assert 0.0 <= a["fraction_without_reader_read"] <= 0.05 and a["n_with_reader_read"] > 0 and a["n_mixed_above_60"] >= 0 and a["n_mixed_below_0"] >= 0
        assert a["n_label_frames_without_read_not_label_gap"] == 0 and a["n_runs"] == a["n_runs_second_routine"] > 0 and a["median_run_length"] >= 1
        assert a["error_statistics_processed_minus_label"]["n_frames"] == a["n_with_reader_read"]
    for s in ("0", "1"):
        assert mix["clean"]["identities"][s]["max_abs_dev_M_plus_W_minus_g_minus_L"] <= 1e-4 and mix["clean"]["identities"][s]["max_abs_dev_S_plus_F_minus_W"] <= 1e-4
    for s in (0, 1):
        assert mix["clean"]["arms"]["W_s%d" % s]["clip_mean_P_minus_L_max_abs"] <= 1e-4


def test_the_eight_joins_are_eight_files_and_each_model_seed_has_its_own(mix):
    shas = {mix["clean"]["arms"][n]["sha256"] for n in mix["clean"]["arms"]}
    assert len(shas) == 8
    for arm in ARMS:
        a = np.load(str(mix["base"] / "clean" / (arm + "_s0.npz")), allow_pickle=False)["lead_gap"]
        b = np.load(str(mix["base"] / "clean" / (arm + "_s1.npz")), allow_pickle=False)["lead_gap"]
        assert not MMJ.equal_key(a, b)


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


def hand_parts(J, vals):
    """m, r, s, f of every mix frame, written a third way: slices of per-run lists. -> {(clip, frame): (L, m, r, s, f)}"""
    vm = MMJ.valid_mask(J)
    out = {}
    for ci, c in enumerate(str(x) for x in J["clip"]):
        o = int(J["off"][ci])
        fr = [f for f in range(int(J["n"][ci])) if vm[o + f] and (c, f) in vals]
        if not fr:
            continue
        L = {f: float(np.float32(J["lead_gap"][o + f])) for f in fr}
        e = {f: vals[(c, f)] - L[f] for f in fr}
        m = float(np.mean([e[f] for f in fr]))
        runs, cur = [], [fr[0]]
        for f in fr[1:]:
            if f == cur[-1] + 1 and abs(L[f] - L[cur[-1]]) <= 3.0:
                cur.append(f)
            else:
                runs.append(cur)
                cur = [f]
        runs.append(cur)
        for run in runs:
            rr = [e[f] - m for f in run]
            for i, f in enumerate(run):
                s = float(np.mean(rr[max(i - 5, 0):i + 6]))
                out[(c, f)] = (L[f], m, rr[i], s, rr[i] - s)
    return out


def row_of(J, c, f):
    names = [str(x) for x in J["clip"]]
    return int(J["off"][names.index(c)]) + f


def test_parts_by_hand_on_the_synthetic_join_for_every_arm_and_both_seeds(mix):
    J = mix["J"]
    for s in (0, 1):
        hp = hand_parts(J, dump_values(mix, s))
        assert len(hp) == mix["clean"]["arms"]["W_s%d" % s]["n_with_reader_read"]
        for arm, k in (("M", 1), ("W", 2), ("S", 3), ("F", 4)):
            g = np.load(str(mix["base"] / "clean" / ("%s_s%d.npz" % (arm, s))), allow_pickle=False)["lead_gap"]
            for (c, f), v in hp.items():
                assert abs(float(g[row_of(J, c, f)]) - (v[0] + v[k])) < 1e-4, (arm, s, c, f)
        assert max(v[3] for v in hp.values()) != min(v[3] for v in hp.values())


def test_runs_exist_with_a_boundary_inside_a_clip_and_a_window_that_is_cut(mix):
    r = mix["clean"]["arms"]["W_s0"]
    assert r["n_runs"] > r["n_clips_with_mix_frames"] and r["median_run_length"] >= 3


def test_part_arrays_by_hand_on_small_arrays():
    # clip 0: rows 0-11, label gap about 10 for six frames then a jump of 9.4 m; clip 1: rows 12-25, label gap 15; clip 2: row 26 only; row 27 (clip 3) is no mix frame
    L = np.concatenate([np.full(6, 10.0), np.full(6, 19.4), np.full(14, 15.0), [30.0, 30.0]])
    e = np.concatenate([np.arange(1.0, 13.0), np.arange(0.0, 14.0), [3.0, 0.0]])
    g = L + e
    clip = np.concatenate([np.zeros(12, int), np.ones(14, int), [2, 3]])
    mixm = np.ones(28, bool)
    mixm[27] = False
    pa = MMJ.part_arrays(g, mixm, clip, L)
    assert list(pa["length"]) == [6, 6, 14, 1] and list(pa["start"]) == [0, 6, 12, 26]
    assert np.allclose(pa["m"][:12], 6.5) and np.allclose(pa["m"][12:26], 6.5) and pa["m"][26] == 3.0
    assert np.allclose(pa["s"][:6], -3.0) and np.allclose(pa["s"][6:12], 3.0)                    # a window never crosses the boundary of the run (the jump of the label gap)
    assert np.allclose(pa["s"][12:26][[0, 5, 7, 13]], [-4.0, -1.5, 0.5, 4.0])                  # windows cut at the ends: frames 0..5, 0..10, 2..12, 8..13 of e = 0..13 minus 6.5
    assert pa["r"][26] == 0.0 and pa["s"][26] == 0.0 and pa["f"][26] == 0.0 and pa["m"][26] == 3.0 and np.isnan(pa["m"][27])
    ok = np.flatnonzero(mixm)
    assert np.allclose(pa["m"][ok] + pa["r"][ok], e[ok]) and np.allclose(pa["s"][ok] + pa["f"][ok], pa["r"][ok])
    for fault, expect_different in (("slow_across_runs", "s"), ("mean_other_clip", "m"), ("window_other", "s")):
        pb = MMJ.part_arrays(g, mixm, clip, L, (fault,))
        assert not np.allclose(pb[expect_different][ok], pa[expect_different][ok]), fault


def test_the_four_joins_of_a_clip_with_one_mix_frame_and_runs_of_one_frame():
    L = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    g = L + np.array([2.0, -1.0, 4.0, 0.5, 1.0])
    clip = np.array([0, 0, 0, 0, 0])
    mixm = np.ones(5, bool)
    pa = MMJ.part_arrays(g, mixm, clip, L)                      # every step is 10 m: five runs of one frame
    assert list(pa["length"]) == [1] * 5
    assert np.allclose(pa["s"], pa["r"]) and np.allclose(pa["f"], 0.0)
    P = {a: MMJ.processed(MMJ.SPEC[a], pa, L) for a in ARMS}
    assert np.allclose(P["F"], L) and np.allclose(P["W"], P["S"]) and np.allclose(P["M"] - L + P["W"] - L, g - L)
    one = MMJ.part_arrays(g, np.array([0, 0, 1, 0, 0], bool), clip, L)
    assert one["m"][2] == 4.0 and one["r"][2] == 0.0 and one["s"][2] == 0.0 and one["f"][2] == 0.0


def test_each_planted_fault_of_control_q4_turns_q1_red(mix):
    for fault, words in (("mean_other_clip", ("second time", "clip mean")), ("slow_across_runs", ("second time", " runs ")), ("swap_parts", ("second time",)),
                         ("window_other", ("second time", "window of 3")), ("other_seed", ("second time",)), ("other_field", ("another field",))):
        r = fault_build(mix, fault)
        assert not r["q1"]["PASS"], fault
        for w in words:
            assert any(w in x for x in r["q1"]["failures"]), (fault, w, r["q1"]["failures"][:3])
        assert r["q0"]["PASS"], fault


def test_the_second_routine_is_needed_the_sum_identities_alone_do_not_see_two_of_the_faults(mix):
    for fault in ("mean_other_clip", "swap_parts"):
        r = fault_build(mix, fault)
        assert not any("sum identity" in x for x in r["q1"]["failures"]), fault
    assert any("clip mean" in x for x in fault_build(mix, "mean_other_clip")["q1"]["failures"])


def test_a_broken_sum_identity_or_clip_mean_in_the_saved_files_is_seen(mix, tmp_path):
    """tamper with a saved W file (one value moved by 0.01 m) and measure again: every clause that holds the value turns red"""
    J, d = mix["J"], mix["clean"]
    p = tmp_path / "t"
    p.mkdir()
    sec, runs = MMJ.second_parts(J, mix["dumps"][0], mix["qmaps"][0], 0)
    ix = sorted(sec)[3]
    for arm in ARMS:
        src = np.load(str(mix["base"] / "clean" / (arm + "_s0.npz")), allow_pickle=False)
        z = {k: src[k] for k in src.files}
        if arm == "W":
            z["lead_gap"] = z["lead_gap"].copy()
            z["lead_gap"][ix] += np.float32(0.01)
        np.savez(str(p / (arm + "_s0.npz")), **z)
    idn = MMJ.identity_facts({a: str(p / (a + "_s0.npz")) for a in ARMS}, J, sec)
    assert idn["max_abs_dev_M_plus_W_minus_g_minus_L"] > 5e-3 and idn["max_abs_dev_S_plus_F_minus_W"] > 5e-3
    tam = copy.deepcopy(d)
    tam["identities"]["0"] = idn
    assert not MMJ.q1_check(tam)["PASS"] and any("sum identity" in x for x in MMJ.q1_check(tam)["failures"])


def test_dump_label_gap_not_equal_to_the_join_turns_q0_red(mix):
    bad = dict(mix["dumps"][0])
    lg = np.array(bad["lab_gap"], copy=True)
    lg[np.flatnonzero(np.isfinite(lg))[7]] += 0.01
    bad["lab_gap"] = lg
    r = MMJ.build(mix["J"], {0: bad, 1: mix["dumps"][1]}, mix["qmaps"], str(mix["base"] / "f_label"), meta=MMJ.synthetic_meta())
    assert not r["q0"]["PASS"] and any("label gap difference" in x for x in r["q0"]["failures"])


def test_q0_red_when_the_dump_is_of_the_other_reader_seed_or_not_on_the_join(mix):
    d1 = dict(mix["dumps"][1])
    d1["seed"] = np.int64(0)
    r = MMJ.build(mix["J"], {0: mix["dumps"][0], 1: d1}, mix["qmaps"], str(mix["base"] / "f_dumpseed"), meta=MMJ.synthetic_meta())
    assert not r["q0"]["PASS"] and any("records seed" in x for x in r["q0"]["failures"])
    d0 = dict(mix["dumps"][0])
    d0["clip_index"] = (np.asarray(d0["clip_index"]) + 1) % len(mix["J"]["n"])
    r = MMJ.build(mix["J"], {0: d0, 1: mix["dumps"][1]}, mix["qmaps"], str(mix["base"] / "f_clipidx"), meta=MMJ.synthetic_meta())
    assert not r["q0"]["PASS"] and any("n_clip_index_mismatch" in x for x in r["q0"]["failures"])


def test_q1_limits_and_record_fields(mix):
    c = mix["clean"]

    def with_(nm, **kw):
        r = copy.deepcopy(c)
        r["arms"][nm].update(kw)
        return r

    def idn(s, **kw):
        r = copy.deepcopy(c)
        r["identities"][s].update(kw)
        return r
    assert MMJ.q1_check(with_("S_s0", fraction_without_reader_read=0.05))["PASS"] is True
    assert MMJ.q1_check(with_("S_s0", fraction_without_reader_read=0.0501))["PASS"] is False
    assert MMJ.q1_check(with_("M_s1", fraction_without_reader_read=0.0501))["PASS"] is False
    assert MMJ.q1_check(with_("S_s1", min_mixed=-0.01, n_mixed_below_0=3))["PASS"] is True          # counted, no bar
    assert MMJ.q1_check(with_("S_s1", n_mixed_below_0=None))["PASS"] is False
    assert MMJ.q1_check(with_("F_s1", n_mixed_above_60=None))["PASS"] is False
    assert MMJ.q1_check(with_("F_s0", n_changed_outside_mixed_frames=1))["PASS"] is False
    assert MMJ.q1_check(with_("F_s0", n_label_frames_without_read_not_label_gap=2))["PASS"] is False
    assert MMJ.q1_check(with_("S_s1", part="r"))["PASS"] is False
    assert MMJ.q1_check(with_("S_s1", reader_seeds=[0]))["PASS"] is False
    assert MMJ.q1_check(with_("S_s1", max_abs_dev_saved_minus_processed_second_routine=1.1e-4))["PASS"] is False
    assert MMJ.q1_check(with_("S_s1", max_abs_dev_saved_minus_processed_second_routine=1e-4))["PASS"] is True
    assert MMJ.q1_check(with_("W_s1", clip_mean_P_minus_L_max_abs=1.1e-4))["PASS"] is False
    assert MMJ.q1_check(with_("W_s1", clip_mean_P_minus_L_max_abs=1e-4))["PASS"] is True
    assert MMJ.q1_check(with_("M_s1", clip_mean_P_minus_L_max_abs=3.0))["PASS"] is True              # the clip mean is 0 only for W
    assert MMJ.q1_check(with_("S_s0", n_runs=c["arms"]["S_s0"]["n_runs"] + 1))["PASS"] is False
    assert MMJ.q1_check(idn("0", max_abs_dev_M_plus_W_minus_g_minus_L=1.1e-4))["PASS"] is False
    assert MMJ.q1_check(idn("1", max_abs_dev_S_plus_F_minus_W=1.1e-4))["PASS"] is False
    assert MMJ.q1_check(idn("1", max_abs_dev_S_plus_F_minus_W=1e-4))["PASS"] is True
    assert MMJ.q1_check(idn("0", n_frames=0))["PASS"] is False
    r = copy.deepcopy(c)
    r["window_half_frames"] = 4
    assert MMJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["run_jump_max_m"] = 4.0
    assert MMJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["seeds"]["1"]["qmap_file"] = "QMAP_LR16_s0_q50.json"
    assert MMJ.q1_check(r)["PASS"] is False
    assert MMJ.q1_check(c, expect_qmap_dir="/workspace/v8proof/integ10k/qmap_sweep")["PASS"] is True
    assert MMJ.q1_check(c, expect_qmap_dir="/somewhere/else")["PASS"] is False
    r = copy.deepcopy(c)
    r["arms"].pop("F_s1")
    assert MMJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["arms"]["M_s1"]["sha256"] = r["arms"]["M_s0"]["sha256"]
    assert MMJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["card"] = "PREREG_V8F_PROOF_ADD97"
    assert MMJ.q1_check(r)["PASS"] is False


def test_q0_record_fields(mix):
    r = copy.deepcopy(mix["clean"])
    r["seeds"]["0"]["e0"]["max_abs_diff"] = 1.1e-4
    assert MMJ.q0_check(r)["PASS"] is False
    r["seeds"]["0"]["e0"]["max_abs_diff"] = 1e-4
    assert MMJ.q0_check(r)["PASS"] is True
    r["seeds"]["0"]["e0"]["n_duplicate_keys"] = 1
    assert MMJ.q0_check(r)["PASS"] is False


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
    rec = json.loads((tmp_path / "out" / "PART_JOINS_RECORD.json").read_text(encoding="utf-8"))
    assert rec["q0"]["PASS"] and rec["q1"]["PASS"] and rec["eval_join_sha256"] == MMJ.sha_file(str(tmp_path / "join.npz"))
    assert rec["seeds"]["0"]["qmap_sha256"] == MMJ.sha_file(str(qd / "QMAP_LR16_s0_q50.json")) and rec["seeds"]["1"]["dump_sha256"] == MMJ.sha_file(str(dd / "reader_s1_score.npz"))
    ok = RE.ctl_q1(rec, str(tmp_path / "out"), rec["eval_join_sha256"])
    assert ok["PASS"], ok
    assert RE.ctl_q0(rec, rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_q1(rec, str(tmp_path / "out"), "0" * 64)["PASS"]
    assert not RE.ctl_q1(rec, str(tmp_path / "nowhere"), rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_q0(rec, "0" * 64)["PASS"]


def test_values_below_0_and_above_60_are_counted_and_left_as_they_are(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    d0 = dict(dumps[0])
    g = np.array(d0["gap"], np.float64)
    ok = np.flatnonzero(np.asarray(d0["valid"], bool) & np.isfinite(g) & np.asarray(d0["lab_present"], bool))
    g[ok[:7]] = -40.0
    g[ok[10:14]] = 300.0
    d0["gap"] = g
    rec = MMJ.build(J, {0: d0, 1: dumps[1]}, qmaps, str(tmp_path / "joins"), meta=MMJ.synthetic_meta())
    for nm in ("W_s0", "F_s0"):
        a = rec["arms"][nm]
        assert a["n_mixed_above_60"] >= 1 and a["max_mixed"] > 60.0 and a["n_mixed_below_0"] >= 1 and a["min_mixed"] < 0.0, nm
    assert rec["q1"]["PASS"] is True and rec["q0"]["PASS"] is True              # counted, no pass criterion; the sum identities hold on float32 values of 300 m
    assert rec["arms"]["W_s0"]["min_mixed"] <= -1.0                              # the value is stored as it is, not cut at 0


# ------------------------------------------------------------------------------------------------ queue (text checks, generator)
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
             all(x.lstrip().startswith("#") or ("RJOIN=" in x and " rollout " in x) or "$PJ" not in x or "PJ=" in x or "admit" in x or "NO-PART" in x for x in outside.split("\n") if "RJOIN" in x or "$PJ" in x)),
            ("no variable of the rule join in the scoring arguments", "$RJOIN" not in f.split("--eval-join")[1].split("\n")[0] and "$PJ" not in f)]


def test_rollout_function_is_byte_identical_to_the_earlier_queues():
    a = fn_text(QUEUE.read_text(encoding="utf-8"))
    assert a == fn_text(PREV_QUEUE.read_text(encoding="utf-8")) == fn_text(MODEL_QUEUE.read_text(encoding="utf-8")) == fn_text(ERR_QUEUE.read_text(encoding="utf-8")) and len(a) > 3000


def test_queue_edits_and_names():
    t = QUEUE.read_text(encoding="utf-8")
    assert "GP=${GR_ROOT:-/workspace/v8proof/errpart10k}" in t and "CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}" in t and "SM=$GP/small_ep" in t
    assert "START_ERRPART10K$PS.txt" in t and "ERRPART10K_DONE$PS.txt" in t and "ERRPART10K_STOP$PS.txt" in t and "queue_errpart$PS.log" in t and 'PS="_s$SEEDSW"' in t
    assert "cbf7063d867c9f4e" in t and t.count("cbf7063d867c9f4e") >= 3 and "PART_JOINS_RECORD.json" in t and "SMOOTH" not in t and "MIX_JOINS" not in t
    assert "GAPRULE" not in t and "planted" not in t.lower() and "ERRCUT" not in t and "errcut" not in t and "smoothgap" not in t
    for arm in ARMS:
        assert ('RJOIN="$PJ/%s_s$k.npz" rollout I10RLcKF5P30Hb20q%s RLc "$k" "$k"' % (arm, arm)) in t
    assert t.count("rollout I10RLcKF5P30Hb20q") == 4 and "rollout I10RLcKF5P30Hb20m" not in t and "rollout I10RLcKF5P30Hb20p" not in t
    adm = t[t.index("admit() {"):t.index("\n}\n", t.index("admit() {"))]
    assert 'rec[c].get("PASS") is not True' in adm and "eval_join_sha256" in adm and "sha256" in adm and '"q0", "q1"' in adm and '"e0"' not in adm and '"p0"' not in adm
    assert 'admit "$PJ/PART_JOINS_RECORD.json" "$PJ" "$JOINE" "M_s0,M_s1,W_s0,W_s1,S_s0,S_s1,F_s0,F_s1"' in t
    assert 'I10RLcKF5P30Hb20mK100_s$k/T1_DONE" ] || stop "NO-BANKED-K100 s$k"' in t
    assert 'NAMES="I10RLcKF5P30Hb20qM I10RLcKF5P30Hb20qW I10RLcKF5P30Hb20qS I10RLcKF5P30Hb20qF"' in t
    assert "28" in t and " rm " not in t and " mv " not in t and " kill " not in t


def test_queue_static_facts_hold_and_each_planted_fault_turns_one_red():
    t = QUEUE.read_text(encoding="utf-8")
    assert all(ok for _f, ok in queue_static_facts(t)), [f for f, ok in queue_static_facts(t) if not ok]
    bad1 = t.replace('--eval-join "$JOINE"', '--eval-join "${RJOIN:-$JOINE}"')
    bad2 = t.replace("JOINE=$V/hierrep/pins/join_eval141.npz", "JOINE=${RJOIN:-$V/hierrep/pins/join_eval141.npz}")
    bad3 = t.replace('extra+=(--rule-join "$RJOIN")', 'extra+=(--rule-join "$RJOIN" --eval-join "$RJOIN")')
    for b in (bad1, bad2, bad3):
        assert b != t and any(not ok for _f, ok in queue_static_facts(b))


def run_generator(src_text, out, tmp_path, name):
    """runs a (possibly edited) copy of the generator as if it stood in errpart/ (its __file__ is the real one); -> (rc, stderr)"""
    p = tmp_path / name
    p.write_text(src_text, encoding="utf-8")
    runner = "import sys; g = {'__file__': sys.argv[2], '__name__': '__main__'}; exec(compile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], 'exec'), g)"
    r = subprocess.run([sys.executable, "-c", runner, str(p), str(GEN)], env=dict(os.environ, EP_OUT=str(out), PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stderr


def test_queue_generator_reproduces_the_shipped_queue(tmp_path):
    out = tmp_path / "q.sh"
    rc, err = run_generator(GEN.read_text(encoding="utf-8"), out, tmp_path, "gen_clean.py")
    assert rc == 0, err[-400:]
    assert out.read_bytes() == QUEUE.read_bytes()


def test_each_asserted_edit_of_the_queue_generator_turns_it_red(tmp_path):
    src = GEN.read_text(encoding="utf-8")
    cases = {
        "edit_misses": ('once("SM=$GP/small_sg", "SM=$GP/small_ep")', 'once("SM=$GP/small_sgX", "SM=$GP/small_ep")'),
        "count_is_wrong": ("'$GP/START_SMOOTHGAP10K$PS.txt', '$GP/START_ERRPART10K$PS.txt', 2)", "'$GP/START_SMOOTHGAP10K$PS.txt', '$GP/START_ERRPART10K$PS.txt', 3)"),
        "wrong_source": ('SRC_SHA_PREFIX = "cef2a3414b240499"', 'SRC_SHA_PREFIX = "0000000000000000"'),
        "rollout_edited": ("# ---- guards\n", "text = text.replace('--limit causal', '--limit causal2')\n# ---- guards\n"),
        "a_banned_name_left": ("# ---- guards\n", "text = text + '# S50\\n'\n# ---- guards\n"),
    }
    for nm, (a, b) in cases.items():
        assert src.count(a) == 1, nm
        rc, err = run_generator(src.replace(a, b), tmp_path / (nm + ".sh"), tmp_path, "gen_%s.py" % nm)
        assert rc != 0 and not (tmp_path / (nm + ".sh")).exists(), (nm, err[-200:])


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
        return re.sub(r"I10RLcKF5P30Hb20[jmpq](G0|M|W|S|F|K100|K50|K25|K12)", "RUN", c)
    assert len(mine) == 8 and len(model) == 12
    assert all("--rule-join" in c and "rzgfhqetj.py" in c and "--form RLc" in c and "--kappa-max 0.2" in c and "--preroll-steps 30" in c for c in mine)
    assert all("--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c for c in mine)
    g0 = [norm(c) for c in model if "jG0" in c]
    assert len(g0) == 2
    for arm in ARMS:
        for s in (0, 1):
            cs = [c for c in mine if ("q%s_s%d " % (arm, s)) in c or ("q%s_s%d/" % (arm, s)) in c]
            assert len(cs) == 1 and ("/%s_s%d.npz" % (arm, s)) in cs[0]
            assert norm(cs[0]) in g0


# ------------------------------------------------------------------------------------------------ reader: names, controls, binding
def iv(lo, hi):
    return [lo, hi]


def test_name_rules_on_planted_intervals():
    n = RE.names_of("M", iv(0.01, 0.05), iv(-0.02, 0.03), iv(-0.03, -0.001))
    assert n["M_CONTACT"].endswith("ERRPART10K-M-CARRIES-CONTACT") and n["M_PRICE"].endswith("-M-FALLING-BEHIND-NOT-SEPARATED") and n["M_K100"].endswith("-M-LESS-CONTACT-THAN-K100")
    c = lambda lo, hi: RE.names_of("W", iv(lo, hi), iv(0, 0), iv(0, 0))["W_CONTACT"]                        # noqa: E731
    assert c(0.0001, 0.05).endswith("CARRIES-CONTACT") and c(0.0, 0.05).endswith("NO-CONTACT-SHOWN") and c(-0.05, -0.01).endswith("NO-CONTACT-SHOWN") and c(-0.01, 0.05).endswith("NO-CONTACT-SHOWN")
    f = lambda lo, hi: RE.names_of("S", iv(0, 0), iv(lo, hi), iv(0, 0))["S_PRICE"]                          # noqa: E731
    assert f(0.001, 0.1).endswith("FALLS-BEHIND-MORE") and f(0.0, 0.1).endswith("FALLING-BEHIND-NOT-SEPARATED") and f(-0.1, -0.001).endswith("FALLS-BEHIND-LESS")
    assert f(-0.1, 0.0).endswith("FALLING-BEHIND-NOT-SEPARATED") and f(-0.1, 0.1).endswith("FALLING-BEHIND-NOT-SEPARATED")
    k = lambda lo, hi: RE.names_of("F", iv(0, 0), iv(0, 0), iv(lo, hi))["F_K100"]                           # noqa: E731
    assert k(-0.02, -0.0001).endswith("LESS-CONTACT-THAN-K100") and k(-0.02, 0.0).endswith("CONTACT-NOT-SEPARATED-FROM-K100") and k(0.0, 0.02).endswith("CONTACT-NOT-SEPARATED-FROM-K100")
    assert k(0.0001, 0.02).endswith("MORE-CONTACT-THAN-K100") and k(-0.02, 0.02).endswith("CONTACT-NOT-SEPARATED-FROM-K100")


def payload(carry=("W",), more=()):
    pr = {}
    for a in RE.ARMS:
        pr[a + "-VS-RR"] = {k: {"interval": iv(-0.05, -0.01), "pooled": -0.03} for k in ("C", "U", "F")}
        pr[a + "-VS-RLc"] = {"C": {"interval": iv(0.01, 0.06) if a in carry else iv(-0.01, 0.03), "pooled": 0.02}, "U": {"interval": iv(-0.02, 0.03), "pooled": 0.0},
                             "F": {"interval": iv(0.01, 0.05) if a in more else iv(-0.02, 0.03), "pooled": 0.0}}
        pr[a + "-VS-K100"] = {k: {"interval": iv(-0.02, 0.01), "pooled": -0.005} for k in ("C", "U", "F")}
    return {"levels": {"RR": {"C": 0.0954}, "RLc": {"C": 0.0}, "K100": {"C": 0.0477}}, "pairs": pr, "families": {}, "strategic": {"pairs": {}}}


def test_card_name_rule_cases():
    assert RE.all_names(payload(carry=("W",)))["CARD"] == "ERRPART10K-THE-PART-WITHIN-THE-CLIP-CARRIES-IT"
    assert RE.all_names(payload(carry=("M",)))["CARD"] == "ERRPART10K-THE-CLIP-PART-CARRIES-IT"
    assert RE.all_names(payload(carry=("M", "W")))["CARD"] == "ERRPART10K-BOTH-PARTS-CARRY-IT"
    assert RE.all_names(payload(carry=()))["CARD"] == "ERRPART10K-NO-PART-SHOWN"
    assert RE.all_names(payload(carry=("S", "F")))["CARD"] == "ERRPART10K-NO-PART-SHOWN"                    # S and F carry contact: the name of the card reads M and W only
    assert RE.all_names(payload(carry=("W", "S"), more=("W",)))["CARD"] == "ERRPART10K-THE-PART-WITHIN-THE-CLIP-CARRIES-IT"
    n = RE.all_names(payload(carry=("S",)))
    assert len(n) == 13 and set(k.split("_")[0] for k in n) == set(RE.ARMS) | {"CARD"} and n["S_CONTACT"].endswith("-S-CARRIES-CONTACT") and n["F_CONTACT"].endswith("-F-NO-CONTACT-SHOWN")


def test_a_card_name_that_the_intervals_do_not_give_turns_the_check_red():
    p = payload(carry=("W",))
    assert RE.card_check(p, "ERRPART10K-THE-CLIP-PART-CARRIES-IT")["PASS"] is False
    assert RE.card_check(p, "ERRPART10K-BOTH-PARTS-CARRY-IT")["PASS"] is False
    assert RE.card_check(p, "ERRPART10K-NO-PART-SHOWN")["PASS"] is False
    assert RE.card_check(p, "ERRPART10K-THE-PART-WITHIN-THE-CLIP-CARRIES-IT")["PASS"] is True
    p = payload(carry=())
    assert RE.card_check(p, "ERRPART10K-THE-PART-WITHIN-THE-CLIP-CARRIES-IT")["PASS"] is False and RE.card_check(p, "ERRPART10K-NO-PART-SHOWN")["PASS"] is True
    p = payload(carry=("M", "W"))
    assert RE.card_check(p, RE.all_names(p)["CARD"])["PASS"] is True and RE.card_check(p, "ERRPART10K-THE-CLIP-PART-CARRIES-IT")["PASS"] is False


def test_controls_q2_q3_q4_q5():
    assert RE.ctl_q2("x\nZZ CHECK-WRAPPER-J PASS\n", "cbf7063d867c9f4e a.py", "cbf7063d867c9f4e")["PASS"]
    assert not RE.ctl_q2("ZZ CHECK-WRAPPER-J FAIL J2A\n", "cbf7063d867c9f4e", "cbf7063d867c9f4e")["PASS"]
    assert not RE.ctl_q2("ZZ CHECK-WRAPPER-J PASS\n", "0123456789abcdef", "cbf7063d867c9f4e")["PASS"]
    assert RE.ctl_q3(0.09541, 0.0, 0.04769)["PASS"] and RE.ctl_q3(0.09536, 0.00004, 0.04774)["PASS"]
    assert not RE.ctl_q3(0.0960, 0.0, 0.0477)["PASS"] and not RE.ctl_q3(0.0954, 0.0138, 0.0477)["PASS"] and not RE.ctl_q3(None, 0.0, 0.0477)["PASS"]
    assert not RE.ctl_q3(0.0954, 0.0, 0.0462)["PASS"] and not RE.ctl_q3(0.0954, 0.0, None)["PASS"] and not RE.ctl_q3(0.0954, 0.0, 0.0954)["PASS"]
    assert RE.ctl_q4("25 passed in 3s")["PASS"] and not RE.ctl_q4("2 failed, 3 passed")["PASS"] and not RE.ctl_q4("")["PASS"]
    assert RE.ctl_q5(True, True, True)["PASS"] and not RE.ctl_q5(False, True, True)["PASS"] and not RE.ctl_q5(True, False, True)["PASS"] and not RE.ctl_q5(True, True, False)["PASS"]


ALL_PASS = {c: {"PASS": True} for c in RE.CONTROLS}


def test_name_written_with_a_control_failed_turns_red(tmp_path):
    ctl = RE.assemble_controls(dict(ALL_PASS, q1={"PASS": False}), payload())
    assert ctl["controls_all_pass"] is False and "payload" not in ctl and "withheld" in ctl
    f = tmp_path / "controls.json"
    f.write_text(json.dumps(ctl), encoding="utf-8")
    out = tmp_path / "result.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 3 and not out.exists()
    f2 = tmp_path / "forged.json"
    f2.write_text(json.dumps(dict(ctl, controls_all_pass=True, payload=payload())), encoding="utf-8")        # an edited flag does not open the names: the controls themselves are re-checked
    assert RE.main(["--stage", "names", "--controls", str(f2), "--out", str(tmp_path / "r2.json")]) == 3
    f3 = tmp_path / "missing.json"
    short = {k: v for k, v in ALL_PASS.items() if k != "q5"}                                                   # a controls file without one control does not open the names
    f3.write_text(json.dumps(dict(RE.assemble_controls(short, payload()))), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(f3), "--out", str(tmp_path / "r3.json")]) == 3 and not (tmp_path / "r3.json").exists()
    for c in RE.CONTROLS:                                                                                      # each one control failed opens nothing, no map of arms to controls
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
    assert r["controls_sha256"] == RE.sha_file(str(f)) and len(r["names"]) == 13 and r["card"] == "PREREG_V8F_PROOF_ADD98"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 2             # refuses an existing output


def test_contact_over_k100_and_the_printed_sections():
    p = payload()
    p["levels"].update({a: {"C": 0.0477 + 0.01} for a in RE.ARMS})
    s = RE.over_k100_rows(p["levels"], p["pairs"])
    assert set(s) == set(RE.ARMS) and abs(s["M"]["difference_over_K100"] - (-0.005 / 0.0477)) < 1e-12 and abs(s["M"]["interval_of_difference_over_K100"][1] - 0.01 / 0.0477) < 1e-12
    assert abs(s["M"]["ratio_to_K100"] - (0.0477 - 0.005) / 0.0477) < 1e-12
    assert RE.over_k100_rows({"K100": {"C": 0.0}}, p["pairs"])["M"] is None
    sec = RE.print_arm_sections(p)
    assert len(sec) == 4 and all("vs RR" in x and "vs RLc" in x and "vs K100" in x and "strategic" in x for x in sec)


def test_reader_source_carries_the_strategic_family_the_k100_reference_and_the_engine_bootstrap_setting():
    t = (HERE / "read_errpart10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and "R.strategic_grid(" in t and "S.pair_read(" in t and "R.family_of(" in t
    assert 'CARD, PRE = "PREREG_V8F_PROOF_ADD98", "ERRPART10K"' in t and '"RLcKF5P30Hb20q" + a' in t and 'ARMS = ("M", "W", "S", "F")' in t
    assert 'GAINS = {"K100": "RLcKF5P30Hb20mK100"}' in t and "runs.update(GAINS)" in t and 'pairs_def[x + "-VS-K100"]' in t and "R.ALL_RUNS.setdefault(r," in t
    assert 'controls = {"q0"' in t and '"q5": ctl_q5(' in t and '"q3": ctl_q3(' in t and "differences_against_K100" in t and "contact_over_k100" in t and "ladder" not in t.lower()
    assert 'for ref in ("RR", "RLc"):' in t and "ZZ READ-ERRPART10K" in t and "ZZ SELFTEST-READ-ERRPART10K" in t


def test_reader_selftest_runs():
    assert RE.selftest() == 0


def test_podread_text():
    t = (HERE / "podread_errpart10k.sh").read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=-1" in t and "GP=$V/errpart10k" in t and "OUT=$GP/read" in t and "CHK=$V/gaprule10kb/check_j2" in t
    assert "ERRPART10K_CONTROLS.json" in t and "ERRPART10K_RESULT.json" in t and "ERRPART10K_DONE_s0.txt" in t and "ERRPART10K_DONE_s1.txt" in t and "lead_pytest.out" in t
    assert "joins/PART_JOINS_RECORD.json" in t and "I10RLcKF5P30Hb20q${x}_s$s" in t and "for x in M W S F; do" in t and "read_errpart10k.py" in t and "q[0-5]" in t
    assert "ZZ PODREAD-ERRPART10K" in t
    assert 'for d in "$W/t1/I10RLcKF5P30Hb20mK100_s0" "$W/t1/I10RLcKF5P30Hb20mK100_s1"; do' in t and '[ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"' in t
    assert "ERRCUT" not in t and "errcut" not in t and "SMOOTH" not in t and "smoothgap" not in t
    assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_errpart10k.sh").replace("\\", "/")]).returncode == 0


def test_last_lines_of_the_tools():
    r = subprocess.run([sys.executable, str(HERE / "read_errpart10k.py"), "--selftest"], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.stdout.strip().splitlines()[-1] == "ZZ SELFTEST-READ-ERRPART10K ok: True" and r.returncode == 0


def test_texts_hold_no_banned_words():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    n = 0
    for f in ("make_part_joins.py", "make_errpart_queue.py", "pod_errpart10k_queue.sh", "read_errpart10k.py", "podread_errpart10k.sh", "test_errpart.py", "ERRPART_DEFS.md"):
        p = HERE / f
        if p.exists():
            n += 1
            t = p.read_text(encoding="utf-8").lower()
            hit = [w for w in bad if w in t]
            assert not hit, (f, hit)
    assert n == 7
    names = [p.name for p in HERE.iterdir() if p.is_file()]
    assert not any(re.fullmatch(r"(aux|con|prn|nul)(\..*)?", x, re.I) or x.lower().endswith("token.json") for x in names)
