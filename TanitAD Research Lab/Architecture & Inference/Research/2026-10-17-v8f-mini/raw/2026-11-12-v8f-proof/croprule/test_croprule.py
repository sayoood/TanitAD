"""test_croprule.py -- tests of the kit of card ADD101 (CROPRULE10K): joins with the pilot's gap, queue, reader controls and names, with planted faults. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest croprule/test_croprule.py -q -p no:cacheprovider --basetemp croprule/tmp_pt/<new name>"""
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
import make_crop_joins as MCJ                # noqa: E402
import read_croprule10k as RE                 # noqa: E402

QUEUE = HERE / "pod_croprule10k_queue.sh"
SG_QUEUE = HERE.parent / "smoothgap" / "pod_smoothgap10k_queue.sh"
ERR_QUEUE = HERE.parent / "errcut" / "pod_errcut10k_queue.sh"
MODEL_QUEUE = HERE.parent / "gaprule" / "pod_gaprule10k_queue.sh"
ARMS = ("KB5", "KB1", "KD5", "KD1")
NAMES8 = ["%s_s%d" % (a, s) for a in ARMS for s in (0, 1)]


# ------------------------------------------------------------------------------------------------ joins: q0 and q1 and their planted faults
@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    base = tmp_path_factory.mktemp("cr")
    J, dumps = MCJ.synthetic_all()
    rec = MCJ.build(J, dumps, str(base / "clean"), meta=MCJ.synthetic_meta())
    return {"J": J, "dumps": dumps, "clean": rec, "base": base}


def fault_build(mix, fault):
    cache = mix.setdefault("fault_cache", {})
    if fault not in cache:
        cache[fault] = MCJ.build(mix["J"], mix["dumps"], str(mix["base"] / ("f_" + fault)), faults=(fault,), meta=MCJ.synthetic_meta())
    return cache[fault]


def saved(mix, name):
    with np.load(str(mix["base"] / "clean" / (name + ".npz")), allow_pickle=False) as z:
        return z["lead_gap"]


def row_of(J, c, f):
    names = [str(x) for x in J["clip"]]
    return int(J["off"][names.index(c)]) + f


def test_clean_build_passes_q0_and_q1(mix):
    r = mix["clean"]
    assert r["q0"]["PASS"], r["q0"]
    assert r["q1"]["PASS"], r["q1"]
    assert sorted(r["arms"]) == sorted(NAMES8) and r["card"] == "PREREG_V8F_PROOF_ADD101" and sorted(r["dumps"]) == sorted(NAMES8)
    for nm, a in r["arms"].items():
        assert a["max_abs_dev_saved_minus_second_routine"] <= 1e-4 and a["other_keys_all_equal"] is True and a["n_changed_outside_mixed_frames"] == 0
        assert a["lead_valid_unchanged"] is True and a["n_label_frames_without_output_not_label_gap"] == 0 and a["n_with_pilot_output"] > 0
        assert a["n_with_pilot_output"] + a["n_without_pilot_output"] == a["n_valid_label_gap"] and a["dump_dir"] == MCJ.ARM_DIR[a["arm"]]
        assert a["error_statistics_pilot_minus_label"]["n_frames"] == a["n_with_pilot_output"]
    assert len({r["arms"][n]["sha256"] for n in r["arms"]}) == 8          # eight different joins (no pair is shared)


def test_join_carries_the_pilot_gap_as_dumped_by_hand(mix):
    """by hand, from the dump arrays: on a frame with a valid label gap and an output the join holds the dumped gap (float32), nothing else is applied"""
    J = mix["J"]
    vm = MCJ.valid_mask(J)
    for arm, s in (("KB5", 0), ("KD1", 1), ("KB1", 1), ("KD5", 0)):
        D = mix["dumps"][(arm, s)]
        g = saved(mix, "%s_s%d" % (arm, s))
        n_hit = 0
        for i in range(0, len(D["frame"]), 7):
            c, f = str(D["clips"][int(D["cid"][i])]), int(D["frame"][i])
            row = row_of(J, c, f)
            if vm[row] and bool(D["valid"][i]) and np.isfinite(D["gap"][i]):
                assert float(g[row]) == float(np.float32(D["gap"][i])), (arm, s, c, f)          # no map, no minimum, no smoothing
                n_hit += 1
            elif vm[row]:
                assert float(g[row]) == float(np.float32(J["lead_gap"][row]))                    # no output: the label gap
            else:
                assert (np.isnan(g[row]) and np.isnan(J["lead_gap"][row])) or float(g[row]) == float(J["lead_gap"][row])
        assert n_hit > 20


def test_the_model_seed_reads_its_own_fit_seed_and_the_arm_its_own_folder(mix):
    for arm in ARMS:
        assert not MCJ.equal_key(saved(mix, arm + "_s0"), saved(mix, arm + "_s1"))
    for s in (0, 1):
        for a, b in (("KB5", "KB1"), ("KB5", "KD5"), ("KD5", "KD1")):
            assert not MCJ.equal_key(saved(mix, "%s_s%d" % (a, s)), saved(mix, "%s_s%d" % (b, s)))
    assert MCJ.ARM_DIR == {"KB5": "B_H5", "KB1": "B_H1", "KD5": "D_H5", "KD1": "D_H1"}
    m = MCJ.synthetic_meta()
    assert m[("KD1", 1)]["dump_file"].endswith("/D_H1/reader_s1_score.npz")


def test_each_planted_fault_turns_q1_red_and_q0_stays_green(mix):
    for fault, word in (("other_seed", "second time"), ("other_head", "second time"), ("other_kind", "second time"), ("minimum", "second time"), ("smoothed", "second time"),
                        ("shifted", "second time"), ("scaled", "second time"), ("outside_mix", "another field or a frame outside"), ("no_output_zero", "do not carry the label gap"),
                        ("other_field", "another field")):
        r = fault_build(mix, fault)
        assert not r["q1"]["PASS"], fault
        assert any(word in x for x in r["q1"]["failures"]), (fault, r["q1"]["failures"][:3])
        assert r["q0"]["PASS"], fault
    assert any("lead marking" in x for x in fault_build(mix, "other_field")["q1"]["failures"])


def test_planted_q0_faults(mix):
    J, dumps = mix["J"], mix["dumps"]

    def run(label, key, fn, word):
        d = dict(dumps)
        d[key] = fn(dict(dumps[key]))
        r = MCJ.build(J, d, str(mix["base"] / ("q0_" + label)), meta=MCJ.synthetic_meta())
        assert not r["q0"]["PASS"], label
        assert any(word in x for x in r["q0"]["failures"]), (label, r["q0"]["failures"][:3])
        assert "%s_s%d" % key in " ".join(r["q0"]["failures"])
    cols = ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")

    def lab(D):
        lg = np.array(D["lab_gap"], copy=True)
        lg[np.flatnonzero(np.isfinite(lg))[7]] += 0.01
        D["lab_gap"] = lg
        return D

    def cidx(D):
        D["clip_index"] = (np.asarray(D["clip_index"]) + 1) % len(J["n"])
        return D

    def beyond(D):
        for k in cols:
            D[k] = np.concatenate([np.asarray(D[k]), np.asarray(D[k])[-1:]])
        D["frame"][-1] = 10 ** 6
        return D

    def dropped(D):
        keep = np.ones(len(D["frame"]), bool)
        keep[3:6] = False
        for k in cols:
            D[k] = np.asarray(D[k])[keep]
        return D

    def dup(D):
        for k in cols:
            D[k] = np.concatenate([np.asarray(D[k]), np.asarray(D[k])[10:12]])
        return D

    def seed_(D):
        D["seed"] = np.int64(1)
        return D
    run("label", ("KB5", 0), lab, "label gap difference")
    run("cidx", ("KB1", 1), cidx, "n_clip_index_mismatch")
    run("beyond", ("KD5", 0), beyond, "n_rows_without_join_row")
    run("dropped", ("KD1", 1), dropped, "n_frame_count_mismatch_clips")
    run("dup", ("KD1", 0), dup, "n_duplicate_keys")
    run("seed", ("KB5", 0), seed_, "records seed")
    d = dict(dumps)
    d[("KB5", 0)] = dict(dumps[("KB5", 0)])
    d[("KB5", 0)]["clips"] = np.asarray([x + "x" for x in d[("KB5", 0)]["clips"]])
    r = MCJ.build(J, d, str(mix["base"] / "q0_names"), meta=MCJ.synthetic_meta())
    assert not r["q0"]["PASS"] and any("n_names_not_in_join" in x for x in r["q0"]["failures"])


def test_frames_without_a_pilot_output_carry_the_label_gap_and_are_counted_with_no_bar(tmp_path):
    J, dumps = MCJ.synthetic_all()
    d = dict(dumps)
    D = dict(dumps[("KB5", 1)])
    ok = np.array(D["valid"], copy=True)
    ok[np.flatnonzero(ok & np.asarray(D["lab_present"], bool))[::2]] = False                # half of the label frames lose their output
    D["valid"] = ok
    d[("KB5", 1)] = D
    rec = MCJ.build(J, d, str(tmp_path / "j"), meta=MCJ.synthetic_meta())
    a = rec["arms"]["KB5_s1"]
    assert rec["q0"]["PASS"] and rec["q1"]["PASS"], rec["q1"]
    assert a["fraction_without_pilot_output"] > 0.4 and a["n_without_pilot_output"] > 100 and a["n_label_frames_without_output_not_label_gap"] == 0
    with np.load(str(tmp_path / "j" / "KB5_s1.npz"), allow_pickle=False) as z:
        sv = np.asarray(z["lead_gap"], np.float64)
    tab = MCJ.frame_table(J, D)
    m = np.flatnonzero(np.asarray(D["lab_present"], bool) & ~MCJ.pilot_read_mask(D, tab))
    r = tab["row"][m[:50]]
    r = r[r >= 0]
    assert len(r) > 0 and all(float(sv[x]) == float(np.float32(J["lead_gap"][x])) for x in r)


def test_values_below_0_and_above_60_are_counted_and_left_as_they_are(tmp_path):
    J, dumps = MCJ.synthetic_all()
    D = dict(dumps[("KD5", 0)])
    g = np.array(D["gap"], np.float64)
    ok = np.flatnonzero(np.asarray(D["valid"], bool) & np.isfinite(g) & np.asarray(D["lab_present"], bool))
    g[ok[:7]] = -4.0
    g[ok[10:14]] = 300.0
    D["gap"] = g
    d = dict(dumps)
    d[("KD5", 0)] = D
    rec = MCJ.build(J, d, str(tmp_path / "joins"), meta=MCJ.synthetic_meta())
    a = rec["arms"]["KD5_s0"]
    assert a["n_mixed_above_60"] >= 1 and a["n_mixed_below_0"] >= 1 and a["min_mixed"] <= -4.0 + 1e-6 and a["max_mixed"] >= 300.0 - 1e-3
    assert rec["q0"]["PASS"] and rec["q1"]["PASS"]


def test_q1_limits_and_record_fields(mix):
    c = mix["clean"]

    def with_(nm, **kw):
        r = copy.deepcopy(c)
        r["arms"][nm].update(kw)
        return r
    assert MCJ.q1_check(with_("KB5_s0", fraction_without_pilot_output=0.9, n_without_pilot_output=500))["PASS"] is True          # counted, no bar
    assert MCJ.q1_check(with_("KB5_s0", fraction_without_pilot_output=None))["PASS"] is False
    assert MCJ.q1_check(with_("KD1_s1", n_mixed_below_0=None))["PASS"] is False
    assert MCJ.q1_check(with_("KD1_s1", n_mixed_above_60=None))["PASS"] is False
    assert MCJ.q1_check(with_("KD1_s1", n_mixed_above_60=5, max_mixed=70.0))["PASS"] is True
    assert MCJ.q1_check(with_("KB1_s0", n_changed_outside_mixed_frames=1))["PASS"] is False
    assert MCJ.q1_check(with_("KB1_s0", n_label_frames_without_output_not_label_gap=2))["PASS"] is False
    assert MCJ.q1_check(with_("KB1_s0", lead_valid_unchanged=False))["PASS"] is False
    assert MCJ.q1_check(with_("KB1_s0", arm="KD1"))["PASS"] is False
    assert MCJ.q1_check(with_("KD5_s1", dump_dir="B_H5"))["PASS"] is False
    assert MCJ.q1_check(with_("KD5_s1", max_abs_dev_saved_minus_second_routine=1.1e-4))["PASS"] is False
    assert MCJ.q1_check(with_("KD5_s1", max_abs_dev_saved_minus_second_routine=1e-4))["PASS"] is True
    assert MCJ.q1_check(with_("KD5_s1", n_with_pilot_output=0))["PASS"] is False
    assert MCJ.q1_check(with_("KD5_s1", lead_gap_dtype_kept=False))["PASS"] is False
    r = copy.deepcopy(c)
    r["dumps"]["KD5_s1"]["dump_file"] = "/x/D_H5/reader_s0_score.npz"
    assert MCJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["dumps"]["KD5_s1"]["dump_dir_name"] = "B_H5"
    assert MCJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["dumps"]["KD5_s1"]["dump_sha256"] = "ab"
    assert MCJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["arms"].pop("KD1_s1")
    assert MCJ.q1_check(r)["PASS"] is False
    r = copy.deepcopy(c)
    r["card"] = "PREREG_V8F_PROOF_ADD97"
    assert MCJ.q1_check(r)["PASS"] is False


def test_q0_record_fields(mix):
    r = copy.deepcopy(mix["clean"])
    r["dumps"]["KB1_s0"]["e0"]["max_abs_diff"] = 1.1e-4
    assert MCJ.q0_check(r)["PASS"] is False
    r["dumps"]["KB1_s0"]["e0"]["max_abs_diff"] = 1e-4
    assert MCJ.q0_check(r)["PASS"] is True
    for key in ("n_duplicate_keys", "n_rows_without_join_row", "n_frame_count_mismatch_clips", "n_clip_index_mismatch"):
        r["dumps"]["KD1_s1"]["e0"][key] = 1
        assert MCJ.q0_check(r)["PASS"] is False, key
        r["dumps"]["KD1_s1"]["e0"][key] = 0
    r["dumps"]["KD1_s1"]["e0"]["n_join_clips_not_in_dump"] = 3             # counted, no bar
    assert MCJ.q0_check(r)["PASS"] is True
    r["dumps"].pop("KD1_s1")
    assert MCJ.q0_check(r)["PASS"] is False


def test_out_dir_refused_and_arguments_required(mix):
    with pytest.raises(SystemExit) as ex:
        MCJ.build(mix["J"], mix["dumps"], str(mix["base"] / "clean"))
    assert "REFUSED" in str(ex.value)
    with pytest.raises(SystemExit) as ex:
        MCJ.main(["--eval-join", "x.npz", "--out-dir", str(mix["base"] / "o")])
    assert "REFUSED" in str(ex.value)


def test_main_through_files(tmp_path):
    J, dumps = MCJ.synthetic_all()
    root = tmp_path / "dump"
    np.savez(str(tmp_path / "join.npz"), **J)
    for a in ARMS:
        (root / MCJ.ARM_DIR[a]).mkdir(parents=True)
        for s in (0, 1):
            np.savez(str(root / MCJ.ARM_DIR[a] / ("reader_s%d_score.npz" % s)), **dumps[(a, s)])
    assert MCJ.main(["--eval-join", str(tmp_path / "join.npz"), "--dump-root", str(root), "--out-dir", str(tmp_path / "out")]) == 0
    rec = json.loads((tmp_path / "out" / "CROP_JOINS_RECORD.json").read_text(encoding="utf-8"))
    assert rec["q0"]["PASS"] and rec["q1"]["PASS"] and rec["eval_join_sha256"] == MCJ.sha_file(str(tmp_path / "join.npz"))
    assert rec["dumps"]["KD1_s1"]["dump_sha256"] == MCJ.sha_file(str(root / "D_H1" / "reader_s1_score.npz"))
    ok = RE.ctl_q1(rec, str(tmp_path / "out"), rec["eval_join_sha256"])
    assert ok["PASS"], ok
    assert RE.ctl_q0(rec, rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_q1(rec, str(tmp_path / "out"), "0" * 64)["PASS"]
    assert not RE.ctl_q1(rec, str(tmp_path / "nowhere"), rec["eval_join_sha256"])["PASS"]
    assert not RE.ctl_q0(rec, "0" * 64)["PASS"]
    assert not RE.ctl_q0(None)["PASS"] and not RE.ctl_q1(None, "x")["PASS"]


# ------------------------------------------------------------------------------------------------ queue (text checks)
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def queue_static_facts(t):
    f = fn_text(t)
    outside = t.replace(f, "")
    return [("scoring gets --eval-join \"$JOINE\" (and the DRY print --eval-join $JOINE)", f.count('--eval-join "$JOINE"') == 1 and f.count("--eval-join $JOINE") == 1),
            ("no other --eval-join anywhere", t.count("--eval-join") == 2 + len(re.findall(r"#.*--eval-join", t))),
            ("JOINE is assigned once, to the pinned evaluation join", len(re.findall(r"^JOINE=", t, re.M)) == 1 and "JOINE=$V/hierrep/pins/join_eval141.npz" in t),
            ("--rule-join once, as extra+=(--rule-join \"$RJOIN\")", f.count("--rule-join") == 1 and 'extra+=(--rule-join "$RJOIN")' in f),
            ("the rule join is named outside the rollout function only in comments or as RJOIN= before a rollout call",
             all(x.lstrip().startswith("#") or ("RJOIN=" in x and " rollout " in x) or "$PJ" not in x or "PJ=" in x or "admit" in x or "NO-CROP" in x for x in outside.split("\n") if "RJOIN" in x or "$PJ" in x)),
            ("no variable of the rule join in the scoring arguments", "$RJOIN" not in f.split("--eval-join")[1].split("\n")[0] and "$PJ" not in f)]


def test_rollout_function_is_byte_identical_to_the_earlier_queues():
    a = fn_text(QUEUE.read_text(encoding="utf-8"))
    assert a == fn_text(SG_QUEUE.read_text(encoding="utf-8")) == fn_text(MODEL_QUEUE.read_text(encoding="utf-8")) == fn_text(ERR_QUEUE.read_text(encoding="utf-8")) and len(a) > 3000


def test_queue_edits_and_names():
    t = QUEUE.read_text(encoding="utf-8")
    assert "GP=${GR_ROOT:-/workspace/v8proof/croprule10k}" in t and "CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}" in t
    assert "START_CROPRULE10K$PS.txt" in t and "CROPRULE10K_DONE$PS.txt" in t and "CROPRULE10K_STOP$PS.txt" in t and "queue_croprule$PS.log" in t and 'PS="_s$SEEDSW"' in t
    assert "cbf7063d867c9f4e" in t and t.count("cbf7063d867c9f4e") >= 3 and "CROP_JOINS_RECORD.json" in t and "SMOOTH" not in t and "MIX_JOINS" not in t
    assert "GAPRULE" not in t and "planted" not in t.lower() and "ERRCUT" not in t and "errcut" not in t
    for arm in ARMS:
        assert ('RJOIN="$PJ/%s_s$k.npz" rollout I10RLcKF5P30Hb20p%s RLc "$k" "$k"' % (arm, arm)) in t
    assert t.count("rollout I10RLcKF5P30Hb20p") == 4 and "rollout I10RLcKF5P30Hb20m" not in t
    adm = t[t.index("admit() {"):t.index("\n}\n", t.index("admit() {"))]
    assert 'rec[c].get("PASS") is not True' in adm and "eval_join_sha256" in adm and "sha256" in adm and '"q0", "q1"' in adm and '"e0"' not in adm
    assert 'admit "$PJ/CROP_JOINS_RECORD.json" "$PJ" "$JOINE" "KB5_s0,KB5_s1,KB1_s0,KB1_s1,KD5_s0,KD5_s1,KD1_s0,KD1_s1"' in t
    assert 'I10RLcKF5P30Hb20mK100_s$k/T1_DONE" ] || stop "NO-BANKED-K100 s$k"' in t
    assert 'NAMES="I10RLcKF5P30Hb20pKB5 I10RLcKF5P30Hb20pKB1 I10RLcKF5P30Hb20pKD5 I10RLcKF5P30Hb20pKD1"' in t
    assert "28" in t and " rm " not in t and " mv " not in t and " kill " not in t


def test_queue_is_what_the_maker_writes():
    import hashlib
    src = (HERE.parent / "smoothgap" / "pod_smoothgap10k_queue.sh").read_bytes()
    assert hashlib.sha256(src).hexdigest()[:16] == "cef2a3414b240499"
    mk = (HERE / "make_croprule_queue.py").read_text(encoding="utf-8")
    assert 'SRC_SHA_PREFIX = "cef2a3414b240499"' in mk and "CR_OVERWRITE_OWN" in mk


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
    mine, model, prev = dry_commands(QUEUE), dry_commands(MODEL_QUEUE), dry_commands(SG_QUEUE)

    def norm(c):
        c = re.sub(r"--rule-join \S+ ", "", c)
        return re.sub(r"I10RLcKF5P30Hb20[jmp](G0|ES25|E|S50|S25|K100|K50|K25|K12|KB5|KB1|KD5|KD1)", "RUN", c)
    assert len(mine) == 8 and len(model) == 12 and len(prev) == 8
    assert all("--rule-join" in c and "rzgfhqetj.py" in c and "--form RLc" in c and "--kappa-max 0.2" in c and "--preroll-steps 30" in c for c in mine)
    assert all("--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c for c in mine)
    g0 = [norm(c) for c in model if "jG0" in c]
    assert len(g0) == 2
    for arm in ARMS:
        for s in (0, 1):
            cs = [c for c in mine if ("p%s_s%d " % (arm, s)) in c or ("p%s_s%d/" % (arm, s)) in c]
            assert cs and all(("/%s_s%d.npz" % (arm, s)) in c for c in cs)
            assert all(norm(c) in g0 for c in cs)
    assert sorted(norm(c) for c in mine)[0] == sorted(norm(c) for c in prev)[0]          # the same command as the one of card ADD97 after the arm and the join


# ------------------------------------------------------------------------------------------------ reader: names, controls, binding
def iv(lo, hi):
    return [lo, hi]


def test_name_rules_on_planted_intervals():
    n = RE.names_of("KB5", iv(-0.05, -0.01), iv(0.0, 0.004), iv(-0.02, 0.03), iv(-0.03, -0.001), 0.0954, 0.0)         # H = 0.0477
    assert n["KB5_CONTACT"] == "CROPRULE10K-KB5-CUTS-CONTACT" and n["KB5_HALF"] == "CROPRULE10K-KB5-WITHIN-HALF-THE-GAP"
    assert n["KB5_PRICE"] == "CROPRULE10K-KB5-FALLING-BEHIND-NOT-SEPARATED" and n["KB5_K100"] == "CROPRULE10K-KB5-LESS-CONTACT-THAN-K100"
    c = lambda lo, hi: RE.names_of("KD1", iv(lo, hi), iv(0, 0), iv(0, 0), iv(0, 0), 0.0954, 0.0)["KD1_CONTACT"]                 # noqa: E731
    assert c(-0.02, -0.0001).endswith("-CUTS-CONTACT") and c(-0.02, 0.0).endswith("-CONTACT-NOT-SEPARATED-FROM-RR") and c(0.0, 0.02).endswith("-CONTACT-NOT-SEPARATED-FROM-RR")
    assert c(0.0001, 0.02) == "CROPRULE10K-KD1-MORE-CONTACT" and c(-0.02, 0.02).endswith("-CONTACT-NOT-SEPARATED-FROM-RR")
    k = lambda lo, hi: RE.names_of("KB1", iv(-0.01, 0.0), iv(0.02, 0.05), iv(0.0, 0.1), iv(lo, hi), 0.0954, 0.0)["KB1_K100"]      # noqa: E731
    assert k(-0.02, -0.0001).endswith("LESS-CONTACT-THAN-K100") and k(-0.02, 0.0).endswith("CONTACT-NOT-SEPARATED-FROM-K100") and k(0.0, 0.02).endswith("CONTACT-NOT-SEPARATED-FROM-K100")
    assert k(0.0001, 0.02).endswith("MORE-CONTACT-THAN-K100") and k(-0.02, 0.02).endswith("CONTACT-NOT-SEPARATED-FROM-K100")
    h = lambda lo, hi: RE.names_of("KB1", iv(-0.01, 0.0), iv(lo, hi), iv(0.0, 0.1), iv(0, 0), 0.0954, 0.0)["KB1_HALF"]           # noqa: E731
    assert h(0.02, 0.0477) == "CROPRULE10K-KB1-WITHIN-HALF-THE-GAP"                                                           # at H: within
    assert h(0.0478, 0.06) == "CROPRULE10K-KB1-HALF-THE-GAP-NOT-REACHED" and h(0.0477, 0.06).endswith("HALF-THE-GAP-NOT-DECIDED") and h(0.02, 0.05).endswith("HALF-THE-GAP-NOT-DECIDED")
    p = lambda lo, hi: RE.names_of("KD5", iv(0, 0), iv(0, 0), iv(lo, hi), iv(0, 0), 0.0954, 0.0)["KD5_PRICE"]                   # noqa: E731
    assert p(0.001, 0.1).endswith("-FALLS-BEHIND-MORE") and p(-0.1, -0.001).endswith("-FALLS-BEHIND-LESS") and p(-0.1, 0.0).endswith("-FALLING-BEHIND-NOT-SEPARATED")
    assert p(0.0, 0.1).endswith("-FALLING-BEHIND-NOT-SEPARATED")


def payload(rr=0.0954, rlc=0.0, kb5=(0.0, 0.004), kd5=(0.0, 0.004)):
    pr = {}
    for a in RE.ARMS:
        hi = {"KB5": kb5, "KD5": kd5}.get(a, (0.0, 0.06))
        pr[a + "-VS-RR"] = {"C": {"interval": iv(-0.05, -0.01), "pooled": -0.03}, "U": {"interval": iv(-0.05, -0.01), "pooled": -0.03}, "F": {"interval": iv(-0.02, 0.03), "pooled": 0.0}}
        pr[a + "-VS-RLc"] = {k: {"interval": iv(*hi), "pooled": 0.002} for k in ("C", "U", "F")}
        pr[a + "-VS-K100"] = {k: {"interval": iv(-0.02, 0.01), "pooled": -0.005} for k in ("C", "U", "F")}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}, "K100": {"C": 0.0477}}, "pairs": pr, "families": {}, "strategic": {"pairs": {}}}


def test_card_names_come_from_kb5_and_kd5():
    n = RE.all_names(payload())
    assert n["CARD_BOUND"] == "CROPRULE10K-BOUND-WITHIN-HALF-THE-GAP" and n["CARD_FIXED"] == "CROPRULE10K-FIXED-WITHIN-HALF-THE-GAP"
    n = RE.all_names(payload(kb5=(0.055, 0.07), kd5=(0.0, 0.07)))
    assert n["CARD_BOUND"] == "CROPRULE10K-BOUND-HALF-THE-GAP-NOT-REACHED" and n["CARD_FIXED"] == "CROPRULE10K-FIXED-HALF-THE-GAP-NOT-DECIDED"
    n = RE.all_names(payload(kb5=(0.0, 0.07), kd5=(0.06, 0.08)))
    assert n["CARD_BOUND"] == "CROPRULE10K-BOUND-HALF-THE-GAP-NOT-DECIDED" and n["CARD_FIXED"] == "CROPRULE10K-FIXED-HALF-THE-GAP-NOT-REACHED"
    assert len(n) == 4 * 4 + 2 and set(k.split("_")[0] for k in n) == set(RE.ARMS) | {"CARD"}
    # the other two arms do not make a name of the card
    n = RE.all_names(payload(kb5=(0.055, 0.07), kd5=(0.055, 0.07)))
    assert n["KB1_HALF"].endswith("NOT-DECIDED") and "BOUND-HALF-THE-GAP-NOT-REACHED" in n["CARD_BOUND"]


def test_a_card_name_that_the_intervals_do_not_give_turns_the_check_red():
    p = payload(kb5=(0.055, 0.07))
    good = {"CARD_BOUND": "CROPRULE10K-BOUND-HALF-THE-GAP-NOT-REACHED", "CARD_FIXED": "CROPRULE10K-FIXED-WITHIN-HALF-THE-GAP"}
    assert RE.card_check(p, good)["PASS"] is True
    assert RE.card_check(p, dict(good, CARD_BOUND="CROPRULE10K-BOUND-WITHIN-HALF-THE-GAP"))["PASS"] is False
    assert RE.card_check(p, dict(good, CARD_FIXED="CROPRULE10K-FIXED-HALF-THE-GAP-NOT-DECIDED"))["PASS"] is False
    assert RE.card_check(p, {"CARD_BOUND": good["CARD_BOUND"]})["PASS"] is False


def test_h_note_is_reported_not_used():
    p = payload(kb5=(0.0, 0.04770), kd5=(0.0, 0.0477))
    assert RE.h_sensitivity(p)["differs"] is False
    p = payload(rr=0.0952, rlc=0.0, kb5=(0.0, 0.04765), kd5=(0.0, 0.0477))                  # H from the point values is 0.0476, the card prints 0.0477
    s = RE.h_sensitivity(p)
    assert s["differs"] is True and s["KD5"]["with_point_values"] == "NOT-DECIDED" and s["KD5"]["with_H_card"] == "WITHIN"
    assert RE.all_names(p)["CARD_FIXED"].endswith("HALF-THE-GAP-NOT-DECIDED")              # the name is made with the point values, as the rule of card ADD97 does


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
    assert r["controls_sha256"] == RE.sha_file(str(f)) and len(r["names"]) == 18 and r["card"] == "PREREG_V8F_PROOF_ADD101" and "h_note" in r
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(out)]) == 2             # refuses an existing output


def test_share_of_the_contact_gap_and_the_u_f_differences_against_k100():
    p = payload()
    s = RE.share_rows(p["levels"], p["pairs"])
    assert abs(s["KB5"]["share"] - 0.03 / 0.0954) < 1e-12 and set(s) == set(RE.ARMS)
    assert RE.print_arm_sections(p) and all("K100" in x for x in RE.print_arm_sections(p))


def test_reader_source_carries_the_strategic_family_the_k100_reference_and_the_engine_bootstrap_setting():
    t = (HERE / "read_croprule10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and t.index("a = ap.parse_args(argv)") < t.index("a.n_boot_control = 300") and "R.strategic_grid(" in t and "S.pair_read(" in t and "R.family_of(" in t
    assert 'CARD, PRE = "PREREG_V8F_PROOF_ADD101", "CROPRULE10K"' in t and '"RLcKF5P30Hb20p" + a' in t and 'ARMS = ("KB5", "KB1", "KD5", "KD1")' in t
    assert 'GAINS = {"K100": "RLcKF5P30Hb20mK100"}' in t and "runs.update(GAINS)" in t and 'pairs_def[x + "-VS-K100"]' in t and "R.ALL_RUNS.setdefault(r," in t
    assert 'controls = {"q0"' in t and '"q5": ctl_q5(' in t and '"q3": ctl_q3(' in t and "differences_against_K100" in t and "qmap" not in t.lower()
    assert 'for ref in ("RR", "RLc"):' in t and "ticks=(2,)" in t


def test_reader_selftest_runs():
    assert RE.selftest() == 0


def test_podread_text():
    t = (HERE / "podread_croprule10k.sh").read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=-1" in t and "GP=$V/croprule10k" in t and "OUT=$GP/read" in t and "CHK=$V/gaprule10kb/check_j2" in t
    assert "CROPRULE10K_CONTROLS.json" in t and "CROPRULE10K_RESULT.json" in t and "CROPRULE10K_DONE_s0.txt" in t and "CROPRULE10K_DONE_s1.txt" in t and "lead_pytest.out" in t
    assert "joins/CROP_JOINS_RECORD.json" in t and "I10RLcKF5P30Hb20p${x}_s$s" in t and "read_croprule10k.py" in t and "q[0-5]" in t and "for x in KB5 KB1 KD5 KD1; do" in t
    assert "--joins-record" in t and "--qmap-dir" not in t and "smooth" not in t.lower()
    assert 'for d in "$W/t1/I10RLcKF5P30Hb20mK100_s0" "$W/t1/I10RLcKF5P30Hb20mK100_s1"; do' in t and '[ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"' in t
    assert "ERRCUT" not in t and "errcut" not in t
    assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_croprule10k.sh").replace("\\", "/")]).returncode == 0
    for f in ("push_croprule.sh", "pull_croprule.sh"):
        assert subprocess.run([bash_exe(), "-n", str(HERE / f).replace("\\", "/")]).returncode == 0


def test_texts_hold_no_banned_words():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    n = 0
    for f in ("make_crop_joins.py", "make_croprule_queue.py", "pod_croprule10k_queue.sh", "read_croprule10k.py", "podread_croprule10k.sh", "test_croprule.py", "CROPRULE_DEFS.md",
              "push_croprule.sh", "pull_croprule.sh"):
        p = HERE / f
        if p.exists():
            n += 1
            t = p.read_text(encoding="utf-8").lower()
            hit = [w for w in bad if w in t]
            assert not hit, (f, hit)
    assert n >= 8
