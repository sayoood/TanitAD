"""test_gaperr_gap.py -- tests of the gap-file and rollout-queue kit of card ADD111 (GAPERR10K): control b0 clause by clause on planted faults, the slow and fast parts on hand-made cases, the two smoothing forms
and their properties, the record, the queue under DRY=1 (eight commands, nothing created), the start-marker first line, the gate (exit 4), the time rule, the admission and the gap-file step of the queue read out of its
own text, the generator, the push script and the wording. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest gaperr/gap/test_gaperr_gap.py -q -p no:cacheprovider --basetemp gaperr/gap/tmp_pt"""
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
V8 = HERE.parent.parent
sys.path.insert(0, str(HERE))
import make_gaperr_gapfiles as MG                # noqa: E402

QUEUE = HERE / "pod_gaperr10k_queue.sh"
PREV_Q = V8 / "fallattr" / "gap" / "pod_fallattr10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
ARMS = ("NR", "BR", "SM", "SL")


def bash_exe():
    return shutil.which("bash")


def posix(p):
    return Path(p).as_posix()


def run_bash(script, env_extra, args=(), cwd=None):
    env = dict(os.environ)
    env.update(env_extra)
    r = subprocess.run([bash_exe(), str(script)] + list(args), env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=120)
    return r.returncode, r.stdout


def fn_text(t, name):
    i0 = t.index(name + "() {")
    return t[i0:t.index("\n}\n", i0) + 3]


# ------------------------------------------------------------------------------------------------ worlds
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("ge")
    jp, bp, rp, op, lp, lop = MG.write_world(str(d / "world"))
    rec = MG.build(jp, bp, rp, lp, str(d / "clean"))
    return {"jp": jp, "bp": bp, "rp": rp, "op": op, "lp": lp, "lop": lop, "rec": rec, "d": d}


def tiny_world(d, gaps, valid=None, base_gap=None, base_has=None):
    """one clip with the given label gaps; base with the given gap and output flags; an LA file as card ADD110 defines it"""
    n = len(gaps)
    valid = np.ones(n, np.int64) if valid is None else np.asarray(valid, np.int64)
    J = {"clip": np.asarray(["c0"]), "off": np.asarray([0], np.int64), "n": np.asarray([n], np.int64), "lead_gap": np.asarray(gaps, np.float32), "lead_valid": valid}
    bh = np.ones(n, bool) if base_has is None else np.asarray(base_has, bool)
    bg = np.where(bh, np.full(n, 7.0) if base_gap is None else np.asarray(base_gap, np.float64), np.nan)
    B = {"clip_index": np.zeros(n, np.int64), "frame": np.arange(n, dtype=np.int64), "gap": bg, "has_out": bh, "clips": np.asarray(["c0"]), "seed": np.int64(-1)}
    os.makedirs(str(d), exist_ok=True)
    jp, bp, rp, lp = str(d / "join.npz"), str(d / "VGAPLE.npz"), str(d / "REC.json"), str(d / "LA.npz")
    np.savez(jp, **J)
    np.savez(bp, **B)
    np.savez(lp, **MG.synthetic_la(J, B))
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump({"files": {"VGAPLE": {"file_sha256": MG.sha_file(bp)}}}, fh)
    return jp, bp, rp, lp


def files_of(d):
    return {a: MG.load_npz(str(Path(d) / ("VGAPA_%s.npz" % a))) for a in ARMS}


# ------------------------------------------------------------------------------------------------ the gap files and control b0
def test_selftest_runs():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(HERE / "make_gaperr_gapfiles.py"), "--selftest"], env=env, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=900)
    assert r.stdout.strip().splitlines()[-1] == "ZZ GAPERR-GAPFILES SELFTEST OK", r.stdout[-500:]


def test_clean_build_passes_every_clause_and_the_record_has_exactly_the_keys_of_the_card(world):
    rec = world["rec"]
    assert sorted(rec) == sorted(["pass_b0", "clauses", "files", "base", "la_file", "eval_join_sha256", "t0", "written_utc"])
    assert rec["pass_b0"] is True and sorted(rec["clauses"]) == sorted(MG.CLAUSES)
    assert all(isinstance(c["pass"], bool) and c["pass"] for c in rec["clauses"].values())
    assert sorted(rec["files"]) == sorted(ARMS)
    for a in ARMS:
        f = rec["files"][a]
        assert sorted(f) == sorted(["n_rows", "n_diff_rows", "n_label_value_rows", "path", "sha256"])
        assert f["sha256"] == MG.sha_file(str(world["d"] / "clean" / ("VGAPA_%s.npz" % a))) and f["path"].endswith("VGAPA_%s.npz" % a)
    assert sorted(rec["base"]) == sorted(["path", "sha256", "named_in_record_add106"]) and rec["base"]["named_in_record_add106"] is True
    assert rec["base"]["sha256"] == MG.sha_file(world["bp"]) and rec["eval_join_sha256"] == MG.sha_file(world["jp"])
    assert sorted(rec["la_file"]) == ["path", "sha256"] and rec["la_file"]["sha256"] == MG.sha_file(world["lp"])
    assert sorted(rec["t0"]) == ["all", "far", "middle", "near"]
    for k, v in rec["t0"].items():
        assert sorted(v) == sorted(["n_error_rows", "rms_e", "rms_s", "rms_f", "rms_sm", "rms_sl"])
    assert rec["t0"]["near"]["n_error_rows"] + rec["t0"]["middle"]["n_error_rows"] + rec["t0"]["far"]["n_error_rows"] == rec["t0"]["all"]["n_error_rows"] > 0
    assert all(isinstance(rec["t0"]["all"][k], float) and rec["t0"]["all"][k] >= 0 for k in ("rms_e", "rms_s", "rms_f", "rms_sm", "rms_sl"))
    on_disk = json.loads((world["d"] / "clean" / "GAPERR_GAPFILES_RECORD.json").read_text(encoding="utf-8"))
    assert on_disk["pass_b0"] is True and sorted(on_disk) == sorted(rec)


def test_record_counts_are_the_independent_counts_and_the_layout_is_the_one_of_the_base_file(world):
    J, B = MG.load_npz(world["jp"]), MG.load_npz(world["bp"])
    names, n, cum, idx, ci, fr = MG.join_frames(J)
    g = np.asarray(J["lead_gap"], np.float64)[idx]
    lab = (np.asarray(J["lead_valid"])[idx] == 1) & np.isfinite(g) & (g >= 0) & (g <= 60)
    assert (lab & ~np.asarray(B["has_out"])).any() and (lab & np.asarray(B["has_out"])).any()
    z = files_of(world["d"] / "clean")
    for a in ARMS:
        assert sorted(z[a]) == sorted(B) == sorted(MG.FIELDS)
        assert all(str(z[a][k].dtype) == str(B[k].dtype) for k in B if k != "clips") and z[a]["clips"].dtype.kind == "U"
        assert int(z[a]["seed"]) == -1 and z[a]["seed"].ndim == 0
        diff = ~((z[a]["gap"] == B["gap"]) | (np.isnan(z[a]["gap"]) & np.isnan(B["gap"]))) | (z[a]["has_out"] != B["has_out"])
        assert world["rec"]["files"][a]["n_diff_rows"] == int(diff.sum()) and world["rec"]["files"][a]["n_rows"] == len(B["gap"])
        assert world["rec"]["files"][a]["n_label_value_rows"] == (int(lab.sum()) if a in ("NR", "BR") else 0)
    for a in ("SM", "SL"):
        assert np.array_equal(z[a]["has_out"], B["has_out"])
        no = ~B["has_out"]
        assert np.array_equal(z[a]["gap"][no], B["gap"][no], equal_nan=True)


def test_slow_and_fast_parts_on_hand_made_cases(tmp_path):
    # case A: one run of 13 error rows, label 30 everywhere, e = 0..12
    jp, bp, rp, lp = tiny_world(tmp_path / "a", [30.0] * 13, base_gap=30.0 + np.arange(13))
    r = MG.build(jp, bp, rp, lp, str(tmp_path / "a_out"))
    assert r["pass_b0"] is True
    z = files_of(tmp_path / "a_out")
    e = np.arange(13, dtype=np.float64)
    s0, s6, s12 = 2.5, 6.0, 9.5                                       # windows: rows 0..5, rows 1..11, rows 7..12
    assert abs(z["NR"]["gap"][0] - (30 + s0)) < 1e-9 and abs(z["NR"]["gap"][6] - (30 + s6)) < 1e-9 and abs(z["NR"]["gap"][12] - (30 + s12)) < 1e-9
    assert abs(z["BR"]["gap"][0] - (30 + e[0] - s0)) < 1e-9 and abs(z["BR"]["gap"][6] - (30 + e[6] - s6)) < 1e-9 and abs(z["BR"]["gap"][12] - (30 + e[12] - s12)) < 1e-9
    assert z["NR"]["has_out"].all() and z["BR"]["has_out"].all()
    # case B: a row that is not a label row splits the error run; its base value is kept
    e_b = [1, 2, 3, 4, 99, 5, 6, 7, 8, 9]
    jp, bp, rp, lp = tiny_world(tmp_path / "b", [30.0] * 10, valid=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1], base_gap=30.0 + np.asarray(e_b, np.float64))
    r = MG.build(jp, bp, rp, lp, str(tmp_path / "b_out"))
    assert r["pass_b0"] is True
    z = files_of(tmp_path / "b_out")
    assert abs(z["NR"]["gap"][3] - 32.5) < 1e-9 and abs(z["NR"]["gap"][5] - 37.0) < 1e-9 and z["NR"]["gap"][4] == 129.0 and z["BR"]["gap"][4] == 129.0
    # case C: a label row with no base output has the label gap and an output in NR and BR, a row that is not a label row and has no output stays as the base
    jp, bp, rp, lp = tiny_world(tmp_path / "c", [30.0, 30.0, 30.0, 30.0], valid=[1, 1, 1, 0], base_gap=[31.0, 32.0, 33.0, 34.0], base_has=[1, 0, 1, 0])
    r = MG.build(jp, bp, rp, lp, str(tmp_path / "c_out"))
    assert r["pass_b0"] is True
    z = files_of(tmp_path / "c_out")
    for a in ("NR", "BR"):
        assert z[a]["gap"][1] == 30.0 and bool(z[a]["has_out"][1]) is True
        assert np.isnan(z[a]["gap"][3]) and bool(z[a]["has_out"][3]) is False
    assert r["files"]["NR"]["n_label_value_rows"] == 3 and r["files"]["SM"]["n_label_value_rows"] == 0
    assert abs(z["NR"]["gap"][0] - 31.0) < 1e-9 and abs(z["NR"]["gap"][2] - 33.0) < 1e-9       # single-row runs: s = e, f = 0


# ------------------------------------------------------------------------------------------------ the smoothing forms
def run_smooth(y, has=None, kind="SM", frames=None, clips=None):
    y = np.asarray(y, np.float64)
    h = np.ones(len(y), bool) if has is None else np.asarray(has, bool)
    fr = np.arange(len(y), dtype=np.int64) if frames is None else np.asarray(frames, np.int64)
    ci = np.zeros(len(y), np.int64) if clips is None else np.asarray(clips, np.int64)
    return MG.smooth_gap(ci, fr, y, h, kind)


def test_smoothing_signature_has_no_label_parameter():
    assert list(inspect.signature(MG.smooth_gap).parameters) == ["clip_index", "frame", "gap", "has_out", "kind", "fault"]


def test_smoothing_properties_on_hand_made_series():
    const = np.full(9, 12.5)
    assert np.array_equal(run_smooth(const), const) and np.abs(run_smooth(const, kind="SL") - const).max() <= 1e-9
    ramp = 40.0 - 1.0 * np.arange(12)
    assert np.abs(run_smooth(ramp, kind="SL") - ramp).max() <= 1e-9
    sm = run_smooth(ramp)
    assert all(abs(sm[i] - ramp[i - 2]) <= 1e-9 for i in range(4, 12))                        # two frames late once five frames are in the run
    assert sm[0] == ramp[0] and abs(sm[1] - 0.5 * (ramp[0] + ramp[1])) <= 1e-9 and abs(sm[2] - ramp[1]) <= 1e-9
    step3 = 10.0 + 3.0 * np.arange(8)                                                        # a step of exactly 3.0 m does not end the run
    assert all(abs(run_smooth(step3)[i] - step3[i - 2]) <= 1e-9 for i in range(4, 8))
    assert run_smooth([10.0, 13.5, 13.5])[1] == 13.5 and run_smooth([10.0, 13.5, 13.5], kind="SL")[1] == 13.5     # 3.5 m ends the run
    jump = np.asarray([10.0] * 5 + [20.0] * 5)
    assert np.array_equal(run_smooth(jump), jump) and np.abs(run_smooth(jump, kind="SL") - jump).max() <= 1e-9
    nan = float("nan")
    y = [10.0, 11.0, 12.0, nan, 14.0, 15.0]
    h = [1, 1, 1, 0, 1, 1]
    sm, sl = run_smooth(y, h), run_smooth(y, h, kind="SL")
    assert np.isnan(sm[3]) and np.isnan(sl[3]) and sm[4] == 14.0 and abs(sm[5] - 14.5) <= 1e-9 and sl[5] == 15.0 and sl[2] == 12.0
    # a run does not cross a clip border or a gap in the frame numbers
    assert run_smooth([10.0, 10.0, 11.0, 11.0], clips=[0, 0, 1, 1])[2] == 11.0
    assert run_smooth([10.0, 10.0, 12.0, 12.0], frames=[0, 1, 5, 6])[2] == 12.0
    # at most five frames: a rising line with a spike 6 frames back does not reach the median
    y = np.asarray([50.0, 10.0, 10.5, 11.0, 11.5, 12.0])
    assert run_smooth(y)[5] == 11.0                                                           # frames 1..5 only (the 50 is a jump and the cap is 5)


def test_smoothing_files_do_not_change_when_the_label_fields_of_the_join_change(world, tmp_path):
    J = MG.load_npz(world["jp"])
    J2 = dict(J)
    J2["lead_gap"] = np.asarray(J["lead_gap"], np.float32)[::-1] + np.float32(3.0)
    J2["lead_valid"] = 1 - np.asarray(J["lead_valid"])
    jp2 = str(tmp_path / "join2.npz")
    np.savez(jp2, **J2)
    r2 = MG.build(jp2, world["bp"], world["rp"], world["lp"], str(tmp_path / "out2"))
    z1, z2 = files_of(world["d"] / "clean"), files_of(tmp_path / "out2")
    for a in ("SM", "SL"):
        assert sorted(z1[a]) == sorted(z2[a]) and all(z1[a][k].tobytes() == z2[a][k].tobytes() for k in z1[a])
    for a in ("NR", "BR"):
        assert not np.array_equal(z1[a]["gap"], z2[a]["gap"], equal_nan=True)
    assert r2["files"]["SM"]["n_label_value_rows"] == 0


@pytest.mark.parametrize("clause,fault", [("second_routine", "win4"), ("second_routine", "nr_nonerr_offset"), ("error_identity", "br_sign"), ("la_equivalence", "nr_nonerr_offset"),
                                           ("smoothing_properties", "sl_mean"), ("smoothing_properties", "sm_median3"), ("smoothing_properties", "sm_flag"),
                                           ("no_label_in_smoothing", "sm_label_leak"), ("layout", "f32gap"), ("layout", "extra_field"), ("layout", "seed0")])
def test_each_planted_fault_turns_its_clause_red_and_the_clean_build_has_it_green(world, clause, fault):
    r = MG.build(world["jp"], world["bp"], world["rp"], world["lp"], str(world["d"] / ("f_%s_%s" % (clause, fault))), fault=fault)
    assert r["clauses"][clause]["pass"] is False and r["pass_b0"] is False
    assert world["rec"]["clauses"][clause]["pass"] is True


def test_an_unrecorded_base_turns_base_is_recorded_red(world):
    r = MG.build(world["jp"], world["bp"], world["rp"], world["lp"], str(world["d"] / "f_base"), base_for_build=world["op"])
    assert r["clauses"]["base_is_recorded"]["pass"] is False and r["pass_b0"] is False and r["base"]["named_in_record_add106"] is False


def test_another_la_file_turns_la_equivalence_red_and_an_unreadable_one_too(world, tmp_path):
    r = MG.build(world["jp"], world["bp"], world["rp"], world["lop"], str(world["d"] / "f_la"))
    assert r["clauses"]["la_equivalence"]["pass"] is False and r["pass_b0"] is False
    bogus = tmp_path / "bogus.npz"
    bogus.write_bytes(b"not an npz")
    r = MG.build(world["jp"], world["bp"], world["rp"], str(bogus), str(tmp_path / "o"))
    assert r["clauses"]["la_equivalence"]["pass"] is False and r["pass_b0"] is False


def test_a_base_with_other_rows_than_the_join_turns_the_layout_red(tmp_path):
    jp, bp, rp, lp = tiny_world(tmp_path / "w", [10.0, 25.0, 45.0])
    B = MG.load_npz(bp)
    B["clips"] = np.asarray(["other"])
    np.savez(bp, **B)
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump({"files": {"VGAPLE": {"file_sha256": MG.sha_file(bp)}}}, fh)
    r = MG.build(jp, bp, rp, lp, str(tmp_path / "out"))
    assert r["clauses"]["layout"]["pass"] is False and r["pass_b0"] is False


def test_main_exit_codes_refusal_and_printout(world, tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = tmp_path / "o1"
    cmd = [sys.executable, str(HERE / "make_gaperr_gapfiles.py"), "--join", world["jp"], "--base", world["bp"], "--base-record", world["rp"], "--la-file", world["lp"], "--out-dir", str(out)]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    lines = r.stdout.strip().splitlines()
    assert r.returncode == 0 and lines[-1] == "ZZ GAPERR-GAPFILES pass_b0 True", r.stdout[-400:]
    assert len(lines) == 1 + len(MG.CLAUSES) and all(re.fullmatch(r"clause [a-z_]+ (True|False)", x) for x in lines[:-1])      # no gap value and no rms value is printed
    r2 = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    assert r2.returncode != 0 and "REFUSED" in r2.stdout
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps({"files": {"VGAPLE": {"file_sha256": "1" * 64}}}), encoding="utf-8")
    cmd3 = list(cmd)
    cmd3[cmd3.index("--out-dir") + 1] = str(tmp_path / "o3")
    cmd3[cmd3.index("--base-record") + 1] = str(bad_rec)
    r3 = subprocess.run(cmd3, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    assert r3.returncode == 1 and r3.stdout.strip().splitlines()[-1] == "ZZ GAPERR-GAPFILES pass_b0 False", r3.stdout[-400:]


# ------------------------------------------------------------------------------------------------ the queue
def dry(env_extra=None, cwd=None):
    rc, out = run_bash(QUEUE, dict({"DRY": "1"}, **(env_extra or {})), cwd=cwd)
    return rc, out, [x for x in out.splitlines() if x.startswith("CMD: DISTRULE_DIR")]


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_syntax_and_dry_run_prints_eight_commands_with_the_right_run_folders_and_gap_files_and_creates_nothing(tmp_path):
    assert subprocess.run([bash_exe(), "-n", str(QUEUE)]).returncode == 0
    ge = tmp_path / "ge_none"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    rc, out, cmds = dry({"GE_ROOT": posix(ge)}, cwd=str(cwd))
    assert rc == 0 and out.strip().splitlines()[-1] == "DRY DONE", out[-600:]
    assert len(cmds) == 8
    assert not ge.exists() and list(cwd.iterdir()) == []
    seen = []
    for c in cmds:
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and c.count("--vis-closing") == 1 and c.count("--vis-gap-file") == 1 and "--rule-join" not in c
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        m = re.search(r"--out-dir /workspace/v8proof/integ10k/t1/(I10RQ50KF5P30Hb20b(NR|BR|SM|SL)_s([01])) ", c)
        assert m, c
        arm, k = m.group(2), m.group(3)
        seen.append((arm, int(k)))
        assert ("--vis-gap-file %s/gapfiles/VGAPA_%s.npz" % (posix(ge), arm)) in c
        assert ("--reader /workspace/v8proof/integ10k/lr16/TK16_s%s/" % k) in c and ("QMAP_LR16_s%s_q50" % k) in c and ("--ckpt /workspace/v8proof/integ10k/ckpt/UE_s%s/A/ckpt_last.pt" % k) in c
    assert seen == [(a, k) for a in ARMS for k in (0, 1)]
    rc2, out2, cmds2 = dry()
    assert rc2 == 0 and len(cmds2) == 8
    assert "--vis-gap-file /workspace/v8proof/gaperr/gapfiles/VGAPA_BR.npz" in cmds2[2] and "I10RQ50KF5P30Hb20bBR_s0" in cmds2[2]
    call = ("make_gaperr_gapfiles.py --join /workspace/v8proof/hierrep/pins/join_eval141.npz --base /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz "
            "--base-record /workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json --la-file /workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz --out-dir /workspace/v8proof/gaperr/gapfiles")
    assert call in out2
    assert "GATE:" in out2 and "STOPRULE" in out2 and "pass_b0" in out2


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_gate_refuses_with_exit_4_without_the_test_outputs_and_passes_on_to_the_next_check_with_them(tmp_path):
    ge = tmp_path / "ge"
    (ge / "kit_gap").mkdir(parents=True)
    (ge / "kit_read").mkdir(parents=True)
    env = {"GE_ROOT": posix(ge), "I10_W": posix(tmp_path / "no_w")}
    rc, out = run_bash(QUEUE, env)
    assert rc == 4 and "REFUSED: step 0 gate" in out
    (ge / "kit_gap" / "lead_pytest.out").write_text("12 passed\n", encoding="utf-8")
    rc, out = run_bash(QUEUE, env)
    assert rc == 4                                                                # one kit only
    (ge / "kit_read" / "lead_pytest.out").write_text("9 passed\n", encoding="utf-8")
    rc, out = run_bash(QUEUE, env)
    assert rc == 2 and "REFUSED: no " in out                                       # past the gate: the pod check refuses (no pod folder here)
    assert not (ge / "GAPERR10K_START.txt").exists() and not (ge / "GAPERR10K_STOP.txt").exists()


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_start_marker_first_line_format_and_time_rule_read_out_of_the_queue(tmp_path):
    q = QUEUE.read_text(encoding="utf-8")
    # the line that writes the start marker, run as it stands in the queue
    m = re.search(r'^\s*\( set -C; (echo "\$\(date \+%s\) \$\(ts\) pid \$\$" > "\$GP/GAPERR10K_START.txt") \) 2>/dev/null', q, re.M)
    assert m, "start-marker line not found"
    gp = tmp_path / "gp"
    gp.mkdir()
    sc = tmp_path / "mk.sh"
    sc.write_text('GP="$1"\nts() { date -u +%Y-%m-%dT%H:%M:%SZ; }\n' + m.group(1) + "\n", encoding="utf-8")
    assert subprocess.run([bash_exe(), str(sc), posix(gp)]).returncode == 0
    first = (gp / "GAPERR10K_START.txt").read_text(encoding="utf-8").splitlines()[0]
    assert re.fullmatch(r"[0-9]+ [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z pid [0-9]+", first), first
    assert abs(int(first.split()[0]) - int(time.time())) < 120
    ct = fn_text(q, "check_time")
    sc2 = tmp_path / "ct.sh"
    sc2.write_text('DRY=0\nGP="$1"\nSTOPS="$2"\nstop() { echo "STOP $*"; exit 3; }\n' + ct + 'check_time "x"; echo OK\n', encoding="utf-8")
    r = subprocess.run([bash_exe(), str(sc2), posix(gp), "10800"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    assert r.returncode == 0 and r.stdout.strip() == "OK"                        # the marker the queue wrote is read by the queue's own time rule
    start = gp / "GAPERR10K_START.txt"
    start.write_text("%d 2026-11-22T00:00:00Z pid 1\n%d resume x pid 2\n" % (int(time.time()) - 100, int(time.time())), encoding="utf-8")   # the FIRST line counts
    r = subprocess.run([bash_exe(), str(sc2), posix(gp), "50"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    assert r.returncode == 3 and "STOP STOP-RULE" in r.stdout
    start.write_text("garbage\n", encoding="utf-8")
    r = subprocess.run([bash_exe(), str(sc2), posix(gp), "10800"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    assert r.returncode == 3 and "START-LINE-UNREADABLE" in r.stdout
    assert 'STOPS=${GE_STOP_S:-10800}' in q and "GAPERR10K_STOP.txt" in q and "ZZ GAPERR10K DONE" in q and "GAPERR10K_DONE.txt" in q and "gaperr_queue.log" in q and "RESUME" in q
    assert "set -C" in q and "$GP/markers" in q


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_admission_and_gap_file_step_read_out_of_the_queue_accept_the_record_the_tool_writes(world, tmp_path):
    q = QUEUE.read_text(encoding="utf-8")
    funcs = fn_text(q, "admit") + fn_text(q, "step_gapfiles")
    sc = tmp_path / "step.sh"
    sc.write_text('DRY=0\nPY="$1"\nKG="$2"\nGF="$3"\nJOINE="$4"\nBASEF="$5"\nBASER="$6"\nLOG="$7"\nLAF="$8"\nstop() { echo "STOP $*"; exit 3; }\nlog() { echo "LOG $*"; }\ncheck_time() { :; }\n'
                  + funcs + "step_gapfiles; echo RC=$?\n", encoding="utf-8")

    def go(gf, baser, laf=None):
        r = subprocess.run([bash_exe(), str(sc), posix(sys.executable), posix(HERE), posix(gf), posix(world["jp"]), posix(world["bp"]), posix(baser), posix(tmp_path / "log.txt"), posix(laf or world["lp"])],
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
        return r.returncode, r.stdout
    gf = tmp_path / "gapfiles"
    rc, out = go(gf, world["rp"])
    assert rc == 0 and "LOG ADMIT-OK files 4" in out and (gf / "GAPERR_GAPFILES_RECORD.json").is_file(), out
    rc, out = go(gf, world["rp"])                                                  # present folder: not built again, admitted again
    assert rc == 0 and "ADMIT-OK" in out
    rc, out = go(gf, world["rp"], world["lop"])                                    # another LA file than the recorded one
    assert rc == 3 and "GAPFILE-ADMISSION LA-FILE-IS-NOT-THE-RECORDED-ONE" in out, out
    z = MG.load_npz(str(gf / "VGAPA_SM.npz"))                                      # a changed file is refused by its hash
    z["gap"] = z["gap"] + 1.0
    np.savez(str(gf / "VGAPA_SM.npz"), **z)
    rc, out = go(gf, world["rp"])
    assert rc == 3 and "GAPFILE-ADMISSION GAPFILE-HASH-DIFFERS SM" in out, out
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps({"files": {"VGAPLE": {"file_sha256": "1" * 64}}}), encoding="utf-8")
    rc, out = go(tmp_path / "gapfiles2", bad_rec)                                  # b0 fails (base not recorded): stop
    assert rc == 3 and "B0-NOT-PASS" in out, out


def test_rollout_function_is_the_one_of_the_previous_queue_and_the_header_facts_hold():
    q, p = QUEUE.read_text(encoding="utf-8"), PREV_Q.read_text(encoding="utf-8")
    assert hashlib.sha256(PREV_Q.read_bytes()).hexdigest()[:16] == "dc6f6c6405092fe3"
    assert fn_text(q, "rollout") == fn_text(p, "rollout") and fn_text(q, "check_time").replace("GAPERR10K", "FALLATTR10K") == fn_text(p, "check_time")
    for need in ('[ "$got" = "28bc31f21ecb4ccb" ]', 'grep -q "28bc31f21ecb4ccb" "$CHK/CHECK_WRAPPER_V.hashes"', "ZZ CHECK-WRAPPER-V PASS", "MAXGPU=3", "MINGB=${I10_MIN_FREE_GB:-28}", "set -C",
                 "kit_gap", "kit_read", "/workspace/v8proof/cropscale/gapfiles/VGAPLE.npz", "/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json", "CUDA_VISIBLE_DEVICES=-1",
                 "GE=${GE_ROOT:-/workspace/v8proof/gaperr}", "LAF=/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz", "I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20aLA",
                 'NAMES="I10RQ50KF5P30Hb20bNR I10RQ50KF5P30Hb20bBR I10RQ50KF5P30Hb20bSM I10RQ50KF5P30Hb20bSL"'):
        assert need in q, need
    base = [x for x in p.splitlines() if "rollout I10RQ50KF5P30Hb20aLA RR" in x][0]
    env0 = base.strip().split(" VGAP=")[0]
    mine_lines = [x for x in q.splitlines() if "rollout I10RQ50KF5P30Hb20b" in x]
    assert len(mine_lines) == 4
    for a, x in zip(ARMS, mine_lines):
        assert x.strip().startswith(env0) and (' VGAP="$GF/VGAPA_%s.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20b%s RR "$k" "$k"' % (a, a)) in x
    for bad in ("FALLATTR", "FA_ROOT", "FA_STOP_S", "pass_a0", "make_attr_gapfiles"):
        assert bad not in q, bad


def test_the_queue_is_regenerated_byte_for_byte_by_its_generator():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VQ_OVERWRITE_OWN="1")
    before = QUEUE.read_bytes()
    r = subprocess.run([sys.executable, str(HERE / "make_gaperr_queue.py")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100)
    assert r.returncode == 0 and "rollouts 8" in r.stdout, r.stdout[-400:]
    assert QUEUE.read_bytes() == before


def test_the_queue_never_deletes_moves_or_kills():
    code = "\n".join(x for x in QUEUE.read_text(encoding="utf-8").splitlines() if not x.lstrip().startswith("#"))
    for bad in (r"\brm\b", r"\brmdir\b", r"\bmv\b", r"\bkill\b", r"\bpkill\b", r"\btruncate\b", r"\bshred\b"):
        assert not re.search(bad, code), bad


# ------------------------------------------------------------------------------------------------ push script, wording, paths
def test_push_script_and_texts():
    pu = (HERE / "push_gaperr_gap.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "lead_pytest.out" in pu and "/workspace/v8proof/gaperr" in pu and "kit_gap" in pu and "sha256sum" in pu and "ZZ PUSH-GAPERR-GAP" in pu
    for f in ("make_gaperr_gapfiles.py", "make_gaperr_queue.py", "pod_gaperr10k_queue.sh"):
        assert f in pu and (HERE / f).is_file()
    assert not re.search(r"ssh |root@|@[0-9]+\.[0-9]+", pu)                                   # no pod address
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "push_gaperr_gap.sh")]).returncode == 0


def test_wording_scan_and_path_length():
    for f in ("make_gaperr_gapfiles.py", "make_gaperr_queue.py", "pod_gaperr10k_queue.sh", "push_gaperr_gap.sh", "test_gaperr_gap.py", "GAPERR_GAP_DEFS.md"):
        p = HERE / f
        assert p.is_file(), f
        t = p.read_text(encoding="utf-8")
        assert not [w for w in BANNED if w in t], (f, [w for w in BANNED if w in t])
        assert "\r" not in t and len(posix(p.relative_to(V8))) <= 200
    for p in HERE.iterdir():
        if p.is_file():
            assert not re.match(r"(?i)^(aux|con|prn|nul)(\.|$)", p.name)
            assert p.stat().st_size < 5 * 1024 * 1024
