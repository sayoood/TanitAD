"""test_fallattr_gap.py -- tests of the gap-file and rollout-queue kit of card ADD110 (FALLATTR10K): control a0 clause by clause on planted faults, the range edges, the source-change count, the record, the queue
under DRY=1 (eight commands, nothing created), the gate (exit 4), the time rule, the admission and the gap-file step of the queue read out of its own text, the generator, the push script and the wording.
Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest fallattr/gap/test_fallattr_gap.py -q -p no:cacheprovider --basetemp fallattr/gap/tmp_pt/<new name>"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
V8 = HERE.parent.parent
sys.path.insert(0, str(HERE))
import make_attr_gapfiles as MAG                # noqa: E402

QUEUE = HERE / "pod_fallattr10k_queue.sh"
MODEL_Q = V8 / "cropscale" / "fit" / "pod_cropscale10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
ARMS = ("LA", "LN", "LM", "LF")


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
    d = tmp_path_factory.mktemp("fa")
    jp, bp, rp, op = MAG.write_world(str(d / "world"))
    rec = MAG.build(jp, bp, rp, str(d / "clean"))
    return {"jp": jp, "bp": bp, "rp": rp, "op": op, "rec": rec, "d": d}


def tiny_world(d, gaps, valid=None, base_gap=None, base_has=None):
    """one clip with the given label gaps; base with the given gap and output flags"""
    n = len(gaps)
    valid = np.ones(n, np.int64) if valid is None else np.asarray(valid, np.int64)
    J = {"clip": np.asarray(["c0"]), "off": np.asarray([0], np.int64), "n": np.asarray([n], np.int64), "lead_gap": np.asarray(gaps, np.float32), "lead_valid": valid}
    bg = np.full(n, 7.0) if base_gap is None else np.asarray(base_gap, np.float64)
    bh = np.ones(n, bool) if base_has is None else np.asarray(base_has, bool)
    bg = np.where(bh, bg, np.nan)
    B = {"clip_index": np.zeros(n, np.int64), "frame": np.arange(n, dtype=np.int64), "gap": bg, "has_out": bh, "clips": np.asarray(["c0"]), "seed": np.int64(-1)}
    os.makedirs(str(d), exist_ok=True)
    jp, bp, rp = str(d / "join.npz"), str(d / "VGAPLE.npz"), str(d / "REC.json")
    np.savez(jp, **J)
    np.savez(bp, **B)
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump({"files": {"VGAPLE": {"file_sha256": MAG.sha_file(bp)}}}, fh)
    return jp, bp, rp


# ------------------------------------------------------------------------------------------------ the gap files and control a0
def test_selftest_runs():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(HERE / "make_attr_gapfiles.py"), "--selftest"], env=env, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=600)
    assert r.stdout.strip().splitlines()[-1] == "ZZ ATTR-GAPFILES SELFTEST OK", r.stdout[-500:]


def test_clean_build_passes_every_clause_and_the_record_has_the_keys_of_the_card(world):
    rec = world["rec"]
    assert sorted(rec) == sorted(["pass_a0", "clauses", "files", "base", "eval_join_sha256", "written_utc"])
    assert rec["pass_a0"] is True and sorted(rec["clauses"]) == sorted(MAG.CLAUSES)
    assert all(isinstance(c["pass"], bool) and c["pass"] for c in rec["clauses"].values())
    assert sorted(rec["files"]) == sorted(ARMS)
    for a in ARMS:
        f = rec["files"][a]
        assert sorted(f) == sorted(["path", "sha256", "n_rows", "n_label_rows", "n_source_changes"])
        assert f["sha256"] == MAG.sha_file(str(world["d"] / "clean" / ("VGAPA_%s.npz" % a))) and f["path"].endswith("VGAPA_%s.npz" % a)
    assert sorted(rec["base"]) == sorted(["path", "sha256", "named_in_record_add106"]) and rec["base"]["named_in_record_add106"] is True
    assert rec["base"]["sha256"] == MAG.sha_file(world["bp"]) and rec["eval_join_sha256"] == MAG.sha_file(world["jp"])
    assert (world["d"] / "clean" / "FALLATTR_GAPFILES_RECORD.json").is_file()
    assert json.loads((world["d"] / "clean" / "FALLATTR_GAPFILES_RECORD.json").read_text(encoding="utf-8"))["pass_a0"] is True


def test_the_four_files_have_the_layout_of_the_base_file_and_the_composition_of_the_card(world):
    J, B = MAG.load_npz(world["jp"]), MAG.load_npz(world["bp"])
    names, n, cum, idx, ci, fr = MAG.join_frames(J)
    g = np.asarray(J["lead_gap"], np.float64)[idx]
    lab = (np.asarray(J["lead_valid"])[idx] == 1) & np.isfinite(g) & (g >= 0) & (g <= 60)
    ranges = {"LA": lab, "LN": lab & (g < 20), "LM": lab & (g >= 20) & (g < 40), "LF": lab & (g >= 40)}
    for a in ARMS:
        z = MAG.load_npz(str(world["d"] / "clean" / ("VGAPA_%s.npz" % a)))
        assert sorted(z) == sorted(B) == sorted(MAG.FIELDS)
        assert all(str(z[k].dtype) == str(B[k].dtype) for k in B if k != "clips") and z["clips"].dtype.kind == "U"
        assert int(z["seed"]) == -1 and z["seed"].ndim == 0
        assert np.array_equal(z["gap"][ranges[a]], g[ranges[a]]) and z["has_out"][ranges[a]].all()
        off = ~ranges[a]
        assert np.array_equal(z["gap"][off], B["gap"][off], equal_nan=True) and np.array_equal(z["has_out"][off], B["has_out"][off])
    assert (ranges["LA"] & ~np.asarray(B["has_out"])).any()                       # the world has label rows where the base has no output: they get an output
    for a in ("LN", "LM", "LF"):
        assert ranges[a].any()


@pytest.mark.parametrize("clause,fault", [("second_routine", "offset_label"), ("second_routine", "outside_zero"), ("ranges_partition", "lm_incl_40"), ("ranges_partition", "lf_drop_last"),
                                           ("outside_is_base", "outside_zero"), ("layout", "f32gap"), ("layout", "extra_field"), ("layout", "seed0")])
def test_each_planted_fault_turns_its_clause_red_and_the_clean_build_has_it_green(world, clause, fault):
    r = MAG.build(world["jp"], world["bp"], world["rp"], str(world["d"] / ("f_%s_%s" % (clause, fault))), fault=fault)
    assert r["clauses"][clause]["pass"] is False and r["pass_a0"] is False
    assert world["rec"]["clauses"][clause]["pass"] is True


def test_an_unrecorded_base_turns_base_is_recorded_red(world):
    r = MAG.build(world["jp"], world["bp"], world["rp"], str(world["d"] / "f_base"), base_for_build=world["op"])
    assert r["clauses"]["base_is_recorded"]["pass"] is False and r["pass_a0"] is False and r["base"]["named_in_record_add106"] is False


def test_a_base_with_other_rows_than_the_join_turns_the_layout_red(tmp_path):
    jp, bp, rp = tiny_world(tmp_path / "w", [10.0, 25.0, 45.0])
    B = MAG.load_npz(bp)
    B["clips"] = np.asarray(["other"])
    np.savez(bp, **B)
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump({"files": {"VGAPLE": {"file_sha256": MAG.sha_file(bp)}}}, fh)
    r = MAG.build(jp, bp, rp, str(tmp_path / "out"))
    assert r["clauses"]["layout"]["pass"] is False and r["pass_a0"] is False


def test_range_edges_20_goes_to_lm_40_to_lf_60_is_a_label_row_and_60_01_is_not(tmp_path):
    gaps = [19.99, 20.0, 39.99, 40.0, 60.0, 60.01, 0.0, -0.5, 30.0, np.nan, 5.0]
    valid = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    jp, bp, rp = tiny_world(tmp_path / "w", gaps, valid, base_gap=np.full(len(gaps), 7.0), base_has=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0])
    r = MAG.build(jp, bp, rp, str(tmp_path / "out"))
    assert r["pass_a0"] is True
    z = {a: MAG.load_npz(str(tmp_path / "out" / ("VGAPA_%s.npz" % a))) for a in ARMS}
    lab = [0, 1, 2, 3, 4, 6, 10]                                                  # 60.01, -0.5, the invalid row and the NaN row are not label rows
    ln, lm, lf = [0, 6, 10], [1, 2], [3, 4]
    for a, rows in (("LA", lab), ("LN", ln), ("LM", lm), ("LF", lf)):
        mask = np.zeros(len(gaps), bool)
        mask[rows] = True
        assert r["files"][a]["n_label_rows"] == len(rows)
        assert np.array_equal(z[a]["gap"][mask], np.asarray(gaps, np.float32).astype(np.float64)[mask]) and z[a]["has_out"][mask].all()
        assert np.array_equal(z[a]["gap"][~mask], np.where(np.asarray([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], bool), 7.0, np.nan)[~mask], equal_nan=True)
    assert z["LM"]["gap"][1] == 20.0 and z["LF"]["gap"][3] == 40.0 and z["LF"]["gap"][4] == 60.0
    assert z["LA"]["gap"][5] == 7.0 and z["LA"]["gap"][7] == 7.0 and z["LA"]["gap"][8] == 7.0           # not label rows: the base
    assert z["LF"]["has_out"][3] and not z["LN"]["has_out"][3]                                           # a label row gets an output where the base has none; elsewhere the base flag stays


def test_source_changes_on_a_hand_made_case(tmp_path):
    assert MAG.count_source_changes([0, 0, 0, 0, 0], [0, 1, 1, 0, 1]) == 3
    assert MAG.count_source_changes([0, 0, 0, 1, 1], [1, 1, 0, 0, 1]) == 2             # the pair across the clip border is not counted
    assert MAG.count_source_changes([0, 0, 0], [0, 0, 0]) == 0 and MAG.count_source_changes([0], [1]) == 0
    jp, bp, rp = tiny_world(tmp_path / "w", [10.0, 50.0, 50.0, 10.0, 30.0, 70.0])
    r = MAG.build(jp, bp, rp, str(tmp_path / "out"))
    # label rows 0..4 (70 is not one). LA: mask 1 1 1 1 1 0 -> 1 change; LN: 1 0 0 1 0 0 -> 3; LM: 0 0 0 0 1 0 -> 2; LF: 0 1 1 0 0 0 -> 2
    assert [r["files"][a]["n_source_changes"] for a in ARMS] == [1, 3, 2, 2]


def test_main_exit_codes_refusal_and_printout(world, tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = tmp_path / "o1"
    cmd = [sys.executable, str(HERE / "make_attr_gapfiles.py"), "--eval-join", world["jp"], "--base", world["bp"], "--base-record", world["rp"], "--out-dir", str(out)]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "ZZ ATTR-GAPFILES pass_a0 True", r.stdout[-400:]
    assert all(x.startswith("clause ") or x.startswith("ZZ ") for x in r.stdout.strip().splitlines())              # no gap value is printed
    r2 = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    assert r2.returncode != 0 and "REFUSED" in r2.stdout
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps({"files": {"VGAPLE": {"file_sha256": "1" * 64}}}), encoding="utf-8")
    cmd3 = cmd[:-1] + [str(tmp_path / "o3")]
    cmd3[cmd3.index("--base-record") + 1] = str(bad_rec)
    r3 = subprocess.run(cmd3, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    assert r3.returncode == 1 and r3.stdout.strip().splitlines()[-1] == "ZZ ATTR-GAPFILES pass_a0 False", r3.stdout[-400:]


# ------------------------------------------------------------------------------------------------ the queue
def dry(env_extra=None, cwd=None):
    rc, out = run_bash(QUEUE, dict({"DRY": "1"}, **(env_extra or {})), cwd=cwd)
    return rc, out, [x for x in out.splitlines() if x.startswith("CMD: DISTRULE_DIR")]


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_syntax_and_dry_run_prints_eight_commands_with_the_right_run_folders_and_gap_files_and_creates_nothing(tmp_path):
    assert subprocess.run([bash_exe(), "-n", str(QUEUE)]).returncode == 0
    fa = tmp_path / "fa_none"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    rc, out, cmds = dry({"FA_ROOT": posix(fa)}, cwd=str(cwd))
    assert rc == 0 and out.strip().splitlines()[-1] == "DRY DONE", out[-600:]
    assert len(cmds) == 8
    assert not fa.exists() and list(cwd.iterdir()) == []
    seen = []
    for c in cmds:
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and c.count("--vis-closing") == 1 and c.count("--vis-gap-file") == 1 and "--rule-join" not in c
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        m = re.search(r"--out-dir /workspace/v8proof/integ10k/t1/(I10RQ50KF5P30Hb20a(L[ANMF])_s([01])) ", c)
        assert m, c
        arm, k = m.group(2), m.group(3)
        seen.append((arm, int(k)))
        assert ("--vis-gap-file %s/gapfiles/VGAPA_%s.npz" % (posix(fa), arm)) in c
        assert ("--reader /workspace/v8proof/integ10k/lr16/TK16_s%s/" % k) in c and ("QMAP_LR16_s%s_q50" % k) in c and ("--ckpt /workspace/v8proof/integ10k/ckpt/UE_s%s/A/ckpt_last.pt" % k) in c
    assert seen == [(a, k) for a in ARMS for k in (0, 1)]
    rc2, out2, cmds2 = dry()
    assert rc2 == 0 and len(cmds2) == 8
    assert "--vis-gap-file /workspace/v8proof/fallattr/gapfiles/VGAPA_LN.npz" in cmds2[2]
    assert "I10RQ50KF5P30Hb20aLN_s0" in cmds2[2]
    assert "make_attr_gapfiles.py --eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz --base /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz --base-record /workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json --out-dir /workspace/v8proof/fallattr/gapfiles" in out2
    assert "GATE:" in out2 and "STOPRULE" in out2


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_gate_refuses_with_exit_4_without_the_test_outputs_and_passes_on_to_the_next_check_with_them(tmp_path):
    fa = tmp_path / "fa"
    (fa / "kit_gap").mkdir(parents=True)
    (fa / "kit_read").mkdir(parents=True)
    env = {"FA_ROOT": posix(fa), "I10_W": posix(tmp_path / "no_w")}
    rc, out = run_bash(QUEUE, env)
    assert rc == 4 and "REFUSED: step 0 gate" in out
    (fa / "kit_gap" / "lead_pytest.out").write_text("12 passed\n", encoding="utf-8")
    rc, out = run_bash(QUEUE, env)
    assert rc == 4                                                                # one kit only
    (fa / "kit_read" / "lead_pytest.out").write_text("9 passed\n", encoding="utf-8")
    rc, out = run_bash(QUEUE, env)
    assert rc == 2 and "REFUSED: no " in out                                       # past the gate: the pod check refuses (no pod folder here)
    assert not (fa / "FALLATTR10K_START.txt").exists() and not (fa / "FALLATTR10K_STOP.txt").exists()


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_time_rule_function_read_out_of_the_queue(tmp_path):
    q = QUEUE.read_text(encoding="utf-8")
    ct = fn_text(q, "check_time")
    gp = tmp_path / "gp"
    gp.mkdir()
    sc = tmp_path / "ct.sh"
    sc.write_text('DRY=0\nGP="$1"\nSTOPS="$2"\nstop() { echo "STOP $*"; exit 3; }\n' + ct + 'check_time "x"; echo OK\n', encoding="utf-8")
    import time
    start = gp / "FALLATTR10K_START.txt"
    start.write_text("%d 2026-11-22T00:00:00Z pid 1\n%d resume x pid 2\n" % (int(time.time()) - 100, int(time.time())), encoding="utf-8")   # the FIRST line counts
    r = subprocess.run([bash_exe(), str(sc), posix(gp), "50"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    assert r.returncode == 3 and "STOP STOP-RULE" in r.stdout
    r = subprocess.run([bash_exe(), str(sc), posix(gp), "10800"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    assert r.returncode == 0 and r.stdout.strip() == "OK"
    start.write_text("garbage\n", encoding="utf-8")
    r = subprocess.run([bash_exe(), str(sc), posix(gp), "10800"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    assert r.returncode == 3 and "START-LINE-UNREADABLE" in r.stdout
    assert 'STOPS=${FA_STOP_S:-10800}' in q and "FALLATTR10K_STOP.txt" in q and "ZZ FALLATTR10K DONE" in q and "FALLATTR10K_DONE.txt" in q and "fallattr_queue.log" in q and "RESUME" in q


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_admission_and_gap_file_step_read_out_of_the_queue_accept_the_record_the_tool_writes(world, tmp_path):
    q = QUEUE.read_text(encoding="utf-8")
    funcs = fn_text(q, "admit") + fn_text(q, "step_gapfiles")
    sc = tmp_path / "step.sh"
    sc.write_text('DRY=0\nPY="$1"\nKG="$2"\nGF="$3"\nJOINE="$4"\nBASEF="$5"\nBASER="$6"\nLOG="$7"\nstop() { echo "STOP $*"; exit 3; }\nlog() { echo "LOG $*"; }\ncheck_time() { :; }\n' + funcs + "step_gapfiles; echo RC=$?\n", encoding="utf-8")

    def go(gf, baser):
        r = subprocess.run([bash_exe(), str(sc), posix(sys.executable), posix(HERE), posix(gf), posix(world["jp"]), posix(world["bp"]), posix(baser), posix(tmp_path / "log.txt")],
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
        return r.returncode, r.stdout
    gf = tmp_path / "gapfiles"
    rc, out = go(gf, world["rp"])
    assert rc == 0 and "LOG ADMIT-OK files 4" in out and (gf / "FALLATTR_GAPFILES_RECORD.json").is_file(), out
    rc, out = go(gf, world["rp"])                                                  # present folder: not built again, admitted again
    assert rc == 0 and "ADMIT-OK" in out
    z = MAG.load_npz(str(gf / "VGAPA_LM.npz"))                                    # a changed file is refused by its hash
    z["gap"] = z["gap"] + 1.0
    np.savez(str(gf / "VGAPA_LM.npz"), **z)
    rc, out = go(gf, world["rp"])
    assert rc == 3 and "GAPFILE-ADMISSION GAPFILE-HASH-DIFFERS LM" in out, out
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps({"files": {"VGAPLE": {"file_sha256": "1" * 64}}}), encoding="utf-8")
    rc, out = go(tmp_path / "gapfiles2", bad_rec)                                  # a0 fails (base not recorded): stop
    assert rc == 3 and "A0-NOT-PASS" in out, out


def test_rollout_function_is_the_model_one_with_two_insertions_and_the_header_facts_hold():
    q, m = QUEUE.read_text(encoding="utf-8"), MODEL_Q.read_text(encoding="utf-8")
    assert hashlib.sha256(MODEL_Q.read_bytes()).hexdigest()[:16] == "5c079e6b279e350b"
    mine, model = fn_text(q, "rollout"), fn_text(m, "rollout")
    assert mine != model
    assert mine.replace('  [ -f "$out/T1_DONE" ] && return 0\n', "").replace('  check_time "rollout $arm"\n', "") == model
    assert mine.count("T1_DONE") == 2 and mine.count("check_time") == 1                       # the second T1_DONE is the touch of the model
    for need in ('[ "$got" = "28bc31f21ecb4ccb" ]', 'grep -q "28bc31f21ecb4ccb" "$CHK/CHECK_WRAPPER_V.hashes"', "ZZ CHECK-WRAPPER-V PASS", "MAXGPU=3", "MINGB=${I10_MIN_FREE_GB:-28}", "set -C",
                 "kit_gap", "kit_read", "/workspace/v8proof/cropscale/gapfiles/VGAPLE.npz", "/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json", "CUDA_VISIBLE_DEVICES=-1",
                 'NAMES="I10RQ50KF5P30Hb20aLA I10RQ50KF5P30Hb20aLN I10RQ50KF5P30Hb20aLM I10RQ50KF5P30Hb20aLF"'):
        assert need in q, need
    base = [x for x in m.splitlines() if "rollout I10RQ50KF5P30Hb20gG2 RR" in x][0]
    env0 = base.strip().split(" VGAP=")[0]
    mine_lines = [x for x in q.splitlines() if "rollout I10RQ50KF5P30Hb20a" in x]
    assert len(mine_lines) == 4
    for a, x in zip(ARMS, mine_lines):
        assert x.strip().startswith(env0) and (' VGAP="$GF/VGAPA_%s.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20a%s RR "$k" "$k"' % (a, a)) in x


def test_the_queue_is_regenerated_byte_for_byte_by_its_generator():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VQ_OVERWRITE_OWN="1")
    before = QUEUE.read_bytes()
    r = subprocess.run([sys.executable, str(HERE / "make_fallattr_queue.py")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100)
    assert r.returncode == 0 and "rollouts 8" in r.stdout, r.stdout[-400:]
    assert QUEUE.read_bytes() == before


def test_the_queue_never_deletes_moves_or_kills():
    code = "\n".join(x for x in QUEUE.read_text(encoding="utf-8").splitlines() if not x.lstrip().startswith("#"))
    for bad in (r"\brm\b", r"\brmdir\b", r"\bmv\b", r"\bkill\b", r"\bpkill\b", r"\btruncate\b", r"\bshred\b"):
        assert not re.search(bad, code), bad


# ------------------------------------------------------------------------------------------------ push script, wording, paths
def test_push_script_and_texts():
    pu = (HERE / "push_fallattr_gap.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "lead_pytest.out" in pu and "/workspace/v8proof/fallattr" in pu and "kit_gap" in pu and "sha256sum" in pu and "ZZ PUSH-FALLATTR-GAP" in pu
    for f in ("make_attr_gapfiles.py", "make_fallattr_queue.py", "pod_fallattr10k_queue.sh"):
        assert f in pu and (HERE / f).is_file()
    assert not re.search(r"ssh |root@|@[0-9]+\.[0-9]+", pu)                                   # no pod address
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "push_fallattr_gap.sh")]).returncode == 0


def test_wording_scan_and_path_length():
    for f in ("make_attr_gapfiles.py", "make_fallattr_queue.py", "pod_fallattr10k_queue.sh", "push_fallattr_gap.sh", "test_fallattr_gap.py", "FALLATTR_GAP_DEFS.md"):
        p = HERE / f
        assert p.is_file(), f
        t = p.read_text(encoding="utf-8")
        assert not [w for w in BANNED if w in t], (f, [w for w in BANNED if w in t])
        assert "\r" not in t and len(posix(p.relative_to(V8))) <= 200
    for p in HERE.iterdir():
        if p.is_file():
            assert not re.match(r"(?i)^(aux|con|prn|nul)(\.|$)", p.name)
            assert p.stat().st_size < 5 * 1024 * 1024
