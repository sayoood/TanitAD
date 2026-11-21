"""test_visseed.py -- tests of the kit of card ADD104 (VISSEED10K): the gap files of the fit seeds 2, 3, 4 with controls x0 and x1, the fit queue, the rollout queue, the reader controls and names, with planted faults.
Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest visseed/test_visseed.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/visseed/tmp_pt/<new name>"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_seed_gapfile as MSG               # noqa: E402
import read_visseed10k as RE                  # noqa: E402

QUEUE = HERE / "pod_visseed10k_queue.sh"
FITQ = HERE / "pod_visseed_fit_queue.sh"
Q103 = HERE.parent / "visrep" / "pod_visrep10k_queue.sh"
Q102 = HERE.parent / "visrule" / "pod_visrule10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
W16 = "28bc31f21ecb4ccb"


def bash_exe():
    return shutil.which("bash")


def posix(p):
    return Path(p).as_posix()


# ------------------------------------------------------------------------------------------------ the gap files, x0 and x1
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("gf")
    J, dumps = MSG.synthetic_all()
    fits, meta, fit_root, old_root = MSG.synthetic_fit_world(str(base / "w"), J, dumps)
    rec = MSG.build(J, dumps, str(base / "clean"), meta=meta, expect_clips=len(J["clip"]), fits=fits, eval_join_sha="9" * 64)
    return {"J": J, "dumps": dumps, "fits": fits, "meta": meta, "fit_root": fit_root, "old_root": old_root, "rec": rec, "base": base}


def test_generator_source_hash_is_the_file_on_disk_and_the_derivation_is_visible():
    src = HERE.parent / "visrule" / "make_vis_gapfile.py"
    gen = (HERE / "gen_seed_gapfile.py").read_text(encoding="utf-8")
    assert hashlib.sha256(src.read_bytes()).hexdigest()[:16] == "97a90ec432bcfc5c" and "97a90ec432bcfc5c" in gen
    t = (HERE / "make_seed_gapfile.py").read_text(encoding="utf-8")
    assert 'DUMP_DIR = "D_H5"' in t and 'SEEDS = (2, 3, 4)' in t and '"VGAP_s%d"' not in t and "VISSEED_GAPFILES_RECORD.json" in t and 'CARD = "PREREG_V8F_PROOF_ADD104"' in t
    gq = (HERE / "make_visseed_queue.py").read_text(encoding="utf-8")
    assert hashlib.sha256(Q103.read_bytes()).hexdigest()[:16] == "ac22955967ea3cb7" and "ac22955967ea3cb7" in gq


def test_clean_build_passes_x0_and_x1_and_has_the_three_files(world):
    r = world["rec"]
    assert r["x0"]["PASS"] and r["x1"]["PASS"] and r["v0"]["PASS"] and r["build"]["PASS"] and r["meta"]["PASS"], (r["x0"], r["x1"])
    assert sorted(r["files"]) == ["VGAPF_s2", "VGAPF_s3", "VGAPF_s4"] and r["card"] == "PREREG_V8F_PROOF_ADD104"
    assert (world["base"] / "clean" / "VISSEED_GAPFILES_RECORD.json").is_file() and (world["base"] / "clean" / "VGAPF_s3.npz").is_file()
    for s in (2, 3, 4):
        a = r["files"]["VGAPF_s%d" % s]
        assert a["dump_dir_name"] == "D_H5" and a["fit_seed"] == s and a["max_abs_dev_saved_minus_second_routine"] <= 1e-9 and a["n_rows"] == a["n_join_rows"]
        assert a["n_without_output"] >= 4 * len(world["J"]["clip"])            # the head H5 has no output at frames 0 to 3
        assert r["dumps"]["VGAPF_s%d" % s]["dump_meta"]["head_pt_sha256"] == r["fits"]["new"]["s%d" % s]["head_disk_sha256"]


def test_each_planted_fault_turns_its_check_red(world, tmp_path):
    vf = MSG.v0_faults(world["J"], world["dumps"])
    assert vf and all(vf.values()), vf
    bf = MSG.build_faults(world["J"], world["dumps"], str(tmp_path / "bf"), world["meta"])
    assert sorted(bf) == ["clip60", "min_label", "scale", "shift", "zero_nooutput"] and all(bf.values()), bf
    xf = MSG.x_faults(str(tmp_path / "xf"), world["J"], world["dumps"])
    assert xf["clean_x0_and_x1_pass"] and all(xf.values()) and len(xf) >= 24, xf
    rec = MSG.build(world["J"], world["dumps"], str(tmp_path / "nofits"), meta=world["meta"])           # no facts of the fits: x0 is not PASS, x1 is not PASS (no meta clause)
    assert not rec["x0"]["PASS"] and not rec["x1"]["PASS"]


def test_x0_names_the_card_clauses_one_by_one(world):
    f = copy.deepcopy(world["fits"])
    assert MSG.x0_check(f)["PASS"]
    f["new"]["s2"]["fit_clips"] = f["new"]["s2"]["fit_clips"][:-1]
    r = MSG.x0_check(f)
    assert not r["PASS"] and any("train clips" in x for x in r["failures"])
    f = copy.deepcopy(world["fits"])
    f["new"]["s4"]["head_pt_sha256_record"] = f["old"]["s1"]["head_pt_sha256_record"]
    f["new"]["s4"]["head_disk_sha256"] = f["old"]["s1"]["head_pt_sha256_record"]
    assert any("same file" in x for x in MSG.x0_check(f)["failures"])
    f = copy.deepcopy(world["fits"])
    f["new"]["s3"]["head"] = "H1"
    assert not MSG.x0_check(f)["PASS"]
    assert not MSG.x0_check(None)["PASS"] and not MSG.x0_check({})["PASS"]


def test_main_through_files_refusals_and_arguments(world, tmp_path):
    ej = tmp_path / "join.npz"
    np.savez(str(ej), **world["J"])
    args = ["--eval-join", str(ej), "--fit-root", world["fit_root"], "--out-dir", str(tmp_path / "out"), "--add99-fits", world["old_root"], "--expect-clips", str(len(world["J"]["clip"]))]
    assert MSG.main(args) == 0
    rec = json.load(open(str(tmp_path / "out" / "VISSEED_GAPFILES_RECORD.json"), encoding="utf-8"))
    assert rec["x0"]["PASS"] and rec["x1"]["PASS"] and rec["eval_join_sha256"] == MSG.sha_file(str(ej))
    with pytest.raises(SystemExit):
        MSG.main(args)                                                          # the out dir exists
    assert MSG.main([]) == 2 and MSG.main(args[:2] + args[4:]) == 2             # --fit-root is required
    bad = tmp_path / "old_bad"
    shutil.copytree(world["old_root"], str(bad))
    os.remove(str(bad / "D_H5_s1" / "FIT_RECORD.json"))
    assert MSG.main(["--eval-join", str(ej), "--fit-root", world["fit_root"], "--out-dir", str(tmp_path / "out2"), "--add99-fits", str(bad), "--expect-clips", str(len(world["J"]["clip"]))]) == 1


# ------------------------------------------------------------------------------------------------ the fit queue
def run_bash(script, env_extra, cwd=None, args=()):
    env = dict(os.environ, **env_extra)
    r = subprocess.run([bash_exe(), str(script)] + list(args), env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=240)
    return r.returncode, r.stdout


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_dry_run_prints_every_command_and_runs_nothing(tmp_path):
    assert subprocess.run([bash_exe(), "-n", str(FITQ)]).returncode == 0
    rc, out = run_bash(FITQ, {"DRY": "1", "VS_ROOT": posix(tmp_path / "gp")})
    assert rc == 0 and not (tmp_path / "gp").exists(), out[-500:]
    cmds = [x for x in out.splitlines() if x.strip().startswith("CMD: ")]
    fits = [x for x in cmds if "crop_fit.py" in x]
    dumps = [x for x in cmds if "crop_dump.py" in x]
    assert len(fits) == 3 and len(dumps) == 3
    for s, c in zip((2, 3, 4), fits):
        assert ("--arm D --head H5 --seed %d --root " % s) in c and "/workspace/v8proof/cropread/kit/fit/crop_fit.py" in c and "--join-train /workspace/v8proof/data/inputs/join_train.npz" in c and "--device cuda" in c
        assert "--root %s/fitroot" % posix(tmp_path / "gp") in c and "--permute-labels" not in c
    for s, c in zip((2, 3, 4), dumps):
        assert ("--arm D --head H5 --seed %d --root " % s) in c and ("--present-dump %s/present_tpl" % posix(tmp_path / "gp")) in c and "/workspace/v8proof/cropread/kit/fit/crop_dump.py" in c
    links = [x for x in cmds if "ln -s" in x]
    assert len(links) == 1 + 3 + 3
    assert any(x.endswith("/workspace/v8proof/cropread/run1/tok %s/fitroot/tok" % posix(tmp_path / "gp")) for x in links)
    for s in (2, 3, 4):
        assert any(("ln -s /workspace/v8proof/closeread/dump/both/reader_s0_score.npz %s/present_tpl/reader_s%d_score.npz" % (posix(tmp_path / "gp"), s)) in x for x in links)
    for x in links:                                                              # nothing is linked INTO the run root of card ADD99
        assert not x.split()[-1].startswith("/workspace/v8proof/cropread/run1")


def make_stub_world(tmp_path):
    r1, pre, kit, gp = tmp_path / "run1", tmp_path / "present", tmp_path / "kit", tmp_path / "gp"
    (r1 / "tok" / "D").mkdir(parents=True)
    (r1 / "tok" / "D" / "MANIFEST.json").write_text("{}", encoding="utf-8")
    (r1 / "lists").mkdir()
    for f in ("train_final.txt", "eval141.txt", "LISTS_RECORD.json"):
        (r1 / "lists" / f).write_text("x\n", encoding="utf-8")
    pre.mkdir()
    (pre / "reader_s0_score.npz").write_bytes(b"x")
    (tmp_path / "join.npz").write_bytes(b"x")
    kit.mkdir()
    (kit / "crop_fit.py").write_text(
        "import argparse, json, os, sys\nap = argparse.ArgumentParser()\nfor n in ('arm', 'head', 'seed', 'root', 'join-train', 'device'):\n    ap.add_argument('--' + n)\na = ap.parse_args()\n"
        "if os.environ.get('STUB_FAIL_SEED') == a.seed:\n    sys.exit(7)\nd = os.path.join(a.root, 'fits', 'D_H5_s' + a.seed)\nos.makedirs(d)\nopen(os.path.join(d, 'head.pt'), 'w').write('h' + a.seed)\n"
        "json.dump({'seed': int(a.seed)}, open(os.path.join(d, 'FIT_RECORD.json'), 'w'))\n", encoding="utf-8")
    (kit / "crop_dump.py").write_text(
        "import argparse, json, os, sys\nap = argparse.ArgumentParser()\nfor n in ('arm', 'head', 'seed', 'root', 'present-dump', 'device'):\n    ap.add_argument('--' + n)\na = ap.parse_args()\n"
        "assert os.path.isfile(os.path.join(a.root, 'fits', 'D_H5_s' + a.seed, 'FIT_RECORD.json'))\nd = os.path.join(a.root, 'dump', 'D_H5')\nos.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'reader_s%s_score.npz' % a.seed), 'w').write('d')\njson.dump({'seed': int(a.seed)}, open(os.path.join(d, 'reader_s%s_meta.json' % a.seed), 'w'))\n", encoding="utf-8")
    shim = tmp_path / "shim"                                                     # no real symbolic links on this file system: a shim `ln -s` and `readlink` that keep the target in a text file
    shim.mkdir()
    (shim / "ln").write_text('#!/bin/bash\n[ "$1" = -s ] || exit 9\nprintf \'SHIMLINK:%s\' "$2" > "$3"\n', encoding="utf-8")
    (shim / "readlink").write_text('#!/bin/bash\nc=$(cat "$1"); echo "${c#SHIMLINK:}"\n', encoding="utf-8")
    env = {"VS_RUN1": posix(r1), "VS_ROOT": posix(gp), "VS_KIT": posix(kit), "VS_PRESENT": posix(pre), "VS_JOIN_TRAIN": posix(tmp_path / "join.npz"), "VS_PY": posix(sys.executable),
           "PATH": posix(shim) + os.pathsep + os.environ["PATH"]}
    return gp, env


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_real_run_with_stub_tools_markers_resume_and_stop(tmp_path):
    gp, env = make_stub_world(tmp_path)
    for p in (tmp_path / "shim" / "ln", tmp_path / "shim" / "readlink"):
        os.chmod(str(p), 0o755)
    env["STUB_FAIL_SEED"] = "4"
    rc, out = run_bash(FITQ, env)
    assert rc == 3 and (gp / "STOP_FITS.txt").is_file() and (gp / "fits_q" / "jobs" / "fit_s4.STOP").is_file() and not (gp / "DONE_FITS.txt").exists(), out[-600:]
    assert (gp / "fits_q" / "jobs" / "fit_s2.DONE").is_file() and (gp / "fits_q" / "jobs" / "fit_s3.DONE").is_file() and not list((gp / "fitroot" / "dump").glob("*/*"))     # no dump before all three fits
    assert (gp / "fitroot" / "tok").read_text(encoding="utf-8").startswith("SHIMLINK:" + posix(tmp_path / "run1" / "tok"))
    rc, out = run_bash(FITQ, env)
    assert rc == 2 and "RESUME=1" in out                                         # a stop needs RESUME=1
    env.pop("STUB_FAIL_SEED")
    rc, out = run_bash(FITQ, dict(env, RESUME="1"))
    assert rc == 0 and (gp / "DONE_FITS.txt").is_file() and "ZZ VISSEED-FITQ DONE" in out, out[-600:]
    for s in (2, 3, 4):
        assert (gp / "fitroot" / "fits" / ("D_H5_s%d" % s) / "FIT_RECORD.json").is_file() and (gp / "fitroot" / "dump" / "D_H5" / ("reader_s%d_meta.json" % s)).is_file()
        assert (gp / "fits_q" / "jobs" / ("dump_s%d.DONE" % s)).is_file() and (gp / "present_tpl" / ("reader_s%d_score.npz" % s)).is_file()
    assert len((gp / "START_FITS.txt").read_text(encoding="utf-8").splitlines()) >= 3
    rc, out = run_bash(FITQ, env)
    assert rc == 0 and "ALREADY-DONE" in out
    assert not (tmp_path / "run1" / "fits").exists() and not (tmp_path / "run1" / "dump").exists()           # nothing is written into the run root of card ADD99


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_refuses_a_missing_input_and_stops(tmp_path):
    gp, env = make_stub_world(tmp_path)
    os.remove(str(tmp_path / "run1" / "lists" / "eval141.txt"))
    rc, out = run_bash(FITQ, env)
    assert rc == 3 and "MISSING-INPUT" in out and (gp / "STOP_FITS.txt").is_file() and not (gp / "fitroot").exists()


# ------------------------------------------------------------------------------------------------ the rollout queue
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def dry_commands(q, seeds="ALL"):
    env = dict(os.environ, DRY="1", GR_SEED=seeds)
    out = subprocess.run([bash_exe(), str(q)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100).stdout
    return [x for x in out.splitlines() if x.startswith("CMD: ")], out


def test_queue_rollout_function_is_the_one_of_cards_add102_and_add103_and_the_header_facts_hold():
    q, q3, q2 = QUEUE.read_text(encoding="utf-8"), Q103.read_text(encoding="utf-8"), Q102.read_text(encoding="utf-8")
    assert hashlib.sha256(q3.encode("utf-8")).hexdigest()[:16] == "ac22955967ea3cb7" and hashlib.sha256(q2.encode("utf-8")).hexdigest()[:16] == "6321d4e6e12c0f47"
    assert fn_text(q) == fn_text(q3) == fn_text(q2)
    assert "\r" not in q and not [w for w in BANNED if w in q]
    for bad in ("VISREP", "visrep", "VISRULE10K_", "queue_visrule", "$G2", "VGAP1", "pKD5", "rollout I10RQ50KF5P30Hb20w", "rollout I10RQ50KF5P30Hb20v"):
        assert bad not in q, bad
    for need in ("GP=${GR_ROOT:-/workspace/v8proof/visseed10k}", "CHK=${GR_CHECK:-/workspace/v8proof/visrule10k/check_v}", '"$GP/VISSEED10K_STOP$PS.txt"', "START_VISSEED10K$PS.txt", "VISSEED10K_DONE$PS.txt",
                 "queue_visseed$PS.log", '[ "$got" = "%s" ]' % W16, 'grep -q "%s" "$CHK/CHECK_WRAPPER_V.hashes"' % W16, "ZZ CHECK-WRAPPER-V PASS",
                 '"VGAPF_s2,VGAPF_s3,VGAPF_s4" "x0,x1")', "VISSEED_GAPFILES_RECORD.json", "I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20wVD5cx", 'NAMES="I10RQ50KF5P30Hb20fF2 I10RQ50KF5P30Hb20fF3 I10RQ50KF5P30Hb20fF4"'):
        assert need in q, need


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_has_six_commands_with_the_right_arguments_and_syntax_is_valid():
    assert subprocess.run([bash_exe(), "-n", str(QUEUE)]).returncode == 0
    cmds, out = dry_commands(QUEUE)
    assert len(cmds) == 6, out[-800:]
    seen = {}
    for c in cmds:
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and "--rule-join" not in c and "--vis-closing" in c
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        for s in (2, 3, 4):
            for k in (0, 1):
                if ("/I10RQ50KF5P30Hb20fF%d_s%d " % (s, k)) in c:
                    seen[(s, k)] = 1
                    assert ("--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/" % k) in c and ("QMAP_LR16_s%d_q50" % k) in c
                    assert "--vis-gap-file /workspace/v8proof/visseed10k/gapfiles/VGAPF_s%d.npz" % s in c, (s, k)          # the gap file of fit seed S, the same for both model seeds
                    assert c.count("--vis-gap-file") == 1
    assert sorted(seen) == [(s, k) for s in (2, 3, 4) for k in (0, 1)]
    one, _ = dry_commands(QUEUE, "0")
    assert len(one) == 3 and all("_s0 " in c and "TK16_s0" in c and "TK16_s1" not in c for c in one)
    assert [x for x in one if "VGAPF_s3.npz" in x] and [x for x in one if "VGAPF_s4.npz" in x]


def test_the_arguments_of_the_rollouts_are_those_of_the_vd5c_call_of_the_add102_queue():
    q0 = Q102.read_text(encoding="utf-8")
    base = [x for x in q0.splitlines() if "rollout I10RQ50KF5P30Hb20vVD5c RR" in x][0]
    env0 = base.split(" VGAP=")[0].strip()
    mine = [x for x in QUEUE.read_text(encoding="utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20fF" in x]
    assert len(mine) == 3
    for s, x in zip((2, 3, 4), mine):
        assert x.strip().startswith(env0) and (' VGAP="$GF/VGAPF_s%d.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20fF%d RR "$k" "$k"' % (s, s)) in x
    assert 'VGAP="$GF/VGAP_s$k.npz" VCLOSE=1' in base


# ------------------------------------------------------------------------------------------------ the reader
def iv(lo, hi):
    return {"interval": [lo, hi], "pooled": 0.5 * (lo + hi)}


def payload(rr=0.0954, rlc=0.0, arms=None):
    """arms {name: (C(X) vs RLc interval)}; default: all three at [0.0, 0.004]"""
    arms = arms or {a: (0.0, 0.004) for a in RE.ARMS}
    pairs = {}
    for a, r in arms.items():
        pairs[a + "-VS-RR"] = {"C": iv(r[0] - rr, r[1] - rr), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-RLc"] = {"C": iv(*r), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-VD5c"] = {"C": iv(r[0] - 0.0169, r[1] - 0.0169), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}}, "pairs": pairs}


def test_name_rules_on_planted_intervals_and_their_edges():
    n = RE.names_of("F2", [-0.05, -0.01], [0.0, 0.004], [0.001, 0.02], [0.001, 0.03], 0.0954, 0.0)
    assert n["F2_CONTACT"] == "VISSEED10K-F2-CUTS-CONTACT" and n["F2_HALF"] == "VISSEED10K-F2-WITHIN-HALF-THE-GAP" and n["F2_VD5C"] == "VISSEED10K-F2-MORE-CONTACT-THAN-VD5c" and n["F2_PRICE"] == "VISSEED10K-F2-FALLS-BEHIND-MORE"
    n = RE.names_of("F3", [0.001, 0.01], [0.05, 0.06], [-0.02, -0.001], [-0.03, -0.01], 0.0954, 0.0)
    assert n["F3_CONTACT"].endswith("-F3-MORE-CONTACT") and n["F3_HALF"].endswith("-F3-HALF-THE-GAP-NOT-REACHED") and n["F3_VD5C"].endswith("-F3-LESS-CONTACT-THAN-VD5c") and n["F3_PRICE"].endswith("-F3-FALLS-BEHIND-LESS")
    n = RE.names_of("F4", [-0.04, 0.01], [0.01, 0.06], [-0.01, 0.01], [-0.02, 0.01], 0.0954, 0.0)
    assert n["F4_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["F4_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["F4_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c") and n["F4_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    assert RE.names_of("F2", [0, 0], [0.0, 0.0477], [0, 0], [0, 0], 0.0954, 0.0)["F2_HALF"].endswith("WITHIN-HALF-THE-GAP")         # an upper end exactly at H is WITHIN
    assert RE.names_of("F2", [0, 0], [0.0477, 0.06], [0, 0], [0, 0], 0.0954, 0.0)["F2_HALF"].endswith("NOT-DECIDED")               # a lower end exactly at H is not NOT-REACHED
    assert RE.names_of("F2", [0, 0], [0, 0], [0.0, 0.01], [0, 0], 0.0954, 0.0)["F2_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c")
    assert RE.names_of("F2", [-0.05, 0.0], [0, 0], [0, 0], [0, 0], 0.0954, 0.0)["F2_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR")   # an upper end exactly at 0 does not cut


def test_the_card_name_follows_the_count_rule_word_for_word():
    P = RE.PRE
    C, M, N = "CUTS-CONTACT", "MORE-CONTACT", "CONTACT-NOT-SEPARATED-FROM-RR"
    W, X, D = "WITHIN-HALF-THE-GAP", "HALF-THE-GAP-NOT-REACHED", "HALF-THE-GAP-NOT-DECIDED"

    def names(c, h):
        out = {}
        for i, a in enumerate(RE.ARMS):
            out[a + "_CONTACT"], out[a + "_HALF"] = "%s-%s-%s" % (P, a, c[i]), "%s-%s-%s" % (P, a, h[i])
        return out
    cases = [([C, C, C], [W, W, W], "HOLDS"), ([C, C, C], [W, W, D], "HOLDS"), ([C, C, C], [W, D, D], "NOT-DECIDED"), ([C, C, C], [D, D, D], "NOT-DECIDED"), ([C, C, C], [W, W, X], "DOES-NOT-HOLD"),
             ([C, C, C], [X, D, D], "DOES-NOT-HOLD"), ([C, C, N], [W, W, W], "DOES-NOT-HOLD"), ([C, M, C], [W, W, W], "DOES-NOT-HOLD"), ([N, N, N], [D, D, D], "DOES-NOT-HOLD"), ([C, C, N], [W, W, D], "DOES-NOT-HOLD")]
    for c, h, want in cases:
        assert RE.holds_name(names(c, h)) == P + "-" + want, (c, h)


def test_all_names_card_name_and_the_second_derivation():
    names = RE.all_names(payload())
    assert names["CARD"] == "VISSEED10K-HOLDS" and len(names) == 3 * 4 + 1
    assert all(k.split("_")[0] in RE.ARMS for k in names if k != "CARD")
    p2 = payload(arms={"F2": (0.0, 0.004), "F3": (0.0, 0.06), "F4": (0.0, 0.07)})                  # one WITHIN: not enough
    assert RE.all_names(p2)["CARD"] == "VISSEED10K-NOT-DECIDED"
    p3 = payload(arms={"F2": (0.0, 0.004), "F3": (0.0, 0.004), "F4": (0.05, 0.06)})                # F4 lower end above H
    assert RE.all_names(p3)["CARD"] == "VISSEED10K-DOES-NOT-HOLD" and RE.all_names(p3)["F4_HALF"].endswith("HALF-THE-GAP-NOT-REACHED")
    p4 = payload(arms={"F2": (0.09, 0.1), "F3": (0.0, 0.004), "F4": (0.0, 0.004)})                 # F2 does not cut: its upper end minus RR is above 0
    assert RE.all_names(p4)["CARD"] == "VISSEED10K-DOES-NOT-HOLD"
    p = payload()
    assert RE.card_check(p, {"CARD": "VISSEED10K-DOES-NOT-HOLD"})["PASS"] is False and RE.card_check(p, RE.all_names(p))["PASS"]
    h = RE.h_sensitivity(p)
    assert h["H_card"] == 0.0477 and abs(h["H_point_values"] - 0.0477) < 1e-12 and set(RE.ARMS) <= set(h)


def test_contact_table_maps_fit_seeds_as_the_card_says():
    ls = {"VD5c": {"C": [0.02, 0.01]}, "VD5cx": {"C": [0.03, 0.04]}, "F2": {"C": [0.05, 0.06]}, "F3": {"C": [0.07, 0.08]}, "F4": {"C": [0.09, 0.10]}}
    t = RE.contact_table(ls)
    assert t["per_fit"]["fit0"]["model_seed_0"] == 0.02 and t["per_fit"]["fit0"]["model_seed_1"] == 0.04          # fit seed 0: VD5c at model seed 0, VD5cx at model seed 1
    assert t["per_fit"]["fit1"]["model_seed_0"] == 0.03 and t["per_fit"]["fit1"]["model_seed_1"] == 0.01          # fit seed 1: VD5cx at model seed 0, VD5c at model seed 1
    assert t["per_fit"]["fit3"]["model_seed_0"] == 0.07 and t["per_fit"]["fit3"]["model_seed_1"] == 0.08 and abs(t["per_fit"]["fit3"]["pooled"] - 0.075) < 1e-12 and sorted(t["per_fit"]) == ["fit0", "fit1", "fit2", "fit3", "fit4"]
    pooled = [0.5 * (0.02 + 0.04), 0.5 * (0.03 + 0.01), 0.055, 0.075, 0.095]
    o = t["over_five_fits_pooled"]
    assert abs(o["mean"] - sum(pooled) / 5) < 1e-12 and o["min"] == min(pooled) and o["max"] == max(pooled)
    assert t["over_five_fits_model_seed_0"]["max"] == 0.09 and t["over_five_fits_model_seed_1"]["min"] == 0.01


def test_control_x4_x5_x6_and_x2():
    lv = {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "VD5cx": 0.0200}
    assert RE.ctl_x4(lv)["PASS"] and RE.ctl_x4(dict(lv, RR=0.09544, VD5cx=0.01996))["PASS"]
    for k, v in (("RR", 0.0953), ("VD5c", 0.0177), ("VD5cx", 0.0208), ("RLc", 0.0005), ("VD5cx", None)):
        assert not RE.ctl_x4(dict(lv, **{k: v}))["PASS"], k
    assert not RE.ctl_x4({k: v for k, v in lv.items() if k != "VD5cx"})["PASS"]
    assert RE.ctl_x5("12 passed in 3s")["PASS"] and not RE.ctl_x5("11 passed, 1 failed in 3s")["PASS"] and not RE.ctl_x5("")["PASS"] and not RE.ctl_x5("1 passed, 2 error in 1s")["PASS"]
    assert RE.ctl_x6(True, True, True)["PASS"] and not RE.ctl_x6(True, True, False)["PASS"] and not RE.ctl_x6(False, True, True)["PASS"] and not RE.ctl_x6(None, True, True)["PASS"]
    shas = {"%s_s%d" % (a, s): W16 + "0" * 48 for a in RE.ARMS for s in (0, 1)}
    assert len(shas) == 6 and RE.ctl_x2(W16, "x\nZZ CHECK-WRAPPER-V PASS\n", "abc " + W16 + " v8f\n", shas)["PASS"]
    assert not RE.ctl_x2("0" * 16, "ZZ CHECK-WRAPPER-V PASS\n", W16, shas)["PASS"]
    assert not RE.ctl_x2(W16, "ZZ CHECK-WRAPPER-V FAIL\n", W16, shas)["PASS"]
    assert not RE.ctl_x2(W16, "ZZ CHECK-WRAPPER-V PASS\n", "no hash here", shas)["PASS"]
    assert not RE.ctl_x2(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, F3_s0="1" * 64))["PASS"]
    assert not RE.ctl_x2(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, F2_s1=None))["PASS"]


def rows(ci, j, pres, g, c=None):
    ci, j, pres, g = np.asarray(ci), np.asarray(j), np.asarray(pres, bool), np.asarray(g, float)
    return (ci.astype(np.int64), j.astype(np.int64), pres, g, np.full(len(g), -0.4) if c is None else np.asarray(c, float))


def test_control_x3_on_planted_rows():
    ref = {(0, 4): 10.0, (0, 5): 11.0, (0, 3): None, (1, 4): 20.0}
    v0 = {(0, 3): [7.0], (0, 4): [9.0], (0, 5): [9.5, 9.5004], (1, 4): [19.0]}
    K = ([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1])
    good = rows(*K, [7.0, 10.0, 11.0005, 20.0])
    r = RE.w2_one(good, ref, v0)
    assert r["PASS"] and r["n_lead_rows_with_head_output"] == 3 and r["n_lead_rows_without_head_output"] == 1
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.002, 20.0]), ref, v0)["PASS"]
    assert not RE.w2_one(rows(*K, [7.1, 10.0, 11.0, 20.0]), ref, v0)["PASS"]
    assert not RE.w2_one(rows(*K, [7.0, 9.0, 11.0, 20.0]), ref, v0)["PASS"]
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0]), ref, v0)["PASS"]
    assert RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0.5]), ref, v0)["PASS"]
    assert not RE.w2_one(rows([0, 0], [3, 4], [1, 1], [7.0, 10.0]), ref, {(0, 4): [9.0]})["PASS"]
    assert not RE.w2_one(rows([0, 0], [3, 4], [0, 0], [7.0, 10.0]), ref, v0)["PASS"]
    exp = {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1}
    assert RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "b" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 0})["PASS"]


def write_logs(pull, run, m, tick, mid, meta):
    d = Path(pull) / ("I10%s_s%d" % (run, m)) / "ticks"
    d.mkdir(parents=True, exist_ok=True)
    for nm, r in (("rule_log_I10%s_s%d.npz" % (run, m), tick), ("rule_log_mid_I10%s_s%d.npz" % (run, m), mid)):
        np.savez(str(d / nm), ci=r[0], j=r[1], present=r[2], g_src=r[3], c_src=r[4], **{k: np.asarray(v) for k, v in meta.items()})
    (d / ("run_I10%s_s%d.txt" % (run, m))).write_text("arm x wrapper sha256 %s started t\n" % (W16 + "0" * 48), encoding="utf-8")


def reader_gap(ci, j):
    return np.float32(20.0 + 0.1 * j + ci)


class Ns:
    pass


def synth_pull(tmp_path, world, spoil=None, closing_zero=None, meta_bad=None, wrong_fit=None):
    pull = tmp_path / "pull"
    J, n = world["J"], np.asarray(world["J"]["n"])
    refs = {s: MSG.second_routine(J, world["dumps"][s]) for s in (2, 3, 4)}
    for m in (0, 1):
        for arm in ("V0", "F2", "F3", "F4"):
            run = RE.V0_RUN if arm == "V0" else RE.RUN[arm]
            fs = int(arm[1:]) if arm != "V0" else None
            if wrong_fit and wrong_fit[0] == (arm, m):
                fs = wrong_fit[1]
            cols = {0: [[], [], [], []], 1: [[], [], [], []]}
            for c in range(len(n)):
                for j in range(int(n[c])):
                    own = reader_gap(c, j)
                    h = refs[fs].get((c, j)) if fs else None
                    g = np.float32(h) if h is not None else own
                    t = cols[j % 2]
                    t[0].append(c), t[1].append(j), t[2].append(g)
                    t[3].append(0.0 if arm == "V0" else -0.5)
                    if arm == "V0" and j % 2 == 0:                                   # a frame logged twice with the float noise of the reader
                        t[0].append(c), t[1].append(j), t[2].append(np.float32(own + 0.0004)), t[3].append(0.0)
            if spoil and spoil[0] == (arm, m):
                cols[0][2][spoil[1]] = cols[0][2][spoil[1]] + 0.01
            if closing_zero == (arm, m):
                cols[0][3] = [0.0] * len(cols[0][3])
                cols[1][3] = [0.0] * len(cols[1][3])
            exp = "" if arm == "V0" else world["rec"]["files"]["VGAPF_s%d" % int(arm[1:])]["file_sha256"]
            if meta_bad == (arm, m):
                exp = "f" * 64
            meta = {"vis_gap_file_sha256": exp, "vis_closing": 0 if arm == "V0" else 1}
            tick, mid = [rows(cols[k][0], cols[k][1], [True] * len(cols[k][0]), cols[k][2], cols[k][3]) for k in (0, 1)]
            write_logs(pull, run, m, tick, mid, meta)
    return pull


def test_control_x3_end_to_end_with_the_three_arms_and_planted_faults(world, tmp_path):
    ej = tmp_path / "join.npz"
    np.savez(str(ej), **world["J"])
    a = Ns()
    a.eval_join, a.fit_root = str(ej), world["fit_root"]
    a.pull = str(synth_pull(tmp_path / "a", world))
    c3 = RE.x3_from_files(a, world["rec"])
    assert c3["PASS"], c3
    assert sorted(c3["per_run"]) == sorted("F%d_s%d" % (s, m) for s in (2, 3, 4) for m in (0, 1))
    assert all(v["n_lead_rows_with_head_output"] > 0 and v["share_of_lead_steps_with_nonzero_closing"] > 0 for v in c3["per_run"].values())
    for label, kw in (("spoil_f3", dict(spoil=(("F3", 1), 5))), ("closing_zero", dict(closing_zero=("F4", 1))), ("meta", dict(meta_bad=("F2", 0))), ("wrong_fit", dict(wrong_fit=((("F2", 0)), 3)))):
        a.pull = str(synth_pull(tmp_path / label, world, **kw))
        assert not RE.x3_from_files(a, world["rec"])["PASS"], label
    a.pull = str(synth_pull(tmp_path / "recwrong", world))
    wrong = copy.deepcopy(world["rec"])
    wrong["files"]["VGAPF_s4"]["file_sha256"] = "e" * 64                                       # the gap file the rule log names is not the one of the record
    assert not RE.x3_from_files(a, wrong)["PASS"]


def test_controls_x0_and_x1_on_the_record_and_the_files(world, tmp_path):
    gdir = str(world["base"] / "clean")
    ejs = "9" * 64
    fits_now, meta_now = MSG.collect(world["fit_root"], world["old_root"], os.path.join(world["fit_root"], "lists", "eval141.txt"), os.path.join(world["fit_root"], "dump"))
    assert RE.ctl_x0(world["rec"], fits_now)["PASS"] and RE.ctl_x1(world["rec"], gdir, ejs, fits_now, meta_now)["PASS"]
    assert not RE.ctl_x1(world["rec"], gdir, "8" * 64, fits_now, meta_now)["PASS"]                 # another evaluation join
    assert not RE.ctl_x1(world["rec"], str(world["base"]), ejs, fits_now, meta_now)["PASS"]        # the gap files are not where the record says
    assert not RE.ctl_x0(None, fits_now)["PASS"] and not RE.ctl_x1(None, gdir, ejs, fits_now, meta_now)["PASS"]
    r2 = copy.deepcopy(world["rec"])
    r2["x0"]["PASS"] = False
    assert not RE.ctl_x0(r2, fits_now)["PASS"]
    r3 = copy.deepcopy(world["rec"])
    r3["x1"]["PASS"] = False
    assert not RE.ctl_x1(r3, gdir, ejs, fits_now, meta_now)["PASS"]
    # a head file changed after the record was built: the facts read now differ from the ones the record saw
    root2 = tmp_path / "w2"
    fits2, meta2, fr2, old2 = MSG.synthetic_fit_world(str(root2), world["J"], world["dumps"])
    rec2 = MSG.build(world["J"], world["dumps"], str(tmp_path / "b2"), meta=meta2, expect_clips=len(world["J"]["clip"]), fits=fits2, eval_join_sha=ejs)
    with open(os.path.join(fr2, "fits", "D_H5_s3", "head.pt"), "ab") as fh:
        fh.write(b"changed")
    f_now, m_now = MSG.collect(fr2, old2, os.path.join(fr2, "lists", "eval141.txt"), os.path.join(fr2, "dump"))
    assert not RE.ctl_x0(rec2, f_now)["PASS"] and not RE.ctl_x1(rec2, str(tmp_path / "b2"), ejs, f_now, m_now)["PASS"]


def test_a_withheld_payload_and_the_names_stage_refuses_with_a_control_failed(tmp_path):
    assert RE.CONTROLS == ("x0", "x1", "x2", "x3", "x4", "x5", "x6")
    ok = {k: {"PASS": True} for k in RE.CONTROLS}
    bad = dict(ok, x3={"PASS": False})
    p = payload()
    r = RE.assemble_controls(bad, p)
    assert r["controls_all_pass"] is False and "payload" not in r and "withheld" in r
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps(r), encoding="utf-8")
    out = tmp_path / "res.json"
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3 and not out.exists()
    cf2 = tmp_path / "c2.json"
    cf2.write_text(json.dumps(RE.assemble_controls(ok, p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf2), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf2)) and res["names"]["CARD"] == "VISSEED10K-HOLDS" and "F4_VD5C" in res["names"]
    for drop in RE.CONTROLS:                                                                 # with any one control missing nothing is opened
        cf3 = tmp_path / ("c3_%s.json" % drop)
        cf3.write_text(json.dumps(RE.assemble_controls({k: v for k, v in ok.items() if k != drop}, p)), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(cf3), "--out", str(tmp_path / ("res3_%s.json" % drop))]) == 3


def test_reader_source_carries_the_strategic_family_the_references_and_the_engine_bootstrap_setting():
    t = (HERE / "read_visseed10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and t.index("a = ap.parse_args(argv)") < t.index("a.n_boot_control = 300")
    assert "R.strategic_grid(" in t and "S.pair_read(" in t and '"-VS-VD5c"' in t and 'pairs_def[x + "-VS-VD5c"]' in t
    assert 'REF_RUNS = {"RR": BANKED["RR"], "RLc": BANKED["RLc"], "VD5c": B102["VD5c"]}' in t and "for ref, rn in REF_RUNS.items()" in t
    assert 'RUN = {"F%d" % s: "RQ50KF5P30Hb20fF%d" % s for s in FITS}' in t and 'B103 = {"VD5cx": "RQ50KF5P30Hb20wVD5cx"}' in t and "C_EXPECTED" in t


def test_podread_text():
    t = (HERE / "podread_visseed10k.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in t and "VISSEED10K_CONTROLS.json" in t and "VISSEED10K_RESULT.json" in t and "ALREADY-READ" in t and "QUEUE-STOPPED" in t
    for x in ("fF2", "fF3", "fF4", "vV0", "vVD5c", "wVD5cx", "D_H5", "--fit-root", "--add99-fits", "--eval-list", "--wrapper-hashes", "--wrapper-file", "VISSEED_GAPFILES_RECORD.json", "FIT_RECORD.json", "meta.json"):
        assert x in t, x
    for bad in (" rm ", " mv ", "kill", "pkill", "/workspace/v8proof/visrep10k", "visrule10k/read"):
        assert bad not in t, bad
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_visseed10k.sh")]).returncode == 0


def test_push_and_pull_scripts():
    pu, pl = (HERE / "push_visseed.sh").read_text(encoding="utf-8"), (HERE / "pull_visseed.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "lead_pytest.out" in pu and "/workspace/v8proof/visseed10k" in pu and "cat $P" not in pu and "ZZ PUSH-VISSEED" in pu
    for f in ("make_seed_gapfile.py", "pod_visseed_fit_queue.sh", "pod_visseed10k_queue.sh", "read_visseed10k.py", "podread_visseed10k.sh"):
        assert f in pu and (HERE / f).is_file()
    assert "pull_add104.tar" in pl and "dry_all.out" in pl and "dry_fits.out" in pl and "pod_add104" in pl and "VISSEED_GAPFILES_RECORD.json" in pl and "FIT_RECORD.json" in pl and "_meta.json" in pl
    assert "head.pt" not in pl and "score.npz" not in pl
    if bash_exe():
        for f in ("push_visseed.sh", "pull_visseed.sh", "pod_visseed_fit_queue.sh"):
            assert subprocess.run([bash_exe(), "-n", str(HERE / f)]).returncode == 0


def test_texts_hold_no_banned_words_and_paths_are_short():
    for f in ("gen_seed_gapfile.py", "INSERT_GAPFILE.txt", "make_seed_gapfile.py", "make_visseed_queue.py", "pod_visseed10k_queue.sh", "pod_visseed_fit_queue.sh", "read_visseed10k.py", "podread_visseed10k.sh",
              "push_visseed.sh", "pull_visseed.sh", "VISSEED_DEFS.md"):
        p = HERE / f
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            assert not [w for w in BANNED if w in t], f
            assert "\r" not in t and len(str(p)) < 200


def test_selftests_run():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for args, last in (([str(HERE / "read_visseed10k.py"), "--selftest"], "ZZ SELFTEST-READ-VISSEED10K ok: True"), ([str(HERE / "make_seed_gapfile.py"), "--selftest"], "ZZ MAKE-SEED-GAPFILE-SELFTEST PASS")):
        r = subprocess.run([sys.executable] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
        assert r.stdout.strip().splitlines()[-1] == last, (args, r.stdout[-400:])
