"""test_cropscale_fit.py -- tests of the fit, gap-file and rollout-queue kit of card ADD106 (CROPSCALE10K): controls g2 and g3 clause by clause on planted faults, the build routines against those of cards ADD104
and ADD105, the fit queue (links, stage PERM, DRY) and the rollout queue (ten commands). Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest cropscale/fit/test_cropscale_fit.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/cropscale/fit/tmp_pt/<new name>"""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
V8 = HERE.parent.parent
sys.path.insert(0, str(HERE))
import make_scale_gapfiles as MSG               # noqa: E402

QUEUE = HERE / "pod_cropscale10k_queue.sh"
FITQ = HERE / "pod_cropscale_fit_queue.sh"
Q104, Q105, Q102 = V8 / "visseed" / "pod_visseed10k_queue.sh", V8 / "visens" / "pod_visens10k_queue.sh", V8 / "visrule" / "pod_visrule10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
W16 = "28bc31f21ecb4ccb"


def bash_exe():
    return shutil.which("bash")


def posix(p):
    return Path(p).as_posix()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ the gap files, g2 and g3
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("cs")
    J, DL, DM, P = MSG.synthetic_world(str(base / "w"))
    F = MSG.collect(P)
    rec = MSG.build(J, DL, DM, F, str(base / "clean"), eval_join_sha="9" * 64)
    return {"J": J, "DL": DL, "DM": DM, "P": P, "F": F, "rec": rec, "base": base}


def test_the_generators_read_the_files_they_are_derived_from():
    assert hashlib.sha256(Q104.read_bytes()).hexdigest()[:16] == "ba9c4bf35430e42f" and hashlib.sha256(Q105.read_bytes()).hexdigest()[:16] == "ee9d4f9d232b04c2"
    assert hashlib.sha256(Q102.read_bytes()).hexdigest()[:16] == "6321d4e6e12c0f47"
    assert hashlib.sha256((V8 / "visrule" / "v8f_t1_eval_reader_rule_rzgfhqetjv.py").read_bytes()).hexdigest()[:16] == W16
    gq = (HERE / "make_cropscale_queue.py").read_text(encoding="utf-8")
    assert "ba9c4bf35430e42f" in gq and "ee9d4f9d232b04c2" in gq and W16 in gq
    t = (HERE / "make_scale_gapfiles.py").read_text(encoding="utf-8")
    assert 'CARD = "PREREG_V8F_PROOF_ADD106"' in t and "CROPSCALE_GAPFILES_RECORD.json" in t and 'DUMP_DIR = "D_H5"' in t


def test_clean_build_passes_g2_and_g3_and_has_the_five_files(world):
    r = world["rec"]
    assert r["g2"]["PASS"] and r["g3"]["PASS"], (r["g2"]["failures"], r["g3"]["failures"])
    assert all(r["g3"]["parts"].values()) and all(r["g2"]["clauses"].values())
    assert sorted(r["files"]) == sorted(["VGAPL_s2", "VGAPL_s3", "VGAPL_s4", "VGAPLE", "VGAPSE3"]) and r["card"] == "PREREG_V8F_PROOF_ADD106"
    for nm in r["files"]:
        assert (world["base"] / "clean" / (nm + ".npz")).is_file() and len(r["files"][nm]["file_sha256"]) == 64
    assert (world["base"] / "clean" / "CROPSCALE_GAPFILES_RECORD.json").is_file()
    ff = r["fit_facts_for_reader"]
    assert sorted(ff) == ["L2", "L3", "L4", "M2", "M3", "M4", "O0", "O1"] and ff["L3"]["kept_epoch"] == 33 and ff["L3"]["n_fit_clips"] == 470 and ff["M3"]["n_fit_clips"] == 450 and ff["L2"]["n_sel_clips"] == 50
    assert r["fits"]["lists"]["n_large"] == 520 and r["fits"]["lists"]["n_large_fit"] == 470 and "t0_diagnostic" in r


def test_the_single_build_is_the_one_of_card_add104_and_the_mean_build_is_the_one_of_card_add105(world):
    m4 = load_mod(V8 / "visseed" / "make_seed_gapfile.py", "msg104")
    m5 = load_mod(V8 / "visens" / "make_ens_gapfile.py", "meg105")
    J, DL, DM = world["J"], world["DL"], world["DM"]
    for s in (2, 3, 4):
        a = MSG.build_arrays(J, DL[s], MSG.frame_table(J, DL[s]), s)
        b = m4.build_arrays(J, DL[s], m4.frame_table(J, DL[s]), s)
        assert sorted(a) == sorted(b) and all(same(a[k], b[k]) for k in a)
    five = {s: MSG.synthetic_dump(J, s) for s in (0, 1, 2, 3, 4)}
    a = MSG.mean_arrays(J, [five[s] for s in (0, 1, 2, 3, 4)])
    b = m5.build_arrays(J, five)
    assert sorted(a) == sorted(b) and all(same(a[k], b[k]) for k in a if k != "seed") and int(a["seed"]) == int(b["seed"]) == -1


def test_the_mean_files_are_the_means_of_the_right_dumps_and_have_an_output_exactly_where_all_three_do(world):
    J, DL, DM, base = world["J"], world["DL"], world["DM"], world["base"] / "clean"
    for nm, dd in (("VGAPLE", DL), ("VGAPSE3", DM)):
        z = MSG.load_npz(str(base / (nm + ".npz")))
        G = np.stack([MSG.on_join_rows(J, dd[s], "gap") for s in (2, 3, 4)], 0)
        allfin = np.isfinite(G).all(axis=0)
        assert np.array_equal(z["has_out"], allfin) and np.allclose(z["gap"][allfin], G[:, allfin].mean(axis=0), atol=1e-12, rtol=0) and np.isnan(z["gap"][~allfin]).all()
        assert int(z["seed"]) == -1 and allfin.any() and (~allfin).any()
    a, b = MSG.load_npz(str(base / "VGAPLE.npz")), MSG.load_npz(str(base / "VGAPSE3.npz"))
    assert not np.array_equal(a["gap"], b["gap"], equal_nan=True)                     # the two sets of dumps differ


def test_g2_has_one_red_fault_per_clause_and_the_clean_facts_pass(world):
    res = MSG.g2_faults(world["F"])
    assert res and all(res.values()), [k for k, v in res.items() if not v]
    for clause in "abcdefghi":
        assert [k for k in res if k.startswith(clause + "_")], clause


def test_g2_names_every_clause_of_the_card_on_the_clean_facts(world):
    r = MSG.g2_check(world["F"])
    assert r["PASS"] and sorted(r["clauses"]) == list("abcdefghi") and all(r["clauses"].values())
    t = MSG.__doc__
    for need in ("seed S", "head H5", "permuted", "role fit", "50 selection clips", "no evaluation clip", "five on record"):
        assert need in t, need


def test_g2_fit_clips_clause_follows_the_role_fit_clips_of_the_large_list(world):
    F = copy.deepcopy(world["F"])
    L = F["lists"]["large"]
    F["lists"]["large"] = L + [["nw999", "fit"]]
    assert not MSG.g2_check(F)["PASS"]                                                # a clip added to the list, not to the fits
    for s in (2, 3, 4):
        F["large"]["s%d" % s]["fit_clips"] = F["large"]["s%d" % s]["fit_clips"] + ["nw999"]
    F["large"]["s2"]["fit_clips"] = F["large"]["s2"]["fit_clips"][:-1] + ["nw998"]
    r = MSG.g2_check(F)
    assert not r["PASS"] and r["clauses"]["d"] is False


def test_g3_has_one_red_fault_per_part_and_the_clean_build_passes(world):
    res = MSG.g3_faults(world["J"], world["DL"], world["DM"], world["F"], str(world["base"]))
    assert res and all(res.values()), [k for k, v in res.items() if not v]
    for part in ("build_", "mean_", "meta_", "mid_", "v0_"):
        assert [k for k in res if k.startswith(part)], part


def test_the_mean_bar_is_1e_minus_6_and_the_single_bar_1e_minus_9(world):
    r = MSG.build(world["J"], world["DL"], world["DM"], world["F"], str(world["base"] / "bar"), fault="m_offset")
    f = r["means"]["VGAPLE"]["facts"]["saved"]
    assert 4e-6 < f["max_abs_dev_saved_minus_second_routine_m"] < 6e-6 and not r["g3"]["PASS"] and r["g3"]["parts"]["mean_large"] is False and r["g3"]["parts"]["build"] is True
    assert MSG.TOL_MEAN_M == 1e-6 and MSG.TOL_SAVED == 1e-9


def test_main_through_files_refusals_and_arguments(world, tmp_path):
    J = world["J"]
    P = dict(world["P"])
    root = tmp_path / "w"
    shutil.copytree(str(world["base"] / "w"), str(root))
    P = {k: str(root / Path(v).relative_to(world["base"] / "w")) for k, v in P.items()}
    ej = tmp_path / "join.npz"
    np.savez(str(ej), **J)

    def args(out, **over):
        d = {"--eval-join": str(ej), "--out-dir": str(out), "--fit-root": P["fit_root"], "--large-list": P["large_list"], "--pilot-list": P["pilot_list"], "--eval-list": P["eval_list"],
             "--old-fits": P["old_fits"], "--mid-root": P["mid_root"], "--rec104": P["rec104"], "--expect-clips": "4"}
        d.update(over)
        return [x for kv in d.items() for x in kv]
    out = tmp_path / "out"
    assert MSG.main(args(out)) == 0
    rec = json.loads((out / "CROPSCALE_GAPFILES_RECORD.json").read_text(encoding="utf-8"))
    assert rec["g2"]["PASS"] and rec["g3"]["PASS"] and rec["eval_join_sha256"] == hashlib.sha256(ej.read_bytes()).hexdigest() and rec["files"]["VGAPLE"]["file_sha256"] == hashlib.sha256((out / "VGAPLE.npz").read_bytes()).hexdigest()
    assert rec["inputs"]["large_list"]["sha256"] == hashlib.sha256(Path(P["large_list"]).read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        MSG.main(args(out))                                                           # an existing out dir is refused
    assert MSG.main([]) == 2
    assert MSG.main(args(tmp_path / "o2", **{"--rec104": str(tmp_path / "none.json")})) == 2
    os.remove(P["fit_root"] + "/dump/D_H5/reader_s3_score.npz")
    assert MSG.main(args(tmp_path / "o3")) == 2                                       # a missing dump is refused


def test_a_record_that_fails_a_clause_exits_1_and_says_which(world, tmp_path):
    J, P = world["J"], world["P"]
    ej = tmp_path / "join.npz"
    np.savez(str(ej), **J)
    r = json.loads(Path(P["fit_root"] + "/fits/D_H5_s4/FIT_RECORD.json").read_text(encoding="utf-8"))
    r["permute_labels"] = True
    p = tmp_path / "fitroot2"
    shutil.copytree(P["fit_root"], str(p))
    Path(str(p) + "/fits/D_H5_s4/FIT_RECORD.json").write_text(json.dumps(r), encoding="utf-8")
    args = ["--eval-join", str(ej), "--out-dir", str(tmp_path / "out"), "--fit-root", str(p), "--large-list", P["large_list"], "--pilot-list", P["pilot_list"], "--eval-list", P["eval_list"],
            "--old-fits", P["old_fits"], "--mid-root", P["mid_root"], "--rec104", P["rec104"], "--expect-clips", "4"]
    assert MSG.main(args) == 1
    rec = json.loads((tmp_path / "out" / "CROPSCALE_GAPFILES_RECORD.json").read_text(encoding="utf-8"))
    assert rec["g2"]["PASS"] is False and rec["g2"]["clauses"]["c"] is False and rec["g3"]["PASS"] is True


# ------------------------------------------------------------------------------------------------ the fit queue
def run_bash(script, env_extra, cwd=None, args=()):
    env = dict(os.environ, **env_extra)
    r = subprocess.run([bash_exe(), str(script)] + list(args), env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=240)
    return r.returncode, r.stdout


def make_stub_world(tmp_path, enc=None, missing_clip=None):
    """a pod-like tree: the pilot's run folder (tokens of its clips and of the evaluation clips, lists), the new token folder, the large list, the encode controls, stub fit and dump tools, shim ln and readlink"""
    r1, pre, kit, gp = tmp_path / "run1", tmp_path / "present", tmp_path / "kit", tmp_path / "gp"
    (r1 / "tok" / "D").mkdir(parents=True)
    (r1 / "lists").mkdir()
    pil = ["pl0 fit", "pl1 fit", "pl2 sel"]
    for c in ("pl0", "pl1", "pl2", "ev0", "ev1"):
        (r1 / "tok" / "D" / (c + ".npy")).write_text("pilot token " + c, encoding="utf-8")
    (gp / "enc_out" / "tok" / "D").mkdir(parents=True)
    for c in ("pl1", "nw0", "nw1"):                                                  # pl1 is in both folders: the pilot's file is linked
        (gp / "enc_out" / "tok" / "D" / (c + ".npy")).write_text("new token " + c, encoding="utf-8")
    if missing_clip:
        os.remove(str(gp / "enc_out" / "tok" / "D" / (missing_clip + ".npy")))
    (r1 / "lists" / "train_final.txt").write_text("\n".join(pil) + "\n", encoding="utf-8")
    (r1 / "lists" / "eval141.txt").write_text("ev0\nev1\n", encoding="utf-8")
    (gp / "lists").mkdir()
    (gp / "lists" / "train_final.txt").write_text("\n".join(pil + ["nw0 fit", "nw1 fit"]) + "\n", encoding="utf-8")
    (gp / "ENC_CONTROLS.json").write_text(json.dumps(enc if enc is not None else {"g0": {"PASS": True}, "g1": {"PASS": True}}), encoding="utf-8")
    pre.mkdir()
    (pre / "reader_s0_score.npz").write_bytes(b"x")
    (tmp_path / "join.npz").write_bytes(b"x")
    kit.mkdir()
    (kit / "crop_fit.py").write_text(
        "import argparse, json, os, sys\nap = argparse.ArgumentParser()\nfor n in ('arm', 'head', 'seed', 'root', 'join-train', 'device'):\n    ap.add_argument('--' + n)\nap.add_argument('--permute-labels', action='store_true')\n"
        "a = ap.parse_args()\nif os.environ.get('STUB_FAIL_SEED') == a.seed and not a.permute_labels:\n    sys.exit(7)\nd = os.path.join(a.root, 'fits', 'D_H5_s' + a.seed + ('_perm' if a.permute_labels else ''))\n"
        "os.makedirs(d)\nopen(os.path.join(d, 'head.pt'), 'w').write('h' + a.seed + str(a.permute_labels))\njson.dump({'seed': int(a.seed), 'perm': a.permute_labels}, open(os.path.join(d, 'FIT_RECORD.json'), 'w'))\n", encoding="utf-8")
    (kit / "crop_dump.py").write_text(
        "import argparse, json, os, sys\nap = argparse.ArgumentParser()\nfor n in ('arm', 'head', 'seed', 'root', 'present-dump', 'device'):\n    ap.add_argument('--' + n)\nap.add_argument('--permute-labels', action='store_true')\n"
        "a = ap.parse_args()\nassert os.path.isfile(os.path.join(a.root, 'fits', 'D_H5_s' + a.seed + ('_perm' if a.permute_labels else ''), 'FIT_RECORD.json'))\n"
        "assert os.path.isfile(os.path.join(a.present_dump, 'reader_s%s_score.npz' % a.seed.strip()))\nd = os.path.join(a.root, 'dump', 'D_H5' + ('_perm' if a.permute_labels else ''))\nos.makedirs(d, exist_ok=True)\n"
        "assert not os.path.exists(os.path.join(d, 'reader_s%s_score.npz' % a.seed))\nopen(os.path.join(d, 'reader_s%s_score.npz' % a.seed), 'w').write('d')\n"
        "json.dump({'seed': int(a.seed)}, open(os.path.join(d, 'reader_s%s_meta.json' % a.seed), 'w'))\n", encoding="utf-8")
    shim = tmp_path / "shim"                                                         # no real symbolic links on this file system: a shim `ln -s` and `readlink` that keep the target in a text file
    shim.mkdir()
    (shim / "ln").write_text('#!/bin/bash\n[ "$1" = -s ] || exit 9\nprintf \'SHIMLINK:%s\' "$2" > "$3"\n', encoding="utf-8")
    (shim / "readlink").write_text('#!/bin/bash\nc=$(cat "$1"); echo "${c#SHIMLINK:}"\n', encoding="utf-8")
    for p in (shim / "ln", shim / "readlink"):
        os.chmod(str(p), 0o755)
    env = {"CS_RUN1": posix(r1), "CS_ROOT": posix(gp), "CS_KIT": posix(kit), "CS_PRESENT": posix(pre), "CS_JOIN_TRAIN": posix(tmp_path / "join.npz"), "CS_PY": posix(sys.executable),
           "PATH": posix(shim) + os.pathsep + os.environ["PATH"]}
    return gp, env


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_dry_run_prints_every_command_and_makes_nothing_in_both_stages(tmp_path):
    assert subprocess.run([bash_exe(), "-n", str(FITQ)]).returncode == 0
    for perm in ("0", "1"):
        gp = tmp_path / ("gp" + perm)
        rc, out = run_bash(FITQ, {"DRY": "1", "PERM": perm, "CS_ROOT": posix(gp)})
        assert rc == 0 and not gp.exists(), out[-500:]
        cmds = [x for x in out.splitlines() if x.strip().startswith("CMD: ")]
        fits = [x for x in cmds if "crop_fit.py" in x]
        dumps = [x for x in cmds if "crop_dump.py" in x]
        seeds = (2, 3, 4) if perm == "0" else (2,)
        assert len(fits) == len(seeds) and len(dumps) == len(seeds)
        for s, c in zip(seeds, fits):
            assert ("--arm D --head H5 --seed %d" % s) in c and "/workspace/v8proof/cropread/kit/fit/crop_fit.py" in c and "--join-train /workspace/v8proof/data/inputs/join_train.npz" in c and "--device cuda" in c
            assert "--root %s/fitroot" % posix(gp) in c and (("--permute-labels" in c) == (perm == "1"))
        for s, c in zip(seeds, dumps):
            assert ("--arm D --head H5 --seed %d" % s) in c and ("--present-dump %s/present_tpl" % posix(gp)) in c and (("--permute-labels" in c) == (perm == "1"))
        links = [x for x in cmds if "ln -s" in x]
        assert any("ln -s /workspace/v8proof/closeread/dump/both/reader_s0_score.npz %s/present_tpl/reader_s2_score.npz" % posix(gp) in x for x in links)
        assert any(x.endswith("%s/fitroot/lists/train_final.txt" % posix(gp)) and ("%s/lists/train_final.txt" % posix(gp)) in x for x in links)
        assert any("<clip>.npy" in x and "/workspace/v8proof/cropread/run1/tok/D" in x and ("%s/enc_out/tok/D" % posix(gp)) in x for x in links)
        for x in links:                                                                 # nothing is linked INTO the run root of card ADD99
            assert not x.split()[-1].startswith("/workspace/v8proof/cropread/run1")
        assert any("ENC_CONTROLS.json reads g0 PASS and g1 PASS" in x for x in cmds) and any("memory free > 28 GB" in x for x in cmds)


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_builds_a_real_token_folder_with_one_link_per_clip_pilot_first(tmp_path):
    gp, env = make_stub_world(tmp_path)
    rc, out = run_bash(FITQ, env)
    assert rc == 0 and "ZZ CROPSCALE-FITS DONE" in out and "ADMIT-OK" in out, out[-800:]
    tok = gp / "fitroot" / "tok" / "D"
    assert tok.is_dir() and not tok.is_symlink()
    r1 = posix(tmp_path / "run1" / "tok" / "D")
    nt = posix(gp / "enc_out" / "tok" / "D")
    want = {"pl0": r1, "pl1": r1, "pl2": r1, "ev0": r1, "ev1": r1, "nw0": nt, "nw1": nt}
    assert sorted(p.name for p in tok.iterdir()) == sorted(c + ".npy" for c in want)
    for c, d in want.items():
        assert (tok / (c + ".npy")).read_text(encoding="utf-8") == "SHIMLINK:%s/%s.npy" % (d, c)
    assert (gp / "fitroot" / "lists" / "train_final.txt").read_text(encoding="utf-8") == "SHIMLINK:" + posix(gp / "lists" / "train_final.txt")
    assert (gp / "fitroot" / "lists" / "eval141.txt").read_text(encoding="utf-8") == "SHIMLINK:" + posix(tmp_path / "run1" / "lists" / "eval141.txt")
    assert "token links 7" in (gp / "START_FITS.txt").read_text(encoding="utf-8")
    assert not (tmp_path / "run1" / "fits").exists() and not (tmp_path / "run1" / "dump").exists()
    assert sorted(p.name for p in (tmp_path / "run1" / "tok" / "D").iterdir()) == sorted(c + ".npy" for c in ("pl0", "pl1", "pl2", "ev0", "ev1"))       # the pilot's folder is untouched


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_stops_on_a_clip_with_no_token_and_on_a_link_that_points_elsewhere(tmp_path):
    gp, env = make_stub_world(tmp_path, missing_clip="nw1")
    rc, out = run_bash(FITQ, env)
    assert rc == 3 and "NO-TOKEN-FOR-CLIP nw1" in out and (gp / "STOP_FITS.txt").is_file() and not (gp / "DONE_FITS.txt").exists()
    assert not list((gp / "fits_q" / "jobs").glob("*"))                                  # no fit started
    gp2, env2 = make_stub_world(tmp_path / "b")
    (gp2 / "fitroot" / "tok" / "D").mkdir(parents=True)
    (gp2 / "fitroot" / "tok" / "D" / "nw0.npy").write_text("SHIMLINK:/somewhere/else.npy", encoding="utf-8")
    rc, out = run_bash(FITQ, env2)
    assert rc == 3 and "LINK-IS-NOT-AS-EXPECTED" in out and "nw0.npy" in out


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_admission_reads_both_forms_of_the_controls_and_refuses_a_failed_one(tmp_path):
    for k, enc in enumerate(({"g0": {"PASS": True}, "g1": {"PASS": True}}, {"controls": {"g0": {"PASS": True}, "g1": {"PASS": True}}, "extra": 1})):
        gp, env = make_stub_world(tmp_path / ("ok%d" % k), enc=enc)
        rc, out = run_bash(FITQ, env)
        assert rc == 0 and "ADMIT-OK g0 g1" in out, (k, out[-400:])
    bad = [{"g0": {"PASS": True}, "g1": {"PASS": False}}, {"g0": {"PASS": True}}, {"controls": {"g0": {"PASS": "true"}, "g1": {"PASS": True}}}, {"g0": True, "g1": True}, {}]
    for k, enc in enumerate(bad):
        gp, env = make_stub_world(tmp_path / ("bad%d" % k), enc=enc)
        rc, out = run_bash(FITQ, env)
        assert rc == 3 and "ENC-CONTROLS" in out and (gp / "STOP_FITS.txt").is_file() and not (gp / "fitroot").exists(), (k, out[-300:])
    gp, env = make_stub_world(tmp_path / "unreadable")
    (gp / "ENC_CONTROLS.json").write_text("{not json", encoding="utf-8")
    rc, out = run_bash(FITQ, env)
    assert rc == 3 and "ENC-CONTROLS" in out
    gp, env = make_stub_world(tmp_path / "nolist")
    (gp / "lists" / "train_final.txt").write_text("pl0 fit\npl9 fit\npl2 sel\nnw0 fit\n", encoding="utf-8")                # does not start with the pilot's list
    rc, out = run_bash(FITQ, env)
    assert rc == 3 and "LARGE-LIST-DOES-NOT-START-WITH-THE-PILOT-LIST" in out and not (gp / "fitroot").exists()


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_markers_resume_and_stop_then_perm_stage_does_not_touch_the_real_fit(tmp_path):
    gp, env = make_stub_world(tmp_path)
    env["STUB_FAIL_SEED"] = "4"
    rc, out = run_bash(FITQ, env)
    assert rc == 3 and (gp / "STOP_FITS.txt").is_file() and (gp / "fits_q" / "jobs" / "fit_s4.STOP").is_file() and not (gp / "DONE_FITS.txt").exists(), out[-600:]
    assert (gp / "fits_q" / "jobs" / "fit_s2.DONE").is_file() and (gp / "fits_q" / "jobs" / "fit_s3.DONE").is_file() and not list((gp / "fitroot" / "dump").glob("*/*"))
    rc, out = run_bash(FITQ, env)
    assert rc == 2 and "RESUME=1" in out
    env.pop("STUB_FAIL_SEED")
    rc, out = run_bash(FITQ, dict(env, RESUME="1"))
    assert rc == 0 and (gp / "DONE_FITS.txt").is_file() and "ZZ CROPSCALE-FITS DONE" in out, out[-600:]
    for s in (2, 3, 4):
        assert (gp / "fitroot" / "fits" / ("D_H5_s%d" % s) / "FIT_RECORD.json").is_file() and (gp / "fitroot" / "dump" / "D_H5" / ("reader_s%d_meta.json" % s)).is_file()
        assert (gp / "fits_q" / "jobs" / ("dump_s%d.DONE" % s)).is_file() and (gp / "present_tpl" / ("reader_s%d_score.npz" % s)).is_file()
    rc, out = run_bash(FITQ, env)
    assert rc == 0 and "ALREADY-DONE" in out
    real = {str(p.relative_to(gp)): p.read_bytes() for p in (gp / "fitroot").rglob("*") if p.is_file() and "tok" not in p.parts}
    stamp = {k: (gp / k).stat().st_mtime_ns for k in real}
    rc, out = run_bash(FITQ, dict(env, PERM="1"))                                        # the stage PERM, beside the finished real fits
    assert rc == 0 and "ZZ CROPSCALE-PERM DONE" in out and (gp / "DONE_PERM.txt").is_file() and (gp / "START_PERM.txt").is_file(), out[-600:]
    assert (gp / "fitroot" / "fits" / "D_H5_s2_perm" / "FIT_RECORD.json").is_file() and (gp / "fitroot" / "dump" / "D_H5_perm" / "reader_s2_score.npz").is_file() and (gp / "fitroot" / "dump" / "D_H5_perm" / "reader_s2_meta.json").is_file()
    assert (gp / "perm_q" / "jobs" / "fit_s2_perm.DONE").is_file() and (gp / "perm_q" / "jobs" / "dump_s2_perm.DONE").is_file()
    assert not (gp / "fitroot" / "fits" / "D_H5_s3_perm").exists() and not (gp / "fitroot" / "fits" / "D_H5_s4_perm").exists()
    for k, b in real.items():                                                           # not one file of the real fits or dumps changed
        assert (gp / k).read_bytes() == b and (gp / k).stat().st_mtime_ns == stamp[k], k
    assert json.loads((gp / "fitroot" / "fits" / "D_H5_s2_perm" / "FIT_RECORD.json").read_text(encoding="utf-8"))["perm"] is True
    assert json.loads((gp / "fitroot" / "fits" / "D_H5_s2" / "FIT_RECORD.json").read_text(encoding="utf-8"))["perm"] is False


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_the_perm_stage_runs_before_the_real_fit_without_collision_too(tmp_path):
    gp, env = make_stub_world(tmp_path)
    rc, out = run_bash(FITQ, dict(env, PERM="1"))
    assert rc == 0, out[-500:]
    rc, out = run_bash(FITQ, env)
    assert rc == 0 and "ZZ CROPSCALE-FITS DONE" in out, out[-500:]
    assert (gp / "fitroot" / "fits" / "D_H5_s2" / "FIT_RECORD.json").is_file() and (gp / "fitroot" / "fits" / "D_H5_s2_perm" / "FIT_RECORD.json").is_file()


# ------------------------------------------------------------------------------------------------ the rollout queue
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def dry_commands(q, seeds="ALL"):
    env = dict(os.environ, DRY="1", GR_SEED=seeds)
    out = subprocess.run([bash_exe(), str(q)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100).stdout
    return [x for x in out.splitlines() if x.startswith("CMD: ")], out


def test_queue_rollout_function_is_the_one_of_cards_add102_add104_add105_and_the_header_facts_hold():
    q = QUEUE.read_text(encoding="utf-8")
    assert fn_text(q) == fn_text(Q104.read_text(encoding="utf-8")) == fn_text(Q105.read_text(encoding="utf-8")) == fn_text(Q102.read_text(encoding="utf-8"))
    assert "\r" not in q and not [w for w in BANNED if w in q]
    for bad in ("VISSEED", "visseed", "VISREP", "VISENS", "VISRULE10K_", "queue_visrule", "$G2", "VGAP1", "VGAPF", "wVD5cx", "pKD5", "rollout I10RQ50KF5P30Hb20w", "rollout I10RQ50KF5P30Hb20v"):
        assert bad not in q, bad
    for need in ("GP=${GR_ROOT:-/workspace/v8proof/cropscale}", "CHK=${GR_CHECK:-/workspace/v8proof/visrule10k/check_v}", '"$GP/CROPSCALE10K_STOP$PS.txt"', "START_CROPSCALE10K$PS.txt", "CROPSCALE10K_DONE$PS.txt",
                 "queue_cropscale$PS.log", '[ "$got" = "%s" ]' % W16, 'grep -q "%s" "$CHK/CHECK_WRAPPER_V.hashes"' % W16, "ZZ CHECK-WRAPPER-V PASS",
                 '"VGAPL_s2,VGAPL_s3,VGAPL_s4,VGAPLE,VGAPSE3" "g2,g3")', "CROPSCALE_GAPFILES_RECORD.json", "SM=$GP/small_cs",
                 "I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20fF2 I10RQ50KF5P30Hb20fF3 I10RQ50KF5P30Hb20fF4 I10RQ50KF5P30Hb20eE5c",
                 'NAMES="I10RQ50KF5P30Hb20gG2 I10RQ50KF5P30Hb20gG3 I10RQ50KF5P30Hb20gG4 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20gE3c"',
                 "I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20; do"):
        assert need in q, need
    assert "GF=${GR_GAPFILES:-$GP/gapfiles}" in q


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_has_ten_commands_five_arms_per_model_seed_in_order_with_the_right_gap_files():
    assert subprocess.run([bash_exe(), "-n", str(QUEUE)]).returncode == 0
    cmds, out = dry_commands(QUEUE)
    assert len(cmds) == 10, out[-800:]
    arms = [("gG2", "VGAPL_s2"), ("gG3", "VGAPL_s3"), ("gG4", "VGAPL_s4"), ("gGEc", "VGAPLE"), ("gE3c", "VGAPSE3")]
    seen = {}
    for c in cmds:
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and "--rule-join" not in c and c.count("--vis-closing") == 1 and c.count("--vis-gap-file") == 1
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        for a, gf in arms:
            for k in (0, 1):
                if ("/I10RQ50KF5P30Hb20%s_s%d " % (a, k)) in c:
                    seen[(a, k)] = 1
                    assert ("--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/" % k) in c and ("QMAP_LR16_s%d_q50" % k) in c
                    assert "--vis-gap-file /workspace/v8proof/cropscale/gapfiles/%s.npz" % gf in c, (a, k)
    assert sorted(seen) == sorted((a, k) for a, _ in arms for k in (0, 1))
    for k in (0, 1):
        one, _ = dry_commands(QUEUE, str(k))
        assert len(one) == 5
        names = [[a for a, _ in arms if ("/I10RQ50KF5P30Hb20%s_s%d " % (a, k)) in c] for c in one]
        assert names == [["gG2"], ["gG3"], ["gG4"], ["gGEc"], ["gE3c"]]                 # the five arms in this order
        assert all(("TK16_s%d" % k) in c and ("TK16_s%d" % (1 - k)) not in c for c in one)


def test_the_arguments_of_the_rollouts_are_those_of_the_vd5c_call_of_the_add102_queue():
    base = [x for x in Q102.read_text(encoding="utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20vVD5c RR" in x][0]
    env0 = base.split(" VGAP=")[0].strip()
    mine = [x for x in QUEUE.read_text(encoding="utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20g" in x]
    assert len(mine) == 5
    for (a, gf), x in zip((("G2", "VGAPL_s2"), ("G3", "VGAPL_s3"), ("G4", "VGAPL_s4"), ("GEc", "VGAPLE"), ("E3c", "VGAPSE3")), mine):
        assert x.strip().startswith(env0) and (' VGAP="$GF/%s.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20g%s RR "$k" "$k"' % (gf, a)) in x
    assert 'VGAP="$GF/VGAP_s$k.npz" VCLOSE=1' in base


def test_the_queue_is_regenerated_byte_for_byte_by_its_generator(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VQ_OVERWRITE_OWN="1")
    before = QUEUE.read_bytes()
    r = subprocess.run([sys.executable, str(HERE / "make_cropscale_queue.py")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100)
    assert r.returncode == 0 and "rollouts 10" in r.stdout, r.stdout[-400:]
    assert QUEUE.read_bytes() == before


# ------------------------------------------------------------------------------------------------ push script, texts, selftest
def test_push_script_and_texts():
    pu = (HERE / "push_cropscale_fit.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "lead_pytest.out" in pu and "/workspace/v8proof/cropscale" in pu and "kit_fit" in pu and "cat $P" not in pu and "ZZ PUSH-CROPSCALE-FIT" in pu
    for f in ("make_scale_gapfiles.py", "pod_cropscale_fit_queue.sh", "pod_cropscale10k_queue.sh"):
        assert f in pu and (HERE / f).is_file()
    if bash_exe():
        for f in ("push_cropscale_fit.sh", "pod_cropscale_fit_queue.sh", "pod_cropscale10k_queue.sh"):
            assert subprocess.run([bash_exe(), "-n", str(HERE / f)]).returncode == 0, f
    for f in ("make_scale_gapfiles.py", "make_cropscale_queue.py", "pod_cropscale10k_queue.sh", "pod_cropscale_fit_queue.sh", "push_cropscale_fit.sh", "test_cropscale_fit.py", "CROPSCALE_FIT_DEFS.md"):
        p = HERE / f
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            assert not [w for w in BANNED if w in t], (f, [w for w in BANNED if w in t])
            assert "\r" not in t and len(str(p.relative_to(V8)).replace("\\", "/")) < 200


def test_the_fit_queue_never_deletes_moves_or_kills():
    t = FITQ.read_text(encoding="utf-8")
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    import re
    for bad in (r"rm", r"rmdir", r"mv", r"kill", r"pkill", r"truncate", r"shred"):
        assert not re.search(bad, code), bad


def test_selftest_runs():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(HERE / "make_scale_gapfiles.py"), "--selftest"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=600)
    assert r.stdout.strip().splitlines()[-1] == "ZZ MAKE-SCALE-GAPFILES-SELFTEST PASS", r.stdout[-400:]


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_fit_queue_admission_reads_the_record_as_the_encode_kit_writes_it(tmp_path):
    # the shape ctl_g0g1.py writes: controls.g0.pass, controls.g1.pass (lower case) and pass_g0, pass_g1, pass_all at the top
    def rec(g0=True, g1=True, top0=True, top1=True):
        return {"card": "ADD106", "controls": {"g0": {"g0_no_duplicates": {"pass": True}, "pass": g0, "n_fit": 3}, "g1": {"g1_pilot_unchanged": {"pass": True}, "pass": g1}},
                "pass_g0": top0, "pass_g1": top1, "pass_all": bool(g0 and g1 and top0 and top1), "seconds": 1.0}
    gp, env = make_stub_world(tmp_path / "ok", enc=rec())
    rc, out = run_bash(FITQ, env)
    assert rc == 0 and "ADMIT-OK g0 g1" in out, out[-400:]
    for k, enc in enumerate((rec(g1=False), rec(g0=False), rec(top0=False), rec(top1=False), rec(g1="true"),
                             {"controls": {"g0": {"pass": False, "PASS": True}, "g1": {"pass": True}}})):
        gp, env = make_stub_world(tmp_path / ("bad%d" % k), enc=enc)
        rc, out = run_bash(FITQ, env)
        assert rc == 3 and "ENC-CONTROLS" in out and (gp / "STOP_FITS.txt").is_file() and not (gp / "fitroot").exists(), (k, out[-300:])
