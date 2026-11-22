"""test_lookgap_dirs.py -- tests of the kit of stage A of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0): the fold rule, the fold run roots read by locator_fit.py unchanged, the direction files (format, neutral frames, no box folder,
refusals), the m0 check with every clause red on its planted fault, the score (S4, S8, S8_out) and the queue (DRY, markers, resume, budget stop) through stubs. Synthetic data only; no file above 5 MB.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest lookgap/dirs/test_lookgap_dirs.py -q -p no:cacheprovider --basetemp lookgap/dirs/tmp_pt/<new name>"""
import inspect
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
KIT_FIT = V8 / "cropread" / "fit"
LOC_DIR = V8 / "lookat"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LOC_DIR))
import loc_dirs as LD               # noqa: E402
import locator_fit as LF            # noqa: E402

BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
QUEUE, PUSH = HERE / "pod_lookgap_dirs_queue.sh", HERE / "push_lookgap_dirs.sh"
LF.set_cft(str(KIT_FIT))
CFT = LF.CFT
NFR = 7
TRAIN = ["t%02d" % i for i in range(10)]
EVAL = ["e0", "e1"]


def posix(p):
    return Path(p).as_posix()


def bash_exe():
    return shutil.which("bash")


def link_dir(target, link):
    """test helper: a link to the token folder; a symbolic link, else a junction, else a copy of the tiny files."""
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return
    except OSError:
        pass
    if subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
        return
    shutil.copytree(str(target), str(link))


# ------------------------------------------------------------------------------------------------ the fold rule
def test_fold_rule_is_sorted_clip_number_mod_five_with_roles_filtered():
    rows = [("c%02d" % i, "sel" if i in (3, 11) else "fit") for i in range(20)]
    rows = rows[::-1]                                                          # the file order does not matter: sorted by clip id
    f = LD.make_folds(rows)
    assert [x["heldout"] for x in f] == [["c%02d" % i for i in range(k, 20, 5)] for k in range(5)]
    assert f[3]["sel"] == ["c11"] and f[3]["heldout"] == ["c03", "c08", "c13", "c18"]    # c03 is in fold 3: not a selection clip of fold 3
    assert f[0]["sel"] == ["c03", "c11"] and f[1]["sel"] == ["c03"] and f[1]["fit"] == [c for c, r in sorted(rows) if r == "fit" and int(c[1:]) % 5 != 1]
    for x in f:
        assert not set(x["heldout"]) & (set(x["fit"]) | set(x["sel"])) and len(x["heldout"]) + len(x["fit"]) + len(x["sel"]) == 20
    assert sorted(sum((x["heldout"] for x in f), [])) == sorted(c for c, _ in rows)


# ------------------------------------------------------------------------------------------------ the tiny world
def make_world(tmp):
    rng = np.random.RandomState(1)
    root = tmp / "run1"
    (root / "tok" / "D").mkdir(parents=True)
    (root / "lists").mkdir()
    box = tmp / "box"
    box.mkdir()
    for c in TRAIN + EVAL:
        np.save(str(root / "tok" / "D" / (c + ".npy")), rng.standard_normal((NFR, 2, 65, 1024)).astype(np.float16))
        has = rng.rand(NFR) < 0.85
        has[4:] = True
        has[0] = True
        np.savez(str(box / (c + ".npz")), has_lead=has, ctr_az=rng.uniform(-20, 20, NFR), ctr_el=rng.uniform(-8, 4, NFR))
    (root / "lists" / "train_final.txt").write_text("".join("%s %s\n" % (c, "sel" if c in ("t00", "t01") else "fit") for c in TRAIN[::-1]), encoding="utf-8")
    (root / "lists" / "eval141.txt").write_text("".join(c + "\n" for c in EVAL), encoding="utf-8")
    return {"root": root, "box": box, "tmp": tmp}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lg")
    w = make_world(tmp)
    fo = tmp / "folds"
    assert LD.main(["folds", "--root", str(w["root"]), "--out-root", str(fo), "--expect-train", "10", "--no-link"]) == 0
    for k in range(5):
        link_dir(w["root"] / "tok", fo / ("f%d" % k) / "run" / "tok")
        assert LF.main(["--seed", "0", "--root", str(fo / ("f%d" % k) / "run"), "--box-dir", str(w["box"]), "--out-root", str(fo / ("f%d" % k) / "loc"), "--kit-fit", str(KIT_FIT),
                        "--device", "cpu", "--epochs", "1", "--batch", "8"]) == 0
    o107 = tmp / "O107"
    assert LF.main(["--seed", "0", "--root", str(w["root"]), "--box-dir", str(w["box"]), "--out-root", str(o107), "--kit-fit", str(KIT_FIT), "--device", "cpu", "--epochs", "1", "--batch", "8"]) == 0
    dd, rd = tmp / "dirs", tmp / "records"
    for k in range(5):
        pa = ["predict", "--root", str(w["root"]), "--head", str(fo / ("f%d" % k) / "loc" / "fits" / "LOC_s0" / "head.pt"), "--fit-record", str(fo / ("f%d" % k) / "loc" / "fits" / "LOC_s0" / "FIT_RECORD.json"),
              "--clips", str(fo / ("f%d" % k) / "heldout.txt"), "--out-dir", str(dd), "--record", str(rd / ("predict_f%d.json" % k)), "--kit-fit", str(KIT_FIT), "--locator-dir", str(LOC_DIR), "--device", "cpu"]
        assert LD.main(pa) == 0
    pe = ["predict", "--root", str(w["root"]), "--head", str(o107 / "fits" / "LOC_s0" / "head.pt"), "--fit-record", str(o107 / "fits" / "LOC_s0" / "FIT_RECORD.json"), "--clips", str(w["root"] / "lists" / "eval141.txt"),
          "--out-dir", str(dd), "--record", str(rd / "predict_eval.json"), "--kit-fit", str(KIT_FIT), "--locator-dir", str(LOC_DIR), "--device", "cpu"]
    assert LD.main(pe) == 0
    w.update({"fo": fo, "o107": o107, "dd": dd, "rd": rd, "pe": pe})
    return w


def run_check(w, dirs=None, rd=None, fo=None, o107_rec=None):
    return LD.run_check(str(w["root"]), str(dirs or w["dd"]), str(w["fo"] / "FOLDS.json"), str(fo or w["fo"]), str(rd or w["rd"]), str(o107_rec or (w["o107"] / "fits" / "LOC_s0" / "FIT_RECORD.json")), 12)


# ------------------------------------------------------------------------------------------------ the fold run roots feed locator_fit.py unchanged
def test_fold_run_roots_are_read_by_locator_fit_unchanged_and_the_records_name_the_fold_lists(world):
    fj = json.loads((world["fo"] / "FOLDS.json").read_text(encoding="utf-8"))
    assert [f["n_heldout"] for f in fj["folds"]] == [2] * 5 and fj["n_train"] == 10 and fj["train_list_sha256"] == CFT.sha256_file(str(world["root"] / "lists" / "train_final.txt"))
    for f in fj["folds"]:
        k = f["fold"]
        rr = world["fo"] / ("f%d" % k) / "run"
        rows = CFT.read_train_list(str(rr / "lists" / "train_final.txt"))            # the tool's own reader accepts the written list
        assert [c for c, r in rows if r == "fit"] == f["fit"] and [c for c, r in rows if r == "sel"] == f["sel"]
        assert CFT.read_ids(str(rr / "lists" / "eval141.txt")) == f["heldout"] and f["heldout_sha256"] == LD.sha256_ids(f["heldout"])
        assert not set(f["heldout"]) & {c for c, _ in rows}
        rec = json.loads((world["fo"] / ("f%d" % k) / "loc" / "fits" / "LOC_s0" / "FIT_RECORD.json").read_text(encoding="utf-8"))
        assert rec["clips_read"] == {"fit": f["fit"], "sel": f["sel"]} and rec["seed"] == 0 and rec["permute_targets"] is False and rec["card"] == "PREREG_V8F_PROOF_ADD107"
    assert [x for x in fj["folds"][0]["fit"]] == ["t02", "t03", "t04", "t06", "t07", "t08", "t09"] and fj["folds"][0]["sel"] == ["t01"] and fj["folds"][1]["sel"] == ["t00"]
    with pytest.raises(SystemExit):
        LD.main(["folds", "--root", str(world["root"]), "--out-root", str(world["fo"]), "--no-link"])        # lists exist: refused
    with pytest.raises(SystemExit):
        LD.main(["folds", "--root", str(world["root"]), "--out-root", str(world["tmp"] / "f2"), "--expect-train", "500", "--no-link"])


def test_folds_mode_makes_the_link_to_the_token_folder_and_copies_no_token_file(world):
    out = world["tmp"] / "folds_link"
    try:
        LD.main(["folds", "--root", str(world["root"]), "--out-root", str(out), "--expect-train", "10"])
    except SystemExit as e:
        pytest.skip("no symbolic links on this machine: %s" % e)
    assert (out / "f0" / "run" / "tok").is_symlink() and os.path.samefile(str(out / "f0" / "run" / "tok" / "D"), str(world["root"] / "tok" / "D"))
    assert sorted(os.listdir(str(out / "f0" / "run"))) == ["lists", "tok"]                   # lists and the link: no token file is a file of the fold root


# ------------------------------------------------------------------------------------------------ the direction files
def test_direction_files_have_the_box_format_a_direction_on_every_frame_and_neutral_frames_without_history(world):
    store = CFT.TokenStore(str(world["root"]), "D")
    model = None
    import torch
    ck = torch.load(str(world["o107"] / "fits" / "LOC_s0" / "head.pt"), map_location="cpu")
    model = LF.make_loc_model()
    model.load_state_dict(ck["state_dict"])
    for c in EVAL:
        z = np.load(str(world["dd"] / (c + ".npz")))
        assert sorted(z.files) == sorted(LD.KEYS) == ["ctr_az", "ctr_el", "has_lead"]
        assert z["has_lead"].dtype == np.bool_ and z["has_lead"].all() and len(z["has_lead"]) == NFR == store.get(c).shape[0]
        assert (z["ctr_az"][:4] == 0.0).all() and (z["ctr_el"][:4] == -2.0).all()
        az, el = LF.predict_clip(model, store.get(c), torch.device("cpu"))
        assert np.allclose(z["ctr_az"][4:], az[4:]) and np.allclose(z["ctr_el"][4:], el[4:]) and np.isfinite(z["ctr_az"]).all()
        b = np.load(str(world["box"] / (c + ".npz")))
        assert z["ctr_az"].dtype == b["ctr_az"].dtype and z["ctr_el"].dtype == b["ctr_el"].dtype                   # the dtypes of the pilot's box files
        assert not np.allclose(z["ctr_az"][4:], b["ctr_az"][4:])                                                   # the file does not carry the label centre


def test_predict_opens_no_label_and_no_box_folder_and_writes_the_run_record(world):
    for fn in (LD.cmd_predict, LD.write_direction_files):
        s = inspect.getsource(fn)
        assert "box_dir" not in s and "load_box" not in s and "box_file" not in s
    rec = json.loads((world["rd"] / "predict_f2.json").read_text(encoding="utf-8"))
    fj = json.loads((world["fo"] / "FOLDS.json").read_text(encoding="utf-8"))
    assert rec["clips"] == fj["folds"][2]["heldout"] and rec["n_clips"] == 2 and rec["n_frames"] == 2 * NFR and rec["n_frames_with_history"] == 2 * (NFR - 4) and rec["n_frames_without_history"] == 2 * 4
    assert rec["head_pt_sha256"] == rec["fit_record_head_pt_sha256"] == CFT.sha256_file(str(world["fo"] / "f2" / "loc" / "fits" / "LOC_s0" / "head.pt")) and rec["box_folder_opened"] is False
    assert rec["keys_written"] == ["has_lead", "ctr_az", "ctr_el"] and all(len(v["sha256"]) == 64 for v in rec["per_clip"].values())


def test_predict_refuses_an_existing_file_a_wrong_head_hash_a_permuted_fit_and_a_wrong_seed(world, tmp_path):
    with pytest.raises(SystemExit):
        LD.main(world["pe"])                                                                         # the files and the record exist
    a = list(world["pe"])
    fr = json.loads((world["o107"] / "fits" / "LOC_s0" / "FIT_RECORD.json").read_text(encoding="utf-8"))

    def variant(**kw):
        r = dict(fr, **kw)
        p = tmp_path / ("fr_%d.json" % len(list(tmp_path.glob("fr_*.json"))))
        p.write_text(json.dumps(r), encoding="utf-8")
        b = list(a)
        b[b.index("--fit-record") + 1] = str(p)
        b[b.index("--out-dir") + 1] = str(tmp_path / "o")
        b[b.index("--record") + 1] = str(tmp_path / ("rec_%s.json" % p.stem))
        return b
    for kw in ({"head_pt_sha256": "0" * 64}, {"permute_targets": True}, {"seed": 1}, {"card": "other"}):
        with pytest.raises(SystemExit):
            LD.main(variant(**kw))
    assert not (tmp_path / "o").exists() or not list((tmp_path / "o").glob("*.npz"))
    with pytest.raises(SystemExit):
        LD.write_direction_files(["x"], lambda c: 7, lambda c: (np.full(7, np.nan),) * 2, str(tmp_path / "never"))        # an output on no frame is refused
    assert not list((tmp_path / "never").glob("*.npz"))


def test_a_clip_shorter_than_five_frames_gets_the_neutral_direction_everywhere(tmp_path):
    st = LD.write_direction_files(["s"], lambda c: 3, lambda c: (np.full(3, np.nan), np.full(3, np.nan)), str(tmp_path))
    z = np.load(str(tmp_path / "s.npz"))
    assert st["s"]["n_with_history"] == 0 and (z["ctr_az"] == 0).all() and (z["ctr_el"] == -2).all() and z["has_lead"].all() and len(z["has_lead"]) == 3


# ------------------------------------------------------------------------------------------------ the check (m0)
def test_check_passes_on_the_finished_folder(world):
    res = run_check(world)
    assert res["all_pass"], {k: v["failures"] for k, v in res["clauses"].items() if not v["pass"]}
    assert len(res["clauses"]) == 9 and res["n_train"] == 10 and res["n_eval"] == 2


def copy_world(world, tmp):
    dd, rd, fo = tmp / "dirs", tmp / "records", tmp / "folds"
    shutil.copytree(str(world["dd"]), str(dd))
    shutil.copytree(str(world["rd"]), str(rd))
    for k in range(5):
        d = fo / ("f%d" % k) / "loc" / "fits" / "LOC_s0"
        d.mkdir(parents=True)
        shutil.copy(str(world["fo"] / ("f%d" % k) / "loc" / "fits" / "LOC_s0" / "FIT_RECORD.json"), str(d / "FIT_RECORD.json"))
    return dd, rd, fo


def red(res, name):
    return not res["clauses"][name]["pass"] and not res["all_pass"]


def test_every_clause_of_the_check_goes_red_on_its_planted_fault(world, tmp_path):
    dd, rd, fo = copy_world(world, tmp_path)
    assert run_check(world, dd, rd, fo)["all_pass"]
    t = TRAIN[3]
    keep = (dd / (t + ".npz")).read_bytes()
    (dd / (t + ".npz")).unlink()                                                                       # a missing file
    assert red(run_check(world, dd, rd, fo), "c1_a_file_for_each_clip")
    (dd / (t + ".npz")).write_bytes(keep)
    (dd / "extra.npz").write_bytes(keep)                                                              # a file of no clip
    assert red(run_check(world, dd, rd, fo), "c1_a_file_for_each_clip")
    (dd / "extra.npz").unlink()
    z = dict(np.load(str(dd / (t + ".npz"))))
    np.savez(str(dd / (t + ".npz")), has_lead=z["has_lead"][:-1], ctr_az=z["ctr_az"][:-1], ctr_el=z["ctr_el"][:-1])         # a frame count other than the token file's
    assert red(run_check(world, dd, rd, fo), "c2_frame_count_equals_the_token_file")
    np.savez(str(dd / (t + ".npz")), has_lead=np.r_[z["has_lead"][:-1], False], ctr_az=z["ctr_az"], ctr_el=z["ctr_el"])
    assert red(run_check(world, dd, rd, fo), "c3_a_direction_on_every_frame")
    np.savez(str(dd / (t + ".npz")), has_lead=z["has_lead"], ctr_az=z["ctr_az"], ctr_el=z["ctr_el"], gap_re=np.zeros(NFR))   # a field from a label
    assert red(run_check(world, dd, rd, fo), "c4_no_field_from_a_label")
    np.savez(str(dd / (t + ".npz")), has_lead=z["has_lead"], ctr_az=z["ctr_az"] + 1.0, ctr_el=z["ctr_el"])              # frames 0 to 3 not neutral
    assert red(run_check(world, dd, rd, fo), "c5_no_history_frames_are_neutral")
    np.savez(str(dd / (t + ".npz")), **z)
    assert run_check(world, dd, rd, fo)["all_pass"]
    k = int(t[1:]) % 5                                                                                   # the fold of clip t; the record of its fit lists it as a fit clip
    fp = fo / ("f%d" % k) / "loc" / "fits" / "LOC_s0" / "FIT_RECORD.json"
    rec = json.loads(fp.read_text(encoding="utf-8"))
    good = json.dumps(rec)
    rec["clips_read"]["fit"] = rec["clips_read"]["fit"] + [t]
    fp.write_text(json.dumps(rec), encoding="utf-8")
    assert red(run_check(world, dd, rd, fo), "c6_fold_fit_records_do_not_list_their_clips")
    fp.write_text(good, encoding="utf-8")
    pf = rd / ("predict_f%d.json" % k)
    prec = json.loads(pf.read_text(encoding="utf-8"))
    pgood = json.dumps(prec)
    prec["head_pt_sha256"] = "0" * 64                                                                 # directions from a head other than the fold's
    pf.write_text(json.dumps(prec), encoding="utf-8")
    assert red(run_check(world, dd, rd, fo), "c7_train_directions_come_from_their_own_fold_head")
    pf.write_text(pgood, encoding="utf-8")
    pe = rd / "predict_eval.json"
    egood = pe.read_text(encoding="utf-8")
    e = json.loads(egood)
    e["head_pt_sha256"] = e["fit_record_head_pt_sha256"] = "1" * 64                                   # evaluation directions from another head
    pe.write_text(json.dumps(e), encoding="utf-8")
    assert red(run_check(world, dd, rd, fo), "c8_evaluation_clips_come_from_the_named_head")
    pe.write_text(egood, encoding="utf-8")
    fj = json.loads((world["fo"] / "FOLDS.json").read_text(encoding="utf-8"))
    bad = tmp_path / "FOLDS_bad.json"
    fj["folds"][0]["fit"] = fj["folds"][0]["fit"][1:]
    bad.write_text(json.dumps(fj), encoding="utf-8")
    res = LD.run_check(str(world["root"]), str(dd), str(bad), str(fo), str(rd), str(world["o107"] / "fits" / "LOC_s0" / "FIT_RECORD.json"), 12)
    assert red(res, "c0_folds_follow_the_rule")


def test_check_command_writes_the_json_refuses_to_overwrite_and_returns_the_verdict(world, tmp_path):
    out = tmp_path / "CHECK.json"
    args = ["check", "--root", str(world["root"]), "--dirs", str(world["dd"]), "--folds-json", str(world["fo"] / "FOLDS.json"), "--fold-root", str(world["fo"]), "--records-dir", str(world["rd"]),
            "--add107-fit-record", str(world["o107"] / "fits" / "LOC_s0" / "FIT_RECORD.json"), "--out", str(out), "--expect-total", "12"]
    assert LD.main(args) == 0 and json.loads(out.read_text(encoding="utf-8"))["all_pass"] is True
    with pytest.raises(SystemExit):
        LD.main(args)
    out2 = tmp_path / "CHECK2.json"
    args[-3] = str(out2)
    args[-1] = "13"
    assert LD.main(args) == 1 and json.loads(out2.read_text(encoding="utf-8"))["all_pass"] is False


# ------------------------------------------------------------------------------------------------ the score
def test_score_matches_a_hand_computation_excludes_frames_without_history_and_refuses_without_a_passing_check(tmp_path, capsys):
    box, dd = tmp_path / "box", tmp_path / "dd"
    box.mkdir()
    dd.mkdir()
    n = 8
    has = np.array([1, 1, 1, 1, 1, 1, 1, 0], bool)
    laz = np.array([0, 0, 0, 0, 2.0, 9.0, 20.0, 5.0])
    lel = np.full(n, -2.0)
    np.savez(str(box / "a.npz"), has_lead=has, ctr_az=laz, ctr_el=lel)
    paz = laz + np.array([50, 50, 50, 50, 3.0, 9.0, 2.0, 0.0])                    # frames 0 to 3 carry the neutral direction and are not scored; frame 7 has no label lead
    np.savez(str(dd / "a.npz"), has_lead=np.ones(n, bool), ctr_az=paz, ctr_el=lel + np.array([0, 0, 0, 0, 0, 0, 5.0, 0]))
    r = LD.score_clips(str(dd), str(box), ["a"])
    assert r["n_rows"] == 3 and r["n_label_lead_frames"] == 7 and r["n_without_history_not_scored"] == 4 and r["n_out_rows"] == 2
    assert r["S4"] == pytest.approx(1 / 3) and r["S8"] == pytest.approx(2 / 3) and r["S8_out"] == pytest.approx(1 / 2)       # d = (3,0), (9,0), (2,5)
    cj = tmp_path / "c.json"
    cj.write_text(json.dumps({"all_pass": False}), encoding="utf-8")
    lst = tmp_path / "l.txt"
    lst.write_text("a fit\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        LD.main(["score", "--dirs", str(dd), "--box-dir", str(box), "--clips", str(lst), "--check-json", str(cj)])
    cj.write_text(json.dumps({"all_pass": True}), encoding="utf-8")
    assert LD.main(["score", "--dirs", str(dd), "--box-dir", str(box), "--clips", str(lst), "--check-json", str(cj), "--label", "x"]) == 0
    out = capsys.readouterr().out
    assert "SCORE x: S4 0.3333 S8 0.6667 S8_out 0.5000" in out


def test_selftest_runs(capsys):
    assert LD.main(["--selftest"]) == 0
    assert "selftest OK" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the queue
STUB = r'''
import json, os, sys, time
name = os.path.basename(sys.argv[0])
args = sys.argv[1:]
def opt(k):
    return args[args.index(k) + 1] if k in args else None
tag = name + ":" + (args[0] if name == "loc_dirs.py" else "fit")
if name == "loc_dirs.py" and args[0] == "predict":
    tag += ":" + os.path.basename(opt("--record"))
if name == "locator_fit.py":
    tag += ":" + os.path.basename(os.path.dirname(opt("--out-root")))
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write("S %s %.3f\n" % (tag, time.time()))
time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
rc = 1 if os.environ.get("STUB_FAIL", "") and os.environ["STUB_FAIL"] in tag else 0
if rc == 0:
    if name == "locator_fit.py":
        d = os.path.join(opt("--out-root"), "fits", "LOC_s0")
        os.makedirs(d, exist_ok=True)
        json.dump({}, open(os.path.join(d, "FIT_RECORD.json"), "w"))
    elif args[0] == "folds":
        os.makedirs(opt("--out-root"), exist_ok=True)
        json.dump({}, open(os.path.join(opt("--out-root"), "FOLDS.json"), "w"))
    elif args[0] == "check":
        json.dump({"all_pass": True}, open(opt("--out"), "w"))
    elif args[0] == "score":
        print("SCORE %s: stub" % opt("--label"))
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write("E %s %.3f\n" % (tag, time.time()))
sys.exit(rc)
'''


def queue_world(tmp, gpu=0, mem_free_gb=100):
    r1, o, kit, kl, kf, l107 = tmp / "run1", tmp / "O", tmp / "kit", tmp / "kitloc", tmp / "kitfit", tmp / "O107"
    (r1 / "lists").mkdir(parents=True)
    (r1 / "tok" / "D").mkdir(parents=True)
    (l107 / "fits" / "LOC_s0").mkdir(parents=True)
    for d in (kit, kl, kf, tmp / "box"):
        d.mkdir()
    for f in (r1 / "lists" / "train_final.txt", r1 / "lists" / "eval141.txt", l107 / "fits" / "LOC_s0" / "head.pt", l107 / "fits" / "LOC_s0" / "FIT_RECORD.json", kf / "crop_fit.py"):
        f.write_text("x", encoding="utf-8")
    (kit / "loc_dirs.py").write_text(STUB, encoding="utf-8")
    (kl / "locator_fit.py").write_text(STUB, encoding="utf-8")
    cg, shim = tmp / "cg", tmp / "shim"
    cg.mkdir()
    shim.mkdir()
    (cg / "memory.max").write_text(str((mem_free_gb + 1) * 1073741824) + "\n", encoding="utf-8")
    (cg / "memory.stat").write_text("anon 1073741824\nfile 5\n", encoding="utf-8")
    (shim / "nvidia-smi").write_bytes(("#!/bin/bash\n" + "".join("echo %d\n" % (100 + i) for i in range(gpu))).encode())
    os.chmod(str(shim / "nvidia-smi"), 0o755)
    env = {"LG_RUN1": posix(r1), "LG_ROOT": posix(o), "LG_KIT": posix(kit), "LG_KIT_LOC": posix(kl), "LG_KIT_FIT": posix(kf), "LG_ADD107": posix(l107), "LG_BOX": posix(tmp / "box"), "LG_PY": posix(sys.executable),
           "LG_CG": posix(cg), "LG_ULIMIT_SOFT": "1", "LG_POLL_S": "0", "LG_WAIT_S": "0", "LG_SPACING_S": "0", "STUB_LOG": posix(tmp / "stub.log"), "PATH": posix(shim) + os.pathsep + os.environ["PATH"]}
    return o, env


def run_q(env, extra=None, script=QUEUE):
    r = subprocess.run([bash_exe(), str(script)], env=dict(os.environ, **env, **(extra or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=240)
    return r.returncode, r.stdout


def stub_events(tmp):
    return [x.split()[:2] for x in (tmp / "stub.log").read_text(encoding="utf-8").splitlines()]


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_prints_every_command_in_order_and_makes_nothing(tmp_path):
    for f in (QUEUE, PUSH):
        assert subprocess.run([bash_exe(), "-n", str(f)]).returncode == 0, f
    gp = tmp_path / "lg"
    rc, out = run_q({"LG_ROOT": posix(gp)}, {"DRY": "1"})
    assert rc == 0 and not gp.exists(), out[-500:]
    cmds = [x.strip()[5:] for x in out.splitlines() if x.strip().startswith("CMD: ") and not x.strip().startswith("CMD: need") and (" loc_dirs.py" in x or "/loc_dirs.py " in x or "/locator_fit.py " in x)]
    assert [c.split()[1].split("/")[-1] for c in cmds] == ["loc_dirs.py"] + ["locator_fit.py"] * 5 + ["loc_dirs.py"] * 9
    fits = [c for c in cmds if "locator_fit.py" in c]
    for k, c in enumerate(fits):
        assert "--seed 0 " in c and "--root %s/f%d/run" % (posix(gp / "folds"), k) in c and "--out-root %s/f%d/loc" % (posix(gp / "folds"), k) in c and "--device cuda" in c and "--permute-targets" not in c
        assert "--box-dir /workspace/v8proof/cropread/enc_out/box" in c and "--kit-fit /workspace/v8proof/cropread/kit/fit" in c
    ev = [c for c in cmds if "predict_eval.json" in c][0]
    assert "--head /workspace/v8proof/lookat/fits/LOC_s0/head.pt" in ev and "--box" not in ev and "eval141.txt" in ev
    assert all("--box" not in c for c in cmds if " predict " in c)
    assert "--expect-train 500" in out and "--expect-total 641" in out
    for m in ("START_DIRS.txt", "DONE_DIRS.txt", "STOP_DIRS.txt", "ulimit -n 16384", "memory free > 28 GB", "fewer than 3 GPU compute processes", "at most 2 at a time", "stop at 5400 s"):
        assert m in out, m


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_runs_the_stages_in_order_writes_markers_and_a_finished_queue_is_not_run_again(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"LG_MAX_JOBS": "2"})
    assert rc == 0 and "ZZ LOOKGAP-DIRS DONE" in out and "SCORE train_heldout: stub" in out and "SCORE evaluation: stub" in out, out[-800:]
    assert (o / "DONE_DIRS.txt").is_file() and (o / "START_DIRS.txt").is_file() and not (o / "STOP_DIRS.txt").exists()
    starts = [e[1] for e in stub_events(tmp_path) if e[0] == "S"]
    assert starts[0] == "loc_dirs.py:folds" and sorted(starts[1:6]) == ["locator_fit.py:fit:f%d" % k for k in range(5)]
    assert starts[6:] == ["loc_dirs.py:predict:predict_f%d.json" % k for k in range(5)] + ["loc_dirs.py:predict:predict_eval.json", "loc_dirs.py:check", "loc_dirs.py:score", "loc_dirs.py:score"]
    for j in ["folds"] + ["fit_f%d" % k for k in range(5)] + ["pred_f%d" % k for k in range(5)] + ["pred_eval", "check", "score_train", "score_eval"]:
        assert (o / "queue" / "jobs" / (j + ".DONE")).is_file() and (o / "queue" / "jobs" / (j + ".SECS")).is_file(), j
    rc2, out2 = run_q(env)
    assert rc2 == 0 and "ALREADY-DONE" in out2


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_failed_job_writes_the_stop_marker_nothing_starts_after_it_and_resume_continues(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"STUB_FAIL": "fit:f1", "LG_MAX_JOBS": "1"})
    assert rc == 3 and (o / "STOP_DIRS.txt").is_file() and not (o / "DONE_DIRS.txt").exists() and (o / "queue" / "jobs" / "fit_f1.STOP").is_file(), out[-600:]
    assert (o / "queue" / "jobs" / "fit_f0.DONE").is_file() and not (o / "folds" / "f2" / "loc").exists()
    rc, out = run_q(env)
    assert rc == 2 and "REFUSED" in out
    rc, out = run_q(env, {"RESUME": "1"})
    assert rc == 0 and (o / "DONE_DIRS.txt").is_file() and "ZZ LOOKGAP-DIRS DONE" in out, out[-600:]
    starts = [e[1] for e in stub_events(tmp_path) if e[0] == "S"]
    assert starts.count("locator_fit.py:fit:f0") == 1 and starts.count("locator_fit.py:fit:f1") == 2 and starts.count("loc_dirs.py:folds") == 1


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_stops_at_the_time_budget_and_waits_for_no_slot_then_stops(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"LG_STOP_S": "0"})
    assert rc == 3 and "BUDGET-BEFORE-folds" in out and (o / "STOP_DIRS.txt").is_file() and not (tmp_path / "stub.log").exists()
    (o / "STOP_DIRS.txt").unlink()
    (o / "queue" / "jobs").mkdir(parents=True, exist_ok=True)
    (o / "queue" / "jobs" / "folds.DONE").write_text("x", encoding="utf-8")
    (o / "queue" / "jobs" / "folds.SECS").write_text("5400\n", encoding="utf-8")
    (o / "folds").mkdir(exist_ok=True)
    rc, out = run_q(env)
    assert rc == 3 and "BUDGET-BEFORE-fit_f0" in (o / "STOP_DIRS.txt").read_text(encoding="utf-8") and not (tmp_path / "stub.log").exists()
    o2, env2 = queue_world(tmp_path / "b", gpu=3)
    rc, out = run_q(env2)
    assert rc == 3 and "no slot for fit_f0" in (o2 / "STOP_DIRS.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_refuses_a_missing_input(tmp_path):
    o, env = queue_world(tmp_path)
    (tmp_path / "O107" / "fits" / "LOC_s0" / "head.pt").unlink()
    rc, out = run_q(env)
    assert rc == 3 and "MISSING-INPUT" in out


# ------------------------------------------------------------------------------------------------ files
def test_no_banned_word_and_short_paths_in_the_files_of_the_kit():
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh", ".md"):
            t = f.read_text(encoding="utf-8").lower()
            for w in BANNED:
                assert w not in t, (f.name, w)
        assert len(str(f)) <= 200 and f.name not in ("aux", "con", "prn", "nul")
        if f.is_file():
            assert f.stat().st_size < 5 * 1024 * 1024
