"""Tests of the fit kit of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0), stage C. CPU only; tiny synthetic data."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
PILOT_FIT = os.path.join(V8, "cropread", "fit")
QUEUE = os.path.join(HERE, "pod_lookgap_fit_queue.sh")


def _load_cd():
    spec = importlib.util.spec_from_file_location("cross_dump_under_test", os.path.join(HERE, "cross_dump.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CD = _load_cd()


# ------------------------------------------------------------------------------------------------ the queue under DRY=1
@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    bash = shutil.which("bash")
    assert bash, "no bash"
    d = tmp_path_factory.mktemp("dryq")
    with open(QUEUE, "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw
    with open(os.path.join(str(d), "q.sh"), "wb") as fh:
        fh.write(raw)
    before = sorted(os.listdir(str(d)))
    env = dict(os.environ, DRY="1")
    r = subprocess.run([bash, "q.sh"], cwd=str(d), env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    after = sorted(os.listdir(str(d)))
    return r, before, after


def test_dry_exit_and_nothing_created(dry):
    r, before, after = dry
    assert r.returncode == 0, r.stdout + r.stderr
    assert before == after == ["q.sh"]
    assert "DRY run complete" in r.stdout


def test_dry_prints_commands_exactly(dry):
    out = dry[0].stdout
    run, k, j = "/workspace/v8proof/lookgap/run", "/workspace/v8proof/cropread/kit/fit", "/workspace/v8proof/data/inputs/join_train.npz"
    py = "/workspace/venv/bin/python"
    for s in (0, 1, 2):
        assert "CMD: %s %s/crop_fit.py --arm B --head H5 --seed %d --root %s --join-train %s --device cuda\n" % (py, k, s, run, j) in out
        assert "CMD: %s %s/crop_dump.py --arm B --head H5 --seed %d --root %s --present-dump %s/present_tpl --device cuda\n" % (py, k, s, run, run) in out
    assert "CMD: %s %s/crop_fit.py --arm B --head H5 --seed 0 --permute-labels --root %s --join-train %s --device cuda\n" % (py, k, run, j) in out
    assert "CMD: %s %s/crop_dump.py --arm B --head H5 --seed 0 --permute-labels --root %s --present-dump %s/present_tpl --device cuda\n" % (py, k, run, run) in out
    kf, r1, pres, root = "/workspace/v8proof/lookgap/kit_fit", "/workspace/v8proof/cropread/run1", "/workspace/v8proof/closeread/dump/both", "/workspace/v8proof/lookgap"
    base = "CMD: %s %s/cross_dump.py --kit-fit %s" % (py, kf, k)
    assert (base + " --fit-dir %s/fits/B_H5_s0 --tok-dir %s/tok/B --present %s/reader_s0_score.npz --seed-index 0 --out %s/xdump/B_on_B_ident/reader_s0_score.npz"
            " --compare %s/dump/B_H5/reader_s0_score.npz --ident-out %s/xdump/XDUMP_IDENT.json\n" % (r1, r1, pres, root, r1, root)) in out
    assert base + " --fit-dir %s/fits/B_H5_s0 --tok-dir %s/tok/P --present %s/reader_s0_score.npz --seed-index 0 --out %s/xdump/B_on_P/reader_s0_score.npz\n" % (r1, root, pres, root) in out
    assert base + " --fit-dir %s/fits/B_H5_s1 --tok-dir %s/tok/P --present %s/reader_s1_score.npz --seed-index 1 --out %s/xdump/B_on_P/reader_s1_score.npz\n" % (r1, root, pres, root) in out
    assert out.count("crop_fit.py") == 4 and out.count("crop_dump.py") == 4 and out.count("cross_dump.py") == 3
    assert "ulimit -n 16384" in out
    assert "ln -s %s/tok/P %s/tok/B" % (root, run) in out
    assert "ln -s %s/reader_s0_score.npz %s/present_tpl/reader_s2_score.npz" % (pres, run) in out
    assert "ln -s %s/reader_s1_score.npz %s/present_tpl/reader_s1_score.npz" % (pres, run) in out


# ------------------------------------------------------------------------------------------------ compare logic
def _d(gap, valid=None):
    gap = np.asarray(gap, float)
    return {"gap": gap, "valid": np.ones(len(gap), bool) if valid is None else np.asarray(valid, bool)}


def test_compare_equal():
    a = _d([np.nan, 1.0, 2.0, 3.0])
    r = CD.compare_dumps(a, _d([np.nan, 1.0, 2.0, 3.0]))
    assert r["pass"] is True and r["max_abs_diff_m"] == 0.0 and r["nan_pattern_equal"] and r["valid_equal"] and r["n_rows"] == 4


def test_compare_one_value_off():
    r = CD.compare_dumps(_d([1.0, 2.0, 3.0]), _d([1.0, 2.0, 3.01]), tol=1e-3)
    assert r["pass"] is False and abs(r["max_abs_diff_m"] - 0.01) < 1e-9
    assert CD.compare_dumps(_d([1.0, 2.0, 3.0]), _d([1.0, 2.0, 3.0005]), tol=1e-3)["pass"] is True


def test_compare_nan_pattern_differs():
    r = CD.compare_dumps(_d([np.nan, 2.0, 3.0]), _d([1.0, 2.0, 3.0]))
    assert r["nan_pattern_equal"] is False and r["pass"] is False and r["max_abs_diff_m"] == 0.0


def test_compare_valid_differs():
    r = CD.compare_dumps(_d([1.0, 2.0], [True, True]), _d([1.0, 2.0], [True, False]))
    assert r["valid_equal"] is False and r["pass"] is False


def test_compare_row_count_differs():
    r = CD.compare_dumps(_d([1.0, 2.0]), _d([1.0, 2.0, 3.0]))
    assert r["pass"] is False and r["max_abs_diff_m"] is None


# ------------------------------------------------------------------------------------------------ selftest, refusals
def test_selftest_and_refusals(tmp_path, monkeypatch, capsys):
    import torch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert torch.cuda.is_available() is False
    monkeypatch.chdir(tmp_path)
    assert CD.selftest(PILOT_FIT, str(tmp_path)) is True
    assert "ZZ CROSS-DUMP SELFTEST OK" in capsys.readouterr().out
    # a world to refuse on
    w = CD.make_world(str(tmp_path / "w"), PILOT_FIT)
    out = str(tmp_path / "o" / "reader_s0_score.npz")
    meta, ident = CD.run_cross(PILOT_FIT, w["fit"], w["tok"], w["present"], 0, out, device="cpu", log=lambda s: None)
    assert ident is None and os.path.isfile(out) and os.path.isfile(str(tmp_path / "o" / "reader_s0_meta.json"))
    for k in ("fit_dir", "tok_dir", "head_pt_sha256", "present_dump", "present_dump_sha256", "dump_sha256", "n_rows", "n_valid", "crop_fit_sha256", "crop_dump_sha256", "seed_index"):
        assert k in meta
    assert meta["dump_sha256"] == hashlib.sha256(open(out, "rb").read()).hexdigest()
    # existing --out
    with pytest.raises(SystemExit) as e:
        CD.run_cross(PILOT_FIT, w["fit"], w["tok"], w["present"], 0, out, device="cpu", log=lambda s: None)
    assert e.value.code == 2
    rc = CD.main(["--kit-fit", PILOT_FIT, "--fit-dir", w["fit"], "--tok-dir", w["tok"], "--present", w["present"], "--seed-index", "0", "--out", out, "--device", "cpu"])
    assert rc == 2
    # existing --ident-out
    idp = tmp_path / "ident.json"
    idp.write_text("{}")
    with pytest.raises(SystemExit) as e:
        CD.run_cross(PILOT_FIT, w["fit"], w["tok"], w["present"], 0, str(tmp_path / "o2" / "reader_s0_score.npz"), compare=out, ident_out=str(idp), device="cpu", log=lambda s: None)
    assert e.value.code == 2
    # a head file that is not H5
    os.makedirs(str(tmp_path / "h1"))
    torch.save({"head": "H1", "arm": "B", "state_dict": {}}, str(tmp_path / "h1" / "head.pt"))
    with pytest.raises(SystemExit) as e:
        CD.run_cross(PILOT_FIT, str(tmp_path / "h1"), w["tok"], w["present"], 0, str(tmp_path / "o3" / "reader_s0_score.npz"), device="cpu", log=lambda s: None)
    assert e.value.code == 2
    assert not os.path.exists(str(tmp_path / "o3" / "reader_s0_score.npz"))
    # compare with the dump it just made: identity passes, exit code of main is 0
    rc = CD.main(["--kit-fit", PILOT_FIT, "--fit-dir", w["fit"], "--tok-dir", w["tok"], "--present", w["present"], "--seed-index", "0", "--out", str(tmp_path / "o4" / "reader_s0_score.npz"),
                  "--compare", out, "--ident-out", str(tmp_path / "ident4.json"), "--device", "cpu"])
    assert rc == 0
    assert json.load(open(str(tmp_path / "ident4.json"), encoding="utf-8"))["pass"] is True
    assert "ZZ CROSS-IDENT pass True max_abs_diff_m 0.0" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the pilot's tools are used unchanged
def test_pilot_tools_unchanged():
    for name, head in (("crop_fit.py", "61bf2f4e8e7ec61f"), ("crop_dump.py", "b892ef27f3c3ae51")):
        assert hashlib.sha256(open(os.path.join(PILOT_FIT, name), "rb").read()).hexdigest().startswith(head)
        assert not os.path.exists(os.path.join(HERE, name)), "the kit carries a copy of " + name


# ------------------------------------------------------------------------------------------------ wording and path length
def _banned():
    return ["vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"]


def test_wording_and_path_length():
    mine = [f for f in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, f)) and f.endswith((".py", ".sh", ".md"))]
    assert {"pod_lookgap_fit_queue.sh", "cross_dump.py", "test_lookgap_fit.py", "LOOKGAP_FIT_DEFS.md", "push_lookgap_fit.sh"} <= set(mine)
    for f in mine:
        text = open(os.path.join(HERE, f), encoding="utf-8").read().lower()
        for w in _banned():
            assert w not in text, "%s contains a banned word" % f
        assert not f.lower().startswith(("aux", "con.", "prn", "nul"))
    for root, dirs, files in os.walk(HERE):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), V8).replace("\\", "/")
            assert len(rel) <= 200, rel
