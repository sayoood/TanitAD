"""test_preshead_fit.py -- tests of the fit kit of card PREREG_V8F_PROOF_ADD118 (PRESHEAD-T0): the generator, the generated fit tool, the dump tool, the pod queue and its wait script.
Tiny planted data only; CPU. Run from the kit folder with the source crop_fit.py at ../../cropread/fit/crop_fit.py (the layout of the dev box and of the pod stage copy)."""
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "cropread", "fit", "crop_fit.py"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import make_pres_fit as MK  # noqa: E402
import pres_fit as PF  # noqa: E402
import pres_dump as PD  # noqa: E402

BASH = shutil.which("bash")
N_FR = 24


def _tokens(rng, lab):
    """[n, 2, 65, 1024] float16 with a planted direction per frame: the first half of the width up and the second down for a lead, the reverse for none"""
    n = len(lab)
    x = rng.normal(0.0, 0.1, (n, 2, 65, 1024))
    s = np.where(np.asarray(lab) > 0.5, 1.0, -1.0)[:, None, None, None]
    x[..., :512] += s
    x[..., 512:] -= s
    return x.astype(np.float16)


def _world(base, seed=3):
    """token root, lists, train join and evaluation join; -> dict of paths and the planted labels"""
    rng = np.random.default_rng(seed)
    root = os.path.join(base, "tokroot")
    os.makedirs(os.path.join(root, "tok", "D"))
    os.makedirs(os.path.join(root, "lists"))
    tr = ["trainclip%02d" % i for i in range(8)]
    ev = ["evalclip%02d" % i for i in range(3)]
    lab = {}
    for c in tr + ev:
        g = np.full(N_FR, np.nan)
        v = np.zeros(N_FR, np.int64)
        a = int(rng.integers(2, N_FR - 8))
        v[a:a + 8] = 1
        g[a:a + 8] = rng.uniform(5.0, 55.0, 8)
        v[0] = 1
        g[0] = 80.0                                   # a marked lead above 60 m is no lead for the fit
        v[N_FR - 1] = 1                               # a marked lead with no finite gap is no lead for the fit
        lab[c] = (g, v)
        y = (v == 1) & np.isfinite(g) & (g >= 0.0) & (g <= 60.0)
        np.save(os.path.join(root, "tok", "D", c + ".npy"), _tokens(rng, y))
    with io.open(os.path.join(root, "lists", "train_final.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for i, c in enumerate(tr):
            fh.write("%s %s\n" % (c, "sel" if i >= 6 else "fit"))
    with io.open(os.path.join(root, "lists", "eval141.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(ev) + "\n")

    def join(path, clips):
        off = np.arange(len(clips)) * N_FR
        np.savez(path, clip=np.array(clips), off=off, n=np.full(len(clips), N_FR), lead_gap=np.concatenate([lab[c][0] for c in clips]), lead_valid=np.concatenate([lab[c][1] for c in clips]))
    jt, je = os.path.join(base, "join_train.npz"), os.path.join(base, "join_eval.npz")
    join(jt, tr)
    join(je, ev)
    return {"root": root, "jt": jt, "je": je, "tr": tr, "ev": ev, "lab": lab, "out": os.path.join(base, "out")}


def _y(lab, c):
    g, v = lab[c]
    return (v == 1) & np.isfinite(g) & (g >= 0.0) & (g <= 60.0)


def _fit(w, seed=0, extra=()):
    return PF.main(["--arm", "D", "--head", "H5", "--seed", str(seed), "--root", w["root"], "--out-root", w["out"], "--join-train", w["jt"], "--device", "cpu",
                    "--epochs", "10", "--batch", "16"] + list(extra))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = _world(str(tmp_path_factory.mktemp("world")))
    assert _fit(w) == 0
    assert PD.main(["--seed", "0", "--root", w["root"], "--out-root", w["out"], "--eval-join", w["je"], "--device", "cpu"]) == 0
    return w


def test_generator_reproduces_the_kit_tool(tmp_path):
    out, rec = str(tmp_path / "v.py"), str(tmp_path / "r.json")
    assert MK.main(["--source", SRC, "--out", out, "--record", rec]) == 0
    with io.open(out, "rb") as a, io.open(os.path.join(HERE, "pres_fit.py"), "rb") as b:
        assert a.read() == b.read().replace(b"\r\n", b"\n")
    d = json.load(io.open(rec, encoding="utf-8"))
    k = json.load(io.open(os.path.join(HERE, "PRES_FIT_GEN_RECORD.json"), encoding="utf-8"))
    assert d["pass_p0"] is True and d["n_substitutions"] == len(MK.SUBS) == 9 and d["n_source_lines_changed"] == 9
    assert d["variant_sha256"] == k["variant_sha256"] and d["source_sha256"].startswith(MK.SOURCE_SHA16)
    with pytest.raises(SystemExit):
        MK.main(["--source", SRC, "--out", out, "--record", rec])                       # the output exists


def test_generator_refuses_a_source_that_does_not_match():
    text = io.open(SRC, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
    MK.make(text)
    with pytest.raises(SystemExit):
        MK.make(text.replace("SmoothL1Loss", "L1Loss"))
    with pytest.raises(SystemExit):
        MK.make(text + '\nCARD = "PREREG_V8F_PROOF_ADD99"\n')


def test_generator_refuses_another_source_hash(tmp_path):
    p = str(tmp_path / "s.py")
    with io.open(p, "wb") as fh:
        fh.write(io.open(SRC, "rb").read() + b"\n# one more line\n")
    with pytest.raises(SystemExit):
        MK.main(["--source", p, "--out", str(tmp_path / "o.py"), "--record", str(tmp_path / "o.json")])


def test_variant_constants():
    assert PF.CARD == "PREREG_V8F_PROOF_ADD118" and PF.CARD_HPARAMS["epochs"] == 8 and PF.CARD_HPARAMS["batch"] == 256 and PF.CARD_HPARAMS["lr"] == 1e-3


def test_fit_record_and_labels(world):
    w = world
    d = json.load(io.open(os.path.join(w["out"], "fits", "D_H5_s0", "FIT_RECORD.json"), encoding="utf-8"))
    fit, sel = w["tr"][:6], w["tr"][6:]
    assert d["label"] == "presence" and d["card"] == "PREREG_V8F_PROOF_ADD118" and d["tier"] == "T0"
    assert d["n_samples"] == {"fit": 6 * (N_FR - 4), "sel": 2 * (N_FR - 4)}                    # every frame f >= 4, with or without a lead
    assert d["n_positive"] == {"fit": int(sum(_y(w["lab"], c)[4:].sum() for c in fit)), "sel": int(sum(_y(w["lab"], c)[4:].sum() for c in sel))}
    assert 0 < d["n_positive"]["fit"] < d["n_samples"]["fit"]
    assert d["hparams_equal_card"] is False and np.isfinite(d["kept_sel_loss"])                # the test runs 10 epochs of batch 16
    assert not os.path.exists(os.path.join(w["root"], "fits"))                                 # nothing is written into the token root
    with pytest.raises(SystemExit):
        _fit(w)                                                                                # a finished fit is refused


def test_dump_rows_and_planted_signal(world):
    w = world
    z = np.load(os.path.join(w["out"], "dump", "D_H5_s0", "pres.npz"), allow_pickle=False)
    m = json.load(io.open(os.path.join(w["out"], "dump", "D_H5_s0", "pres_meta.json"), encoding="utf-8"))
    p = z["p"]
    assert len(p) == 3 * N_FR == m["n_rows"] == m["n_rows_of_the_join"] and [str(c) for c in z["clips"]] == w["ev"]
    fr = np.tile(np.arange(N_FR), 3)
    assert np.isnan(p[fr < 4]).all() and np.isfinite(p[fr >= 4]).all() and m["n_finite"] == 3 * (N_FR - 4)
    assert (z["has_out"] == np.isfinite(p)).all() and ((p[fr >= 4] >= 0.0) & (p[fr >= 4] <= 1.0)).all()
    y = np.concatenate([_y(w["lab"], c) for c in w["ev"]])
    k = fr >= 4
    assert p[k & y].mean() > p[k & ~y].mean() + 0.3                                            # the planted direction is read
    assert m["head_pt_sha256"] == m["head_pt_sha256_of_the_fit_record"] and m["card"] == "PREREG_V8F_PROOF_ADD118"
    with pytest.raises(SystemExit):
        PD.main(["--seed", "0", "--root", w["root"], "--out-root", w["out"], "--eval-join", w["je"], "--device", "cpu"])


def test_dump_refuses_a_gap_fit_and_a_frame_count(world, tmp_path):
    w = world
    with pytest.raises(SystemExit):
        PD.main(["--seed", "1", "--root", w["root"], "--out-root", w["out"], "--eval-join", w["je"], "--device", "cpu"])       # no fit of seed 1
    J = np.load(w["je"], allow_pickle=False)
    bad = str(tmp_path / "je_bad.npz")
    np.savez(bad, clip=J["clip"], off=J["off"], n=J["n"] - 1, lead_gap=J["lead_gap"], lead_valid=J["lead_valid"])
    out2 = str(tmp_path / "out2")
    shutil.copytree(os.path.join(w["out"], "fits"), os.path.join(out2, "fits"))
    with pytest.raises(SystemExit):
        PD.main(["--seed", "0", "--root", w["root"], "--out-root", out2, "--eval-join", bad, "--device", "cpu"])


def test_permuted_fit_has_its_own_folder_and_overlap_is_refused(world, tmp_path):
    w = world
    assert _fit(w, extra=["--permute-labels", "--epochs", "1"]) == 0
    d = json.load(io.open(os.path.join(w["out"], "fits", "D_H5_s0_perm", "FIT_RECORD.json"), encoding="utf-8"))
    assert d["permute_labels"] is True and d["n_positive"]["fit"] == json.load(io.open(os.path.join(w["out"], "fits", "D_H5_s0", "FIT_RECORD.json"), encoding="utf-8"))["n_positive"]["fit"]
    w2 = _world(str(tmp_path / "w2"))
    with io.open(os.path.join(w2["root"], "lists", "train_final.txt"), "a", encoding="utf-8", newline="\n") as fh:
        fh.write("%s fit\n" % w2["ev"][0])
    with pytest.raises(SystemExit):
        _fit(w2)


def _env(w, gp, **kw):
    e = dict(os.environ)
    e.update({"PH_ROOT": gp, "PH_KIT": HERE, "PH_TOKROOT": w["root"], "PH_JOIN_TRAIN": w["jt"], "PH_EVAL_JOIN": w["je"], "PH_PY": sys.executable, "PH_DEVICE": "cpu",
              "PH_FIT_EXTRA": "--epochs 1 --batch 16", "PH_MIN_FREE_GB": "-1", "PH_TESTS": os.path.join(gp, "lead_tests", "lead_pytest_fit.out"), "OMP_NUM_THREADS": "2"})
    e.pop("DRY", None)
    e.update(kw)
    return e


def _run(script, env):
    return subprocess.run([BASH, os.path.join(HERE, script)], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)


def _gate(gp, text="7 passed in 1.00s\n"):
    os.makedirs(os.path.join(gp, "lead_tests"), exist_ok=True)
    with io.open(os.path.join(gp, "lead_tests", "lead_pytest_fit.out"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def test_queue_dry_makes_nothing(world, tmp_path):
    gp = str(tmp_path / "gp")
    r = _run("pod_preshead_fit_queue.sh", _env(world, gp, DRY="1"))
    assert r.returncode == 0 and not os.path.exists(gp)
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("  CMD:")]
    assert len(cmds) == 6 and sum("pres_fit.py" in c for c in cmds) == 3 and sum("pres_dump.py" in c for c in cmds) == 3 and sum("--permute-labels" in c for c in cmds) == 2
    assert all("--out-root " + gp in c for c in cmds) and all("--arm D --head H5" in c for c in cmds if "pres_fit.py" in c)


def test_queue_gates(world, tmp_path):
    gp = str(tmp_path / "gp")
    os.makedirs(gp)
    assert _run("pod_preshead_fit_queue.sh", _env(world, gp)).returncode == 4                  # no lead test output
    _gate(gp, "1 failed, 6 passed in 1.00s\n")
    assert _run("pod_preshead_fit_queue.sh", _env(world, gp)).returncode == 4
    _gate(gp)
    r = _run("pod_preshead_fit_queue.sh", _env(world, gp, PH_STOP_S="0"))                      # the stop rule is reached before the first step
    assert r.returncode == 3 and os.path.isfile(os.path.join(gp, "PRESHEAD_STOP.txt")) and not os.path.exists(os.path.join(gp, "PRESHEAD_DONE.txt"))
    assert _run("pod_preshead_fit_queue.sh", _env(world, gp)).returncode == 2                  # a stop marker is not run over


def test_queue_runs_all_steps(world, tmp_path):
    gp = str(tmp_path / "gp")
    _gate(gp)
    r = _run("pod_preshead_fit_queue.sh", _env(world, gp))
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert io.open(os.path.join(gp, "PRESHEAD_DONE.txt"), encoding="utf-8").read().startswith("ZZ PRESHEAD DONE ")
    assert len(io.open(os.path.join(gp, "PRESHEAD_START.txt"), encoding="utf-8").read().split()) == 4
    for f in ("D_H5_s0", "D_H5_s1", "D_H5_s0_perm"):
        assert os.path.isfile(os.path.join(gp, "fits", f, "FIT_RECORD.json")) and os.path.isfile(os.path.join(gp, "dump", f, "pres.npz")) and os.path.isfile(os.path.join(gp, "dump", f, "pres_meta.json"))
    assert sorted(os.listdir(os.path.join(gp, "jobs"))) == sorted("%s.%s" % (j, s) for j in ("fit_s0", "fit_s1", "fit_s0_perm", "dump_s0", "dump_s1", "dump_s0_perm") for s in ("START", "DONE"))
    assert "ALREADY-DONE" in _run("pod_preshead_fit_queue.sh", _env(world, gp)).stdout


def test_after_script(world, tmp_path):
    gp, prev = str(tmp_path / "gp"), str(tmp_path / "prev")
    os.makedirs(gp)
    os.makedirs(prev)
    with io.open(os.path.join(prev, "GAPMARGIN10K_STOP.txt"), "w", encoding="utf-8") as fh:
        fh.write("stop\n")
    r = _run("pod_preshead_after.sh", _env(world, gp, PH_PREV=prev, PH_STEP_S="1"))
    assert r.returncode == 5 and os.path.isfile(os.path.join(gp, "PRESHEAD_AFTER_NOT_STARTED.txt")) and not os.path.exists(os.path.join(gp, "PRESHEAD_START.txt"))
    assert _run("pod_preshead_after.sh", _env(world, gp, PH_PREV=prev)).returncode == 2        # its start marker exists
    gp2, prev2 = str(tmp_path / "gp2"), str(tmp_path / "prev2")
    os.makedirs(gp2)
    os.makedirs(prev2)
    assert _run("pod_preshead_after.sh", _env(world, gp2, PH_PREV=prev2, PH_WAIT_S="1", PH_STEP_S="1")).returncode == 6
    gp3 = str(tmp_path / "gp3")
    os.makedirs(gp3)
    with io.open(os.path.join(prev2, "GAPMARGIN10K_DONE.txt"), "w", encoding="utf-8") as fh:
        fh.write("done\n")
    r = _run("pod_preshead_after.sh", _env(world, gp3, PH_PREV=prev2, PH_STEP_S="1", DRY="1"))
    assert r.returncode == 0 and "DRY run complete" in r.stdout and "the queue starts" in io.open(os.path.join(gp3, "PRESHEAD_AFTER_START.txt"), encoding="utf-8").read()
