"""test_boxsize_fit.py -- tests of the kit of card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0): encoding, the replacement box loader, the restore of the module value,
the record, one tiny fit through the tool on CPU (and its identity run), the queue script (syntax, DRY, no delete or relocate command). Synthetic data only.
Run (one file): PYCAP_MB=3000 bash F:/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest <this file> -q -p no:cacheprovider --basetemp C:/Users/Admin/AppData/Local/Temp/a124fit_bt"""
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
LOC = Path(os.environ.get("BS_TEST_LOCATOR", str(V8 / "lookat" / "locator_fit.py")))      # on the pod the banked tools lie in kit folders
KIT_FIT = Path(os.environ.get("BS_TEST_KIT_FIT", str(V8 / "cropread" / "fit")))
sys.path.insert(0, str(HERE))
import boxsize_fit as BF        # noqa: E402

BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
QUEUE = HERE / "pod_boxsize_queue.sh"
RECORD_KEYS = {"card", "tier", "module_values_changed", "module_values_restored", "locator_fit_sha256", "tool_sha256", "argv", "seed", "permute_targets", "finished_utc"}
LF = BF.import_locator(str(LOC))
LF.set_cft(str(KIT_FIT))
CFT = LF.CFT
ORIGINAL_LOAD_BOX = LF.load_box
NFR = 9


def posix(p):
    return Path(p).as_posix()


# ------------------------------------------------------------------------------------------------ encoding
def test_encode_decode_round_trip_inside_the_limits():
    assert abs(BF.LIMITS[0] - np.exp(-0.4)) < 1e-12 and abs(BF.LIMITS[1] - np.exp(3.6)) < 1e-12
    assert 0.67 < BF.LIMITS[0] < 0.68 and 36.5 < BF.LIMITS[1] < 36.7
    rng = np.random.RandomState(1)
    w, h = rng.uniform(BF.LIMITS[0], BF.LIMITS[1], 500), rng.uniform(BF.LIMITS[0], BF.LIMITS[1], 500)
    a, b = BF.encode(w, h)
    w2, h2 = BF.decode(a, b)
    assert np.max(np.abs(w2 - w)) < 1e-12 and np.max(np.abs(h2 - h)) < 1e-12
    a, b = BF.encode([np.exp(1.6)], [np.exp(1.6 + 0.5)])                # u_w = 0, u_h = 0.5
    assert abs(a[0]) < 1e-12 and abs(b[0] - 2.0) < 1e-12
    a, b = BF.encode([BF.LIMITS[1]], [BF.LIMITS[0]])
    assert abs(a[0] - 32.0) < 1e-9 and abs(b[0] - (-18.0)) < 1e-9       # the limits of the locator's clips: 16 * 2 and -2 + 8 * (-2)


# ------------------------------------------------------------------------------------------------ the replacement loader
def test_replacement_load_box_returns_encoded_sizes_and_drops_bad_frames(tmp_path):
    n = 8
    has = np.array([1, 1, 1, 1, 1, 1, 0, 1], bool)
    w = np.array([2.0, 4.0, np.nan, 5.0, 0.0, 3.0, 2.0, 40.0])
    h = np.array([1.0, 8.0, 3.0, -1.0, 2.0, np.inf, 2.0, 0.1])
    az = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    el = np.zeros(n)
    np.savez(str(tmp_path / "c0.npz"), has_lead=has, ctr_az=az, ctr_el=el, ang_w=w, ang_h=h)
    ok, a, b = BF.box_size_loader(str(tmp_path), "c0", n)
    # frame 0, 1 are good; 2 nan width; 3 negative height; 4 zero width; 5 infinite height; 6 no lead; 7 nan centre
    assert ok.tolist() == [True, True, False, False, False, False, False, False]
    assert ok.dtype == bool and a.shape == (n,) and b.shape == (n,) and a.dtype == np.float64
    ea, eb = BF.encode(w[:2], h[:2])
    assert np.allclose(a[:2], 16 * (np.log(w[:2]) - 1.6)) and np.allclose(b[:2], -2 + 8 * (np.log(h[:2]) - 1.6)) and np.allclose(a[:2], ea) and np.allclose(b[:2], eb)
    assert (a[2:] == 0).all() and (b[2:] == 0).all()
    with pytest.raises(SystemExit):
        BF.box_size_loader(str(tmp_path), "c0", n + 1)
    with pytest.raises(SystemExit):
        BF.box_size_loader(str(tmp_path), "nope", n)
    LF.load_box = BF.box_size_loader
    try:
        t = LF.targets_of(a, b)
    finally:
        LF.load_box = ORIGINAL_LOAD_BOX
    u_w, u_h = np.log(w[:2]) - 1.6, np.log(h[:2]) - 1.6
    assert np.allclose(t[:2, 0], np.clip(u_w, -2, 2)) and np.allclose(t[:2, 1], np.clip(u_h, -2, 2))
    big = BF.encode([40.0], [0.1])                                      # outside the limits: clipped by the locator's own targets
    tb = LF.targets_of(*big)
    assert np.allclose(tb, [[2.0, -2.0]])
    ra, rb = LF.readout(t[:2, 0], t[:2, 1])
    w2, h2 = BF.decode(ra, rb)
    assert np.allclose(w2, w[:2], atol=1e-5) and np.allclose(h2, h[:2], atol=1e-5)


# ------------------------------------------------------------------------------------------------ the tiny world
def make_tiny(tmp):
    rng = np.random.RandomState(0)
    root = tmp / "run1"
    (root / "tok" / "D").mkdir(parents=True)
    (root / "lists").mkdir()
    box = tmp / "box"
    box.mkdir()
    fit, sel, ev = ["f0", "f1", "f2"], ["s0"], ["e0", "e1"]
    man = {}
    for c in fit + sel + ev:
        tok = rng.standard_normal((NFR, 2, 65, 1024)).astype(np.float16)
        p = root / "tok" / "D" / (c + ".npy")
        np.save(str(p), tok)
        man[c] = {"n_frames": NFR, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        has = rng.rand(NFR) < 0.8
        has[5:8] = True
        az, el = rng.uniform(-20, 20, NFR), rng.uniform(-8, 4, NFR)
        az[1] = np.nan
        w, h = rng.uniform(0.5, 40.0, NFR), rng.uniform(0.5, 40.0, NFR)
        w[2] = np.nan
        h[3] = -1.0
        np.savez(str(box / (c + ".npz")), has_lead=has, ctr_az=az, ctr_el=el, ang_w=w, ang_h=h)
    (root / "tok" / "D" / "MANIFEST.json").write_text(json.dumps({"clips": man}), encoding="utf-8")
    (root / "lists" / "train_final.txt").write_text("".join("%s fit\n" % c for c in fit) + "s0 sel\n", encoding="utf-8")
    (root / "lists" / "eval141.txt").write_text("e0\ne1\n", encoding="utf-8")
    names = fit + sel
    np.savez(str(tmp / "join_train.npz"), clip=np.asarray(names), n=np.full(4, NFR), off=np.arange(4) * NFR, lead_gap=rng.uniform(0, 60, 4 * NFR), lead_valid=np.ones(4 * NFR, np.int8))
    return {"root": root, "box": box, "jt": tmp / "join_train.npz"}


def fit_args(w, out, *extra, seed=0):
    return ["--locator", str(LOC), "--seed", str(seed), "--root", str(w["root"]), "--box-dir", str(w["box"]), "--out-root", str(out), "--kit-fit", str(KIT_FIT),
            "--device", "cpu", "--epochs", "2", "--batch", "8"] + list(extra)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bs")
    w = make_tiny(tmp)
    out = tmp / "O"
    assert BF.run(fit_args(w, out, seed=0)) == 0
    assert BF.run(fit_args(w, out, "--permute-targets", seed=0)) == 0
    assert CFT.main(["--arm", "D", "--head", "H5", "--seed", "0", "--epochs", "1", "--batch", "8", "--root", str(w["root"]), "--join-train", str(w["jt"]), "--device", "cpu"]) == 0
    assert BF.run(["--locator", str(LOC), "--identity", "--root", str(w["root"]), "--out-root", str(out), "--kit-fit", str(KIT_FIT), "--join-train", str(w["jt"]), "--device", "cpu"]) == 0
    w["out"], w["tmp"] = out, tmp
    return w


def test_one_tiny_fit_writes_prediction_that_decodes_to_positive_sizes(world):
    out = world["out"]
    assert (out / "fits" / "LOC_s0" / "FIT_RECORD.json").is_file() and (out / "fits" / "LOC_s0" / "head.pt").is_file()
    z = np.load(str(out / "pred" / "LOC_s0.npz"))
    assert len(z["frame"]) == 2 * NFR and np.isfinite(z["az_pred"][z["frame"] >= 4]).all()
    ok = np.isfinite(z["az_pred"]) & np.isfinite(z["el_pred"])
    w, h = BF.decode(z["az_pred"][ok], z["el_pred"][ok])
    assert ok.sum() == 2 * (NFR - 4) and (w > 0).all() and (h > 0).all() and np.isfinite(w).all() and np.isfinite(h).all()
    assert (out / "pred" / "LOC_s0_perm.npz").is_file() and (out / "fits" / "LOC_s0_perm" / "BOXSIZE_FIT_RECORD.json").is_file()
    assert LF.load_box is ORIGINAL_LOAD_BOX


def test_record_has_exactly_the_keys_and_the_hashes(world):
    out = world["out"]
    rec = json.loads((out / "fits" / "LOC_s0" / "BOXSIZE_FIT_RECORD.json").read_text(encoding="utf-8"))
    assert set(rec) == RECORD_KEYS
    assert rec["card"] == "PREREG_V8F_PROOF_ADD124" and rec["tier"] == "T0" and rec["module_values_changed"] == [{"name": "load_box"}] and rec["module_values_restored"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", rec["locator_fit_sha256"]) and re.fullmatch(r"[0-9a-f]{64}", rec["tool_sha256"])
    assert rec["locator_fit_sha256"] == hashlib.sha256(LOC.read_bytes()).hexdigest() and rec["tool_sha256"] == hashlib.sha256((HERE / "boxsize_fit.py").read_bytes()).hexdigest()
    assert rec["seed"] == 0 and rec["permute_targets"] is False and isinstance(rec["argv"], list) and rec["finished_utc"].endswith("Z")
    perm = json.loads((out / "fits" / "LOC_s0_perm" / "BOXSIZE_FIT_RECORD.json").read_text(encoding="utf-8"))
    assert set(perm) == RECORD_KEYS and perm["permute_targets"] is True
    ident = json.loads((out / "BOXSIZE_IDENTITY_RECORD.json").read_text(encoding="utf-8"))
    assert set(ident) == RECORD_KEYS and ident["seed"] is None
    assert json.loads((out / "IDENTITY_K5.json").read_text(encoding="utf-8"))["PASS"] is True


def test_the_fit_trains_on_encoded_sizes_not_centres(world):
    store = CFT.TokenStore(str(world["root"]), "D")
    LF.load_box = BF.box_size_loader
    try:
        sc, sf, sy = LF.build_samples(["f0"], store, "box", box_dir=str(world["box"]))
    finally:
        LF.load_box = ORIGINAL_LOAD_BOX
    z = np.load(str(world["box"] / "f0.npz"))
    ok = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"]) & np.isfinite(z["ang_w"]) & np.isfinite(z["ang_h"]) & (z["ang_w"] > 0) & (z["ang_h"] > 0)
    want = [f for f in range(NFR) if ok[f] and f >= 4]
    assert list(sf) == want
    uw, uh = np.log(z["ang_w"][want]) - 1.6, np.log(z["ang_h"][want]) - 1.6
    assert np.allclose(sy[:, 0], np.clip(uw, -2, 2), atol=1e-6) and np.allclose(sy[:, 1], np.clip(uh, -2, 2), atol=1e-6)


def test_the_module_value_is_restored_after_a_run_that_raises(world):
    assert LF.load_box is ORIGINAL_LOAD_BOX
    bad = fit_args(world, world["tmp"] / "O_bad", seed=7)
    bad[bad.index("--box-dir") + 1] = str(world["tmp"] / "no_such_box_dir")
    with pytest.raises(SystemExit):
        BF.run(bad)                                                     # the replacement loader refuses a missing box file inside the imported main
    assert LF.load_box is ORIGINAL_LOAD_BOX
    with pytest.raises(SystemExit):
        BF.run(["--locator", str(LOC), "--seed", "0"])                  # the imported main refuses missing arguments
    assert LF.load_box is ORIGINAL_LOAD_BOX
    with pytest.raises(SystemExit):
        BF.run(fit_args(world, world["out"], seed=0))                   # a finished fit is refused, nothing overwritten
    assert LF.load_box is ORIGINAL_LOAD_BOX
    assert not (world["tmp"] / "O_bad" / "fits" / "LOC_s7" / "BOXSIZE_FIT_RECORD.json").exists()


# ------------------------------------------------------------------------------------------------ the queue script
def bash_exe():
    b = shutil.which("bash")
    if not b:
        pytest.skip("no bash")
    return b


def test_queue_syntax_dry_run_and_no_delete_or_relocate_command(tmp_path):
    b = bash_exe()
    text = QUEUE.read_text(encoding="utf-8")
    r = subprocess.run([b, "-n", posix(QUEUE)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    root = tmp_path / "bsroot"
    env = dict(os.environ, DRY="1", BS_ROOT=posix(root), BS_OUT=posix(root / "out"), BS_QUEUE=posix(root / "queue"))
    r = subprocess.run([b, posix(QUEUE)], capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    o = r.stdout
    pos = [o.index(s) for s in ("STEP 1 ident", "STEP 2 fit_s0:", "STEP 3 fit_s1:", "STEP 4 fit_s2:", "STEP 5 fit_s0_perm:")]
    assert pos == sorted(pos) and "--identity" in o and "--permute-targets" in o and "ulimit -n 16384" in o
    assert o.count("boxsize_fit.py") >= 5 and "BOXSIZE_START.txt" in o and "BOXSIZE_DONE.txt" in o and "BOXSIZE_STOP.txt" in o
    assert not root.exists() and list(tmp_path.iterdir()) == []         # nothing created
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"(^|[\s;&|(`])(rm|mv|unlink|rmdir|shred|kill|pkill|killall)(\s|$)", code, re.M)
    assert "ulimit -n 16384" in "\n".join(text.splitlines()[:12])
    assert "BS_STOP_S:-5400" in text and "V=/workspace/v8proof" in text and "$V/boxsize" in text and "$B/out" in text and "$B/queue" in text and "-lt 3" in text and "MINGB" in text


def test_no_banned_words_in_the_kit_files():
    for p in (HERE / "boxsize_fit.py", QUEUE, HERE / "BOXSIZE_FIT_DEFS.md", Path(__file__)):
        t = p.read_text(encoding="utf-8").lower()
        for w in BANNED:
            assert w not in t, (p.name, w)
