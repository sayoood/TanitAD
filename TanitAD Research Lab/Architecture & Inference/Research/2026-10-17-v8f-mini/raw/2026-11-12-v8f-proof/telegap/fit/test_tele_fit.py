"""pytest for tele_fit.py, tele_dump.py and pod_telegap_fit_queue.sh (card PREREG_V8F_PROOF_ADD122). Synthetic data only; the token width is patched to 8 and 3 tokens per window."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tele_fit as TF  # noqa: E402
import tele_dump as TD  # noqa: E402

KIT, CFT, CDP = TF.load_kit(None)
NF, WID, NTW = 8, 8, 3
QUEUE = os.path.join(HERE, "pod_telegap_fit_queue.sh")


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    monkeypatch.setattr(CFT, "WIDTH", WID)
    monkeypatch.setattr(CFT, "N_TOK_WIN", NTW)
    monkeypatch.setattr(CFT, "N_TOK", CFT.N_WIN * NTW)


def toks(seed):
    return (np.random.default_rng(seed).standard_normal((NF, 2, NTW, WID)) * 0.5).astype(np.float16)


def make_world(base):
    """base/tok/D (wide), base/tok_T (tele), base/match, base/lists, join.npz, present.npz. clips c0..c2 fit, c3 sel, e0 eval. c0 masked at frame 5, e0 masked at frame 6."""
    base = str(base)
    wide, tele, match, lists = (os.path.join(base, p) for p in ("tok/D", "tok_T", "match", "lists"))
    for d in (wide, tele, match, lists):
        os.makedirs(d, exist_ok=True)
    clips = ["c0", "c1", "c2", "c3", "e0"]
    for i, c in enumerate(clips):
        np.save(os.path.join(wide, c + ".npy"), toks(10 + i))
        np.save(os.path.join(tele, c + ".npy"), toks(100 + i))
        m = np.ones(NF, bool)
        if c == "c0":
            m[5] = False
        if c == "e0":
            m[6] = False
        np.savez(os.path.join(match, c + ".npz"), mask=m)
    with open(os.path.join(lists, "train_final.txt"), "w") as fh:
        fh.write("c0 fit\nc1 fit\nc2 fit\nc3 sel\n")
    with open(os.path.join(lists, "eval141.txt"), "w") as fh:
        fh.write("e0\n")
    tc = clips[:4]
    gap = np.random.default_rng(5).uniform(5, 50, NF * 4)
    np.savez(os.path.join(base, "join.npz"), clip=np.array(tc), off=np.arange(4) * NF, n=np.full(4, NF), lead_gap=gap, lead_valid=np.ones(NF * 4))
    fr = np.arange(NF)
    np.savez(os.path.join(base, "present.npz"), cid=np.zeros(NF, np.int64), clip_index=np.zeros(NF, np.int64), frame=fr, lab_present=np.ones(NF), lab_gap=np.full(NF, 20.0),
             lab_closing=np.zeros(NF), lab_run=np.ones(NF), clips=np.array(["e0"]), dt_s=np.float64(0.2), valid=np.ones(NF, bool))
    return {"wide": wide, "tele": tele, "match": match, "lists": lists, "join": os.path.join(base, "join.npz"), "present": os.path.join(base, "present.npz"), "base": base}


def store_for(arm, w):
    return TF.TeleStore(CFT, arm, w["wide"], w["tele"], w["match"])


def test_combined_windows_and_order(tmp_path):
    w = make_world(tmp_path)
    W, T = np.load(os.path.join(w["wide"], "c1.npy")), np.load(os.path.join(w["tele"], "c1.npy"))
    exp = {"D2": np.stack([W[:, 0], W[:, 1]], 1), "M": np.stack([W[:, 1], T[:, 0]], 1), "F": np.stack([W[:, 0], W[:, 1], T[:, 0], T[:, 1]], 1)}
    for arm, e in exp.items():
        v = store_for(arm, w).get("c1")
        assert v.shape == e.shape and v.ndim == 4 and v.dtype == np.float16 and len(v) == NF
        assert np.array_equal(v[:], e)
        assert np.array_equal(v[2:5], e[2:5])
        assert np.array_equal(v[3], e[3])
        assert np.array_equal(v[-1], e[-1])
        assert np.array_equal(np.asarray(v[1:4]).reshape(3, -1, WID), e[1:4].reshape(3, -1, WID))
        assert np.array_equal(v[::2], e[::2])
        with pytest.raises(IndexError):
            v[NF]
    assert TF.ARM_WINDOW_NAMES == {"D2": ["W32", "W16"], "M": ["W16", "T8"], "F": ["W32", "W16", "T8", "T4"]}


def test_d2_equals_crop_fit_store(tmp_path):
    w = make_world(tmp_path)
    ref = CFT.TokenStore(w["base"], "D")           # <base>/tok/D
    for c in ("c0", "e0"):
        assert np.array_equal(store_for("D2", w).get(c)[:], np.asarray(ref.get(c)))
    s = TF.TeleStore(CFT, "D2", w["wide"])         # D2 needs neither tele folder nor match folder
    assert s.mask("c0") is None and not s.drop_frames("c0", "H5").any()


def test_refusals_of_the_store(tmp_path):
    w = make_world(tmp_path)
    with pytest.raises(SystemExit):
        TF.TeleStore(CFT, "M", w["wide"], None, None)
    np.save(os.path.join(w["tele"], "c2.npy"), toks(1)[:5])          # fewer tele frames than wide frames
    with pytest.raises(SystemExit):
        store_for("M", w).get("c2")
    np.savez(os.path.join(w["match"], "c3.npz"), mask=np.ones(3, bool))   # wrong length
    with pytest.raises(SystemExit):
        store_for("M", w).mask("c3")
    np.save(os.path.join(w["wide"], "c3.npy"), np.zeros((NF, 2, NTW, WID + 1), np.float16))   # wrong shape
    with pytest.raises(SystemExit):
        store_for("D2", w).get("c3")


def test_mask_drops_exactly_the_samples_whose_run_holds_the_frame(tmp_path):
    w = make_world(tmp_path)
    labels = {c: (np.full(NF, 10.0), np.ones(NF, bool)) for c in ("c0", "c1")}
    for head, dropped_expected in (("H5", [5, 6, 7]), ("H1", [5])):
        for arm in ("M", "F"):
            counters = []
            st = store_for(arm, w)
            with TF.patched(CFT, TF.arm_values(CFT, arm, st, TF.make_wrapped_builder(CFT, CFT.build_samples, counters))):
                sc, sf, sy = CFT.build_samples(["c0", "c1"], labels, st, head)
            lo = 4 if head == "H5" else 0
            all_f0 = [f for f in range(lo, NF)]
            assert sorted(sf[sc == 0].tolist()) == [f for f in all_f0 if f not in dropped_expected]
            assert sorted(sf[sc == 1].tolist()) == all_f0            # a clip with every mask true loses nothing
            assert counters == [{"total": 2 * len(all_f0), "kept": 2 * len(all_f0) - len(dropped_expected), "dropped": len(dropped_expected)}]
            assert len(sy) == len(sf)
    counters = []
    st = store_for("D2", w)
    with TF.patched(CFT, TF.arm_values(CFT, "D2", st, TF.make_wrapped_builder(CFT, CFT.build_samples, counters))):
        sc, sf, sy = CFT.build_samples(["c0", "c1"], labels, st, "H5")
    assert counters[0]["dropped"] == 0 and counters[0]["kept"] == len(sf) == 8
    m = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 1], bool)
    assert TF.drop_frames_of(m, "H1", 5).tolist() == (~m).tolist()
    assert TF.drop_frames_of(m, "H5", 5).tolist() == [False, False, True, True, True, True, True, True, True, True]


def test_drop_run_windows_exact():
    m = np.ones(12, bool)
    m[6] = False
    d = TF.drop_frames_of(m, "H5", 5)
    assert np.flatnonzero(d).tolist() == [6, 7, 8, 9, 10]
    assert np.flatnonzero(TF.drop_frames_of(m, "H1", 5)).tolist() == [6]


def test_module_values_restored_after_exception_and_run(tmp_path):
    w = make_world(tmp_path)
    keys = ("N_WIN", "N_TOK", "ARMS", "TokenStore", "build_samples")
    before = {k: getattr(CFT, k) for k in keys}
    st = store_for("F", w)
    with pytest.raises(RuntimeError):
        with TF.patched(CFT, TF.arm_values(CFT, "F", st, TF.make_wrapped_builder(CFT, CFT.build_samples, []))) as changed:
            assert CFT.N_WIN == 4 and CFT.N_TOK == 4 * NTW and "F" in CFT.ARMS
            assert {c["name"] for c in changed} == set(keys)
            raise RuntimeError("boom")
    assert all(getattr(CFT, k) is before[k] for k in keys)
    st = store_for("M", w)
    with TF.patched(CFT, TF.arm_values(CFT, "M", st, TF.make_wrapped_builder(CFT, CFT.build_samples, []))) as changed:
        assert "N_WIN" not in {c["name"] for c in changed} and "N_TOK" not in {c["name"] for c in changed}   # two windows: unchanged values are not listed
    assert all(getattr(CFT, k) is before[k] for k in keys)


def fit_args(w, arm, out, seed=0, perm=False):
    a = ["--arm", arm, "--head", "H5", "--seed", str(seed), "--wide-tok-dir", w["wide"], "--tele-tok-dir", w["tele"], "--match-dir", w["match"], "--out-root", out,
         "--lists-dir", w["lists"], "--join-train", w["join"], "--device", "cpu", "--epochs", "1", "--batch", "4"]
    return a + (["--permute-labels"] if perm else [])


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_tiny_fit_records_and_dump(tmp_path):
    w = make_world(tmp_path / "w")
    out = str(tmp_path / "out")
    keys = ("N_WIN", "N_TOK", "ARMS", "TokenStore", "build_samples")
    before = {k: getattr(CFT, k) for k in keys}
    assert TF.main(fit_args(w, "F", out)) == 0
    assert all(getattr(CFT, k) is before[k] for k in keys)
    fd = os.path.join(out, "fits", "F_H5_s0")
    rec = json.load(open(os.path.join(fd, "TELE_FIT_RECORD.json")))
    assert rec["arm"] == "F" and rec["seed"] == 0 and rec["n_win"] == 4 and rec["n_tok"] == 4 * NTW and rec["windows"] == ["W32", "W16", "T8", "T4"]
    assert rec["crop_fit_sha256"] == sha(os.path.join(KIT, "crop_fit.py")) and rec["module_values_restored"] is True
    assert {c["name"] for c in rec["module_values_changed"]} == {"N_WIN", "N_TOK", "ARMS", "TokenStore", "build_samples"}
    # fit clips c0 (H5 frames 4..7, three of them dropped), c1, c2: 4 + 4 + 4 total 12, kept 9; sel c3: 4 kept
    assert rec["n_samples"]["fit"] == {"total": 12, "kept": 9, "dropped": 3} and rec["n_samples"]["sel"] == {"total": 4, "kept": 4, "dropped": 0}
    fr = json.load(open(os.path.join(fd, "FIT_RECORD.json")))
    assert fr["n_samples"] == {"fit": 9, "sel": 4} and fr["arm"] == "F"
    assert not os.path.exists(os.path.join(w["base"], "fits"))              # nothing written beside the inputs
    assert os.path.isfile(os.path.join(out, "lists", "train_final.txt"))
    with pytest.raises(SystemExit):                                          # a finished fit is not run again
        TF.main(fit_args(w, "F", out))
    # dump
    da = ["--arm", "F", "--head", "H5", "--seed", "0", "--wide-tok-dir", w["wide"], "--tele-tok-dir", w["tele"], "--match-dir", w["match"], "--out-root", out,
          "--present-file", w["present"], "--device", "cpu"]
    assert TD.main(da) == 0
    z = np.load(os.path.join(out, "dump", "F_H5", "reader_s0_score.npz"), allow_pickle=False)
    assert set(z.files) == {"cid", "clip_index", "frame", "lab_present", "lab_gap", "lab_closing", "lab_run", "clips", "dt_s", "gap", "p", "clos", "valid", "seed", "tele_dropped"}
    assert set(z.files) - {"tele_dropped"} == set(CDP.COPIED) | {"gap", "p", "clos", "valid", "seed"}
    # e0 masked at frame 6: H5 runs ending at 6, 7 hold it (frame 8 does not exist); frames 0..3 are NaN for H5 anyway
    assert np.flatnonzero(z["tele_dropped"]).tolist() == [6, 7]
    assert np.isnan(z["gap"][:4]).all() and np.isnan(z["gap"][6:]).all() and np.isfinite(z["gap"][4:6]).all()
    assert np.flatnonzero(z["valid"]).tolist() == [4, 5]
    meta = json.load(open(os.path.join(out, "dump", "F_H5", "reader_s0_meta.json")))
    assert meta["n_tele_dropped_rows"] == 2 and meta["dump_sha256"] == sha(os.path.join(out, "dump", "F_H5", "reader_s0_score.npz"))
    with pytest.raises(SystemExit):
        TD.main(da)
    assert all(getattr(CFT, k) is before[k] for k in keys)


def test_permuted_and_m_arm(tmp_path):
    w = make_world(tmp_path / "w")
    out = str(tmp_path / "out")
    assert TF.main(fit_args(w, "M", out, seed=1, perm=True)) == 0
    assert os.path.isfile(os.path.join(out, "fits", "M_H5_s1_perm", "TELE_FIT_RECORD.json"))
    rec = json.load(open(os.path.join(out, "fits", "M_H5_s1_perm", "TELE_FIT_RECORD.json")))
    assert rec["permute_labels"] is True and rec["n_win"] == 2 and {c["name"] for c in rec["module_values_changed"]} == {"ARMS", "TokenStore", "build_samples"}


def test_d2_fit_equals_crop_fit_on_arm_d(tmp_path):
    import torch
    w = make_world(tmp_path / "w")
    out = str(tmp_path / "out")
    assert TF.main(fit_args(w, "D2", out)) == 0
    rec = json.load(open(os.path.join(out, "fits", "D2_H5_s0", "TELE_FIT_RECORD.json")))
    assert rec["n_samples"]["fit"]["dropped"] == 0 and rec["n_samples"]["sel"]["dropped"] == 0
    base = w["base"]                                     # crop_fit's own run: root=base has lists/ and tok/D
    rc = CFT.main(["--arm", "D", "--head", "H5", "--seed", "0", "--root", base, "--join-train", w["join"], "--device", "cpu", "--epochs", "1", "--batch", "4"])
    assert rc == 0
    a = torch.load(os.path.join(out, "fits", "D2_H5_s0", "head.pt"), map_location="cpu")
    b = torch.load(os.path.join(base, "fits", "D_H5_s0", "head.pt"), map_location="cpu")
    assert a["n_tok"] == b["n_tok"]
    for k in a["state_dict"]:
        assert torch.allclose(a["state_dict"][k], b["state_dict"][k], atol=1e-6), k


# ============================================================================================ the queue
def find_bash():
    g = "C:/Program Files/Git/bin/bash.exe"
    if os.path.isfile(g):
        return g
    return shutil.which("bash")


def run_q(env_extra, tmp):
    bash = find_bash()
    if bash is None:
        pytest.skip("no bash")
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([bash, QUEUE.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(tmp), timeout=120)


def test_queue_dry_order(tmp_path):
    root = str(tmp_path / "tg").replace("\\", "/")
    r = run_q({"DRY": "1", "TG_ROOT": root}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert not os.path.exists(root)                                         # DRY creates nothing
    jobs = [ln.strip() for ln in r.stdout.splitlines() if ln.startswith("JOB ")]
    assert jobs == ["JOB D2_H5_s0:", "JOB F_H5_s0_perm:", "JOB M_H5_s0:", "JOB M_H5_s1:", "JOB M_H5_s2:", "JOB F_H5_s0:", "JOB F_H5_s1:", "JOB F_H5_s2:"]
    cmds = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith("CMD: ") and ("tele_fit.py" in ln or "tele_dump.py" in ln)]
    assert len(cmds) == 16
    for i in range(8):
        assert "tele_fit.py" in cmds[2 * i] and "tele_dump.py" in cmds[2 * i + 1]
    assert "--permute-labels" in cmds[2] and "--permute-labels" in cmds[3] and sum("--permute-labels" in c for c in cmds) == 2
    assert all("--head H5" in c for c in cmds)
    assert all("--out-root %s/fit_out" % root in c for c in cmds)
    assert "reader_s0_score.npz" in cmds[1 + 2 * 4]                         # seed 2 reads the present dump of seed 0
    assert "reader_s1_score.npz" in cmds[1 + 2 * 3]
    assert "--wide-tok-dir /workspace/v8proof/cropread/enc_out/tok/D" in cmds[0]
    assert "--tele-tok-dir %s/enc_out/tok_T" % root in cmds[0] and "--match-dir %s/enc_out/match" % root in cmds[0]
    assert "28 GB" in r.stdout and "10800" in r.stdout


def test_queue_refusals_create_nothing(tmp_path):
    root = tmp_path / "tg"
    root.mkdir()
    rs = str(root).replace("\\", "/")
    r = run_q({"TG_ROOT": rs}, tmp_path)
    assert r.returncode == 2 and "TELEGAP_ENC_DONE" in r.stdout
    (root / "TELEGAP_ENC_DONE.txt").write_text("x\n")
    (root / "TELEGAP_ENC_STOP.txt").write_text("x\n")
    r = run_q({"TG_ROOT": rs}, tmp_path)
    assert r.returncode == 2 and "TELEGAP_ENC_STOP" in r.stdout
    os.remove(root / "TELEGAP_ENC_STOP.txt")
    r = run_q({"TG_ROOT": rs, "TG_CROPREAD": rs + "/nocrop"}, tmp_path)      # inputs missing
    assert r.returncode == 2 and "missing input" in r.stdout
    (root / "TELEGAP_FIT_STOP.txt").write_text("old stop\n")
    r = run_q({"TG_ROOT": rs}, tmp_path)
    assert r.returncode == 2 and "RESUME=1" in r.stdout
    assert sorted(os.listdir(root)) == ["TELEGAP_ENC_DONE.txt", "TELEGAP_FIT_STOP.txt"]     # no marker, no folder made by a refusal


def test_queue_text_rules():
    t = open(QUEUE, encoding="utf-8").read()
    import re
    body = t.replace("kills nothing", "")
    assert not re.search(r"(?<![A-Za-z])(rm|mv)\s", body) and "kill" not in body and "nvidia-smi -r" not in body
    assert "set -C" in t and "TELEGAP_FIT_START.txt" in t and "ZZ TELEGAP_FIT DONE" in t and "memory.max" in t and "memory.stat" in t and "free -" not in t
