"""Dry test of train_full_box2.py on fake files (CPU): the run completes, every control is computed, the planted read is 1, and the names follow card ADD149."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_full_box2 as F  # noqa: E402

torch = pytest.importorskip("torch")
T = 40


def _clip(path, rng):
    tok = (rng.normal(size=(T, 40, 1024)) * 0.5).astype(np.float32)
    torch.save({"pool4x10": torch.from_numpy(tok).to(torch.float8_e4m3fn), "v": torch.from_numpy(rng.uniform(5, 15, T).astype(np.float32))}, path)


def _fake(tmp, rng, n_corpus=24, n_eval=4):
    corp, ev = os.path.join(tmp, "corpus"), os.path.join(tmp, "eval")
    for d in (corp, ev):
        os.makedirs(os.path.join(d, "clips"))
    man, sha, k, has, lead, gap = [], [], [], [], [], []
    for i in range(n_corpus):
        cid, s12 = "c%04d" % i, "s%011d" % i
        _clip(os.path.join(corp, "clips", cid + ".pt"), rng)
        man.append({"clip": cid, "sha12": s12, "file": "clips/%s.pt" % cid, "shapes": {"pool4x10": [T, 40, 1024]}})
        g0 = rng.uniform(10, 40)
        for kk in range(2 * T):
            sha.append(s12); k.append(kk); has.append(1); lead.append(1); gap.append(g0 + 0.05 * kk)
    open(os.path.join(corp, "manifest.jsonl"), "w").write("\n".join(json.dumps(r) for r in man))
    lf = os.path.join(tmp, "lead.npz")
    np.savez(lf, sha12=np.array(sha), k=np.array(k), has=np.array(has), lead=np.array(lead), gap=np.array(gap, np.float32))
    em, off = [], []
    for i in range(n_eval):
        cid = "e%04d" % i
        _clip(os.path.join(ev, "clips", cid + ".pt"), rng)
        em.append({"clip": cid, "file": "clips/%s.pt" % cid, "shapes": {"pool4x10": [T, 40, 1024]}})
        off.append(i * T)
    open(os.path.join(ev, "manifest.jsonl"), "w").write("\n".join(json.dumps(r) for r in em))
    jf = os.path.join(tmp, "join.npz")
    N = n_eval * T
    np.savez(jf, clip=np.array([r["clip"] for r in em]), off=np.array(off), n=np.full(n_eval, T), has=np.ones(N, bool), lead_valid=np.ones(N, np.int8),
             lead_gap=np.tile(20 + 0.1 * np.arange(T), n_eval).astype(np.float32))
    return corp, ev, lf, jf


def test_dry_run(tmp_path):
    rng = np.random.default_rng(0)
    corp, ev, lf, jf = _fake(str(tmp_path), rng)
    out = os.path.join(str(tmp_path), "out")
    F.main(["--out", out, "--steps", "40", "--ckpt-every", "20", "--n-fit", "12", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard"])
    r = json.load(open(os.path.join(out, "FULL.json")))
    assert "error" not in r, r.get("traceback")
    assert r["split"]["fit_used"] == 12 and r["split"]["held_out"] == 2
    assert r["controls"]["c2_disjoint"]["PASS"]
    assert r["controls"]["c3_planted_read"]["PASS"]
    assert r["controls"]["c4_checkpoint"]["PASS"]
    assert r["controls"]["c5_token_bytes"]["PASS"], r["controls"]["c5_token_bytes"]
    assert set(r["train"]["ckpt_sha256"]) == {"20", "40"}
    assert r["held_out"]["n_S"] > 0 and r["held_out"]["n_A"] > 0
    assert r["eval141"]["n_S"] > 0
    assert os.path.exists(os.path.join(out, "head_40.pt"))
    # card ADD150: the reference head read beside; the run's own checkpoint as the reference gives exactly 0 with CI [0, 0]
    ref = os.path.join(out, "head_40.pt")
    sha = F.sha_file(ref)
    out2 = os.path.join(str(tmp_path), "out2")
    F.main(["--out", out2, "--steps", "40", "--ckpt-every", "20", "--n-fit", "12", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard",
            "--ref-ckpt", ref, "--ref-sha", sha[:16]])
    r2 = json.load(open(os.path.join(out2, "FULL.json")))
    assert "error" not in r2, r2.get("traceback")
    pb = r2["paired_vs_ref_held_out"]
    assert r2["held_out"]["S"] == pytest.approx(pb["S_head"]) and r2["held_out"]["A"] == pytest.approx(pb["A_head"])
    if F.sha_file(os.path.join(out2, "head_40.pt")) == sha:                                  # bit-identical retrain on CPU: the difference is an identity
        assert pb["dS"] == 0.0 and pb["dA"] == 0.0 and pb["dS_ci95"] == [0.0, 0.0]
    out3 = os.path.join(str(tmp_path), "out3")
    F.main(["--out", out3, "--steps", "4", "--ckpt-every", "4", "--n-fit", "12", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard",
            "--ref-ckpt", ref, "--ref-sha", "0000dead"])
    assert "does not start with" in json.load(open(os.path.join(out3, "FULL.json")))["error"]


def test_paired_bootstrap_planted():
    rng = np.random.default_rng(5)
    clips = []
    for _ in range(30):
        c = rng.normal(size=40)
        clips.append({"v": rng.uniform(5, 10, 40), "clos_lab": c, "clos_head": c + 0.1 * rng.normal(size=40), "clos_ref": c + 2.0 * rng.normal(size=40)})
    pb = F.paired_bootstrap(clips, n_boot=300)
    assert pb["dS"] > 0.3 and pb["dS_ci95"][0] > 0                                           # the less noisy head reads higher, separated
    same = [dict(d, clos_ref=d["clos_head"]) for d in clips]
    pz = F.paired_bootstrap(same, n_boot=50)
    assert pz["dS"] == 0.0 and pz["dA"] == 0.0 and pz["dS_ci95"] == [0.0, 0.0] and pz["dA_ci95"] == [0.0, 0.0]


def test_guard_stops(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=6, n_eval=2)
    monkeypatch.setattr(F, "commit_avail_gb", lambda: 1.0)
    out = os.path.join(str(tmp_path), "out")
    F.main(["--out", out, "--steps", "4", "--n-fit", "4", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf])
    r = json.load(open(os.path.join(out, "FULL.json")))
    assert r["name"] == "LEADSPEEDFULL10K-STOPPED-MEMORY"
    assert "train" not in r


def test_guard_counts_process_once(tmp_path, monkeypatch):
    """AM1: need = 5.5 + clips + max(0, 2.5 - own). AM3: 4 fit clips at 0.1 MB (memmap tokens) = 0.0004 GB; with own 1.3 the need is 6.7004 GB."""
    rng = np.random.default_rng(3)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=6, n_eval=2)
    monkeypatch.setattr(F, "own_commit_gb", lambda: 1.3)
    for avail, stopped in ((6.70, True), (6.75, False)):
        out = os.path.join(str(tmp_path), "out%s" % avail)
        monkeypatch.setattr(F, "commit_avail_gb", lambda a=avail: a)
        F.main(["--out", out, "--steps", "4", "--ckpt-every", "4", "--n-fit", "4", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf])
        r = json.load(open(os.path.join(out, "FULL.json")))
        assert (r.get("name") == "LEADSPEEDFULL10K-STOPPED-MEMORY") == stopped, (avail, r.get("error"))
        assert abs(r["guard"]["need_GB"] - 6.70) < 0.005


def test_names_and_consequence():
    assert F.name_of(0.30, 0.0) == "LEADSPEEDFULL10K-CARRIED"
    assert F.name_of(0.29, 0.30) == "LEADSPEEDFULL10K-SPEED-ONLY"
    assert F.name_of(0.29, 0.29) == "LEADSPEEDFULL10K-NOT-CARRIED"
    assert F.consequence("LEADSPEEDFULL10K-CARRIED", 0.31, 0.0).startswith("WITH-FORECAST")
    assert F.consequence("LEADSPEEDFULL10K-CARRIED", 0.20, 0.5).startswith("BLOCKER")
    assert F.consequence("LEADSPEEDFULL10K-SPEED-ONLY", 0.0, 0.31).startswith("CLOSING-ONLY")
    assert F.consequence("LEADSPEEDFULL10K-NOT-CARRIED", 0.9, 0.9).startswith("BLOCKER")


def test_read_quantities_planted_identity():
    rng = np.random.default_rng(2)
    v = rng.uniform(5, 10, 30)
    c = rng.normal(size=30)
    c[:3] = np.nan
    d = {"v": v, "clos_lab": c, "clos_head": c.copy(), "pres_lab": np.ones(30, bool), "pres_head": rng.uniform(size=30), "gap_lab": np.full(30, 20.0), "gap_head": np.full(30, 21.0)}
    q = F.read_quantities([d])
    assert abs(q["S"] - 1) < 1e-12 and abs(q["A"] - 1) < 1e-12 and abs(q["gap_mae_m"] - 1.0) < 1e-12


def test_train_seed_changes_training_not_split(tmp_path):
    rng = np.random.default_rng(7)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=12, n_eval=2)
    outs = {}
    for name, extra in (("a", []), ("b", []), ("c", ["--train-seed", "149"])):
        out = os.path.join(str(tmp_path), name)
        F.main(["--out", out, "--steps", "6", "--ckpt-every", "6", "--n-fit", "6", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard"] + extra)
        r = json.load(open(os.path.join(out, "FULL.json")))
        assert "error" not in r, r.get("traceback")
        outs[name] = (r["seeds"], r["split"], r["train"]["ckpt_sha256"]["6"])
    assert outs["a"][0] == {"split": 148, "train": 148} and outs["c"][0] == {"split": 148, "train": 149}
    assert outs["a"][1] == outs["c"][1]
    assert outs["a"][2] == outs["b"][2]                                                     # same seed, same weights (CPU)
    assert outs["a"][2] != outs["c"][2]                                                     # another training seed, other weights
