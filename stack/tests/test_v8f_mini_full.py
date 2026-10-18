"""PREREG_V8F_MINI Amendment 4 (v8F-mini on the full corpus, box 2): the queue and the verdict. CPU only, < 30 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_mini_full.py

What is pinned, and the control that must go red for each:
  * the queue (CRLF) trains every arm on cache_full with --exclude 540:600, --kin-mode yaw, cache_full's native_tracks.pt, the 4S budget
    and P10-yaw-4512 (copied out of the ampersand path), and never names the 600-clip cache root;
  * its dry run passes ARM / SEED / GRID to train and eval in that order (a goto-dispatched stage once read the stage name as the arm),
    and an 8x20 run gets its own directory and tag;
  * the verdict REFUSES mixed provenance (another clip list, a missing holdout, an unfinished run), reads K-1 per arm with no selection,
    and WM-1 passes only when BOTH seeds are separated, negative and larger than 2F.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
QUEUE = PKG / "code" / "v8f_mini_box2_queue.cmd"
VERDICT = PKG / "code" / "v8f_mini_full_verdict.py"
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


# ================================================================================================ the queue
@pytest.mark.skipif(not QUEUE.exists(), reason="queue not in this checkout")
def test_queue_trains_on_cache_full_with_the_ra1_holdout_withheld_and_the_compliant_flags():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 50, "the queue must stay CRLF"
    txt = raw.decode("utf-8")
    a = [ln for ln in txt.splitlines() if ln.startswith('set "A=')]
    assert len(a) == 1
    for flag in ("--exclude 540:600", "--kin-mode yaw", "--tracks %TRACKS%", "--steps-mult 4", "--batch 64", "--p10 %P10%",
                 "--cache-dir %CACHEFULL%", "--v9-train-md5 83e30bd3791f0767707eade5f4671759"):
        assert flag in a[0], flag
    assert 'set "TRACKS=%CACHEFULL%\\native_tracks.pt"' in txt
    assert "\\raw\\p10_yaw4512\"" in txt and "p10_yaw540" not in txt and "p10_yaw\\" not in txt
    assert "D:\\tanitad\\cache\\" not in txt and 'CACHE=D:\\tanitad\\cache"' not in txt   # the 600-clip root is never named


@pytest.mark.skipif(os.name != "nt" or not QUEUE.exists(), reason="the queue is a Windows batch file")
def test_queue_dry_run_passes_arm_seed_grid_in_order_and_8x20_gets_its_own_run(tmp_path):
    env = dict(os.environ, V8F_DRYRUN="1", REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"),
               V9=str(tmp_path / "v9"))

    def q(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        return [ln for ln in r.stdout.splitlines() if ln.strip()]

    g = q("gate")
    assert len(g) == 2 and all("--selfcheck-only" in ln and "--exclude 540:600" in ln for ln in g)
    t = q("train M2b 1")
    assert len(t) == 1 and "--arm M2b --seed 1 --grid pool4x10" in t[0] and str(tmp_path / "runsm" / "M2b_s1") + " " in t[0]
    t8 = q("train M1 0 pool8x20")
    assert "--arm M1 --seed 0 --grid pool8x20" in t8[0] and "M1_s0_8x20" in t8[0]
    e = q("eval M1 1")
    assert "--tag M1_s1 " in e[0] and "M1_s1" + os.sep + "ckpt_last.pt" in e[0] and "--kin-mode yaw" in e[0] and "native_tracks.pt" in e[0]
    e8 = q("eval M2b 0 pool8x20")
    assert "--tag M2b_8x20_s0 " in e8[0]
    c = q("compare")
    assert all(f"{k}=" in c[0] for k in ("M1_s0", "M1_s1", "M2b_s0", "M2b_s1", "M1-navshuf_s0", "M1-navshuf_s1"))
    assert "p10_eval_pred6.npy" in q("floors")[0]


# ================================================================================================ the verdict
def _verdict():
    spec = importlib.util.spec_from_file_location("v8f_mini_full_verdict", VERDICT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _world(tmp, gain=0.30, seed_gap=0.01, s1_gain=None, s1_split=0.0, train_sha="t" * 64, holdout=HOLDOUT_SHA, drop=None):
    """4 runs on 20 episodes x 10 windows. M2b = M1 - gain (seed 1: s1_gain); the seeds differ by seed_gap. s1_split adds +x to the
    even and -x to the odd episodes of M2b seed 1: its MEAN (so F) is unchanged, only its episode spread grows."""
    rng = np.random.default_rng(0)
    eid = np.repeat(np.arange(20), 10)
    base = 2.5 + rng.normal(0, 0.4, eid.size) + np.repeat(rng.normal(0, 0.3, 20), 10)
    ev, rd = tmp / "eval", tmp / "runs"
    ev.mkdir(parents=True, exist_ok=True)
    g = {0: gain, 1: gain if s1_gain is None else s1_gain}
    for arm in ("M1", "M2b"):
        for s in (0, 1):
            ade = base + s * seed_gap - (g[s] if arm == "M2b" else 0.0) + rng.normal(0, 0.02, eid.size)
            if arm == "M2b" and s == 1:
                ade = ade + s1_split * np.where(eid % 2 == 0, 1.0, -1.0)
            np.savez(ev / f"eval_{arm}_s{s}.npz", ade6=ade, eid=eid)
            json.dump({"raw": {"traj": {"ade6_m": {"mean": float(ade.mean())}}, "families": {"lateral": {}}}},
                      open(ev / f"eval_{arm}_s{s}.json", "w"))
            if drop == (arm, s):
                continue
            (rd / f"{arm}_s{s}").mkdir(parents=True, exist_ok=True)
            json.dump({"done": True, "steps": 100, "S": 25, "arm": arm, "seed": s, "n_clips": 4512, "train_sha256": train_sha,
                       "excluded_sha256": holdout}, open(rd / f"{arm}_s{s}" / "summary.json", "w"))
    arms = {a: {"K1_ade6_minus_P10": [{"mean": -0.1, "separated": True}] * 2, "seed_floor_F_m": 0.01, "K1_pass": a == "M2b"}
            for a in ("M1", "M2b")}
    json.dump({"arms": arms, "P10_consistency_a_rate": 0.32}, open(ev / "compare.json", "w"))
    return ev, rd


def _run(tmp, **kw):
    ev, rd = _world(tmp, **kw)
    out = tmp / "verdict.json"
    assert _verdict().main(["--eval-dir", str(ev), "--runs-dir", str(rd), "--out", str(out)]) == 0
    return json.load(open(out))


@pytest.mark.skipif(not VERDICT.exists(), reason="verdict not in this checkout")
def test_verdict_wm1_passes_only_when_both_seeds_beat_2F(tmp_path):
    v = _run(tmp_path / "a")
    assert v["WM1"]["pass"] and all(c["separated"] and c["mean"] < 0 for c in v["WM1"]["per_seed"])
    assert v["K1"]["pass"] and v["K1"]["per_arm"]["M1"]["K1_pass"] is False and "no selection" in v["K1"]["multiplicity"]
    assert v["provenance"]["excluded_sha256"] == HOLDOUT_SHA and "families" in v and "_floors" not in v["families"]
    big_floor = _run(tmp_path / "b", gain=0.30, seed_gap=0.40)                  # separated, negative, but |delta| < 2F
    assert not big_floor["WM1"]["pass"] and big_floor["WM1"]["seed_floor_F_m"]["max"] > 0.15
    one_seed = _run(tmp_path / "c", gain=0.30, s1_gain=0.0)                     # seed 1 shows no gain
    assert not one_seed["WM1"]["pass"] and one_seed["WM1"]["per_seed"][0]["separated"] and not one_seed["WM1"]["per_seed"][1]["separated"]
    split = _run(tmp_path / "e", gain=0.30, s1_split=2.0)                       # same means and F; seed 1 NOT separated
    assert split["WM1"]["per_seed"][0]["separated"] and not split["WM1"]["per_seed"][1]["separated"]
    assert split["WM1"]["seed_floor_F_m"]["max"] < 0.05 and not split["WM1"]["pass"]       # one seed is not both seeds
    worse = _run(tmp_path / "d", gain=-0.30)                                    # M2b separated WORSE
    assert not worse["WM1"]["pass"] and all(c["mean"] > 0 for c in worse["WM1"]["per_seed"])
    assert "WM-1 FAIL" in worse["summary"] and "rung beta" in worse["next"][1]


@pytest.mark.skipif(not VERDICT.exists(), reason="verdict not in this checkout")
def test_verdict_refuses_mixed_provenance(tmp_path):
    V = _verdict()
    for i, kw in enumerate((dict(holdout="0" * 64), dict(drop=("M2b", 1)))):
        ev, rd = _world(tmp_path / f"r{i}", **kw)
        with pytest.raises(SystemExit, match="REFUSED"):
            V.main(["--eval-dir", str(ev), "--runs-dir", str(rd), "--out", str(tmp_path / "x.json")])
    ev, rd = _world(tmp_path / "r2")
    s = json.load(open(rd / "M1_s1" / "summary.json"))
    s["train_sha256"] = "u" * 64
    json.dump(s, open(rd / "M1_s1" / "summary.json", "w"))
    with pytest.raises(SystemExit, match="one clip list"):
        V.main(["--eval-dir", str(ev), "--runs-dir", str(rd), "--out", str(tmp_path / "y.json")])
