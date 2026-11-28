"""Gates of the entry (AGENT C): the card-6b starts, the ground-follower fix, the dispatcher over every arm / condition / agent mode, ZERO == P10 through the entry.
Every `check_*` takes `mut=None`; `code/run_mutations_c.py` runs each with a deliberate regression that must raise AssertionError (RED).
CPU only, stub renderer, stub encoder. Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_run.py -q"""
from __future__ import annotations

import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["T2A_STUB"] = "1"
os.environ.setdefault("T2A_KIT", "C:/Users/Admin/t2a_scratch/box2_kit")
os.environ.setdefault("T2V8F_V9_DIR", "F:/TanitAD-scratch/t2v8f/agentA/v9")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import numpy as np  # noqa: E402

import t2v8f_kit as K  # noqa: E402

TMP = os.environ.get("T2V8F_TEST_TMP", "C:/Users/Admin/AppData/Local/Temp/claude/t2v8f_run_tests")


def table():
    return json.load(open(K.SCREEN_TABLE))["scenes"]


# ------------------------------------------------------------------------------------------------ item 1: the starts (card 6b)
def check_starts(mut=None):
    """COUNT: 42 launch scenes all keep >= 1 start (two unless clamped together); every guard has two (distinct); CLAMP: every start >= 18 and the three scenes whose t_c - 4 s
    is earlier than 18 ticks start exactly at 18; COLLAPSE: clamped-together starts are one; GUARD STREAM: the draws equal one hand-rolled default_rng(1) in table order."""
    saved = (K.F_MIN, K.START_DELTAS_S, K.GUARD_N, K.GUARD_SEED, K._collapse)
    try:
        if mut == "no_clamp":
            K.F_MIN = 0
        if mut == "one_delta":
            K.START_DELTAS_S = (4.0,)
        if mut == "one_guard_start":
            K.GUARD_N = 1
        if mut == "guard_seed":
            K.GUARD_SEED = 2
        if mut == "no_collapse":
            K._collapse = lambda c: list(c)
        tab = table()
        st = K.all_starts(tab)
        sel = sorted((e for e in tab if e.get("selected")), key=lambda e: int(e["order_index"]))
        launch = [e for e in sel if e["slot"].startswith("launch")]
        guard = [e for e in sel if e["slot"].startswith("guard")]
        assert len(launch) == 42 and len(guard) == 20
        assert all(len(st[e["scene"]]) >= 1 for e in launch), "a launch scene lost every start"
        assert all(f >= 18 for e in sel for f in st[e["scene"]]), "a start before 18 ticks of history"
        assert all(f + 60 < e["n_ticks"] for e in sel for f in st[e["scene"]])
        for e in launch:                                       # independent recomputation of the clamp and the collapse
            tc = float(e["launch_events"][0]["t_c_s"])
            want = []
            for d in (4.0, 2.0):
                f = max(int(round((tc - d) * 10)), 18)
                if f <= e["n_ticks"] - 61 and f not in want:
                    want.append(f)
            assert st[e["scene"]] == want, f"{e['slot']}: {st[e['scene']]} != {want}"
        clamped = [e["slot"] for e in launch if round((float(e["launch_events"][0]["t_c_s"]) - 4.0) * 10) < 18]
        assert {"launch_17", "launch_25", "launch_38"} <= set(clamped)
        assert all(st[next(e for e in launch if e["slot"] == s)["scene"]][0] == 18 for s in clamped), "a clamped start is not at 18"
        collapsed = [e for e in launch if len(st[e["scene"]]) == 1]
        assert collapsed and all(round((float(e["launch_events"][0]["t_c_s"]) - 2.0) * 10) <= 18 or round((float(e["launch_events"][0]["t_c_s"]) - 2.0) * 10) > e["n_ticks"] - 61
                                 for e in collapsed), "a launch scene has one start for no reason"
        rng = np.random.default_rng(1)                         # card 3: ONE default_rng(1) stream, two starts per guard scene, table order
        for e in guard:
            want = _dedupe([int(rng.integers(18, e["n_ticks"] - 61 + 1)) for _ in range(2)])
            assert st[e["scene"]] == want, f"{e['slot']}: guard starts {st[e['scene']]} != {want}"
        n_guard = sum(len(st[e["scene"]]) for e in guard)
        assert n_guard >= 2 * len(guard) - 1, f"guard starts {n_guard}"
        return {"launch_starts": sum(len(st[e["scene"]]) for e in launch), "guard_starts": n_guard, "clamped_at_18": sorted(clamped),
                "single_start_launch": sorted(e["slot"] for e in collapsed)}
    finally:
        K.F_MIN, K.START_DELTAS_S, K.GUARD_N, K.GUARD_SEED, K._collapse = saved


def _dedupe(c):
    out = []
    for f in c:
        if f not in out:
            out.append(f)
    return out


# ------------------------------------------------------------------------------------------------ the ground follower (fixed at source in t2v8f_exec)
def _ground_correct_4x4_bug(X, gt_T, xy, yaw):
    """The first version of `t2v8f_exec.ground_correct` (kept here as the deliberate-regression arm): a 4x4 `rz` multiplied with a 3x3 block."""
    P_ = np.stack([T[:2, 3] for T in gt_T])
    G = gt_T[int(np.argmin(np.linalg.norm(P_ - np.asarray(xy)[None, :2], axis=1)))]
    out = np.eye(4)
    out[:3, :3] = X.rz(yaw) @ (X.rz(-X.yaw_of(G)) @ G[:3, :3])
    out[:3, 3] = (xy[0], xy[1], G[2, 3])
    return out


def check_ground_correct(mut=None):
    """`t2v8f_exec.ground_correct` equals the harness' `GroundFollower.correct` on rolled / pitched poses, and the entry installs nothing over it."""
    import t2v8f_run as R
    R.setup()
    X, cd = R.STATE["X"], R.STATE["cd"]
    assert "gc_e3" not in R.STATE and R.STATE["K"].ground_correct(*[[np.eye(4)], [0.0, 0.0], 0.0]).shape == (4, 4), "the entry still overrides ground_correct"
    rng = np.random.default_rng(7)
    gt = []
    for i in range(40):
        T = X.rz(0.05 * i)
        c, s = np.cos(0.02 * rng.standard_normal()), np.sin(0.02 * rng.standard_normal())
        T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])        # a small roll
        T[0, 3], T[1, 3], T[2, 3] = 3.0 * i, 0.5 * i, 0.01 * i
        gt.append(T)
    fns = {"e3_version": lambda g, xy, yaw: _ground_correct_4x4_bug(X, g, xy, yaw),
           "yaw_from_log": lambda g, xy, yaw: X.ground_correct(g, xy, X.yaw_of(g[int(np.argmin(np.linalg.norm(np.stack([t[:2, 3] for t in g]) - np.asarray(xy)[None, :2], axis=1)))]))}
    fn = fns.get(mut, X.ground_correct)
    gf = cd.GroundFollower(gt)
    worst = 0.0
    for q in range(25):
        Tn = X.rz(0.1 * q + 0.03)
        Tn[0, 3], Tn[1, 3] = 4.0 * q + 0.3, 0.4 * q - 0.2
        want = gf.correct(Tn)[0]
        got = fn(gt, Tn[:2, 3], X.yaw_of(Tn))
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1e-12, f"ground_correct differs from GroundFollower.correct by {worst:.3g}"
    return {"max_dev": worst}


# ------------------------------------------------------------------------------------------------ the dispatcher: every arm, every condition, both agent modes
def run_set(arm, tcond, agents, out, starts="60", scene_idx=0, scene_id="stubscene0", extra=()):
    import t2v8f_run as R
    R.setup()
    R.STATE["nav"] = {}
    blind = os.path.join(TMP, "blind_mean.pt")
    if not os.path.exists(blind):
        import torch
        os.makedirs(TMP, exist_ok=True)
        torch.save(torch.zeros(40, 1024, dtype=torch.float16), blind)           # a stand-in mean: the entry's plumbing, not the banked values
    av = ["--scene-dir", "stub", "--arm", arm, "--condition", agents, "--starts", starts, "--steps", "60", "--out", out, "--t2-cond", tcond, "--t2-scene-idx", str(scene_idx),
          "--t2-scene-id", scene_id, "--t2-encoder", "stub", "--t2-device", "cpu", "--t2-plumbing", "--t2-blind-mean", blind, *extra]
    R.rollout_main(av)
    return [json.loads(line) for line in open(os.path.join(out, "records.jsonl"), encoding="utf-8")]


def _fake_attach():
    m = types.ModuleType("actor_map")
    m.attach_actors_verified = lambda r, sd: {"verdict": "ACCEPTED", "n_mapped": 0}
    sys.modules["actor_map"] = m


NEED = ("scene", "f0", "t0_step", "cl", "g", "v0", "track", "extra", "plans", "arm", "tcond", "agents", "tier", "purpose")


def check_dispatch(mut=None):
    """All 7 arms under T2, C0_s0 under RR and logged, both agent modes (objects through main with the actor attach faked), records in the scorer's input format;
    ZERO == P10 exactly through the entry (T3); RR renders poses different from T2 (the condition reaches the loop)."""
    import t2v8f_exec as X
    import t2v8f_run as R
    R.setup()
    orig = X.run_t2_rollout
    if mut == "cond_dropped":
        X.run_t2_rollout = lambda *a, **k: orig(*a, **dict(k, condition="T2"))
    try:
        _fake_attach()
        recs = {}
        for arm in R.ARMS:
            recs[(arm, "T2")] = run_set(arm, "T2", "empty", f"{TMP}/{mut}/d_{arm}_T2")[0]
        for tc in ("RR", "logged"):
            recs[("C0_s0", tc)] = run_set("C0_s0", tc, "empty", f"{TMP}/{mut}/d_C0_s0_{tc}")[0]
        recs[("C0_s0", "T2", "objects")] = run_set("C0_s0", "T2", "objects", f"{TMP}/{mut}/d_C0_s0_obj")[0]
    finally:
        X.run_t2_rollout = orig
    for k, r in recs.items():
        assert all(n in r for n in NEED), (k, [n for n in NEED if n not in r])
        assert np.asarray(r["cl"]).shape == (30, 2) and np.asarray(r["g"]).shape == (30, 2) and len(r["plans"]) == 3 and len(r["extra"]) == 3
        assert all("s_ckpt" in e and "s_pose" in e for e in r["extra"]), "S-POS inputs missing from the record"
        assert r["purpose"].startswith("PLUMBING ONLY") and r["tier"].startswith("T2 re-perception")
        assert r["policy_track_equals_executor"] is True
    assert recs[("C0_s0", "T2", "objects")]["agents"] == "objects" and os.path.exists(f"{TMP}/{mut}/d_C0_s0_obj/actor_attach.json")
    assert recs[("ZERO", "T2")]["plans"] == recs[("P10", "T2")]["plans"] and recs[("ZERO", "T2")]["cl"] == recs[("P10", "T2")]["cl"], "T3 through the entry: ZERO != P10"
    assert recs[("C0_s0", "T2")]["plans"] != recs[("P10", "T2")]["plans"], "the model arm equals the prior: the control would be vacuous"

    def pose(r):
        return np.array([[c["x"], c["y"]] for c in r["calls"] if c["phase"] == "exec"])
    d = float(np.abs(pose(recs[("C0_s0", "RR")]) - pose(recs[("C0_s0", "T2")])).max())
    assert d > 1e-3, f"RR renders the same poses as T2 (max |d| {d:.3g} m): the condition did not reach the loop"
    assert recs[("C0_s0", "logged")]["calls"][-1]["tau"] == recs[("C0_s0", "logged")]["t0_step"] + 30
    return {"arms": len(R.ARMS), "conditions": ["T2", "RR", "logged"], "agents": ["empty", "objects"], "rr_vs_t2_max_pose_dev_m": d}


def test_starts():
    print(check_starts())


def test_ground_correct():
    print(check_ground_correct())


def test_dispatch():
    print(check_dispatch())
