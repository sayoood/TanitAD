"""W3 v1.1 agents + driver arithmetic (run in the NAVSIM venv, PYTHONPATH = v1.1 tree).

    PYTHONPATH=<v1.1 tree> C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -m pytest -q tests/test_w3_agents.py

Literal expectations; every guard has an arm that must go RED.
"""
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import pytest

V11 = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
E2_SEAM = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
           "2026-09-19-navsim-refcv4b-bridge/code/tanitad_seam_agent.py")
sys.path[:0] = [V11, CODE]

from navsim.common.dataclasses import EgoStatus  # noqa: E402

import run_v1  # noqa: E402
import w3_agents_v1 as W  # noqa: E402


def _ego(seed):
    r = np.random.default_rng(seed)
    return [EgoStatus(ego_pose=r.normal(size=3).astype(np.float32),
                      ego_velocity=r.normal(size=2).astype(np.float32),
                      ego_acceleration=r.normal(size=2).astype(np.float32),
                      driving_command=np.eye(4, dtype=np.float32)[seed % 4]) for _ in range(4)]


def test_fingerprint_equals_e2_verbatim():
    spec = importlib.util.spec_from_file_location("e2seam", E2_SEAM)
    e2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2)                     # module-level import only; its v2 class is not built
    for s in range(5):
        assert W.fingerprint(_ego(s)) == e2.fingerprint(_ego(s))
    assert W.fingerprint(_ego(0)) != W.fingerprint(_ego(1))


def test_fingerprint_literal():
    es = [EgoStatus(ego_pose=np.zeros(3, np.float32), ego_velocity=np.zeros(2, np.float32),
                    ego_acceleration=np.zeros(2, np.float32),
                    driving_command=np.array([0, 1, 0, 0], np.float32))]
    # literal derived independently: sha1 of struct.pack("<3d2d2d4d", 0,0,0, 0,0, 0,0, 0,1,0,0)
    assert W.fingerprint(es) == "643f9097d3a70bf0f2ce6d914bc27ac7ceb6925c"


def test_stop_agent_is_exactly_zero():
    t = W.StopAgent().compute_trajectory(None)
    assert t.poses.shape == (8, 3) and t.poses.dtype == np.float32
    assert float(np.abs(t.poses).max()) == 0.0
    assert (t.trajectory_sampling.num_poses, t.trajectory_sampling.interval_length) == (8, 0.5)
    assert W.StopAgent.requires_scene is False


class _Meta:
    def __init__(self, tok):
        self.initial_token = tok


class _Scene:
    def __init__(self, tok):
        self.scene_metadata = _Meta(tok)
        self.frames = "POISON"                      # the agent must read nothing but the token


class _AI:
    def __init__(self, es):
        self.ego_statuses = es


def _seam(tmp, toks, fps, poses):
    p = os.path.join(tmp, "seam.npz")
    np.savez(p, token=np.asarray(toks), fingerprint=np.asarray(fps), poses=poses,
             sampling=np.asarray([8, 0.5]), arm=np.asarray("t"))
    return p


def test_seam_agent_lookup_and_refusals():
    with tempfile.TemporaryDirectory() as d:
        es = _ego(3)
        poses = np.arange(48, dtype=np.float32).reshape(2, 8, 3)
        p = _seam(d, ["aaaa", "bbbb"], [W.fingerprint(es), W.fingerprint(_ego(4))], poses)
        ag = W.SeamAgentV1(seam_file=p, call_log=os.path.join(d, "c.jsonl"))
        ag.initialize()
        out = ag.compute_trajectory(_AI(es), _Scene("aaaa"))
        assert np.array_equal(out.poses, poses[0])
        with pytest.raises(KeyError):                               # unknown token: RED
            ag.compute_trajectory(_AI(es), _Scene("cccc"))
        with pytest.raises(ValueError):                             # wrong AgentInput: RED
            ag.compute_trajectory(_AI(es), _Scene("bbbb"))
        calls = [json.loads(x) for x in open(os.path.join(d, "c.jsonl"))]
        assert [c.get("source") for c in calls if c["event"] == "call"] == [
            "seam", "UNKNOWN_TOKEN", "FINGERPRINT_MISMATCH"]


def test_seam_agent_refuses_nonfinite_and_bad_shape():
    with tempfile.TemporaryDirectory() as d:
        bad = np.zeros((1, 8, 3), np.float32)
        bad[0, 3, 1] = np.nan
        ag = W.SeamAgentV1(seam_file=_seam(d, ["aaaa"], ["x"], bad))
        with pytest.raises(ValueError):
            ag.initialize()
        ag = W.SeamAgentV1(seam_file=_seam(d, ["aaaa"], ["x"], np.zeros((1, 6, 3), np.float32)))
        with pytest.raises(ValueError):
            ag.initialize()


# --- driver arithmetic ------------------------------------------------------ #
def _row(nc, dac, ep, ttc, c):
    return {"no_at_fault_collisions": nc, "drivable_area_compliance": dac, "ego_progress": ep,
            "time_to_collision_within_bound": ttc, "comfort": c}


def test_pdms_formula_literals():
    assert run_v1.formula(_row(1, 1, 0.5, 1, 0)) == 0.625            # (2.5 + 5 + 0) / 12
    assert run_v1.formula(_row(0.5, 1, 1, 1, 1)) == 0.5
    assert run_v1.formula(_row(1, 0, 1, 1, 1)) == 0.0
    assert run_v1.formula(_row(1, 1, 0, 1, 0)) == 5 / 12              # the STOP floor term


def test_pdms_formula_is_not_the_v2_denominator():                  # mutation: /14 must differ
    r = _row(1, 1, 0.5, 1, 0)
    assert run_v1.formula(r) != (5 * 0.5 + 5 * 1 + 2 * 0) / 14


def test_navtest_yaml_parse_literal_counts():
    logs, toks = run_v1.navtest_tokens()
    assert (len(logs), len(toks), len(set(toks))) == (136, 12146, 12146)
    import yaml
    d = yaml.safe_load(open(run_v1.NAVTEST_YAML))
    assert logs == d["log_names"] and toks == d["tokens"]
