"""refcv7 — the TRAINER wiring: pin refusals (each with its GREEN control), the weight-gate
rows, and the future-agent oracle block's frame transform (analytic targets)."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T                                    # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

BASE = ["--arm", "hier", "--size", "tiny", "--out", "X",
        "--v7-labels", "labels.jsonl.gz",
        "--agents", "head", "--w-agent", "1.0",
        "--agent-join", "join.jsonl.xz", "--agent-join-verify", "off",
        "--trunk", "timm", "--tac-decoder-v6", "--w-tac-v6", "1.0"]
R7 = ["--refcv7", "--w-r7-wta", "1.0", "--w-r7-scorer", "1.0",
      "--r7-nav-tau-rad", "0.2", "--w-map", "1.0", "--map-gt-root", "maps"]


def _pin(argv):
    args = T.build_parser().parse_args(argv)
    cfg = v3.RefCV3Config(hier=(args.arm == "hier"))
    try:
        T._pin_refcv7(cfg, args)          # the refcv7 pin alone: isolates ITS refusals
        return True, None, cfg
    except SystemExit as exc:
        return False, str(exc), None


def _drop(argv, flag, n_values=1):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1 + n_values
            continue
        out.append(argv[i])
        i += 1
    assert len(out) != len(argv), f"drop({flag!r}) removed nothing"
    return out


def _set(argv, flag, value):
    out = list(argv)
    assert flag in out
    out[out.index(flag) + 1] = value
    return out


def test_GREEN_the_full_refcv7_arm_passes_and_pins_the_config():
    ok, msg, cfg = _pin(BASE + R7)
    assert ok, msg
    assert cfg.refcv7 is True and cfg.refcv7_select is True
    assert cfg.refcv7_head_cfg.n_queries == 64 and cfg.refcv7_head_cfg.d_model == 256
    assert cfg.refcv7_head_cfg.scorer_self_attn is False


def test_GREEN_no_flag_is_a_no_op():
    ok, msg, cfg = _pin(BASE)
    assert ok and cfg.refcv7 is False


@pytest.mark.parametrize("argv,needle", [
    (_drop(BASE + R7, "--tac-decoder-v6", 0), "--tac-decoder-v6"),
    (_set(BASE + R7, "--trunk", "refc"), "--trunk timm"),
    (_set(_set(BASE + R7, "--w-r7-wta", "0"), "--w-r7-scorer", "0"), "BOTH weights 0"),
    (_set(BASE + R7, "--w-r7-wta", "0"), "--w-r7-wta 0"),
    (_set(BASE + R7, "--w-r7-scorer", "0"), "untrained scorer"),
    (_drop(BASE + R7, "--r7-nav-tau-rad"), "--r7-nav-tau-rad"),
    (_set(BASE + R7, "--w-map", "0"), "SAM3 map"),
    (_drop(BASE + R7, "--refcv7", 0), "without --refcv7"),
    (_set(BASE + R7, "--arm", "flat"), "--arm hier"),
])
def test_RED_each_dead_configuration_refuses_for_its_own_reason(argv, needle):
    ok, msg, _ = _pin(argv)
    assert not ok, f"expected a refusal: {argv}"
    assert needle in msg, f"refused for the wrong reason: {msg[:220]}"


def test_GREEN_scorer_off_is_legal_when_selection_is_off():
    ok, msg, cfg = _pin(_set(BASE + R7, "--w-r7-scorer", "0") + ["--r7-no-select"])
    assert ok, msg
    assert cfg.refcv7_select is False


def test_both_new_weights_have_discriminating_gate_rows():
    for key in ("w_r7_wta", "w_r7_scorer"):
        g = T.REFC_WEIGHT_GATES[key]["gate"]
        met, _ = g(T.build_parser().parse_args(BASE + R7))
        assert met, key
        not_met, _ = g(T.build_parser().parse_args(BASE))
        assert not not_met, key


def test_conflict_detector_sees_both_refcv7_terms():
    names = {n for n, _ in T.CONFLICT_PERCEPTION_TERMS}
    assert {"r7_wta", "r7_scorer"} <= names


# --------------------------------------------------------------------------- #
# the future-agent block: rig(f+h) -> world -> rig(f), analytic               #
# --------------------------------------------------------------------------- #
class _Join:
    """A static agent at world (wx, wy), heading 0, seen from the ego's rig at
    every frame: returns (cx, cy, yaw, l, w, occ) in THAT frame's rig."""

    def __init__(self, poses, wx, wy, missing=()):
        self.poses, self.wx, self.wy, self.missing = poses, wx, wy, set(missing)

    def lookup(self, eid, f):
        if f in self.missing:
            return None
        x, y, h = (float(v) for v in self.poses[f][:3])
        dx, dy = self.wx - x, self.wy - y
        cx = math.cos(h) * dx + math.sin(h) * dy
        cy = -math.sin(h) * dx + math.cos(h) * dy
        return np.array([[cx, cy, -h, 4.5, 2.0, -1.0]])


def _ep(yaw0=0.0):
    t = torch.arange(200, dtype=torch.float32) * 0.1
    v = 10.0
    x = v * t * math.cos(yaw0)
    y = v * t * math.sin(yaw0)
    poses = torch.stack([x, y, torch.full_like(t, yaw0), torch.full_like(t, v)], 1)
    return SimpleNamespace(episode_id=7, poses=poses)


@pytest.mark.parametrize("yaw0", [0.0, math.pi / 2, -2.0])
def test_future_agents_land_where_the_world_puts_them(yaw0):
    ep = _ep(yaw0)
    f = 20
    # a static agent 50 m ahead of the ego at frame f, in world coordinates
    x0, y0 = float(ep.poses[f][0]), float(ep.poses[f][1])
    wx, wy = x0 + 50 * math.cos(yaw0), y0 + 50 * math.sin(yaw0)
    fake = SimpleNamespace(agent_join=_Join(ep.poses, wx, wy), agent_pad=4)
    out = T.V3Dataset._agent_future_item(fake, ep, f)
    box, valid = out["r7_agent_fut"], out["r7_agent_fut_valid"]
    assert bool(out["r7_agent_fut_label"].all())
    # static in the world => the SAME point in rig(f) at every slot: (50, 0)
    assert torch.allclose(box[:, 0, 0], torch.full((8,), 50.0), atol=1e-3)
    assert torch.allclose(box[:, 0, 1], torch.zeros(8), atol=1e-3)
    assert bool(valid[:, 0].all()) and not bool(valid[:, 1:].any())


def test_unlabelled_future_frames_are_marked_not_invented():
    ep = _ep()
    f = 20
    miss = {f + 30}                                    # slot 5 (h = 30)
    fake = SimpleNamespace(agent_join=_Join(ep.poses, 100.0, 0.0, miss), agent_pad=4)
    lab = T.V3Dataset._agent_future_item(fake, ep, f)["r7_agent_fut_label"]
    assert lab.tolist() == [True, True, True, True, False, True, True, True]


def test_slots_past_the_clip_end_are_unlabelled():
    ep = _ep()
    f = 195                                            # T = 200: only h <= 4 fits -> none
    fake = SimpleNamespace(agent_join=_Join(ep.poses, 100.0, 0.0), agent_pad=4)
    lab = T.V3Dataset._agent_future_item(fake, ep, f)["r7_agent_fut_label"]
    assert not bool(lab.any())


def test_RED_toad_without_a_trained_scorer_refuses():
    ok, msg, _ = _pin(_set(BASE + R7, "--w-r7-scorer", "0")
                      + ["--r7-no-select", "--r7-toad"])
    assert not ok and "reward IS the scorer" in msg


def test_GREEN_toad_with_the_scorer_passes_and_pins():
    ok, msg, cfg = _pin(BASE + R7 + ["--r7-toad", "--r7-toad-seed", "3"])
    assert ok, msg
    assert cfg.refcv7_toad is True and cfg.refcv7_toad_seed == 3
