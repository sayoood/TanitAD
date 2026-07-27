"""comma2k19 heading regime — the DEFAULT FLIP, its guard, and the pin that
keeps the deployed (legacy) path reproducible.

WHY THIS FILE EXISTS
--------------------
The standstill repair shipped OPT-IN on 2026-07-27 so that no existing cache
would move. The consequence was that **the defect became the default**: every
new comma build got a heading that is ``arctan2`` of the ENU velocity and is
UNDEFINED AT STANDSTILL unless the caller opted in. MEASURED on the 64-segment
val build (``…/2026-07-27-idm-v3/results/labels_v3.json → yaw_audit_by_speed``):
**26.27 % of frames below 0.5 m/s carry a physically impossible |yaw_rate|**
(max 15.53 rad/s = 890 deg/s); **0.000 % in every bin above 0.5 m/s**;
PhysicalAI has **zero in every bin**. Repairing the label moved the DEPLOYED
head's comma yaw R² **+0.0114 → +0.3308 with nothing retrained**.

⭐ **THE FAILING DIRECTION IS DEMONSTRATED HERE, not asserted.** A guard that
cannot fire is worse than none (RETRACTION_LOG class C13). Every refusal below
is exercised on the input that should trigger it, and
``test_the_defect_is_reproduced_by_the_legacy_path`` builds the physically
impossible yaw rate from a standstill fixture so the repair is measured against
a defect that is actually present, not against a fixture where both modes agree.

THE TWO INVARIANTS
------------------
1. **A caller cannot build a new comma cache on broken labels without knowing.**
   The default is the repair; ``None``/missing-config resolves to the repair;
   legacy needs ``allow_legacy=True`` AND a written reason.
2. **No existing cache silently changes meaning.** ``label_params`` contributes
   NO key for legacy (so every pre-2026-07-27 comma cache dir keeps its exact
   name) and a key for the repair (so a repaired build cannot land in one), and
   the legacy label itself is pinned bit-identical against an independent
   reimplementation of the pre-flip formula.
"""

import numpy as np
import pytest
import torch

from tanitad.data.comma2k19 import (DEFAULT_HEADING_MODE, HEADING_MODE_HOLD,
                                    HEADING_MODE_LEGACY, HEADING_MODES,
                                    HEADING_OBSERVABLE_V_MPS,
                                    LEGACY_HEADING_REASON, Comma2k19Dataset,
                                    LegacyHeadingRefused, actions_and_poses,
                                    build_episode, cache_build_params,
                                    ecef_to_enu, label_params,
                                    resolve_heading_mode)
from tanitad.data.epcache import cache_key

# MEASURED threshold above which a comma yaw rate is physically implausible for
# a road vehicle (IDM v3's audit uses |ω| > 1.5 rad/s = 86 deg/s).
IMPOSSIBLE_YAW_RATE = 1.5

REF_ECEF = np.array([4278000.0, 635000.0, 4672000.0])   # Zurich-ish
EAST = np.array([-0.147, 0.989, 0.0])                   # ~unit east in ECEF
UP = np.array([0.673, 0.100, 0.733])                    # ~unit up in ECEF
N_STILL, N_MOVING, DT = 6, 24, 0.05                     # 20 fps


def standstill_then_drive():
    """frame_times / ECEF positions / ECEF velocities for a segment that is
    STATIONARY (jittering below 0.05 m/s in random directions) and then drives
    east at 10 m/s — the real shape of the defect, which is why comma's heading
    goes wild exactly where the vehicle is not moving."""
    n = N_STILL + N_MOVING
    ft = np.arange(n) * DT
    rng = np.random.default_rng(0)
    vel = np.zeros((n, 3))
    # standstill: |v| ~ 0.01 m/s pointing anywhere (GNSS noise, not motion)
    d = rng.normal(size=(N_STILL, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    vel[:N_STILL] = 0.01 * d
    vel[N_STILL:] = 10.0 * EAST
    pos = REF_ECEF[None] + np.cumsum(vel * DT, axis=0)
    return ft, pos, vel


def _yaw_rate(poses: np.ndarray, dt: float) -> np.ndarray:
    """Wrapped d(yaw)/dt — the quantity the IDM head is scored on."""
    dy = np.diff(poses[:, 2].astype(np.float64))
    return np.arctan2(np.sin(dy), np.cos(dy)) / dt


def _legacy_poses_reference(ft, pos, vel, stride):
    """The PRE-FLIP default, REIMPLEMENTED from its specification rather than
    called — so the reproducibility pin cannot be satisfied by the same bug on
    both sides. Heading = arctan2 of the ENU velocity, no standstill handling.
    """
    enu_p, enu_v = ecef_to_enu(pos[::stride], vel[::stride])
    yaw = np.arctan2(enu_v[:, 1], enu_v[:, 0])
    v = np.linalg.norm(enu_v, axis=1)
    return np.column_stack([enu_p[:, 0], enu_p[:, 1], yaw, v]).astype(np.float32)


def _can(ft):
    return (ft, np.full(ft.size, 10.0)), (ft, np.full(ft.size, 15.3))


# --------------------------------------------------------------------------- #
# 1. THE DEFAULT — and the defect it now avoids, DEMONSTRATED                   #
# --------------------------------------------------------------------------- #

def test_default_is_the_repair():
    assert DEFAULT_HEADING_MODE == HEADING_MODE_HOLD
    assert HEADING_MODE_LEGACY in HEADING_MODES, "legacy stays REACHABLE"


def test_the_defect_is_reproduced_by_the_legacy_path():
    """⭐ The failing direction. Without this the whole regime is unfalsifiable.

    Legacy heading on a standstill segment must produce a PHYSICALLY IMPOSSIBLE
    yaw rate; the repaired one must not.
    """
    ft, pos, vel = standstill_then_drive()
    speed, steer = _can(ft)
    _, p_leg = actions_and_poses(ft, pos, vel, speed, steer, stride=1,
                                 heading_mode=HEADING_MODE_LEGACY,
                                 allow_legacy_heading=True,
                                 legacy_heading_reason=LEGACY_HEADING_REASON)
    _, p_def = actions_and_poses(ft, pos, vel, speed, steer, stride=1)

    yr_leg, yr_def = _yaw_rate(p_leg, DT), _yaw_rate(p_def, DT)
    still = slice(0, N_STILL - 1)
    # the defect, reproduced: a stationary vehicle "rotating" far past any
    # physical rate (the corpus's measured max is 15.53 rad/s = 890 deg/s)
    assert np.abs(yr_leg[still]).max() > IMPOSSIBLE_YAW_RATE
    # ... and gone under the default
    assert np.abs(yr_def[still]).max() < 1e-6
    # the MOVING part is untouched by the repair — it repairs, it does not smooth
    assert np.allclose(p_leg[N_STILL:, 2], p_def[N_STILL:, 2], atol=1e-12)
    # and the speed channel never moves at all
    assert np.array_equal(p_leg[:, 3], p_def[:, 3])


def test_the_repair_only_touches_below_the_measured_threshold():
    """The threshold is MEASURED (0.000 % impossible above 0.5 m/s), so the
    repair must not reach above it."""
    ft, pos, vel = standstill_then_drive()
    speed, steer = _can(ft)
    _, p_def = actions_and_poses(ft, pos, vel, speed, steer, stride=1)
    _, p_leg = actions_and_poses(ft, pos, vel, speed, steer, stride=1,
                                 heading_mode=HEADING_MODE_LEGACY,
                                 allow_legacy_heading=True,
                                 legacy_heading_reason="test")
    observable = p_leg[:, 3] >= HEADING_OBSERVABLE_V_MPS
    assert observable.any() and not observable.all()      # fixture spans both
    assert np.array_equal(p_leg[observable, 2], p_def[observable, 2])
    assert not np.array_equal(p_leg[~observable, 2], p_def[~observable, 2])


# --------------------------------------------------------------------------- #
# 2. THE GUARD — every refusal exercised on the input that must trigger it      #
# --------------------------------------------------------------------------- #

def test_legacy_is_refused_without_an_acknowledgement():
    with pytest.raises(LegacyHeadingRefused, match="UNDEFINED"):
        resolve_heading_mode(HEADING_MODE_LEGACY)


@pytest.mark.parametrize("allow,reason", [(True, ""), (True, "   "),
                                          (False, "I want the old labels")])
def test_half_an_acknowledgement_is_still_refused(allow, reason):
    """A boolean can be flipped absent-mindedly and a sentence can be written
    without the flag — neither alone is an acknowledgement."""
    with pytest.raises(LegacyHeadingRefused):
        resolve_heading_mode(HEADING_MODE_LEGACY, allow_legacy=allow,
                             reason=reason)


def test_a_full_acknowledgement_returns_legacy():
    assert resolve_heading_mode(HEADING_MODE_LEGACY, allow_legacy=True,
                                reason=LEGACY_HEADING_REASON) \
        == HEADING_MODE_LEGACY


@pytest.mark.parametrize("accident", [None, HEADING_MODE_HOLD])
def test_accident_modes_land_on_the_repair(accident):
    """A missing config key, an unset variable, a ``.get`` that found nothing —
    every silent path resolves to the CORRECT label, never the broken one."""
    assert resolve_heading_mode(accident) in (DEFAULT_HEADING_MODE,
                                              HEADING_MODE_HOLD)


def test_unknown_mode_is_refused_as_a_plain_ValueError():
    with pytest.raises(ValueError, match="unknown heading_mode"):
        resolve_heading_mode("enu_velocity_v2")
    # ... and a typo is NOT quietly treated as legacy
    with pytest.raises(ValueError):
        resolve_heading_mode("enu_velocty")


def test_actions_and_poses_refuses_legacy():
    ft, pos, vel = standstill_then_drive()
    speed, steer = _can(ft)
    with pytest.raises(LegacyHeadingRefused):
        actions_and_poses(ft, pos, vel, speed, steer, stride=1,
                          heading_mode=HEADING_MODE_LEGACY)


def test_build_episode_refuses_legacy(tmp_path):
    from tests.test_comma2k19 import fake_decode, make_fake_segment
    seg = make_fake_segment(tmp_path, "d0_2018-01-01--10-00-00", "3")
    with pytest.raises(LegacyHeadingRefused):
        build_episode(seg, size=32, max_steps=10, decode_fn=fake_decode,
                      heading_mode=HEADING_MODE_LEGACY)


def test_dataset_refuses_legacy_BEFORE_decoding_anything(tmp_path):
    """The dataset resolves in its first millisecond. A refusal that arrives
    after 40 minutes of video decode is a refusal nobody waits for."""
    from tests.test_comma2k19 import make_fake_segment
    seg = make_fake_segment(tmp_path, "d0_2018-01-01--10-00-00", "3")
    calls = []

    def spy_decode(*a, **k):
        calls.append(a)
        return torch.zeros((10, 3, 32, 32), dtype=torch.uint8)

    with pytest.raises(LegacyHeadingRefused):
        Comma2k19Dataset([seg], size=32, max_steps=8, decode_fn=spy_decode,
                         heading_mode=HEADING_MODE_LEGACY)
    assert calls == [], "refused only AFTER decoding — the guard is too late"


# --------------------------------------------------------------------------- #
# 3. THE PIN — the deployed (legacy) path still reproduces BIT-IDENTICALLY      #
# --------------------------------------------------------------------------- #

def test_legacy_with_acknowledgement_is_bit_identical_to_the_preflip_default():
    """⭐ Committed comma results were measured on LEGACY. This asserts the
    acknowledged legacy path equals an INDEPENDENT reimplementation of the
    pre-flip formula, exactly — not approximately."""
    ft, pos, vel = standstill_then_drive()
    speed, steer = _can(ft)
    for stride in (1, 2):
        _, p_leg = actions_and_poses(ft, pos, vel, speed, steer, stride=stride,
                                     heading_mode=HEADING_MODE_LEGACY,
                                     allow_legacy_heading=True,
                                     legacy_heading_reason=LEGACY_HEADING_REASON)
        ref = _legacy_poses_reference(ft, pos, vel, stride)
        assert np.array_equal(p_leg, ref), f"legacy moved at stride={stride}"


def test_build_episode_legacy_path_reproduces_bit_identically(tmp_path):
    from tests.test_comma2k19 import fake_decode, make_fake_segment
    seg = make_fake_segment(tmp_path, "d0_2018-01-01--10-00-00", "3")
    ep = build_episode(seg, size=32, max_steps=10, decode_fn=fake_decode,
                       heading_mode=HEADING_MODE_LEGACY,
                       allow_legacy_heading=True,
                       legacy_heading_reason=LEGACY_HEADING_REASON)
    ft = np.load(seg / "global_pose" / "frame_times")
    pos = np.load(seg / "global_pose" / "frame_positions")
    vel = np.load(seg / "global_pose" / "frame_velocities")
    ref = _legacy_poses_reference(ft, pos, vel, 2)[2:ep.poses.shape[0] + 2]
    assert np.array_equal(ep.poses.numpy(), ref)


# --------------------------------------------------------------------------- #
# 4. THE CACHE KEY — no existing dir changes meaning, no new dir collides       #
# --------------------------------------------------------------------------- #

# The exact params dict the comma cache is minted with, copied from
# train_worldmodel._build_datasets as of 2026-07-27 (canonical frame -> the
# geometry fragment is empty).
COMMA_PARAMS = {"size": 256, "n_stack": 3, "stride": 2, "max_steps": 300}
SRCS = [f"Chunk_1/route{i}/seg{i}" for i in range(50)]


def test_legacy_contributes_no_build_param():
    """Every comma cache dir minted before 2026-07-27 keeps its EXACT name."""
    assert label_params(HEADING_MODE_LEGACY) == {}
    merged = {**COMMA_PARAMS, **label_params(HEADING_MODE_LEGACY)}
    assert merged == COMMA_PARAMS
    assert cache_key(SRCS, merged) == cache_key(SRCS, COMMA_PARAMS)


def test_the_repaired_regime_changes_the_key():
    """⛔ The failing direction for the cache: a repaired build must be
    STRUCTURALLY unable to land in — or overwrite — a legacy-keyed dir."""
    legacy = {**COMMA_PARAMS, **label_params(HEADING_MODE_LEGACY)}
    repaired = {**COMMA_PARAMS, **label_params(HEADING_MODE_HOLD)}
    assert repaired != legacy
    assert cache_key(SRCS, repaired) != cache_key(SRCS, legacy)
    assert label_params() == {"heading_mode": HEADING_MODE_HOLD}, \
        "the DEFAULT must carry a key, or a repaired build reuses a legacy dir"


def test_label_params_refuses_an_unknown_regime():
    with pytest.raises(ValueError):
        label_params("enu_velocity_hold_v2")


def test_cache_build_params_refuses_legacy_without_a_reason():
    """This is the call the trainer makes. If it could be satisfied silently,
    the whole regime would be decorative."""
    with pytest.raises(LegacyHeadingRefused):
        cache_build_params(COMMA_PARAMS, HEADING_MODE_LEGACY)
    with pytest.raises(LegacyHeadingRefused):
        cache_build_params(COMMA_PARAMS, HEADING_MODE_LEGACY, allow_legacy=True)


def test_cache_build_params_round_trip():
    assert cache_build_params(COMMA_PARAMS) == {
        **COMMA_PARAMS, "heading_mode": HEADING_MODE_HOLD}
    assert cache_build_params(COMMA_PARAMS, None) \
        == cache_build_params(COMMA_PARAMS)
    assert cache_build_params(COMMA_PARAMS, HEADING_MODE_LEGACY,
                              allow_legacy=True,
                              reason=LEGACY_HEADING_REASON) == COMMA_PARAMS


# --------------------------------------------------------------------------- #
# 5. THE TRAINER — wired end-to-end, because forgetting to wire it is the       #
#    failure that makes a repaired build silently reuse a legacy cache dir      #
# --------------------------------------------------------------------------- #

@pytest.fixture
def comma_root(tmp_path):
    """Two routes (route-level split needs >= 2) of short segments."""
    from tests.test_comma2k19 import make_fake_segment
    for i in range(4):
        make_fake_segment(tmp_path, f"d{i % 2}_2018-01-0{i + 1}--10-00-00",
                          str(i))
    return tmp_path


def _run_trainer_dataset_build(root, monkeypatch, **kw):
    """Drive train_worldmodel's REAL comma branch with a mocked video decode and
    report which cache dirs it created."""
    from tanitad.config import base250cam_config
    from tanitad.data import comma2k19 as C
    from tanitad.train import train_worldmodel as TW

    real_build = build_episode          # the module-level import, never patched

    def light_build(seg, **kwargs):
        kwargs.pop("size", None)
        return real_build(seg, size=32, max_steps=6,
                          decode_fn=lambda s, st, sz, mf, **_: torch.zeros(
                              (min(mf or 8, 8), 3, 32, 32), dtype=torch.uint8),
                          **{k: v for k, v in kwargs.items() if k != "frame"})

    monkeypatch.setattr(C, "build_episode", light_build)
    TW._build_datasets(base250cam_config(), 4, "comma2k19", str(root), **kw)
    return sorted(p.name for p in (root / "_epcache").iterdir() if p.is_dir())


def test_trainer_default_build_does_NOT_reuse_the_legacy_cache_dir(
        comma_root, monkeypatch):
    """⭐ The end-to-end failing direction, on the REAL trainer branch. Build
    with the default (repair) and then with acknowledged legacy against the SAME
    root: the two must occupy DIFFERENT dirs, and the legacy dirs must carry the
    unchanged pre-flip key so no existing cache is orphaned or overwritten."""
    dirs_default = _run_trainer_dataset_build(comma_root, monkeypatch)
    dirs_both = _run_trainer_dataset_build(
        comma_root, monkeypatch, comma_heading_mode=HEADING_MODE_LEGACY,
        comma_legacy_heading_reason=LEGACY_HEADING_REASON)
    new = set(dirs_both) - set(dirs_default)
    assert new, "legacy landed in the SAME dirs as the repair — caches collide"
    assert len(dirs_both) == 2 * len(dirs_default)        # train+val, both modes

    # ... and the legacy dirs are keyed EXACTLY as they were before the flip:
    # the params dict with NO label fragment, over the trainer's own selection.
    from tanitad.config import base250cam_config
    from tanitad.data.comma2k19 import (discover_segments,
                                        sample_segments_across_routes,
                                        split_by_route)
    from tanitad.geometry import build_params
    cfg = base250cam_config()
    base = build_params(cfg, {"size": 256, "n_stack": 3, "stride": 2,
                              "max_steps": 300})
    segs = sample_segments_across_routes(discover_segments(comma_root), 4,
                                         seed=cfg.train.seed)
    tr, va = split_by_route(segs, val_frac=0.2, seed=cfg.train.seed)
    preflip = {f"comma2k19-train-{cache_key(tr, base)}",
               f"comma2k19-val-{cache_key(va, base)}"}
    assert new == preflip, \
        "the legacy dir is NOT the pre-flip key — an existing cache is orphaned"


def test_trainer_refuses_legacy_without_a_written_reason(comma_root,
                                                         monkeypatch):
    with pytest.raises(LegacyHeadingRefused):
        _run_trainer_dataset_build(
            comma_root, monkeypatch, comma_heading_mode=HEADING_MODE_LEGACY)
