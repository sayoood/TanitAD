"""CPU tests — Stage 3 CURATION (``lake.curation``): inverse-frequency clamped
strata weights, weakness up-sampling, safety-event mining, and the frozen
hash-pinned per-tier eval holdout. Synthetic trajectories only.
"""

import math

import torch

from tanitad.lake import curation as CU
from tanitad.lake import goal_labels as GL
from tanitad.lake import vocab as V


def _roll(yaw_rate, speed, T, dt=0.1):
    rows, x, y, yaw = [], 0.0, 0.0, 0.0
    for t in range(T):
        v = float(speed[t])
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += float(yaw_rate[t]) * dt
    return torch.tensor(rows, dtype=torch.float32)


def _straight(v, T):
    return _roll(torch.zeros(T), torch.full((T,), float(v)), T)


# --------------------------------------------------------------------------- #
# 1. inverse-frequency clamped weights (the refb_train scheme)                 #
# --------------------------------------------------------------------------- #
def test_inverse_frequency_up_weights_rare_and_clamps():
    counts = {"maj": 90, "rare": 10}
    w = CU.inverse_frequency_weights(counts, clamp=10.0)
    assert w["rare"] > w["maj"]                        # rarity up-weighted
    assert w["maj"] < 1.0                              # majority pushed below 1
    # a singleton stratum in a big corpus would blow up without the clamp
    big = {"maj": 10_000, "singleton": 1}
    assert CU.inverse_frequency_weights(big, clamp=10.0)["singleton"] == 10.0


def test_stratum_key_scene_unknown_behavior_kinematic():
    g = GL.mint_kinematic_goal(_straight(30.0, 300), 0)
    key = CU.stratum_key(None, g)
    # first 4 entries are the (unknown) VLM scene axes; the rest the behavior tokens
    assert key[:4] == ("unknown", "unknown", "unknown", "unknown")
    assert g["LONMODE"]["token"] in key and g["VTARGET"]["token"] in key


# --------------------------------------------------------------------------- #
# 2. weakness strata                                                          #
# --------------------------------------------------------------------------- #
def test_weakness_high_speed_and_curves_and_stopgo():
    hi = GL.mint_kinematic_goal(_straight(35.0, 300), 0)
    assert "high_speed_longitudinal" in CU.weakness_strata(hi)
    # a tight sustained curve -> VSOURCE curve_constrained -> curves weakness
    T = 320
    curve = _roll(torch.full((T,), 8.0 / 40.0), torch.full((T,), 8.0), T)
    assert "curves" in CU.weakness_strata(GL.mint_kinematic_goal(curve, 40))
    # a plain mid-speed cruise is NOT a weakness stratum
    assert CU.weakness_strata(GL.mint_kinematic_goal(_straight(18.0, 300), 0)) == []


def test_episode_weakness_detects_contained_segment():
    # mostly cruise but with a stop-and-go segment in the middle
    sp = torch.cat([torch.full((60,), 8.0), torch.linspace(8, 0, 20),
                    torch.zeros(40), torch.linspace(0, 8, 20),
                    torch.full((160,), 8.0)])
    p = _roll(torch.zeros(len(sp)), sp, len(sp))
    assert "stop_and_go" in CU.episode_weakness(p)


# --------------------------------------------------------------------------- #
# 3. safety-event mining                                                      #
# --------------------------------------------------------------------------- #
def test_hard_brake_mined_kinematically():
    sp = torch.cat([torch.full((30,), 15.0), torch.linspace(15, 0, 15),
                    torch.zeros(35)])
    p = _roll(torch.zeros(len(sp)), sp, len(sp))
    assert CU.safety_event(p, {"DYN": {"token": "max"}}) == "hard_brake"
    # gentle cruise is not a safety event
    assert CU.safety_event(_straight(20.0, 80), {"DYN": {"token": "gentle"}}) is None


def test_vlm_safety_classes_dormant_without_lead_or_coc():
    p = _straight(20.0, 80)
    # near_miss needs a VLM lead ttc; anomaly needs a CoC physics_flag / notable
    assert CU.safety_event(p, {}, lead_state=None) is None
    assert CU.safety_event(p, {}, coc_trace={"physics_flag": "impossible"}) == "anomaly"


# --------------------------------------------------------------------------- #
# 4. frozen, hash-pinned per-tier eval holdout                                 #
# --------------------------------------------------------------------------- #
def test_holdout_is_deterministic_and_split_unit_granular():
    a = [CU.is_eval_holdout(f"route{i}", "ship") for i in range(200)]
    b = [CU.is_eval_holdout(f"route{i}", "ship") for i in range(200)]
    assert a == b                                      # frozen across calls
    # same split_unit -> same verdict regardless of how many windows it has
    assert CU.is_eval_holdout("routeX", "ship") == CU.is_eval_holdout("routeX", "ship")
    assert CU.is_eval_holdout("", "ship") is False     # no unit id -> never holdout
    frac = sum(a) / len(a)
    assert 0.03 < frac < 0.20                           # ~HOLDOUT_FRAC (0.1)


def test_holdout_independent_per_tier():
    ship = [CU.is_eval_holdout(f"r{i}", "ship") for i in range(200)]
    nc = [CU.is_eval_holdout(f"r{i}", "nc") for i in range(200)]
    assert ship != nc                                  # C-eval and R-eval differ


# --------------------------------------------------------------------------- #
# 5. corpus-level compose                                                      #
# --------------------------------------------------------------------------- #
def test_curate_corpus_down_weights_majority_up_weights_rare():
    recs = []
    for i in range(8):                                 # the straight-highway majority
        p = _straight(20.0, 300)
        recs.append({"id": i, "split_unit_id": f"r{i}", "tier": "ship",
                     "goal": GL.episode_goal_summary(p), "poses": p})
    ph = _straight(35.0, 300)                           # one rare high-speed clip
    recs.append({"id": 99, "split_unit_id": "r99", "tier": "ship",
                 "goal": GL.episode_goal_summary(ph), "poses": ph})
    verd = CU.curate_corpus(recs)
    assert verd[99].weight > verd[0].weight             # rare + weakness up-weighted
    assert "high_speed_longitudinal" in verd[99].weakness
    assert all(0.0 < verd[i].weight for i in range(8))
    d = verd[0].to_dict()
    assert set(d) == {"strata", "weight", "safety_event", "is_eval_holdout",
                      "weakness"}
