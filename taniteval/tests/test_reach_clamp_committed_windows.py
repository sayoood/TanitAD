"""FIX 2 — the reachability clamp, proved INERT on the committed windows.

The claim being pinned, MEASURED on ``taniteval/results/fan_refc-xl-30k.pt``
(REF-C-XL @30k, **881 canonical val windows / 40 episodes**, the emitted
256-candidate fan WITH its real per-candidate logits) and published in
``…/2026-07-27-percandidate-labels/raw/t1_clip_fansize.json``:

    candidates removed                       72.08 %
    windows with an EMPTY survivor set        0.00 %
    ADE-oracle survives                      100.00 %
    paired Δ ADE (episode-cluster, B=2000)   **exactly 0.0000**
    miss@2m                                  0.0159 -> 0.0159

⇒ the clamp is FREE. It deletes 72.08 % of the fan, moves the pick in **zero**
windows, and makes any per-candidate computation 3.58x cheaper.

⚠️ **"Δ = 0" is only evidence if the instrument CAN move the pick.** A mask that
never fires produces the identical number and means nothing — that is the C13
class, a guard that cannot fail. So this file also drives the SAME code with a
band tightened until it must bite, and asserts the pick then moves and the ADE
degrades. Both directions, or neither reading is admissible.
"""
import os

import numpy as np
import pytest
import torch

from taniteval import ci as _ci

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAN = os.path.join(_REPO, "taniteval", "results", "fan_refc-xl-30k.pt")

pytestmark = pytest.mark.skipif(not os.path.exists(FAN),
                                reason=f"committed fan dump missing: {FAN}")

SEL_ACCEL_MAX = 2.5          # flagship_v15.V15Config.sel_accel_max — NOT tuned
HORIZON_S = 2.0


@pytest.fixture(scope="module")
def dump():
    d = torch.load(FAN, map_location="cpu", weights_only=False)
    fan = d["fan"].double()                                  # [W, C, 4, 2]
    gt = d["gt"].double()                                    # [W, 4, 2]
    out = {"fan": fan, "gt": gt, "logit": d["logits"].double(),
           "v0": d["v0"].double(), "eid": [str(e) for e in d["eid"]],
           "sel": d["sel"], "fe": (fan - gt[:, None]).norm(dim=-1).mean(-1)}
    return out


def _pick(logit, mask, fallback):
    """argmax over the surviving candidates; an empty row keeps its whole fan."""
    m = logit.masked_fill(~mask, float("-inf"))
    dead = ~mask.any(dim=1)
    p = m.argmax(dim=1)
    p[dead] = fallback[dead]
    return p, dead


def _ade(fe, pick):
    return fe.gather(1, pick[:, None])[:, 0].numpy()


def test_fidelity_the_dumps_own_pick_is_the_argmax_of_its_logits(dump):
    """Reproduce a committed number before quoting a new one."""
    assert torch.equal(dump["logit"].argmax(dim=1), dump["sel"]), (
        "the cached fan's recorded pick must be the argmax of its cached "
        "logits, or nothing downstream of it is interpretable")
    assert dump["fan"].shape[0] == 881 and dump["fan"].shape[1] == 256
    assert len(set(dump["eid"])) == 40


def test_the_clamp_is_free_on_the_committed_windows(dump):
    """72.08 % of the fan deleted, oracle 100 % intact, paired Δ EXACTLY 0."""
    from tanitad.models.flagship_v15 import reachability_mask

    keep = reachability_mask(dump["fan"], dump["v0"],
                             accel_max=SEL_ACCEL_MAX, horizon_s=HORIZON_S)
    base_pick = dump["logit"].argmax(dim=1)
    clip_pick, dead = _pick(dump["logit"], keep, base_pick)

    frac_removed = float(1.0 - keep.double().mean())
    assert round(frac_removed, 4) == 0.7208, frac_removed
    assert float(dead.double().mean()) == 0.0, "no window may lose its whole fan"

    fe = dump["fe"]
    orc = fe.argmin(dim=1)
    assert float(keep.gather(1, orc[:, None])[:, 0].double().mean()) == 1.0, (
        "the ADE-oracle must survive the clamp in EVERY window — if it does "
        "not, the clamp is buying speed with ceiling")

    base_ade, clip_ade = _ade(fe, base_pick), _ade(fe, clip_pick)
    assert int((clip_pick != base_pick).sum()) == 0, "the pick must not move"
    paired = _ci.paired_episode_cluster_bootstrap(clip_ade, base_ade,
                                                  dump["eid"], n_boot=2000)
    assert paired["delta"] == 0.0 and paired["lo"] == 0.0 and paired["hi"] == 0.0
    assert paired["estimator"] == "paired_episode_cluster_bootstrap"
    assert paired["n_windows"] == 881 and paired["n_episodes"] == 40
    assert round(float(base_ade.mean()), 4) == 0.4714
    assert round(float((base_ade > 2.0).mean()), 4) == 0.0159
    assert round(float((clip_ade > 2.0).mean()), 4) == 0.0159
    # the whole point: 1 / (1 - 0.7208) = 3.58x fewer per-candidate rollouts
    assert 3.5 < 1.0 / (1.0 - frac_removed) < 3.7


def test_the_mask_CAN_move_the_pick_so_the_zero_is_evidence(dump):
    """DESIGNED TO FAIL: tighten the band until it must bite.

    Without this, `Δ = 0.0000` is indistinguishable from a mask that never
    fires. Here the identical code, given a band it cannot satisfy, moves the
    pick in many windows and makes the ADE strictly WORSE — which is what a
    live instrument looks like."""
    from tanitad.models.flagship_v15 import reachability_mask

    base_pick = dump["logit"].argmax(dim=1)
    fe = dump["fe"]
    base_ade = _ade(fe, base_pick)
    moved_any = False
    for accel_max in (0.5, 0.2, 0.05):
        keep = reachability_mask(dump["fan"], dump["v0"],
                                 accel_max=accel_max, horizon_s=HORIZON_S)
        pick, dead = _pick(dump["logit"], keep, base_pick)
        n_moved = int((pick != base_pick).sum())
        if n_moved:
            moved_any = True
            assert _ade(fe, pick).mean() > base_ade.mean(), (
                "a band tight enough to move the pick must also HURT — if it "
                "helped, the clamp would be a tuned selector, not physics")
    assert moved_any, (
        "the reachability mask never moved the pick at ANY band width, so the "
        "Δ = 0.0000 result would be vacuous — this is the C13 class")


def test_an_out_of_band_candidate_is_actually_present_in_the_fan(dump):
    """The clamp is not deleting nothing: the offset head really does emit
    unflyable candidates. Val GT max is 132.4 km/h; the fan goes far past it."""
    from tanitad.models.flagship_v15 import candidate_mean_speed

    v_mean = candidate_mean_speed(dump["fan"], HORIZON_S).numpy() * 3.6   # km/h
    gt_max = float((dump["gt"][:, -1].norm(dim=-1) / HORIZON_S).max()) * 3.6
    assert round(gt_max, 1) == 132.4, gt_max
    assert v_mean.max() > gt_max, "the fan must exceed the GT envelope"
    assert round(float(v_mean.max()), 1) == 171.5, float(v_mean.max())
    assert round(float(np.percentile(v_mean, 99)), 1) == 159.6

    # ...and the vocabulary is NOT the culprit: the anchors are real windows.
    # The excess lives in the OFFSET, which is what the clamp catches.
    assert v_mean.max() / gt_max > 1.25


def test_the_band_itself_is_conservative_not_tuned(dump):
    """reach = a_max * T is 2x the kinematic bound on a MEAN speed, so the
    72.08 % is a statement about the offset head, not about a tight band."""
    from tanitad.models.flagship_v15 import reachability_mask

    loose = reachability_mask(dump["fan"], dump["v0"], accel_max=SEL_ACCEL_MAX,
                              horizon_s=HORIZON_S)
    strict = reachability_mask(dump["fan"], dump["v0"],
                               accel_max=SEL_ACCEL_MAX / 2.0,
                               horizon_s=HORIZON_S)
    assert float(strict.double().mean()) < float(loose.double().mean())
    assert float(loose.double().mean()) == pytest.approx(0.2792, abs=1e-4)
