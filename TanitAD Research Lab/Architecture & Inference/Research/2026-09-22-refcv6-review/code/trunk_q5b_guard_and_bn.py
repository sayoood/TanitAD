"""Q5 corrections — two things the first pass got wrong or incomplete.

**(A) WHERE the model-level future guard is actually sensitive.**
`refc.py:4267-4269` calls `ego_channels_from_poses(ego_poses, n_past)` and then
`self.ego_hist(ch)` **with no `n_past`**. So the ONE load-bearing slice is
`ego_history.py:92` (`past = poses[:, :n]`); the encoder's own slice
(`ego_history.py:176`) is a no-op on an already-sliced `[B, n_past, 3]`.
Two mutations, both run against the REAL model-level assertion from
`test_refcv6_tactical.py:1243`:
  * break the ENCODER's slice  -> expected: guard stays GREEN (blind spot);
  * break the CHANNEL BUILDER's slice -> expected: guard goes RED.
A guard that goes red on neither is worse than none.

**(B) `fuse_identity_init`'s "BIT-IDENTICAL" claim and BatchNorm.**
`timm_trunk.py:300-306` states a fresh K-frame trunk emits *bit-identically*
what the single-frame trunk emits. The K-pass path folds K into the BATCH
(`timm_trunk.py:583-586`), so in TRAIN mode every BatchNorm sees a K-times
larger, differently-composed batch. The same module already knows this
mechanism for the chunking lever (`timm_trunk.py:178-182`: *"chunking alone
changes BN and MEASURED a -40 % shift in ga_trunk"*). Measured here in three
regimes: eval, train, train+frozen_bn.
"""
from __future__ import annotations

import json
import sys

import torch

OUT = {}


def _guard(m, cfg, patch=None):
    """Run the EXACT assertion of test_refcv6_tactical.py:1243, return the gap."""
    w = cfg.core.window
    n_past = max(1, w // 2)
    torch.manual_seed(7)
    frames = torch.randn(2, w, 1, 64, 64)
    nav = torch.zeros(2, dtype=torch.long)
    poses = torch.randn(2, w, 4)
    fut = poses.clone()
    fut[:, n_past:] = fut[:, n_past:] + 1e4
    with torch.no_grad():
        a = m(frames, nav_cmd=nav, v0=torch.zeros(2),
              ego_poses=poses, ego_n_past=n_past)["traj"]
        b = m(frames, nav_cmd=nav, v0=torch.zeros(2),
              ego_poses=fut, ego_n_past=n_past)["traj"]
    return {"bit_identical": bool(torch.equal(a, b)),
            "max_abs_diff": float((a - b).abs().max())}


def main() -> int:
    sys.path.insert(0, "D:/Projects/TanitAD/stack/tests")
    import importlib

    import tanitad.models.ego_history as eh
    from tanitad.refs import refc as R

    tac = importlib.import_module("test_refcv6_tactical")

    # ------------------------------------------------------------ (A) --- #
    res = {}
    m, cfg = tac._ego_hist_model()
    res["baseline"] = _guard(m, cfg)

    # -- mutation 1: the ENCODER forgets to slice --------------------------
    enc = m.core.ego_hist
    orig_fwd = enc.forward
    enc.forward = lambda seq, n_past=None, _o=orig_fwd: _o(seq, None)
    res["MUT1_encoder_ignores_n_past"] = _guard(m, cfg)
    enc.forward = orig_fwd

    # -- mutation 2: the CHANNEL BUILDER forgets to slice ------------------
    # ⛔ this is the real historical defect shape: a builder that hands the
    # whole window to a causal encoder. Patch the symbol the model imports.
    orig_ch = eh.ego_channels_from_poses

    def leaky_channels(poses, n_past, dt=0.1, _o=orig_ch):
        return _o(poses, poses.shape[1], dt)      # ⛔ n_past := the WHOLE window
    eh.ego_channels_from_poses = leaky_channels
    try:
        res["MUT2_channel_builder_ignores_n_past"] = _guard(m, cfg)
    except Exception as e:
        res["MUT2_channel_builder_ignores_n_past"] = {
            "RAISED": f"{type(e).__name__}: {e}"}
    finally:
        eh.ego_channels_from_poses = orig_ch

    # -- mutation 3: a CENTRED difference at the last past step ------------
    # the specific leak ego_history.py:39-42 names. n_past stays honest; only
    # the derivative reaches one step forward.
    def centred(poses, n_past, dt=0.1):
        n = int(n_past)
        past = poses[:, :n]
        v, yaw = past[..., 3], past[..., 2]
        dv = torch.zeros_like(v)
        dv[:, 1:] = (v[:, 1:] - v[:, :-1]) / float(dt)
        # ⛔ the leak: the LAST past step uses the FIRST future sample
        if poses.shape[1] > n:
            dv[:, -1] = (poses[:, n, 3] - v[:, -2]) / (2 * float(dt))
        dyaw = torch.zeros_like(yaw)
        dyaw[:, 1:] = eh._wrap_pi(yaw[:, 1:] - yaw[:, :-1]) / float(dt)
        return torch.stack([v, dv, dyaw], dim=-1)
    eh.ego_channels_from_poses = centred
    try:
        res["MUT3_centred_difference_at_last_past_step"] = _guard(m, cfg)
    except Exception as e:
        res["MUT3_centred_difference_at_last_past_step"] = {
            "RAISED": f"{type(e).__name__}: {e}"}
    finally:
        eh.ego_channels_from_poses = orig_ch

    res["GUARD_IS_ALIVE"] = (
        res["baseline"]["bit_identical"]
        and not res["MUT2_channel_builder_ignores_n_past"].get("bit_identical",
                                                               True))
    res["GUARD_BLIND_TO_ENCODER_REGRESSION"] = res[
        "MUT1_encoder_ignores_n_past"].get("bit_identical", False)
    res["call_site"] = ("refc.py:4267-4269 -- ego_channels_from_poses(poses, "
                        "n_past) then self.ego_hist(ch) with NO n_past")
    OUT["Q5A_model_level_guard_mutation_audit"] = res

    # ------------------------------------------------------------ (B) --- #
    from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig

    def pair(train: bool, frozen_bn: bool):
        k3 = TimmResNetTrunk(TimmTrunkConfig(
            model_name="resnet34.a1_in1k", frames=3, mode="shared",
            image_hw=(416, 1024), pretrained=True, frozen_bn=frozen_bn))
        k1 = TimmResNetTrunk(TimmTrunkConfig(
            model_name="resnet34.a1_in1k", frames=1, mode="shared",
            image_hw=(416, 1024), pretrained=True, frozen_bn=frozen_bn))
        k3.train(train)
        k1.train(train)
        torch.manual_seed(3)
        x = torch.rand(2, 9, 64, 128)
        with torch.no_grad():
            a16, a32, _ = k3.forward_features(x)
            b16, b32, _ = k1.forward_features(x[:, 6:9])      # NEWEST frame
        return {"s16_max_abs_diff": float((a16 - b16).abs().max()),
                "s32_max_abs_diff": float((a32 - b32).abs().max()),
                "BIT_IDENTICAL": bool(torch.equal(a16, b16)),
                "k3_bn_in_training_mode": int(k3.bn_training_count()),
                "k1_bn_in_training_mode": int(k1.bn_training_count())}

    OUT["Q5B_identity_init_bit_identity"] = {
        "eval_mode": pair(False, False),
        "TRAIN_mode": pair(True, False),
        "TRAIN_mode_frozen_bn": pair(True, True),
        "claim_in_code": ("timm_trunk.py:300-306 -- 'a freshly built K-frame "
                          "trunk emits BIT-IDENTICALLY what the single-frame "
                          "trunk emits'"),
        "mechanism": ("the K passes are folded into the BATCH "
                      "(timm_trunk.py:583-586), so train-mode BatchNorm "
                      "normalises over B*K images instead of B"),
    }
    # the sibling lever the module already documents for the SAME mechanism
    OUT["Q5B_known_sibling"] = ("timm_trunk.py:178-182 -- 'chunking alone "
                                "changes BN and MEASURED a -40 % shift in "
                                "ga_trunk on resnet34 ... With BN frozen, "
                                "chunked and unchunked agree to 7.2e-6'")

    json.dump(OUT, sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
