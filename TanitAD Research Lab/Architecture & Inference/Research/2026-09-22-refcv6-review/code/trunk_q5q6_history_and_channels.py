"""Q5 (history inputs) and Q6 (channel counts), both by CONSTRUCTION.

⛔ A GREEN TEST IS NOT EVIDENCE. `stack/tests/test_refcv6_trunk.py:418` has a
paired deliberate-regression arm (`..._GOES_RED`) for the ENCODER-level future
read. The **model-level** guard —
`test_refcv6_tactical.py:1243 test_a_FUTURE_index_can_NEVER_enter_the_ego_history`
— has **no such arm**, and it is the one that guards the real wiring. This file
constructs that violation and confirms the assertion goes RED.

Q5 also measures, rather than reads:
  * K=3 runs the backbone K TIMES with ONE set of weights (counted at the stem);
  * the stem really sees **3** channels (not 9) on the default `shared` arm;
  * fusion happens AFTER the trunk, and `fuse_identity_init` makes a fresh
    K-frame trunk BIT-IDENTICAL to the single-frame trunk on the newest frame;
  * the `inflate` knockout reproduces the 3-channel response on repeated frames.

Q6 reads stride-16/32 channels from `timm.feature_info` for BOTH backbones and
cross-checks against an ACTUAL forward's tensor shapes — an independent
derivation, not a re-run of `feature_info`.
"""
from __future__ import annotations

import json
import sys

import torch

OUT = {}


def main() -> int:
    import timm

    from tanitad.models.timm_trunk import (TimmResNetTrunk, TimmTrunkConfig,
                                           _find_stem)
    from tanitad.models.trunk_shapes import FRAME_416x1024, TrunkSpec

    # ======================= Q6 — channel counts ======================== #
    q6 = {}
    for name in ("resnet34.a1_in1k", "resnet101.a1_in1k"):
        m = timm.create_model(name, pretrained=False, features_only=True,
                              out_indices=(3, 4))
        fi = {int(r): int(c) for r, c in zip(m.feature_info.reduction(),
                                             m.feature_info.channels())}
        with torch.no_grad():
            f16, f32 = m(torch.zeros(1, 3, 416, 1024))
        spec = TrunkSpec.from_timm(name, FRAME_416x1024)
        q6[name] = {
            "feature_info_channels": fi,
            "FORWARD_s16_shape": list(f16.shape),
            "FORWARD_s32_shape": list(f32.shape),
            "forward_agrees_with_feature_info":
                (int(f16.shape[1]), int(f32.shape[1])) == (fi[16], fi[32]),
            "TrunkSpec_perception": spec.perception.as_dict(),
            "TrunkSpec_planner": spec.planner.as_dict(),
            "spec_agrees_with_forward":
                (spec.perception.channels, tuple(spec.perception.hw)) ==
                (int(f16.shape[1]), tuple(f16.shape[2:])) and
                (spec.planner.channels, tuple(spec.planner.hw)) ==
                (int(f32.shape[1]), tuple(f32.shape[2:])),
            "source": spec.source,
        }
    OUT["Q6_channels"] = q6
    # SPEC §10.2's table, as LITERALS, compared against what was measured.
    OUT["Q6_vs_SPEC_10_2"] = {
        "resnet101_spec_1024_2048": (q6["resnet101.a1_in1k"]
                                     ["feature_info_channels"] == {16: 1024, 32: 2048}),
        "resnet34_spec_256_512": (q6["resnet34.a1_in1k"]
                                  ["feature_info_channels"] == {16: 256, 32: 512}),
    }

    # ================== Q5a — K passes, SHARED weights ================== #
    small = (64, 128)
    t = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=3, mode="shared",
        image_hw=(416, 1024), pretrained=True))
    calls = {"n": 0, "in_ch": set(), "batch": set()}
    _n, stem = _find_stem(t.net)

    def hook(_m, inp, _o):
        calls["n"] += 1
        calls["in_ch"].add(int(inp[0].shape[1]))
        calls["batch"].add(int(inp[0].shape[0]))
    h = stem.register_forward_hook(hook)
    x = torch.rand(2, 9, *small)
    with torch.no_grad():
        s16, s32, _p = t.forward_features(x)
    h.remove()
    OUT["Q5a_shared_weights"] = {
        "stem_forward_calls_per_trunk_forward": calls["n"],
        "stem_input_channels_seen": sorted(calls["in_ch"]),
        "stem_input_batch_seen": sorted(calls["batch"]),
        "note": ("ONE call whose batch is B*K == 6 for B=2, K=3: the K passes "
                 "are folded into the batch, through ONE set of weights, and "
                 "the stem input is 3 channels — NOT a 9-channel stem."),
        "n_distinct_stem_weight_tensors": len(
            {id(p) for p in stem.parameters()}),
        "fuse16_is_after_trunk": type(t.fuse16).__name__,
        "fuse32_is_after_trunk": type(t.fuse32).__name__,
        "fuse_kind": t.cfg.fuse,
    }

    # ---- fusion AFTER the trunk, and identity-initialised --------------- #
    t1 = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=1, mode="shared",
        image_hw=(416, 1024), pretrained=True))
    newest = x[:, 6:9]                               # OLDEST -> NEWEST order
    with torch.no_grad():
        a16, a32, _ = t.forward_features(x)
        b16, b32, _ = t1.forward_features(newest)
    OUT["Q5b_identity_init_makes_history_removable"] = {
        "s16_max_abs_diff_vs_single_frame": float((a16 - b16).abs().max()),
        "s32_max_abs_diff_vs_single_frame": float((a32 - b32).abs().max()),
        "BIT_IDENTICAL_s16": bool(torch.equal(a16, b16)),
        # CONTROL: mutate the OLDEST frame — with identity init it must NOT move
        # (weight sits on frame K-1), which is the claim being tested.
    }
    x_old = x.clone()
    x_old[:, 0:3] = 0.0
    with torch.no_grad():
        c16, _c32, _ = t.forward_features(x_old)
    OUT["Q5b_identity_init_makes_history_removable"][
        "zeroing_OLDEST_frame_moves_s16_by"] = float((a16 - c16).abs().max())
    x_new = x.clone()
    x_new[:, 6:9] = 0.0
    with torch.no_grad():
        d16, _d32, _ = t.forward_features(x_new)
    OUT["Q5b_identity_init_makes_history_removable"][
        "CONTROL_zeroing_NEWEST_frame_moves_s16_by"] = float(
            (a16 - d16).abs().max())

    # ---- the 9-channel inflate arm is the KNOCKOUT, not the default ----- #
    ti = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=3, mode="inflate",
        image_hw=(416, 1024), pretrained=True))
    # ⚠️ `_find_stem` looks for a 3-channel conv and CORRECTLY raises on an
    # already-inflated stem; read the conv directly instead.
    stem_i = next(m for m in ti.net.modules()
                  if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (7, 7))
    rep = torch.cat([newest, newest, newest], dim=1)
    with torch.no_grad():
        e16, _e32, _ = ti.forward_features(rep)
        f16b, _f32, _ = t1.forward_features(newest)
    OUT["Q5c_inflate_arm"] = {
        "inflate_stem_in_channels": int(stem_i.in_channels),
        "default_mode_in_TimmTrunkConfig": TimmTrunkConfig().mode,
        "shared_stem_in_channels": int(stem.in_channels),
        "repeated_frame_max_abs_diff_vs_3ch": float((e16 - f16b).abs().max()),
    }

    # =============== Q5d — the MODEL-LEVEL future-read guard ============= #
    # Reproduce test_refcv6_tactical.py:1243 EXACTLY, then break the wiring and
    # confirm the same assertion goes RED.
    sys.path.insert(0, "D:/Projects/TanitAD/stack/tests")
    import importlib
    tac = importlib.import_module("test_refcv6_tactical")
    res = {}
    try:
        m, cfg = tac._ego_hist_model()
        w = cfg.core.window
        n_past = max(1, w // 2)
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
        res["baseline_bit_identical"] = bool(torch.equal(a, b))
        res["baseline_max_abs_diff"] = float((a - b).abs().max())

        # ⛔ THE CONSTRUCTED REGRESSION: make the core's ego encoder ignore
        # n_past, exactly as a careless refactor would.
        enc = m.core.ego_hist
        orig = enc.forward

        def leaky(seq, n_past=None, _o=orig):
            return _o(seq, None)
        enc.forward = leaky
        with torch.no_grad():
            a2 = m(frames, nav_cmd=nav, v0=torch.zeros(2),
                   ego_poses=poses, ego_n_past=n_past)["traj"]
            b2 = m(frames, nav_cmd=nav, v0=torch.zeros(2),
                   ego_poses=fut, ego_n_past=n_past)["traj"]
        enc.forward = orig
        res["MUTATED_bit_identical"] = bool(torch.equal(a2, b2))
        res["MUTATED_max_abs_diff"] = float((a2 - b2).abs().max())
        res["GUARD_GOES_RED_ON_THE_REAL_WIRING"] = (
            res["baseline_bit_identical"] and not res["MUTATED_bit_identical"])
        # and the ego path is genuinely live (else the guard is vacuous)
        alt = poses.clone()
        alt[:, :n_past] += 3.0
        with torch.no_grad():
            c = m(frames, nav_cmd=nav, v0=torch.zeros(2),
                  ego_poses=alt, ego_n_past=n_past)["traj"]
        res["CONTROL_past_ego_moves_the_plan_by"] = float((a - c).abs().max())
    except Exception as e:                                  # pragma: no cover
        res["INCONCLUSIVE"] = f"{type(e).__name__}: {e}"
    OUT["Q5d_model_level_future_guard"] = res

    # ---- ego channels: are the differences BACKWARD? (analytic target) --- #
    from tanitad.models.ego_history import ego_channels_from_poses
    poses = torch.zeros(1, 6, 4)
    poses[0, :, 3] = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0, 99.0])
    poses[0, :, 2] = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0, 9.0])
    ch = ego_channels_from_poses(poses, n_past=4, dt=0.1)
    OUT["Q5e_backward_differences_ANALYTIC"] = {
        "shape": list(ch.shape),
        "speed": [round(float(v), 4) for v in ch[0, :, 0]],
        "accel": [round(float(v), 4) for v in ch[0, :, 1]],
        "accel_expected_backward": [0.0, 10.0, 20.0, 30.0],
        "accel_would_be_if_CENTRED_at_last_past": round(
            float((10.0 - 3.0) / 0.2), 4),
        "future_value_99_absent": bool(float(ch.abs().max()) < 99.0),
    }

    json.dump(OUT, sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
