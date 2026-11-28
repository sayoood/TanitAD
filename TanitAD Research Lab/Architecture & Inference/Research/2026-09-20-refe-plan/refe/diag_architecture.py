"""Proof that the three ARCHITECTURE defects from the 2026-09-20 conformance review are fixed.

Each check is written against the review's own MEASURED evidence, so a regression reproduces the
exact symptom it recorded rather than some paraphrase of it.

  D2  the registers did not compress: every decoder layer received `visual + 16` tokens with the
      IDENTICAL storage pointer, i.e. 1,936 tokens instead of 16, and the paper's scene-vs-visual
      asymmetry was absent.
  D6  the WTA distance averaged over points AND channels, so 1 radian cost what 1 metre cost. The
      review's counterexample: 0.00 m + 0.60 rad scored 0.2000 and 0.30 m + 0.00 rad scored 0.1000,
      so the old metric picked the SECOND. The winner flips.
  D7  the trunk carried a `trunc_normal_` frozen table with no checkpoint counterpart
      (305,045,504 - 303,079,424 = 1,966,080 = 1920 x 1024) while DINOv3 computes axial RoPE on the
      fly and stores none.

⭐ Every check has a CONTROL that must read the no-effect value, because a check that cannot go red
proves nothing. Run at ViT-S and a small geometry so this is a seconds-long CPU test.

Usage:  python diag_architecture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import model as M

    cfg = M.REFeConfig.for_backbone("vits16")
    cfg.img_h, cfg.img_w = 128, 256          # 8 x 16 = 128 patches, enough to tell 128 from 16
    net = M.REFe(cfg).eval()
    B = 2
    # ⛔ THIS WAS `randn(B, 3, ...)` -- FOUR-DIMENSIONAL, i.e. ONE CAMERA -- while :65 below already
    # computed `exp_scene = cfg.n_cameras * cfg.n_registers`. Half-migrated: the instrument died at
    # `model.py` forward with "configured for 4 cameras but received 1", exit 1, so every check in
    # this file (register compression, the two decoder contexts, the WTA metric) stopped running the
    # moment the rig went to four cameras. A dead instrument reports nothing and looks like silence.
    img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)
    ego = torch.randn(B, cfg.ego_dim)
    goal = torch.randn(B, 2 * cfg.n_goal_points)
    n_patch = (cfg.img_h // cfg.patch) * (cfg.img_w // cfg.patch)
    # the scoring decoder attends to the VISUAL tokens of the WHOLE RIG, so the expectation is
    # per-camera patches times cameras -- written out rather than read back off the model.
    n_visual = n_patch * cfg.n_cameras

    seen: dict = {}

    def hook(tag):
        def f(mod, args, kwargs, out):
            ctx = kwargs.get("ctx", args[1] if len(args) > 1 else None)
            if ctx is not None:
                seen.setdefault(tag, []).append((tuple(ctx.shape), ctx.data_ptr()))
        return f

    hs = [blk.register_forward_pre_hook(
        lambda m, a, kw, t=t: seen.setdefault(t, []).append(
            (tuple(a[1].shape), a[1].data_ptr())) or None, with_kwargs=True)
        for t, blks in (("dec", net.dec), ("score", net.score_dec)) for blk in blks]
    with torch.no_grad():
        traj, score = net(img, ego, goal)
    for h in hs:
        h.remove()

    dec_ctx = {s for s, _ in seen.get("dec", [])}
    sc_ctx = {s for s, _ in seen.get("score", [])}
    dec_ptr = {p for _, p in seen.get("dec", [])}
    sc_ptr = {p for _, p in seen.get("score", [])}
    exp_scene = cfg.n_cameras * cfg.n_registers

    print("--- D2: do the registers COMPRESS, and do the decoders read DIFFERENT tensors? ---")
    print(f"  patches per camera             : {n_patch}   x {cfg.n_cameras} cameras = {n_visual}")
    print(f"  trajectory decoder context     : {sorted(dec_ctx)}   (paper: {exp_scene} scene tokens)")
    print(f"  scoring decoder context        : {sorted(sc_ctx)}   (paper: the visual tokens)")
    print(f"  same storage pointer?          : {'YES  <- the old defect' if dec_ptr & sc_ptr else 'no'}")

    print("\n--- D7: is the pretrained position encoding back? ---")
    has_pos = any(n.endswith("backbone.pos") for n, _ in net.named_parameters())
    cos, sin = M.build_axial_rope(4, 4, cfg.width // cfg.heads)
    t = torch.randn(1, 1, 16, cfg.width // cfg.heads)
    rot = M.apply_rope(t.clone(), cos, sin, 0)
    moved = float((rot - t).abs().max())
    norm_kept = float((rot.norm(dim=-1) - t.norm(dim=-1)).abs().max())
    print(f"  backbone.pos parameter exists  : {has_pos}   (must be False)")
    print(f"  rope changes the vectors       : max |delta| {moved:.4f}   (must be > 0)")
    print(f"  rope PRESERVES the norm        : max |delta norm| {norm_kept:.2e}   (a rotation must)")

    print("\n--- D6: the review's counterexample, which used to flip the winner ---")
    tgt = torch.zeros(1, cfg.horizon_steps, 3)
    cand = torch.zeros(1, 2, cfg.horizon_steps, 3)
    cand[0, 0, :, 2] = 0.60          # perfect position, 0.60 rad yaw error
    cand[0, 1, :, 0] = 0.30          # 0.30 m position error, perfect yaw
    old = (cand - tgt.unsqueeze(1)).abs().mean(dim=(2, 3))
    _, idx = M.wta_loss(cand, tgt)
    print(f"  old metric                     : {old[0].tolist()} -> picks #{int(old[0].argmin())}")
    print(f"  new metric picks               : #{int(idx[0])}   (must be 0, the one that is "
          f"exactly on the path)")

    checks = {
        "D2 trajectory decoder sees exactly the scene tokens":
            dec_ctx == {(B, exp_scene, cfg.dec_width)},
        "D2 scoring decoder sees the VISUAL tokens of ALL cameras":
            sc_ctx == {(B, n_visual, cfg.dec_width)},
        "D2 the two contexts are DIFFERENT tensors": not (dec_ptr & sc_ptr),
        "D7 the random frozen pos table is gone": not has_pos,
        "D7 CONTROL rope actually rotates (non-zero delta)": moved > 1e-6,
        "D7 CONTROL rope preserves the norm (it is a rotation)": norm_kept < 1e-4,
        "D6 the winner is the proposal on the path": int(idx[0]) == 0,
        "D6 CONTROL the OLD metric picked the other one": int(old[0].argmin()) == 1,
        "shapes still correct": traj.shape == (B, cfg.n_proposals, cfg.horizon_steps, 3)
            and score.shape == (B, cfg.n_proposals, cfg.n_score_components),
    }
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    good = all(checks.values())
    print("\n" + ("ARCHITECTURE_CONFORMS" if good else "ARCHITECTURE_DEFECTIVE"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
