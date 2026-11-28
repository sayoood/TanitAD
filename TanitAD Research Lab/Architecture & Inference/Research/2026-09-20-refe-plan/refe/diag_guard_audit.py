"""CAN EACH GUARD ACTUALLY GO RED? A mutation audit over every instrument in the package.

⛔ WHY THIS EXISTS. Five reviews found guards that could not fail: assertions written as an
expression over the code under test, a probe that hardcoded the value it reported, a test that
STUBBED the method under test, three rotary assertions satisfied by ANY rotation, an allow-list
excusing an absence nobody re-read, and a regression arm that restated the forward pass and drifted
three times in one day. Each let a real defect through a green suite.

⭐ THE ONLY HONEST TEST OF A GUARD IS A MUTATION: reintroduce the defect it claims to catch and
require it to go RED. This file does that mechanically -- it patches a live object, runs the
instrument's own check, and reports whether the verdict flipped. An arm that stays green under its
own defect is named, not excused.

Usage:  python diag_guard_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import model as M
    results = []

    def arm(name, mutate, check):
        """mutate() breaks something; check() must return False (RED) when it is broken."""
        try:
            before = check()
        except Exception:
            before = False
        undo = mutate()
        try:
            after = check()
        except Exception:
            after = False          # an exception IS a red verdict
        undo()
        caught = bool(before) and not bool(after)
        results.append((name, before, after, caught))
        print(f"  [{'CATCHES' if caught else 'BLIND  '}] {name:52s} clean={before} mutated={after}")

    cfg = M.REFeConfig.for_backbone("vits16")
    cfg.img_h, cfg.img_w = 128, 256
    net = M.REFe(cfg).eval()
    B, N = 1, cfg.n_cameras
    img = torch.randn(B, N, 3, cfg.img_h, cfg.img_w)
    ego = torch.randn(B, cfg.ego_dim)
    goal = torch.randn(B, 2 * cfg.n_goal_points)

    # --- rope: the reference comparison must fail if the rotation changes -------------------
    def rope_check():
        c, s = M.build_axial_rope(8, 16, 64)
        return abs(float(c[:, 0].max()) - 1.0) < 1.0 and float(s.abs().max()) > 0

    def rope_mutate():
        orig = M.build_axial_rope
        M.build_axial_rope = lambda gh, gw, dh, device=None, dtype=torch.float32, base=100.0: (
            torch.ones(gh * gw, dh), torch.zeros(gh * gw, dh))      # identity rotation
        def undo(): M.build_axial_rope = orig
        return undo
    arm("rope: an identity rotation is detected", rope_mutate,
        lambda: float(M.build_axial_rope(8, 16, 64)[1].abs().max()) > 1e-6)

    # --- normalisation: removing it must change the features ------------------------------
    def norm_mutate():
        mean, std = net.backbone.img_mean.clone(), net.backbone.img_std.clone()
        net.backbone.img_mean.zero_(); net.backbone.img_std.fill_(1.0)
        def undo():
            net.backbone.img_mean.copy_(mean); net.backbone.img_std.copy_(std)
        return undo
    base_feat = net.backbone(img.reshape(B * N, *img.shape[2:])).detach().clone()
    arm("normalisation: removing mean/std changes the features", norm_mutate,
        lambda: torch.allclose(
            net.backbone(img.reshape(B * N, *img.shape[2:])), base_feat, atol=1e-5))

    # --- camera-count guard: a 1-camera batch into a 4-camera model must RAISE -------------
    def cam_mutate():
        net.allow_camera_mismatch = True
        def undo(): net.allow_camera_mismatch = False
        return undo

    def cam_check():
        try:
            net(img[:, :1], ego, goal)
            return False          # accepted -> the guard is off
        except ValueError:
            return True           # refused  -> the guard is on
    arm("camera guard: a 1-cam batch into a 4-cam model is refused", cam_mutate, cam_check)

    # --- scorer detach: removing it must let the score loss reach the trajectory ------------
    def detach_check():
        net.zero_grad(set_to_none=True)
        traj, score = net(img, ego, goal)
        score.sum().backward()
        g = net.queries.grad
        return g is None or float(g.abs().sum()) == 0.0

    def detach_mutate():
        orig = net.forward
        net.forward = lambda *a, **k: M.REFe.forward(net, *a, **{**k, "detach_scorer": False})
        def undo(): net.forward = orig
        return undo
    arm("scorer detach: re-attaching leaks into the trajectory head", detach_mutate, detach_check)

    # --- pos3d: it must be camera-CONDITIONED, not a shared table ---------------------------
    def pos_check():
        f = net._frustum(8, 16, "cpu", torch.float32)[0].reshape(cfg.n_cameras, 128, -1)
        return max(float((f[i] - f[j]).abs().max())
                   for i in range(cfg.n_cameras) for j in range(i + 1, cfg.n_cameras)) > 1e-6

    def pos_mutate():
        t, q = cfg.cam_t, cfg.cam_q
        cfg.cam_t = tuple(cfg.cam_t[0] for _ in cfg.cam_t)     # every camera at the same pose
        cfg.cam_q = tuple(cfg.cam_q[0] for _ in cfg.cam_q)
        net._pos3d_cache.clear()
        def undo():
            cfg.cam_t, cfg.cam_q = t, q
            net._pos3d_cache.clear()
        return undo
    arm("pos3d: identical extrinsics collapse it to a table", pos_mutate, pos_check)

    # --- LoRA: zeroing its scale must remove the gradient path ------------------------------
    def lora_check():
        net.zero_grad(set_to_none=True)
        traj, _ = net(img, ego, goal)
        traj.sum().backward()
        return sum(float(p.grad.abs().sum()) for n, p in net.named_parameters()
                   if n.startswith("backbone") and p.requires_grad and p.grad is not None) > 0

    def lora_mutate():
        mods = [m for m in net.backbone.modules() if hasattr(m, "scale")]
        old = [m.scale for m in mods]
        for m in mods:
            m.scale = 0.0
        def undo():
            for m, s in zip(mods, old):
                m.scale = s
        return undo
    arm("LoRA: a zeroed scale makes the adapters inert", lora_mutate, lora_check)

    caught = sum(1 for _, _, _, c in results if c)
    print(f"\n  {caught} of {len(results)} guards go RED under their own defect")
    print("\n" + ("GUARDS_CAN_FAIL" if caught == len(results) else "SOME_GUARDS_ARE_BLIND"))
    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
