"""
render_probe.py -- render a NuRec gaussian scene with gsplat and FALSIFY the result
against the reference render shipped in the scene folder.

The point of this script is not to produce a picture. It is to answer, with a number,
whether our decode of volume.nurec is CORRECT. A wrong decode that renders "something"
is the dangerous outcome, so every run reports PSNR / MAE against
camera_front_wide_120fov.mp4 and writes a side-by-side PNG.

Usage (on Thor, edge venv):
  python render_probe.py --scene-dir ~/nurec_work/x --ref <path to mp4> \
      --frame 0 --layers road,background --out out/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nurec_loader import (  # noqa: E402
    NuRecScene,
    RigTrajectories,
    K_for,
    ftheta_coeffs_for,
    read_volume_nurec,
    quat_layout_selftest,
    sample_cubemap,
)


# ---------------------------------------------------------------------------------
def time_basis(kind: str, tau: float, F: int) -> np.ndarray:
    """Candidate bases over the per-layer appearance ("fourier") features."""
    b = np.zeros(F, np.float32)
    if kind == "f0":
        b[0] = 1.0
    elif kind == "fourier_cs":  # [1, cos2pit, sin2pit, cos4pit, sin4pit, ...]
        b[0] = 1.0
        for k in range(1, (F - 1) // 2 + 1):
            if 2 * k - 1 < F:
                b[2 * k - 1] = np.cos(2 * np.pi * k * tau)
            if 2 * k < F:
                b[2 * k] = np.sin(2 * np.pi * k * tau)
    elif kind == "fourier_sc":  # [1, sin, cos, sin2, cos2, ...]
        b[0] = 1.0
        for k in range(1, (F - 1) // 2 + 1):
            if 2 * k - 1 < F:
                b[2 * k - 1] = np.sin(2 * np.pi * k * tau)
            if 2 * k < F:
                b[2 * k] = np.cos(2 * np.pi * k * tau)
    elif kind == "tent":  # piecewise-linear nodes at tau = i/(F-1)
        if F == 1:
            b[0] = 1.0
        else:
            x = tau * (F - 1)
            i = int(np.clip(np.floor(x), 0, F - 2))
            w = x - i
            b[i] = 1.0 - w
            b[i + 1] = w
    elif kind == "poly":  # [1, tau, tau^2, ...]
        for i in range(F):
            b[i] = tau**i
    else:
        raise ValueError(kind)
    return b


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def read_ref_frame(mp4: str, frame: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame} of {mp4} (n={n})")
    return img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]


# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="dir with volume.nurec + rig_trajectories.json")
    ap.add_argument("--ref", default=None, help="reference mp4 for the falsifier")
    ap.add_argument("--cam", default="camera_front_wide_120fov")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--frames", default=None, help="comma list; overrides --frame")
    ap.add_argument("--layers", default="background,road")
    ap.add_argument("--basis", default="f0", choices=["f0", "fourier_cs", "fourier_sc", "tent", "poly"])
    ap.add_argument("--quat-layout", default="auto", choices=["auto", "wxyz", "xyzw"])
    ap.add_argument("--sky", action="store_true", help="composite the sky env cubemap behind")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--opacity-floor", type=float, default=0.0)
    ap.add_argument("--rolling-shutter", action="store_true")
    ap.add_argument("--negative-control", action="store_true",
                    help="also score the render against WRONG reference frames")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "6")
    import torch
    from gsplat import rasterization

    sd = Path(args.scene_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    report = {"scene_dir": str(sd), "cam": args.cam, "layers": args.layers, "basis": args.basis}

    t0 = time.time()
    nre = read_volume_nurec(sd / "volume.nurec")
    report["nurec_version"] = nre["version"]
    print(f"[load] volume.nurec  version={nre['version']}  {time.time()-t0:.1f}s", flush=True)

    # ---- quaternion layout: decided from geometry, not from the reference image ----
    if args.quat_layout == "auto":
        probe = NuRecScene(nre)
        st = quat_layout_selftest(probe, "road", n_patches=120)
        layout = "wxyz" if st["winner"] == 1.0 else "xyzw"
        print(f"[quat] road self-test over {int(st['n_patches_used'])} patches:\n"
              f"       mean|dot| (near-degenerate): wxyz={st['absdot_wxyz']:.4f} xyzw={st['absdot_xyzw']:.4f}\n"
              f"       TILT correlation (decisive): wxyz={st['tiltcorr_wxyz']:+.4f} "
              f"(x {st['tiltcorr_wxyz_x']:+.3f}, y {st['tiltcorr_wxyz_y']:+.3f})  "
              f"xyzw={st['tiltcorr_xyzw']:+.4f} (x {st['tiltcorr_xyzw_x']:+.3f}, y {st['tiltcorr_xyzw_y']:+.3f})\n"
              f"       -> {layout}")
        report["quat_selftest"] = {k: v for k, v in st.items()}
    else:
        layout = args.quat_layout
    report["quat_layout"] = layout

    scene = NuRecScene(nre, quat_layout=layout)
    # the file declares every array's shape; verify each against its byte count under
    # fp16 -- this is the proof of the dtype, and it fails loudly if the file changes.
    for L in [x for x in args.layers.split(",") if x]:
        vs = scene.verify_shapes(L)
        print(f"[shape] {L}: " + "  ".join(f"{k}{v[0]}" for k, v in vs.items()))
    report["verified_shapes"] = {L: {k: list(v[0]) for k, v in scene.verify_shapes(L).items()}
                                 for L in [x for x in args.layers.split(",") if x]}
    rig = RigTrajectories(sd / "rig_trajectories.json")
    cam = rig.camera(args.cam)
    print(f"[cam ] {cam.name} {cam.width}x{cam.height} pp=({cam.cx:.2f},{cam.cy:.2f}) "
          f"ref_poly={cam.reference_poly} max_angle={cam.max_angle:.4f} shutter={cam.shutter_type}")

    t_lo = int(scene.sd[".gaussians_nodes.background.time_embed._extra_state"]["timestamps_us_min"])
    t_hi = int(scene.sd[".gaussians_nodes.background.time_embed._extra_state"]["timestamps_us_max"])

    frames = [int(x) for x in args.frames.split(",")] if args.frames else [args.frame]
    layers = [x for x in args.layers.split(",") if x]

    # ---- assemble gaussians (time basis depends on the frame; rebuild per frame) ----
    results = []
    for fi in frames:
        ts0, ts1 = rig.frame_timestamps_us(args.cam, fi)
        tau = (ts1 - t_lo) / float(t_hi - t_lo)
        means, quats, scales, opac, sh = [], [], [], [], []
        dropped = {}
        for L in layers:
            F = scene.fourier_dim(L)
            b = time_basis(args.basis, tau, F)
            g = scene.gaussians(L, time_basis=b, opacity_floor=args.opacity_floor)
            dropped[L] = g.n_dropped
            print(f"[gs  ] {L}: {g.means.shape[0]} gaussians (dropped {g.n_dropped}: {g.drop_reason})")
            means.append(g.means); quats.append(g.quats); scales.append(g.scales)
            opac.append(g.opacities); sh.append(g.sh)
        means = torch.from_numpy(np.concatenate(means)).to(dev)
        quats = torch.from_numpy(np.concatenate(quats)).to(dev)
        scales = torch.from_numpy(np.concatenate(scales)).to(dev)
        opac = torch.from_numpy(np.concatenate(opac)).to(dev)
        sh = torch.from_numpy(np.concatenate(sh)).to(dev)

        W, H = cam.width // args.downscale, cam.height // args.downscale
        K = K_for(cam).copy()
        if args.downscale != 1:
            K[0, 2] /= args.downscale
            K[1, 2] /= args.downscale
            K[0, 0] /= args.downscale
            K[1, 1] /= args.downscale
        Kt = torch.from_numpy(K[None]).to(dev)

        vm_end = torch.from_numpy(rig.viewmat(args.cam, fi, shutter=1)[None].astype(np.float32)).to(dev)
        kw = {}
        if args.rolling_shutter:
            from gsplat.cuda._wrapper import RollingShutterType

            vm_start = torch.from_numpy(rig.viewmat(args.cam, fi, shutter=0)[None].astype(np.float32)).to(dev)
            # viewmats = shutter start, viewmats_rs = shutter end
            kw = dict(rolling_shutter=RollingShutterType.TOP_TO_BOTTOM, viewmats_rs=vm_end)
            vm = vm_start
        else:
            vm = vm_end

        torch.cuda.synchronize()
        t1 = time.time()
        colors, alphas, meta = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=sh,
            viewmats=vm, Ks=Kt, width=W, height=H,
            sh_degree=3, packed=False, with_ut=True, with_eval3d=True,
            camera_model="ftheta", ftheta_coeffs=ftheta_coeffs_for(cam),
            near_plane=0.05, far_plane=2000.0, **kw,
        )
        torch.cuda.synchronize()
        dt = time.time() - t1
        img = colors[0].clamp(0, 1).cpu().numpy()
        alpha = alphas[0].cpu().numpy()
        print(f"[rend] frame {fi} tau={tau:.4f} {W}x{H} in {dt*1000:.0f} ms "
              f"({1/dt:.1f} FPS)  mean_alpha={alpha.mean():.4f}")

        # ---- sky ----
        if args.sky:
            cube = scene.sky_cubemap()
            if cube is not None:
                dirs = camera_ray_dirs(cam, W, H, args.downscale)
                c2n = rig.cam_to_nre(args.cam, fi, shutter=1)
                world_dirs = dirs @ c2n[:3, :3].T.astype(np.float32)
                sky = sample_cubemap(cube, world_dirs).astype(np.float32)
                img = img + (1.0 - alpha) * np.clip(sky, 0, 1)
                img = np.clip(img, 0, 1)

        res = {"frame": fi, "tau": tau, "render_ms": dt * 1000, "mean_alpha": float(alpha.mean()),
               "n_gaussians": int(means.shape[0]), "dropped": dropped}

        # ---- FALSIFIER ----
        if args.ref:
            ref = read_ref_frame(args.ref, fi)
            if ref.shape[:2] != img.shape[:2]:
                import cv2

                # the shipped reference mp4 is 3840x2160 = 2x the calibrated sensor
                # resolution (1920x1080); area-downsample it onto our grid.
                print(f"[FALS] resampling reference {ref.shape[1]}x{ref.shape[0]} -> {W}x{H}")
                res_note = f"ref_resampled_from_{ref.shape[1]}x{ref.shape[0]}"
                ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
                res["ref_note"] = res_note
            res["psnr"] = psnr(img, ref)
            res["mae"] = float(np.mean(np.abs(img - ref)))
            res["ref_mean"] = float(ref.mean())
            res["render_mean"] = float(img.mean())
            print(f"[FALS] frame {fi}: PSNR={res['psnr']:.2f} dB  MAE={res['mae']:.4f}  "
                  f"mean ref={res['ref_mean']:.3f} render={res['render_mean']:.3f}")

            # ---- DIAGNOSTIC, NOT A RESULT --------------------------------------
            # The scene ships a trained per-frame ISP (exposure / vignetting / colour
            # / CRF) that we do not reproduce. Fitting a 6-parameter per-channel
            # affine bounds how much of the residual is a pure radiometric transform
            # and how much is genuine geometric/appearance error. Always reported
            # separately from the raw PSNR above; never quote it as "the" PSNR.
            fit = np.empty_like(img)
            gains = []
            for c in range(3):
                x, y = img[..., c].ravel(), ref[..., c].ravel()
                A = np.stack([x, np.ones_like(x)], 1)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                fit[..., c] = (x * sol[0] + sol[1]).reshape(img.shape[:2])
                gains.append([float(sol[0]), float(sol[1])])
            fit = np.clip(fit, 0, 1)
            res["psnr_after_affine_colour_fit"] = psnr(fit, ref)
            res["affine_gain_bias"] = gains
            print(f"[DIAG] frame {fi}: PSNR after 6-param colour fit = "
                  f"{res['psnr_after_affine_colour_fit']:.2f} dB  (DIAGNOSTIC, not a result) "
                  f"gains={[round(g[0],3) for g in gains]}")

            # ---- NEGATIVE CONTROL ----------------------------------------------
            # Every frame of this clip is a dark night street, so a decent PSNR could
            # in principle come from "both images are dark" rather than from a correct
            # pose. Score our ONE render against the CORRECT reference frame and
            # against several WRONG ones. If the mapping is right, the correct frame
            # must win by a clear margin.
            if args.negative_control:
                import cv2

                lum = lambda a: a @ np.array([0.299, 0.587, 0.114], np.float32)

                def _ncc1(a, b):
                    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
                    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

                def ncc(a, b):
                    return _ncc1(lum(a), lum(b))

                def grad_ncc(a, b):
                    """NCC on image GRADIENTS. Raw-intensity NCC is a weak statistic on
                    this clip -- every frame is the same dark street, so the low
                    frequencies match everywhere. Gradients carry the actual geometry
                    (lamp posts, facade edges, lane markings), which is what a correct
                    pose must reproduce."""
                    ga, gb = [], []
                    for im in (a, b):
                        g = lum(im)
                        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                        (ga if im is a else gb).append(np.hypot(gx, gy))
                    return _ncc1(ga[0], gb[0])

                nc = {}
                cands = sorted({fi, max(0, fi - 60), min(598, fi + 60),
                                (fi + 150) % 599, (fi + 300) % 599, (fi + 450) % 599})
                for j in cands:
                    rj = read_ref_frame(args.ref, j)
                    if rj.shape[:2] != img.shape[:2]:
                        import cv2

                        rj = cv2.resize(rj, (W, H), interpolation=cv2.INTER_AREA)
                    nc[j] = {"psnr": psnr(img, rj), "ncc": ncc(img, rj),
                             "grad_ncc": grad_ncc(img, rj)}
                res["negative_control"] = nc
                best = max(nc, key=lambda j: nc[j]["grad_ncc"])
                res["negative_control_argmax_grad_ncc"] = int(best)
                res["negative_control_margin"] = (
                    nc[fi]["grad_ncc"] - max(v["grad_ncc"] for j, v in nc.items() if j != fi)
                )
                line = "  ".join(f"{'*' if j==fi else ' '}f{j}:g={nc[j]['grad_ncc']:+.3f}"
                                 for j in cands)
                print(f"[NCTL] render@{fi} gradNCC vs refs -> {line}   "
                      f"{'PASS' if best == fi else 'FAIL'} (argmax=f{best}, "
                      f"margin={res['negative_control_margin']:+.3f})")
            save_side_by_side(out / f"sbs_frame{fi:04d}_{args.basis}_{'-'.join(layers)}.png", img, ref, res)
        save_png(out / f"render_frame{fi:04d}_{args.basis}_{'-'.join(layers)}.png", img)
        results.append(res)

    report["results"] = results
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def camera_ray_dirs(cam, W: int, H: int, downscale: int) -> np.ndarray:
    """Unit ray directions in CAMERA optical coords for every pixel, using the file's
    own ftheta polynomial (bw poly: pixel distance -> angle)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    px = (xs + 0.5) * downscale - (cam.cx + 0.5)
    py = (ys + 0.5) * downscale - (cam.cy + 0.5)
    r = np.hypot(px, py)
    bw = np.array(cam.pixeldist_to_angle_poly, np.float64)
    theta = np.polyval(bw[::-1], r)
    theta = np.clip(theta, 0.0, cam.max_angle)
    s = np.where(r > 1e-9, np.sin(theta) / np.maximum(r, 1e-9), 0.0)
    d = np.stack([px * s, py * s, np.cos(theta)], -1)
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)


def save_png(path: Path, img: np.ndarray):
    import cv2

    cv2.imwrite(str(path), (np.clip(img, 0, 1)[:, :, ::-1] * 255).astype(np.uint8))


def save_side_by_side(path: Path, render: np.ndarray, ref: np.ndarray, res: dict):
    import cv2

    h, w = render.shape[:2]
    pad = np.ones((h, 8, 3), np.float32)
    diff = np.abs(render - ref).mean(-1, keepdims=True).repeat(3, -1)
    canvas = np.concatenate([render, pad, ref, pad, np.clip(diff * 3, 0, 1)], 1)
    canvas = (np.clip(canvas, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
    lab = f"OURS (gsplat)            REFERENCE (NuRec mp4)          |diff|x3   PSNR={res.get('psnr',float('nan')):.2f}dB"
    bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
    cv2.putText(bar, lab, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.concatenate([bar, canvas], 0))


if __name__ == "__main__":
    main()
