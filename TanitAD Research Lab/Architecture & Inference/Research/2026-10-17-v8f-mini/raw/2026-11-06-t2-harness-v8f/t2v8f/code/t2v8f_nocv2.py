"""OpenCV-free replacements for the two harness helpers that need `cv2`, so the `objects` condition (and its actor falsifier) runs on box 2 where `cv2` is not installed.

WHICH HARNESS PATHS NEED cv2 (grep of the exported kit, MEASURED):
  * `closedloop_drive.main --condition objects` -> `actor_map.attach_actors_verified` -> `actor_map.falsify_actors` -> `gsplat_renderer.read_ref_frame` (cv2.VideoCapture, cv2.resize)
    and `gsplat_renderer.grad_ncc` (cv2.cvtColor, cv2.Sobel).   <- this is the only path a T2 run takes
  * `closedloop_drive.main --save-video-frames` (cv2.imwrite)    <- never used by this package
  * frame_align / overlay_video / make_before_after / openloop_drive                                     <- never called
`install()` replaces the first two with numpy / PIL / PyAV versions. The reference frame comes from, in order: the pre-extracted `ref_frames.npz` beside the mp4
(decoded on the dev box from the same mp4, hashes in the package manifest), then PyAV, then imageio, then cv2 if it exists. No decoder and no pre-extracted frame -> a named error.
`grad_ncc` is a re-statement of the cv2 formula (RGB2GRAY integer weights, 3x3 Sobel, reflect-101 border); cv2 is absent on both boxes, so it is not compared against cv2.
Every use of it in the falsifier compares two values of the SAME function (on vs wrong-time), so a constant difference from cv2 cancels.
"""
from __future__ import annotations

import os
import sys

import numpy as np


def gray_u8(rgb: np.ndarray) -> np.ndarray:
    """cv2.COLOR_RGB2GRAY on uint8: (4899 R + 9617 G + 1868 B + 8192) >> 14."""
    a = rgb.astype(np.int32)
    return ((4899 * a[..., 0] + 9617 * a[..., 1] + 1868 * a[..., 2] + 8192) >> 14).astype(np.uint8)


def sobel3(g: np.ndarray, axis: int) -> np.ndarray:
    """cv2.Sobel(g, CV_32F, dx, dy, ksize=3) with the default BORDER_REFLECT_101 (numpy 'reflect'). axis=1: d/dx, axis=0: d/dy."""
    p = np.pad(g.astype(np.float32), 1, mode="reflect")
    if axis == 1:
        d = p[:, 2:] - p[:, :-2]
        return (d[:-2] + 2.0 * d[1:-1] + d[2:]).astype(np.float32)
    d = p[2:, :] - p[:-2, :]
    return (d[:, :-2] + 2.0 * d[:, 1:-1] + d[:, 2:]).astype(np.float32)


def grad_ncc(a: np.ndarray, b: np.ndarray) -> float:
    la, lb = gray_u8(a).astype(np.float32), gray_u8(b).astype(np.float32)
    ga = np.hypot(sobel3(la, 1), sobel3(la, 0)).ravel()
    gb = np.hypot(sobel3(lb, 1), sobel3(lb, 0)).ravel()
    ga, gb = ga - ga.mean(), gb - gb.mean()
    return float(ga @ gb / max(np.linalg.norm(ga) * np.linalg.norm(gb), 1e-12))


def resize_area(img: np.ndarray, size_wh) -> np.ndarray:
    """Box-filter resize (PIL), the closest numpy-side match of cv2.INTER_AREA; only used when the mp4 and the renderer disagree on the frame size."""
    from PIL import Image
    return np.ascontiguousarray(np.asarray(Image.fromarray(img).resize(tuple(size_wh), Image.BOX)))


def decoders() -> dict:
    """Which mp4 decoders import here (name -> version)."""
    out = {}
    for m in ("av", "imageio", "cv2"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:                                                          # noqa: BLE001
            pass
    return out


def decode_frame(mp4: str, idx: int) -> np.ndarray:
    """Frame `idx` of an mp4 as uint8 RGB HxWx3, by the first decoder that imports."""
    try:
        import av
        with av.open(str(mp4)) as c:
            for i, vf in enumerate(c.decode(c.streams.video[0])):
                if i == idx:
                    return vf.to_ndarray(format="rgb24")
        raise RuntimeError(f"{mp4}: fewer than {idx + 1} frames")
    except ImportError:
        pass
    try:
        import imageio.v3 as iio
        return np.ascontiguousarray(iio.imread(str(mp4), index=idx, plugin="pyav"))
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {mp4}")
    return np.ascontiguousarray(img[:, :, ::-1])


def extract_frames(mp4: str, idxs, out_npz: str) -> dict:
    """Dev box, package build: decode `idxs` once (one pass) and bank them losslessly as `f<idx>` arrays. -> {idx: shape}."""
    import av
    want, got = set(int(i) for i in idxs), {}
    with av.open(str(mp4)) as c:
        for i, vf in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                got[f"f{i}"] = vf.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    miss = want - {int(k[1:]) for k in got}
    if miss:
        raise RuntimeError(f"{mp4}: frames {sorted(miss)} not decoded")
    np.savez_compressed(out_npz, **got)
    return {k: v.shape for k, v in got.items()}


def read_ref_frame(mp4: str, frame: int, size_wh) -> np.ndarray:
    """Same contract as `gsplat_renderer.read_ref_frame`: RGB uint8 [H, W, 3] at `size_wh` = (width, height)."""
    npz = os.path.join(os.path.dirname(str(mp4)), "ref_frames.npz")
    img = None
    if os.path.isfile(npz):
        with np.load(npz) as z:
            if f"f{int(frame)}" in z.files:
                img = np.ascontiguousarray(z[f"f{int(frame)}"])
    if img is None:
        try:
            img = decode_frame(mp4, int(frame))
        except ImportError as e:
            raise RuntimeError(f"no mp4 decoder here and frame {frame} is not in {npz}: {e}") from e
    if (img.shape[1], img.shape[0]) != tuple(size_wh):
        img = resize_area(img, size_wh)
    return np.ascontiguousarray(img)


def install() -> dict:
    """Replace `gsplat_renderer.grad_ncc` and `.read_ref_frame` (the harness imports them by name at call time inside `falsify_actors`)."""
    import gsplat_renderer as G
    old = (G.grad_ncc, G.read_ref_frame)
    G.grad_ncc, G.read_ref_frame = grad_ncc, read_ref_frame
    return {"replaced": ["grad_ncc", "read_ref_frame"], "module": getattr(G, "__file__", "stub")}


if __name__ == "__main__":                                                         # self-check: identity, a shifted copy, border handling
    rng = np.random.default_rng(0)
    im = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
    print({"self": round(grad_ncc(im, im), 6), "shifted": round(grad_ncc(im, np.roll(im, 7, 1)), 4)})
    sys.exit(0)
