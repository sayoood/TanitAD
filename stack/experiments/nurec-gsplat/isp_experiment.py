"""isp_experiment.py -- does the scene's own PPISP explain the render/reference residual?

Runs against a banked linear dump (``--dump``, produced by ``dump_linear.py``) so the
whole thing is 0-GPU and re-runnable.

WHAT IT TESTS, with both outcomes committed in advance
  H_ISP     the residual is the scene's trained post-processing ISP. Then it is
            PER-CAMERA and FRAME-INDEPENDENT (the two per-view stages are frozen by
            ``per_frame_ppisp_enabled: false``), so one fixed curve must serve all
            frames, and applying the file's own parameters must IMPROVE the render.
  H_COVER   the residual is dominated by the ~80% of pixels no gaussian covers (missing
            sky / far field). Then it is not radiometric at all, the apparent
            "near-equal per-channel affine gain" is an artifact of those pixels, and it
            must VANISH when the fit is restricted to covered pixels.

METRIC DISCIPLINE
  PSNR and plain NCC are RETRACTED on this clip: the negative control ranks a wrong
  frame first under both. grad-NCC is the metric that discriminates, and every score
  below is reported with its negative control (does the CORRECT reference frame win?).
  PSNR is printed only as context, never as a verdict.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppisp import apply as ppisp_apply  # noqa: E402
from ppisp import PUBLISHED_CRF_PARAMS_PER_CHANNEL, crf_curve, decode_crf, read_ppisp  # noqa: E402

LUM = np.array([0.299, 0.587, 0.114], np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def grad_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC of gradient magnitude. Every frame of a night clip is a dark street, so
    intensity NCC measures 'both are dark'; gradients carry the geometry."""
    import cv2

    def g(im):
        y = im.astype(np.float32) @ LUM
        return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))

    return _ncc(g(a), g(b))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 999.0 if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def quantile_curve(render, ref, alpha, grid, amin=0.995, nq=199):
    """Registration-invariant transfer curve: match the DISTRIBUTIONS over the covered
    pixels. Conditioning ref on the render value pixel-by-pixel does NOT work here --
    sub-pixel misalignment of small bright sources (street lamps) makes it non-monotone,
    which no camera response function can be."""
    m = alpha >= amin
    qs = np.linspace(0.001, 0.999, nq)
    out = np.empty((3, grid.size))
    for c in range(3):
        qx = np.quantile(np.clip(render[..., c][m], 0, 1), qs)
        qy = np.quantile(ref[..., c][m], qs)
        out[c] = np.interp(grid, qx, qy, left=qy[0], right=qy[-1])
    return out, int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="npz from dump_linear.py")
    ap.add_argument("--msgpack", required=True, help="gunzipped volume.nurec")
    ap.add_argument("--ref-mp4", required=True)
    ap.add_argument("--cam-idx", type=int, required=True,
                    help="index into the scene's sensor list (NOT the mp4 name)")
    ap.add_argument("--neg-frames", default="0,60,150,300,450")
    ap.add_argument("--amin", type=float, default=0.995)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    import cv2
    import msgpack

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.dump)
    frames, render, alpha, ref = d["frames"], d["render"], d["alpha"], d["ref"]
    n_f, h, w = render.shape[0], render.shape[1], render.shape[2]

    nre = msgpack.unpackb(open(args.msgpack, "rb").read(), raw=False,
                          strict_map_key=False)["nre_data"]
    pp = read_ppisp(nre)
    cam = args.cam_idx
    report = {"nurec_version": nre["version"], "cam_idx": cam,
              "describe": pp.describe(cam), "frames": frames.tolist()}

    print(f"=== PPISP as stored (nre {nre['version']}, cam {cam}) ===")
    for k, v in pp.describe(cam).items():
        print(f"  {k}: {v}")
    if pp.crf_params_per_channel != PUBLISHED_CRF_PARAMS_PER_CHANNEL:
        print(f"  !! stores {pp.crf_params_per_channel} CRF params/channel; published "
              f"layout has {PUBLISHED_CRF_PARAMS_PER_CHANNEL} -> slot assignment UNKNOWN")

    # ---- 1. how much can the three unambiguous stages possibly do? ------------------
    print("\n=== stage-by-stage effect, the file's own numbers, NO fitting (frame 0) ===")
    base = np.clip(render[0], 0, 1)
    stage_fx = {}
    for st in ("exposure", "vignetting", "color"):
        o = ppisp_apply(base, pp, cam, int(frames[0]), stages=(st,))
        stage_fx[st] = {"max_abs": float(np.abs(o - base).max()),
                        "mean_abs": float(np.abs(o - base).mean())}
        print(f"  {st:12s}: max|d|={stage_fx[st]['max_abs']:.6f}  "
              f"mean|d|={stage_fx[st]['mean_abs']:.6f}")
    o = ppisp_apply(base, pp, cam, int(frames[0]), stages=("exposure", "vignetting", "color"))
    stage_fx["all_three"] = {"max_abs": float(np.abs(o - base).max()),
                             "mean_abs": float(np.abs(o - base).mean())}
    print(f"  {'all three':12s}: max|d|={stage_fx['all_three']['max_abs']:.6f}  "
          f"mean|d|={stage_fx['all_three']['mean_abs']:.6f}")
    report["stage_effect"] = stage_fx

    # ---- 2. H_COVER: is the affine gain an uncovered-pixel artifact? ----------------
    print("\n=== affine gain, FULL-FRAME vs COVERED-ONLY (H_COVER) ===")
    aff = []
    for i in range(n_f):
        img, r = np.clip(render[i], 0, 1), ref[i]
        cov = alpha[i] >= args.amin
        row = {"frame": int(frames[i]), "covered_frac": float(cov.mean())}
        for tag, msk in (("full", None), ("covered", cov)):
            g = []
            for c in range(3):
                x, y = img[..., c].ravel(), r[..., c].ravel()
                if msk is not None:
                    x, y = x[msk.ravel()], y[msk.ravel()]
                sol, *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
                g.append(float(sol[0]))
            row[f"gain_{tag}"] = g
        err = np.abs(img - r).mean(-1)
        row["uncovered_share_of_abs_err"] = float(err[~cov].sum() / err.sum())
        aff.append(row)
        print(f"  f{row['frame']:<4d} covered={100*row['covered_frac']:.1f}%  "
              f"gain_full={[round(x,3) for x in row['gain_full']]}  "
              f"gain_covered={[round(x,3) for x in row['gain_covered']]}  "
              f"uncovered share of |err| = {100*row['uncovered_share_of_abs_err']:.1f}%")
    report["affine"] = aff

    # ---- 3. H_ISP: is the required transform frame-independent? ---------------------
    grid = np.linspace(0.0, 1.0, 101)
    curves = np.stack([quantile_curve(render[i], ref[i], alpha[i], grid, args.amin)[0]
                       for i in range(n_f)])
    print("\n=== quantile-matched transform: frame-independent? (H_ISP) ===")
    tr = {}
    for c, nm in enumerate("RGB"):
        corr = float(np.abs(curves[0, c] - grid).max())
        spread = np.abs(curves[:, c] - curves[0, c]).max(axis=1)
        tr[nm] = {"correction_size": corr, "cross_frame_spread": spread.tolist(),
                  "ratio": (spread[1:] / max(corr, 1e-9)).tolist()}
        print(f"  {nm}: correction={corr:.4f}  spread={np.round(spread,4).tolist()}  "
              f"spread/correction={np.round(spread[1:]/max(corr,1e-9),3).tolist()}")
    report["frame_independence"] = tr

    # ---- 4. grad-NCC before/after, each with its negative control -------------------
    neg = [int(x) for x in args.neg_frames.split(",")]
    cap = cv2.VideoCapture(args.ref_mp4)

    def read_ref(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read reference frame {i}")
        im = im[:, :, ::-1].astype(np.float32) / 255.0
        return im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h),
                                                            interpolation=cv2.INTER_AREA)

    refs = {j: read_ref(j) for j in neg}
    cap.release()
    f0 = int(frames[0])

    def score(img, tag):
        g = {j: grad_ncc(img, refs[j]) for j in neg}
        best = max(g, key=g.get)
        margin = g[f0] - max(v for j, v in g.items() if j != f0)
        print(f"  {tag:38s} gradNCC={g[f0]:+.4f} argmax=f{best} "
              f"{'PASS' if best == f0 else 'FAIL'} margin={margin:+.4f} "
              f"(PSNR {psnr(img, refs[f0]):.2f}, mean {img.mean():.4f})")
        return {"grad_ncc": g[f0], "argmax": best, "pass": best == f0,
                "margin": margin, "all": {str(k): v for k, v in g.items()},
                "psnr_context_only": psnr(img, refs[f0]), "mean": float(img.mean())}

    print(f"\n=== grad-NCC BEFORE / AFTER (reference mean {refs[f0].mean():.4f}) ===")
    scores = {"before": score(base, "BEFORE (linear, no ISP)")}

    # every ordered 4-of-K assignment, ranked against the frame-0 quantile curve
    ranked = []
    for perm in itertools.permutations(range(pp.crf_params_per_channel), 4):
        e = 0.0
        for c in range(3):
            y = crf_curve(grid, *decode_crf(pp.crf[cam, c], perm))
            e += float(np.mean((y - curves[0, c]) ** 2))
        ranked.append((e / 3.0, perm))
    ranked.sort()
    ident = float(np.mean([(grid - curves[0, c]) ** 2 for c in range(3)]))
    report["crf_search"] = {"n_candidates": len(ranked),
                            "identity_rmse": float(np.sqrt(ident)),
                            "top": [{"perm": list(p), "rmse": float(np.sqrt(e))}
                                    for e, p in ranked[:5]],
                            "worst_rmse": float(np.sqrt(ranked[-1][0]))}
    print(f"  [CRF slot search] identity RMSE={np.sqrt(ident):.5f}; best of {len(ranked)} "
          f"= {np.sqrt(ranked[0][0]):.5f} at idx{ranked[0][1]}; worst "
          f"{np.sqrt(ranked[-1][0]):.5f}")

    pub = tuple(range(4))
    scores["after_published_layout"] = score(
        ppisp_apply(base, pp, cam, f0, crf_perm=pub), f"AFTER PPISP published idx{pub}")
    best_perm = ranked[0][1]
    best_img = ppisp_apply(base, pp, cam, f0, crf_perm=best_perm)
    scores["after_best_of_search"] = score(
        best_img, f"AFTER PPISP best-of-search idx{best_perm}")
    scores["after_best_of_search"]["SELECTION_BIASED"] = True

    qimg = np.stack([np.interp(base[..., c], grid, curves[0, c]) for c in range(3)], -1)
    scores["after_free_empirical_curve"] = score(
        qimg, "AFTER free empirical curve (not ISP)")

    # out-of-sample: does the frame-0 curve carry to an unseen frame?
    oos = []
    for i in range(1, n_f):
        bi = np.clip(render[i], 0, 1)
        qi = np.stack([np.interp(bi[..., c], grid, curves[0, c]) for c in range(3)], -1)
        raw = grad_ncc(bi, read_ref_cached(refs, args.ref_mp4, int(frames[i]), h, w))
        fix = grad_ncc(qi, read_ref_cached(refs, args.ref_mp4, int(frames[i]), h, w))
        oos.append({"frame": int(frames[i]), "raw": raw, "after_f0_curve": fix,
                    "delta": fix - raw})
        print(f"  out-of-sample f{frames[i]}: gradNCC {raw:+.4f} -> {fix:+.4f} "
              f"(delta {fix-raw:+.4f})")
    in_s = scores["after_free_empirical_curve"]["grad_ncc"] - scores["before"]["grad_ncc"]
    ratio = float(np.mean([o["delta"] for o in oos]) / max(in_s, 1e-9))
    report["out_of_sample"] = {"per_frame": oos, "in_sample_delta": in_s,
                               "mean_oos_over_in_sample": ratio}
    print(f"  in-sample delta {in_s:+.4f}; mean out-of-sample/in-sample = {ratio:.3f}")
    report["scores"] = scores

    # ---- 5. side-by-side ------------------------------------------------------------
    def bar(text, width):
        b = np.zeros((30, width, 3), np.uint8)
        cv2.putText(b, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return b

    pad = np.ones((h, 8, 3), np.float32)
    diff = np.clip(np.abs(best_img - refs[f0]).mean(-1, keepdims=True).repeat(3, -1) * 3, 0, 1)
    cover = (alpha[0][..., None] >= args.amin).astype(np.float32).repeat(3, -1)
    row = np.concatenate([base, pad, best_img, pad, refs[f0], pad, diff, pad, cover], 1)
    row = (np.clip(row, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
    label = (f"BEFORE linear no-ISP   AFTER PPISP idx{best_perm} (UNIDENTIFIED layout)   "
             f"REFERENCE   |diff|x3   COVERED alpha>={args.amin} "
             f"({100*float((alpha[0]>=args.amin).mean()):.0f}%)   "
             f"gradNCC {scores['before']['grad_ncc']:.4f} -> "
             f"{scores['after_best_of_search']['grad_ncc']:.4f}")
    canvas = np.concatenate([bar(label, row.shape[1]), row], 0)
    canvas = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2),
                        interpolation=cv2.INTER_AREA)
    png = out_dir / "sbs_frame0_isp_before_after.png"
    cv2.imwrite(str(png), canvas)
    (out_dir / "isp_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {png}\nwrote {out_dir / 'isp_report.json'}")


_REF_CACHE: dict = {}


def read_ref_cached(seed: dict, mp4: str, idx: int, h: int, w: int) -> np.ndarray:
    if idx in seed:
        return seed[idx]
    if idx in _REF_CACHE:
        return _REF_CACHE[idx]
    import cv2

    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, im = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read reference frame {idx}")
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    if im.shape[:2] != (h, w):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    _REF_CACHE[idx] = im
    return im


if __name__ == "__main__":
    main()
