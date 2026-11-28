"""Stage 3: apply the scene's OWN PPISP parameters and score BEFORE vs AFTER.

Faithful numpy port of nv-tlabs/ppisp v1.0.0 tests/torch_reference.py (Apache-2.0),
driven entirely by the numbers stored in this scene's volume.nurec. No fitting.
Scored with grad-NCC (PSNR/NCC are retracted on this clip) + the negative control.
"""
import numpy as np, msgpack, json, cv2

SD = "/home/nvidia/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430"
REF = SD + "/camera_front_wide_120fov.mp4"
CAM = 3
D = np.load("/home/nvidia/nurec_work/linear_dump.npz")
frames, render, alpha, ref = D["frames"], D["render"], D["alpha"], D["ref"]
H, W = render.shape[1], render.shape[2]

obj = msgpack.unpackb(open("/home/nvidia/nurec_work/x/volume.msgpack", "rb").read(),
                      raw=False, strict_map_key=False)
sd = obj["nre_data"]["state_dict"]
f16 = lambda k, shp: np.frombuffer(sd[k], np.float16).astype(np.float64).reshape(shp)
P = ".post_processings.0.ppisp."
EXP = f16(P + "exposure_params", (3594,))
VIG = f16(P + "vignetting_params", (6, 3, 5))
COL = f16(P + "color_params", (3594, 8))
CRF = f16(P + "crf_params", (6, 3, 7))

softplus = lambda z: np.logaddexp(0.0, z)
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))

# ---- faithful port of _COLOR_PINV_BLOCK_DIAG / _get_homography_torch --------------
_BLK = np.zeros((8, 8))
_ZCA = [(0.0480542, -0.0043631, 0.0481283), (0.0580570, -0.0179872, 0.0431061),
        (0.0433336, -0.0180537, 0.0580500), (0.0128369, -0.0034654, 0.0128158)]
for i, (a, b, c) in enumerate(_ZCA):
    _BLK[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[a, b], [b, c]]


def homography(cp):
    off = cp @ _BLK
    bd, rd, gd, nd = off[0:2], off[2:4], off[4:6], off[6:8]
    t_b = np.array([0.0 + bd[0], 0.0 + bd[1], 1.0])
    t_r = np.array([1.0 + rd[0], 0.0 + rd[1], 1.0])
    t_g = np.array([0.0 + gd[0], 1.0 + gd[1], 1.0])
    t_k = np.array([1 / 3 + nd[0], 1 / 3 + nd[1], 1.0])
    T = np.stack([t_b, t_r, t_g], axis=1)
    skew = np.array([[0, -t_k[2], t_k[1]], [t_k[2], 0, -t_k[0]], [-t_k[1], t_k[0], 0]])
    M = skew @ T
    lams = [np.cross(M[0], M[1]), np.cross(M[0], M[2]), np.cross(M[1], M[2])]
    lam = lams[int(np.argmax([l @ l for l in lams]))]
    S_inv = np.array([[-1., -1., 1.], [1., 0., 0.], [0., 1., 0.]])
    Hm = T @ np.diag(lam) @ S_inv
    return Hm / (Hm[2, 2] + 1e-10)


def crf_apply(x, toe, shoulder, gamma, center):
    x = np.clip(x, 0.0, 1.0)
    eps = 1e-6
    lerp = toe + center * (shoulder - toe)
    a = (shoulder * center) / lerp
    b = 1.0 - a
    lo = a * np.power(np.clip(x / center, eps, None), toe)
    hi = 1.0 - b * np.power(np.clip((1.0 - x) / (1.0 - center), eps, None), shoulder)
    return np.power(np.clip(np.where(x <= center, lo, hi), eps, None), gamma)


def ppisp(img, cam_idx, view_idx, crf_perm, stages=("exp", "vig", "col", "crf")):
    rgb = img.copy()
    if "exp" in stages and view_idx is not None:
        rgb = rgb * (2.0 ** EXP[view_idx])
    if "vig" in stages and cam_idx is not None:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        mx = float(max(W, H))
        uv = np.stack([(xs - W * 0.5) / mx, (ys - H * 0.5) / mx], -1)
        out = np.empty_like(rgb)
        for ch in range(3):
            v = VIG[cam_idx, ch]
            d = uv - v[:2]
            r2 = (d * d).sum(-1)
            fall = np.ones_like(r2)
            r2p = r2.copy()
            for al in v[2:]:
                fall = fall + al * r2p
                r2p = r2p * r2
            out[..., ch] = rgb[..., ch] * np.clip(fall, 0, 1)
        rgb = out
    if "col" in stages and view_idx is not None:
        Hm = homography(COL[view_idx])
        inten = rgb.sum(-1, keepdims=True)
        rgi = np.concatenate([rgb[..., 0:1], rgb[..., 1:2], inten], -1)
        rgi = rgi @ Hm.T
        rgi = rgi * (inten / (rgi[..., 2:3] + 1e-5))
        r_, g_ = rgi[..., 0], rgi[..., 1]
        rgb = np.stack([r_, g_, rgi[..., 2] - r_ - g_], -1)
    if "crf" in stages and cam_idx is not None:
        out = np.empty_like(rgb)
        for ch in range(3):
            p = CRF[cam_idx, ch]
            out[..., ch] = crf_apply(rgb[..., ch],
                                     0.3 + softplus(p[crf_perm[0]]),
                                     0.3 + softplus(p[crf_perm[1]]),
                                     0.1 + softplus(p[crf_perm[2]]),
                                     sigmoid(p[crf_perm[3]]))
        rgb = out
    return np.clip(rgb, 0, 1)


# ---- metrics (grad-NCC; PSNR/NCC retracted on this clip) --------------------------
LUM = np.array([0.299, 0.587, 0.114], np.float32)


def _ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def gmag(im):
    g = (im.astype(np.float32) @ LUM)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def grad_ncc(a, b):
    return _ncc(gmag(a), gmag(b))


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 999.0 if m <= 0 else float(10 * np.log10(1.0 / m))


cap = cv2.VideoCapture(REF)


def ref_frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    assert ok, i
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) if im.shape[:2] != (H, W) else im


NEG = [0, 60, 150, 300, 450]
REFS = {j: ref_frame(j) for j in NEG}
cap.release()

# view index: 3594 = 6 cameras x 599 frames, camera-major (a block of 599 per camera)
VIEW0 = CAM * 599 + 0
print("view index for cam%d frame0 = %d" % (CAM, VIEW0))
print("exposure[%d] = %.6f  -> gain 2^x = %.6f" % (VIEW0, EXP[VIEW0], 2.0 ** EXP[VIEW0]))
print("color  [%d] = %s" % (VIEW0, COL[VIEW0]))
print("homography =\n%s" % np.round(homography(COL[VIEW0]), 6))

base = np.clip(render[0], 0, 1)
print("\n=== stage-by-stage effect on frame 0 (the file's own numbers, NO fitting) ===")
for st, lab in [(("exp",), "exposure only"), (("vig",), "vignetting only"),
                (("col",), "colour only"), (("exp", "vig", "col"), "exposure+vignetting+colour")]:
    o = ppisp(base, CAM, VIEW0, (0, 1, 2, 3), stages=st)
    print("  %-28s: max|delta|=%.6f  mean|delta|=%.6f"
          % (lab, np.abs(o - base).max(), np.abs(o - base).mean()))


def score(img, tag):
    g = {j: grad_ncc(img, REFS[j]) for j in NEG}
    best = max(g, key=g.get)
    marg = g[0] - max(v for j, v in g.items() if j != 0)
    wrong = " ".join("f%d=%+.4f" % (j, g[j]) for j in NEG if j)
    print("  %-38s gradNCC f0=%+.4f | wrong: %s | argmax=f%d %s margin=%+.4f | PSNR=%.2f mean=%.4f"
          % (tag, g[0], wrong, best, "PASS" if best == 0 else "FAIL", marg,
             psnr(img, REFS[0]), img.mean()))
    return g[0], best, marg


print("\n=== grad-NCC BEFORE / AFTER (reference mean = %.4f) ===" % REFS[0].mean())
b_g, b_arg, b_m = score(base, "BEFORE (linear, no ISP)")

cands = {"public v1.0.0 first-4 idx(0,1,2,3)": (0, 1, 2, 3),
         "best-of-840 vs quantile idx(2,4,0,6)": (2, 4, 0, 6),
         "runner-up idx(2,4,0,3)": (2, 4, 0, 3),
         "runner-up idx(5,4,1,6)": (5, 4, 1, 6)}
after = {}
best_img = pub_img = None
for lab, perm in cands.items():
    img = ppisp(base, CAM, VIEW0, perm)
    after[lab] = score(img, "AFTER full PPISP %s" % (perm,))
    if perm == (2, 4, 0, 6):
        best_img = img
    if perm == (0, 1, 2, 3):
        pub_img = img

# ---- the empirical alternative: per-frame quantile curve (NOT an ISP) -------------
Q = np.load("/home/nvidia/nurec_work/quantile_curves.npz")
XG, curves = Q["XG"], Q["curves"]
qimg = np.stack([np.interp(base[..., c], XG, curves[0, c]) for c in range(3)], -1)
q_res = score(qimg, "AFTER empirical quantile curve (f0)")
# cross-frame: apply the frame-0 curve to frame 300's render, score vs ref 300
b300 = np.clip(render[2], 0, 1)
q300 = np.stack([np.interp(b300[..., c], XG, curves[0, c]) for c in range(3)], -1)
print("  cross-frame transfer of the f0 curve onto frame 300: "
      "gradNCC(f300 render, f300 ref) raw=%+.4f  after-f0-curve=%+.4f"
      % (grad_ncc(b300, REFS[300]), grad_ncc(q300, REFS[300])))

json.dump({"before": {"grad_ncc_f0": b_g, "argmax": int(b_arg), "margin": b_m},
           "after": {k: {"grad_ncc_f0": v[0], "argmax": int(v[1]), "margin": v[2]}
                     for k, v in after.items()},
           "after_quantile": {"grad_ncc_f0": q_res[0], "argmax": int(q_res[1]),
                              "margin": q_res[2]}},
          open("/home/nvidia/nurec_work/ppisp_scores.json", "w"), indent=1)


def bar(text, w):
    b = np.zeros((30, w, 3), np.uint8)
    cv2.putText(b, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return b


pad = np.ones((H, 8, 3), np.float32)
diff = np.clip(np.abs(best_img - REFS[0]).mean(-1, keepdims=True).repeat(3, -1) * 3, 0, 1)
row = np.concatenate([base, pad, best_img, pad, REFS[0], pad, diff], 1)
row = (np.clip(row, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
lab = ("BEFORE linear no-ISP        AFTER PPISP idx(2,4,0,6)         REFERENCE NuRec mp4"
       "          |diff|x3      gradNCC %.4f -> %.4f"
       % (b_g, after["best-of-840 vs quantile idx(2,4,0,6)"][0]))
out = np.concatenate([bar(lab, row.shape[1]), row], 0)
out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
cv2.imwrite("/home/nvidia/nurec_work/sbs_frame0_isp_before_after.png", out)
print("\nwrote /home/nvidia/nurec_work/sbs_frame0_isp_before_after.png")
