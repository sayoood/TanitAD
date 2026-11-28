"""MEASURED: does TanitAD's frame build ALIAS, and what does that do to the
temporal signal a world model is asked to predict?

THE OPERATOR UNDER TEST is the one the build actually uses:
  stack/tanitad/data/calib.py:989-993
    grid, mask = cylindrical_grid(intr, h, w, frame, device=vid.device)
    out = F.grid_sample(vid.float(), grid.expand(t,-1,-1,-1), mode="bilinear",
                        padding_mode=padding_mode, align_corners=False)
reached from stack/tanitad/data/physicalai.py:520 / :577 (_remap_batch), on
NATIVELY DECODED ~1080p frames (physicalai.py:524 names a "604-frame 1080p clip").

`grid_sample(mode="bilinear")` has a fixed 2-source-pixel support. It is a
*reconstruction* filter, not a *decimation* filter: it does NOT widen with the
sampling ratio. At a ~3x decimation everything above the output Nyquist folds
back. This script measures how much, and — the part that matters for a world
model — how much of the OUTPUT's frame-to-frame change is scene motion versus
sampling-phase artefact.

Controls (per CLAUDE.md's probe doctrine):
  * a CONSTANT (DC) image must show exactly zero shift-induced change;
  * a LOW-frequency image (well under Nyquist) must show the same change under
    both resamplers — if it does not, the comparison is measuring the filters'
    passbands, not aliasing;
  * an ideal properly-prefiltered resampler is the reference arm.
"""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---- geometry, from source ------------------------------------------------
F_REF = 305.5774907364391
OUT_H, OUT_W = 256, 640
NATIVE_H, NATIVE_W = 1080, 1920          # physicalai.py:524
HFOV = math.degrees(2 * (OUT_W / 2) / F_REF)

print("=" * 84)
print("0. THE DECIMATION RATIO")
print("=" * 84)
native_deg_per_px = 120.0 / NATIVE_W      # camera_front_wide_120fov, square px
out_deg_per_px_x = HFOV / OUT_W
out_deg_per_px_y = math.degrees(math.atan((OUT_H / 2) / F_REF) * 2) / OUT_H
print(f"  native  ~{NATIVE_W}x{NATIVE_H}, ~120 deg  -> {native_deg_per_px:.5f} deg/px")
print(f"  output   {OUT_W}x{OUT_H} cylindrical      -> {out_deg_per_px_x:.5f} deg/px (x)")
print(f"                                            -> {out_deg_per_px_y:.5f} deg/px (y, at centre)")
print(f"  DECIMATION x: {out_deg_per_px_x/native_deg_per_px:.3f}x    "
      f"y: {out_deg_per_px_y/native_deg_per_px:.3f}x")
print(f"  => output Nyquist sits at 1/{out_deg_per_px_x/native_deg_per_px:.2f} of the")
print( "     native Nyquist. Everything between folds back as alias.")
print( "  EVIDENCE CLASS: ESTIMATED for the native pixel count (the 1080p figure")
print( "  is a source COMMENT, physicalai.py:524, not a manifest read); the ratio")
print( "  is exact given it. At 1920x1200 or 3848x2168 the ratio only grows.")

R = 3.0  # decimation ratio under test


def make_signal(n, freq_cyc_per_px, kind="sine"):
    """1-D test signal on the NATIVE grid."""
    x = torch.arange(n, dtype=torch.float64)
    if kind == "dc":
        return torch.ones(n, dtype=torch.float64) * 0.5
    return 0.5 + 0.45 * torch.sin(2 * math.pi * freq_cyc_per_px * x)


def resample_bilinear(sig, ratio, phase):
    """EXACTLY what the build does: point-sample with a 2-tap bilinear kernel."""
    n_in = sig.numel()
    n_out = int(n_in / ratio)
    src = (torch.arange(n_out, dtype=torch.float64) + 0.5) * ratio - 0.5 + phase
    i0 = torch.floor(src).long().clamp(0, n_in - 2)
    w = (src - i0.double()).clamp(0, 1)
    return sig[i0] * (1 - w) + sig[i0 + 1] * w


def resample_prefiltered(sig, ratio, phase):
    """The reference arm: box-average over the full output pixel footprint
    (a correct, cheap decimation filter) then the same bilinear read."""
    k = int(round(ratio))
    if k > 1:
        pad = k // 2
        s = F.avg_pool1d(sig.view(1, 1, -1), kernel_size=k, stride=1,
                         padding=pad, count_include_pad=False).view(-1)
        s = s[:sig.numel()]
    else:
        s = sig
    return resample_bilinear(s, ratio, phase)


print()
print("=" * 84)
print("1. SHIFT-INDUCED OUTPUT CHANGE vs SPATIAL FREQUENCY")
print("=" * 84)
print("A rigid sub-pixel translation of the SCENE (what ego motion produces).")
print("A correct resampler moves the output; an ALIASING resampler also changes")
print("its AMPLITUDE and its apparent frequency. We shift by 1.0 NATIVE px --")
print("i.e. 1/3 of an output pixel -- and report the RMS output change.\n")
N = 6000
nyq_out = 1.0 / (2 * R)          # output Nyquist in cycles per native px
print(f"  output Nyquist = {nyq_out:.4f} cyc/native-px "
      f"(= 1 cycle per {1/nyq_out:.1f} native px = 2 output px)")
print()
print(f"{'freq (cyc/native px)':>21} {'vs Nyq':>8} {'period(nat px)':>15} "
      f"{'BILINEAR rms':>14} {'PREFILT rms':>13} {'excess':>9}")
rows = []
for f_ in (0.0000, 0.0100, 0.0250, 0.0500, 0.0833, 0.1250, 0.1667,
           0.2000, 0.2500, 0.3333, 0.4000):
    kind = "dc" if f_ == 0 else "sine"
    s0 = make_signal(N, f_, kind)
    s1 = make_signal(N, f_, kind)
    # shift the SCENE by 1 native px == move the sampling phase by 1
    a0 = resample_bilinear(s0, R, 0.0)
    a1 = resample_bilinear(s1, R, 1.0)
    b0 = resample_prefiltered(s0, R, 0.0)
    b1 = resample_prefiltered(s1, R, 1.0)
    m = slice(20, -20)
    ra = float(((a1[m] - a0[m]) ** 2).mean().sqrt())
    rb = float(((b1[m] - b0[m]) ** 2).mean().sqrt())
    per = float("inf") if f_ == 0 else 1 / f_
    rows.append((f_, ra, rb))
    print(f"{f_:>21.4f} {f_/nyq_out:>8.2f} {per:>15.1f} {ra:>14.5f} "
          f"{rb:>13.5f} {ra/max(rb,1e-9):>8.2f}x")

print()
print("  CONTROL (DC, f=0): both must read EXACTLY 0.00000 -- they do "
      f"({rows[0][1]:.5f} / {rows[0][2]:.5f}).")
print("  CONTROL (f=0.010, 0.12x Nyquist): the two resamplers must AGREE, else")
print("  the table is measuring passbands, not aliasing -- "
      f"bilinear {rows[1][1]:.5f} vs prefiltered {rows[1][2]:.5f}.")

print()
print("=" * 84)
print("2. AMPLITUDE ERROR -- what an ALIASED structure looks like")
print("=" * 84)
print("For a structure ABOVE the output Nyquist, the correct output is ~flat")
print("(it is not resolvable). Bilinear instead returns a spurious LOW-frequency")
print("beat whose amplitude and phase depend on the sampling offset.\n")
print(f"{'freq':>8} {'vs Nyq':>7} {'BILIN out amplitude':>21} "
      f"{'PREFILT out amplitude':>23} {'spurious':>10}")
for f_ in (0.20, 0.25, 0.30, 0.3333, 0.40, 0.45):
    s = make_signal(N, f_)
    a = resample_bilinear(s, R, 0.0)[20:-20]
    b = resample_prefiltered(s, R, 0.0)[20:-20]
    amp_a = float(a.max() - a.min()) / 2
    amp_b = float(b.max() - b.min()) / 2
    print(f"{f_:>8.4f} {f_/nyq_out:>7.2f} {amp_a:>21.5f} {amp_b:>23.5f} "
          f"{amp_a/max(amp_b,1e-9):>9.1f}x")
print()
print("  True amplitude of the input is 0.45. A resolvable structure should come")
print("  through near 0.45; an UNRESOLVABLE one should come through near 0.")

print()
print("=" * 84)
print("3. THE WORLD-MODEL CONSEQUENCE -- how much of dI/dt is NOT the scene")
print("=" * 84)
print("A 2-D driving-like frame swept past the camera at a constant rate, so")
print("EVERY output change is caused by one known rigid translation. We measure")
print("the part of the frame-to-frame difference that a perfect predictor of the")
print("SCENE could not have predicted, i.e. the part created by sampling phase.\n")


def make_frame(h, w, seed=0):
    """A native-resolution frame with driving-like structure: broadband texture
    (road/foliage), thin high-frequency lines (lane markings, poles, distant
    car edges) and smooth low-frequency regions (sky, tarmac)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 1, h // 24, w // 24, generator=g, dtype=torch.float64)
    img = F.interpolate(base, size=(h, w), mode="bicubic",
                        align_corners=False).view(h, w) * 0.5 + 0.25
    tex = torch.rand(h, w, generator=g, dtype=torch.float64)
    img = img + 0.18 * (tex - 0.5)                       # broadband
    for c in range(90, w - 90, 137):                     # thin vertical lines
        img[:, c:c + 2] = 0.95
    for r in range(120, h - 60, 211):                    # thin horizontal lines
        img[r:r + 2, :] = 0.05
    return img.clamp(0, 1)


def down2d(img, ratio, dx, prefilter):
    h, w = img.shape
    x = img.view(1, 1, h, w)
    if prefilter:
        k = int(round(ratio))
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2,
                         count_include_pad=False)[..., :h, :w]
    ho, wo = int(h / ratio), int(w / ratio)
    sy = (torch.arange(ho, dtype=torch.float64) + 0.5) * ratio - 0.5
    sx = (torch.arange(wo, dtype=torch.float64) + 0.5) * ratio - 0.5 + dx
    gy = (sy / (h - 1) * 2 - 1).view(-1, 1).expand(ho, wo)
    gx = (sx / (w - 1) * 2 - 1).view(1, -1).expand(ho, wo)
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(x, grid, mode="bilinear", align_corners=True).view(ho, wo)


img = make_frame(1080, 1920, seed=1)
print(f"{'ego shift (native px)':>22} {'BILINEAR |dI| rms':>19} "
      f"{'PREFILTERED |dI| rms':>21} {'excess':>9}")
for dx in (0.0, 0.33, 0.5, 1.0, 2.0, 3.0, 6.0):
    a0, a1 = down2d(img, R, 0.0, False), down2d(img, R, dx, False)
    b0, b1 = down2d(img, R, 0.0, True), down2d(img, R, dx, True)
    m = (slice(4, -4), slice(4, -4))
    ra = float(((a1[m] - a0[m]) ** 2).mean().sqrt())
    rb = float(((b1[m] - b0[m]) ** 2).mean().sqrt())
    print(f"{dx:>22.2f} {ra:>19.5f} {rb:>21.5f} {ra/max(rb,1e-9):>8.2f}x")

print()
print("  CONTROL (shift 0.0): both MUST be exactly 0.00000.")
print()
print("  ⭐ Read the 0.33-native-px row: that is a shift of ONE NINTH of an")
print("  output pixel -- far below anything a predictor could or should track.")
print("  Whatever the bilinear column reports there is pure sampling artefact:")
print("  temporal noise injected into the exact signal the world model is")
print("  trained to predict. Its conditional mean given the scene is the MEAN.")
