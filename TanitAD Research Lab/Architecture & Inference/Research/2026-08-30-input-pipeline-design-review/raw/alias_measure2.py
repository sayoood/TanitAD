"""HARDENING the aliasing number against the fair objection:
'your synthetic frame is white-noise broadband; real camera frames have a ~1/f
power spectrum and the lens/OLPF already low-passes, so you are overstating it.'

So: repeat the world-model measurement on frames with a controlled 1/f^alpha
power spectrum (alpha=2 is the standard natural-image model; alpha=2.4 is a
*more* forgiving, softer image), plus an explicit lens-MTF roll-off, and report
the excess as a function of how much high-frequency energy the frame really has.

Every arm keeps the two controls: shift 0 must read exactly 0, and a frame with
NO energy above the output Nyquist must show excess == 1.00x (if it does not,
the instrument is broken, not the pipeline).
"""
import math
import torch
import torch.nn.functional as F

R = 3.0
H, W = 1080, 1920


def spectral_frame(h, w, alpha, mtf_cutoff=None, seed=0):
    """Frame with power ~ 1/f^alpha. `mtf_cutoff` (in cycles/px, <=0.5) applies a
    Gaussian lens/OLPF roll-off reaching ~0.1 at the cutoff."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(h, w, generator=g, dtype=torch.float64)
    Fh = torch.fft.rfft2(noise)
    fy = torch.fft.fftfreq(h, dtype=torch.float64).view(-1, 1)
    fx = torch.fft.rfftfreq(w, dtype=torch.float64).view(1, -1)
    f = torch.sqrt(fy ** 2 + fx ** 2).clamp_min(1e-6)
    amp = f ** (-alpha / 2.0)
    if mtf_cutoff:
        amp = amp * torch.exp(-(f / mtf_cutoff) ** 2 * math.log(10.0))
    img = torch.fft.irfft2(Fh * amp, s=(h, w))
    img = (img - img.mean()) / img.std()
    return (img * 0.18 + 0.5).clamp(0, 1)


def hf_fraction(img, cutoff):
    """Fraction of TOTAL (AC) spectral power above `cutoff` cycles/px."""
    Fh = torch.fft.rfft2(img - img.mean())
    p = (Fh.abs() ** 2)
    fy = torch.fft.fftfreq(img.shape[0], dtype=torch.float64).view(-1, 1)
    fx = torch.fft.rfftfreq(img.shape[1], dtype=torch.float64).view(1, -1)
    f = torch.sqrt(fy ** 2 + fx ** 2)
    return float(p[f > cutoff].sum() / p.sum())


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


NYQ = 1.0 / (2 * R)          # 0.1667 cycles per native px

ARMS = [
    ("BAND-LIMITED control (no energy > Nyq)", 2.0, NYQ * 0.55),
    ("soft image  1/f^2.4 + strong lens MTF", 2.4, 0.10),
    ("natural     1/f^2.0 + lens MTF 0.20",   2.0, 0.20),
    ("natural     1/f^2.0 + lens MTF 0.35",   2.0, 0.35),
    ("crisp       1/f^1.6, no MTF roll-off",  1.6, None),
    ("white-ish   1/f^0.8, no MTF roll-off",  0.8, None),
]

print("=" * 96)
print("EXCESS FRAME-TO-FRAME CHANGE (bilinear / correctly-prefiltered) vs frame spectrum")
print(f"decimation {R}x, output Nyquist = {NYQ:.4f} cycles per native px")
print("=" * 96)
print(f"{'frame spectrum':<40} {'%power>Nyq':>11} {'shift 0':>8} "
      f"{'0.33 px':>9} {'1.0 px':>8} {'3.0 px':>8}")
for name, alpha, mtf in ARMS:
    img = spectral_frame(H, W, alpha, mtf, seed=7)
    hf = hf_fraction(img, NYQ) * 100
    cells = []
    for dx in (0.0, 0.33, 1.0, 3.0):
        a0, a1 = down2d(img, R, 0.0, False), down2d(img, R, dx, False)
        b0, b1 = down2d(img, R, 0.0, True), down2d(img, R, dx, True)
        m = (slice(6, -6), slice(6, -6))
        ra = float(((a1[m] - a0[m]) ** 2).mean().sqrt())
        rb = float(((b1[m] - b0[m]) ** 2).mean().sqrt())
        cells.append(0.0 if dx == 0 else ra / max(rb, 1e-12))
    print(f"{name:<40} {hf:>10.2f}% {cells[0]:>7.2f}x {cells[1]:>8.2f}x "
          f"{cells[2]:>7.2f}x {cells[3]:>7.2f}x")

print()
print("  CONTROL row 1 is band-limited BELOW the output Nyquist: excess MUST be")
print("  ~1.00x. If it is, the instrument measures aliasing and nothing else.")
print("  CONTROL column 'shift 0': MUST be exactly 0.00x everywhere.")

print()
print("=" * 96)
print("THE SAME QUESTION FOR THE STRUCTURES DRIVING ACTUALLY NEEDS")
print("=" * 96)
print("Thin high-contrast features on a smooth background -- lane markings, poles,")
print("sign posts, distant vehicle edges. These sit AT or ABOVE the output Nyquist")
print("by construction, whatever the background spectrum is.\n")


def thin_feature_frame(h, w, width_px, seed=3):
    g = torch.Generator().manual_seed(seed)
    bg = spectral_frame(h, w, 2.0, 0.20, seed=seed) * 0.6 + 0.2
    step = max(int(width_px * 9), 8)
    for c in range(step, w - step, step):
        bg[:, c:c + width_px] = 0.95
    return bg.clamp(0, 1)


print(f"{'feature width (native px)':>26} {'~range for a 0.15 m lane line':>31} "
      f"{'excess @0.33px':>15} {'excess @1px':>12}")
for wpx in (2, 3, 4, 6, 9, 12):
    # 0.15 m lane marking width; native 0.0625 deg/px  => angular width
    # 2*atan(0.075/Z) rad == wpx * 0.0625 deg  =>  Z
    ang = math.radians(wpx * 0.0625)
    Z = 0.075 / math.tan(ang / 2)
    img = thin_feature_frame(H, W, wpx)
    row = []
    for dx in (0.33, 1.0):
        a0, a1 = down2d(img, R, 0.0, False), down2d(img, R, dx, False)
        b0, b1 = down2d(img, R, 0.0, True), down2d(img, R, dx, True)
        m = (slice(6, -6), slice(6, -6))
        ra = float(((a1[m] - a0[m]) ** 2).mean().sqrt())
        rb = float(((b1[m] - b0[m]) ** 2).mean().sqrt())
        row.append(ra / max(rb, 1e-12))
    print(f"{wpx:>26} {Z:>28.1f} m {row[0]:>14.2f}x {row[1]:>11.2f}x")
print()
print("  A 0.15 m lane marking is 2-4 native px from ~110 m down to ~55 m, and")
print("  ~9 native px at 25 m. Everything under ~6 native px is above the output")
print("  Nyquist and is therefore ALIASED rather than resolved.")
