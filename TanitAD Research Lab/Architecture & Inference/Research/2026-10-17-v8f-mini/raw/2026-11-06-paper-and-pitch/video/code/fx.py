"""Drawing toolkit for the TanitAD pitch film: supersampled PIL canvas, palette, easing, widgets."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, SS, FPS = 1920, 1080, 2, 30
FD = "C:/Windows/Fonts/"
FONTFILES = {"light": "segoeuil.ttf", "reg": "segoeui.ttf", "semi": "seguisb.ttf", "bold": "segoeuib.ttf"}
_fc = {}


def font(weight, px):
    k = (weight, int(round(px * SS)))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FD + FONTFILES[weight], k[1])
    return _fc[k]


def tw(s, size, weight="reg"):
    return font(weight, size).getlength(s) / SS


# palette
BG0, BG1 = (6, 12, 27), (15, 26, 52)
TXT, TXT2, TXT3 = (236, 241, 250), (160, 174, 200), (108, 123, 152)
S_C, T_C, O_C, F_C = (176, 74, 134), (31, 111, 178), (201, 133, 0), (122, 142, 172)
MINT = (104, 196, 160)
CARD, CARD2, LINE = (15, 26, 50), (21, 35, 66), (48, 65, 100)


def tint(c, t):
    return tuple(int(v + (255 - v) * t) for v in c)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def prog(t, a, b):
    return clamp((t - a) / (b - a)) if b > a else (1.0 if t >= a else 0.0)


def ep(t, a, b):
    return ease(prog(t, a, b))


_bg = {}


def base_bg():
    if "bg" not in _bg:
        w, h = W * SS, H * SS
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * 0.62, h * 0.38
        r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        k = np.clip(1 - r * 1.35, 0, 1) ** 1.6
        vy = yy / h
        c0 = np.array(BG0, np.float32)
        c1 = np.array(BG1, np.float32)
        a = c0 + (np.array((10, 18, 38), np.float32) - c0) * vy[..., None]
        a = a + (c1 - c0)[None, None, :] * k[..., None] * 0.9
        rng = np.random.default_rng(7)
        a = a + rng.normal(0, 0.9, (h, w, 1)).astype(np.float32)
        _bg["bg"] = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return _bg["bg"]


_glow = {}


def glow_mask(r):
    r = max(2, int(r))
    if r not in _glow:
        n = 2 * r
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.sqrt((xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2) / r
        _glow[r] = Image.fromarray((np.clip(1 - d, 0, 1) ** 2.2 * 255).astype(np.uint8), "L")
    return _glow[r]


class Canvas:
    def __init__(self, base=None):
        self.im = (base or base_bg()).copy()
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def c(v):
        return int(round(v * SS))

    def _paste(self, mask, ox, oy, color, alpha):
        if alpha <= 0:
            return
        if alpha < 0.999:
            mask = mask.point([int(i * alpha + 0.5) for i in range(256)])
        self.im.paste(color, (ox, oy), mask)

    def _draw(self, bb, fn, color, alpha):
        if alpha <= 0:
            return
        if alpha >= 0.999:
            fn(self.d, 0, 0, color)
            return
        x0, y0, x1, y1 = [int(v) for v in bb]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.im.width, x1), min(self.im.height, y1)
        if x1 <= x0 or y1 <= y0:
            return
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        fn(ImageDraw.Draw(m), -x0, -y0, 255)
        self._paste(m, x0, y0, color, alpha)

    # ---------- text
    def text(self, x, y, s, size, color=TXT, weight="reg", alpha=1.0, anchor="la", tracking=0):
        if alpha <= 0 or not s:
            return
        if tracking:
            tot = sum(font(weight, size).getlength(ch) / SS + tracking for ch in s) - tracking
            x0 = x - (tot / 2 if anchor[0] == "m" else tot if anchor[0] == "r" else 0)
            for ch in s:
                self.text(x0, y, ch, size, color, weight, alpha, "l" + anchor[1])
                x0 += font(weight, size).getlength(ch) / SS + tracking
            return
        f = font(weight, size)
        px, py = x * SS, y * SS
        if alpha >= 0.999:
            self.d.text((px, py), s, font=f, fill=color, anchor=anchor)
            return
        bb = self.d.textbbox((0, 0), s, font=f, anchor=anchor)
        m = Image.new("L", (bb[2] - bb[0] + 6, bb[3] - bb[1] + 6), 0)
        ImageDraw.Draw(m).text((-bb[0] + 3, -bb[1] + 3), s, font=f, fill=255, anchor=anchor)
        self._paste(m, int(px + bb[0] - 3), int(py + bb[1] - 3), color, alpha)

    def para(self, x, y, s, size, color=TXT2, weight="reg", maxw=600, lh=1.32, alpha=1.0, anchor="l"):
        words, lines, cur = s.split(), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if tw(t, size, weight) <= maxw or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w_
        lines.append(cur)
        for i, ln in enumerate(lines):
            self.text(x, y + i * size * lh, ln, size, color, weight, alpha, anchor + "a")
        return y + len(lines) * size * lh

    # ---------- shapes
    def rrect(self, x0, y0, x1, y1, r=12, fill=None, outline=None, width=2, alpha=1.0):
        c = self.c
        b = (c(x0), c(y0), c(x1), c(y1))
        pad = c(width) + 2
        bb = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        if fill is not None:
            self._draw(bb, lambda d, ox, oy, v: d.rounded_rectangle([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], radius=c(r), fill=v), fill, alpha)
        if outline is not None:
            self._draw(bb, lambda d, ox, oy, v: d.rounded_rectangle([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], radius=c(r), outline=v, width=c(width)), outline, alpha)

    def circle(self, cx, cy, r, fill=None, outline=None, width=2, alpha=1.0):
        c = self.c
        b = (c(cx - r), c(cy - r), c(cx + r), c(cy + r))
        pad = c(width) + 2
        bb = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        if fill is not None:
            self._draw(bb, lambda d, ox, oy, v: d.ellipse([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], fill=v), fill, alpha)
        if outline is not None:
            self._draw(bb, lambda d, ox, oy, v: d.ellipse([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], outline=v, width=c(width)), outline, alpha)

    def arc(self, cx, cy, r, a0, a1, color, width=3, alpha=1.0):
        c = self.c
        b = (c(cx - r), c(cy - r), c(cx + r), c(cy + r))
        pad = c(width) + 2
        bb = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        self._draw(bb, lambda d, ox, oy, v: d.arc([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], a0, a1, fill=v, width=c(width)), color, alpha)

    def poly(self, pts, fill, alpha=1.0):
        c = self.c
        P = [(c(x), c(y)) for x, y in pts]
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        bb = (min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3)
        self._draw(bb, lambda d, ox, oy, v: d.polygon([(x + ox, y + oy) for x, y in P], fill=v), fill, alpha)

    def line(self, pts, color, width=3, alpha=1.0, cap=True):
        if len(pts) < 2:
            return
        c = self.c
        P = [(c(x), c(y)) for x, y in pts]
        w = max(1, c(width))
        pad = w + 3
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        bb = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

        def fn(d, ox, oy, v):
            Q = [(x + ox, y + oy) for x, y in P]
            d.line(Q, fill=v, width=w, joint="curve")
            if cap:
                r = w / 2
                for q in (Q if (len(Q) > 2 and w >= 6) else (Q[0], Q[-1])):
                    d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=v)
        self._draw(bb, fn, color, alpha)

    def dashed(self, p0, p1, color, width=2, dash=14, gap=10, alpha=1.0, frac=1.0):
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * frac
        if L <= 0:
            return
        ux, uy = (p1[0] - p0[0]) / math.hypot(p1[0] - p0[0], p1[1] - p0[1]), (p1[1] - p0[1]) / math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            self.line([(p0[0] + ux * s, p0[1] + uy * s), (p0[0] + ux * e, p0[1] + uy * e)], color, width, alpha, cap=False)
            s += dash + gap

    def arrow(self, p0, p1, color, width=4, alpha=1.0, head=14, frac=1.0):
        x0, y0 = p0
        x1, y1 = p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 2:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        self.line([(x0, y0), (x1 - ux * head * 0.6, y1 - uy * head * 0.6)], color, width, alpha)
        nx, ny = -uy, ux
        self.poly([(x1, y1), (x1 - ux * head + nx * head * 0.55, y1 - uy * head + ny * head * 0.55),
                   (x1 - ux * head - nx * head * 0.55, y1 - uy * head - ny * head * 0.55)], color, alpha)

    def check(self, cx, cy, s, color, width=4, alpha=1.0, frac=1.0):
        pts = [(cx - 0.45 * s, cy + 0.02 * s), (cx - 0.12 * s, cy + 0.36 * s), (cx + 0.5 * s, cy - 0.38 * s)]
        if frac < 1:
            seg1 = 0.4
            if frac < seg1:
                k = frac / seg1
                pts = [pts[0], (pts[0][0] + (pts[1][0] - pts[0][0]) * k, pts[0][1] + (pts[1][1] - pts[0][1]) * k)]
            else:
                k = (frac - seg1) / (1 - seg1)
                pts = [pts[0], pts[1], (pts[1][0] + (pts[2][0] - pts[1][0]) * k, pts[1][1] + (pts[2][1] - pts[1][1]) * k)]
        self.line(pts, color, width, alpha)

    def glow(self, cx, cy, r, color, alpha=0.5):
        r = int(r * SS)
        m = glow_mask(r)
        self._paste(m, int(cx * SS) - r, int(cy * SS) - r, color, alpha)

    def image(self, im, x, y, w=None, h=None, alpha=1.0, radius=0):
        if w is not None:
            im = im.resize((int(w * SS), int(h * SS)), Image.LANCZOS)
        if radius:
            m = Image.new("L", im.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius=self.c(radius), fill=int(255 * alpha))
        else:
            m = Image.new("L", im.size, int(255 * alpha))
        self.im.paste(im, (self.c(x), self.c(y)), m)

    def out(self):
        return self.im.reduce(SS)


# ---------- widgets
def kicker(cv, t, text, a=1.0, t0=0.15):
    cv.text(110, 66, text, 22, TXT2, "semi", alpha=ep(t, t0, t0 + 0.6) * a, tracking=3.2)


def headline(cv, t, text, a=1.0, t0=0.3, size=54):
    k = ep(t, t0, t0 + 0.8)
    cv.text(110, 104 + 12 * (1 - k), text, size, TXT, "semi", alpha=k * a)


def tier_chip(cv, x, y, label, detail="", alpha=1.0, color=TXT2, anchor="l"):
    f1, f2 = 20, 22
    w1 = tw(label, f1, "bold") + 3 * len(label) * 0.9
    w2 = tw(detail, f2, "reg") if detail else 0
    w = 28 + w1 + (18 + w2 if detail else 0)
    x0 = x - (w / 2 if anchor == "m" else w if anchor == "r" else 0)
    cv.rrect(x0, y, x0 + w, y + 42, 21, fill=CARD, outline=mix(color, BG0, 0.35), width=1.6, alpha=alpha)
    cv.text(x0 + 16, y + 21, label, f1, tint(color, 0.15), "bold", alpha=alpha, anchor="lm", tracking=2.6)
    if detail:
        cv.text(x0 + 16 + w1 + 18, y + 21, detail, f2, TXT2, "reg", alpha=alpha, anchor="lm")
    return w


def badge(cv, x, y, text, color, alpha=1.0, size=24, anchor="l"):
    w = tw(text, size, "bold") + 3.5 * len(text) + 36
    x0 = x - (w if anchor == "r" else w / 2 if anchor == "m" else 0)
    cv.rrect(x0, y, x0 + w, y + size + 22, (size + 22) / 2, fill=mix(color, BG0, 0.78), outline=color, width=2, alpha=alpha)
    cv.text(x0 + w / 2, y + (size + 22) / 2, text, size, tint(color, 0.35), "bold", alpha=alpha, anchor="mm", tracking=3.5)
    return w


def particles(cv, gt, alpha=1.0):
    rng = np.random.default_rng(11)
    cols = [S_C, T_C, O_C]
    for i in range(46):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        sp, ph, am = rng.uniform(6, 20), rng.uniform(0, 6.28), rng.uniform(10, 40)
        r, a = rng.uniform(5, 16), rng.uniform(0.10, 0.28)
        y = (y0 - sp * gt) % H
        x = x0 + math.sin(gt * 0.25 + ph) * am
        cv.glow(x, y, r * 2.2, cols[i % 3], a * alpha)
        cv.circle(x, y, max(1.0, r * 0.12), fill=tint(cols[i % 3], 0.4), alpha=a * 1.6 * alpha)


def counter_text(v, fmt):
    return fmt.format(v)
