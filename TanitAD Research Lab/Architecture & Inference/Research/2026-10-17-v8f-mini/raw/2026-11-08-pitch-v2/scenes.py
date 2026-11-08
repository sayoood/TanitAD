"""The ten scenes of the TanitAD pitch film.  Every scene is  fn(cv, t)  with t = local seconds."""
import math
from PIL import Image
from fx import *

ATT = "F:/TanitAD-scratch/pitch/assets/att"
LV = [("S", "Strategic", S_C), ("T", "Tactical", T_C), ("O", "Operative", O_C)]
MINUS = "\u2212"


def st(t, i, t0, step, dur=0.8):
    return ep(t, t0 + i * step, t0 + i * step + dur)


def letter_dot(cv, x, y, ch, col, r=30, a=1.0, size=34):
    cv.circle(x, y, r, fill=mix(col, BG0, 0.72), outline=col, width=2.5, alpha=a)
    cv.text(x, y, ch, size, tint(col, 0.35), "bold", alpha=a, anchor="mm")


def card(cv, x0, y0, x1, y1, a=1.0, col=LINE, fill=CARD, r=18, w=1.8):
    cv.rrect(x0, y0, x1, y1, r, fill=fill, outline=col, width=w, alpha=a)


def pbar(cv, x, y, w, h, frac, col, a=1.0, r=None):
    ww = w * frac
    if ww >= 3:
        cv.rrect(x, y, x + ww, y + h, r if r is not None else min(h / 2, 10), fill=col, alpha=a)


# ================================================================= 1 title
def s1(cv, t):
    ys, cols, spd, nds = [800, 852, 904], [S_C, T_C, O_C], [110, 300, 720], [3, 6, 12]
    names = ["S  strategic", "T  tactical", "O  operative"]
    L = 1700
    for i in range(3):
        a = ep(t, 0.2 + 0.2 * i, 1.2 + 0.2 * i)
        y, c = ys[i], cols[i]
        cv.line([(110, y), (110 + L * a, y)], mix(c, BG1, 0.45), 2.5)
        for k in range(nds[i]):
            x = 110 + ((t * spd[i] + k * L / nds[i]) % L)
            if x < 110 + L * a:
                cv.glow(x, y, 17, c, 0.6 * a)
                cv.circle(x, y, 5, fill=tint(c, 0.35), alpha=a)
        cv.text(1810, y - 34, names[i], 22, tint(c, 0.35), "semi", alpha=a, anchor="ra", tracking=2)
    k = ep(t, 0.5, 1.5)
    cv.text(106, 250 + 22 * (1 - k), "TanitAD", 170, TXT, "bold", alpha=k)
    for i, c in enumerate(cols):
        w = 148 * ep(t, 1.2 + 0.2 * i, 1.9 + 0.2 * i)
        if w > 3:
            cv.rrect(114 + i * 160, 478, 114 + i * 160 + w, 486, 4, fill=c)
    k = ep(t, 1.6, 2.5)
    cv.text(112, 522 + 10 * (1 - k), "A hierarchical world model that drives at every level of abstraction", 44, TXT, "light", alpha=k)
    k = ep(t, 2.5, 3.4)
    cv.text(112, 610, "Dr.-Ing. Sayed Bouzouraa  \u00b7  TanitAD Research", 30, TXT2, "reg", alpha=k)


# ================================================================= 2 problem
def ico(cv, kind, cx, cy, s, c, a):
    if kind == 0:
        for i, (w_, yo) in enumerate(((0.45, -0.34), (0.75, 0.0), (1.05, 0.34))):
            cv.rrect(cx - w_ * s / 2, cy + yo * s - 0.1 * s, cx + w_ * s / 2, cy + yo * s + 0.1 * s, 7, fill=mix(c, BG1, 0.1 * (2 - i)), alpha=a)
    elif kind == 1:
        cv.rrect(cx - 0.5 * s, cy - 0.5 * s, cx + 0.5 * s, cy + 0.5 * s, 14, outline=c, width=4, alpha=a)
        cv.text(cx, cy + 0.03 * s, "?", int(0.8 * s), c, "semi", alpha=a, anchor="mm")
    elif kind == 2:
        cv.arc(cx, cy + 0.25 * s, 0.55 * s, 180, 360, c, 5, a)
        cv.line([(cx, cy + 0.25 * s), (cx + 0.3 * s, cy - 0.12 * s)], c, 5, a)
        cv.circle(cx, cy + 0.25 * s, 0.07 * s, fill=c, alpha=a)
        cv.line([(cx - 0.55 * s, cy + 0.25 * s), (cx + 0.55 * s, cy + 0.25 * s)], mix(c, BG1, 0.5), 3, a)
    else:
        cv.rrect(cx - 0.35 * s, cy - 0.35 * s, cx + 0.35 * s, cy + 0.35 * s, 10, outline=c, width=4, alpha=a)
        cv.rrect(cx - 0.17 * s, cy - 0.17 * s, cx + 0.17 * s, cy + 0.17 * s, 5, fill=c, alpha=a)
        for j in (-0.2, 0, 0.2):
            cv.line([(cx + j * s, cy - 0.35 * s), (cx + j * s, cy - 0.52 * s)], c, 4, a)
            cv.line([(cx + j * s, cy + 0.35 * s), (cx + j * s, cy + 0.52 * s)], c, 4, a)
            cv.line([(cx - 0.35 * s, cy + j * s), (cx - 0.52 * s, cy + j * s)], c, 4, a)
            cv.line([(cx + 0.35 * s, cy + j * s), (cx + 0.52 * s, cy + j * s)], c, 4, a)


def s2(cv, t):
    kicker(cv, t, "02  \u00b7  THE PROBLEM")
    headline(cv, t, "Four limits of end-to-end driving models")
    items = [("Data-hungry", "Performance keeps depending on ever more driving data."),
             ("Opaque", "One network, with no inspectable intermediate decisions."),
             ("Unaware of its limits", "No built-in sense of when the model should not be trusted."),
             ("Expensive at inference", "Large models that are heavy to run on the vehicle.")]
    for i, (h_, d_) in enumerate(items):
        a = st(t, i, 1.0, 1.15, 0.8)
        x0 = 110 + i * 425
        y = 340 + 20 * (1 - a)
        card(cv, x0, y, x0 + 395, y + 400, a)
        ico(cv, i, x0 + 197, y + 105, 100, tint(F_C, 0.25), a)
        cv.text(x0 + 30, y + 215, h_, 32, TXT, "semi", alpha=a)
        cv.para(x0 + 30, y + 268, d_, 25, TXT2, "reg", 340, alpha=a)
    a = ep(t, 5.9, 6.9)
    cv.text(110, 850, "TanitAD takes a different route:", 34, TXT2, "light", alpha=a)
    cv.text(110, 898, "a structured hierarchy of world models, at most 800 M parameters.", 44, TXT, "semi", alpha=ep(t, 6.3, 7.3))


# ================================================================= 3 the 4B idea
def s3(cv, t):
    kicker(cv, t, "03  \u00b7  THE 4B IDEA")
    headline(cv, t, "Four brains, four clocks")
    rows = [("S", "Strategic", S_C, "\u2264 0.5 Hz", "6\u201330 s", "Route checkpoints, detected context and derived goals", 2.4),
            ("T", "Tactical", T_C, "1\u20132 Hz", "up to 6 s", "Emits goals and constraints: the action for the operative level", 1.0),
            ("O", "Operative", O_C, "10\u201320 Hz", "up to 2 s", "Control, and one or more trajectories up to 6 s", 0.35),
            ("F", "Fallback", F_C, "continuous", "", "Self-monitoring; owns the minimal-risk manoeuvre", 0)]
    X0, X1, Hh = 110, 1330, 140
    order = {3: 0, 2: 1, 1: 2, 0: 3}
    for i, (ch, nm, col, hz, hor, desc, per) in enumerate(rows):
        a = ep(t, 1.0 + 1.1 * order[i], 1.8 + 1.1 * order[i])
        y0 = 250 + i * 160 + 26 * (1 - a)
        cv.rrect(X0, y0, X1, y0 + Hh, 18, fill=mix(col, BG0, 0.86), outline=mix(col, BG0, 0.35), width=2, alpha=a)
        cv.rrect(X0, y0, X0 + 10, y0 + Hh, 5, fill=col, alpha=a)
        letter_dot(cv, X0 + 70, y0 + Hh / 2, ch, col, 32, a)
        cv.text(X0 + 128, y0 + 24, nm, 42, TXT, "bold", alpha=a)
        cv.para(X0 + 128, y0 + 80, desc, 25, TXT2, "reg", 720, alpha=a)
        cy = y0 + Hh / 2
        for j, (txt) in enumerate([hz, hor]):
            if txt:
                w = tw(txt, 25, "semi") + 34
                xr = X1 - 130
                yy = y0 + 22 + j * 58
                cv.rrect(xr - w, yy, xr, yy + 42, 21, fill=CARD, outline=mix(col, BG0, 0.2), width=1.8, alpha=a)
                cv.text(xr - w / 2, yy + 21, txt, 25, tint(col, 0.35), "semi", alpha=a, anchor="mm")
        # clock
        cx = X1 - 62
        if per:
            ph = (t / per) % 1.0
            cv.circle(cx, cy, 10 + 28 * ph, outline=col, width=2.5, alpha=a * (1 - ph) * 0.9)
        else:
            cv.circle(cx, cy, 22 + 3 * math.sin(t * 2), outline=col, width=2.5, alpha=a * 0.55)
        cv.circle(cx, cy, 8, fill=tint(col, 0.2), alpha=a)
    # flows
    xd, xu = 1470, 1700
    a1 = ep(t, 6.4, 7.2)
    cv.text(xd, 190, "Goals", 32, tint(MINT, 0.1), "bold", alpha=a1, anchor="ma")
    cv.para(xd, 232, "condition strictly downward", 22, TXT2, "reg", 210, alpha=a1, anchor="m")
    cv.arrow((xd, 350), (xd, 860), mix(TXT2, BG0, 0.2), 4, alpha=a1, frac=ep(t, 6.6, 8.6), head=20)
    a2 = ep(t, 9.0, 9.8)
    cv.text(xu, 190, "Latents", 32, tint(MINT, 0.1), "bold", alpha=a2, anchor="ma")
    cv.para(xu, 232, "flow upward through a stop-gradient barrier", 22, TXT2, "reg", 210, alpha=a2, anchor="m")
    cv.arrow((xu, 860), (xu, 350), mix(TXT2, BG0, 0.2), 4, alpha=a2, frac=ep(t, 9.2, 11.2), head=20)
    for k, yb in enumerate((400, 560, 720)):
        ab = ep(t, 10.4 + 0.3 * k, 11.0 + 0.3 * k)
        cv.rrect(xu - 38, yb - 5, xu + 38, yb + 5, 3, fill=O_C if False else tint(S_C, 0.3), alpha=ab)
        cv.line([(xu - 38, yb - 17), (xu - 38, yb + 17)], tint(S_C, 0.3), 4, ab)
        cv.line([(xu + 38, yb - 17), (xu + 38, yb + 17)], tint(S_C, 0.3), 4, ab)
    cv.text(xu + 56, 560, "stop-gradient", 22, tint(S_C, 0.45), "semi", alpha=ep(t, 10.8, 11.4), anchor="lm")
    # flowing dots
    if t > 8.4:
        f = ep(t, 8.4, 9.2)
        for k in range(6):
            y = 350 + ((t * 120 + k * 510 / 6) % 510)
            c = mix(S_C, O_C, (y - 350) / 510)
            cv.glow(xd, y, 15, c, 0.6 * f)
            cv.circle(xd, y, 5, fill=tint(c, 0.4), alpha=f)
    if t > 11.2:
        f = ep(t, 11.2, 12.0)
        for k in range(6):
            y = 860 - ((t * 120 + k * 510 / 6) % 510)
            c = mix(S_C, O_C, (y - 350) / 510)
            cv.glow(xu, y, 15, c, 0.6 * f)
            cv.circle(xu, y, 5, fill=tint(c, 0.4), alpha=f)
    cv.text(110, 920, "Every level reads the current ego state and has its own predictor in its own latent space.", 26, TXT2, "reg", alpha=ep(t, 12.4, 13.4))
    cv.text(110, 966, "A self-supervised, hierarchical latent world model, within at most 800 M parameters.", 30, TXT, "semi", alpha=ep(t, 13.2, 14.2))


# ================================================================= 4 v8F-mini
def poly_pts(f, n=40):
    return [f(i / (n - 1)) for i in range(n)]


def s4(cv, t):
    kicker(cv, t, "04  \u00b7  THE REFERENCE IMPLEMENTATION")
    headline(cv, t, "v8F-mini: three supervised levels over frozen features")
    Y0, Y1 = 250, 690
    A = lambda a, b: ep(t, a, b)
    # camera
    a = A(0.6, 1.4)
    card(cv, 110, 340, 290, 600, a)
    cx, cy = 200, 440
    cv.rrect(cx - 44, cy - 28, cx + 44, cy + 28, 10, outline=tint(F_C, 0.3), width=4, alpha=a)
    cv.circle(cx, cy, 15, outline=tint(F_C, 0.3), width=4, alpha=a)
    cv.rrect(cx - 18, cy - 38, cx + 18, cy - 28, 4, fill=tint(F_C, 0.3), alpha=a)
    cv.text(200, 510, "Front-wide", 24, TXT, "semi", alpha=a, anchor="ma")
    cv.text(200, 540, "camera", 24, TXT, "semi", alpha=a, anchor="ma")
    cv.arrow((298, 470), (342, 470), TXT3, 3, alpha=A(1.2, 1.8), head=12)
    # DINO
    a = A(1.4, 2.4)
    card(cv, 350, Y0, 620, Y1, a, col=mix(F_C, BG0, 0.2))
    cv.text(380, Y0 + 24, "Frozen DINOv3", 30, TXT, "semi", alpha=a)
    cv.text(380, Y0 + 62, "L/16 features", 24, TXT2, "reg", alpha=a)
    for r in range(5):
        for c_ in range(6):
            sh = 0.5 + 0.5 * math.sin(t * 2.2 + r * 0.9 + c_ * 0.6)
            cv.rrect(382 + c_ * 36, 360 + r * 36, 382 + c_ * 36 + 28, 360 + r * 36 + 28, 6, fill=mix(CARD2, tint(F_C, 0.2), 0.25 + 0.5 * sh), alpha=a)
    cv.text(380, 600, "frozen", 22, tint(F_C, 0.3), "bold", alpha=a, tracking=3)
    cv.text(380, 636, "shared adapter", 22, TXT2, "reg", alpha=a)
    cv.arrow((628, 470), (682, 470), TXT3, 3, alpha=A(2.2, 2.8), head=12)
    # S
    a = A(2.6, 3.6)
    card(cv, 690, Y0, 960, Y1, a, col=mix(S_C, BG0, 0.3))
    letter_dot(cv, 728, Y0 + 44, "S", S_C, 24, a, 26)
    cv.text(764, Y0 + 22, "Strategic", 30, TXT, "semi", alpha=a)
    cv.text(718, Y0 + 82, "route checkpoints", 24, TXT2, "reg", alpha=a)
    pf = lambda u: (815 + 95 * u * u + 8 * u, 585 - 205 * u)
    cv.line(poly_pts(pf), mix(S_C, BG0, 0.45), 3, alpha=a)
    cv.poly([(815, 570), (806, 594), (824, 594)], tint(F_C, 0.3), alpha=a)
    for k, (u, tt) in enumerate(((0.375, 3.4), (0.625, 3.9), (1.0, 4.4))):
        ak = A(tt, tt + 0.5)
        px, py = pf(u)
        cv.glow(px, py, 26, S_C, 0.5 * ak)
        cv.poly([(px, py - 13), (px + 11, py), (px, py + 13), (px - 11, py)], S_C, alpha=ak)
    cv.text(718, 622, "30 / 50 / 80 m", 26, TXT, "semi", alpha=A(4.2, 4.8))
    cv.text(718, 656, "about 1 Hz", 22, TXT3, "reg", alpha=A(4.2, 4.8))
    cv.arrow((968, 470), (1012, 470), TXT3, 3, alpha=A(5.0, 5.6), head=12)
    # T
    a = A(5.2, 6.2)
    card(cv, 1020, Y0, 1400, Y1, a, col=mix(T_C, BG0, 0.3))
    letter_dot(cv, 1058, Y0 + 44, "T", T_C, 24, a, 26)
    cv.text(1094, Y0 + 22, "Tactical", 30, TXT, "semi", alpha=a)
    cv.text(1048, Y0 + 82, "typed 60-d code", 24, TXT2, "reg", alpha=a)
    groups = [(22, (31, 111, 178), "22 goals"), (5, (66, 146, 214), "5 lateral classes"),
              (7, (112, 176, 230), "7 longitudinal classes"), (26, (166, 206, 240), "26 continuous constraints")]
    n, fillp = 0, A(6.0, 8.8) * 60
    for gi, (cnt, gc, _) in enumerate(groups):
        for _k in range(cnt):
            on = 1.0 if n < fillp else (clamp(fillp - n + 1) if n < fillp + 1 else 0.0)
            col = mix(CARD2, gc, 0.18 + 0.82 * on)
            x = 1048 + n * 5.55
            cv.rrect(x, 400, x + 4.5, 446, 2, fill=col, alpha=a)
            n += 1
    off = 0
    for gi, (cnt, gc, lab) in enumerate(groups):
        ag = A(6.6 + gi * 0.7, 7.3 + gi * 0.7)
        yy = 478 + gi * 36
        cv.rrect(1048, yy + 6, 1066, yy + 22, 4, fill=gc, alpha=ag)
        cv.text(1080, yy + 14, lab, 22, TXT, "reg", alpha=ag, anchor="lm")
    cv.text(1048, 656, "about 2 Hz \u00b7 the action for O", 22, TXT3, "reg", alpha=A(9.0, 9.6) * a)
    cv.arrow((1408, 470), (1452, 470), TXT3, 3, alpha=A(9.0, 9.6), head=12)
    # O
    a = A(9.2, 10.2)
    card(cv, 1460, Y0, 1810, Y1, a, col=mix(O_C, BG0, 0.3))
    letter_dot(cv, 1498, Y0 + 44, "O", O_C, 24, a, 26)
    cv.text(1534, Y0 + 22, "Operative", 30, TXT, "semi", alpha=a)
    cv.text(1488, Y0 + 82, "6 s trajectory \u00b7 10 Hz", 24, TXT2, "reg", alpha=a)
    prior = lambda u: (1500 + 270 * u, 510 - 120 * u * u - 30 * u)
    resid = lambda u: 60 * math.sin(math.pi * min(1, u) * 0.9) * u
    learned = lambda u: (prior(u)[0], prior(u)[1] - resid(u))
    pp = poly_pts(prior, 30)
    for i in range(0, 29, 2):
        cv.line([pp[i], pp[i + 1]], tint(F_C, 0.15), 3.5, alpha=a, cap=False)
    fr = A(10.4, 12.6)
    if fr > 0.02:
        cv.line(poly_pts(lambda u: learned(u * fr), 40), O_C, 5, alpha=a)
        ex, ey = learned(fr)
        cv.glow(ex, ey, 20, O_C, 0.6)
    ar = A(12.2, 13.0)
    for u in (0.4, 0.65, 0.9):
        p0, p1 = prior(u), learned(u)
        cv.arrow(p0, (p0[0] + (p1[0] - p0[0]), p0[1] + (p1[1] - p0[1])), tint(O_C, 0.4), 2.5, alpha=ar, head=9)
    cv.para(1488, 532, "residual on a kinematic prior (P10)", 22, TXT2, "reg", 290, alpha=A(10.8, 11.6))
    cv.para(1488, 604, "zero residual = the prior exactly, bit for bit", 22, MINT, "semi", 290, alpha=A(13.0, 13.8))
    # band, inputs, counters
    a = A(11.0, 12.0)
    cv.rrect(350, 716, 1810, 762, 23, fill=mix(MINT, BG0, 0.9), outline=mix(MINT, BG0, 0.45), width=1.6, alpha=a)
    cv.text(1080, 739, "plus an H-JEPA world-model auxiliary on the shared adapter", 24, tint(MINT, 0.25), "semi", alpha=a, anchor="mm")
    cv.text(110, 790, "Inputs at inference: front-wide camera, measured ego motion including yaw rate, nav, speed bucket", 24, TXT2, "reg", alpha=A(12.0, 13.0))
    vals = [(7.2, "{:.1f} M", "trainable parameters"), (4, "\u2248 {:.0f} h", "training on one 8 GB consumer GPU"),
            (4512, "{:,.0f}", "training clips \u00b7 141 held-out episodes")]
    for i, (v, f_, cap) in enumerate(vals):
        a = A(13.2 + 0.25 * i, 14.0 + 0.25 * i)
        k = ep(t, 13.4 + 0.25 * i, 15.4 + 0.25 * i)
        xc = 330 + i * 630
        cv.text(xc, 842, f_.format(v * k), 78, TXT, "bold", alpha=a, anchor="ma")
        cv.text(xc, 940, cap, 25, TXT2, "reg", alpha=a, anchor="ma")


# ================================================================= 5 seeing in layers
PW_, PH_ = 562, 223
_att = {}


def att(row, p, i):
    k = (row, p, i)
    if k not in _att:
        _att[k] = Image.open(f"{ATT}/{row}/{p}_{i:03d}.png").convert("RGB").resize((PW_ * SS, PH_ * SS), Image.LANCZOS)
    return _att[k]


def att_frame(row, p, pos, n=48):
    i = int(max(0, min(n - 1, pos)))
    f = max(0.0, min(1.0, pos - i)) if i < n - 1 else 0.0
    a = att(row, p, i)
    return a if f < 0.02 else Image.blend(a, att(row, p, i + 1), f)


def s5(cv, t):
    kicker(cv, t, "05  \u00b7  SEEING IN LAYERS")
    headline(cv, t, "What each level looks at")
    xs = [62, 677, 1291]
    meta = [("S", "Strategic", "about 1 Hz \u00b7 checkpoints, 6\u201330 s", S_C), ("T", "Tactical", "about 2 Hz \u00b7 goals, constraints, up to 6 s", T_C),
            ("O", "Operative", "10 Hz \u00b7 control, up to 2 s", O_C)]
    a0 = ep(t, 0.5, 1.3)
    for j, (ch, nm, sub, col) in enumerate(meta):
        cv.text(xs[j], 196, f"{ch}  {nm}", 30, tint(col, 0.4), "bold", alpha=a0)
        cv.text(xs[j] + 560, 200, sub, 22, TXT2, "reg", alpha=a0, anchor="ra")
    pos = max(0.0, (t - 0.8)) * 4.0
    for r, row in enumerate(("A", "B")):
        y = 252 + r * 258
        a = ep(t, 0.8 + 0.4 * r, 1.6 + 0.4 * r)
        for j, (ch, nm, sub, col) in enumerate(meta):
            fr = att_frame(row, ch, pos)
            cv.rrect(xs[j] - 3, y - 3, xs[j] + PW_ + 3, y + PH_ + 3, 12, fill=col, alpha=a)
            cv.image(fr, xs[j], y, alpha=a, radius=10)
    cv.text(960, 790, "Each level attends to its own abstraction", 50, TXT, "semi", alpha=ep(t, 2.2, 3.2), anchor="ma")
    cv.text(960, 862, "Attention overlays from the model on two example clips", 26, TXT2, "reg", alpha=ep(t, 3.0, 4.0), anchor="ma")
    tier_chip(cv, 960, 932, "T0 DIAGNOSTIC", "teacher-forced on the true past \u00b7 qualitative view, not an explanation", ep(t, 3.4, 4.4), anchor="m")


# ================================================================= 6 results
def dots(cv, k, n=5, a=1.0):
    for i in range(n):
        cv.circle(1810 - (n - 1 - i) * 26, 84, 8 if i == k else 5, fill=tint(MINT, 0.1) if i == k else LINE, alpha=a)


def r_head(cv, u, a, tag, text, sub=None):
    cv.text(110, 150, tag, 22, tint(MINT, 0.1), "bold", alpha=ep(u, 0.15, 0.7) * a, tracking=3)
    k = ep(u, 0.2, 0.9)
    cv.text(110, 184 + 10 * (1 - k), text, 48, TXT, "semi", alpha=k * a)
    if sub:
        cv.text(110, 252, sub, 25, TXT2, "reg", alpha=ep(u, 0.5, 1.2) * a)


def r1(cv, u, a):
    r_head(cv, u, a, "K-1", "Beats the learned kinematic floor on 6 s error", "6 s ADE improvement of M2b over the frozen kinematic prior P10, two training seeds")
    x0, sc = 330, 1500.0
    vals = [("Seed 0", 0.312), ("Seed 1", 0.432)]
    cv.line([(x0, 330), (x0, 590)], LINE, 2, alpha=a)
    for i, (nm, v) in enumerate(vals):
        y = 360 + i * 120
        k = ep(u, 0.9 + 0.4 * i, 2.3 + 0.4 * i)
        cv.text(x0 - 24, y + 36, nm, 28, TXT2, "semi", alpha=a, anchor="rm")
        pbar(cv, x0, y, v * sc, 72, k, S_C, a, 8)
        cv.text(x0 + v * sc * k + 20, y + 36, f"{MINUS}{v * k:.2f} m", 44, TXT, "bold", alpha=a * clamp(k * 3), anchor="lm")
    bx = x0 + 0.2415 * sc
    ab = ep(u, 2.4, 3.2) * a
    cv.dashed((bx, 332), (bx, 600), tint(MINT, 0.1), 3, 12, 8, alpha=ab)
    cv.text(bx, 300, "pre-registered bar 0.2415 m", 24, tint(MINT, 0.2), "semi", alpha=ab, anchor="ma")
    ks = ep(u, 3.3, 4.2) * a
    cv.rrect(1250, 340, 1810, 600, 20, fill=CARD, outline=mix(MINT, BG0, 0.5), width=2, alpha=ks)
    cv.text(1530, 392, "12.5–17.4 %", 78, tint(MINT, 0.1), "bold", alpha=ks, anchor="ma")
    cv.text(1530, 500, "lower 6 s error", 36, TXT, "semi", alpha=ks, anchor="ma")
    cv.text(1530, 550, "against the kinematic floor", 24, TXT2, "reg", alpha=ks, anchor="ma")
    cv.text(x0, 690, "Both seeds separated, and beyond the pre-registered bar.", 28, TXT2, "reg", alpha=ep(u, 3.8, 4.7) * a)
    tier_chip(cv, 110, 950, "OPEN LOOP", "the model's own held outputs", ep(u, 1.0, 1.8) * a)


def r2(cv, u, a):
    r_head(cv, u, a, "HACC", "Better than the floor on every row of the pre-registered set", "A zero-training post-map, both training seeds")
    rows = [("ADE at 2 s", f"{MINUS}0.011 / {MINUS}0.008 m"), ("ADE at 6 s", f"{MINUS}0.50 / {MINUS}0.47 m"), ("Acceleration error", f"{MINUS}0.022 / {MINUS}0.040 m/s\u00b2"),
            ("Curvature", ""), ("Speed", ""), ("Heading", ""), ("Yaw rate", ""), ("Decision agreement", ""), ("Consistency", "")]
    for i, (nm, val) in enumerate(rows):
        r_, c_ = divmod(i, 3)
        x0, y0 = 200 + c_ * 510, 320 + r_ * 125
        k = ep(u, 0.7 + 0.33 * i, 1.3 + 0.33 * i)
        cv.rrect(x0, y0, x0 + 480, y0 + 100, 16, fill=CARD, outline=mix(MINT, BG0, 0.55), width=1.8, alpha=k * a)
        cv.circle(x0 + 48, y0 + 50, 24, fill=mix(MINT, BG0, 0.82), outline=MINT, width=2, alpha=k * a)
        cv.check(x0 + 48, y0 + 50, 28, MINT, 4.5, alpha=k * a, frac=ep(u, 0.9 + 0.33 * i, 1.5 + 0.33 * i))
        cv.text(x0 + 92, y0 + (30 if val else 50), nm, 30, TXT, "semi", alpha=k * a, anchor="lm")
        if val:
            cv.text(x0 + 92, y0 + 70, val, 24, TXT2, "reg", alpha=k * a, anchor="lm")
    cv.text(200, 725, "Nothing is separated-worse.", 32, TXT, "semi", alpha=ep(u, 4.2, 5.0) * a)
    cv.text(200, 772, "Distance keeping is not separated.", 22, TXT3, "reg", alpha=ep(u, 4.5, 5.3) * a)
    tier_chip(cv, 110, 950, "OPEN LOOP", "zero-training post-map, both seeds", ep(u, 1.0, 1.8) * a)


def r3(cv, u, a):
    r_head(cv, u, a, "T1-KIN", f"On its own executed motion: 6 s ADE {MINUS}1.26 / {MINUS}1.19 m", "Adopted base DAG-OACC against the kinematic floor P10, two training seeds")
    x0, sc = 330, 800.0
    cv.line([(x0, 320), (x0, 520)], LINE, 2, alpha=a)
    for i, (nm, v) in enumerate((("Seed 0", 1.2556), ("Seed 1", 1.1903))):
        y = 330 + i * 100
        k = ep(u, 0.9 + 0.4 * i, 2.4 + 0.4 * i)
        cv.text(x0 - 24, y + 34, nm, 28, TXT2, "semi", alpha=a, anchor="rm")
        pbar(cv, x0, y, v * sc, 68, k, T_C, a, 8)
        cv.text(x0 + v * sc * k + 20, y + 34, f"{MINUS}{v * k:.2f} m", 44, TXT, "bold", alpha=a * clamp(k * 3), anchor="lm")
    cv.text(x0, 570, "Four of four metric families improved somewhere", 34, TXT, "semi", alpha=ep(u, 2.6, 3.4) * a)
    for i, nm in enumerate(("Longitudinal", "Lateral", "Tactical", "Strategic")):
        k = ep(u, 3.0 + 0.4 * i, 3.7 + 0.4 * i)
        x = x0 + i * 330
        cv.rrect(x, 640, x + 300, 716, 38, fill=CARD, outline=mix(MINT, BG0, 0.5), width=1.8, alpha=k * a)
        cv.check(x + 44, 678, 26, MINT, 4.5, alpha=k * a, frac=ep(u, 3.2 + 0.4 * i, 3.9 + 0.4 * i))
        cv.text(x + 82, 678, nm, 28, TXT, "semi", alpha=k * a, anchor="lm")
    tier_chip(cv, 110, 950, "T1-KIN", "own executed motion, logged cameras", ep(u, 1.0, 1.8) * a)


def r4(cv, u, a):
    r_head(cv, u, a, "T1-KIN \u00b7 STRATEGIC", "Route checkpoints beat a kinematic arc by up to 8.8 m", "Advantage of S's route checkpoints over a kinematic arc, in metres, two training seeds")
    base, sc = 790, 42.0
    cv.line([(250, base), (1670, base)], LINE, 2, alpha=a)
    data = [("30 m", 1.2, 1.3), ("50 m", 3.3, 3.5), ("80 m", 7.8, 8.8)]
    for g, (lab, v0, v1) in enumerate(data):
        xc = 470 + g * 490
        k = ep(u, 0.9 + 0.7 * g, 2.3 + 0.7 * g)
        for s_, v in enumerate((v0, v1)):
            x = xc - 110 + s_ * 120
            h_ = v * sc * k
            if h_ > 3:
                cv.rrect(x, base - h_, x + 100, base, 8, fill=mix(S_C, tint(S_C, 0.35), s_ * 0.6), alpha=a)
            cv.text(x + 50, base - h_ - 12, f"+{v * k:.1f}", 32, TXT, "bold", alpha=a * clamp(k * 3), anchor="ms")
        cv.text(xc, base + 18, lab, 30, TXT2, "semi", alpha=a, anchor="ma")
    cv.rrect(250, 322, 270, 338, 4, fill=S_C, alpha=a)
    cv.text(282, 330, "seed 0", 22, TXT2, "reg", alpha=a, anchor="lm")
    cv.rrect(380, 322, 400, 338, 4, fill=tint(S_C, 0.35), alpha=a)
    cv.text(412, 330, "seed 1", 22, TXT2, "reg", alpha=a, anchor="lm")
    cv.text(1670, 868, "checkpoint distance ahead", 22, TXT3, "reg", alpha=a, anchor="ra")
    tier_chip(cv, 110, 950, "T1-KIN", "own executed motion, logged cameras", ep(u, 1.0, 1.8) * a)


def r5(cv, u, a):
    r_head(cv, u, a, "INTERFACE", "A typed interface carries about 1.1 m of headroom", "6 s ADE of the operative level, with and without the ground-truth tactical code")
    base, sc = 840, 170.0
    cv.line([(300, base), (1300, base)], LINE, 2, alpha=a)
    for i, (lab, v, col) in enumerate((("O with its own tactical code", 2.1, F_C), ("O fed the ground-truth tactical code", 1.0, T_C))):
        k = ep(u, 0.9 + 0.8 * i, 2.4 + 0.8 * i)
        x = 400 + i * 400
        h_ = v * sc * k
        cv.rrect(x, base - h_, x + 220, base, 10, fill=col, alpha=a)
        cv.text(x + 110, base - h_ - 14, f"about {v * k:.1f} m", 34, TXT, "bold", alpha=a * clamp(k * 3), anchor="ms")
        cv.para(x + 110, base + 14, lab, 24, TXT2, "reg", 340, alpha=a, anchor="m")
    kb = ep(u, 2.8, 3.8)
    y_top, y_bot = base - 2.1 * sc, base - 1.0 * sc
    cv.dashed((620, y_top), (1230, y_top), TXT3, 2, 10, 8, alpha=kb * a)
    cv.arrow((1190, y_top), (1190, y_top + (y_bot - y_top) * kb), tint(MINT, 0.1), 5, alpha=a, head=18)
    cv.text(1240, (y_top + y_bot) / 2, "\u2248 1.1 m", 60, tint(MINT, 0.1), "bold", alpha=kb * a, anchor="lm")
    cv.text(1240, (y_top + y_bot) / 2 + 52, "of headroom through the interface", 24, TXT2, "reg", alpha=ep(u, 3.4, 4.2) * a, anchor="lm")
    tier_chip(cv, 110, 950, "DIAGNOSTIC \u00b7 T0", "ground-truth tactical code fed to O (oracle by interface)", ep(u, 1.0, 1.8) * a)


def s6(cv, t):
    kicker(cv, t, "06  \u00b7  RESULTS  \u00b7  MEASURED")
    segs = [(0, 6.0, r1), (6.0, 11.4, r2), (11.4, 17.4, r3), (17.4, 22.9, r4), (22.9, 28.0, r5)]
    for k, (a_, b_, fn) in enumerate(segs):
        if a_ <= t < b_ + 0.01:
            u = t - a_
            al = ep(u, 0, 0.35) * (1 - ep(u, (b_ - a_) - 0.35, b_ - a_) if k < 4 else 1.0)
            if k == 0:
                al = 1.0 * (1 - ep(u, (b_ - a_) - 0.35, b_ - a_))
            dots(cv, k)
            fn(cv, u, al)
            break


# ================================================================= 7 rigour
def glyph(cv, kind, cx, cy, s, c, a):
    if kind == 0:   # fork: both outcomes
        cv.line([(cx - 0.45 * s, cy), (cx - 0.05 * s, cy)], c, 4, a)
        cv.line([(cx - 0.05 * s, cy), (cx + 0.4 * s, cy - 0.32 * s)], c, 4, a)
        cv.line([(cx - 0.05 * s, cy), (cx + 0.4 * s, cy + 0.32 * s)], c, 4, a)
        for dy in (-0.32, 0.32):
            cv.circle(cx + 0.45 * s, cy + dy * s, 0.09 * s, fill=c, alpha=a)
        cv.circle(cx - 0.5 * s, cy, 0.09 * s, fill=c, alpha=a)
    elif kind == 1:  # lock
        cv.rrect(cx - 0.32 * s, cy - 0.05 * s, cx + 0.32 * s, cy + 0.42 * s, 8, fill=c, alpha=a)
        cv.arc(cx, cy - 0.05 * s, 0.22 * s, 180, 360, c, 5, a)
        cv.circle(cx, cy + 0.2 * s, 0.07 * s, fill=BG0, alpha=a)
    elif kind == 2:  # can fail: circle with cross
        cv.circle(cx, cy, 0.42 * s, outline=c, width=4, alpha=a)
        cv.line([(cx - 0.2 * s, cy - 0.2 * s), (cx + 0.2 * s, cy + 0.2 * s)], c, 4, a)
        cv.line([(cx - 0.2 * s, cy + 0.2 * s), (cx + 0.2 * s, cy - 0.2 * s)], c, 4, a)
    elif kind == 3:  # three overlapping distributions
        for j, off in enumerate((-0.26, 0.0, 0.26)):
            pts = [(cx + off * s + (i / 20 - 0.5) * 0.7 * s, cy + 0.35 * s - 0.6 * s * math.exp(-((i / 20 - 0.5) * 4.2) ** 2)) for i in range(21)]
            cv.line(pts, mix(c, tint(c, 0.4), j / 2), 3.5, a)
    else:            # four bars
        for j, (hh, col) in enumerate(((0.55, S_C), (0.8, T_C), (0.45, O_C), (0.65, F_C))):
            x = cx + (j - 1.5) * 0.28 * s
            cv.rrect(x - 0.09 * s, cy + 0.4 * s - hh * s, x + 0.09 * s, cy + 0.4 * s, 4, fill=col, alpha=a)


def s7(cv, t):
    kicker(cv, t, "07  \u00b7  RIGOUR")
    headline(cv, t, "Rigour as a moat")
    items = [("Pre-registered", "Every arm commits both outcomes before it sees any data."),
             ("Launch gates", "Each training run is bound to its exact commit, argv and data hashes."),
             ("Deliberate-regression arms", "They prove that every check can fail."),
             ("Episode-cluster bootstrap and seed replicates", "Three variance questions (episodes, training seeds, inference); each claim names which one it answers."),
             ("Four metric families, never one score", "Longitudinal, lateral, tactical and strategic are reported on every evaluation.")]
    cols = [T_C, S_C, O_C, T_C, S_C]
    for i, (h_, d_) in enumerate(items):
        a = st(t, i, 1.0, 1.0, 0.8)
        y = 215 + i * 142 + 18 * (1 - a)
        card(cv, 110, y, 1810, y + 122, a)
        cv.circle(200, y + 61, 44, fill=mix(cols[i], BG0, 0.78), outline=cols[i], width=2.5, alpha=a)
        glyph(cv, i, 200, y + 61, 62, tint(cols[i], 0.4), a)
        cv.text(280, y + 22, h_, 36, TXT, "bold", alpha=a)
        cv.text(280, y + 72, d_, 25, TXT2, "reg", alpha=a)
    cv.text(110, 955, "An append-only log records every revised claim with its root-cause class.", 26, TXT2, "reg", alpha=ep(t, 6.8, 7.8))


# ================================================================= 8 frontier + next step
def s8(cv, t):
    kicker(cv, t, "08  \u00b7  THE FRONTIER AND THE NEXT STEP")
    headline(cv, t, "One open behaviour, and a new kind of evidence")
    a = ep(t, 0.9, 1.7)
    cv.para(110, 215, "One behaviour remains: assertive launch from standstill", 38, TXT, "semi", 800, alpha=a)
    cv.para(110, 320, "Every zero-training lever has been measured; the analysis attributes the remainder to what logged cameras can show.", 25, TXT2, "reg", 800, alpha=ep(t, 1.5, 2.3))
    # schematic speed curves
    ac = ep(t, 2.0, 2.8)
    X0, X1, Y0, Y1 = 130, 860, 480, 780
    cv.line([(X0, Y0), (X0, Y1), (X1, Y1)], LINE, 2.5, alpha=ac)
    cv.text(X0 + 10, Y0 - 4, "speed", 22, TXT3, "reg", alpha=ac)
    cv.text(X1, Y1 + 10, "time from standstill", 22, TXT3, "reg", alpha=ac, anchor="ra")
    sg = lambda x, k, t0: 1 / (1 + math.exp(-k * (x - t0)))
    kk = ep(t, 2.6, 5.6)
    for nm, k_, t0, col, lab, ly in (("human", 11, 0.3, TXT, "human", 0), ("model", 7, 0.5, O_C, "model", 1)):
        pts = []
        for i in range(60):
            x = i / 59 * kk
            v = (sg(x, k_, t0) - sg(0, k_, t0)) / (1 - sg(0, k_, t0)) if True else 0
            pts.append((X0 + x * (X1 - X0 - 30), Y1 - 12 - v * (Y1 - Y0 - 60)))
        if kk > 0.03:
            cv.line(pts, col, 5, alpha=ac)
        cv.text(X1 - 20, Y0 + 28 + ly * 40, lab, 26, tint(col, 0.1) if ly == 0 else tint(O_C, 0.3), "semi", alpha=ep(t, 5.0, 5.8))
    cv.text(X0 + 10, Y1 + 46, "schematic: pulling away from near-standstill", 22, TXT3, "reg", alpha=ac)
    # right panel
    ar = ep(t, 3.0, 3.9)
    badge(cv, 1000, 205, "FIRST READS", O_C, ar)
    cv.para(1000, 275, "Re-rendered simulation on neural reconstructions", 38, TXT, "semi", 810, alpha=ar)
    cv.para(1000, 375, "Logged cameras cannot show a car its own consequences. Re-rendered scenes can.", 26, TXT2, "reg", 800, alpha=ep(t, 3.6, 4.4))
    cv.text(1000, 472, "62 scenes selected by a rule frozen before any log was read", 24, TXT2, "reg", alpha=ep(t, 4.2, 5.0))
    kg = ep(t, 5.0, 8.4) * 62
    for i in range(62):
        r_, c_ = divmod(i, 16)
        x, y = 1000 + c_ * 51, 520 + r_ * 51
        on = clamp(kg - i)
        col = O_C if i < 42 else tint(F_C, 0.3)
        cv.rrect(x, y, x + 40, y + 40, 8, fill=mix(CARD2, col, 0.15 + 0.85 * on), alpha=1.0)
    al = ep(t, 8.2, 9.0)
    cv.rrect(1000, 742, 1018, 758, 4, fill=O_C, alpha=al)
    cv.text(1030, 750, "42 launch", 24, TXT, "reg", alpha=al, anchor="lm")
    cv.rrect(1180, 742, 1198, 758, 4, fill=tint(F_C, 0.3), alpha=al)
    cv.text(1210, 750, "20 guard scenes, staged", 24, TXT, "reg", alpha=al, anchor="lm")
    cv.text(1000, 800, "The harness reproduces the logged-camera plans bit for bit.", 25, TXT2, "reg", alpha=ep(t, 9.2, 10.0))
    tier_chip(cv, 1000, 950, "T2 \u00b7 RE-RENDERED SIMULATION", "first reads on 48 scenes, replayed traffic", ep(t, 9.6, 10.4), color=O_C)
    tier_chip(cv, 110, 950, "T1-KIN", "logged cameras today", ep(t, 2.4, 3.2))


# ================================================================= 9 outlook
def glyph2(cv, k, cx, cy, s, c, a):
    if k == 0:   # eye
        cv.arc(cx, cy + 0.2 * s, 0.5 * s, 200, 340, c, 4, a)
        cv.arc(cx, cy - 0.2 * s, 0.5 * s, 20, 160, c, 4, a)
        cv.circle(cx, cy, 0.17 * s, fill=c, alpha=a)
    elif k == 1:  # growth bars
        for j in range(3):
            h_ = (0.3, 0.6, 1.0)[j]
            cv.rrect(cx + (j - 1) * 0.36 * s - 0.12 * s, cy + 0.5 * s - h_ * s, cx + (j - 1) * 0.36 * s + 0.12 * s, cy + 0.5 * s, 4, fill=c, alpha=a)
    elif k == 2:  # graph
        P = [(cx - 0.4 * s, cy + 0.3 * s), (cx, cy - 0.3 * s), (cx + 0.4 * s, cy + 0.25 * s), (cx + 0.05 * s, cy + 0.45 * s)]
        for i_, j_ in ((0, 1), (1, 2), (0, 3), (3, 2), (1, 3)):
            cv.line([P[i_], P[j_]], mix(c, BG0, 0.3), 3, a)
        for p in P:
            cv.circle(p[0], p[1], 0.1 * s, fill=c, alpha=a)
    elif k == 3:  # shield
        cv.poly([(cx, cy - 0.5 * s), (cx + 0.42 * s, cy - 0.3 * s), (cx + 0.36 * s, cy + 0.2 * s), (cx, cy + 0.5 * s), (cx - 0.36 * s, cy + 0.2 * s), (cx - 0.42 * s, cy - 0.3 * s)], mix(c, BG0, 0.55), a)
        cv.line([(cx, cy - 0.5 * s), (cx + 0.42 * s, cy - 0.3 * s), (cx + 0.36 * s, cy + 0.2 * s), (cx, cy + 0.5 * s), (cx - 0.36 * s, cy + 0.2 * s), (cx - 0.42 * s, cy - 0.3 * s), (cx, cy - 0.5 * s)], c, 3.5, a)
    else:
        ico(cv, 3, cx, cy, s * 1.0, c, a)


def s9(cv, t):
    kicker(cv, t, "09  \u00b7  OUTLOOK")
    headline(cv, t, "What more work unlocks")
    badge(cv, 1810, 60, "OUTLOOK", MINT, ep(t, 0.3, 1.0), anchor="r")
    items = [("Perception", "LoRA fine-tune of the visual encoder for motion cues.", "A40-class GPU, about 100 GPU-h per seed (estimated)", O_C),
             ("Scale", "More data and a larger model: from 7 M toward the 4B budget of at most 800 M.", "The corpus is far larger than the 4,512 clips used", MINT),
             ("Strategic level", "Checkpoints to 30 s, detected context such as country and weather, and derived goals.", "World-model planning is the baseline", S_C),
             ("Fallback brain", "Imagination-error self-monitoring and the minimal-risk manoeuvre.", "ISMR / DSSAD logging", F_C),
             ("Deployment", "Jetson Thor: at most 100 ms end to end with 4 cameras.", "Optimisation follows results", T_C)]
    for i, (h_, d_, x_, col) in enumerate(items):
        a = st(t, i, 1.0, 1.5, 0.9)
        x0 = 110 + i * 345
        y0 = 235 + 20 * (1 - a)
        cv.rrect(x0, y0, x0 + 320, y0 + 500, 18, fill=mix(col, BG0, 0.9), alpha=a)
        for e in (((x0, y0), (x0 + 320, y0)), ((x0 + 320, y0), (x0 + 320, y0 + 500)), ((x0 + 320, y0 + 500), (x0, y0 + 500)), ((x0, y0 + 500), (x0, y0))):
            cv.dashed(e[0], e[1], mix(col, BG0, 0.25), 2.5, 14, 9, alpha=a)
        glyph2(cv, i, x0 + 160, y0 + 80, 76, tint(col, 0.3), a)
        yy = cv.para(x0 + 28, y0 + 150, h_, 34, tint(col, 0.5), "bold", 270, alpha=a)
        yy = cv.para(x0 + 28, yy + 10, d_, 27, TXT, "reg", 270, alpha=a)
        if x_:
            cv.para(x0 + 28, yy + 14, x_, 24, TXT2, "reg", 270, alpha=a)
    cv.text(110, 790, "Also planned: calibrated imagination of occluded areas \u00b7 NavSim / navtest benchmarks \u00b7 T2 re-rendered-simulation reads", 24, TXT2, "reg", alpha=ep(t, 9.0, 10.0))
    cv.text(110, 846, "Each item is an outlook, not yet implemented.", 24, TXT3, "reg", alpha=ep(t, 9.6, 10.6))


# ================================================================= 10 close
def s10(cv, t):
    cv.text(110, 150, "WHY NOW", 24, tint(MINT, 0.1), "bold", alpha=ep(t, 0.2, 0.9), tracking=5)
    k = ep(t, 0.5, 1.5)
    cv.para(110, 215 + 12 * (1 - k), "More data, more compute, and re-rendered simulation turn a validated architecture into a driver.", 56, TXT, "semi", 1500, lh=1.25, alpha=k)
    for i, c in enumerate((S_C, T_C, O_C)):
        w = 150 * ep(t, 2.2 + 0.2 * i, 2.9 + 0.2 * i)
        if w > 3:
            cv.rrect(114 + i * 160, 492, 114 + i * 160 + w, 500, 4, fill=c)
    k = ep(t, 3.0, 4.2)
    cv.text(106, 560 + 18 * (1 - k), "TanitAD", 150, TXT, "bold", alpha=k)
    cv.text(112, 745, "Dr.-Ing. Sayed Bouzouraa", 40, TXT2, "light", alpha=ep(t, 4.2, 5.2))


SCENES = [("title", 6.0, s1), ("problem", 9.0, s2), ("idea4b", 16.0, s3), ("v8fmini", 17.0, s4), ("layers", 12.0, s5),
          ("results", 28.0, s6), ("rigour", 10.0, s7), ("frontier", 13.0, s8), ("outlook", 13.0, s9), ("close", 8.0, s10)]
