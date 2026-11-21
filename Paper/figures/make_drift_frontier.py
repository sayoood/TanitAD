"""Generate Figure 5 — the (drift, prediction) plane and the coupling hypothesis.

EVERY NUMBER IS READ FROM THE BANKED PROBE JSONs AT GENERATION TIME. Nothing is
typed in by hand: if a source artifact is missing the script exits loudly rather
than falling back to a literal, so the figure cannot drift from the measurement.
(Same contract as make_drift_story.py, which draws Figure 4.)

Sources (MEASURED, ours; all T0-DIAGNOSTIC):
  e4-drift-ladder raw/      e4_e4_l1_inno_drift.json, e4_e4_l1_inno_nrmse.json
                            (MM-E4 L1: innovation-SIGReg + its base o14fut10)
  o14-tiny-ladder raw/      o14drift.json, o14nrmse.json (the 2k ladder: the
                            same base o14fut10 read in a DIFFERENT package,
                            o14base2k, and the ladder's DR control o14dr10),
                            o14fut30k_*.json / ema30k_*.json (the 30k pair)
  p0-ema-bakeoff raw/       p0_drift.json, p0_nrmse.json (MM-E1 stage 1: the
                            2k EMA pair + the same 2k incumbent)

Cross-artifact tripwires (the figure refuses to draw on disagreement):
  * o14fut10's drift must agree between the O14 ladder's and the drift-ladder's
    raws (same arm, same instrument, two packages, two nights);
  * o14fut10's centred cosine likewise;
  * o14base2k's drift must agree between the ladder's and the bake-off's raws;
  * the 30k incumbent's centred cosine must agree between its own raw and the
    EMA pair's raw.

⛔ SCALE HONESTY IS THE FIGURE'S MAIN DESIGN CONSTRAINT. A 2k drift number and a
30k drift number are NOT commensurable (the rig's drift-validity band is
30k-calibrated; 2k arms are admissible vs-base RELATIVE only), so:
  * the plane is drawn TWICE, once per scale, and never once for both;
  * the two panels do NOT share axes, and each panel prints its own spans;
  * every intervention is drawn as an ARROW FROM ITS OWN BASE TO ITSELF — the
    one-variable delta is the only comparison the design licenses.

Design notes (dataviz discipline, as make_drift_story.py):
  * form first — the story is "the two measured deltas agree in SIGN on both
    axes", so the encoding is a vector field of within-pair arrows, not a
    scatter of levels.
  * the hypothesis is drawn as SHADED QUADRANTS at each base point, never as a
    fitted curve: sign agreement is the entire content of the coupling claim,
    and a drawn frontier line would assert a slope no measurement supports.
  * colour by job — ARM (blue) = the decision-grade intervention, CTRL (orange)
    = the base/incumbent it is one variable from, GATE (violet) = the
    motivation-only 2k EMA arms and pending/registered elements.
    Palette #2a78d6,#eb6834,#4a3aa7 — validated for the paper's figures.
  * identity is never colour-alone: every series is direct-labelled.
  * derived percentages are arithmetic on loaded values, never literals.

Honesty notes carried on the figure itself:
  * the 2k EMA arms are MOTIVATION-ONLY (off the published tau operating map in
    both directions, §14.7) and are NOT counted in the hypothesis's n = 2; they
    are drawn muted so a reader cannot mistake them for decision-grade points.
  * o14dr10 is the O14 ladder's deliberate-regression arm — a DIFFERENT cell's
    control, drawn only to show the plane's inert scale, and labelled as such.
    L1's OWN shuffle control is registered but not yet read.
  * panel C's dose axis is CATEGORICAL: the four registered weights are
    0 / 0.01 / 0.03 / 0.1 and only two of them have been measured.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
AI = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference")
E4 = os.path.join(AI, "Implementation", "incoming",
                  "2026-08-29-e4-drift-ladder", "raw")
LADDER = os.path.join(AI, "Implementation", "incoming",
                      "2026-08-27-o14-tiny-ladder", "raw")
P0 = os.path.join(AI, "Implementation", "incoming",
                  "2026-08-28-p0-ema-bakeoff", "raw")

W, H = 1240, 792
SURFACE = "#fcfcfb"
CARD = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"
EDGE = "#dedcd6"
ARM = "#2a78d6"      # categorical slot 1 — the decision-grade intervention
CTRL = "#eb6834"     # categorical slot 2 — the base it is one variable from
GATE = "#4a3aa7"     # violet — motivation-only / registered-but-unrun
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"


def esc(s):
    return html.escape(str(s))


def load(base, name):
    p = os.path.join(base, name)
    if not os.path.exists(p):
        sys.exit(f"missing source artifact: {p}\n"
                 "This figure is generated FROM the banked probe JSONs; it has "
                 "no hand-typed fallback by design.")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def drift_of(doc, arm):
    return doc["arms"][arm]["columns"]["z_t (DRIFT / POSITIVE CONTROL)"]["r"]


def cos_of(doc, arm):
    return doc["arms"][arm]["cos_centred"]


# ---------------------------------------------------------------- the literals
E4D = load(E4, "e4_e4_l1_inno_drift.json")    # L1 + its base, drift
E4N = load(E4, "e4_e4_l1_inno_nrmse.json")    # L1 + its base, cos/nrmse
LD = load(LADDER, "o14drift.json")            # the 2k ladder, drift
LN = load(LADDER, "o14nrmse.json")            # the 2k ladder, cos
F30D = load(LADDER, "o14fut30k_drift.json")   # 30k incumbent drift
F30N = load(LADDER, "o14fut30k_nrmse.json")   # 30k incumbent cos
E30D = load(LADDER, "ema30k_drift.json")      # 30k EMA drift
E30N = load(LADDER, "ema30k_nrmse.json")      # 30k EMA cos (+ incumbent cos)
P0D = load(P0, "p0_drift.json")               # 2k EMA pair drift (+ base)
P0N = load(P0, "p0_nrmse.json")               # 2k EMA pair cos (+ base)

# panel A — 2k
BASE_D, BASE_C = drift_of(E4D, "o14fut10"), cos_of(E4N, "o14fut10")
L1_D, L1_C = drift_of(E4D, "e4_l1_inno"), cos_of(E4N, "e4_l1_inno")
L1_NRMSE = E4N["arms"]["e4_l1_inno"]["nrmse"]
L1_MEANONLY = E4N["arms"]["e4_l1_inno"]["nrmse_meanonly_control"]
L1_MEANFRAC = E4N["arms"]["e4_l1_inno"]["mean_fraction_of_prediction"]
L1_Z = E4N["arms"]["e4_l1_inno"]["z_centred"]
BASE_Z = E4N["arms"]["o14fut10"]["z_centred"]
INC2K_D, INC2K_C = drift_of(LD, "o14base2k"), cos_of(LN, "o14base2k")
DR_D, DR_C = drift_of(LD, "o14dr10"), cos_of(LN, "o14dr10")
EMA2K = [(drift_of(P0D, a), cos_of(P0N, a)) for a in ("ema2k_s0", "ema2k_s1")]

# panel B — 30k
INC30_D, INC30_C = drift_of(F30D, "o14fut30k"), cos_of(F30N, "o14fut30k")
EMA30_D, EMA30_C = drift_of(E30D, "emao14_30k"), cos_of(E30N, "emao14_30k")

# cross-artifact tripwires: same quantity, two independent packages.
for a, b, what in ((BASE_D, drift_of(LD, "o14fut10"),
                    "base o14fut10 drift (drift-ladder vs O14 ladder)"),
                   (BASE_C, cos_of(LN, "o14fut10"),
                    "base o14fut10 cos (drift-ladder vs O14 ladder)"),
                   (INC2K_D, drift_of(P0D, "o14base2k"),
                    "2k incumbent drift (ladder vs bake-off)"),
                   (INC30_C, cos_of(E30N, "o14fut30k"),
                    "30k incumbent cos (own raw vs EMA-pair raw)")):
    if abs(a - b) > 5e-5:
        sys.exit(f"{what} disagrees across artifacts: {a} vs {b} — do not "
                 "draw the figure until this is resolved.")


def pct(new, old):
    return f"{(new / old - 1.0) * 100.0:+.1f} %"


def pctv(new, old):
    return (new / old - 1.0) * 100.0


# ------------------------------------------------------------------ primitives
def card(ox, oy, w, h, title, sub):
    o = [f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="7" '
         f'fill="{CARD}" stroke="{EDGE}" stroke-width="1"/>',
         f'<text x="{ox+16}" y="{oy+26}" font-size="14" font-weight="700" '
         f'fill="{INK}">{esc(title)}</text>']
    for i, line in enumerate(sub):
        o.append(f'<text x="{ox+16}" y="{oy+43+i*14}" font-size="10.2" '
                 f'fill="{INK2 if i == 0 else MUTED}">{esc(line)}</text>')
    return o


def halo(x, y, txt, size=10.6, weight="700", fill=INK, anchor="start"):
    """Direct label with a card-coloured halo drawn as a SEPARATE underlay
    (paint-order='stroke' is not honoured by every rasteriser)."""
    common = (f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
              f'font-weight="{weight}" text-anchor="{anchor}"')
    return (f'<text {common} fill="{CARD}" stroke="{CARD}" stroke-width="3.4" '
            f'stroke-linejoin="round">{esc(txt)}</text>'
            f'<text {common} fill="{fill}">{esc(txt)}</text>')


def arrow(x1, y1, x2, y2, col, width=2.0, dash=None, head=7.0):
    """A line with an EXPLICIT polygon head (markers are not universally
    honoured by rasterisers — the same reason halo() underlays its stroke)."""
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    px, py = -math.sin(ang) * head * 0.42, math.cos(ang) * head * 0.42
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return [f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="{col}" stroke-width="{width}"{d}/>',
            f'<polygon points="{x2:.1f},{y2:.1f} {bx+px:.1f},{by+py:.1f} '
            f'{bx-px:.1f},{by-py:.1f}" fill="{col}"/>']


def plane(ox, oy, w, h, title, sub, pts, arrows, xlo, xhi, ylo, yhi,
          quad_at, foot):
    """One (drift, cos) plane at ONE scale. Axes are per-panel by design."""
    o = card(ox, oy, w, h, title, sub)
    px, py = ox + 54, oy + 124
    pw, ph = w - 54 - 24, h - 124 - 76

    def sx(v):
        return px + (v - xlo) / (xhi - xlo) * pw

    def sy(v):
        return py + ph - (v - ylo) / (yhi - ylo) * ph

    # the quadrants the coupling hypothesis predicts — SIGN ONLY, no slope.
    qx, qy = sx(quad_at[0]), sy(quad_at[1])
    for x0, y0, ww, hh in ((qx, py, px + pw - qx, qy - py),
                           (px, qy, qx - px, py + ph - qy)):
        if ww > 0 and hh > 0:
            o.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{ww:.1f}" '
                     f'height="{hh:.1f}" fill="{GATE}" fill-opacity="0.075"/>')
    o.append(f'<line x1="{px}" y1="{qy:.1f}" x2="{px+pw}" y2="{qy:.1f}" '
             f'stroke="{GATE}" stroke-width="0.9" stroke-dasharray="2 3" '
             f'stroke-opacity="0.55"/>')
    o.append(f'<line x1="{qx:.1f}" y1="{py}" x2="{qx:.1f}" y2="{py+ph}" '
             f'stroke="{GATE}" stroke-width="0.9" stroke-dasharray="2 3" '
             f'stroke-opacity="0.55"/>')

    for i in range(5):
        gv = xlo + (xhi - xlo) * i / 4
        o.append(f'<line x1="{sx(gv):.1f}" y1="{py}" x2="{sx(gv):.1f}" '
                 f'y2="{py+ph}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{sx(gv):.1f}" y="{py+ph+15}" font-size="9.6" '
                 f'text-anchor="middle" fill="{MUTED}">{gv:.2f}</text>')
        gh = ylo + (yhi - ylo) * i / 4
        o.append(f'<line x1="{px}" y1="{sy(gh):.1f}" x2="{px+pw}" '
                 f'y2="{sy(gh):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{px-7}" y="{sy(gh)+3.4:.1f}" font-size="9.6" '
                 f'text-anchor="end" fill="{MUTED}">{gh:.2f}</text>')
    o.append(f'<text x="{px+pw*0.5:.1f}" y="{py+ph+31}" font-size="10.2" '
             f'text-anchor="middle" fill="{INK2}">drift r  '
             f'← less self-prediction</text>')
    o.append(f'<text x="{px-40}" y="{py+ph/2:.1f}" font-size="10.2" '
             f'fill="{INK2}" transform="rotate(-90 {px-40} {py+ph/2:.1f})" '
             f'text-anchor="middle">cos (centred) → better</text>')

    for (x1, y1), (x2, y2), col, wd, dash in arrows:
        o += arrow(sx(x1), sy(y1), sx(x2), sy(y2), col, wd, dash)
    for x, y, r, col, lab, dx, dy, anc, sz in pts:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r}" '
                 f'fill="{col}" stroke="{CARD}" stroke-width="1.8"/>')
        if lab:
            o.append(halo(sx(x) + dx, sy(y) + dy, lab, size=sz, fill=col,
                          anchor=anc))
    for i, line in enumerate(foot):
        o.append(f'<text x="{ox+16}" y="{oy+h-34+i*13}" font-size="9.2" '
                 f'fill="{MUTED}">{esc(line)}</text>')
    return o


# --------------------------------------------------------------------- panel C
def panel_c(ox, oy, w, h):
    dd, dc = pctv(L1_D, BASE_D), pctv(L1_C, BASE_C)
    o = card(ox, oy, w, h, "The registered discriminating tests",
             ["MM-E5 (PREREG_DRIFT_PREDICTION_FRONTIER.md, 6e719a493 —",
              "registered BEFORE the discriminating arms reported)",
              "test 2: a DOSE sweep of the one mechanism — four weights,",
              "two measured, two registered and not yet run"])
    px, py = ox + 52, oy + 116
    pw, ph = w - 52 - 22, 150
    xs = [px + pw * f for f in (0.08, 0.36, 0.64, 0.92)]
    ylo, yhi = -105.0, 12.0

    def sy(v):
        return py + ph - (v - ylo) / (yhi - ylo) * ph

    for gv in (0, -25, -50, -75, -100):
        o.append(f'<line x1="{px}" y1="{sy(gv):.1f}" x2="{px+pw}" '
                 f'y2="{sy(gv):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{px-7}" y="{sy(gv)+3.4:.1f}" font-size="9.6" '
                 f'text-anchor="end" fill="{MUTED}">{gv:+d}%</text>')
    for x, lab in zip(xs, ("0\n(base)", "0.01", "0.03", "0.1\n(L1)")):
        first = lab.split("\n")
        o.append(f'<text x="{x:.1f}" y="{py+ph+15}" font-size="9.8" '
                 f'text-anchor="middle" fill="{INK2}">{first[0]}</text>')
        if len(first) > 1:
            o.append(f'<text x="{x:.1f}" y="{py+ph+26}" font-size="8.6" '
                     f'text-anchor="middle" fill="{MUTED}">{first[1]}</text>')
    o.append(f'<text x="{px+pw*0.5:.1f}" y="{py+ph+42}" font-size="9.8" '
             f'text-anchor="middle" fill="{INK2}">innovation weight '
             f'--w-o6 · CATEGORICAL spacing</text>')

    # The two NOT-YET-RUN doses are drawn as ABSENCE — a vertical marker and
    # nothing else. A point drawn at any height would assert a value we have
    # not measured, which is the one thing this panel must not do.
    for x in xs[1:3]:
        o.append(f'<line x1="{x:.1f}" y1="{py-4}" x2="{x:.1f}" '
                 f'y2="{py+ph}" stroke="{GATE}" stroke-width="1.1" '
                 f'stroke-dasharray="3 4"/>')
    o.append(halo(xs[1] + 6, py + ph - 10, "registered — not yet run",
                  size=9.4, fill=GATE))

    for series, col, lab in (((0.0, dd), ARM, "drift"),
                             ((0.0, dc), CTRL, "cos (centred)")):
        y0, y1 = sy(series[0]), sy(series[1])
        o.append(f'<line x1="{xs[0]:.1f}" y1="{y0:.1f}" x2="{xs[3]:.1f}" '
                 f'y2="{y1:.1f}" stroke="{col}" stroke-width="1.6" '
                 f'stroke-dasharray="6 5"/>')
        for x, y in ((xs[0], y0), (xs[3], y1)):
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="{col}" '
                     f'stroke="{CARD}" stroke-width="1.8"/>')
        o.append(halo(xs[3] - 8, y1 - 10, f"{lab} {series[1]:+.1f} %",
                      size=10.0, fill=col, anchor="end"))
    o.append(halo(xs[0] + 8, sy(0) - 11, "base = 0 % by definition", size=9.4,
                  fill=MUTED))
    o.append(f'<text x="{px+pw*0.5:.1f}" y="{py+ph+55}" font-size="9.2" '
             f'text-anchor="middle" fill="{MUTED}">the dashed joins are NOT '
             f'a curve: two points cannot show monotonicity</text>')

    rows = [("COUPLED", "drift and cos move monotonically together across "
             "the four"),
            ("DECOUPLED", "a point has lower drift with cos inside the base "
             "seed band"),
            ("L1-SPECIFIC", "the disjoint-mechanism arms do NOT trade along "
             "that line"),
            ("NON-MONOTONE", "no verdict — report the points and stop")]
    y = py + ph + 76
    o.append(f'<text x="{ox+16}" y="{y}" font-size="10.4" font-weight="700" '
             f'fill="{INK}">Outcomes committed in advance</text>')
    for i, (k, v) in enumerate(rows):
        o.append(f'<text x="{ox+16}" y="{y+16+i*22:.0f}" font-size="9.4" '
                 f'font-weight="700" fill="{GATE}">{esc(k)}</text>')
        o.append(f'<text x="{ox+16}" y="{y+28+i*22:.0f}" font-size="8.8" '
                 f'fill="{MUTED}">{esc(v)}</text>')
    o.append(f'<text x="{ox+16}" y="{oy+h-16}" font-size="9.2" '
             f'fill="{MUTED}">test 1 (L2 frozen-teacher, L4 azimuthal crop) '
             f'is the load-bearing one: only a</text>')
    o.append(f'<text x="{ox+16}" y="{oy+h-5}" font-size="9.2" '
             f'fill="{MUTED}">DISJOINT mechanism can tell a frontier from a '
             f'selection effect over levers.</text>')
    return o


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="40" y="44" font-size="24" font-weight="700" '
           f'fill="{INK}">Prediction at matched drift: two measured deltas, '
           f'and the hypothesis they raise</text>',
           f'<text x="40" y="68" font-size="12.5" fill="{INK2}">MM-E4 L1 cut '
           f'drift by {abs(pctv(L1_D, BASE_D)):.1f} % — the largest '
           f'reduction ever measured on the trainable line — and cost '
           f'{abs(pctv(L1_C, BASE_C)):.1f} % of prediction, failing the '
           f'pre-committed protection. The EMA teacher at 30 k moved BOTH the '
           f'other way.</text>',
           f'<text x="40" y="86" font-size="12.5" fill="{INK2}">Tier '
           f'T0-DIAGNOSTIC — world-model diagnostics, never driving '
           f'performance · drift instrument: RFF+ridge, K-fold '
           f'clip-disjoint, top-8 PCA of Δz at k = 4, 80 held-out clips '
           f'· every value read from the banked probe JSONs at generation '
           f'time.</text>']
    out.append(f'<circle cx="46" cy="104" r="5" fill="{ARM}"/>')
    out.append(f'<text x="58" y="108" font-size="11" fill="{INK2}">the '
               f'decision-grade intervention</text>')
    out.append(f'<circle cx="256" cy="104" r="5" fill="{CTRL}"/>')
    out.append(f'<text x="268" y="108" font-size="11" fill="{INK2}">the base '
               f'it is ONE VARIABLE from</text>')
    out.append(f'<circle cx="512" cy="104" r="5" fill="{GATE}" '
               f'fill-opacity="0.45"/>')
    out.append(f'<text x="524" y="108" font-size="11" fill="{INK2}">'
               f'motivation-only / registered-but-unrun</text>')
    out.append(f'<rect x="820" y="99" width="14" height="10" fill="{GATE}" '
               f'fill-opacity="0.10"/>')
    out.append(f'<text x="840" y="108" font-size="11" fill="{INK2}">the '
               f'quadrants the coupling HYPOTHESIS predicts (sign only — '
               f'no slope asserted, no fit)</text>')
    out.append(f'<line x1="40" y1="118" x2="{W-40}" y2="118" stroke="{GRID}" '
               f'stroke-width="1.2"/>')

    # ---- panel A: 2k -------------------------------------------------------
    axlo, axhi = 0.325, 0.505
    aylo, ayhi = -0.02, 0.28
    out += plane(
        40, 132, 380, 470,
        "A · the plane at 2 k",
        [f"MM-E4 L1 vs its base o14fut10 — one variable",
         f"drift {pct(L1_D, BASE_D)} · cos {pct(L1_C, BASE_C)} "
         f"— BOTH DOWN",
         f"spans: drift {axlo:.2f}–{axhi:.2f} · "
         f"cos {aylo:.2f}–{ayhi:.2f}",
         "⚠ axes NOT shared with panel B"],
        [(BASE_D, BASE_C, 5.0, CTRL, f"base o14fut10  {BASE_D:.4f}, "
          f"{BASE_C:.4f}", -9, -10, "end", 10.0),
         (L1_D, L1_C, 5.4, ARM, f"L1 innovation-SIGReg  {L1_D:.4f}, "
          f"{L1_C:.4f}", 9, 14, "start", 10.0),
         (EMA2K[0][0], EMA2K[0][1], 3.8, GATE, "", 0, 0, "start", 9.0),
         (EMA2K[1][0], EMA2K[1][1], 3.8, GATE, "", 0, 0, "start", 9.0),
         (INC2K_D, INC2K_C, 3.8, GATE, "o14base2k", 8, 12, "start", 9.2)],
        [((BASE_D, BASE_C), (L1_D, L1_C), ARM, 2.4, None),
         ((INC2K_D, INC2K_C), EMA2K[0], GATE, 1.3, "5 4"),
         ((INC2K_D, INC2K_C), EMA2K[1], GATE, 1.3, "5 4")],
        axlo, axhi, aylo, ayhi, (BASE_D, BASE_C),
        ["violet: the 2 k EMA pair off its own base o14base2k — same sign "
         "on both axes, but",
         "MOTIVATION-ONLY (off the published τ map, §14.7) and NOT "
         "counted in the n = 2.",
         f"DR-arm inert scale: {DR_D:.4f} / {DR_C:.4f} vs base "
         f"{INC2K_D:.4f} / {INC2K_C:.4f}."])

    # ---- panel B: 30k ------------------------------------------------------
    bxlo, bxhi = 0.655, 0.705
    bylo, byhi = 0.56, 0.80
    out += plane(
        432, 132, 380, 470,
        "B · the plane at 30 k",
        ["E-DEC-69: the EMA teacher vs o14fut30k — one variable",
         f"drift {pct(EMA30_D, INC30_D)} · cos "
         f"{pct(EMA30_C, INC30_C)} — BOTH UP",
         f"spans: drift {bxlo:.3f}–{bxhi:.3f} · "
         f"cos {bylo:.2f}–{byhi:.2f}",
         "⚠ a 30 k drift number is not commensurable with a 2 k one"],
        [(INC30_D, INC30_C, 5.0, CTRL, f"o14fut30k  {INC30_D:.4f}, "
          f"{INC30_C:.4f}", -9, 15, "end", 10.0),
         (EMA30_D, EMA30_C, 5.4, ARM, f"EMA teacher  {EMA30_D:.4f}, "
          f"{EMA30_C:.4f}", -9, -9, "end", 10.0)],
        [((INC30_D, INC30_C), (EMA30_D, EMA30_C), ARM, 2.4, None)],
        bxlo, bxhi, bylo, byhi, (INC30_D, INC30_C),
        ["the two arrows agree in SIGN on both axes — that agreement, at "
         "n = 2, is the whole",
         "of the coupling observation. Different mechanisms, different scales, "
         "different parts",
         "of the cos axis (0.01–0.25 here vs 0.60–0.75 there): a "
         "HYPOTHESIS, not a result."])

    out += panel_c(824, 132, 376, 470)

    out.append(f'<text x="40" y="{H-64}" font-size="9.8" fill="{MUTED}">'
               f'Sources (MEASURED, ours; T0-DIAGNOSTIC): '
               f'e4_e4_l1_inno_drift/_nrmse.json (2026-08-29-e4-drift-ladder), '
               f'o14drift/o14nrmse/o14fut30k_*/ema30k_*.json '
               f'(2026-08-27-o14-tiny-ladder),</text>')
    out.append(f'<text x="40" y="{H-50}" font-size="9.8" fill="{MUTED}">'
               f'p0_drift/p0_nrmse.json (2026-08-28-p0-ema-bakeoff). Preregs: '
               f'PREREG_DRIFT_ATTACK_LADDER.md (MM-E4, protection rule '
               f'registered 19:00, read 21:40) · '
               f'PREREG_DRIFT_PREDICTION_FRONTIER.md (MM-E5, 22:02).</text>')
    out.append(f'<text x="40" y="{H-36}" font-size="9.8" fill="{MUTED}">'
               f'L1&#8217;s prediction collapse, four ways: nrmse '
               f'{L1_NRMSE:.4f} exceeds 1.0 AND its own mean-only control '
               f'{L1_MEANONLY:.4f}; {L1_MEANFRAC*100:.1f} % of the predicted '
               f'Δz IS the batch mean; the centred cosine falls from '
               f'{BASE_Z:.2f}σ to {L1_Z:.2f}σ against its '
               f'permutation null.</text>')
    out.append(f'<text x="40" y="{H-22}" font-size="9.8" fill="{MUTED}">'
               f'Four cross-artifact tripwires passed: the base arm&#8217;s '
               f'drift and cos agree across the O14 and drift ladders; the 2 k '
               f'incumbent&#8217;s drift across ladder and bake-off; the 30 k '
               f'incumbent&#8217;s cos across its own raw and the EMA '
               f'pair&#8217;s. Deltas are arithmetic on loaded values.</text>')
    out.append("</svg>")
    svg = "\n".join(out)
    sp = os.path.join(HERE, "drift_frontier.svg")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print("wrote", sp)
    try:
        import cairosvg
        pp = os.path.join(HERE, "drift_frontier.png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=pp, scale=2.0)
        print("wrote", pp)
    except (ImportError, OSError):
        # OSError: cairosvg installed without its native cairo (Windows).
        print("cairosvg unavailable — SVG only")


if __name__ == "__main__":
    main()
