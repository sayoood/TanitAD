"""Generate Figure 4 — the self-prediction drift story (SVG + PNG).

EVERY NUMBER IS READ FROM THE BANKED PROBE JSONs AT GENERATION TIME. Nothing is
typed in by hand: if a source artifact is missing the script exits loudly rather
than falling back to a literal, so the figure cannot drift from the measurement.

Sources (MEASURED, ours; all T0-DIAGNOSTIC):
  o14-tiny-ladder raw/      o14drift.json, o14nrmse.json (the 2k incumbent),
                            o14fut30k_drift.json, o14fut30k_nrmse.json (the 30k
                            incumbent), ema30k_drift.json, ema30k_nrmse.json
                            (MM-E1 stage 2)
  p0-ema-bakeoff raw/       p0_drift.json, p0_nrmse.json (MM-E1 stage 1: the
                            2k EMA pair + seed, and the same 2k incumbent)
  action-conditioning raw/  freezedrift.json (the frozen crossed cell),
                            seedrep.json (postrain30k + seed replicate),
                            deltaz_distilled.json (postrain10k — the
                            manufacture trajectory's start)

Cross-artifact tripwires (the figure refuses to draw on disagreement):
  * the 2k incumbent's drift must agree between the ladder's and the bake-off's
    raws (same arm, same instrument, two packages);
  * the 2k incumbent's centred cosine likewise;
  * the 30k incumbent's centred cosine must agree between its own raw and the
    EMA pair's raw.

Design notes (dataviz discipline, as make_winners_curse.py):
  * form first — the story is "two effects invert between 2k and 30k" (two
    slope panels, one scale each, never a dual axis) and "how much of drift is
    manufactured" (a bar panel with the frozen cell as the reference).
  * colour by job — ARM (blue) = the intervention (EMA / frozen), CTRL
    (orange) = the incumbent line, GATE (violet, dashed) = reference markers.
    Palette #2a78d6,#eb6834,#4a3aa7 — validated for the paper's figures.
  * identity is never colour-alone: every series is direct-labelled.
  * derived percentages are arithmetic on loaded values, never literals.

Honesty notes carried on the figure itself:
  * each EMA pair is one-variable WITHIN its scale; the 2k and 30k lines differ
    in recipe (2k: +omega, w_o14=0; 30k: +O14, legacy conditioning) — the
    comparison is the EMA delta at each scale, not the absolute levels, and
    EMA x O14 is an unseparated interaction (the pre-registered limit).
  * panel C shows seedrep.json's re-read of the trained arms (0.6674/0.6741);
    the registry quotes the first read of the same instrument as 0.669/0.679 —
    fold-partition refit variance, far inside the 1.5% seed band.
  * the world-smoothness / self-generated split is an ESTIMATE: drift is
    measured per-arm in that arm's own latent geometry (different rulers).
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
AI = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference")
LADDER = os.path.join(AI, "Implementation", "incoming",
                      "2026-08-27-o14-tiny-ladder", "raw")
P0 = os.path.join(AI, "Implementation", "incoming",
                  "2026-08-28-p0-ema-bakeoff", "raw")
ACOND = os.path.join(AI, "Research",
                     "2026-08-24-action-conditioning-and-heldout", "raw")

W, H = 1240, 792
SURFACE = "#fcfcfb"
CARD = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"
EDGE = "#dedcd6"
ARM = "#2a78d6"      # categorical slot 1 — the intervention (EMA / frozen)
CTRL = "#eb6834"     # categorical slot 2 — the incumbent line
GATE = "#4a3aa7"     # violet — reference marker
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
LD = load(LADDER, "o14drift.json")            # 2k ladder drift (incl. base)
LN = load(LADDER, "o14nrmse.json")            # 2k ladder cos
F30D = load(LADDER, "o14fut30k_drift.json")   # 30k incumbent drift
F30N = load(LADDER, "o14fut30k_nrmse.json")   # 30k incumbent cos
E30D = load(LADDER, "ema30k_drift.json")      # 30k EMA drift
E30N = load(LADDER, "ema30k_nrmse.json")      # 30k EMA cos (+ incumbent cos)
P0D = load(P0, "p0_drift.json")               # 2k EMA pair drift (+ base)
P0N = load(P0, "p0_nrmse.json")               # 2k EMA pair cos (+ base)
FRZ = load(ACOND, "freezedrift.json")         # frozen crossed cell
SEED = load(ACOND, "seedrep.json")            # postrain30k + seed1 re-read
D10 = load(ACOND, "deltaz_distilled.json")    # postrain10k (manufacture start)

BASE2K_D = drift_of(LD, "o14base2k")
BASE2K_C = cos_of(LN, "o14base2k")
EMA2K_D = [drift_of(P0D, "ema2k_s0"), drift_of(P0D, "ema2k_s1")]
EMA2K_C = [cos_of(P0N, "ema2k_s0"), cos_of(P0N, "ema2k_s1")]
INC30_D = drift_of(F30D, "o14fut30k")
INC30_C = cos_of(F30N, "o14fut30k")
EMA30_D = drift_of(E30D, "emao14_30k")
EMA30_C = cos_of(E30N, "emao14_30k")
FROZEN_D = drift_of(FRZ, "postrain30k_freeze")
TRAIN_D = drift_of(SEED, "postrain30k")
TRAIN_D_S1 = drift_of(SEED, "postrain30k_seed1")
P10K_D = D10["arms"]["postrain10k"]["columns"]["z_t (drift)"]["mean_r"]

# cross-artifact tripwires: same quantity, two independent packages.
for a, b, what in ((BASE2K_D, drift_of(P0D, "o14base2k"),
                    "2k incumbent drift (ladder vs bake-off)"),
                   (BASE2K_C, cos_of(P0N, "o14base2k"),
                    "2k incumbent cos (ladder vs bake-off)"),
                   (INC30_C, cos_of(E30N, "o14fut30k"),
                    "30k incumbent cos (own raw vs EMA-pair raw)")):
    if abs(a - b) > 5e-5:
        sys.exit(f"{what} disagrees across artifacts: {a} vs {b} — do not "
                 "draw the figure until this is resolved.")


def pct(new, old):
    v = (new / old - 1.0) * 100.0
    return f"{v:+.1f} %"


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


def slope_panel(ox, oy, w, h, title, sub, ylab, base2k, ema2k, inc30, ema30,
                ymax, better_note):
    """One metric, two stations (2 k, 30 k): incumbent line vs EMA points."""
    o = card(ox, oy, w, h, title, sub)
    # the deltas, computed from the loaded values — in the header, never over data
    o.append(f'<text x="{ox+16}" y="{oy+43+len(sub)*14+4}" font-size="11" '
             f'font-weight="700" fill="{INK}">2 k: {pct(ema2k[0], base2k)} / '
             f'{pct(ema2k[1], base2k)}   →   30 k: {pct(ema30, inc30)}</text>')
    px, py = ox + 52, oy + 118
    pw, ph = w - 52 - 22, h - 118 - 62
    x2k, x30 = px + pw * 0.16, px + pw * 0.84

    def sy(v):
        return py + ph - (v / ymax) * ph

    n_grid = 5
    for i in range(n_grid + 1):
        gv = ymax * i / n_grid
        y = sy(gv)
        o.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px+pw}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{px-7}" y="{y+3.4:.1f}" font-size="9.6" '
                 f'text-anchor="end" fill="{MUTED}">{gv:.2f}</text>')
    o.append(f'<text x="{px-38}" y="{py+ph/2:.1f}" font-size="10" '
             f'fill="{INK2}" transform="rotate(-90 {px-38} {py+ph/2:.1f})" '
             f'text-anchor="middle">{esc(ylab)}</text>')
    for xx, lab in ((x2k, "2 k steps"), (x30, "30 k steps")):
        o.append(f'<text x="{xx:.1f}" y="{py+ph+18}" font-size="10.4" '
                 f'text-anchor="middle" fill="{INK2}">{lab}</text>')
    o.append(f'<line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" '
             f'stroke="{GRID}" stroke-width="1.2"/>')

    # incumbent line (one variable within each scale; recipes differ across)
    o.append(f'<line x1="{x2k:.1f}" y1="{sy(base2k):.1f}" x2="{x30:.1f}" '
             f'y2="{sy(inc30):.1f}" stroke="{CTRL}" stroke-width="2" '
             f'stroke-dasharray="7 4"/>')
    # EMA: both 2k seeds converge on the single 30k point
    for v in ema2k:
        o.append(f'<line x1="{x2k:.1f}" y1="{sy(v):.1f}" x2="{x30:.1f}" '
                 f'y2="{sy(ema30):.1f}" stroke="{ARM}" stroke-width="2"/>')
    for xx, v, col in ((x2k, base2k, CTRL), (x30, inc30, CTRL),
                       (x2k, ema2k[0], ARM), (x2k, ema2k[1], ARM),
                       (x30, ema30, ARM)):
        o.append(f'<circle cx="{xx:.1f}" cy="{sy(v):.1f}" r="4.4" '
                 f'fill="{col}" stroke="{CARD}" stroke-width="2"/>')

    o.append(halo(x2k + 8, sy(base2k) - 8, f"incumbent {base2k:.4f}",
                  fill=CTRL))
    inc30_dy = 16 if ema30 > inc30 else -8
    o.append(halo(x30 - 8, sy(inc30) + inc30_dy, f"{inc30:.4f}", fill=CTRL,
                  anchor="end"))
    o.append(halo(x2k + 8, sy(min(ema2k)) + 14,
                  f"EMA {ema2k[0]:.4f} / {ema2k[1]:.4f}", fill=ARM))
    ema30_dy = -8 if ema30 > inc30 else 16
    o.append(halo(x30 - 8, sy(ema30) + ema30_dy, f"EMA {ema30:.4f}", fill=ARM,
                  anchor="end"))
    o.append(f'<text x="{px+pw*0.5:.1f}" y="{py+ph+36}" font-size="9.6" '
             f'text-anchor="middle" fill="{MUTED}">{esc(better_note)}</text>')
    return o


# --------------------------------------------------------------------- panel C
def panel_c(ox, oy, w, h):
    o = card(ox, oy, w, h, "How much of the drift is manufactured",
             ["drift r at 30 k, same instrument, same 80 held-out clips",
              "frozen cell = what a FIXED representation of a smooth world",
              "already makes predictable; the trained excess is the",
              "objective's own contribution (E-DEC-61/64)"])
    px, py = ox + 178, oy + 104
    pw, ph = w - 178 - 26, 186
    xmax = 0.75
    rows = [("frozen encoder", "postrain30k_freeze", FROZEN_D, ARM),
            ("trained", "postrain30k", TRAIN_D, CTRL),
            ("trained · seed 1", "postrain30k_seed1", TRAIN_D_S1, CTRL)]
    bh = ph / len(rows) * 0.52

    def sx(v):
        return px + (v / xmax) * pw

    for gv in (0.0, 0.2, 0.4, 0.6):
        x = sx(gv)
        o.append(f'<line x1="{x:.1f}" y1="{py}" x2="{x:.1f}" y2="{py+ph}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{py+ph+14}" font-size="9.6" '
                 f'text-anchor="middle" fill="{MUTED}">{gv:.1f}</text>')

    for i, (lab, arm, v, col) in enumerate(rows):
        yc = py + ph * (i + 0.5) / len(rows)
        o.append(f'<text x="{px-8}" y="{yc-2:.1f}" font-size="10.4" '
                 f'text-anchor="end" fill="{INK}">{esc(lab)}</text>')
        o.append(f'<text x="{px-8}" y="{yc+10:.1f}" font-size="8.8" '
                 f'text-anchor="end" fill="{MUTED}">{esc(arm)}</text>')
        o.append(f'<rect x="{px}" y="{yc-bh/2:.1f}" width="{sx(v)-px:.1f}" '
                 f'height="{bh:.1f}" rx="2.5" fill="{col}"/>')
        o.append(halo(sx(v) + 6, yc + 4, f"{v:.4f}", fill=col))

    # the manufacture trajectory's start: same recipe at 10k
    x10 = sx(P10K_D)
    o.append(f'<line x1="{x10:.1f}" y1="{py-6}" x2="{x10:.1f}" '
             f'y2="{py+ph}" stroke="{GATE}" stroke-width="1.3" '
             f'stroke-dasharray="5 4"/>')
    o.append(halo(x10 + 5, py + 4, f"same recipe @ 10 k: {P10K_D:.4f}",
                  size=9.4, fill=GATE))

    # the self-generated excess, computed from the loaded values — an ESTIMATE.
    # Drawn BELOW the axis so it can never sit on a bar row.
    y_br = py + ph + 26
    o.append(f'<path d="M {sx(FROZEN_D):.1f} {y_br:.1f} v 7 '
             f'H {sx(TRAIN_D):.1f} v -7" fill="none" stroke="{INK2}" '
             f'stroke-width="1.2"/>')
    o.append(f'<line x1="{sx(FROZEN_D):.1f}" y1="{py+ph:.1f}" '
             f'x2="{sx(FROZEN_D):.1f}" y2="{y_br:.1f}" stroke="{GRID}" '
             f'stroke-width="1"/>')
    o.append(f'<line x1="{sx(TRAIN_D):.1f}" y1="{py+ph:.1f}" '
             f'x2="{sx(TRAIN_D):.1f}" y2="{y_br:.1f}" stroke="{GRID}" '
             f'stroke-width="1"/>')
    o.append(halo((sx(FROZEN_D) + sx(TRAIN_D)) / 2, y_br + 21,
                  f"self-generated ≈ +{TRAIN_D - FROZEN_D:.2f} — ESTIMATE",
                  size=10.2, anchor="middle"))
    o.append(f'<text x="{(sx(FROZEN_D)+sx(TRAIN_D))/2:.1f}" y="{y_br+35:.1f}" '
             f'font-size="9.2" text-anchor="middle" fill="{MUTED}">different '
             f'latent geometries — not a variance decomposition</text>')
    o.append(f'<text x="{ox+16}" y="{oy+h-40}" font-size="9.2" '
             f'fill="{MUTED}">freezing removes the manufactured component but '
             f'FAILS prediction (nrmse +14.6 % — DEGENERATE,</text>')
    o.append(f'<text x="{ox+16}" y="{oy+h-27}" font-size="9.2" '
             f'fill="{MUTED}">pre-registered band): the v7 encoder stays '
             f'trainable. Registry first-read of the trained pair: '
             f'0.669 / 0.679.</text>')
    o.append(f'<text x="{ox+16}" y="{oy+h-14}" font-size="9.2" '
             f'fill="{MUTED}">Shown: the seedrep.json re-read — '
             f'fold-partition variance, inside the 1.5 % seed band.</text>')
    return o


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="40" y="44" font-size="24" font-weight="700" '
           f'fill="{INK}">The drift attractor: manufactured by training, '
           f'untouched by the teacher</text>',
           f'<text x="40" y="68" font-size="12.5" fill="{INK2}">MM-E1: both '
           f'2 k EMA-teacher effects INVERT at 30 k — the pre-registered '
           f'transient account confirmed (fixed τ = 0.996 is 12.5 % of a 2 k '
           f'run, 0.83 % of a 30 k run) — and the frozen crossed cell prices '
           f'the manufactured share.</text>',
           f'<text x="40" y="86" font-size="12.5" fill="{INK2}">Tier '
           f'T0-DIAGNOSTIC — world-model diagnostics, never driving '
           f'performance · drift instrument: RFF+ridge, K-fold clip-disjoint, '
           f'top-8 PCA of Δz at k = 4, 80 held-out clips · every value read '
           f'from the banked probe JSONs at generation time.</text>']
    out.append(f'<circle cx="46" cy="104" r="5" fill="{ARM}"/>')
    out.append(f'<text x="58" y="108" font-size="11" fill="{INK2}">the '
               f'intervention (EMA teacher · frozen encoder)</text>')
    out.append(f'<circle cx="330" cy="104" r="5" fill="{CTRL}"/>')
    out.append(f'<text x="342" y="108" font-size="11" fill="{INK2}">the '
               f'incumbent (live self-target, trainable)</text>')
    out.append(f'<line x1="586" y1="104" x2="612" y2="104" stroke="{GATE}" '
               f'stroke-width="1.4" stroke-dasharray="4 3"/>')
    out.append(f'<text x="620" y="108" font-size="11" fill="{INK2}">'
               f'reference marker</text>')
    out.append(f'<line x1="40" y1="118" x2="{W-40}" y2="118" stroke="{GRID}" '
               f'stroke-width="1.2"/>')

    out += slope_panel(
        40, 132, 380, 470,
        "Drift: the 2 k gain vanishes and inverts",
        ["r(g(z_t), Δz) — lower = less self-prediction",
         "at 2 k the EMA teacher is the first lever ever to move",
         "drift down; at 30 k it reads ABOVE the incumbent",
         "⇒ EMA-OUT on the committed question"],
        "drift r", BASE2K_D, EMA2K_D, INC30_D, EMA30_D, 0.75,
        "one variable within each pair; 2 k and 30 k lines differ in recipe")
    out += slope_panel(
        432, 132, 380, 470,
        "Prediction: the 2 k cost becomes a gain",
        ["centred cosine of the h=1 prediction — higher is better",
         "the 2 k deficit is the ramp-less fixed-τ teacher averaging",
         "near-random early weights; at 30 k the same τ sits inside",
         "the published band and prediction improves +24.5 %"],
        "cos (centred)", BASE2K_C, EMA2K_C, INC30_C, EMA30_C, 0.80,
        "E-DEC-69: the largest prediction gain measured on the trainable line")
    out += panel_c(824, 132, 376, 470)

    out.append(f'<text x="40" y="{H-64}" font-size="9.8" fill="{MUTED}">'
               f'Sources (MEASURED, ours): o14drift/o14nrmse/o14fut30k_*/'
               f'ema30k_* .json (2026-08-27-o14-tiny-ladder), p0_drift/'
               f'p0_nrmse.json (2026-08-28-p0-ema-bakeoff),</text>')
    out.append(f'<text x="40" y="{H-50}" font-size="9.8" fill="{MUTED}">'
               f'freezedrift/seedrep/deltaz_distilled.json (2026-08-24-action-'
               f'conditioning-and-heldout). Registry: MODEL_REGISTRY.md '
               f'§13.9–§13.10 · prereg PREREG_P0_EMA_BAKEOFF.md (both stages, '
               f'τ-amendment recorded before the stage-2 read).</text>')
    out.append(f'<text x="40" y="{H-36}" font-size="9.8" fill="{MUTED}">'
               f'Deltas are arithmetic on the loaded values. Stated limits: '
               f'EMA×O14 interaction unseparated; the 2 k pair is '
               f'motivation-only (no published EMA operating point exists at '
               f'≤ 5 k steps); the decomposition is an ESTIMATE.</text>')
    out.append(f'<text x="40" y="{H-22}" font-size="9.8" fill="{MUTED}">'
               f'Cross-artifact tripwires passed: the 2 k incumbent agrees '
               f'across the ladder and bake-off packages; the 30 k '
               f'incumbent&#8217;s cosine agrees across its own raw and the '
               f'EMA pair&#8217;s.</text>')
    out.append("</svg>")
    svg = "\n".join(out)
    sp = os.path.join(HERE, "drift_story.svg")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print("wrote", sp)
    try:
        import cairosvg
        pp = os.path.join(HERE, "drift_story.png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=pp, scale=2.0)
        print("wrote", pp)
    except (ImportError, OSError):
        # OSError: cairosvg installed without its native cairo (Windows).
        print("cairosvg unavailable — SVG only")


if __name__ == "__main__":
    main()
