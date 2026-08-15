"""Generate Figure 3 — the winner's curse in TanitAD's selection stage (SVG + PNG).

EVERY NUMBER IS READ FROM THE GATE JSONs AT GENERATION TIME. Nothing is typed in
by hand: if a source artifact is missing the script exits loudly rather than
falling back to a literal, so the figure cannot drift from the measurement.

Sources (MEASURED, ours):
  w7_full_gate.json        W7-FULL selector-free run (T0, 881-window grid)
  w7_selection_rules.json  the 0-GPU selection-rule sweep over its banked windows
                           (EXPLORATORY class — stamped on the figure)
  w4r_gate.json            the unicycle head refit on the repaired trunk
Both live in the redesign incoming directory of the Research Hub.

Design notes (dataviz discipline):
  * form first — the data's job is "does conditioning on the cost change the error
    distribution?", which is a magnitude-vs-m comparison (panel A), a position on a
    rank ruler (panel B), and a small ranked comparison (panel C). No dual axis
    anywhere: every panel carries exactly one scale, and the rank ruler is its own
    panel precisely so rank is never plotted against metres.
  * colour by job — ARM (blue) = headroom that exists, CTRL (orange) = what the
    rule actually delivers, GATE (violet, dashed) = pre-registered reference.
    Palette validated: node dataviz/scripts/validate_palette.js
    "#2a78d6,#eb6834,#4a3aa7" --mode light --pairs all -> ALL CHECKS PASS
    (worst all-pairs CVD dE 13.0 deutan, normal-vision 16.3).
  * identity is never colour-alone: every series is direct-labelled.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
GATES = os.path.join(
    REPO, "TanitAD Research Hub", "Architecture & Inference", "Implementation",
    "incoming", "2026-08-07-hierarchical-wm-redesign")

W, H = 1240, 806
SURFACE = "#fcfcfb"
CARD = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"
EDGE = "#dedcd6"
ARM = "#2a78d6"      # categorical slot 1 — headroom that exists in the fan
CTRL = "#eb6834"     # categorical slot 2 — what the argmin rule delivers
GATE = "#4a3aa7"     # violet — pre-registered gate / reference marker
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"


def esc(s):
    return html.escape(str(s))


def load(name):
    p = os.path.join(GATES, name)
    if not os.path.exists(p):
        sys.exit(f"missing source artifact: {p}\n"
                 "This figure is generated FROM the gate JSONs; it has no "
                 "hand-typed fallback by design.")
    with open(p) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- the literals
FULL = load("w7_full_gate.json")
RULES = load("w7_selection_rules.json")
W4R = load("w4r_gate.json")

ORACLE = RULES["oracle_ade"]                        # 0.1273
W7_ADE = RULES["w7_deployed_combined_cost_ade"]     # 3.3348
ROLL_ONLY = RULES["argmin_on_roll_cost_alone_ade"]  # 5.2898
RANK_BLEND = RULES["rank_blend_ade"]                # 5.4615
ARGMIN_RANK = RULES["mean_error_rank_of_cost_argmin"]   # 132.32
CHANCE_RANK = RULES["median_rank_would_be"]             # 128
N_CAND = RULES["n_candidates"]                          # 256
N_WIN = RULES["n_windows"]                              # 881
CEIL = {int(k): v for k, v in RULES["top_m_ceiling_ade"].items()}
TOPM = {int(k): v for k, v in RULES["top_m_mean_error_ade"].items()}
MS = sorted(CEIL)
# The artifact's own read calls the top-m plateau "the fan's own mean". We
# plot the MEASURED plateau level (the m=32 value) rather than deriving an
# average of the series, so no number on the figure is computed by us.
PLATEAU = TOPM[max(MS)]

THR = FULL["gate_W7_selgap_closed"]["threshold_m"]      # 0.4505
FROZEN = FULL["mini_eval"]["frozen_selected_ade_in_run"]  # 4.4159
RHO_W = FULL["calibration_p7"]["within_window_cost_vs_error_over_shortlist"]
RHO_A = FULL["calibration_p7"]["across_windows_cost_vs_realised_error"]
IN_SHORT = FULL["mini_eval"]["winner_in_shortlist_frac"]  # 1.0
W4R_ORACLE = W4R["oracle_ade"]                            # 0.1273

# cross-artifact tripwire: the sweep rounds the oracle to 4 dp, the gate does not.
assert abs(W4R_ORACLE - ORACLE) < 5e-5, (
    f"fan oracle disagrees across artifacts: w4r_gate {W4R_ORACLE} vs "
    f"w7_selection_rules {ORACLE} — do not draw the figure until this is "
    "resolved.")


# ------------------------------------------------------------------- primitives
def card(ox, oy, w, h, title, sub, fill=CARD, stroke=EDGE):
    o = [f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="7" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
         f'<text x="{ox+18}" y="{oy+27}" font-size="14.5" font-weight="700" '
         f'fill="{INK}">{esc(title)}</text>']
    for i, line in enumerate(sub):
        o.append(f'<text x="{ox+18}" y="{oy+45+i*15}" font-size="10.6" '
                 f'fill="{INK2 if i == 0 else MUTED}">{esc(line)}</text>')
    return o


def halo(x, y, txt, size=11, weight="700", fill=INK, anchor="start"):
    """Direct label with a surface-coloured halo.

    NOTE: `paint-order="stroke"` is NOT honoured by every SVG rasteriser (our
    PNG export ignores it), which silently erases the glyph under its own white
    stroke — the label is present in the file and invisible in the render. So
    the halo is drawn as a SEPARATE underlay element, which works everywhere.
    """
    common = (f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
              f'font-weight="{weight}" text-anchor="{anchor}"')
    return (f'<text {common} fill="{CARD}" stroke="{CARD}" stroke-width="3.6" '
            f'stroke-linejoin="round">{esc(txt)}</text>'
            f'<text {common} fill="{fill}">{esc(txt)}</text>')


# --------------------------------------------------------------------- panel A
def panel_a(ox, oy, w, h):
    o = card(ox, oy, w, h,
             "Conditioning on the cost does not change the error distribution",
             [f"realised ADE (m) of the candidates inside the cost's own top-m, "
              f"over {N_CAND} candidates x {N_WIN} windows",
              "if the cost selected, the orange line would fall with m. It does "
              "not: it sits on the fan's own mean."])
    px, py = ox + 66, oy + 92
    pw, ph = w - 66 - 118, h - 92 - 52
    ymax = 6.0

    def sy(v):
        return py + ph - (min(v, ymax) / ymax) * ph

    def sx(i):
        return px + (i / (len(MS) - 1)) * pw

    for gv in (0, 1, 2, 3, 4, 5, 6):
        y = sy(gv)
        o.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px+pw}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{px-9}" y="{y+3.6:.1f}" font-size="10" '
                 f'text-anchor="end" fill="{MUTED}">{gv}</text>')
    o.append(f'<text x="{px-46}" y="{py+ph/2:.1f}" font-size="10.4" '
             f'fill="{INK2}" transform="rotate(-90 {px-46} {py+ph/2:.1f})" '
             f'text-anchor="middle">ADE (m)</text>')

    # the plateau the orange line lies on = the fan's own mean (artifact read)
    ymean = sy(PLATEAU)
    o.append(f'<line x1="{px}" y1="{ymean:.1f}" x2="{px+pw+8}" y2="{ymean:.1f}" '
             f'stroke="{GATE}" stroke-width="1.3" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{px+pw+13}" y="{ymean-5:.1f}" font-size="10" '
             f'fill="{GATE}">the fan\'s</text>')
    o.append(f'<text x="{px+pw+13}" y="{ymean+8:.1f}" font-size="10" '
             f'fill="{GATE}">own mean</text>')
    o.append(f'<text x="{px+pw+13}" y="{ymean+21:.1f}" font-size="10" '
             f'fill="{GATE}">{PLATEAU:.2f} m</text>')

    # full-fan oracle
    yor = sy(ORACLE)
    o.append(f'<line x1="{px}" y1="{yor:.1f}" x2="{px+pw+8}" y2="{yor:.1f}" '
             f'stroke="{GATE}" stroke-width="1.3" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{px+pw+13}" y="{yor+3.6:.1f}" font-size="10" '
             f'fill="{GATE}">oracle {ORACLE:.4f} m</text>')

    for key, colour, label in ((TOPM, CTRL, "mean error inside the top-m"),
                               (CEIL, ARM, "best available in the top-m")):
        pts = " ".join(f"{sx(i):.1f},{sy(key[m]):.1f}" for i, m in enumerate(MS))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        for i, m in enumerate(MS):
            o.append(f'<circle cx="{sx(i):.1f}" cy="{sy(key[m]):.1f}" r="4.6" '
                     f'fill="{colour}" stroke="{CARD}" stroke-width="2"/>')

    o.append(halo(sx(0) + 12, sy(TOPM[MS[0]]) - 12,
                  f"{TOPM[MS[0]]:.2f} m", fill=CTRL))
    o.append(halo(sx(len(MS) - 1) - 6, sy(TOPM[MS[-1]]) - 12,
                  f"{TOPM[MS[-1]]:.2f} m — flat", fill=CTRL, anchor="end"))
    o.append(halo(sx(0) + 12, sy(CEIL[MS[0]]) - 12, f"{CEIL[MS[0]]:.2f} m",
                  fill=ARM))
    o.append(halo(sx(len(MS) - 1) - 6, sy(CEIL[MS[-1]]) - 13,
                  f"{CEIL[MS[-1]]:.3f} m", fill=ARM, anchor="end"))

    for i, m in enumerate(MS):
        o.append(f'<text x="{sx(i):.1f}" y="{py+ph+18}" font-size="10.4" '
                 f'text-anchor="middle" fill="{INK2}">{m}</text>')
    o.append(f'<text x="{px+pw/2:.1f}" y="{py+ph+34}" font-size="10.4" '
             f'text-anchor="middle" fill="{INK2}">m = size of the cost\'s '
             f'lowest-cost set</text>')
    o.append(f'<line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" '
             f'stroke="{GRID}" stroke-width="1.2"/>')
    return o


# --------------------------------------------------------------------- panel B
def panel_b(ox, oy, w, h):
    o = card(ox, oy, w, h, "Where the argmin actually lands",
             [f"rank of the chosen candidate's realised error, 1 = best of "
              f"{N_CAND}",
              "a selector with tail information lands near 1. This one lands on "
              "the median."])
    px, py = ox + 26, oy + 118
    pw = w - 52
    o.append(f'<rect x="{px}" y="{py}" width="{pw}" height="13" rx="6" '
             f'fill="{GRID}"/>')

    def rx(r):
        return px + ((r - 1) / (N_CAND - 1)) * pw

    o.append(f'<circle cx="{rx(1):.1f}" cy="{py+6.5}" r="6" fill="{ARM}" '
             f'stroke="{CARD}" stroke-width="2"/>')
    o.append(f'<text x="{rx(1):.1f}" y="{py-10}" font-size="10.2" '
             f'text-anchor="start" fill="{ARM}">rank 1 = oracle</text>')
    o.append(f'<text x="{rx(1):.1f}" y="{py+30}" font-size="10.2" '
             f'text-anchor="start" fill="{MUTED}">{ORACLE:.4f} m</text>')

    xc = rx(CHANCE_RANK)
    o.append(f'<line x1="{xc:.1f}" y1="{py-6}" x2="{xc:.1f}" y2="{py+19}" '
             f'stroke="{GATE}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    o.append(f'<text x="{xc:.1f}" y="{py-10}" font-size="10.2" '
             f'text-anchor="middle" fill="{GATE}">chance = {CHANCE_RANK}</text>')

    xa = rx(ARGMIN_RANK)
    o.append(f'<circle cx="{xa:.1f}" cy="{py+6.5}" r="7.5" fill="{CTRL}" '
             f'stroke="{CARD}" stroke-width="2.4"/>')
    o.append(halo(xa, py + 44, f"argmin: rank {ARGMIN_RANK:.1f}", size=12,
                  fill=CTRL, anchor="middle"))
    o.append(f'<text x="{px+pw:.1f}" y="{py+30}" font-size="10.2" '
             f'text-anchor="end" fill="{MUTED}">{N_CAND}</text>')

    ly = py + 70
    rows = [
        ("within-window rank corr. (over the 256)",
         f"rho {RHO_W['rho_mean']:.3f} mean / {RHO_W['rho_median']:.3f} median"),
        ("across-window calibration",
         f"rho {RHO_A['spearman_rho']:.4f} "
         f"[{RHO_A['rho_ci_cluster'][0]:.4f}, {RHO_A['rho_ci_cluster'][1]:.4f}]"),
        ("true best candidate available",
         f"{IN_SHORT*100:.0f}% of windows (no shortlist)"),
    ]
    for i, (k, v) in enumerate(rows):
        o.append(f'<text x="{ox+18}" y="{ly+i*28}" font-size="10.4" '
                 f'fill="{MUTED}">{esc(k)}</text>')
        o.append(f'<text x="{ox+18}" y="{ly+14+i*28}" font-size="11.4" '
                 f'font-weight="700" fill="{INK}">{esc(v)}</text>')
    o.append(f'<text x="{ox+18}" y="{oy+h-26}" font-size="10.2" fill="{INK2}">'
             'The cost ranks the bulk correctly and carries no information in '
             'the tail —</text>')
    o.append(f'<text x="{ox+18}" y="{oy+h-12}" font-size="10.2" fill="{INK2}">'
             'and the tail is the only part an argmin reads.</text>')
    return o


# --------------------------------------------------------------------- panel C
def panel_c(ox, oy, w, h):
    o = card(ox, oy, w, h, "What each selection rule scores, same fan",
             ["selected ADE (m), lower is better · T0 diagnostic on the "
              "881-window held-out grid",
              "the deployed number comes from the kinematic term; roll "
              "consistency alone selects nothing."])
    bars = [("fan oracle (best available)", ORACLE, ARM),
            ("pre-registered gate", THR, GATE),
            ("W7-FULL: roll + 0.2 x kinematic", W7_ADE, CTRL),
            ("frozen selector on the same fan", FROZEN, CTRL),
            ("argmin on roll consistency alone", ROLL_ONLY, CTRL),
            ("rank blend of the two costs", RANK_BLEND, CTRL)]
    lw = 218
    px = ox + 18 + lw
    pw = w - (18 + lw) - 86
    xmax = 5.8
    top = oy + 88
    bh, bg = 19, 13
    for i, (name, val, colour) in enumerate(bars):
        y = top + i * (bh + bg)
        o.append(f'<text x="{px-10}" y="{y+bh*0.72:.0f}" font-size="10.6" '
                 f'text-anchor="end" fill="{INK2}">{esc(name)}</text>')
        bw = max(2.5, (min(val, xmax) / xmax) * pw)
        if colour is GATE:
            o.append(f'<line x1="{px+bw:.1f}" y1="{y-4}" x2="{px+bw:.1f}" '
                     f'y2="{y+bh+4}" stroke="{GATE}" stroke-width="1.6" '
                     f'stroke-dasharray="4 3"/>')
        else:
            o.append(f'<rect x="{px}" y="{y}" width="{bw:.1f}" height="{bh}" '
                     f'rx="4" fill="{colour}"/>')
        o.append(halo(px + bw + 8, y + bh * 0.74, f"{val:.4f}".rstrip("0")
                      if val < 1 else f"{val:.3f}",
                      size=11, fill=INK if colour is not GATE else GATE))
    o.append(f'<line x1="{px}" y1="{top-8}" x2="{px}" '
             f'y2="{top+len(bars)*(bh+bg)-4}" stroke="{GRID}" '
             f'stroke-width="1.2"/>')
    return o


# ------------------------------------------------------------------ takeaway
def panel_d(ox, oy, w, h):
    o = card(ox, oy, w, h, "The mechanism", [], fill="#f3f6fa", stroke="#d6dde6")
    lines = [
        "A rank correlation is a BULK statistic. An argmin is an",
        "EXTREME one. What governs selection is lower-tail",
        "dependence — and here it is zero.",
        "",
        "A self-consistency cost (does the world model reproduce",
        "this candidate?) is minimised by a near-stationary",
        "candidate, whose realised error is large. Deepening the",
        "fan makes such a candidate more likely, so the argmin",
        "degrades while the oracle improves.",
        "",
        "V-JEPA 2-AC and DINO-WM minimise distance to a GOAL,",
        "which inaction cannot minimise. We copied the planning",
        "loop and dropped that term; W7's own progress term has",
        "been at weight 0.0 in every run to date.",
    ]
    yy = oy + 46
    for line in lines:
        if line:
            o.append(f'<text x="{ox+18}" y="{yy}" font-size="11.2" '
                     f'fill="{INK2}">{esc(line)}</text>')
        yy += 16
    return o


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="40" y="46" font-size="25" font-weight="700" fill="{INK}">'
           "The winner&#8217;s curse: a fan that contains the answer, and a rule "
           "that cannot find it</text>",
           f'<text x="40" y="70" font-size="12.5" fill="{INK2}">'
           f'W7-FULL, selector-free: all {N_CAND} candidates available in every '
           f'window (the true best is never excluded), on the repaired trunk with '
           f'a refit emission head.</text>',
           f'<text x="40" y="88" font-size="12.5" fill="{INK2}">'
           f'Tier T0 — a world-model diagnostic, never driving performance · '
           f'{N_WIN} held-out windows / 40 episodes · every value read from the '
           f'gate JSONs at generation time.</text>']
    # legend — identity is also carried by direct labels on every series
    out.append(f'<circle cx="46" cy="106" r="5" fill="{ARM}"/>')
    out.append(f'<text x="58" y="110" font-size="11" fill="{INK2}">headroom that '
               'exists in the fan</text>')
    out.append(f'<circle cx="272" cy="106" r="5" fill="{CTRL}"/>')
    out.append(f'<text x="284" y="110" font-size="11" fill="{INK2}">what an '
               'argmin rule delivers</text>')
    out.append(f'<line x1="500" y1="106" x2="526" y2="106" stroke="{GATE}" '
               'stroke-width="1.4" stroke-dasharray="4 3"/>')
    out.append(f'<text x="534" y="110" font-size="11" fill="{INK2}">'
               'pre-registered gate or reference</text>')
    out.append(f'<line x1="40" y1="120" x2="{W-40}" y2="120" stroke="{GRID}" '
               'stroke-width="1.2"/>')

    out += panel_a(40, 134, 700, 306)
    out += panel_b(764, 134, 436, 306)
    out += panel_c(40, 460, 700, 292)
    out += panel_d(764, 460, 436, 292)

    out.append(f'<text x="40" y="{H-30}" font-size="9.8" fill="{MUTED}">'
               'Sources (MEASURED, ours): w7_full_gate.json and w4r_gate.json '
               '(gate PASS), plus the 0-GPU sweep w7_selection_rules.json, whose '
               'own artifact stamps it EXPLORATORY —</text>')
    out.append(f'<text x="40" y="{H-16}" font-size="9.8" fill="{MUTED}">'
               'a rule chosen from it must be re-measured on a fresh grid. '
               'Point estimates are corpus-grid means over the fixed grid; the '
               'two correlations carry the episode-cluster bootstrap.</text>')
    out.append("</svg>")
    svg = "\n".join(out)
    sp = os.path.join(HERE, "winners_curse.svg")
    with open(sp, "w") as fh:
        fh.write(svg)
    print("wrote", sp)
    try:
        import cairosvg
        pp = os.path.join(HERE, "winners_curse.png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=pp, scale=2.0)
        print("wrote", pp)
    except ImportError:
        print("cairosvg absent — SVG only")


if __name__ == "__main__":
    main()
