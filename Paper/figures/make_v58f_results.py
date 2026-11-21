"""Generate the v5.8f repair-arc results figure (SVG).

Every number is MEASURED and carries its registry section / artifact in the
`src` field — the figure is generated FROM those literals so it cannot drift
from the source of truth by hand-editing.
"""
import html
import os

W, H = 1240, 706
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"
CARD = "#ffffff"
ARM = "#2a78d6"        # categorical slot 1 — measured pipeline arm
CTRL = "#eb6834"       # categorical slot 2 — control / pre-repair / ablation
GATE = "#4a3aa7"       # violet — pre-registered gate marker

FONT = ("Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif")

PANELS = [
    dict(title="Action-response gain", unit="× the unicycle analytic",
         note="the root defect, repaired at predictor scale",
         gate=(0.5, 2.0), gate_label="pre-registered band [0.5, 2.0]",
         better=None, xmax=2.1,
         bars=[("v5f trunk · left", 0.27, CTRL),
               ("v5f trunk · right", 0.23, CTRL),
               ("stage-A · left", 0.971, ARM),
               ("stage-A · right", 0.966, ARM)],
         src="registry §1.13c · w3_gate.json / stage_a_gate.json"),
    dict(title="Fan oracle ADE", unit="m · lower is better",
         note="the reachable best inside the candidate fan",
         gate=None, better="lower", xmax=0.24,
         bars=[("v5f fan", 0.1975, CTRL),
               ("W4 head · frozen trunk", 0.1077, ARM),
               ("W4r head · repaired trunk", 0.1273, ARM)],
         src="registry §1.13/§1.14 · w4_gate.json / w4r_gate.json"),
    dict(title="Selected trajectory · accel MAE", unit="m/s² · lower is better",
         note="kinematic feasibility of what the planner actually picks",
         gate=(None, 1.5), gate_label="gate < 1.5", better="lower", xmax=9.0,
         bars=[("v5f selected", 8.10, CTRL),
               ("W4 selected", 0.774, ARM),
               ("W4r winner", 0.276, ARM)],
         src="registry §1.13/§1.14 · w4_gate.json / w4r_gate.json"),
    dict(title="Selector calibration", unit="Spearman ρ · higher is better",
         note="does the score know where the plan will err?",
         gate=(0.3, None), gate_label="P7 gate ρ ≥ 0.3", better="higher",
         xmax=0.82,
         bars=[("W4b learned rescorer", 0.054, CTRL),
               ("frozen argmax", 0.262, CTRL),
               ("W7 roll-cost · K=8", 0.399, ARM),
               ("W7 roll-cost · repaired trunk", 0.716, ARM)],
         src="registry §1.14 · p7_regrade.json / w7_gate_*.json"),
    dict(title="Imagination ablation", unit="selected ADE, m · lower is better",
         note="break the imagination input, keep everything else",
         gate=None, better="lower", xmax=8.2,
         bars=[("intact", 0.4011, ARM),
               ("shuffled (correspondence broken)", 1.2492, CTRL),
               ("zeroed (content removed)", 7.6493, CTRL)],
         src="registry §1.14 · i4a_{none,shuffle,zero}.json"),
]

PANEL_W, PANEL_H = 372, 246
COLS, PAD_X, PAD_Y = 3, 40, 26
X0, Y0 = 40, 132
BAR_H, BAR_GAP = 19, 12
LABEL_W = 176


def esc(s):
    return html.escape(str(s))


def panel_svg(p, ox, oy):
    o = []
    o.append(f'<rect x="{ox}" y="{oy}" width="{PANEL_W}" height="{PANEL_H}" '
             f'rx="7" fill="#ffffff" stroke="#dedcd6" stroke-width="1"/>')
    o.append(f'<text x="{ox+16}" y="{oy+26}" font-size="14.5" font-weight="700" '
             f'fill="{INK}">{esc(p["title"])}</text>')
    o.append(f'<text x="{ox+16}" y="{oy+44}" font-size="10.6" fill="{INK2}">'
             f'{esc(p["unit"])}</text>')
    o.append(f'<text x="{ox+16}" y="{oy+60}" font-size="10.2" fill="{MUTED}">'
             f'{esc(p["note"])}</text>')

    plot_x = ox + 16 + LABEL_W
    plot_w = PANEL_W - (16 + LABEL_W) - 58
    top = oy + 78
    xmax = p["xmax"]

    def sx(v):
        return plot_x + max(0.0, min(v / xmax, 1.0)) * plot_w

    # gate band / line
    g = p.get("gate")
    if g:
        lo, hi = g
        if lo is not None and hi is not None:
            x1, x2 = sx(lo), sx(hi)
            o.append(f'<rect x="{x1:.1f}" y="{top-6}" width="{x2-x1:.1f}" '
                     f'height="{len(p["bars"])*(BAR_H+BAR_GAP)+4}" '
                     f'fill="{GATE}" fill-opacity="0.07"/>')
            for xx in (x1, x2):
                o.append(f'<line x1="{xx:.1f}" y1="{top-6}" x2="{xx:.1f}" '
                         f'y2="{top+len(p["bars"])*(BAR_H+BAR_GAP)-2}" '
                         f'stroke="{GATE}" stroke-width="1.2" '
                         f'stroke-dasharray="4 3"/>')
        else:
            v = lo if lo is not None else hi
            xx = sx(v)
            o.append(f'<line x1="{xx:.1f}" y1="{top-6}" x2="{xx:.1f}" '
                     f'y2="{top+len(p["bars"])*(BAR_H+BAR_GAP)-2}" '
                     f'stroke="{GATE}" stroke-width="1.2" '
                     f'stroke-dasharray="4 3"/>')
        o.append(f'<text x="{plot_x}" y="{oy+PANEL_H-30}" font-size="9.6" '
                 f'fill="{GATE}">{esc(p["gate_label"])}</text>')

    for i, (name, val, colour) in enumerate(p["bars"]):
        y = top + i * (BAR_H + BAR_GAP)
        o.append(f'<text x="{plot_x-10}" y="{y+BAR_H*0.72:.0f}" font-size="10.6" '
                 f'text-anchor="end" fill="{INK2}">{esc(name)}</text>')
        bw = max(2.0, sx(val) - plot_x)
        o.append(f'<rect x="{plot_x}" y="{y}" width="{bw:.1f}" height="{BAR_H}" '
                 f'rx="4" fill="{colour}"/>')
        txt = f"{val:.3f}" if val < 10 else f"{val:.2f}"
        # NOTE: paint-order="stroke" is NOT honoured by every SVG rasteriser
        # (our PNG export ignores it), which erases the glyph under its own
        # halo — present in the file, invisible in the render. Draw the halo as
        # a separate underlay element, which works everywhere.
        lbl = (f'x="{plot_x+bw+7:.1f}" y="{y+BAR_H*0.74:.0f}" font-size="11" '
               f'font-weight="700"')
        o.append(f'<text {lbl} fill="{CARD}" stroke="{CARD}" stroke-width="3.4" '
                 f'stroke-linejoin="round">{txt}</text>')
        o.append(f'<text {lbl} fill="{INK}">{txt}</text>')

    o.append(f'<line x1="{plot_x}" y1="{top-8}" x2="{plot_x}" '
             f'y2="{top+len(p["bars"])*(BAR_H+BAR_GAP)-2}" stroke="{GRID}" '
             f'stroke-width="1.2"/>')
    o.append(f'<text x="{ox+16}" y="{oy+PANEL_H-13}" font-size="9.2" '
             f'fill="{MUTED}">{esc(p["src"])}</text>')
    return "\n".join(o)


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']
    out.append(f'<text x="40" y="48" font-size="25" font-weight="700" '
               f'fill="{INK}">v5.8f — the repair arc, measured</text>')
    out.append(f'<text x="40" y="72" font-size="12.5" fill="{INK2}">'
               'One defect (a muffled action interface) explained four failures; '
               'repairing it at predictor scale restored the action response, the '
               'fan and the selector calibration.</text>')
    out.append(f'<text x="40" y="90" font-size="12.5" fill="{INK2}">'
               'Tier T0 unless stated · 881-window eval grid · every bar cites its '
               'registry section and artifact.</text>')
    # legend
    out.append(f'<rect x="40" y="102" width="11" height="11" rx="3" fill="{ARM}"/>')
    out.append(f'<text x="57" y="112" font-size="11" fill="{INK2}">measured arm of '
               'the pipeline</text>')
    out.append(f'<rect x="248" y="102" width="11" height="11" rx="3" fill="{CTRL}"/>')
    out.append(f'<text x="265" y="112" font-size="11" fill="{INK2}">control, '
               'pre-repair state, or ablation</text>')
    out.append(f'<line x1="530" y1="108" x2="556" y2="108" stroke="{GATE}" '
               'stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<text x="562" y="112" font-size="11" fill="{INK2}">'
               'pre-registered gate</text>')
    out.append(f'<line x1="40" y1="120" x2="{W-40}" y2="120" stroke="{GRID}" '
               'stroke-width="1.2"/>')

    for i, p in enumerate(PANELS):
        ox = X0 + (i % COLS) * (PANEL_W + PAD_X)
        oy = Y0 + (i // COLS) * (PANEL_H + PAD_Y)
        out.append(panel_svg(p, ox, oy))

    # takeaway cell in the empty 6th slot
    ox = X0 + 2 * (PANEL_W + PAD_X)
    oy = Y0 + 1 * (PANEL_H + PAD_Y)
    out.append(f'<rect x="{ox}" y="{oy}" width="{PANEL_W}" height="{PANEL_H}" '
               f'rx="7" fill="#f3f6fa" stroke="#d6dde6" stroke-width="1"/>')
    lines = [
        ("What the arc shows", True),
        ("", False),
        ("The world model turned the right way but at ¼", False),
        ("the physical magnitude. Predictor-only post-", False),
        ("training moved the gain 0.27 → 0.97 without", False),
        ("touching encoder, head or emission.", False),
        ("", False),
        ("On the repaired trunk the fan is healthy and", False),
        ("the roll-cost becomes the best-calibrated", False),
        ("selection signal the programme has measured", False),
        ("(ρ 0.716 vs ≤ 0.26 for every learned scorer).", False),
        ("", False),
        ("Imagination is load-bearing, not decorative:", False),
        ("removing its content costs 19× in ADE, and", False),
        ("breaking only the correspondence still costs 3×.", False),
    ]
    yy = oy + 26
    for text, bold in lines:
        if text:
            out.append(f'<text x="{ox+16}" y="{yy}" font-size="'
                       f'{"13.5" if bold else "11.2"}" '
                       f'font-weight="{"700" if bold else "400"}" '
                       f'fill="{INK if bold else INK2}">{esc(text)}</text>')
        yy += 15 if not bold else 21

    out.append(f'<text x="40" y="{H-22}" font-size="9.8" fill="{MUTED}">'
               'Selection is NOT closed by this arc: the selector-free full-fan run '
               'has since failed the same ≤ 0.4505 gate at 3.335 m over a 0.127 m '
               'oracle — see Figure 3. Failed gates are reported as results.</text>')
    out.append("</svg>")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v58f_results.svg")
    svg = "\n".join(out)
    open(p, "w", encoding="utf-8").write(svg)
    print("wrote", p)
    try:
        import cairosvg
        pp = p.replace(".svg", ".png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=pp, scale=2.0)
        print("wrote", pp)
    except ImportError:
        print("cairosvg absent — SVG only")


if __name__ == "__main__":
    main()
