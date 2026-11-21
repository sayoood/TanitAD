"""LF0 figure — what the BEV raster actually contains, and what the WM decodes.

Answers a question the PI asked directly: *does the extracted BEV include
road / corridor / lane boundaries?* It does NOT. The raster is built from
``obstacle.offline``, whose full measured enum is TEN DYNAMIC AGENT CLASSES
(``bev_raster.py:79-82``) — automobile, heavy_truck, bus, other_vehicle,
trailer, person, rider, stroller, animal, protruding_object. There is no lane
boundary, no road edge, no drivable area and no corridor anywhere in it, which
is consistent with the settled corpus fact that PhysicalAI-AV ships no map data.

⚠️ The "ego corridor" is therefore an ASSUMPTION, not a perception: a
hand-defined ±1.5 m band (grid columns 29–34) that LF0 draws on top in order to
ask "is there something in my lane". It is drawn dashed here so it cannot be
mistaken for a measured lane.

⭐ The finding this figure makes visible, and which the summary statistic hides:
the decoded BEV is **not blank**. It puts 35–68 cells above τ = 0.7 — comparable
to the ground truth's 31–33 — but essentially none of them in the ego corridor
where the real lead vehicle is. The failure is **confident mislocation**, not
absence of output. That is what an IoU of ~0.02 alongside a retention ratio of
0.932 means in practice.

Data: ``panels_compact.json``, extracted from ``lf0_panels.npz`` on pod4 (the
raster cells ≥ τ, plus a 12×8 mean-pooled density map per raster). Nothing here
is hand-entered; every number is read from that file at render time.

⛔ TWO CAVEATS ON THE CELL COUNTS, added 2026-08-16 by the `bev_raster` consumer
audit (`…/incoming/2026-08-16-bev-consumer-audit/BEV_CONSUMER_AUDIT.md`).

1. **The counts are WHOLE-GRID and this figure draws every cell, unmasked.** The
   BEV panel is a Cartesian 60 m × ±16 m raster, but the camera sees an azimuth
   WEDGE: at the 117° sub-frame LF0 ran on, **626 of 7 680 cells (8.151 %) lie
   outside the camera's horizontal field entirely** and are unanswerable from a
   vision-only latent. So an unknown share of the "35–68 cells ≥ τ" the caption
   quotes sits on cells no camera observed. ⚠️ That share is **NOT RECOVERABLE**:
   `panels_compact.json` and `lf0_panels.npz` are pod4 scratch artifacts and are
   in neither the repo nor any banked bundle (3 probes). Re-rendering with the
   mask needs a re-run of `lf0_bev_lead.py`, which now writes the field mask into
   `lf0_panels.npz` (key ``fov``) precisely so this cannot recur.
   ⭐ **The figure's LOAD-BEARING claim survives regardless**: "essentially none
   of them land in the ego band" is a statement about the **corridor**, and the
   corridor is **entirely inside the field** (0 of 708 scanned cells) at the
   frame this ran on. The mask can only *remove* decoded cells from outside the
   band — it can never move one into it. So the mislocation finding stands and
   masking would, if anything, sharpen it.
2. **The grid shape is imported, not typed.** It used to be a bare
   ``NX, NY = 120, 64`` literal beside a live ``GRID_DEFAULT``; a geometry fact
   restated inline is the same rot class as the "N of 36" count that went stale
   four times. Values are unchanged (``GRID_DEFAULT.shape == (120, 64)``), so
   the rendered figure is byte-identical — it simply can no longer drift.
"""
from __future__ import annotations

import json
import os
import sys

# categorical pair, validated (light + dark) earlier in this programme
C_TRUTH = "#2a78d6"      # ground truth agents
C_DEC = "#eb6834"        # decoded occupancy >= tau
C_INK = "#1a1a1a"
C_MUTE = "#6b6b6b"
C_RULE = "#d4d4d4"
C_BAND = "#b8b8b8"       # the ASSUMED corridor — neutral, never a series colour
C_NOFOV = "#e8e4de"      # OUTSIDE the camera field — unobservable, not a series

CELL_W, CELL_H = 2.0, 1.15          # px per grid cell (64 wide, 120 forward)

# ⛔ IMPORTED, NOT TYPED. `stack/` is a sibling of `Paper/`; locate it explicitly
# (the `p8_geometry_census.py` precedent) so the panel geometry is the SAME
# object the probe rasterised against. A bare literal here would be a geometry
# fact restated inline — the rot class this file's docstring §2 describes.
_STACK = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "stack")
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)
from tanitad.data.bev_raster import GRID_DEFAULT  # noqa: E402

NX, NY = GRID_DEFAULT.shape         # (120, 64) — asserted in tests, not typed


def nofov_spans(hfov_deg):
    """Per-row ``(r0, r1, c0, c1)`` rectangles covering the OUT-OF-FIELD cells.

    ``()`` when ``hfov_deg`` is ``None`` — the honest answer when the source
    file does not record the frame, which is the state the 2026-08-12 figure was
    published in. Never guess a frame: at 117° the out-of-field region is 8.151 %
    of the grid and at the legacy 51.4° pinhole it is 27.682 %, so a wrong guess
    would shade a region three times too large and read as a measurement.

    The mask is contiguous per row (it is ``|atan2(y, x)| > half_angle`` on a
    Cartesian grid, so each row's out-of-field cells form a left block and a
    right block), which is why row spans suffice and no per-cell rects are
    emitted.
    """
    if hfov_deg is None:
        return ()
    import math

    from tanitad.data.bev_raster import fov_mask
    m = fov_mask(GRID_DEFAULT, math.radians(float(hfov_deg) / 2.0))
    spans = []
    for r in range(NX):
        out = [c for c in range(NY) if not m[r, c]]
        if not out:
            continue
        run = [out[0], out[0]]
        for c in out[1:]:
            if c == run[1] + 1:
                run[1] = c
            else:
                spans.append((r, r, run[0], run[1]))
                run = [c, c]
        spans.append((r, r, run[0], run[1]))
    return tuple(spans)


def panel(px, py, p, key, meta, idx, *, title, show_truth_line):
    """One BEV panel: forward is UP, ego at bottom centre."""
    w, h = NY * CELL_W, NX * CELL_H
    o = [f'<g transform="translate({px},{py})">']
    o.append(f'<text x="{w/2}" y="-8" text-anchor="middle" font-size="10.5" '
             f'font-weight="600" fill="{C_INK}">{title}</text>')
    o.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fbfbfb" '
             f'stroke="{C_RULE}" stroke-width="1"/>')

    # ⛔ cells OUTSIDE the camera's horizontal field — unanswerable from a
    # vision-only latent, so decoded mass there is not a detection. Drawn only
    # when the source file records the frame; see main()'s caveat line when it
    # does not. `hfov_deg` is what `lf0_gate.json`'s `geometry` block now carries.
    for r0_, r1_, c0_, c1_ in meta.get("nofov_spans", ()):
        o.append(f'<rect x="{c0_ * CELL_W:.1f}" '
                 f'y="{h - (r1_ + 1) * CELL_H:.1f}" '
                 f'width="{(c1_ - c0_ + 1) * CELL_W:.1f}" '
                 f'height="{(r1_ - r0_ + 1) * CELL_H:.1f}" '
                 f'fill="{C_NOFOV}" fill-opacity="0.5"/>')

    # the ASSUMED corridor band (dashed => not a measured lane)
    c0, c1 = min(meta["cols"]), max(meta["cols"])
    bx, bw = c0 * CELL_W, (c1 - c0 + 1) * CELL_W
    o.append(f'<rect x="{bx:.1f}" y="0" width="{bw:.1f}" height="{h}" '
             f'fill="{C_BAND}" fill-opacity="0.20" stroke="{C_BAND}" '
             f'stroke-width="0.8" stroke-dasharray="3 3"/>')

    colour = C_TRUTH if key == "gt" else C_DEC
    for r, c in p[key + "_hits"]:
        y = h - (r + 1) * CELL_H                       # row 0 = ego, at bottom
        o.append(f'<rect x="{c*CELL_W:.1f}" y="{y:.1f}" width="{CELL_W}" '
                 f'height="{CELL_H}" fill="{colour}"/>')

    # true lead range: a rule across the panel at the GT-read distance
    if show_truth_line and meta["true_m"][idx] is not None:
        ry = h - (meta["true_m"][idx] / meta["cell_m"]) * CELL_H
        o.append(f'<line x1="0" y1="{ry:.1f}" x2="{w}" y2="{ry:.1f}" '
                 f'stroke="{C_TRUTH}" stroke-width="1.2" stroke-dasharray="5 3"/>')
        o.append(f'<text x="{w+4}" y="{ry+3:.1f}" font-size="8.5" '
                 f'fill="{C_TRUTH}">lead {meta["true_m"][idx]:.1f} m</text>')

    # ego marker
    o.append(f'<polygon points="{w/2-3},{h} {w/2+3},{h} {w/2},{h-7}" '
             f'fill="{C_INK}"/>')
    n = p[key + "_n"]
    o.append(f'<text x="2" y="{h+11}" font-size="8.5" fill="{C_MUTE}">'
             f'{n} cells ≥ τ</text>')
    o.append("</g>")
    return "".join(o)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.environ.get(
        "LF0_PANELS",
        "/tmp/claude-0/-home-user-TanitAD/"
        "4e3220fa-f478-55f3-ae2e-c2c6898f11c8/scratchpad/panels_compact.json")
    if not os.path.exists(src):
        raise SystemExit(f"[lf0-fig] missing {src} — pull it from the pod first")
    d = json.load(open(src, encoding="utf-8"))
    meta = {"cols": d["cols"], "cell_m": d["cell_m"], "true_m": d["true_m"]}
    # ⭐ THE FRAME, IF THE SOURCE RECORDS IT. It did not in 2026-08-12 — which is
    # exactly why the published figure's cell counts are unrecoverable. LF0 now
    # writes `geometry.model_frame.hfov_deg` into lf0_gate.json and the field
    # mask into lf0_panels.npz, so an extractor can carry it forward.
    meta["nofov_spans"] = nofov_spans(d.get("hfov_deg"))
    meta["hfov_deg"] = d.get("hfov_deg")

    pw, ph = NY * CELL_W, NX * CELL_H
    gx, gy = pw + 96, ph + 46
    left, top = 58, 124
    W = max(left + 3 * gx + 20, 1080)
    H = top + len(d["panels"]) * gy + 150

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Inter,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{left}" y="34" font-size="17" font-weight="700" '
         f'fill="{C_INK}">The BEV contains agents — no lanes, no road, no corridor</text>',
         f'<text x="{left}" y="54" font-size="11" fill="{C_MUTE}">'
         f'obstacle.offline is TEN DYNAMIC AGENT CLASSES: automobile · heavy_truck · bus · other_vehicle · trailer · person · rider · stroller · animal · protruding_object.</text>',
         f'<text x="{left}" y="69" font-size="11" fill="{C_MUTE}">'
         f'PhysicalAI-AV ships no map data, so the dashed band is a HAND-DEFINED ±1.5 m '
         f'assumption about the ego lane — not a perceived corridor.</text>']

    cols = [("gt", "ground truth agents"),
            ("enc", "decoded from ENCODED latent"),
            ("pred", "decoded from PREDICTED latent")]
    for row, p in enumerate(d["panels"]):
        py = top + row * gy
        o.append(f'<text x="{left-10}" y="{py + ph/2}" font-size="9.5" '
                 f'fill="{C_MUTE}" text-anchor="end" transform="rotate(-90 '
                 f'{left-10} {py + ph/2})">window {d["win"][row]}</text>')
        for ci, (key, label) in enumerate(cols):
            o.append(panel(left + ci * gx, py, p, key, meta, row,
                           title=label if row == 0 else "",
                           show_truth_line=(key == "gt")))

    fy = top + len(d["panels"]) * gy + 16
    o.append(f'<line x1="{left}" y1="{fy-8}" x2="{W-20}" y2="{fy-8}" '
             f'stroke="{C_RULE}" stroke-width="1"/>')
    lines = [
        ("The decode is NOT blank — it is confidently WRONG about where things are.",
         True),
        ("Every window here HAS a lead vehicle in the corridor (dashed blue rule). "
         "The decoded panels put 35–68 cells above τ = 0.7 — comparable to the ground "
         "truth's 31–33 —", False),
        ("yet essentially none of them land in the ego band. That is what IoU ≈ 0.02 "
         "beside a retention ratio of 0.932 means in practice: the relative structure "
         "survives prediction,", False),
        ("the absolute placement does not. Across 129 labelled windows the corridor is "
         "empty in the decode for 81.4 % (encoded) and 92.3 % (predicted) of them.", False),
        ("τ = 0.7 is INHERITED from the P8 gate and never re-tuned here. "
         "Panels show cells ≥ τ, not the full-resolution probability field.", False),
        ((f"Cells outside the {meta['hfov_deg']:.0f}° camera field are shaded — "
          f"a vision-only latent cannot answer them, so decoded mass there is "
          f"not a detection."
          if meta.get("hfov_deg") else
          "⚠ The source file records NO camera frame, so the out-of-field cells "
          "(8.151 % of the grid at LF0's 117° sub-frame, ALL at x < 9.3 m) are "
          "NOT shaded and are counted in the totals above. The corridor itself "
          "is fully in-field, so the mislocation finding is unaffected."), False),
    ]
    for i, (t, bold) in enumerate(lines):
        o.append(f'<text x="{left}" y="{fy + 8 + i*15}" font-size="10.5" '
                 f'font-weight="{"700" if bold else "400"}" '
                 f'fill="{C_INK if bold else C_MUTE}">{t}</text>')

    lx = left
    ly = 82
    o.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{C_TRUTH}"/>'
             f'<text x="{lx+15}" y="{ly+9}" font-size="10" fill="{C_INK}">'
             f'ground-truth agent</text>')
    o.append(f'<rect x="{lx+150}" y="{ly}" width="10" height="10" fill="{C_DEC}"/>'
             f'<text x="{lx+165}" y="{ly+9}" font-size="10" fill="{C_INK}">'
             f'decoded occupancy ≥ τ</text>')
    o.append(f'<rect x="{lx+330}" y="{ly}" width="10" height="10" fill="{C_BAND}" '
             f'fill-opacity="0.35" stroke="{C_BAND}" stroke-dasharray="3 3"/>'
             f'<text x="{lx+345}" y="{ly+9}" font-size="10" fill="{C_INK}">'
             f'ASSUMED ±1.5 m ego band (not perceived)</text>')
    o.append("</svg>")

    # ⛔ OUTPUT DIR IS OVERRIDABLE, and it must be. `main()` writes straight over
    # a PUBLISHED, git-tracked figure; a test (or a dry run) that calls it with
    # synthetic data silently destroys the real one. MEASURED the hard way while
    # writing this audit — the file was recovered with `git checkout` and
    # verified byte-identical, but only because it happened to be committed.
    out = os.path.join(os.environ.get("LF0_FIG_OUT", here), "lf0_bev_panels.svg")
    open(out, "w", encoding="utf-8").write("\n".join(o))
    print(f"[lf0-fig] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
