"""Render the overnight validation report as a single self-contained page."""
from __future__ import annotations

import base64
import html
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data import alpamayo_records as AR    # noqa: E402
from tanitad.data import egomotion_source as ES    # noqa: E402
from tanitad.models import vocab_v7 as V7          # noqa: E402

CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
OFFS = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
LAB = json.load(open("C:/Users/Admin/tanitad-wt/_s2build/report/sample_labels.json"))
SUM = json.load(open("C:/Users/Admin/tanitad-wt/_s2build/v7_time/summary.json"))

#: My verdict per clip, written from the FRAMES before the labels were read.
VERDICT = {
    "e55a16e8": (True, "Signalised intersection, several light heads on masts, a lead car at the "
                       "stop line. The scene is essentially FROZEN across all 10 s — the ego is "
                       "stopped. Labels: STOP_POINT + TRAFFIC_LIGHT_REACT_RED + lon=HOLD. Correct "
                       "on every axis, and Alpamayo's <code>Stop</code> state agrees."),
    "5355f3cc": (True, "A pedestrian with a backpack crosses the zebra directly in front of the ego "
                       "at +2 s and clears it by +6 s. Labels: STOP_POINT + YIELD with "
                       "lon=BRAKE_TO, and Alpamayo names <code>pedestrian at crosswalk</code> as "
                       "the critical component. The case the pipeline most needs to get right, and "
                       "it does. <b>lon read FOLLOW before tonight.</b>"),
    "ec075947": (True, "Urban junction: the view sweeps ~90° past a corner building. A genuine "
                       "junction turn — tight arc, low speed — so it survives the new turn gate: "
                       "TURN_R tactical, lat=TURN_R, NAV_TURN_R. Strategic is now "
                       "<b>FOLLOW_ROUTE</b>, correctly: the turn happens inside the plan, and "
                       "nothing else falls in the 8–30 s band."),
    "43bbcbf9": (True, "⭐ <b>The clip that exposed three defects at once.</b> The PI said he saw "
                       "no turning manoeuvre here and was right: it is a motorway S-bend — +38° by "
                       "t+3 s, back to +24° by t+6 s, then −53° by t+14 s — driven at 12.0→14.3 m/s "
                       "<b>while accelerating</b>. It was labelled TURN_L with radius 40 m, a number "
                       "from peak instantaneous curvature; the arc radius is <b>144 m</b>. It now "
                       "emits no turn at all: LANE_KEEP + ADAPT_SPEED_FOR_CURVE, "
                       "NAV_FOLLOW_ROAD, strategic FOLLOW_ROUTE."),
    "683d37fb": (True, "⚠️ <b>I read this wrong and the data corrected me.</b> On the montage I saw "
                       "'empty dusk road' and called its EVADE a false positive. Cropping "
                       "Alpamayo's box at full resolution shows <b>a real pedestrian walking in the "
                       "road</b>, lit by the headlights. EVADE + NUDGE_L is <b>correct</b>; my "
                       "400 px thumbnail was the unreliable instrument."),
    "d94365be": (False, "⭐ <b>The clip whose ONCOMING timing you questioned — you were right to.</b> "
                        "<code>REACT_ON_ONCOMING</code> is marked <b>untimed</b>: nothing in the "
                        "source places it in 2–6 s. This clip has <b>no motion segments and no "
                        "components analysis at all</b>, so there is no time evidence of any kind, "
                        "and Alpamayo's anchor sits 2.9 s before ours — an event it names can have "
                        "finished before our band opens, which is what you saw in the frames. Worse, "
                        "its two text fields contradict each other: <i>cot</i> says nudge LEFT for a "
                        "parked car, <i>chain_of_causation</i> says nudge RIGHT for an oncoming "
                        "vehicle. Corpus-wide <b>REACT_ON_ONCOMING is 0 timed / 344 untimed</b>."),
    "59b57590": (True, "⭐ <b>The clip whose MERGE you questioned — you were right, there is no "
                       "merge.</b> The only occurrence of \"merg\" in the whole text is "
                       "<i>\"potential door-opening/merge hazards\"</i>, a hypothetical risk class "
                       "in a compound noun. That token is gone. <b>The lane change the CoT claims "
                       "is deliberately NOT emitted</b>: measured with the road's arc removed, this "
                       "ego displaces <b>0.01–0.05 m</b> in every window — a lane is ~3.5 m, so it "
                       "never changes lane in view. The CoT describes an intent the clip does not "
                       "execute, and the lateral-evidence gate is right to withhold it."),
    "472944a4": (True, "⚠️ <b>My second wrong call, corrected by the box.</b> I read the growing "
                       "light as an oncoming headlight and reported REACT_ON_ONCOMING as missing. "
                       "The box, cropped at full resolution, shows <b>a car ahead with red TAIL "
                       "lights</b> — same direction, not oncoming — and the source says "
                       "<code>merging vehicle from the left</code>. YIELD + MERGE + GAP_TARGET is "
                       "<b>correct</b>. At night, on a downscaled tile, I cannot reliably tell "
                       "headlights from tail-lights."),
    "0d932392": (True, "Night elevated road, no traffic, no light anywhere in 10 s. It emitted a "
                       "phantom TRAFFIC_LIGHT_REACT when I first reviewed it; that token is now "
                       "gone and the label is SPEED_BAND alone. <b>This clip exposed the 408× "
                       "negation inversion</b> — 515 false light tokens across the corpus."),
    "d452ea24": (True, "Red lights at −4 s, a crossing, and the ego passing a bus from +2 s to "
                       "+6 s. It was labelled FOLLOW_LANE and nothing else when I first reviewed "
                       "it. The source reads <i>'nudge left to pass the stopped bus'</i> and "
                       "geometry shows NUDGE_L — <b>a stopped vehicle is a STATIC obstacle, so this "
                       "is EVADE, not OVERTAKE</b>, which is now what it emits. The traffic light "
                       "at −4 s is still not captured."),
    "295aba84": (None, "Night multi-lane arterial, a white pickup close in the adjacent left lane "
                       "at −4 s. YIELD + MERGE with critical component <code>merging vehicle from "
                       "the left</code> is <b>plausible but not confirmable</b> from six frames. "
                       "Recorded as unverified rather than counted either way."),
    "6faad52e": (None, "Night suburban, a vehicle passes on the left at −4 s, then a railed bridge. "
                       "The strategic TURN_LEFT_FOLLOW_ROUTE is for an 86° turn at <b>t+26 s</b> — "
                       "well beyond where the frame strip ends, so it is <b>unverifiable from this "
                       "evidence</b>, not wrong."),
}


def b64(fig=None, img=None, fmt="png") -> str:
    b = io.BytesIO()
    if fig is not None:
        fig.savefig(b, format=fmt, dpi=104, bbox_inches="tight")
        plt.close(fig)
    else:
        img.save(b, format=fmt, quality=80)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def strip(cid: str) -> str:
    p = f"{CAM}/{cid}.mp4"
    if not os.path.exists(p):
        return '<p class="warn">no frames available</p>'
    import av
    from PIL import Image
    tiles = {}
    with av.open(p) as c:
        fps = float(c.streams.video[0].average_rate or 30.0)
        want = {int(round((8.0 + o) * fps)): o for o in OFFS}
        last = max(want)
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                im = fr.to_image()
                im.thumbnail((330, 190), Image.LANCZOS)
                tiles[want[i]] = im
            if i > last:
                break
    return "".join(
        f'<figure><img src="{b64(img=tiles[o], fmt="JPEG")}">'
        f'<figcaption>{"KEY t=8.0s" if o == 0 else f"{o:+.0f}s"}</figcaption></figure>'
        for o in sorted(tiles))


def bev(cid: str, rec: dict) -> str:
    tr = ES.load(cid, max_s=8.0 + 32.0)
    k, p = tr.key_index, tr.poses
    c, s = np.cos(-p[k, 2]), np.sin(-p[k, 2])
    dx, dy = p[:, 0] - p[k, 0], p[:, 1] - p[k, 1]
    ex, ey = c * dx - s * dy, s * dx + c * dy
    n6 = int(round(6.0 * ES.HZ))
    hi6 = min(len(p) - 1, k + n6)
    fig, ax = plt.subplots(figsize=(3.0, 3.9))
    ax.plot(ey[:k + 1], ex[:k + 1], color="#999", lw=1.5, label="past")
    ax.plot(ey[k:hi6 + 1], ex[k:hi6 + 1], color="#0aa06a", lw=3.0, label="tactical 2-6 s")
    ax.plot(ey[hi6:], ex[hi6:], color="#2f7fd0", lw=1.7, ls="--", label="strategic band")
    ax.plot([0], [0], "o", color="#d0342c", ms=7, zorder=6, label="anchor 8.0 s")
    a = rec["g_tac"]["anchor"]
    ax.plot([a["goal_y_m"]], [a["goal_x_m"]], "*", color="#e8890c", ms=14, zorder=6,
            label="6 s goal anchor")
    ax.set_xlabel("lateral (m)", fontsize=7)
    ax.set_ylabel("along-track (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.axis("equal"); ax.grid(alpha=.22); ax.invert_xaxis()
    ax.legend(fontsize=5.4, loc="best", framealpha=.9)
    return b64(fig=fig)


def badge(prov, disputed, grounded):
    if grounded:
        return '<span class="p g">box-grounded</span>'
    if prov == "geometry":
        return '<span class="p geo">geometry</span>'
    if prov == "alpamayo-structured":
        return '<span class="p as">alpamayo-struct</span>'
    if disputed:
        return '<span class="p d">vlm-cot · disputed</span>'
    return f'<span class="p">{html.escape(str(prov))}</span>'


def clip_section(cid: str, rec: dict) -> str:
    ok, note = VERDICT.get(cid[:8], (None, ""))
    al = rec.get("alpamayo") or {}
    lon, lat = al.get("longitudinal") or {}, al.get("lateral") or {}
    comp = al.get("critical_component") or {}
    arec = AR.get(cid)

    rows = []
    for t, a in rec["g_tac"]["goals"].items():
        a = a if isinstance(a, dict) else {}
        args = {k: v for k, v in a.items()
                if k not in ("provenance", "disputed", "grounded", "grounding", "reason")}
        rows.append(
            f'<tr><td class="tok">{t}</td><td class="args">'
            f'{html.escape(json.dumps(args)) if args else "—"}</td>'
            f'<td>{badge(a.get("provenance", "geometry"), a.get("disputed"), a.get("grounded"))}'
            f'{"" if not a.get("time_basis") else (chr(32) + chr(60) + "span class=" + chr(34) + "p seg" + chr(34) + chr(62) + "timed" + chr(60) + "/span" + chr(62)) if a.get("time_basis") == "segment" else (chr(32) + chr(60) + "span class=" + chr(34) + "p untimed" + chr(34) + chr(62) + "untimed" + chr(60) + "/span" + chr(62))}'
            f'</td></tr>')

    src = rec.get("cot_source") or {}
    ma = src.get("meta_action") or {}
    meta_html = (" · ".join(f"<b>{html.escape(k)}</b> {html.escape(str(v))}"
                            for k, v in ma.items()) or "—")
    chain_html = html.escape(src.get("chain_of_causation") or "—")
    if (src.get("conflict") or {}).get("conflict"):
        chain_html += ('<br><span class="conflictwarn">&#9888; CONTRADICTS '
                       '<code>cot</code> on direction — DROPPED for extraction</span>')
    comp_html = html.escape((src.get("components_analysis") or "—")[:1400]).replace(chr(10), "<br>")
    motion_html = html.escape((src.get("motion_analysis") or "—")[:1400]).replace(chr(10), "<br>")
    cls = "ok" if ok is True else ("bad" if ok is False else "unk")
    mark = "✓" if ok is True else ("✗" if ok is False else "?")
    return f"""
<section class="clip">
 <h3><span class="cid">{cid[:8]}</span>
     <span class="v {cls}">{mark} {"confirmed" if ok else ("defect" if ok is False else "unverifiable")}</span></h3>
 <div class="strip">{strip(cid)}</div>
 <div class="row">
  <img class="bev" src="{bev(cid, rec)}" alt="BEV">
  <div class="lab">
   <table>
    <tr><th colspan="3">STRATEGIC · 8–30 s</th></tr>
    <tr><td class="tok"><span class="role">goal</span> {rec['g_str']['token']}</td>
        <td class="args">{html.escape(json.dumps(rec['g_str']['args'])) or '—'}</td>
        <td>{badge(rec['g_str']['provenance'], False, False)}</td></tr>
    <tr><td class="tok"><span class="role">action</span> {rec['a_str']['token']}</td>
        <td class="args">{html.escape(json.dumps(rec['a_str']['args'])) or '—'}</td>
        <td>{badge(rec['a_str']['provenance'], False, False)}</td></tr>
    <tr><th colspan="3">TACTICAL GOALS · 2–6 s · multi-label · GOAL = what to achieve</th></tr>
    {''.join(rows)}
    <tr><th colspan="3">TACTICAL ACTIONS</th></tr>
    <tr><td class="tok">lat · {rec['a_tac']['lat']}</td>
        <td class="args">{html.escape(json.dumps(rec['a_tac'].get('lat_args', {})))}</td>
        <td>{badge('geometry', False, False)}</td></tr>
    <tr><td class="tok">lon · {rec['a_tac']['lon']}</td>
        <td class="args">{html.escape(json.dumps(rec['a_tac'].get('lon_args', {})))}</td>
        <td>{badge('geometry', False, False)}</td></tr>
    <tr><th colspan="3">NAV COMMAND · model INPUT, oracle</th></tr>
    <tr><td class="tok">{rec['nav_command']['token']}</td>
        <td class="args">{html.escape(json.dumps(rec['nav_command'].get('args', {})))}</td>
        <td><span class="p or">ego-future oracle</span></td></tr>
   </table>
  </div>
 </div>
 <div class="alpa">
  <b>Alpamayo</b> · longitudinal <code>{lon.get('phrase')}</code>
  (calibrated {lon.get('dv_expected_ms')} m/s, measured {lon.get('dv_measured_ms')} m/s —
  <b>{'agrees' if lon.get('agree') else 'disagrees'}</b>, scored on its own window)
  · lateral <code>{lat.get('alpamayo_side')}</code>
  <b>{'agrees' if lat.get('agree') else 'disagrees'}</b><br>
  <b>critical component</b> <code>{comp.get('type')}</code>
  {'<i>(explicit NONE)</i>' if comp.get('is_explicit_none') else ''}
  · <b>box</b> {', '.join(al.get('boxes', [])) or '—'}
 </div>
 <details class="src" open>
  <summary>Alpamayo source, verbatim — meta_action and all four text fields</summary>
  <table class="srct">
   <tr><th>meta_action</th><td>{meta_html}</td></tr>
   <tr><th>cot</th><td>{html.escape(src.get('cot') or '—')}</td></tr>
   <tr><th>chain_of_causation</th><td>{chain_html}</td></tr>
   <tr><th>critical_components</th><td>{comp_html}</td></tr>
   <tr><th>motion_analysis</th><td>{motion_html}</td></tr>
  </table>
 </details>
 <div class="verdict {cls}"><b>My reading of the frames:</b> {note}</div>
</section>"""


def defs_table() -> str:
    groups = [("Strategic GOALS · 8–30 s · WHAT comes next on the route, and when", V7.STRATEGIC_GOAL_TOKENS_V7),
              ("Strategic ACTIONS · what to DO NOW about that goal — never changes the current tactical manoeuvre", V7.STRATEGIC_ACTION_TOKENS_V7),
              ("Tactical GOALS · 2–6 s · multi-label · what is to be achieved", V7.TACTICAL_GOAL_TOKENS_V7),
              ("Tactical lateral ACTIONS · the control output, 2–6 s", V7.TACTICAL_LAT_ACTIONS_V7),
              ("Tactical longitudinal ACTIONS · the control output, 2–6 s", V7.TACTICAL_LON_ACTIONS_V7),
              ("Nav commands · model INPUT, not a target", V7.NAV_COMMAND_TOKENS)]
    out = []
    for title, toks in groups:
        rows = "".join(
            f'<tr><td class="tok">{t}</td><td>{html.escape(V7.DEFINITIONS.get(t, "—"))}</td></tr>'
            for t in toks)
        out.append(f'<h3>{title} <span class="n">{len(toks)} tokens</span></h3>'
                   f'<table class="defs">{rows}</table>')
    return "".join(out)


def dist_chart() -> str:
    g = SUM["g_tac"]
    ks = list(g)[:16][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(range(len(ks)), [g[k] for k in ks], color="#2f7fd0")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=7.5, family="monospace")
    ax.set_xlabel("clips (of 4,719)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=.25)
    for i, k in enumerate(ks):
        ax.text(g[k] + 12, i, str(g[k]), va="center", fontsize=6.6)
    return b64(fig=fig)
