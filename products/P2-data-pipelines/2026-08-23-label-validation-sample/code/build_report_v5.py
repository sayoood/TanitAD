"""Render the label-provenance report: every layer, and where each value came from."""
from __future__ import annotations

import collections
import html
import json
from pathlib import Path

V = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/label_validation_report.html")

SHOW = ["0e56dae2", "2cf5d4c8", "416601c0", "3a0165bd", "1ad7bf7b", "e850f1fb",
        "82b8780b", "e084c7c3", "12bb97af", "01b24287", "028eff14", "84f5bb0d",
        "6c3688c0", "62a7e92a"]

JUDGE = {
 "0e56dae2": "The scene we analysed together, now layered the way you described it. The turn begins at t+2.5 s and completes inside the 6 s plan, so it is <b>tactical</b>: <span class='mono'>a_tac.lat = TURN_L</span>. Across the strategic band (8–30 s) the ego simply follows the road, so <span class='mono'>g_str = FOLLOW_MAIN_ROAD</span> and <span class='mono'>a_str = HOLD_CORRIDOR</span> — “follow the road after turning”. Note the VLM does <i>not</i> corroborate a junction manoeuvre here: its CoT says “keep speed through the intersection”, a junction referent with no turn verb. Geometry says turn; the disagreement is recorded, not hidden.",
 "2cf5d4c8": "Validated by eye. Right turn into a residential street: rotation between t0 and t+2s, decelerating 9.2→5.5. TURN_RIGHT at R=10.1 m.",
 "416601c0": "Validated by eye. Stopped at a red light through t+2s, green at t+4s, then launches and rotates. The old label said FOLLOW_MAIN_ROAD; the ego does not go straight. Note the stop is correctly demoted to the tactical layer (STOP_POINT) while the turn carries the strategic goal.",
 "3a0165bd": "A composite the frames make honest: the ego stops, creeps, stops again, and turns later. Both STOP_AT and TURN_RIGHT are defensible. The rule here is a design choice, not a measurement — the turn takes the strategic slot because it is the route decision, and the stop appears in g_tac as STOP_POINT.",
 "1ad7bf7b": "Validated by eye. Stopped at a red light with a pedestrian crossing, green, then straight down the boulevard with no rotation. FOLLOW_MAIN_ROAD correct — a genuine negative, which is what a false-negative check needs.",
 "e850f1fb": "Validated by eye. Straight down a narrow old-town street the whole way, decelerating into a square at the end. FOLLOW_MAIN_ROAD correct; the corridor offset is the ego passing a parked car.",
 "82b8780b": "Red at t−4s, green at t−2s, then 0.0→12.1 m/s and a right turn onto a boulevard. Alpamayo's CoT reads “resume speed from stop since the traffic light turns green” — one of the cases where the VLM is exactly right, and the frames confirm it.",
 "e084c7c3": "The reference controlled stop. Red lights unmistakable on the gantry, 12.4→0.0 and held. Geometry finds the stop; the CoT supplies the reason. Both agree.",
 "12bb97af": "The queue case. Geometry sees a stop; the CoT names a lead vehicle, so the resolved reason is queueing rather than a signal. This is the jam-vs-light distinction you asked for, and it is the only place the VLM changes an outcome.",
 "01b24287": "Kept because it is the clip my very first pass scored WRONG. Over 4 s the ego barely rotates; the turn is real and begins 7.3 s out. The label was right and my window was too short — the reason g_str is now derived over the full observed band.",
 "028eff14": "A 173° rotation at R=6.4 m — a U-turn or a roundabout. The vocabulary has no U_TURN or ROUNDABOUT token, so wide rotations are absorbed into the turn classes. Flagged as a vocabulary gap, not a labelling error.",
 "84f5bb0d": "A stop with essentially no rotation, and the ego is near-stationary across the tactical band — so the anchor goal would land on top of the vehicle. It abstains on the lateral axis instead of emitting a goal point the car never travels to.",
 "6c3688c0": "Tight right turn, R=5.9 m, −88°. No Alpamayo row for this clip, so every value here is geometry — which is the normal case: the VLM covers 100% of aug120 but only 9.3% of the validation split.",
 "62a7e92a": "A narrow rural lane with gentle bends at R=81.6 m. This is precisely the case a yaw-magnitude rule gets wrong: 28.9° of heading change on a country road is geometry, not a junction decision.",
}

SRC_CLASS = {"geometry": "src-geo", "abstain": "src-abs", "vlm": "src-vlm"}


def src_kind(prov: str) -> str:
    p = (prov or "").lower()
    if "vlm" in p:
        return "vlm"
    if "abstain" in p or "unconstrained" in p:
        return "abstain"
    return "geometry"


def src_chip(prov: str) -> str:
    k = src_kind(prov)
    lbl = {"geometry": "EGO GEOMETRY", "vlm": "GEOMETRY + VLM REASON",
           "abstain": "ABSTAIN"}[k]
    return (f'<span class="src {SRC_CLASS[k]}">{lbl}</span>'
            f'<span class="srcd">{html.escape(prov or "")}</span>')


def args_html(d):
    if not d:
        return '<span class="arg none">—</span>'
    live = [f'<span class="arg"><i>{k}</i>{v}</span>' for k, v in d.items()
            if v is not None]
    return "".join(live) or '<span class="arg none">unset</span>'


def bev(t, w=340, h=250):
    px, py, fx, fy = t["px"], t["py"], t["fx"], t["fy"]
    xs, ys = px + fx, py + fy
    if not xs:
        return ""
    sx, sy = [-q for q in ys], [-q for q in xs]
    span = max(max(sx) - min(sx), max(sy) - min(sy), 12.0) * 1.18
    cx, cy = (min(sx) + max(sx)) / 2, (min(sy) + max(sy)) / 2

    def P(a, b):
        return ((a - cx) / span * w + w / 2, (b - cy) / span * h + h / 2)

    def path(ax, ay, i0=0, i1=None):
        pts = [P(-b, -a) for a, b in list(zip(ax, ay))[i0:i1]]
        return " ".join(f"{'M' if i == 0 else 'L'}{q:.1f},{r:.1f}"
                        for i, (q, r) in enumerate(pts))
    bi = t.get("band_i", 80)
    ox, oy = P(0, 0)
    grid = "".join(f'<line x1="0" y1="{g*h/5:.0f}" x2="{w}" y2="{g*h/5:.0f}"/>'
                   f'<line x1="{g*w/5:.0f}" y1="0" x2="{g*w/5:.0f}" y2="{h}"/>'
                   for g in range(1, 5))
    return f'''<svg class="bev" viewBox="0 0 {w} {h}" role="img"
 aria-label="Bird's-eye view in the ego frame: dashed past, solid future, highlighted strategic band">
<g class="bev-grid">{grid}</g>
<path class="bev-past" d="{path(px, py)}"/>
<path class="bev-fut" d="{path(fx, fy, 0, bi + 1)}"/>
<path class="bev-band" d="{path(fx, fy, bi)}"/>
<circle class="bev-ego" cx="{ox:.1f}" cy="{oy:.1f}" r="5"/>
<text class="bev-sc" x="8" y="{h-8}">grid {round(span/4)} m</text></svg>'''


def card(r):
    cid = r["clip_id"][:8]
    g, a, gt, at = r["g_str"], r["a_str"], r["g_tac"], r["a_tac"]
    m, hz, sem = r["manoeuvre"], r["horizon"], r["semantics"]

    figs = ""
    for fm in r["frame_meta"]:
        o = fm["offset_s"]
        cls = "key" if o == 0 else ("band" if fm["strategic"] else "")
        cap = (f'KEY t0 · {fm["v"]} m/s' if o == 0
               else f't{o:+.0f}s · {fm["v"]} m/s')
        if fm["strategic"]:
            cap += " · strat"
        figs += (f'<figure class="{cls}"><img loading="lazy" '
                 f'src="data:image/jpeg;base64,{r["frames"][fm["key"]]}" '
                 f'alt="Forward camera at t{o:+.0f} seconds from the anchor">'
                 f'<figcaption>{cap}</figcaption></figure>')

    rows = [
        ("STRATEGIC", "goal", g["token"], {}, g.get("provenance"), g.get("reason")),
        ("STRATEGIC", "action", a["token"], {}, a.get("provenance"), a.get("reason")),
        ("TACTICAL", "goal · lat", gt["lat_token"], gt["lat_args"],
         gt["lat_provenance"], None),
        ("TACTICAL", "goal · lon", gt["lon_token"], gt["lon_args"],
         gt["lon_provenance"], None),
        ("TACTICAL", "action · lat", at["lat"], {},
         f'geometry({at.get("window_s",[0,6])[0]:.0f}–{at.get("window_s",[0,6])[1]:.0f} s plan · vocab {at.get("vocab","v6.1")})',
         at.get("lateral_class")),
        ("TACTICAL", "action · lon", at["lon"], {},
         f'geometry({at.get("window_s",[0,6])[0]:.0f}–{at.get("window_s",[0,6])[1]:.0f} s plan)', None),
    ]
    # ⚠️ THE ROWSPAN MUST COUNT THE ROWS IT ACTUALLY SPANS. A hardcoded
    # rowspan=2 left TACTICAL labelling only 2 of its 4 rows, so the two action
    # rows appeared to belong to nothing — the inconsistency the PI saw.
    spans = collections.Counter(fam for fam, *_ in rows)
    body, seen = "", set()
    for fam, name, tok, ar, prov, why in rows:
        famcell = ""
        if fam not in seen:
            seen.add(fam)
            famcell = f'<td class="fam" rowspan="{spans[fam]}">{fam}</td>'
        body += (f'<tr>{famcell}<td class="lay">{name}</td>'
                 f'<td><span class="tok {"t-str" if fam=="STRATEGIC" else "t-tac"}">'
                 f'{html.escape(str(tok))}</span></td>'
                 f'<td class="argc">{args_html(ar)}</td>'
                 f'<td class="srcc">{src_chip(prov)}</td>'
                 f'<td class="whyc">{html.escape(why or "")}</td></tr>')

    if sem:
        refs = "".join(f'<span class="ref">{x}</span>'
                       for x in sem.get("referents", []))
        used = (f'stop reason <b>{sem["stop_reason"]}</b> — used ONLY to refine a '
                f'stop geometry already found' if sem.get("stop_reason")
                else 'no stop reason — <b>contributed nothing</b> to this label')
        alp = (f'<div class="alp"><div class="alp-h">ALPAMAYO&#8209;SUPER '
               f'<span class="alp-t">VLM · never supervises kinematics</span></div>'
               f'<p class="cot">&ldquo;{html.escape(sem.get("cot",""))}&rdquo;</p>'
               f'{f"<div class=refs>{refs}</div>" if refs else ""}'
               f'<p class="alp-u">{used}</p></div>')
    else:
        alp = ('<div class="alp none"><div class="alp-h">ALPAMAYO&#8209;SUPER</div>'
               '<p class="alp-u">No inference for this clip — <b>every value above '
               'is ego geometry</b>. Coverage is 100% of aug120, 9.3% of w120val.</p></div>')

    ev = ""
    if m:
        R = m["turn_radius_m"]
        Rs = "&infin;" if (R is None or R > 9e5) else f'{R:.1f} m'
        for k, v in [("lateral class", m["lateral_class"]), ("turn radius", Rs),
                     ("peak yaw", f'{m["peak_yaw_deg"]:+.1f}&deg;'),
                     ("onset", f't+{m["yaw_onset_s"]}s' if m["yaw_onset_s"] else "—"),
                     ("turn segments", m["n_turn_segments"]),
                     ("stop type", m["stop_type"]),
                     ("decel cycles", m["n_decel_events"]),
                     ("speed", f'{m["v_at_key"]:.2f}&rarr;{m["v_end"]:.2f} m/s'),
                     ("horizon used", f'{m["horizon_s"]}s'),
                     ("band observed", f'{hz["band_observable_s"]}/22 s'),
                     ("recording", f'{hz["recording_span_s"]}s'),
                     ("confidence", m["confidence"])]:
            ev += f'<div class="ev"><dt>{k}</dt><dd>{v}</dd></div>'

    return f'''<article class="card">
 <header class="card-h">
  <div class="cid"><span class="mono big">{cid}</span>
   <span class="tag ok">join-free frames</span>
   <span class="tag">horizon {hz["available_s"]}s</span></div>
 </header>
 <div class="strip">{figs}</div>
 <div class="body">
  <div class="left">{bev(r["traj"])}
   <div class="bevkey"><span class="k-past"></span>past<span class="k-fut"></span>future
    <span class="k-band"></span>strategic band</div>{alp}</div>
  <div class="right"><div class="tw"><table class="stack">
   <thead><tr><th>layer</th><th>field</th><th>token</th><th>args</th>
    <th>source of the value</th><th>derivation</th></tr></thead>
   <tbody>{body}</tbody></table></div>
   <dl class="evid">{ev}</dl></div>
 </div>
 <div class="verd"><p><span class="vl">my read</span>{JUDGE.get(cid, "")}</p></div>
</article>'''


def main() -> None:
    rows = {r["clip_id"][:8]: r for r in
            json.loads((V / "sample_v5.json").read_text(encoding="utf-8"))}
    cards = "\n".join(card(rows[c]) for c in SHOW if c in rows)
    doc = TEMPLATE.replace("@@CARDS@@", cards)
    assert "@@" not in doc
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), "
          f"{sum(1 for c in SHOW if c in rows)} cards")


TEMPLATE = r'''<title>Label Provenance Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
 --ground:#F5F7FA;--panel:#FFF;--panel-2:#EDF1F6;--panel-3:#E2E9F1;
 --ink:#0E1620;--ink-2:#43525F;--ink-3:#70818F;
 --line:#D6E0E9;--line-2:#BECBD8;
 --accent:#0057D9;--accent-soft:#E2ECFC;
 --past:#8A9CAE;--band:#7A3DBD;--band-soft:#F0E7FA;
 --ok:#0F7A50;--ok-bg:#E2F3EA;--flag:#B4400C;--flag-bg:#FBE8DF;
 --shadow:0 1px 2px rgba(14,22,32,.06),0 10px 28px -14px rgba(14,22,32,.2);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0A0F14;--panel:#121A22;--panel-2:#19232D;--panel-3:#21303F;
 --ink:#E6EDF4;--ink-2:#A3B3C1;--ink-3:#798A98;
 --line:#212D37;--line-2:#2F3E4B;
 --accent:#5C9BFF;--accent-soft:#14233A;
 --past:#68788A;--band:#B489EE;--band-soft:#1D1530;
 --ok:#4FC08A;--ok-bg:#11281E;--flag:#FF8A5B;--flag-bg:#2D1710;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -16px rgba(0,0,0,.75);
}}
:root[data-theme="dark"]{
 --ground:#0A0F14;--panel:#121A22;--panel-2:#19232D;--panel-3:#21303F;
 --ink:#E6EDF4;--ink-2:#A3B3C1;--ink-3:#798A98;
 --line:#212D37;--line-2:#2F3E4B;
 --accent:#5C9BFF;--accent-soft:#14233A;
 --past:#68788A;--band:#B489EE;--band-soft:#1D1530;
 --ok:#4FC08A;--ok-bg:#11281E;--flag:#FF8A5B;--flag-bg:#2D1710;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -16px rgba(0,0,0,.75);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"Source Sans 3",ui-sans-serif,system-ui,sans-serif;font-size:16px;
 line-height:1.6;-webkit-font-smoothing:antialiased}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;
 font-variant-numeric:tabular-nums}
.wrap{max-width:1560px;margin:0 auto;padding:46px 22px 92px}
h1,h2,h3{font-family:Archivo,ui-sans-serif,system-ui,sans-serif;text-wrap:balance;
 margin:0;line-height:1.14;letter-spacing:-.012em}
h1{font-size:clamp(30px,4.2vw,46px);font-weight:700}
h2{font-size:25px;font-weight:600}
p{margin:0}
.eyebrow{font-family:Archivo,sans-serif;font-size:11.5px;font-weight:600;
 letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
.lede{max-width:72ch;color:var(--ink-2);margin-top:14px;font-size:17px}
header.top{border-bottom:1px solid var(--line);padding-bottom:30px;
 display:flex;flex-direction:column;gap:6px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.mtag{font-size:12.5px;padding:4px 10px;border:1px solid var(--line-2);
 border-radius:2px;color:var(--ink-2);background:var(--panel)}
.mtag.good{border-color:var(--ok);color:var(--ok);background:var(--ok-bg)}
.mtag.warn{border-color:var(--flag);color:var(--flag);background:var(--flag-bg)}
section{margin-top:50px}
.sechead{display:flex;flex-direction:column;gap:6px;margin-bottom:20px}
.sechead p{color:var(--ink-2);max-width:80ch}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 padding:20px 22px;box-shadow:var(--shadow)}
.box h3{font-size:12px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--ink-3);margin-bottom:12px;font-weight:600}
.box p{color:var(--ink-2);font-size:14.5px}.box p+p{margin-top:8px}
.box b{color:var(--ink)}
.bign{font-family:Archivo,sans-serif;font-size:38px;font-weight:700;
 font-variant-numeric:tabular-nums;line-height:1;letter-spacing:-.02em}
.bign.good{color:var(--ok)}.bign.bad{color:var(--flag)}
table{width:100%;border-collapse:collapse;font-size:14px}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
 border-radius:3px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);
 vertical-align:top}
th{font-family:Archivo,sans-serif;font-size:10.5px;text-transform:uppercase;
 letter-spacing:.08em;color:var(--ink-3);font-weight:600;white-space:nowrap}
tr:last-child td{border-bottom:none}
td.n{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
.good{color:var(--ok);font-weight:600}.bad{color:var(--flag);font-weight:600}

.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 margin-bottom:22px;overflow:hidden;box-shadow:var(--shadow)}
.card-h{padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel-2)}
.cid{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.big{font-size:16px;font-weight:600}
.tag{font-size:10.5px;color:var(--ink-3);text-transform:uppercase;
 letter-spacing:.07em;border:1px solid var(--line-2);padding:2px 7px;border-radius:2px}
.tag.ok{color:var(--ok);border-color:var(--ok);background:var(--ok-bg)}
.strip{display:flex;gap:10px;padding:16px 18px;overflow-x:auto;background:var(--panel-3)}
.strip figure{margin:0;flex:0 0 216px;display:flex;flex-direction:column;gap:6px}
.strip img{width:216px;height:216px;object-fit:cover;border-radius:2px;
 border:1px solid var(--line-2);display:block}
.strip figcaption{font-family:"JetBrains Mono",monospace;font-size:10.5px;color:var(--ink-3)}
.strip .key img{border:3px solid var(--accent)}
.strip .key figcaption{color:var(--accent);font-weight:600}
.strip .band img{border:3px solid var(--band)}
.strip .band figcaption{color:var(--band);font-weight:600}
.body{display:grid;grid-template-columns:360px 1fr;gap:20px;padding:16px 18px 6px}
.bev{width:100%;background:var(--panel-2);border:1px solid var(--line);
 border-radius:2px;aspect-ratio:340/250}
.bev-grid line{stroke:var(--line-2);stroke-width:.6}
.bev-past{fill:none;stroke:var(--past);stroke-width:2.2;stroke-dasharray:3 3}
.bev-fut{fill:none;stroke:var(--accent);stroke-width:2.8;stroke-linecap:round}
.bev-band{fill:none;stroke:var(--band);stroke-width:3.4;stroke-linecap:round}
.bev-ego{fill:var(--ink);stroke:var(--panel);stroke-width:1.6}
.bev-sc{fill:var(--ink-3);font-size:10px;font-family:"JetBrains Mono",monospace}
.bevkey{font-family:"JetBrains Mono",monospace;font-size:10px;color:var(--ink-3);
 display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin:6px 0 12px}
.bevkey span{width:13px;height:2px;display:inline-block}
.k-past{background:var(--past)}.k-fut{background:var(--accent)}.k-band{background:var(--band)}
table.stack td.fam{font-family:Archivo,sans-serif;font-size:11px;font-weight:600;
 letter-spacing:.08em;color:var(--ink-3);white-space:nowrap;
 border-right:1px solid var(--line)}
td.lay{font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--ink-2);
 white-space:nowrap}
.tok{font-family:"JetBrains Mono",monospace;font-size:13px;font-weight:600;
 padding:3px 9px;border-radius:2px;white-space:nowrap;display:inline-block}
.tok.t-str{color:var(--accent);border:1px solid var(--accent);background:var(--accent-soft)}
.tok.t-tac{color:var(--band);border:1px solid var(--band);background:var(--band-soft)}
.argc{min-width:120px}
.arg{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--ink-2);
 background:var(--panel-2);border:1px dashed var(--line-2);padding:1px 6px;
 border-radius:2px;margin:0 3px 3px 0;display:inline-block}
.arg i{font-style:normal;color:var(--ink-3);margin-right:4px}
.arg.none{border-style:dotted;color:var(--ink-3)}
.srcc{white-space:nowrap}
.src{font-family:"JetBrains Mono",monospace;font-size:9.5px;font-weight:600;
 letter-spacing:.05em;padding:2px 7px;border-radius:2px;display:inline-block}
.src-geo{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent)}
.src-vlm{background:var(--band-soft);color:var(--band);border:1px solid var(--band)}
.src-abs{background:var(--panel-2);color:var(--ink-3);border:1px solid var(--line-2)}
.srcd{display:block;font-family:"JetBrains Mono",monospace;font-size:10px;
 color:var(--ink-3);margin-top:3px}
.whyc{font-size:12.5px;color:var(--ink-3);max-width:34ch}
.alp{background:var(--band-soft);border:1px solid var(--band);border-radius:2px;
 padding:11px 13px}
.alp.none{background:var(--panel-2);border-color:var(--line-2)}
.alp-h{font-family:"JetBrains Mono",monospace;font-size:9.5px;letter-spacing:.09em;
 color:var(--ink-3);margin-bottom:7px}
.alp-t{color:var(--band)}
.cot{font-size:13.5px;font-style:italic;color:var(--ink)}
.refs{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}
.ref{font-family:"JetBrains Mono",monospace;font-size:10px;background:var(--panel);
 border:1px solid var(--line-2);padding:2px 6px;border-radius:2px;color:var(--ink-2)}
.alp-u{font-size:12.5px;color:var(--ink-2);margin-top:7px}
.evid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
 gap:9px 14px;margin:12px 0 0}
.ev dt{font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.07em}
.ev dd{margin:2px 0 0;font-family:"JetBrains Mono",monospace;font-size:12.5px;
 font-weight:600}
.verd{border-top:1px solid var(--line);padding:13px 18px 16px;background:var(--panel-2)}
.verd p{font-size:14.5px;color:var(--ink);max-width:108ch}
.vl{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.08em;
 text-transform:uppercase;color:var(--accent);margin-right:8px}
footer{margin-top:50px;padding-top:22px;border-top:1px solid var(--line);
 color:var(--ink-3);font-size:13px;max-width:88ch}
@media (max-width:1120px){.body{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="wrap">
<header class="top">
 <p class="eyebrow">TanitAD · DataFlyWheel · 2026-08-23 · rebuilt pipeline</p>
 <h1>Label Provenance Report</h1>
 <p class="lede">The re-emitted label set, shown layer by layer — strategic goal and action,
  tactical goal and action — with the <b>source of every filled value</b> beside it. Frames are
  decoded from the clip mp4 whose filename IS the clip UUID, so no join can put the wrong
  pictures next to a label.</p>
 <div class="meta">
  <span class="mtag mono">801/801 labels re-emitted</span>
  <span class="mtag good">guard REFUSE 7.7% → 0.00%</span>
  <span class="mtag good">strategic band 94.5% full</span>
  <span class="mtag mono">33 join-free clips</span>
  <span class="mtag warn">NON-PARITY</span>
 </div>
</header>

<section>
 <div class="sechead"><p class="eyebrow">Read this first</p>
  <h2>Where every value comes from</h2>
  <p>Three sources fill the label stack, and they are never mixed. This is the answer to
   “what filled this field?” for every row in every card below.</p></div>
 <div class="grid3">
  <div class="box" style="border-left:3px solid var(--accent)">
   <h3><span class="src src-geo">EGO GEOMETRY</span></h3>
   <p>Derived from the provider's egomotion at 100 Hz, keyed by clip UUID. <b>Every
    kinematic token</b> — turns, stops, speeds, goal points — comes from here and only here.</p>
   <p>Instantaneous curvature κ = ω⁄v decides turn vs bend; the first sustained segment
    decides direction; stop repetition and recovery decide jam vs signal.</p>
  </div>
  <div class="box" style="border-left:3px solid var(--band)">
   <h3><span class="src src-vlm">GEOMETRY + VLM REASON</span></h3>
   <p>Alpamayo's chain-of-thought may <b>only refine the REASON of a stop that geometry has
    already found</b>. It can never create an event, and it never supervises a lateral token.</p>
   <p>Why so restricted: its lateral axis is <b>at chance</b> (31.2% vs 23.9% shuffled,
    p=0.335) and its CoT hallucinates — 3 correct, 2 wrong on visually checkable claims.</p>
  </div>
  <div class="box" style="border-left:3px solid var(--line-2)">
   <h3><span class="src src-abs">ABSTAIN</span></h3>
   <p>An explicit refusal with its reason, never a silent guess. Used when the horizon is too
    short to see the band, when the ego is near-stationary so a goal point would land on the
    car, or when the band constrains nothing nameable from ego alone.</p>
   <p>Abstains are <b>24 lateral</b> and <b>172 longitudinal</b> of 801 — visible, not hidden.</p>
  </div>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">Four fixes from your review of the first scene</p>
  <h2>Which layer owns a manoeuvre</h2>
  <p>Analysing <span class="mono">0e56dae2</span> together surfaced five defects — four in the label logic and one in this page. The deepest one is a question of <b>which layer owns a manoeuvre</b>: a turn that completes inside the 6 s plan is <i>tactical</i>, and the strategic layer should describe what lies beyond it.</p></div>
 <div class="tw"><table>
  <thead><tr><th>What you saw</th><th>Why it happened</th><th>Fix</th></tr></thead>
  <tbody>
  <tr><td><b>The turn was labelled STRATEGIC at all</b></td>
   <td>The strategic label was derived over <b>[t0, t0+30 s]</b> — a window that CONTAINS the
    tactical band — so a manoeuvre the plan already executes was promoted to the strategic
    layer. Measured on this clip the three bands disagree completely:
    <span class="mono">operative 0–2 s = STRAIGHT</span>,
    <span class="mono">tactical 2–6 s = JUNCTION_TURN_L +102.6°</span>,
    <span class="mono">strategic 8–30 s = follows the road</span>.</td>
   <td>The strategic label is now derived over the <b>strategic band only (8–30 s)</b>.
    This clip becomes <span class="mono">FOLLOW_MAIN_ROAD</span> + <span class="mono">HOLD_CORRIDOR</span>
    — “follow the road after turning” — and the turn belongs to the tactical layer.
    <b>83 clips</b> moved out of strategic-turn (271 → 188).</td></tr>
  <tr><td><b>Strategic action <span class="mono">HOLD_CORRIDOR</span> while the vehicle
    turns</b></td>
   <td>v6.0's six strategic actions have <b>no token for committing to a junction turn</b> —
    <span class="mono">PREPARE_EXIT</span> is a motorway exit,
    <span class="mono">PREPARE_LANE_CHANGE</span> is lane-relative. Every turn was forced onto
    a token that denies it.</td>
   <td>v6.1 appends <span class="mono">PREPARE_TURN_L/R</span>; the turn case is tested before
    the longitudinal fallbacks. <b>240 clips</b> now carry it.</td></tr>
  <tr><td><b>Tactical action <span class="mono">LANE_KEEP</span> inside an intersection</b></td>
   <td>Two causes at once. The window was <b>0–2 s</b> — inherited from the legacy label
    horizon — while this turn <b>begins at t+2.5 s and reaches +120°</b>. And v6.0's lateral
    actions are all lane-relative, so even seen, it could not be named.</td>
   <td>The window is now the <b>full 0–6 s plan rollout</b> (HIERARCHY §4b), and the emitter
    selects v6.1 with <span class="mono">TURN_L/TURN_R</span>. <b>123 clips</b> now read as
    turns instead of lane-keeps.</td></tr>
  <tr><td><b>Goals and actions inconsistent in the table</b></td>
   <td>A hardcoded <span class="mono">rowspan=2</span> labelled only 2 of TACTICAL's 4 rows,
    so the two action rows appeared to belong to nothing.</td>
   <td>The rowspan now counts the rows it actually spans.</td></tr>
  <tr><td><b>Your idea: turn/exit terms combined with intersection/roundabout</b></td>
   <td>A turn verb alone is a kinematic claim, and Alpamayo's kinematic axes are at chance.
    A turn verb <i>together with</i> a junction referent is a claim about road
    <b>topology</b> — which ego poses cannot see at all.</td>
   <td>Implemented as <b>corroboration, never an override</b>: it raises confidence and never
    flips a class. Where it fires, geometry independently agrees on <b>6 of 7</b>.</td></tr>
  </tbody></table></div>

 <div class="grid3" style="margin-top:16px">
  <div class="box" style="border-left:3px solid var(--flag)">
   <h3>⚠️ This changes model tensors</h3>
   <p>Both vocabularies <b>size live embedding tables</b>, and resumes are tensor-strict. So
    v6.1 is an <b>APPEND behind a version switch</b>, never an edit: indices 0–5 keep their
    meaning, existing labels and checkpoints stay valid, and a 6-wide head widens to 8 by
    <b>padding rather than retraining</b>.</p>
   <p>The default stays v6.0, so importing changes nothing. <b>The Master Mind and Training
    FlyWheel need to know before the next run</b> — a transfer note is filed with the work
    package.</p>
  </div>
  <div class="box"><h3>Strategic actions after the fix</h3>
   <p><span class="bign">240</span></p>
   <p><span class="mono">PREPARE_TURN_L</span> 126 · <span class="mono">PREPARE_TURN_R</span>
    114. <span class="mono">HOLD_CORRIDOR</span> drops 478 → 313 — it now means what it says.</p>
  </div>
  <div class="box"><h3>Tactical actions after the fix</h3>
   <p><span class="bign">123</span></p>
   <p><span class="mono">TURN_L</span> 73 · <span class="mono">TURN_R</span> 50, over the
    0–6 s plan. Previously every one of these read <span class="mono">LANE_KEEP</span>.</p>
   <p>⚠️ Representable, not scoreable: per-class metrics on these are refused below n=200.</p>
  </div>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">What changed</p>
  <h2>The rebuilt pipeline, measured</h2></div>
 <div class="grid3">
  <div class="box"><h3>Self-consistency</h3>
   <p><span class="bign good">0.00%</span></p>
   <p>The emitter's output now passes the guard that judges it: <b>REFUSE 61 (7.7%) → 0</b>,
    CLEAN 89.5% → <b>98.1%</b>. <span class="mono">G1-fallback-absorbs-turn</span> — the one
    systematic defect — is zero.</p>
  </div>
  <div class="box"><h3>Strategic horizon recovered</h3>
   <p><span class="bign good">94.5%</span></p>
   <p>Reading the provider's egomotion (20–140 s) instead of the 20 s episode cache,
    <b>757/801 clips carry the full 22 s strategic band</b>; median horizon 37.0 s. The
    “81% unobservable” blocker was an artefact of the cache.</p>
  </div>
  <div class="box"><h3>Labels changed</h3>
   <p><span class="bign">154</span></p>
   <p>Of 797 comparable: <b>75</b> FOLLOW_MAIN_ROAD that hid a real manoeuvre, <b>36</b> bends
    miscalled turns, and <b>all 14</b> NONE_ABSTAIN plus <b>all 80</b> action-abstains
    resolved against the real horizon.</p>
  </div>
 </div>
 <div class="tw" style="margin-top:16px"><table>
  <thead><tr><th>Defect found by emitting at corpus scale</th><th>Measured</th>
   <th>Fix</th></tr></thead>
  <tbody>
  <tr><td>the label↔frame join used a colliding 16-bit id</td>
   <td class="n bad">8/39 wrong (20.5%)</td><td>key everything on the clip UUID</td></tr>
  <tr><td><span class="mono">EVADE_IN_CORRIDOR</span> fired on lane-keeping jitter</td>
   <td class="n bad">79/132 below 1.0 m</td><td>magnitude floor</td></tr>
  <tr><td>…and none had the out-and-back signature of an evasion</td>
   <td class="n bad">0 of 40 returned</td><td>require return + net-yaw ≈ 0 ⇒ now emits <b>0</b></td></tr>
  <tr><td><span class="mono">ANCHOR_GOAL</span> on a stationary ego</td>
   <td class="n bad">4 behind the car</td><td>degenerate-goal abstain</td></tr>
  <tr><td>the emitter failed its OWN guard</td><td class="n bad">9 labels</td>
   <td>emitter imports the guard's constants</td></tr>
  </tbody></table></div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">The sample — 14 clips, every layer, every source</p>
  <h2>Label stacks on join-free frames</h2>
  <p>Eight frames per clip from t−4 s to t+12 s; blue is the anchor, purple frames and the
   purple track segment lie inside the strategic band. Each table row gives the token, its
   arguments, the source that filled it, and the derivation.</p></div>
 @@CARDS@@
</section>

<footer><p><strong>Provenance and limits.</strong> Poses: provider egomotion, 100 Hz, keyed by
 clip UUID — 801/801 coverage, no legacy-id lookup anywhere. Frames: the clip mp4 whose filename
 is the UUID, so the frames beside a label cannot belong to another clip. Labels:
 <span class="mono">stack/scripts/s2_geom_emit.py</span>, 801/801 emitted, 0 failures.
 Alpamayo covers 100% of aug120 and 9.3% of w120val; where absent, every value is geometry.
 ⚠️ This run is <b>NON-PARITY</b> — not <span class="mono">e438721ae894</span>/<span class="mono">f09e44db</span> —
 so nothing here is cross-arm comparable; the emitter reads the provider source directly, so a
 parity run is a re-invocation rather than a rewrite. Visual validation covers 8 clips of 801:
 zero label errors found, which is a coverage statement and not a clean bill of health. The
 composite stop+turn ordering (turn takes the strategic slot, stop moves to
 <span class="mono">g_tac</span>) is a documented design choice, not a measurement.</p></footer>
</div>
'''

if __name__ == "__main__":
    main()
