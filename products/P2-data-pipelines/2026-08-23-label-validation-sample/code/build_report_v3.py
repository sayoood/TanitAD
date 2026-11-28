"""Render the label-validation visual report v3 (self-contained HTML)."""
from __future__ import annotations

import html
import json
from pathlib import Path

V = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/label_validation_report.html")

SHOW = ["5b4eef8f", "4d389996", "5aef0388", "b0499b70", "1a293863", "a8a381bf",
        "00d05901", "c84534a9", "a2524c12", "e084c7c3", "12bb97af", "01b24287"]

JUDGE = {
 "5b4eef8f": "The clip that overturned my own fix. Peak yaw +69&deg; over a 12 s horizon reads as a gentle bend if you divide it by the whole arc &mdash; and I nearly weakened the guard on that basis. Instantaneous curvature says <b>R = 12.4 m</b>: a tight junction turn. FOLLOW_MAIN_ROAD is wrong here. Note Alpamayo says &ldquo;nudge left to pass the cyclist&rdquo; &mdash; the VLM is describing a different event, which is why its claims are tagged <span class='mono'>disputed</span> and never override geometry.",
 "4d389996": "The same miss with a second failure stacked on it: R = 9.2 m junction turn labelled corridor-follow, while the ego decelerates 3.95 &rarr; 0.43 m/s under a hold-corridor action. No Alpamayo row exists for this clip, so nothing external would have caught it.",
 "5aef0388": "An 89&deg; right turn at R = 12.2 m where the goal head abstained. Abstaining on ambiguous geometry is right; this geometry is not ambiguous. The action family stays confident through the same turn.",
 "b0499b70": "The clean longitudinal inversion. Straight road, ego launches from rest 0.17 &rarr; 13.27 m/s, and the label says prepare-to-stop. The manoeuvre analysis reads <span class='mono'>LAUNCH</span> / <span class='mono'>ALREADY_STOPPED</span> &mdash; the pipeline already had the signal to reject this.",
 "1a293863": "I previously called this borderline and let it pass. Instantaneous curvature settles it: <b>R = 8.0 m</b> is a tight junction turn, not a highway curve. My earlier arc-based estimate of 103 m was the artefact. Correctly refused now.",
 "a8a381bf": "A full stop from 9.69 m/s described as hold-corridor. Not an inversion &mdash; hold-corridor does not contradict stopping, it simply fails to describe the dominant event.",
 "00d05901": "The control case, and it now passes. Alpamayo states plainly &ldquo;turn right at the intersection since the traffic light is green&rdquo;. At R = 37.4 m this is a wide sweeping junction &mdash; the widest true turn in the sample &mdash; and the classifier calls it JUNCTION_TURN_R at MEDIUM confidence. An earlier arc-based version called it a road bend, i.e. the VLM caught a real classifier error.",
 "c84534a9": "An 87&deg; right turn at R = 9.8 m with the textbook signature you described: decelerate into it (7.5 &rarr; 4.2 m/s), turn at the apex, accelerate out to 11.2 m/s. Alpamayo again says &ldquo;nudge left to pass the cyclist&rdquo; and is again wrong &mdash; two independent clips where the CoT contradicts unambiguous geometry.",
 "a2524c12": "Exactly the composite you asked us to detect: <span class='mono'>YIELD</span> stop plus a junction turn. Alpamayo supplies the reason geometry cannot &mdash; &ldquo;slow down due to the stop sign ahead&rdquo; &mdash; so the resolved stop is sign-controlled, not a queue.",
 "e084c7c3": "The reference controlled stop. One monotonic deceleration 6.40 &rarr; 0 with no rotation, and the CoT names the cause: &ldquo;stop for the red traffic light&rdquo;. Kinematics gives the shape, semantics gives the reason, and they agree.",
 "12bb97af": "The queue case, and the one that separates jam from signal. The ego stops but the CoT says &ldquo;wait for a gap&hellip; the lead vehicle ahead is slowing&rdquo; &mdash; a lead-vehicle referent, so the resolved type is <span class='mono'>QUEUE</span> even where the kinematic shape alone looked controlled. This is the strategic distinction you asked for.",
 "01b24287": "Kept as the worked example of my own retracted error. Over 4 s the ego barely rotates and I scored the TURN_LEFT as wrong; the turn is real, 100&deg; beginning 7.3 s out, visible in the +8 s frame. The label was right and my window was too short.",
}


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
 aria-label="Bird's-eye view in the ego frame: dashed past track, solid future track, and the segment inside the strategic band">
<g class="bev-grid">{grid}</g>
<path class="bev-past" d="{path(px, py)}"/>
<path class="bev-fut" d="{path(fx, fy, 0, bi + 1)}"/>
<path class="bev-band" d="{path(fx, fy, bi)}"/>
<circle class="bev-ego" cx="{ox:.1f}" cy="{oy:.1f}" r="5"/>
<text class="bev-sc" x="8" y="{h-8}">grid {round(span/4)} m</text></svg>'''


ARG_NAMES = ["dist_m", "lat_off", "lane_idx", "v_target", "deadline_m",
             "text_id", "arc_m", "hold_s"]


def args_html(block):
    a, m = block.get("args"), block.get("arg_mask")
    if not a or not m:
        return ""
    live = [f'<span class="arg"><i>{ARG_NAMES[i]}</i>{a[i]:g}</span>'
            for i in range(min(len(a), len(m), 8)) if m[i]]
    return ("".join(live) if live
            else '<span class="arg none">no args set</span>')


def card(r):
    cid = r["clip_id"][:8]
    man, g, a, tac = r["manoeuvre"], r["g_str"], r["a_str"], r["tactical"]
    gd, sem = r["guard"], r["semantics"]

    figs = ""
    for m in r["frame_meta"]:
        off = m["offset_s"]
        cls = "key" if off == 0 else ("band" if m["strategic"] else "")
        cap = ("KEY FRAME  t0" if off == 0 else f"t{off:+.0f}s")
        if m["strategic"]:
            cap += " · strategic"
        alt = ("Forward camera at the labelled anchor" if off == 0 else
               f"Forward camera {abs(off):.0f}s "
               f"{'after' if off > 0 else 'before'} the anchor")
        figs += (f'<figure class="{cls}"><img loading="lazy" '
                 f'src="data:image/jpeg;base64,{r["frames"][m["key"]]}" '
                 f'alt="{alt}"><figcaption>{cap}</figcaption></figure>')

    chips = "".join(
        f'<span class="chip {"refuse" if f["severity"]=="REFUSE" else "flag"}">'
        f'{f["rule"]}</span>' for f in gd["findings"]) or \
        '<span class="chip ok">no findings</span>'

    R = man["turn_radius_m"]
    Rs = "&infin;" if R == float("inf") or R > 9e5 else f'{R:.1f} m'
    ev = [("lateral", man["lateral_class"]),
          ("turn radius", Rs),
          ("peak yaw", f'{man["peak_yaw_deg"]:+.1f}&deg;'),
          ("onset", f't+{man["yaw_onset_s"]}s' if man["yaw_onset_s"] else "&mdash;"),
          ("longitudinal", man["longitudinal_class"]),
          ("stop type", man["stop_type"]),
          ("stop episodes", f'{man["n_stop_episodes"]} · {man["longest_stop_s"]}s'),
          ("decel cycles", man["n_decel_events"]),
          ("speed", f'{man["v_at_key"]:.2f}&rarr;{man["v_end"]:.2f} m/s'),
          ("slowed for turn", {True: "yes", False: "no", None: "n/a"}[man["slowed_for_turn"]]),
          ("confidence", man["confidence"]),
          ("horizon", f'{man["horizon_s"]}s')]
    evh = "".join(f'<div class="ev"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in ev)

    if r["alpamayo"]:
        al = r["alpamayo"]
        refs = "".join(f'<span class="ref">{x}</span>'
                       for x in (sem["referents"] if sem else []))
        toks = "".join(
            f'<span class="tokp">{t["token"]}'
            + (f'<i>{", ".join(f"{k}={v}" for k, v in t["args"].items() if v)}</i>'
               if any(t["args"].values()) else "")
            + '</span>' for t in r["tokens"])
        alph = f'''<div class="blk alp"><div class="blk-h">ALPAMAYO&#8209;SUPER
     <span class="prov">VLM · disputed until grounded</span></div>
     <div class="axes"><span class="ax">{html.escape(al["lane"])}</span>
      <span class="ax">{html.escape(al["lateral"])}</span>
      <span class="ax">{html.escape(al["longitudinal"])}</span></div>
     <p class="cot">&ldquo;{html.escape(al["cot"])}&rdquo;</p>
     {f'<div class="refs">{refs}</div>' if refs else ''}
     {f'<div class="toks"><span class="tl">tokens unblocked</span>{toks}</div>' if toks else ''}
    </div>'''
    else:
        alph = ('<div class="blk alp none"><div class="blk-h">ALPAMAYO&#8209;SUPER</div>'
                '<p class="miss">No inference for this clip. Alpamayo covers '
                '<b>100% of aug120</b> but only <b>9.3% of w120val</b> &mdash; '
                'this clip is in the val split.</p></div>')

    sr = r["stop_resolved"]
    return f'''<article class="card{' is-flag' if gd['findings'] else ''}">
 <header class="card-h">
  <div class="card-id"><span class="mono big">{cid}</span>
   <span class="split">{r["split"]}</span>
   <span class="split">{man["lateral_class"]}</span></div>
  <div class="chips">{chips}</div>
 </header>
 <div class="strip">{figs}</div>
 <div class="body">
  <div class="left">{bev(r["traj"])}
   <div class="bevkey"><span class="k-past"></span>past
    <span class="k-fut"></span>future<span class="k-band"></span>strategic band</div>
  </div>
  <div class="right">
   <div class="blk">
    <div class="blk-h">STRATEGIC <span class="prov">{g.get("provenance") or ""}</span></div>
    <div class="row"><span class="fam">goal</span>
     <span class="tok">{html.escape(g["token"])}</span>
     <span class="argw">{args_html(g)}</span>
     <span class="conf">conf {g.get("confidence")}</span></div>
    <div class="row"><span class="fam">action</span>
     <span class="tok">{"<em>ABSTAIN</em>" if a["abstain"] else html.escape(a["token"] or "—")}</span>
     <span class="argw">{args_html(a)}</span>
     <span class="conf">conf {a.get("confidence") or "&mdash;"}</span></div>
   </div>
   <div class="blk">
    <div class="blk-h">TACTICAL <span class="prov">{tac["horizon_s"]}s horizon · factored</span></div>
    <div class="row"><span class="fam">lat</span><span class="tok t2">{tac["lat"]}</span>
     <span class="fam">lon</span><span class="tok t2">{tac["lon"]}</span></div>
   </div>
   {alph}
   <div class="blk">
    <div class="blk-h">RESOLVED STOP</div>
    <div class="row"><span class="tok t3">{sr["type"]}</span>
     <span class="conf">{html.escape(sr["provenance"])}</span></div>
   </div>
  </div>
 </div>
 <dl class="evid">{evh}</dl>
 <div class="verd"><p><span class="vl">my read</span>{JUDGE.get(cid, "")}</p></div>
</article>'''


def main() -> None:
    rows = {r["clip_id"][:8]: r for r in
            json.loads((V / "sample_v3.json").read_text(encoding="utf-8"))}
    cards = "\n".join(card(rows[c]) for c in SHOW if c in rows)
    doc = TEMPLATE.replace("@@CARDS@@", cards)
    assert "@@" not in doc
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), "
          f"{sum(1 for c in SHOW if c in rows)} cards")


TEMPLATE = r'''<title>Ego Label Forensics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
 --ground:#F5F7FA;--panel:#FFF;--panel-2:#EDF1F6;--panel-3:#E3EAF2;
 --ink:#0E1620;--ink-2:#43525F;--ink-3:#71828F;
 --line:#D7E0E9;--line-2:#BFCCD9;
 --accent:#0057D9;--accent-soft:#E3EDFC;
 --past:#8B9DAF;--band:#7A3DBD;--band-soft:#F1E8FB;
 --ok:#0F7A50;--ok-bg:#E2F3EA;--flag:#B4400C;--flag-bg:#FBE8DF;
 --refuse:#9B1C1C;--refuse-bg:#FBE3E3;
 --shadow:0 1px 2px rgba(14,22,32,.06),0 10px 28px -14px rgba(14,22,32,.2);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0A0F14;--panel:#121A22;--panel-2:#19232D;--panel-3:#213040;
 --ink:#E6EDF4;--ink-2:#A4B4C2;--ink-3:#7A8B99;
 --line:#222E38;--line-2:#30404E;
 --accent:#5C9BFF;--accent-soft:#14233A;
 --past:#69798A;--band:#B489EE;--band-soft:#1D1530;
 --ok:#4FC08A;--ok-bg:#11281E;--flag:#FF8A5B;--flag-bg:#2D1710;
 --refuse:#FF6B6B;--refuse-bg:#331414;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -16px rgba(0,0,0,.75);
}}
:root[data-theme="dark"]{
 --ground:#0A0F14;--panel:#121A22;--panel-2:#19232D;--panel-3:#213040;
 --ink:#E6EDF4;--ink-2:#A4B4C2;--ink-3:#7A8B99;
 --line:#222E38;--line-2:#30404E;
 --accent:#5C9BFF;--accent-soft:#14233A;
 --past:#69798A;--band:#B489EE;--band-soft:#1D1530;
 --ok:#4FC08A;--ok-bg:#11281E;--flag:#FF8A5B;--flag-bg:#2D1710;
 --refuse:#FF6B6B;--refuse-bg:#331414;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -16px rgba(0,0,0,.75);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"Source Sans 3",ui-sans-serif,system-ui,sans-serif;font-size:16px;
 line-height:1.6;-webkit-font-smoothing:antialiased}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;
 font-variant-numeric:tabular-nums}
.wrap{max-width:1500px;margin:0 auto;padding:48px 22px 96px}
h1,h2,h3{font-family:Archivo,ui-sans-serif,system-ui,sans-serif;text-wrap:balance;
 margin:0;line-height:1.14;letter-spacing:-.012em}
h1{font-size:clamp(30px,4.2vw,46px);font-weight:700}
h2{font-size:25px;font-weight:600}
p{margin:0}
.eyebrow{font-family:Archivo,sans-serif;font-size:11.5px;font-weight:600;
 letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
.lede{max-width:70ch;color:var(--ink-2);margin-top:14px;font-size:17px}
header.top{border-bottom:1px solid var(--line);padding-bottom:30px;
 display:flex;flex-direction:column;gap:6px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.mtag{font-size:12.5px;padding:4px 10px;border:1px solid var(--line-2);
 border-radius:2px;color:var(--ink-2);background:var(--panel)}
.mtag.warn{border-color:var(--flag);color:var(--flag);background:var(--flag-bg)}
.mtag.good{border-color:var(--ok);color:var(--ok);background:var(--ok-bg)}
section{margin-top:52px}
.sechead{display:flex;flex-direction:column;gap:6px;margin-bottom:20px}
.sechead p{color:var(--ink-2);max-width:78ch}
.alert{background:var(--band-soft);border:1px solid var(--band);border-left-width:3px;
 border-radius:3px;padding:22px 24px}
.alert h3{font-size:18px;margin-bottom:10px}
.alert p{color:var(--ink-2);max-width:80ch}.alert p+p{margin-top:9px}
.alert b{color:var(--ink)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 padding:20px 22px;box-shadow:var(--shadow)}
.box h3{font-size:12px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--ink-3);margin-bottom:12px;font-weight:600}
.box p{color:var(--ink-2);font-size:14.5px}
.box p+p{margin-top:8px}
.big-n{font-family:Archivo,sans-serif;font-size:40px;font-weight:700;
 font-variant-numeric:tabular-nums;line-height:1;letter-spacing:-.02em}
.big-n.bad{color:var(--flag)}.big-n.good{color:var(--ok)}
table{width:100%;border-collapse:collapse;font-size:14px}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
 border-radius:3px;box-shadow:var(--shadow)}
th,td{text-align:left;padding:9px 13px;border-bottom:1px solid var(--line)}
th{font-family:Archivo,sans-serif;font-size:10.5px;text-transform:uppercase;
 letter-spacing:.08em;color:var(--ink-3);font-weight:600}
tr:last-child td{border-bottom:none}
td.n{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
.bad{color:var(--flag);font-weight:600}.good{color:var(--ok);font-weight:600}

.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 margin-bottom:22px;overflow:hidden;box-shadow:var(--shadow)}
.card.is-flag{border-left:3px solid var(--flag)}
.card-h{display:flex;justify-content:space-between;align-items:center;gap:12px;
 padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel-2);
 flex-wrap:wrap}
.card-id{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.big{font-size:16px;font-weight:600}
.split{font-size:10.5px;color:var(--ink-3);text-transform:uppercase;
 letter-spacing:.08em;border:1px solid var(--line-2);padding:2px 7px;border-radius:2px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-family:"JetBrains Mono",monospace;font-size:10px;font-weight:600;
 letter-spacing:.04em;padding:3px 8px;border-radius:2px}
.chip.ok{background:var(--ok-bg);color:var(--ok)}
.chip.flag{background:var(--flag-bg);color:var(--flag)}
.chip.refuse{background:var(--refuse-bg);color:var(--refuse)}

/* FRAMES — the PI asked for bigger; fixed 210px cells, strip scrolls itself */
.strip{display:flex;gap:10px;padding:16px 18px;overflow-x:auto;
 scrollbar-width:thin;background:var(--panel-3)}
.strip figure{margin:0;flex:0 0 210px;display:flex;flex-direction:column;gap:6px}
.strip img{width:210px;height:210px;object-fit:cover;border-radius:2px;
 border:1px solid var(--line-2);display:block}
.strip figcaption{font-family:"JetBrains Mono",monospace;font-size:10.5px;
 color:var(--ink-3)}
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
 display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-top:6px}
.bevkey span{width:13px;height:2px;display:inline-block}
.k-past{background:var(--past)}.k-fut{background:var(--accent)}
.k-band{background:var(--band)}
.right{display:flex;flex-direction:column;gap:10px}
.blk{background:var(--panel-2);border:1px solid var(--line);border-radius:2px;
 padding:11px 13px}
.blk.alp{background:var(--accent-soft)}
.blk.alp.none{background:var(--panel-2)}
.blk-h{font-family:"JetBrains Mono",monospace;font-size:9.5px;letter-spacing:.1em;
 color:var(--ink-3);margin-bottom:8px;display:flex;justify-content:space-between;
 gap:10px;flex-wrap:wrap}
.prov{color:var(--ink-3);opacity:.85;letter-spacing:.02em}
.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.row+.row{margin-top:7px}
.fam{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--ink-3);
 letter-spacing:.06em;min-width:38px}
.tok{font-family:"JetBrains Mono",monospace;font-size:13.5px;font-weight:600;
 background:var(--panel);border:1px solid var(--line-2);padding:3px 9px;border-radius:2px}
.tok.t2{color:var(--accent);border-color:var(--accent)}
.tok.t3{color:var(--band);border-color:var(--band)}
.tok em{font-style:normal;color:var(--ink-3)}
.argw{display:flex;gap:5px;flex-wrap:wrap}
.arg{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--ink-2);
 background:var(--panel);border:1px dashed var(--line-2);padding:1px 6px;border-radius:2px}
.arg i{font-style:normal;color:var(--ink-3);margin-right:4px}
.arg.none{border-style:dotted;color:var(--ink-3)}
.conf{font-size:11px;color:var(--ink-3)}
.axes{display:flex;gap:6px;flex-wrap:wrap}
.ax{font-size:12px;background:var(--panel);border:1px solid var(--line-2);
 padding:2px 8px;border-radius:2px;color:var(--ink-2)}
.cot{font-size:13.5px;color:var(--ink);font-style:italic;margin-top:7px}
.refs,.toks{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px;align-items:center}
.ref{font-family:"JetBrains Mono",monospace;font-size:10px;background:var(--panel);
 border:1px solid var(--line-2);padding:2px 6px;border-radius:2px;color:var(--ink-2)}
.tl{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--ink-3);
 letter-spacing:.06em}
.tokp{font-family:"JetBrains Mono",monospace;font-size:10.5px;font-weight:600;
 background:var(--band-soft);color:var(--band);border:1px solid var(--band);
 padding:2px 7px;border-radius:2px}
.tokp i{font-style:normal;font-weight:400;opacity:.8;margin-left:5px}
.miss{font-size:13px;color:var(--ink-3)}
.evid{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
 gap:10px 14px;margin:0;padding:12px 18px 14px;border-top:1px solid var(--line)}
.ev dt{font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.07em}
.ev dd{margin:2px 0 0;font-family:"JetBrains Mono",monospace;font-size:12.5px;
 font-weight:600}
.verd{border-top:1px solid var(--line);padding:13px 18px 16px;background:var(--panel-2)}
.verd p{font-size:14.5px;color:var(--ink);max-width:100ch}
.vl{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.08em;
 text-transform:uppercase;color:var(--accent);margin-right:8px}
.rec{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 padding:26px;box-shadow:var(--shadow)}
.rec ol{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:13px}
.rec li{color:var(--ink-2);max-width:86ch}.rec li b{color:var(--ink)}
footer{margin-top:52px;padding-top:22px;border-top:1px solid var(--line);
 color:var(--ink-3);font-size:13px;max-width:88ch}
@media (max-width:1080px){.body{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="wrap">
<header class="top">
 <p class="eyebrow">TanitAD · DataFlyWheel · 2026-08-23 · iteration 2</p>
 <h1>Ego Label Forensics</h1>
 <p class="lede">Every layer the pipeline produces, on clips where labels join to real
  frames and real ego poses: strategic goal and action with their arguments, the factored
  tactical pair, the ego manoeuvre analysis, and Alpamayo-Super's own reading — each
  scored against geometry computed independently from the poses.</p>
 <div class="meta">
  <span class="mtag mono">n = 39 clips</span>
  <span class="mtag mono">8 frames/clip · t−4s → t+12s</span>
  <span class="mtag good">Alpamayo 100% of aug120</span>
  <span class="mtag warn">9.3% of w120val</span>
  <span class="mtag warn">strategic band 18% observable</span>
  <span class="mtag warn">NON-PARITY cache</span>
 </div>
</header>

<section>
 <div class="sechead"><p class="eyebrow">Answering the Alpamayo question</p>
  <h2>The augmented data is used everywhere it exists</h2></div>
 <div class="grid2">
  <div class="box"><h3>Coverage by split</h3>
   <p><span class="big-n good">201/201</span></p>
   <p><b>aug120 — 100%.</b> Every clip in the augmentation split carries an Alpamayo
    inference row.</p>
   <p style="margin-top:14px"><span class="big-n bad">56/600</span></p>
   <p><b>w120val — 9.3%.</b> The validation split was largely never augmented.</p>
  </div>
  <div class="box"><h3>Why only 16 of 39 clips showed it</h3>
   <p>The augmentation covers <b>257 of the 801 labelled clips (32.1%)</b>, and all 544
    uncovered clips are in <b>w120val</b>. The previous sample happened to be 27/39
    w120val, because that is what sits in the local episode cache — so the gap was in the
    <i>sample</i>, not in how the directive is applied.</p>
   <p>Of the 12 aug120 clips reachable locally, <b>12/12 carry Alpamayo</b>. The ceiling
    here is the episode cache, not the augmentation.</p>
   <p>⇒ Two real work items: <b>extend the augmentation to w120val</b> so validation
    clips can be checked the same way, and cache more aug120 episodes locally.</p>
  </div>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">The idea you proposed, measured</p>
  <h2>Term search on the CoT unblocks tokens we could not emit at all</h2>
  <p>HIERARCHY_VOCABULARY lists six tactical tokens as blocked because they need non-ego
   inputs the corpus does not ship. The chain-of-thought names exactly those referents.
   Measured over all 4,729 augmented clips:</p></div>
 <div class="tw"><table>
  <thead><tr><th>Token (previously unemittable)</th><th>Clips</th><th>Share</th>
   <th>What in the CoT supplies it</th></tr></thead>
  <tbody>
  <tr><td class="mono">GAP_TARGET</td><td class="n">1042</td><td class="n">22.0%</td>
   <td>lead-vehicle (985) and merge-gap (68) references</td></tr>
  <tr><td class="mono">STOP_POINT</td><td class="n">859</td><td class="n">18.2%</td>
   <td>explicit stop verbs plus a light/sign reason</td></tr>
  <tr><td class="mono">EVADE_IN_CORRIDOR</td><td class="n">792</td><td class="n">16.7%</td>
   <td>“nudge left” 608 + “nudge right” 184, with the obstacle class</td></tr>
  <tr><td class="mono">TRAFFIC_LIGHT_REACT</td><td class="n">629</td><td class="n">13.3%</td>
   <td>light state named directly — red 175, yellow 26, green 428</td></tr>
  <tr><td class="mono">YIELD_AT</td><td class="n">395</td><td class="n">8.4%</td>
   <td>“yield” verbs (156) and yield-sign references</td></tr>
  </tbody></table></div>
 <div class="grid2" style="margin-top:16px">
  <div class="box"><h3>Referent coverage</h3>
   <p><span class="big-n">83.0%</span></p>
   <p>3,926 of 4,729 CoTs carry at least one recognised referent. The remaining 39% that
    yield no <i>token</i> are logged verbatim, never dropped — vocabulary completeness is
    measured as coverage of the phrase distribution.</p>
   <p>The unmapped set is itself informative: <b>“adapt speed for the right/left curve
    ahead” appears 366 times</b>, which is an independent witness for road curvature.</p>
  </div>
  <div class="box"><h3>⚠️ What these tokens are not</h3>
   <p>Every proposed token carries <span class="mono">provenance="vlm-cot"</span>, its
    verbatim evidence sentence, and <span class="mono">disputed=true</span>. They have no
    image-space grounding — no box, no frame index — so per the fusion gate they may
    supervise a goal or interpretation head but must <b>not</b> be promoted to trusted
    perception without a grounding pass.</p>
   <p>They are also label-side only: never an inference-time input, and never inside a goal
    signal that must stay disjoint from the situation classifier.</p>
  </div>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">Rebuilt ego logic</p>
  <h2>Curvature and the speed profile, not a yaw threshold</h2></div>
 <div class="alert">
  <h3>The estimator error this fixes — it fooled me for a full pass</h3>
  <p>Turn radius measured as arc ÷ Δyaw over the horizon divides a turn's heading change by
   an arc that <b>includes the straight road after the turn</b>. On clip <span class="mono">5b4eef8f</span>
   that read <b>83.7 m</b> — a gentle bend — when the true apex radius is <b>12.4 m</b>, a
   tight junction turn. I had begun weakening a correct guard on the strength of that number.</p>
  <p>The classifier now uses <b>instantaneous curvature, κ = ω ⁄ v</b>, minimised over the
   manoeuvre. Thresholds are calibrated on the real 39-clip distribution rather than chosen:
   sorted by R<sub>min</sub> the sample splits with an <b>empty margin between 37.4 m and
   77.9 m</b> — every clip below has |peak yaw| ≥ 38°, every clip above ≤ 13°. The gates sit
   inside that gap.</p>
  <p>Independent check: on <span class="mono">00d05901</span>, where Alpamayo states “turn right at the
   intersection”, the arc-based version said <i>road bend</i> and the curvature version says
   <b>JUNCTION_TURN_R</b>. The VLM caught a real classifier error.</p>
 </div>
 <div class="grid2" style="margin-top:16px">
  <div class="box"><h3>Stop type — jam vs signal</h3>
   <p>You asked for prepare-to-stop to be separable from traffic jams. Kinematically the two
    are <b>identical for the first seconds</b>, so depth of the stop cannot decide it. Two
    things can: <b>repetition</b> (a jam stops repeatedly) and <b>recovery</b> (a signal
    releases to cruise).</p>
   <p>Classes: <span class="mono">CONTROLLED</span> · <span class="mono">QUEUE</span> ·
    <span class="mono">YIELD</span> · <span class="mono">ALREADY_STOPPED</span> ·
    <span class="mono">NONE</span>. Where Alpamayo names a reason, semantics resolves what
    kinematics cannot — and a conflict is <b>surfaced, not silently resolved</b>.</p>
   <p>On this sample: 8 CONTROLLED · 5 ALREADY_STOPPED · 2 YIELD · 1 QUEUE · 23 none.</p>
  </div>
  <div class="box"><h3>A bug my own tests caught</h3>
   <p>The deceleration-cycle counter re-armed <i>during</i> a descent, so one smooth
    10 → 0 m/s stop counted as <b>six</b> decelerations. That would have made every ordinary
    stop look like stop-and-go traffic and destroyed the QUEUE discriminator this function
    exists to provide.</p>
   <p>It only surfaced because the test asserted the count, not merely that the code ran.
    After the fix, QUEUE on this sample drops from 3 to 1 — the earlier three were mostly
    that bug.</p>
  </div>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">The sample — 12 clips, every layer</p>
  <h2>What the pipeline says, and what the ego actually did</h2>
  <p>Eight frames per clip from t−4 s to t+12 s. Blue is the anchor; purple frames and the
   purple track segment fall inside the strategic band. Goals and actions are shown with
   their arguments. All flagged clips are included, plus reference cases.</p></div>
 @@CARDS@@
</section>

<section>
 <div class="sechead"><p class="eyebrow">Proposals</p>
  <h2>Solutions for each problem found</h2></div>
 <div class="rec"><ol>
  <li><b>Adopt curvature-based manoeuvre detection as the labelling primitive.</b> Shipped
   as <span class="mono">ego_manoeuvre.py</span> with 36 tests, including a fixture built so
   the old and new estimators disagree — otherwise nothing holds the correct one in place.</li>
  <li><b>Split kinematics from semantics, permanently.</b> Alpamayo's longitudinal axis
   agrees with ego dv on only <b>56%</b> with 25% sign contradictions, and I tested and
   <b>refuted</b> the obvious excuse — it is not a temporal-alignment artefact; our anchor is
   already the best of six tested. Geometry decides <i>what</i>, the CoT decides <i>why</i>,
   and neither overrides the other.</li>
  <li><b>Emit the five unblocked tactical tokens as disputed labels</b> and grade them
   against a grounding pass before any promotion. This is the largest single gain available:
   tokens that were previously unemittable now have a source on 61% of augmented clips.</li>
  <li><b>Extend the Alpamayo augmentation to w120val.</b> At 9.3% coverage the validation
   split cannot be checked the way the train split can, which will block evaluation of any
   head these labels supervise.</li>
  <li><b>Resolve the strategic horizon.</b> Still the largest open issue: 81% of the
   8–30 s band is past the end of a 19.9 s clip. Either narrow the definition to ~0–12 s or
   source longer sequences.</li>
  <li><b>Then re-run at n ≈ 300 on the parity corpus.</b> At n=39 the per-class counts
   cannot set a threshold — and the curvature gate deserves calibration on a sample that
   actually populates the 40–60 m band, which this one leaves empty.</li>
 </ol></div>
</section>

<footer><p><strong>How to read these numbers.</strong> Frames and poses come from the local
 cache <span class="mono">physicalai-train-14231cd29c74</span>, which is <b>not</b> the parity
 corpus <span class="mono">e438721ae894</span>/<span class="mono">f09e44db</span> — nothing here
 is cross-arm comparable. The sample is every clip both labelled and locally cached: a
 convenience sample whose class frequencies say nothing about the corpus. The label↔frame join
 runs through <span class="mono">episode_id_legacy</span>, which collides by design; 3 label-side
 and 3 cache-side colliding ids were refused rather than resolved. Camera frames are the
 256×256 model-input crops upscaled for display, not the full field of view. Tactical labels are
 <span class="mono">window_factored_labels_v2</span> at 2.0 s. Alpamayo rows are the banked
 per-clip taxonomy. The geometric analysis is a second opinion from ego poses alone and is not
 ground truth; the CoT is generative-model output and can be wrong — two clips here say “nudge
 left to pass the cyclist” through unambiguous 69° and 87° junction turns.</p></footer>
</div>
'''

if __name__ == "__main__":
    main()
