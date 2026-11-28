"""Render the label-validation visual report v2."""
from __future__ import annotations

import html
import json
from pathlib import Path

V = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/label_validation_report.html")

SHOW = ["5b4eef8f", "4d389996", "5aef0388", "90006660", "b0499b70",
        "a8a381bf", "01b24287", "2cf5d4c8", "a5acfb2a", "01bee851"]
FLAG = {"5b4eef8f", "4d389996", "5aef0388", "90006660", "b0499b70", "a8a381bf"}

JUDGE = {
 "5b4eef8f": "The clearest defect here. A 69&deg; left turn begins 1.7 s after the anchor and the strategic label says the ego is following the main road. Our own tactical head disagrees with our own strategic head on this clip &mdash; tactical reads <span class=\"mono\">turn_left</span>. When two families we derive from the same poses contradict each other, that is free supervision for a guard.",
 "4d389996": "The same miss, twice over. A 72&deg; turn from 1.0 s out labelled corridor-follow, while the ego decelerates 3.95 &rarr; 0.43 m/s under a hold-corridor action. Alpamayo is absent on this clip, so nothing external would have caught it either.",
 "5aef0388": "An 89&deg; right turn where the strategic goal abstained. Abstaining on ambiguous geometry is right; this geometry is not ambiguous. Note the action family stays confident (hold corridor) straight through the turn.",
 "90006660": "The one case I cannot settle from the trace. Net yaw over 4 s is &minus;35&deg; (right), but the largest excursion on the horizon is +56&deg; (left). Either a right turn followed by a left, or a curve then a junction. The frames are the only way to call it &mdash; which is exactly why this report shows them.",
 "b0499b70": "An inverted longitudinal label. The ego accelerates from rest, 0.17 &rarr; 13.27 m/s, and the action label says prepare-to-stop. Our tactical LON reads <span class=\"mono\">accelerate</span> on the same window, so again the two families already disagree &mdash; the information to catch this is in the pipeline, unused.",
 "a8a381bf": "A full stop from 9.69 m/s labelled hold-corridor. Milder than an inversion: hold-corridor does not contradict stopping, it just fails to describe it. Alpamayo reads Gentle Deceleration here, agreeing with the geometry and not with our label.",
 "01b24287": "The case my first pass scored WRONG, kept in deliberately. Over 4 s the ego barely rotates and I called the TURN_LEFT bad. The turn is real &mdash; 100&deg;, starting 7.3 s out, visible in the +8 s frame. The label was right; my test was too short.",
 "2cf5d4c8": "A wide sustained rotation &mdash; roundabout or U-turn &mdash; where the initial commitment is right-hand (&minus;73.9&deg; at 4 s) and the full excursion reaches 168&deg;. Labelled TURN_RIGHT, which is defensible. Worth flagging that the vocabulary has no U_TURN or ROUNDABOUT token, so every wide rotation is absorbed into a turn class.",
 "a5acfb2a": "Clean on every family and the easiest class in the set: 11.86 m/s decaying to a full stop with almost no rotation. STOP_AT / PREPARE_STOP, tactical <span class=\"mono\">brake_stop</span>. All seven STOP_AT clips look like this.",
 "01bee851": "A good one to see the layers line up. A 62&deg; left turn from 2.4 s out with the speed falling 5.80 &rarr; 1.60 m/s: strategic TURN_LEFT / REDUCE_TO, tactical <span class=\"mono\">turn_left</span> + <span class=\"mono\">brake_stop</span>, and Alpamayo independently reading Gentle Deceleration. Three sources, one story.",
}


def bev(t, w=300, h=200):
    px, py, fx, fy = (t["past_x"], t["past_y"], t["fut_x"], t["fut_y"])
    xs, ys = px + fx, py + fy
    if not xs:
        return ""
    sx, sy = [-q for q in ys], [-q for q in xs]
    span = max(max(sx) - min(sx), max(sy) - min(sy), 12.0) * 1.15
    cx, cy = (min(sx) + max(sx)) / 2, (min(sy) + max(sy)) / 2

    def P(a, b):
        return ((a - cx) / span * w + w / 2, (b - cy) / span * h + h / 2)

    def path(ax, ay, i0=0, i1=None):
        seg = list(zip(ax, ay))[i0:i1]
        pts = [P(-b, -a) for a, b in seg]
        return " ".join(f"{'M' if i == 0 else 'L'}{q:.1f},{r:.1f}"
                        for i, (q, r) in enumerate(pts))
    bi = t.get("band_start_i", 80)
    ox, oy = P(0, 0)
    grid = "".join(f'<line x1="0" y1="{g*h/4:.0f}" x2="{w}" y2="{g*h/4:.0f}"/>'
                   f'<line x1="{g*w/4:.0f}" y1="0" x2="{g*w/4:.0f}" y2="{h}"/>'
                   for g in range(1, 4))
    return f'''<svg class="bev" viewBox="0 0 {w} {h}" role="img"
 aria-label="Bird's-eye view: dashed past track, solid future track, and the portion falling inside the strategic band">
<g class="bev-grid">{grid}</g>
<path class="bev-past" d="{path(px, py)}"/>
<path class="bev-fut" d="{path(fx, fy, 0, bi + 1)}"/>
<path class="bev-band" d="{path(fx, fy, bi)}"/>
<circle class="bev-ego" cx="{ox:.1f}" cy="{oy:.1f}" r="4"/>
<text class="bev-sc" x="7" y="{h-7}">grid {round(span/3)} m</text></svg>'''


def card(r):
    cid, g, a = r["clip_id"][:8], r["g_str"], r["a_str"]
    tv, geo, hz = r["tactical"]["v2"], r["geometry"], r["horizon"]
    flagged = cid in FLAG
    fm = {m["key"]: m for m in r["frame_meta"]}

    figs = ""
    for m in r["frame_meta"]:
        k, off = m["key"], m["offset_s"]
        cls = ("key" if off == 0 else
               ("band" if m["in_strategic_band"] else ""))
        cap = ("KEY FRAME" if off == 0 else
               f"t{off:+.0f}s" + (" &middot; strat" if m["in_strategic_band"]
                                  else ""))
        alt = ("Forward camera at the labelled anchor frame" if off == 0 else
               f"Forward camera {abs(off):.0f} seconds "
               f"{'after' if off > 0 else 'before'} the anchor")
        figs += (f'<figure class="{cls}"><img src="data:image/jpeg;base64,'
                 f'{r["frames"][k]}" alt="{alt}">'
                 f'<figcaption>{cap}</figcaption></figure>')

    alp = r["alpamayo"]
    if alp:
        alph = (f'<div class="al"><span class="src">ALPAMAYO-SUPER</span>'
                f'<span class="av">{html.escape(alp["lane"])}</span>'
                f'<span class="av">{html.escape(alp["lateral"])}</span>'
                f'<span class="av">{html.escape(alp["longitudinal"])}</span>'
                f'<p class="cot">&ldquo;{html.escape(alp["cot"])}&rdquo;</p></div>')
    else:
        alph = ('<div class="al none"><span class="src">ALPAMAYO-SUPER</span>'
                '<span class="av miss">no inference for this clip</span></div>')

    onset = geo["t_yaw_onset_25deg_s"]
    band_yaw = geo["yaw_in_strategic_band_deg"]
    ev = [("peak yaw", f'{geo["peak_yaw_after_key_deg"]:+.1f}&deg;'),
          ("turn onset", f"t+{onset}s" if onset is not None else "&mdash;"),
          ("yaw in strat band", f"{band_yaw:+.1f}&deg;" if band_yaw is not None
           else "&mdash;"),
          ("speed", f'{geo["v_at_key_mps"]:.2f}&rarr;{geo["v_end_mps"]:.2f} m/s'),
          ("min speed", f'{geo["v_min_future_mps"]:.2f} m/s'),
          ("stops", "yes" if geo["comes_to_a_stop"] else "no")]
    evh = "".join(f'<div class="ev"><dt>{k}</dt><dd>{v}</dd></div>'
                  for k, v in ev)
    a_tok = "<em>ABSTAIN</em>" if a["abstain"] else html.escape(a["token"] or "—")

    return f'''<article class="card{' is-flag' if flagged else ''}">
 <header class="card-h">
  <div class="card-id"><span class="mono">{cid}</span>
   <span class="split">{r["label_split"]}</span></div>
  <div class="hz mono">future {hz["future_available_s"]}s &middot;
   strat band {hz["band_observable_s"]}s/22s</div>
 </header>
 <div class="strip">{figs}<div class="bevwrap">{bev(r["traj_ego"])}
  <div class="bevkey"><span class="k-past"></span>past<span class="k-fut"></span>future<span class="k-band"></span>strat</div></div></div>
 <div class="body">
  <div class="labels">
   <div class="lab"><span class="lab-f">STRATEGIC</span>
    <span class="tok">{html.escape(g["token"])}</span>
    <span class="tok">{a_tok}</span></div>
   <div class="lab"><span class="lab-f">TACTICAL</span>
    <span class="tok t2">{tv["lat"]}</span>
    <span class="tok t2">{tv["lon"]}</span>
    <span class="conf">2.0 s horizon</span></div>
   {alph}
  </div>
  <dl class="evid">{evh}</dl>
 </div>
 <div class="verd"><p class="mine"><span class="vl">my read</span>
  {JUDGE.get(cid, "")}</p></div>
</article>'''


def main() -> None:
    rows = {r["clip_id"][:8]: r for r in
            json.loads((V / "sample_v2.json").read_text(encoding="utf-8"))}
    cards = "\n".join(card(rows[c]) for c in SHOW if c in rows)
    doc = TEMPLATE.replace("@@CARDS@@", cards)
    assert "@@" not in doc, "unreplaced placeholder"
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), {len(SHOW)} cards")


TEMPLATE = r'''<title>Strategic Label Spot-Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
 --ground:#F6F8FA;--panel:#FFF;--panel-2:#EEF2F6;
 --ink:#0F1720;--ink-2:#44525F;--ink-3:#71818F;
 --line:#D9E1E9;--line-2:#C3CFDA;
 --accent:#0057D9;--accent-soft:#E4EDFC;
 --past:#8FA1B3;--band:#7A3DBD;--band-soft:#F0E7FA;
 --ok:#0F7A50;--ok-bg:#E3F3EB;--flag:#B4400C;--flag-bg:#FBE9E0;
 --shadow:0 1px 2px rgba(15,23,32,.06),0 8px 24px -12px rgba(15,23,32,.18);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0B1015;--panel:#131B23;--panel-2:#1A242E;
 --ink:#E7EEF5;--ink-2:#A6B6C4;--ink-3:#7C8D9C;
 --line:#243039;--line-2:#31404C;
 --accent:#5C9BFF;--accent-soft:#152439;
 --past:#6B7C8C;--band:#B489EE;--band-soft:#1E1630;
 --ok:#4FC08A;--ok-bg:#12291F;--flag:#FF8A5B;--flag-bg:#2E1810;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
 --ground:#0B1015;--panel:#131B23;--panel-2:#1A242E;
 --ink:#E7EEF5;--ink-2:#A6B6C4;--ink-3:#7C8D9C;
 --line:#243039;--line-2:#31404C;
 --accent:#5C9BFF;--accent-soft:#152439;
 --past:#6B7C8C;--band:#B489EE;--band-soft:#1E1630;
 --ok:#4FC08A;--ok-bg:#12291F;--flag:#FF8A5B;--flag-bg:#2E1810;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"Source Sans 3",ui-sans-serif,system-ui,sans-serif;
 font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;
 font-variant-numeric:tabular-nums}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 96px}
h1,h2,h3{font-family:Archivo,ui-sans-serif,system-ui,sans-serif;
 text-wrap:balance;margin:0;line-height:1.15;letter-spacing:-.012em}
h1{font-size:clamp(30px,4.4vw,46px);font-weight:700}
h2{font-size:24px;font-weight:600}
p{margin:0}
.eyebrow{font-family:Archivo,sans-serif;font-size:11.5px;font-weight:600;
 letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
.lede{max-width:68ch;color:var(--ink-2);margin-top:14px;font-size:17px}
header.top{border-bottom:1px solid var(--line);padding-bottom:32px;
 display:flex;flex-direction:column;gap:6px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.mtag{font-size:12.5px;padding:4px 10px;border:1px solid var(--line-2);
 border-radius:2px;color:var(--ink-2);background:var(--panel)}
.mtag.warn{border-color:var(--flag);color:var(--flag);background:var(--flag-bg)}
section{margin-top:56px}
.sechead{display:flex;flex-direction:column;gap:6px;margin-bottom:22px}
.sechead p{color:var(--ink-2);max-width:74ch}

.alert{background:var(--band-soft);border:1px solid var(--band);
 border-left-width:3px;border-radius:3px;padding:24px 26px}
.alert h3{font-size:19px;margin-bottom:12px;color:var(--ink)}
.alert p{color:var(--ink-2);max-width:76ch}
.alert p+p{margin-top:10px}
.alert strong{color:var(--ink)}
.hbar{margin:18px 0 6px;height:30px;display:flex;border-radius:2px;
 overflow:hidden;border:1px solid var(--line-2)}
.hbar .seen{background:var(--band);color:#fff;display:flex;align-items:center;
 justify-content:center;font-size:11.5px;font-weight:600;
 font-family:"JetBrains Mono",monospace}
.hbar .unseen{background:repeating-linear-gradient(45deg,var(--panel-2),
 var(--panel-2) 6px,var(--line) 6px,var(--line) 12px);flex:1;display:flex;
 align-items:center;justify-content:center;font-size:11.5px;color:var(--ink-3);
 font-family:"JetBrains Mono",monospace}

.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:16px}
.score{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 padding:20px;box-shadow:var(--shadow)}
.score .n{font-family:Archivo,sans-serif;font-size:38px;font-weight:700;
 font-variant-numeric:tabular-nums;line-height:1;letter-spacing:-.02em}
.score .n.bad{color:var(--flag)}
.score .l{font-size:12px;color:var(--ink-3);margin-top:8px;
 text-transform:uppercase;letter-spacing:.08em}
.score .s{font-size:13.5px;color:var(--ink-2);margin-top:9px}

.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 margin-bottom:20px;overflow:hidden;box-shadow:var(--shadow)}
.card.is-flag{border-left:3px solid var(--flag)}
.card-h{display:flex;justify-content:space-between;align-items:center;gap:12px;
 padding:11px 18px;border-bottom:1px solid var(--line);background:var(--panel-2)}
.card-id{display:flex;align-items:baseline;gap:10px;font-size:14px;font-weight:600}
.split{font-size:11px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em}
.hz{font-size:11px;color:var(--ink-3)}
.strip{display:grid;grid-template-columns:repeat(6,1fr) 1.4fr;gap:8px;padding:16px}
.strip figure{margin:0;display:flex;flex-direction:column;gap:5px}
.strip img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:2px;
 border:1px solid var(--line);display:block}
.strip figcaption{font-family:"JetBrains Mono",monospace;font-size:9.5px;
 color:var(--ink-3);letter-spacing:.02em}
.strip .key img{border:2px solid var(--accent)}
.strip .key figcaption{color:var(--accent);font-weight:600}
.strip .band img{border:2px solid var(--band)}
.strip .band figcaption{color:var(--band);font-weight:600}
.bevwrap{display:flex;flex-direction:column;gap:5px}
.bev{width:100%;background:var(--panel-2);border:1px solid var(--line);
 border-radius:2px;aspect-ratio:300/200}
.bev-grid line{stroke:var(--line-2);stroke-width:.6}
.bev-past{fill:none;stroke:var(--past);stroke-width:2;stroke-dasharray:3 3}
.bev-fut{fill:none;stroke:var(--accent);stroke-width:2.4;stroke-linecap:round}
.bev-band{fill:none;stroke:var(--band);stroke-width:3;stroke-linecap:round}
.bev-ego{fill:var(--ink);stroke:var(--panel);stroke-width:1.4}
.bev-sc{fill:var(--ink-3);font-size:9px;font-family:"JetBrains Mono",monospace}
.bevkey{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--ink-3);
 display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.bevkey span{width:12px;height:2px;display:inline-block}
.k-past{background:var(--past)}.k-fut{background:var(--accent)}
.k-band{background:var(--band)}
.body{display:grid;grid-template-columns:330px 1fr;gap:20px;padding:0 18px 14px}
.labels{display:flex;flex-direction:column;gap:9px}
.lab{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.lab-f{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--ink-3);
 letter-spacing:.08em;min-width:66px}
.tok{font-family:"JetBrains Mono",monospace;font-size:12.5px;font-weight:600;
 color:var(--ink);background:var(--panel-2);padding:2px 7px;border-radius:2px;
 border:1px solid var(--line)}
.tok.t2{color:var(--accent);border-color:var(--accent)}
.tok em{font-style:normal;color:var(--ink-3)}
.conf{font-size:11px;color:var(--ink-3)}
.al{border-top:1px dashed var(--line-2);padding-top:9px;display:flex;
 flex-wrap:wrap;align-items:center;gap:6px}
.src{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--ink-3);
 letter-spacing:.08em;width:100%}
.av{font-size:12px;background:var(--accent-soft);color:var(--ink-2);
 padding:2px 7px;border-radius:2px}
.av.miss{background:transparent;color:var(--ink-3);font-style:italic}
.cot{width:100%;font-size:13px;color:var(--ink-2);font-style:italic;margin-top:4px}
.evid{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));
 gap:10px 14px;margin:0}
.ev dt{font-size:10px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.07em}
.ev dd{margin:2px 0 0;font-family:"JetBrains Mono",monospace;font-size:12.5px;
 color:var(--ink);font-weight:600}
.verd{border-top:1px solid var(--line);padding:13px 18px 16px;background:var(--panel-2)}
.verd p{font-size:14.5px;color:var(--ink);max-width:92ch}
.vl{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.08em;
 text-transform:uppercase;color:var(--accent);margin-right:8px}
.rec{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 padding:26px;box-shadow:var(--shadow)}
.rec ol{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:13px}
.rec li{color:var(--ink-2);max-width:82ch}
.rec li strong{color:var(--ink)}
table{width:100%;border-collapse:collapse;font-size:14px}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
 border-radius:3px;box-shadow:var(--shadow)}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line)}
th{font-family:Archivo,sans-serif;font-size:11px;text-transform:uppercase;
 letter-spacing:.08em;color:var(--ink-3);font-weight:600}
tr:last-child td{border-bottom:none}
td.n{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
td .bad{color:var(--flag);font-weight:600}
footer{margin-top:56px;padding-top:22px;border-top:1px solid var(--line);
 color:var(--ink-3);font-size:13px;max-width:84ch}
@media (max-width:1000px){
 .strip{grid-template-columns:repeat(3,1fr)}
 .bevwrap{grid-column:1/-1}
 .body{grid-template-columns:1fr}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="wrap">
<header class="top">
 <p class="eyebrow">TanitAD &middot; DataFlyWheel &middot; 2026-08-23</p>
 <h1>Strategic Label Spot-Check</h1>
 <p class="lede">Thirty-nine clips where an extracted label could be joined to real frames and
  real ego poses. Every layer we produce is shown together &mdash; strategic goal and action,
  the factored tactical pair, and Alpamayo-Super's own read &mdash; against geometry computed
  independently from the poses.</p>
 <div class="meta">
  <span class="mtag mono">n = 39 clips</span>
  <span class="mtag mono">strategic 87.2% / 92.3%</span>
  <span class="mtag mono">Alpamayo on 16/39</span>
  <span class="mtag warn">strategic band only 18% observable</span>
  <span class="mtag warn">NON-PARITY cache</span>
 </div>
</header>

<section>
 <div class="sechead"><p class="eyebrow">Read this first</p>
  <h2>The corpus cannot show us the strategic horizon</h2></div>
 <div class="alert">
  <h3>81% of a strategic label's own definition lies past the end of the recording</h3>
  <p>A strategic goal is defined over <strong>key frame + 8 s to 30 s</strong>. These clips are
   <strong>19.9 s long</strong> and the label anchor sits at <strong>7.8 s</strong>, leaving
   12.0 s of future. The required band is 15.8&ndash;37.8 s of clip time; the recording stops
   at 19.9 s.</p>
  <div class="hbar"><div class="seen" style="width:18.2%">4.0 s observed</div>
   <div class="unseen">17.9 s past the end of the clip &mdash; never recorded</div></div>
  <p>So <strong>4.0 s of a 22 s band is observable, 18.2%</strong>. Every strategic label on
   this corpus is an extrapolation over the remaining 81%, and no amount of label-side work
   changes that &mdash; it is a property of 20-second clips, not of the extractor.</p>
  <p>Two consequences worth deciding on. <strong>Validation:</strong> we cannot confirm or
   refute the strategic band from PhysicalAI at all; the checks below cover 0&ndash;12 s, which
   is the tactical and near-strategic range only. <strong>Supervision:</strong> a head trained
   to predict 8&ndash;30 s from these labels is being taught to guess &mdash; the target beyond
   12 s was inferred, not observed. If the strategic layer is to be evidenced rather than
   asserted, it needs a corpus with longer sequences.</p>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">Agreement</p>
  <h2>Where each layer stands</h2>
  <p>The geometric check reads ego poses only &mdash; heading, speed, stop events. It cannot see
   lanes, signs or traffic, so a disagreement is a flag for review, not proof of error.</p></div>
 <div class="scores">
  <div class="score"><div class="n">34/39</div><div class="l">strategic goal</div>
   <p class="s">87.2%. Three of the five flags are one defect.</p></div>
  <div class="score"><div class="n">36/39</div><div class="l">strategic action</div>
   <p class="s">92.3%. One outright inversion.</p></div>
  <div class="score"><div class="n">37/39</div><div class="l">tactical v1 vs v2</div>
   <p class="s">The curvature gate changes 2 clips &mdash; it is doing little here.</p></div>
  <div class="score"><div class="n bad">9/16</div><div class="l">Alpamayo LON vs ours</div>
   <p class="s">56%, with <strong>25% outright sign contradictions</strong>.</p></div>
 </div>

 <div class="tw" style="margin-top:16px"><table>
  <thead><tr><th>Layer</th><th>Agreement</th><th>n</th><th>What it means</th></tr></thead>
  <tbody>
  <tr><td>g_str &mdash; TURN_LEFT</td><td class="n">100%</td><td class="n">13</td>
   <td>Turn classes are the strong part of the extractor.</td></tr>
  <tr><td>g_str &mdash; STOP_AT</td><td class="n">100%</td><td class="n">7</td>
   <td>Every one reaches v=0. Cleanest class in the set.</td></tr>
  <tr><td>g_str &mdash; TURN_RIGHT</td><td class="n">87.5%</td><td class="n">8</td>
   <td>One sign conflict that needs the video to settle.</td></tr>
  <tr><td>g_str &mdash; FOLLOW_MAIN_ROAD</td><td class="n"><span class="bad">70%</span></td><td class="n">10</td>
   <td>The weak class. All three failures are junction turns absorbed by the fallback token.</td></tr>
  <tr><td>Alpamayo lane &rarr; tactical LAT</td><td class="n">80%</td><td class="n">15</td>
   <td>Consistent with the banked finding that a lane change is indistinguishable from a lane keep on that axis.</td></tr>
  <tr><td>Alpamayo LON &rarr; tactical LON</td><td class="n"><span class="bad">56%</span></td><td class="n">16</td>
   <td>4 of 16 are decel-vs-accel contradictions. Either the horizons differ or one source is wrong; unresolved.</td></tr>
  <tr><td>strategic TURN_* with tactical lane_keep</td><td class="n">52%</td><td class="n">21</td>
   <td><strong>Expected, not a defect</strong> &mdash; tactical spans 2.0 s and the turns start at a median of 3.7 s.</td></tr>
  </tbody></table></div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">The sample &mdash; 10 clips</p>
  <h2>Every layer, on every clip</h2>
  <p>Six frames per clip from t&minus;2 s to t+12 s. The blue frame is the label anchor; purple
   frames and the purple track segment fall inside the strategic band. Bird's-eye is the ego
   frame at the anchor, past dashed. All six flagged clips are here, plus four the extractor
   gets right.</p></div>
 @@CARDS@@
</section>

<section>
 <div class="sechead"><p class="eyebrow">Recommendation</p>
  <h2>What to fix before scaling</h2></div>
 <div class="rec"><ol>
  <li><strong>Decide the strategic horizon question first &mdash; it outranks every label fix.</strong>
   Either narrow the strategic definition to what 20 s clips can evidence (roughly 0&ndash;12 s),
   or source longer sequences. Scaling label extraction now bakes an 81% unobserved
   extrapolation into the whole dataset.</li>
  <li><strong>Use the disagreement we already compute.</strong> On two of the six flagged clips
   our own tactical family already contradicts our strategic label, and on a third Alpamayo
   does. Nothing consumes that today. A cross-family consistency check is the cheapest guard
   available and needs no new data.</li>
  <li><strong>Guard FOLLOW_MAIN_ROAD.</strong> At 70% on n=10 it is the one class with a
   systematic, one-directional defect: it is the fallback token, so undetected turns land in it
   silently. If the hindsight path rotates more than ~25&deg; on the horizon, the clip must emit
   a turn token or abstain &mdash; never corridor-follow.</li>
  <li><strong>Settle the Alpamayo longitudinal conflict.</strong> 56% agreement with 25% sign
   contradictions is too low to consume as corroboration. Establish whether the two read
   different horizons before either is trusted; right now we have two longitudinal opinions and
   no basis for preferring one.</li>
  <li><strong>Validate the 80 abstained rows separately.</strong> None had local frames, so this
   session's abstain change is <em>not</em> evidenced here.</li>
  <li><strong>Then re-run at n&nbsp;&asymp;&nbsp;300 on the parity corpus.</strong> At n=39 the
   per-class counts cannot set a threshold &mdash; FOLLOW_MAIN_ROAD's 70% is 7 of 10.</li>
 </ol></div>
</section>

<footer><p><strong>How to read these numbers.</strong> Frames and poses come from the local cache
 <span class="mono">physicalai-train-14231cd29c74</span>, which is <strong>not</strong> the parity
 corpus <span class="mono">e438721ae894</span>/<span class="mono">f09e44db</span> &mdash; nothing
 here is cross-arm comparable. The sample is every clip that happened to be both labelled and
 locally cached: a convenience sample whose class frequencies say nothing about the corpus. The
 label&harr;frame join runs through <span class="mono">episode_id_legacy</span>, which collides by
 design; 3 label-side and 3 cache-side colliding ids were refused rather than resolved. Camera
 frames are the 256&times;256 model-input crops, not the full field of view. Tactical labels are
 <span class="mono">refc_tactical.window_factored_labels_v2</span> at a 2.0 s horizon. Alpamayo
 rows are the banked per-clip taxonomy and cover 16 of 39 clips. The geometric check is a second
 opinion from ego poses alone and is not ground truth.</p></footer>
</div>
'''

if __name__ == "__main__":
    main()
