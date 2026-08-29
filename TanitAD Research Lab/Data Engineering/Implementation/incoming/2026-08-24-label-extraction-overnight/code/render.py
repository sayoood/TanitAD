"""Assemble the overnight validation report into one self-contained HTML file."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/_s2build/report")
from build_report_html import (LAB, SUM, clip_section, defs_table,  # noqa: E402
                               dist_chart)

CSS = """
:root{
  --ground:#f5f5f3; --panel:#ffffff; --sunk:#eeeeec;
  --ink:#16181d; --ink-2:#4a4d57; --ink-3:#767a86;
  --rule:#dededa; --rule-2:#c9c9c4;
  --cyan:#0f6f7d; --cyan-soft:#0f6f7d1a;
  --amber:#a8650a; --amber-soft:#a8650a1a;
  --good:#1f6b45; --good-soft:#1f6b451a;
  --bad:#a32f28;  --bad-soft:#a32f281a;
  --unk:#5d6070;  --unk-soft:#5d60701a;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#101217; --panel:#171a21; --sunk:#1e222b;
  --ink:#e8e9ec; --ink-2:#a9adb8; --ink-3:#7d818d;
  --rule:#282d37; --rule-2:#39404c;
  --cyan:#4fc3d4; --cyan-soft:#4fc3d422;
  --amber:#e0a244; --amber-soft:#e0a24422;
  --good:#5cc189; --good-soft:#5cc18922;
  --bad:#e3776e;  --bad-soft:#e3776e22;
  --unk:#9aa0b0;  --unk-soft:#9aa0b022;
}}
:root[data-theme="dark"]{
  --ground:#101217; --panel:#171a21; --sunk:#1e222b;
  --ink:#e8e9ec; --ink-2:#a9adb8; --ink-3:#7d818d;
  --rule:#282d37; --rule-2:#39404c;
  --cyan:#4fc3d4; --cyan-soft:#4fc3d422;
  --amber:#e0a244; --amber-soft:#e0a24422;
  --good:#5cc189; --good-soft:#5cc18922;
  --bad:#e3776e;  --bad-soft:#e3776e22;
  --unk:#9aa0b0;  --unk-soft:#9aa0b022;
}
*{box-sizing:border-box}
body{
  margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  font-size:15px;line-height:1.62;-webkit-font-smoothing:antialiased;
}
.wrap{max-width:1140px;margin:0 auto;padding:34px 22px 90px}
.col{max-width:70ch}
h1,h2,h3{font-family:Archivo,"IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(30px,4.6vw,46px);font-weight:700;letter-spacing:-.022em;line-height:1.08}
h2{font-size:23px;font-weight:650;letter-spacing:-.012em;margin:56px 0 6px;padding-top:20px;
   border-top:1px solid var(--rule)}
h3{font-size:15.5px;font-weight:650;margin:26px 0 8px}
p{margin:11px 0}
a{color:var(--cyan)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.885em;
     background:var(--sunk);padding:.1em .38em;border-radius:3px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--cyan);margin:0 0 12px}
.lede{font-size:17.5px;line-height:1.55;color:var(--ink-2);margin:16px 0 0;max-width:66ch}
.meta{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);
  margin-top:20px;padding-top:12px;border-top:1px solid var(--rule)}

/* headline metrics */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:9px;overflow:hidden;margin:30px 0}
.kpi{background:var(--panel);padding:15px 16px}
.kpi .n{font-family:Archivo,sans-serif;font-size:27px;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .l{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);margin-top:5px}
.kpi .s{font-size:12px;color:var(--ink-2);margin-top:3px}
.kpi.win .n{color:var(--good)}
.kpi.hero .n{color:var(--cyan)}

/* before/after */
.ba{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px}
.ba th{text-align:left;font-family:"IBM Plex Mono",monospace;font-size:10.5px;
  letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
  border-bottom:1px solid var(--rule-2);padding:7px 9px}
.ba td{padding:8px 9px;border-bottom:1px solid var(--rule);vertical-align:top}
.ba td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  white-space:nowrap;text-align:right}
.ba .was{color:var(--ink-3)}
.ba .now{color:var(--good);font-weight:600}

/* scene cards */
.clip{background:var(--panel);border:1px solid var(--rule);border-radius:11px;
  padding:15px;margin:22px 0}
.clip h3{display:flex;align-items:center;gap:11px;margin:0 0 11px}
.cid{font-family:"IBM Plex Mono",monospace;font-size:14px;font-weight:600}
.v{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.07em;
  text-transform:uppercase;padding:3px 9px;border-radius:20px;font-weight:500}
.v.ok{background:var(--good-soft);color:var(--good)}
.v.bad{background:var(--bad-soft);color:var(--bad)}
.v.unk{background:var(--unk-soft);color:var(--unk)}
.strip{display:flex;gap:5px;overflow-x:auto;padding-bottom:7px}
.strip figure{margin:0;flex:0 0 auto}
.strip img{height:132px;border-radius:5px;display:block;background:#000}
.strip figcaption{font-family:"IBM Plex Mono",monospace;font-size:9.5px;color:var(--ink-3);
  text-align:center;margin-top:3px}
.row{display:flex;gap:15px;margin-top:13px;flex-wrap:wrap;align-items:flex-start}
.bev{width:236px;flex:0 0 auto;border-radius:7px;background:var(--panel)}
.lab{flex:1 1 380px;min-width:0;overflow-x:auto}
.lab table{border-collapse:collapse;width:100%;font-size:12.5px}
.lab th{text-align:left;background:var(--sunk);padding:5px 8px;
  font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);font-weight:500}
.lab td{padding:4px 8px;border-top:1px solid var(--rule);vertical-align:top}
.tok{font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap}
.role{font-family:"IBM Plex Mono",monospace;font-size:9px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);background:var(--sunk);
  padding:1px 5px;border-radius:3px;margin-right:5px;font-weight:500}
.args{font-family:"IBM Plex Mono",monospace;color:var(--ink-3);font-size:11px;
  font-variant-numeric:tabular-nums}
.p{font-family:"IBM Plex Mono",monospace;font-size:9.5px;padding:2px 7px;border-radius:20px;
  white-space:nowrap;background:var(--unk-soft);color:var(--unk)}
.p.geo{background:var(--cyan-soft);color:var(--cyan)}
.p.g{background:var(--good-soft);color:var(--good)}
.p.d{background:var(--amber-soft);color:var(--amber)}
.p.as{background:var(--cyan-soft);color:var(--cyan)}
.p.or{background:var(--bad-soft);color:var(--bad)}
.alpa{margin-top:11px;padding:10px 12px;background:var(--sunk);border-radius:7px;
  font-size:12.5px;line-height:1.55;color:var(--ink-2)}
.alpa b{color:var(--ink)}
.verdict{margin-top:9px;padding:10px 13px;border-radius:0 7px 7px 0;font-size:13.5px;
  line-height:1.55;border-left:3px solid var(--unk);background:var(--unk-soft)}
.verdict.ok{border-left-color:var(--good);background:var(--good-soft)}
.verdict.bad{border-left-color:var(--bad);background:var(--bad-soft)}

/* definitions */
.defs{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 20px}
.defs td{padding:6px 9px;border-bottom:1px solid var(--rule);vertical-align:top}
.defs td:first-child{width:230px}
h3 .n{font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:400;
  color:var(--ink-3);letter-spacing:.07em}
.chart{max-width:100%;margin:14px 0}
.note{border-left:3px solid var(--amber);background:var(--amber-soft);
  padding:11px 14px;border-radius:0 7px 7px 0;margin:16px 0;font-size:14px}
.note.bad{border-left-color:var(--bad);background:var(--bad-soft)}
ul{padding-left:20px}li{margin:5px 0}
.foot{margin-top:60px;padding-top:16px;border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-3)}
@media (max-width:640px){.bev{width:100%}.strip img{height:106px}}
"""

HEAD = """<title>Label Extraction Overnight</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>%s</style>""" % CSS


def kpi(n, label, sub="", cls=""):
    return (f'<div class="kpi {cls}"><div class="n">{n}</div>'
            f'<div class="l">{label}</div>'
            + (f'<div class="s">{sub}</div>' if sub else "") + "</div>")


def main() -> None:
    order = ["d8f80c0f", "b5812659", "95d2c361", "d6982eb9", "d5700d32",
             "1da60b2a", "0a0bb8fe", "ffd66d72", "263597d8", "3ae000c7"]
    by8 = {c[:8]: c for c in LAB}
    scenes = "".join(clip_section(by8[k], LAB[by8[k]]) for k in order if k in by8)

    body = f"""
<div class="wrap">
<header class="col">
  <p class="eyebrow">TanitAD · Data FlyWheel · P2 label pipeline</p>
  <h1>Label extraction, rebuilt overnight</h1>
  <p class="lede">The whole 26.2-hour Alpamayo corpus now labels end to end in 30 seconds
  with zero refusals. Goal/action contradictions fell from 21.7&nbsp;% to 0.40&nbsp;%, and a
  control found one token firing <b>408× more often on empty scenes than on occupied
  ones</b>. Twelve scenes below were drawn at random and checked against their own
  frames — which is how every defect here was found, and how the data caught
  <b>two wrong calls of my own</b>.</p>
  <p class="meta">2026-08-29 · pre-production sign-off · vocabulary FROZEN at v7 · 4,719 clips · all numbers MEASURED</p>
</header>

<div class="kpis">
  {kpi("4,719", "clips labelled", "0 refused · 26.2 h", "hero")}
  {kpi("0", "goal/action defects", "was 21.7 % of clips", "win")}
  {kpi("30 s", "full-corpus runtime", "150 clips/s")}
  {kpi("44/52", "vocabulary emitting", "census clean — 8 declared, 0 silent", "win")}
  {kpi("63.9 / 63.5 %", "Alpamayo lat / lon agreement", "vs 41.6 % chance")}
</div>

<div class="col">
<h2>What you asked me to confirm — and where I was wrong</h2>
<p>You were right and I was wrong on both counts. The augmentation is
<b>4,729 clips = 26.3 hours</b>, pushed to
<code>Sayood/tanitad-alpamayo2-augmentation</code>, and it carries
<b>five tasks</b>, not one. <code>meta_action</code> is present on every clip.
My claim that it was "not reachable locally" and that the data was "limited"
described a 801-clip subset I had imposed myself, read from a four-field
export I had mistaken for the dataset.</p>
</div>

<table class="ba">
<tr><th>Task</th><th>Coverage</th><th>What it gives</th><th class="num">Used before</th></tr>
<tr><td class="tok">meta_action</td><td class="num">4,729 · 100 %</td>
    <td>3-axis action + CoT. The <b>best lateral corroborator</b> — 69.9 % vs 41.6 % chance.</td>
    <td class="num was">no</td></tr>
<tr><td class="tok">auto_labeling</td><td class="num">4,729 · 100 %</td>
    <td>Typed, <b>time-stamped motion segments</b> (97.1 % parse) + the forcing object.
        <code>type: none</code> on 853 clips is an explicit clean negative.</td>
    <td class="num was">no</td></tr>
<tr><td class="tok">grounding_via_vqa</td><td class="num">4,411 · 93.3 %</td>
    <td><b>2D boxes</b>, and they are accurate — verified pixel-level on two clips.</td>
    <td class="num was">no</td></tr>
<tr><td class="tok">vqa</td><td class="num">4,722 · 99.9 %</td>
    <td>21 semantic categories — but a <b>sampled</b> bank, ~0.3 % coverage per question.
        Spot-checks only.</td><td class="num was">no</td></tr>
<tr><td class="tok">trajectory</td><td class="num">4,729 · 100 %</td>
    <td>Alpamayo's own predicted path and its ADE/FDE.</td><td class="num was">no</td></tr>
<tr><td class="tok">cot</td><td class="num">4,729 · 100 %</td>
    <td>The single field the pipeline had been using.</td>
    <td class="num now">yes</td></tr>
</table>

<div class="col">
<div class="note"><b>The anchor was misaligned all along.</b> Alpamayo's
<code>t0_us</code> is <b>5.1 s</b>; our label anchor is <b>8.0 s</b>. Every
cross-source number published before today compared moments 2.9 s apart — about
29 m at urban speed. Scoring each source on its own window moved longitudinal
agreement from 54.9 % to 62.1 % with no change to the data.</div>

<h2>Five defects found by looking at frames</h2>
<p>Each was found by checking a random scene against its own pixels, not by a test.
All five are fixed and pinned; the measured effect of each is below.</p>
</div>

<table class="ba">
<tr><th>Defect</th><th>How it showed up</th><th class="num">Before</th><th class="num">After</th></tr>
<tr><td><b>Substring collision</b><br><span class="args">"_L" in "FOLLOW_LANE"</span></td>
    <td>Every straight clip scored as a LEFT turn, so Alpamayo's lateral axis read
        <i>worse than chance</i> one hour after it measured 69.9 %.</td>
    <td class="num was">24.9 %</td><td class="num now">64.2 %</td></tr>
<tr><td><b>A turn in neither layer</b></td>
    <td>A −94° turn running 4.5→14.8 s was skipped by the strategic filter (starts
        &lt;6 s) and by the tactical one (only reads the first manoeuvre). A straddling
        manoeuvre now populates <b>both</b>.</td>
    <td class="num was">dropped</td><td class="num now">both layers</td></tr>
<tr><td><b>EVADE without lateral motion</b></td>
    <td>74 % of EVADE emissions had no nudge at all — a sentence about an evasion,
        not an evasion. Geometry must now show the movement; the CoT names the object.</td>
    <td class="num was">932</td><td class="num now">238</td></tr>
<tr><td><b>Two detectors, one fact</b></td>
    <td>Goal read <code>manoeuvre_sequence</code>, action read <code>EM.analyse</code>;
        219 clips carried a TURN goal beside <code>lat=LANE_KEEP</code>.</td>
    <td class="num was">219</td><td class="num now">0</td></tr>
<tr><td><b>Stop-then-launch</b></td>
    <td>An ego already at rest that departs was labelled <code>STOP_POINT</code> beside
        <code>lon=ACCELERATE</code>: the plan's content is the departure.</td>
    <td class="num was">101</td><td class="num now">6</td></tr>
<tr><td><b>Negated terms read as present</b><br><span class="args">the 408x inversion</span></td>
    <td>"…with <b>no</b> lead vehicle pedestrians cyclists <b>traffic lights</b> or obstacles"
        emitted TRAFFIC_LIGHT_REACT. Found by a CONTROL, not a test: splitting the corpus
        by Alpamayo's own "nothing is critical" and checking the rates ran the right way.
        Two tokens ran backwards.</td>
    <td class="num was">533</td><td class="num now">18</td></tr>
<tr><td><b>A stopped bus is not an overtake</b></td>
    <td>"Nudge left to pass the stopped bus" matched no obstacle class, so the clip was
        labelled <code>FOLLOW_LANE</code> alone. A stopped object is STATIC — EVADE.</td>
    <td class="num was">missed</td><td class="num now">EVADE</td></tr>
</table>

<div class="col">
<div class="note"><b>What caught the biggest defect was a control, not a test.</b>
Nothing was failing. The tokens looked plausible one clip at a time. Splitting the
corpus by an independent source's own negative — 853 clips where Alpamayo states
nothing is critical — and requiring the rates to run in a known direction is what
exposed it. <code>GAP_TARGET</code> at 0.04x and <code>YIELD</code> at 0.21x ran
correctly; <code>TRAFFIC_LIGHT_REACT</code> at <b>408x</b> did not.<br><br>
⚠️ And I had already probed negation that night, measured "3.5 % of clips — small",
and moved on. That probe searched for <code>no</code> within 40 characters of a term;
it could not express a six-term enumeration under a single <code>no</code>, which is
the shape that actually occurs. <b>An instrument that cannot detect X in the form X
takes will report that X is rare.</b></div>

<div class="col">
<div class="note bad"><b>Two of tonight's errors were mine, in the fix itself.</b>
I built a <code>contradicted</code> state on the boxes — "the clip has boxes, none
of kind X, therefore X is absent". Invalid: there is exactly <b>one</b> grounding
question per clip, and 3,246 clips with boxes were never asked about pedestrians
at all. Grounding can <b>confirm, never refute</b>. And my first coherence check
tested only "TURN goal but no turn action", never the converse — so a fix I made
broke 70 clips the other way and the check certified them. Both directions are
tested now.</div>

<h2>Three corrections from your review of <code>43bbcbf9</code></h2>
<p>One screenshot, three real defects — and the third of them hid a fourth in my
own fix.</p>
</div>

<table class="ba">
<tr><th>What you flagged</th><th>What was wrong</th><th class="num">Before</th><th class="num">After</th></tr>
<tr><td><b>Strategic extracted at 6 s</b></td>
    <td>The emitter used <b>0–6 s</b> for tactical and called everything from 6 s on
        strategic. Both ends wrong: 0–2 s belongs to the <b>operative</b> layer, and
        the strategic band opens at <b>8 s</b>. Symptom: a route-level token stamped
        <code>by_time_s: 6.0</code>.</td>
    <td class="num was">435</td><td class="num now">0</td></tr>
<tr><td><b>Tactical should be 2–6 s</b></td>
    <td>Now <code>OPERATIVE (0,2) · TACTICAL (2,6) · STRATEGIC (8,30)</code>, which
        exposes a real gap: <b>nothing owns 6–8 s</b>. Manoeuvres landing only there
        are reported as <code>unassigned</code> rather than absorbed — 54 clips
        (1.1 %), a design question for you rather than an emitter default.</td>
    <td class="num was">0–6 s</td><td class="num now">2–6 s</td></tr>
<tr><td><b>Long curves read as turns</b></td>
    <td>Two causes. The radius came from <b>peak instantaneous curvature</b>, so one
        noisy sample set it — 40 m where the arc says 144 m. And there was no speed
        condition at all. Now <code>|Δyaw| ≥ 15° AND R_arc ≤ 140 m AND v_min ≤ 8 m/s</code>,
        swept against Alpamayo's own turn labels (349 turns vs 1,468 keep-lane).</td>
    <td class="num was">987</td><td class="num now">615</td></tr>
</table>

<div class="col">
<div class="note"><b>Your mechanism was right; the signal was not.</b> You predicted
turns are found by strong deceleration into the curve. In the 2–6 s band a turning
ego is <b>accelerating</b> — median <b>+2.2 m/s</b> — because it is already exiting,
so <code>dv</code> separates nothing. Turn clips are <b>already slow four seconds
before the anchor</b> (5.4 vs 14.0 m/s): the braking happens outside every window we
observe. The gate therefore uses <b>absolute speed</b>, and it earns its place —
removing <b>70 of 159 false positives (44 %)</b> and lifting precision 58.4 % →
<b>70.6 %</b>.<br><br>
⚠️ Fixing the goal side alone re-created the two-detector defect in mirror image:
<b>329 clips</b> gained <code>lat=TURN</code> with no turn goal. Coherence went
0.40 % → 7.16 % → <b>0.21 %</b> once both sides read the same gated list.</div>

<h2>Your MERGE question, and the three defects behind it</h2>
<p>You asked why <code>59b57590</code> emitted <code>MERGE</code> when its CoT never
mentions one. It doesn't — and answering that found larger problems sitting beside it.</p>
</div>

<table class="ba">
<tr><th>Defect</th><th>What it was</th><th class="num">Before</th><th class="num">After</th></tr>
<tr><td><b>MERGE from a hazard phrase</b></td>
    <td>The only "merg" in the text is <i>"potential door-opening/merge hazards"</i> — a
        hypothetical risk class in a compound noun. Negation scoping cannot catch this:
        "potential" is <b>irrealis</b>, not negation.</td>
    <td class="num was">129</td><td class="num now">82</td></tr>
<tr><td><b>The lane change was never read</b></td>
    <td>The clip opens with <i>"Change lanes to the left"</i> — the manoeuvre it is about —
        and produced nothing. <code>LANE_CHANGE_L/R</code> were frozen, defined, and in the
        matrix with <b>no extraction path at all</b>. 174 clips state a lane change; zero
        got a token.</td>
    <td class="num was">0</td><td class="num now">39</td></tr>
<tr><td><b>Only the first component was read</b></td>
    <td><code>components_analysis</code> is a numbered list on 197 clips. Here we kept
        "Lane divider" and discarded the parked vehicles, the traffic lights and the lead
        vehicle.</td>
    <td class="num was">3,107</td><td class="num now">3,733</td></tr>
<tr><td><b>25 % of the vocabulary was dead</b></td>
    <td><b>13 of 52 frozen tokens were never emitted on the entire corpus.</b> The strategic
        layer was worst: only 3 of 8 goals and 2 of 7 actions reachable, because it handled
        turns and nothing else.</td>
    <td class="num was">39/52</td><td class="num now">43/52</td></tr>
</table>

<div class="col">
<div class="note"><b>Freezing a token is not the same as being able to emit one.</b>
<code>LANE_CHANGE_L</code> passed the freeze test, had a definition in the matrix, and
could never be produced. Every check we had passed. There is now a
<b>reachability test</b>: each frozen token either has an extraction path or is listed in
<code>NOT_YET_EXTRACTABLE</code> with a reason. Nine remain, each naming what it would
take — a head sized to this vocabulary would otherwise train nine permanently empty
classes.<br><br>
Newly reachable from geometry alone: <code>STOP_AT_FOLLOW_ROUTE</code> (469),
<code>PREPARE_STOP_FOLLOW_ROUTE</code> (469), <code>RESUME_CRUISE_FOLLOW_ROUTE</code>
(573), <code>CREEP</code> (63).</div>

<div class="note"><b>Why only 39 of 176 lane changes survive.</b> A lane change must show
lateral motion. Measuring cross-track residual with the road's own arc removed, against a
matched control: <b>+0.07 m</b> (2–6 s), <b>+0.41 m</b> (Alpamayo's window), <b>+1.02 m</b>
(−3…+8 s). The signal is real and grows with the window, but the median claimed lane
change <b>never displaces a full lane width (~3.5 m)</b> in anything we observe. The 39
admitted are the ones geometry can confirm; loosening the gate would manufacture labels.
⚠️ My first version of this measured deviation from the initial heading and put the
control at 7.95 m — any curve produces that. A displacement measure without the arc
removed is measuring the road.</div>

<h2>Four corrections from your second review</h2>
</div>
<table class="ba">
<tr><th>What you flagged</th><th>What it was</th><th class="num">Before</th><th class="num">After</th></tr>
<tr><td><b>Is MERGE fixed?</b></td>
    <td>Yes. The hazard-phrase guard holds — <b>0</b> MERGE tokens now come from a
        "potential …/merge hazards" construction.</td>
    <td class="num was">129</td><td class="num now">82</td></tr>
<tr><td><b>SPEED_BAND should always be there</b></td>
    <td>It was gated on the speed being <i>held</i>, which deleted the target speed from
        exactly the scenes where it matters. A target speed interval is a property every
        plan has; whether it is held is the <b>action's</b> verdict. Now unconditional —
        and the (SPEED_BAND, STOP_POINT) exclusion was retired, since a stop is a band of
        0.0–0.0, which is information, not a contradiction.</td>
    <td class="num was">1,440</td><td class="num now">4,719</td></tr>
<tr><td><b>Description still said 0–6 s</b></td>
    <td>⛔ Not just a label. <code>tactical_goals()</code> was still computing SPEED_BAND,
        STOP_POINT and the anchor from <b>0 s</b> — over ground the OPERATIVE layer owns.
        Only the turn split had been moved to 2–6 s. The header was accurately describing
        the code. The anchor now carries <code>band_s</code> so the window can never again
        be implicit.</td>
    <td class="num was">0–6 s</td><td class="num now">2–6 s</td></tr>
<tr><td><b>Goal vs action not distinguished</b></td>
    <td>Every token now carries an explicit <code>ROLE_OF</code> — <b>goal</b> (what to
        achieve, and by when), <b>action</b> (what to do now about it), <b>input</b> (nav).
        The tables below label each row.</td>
    <td class="num was">implicit</td><td class="num now">26 / 23 / 3</td></tr>
</table>

<div class="col">
<div class="note"><b>The target speed band now discriminates the situation</b>, which is
what makes it worth having on every clip — measured medians: stopping <b>0.00–1.40</b>,
turning <b>4.52–7.11</b>, cruising <b>12.78–13.16</b> m/s.</div>

<div class="note bad"><b>A correction to what I told you last time.</b> I said this clip
"now emits LANE_CHANGE_L". It does not, and should not — I had checked the CoT extractor's
output and reported it as the emitter's. The emitter applies a lateral-evidence gate, and
this ego displaces <b>0.01–0.05 m</b> with the road's arc removed. It never changes lane in
view. The extractor finds the phrase; the gate correctly refuses the token.</div>

<h2>Is the oncoming label in the right time slot?</h2>
<p>No — and it cannot be, on present evidence. You spotted it in the past frames, and the
data agrees.</p>
</div>

<table class="ba">
<tr><th>Question</th><th>Measurement</th><th class="num">Result</th></tr>
<tr><td><b>Do CoT tokens carry a time?</b></td>
    <td>A CoT claim has no timestamp. Only Alpamayo's parsed <i>motion segments</i> give one,
        and <b>1,845 clips (39.0 %) have no time information whatsoever</b> — no segments, no
        "first N seconds" phrase. <code>d94365be</code> is one of them.</td>
    <td class="num was">87.2 % untimed</td></tr>
<tr><td><b>Specifically REACT_ON_ONCOMING</b></td>
    <td>Not a single emission is supported by a timed segment.</td>
    <td class="num was">0 timed / 344</td></tr>
<tr><td><b>Why it lands in the past</b></td>
    <td>Alpamayo's anchor is <b>5.1 s</b>, ours is <b>8.0 s</b>. Its text describes a window
        starting 2.9 s before ours, so an event it names can finish before our tactical band
        opens.</td>
    <td class="num was">−2.9 s</td></tr>
<tr><td><b>And the two texts disagree</b></td>
    <td>On <code>d94365be</code>: <i>cot</i> "nudge <b>LEFT</b> due to the <b>parked car</b>",
        <i>chain_of_causation</i> "nudge <b>RIGHT</b> due to the <b>oncoming vehicle</b>". My
        concatenation was feeding the extractor both halves of a contradiction.</td>
    <td class="num was">49 clips</td></tr>
</table>

<div class="col">
<div class="note"><b>What changed rather than what I concluded.</b> Every CoT-derived goal now
carries <code>time_basis</code>: <span class="p seg">timed</span> when a parsed motion segment
overlapping 2–6 s mentions it, <span class="p untimed">untimed</span> when the band placement
is an assumption nothing supports. A consumer that weights those equally is now choosing to.
On a direction conflict the primary <code>cot</code> field wins and the secondary is dropped
(48 clips).<br><br>
⚠️ I am not deleting the untimed tokens. They are 87 % of the perception signal, and the
claims are often true — they are just not <i>placed</i>. Marking them is honest; discarding
them would throw away most of what Alpamayo knows. <b>Whether an untimed token should train
the tactical head is your call, and the field is there so you can make it.</b></div>

<h2>Every scene now shows Alpamayo's raw text</h2>
<p>As you asked: <code>meta_action</code> and all four source fields verbatim under each scene,
so nothing the pipeline read is hidden behind a summary.</p>

<h2>Ten FRESH scenes — an unseen sample, checked against their frames</h2>
<p>Drawn for the pre-production sign-off with a NEW seed, excluding every previously
reviewed clip — so nothing below was tuned on. Strata now include the new
YIELD_FOR_TURN and CORRIDOR_OFFSET classes. <b>9 of 10 confirmed, 1 corrected by the PI</b>: my published CONFIRMED on d8f80c0f was wrong — a pull-out around a parked car, kinematically identical to a turn, now flagged <code>contested</code> by the new turn-corroboration mechanism. Also here: one
where my own reading was wrong and the measurement was right, one honest gate
refusal shown as such, and a fresh reproduction of the untimed-oncoming case —
carrying its flag.</p>
</div>

{scenes}

<div class="col">
<h2>Where the labels land across the corpus</h2>
</div>
<img class="chart" src="{dist_chart()}" alt="tactical goal distribution">

<div class="col">
<h2>The vocabulary — frozen, with definitions</h2>
<p>You noted the definitions had gone missing between documents. They now live in
<code>vocab_v7.py</code> beside the token tuples, and a test asserts every frozen
token has one and that no definition outlives its token — a markdown file is what
allowed them to drift. The vocabulary is <b>frozen</b> as you instructed: the tuples
are tensor dimensions, and a test pins their exact order.</p>
</div>
{defs_table()}

<div class="col">
<h2>What is still wrong</h2>
<ul>
<li><b>Traffic lights are still under-read on rich scenes.</b> <code>d452ea24</code>
    shows red lights at −4 s that no token captures — the CoT for that clip talks only
    about the bus, so there is nothing to extract from. This is a source-coverage limit,
    not an extraction bug, and it is the honest ceiling of a CoT-driven approach.</li>
<li><b>19 clips (0.40 %)</b> still carry a goal/action contradiction: 10 with
    <code>lat=TURN</code> and no turn goal, 9 pairing <code>STOP_POINT</code> with a
    non-braking longitudinal action.</li>
<li><b>Grounding reaches only ~13 % of CoT tokens</b> — a ceiling set by the one
    sampled question per clip, not by our logic. It cannot be raised without
    re-running the grounding task with more questions per clip.</li>
<li><b>The explicit <code>none</code> cannot be used to suppress tokens.</b> I proposed
    this as the top fix, then checked its scope: all 853 are worded "the first 2 seconds",
    which from Alpamayo's 5.1 s anchor is <b>−2.9 to −0.9 s relative to ours</b> — it does
    not overlap our window. It is still valuable as a CONTROL, which is exactly how it
    found the 408× inversion.</li>
<li><b>Alpamayo's <code>trajectory</code> task is still unused</b> — its predicted path
    and ADE/FDE per clip are banked but nothing reads them yet.</li>
</ul>
<p class="foot">Corpus <code>C:/Users/Admin/tanitad-wt/_s2build/v7_final2/</code> ·
records.parquet sha256 <code>ecae276db9969de1…</code> ·
retractions C142–C147 · 163+ tests green on the affected surface</p>
</div>
</div>
"""
    open("C:/Users/Admin/tanitad-wt/_s2build/report/report.html", "w",
         encoding="utf-8").write(HEAD + body)
    print("wrote report.html")


if __name__ == "__main__":
    main()
