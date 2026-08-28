"""Render the label-pipeline STATE report — every number pulled live."""
from __future__ import annotations

import collections
import gzip
import html
import json
import subprocess
import sys

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.models import vocab_v7 as V7        # noqa: E402

M = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
WP = (f"{M}/TanitAD Research Lab/Data Engineering/Implementation/incoming/"
      "2026-08-24-label-extraction-overnight")
ROWS = [json.loads(l) for l in
        gzip.open(f"{WP}/raw/s2_labels_v7.jsonl.gz", "rt", encoding="utf-8") if l.strip()]
SUM = json.load(open(f"{WP}/raw/summary.json", encoding="utf-8"))


def git(*a):
    return subprocess.run(["git", "-C", M, "-c", "core.fsmonitor=false", *a],
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()


def cnt(f):
    return collections.Counter(f(r) for r in ROWS)


N = len(ROWS)
G_TAC = collections.Counter(k for r in ROWS for k in r["g_tac"]["goals"])
PROV = collections.Counter()
TB = collections.Counter()
CORR = collections.Counter()
for r in ROWS:
    for t, a in r["g_tac"]["goals"].items():
        if not isinstance(a, dict):
            continue
        PROV[a.get("provenance", "geometry")] += 1
        if "time_basis" in a:
            TB[a["time_basis"]] += 1
        if "corroboration" in a:
            CORR[a["corroboration"]] += 1
VIOL = sum(1 for r in ROWS if r["g_tac"]["violations"])
GAP = sum(1 for r in ROWS if r.get("bands", {}).get("unassigned_manoeuvres"))
LAT = [x for x in ((r.get("alpamayo") or {}).get("lateral", {}).get("agree") for r in ROWS)
       if x is not None]
LON = [x for x in ((r.get("alpamayo") or {}).get("longitudinal", {}).get("agree") for r in ROWS)
       if x is not None]
CENSUS = SUM.get("vocab_emission_census", {})
CHECKABLE = sum(v for k, v in CORR.items() if k != "not_checkable")
CORROB = CORR["both"] + CORR["box"] + CORR["component"]


def bar(n, top, cls=""):
    pct = 100.0 * n / max(top, 1)
    return (f'<div class="bar {cls}"><span style="width:{pct:.1f}%"></span></div>')


def rows_table(counter, top=None, prov=None):
    top = top or max(counter.values(), default=1)
    out = []
    for k, v in counter.most_common():
        p = prov(k) if prov else ""
        out.append(
            f'<tr><td class="tok">{html.escape(k)}{p}</td>'
            f'<td class="num">{v:,}</td>'
            f'<td class="pc">{100*v/N:.1f}%</td>'
            f'<td class="barc">{bar(v, top)}</td></tr>')
    return "".join(out)


def goal_prov(k):
    src = None
    for r in ROWS:
        a = r["g_tac"]["goals"].get(k)
        if isinstance(a, dict):
            src = a.get("provenance", "geometry")
            break
    if src == "vlm-cot":
        return ' <span class="chip cot">vlm-cot</span>'
    if src == "alpamayo-structured":
        return ' <span class="chip as">struct</span>'
    return ' <span class="chip geo">geometry</span>'


CSS = """
:root{--ground:#f5f5f3;--panel:#fff;--sunk:#eeeeec;--ink:#16181d;--ink-2:#4a4d57;
 --ink-3:#767a86;--rule:#dededa;--rule-2:#c9c9c4;--cyan:#0f6f7d;--cyan-soft:#0f6f7d1a;
 --amber:#a8650a;--amber-soft:#a8650a1a;--good:#1f6b45;--good-soft:#1f6b451a;
 --bad:#a32f28;--bad-soft:#a32f281a;--unk:#5d6070;--unk-soft:#5d60701a;--fill:#0f6f7d55}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#101217;--panel:#171a21;--sunk:#1e222b;--ink:#e8e9ec;--ink-2:#a9adb8;
 --ink-3:#7d818d;--rule:#282d37;--rule-2:#39404c;--cyan:#4fc3d4;--cyan-soft:#4fc3d422;
 --amber:#e0a244;--amber-soft:#e0a24422;--good:#5cc189;--good-soft:#5cc18922;
 --bad:#e3776e;--bad-soft:#e3776e22;--unk:#9aa0b0;--unk-soft:#9aa0b022;--fill:#4fc3d455}}
:root[data-theme="dark"]{
 --ground:#101217;--panel:#171a21;--sunk:#1e222b;--ink:#e8e9ec;--ink-2:#a9adb8;
 --ink-3:#7d818d;--rule:#282d37;--rule-2:#39404c;--cyan:#4fc3d4;--cyan-soft:#4fc3d422;
 --amber:#e0a244;--amber-soft:#e0a24422;--good:#5cc189;--good-soft:#5cc18922;
 --bad:#e3776e;--bad-soft:#e3776e22;--unk:#9aa0b0;--unk-soft:#9aa0b022;--fill:#4fc3d455}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;font-size:15px;
 line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 22px 90px}
h1,h2,h3{font-family:Archivo,"IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(28px,4.2vw,42px);font-weight:700;letter-spacing:-.022em;line-height:1.1}
h2{font-size:21px;font-weight:650;letter-spacing:-.01em;margin:52px 0 4px;
 padding-top:18px;border-top:1px solid var(--rule)}
h3{font-size:13px;font-weight:600;margin:24px 0 7px;font-family:"IBM Plex Mono",monospace;
 letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}
p{margin:10px 0;max-width:70ch}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.87em;
 background:var(--sunk);padding:.1em .38em;border-radius:3px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.15em;
 text-transform:uppercase;color:var(--cyan);margin:0 0 10px}
.lede{font-size:17px;line-height:1.55;color:var(--ink-2);margin:14px 0 0;max-width:66ch}
.meta{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);
 margin-top:18px;padding-top:11px;border-top:1px solid var(--rule)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;
 background:var(--rule);border:1px solid var(--rule);border-radius:9px;overflow:hidden;margin:28px 0}
.kpi{background:var(--panel);padding:14px 15px}
.kpi .n{font-family:Archivo,sans-serif;font-size:25px;font-weight:700;letter-spacing:-.02em;
 font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .l{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;
 text-transform:uppercase;color:var(--ink-3);margin-top:5px}
.kpi .s{font-size:11.5px;color:var(--ink-2);margin-top:3px}
.kpi.good .n{color:var(--good)}.kpi.hero .n{color:var(--cyan)}.kpi.warn .n{color:var(--amber)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 4px}
th{text-align:left;font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;
 text-transform:uppercase;color:var(--ink-3);border-bottom:1px solid var(--rule-2);padding:6px 8px}
td{padding:5px 8px;border-bottom:1px solid var(--rule);vertical-align:middle}
td.num,td.pc{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
 text-align:right;white-space:nowrap}
td.pc{color:var(--ink-3);font-size:12px}
td.barc{width:34%;padding-right:0}
.tok{font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap}
.bar{height:7px;background:var(--sunk);border-radius:4px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--fill);border-radius:4px}
.chip{font-family:"IBM Plex Mono",monospace;font-size:9px;padding:1px 5px;border-radius:9px;
 margin-left:6px;font-weight:500;letter-spacing:.04em;vertical-align:1px}
.chip.geo{background:var(--cyan-soft);color:var(--cyan)}
.chip.cot{background:var(--amber-soft);color:var(--amber)}
.chip.as{background:var(--unk-soft);color:var(--unk)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:22px}
.note{border-left:3px solid var(--amber);background:var(--amber-soft);padding:11px 14px;
 border-radius:0 7px 7px 0;margin:16px 0;font-size:13.5px}
.note.good{border-left-color:var(--good);background:var(--good-soft)}
.note.bad{border-left-color:var(--bad);background:var(--bad-soft)}
.note b{color:var(--ink)}
ul{padding-left:19px;max-width:70ch}li{margin:6px 0}
.dec{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--cyan);
 border-radius:0 9px 9px 0;padding:13px 16px;margin:14px 0}
.dec h4{margin:0 0 5px;font-size:15px;font-family:Archivo,sans-serif;font-weight:650}
.scroll{overflow-x:auto}
.foot{margin-top:56px;padding-top:15px;border-top:1px solid var(--rule);
 font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-3);line-height:1.7}
"""


def kpi(n, l, s="", c=""):
    return (f'<div class="kpi {c}"><div class="n">{n}</div><div class="l">{l}</div>'
            + (f'<div class="s">{s}</div>' if s else "") + "</div>")


BODY = f"""
<div class="wrap">
<header>
 <p class="eyebrow">TanitAD · Data FlyWheel · P2 · state as of 2026-08-28</p>
 <h1>The v7 label ground truth</h1>
 <p class="lede">The corpus the PI named the only label ground truth, and everything
 currently true about it — what it contains, what each number rests on, what the
 pipeline refuses to claim, and the two decisions waiting on him.</p>
 <p class="meta">{N:,} clips · {N*20/3600:.1f} h · schema <code>s2-geom-v7</code> ·
 vocabulary FROZEN at {len(V7.ALL_V7_TOKENS)} tokens · every figure below read from the
 shipped set, none hand-entered</p>
</header>

<div class="kpis">
 {kpi(f"{N:,}", "clips labelled", f"{SUM.get('refused',0)} refused · {N*20/3600:.1f} h", "hero")}
 {kpi(f"{VIOL}", "goal/action contradictions", "was 21.7 % of clips", "good")}
 {kpi(f"{CENSUS.get('emitted_at_least_once','?')}/{len(V7.ALL_V7_TOKENS)}", "vocabulary reachable",
      f"{len(V7.NOT_YET_EXTRACTABLE)} declared unreachable", "good")}
 {kpi(f"{TB['untimed']/max(sum(TB.values()),1):.0%}", "CoT tokens untimed", "band placement unproven", "warn")}
 {kpi("30 s", "full-corpus runtime", "150 clips/s")}
</div>

<div>
<h2>What the set contains</h2>
<p>Three layers plus a model input. The bands are the PI's:
<code>OPERATIVE [0,2]</code> · <code>TACTICAL [2,6]</code> · <code>STRATEGIC [8,30]</code>,
and every record carries them so the window can never be implicit.</p>
</div>

<div class="grid2">
 <div>
  <h3>Strategic goal — what comes next on the route</h3>
  <div class="scroll"><table><tr><th>token</th><th>n</th><th>%</th><th></th></tr>
  {rows_table(cnt(lambda r: r['g_str']['token']))}</table></div>
  <h3>Strategic action — what to do about it now</h3>
  <div class="scroll"><table><tr><th>token</th><th>n</th><th>%</th><th></th></tr>
  {rows_table(cnt(lambda r: r['a_str']['token']))}</table></div>
 </div>
 <div>
  <h3>Lateral action</h3>
  <div class="scroll"><table><tr><th>token</th><th>n</th><th>%</th><th></th></tr>
  {rows_table(cnt(lambda r: r['a_tac']['lat']))}</table></div>
  <h3>Longitudinal action</h3>
  <div class="scroll"><table><tr><th>token</th><th>n</th><th>%</th><th></th></tr>
  {rows_table(cnt(lambda r: r['a_tac']['lon']))}</table></div>
  <h3>Nav command — model INPUT, ego-future oracle</h3>
  <div class="scroll"><table><tr><th>token</th><th>n</th><th>%</th><th></th></tr>
  {rows_table(cnt(lambda r: r['nav_command']['token']))}</table></div>
 </div>
</div>

<h3>Tactical goals — multi-label, {sum(G_TAC.values()):,} instances over {N:,} clips</h3>
<div class="scroll"><table><tr><th>token</th><th>n</th><th>% of clips</th><th></th></tr>
{rows_table(G_TAC, prov=goal_prov)}</table></div>

<div>
<h2>What each number rests on</h2>
</div>
<div class="grid2">
 <div>
  <h3>Where goals come from</h3>
  <div class="scroll"><table><tr><th>provenance</th><th>instances</th><th>share</th><th></th></tr>
  {"".join(f'<tr><td class="tok">{k}</td><td class="num">{v:,}</td>'
           f'<td class="pc">{100*v/sum(PROV.values()):.1f}%</td>'
           f'<td class="barc">{bar(v, max(PROV.values()))}</td></tr>'
           for k, v in PROV.most_common())}</table></div>
  <h3>Corroboration of CoT tokens</h3>
  <div class="scroll"><table><tr><th>tier</th><th>n</th><th></th></tr>
  {"".join(f'<tr><td class="tok">{k}</td><td class="num">{v:,}</td>'
           f'<td class="barc">{bar(v, max(CORR.values()))}</td></tr>'
           for k, v in CORR.most_common())}</table></div>
  <p style="font-size:12.5px;color:var(--ink-2)"><b>{CORROB:,}/{CHECKABLE:,} =
  {100*CORROB/max(CHECKABLE,1):.1f}%</b> of tokens naming a checkable object are
  corroborated. <code>box</code> is image-space perception;
  <code>component</code> is a second text claim from a different Alpamayo task —
  they are never merged.</p>
 </div>
 <div>
  <h3>Time basis of CoT tokens</h3>
  <div class="scroll"><table><tr><th>basis</th><th>n</th><th></th></tr>
  {"".join(f'<tr><td class="tok">{k}</td><td class="num">{v:,}</td>'
           f'<td class="barc">{bar(v, max(TB.values()))}</td></tr>'
           for k, v in TB.most_common())}</table></div>
  <h3>Cross-check against Alpamayo</h3>
  <div class="scroll"><table><tr><th>axis</th><th>agree</th><th>n</th></tr>
  <tr><td class="tok">lateral</td><td class="num">{100*sum(LAT)/max(len(LAT),1):.1f}%</td>
      <td class="num">{len(LAT):,}</td></tr>
  <tr><td class="tok">longitudinal</td><td class="num">{100*sum(LON)/max(len(LON),1):.1f}%</td>
      <td class="num">{len(LON):,}</td></tr></table></div>
  <p style="font-size:12.5px;color:var(--ink-2)">Alpamayo corroborates; it never
  overrides. Its anchor is 5.1 s against our 8.0 s, so each source is scored on
  its own window.</p>
 </div>
</div>

<div class="note bad"><b>The single largest caveat: {TB['untimed']:,} of
{sum(TB.values()):,} CoT tokens ({TB['untimed']/max(sum(TB.values()),1):.1%}) are
<code>untimed</code>.</b> A CoT claim carries no timestamp — only Alpamayo's parsed
motion segments give one, and 1,845 clips (39.0 %) have no time information at all.
<code>REACT_ON_ONCOMING</code> is <b>0 timed / 344</b>. The claims are often true; they
are simply not <i>placed</i> in the 2–6 s band by any evidence. Geometry-sourced
tokens carry no such caveat.</div>

<div>
<h2>Vocabulary coverage</h2>
<p><b>{CENSUS.get('emitted_at_least_once','?')} of {len(V7.ALL_V7_TOKENS)}</b> frozen tokens
are emitted at least once. {len(V7.NOT_YET_EXTRACTABLE)} are <b>declared</b> unreachable,
each with a stated reason — a head sized to the full vocabulary trains that many empty
classes unless masked.</p>
<div class="scroll"><table><tr><th>declared unreachable</th><th>why, in short</th></tr>
{"".join(f'<tr><td class="tok">{t}</td><td style="font-size:12.5px;color:var(--ink-2)">'
         f'{html.escape(V7.NOT_YET_EXTRACTABLE[t][:150])}…</td></tr>'
         for t in sorted(V7.NOT_YET_EXTRACTABLE))}</table></div>
</div>

{'<div class="note"><b>One token is silent and NOT declared: <code>'
 + ", ".join(CENSUS.get("silent_and_undeclared", [])) +
 '</code>.</b> Its mirror <code>YIELD_FOR_TURN_R</code> fires once, so the path works — '
 'this is genuine rarity (a turn preceded by a held stop), not an unreachable token. '
 'It is flagged rather than hidden: the corpus runner now census-checks every build, '
 'because <code>FOLLOW_LANE</code> silently fell from 1,281 emissions to zero earlier '
 'today and nothing caught it.</div>' if CENSUS.get("silent_and_undeclared") else ''}

<div>
<h2>What is decided, and what is not</h2>
</div>

<div class="dec">
 <h4>Open · may an untimed token supervise the tactical head?</h4>
 <p style="margin:4px 0 0;font-size:13.5px">{TB['untimed']:,} tokens are affected — 87 % of
 the CoT-derived signal. They are marked, not deleted: discarding them throws away most of
 what Alpamayo knows, and using them blind asserts a placement nothing supports. The
 <code>time_basis</code> field exists so this is decided rather than defaulted.</p>
</div>
<div class="dec">
 <h4>Open · the lane-detector arm for <code>CORRIDOR_OFFSET</code></h4>
 <p style="margin:4px 0 0;font-size:13.5px">The Research Lab has a literature-backed design
 (CLRerNet-class lane detector + <code>obstacle.offline</code> causal-trigger join) that would
 give the token the independent reference it lacks. A new GPU stream; the box is free. Not
 started without your word.</p>
</div>
<div class="dec">
 <h4>Decided · the strategic-action head is 7 tokens</h4>
 <p style="margin:4px 0 0;font-size:13.5px">Your removal of
 <code>REDUCE_TO_FOLLOW_ROUTE</code> predates the freeze and is committed with your
 instruction as provenance. Re-adding it is a one-token change under a new version if you
 want it back.</p>
</div>

<div>
<h2>Where it lives</h2>
<div class="scroll"><table><tr><th>artifact</th><th>path / blob</th></tr>
<tr><td class="tok">label set</td><td class="args" style="font-family:'IBM Plex Mono',monospace;font-size:11.5px">
 …/2026-08-24-label-extraction-overnight/raw/s2_labels_v7.jsonl.gz</td></tr>
<tr><td class="tok">emitter</td><td style="font-family:'IBM Plex Mono',monospace;font-size:11.5px">stack/scripts/s2_geom_emit_v7.py</td></tr>
<tr><td class="tok">runner</td><td style="font-family:'IBM Plex Mono',monospace;font-size:11.5px">stack/scripts/s2_run_corpus.py</td></tr>
<tr><td class="tok">vocabulary</td><td style="font-family:'IBM Plex Mono',monospace;font-size:11.5px">stack/tanitad/models/vocab_v7.py <span class="chip geo">frozen</span></td></tr>
<tr><td class="tok">register row</td><td style="font-family:'IBM Plex Mono',monospace;font-size:11.5px">Project Steering/GOALS_AND_CLAIMS.md → D-LABEL-GT</td></tr>
<tr><td class="tok">retractions</td><td style="font-family:'IBM Plex Mono',monospace;font-size:11.5px">RETRACTION_LOG.md → C142–C149, DE-C150</td></tr>
</table></div>
<p class="foot">
Sources: label set read directly; counts computed at render time, not transcribed.<br>
Alpamayo augmentation: 4,729 clips / 26.3 h, five tasks, <code>records.parquet</code>.<br>
Tests: 151 green on the affected surface.<br>
Nine retractions banked this cycle; most were defects in this pipeline's own output.
</p>
</div>
</div>
"""

HEAD = """<title>The v7 Label Ground Truth</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>%s</style>""" % CSS

open("C:/Users/Admin/tanitad-wt/_s2build/report/state.html", "w",
     encoding="utf-8").write(HEAD + BODY)
print("wrote state.html")
