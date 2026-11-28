"""Render the v7 vocabulary definition, including the goal->action matrix."""
from __future__ import annotations

import json
from pathlib import Path

D = Path("C:/Users/Admin/tanitad-wt/_s2build/design")
M = json.loads((D / "v7_matrix.json").read_text(encoding="utf-8"))
OUT = D / "tac_vocab_v7.html"

LAT = M["lat_actions"]
LON = M["lon_actions"]
GOALS = [g for g in M["lat_map"]]


def matrix(kind: str) -> str:
    acts = LAT if kind == "lat" else LON
    amap = M["lat_map"] if kind == "lat" else M["lon_map"]
    head = "".join(f'<th class="rot"><span>{a}</span></th>' for a in acts)
    body = ""
    for g in GOALS:
        cnt = M["goal_counts"].get(g, 0)
        cells = "".join(
            f'<td class="{"yes" if a in amap.get(g, ()) else "no"}">'
            f'{"●" if a in amap.get(g, ()) else ""}</td>' for a in acts)
        body += (f'<tr><td class="gname">{g}'
                 f'<span class="gn">{cnt or "—"}</span></td>{cells}</tr>')
    return (f'<div class="tw"><table class="mx"><thead><tr>'
            f'<th class="corner">goal ↓ / action →</th>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def cooc() -> str:
    mx = max(p["n"] for p in M["pairs"]) if M["pairs"] else 1
    rows = ""
    for p in M["pairs"]:
        w = 100 * p["n"] / mx
        rows += (f'<div class="cor"><span class="ca mono">{p["a"]}</span>'
                 f'<span class="cp">+</span>'
                 f'<span class="cb mono">{p["b"]}</span>'
                 f'<span class="cbar"><span style="width:{w:.0f}%"></span></span>'
                 f'<span class="cn mono">{p["n"]}</span></div>')
    return rows


def sizes() -> str:
    tot = sum(M["set_sizes"].values())
    out = ""
    for k in sorted(M["set_sizes"], key=int):
        v = M["set_sizes"][k]
        out += (f'<div class="szr"><span class="szk mono">{k} goal'
                f'{"s" if int(k) > 1 else ""}</span>'
                f'<span class="szbar"><span style="width:{100*v/tot:.1f}%"></span></span>'
                f'<span class="szn mono">{v} · {100*v/tot:.1f}%</span></div>')
    return out


def excl() -> str:
    return "".join(f'<span class="ex mono">{a} ⊕ {b}</span>'
                   for a, b in M["exclusive"])


def main() -> None:
    c = M["consistency"]
    doc = TEMPLATE
    for k, v in (("@@LAT_MATRIX@@", matrix("lat")),
                 ("@@LON_MATRIX@@", matrix("lon")),
                 ("@@COOC@@", cooc()), ("@@SIZES@@", sizes()),
                 ("@@EXCL@@", excl()),
                 ("@@LATPCT@@", f'{100*c["lat"]/c["n"]:.1f}'),
                 ("@@LONPCT@@", f'{100*c["lon"]/c["n"]:.1f}'),
                 ("@@NEXCL@@", str(len(M["exclusive"])))):
        assert k in doc, k
        doc = doc.replace(k, v)
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


TEMPLATE = r'''<title>Hierarchy Vocabulary v7</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
 --ground:#F5F7FA;--panel:#FFF;--panel-2:#EDF1F6;
 --ink:#0E1620;--ink-2:#43525F;--ink-3:#70818F;
 --line:#D6E0E9;--line-2:#BECBD8;
 --accent:#0057D9;--accent-soft:#E2ECFC;
 --band:#7A3DBD;--band-soft:#F0E7FA;
 --ok:#0F7A50;--ok-bg:#E2F3EA;--flag:#B4400C;--flag-bg:#FBE8DF;
 --shadow:0 1px 2px rgba(14,22,32,.06),0 10px 28px -14px rgba(14,22,32,.2);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0A0F14;--panel:#121A22;--panel-2:#19232D;
 --ink:#E6EDF4;--ink-2:#A3B3C1;--ink-3:#798A98;
 --line:#212D37;--line-2:#2F3E4B;
 --accent:#5C9BFF;--accent-soft:#14233A;
 --band:#B489EE;--band-soft:#1D1530;
 --ok:#4FC08A;--ok-bg:#11281E;--flag:#FF8A5B;--flag-bg:#2D1710;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -16px rgba(0,0,0,.75);
}}
:root[data-theme="dark"]{
 --ground:#0A0F14;--panel:#121A22;--panel-2:#19232D;
 --ink:#E6EDF4;--ink-2:#A3B3C1;--ink-3:#798A98;
 --line:#212D37;--line-2:#2F3E4B;
 --accent:#5C9BFF;--accent-soft:#14233A;
 --band:#B489EE;--band-soft:#1D1530;
 --ok:#4FC08A;--ok-bg:#11281E;--flag:#FF8A5B;--flag-bg:#2D1710;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -16px rgba(0,0,0,.75);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"Source Sans 3",ui-sans-serif,system-ui,sans-serif;font-size:16px;
 line-height:1.62;-webkit-font-smoothing:antialiased}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;
 font-variant-numeric:tabular-nums}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 96px}
h1,h2,h3{font-family:Archivo,ui-sans-serif,system-ui,sans-serif;text-wrap:balance;
 margin:0;line-height:1.15;letter-spacing:-.012em}
h1{font-size:clamp(30px,4.2vw,44px);font-weight:700}
h2{font-size:24px;font-weight:600}
p{margin:0}
.eyebrow{font-family:Archivo,sans-serif;font-size:11.5px;font-weight:600;
 letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
.lede{max-width:74ch;color:var(--ink-2);margin-top:14px;font-size:17px}
header.top{border-bottom:1px solid var(--line);padding-bottom:30px;
 display:flex;flex-direction:column;gap:6px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.mtag{font-size:12.5px;padding:4px 10px;border:1px solid var(--line-2);
 border-radius:2px;color:var(--ink-2);background:var(--panel)}
.mtag.good{border-color:var(--ok);color:var(--ok);background:var(--ok-bg)}
.mtag.warn{border-color:var(--flag);color:var(--flag);background:var(--flag-bg)}
section{margin-top:46px}
.sechead{display:flex;flex-direction:column;gap:6px;margin-bottom:18px}
.sechead p{color:var(--ink-2);max-width:82ch}
.box{background:var(--panel);border:1px solid var(--line);border-radius:3px;
 padding:20px 22px;box-shadow:var(--shadow)}
.box h3{font-size:12px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--ink-3);margin-bottom:12px;font-weight:600}
.box p{color:var(--ink-2);font-size:14.7px}.box p+p{margin-top:9px}
.box b{color:var(--ink)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
table{width:100%;border-collapse:collapse;font-size:14.2px}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
 border-radius:3px;box-shadow:var(--shadow)}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);
 vertical-align:top}
th{font-family:Archivo,sans-serif;font-size:10.5px;text-transform:uppercase;
 letter-spacing:.08em;color:var(--ink-3);font-weight:600;white-space:nowrap}
tr:last-child td{border-bottom:none}
td.n{font-family:"JetBrains Mono",monospace;white-space:nowrap}
/* goal x action matrix */
table.mx{font-size:12.5px}
table.mx th.corner{font-size:10px;color:var(--ink-3);white-space:nowrap;
 border-right:1px solid var(--line)}
table.mx th.rot{height:112px;padding:0;width:34px;vertical-align:bottom;
 border-bottom:1px solid var(--line)}
table.mx th.rot span{writing-mode:vertical-rl;transform:rotate(180deg);
 font-family:"JetBrains Mono",monospace;font-size:10.5px;font-weight:600;
 color:var(--ink-2);letter-spacing:.02em;padding:6px 0;text-transform:none}
table.mx td{padding:5px 6px;text-align:center}
table.mx td.gname{font-family:"JetBrains Mono",monospace;font-size:11.5px;
 text-align:left;white-space:nowrap;color:var(--ink);
 border-right:1px solid var(--line);font-weight:600}
table.mx td.gname .gn{color:var(--ink-3);font-weight:400;margin-left:8px}
table.mx td.yes{color:var(--ok);background:var(--ok-bg);font-size:13px}
table.mx td.no{color:var(--line-2)}
/* co-occurrence */
.cor{display:grid;grid-template-columns:1fr 14px 1fr 130px 44px;gap:8px;
 align-items:center;padding:5px 0;border-bottom:1px solid var(--line)}
.cor:last-child{border-bottom:none}
.ca,.cb{font-size:11.5px;color:var(--band);background:var(--band-soft);
 border:1px solid var(--band);padding:2px 7px;border-radius:2px;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cp{color:var(--ink-3);text-align:center}
.cbar{background:var(--panel-2);height:8px;border-radius:1px;overflow:hidden}
.cbar span{display:block;height:100%;background:var(--band)}
.cn{font-size:12px;color:var(--ink-3);text-align:right}
.szr{display:grid;grid-template-columns:90px 1fr 120px;gap:10px;align-items:center;
 padding:5px 0}
.szk{font-size:12.5px;color:var(--ink-2)}
.szbar{background:var(--panel-2);height:10px;border-radius:1px;overflow:hidden}
.szbar span{display:block;height:100%;background:var(--accent)}
.szn{font-size:12px;color:var(--ink-3);text-align:right}
.ex{font-size:10.5px;color:var(--flag);background:var(--flag-bg);
 border:1px solid var(--flag);padding:2px 6px;border-radius:2px;
 margin:0 4px 4px 0;display:inline-block}
.tok{font-family:"JetBrains Mono",monospace;font-size:12.4px;font-weight:600;
 padding:2px 8px;border-radius:2px;white-space:nowrap;display:inline-block;
 margin:1px 2px 1px 0;border:1px solid var(--line-2);background:var(--panel-2)}
.tok.str{color:var(--accent);border-color:var(--accent);background:var(--accent-soft)}
.tok.tac{color:var(--band);border-color:var(--band);background:var(--band-soft)}
.tok.new{color:var(--ok);border-color:var(--ok);background:var(--ok-bg)}
.tok.perc{color:var(--flag);border-color:var(--flag);background:var(--flag-bg)}
.arg{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--ink-2);
 background:var(--panel-2);border:1px dashed var(--line-2);padding:1px 6px;
 border-radius:2px;margin:0 3px 3px 0;display:inline-block}
.alert{background:var(--flag-bg);border:1px solid var(--flag);border-left-width:3px;
 border-radius:3px;padding:20px 22px}
.alert h3{font-size:17px;margin-bottom:10px;color:var(--ink)}
.alert p{color:var(--ink-2);max-width:84ch}.alert p+p{margin-top:9px}
.alert b{color:var(--ink)}
.bign{font-family:Archivo,sans-serif;font-size:34px;font-weight:700;
 font-variant-numeric:tabular-nums;line-height:1;letter-spacing:-.02em}
.bign.good{color:var(--ok)}
footer{margin-top:46px;padding-top:22px;border-top:1px solid var(--line);
 color:var(--ink-3);font-size:13px;max-width:88ch}
@media (max-width:900px){.grid2{grid-template-columns:1fr}
 .cor{grid-template-columns:1fr 14px 1fr 60px 40px}}
</style>
<div class="wrap">
<header class="top">
 <p class="eyebrow">TanitAD · DataFlyWheel · 2026-08-23 · IMPLEMENTED</p>
 <h1>Hierarchy Vocabulary v7</h1>
 <p class="lede">Layers separated by <b>position in the manoeuvre sequence</b>. The tactical
  goal is a <b>set</b>; the strategic goal is the overnext manoeuvre, to follow the route.
  Every goal is linked to the actions that may serve it, and the link is checked per clip.</p>
 <div class="meta">
  <span class="mtag good">801/801 emitted · 0 violations</span>
  <span class="mtag good">goal↔action lat @@LATPCT@@% · lon @@LONPCT@@%</span>
  <span class="mtag mono">189 tests pass</span>
  <span class="mtag warn">nav command is ORACLE</span>
 </div>
</header>

<section>
 <div class="sechead"><p class="eyebrow">§1 — what you asked, and what the corpus said</p>
  <h2>Your ten items</h2></div>
 <div class="tw"><table>
  <thead><tr><th>Item</th><th>Done</th><th>Measured</th></tr></thead>
  <tbody>
  <tr><td><b>WAIT_FOR_ONCOMING → react on oncoming</b></td>
   <td><span class="tok new">REACT_ON_ONCOMING</span>, fires on any “oncoming”</td>
   <td class="n">0 → <b>142</b> (3.0%)</td></tr>
  <tr><td><b>Yield from the CoT</b></td><td><span class="tok tac">YIELD</span> with
   <span class="arg">reason</span> sign|hazard</td><td class="n"><b>400</b> (8.5%)</td></tr>
  <tr><td><b>Exit from terms, not geometry</b></td>
   <td><span class="tok new">TAKE_EXIT_L</span> <span class="tok new">TAKE_EXIT_R</span></td>
   <td class="n">54 (exit 17 + ramp 26)</td></tr>
  <tr><td><b>Merge in Alpamayo?</b> — yes</td><td><span class="tok new">MERGE</span> +
   <span class="mono">YIELD_MERGE</span> action</td><td class="n"><b>45</b> (1.0%)</td></tr>
  <tr><td><b>Target speed band restored</b></td><td><span class="tok tac">SPEED_BAND</span>
   from geometry</td><td class="n">256 clips</td></tr>
  <tr><td><b>Overtake vs evade, precisely</b></td>
   <td>moving obstacle ⇒ overtake; static/VRU ⇒ evade</td>
   <td class="n">45 / <b>574</b></td></tr>
  <tr><td><b>ACCELERATE action</b></td><td><span class="tok new">ACCELERATE</span></td>
   <td class="n">189 clips</td></tr>
  <tr><td><b>Nav commands, kept simple</b></td>
   <td>next manoeuvre, time-independent</td><td class="n">339 turn / 462 follow</td></tr>
  <tr><td><b>Goal↔action documentation</b></td><td>§3 matrix, checked per clip</td>
   <td class="n">lat @@LATPCT@@% lon @@LONPCT@@%</td></tr>
  <tr><td><b>Parallel goals visualised</b></td><td>§4</td><td class="n">68 clips ≥2 goals</td></tr>
  </tbody></table></div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">§2 — what REDUCE_TO_FOLLOW_ROUTE means</p>
  <h2>You asked, and it had no definition</h2></div>
 <div class="box">
  <p>In the first v7 run it was emitted <b>0 times</b> — a token inherited from v6 with no
   meaning in the ordinal design, because <span class="mono">PREPARE_TURN</span> and
   <span class="mono">PREPARE_STOP</span> already cover “slow down for the overnext
   manoeuvre”.</p>
  <p><b>It now has one distinct meaning:</b> a <b>sustained</b> speed drop across the
   strategic horizon that <i>no manoeuvre explains</i> — the ego entering a slower road class
   (town, zone) and staying slower. That is a route-level fact, so it earns its place;
   anything shorter is tactical.</p>
  <p>Definition: mean speed over the last 10 s of the lookahead is ≥3 m/s below the first 6 s,
   and its maximum stays ≥1 m/s below that. <b>Emitted on 90 clips.</b> If you would rather
   drop the token than keep this meaning, say so — it is the one token here whose definition
   I chose rather than derived.</p>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">§3 — goal ↔ action</p>
  <h2>Which actions may serve which goal</h2>
  <p>A dot means the action can further that goal. An emitted action serving <b>none</b> of its
   goals is the <span class="mono">TURN_LEFT</span> + <span class="mono">HOLD_CORRIDOR</span>
   shape you found — so this is checked on every clip and written into the label.</p></div>
 <h3 style="font-size:13px;color:var(--ink-3);margin-bottom:8px;letter-spacing:.08em;
  text-transform:uppercase">Lateral</h3>
 @@LAT_MATRIX@@
 <h3 style="font-size:13px;color:var(--ink-3);margin:20px 0 8px;letter-spacing:.08em;
  text-transform:uppercase">Longitudinal</h3>
 @@LON_MATRIX@@
 <div class="alert" style="margin-top:16px">
  <h3>The matrix found two incoherences of mine within minutes</h3>
  <p><b>1. <span class="mono">FOLLOW_LANE</span> + <span class="mono">BRAKE_TO</span> — 69
   clips.</b> My map omitted braking from lane-following. Braking <i>while</i> following a lane
   is ordinary (traffic ahead); the map was describing an idealised taxonomy rather than
   driving. Fixed.</p>
  <p><b>2. <span class="mono">SPEED_BAND</span> + <span class="mono">ACCELERATE</span>/<span class="mono">BRAKE_TO</span>
   — 103 clips.</b> The goal allowed a 3.0 m/s spread while the CRUISE action required
   |dv| &lt; 1.0 — <b>two thresholds for one fact</b>, the same defect as the emitter/guard
   constants earlier today. The band now uses the action's own measure. Fixed.</p>
  <p>Longitudinal consistency <b>72.7% → @@LONPCT@@%</b>. The remaining 45 are mostly temporal
   ordering inside the 6 s window (the goal is later, the action is now) — e.g.
   <span class="mono">STOP_POINT</span>+<span class="mono">ACCELERATE</span> ×14, where the ego
   accelerates now and stops later. <b>Left visible rather than tuned away.</b></p>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">§4 — parallel goals</p>
  <h2>Which goals actually co-occur</h2>
  <p>@@NEXCL@@ pairs are declared mutually exclusive and checked; everything else may hold at
   once. <b>0 violations across 801 clips.</b></p></div>
 <div class="grid2">
  <div class="box"><h3>Goals held simultaneously</h3>@@SIZES@@
   <p style="margin-top:10px">Multi-goal clips are the minority — but they are the interesting
    ones, and a single-token head could not express any of them.</p></div>
  <div class="box"><h3>Mutually exclusive pairs</h3><div>@@EXCL@@</div></div>
 </div>
 <div class="box" style="margin-top:16px"><h3>Most frequent co-occurring pairs</h3>@@COOC@@</div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">§5 — nav commands</p>
  <h2>Simple, and oracle</h2></div>
 <div class="grid2">
  <div class="box"><h3>Exactly as specified</h3>
   <p><span class="tok new">NAV_TURN_L</span> <span class="tok new">NAV_TURN_R</span>
    <span class="tok new">NAV_FOLLOW_ROAD</span> ·
    <span class="arg">distance_m</span><span class="arg">time_s</span></p>
   <p>The <b>next manoeuvre, independent of time</b> — turn detected, else follow road. Same
    extraction as the manoeuvre sequence, so it cannot drift from the labels.</p>
   <p><b>Emitted:</b> NAV_FOLLOW_ROAD 462 · NAV_TURN_L 175 · NAV_TURN_R 164.</p>
  </div>
  <div class="box" style="border-left:3px solid var(--flag)"><h3>⛔ It is oracle here</h3>
   <p>Our only route supplier on PhysicalAI is <b>the ego's own recorded future</b>, so a nav
    command derived from it and fed back as an input <b>tells the model the answer</b>.</p>
   <p>Every command carries <span class="mono">provenance="ego-future"</span>,
    <span class="mono">oracle=true</span>. <span class="mono">nav-system</span> provenance is
    admissible at inference; <span class="mono">ego-future</span> is <b>training only</b>, and
    any eval using it may never be compared against a vision-only arm.</p>
  </div>
 </div>
</section>

<section>
 <div class="sechead"><p class="eyebrow">§6 — limits</p>
  <h2>What the corpus will not give you</h2></div>
 <div class="grid2">
  <div class="box"><h3>Open door — 0 clips</h3>
   <p>You named the open-door case for <span class="mono">EVADE_IN_CORRIDOR</span>. The term
    appears in <b>0 of 4,729</b> CoTs. The evade class is instead dominated by
    <b>parked vehicles (419)</b>, <b>pedestrians (278)</b> and <b>cyclists (64)</b>.</p>
   <p>The extractor accepts a door referent; the corpus simply has none.</p>
  </div>
  <div class="box"><h3>meta_action is not reachable</h3>
   <p>You asked whether Alpamayo's meta actions help. The raw records carry
    <span class="mono">answer|box|cot|cot_auto_labeling|meta_action|raw_outputs</span>, but only
    <b>cot, lane, lateral, longitudinal</b> were exported per clip — <b>meta_action is not
    available locally</b>. Using it needs the source parquet.</p>
   <p>Given the lane/lateral axes measured <b>at chance</b>, I would want meta_action tested the
    same way before trusting it.</p>
  </div>
 </div>
 <div class="box" style="margin-top:16px"><h3>⚠️ Representable ≠ scoreable</h3>
  <p>On the 801-clip labelled set the CoT-derived tokens are thin because only 257 clips carry
   Alpamayo: <span class="mono">MERGE</span> 1, <span class="mono">TAKE_EXIT_R</span> 1,
   <span class="mono">OVERTAKE_VEHICLE</span> 2, <span class="mono">GAP_TARGET</span> 5,
   <span class="mono">REACT_ON_ONCOMING</span> 6. All far below the n=200 floor ⇒ emit and
   supervise, but <b>refuse any per-class metric</b>.</p>
  <p>Corpus-wide (4,729 clips) the same tokens are far better populated — EVADE 574, YIELD 400,
   ONCOMING 142, OVERTAKE 45, MERGE 44 — which is another reason to extend the augmentation
   beyond aug120.</p>
 </div>
</section>

<footer><p><strong>Implementation.</strong>
 <span class="mono">stack/tanitad/models/vocab_v7.py</span> ·
 <span class="mono">stack/tanitad/data/cot_tokens_v7.py</span> ·
 <span class="mono">stack/scripts/s2_geom_emit_v7.py</span> ·
 <span class="mono">stack/tests/test_vocab_v7.py</span>. 801/801 emitted, 0 failures, 0 goal-set
 violations, 189 tests pass. ⛔ <b>No v6 tuple is edited, reordered or truncated</b> — v7 adds
 new names only, so no checkpoint changes shape. Counts are from the NON-PARITY
 <span class="mono">14231cd29c74</span> lineage.</p></footer>
</div>
'''

if __name__ == "__main__":
    main()
