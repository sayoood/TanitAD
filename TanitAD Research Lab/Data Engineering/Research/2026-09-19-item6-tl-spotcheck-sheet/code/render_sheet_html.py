"""Render raw/sample.json as sheet.html — one row per frame, a verdict picker, CSV out.
Offline, self-contained (images are relative `media/` paths), sha12 only."""
import html
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
d = json.load(open(OUT / "raw" / "sample.json", encoding="utf-8"))
CHIP = {"RED": "#c62828", "GREEN": "#2e7d32", "YELLOW": "#f9a825", "colourless": "#607d8b"}

rows = []
for s in d["rows"]:
    lab = s["label"]
    rows.append(f"""
<tr data-n="{s['n']}" data-sha="{s['sha12']}" data-label="{lab}">
  <td class="n">{s['n']}</td>
  <td class="img"><a href="{s['image']}" target="_blank" rel="noopener"><img src="{s['image']}" alt="frame {s['n']}" loading="lazy"></a></td>
  <td class="meta">
    <div class="chip" style="background:{CHIP[lab]}">{lab if lab != 'colourless' else 'colourless (no colour stated)'}</div>
    <div class="k">clip <code>{s['sha12']}</code> · rig {s['rig']} · {s['frame_time_s']:.2f} s</div>
    <div class="k">{'box-grounded' if s['grounded'] else 'not box-grounded'}</div>
    <p class="reason">“{html.escape(s['reason'])}”</p>
    <label>Verdict
      <select class="v"><option value=""></option>
        <option value="match">✓ lamp shows this colour</option>
        <option value="wrong">✗ lamp shows a different colour</option>
        <option value="nolamp">no relevant lamp visible</option>
        <option value="unclear">can't tell</option></select></label>
    <label>Notes <input class="note" type="text"></label>
  </td>
</tr>""")

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic-light colour spot-check</title>
<style>
:root{{--ink:#1d2327;--mute:#5b6770;--line:#d9dee2;--bg:#f7f8f9;--card:#fff;--accent:#1f5f8b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}}
header,footer{{max-width:1180px;margin:0 auto;padding:20px 24px}}
h1{{font-size:22px;margin:0 0 8px}} p{{margin:6px 0;max-width:75ch}}
.lead{{color:var(--mute)}} code{{font:13px ui-monospace,Consolas,monospace}}
table{{width:min(1180px,100%);margin:0 auto;border-collapse:collapse;background:var(--card)}}
td{{border-top:1px solid var(--line);vertical-align:top;padding:12px}}
td.n{{width:40px;color:var(--mute);font-variant-numeric:tabular-nums}}
td.img img{{width:640px;max-width:100%;height:auto;display:block;border-radius:4px}}
td.meta{{min-width:260px}} .chip{{display:inline-block;color:#fff;font-weight:600;padding:2px 10px;border-radius:12px;margin-bottom:6px}}
.k{{color:var(--mute);font-size:13px}} .reason{{font-style:italic;color:#37424a;font-size:14px}}
label{{display:block;margin-top:8px;font-size:13px;color:var(--mute)}} select,input{{display:block;width:100%;margin-top:2px;font:inherit;padding:4px}}
select:focus,input:focus,button:focus{{outline:2px solid var(--accent);outline-offset:1px}}
button{{font:inherit;padding:8px 14px;border:1px solid var(--accent);background:var(--accent);color:#fff;border-radius:4px;cursor:pointer}}
textarea{{width:100%;height:180px;font:13px ui-monospace,Consolas,monospace;margin-top:8px}}
.count{{font-variant-numeric:tabular-nums;color:var(--mute)}}
</style></head><body>
<header>
<h1>Traffic-light colour spot-check — {len(d['rows'])} frames, about 15 minutes</h1>
<p>Each row is one clip. The label beside the image is the traffic-light colour the Alpamayo reasoning text assigned to it — an <b>Alpamayo-derived teacher signal</b>, not ground truth. The question for every row: <b>does the lamp the car is responding to show that colour?</b></p>
<p class="lead">The top image is the last frame Alpamayo saw ({d['alpamayo_t0_s']} s into the clip). The bottom image is the band inside the yellow box at full resolution, placed on each clip's own horizon, where lamps usually sit. Click an image to open it full size. Some labels describe a change (“the light turns green”); judge the colour that fits the reason text, or choose “can't tell”.</p>
<p class="lead">Sample: {', '.join(f"{k} {v}" for k, v in d['taken'].items())} (fixed seed {d['seed']}; drawn from {', '.join(f"{k} {v}" for k, v in d['pool'].items())}). Clips are named by sha12.</p>
</header>
<table><tbody>{''.join(rows)}</tbody></table>
<footer>
<p><button id="go">Show results as CSV</button> <span class="count" id="count"></span></p>
<textarea id="csv" readonly placeholder="Choose verdicts, then press the button. Copy this text back."></textarea>
</footer>
<script>
const K='tl-spotcheck-2026-09-19';
const rows=[...document.querySelectorAll('tr[data-n]')];
function save(){{try{{localStorage.setItem(K,JSON.stringify(rows.map(r=>[r.querySelector('.v').value,r.querySelector('.note').value])))}}catch(e){{}}}}
function count(){{const n=rows.filter(r=>r.querySelector('.v').value).length;document.getElementById('count').textContent=n+' of '+rows.length+' judged';}}
try{{const s=JSON.parse(localStorage.getItem(K)||'null');if(s)rows.forEach((r,i)=>{{if(s[i]){{r.querySelector('.v').value=s[i][0];r.querySelector('.note').value=s[i][1];}}}})}}catch(e){{}}
rows.forEach(r=>{{r.querySelector('.v').addEventListener('change',()=>{{save();count();}});r.querySelector('.note').addEventListener('input',save);}});
count();
document.getElementById('go').addEventListener('click',()=>{{
  const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
  const lines=['n,sha12,label,verdict,notes'].concat(rows.map(r=>[r.dataset.n,r.dataset.sha,r.dataset.label,r.querySelector('.v').value,q(r.querySelector('.note').value)].join(',')));
  document.getElementById('csv').value=lines.join('\\n');
}});
</script></body></html>
"""
(OUT / "sheet.html").write_text(page, encoding="utf-8")
print("sheet.html written: %d rows, %d bytes" % (len(d["rows"]), len(page.encode("utf-8"))))
