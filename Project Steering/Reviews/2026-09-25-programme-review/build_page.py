#!/usr/bin/env python3
"""Render the programme-review Markdown into a single designed HTML page.

Usage: build_page.py <report.md> <out.html> [chart.svg]
- Markdown -> HTML (tables, fenced code, toc ids)
- evidence-class words become chips (MEASURED, PUBLISHED, ...)
- tables get their own horizontal scroll container
- h2/h3 feed a sticky table of contents
"""
import re
import sys
import html

import markdown

SRC, OUT = sys.argv[1], sys.argv[2]
CHART = sys.argv[3] if len(sys.argv) > 3 else None

md_text = open(SRC, encoding="utf-8").read()

# Title line of the markdown becomes the page hero; drop it from the body.
m = re.match(r"#\s+(.+)\n", md_text)
hero_title = m.group(1).strip() if m else "TanitAD Programme Review"
body_md = md_text[m.end():] if m else md_text

conv = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    extension_configs={"toc": {"toc_depth": "2-3", "permalink": False}},
)
body = conv.convert(body_md)
toc_tokens = conv.toc_tokens

EVID = {
    "MEASURED": "ev-m", "PUBLISHED": "ev-p", "INHERITED": "ev-i",
    "ESTIMATED": "ev-e", "HYPOTHESIS": "ev-h", "UNVERIFIED": "ev-u",
}


def chipify(fragment: str) -> str:
    # only touch text outside tags and outside <code>/<pre>
    out, i = [], 0
    for part in re.split(r"(<pre.*?</pre>|<code.*?</code>|<[^>]+>)", fragment, flags=re.S):
        if not part:
            continue
        if part.startswith("<"):
            out.append(part)
            continue
        part = re.sub(
            r"\b(MEASURED|PUBLISHED|INHERITED|ESTIMATED|HYPOTHESIS|UNVERIFIED)\b",
            lambda mm: f'<span class="ev {EVID[mm.group(1)]}">{mm.group(1)}</span>',
            part,
        )
        out.append(part)
    return "".join(out)


body = chipify(body)
body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
if CHART:
    body = body.replace("<p>[[CHART]]</p>", open(CHART, encoding="utf-8").read())


def toc_html(tokens, depth=0):
    if not tokens:
        return ""
    items = []
    for t in tokens:
        name = re.sub(r"<[^>]+>", "", t["name"])
        sub = toc_html(t.get("children", []), depth + 1) if depth == 0 else ""
        items.append(f'<li><a href="#{t["id"]}">{name}</a>{sub}</li>')
    return f'<ol class="toc-l{depth}">' + "".join(items) + "</ol>"


# toc_tokens root is the (removed) h1 level or the h2 list
roots = toc_tokens
if len(roots) == 1 and roots[0]["level"] == 1:
    roots = roots[0]["children"]
toc = toc_html(roots)

CSS = r"""
:root{
  --bg:#F4F6F9; --surface:#FFFFFF; --ink:#121820; --muted:#556070; --rule:#D8DDE5;
  --accent:#9C6200; --accent-soft:#FFF3D6; --lane:#E3A21A; --code-bg:#EEF1F5;
  --ev-m-fg:#17693F; --ev-m-bg:#DFF3E7; --ev-p-fg:#1D56A0; --ev-p-bg:#E2ECF9;
  --ev-i-fg:#6A4A9E; --ev-i-bg:#EEE8F8; --ev-e-fg:#865300; --ev-e-bg:#FAEED3;
  --ev-h-fg:#46505E; --ev-h-bg:#E9EDF2; --ev-u-fg:#A02828; --ev-u-bg:#FBE5E5;
  --callout:#F8F1E1;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#0E1216; --surface:#151A21; --ink:#E3E7ED; --muted:#98A3B2; --rule:#28303A;
    --accent:#F0B13A; --accent-soft:#2A2110; --lane:#E9AE2E; --code-bg:#1B222B;
    --ev-m-fg:#7BD7A2; --ev-m-bg:#12301F; --ev-p-fg:#8DB9F2; --ev-p-bg:#142741;
    --ev-i-fg:#C3A8F0; --ev-i-bg:#261C3A; --ev-e-fg:#F2C572; --ev-e-bg:#33260C;
    --ev-h-fg:#B9C2CE; --ev-h-bg:#222A34; --ev-u-fg:#F29A9A; --ev-u-bg:#3A1616;
    --callout:#1E1A10;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0E1216; --surface:#151A21; --ink:#E3E7ED; --muted:#98A3B2; --rule:#28303A;
  --accent:#F0B13A; --accent-soft:#2A2110; --lane:#E9AE2E; --code-bg:#1B222B;
  --ev-m-fg:#7BD7A2; --ev-m-bg:#12301F; --ev-p-fg:#8DB9F2; --ev-p-bg:#142741;
  --ev-i-fg:#C3A8F0; --ev-i-bg:#261C3A; --ev-e-fg:#F2C572; --ev-e-bg:#33260C;
  --ev-h-fg:#B9C2CE; --ev-h-bg:#222A34; --ev-u-fg:#F29A9A; --ev-u-bg:#3A1616;
  --callout:#1E1A10;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:"Barlow",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16.5px;line-height:1.62;padding-inline:16px;padding-block:0 64px}
.hero{max-width:1240px;margin:0 auto;padding-block:40px 20px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.hero h1{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:700;font-size:clamp(34px,5.4vw,58px);
  line-height:1.02;letter-spacing:-.005em;margin:.25em 0 .2em;text-wrap:balance}
.hero p.sub{max-width:72ch;color:var(--muted);font-size:18px;margin:0}
.lane{height:6px;margin-block:22px 0;background:repeating-linear-gradient(90deg,var(--lane) 0 46px,transparent 46px 78px);border-radius:2px;opacity:.9}
.legend{display:flex;flex-wrap:wrap;gap:8px 10px;margin-top:16px;align-items:center;font-size:13.5px;color:var(--muted)}
.wrap{max-width:1240px;margin:0 auto;display:grid;grid-template-columns:1fr;gap:28px}
@media (min-width:1080px){.wrap{grid-template-columns:260px minmax(0,1fr)}}
nav.toc{font-size:14px}
@media (min-width:1080px){nav.toc{position:sticky;top:calc(env(safe-area-inset-top,0px) + 16px);align-self:start;
  max-height:calc(100vh - 32px);overflow:auto;padding-right:8px}}
nav.toc details summary{cursor:pointer;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);padding-block:6px}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc .toc-l0>li{margin-block:6px}
nav.toc .toc-l0>li>a{font-weight:600;color:var(--ink)}
nav.toc .toc-l1{margin:4px 0 8px 10px;border-left:2px solid var(--rule);padding-left:10px}
nav.toc .toc-l1 a{color:var(--muted);font-size:13.5px}
nav.toc a{text-decoration:none;display:block;padding-block:1px}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent)}
main{min-width:0;background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:clamp(18px,3.2vw,44px)}
main>*{max-width:80ch}
main>.tbl,main>pre,main>figure{max-width:none}
h2,h3,h4{font-family:"Barlow Condensed","Arial Narrow",sans-serif;line-height:1.15;text-wrap:balance;scroll-margin-top:16px}
h2{font-size:32px;font-weight:700;margin:2.2em 0 .5em;padding-top:.6em;border-top:1px solid var(--rule)}
main>h2:first-child{margin-top:0;border-top:0;padding-top:0}
h3{font-size:23px;font-weight:600;margin:1.8em 0 .4em}
h4{font-size:18px;font-weight:600;margin:1.4em 0 .3em;letter-spacing:.01em}
p,li{max-width:80ch}
a{color:var(--accent)}
strong{font-weight:600}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86em;background:var(--code-bg);padding:.08em .35em;border-radius:4px;overflow-wrap:anywhere}
pre{background:var(--code-bg);border-radius:8px;padding:14px 16px;overflow-x:auto;font-size:13.5px;line-height:1.5}
pre code{background:none;padding:0;overflow-wrap:normal}
blockquote{margin:1.2em 0;padding:12px 18px;background:var(--callout);border-left:4px solid var(--lane);border-radius:0 8px 8px 0}
blockquote p{margin:.4em 0}
.tbl{overflow-x:auto;margin:1.1em 0;border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;vertical-align:top;padding:8px 11px;border-bottom:1px solid var(--rule)}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  background:var(--bg);font-weight:500;white-space:nowrap}
tr:last-child td{border-bottom:0}
td{min-width:7ch}
hr{border:0;border-top:1px dashed var(--rule);margin:2em 0}
.ev{display:inline-block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;font-weight:500;letter-spacing:.06em;
  padding:1px 6px;border-radius:999px;vertical-align:1px;line-height:1.55;white-space:nowrap}
.ev-m{color:var(--ev-m-fg);background:var(--ev-m-bg)} .ev-p{color:var(--ev-p-fg);background:var(--ev-p-bg)}
.ev-i{color:var(--ev-i-fg);background:var(--ev-i-bg)} .ev-e{color:var(--ev-e-fg);background:var(--ev-e-bg)}
.ev-h{color:var(--ev-h-fg);background:var(--ev-h-bg)} .ev-u{color:var(--ev-u-fg);background:var(--ev-u-bg)}
figure.chart{margin:1.4em 0;padding:14px 12px 8px;border:1px solid var(--rule);border-radius:8px;overflow-x:auto}
figure.chart figcaption{font-size:13.5px;color:var(--muted);margin-top:6px;max-width:80ch}
footer{max-width:1240px;margin:28px auto 0;color:var(--muted);font-size:13.5px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
"""

legend = " ".join(
    f'<span class="ev {c}">{k}</span>' for k, c in EVID.items()
)

page = f"""<title>TanitAD Programme Review</title>
<meta name="description" content="Whole-programme review of TanitAD: 4-brain world model, training, data, evaluation, agent harness, frontier research and plan.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,400;0,500;0,600;1,400&family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<header class="hero">
  <div class="eyebrow">TanitAD · independent programme review · 2026-09-25</div>
  <h1>{html.escape(hero_title)}</h1>
  <p class="sub">Model, training, data, evaluation and agent-harness review of the hierarchical latent world model, with frontier research mapped to the programme's defects and a sequenced plan.</p>
  <div class="lane" aria-hidden="true"></div>
  <div class="legend"><span>Evidence classes:</span> {legend}</div>
</header>
<div class="wrap">
  <nav class="toc" aria-label="Contents"><details open><summary>Contents</summary>{toc}</details></nav>
  <main>{body}</main>
</div>
<footer>Source of record: <code>Project Steering/Reviews/2026-09-25-programme-review/</code> in the TanitAD repository (branch <code>claude/optimistic-shannon-vixpo6</code>). Model facts follow <code>MODEL_REGISTRY.md</code>; research claims cite their papers.</footer>
<script>
(function(){{try{{var d=document.querySelector('nav.toc details');if(window.matchMedia('(max-width:1079px)').matches)d.removeAttribute('open');}}catch(e){{}}}})();
</script>
"""
open(OUT, "w", encoding="utf-8").write(page)
print("wrote", OUT, len(page), "bytes")
