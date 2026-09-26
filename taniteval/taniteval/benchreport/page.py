"""The page shell of a W5 report: inline CSS + inline JS, no CDN, no web font, offline by construction.

* light AND dark are both SELECTED token sets (palette.css_tokens); a toggle cycles auto -> dark -> light
  (per-viewer convenience in localStorage, wrapped in try/catch — the page renders without it);
* print: forced light tokens, the stamp becomes a ``position: fixed`` running header so it repeats on
  EVERY printed page, table twins open, figures never split;
* one tooltip for every ``[data-tip]`` mark, on hover AND keyboard focus (textContent only).
"""
from __future__ import annotations

from .palette import DARK, LIGHT, N_HEAT, css_tokens
from .svg import CHART_CSS, esc

PAGE_CSS = """
*{box-sizing:border-box}
html{background:var(--page)}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--s1)}
.stamp{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--border);font-size:12px;color:var(--ink-2);padding:6px 16px;display:flex;flex-wrap:wrap;gap:2px 16px;align-items:center}
.stamp b{color:var(--ink);font-weight:600}
.stamp .sp{flex:1}
.stamp button{font:inherit;font-size:12px;color:var(--ink-2);background:transparent;border:1px solid var(--border);border-radius:6px;padding:1px 8px;cursor:pointer}
main{max-width:1180px;margin:0 auto;padding:18px 16px 64px}
h1{font-size:22px;margin:10px 0 2px;letter-spacing:-.01em;line-height:1.25}
h2{font-size:17px;margin:36px 0 4px;padding-top:12px;border-top:1px solid var(--grid)}
h3{font-size:14px;margin:18px 0 4px}
.sub{color:var(--ink-2);font-size:12.5px;margin:0 0 10px}
.lede{color:var(--ink-2);margin:4px 0 12px;max-width:92ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:12px 0;overflow-x:auto}
figure{margin:0}
figure svg{max-width:100%;height:auto;display:block}
figcaption{font-size:12.5px;color:var(--ink-2);margin-top:8px;max-width:100ch}
details.twin>summary,details.more>summary{cursor:pointer;color:var(--ink-2);font-size:12.5px;margin-top:8px}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0;font-variant-numeric:tabular-nums}
caption{text-align:left;font-size:12.5px;color:var(--ink-2);padding:2px 0 6px;caption-side:top}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;vertical-align:top}
th[scope=row]{text-align:left;font-weight:500}
thead th{font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis)}
td.refused,td.na{color:var(--muted);font-style:italic;text-align:left;white-space:normal;min-width:10ch}
td.wrap,th.wrap{white-space:normal;text-align:left;max-width:60ch}
.heat td.h{font-weight:500}
QCSS
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;margin:12px 0}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.kpi .l{font-size:12px;color:var(--ink-2)}
.kpi .v{font-size:30px;font-weight:600;line-height:1.2;margin:2px 0}
.kpi .d{font-size:12.5px;color:var(--ink-2)}
.banner{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--muted);border-radius:8px;padding:9px 14px;margin:8px 0;font-size:13px}
.banner.critical{border-left-color:var(--st-critical)}.banner.serious{border-left-color:var(--st-serious)}
.banner.warning{border-left-color:var(--st-warning)}.banner.good{border-left-color:var(--st-good)}
.banner .ic{display:inline-block;width:1.4em;font-weight:700}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:0 7px;border-radius:10px;border:1px solid var(--border);white-space:nowrap}
.pill.ok{color:var(--good-text)}.pill.partial{color:var(--ink-2)}.pill.unavailable{color:var(--muted)}
.pill.oracle{color:var(--st-serious);border-color:var(--st-serious)}
.kpi.caveat{border-left:4px solid var(--st-serious)}
.chip{white-space:nowrap}
ul.sub.cols{columns:2;column-gap:28px;margin:4px 0 0;padding-left:18px}
ul.sub.cols li{break-inside:avoid;margin:2px 0}
.num{font-variant-numeric:tabular-nums}
.mut{color:var(--muted)}.ink2{color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--ink-2);margin:4px 0}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.gallery figure{margin:14px 0}
.gallery img{max-width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:#fff}
.fam td.reason{white-space:normal;text-align:left;width:34ch;min-width:26ch;color:var(--ink-2);font-size:11.5px;line-height:1.35}
.fam th[scope=row]{max-width:26ch;white-space:normal}
.fam td{vertical-align:top}
code{font-size:12px}
.foot{margin-top:40px;padding-top:12px;border-top:1px solid var(--grid);font-size:11.5px;color:var(--muted)}
#tt{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);box-shadow:0 2px 10px rgba(0,0,0,.18);border-radius:6px;padding:6px 8px;font-size:12px;max-width:380px;display:none;z-index:20;white-space:pre-line}
@media (max-width:640px){.stamp{position:static}.kpi .v{font-size:24px}}
@media print{
  PRINTLIGHT
  html,body{background:#fff}
  .stamp{position:fixed;top:0;left:0;right:0;border-bottom:1px solid #c3c2b7}
  .stamp button,.noprint,#tt{display:none!important}
  main{padding-top:46px;max-width:none}
  .card,figure,.kpi,table{break-inside:avoid}
  h2,h3{break-after:avoid}
  @page{margin:12mm 9mm 12mm 9mm}
}
"""

JS = r"""
(function(){
  var tt=document.getElementById('tt');
  function place(x,y){tt.style.left=Math.max(4,Math.min(x,window.innerWidth-390))+'px';tt.style.top=Math.max(4,Math.min(y,window.innerHeight-80))+'px';}
  function show(e){var t=e.target&&e.target.closest?e.target.closest('[data-tip]'):null;
    if(!t){tt.style.display='none';return;}
    tt.textContent=t.getAttribute('data-tip');tt.style.display='block';
    if(e.type==='focusin'){var r=t.getBoundingClientRect();place(r.right+8,r.top);}else{place((e.clientX||0)+14,(e.clientY||0)+14);}}
  document.addEventListener('pointermove',show);document.addEventListener('focusin',show);
  document.addEventListener('pointerleave',function(){tt.style.display='none';});
  window.addEventListener('beforeprint',function(){document.querySelectorAll('details').forEach(function(d){d.setAttribute('data-was',d.open?'1':'0');d.open=true;});});
  window.addEventListener('afterprint',function(){document.querySelectorAll('details').forEach(function(d){d.open=d.getAttribute('data-was')==='1';});});
  var root=document.documentElement,KEY='tanitad-benchreport-theme',btn=document.getElementById('theme');
  function label(){var t=root.getAttribute('data-theme');if(btn)btn.textContent='theme: '+(t||'auto');}
  try{var s=localStorage.getItem(KEY);if(s)root.setAttribute('data-theme',s);}catch(e){}
  label();
  if(btn)btn.addEventListener('click',function(){var t=root.getAttribute('data-theme');var n=t==='dark'?'light':(t==='light'?'':'dark');
    if(n)root.setAttribute('data-theme',n);else root.removeAttribute('data-theme');
    try{if(n)localStorage.setItem(KEY,n);else localStorage.removeItem(KEY);}catch(e){}label();});
})();
"""


def _qcss() -> str:
    return "".join(f".q{i}{{background:var(--q{i});color:var(--q{i}-ink)}}" for i in range(N_HEAT))


def _print_light() -> str:
    # force the LIGHT token set in print regardless of the viewer's theme
    toks = css_tokens()
    light_block = toks[toks.index(":root{") + len(":root{"): toks.index("}")]
    return f":root,:root[data-theme=\"dark\"]{{{light_block}}}"


def page(title: str, stamp_html: str, body_html: str, meta: dict) -> str:
    css = css_tokens() + PAGE_CSS.replace("QCSS", _qcss()).replace("PRINTLIGHT", _print_light()) + CHART_CSS
    metas = "".join(f'<meta name="{esc(k)}" content="{esc(v)}">' for k, v in meta.items())
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(title)}</title>{metas}<style>{css}</style></head><body>"
            f"<header class=\"stamp\" id=\"stamp\">{stamp_html}<span class=\"sp\"></span>"
            "<button id=\"theme\" class=\"noprint\" type=\"button\">theme: auto</button></header>"
            f"<main>{body_html}</main><div id=\"tt\" role=\"tooltip\"></div>"
            f"<script>{JS}</script></body></html>\n")


def standalone_svg(svg_markup: str) -> str:
    """A ``fig/*.svg`` file: the same markup plus its own token block (light + dark) so it renders
    correctly outside the page."""
    toks = css_tokens().replace(":root", "svg")
    css = toks + CHART_CSS + "svg{background:var(--surface)}"
    i = svg_markup.index(">") + 1
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_markup[:i] + f"<style>{css}</style>" + svg_markup[i:] + "\n"


_ = (LIGHT, DARK)
