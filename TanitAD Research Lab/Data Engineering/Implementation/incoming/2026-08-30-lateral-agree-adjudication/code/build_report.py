"""Emit the D-LAT-AGREE inspection report with the panels embedded."""
import base64
import json
from pathlib import Path

D = Path("C:/Users/Admin/tanitad-wt/_s2build/release/lat_panels")
meta = {m["clip"][:8]: m for m in json.load(open(D / "panels.json"))}


def img(stem: str) -> str:
    b = (D / "jpg" / f"{stem}.jpg").read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def case(stem: str, verdict: str, note: str) -> str:
    m = meta[stem.split("_")[1]]
    alp = str(m["alp"]).upper()
    verb = m["verb"] or "—"
    cot = (m["cot"] or "").split("type:")[0].strip()
    return f"""
<figure class="case">
  <img src="{img(stem)}" alt="Camera frames and lateral trajectory for clip {m['clip'][:8]}">
  <figcaption>
    <div class="strip">
      <span class="k">clip</span><span class="v mono">{m['clip'][:8]}</span>
      <span class="k">Alpamayo side</span><span class="v mono">{alp}</span>
      <span class="k">geometry</span><span class="v mono">{m['geom']}</span>
      <span class="k">peak lateral</span><span class="v mono num">{m['peak_m']:+.2f} m</span>
      <span class="k">its own text says</span><span class="v mono">{verb}</span>
    </div>
    <p class="cot">&ldquo;{cot}&rdquo;</p>
    <p class="verdict"><strong>{verdict}</strong> {note}</p>
  </figcaption>
</figure>"""


CASES = {
    "extractor": [
        ("EXTRACTOR-WRONG_002646e7",
         "The text and the geometry agree; only the extracted side disagrees.",
         "Parked cars line the right. The car moves 2.96&nbsp;m <em>left</em> to clear them, "
         "which is exactly what its own sentence says. The side field read "
         "&ldquo;right&rdquo; because that is where the <em>parked car</em> was."),
        ("EXTRACTOR-WRONG_00c15800",
         "Same failure, opposite hand.",
         "&ldquo;Nudge right &hellip; to the parked car on the left.&rdquo; The car goes "
         "13.09&nbsp;m right. The side field took the obstacle&rsquo;s side again."),
        ("EXTRACTOR-WRONG_0618670a",
         "A stated right turn, extracted as left.",
         "The sentence opens &ldquo;Turn right into the parking lot&rdquo;; the measured path "
         "goes right. Nothing here supports &ldquo;left&rdquo;."),
    ],
    "genuine": [
        ("GENUINE-CONTRADICTION_038f1d03",
         "A real conflict, and the geometry has the stronger evidence.",
         "The text says turn right. The car displaces 14.15&nbsp;m to the <em>left</em> — far "
         "past any measurement ambiguity. The narration is simply wrong."),
        ("GENUINE-CONTRADICTION_04e1c744",
         "Real conflict, same shape.",
         "&ldquo;Turn right at the intersection&rdquo; against 10.94&nbsp;m of measured left "
         "displacement."),
        ("GENUINE-CONTRADICTION_004bfe84",
         "Real conflict, but weak on both sides.",
         "1.48&nbsp;m is barely past the 1.0&nbsp;m trigger, and the text makes no lateral "
         "claim at all — it is about the queue ahead. This is the class where I would want "
         "your call rather than a rule."),
    ],
    "geom": [
        ("ONE-SIDED-GEOM_00c628f3",
         "Following the road, labelled as a manoeuvre.",
         "6.62&nbsp;m over 123&nbsp;m of travel, on a road that is visibly bending. The text "
         "says &ldquo;keep lane&rdquo; and the text is right. There is no lateral "
         "<em>decision</em> here at all."),
        ("ONE-SIDED-GEOM_00f3bda1",
         "A lane change, labelled NUDGE.",
         "The text states it outright: &ldquo;Lane change to the left.&rdquo; 9.67&nbsp;m of "
         "displacement. This is the absorption we already accepted — but it is worth seeing "
         "what it looks like in the data."),
        ("ONE-SIDED-GEOM_01acc9de",
         "Road curvature again.",
         "&ldquo;Adapt speed for the road curvature&rdquo; — 3.47&nbsp;m of lateral offset "
         "that is the road, not the driver."),
    ],
    "alpamayo": [
        ("ONE-SIDED-ALPAMAYO_00869080",
         "Here the geometry under-calls and the narration is right.",
         "3.69&nbsp;m of measured offset, and the text says &ldquo;nudge right to follow the "
         "lane bend&rdquo; — yet the label is LANE_KEEP. This is the case that argues against "
         "any blanket &ldquo;geometry always wins&rdquo; rule."),
        ("ONE-SIDED-ALPAMAYO_00759fe8",
         "A claim the geometry cannot see.",
         "1.32&nbsp;m — under the trigger. The narration describes a right turn through an "
         "intersection that the six-second window does not contain."),
    ],
    "control": [
        ("CONTROL-AGREE_006a389c",
         "The control, and it reads as it must.",
         "Straight highway, 0.30&nbsp;m of drift, both sources say straight. If this case ever "
         "classifies as anything else, the analysis above is broken."),
    ],
}

sections = {k: "\n".join(case(*c) for c in v) for k, v in CASES.items()}

HTML = f"""<title>Lateral Label Disagreements</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --ground: #FAFAFB; --surface: #FFFFFF; --ink: #13161B; --muted: #626C7A;
  --line: #DFE3E9; --path: #1F6FB4; --thresh: #C0392F; --ok: #2E7D5B;
  --warn: #B4741C; --chip: #EEF2F7;
  --sans: "IBM Plex Sans", system-ui, sans-serif;
  --serif: "Source Serif 4", Georgia, serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #0F1216; --surface: #171B21; --ink: #E4E8EE; --muted: #98A3B2;
    --line: #2A313B; --path: #5AA3DE; --thresh: #E4695E; --ok: #62B98D;
    --warn: #D9A24E; --chip: #1E242C;
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0F1216; --surface: #171B21; --ink: #E4E8EE; --muted: #98A3B2;
  --line: #2A313B; --path: #5AA3DE; --thresh: #E4695E; --ok: #62B98D;
  --warn: #D9A24E; --chip: #1E242C;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--ground); color: var(--ink);
  font-family: var(--serif); font-size: 17px; line-height: 1.65;
  margin: 0; padding: 0 1.25rem 6rem;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
.col {{ max-width: 68ch; }}
header {{ padding: 4.5rem 0 2.5rem; border-bottom: 2px solid var(--ink); }}
.eyebrow {{
  font-family: var(--mono); font-size: .74rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 1.1rem;
}}
h1 {{
  font-family: var(--sans); font-weight: 700; font-size: clamp(2.1rem, 5vw, 3.1rem);
  line-height: 1.08; letter-spacing: -.02em; margin: 0 0 1.2rem; text-wrap: balance;
}}
.lede {{ font-size: 1.16rem; color: var(--ink); margin: 0 0 1.4rem; }}
h2 {{
  font-family: var(--sans); font-weight: 600; font-size: 1.6rem; letter-spacing: -.01em;
  margin: 4rem 0 .5rem; text-wrap: balance;
}}
h3 {{ font-family: var(--sans); font-weight: 600; font-size: 1.06rem; margin: 2.4rem 0 .4rem; }}
p {{ margin: 0 0 1.05rem; }}
.mono {{ font-family: var(--mono); }}
.num {{ font-variant-numeric: tabular-nums; }}
.meta {{
  font-family: var(--mono); font-size: .78rem; color: var(--muted);
  display: flex; flex-wrap: wrap; gap: .4rem 1.6rem; margin-top: 1.6rem;
}}
.tblwrap {{ overflow-x: auto; margin: 1.6rem 0 .6rem; }}
table {{ border-collapse: collapse; width: 100%; font-family: var(--sans); font-size: .95rem; }}
th, td {{ text-align: left; padding: .62rem .9rem; border-bottom: 1px solid var(--line); }}
th {{
  font-family: var(--mono); font-size: .72rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--muted); font-weight: 500;
}}
td.n {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }}
tr.hi td {{ background: color-mix(in srgb, var(--thresh) 9%, transparent); font-weight: 600; }}
.chip {{
  display: inline-block; font-family: var(--mono); font-size: .7rem; letter-spacing: .08em;
  text-transform: uppercase; padding: .2rem .5rem; border-radius: 3px;
  background: var(--chip); color: var(--muted); white-space: nowrap;
}}
.chip.bad {{ background: color-mix(in srgb, var(--thresh) 16%, transparent); color: var(--thresh); }}
.chip.warn {{ background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }}
.chip.good {{ background: color-mix(in srgb, var(--ok) 16%, transparent); color: var(--ok); }}
.case {{
  margin: 2.2rem 0 3rem; background: var(--surface);
  border: 1px solid var(--line); border-radius: 4px; overflow: hidden;
}}
.case img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: 1.1rem 1.3rem 1.4rem; border-top: 1px solid var(--line); }}
.strip {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .55rem 1.4rem; margin-bottom: .95rem;
}}
.strip .k {{
  font-family: var(--mono); font-size: .68rem; letter-spacing: .09em;
  text-transform: uppercase; color: var(--muted); display: block;
}}
.strip .v {{ font-size: .92rem; font-weight: 500; display: block; margin-top: .1rem; }}
.cot {{
  font-size: .95rem; color: var(--muted); font-style: italic;
  border-left: 2px solid var(--line); padding-left: .9rem; margin: 0 0 .9rem;
}}
.verdict {{ margin: 0; font-size: 1rem; }}
.verdict strong {{ font-family: var(--sans); font-weight: 600; }}
.callout {{
  border-left: 3px solid var(--thresh); padding: .1rem 0 .1rem 1.1rem;
  margin: 1.8rem 0;
}}
.callout.ask {{ border-left-color: var(--path); }}
ul {{ margin: 0 0 1.05rem; padding-left: 1.15rem; }}
li {{ margin-bottom: .5rem; }}
a {{ color: var(--path); }}
:focus-visible {{ outline: 2px solid var(--path); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>

<div class="wrap">
<header class="col">
  <p class="eyebrow">D-LAT-AGREE &middot; label forensics</p>
  <h1>What the lateral disagreements actually look like</h1>
  <p class="lede">The corpus reports that its two label sources disagree on lateral direction
  41.7&nbsp;% of the time. Opening the cases one by one shows that most of that number is not
  disagreement at all &mdash; and that the smaller part which <em>is</em> real splits three ways,
  only one of which needs your ruling.</p>
  <div class="meta">
    <span>blob md5 ee44875916ae7c0ac002c6716b9658ea</span>
    <span>4,719 clips &middot; 4,416 with both fields</span>
    <span>camera + egomotion, measured</span>
  </div>
</header>

<section class="col">
<h2>The number, decomposed</h2>
<p>Every panel below plots the car&rsquo;s own path in the frame of its key
frame &mdash; positive is left &mdash; against the &plusmn;1.0&nbsp;m threshold that decides a
NUDGE label. The camera strip above each plot is the same six seconds.</p>
<div class="tblwrap">
<table>
<thead><tr><th>class</th><th>what it means</th><th class="n">clips</th><th class="n">share</th></tr></thead>
<tbody>
<tr><td><span class="chip good">both agree</span></td><td>same side, or both straight</td><td class="n">2,827</td><td class="n">64.0&nbsp;%</td></tr>
<tr><td><span class="chip warn">one-sided</span></td><td>one names a side, the other says straight</td><td class="n">1,373</td><td class="n">31.1&nbsp;%</td></tr>
<tr class="hi"><td><span class="chip bad">opposite</span></td><td>both name a side, and the sides conflict</td><td class="n">216</td><td class="n">4.9&nbsp;%</td></tr>
</tbody>
</table>
</div>
<p>Of those 216, a quarter turn out not to be conflicts either. That is where to start.</p>
</section>

<section>
<div class="col">
<h2>1 &middot; Alpamayo contradicts itself <span class="chip bad">54 clips</span></h2>
<div class="callout">
<p><strong>Correction, and it is mine.</strong> I first published this section claiming our
extractor was buggy &mdash; that it grabbed an obstacle&rsquo;s side instead of the car&rsquo;s.
Two things were wrong with that. The side does not come from the reasoning text at all: it is
Alpamayo&rsquo;s own structured field, and our reading of it is faithful. And I reached the
conclusion by <em>selecting cases where the text matched the geometry</em>, then concluding the
text was reliable &mdash; which is circular.</p>
<p>Measured without that filter, on the 994 clips where Alpamayo supplies both a structured side
and a manoeuvre sentence: the <strong>structured field matches geometry 59.7&nbsp;%</strong> of the
time, the <strong>reasoning text only 17.9&nbsp;%</strong>. Switching to the text &mdash; the fix I
was about to propose &mdash; would have made the labels substantially worse.</p>
</div>
<p>What the cases below still show is real, and it is a fact about Alpamayo rather than about us:
its structured label and its own reasoning disagree, and here the geometry sides with the
reasoning. That is worth seeing. It is not a licence to trust the reasoning in general.</p>
</div>
{sections['extractor']}
</section>

<section>
<div class="col">
<h2>2 &middot; Genuine conflicts <span class="chip bad">~162 clips</span></h2>
<p>Here the narration really does claim one direction and the car really does go the other way.
In the large-displacement cases the measurement is not in doubt, so the narration is wrong. The
weak ones are less obvious.</p>
</div>
{sections['genuine']}
</section>

<section>
<div class="col">
<h2>3 &middot; Geometry sees a manoeuvre, the text does not <span class="chip warn">505 clips</span></h2>
<p>This is the largest block and, to my eye, the most consequential &mdash; not because the
sources conflict, but because of <em>what the label is calling a nudge</em>.</p>
</div>
{sections['geom']}
<div class="col">
<div class="callout">
<p>Across all NUDGE-labelled clips the median peak offset is <strong>6.2&nbsp;to&nbsp;8.8&nbsp;m</strong>,
<strong>71&ndash;81&nbsp;%</strong> exceed a full lane width, and <strong>97&nbsp;%</strong> never
return toward where they started. A nudge &mdash; swerve out, come back &mdash; is about
<strong>1&nbsp;%</strong> of the class it names.</p>
</div>
<p>The threshold is 1.0&nbsp;m with no upper bound and no lane reference, so following a bend and
changing lane both land in the same bucket as easing around a parked car. Telling those apart
needs a lane-relative measurement, which is exactly what we do not have today.</p>
</div>
</section>

<section>
<div class="col">
<h2>4 &middot; The text sees a manoeuvre, geometry does not <span class="chip warn">868 clips</span></h2>
<p>Mostly narration describing something outside the six-second window. But not always &mdash;
and the exception matters for the rule you are being asked to set.</p>
</div>
{sections['alpamayo']}
</section>

<section>
<div class="col">
<h2>Control</h2>
<p>A case where the answer is known in advance. If the classification above is sound, this
lands in &ldquo;both agree&rdquo; and nowhere else &mdash; all 1,994 of its kind do.</p>
</div>
{sections['control']}
</section>

<section class="col">
<h2>What I would ask you to decide</h2>
<div class="callout ask">
<p>Nothing here blocks training: this flag is diagnostic and no trainer reads it. These are
label-quality calls for the next extraction.</p>
</div>
<ul>
<li><strong>Alpamayo&rsquo;s self-contradiction (54 clips)</strong> &mdash; nothing for me to fix in
our code; the parse is faithful. The open question is whether a clip whose two Alpamayo outputs
disagree should carry the corroboration flag at all, or be marked unusable for it. My inclination
is the latter: a source that contradicts itself is not a second opinion.</li>
<li><strong>Whether NUDGE should keep its name and its unbounded threshold.</strong> On the
evidence above it is mostly not nudging. An upper bound, or a split once a lane reference
exists, are both real options.</li>
<li><strong>Whether geometry always wins.</strong> The last panel in section&nbsp;4 is a case
where it under-calls and the narration is right; a blanket rule would keep that one wrong.</li>
</ul>
</section>
</div>
"""

out = Path("C:/Users/Admin/tanitad-wt/_s2build/release/lat_report.html")
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out} — {out.stat().st_size/1e6:.2f} MB")
