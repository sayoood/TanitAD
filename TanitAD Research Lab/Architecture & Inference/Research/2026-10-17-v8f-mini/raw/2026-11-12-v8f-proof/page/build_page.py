"""Build the PI-facing page from V8_THEORY.md (+ optional RESULTS_BLOCK.md). Usage: python build_page.py <theory.md> <out.html> [results.md]"""
import html, math, re, sys

SRC, OUT = sys.argv[1], sys.argv[2]
RES = sys.argv[3] if len(sys.argv) > 3 else None


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    return t


def md(text):
    out, in_ul = [], False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("* "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(line[2:])}</li>"); continue
        if in_ul:
            out.append("</ul>"); in_ul = False
        if not line:
            continue
        if line.startswith("# "):
            continue  # the page has its own title block
        if line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif re.match(r"^\*\*T\d+\. ", line):
            m = re.match(r"^\*\*(T\d+)\. (.*)\*\*$", line)
            out.append(f'<h3 class="thm"><span class="tag">{m.group(1)}</span>{inline(m.group(2))}</h3>' if m else f"<h3>{inline(line)}</h3>")
        elif re.match(r"^\d+\. ", line):
            out.append(f'<p class="step">{inline(line)}</p>')
        else:
            out.append(f"<p>{inline(line)}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def chart():
    """Pixel size of rule-relevant objects against range, corpus scale (solid) and native scale (dashed). Log-log."""
    W, H, L, R, T, B = 720, 360, 56, 150, 16, 44
    f_c, f_n, h = 305.58, 916.73, 1.31
    xs = [5 * (80 / 5) ** (i / 60) for i in range(61)]
    lx = lambda d: L + (W - L - R) * (math.log(d / 5) / math.log(80 / 5))
    ly = lambda p: T + (H - T - B) * (1 - (math.log(max(p, 0.03) / 0.03) / math.log(300 / 0.03)))
    series = [("signal head height 0.9 m", "a", lambda f, d: f * 0.9 / d), ("car width 1.8 m", "b", lambda f, d: f * 1.8 / d),
              ("stop line 0.4 m on the ground", "c", lambda f, d: f * h * 0.4 / (d * (d + 0.4)))]
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Pixel size of a signal head, a car and a stop line against range, at the corpus scale and at native scale">']
    for p, lab in ((1, "1 px"), (16, "16 px patch"), (64, "64 px pooled cell")):
        g.append(f'<line x1="{L}" x2="{W-R}" y1="{ly(p):.1f}" y2="{ly(p):.1f}" class="ref"/><text x="{W-R+6}" y="{ly(p)+4:.1f}" class="reft">{lab}</text>')
    for d in (5, 10, 20, 30, 50, 80):
        g.append(f'<line x1="{lx(d):.1f}" x2="{lx(d):.1f}" y1="{T}" y2="{H-B}" class="grid"/><text x="{lx(d):.1f}" y="{H-B+16}" class="tick" text-anchor="middle">{d} m</text>')
    for p in (0.1, 1, 10, 100):
        g.append(f'<text x="{L-6}" y="{ly(p)+4:.1f}" class="tick" text-anchor="end">{p:g}</text>')
    for name, cls, fn in series:
        for f, dash in ((f_c, ""), (f_n, ' stroke-dasharray="5 4"')):
            pts = " ".join(f"{lx(d):.1f},{ly(fn(f, d)):.1f}" for d in xs)
            g.append(f'<polyline points="{pts}" class="s {cls}"{dash}/>')
        g.append(f'<text x="{W-R+6}" y="{ly(fn(f_c, 80))+4:.1f}" class="lab {cls}">{name}</text>')
    g.append(f'<text x="{L}" y="{H-6}" class="tick">range to the object</text><text x="{L-44}" y="{T+10}" class="tick">pixels</text></svg>')
    return "\n".join(g)


theory = md(open(SRC, encoding="utf-8").read())
results = md(open(RES, encoding="utf-8").read()) if RES else "<p>No experiment of this programme has finished yet. Three CPU reads are running on banked checkpoints and cached tokens.</p>"

page = f"""<title>v8 Proof Programme</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>
:root {{ --bg:#f6f5f1; --panel:#ffffff; --fg:#1d2126; --mute:#5c6670; --line:#d9d6cd; --accent:#0b6e6e; --warn:#a4541a; --a:#b3261e; --b:#1f5fa8; --c:#6a4c93; --code:#eeece5 }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14171a; --panel:#1c2024; --fg:#e6e3db; --mute:#9aa4ad; --line:#30363c; --accent:#5cc8c0; --warn:#e0a060; --a:#ff8a80; --b:#82b6f5; --c:#c3a6e8; --code:#23282d; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg:#14171a; --panel:#1c2024; --fg:#e6e3db; --mute:#9aa4ad; --line:#30363c; --accent:#5cc8c0; --warn:#e0a060; --a:#ff8a80; --b:#82b6f5; --c:#c3a6e8; --code:#23282d; color-scheme: dark }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.55 "IBM Plex Sans", system-ui, sans-serif; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:28px 64px }}
main {{ max-width:860px; margin:0 auto; display:flex; flex-direction:column; gap:14px; min-width:0 }}
h1 {{ font:600 34px/1.15 "Source Serif 4", Georgia, serif; margin:0; text-wrap:balance }}
h2 {{ font:600 22px/1.25 "Source Serif 4", Georgia, serif; margin:26px 0 0; padding-top:18px; border-top:1px solid var(--line); text-wrap:balance }}
h3.thm {{ font-size:17px; font-weight:600; margin:18px 0 0; text-wrap:balance }}
.tag {{ font:500 12px "IBM Plex Mono", monospace; color:var(--bg); background:var(--accent); border-radius:4px; padding:2px 7px; margin-right:9px; vertical-align:2px }}
p, li {{ max-width:72ch; margin:0 }}
ul {{ margin:0; padding-left:20px; display:flex; flex-direction:column; gap:6px }}
code {{ font:13.5px "IBM Plex Mono", ui-monospace, monospace; background:var(--code); border-radius:3px; padding:1px 4px; overflow-wrap:anywhere }}
.kicker {{ font:500 12px "IBM Plex Mono", monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--mute) }}
.lead {{ font-size:18px; color:var(--mute) }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px 18px; display:flex; flex-direction:column; gap:10px; min-width:0 }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px }}
.stat b {{ display:block; font:600 26px/1.1 "Source Serif 4", Georgia, serif; color:var(--accent) }}
.stat span {{ font-size:13.5px; color:var(--mute) }}
.ask {{ border-left:3px solid var(--warn) }}
.step {{ padding-left:2px }}
.fig {{ overflow-x:auto }}
svg {{ width:100%; min-width:560px; height:auto; display:block }}
.grid {{ stroke:var(--line); stroke-width:1 }} .ref {{ stroke:var(--mute); stroke-width:1; stroke-dasharray:2 3 }}
.tick, .reft {{ font:11px "IBM Plex Mono", monospace; fill:var(--mute) }} .lab {{ font:500 11.5px "IBM Plex Sans", sans-serif }}
.s {{ fill:none; stroke-width:2 }} .s.a {{ stroke:var(--a) }} .s.b {{ stroke:var(--b) }} .s.c {{ stroke:var(--c) }}
.lab.a {{ fill:var(--a) }} .lab.b {{ fill:var(--b) }} .lab.c {{ fill:var(--c) }}
figcaption {{ font-size:13.5px; color:var(--mute); max-width:72ch }}
</style>
<main>
<div class="kicker">TanitAD · v8F · first edition, 13 Nov 2026</div>
<h1>v8 proof programme: what is proved, what is measured, what comes next</h1>
<p class="lead">Nine theorems tie the measured state of v8F to a way forward. The algebra behind each is checked by a solver; the premises are experiments. Twelve were read. One refuted a prediction of this page, and a later read showed part of that refutation was a limit of the probe; one gave a mixed answer and one follow-up failed its bar; the weight average of two replicates realised 0.74 of the ensemble gain; the direct read on un-pooled and higher-resolution tokens found a small gain from un-pooling and none from the full higher-resolution grid; a one-scalar weight edit reproduced a trained lever on its ROC; one is not read yet.</p>
<div class="stats">
<div class="stat"><b>47 / 47</b><span>verification conditions proved by Z3, statements frozen first</span></div>
<div class="stat"><b>19 / 19</b><span>mutated twins refuted with a counterexample</span></div>
<div class="stat"><b>6.3 m</b><span>beyond this range all ground falls in one row of the 4x10 token grid</span></div>
<div class="stat"><b>-3.4 m</b><span>stop-window error at 2 to 6 s with the verified shield and a perfect stop cue (T1-kin, a ceiling)</span></div>
<div class="stat"><b>0.18 px</b><span>height of a 0.4 m stop line at 30 m in the 256x640 input</span></div>
</div>
<div class="panel">
<strong>The short version</strong>
<ul>
<li>v8F-mini is a three-reader hierarchy on frozen, heavily pooled vision tokens. It beats its kinematic prior by about 1.2 m at 6 s on its own motion, and its levels do listen to each other. It has no planner, no selector and no working predictive world model yet.</li>
<li>The camera geometry is fixed and solver-checked: a 0.4 m stop line is below one pixel beyond about 12.5 m, and the 4x10 grid puts all ground beyond 6.3 m in one row. The prediction drawn from it failed its first test. The pooled 4x10 tokens do carry coarse lead range, lead presence, nearest-agent range and agent count beyond ego motion, and the finer 8x20 grid adds nothing measurable under a rank-256 probe. A later read with the full kernel shows the finer grid does carry more lead content (lead present +0.043 AUROC), so a finer interface is a live lever beside the training signal; the probes are also still data-limited at 111 episodes.</li>
<li>A hierarchy cannot beat a flat model through squared error alone. The matched flat control, already run by the programme, is not established (-0.35 m at one seed, -0.05 m at the other). Read by window class on the banked rollouts, the hierarchy is ahead where the future is not the common case (-0.48 / -0.16 m) and, without own-state supervision, behind where it is (+0.32 / +0.49 m). The margin did not sit in the selection term as the theory was read to predict: most of it is on windows where both models miss the logged mode. A commit rule gated by the tactical confidence does not rescue it (that confidence tells the two window groups apart with AUROC 0.64): the accuracy of the decision is the lever.</li>
<li>Rules can be guaranteed by a shield for every possible network; a loss term cannot do that. What limits the stop rule today is the cue, not the mechanism. Measured this night on real plans (T1-kin, a diagnostic): with a perfect stop cue the verified shield removes every overrun and lowers stop-window error by 3.5 / 3.4 m at the two seeds, at no measured cost; with the cue the model emits it gains nothing. A probe on the pooled tokens does not supply the cue either: vision adds nothing to ego motion for the stop flag and nothing to a constant for the stop distance. The conformal margin holds its 0.9 coverage, at the price of stopping about 7 m short.</li>
<li>Imagination-driven improvement is admissible only when the imagined gain exceeds twice the model gap. With world-model skill at zero that is not yet the case. Measured this night: a closed-form stacked predictor beats the learned world model at every horizon (at 2 s, skill -0.15 becomes +0.17), and the multi-level criterion holds at 0.42. Even so it explains at most 17 % of the latent change.</li>
</ul>
</div>
<div class="panel ask">
<strong>Two things only you can unblock</strong>
<ul>
<li>The Lean 4 proof assistant on the pod (official installer, about 6 to 8 GB). The permission layer refused the install; until then the induction and summation steps stay paper arguments.</li>
<li>The Hugging Face token on the new pod, to stage the 456 GB corpus and the gated encoder weights there. The project file says never copy tokens; an earlier ruling of yours allowed it for pods. I have not done it.</li>
</ul>
</div>
<h2>Results of this programme so far</h2>
{results}
<h2>The camera interface, drawn</h2>
<figure class="panel" style="margin:0">
<div class="fig">{chart()}</div>
<figcaption>Solid lines: the 256x640 cylindrical input used today (5.33 px per degree). Dashed lines: the same objects at native width (16 px per degree; the native width is inherited and still to be verified). Formulas are checked as VC-H4 to VC-H6; focal length and camera height are measured from source.</figcaption>
</figure>
{theory}
</main>
"""
open(OUT, "w", encoding="utf-8", newline="\n").write(page)
print("wrote", OUT, len(page))
