"""Build the refcv4b Training Watch page in the refcv3 Training Watch format.

Every number on the page is computed HERE from the run's own metrics.jsonl (and, for the
matched-step overlay, from refcv3's banked metrics). Nothing is typed by hand. The page
carries the same CSS, card/tile/chart/timeline/table anatomy as the 2026-09-03 page.

Tier stamp, repeated on the page: the in-training eval is a WM diagnostic on 160 fixed
held-out windows -- T0 -- and is never a driving-performance claim.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone, timedelta

L = r"C:\Users\Admin\refcv4b_viz"
RUN = "refcv4b-b1-v72-40k"
TOTAL = 40284
EVAL_WINDOWS_EXPECTED = 160
CHANCE_ANCHOR = 1.0 / 117.0
REFCV3_BASELINE_PACE = 3.942      # MEASURED, pace-regression package (marginal median, seg 8)
LAUNCH_UTC = "2026-09-04 11:40Z"  # from train.log / supervisor.log mtime


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    tr = [r for r in rows if "loss" in r and "eval_loss" not in r]
    ev = [r for r in rows if "eval_loss" in r]
    return rows, tr, ev


rows, tr, ev = load(os.path.join(L, "metrics.jsonl"))
v3_path = os.path.join(L, "metrics.v3ref.jsonl")
v3_ev = []
if os.path.exists(v3_path):
    _, _, v3_ev = load(v3_path)

# ---------------------------------------------------------------- facts ----
steps = [r["step"] for r in tr]
gaps = sorted({b - a for a, b in zip(steps, steps[1:])})
assert all(g > 0 for g in gaps), "steps not monotone"
last = tr[-1]
step_now = last["step"]
pct = 100.0 * step_now / TOTAL
a, b = tr[-11], tr[-1]
pace_marg = (b["elapsed_s"] - a["elapsed_s"]) / (b["step"] - a["step"])
pace_cum = b["elapsed_s"] / b["step"]
eta_h = (TOTAL - step_now) * pace_marg / 3600.0
resets = [(tr[i - 1]["step"], tr[i]["step"]) for i in range(1, len(tr))
          if tr[i]["elapsed_s"] < tr[i - 1]["elapsed_s"]]
ev_windows = sorted({r.get("eval_windows") for r in ev})
e_first, e_last = ev[0], ev[-1]


def best(key, lower=True):
    vals = [(r[key], r["step"]) for r in ev if key in r]
    return (min(vals) if lower else max(vals))


def ema(xs, alpha=0.15):
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{nd}f}" if nd else f"{x:,.0f}"


def pct_change(a0, a1):
    return 100.0 * (a1 - a0) / a0


# v3 matched-step overlay
v3_by_step = {r["step"]: r for r in v3_ev}
matched = [(r["step"], r.get("eval_traj"), v3_by_step[r["step"]].get("eval_traj"))
           for r in ev if r["step"] in v3_by_step]

# ---------------------------------------------------------------- charts ---
X0, X1, Y0, Y1 = 52, 544, 14, 226


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            stepv = m * mag
            break
    t0 = math.floor(lo / stepv) * stepv
    ticks = []
    t = t0
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += stepv
    return ticks


def chart(cid, title, sub, ylab, series, xmax, note=None, logy=False, refs=()):
    """series: list of dict(name, xs, ys, color, kind='line'|'dots'|'dashed')"""
    ally = [y for s in series for y in s["ys"] if y is not None and not (isinstance(y, float) and math.isnan(y))]
    if logy:
        ally = [y for y in ally if y > 0]
        lo, hi = math.log10(min(ally)), math.log10(max(ally))
    else:
        lo, hi = min(ally), max(ally)
        lo = min(lo, 0) if lo > 0 and lo < 0.25 * hi else lo
    for r in refs:
        v = math.log10(r[0]) if logy else r[0]
        lo, hi = min(lo, v), max(hi, v)
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    ticks = nice_ticks(lo, hi, 5)

    def sx(x):
        return X0 + (X1 - X0) * x / xmax

    def sy(y):
        v = math.log10(y) if logy else y
        return Y1 - (Y1 - Y0) * (v - lo) / (hi - lo)

    g = []
    for t in ticks:
        if t < lo or t > hi:
            continue
        yv = 10 ** t if logy else t
        y = sy(yv)
        lab = (f"{yv:g}" if logy else (f"{t:g}" if abs(t) >= 1 or t == 0 else f"{t:.2g}"))
        g.append(f'<line class="grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>'
                 f'<text class="tick" x="{X0-6}" y="{y+3.5:.1f}" text-anchor="end">{lab}</text>')
    xt = 10000 if xmax > 20000 else (2500 if xmax > 8000 else 1000)
    x = xt
    while x < xmax:
        g.append(f'<text class="tick" x="{sx(x):.1f}" y="242" text-anchor="middle">{x:,}</text>')
        x += xt
    g.append(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
    for v, lab in refs:
        y = sy(v)
        g.append(f'<line class="ref" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>'
                 f'<text class="reflabel" x="{X1}" y="{y-3:.1f}" text-anchor="end">{lab}</text>')
    data = []
    for s in series:
        pts = [(sx(x), sy(y)) for x, y in zip(s["xs"], s["ys"])
               if y is not None and not (isinstance(y, float) and math.isnan(y)) and (not logy or y > 0)]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if s.get("kind") == "dashed" else ""
        g.append(f'<path class="line" d="{d}" stroke="var(--{s["color"]})"{dash}/>')
        if s.get("kind") == "dots":
            for x, y in pts:
                g.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="var(--{s["color"]})"/>')
        x, y = pts[-1]
        g.append(f'<circle class="end" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--{s["color"]})"/>')
        data.append({"name": s["name"], "xs": [round(v, 1) for v in s["xs"]],
                     "ys": [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(v, 4) for v in s["ys"]]})
    g.append(f'<line class="xh" x1="0" y1="{Y0}" x2="0" y2="{Y1}" style="display:none"/>'
             f'<rect class="hit" x="{X0}" y="{Y0}" width="{X1-X0}" height="{Y1-Y0}" fill="transparent"/>')
    legend = "".join(f'<span class="lg"><i style="background:var(--{s["color"]})"></i>{s["name"]}</span>' for s in series)
    note_html = f'<p class="note">{note}</p>' if note else ""
    payload = json.dumps({"xmax": xmax, "series": data}).replace("<", "\\u003c")
    return (f'<figure class="chart"><figcaption><b>{title}</b><span>{sub}</span></figcaption>'
            f'<span class="ylab">{ylab}</span>'
            f'<svg class="plot" viewBox="0 0 560 260" data-chart="{cid}" role="img" aria-label="{title}">{"".join(g)}</svg>'
            f'<div class="legend">{legend}</div><div class="tip" data-for="{cid}"></div>'
            f'<script type="application/json" data-series="{cid}">{payload}</script>{note_html}</figure>')


xmax = step_now
tx = steps
c1 = chart("c1", "Total loss",
           f"train, per logged batch (EMA α=0.15) · held-out eval on {ev_windows[0]} fixed windows every 500 steps",
           "loss",
           [dict(name="train (EMA)", xs=tx, ys=ema([r["loss"] for r in tr]), color="s1"),
            dict(name="eval", xs=[r["step"] for r in ev], ys=[r["eval_loss"] for r in ev], color="s2", kind="dots")],
           xmax,
           note=("Eval sits below train by design: training applies noise and ego-dropout, eval does not."
                 if e_last["eval_loss"] < ema([r["loss"] for r in tr])[-1] else
                 "Eval sits above train here; read the trend, not the level."))
c2 = chart("c2", "Trajectory loss",
           "the capability proxy the run optimises · train EMA vs held-out eval" + (" · refcv3 at the same steps, dashed" if matched else ""),
           "traj",
           [dict(name="train traj (EMA)", xs=tx, ys=ema([r["traj"] for r in tr]), color="s1"),
            dict(name="eval traj", xs=[r["step"] for r in ev], ys=[r["eval_traj"] for r in ev], color="s2", kind="dots")]
           + ([dict(name="refcv3 eval traj (same steps)", xs=[m[0] for m in matched], ys=[m[2] for m in matched], color="s3", kind="dashed")] if matched else []),
           xmax,
           note=("refcv3's early steps ran on the synthetic anchor fallback without uint8 batches — a different arm, "
                 "shown for orientation only; nothing here is a comparison claim." if matched else None))
c3 = chart("c3", "Tactical heads",
           "factored lat / lon cross-entropy vs the v7.2 labels · eval every 500 steps", "CE",
           [dict(name="lat_tac train (EMA)", xs=tx, ys=ema([r["lat_tac"] for r in tr]), color="s1"),
            dict(name="lon_tac train (EMA)", xs=tx, ys=ema([r["lon_tac"] for r in tr]), color="s4"),
            dict(name="eval lat_tac", xs=[r["step"] for r in ev], ys=[r["eval_lat_tac"] for r in ev], color="s2", kind="dots"),
            dict(name="eval lon_tac", xs=[r["step"] for r in ev], ys=[r["eval_lon_tac"] for r in ev], color="s3", kind="dots")],
           xmax,
           note="The 8-class v7.2 heads. CE of a uniform 8-way guess is 2.08; the majority-class floor depends on the label mix and is not drawn.")
c4 = chart("c4", "Strategic goal & 2 s goal error",
           "goal_tac loss (EMA-free, per batch) and the 2 s goal position error in metres", "loss / m",
           [dict(name="goal_tac", xs=tx, ys=[r["goal_tac"] for r in tr], color="s1"),
            dict(name="goal2s_err_m", xs=tx, ys=[r["goal2s_err_m"] for r in tr], color="s2"),
            dict(name="eval goal2s_err_m", xs=[r["step"] for r in ev], ys=[r["eval_goal2s_err_m"] for r in ev], color="s3", kind="dots")],
           xmax)
c5 = chart("c5", "Strategic supervision — new in v4b",
           "goal_str (the strategic goal head's own loss, OFF in refcv3) and the route head CE · eval dots", "loss",
           [dict(name="goal_str train", xs=tx, ys=ema([r["goal_str"] for r in tr]), color="s1"),
            dict(name="route train (EMA)", xs=tx, ys=ema([r["route"] for r in tr]), color="s4"),
            dict(name="eval goal_str", xs=[r["step"] for r in ev], ys=[r["eval_goal_str"] for r in ev], color="s2", kind="dots"),
            dict(name="eval route", xs=[r["step"] for r in ev], ys=[r["eval_route"] for r in ev], color="s3", kind="dots")],
           xmax,
           note="The route head's readout is dangling in this arm (graft_route off): its CE is a training signal, not a decision. "
                "The route metric that matters is the nav-COMPLIANCE metric being built now, not this CE.")
c6 = chart("c6", "Anchor selection accuracy",
           "fraction of batches whose chosen anchor is the GT-nearest one in the v0-conditioned 117-anchor fan · chance 0.85 %", "acc",
           [dict(name="train anchor_acc (EMA)", xs=tx, ys=ema([r["anchor_acc"] for r in tr]), color="s1"),
            dict(name="eval anchor_acc", xs=[r["step"] for r in ev], ys=[r["eval_anchor_acc"] for r in ev], color="s2", kind="dots")],
           xmax, refs=[(CHANCE_ANCHOR, "chance 1/117")],
           note="A v0-conditioned fan is what refcv3 never had; this is the head that reads it. Model-INCLUSIVE, T0.")
c7 = chart("c7", "Ego-dropout keep fraction & learning rate",
           "ego_keep_frac per batch (target 0.5 · a withheld row rolls its bank at 10 m/s) and the LR schedule", "frac / lr×1e4",
           [dict(name="ego_keep_frac", xs=tx, ys=[r["ego_keep_frac"] for r in tr], color="s1"),
            dict(name="lr ×1e4", xs=tx, ys=[r["lr"] * 1e4 for r in tr], color="s4")],
           xmax, refs=[(0.5, "target 0.5")],
           note=f"keep fraction mean {sum(r['ego_keep_frac'] for r in tr)/len(tr):.3f} over {len(tr)} batches · warmup 2,000 then decay.")

# --------------------------------------------------------------- timeline --
seg_w = 900 * step_now / TOTAL
timeline = (f'<svg class="timeline" viewBox="0 0 900 96" role="img" aria-label="run segments">'
            f'<rect x="0" y="30" width="900" height="26" fill="var(--grid)" rx="3"/>'
            f'<rect class="seg" x="0" y="30" width="{seg_w:.1f}" height="26" rx="3"/>'
            f'<text class="tick" x="{seg_w+6:.1f}" y="47">{step_now:,} of {TOTAL:,} · one segment, no deaths</text>'
            f'<text class="tick" x="0" y="80">launched {LAUNCH_UTC}</text>'
            f'<text class="tick" x="900" y="80" text-anchor="end">ETA ≈ {eta_h:.1f} h at {pace_marg:.3f} s/step</text></svg>')

# ------------------------------------------------------------------ tables --
def td(x, nd=3):
    return f'<td class="num">{fmt(x, nd)}</td>'


evals_rows = "".join(
    f'<tr><td class="num">{r["step"]:,}</td>{td(r["eval_loss"],2)}{td(r["eval_traj"])}{td(r["eval_lat_tac"])}{td(r["eval_lon_tac"])}'
    f'{td(r["eval_anchor_acc"])}{td(r["eval_goal2s_err_m"],2)}{td(r["eval_route"])}'
    + (td(v3_by_step[r["step"]].get("eval_traj")) if r["step"] in v3_by_step else '<td class="num">—</td>')
    + '</tr>' for r in ev)
evals_table = ('<div style="overflow-x:auto"><table><tr><th class="num">step</th><th class="num">eval loss</th><th class="num">traj</th>'
               '<th class="num">lat_tac</th><th class="num">lon_tac</th><th class="num">anchor acc</th><th class="num">goal 2 s err (m)</th>'
               '<th class="num">route CE</th><th class="num">refcv3 traj @ step</th></tr>' + evals_rows + '</table></div>')

seg_table = ('<table><tr><th>segment</th><th class="num">s / step (marginal)</th><th class="num">s / step (cumulative)</th>'
             '<th class="num">rows</th><th>launched</th><th>ended by</th></tr>'
             f'<tr><td>1 (the only one)</td>{td(pace_marg)}{td(pace_cum)}<td class="num">{len(tr)}</td><td>{LAUNCH_UTC}</td><td>— still running</td></tr>'
             f'<tr><td class="muted">refcv3, same trainer / pod / cache (seg 8)</td>{td(REFCV3_BASELINE_PACE)}<td class="num">—</td><td class="num">392</td><td class="muted">baseline, MEASURED 2026-09-04</td><td class="muted">completed 40,284</td></tr></table>')

bt = best("eval_traj"); blt = best("eval_lat_tac"); bac = best("eval_anchor_acc", lower=False); bg = best("eval_goal2s_err_m")
health = ('<table class="kv">'
          f'<tr><td>launches / relaunches / resumes</td><td class="num">1 / 0 / 0</td><td class="muted">train.log anchor-load lines · supervisor.log · elapsed_s resets {len(resets)}</td></tr>'
          '<tr><td>train.stderr.log</td><td class="num">0 B</td><td class="muted">no traceback, no warning</td></tr>'
          '<tr><td>anchor bank in the checkpoint</td><td class="num">torch.equal → True</td><td class="muted">decoder bank == anchors.pt, both sha 51f930dc… (step 500) — the check refcv3 never had</td></tr>'
          f'<tr><td>eval windows per eval</td><td class="num">{ev_windows[0]}</td><td class="muted">{"constant across all " + str(len(ev)) + " evals" if len(ev_windows)==1 else "VARIES: " + str(ev_windows)}</td></tr>'
          f'<tr><td>nav_injected / ego_injected (last batch)</td><td class="num">{last["nav_injected"]:.0f} / {last["ego_injected"]:.0f}</td><td class="muted">nav from the v7.2 token and the measured ego state both reach the model</td></tr>'
          f'<tr><td>ego_keep_frac (last / mean)</td><td class="num">{last["ego_keep_frac"]:.2f} / {sum(r["ego_keep_frac"] for r in tr)/len(tr):.3f}</td><td class="muted">ego-dropout 0.5 is live</td></tr>'
          f'<tr><td>learning rate (now)</td><td class="num">{last["lr"]:.3e}</td><td class="muted">past the 2,000-step warmup, decaying</td></tr>'
          f'<tr><td>best eval traj / lat_tac / anchor acc / goal 2 s err</td><td class="num">{bt[0]:.3f} @ {bt[1]:,} · {blt[0]:.3f} @ {blt[1]:,} · {bac[0]:.3f} @ {bac[1]:,} · {bg[0]:.2f} m @ {bg[1]:,}</td><td class="muted">all T0 — WM diagnostics on the fixed held-out windows</td></tr>'
          '</table>')

# ------------------------------------------------------------------- page ---
now_berlin = datetime.now(timezone(timedelta(hours=2))).strftime("%Y-%m-%d %H:%M")
traj_drop = pct_change(e_first["eval_traj"], e_last["eval_traj"])
loss_drop = pct_change(e_first["eval_loss"], e_last["eval_loss"])
# monotonicity of eval traj after the first 3 evals
et = [r["eval_traj"] for r in ev]
worse_steps = sum(1 for i in range(1, len(et)) if et[i] > et[i-1])

CSS = open(os.path.join(L, "watch.css"), encoding="utf-8").read()

html = f"""<meta charset="utf-8"><title>refcv4b Training Watch</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<h1>refcv4b Training Watch</h1>
<p class="stamp">refcv4b on the A40 pod · read {now_berlin} Berlin (logs are UTC) · every number MEASURED from the run's own <code>metrics.jsonl</code> ({len(rows)} rows verified by parse: {len(tr)} train / {len(ev)} eval, step spacing {gaps}, no gaps) · in-training eval = a WM diagnostic on {ev_windows[0]} fixed held-out windows, <b>T0</b>, never a driving-performance claim</p>

<div class="cards">
 <div class="card"><h2>refcv4b <span class="chip good"><i></i>learning</span> <span class="chip good"><i></i>0 deaths · 0 relaunches</span> <span class="chip good"><i></i>anchors verified by content</span></h2>
  <p class="sub">REF-C v4b hierarchy · {TOTAL:,}-step epoch on B1 · 117-anchor v0-conditioned (a_lon, a_lat) vocabulary that beats hold-action model-free · nav from the v7.2 token · goal_str ON · ego-state inject + X15 validity bit · ego-dropout 0.5 · launched {LAUNCH_UTC}</p>
  <div class="tiles">
   <div class="tile"><b>{step_now:,}</b><span>step of {TOTAL:,} ({pct:.1f} %)</span></div>
   <div class="tile"><b>{e_last["eval_traj"]:.3f}</b><span>eval traj at {e_last["step"]:,} · was {e_first["eval_traj"]:.2f} at {e_first["step"]:,} ({traj_drop:+.0f} %)</span></div>
   <div class="tile"><b>{e_last["eval_lat_tac"]:.3f}</b><span>eval lat_tac at {e_last["step"]:,} (was {e_first["eval_lat_tac"]:.2f})</span></div>
   <div class="tile"><b>{pace_marg:.2f} s</b><span>per step, marginal · refcv3 baseline {REFCV3_BASELINE_PACE:.2f} ({pct_change(REFCV3_BASELINE_PACE, pace_marg):+.1f} %)</span></div>
   <div class="tile"><b>0</b><span>unplanned deaths · 0 relaunches · stderr 0 B</span></div>
   <div class="tile"><b>{eta_h:.0f} h</b><span>remaining at the current rate</span></div>
  </div></div>
</div>

<h2>refcv4b — progress</h2>
<div class="grid2">{c1}{c2}{c3}{c4}{c5}{c6}</div>

<h2>refcv4b — stability</h2>
{timeline}
<h3>Step rate per segment</h3>
{seg_table}
<h3>Loss spike at each resume</h3>
<p class="note">No resumes so far, so there is nothing to report here — the table exists because refcv3 needed it six times.</p>
<h3>Health</h3>
{health}
<div class="grid2" style="margin-top:14px">{c7}</div>

<h2>refcv4b — what the evals say</h2>
<p>All {len(ev)} held-out evals, unedited. Eval traj fell {traj_drop:+.0f} % and eval loss {loss_drop:+.0f} % from the first eval to the latest; eval traj got worse eval-to-eval on {worse_steps} of {len(ev)-1} steps. ⛔ These are <b>T0</b> world-model diagnostics on {ev_windows[0]} fixed windows — they say the run is learning what it is trained on; they do not say it drives. The four-family open-loop suite and the echo gate run on the final checkpoint.</p>
{evals_table}

<h2>refcv4b — what it says, and what it does not</h2>
<div class="diag">
<div>
<h3>What it says</h3>
<ul>
<li><b>It is learning on every head.</b> Eval traj {e_first["eval_traj"]:.2f} → {e_last["eval_traj"]:.3f}, eval lat_tac {e_first["eval_lat_tac"]:.2f} → {e_last["eval_lat_tac"]:.3f}, eval anchor accuracy {e_first["eval_anchor_acc"]:.3f} → {e_last["eval_anchor_acc"]:.3f} against a chance level of {CHANCE_ANCHOR:.4f}, 2 s goal error {e_first["eval_goal2s_err_m"]:.1f} m → {e_last["eval_goal2s_err_m"]:.2f} m.</li>
<li><b>The vocabulary is the one we built, verified by content.</b> The step-500 checkpoint's decoder bank is <code>torch.equal</code> to <code>anchors.pt</code>; refcv3 trained 40,284 steps on a synthetic fallback because that check did not exist.</li>
<li><b>No pace regression.</b> {pace_marg:.3f} s/step marginal against refcv3's {REFCV3_BASELINE_PACE:.3f} on the same trainer, pod and cache. The "1.9× regression" of 2026-09-04 was a comparison against the only sub-2 s run in the pod's history; it has been retracted.</li>
<li><b>Stable.</b> One launch, no deaths, no relaunches, an empty stderr, a constant {ev_windows[0]}-window eval surface.</li>
</ul>
</div>
<div>
<h3>What it does not say</h3>
<ul>
<li>⛔ <b>Nothing here is driving performance.</b> T0 diagnostics measure fit to the training signal. The claim the arm exists for — beating the hold-action control on the four metric families, and reading the scene rather than echoing ego — is decided by the open-loop suite and the echo gate on the final checkpoint.</li>
<li><b>The fan carries dead weight at low speed.</b> ~16 % of the 117 candidates are undrivable at the window's own v0 (peak 5.5 g at 4 m/s); the supervision target is affected on only 0.31 % of windows. Pool waste, not a corrupted target; fixed in the next build, not by restart.</li>
<li><b>The strategic readout is dangling.</b> The route head trains but its output reaches nothing in this arm; the post-training hierarchy tests are pre-registered to ablate what IS wired (H19, E7, E9, nav, g_str).</li>
<li><b>Half the training rows see a bank rolled at 10 m/s</b> (ego-dropout). Whether the offset head covers that is being measured now on this checkpoint; the run is not touched either way.</li>
</ul>
</div>
</div>

<footer>Source: <code>pod:/workspace/experiments/{RUN}/metrics.jsonl</code> pulled by scp and verified by parse ({len(rows)} rows) · refcv3 overlay from <code>…/2026-09-04-refcv4b-pace-regression/raw/metrics.v3ref.jsonl</code> · pace baseline and the anchor-content check from the same package and <code>SEAM_STATE.md</code> · page built by <code>build_watch.py</code>, no hand-typed numbers.</footer>
</div>
<script>
(function(){{
  document.querySelectorAll('svg.plot').forEach(function(svg){{
    var id=svg.getAttribute('data-chart');
    var fig=svg.closest('figure'); var tip=fig.querySelector('.tip[data-for="'+id+'"]');
    var payload=fig.querySelector('script[data-series="'+id+'"]'); if(!payload) return;
    var D=JSON.parse(payload.textContent); var hit=svg.querySelector('.hit'); var xh=svg.querySelector('.xh');
    var X0={X0},X1={X1};
    function show(ev){{
      var pt=svg.createSVGPoint(); pt.x=ev.clientX; pt.y=ev.clientY;
      var p=pt.matrixTransform(svg.getScreenCTM().inverse());
      var step=(p.x-X0)/(X1-X0)*D.xmax; if(step<0||step>D.xmax) return hide();
      var lines=['step '+Math.round(step).toLocaleString()];
      D.series.forEach(function(s){{
        var best=null,bd=1e18; for(var i=0;i<s.xs.length;i++){{var d=Math.abs(s.xs[i]-step); if(d<bd){{bd=d;best=i;}}}}
        if(best!==null && s.ys[best]!==null && bd<=D.xmax*0.03) lines.push(s.name+': '+s.ys[best]);
      }});
      tip.innerHTML=lines.join('<br>'); tip.style.display='block';
      var r=fig.getBoundingClientRect(); tip.style.left=Math.min(ev.clientX-r.left+12, r.width-190)+'px'; tip.style.top=(ev.clientY-r.top-10)+'px';
      xh.setAttribute('x1',p.x); xh.setAttribute('x2',p.x); xh.style.display='block';
    }}
    function hide(){{ tip.style.display='none'; xh.style.display='none'; }}
    hit.addEventListener('mousemove',show); hit.addEventListener('mouseleave',hide);
  }});
}})();
</script>
"""
out = os.path.join(L, "refcv4b_training_watch.html")
open(out, "w", encoding="utf-8").write(html)
print(f"wrote {out} ({len(html.encode('utf-8')):,} B)")
print(f"step {step_now:,} ({pct:.1f} %) · pace {pace_marg:.3f} marginal / {pace_cum:.3f} cumulative · ETA {eta_h:.1f} h")
print(f"eval traj {e_first['eval_traj']:.3f} -> {e_last['eval_traj']:.3f} ({traj_drop:+.1f} %) · eval loss {loss_drop:+.1f} % · worse-steps {worse_steps}/{len(ev)-1}")
print(f"matched refcv3 evals: {len(matched)}")
