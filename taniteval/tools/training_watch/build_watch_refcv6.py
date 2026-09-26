"""Build the refcv6 Training Watch page -- the refcv3 / refcv4b Training Watch format.

Pulls the run's own artifacts from Thor (metrics.jsonl, config.json, the supervisor log, the stderr
size AND content, the live pids, the checkpoint stamp) and computes every number on the page from
them. Every stderr line must be diagnosed in FACTS["stderr_diagnosed"] or it fails the health chip.
Nothing is typed by hand except the fixed launch facts in FACTS, each of which is banked in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-23-refcv6-fixes/`.

Tier stamp, repeated on the page: the in-training eval is a world-model diagnostic on a FIXED
seeded subset of held-out windows (the same windows every eval) -- T0 -- and is never a
driving-performance claim. The four-family T1 battery belongs to the EvalFlyWheel.

    python build_watch_refcv6.py            # pull from Thor, then build
    python build_watch_refcv6.py --no-pull  # rebuild from the last pull
"""
from __future__ import annotations

import html as _html
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

HOST = "tanitad-thor-wifi"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"      # MSYS ssh deadlocks under Python pipes
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
RUN = "/home/nvidia/refcv6_run/runs/refcv6-r101-s0"
HERE = os.path.dirname(os.path.abspath(__file__))
L = os.environ.get("REFCV6_WATCH_DIR", r"C:\Users\Admin\qland\work\watch6")
TOTAL = 50400
CHANCE_ANCHOR = 1.0 / 117.0
BERLIN = timezone(timedelta(hours=2))     # CEST; the run ends before the October change

# Fixed launch facts -- each one banked in the 2026-09-23-refcv6-fixes package (sec. 11).
FACTS = {
    "launch_berlin": "2026-09-23 20:42",
    "segments": [
        ("284393c", "launch", "2026-09-23 20:42"),
        ("287d72e", "logging-only switch at the step-500 checkpoint", "2026-09-23 21:45"),
        ("82c2331", "A16 fixes (F3 cascade loss live, true label clock) at the step-34,500 checkpoint -- "
                    "the PI's 'stop now, resume with fixes'", "2026-09-26 13:30"),
    ],
    "pi": ('"Full, ~4.0 days (Recommended)"', '"Keep every 10th (Recommended)"'),
    # Every stderr line is either DIAGNOSED here -- exact text, with the diagnosis and its date --
    # or it counts as undiagnosed and fails the health check. A new instance of a known warning
    # carries a new timestamp, so it is undiagnosed again: a second recompile is news.
    "stderr_diagnosed": {
        "W0924 09:03:45.814000 3346338 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use "
        "max_autotune_gemm mode": (
            "benign, diagnosed 2026-09-26: Inductor's notice that Thor's 20 SMs are too few for "
            "max-autotune GEMM, a mode this run does not use. Printed when the backbone compiled a "
            "third time ([0/2]) at step ~6,571 (2026-09-24 09:03 Berlin): ~35 s once; median plain "
            "pace 6.552 s/step over steps 3,000-6,500 vs 6.606 over 6,600-9,000."),
    },
}
# the supervisor's traceback / OOM pattern (sup_refcv6.sh), applied here to the stderr CONTENT
ERR_PAT = re.compile("Trace" "back|CUDA out of mem" "ory|OutOfMemory")
# ZZ<arm>-<step>-<steps>-<tracebacks>-<launch>ZZ; <step> is -1 before the first row, and
# <tracebacks> can span a newline ("0\n0": grep -c prints 0 AND exits 1, then `|| echo 0`)
TOKEN_RE = re.compile(r"ZZrefcv6-r101-s0-(-?\d+)-(\d+)-([^Z]*?)-(\d+)ZZ")
STDERR_CAP = 65536


# ------------------------------------------------------------------ pull ----
def _ssh(cmd: str, timeout: int = 60) -> str:
    r = subprocess.run([SSH, "-n", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", HOST, cmd],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"ZZWATCH-PULL-FAIL ssh rc={r.returncode}: {r.stderr.decode()[:300]}")
    return r.stdout.decode("utf-8", "replace")


def _scp(remote: str, local: str, timeout: int = 300) -> None:
    r = subprocess.run([SCP, "-q", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                        f"{HOST}:{remote}", local], capture_output=True, timeout=timeout)
    if r.returncode != 0 or not os.path.getsize(local):
        raise SystemExit(f"ZZWATCH-PULL-FAIL scp {remote}: {r.stderr.decode()[:300]}")


def pull() -> None:
    os.makedirs(L, exist_ok=True)
    _scp(f"{RUN}/metrics.jsonl", os.path.join(L, "metrics.jsonl"))
    _scp(f"{RUN}/config.json", os.path.join(L, "config.json"))
    _scp(f"{RUN}/sup_refcv6-r101-s0.log", os.path.join(L, "sup.log"))
    o = RUN
    cmd = "; ".join([
        f"O={o}",
        "SP=$(grep -o 'supervisor pid [0-9]*' $O/sup_refcv6-r101-s0.log | tail -1 | cut -d' ' -f3)",
        "TP=$(cat $O/train.pid 2>/dev/null)",
        'echo "K:sup_pid=$SP"', 'echo "K:train_pid=$TP"',
        'ps -p "$SP" >/dev/null 2>&1 && echo K:sup_alive=1 || echo K:sup_alive=0',
        'ps -p "$TP" >/dev/null 2>&1 && echo K:train_alive=1 || echo K:train_alive=0',
        'echo "K:stderr_bytes=$(stat -c %s $O/train.stderr.log 2>/dev/null || echo -1)"',
        'echo "K:ckpt=$(stat -c \'%s %Y\' $O/ckpt.pt 2>/dev/null)"',
        'echo "K:done=$(test -e $O/summary.json && echo 1 || echo 0)"',
        'echo "K:now=$(date -u +%s)"',
        # the stderr CONTENT in the same breath as its size: a non-empty stderr is read and
        # diagnosed line by line, never judged by its byte count
        "echo ZZSTDERR-BEGIN", f"tail -c {STDERR_CAP} $O/train.stderr.log 2>/dev/null",
        "echo", "echo ZZSTDERR-END",
    ])
    out = _ssh(cmd)
    head, sep, rest = out.partition("ZZSTDERR-BEGIN\n")
    body, sep2, _ = rest.rpartition("ZZSTDERR-END")
    if not sep or not sep2:
        raise SystemExit("ZZWATCH-PULL-FAIL stderr section missing from the remote state")
    kv = {}
    for line in head.splitlines():
        if line.startswith("K:") and "=" in line:
            k, _, v = line[2:].partition("=")
            kv[k.strip()] = v.strip()
    need = ("sup_pid", "train_pid", "sup_alive", "train_alive", "stderr_bytes", "now")
    missing = [k for k in need if k not in kv]
    if missing:
        raise SystemExit(f"ZZWATCH-PULL-FAIL remote state incomplete: {missing}")
    json.dump(kv, open(os.path.join(L, "remote_state.json"), "w"), indent=1)
    with open(os.path.join(L, "stderr_tail.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body[:-1] if body.endswith("\n") else body)     # drop the separator `echo`


# ------------------------------------------------------------------ load ----
def load():
    rows = []
    for line in open(os.path.join(L, "metrics.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                       # a torn last line while the trainer writes
    tr = [r for r in rows if "loss" in r and "eval_loss" not in r and isinstance(r.get("step"), int)]
    ev = [r for r in rows if "eval_loss" in r]
    cd = [r for r in rows if any(k.startswith("cd_") and k != "cd_deferred" for k in r)]
    return rows, tr, ev, cd


def ema(xs, alpha=0.15):
    out, m = [], None
    for x in xs:
        if x is None:
            out.append(m)
            continue
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{nd}f}" if nd else f"{x:,.0f}"


def esc(s) -> str:
    return _html.escape(str(s), quote=True)


# ---------------------------------------------------------------- charts ---
X0, X1, Y0, Y1 = 52, 544, 14, 226


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    stepv = 10 * mag
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            stepv = m * mag
            break
    t = math.floor(lo / stepv) * stepv
    ticks = []
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += stepv
    return ticks


def chart(cid, title, sub, ylab, series, xmax, note=None, refs=(), ylo=None, yhi=None):
    """series: list of dict(name, xs, ys, color, kind='line'|'dots'|'dashed')"""
    ally = [y for s in series for y in s["ys"]
            if y is not None and not (isinstance(y, float) and math.isnan(y))]
    if not ally:
        return (f'<figure class="chart"><figcaption><b>{esc(title)}</b><span>{esc(sub)}</span>'
                f'</figcaption><p class="note">No data yet.</p></figure>')
    lo, hi = min(ally), max(ally)
    lo = min(lo, 0) if 0 < lo < 0.25 * hi else lo
    for v, _ in refs:
        lo, hi = min(lo, v), max(hi, v)
    if ylo is not None:
        lo = ylo
    if yhi is not None:
        hi = yhi
    pad = (hi - lo) * 0.06 or 1
    lo, hi = (lo if ylo is not None else lo - pad), (hi if yhi is not None else hi + pad)

    def sx(x):
        return X0 + (X1 - X0) * x / xmax

    def sy(y):
        return Y1 - (Y1 - Y0) * (y - lo) / (hi - lo)

    g = []
    for t in nice_ticks(lo, hi, 5):
        if t < lo - 1e-12 or t > hi + 1e-12:
            continue
        y = sy(t)
        lab = f"{t:g}" if abs(t) >= 1 or t == 0 else f"{t:.2g}"
        g.append(f'<line class="grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>'
                 f'<text class="tick" x="{X0-6}" y="{y+3.5:.1f}" text-anchor="end">{lab}</text>')
    xt = 10000 if xmax > 20000 else (2500 if xmax > 8000 else (500 if xmax > 2000 else 250))
    x = xt
    while x < xmax:
        g.append(f'<text class="tick" x="{sx(x):.1f}" y="242" text-anchor="middle">{x:,}</text>')
        x += xt
    g.append(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
    for v, lab in refs:
        y = sy(v)
        g.append(f'<line class="ref" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>'
                 f'<text class="reflabel" x="{X1}" y="{y-3:.1f}" text-anchor="end">{esc(lab)}</text>')
    for sstep in SWITCH_STEPS:
        if 0 < sstep < xmax:
            g.append(f'<line class="mark switch" x1="{sx(sstep):.1f}" y1="{Y0}" x2="{sx(sstep):.1f}" y2="{Y1}"/>')
    for dstep in DEATH_STEPS:
        if 0 < dstep < xmax:
            g.append(f'<line class="mark death" x1="{sx(dstep):.1f}" y1="{Y0}" x2="{sx(dstep):.1f}" y2="{Y1}"/>')
    data = []
    for s in series:
        pts = [(sx(x), sy(y)) for x, y in zip(s["xs"], s["ys"])
               if y is not None and not (isinstance(y, float) and math.isnan(y))]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if s.get("kind") == "dashed" else ""
        if s.get("kind") != "dots" or len(pts) > 1:
            g.append(f'<path class="line" d="{d}" stroke="var(--{s["color"]})"{dash}/>')
        if s.get("kind") == "dots":
            for x, y in pts:
                g.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="var(--{s["color"]})"/>')
        x, y = pts[-1]
        g.append(f'<circle class="end" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--{s["color"]})"/>')
        data.append({"name": s["name"], "xs": [round(v, 1) for v in s["xs"]],
                     "ys": [None if (v is None or (isinstance(v, float) and math.isnan(v)))
                            else round(v, 4) for v in s["ys"]]})
    g.append(f'<line class="xh" x1="0" y1="{Y0}" x2="0" y2="{Y1}" style="display:none"/>'
             f'<rect class="hit" x="{X0}" y="{Y0}" width="{X1-X0}" height="{Y1-Y0}" fill="transparent"/>')
    legend = "".join(f'<span class="lg"><i style="background:var(--{s["color"]})"></i>{esc(s["name"])}</span>'
                     for s in series)
    note_html = f'<p class="note">{note}</p>' if note else ""
    payload = json.dumps({"xmax": xmax, "series": data}).replace("<", "\\u003c")
    return (f'<figure class="chart"><figcaption><b>{esc(title)}</b><span>{sub}</span></figcaption>'
            f'<span class="ylab">{esc(ylab)}</span>'
            f'<svg class="plot" viewBox="0 0 560 260" data-chart="{cid}" role="img" aria-label="{esc(title)}">{"".join(g)}</svg>'
            f'<div class="legend">{legend}</div><div class="tip" data-for="{cid}"></div>'
            f'<script type="application/json" data-series="{cid}">{payload}</script>{note_html}</figure>')


SWITCH_STEPS: list = []
DEATH_STEPS: list = []


def build() -> str:
    global SWITCH_STEPS, DEATH_STEPS
    rows, tr, ev, cd = load()
    st = json.load(open(os.path.join(L, "remote_state.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(L, "config.json"), encoding="utf-8"))
    sup_log = open(os.path.join(L, "sup.log"), encoding="utf-8", errors="replace").read()
    if not tr:
        raise SystemExit("ZZWATCH-NO-TRAINING-ROWS")

    # ---------------------------------------------------------- segments --
    segs, cur = [], [tr[0]]
    for a, b in zip(tr, tr[1:]):
        if b["elapsed_s"] < a["elapsed_s"]:
            segs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    segs.append(cur)
    planned = len(FACTS["segments"]) - 1
    SWITCH_STEPS = [s[0]["step"] - 50 for s in segs[1:1 + planned]]
    DEATH_STEPS = [s[0]["step"] - 50 for s in segs[1 + planned:]]
    unplanned = max(0, len(segs) - 1 - planned)

    last = tr[-1]
    step_now = last["step"]
    pct = 100.0 * step_now / TOTAL
    seg_now = segs[-1]
    k = min(10, len(seg_now) - 1)
    if k >= 1:
        a, b = seg_now[-1 - k], seg_now[-1]
        pace = (b["elapsed_s"] - a["elapsed_s"]) / (b["step"] - a["step"])
    else:
        pace = last["elapsed_s"] / max(1, last["step"] - (segs[-2][-1]["step"] if len(segs) > 1 else 0))
    now_utc = datetime.fromtimestamp(int(st["now"]), timezone.utc)
    eta_s = (TOTAL - step_now) * pace
    finish = (now_utc + timedelta(seconds=eta_s)).astimezone(BERLIN)
    now_b = now_utc.astimezone(BERLIN)

    # ------------------------------------------------------------ status --
    sup_alive, tr_alive = st.get("sup_alive") == "1", st.get("train_alive") == "1"
    stderr_b = int(st.get("stderr_bytes", "-1"))
    # ⛔ the token is matched WHOLE across line breaks. Splitting the log on whitespace once read
    # the first half of a split token ("...-50400-0" / "0-1ZZ") and reported <steps> as n_err.
    toks = list(TOKEN_RE.finditer(sup_log))
    token = toks[-1].group(0) if toks else None
    n_err = launch_no = None
    token_split = False
    field = ""
    if toks:
        field = toks[-1].group(3)            # "U" = the supervisor could not READ its stderr
        token_split = any(c.isspace() for c in field)
        ints = re.findall(r"\d+", field)
        n_err = int(ints[0]) if ints else None   # grep -c's own count; `|| echo 0` only APPENDS a 0
        launch_no = int(toks[-1].group(4))
    token_ok = n_err is not None or "ZZrefcv6-r101-s0-" not in sup_log
    # the stderr CONTENT, read line by line: every line is diagnosed in FACTS or it fails
    sp = os.path.join(L, "stderr_tail.log")
    stderr_txt = open(sp, encoding="utf-8", errors="replace").read() if os.path.exists(sp) else ""
    stderr_read_ok = stderr_b >= 0 and len(stderr_txt.encode("utf-8")) >= min(stderr_b, STDERR_CAP)
    stderr_truncated = stderr_b > STDERR_CAP
    stderr_lines = [ln for ln in stderr_txt.splitlines() if ln.strip()]
    diag = FACTS["stderr_diagnosed"]
    stderr_undiag = [ln for ln in stderr_lines if ln not in diag]
    n_err_client = sum(1 for ln in stderr_lines if ERR_PAT.search(ln))
    stderr_ok = stderr_read_ok and not stderr_truncated and not stderr_undiag and n_err_client == 0
    launches_total = sup_log.count("lock acquired")
    done = st.get("done") == "1"
    ckpt = st.get("ckpt", "").split()
    ckpt_gb = int(ckpt[0]) / 1e9 if len(ckpt) == 2 else None
    ckpt_age_min = (int(st["now"]) - int(ckpt[1])) / 60 if len(ckpt) == 2 else None
    mem_peak = max((r.get("cuda_max_mem_gb") or 0) for r in tr)
    slots, comp = last.get("trunk_frame_slots"), last.get("trunk_frames_computed")
    dd_frac = comp / slots if slots else None
    ev_windows = sorted({r.get("eval_windows") for r in ev})
    cd_after = [r for r in cd if r["step"] > (SWITCH_STEPS[0] if SWITCH_STEPS else 0)]
    cd_last = cd[-1] if cd else None
    cd_lag = (step_now - cd_last["step"]) if cd_last else None
    readings_ok = cd_last is not None and cd_lag is not None and cd_lag <= 60
    e_first, e_last = (ev[0], ev[-1]) if ev else (None, None)
    eh = (cfg.get("seams", {}) or {}).get("ego_history") or {}
    levers = cfg.get("trunk_memory_levers") or {}

    def chip(ok, good, bad, warn=False):
        cls = "good" if ok else ("warn" if warn else "crit")
        return f'<span class="chip {cls}"><i></i>{esc(good if ok else bad)}</span>'

    learning = bool(ev) and len(ev) >= 2 and e_last["eval_traj"] < e_first["eval_traj"]
    chips = (chip(tr_alive and sup_alive and not done, "training", "finished" if done else "NOT RUNNING", warn=done)
             + chip(learning or len(ev) < 2, "learning" if len(ev) >= 2 else "first eval in", "eval traj not falling", warn=True)
             + chip(unplanned == 0 and token_ok and (n_err in (0, None)),
                    f"{unplanned} unplanned deaths · {planned} planned switch",
                    f"{unplanned} unplanned deaths · tracebacks "
                    f"{n_err if token_ok else 'NOT READ' if field.strip() == 'U' else 'UNPARSED'}")
             + chip(stderr_ok,
                    "stderr empty" if not stderr_lines else f"stderr: {len(stderr_lines)} line(s), all diagnosed",
                    ("stderr NOT READ" if not stderr_read_ok else "stderr past the read cap" if stderr_truncated
                     else f"stderr: {len(stderr_undiag)} UNDIAGNOSED line(s)"))
             + chip(readings_ok, "conflict readings recording", "conflict readings MISSING"))

    # ------------------------------------------------------------ charts --
    xmax = max(step_now, 1000)
    tx = [r["step"] for r in tr]
    exs = [r["step"] for r in ev]

    def trs(key):
        return [r.get(key) for r in tr]

    def evs(key):
        return [r.get(key) for r in ev]

    c1 = chart("c1", "Total loss",
               f"train, per logged batch (EMA α=0.15) · held-out eval every 500 steps on {ev_windows[0] if ev_windows else '—'} fixed windows",
               "loss",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("loss")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_loss"), color="s2", kind="dots")], xmax,
               note="The blue rule marks the step-500 switch to 287d72e (logging only). Loss mixes every "
                    "supervised term; read the per-head charts for what moved.")
    c2 = chart("c2", "Trajectory loss",
               "the diffusion planner's trajectory term · train EMA vs held-out eval", "traj",
               [dict(name="train traj (EMA)", xs=tx, ys=ema(trs("traj")), color="s1"),
                dict(name="eval traj", xs=exs, ys=evs("eval_traj"), color="s2", kind="dots")], xmax)
    c3 = chart("c3", "2 s goal error",
               "distance between the predicted and the true position 2 s ahead, in metres", "m",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("goal2s_err_m")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_goal2s_err_m"), color="s2", kind="dots")], xmax,
               note="Model-inclusive and T0: it scores fit to the held-out futures, not driving.")
    c4 = chart("c4", "Anchor selection accuracy",
               "fraction of windows whose selected anchor is the GT-nearest of the 117 v0-conditioned anchors", "acc",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("anchor_acc")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_anchor_acc"), color="s2", kind="dots")], xmax,
               refs=[(CHANCE_ANCHOR, "chance 1/117")])
    c5 = chart("c5", "Tactical decoder v6 — lateral and longitudinal",
               "cross-entropy against the v8 labels · eval dots", "CE",
               [dict(name="lat train (EMA)", xs=tx, ys=ema(trs("tacv6_lat_ce")), color="s1"),
                dict(name="lon train (EMA)", xs=tx, ys=ema(trs("tacv6_lon_ce")), color="s4"),
                dict(name="eval lat", xs=exs, ys=evs("eval_tacv6_lat_ce"), color="s2", kind="dots"),
                dict(name="eval lon", xs=exs, ys=evs("eval_tacv6_lon_ce"), color="s3", kind="dots")], xmax)
    c6 = chart("c6", "Tactical decoder v6 — the 22-token goal set",
               "multi-label BCE over the tactical goal tokens · goal-confidence BCE", "BCE",
               [dict(name="goal BCE train (EMA)", xs=tx, ys=ema(trs("tacv6_goal_bce")), color="s1"),
                dict(name="confidence BCE train (EMA)", xs=tx, ys=ema(trs("tacv6_goal_conf_bce")), color="s4"),
                dict(name="eval goal BCE", xs=exs, ys=evs("eval_tacv6_goal_bce"), color="s2", kind="dots")], xmax)
    c7 = chart("c7", "BEV map — drivable IoU",
               "intersection-over-union of the predicted and the SAM3 drivable area in the BEV grid (higher is better)", "IoU",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("map_iou_drivable")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_map_iou_drivable"), color="s2", kind="dots")], xmax,
               ylo=0.0, yhi=1.0)
    c7b = chart("c7b", "BEV map — loss",
                "soft cross-entropy of the BEV map head against the SAM3 map GT (lower is better)", "CE",
                [dict(name="train (EMA)", xs=tx, ys=ema(trs("map")), color="s1"),
                 dict(name="eval", xs=exs, ys=evs("eval_map"), color="s2", kind="dots")], xmax)
    c8 = chart("c8", "3-D boxes",
               "Hungarian-matched cuboid set loss against the 3-D agent join", "loss",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("box3d")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_box3d"), color="s2", kind="dots")], xmax)
    c9 = chart("c9", "Agents",
               "agent-slot classification and centre losses", "loss",
               [dict(name="class train (EMA)", xs=tx, ys=ema(trs("agent_cls")), color="s1"),
                dict(name="centre train (EMA)", xs=tx, ys=ema(trs("agent_centre")), color="s4"),
                dict(name="eval class", xs=exs, ys=evs("eval_agent_cls"), color="s2", kind="dots"),
                dict(name="eval centre", xs=exs, ys=evs("eval_agent_centre"), color="s3", kind="dots")], xmax)
    cdx = [r["step"] for r in cd]

    def cds(key):
        return [r.get(key) for r in cd]

    c10 = chart("c10", "Gradient conflict on the trunk",
                "cosine between the planning gradient and the perception gradient, per reading (every 10th step since 500) · EMA", "cos",
                [dict(name="whole trunk", xs=cdx, ys=ema(cds("cd_cos"), 0.2), color="s1"),
                 dict(name="layer4", xs=cdx, ys=ema(cds("cd_stage_layer4_cos"), 0.2), color="s2"),
                 dict(name="fusion", xs=cdx, ys=ema(cds("cd_fuse_cos"), 0.2), color="s3"),
                 dict(name="stem", xs=cdx, ys=ema(cds("cd_stem_cos"), 0.2), color="s4")], xmax,
                refs=[(0.0, "orthogonal")], ylo=-1.0 if cd and min(x for x in cds("cd_cos") if x is not None) < -0.5 else None,
                note="Below zero the two objectives pull the shared trunk apart; near zero they are independent. "
                     "Readings before step 500 were computed but never logged (sec. 11).")
    c11 = chart("c11", "Learning rate", "warmup 2,000 steps then cosine to step 50,400 · ×1e4", "lr×1e4",
                [dict(name="lr ×1e4", xs=tx, ys=[(r.get("lr") or 0) * 1e4 for r in tr], color="s1")], xmax)

    # ---------------------------------------------------------- timeline --
    tl = ['<svg class="timeline" viewBox="0 0 900 96" role="img" aria-label="run segments">',
          '<rect x="0" y="30" width="900" height="26" fill="var(--grid)" rx="3"/>']
    starts = [0] + SWITCH_STEPS + DEATH_STEPS
    starts = sorted(starts)
    ends = starts[1:] + [step_now]
    for i, (s0, s1) in enumerate(zip(starts, ends)):
        cls = "seg" if i == 0 else ("seg switch" if s0 in SWITCH_STEPS else "seg")
        tl.append(f'<rect class="{cls}" x="{900*s0/TOTAL:.1f}" y="30" width="{max(1.5, 900*(s1-s0)/TOTAL):.1f}" height="26" rx="2"/>')
    tl.append(f'<text class="tick" x="{min(900*step_now/TOTAL+6, 640):.1f}" y="47">{step_now:,} of {TOTAL:,}</text>')
    tl.append(f'<text class="tick" x="0" y="80">launched {FACTS["launch_berlin"]} Berlin</text>')
    tl.append(f'<text class="tick" x="900" y="80" text-anchor="end">finish ≈ {finish:%a %d %b %H:%M} Berlin at {pace:.2f} s/step</text></svg>')
    timeline = "".join(tl)

    # ------------------------------------------------------------ tables --
    def td(x, nd=3):
        return f'<td class="num">{fmt(x, nd)}</td>'

    seg_rows = []
    for i, s in enumerate(segs):
        lab = (f"{i+1} · {FACTS['segments'][i][0]} ({FACTS['segments'][i][1]})"
               if i < len(FACTS["segments"]) else f"{i+1} · UNPLANNED relaunch")
        started = FACTS["segments"][i][2] + " Berlin" if i < len(FACTS["segments"]) else "see sup log"
        if len(s) >= 2:
            k2 = min(10, len(s) - 1)
            pm = (s[-1]["elapsed_s"] - s[-1 - k2]["elapsed_s"]) / (s[-1]["step"] - s[-1 - k2]["step"])
        else:
            pm = None
        ended = "— still running" if i == len(segs) - 1 and not done else "next segment"
        seg_rows.append(f'<tr><td>{esc(lab)}</td><td class="num">{s[0]["step"]:,}–{s[-1]["step"]:,}</td>'
                        f'{td(pm, 2)}<td class="num">{len(s)}</td><td>{esc(started)}</td><td>{ended}</td></tr>')
    seg_table = ('<div style="overflow-x:auto"><table><tr><th>segment</th><th class="num">steps logged</th>'
                 '<th class="num">s / step (marginal)</th><th class="num">rows</th><th>started</th><th>ended by</th></tr>'
                 + "".join(seg_rows) + '</table></div>')

    eval_rows = "".join(
        f'<tr><td class="num">{r["step"]:,}</td>{td(r.get("eval_loss"),2)}{td(r.get("eval_traj"))}'
        f'{td(r.get("eval_goal2s_err_m"),2)}{td(r.get("eval_anchor_acc"))}{td(r.get("eval_tacv6_lat_ce"))}'
        f'{td(r.get("eval_tacv6_lon_ce"))}{td(r.get("eval_tacv6_goal_bce"))}{td(r.get("eval_map_iou_drivable"))}'
        f'{td(r.get("eval_box3d"),2)}{td(r.get("eval_agent_cls"))}</tr>' for r in ev)
    evals_table = ('<div style="overflow-x:auto"><table><tr><th class="num">step</th><th class="num">eval loss</th>'
                   '<th class="num">traj</th><th class="num">goal 2 s (m)</th><th class="num">anchor acc</th>'
                   '<th class="num">tac lat CE</th><th class="num">tac lon CE</th><th class="num">goal BCE</th>'
                   '<th class="num">map IoU</th><th class="num">box3d</th><th class="num">agent cls</th></tr>'
                   + eval_rows + '</table></div>')

    health = ('<div style="overflow-x:auto"><table class="kv">'
              f'<tr><td>supervisor / trainer pid</td><td class="num">{esc(st.get("sup_pid"))} {"alive" if sup_alive else "GONE"} · '
              f'{esc(st.get("train_pid"))} {"alive" if tr_alive else "GONE"}</td><td class="muted">checked by pid with <code>ps -p</code></td></tr>'
              f'<tr><td>launches in the supervisor log</td><td class="num">{launches_total}</td><td class="muted">{planned} planned (the step-500 switch) · {unplanned} unplanned</td></tr>'
              f'<tr><td>supervisor token</td><td class="num">{esc(token.replace(chr(10), " ⏎ ")) if token else "—"}</td><td class="muted">step · steps · tracebacks · launch'
              + (f' — read as {n_err} tracebacks: the supervisor printed its zero twice across a line break '
                 '(<code>grep -c</code> prints 0 and exits 1, then <code>|| echo 0</code>); the live supervisor '
                 'is left untouched, the script is fixed for the next launch' if token_split else '') + '</td></tr>'
              f'<tr><td>train.stderr.log</td><td class="num">{stderr_b:,} B · {len(stderr_lines)} line(s)</td><td class="muted">'
              + ("empty" if not stderr_lines and stderr_read_ok else "NOT READ — the pulled content is shorter than the file"
                 if not stderr_read_ok else f"{len(stderr_undiag)} undiagnosed · {n_err_client} traceback/OOM lines, "
                 "counted from the content itself (below)") + '</td></tr>'
              f'<tr><td>peak device memory</td><td class="num">{mem_peak:.2f} GB</td><td class="muted"><code>cuda_max_mem_gb</code> — the only admissible probe on Thor</td></tr>'
              f'<tr><td>frames computed / slots</td><td class="num">{fmt(comp,0)} / {fmt(slots,0)} ({fmt(dd_frac*100 if dd_frac else None,1)} %)</td><td class="muted">each distinct frame once (sec. 10); a window needs 10 of 24</td></tr>'
              f'<tr><td>conflict readings</td><td class="num">{len(cd)} · last at step {cd_last["step"] if cd_last else "—"}</td><td class="muted">every 10th step since 500; {len(cd_after)} since the switch</td></tr>'
              f'<tr><td>ego input</td><td class="num">{"GRU over " + str(eh.get("steps")) + " steps × " + str(eh.get("channels")) + " ch" if eh.get("enable") else "OFF"}</td><td class="muted">the ego-history encoder, from <code>config.json</code> seams (built and checked at launch)</td></tr>'
              f'<tr><td>nav input (last batch)</td><td class="num">{fmt(last.get("nav_injected"),0)}</td><td class="muted">nav from the v7 token reaches the model</td></tr>'
              f'<tr><td>eval windows per eval</td><td class="num">{", ".join(str(w) for w in ev_windows) or "—"}</td><td class="muted">a fixed seeded subset: the SAME windows every eval</td></tr>'
              f'<tr><td>backbone levers (built)</td><td class="num">{esc(", ".join(f"{k}={v}" for k, v in levers.items()))}</td><td class="muted">read off the built trunk, not the flags</td></tr>'
              f'<tr><td>checkpoint</td><td class="num">{fmt(ckpt_gb,2)} GB · {fmt(ckpt_age_min,0)} min old</td><td class="muted">overwritten every 500 steps</td></tr>'
              f'<tr><td>learning rate (now)</td><td class="num">{(last.get("lr") or 0):.3e}</td><td class="muted">{"in the 2,000-step warmup" if step_now < 2000 else "cosine decay to 50,400"}</td></tr>'
              '</table></div>')
    stderr_tbl = ('<h3>train.stderr.log, line by line</h3><div style="overflow-x:auto"><table><tr><th>line</th><th>diagnosis</th></tr>'
                  + "".join(f'<tr><td><code>{esc(ln)}</code></td><td>{esc(diag[ln]) if ln in diag else "<b>UNDIAGNOSED</b> — read it on Thor"}</td></tr>'
                            for ln in stderr_lines[-40:])
                  + '</table></div>') if stderr_lines else ""

    fam = ('<div style="overflow-x:auto"><table><tr><th>family</th><th>what the in-run eval shows (T0)</th><th>the decision-grade test</th></tr>'
           f'<tr><td>LONGITUDINAL</td><td>2 s goal error {fmt(e_last.get("eval_goal2s_err_m") if e_last else None,2)} m (along- and cross-track mixed)</td><td>speed accuracy, headway, TTC — EvalFlyWheel battery, requested 2026-09-23</td></tr>'
           '<tr><td>LATERAL</td><td>no dedicated in-run metric</td><td>heading, curvature, yaw-rate, cross-track — EvalFlyWheel battery</td></tr>'
           f'<tr><td>TACTICAL</td><td>tac lat CE {fmt(e_last.get("eval_tacv6_lat_ce") if e_last else None)} · lon CE {fmt(e_last.get("eval_tacv6_lon_ce") if e_last else None)} · goal BCE {fmt(e_last.get("eval_tacv6_goal_bce") if e_last else None)}</td><td>manoeuvre confusion, goal selection — EvalFlyWheel battery</td></tr>'
           '<tr><td>STRATEGIC</td><td>not applicable: the strategic layer is OFF in this arm (SPEC_REFCV6_V2, PI 2026-09-16)</td><td>—</td></tr>'
           '</table></div>')

    says = []
    if ev:
        says.append(f"<li><b>Held-out trend.</b> Eval traj {e_first['eval_traj']:.3f} → {e_last['eval_traj']:.3f}, "
                    f"2 s goal error {e_first['eval_goal2s_err_m']:.2f} → {e_last['eval_goal2s_err_m']:.2f} m, "
                    f"anchor accuracy {e_first['eval_anchor_acc']:.3f} → {e_last['eval_anchor_acc']:.3f} (chance {CHANCE_ANCHOR:.4f}), "
                    f"map IoU {e_first['eval_map_iou_drivable']:.3f} → {e_last['eval_map_iou_drivable']:.3f} "
                    f"over {len(ev)} eval(s) from step {e_first['step']:,} to {e_last['step']:,}.</li>")
    says.append(f"<li><b>Pace.</b> {pace:.2f} s/step marginal over the last {k} logged rows ⇒ finish ≈ {finish:%a %d %b %H:%M} Berlin.</li>")
    says.append(f"<li><b>Stability.</b> {len(segs)} segment(s): {planned} planned switch, {unplanned} unplanned; "
                f"stderr {stderr_b:,} B in {len(stderr_lines)} line(s), {len(stderr_undiag)} undiagnosed, "
                f"{n_err_client} traceback/OOM; peak {mem_peak:.2f} GB.</li>")
    # A16 (2026-09-26): F3's per-stage cascade loss never ran until the PI's switch. The row that
    # first carries `cascade` is read from the log itself, so the page cannot claim it early.
    cas = [r["step"] for r in tr if "cascade" in r]
    says.append("<li><b>F3 cascade loss (A16).</b> " + (
        f"active from step {cas[0]:,} ({len(cas)} logged rows carry it); before that the run trained "
        "F3 detach-only with F4 on the last layer only (GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST), so "
        "the run is a hybrid from that step on (the PI's ruling, 2026-09-26).</li>" if cas else
        "not in any logged row yet: F3 detach-only, F4 on the last layer only "
        "(GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST).</li>"))
    if cd_after:
        cvals = [r["cd_cos"] for r in cd_after if r.get("cd_cos") is not None]
        neg = sum(1 for v in cvals if v < 0)
        says.append(f"<li><b>Gradient conflict.</b> {len(cvals)} readings since the switch: mean cosine "
                    f"{sum(cvals)/len(cvals):+.3f}, {neg} of {len(cvals)} below zero.</li>")
    doesnt = ["<li>⛔ <b>Nothing here is driving performance.</b> The in-run eval is a T0 world-model diagnostic on "
              f"{ev_windows[0] if ev_windows else '—'} fixed held-out windows. Driving claims come from the four-family "
              "T1 battery and the NavSim suite (EvalFlyWheel), on snapshotted checkpoints — never on Thor while it trains.</li>",
              "<li><b>No interval.</b> Single-run curves carry no estimator; a trend is a trend, not a separated difference.</li>",
              "<li><b>Train and eval are not the same distribution.</b> Training rows are one batch each (noisy); "
              "eval rows average the same fixed windows. Compare eval to eval.</li>"]

    CSS = open(os.path.join(HERE, "watch.css"), encoding="utf-8").read()
    head_stamp = (f"refcv6-r101-s0 on Thor · read {now_b:%Y-%m-%d %H:%M} Berlin (logs are UTC) · every number MEASURED from "
                  f"the run's own <code>metrics.jsonl</code> ({len(rows)} rows by parse: {len(tr)} train / {len(ev)} eval / "
                  f"{len(cd)} conflict) · in-training eval = a WM diagnostic on {ev_windows[0] if ev_windows else '—'} fixed "
                  "held-out windows, <b>T0</b>, never a driving-performance claim")
    e_tile = (f'<div class="tile"><b>{e_last["eval_traj"]:.3f}</b><span>eval traj at {e_last["step"]:,} · first eval '
              f'{e_first["eval_traj"]:.3f} at {e_first["step"]:,}</span></div>'
              f'<div class="tile"><b>{e_last["eval_goal2s_err_m"]:.2f} m</b><span>eval 2 s goal error (first {e_first["eval_goal2s_err_m"]:.2f})</span></div>'
              f'<div class="tile"><b>{e_last["eval_map_iou_drivable"]:.3f}</b><span>eval map drivable IoU (first {e_first["eval_map_iou_drivable"]:.3f})</span></div>'
              if ev else '<div class="tile"><b>—</b><span>no eval yet</span></div>')
    page = f"""<meta charset="utf-8"><title>refcv6 Training Watch</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}
@media (max-width:640px){{.wrap{{padding:20px 16px 48px}}.grid2{{grid-template-columns:1fr}}.tiles{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:28px}}}}</style>
<div class="wrap">
<h1>refcv6 Training Watch</h1>
<p class="stamp">{head_stamp}</p>

<div class="cards">
 <div class="card"><h2>refcv6 {chips}</h2>
  <p class="sub">resnet101 at 416×1024 · nav from the v7 token · ego history · max-speed input · tactical decoder v6 · SAM3 map + agents + 3-D boxes · DDIM diffusion over 117 anchors · batch 16 × {TOTAL:,} steps (the PI's {FACTS["pi"][0]}) · conflict probe every 10th step ({FACTS["pi"][1]})</p>
  <div class="tiles">
   <div class="tile"><b>{step_now:,}</b><span>step of {TOTAL:,} ({pct:.1f} %)</span></div>
   <div class="tile"><b>{pace:.2f} s</b><span>per step, marginal</span></div>
   <div class="tile"><b>{finish:%a %H:%M}</b><span>finish ≈ {finish:%d %b} Berlin ({eta_s/3600:.0f} h)</span></div>
   {e_tile}
  </div></div>
</div>

<h2>Progress — planning</h2>
<div class="grid2">{c1}{c2}{c3}{c4}</div>
<h2>Progress — tactical</h2>
<div class="grid2">{c5}{c6}</div>
<h2>Progress — perception</h2>
<div class="grid2">{c7}{c7b}{c8}{c9}</div>
<h2>Gradient conflict</h2>
<div class="grid2">{c10}{c11}</div>

<h2>Stability</h2>
{timeline}
<h3>Segments</h3>
{seg_table}
<h3>Health</h3>
{health}
{stderr_tbl}

<h2>What the evals say</h2>
<p>All {len(ev)} held-out evals, unedited — the same {ev_windows[0] if ev_windows else '—'} windows each time.</p>
{evals_table}
<h3>The four metric families</h3>
{fam}

<h2>What it says, and what it does not</h2>
<div class="diag"><div><h3>What it says</h3><ul>{"".join(says)}</ul></div>
<div><h3>What it does not say</h3><ul>{"".join(doesnt)}</ul></div></div>

<footer>Source: <code>thor:{RUN}/metrics.jsonl</code>, <code>config.json</code> and the supervisor log, pulled by scp and verified by parse ·
built by <code>taniteval/tools/training_watch/build_watch_refcv6.py</code>, no hand-typed numbers · launch facts:
<code>…/2026-09-23-refcv6-fixes/LAUNCH_READINESS_FIXES.md</code> sec. 11.</footer>
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
    summary = {"step": step_now, "pct": round(pct, 2), "pace_s": round(pace, 3),
               "finish_berlin": f"{finish:%Y-%m-%d %H:%M}", "eval_last": e_last and {
                   k2: e_last.get(k2) for k2 in ("step", "eval_loss", "eval_traj", "eval_goal2s_err_m",
                                                 "eval_anchor_acc", "eval_map_iou_drivable")},
               "segments": len(segs), "unplanned": unplanned, "n_err": n_err, "stderr_bytes": stderr_b,
               "n_err_client": n_err_client, "token_ok": token_ok, "token_split": token_split,
               "stderr_lines": len(stderr_lines), "stderr_undiagnosed": len(stderr_undiag),
               "stderr_read_ok": stderr_read_ok and not stderr_truncated,
               "sup_alive": sup_alive, "train_alive": tr_alive, "done": done,
               "cd_last_step": cd_last and cd_last["step"], "cd_cos_last": cd_last and cd_last.get("cd_cos"),
               "readings_ok": readings_ok, "mem_peak_gb": round(mem_peak, 3)}
    return page, summary


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if "--no-pull" not in a:
        pull()
    out = os.path.join(L, "refcv6_training_watch.html")
    if "--out" in a:
        out = a[a.index("--out") + 1]
    page, summary = build()
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    json.dump(summary, open(os.path.join(L, "watch_summary.json"), "w"), indent=1)
    print(f"wrote {out} ({len(page.encode('utf-8')):,} B)")
    print("ZZWATCH " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
