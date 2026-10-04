"""Build the refcv7 Training Watch page -- the refcv6 Training Watch, parameterised by run, plus what
refcv7 adds (Project Steering/SPEC_REFCV7.md sec. 6-15):

* the 10 cm map's 40-key contract (A8 item 3) as a class x band heatmap with per-class trends, and
  the THIN-CLASS ALARM tile -- the REGISTERED rule only (LOGGING_SPEC_MAP10 sec. 5 item 3), with
  20-60 m amber and beyond 60 m values only (the Master Mind's ruling 2026-09-27);
* the box heads' P0 detection metrics (A9) with the A10 confidence-ratio ALARM band [0.5, 1.5], armed
  from step 5,000 (the same ruling);
* the DECLARED gradient reach (D3): an alarm on any declared ``ga_*`` key absent or exactly 0;
* NEW-1's residual prior as the run records it, and the marginal pace against the 8.0 s/step line
  (shown, not decided: the PI rule itself is applied at the G-LIVE smoke).

It pulls the run's own artifacts from Thor (metrics.jsonl, config.json, the supervisor log, the
stderr size AND content, the live pids, the checkpoint stamp) and computes every number on the page
from them. Nothing is typed by hand except FACTS (planned segments, diagnosed stderr lines) and the
literal contracts below, each with its source. A key the page needs and the log lacks renders
UNAVAILABLE and raises that section's alarm; it is never drawn as 0.

Tier stamp, repeated on the page: the in-training eval is a world-model diagnostic on a FIXED seeded
subset of held-out windows (the same windows every eval) -- T0 -- and never a driving-performance
claim. The four-family T1 battery and NavSim belong to the EvalFlyWheel.

    python build_watch_refcv7.py                       # pull from Thor, then build
    python build_watch_refcv7.py --no-pull             # rebuild from the last pull
    python build_watch_refcv7.py --metrics M --out O   # build from ONE metrics.jsonl, never pulls:
                                                       # the launch gate's G-MAP item 4 form

--run <thor dir>   the run directory (else $REFCV7_RUN, else `--out` of the gated argv file
                   stack/ops/runs.d/<arm>.argv.json in this tree -- the file sup_refcv7.sh runs)
--arm <arm>        else $REFCV7_ARM, else refcv7-r101-s0
--dir <dir>        the local watch dir (else $REFCV7_WATCH_DIR)
--out <page>       --summary <json>   (default: beside the page, <page stem>.summary.json)
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
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_ARM = "refcv7-r101-s0"
ARM = os.environ.get("REFCV7_ARM", DEFAULT_ARM)
L = os.environ.get("REFCV7_WATCH_DIR", r"C:\Users\Admin\qland\work\watch7")
TOTAL_SPEC = 50400             # SPEC_REFCV7 sec. 4: b16 x 50,400 steps (the refcv6 budget)
CKPT_EVERY_SPEC = 500          # SPEC_REFCV7 sec. 4: checkpoints every 500
EVAL_EVERY_SPEC = 500          # the gated argv's --eval-every
PACE_REF_S = 6.4               # refcv6-r101-s0 on Thor, MEASURED (SPEC sec. 6.2 item 5; the gate's cost_ref)
PACE_PI_LINE_S = 8.0           # 6.4 x 1.25: above it the cost goes to the PI (SPEC 6.2 item 5, 11.1, 12 item 4)
CHANCE_ANCHOR = 1.0 / 117.0

# Fixed facts. Planned segments, in order: (label, what). The first is the launch; every LATER one is
# a switch the PI ordered (a new gate run and a new token, SPEC sec. 2). An elapsed_s reset beyond
# these is an UNPLANNED relaunch. The launch TIME is read from the supervisor log, never typed.
FACTS = {
    "segments": [("launch", "the gated launch: sup_refcv7.sh -> launch_gate.py exec")],
    # Every stderr line is either DIAGNOSED here -- exact text, with the diagnosis and its date --
    # or it counts as undiagnosed and fails the health chip. A new instance of a known warning
    # carries a new timestamp, so it is undiagnosed again. Empty at launch.
    "stderr_diagnosed": {},
}
# the supervisor's traceback / OOM pattern (sup_refcv7.sh), applied here to the stderr CONTENT;
# built from pieces so this source can never match itself in an echoing PTY
ERR_PAT = re.compile("Trace" "back|CUDA out of mem" "ory|OutOfMemory")
STDERR_CAP = 65536

# ---- the 10 cm map (NEW-2) -------------------------------------------------------------------------
#: SPEC_REFCV7 A8 item 3 (sec. 13, landed): the 40-key Watch contract `eval_map_hires_iou_{cls}_{band}`;
#: the spelling is NEW-2's `map_head_hires.per_class_key` (LOGGING_SPEC_MAP10 sec. 1)
MAP_CLASSES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
MAP_BANDS = ("0_20", "20_40", "40_60", "60_80", "80_100")
#: the five thin classes (map-signal audit RESULT.md sec. 1; the launch gate's `thin_classes`) -- a LABEL
#: on this page; the alarm below reads every class
THIN_CLASSES = ("lane", "crosswalk", "arrow", "edge", "hatched")
#: THE THIN-CLASS ALARM IS THE REGISTERED RULE ONLY (LOGGING_SPEC_MAP10 sec. 5 item 3, landed; the Master
#: Mind's ruling 2026-09-27): RED iff ANY class's 0_20 eval IoU is <= 0.05 at ANY eval at or after step
#: 5,000, or ANY eval row lacks a required key. Latched, as registered ("at any eval").
THIN_FLOOR = 0.05
THIN_FROM_STEP = 5000
RED_BAND = "0_20"
#: the same ruling: 20-60 m at or below the floor from step 5,000 is AMBER, informative ...
AMBER_BANDS = ("20_40", "40_60")
#: ... and beyond 60 m the page shows VALUES ONLY, no colour: refcv6 predicted nothing there, and A7
#: expects far bands low. ⛔ The per-band BAR against refcv6@38k (A8, BAR-M7-1..3) needs the paired
#: interval on the eval kit, so it is NOT a Watch alarm: the EvalFlyWheel battery reads it at milestones.
PLAIN_BANDS = ("60_80", "80_100")
#: the heatmap's bin edges (bin i holds v <= edge i); the first edge IS the registered floor
HEAT_EDGES = (0.05, 0.2, 0.4, 0.6, 0.8)
BAND_COLORS = ("s1", "s2", "s3", "s4", "s5")      # fixed order, never cycled (validated palette)

# ---- the box heads (A9 P0, A10) ------------------------------------------------------------------------
#: ⚠️ UNVERIFIED until the box-head package lands: the spelling is the box builder's in-progress
#: `stack/tanitad/eval/detection_metrics.py` (`metric_keys`, `calib_key_names`), read 2026-09-27 03:34.
BOX_HEADS = ("box3d", "agent")
BOX_RATIO_BAND = (0.5, 1.5)                        # SPEC_REFCV7 A10 sec. 15.3: outside is a Watch ALARM
#: The ratio alarm is ARMED from step 5,000 (the Master Mind's ruling 2026-09-27, registered by the MM in
#: the SPEC). Before it the tile is grey, "warming up (presence prior 0.01)", with the value shown: at
#: the 0.01 prior (logit -4.6) almost no slot starts above 0.5, so the early ratio is expected near 0.
BOX_ARM_STEP = 5000
BOX_WARMUP = "warming up (presence prior 0.01)"
BOX_THR = (("0p5", "0.5 m"), ("1", "1 m"), ("2", "2 m"), ("4", "4 m"))
BOX_BANDS = (("all", "all"), ("0_20", "0–20 m"), ("20_40", "20–40 m"), ("40_60", "40–60 m"))
#: refcv6@38k, MEASURED by the landed box-head audit (`…/2026-09-26-box-head-audit/RESULT.md` sec. 0,
#: 3, 5.2, 5.3, 7.2): 139 eval clips x 4 windows = 556 "clipgrid" windows, the 0.5 gate, scored against
#: the TRAINER's targets; the `_vis1` values are the same checkpoint re-scored under VIS-1. A reference
#: on OTHER windows, never a paired comparison.
REF6 = {"box3d": {"ap2m": 0.210, "ap2m_vis1": 0.151, "prec": 0.162, "rec": 0.551, "prec_vis1": 0.109,
                  "rec_vis1": 0.578, "ratio": 3.40, "auroc": 0.855},
        "agent": {"ratio": 3.75, "auroc": 0.860}}
BOX_REF_SRC = ("refcv6@38k, box-head audit RESULT.md (landed 35e8207): 556 clipgrid eval windows, "
               "gate 0.5, vs the trainer's targets; VIS-1 re-score where marked. Other windows than "
               "this run's in-run eval: a reference, not a paired comparison.")

# ---- NavSim + battery (EvalFlyWheel) -----------------------------------------------------------------
#: The EvalFlyWheel's refcv7 package (the Master Mind's ruling 2026-09-27; created at the first
#: milestone -- until then the section reads "nothing is shown rather than a guess"). The live lane is
#: not named: without $REFCV7_EVAL_LIVE an unbanked split reads "not banked", never "not run".
EVAL_PKG_DEFAULT = (r"D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming"
                    r"\2026-09-27-refcv7-standard-tests")
EVAL_PKG = os.environ.get("REFCV7_EVAL_PKG", EVAL_PKG_DEFAULT)
EVAL_LIVE = os.environ.get("REFCV7_EVAL_LIVE")
#: the arm id the EvalFlyWheel banks refcv7 under (the Master Mind's ruling 2026-09-27). A wrong id reads
#: as missing counts and the count guard REFUSES the split -- loud, never a wrong number.
NAVSIM_ARM = os.environ.get("REFCV7_NAVSIM_ARM", "R7_A1")
#: The published split sizes, written here as literals from the NAVSIM protocol, independently of any
#: summary this page reads (refcv6's builder, and navsim/code/score_arm6.py:72-76, agree). A banked
#: summary must match them. MEASURED 2026-09-26: a scorer that stopped after 1,464 of 12,146 navtest rows
#: was parsed anyway and rendered as "64.14 ... SUBSET". A FAILED run's partial output is never a smaller
#: valid run (RETR-2026-09-26-NAVTEST30K-PARTIAL).
EXPECTED_N = {"navtest": {"n_tokens": 12146}, "navhard": {"n_stage1": 450, "n_stage2": 5462},
              "warmup": {"n_stage1": 16, "n_stage2": 204}}

SWITCH_STEPS: list = []
DEATH_STEPS: list = []


# ------------------------------------------------------------------ small helpers ----
def esc(s) -> str:
    return _html.escape(str(s), quote=True)


def fmt(x, nd=3):
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)) or \
            (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:,.{nd}f}" if nd else f"{x:,.0f}"


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _cnt(x):
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else "missing"


def _jload(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return None


def ema(xs, alpha=0.15):
    out, m = [], None
    for x in xs:
        if x is None:
            out.append(m)
            continue
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def berlin(t: datetime) -> datetime:
    """UTC -> Europe/Berlin by the EU rule (CEST from the last Sunday of March 01:00 UTC to the last
    Sunday of October 01:00 UTC). No tz database needed: the dev box has none."""
    t = t.astimezone(timezone.utc)

    def last_sunday(month):
        d = datetime(t.year, month + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        return d - timedelta(days=(d.weekday() + 1) % 7)
    summer = last_sunday(3) <= t < last_sunday(10)
    return t.astimezone(timezone(timedelta(hours=2 if summer else 1), "CEST" if summer else "CET"))


def band_label(b: str) -> str:
    lo, hi = b.split("_")
    return f"{lo}–{hi} m"


def _argv_vals(argv, flag, n=1):
    """The `n` values after the LAST occurrence of `flag`, or None."""
    v = None
    if isinstance(argv, list):
        for i, t in enumerate(argv):
            if t == flag and i + n < len(argv):
                v = argv[i + 1:i + 1 + n]
    return v


def _argv_val(argv, flag):
    v = _argv_vals(argv, flag, 1)
    return v[0] if v else None


# ------------------------------------------------------------------ the run dir ----
def gated_argv_file(arm: str) -> str:
    return os.path.join(TREE, "stack", "ops", "runs.d", f"{arm}.argv.json")


def resolve_run(arm: str, cli_run: str | None = None):
    """(run dir, where it came from) or (None, why not). One source of truth, never a guess:
    --run, then $REFCV7_RUN, then `--out` of the gated argv file the supervisor runs."""
    if cli_run:
        return cli_run, "--run"
    if os.environ.get("REFCV7_RUN"):
        return os.environ["REFCV7_RUN"], "$REFCV7_RUN"
    p = gated_argv_file(arm)
    d = _jload(p)
    a = d.get("argv") if isinstance(d, dict) else d
    out = _argv_val(a, "--out")
    if out:
        return out, f"--out of stack/ops/runs.d/{arm}.argv.json"
    return None, (f"no --run, no $REFCV7_RUN, and no --out in {p} (the gated argv lands with the "
                  "launch gate)")


# ------------------------------------------------------------------ pull ----
def _ssh(cmd: str, timeout: int = 60) -> str:
    r = subprocess.run([SSH, "-n", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", HOST, cmd],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"ZZWATCH-PULL-FAIL ssh rc={r.returncode}: {r.stderr.decode()[:300]}")
    return r.stdout.decode("utf-8", "replace")


def _scp(remote: str, local: str, timeout: int = 600) -> None:
    r = subprocess.run([SCP, "-q", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                        f"{HOST}:{remote}", local], capture_output=True, timeout=timeout)
    if r.returncode != 0 or not os.path.exists(local) or not os.path.getsize(local):
        raise SystemExit(f"ZZWATCH-PULL-FAIL scp {remote}: {r.stderr.decode()[:300]}")


def pull(run: str, arm: str, out_dir: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", run or "") or not re.fullmatch(r"[A-Za-z0-9_.-]+", arm):
        raise SystemExit(f"ZZWATCH-PULL-FAIL refusing an unsafe run dir / arm: {run!r} / {arm!r}")
    os.makedirs(out_dir, exist_ok=True)
    _scp(f"{run}/metrics.jsonl", os.path.join(out_dir, "metrics.jsonl"))
    _scp(f"{run}/config.json", os.path.join(out_dir, "config.json"))
    _scp(f"{run}/sup_{arm}.log", os.path.join(out_dir, "sup.log"))
    cmd = "; ".join([
        f"O={run}",
        f"SP=$(grep -o 'supervisor pid [0-9]*' $O/sup_{arm}.log | tail -1 | cut -d' ' -f3)",
        "TP=$(cat $O/train.pid 2>/dev/null)",
        'echo "K:sup_pid=$SP"', 'echo "K:train_pid=$TP"',
        'ps -p "$SP" >/dev/null 2>&1 && echo K:sup_alive=1 || echo K:sup_alive=0',
        'ps -p "$TP" >/dev/null 2>&1 && echo K:train_alive=1 || echo K:train_alive=0',
        'echo "K:stderr_bytes=$(stat -c %s $O/train.stderr.log 2>/dev/null || echo -1)"',
        'echo "K:ckpt=$(stat -c \'%s %Y\' $O/ckpt.pt 2>/dev/null)"',
        'echo "K:done=$(test -e $O/summary.json && echo 1 || echo 0)"',
        'echo "K:stopped=$(test -e $O/STOPPED_BY_PI.json && echo 1 || echo 0)"',
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
    kv["run"], kv["arm"] = run, arm
    with open(os.path.join(out_dir, "remote_state.json"), "w", encoding="utf-8") as fh:
        json.dump(kv, fh, indent=1)
    with open(os.path.join(out_dir, "stderr_tail.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body[:-1] if body.endswith("\n") else body)     # drop the separator `echo`


# ------------------------------------------------------------------ load ----
def load(metrics_path: str):
    rows = []
    with open(metrics_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                       # a torn last line while the trainer writes
    tr = [r for r in rows if "loss" in r and "eval_loss" not in r and isinstance(r.get("step"), int)]
    ev = [r for r in rows if "eval_loss" in r and isinstance(r.get("step"), int)]
    cd = [r for r in rows if any(k.startswith("cd_") and k != "cd_deferred" for k in r)]
    er = [r for r in rows if "eval_error" in r]
    return rows, tr, ev, cd, er


# ------------------------------------------------------------------ the 10 cm map ----
def map_key(stat: str, c: str, b: str) -> str:
    return f"eval_map_hires_{stat}_{c}_{b}"


def _cell(row, k):
    """('value', v) | ('undef', None) -- JSON null: undefined, never 0 | ('missing', None)."""
    if row is None or k not in row:
        return "missing", None
    v = row[k]
    if v is None:
        return "undef", None
    v = _num(v)
    return ("value", float(v)) if v is not None else ("missing", None)


def map_read(ev: list, cfg: dict | None = None) -> dict:
    """The 40-key contract on every eval row, the thin-class alarm, and the latest eval's cells.

    The alarm is the REGISTERED rule only (LOGGING_SPEC_MAP10 5.3; the Master Mind's ruling 2026-09-27):
    RED iff any class's 0_20 IoU <= 0.05 at any eval at or after step 5,000, or any eval row lacks a
    required key -- latched. 20-60 m at or below 0.05 from step 5,000 is AMBER (informative). Beyond
    60 m: values only. A null IoU (the class absent from GT and every prediction) is no value, so it is
    neither a breach nor a missing key."""
    out = {"n_evals": len(ev), "latest_step": ev[-1]["step"] if ev else None, "state": "pending",
           "alarm": False, "reasons": [], "cells": {}, "raw": {}, "share": {}, "n": {},
           "missing_latest": [], "rows_missing": [], "breaches": [], "breaches_latest": [],
           "amber": [], "amber_latest": [], "first_red_step": None, "first_red_what": None,
           "now_min_0_20": None, "undefined_0_20": [],
           "series_keys": [], "iou_count_mismatch": [], "iou_count_checked": 0, "decision_rule": None,
           "legacy_05m": False}
    mh = cfg.get("map_hires") if isinstance(cfg, dict) else None
    dr = mh.get("decision_rule") if isinstance(mh, dict) else None
    out["decision_rule"] = dr if isinstance(dr, str) else None
    out["series_keys"] = sorted({map_key("iou", c, b) for r in ev for c in MAP_CLASSES for b in MAP_BANDS
                                 if map_key("iou", c, b) in r})
    out["legacy_05m"] = any("map_iou_drivable" in r or "eval_map_iou_drivable" in r for r in ev)
    if not ev:
        out["reasons"].append("no in-run eval row yet: the 40-key contract cannot be read")
        return out
    for r in ev:
        miss = [(c, b) for c in MAP_CLASSES for b in MAP_BANDS if _cell(r, map_key("iou", c, b))[0] == "missing"]
        if miss:
            out["rows_missing"].append((r["step"], len(miss)))
        if r["step"] < THIN_FROM_STEP:
            continue
        for c in MAP_CLASSES:
            kind, v = _cell(r, map_key("iou", c, RED_BAND))
            if kind == "value" and v <= THIN_FLOOR:
                out["breaches"].append((r["step"], c, RED_BAND, v, THIN_FLOOR))
            for b in AMBER_BANDS:
                kind, v = _cell(r, map_key("iou", c, b))
                if kind == "value" and v <= THIN_FLOOR:
                    out["amber"].append((r["step"], c, b, v, THIN_FLOOR))
    last = ev[-1]
    eb = _num(last.get("eval_batches"))
    for c in MAP_CLASSES:
        for b in MAP_BANDS:
            out["cells"][(c, b)] = _cell(last, map_key("iou", c, b))
            out["raw"][(c, b)] = _cell(last, map_key("iouraw", c, b))
            out["share"][(c, b)] = _cell(last, map_key("lshare", c, b))
            n = _num(last.get(map_key("n", c, b)))
            out["n"][(c, b)] = (n * eb if (n is not None and eb) else n)
            # the IoU re-derived from the row's OWN counts -- independently of the logged ratio
            i, u = _num(last.get(map_key("inter", c, b))), _num(last.get(map_key("union", c, b)))
            kind, v = out["cells"][(c, b)]
            if i is not None and u is not None and kind != "missing":
                out["iou_count_checked"] += 1
                want = (i / u) if u > 0 else None
                if (want is None) != (v is None) or (want is not None and abs(want - v) > 1e-4):
                    out["iou_count_mismatch"].append((c, b, v, want))
    out["missing_latest"] = [map_key("iou", c, b) for (c, b), (k, _) in out["cells"].items() if k == "missing"]
    out["breaches_latest"] = [x for x in out["breaches"] if x[0] == last["step"]]
    out["amber_latest"] = [x for x in out["amber"] if x[0] == last["step"]]
    now = [(c, out["cells"][(c, RED_BAND)][1]) for c in MAP_CLASSES if out["cells"][(c, RED_BAND)][0] == "value"]
    out["now_min_0_20"] = min(now, key=lambda t: t[1]) if now else None
    out["undefined_0_20"] = [c for c in MAP_CLASSES if out["cells"][(c, RED_BAND)][0] == "undef"]
    # the latch: the FIRST eval that met either registered condition
    firsts = [(s, f"the eval row lacked {n} of the 40 keys") for s, n in out["rows_missing"][:1]]
    firsts += [(s, f"{c} 0–20 m IoU {v:.3f} ≤ {THIN_FLOOR:g}") for s, c, _b, v, _f in out["breaches"][:1]]
    if firsts:
        out["first_red_step"], out["first_red_what"] = min(firsts)
    out["alarm"] = bool(firsts)
    if out["alarm"]:
        out["reasons"].append(f"first red at step {out['first_red_step']:,}: {out['first_red_what']} (latched, "
                              "LOGGING_SPEC_MAP10 5.3: at any eval)")
    if out["now_min_0_20"] is not None:
        c, v = out["now_min_0_20"]
        armed = last["step"] >= THIN_FROM_STEP
        out["reasons"].append(f"now (step {last['step']:,}): lowest 0–20 m IoU {esc(c)} {v:.3f}"
                              + ((" — at or below 0.05" if v <= THIN_FLOOR else " — above 0.05") if armed
                                 else f" (the IoU floor arms at step {THIN_FROM_STEP:,})"))
    if out["missing_latest"]:
        out["reasons"].append(f"{len(out['missing_latest'])} of 40 keys ABSENT from the latest eval row "
                              f"(step {last['step']:,})")
    older = [(s, n) for s, n in out["rows_missing"] if s != last["step"]]
    if older:
        out["reasons"].append(f"{len(older)} earlier eval row(s) lacked keys (first: step {older[0][0]:,}, "
                              f"{older[0][1]} missing)")
    if out["breaches"]:
        steps = sorted({x[0] for x in out["breaches"]})
        out["reasons"].append(f"{len(out['breaches'])} 0–20 m reading(s) at or below 0.05 in {len(steps)} eval(s) "
                              f"since step {THIN_FROM_STEP:,}; {len(out['breaches_latest'])} in the latest")
    if out["undefined_0_20"]:
        out["reasons"].append("undefined at 0–20 m (absent from GT and every prediction): "
                              + ", ".join(esc(c) for c in out["undefined_0_20"]))
    out["state"] = ("UNAVAILABLE" if out["missing_latest"] else "ALARM" if out["alarm"]
                    else "AMBER" if out["amber_latest"] else "clear")
    return out


def map_signal(tr: list) -> dict:
    """LOGGING_SPEC_MAP10 sec. 5 item 2, on the TRAIN rows: per class, the loss share (sum over the bands
    of `map_hires_lshare_*`), the label-mass share (its labelled cells over all labelled cells) and the
    pull on its OWN cells, `gno` (L2 over the bands of `map_hires_gno_*`). A class with labelled cells
    and a pull of exactly 0 gets NO signal -- the spec's alarm. A class with no cells in a row is no
    evidence either way."""
    out = {"state": "UNAVAILABLE", "alarm": bool(tr), "classes": {}, "dead_latest": [], "latest_step": None,
           "missing_gno": list(MAP_CLASSES)}
    rows = [r for r in tr if any(f"map_hires_gno_{c}_{b}" in r or f"map_hires_lshare_{c}_{b}" in r
                                 for c in MAP_CLASSES for b in MAP_BANDS)]
    if not rows:
        return out
    out["state"], out["alarm"], out["latest_step"] = "clear", False, rows[-1]["step"]
    for c in MAP_CLASSES:
        ls, mass, gno, dead = [], [], [], []
        for r in rows:
            sh = [_num(r.get(f"map_hires_lshare_{c}_{b}")) for b in MAP_BANDS]
            ls.append(sum(sh) if all(v is not None for v in sh) else None)
            n_c = [_num(r.get(f"map_hires_n_{c}_{b}")) for b in MAP_BANDS]
            n_all = [_num(r.get(f"map_hires_n_{cc}_{b}")) for cc in MAP_CLASSES for b in MAP_BANDS]
            nc = sum(n_c) if all(v is not None for v in n_c) else None
            na = sum(n_all) if all(v is not None for v in n_all) else None
            mass.append(nc / na if (nc is not None and na) else None)
            g = [_num(r.get(f"map_hires_gno_{c}_{b}")) for b in MAP_BANDS]
            gv = math.sqrt(sum(v * v for v in g)) if all(v is not None for v in g) else None
            gno.append((r["step"], gv))
            if gv is not None and gv == 0 and nc:
                dead.append(r["step"])
        last_n = [_num(rows[-1].get(f"map_hires_n_{c}_{b}")) for b in MAP_BANDS]
        out["classes"][c] = {"lshare_ema": ema(ls)[-1], "mass": mass[-1], "gno": gno[-1][1], "gno_series": gno,
                             "dead_rows": dead, "n_latest": (sum(last_n) if all(v is not None for v in last_n) else None)}
        if dead and dead[-1] == rows[-1]["step"]:
            out["dead_latest"].append(c)
    missing = [c for c, d in out["classes"].items() if d["gno"] is None]
    if out["dead_latest"]:
        out["state"], out["alarm"] = "ALARM", True
    elif missing:
        out["state"], out["alarm"] = "UNAVAILABLE", True
    out["missing_gno"] = missing
    return out


def _hbin(v: float) -> int:
    for i, e in enumerate(HEAT_EDGES):
        if v <= e:
            return i
    return len(HEAT_EDGES)


def heatmap(m: dict, which: str, caption: str) -> str:
    """A class x band table of the latest eval. `which` = cells (the declared rule) | raw.
    ⛔ A key absent from the row prints NO key name (the launch gate reads the key literals as
    'this series is carried'), says UNAVAILABLE, and is never coloured as a value."""
    stat = "iou" if which == "cells" else "iouraw"
    brs = {(c, b) for (_, c, b, _, _) in m["breaches_latest"]} if which == "cells" else set()
    amb = {(c, b) for (_, c, b, _, _) in m["amber_latest"]} if which == "cells" else set()
    head = "".join(f'<th class="num">{band_label(b)}</th>' for b in MAP_BANDS)
    body = []
    for c in MAP_CLASSES:
        tds = []
        for b in MAP_BANDS:
            kind, v = m[which][(c, b)]
            n = m["n"].get((c, b))
            ntxt = f"n {fmt(n, 0)}" if n is not None else "n —"
            if kind == "missing":
                tds.append(f'<td class="hm unav" data-cell="{c} {b}" title="{esc(c)} · {band_label(b)} · the key is '
                           f'ABSENT from the latest eval row">UNAVAILABLE</td>')
            elif kind == "undef":
                k = map_key(stat, c, b)
                cl = "hm plain" if b in PLAIN_BANDS else "hm undef"
                tds.append(f'<td class="{cl}" data-cell="{c} {b}" data-key="{k}" title="{esc(c)} · '
                           f'{band_label(b)} · undefined: the class is absent from GT and every prediction '
                           f'(A8: reported, never dropped)">undefined<small>{ntxt}</small></td>')
            elif b in PLAIN_BANDS:
                # beyond 60 m: values only, no colour (the Master Mind's ruling 2026-09-27)
                k = map_key(stat, c, b)
                tds.append(f'<td class="hm plain" data-cell="{c} {b}" data-key="{k}" title="{esc(c)} · '
                           f'{band_label(b)} · IoU {v:.4f} · {ntxt} labelled cells · beyond 60 m: values only · {k}">'
                           f'{v:.3f}<small>{ntxt}</small></td>')
            else:
                k = map_key(stat, c, b)
                status = "breach" if (c, b) in brs else ("amber" if (c, b) in amb else "")
                flag = {"breach": '<span class="flag" aria-hidden="true">! </span>',
                        "amber": '<span class="flag" aria-hidden="true">~ </span>'}.get(status, "")
                why = {"breach": " · RED: at or below 0.05 at 0–20 m (registered)",
                       "amber": " · AMBER: at or below 0.05 at 20–60 m (informative)"}.get(status, "")
                tds.append(f'<td class="hm hb{_hbin(v)}{" " + status if status else ""}" data-cell="{c} {b}" '
                           f'data-key="{k}" title="{esc(c)} · {band_label(b)} · IoU {v:.4f} · {ntxt} labelled cells'
                           f'{why} · {k}">{flag}{v:.3f}<small>{ntxt}</small></td>')
        thin = ' <span class="muted">· thin</span>' if c in THIN_CLASSES else ""
        body.append(f'<tr><th scope="row">{esc(c)}{thin}</th>{"".join(tds)}</tr>')
    return (f'<figure class="hmfig"><figcaption><b>{esc(caption)}</b></figcaption><div style="overflow-x:auto">'
            f'<table class="hm"><tr><th>class</th>{head}</tr>{"".join(body)}</table></div></figure>')


def heat_legend() -> str:
    labs = ["≤ 0.05 (the floor)", "0.05–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "> 0.8"]
    sw = "".join(f'<span><i style="background:var(--h{i})"></i>{esc(t)}</span>' for i, t in enumerate(labs))
    return (f'<div class="hmlegend"><span>IoU, 0–60 m:</span>{sw}<span><i class="lg-plain"></i>beyond 60 m: values '
            'only</span><span><i class="lg-undef"></i>undefined: absent from GT and every prediction</span>'
            '<span><i class="lg-unav"></i>UNAVAILABLE: the key is absent (an alarm, never 0)</span>'
            '<span><i class="lg-breach"></i><b>!</b> RED: 0–20 m ≤ 0.05 (registered)</span>'
            '<span><i class="lg-amber"></i><b>~</b> AMBER: 20–60 m ≤ 0.05 (informative)</span></div>')


def share_table(m: dict) -> str:
    head = "".join(f'<th class="num">{band_label(b)}</th>' for b in MAP_BANDS)
    body = []
    for c in MAP_CLASSES:
        tds = []
        for b in MAP_BANDS:
            kind, v = m["share"][(c, b)]
            tds.append('<td class="num muted">UNAVAILABLE</td>' if kind == "missing" else
                       '<td class="num muted">undefined</td>' if kind == "undef" else
                       f'<td class="num">{100 * v:.2f} %</td>')
        body.append(f"<tr><td>{esc(c)}</td>{''.join(tds)}</tr>")
    return ('<div style="overflow-x:auto"><table><tr><th>class · loss share</th>' + head + "</tr>"
            + "".join(body) + "</table></div>")


# ------------------------------------------------------------------ the box heads ----
def bkey(h: str, s: str) -> str:
    return f"eval_{h}_{s}"


def box_read(ev: list) -> dict:
    """Per head: the latest eval's P0 readings, the A10 ratio alarm, and independent cross-checks."""
    lo, hi = BOX_RATIO_BAND
    out = {"heads": {}, "alarm": False, "any_p0": any(bkey(h, "conf_ratio") in r for r in ev for h in BOX_HEADS)}
    last = ev[-1] if ev else None
    for h in BOX_HEADS:
        d = {"state": "pending", "alarm": False, "armed": False, "reasons": [], "ratio": None, "checks": []}
        hist = [(r["step"], _num(r.get(bkey(h, "conf_ratio")))) for r in ev if bkey(h, "conf_ratio") in r]
        d["history"] = hist
        if last is None:
            d["reasons"].append("no in-run eval row yet")
        else:
            kind, r = _cell(last, bkey(h, "conf_ratio"))
            d["armed"] = armed = last["step"] >= BOX_ARM_STEP
            if kind == "value":
                d["ratio"] = r
            outside = kind == "value" and not (lo <= r <= hi)
            if not armed:
                # the Master Mind's ruling 2026-09-27: grey, with the value shown, until step 5,000
                d.update(state=BOX_WARMUP, alarm=False)
                shown = (f"ratio {r:.2f}" if kind == "value" else
                         "the ratio is ABSENT from the row" if kind == "missing" else "the ratio is null")
                d["reasons"].append(f"armed from step {BOX_ARM_STEP:,}; the latest eval is step {last['step']:,}: {shown}")
            elif kind == "missing":
                d.update(state="UNAVAILABLE", alarm=True)
                d["reasons"].append(f"the confidence ratio is ABSENT from the latest eval row (step {last['step']:,})")
            elif kind == "undef":
                d.update(state="UNDEFINED", alarm=True)
                d["reasons"].append("the ratio is null: no VIS-1 positives in the eval windows")
            else:
                d.update(state="ALARM" if outside else "in band", alarm=outside)
                if outside:
                    d["reasons"].append(f"confident / VIS-1 positives = {r:.2f}, outside [{lo}, {hi}] "
                                        f"(A10 15.3) at step {last['step']:,}")
                out_steps = [s for s, v in hist if v is not None and s >= BOX_ARM_STEP and not (lo <= v <= hi)]
                if out_steps and not outside:
                    d["reasons"].append(f"{len(out_steps)} earlier armed eval(s) outside the band (last: step "
                                        f"{out_steps[-1]:,})")
            g = {k: _num(last.get(bkey(h, k))) for k in
                 ("prec@gate", "rec@gate", "f1@gate", "ap2m", "auroc_matched", "auroc_objectness", "cls_acc_tp",
                  "cls_acc_tp_priorcorr", "centre_err_p50", "n_conf", "tp@gate", "n_pos", "n_ignore",
                  "n_dropped_hidden", "n_ignore_masked_slots", "n_windows", "conf_ratio_alarm", "calib_pr_gate",
                  "calib_prec", "calib_rec", "calib_n_pos", "calib_n_windows")}
            d["vals"] = g
            d["grid"] = {(t, b): _cell(last, bkey(h, f"det_map{t}_{b}")) for t, _ in BOX_THR for b, _ in BOX_BANDS}
            # cross-checks DERIVED INDEPENDENTLY of the logged ratios: the pooled counts in the same row
            nc, npos, tp = g["n_conf"], g["n_pos"], g["tp@gate"]
            if d["ratio"] is not None and nc is not None and npos:
                want = nc / npos
                if abs(want - d["ratio"]) > 1e-4 * max(1.0, abs(want)):
                    d["checks"].append(f"conf_ratio {d['ratio']:.5f} does not reproduce from n_conf / n_pos = {want:.5f}")
            if g["prec@gate"] is not None and tp is not None and nc:
                if abs(tp / nc - g["prec@gate"]) > 1e-4:
                    d["checks"].append(f"prec@gate {g['prec@gate']:.5f} does not reproduce from tp / n_conf = {tp / nc:.5f}")
            if g["rec@gate"] is not None and tp is not None and npos:
                if abs(tp / npos - g["rec@gate"]) > 1e-4:
                    d["checks"].append(f"rec@gate {g['rec@gate']:.5f} does not reproduce from tp / n_pos = {tp / npos:.5f}")
            # the trainer flags the BAND at every eval; it is held against this page's band reading, never
            # against the armed state (a warm-up ratio outside the band is flagged by both, and agrees)
            fl = g["conf_ratio_alarm"]
            if d["ratio"] is not None and fl is not None and bool(fl) != outside:
                d["checks"].append(f"the trainer's own conf_ratio_alarm flag reads {fl:g}; this page reads the ratio "
                                   f"{'outside' if outside else 'inside'} the band -- they disagree")
            if last.get("eval_calib_error"):
                d["checks"].append(f"the INFORMATIVE calibration pass failed: {last['eval_calib_error']}")
        out["heads"][h] = d
    out["alarm"] = any(d["alarm"] for d in out["heads"].values())
    return out


# ------------------------------------------------------------------ gradient reach (D3) ----
def ga_read(tr: list, cfg: dict | None) -> dict:
    """The DECLARED ga_* keys (config.json `grad_reach_logging`, written by refc_v3_train.py's
    `_grad_reach_declaration` before step 1) held against the latest train row: absent or exactly 0
    is an ALARM. Batch 3's `declared_vs_built.grad_unreachable` is shown as information."""
    out = {"declared": None, "source": "", "absent": [], "zero": [], "alarm": False, "per_key": {},
           "observed": [], "undeclared": [], "grad_unreachable": None, "rows_bad": 0, "first_bad_step": None,
           "latest_step": tr[-1]["step"] if tr else None}
    if not isinstance(cfg, dict):
        out["source"] = "config.json UNAVAILABLE"
    else:
        decl = cfg.get("grad_reach_logging")
        if not isinstance(decl, dict):
            out["source"] = "config.json carries no grad_reach_logging declaration (a pre-D3 trainer)"
        elif not decl.get("declared"):
            out["declared"] = []
            out["source"] = "config.json declares NO gradient reach (declared: false)"
        else:
            out["declared"] = sorted(str(k) for k in (decl.get("keys") or []))
            out["source"] = (f"config.json grad_reach_logging: {len(out['declared'])} keys, "
                             f"{esc(decl.get('source', ''))}")
        dvb = cfg.get("declared_vs_built")
        if isinstance(dvb, dict) and "grad_unreachable" in dvb:
            out["grad_unreachable"] = dvb["grad_unreachable"]
    out["observed"] = sorted({k for r in tr for k in r if k.startswith("ga_")})
    decl = out["declared"] or []
    out["undeclared"] = [k for k in out["observed"] if k not in decl]
    last = tr[-1] if tr else None
    for k in decl:
        vals = [(r["step"], r.get(k, "ABSENT")) for r in tr]
        bad = [s for s, v in vals if v == "ABSENT" or _num(v) is None or _num(v) == 0]
        out["per_key"][k] = {"latest": (last.get(k, "ABSENT") if last else "ABSENT"),
                             "min": min((v for _, v in vals if _num(v) is not None), default=None),
                             "rows_bad": len(bad), "first_bad": bad[0] if bad else None,
                             "series": [(s, _num(v)) for s, v in vals if v != "ABSENT"]}
    if last is not None:
        for k in decl:
            v = last.get(k, "ABSENT")
            if v == "ABSENT" or _num(v) is None:
                out["absent"].append(k)
            elif _num(v) == 0:
                out["zero"].append(k)
        bad_rows = [r["step"] for r in tr if any(k not in r or _num(r.get(k)) in (None, 0) for k in decl)]
        out["rows_bad"] = len(bad_rows)
        out["first_bad_step"] = bad_rows[0] if bad_rows else None
    out["alarm"] = (out["declared"] is None or out["declared"] == [] or bool(out["absent"])
                    or bool(out["zero"]) or last is None)
    return out


def sparkline(series, w=150, h=26) -> str:
    """log10 of a per-row gradient magnitude; an exact 0 is a red tick on the baseline."""
    pos = [(s, v) for s, v in series if v is not None and v > 0]
    zer = [s for s, v in series if v is not None and v == 0]
    if not series or (len(pos) < 2 and not zer):
        return '<span class="muted">—</span>'
    xs = [s for s, _ in series]
    x0, x1 = min(xs), max(xs)
    ly = [math.log10(v) for _, v in pos]
    lo, hi = (min(ly), max(ly)) if ly else (0.0, 1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5

    def sx(s):
        return 2 + (w - 4) * ((s - x0) / (x1 - x0) if x1 > x0 else 1.0)

    def sy(v):
        return h - 3 - (h - 6) * (math.log10(v) - lo) / (hi - lo)
    d = "M" + " L".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in pos) if len(pos) >= 2 else ""
    z = "".join(f'<rect class="z" x="{sx(s) - 1:.1f}" y="{h - 4}" width="2" height="4"/>' for s in zer)
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" role="img" aria-label="log10 trend, {len(zer)} exact zeros">'
            f'{"<path d=" + chr(34) + d + chr(34) + "/>" if d else ""}{z}</svg>')


# ------------------------------------------------------------------ NEW-1 residual prior ----
PRIOR_KEY_RE = re.compile(r"(^|_)(residual|prior)(_|$)")


def prior_read(cfg: dict | None, rows: list) -> dict:
    out = {"stamp": None, "argv_mode": None, "keys": [], "latest": {}}
    if isinstance(cfg, dict):
        st = (cfg.get("seams") or {}).get("residual_prior") if isinstance(cfg.get("seams"), dict) else None
        out["stamp"] = st if isinstance(st, dict) else None
        out["argv_mode"] = _argv_val(cfg.get("argv"), "--residual-prior")
    keys = sorted({k for r in rows for k in r if PRIOR_KEY_RE.search(k)})
    out["keys"] = keys
    for k in keys:
        last = [r for r in rows if k in r]
        out["latest"][k] = (last[-1].get("step"), last[-1][k]) if last else None
    return out


# ------------------------------------------------------------------ NavSim + battery ----
def stamp_for(step: int) -> str:
    labs = FACTS["segments"]
    if len(labs) < 2:
        return "as launched"
    i = min(sum(1 for s in SWITCH_STEPS if step > s), len(labs) - 1)
    return f"{labs[i][0]}: {labs[i][1]}"


def _steps_with(root, marker):
    out = []
    if root and os.path.isdir(root):
        for name in os.listdir(root):
            m = re.fullmatch(r"step(\d+)", name)
            if m and os.path.isfile(os.path.join(root, name, marker)):
                out.append(int(m.group(1)))
    return sorted(out)


def _count_guard(split, s):
    """None when the summary's sample counts are the published split's, or when the summary is a DESIGNED
    navtest token subset (it names its token file and carries its own n). Otherwise the refusal text."""
    if split == "navtest":
        got, want = s.get("n_tokens"), EXPECTED_N["navtest"]["n_tokens"]
        if s.get("tokens_subset"):
            return None if isinstance(got, int) and got > 0 else "UNAVAILABLE — a token subset without its count"
        return None if got == want else f"UNAVAILABLE — count guard FAIL ({_cnt(got)} of {want:,})"
    a1 = (s.get("arms") or {}).get(NAVSIM_ARM) or {}
    bad = [f"{k} {_cnt(a1.get(k))} of {v:,}" for k, v in EXPECTED_N[split].items() if a1.get(k) != v]
    return "UNAVAILABLE — count guard FAIL (" + "; ".join(bad) + ")" if bad else None


def navsim_read():
    """Per checkpoint and split: the headline KPI, its controls, the paired read vs STOP and the bar
    verdict -- or the split's status ("running" / "not run") when nothing is banked for it, or an
    UNAVAILABLE refusal when the banked summary's counts are not the published split's."""
    if not EVAL_PKG:
        return {}, {}
    ms = os.path.join(EVAL_PKG, "navsim", "raw", "milestones")
    live = os.path.join(EVAL_LIVE, "navsim", "raw", "milestones") if EVAL_LIVE else None
    A = NAVSIM_ARM
    res, published = {}, {}
    for step in _steps_with(ms, "BARS.json"):
        d = os.path.join(ms, f"step{step}")
        bars = (_jload(os.path.join(d, "BARS.json")) or {}).get("bars", {})
        pub = (bars.get("navtest") or {}).get("stretch_published") if isinstance(bars.get("navtest"), dict) else None
        if isinstance(pub, dict) and pub:
            published = pub                          # the banked reference text, read, never typed
        row = {}
        for split in ("navtest", "navhard", "warmup"):
            s = _jload(os.path.join(d, f"summary_{split}.json"))
            b = bars.get(split, {}) if isinstance(bars.get(split), dict) else {}
            if not s or b.get("status") == "UNAVAILABLE":
                ld = os.path.join(live, f"step{step}") if live else None
                started = bool(ld) and any(os.path.exists(os.path.join(ld, f"{p}_{split}")) for p in ("bridge", "scores"))
                # without a named live lane this page cannot tell "running" from "not run": it says so
                row[split] = {"status": "running" if started else ("not run" if live else "not banked")}
                continue
            refusal = _count_guard(split, s)
            if refusal:
                row[split] = {"status": refusal}
                continue
            arms = s.get("arms", {})
            if split == "navtest":
                def pd(a):
                    return _num((arms.get(a) or {}).get("PDMS"))
                a1 = arms.get(A) or {}
                iv = a1.get("interval") or {}
                pv = ((s.get("pairs") or {}).get(f"{A}__minus__STOP") or {}).get("interval") or {}
                row[split] = {"status": "ok", "metric": "PDMS (×100)", "value": pd(A),
                              "lo": _num(iv.get("lo")) and iv["lo"] * 100, "hi": _num(iv.get("hi")) and iv["hi"] * 100,
                              "STOP": pd("STOP"), "CV": pd("CV"), "HUMAN": pd("HUMAN"), "refcv6": pd("R6_A1"),
                              "d_stop": _num(pv.get("delta")) and pv["delta"] * 100,
                              "d_lo": _num(pv.get("lo")) and pv["lo"] * 100, "d_hi": _num(pv.get("hi")) and pv["hi"] * 100,
                              "n": s.get("n_tokens"), "verdict": b.get("verdict"),
                              "sub": {a: {k: _num((arms.get(a) or {}).get(k)) for k in ("NC", "DAC", "TTC", "EP", "C", "DDC", "PDMS")}
                                      for a in (A, "STOP", "CV", "HUMAN", "R6_A1") if a in arms}}
            elif split == "navhard":
                def ep(a):
                    return _num((arms.get(a) or {}).get("official_two_stage_EPDMS"))
                pv = b.get("paired_vs_STOP") or {}
                row[split] = {"status": "ok", "metric": "two-stage EPDMS", "value": ep(A),
                              "seed1": ep(f"{A}_s1"), "STOP": ep("STOP_zero"), "CV": ep("CV_official"),
                              "ECHO": ep("ECHO_ha0_ext"), "d_stop": _num(pv.get("delta")),
                              "d_lo": _num(pv.get("lo")), "d_hi": _num(pv.get("hi")),
                              "n": (arms.get(A) or {}).get("n_stage2"), "verdict": b.get("verdict")}
            else:
                v = b.get("values") or {}
                row[split] = {"status": "ok", "metric": "S2-EPDMS-u", "value": _num(v.get(A)),
                              "STOP": _num(v.get("STOP_zero")), "CV": _num(v.get("CV_official")),
                              "ECHO": _num(v.get("ECHO_ha0_ext")), "margin": _num(b.get("margin")),
                              "floor": _num(b.get("seed_floor")), "interval": b.get("interval"),
                              "n": (arms.get(A) or {}).get("n_stage2"), "verdict": b.get("verdict")}
        res[step] = row
    return res, published


def battery_read():
    """The battery's pre-registered bars per banked tag (step<N>), both inference seeds."""
    if not EVAL_PKG:
        return {}
    root = os.path.join(EVAL_PKG, "battery", "raw")
    res = {}
    for step in _steps_with(root, "battery_summary.json"):
        s = _jload(os.path.join(root, f"step{step}", "battery_summary.json")) or {}
        bars = []
        for b in s.get("bars", []):
            seeds = b.get("per_inference_seed") or {}
            bars.append({"id": b.get("id"), "statement": b.get("statement"), "verdict": b.get("verdict"),
                         "seeds": {k: (_num(v.get("delta")), _num(v.get("lo")), _num(v.get("hi")), v.get("separated"),
                                       v.get("n_windows"), v.get("n_episodes")) for k, v in seeds.items()}})
        if bars:
            res[step] = {"bars": bars}
    return res


def _verdict(v):
    s = str(v or "")
    rest = lambda k: (f' <span class="muted">{esc(s[k:].strip())}</span>' if s[k:].strip() else "")
    if s.startswith("PASS"):                    # a qualifier ("SUBSET — not the published split") is KEPT
        return '<span class="chip good verdict"><i></i>bar passed</span>' + rest(4)
    if s.startswith("FAIL"):
        return '<span class="chip crit verdict"><i></i>bar failed</span>' + rest(4)
    return esc(v or "—")


def navsim_html(ns):
    if not ns:
        where = (f"under <code>{esc(EVAL_PKG)}</code>" if EVAL_PKG else
                 "because no refcv7 eval package is configured (<code>$REFCV7_EVAL_PKG</code>)")
        return (f'<p class="muted">No banked NavSim milestone was readable {where} — nothing is shown rather '
                'than a guess.</p>'), []
    steps = sorted(ns)
    ctrl = {}
    for st_ in reversed(steps):                     # the controls are banked floors: take the latest reading
        for split, r in ns[st_].items():
            if r.get("status") == "ok" and split not in ctrl:
                ctrl[split] = r

    def cell(r, split):
        if r.get("status") != "ok":
            return f'<td class="muted">{esc(r.get("status"))}</td>'
        if split == "navtest":
            v = f'<b>{fmt(r["value"], 2)}</b> [{fmt(r["lo"], 2)}, {fmt(r["hi"], 2)}]'
            d = f'vs STOP {fmt(r["d_stop"], 2)} [{fmt(r["d_lo"], 2)}, {fmt(r["d_hi"], 2)}]'
        elif split == "navhard":
            v = f'<b>{fmt(r["value"], 4)}</b> (seed 1: {fmt(r.get("seed1"), 4)})'
            d = f'vs STOP {fmt(r["d_stop"], 4)} [{fmt(r["d_lo"], 4)}, {fmt(r["d_hi"], 4)}]'
        else:
            v = f'<b>{fmt(r["value"], 4)}</b>'
            d = f'margin {fmt(r["margin"], 4)} (seed floor {fmt(r["floor"], 4)}); no interval: {esc(r.get("interval") or "—")}'
        nd = 2 if split == "navtest" else 4
        n = r.get("n")
        nn = f'n = {n:,}' if isinstance(n, int) else "n —"
        return (f'<td class="num">{v}<br><span class="muted">{d}</span><br><span class="muted">{nn} · STOP on '
                f'these samples {fmt(r.get("STOP"), nd)}</span><br>{_verdict(r.get("verdict"))}</td>')
    head = "".join(f'<th class="num">step {s:,}<br><span class="muted">{esc(stamp_for(s))}</span></th>' for s in steps)
    lab = {"navtest": "navtest · PDMS (×100)", "navhard": "navhard · official two-stage EPDMS",
           "warmup": "warmup · S2-EPDMS-u"}
    rows = []
    for split in ("navtest", "navhard", "warmup"):
        c = ctrl.get(split, {})
        nd = 2 if split == "navtest" else 4
        rows.append(f'<tr><td>{lab[split]}</td>' + "".join(cell(ns[s].get(split, {"status": "not run"}), split) for s in steps)
                    + f'<td class="num">{fmt(c.get("STOP"), nd)}</td><td class="num">{fmt(c.get("CV"), nd)}</td>'
                    f'<td class="num">{fmt(c.get("ECHO"), nd)}</td><td class="num">{fmt(c.get("HUMAN"), nd)}</td></tr>')
    table = ('<div style="overflow-x:auto"><table><tr><th>split · metric</th>' + head
             + '<th class="num">STOP</th><th class="num">CV</th><th class="num">ECHO</th><th class="num">human</th></tr>'
             + "".join(rows) + '</table></div>')
    subs = []
    names = {NAVSIM_ARM: "refcv7", "STOP": "STOP", "CV": "CV", "HUMAN": "human", "R6_A1": "refcv6"}
    for s in steps:
        r = ns[s].get("navtest", {})
        if r.get("status") != "ok" or not r.get("sub"):
            continue
        subs.append(f'<h3>navtest PDMS sub-scores at step {s:,}</h3><div style="overflow-x:auto"><table><tr><th>arm</th>'
                    + "".join(f'<th class="num">{k}</th>' for k in ("NC", "DAC", "TTC", "EP", "C", "DDC", "PDMS"))
                    + "</tr>" + "".join(
                        f'<tr><td>{names.get(a, a)}</td>' + "".join(
                            f'<td class="num">{"<b>" if k == "PDMS" else ""}{fmt(v.get(k), 2)}{"</b>" if k == "PDMS" else ""}</td>'
                            for k in ("NC", "DAC", "TTC", "EP", "C", "DDC", "PDMS")) + "</tr>"
                        for a, v in r["sub"].items()) + "</table></div>")
    tiles = []
    for split, unit in (("navtest", 2), ("navhard", 4), ("warmup", 4)):
        have = [(s, ns[s][split]) for s in steps if ns[s].get(split, {}).get("status") == "ok"]
        if have:
            s, r = have[-1]
            tiles.append(f'<div class="tile"><b>{fmt(r["value"], unit)}</b><span>{esc(lab[split].split(" · ")[0])} '
                         f'{esc(r["metric"])} at step {s:,} · STOP {fmt(r.get("STOP"), unit)}</span></div>')
    return ('<div class="tiles">' + "".join(tiles) + "</div>" + table + "".join(subs)), steps


def battery_html(bt):
    if not bt:
        return ""
    out = []
    for s in sorted(bt):
        rows = "".join(
            f'<tr><td><b>{esc(x["id"])}</b><br><span class="muted">{esc(x["statement"])}</span></td>'
            + "".join(f'<td class="num">{fmt(v[0], 4)} [{fmt(v[1], 4)}, {fmt(v[2], 4)}]'
                      f'{" sep" if v[3] else ""}</td>' for _, v in sorted(x["seeds"].items()))
            + f'<td>{_verdict(x["verdict"])}</td></tr>' for x in bt[s]["bars"])
        out.append(f'<h3>Four-family battery (T1) at step {s:,} <span class="muted">— {esc(stamp_for(s))}</span></h3>'
                   '<div style="overflow-x:auto"><table><tr><th>bar</th><th class="num">inference seed 0</th>'
                   '<th class="num">inference seed 1</th><th>verdict</th></tr>' + rows + "</table></div>")
    return "".join(out)


# ------------------------------------------------------------------ charts ----
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


def chart(cid, title, sub, ylab, series, xmax, note=None, refs=(), ylo=None, yhi=None, empty=None):
    """series: list of dict(name, xs, ys, color, kind='line'|'dots'|'dashed', key=None). `key` is the
    metrics key the series is read from, carried in the hover payload."""
    ally = [y for s in series for y in s["ys"]
            if y is not None and not (isinstance(y, float) and math.isnan(y))]
    if not ally:
        return (f'<figure class="chart"><figcaption><b>{esc(title)}</b><span>{sub}</span>'
                f'</figcaption><p class="note">{empty or "No data yet."}</p></figure>')
    lo, hi = min(ally), max(ally)
    lo = min(lo, 0) if 0 < lo < 0.25 * hi else lo
    for v, _ in refs:
        lo, hi = min(lo, v), max(hi, v)
    if ylo is not None:
        lo = ylo
    if yhi is not None:
        hi = max(yhi, hi)
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
        item = {"name": s["name"], "xs": [round(v, 1) for v in s["xs"]],
                "ys": [None if (v is None or (isinstance(v, float) and math.isnan(v)))
                       else round(v, 4) for v in s["ys"]]}
        if s.get("key"):
            item["key"] = s["key"]
        data.append(item)
    g.append(f'<line class="xh" x1="0" y1="{Y0}" x2="0" y2="{Y1}" style="display:none"/>'
             f'<rect class="hit" x="{X0}" y="{Y0}" width="{X1-X0}" height="{Y1-Y0}" fill="transparent"/>')
    legend = "".join(f'<span class="lg"><i style="background:var(--{s["color"]})"></i>{esc(s["name"])}</span>'
                     for s in series if len(series) > 1)
    note_html = f'<p class="note">{note}</p>' if note else ""
    payload = json.dumps({"xmax": xmax, "series": data}).replace("<", "\\u003c")
    return (f'<figure class="chart"><figcaption><b>{esc(title)}</b><span>{sub}</span></figcaption>'
            f'<span class="ylab">{esc(ylab)}</span>'
            f'<svg class="plot" viewBox="0 0 560 260" data-chart="{cid}" role="img" aria-label="{esc(title)}">{"".join(g)}</svg>'
            f'<div class="legend">{legend}</div><div class="tip" data-for="{cid}"></div>'
            f'<script type="application/json" data-series="{cid}">{payload}</script>{note_html}</figure>')


# ------------------------------------------------------------------ page CSS (refcv7 additions) ----
EXTRA_CSS = """
:root{--s5:#e87ba4;--h0:#cde2fb;--h1:#9ec5f4;--h2:#6da7ec;--h3:#3987e5;--h4:#1c5cab;--h5:#104281;
 --ht0:#0b0b0b;--ht1:#0b0b0b;--ht2:#0b0b0b;--ht3:#ffffff;--ht4:#ffffff;--ht5:#ffffff;--undef:#e6e4de}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--s5:#d55181;--h0:#0d366b;--h1:#184f95;
 --h2:#256abf;--h3:#2a78d6;--h4:#6da7ec;--h5:#9ec5f4;--ht0:#ffffff;--ht1:#ffffff;--ht2:#ffffff;--ht3:#ffffff;
 --ht4:#0b0b0b;--ht5:#0b0b0b;--undef:#2e2e2c}}
:root[data-theme="dark"]{--s5:#d55181;--h0:#0d366b;--h1:#184f95;--h2:#256abf;--h3:#2a78d6;--h4:#6da7ec;
 --h5:#9ec5f4;--ht0:#ffffff;--ht1:#ffffff;--ht2:#ffffff;--ht3:#ffffff;--ht4:#0b0b0b;--ht5:#0b0b0b;--undef:#2e2e2c}
figure.hmfig{margin:6px 0 10px}figure.hmfig figcaption b{font-size:14px}
table.hm{border-collapse:separate;border-spacing:3px}table.hm th{border:0}
td.hm{min-width:76px;text-align:center;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
 border:0;border-radius:3px;padding:5px 6px;font-size:13px}
td.hm small{display:block;font-size:10.5px;opacity:.85}
td.hb0{background:var(--h0);color:var(--ht0)}td.hb1{background:var(--h1);color:var(--ht1)}
td.hb2{background:var(--h2);color:var(--ht2)}td.hb3{background:var(--h3);color:var(--ht3)}
td.hb4{background:var(--h4);color:var(--ht4)}td.hb5{background:var(--h5);color:var(--ht5)}
td.hm.undef{background:var(--undef);color:var(--ink2)}
td.hm.unav{background:transparent;color:var(--crit);outline:2px dashed var(--crit);outline-offset:-3px;font-weight:600;font-size:11px}
td.hm.breach{outline:2px solid var(--crit);outline-offset:-3px}.flag{font-weight:700}
td.hm.amber{outline:2px solid var(--warn);outline-offset:-3px}
td.hm.plain{background:transparent;color:var(--ink);box-shadow:inset 0 0 0 1px var(--line)}
.hmlegend i.lg-plain{box-shadow:inset 0 0 0 1px var(--line)}
.hmlegend i.lg-breach{outline:2px solid var(--crit);outline-offset:-2px}
.hmlegend i.lg-amber{outline:2px solid var(--warn);outline-offset:-2px}
.chip.info i{background:var(--muted)}
.hmlegend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--ink2);margin:4px 0 10px}
.hmlegend i{display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:5px}
.hmlegend i.lg-undef{background:var(--undef)}.hmlegend i.lg-unav{outline:2px dashed var(--crit);outline-offset:-2px}
.alarms{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:10px 0 4px}
.alarm{border:1px solid var(--line);border-left:5px solid var(--good);border-radius:6px;padding:10px 12px;background:var(--surface)}
.alarm.crit{border-left-color:var(--crit)}.alarm.warn{border-left-color:var(--warn)}.alarm.info{border-left-color:var(--muted)}
.alarm h3{margin:0 0 2px;font-size:12.5px}
.alarm .state{display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:18px;font-weight:600;margin:2px 0}
.alarm ul{margin:4px 0 0;padding-left:18px;font-size:12.5px;color:var(--ink2)}.alarm li{margin:0 0 3px}
.alarm .src{display:block;font-size:11.5px;color:var(--muted);margin-top:6px}
code{overflow-wrap:anywhere}
svg.spark{width:150px;height:26px;display:block}.spark path{fill:none;stroke:var(--s1);stroke-width:1.5}.spark .z{fill:var(--crit)}
.grid4{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px 18px}
.grid2.fill{grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
@media (max-width:640px){.wrap{padding:20px 16px 48px}.grid2,.grid4{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}}
"""


def _tile(kind, title, state, reasons, src, tid=None):
    cls = {"crit": "crit", "warn": "warn", "good": "good", "info": "info"}[kind]
    lis = "".join(f"<li>{r}</li>" for r in reasons)
    i = f' id="{tid}"' if tid else ""
    return (f'<div class="alarm {cls}"{i}><h3>{esc(title)}</h3><span class="state">{esc(state)}</span>'
            f'{"<ul>" + lis + "</ul>" if lis else ""}<span class="src">{src}</span></div>')


# ------------------------------------------------------------------ build ----
def build(watch_dir: str | None = None, metrics_path: str | None = None, arm: str | None = None):
    """(page, summary) from the artifacts in `watch_dir` (default L). `metrics_path` builds from ONE
    metrics.jsonl (config.json beside it when present); every artifact that is absent renders as
    UNAVAILABLE rather than stopping the build."""
    global SWITCH_STEPS, DEATH_STEPS
    arm = arm or ARM
    wd = watch_dir or (os.path.dirname(os.path.abspath(metrics_path)) if metrics_path else L)
    mp = metrics_path or os.path.join(wd, "metrics.jsonl")
    rows, tr, ev, cd, er = load(mp)
    if not tr:
        raise SystemExit("ZZWATCH-NO-TRAINING-ROWS")
    st = _jload(os.path.join(wd, "remote_state.json"))
    state_read = isinstance(st, dict) and "now" in st
    st = st if state_read else {}
    cfg = _jload(os.path.join(wd, "config.json"))
    cfg = cfg if isinstance(cfg, dict) else None
    sp_ = os.path.join(wd, "sup.log")
    sup_log = open(sp_, encoding="utf-8", errors="replace").read() if os.path.exists(sp_) else None
    argv = cfg.get("argv") if cfg else None
    total = TOTAL_SPEC
    steps_argv = _argv_val(argv, "--steps")
    total_note = ""
    if steps_argv is not None:
        try:
            total = int(steps_argv)
        except ValueError:
            total_note = f"config.json --steps {steps_argv!r} is not an integer"
        if total != TOTAL_SPEC:
            total_note = (f"the run's own --steps {total:,} differs from SPEC_REFCV7 sec. 4's {TOTAL_SPEC:,}; "
                          "the ETA uses the run's")
    log_every = int(_argv_val(argv, "--log-every") or 50)
    save_every = _argv_val(argv, "--save-every")
    eval_every = _argv_val(argv, "--eval-every")

    # ---------------------------------------------------------- segments --
    segs, cur = [], [tr[0]]
    for a, b in zip(tr, tr[1:]):
        if (b.get("elapsed_s") or 0) < (a.get("elapsed_s") or 0):
            segs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    segs.append(cur)
    planned = len(FACTS["segments"]) - 1
    SWITCH_STEPS = [s[0]["step"] - log_every for s in segs[1:1 + planned]]
    DEATH_STEPS = [s[0]["step"] - log_every for s in segs[1 + planned:]]
    unplanned = max(0, len(segs) - 1 - planned)

    last = tr[-1]
    step_now = last["step"]
    pct = 100.0 * step_now / total
    seg_now = segs[-1]
    k = min(10, len(seg_now) - 1)
    if k >= 1 and _num(seg_now[-1].get("elapsed_s")) is not None and _num(seg_now[-1 - k].get("elapsed_s")) is not None:
        a, b = seg_now[-1 - k], seg_now[-1]
        pace = (b["elapsed_s"] - a["elapsed_s"]) / max(1, b["step"] - a["step"])
    elif len(seg_now) == 1 and (_num(last.get("elapsed_s")) or 0) > 0:
        base = segs[-2][-1]["step"] if len(segs) > 1 else 0
        pace = last["elapsed_s"] / max(1, last["step"] - base)
    else:
        pace = None
    # per-row pace inside each segment; the WARM median drops the intervals that hold an eval/checkpoint
    pxs, pys, warm = [], [], []
    ev_int = int(eval_every or EVAL_EVERY_SPEC)
    for sg in segs:
        for a, b in zip(sg, sg[1:]):
            ds = b["step"] - a["step"]
            if ds <= 0 or b.get("elapsed_s") is None or a.get("elapsed_s") is None:
                continue
            p = (b["elapsed_s"] - a["elapsed_s"]) / ds
            pxs.append(b["step"])
            pys.append(p)
            if not any(a["step"] <= m < b["step"] for m in range((a["step"] // ev_int) * ev_int, b["step"] + 1, ev_int)):
                warm.append(p)
    warm_med = sorted(warm)[len(warm) // 2] if warm else None
    now_utc = (datetime.fromtimestamp(int(st["now"]), timezone.utc) if state_read
               else datetime.now(timezone.utc))
    eta_s = max(0, total - step_now) * pace if pace is not None else None
    finish = berlin(now_utc + timedelta(seconds=eta_s)) if eta_s is not None else None
    now_b = berlin(now_utc)
    pace_above = pace is not None and pace > PACE_PI_LINE_S

    # ------------------------------------------------------------ status --
    sup_alive, tr_alive = st.get("sup_alive") == "1", st.get("train_alive") == "1"
    stderr_b = int(st.get("stderr_bytes", "-1") or -1) if state_read else -1
    token_re = re.compile(r"ZZ" + re.escape(arm) + r"-(-?\d+)-(\d+)-([^Z]*?)-(\d+)ZZ")
    gate_re = re.compile(r"ZZGATE(OK|REFUSED)-" + re.escape(arm) + r"-(\d+)ZZ")
    toks = list(token_re.finditer(sup_log or ""))
    token = toks[-1].group(0) if toks else None
    n_err = launch_no = None
    token_split = False
    field = ""
    if toks:
        field = toks[-1].group(3)            # "U" = the supervisor could not READ its stderr
        token_split = any(c.isspace() for c in field)
        ints = re.findall(r"\d+", field)
        n_err = int(ints[0]) if ints else None
        launch_no = int(toks[-1].group(4))
    token_ok = sup_log is not None and (n_err is not None or f"ZZ{arm}-" not in sup_log)
    gates = [(m.group(1), int(m.group(2))) for m in gate_re.finditer(sup_log or "")]
    gate_ok = sum(1 for g in gates if g[0] == "OK")
    gate_ref = [n for g, n in gates if g == "REFUSED"]
    ev_lines = re.findall(r"^\[sup:" + re.escape(arm) + r"\] (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) lock acquired",
                          sup_log or "", flags=re.M)
    launch_b = (berlin(datetime.strptime(ev_lines[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
                if ev_lines else None)
    sp = os.path.join(wd, "stderr_tail.log")
    stderr_txt = open(sp, encoding="utf-8", errors="replace").read() if os.path.exists(sp) else ""
    stderr_read_ok = state_read and stderr_b >= 0 and len(stderr_txt.encode("utf-8")) >= min(stderr_b, STDERR_CAP)
    stderr_truncated = stderr_b > STDERR_CAP
    stderr_lines = [ln for ln in stderr_txt.splitlines() if ln.strip()]
    diag = FACTS["stderr_diagnosed"]
    stderr_undiag = [ln for ln in stderr_lines if ln not in diag]
    n_err_client = sum(1 for ln in stderr_lines if ERR_PAT.search(ln))
    stderr_ok = stderr_read_ok and not stderr_truncated and not stderr_undiag and n_err_client == 0
    launches_total = (sup_log or "").count("lock acquired")
    trainer_launches = len(re.findall(r" launch #\d+ of \d+: ", sup_log or ""))
    done = st.get("done") == "1"
    stopped = st.get("stopped") == "1"
    ckpt = str(st.get("ckpt", "")).split()
    ckpt_gb = int(ckpt[0]) / 1e9 if len(ckpt) == 2 else None
    ckpt_age_min = (int(st["now"]) - int(ckpt[1])) / 60 if len(ckpt) == 2 and state_read else None
    mem_peak = max(((r.get("cuda_max_mem_gb") or 0) for r in tr), default=0)
    slots, comp = last.get("trunk_frame_slots"), last.get("trunk_frames_computed")
    dd_frac = comp / slots if slots and comp is not None else None
    ev_windows = sorted({r.get("eval_windows") for r in ev if r.get("eval_windows") is not None})
    ev_w = ev_windows[0] if ev_windows else "—"
    cd_last = cd[-1] if cd else None
    cd_lag = (step_now - cd_last["step"]) if cd_last else None
    readings_ok = cd_last is not None and cd_lag is not None and cd_lag <= 60
    e_first, e_last = (ev[0], ev[-1]) if ev else (None, None)
    eh = ((cfg or {}).get("seams", {}) or {}).get("ego_history") or {}
    levers = (cfg or {}).get("trunk_memory_levers") or {}
    cas_rows = sum(1 for r in tr if "cascade" in r)

    m = map_read(ev, cfg)
    # refcv6's 0.5 m keys, on ANY row (train rows carry `map_iou_drivable`, eval rows its `eval_` twin)
    m["legacy_05m"] = any("map_iou_drivable" in r or "eval_map_iou_drivable" in r for r in rows)
    ms = map_signal(tr)
    bx = box_read(ev)
    ga = ga_read(tr, cfg)
    pr = prior_read(cfg, rows)

    def chip(ok, good, bad, warn=False, info=False):
        cls = "good" if ok else ("info" if info else "warn" if warn else "crit")
        return f'<span class="chip {cls}"><i></i>{esc(good if ok else bad)}</span>'

    def map_chip():
        if m["state"] == "pending":
            return chip(False, "", "map 10 cm: no eval yet", info=True)
        if m["missing_latest"]:
            return chip(False, "", f"map 10 cm: {len(m['missing_latest'])} keys UNAVAILABLE")
        if m["alarm"]:
            return chip(False, "", f"THIN-CLASS ALARM · first red at step {m['first_red_step']:,}")
        if m["amber_latest"]:
            return chip(False, "", f"map 10 cm: 20–60 m ≤ 0.05 on {len(m['amber_latest'])} cell(s), informative",
                        warn=True)
        return chip(True, "map 10 cm: 40/40 keys · 0–20 m above 0.05", "")

    def box_chip(h):
        d = bx["heads"][h]
        if d["state"] == "pending":
            return chip(False, "", f"{h}: no eval yet", info=True)
        if not d["armed"]:
            return chip(False, "", f"{h} warming up · ratio {fmt(d['ratio'], 2)}", info=True)
        if d["ratio"] is None:
            return chip(False, "", f"{h} ratio {d['state']}")
        return chip(not d["alarm"], f"{h} ratio {d['ratio']:.2f} in [0.5, 1.5]", f"{h} RATIO ALARM {d['ratio']:.2f}")
    pace_lab = "(marginal, incl. eval + ckpt)"

    learning = len(ev) >= 2 and (_num(e_last.get("eval_traj")) or 0) < (_num(e_first.get("eval_traj")) or 0)
    run_chip = (chip(tr_alive and sup_alive and not done and not stopped, "training",
                     "finished" if done else ("stopped by the PI" if stopped else "NOT RUNNING"), warn=done or stopped)
                if state_read else chip(False, "", "run state UNAVAILABLE (not pulled)", warn=True))
    chips = (run_chip
             + chip(learning or len(ev) < 2, "learning" if len(ev) >= 2 else ("first eval in" if ev else "no eval yet"),
                    "eval traj not falling", warn=True)
             + (chip(unplanned == 0 and token_ok and (n_err in (0, None)),
                     f"{unplanned} unplanned deaths · {planned} planned switch",
                     f"{unplanned} unplanned deaths · tracebacks "
                     f"{n_err if token_ok else 'NOT READ' if field.strip() == 'U' else 'UNPARSED'}")
                if sup_log is not None else chip(False, "", "supervisor log UNAVAILABLE", warn=True))
             + (chip(stderr_ok,
                     "stderr empty" if not stderr_lines else f"stderr: {len(stderr_lines)} line(s), all diagnosed",
                     ("stderr NOT READ" if not stderr_read_ok else "stderr past the read cap" if stderr_truncated
                      else f"stderr: {len(stderr_undiag)} UNDIAGNOSED line(s)"))
                if state_read else chip(False, "", "stderr UNAVAILABLE (not pulled)", warn=True))
             + chip(readings_ok, "conflict readings recording", "conflict readings MISSING")
             + (chip(not gate_ref and gate_ok > 0, f"gate MATCH ×{gate_ok}",
                     f"gate REFUSED launch #{gate_ref[-1]}" if gate_ref else "no gate verdict in the supervisor log",
                     warn=not gate_ref)
                if sup_log is not None else "")
             + map_chip()
             + chip(not ms["alarm"], "map signal: every class with cells is pulled",
                    ("MAP SIGNAL: no pull on " + ", ".join(ms["dead_latest"])) if ms["dead_latest"] else
                    "map signal UNAVAILABLE")
             + "".join(box_chip(h) for h in BOX_HEADS)
             + chip(not ga["alarm"], f"grad reach: {len(ga['declared'] or [])} declared keys, none dead",
                    ("grad reach declaration UNAVAILABLE" if ga["declared"] is None else
                     "grad reach: NOTHING declared" if not ga["declared"] else
                     f"GRAD REACH ALARM: {len(ga['absent']) + len(ga['zero'])} dead or absent"))
             + (chip(not pace_above,
                     f"pace {pace:.2f} s/step {pace_lab} ≤ {PACE_PI_LINE_S:.1f} · warm median {fmt(warm_med, 2)}",
                     f"pace {pace:.2f} s/step {pace_lab} > {PACE_PI_LINE_S:.1f} · warm median {fmt(warm_med, 2)}",
                     warn=True)
                if pace is not None else ""))

    # ------------------------------------------------------------ charts --
    xmax = max(step_now, 1000)
    tx = [r["step"] for r in tr]
    exs = [r["step"] for r in ev]

    def trs(key):
        return [_num(r.get(key)) for r in tr]

    def evs(key):
        return [_num(r.get(key)) for r in ev]

    c1 = chart("c1", "Total loss",
               f"train, per logged batch (EMA α=0.15) · held-out eval every {esc(eval_every or EVAL_EVERY_SPEC)} steps on {ev_w} fixed windows",
               "loss",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("loss")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_loss"), color="s2", kind="dots")], xmax,
               note="Loss mixes every supervised term; read the per-head charts for what moved. Rules mark "
                    "planned switches (blue) and unplanned relaunches (red).")
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
               "fraction of windows whose selected anchor is the GT-nearest of the 117 v0-conditioned anchors "
               "(residual on the prior under NEW-1)", "acc",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("anchor_acc")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_anchor_acc"), color="s2", kind="dots")], xmax,
               refs=[(CHANCE_ANCHOR, "chance 1/117")])
    c5 = chart("c5", "Tactical decoder v6 — lateral and longitudinal",
               "cross-entropy against the v8 labels on the true clock (FIX-2) · eval dots", "CE",
               [dict(name="lat train (EMA)", xs=tx, ys=ema(trs("tacv6_lat_ce")), color="s1"),
                dict(name="lon train (EMA)", xs=tx, ys=ema(trs("tacv6_lon_ce")), color="s4"),
                dict(name="eval lat", xs=exs, ys=evs("eval_tacv6_lat_ce"), color="s2", kind="dots"),
                dict(name="eval lon", xs=exs, ys=evs("eval_tacv6_lon_ce"), color="s3", kind="dots")], xmax)
    c6 = chart("c6", "Tactical decoder v6 — the 22-token goal set",
               "multi-label BCE over the tactical goal tokens · goal-confidence BCE", "BCE",
               [dict(name="goal BCE train (EMA)", xs=tx, ys=ema(trs("tacv6_goal_bce")), color="s1"),
                dict(name="confidence BCE train (EMA)", xs=tx, ys=ema(trs("tacv6_goal_conf_bce")), color="s4"),
                dict(name="eval goal BCE", xs=exs, ys=evs("eval_tacv6_goal_bce"), color="s2", kind="dots")], xmax)

    # the 10 cm map -- 8 charts x 5 bands = the 40 series, each carrying its key
    na_map = "UNAVAILABLE — no eval row carries this class's 10 cm IoU keys"
    mcharts = []
    for c in MAP_CLASSES:
        ser = []
        for i, b in enumerate(MAP_BANDS):
            key = map_key("iou", c, b)
            pts = [(r["step"], _num(r.get(key))) for r in ev if key in r]
            if not pts:
                continue
            n = m["n"].get((c, b))
            ser.append(dict(name=f"{band_label(b)} (n {fmt(n, 0) if n is not None else '—'})",
                            xs=[p[0] for p in pts], ys=[p[1] for p in pts], color=BAND_COLORS[i],
                            kind="dots", key=key))
        mcharts.append(chart(f"m_{c}", f"{c} — IoU at 10 cm", "pooled, per band · 0–20 m ≤ 0.05 is RED",
                             "IoU", ser, xmax, ylo=0.0, yhi=1.0, empty=na_map,
                             refs=[(THIN_FLOOR, "floor 0.05")] if ser else ()))
    cmh = chart("cmh", "Map (10 cm) — loss", "the 10 cm hard-label CE as optimised (map_hires) · train EMA vs eval",
                "CE", [dict(name="train (EMA)", xs=tx, ys=ema(trs("map_hires")), color="s1", key="map_hires"),
                       dict(name="eval", xs=exs, ys=evs("eval_map_hires"), color="s2", kind="dots", key="eval_map_hires")],
                xmax, empty="UNAVAILABLE — no row carries map_hires")

    # boxes
    bcharts = []
    for h in BOX_HEADS:
        r6 = REF6.get(h, {})
        refs = [(BOX_RATIO_BAND[0], "band 0.5"), (BOX_RATIO_BAND[1], "band 1.5")]
        if r6.get("ratio") is not None:
            refs.append((r6["ratio"], f"refcv6@38k {r6['ratio']:.2f} (vs trainer targets)"))
        rk = bkey(h, "conf_ratio")
        pts = [(r["step"], _num(r.get(rk))) for r in ev if rk in r]
        bcharts.append(chart(f"b_{h}_ratio", f"{h} — confident / VIS-1 positives",
                             "slots with σ ≥ 0.5 (IGNORE-matched excluded) per VIS-1 positive, pooled over the eval "
                             f"windows · A10 ALARM outside [0.5, 1.5], armed from step {BOX_ARM_STEP:,}", "ratio",
                             [dict(name="ratio", xs=[p[0] for p in pts], ys=[p[1] for p in pts], color="s1",
                                   kind="dots", key=rk)] if pts else [],
                             xmax, refs=refs if pts else (), empty="UNAVAILABLE — no eval row carries the P0 ratio"))
        qs = []
        for i, (kk, nm) in enumerate((("ap2m", "AP@2 m"), ("prec@gate", "precision @ 0.5"),
                                      ("rec@gate", "recall @ 0.5"), ("auroc_matched", "presence AUROC"))):
            key = bkey(h, kk)
            pts = [(r["step"], _num(r.get(key))) for r in ev if key in r]
            if pts:
                qs.append(dict(name=nm, xs=[p[0] for p in pts], ys=[p[1] for p in pts], color=BAND_COLORS[i],
                               kind="dots", key=key))
        qrefs = [(r6["ap2m"], f"refcv6@38k AP@2 m {r6['ap2m']:.2f}")] if (r6.get("ap2m") and qs) else []
        bcharts.append(chart(f"b_{h}_q", f"{h} — detection quality",
                             "AP@2 m BEV (pooled classes), precision and recall at the declared gate σ ≥ 0.5, "
                             "presence AUROC · VIS-1 positives, IGNORE as DontCare", "value",
                             qs, xmax, ylo=0.0, yhi=1.0, refs=qrefs,
                             empty="UNAVAILABLE — no eval row carries the P0 detection keys"))
    c8 = chart("c8", "3-D boxes — loss",
               "Hungarian-matched cuboid set loss (focal presence under A9 R1)", "loss",
               [dict(name="train (EMA)", xs=tx, ys=ema(trs("box3d")), color="s1"),
                dict(name="eval", xs=exs, ys=evs("eval_box3d"), color="s2", kind="dots")], xmax)
    c9 = chart("c9", "Agents — loss",
               "agent-slot classification and centre losses", "loss",
               [dict(name="class train (EMA)", xs=tx, ys=ema(trs("agent_cls")), color="s1"),
                dict(name="centre train (EMA)", xs=tx, ys=ema(trs("agent_centre")), color="s4"),
                dict(name="eval class", xs=exs, ys=evs("eval_agent_cls"), color="s2", kind="dots"),
                dict(name="eval centre", xs=exs, ys=evs("eval_agent_centre"), color="s3", kind="dots")], xmax)
    cdx = [r["step"] for r in cd]

    def cds(key):
        return [_num(r.get(key)) for r in cd]

    cmin = min((x for x in cds("cd_cos") if x is not None), default=0.0)
    c10 = chart("c10", "Gradient conflict on the trunk",
                "cosine between the planning gradient and the perception gradient, per reading · EMA", "cos",
                [dict(name="whole trunk", xs=cdx, ys=ema(cds("cd_cos"), 0.2), color="s1"),
                 dict(name="layer4", xs=cdx, ys=ema(cds("cd_stage_layer4_cos"), 0.2), color="s2"),
                 dict(name="fusion", xs=cdx, ys=ema(cds("cd_fuse_cos"), 0.2), color="s3"),
                 dict(name="stem", xs=cdx, ys=ema(cds("cd_stem_cos"), 0.2), color="s4")], xmax,
                refs=[(0.0, "orthogonal")], ylo=-1.0 if cmin < -0.5 else None,
                note="Below zero the two objectives pull the shared trunk apart; near zero they are independent.")
    c11 = chart("c11", "Learning rate", f"warmup then cosine to step {total:,} · ×1e4", "lr×1e4",
                [dict(name="lr ×1e4", xs=tx, ys=[(r.get("lr") or 0) * 1e4 for r in tr], color="s1")], xmax)
    c12 = chart("c12", "Pace", "seconds per step between consecutive logged rows (EMA α=0.15), including the "
                "eval and the checkpoint every 500 steps", "s/step",
                [dict(name="s/step (EMA)", xs=pxs, ys=ema(pys), color="s1")], xmax,
                refs=[(PACE_REF_S, f"refcv6 {PACE_REF_S} s/step"),
                      (PACE_PI_LINE_S, f"PI line {PACE_PI_LINE_S} s/step (+25 %)")],
                note="Shown, not decided: more than +25 % over refcv6's 6.4 s/step is the PI's reserved decision "
                     "(SPEC_REFCV7 6.2 item 5, 12 item 4).")

    # ---------------------------------------------------------- timeline --
    tl = ['<svg class="timeline" viewBox="0 0 900 96" role="img" aria-label="run segments">',
          '<rect x="0" y="30" width="900" height="26" fill="var(--grid)" rx="3"/>']
    starts = sorted([0] + SWITCH_STEPS + DEATH_STEPS)
    ends = starts[1:] + [step_now]
    for i, (s0, s1) in enumerate(zip(starts, ends)):
        cls = "seg switch" if (i > 0 and s0 in SWITCH_STEPS) else "seg"
        tl.append(f'<rect class="{cls}" x="{900*s0/total:.1f}" y="30" width="{max(1.5, 900*(s1-s0)/total):.1f}" height="26" rx="2"/>')
    tl.append(f'<text class="tick" x="{min(900*step_now/total+6, 640):.1f}" y="47">{step_now:,} of {total:,}</text>')
    tl.append(f'<text class="tick" x="0" y="80">launched {launch_b:%Y-%m-%d %H:%M} Berlin</text>' if launch_b else
              '<text class="tick" x="0" y="80">launch time UNAVAILABLE (no supervisor log)</text>')
    tl.append((f'<text class="tick" x="900" y="80" text-anchor="end">stopped by the PI at step {step_now:,}</text></svg>')
              if stopped else
              (f'<text class="tick" x="900" y="80" text-anchor="end">finish ≈ {finish:%a %d %b %H:%M} Berlin at {pace:.2f} s/step</text></svg>'
               if finish else '<text class="tick" x="900" y="80" text-anchor="end">finish — pace not measurable yet</text></svg>'))
    timeline = "".join(tl)

    # ------------------------------------------------------------ tables --
    def td(x, nd=3):
        return f'<td class="num">{fmt(_num(x), nd)}</td>'

    seg_rows = []
    for i, s in enumerate(segs):
        lab = (f"{i+1} · {FACTS['segments'][i][0]} ({FACTS['segments'][i][1]})"
               if i < len(FACTS["segments"]) else f"{i+1} · UNPLANNED relaunch")
        if len(s) >= 2:
            k2 = min(10, len(s) - 1)
            pm = (s[-1]["elapsed_s"] - s[-1 - k2]["elapsed_s"]) / max(1, s[-1]["step"] - s[-1 - k2]["step"])
        else:
            pm = None
        ended = "— still running" if i == len(segs) - 1 and not done else "next segment"
        seg_rows.append(f'<tr><td>{esc(lab)}</td><td class="num">{s[0]["step"]:,}–{s[-1]["step"]:,}</td>'
                        f'{td(pm, 2)}<td class="num">{len(s)}</td><td>{ended}</td></tr>')
    seg_table = ('<div style="overflow-x:auto"><table><tr><th>segment</th><th class="num">steps logged</th>'
                 '<th class="num">s / step (marginal)</th><th class="num">rows</th><th>ended by</th></tr>'
                 + "".join(seg_rows) + '</table></div>')

    eval_rows = "".join(
        f'<tr><td class="num">{r["step"]:,}</td>{td(r.get("eval_loss"),2)}{td(r.get("eval_traj"))}'
        f'{td(r.get("eval_goal2s_err_m"),2)}{td(r.get("eval_anchor_acc"))}{td(r.get("eval_tacv6_lat_ce"))}'
        f'{td(r.get("eval_tacv6_lon_ce"))}{td(r.get("eval_tacv6_goal_bce"))}{td(r.get("eval_map_hires"))}'
        f'{td(r.get(map_key("iou", "drivable", "0_20")))}{td(r.get(map_key("iou", "lane", "0_20")))}'
        f'{td(r.get(bkey("box3d", "ap2m")))}{td(r.get(bkey("box3d", "conf_ratio")),2)}'
        f'{td(r.get(bkey("agent", "conf_ratio")),2)}</tr>' for r in ev)
    evals_table = ('<div style="overflow-x:auto"><table><tr><th class="num">step</th><th class="num">eval loss</th>'
                   '<th class="num">traj</th><th class="num">goal 2 s (m)</th><th class="num">anchor acc</th>'
                   '<th class="num">tac lat CE</th><th class="num">tac lon CE</th><th class="num">goal BCE</th>'
                   '<th class="num">map 10 cm CE</th><th class="num">drivable IoU 0–20</th><th class="num">lane IoU 0–20</th>'
                   '<th class="num">box3d AP@2 m</th><th class="num">box3d ratio</th><th class="num">agent ratio</th></tr>'
                   + eval_rows + '</table></div>')

    ckpt_int = int(save_every) if (save_every or "").isdigit() else None
    ckpt_note = (f"--save-every {ckpt_int} (config.json argv)" if ckpt_int else "--save-every not in config.json argv")
    if ckpt_int and ckpt_int != CKPT_EVERY_SPEC:
        ckpt_note += f" ≠ SPEC_REFCV7 sec. 4's {CKPT_EVERY_SPEC}"
    ckpt_late = (ckpt_age_min is not None and pace and ckpt_int
                 and ckpt_age_min > 2 * ckpt_int * pace / 60 + 10)
    health = ('<div style="overflow-x:auto"><table class="kv">'
              f'<tr><td>run directory</td><td class="num">{esc(st.get("run") or "—")}</td><td class="muted">{esc(arm)}</td></tr>'
              f'<tr><td>supervisor / trainer pid</td><td class="num">'
              + (f'{esc(st.get("sup_pid"))} {"alive" if sup_alive else "GONE"} · {esc(st.get("train_pid"))} '
                 f'{"alive" if tr_alive else "GONE"}' if state_read else "UNAVAILABLE (not pulled)")
              + '</td><td class="muted">checked by pid with <code>ps -p</code></td></tr>'
              f'<tr><td>launch gate verdicts</td><td class="num">{gate_ok} MATCH · {len(gate_ref)} REFUSED</td>'
              '<td class="muted"><code>ZZGATEOK</code> / <code>ZZGATEREFUSED</code> in the supervisor log; a refusal is final (G4)</td></tr>'
              f'<tr><td>launches in the supervisor log</td><td class="num">{launches_total} supervisor · {trainer_launches} trainer</td>'
              f'<td class="muted">{planned} planned switch(es) · {unplanned} unplanned relaunch(es) by <code>elapsed_s</code> resets</td></tr>'
              f'<tr><td>supervisor token</td><td class="num">{esc(token.replace(chr(10), " ⏎ ")) if token else "—"}</td><td class="muted">step · steps · tracebacks · launch'
              + (f' — read as {n_err} tracebacks: the zero was printed twice across a line break' if token_split else '') + '</td></tr>'
              f'<tr><td>train.stderr.log</td><td class="num">'
              + (f'{stderr_b:,} B · {len(stderr_lines)} line(s)' if state_read else "UNAVAILABLE") + '</td><td class="muted">'
              + ("not pulled" if not state_read else "empty" if not stderr_lines and stderr_read_ok else
                 "NOT READ — the pulled content is shorter than the file" if not stderr_read_ok else
                 f"{len(stderr_undiag)} undiagnosed · {n_err_client} traceback/OOM lines, counted from the content itself (below)")
              + '</td></tr>'
              f'<tr><td>in-run eval failures</td><td class="num">{len(er)}</td><td class="muted">'
              + (f'last at step {er[-1].get("step")}: {esc(str(er[-1].get("eval_error"))[:160])}' if er else
                 "an eval that raises writes an <code>eval_error</code> row and training continues") + '</td></tr>'
              f'<tr><td>peak device memory</td><td class="num">{mem_peak:.2f} GB</td><td class="muted"><code>cuda_max_mem_gb</code> — the only admissible probe on Thor</td></tr>'
              f'<tr><td>F3 cascade loss (FIX-1)</td><td class="num">{cas_rows} of {len(tr)} train rows</td><td class="muted">'
              + ("present on every logged row" if cas_rows == len(tr) else "⚠ MISSING on some rows — FIX-1 says it is live from step 1")
              + '</td></tr>'
              f'<tr><td>frames computed / slots</td><td class="num">{fmt(comp,0)} / {fmt(slots,0)} ({fmt(dd_frac*100 if dd_frac else None,1)} %)</td><td class="muted">each distinct frame once</td></tr>'
              f'<tr><td>conflict readings</td><td class="num">{len(cd)} · last at step {cd_last["step"] if cd_last else "—"}</td><td class="muted">every 10th step</td></tr>'
              f'<tr><td>ego input</td><td class="num">{"GRU over " + str(eh.get("steps")) + " steps × " + str(eh.get("channels")) + " ch" if eh.get("enable") else ("OFF" if cfg else "config.json UNAVAILABLE")}</td><td class="muted">the ego-history encoder, from <code>config.json</code> seams</td></tr>'
              f'<tr><td>nav input (last batch)</td><td class="num">{fmt(_num(last.get("nav_injected")),0)}</td><td class="muted">nav from the v7 token reaches the model</td></tr>'
              f'<tr><td>eval windows per eval</td><td class="num">{", ".join(str(w) for w in ev_windows) or "—"}</td><td class="muted">a fixed seeded subset: the SAME windows every eval</td></tr>'
              f'<tr><td>backbone levers (built)</td><td class="num">{esc(", ".join(f"{k}={v}" for k, v in levers.items())) or "—"}</td><td class="muted">read off the built trunk, not the flags</td></tr>'
              f'<tr><td>checkpoint</td><td class="num">{fmt(ckpt_gb,2)} GB · {fmt(ckpt_age_min,0)} min old</td><td class="muted">{esc(ckpt_note)}'
              + (" · ⚠ older than two checkpoint intervals at this pace" if ckpt_late else "") + '</td></tr>'
              f'<tr><td>pace</td><td class="num">{fmt(pace, 2)} s/step marginal · {fmt(warm_med, 2)} warm median</td>'
              f'<td class="muted">marginal = the last {k} rows, incl. eval + ckpt; warm = the median over intervals '
              f'without an eval or checkpoint · refcv6 {PACE_REF_S} · line {PACE_PI_LINE_S} (+25 %) · the PI rule itself is '
              'applied at the G-LIVE smoke, not by this page</td></tr>'
              f'<tr><td>learning rate (now)</td><td class="num">{(last.get("lr") or 0):.3e}</td><td class="muted">{"in the 2,000-step warmup" if step_now < 2000 else f"cosine decay to {total:,}"}</td></tr>'
              '</table></div>')
    stderr_tbl = ('<h3>train.stderr.log, line by line</h3><div style="overflow-x:auto"><table><tr><th>line</th><th>diagnosis</th></tr>'
                  + "".join(f'<tr><td><code>{esc(ln)}</code></td><td>{esc(diag[ln]) if ln in diag else "<b>UNDIAGNOSED</b> — read it on Thor"}</td></tr>'
                            for ln in stderr_lines[-40:])
                  + '</table></div>') if stderr_lines else ""

    # ------------------------------------------------------------ map section --
    thin_lines = list(m["reasons"])
    for (s, c, b, v, bar) in m["breaches_latest"][:12]:
        thin_lines.append(f"RED now: {esc(c)} · {band_label(b)}: IoU {v:.3f} ≤ {bar:g}")
    if m["missing_latest"]:
        miss_cells = [(c, b) for (c, b), (kk, _) in m["cells"].items() if kk == "missing"]
        thin_lines.append("absent: " + ", ".join(f"{esc(c)} {band_label(b)}" for c, b in miss_cells[:12])
                          + (" …" if len(miss_cells) > 12 else ""))
    if m["amber_latest"]:
        thin_lines.append("AMBER, informative — 20–60 m at or below 0.05 now: "
                          + ", ".join(f"{esc(c)} {band_label(b)} {v:.3f}" for _s, c, b, v, _f in m["amber_latest"][:10])
                          + (" …" if len(m["amber_latest"]) > 10 else ""))
    if m["state"] == "pending":
        thin_kind, thin_state = "info", "no eval yet"
    elif m["alarm"]:
        thin_kind = "crit"
        thin_state = f"RED (latched) · first red at step {m['first_red_step']:,}"
    elif m["amber_latest"]:
        thin_kind, thin_state = "warn", "clear · amber 20–60 m"
    else:
        thin_kind, thin_state = "good", "clear"
    thin_tile = _tile(thin_kind, "Thin-class alarm", thin_state, thin_lines,
                      "The registered rule only (LOGGING_SPEC_MAP10 5.3): RED iff any class's 0–20 m eval IoU ≤ 0.05 "
                      f"at any eval from step {THIN_FROM_STEP:,}, or any eval row lacks one of the 40 keys; latched. "
                      "20–60 m ≤ 0.05: amber, informative. Beyond 60 m: values only. The per-band bar against "
                      "refcv6@38k (A8) needs the paired interval: the battery reads it at milestones, never this page.",
                      tid="thin-class-alarm")
    rule = m["decision_rule"]
    map_intro = (f'<p>The latest in-run eval (step {m["latest_step"]:,}) on the {ev_w} fixed windows, 8 classes × 5 '
                 f'bands over 100 m × ±30 m. Decision rule declared in config.json: <b>{esc(rule) if rule else "UNAVAILABLE"}</b>'
                 f'{" (argmax of z − log w with the frozen class weights)" if rule == "prior_corrected" else ""}. '
                 'n = labelled 10 cm cells over the eval. These in-run IoUs are T0 on the in-run windows.</p>'
                 '<p><b>What is and is not an alarm here.</b> RED is the registered rule only: any class at or below '
                 f'0.05 IoU at 0–20 m at any eval from step {THIN_FROM_STEP:,}, or an eval row that lacks one of the '
                 '40 keys (latched). 20–60 m at or below 0.05 is amber and informative. Beyond 60 m the page shows '
                 'values only: refcv6 predicted nothing there, and A7 expects far bands low. ⛔ The per-band bars '
                 "against refcv6@38k (A8, BAR-M7-1..3) need the paired interval on the eval kit's 137 clips, so they "
                 'are NOT a Watch alarm: the EvalFlyWheel battery reads them at milestones.</p>'
                 if ev else '<p class="muted">No in-run eval row yet: the 40-key contract cannot be read.</p>')
    cross = ""
    if ev:
        mism, nchk = m["iou_count_mismatch"], m["iou_count_checked"]
        if not nchk:
            verdict = ("could not run: no key of the latest row carries both its IoU and its inter / union counts "
                       "(a check that reads nothing certifies nothing)")
        elif mism:
            verdict = (f"DISAGREES on {len(mism)} key(s): "
                       + ", ".join(f"{esc(c)} {band_label(b)}" for c, b, _, _ in mism[:6]))
        else:
            verdict = f"agrees on all {nchk} keys that carry their counts"
        cross = ('<p class="muted">Cross-check, derived independently of the logged ratios: the IoU re-computed from '
                 f"the row's own inter / union counts {verdict}.</p>")
    legacy = ('<p class="muted">⚠ This log also carries the 0.5 m keys (<code>map</code>, <code>map_iou_drivable</code>): '
              "that is refcv6's 0.5 m head, NOT the refcv7 map, and it is not drawn here (SPEC_REFCV7 A3).</p>"
              if m["legacy_05m"] else "")
    if ev:
        map_html = (map_intro + heat_legend()
                    + heatmap(m, "cells", f"IoU at 10 cm, declared rule ({rule or 'rule UNAVAILABLE'}) — latest eval")
                    + cross + legacy
                    + '<h3>Per class over the run — all 40 series</h3><div class="grid4">' + "".join(mcharts)
                    + cmh + "</div>"
                    + heatmap(m, "raw", "IoU at 10 cm, RAW argmax (the diagnostic; the declared rule gates) — latest eval")
                    + '<h3>Per-class loss share — latest eval</h3>' + share_table(m))
    else:
        map_html = map_intro + legacy + '<div class="grid2 fill">' + cmh + "</div>"
    # the training signal per class (LOGGING_SPEC_MAP10 5 item 2), from the TRAIN rows
    if ms["classes"]:
        srows = []
        for c in MAP_CLASSES:
            dd = ms["classes"][c]
            ratio = (dd["lshare_ema"] / dd["mass"]) if (dd["lshare_ema"] is not None and dd["mass"]) else None
            stc = ('<span class="chip crit"><i></i>no signal</span>' if c in ms["dead_latest"] else
                   '<span class="chip good"><i></i>pulled</span>' if dd["gno"] else
                   '<span class="chip warn"><i></i>no cells in the last row</span>' if dd["n_latest"] == 0 else
                   '<span class="muted">UNAVAILABLE</span>')
            srows.append(f'<tr><td>{esc(c)}{" · thin" if c in THIN_CLASSES else ""}</td>'
                         f'<td class="num">{fmt(dd["lshare_ema"], 4)}</td><td class="num">{fmt(dd["mass"], 4)}</td>'
                         f'<td class="num">{fmt(ratio, 2)}</td>'
                         f'<td class="num">{("%.3g" % dd["gno"]) if dd["gno"] is not None else "UNAVAILABLE"}</td>'
                         f'<td class="num">{len(dd["dead_rows"])}</td><td>{sparkline(dd["gno_series"])}</td>'
                         f"<td>{stc}</td></tr>")
        sig_html = ('<h3>The training signal per class (train rows)</h3><div style="overflow-x:auto"><table>'
                    '<tr><th>class</th><th class="num">loss share (EMA)</th><th class="num">label-mass share</th>'
                    '<th class="num">share ÷ mass</th><th class="num">pull on own cells (last row)</th>'
                    '<th class="num">rows with cells, no pull</th><th>log10 pull</th><th>last row</th></tr>'
                    + "".join(srows) + "</table></div>"
                    '<p class="muted">Loss share = Σ over the bands of <code>map_hires_lshare_*</code>; label-mass share '
                    '= the class\'s labelled cells over all labelled cells; pull = ‖∂L/∂z‖ on the class\'s own cells '
                    '(<code>map_hires_gno_*</code>, L2 over the bands). A class with cells and a pull of exactly 0 '
                    'gets NO signal (LOGGING_SPEC_MAP10 5 item 2); a class with no cells in a batch is no evidence.</p>')
    else:
        sig_html = ('<p class="muted">The training signal per class is UNAVAILABLE: no train row carries '
                    '<code>map_hires_lshare_*</code> or <code>map_hires_gno_*</code>.</p>')
    map_html += sig_html

    # ------------------------------------------------------------ box section --
    box_tiles, box_tables = [], []
    for h in BOX_HEADS:
        d = bx["heads"][h]
        r6 = REF6.get(h, {})
        lines = list(d["reasons"]) + [f"⚠ {esc(x)}" for x in d.get("checks", [])]
        warming = d["state"] == BOX_WARMUP
        kind_ = ("info" if (d["state"] == "pending" or warming) else "crit" if d["alarm"]
                 else "warn" if d.get("checks") else "good")
        state_ = (f"{d['ratio']:.2f} · {d['state']}" if d["ratio"] is not None else
                  f"UNAVAILABLE · {d['state']}" if warming else d["state"])
        box_tiles.append(_tile(kind_, f"{h} — box ratio alarm", state_, lines,
                               f"confident / VIS-1 positives, band [{BOX_RATIO_BAND[0]}, {BOX_RATIO_BAND[1]}] (SPEC_REFCV7 "
                               f"A10 15.3), armed from step {BOX_ARM_STEP:,} · refcv6@38k read "
                               f"{fmt(r6.get('ratio'), 2)} at its 0.5 gate"))
        if d["state"] == "pending" or "vals" not in d:
            continue
        g = d["vals"]
        grid_rows = "".join(
            f'<tr><td>mAP @ {tl_}</td>' + "".join(
                (f'<td class="num">{d["grid"][(t, b)][1]:.3f}</td>' if d["grid"][(t, b)][0] == "value" else
                 '<td class="num muted">undefined</td>' if d["grid"][(t, b)][0] == "undef" else
                 '<td class="num muted">UNAVAILABLE</td>') for b, _ in BOX_BANDS) + "</tr>"
            for t, tl_ in BOX_THR)
        ref_ap = (f'refcv6@38k {fmt(r6["ap2m"], 3)} vs targets'
                  + (f', {fmt(r6["ap2m_vis1"], 3)} re-scored under VIS-1' if r6.get("ap2m_vis1") else "")
                  if r6.get("ap2m") else "no refcv6 reference banked for this head")
        ref_auc = f'refcv6@38k {fmt(r6["auroc"], 3)} (matched)' if r6.get("auroc") else "no refcv6 reference"
        ref_pr = (f'refcv6@38k {fmt(r6["prec"], 3)} / {fmt(r6["rec"], 3)} vs targets'
                  + (f'; {fmt(r6["prec_vis1"], 3)} / {fmt(r6["rec_vis1"], 3)} under VIS-1' if r6.get("prec_vis1") else "")
                  if r6.get("prec") else "no refcv6 reference banked for this head")
        box_tables.append(
            f'<h3>{h} — the latest eval (step {e_last["step"]:,})</h3><div style="overflow-x:auto"><table>'
            '<tr><th>class-mean AP</th>' + "".join(f'<th class="num">{bl}</th>' for _, bl in BOX_BANDS) + "</tr>"
            + grid_rows + '</table></div><div style="overflow-x:auto"><table class="kv">'
            f'<tr><td>AP@2 m BEV, pooled classes</td>{td(g["ap2m"])}<td class="muted">{ref_ap}</td></tr>'
            f'<tr><td>presence AUROC (matched / objectness)</td><td class="num">{fmt(g["auroc_matched"])} / {fmt(g["auroc_objectness"])}</td><td class="muted">{ref_auc}</td></tr>'
            f'<tr><td>precision / recall / F1 at σ ≥ 0.5</td><td class="num">{fmt(g["prec@gate"])} / {fmt(g["rec@gate"])} / {fmt(g["f1@gate"])}</td>'
            f'<td class="muted">{ref_pr}</td></tr>'
            f'<tr><td>confident / VIS-1 positives</td><td class="num">{fmt(d["ratio"], 3)}</td><td class="muted">n_conf {fmt(g["n_conf"], 0)} · positives {fmt(g["n_pos"], 0)} · TP {fmt(g["tp@gate"], 0)} · IGNORE {fmt(g["n_ignore"], 0)} · {fmt(g["n_windows"], 0)} windows</td></tr>'
            f'<tr><td>class accuracy on TPs (raw / prior-corrected)</td><td class="num">{fmt(g["cls_acc_tp"])} / {fmt(g["cls_acc_tp_priorcorr"])}</td><td class="muted">prior-corrected is INFORMATIVE (A10 15.3)</td></tr>'
            f'<tr><td>median matched centre error</td><td class="num">{fmt(g["centre_err_p50"], 2)} m</td><td class="muted">Hungarian pairs, last decoder layer</td></tr>'
            f'<tr><td><b>INFORMATIVE</b>: the TRAIN P = R gate</td><td class="num">{fmt(g["calib_pr_gate"])} (P {fmt(g["calib_prec"])} / R {fmt(g["calib_rec"])})</td>'
            f'<td class="muted">on the FIXED TRAIN calibration windows ({fmt(g["calib_n_windows"], 0)} windows, {fmt(g["calib_n_pos"], 0)} positives); '
            'never replaces the declared 0.5 rule and enters no bar</td></tr></table></div>')
    box_html = ('<p class="muted">The two ratio alarms are the tiles under Alarms, above. Per head, on the last '
                'decoder layer (the one inference reads), on VIS-1 positives with IGNORE rows as DontCare.</p>'
                + ("".join(box_tables) if box_tables else
                   '<p class="muted">UNAVAILABLE: no eval row carries the P0 keys yet (A9: detection is measured every eval).</p>')
                + '<div class="grid2">' + "".join(bcharts) + c8 + c9 + "</div>"
                + f'<p class="muted">Reference: {esc(BOX_REF_SRC)} Key names: the box builder\'s '
                  '<code>detection_metrics.py</code> (UNVERIFIED until the box-head package lands).</p>')

    # ------------------------------------------------------------ gradient reach --
    parts = []
    decl_set = set(ga["declared"] or [])
    for kk in sorted(decl_set | set(ga["observed"])):
        if kk.endswith("_n") and kk[:-2] in (decl_set | set(ga["observed"])):
            continue
        parts.append(kk)
    ga_rows = []
    for kk in parts:
        pk = ga["per_key"].get(kk) or {}
        nk = kk + "_n"
        pn = ga["per_key"].get(nk) or {}
        declared = kk in decl_set
        lv = pk.get("latest", "ABSENT") if declared else (tr[-1].get(kk, "ABSENT"))
        dead = declared and (kk in ga["absent"] or kk in ga["zero"] or nk in ga["absent"] or nk in ga["zero"])
        lv_txt = ("<b>ABSENT</b>" if lv == "ABSENT" else "<b>0 (exactly)</b>" if _num(lv) == 0 else f"{_num(lv):.4g}"
                  if _num(lv) is not None else "<b>non-numeric</b>")
        nv = pn.get("latest", tr[-1].get(nk, "ABSENT"))
        series = pk.get("series") or [(r["step"], _num(r.get(kk))) for r in tr if kk in r]
        state_chip = ('<span class="chip crit"><i></i>dead or absent</span>' if dead else
                      '<span class="chip good"><i></i>reached</span>' if declared else "")
        not_decl = "" if declared else ' <span class="muted">(not declared)</span>'
        n_txt = "ABSENT" if nv == "ABSENT" else fmt(_num(nv), 0)
        min_txt = fmt(_num(pk.get("min")), 4) if declared else "—"
        bad_txt = pk.get("rows_bad", "—") if declared else "—"
        ga_rows.append(f'<tr><td><code>{esc(kk)}</code>{not_decl}</td>'
                       f'<td class="num">{lv_txt}</td><td class="num">{n_txt}</td>'
                       f'<td class="num">{min_txt}</td><td class="num">{bad_txt}</td>'
                       f'<td>{sparkline(series)}</td><td>{state_chip}</td></tr>')
    ga_table = ('<div style="overflow-x:auto"><table><tr><th>group</th><th class="num">grad |sum| (latest row)</th>'
                '<th class="num">params with grad</th><th class="num">min over the run</th><th class="num">rows dead/absent</th>'
                '<th>log10 trend</th><th>latest row</th></tr>' + "".join(ga_rows) + "</table></div>") if parts else \
        '<p class="muted">UNAVAILABLE: no <code>ga_*</code> key in any train row and none declared.</p>'
    gu = ga["grad_unreachable"]
    if isinstance(gu, dict) and gu:
        gu_html = ('<h3>Declared grad-unreachable (information, not an alarm)</h3><div style="overflow-x:auto"><table>'
                   '<tr><th>module</th><th>why it gets no gradient by design</th></tr>'
                   + "".join(f"<tr><td><code>{esc(p)}</code></td><td>{esc(w)}</td></tr>" for p, w in gu.items())
                   + "</table></div>")
    elif gu is not None:
        gu_html = f'<p class="muted">config.json <code>declared_vs_built.grad_unreachable</code>: {esc(json.dumps(gu))}</p>'
    else:
        gu_html = ('<p class="muted">config.json carries no <code>declared_vs_built.grad_unreachable</code> '
                   '(fixes batch 3 records the modules frozen by design there).</p>')
    ga_lines = []
    if ga["absent"]:
        ga_lines.append("absent or non-numeric in the latest row: " + ", ".join(f"<code>{esc(x)}</code>" for x in ga["absent"][:8]))
    if ga["zero"]:
        ga_lines.append("exactly 0 in the latest row: " + ", ".join(f"<code>{esc(x)}</code>" for x in ga["zero"][:8]))
    if ga["rows_bad"] and not (ga["absent"] or ga["zero"]):
        ga_lines.append(f"{ga['rows_bad']} earlier train row(s) had a dead or absent declared key (first: step {ga['first_bad_step']:,})")
    if ga["undeclared"]:
        ga_lines.append(f"{len(ga['undeclared'])} logged ga_* key(s) not in the declaration")
    ga_tile = _tile("crit" if ga["alarm"] else ("warn" if ga["rows_bad"] else "good"), "Gradient reach alarm",
                    ("UNAVAILABLE" if ga["declared"] is None else "ALARM" if ga["alarm"] else "all reached"),
                    ga_lines or [esc(ga["source"])],
                    "ALARM on any DECLARED ga_* key absent or exactly 0 in the latest train row (D3; the declaration "
                    "is config.json grad_reach_logging, read off the BUILT model)")

    # ------------------------------------------------------------ residual prior --
    ps = pr["stamp"]
    prior_rows = ""
    if pr["keys"]:
        prior_rows = ('<div style="overflow-x:auto"><table><tr><th>logged key</th><th class="num">latest</th><th class="num">at step</th></tr>'
                      + "".join(f'<tr><td><code>{esc(kk)}</code></td><td class="num">{fmt(_num(v[1]), 4) if v else "—"}</td>'
                                f'<td class="num">{v[0] if v else "—"}</td></tr>' for kk, v in pr["latest"].items())
                      + "</table></div>")
    prior_html = (
        (f'<p>config.json <code>seams.residual_prior</code>: <b>{esc(ps.get("residual_prior"))}</b> — '
         f'{esc(ps.get("definition", ""))}. Reads: {esc(", ".join(ps.get("reads") or []) or "—")}. Vocabulary: '
         f'{esc(ps.get("vocabulary_space", "—"))}. Equals the battery echo: {esc(ps.get("equals_battery_echo"))}. '
         f'argv <code>--residual-prior</code>: {esc(pr["argv_mode"] or "absent")}'
         + (' <span class="chip crit"><i></i>argv and stamp disagree</span>'
            if pr["argv_mode"] and pr["argv_mode"] != ps.get("residual_prior") else "") + '.</p>')
        if ps else '<p class="muted">UNAVAILABLE: config.json carries no <code>seams.residual_prior</code> stamp.</p>')
    prior_html += (prior_rows if pr["keys"] else
                   '<p class="muted">No logged key carries <code>residual</code> or <code>prior</code> in its name. '
                   "MEASURED from source at the tip (a3db3a8): the trainer emits the prior only in the forward's output "
                   "(<code>residual_prior_ctrl</code> / <code>_v</code> / <code>_path</code>), never into a log row; "
                   "the prior's liveness is G-LIVE's check at the launch smoke.</p>")

    # ------------------------------------------------------------ families + says --
    def ev_get(kk):
        return _num(e_last.get(kk)) if e_last else None
    fam = ('<div style="overflow-x:auto"><table><tr><th>family</th><th>what the in-run eval shows (T0)</th><th>the decision-grade test</th></tr>'
           f'<tr><td>LONGITUDINAL</td><td>2 s goal error {fmt(ev_get("eval_goal2s_err_m"),2)} m (along- and cross-track mixed)</td><td>speed accuracy, headway, TTC — EvalFlyWheel battery</td></tr>'
           '<tr><td>LATERAL</td><td>no dedicated in-run metric</td><td>heading, curvature, yaw-rate, cross-track — EvalFlyWheel battery</td></tr>'
           f'<tr><td>TACTICAL</td><td>tac lat CE {fmt(ev_get("eval_tacv6_lat_ce"))} · lon CE {fmt(ev_get("eval_tacv6_lon_ce"))} · goal BCE {fmt(ev_get("eval_tacv6_goal_bce"))} (true label clock, FIX-2)</td><td>manoeuvre confusion, goal selection — EvalFlyWheel battery</td></tr>'
           '<tr><td>STRATEGIC</td><td>not applicable: the strategic layer is OFF in this arm (SPEC_REFCV7 sec. 1)</td><td>—</td></tr>'
           '</table></div>')

    says = []
    if ev and len(ev) >= 1:
        def tr2(kk, nd=3):
            return f"{fmt(_num(e_first.get(kk)), nd)} → {fmt(_num(e_last.get(kk)), nd)}"
        says.append(f"<li><b>Held-out trend.</b> Eval traj {tr2('eval_traj')}, 2 s goal error {tr2('eval_goal2s_err_m', 2)} m, "
                    f"anchor accuracy {tr2('eval_anchor_acc')} (chance {CHANCE_ANCHOR:.4f}) over {len(ev)} eval(s) from step "
                    f"{e_first['step']:,} to {e_last['step']:,}.</li>")
    if stopped:
        says.append(f"<li><b>Stopped by the PI.</b> The run was stopped at step {step_now:,} (<code>STOPPED_BY_PI.json</code>); "
                    "no finish time is quoted.</li>")
    elif pace is not None:
        says.append(f"<li><b>Pace.</b> {pace:.2f} s/step marginal, incl. eval + ckpt, over the last {k} logged rows "
                    f"(warm median {fmt(warm_med, 2)}) ⇒ finish ≈ {finish:%a %d %b %H:%M} Berlin. "
                    + (f"That is ABOVE the {PACE_PI_LINE_S} s/step line (+25 % over refcv6's {PACE_REF_S}), shown and "
                       "not decided: the PI rule itself is applied at the G-LIVE smoke." if pace_above else
                       f"Within the {PACE_PI_LINE_S} s/step line (+25 % over refcv6's {PACE_REF_S}).") + "</li>")
    says.append(f"<li><b>Stability.</b> {len(segs)} segment(s): {planned} planned switch, {unplanned} unplanned; "
                + (f"stderr {stderr_b:,} B in {len(stderr_lines)} line(s), {len(stderr_undiag)} undiagnosed, "
                   f"{n_err_client} traceback/OOM" if state_read else "stderr not pulled")
                + f"; {len(er)} failed in-run eval(s); peak {mem_peak:.2f} GB.</li>")
    if ev:
        def cell_word(cl):
            return f"{cl[1]:.3f}" if cl[0] == "value" else ("undefined" if cl[0] == "undef" else "UNAVAILABLE")
        dl = m["cells"].get(("drivable", "0_20"), ("missing", None))
        ln = m["cells"].get(("lane", "0_20"), ("missing", None))
        red = (f"RED (latched), first red at step {m['first_red_step']:,}" if m["alarm"] else "clear")
        amb = (f"; {len(m['amber_latest'])} cell(s) at 20–60 m at or below 0.05 now (amber, informative)"
               if m["amber_latest"] else "")
        says.append(f"<li><b>Map (10 cm).</b> {40 - len(m['missing_latest'])} of 40 keys in the latest eval; "
                    f"0–20 m IoU drivable {cell_word(dl)}, lane {cell_word(ln)}; thin-class alarm {red}{amb}.</li>")
        says.append("<li><b>Boxes (P0).</b> " + "; ".join(
            f"{h}: confident/positives {fmt(bx['heads'][h]['ratio'], 2)} ({bx['heads'][h]['state']}), "
            f"AP@2 m {fmt((bx['heads'][h].get('vals') or {}).get('ap2m'))}" for h in BOX_HEADS) + ".</li>")
    says.append(f"<li><b>Gradient reach.</b> {len(ga['declared'] or [])} declared key(s); "
                + (f"{len(ga['absent'])} absent and {len(ga['zero'])} exactly 0 in the latest row"
                   if ga["declared"] else esc(ga["source"])) + ".</li>")
    ns, ns_pub = navsim_read()
    bt = battery_read()
    ns_html, ns_steps = navsim_html(ns)
    bt_html = battery_html(bt)
    if ns_steps:
        parts_ = []
        for s in ns_steps:
            got = [f"{sp_} {ns[s][sp_]['metric']} {fmt(ns[s][sp_]['value'], 2 if sp_ == 'navtest' else 4)} "
                   f"(STOP {fmt(ns[s][sp_].get('STOP'), 2 if sp_ == 'navtest' else 4)}, n {ns[s][sp_].get('n')}; "
                   f"bar: {esc(ns[s][sp_].get('verdict') or '—')})"
                   for sp_ in ("navtest", "navhard", "warmup") if ns[s].get(sp_, {}).get("status") == "ok"]
            run = [sp_ for sp_ in ("navtest", "navhard", "warmup") if ns[s].get(sp_, {}).get("status") == "running"]
            parts_.append(f"step {s:,}: " + ("; ".join(got) or "nothing banked") + (f"; {', '.join(run)} running" if run else ""))
        beats = any(ns[s][sp_].get("value") is not None and ns[s][sp_].get("STOP") is not None
                    and ns[s][sp_]["value"] >= ns[s][sp_]["STOP"]
                    for s in ns_steps for sp_ in ns[s] if ns[s][sp_].get("status") == "ok")
        says.append("<li><b>NavSim (open loop, EvalFlyWheel).</b> " + " · ".join(parts_)
                    + (". No checkpoint beats the STOP plan yet on any split.</li>" if not beats else ".</li>"))
    doesnt = ["<li>⛔ <b>Nothing here is driving performance.</b> The in-run eval is a T0 world-model diagnostic on "
              f"{ev_w} fixed held-out windows. Driving claims come from the four-family T1 battery and the NavSim "
              "suite (EvalFlyWheel), on snapshotted checkpoints — never on Thor while it trains.</li>",
              "<li><b>No interval.</b> Single-run curves carry no estimator; a trend is a trend, not a separated "
              "difference. The map and box bars (BAR-M7, BAR-B7) are paired bootstraps on the eval kit, not these rows.</li>",
              "<li><b>The refcv6 reference lines are on other windows.</b> The box audit's 556 clipgrid windows are "
              "not this run's in-run eval windows: a reference, not a paired comparison.</li>",
              "<li><b>Train and eval are not the same distribution.</b> Training rows are one batch each (noisy); "
              "eval rows average the same fixed windows. Compare eval to eval.</li>"]

    CSS = open(os.path.join(HERE, "watch.css"), encoding="utf-8").read()
    head_stamp = (f"{esc(arm)} on Thor · read {now_b:%Y-%m-%d %H:%M} Berlin (logs are UTC)"
                  + ("" if state_read else " — no run state pulled: built from the metrics file alone")
                  + f" · every number MEASURED from the run's own <code>metrics.jsonl</code> ({len(rows)} rows by parse: "
                  f"{len(tr)} train / {len(ev)} eval / {len(cd)} conflict) · in-training eval = a WM diagnostic on "
                  f"{ev_w} fixed held-out windows, <b>T0</b>, never a driving-performance claim")
    dcell = m["cells"].get(("drivable", "0_20"), ("missing", None)) if ev else ("missing", None)
    lcell = m["cells"].get(("lane", "0_20"), ("missing", None)) if ev else ("missing", None)

    def celltxt(cl):
        return f"{cl[1]:.3f}" if cl[0] == "value" else ("undefined" if cl[0] == "undef" else "UNAVAILABLE")
    e_tile = ((f'<div class="tile"><b>{fmt(ev_get("eval_traj"))}</b><span>eval traj at {e_last["step"]:,} · first eval '
               f'{fmt(_num(e_first.get("eval_traj")))} at {e_first["step"]:,}</span></div>'
               f'<div class="tile"><b>{fmt(ev_get("eval_goal2s_err_m"), 2)} m</b><span>eval 2 s goal error (first '
               f'{fmt(_num(e_first.get("eval_goal2s_err_m")), 2)})</span></div>'
               f'<div class="tile"><b>{celltxt(dcell)}</b><span>map 10 cm drivable IoU, 0–20 m</span></div>'
               f'<div class="tile"><b>{celltxt(lcell)}</b><span>map 10 cm lane IoU, 0–20 m</span></div>')
              if ev else '<div class="tile"><b>—</b><span>no eval yet</span></div>')
    prof = []
    if argv:
        hw = _argv_vals(argv, "--image-hw", 2)
        prof.append(f"{_argv_val(argv, '--trunk-name') or 'trunk ?'} at {'×'.join(hw) if hw else '?'}")
        if _argv_val(argv, "--residual-prior"):
            prof.append(f"residual on the {_argv_val(argv, '--residual-prior')} prior (NEW-1)")
        if _argv_val(argv, "--map-hires") == "on":
            prof.append(f"10 cm map {_argv_val(argv, '--map-hires-x-max-m') or '100'} m × ±"
                        f"{_argv_val(argv, '--map-hires-y-half-m') or '30'} m, {_argv_val(argv, '--map-hires-decision-rule') or '?'} rule (NEW-2)")
        if _argv_val(argv, "--bev-source"):
            prof.append(f"BEV from {_argv_val(argv, '--bev-source')}")
        sel = [n for f, n in (("--graft-tac8-prior", "8×8 tactical prior"), ("--graft-nav-compliance", "nav compliance"),
                              ("--speed-ceiling-filter", "speed ceiling (inference only)")) if f in argv]
        if sel:
            prof.append("selection: " + ", ".join(sel))
        prof.append(f"batch {_argv_val(argv, '--batch') or '?'} × {total:,} steps")
    card_sub = (" · ".join(esc(p) for p in prof) + " — read from config.json argv") if prof else \
        "config.json UNAVAILABLE: the run profile cannot be read"
    total_warn = f'<p class="note"><b>⚠ {esc(total_note)}</b></p>' if total_note else ""
    pace_tile = _tile("warn" if pace_above else ("info" if pace is None else "good"),
                      "Pace — marginal, incl. eval + ckpt",
                      f"{fmt(pace, 2)} s/step" if pace is not None else "not measurable yet",
                      [f"warm median {fmt(warm_med, 2)} s/step (intervals without an eval or checkpoint)",
                       f"line {PACE_PI_LINE_S} s/step = refcv6's {PACE_REF_S} × 1.25"],
                      "Shown, not decided: amber above the line. The PI rule itself is applied at the G-LIVE smoke, "
                      "not by the Watch (SPEC_REFCV7 6.2, 12)")
    alarms = '<div class="alarms">' + thin_tile + "".join(box_tiles) + ga_tile + pace_tile + "</div>"

    page = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>refcv7 Training Watch</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}
{EXTRA_CSS}</style>
<div class="wrap">
<h1>refcv7 Training Watch</h1>
<p class="stamp">{head_stamp}</p>

<div class="cards">
 <div class="card"><h2>{esc(arm)} {chips}</h2>
  <p class="sub">{card_sub}</p>
  {total_warn}
  <div class="tiles">
   <div class="tile"><b>{step_now:,}</b><span>step of {total:,} ({pct:.1f} %)</span></div>
   <div class="tile"><b>{fmt(pace, 2)} s</b><span>per step, marginal (PI line {PACE_PI_LINE_S})</span></div>
   {('<div class="tile"><b>stopped</b><span>by the PI at step ' + format(step_now, ',') + '</span></div>') if stopped else (('<div class="tile"><b>' + format(finish, '%a %H:%M') + '</b><span>finish ≈ ' + format(finish, '%d %b') + ' Berlin (' + format(eta_s / 3600, '.0f') + ' h)</span></div>') if finish else '<div class="tile"><b>—</b><span>finish: pace not measurable yet</span></div>')}
   {e_tile}
  </div></div>
</div>

<h2>Alarms</h2>
{alarms}

<h2>Progress — planning</h2>
<div class="grid2">{c1}{c2}{c3}{c4}</div>
<h2>Progress — tactical</h2>
<div class="grid2">{c5}{c6}</div>

<h2>Map (10 cm) — per class and band</h2>
{map_html}

<h2>Boxes — detection (P0)</h2>
{box_html}

<h2>Gradient reach</h2>
<p class="muted">{esc(ga["source"])}. The latest train row is step {fmt(ga["latest_step"], 0)}.</p>
{ga_table}
{gu_html}

<h2>Residual prior (NEW-1)</h2>
{prior_html}

<h2>Gradient conflict, learning rate and pace</h2>
<div class="grid2">{c10}{c11}{c12}</div>

<h2>Stability</h2>
{timeline}
<h3>Segments</h3>
{seg_table}
<h3>Health</h3>
{health}
{stderr_tbl}

<h2>What the evals say</h2>
<p>All {len(ev)} held-out evals, unedited — the same {ev_w} windows each time. — = the row does not carry the key.</p>
{evals_table}
<h3>The four metric families</h3>
{fam}
{bt_html}

<h2>NavSim — benchmark KPIs per checkpoint</h2>
<p>The EvalFlyWheel's NavSim suite on the snapshotted checkpoints (never on Thor while it trains): an <b>open-loop
benchmark</b>, zero-shot, never closed loop. Each KPI is paired against the model-free controls on the same scenes: STOP
(an all-zero plan), CV (constant velocity), ECHO (the kinematic extrapolation) and the human log. BAR-R7-N1 is navtest PDMS
above STOP with the paired log-cluster interval excluding 0, on the FULL split.</p>
{ns_html}
<p class="muted">{("Published references, not our measurements (as banked by the suite): " + " · ".join(esc(k) + ": " + esc(v) for k, v in ns_pub.items()) + ". ") if ns_pub else ""}Source: the suite's <code>raw/milestones/step*/BARS.json</code> and
<code>summary_*.json</code>, read by this builder; a split whose counts are not the published split's is refused.</p>

<h2>What it says, and what it does not</h2>
<div class="diag"><div><h3>What it says</h3><ul>{"".join(says)}</ul></div>
<div><h3>What it does not say</h3><ul>{"".join(doesnt)}</ul></div></div>

<footer>Source: <code>thor:{esc(st.get("run") or "&lt;run&gt;")}/metrics.jsonl</code>, <code>config.json</code> and the supervisor log, pulled by scp and verified by parse ·
built by <code>taniteval/tools/training_watch/build_watch_refcv7.py</code>, no hand-typed numbers · contracts: SPEC_REFCV7 A8 (40 map keys),
LOGGING_SPEC_MAP10 5.3 (the thin-class alarm), A9/A10 (P0, the ratio band), D3 (the declared gradient reach).</footer>
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
    summary = {
        "arm": arm, "run": st.get("run"), "state_read": state_read, "step": step_now, "total": total,
        "pct": round(pct, 2), "pace_s": round(pace, 3) if pace is not None else None,
        "pace_warm_median_s": round(warm_med, 3) if warm_med is not None else None,
        "pace_above_pi_line": pace_above, "finish_berlin": f"{finish:%Y-%m-%d %H:%M}" if finish else None,
        "eval_last": e_last and {k2: e_last.get(k2) for k2 in ("step", "eval_loss", "eval_traj", "eval_goal2s_err_m",
                                                               "eval_anchor_acc", "eval_map_hires")},
        "segments": len(segs), "unplanned": unplanned, "n_err": n_err, "stderr_bytes": stderr_b,
        "n_err_client": n_err_client, "token_ok": token_ok, "token_split": token_split,
        "stderr_lines": len(stderr_lines), "stderr_undiagnosed": len(stderr_undiag),
        "stderr_read_ok": stderr_read_ok and not stderr_truncated,
        "sup_alive": sup_alive, "train_alive": tr_alive, "done": done, "stopped": stopped,
        "gate": {"match": gate_ok, "refused": gate_ref}, "eval_failures": len(er),
        "cd_last_step": cd_last and cd_last["step"], "cd_cos_last": cd_last and cd_last.get("cd_cos"),
        "readings_ok": readings_ok, "mem_peak_gb": round(mem_peak, 3), "cascade_rows": cas_rows,
        "map": {"state": m["state"], "alarm": m["alarm"], "n_evals": m["n_evals"], "latest_step": m["latest_step"],
                "present_latest": (40 - len(m["missing_latest"])) if ev else 0,
                "missing_latest": m["missing_latest"], "rows_missing": [list(x) for x in m["rows_missing"]],
                "breaches": [list(x) for x in m["breaches"]], "breaches_latest": [list(x) for x in m["breaches_latest"]],
                "amber_latest": [list(x) for x in m["amber_latest"]], "first_red_step": m["first_red_step"],
                "now_min_0_20": list(m["now_min_0_20"]) if m["now_min_0_20"] else None,
                "series_keys": len(m["series_keys"]), "iou_count_mismatch": [list(x) for x in m["iou_count_mismatch"]],
                "iou_count_checked": m["iou_count_checked"],
                "signal": {"state": ms["state"], "alarm": ms["alarm"], "dead_latest": ms["dead_latest"],
                           "missing_gno": ms.get("missing_gno", [])},
                "decision_rule": m["decision_rule"]},
        "box": {"alarm": bx["alarm"], **{h: {"state": d["state"], "alarm": d["alarm"], "armed": d["armed"],
                                              "ratio": d["ratio"],
                                              "checks": d.get("checks", []),
                                              "ap2m": (d.get("vals") or {}).get("ap2m"),
                                              "prec": (d.get("vals") or {}).get("prec@gate"),
                                              "rec": (d.get("vals") or {}).get("rec@gate"),
                                              "auroc": (d.get("vals") or {}).get("auroc_matched")}
                                          for h, d in bx["heads"].items()}},
        "ga": {"alarm": ga["alarm"], "declared": ga["declared"], "absent": ga["absent"], "zero": ga["zero"],
               "rows_bad": ga["rows_bad"], "undeclared": ga["undeclared"],
               "grad_unreachable": sorted(ga["grad_unreachable"]) if isinstance(ga["grad_unreachable"], dict) else None},
        "prior": {"mode": (pr["stamp"] or {}).get("residual_prior"), "argv_mode": pr["argv_mode"], "keys": pr["keys"]},
        "navsim": {str(s): {sp_: (r.get("value") if r.get("status") == "ok" else r.get("status"))
                            for sp_, r in ns[s].items()} for s in ns_steps},
        "battery_steps": sorted(bt),
    }
    return page, summary


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])

    def opt(name):
        return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else None
    arm = opt("--arm") or ARM
    metrics = opt("--metrics")
    wd = opt("--dir") or (None if metrics else L)
    if metrics is None and "--no-pull" not in a:
        run, why = resolve_run(arm, opt("--run"))
        if not run:
            raise SystemExit(f"ZZWATCH-PULL-FAIL {why}")
        pull(run, arm, wd)
    out = opt("--out") or os.path.join(wd or os.path.dirname(os.path.abspath(metrics)), "refcv7_training_watch.html")
    page, summary = build(watch_dir=wd, metrics_path=metrics, arm=arm)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    sp = opt("--summary") or (os.path.splitext(out)[0] + ".summary.json")
    with open(sp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {out} ({len(page.encode('utf-8')):,} B)")
    print("ZZWATCH " + json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
