"""Build the refcv8 Training Watch page: the refcv7 Training Watch, parameterised for refcv8, plus

* a **"vs refcv7"** section -- refcv8 starts FROM refcv7's step-50,400 weights, so for every in-run eval
  metric both logs carry: refcv8's curve against its own steps, refcv7's FINAL LEVEL as a reference band
  (mean +- sd of its last 6 eval rows, i.e. the level refcv8 inherited and must not regress from) and
  refcv7's own curve over its 50,400 steps, dashed, for context; a table (metric | refcv7 final level |
  refcv8 latest | delta | direction); and a SAME-WINDOWS verdict that is DERIVED from the two runs'
  own config.json / logs / trainer source, never typed;
* refcv8's own additions where the log carries them: the B1 guard decision row, the fp32 gradient-share
  readings (`gs_*`), the allocation emission switch, the EMA eval (`ema_eval_*`) against the raw eval, and
  the r8 loss terms (`r8_*`, `eval_r8_*`).

It REUSES `build_watch_refcv7.py` -- loaded as a FRESH module instance and configured for refcv8 (arm, total
steps, no refcv7 NavSim package, the diagnosed stderr line) -- and post-processes the page it builds. The
refcv7 builder is not modified, and a separate refcv7 instance is never touched by this one. Every hook the
post-processing needs is asserted; a hook that is missing fails LOUD (`ZZWATCH-HOOK-MISSING`), it is never
skipped.

Nothing on the page is typed by hand except the `HEADLINE` metric list (a key, a label and the metric's
SENSE -- higher or lower is better) and the facts marked INHERITED below. A key the page needs and a log
lacks renders UNAVAILABLE; a JSON null renders "undefined". Neither is ever drawn as 0.

Tier stamp: both runs' numbers are the TRAINER'S IN-RUN EVAL -- a world-model diagnostic on a FIXED seeded
subset of held-out windows -- T0, never driving performance; one seed each. The four-family T1 battery and
NavSim belong to the EvalFlyWheel and are not on this page.

    python build_watch_refcv8.py                      # read-only pull from Thor, then build
    python build_watch_refcv8.py --no-pull            # rebuild from the last pull
    python build_watch_refcv8.py --metrics M --out O  # build from ONE metrics.jsonl, never pulls

--run <thor dir>  the run (else $REFCV8_RUN, else `--out` of stack/ops/runs.d/refcv8-r101-s0.argv.json in
                  the tree the refcv7 builder sits in, else the constant RUN8_DEFAULT)
--dir <dir>       the local watch dir (else $REFCV8_WATCH_DIR)        --arm <arm>   (else refcv8-r101-s0)
--ref-dir <dir>   refcv7's COMPLETE record: metrics.jsonl + config.json (else $REFCV7_REF_DIR)
--out <page>      --summary <json>   (default: beside the page)
$REFCV7_BUILDER   path of build_watch_refcv7.py when it is not this file's sibling
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ refcv8 constants ----
ARM = "refcv8-r101-s0"
L = os.environ.get("REFCV8_WATCH_DIR", r"C:\Users\Admin\qland\work\watch8")
REF_DIR = os.environ.get("REFCV7_REF_DIR", r"C:\Users\Admin\qland\work\watch7")
RUN8_DEFAULT = "/home/nvidia/refcv8_run/runs/refcv8-r101-s0"      # INHERITED: the brief / the gated argv's --out
TOTAL_SPEC = 48852            # the refcv8 budget: 48,852 steps (the gated argv's --steps; asserted against it)
#: refcv7's COMPLETE record must end here, or the "final level" is not final
REF_FINAL_STEP = 50400
REF_BAND_N = 6                # the band is mean +- sd (n-1) over refcv7's LAST 6 eval rows
#: pace lines kept from the refcv7 builder (6.4 refcv6 / 8.0 pre-approval line / 10.5 PI ceiling)
SMOKE_PACE_S = 8.94           # INHERITED: the G-LIVE smoke pace, MEASURED by the launch (not re-measured here)
GS_PLANNED_FALLBACK = "1000,16250,32500,48750"   # only used when the argv lacks --grad-share-at

#: where the two trainers' held-out-eval subset code lives on Thor (read-only grep). refcv7's tree is the
#: commit it launched from; refcv8's is the run's own launch tree.
EVID_TRAINER7 = "/home/nvidia/refcv7_run/fec3a0dccf/stack/scripts/refc_v3_train.py"
EVID_RUN7 = "/home/nvidia/refcv7_run/runs/refcv7-r101-s0"
EVID_FILE = "eval_set_evidence.json"
#: the line the trainer prints for the held-out eval, and the two lines that DEFINE which windows it takes
HELD_OUT_RE = re.compile(r"held-out eval: (\d+) episodes -> (\d+) windows, (\d+) fixed batches every (\d+) steps")
SEED_LINE = "manual_seed(12345)"
PERM_LINE = "randperm(len(e_ds)"

#: refcv8's own stderr line, DIAGNOSED (exact text -- a relaunch carries a new pid/timestamp and so reads
#: UNDIAGNOSED again, on purpose: the refcv7 builder's rule). The diagnosis is read off the warning's own
#: text and is UNVERIFIED against the torch source.
STDERR_DIAGNOSED = {
    "W1015 10:27:37.197000 1077980 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use max_autotune_gemm mode":
        "torch-inductor informational warning (max_autotune_gemm needs more SMs than this GPU has; inductor "
        "skips that mode). Not a traceback / OOM. Diagnosed 2026-10-15 from the warning text; UNVERIFIED "
        "against the torch source.",
}
SEGMENTS = [("launch", "the gated launch (supervisor log sup_refcv8-r101-s0.log); warm start from "
                       "refcv7-r101-s0 @ 50,400")]

# ------------------------------------------------------------------ the metric list (the only typed spec) ----
#: (id, key, label, sense, group). `sense` = which direction is BETTER: "higher" | "lower".
#: There is NO ADE key in the trainer's in-run eval; trajectory-type = eval_traj (a loss) and
#: eval_goal2s_err_m (metres). The map IoU keys are the 10 cm map's.
HEADLINE = [
    ("loss", "eval_loss", "Eval loss (total)", "lower", "planning"),
    ("loss_ex", "eval_loss_ex_r8", "Eval loss excluding r8 terms", "lower", "planning"),
    ("traj", "eval_traj", "Trajectory loss", "lower", "planning"),
    ("goal2s", "eval_goal2s_err_m", "Goal @ 2 s error (m)", "lower", "planning"),
    ("anchor", "eval_anchor_acc", "Anchor accuracy", "higher", "planning"),
    ("lat", "eval_lat", "Lateral loss", "lower", "planning"),
    ("lon", "eval_lon", "Longitudinal loss", "lower", "planning"),
    ("box_ap", "eval_box3d_ap2m", "Box3D AP @ 2 m", "higher", "boxes"),
    ("agent_ap", "eval_agent_ap2m", "Agent AP @ 2 m", "higher", "boxes"),
    ("box_auc", "eval_box3d_auroc_matched", "Box3D AUROC (matched)", "higher", "boxes"),
    ("box_loss", "eval_box3d", "Box3D loss", "lower", "boxes"),
    ("map_loss", "eval_map_hires", "Map 10 cm loss", "lower", "map"),
    ("iou_driv", "eval_map_hires_iou_drivable_0_20", "Map IoU drivable, 0–20 m", "higher", "map"),
    ("iou_lane", "eval_map_hires_iou_lane_0_20", "Map IoU lane, 0–20 m", "higher", "map"),
    ("iou_cross", "eval_map_hires_iou_crosswalk_0_20", "Map IoU crosswalk, 0–20 m", "higher", "map"),
    ("tac", "eval_tac_v6", "Tactical loss (v6, total)", "lower", "tactical"),
    ("tac_lat", "eval_tacv6_lat_ce", "Tactical lateral CE", "lower", "tactical"),
    ("tac_lon", "eval_tacv6_lon_ce", "Tactical longitudinal CE", "lower", "tactical"),
    ("tac_goal", "eval_tacv6_goal_bce", "Tactical goal BCE", "lower", "tactical"),
]
#: metrics whose DEFINITION differs between the runs (a chip beside the metric; the Δ is not like-for-like).
#: eval_loss: the trainer adds the weighted r8 terms to the total (`loss = loss + _l8`, refc_v3_train.py), which
#: refcv7 does not have; the derived `eval_loss_ex_r8` = eval_loss − eval_r8_total removes exactly that term.
#: The other loss terms carry the same effective weights in both configs (config.json effective_weights).
DEFINITION_CHANGED = {
    "eval_loss": "NOT like-for-like — refcv8's total includes its weighted r8 terms; see the excl.-r8 row",
    "eval_sel_v3": "different candidates — refcv8 emits its allocated candidates from step 1,000",
}
DERIVED_NOTE = ("eval_loss_ex_r8 is DERIVED here: eval_loss − eval_r8_total for refcv8 (the trainer adds the already-weighted "
                "r8 total to the loss at weight 1.0), and eval_loss itself for refcv7, whose rows carry no r8 term.")
MAP_CLASSES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
MAP_BANDS = ("0_20", "20_40", "40_60", "60_80", "80_100")

#: Which shared keys have a defined better/worse sense in the all-keys table. Anything else (counters, label
#: shares, flags, gates, confidences) is NOT a quality metric and is counted, not tabulated.
_CLS = "(?:automobile|bus|heavy_truck|other_vehicle|person|protruding_object|rider|stroller|trailer|animal|all)"
_HEAD = "(?:box3d|agent)"
HIGHER_RES = [re.compile(p) for p in (
    rf"^eval_{_HEAD}_(?:ap2m|auroc_matched|auroc_objectness|calib_pr_gate|calib_prec|calib_rec|prec@gate|f1@gate|"
    r"cls_acc_tp|cls_acc_tp_priorcorr)$",
    rf"^eval_{_HEAD}_rec@gate(?:_\w+)?$",
    rf"^eval_{_HEAD}_det_(?:ap0p5|ap1|ap2|ap4)_{_CLS}_(?:all|\d+_\d+)$",
    rf"^eval_{_HEAD}_det_map(?:0p5|1|2|4)_(?:all|\d+_\d+)$",
    r"^eval_anchor_acc$",
    r"^eval_map_hires_(?:iou|iouraw)_(?:%s)_\d+_\d+$" % "|".join(MAP_CLASSES),
)]
LOWER_RES = [re.compile(p) for p in (
    r"^eval_(?:loss|loss_ex_r8|traj|lat|lon|cascade|sel_v3|route|goal2s_err_m|goal_tac|lat_tac|lon_tac|tac_v6|cls|law|"
    r"map_hires|box3d|agent)$",
    rf"^eval_{_HEAD}_(?:centre|centre_err_p50|cls|h|size|yaw|z|rates|presence|layer\d|presence_layer\d)$",
    r"^eval_tacv6_(?:lat_ce|lon_ce|goal_bce|goal_conf_bce|total_weighted)$",
)]
#: counters used by the SAME-WINDOWS check (label-derived counts do not depend on the model)
COUNT_RE = re.compile(r"(?:^|_)n_|npos|_n$|^eval_windows$|^eval_batches$|(?:^|_)rows(?:_|$)|n_windows$")
TAC_RE = re.compile(r"(?:^|_)tac")


# ------------------------------------------------------------------ load the refcv7 builder ----
def load_refcv7():
    """A FRESH instance of build_watch_refcv7 -- never the shared sys.modules one, so nothing configured
    here can leak into another importer (the refcv7 tests load it the same way)."""
    path = os.environ.get("REFCV7_BUILDER") or os.path.join(HERE, "build_watch_refcv7.py")
    if not os.path.isfile(path):
        raise SystemExit(f"ZZWATCH-NO-REFCV7-BUILDER {path} (set $REFCV7_BUILDER)")
    spec = importlib.util.spec_from_file_location("build_watch_refcv7_for_refcv8", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def configure(b7, arm: str):
    """The refcv7 builder's parameters, set for refcv8 on THIS instance only."""
    b7.ARM = arm
    b7.L = L
    b7.TOTAL_SPEC = TOTAL_SPEC
    b7.EVAL_PKG = ""             # no refcv7 NavSim/battery package may be read as refcv8's
    b7.EVAL_LIVE = None
    b7.FACTS = {"segments": list(SEGMENTS), "stderr_diagnosed": dict(STDERR_DIAGNOSED)}
    b7.navsim_html = lambda ns: ("", [])
    return b7


# ------------------------------------------------------------------ small helpers ----
def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _g(x, nd=4):
    """compact number: 4 significant figures."""
    if not _isnum(x):
        return "—"
    if x != 0 and abs(x) < 10 ** -nd:
        return f"{x:.2e}"
    return f"{x:.{nd}g}"


def _d(x, nd=3):
    if not _isnum(x):
        return "—"
    if x != 0 and abs(x) < 10 ** -nd:
        return f"{x:+.1e}"
    return f"{x:+.{nd}g}"


def _find_path(cfg, *path):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def points(ev, key):
    """[(step, value-or-None)] for every eval row that carries `key` (a JSON null is kept as None)."""
    return [(r["step"], r[key] if _isnum(r.get(key)) else None) for r in ev if key in r]


def latest(ev, key):
    """('value', v, step) | ('undef', None, step) -- the key is present and null | ('missing', None, step):
    the LATEST eval row does not carry it. Never a 0."""
    if not ev:
        return ("missing", None, None)
    r = ev[-1]
    if key not in r:
        return ("missing", None, r["step"])
    return (("value", r[key], r["step"]) if _isnum(r[key]) else ("undef", None, r["step"]))


def ref_level(ev7, key, n=REF_BAND_N):
    """refcv7's final level: mean and sd (n-1) over its LAST `n` eval rows -- computed from refcv7's rows
    and from nothing else. None when fewer than 3 of them carry a number."""
    vals = [r[key] for r in ev7[-n:] if key in r and _isnum(r[key])]
    if len(vals) < 3:
        return None
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals), "n": len(vals),
            "steps": [r["step"] for r in ev7[-n:]]}


def own_scatter(ev8, key, n=REF_BAND_N):
    vals = [r[key] for r in ev8[-n:] if key in r and _isnum(r[key])]
    return statistics.stdev(vals) if len(vals) >= 3 else None


def judge(sense, lvl, cell, own_sd):
    """-> dict(delta, word, in_band, in_scatter). `cell` = latest(...). word in better|worse|equal|undefined|
    UNAVAILABLE|no-reference."""
    st, v, _ = cell
    if st == "missing":
        return {"delta": None, "word": "UNAVAILABLE", "in_band": None, "in_scatter": None}
    if st == "undef":
        return {"delta": None, "word": "undefined", "in_band": None, "in_scatter": None}
    if lvl is None:
        return {"delta": None, "word": "no-reference", "in_band": None, "in_scatter": None}
    dl = v - lvl["mean"]
    if dl == 0:
        word = "equal"
    else:
        word = "better" if ((dl > 0) == (sense == "higher")) else "worse"
    return {"delta": dl, "word": word, "in_band": abs(dl) <= lvl["sd"],
            "in_scatter": (abs(dl) <= own_sd) if own_sd is not None else None}


def metric_row(key, label, sense, ev7, ev8):
    lvl = ref_level(ev7, key)
    raw = latest(ev8, key)
    ema = latest(ev8, "ema_" + key) if any(("ema_" + key) in r for r in ev8) else ("missing", None, None)
    has_ema = any(("ema_" + key) in r for r in ev8)
    sd_raw = own_scatter(ev8, key)
    sd_ema = own_scatter(ev8, "ema_" + key) if has_ema else None
    present7 = any(key in r for r in ev7)
    return {"key": key, "label": label, "sense": sense, "lvl": lvl, "present7": present7,
            "raw": raw, "ema": ema, "has_ema": has_ema,
            "j_raw": judge(sense, lvl, raw, sd_raw), "j_ema": judge(sense, lvl, ema, sd_ema) if has_ema else None,
            "sd_raw": sd_raw, "sd_ema": sd_ema}


# ------------------------------------------------------------------ the SAME-WINDOWS verdict ----
def _argv_get(cfg, flag, n=1):
    argv = cfg.get("argv") if isinstance(cfg, dict) else None
    if not isinstance(argv, list):
        return None
    v = None
    for i, t in enumerate(argv):
        if t == flag and i + n < len(argv):
            v = argv[i + 1:i + 1 + n]
    return v[0] if (v and n == 1) else v


def count_keys(ev):
    ks = set()
    for r in ev:
        for k in r:
            if k.startswith("eval_") and COUNT_RE.search(k):
                ks.add(k)
    return ks


def windows_verdict(cfg7, cfg8, ev7, ev8, evid):
    """Are the two runs' in-run evals on the SAME windows? DERIVED from the runs' own artifacts:
    the trainer takes `randperm(len(eval_dataset), generator=seed 12345)[:eval_batches * batch]`, so the
    windows are a pure function of (the eval cache, the eval labels, that seed, the batch count). The
    checks below establish those inputs are the same in both runs. This is NOT a per-window-id dump
    (the trainer has `--eval-window-dump` for that). Every check is listed with both sides."""
    checks = []

    def add(name, v7, v8, ok, note=""):
        checks.append({"name": name, "refcv7": v7, "refcv8": v8, "ok": bool(ok), "note": note})

    for flag, nm in (("--eval-cache", "eval cache dir (--eval-cache)"), ("--eval-labels", "eval labels (--eval-labels)"),
                     ("--speed-max-sidecar-v6-eval", "eval speed sidecar"),
                     ("--eval-batches", "eval batches"), ("--batch", "batch")):
        a, b = _argv_get(cfg7, flag), _argv_get(cfg8, flag)
        add(nm, a, b, a is not None and a == b)
    n7 = _find_path(cfg7, "agent_join_stats", "eval", "n_windows")
    n8 = _find_path(cfg8, "agent_join_stats", "eval", "n_windows")
    add("eval-set windows in config.json (agent_join_stats.eval.n_windows)", n7, n8, n7 is not None and n7 == n8)
    w7 = sorted({r.get("eval_windows") for r in ev7 if r.get("eval_windows") is not None})
    w8 = sorted({r.get("eval_windows") for r in ev8 if r.get("eval_windows") is not None})
    add("windows scored per eval row (eval_windows)", w7, w8, len(w7) == 1 and w7 == w8)
    evd = evid if isinstance(evid, dict) else {}
    h7, h8 = HELD_OUT_RE.search(evd.get("ev7") or ""), HELD_OUT_RE.search(evd.get("ev8") or "")
    add("trainer log: held-out eval episodes -> windows (train.log)",
        h7.group(0) if h7 else "NOT READ", h8.group(0) if h8 else "NOT READ",
        bool(h7 and h8 and h7.groups() == h8.groups()))
    s7 = SEED_LINE in (evd.get("seed7") or "") and PERM_LINE in (evd.get("perm7") or "")
    s8 = SEED_LINE in (evd.get("seed8") or "") and PERM_LINE in (evd.get("perm8") or "")
    add("trainer source: subset = randperm(len(e_ds), seed 12345)[:n]",
        "found" if s7 else "NOT READ", "found" if s8 else "NOT READ", s7 and s8)
    # the label-derived counters: constant inside each run and identical across the two
    k7, k8 = count_keys(ev7), count_keys(ev8)
    shared = sorted(k7 & k8)
    same, varying, label_changed = [], [], []
    for k in shared:
        v7s = {r[k] for r in ev7 if k in r and _isnum(r[k])}
        v8s = {r[k] for r in ev8 if k in r and _isnum(r[k])}
        if len(v7s) == 1 and len(v8s) == 1:
            (same if v7s == v8s else label_changed).append(k)
        else:
            varying.append(k)
    n_const = len(same) + len(label_changed)
    # a window set that differs changes MANY counters at once; the only counters allowed to differ across the
    # runs are the TACTICAL ones (the tactical supervision counters differ -- stated on the page), never a
    # perception / box / map / plan counter
    non_tac = sorted(k for k in label_changed if not TAC_RE.search(k))
    add("label-derived counters identical in both runs (tactical-label counters excepted)",
        f"{n_const} constant counters",
        f"{len(same)} identical; {len(label_changed) - len(non_tac)} tactical differ; "
        f"{len(non_tac)} other differ" + (f" ({', '.join(non_tac[:4])})" if non_tac else ""),
        n_const > 0 and not non_tac,
        f"{len(varying)} counters vary with the model inside each run and are not window evidence")
    failed = [c["name"] for c in checks if not c["ok"]]
    same_windows = not failed
    tac_changed = sorted(k for k in label_changed if TAC_RE.search(k))
    return {"same": same_windows, "failed": failed, "checks": checks, "n_counters_constant": n_const,
            "n_counters_identical": len(same), "label_changed_counters": label_changed,
            "label_changed_tactical": tac_changed, "counters_varying": len(varying),
            "verdict": ("SAME EVAL WINDOWS — derived" if same_windows else "INDICATIVE ONLY — " + "; ".join(failed)),
            "proof_class": "DERIVED from config + logs + trainer source; NOT a per-window-id dump"}


# ------------------------------------------------------------------ evidence pull (read-only) ----
def pull_evidence(run: str, out_dir: str, cfg8: dict | None):
    """Read-only: the trainer's held-out-eval log line and the two subset-defining source lines, for both
    runs, plus the eval cache / labels mtimes. One ssh, `grep`/`stat`/`ls` only -- nothing that loads Thor's
    CPU, GPU or RAM. Values already pulled are kept when a re-read comes back empty."""
    b7 = load_refcv7()
    for p in (run, EVID_RUN7, EVID_TRAINER7):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", p):
            raise SystemExit(f"ZZWATCH-PULL-FAIL refusing an unsafe path: {p!r}")
    base8 = run.rsplit("/runs/", 1)[0]
    trainer8 = f"{base8}/launch_tree/stack/scripts/refc_v3_train.py"
    cache = _argv_get(cfg8, "--eval-cache") or ""
    labels = _argv_get(cfg8, "--eval-labels") or ""
    if (cache and not re.fullmatch(r"[A-Za-z0-9_./-]+", cache)) or (labels and not re.fullmatch(r"[A-Za-z0-9_./-]+", labels)):
        raise SystemExit("ZZWATCH-PULL-FAIL refusing an unsafe eval path")

    def g(path, pat):
        return f"$(grep -a -m1 '{pat}' {path} 2>/dev/null | cut -c1-200)"
    cmd = "; ".join([
        f'echo "ZZE:ev8={g(run + "/train.log", "held-out eval")}"',
        f'echo "ZZE:ev7={g(EVID_RUN7 + "/train.log", "held-out eval")}"',
        f'echo "ZZE:seed8={g(trainer8, SEED_LINE)}"', f'echo "ZZE:perm8={g(trainer8, PERM_LINE)}"',
        f'echo "ZZE:seed7={g(EVID_TRAINER7, SEED_LINE)}"', f'echo "ZZE:perm7={g(EVID_TRAINER7, PERM_LINE)}"',
        f'echo "ZZE:cache_mtime=$(stat -c %Y {cache} 2>/dev/null)"' if cache else "true",
        f'echo "ZZE:labels_mtime=$(stat -c %Y {labels} 2>/dev/null)"' if labels else "true",
    ])
    out = b7._ssh(cmd)
    got = {}
    for line in out.splitlines():
        if line.startswith("ZZE:") and "=" in line:
            k, _, v = line[4:].partition("=")
            got[k.strip()] = v.strip()
    path = os.path.join(out_dir, EVID_FILE)
    prev = b7._jload(path) or {}
    merged = {**{k: v for k, v in prev.items() if isinstance(v, str) and v}, **{k: v for k, v in got.items() if v}}
    merged["trainer7"], merged["trainer8"] = EVID_TRAINER7, trainer8
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(merged, fh, indent=1)


# ------------------------------------------------------------------ SVG charts (the vs-refcv7 curves) ----
def _ticks(lo, hi, n=5):
    """Round tick values covering [lo, hi]: the refcv7 `nice_ticks` rule, but counted by INTEGER index so it
    always terminates (at most n * 4 + 2 ticks), whatever the magnitude of lo against the step."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = 10 * mag
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    k0 = math.floor(lo / step)
    k1 = min(math.floor(hi / step), k0 + n * 4 + 2)
    return [round(k * step, 10) for k in range(k0, k1 + 1)]


def _quant(ys):
    """6 significant digits: float noise between nominally equal values becomes exact equality, so the
    refcv7 `chart()` (which has the same one-ulp hazard) never sees a noise-sized spread."""
    return [None if y is None else float(f"{y:.6g}") for y in ys]


def _clamp(y, lo, hi):
    return min(max(y, lo), hi)


def vs_chart(b7, cid, title, sub, key, ev7, ev8, lvl, mode, full_xmax, zoom_xmax, tag=""):
    """One figure: refcv8 raw (dots+line), refcv8 EMA (line), refcv7's final band, and -- in the FULL view
    only -- refcv7's own curve dashed. Same SVG geometry, classes and hover payload as the refcv7 builder's
    `chart()`, so the page's own tooltip script drives it."""
    esc = b7.esc
    X0, X1, Y0, Y1 = b7.X0, b7.X1, b7.Y0, b7.Y1
    xmax = full_xmax if mode == "full" else zoom_xmax
    s8 = [(x, y) for x, y in points(ev8, key) if x <= xmax]
    se = [(x, y) for x, y in points(ev8, "ema_" + key) if x <= xmax]
    s7 = points(ev7, key) if mode == "full" else []
    vals = [y for _, y in s8 + se if y is not None]
    if lvl:
        vals += [lvl["mean"] - lvl["sd"], lvl["mean"] + lvl["sd"]]
    if mode == "full":
        vals += [y for x, y in s7 if y is not None and x >= 10000]     # refcv7's early (cold-start) values are clipped
    cls = f"vs{'f' if mode == 'full' else 'z'}"
    head = (f'<figure class="chart {cls}"><figcaption><b>{esc(title)}</b><span>{sub}</span>'
            f'{("<span>" + tag + "</span>") if tag else ""}</figcaption>')
    if not vals:
        return head + '<p class="note">UNAVAILABLE — neither run carries this key (nothing drawn).</p></figure>'
    lo, hi = min(vals), max(vals)
    scale = max(abs(lo), abs(hi))
    if hi - lo <= 1e-9 * scale or hi - lo <= 1e-300:
        # a constant series -- or one whose spread is float noise: the noise must not set the axis (the refcv7
        # builder's nice_ticks never advances when its step is below one ulp of the tick, i.e. it HANGS)
        pad = scale * 0.05 or 1.0
        lo, hi = (lo + hi) / 2 - pad, (lo + hi) / 2 + pad
    else:
        pad = (hi - lo) * 0.10
        lo, hi = lo - pad, hi + pad

    def sx(x):
        return X0 + (X1 - X0) * x / xmax

    def sy(y):
        return Y1 - (Y1 - Y0) * (y - lo) / (hi - lo)

    g = []
    for t in _ticks(lo, hi, 5):
        if t < lo - 1e-12 or t > hi + 1e-12:
            continue
        yy = sy(t)
        lab = f"{t:g}" if abs(t) >= 1 or t == 0 else f"{t:.2g}"
        g.append(f'<line class="grid" x1="{X0}" y1="{yy:.1f}" x2="{X1}" y2="{yy:.1f}"/>'
                 f'<text class="tick" x="{X0 - 6}" y="{yy + 3.5:.1f}" text-anchor="end">{lab}</text>')
    xt = 10000 if xmax > 20000 else (2500 if xmax > 8000 else (500 if xmax > 2000 else 250))
    x = xt
    while x < xmax:
        g.append(f'<text class="tick" x="{sx(x):.1f}" y="242" text-anchor="middle">{x:,}</text>')
        x += xt
    g.append(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
    if lvl:
        ytop, ybot = sy(lvl["mean"] + lvl["sd"]), sy(lvl["mean"] - lvl["sd"])
        h = max(ybot - ytop, 2.0)
        g.append(f'<rect class="band" x="{X0}" y="{(ytop + ybot) / 2 - h / 2:.1f}" width="{X1 - X0}" height="{h:.1f}"/>')
        ym = sy(lvl["mean"])
        g.append(f'<line class="ref" x1="{X0}" y1="{ym:.1f}" x2="{X1}" y2="{ym:.1f}"/>'
                 f'<text class="reflabel" x="{X1}" y="{ym - 3:.1f}" text-anchor="end">refcv7 final {esc(_g(lvl["mean"]))}</text>')
    data = []

    def path(pts, color, dash=False, dots=False, name="", end=True):
        pp = [(sx(x), sy(_clamp(y, lo, hi))) for x, y in pts if y is not None]
        if not pp:
            return
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pp)
        dash_attr = ' stroke-dasharray="5 4"' if dash else ""
        g.append(f'<path class="line" d="{d}" stroke="var(--{color})"{dash_attr}/>')
        if dots:
            for a, b in pp:
                g.append(f'<circle class="dot" cx="{a:.1f}" cy="{b:.1f}" r="2.6" fill="var(--{color})"/>')
        if end:
            a, b = pp[-1]
            g.append(f'<circle class="end" cx="{a:.1f}" cy="{b:.1f}" r="4" fill="var(--{color})"/>')
        data.append({"name": name, "xs": [round(x, 1) for x, y in pts],
                     "ys": [None if y is None else float(f"{y:.5g}") for x, y in pts], "key": key})

    path(s7, "ink2", dash=True, name="refcv7 (own steps)", end=False)
    path(se, "s3", name="refcv8 EMA")
    path(s8, "s1", dots=True, name="refcv8 raw")
    g.append(f'<line class="xh" x1="0" y1="{Y0}" x2="0" y2="{Y1}" style="display:none"/>'
             f'<rect class="hit" x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}" fill="transparent"/>')
    leg = [('<span class="lg"><i style="background:var(--s1)"></i>refcv8 raw eval</span>')]
    if se:
        leg.append('<span class="lg"><i style="background:var(--s3)"></i>refcv8 EMA eval</span>')
    if lvl:
        leg.append('<span class="lg"><i class="lg-band"></i>refcv7 final: mean ± sd of its last '
                   f'{lvl["n"]} evals</span>')
    if mode == "full" and s7:
        leg.append('<span class="lg"><i class="lg-dash"></i>refcv7 own curve (early cold-start values clipped)</span>')
    note = ""
    if not se:
        note = '<p class="note">No EMA eval key for this metric in the refcv8 log (<code>ema_' + esc(key) + '</code>).</p>'
    payload = json.dumps({"xmax": xmax, "series": data}).replace("<", "\\u003c")
    return (head + f'<span class="ylab">{esc(key)}</span>'
            f'<svg class="plot vsplot" viewBox="0 0 560 260" data-chart="{cid}" role="img" aria-label="{esc(title)}">{"".join(g)}</svg>'
            f'<div class="legend">{"".join(leg)}</div><div class="tip" data-for="{cid}"></div>'
            f'<script type="application/json" data-series="{cid}">{payload}</script>{note}</figure>')


# ------------------------------------------------------------------ table cells ----
def _dir_span(b7, j):
    w = j["word"]
    cls = {"better": "good", "worse": "crit"}.get(w, "mute")
    return f'<span class="dir {cls}">{b7.esc(w)}</span>'


def _noise(j):
    bits = []
    if j["in_band"] is not None:
        bits.append("inside refcv7 band" if j["in_band"] else "outside refcv7 band")
    if j["in_scatter"] is not None:
        bits.append("within refcv8 scatter" if j["in_scatter"] else "beyond refcv8 scatter")
    return "; ".join(bits)


def _val_cell(b7, cell, with_step=True):
    st, v, step = cell
    if st == "value":
        s = f'<b>{b7.esc(_g(v))}</b>'
        return s + (f'<br><span class="muted">step {step:,}</span>' if with_step and step is not None else "")
    if st == "undef":
        return f'<span class="muted" title="the log carries JSON null: undefined, not 0">undefined</span>'
    return '<span class="unav" title="the latest eval row does not carry this key">UNAVAILABLE</span>'


def _delta_cell(b7, j):
    if j is None:
        return '<span class="muted">no EMA key</span>'
    if j["delta"] is None:
        word = {"UNAVAILABLE": '<span class="unav">UNAVAILABLE</span>', "undefined": '<span class="muted">undefined</span>',
                "no-reference": '<span class="unav">no refcv7 reference</span>'}.get(j["word"], b7.esc(j["word"]))
        return word
    return (f'<b>{b7.esc(_d(j["delta"]))}</b> {_dir_span(b7, j)}'
            f'<br><span class="muted">{b7.esc(_noise(j))}</span>')


def _lvl_cell(b7, row):
    lvl = row["lvl"]
    if lvl is None:
        return ('<span class="unav">UNAVAILABLE</span>' if not row["present7"]
                else '<span class="muted">undefined (fewer than 3 numeric)</span>')
    return f'<b>{b7.esc(_g(lvl["mean"]))}</b> <span class="muted">± {b7.esc(_g(lvl["sd"], 2))}</span>'


NO_EMA = '<span class="muted">no EMA key</span>'


def table_rows(b7, rows, flag_fn):
    out = []
    for r in rows:
        flag = flag_fn(r["key"])
        tag = f' <span class="chip info flagchip">{b7.esc(flag)}</span>' if flag else ""
        out.append(
            f'<tr><td>{b7.esc(r["label"])}{tag}<br><code>{b7.esc(r["key"])}</code>'
            f'<br><span class="muted">{"higher" if r["sense"] == "higher" else "lower"} is better</span></td>'
            f'<td class="num">{_lvl_cell(b7, r)}</td>'
            f'<td class="num">{_val_cell(b7, r["raw"])}</td><td class="num">{_delta_cell(b7, r["j_raw"])}</td>'
            f'<td class="num">{_val_cell(b7, r["ema"]) if r["has_ema"] else NO_EMA}</td>'
            f'<td class="num">{_delta_cell(b7, r["j_ema"])}</td></tr>')
    return "".join(out)


TABLE_HEAD = ('<tr><th>metric</th><th class="num">refcv7 final level<br>(mean ± sd, last 6 evals)</th>'
              '<th class="num">refcv8 latest<br>raw eval</th><th class="num">Δ raw vs refcv7</th>'
              '<th class="num">refcv8 latest<br>EMA eval</th><th class="num">Δ EMA vs refcv7</th></tr>')


def classify(key):
    if any(p.match(key) for p in HIGHER_RES):
        return "higher"
    if any(p.match(key) for p in LOWER_RES):
        return "lower"
    return None


def shared_keys(ev7, ev8):
    k7 = {k for r in ev7 for k in r if k.startswith("eval_")}
    k8 = {k for r in ev8 for k in r if k.startswith("eval_")}
    return sorted(k7 & k8), len(k7), len(k8)


# ------------------------------------------------------------------ refcv8's own additions ----
def b1_read(rows8, cfg8, step_now):
    guard_cfg = _find_path(cfg8, "ema", "b1_guard") or {}
    gstep = guard_cfg.get("guard_step")
    drow = [r for r in rows8 if "b1_guard" in r]
    return {"row": drow[-1] if drow else None, "guard_step": gstep, "ref": guard_cfg.get("ref_ap2m"),
            "metric": guard_cfg.get("metric"), "step_now": step_now}


def b1_tile(b7, b):
    r = b["row"]
    gstep = b["guard_step"]
    src = ("<code>b1_*</code> decision row of metrics.jsonl; reference and step from <code>config.json</code> "
           "<code>ema.b1_guard</code>")
    if r is not None:
        cut = r.get("b1_cut")
        kind = "good" if (r.get("b1_guard") == "keep" and cut is False) else "warn"
        reasons = [f"EMA box3d AP @ 2 m {b7.esc(_g(r.get('b1_ema_box3d_ap2m')))} vs the guard reference "
                   f"{b7.esc(_g(r.get('b1_ref_box3d_ap2m')))}",
                   f"head peak LR cut: {b7.esc(str(cut))} · before {b7.esc(_g(r.get('b1_head_peak_lr_before')))} · "
                   f"after {b7.esc(_g(r.get('b1_head_peak_lr_after')))} (— = JSON null: undefined, no cut applied)"]
        return b7._tile(kind, "B1 guard — box head", f"{r.get('b1_guard')} @ step {r.get('step'):,}", reasons, src)
    if gstep is None:
        return b7._tile("crit", "B1 guard — box head", "UNAVAILABLE",
                        ["config.json carries no ema.b1_guard block: the guard step is unknown"], src)
    if b["step_now"] is not None and b["step_now"] >= gstep:
        return b7._tile("crit", "B1 guard — box head", "decision row MISSING",
                        [f"the run is at step {b['step_now']:,}, past the guard step {gstep:,}, and no row carries "
                         "<code>b1_guard</code>"], src)
    return b7._tile("info", "B1 guard — box head", f"pending (decision at step {gstep:,})",
                    [f"reference EMA box3d AP @ 2 m {b7.esc(_g(b['ref']))}"], src)


def gs_read(rows8, argv_planned):
    reads = [r for r in rows8 if "gs_n_terms" in r]
    planned = []
    for t in (argv_planned or GS_PLANNED_FALLBACK).split(","):
        t = t.strip()
        if t.isdigit():
            planned.append(int(t))
    return reads, planned


def gs_html(b7, reads, planned, step_now, pace):
    esc = b7.esc
    done = {r["step"] for r in reads}
    plan = []
    for p in planned:
        if p in done:
            plan.append(f"<b>{p:,}</b> done")
        elif step_now is not None and p > step_now and pace:
            plan.append(f"<b>{p:,}</b> pending (≈ {(p - step_now) * pace / 3600:.1f} h at {pace:.2f} s/step)")
        else:
            plan.append(f"<b>{p:,}</b> pending")
    head = f'<p>Planned fp32 gradient-share readings: {" · ".join(plan)}.</p>' if planned else ""
    if not reads:
        return head + '<p class="muted">No <code>gs_*</code> reading in the log yet — nothing is shown rather than a guess.</p>'
    r = reads[-1]
    pat = re.compile(r"^gs_(.+?)_(proj|norm)_(.+)$")
    locs, terms = [], []
    for k in r:
        m = pat.match(k)
        if m and m.group(1) not in ("n", "fp32") and m.group(3) != "total":
            if m.group(1) not in locs:
                locs.append(m.group(1))
            if m.group(3) not in terms:
                terms.append(m.group(3))
    terms = sorted(terms, key=lambda t: (t == "rest", t))
    body = []
    for loc in locs:
        cells = []
        for t in terms:
            p, n = r.get(f"gs_{loc}_proj_{t}"), r.get(f"gs_{loc}_norm_{t}")
            if p is None and f"gs_{loc}_proj_{t}" not in r:
                cells.append('<td class="num"><span class="unav">UNAVAILABLE</span></td>')
            else:
                cells.append(f'<td class="num"><b>{esc(_g(p, 3))}</b><br><span class="muted">‖g‖ {esc(_g(n, 3))}</span></td>')
        shares = [r.get(f"gs_{loc}_proj_{t}") for t in terms]
        ssum = sum(shares) if all(_isnum(s) for s in shares) else None
        body.append(f'<tr><td><code>{esc(loc)}</code></td>{"".join(cells)}'
                    f'<td class="num">{esc(_g(ssum, 4))}</td><td class="num">{esc(_g(r.get(f"gs_{loc}_norm_total"), 4))}</td>'
                    f'<td class="num">{esc(_g(r.get(f"gs_{loc}_lin_rel_err"), 3))}</td></tr>')
    th = "".join(f'<th class="num">{esc(t)}<br>share (proj) / ‖g‖</th>' for t in terms)
    fp32 = ", ".join(f"{k[3:]} {_g(r[k], 3)}" for k in sorted(r) if k.startswith("gs_fp32_replay")) or "—"
    return (head + f'<h3>Latest reading — step {r["step"]:,}</h3>'
            '<div style="overflow-x:auto"><table class="vs"><tr><th>location</th>' + th +
            '<th class="num">Σ share</th><th class="num">‖g‖ total</th><th class="num">linearity rel. err</th></tr>'
            + "".join(body) + '</table></div>'
            f'<p class="note">Read from the log\'s <code>gs_*</code> keys: <code>proj</code> is each term\'s projection share '
            f'(it sums to ≈ 1 across the terms, shown as Σ share — an inference from the numbers, UNVERIFIED against '
            f'the trainer source), <code>norm</code> the gradient norm. Measurement cost {esc(_g(r.get("gs_measure_s"), 4))} s over '
            f'{esc(_g(r.get("gs_n_passes"), 2))} passes. fp32 replay flags: {esc(fp32)}.</p>')


def r8_charts(b7, tr, ev, total):
    """The r8 loss terms: train rows (EMA-smoothed, alpha .15) and the matching `eval_r8_*` dots."""
    def ser(rows, key, color, name, kind="line", smooth=True):
        xs = [r["step"] for r in rows if key in r]
        ys = [r[key] if _isnum(r.get(key)) else None for r in rows if key in r]
        return {"name": name, "xs": xs, "ys": _quant(b7.ema(ys) if smooth else ys), "color": color, "kind": kind,
                "key": key}
    xmax = max([total] + [r["step"] for r in tr[-1:]])
    out = []
    out.append(b7.chart("r8_total", "r8 total", "train (smoothed) and eval", "loss",
                        [ser(tr, "r8_total", "s1", "train r8_total"), ser(ev, "eval_r8_total", "s2", "eval r8_total", "dots", False)],
                        xmax, empty="No r8 term in the log yet."))
    out.append(b7.chart("r8_alloc", "allocation / selection terms", "train (smoothed)", "loss",
                        [ser(tr, "r8_alloc_l1", "s1", "alloc_l1"), ser(tr, "r8_sat", "s2", "sat"),
                         ser(tr, "r8_subscore", "s3", "subscore"), ser(tr, "r8_cons", "s4", "cons")],
                        xmax, empty="No r8 term in the log yet."))
    out.append(b7.chart("r8_v9", "v9 constraint and label errors", "train (smoothed)", "value",
                        [ser(tr, "r8_v9_cons", "s1", "v9_cons"), ser(tr, "r8_v9_mae_lat_c", "s2", "mae lat_c"),
                         ser(tr, "r8_v9_mae_lon_c", "s3", "mae lon_c"), ser(tr, "r8_v9_mae_speed", "s4", "mae speed")],
                        xmax, empty="No r8 term in the log yet."))
    return out


# ------------------------------------------------------------------ the page sections ----
EXTRA8_CSS = """
.band{fill:var(--s2);opacity:.20}
svg.vsplot .reflabel{paint-order:stroke;stroke:var(--surface);stroke-width:3px;stroke-linejoin:round}
.lg i.lg-band{background:var(--s2);opacity:.45;height:8px}
.lg i.lg-dash{background:none;border-top:2px dashed var(--ink2);height:0}
.dir{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.dir.good{color:var(--good)}.dir.crit{color:var(--crit)}.dir.mute{color:var(--muted)}
.unav{color:var(--crit);font-weight:600;font-size:12px}
table.vs{border-collapse:collapse;font-size:13px;margin:6px 0 4px}
table.vs td,table.vs th{padding:6px 14px 6px 0;vertical-align:top}
.vsbox{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 16px;margin:10px 0 14px}
.vsbox ul{margin:6px 0 0;padding-left:18px}.vsbox li{margin:0 0 6px;max-width:none}
.vsverdict{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:8px 0}
.flagchip{text-transform:none;letter-spacing:0;font-weight:600}
#vs-toggle{display:flex;gap:8px;margin:6px 0 10px}
#vs-toggle button{font:inherit;font-size:13px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);
 background:var(--surface);color:var(--ink);cursor:pointer}
#vs-toggle button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
#vs-curves[data-mode="full"] .vsz{display:none}#vs-curves[data-mode="zoom"] .vsf{display:none}
details.vsd{margin:10px 0}details.vsd>summary{cursor:pointer;color:var(--accent);font-weight:600}
td.gcell{text-align:center;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;padding:5px 8px}
td.gcell.good{box-shadow:inset 0 0 0 2px var(--good)}td.gcell.crit{box-shadow:inset 0 0 0 2px var(--crit)}
td.gcell.mute{box-shadow:inset 0 0 0 1px var(--line)}
.tableWrap{overflow-x:auto}
"""
TOGGLE_JS = """<script>
(function(){
  var s=document.getElementById('vs-curves'), bar=document.getElementById('vs-toggle');
  if(!s||!bar) return; bar.hidden=false;
  bar.addEventListener('click',function(e){
    var b=e.target.closest('button'); if(!b) return;
    s.setAttribute('data-mode',b.getAttribute('data-mode'));
    [].forEach.call(bar.querySelectorAll('button'),function(x){x.setAttribute('aria-pressed',x===b?'true':'false');});
  });
})();
</script>
"""


def _flag_fn(verdict):
    tac = bool(verdict["label_changed_tactical"])

    def fn(key):
        if tac and TAC_RE.search(key):
            return "tactical label counters differ — INDICATIVE ONLY"
        return DEFINITION_CHANGED.get(key)
    return fn


def augment(ev, subtract_r8):
    """Eval rows plus the one derived key `eval_loss_ex_r8` (and its `ema_` twin).
    refcv8 (`subtract_r8`): eval_loss - eval_r8_total; a row that lacks either KEY gets no derived key (it
    reads UNAVAILABLE), a null input gives None (undefined) -- never eval_loss itself.
    refcv7: its rows carry no r8 term by construction, so the key is eval_loss."""
    out = []
    for r in ev:
        r2 = dict(r)
        if subtract_r8:
            for pre in ("", "ema_"):
                if (pre + "eval_loss") in r and (pre + "eval_r8_total") in r:
                    a, b = r[pre + "eval_loss"], r[pre + "eval_r8_total"]
                    r2[pre + "eval_loss_ex_r8"] = (a - b) if (_isnum(a) and _isnum(b)) else None
        elif "eval_loss" in r:
            r2["eval_loss_ex_r8"] = r["eval_loss"] if _isnum(r["eval_loss"]) else None
        out.append(r2)
    return out


def vs_section(b7, ev7, ev8, cfg7, cfg8, verdict, ref_state, total, ref_dir, evid=None):
    esc = b7.esc
    parts = ['<h2 id="vs-refcv7">refcv8 vs refcv7 — the level refcv8 inherited</h2>']
    if ref_state["problem"]:
        parts.append(f'<div class="vsbox"><p><b class="unav">UNAVAILABLE.</b> {esc(ref_state["problem"])} '
                     'No comparison is drawn: a band computed from an incomplete reference would be a guess.</p></div>')
        return "".join(parts), {"state": "UNAVAILABLE", "problem": ref_state["problem"]}
    flag = _flag_fn(verdict)
    chip = ('<span class="chip good"><i></i>' + esc(verdict["verdict"]) + "</span>") if verdict["same"] else \
        ('<span class="chip crit"><i></i>' + esc(verdict["verdict"]) + "</span>")
    w8 = sorted({r.get("eval_windows") for r in ev8 if r.get("eval_windows") is not None})
    n_tr7 = _find_path(cfg7, "agent_join_stats", "train", "n_windows")
    n_tr8 = _find_path(cfg8, "agent_join_stats", "train", "n_windows")
    c7 = _find_path(cfg7, "v2_parity", "clips_present")
    c8 = _find_path(cfg8, "v2_parity", "clips_present")
    seed7, seed8 = (cfg7 or {}).get("seed"), (cfg8 or {}).get("seed")
    lr_last = ref_state.get("lr_last")        # refcv7's final LR: a train-row key of ITS own record
    lr8 = ref_state.get("lr8")
    cmp_note = ("Compare the <b>EMA</b> column first: refcv7's band is narrow because its learning rate had annealed to "
                f"{esc(_g(lr_last, 2))} by step {REF_FINAL_STEP:,}, so its weights stopped moving; refcv8 re-warmed to "
                f"{esc(_g(lr8, 2))}, so its raw evals scatter far more than that band.") if lr_last is not None and lr8 is not None else \
        "Compare the <b>EMA</b> column first: refcv8 re-warms its learning rate after refcv7's annealed finish, so its raw evals scatter."
    parts.append(
        '<div class="vsbox"><b>Read this first.</b><ul>'
        '<li><b>Tier T0, never driving performance.</b> Both runs\' numbers are the TRAINER\'S IN-RUN EVAL — a world-model '
        'diagnostic on a fixed held-out subset, not a closed-loop result. There is no ADE key in this eval: '
        'trajectory-type = <code>eval_traj</code> (a loss) and <code>eval_goal2s_err_m</code> (m).</li>'
        f'<li><b>One seed each</b> (refcv7 seed {esc(seed7)}, refcv8 seed {esc(seed8)}), so there is <b>no interval</b> '
        'on any Δ: a difference is a reading, not a separated effect (H-ESTIM-SEED-1: the training-seed floor is not '
        'estimated here). “Inside refcv7 band” / “within refcv8 scatter” only say a Δ is no larger than that run\'s own '
        'eval-to-eval spread.</li>'
        f'<li><b>The training corpora differ.</b> refcv7: {esc(c7)} clips / {esc(_cnt(n_tr7))} windows; refcv8: '
        f'{esc(c8)} clips / {esc(_cnt(n_tr8))} windows (the 4,572-clip view). refcv8 also changes the recipe '
        '(v9 nav command + turn distance, road-type speed label, constraint and allocation terms, EMA): a Δ is the '
        'whole refcv8 recipe on the same frames, not one lever.</li>'
        f'<li>{cmp_note}</li>'
        f'<li><b>Eval windows.</b> {chip} Both runs score {esc(", ".join(str(w) for w in w8) or "?")} windows per '
        'eval. Proof class: ' + esc(verdict["proof_class"]) + '. ' +
        ('Tactical keys are marked INDICATIVE ONLY: the tactical supervision counters (' +
         esc(", ".join(verdict["label_changed_tactical"])) + ') differ between the runs on the same windows — the '
         'tactical labels the two runs score against are not the same (refcv8 reads the v9 release via '
         '<code>--r8-v9-labels-eval</code>).' if verdict["label_changed_tactical"] else '') + '</li>'
        '</ul></div>')
    # ---- headline table
    rows = [dict(metric_row(k, lab, sense, ev7, ev8), id=i, group=g) for i, k, lab, sense, g in HEADLINE]
    parts.append('<h3>Headline metrics</h3><div class="tableWrap"><table class="vs">' + TABLE_HEAD +
                 table_rows(b7, rows, flag) + "</table></div>")
    parts.append('<p class="note">Δ = refcv8 latest − refcv7 final level; “better / worse” follows the metric\'s sense '
                 '(higher or lower is better), regardless of the size of Δ. “undefined” = the log carries a JSON null; '
                 '“UNAVAILABLE” = the key is absent from the latest eval row. Neither is drawn as 0. ' + esc(DERIVED_NOTE) + '</p>')
    # ---- window check
    def _res(c):
        w = ('<span class="dir good">ok</span>' if c["ok"] else '<span class="dir crit">FAIL</span>')
        return w + (f' <span class="muted">{esc(c["note"])}</span>' if c["note"] else "")
    ck = "".join(f'<tr><td>{esc(c["name"])}</td><td class="num">{esc(c["refcv7"])}</td>'
                 f'<td class="num">{esc(c["refcv8"])}</td><td>{_res(c)}</td></tr>' for c in verdict["checks"])
    def _mt(k):
        v = (evid or {}).get(k)
        if not (isinstance(v, str) and v.isdigit()):
            return "NOT READ"
        return f"{b7.berlin(datetime.fromtimestamp(int(v), timezone.utc)):%Y-%m-%d %H:%M} Berlin"
    lc = (f'<p class="note">Label-pipeline-changed counters (constant in each run, different across them): '
          f'{esc(", ".join(verdict["label_changed_counters"]) or "none")}. The shared eval cache was last modified '
          f'{esc(_mt("cache_mtime"))} and the eval labels {esc(_mt("labels_mtime"))} (read-only <code>stat</code> on Thor).</p>')
    parts.append('<details class="vsd"><summary>Same-windows evidence — ' + esc(verdict["verdict"]) + '</summary>'
                 '<div class="tableWrap"><table class="vs"><tr><th>check</th><th class="num">refcv7</th>'
                 '<th class="num">refcv8</th><th>result</th></tr>' + ck + '</table></div>' + lc + '</details>')
    # ---- curves
    step_last = ev8[-1]["step"] if ev8 else 0
    full_xmax = max(REF_FINAL_STEP, total)
    zoom_xmax = max(2000, int(math.ceil(step_last * 1.1 / 500.0)) * 500)
    figs = []
    for i, key, lab, sense, g in HEADLINE:
        lvl = ref_level(ev7, key)
        sub = f'{"higher" if sense == "higher" else "lower"} is better · T0 in-run eval'
        tag = (f' <span class="chip info flagchip">{esc(flag(key))}</span>' if flag(key) else "")
        for mode in ("full", "zoom"):
            figs.append(vs_chart(b7, f"vs_{i}_{mode}", lab, sub, key, ev7, ev8, lvl, mode, full_xmax, zoom_xmax, tag))
    pct = 100.0 * step_last / total if total else 0
    parts.append('<h3>Curves</h3><p>Orange band = refcv7\'s final level (what refcv8 inherited). Solid = refcv8 '
                 f'(latest eval at step {step_last:,}, {pct:.1f} % of its {total:,} steps). The full view adds refcv7\'s '
                 f'own curve over its {REF_FINAL_STEP:,} steps, dashed; the zoomed view shows refcv8\'s steps only.</p>'
                 '<div id="vs-toggle" hidden role="group" aria-label="x axis"><button type="button" data-mode="full" '
                 'aria-pressed="true">Full range</button><button type="button" data-mode="zoom" aria-pressed="false">'
                 'Zoom to refcv8 progress</button></div>'
                 '<div id="vs-curves" data-mode="full" class="grid2">' + "".join(figs) + "</div>")
    # ---- map IoU grids
    gr = {}
    for which, ek in (("raw", ""), ("EMA", "ema_")):
        trs = []
        for c in MAP_CLASSES:
            tds = []
            for b in MAP_BANDS:
                key = f"eval_map_hires_iou_{c}_{b}"
                lvl = ref_level(ev7, key)
                has = any((ek + key) in r for r in ev8)
                cell = latest(ev8, ek + key) if has else ("missing", None, None)
                j = judge("higher", lvl, cell, own_scatter(ev8, ek + key) if has else None)
                if j["delta"] is None:
                    txt, cls = ("UNAVAILABLE" if j["word"] == "UNAVAILABLE" else "undefined"), "mute"
                elif j["word"] == "equal":
                    txt, cls = f"= {_g(cell[1], 3)}", "mute"
                else:
                    txt, cls = f"{j['delta']:+.3f}", {"better": "good", "worse": "crit"}.get(j["word"], "mute")
                tds.append(f'<td class="gcell {cls}" title="{esc(key)}: refcv7 {esc(_g(lvl["mean"]) if lvl else "—")}, '
                           f'refcv8 {esc(_g(cell[1]))}">{esc(txt)}</td>')
                gr[(which, c, b)] = j["word"]
            trs.append(f'<tr><th>{esc(c)}</th>{"".join(tds)}</tr>')
        head = "".join(f"<th>{esc(b.replace('_', '–'))} m</th>" for b in MAP_BANDS)
        parts.append(f'<h3>Map IoU — Δ {which} vs refcv7 final (8 classes × 5 bands)</h3><div class="tableWrap">'
                     f'<table class="vs"><tr><th></th>{head}</tr>{"".join(trs)}</table></div>')
    parts.append('<p class="note">Green outline = better, red = worse (sign of Δ only); “= v” = identical to refcv7\'s level; hover a cell for both levels. IoU '
                 'cells with no ground-truth cells in the 128 windows read “undefined”, not 0.</p>')
    # ---- all shared keys
    keys, n7k, n8k = shared_keys(ev7, ev8)
    allrows, no_sense = [], 0
    for k in keys:
        s = classify(k)
        if s is None:
            no_sense += 1
            continue
        allrows.append(metric_row(k, k, s, ev7, ev8))
    tally = {}
    for which, f in (("raw", lambda r: r["j_raw"]), ("EMA", lambda r: r["j_ema"])):
        c = {"better": 0, "worse": 0, "equal": 0, "other": 0}
        for r in allrows:
            j = f(r)
            if j is None:
                continue
            c[j["word"] if j["word"] in c else "other"] += 1
        tally[which] = c
    parts.append(
        f'<details class="vsd"><summary>All {len(allrows)} shared metrics with a defined better/worse sense '
        f'(of {len(keys)} shared eval keys; {no_sense} are counters, label shares, flags or gates)</summary>'
        f'<p class="note">Row tally, not evidence — the rows are strongly correlated. raw: {tally["raw"]["better"]} better / '
        f'{tally["raw"]["worse"]} worse / {tally["raw"]["equal"]} equal / {tally["raw"]["other"]} undefined-or-unavailable; '
        f'EMA: {tally["EMA"]["better"]} better / {tally["EMA"]["worse"]} worse / {tally["EMA"]["equal"]} equal / '
        f'{tally["EMA"]["other"]} undefined-or-unavailable.</p>'
        '<div class="tableWrap"><table class="vs">' + TABLE_HEAD + table_rows(b7, allrows, flag) + '</table></div></details>')

    def ser_row(r):
        lv = r["lvl"]
        return {"key": r["key"], "sense": r["sense"],
                "refcv7_mean": lv["mean"] if lv else None, "refcv7_sd": lv["sd"] if lv else None,
                "refcv7_n": lv["n"] if lv else 0, "refcv8_raw": r["raw"][1], "refcv8_raw_state": r["raw"][0],
                "refcv8_step": r["raw"][2], "delta_raw": r["j_raw"]["delta"], "direction_raw": r["j_raw"]["word"],
                "refcv8_ema": r["ema"][1] if r["has_ema"] else None, "refcv8_ema_state": r["ema"][0] if r["has_ema"] else "no-key",
                "delta_ema": r["j_ema"]["delta"] if r["j_ema"] else None,
                "direction_ema": r["j_ema"]["word"] if r["j_ema"] else None}
    summ = {"state": "built", "windows": {k: verdict[k] for k in ("same", "verdict", "failed", "proof_class",
                                                                  "n_counters_constant", "n_counters_identical",
                                                                  "label_changed_counters", "label_changed_tactical",
                                                                  "counters_varying")},
            "windows_checks": verdict["checks"], "reference": {"last_step": ref_state["last_step"], "band_n": REF_BAND_N,
                                                                "dir": ref_dir},
            "headline": [ser_row(r) for r in rows], "all_shared": {"n": len(allrows), "tally": tally,
                                                                   "no_sense": no_sense, "shared_keys": len(keys)}}
    return "".join(parts), summ


def _cnt(x):
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else "?"


def additions_section(b7, rows8, tr, ev, cfg8, summary8):
    esc = b7.esc
    argv = (cfg8 or {}).get("argv")
    step_now = summary8["step"]
    pace = summary8.get("pace_s")
    tiles = []
    b = b1_read(rows8, cfg8, step_now)
    tiles.append(b1_tile(b7, b))
    # allocation emission
    emit_start = b7._argv_val(argv, "--r8-alloc-emit-start")
    es = [(r["step"], r["r8_emit_frac"]) for r in tr if _isnum(r.get("r8_emit_frac"))]
    first_full = next((s for s, v in es if v >= 1.0), None)
    if not es:
        tiles.append(b7._tile("crit", "Allocation emission", "UNAVAILABLE",
                              ["no train row carries <code>r8_emit_frac</code>"], "train rows of metrics.jsonl"))
    else:
        kind = "good" if (emit_start and first_full is not None and first_full >= int(emit_start)) else "warn"
        tiles.append(b7._tile(kind, "Allocation emission", f"emit_frac {es[-1][1]:.3f} at step {es[-1][0]:,}",
                              [f"switch step from the argv: {esc(emit_start or 'UNAVAILABLE')} (<code>--r8-alloc-emit-start</code>)",
                               f"first logged row at emit_frac 1.000: {esc(f'{first_full:,}' if first_full else 'none yet')} "
                               f"(rows are logged every 50 steps; the switch row itself is logged before it takes effect)",
                               f"before the switch the logged value is {esc(_g(es[0][1], 4))} — the meaning of a value "
                               "below 1 is UNVERIFIED (read as a fraction of the batch)"],
                              "<code>r8_emit_frac</code> of the train rows"))
    # pace vs smoke
    if pace is None:
        tiles.append(b7._tile("info", "Pace vs the smoke", "not measurable yet", [], "train rows"))
    else:
        tiles.append(b7._tile("good" if pace <= b7.PACE_PI_APPROVED_S else "warn", "Pace vs the smoke",
                              f"{pace:.2f} s/step",
                              [f"G-LIVE smoke pace {SMOKE_PACE_S} s/step (INHERITED: measured by the launch, not here); "
                               f"now {pace - SMOKE_PACE_S:+.2f} s/step against it",
                               f"warm median {esc(_g(summary8.get('pace_warm_median_s'), 3))} s/step (intervals with no eval "
                               f"or checkpoint)",
                               f"PI-approved ceiling {b7.PACE_PI_APPROVED_S} s/step — refcv7's approval, which SPEC_REFCV8 "
                               "prescribes; amber only above it"], "marginal pace over the last logged rows"))
    parts = ['<h2 id="refcv8-additions">refcv8\'s own additions</h2><div class="alarms">' + "".join(tiles) + "</div>"]
    # loss weights from the argv
    wts = [(f, b7._argv_val(argv, f)) for f in ("--w-r8-cons", "--w-r8-alloc-l1", "--w-r8-listwise", "--w-r8-sat",
                                                 "--w-r8-subscore", "--w-r8-v9-cons")]
    parts.append('<p>r8 loss weights as launched (from <code>config.json</code> argv): ' +
                 " · ".join(f"<code>{esc(f)}</code> {esc(v if v is not None else 'UNAVAILABLE')}" for f, v in wts) +
                 f'. EMA decay {esc(_find_path(cfg8, "ema", "decay") if cfg8 else "UNAVAILABLE")}; its eval keys are '
                 '<code>ema_eval_*</code> and are drawn beside the raw eval in every curve above.</p>')
    reads, planned = gs_read(rows8, b7._argv_val(argv, "--grad-share-at"))
    parts.append("<h3>Gradient-share readings (fp32)</h3>" + gs_html(b7, reads, planned, step_now, pace))
    parts.append('<h3>r8 loss terms</h3><div class="grid2">' + "".join(r8_charts(b7, tr, ev, summary8["total"])) + "</div>")
    summ = {"b1_guard": {"decision": (b["row"] or {}).get("b1_guard"), "step": (b["row"] or {}).get("step"),
                         "cut": (b["row"] or {}).get("b1_cut"), "guard_step": b["guard_step"], "ref_ap2m": b["ref"]},
            "grad_share": {"planned": planned, "done": [r["step"] for r in reads]},
            "alloc_emit": {"start": emit_start, "first_full_step": first_full,
                           "last": es[-1] if es else None},
            "pace_vs_smoke_s": (pace - SMOKE_PACE_S) if pace is not None else None}
    return "".join(parts), summ


NAVSIM_REPLACEMENT = (
    '<h2>NavSim and the T1 battery</h2>\n<p class="muted">No refcv8 NavSim or four-family battery milestone is read by '
    'this page: the EvalFlyWheel banks them per checkpoint, separately, and the refcv7 package is deliberately NOT '
    'shown here — nothing is shown rather than another arm\'s numbers.</p>\n')


def inject(page, hooks):
    """Apply (old, new, count) literal replacements; every one must match EXACTLY `count` times."""
    for old, new, count in hooks:
        n = page.count(old)
        if n != count:
            raise SystemExit(f"ZZWATCH-HOOK-MISSING expected {count} x {old[:80]!r}, found {n}: the refcv7 page "
                             "layout changed under build_watch_refcv8.py")
        page = page.replace(old, new)
    return page


def post_process(page, vs_html, add_html):
    page = re.sub(r'<link rel="preconnect"[^>]*>', "", page)
    page = re.sub(r'<link rel="stylesheet" href="https?://[^"]*">', "", page)
    m = re.search(r"<h2>NavSim — benchmark KPIs per checkpoint</h2>.*?(?=<h2>What it says, and what it does not</h2>)",
                  page, flags=re.S)
    if not m:
        raise SystemExit("ZZWATCH-HOOK-MISSING NavSim section not found in the refcv7 page")
    page = page[:m.start()] + NAVSIM_REPLACEMENT + page[m.end():]
    page = inject(page, [
        ("<title>refcv7 Training Watch</title>", "<title>refcv8 Training Watch</title>", 1),
        ("<h1>refcv7 Training Watch</h1>", "<h1>refcv8 Training Watch</h1>", 1),
        ("</style>\n<div class=\"wrap\">", EXTRA8_CSS + "</style>\n<div class=\"wrap\">", 1),
        ("<h2>Progress — planning</h2>", vs_html + add_html + "\n<h2>Progress — planning</h2>", 1),
        ("built by <code>taniteval/tools/training_watch/build_watch_refcv7.py</code>",
         "built by <code>taniteval/tools/training_watch/build_watch_refcv8.py</code> (reusing "
         "<code>build_watch_refcv7.py</code>)", 1),
        ("contracts: SPEC_REFCV7 A8", "contracts (refcv7's, unchanged; SPEC_REFCV8 is a draft): SPEC_REFCV7 A8", 1),
    ])
    if re.search(r'(?:src|href)="https?://', page):
        raise SystemExit("ZZWATCH-NOT-SELF-CONTAINED the page still references an external URL")
    return page + TOGGLE_JS


# ------------------------------------------------------------------ build ----
def build(watch_dir: str | None = None, metrics_path: str | None = None, arm: str | None = None,
          ref_dir: str | None = None):
    """(page, summary). Reads refcv8's artifacts from `watch_dir` (default L) and refcv7's complete record
    from `ref_dir` (default REF_DIR). Every missing artifact renders UNAVAILABLE rather than stopping the build."""
    arm = arm or ARM
    b7 = configure(load_refcv7(), arm)
    wd = watch_dir or (os.path.dirname(os.path.abspath(metrics_path)) if metrics_path else L)
    mp = metrics_path or os.path.join(wd, "metrics.jsonl")
    cfg8 = b7._jload(os.path.join(wd, "config.json"))
    cfg8 = cfg8 if isinstance(cfg8, dict) else {}
    # the refcv7 builder reads the run's own --steps and prints a note when it differs from TOTAL_SPEC
    page, summary = b7.build(watch_dir=wd, metrics_path=metrics_path, arm=arm)
    total = summary["total"]
    rows8, tr, ev8, _cd, _er = b7.load(mp)
    # ---- refcv7's complete record
    rd = ref_dir or REF_DIR
    ref_state = {"problem": None, "last_step": None, "lr_last": None, "lr8": None}
    ev7, cfg7 = [], {}
    rmp = os.path.join(rd, "metrics.jsonl")
    if not os.path.isfile(rmp):
        ref_state["problem"] = f"refcv7's complete record is not at {rmp}"
    else:
        rows7, tr7, ev7, _a, _b = b7.load(rmp)
        cfg7 = b7._jload(os.path.join(rd, "config.json"))
        cfg7 = cfg7 if isinstance(cfg7, dict) else {}
        ref_state["last_step"] = rows7[-1].get("step") if rows7 else None
        if not ev7 or ref_state["last_step"] != REF_FINAL_STEP or ev7[-1]["step"] != REF_FINAL_STEP:
            ref_state["problem"] = (f"refcv7's record ends at step {ref_state['last_step']}, not {REF_FINAL_STEP:,}: "
                                    "its final level is not final")
        ref_state["lr_last"] = tr7[-1].get("lr") if tr7 else None
    ref_state["lr8"] = tr[-1].get("lr") if tr else None
    evid = b7._jload(os.path.join(wd, EVID_FILE))
    verdict = windows_verdict(cfg7, cfg8, ev7, ev8, evid)      # on the logged rows, before the derived key
    ev7 = augment(ev7, False) if not any(k.startswith("eval_r8_") for r in ev7 for k in r) else ev7
    ev8 = augment(ev8, True)
    vs_html, vs_summ = vs_section(b7, ev7, ev8, cfg7, cfg8, verdict, ref_state, total, rd, evid)
    add_html, add_summ = additions_section(b7, rows8, tr, ev8, cfg8, summary)
    page = post_process(page, vs_html, add_html)
    summary = dict(summary, vs_refcv7=vs_summ, refcv8=add_summ, builder="build_watch_refcv8.py", total_spec=TOTAL_SPEC)
    return page, summary


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])

    def opt(name):
        return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else None
    arm = opt("--arm") or ARM
    metrics = opt("--metrics")
    wd = opt("--dir") or (None if metrics else L)
    if metrics is None and "--no-pull" not in a:
        b7 = load_refcv7()
        run = opt("--run") or os.environ.get("REFCV8_RUN")
        if not run:
            run, _why = b7.resolve_run(arm, None)
        run = run or RUN8_DEFAULT
        if arm not in run:      # b7.resolve_run honours $REFCV7_RUN: never pull another arm's run by accident
            raise SystemExit(f"ZZWATCH-PULL-FAIL the run {run!r} does not belong to {arm!r} "
                             "(is $REFCV7_RUN set? pass --run)")
        b7.pull(run, arm, wd)
        pull_evidence(run, wd, b7._jload(os.path.join(wd, "config.json")))
    out = opt("--out") or os.path.join(wd or os.path.dirname(os.path.abspath(metrics)), "refcv8_training_watch.html")
    page, summary = build(watch_dir=wd, metrics_path=metrics, arm=arm, ref_dir=opt("--ref-dir"))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    sp = opt("--summary") or (os.path.splitext(out)[0] + ".summary.json")
    with open(sp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"wrote {out} ({len(page.encode('utf-8')):,} B)")
    print(f"wrote {sp}")
    w = summary["vs_refcv7"].get("windows", {})
    print(f"ZZWATCH8 step={summary['step']}/{summary['total']} windows={w.get('verdict', 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
