#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_mapft_argv.py -- MAPFT launch argv generator (PREREG_MAPFT sealed sha256 f8774a7b..., section 1).

Builds, from refcv8's COMMITTED launch argv (stack/ops/runs.d/refcv8-r101-s0.argv.json, argv sha256 188aa439...),

  smoke : the 50-step TIMING SMOKE argv  (mapft_argv_smoke.json)
  derive: the budget N from the MEASURED  s_step  and  smoke_wall_s       (no files written)
  build : the FULL launch argv  (mapft_argv_full.json)  -- budget rule applied EXACTLY, or STOP
  audit : re-derive N from the same two numbers and compare with an argv file
  self-test

THE SEALED BUDGET RULE (PREREG_MAPFT s1, verbatim):

    N = min(10000, floor(0.85 * (21600 - smoke_wall_s - 1800) / s_step))        1800 s reserved for in-run evals + save
    if N < 1000: the run does NOT launch (STOP; report s_step to the PI)
    --steps N   --warmup 200   --lr 1e-4   --eval-every max(500, N // 4)   --ema-decay round(1 - 10 / N, 5)

  * `floor(...)` is evaluated in EXACT rational arithmetic (fractions.Fraction over the decimal strings), not in IEEE
    doubles: the rule is a mathematical statement, and a double can turn an exact integer into k-eps (-> k-1) and
    flip a STOP. The float value is computed too and a disagreement is PRINTED (it never decides).
  * `round(1 - 10 / N, 5)` is the literal Python expression of the seal (a double, Python's own rounding): at an exact
    tie (e.g. N = 3200 -> 0.996875) it gives 0.99687, and that is what the file carries.

  s_step is the trainer's own per-step figure: refc_v3_train.py writes `elapsed_s` = time.time() - t0 at every LOGGED
  step -- CUMULATIVE wall seconds, NOT divided by anything (`elapsed_s=round(time.time() - t0, 1)`, :10996). So s_step =
  (elapsed_s[last] - elapsed_s[first_used]) / (step[last] - step[first_used]), the divisor being the STEP DIFFERENCE.
  (Contrast train_v6_staged.py, which logs an already-divided step_s: CLAUDE.md "the rule is INVERTED" trap -- this
  trainer is the cumulative kind.) The first rows (steps < --from-step, default 20) carry cuDNN autotune / lazy
  allocation and are not the run's cost.

Exit codes: 0 ok | 2 bad input | 3 STOP (N < 1000; the marker line is  ZZMAPFT-STOPZZ ).
Stdout markers are opaque ZZ...ZZ tokens, disjoint from anything a watcher greps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

# ----------------------------------------------------------------------------------------------- the seal
BUDGET_S = 21600            # 6 Thor-hours, the PI's cap (D-R9-MAPHEAD-PRECISION)
RESERVE_S = 1800            # in-run evals + the save
FRAC = Fraction(85, 100)    # 0.85
N_CAP = 10000
N_MIN = 1000
WARMUP = 200
LR = "1e-4"
EVAL_FLOOR = 500
EMA_HORIZON_DIV = 10        # ema horizon = N / 10

BASE_ARGV_SHA256 = "188aa439258d28e712e9a1b4a8ebda982d5f068b4515022f54ba1172e18fd956"   # refcv8-r101-s0 launch argv
BASE_NTOKENS = 210
ARM = "refcv8-mapft-s0"
# Thor layout (read-only verified 2026-10-26; /home/nvidia/r8ns is the refcv8/NT launch root)
THOR_ROOT = "/home/nvidia/r8ns/mapft"
LAUNCH_TREE = THOR_ROOT + "/launch_tree"
RUN_OUT = THOR_ROOT + "/runs/" + ARM
SMOKE_OUT = THOR_ROOT + "/runs/refcv8-mapft-smoke50"
INIT_FROM = "/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt"          # md5 b4364029b68f29a33eeb4aebbbbd002b, 403,200,712 B
JOIN_MASKS_REL = "stack/tanitad/configs/refcv8_join_label_defects_train4572.json"

#: flags the map-only fine-tune REMOVES from refcv8's argv (map_only_ft.pin_args refuses each; gate profile forbids)
REMOVE = ("--b1-guard-ref-ap2m", "--b1-guard-step", "--grad-share-every", "--grad-share-terms", "--grad-share-at",
          "--conflict-every")


class MapftError(SystemExit):
    pass


# ----------------------------------------------------------------------------------------------- the budget
def _frac(x) -> Fraction:
    """Decimal intent, not the binary value: Fraction('3.1') not Fraction(3.1)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise MapftError("a number was expected, got a bool")
    if isinstance(x, int):
        return Fraction(x)
    try:
        f = Fraction(str(x).strip())
    except (ValueError, ZeroDivisionError):
        raise MapftError(f"not a number: {x!r}")
    return f


def derive_budget(s_step, smoke_wall_s) -> dict:
    """The sealed rule. Returns the whole record; `stop` is True when N < 1000 (then no argv may be written)."""
    s = _frac(s_step)
    w = _frac(smoke_wall_s)
    if s <= 0:
        raise MapftError(f"s_step must be > 0 (got {s_step!r})")
    if w < 0:
        raise MapftError(f"smoke_wall_s must be >= 0 (got {smoke_wall_s!r})")
    avail = Fraction(BUDGET_S) - w - Fraction(RESERVE_S)
    raw = FRAC * avail / s                                  # exact
    n_exact = math.floor(raw) if avail > 0 else 0
    n_float = None
    try:                                                    # the double version of the seal's text, DISCLOSED only
        n_float = math.floor(0.85 * (21600 - float(w) - 1800) / float(s))
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    n = min(N_CAP, n_exact)
    stop = n < N_MIN
    rec = {
        "rule": "N = min(10000, floor(0.85 * (21600 - smoke_wall_s - 1800) / s_step)); N < 1000 -> STOP",
        "s_step": str(s_step), "smoke_wall_s": str(smoke_wall_s),
        "available_s": str(avail), "n_raw_exact": float(raw), "n_floor_exact": int(n_exact),
        "n_floor_float": n_float, "n_float_disagrees": (n_float is not None and n_float != n_exact),
        "N": int(n), "capped": bool(n_exact > N_CAP), "stop": bool(stop),
    }
    if not stop:
        rec.update({
            "steps": int(n), "warmup": WARMUP, "lr": LR,
            "eval_every": max(EVAL_FLOOR, n // 4),
            "ema_decay": round(1 - EMA_HORIZON_DIV / n, 5),
            "projected_train_s": float(n * s),
            "projected_total_s": float(n * s + w + RESERVE_S),
        })
    return rec


# ----------------------------------------------------------------------------------------------- smoke reader
def read_smoke_metrics(path, from_step: int = 20) -> dict:
    """s_step from a smoke's metrics.jsonl: Delta(elapsed_s) / Delta(step) over the logged rows with step >= from_step.
    Structural problems are REFUSED (a number computed from a malformed log is worse than none)."""
    rows = {}
    n_bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line, parse_float=str)       # keep elapsed_s as the decimal string the trainer wrote
            except Exception:
                n_bad += 1                                  # a torn final line is skipped, counted
                continue
            if isinstance(r, dict) and "step" in r and "elapsed_s" in r and "loss" in r:
                rows[int(r["step"])] = r["elapsed_s"]
    steps = sorted(rows)
    use = [s for s in steps if s >= from_step]
    if len(use) < 2:
        raise MapftError(f"need >= 2 logged train rows with step >= {from_step}; have steps {steps}")
    e = [_frac(rows[s]) for s in use]
    if any(b <= a for a, b in zip(e, e[1:])):
        raise MapftError(f"elapsed_s is not strictly increasing over steps {use}: {[str(x) for x in e]}")
    s_step = (e[-1] - e[0]) / (use[-1] - use[0])
    ivs = [float((e[i + 1] - e[i]) / (use[i + 1] - use[i])) for i in range(len(use) - 1)]
    spread = (max(ivs) / min(ivs) - 1.0) if min(ivs) > 0 else float("inf")
    return {"s_step": s_step, "s_step_float": float(s_step), "rows_used": use, "intervals_s_per_step": ivs,
            "interval_spread": spread, "from_step": from_step, "n_unparseable_lines": n_bad,
            "warn_unstable": bool(spread > 0.25),
            "divisor": "step difference (elapsed_s is cumulative wall since the trainer's t0, not per-step)"}


# ----------------------------------------------------------------------------------------------- argv algebra
def groups(argv: list) -> list:
    """[(flag, [values...])] -- refuses a leading/positional token (the base has none)."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if not t.startswith("--"):
            raise MapftError(f"positional token {t!r} at {i}: the base argv has none")
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        out.append((t, list(argv[i + 1:j])))
        i = j
    return out


def flatten(gs: list) -> list:
    return [x for f, v in gs for x in (f, *v)]


def _find(gs, flag):
    idx = [i for i, (f, _v) in enumerate(gs) if f == flag]
    if len(idx) > 1:
        raise MapftError(f"{flag} appears {len(idx)} times")
    return idx[0] if idx else None


def op_remove(gs, flag, edits):
    i = _find(gs, flag)
    if i is None:
        raise MapftError(f"cannot remove {flag}: not in the base argv")
    edits.append(f"remove {flag} {' '.join(gs[i][1])}".rstrip())
    del gs[i]


def op_set(gs, flag, vals, edits, must_exist=True):
    vals = [str(v) for v in vals]
    i = _find(gs, flag)
    if i is None:
        if must_exist:
            raise MapftError(f"cannot set {flag}: not in the base argv")
        gs.append((flag, vals))
        edits.append(f"add {flag} {' '.join(vals)}".rstrip())
    else:
        edits.append(f"set {flag}: {' '.join(gs[i][1])} -> {' '.join(vals)}")
        gs[i] = (flag, vals)


def load_base(path) -> list:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    a = d["argv"] if isinstance(d, dict) else d
    if not (isinstance(a, list) and all(isinstance(x, str) for x in a)):
        raise MapftError("base argv is not a list of strings")
    sha = hashlib.sha256(json.dumps(a, separators=(",", ":")).encode("utf-8")).hexdigest()
    if sha != BASE_ARGV_SHA256 or len(a) != BASE_NTOKENS:
        raise MapftError(f"base argv is NOT refcv8-r101-s0's launch argv (sha256 {sha[:16]} / {len(a)} tokens; "
                         f"want {BASE_ARGV_SHA256[:16]} / {BASE_NTOKENS})")
    return a


def mapft_core(base: list, launch_tree: str, init_from: str) -> tuple:
    """refcv8's launch argv -> the map-only argv with NO budget yet. (groups, edits)"""
    gs, edits = groups(list(base)), []
    for f in REMOVE:
        op_remove(gs, f, edits)
    op_set(gs, "--init-from", [init_from], edits)
    op_set(gs, "--join-defect-masks", [f"{launch_tree}/{JOIN_MASKS_REL}"], edits)
    if gs[_find(gs, "--lr")][1] != [LR]:
        raise MapftError(f"base --lr is {gs[_find(gs, '--lr')][1]}, expected [{LR}] (refcv8's peak)")
    op_set(gs, "--r8-train-only-map", [], edits, must_exist=False)
    op_set(gs, "--conflict-detector", ["off"], edits, must_exist=False)
    return gs, edits


def build_full(base: list, budget: dict, launch_tree=LAUNCH_TREE, run_out=RUN_OUT, init_from=INIT_FROM) -> tuple:
    if budget["stop"]:
        raise MapftError("N < 1000: no launch argv may be built (STOP)")
    gs, edits = mapft_core(base, launch_tree, init_from)
    op_set(gs, "--out", [run_out], edits)
    op_set(gs, "--steps", [budget["steps"]], edits)
    op_set(gs, "--warmup", [budget["warmup"]], edits)
    op_set(gs, "--eval-every", [budget["eval_every"]], edits)
    op_set(gs, "--ema-decay", [repr(budget["ema_decay"])], edits)
    return flatten(gs), edits


def build_smoke(base: list, launch_tree=LAUNCH_TREE, smoke_out=SMOKE_OUT, init_from=INIT_FROM) -> tuple:
    """The 50-step TIMING smoke: the launch argv with a 50-step budget. --log-every 10 gives >= 2 rows after the warm-up
    (elapsed_s is cumulative, so s_step needs a difference); no in-run eval (the gate's G-LIVE smoke owns the eval
    path); one save at step 50 (rolling ckpt + EMA + the FrozenStateGuard at a save step)."""
    gs, edits = mapft_core(base, launch_tree, init_from)
    op_set(gs, "--out", [smoke_out], edits)
    op_set(gs, "--steps", ["50"], edits)
    op_set(gs, "--warmup", ["10"], edits)
    op_set(gs, "--eval-every", ["0"], edits)
    op_set(gs, "--ema-decay", ["0.99"], edits)
    op_set(gs, "--log-every", ["10"], edits)
    op_set(gs, "--save-every", ["50"], edits)
    return flatten(gs), edits


def sha_compact(argv: list) -> str:
    return hashlib.sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def flag_val(argv: list, flag: str):
    for gf, vals in groups(list(argv)):
        if gf == flag:
            return vals
    return None


def audit_argv(argv: list, budget: dict) -> list:
    """Compare an argv's budget flags with the rule's output. [] = consistent. (Determinism, not correctness:
    the correctness of derive_budget is pinned by the literal cases in test_make_mapft_argv.py.)"""
    bad = []
    if budget["stop"]:
        return ["the rule says STOP (N < 1000): no argv may exist"]
    want = {"--steps": [str(budget["steps"])], "--warmup": [str(budget["warmup"])], "--lr": [LR],
            "--eval-every": [str(budget["eval_every"])], "--ema-decay": [repr(budget["ema_decay"])]}
    for f, v in want.items():
        got = flag_val(argv, f)
        if got != v:
            bad.append(f"{f}: argv has {got}, the rule gives {v}")
    for f in REMOVE:
        if flag_val(argv, f) is not None:
            bad.append(f"{f} present (must be removed)")
    if flag_val(argv, "--r8-train-only-map") != []:
        bad.append("--r8-train-only-map missing (or carries a value)")
    if flag_val(argv, "--conflict-detector") != ["off"]:
        bad.append("--conflict-detector is not `off`")
    return bad


# ----------------------------------------------------------------------------------------------- self-test
def self_test() -> int:
    # LITERALS worked by hand (see test_make_mapft_argv.py for the full set): this is only a smoke of the plumbing
    b = derive_budget("3.0", "600")
    assert (b["N"], b["eval_every"], b["ema_decay"], b["stop"]) == (5440, 1360, 0.99816, False), b
    assert derive_budget("17.0", "600")["stop"] is True
    print("SELFTEST-OK")
    return 0


# ----------------------------------------------------------------------------------------------- CLI
def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # default=str: the smoke record carries exact Fractions -> written as "p/q" strings (a Fraction is not JSON)
    # bytes, not write_text: on Windows text mode turns every \n into \r\n, so a file built on the dev box would differ from one built on Thor
    Path(path).write_bytes((json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n").encode("utf-8"))


def _budget_from_args(a) -> tuple:
    smoke_info = None
    if getattr(a, "from_smoke_metrics", None):
        smoke_info = read_smoke_metrics(a.from_smoke_metrics, a.from_step)
        if smoke_info["warn_unstable"]:
            print(f"WARN-UNSTABLE: per-interval s/step spread {smoke_info['interval_spread']:.2f} (> 0.25): "
                  f"{smoke_info['intervals_s_per_step']} -- the budget uses the pooled value, as sealed")
        s_step = smoke_info["s_step"]
    elif a.s_step is not None:
        s_step = a.s_step
    else:
        raise MapftError("give --s-step or --from-smoke-metrics")
    if a.smoke_wall_s is None:
        raise MapftError("--smoke-wall-s is required (the smoke process's wall seconds, launch to exit)")
    return derive_budget(s_step, a.smoke_wall_s), smoke_info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_budget=True):
        if need_budget:
            p.add_argument("--s-step", default=None, help="measured seconds per step (decimal)")
            p.add_argument("--from-smoke-metrics", default=None, help="the smoke's metrics.jsonl (derives s_step)")
            p.add_argument("--from-step", type=int, default=20)
            p.add_argument("--smoke-wall-s", default=None, help="wall seconds of the smoke process, launch to exit")

    p = sub.add_parser("derive"); common(p)
    p = sub.add_parser("smoke")
    p.add_argument("--base", required=True); p.add_argument("--out", required=True)
    p.add_argument("--launch-tree", default=LAUNCH_TREE); p.add_argument("--smoke-out", default=SMOKE_OUT)
    p.add_argument("--init-from", default=INIT_FROM)
    p = sub.add_parser("build"); common(p)
    p.add_argument("--base", required=True); p.add_argument("--out", required=True)
    p.add_argument("--launch-tree", default=LAUNCH_TREE); p.add_argument("--run-out", default=RUN_OUT)
    p.add_argument("--init-from", default=INIT_FROM)
    p = sub.add_parser("audit"); common(p)
    p.add_argument("--argv", required=True)
    sub.add_parser("self-test")
    a = ap.parse_args(argv)

    if a.cmd == "self-test":
        return self_test()
    if a.cmd == "smoke":
        base = load_base(a.base)
        av, edits = build_smoke(base, a.launch_tree, a.smoke_out, a.init_from)
        _write_json(a.out, {"schema": "tanitad.launch_argv/1", "arm": ARM + "-smoke50", "stage": "timing-smoke",
                            "what": "PREREG_MAPFT s1 50-step timing smoke: the launch argv with a 50-step budget "
                                    "(NOT gated, NOT a launch; measures s_step and smoke_wall_s)",
                            "base_argv_sha256": BASE_ARGV_SHA256, "edits": edits, "argv": av,
                            "argv_sha256": sha_compact(av)})
        print(f"SMOKE-ARGV {len(av)} tokens sha256 {sha_compact(av)[:16]} -> {a.out}")
        print("ZZMAPFT-SMOKEARGV-OKZZ")
        return 0

    budget, smoke_info = _budget_from_args(a)
    if budget["n_float_disagrees"]:
        print(f"NOTE: the double-precision reading of the rule gives N={budget['n_floor_float']}, the exact reading "
              f"gives {budget['n_floor_exact']} -- the exact one is used (the rule is a mathematical statement)")
    if a.cmd == "derive":
        print(json.dumps({"budget": budget, "smoke": smoke_info}, indent=1, default=str))
        if budget["stop"]:
            print(f"ZZMAPFT-STOPZZ N={budget['N']} < {N_MIN}: do NOT launch; report s_step={budget['s_step']} to the PI")
            return 3
        print(f"ZZMAPFT-N-{budget['N']}ZZ")
        return 0
    if a.cmd == "audit":
        d = json.loads(Path(a.argv).read_text(encoding="utf-8"))
        av = d["argv"] if isinstance(d, dict) else d
        bad = audit_argv(av, budget)
        for b in bad:
            print("AUDIT-FAIL:", b)
        if not bad:
            print(f"AUDIT-OK N={budget['N']} (the argv equals the rule's output for these two numbers)")
        return 0 if not bad else 2
    if a.cmd == "build":
        if budget["stop"]:
            _write_json(str(Path(a.out).with_suffix("")) + "_STOP.json", {"budget": budget, "smoke": smoke_info})
            print(f"ZZMAPFT-STOPZZ N={budget['N']} < {N_MIN}: s_step={budget['s_step']} smoke_wall_s={budget['smoke_wall_s']} "
                  f"-> do NOT launch; report to the PI (cap 6 Thor-h)")
            return 3
        base = load_base(a.base)
        av, edits = build_full(base, budget, a.launch_tree, a.run_out, a.init_from)
        bad = audit_argv(av, budget)
        if bad:
            raise MapftError("internal: the built argv fails its own audit: " + "; ".join(bad))
        _write_json(a.out, {
            "schema": "tanitad.launch_argv/1", "arm": ARM, "stage": "launch",
            "what": "refcv8 map-only fine-tune (PI D-R9-MAPHEAD-PRECISION), PREREG_MAPFT sealed f8774a7b: refcv8's "
                    "launch argv + --r8-train-only-map --conflict-detector off, warm start = the shipped EMA model, "
                    "budget from the measured smoke",
            "base_argv_sha256": BASE_ARGV_SHA256, "budget": budget, "smoke": smoke_info, "edits": edits,
            "argv": av, "argv_sha256": sha_compact(av)})
        print(f"BUILT {len(av)} tokens, N={budget['N']} steps, warmup {WARMUP}, eval-every {budget['eval_every']}, "
              f"ema-decay {budget['ema_decay']}, projected train {budget['projected_train_s']/3600:.2f} h "
              f"(total incl. smoke+reserve {budget['projected_total_s']/3600:.2f} h) sha256 {sha_compact(av)[:16]} -> {a.out}")
        print(f"ZZMAPFT-BUILT-{budget['N']}ZZ")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
