#!/usr/bin/env python3
"""p_floor.py — report the P-panel's run-to-run floor, in the order committed BEFORE the numbers.

⛔ THE RULE THIS FILE ENFORCES (Master Mind, 2026-09-20, committed in advance precisely so it
could not drift once the numbers existed):

* **`|P0 − P0b|` — same pinned tree, seeds 1 vs 2, nothing else moved — IS THE SEED FLOOR.**
  That is the number that gets quoted, and no arm may be called SUPPORTED by a smaller margin.
* ⛔ **`|P0 − A8|` is an UPPER BOUND and is NEVER called the floor.** A8 carries no commit, no
  code hash and no trainer md5, and the repo's trainer already differed from the pinned tree by
  313 diff lines, so that difference is `seed + an unquantified CODE DELTA`. It is reported
  BESIDE the floor, labelled — because the **gap between the two is the size of the code delta**,
  which is worth knowing on its own.
* **The floor is reported BEFORE anything is interpreted against it.**

:func:`report` therefore emits the floor first and refuses to emit anything at all when the
second replicate is missing — a floor derived from one replicate is not a floor, it is a guess,
and `H-ESTIM-SEED-1` is the rule it would violate.

⛔ AND EVERY FAMILY IS REPORTED SEPARATELY. `EVAL_DOCTRINE`'s four-families rule binds here:
longitudinal · lateral · tactical · strategic, plus the perception/box family this panel is about.
A single pooled score hides exactly the trade-off the panel exists to see, so :func:`families`
returns a dict per family and :func:`report` never sums across them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: ⛔⛔ THE HEADLINE IS THE **360°** POPULATION, NOT THE GATE'S. `eval_box3d_centre` is
#: `slot_set_loss`'s centre term: `Σ|pred − tgt|` over **every matched row**, divided by their
#: count (`agent_slots.py:577`) — no population filter. `PREREG_PERCEPTION_BOX_QUALITY` states
#: its targets per population (**near-forward 6.06 m PRIMARY**, all-matched 12.04 m secondary)
#: and every SUPPORTED criterion for `P1`/`P3`/`P5` is written on the **near-forward** error.
#: ⇒ A floor quoted from this key and compared against a near-forward verdict is a
#: CROSS-POPULATION comparison and is inadmissible. The two populations differ by ~2× on the
#: banked read, so it is not a rounding matter.
#: ⛔ The near-forward number needs per-window pred/target pairs (`box_quality.match_pairs`),
#: i.e. `--eval-window-dump`, which the P arms' base config does NOT carry. Until that is
#: resolved (see `P_PANEL_DUMP_GAP.md`) this reporter's headline is the SECONDARY metric, and
#: :func:`render` says so on every line it prints.
HEAD = "eval_box3d_centre"
HEAD_POPULATION = "all_360 (SECONDARY) - NOT the near-forward primary the verdicts use"

#: ⛔ Per family, NEVER pooled. Order is the doctrine's order; `perception` is this panel's own.
FAMILIES: dict[str, tuple[str, ...]] = {
    "longitudinal": ("eval_lon", "eval_lon_tac", "eval_tacv6_lon_ce"),
    "lateral": ("eval_lat", "eval_lat_tac", "eval_tacv6_lat_ce", "eval_agent_yaw"),
    "tactical": ("eval_tac_v6", "eval_goal_tac", "eval_tacv6_goal_bce",
                 "eval_tacv6_goal_conf_bce"),
    "strategic": ("eval_route", "eval_nav_injected"),
    "perception": ("eval_box3d_centre", "eval_box3d_size", "eval_box3d_yaw",
                   "eval_box3d_presence", "eval_box3d_cls", "eval_box3d_z", "eval_box3d_h"),
}


#: ⛔ Unparseable metrics lines, per file, so a damaged artifact is VISIBLE rather than
#: silently skipped. A bare `except: continue` cannot tell "different kind of row" from
#: "this file is truncated", and a trainer killed mid-write leaves exactly that.
_UNPARSEABLE: dict[str, int] = {}


def final_eval_row(run_dir: Path) -> dict | None:
    """The LAST metrics row carrying the headline. ⛔ Not the last line — train rows have none."""
    met = Path(run_dir) / "metrics.jsonl"
    if not met.exists():
        return None
    best = None
    with open(met, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:                                # noqa: BLE001 — COUNTED
                _UNPARSEABLE[str(met)] = _UNPARSEABLE.get(str(met), 0) + 1
                continue
            if HEAD in d:
                best = d
    return best


def _seed_of(run_dir: Path):
    cfg = Path(run_dir) / "config.json"
    if not cfg.exists():
        return None
    try:
        argv = list(json.loads(cfg.read_text(encoding="utf-8")).get("argv") or [])
    except Exception:                                        # noqa: BLE001
        return None
    for i, t in enumerate(argv):
        if t == "--seed":
            return argv[i + 1] if i + 1 < len(argv) else None
    return None


def delta(a: dict, b: dict, keys) -> dict:
    """``|a − b|`` per key, only where BOTH sides carry a finite value.

    A key present on one side only is reported as ``None`` with a reason, never silently
    dropped — a family that quietly loses a metric is the omission the doctrine forbids.
    """
    out = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            out[k] = {"abs_delta": None, "reason": "absent in %s" % (
                "both" if va is None and vb is None else ("A" if va is None else "B"))}
            continue
        try:
            fa, fb = float(va), float(vb)
        except (TypeError, ValueError):
            out[k] = {"abs_delta": None, "reason": "non-numeric"}
            continue
        if not (math.isfinite(fa) and math.isfinite(fb)):
            out[k] = {"abs_delta": None, "reason": "non-finite"}
            continue
        out[k] = {"abs_delta": abs(fa - fb), "a": fa, "b": fb}
    return out


def families(a: dict, b: dict) -> dict:
    """The five families, each on its own. ⛔ Never summed — see the module docstring."""
    return {fam: delta(a, b, keys) for fam, keys in FAMILIES.items()}


def report(p0: dict | None, p0b: dict | None, a8: dict | None = None,
           seeds: dict | None = None) -> dict:
    """The whole read, in the committed order. ⇒ floor FIRST, upper bound BESIDE it, labelled.

    ⛔ Refuses (``status`` ``INCONCLUSIVE``) when the second replicate is missing: a floor from
    one replicate is not a floor. The upper bound alone is NEVER promoted to fill the gap.
    """
    if p0 is None or p0b is None:
        return {"status": "INCONCLUSIVE",
                "reason": ("the SEED FLOOR needs BOTH replicates; "
                           "missing %s. |P0 - A8| is an upper bound and may NOT stand in for it"
                           % ("P0" if p0 is None else "P0b")),
                "seed_floor": None, "upper_bound_not_the_floor": None}
    floor = families(p0, p0b)
    out = {
        "status": "OK",
        # ⭐ FIRST KEY, DELIBERATELY: the floor is read before anything is judged against it.
        "seed_floor": {
            "what": "|P0 - P0b|, same pinned tree, seeds differ, nothing else moved",
            "admissible_as": "THE SEED FLOOR",
            "headline_key": HEAD,
            "headline_population": HEAD_POPULATION,
            "headline_abs_delta": floor["perception"][HEAD]["abs_delta"],
            "by_family": floor,
            "seeds": (seeds or {}),
        },
        "upper_bound_not_the_floor": None,
        "code_delta_gap": None,
    }
    if a8 is not None:
        ub = families(p0, a8)
        out["upper_bound_not_the_floor"] = {
            "what": "|P0 - A8|, DIFFERENT and unrecorded code (A8 stamps no commit/hash/md5)",
            # ⛔ the label is load-bearing, not decoration: a mutation dropping it goes RED
            "admissible_as": "UPPER BOUND ONLY - never quote this as the floor",
            "headline_key": HEAD,
            "headline_abs_delta": ub["perception"][HEAD]["abs_delta"],
            "by_family": ub,
        }
        f, u = (out["seed_floor"]["headline_abs_delta"],
                out["upper_bound_not_the_floor"]["headline_abs_delta"])
        if f is not None and u is not None:
            out["code_delta_gap"] = {
                "abs": abs(u - f),
                "what": "|upper bound - floor| on the headline = the SIZE OF THE CODE DELTA, "
                        "which is worth knowing on its own",
            }
    return out


def render(rep: dict) -> str:
    """Text in the committed order. ⛔ Raises if asked to render the upper bound as a floor."""
    if rep.get("status") != "OK":
        return "STATUS %s\n  %s" % (rep.get("status"), rep.get("reason"))
    sf = rep["seed_floor"]
    if sf.get("admissible_as") != "THE SEED FLOOR":
        raise ValueError("the seed-floor block is not labelled as the floor")
    ub = rep.get("upper_bound_not_the_floor")
    if ub is not None and "UPPER BOUND ONLY" not in str(ub.get("admissible_as")):
        raise ValueError("the |P0 - A8| block MUST carry its upper-bound label; refusing to "
                         "render a report that could be read as two floors")
    lines = ["SEED FLOOR (quote this): |P0 - P0b| %s = %s"
             % (HEAD, _fmt(sf["headline_abs_delta"])),
             "  population: %s" % HEAD_POPULATION,
             "  seeds: %s" % (sf.get("seeds") or {})]
    for fam, d in sf["by_family"].items():
        got = {k: _fmt(v["abs_delta"]) for k, v in d.items() if v["abs_delta"] is not None}
        miss = [k for k, v in d.items() if v["abs_delta"] is None]
        lines.append("  %-13s %s%s" % (fam, got, ("  [absent: %s]" % miss) if miss else ""))
    if ub is not None:
        lines += ["", "UPPER BOUND - NOT THE FLOOR: |P0 - A8| %s = %s"
                  % (HEAD, _fmt(ub["headline_abs_delta"])),
                  "  (A8's code is unrecorded, so this is seed + an unquantified code delta)"]
        g = rep.get("code_delta_gap")
        if g:
            lines.append("  code-delta gap |upper - floor| = %s" % _fmt(g["abs"]))
    return "\n".join(lines)


def _fmt(v):
    return "n/a" if v is None else ("%.6g" % v)


# ⛔ A CHECKER MUST NOT DIE ON ITS OWN OUTPUT. MEASURED 2026-09-20: three separate
# readouts crashed with a cp1252 `UnicodeEncodeError` on this box mid-print -- one of
# them after reporting "lines lost = 1" but BEFORE naming the line, i.e. it had verified
# nothing while looking like it had. Relying on the caller to export PYTHONIOENCODING is
# a habit; this is a guard. `errors="replace"` means the print degrades instead of
# raising even if the stream cannot take utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 -- a stream that cannot be reconfigured is not fatal
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", required=True, help="the P0-REPLICATE arm dir (containing run/)")
    ap.add_argument("--p0b", required=True, help="the P0B-REPLICATE arm dir")
    ap.add_argument("--a8", default=None, help="the BASE run dir, for the labelled upper bound")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    def _row(d):
        if d is None:
            return None
        p = Path(d)
        return final_eval_row(p / "run" if (p / "run").exists() else p)
    p0, p0b, a8 = _row(a.p0), _row(a.p0b), _row(a.a8)
    seeds = {k: _seed_of(Path(v) / "run" if (Path(v) / "run").exists() else Path(v))
             for k, v in (("P0", a.p0), ("P0b", a.p0b)) if v}
    rep = report(p0, p0b, a8, seeds=seeds)
    if _UNPARSEABLE:
        rep["unparseable_metrics_lines"] = dict(_UNPARSEABLE)   # ⛔ surfaced, never swallowed
    print(render(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print("\nZZPF-WROTE %s ZZ" % a.out, flush=True)
    return 0 if rep.get("status") == "OK" else 3


if __name__ == "__main__":
    raise SystemExit(main())
