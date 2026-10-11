"""Daily Claude Code token meter (token-budget fix P8, 2026-10-11).

Reads every Claude Code transcript under ~/.claude/projects (main sessions and subagents), takes the
`usage` record of each model call once, and reports weighted spend per day, per model and per transcript
file, plus the two measured burn mechanisms:
  * cache writes after an idle gap of >= 5 min  (subagent prompt caches expire after 5 min), and
  * cache reads on context above 200k tokens.

Weights are API price RATIOS (input 1, cache write 1.25, cache read 0.1, output 5; Sonnet 0.6x and
Haiku 0.2x of Opus). The subscription meter's absolute scale is not public: compare days and shares,
not absolute units.

D: is a `subst` of E:, so the same session appears under two project folders; calls are de-duplicated
on (timestamp, model, token counts).

Usage:  python tools/token_meter/token_meter.py [--days 2] [--top 10]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os

W_IN, W_CW, W_CR, W_OUT = 1.0, 1.25, 0.1, 5.0
IDLE_S = 300
BIG_CTX = 200_000


def model_scale(model: str) -> float:
    m = (model or "").lower()
    return 0.2 if "haiku" in m else 0.6 if "sonnet" in m else 1.0


def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def scan(root: str, since: str) -> list[dict]:
    calls, seen = [], set()
    for path in glob.glob(os.path.join(root, "*", "**", "*.jsonl"), recursive=True):
        if dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d") < since:
            continue
        per_msg = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"usage"' not in line:
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                msg = o.get("message") or {}
                u = msg.get("usage")
                if isinstance(u, dict) and msg.get("model") != "<synthetic>":
                    per_msg[msg.get("id") or o.get("uuid")] = (o.get("timestamp", ""), msg.get("model", ""), u)
        for t, model, u in per_msg.values():
            if not t or t[:10] < since:
                continue
            c = dict(inp=u.get("input_tokens") or 0, cw=u.get("cache_creation_input_tokens") or 0,
                     cr=u.get("cache_read_input_tokens") or 0, out=u.get("output_tokens") or 0)
            key = (t, model, c["inp"], c["cw"], c["cr"], c["out"])
            if key in seen:
                continue
            seen.add(key)
            s = model_scale(model)
            c.update(ts=t, model=model, path=path, sub="subagents" in path.replace("\\", "/"),
                     w=s * (W_IN * c["inp"] + W_CW * c["cw"] + W_CR * c["cr"] + W_OUT * c["out"]), s=s)
            calls.append(c)
    return calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), ".claude", "projects"))
    a = ap.parse_args()
    since = (dt.date.today() - dt.timedelta(days=a.days - 1)).isoformat()
    calls = scan(a.root, since)
    if not calls:
        print(f"no calls since {since}")
        return
    tot = sum(c["w"] for c in calls)
    by_file = collections.defaultdict(list)
    for c in calls:
        by_file[c["path"]].append(c)
    idle = 0.0
    for cs in by_file.values():
        cs.sort(key=lambda c: c["ts"])
        for prev, cur in zip(cs, cs[1:]):
            if (ts(cur["ts"]) - ts(prev["ts"])).total_seconds() >= IDLE_S:
                idle += cur["s"] * W_CW * cur["cw"]
    big = sum(c["s"] * W_CR * c["cr"] * max(0, c["inp"] + c["cw"] + c["cr"] - BIG_CTX)
              / max(1, c["inp"] + c["cw"] + c["cr"]) for c in calls)
    out = sum(c["s"] * W_OUT * c["out"] for c in calls)
    print(f"since {since}: {len(calls)} calls, weighted {tot / 1e6:.1f}M units")
    print(f"  idle-expiry cache writes {100 * idle / tot:4.1f}%   reads above {BIG_CTX // 1000}k context "
          f"{100 * big / tot:4.1f}%   output {100 * out / tot:4.1f}%   subagents "
          f"{100 * sum(c['w'] for c in calls if c['sub']) / tot:4.1f}%")
    days = collections.Counter()
    models = collections.Counter()
    for c in calls:
        days[c["ts"][:10]] += c["w"]
        models[c["model"]] += c["w"]
    print("per day:  " + "  ".join(f"{d} {w / 1e6:.1f}M" for d, w in sorted(days.items())))
    print("per model: " + "  ".join(f"{m} {100 * w / tot:.0f}%" for m, w in models.most_common()))
    print(f"top {a.top} transcripts (weighted, calls, mean context):")
    for p, cs in sorted(by_file.items(), key=lambda kv: -sum(c["w"] for c in kv[1]))[: a.top]:
        w = sum(c["w"] for c in cs)
        ctx = sum(c["inp"] + c["cw"] + c["cr"] for c in cs) / len(cs)
        kind = "SUB " if cs[0]["sub"] else "MAIN"
        print(f"  {w / 1e6:7.1f}M {100 * w / tot:5.1f}% {len(cs):5d} calls {ctx / 1e3:5.0f}k  {kind} "
              f"{os.path.basename(p)[:40]}")


if __name__ == "__main__":
    main()
