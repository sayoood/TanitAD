#!/usr/bin/env python3
"""Tests for refe/rl_pool_available.py (RL-S1 priority 2).  Run: python refe/test_rl_pool_available.py  ->  ZZTEST_PASS <n>.

The claim to defend is "the same tool and seed, with an availability filter": so the property tests run the REAL `rl_partition.partition()` on a synthetic pool
before and after the filter.  Expected values are hand-derived or recomputed independently.  Each mutation edits the module's SOURCE (the snippet must occur
exactly once) and the suite must then report >= 1 failing check.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_pool_available.py")
sys.path.insert(0, HERE)
import rl_partition as RP  # noqa: E402

SZ = lambda k: 6 + 2 * (k % 5)                                                       # noqa: E731  log k has 6,8,10,12,14 scenes
LOGS = [f"L{k:02d}" for k in range(30)]
POOL = [(f"tok_{k:02d}_{j:02d}", LOGS[k]) for k in range(30) for j in range(SZ(k))]
HEADER = "token\tlog\tin_navtest\tserved"
SEED = 11
KW = dict(n_train=60, n_val=20, min_val_logs=5, seed=SEED)                           # cap 20 // 5 = 4 per val log


def load(src, name):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


def pool_text():
    return HEADER + "\n" + "".join(f"{t}\t{l}\t0\t{i % 3}\n" for i, (t, l) in enumerate(POOL))


def write_unavail(path, toks, extra_cols=True):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("token\tlog\trank\n" + "".join(f"{t}\t{l}\t0\n" for t, l in toks))


def suite(rl):
    out = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    def guarded(name, fn):
        try:
            fn()
        except Exception as e:                                                       # noqa: BLE001
            check(name, False, f"raised {type(e).__name__}: {str(e)[:120]}")

    with tempfile.TemporaryDirectory() as d:
        pool = os.path.join(d, "pool.tsv")
        open(pool, "w", encoding="utf-8", newline="\n").write(pool_text())
        base = RP.partition(POOL, **KW)

        def run(argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = rl.main(argv)
            return rc, buf.getvalue()

        def t_filter():
            gone = [POOL[3], POOL[40], POOL[41]]                                      # three scenes; two of them adjacent in one log
            ua = os.path.join(d, "u.tsv")
            write_unavail(ua, gone)
            rc, so = run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "o.tsv")])
            txt = open(os.path.join(d, "o.tsv"), encoding="utf-8", newline="").read()
            exp = HEADER + "\n" + "".join(f"{t}\t{l}\t0\t{i % 3}\n" for i, (t, l) in enumerate(POOL) if (t, l) not in gone)
            check("filter: exactly the listed tokens are dropped; header, columns and row order kept (hand-derived)", rc == 0 and txt == exp, f"rc {rc}")
            check("filter: only the token matters (not the log): the other scenes of the same log stay",
                  all(f"{t}\t" in txt for t, l in POOL if l == POOL[3][1] and (t, l) not in gone))
            side = json.load(open(os.path.join(d, "o.tsv.available.json")))
            check("sidecar: counts (rows in 330 -> 327)", side["rows_in"] == len(POOL) and side["rows_out"] == len(POOL) - 3 and side["removed"] == 3, str(side))
            rc2, _ = run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "o2.tsv.gz")])
            with gzip.open(os.path.join(d, "o2.tsv.gz"), "rt", encoding="utf-8", newline="") as f:
                check("filter: .gz output decompresses to the same text", rc2 == 0 and f.read() == exp)
            rc3, _ = run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "o3.tsv")])
            check("filter: deterministic (same inputs -> same bytes)", open(os.path.join(d, "o3.tsv"), "rb").read() == open(os.path.join(d, "o.tsv"), "rb").read())
        guarded("filter", t_filter)

        def t_empty():
            ua = os.path.join(d, "empty.tsv")
            write_unavail(ua, [])
            rc, so = run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "same.tsv")])
            check("an EMPTY unavailable list reproduces the pool byte for byte (the expected case: nothing to filter)",
                  rc == 0 and open(os.path.join(d, "same.tsv"), "rb").read() == open(pool, "rb").read())
        guarded("empty", t_empty)

        def t_refusals():
            ua = os.path.join(d, "u2.tsv")
            write_unavail(ua, [("not_in_pool", "L00")])
            rc, so = run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "r1.tsv")])
            check("an unavailable token that is NOT in the pool -> exit 4, nothing written", rc == 4 and not os.path.exists(os.path.join(d, "r1.tsv")), f"rc {rc}")
            rc, so = run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "r2.tsv"), "--allow-unknown"])
            check("--allow-unknown writes it and records the unknown token in the sidecar",
                  rc == 0 and json.load(open(os.path.join(d, "r2.tsv.available.json")))["unknown"] == ["not_in_pool"])
            bad = os.path.join(d, "bad.tsv")
            open(bad, "w", newline="\n").write("scene\tlog\ntok_00_00\tL00\n")           # column 0 holds a REAL pool token under the wrong header
            rc, so = run(["--pool", pool, "--unavailable", bad, "--out", os.path.join(d, "r3.tsv")])
            check("an unavailable file with no 'token' column -> exit 4", rc == 4)
            nolog = os.path.join(d, "nolog.tsv")
            open(nolog, "w", newline="\n").write("token\tother\na\tb\n")
            ua0 = os.path.join(d, "u0.tsv")
            write_unavail(ua0, [])
            rc, so = run(["--pool", nolog, "--unavailable", ua0, "--out", os.path.join(d, "r4.tsv")])
            check("a pool with no 'log' column -> exit 4", rc == 4)
        guarded("refusals", t_refusals)

        def t_partition_property():
            """THE CLAIM: 'the same tool and seed with an availability filter'."""
            used = {t for t, _ in base["train"]} | {t for t, _ in base["val"]}
            unused = [s for s in POOL if s[0] not in used]
            check("setup: the synthetic partition leaves scenes unused (so the property is testable)", len(unused) > 5, str(len(unused)))
            drop = [unused[0], unused[len(unused) // 2]]
            ua = os.path.join(d, "up.tsv")
            write_unavail(ua, drop)
            run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "pp.tsv")])
            lines = open(os.path.join(d, "pp.tsv"), encoding="utf-8", newline="").read().split("\n")
            pool2 = [(r.split("\t")[0], r.split("\t")[1]) for r in lines[1:] if r]
            r2 = RP.partition(pool2, **KW)
            check("filtering UNUSED scenes leaves RL-train and RL-val identical (same tool, same seed)", r2["train"] == base["train"] and r2["val"] == base["val"])
            tr = base["train"]
            victim = tr[len(tr) // 2]
            write_unavail(ua, [victim])
            run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "pp2.tsv")])
            lines = open(os.path.join(d, "pp2.tsv"), encoding="utf-8", newline="").read().split("\n")
            pool3 = [(r.split("\t")[0], r.split("\t")[1]) for r in lines[1:] if r]
            r3 = RP.partition(pool3, **KW)
            check("filtering a TRAIN scene removes exactly it from RL-train and keeps RL-val and the train size", victim not in r3["train"] and r3["val"] == base["val"] and len(r3["train"]) == len(tr),
                  f"{len(r3['train'])} vs {len(tr)}")
            check("... and the other train scenes of the OLD partition are all still there",
                  set(tr) - {victim} <= set(r3["train"]))
            v = base["val"][0]
            write_unavail(ua, [v])
            run(["--pool", pool, "--unavailable", ua, "--out", os.path.join(d, "pp3.tsv")])
            lines = open(os.path.join(d, "pp3.tsv"), encoding="utf-8", newline="").read().split("\n")
            r4 = RP.partition([(r.split("\t")[0], r.split("\t")[1]) for r in lines[1:] if r], **KW)
            check("filtering a VAL scene keeps RL-val at its size and the train/val logs disjoint (the tool's own verify)",
                  v not in r4["val"] and len(r4["val"]) == len(base["val"]) and not ({l for _, l in r4["val"]} & {l for _, l in r4["train"]}))
        guarded("partition_property", t_partition_property)
    return out


MUTATIONS = [
    ("drops_by_log_not_token", "filter", "        if tok in unavailable:", "        if row.rstrip(\"\\r\").split(\"\\t\")[1] in {POOL_LOG.get(t) for t in unavailable}:"),
    ("keeps_the_unavailable", "filter", "        if tok in unavailable:", "        if tok in unavailable and False:"),
    ("drops_everything_after", "filter", "            removed.append(tok)\n        else:\n            kept.append(row)", "            removed.append(tok)\n            break\n        else:\n            kept.append(row)"),
    ("header_dropped", "filter", "    kept, removed = [lines[0]], []", "    kept, removed = [], []"),
    ("rows_sorted", "filter", "    unknown = sorted(unavailable - set(removed))", "    kept = kept[:1] + sorted(kept[1:], reverse=True)\n    unknown = sorted(unavailable - set(removed))"),
    ("unknown_token_allowed", "refuse", "    if unknown and not allow_unknown:", "    if False:"),
    ("empty_list_rewrites", "filter", "    text = \"\\n\".join(kept)", "    text = \"\\n\".join(kept).replace(\"\\t0\\t\", \"\\t1\\t\", 1)"),
    ("no_token_column_accepted", "refuse", "        if \"token\" not in head:\n            raise ValueError(f\"{p}: no 'token' column in {head!r}\")\n        it = head.index(\"token\")", "        it = 0"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    base = suite(load(src, "rl_pool_available_under_test"))
    bad = [(n, d) for n, ok, d in base if not ok]
    for n, d in bad:
        print(f"  FAIL {n}  {d}")
    assert not bad, f"the unmutated module FAILS its own checks: {[n for n, _ in bad]}"
    print(f"unmutated: {len(base)} checks green")
    print(f"{'mutation':28s} {'class':8s} caught-by")
    for name, cls, old, new in MUTATIONS:
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1)"
        mod = load(src.replace(old, new), f"rl_pool_available_mut_{name}")
        mod.POOL_LOG = {t: l for t, l in POOL}
        failing = [n for n, ok, _ in suite(mod) if not ok]
        print(f"{name:28s} {cls:8s} {len(failing)}  {failing[0][:80] if failing else ''}")
        assert failing, f"mutation {name} stayed GREEN: the suite cannot see it"
    print(f"ZZTEST_PASS {1 + len(MUTATIONS) + 1}  (1 unmutated suite of {len(base)} checks + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
