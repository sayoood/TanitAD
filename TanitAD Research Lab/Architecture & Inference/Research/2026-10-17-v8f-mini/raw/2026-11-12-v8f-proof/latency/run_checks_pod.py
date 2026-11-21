#!/usr/bin/env python3
"""run_checks_pod.py -- a plain-python runner (no pytest, no torch needed) of the self-checks of latency_proxy.py on a pure-python stub, of its JSON / MD writer and its gate,
and of a list of mutations of the timing harness, each of which must turn a check red.

  python run_checks_pod.py [--work-dir DIR]        (DIR is created; default: checks_work next to this file; nothing is deleted)
Last line: ZZ CHECKS-POD DONE <n_pass> <n_fail>.  Exit code 0 when n_fail is 0.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import latency_proxy as LP  # noqa: E402

SPEC_FIRST_LINE = "Latency proxy on the pod GPU (not Thor, not a product latency; the image encoder is not included). Tier: none (a cost measurement, not a driving metric)."
SPEC_FAMILY_LINE = "LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement)"

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# ---------------------------------------------------------------------------------------------------- mutations of the harness
class Patch:
    """Replace attributes of the module (or a class) for the duration of a `with`; always restored."""

    def __init__(self, *items):
        self.items = items
        self.saved = []

    def __enter__(self):
        for obj, name, new in self.items:
            self.saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, new)
        return self

    def __exit__(self, *exc):
        for obj, name, old in reversed(self.saved):
            setattr(obj, name, old)
        return False


_orig_time_fn = LP.time_fn
_orig_stats = LP.stats_ms
_orig_wrap = LP.Counter.wrap
_orig_reset = LP.Counter.reset
_orig_roll = LP.StubSubject._roll
_orig_rstates = LP.StubSubject.rollout_states


def mut_time_unit(fn, n, warmup, sync, clock=None):
    s, f = _orig_time_fn(fn, n, warmup, sync, clock)
    return [x * 1000.0 for x in s], f


def mut_time_offset(fn, n, warmup, sync, clock=None):
    s, f = _orig_time_fn(fn, n, warmup, sync, clock)
    return [x + 0.030 for x in s], f


def mut_time_fn_not_called(fn, n, warmup, sync, clock=None):
    return _orig_time_fn(lambda: None, n, warmup, sync, clock)


def mut_time_no_sync(fn, n, warmup, sync, clock=None):
    return _orig_time_fn(fn, n, warmup, lambda: None, clock)


def mut_time_first_only(fn, n, warmup, sync, clock=None):
    s, f = _orig_time_fn(fn, 1, warmup, sync, clock)       # one sample repeated: the spread is gone, and the stub 50 ms stays right -> must be caught by the sync count test, not s1
    return s * n, f


def mut_stats_p95_max(samples_s):
    d = _orig_stats(samples_s)
    d["p95"] = d["max"]
    return d


def mut_stats_p50_is_mean(samples_s):
    d = _orig_stats(samples_s)
    d["p50"] = d["mean"]
    return d


def mut_wrap_no_count(self, fn, rows_arg=0):
    return fn


def mut_reset_noop(self):
    return None


def mut_roll_fixed_horizon(self, z0, A):
    return _orig_roll(self, z0, [row[:10] for row in A])


def mut_roll_states_drop_last(self, K, H, mode):
    out = _orig_rstates(self, K, H, mode)
    return [row[:-1] for row in out] if mode == "batch" else out


def mut_roll_states_zero(self, K, H, mode):
    out = _orig_rstates(self, K, H, mode)
    return [[[0.0 for _ in s] for s in row] for row in out]


def sync_count_test():
    calls = [0]

    def sync():
        calls[0] += 1
    n, w = 7, 3
    LP.time_fn(lambda: None, n, w, sync)
    return {"id": "sync_count", "pass": calls[0] == 2 * n + w, "value": calls[0], "bar": f"{2 * n + w} (one sync per warm-up call, two per timed region)"}


def s1_check():
    return LP.check_s1(lambda: None)


def s2_check():
    return LP.check_s2(LP.StubSubject(), 4, 10)


def s3_check():
    return LP.check_s3(LP.StubSubject(), 4, 10)


def s4_check():
    return LP.check_s4(LP.StubSubject())


MUTATIONS = [
    ("M1 time unit off by 1000", [(LP, "time_fn", mut_time_unit)], s1_check),
    ("M2 timed function never called", [(LP, "time_fn", mut_time_fn_not_called)], s1_check),
    ("M3 no device synchronisation (asynchronous stub)", [(LP, "time_fn", mut_time_no_sync)], LP.check_s1a),
    ("M4 constant 30 ms added to every sample", [(LP, "time_fn", mut_time_offset)], s1_check),
    ("M5 p95 is the maximum", [(LP, "stats_ms", mut_stats_p95_max)], LP.check_s5),
    ("M6 p50 is the mean", [(LP, "stats_ms", mut_stats_p50_is_mean)], LP.check_s5),
    ("M7 predictor-call counter does not count", [(LP.Counter, "wrap", mut_wrap_no_count)], s4_check),
    ("M8 counter never reset (counts accumulate across ticks)", [(LP.Counter, "reset", mut_reset_noop)], s4_check),
    ("M9 rollout ignores the requested horizon (work does not scale)", [(LP.StubSubject, "_roll", mut_roll_fixed_horizon)], s3_check),
    ("M10 batched rollout drops its last step", [(LP.StubSubject, "rollout_states", mut_roll_states_drop_last)], s2_check),
    ("M11 both rollouts return zeros", [(LP.StubSubject, "rollout_states", mut_roll_states_zero)], s2_check),
    ("M12 one repeated sample instead of n (sync count)", [(LP, "time_fn", mut_time_first_only)], sync_count_test),
]


def run_mutations():
    for name, patches, probe in MUTATIONS:
        with Patch(*patches):
            res = probe()
        res = res if isinstance(res, dict) else {"pass": False}
        record(f"mutation killed: {name}", not res["pass"], f"check {res.get('id')} read {'RED' if not res['pass'] else 'GREEN (mutation survived)'}")
        again = probe()
        record(f"harness restored after: {name}", again["pass"], f"check {again.get('id')}")


# ---------------------------------------------------------------------------------------------------- the checks, the writer and the gate
def run_baseline():
    sub = LP.StubSubject()
    checks = LP.run_self_checks(sub, 4, 4, 10)
    for c in checks:
        record(f"baseline check {c['id']}", c["pass"], json.dumps({k: v for k, v in c.items() if k in ("value_p50_ms", "value_max_abs_diff", "value_ratio", "value")})[:160])
    record("sync discipline: one sync per warm-up call and two per timed region", sync_count_test()["pass"])
    st = LP.stats_ms([0.010])
    record("statistics of one sample", st["p50"] == st["p95"] == st["mean"] == 10.0 and st["n"] == 1)
    try:
        LP.stats_ms([])
        record("statistics of no samples refuse", False)
    except Exception:
        record("statistics of no samples refuse", True)
    c = LP.Counter()
    w = c.wrap(lambda x: x)
    w(1)
    w(2)
    record("counter counts two calls", c.n == 2)


def run_writer_and_gate(work):
    d1 = os.path.join(work, "stub_ok")
    rc = LP.main(["--stub", "--out-dir", d1, "--n", "8", "--warmup", "2", "--k-list", "1,4", "--extra-horizons", "20"])
    record("stub flow exits 0", rc == 0, f"rc {rc}")
    jp, mp = os.path.join(d1, "LATENCY_PROXY_STUB.json"), os.path.join(d1, "LATENCY_PROXY_STUB.md")
    record("stub flow writes the JSON and the MD", os.path.isfile(jp) and os.path.isfile(mp))
    if os.path.isfile(mp) and os.path.isfile(jp):
        with open(mp, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        with open(jp, encoding="utf-8") as fh:
            R = json.load(fh)
        record("MD first line is the required text", lines[0] == SPEC_FIRST_LINE)
        record("MD second line is the family line", lines[1] == SPEC_FAMILY_LINE)
        record("JSON carries both lines and the tier", R["first_line"] == SPEC_FIRST_LINE and R["family_line"] == SPEC_FAMILY_LINE and R["tier"].startswith("none"))
        record("JSON carries all six checks passed", len(R["checks"]) == 6 and all(c["pass"] for c in R["checks"]))
        v = R["variants"]["eager"]
        record("JSON eager variant has a, b, c, d sections", all(k in v for k in ("a_predictor_step_batch1", "b_rollout", "c_heads", "d_planning_tick")))
        e = v["d_planning_tick"]["stub-tick"]
        record("tick section carries counted predictor calls", e["predictor_calls_per_tick_values"] == [0, 10] and e["n_inputs_with_rollout"] == 2, json.dumps(e["predictor_calls_per_tick_counted"]))
        b = v["b_rollout"]["H10_K4"]
        record("seq total is slower than the batch for K = 4", b["seq_total"]["p50"] > b["batch"]["p50"], f"seq/batch {b['seq_over_batch_p50']:.2f}")
        record("fp16 and compile variants are marked NOT RUN with a text on the stub", R["variants"]["fp16"].get("status") == "NOT RUN" and bool(R["variants"]["fp16"].get("error")) and R["variants"]["compile"].get("status") == "NOT RUN")
        text = "\n".join(lines).lower()
        banned = ("vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "closed" + " loop")   # spelled in parts so this file holds none of them
        hits = [w for w in banned if re.search(chr(92) + "b" + re.escape(w) + chr(92) + "b", text)]
        record("MD holds none of the banned words", not hits, ",".join(hits))
    rc2 = LP.main(["--stub", "--out-dir", d1, "--n", "8", "--warmup", "2", "--k-list", "1,4"])
    record("a second write to the same folder is refused (rc 4)", rc2 == 4, f"rc {rc2}")
    d2 = os.path.join(work, "stub_gate")
    bad = lambda: {"id": "s5", "pass": False, "name": "forced failure"}        # noqa: E731
    with Patch((LP, "check_s5", bad)):
        rc3 = LP.main(["--stub", "--out-dir", d2, "--n", "8", "--warmup", "2", "--k-list", "1,4"])
    record("a failed self-check exits 2", rc3 == 2, f"rc {rc3}")
    record("a failed self-check writes no timing table", not os.path.exists(os.path.join(d2, "LATENCY_PROXY_STUB.json")) and not os.path.exists(os.path.join(d2, "LATENCY_PROXY_STUB.md")))
    record("a failed self-check writes SELF_CHECKS_FAILED.json", os.path.isfile(os.path.join(d2, "SELF_CHECKS_FAILED.json")))
    d3 = os.path.join(work, "stub_mut_first_line")
    with Patch((LP, "FIRST_LINE", "Latency proxy (changed text)")):
        LP.main(["--stub", "--out-dir", d3, "--n", "8", "--warmup", "2", "--k-list", "1,4"])
    with open(os.path.join(d3, "LATENCY_PROXY_STUB.md"), encoding="utf-8") as fh:
        first = fh.readline().rstrip("\n")
    record("mutation killed: changed first line is caught by the writer test", first != SPEC_FIRST_LINE, "the MD first line differs from the required text")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=os.path.join(HERE, "checks_work"))
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    work = os.path.join(a.work_dir, "run_" + stamp + "_" + str(os.getpid()))
    os.makedirs(work, exist_ok=False)
    print("work dir", work, flush=True)
    run_baseline()
    run_writer_and_gate(work)
    run_mutations()
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    n_mut = sum(1 for n, _, _ in RESULTS if n.startswith("mutation killed"))
    print(f"mutations listed: {len(MUTATIONS)} harness mutations + 1 writer mutation; killed {sum(1 for n, ok, _ in RESULTS if n.startswith('mutation killed') and ok)} of {n_mut}", flush=True)
    print(f"ZZ CHECKS-POD DONE {n_pass} {n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
