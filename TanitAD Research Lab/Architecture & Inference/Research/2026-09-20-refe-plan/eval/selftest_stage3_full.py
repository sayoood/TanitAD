#!/usr/bin/env python3
"""Fast, data-free tests of eval/stage3_full.py (synthetic tables; the real-table checks are `stage3_full.py --selfcheck`).

    python eval/selftest_stage3_full.py        # prints ZZSELFTEST_STAGE3 PASS n/n, exits 0 only if every test passes

Each test that guards a refusal also has its deliberate-regression twin: a check that cannot fail proves nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage3_full as S  # noqa: E402

NB = 300                      # resamples: speed, not precision (the estimator code path is the same)


def tables(n_logs=12, per=20, seed=0):
    rng = np.random.default_rng(seed)
    toks = [f"t{i:04d}" for i in range(n_logs * per)]
    tl = {t: f"log{i // per:02d}" for i, t in enumerate(toks)}
    def row(base):
        r = {k: float(np.clip(base + rng.normal(0, 8), 0, 100)) for k in S.SUBS}
        r["PDMS"] = float(np.clip(base + rng.normal(0, 8), 0, 100))
        return r
    a = {t: row(60) for t in toks}
    return toks, tl, a


def shifted(tbl, dpdms=0.0, dsub=None):
    out = {t: dict(v) for t, v in tbl.items()}
    for v in out.values():
        v["PDMS"] += dpdms
        for k, d in (dsub or {}).items():
            v[k] += d
    return out


def t_self_pair_is_zero():
    toks, tl, a = tables()
    r = S.read_pair(a, {t: dict(v) for t, v in a.items()}, toks, tl, n=NB)
    return S.check_selfpair(r)["ok"]


def t_shuffled_arm_fails_self_pair():                      # the regression arm
    toks, tl, a = tables()
    r = S.read_pair(S.shuffled(a), a, toks, tl, n=NB)
    return not S.check_selfpair(r)["ok"]


def noisy(tbl, d):
    """a table whose per-token difference to `tbl` VARIES (a constant shift makes every resample mean identical)"""
    return {t: dict(v, PDMS=v["PDMS"] + d + float(np.random.default_rng(i).normal(0, 10)),
                    EP=v["EP"] + 0.7 + float(np.random.default_rng(10_000 + i).normal(0, 5))) for i, (t, v) in enumerate(tbl.items())}


def t_swap_flips_sign():
    toks, tl, a = tables()
    b = noisy(a, 1.5)
    f, r = S.read_pair(b, a, toks, tl, n=NB), S.read_pair(a, b, toks, tl, n=NB)
    return (S.check_swap(f, r)["ok"] and f["primary"]["mean"] > 0 > r["primary"]["mean"]
            and f["primary"]["ci95"][0] != f["primary"]["ci95"][1])                 # the test has teeth: the interval is not a point


def t_swap_with_wrong_seed_fails():                        # the regression arm
    # ⚠ the per-token differences must VARY: with a constant shift every resample mean is identical, the interval is a
    # point, and no seed can move it -- the first version of this test used a constant shift and could not fail
    toks, tl, a = tables()
    b = {t: dict(v, PDMS=v["PDMS"] + 1.5 + float(np.random.default_rng(i).normal(0, 10))) for i, (t, v) in enumerate(a.items())}
    f, r = S.read_pair(b, a, toks, tl, n=NB), S.read_pair(a, b, toks, tl, n=NB, seed=1)
    ok_same_seed = S.check_swap(f, S.read_pair(a, b, toks, tl, n=NB))["ok"]
    return (not S.check_swap(f, r)["ok"]) and ok_same_seed


def t_decision_rows():
    d = S.decide
    return (d(0.1, 2.0, [], True)["outcome"] == "GO"
            and d(0.1, 2.0, ["EP"], True)["outcome"] == "NO-GO (adverse sub-score)"
            and "EP" in d(0.1, 2.0, ["EP"], True)["why"]
            and d(-0.1, 2.0, [], True)["outcome"] == "NO-GO (NOT PROVEN)"
            and d(-2.0, -0.1, [], True)["outcome"] == "NO-GO (P2 below u400)"
            and d(0.0, 2.0, [], True)["outcome"] == "NO-GO (NOT PROVEN)"       # lo > 0 is STRICT
            and d(5.0, 6.0, [], False)["outcome"] == "INVALID")                   # a failed gate is never a GO


def t_adverse_needs_whole_interval_below_zero():
    toks, tl, a = tables(n_logs=20)
    b = shifted(a, 0.0, {"TTC": -6.0})                      # a large, consistent loss on one sub-score
    r = S.read_pair(b, a, toks, tl, n=NB)
    c = S.read_pair(shifted(a, 0.0, {"C": 0.0}), a, toks, tl, n=NB)
    return r["adverse_subscores"] == ["TTC"] and c["adverse_subscores"] == []


def t_go_requires_all_three():
    toks, tl, a = tables(n_logs=20)
    better = shifted(a, 4.0)                                # PDMS up, every sub-score unchanged
    r = S.read_pair(better, a, toks, tl, n=NB)
    d = S.decide(r["primary"]["ci95"][0], r["primary"]["ci95"][1], r["adverse_subscores"], True)
    worse_sub = shifted(a, 4.0, {"DAC": -8.0})
    r2 = S.read_pair(worse_sub, a, toks, tl, n=NB)
    d2 = S.decide(r2["primary"]["ci95"][0], r2["primary"]["ci95"][1], r2["adverse_subscores"], True)
    return d["outcome"] == "GO" and d2["outcome"] == "NO-GO (adverse sub-score)" and d2["why"].endswith("DAC")


def t_token_gate():
    logs = {f"t{i}": f"l{i // 3}" for i in range(12)}
    pins = {"md5": None, "n": 12, "logs": 4}
    raw = ('{"tokens": [' + ",".join(f'"t{i}"' for i in range(12)) + ']}').encode()
    pins["md5"] = S.md5_bytes(raw)
    ok = S.check_tokens_bytes(raw, pins, logs)["ok"]
    cut = ('{"tokens": [' + ",".join(f'"t{i}"' for i in range(11)) + ']}').encode()
    return ok and not S.check_tokens_bytes(cut, pins, logs)["ok"]


def t_pins_are_the_registered_values():
    return (S.FULL_PINS["md5"].startswith("3a8b3653") and S.FULL_PINS["n"] == 12146 and S.FULL_PINS["logs"] == 136
            and S.C1_PINS["md5"].startswith("bc813be8") and S.N_BOOT == 10000 and S.SEED == 20260927
            and list(S.SUBS) == ["NC", "DAC", "EP", "TTC", "C", "DDC"])


def t_stage2_instrument_untouched():
    return S.md5_file(S.STAGE2_SFT) == S.STAGE2_SFT_MD5


def t_baselines_pinned_by_content():
    p, why = S.first_matching([Path(__file__)], "0" * 32)   # an existing file with the WRONG hash must be refused
    q, _ = S.first_matching([Path(__file__)], S.md5_file(Path(__file__)))
    return p is None and q is not None and "md5 differs" in why[0]


TESTS = [t_self_pair_is_zero, t_shuffled_arm_fails_self_pair, t_swap_flips_sign, t_swap_with_wrong_seed_fails, t_decision_rows,
         t_adverse_needs_whole_interval_below_zero, t_go_requires_all_three, t_token_gate, t_pins_are_the_registered_values,
         t_stage2_instrument_untouched, t_baselines_pinned_by_content]


def main() -> int:
    bad = []
    for t in TESTS:
        try:
            ok = bool(t())
        except Exception as e:  # a crashing test is a failing test
            ok = False
            print("  EXC", t.__name__, repr(e))
        print(("  ok  " if ok else "  FAIL"), t.__name__)
        if not ok:
            bad.append(t.__name__)
    print("ZZSELFTEST_STAGE3", "PASS" if not bad else "FAIL", f"{len(TESTS) - len(bad)}/{len(TESTS)}", *bad)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
