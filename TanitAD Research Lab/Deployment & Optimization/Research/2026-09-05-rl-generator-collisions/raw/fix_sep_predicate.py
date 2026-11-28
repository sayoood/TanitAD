"""Fix the `sep` predicate in `rl_refcv3_min.paired_delta` and pin it with a test.

THE DEFECT (MEASURED 2026-09-05, stack/scripts/rl_refcv3_min.py:598):

    "sep": bool((lo > 0) == (hi > 0))

`sep` is supposed to mean "the confidence interval EXCLUDES zero". This predicate instead asks
"do the two endpoints agree about being strictly positive", which is a different question, and
it is WRONG in two cases -- ASYMMETRICALLY:

    lo    hi     buggy   correct
    0.0   0.0    True    False    <- an exactly-zero interval read as separated
   -1.0   0.0    True    False    <- touches zero FROM BELOW, read as separated
    0.0   1.0    False   False    <- touches zero FROM ABOVE, correctly not separated

⛔ The asymmetry is the serious part: an interval that touches zero from below is called
separated while its mirror image is not. On this rig's safety metrics NEGATIVE means BETTER, so
the defect systematically favours reporting IMPROVEMENTS.

MEASURED blast radius on the banked arms: 36 of `ctrl0`'s 40 `sep=True` rows and 11 of
`ctrl_null`'s 35 are exact-zero rows that this predicate mislabels. The circulating figure
"a zero-information arm separates 35 of 57 metrics" should read 24 of 57. The claim's substance
is unaffected -- those 24 are real -- but the count is not.

Idempotent, content-asserted, and it verifies the truth table after writing.
"""
import io
import os
import sys
import time

REPO = sys.argv[1] if len(sys.argv) > 1 else r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PATH = os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py")

OLD = '''            "sep": bool((lo > 0) == (hi > 0)), "_estimator": "paired episode-cluster bootstrap"}'''
NEW = '''            #: ⛔ `sep` means THE INTERVAL EXCLUDES ZERO, and nothing else. It was written
            #: `bool((lo > 0) == (hi > 0))`, which asks whether the endpoints AGREE about being
            #: strictly positive -- a different question, wrong on two cases and ASYMMETRICALLY
            #: so: `[0, 0]` and `[-1, 0]` both read True while their mirror `[0, 1]` reads
            #: False. On this rig's safety metrics NEGATIVE means BETTER, so the defect
            #: systematically favoured reporting IMPROVEMENTS. MEASURED 2026-09-05: it
            #: mislabelled 36 of `ctrl0`'s 40 `sep=True` rows (a FROZEN-WEIGHT arm) and 11 of
            #: `ctrl_null`'s 35 -- the circulating "a zero-information arm separates 35 of 57"
            #: is 24 of 57. Pinned by `stack/tests/test_rl_paired_delta_sep.py`.
            "sep": bool(lo > 0 or hi < 0), "_estimator": "paired episode-cluster bootstrap"}'''

src = None
for _ in range(14):
    try:
        with io.open(PATH, encoding="utf-8") as fh:
            src = fh.read()
        break
    except OSError:
        time.sleep(4)
if src is None:
    raise SystemExit("INCONCLUSIVE: could not read %s (mount)" % PATH)

if '"sep": bool(lo > 0 or hi < 0)' in src:
    print("ALREADY APPLIED")
elif OLD not in src:
    raise SystemExit("REFUSED: the buggy predicate was not found verbatim -- not editing blind")
else:
    src = src.replace(OLD, NEW, 1)
    for _ in range(14):
        try:
            with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
                fh.write(src)
            break
        except OSError:
            time.sleep(4)
    print("APPLIED")

# ---- verify by extracting the function and exercising its truth table ----
with io.open(PATH, encoding="utf-8") as fh:
    back = fh.read()
assert '"sep": bool(lo > 0 or hi < 0)' in back, "VERIFY FAILED: fix absent after write"
# ⚠️ Scope this to the CODE, not the file: the comment written above deliberately QUOTES the
# buggy predicate, so a whole-file search matches my own text. That is the self-match trap the
# monitor rule warns about, in a verification costume -- it fired here on the first run.
assert '"sep": bool((lo > 0) == (hi > 0))' not in back,     "VERIFY FAILED: the buggy predicate is still LIVE (not merely quoted in a comment)"

import numpy as np  # noqa: E402

start = back.index("def paired_delta(")
end = back.index("\ndef ", start + 1)
ns = {"np": np}
exec(compile(back[start:end], PATH, "exec"), ns)
pd = ns["paired_delta"]


def run(vals):
    """vals: per-episode deltas. Builds the before/after dicts paired_delta expects."""
    before = {"per_window": [{"wi": i, "eid": i, "k": 0.0} for i in range(len(vals))]}
    after = {"per_window": [{"wi": i, "eid": i, "k": float(v)} for i, v in enumerate(vals)]}
    return pd(before, after, "k", reps=200, seed=0)


cases = [
    ("all exactly zero            ", [0.0] * 12, False),
    ("all clearly positive        ", [1.0] * 12, True),
    ("all clearly negative        ", [-1.0] * 12, True),
    ("straddling zero             ", [-1.0, 1.0] * 6, False),
]
print()
print("%-30s %-12s %-10s %-10s %s" % ("case", "delta", "sep", "expected", ""))
ok = True
for name, vals, want in cases:
    r = run(vals)
    good = r["sep"] == want
    ok = ok and good
    print("%-30s %-12.4f %-10s %-10s %s"
          % (name, r["delta"], r["sep"], want, "OK" if good else "<-- FAIL"))
print()
print("TRUTH_TABLE=%s" % ("PASS" if ok else "FAIL"))
if not ok:
    raise SystemExit(1)
