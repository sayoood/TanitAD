"""The corrected test's own mutations that touch the edited code, run in isolation: each must go RED.  Usage: mutcheck.py <refe_dir>"""
import os, sys, tempfile
from pathlib import Path
HERE = Path(sys.argv[1]); os.chdir(HERE); sys.path.insert(0, str(HERE))
import test_rls1_queue as T
text = T.CHAIN.read_text(encoding="utf-8")
sw = lambda a, b: text.replace(a, b) if text.count(a) == 1 else (_ for _ in ()).throw(AssertionError(a))  # noqa: E731
cases = [
    ("handshake off (s1_slow_start)", None, {"s1_slow_start"}),
    ("S1 is not ahead", sw('  if lock_held "$S1"; then echo "the S1 chain holds $S1/chain.lock"; return 0; fi\n', ""), {"s1"}),
    ("chain.lock held from the start, never given back",
     sw("local t0=$SECONDS held=0", 'exec 200>"$R/chain.lock"; flock -n 200; local t0=$SECONDS held=1').replace('if [ "$held" = 1 ]; then exec 200>&-; held=0;', 'if false; then exec 200>&-; held=0;'),
     {"s1", "both"}),
]
red = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, mut, only in cases:
        if mut is None:
            T.SANDBOX["handshake"] = False
        try:
            T.queue_scenarios(mut or text, Path(tmp), only)
            print(f"GREEN (BAD): {name}")
        except AssertionError as e:
            red += 1
            print(f"RED: {name}: {str(e)[:120]!r}")
        finally:
            T.SANDBOX["handshake"] = True
    # and the unmutated text on the same scenarios must be GREEN
    try:
        T.queue_scenarios(text, Path(tmp), {"s1", "s1_slow_start", "both"}); print("unmutated s1+s1_slow_start+both: GREEN")
    except AssertionError as e:
        print(f"unmutated FAILED: {str(e)[:200]!r}")
print(f"ZZMUTCHECK red={red}/{len(cases)}")
