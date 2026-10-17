import sys, tempfile, time
from pathlib import Path
tree = sys.argv[1]
pk = Path(tree) / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, str(pk / "refe")); sys.path.insert(0, str(pk / "eval"))
import test_rls1_queue as T
text = T.CHAIN.read_text(encoding="utf-8")
kill_line = '  [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED ($(cat'
assert text.count(kill_line) == 1
for arm, key in (("R0d", "r0d_kill"), ("R1d", "r1d_kill")):
    mut = text.replace(kill_line, '  [ "$ARM" != %s ] && [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED ($(cat' % arm)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        try:
            T.rls1d_checks(mut, tmp, {key})
            print(arm, "MUTATION STAYED GREEN")
        except AssertionError as e:
            print(arm, "RED in", round(time.time() - t0, 1), "s; head:", str(e)[:230].replace("\n", " | "))
    with tempfile.TemporaryDirectory() as tmp:
        print(arm, "must_fail_with ->", T.must_fail_with(lambda: T.rls1d_checks(mut, tmp, {key}), "m", "KILL-STOP"))
# self-check of the helper: a RED with the wrong message must be refused
try:
    T.must_fail_with(lambda: (_ for _ in ()).throw(AssertionError("something else")), "demo", "KILL-STOP")
    print("helper: WRONG-REASON NOT DETECTED")
except AssertionError as e:
    print("helper refuses a wrong-reason RED:", str(e)[:90])
