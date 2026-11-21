"""compare_test_logs.py <landed log> <tip log>: the FAILED/ERROR test ids of two pytest -rfEs logs of the SAME test
list, set-compared. A regression is an id that fails on the landed tree and NOT on the tip. Also prints each log's
final summary line and its EXIT line, and REFUSES (exit 2) a log without them -- a truncated log is not a clean run."""
import re
import sys
from pathlib import Path

ID = re.compile(r"^(FAILED|ERROR) (\S+)")


def read(p):
    lines = Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
    ids = {m.group(2) for ln in lines if (m := ID.match(ln))}
    summ = [ln for ln in lines if re.search(r"\d+ passed", ln) and " in " in ln]
    ex = [ln for ln in lines if ln.startswith("EXIT ")]
    if not summ or not ex:
        print(f"{p}: NO summary line or NO EXIT line -- truncated or crashed")
        sys.exit(2)
    return ids, summ[-1].strip("= "), ex[-1]


a, a_sum, a_ex = read(sys.argv[1])
t, t_sum, t_ex = read(sys.argv[2])
print(f"landed: {a_sum} | {a_ex}")
print(f"tip   : {t_sum} | {t_ex}")
print(f"failing ids: landed {len(a)}, tip {len(t)}, common {len(a & t)}")
reg, fixed = sorted(a - t), sorted(t - a)
print(f"REGRESSIONS (fail on the landed tree only): {len(reg)}")
for x in reg:
    print("  ", x)
print(f"fixed by the landing (fail on the tip only): {len(fixed)}")
for x in fixed:
    print("  ", x)
sys.exit(1 if reg else 0)
