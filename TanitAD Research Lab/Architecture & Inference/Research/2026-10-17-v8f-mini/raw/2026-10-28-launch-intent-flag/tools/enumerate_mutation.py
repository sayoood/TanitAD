"""Mutation proof of enumerate_cells_li.py: break ONE line at a time and require exit code 1 (the checks catch it). python enumerate_mutation.py <enumerate_cells_li.py> <out file>"""
import os
import subprocess
import sys
import tempfile

SRC, OUT = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
M = [
    ("E01_harms_without_control", '"HARMS": ctrl and not guards_held,', '"HARMS": not guards_held,'),
    ("E02_one_seed_is_at_least_one", '"PARTIAL / ONE SEED": ctrl and guards_held and rm_n == 1,', '"PARTIAL / ONE SEED": ctrl and guards_held and rm_n >= 1,'),
    ("E03_adoption_needs_only_one_seed", "adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)", "adopt_ok = any(keep[s] and nobase[s] for s in SEEDS)"),
    ("E04_fixed_adopted_skips_adoption", '"FIXED / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,', '"FIXED / ADOPTED": ctrl and guards_held and rm_both and rc_both,'),
    ("E05_refused_never", '"REFUSED": not ctrl,', '"REFUSED": False,'),
    ("E06_registered_count_changed", '"PARTIAL / NOT ADOPTED": 45,', '"PARTIAL / NOT ADOPTED": 44,'),
    ("E07_harms_not_sent_to_B", 'NOT_ADOPTED_SENDS_TO_B = {"HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED"}', 'NOT_ADOPTED_SENDS_TO_B = {"FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED"}'),
    ("E08_dose_constant_changed", "wa == 1.58926 and wb == 0.0357048", "wa == 1.58927 and wb == 0.0357048"),
    ("E09_k_share_exclusive", "return abs(share - target) <= tol + 1e-12", "return abs(share - target) < tol - 1e-12"),
    ("E10_bar_halved", "return 2.0 * max(F_ADE if row", "return 1.0 * max(F_ADE if row"),
    ("E11_bar_exclusive", "return hi < 0 and -point >= b", "return hi < 0 and -point > b"),
    ("E12_next_launch_inverted", "    if mlaunch_moved:\n        return \"LOOP-STATE", "    if not mlaunch_moved:\n        return \"LOOP-STATE"),
    ("E13_partial_adopted_needs_crawl", '"PARTIAL / ADOPTED": ctrl and guards_held and rm_both and not rc_both and adopt_ok,', '"PARTIAL / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,'),
    ("E14_launch_route_inverted", 'return "LAUNCH as written" if base_is_dagoacc else', 'return "LAUNCH as written" if not base_is_dagoacc else'),
]
res = []
base = subprocess.run([sys.executable, SRC], capture_output=True, text=True)
res.append(f"CONTROL (unmutated): rc={base.returncode}  <- must be 0")
bad = base.returncode != 0
for name, old, new in M:
    assert s.count(old) == 1, f"{name}: anchor found {s.count(old)} times: {old[:70]!r}"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.py")
        open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
        r = subprocess.run([sys.executable, p], capture_output=True, text=True)
    ok = r.returncode == 1
    bad |= not ok
    res.append(f"{name}: rc={r.returncode} {'KILLED' if ok else 'SURVIVED  <--'}")
res.append(f"SUMMARY: {sum(1 for l in res if l.endswith('KILLED'))}/{len(M)} mutants killed" + ("" if not bad else "  (FAILURE)"))
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(res) + "\n")
print("\n".join(res))
sys.exit(1 if bad else 0)
