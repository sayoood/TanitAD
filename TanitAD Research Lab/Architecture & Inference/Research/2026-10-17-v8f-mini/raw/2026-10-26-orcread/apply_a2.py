"""Bring orc_read.py and its tests to PREREG_V8F_ORECOVER Amendment 2 (A2-2026-10-26, written before any run). Originals kept as *.pre_a2.
python apply_a2.py <amended PREREG file (the tip blob + Amendment 2)>"""
import io
import shutil
import sys

H = "C:/Users/Admin/orcread/"
AMENDED = sys.argv[1]
A2_NO_EFFECT = "no F1 row separated-better and ≥ 2F at both seeds, guards held"
assert A2_NO_EFFECT in io.open(AMENDED, encoding="utf-8").read().split("## Amendment 2")[1]


def rep(name, pairs):
    p = H + name
    shutil.copyfile(p, p + ".pre_a2")
    s = io.open(p, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:90])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="").write(s)


rep("orc_read.py", [
    ('PREREG_ID = "Project Steering/PREREG_V8F_ORECOVER.md @28e28c88 + Amendment 1 (A1-2026-10-26)"',
     'PREREG_ID = "Project Steering/PREREG_V8F_ORECOVER.md @28e28c88 + Amendment 1 (A1-2026-10-26) + Amendment 2 (A2-2026-10-26)"\n'
     '# Amendment 2 replaces the NO EFFECT condition (the as-written table had a hole); the as-written table above stays verbatim, this is the condition in force.\n'
     'A2_NO_EFFECT = "' + A2_NO_EFFECT + '"'),
    ('        "NO EFFECT": bool(not sep_both["lat"] and not sep_both["lon"] and guards_held),',
     '        "NO EFFECT": bool(not moved_both["lat"] and not moved_both["lon"] and guards_held),        # A2-2026-10-26 (as written: not sep_both)'),
    ('             "guards_held": guards_held, "head_moved_at_both_seeds": moved_both}',
     '             "guards_held": guards_held, "head_moved_at_both_seeds": moved_both,\n'
     '             "sub_bar_separated_better_at_both_seeds": {hd: bool(sep_both[hd] and not moved_both[hd]) for hd in ("lat", "lon")}}'),
])
s = io.open(H + "orc_read.py", encoding="utf-8").read()
i = s.index("def outcome_block(d):")
j = s.index("\n\n\n", i)
blk = s[i:j]
assert "cond, nxt = next((c, n) for nm, c, n in OUTCOME_TABLE if nm == out)" in blk
print("outcome_block:\n" + blk)
