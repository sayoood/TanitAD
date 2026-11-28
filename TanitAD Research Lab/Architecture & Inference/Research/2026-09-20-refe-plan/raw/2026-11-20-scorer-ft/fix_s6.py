import sys, pathlib
p = pathlib.Path(sys.argv[1])
b = p.read_bytes()
old = rb'P5-C|traj_label|TLT\.|tgt_t", "\n".join('
new = rb'P5-C|traj_label|TLT\.|tgt_t|rank|RK\.|SCORER-FT", "\n".join('
assert b.count(old) == 1, b.count(old)
b = b.replace(old, new)
old2 = rb'# P5-C (2026-11-12): the label-table hunks are named too;'
assert b.count(old2) == 1, b.count(old2)
b = b.replace(old2, rb'# SCORER-FT (2026-11-20): the ranking-term hunks are named too; P5-C (2026-11-12): the label-table hunks are named too;')
p.write_bytes(b)
print("ZZFIXS6_OK")
