import io
import os

D = "F:/TanitAD-scratch/v8proof/integ10k/"
src = io.open(D + "podread_agerule_decomp_n2.sh", encoding="utf-8", newline="").read()
assert "\r" not in src
reps = [
    ("podread_agerule_decomp.sh", "podread_agedose_decomp.sh", 1),
    ("pod-side read of part B of card PREREG_V8F_PROOF_ADD71 with its notes N1 and N2 (carried pose against own motion state on the banked runs I10R0KNlHb20 and\n# I10R0KNlP30Hb20 of card ADD70); CPU only, no rollout.",
     "pod-side read of card PREREG_V8F_PROOF_ADD72 (AGEDOSE10K) under its amendment 1 (carried pose against own motion state at 6 and 12 s on the banked runs I10R0KHb20,\n# I10R0KP30Hb20, I10R0KP60Hb20 and I10R0Hb20 of card ADD70, on the 284 windows of the P = 60 grid); CPU only, no rollout.", 1),
    ("ZZ PODREAD-AGERULE-DECOMP", "ZZ PODREAD-AGEDOSE-DECOMP", 4),
    ("read_agerule_decomp_n2", "read_agedose_decomp", 2),
    ("READ_AGERULE_DECOMP_N2.json", "READ_AGEDOSE_DECOMP.json", 3),
    ("read_agerule_decomp.log", "read_agedose_decomp.log", 2),
    ("for f in \"$P/root/integ10k/read_longroll10k.py\" \"$P/root/integ10k/read_longroll10k_n3.py\" \"$P/root/integ10k/read_longroll10k_n4.py\" \"$P/root/integ10k/read_agerule_decomp.py\" \"$LR/grid_p30.json\" \"$ADD70JSON\"; do",
     "for f in \"$P/root/integ10k/read_longroll10k.py\" \"$P/root/integ10k/read_longroll10k_n3.py\" \"$P/root/integ10k/read_longroll10k_n4.py\" \"$P/root/integ10k/read_agerule_decomp.py\" \"$P/root/integ10k/read_agedose_decomp.py\" \"$LR/grid_p30.json\" \"$LR/grid_p60.json\" \"$ADD70JSON\"; do", 1),
    ("for r in R0KNlHb20 R0KNlP30Hb20 R0Hb20; do", "for r in R0KHb20 R0KP30Hb20 R0KP60Hb20 R0Hb20; do", 1),
    ("sha256sum \"$P/root/integ10k/read_agerule_decomp.py\" \"$P/root/integ10k/read_longroll10k_n4.py\" \"$ADD70JSON\"",
     "sha256sum \"$P/root/integ10k/read_agedose_decomp.py\" \"$P/root/integ10k/read_agerule_decomp.py\" \"$P/root/integ10k/read_longroll10k_n4.py\" \"$ADD70JSON\"", 1),
    ("\"$PY\" read_agerule_decomp.py --pull", "\"$PY\" read_agedose_decomp.py --pull", 1),
]
out = src
for a, b, n in reps:
    c = out.count(a)
    assert c == n, (a[:60], c, n)
    out = out.replace(a, b)
io.open(D + "podread_agedose_decomp.sh", "w", encoding="utf-8", newline="\n").write(out)
print("written", len(out))
