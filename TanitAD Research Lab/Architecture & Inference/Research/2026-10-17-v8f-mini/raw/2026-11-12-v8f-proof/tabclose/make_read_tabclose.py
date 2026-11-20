"""make_read_tabclose.py -- writes read_tabclose10k.py from shufclose/read_shufclose10k.py (sha256 prefix 8d5391d73cde4ca8; blocks copied with counted edits) and the new code of
tabclose_frag.txt (read_tabclose_frag.txt) and tabclose_selftest_frag.txt. Usage: python make_read_tabclose.py   (run in tabclose/). Standard library only."""
import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "shufclose", "read_shufclose10k.py"))
OUT = os.path.join(HERE, "read_tabclose10k.py")
SRC_SHA = "8d5391d73cde4ca8"
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the parent reader is not the one of card ADD83")
if os.path.exists(OUT) and os.environ.get("TB_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set TB_OVERWRITE_OWN=1 to rewrite it)" % OUT)
PL = raw.decode("utf-8").split("\n")
N_EDITS = 0


def block(name):
    i = next(k for k, x in enumerate(PL) if x.startswith("def %s(" % name))
    j = i + 1
    while j < len(PL) and not (PL[j].startswith("def ") or PL[j].startswith("# ====") or PL[j].startswith("class ")):
        j += 1
    body = PL[i:j]
    while body and body[-1].strip() == "":
        body.pop()
    return "\n".join(body)


def edit(text, a, b, n=1):
    global N_EDITS
    assert text.count(a) == n, (a, text.count(a), n)
    N_EDITS += 1
    return text.replace(a, b)


B = {}
for nm in ("log_input", "join_inputs", "ctl_x1s", "ctl_m1s", "read_e_json", "register_runs", "log_arrays", "sha_file"):
    B[nm] = block(nm)
B["ctl_r1s"] = edit(block("ctl_r1s"), 'shas.get((f[len("RRe_S"):], s))', 'shas.get((f[len("RRe_"):], s))')
t = block("ctl_u2")
t = edit(t, "def ctl_u2(tables, logs, tol=TOL_U2):", "def ctl_u2(tables, logs, arms, tol=TOL_U2):")
t = edit(t, "    for e in ESTS:\n", "    for e in arms:\n")
t = edit(t, 'logs.get(("RRe_S" + e, s))', 'logs.get(("RRe_" + e, s))')
B["ctl_u2"] = t
t = block("ctl_h1")
t = edit(t, 'cname("EST-VS-SHUF-N10", "AGE", "C", got)', 'cname("TOK-VS-RR", "AGE", "C", got)')
t = edit(t, 'cname("EST-VS-SHUF-N10", "AGE", "C", want)', 'cname("TOK-VS-RR", "AGE", "C", want)')
B["ctl_h1"] = t
t = block("run_setting")
t = edit(t, "    cfg = SET[st]\n", "    cfg = SET[st]\n    set_active(st)\n")
t = edit(t, 'ctl["r1_" + st] = ctl_r1s(texts, n_files, n_clips, cfg["P"] if st == "AGE" else 0, a.wrapper_sha, {(e, s): shas.get("surr_%s_s%d.npz" % (e, s)) for e in ESTS for s in SEEDS})',
         'ctl["r1_" + st] = ctl_r1s(texts, n_files, n_clips, cfg["P"] if st == "AGE" else 0, a.wrapper_sha, {(e, s): shas.get("tab_%s_s%d.npz" % (e, s)) for e in NEW_ARMS for s in SEEDS})')
t = edit(t, '    ctl["u2_" + st] = ctl_u2(tables, elogs)\n', '    ctl["a2_" + st] = ctl_u2(tables, elogs, [f[len("RRe_"):] for f in NEWS])\n')
t = edit(t, '    ctl["u3_" + st] = ctl_u3(stats, st)\n    extra["u3_gate"] = ctl["u3_" + st]["gate_u3"]\n',
         '    if st == "AGE" and "RRe_STK" in NEWS and "RRe_TK" in NEWS:\n        ctl["a4_u3_" + st] = ctl_a4_u3(stats, st)\n')
t = edit(t, '    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl)\n',
         '    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl)\n'
         '    if st == "AGE" and "RRe_L100" in NEWS:\n        ctl["a3_" + st] = ctl_a3(G["levels"][runs["RRe_L100"]], G["levels"][runs["RRc"]])\n')
t = edit(t, '    if S is None or cells_s is None or not all(v.get("PASS") is True for v in ctl.values()):\n',
         '    if S is None or cells_s is None or not all(v.get("PASS") is True for k_, v in ctl.items() if not k_.startswith(CARD_SPECIFIC)):\n')
t = edit(t, '"levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text)}',
         '"levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text),\n            "table": (arm_table(G["levels"], runs) if q else hw_text), "pair_forms": dict(PAIRS), "pair_part": dict(PAIR_PART)}')
assert "ESTS" not in t and "surr_" not in t and "u3" not in t.replace("a4_u3", ""), "a name of the parent is left in run_setting"
B["run_setting"] = t
frag = io.open(os.path.join(HERE, "read_tabclose_frag.txt"), encoding="utf-8").read()
st_frag = io.open(os.path.join(HERE, "tabclose_selftest_frag.txt"), encoding="utf-8").read().rstrip("\n")
for nm, body in list(B.items()) + [("SELFTEST", st_frag)]:
    tag = "#@@%s@@" % nm
    assert frag.count(tag) == 1, tag
    frag = frag.replace(tag, body)
assert "#@@" not in frag
for bad in ("SHUFCLOSE", "ADD83", "surr_", "RRe_S\" +"):
    assert bad not in frag.replace("PREREG_V8F_PROOF_ADD84", ""), bad
assert "\r" not in frag
io.open(OUT, "w", encoding="utf-8", newline="\n").write(frag)
print("wrote %s sha256 %s lines %d edits %d" % (os.path.basename(OUT), hashlib.sha256(frag.encode("utf-8")).hexdigest()[:16], frag.count("\n"), N_EDITS))
