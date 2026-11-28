"""make_read_smooth.py -- writes read_smoothreq10k.py from tabclose/read_tabclose10k.py (sha256 prefix below; blocks copied with counted edits) and the new code of read_smooth_frag.txt and
smooth_selftest_frag.txt. Usage: python make_read_smooth.py   (run in smoothreq/). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "tabclose", "read_tabclose10k.py"))
OUT = os.path.join(HERE, "read_smoothreq10k.py")
SRC_SHA = os.environ.get("SR_READ_SRC_SHA", "5fa729f4d5880254")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the parent reader is not the one of card ADD84 (%s)" % hashlib.sha256(raw).hexdigest()[:16])
if os.path.exists(OUT) and os.environ.get("SR_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set SR_OVERWRITE_OWN=1 to rewrite it)" % OUT)
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
for nm in ("ctl_rec", "log_input", "join_inputs", "ctl_r1s", "ctl_x1s", "ctl_m1s", "ctl_u2", "ctl_h1", "assemble", "read_e_json", "register_runs", "log_arrays", "sha_file"):
    B[nm] = block(nm)
t = block("run_setting")
t = edit(t, '    ctl["a2_" + st] = ctl_u2(tables, elogs, [f[len("RRe_"):] for f in NEWS])\n', '    ctl["b3_" + st] = ctl_u2(tables, elogs, [f[len("RRe_"):] for f in NEWS])\n')
t = edit(t, '    if st == "AGE" and "RRe_STK" in NEWS and "RRe_TK" in NEWS:\n        ctl["a4_u3_" + st] = ctl_a4_u3(stats, st)\n', '    ctl["b4_" + st] = ctl_b4(stats, st)\n')
t = edit(t, '    if st == "AGE" and "RRe_L100" in NEWS:\n        ctl["a3_" + st] = ctl_a3(G["levels"][runs["RRe_L100"]], G["levels"][runs["RRc"]])\n',
         '    ctl["k4_" + st] = ctl_k4(G["levels"], runs, K4["json"], K4["reason"])\n')
t = edit(t, 'if f in NEWS or f in EEST or f == "RR":', 'if f in NEWS or f in EEST or f == "RR" or f == "RRe_TK":')
t = edit(t, "                if f in NEWS or f in EEST:\n", '                if f in NEWS or f in EEST or f == "RRe_TK":\n')
t = edit(t, 'for n_ in NEWS + EEST]', 'for n_ in NEWS + BANK84F + EEST]')
for bad in ("a2_", "a3_", "a4_u3", "ctl_a3"):
    assert bad not in t, bad
B["run_setting"] = t
frag = io.open(os.path.join(HERE, "read_smooth_frag.txt"), encoding="utf-8").read()
st_frag = io.open(os.path.join(HERE, "smooth_selftest_frag.txt"), encoding="utf-8").read().rstrip("\n")
for nm, body in list(B.items()) + [("SELFTEST", st_frag)]:
    tag = "#@@%s@@" % nm
    assert frag.count(tag) == 1, tag
    frag = frag.replace(tag, body)
assert "#@@" not in frag
for bad in ("TABCLOSE10K", "ADD84_x"):
    assert bad not in frag, bad
assert "\r" not in frag
io.open(OUT, "w", encoding="utf-8", newline="\n").write(frag)
print("wrote %s sha256 %s lines %d edits %d" % (os.path.basename(OUT), hashlib.sha256(frag.encode("utf-8")).hexdigest()[:16], frag.count("\n"), N_EDITS))
