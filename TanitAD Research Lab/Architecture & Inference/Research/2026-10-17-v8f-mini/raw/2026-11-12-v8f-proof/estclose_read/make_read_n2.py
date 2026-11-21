#!/usr/bin/env python
"""make_read_n2.py -- note N2 of card ADD79 and note N1 of card ADD80: the tool clause "rows of the first tick among the compared rows" of the estimate controls (e2 of ADD79, k2 of ADD80)
asked for tick 0 in the mid-look log, which holds the ticks 5, 15, 25 by construction. ONE counted text edit per reader: the first tick of the replan log is 0, the first tick of the
mid-look log is its smallest tick. Writes read_estclose10k_n2.py + podread_estclose10k_n2.sh (this folder) and ../constclose/read_constclose10k_n1.py + podread_constclose10k_n1.sh."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "constclose")


def sha(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def edit(src, dst, want_sha, old, new, n_old=1):
    raw = io.open(src, "rb").read()
    assert hashlib.sha256(raw).hexdigest().startswith(want_sha), "%s is not the reviewed file" % src
    s = raw.decode("utf-8")
    assert s.count(old) == n_old, (src, s.count(old))
    s = s.replace(old, new)
    io.open(dst, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote %s %s" % (os.path.basename(dst), sha(dst)[:16]))


FIRST = '(0 if tag == "rule" else (int(kk.min()) if len(kk) else 0))'

# card ADD79, control e2
edit(os.path.join(HERE, "read_estclose10k.py"), os.path.join(HERE, "read_estclose10k_n2.py"), "826bacc8d0e36be1",
     "                n_first = int(((kk == 0) & eo).sum())",
     "                n_first = int(((kk == " + FIRST + ") & eo).sum())      # note N2 of card ADD79: the first look of the mid-look log is its smallest tick")
s = io.open(os.path.join(HERE, "podread_estclose10k.sh"), "rb").read().decode("utf-8")
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("387e67e07ba93925")
n = s.count("read_estclose10k.py")
assert n >= 3, n
io.open(os.path.join(HERE, "podread_estclose10k_n2.sh"), "w", encoding="utf-8", newline="\n").write(s.replace("read_estclose10k.py", "read_estclose10k_n2.py"))
print("wrote podread_estclose10k_n2.sh %s (%d names replaced)" % (sha(os.path.join(HERE, "podread_estclose10k_n2.sh"))[:16], n))

# card ADD80, control k2
edit(os.path.join(CC, "read_constclose10k.py"), os.path.join(CC, "read_constclose10k_n1.py"), "0eed065b41f04777",
     '                n_first = int(((np.asarray(z["k"]) == 0) & eo).sum())',
     '                kk = np.asarray(z["k"]); n_first = int(((kk == ' + FIRST + ") & eo).sum())      # note N1 of card ADD80: the first look of the mid-look log is its smallest tick")
s = io.open(os.path.join(CC, "podread_constclose10k.sh"), "rb").read().decode("utf-8")
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("53f4f5376da3659b")
n = s.count("read_constclose10k.py")
assert n >= 3, n
io.open(os.path.join(CC, "podread_constclose10k_n1.sh"), "w", encoding="utf-8", newline="\n").write(s.replace("read_constclose10k.py", "read_constclose10k_n1.py"))
print("wrote podread_constclose10k_n1.sh %s (%d names replaced)" % (sha(os.path.join(CC, "podread_constclose10k_n1.sh"))[:16], n))
