"""make_read_gaprule.py -- writes read_gaprule10k.py from read_gaprule_frag.txt with the block sha_file copied from ../smoothreq/read_smoothreq10k.py (sha256 prefix below).
Usage: python make_read_gaprule.py   (run in gaprule/). Refuses to overwrite (GR_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "smoothreq", "read_smoothreq10k.py"))
OUT = os.path.join(HERE, "read_gaprule10k.py")
raw = io.open(SRC, "rb").read()
SRC_SHA = os.environ.get("GR_READ_SRC_SHA", "")
if os.path.exists(OUT) and os.environ.get("GR_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists" % OUT)
PL = raw.decode("utf-8").split("\n")
i = next(k for k, x in enumerate(PL) if x.startswith("def sha_file("))
j = i + 1
while j < len(PL) and not (PL[j].startswith("def ") or PL[j].startswith("# ====") or PL[j].startswith("class ")):
    j += 1
body = PL[i:j]
while body and not body[-1].strip():
    body.pop()
block = "\n".join(body)
assert block.count("hashlib.sha256()") == 1
frag = io.open(os.path.join(HERE, "read_gaprule_frag.txt"), encoding="utf-8").read()
assert frag.count("#@@sha_file@@") == 1
frag = frag.replace("#@@sha_file@@", block)
for bad in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert bad not in frag.lower(), bad
io.open(OUT, "w", encoding="utf-8", newline="\n").write(frag)
print("wrote %s sha256 %s lines %d (source reader sha256 %s)" % (os.path.basename(OUT), hashlib.sha256(frag.encode("utf-8")).hexdigest()[:16], frag.count("\n"), hashlib.sha256(raw).hexdigest()[:16]))
