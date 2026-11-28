"""Copy the LF working copies into the tree, restoring each file's tip EOL convention (CRLF files stay CRLF, LF and NEW
files LF). Verifies CR count == LF count for CRLF files and CR == 0 for LF files after the write."""
from pathlib import Path

LF = Path("F:/refcv9_code2/lf")
TREE = Path("F:/refcv9_code2/tree/stack")
PRISTINE = Path("F:/refcv9_code2/pristine/stack")

for f in sorted(p for p in LF.rglob("*") if p.is_file() and p.suffix == ".py"):
    rel = f.relative_to(LF)
    pri = PRISTINE / rel
    crlf = pri.is_file() and pri.read_bytes().count(b"\r\n") > 0
    b = f.read_bytes()
    assert b"\r" not in b, rel
    out = b.replace(b"\n", b"\r\n") if crlf else b
    (TREE / rel).write_bytes(out)
    nb = (TREE / rel).read_bytes()
    cr, lf = nb.count(b"\r"), nb.count(b"\n")
    assert (cr == lf) if crlf else (cr == 0), rel
    print(f"{rel}  {'CRLF' if crlf else 'LF'}{'' if pri.is_file() else ' (NEW)'}  lines={lf}")
