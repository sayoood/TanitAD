"""mk_files3.py -- writes viz3/FILES.txt: path <TAB> bytes <TAB> git blob sha1 of the RAW bytes (sha1('blob <n>\\0' + bytes), no filters).  FILES.txt itself is excluded."""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
rows = []
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", "_qa_tmp")]
    for f in sorted(files):
        p = os.path.join(root, f).replace("\\", "/")
        if f in ("FILES.txt",) or f.endswith(".tmp.mp4") or f.endswith(".pyc"):
            continue
        b = open(p, "rb").read()
        rows.append((p, len(b), hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()))
rows.sort()
with open(f"{HERE}/FILES.txt", "w", newline="\n") as fh:
    fh.write("# path<TAB>bytes<TAB>git blob sha1 of the raw bytes (FILES.txt itself excluded; nothing is staged, committed or pushed)\n")
    for p, n, h in rows:
        fh.write(f"{p}\t{n}\t{h}\n")
print(len(rows), "files,", sum(r[1] for r in rows), "bytes")
