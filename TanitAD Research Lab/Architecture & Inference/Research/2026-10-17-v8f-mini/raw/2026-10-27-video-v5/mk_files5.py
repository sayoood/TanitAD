"""mk_files5.py -- FILES.txt: path, bytes, git blob sha1 of the raw bytes, for every file of video5 (FILES.txt itself and the temporary files excluded).  Nothing is staged, committed or pushed."""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
rows = ["# path<TAB>bytes<TAB>git blob sha1 of the raw bytes (FILES.txt itself excluded; nothing is staged, committed or pushed)"]
for root, _, fs in os.walk(HERE):
    for f in sorted(fs):
        p = os.path.join(root, f).replace("\\", "/")
        if f in ("FILES.txt",) or f.endswith(".tmp.mp4") or "__pycache__" in p or f.startswith("_patch"):
            continue
        b = open(p, "rb").read()
        rows.append(f"{p}\t{len(b)}\t{hashlib.sha1(b'blob ' + str(len(b)).encode() + bytes([0]) + b).hexdigest()}")
open(f"{HERE}/FILES.txt", "w", encoding="utf-8").write("\n".join(rows) + "\n")
print(len(rows) - 1, "files listed")
