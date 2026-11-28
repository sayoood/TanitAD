"""verify_blobs.py <tree> <blobs.txt>: every landed file in <tree> must hash (git blob sha1) to the LANDING_READY
"new" blob. Exit 9 on the first mismatch; prints the count on success."""
import hashlib
import sys
from pathlib import Path

tree, lst = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for line in lst.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    blob, path = line.split(" ", 1)
    data = (tree / path).read_bytes()
    got = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
    if got != blob:
        print(f"MISMATCH {path}: {got} != {blob}")
        sys.exit(9)
    n += 1
print(f"verified {n} landed files by git blob in {tree}")
