"""Stage the S-T launch closure for shipping to Thor.

CRLF -> LF normalisation is applied because the dev-box working tree is mixed
(41 of 76 files are CRLF) and Thor is Linux; the md5 recorded here is of the
NORMALISED bytes, i.e. exactly the bytes that will land on Thor.
"""
import hashlib
import json
import os
import sys
import tarfile

REPO = os.path.abspath(sys.argv[1])
LIST = sys.argv[2]
STAGE = os.path.abspath(sys.argv[3])
TARBALL = os.path.abspath(sys.argv[4])

paths = [ln.strip() for ln in open(LIST, encoding="utf-8") if ln.strip()]
md5s = {}
for rel in paths:
    src = os.path.join(REPO, rel)
    data = open(src, "rb").read().replace(b"\r\n", b"\n")
    dst = os.path.join(STAGE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(data)
    md5s[rel] = hashlib.md5(data).hexdigest()

with tarfile.open(TARBALL, "w:gz") as tf:
    for rel in paths:
        ti = tf.gettarinfo(os.path.join(STAGE, rel), arcname=rel)
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = "root"
        ti.mode = 0o644
        with open(os.path.join(STAGE, rel), "rb") as fh:
            tf.addfile(ti, fh)

open(os.path.join(STAGE, "MD5S.json"), "w", encoding="utf-8").write(
    json.dumps(md5s, indent=1, sort_keys=True))
print("ZZJSONZZ" + json.dumps({
    "n": len(paths), "tar_bytes": os.path.getsize(TARBALL),
    "tar_md5": hashlib.md5(open(TARBALL, "rb").read()).hexdigest()}) + "ZZENDZZ")
