"""Make dscale10k_merge_lean.py from dscale10k_merge.py (sha256 89d40399...) by two asserted replacements that change
memory use only: (1) the deterministic npz is written straight to a temporary FILE, not a BytesIO (same zipfile calls,
same entry order, same fixed timestamp and attributes), then renamed; (2) merge_v9 drops its references to the input
row arrays once the union rows are built. Control b2 of card PREREG_V8F_LEVER3 (md5 of the union npz equal to the
recorded 9881b170...) decides whether the bytes are identical."""
import hashlib, sys

SRC = "F:/TanitAD-scratch/datascale/pod/deliver/study/dscale10k_merge.py"
DST = "F:/TanitAD-scratch/rollhist/lever3/build/dscale10k_merge_lean.py"
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("89d4039966cd"), "source sha"
CR, LF = chr(13), chr(10)
s = raw.decode("utf-8").replace(CR + LF, LF)

old1 = '''    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:'''
new1 = '''    tmp_path = str(path) + ".lean.tmp"
    with open(tmp_path, "wb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED) as zf:'''
old2 = '''            zf.writestr(zi, b.getvalue())
    _write_bytes(path, buf.getvalue())'''
new2 = '''            zf.writestr(zi, b.getbuffer())
            del b, a
    os.replace(tmp_path, path)'''
old3 = '''    for k in rc:
        out[k] = np.concatenate([rc[k][idx], rt[k]], axis=0)
'''
new3 = '''    for k in rc:
        out[k] = np.concatenate([rc[k][idx], rt[k]], axis=0)
    rc = {k: None for k in rc}; rt = None; Zc = Zt = None          # lean: free the input row arrays (memory only)
'''
for o, n in ((old1, new1), (old2, new2), (old3, new3)):
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n)
out = s.encode("utf-8")
assert b"\x08" not in out and CR.encode() not in out
open(DST, "wb").write(out)
print("wrote", DST, hashlib.sha256(out).hexdigest()[:16])
