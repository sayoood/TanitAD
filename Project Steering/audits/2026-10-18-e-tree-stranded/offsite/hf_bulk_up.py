"""Off-site copy of the 2026-10-18 E:-tree stranded bulk (C:/Users/Admin/estrand_bulk, 2,252 files) to the PRIVATE
dataset Sayood/tanitad-internal-data under stranded-bulk-2026-10-18/. Token read in place from Keys.txt (never printed).
Chunked commits with retries; final per-file verification (LFS sha256 or git blob sha1) -> HF_VERIFIED.json."""
import os, re, json, time, hashlib, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, CommitOperationAdd
B = "C:/Users/Admin/estrand_bulk"; RID = "Sayood/tanitad-internal-data"; PFX = "stranded-bulk-2026-10-18"
OUT = "C:/Users/Admin/estrand/hfup/"
BS = chr(92); PRE = BS + BS + "?" + BS
LP = lambda rel: PRE + (B + "/" + rel).replace("/", BS)
tok = re.search(r"hf_[A-Za-z0-9]{30,}", open("D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
api = HfApi(token=tok)
man = json.load(open(B + "/MANIFEST.sha256.json"))
rels = sorted(man) + ["MANIFEST.sha256.json"]
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def remote_set():
    return {f[len(PFX) + 1:] for f in api.list_repo_files(RID, repo_type="dataset") if f.startswith(PFX + "/")}
have = remote_set(); log("remote already has", len(have), "of", len(rels))
todo = [r for r in rels if r not in have]
chunks, cur, cb = [], [], 0
for r in todo:
    s = os.path.getsize(LP(r))
    if cur and (cb + s > 600e6 or len(cur) >= 400): chunks.append(cur); cur, cb = [], 0
    cur.append(r); cb += s
if cur: chunks.append(cur)
log("chunks", len(chunks), [len(c) for c in chunks])
for n, c in enumerate(chunks):
    for att in range(1, 6):
        try:
            ops = [CommitOperationAdd(path_in_repo=f"{PFX}/{r}", path_or_fileobj=LP(r)) for r in c]
            api.create_commit(RID, repo_type="dataset", operations=ops,
                              commit_message=f"stranded bulk 2026-10-18 chunk {n+1}/{len(chunks)} ({len(c)} files)")
            log(f"CHUNK {n} ok ({len(c)} files) attempt {att}"); break
        except Exception as e:
            log(f"CHUNK {n} attempt {att} FAILED: {type(e).__name__}: {str(e)[:300]}"); time.sleep(60 * att)
            have = remote_set(); c = [r for r in c if r not in have]
            if not c: log(f"CHUNK {n} present after failure"); break
    else:
        log("ZZHFUP_STOP chunk", n); sys.exit(3)
# verify every file by content
log("verifying")
bad, ok = [], 0
paths = [f"{PFX}/{r}" for r in rels]
info = {}
for i in range(0, len(paths), 200):
    for p in api.get_paths_info(RID, paths[i:i + 200], repo_type="dataset"):
        info[p.path] = p
for r in rels:
    p = info.get(f"{PFX}/{r}")
    b = open(LP(r), "rb").read()
    if p is None: bad.append((r, "absent")); continue
    if getattr(p, "lfs", None):
        good = p.lfs.sha256 == hashlib.sha256(b).hexdigest()
    else:
        good = p.blob_id == hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    if r != "MANIFEST.sha256.json" and good:
        good = man[r]["sha256"] == hashlib.sha256(b).hexdigest()
    ok += good
    if not good: bad.append((r, "mismatch"))
res = {"repo": RID, "prefix": PFX, "n_files": len(rels), "verified_ok": ok, "bad": bad[:50], "n_bad": len(bad),
       "bytes": sum(os.path.getsize(LP(r)) for r in rels), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
json.dump(res, open(OUT + "HF_VERIFIED.json", "w"), indent=1)
log("ZZHFUP_DONE" if not bad else "ZZHFUP_BAD", json.dumps({k: res[k] for k in ("n_files", "verified_ok", "n_bad")}))
