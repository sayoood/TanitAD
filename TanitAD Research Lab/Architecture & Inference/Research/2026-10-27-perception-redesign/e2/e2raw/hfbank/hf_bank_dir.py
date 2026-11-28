"""hf_bank_dir.py <src_dir> <name> <out_dir> -- MM 4a3d05, 2026-11-20.

Banks one directory to the PI's PRIVATE HF dataset repo as a tar stream cut into 1,500 MB parts.
* Every member's sha256 is computed from the bytes tar reads (one read of the source).
* Each part is written to <out_dir>/parts, hashed, uploaded (HF_HUB_DISABLE_XET=1), and verified by the remote LFS sha256.
  Only a VERIFIED part's local temp copy is removed, so the local disk holds at most one part.
* The manifest (members, parts, remote revisions) is written to <out_dir>/<name>.manifest.json and uploaded beside the parts.
* It never deletes or changes anything in <src_dir>. Deleting the source is a separate, PI-approved step.
The token is read in place from Keys.txt and never printed.
"""
import hashlib
import json
import os
import re
import sys
import tarfile
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi  # noqa: E402

REPO = "Sayood/tanitad-refcv9-e2-raw-runs"
PART = int(os.environ.get("HFBANK_PART_MB", "1500")) * 1024 * 1024
src, name, out = sys.argv[1].rstrip("/\\"), sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, "parts"), exist_ok=True)
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok)
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(REPO, repo_type="dataset").private, "repo is not private: refusing to upload"
log = open(os.path.join(out, name + ".log"), "a", encoding="utf-8")


def ev(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


parts = []


def ship(path, idx, sha):
    pin = f"{name}/{name}.tar.part{idx:03d}"
    for attempt in range(1, 4):
        try:
            ci = api.upload_file(path_or_fileobj=path, path_in_repo=pin, repo_id=REPO, repo_type="dataset",
                                 commit_message=f"bank {name} part {idx:03d} (refcv9 E2 raw pull, only copy before this)")
            pi = api.get_paths_info(REPO, [pin], repo_type="dataset", revision=ci.oid)[0]
            got = pi.lfs.sha256 if pi.lfs else None
            if got == sha:
                rec = {"part": idx, "path_in_repo": pin, "bytes": os.path.getsize(path), "sha256": sha, "revision": ci.oid,
                       "remote_lfs_sha256": got, "verified": True}
                parts.append(rec)
                os.remove(path)          # this job's own temp part, removed only after the remote sha256 matched
                ev(f"PART {idx:03d} VERIFIED {rec['bytes']} B sha256 {sha[:16]} rev {ci.oid[:12]}; local temp part removed")
                return
            ev(f"PART {idx:03d} attempt {attempt}: remote sha256 {got} != local {sha}")
        except Exception as e:  # noqa: BLE001
            ev(f"PART {idx:03d} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(60)
    ev(f"ZZBANK-FAIL part {idx:03d} not verified after 3 attempts; the local part is KEPT at {path}")
    sys.exit(3)


class PartWriter:
    def __init__(self):
        self.idx, self.f, self.h, self.n = -1, None, None, 0
        self._open()

    def _open(self):
        self.idx += 1
        self.path = os.path.join(out, "parts", f"{name}.tar.part{self.idx:03d}")
        self.f, self.h, self.n = open(self.path, "wb"), hashlib.sha256(), 0

    def _close_and_ship(self):
        self.f.close()
        ship(self.path, self.idx, self.h.hexdigest())

    def write(self, b):
        b = memoryview(b)
        while len(b):
            k = min(len(b), PART - self.n)
            self.f.write(b[:k])
            self.h.update(b[:k])
            self.n += k
            b = b[k:]
            if self.n == PART:
                self._close_and_ship()
                self._open()
        return None

    def finish(self):
        if self.n:
            self._close_and_ship()
        else:
            self.f.close()
            os.remove(self.path)         # an empty temp part this job just created


class HashReader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b


base = os.path.dirname(src)
files, skipped = [], []
for root, dirs, fs in os.walk(src):
    dirs.sort()
    for fn in sorted(fs):
        p = os.path.join(root, fn)
        (files if os.path.isfile(p) and not os.path.islink(p) else skipped).append(p)
ev(f"START {name}: {len(files)} files under {src} (skipped non-regular: {len(skipped)}) -> {REPO}")
w = PartWriter()
members = []
total = 0
with tarfile.open(fileobj=w, mode="w|", format=tarfile.PAX_FORMAT) as tf:
    for i, p in enumerate(files):
        arc = os.path.relpath(p, base).replace("\\", "/")
        ti = tf.gettarinfo(p, arcname=arc)
        with open(p, "rb") as fh:
            hr = HashReader(fh)
            tf.addfile(ti, hr)
        members.append({"path": arc, "bytes": ti.size, "sha256": hr.h.hexdigest()})
        total += ti.size
        if i % 2000 == 0:
            ev(f"  member {i}/{len(files)} ({total // 2**20} MB)")
w.finish()
man = {"name": name, "repo": REPO, "src": src, "n_files": len(members), "bytes": total, "skipped_non_regular": skipped,
       "part_bytes": PART, "parts": parts, "members": members,
       "restore": f"cat {name}.tar.part* | tar xf -  (parts in numeric order); check every member against 'members'",
       "all_parts_verified": all(p["verified"] for p in parts) and len(parts) > 0}
mp = os.path.join(out, name + ".manifest.json")
json.dump(man, open(mp, "w", encoding="utf-8"), indent=0)
msha = hashlib.sha256(open(mp, "rb").read()).hexdigest()
ci = api.upload_file(path_or_fileobj=mp, path_in_repo=f"{name}/{name}.manifest.json", repo_id=REPO, repo_type="dataset",
                     commit_message=f"bank {name} manifest ({len(members)} members, {len(parts)} parts)")
pi = api.get_paths_info(REPO, [f"{name}/{name}.manifest.json"], repo_type="dataset", revision=ci.oid)[0]
mdata = open(mp, "rb").read()
if pi.lfs:
    rsha, mok = pi.lfs.sha256, pi.lfs.sha256 == msha
else:                                    # a small file is a plain git blob: compare its git blob id
    rsha = pi.blob_id
    mok = rsha == hashlib.sha1(b"blob %d\0" % len(mdata) + mdata).hexdigest()
ok = man["all_parts_verified"] and mok
ev(f"{'ZZBANK-VERIFIED' if ok else 'ZZBANK-FAIL'} {name}: {len(members)} members, {total} B, {len(parts)} parts all verified="
   f"{man['all_parts_verified']}; manifest sha256 {msha[:16]} (remote lfs {str(rsha)[:16]}, rev {ci.oid[:12]})")
sys.exit(0 if ok else 3)
