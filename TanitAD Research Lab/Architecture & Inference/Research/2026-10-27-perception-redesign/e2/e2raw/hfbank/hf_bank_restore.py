"""hf_bank_restore.py <name> <dest_parent> -- MM 4a3d05, 2026-11-21: RESTORE a bank made by hf_bank_dir.py onto disk.

* Streams the parts over HTTPS in numeric order straight into tarfile: NO temp part is written anywhere, so the disk needs only the
  restored bytes (manifest 'bytes').
* Every part's sha256 is checked against the manifest while it streams; every member's sha256 is checked as it is written.
* Writes only regular members whose path is in the manifest, relative and without '..', under <dest_parent>/<name-of-src-dir>/...
  It REFUSES if <dest_parent>/<top> already exists (it never overwrites or merges), and refuses if <dest_parent> lacks room.
* Ends with RESTORE-PASS (every manifest member written once, sha256-equal, every part sha256-equal) or RESTORE-FAIL. On FAIL the
  partial tree is left in place, named in the message; removing it is a separate step (it is never the only copy: the bank is).
The token is read in place from Keys.txt and never printed.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time

import requests
from huggingface_hub import hf_hub_url

REPO = "Sayood/tanitad-refcv9-e2-raw-runs"
name, dest = sys.argv[1], sys.argv[2].rstrip("/\\")
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
H = {"authorization": f"Bearer {tok}"}


def say(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg, flush=True)


r = requests.get(hf_hub_url(REPO, f"{name}/{name}.manifest.json", repo_type="dataset"), headers=H, timeout=120)
r.raise_for_status()
man = json.loads(r.content)
if not man.get("all_parts_verified"):
    say(f"RESTORE-FAIL {name}: the manifest does not say all parts verified")
    sys.exit(3)
want = {m["path"]: m["sha256"] for m in man["members"]}
tops = {p.split("/", 1)[0] for p in want}
if len(tops) != 1:
    say(f"RESTORE-FAIL {name}: members have {len(tops)} top folders, expected 1")
    sys.exit(3)
top = tops.pop()
for p in want:
    if p.startswith("/") or ":" in p or ".." in p.split("/"):
        say(f"RESTORE-FAIL {name}: unsafe member path {p!r}")
        sys.exit(3)
CHECK = len(sys.argv) > 3 and sys.argv[3] == "--check-only"   # stream and hash every member, write NOTHING (proves the bank restores)
root = os.path.join(dest, top)
if not CHECK and os.path.exists(root):
    say(f"RESTORE-FAIL {name}: {root} already exists (never overwritten or merged)")
    sys.exit(4)
free = shutil.disk_usage(dest).free
if not CHECK and free < man["bytes"] + 512 * 2**20:
    say(f"RESTORE-FAIL {name}: {dest} has {free // 2**20} MB free, needs {man['bytes'] // 2**20} MB + 512 MB")
    sys.exit(5)
parts = sorted(man["parts"], key=lambda q: q["part"])
say(f"START {name}: {len(want)} members, {man['bytes']} B, {len(parts)} parts -> {root}")


class PartStream:
    """Concatenates the parts' HTTPS bodies in order, checking each part's sha256 and byte count when it ends."""

    def __init__(self, ps):
        self.ps, self.it, self.h, self.n, self.cur, self.bad, self.seen = list(ps), None, None, 0, None, [], []
        self.buf, self.off = b"", 0

    def _finish_part(self):
        got = self.h.hexdigest()
        ok = got == self.cur["sha256"] and self.n == self.cur["bytes"]
        say(f"  part {self.cur['part']:03d} {'OK' if ok else 'MISMATCH'} {self.n} B sha256 {got[:16]}")
        if not ok:
            self.bad.append(self.cur["part"])
        self.seen.append(self.cur["part"])
        self.cur, self.it = None, None

    def _chunk(self):
        while True:
            if self.it is None:
                if not self.ps:
                    return None
                self.cur = self.ps.pop(0)
                resp = requests.get(hf_hub_url(REPO, self.cur["path_in_repo"], repo_type="dataset"), headers=H, stream=True, timeout=300)
                resp.raise_for_status()
                self.it, self.h, self.n = resp.iter_content(chunk_size=1 << 20), hashlib.sha256(), 0
            try:
                b = next(self.it)
            except StopIteration:
                self._finish_part()
                continue
            if b:
                self.h.update(b)
                self.n += len(b)
                return b

    def read(self, n=-1):
        out, need = [], n
        while n < 0 or need > 0:
            if self.off >= len(self.buf):
                b = self._chunk()
                if b is None:
                    break
                self.buf, self.off = b, 0
            take = len(self.buf) - self.off if n < 0 else min(need, len(self.buf) - self.off)
            out.append(self.buf[self.off:self.off + take])
            self.off += take
            if n >= 0:
                need -= take
        return b"".join(out)


ps = PartStream(parts)
done, bad, extra = set(), [], []
with tarfile.open(fileobj=ps, mode="r|") as tf:
    for m in tf:
        if not m.isfile():
            continue
        if m.name not in want or m.name in done:
            extra.append(m.name)
            continue
        h = hashlib.sha256()
        src = tf.extractfile(m)
        if CHECK:
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                h.update(b)
        else:
            out = os.path.join(dest, *m.name.split("/"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as fo:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    fo.write(b)
        if h.hexdigest() != want[m.name]:
            bad.append(m.name)
        done.add(m.name)
        if len(done) % 2000 == 0:
            say(f"  member {len(done)}/{len(want)}")
while ps.read(1 << 20):   # drain the tar padding so the last part's sha256 is checked
    pass
ok = len(done) == len(want) and not bad and not extra and not ps.bad and len(ps.seen) == len(parts)
say(f"{'RESTORE-PASS' if ok else 'RESTORE-FAIL'} {name}: {len(done)}/{len(want)} members "
    f"{'streamed and hashed (check only, nothing written)' if CHECK else 'written to ' + root}, {len(bad)} sha256 mismatches, "
    f"{len(extra)} unexpected members, parts mismatched {ps.bad or 'none'}")
sys.exit(0 if ok else 3)
