"""hf_bank_restore_check.py <name> <dl_dir> -- MM 4a3d05, 2026-11-20: download a bank made by hf_bank_dir.py, stream its parts
in order through tarfile, and check every member's sha256 against the manifest. Proves that the bank restores. Prints RESTORE-PASS/FAIL."""
import hashlib
import json
import os
import re
import sys
import tarfile

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Sayood/tanitad-refcv9-e2-raw-runs"
name, dl = sys.argv[1], sys.argv[2]
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
mp = hf_hub_download(REPO, f"{name}/{name}.manifest.json", repo_type="dataset", token=tok, local_dir=dl)
man = json.load(open(mp, encoding="utf-8"))
paths = [hf_hub_download(REPO, p["path_in_repo"], repo_type="dataset", token=tok, local_dir=dl)
         for p in sorted(man["parts"], key=lambda r: r["part"])]


class Cat:
    def __init__(self, ps):
        self.ps, self.f = list(ps), None

    def read(self, n=-1):
        out = b""
        while n < 0 or len(out) < n:
            if self.f is None:
                if not self.ps:
                    break
                self.f = open(self.ps.pop(0), "rb")
            b = self.f.read(-1 if n < 0 else n - len(out))
            if not b:
                self.f.close()
                self.f = None
                continue
            out += b
        return out


want = {m["path"]: m["sha256"] for m in man["members"]}
seen, bad = 0, []
with tarfile.open(fileobj=Cat(paths), mode="r|") as tf:
    for m in tf:
        if not m.isfile():
            continue
        h = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
        seen += 1
        if want.get(m.name) != h:
            bad.append(m.name)
ok = seen == len(want) and not bad
print(f"{'RESTORE-PASS' if ok else 'RESTORE-FAIL'} {name}: {seen}/{len(want)} members, {len(bad)} sha256 mismatches, {len(paths)} parts")
sys.exit(0 if ok else 3)
