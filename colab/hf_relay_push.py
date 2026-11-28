"""Push files to the PRIVATE HF relay for Colab jobs, with the MANIFEST written LAST (a listed file is complete).

    C:\\Users\\Admin\\venvs\\tanitad\\Scripts\\python.exe colab\\hf_relay_push.py --prefix refe/snapshots \\
        --quota colab\\raw\\<date>-hf-relay\\hf_quota_check.json FILE [FILE ...]

Relay repo: Sayood/tanitad-colab-relay (DATASET, private; created on first use). PI, 2026-09-27: data moves pod/dev
box -> private HF -> Colab ("follow your recommendation regarding data transfer").
⛔ PRE-PUSH ARITHMETIC IS THE ONLY GUARD (memory: hf-quota-is-a-hard-ceiling): private storage above 1 TB is BILLED,
not refused. So this refuses unless --quota names a hf_quota_check.json (the DataFlyWheel instrument's output) that
is at most 24 h old AND private_used + this push stays under --ceiling-gb (default 1000). Deleting relay files later
frees nothing until a super-squash, so every byte pushed here counts against the ceiling.
Token read in place from the git-ignored Keys.txt; never printed, never on argv.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.utils import EntryNotFoundError  # noqa: E402

REPO = "Sayood/tanitad-colab-relay"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--quota", required=True, help="hf_quota_check.json, at most 24 h old")
    ap.add_argument("--ceiling-gb", type=float, default=1000.0)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    tok = re.findall(r"hf_[A-Za-z0-9]{20,}", Path("D:/Projects/TanitAD/Keys.txt").read_text(encoding="utf-8", errors="replace"))
    if len(tok) != 1:
        print(f"ZZPUSH_REFUSED expected exactly one hf_ token in Keys.txt, found {len(tok)}")
        return 1
    age_h = (time.time() - os.path.getmtime(a.quota)) / 3600
    q = json.load(open(a.quota, encoding="utf-8"))
    priv = q["used_bytes_private"] / 1e9
    assert abs(priv - sum(r["used_bytes"] for r in q["repos"] if r.get("private")) / 1e9) < 1e-6, "readout inconsistent"
    assert q.get("n_repos_missing_used_storage", 1) == 0, "a repo without usedStorage makes the sum a floor, not a total"
    push = sum(os.path.getsize(f) for f in a.files) / 1e9
    print(f"pre-push: private used {priv:.3f} GB (readout {age_h:.1f} h old) + push {push:.3f} GB = "
          f"{priv + push:.3f} GB vs ceiling {a.ceiling_gb:.0f} GB")
    if age_h > 24 or priv + push >= a.ceiling_gb:
        print("ZZPUSH_REFUSED quota readout too old or the push would cross the private ceiling")
        return 1
    api = HfApi(token=tok[0])
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        print("ZZPUSH_REFUSED the relay repo exists but is NOT private")
        return 1
    try:
        man = json.load(open(hf_hub_download(REPO, f"{a.prefix}/MANIFEST.json", repo_type="dataset", token=tok[0],
                                             force_download=True), encoding="utf-8"))
    except EntryNotFoundError:
        man = {"repo": REPO, "prefix": a.prefix, "files": {}}
    for f in a.files:
        rel = os.path.basename(f)
        ent = {"bytes": os.path.getsize(f), "md5": md5(f), "source": f,
               "pushed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        t0 = time.time()
        api.upload_file(path_or_fileobj=f, path_in_repo=f"{a.prefix}/{rel}", repo_id=REPO, repo_type="dataset",
                        commit_message=f"relay {rel} md5 {ent['md5']}")
        ent["upload_s"] = round(time.time() - t0, 1)
        man["files"][rel] = ent
        print(f"pushed {rel} {ent['bytes']} B md5 {ent['md5']} in {ent['upload_s']} s")
    tmp = Path(os.environ.get("TEMP", ".")) / "relay_manifest.json"
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo=f"{a.prefix}/MANIFEST.json", repo_id=REPO,
                    repo_type="dataset", commit_message=f"MANIFEST ({len(man['files'])} files)")
    print(f"ZZPUSH_OK {REPO}:{a.prefix} ({len(man['files'])} files in MANIFEST)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
