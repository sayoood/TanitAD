"""bank_fork_pod.py -- banks the only-on-pod set of the v8proof fork (PI 2026-11-24, "Yes, bank it") to a private HF dataset.

Runs on the fork pod. Tars the set into /workspace/v8proof/hfbank/tars (rollout folders of integ10k/t1 in chunks of about 1 GB; one tar of the small set),
writes MANIFEST.json (per tar: size, sha256, members) and MANIFEST_FILES.txt (every member path and size), uploads them one by one with retries
(HF_HUB_DISABLE_XET=1), then reads the hub's own sha256 of every uploaded file and compares. Writes BANK_VERIFY.json and BANK_DONE.txt or BANK_STOP.txt.
The token is read in this process from /root/.hf_token and never printed. It deletes, moves and stops nothing.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi  # noqa: E402

V = "/workspace/v8proof"
OUT = V + "/hfbank"
TARS = OUT + "/tars"
REPO_NAME = "tanitad-v8proof-fork-bank"
CHUNK = 1 << 30
SMALL = ["integ10k/ckpt", "integ10k/lr16", "integ10k/qmap_sweep", "cropscale/gapfiles", "gapmargin/gapfiles", "boxfuse10k/gapfiles", "gapspread10k/gapfiles",
         "boxfuse10k/det_out"]
SMALL_GLOB_ROOT = "cropscale"
SMALL_MAX = 20 << 20


def ts():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    with io.open(OUT + "/bank.log", "a", encoding="utf-8") as fh:
        fh.write("[%s] %s\n" % (ts(), msg))
    print("[%s] %s" % (ts(), msg), flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files_under(rel):
    out = []
    for root, _, fs in os.walk(os.path.join(V, rel)):
        for f in sorted(fs):
            p = os.path.join(root, f)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(os.path.relpath(p, V))
    return sorted(out)


def small_set():
    out = []
    for rel in SMALL:
        if os.path.isdir(os.path.join(V, rel)):
            out += files_under(rel)
    for root, dirs, fs in os.walk(os.path.join(V, SMALL_GLOB_ROOT)):
        if root.count(os.sep) - os.path.join(V, SMALL_GLOB_ROOT).count(os.sep) >= 3:
            dirs[:] = []
        for f in fs:
            p = os.path.join(root, f)
            if (f.endswith(".pt") or f.endswith(".json")) and os.path.isfile(p) and os.path.getsize(p) <= SMALL_MAX:
                out.append(os.path.relpath(p, V))
    return sorted(set(out))


def chunks_of_t1():
    base = os.path.join(V, "integ10k/t1")
    runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    groups, cur, size = [], [], 0
    for r in runs:
        fs = files_under("integ10k/t1/" + r)
        s = sum(os.path.getsize(os.path.join(V, f)) for f in fs)
        if cur and size + s > CHUNK:
            groups.append(cur)
            cur, size = [], 0
        cur += fs
        size += s
    if cur:
        groups.append(cur)
    loose = [os.path.relpath(os.path.join(base, f), V) for f in sorted(os.listdir(base)) if os.path.isfile(os.path.join(base, f))]
    if loose:
        groups.append(loose)
    return groups


def make_tar(name, members):
    p = os.path.join(TARS, name)
    if os.path.exists(p):
        raise SystemExit("REFUSED: %s exists" % p)
    with tarfile.open(p, "w") as tf:
        for m in members:
            tf.add(os.path.join(V, m), arcname=m, recursive=False)
    return {"file": name, "size": os.path.getsize(p), "sha256": sha_file(p), "n_members": len(members)}


def upload(api, repo, path, name, tries=4):
    for k in range(tries):
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo, repo_type="dataset", commit_message="bank %s" % name)
            return True
        except Exception as ex:  # noqa: BLE001
            log("upload %s attempt %d failed: %s" % (name, k + 1, type(ex).__name__))
            time.sleep(30 * (k + 1))
    return False


def main():
    os.makedirs(TARS, exist_ok=False)
    tok = io.open("/root/.hf_token", encoding="utf-8").read().strip()
    api = HfApi(token=tok)
    user = api.whoami()["name"]
    repo = "%s/%s" % (user, REPO_NAME)
    log("start; repo %s (private dataset)" % REPO_NAME)
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    if not getattr(info, "private", False):
        io.open(OUT + "/BANK_STOP.txt", "w").write("ZZ BANK STOP the repo is not private %s\n" % ts())
        log("STOP: the repo is not private")
        return 3
    plan = [("small_set.tar", small_set())] + [("integ10k_t1_part%02d.tar" % i, g) for i, g in enumerate(chunks_of_t1())]
    man, listing = {"card_go": "PI 2026-11-24 'Yes, bank it'", "root": V, "repo_name": REPO_NAME, "tars": []}, []
    for name, members in plan:
        rec = make_tar(name, members)
        man["tars"].append(rec)
        listing += ["%s\t%s\t%d" % (name, m, os.path.getsize(os.path.join(V, m))) for m in members]
        log("tar %s %d members %.1f MB sha256 %s" % (name, rec["n_members"], rec["size"] / 1048576, rec["sha256"][:16]))
    io.open(os.path.join(TARS, "MANIFEST.json"), "w", encoding="utf-8").write(json.dumps(man, indent=1))
    io.open(os.path.join(TARS, "MANIFEST_FILES.txt"), "w", encoding="utf-8").write("\n".join(listing) + "\n")
    names = [r["file"] for r in man["tars"]] + ["MANIFEST.json", "MANIFEST_FILES.txt"]
    for n in names:
        ok = upload(api, repo, os.path.join(TARS, n), n)
        log("upload %s %s" % (n, "OK" if ok else "FAILED"))
        if not ok:
            io.open(OUT + "/BANK_STOP.txt", "w").write("ZZ BANK STOP upload of %s failed %s\n" % (n, ts()))
            return 1
    want = {r["file"]: r["sha256"] for r in man["tars"]}
    want["MANIFEST.json"] = sha_file(os.path.join(TARS, "MANIFEST.json"))
    want["MANIFEST_FILES.txt"] = sha_file(os.path.join(TARS, "MANIFEST_FILES.txt"))
    got = {}
    for p in api.get_paths_info(repo_id=repo, paths=names, repo_type="dataset"):
        lfs = getattr(p, "lfs", None)
        got[p.path] = getattr(lfs, "sha256", None) if lfs else None
    bad = []
    for n, h in want.items():
        g = got.get(n)
        if g is None:
            # a small file stored without LFS: compare by downloading it
            q = api.hf_hub_download(repo_id=repo, filename=n, repo_type="dataset", local_dir=OUT + "/verify_dl")
            g = sha_file(q)
        if g != h:
            bad.append(n)
    ver = {"repo_name": REPO_NAME, "n_files": len(want), "all_equal": not bad, "mismatch": bad, "sha256": want, "verified_utc": ts()}
    io.open(OUT + "/BANK_VERIFY.json", "w", encoding="utf-8").write(json.dumps(ver, indent=1))
    if bad:
        io.open(OUT + "/BANK_STOP.txt", "w").write("ZZ BANK STOP sha256 mismatch %s %s\n" % (bad, ts()))
        log("STOP: mismatch %s" % bad)
        return 1
    tot = sum(r["size"] for r in man["tars"])
    io.open(OUT + "/BANK_DONE.txt", "w").write("ZZ BANK DONE %d files %.2f GB all sha256 equal on the hub %s\n" % (len(want), tot / 1e9, ts()))
    log("DONE %d files, %.2f GB, all sha256 equal on the hub" % (len(want), tot / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
