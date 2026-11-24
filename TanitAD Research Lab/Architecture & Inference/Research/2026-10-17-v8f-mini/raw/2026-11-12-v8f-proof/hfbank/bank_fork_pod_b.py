"""bank_fork_pod_b.py -- second bank of the v8proof fork pod (PI 2026-11-24, "Bank encodes and runs") to the same private HF dataset.

Runs on the fork pod. Two parts:
  (a) cropscale/enc_out (the token encodes of card ADD104/ADD106 and after): every file uploaded as itself under cropscale_enc_out/<path>, in batched commits.
  (b) runs (rollout and training runs of the fork, 62 run folders): tarred in chunks of about 2 GB into /workspace/v8proof/hfbank_b/tars, uploaded one by one.
Then reads the hub's own hash of every uploaded file and compares (LFS sha256; for a file stored without LFS, the git blob sha1).
Writes MANIFEST_B.json, MANIFEST_B_FILES.txt, BANKB_VERIFY.json, and BANKB_DONE.txt or BANKB_STOP.txt in /workspace/v8proof/hfbank_b.
Not banked by this script: cropscale/stage and cropscale/hfdl (copies of HF inputs, re-creatable), and the other folders outside the go.
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
from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

V = "/workspace/v8proof"
OUT = V + "/hfbank_b"
TARS = OUT + "/tars"
REPO_NAME = "tanitad-v8proof-fork-bank"
ENC = "cropscale/enc_out"
ENC_PREFIX = "cropscale_enc_out/"
RUNS = "runs"
CHUNK = 2 << 30
BATCH_BYTES = 4 << 30
BATCH_FILES = 150


def ts():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    with io.open(OUT + "/bank_b.log", "a", encoding="utf-8") as fh:
        fh.write("[%s] %s\n" % (ts(), msg))
    print("[%s] %s" % (ts(), msg), flush=True)


def stop(msg):
    io.open(OUT + "/BANKB_STOP.txt", "w").write("ZZ BANKB STOP %s %s\n" % (msg, ts()))
    log("STOP: " + msg)
    return 1


def hashes(p):
    """sha256 and git blob sha1 of a file."""
    h, g = hashlib.sha256(), hashlib.sha1()
    g.update(b"blob %d\0" % os.path.getsize(p))
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
            g.update(b)
    return h.hexdigest(), g.hexdigest()


def files_under(rel):
    out = []
    for root, _, fs in os.walk(os.path.join(V, rel)):
        for f in sorted(fs):
            p = os.path.join(root, f)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(os.path.relpath(p, V))
    return sorted(out)


def run_chunks():
    base = os.path.join(V, RUNS)
    runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not os.path.islink(os.path.join(base, d)))
    groups, cur, size = [], [], 0
    for r in runs:
        fs = files_under(RUNS + "/" + r)
        s = sum(os.path.getsize(os.path.join(V, f)) for f in fs)
        if cur and size + s > CHUNK:
            groups.append(cur)
            cur, size = [], 0
        cur += fs
        size += s
    if cur:
        groups.append(cur)
    loose = [os.path.relpath(os.path.join(base, f), V) for f in sorted(os.listdir(base))
             if os.path.isfile(os.path.join(base, f)) and not os.path.islink(os.path.join(base, f))]
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
    s256, b1 = hashes(p)
    return {"file": name, "size": os.path.getsize(p), "sha256": s256, "blob_sha1": b1, "n_members": len(members)}


def retry(fn, what, tries=4):
    for k in range(tries):
        try:
            fn()
            return True
        except Exception as ex:  # noqa: BLE001
            log("%s attempt %d failed: %s" % (what, k + 1, type(ex).__name__))
            time.sleep(60 * (k + 1))
    return False


def main():
    os.makedirs(TARS, exist_ok=False)
    tok = io.open("/root/.hf_token", encoding="utf-8").read().strip()
    api = HfApi(token=tok)
    repo = "%s/%s" % (api.whoami()["name"], REPO_NAME)
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    if not getattr(info, "private", False):
        return stop("the repo is not private")
    log("start; repo %s (private dataset)" % REPO_NAME)
    want = {}  # path_in_repo -> (sha256, blob_sha1, size)
    listing = []
    # (a) encodes, file by file, batched commits
    enc = files_under(ENC)
    log("enc_out: %d files %.1f GB" % (len(enc), sum(os.path.getsize(os.path.join(V, f)) for f in enc) / 1e9))
    for f in enc:
        p = os.path.join(V, f)
        s256, b1 = hashes(p)
        rp = ENC_PREFIX + os.path.relpath(f, ENC)
        want[rp] = (s256, b1, os.path.getsize(p))
        listing.append("%s\t%s\t%d\t%s" % (rp, f, os.path.getsize(p), s256))
    log("enc_out hashed")
    batches, cur, size = [], [], 0
    for f in enc:
        s = os.path.getsize(os.path.join(V, f))
        if cur and (size + s > BATCH_BYTES or len(cur) >= BATCH_FILES):
            batches.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += s
    if cur:
        batches.append(cur)
    for i, b in enumerate(batches):
        ops = [CommitOperationAdd(path_in_repo=ENC_PREFIX + os.path.relpath(f, ENC), path_or_fileobj=os.path.join(V, f)) for f in b]
        ok = retry(lambda: api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message="bank enc_out batch %d" % i),
                   "enc_out batch %d" % i)
        log("enc_out batch %d/%d (%d files) %s" % (i + 1, len(batches), len(b), "OK" if ok else "FAILED"))
        if not ok:
            return stop("upload of enc_out batch %d failed" % i)
    # (b) runs, chunked tars
    man = {"card_go": "PI 2026-11-24 'Bank encodes and runs'", "root": V, "repo_name": REPO_NAME, "enc_prefix": ENC_PREFIX,
           "enc_files": len(enc), "tars": []}
    for i, g in enumerate(run_chunks()):
        name = "runs_part%02d.tar" % i
        rec = make_tar(name, g)
        man["tars"].append(rec)
        want[name] = (rec["sha256"], rec["blob_sha1"], rec["size"])
        listing += ["%s\t%s\t%d" % (name, m, os.path.getsize(os.path.join(V, m))) for m in g]
        log("tar %s %d members %.1f MB sha256 %s" % (name, rec["n_members"], rec["size"] / 1048576, rec["sha256"][:16]))
        ok = retry(lambda: api.upload_file(path_or_fileobj=os.path.join(TARS, name), path_in_repo=name, repo_id=repo, repo_type="dataset",
                                           commit_message="bank %s" % name), "upload " + name)
        log("upload %s %s" % (name, "OK" if ok else "FAILED"))
        if not ok:
            return stop("upload of %s failed" % name)
    io.open(os.path.join(OUT, "MANIFEST_B.json"), "w", encoding="utf-8").write(json.dumps(man, indent=1))
    io.open(os.path.join(OUT, "MANIFEST_B_FILES.txt"), "w", encoding="utf-8").write("\n".join(listing) + "\n")
    for n in ["MANIFEST_B.json", "MANIFEST_B_FILES.txt"]:
        p = os.path.join(OUT, n)
        s256, b1 = hashes(p)
        want[n] = (s256, b1, os.path.getsize(p))
        ok = retry(lambda: api.upload_file(path_or_fileobj=p, path_in_repo=n, repo_id=repo, repo_type="dataset", commit_message="bank " + n), "upload " + n)
        if not ok:
            return stop("upload of %s failed" % n)
    # verification against the hub's own hashes
    names = sorted(want)
    got = {}
    for k in range(0, len(names), 200):
        for p in api.get_paths_info(repo_id=repo, paths=names[k:k + 200], repo_type="dataset"):
            lfs = getattr(p, "lfs", None)
            got[p.path] = ("lfs", getattr(lfs, "sha256", None)) if lfs else ("blob", getattr(p, "blob_id", None))
    bad = []
    n_lfs = 0
    for n in names:
        kind, h = got.get(n, (None, None))
        if kind == "lfs":
            n_lfs += 1
            ok = h == want[n][0]
        elif kind == "blob":
            ok = h == want[n][1]
        else:
            ok = False
        if not ok:
            bad.append(n)
    tot = sum(v[2] for v in want.values())
    ver = {"repo_name": REPO_NAME, "n_files": len(names), "n_lfs": n_lfs, "all_equal": not bad, "mismatch": bad[:50], "n_mismatch": len(bad),
           "bytes": tot, "verified_utc": ts()}
    io.open(OUT + "/BANKB_VERIFY.json", "w", encoding="utf-8").write(json.dumps(ver, indent=1))
    if bad:
        return stop("hash mismatch on %d files" % len(bad))
    io.open(OUT + "/BANKB_DONE.txt", "w").write("ZZ BANKB DONE %d files %.2f GB all hashes equal on the hub %s\n" % (len(names), tot / 1e9, ts()))
    log("DONE %d files, %.2f GB, all hashes equal on the hub" % (len(names), tot / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
