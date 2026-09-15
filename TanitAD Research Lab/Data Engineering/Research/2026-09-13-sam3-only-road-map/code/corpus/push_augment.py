"""Push staged augmentation files to the private HF corpus repo (PI 2026-09-15). Commits of at most 90 files (HF: manual commits ~50-100
files), data files before manifests; after each commit every file is verified on the Hub -- LFS/xet files by the sha256 the Hub reports,
small non-LFS files by downloading them back and hashing. Files already on the Hub with the same sha256 are skipped (resumable).
Refuses when the account's private storage plus this push would pass 900 GB (Pro includes 1 TB private; beyond it is billed).
Usage: push_augment.py <stage dir> <prefix>[,<prefix>...]      e.g. push_augment.py D:/.../stage calibration,lidar_bev_gt,agents"""
import hashlib, json, re, sys, tempfile, time
from pathlib import Path
import truststore; truststore.inject_into_ssl()
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = "Sayood/tanitad-v7-training-corpus"; CEILING_GB = 900.0; PER_COMMIT = 90
STAGE = Path(sys.argv[1]); PREFIXES = tuple(sys.argv[2].split(","))
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok)
LOG = STAGE.parent / "push_augment.log"


def log(m):
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + m; print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def remote_info(paths):
    out = {}
    for k in range(0, len(paths), 400):
        for f in api.get_paths_info(REPO, paths[k:k + 400], repo_type="dataset", expand=True):
            if type(f).__name__ == "RepoFile":
                out[f.path] = f
    return out


def matches(f, want):
    return f is not None and f.size == want["bytes"] and f.lfs is not None and f.lfs.sha256 == want["sha256"]


def main():
    listed = json.loads((STAGE / "stage_files.json").read_text(encoding="utf-8"))
    files = {p: v for p, v in listed.items() if p.startswith(PREFIXES)}
    for p, v in files.items():                                                 # the stage must still hold what was listed
        assert (STAGE / p).stat().st_size == v["bytes"], p
    order = sorted(files, key=lambda p: ("MANIFEST" in p, p))
    rem = remote_info(order)
    todo = [p for p in order if not matches(rem.get(p), files[p])]
    log(f"{len(files)} staged files under {PREFIXES}; {len(files) - len(todo)} already on the Hub with the same sha256; {len(todo)} to push "
        f"({sum(files[p]['bytes'] for p in todo) / 1e9:.3f} GB)")
    if not todo:
        print("ZZPUSH-0-0ZZ"); return
    me = api.whoami()["name"]; priv = 0
    for lister, info in ((api.list_models, api.model_info), (api.list_datasets, api.dataset_info), (api.list_spaces, api.space_info)):
        for r in lister(author=me):
            i = info(r.id, expand=["usedStorage", "private"]); priv += (getattr(i, "used_storage", 0) or 0) if i.private else 0
    push_gb = sum(files[p]["bytes"] for p in todo) / 1e9
    log(f"account private storage {priv / 1e9:.2f} GB + this push {push_gb:.3f} GB (ceiling {CEILING_GB} GB)")
    if priv / 1e9 + push_gb > CEILING_GB:
        log("REFUSED: would pass the private storage ceiling"); sys.exit(2)
    n_ok = n_bad = 0
    for k in range(0, len(todo), PER_COMMIT):
        batch = todo[k:k + PER_COMMIT]; comps = sorted({p.split("/")[0] for p in batch})
        t0 = time.time()
        info = api.create_commit(REPO, repo_type="dataset", commit_message=f"augmentations 2026-09 ({', '.join(comps)}): {len(batch)} files, batch {k // PER_COMMIT + 1}",
                                 operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(STAGE / p)) for p in batch])
        rem = remote_info(batch); bad = []
        for p in batch:
            f = rem.get(p)
            if f is not None and f.lfs is None and f.size == files[p]["bytes"]:        # small non-LFS file: download back and hash
                with tempfile.TemporaryDirectory() as td:
                    good = sha256(hf_hub_download(REPO, p, repo_type="dataset", revision=info.oid, token=tok, local_dir=td)) == files[p]["sha256"]
            else:
                good = matches(f, files[p])
            n_ok += good
            if not good:
                bad.append(p)
        n_bad += len(bad)
        log(f"commit {info.oid[:10]}: {len(batch)} files ({comps}) in {time.time() - t0:.0f}s, verified {len(batch) - len(bad)}, bad {len(bad)} {bad[:3]}")
        if bad:
            break
    print(f"ZZPUSH-{n_ok}-{n_bad}ZZ")


if __name__ == "__main__":
    main()
