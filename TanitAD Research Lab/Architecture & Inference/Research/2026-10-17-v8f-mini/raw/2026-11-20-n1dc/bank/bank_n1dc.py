"""MM 2026-11-20: bank the N1DC arms to a PRIVATE HF dataset, then verify by sha256 (pattern of raw/2026-11-19-v8f-recover-bank/bank_recover.py).
Objects: the two trained checkpoints (each must equal the pod's own sha256, recorded in the guard chain's log) and one tar of the eval-141 dumps
(T1-kin/hb0.2 and T1-kin/old; the dev box holds the only copy). The D1-HO dumps stay on the pod: they are re-creatable bit for bit from the
checkpoints (deterministic executor; K-HB0 control). Writes BANK_N1DC.json; ends on ZZ BANKN1DC DONE ZZ / ZZ BANKN1DC FAIL ZZ <reason>."""
import hashlib
import json
import os
import sys
import tarfile
import time

from huggingface_hub import HfApi

E = "F:/TanitAD-scratch/n1dc/read/eval141"
OUT = "F:/TanitAD-scratch/n1dc/bank"
REPO = "Sayood/tanitad-v8f-n1dc-runs"
CKPT = {"N1DC_s0/train/ckpt_last.pt": (E + "/ckpts/N1DC_s0/ckpt_last.pt", "1fdfed1878d0823aa05f7c3e7051a00bccc01ee068cb65d85252a427cdb4f1a7"),
        "N1DC_s1/train/ckpt_last.pt": (E + "/ckpts/N1DC_s1/ckpt_last.pt", "9d5266a1e5a5bbf94ed075d5b5d82c29041142202985f82e56de0e85d26fc45a")}
DUMP_DIRS = ("dumps141_hb02", "dumps141_old")
TAR_REL = "eval141/n1dc_eval141_dumps.tar"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hub_sha(api):
    out = {}
    for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
        lfs = getattr(e, "lfs", None)
        if lfs is not None:
            out[e.path] = lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
    return out


def fail(msg):
    print(f"ZZ BANKN1DC FAIL ZZ {msg}", flush=True)
    sys.exit(1)


def main():
    items, local = [], {}
    for rel, (p, want) in CKPT.items():
        got = sha(p)
        if got != want:
            fail(f"{p} sha256 {got[:16]} is not the pod's {want[:16]}")
        items.append((p, rel))
        local[rel] = got
    tar_p = os.path.join(OUT, "n1dc_eval141_dumps.tar")
    n_files = 0
    with tarfile.open(tar_p, "w") as t:
        for d in DUMP_DIRS:
            for root, _, files in sorted(os.walk(os.path.join(E, d))):
                for f in sorted(files):
                    p = os.path.join(root, f)
                    t.add(p, arcname=os.path.relpath(p, E).replace("\\", "/"))
                    n_files += 1
    items.append((tar_p, TAR_REL))
    local[TAR_REL] = sha(tar_p)
    print(f"local hashed: {len(local)} objects; tar {n_files} files {os.path.getsize(tar_p)} B", flush=True)
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(REPO, repo_type="dataset").private:
        fail("repo is not private")
    have = hub_sha(api)
    for p, rel in items:
        if have.get(rel) == local[rel]:
            print(f"skip (equal on hub) {rel}", flush=True)
            continue
        for k in range(4):
            try:
                api.upload_file(path_or_fileobj=p, path_in_repo=rel, repo_id=REPO, repo_type="dataset", commit_message=f"bank {rel}")
                print(f"uploaded {rel} {os.path.getsize(p)}", flush=True)
                break
            except Exception as e:
                print(f"retry {k} {rel}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 * (k + 1))
        else:
            fail(f"upload failed {rel}")
    hub = hub_sha(api)
    rows = {rel: {"local_sha256": local[rel], "hub_lfs_sha256": hub.get(rel), "equal": hub.get(rel) == local[rel]} for _, rel in items}
    bad = [r for r, v in rows.items() if not v["equal"]]
    with open(os.path.join(OUT, "BANK_N1DC.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": REPO, "repo_type": "dataset", "private": True, "n_objects": len(rows), "tar_dirs": list(DUMP_DIRS), "tar_n_files": n_files,
                   "d1ho_dumps": "not banked here: on the pod, re-creatable bit for bit from the banked checkpoints", "objects": rows, "not_equal": bad}, f, indent=1)
    if bad:
        fail(f"{len(bad)} objects not equal on hub: {bad}")
    print(f"ZZ BANKN1DC DONE ZZ {len(rows)} objects equal on hub (2 checkpoints + 1 tar of {n_files} files)", flush=True)


if __name__ == "__main__":
    main()
