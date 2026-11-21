"""MM 2026-11-21: bank the four ROLLHIST arms (RH_s0, B0_s0, RH_s1, B0_s1) to a PRIVATE HF dataset, then verify by sha256 (pattern of
F:/TanitAD-scratch/n1dc/bank/bank_n1dc.py). Each checkpoint must equal the pod's own sha256, as recorded by the read chain's pull
(F:/TanitAD-scratch/rollhist/read/ckpts/SHA256_PODS.txt). The banks (2 x 2.0 GB, /root/rollhist/bank/s0, s1) stay on the pod: they are re-creatable from
the UE.A checkpoints by the generator (K-B1 replay identity). The read's dumps are banked with the read result, not here.
Writes BANK_ROLLHIST.json; ends on ZZ BANKRH DONE ZZ / ZZ BANKRH FAIL ZZ <reason>. Token: HF_TOKEN in the environment (read in place from Keys.txt by the launcher)."""
import hashlib
import json
import os
import sys
import time

from huggingface_hub import HfApi

CK = "F:/TanitAD-scratch/rollhist/read/ckpts"
OUT = "F:/TanitAD-scratch/rollhist/bank"
REPO = "Sayood/tanitad-v8f-rollhist-runs"
ARMS = ("RH_s0", "B0_s0", "RH_s1", "B0_s1")


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
    print(f"ZZ BANKRH FAIL ZZ {msg}", flush=True)
    sys.exit(1)


def main():
    pods = {}
    for line in open(os.path.join(CK, "SHA256_PODS.txt"), encoding="utf-8"):
        h, p = line.split()
        pods[p.split("/")[0]] = h
    items, local = [], {}
    for arm in ARMS:
        p = f"{CK}/{arm}/ckpt_last.pt"
        rel = f"{arm}/train/ckpt_last.pt"
        if arm not in pods:
            fail(f"no pod sha256 recorded for {arm}")
        got = sha(p)
        if got != pods[arm]:
            fail(f"{p} sha256 {got[:16]} is not the pod's {pods[arm][:16]}")
        items.append((p, rel))
        local[rel] = got
    print(f"local hashed: {len(local)} checkpoints, each equal to the pod's sha256", flush=True)
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
            except Exception as e:  # noqa: BLE001 -- retried, then reported
                print(f"retry {k} {rel}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 * (k + 1))
        else:
            fail(f"upload failed {rel}")
    hub = hub_sha(api)
    rows = {rel: {"local_sha256": local[rel], "hub_lfs_sha256": hub.get(rel), "equal": hub.get(rel) == local[rel]} for _, rel in items}
    bad = [r for r, v in rows.items() if not v["equal"]]
    with open(os.path.join(OUT, "BANK_ROLLHIST.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": REPO, "repo_type": "dataset", "private": True, "n_objects": len(rows), "pod_sha_source": "read/ckpts/SHA256_PODS.txt (the pod's own sha256sum at pull)",
                   "banks": "not banked: /root/rollhist/bank/s0, s1 (2.0 GB each) on the pod, re-creatable by the generator from the UE.A checkpoints",
                   "objects": rows, "not_equal": bad}, f, indent=1)
    if bad:
        fail(f"hub sha256 differs for {bad}")
    print("ZZ BANKRH DONE ZZ", flush=True)


if __name__ == "__main__":
    main()
