"""Bank the D:-only scene-tap probe artifacts to the private HF repo and verify by content.

Uploads each local dir under a repo prefix, then lists the repo tree and requires, for EVERY
local file, a remote entry with the same size and (for LFS files) the same sha256. Writes
manifest.json and a marker: BANK_DONE (all verified) or BANK_FAILED <why>. Never deletes.
The token is read in place from Keys.txt and never printed.
"""
import hashlib, json, os, re, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

KEYS = r"E:/Projects/TanitAD/Keys.txt"
REPO = "Sayood/tanitad-refe-s1aux"
JOBS = [
    (r"D:/Projects/TanitAD-artifacts/scene_tap_probe", "probes/2026-10-15-scene-tap-probe"),
    (r"D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_scene_ctx",
     "probes/2026-10-16-aux-navtest-forward/c3_full_scene_ctx"),
]
OUT = os.path.dirname(os.path.abspath(__file__))


def log(m):
    with open(os.path.join(OUT, "bank.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    tok = re.search(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8").read()).group(0)
    api = HfApi(token=tok)
    info = api.repo_info(REPO, repo_type="model")
    if not info.private:
        raise SystemExit("REFUSE: repo is not private")
    local = {}
    for src, pre in JOBS:
        for root, _, files in os.walk(src):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, src).replace("\\", "/")
                local[f"{pre}/{rel}"] = p
    log(f"local files {len(local)}")
    for src, pre in JOBS:
        for attempt in range(3):
            try:
                api.upload_folder(folder_path=src, path_in_repo=pre, repo_id=REPO,
                                  repo_type="model",
                                  commit_message=f"bank D:-only probe artifacts: {pre}")
                log(f"uploaded {pre}")
                break
            except Exception as e:  # retried, then the verify decides
                log(f"upload {pre} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
                time.sleep(60)
    remote = {}
    for _, pre in JOBS:
        for e in api.list_repo_tree(REPO, path_in_repo=pre, recursive=True, expand=True):
            if hasattr(e, "size") and getattr(e, "size", None) is not None and not hasattr(e, "tree_id"):
                remote[e.path] = (e.size, (e.lfs.sha256 if getattr(e, "lfs", None) else None))
    rows, bad = [], []
    for rp, lp in sorted(local.items()):
        lsz, lsh = os.path.getsize(lp), sha(lp)
        r = remote.get(rp)
        ok = r is not None and r[0] == lsz and (r[1] is None or r[1] == lsh)
        rows.append({"repo_path": rp, "local": lp, "size": lsz, "sha256": lsh,
                     "remote_size": r[0] if r else None, "remote_sha256": r[1] if r else None,
                     "lfs_checked": bool(r and r[1]), "ok": ok})
        if not ok:
            bad.append(rp)
    man = {"repo": REPO, "n_local": len(local), "n_ok": len(rows) - len(bad), "bad": bad[:50],
           "n_lfs_checked": sum(r["lfs_checked"] for r in rows), "files": rows}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    marker = "BANK_DONE" if not bad else "BANK_FAILED"
    open(os.path.join(OUT, marker), "w").write(
        f"{marker} ok {man['n_ok']}/{man['n_local']} lfs_checked {man['n_lfs_checked']}\n")
    log(f"{marker} ok {man['n_ok']}/{man['n_local']} lfs_checked {man['n_lfs_checked']}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        open(os.path.join(OUT, "BANK_FAILED"), "w").write(f"BANK_FAILED {type(e).__name__}: {str(e)[:300]}\n")
        log(f"BANK_FAILED {type(e).__name__}: {str(e)[:300]}")
        raise
