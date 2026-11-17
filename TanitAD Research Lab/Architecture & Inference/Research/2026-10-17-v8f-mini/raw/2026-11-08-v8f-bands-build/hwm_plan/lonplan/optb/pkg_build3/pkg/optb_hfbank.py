"""Bank one finished Option B run PRIVATE to HF and verify it by content (dev box, tanitad venv).
    python optb_hfbank.py --tag FTRSP_s0 --bank-dir C:/Users/Admin/t2v8f_pulls/optb_FTRSP_s0 --repo <owner/name>
The token is read IN PLACE from Keys.txt (never printed, never put in an argument or a log). HF_HUB_DISABLE_XET=1 is set before huggingface_hub is imported (hf_xet hangs on the dev box).
The repo is created PRIVATE when absent and REFUSED when it exists as public. Every banked file is compared after the upload: LFS files by sha256, small files by the git blob id. Exit 0 iff every file verifies."""
import argparse
import hashlib
import os
import re
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"


def token(keys):
    m = re.search(r"hf_[A-Za-z0-9]+", open(keys, encoding="utf-8", errors="replace").read())
    if not m:
        raise SystemExit("no HF token found in " + keys)
    return m.group(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--keys", default=os.environ.get("OPTB_KEYS", "E:/Projects/TanitAD/Keys.txt"))
    a = ap.parse_args()
    from huggingface_hub import HfApi
    tok = token(a.keys)
    api = HfApi(token=tok)
    try:
        info = api.repo_info(a.repo, repo_type="model")
        if not info.private:
            print(f"REFUSED: {a.repo} exists and is PUBLIC", file=sys.stderr)
            return 2
    except Exception as e:  # noqa: BLE001
        if "404" not in str(e) and "not found" not in str(e).lower():
            print(f"REFUSED: cannot read {a.repo}: {type(e).__name__}", file=sys.stderr)
            return 3
        api.create_repo(a.repo, repo_type="model", private=True, exist_ok=False)
    files = sorted(f for f in os.listdir(a.bank_dir) if os.path.isfile(os.path.join(a.bank_dir, f)))
    if not files or "ckpt_last.pt" not in files:
        print(f"REFUSED: {a.bank_dir} holds no ckpt_last.pt", file=sys.stderr)
        return 4
    prefix = f"optb/{a.tag}"
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=a.bank_dir, path_in_repo=prefix, commit_message=f"OPTB {a.tag} bank (private)")
    remote = {e.path: e for e in api.list_repo_tree(a.repo, path_in_repo=prefix, recursive=True, expand=True) if hasattr(e, "size")}
    bad = []
    for f in files:
        p = os.path.join(a.bank_dir, f)
        data = open(p, "rb").read()
        e = remote.get(f"{prefix}/{f}")
        if e is None or e.size != len(data):
            bad.append(f"{f}: missing or size differs")
            continue
        lfs = getattr(e, "lfs", None)
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if sha != hashlib.sha256(data).hexdigest():
                bad.append(f"{f}: LFS sha256 differs")
        elif e.blob_id != hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest():
            bad.append(f"{f}: blob id differs")
    if bad:
        print("VERIFY-FAIL " + "; ".join(bad), file=sys.stderr)
        return 5
    print(f"ZZ OPTB-HF-OK {a.tag} {len(files)} files verified in a private repo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
