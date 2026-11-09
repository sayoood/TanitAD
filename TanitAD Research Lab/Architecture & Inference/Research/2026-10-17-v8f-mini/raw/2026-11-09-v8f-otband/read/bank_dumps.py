"""bank_dumps.py -- bank the OTBAND read's T1-kin dumps (numeric plans and head outputs, no camera frame) PRIVATE on the PI's HF
account, one uncompressed tar per tag, each verified by sha256 after a re-download.  Usage: python bank_dumps.py TAG [TAG ...]
Writes HF_BANK_<TAG>.json next to this file.  Deletes only its own tar and verify copy; the dumps stay until the folder is swept.
"""
import hashlib
import json
import os
import sys
import tarfile

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels")
import publish_context_labels as PC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = "F:/TanitAD-scratch/otband_read/t1"
WORK = "F:/TanitAD-scratch/otband_read/bank"
NAME = "tanitad-v8f-otband-read"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def members(full):
    d = f"{T1}/dump_{full}"
    out = [(f"{d}/{f}", f"dump_{full}/{f}") for f in sorted(os.listdir(d))]
    for f in (f"ticks/ticks_{full}.npz", f"done_{full}.txt", f"t1v8f_{full}.json", f"argv_{full}.txt", f"log_{full}.txt"):
        if os.path.exists(f"{T1}/{f}"):
            out.append((f"{T1}/{f}", f))
    return out


def main():
    from huggingface_hub import hf_hub_download
    api, tok, user = PC.authenticated()
    repo = f"{user}/{NAME}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(repo, repo_type="dataset").private is True, "the repo is not private"
    os.makedirs(WORK, exist_ok=True)
    rc = 0
    for tag in sys.argv[1:]:
        full = f"{tag}_m10_causal_heads"
        assert os.path.exists(f"{T1}/done_{full}.txt"), f"no done file for {tag}"
        tar = f"{WORK}/{full}.tar"
        ms = members(full)
        with tarfile.open(tar, "w") as t:
            for src, arc in ms:
                t.add(src, arcname=arc)
        want = sha(tar)
        inrepo = f"t1/{full}.tar"
        api.upload_file(path_or_fileobj=tar, path_in_repo=inrepo, repo_id=repo, repo_type="dataset", commit_message=f"OTBAND read: {tag} ({len(ms)} files)")
        info = api.repo_info(repo, repo_type="dataset")
        assert info.private is True, "the repo is not private after the upload"
        q = hf_hub_download(repo, inrepo, repo_type="dataset", revision=info.sha, token=tok, local_dir=f"{WORK}/verify")
        ok = sha(q) == want
        rec = {"repo": repo, "path": inrepo, "revision": info.sha, "n_files": len(ms), "bytes": os.path.getsize(tar), "sha256": want, "verified": ok}
        print(json.dumps(rec), flush=True)
        json.dump(rec, open(os.path.join(HERE, f"HF_BANK_{tag}.json"), "w", encoding="utf-8"), indent=1)
        if ok:
            os.remove(q)
            os.remove(tar)
        else:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
