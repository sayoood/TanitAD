r"""OTBAND tree refresh on box 2 (stdlib only). Reuses dt2_refresh.py (the earlier chain; RECOVER-FT's refresh.py before it) with the commit a PARAMETER: overlays the shipped tree tar (otband_ship_tree.py: the OTBAND trainer-change
commit's import closure) onto the OTBAND-owned tree D:\tanitad\otband\repo (never the shared D:\tanitad\repo). FIRST every member whose bytes differ is COPIED to <aside-root>\repo_aside_otband_<commit12>\<same relative path> (nothing is deleted; the aside is the exact pre-state
of every overwritten file; an aside that already exists is kept, it is the older pre-state), then each member is written through a temp file and re-hashed. Any failure puts every overwritten file back from the aside and
exits 1 WITHOUT writing TIP_OTBAND.txt. On success: <repo>\TIP_OTBAND.txt = the commit and <home>\tree_manifest.json = the manifest (otband_check.py preflight re-hashes every member against it before each gate and each
training). Unlike the earlier chain's refresh there is no second tree: the bank-tree step is gone.
    python -I otband_refresh.py --commit <40-hex> [--tar F] [--manifest F] [--result F]
Paths come from OTBAND_HOME / OTBAND_REPO / OTBAND_ASIDE_ROOT (defaults are box 2's). Exit 0 iff every member verifies."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def dst_of(root, rel):
    return os.path.join(root, *rel.split("/"))


def refresh(commit, tar_path, man_path, repo, home, aside_root, result):
    res = {"commit": commit, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": False, "verify_failures": [], "same": 0, "moved": 0, "new": 0, "rolled_back": []}
    aside = os.path.join(aside_root, f"repo_aside_otband_{commit[:12]}")
    res["aside"] = aside

    def done(ok, why=None):
        res["ok"], res["why"] = ok, why
        os.makedirs(os.path.dirname(result), exist_ok=True)
        json.dump(res, open(result, "w", encoding="utf-8", newline="\n"), indent=1)
        print(("ZZOTBAND-SHIP-OK " if ok else "ZZOTBAND-SHIP-FAIL ") + json.dumps({k: res[k] for k in ("commit", "same", "moved", "new", "verify_failures", "why")}))
        return 0 if ok else 1

    if not HEX40.match(commit or ""):
        return done(False, f"--commit {commit!r} is not a full 40-hex sha")
    man = json.load(open(man_path, encoding="utf-8"))
    if man.get("commit") != commit:
        return done(False, f"manifest commit {man.get('commit')!r} != {commit}")
    if sha_file(tar_path) != man.get("tar_sha256"):
        return done(False, f"tar sha256 {sha_file(tar_path)} != manifest {man.get('tar_sha256')}")
    tf = tarfile.open(tar_path)
    names = []
    for m in tf.getmembers():
        n = m.name
        if not m.isfile() or n.startswith("/") or ".." in n.split("/") or ":" in n or "\\" in n:
            return done(False, f"unsafe or non-file tar member {n!r}")
        names.append(n)
    if sorted(names) != sorted(man["members"]):
        return done(False, "tar members differ from the manifest's member list")
    for rel in REQUIRED:
        if rel not in man["members"]:
            return done(False, f"{rel} is not in the shipped tree (not the OTBAND trainer-change commit)")
    changed = []
    for m in tf.getmembers():
        dst = dst_of(repo, m.name)
        if os.path.exists(dst):
            if sha_file(dst) == man["members"][m.name]:
                res["same"] += 1
                continue
            a = dst_of(aside, m.name)
            if os.path.exists(a):
                print("ASIDE EXISTS (the older pre-state is kept):", a)
            else:
                os.makedirs(os.path.dirname(a), exist_ok=True)
                shutil.copy2(dst, a)
            res["moved"] += 1
            changed.append(m.name)
        else:
            res["new"] += 1
    written = []
    try:
        for m in tf.getmembers():
            dst = dst_of(repo, m.name)
            if os.path.exists(dst) and sha_file(dst) == man["members"][m.name]:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".otbandtmp"
            with open(tmp, "wb") as f:
                shutil.copyfileobj(tf.extractfile(m), f)
            os.replace(tmp, dst)
            written.append(m.name)
        res["verify_failures"] = [n for n, h in man["members"].items() if not os.path.exists(dst_of(repo, n)) or sha_file(dst_of(repo, n)) != h]
        if res["verify_failures"]:
            raise RuntimeError(f"{len(res['verify_failures'])} member(s) differ after the write")
    except Exception as e:                                  # noqa: BLE001  roll back what was overwritten; new files stay (nothing is deleted)
        for n in changed:
            if n in written:
                shutil.copy2(dst_of(aside, n), dst_of(repo, n))
                res["rolled_back"].append(n)
        return done(False, f"{type(e).__name__}: {e}")
    os.makedirs(home, exist_ok=True)
    prev = os.path.join(home, "tree_manifest.json")
    if os.path.exists(prev) and sha_file(prev) != sha_file(man_path):
        os.replace(prev, prev + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(man_path, prev)
    with open(os.path.join(repo, "TIP_OTBAND.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(commit + "\n")
    return done(True)


def main(argv=None):
    home = os.environ.get("OTBAND_HOME", r"D:\tanitad\otband")
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--tar", default=os.path.join(home, "ship", "otband_tree.tar"))
    ap.add_argument("--manifest", default=os.path.join(home, "ship", "otband_tree_manifest.json"))
    ap.add_argument("--result", default=os.path.join(home, "ship", "SHIP_RESULT.json"))
    a = ap.parse_args(argv)
    return refresh(a.commit, a.tar, a.manifest, os.environ.get("OTBAND_REPO", r"D:\tanitad\otband\repo"), home, os.environ.get("OTBAND_ASIDE_ROOT", r"D:\tanitad\otband"), a.result)


if __name__ == "__main__":
    sys.exit(main())
