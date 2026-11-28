#!/usr/bin/env python3
"""l3_dry_root.py -- builds the DRY root of the D3K_s0 / D9K_s0 dry mint on the 20-clip slice (card PREREG_V8F_LEVER3_V2 control b3 slice), under C:/Users/Admin/lever3_dry (NTFS, own data).
Layout = the real root's layout, so the argv, the templates and the checker see the same shape:
  tree_cdb   junction -> C:/Users/Admin/lever3/tree_cdb            (read only)
  init       junction -> C:/Users/Admin/lever3/init                (read only)
  cache_union junction -> E:/TanitAD-scratch/lever3/b3/repack      (the repacked 20-clip slice, read only)
  merged/    hard links to the REAL v9 union, lead union (read only, same bytes); the real speed file and the slice's own tracks / join files copied in under the union names
  bank/s0    copy of the 4-shard smoke bank of the kit (E:/TanitAD-scratch/lever3/b3/work/bank_smoke4; PARTIAL: 1,357 windows)
  runs/D3K_s0, runs/D9K_s0  argv rendered by l3_argv.render with the smoke bank's sha256 and verified (control b4)
usage: l3_dry_root.py build | remove      (remove deletes only this folder tree, by literal path, after checking the junctions are links)"""
import hashlib, json, os, shutil, subprocess, sys
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, KIT)
import l3_argv as A

DRY = "C:/Users/Admin/lever3_dry"
REAL = "C:/Users/Admin/lever3"
B3 = "E:/TanitAD-scratch/lever3/b3"


def junction(link, target):
    if os.path.lexists(link):
        return
    r = subprocess.run(["cmd", "/c", "mklink", "/J", link.replace("/", "\\"), target.replace("/", "\\")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout


def build():
    os.makedirs(f"{DRY}/merged", exist_ok=True)
    junction(f"{DRY}/tree_cdb", f"{REAL}/tree_cdb")
    junction(f"{DRY}/init", f"{REAL}/init")
    junction(f"{DRY}/cache_union", f"{B3}/repack")
    for f in ("v9_labels_union.npz", "v9_labels_union.manifest.json", "lead_union.npz"):
        if not os.path.exists(f"{DRY}/merged/{f}"):
            os.link(f"{REAL}/merged/{f}", f"{DRY}/merged/{f}")
    for src, dst in ((f"{REAL}/merged/speed_union.jsonl", "speed_union.jsonl"), (f"{B3}/labels/native_tracks_slice.pt", "native_tracks_union.pt"), (f"{B3}/join_repack.npz", "join_union.npz")):
        if not os.path.exists(f"{DRY}/merged/{dst}"):
            shutil.copy2(src, f"{DRY}/merged/{dst}")
    if not os.path.isdir(f"{DRY}/bank/s0"):
        shutil.copytree(f"{B3}/work/bank_smoke4", f"{DRY}/bank/s0")
    sha = hashlib.sha256(open(f"{DRY}/bank/s0/bank.json", "rb").read()).hexdigest()
    for arm in ("D3K_s0", "D9K_s0"):
        t = A.render(arm, sha, DRY)
        bad = A.verify_one(arm, t, DRY, sha)
        assert not bad, bad
        os.makedirs(f"{DRY}/runs/{arm}", exist_ok=True)
        json.dump({"argv": t}, open(f"{DRY}/runs/{arm}/argv.json", "w"), indent=1)
    print("dry root built; bank sha256", sha)


def remove():
    for j in ("tree_cdb", "init", "cache_union"):
        p = f"{DRY}/{j}"
        if os.path.lexists(p):
            os.rmdir(p)                                   # a junction: removes the link, never the target
    shutil.rmtree(DRY)
    print("removed", DRY)


if __name__ == "__main__":
    {"build": build, "remove": remove}[sys.argv[1]]()
