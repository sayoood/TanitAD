# ship_div.py (from ship_dose.py, card PREREG_DIV_P6): ship_s1zsm.py with the chain, profile and tar name as arguments, and the frozen card PREREG_DOSE_P6 (sha256 037c1272...) copied next to the tar.  Arm S1ZD (PREREG_DOSE_P6).
#!/usr/bin/env python3
"""ship_scorer1p6.py: build the SCORER1P6 pod package from a landed commit, as raw blobs, with the SHIP_MANIFEST.json the chain verifies.  READ-ONLY on git (archive / show only); writes only under --work and --out.

  python ship_dose.py --tip <40-hex or ref> --out <dir> --chain <refe/ chain script name> --profile <launch-gate profile> [--tar-name dose_pkg.tar] [--work <scratch dir>] [--git-dir <mirror .git>] [--allow-unpinned] [--card <PREREG_DOSE_P6.md>]
Package (extract with:  tar -xf <tar-name> -C <the chain's package dir> --no-same-owner ):
  refe/ code/ splits/            WHOLE, from `git archive <tip>:<PK>` (core.autocrlf=false): gate_token.TREE_ROOTS are hashed whole by the pod's tree check, and test_launch_gate.py globs refe/*.py
  eval/rls1b_arms.json, eval/rls1d_arms.json, eval/rls1e_arms.json, raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md      the files test_launch_gate (G-SUITE) opens outside the roots (p7_ship_list.json, traced)
  SHIP_MANIFEST.json             made by the TIP'S OWN gate_token.ship_manifest (the code the chain imports), with commit <tip> and repo prefix "<PK>/": every manifested file must equal its blob at the tip
Refuses (rc 1, ZZSHIP6_REFUSED): a tip whose launch_gate.SCORER1P6_DATA_PIN is None (unless --allow-unpinned, a dry run), a missing chain-required file (read from the tip's own scorer1p6_chain.sh), a CR in any .sh (CR in other text blobs is a NOTE), a Windows reserved name (aux con prn nul), a tar that differs from the manifest or from the tip's blobs.  Success: ZZSHIP6_OK <n> files ... tar sha256 <64-hex>.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
ROOTS = ("refe", "code", "splits")
OUTSIDE = ("eval/rls1b_arms.json", "eval/rls1d_arms.json", "eval/rls1e_arms.json", "raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md")
RESERVED = {"aux", "con", "prn", "nul"}
CARD_SHA256 = "e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac"          # PREREG_DIV_P6.md as frozen 2026-11-10
TEXT_SUF = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".patch")


def refuse(msg):
    print("ZZSHIP6_REFUSED " + msg)
    sys.exit(1)


def git(gd, *args, check=True):
    r = subprocess.run(["git", "--git-dir", gd, "-c", "core.autocrlf=false", *args], capture_output=True)
    if check and r.returncode != 0:
        refuse("git %s failed: %s" % (" ".join(args)[:120], r.stderr.decode("utf-8", "replace").strip()[-200:]))
    return r


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="F:/tanitad_scratch/p6ship/work")
    ap.add_argument("--git-dir", default="C:/Users/Admin/tanitad-push/.git")
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--card", default="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_DIV_P6.md")
    ap.add_argument("--chain", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--tar-name", default="div_pkg.tar")
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    tip = git(a.git_dir, "rev-parse", "--verify", a.tip + "^{commit}").stdout.decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", tip):
        refuse("tip did not resolve to 40-hex: %r" % tip)
    pkg = os.path.join(a.work, "pkg")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg)
    os.makedirs(a.out, exist_ok=True)
    # 1. the three roots, raw blobs
    arc = git(a.git_dir, "archive", "--format=tar", "%s:%s" % (tip, PK), *ROOTS).stdout
    staged = {}
    with tarfile.open(fileobj=io.BytesIO(arc)) as tf:
        for m in tf.getmembers():
            if m.isdir():
                continue
            if not m.isfile() or m.name.startswith("/") or ".." in m.name.split("/"):
                refuse("unsafe or non-regular tar member %r" % m.name)
            staged[m.name] = tf.extractfile(m).read()
    for rel in OUTSIDE:
        r = git(a.git_dir, "show", "%s:%s/%s" % (tip, PK, rel), check=False)
        if r.returncode != 0:
            refuse("outside file %s is not in the tip" % rel)
        staged[rel] = r.stdout
    for rel, b in staged.items():
        stem = os.path.basename(rel).split(".")[0].lower()
        if stem in RESERVED or any(p.lower() in RESERVED for p in rel.split("/")[:-1]):
            refuse("Windows reserved name in %s" % rel)
        if rel.endswith(".sh") and b"\r" in b:
            refuse("CR in %s (a CRLF .sh dies on Linux; the blob must be LF)" % rel)
        if rel.endswith(TEXT_SUF) and not rel.endswith(".sh") and b"\r" in b:
            print("NOTE %s carries CRLF in the blob (shipped as is; harmless for .py/.json)" % rel)
    # 2. the pin and the chain's own required list
    lg = staged["refe/launch_gate.py"]
    pinned = re.search(rb"^SCORER1P6_DATA_PIN = None\n", lg, re.M) is None
    if not pinned and not a.allow_unpinned:
        refuse("launch_gate.SCORER1P6_DATA_PIN is None at %s: fill and LAND it first (fill_pin.py), then ship the new tip" % tip[:12])
    if "refe/" + a.chain not in staged:
        refuse("chain refe/%s is not in the tip" % a.chain)
    chain = staged["refe/" + a.chain].decode("utf-8")
    block = re.search(r"for f in (.*?); do\n\s*\[ -f \"\$PKG/\$f\" \]", chain, re.S)
    if not block:
        refuse("could not read the chain's required-file list")
    lst = block.group(1)
    if lst.strip() == "$NEED":                         # dose_chain.sh / rep_chain.sh keep the list in NEED="..." strings (ship_rep.py's rule)
        lst = " ".join(re.findall(r'NEED="([^"]*)"', chain, re.S))
    required = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|sh|yaml|json)", lst)
    if not required:
        refuse("the chain's required-file list is empty (a loop over a variable this script cannot read?)")
    gone = [f for f in required if f != "SHIP_MANIFEST.json" and f not in staged]
    if gone:
        refuse("chain-required files missing from the package: %s" % gone)
    for rel, b in sorted(staged.items()):
        p = os.path.join(pkg, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b)
    # 3. SHIP_MANIFEST.json by the tip's own code
    GT = load(os.path.join(pkg, "refe", "gate_token.py"), "gate_token")
    man = GT.ship_manifest(pkg, commit=tip, git_dir=a.git_dir, repo_prefix=PK + "/")
    GT.write_json(os.path.join(pkg, GT.SHIP_MANIFEST), man)
    with open(os.path.join(pkg, GT.SHIP_MANIFEST), "rb") as fh:
        staged[GT.SHIP_MANIFEST] = fh.read()
    # 4. the tar (sorted, mtime 0, owner 0): raw bytes
    tar_path = os.path.join(a.out, a.tar_name)
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        for rel in sorted(staged):
            ti = tarfile.TarInfo(rel)
            ti.size = len(staged[rel])
            ti.mtime = 0
            ti.mode = 0o755 if rel.endswith(".sh") else 0o644
            tf.addfile(ti, io.BytesIO(staged[rel]))
    h = hashlib.sha256()
    with open(tar_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    tsha = h.hexdigest()
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%s  %s\n" % (tsha, a.tar_name))
    shutil.copyfile(os.path.join(pkg, GT.SHIP_MANIFEST), os.path.join(a.out, GT.SHIP_MANIFEST))
    # 5. verify the tar: against the manifest (the chain's own check, on an extraction) and against the tip's blobs
    ex = os.path.join(a.work, "extract")
    if os.path.exists(ex):
        shutil.rmtree(ex)
    os.makedirs(ex)
    bad = []
    names = []
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            names.append(m.name)
            b = tf.extractfile(m).read()
            p = os.path.join(ex, *m.name.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(b)
            if m.name != GT.SHIP_MANIFEST:
                blob = git(a.git_dir, "show", "%s:%s/%s" % (tip, PK, m.name), check=False)
                if blob.returncode != 0 or blob.stdout != b:
                    bad.append(m.name)
    if sorted(names) != sorted(staged) or bad:
        refuse("the tar differs from the staged set / the tip's blobs: %s" % bad[:6])
    man2 = GT.read_json(os.path.join(ex, GT.SHIP_MANIFEST))
    tree = GT.tree_manifest(ex)
    if man2.get("files") != tree or man2.get("commit") != tip:
        refuse("extracted tree != SHIP_MANIFEST.json: %s" % GT.tree_diff(man2.get("files") or {}, tree))
    shutil.rmtree(ex)
    # 5b. the frozen card, next to the tar (staged on the pod where the dose chain reads it)
    cardb = open(a.card, "rb").read() if os.path.isfile(a.card) else None
    if cardb is None or hashlib.sha256(cardb).hexdigest() != CARD_SHA256:
        refuse("the card %s is missing or its sha256 is not the frozen %s" % (a.card, CARD_SHA256))
    with open(os.path.join(a.out, "PREREG_DIV_P6.md"), "wb") as fh:
        fh.write(cardb)
    # 6. the file list with reasons
    p7 = load(os.path.join(pkg, "refe", "p7_ship_list.py"), "p7_ship_list")
    clos = set(p7.closure(Path(pkg), p7.load_suite(Path(pkg), a.profile, [])))
    rows = []
    for rel in sorted(staged):
        if rel in required:
            why = "chain-required (precheck list)"
        elif rel == GT.SHIP_MANIFEST:
            why = "the manifest the chain's tree check reads"
        elif rel in OUTSIDE:
            why = "opened by test_launch_gate (G-SUITE), outside the roots"
        elif rel in clos:
            why = "G-SUITE closure (imported or named by the suite scripts)"
        else:
            why = "tree root %s/ (hashed whole by the pod tree check; test_launch_gate globs refe/*.py)" % rel.split("/")[0]
        rows.append((rel, len(staged[rel]), hashlib.sha256(staged[rel]).hexdigest(), why))
    with open(os.path.join(a.out, "ship_files.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("path\tbytes\tsha256\treason\n")
        for r in rows:
            fh.write("%s\t%d\t%s\t%s\n" % r)
    rep = {"tip": tip, "n_files": len(rows), "n_manifested": len(man["files"]), "tree_sha256": man["tree_sha256"], "tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tsha,
           "pin_set": pinned, "required_by_chain": required}
    with open(os.path.join(a.out, "ship_report.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    cnt = {}
    for r in rows:
        k = r[3].split(" (")[0]
        cnt[k] = cnt.get(k, 0) + 1
    print("ZZSHIP6_OK %d files (%d manifested) tip %s tree %s pin_set=%s tar %d bytes sha256 %s" % (len(rows), len(man["files"]), tip[:12], man["tree_sha256"][:12], pinned, rep["tar_bytes"], tsha))
    for k, v in sorted(cnt.items()):
        print("  %4d  %s" % (v, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
