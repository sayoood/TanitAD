"""OBD tree export (dev box; stdlib + git). Reuses dt2_ship_tree.py (the earlier chain, itself modelled on RECOVER-FT's repo_recft_tip.tar): the overlay that puts the LANDED OBD trainer-change commit on box 2 is a tar of the
trainer's import closure taken byte for byte from the commit, and the manifest `obd_tree_manifest.json` (commit, tar sha256, every member's sha256 and git blob id). Changes against the the earlier chain file: the needles are OBD's
(obd_check.NEEDLES: trainer, model and data module), no T2-bank library is required, names are obd_*.

    python -I obd_ship_tree.py build --repo <git repo> --commit <40-hex> --export-list <file> --out-dir <dir>

Rules (each is a refusal that names its reason, exit 3; nothing is written on a refusal):
  * the commit is the FULL 40-hex sha and `git rev-parse --verify <sha>^{commit}` returns exactly it (an abbreviation or a branch name identifies nothing: the gate token binds to it);
  * every member is read with `git cat-file --batch` (raw blob, no eol conversion: the mirror is autocrlf=true and `git archive` would CRLF the source and change every hash) and its blob id is re-computed;
  * the import closure of the trainer (ast, every `tanitad` / `taniteval` import, lazy ones included) is added to the export list ("auto_added" in the manifest) and the list must name nothing that is not in the commit;
  * the commit's trainer, model and data module must carry every string of obd_check.NEEDLES (the --w-op / --w-tac-traj / --t-stok-dir / --freeze flags, bad_oband, oband_probe, frozen_groups): a commit that is not the OBD trainer-change commit is refused HERE, not at the box-2 gate;
  * the output folder is keyed by the commit and is never overwritten.
"""
import argparse
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import obd_check as CK  # noqa: E402

HEX40 = re.compile(r"^[0-9a-f]{40}$")
ROOTS = (("tanitad", "stack/tanitad"), ("taniteval", "taniteval/taniteval"))      # import root -> repo folder holding the package


class Refused(Exception):
    pass


def git(repo, *args, data=None):
    r = subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", repo, *args], input=data, capture_output=True)
    if r.returncode != 0:
        raise Refused(f"git {' '.join(args[:3])} failed: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout


def tree_paths(repo, commit):
    return set(p for p in git(repo, "ls-tree", "-r", "-z", "--name-only", commit).decode("utf-8").split("\0") if p)


class Blobs:
    """raw blobs of one commit through ONE `git cat-file --batch` process"""

    def __init__(self, repo, commit):
        self.commit = commit
        self.p = subprocess.Popen(["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def get(self, path):
        self.p.stdin.write(f"{self.commit}:{path}\n".encode("utf-8"))
        self.p.stdin.flush()
        head = self.p.stdout.readline().decode("utf-8", "replace").split()
        if len(head) != 3 or head[1] != "blob":
            raise Refused(f"{path} is not a blob of {self.commit[:12]} ({' '.join(head)})")
        data = self.p.stdout.read(int(head[2]))
        self.p.stdout.read(1)
        if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != head[0]:
            raise Refused(f"{path}: the bytes read do not hash to the blob id {head[0]}")
        return data, head[0]

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def resolve(mod, paths):
    """module name -> the repo file (or None): a package's __init__ or a plain module"""
    top = mod.split(".")[0]
    for root, base in ROOTS:
        if top == root:
            rel = base + "/" + "/".join(mod.split(".")[1:]) if "." in mod else base
            rel = rel.rstrip("/")
            for cand in (rel + ".py", rel + "/__init__.py"):
                if cand in paths:
                    return cand
    return None


def package_of(f):
    """repo file -> its dotted package (the folder it sits in), or None when it is outside the import roots (a script)"""
    for root, base in ROOTS:
        if f.startswith(base + "/"):
            return ".".join([root] + f[len(base) + 1:].split("/")[:-1])
    return None


def imports_of(src, pkg=None):
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module)
            out.update(f"{n.module}.{a.name}" for a in n.names)                # `from pkg import sub` may name a submodule; an attribute resolves to nothing and is ignored
        elif isinstance(n, ast.ImportFrom) and n.level > 0 and pkg:            # `from . import x`, `from .y import z`: resolved against the importing file's package
            base = pkg.split(".")[:len(pkg.split(".")) - (n.level - 1)]
            if not base:
                continue
            mod = ".".join(base + ([n.module] if n.module else []))
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in n.names)
    return out


def closure(blobs, paths, start):
    """every repo file the trainer can import, plus the __init__.py of each parent package"""
    seen, todo = set(), [start]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for m in imports_of(blobs.get(f)[0].decode("utf-8", "replace"), package_of(f)):
            r = resolve(m, paths)
            if r and r not in seen:
                todo.append(r)
            parts = m.split(".")
            for i in range(1, len(parts)):                                       # parent packages
                pr = resolve(".".join(parts[:i]), paths)
                if pr and pr not in seen:
                    todo.append(pr)
    return seen


def read_list(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.lstrip().startswith("#")]


def build(repo, commit, export_list, out_dir):
    if not HEX40.match(commit or ""):
        raise Refused(f"--commit {commit!r} is not a full 40-hex sha (an abbreviation or a branch name identifies nothing)")
    if git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise Refused(f"{commit} is not a commit of {repo}")
    out_dir = out_dir.replace("\\", "/")
    if os.path.exists(f"{out_dir}/obd_tree.tar"):
        raise Refused(f"{out_dir}/obd_tree.tar exists (the builder never overwrites; move the folder aside)")
    paths = tree_paths(repo, commit)
    listed = read_list(export_list)
    missing = sorted(set(listed) - paths)
    if missing:
        raise Refused(f"the export list names {len(missing)} path(s) that are not in {commit[:12]}: {missing[:5]}")
    blobs = Blobs(repo, commit)
    try:
        for rel, needles in CK.NEEDLES.items():
            if rel not in paths:
                raise Refused(f"{commit[:12]} has no {rel}")
            src = blobs.get(rel)[0].decode("utf-8", "replace")
            lacks = [n for n in needles if n not in src]
            if lacks:
                raise Refused(f"{commit[:12]}'s {rel} lacks {lacks}: this is not the OBD trainer-change commit")
        clo = closure(blobs, paths, CK.TRAINER_REL) | {CK.MODEL_REL, CK.DATA_REL}
        auto = sorted(clo - set(listed))
        members = sorted(set(listed) | clo)
        data, blob = {}, {}
        for m in members:
            data[m], blob[m] = blobs.get(m)
    finally:
        blobs.close()
    os.makedirs(out_dir, exist_ok=True)
    tar_path = f"{out_dir}/obd_tree.tar"
    with tarfile.open(tar_path + ".tmp", "w") as tf:
        for m in members:
            ti = tarfile.TarInfo(m)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data[m]), 1700000000, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data[m]))
    os.replace(tar_path + ".tmp", tar_path)
    tsha = hashlib.sha256(open(tar_path, "rb").read()).hexdigest()
    man = {"schema": "obd-tree-v1", "commit": commit, "tar_sha256": tsha, "n_members": len(members), "auto_added": auto,
           "export_list_sha256": hashlib.sha256(open(export_list, "rb").read()).hexdigest(),
           "members": {m: hashlib.sha256(data[m]).hexdigest() for m in members}, "git_blob_ids": blob}
    with open(f"{out_dir}/obd_tree_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{tsha} *obd_tree.tar\n")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for f in ("--repo", "--commit", "--export-list", "--out-dir"):
        b.add_argument(f, required=True)
    a = ap.parse_args(argv)
    try:
        man = build(a.repo, a.commit, a.export_list, a.out_dir)
    except (Refused, OSError, SyntaxError) as e:
        print(f"ZZ OBD-SHIP-REFUSED {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    print(json.dumps({k: man[k] for k in ("commit", "tar_sha256", "n_members", "auto_added")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
