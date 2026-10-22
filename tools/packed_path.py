"""Resolve, extract and verify repo paths that were PACKED into a tar to keep every repo path <= 220 chars.

Why packing exists (Master Mind 2026-10-22, PI decision "Shorten the paths"): the launch gate's dev-box half
(G-SUITE-PINNED) archives the FULL commit onto Windows, where a path over MAX_PATH (260) cannot be created
without LongPathsEnabled. 171 repo paths exceeded 220 chars (max 286), nearly all NavSim scorer outputs whose
names repeat the run name three times. Renaming them would have rewritten hundreds of historical references
(logs, manifests, pod paths); packing keeps every original name and byte INSIDE a tar beside where it lived,
so every existing reference stays true and resolves through this tool.

Registry: tools/packed_paths.json. Each pack: the tar, its MEMBERS sidecar (arcname, git blob, sha256, size,
mode), and `prefix` -- the repo dir the arcnames are relative to (old repo path = prefix + "/" + arcname).

    python tools/packed_path.py "<old repo path>"            # which tar holds it, under which member name
    python tools/packed_path.py "<old repo path>" --extract DIR
    python tools/packed_path.py --verify                      # every tar vs its sidecar and the registry
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = "tools/packed_paths.json"
SCHEMA = "tanitad.packed_paths/1"


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_registry(repo: Path = REPO) -> dict:
    reg = json.loads((repo / REGISTRY).read_text(encoding="utf-8"))
    if reg.get("schema") != SCHEMA:
        raise ValueError(f"{REGISTRY}: schema {reg.get('schema')!r} is not {SCHEMA!r}")
    return reg


def read_sidecar(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        arc, blob, sha, size, mode = line.split("\t")
        if arc in out:
            raise ValueError(f"duplicate member {arc!r} in sidecar")
        out[arc] = {"git_blob": blob, "sha256": sha, "size": int(size), "mode": mode}
    return out


def resolve(old_path: str, reg: dict) -> tuple[dict, str] | None:
    """-> (pack, arcname) when `old_path` was packed, else None."""
    p = old_path.replace("\\", "/").strip("/")
    for pack in reg["packs"]:
        pre = pack["prefix"].rstrip("/") + "/"
        if p.startswith(pre):
            arc = p[len(pre):]
            if arc in set(pack["members"]):
                return pack, arc
    return None


def verify(repo: Path = REPO) -> list[str]:
    """Every problem found (empty = all packs intact)."""
    reg = load_registry(repo)
    probs: list[str] = []
    for pack in reg["packs"]:
        tp, sp = repo / pack["tar"], repo / pack["sidecar"]
        if not tp.is_file() or not sp.is_file():
            probs.append(f"{pack['tar']}: tar or sidecar missing")
            continue
        raw = tp.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pack["tar_sha256"]:
            probs.append(f"{pack['tar']}: tar sha256 differs from the registry")
        side = read_sidecar(sp.read_text(encoding="utf-8"))
        if sorted(side) != sorted(pack["members"]):
            probs.append(f"{pack['tar']}: sidecar members differ from the registry")
        seen = set()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    probs.append(f"{pack['tar']}: non-file member {m.name!r}")
                    continue
                data = tf.extractfile(m).read()
                want = side.get(m.name)
                if want is None:
                    probs.append(f"{pack['tar']}: member {m.name!r} not in the sidecar")
                    continue
                seen.add(m.name)
                if git_blob_id(data) != want["git_blob"] or hashlib.sha256(data).hexdigest() != want["sha256"]:
                    probs.append(f"{pack['tar']}: member {m.name!r} content differs from its sidecar")
        missing = sorted(set(side) - seen)
        if missing:
            probs.append(f"{pack['tar']}: {len(missing)} sidecar member(s) absent from the tar, e.g. {missing[0]!r}")
    return probs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--extract", metavar="DIR")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    if a.verify:
        probs = verify(repo)
        reg = load_registry(repo)
        n = sum(len(p["members"]) for p in reg["packs"])
        for x in probs:
            print("PROBLEM", x)
        print(f"{'OK' if not probs else 'FAIL'}: {len(reg['packs'])} packs, {n} members, {len(probs)} problem(s)")
        return 0 if not probs else 1
    if not a.path:
        ap.error("give a path, or --verify")
    hit = resolve(a.path, load_registry(repo))
    if hit is None:
        print(f"not packed: {a.path}")
        return 2
    pack, arc = hit
    print(f"tar: {pack['tar']}\nmember: {arc}")
    if a.extract:
        with tarfile.open(repo / pack["tar"], mode="r:") as tf:
            data = tf.extractfile(tf.getmember(arc)).read()
        dest = Path(a.extract) / Path(arc).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"extracted -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
