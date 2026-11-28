"""Enumerate every repo under Sayood/ on HF and dump the file tree with sizes + sha.

Never prints the token. Reads it in place from Keys.txt.
"""
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()

from huggingface_hub import HfApi

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"


def token():
    with open(KEYS, "r", encoding="utf-8", errors="replace") as fh:
        m = re.search(r"hf_[A-Za-z0-9]+", fh.read())
    if not m:
        raise SystemExit("no hf token found")
    return m.group(0)


def main():
    tok = token()
    api = HfApi(token=tok)
    out = {"models": {}, "datasets": {}}

    for kind, lister in (("models", api.list_models), ("datasets", api.list_datasets)):
        try:
            repos = list(lister(author="Sayood"))
        except Exception as exc:  # noqa: BLE001
            out[kind]["_ERROR"] = f"{type(exc).__name__}: {exc}"
            continue
        for r in repos:
            rid = r.id
            entry = {"private": getattr(r, "private", None), "files": None, "error": None}
            try:
                infos = api.list_repo_tree(
                    rid,
                    repo_type=kind[:-1],
                    recursive=True,
                    expand=True,
                    token=tok,
                )
                files = []
                for f in infos:
                    if getattr(f, "size", None) is None:
                        continue  # directory
                    lfs = getattr(f, "lfs", None)
                    files.append(
                        {
                            "path": f.path,
                            "size": f.size,
                            "sha256": getattr(lfs, "sha256", None) if lfs else None,
                            "blob_id": getattr(f, "blob_id", None),
                        }
                    )
                entry["files"] = files
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
            out[kind][rid] = entry

    dest = sys.argv[1]
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    for kind in ("models", "datasets"):
        for rid, e in sorted(out[kind].items()):
            if rid == "_ERROR":
                print(f"[{kind}] LIST ERROR: {e}")
                continue
            if e["error"]:
                print(f"[{kind}] {rid}: TREE ERROR {e['error']}")
            else:
                n = len(e["files"])
                tot = sum(f["size"] for f in e["files"])
                print(f"[{kind}] {rid}: {n} files, {tot/1e9:.2f} GB, private={e['private']}")


if __name__ == "__main__":
    main()
