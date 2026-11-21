"""MASTER-MIND-RUN: seal Operational Amendment 1.  Nothing is hand-written: this uses the package's own hash and check (pgc_seal.sha_file / check_seal) and rolls back on any failure.

    python -B ops1/seal_amendment_ops1.py

Refuses unless: PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md reads STATUS DRAFT; its .sha256 does not exist; the sealed prereg / SEALED_AT.txt are untouched (hash re-checked); every `to` in its OPS_REPIN block equals the file on disk.
Then: flips `**STATUS: DRAFT` -> `**STATUS: SEALED` and `Sealed at: NOT SEALED` -> the time, writes PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256 (`<sha256>  <name>`, the format of the prereg's own), runs
pgc_seal.check_seal(inputs=True) with the read's own requirements, and restores the DRAFT and deletes the .sha256 if that raises.  Label-free: hashes files, opens no label, no array, no scalar store.
"""
import datetime
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{ROOT}/code")
import pgc_run                                                                           # noqa: E402
import pgc_seal as S                                                                     # noqa: E402

NAME = "PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1"
MD, SHA = f"{ROOT}/{NAME}.md", f"{ROOT}/{NAME}.sha256"


def main() -> int:
    if os.path.exists(SHA):
        print(f"REFUSED: {NAME}.sha256 already exists (already sealed)")
        return 2
    raw = open(MD, "rb").read()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if S.status_of(text) != "DRAFT" or text.count("**STATUS: DRAFT") != 1 or text.count("Sealed at: NOT SEALED") != 1:
        print("REFUSED: the amendment is not an unsealed DRAFT")
        return 2
    if S.sha_file(f"{ROOT}/PREREG_PRIVGEOM_CLEAN.md") != "148ebf0f2162b43ba7a1c29243fceac5900fe6b88ace3cae04fc27bdf29c6cf9":
        print("REFUSED: the sealed prereg does not hash to its sealed sha256")
        return 2
    for name, e in S.repin_block(text, "Operational amendment 1").items():
        live = S.sha_file(f"{ROOT}/{name}")
        if live != e["to"]:
            print(f"REFUSED: {name} on disk is {live[:12]}, the amendment says {e['to'][:12]}")
            return 2
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    sealed = text.replace("**STATUS: DRAFT", "**STATUS: SEALED", 1).replace("Sealed at: NOT SEALED", f"Sealed at: {now}", 1)
    try:
        with open(MD, "w", encoding="utf-8", newline="\n") as f:
            f.write(sealed)
        with open(SHA, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{S.sha_file(MD)}  {NAME}.md\n")
        info = S.check_seal(ROOT, required=pgc_run.SEAL_REQUIRED, inputs=True)
    except BaseException as e:                                                           # noqa: BLE001
        with open(MD, "wb") as f:
            f.write(raw)
        if os.path.exists(SHA):
            os.remove(SHA)
        print(f"ROLLED BACK: {type(e).__name__}: {e}")
        return 3
    print("SEALED", json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
