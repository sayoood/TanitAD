"""PRIVGEOM-CLEAN seal: write and verify the seal of the pre-registration.  LABEL-FREE: this module opens no label, no array, no scalar store (it hashes files and reads two small text files).

    python pgc_seal.py seal  --sealed-by NAME     # ONE-TIME.  Validates everything below, then writes PREREG_PRIVGEOM_CLEAN.sha256 + SEALED_AT.txt.  REFUSES if either already exists
    python pgc_seal.py check [--inputs]           # exit 0 only if the seal holds (prints a JSON summary); --inputs also hashes the label-free input files

What the seal is (every point is a REFUSAL; each has a RED arm in test_pgc_seal.py and in pgc_seal_proofs.py):
  1. The prereg hashes to the sha256 recorded in PREREG_PRIVGEOM_CLEAN.sha256            (a ONE-BYTE edit is refused).
  2. The prereg's own STATUS line reads SEALED                                           (a DRAFT-status copy is refused EVEN WHEN it hashes to its own .sha256).
  3. SEALED_AT.txt (JSON) exists, records the same prereg sha256, and records the sha256 of EVERY code file the read uses; each must equal the file on disk NOW
     (an edit to any pinned file after the seal is refused; the set must cover everything the caller requires).
  4. Every row of the prereg's section-13 pins table equals the sealed pin of that file, and the table covers every sealed pin (the table cannot go stale or omit a file).
  5. The Revision-1 block of the prereg is the MM ruling VERBATIM (de-quoted, it equals MM_RULING_PGCLEAN.md) and that file hashes to its sealed sha256.
  6. Amendments (PREREG_PRIVGEOM_CLEAN_AMENDMENT<k>.md + .sha256) are each verified the same way as the prereg's hash.
  7. With --inputs: the label-free input files (census meta, token bank, coverage, plant calibration, G-LIVE registration) hash to their sealed sha256.
Hashes are over RAW BYTES (every pinned file is LF-only, so the staged repo copy hashes the same).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SCHEMA = "privgeom-clean-seal-v1"
PREREG_NAME = "PREREG_PRIVGEOM_CLEAN.md"
RULING_NAME = "MM_RULING_PGCLEAN.md"
SEALED_AT_NAME = "SEALED_AT.txt"
BEGIN, END = "<!-- MM_RULING_VERBATIM_BEGIN -->", "<!-- MM_RULING_VERBATIM_END -->"
STATUS_RE = re.compile(r"^\*\*STATUS:\s*([A-Za-z][A-Za-z-]*)", re.M)
TABLE_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*`([0-9a-f]{64})`\s*\|\s*$", re.M)
SELF_DERIVED = ("code/PINS.json",)                 # derived from the pins by pgc_pins.py, so not a row of the prereg table


class SealError(RuntimeError):
    pass


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_text(path: str) -> str:
    return open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def status_of(text: str):
    """The word after `**STATUS:` on the first such line within the first 40 lines (None if absent)."""
    m = STATUS_RE.search("\n".join(text.splitlines()[:40]))
    return m.group(1).upper() if m else None


def ruling_block(text: str):
    """The de-quoted text between the two markers of Revision 1 (None if the markers are absent or a line is not quoted)."""
    a, b = text.find(BEGIN), text.find(END)
    if a < 0 or b < 0 or b < a:
        return None
    out = []
    for ln in text[a + len(BEGIN):b].strip("\n").splitlines():
        out.append(ln[2:] if ln.startswith("> ") else ("" if ln == ">" else None))
    return None if any(x is None for x in out) else "\n".join(out)


def table_pins(text: str) -> dict:
    return {m.group(1): m.group(2) for m in TABLE_RE.finditer(text)}


def _rel(name: str) -> str:
    """a prereg-table name -> the root-relative path of the pinned file."""
    return name if name.startswith("ref/") else f"code/{name}"


def _verbatim_ok(text: str, ruling_path: str) -> bool:
    blk = ruling_block(text)
    return blk is not None and blk.strip("\n") == read_text(ruling_path).strip("\n")


def check_seal(root: str = ROOT, *, prereg_name: str = PREREG_NAME, required=(), inputs: bool = False) -> dict:
    """Raise SealError unless the seal holds.  Reads only the prereg, its .sha256, SEALED_AT.txt, the ruling, the pinned files (and the inputs with inputs=True)."""
    prereg = f"{root}/{prereg_name}"
    shaf, sealed_at, ruling = prereg[:-3] + ".sha256", f"{root}/{SEALED_AT_NAME}", f"{root}/{RULING_NAME}"
    for p in (prereg, shaf, sealed_at):
        if not os.path.exists(p):
            raise SealError(f"{os.path.basename(p)} is missing: the pre-registration is NOT SEALED")
    rec = open(shaf, encoding="utf-8").read().split()[0]
    got = sha_file(prereg)
    if got != rec:
        raise SealError(f"PREREG does not hash to its recorded sha256 {rec} (got {got}): every stage refuses")
    text = read_text(prereg)
    st = status_of(text)
    if st != "SEALED":
        raise SealError(f"PREREG status is {st!r}, not SEALED: a draft-status copy is refused (even if it hashes to its own .sha256)")
    try:
        S = json.load(open(sealed_at, encoding="utf-8"))
    except Exception as e:                                                               # noqa: BLE001
        raise SealError(f"{SEALED_AT_NAME} is unreadable: {e}") from None
    if S.get("schema") != SCHEMA:
        raise SealError(f"{SEALED_AT_NAME}: schema {S.get('schema')!r} != {SCHEMA!r}")
    if S.get("prereg_sha256") != rec:
        raise SealError(f"{SEALED_AT_NAME} records prereg {S.get('prereg_sha256')} but {os.path.basename(shaf)} records {rec}")
    pins = S.get("code_sha256")
    if not isinstance(pins, dict) or not pins:
        raise SealError(f"{SEALED_AT_NAME}: no code pins")
    miss = [n for n in required if n not in pins]
    if miss:
        raise SealError(f"{SEALED_AT_NAME} does not pin file(s) the read uses: {miss}")
    bad = [n for n, h in sorted(pins.items()) if not os.path.exists(f"{root}/{n}") or sha_file(f"{root}/{n}") != h]
    if bad:
        raise SealError(f"pinned file(s) differ from the seal (edited after sealing?): {bad}")
    tab = table_pins(text)                                                               # the section-13 table must be the sealed pins, in both directions
    tab_bad = [n for n, h in tab.items() if pins.get(_rel(n)) != h]
    tab_missing = [n for n in pins if n not in SELF_DERIVED and not any(_rel(t) == n for t in tab)]
    if tab_bad or tab_missing:
        raise SealError(f"the prereg's section-13 pins table disagrees with the seal: wrong {tab_bad}, absent {tab_missing}")
    if not os.path.exists(ruling) or sha_file(ruling) != S.get("ruling_sha256"):
        raise SealError("MM_RULING_PGCLEAN.md is missing or differs from its sealed sha256")
    if not _verbatim_ok(text, ruling):
        raise SealError("Revision 1 is not the MM ruling verbatim (the de-quoted block differs from MM_RULING_PGCLEAN.md)")
    amend, k = {}, 1
    while os.path.exists(f"{root}/{prereg_name[:-3]}_AMENDMENT{k}.md"):
        a = f"{root}/{prereg_name[:-3]}_AMENDMENT{k}.md"
        ash = a[:-3] + ".sha256"
        if not os.path.exists(ash) or sha_file(a) != open(ash, encoding="utf-8").read().split()[0]:
            raise SealError(f"Amendment {k} does not hash to its recorded sha256")
        amend[f"amend{k}"] = open(ash, encoding="utf-8").read().split()[0]
        k += 1
    if inputs:
        ins = S.get("inputs") or {}
        bad_in = [key for key, r in ins.items() if not os.path.exists(r["path"] if os.path.isabs(r["path"]) else f"{root}/{r['path']}")
                  or sha_file(r["path"] if os.path.isabs(r["path"]) else f"{root}/{r['path']}") != r["sha256"]]
        if bad_in or not ins:
            raise SealError(f"input file(s) differ from the seal (or none sealed): {bad_in}")
    return {"pins": {"prereg": rec, **amend}, "sealed_at": S.get("sealed_at"), "revision": S.get("revision"), "n_code_pins": len(pins), "status": st, "inputs_checked": bool(inputs)}


def seal(root: str = ROOT, *, sealed_by: str, inputs: dict, required=(), prereg_name: str = PREREG_NAME) -> dict:
    """ONE-TIME.  Validates in memory, writes last.  `inputs` = {key: path}."""
    prereg = f"{root}/{prereg_name}"
    shaf, sealed_at, ruling = prereg[:-3] + ".sha256", f"{root}/{SEALED_AT_NAME}", f"{root}/{RULING_NAME}"
    if os.path.exists(shaf) or os.path.exists(sealed_at):
        raise SealError("already sealed: a change after the seal is an AMENDMENT, never a re-seal")
    text = read_text(prereg)
    if status_of(text) != "SEALED":
        raise SealError(f"the prereg's STATUS line must read SEALED before it is hashed (reads {status_of(text)!r})")
    pj = json.load(open(f"{root}/code/PINS.json", encoding="utf-8"))
    if not pj.get("frozen_ok"):
        raise SealError("PINS.json: a FROZEN file differs from PRIVGEOM-SAFE's pin")
    live = {_rel(n): sha_file(f"{root}/{_rel(n)}") for n in pj["sha256"]}
    live["code/PINS.json"] = sha_file(f"{root}/code/PINS.json")
    stale = [n for n in pj["sha256"] if live[_rel(n)] != pj["sha256"][n]]
    if stale:
        raise SealError(f"PINS.json is stale (regenerate with pgc_pins.py): {stale}")
    miss = [n for n in required if n not in live]
    if miss:
        raise SealError(f"the caller requires pins that PINS.json lacks: {miss}")
    if not _verbatim_ok(text, ruling):
        raise SealError("Revision 1 is not the MM ruling verbatim")
    tab = table_pins(text)
    bad = [n for n, h in tab.items() if live.get(_rel(n)) != h]
    absent = [n for n in live if n not in SELF_DERIVED and not any(_rel(t) == n for t in tab)]
    if bad or absent:
        raise SealError(f"the prereg's pins table is wrong {bad} / absent {absent}")
    rec = sha_file(prereg)
    S = {"schema": SCHEMA, "sealed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "sealed_by": sealed_by, "revision": 1,
         "prereg": prereg_name, "prereg_sha256": rec, "ruling": RULING_NAME, "ruling_sha256": sha_file(ruling), "primary_arm": "LGc", "code_sha256": live,
         "inputs": {key: {"path": p, "sha256": sha_file(p), "bytes": os.path.getsize(p)} for key, p in inputs.items()},
         "note": "hashes are over RAW BYTES (every pinned file is LF-only).  A change after the seal is an Amendment (PREREG_PRIVGEOM_CLEAN_AMENDMENT<k>.md + .sha256), never a re-seal."}
    with open(sealed_at, "w", encoding="utf-8", newline="\n") as f:
        json.dump(S, f, indent=1)
    with open(shaf, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{rec}  {prereg_name}\n")
    info = check_seal(root, prereg_name=prereg_name, required=required, inputs=True)
    return {"sealed_sha256": rec, "check": info}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("seal", "check"))
    ap.add_argument("--sealed-by", default="")
    ap.add_argument("--inputs", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            print(json.dumps(check_seal(ROOT, inputs=a.inputs), indent=1))
            return 0
        sys.path.insert(0, HERE)
        import pgc_run                                                                   # noqa: E402  (the caller's own requirements and input list)
        print(json.dumps(seal(ROOT, sealed_by=a.sealed_by, inputs=pgc_run.SEAL_INPUTS, required=pgc_run.SEAL_REQUIRED), indent=1))
        return 0
    except SealError as e:
        print(f"SEAL REFUSED: {e}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
