#!/usr/bin/env python3
"""P5-C read instrument B11: SEALED.json (read order, card section 11) and the K-ledger entry writer / validator.

Card section 11, read order:  1. both arms' `model_final.pt` sha256s are written to `SG/p5c/read/SEALED.json` BEFORE either navtest pass starts;
2. both arms are scored in one session;  3. no per-arm number is printed before both are complete.  One ledger entry per arm read:
  {"key": "P5C:<arm>:<sha256 of model_final.pt>:E3", "n": <next free>, "time": <UTC at read>, "nominee": <arm>, "family": "P5C", "set": "E3",
   "smoke": false, "scorers": ["lqr_tracked", "tracker_transparent"], "card_sha256": ..., "base_sha256": ...}
C0r only if card 6.3 triggers AND the PI approves. `n` is the next free value at read time. THIS TOOL NEVER WRITES THE REAL LEDGER: it writes the ENTRIES
file and a merged COPY at a path that must differ from the ledger; the Master Mind appends at read time.

Subcommands
  seal           --arm C1=<model_final.pt> --arm C0=<model_final.pt> [--arm C0r=...] --card PREREG_P5C.md --base-sha <hex> --out SEALED.json
                   sha256 of each file is COMPUTED here (a typed hash is accepted only as --arm-sha NAME=<hex> and is then marked hash_source "declared")
  verify         --sealed SEALED.json [--card PREREG_P5C.md] [--recheck-files] [--rows-glob G ...] --out result.json
                   self-hash, schema, C1 and C0 present, 64-hex, distinct from each other and from the base, card sha, optional file re-hash,
                   optional "the rows were written AFTER the seal" (file mtime >= sealed_utc). Boolean `pass`; kind `B11_sealed`.
  validate-ledger --ledger ledger_K.json            n = 1..K in order, keys unique, required fields, time non-decreasing
  entries        --sealed S --ledger L --out entries.json [--time UTC] [--c0r-approved]
  check          --ledger L --entries E [--sealed S] --out result.json    the ledger plus the entries must still validate, entries must equal the sealed arms
  merge          --ledger L --entries E --out MERGED.json                 (never in place; the MM copies it over the ledger at read time)
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import sys

KIND_SEALED = "B11_sealed"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCORERS = ["lqr_tracked", "tracker_transparent"]
REQUIRED_ARMS = ("C1", "C0")
OPTIONAL_ARMS = ("C0r",)
LEDGER_REQ = ("key", "n", "time", "family", "set", "smoke")
ENTRY_FIELDS = ("key", "n", "time", "nominee", "family", "set", "smoke", "scorers", "card_sha256", "base_sha256")
TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(TIME_FMT)


def parse_utc(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, TIME_FMT).replace(tzinfo=dt.timezone.utc)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(o) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def self_hash(doc: dict) -> str:
    d = {k: v for k, v in doc.items() if k != "sealed_sha256"}
    return hashlib.sha256(canon(d)).hexdigest()


# ------------------------------------------------------------------------------------------------------ SEALED.json
def build_sealed(arms: dict, card_sha256: str, base_sha256: str, sealed_utc: str | None = None) -> dict:
    """arms: {name: {"model_final_sha256", "hash_source", ["path", "size_bytes"]}}"""
    doc = {"kind": "P5C_SEALED", "schema": 1, "card": "PREREG_P5C.md", "card_sha256": card_sha256, "base_sha256": base_sha256,
           "set": "E3", "scorers": SCORERS, "sealed_utc": sealed_utc or utc_now(), "arms": arms,
           "rule": "card section 11: both model_final.pt sha256s are written here before either navtest pass starts; the read refuses a SEALED.json that fails `verify`"}
    doc["sealed_sha256"] = self_hash(doc)
    return doc


def verify_sealed(doc: dict, card_path: str | None = None, recheck_files: bool = False, rows_paths=None) -> dict:
    bad = []
    if doc.get("kind") != "P5C_SEALED" or doc.get("schema") != 1:
        bad.append("not a P5C_SEALED schema-1 document")
    if doc.get("sealed_sha256") != self_hash(doc):
        bad.append("sealed_sha256 does not match the content: the file was edited after sealing")
    arms = doc.get("arms") or {}
    for a in REQUIRED_ARMS:
        if a not in arms:
            bad.append(f"arm {a} is not sealed (both arms are sealed in ONE manifest before either read)")
    unknown = [a for a in arms if a not in REQUIRED_ARMS + OPTIONAL_ARMS]
    if unknown:
        bad.append(f"unknown arms {unknown}")
    shas = {}
    for a, e in arms.items():
        s = (e or {}).get("model_final_sha256", "")
        if not HEX64.match(str(s)):
            bad.append(f"arm {a}: model_final_sha256 is not 64 lower-case hex")
        shas[a] = s
    if len(set(shas.values())) != len(shas):
        bad.append("two arms carry the same model_final_sha256 (a sealing error: C1 and C0 are different trainings)")
    base = doc.get("base_sha256", "")
    if not HEX64.match(str(base)):
        bad.append("base_sha256 is not 64 lower-case hex")
    elif base in shas.values():
        bad.append("an arm's model_final_sha256 equals the base's: that arm did not move off the base")
    if not HEX64.match(str(doc.get("card_sha256", ""))):
        bad.append("card_sha256 is not 64 lower-case hex")
    if card_path:
        if sha256_file(card_path) != doc.get("card_sha256"):
            bad.append("card_sha256 differs from the card file on disk")
    try:
        t_seal = parse_utc(doc.get("sealed_utc", ""))
        if t_seal > dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5):
            bad.append("sealed_utc is in the future")
    except ValueError:
        t_seal = None
        bad.append("sealed_utc is not UTC `%Y-%m-%dT%H:%M:%SZ`")
    if recheck_files:
        for a, e in arms.items():
            p = (e or {}).get("path")
            if e.get("hash_source") != "file":
                bad.append(f"arm {a}: hash was declared, not computed from a file; cannot recheck")
            elif not p or not os.path.exists(p):
                bad.append(f"arm {a}: file {p} not found for the recheck")
            elif sha256_file(p) != e["model_final_sha256"]:
                bad.append(f"arm {a}: the file on disk no longer hashes to the sealed sha256")
    stale = []
    if rows_paths and t_seal is not None:
        for p in rows_paths:
            if os.path.exists(p):
                m = dt.datetime.fromtimestamp(os.path.getmtime(p), dt.timezone.utc)
                if m < t_seal:
                    stale.append(p)
        if stale:
            bad.append(f"{len(stale)} score-row files are OLDER than the seal (first {stale[0]}): the read started before the arms were sealed")
    ok = not bad
    return {"control": KIND_SEALED, "instrument": "p5c_seal.py B11", "card": "PREREG_P5C.md section 11 (read order)", "sealed_sha256": doc.get("sealed_sha256"),
            "arms": {a: shas[a] for a in shas}, "problems": bad, "pass": bool(ok), "verdict": "PASS" if ok else "FAIL"}


# ------------------------------------------------------------------------------------------------------ the K ledger
def validate_ledger(doc: dict) -> dict:
    bad = []
    reads = doc.get("reads")
    if not isinstance(reads, list):
        return {"pass": False, "K": None, "problems": ["no `reads` list"], "verdict": "FAIL"}
    seen, last_t = set(), None
    for i, r in enumerate(reads):
        miss = [k for k in LEDGER_REQ if k not in r]
        if miss:
            bad.append(f"entry {i + 1}: missing {miss}")
            continue
        if r["n"] != i + 1:
            bad.append(f"entry {i + 1}: n = {r['n']} (n must be 1..K in order, no gap, no repeat, no reorder)")
        if r["key"] in seen:
            bad.append(f"entry {i + 1}: duplicate key {r['key']}")
        seen.add(r["key"])
        try:
            t = parse_utc(r["time"])
            if last_t is not None and t < last_t:
                bad.append(f"entry {i + 1}: time {r['time']} is earlier than the previous entry's (out of order)")
            last_t = t
        except ValueError:
            bad.append(f"entry {i + 1}: time {r['time']!r} is not UTC `%Y-%m-%dT%H:%M:%SZ`")
    return {"pass": not bad, "K": len(reads), "problems": bad, "verdict": "PASS" if not bad else "FAIL"}


def make_entries(sealed: dict, ledger: dict, time: str | None = None, c0r_approved: bool = False) -> list:
    v = verify_sealed(sealed)
    if not v["pass"]:
        raise ValueError("SEALED.json fails verify: " + "; ".join(v["problems"]))
    lv = validate_ledger(ledger)
    if not lv["pass"]:
        raise ValueError("the ledger fails validation: " + "; ".join(lv["problems"]))
    arms = [a for a in REQUIRED_ARMS + OPTIONAL_ARMS if a in sealed["arms"]]
    if "C0r" in arms and not c0r_approved:
        raise ValueError("C0r is sealed but --c0r-approved was not given (card 6.3: only if the clause triggers AND the PI approves)")
    t = time or utc_now()
    if lv["K"]:
        last = parse_utc(ledger["reads"][-1]["time"])
        if parse_utc(t) < last:
            raise ValueError(f"entry time {t} is earlier than the ledger's last entry {ledger['reads'][-1]['time']}")
    have = {r["key"] for r in ledger["reads"]}
    out = []
    n = lv["K"]
    for a in arms:
        n += 1
        key = f"P5C:{a}:{sealed['arms'][a]['model_final_sha256']}:E3"
        if key in have:
            raise ValueError(f"duplicate: {key} is already in the ledger (one entry per arm read)")
        out.append({"key": key, "n": n, "time": t, "nominee": a, "family": "P5C", "set": "E3", "smoke": False, "scorers": list(SCORERS),
                    "card_sha256": sealed["card_sha256"], "base_sha256": sealed["base_sha256"]})
    return out


def check_entries(ledger: dict, entries: list, sealed: dict | None = None) -> dict:
    bad = []
    merged = copy.deepcopy(ledger)
    merged["reads"] = list(merged.get("reads", [])) + copy.deepcopy(entries)
    lv = validate_ledger(merged)
    bad += lv["problems"]
    K0 = len(ledger.get("reads", []))
    for j, e in enumerate(entries):
        miss = [k for k in ENTRY_FIELDS if k not in e]
        if miss:
            bad.append(f"new entry {j + 1}: missing {miss}")
            continue
        if e["n"] != K0 + j + 1:
            bad.append(f"new entry {j + 1}: n = {e['n']}, the next free value is {K0 + j + 1}")
        if e["family"] != "P5C" or e["set"] != "E3" or e["smoke"] is not False or e["scorers"] != SCORERS:
            bad.append(f"new entry {j + 1}: family / set / smoke / scorers differ from the card's template")
        m = re.match(r"^P5C:(C1|C0|C0r):([0-9a-f]{64}):E3$", e["key"])
        if not m or m.group(1) != e["nominee"]:
            bad.append(f"new entry {j + 1}: key {e['key']!r} is not P5C:<nominee>:<sha256>:E3")
        elif sealed is not None:
            s = (sealed.get("arms") or {}).get(e["nominee"], {}).get("model_final_sha256")
            if s != m.group(2):
                bad.append(f"new entry {j + 1}: key sha256 is not the SEALED sha256 of {e['nominee']}")
            if e["card_sha256"] != sealed.get("card_sha256") or e["base_sha256"] != sealed.get("base_sha256"):
                bad.append(f"new entry {j + 1}: card_sha256 / base_sha256 differ from SEALED.json")
    if sealed is not None:
        want = {a for a in sealed.get("arms", {}) if a != "C0r"}
        got = {e.get("nominee") for e in entries}
        if not want <= got:
            bad.append(f"arms sealed but with no ledger entry: {sorted(want - got)} (one entry per arm read)")
    return {"control": "B11_ledger", "pass": not bad, "K_before": K0, "K_after": K0 + len(entries), "problems": bad, "verdict": "PASS" if not bad else "FAIL"}


# ------------------------------------------------------------------------------------------------------ CLI
def _write(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


def cmd_seal(a) -> int:
    arms = {}
    for s in a.arm or []:
        n, p = s.split("=", 1)
        if not os.path.isfile(p):
            print(f"REFUSED: {p} is not a file", file=sys.stderr)
            return 2
        arms[n] = {"model_final_sha256": sha256_file(p), "hash_source": "file", "path": os.path.abspath(p), "size_bytes": os.path.getsize(p)}
    for s in a.arm_sha or []:
        n, h = s.split("=", 1)
        arms[n] = {"model_final_sha256": h.lower(), "hash_source": "declared"}
    doc = build_sealed(arms, sha256_file(a.card), a.base_sha.lower())
    v = verify_sealed(doc, a.card)
    if not v["pass"]:
        print("REFUSED:", "; ".join(v["problems"]), file=sys.stderr)
        return 2
    if os.path.exists(a.out):
        print(f"REFUSED: {a.out} exists (a seal is written once)", file=sys.stderr)
        return 2
    _write(a.out, doc)
    print("ZZP5C_SEALED", {k: v_["model_final_sha256"][:12] for k, v_ in arms.items()}, doc["sealed_sha256"][:12])
    return 0


def cmd_verify(a) -> int:
    rows = [p for g in (a.rows_glob or []) for p in __import__("glob").glob(g)]
    res = verify_sealed(json.load(open(a.sealed, encoding="utf-8")), a.card, a.recheck_files, rows or None)
    _write(a.out, res)
    print("ZZP5C_SEALED_VERIFY", res["verdict"], res["problems"][:2])
    return 0 if res["pass"] else 1


def cmd_validate_ledger(a) -> int:
    r = validate_ledger(json.load(open(a.ledger, encoding="utf-8")))
    print("ZZP5C_LEDGER", r["verdict"], "K", r["K"], r["problems"][:3])
    return 0 if r["pass"] else 1


def cmd_entries(a) -> int:
    try:
        ents = make_entries(json.load(open(a.sealed, encoding="utf-8")), json.load(open(a.ledger, encoding="utf-8")), a.time, a.c0r_approved)
    except ValueError as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    _write(a.out, ents)
    print("ZZP5C_ENTRIES", [(e["nominee"], e["n"]) for e in ents])
    return 0


def cmd_check(a) -> int:
    sealed = json.load(open(a.sealed, encoding="utf-8")) if a.sealed else None
    r = check_entries(json.load(open(a.ledger, encoding="utf-8")), json.load(open(a.entries, encoding="utf-8")), sealed)
    _write(a.out, r)
    print("ZZP5C_LEDGER_CHECK", r["verdict"], r["problems"][:3])
    return 0 if r["pass"] else 1


def cmd_merge(a) -> int:
    if os.path.abspath(a.out) == os.path.abspath(a.ledger):
        print("REFUSED: --out is the ledger itself; the Master Mind appends the real ledger", file=sys.stderr)
        return 2
    led = json.load(open(a.ledger, encoding="utf-8"))
    ents = json.load(open(a.entries, encoding="utf-8"))
    r = check_entries(led, ents)
    if not r["pass"]:
        print("REFUSED:", "; ".join(r["problems"]), file=sys.stderr)
        return 2
    led["reads"] = led["reads"] + ents
    _write(a.out, led)
    print("ZZP5C_MERGED K", len(led["reads"]), "->", a.out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("seal")
    s.add_argument("--arm", action="append")
    s.add_argument("--arm-sha", action="append")
    s.add_argument("--card", required=True)
    s.add_argument("--base-sha", required=True)
    s.add_argument("--out", required=True)
    v = sp.add_parser("verify")
    v.add_argument("--sealed", required=True)
    v.add_argument("--card", default=None)
    v.add_argument("--recheck-files", action="store_true")
    v.add_argument("--rows-glob", nargs="*", default=None)
    v.add_argument("--out", required=True)
    vl = sp.add_parser("validate-ledger")
    vl.add_argument("--ledger", required=True)
    e = sp.add_parser("entries")
    e.add_argument("--sealed", required=True)
    e.add_argument("--ledger", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--time", default=None)
    e.add_argument("--c0r-approved", action="store_true")
    c = sp.add_parser("check")
    c.add_argument("--ledger", required=True)
    c.add_argument("--entries", required=True)
    c.add_argument("--sealed", default=None)
    c.add_argument("--out", required=True)
    m = sp.add_parser("merge")
    m.add_argument("--ledger", required=True)
    m.add_argument("--entries", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"seal": cmd_seal, "verify": cmd_verify, "validate-ledger": cmd_validate_ledger, "entries": cmd_entries, "check": cmd_check, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
