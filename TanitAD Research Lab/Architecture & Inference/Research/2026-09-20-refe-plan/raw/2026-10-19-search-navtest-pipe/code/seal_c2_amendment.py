#!/usr/bin/env python3
"""Seal AMENDMENT C2 (MM_RULING_C2AMEND.md DQ9).  The Master Mind runs this ONCE, when the landed code is final AND the fixture record reads IDENTICAL (DQ8); nothing here chooses a term or a bar.

    python seal_c2_amendment.py --src AMENDMENT_C2_DRAFT.md --ruling MM_RULING_C2AMEND.md --pkg <dir for the sealed files> --fixture-record <FIXTURE_RECORD.json> [--check-only]
        [--code-dir D] [--dose-chain F] [--banked-seam F] [--banked-csv F] [--token-file F] [--ckpt F] [--head F] [--harness F]

Modelled on `seal_spc.py` (C1's sealer).  It
  (1) REFUSES a source that is not a FINAL DRAFT (`Status: **DRAFT`), that still carries a `@@` placeholder outside the PINS block, or whose PINS block carries a `@@SEAL@@` placeholder this tool does not know how to fill
      (a pin left as a placeholder must refuse: that is the RED arm of DQ9);
  (2) FILLS the `@@SEAL@@` pins from the files on disk: the sha256 of the banked seam and the banked CSV (bytes only, no label is parsed), of the dose package's `dose_chain.py` (LF) and of the fixture record, and the LF sha256 of
      every operative code file and of this tool; a pin the draft already carries for a computed key must EQUAL the file (else it refuses), and every pin that names a file is checked against that file;
  (3) checks the ruling file is `MM_RULING_C2AMEND.md` (sha256 fe4d9ce6...), that the draft pins it, and that the FIXTURE RECORD reads IDENTICAL with code shas equal to the pins it is about to write (the chain's own check);
  (4) writes the sealed text (LF, `Status: **SEALED**`; the terms text is BYTE-IDENTICAL to the draft's: only the status line and the PINS block change, and the tool checks that), `AMENDMENT_C2.sha256` and `SEALED_AT.txt`;
  (5) runs the chain's OWN amendment seal check (`navpipe_core.check_amendment_seal`) on the written text in a scratch directory first and only then moves the files in; a refusal there leaves nothing behind.
It never overwrites a sealed file: a re-pin is a "Revision n" (DQ8), written by hand with the terms byte-identical.  No GPU, no pod, no navtest label, score or search output is read.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402

SEALED_NAME, SIDECAR_NAME, STAMP_NAME = "AMENDMENT_C2.md", "AMENDMENT_C2.sha256", "SEALED_AT.txt"
PLACEHOLDER = "@@SEAL@@"
SELF = "seal_c2_amendment.py"
CODE_PINS = tuple(NC.OPERATIVE) + (SELF,)
RULED_HEAD = "RULED (`MM_RULING_C2AMEND.md` sha256 `" + NC.C2AMEND_RULING_SHA256[:8]
BEGIN, END = "<!-- PINS-BEGIN -->", "<!-- PINS-END -->"


class SealRefused(RuntimeError):
    pass


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def split_pins(text: str) -> tuple:
    """-> (head, pins block lines, tail); the block is whatever sits between the markers (the ``` fence included)"""
    if BEGIN not in text or END not in text:
        raise SealRefused("the source has no PINS block")
    head, _, rest = text.partition(BEGIN)
    blk, _, tail = rest.partition(END)
    return head, blk, tail


def computed_pins(a) -> dict:
    """the pins this tool computes from files on disk: key -> sha256"""
    out = {"banked_seam_sha256": NC.sha256_file(a.banked_seam), "banked_csv_sha256": NC.sha256_file(a.banked_csv), "dose_chain_sha256": NC.sha_lf(a.dose_chain),
           "fixture_record_sha256": NC.sha256_file(a.fixture_record)}
    for f in CODE_PINS:
        out["code:" + f] = NC.sha_lf(os.path.join(a.code_dir, f))
    return out


def fill_pins(src_text: str, fills: dict) -> tuple:
    """replace every `@@SEAL@@` value in the PINS block by its computed sha; -> (text, the pins dict).  Refuses an unknown placeholder, a missing key, and a pre-filled value that differs from the file."""
    head, blk, tail = split_pins(src_text)
    if "@@" in head or "@@" in tail:
        raise SealRefused("the source still has an unfilled @@ placeholder outside the PINS block")      # @GUARD:S-PLACEHOLDER-OUTSIDE
    lines, seen = [], set()
    for line in blk.split("\n"):
        st = line.strip()
        if not st or st.startswith(("`", "#")):
            lines.append(line)
            continue
        k, _, v = st.partition(" ")
        v = v.strip()
        if v == PLACEHOLDER:
            if k not in fills:                                                                            # @GUARD:S-PLACEHOLDER-UNKNOWN
                raise SealRefused(f"pin {k} is still a {PLACEHOLDER} placeholder and this tool has no file to fill it from")
            line = f"{k} {fills[k]}"
        elif "@@" in v:
            raise SealRefused(f"pin {k} carries a placeholder this tool does not know: {v!r}")
        elif k in fills and v.lower() != fills[k]:                                                         # @GUARD:S-PIN-DISAGREES
            raise SealRefused(f"pin {k} is {v[:12]}... in the draft but its file hashes to {fills[k][:12]}...")
        seen.add(k)
        lines.append(line)
    missing = [k for k in fills if k not in seen]
    if missing:                                                                                           # @GUARD:S-PIN-NOT-IN-DRAFT
        raise SealRefused("the draft's PINS block has no line for: " + ", ".join(missing))
    text = head + BEGIN + "\n".join(lines) + END + tail
    return text, NC.parse_pins(text)


def check_ruling(ruling_path: str, text: str) -> str:
    got = hashlib.sha256(lf(open(ruling_path, "rb").read())).hexdigest()
    if got != NC.C2AMEND_RULING_SHA256:                                                                   # @GUARD:S-RULING-SHA
        raise SealRefused(f"--ruling hashes to {got[:12]}..., MM_RULING_C2AMEND.md is {NC.C2AMEND_RULING_SHA256[:12]}...")
    if RULED_HEAD not in text:                                                                            # @GUARD:S-RULED-MARK
        raise SealRefused("the draft's section 14 is not marked RULED with the ruling's sha256 (fold the ruling in first, brief item 6)")
    return got


def build_sealed(src_text: str, fills: dict, when: str | None = None) -> tuple:
    """-> (the sealed text, its pins).  Only the first Status line and the PINS block change."""
    t = src_text.replace("\r\n", "\n")
    if not re.search(r"^Status: \*\*DRAFT", t, re.M):                                                     # @GUARD:S-STATUS-DRAFT
        raise SealRefused("the source has no 'Status: **DRAFT' line to turn into the seal")
    if "FINAL DRAFT" not in t.split("\n", 1)[0]:                                           # @GUARD:S-TITLE-FINAL
        raise SealRefused("the source is not titled FINAL DRAFT (brief item 6)")
    t2, pins = fill_pins(t, fills)
    when = when or datetime.datetime.now().astimezone().isoformat(timespec="minutes")
    sealed = re.sub(r"^Status: \*\*DRAFT.*$", f"Status: **SEALED** {when} (the sha256 of this text is in `{SIDECAR_NAME}`; `navpipe_chain.py --mode navtest` REFUSES to run without it, without every pin below and without the "
                    f"licence, the config and the fixture record they bind). Terms text byte-identical to the draft; only this line and the PINS block were written by `{SELF}`.)", t2, count=1, flags=re.M)
    sealed = sealed.replace("(FINAL DRAFT)", "(SEALED)", 1)
    return sealed, pins


def terms_identical(draft: str, sealed: str) -> bool:
    """the terms text (everything but the first Status line, the one title word and the PINS block) is byte-identical"""
    def core(x):
        h, _, tl = x.replace("\r\n", "\n").partition(BEGIN)
        _, _, tl = tl.partition(END)
        h = re.sub(r"^Status: .*$", "Status:", h, count=1, flags=re.M)
        return h.replace("(FINAL DRAFT)", "(X)", 1).replace("(SEALED)", "(X)", 1) + tl
    return core(draft) == core(sealed)


def seal(a) -> dict:
    src = open(a.src, encoding="utf-8", newline="").read()
    check_ruling(a.ruling, src)
    fills = computed_pins(a)
    sealed, pins = build_sealed(src, fills)
    if "@@" in sealed:                                                                                    # @GUARD:S-POST-PLACEHOLDER
        raise SealRefused("the sealed text still carries a @@ placeholder")
    if not terms_identical(src, sealed):                                                                  # @GUARD:S-TERMS-IDENTICAL
        raise SealRefused("the sealed text's terms differ from the draft's (only the status line and the PINS block may change)")
    bad = NC.file_pin_mismatches(pins, {"token_file_sha256": a.token_file, "ckpt_sha256": a.ckpt, "head_s0_sha256": a.head, "harness_script_sha256": a.harness,
                                        "seam_module_sha256": os.path.join(os.path.dirname(a.harness), "refe_navtest_seam.py")})
    if bad:                                                                                               # @GUARD:S-FILE-PINS
        raise SealRefused("pins that do not equal their files: " + ", ".join(bad))
    try:
        NC.check_fixture_record(a.fixture_record, pins, NC.CARVE_B_SHA256, NC.OPERATIVE)
    except NC.Refusal as e:
        raise SealRefused(f"the fixture record is refused by the chain's own check: {e.reason} {e.detail}")
    sha = hashlib.sha256(sealed.encode("utf-8")).hexdigest()
    return {"text": sealed, "sha": sha, "pins": pins}


def main(argv=None) -> int:
    import navpipe_chain as CH
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ruling", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--fixture-record", required=True)
    ap.add_argument("--code-dir", default=HERE)
    ap.add_argument("--dose-chain", default=NC.DOSE_CHAIN)
    ap.add_argument("--banked-seam", default=CH.BANKED_SEAM)
    ap.add_argument("--banked-csv", default=CH.BANKED_CSV)
    ap.add_argument("--token-file", default=CH.NAVTEST_TOKENS)
    ap.add_argument("--ckpt", default=CH.CKPT)
    ap.add_argument("--head", default=CH.HEAD_S0)
    ap.add_argument("--harness", default=CH.HARNESS)
    ap.add_argument("--check-only", action="store_true", help="build and verify in a scratch directory, write nothing")
    a = ap.parse_args(argv)
    pkg = os.path.abspath(a.pkg)
    try:
        r = seal(a)
        out = {n: os.path.join(pkg, n) for n in (SEALED_NAME, SIDECAR_NAME, STAMP_NAME)}
        if any(os.path.exists(p) for p in out.values()):                                                  # @GUARD:S-NO-OVERWRITE
            raise SealRefused(f"{pkg} already holds sealed files: a re-pin is a Revision (DQ8), never an overwrite")
        scratch = tempfile.mkdtemp(prefix="c2seal_", dir=os.environ.get("C2SEAL_TMP"))
        try:
            p = os.path.join(scratch, SEALED_NAME)
            open(p, "w", encoding="utf-8", newline="\n").write(r["text"])
            open(os.path.join(scratch, SIDECAR_NAME), "w", encoding="utf-8", newline="\n").write(r["sha"] + "\n")
            try:
                NC.check_amendment_seal(p, r["sha"], a.code_dir, (), NC.OPERATIVE, NC.REQUIRED_PINS, True)      # the chain's OWN seal check, on the text as written   @GUARD:S-OWN-CHECK
            except NC.Refusal as e:
                raise SealRefused(f"the chain's own seal check refuses the text: {e.reason} {e.detail}")
            if a.check_only:
                print("ZZSEAL_CHECK_OK", r["sha"])
                return 0
            os.makedirs(pkg, exist_ok=True)
            shutil.copyfile(p, out[SEALED_NAME])
            shutil.copyfile(os.path.join(scratch, SIDECAR_NAME), out[SIDECAR_NAME])
            open(out[STAMP_NAME], "w", encoding="utf-8", newline="\n").write(datetime.datetime.now().astimezone().isoformat(timespec="seconds") + "\n" + r["sha"] + "\n")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except SealRefused as e:
        print("ZZSEAL_REFUSED", e)
        return 1
    print("ZZSEALED", r["sha"], out[SEALED_NAME])
    return 0


if __name__ == "__main__":
    sys.exit(main())
