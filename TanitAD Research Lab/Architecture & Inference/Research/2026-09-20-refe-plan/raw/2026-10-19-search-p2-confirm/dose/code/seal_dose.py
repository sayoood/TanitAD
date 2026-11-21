#!/usr/bin/env python3
"""Seal the SEARCH-DOSE amendment: AMENDMENT_DOSE_DRAFT.md -> AMENDMENT_DOSE.md + AMENDMENT_DOSE.sha256 + SEALED_AT.txt, with the pins the dose chain re-checks (`DoseChain.check_amendment`).

    python seal_dose.py --src ../AMENDMENT_DOSE_DRAFT.md --ruling ../MM_RULING_DOSE.md [--pkg ..] [--c1-pkg C:/Users/Admin/par18/spc]
    python seal_dose.py --revision1 --pkg <package dir> [--note <pkg>/REVISION1_NOTE.md]     (REVISION 1, MM_RULING_C2FINAL.md point 1: AMENDMENT_DOSE.md + the note -> AMENDMENT_DOSE_R1.md; see `seal_revision1`)

Fills the draft's two placeholders: `@@RULING@@` (the MM's pre-seal ruling folded in VERBATIM with its sha256; REQUIRED, and pinned as `ruling_sha256`) and `@@PINS@@` (the machine-readable pins block).  Refuses unless the draft's first
Status line is DRAFT, both placeholders occur exactly once, C1's prereg is itself sealed, and the O4 file equals the pinned constant; never overwrites an existing sealed file.  After the seal, NO pinned file may change."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dose_chain as DC  # noqa: E402

CH = DC.CH
PINNED_CODE = ("dose_chain.py", "test_dose.py", "mutation_dose.py", "seal_dose.py", "cost_dose.py")


class SealRefused(Exception):
    pass


def build_pins(c1_pkg: str, code_dir: str = HERE, ruling: str | None = None, extra: tuple = ()) -> str:
    sha_file = os.path.join(c1_pkg, "PREREG_SEARCH_P2_CONFIRM.sha256")
    prereg = os.path.join(c1_pkg, CH.SEALED_NAME)
    if not (os.path.isfile(sha_file) and os.path.isfile(prereg)):
        raise SealRefused("C1's sealed prereg / its .sha256 are missing")
    c1_sha = open(sha_file, encoding="utf-8").read().split()[0].strip().lower()
    if hashlib.sha256(open(prereg, "rb").read().replace(b"\r\n", b"\n")).hexdigest() != c1_sha:
        raise SealRefused("C1's prereg does not equal its .sha256")
    if CH.sha_lf(DC.CONFIG_O4) != DC.O4_SHA256:
        raise SealRefused("the O4 config file differs from the pinned constant")
    lines = [f"c1_prereg_sha256 {c1_sha}", f"config_o4_sha256 {DC.O4_SHA256}", f"config_c8_sha256 {CH.sha_lf(DC.CONFIG_C8)}"]
    if ruling:
        lines.append(f"ruling_sha256 {CH.sha_lf(ruling)}")
    lines += list(extra)
    lines += [f"code:{f} {CH.sha_lf(os.path.join(code_dir, f))}" for f in PINNED_CODE]
    return "<!-- PINS-BEGIN -->\n```\n" + "\n".join(lines) + "\n```\n<!-- PINS-END -->"


def seal(src: str, pkg: str, c1_pkg: str = DC.SPC_PKG, ruling: str | None = None, code_dir: str = HERE, now: datetime.datetime | None = None) -> str:
    text = open(src, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
    m = re.search(r"^Status: .*$", text, re.M)
    if not m or not m.group(0).startswith("Status: **DRAFT"):
        raise SealRefused(f"the draft's first Status line is not DRAFT: {(m.group(0)[:40] if m else None)!r}")
    for ph in ("@@RULING@@", "@@PINS@@"):
        if text.count(ph) != 1:
            raise SealRefused(f"placeholder {ph} occurs {text.count(ph)} times (must be exactly 1)")
    out_md, out_sha, out_at = (os.path.join(pkg, n) for n in (DC.AMEND_NAME, DC.AMEND_SHA, "AMENDMENT_DOSE.SEALED_AT.txt"))
    if any(os.path.exists(p) for p in (out_md, out_sha, out_at)):
        raise SealRefused("a sealed amendment already exists here: never overwritten")
    if not ruling or not os.path.isfile(ruling):
        raise SealRefused("the MM ruling file is required (it is folded in verbatim and its sha256 is pinned)")
    if os.path.basename(ruling) != DC.RULING_NAME or os.path.abspath(os.path.dirname(ruling)) != os.path.abspath(pkg):
        raise SealRefused(f"the ruling must be {DC.RULING_NAME} inside the sealed package dir (the chain re-hashes it there)")
    pins = build_pins(c1_pkg, code_dir, ruling)
    if ruling:
        rt = open(ruling, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
        rblock = ("## Revision 1 (MM ruling, pre-seal)\nThe Master Mind's ruling on DQ1..DQn, folded in VERBATIM (sha256 of the ruling file: " + hashlib.sha256(rt.encode("utf-8")).hexdigest()
                  + "). **Where the body and the ruling differ, THE RULING PREVAILS.**\n\n````markdown\n" + rt.rstrip("\n") + "\n````\n")
    else:
        rblock = "## Revision 1\nNo pre-seal ruling was folded in: the body below is the text sealed, with the recommendations of its DQ section adopted as written.\n"
    now = now or datetime.datetime.now().astimezone()
    stamp = now.isoformat(timespec="seconds")
    sealed = text.replace("@@RULING@@", rblock).replace("@@PINS@@", pins)
    sealed = re.sub(r"^Status: \*\*DRAFT.*$", f"Status: **SEALED** {stamp} (the sha256 of this text is in AMENDMENT_DOSE.sha256; code/dose_chain.py REFUSES to run without both and without every pin below)", sealed, count=1, flags=re.M)
    os.makedirs(pkg, exist_ok=True)
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(sealed)
    sha = hashlib.sha256(sealed.encode("utf-8")).hexdigest()
    open(out_sha, "w", encoding="utf-8", newline="\n").write(sha + "\n")
    open(out_at, "w", encoding="utf-8", newline="\n").write(stamp + "\n" + sha + "\n")
    return sha


def build_r1_text(orig_text: str, note: str, pins_block: str, status: str) -> str:
    """REVISION 1 text = the ORIGINAL sealed text with exactly three edits: its Status line replaced by `status`, the dated note block inserted directly after it, and its pins block replaced.  Pure."""
    t = orig_text.replace("\r\n", "\n")
    m = re.search(r"^Status: .*\n", t, re.M)
    if not m:
        raise SealRefused("the original has no Status line")
    if t.count("<!-- PINS-BEGIN -->") != 1 or t.count("<!-- PINS-END -->") != 1:
        raise SealRefused("the original must carry exactly one pins block")
    if DC.REV1_BEGIN in t or DC.REV1_END in t:
        raise SealRefused("the original already carries a REVISION 1 note block")
    block = DC.REV1_BEGIN + "\n" + note.strip("\n") + "\n" + DC.REV1_END + "\n"
    t = t[:m.start()] + status + "\n" + block + t[m.end():]
    return re.sub(r"<!-- PINS-BEGIN -->.*?<!-- PINS-END -->", lambda _: pins_block, t, count=1, flags=re.S)


def seal_revision1(pkg: str, note: str, c1_pkg: str = DC.SPC_PKG, code_dir: str = HERE, now: datetime.datetime | None = None) -> str:
    """REVISION 1 (MM_RULING_C2FINAL.md point 1): the sealed AMENDMENT_DOSE.md (never edited) + the dated note -> AMENDMENT_DOSE_R1.md + AMENDMENT_DOSE_R1.sha256 + AMENDMENT_DOSE_R1.SEALED_AT.txt in `pkg`.
    Everything it reads lives in `pkg`: the original and its .sha256, MM_RULING_DOSE.md (unchanged: still pinned as `ruling_sha256`), MM_RULING_C2FINAL.md (must equal `DC.RULING_C2FINAL_SHA256`).  The terms text of
    the result is asserted equal to the original's before anything is written; the code shas are re-pinned from `code_dir`.  Never overwrites; a second call refuses."""
    j = lambda n: os.path.join(pkg, n)       # noqa: E731
    out_md, out_sha, out_at = j(DC.AMEND_R1_NAME), j(DC.AMEND_R1_SHA), j("AMENDMENT_DOSE_R1.SEALED_AT.txt")
    if any(os.path.exists(p) for p in (out_md, out_sha, out_at)):
        raise SealRefused("a sealed REVISION 1 already exists here: never overwritten")
    for need in (j(DC.AMEND_NAME), j(DC.AMEND_SHA), j(DC.RULING_NAME), j(DC.RULING_C2FINAL_NAME), note):
        if not os.path.isfile(need):
            raise SealRefused(f"missing {need}")
    orig_bytes = open(j(DC.AMEND_NAME), "rb").read().replace(b"\r\n", b"\n")
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    if orig_sha != open(j(DC.AMEND_SHA), encoding="utf-8").read().split()[0].strip().lower():
        raise SealRefused("the original AMENDMENT_DOSE.md does not equal its .sha256")
    orig_text = orig_bytes.decode("utf-8")
    m = re.search(r"^Status: .*$", orig_text, re.M)
    if not m or not m.group(0).startswith("Status: **SEALED"):
        raise SealRefused("the original's first Status line is not SEALED")
    if CH.sha_lf(j(DC.RULING_C2FINAL_NAME)) != DC.RULING_C2FINAL_SHA256:
        raise SealRefused("MM_RULING_C2FINAL.md differs from the pinned constant")
    if CH.parse_pins(orig_text).get("ruling_sha256") != CH.sha_lf(j(DC.RULING_NAME)):
        raise SealRefused("MM_RULING_DOSE.md differs from the ruling the original pinned")
    note_text = open(note, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
    if DC.REV1_BEGIN in note_text or DC.REV1_END in note_text or "<!-- PINS-" in note_text:
        raise SealRefused("the note carries a block delimiter")
    if DC.RULING_C2FINAL_SHA256[:8] not in note_text:                                       # also refuses an empty note
        raise SealRefused("the note does not name MM_RULING_C2FINAL.md by its sha256")
    pins = build_pins(c1_pkg, code_dir, j(DC.RULING_NAME), extra=(f"ruling_c2final_sha256 {CH.sha_lf(j(DC.RULING_C2FINAL_NAME))}", f"amendment_original_sha256 {orig_sha}"))
    now = now or datetime.datetime.now().astimezone()
    stamp = now.isoformat(timespec="seconds")
    status = (f"Status: **SEALED** {stamp} (REVISION 1 of AMENDMENT_DOSE.md [sha256 {orig_sha[:8]}...], the terms text byte-identical to it; the sha256 of this text is in AMENDMENT_DOSE_R1.sha256; "
              "code/dose_chain.py REFUSES to run without both and without every pin below)")
    sealed = build_r1_text(orig_text, note_text, pins, status)
    if DC.terms_text(sealed) != DC.terms_text(orig_text):
        raise SealRefused("the terms text of REVISION 1 differs from the original's")
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(sealed)
    sha = hashlib.sha256(sealed.encode("utf-8")).hexdigest()
    open(out_sha, "w", encoding="utf-8", newline="\n").write(sha + "\n")
    open(out_at, "w", encoding="utf-8", newline="\n").write(stamp + "\n" + sha + "\n")
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="the DRAFT (not with --revision1)")
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--c1-pkg", default=DC.SPC_PKG)
    ap.add_argument("--ruling", help="MM_RULING_DOSE.md inside --pkg (not with --revision1)")
    ap.add_argument("--revision1", action="store_true", help="seal REVISION 1: AMENDMENT_DOSE.md + --note -> AMENDMENT_DOSE_R1.md (+ .sha256, .SEALED_AT.txt) in --pkg")
    ap.add_argument("--note", help="the REVISION 1 note file (default <pkg>/REVISION1_NOTE.md)")
    a = ap.parse_args(argv)
    if a.revision1:
        try:
            sha = seal_revision1(a.pkg, a.note or os.path.join(a.pkg, "REVISION1_NOTE.md"), a.c1_pkg)
        except SealRefused as e:
            print("ZZSEAL_REFUSED", e, flush=True)
            return 5
        print("ZZSEAL_R1_OK", sha, flush=True)
        return 0
    if not a.src or not a.ruling:
        ap.error("--src and --ruling are required (without --revision1)")
    try:
        sha = seal(a.src, a.pkg, a.c1_pkg, a.ruling)
    except SealRefused as e:
        print("ZZSEAL_REFUSED", e, flush=True)
        return 5
    print("ZZSEAL_OK", sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
