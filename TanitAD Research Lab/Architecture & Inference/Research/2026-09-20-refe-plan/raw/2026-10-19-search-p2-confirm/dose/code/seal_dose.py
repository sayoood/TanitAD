#!/usr/bin/env python3
"""Seal the SEARCH-DOSE amendment: AMENDMENT_DOSE_DRAFT.md -> AMENDMENT_DOSE.md + AMENDMENT_DOSE.sha256 + SEALED_AT.txt, with the pins the dose chain re-checks (`DoseChain.check_amendment`).

    python seal_dose.py --src ../AMENDMENT_DOSE_DRAFT.md --ruling ../MM_RULING_DOSE.md [--pkg ..] [--c1-pkg C:/Users/Admin/par18/spc]

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


def build_pins(c1_pkg: str, code_dir: str = HERE, ruling: str | None = None) -> str:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--c1-pkg", default=DC.SPC_PKG)
    ap.add_argument("--ruling", required=True, help="MM_RULING_DOSE.md inside --pkg")
    a = ap.parse_args(argv)
    try:
        sha = seal(a.src, a.pkg, a.c1_pkg, a.ruling)
    except SealRefused as e:
        print("ZZSEAL_REFUSED", e, flush=True)
        return 5
    print("ZZSEAL_OK", sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
