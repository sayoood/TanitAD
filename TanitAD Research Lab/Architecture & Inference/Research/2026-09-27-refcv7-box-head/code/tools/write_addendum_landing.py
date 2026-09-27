#!/usr/bin/env python3
"""write_addendum_landing.py -- a landing that applies ON TOP OF another, not-yet-landed landing: each repo file's BASE
is the blob that landing declares as NEW (read from its list), not the current tip's. Package files are listed
EXPLICITLY (never swept from a directory diff), so nothing another open landing lists can leak into this one.

    python write_addendum_landing.py --repo-root D:/Projects/TanitAD --package-rel REL --on LANDING_READY_A17.txt
        --subdir a17_toy --src-root <dir with the changed files at repo paths> --files <repo path> ...
        --pkg-files <package-relative path> ... --heading H --out LANDING_READY_A17_TOY.txt

Refuses: a repo file the ``--on`` list does not carry; a CRLF/LF flip against that base; a file whose blob equals the
base; a listed package file that is missing, carries a raw UUID-shaped string or an HF-token-shaped string, or is
itself listed by the ``--on`` landing.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}")
HEAD = re.compile(r"^# (\S+): base (?:NEW|[0-9a-f]{40}) \| new ([0-9a-f]{40}) \| (CRLF|LF)$")


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def eol(data: bytes) -> str:
    return "CRLF" if b"\r\n" in data else "LF"


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--repo-root", "--package-rel", "--on", "--subdir", "--src-root", "--heading", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--pkg-files", nargs="*", default=[])
    a = ap.parse_args()
    rel = a.package_rel.rstrip("/")
    pkg = Path(a.repo_root) / rel
    on_lines = (pkg / a.on).read_text(encoding="utf-8").splitlines()
    on_new = {m.group(1): (m.group(2), m.group(3)) for ln in on_lines if (m := HEAD.match(ln))}
    on_pkg = {ln for ln in on_lines if ln and not ln.startswith("#") and "  ->  " not in ln}
    lines = [f"## {a.heading}",
             f"# Paths are REPO-relative. Applies ON TOP OF {a.on}: each base below is the blob {a.on} declares NEW",
             f"# for that path -- land {a.on} first. Full files; line endings kept. Below: this landing's package files.",
             ""]
    for rp in a.files:
        if rp not in on_new:
            raise SystemExit(f"{rp}: {a.on} does not carry it -- not an addendum file")
        base, base_eol = on_new[rp]
        data = (Path(a.src_root) / rp).read_bytes()
        if eol(data) != base_eol or (base_eol == "CRLF" and data.count(b"\r\n") != data.count(b"\n")):
            raise SystemExit(f"{rp}: line endings {eol(data)} vs the base's {base_eol}")
        if blob(data) == base:
            raise SystemExit(f"{rp}: unchanged vs its base")
        dst = pkg / "code" / a.subdir / rp
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(a.src_root) / rp, dst)
        lines += [f"# {rp}: base {base} | new {blob(data)} | {eol(data)}", f"{rel}/code/{a.subdir}/{rp}  ->  {rp}"]
    lines += ["", f"# ---- this landing's package files ({rel}/) ----"]
    for pf in a.pkg_files:
        p = pkg / pf
        rr = f"{rel}/{pf}"
        if rr in on_pkg:
            raise SystemExit(f"{pf}: already listed by {a.on} -- an addendum must not change it")
        data = p.read_bytes()
        if UUID.search(data):
            raise SystemExit(f"{pf}: a raw UUID-shaped string -- clip ids must be sha12")
        if TOKEN.search(data):
            raise SystemExit(f"{pf}: an HF-token-shaped string")
        lines.append(rr)
    (pkg / a.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.out}: {len(a.files)} repo file(s) on top of {a.on} + {len(a.pkg_files)} package files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
