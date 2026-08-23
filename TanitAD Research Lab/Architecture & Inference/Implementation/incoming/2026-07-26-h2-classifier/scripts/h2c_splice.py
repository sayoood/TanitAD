"""Splice the rendered tables into `H2_CLASSIFIER.md` at its `<!-- TABLES:* -->` markers.

The report's numbers must come from the artifacts, not from a human retyping them — three of the
program's own retractions entered by transcription. `h2c_report.py` renders; this file places.
Idempotent: re-running replaces the previously spliced block, so the markers survive.

usage:  python h2c_splice.py <artifacts dir> <H2_CLASSIFIER.md>
"""
from __future__ import annotations

import re
import subprocess
import sys
import os

MAP = {
    "RESULTS": ["Discrimination", "Is any arm above CHANCE", "Paired AP deltas",
                "The operating point", "The efficiency trade-off CURVE",
                "The efficiency ledger", "Measured compute"],
    "SENS": ["C12 — the LABEL's own structure", "C12 — the composite decomposed",
             "C12 — the CORRECTED", "Sensitivities"],
    "CV": ["Training-side cross-validation"],
}
BEGIN = "<!-- TABLES:%s -->"
END = "<!-- /TABLES:%s -->"


def main():
    art, doc = sys.argv[1:3]
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, os.path.join(here, "h2c_report.py"), art],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    sections = {}
    cur, buf = None, []
    for line in out.splitlines():
        if line.startswith("### "):
            if cur:
                sections[cur] = "\n".join(buf).rstrip()
            cur, buf = line[4:].strip(), [line]
        elif cur:
            buf.append(line)
    if cur:
        sections[cur] = "\n".join(buf).rstrip()

    txt = open(doc, encoding="utf-8").read()
    for key, wanted in MAP.items():
        picked = [v for w in wanted for k, v in sections.items() if k.startswith(w)]
        block = f"{BEGIN % key}\n\n" + "\n\n".join(picked) + f"\n\n{END % key}"
        pat = re.compile(re.escape(BEGIN % key) + r".*?" + re.escape(END % key), re.S)
        txt = pat.sub(lambda _m: block, txt) if pat.search(txt) \
            else txt.replace(BEGIN % key, block)
        print(f"[splice] {key}: {len(picked)} sections")
    open(doc, "w", encoding="utf-8").write(txt)
    print(f"[splice] wrote {doc}")


if __name__ == "__main__":
    main()
