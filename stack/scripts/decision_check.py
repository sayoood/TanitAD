"""decision_check — before recommending an architecture change, ASK THE RECORD.

⛔ WHY THIS EXISTS (C129, 2026-08-20). I recommended "freeze a strong encoder"
off the back of a T0 decodability probe. Two things already in the repo refuted
it and I had opened neither:

  * `D-003` — *"frozen-encoder is a COMPARISON ARM, NOT A HEDGE TO ADOPT"*,
    a standing PI decision, one of **48** in `DECISIONS.md`;
  * `refa-4brain-speed-30k` — the frozen-DINOv2 arm that ALREADY RAN it:
    ADE@2s 2.1322, plateaued, *"at a capability ceiling"*, and **2.62 m worse
    than the flagship**.

The failure was not ignorance, it was COST: checking meant knowing which of three
large documents to grep and how. So the fix is not a rule, it is making the check
one command.

⭐ THE STANDING RULE IT SERVES: **an architectural recommendation must cite (a)
the `D-` row it is consistent with and (b) the measured arm that already tested
it — or state explicitly that neither exists.** A probe result is admissible as
EVIDENCE toward such a recommendation and is NEVER sufficient for one.

    python stack/scripts/decision_check.py frozen encoder
    python stack/scripts/decision_check.py --context 3 sigreg collapse

⚠️ This tool SURFACES, it does not adjudicate. A hit is something to read, not a
verdict; an empty result means "nothing matched these words", never "no decision
exists" — the absence rule applies to this instrument like any other, so try a
second vocabulary before concluding the record is silent.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def _first_existing(*rel: str) -> Path:
    """Resolve a record from CANDIDATE paths, newest-migration-first.

    ⛔ MEASURED 2026-08-23: migration P7 moved `DECISIONS.md` to
    `archive/DECISIONS_2026-07-20.md` and this module still named the root path.
    ``units()`` returns [] for a missing file WITHOUT raising, so the tool kept
    exiting 0 while surfacing zero decisions -- a guard whose green result meant
    "I could not read the register", indistinguishable from "nothing applies".
    Resolving from candidates keeps the guard working across moves; the last
    entry is returned when none exist so the caller still gets a real path to
    report as absent.
    """
    for r in rel:
        p = REPO / r
        if p.exists():
            return p
    return REPO / rel[-1]


#: (label, path, regex marking the start of a citable unit)
SOURCES: tuple[tuple[str, Path, str], ...] = (
    ("DECISION",
     _first_existing("DECISIONS.md", "archive/DECISIONS_2026-07-20.md"),
     r"^## (D-[A-Za-z0-9]+)\b"),
    ("REGISTRY", REPO / "Project Steering/MODEL_REGISTRY.md",
     r"^#{2,4} ([0-9][0-9.]*\s.*)$"),
    ("RETRACTION", REPO / "Project Steering/RETRACTION_LOG.md",
     r"^## (C[0-9]+)\b"),
)

#: rows in the registry that ARE a measured arm — these are the "(b)" half.
ARM_ROW = re.compile(r"^\|\s*[0-9]+\s*\|\s*(.+?)\s*\|")


def units(path: Path, header_re: str):
    """-> [(anchor, [lines])] — the document split into citable units."""
    if not path.exists():
        return []
    rx = re.compile(header_re)
    out, anchor, buf = [], "(preamble)", []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = rx.match(line)
        if m:
            if buf:
                out.append((anchor, buf))
            anchor, buf = m.group(1).strip(), [line]
        else:
            buf.append(line)
    if buf:
        out.append((anchor, buf))
    return out


def search(terms: list[str], context: int, require_all: bool):
    pats = [re.compile(re.escape(t), re.I) for t in terms]
    hits = []
    for label, path, header_re in SOURCES:
        if not path.exists():
            print(f"  ⚠️  {label}: {path.name} NOT FOUND — the record this tool "
                  f"checks is incomplete, and that is itself a finding")
            continue
        for anchor, lines in units(path, header_re):
            blob = "\n".join(lines)
            found = [p for p in pats if p.search(blob)]
            ok = len(found) == len(pats) if require_all else bool(found)
            if not ok:
                continue
            # the most informative line: the one matching the most terms
            best_i, best_n = 0, -1
            for i, ln in enumerate(lines):
                n = sum(bool(p.search(ln)) for p in pats)
                if n > best_n:
                    best_i, best_n = i, n
            lo = max(0, best_i - context)
            hi = min(len(lines), best_i + context + 1)
            hits.append((label, path.name, anchor, lines[lo:hi]))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("terms", nargs="+", help="keywords from the proposal")
    ap.add_argument("--context", type=int, default=2,
                    help="lines of context around the best-matching line")
    ap.add_argument("--any", action="store_true",
                    help="match ANY term (default: ALL terms)")
    ap.add_argument("--max", type=int, default=12, help="max hits to print")
    a = ap.parse_args(argv)

    hits = search(a.terms, a.context, require_all=not a.any)
    q = " ".join(a.terms)
    if not hits:
        print(f"\nno record matched {q!r} "
              f"({'ANY' if a.any else 'ALL'} terms).")
        print("⚠️  That is 'nothing matched these WORDS', NOT 'no decision "
              "exists'. Try a second vocabulary before concluding the record "
              "is silent — absence found at one location is not absence.")
        return 1

    print(f"\n{len(hits)} unit(s) in the record mention {q!r}:\n")
    for label, fname, anchor, lines in hits[:a.max]:
        print(f"─── [{label}] {fname} · {anchor}")
        for ln in lines:
            print(f"    {ln[:160]}")
        print()
    if len(hits) > a.max:
        print(f"… {len(hits) - a.max} more not shown (--max to raise).")

    print("⛔ A recommendation must cite (a) the D- row it is consistent with "
          "and (b) the measured arm that already tested it — or say neither "
          "exists. A probe is EVIDENCE, never sufficient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
