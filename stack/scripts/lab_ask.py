"""The FlyWheel <-> Research Lab channel. Append, list, and answer asks.

⭐ PI DECISION 2026-08-31: "let allow the fly wheel agents to communicate directly with
the research lab."

⛔ WHY THIS IS A FILE AND NOT A MESSAGE, stated plainly rather than pretended around.
FlyWheel agents are peer SESSIONS; the Research Lab agent is a SUBAGENT that exists only
while its daily pass runs. A FlyWheel physically cannot send it a message -- the Lab is
not addressable when it is not alive, and no permission change alters that. So the honest
form of "direct" is an ASYNCHRONOUS channel both sides own, with the Master Mind out of
the interpretation path:

    FlyWheel  --ask-->  LAB_ASKS.md  --(standing brief reads it)-->  Research Lab
    Research Lab  --answer-->  LAB_ASKS.md + a package  -->  FlyWheel reads the digest

⭐ AND ASYNCHRONOUS IS BETTER HERE, not a consolation. The Lab runs once a day; a
FlyWheel asking at 03:00 needs the question to survive until the pass. It is auditable,
it lives in the repo, and it survives session death -- which is not hypothetical: the
2026-08-31 Lab agent died twice mid-run on API errors and its banked files were what
made the work recoverable.

⚠️ CONCURRENCY IS A REAL HAZARD HERE, not a theoretical one. Several agents may append
to one file on a mount that intermittently fails reads AND writes with errno 22. Every
write is read-modify-write to a temp file then os.replace (atomic on the same volume),
with a retry loop, and a read-back verification. A lost ask is a question nobody knows
was asked.

⚠️ OUTPUT IS ASCII-ONLY on stdout (cp1252 dev box); the FILE is utf-8 and keeps glyphs.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ASKS = "TanitAD Research Lab/LAB_ASKS.md"
HEADER = """<title>LAB_ASKS - the FlyWheel <-> Research Lab channel</title>

# LAB_ASKS

⭐ **PI decision 2026-08-31:** FlyWheel agents talk to the Research Lab **directly**,
through this file. The Master Mind is no longer in the interpretation path.

⛔ **APPEND WITH THE TOOL, NEVER BY HAND:** `python stack/scripts/lab_ask.py --ask "..."
--from <agent> [--field <field>] [--why "..."]`. Several agents write here and the mount
intermittently fails mid-write; the tool does read-modify-write + atomic replace +
read-back. A hand-edit during another agent's write loses an ask, and **a lost ask is a
question nobody knows was asked**.

**The contract**
| motion | who | when |
|---|---|---|
| **ask** — append an OPEN row: the question, why it matters, what would answer it | any FlyWheel, the MM, the PI | any time |
| **read** — the Research Lab's standing brief MUST read this file and address OPEN rows before pulling backlog seeds | Research Lab | every daily pass |
| **answer** — `--answer ASK-N --package <path>`; the row becomes ANSWERED and points at the evidence | Research Lab | same pass |
| ⚠️ **a question the Lab cannot answer is answered with WHY NOT**, never left silent — an unanswered ask that looks pending forever is the failure this file replaces | Research Lab | same pass |

---

"""


def _read(p: Path) -> str:
    for i in range(8):
        try:
            return p.read_text(encoding="utf-8") if p.exists() else HEADER
        except OSError as e:
            if i == 7:
                raise SystemExit(f"[ask] cannot read {p}: errno {e.errno}")
            time.sleep(2)
    return HEADER


def _write(p: Path, text: str) -> None:
    """Atomic replace + read-back. The mount lies about success often enough."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    for i in range(8):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)
            if p.read_text(encoding="utf-8") == text:
                return
        except OSError as e:
            if i == 7:
                raise SystemExit(f"[ask] cannot write {p}: errno {e.errno}")
            time.sleep(2)
    raise SystemExit(f"[ask] write to {p} did not verify after 8 attempts")


def _next_id(text: str) -> str:
    ids = [int(m) for m in re.findall(r"ASK-(\d+)", text)]
    return f"ASK-{(max(ids) + 1) if ids else 1}"


def add(root: Path, q: str, who: str, field: str, why: str, date: str) -> str:
    p = root / ASKS
    txt = _read(p)
    aid = _next_id(txt)
    row = [f"### {aid} · OPEN · {who}" + (f" · {field}" if field else ""),
           f"*asked {date}*", "", f"**Q.** {q}"]
    if why:
        row += ["", f"**Why it matters.** {why}"]
    row += ["", ""]
    _write(p, txt.rstrip("\n") + "\n\n" + "\n".join(row))
    return aid


def answer(root: Path, aid: str, package: str, note: str, date: str) -> bool:
    p = root / ASKS
    txt = _read(p)
    pat = re.compile(rf"^### {re.escape(aid)} · OPEN ·", re.M)
    if not pat.search(txt):
        return False
    txt = pat.sub(f"### {aid} · ANSWERED ·", txt, count=1)
    anchor = re.search(rf"^### {re.escape(aid)} · ANSWERED ·.*$", txt, re.M)
    ins = txt.index("\n", anchor.end()) + 1
    add_txt = f"\n**A.** *answered {date}* -> `{package}`"
    if note:
        add_txt += f" — {note}"
    add_txt += "\n"
    _write(p, txt[:ins] + add_txt + txt[ins:])
    return True


def listing(root: Path, only_open: bool) -> list[str]:
    txt = _read(root / ASKS)
    rows = re.findall(r"^### (ASK-\d+) · (OPEN|ANSWERED) · (.+)$", txt, re.M)
    return [f"{a}  {s:<9} {rest}" for a, s, rest in rows
            if not only_open or s == "OPEN"]


def main() -> int:
    ap = argparse.ArgumentParser(description="FlyWheel <-> Research Lab channel")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ask")
    ap.add_argument("--from", dest="who", default="")
    ap.add_argument("--field", default="")
    ap.add_argument("--why", default="")
    ap.add_argument("--answer")
    ap.add_argument("--package", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--date", default="", help="YYYY-MM-DD; required with --ask")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--open", action="store_true", help="with --list: OPEN only")
    a = ap.parse_args()
    root = Path(a.repo).resolve()

    if a.ask:
        if not a.who:
            raise SystemExit("[ask] --from is required: an anonymous ask cannot be "
                             "answered back to anyone.")
        if not a.date:
            raise SystemExit("[ask] --date YYYY-MM-DD is required. The system clock "
                             "is not trusted here (narrative dates have run ahead of "
                             "wall-clock before); pass the date you mean.")
        aid = add(root, a.ask, a.who, a.field, a.why, a.date)
        print(f"[ask] {aid} recorded from {a.who}. The Lab reads OPEN rows each pass.")
        return 0
    if a.answer:
        if not a.package:
            raise SystemExit("[ask] --package is required: an answer without evidence "
                             "is an opinion, and this channel carries neither silently.")
        if not a.date:
            raise SystemExit("[ask] --date YYYY-MM-DD is required.")
        ok = answer(root, a.answer, a.package, a.note, a.date)
        print(f"[ask] {a.answer} marked ANSWERED." if ok
              else f"[ask] {a.answer} not found as an OPEN row.")
        return 0 if ok else 1
    rows = listing(root, a.open)
    print(f"[ask] {len(rows)} row(s){' OPEN' if a.open else ''}:")
    for r in rows:
        print("  " + r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
