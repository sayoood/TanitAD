#!/usr/bin/env python3
"""PreToolUse guard for Bash and PowerShell: protects the handoff folder dagland/ from stray files.

PI ruling 2026-11-11 ("follow your recommendation", option A, narrow). A shell command is REFUSED when, outside
heredoc / here-string bodies, it
  1. names a path under dagland/ that does not exist yet (a `>>`, `>`, `touch`, `tee` ... would create it), or
  2. names any dagland/ path AND contains a delete / move / rename verb.
Everything else passes: reads of existing files, and the real append `cat >> <existing LOOP_STATE> <<'EOF' ... EOF`.
New dagland files are made with the Write tool, which this hook does not see.

Why: 8 recorded times (2026-11-04..10) a reflexive "no-op" shell command created an empty stray in dagland/,
twice followed by `rm -f` in the same command (memory: no-rm-in-state-appends.md).

Protocol: stdin = the hook JSON ({"tool_name", "tool_input": {"command"}, "cwd", ...}); exit 2 + stderr = block,
exit 0 = allow. Any internal error allows (exit 0) with a note on stderr: this is a guard against a reflex, not a
security boundary, and it must never wedge every shell command.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

DAGLAND = os.environ.get("TANITAD_DAGLAND_DIR", "E:/Projects/TanitAD-artifacts/dagland")

_STOP = r"\s'\"`<>|;&(){},"
TOKEN_RE = re.compile(r"[^%s]*dagland[\\/]+[^%s]*" % (_STOP, _STOP), re.I)
DELETE_RE = re.compile(
    r"(?<![\w.\-])(rm|rmdir|unlink|shred|mv|del|erase|rd|ren|rename|move|ri|mi|rni|"
    r"Remove-Item|Move-Item|Rename-Item|Clear-Content)(?![\w\-])"
    r"|os\.(remove|unlink|rename|replace|rmdir)\b|shutil\.(rmtree|move)\b|\.unlink\(|\.rmdir\(",
    re.I,
)
HEREDOC_RE = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][\w]*)\1")


def strip_bodies(cmd: str) -> str:
    """Drop heredoc bodies (bash) and here-string bodies (PowerShell); keep every command line."""
    lines = cmd.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for _q, delim in HEREDOC_RE.findall(line):
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
            i += 1
        if re.search(r"@['\"]\s*$", line):
            close = "'@" if line.rstrip().endswith("'") else '"@'
            while i < len(lines) and not lines[i].lstrip().startswith(close):
                i += 1
            if i < len(lines):
                out.append(lines[i][lines[i].index(close) + 2:])
            i += 1
    return "\n".join(out)


def _win(p: str) -> str:
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if m else p


def _exists(p: str) -> bool:
    if any(c in p for c in "*?["):
        return bool(glob.glob(p))
    return os.path.exists(p)


def path_exists(token: str, cwd: str) -> bool:
    tok = token.strip()
    cands = [tok, tok.rstrip(".:")]
    for t in cands:
        t = _win(t)
        rel = re.split(r"dagland[\\/]+", t, maxsplit=1, flags=re.I)[1].rstrip("\\/")
        tries = [os.path.join(DAGLAND, rel) if rel else DAGLAND]
        if re.match(r"^[A-Za-z]:[\\/]", t) or t.startswith("/"):
            tries.append(t)
        elif cwd:
            tries.append(os.path.join(cwd, t))
        if any(_exists(x) for x in tries):
            return True
    return False


def decide(cmd: str, cwd: str = "") -> tuple[bool, str]:
    if not cmd or "dagland" not in cmd.lower():
        return False, ""
    scan = strip_bodies(cmd)
    toks = [t for t in TOKEN_RE.findall(scan) if re.search(r"dagland[\\/]", t, re.I)]
    if not toks:
        return False, ""
    verb = DELETE_RE.search(scan)
    if verb:
        return True, (f"dagland guard: this command names {toks[0]!r} and contains '{verb.group(0)}'. "
                      "Never delete, move or rename in a command that names a dagland/ path (PI 2026-11-11).")
    new = [t for t in toks if not path_exists(t, cwd)]
    if new:
        return True, (f"dagland guard: {new[0]!r} does not exist, so this command could create a stray file in "
                      "dagland/. Append only to the existing live LOOP_STATE file; if there is nothing to append, "
                      "send no command. Create a new dagland file with the Write tool (PI 2026-11-11).")
    return False, ""


def main() -> int:
    try:
        data = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
        ti = data.get("tool_input") or {}
        block, why = decide(str(ti.get("command") or ""), str(data.get("cwd") or ""))
    except Exception as e:  # never wedge the shell on a guard bug
        sys.stderr.write(f"dagland guard: internal error, allowing ({type(e).__name__}: {e})\n")
        return 0
    if block:
        sys.stderr.write(why + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
