#!/usr/bin/env python3
"""Build the staged box-2 queue: the tip's v8f_mini_box2_queue.cmd (2a78f059) + the DAgger stages, ADDITIVELY.

    tip_v8f_mini_box2_queue.cmd  (git show 2a78f059:<PK>/code/v8f_mini_box2_queue.cmd, CRLF)
  + queue_additions/doc.txt       after the `evalorc` documentation line
  + queue_additions/dispatch.txt  after the `evalorc` dispatch line
  + queue_additions/stages.txt    appended at the end of the file

Every anchor must occur EXACTLY once; every output line keeps the CRLF the tip uses. The result is written to stage/<PK>/code/v8f_mini_box2_queue.cmd, with a unified diff against the tip beside it
(queue.diff). The selftest re-derives the file and asserts the diff has no removed line.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TIP = HERE / "tip_v8f_mini_box2_queue.cmd"
OUT = HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
ADD = HERE / "queue_additions"
DOC_ANCHOR = "rem   v8f_mini_box2_queue.cmd evalorc ARM SEED "
DISPATCH_ANCHOR = 'if /i "%~1"=="evalorc" goto :evalorc'


def lines_of(path: Path) -> list:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n").split("\n")


def build() -> bytes:
    tip = TIP.read_bytes().decode("utf-8")
    assert "\r\n" in tip and tip.endswith("\r\n"), "the tip queue is CRLF"
    L = tip[:-2].split("\r\n")
    doc, disp, stages = (lines_of(ADD / n) for n in ("doc.txt", "dispatch.txt", "stages.txt"))
    for part in (doc, disp, stages):
        while part and part[-1] == "":
            part.pop()
    i_doc = [i for i, x in enumerate(L) if x.startswith(DOC_ANCHOR)]
    i_dis = [i for i, x in enumerate(L) if x == DISPATCH_ANCHOR]
    if len(i_doc) != 1 or len(i_dis) != 1:
        sys.exit(f"anchors: doc {i_doc} dispatch {i_dis}: each must occur exactly once")
    new = L[:i_doc[0] + 1] + doc + L[i_doc[0] + 1:i_dis[0] + 1] + disp + L[i_dis[0] + 1:] + stages
    return ("\r\n".join(new) + "\r\n").encode("utf-8")


def main() -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    a = TIP.read_bytes().decode("utf-8").splitlines(keepends=False)
    b = data.decode("utf-8").splitlines(keepends=False)
    diff = list(difflib.unified_diff(a, b, "a/" + PK + "/code/v8f_mini_box2_queue.cmd", "b/" + PK + "/code/v8f_mini_box2_queue.cmd", lineterm="", n=2))
    (HERE / "queue.diff").write_text("\n".join(diff) + "\n", encoding="utf-8", newline="\n")
    removed = [x for x in diff if x.startswith("-") and not x.startswith("---")]
    print(f"wrote {OUT}  sha256 {hashlib.sha256(data).hexdigest()}  lines {len(b)} (tip {len(a)})  removed lines {len(removed)}")
    return 1 if removed else 0


if __name__ == "__main__":
    sys.exit(main())
