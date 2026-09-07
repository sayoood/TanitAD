# -*- coding: utf-8 -*-
"""Apply the per-batch conditioning block to refc_adapter.py by EXACT replacement.

Mechanical, idempotent-checked, and it refuses if the anchor text is not found
verbatim — so the patch can never land half-applied on a file that moved.
"""
from __future__ import annotations

import io
import pathlib
import sys

from new_block import NEW

TARGET = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                      else r"C:\Users\Admin\tanitad-perbatch\stack\tanitad\rl\refc_adapter.py")

START = "def assert_conditioning(model, batch) -> dict[str, bool]:"
END_MARK = "    return sample_fn\n"

src = TARGET.read_text(encoding="utf-8")
i = src.index(START)
j = src.index(END_MARK, i) + len(END_MARK)
old = src[i:j]
assert "checked = {\"done\": False}" in old, "anchor block does not contain the defect"
new_src = src[:i] + NEW + src[j:]

# __all__ gains the two new public names
new_src = new_src.replace(
    '__all__ = ["FORWARD_KEYS", "ConditioningError", "conditioning_requirements",',
    '__all__ = ["FORWARD_KEYS", "ConditioningError", "conditioning_requirements",\n'
    '           "ConditioningContract",')
assert '"ConditioningContract",' in new_src

with io.open(TARGET, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(new_src)
print(f"patched {TARGET}  ({len(src)} -> {len(new_src)} chars)")
