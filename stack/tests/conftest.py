"""Shared test environment — Windows parity for the suite.

WHY THIS EXISTS (2026-08-15). The cloud campaign certified the stop-state tree at
**2,804 passed / 0 failed on Linux**. The identical tree on the Windows dev box
failed **12 tests, all one environment class, zero logic defects**:

* 11 of 12: a CLI under test prints ``⛔`` (U+26D4) in its help/refusal text; the
  child Python on Windows inherits the **cp1252** console encoding, raises
  ``UnicodeEncodeError`` inside ``print()``, and dies with exit 1 before reaching
  the exit code the test asserts. The *refusal logic was working* — the process
  crashed while saying so. (Same family as the MooseFS ``Errno 5`` lesson: a
  reporter that cannot write cannot report.)
* 1 of 12: a POSIX path-separator assertion (`/root/abc.mp4` vs `\\root\\abc.mp4`)
  — patched in the test itself, not here.

The fix belongs at the environment seam, not in 11 call sites: every subprocess
spawned by a test inherits ``PYTHONIOENCODING=utf-8``, which is a no-op on Linux
(already UTF-8) and makes Windows children encode their own output correctly.
"""

import os

# Set at import time so it is inherited by every subprocess the tests spawn,
# regardless of whether they pass ``env=`` explicitly (those that do build on
# ``os.environ`` copies).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ⚠️ One test also DECODES a child's output in the parent (``text=True`` uses
# the locale codec, fixed at interpreter start — too late to change here). On
# Windows, run the suite as:  PYTHONUTF8=1 python -m pytest
# MEASURED 2026-08-15 with both set, FULL suite on this box:
# 2,810 passed / 0 failed / 17 skipped / 2 xfailed — matching the campaign's
# Linux certification (2,804/0 at stop; the delta is the locally-banked tests
# plus the one win32 platform skip).
os.environ.setdefault("PYTHONUTF8", "1")   # for grandchildren, at minimum
