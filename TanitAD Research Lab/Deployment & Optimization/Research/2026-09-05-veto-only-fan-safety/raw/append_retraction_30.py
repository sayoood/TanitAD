#!/usr/bin/env python3
"""Append retraction #28 (append-only). Refuses if the heading is already present."""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "RETRACTION_LOG.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
NE, ST, W, ED, X, ARR, SEC = (chr(9940), chr(11088), chr(9888), chr(8212), chr(215),
                              chr(8594), chr(167))
LAM = chr(955)
MARKER = "the MECHANISM LABEL is withdrawn, and the sweep built on it"

TEXT = """

# 2026-09-05 (#NN) {ED} *"the decoder's OFFSET HEAD manufactures the fan's infeasibility"* {ED} the MEASUREMENT stands, the MECHANISM LABEL is withdrawn, and the sweep built on it was interpolating an object that exists nowhere in the decode

**What was claimed**, in `.../2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}6 and register row
`D-REFC-OFFSET-FEAS-1`: *"the vocabulary is almost entirely drivable {ED} 1.6 % envelope violation
at 0.48 g {ED} and the decoder's OFFSET HEAD manufactures the other 87 points, an 8.56{X} blow-up in
peak friction load."*

**What is true, read from source rather than from the docstring** (`stack/tanitad/refs/refc.py`):

```
:1590   bank = self.roll_bank(...)          # refcv3: anchor_v0_cond False -> anchors[None].expand
:1640   conf0, offset = self._decode(kv, cond, x0, 0);   x = bank + offset      # classifier pass
:1676   for i in range(steps):  x_in = x + noise;  _, off = self._decode(...);  x = x_in + off
```

The path from bank to emitted fan is **three stages**, and `out["offset"]` is the
**classifier-pass offset ONLY** {ED} it is assigned once at `:1640` and never reassigned inside the
refinement loop, which adds `off`. Two consequences, one small and one not:

1. **The label.** *"the offset head"* names one stage of a decode that has several. The blow-up is
   created by the **whole decode**; which stage does it was not measured by that section and is now
   measured explicitly (a stage decomposition, free from the same forward).
2. {NE} **The sweep.** A shrink sweep written as `bank8 = fan - out["offset"]` interpolates
   `bank + sum(loop offsets)` {ED} an intermediate that **exists at no point in the decode**. Its
   {LAM} = 0 row read `fan_peak_g` **5.9955** against the anchor bank's independently measured
   **0.4808**, which is what exposed it. Those numbers are **WITHDRAWN**; the artifact is
   **quarantined, not deleted** (`raw/fan_rerank_WITHDRAWN_wrong_lambda_operand.json`) and the sweep
   re-run over `out["anchor_bank"]` (`refc.py:1793`), which is the object the decode actually
   started from.

{W}{W} **THE PART WORTH CARRYING: THE IDENTITY CONTROL PASSED, AND WAS BLIND BY CONSTRUCTION.**
The sweep shipped with a control asserting that {LAM} = 1 reproduces the emitted fan to 1e-5 m, and
it passed {ED} because {LAM} = 1 IS the emitted fan **whatever base you interpolate from**. The
control checked the ARITHMETIC and could not check the OBJECT. That is a control chosen for the
step that was easy to verify rather than for the step that could be wrong, and it is the same
family as CLAUDE.md's *"a tool reporting success is not evidence that its output is right"*.
{ARR} **The repair generalises: an interpolation control must pin the END that is NOT trivially
reproduced.** The corrected sweep asserts that at {LAM} = 0 the rates match the bank rates obtained
**by a different route** (`bank_vs_fan_feasibility.py`, a separate probe reading the checkpoint
tensor directly). A control that must reproduce a value computed through another path is an object
check; one that reproduces a value the code trivially returns is not.

**What is NOT retracted, and it was verified rather than assumed.** {SEC}6's table stands.
`roll_bank` for a fixed vocabulary returns `self.anchors[None].expand(...)` {ED} the source calls it
*"byte-identical to the pre-2026-09-04 `anchors[None].expand(...)`"* {ED} and refcv3's
`config.json['argv']` carries **no** `--anchor-v0-cond` flag, so `core.decoder.anchors` **is** the
bank that decode started from. bank `envelope` **0.0156** / `peak_g` **0.4808 g** vs emitted
**0.8877** / **4.1131 g** is a measurement of the right object.

**Root-cause CLASS {ED} the `df` / `free` / `step_s` / units / tensor-rank family, with the scope
being WHICH TENSOR.** A field named `offset` in an output dict, in a module whose docstring
describes an "offset head", was read as *the* offset. It is *an* offset. {ARR} **Durable rule: before
differencing two tensors from a model's output dict, find the line that ASSIGNS each of them.** The
name is not the definition, and in a loop the last assignment is the one that counts.

{ARR} **Pinned:** `RESULT.md` {SEC}6 (corrected in the same turn, with the source lines quoted),
{SEC}8 (the stage decomposition that replaces the guess); `raw/patch_rerank_lambda_fix.py`;
`stack/scripts/rl_fan_rerank_probe.py`; register row `D-REFC-OFFSET-FEAS-1` (re-worded).
""".replace("{NE}", NE).replace("{ST}", ST).replace("{W}", W).replace("{ED}", ED) \
   .replace("{X}", X).replace("{ARR}", ARR).replace("{SEC}", SEC).replace("{LAM}", LAM)

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)
# {NE} AUTO-NUMBER. Several agents append to this log concurrently: a hard-coded number
# collides silently, and the first version of this script refused with "#28 already
# present" because a SIBLING had taken 28 AND 29 while this one was being written. The
# refusal was correct; the fix is to take max+1 rather than to guess.
import re as _re
nums = [int(m) for m in _re.findall(r"^# \d{4}-\d{2}-\d{2} \(#(\d+)\)", s, _re.M)]
if not nums:
    raise SystemExit("[retraction] could not parse any existing retraction numbers - "
                     "refusing rather than appending an unnumbered entry")
N = max(nums) + 1
if MARKER in s:
    raise SystemExit("[retraction] this entry's body is already present")
s = s.rstrip(LF) + LF + TEXT.replace("(#NN)", "(#%d)" % N)
print("[retraction] taking number #%d (existing max %d)" % (N, max(nums)))
open(PATH, "wb").write((s.replace(LF, CRLF) if is_crlf else s).encode("utf-8"))
print("[retraction] appended (%s)" % ("CRLF" if is_crlf else "LF"))
