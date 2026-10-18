#!/usr/bin/env python3
"""assemble.py -- build Paper v1.5 from the tip blob of Paper/TANITAD_PAPER.md (5fb509a, blob 188d00e3...)

Every edit is an exact-anchor insertion/replacement asserted to match ONCE, performed on the bytes of the tip blob, so the
rest of the paper stays byte-identical (line endings: the tip blob is LF-only; this script never translates newlines).
Inputs : base_TANITAD_PAPER.md (git show 5fb509a:Paper/TANITAD_PAPER.md), sec721_draft.md
Outputs: TANITAD_PAPER.md, inserted_text.txt (every inserted/changed piece, for verify_paper_numbers.py)
"""
import hashlib
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = open(os.path.join(HERE, "base_TANITAD_PAPER.md"), "rb").read().decode("utf-8")
assert "\r" not in BASE, "tip blob is expected to be LF-only"
assert hashlib.sha1(("blob %d\0" % len(BASE.encode("utf-8"))).encode() + BASE.encode("utf-8")).hexdigest().startswith("188d00e3"), "base blob mismatch"

W = 118


def wrap_par(text, width=W, indent=""):
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False,
                         subsequent_indent=indent)


def wrap_block(md, width=W):
    """hard-wrap prose paragraphs the way the paper does; leave tables, headings, blank lines alone"""
    out = []
    for line in md.split("\n"):
        if line.startswith("|") or line.startswith("#") or line.strip() == "" or line.startswith(">"):
            out.append(line)
        else:
            out.append(wrap_par(line, width))
    return "\n".join(out)


def once(s, anchor, repl):
    assert s.count(anchor) == 1, (s.count(anchor), anchor[:80])
    return s.replace(anchor, repl)


inserted = []
new = BASE

# 1. status line ---------------------------------------------------------------------------------------------------
new = once(new, "living paper, **v1.4 (2026-10-17)**. Maintained", "living paper, **v1.5 (2026-10-18)**. Maintained")
inserted.append("living paper, **v1.5 (2026-10-18)**.")

# 2. version note ----------------------------------------------------------------------------------------------------
status_add = (
    "**v1.5 (2026-10-18)** adds §7.21 — what four packages that landed in the three days after §7.20 add to it: the "
    "oracle − pick headroom inside P2's own 64-proposal fan (9.41 PDMS points, 62 % of it ranking among non-zero "
    "proposals, which the lever ranking of §7.20(g) left out), the navtest anatomy of the RL lever (it buys progress and "
    "pays in NC and TTC, so no free point is deployable-safe), why the safety constraint built to remove that payment "
    "failed (it held on an expectation under the training policy while its acceptance test reads the argmax), and a "
    "selection rule that is inert because the drivable-area failures are confident. It is v1.5 rather than v1.4.1 "
    "because it changes the lever ranking the programme works from and states one new method (a constraint must bind "
    "the statistic its acceptance test reads). The abstract is unchanged: no number in it moves."
)
status_block = wrap_par(status_add, 104)
new = once(new, "Status line, this note and the changelog move in the same edit.)*",
           "Status line, this note and the changelog move in the same edit.\n" + status_block + ")*")
inserted.append(status_add)

# 3. §7.20(g) dated pointer ------------------------------------------------------------------------------------------
ptr = (
    "*(Update v1.5, 2026-10-18: the ranking above priced only the zero tail. §7.21(a) measures the whole oracle − pick "
    "headroom inside P2's own fan, 9.41 PDMS points, of which the zero tail is 3.54 and ranking among non-zero "
    "proposals 5.875, and §7.21(b)–(d) report the three levers tried at the scorer since: an RL point that buys progress "
    "and pays in NC and TTC, a safety constraint that held on the wrong quantity, and a selection rule that is inert. P6 "
    "is unchanged: it is still the only lever with a non-zero share on the near-twin class (d), and it has still not "
    "trained.)*"
)
new = once(new, "pair's +0.33 as the floor it must clear.\n\n**What §7.20 does not establish.**",
           "pair's +0.33 as the floor it must clear.\n\n" + wrap_par(ptr) + "\n\n**What §7.20 does not establish.**")
inserted.append(ptr)

# 4. the new section --------------------------------------------------------------------------------------------------
sec = open(os.path.join(HERE, "sec721_draft.md"), "rb").read().decode("utf-8")
assert "\r" not in sec
sec = sec.rstrip("\n")
sec_w = wrap_block(sec)
new = once(new, "\n\n## 8. Discussion: self-supervision", "\n\n" + sec_w + "\n\n## 8. Discussion: self-supervision")
inserted.append(sec)

# 5. references --------------------------------------------------------------------------------------------------------
ref = ("Added v1.5 (§7.21; PUBLISHED and banked, sha256 in `TanitAD Research Lab/Library/library.json`): CLOVER "
       "arXiv:2605.15120; DriveZero arXiv:2609.06055 and NAVSIM arXiv:2406.15349, which §7.20 has cited by library key "
       "since v1.4 and which are listed here for the first time.")
new = once(new, "(already cited in §13.1).)", "(already cited in §13.1). " + "\n".join(wrap_par(ref, 104).split("\n")) + ")")
inserted.append(ref)

# 6. changelog -----------------------------------------------------------------------------------------------------------
cl = (
    "- **v1.5 (2026-10-18): where the REFe headroom sits, and the first levers aimed at it.** **New §7.21** (tier T1 for "
    "(a), (b), (d); the RL-val selection tier, not T1, for (c)). (a) P2's own 64-proposal table over the full navtest, "
    "proven P2's by a same-pick check on all 12,146 tokens: best-of-64 oracle 98.23 [97.82, 98.62] against the pick "
    "88.82 [87.86, 89.77], a 9.41 [8.68, 10.18] ceiling of which 62.4 % is ranking among non-zero proposals and 37.6 % "
    "the zero tail §7.20 priced: the scorer, not the fan, holds it (a ceiling, not an expected gain). (b) The RL lever's "
    "navtest anatomy, row FP-NONE: R0B over the deployed scorer PDMS +0.55 [+0.24, +0.84], EP +1.81, with NC −0.25 and "
    "TTC −0.40 both separated adverse and 36 more zero-PDMS picks; the family screen flags 0 of 21 but has low power, "
    "and the 'slow bias removed' reading in the package prose does not survive its own JSON (the two intervals overlap). "
    "(c) Why the safety constraint failed: the dual held in-sample on E_π[TTC] at the training temperature T = 0.1 "
    "(signed g −0.000343) while the pick lost TTC −0.767 [−1.487, −0.065] on RL-val (K3-killed, outcome O4); on the "
    "unconstrained R0's full RL-val the constrained quantity understates the pick's loss (+0.223 against +1.000 at "
    "u1000, paired difference −0.777 [−1.320, −0.251], separated at one of three states), a train-to-val gap is not "
    "demonstrated, and a pick-aligned temperature (T = 0.02) is PROVISIONAL because its control 3 failed as registered. "
    "(d) SELRULE-DAC reads ADOPT, +0.0235 [+0.0002, +0.0546] PDMS = 0.7 % of the O-DAC oracle reach: inert, consistent "
    "with confident drivable-area errors (median p̂_DAC 0.932 at the failing picks). **Stated as method:** a constraint "
    "must bind the statistic its acceptance test reads. Four metric families reported where the source carries them; "
    "STRATEGIC PAUSED (PI, 2026-09-19). §7.20(g) gains a dated pointer; the one literature row used (CLOVER, library "
    "`2605.15120`) was re-read in the banked PDF; the abstract is unchanged. One-seed states throughout: no lever-effect "
    "claim."
)
cl_w = textwrap.fill(cl, width=W, break_long_words=False, break_on_hyphens=False, subsequent_indent="  ")
assert new.endswith("\n") and not new.endswith("\n\n")
new = new + "\n" + cl_w + "\n"
inserted.append(cl)

out = os.path.join(HERE, "TANITAD_PAPER.md")
with open(out, "wb") as f:
    f.write(new.encode("utf-8"))
with open(os.path.join(HERE, "inserted_text.txt"), "wb") as f:
    f.write(("\n\n".join(inserted) + "\n").encode("utf-8"))
print("wrote", out, len(new.encode("utf-8")), "bytes;", new.count("\n"), "lines; CR count", new.count("\r"))
