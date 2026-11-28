"""Append the RL-generator-collisions retraction to RETRACTION_LOG.md.

Takes max(#NN)+1 rather than a hard-coded number (sibling streams take numbers while this
is being written). Idempotent on the marker, content-verified after the write.
"""
import io
import os
import re
import sys
import time

F = os.path.join(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD",
                 "Project Steering", "RETRACTION_LOG.md")
MARKER = "D-RL-TOP32-NOMOVE-1"

src = None
for _ in range(12):
    try:
        with io.open(F, encoding="utf-8") as fh:
            src = fh.read()
        break
    except OSError:
        time.sleep(3)
if src is None:
    raise SystemExit("INCONCLUSIVE: could not read RETRACTION_LOG.md (mount)")

if MARKER in src:
    print("ALREADY APPLIED")
    sys.exit(0)

nums = [int(m) for m in re.findall(r"\(#(\d+)\)", src)]
n = (max(nums) + 1) if nums else 32
print("highest existing retraction = #%d -> writing #%d" % (max(nums) if nums else 0, n))

ENTRY = u"""

# 2026-09-05 (#%d) \u2014 A RETRACTION ENTRY THAT ITSELF CARRIED AN UNVERIFIED NUMBER, AND THE COLLISION DEFINITION THAT MADE EVERY PRIOR `contact` NUMBER 37.8 %% LOW (Architecture & Inference FlyWheel)

## 1. RETRACTED \u2014 `top32_contact` "0.03299 \u2192 0.02691 = \u22120.00729, the one gain" DOES NOT EXIST

Retraction **#31** corrected a real error (reading `sel_contact` when the question was about the
generator) and, in its own durable-fix list, wrote: *"Carry a stream's own 'one gain' forward. The
\u22120.00729 on `top32_contact` was measured, labelled a gain, and dropped at the summary layer."*
It pinned that number to `\u2026/2026-09-05-veto-only-fan-safety/RESULT.md`. **That number is not in
that file, and it is not in any banked artifact.**

**MEASURED** in `\u2026/veto-only-fan-safety/raw/veto_verdict_veto200.json` and
`\u2026/veto_verdict_veto2k.json`:

| arm | `top32_contact` base | after s0 | after s1 | \u0394 s0 |
|---|---|---|---|---|
| `veto200` | 0.03298611 | 0.03298611 | 0.03298611 | **+0.000000, CI [0, 0]** |
| `veto2k` | 0.03298611 | 0.03298611 | 0.03298611 | **+0.000000, CI [0, 0]** |

and `fan_contact` moved the **wrong way** \u2014 **+0.000391** (200 steps), **+0.001693** (2 000
steps), neither separated. The strings `0.02691`, `0.0269` and `0.00729` appear **nowhere** in that
package's `RESULT.md`, searched in **bare AND comma-grouped** form (the `45,456` lesson). `0.0269`
occurs once in `raw/fan_rerank_veto200s0.json`, whose `*__contact` entries are **all 0.0 for every
selection rule** \u2014 those are **selected-path** metrics, so that file cannot be the source of a
*fan* number.

\u21d2 **No RL arm has ever moved the generator's collision rate.**

\u2192 **ROOT-CAUSE CLASS: a number quoted without its arm and its artifact path \u2014 propagated
through the RETRACTION LOG, which is supposed to be where that stops.** #31 was written to fix a
scope error and, in the same breath, introduced a provenance error. The mechanism is the one
`CLAUDE.md` opens with: *"this rule exists because prose lied to us"*. A retraction entry is prose.

\u2192 **Why it mattered.** #31's number became the **motivating premise of the successor brief** \u2014
*"RL ALREADY moved it \u2026 I did not carry that forward"* \u2014 and therefore of a whole work
package's framing. A fabricated gain is worse than a missing one: it makes the next experiment look
like a confirmation rather than a first attempt.

\u2192 **Durable fix.** \u26d4 **A retraction entry's own numbers are claims and must cite the RAW
JSON, not the report being retracted.** Where #31 says a value "was measured", the entry must name
the file and key. Applied here: every number above names its verdict file.

\u26a0 **Scope.** #31's PRIMARY correction \u2014 that `sel_contact` measures the SELECTOR and the RL
target is the generator \u2014 **STANDS and is strengthened**: MEASURED, the selector picks a
non-collider in **240/240** windows, and the generator's own rate is **larger** than #31 said (\u00a72).
Only the "one gain" sentence and its pin are withdrawn.

## 2. CORRECTED \u2014 the banked collision flag uses a SUPERSEDED definition; every prior `contact` number is 37.8 %% low

`rewards._collision` is **SWEPT in the relative frame** (a segment crossing the 2 m disc counts even
when neither endpoint is inside). `fan_bank_base_240w.npz`'s `f_contact` **predates** it.

**POSITIVELY IDENTIFIED, not inferred** \u2014 the per-step **POINT** test reproduces the banked flag
with **0 disagreements over 30,720 candidates**, while every structural flag (`kamm_over`,
`envelope`, `off_reach`, `infeasible`) reproduces **exactly** under the current scorer and only
`contact` differs, **one-sidedly** (289 extra, 0 fewer):

| | POINT (banked, superseded) | **SWEPT (current)** |
|---|---|---|
| `fan_contact`, all 240 windows | 0.024870 (764 / 30,720) | **0.034277 (1,053 / 30,720)** |
| `fan_contact`, 65 lead windows | 0.095673 | **0.126562** |

\u2b50 Corroborated by an independent route: `_swept_hit` landed in commit `9765634`, which
`\u2026/veto-only-fan-safety/raw/chain_suite.sh` records as arriving **after** the s0/s1 arms were
frozen. And by a third, from the other side: **289 of 1,053 colliders are "swept-only"** (no sampled
point inside the disc), and **1,053 \u2212 764 = 289** exactly.

\u2192 **ROOT-CAUSE CLASS: a true quantity quoted outside its scope \u2014 here the scope is the
DEFINITION**, the same family as the `control_units` trap, the cylindrical-FOV trap and the `step_s`
divisor. \u2192 **Durable fix: state the collision definition beside any `contact` number, or it is
not quotable.**

## 3. MY OWN, caught before it shipped \u2014 a rank probe whose CONSTANT control read +0.6464 instead of 0

Measuring whether `progress` ranks colliding candidates higher, I used `argsort(argsort(\u00b7))`
ordinal ranks. The contact flag is binary and **~97 %% ties**, and `np.argsort` is **stable**, so a
**constant** score received the ranks 0\u2026K\u22121 **in index order** and correlated with wherever the
colliders happened to sit. The **K1 constant control read +0.6464** (must be 0) and the **K2 oracle
control read +0.0810** (must be \u22121). It would have shipped `progress` at **+0.49** as a measured
result.

\u2192 Fixed by switching to a **tie-safe AUC**; both controls then read their no-information values
**exactly** (0.5000 and 0.0000). \u2192 **ROOT-CAUSE CLASS: a probe that tunes/ranks on a quantity
whose tie structure it ignores** \u2014 the `CLAUDE.md` probe-panel rule (*"every probe panel carries
a CONSTANT-ONLY control that must read the no-information value exactly"*) earning its place again.
**Nothing was retracted because the control fired first**, which is the entire argument for the rule.

\u2192 **Pinned:** `\u2026/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions/`
\u2014 `SPEC.md` \u00a70, `RESULT.md` \u00a71, `raw/p2_feasible_vs_contact.json` (control C1b),
`raw/p3_progress_binds.json` (controls K1/K2), `raw/p6_graded_term_headroom.json`.
Register rows: `D-RL-TOP32-NOMOVE-1`, `D-RL-CONTACT-DEFN-1`, `D-RL-PROGRESS-COMPOSED-1`.
""" % n

out = src.rstrip("\n") + ENTRY

for _ in range(12):
    try:
        with io.open(F, "w", encoding="utf-8", newline="") as fh:
            fh.write(out)
        break
    except OSError:
        time.sleep(3)

with io.open(F, encoding="utf-8") as fh:
    back = fh.read()
a = back.count(MARKER)
b = back.count("(#%d)" % n)
c = back.count("(#31)")
print("marker %r occurrences=%d (want 1)" % (MARKER, a))
print("(#%d) occurrences=%d (want 1)" % (n, b))
print("(#31) CONTROL occurrences=%d (want >=1; 0 means unreadable)" % c)
print("bytes %d -> %d (+%d)" % (len(src), len(back), len(back) - len(src)))
print("APPEND=%s" % ("PASS" if (a == 1 and b == 1 and c >= 1) else "FAIL"))
if not (a == 1 and b == 1 and c >= 1):
    raise SystemExit(1)
