"""Promote PREREG_V8F_LAUNCH_INTENT from DRAFT v2 (sha256 aec8156a...) to PRE-REGISTERED, NOT RUN: the title, the status block, the four
open section-0 cells (items 1, 4, 6, 8) and a 'met' note on section 13 precondition 4. Nothing else changes (PROMOTION_DIFF.diff).
python promote_prereg.py"""
import difflib
import hashlib
import os

D = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_v2/"
SRC = D + "PREREG_V8F_LAUNCH_INTENT.md"
OUT = D + "promote/PREREG_V8F_LAUNCH_INTENT.md"
src = open(SRC, "rb").read()
assert hashlib.sha256(src).hexdigest().startswith("aec8156a4a24"), "the draft moved"
t = src.decode("utf-8")
assert "\r" not in t


def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)


rep("# PREREG_V8F_LAUNCH_INTENT (DRAFT v2): ", "# PREREG_V8F_LAUNCH_INTENT: ")
rep("**Status: DRAFT v2 for the Master Mind's promotion.** ",
    "**Status: PRE-REGISTERED, NOT RUN (2026-10-28).** Promoted by the Master Mind from DRAFT v2 (sha256 `aec8156a…`): the open"
    " items of §0 (1, 4, 6, 8) are confirmed in place and nothing else changes (`raw/2026-10-28-launch-intent-v2/PROMOTION_DIFF.diff`)."
    " Box-2 time: the PI's go, *\"Yes, queue it\"* (2026-10-28), for the four runs of §4, each through its launch gate. ")
rep("Proposed claim row `H-V8F-LAUNCH-INTENT-1` (the Master Mind registers it; this draft edits no register).",
    "Claim row `H-V8F-LAUNCH-INTENT-1`, registered at promotion.")
rep("| **OPEN.** This draft builds the registered object as an **additive term** (§2). Confirm the reading, or name another. |",
    "| **CONFIRMED** by the Master Mind at promotion (2026-10-28): the **additive term** of §2 is the registered object. It adds to"
    " O's unchanged 30-knot loss; it replaces no term. The flag is landed at `b09ac89f` (default 0.0 = off). |")
rep("| **OPEN.** Rule Zero: an outcome that ends at \"not a lever\" must register what follows. Edit freely. |",
    "| **CONFIRMED** by the Master Mind at promotion (2026-10-28): §11 as drafted. M-LAUNCH moved routes to LOOP-STATE (zero"
    " training); not moved routes to LAUNCH-INTENT-HI (W_HI 4.17852, the pair design, its own PI go); if both are exhausted, §11"
    " names the blockers. |")
rep("| **OPEN** (§13). |",
    "| **CONFIRMED** by the Master Mind at promotion (2026-10-28): **met.** `PREREG_V8F_DAG_ORC`'s arm A read HARMS (not adopted)."
    " Its Amendment 3 (A3-2026-10-28, the PI's routing) sends NEXT-CRAWL here and does not train B. No read adopted another base."
    " So DAG-OACC is the base and the main pair's control (§13 precondition 4). |")
rep("(MEASURED: neither `summary.json` has a `launch` key) (§9). | **OPEN.** |",
    "(MEASURED: neither `summary.json` has a `launch` key) (§9). | **CONFIRMED** by the Master Mind at promotion (2026-10-28):"
    " readings (a)–(e) as written. |")
P4 = ("If a read adopts another base before A's gate, the same.")
rep(P4, P4 + " **Met at promotion (2026-10-28):** arm A of DAG-ORC read HARMS, and `PREREG_V8F_DAG_ORC` Amendment 3 routes"
    " NEXT-CRAWL to LAUNCH-INTENT without training B. No read adopted another base, so the base is DAG-OACC.")
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert w not in t.lower(), w
assert "**OPEN" not in t, [l[:60] for l in t.split("\n") if "**OPEN" in l]
open(OUT, "wb").write(t.encode("utf-8"))
diff = "".join(difflib.unified_diff(src.decode("utf-8").splitlines(True), t.splitlines(True),
                                    "DRAFT_v2/PREREG_V8F_LAUNCH_INTENT.md", "promoted/PREREG_V8F_LAUNCH_INTENT.md", n=0))
open(D + "promote/PROMOTION_DIFF.diff", "wb").write(diff.encode("utf-8"))
print("promoted", hashlib.sha256(t.encode("utf-8")).hexdigest(), "diff lines", diff.count("\n"))
