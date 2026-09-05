"""Fold two things into the successor doc: the RULE ZERO next arms, and the sibling module.

(a) The two conditioning arms this stream ran (0 GPU), one admissible and one NOT, with the
    control that stopped the inadmissible one from being quoted.
(b) ⛔ ESCALATION: `stack/tanitad/refs/feasible_decode.py` (13,672 B) and
    `stack/tests/test_feasible_decode.py` (7,791 B) exist ONLY in the frozen clone
    C:/Users/Admin/refcv4b_repo -- `git cat-file -e HEAD:...` says ABSENT from the repo while
    the same probe reads refc.py PRESENT, so it is a genuine absence, not a failed read. They
    were written minutes ago by a LIVE sibling stream, so this document ESCALATES them rather
    than sweeping them into this package's commit.
"""
import io
import os

os.chdir("C:/Users/Admin/kingate")
p = "SUCCESSOR_FEASIBILITY_AWARE_DECODE.md"
s = io.open(p, encoding="utf-8").read()

a = "## 5. What this successor is NOT"
b = """## 4c. RULE ZERO's next arms — both run, one admissible, and the control that stopped the other

`CLAUDE.md`'s RULE ZERO (2026-09-05) requires that a refuted arm leave behind the NEXT arm.
§4b's diagnosis was *"the bank is speed-blind"*, so the next arm is **condition the vocabulary**.
Two forms of that were run here at **zero GPU**.

### ARM 1 — similarity scaling `bank_v0 = bank * (v0 / v_ref)` — ADMISSIBLE, and NEGATIVE

MEASURED (`raw/v0_conditioned_bank.json`, 400 windows × 128 candidates, `v_ref` = 10.0 m/s from
`refc.py:361`; C1 and C2-OBJECT both pass). At λ = 0:

| | fixed bank | scaled bank | delta |
|---|---|---|---|
| `off_reach` | 0.7822 | 0.6770 | **−0.1052** |
| `envelope` | **0.0156** | **0.2421** | **+0.2265 (15.5×)** |
| `peak_g` | 0.4808 g | 0.5498 g | +0.0690 |
| `oracle_ade` | 1.0985 m | 1.1571 m | +0.0586 |

⛔ **A similarity scale is the WRONG OPERATION, and the arithmetic says why before the data
does.** Under a spatial scale `s` on a fixed time grid, speed scales by `s` and curvature by
`1/s`, so **`lat_acc = v²κ` scales by `s`** — and this corpus needs scales up to **3.63×**, so
the friction load is multiplied by up to 3.63. It buys a little reach and pays for it in exactly
the currency the successor exists to save.

### ARM 2 — a control-space roll (constant-speed re-integration) — ⛔ INADMISSIBLE, NOT QUOTED

The right operation is `roll_bank`'s: re-integrate the anchor's own control content at the
window's speed, which changes **arc length** without multiplying the friction load. A stand-in
was implemented (`raw/v0_roll_bank_probe.py`) and it produced a **spectacular-looking** result —
at λ = 0, `off_reach` 0.7822 → 0.0000, `envelope` 0.8879 → 0.0009, `oracle_ade` 1.0985 → 0.4715.

⛔ **It is not reported as a finding, because its known-answer control FAILED.** `C3b` rolls a
**synthetic constant-speed arc**, whose answer is known exactly, and the round-trip error is
**3.17e-02 m** against a 1e-4 m bar: the integrator takes the heading of the segment containing
the current arc position and steps a full `v·dt` along it, which is first-order wrong on a curve.
`C3a` (the near-constant-speed anchor subset, 34 of 128) reads 0.75 m and also fails.
⭐ **That column would have been the headline of this stream.** Two earlier grid errors were
caught the same way and fixed (`ARM_HORIZONS` is uniform at 0.5 s only over the first four slots,
and anchor slot 0 is at t = 0.5 s rather than the origin — both are the `dt`/scope family). The
control is what separates "the vocabulary is the answer" from "my integrator is first-order".

⇒ **What would settle it, and what blocks it:** the repo's own tested integrator,
`refc_sampler.roll_controls` (present in the repo tree, **absent from the frozen clone this
stream ran in**), applied to anchors that carry a **control sequence**. refcv3's
`core.decoder.anchors` are WAYPOINTS; **refcv4b's anchors declare `control_units` via
`anchor_meta.py`**. ⇒ **The conditioning experiment belongs on refcv4b, not on refcv3**, and it
needs the un-frozen tree. That is the blocker, named, and it is not a compute blocker.

## 4d. ⛔ ESCALATION — a feasibility-aware decode ALREADY EXISTS AND IS NOT IN THE REPO

MEASURED 2026-09-05 17:15Z, by a positive assertion with a same-breath control:

```
git cat-file -e HEAD:stack/tanitad/refs/feasible_decode.py  -> does not exist in HEAD
git cat-file -e HEAD:stack/tanitad/refs/refc.py             -> PRESENT   (control)
```

while the frozen clone `C:/Users/Admin/refcv4b_repo` holds
**`stack/tanitad/refs/feasible_decode.py` (13,672 B)** and
**`stack/tests/test_feasible_decode.py` (7,791 B)**, both written minutes earlier. Its own
docstring opens on the same 8.56× measurement this document is built on and states its thesis
as *"after `project_feasible`, an envelope-violating or Kamm-violating path is not merely
penalised, it is UNREPRESENTABLE — a structural zero, not an estimate"*.

⭐ **So the successor is already being BUILT by a sibling stream, and this document's job
changes from proposing it to supplying its evidence.** Concretely, §4b and §4c hand that module
three things it needs and does not yet have:

1. **The frontier it must beat** — `raw/displacement_frontier.json`: on the *shipped* vocabulary
   there is no knee, so a projection that only shrinks displacement cannot win.
2. **The operation NOT to use** — a similarity scale multiplies `lat_acc` by `s`; it must roll,
   not scale.
3. **The guard its arms must carry** — `off_reach` reported beside `envelope`, because on this
   bank they are one trade, and a projection that fixes the envelope by shortening paths will
   read as a win while making the fan unreachable.

⛔ **This is an INTEGRATION ESCALATION, not a merge request written into a README** (the failure
mode that cost 10 days once already). The two files live on **one disk** and are absent from
git. They are **not** committed by this package, because they are a live sibling's in-progress
work and sweeping them under this stream's message is the documented git-hygiene failure. ⇒ The
owning stream must stage them, and until it does they are a single-disk deliverable.

## 5. What this successor is NOT"""
assert a in s
s = s.replace(a, b, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("successor doc updated")
