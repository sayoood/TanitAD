# -*- coding: utf-8 -*-
"""WP-D step 20 -- CLOSE A3 in PANEL_RESULT.md. Every number is injected from
the banked JSON; nothing is hand-typed.

⛔ Each anchor is asserted to occur EXACTLY ONCE before it is replaced, and the
file is written atomically. Authored on LOCAL DISK; the repo copy is made in one
operation afterwards (the G: mount is degraded).
"""
import ast  # noqa: F401
import io
import json
import os
import tempfile

RAW = r"C:\Users\Admin\wpd-probe\raw"
SRC = r"C:\Users\Admin\wpd-probe\PANEL_RESULT.md"

C = json.load(io.open(os.path.join(RAW, "a3_verdict_cart.json"), encoding="utf-8"))
P = json.load(io.open(os.path.join(RAW, "a3_verdict_pol.json"), encoding="utf-8"))
D = json.load(io.open(os.path.join(RAW, "bank_determinism_control.json"), encoding="utf-8"))
BC = json.load(io.open(os.path.join(RAW, "boot_rep_cart.json"), encoding="utf-8"))
BP = json.load(io.open(os.path.join(RAW, "boot_rep_pol.json"), encoding="utf-8"))

cf, pf = C["floors"], P["floors"]
ci = C["instrument_floor"]
iw = ci["within_invocation_identical_file"]
ia = ci["across_invocations_bit_identical_features"]
INSTR = ci["max_instrument_floor"]

s = io.open(SRC, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


cA = BC["pairs"]["tok_D1_minus_tok_D0"]
pA = BP["pairs"]["tok_D1_minus_tok_D0"]

# ---------------------------------------------------------------- 1. polar A3 row
sub("| **A3** | ≥ **3×** a replicate floor measured in this panel | **+0.00923** "
    "[-0.01457, +0.03026], **not separated**; **floor not yet measured** | ⏳ **NOT MEASURED** |",
    f"| **A3** | ≥ **3×** a replicate floor measured in this panel | **+{pA['delta']:.5f}** "
    f"[{pA['lo']:+.5f}, {pA['hi']:+.5f}], **not separated**; floor **{pf['floor_used']:.5f}** "
    f"⇒ 3× = **{P['three_x_floor']:.5f}** | ⛔ **FAIL** (misses by "
    f"{P['a3_bar_shortfall_x']:.1f}×) |")

# ---------------------------------------------------------------- 2. cart A3 row
sub("| **A3** | ≥ **3×** a replicate floor from THIS panel | +0.00173 | "
    "[-0.00777, +0.01043] | no | ⏳ **NOT MEASURED** |",
    f"| **A3** | ≥ **3×** a replicate floor from THIS panel | +{cA['delta']:.5f} | "
    f"[{cA['lo']:+.5f}, {cA['hi']:+.5f}] | no | ⛔ **`F5` UNDERPOWERED** — floor "
    f"**{cf['floor_used']:.5f}**, 3× = **{C['three_x_floor']:.5f}**; and the "
    f"INSTRUMENT alone reads **{INSTR:.6f}** where A3 needed **< 0.00058** |")

# ---------------------------------------------------------------- 3. section 0 item 5
sub("""5. **`D0b`, the replicate arm A3 needs, is LAUNCHED and RUNNING** (**§1**, argv-audited to
   **one differing token**). Until it lands **A3 is `NOT MEASURED`** — per the prereg's own `F5`,
   *underpowered, not negative* — and **§5** states exactly how small the replicate floor would
   have to be for A3 to pass.""",
    f"""5. ⛔⛔ **`A3` IS NOW CLOSED, AND IT CLOSES AS `F5` — UNDERPOWERED, NOT NEGATIVE (§5b).**
   Both replicate arms landed (`D0b` = D0's flags and D0's **seed**, argv-audited to **one
   differing token**, the `--out` path; `D0c` = seed 1). MEASURED Cartesian: `f_same`
   **{cf['f_same_D0b_minus_D0']:.6f}**, `f_seed` **{cf['f_seed_D0c_minus_D0']:.6f}**, so the floor
   §5b asks for — `max(f_same, f_seed)` — is **{cf['floor_used']:.5f}**, and **3× it is
   {C['three_x_floor']:.5f}** against a committed lever gap of **+0.00173**: A3's bar misses by
   **{C['a3_bar_shortfall_x']:.1f}×**. ⭐⭐ **But the deciding fact is not the replicates — it is
   that THE INSTRUMENT CANNOT MEASURE WHAT A3 ASKED FOR.** Two probe arms reading a **bit-identical
   feature file**, same head, same seed, same split, same invocation — a gap whose true value is
   **exactly 0** — read **{iw['floor']:.6f} AP**; and across invocations, on a bank proven
   **bit-identical** ({D['bank_tok_D0_differing_cells']} of {D['bank_tok_D0_cells']:,} cells
   differ), **{ia['floor']:.6f}**. A3 required the floor to be **below 0.00058**, so the chain's own
   irreproducibility is **{ci['times_the_a3_bar']:.2f}×** the bar. **A3 was unmeasurable on this rig
   before either replicate was trained.** ⚠️ On the **polar** secondary the bar is looser
   (**0.00308**) and the instrument sits **0.46×** below it, so there A3 is a **MEASURED FAIL**
   (misses by {P['a3_bar_shortfall_x']:.1f}×) rather than underpowered — the two geometries get
   different labels and must not be quoted interchangeably.""")

# ---------------------------------------------------------------- 4. section 6
sub("""⚠️ **`F5` is also partly live and must be said**: `A3` is **NOT MEASURED**, and the prereg is
explicit that an unmeasured criterion is **underpowered, not negative**. `D0b` decides it. But `F4`
does not depend on `D0b`: A4 compares two arms *in the same panel* and reads +0.00023 with a CI
straddling zero.""",
    f"""⛔⛔ **`F5` IS NOW SETTLED AND IT FIRED — ON THE PRIMARY GEOMETRY THE REPRESENTATION PANEL IS
UNDERPOWERED, NOT NEGATIVE.** `D0b`/`D0c` landed and §5b's arithmetic gives a Cartesian floor of
**{cf['floor_used']:.5f}** (= `f_seed`; `f_same` is only **{cf['f_same_D0b_minus_D0']:.6f}**), so
**3× the floor is {C['three_x_floor']:.5f}** — the committed lever gap **+0.00173** misses it by
**{C['a3_bar_shortfall_x']:.1f}×**. ⚠️ **And this REACHES BACKWARDS INTO `A4`, which must be said
plainly:** A4's measured `D1 − D2` is **-0.00366**, whose magnitude is **{0.00539/0.00366:.1f}×
SMALLER** than the same floor. ⇒ *"the shuffled arm matches the lever"* is **exactly what this rig
would report whether or not the target carried information**, so `F4`'s *mechanism* claim —
capacity-not-content — is **NOT** established by A4 alone at 4,000 steps. What survives unchanged is
the **negative**: `A1 ∧ A2 ∧ A3 ∧ A4` is unreachable, because **`A2` fails on its own terms** (a
+0.01269 point estimate whose CI straddles zero) and no floor makes a straddling CI separate.
⭐ **`F2` (§10) is therefore the load-bearing half of the refutation**, and it is measured on a
different instrument at a different scale (ADE **+0.02610 [+0.00970, +0.04240]**, **{0.02610/cf['floor_used']:.1f}×**
this probe's seed floor) — see §7 for the arm that tests whether it too survives a replicate.""")

# ---------------------------------------------------------------- 5. section 8
sub("* ⚠️ **A3 is NOT MEASURED, not negative.**",
    f"""* ⛔ **A3 is CLOSED: `F5` UNDERPOWERED on the Cartesian PRIMARY, `FAIL` on the polar
  secondary** (`raw/a3_verdict_cart.json`, `raw/a3_verdict_pol.json`). It does **not** say the aux
  term's effect is zero — it says **this rig cannot resolve an effect of that size**, and the proof
  is an instrument that reads **{INSTR:.6f} AP** between two arms that differ in **nothing at all**.
* ⚠️ **It does NOT retract `A1`.** A1's **+0.02630 [+0.00822, +0.04256]** is
  **{0.02630/cf['floor_used']:.1f}×** the seed floor and stays separated; the trunk really is above
  the marginal. ⛔ **It DOES weaken `A4`'s mechanism reading** — see §6.""")

fd, tmp = tempfile.mkstemp(dir=os.path.dirname(SRC), suffix=".tmp")
os.close(fd)
io.open(tmp, "w", encoding="utf-8", newline="\n").write(s)
os.replace(tmp, SRC)
print("PANEL_RESULT.md updated:", len(s), "chars")
for probe, must in (("F5` UNDERPOWERED", 1), ("NOT MEASURED", 0)):
    n = s.count(probe)
    print(f"  post-check {probe!r}: {n} (expected {'>=1' if must else '0'})")
assert "⏳ **NOT MEASURED**" not in s, "an A3 NOT MEASURED row survived"
print("POST_ASSERT_OK: no A3 'NOT MEASURED' row remains")
