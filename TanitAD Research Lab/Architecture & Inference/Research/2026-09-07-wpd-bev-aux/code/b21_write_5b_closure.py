# -*- coding: utf-8 -*-
"""WP-D step 21 -- write the §5b CLOSURE. Numbers injected from banked JSON.

The RECIPE in §5b is kept verbatim as the record of what was actually run; the
closure is appended under it so a reader sees the instruction and its result in
one place.
"""
import io
import json
import os
import tempfile

RAW = r"C:\Users\Admin\wpd-probe\raw"
SRC = r"C:\Users\Admin\wpd-probe\PANEL_RESULT.md"
J = lambda n: json.load(io.open(os.path.join(RAW, n), encoding="utf-8"))

C, P = J("a3_verdict_cart.json"), J("a3_verdict_pol.json")
D = J("bank_determinism_control.json")
BC, BP = J("boot_rep_cart.json"), J("boot_rep_pol.json")
IB = J("boot_instrument_cart.json")
AU = J("D0c_argv_audit.json")

cf, pf, ci = C["floors"], P["floors"], C["instrument_floor"]
iw, ia = ci["within_invocation_identical_file"], ci["across_invocations_bit_identical_features"]
ipair = IB["pairs"]["tok_D0dup_minus_tok_D0"]
ac, ap_ = BC["point_ap"], BP["point_ap"]

s = io.open(SRC, encoding="utf-8").read()
ANCHOR = ("⭐ **The floor will not be zero.** MEASURED already, before either arm finishes: at step "
          "50, D0\nreads `loss 51.63894 / traj 2.30219` and D0b reads `loss 51.65258 / traj "
          "2.30269` — two runs\ndiffering in **nothing** have already diverged, so this arm is "
          "measuring a real quantity and not an\nidentity.")
assert s.count(ANCHOR) == 1, s.count(ANCHOR)

CLOSURE = ANCHOR + f"""

---

### 5b.1 ⛔⛔ `A3` — CLOSED 2026-09-10. **`F5` UNDERPOWERED on the PRIMARY geometry.**

Both replicate arms landed on `tanitad-thor-wifi` (`summary.json` `done: true`, step 4000;
`D0b` 16,433 s, `D0c` 16,443 s) and were put through **this panel's own pipeline** —
`b1_bank_wpd.py` → `b2_probe_wpd.py --az-sign prog` → `b5_fast_boot.py --n-boot 2000` — in
**one bank and one probe invocation** alongside D0/D1/D2, so no arm is compared across instruments.

⭐ **The levers, audited rather than asserted** (`raw/D0c_argv_audit.json`): against D0's 59 argv
tokens, **`D0b` differs in ONE — the `--out` path — and in nothing else** (same seed 0);
**`D0c` differs in the seed alone** (0 → 1). Levers moved, excluding the output path:
**{AU['arms']['D0b']['n_diff_blocks_excluding_out_path']}** and
**{AU['arms']['D0c']['n_diff_blocks_excluding_out_path']}**.

| the floor, as §5b defines it | Cartesian (**PRIMARY**) | polar (secondary) |
|---|---|---|
| `AP(D0)` | {ac['tok_D0']:.6f} | {ap_['tok_D0']:.6f} |
| `AP(D0b)` — same flags, **same seed** | {ac['tok_D0b']:.6f} | {ap_['tok_D0b']:.6f} |
| `AP(D0c)` — **seed 1**, prereg §4 | {ac['tok_D0c']:.6f} | {ap_['tok_D0c']:.6f} |
| `f_same = |AP(D0b) − AP(D0)|` | {cf['f_same_D0b_minus_D0']:.6f} | {pf['f_same_D0b_minus_D0']:.6f} |
| `f_seed = |AP(D0c) − AP(D0)|` | **{cf['f_seed_D0c_minus_D0']:.6f}** | **{pf['f_seed_D0c_minus_D0']:.6f}** |
| **floor = `max(f_same, f_seed)`** | **{cf['floor_used']:.6f}** ← `f_seed` | **{pf['floor_used']:.6f}** ← `f_seed` |
| **3 × floor** | **{C['three_x_floor']:.6f}** | **{P['three_x_floor']:.6f}** |
| committed lever gap | **+0.00173** (§5b, verbatim) | +0.00923 (*derived* from §5's polar table) |
| **A3's own bar** | ⛔ **FAIL**, misses by **{C['a3_bar_shortfall_x']:.1f}×** | ⛔ **FAIL**, misses by **{P['a3_bar_shortfall_x']:.1f}×** |
| **verdict** | ⛔ **`F5` UNDERPOWERED** | ⛔ **`A3` FAIL** (measured, *not* underpowered) |

⚠️ **The two geometries get DIFFERENT labels and must not be quoted interchangeably.** The reason is
the bar, not the data: polar's required floor is **{P['prereg_literals_from_PANEL_RESULT_5b']['a3_floor_must_be_below']:.5f}**,
which the instrument (below) sits **0.46×** under, so polar *can* be measured and simply fails;
Cartesian's is **0.00058**, which it cannot.

#### ⭐⭐ The control that decides the LABEL — and it was not in §5b's recipe

§5b's recipe assumes the measurement chain is reproducible. **It is not**, and that is measured here
in two ways whose true value is **known to be exactly zero**:

| control | what differs | reads | true value |
|---|---|---|---|
| `tok_D0dup` vs `tok_D0`, **same invocation**, IDENTICAL feature file, same head, same seed, same split | **nothing at all** | **{iw['floor']:.6f}** AP, CI [{ipair['lo']:+.5f}, {ipair['hi']:+.5f}] | **0** |
| `AP(D0)` panel 2026-09-08 vs this run, on a bank proven **bit-identical** | **nothing at all** | **{ia['floor']:.6f}** AP | **0** |

⛔ **A3 required the replicate floor to be BELOW 0.00058 AP. The chain cannot reproduce ITSELF to
better than {ci['max_instrument_floor']:.6f} — {ci['times_the_a3_bar']:.2f}× that bar. A3 was
UNMEASURABLE on this rig before either replicate arm was trained**, and per the prereg's own `F5`
an unmeasurable criterion is reported as **UNDERPOWERED, never as a negative**.

⭐ **And the variance is localised, not merely observed.** Re-banking `D0` from the same trunk gave
**{D['bank_tok_D0_differing_cells']} of {D['bank_tok_D0_cells']:,} differing cells** (max |diff|
**{D['bank_tok_D0_max_abs_diff']:g}**) — the **ENCODER pass is bit-exact** — while the discriminating
half confirms the replicate is a real second run: `tok_D0b` differs from `tok_D0` in
**{100.0*D['tok_D0b_vs_D0_differing_cells']/D['bank_tok_D0_cells']:.1f} %** of cells. ⇒ **every bit of
this floor is the PROBE HEAD's own training**, and none of it is the trunk.

#### ⚠️ Both readings of §5b's F5 clause, reported side by side rather than chosen

§5b states the F4/F5 test two ways and on the Cartesian read **they disagree** — by 0.9 %:

* **(a) the LITERAL** — *"if the floor exceeds **0.00539**"*: floor **{cf['floor_used']:.5f}** ⇒
  F5 = **{C['f5_by_literal_threshold']}**.
* **(b) §5b's WORDS** — *"exceed the whole D0→D2 spread"*, with the floor measured **"IN THE SAME
  PANEL"**: this panel's D0→D2 spread is **{C['spread_D0_to_D2_same_panel']:.5f}** ⇒
  F5 = **{C['f5_by_same_panel_spread']}**.

⛔ **The verdict rests on NEITHER**, precisely because a 0.9 % knife-edge is not evidence. It rests
on the instrument floor above, which is independent of both. ⭐ The invariant statement, for a reader
who wants one number: **three arms differing in NOTHING BUT A SEED span
{C['spread_replicates_same_panel']:.5f} AP, while D0/D1/D2 span {C['spread_D0_to_D2_same_panel']:.5f}
— a ratio of {C['replicate_spread_over_lever_spread_x']:.2f}×.**

#### What would make A3 readable, and what it costs

1. ⭐ **Cheap and it fixes the INSTRUMENT, not the science:** average each arm's probe over **N head
   seeds**. The head is 67.2 k params and **43 s** on the dev-box 4060, so N = 10 costs
   **~45 min for six arms** and shrinks the {ci['max_instrument_floor']:.6f} instrument floor by
   ≈ √10 → **≈ 0.00045**, under the 0.00058 bar. ⛔ **This does not rescue A3** — it only stops the
   instrument being the binding constraint.
2. ⛔ **The binding floor is TRAINING, and it is expensive.** Resolving a **+0.00173** effect against
   a **{cf['floor_used']:.5f}** seed floor at A3's 3× bar needs the arm-mean SE down to ≈ 0.00058,
   i.e. **n ≈ (0.00534/0.00058)² ≈ 85 training runs per arm** at 4.7 h each ≈ **400 GPU-hours per
   arm**. ⇒ **Not worth spending.** The honest conclusion is that **a 4,000-step tiny rig cannot
   adjudicate a 0.0017 AP representation effect at all**, and A3 as written should not be re-run at
   this scale — it should be restated at a scale where the effect is larger, or retired.
3. ⭐ **What IS worth spending is on the OTHER half of the claim** — see §7.
"""

fd, tmp = tempfile.mkstemp(dir=os.path.dirname(SRC), suffix=".tmp")
os.close(fd)
io.open(tmp, "w", encoding="utf-8", newline="\n").write(s.replace(ANCHOR, CLOSURE))
os.replace(tmp, SRC)
out = io.open(SRC, encoding="utf-8").read()
print("WROTE §5b.1 closure ->", len(out), "chars")
for tok in ("5b.1", "UNMEASURABLE on this rig", "n ≈ (0.00534/0.00058)"):
    print(f"  contains {tok!r}: {out.count(tok)}")
