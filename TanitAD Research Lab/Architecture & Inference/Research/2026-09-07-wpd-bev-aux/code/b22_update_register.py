# -*- coding: utf-8 -*-
"""WP-D step 22 -- update `E-BEV-AUX-1` in GOALS_AND_CLAIMS.md for A3's closure.

⛔ The register is SHARED and other agents write it. This script therefore
(a) READS IT FRESH at apply time, retrying through the degraded G: mount with an
interleaved same-breath control, (b) makes only TARGETED replacements whose
anchors must each occur EXACTLY ONCE, and (c) writes the result to LOCAL DISK.
The copy back into the repo is a separate, single operation.

⚠️ It never rewrites the whole row and never touches any other claim's row, so a
sibling's concurrent edit elsewhere in the file survives.
"""
import io
import json
import os
import sys
import time

RAW = r"C:\Users\Admin\wpd-probe\raw"
REG = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Project Steering\GOALS_AND_CLAIMS.md"
CTRL = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\CLAUDE.md"
OUT = r"C:\Users\Admin\wpd-probe\GOALS_AND_CLAIMS.new.md"
J = lambda n: json.load(io.open(os.path.join(RAW, n), encoding="utf-8"))

C, P = J("a3_verdict_cart.json"), J("a3_verdict_pol.json")
D = J("bank_determinism_control.json")
BC = J("boot_rep_cart.json")
cf, pf, ci = C["floors"], P["floors"], C["instrument_floor"]
iw, ia = ci["within_invocation_identical_file"], ci["across_invocations_bit_identical_features"]

# ---- read FRESH, with an interleaved same-breath control ---------------------
src = None
for i in range(120):
    ok_c = False
    try:
        ok_c = len(io.open(CTRL, "rb").read()) > 1000
    except Exception:
        pass
    try:
        src = io.open(REG, encoding="utf-8").read()
        print(f"[read] OK on attempt {i}, {len(src)} chars, control_ok={ok_c}")
        break
    except Exception as e:
        if i % 10 == 0:
            print(f"[read] attempt {i}: control_ok={ok_c} target=FAIL {e!r}"[:110], flush=True)
        time.sleep(3)
if src is None:
    print("EXHAUSTED -- INCONCLUSIVE (the mount, not the file). Nothing written.")
    sys.exit(2)

assert src.count("| E-BEV-AUX-1 |") == 1, src.count("| E-BEV-AUX-1 |")


def sub(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:80])
    src = src.replace(old, new)


A3_CLOSED = (
 f"⛔⛔ **`A3` IS CLOSED (MEASURED 2026-09-10) AND IT CLOSES AS `F5` — UNDERPOWERED, NOT NEGATIVE, "
 f"ON THE CARTESIAN PRIMARY.** Both replicate arms landed on Thor (`summary.json` `done: true`, "
 f"step 4000; 16,433 s / 16,443 s) and were processed through the panel's OWN pipeline in ONE bank "
 f"and ONE probe invocation alongside D0/D1/D2. Argv-audited (`raw/D0c_argv_audit.json`): **`D0b` "
 f"differs from D0 in exactly ONE of 59 tokens — the `--out` path — same seed 0**; `D0c` differs in "
 f"the **seed alone**. MEASURED Cartesian floors: `f_same` **{cf['f_same_D0b_minus_D0']:.6f}**, "
 f"`f_seed` **{cf['f_seed_D0c_minus_D0']:.6f}** ⇒ §5b's `max(f_same, f_seed)` = "
 f"**{cf['floor_used']:.5f}**, so **3× = {C['three_x_floor']:.5f}** and the committed lever gap "
 f"**+0.00173** misses A3's bar by **{C['a3_bar_shortfall_x']:.1f}×**. "
 f"⭐⭐ **THE DECIDING FACT IS NOT THE REPLICATES — IT IS THAT THE INSTRUMENT CANNOT MEASURE WHAT A3 "
 f"ASKED FOR.** Two probe arms reading a **bit-identical feature file** (same head, same seed, same "
 f"split, same invocation — a gap whose TRUE value is **exactly 0**) read **{iw['floor']:.6f} AP** "
 f"[{iw['ci'][0]:+.5f}, {iw['ci'][1]:+.5f}]; and across invocations, on a bank proven bit-identical "
 f"(**{D['bank_tok_D0_differing_cells']} of {D['bank_tok_D0_cells']:,} cells differ**, max |diff| "
 f"{D['bank_tok_D0_max_abs_diff']:g}), **{ia['floor']:.6f}**. A3 required the floor **below "
 f"0.00058**, so the chain's own irreproducibility is **{ci['times_the_a3_bar']:.2f}× the bar**: "
 f"**A3 was unmeasurable on this rig before either replicate was trained.** ⭐ The variance is "
 f"LOCALISED: the ENCODER pass is **bit-exact**, `tok_D0b` differs from `tok_D0` in "
 f"**{100.0*D['tok_D0b_vs_D0_differing_cells']/D['bank_tok_D0_cells']:.1f} %** of cells (a real "
 f"second run, not an identity), so **every bit of the floor is the PROBE HEAD's own training**. "
 f"⚠️ On the **polar** secondary the required bar is looser (**0.00308**) and the instrument sits "
 f"**0.46×** below it, so there A3 is a **MEASURED FAIL** (misses by "
 f"{P['a3_bar_shortfall_x']:.1f}×) — ⛔ the two geometries carry DIFFERENT labels and must not be "
 f"quoted interchangeably. ⚠️ **§5b states its F4/F5 test two ways and on Cartesian they disagree "
 f"by 0.9 %** (literal 0.00539 ⇒ F5={C['f5_by_literal_threshold']}; same-panel D0→D2 spread "
 f"{C['spread_D0_to_D2_same_panel']:.5f} ⇒ F5={C['f5_by_same_panel_spread']}); the verdict rests on "
 f"NEITHER, but on the instrument floor, which is independent of both. Invariant form: **three arms "
 f"differing in NOTHING BUT A SEED span {C['spread_replicates_same_panel']:.5f} AP while D0/D1/D2 "
 f"span {C['spread_D0_to_D2_same_panel']:.5f} — {C['replicate_spread_over_lever_spread_x']:.2f}×.**")

# (a) the "A3 is running on Thor" clause
sub("⏳ **A3 is `NOT MEASURED`, NOT negative** (the prereg's own `F5`): the replicate arm "
    "`wpd-D0b-4k` is RUNNING on Thor.",
    A3_CLOSED)

# (b) the bar line inside the same row
sub("A3 +0.00173 [-0.00777, +0.01043], NOT separated ⇒ **NOT MEASURED** (no replicate floor yet);",
    f"A3 +0.00173 [-0.00777, +0.01043], NOT separated ⇒ **`F5` UNDERPOWERED** "
    f"(replicate floor MEASURED **{cf['floor_used']:.5f}**; 3× = {C['three_x_floor']:.5f}; and the "
    f"INSTRUMENT alone reads **{ci['max_instrument_floor']:.6f}** between two arms differing in "
    f"NOTHING, against a required **< 0.00058**);")

# (c) the summary clause
sub("⇒ **TWO of the four bars fail and a third is unmeasured**",
    "⇒ **TWO of the four bars fail and the third is UNDERPOWERED, not negative**")

# (d) ⛔ A4's mechanism reading is weakened by the same floor -- say it in the row.
sub("On the CARTESIAN target the shuffled arm is **AHEAD** (**0.0602** vs **0.0565**), so A4 fails "
    "on the POINT ESTIMATE alone (-0.00366 [-0.01173, +0.00397], NOT separated).",
    f"On the CARTESIAN target the shuffled arm is **AHEAD** (**0.0602** vs **0.0565**), so A4 fails "
    f"on the POINT ESTIMATE alone (-0.00366 [-0.01173, +0.00397], NOT separated). "
    f"⛔⛔ **BUT A3's MEASURED SEED FLOOR REACHES BACKWARDS INTO A4 AND MUST BE SAID: "
    f"|D1 − D2| = 0.00366 is {cf['floor_used']/0.00366:.1f}× SMALLER than the "
    f"{cf['floor_used']:.5f} floor**, so *\"the shuffled arm matches the lever\"* is exactly what "
    f"this rig reports whether or not the target carried information ⇒ **`F4`'s MECHANISM claim "
    f"(capacity-not-content) is NOT established by A4 alone at 4,000 steps.** What is untouched is "
    f"the NEGATIVE: `A2` fails on its own terms (a straddling CI, which no floor makes separate), "
    f"so SUCCESS remains unreachable — and **`F2` (the planner regression) is now the load-bearing "
    f"half of the refutation**, measured on a different instrument at "
    f"{0.02610/cf['floor_used']:.1f}× this probe's seed floor.")

io.open(OUT, "w", encoding="utf-8", newline="\n").write(src)
print("WROTE", OUT, len(src), "chars")
for tok in ("A3` IS CLOSED", "F5` UNDERPOWERED", "MECHANISM claim"):
    print(f"  contains {tok!r}: {src.count(tok)}")
assert "the replicate arm `wpd-D0b-4k` is RUNNING on Thor" not in src
print("POST_ASSERT_OK: the stale 'RUNNING on Thor' clause is gone")
