"""PREREG_V8F_LAUNCH_INTENT (DRAFT v2): prove that the outcome table, the dose cascade and NEXT-LAUNCH are complete and exclusive, that the table's cell mapping is DAG-ORC's, and that the v2 per-seed composition over the two pairs maps onto it.

python enumerate_cells_li.py > enumerate_cells_li.out     (exit 0 iff every check below passes)

It is `raw/2026-10-27-dag-orc-draft/enumerate_cells.py` with: section 1's `outcome_rows` UNCHANGED (the same eight literal predicates over the same 524,288 cells); the committed-next texts and the NEXT-CRAWL section
replaced by LAUNCH-INTENT's; and added checks: (8) the per-row cell counts equal the counts DAG-ORC's own `enumerate_cells.out` registers (so the mapping from conditions to outcome rows is identical), (9) the
dose rule and the K-SHARE band on hand-made edge cases, and, v2 (MM_REDESIGN_2026-10-28): (7) the bar is 2 * max(F, RF0) and the four crawl bars are reachable, (9) arm B at the share 1/4, and (10) the per-seed AND over the seed's two pairs.

What it proves, each over EVERY cell (no sampling):
  1. The 8 outcome rows of section 7, each coded as its own literal predicate, match EXACTLY ONE row in each of the 2 x 2^18 = 524,288 cells (controls x 5 guards x 2 seeds x R-CRAWL, R-MOVE, R-KEEP, R-NOBASE x 2 seeds).
  2. Every row has exactly one committed next, for arm A and for arm B.
  3. Adoption is reachable only with R-MOVE at BOTH seeds (H-ESTIM-SEED-1: a one-seed move adopts nothing).
  4. With adoption projected away, the (R-CRAWL, R-MOVE) x 2-seed guarded cells map exactly as PREREG_V8F_DAGGER2 Amendment 1 (FIXED 1 / PARTIAL 11 / NO EFFECT 4).
  5. The A -> B dose cascade is total over A's 8 rows x B trained or not x B's 8 rows.
  6. NEXT-LAUNCH is total over (M-LAUNCH moved at both seeds: 2) x (the outcome row sending it there: 4).
  7. The bar (v2: 2 * max(F, RF0), RF is not a term) and clear / fire predicates on hand-made edge cases (inclusive bar, direction, structural zero); the four crawl bars equal the registered values and lie below the CTRL - P10 gap at both seeds.
  8. The cell counts per row equal DAG-ORC's registered counts.
  9. The dose rule gives the registered share at the measured f (A 1/3, B 1/4, HI 1/2); the K-SHARE band is inclusive.
  10. (v2) A seed's predicate is the AND of its main-pair and replicate-pair values: all four combinations, every per-seed value reachable, and 200,000 seeded random pair-level assignments each match exactly one row.
"""
import itertools
import sys
from collections import Counter

SEEDS = (0, 1)
GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")
F_ADE = 0.1207          # F_M2b, m (MODEL_REGISTRY.md section 17, H-V8F-SEED-1)
HALF_F = 0.0604         # +F/2, the R-T0 and R-KEEP point bar, m
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


# ---------------------------------------------------------------- 1. the outcome rows, each a literal predicate (UNCHANGED from DAG-ORC's enumerate_cells.py)
def outcome_rows(ctrl, held, rc, rm, keep, nobase):
    """held[g][s], rc[s], rm[s], keep[s], nobase[s]: booleans. Returns {row: bool} exactly as section 7 writes it."""
    guards_held = all(held[g][s] for g in GUARDS for s in SEEDS)
    rm_both, rm_n, rc_both = all(rm), sum(rm), all(rc)
    adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)
    return {
        "REFUSED": not ctrl,
        "HARMS": ctrl and not guards_held,
        "FIXED / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,
        "FIXED / NOT ADOPTED": ctrl and guards_held and rm_both and rc_both and not adopt_ok,
        "PARTIAL / ADOPTED": ctrl and guards_held and rm_both and not rc_both and adopt_ok,
        "PARTIAL / NOT ADOPTED": ctrl and guards_held and rm_both and not rc_both and not adopt_ok,
        "PARTIAL / ONE SEED": ctrl and guards_held and rm_n == 1,
        "NO EFFECT": ctrl and guards_held and rm_n == 0,
    }


ROWS = list(outcome_rows(True, {g: {s: True for s in SEEDS} for g in GUARDS}, (1, 1), (1, 1), (1, 1), (1, 1)).keys())
ADOPTING = {"FIXED / ADOPTED", "PARTIAL / ADOPTED"}

NEXT_A = {
    "REFUSED": "fix the failing control and re-read; no outcome row is reported",
    "HARMS": "A is not adopted; read B (the lower dose) under this same table (train B first, with its own PI go)",
    "FIXED / ADOPTED": "adopt A as the base; Read A re-run on A (DOMINANT there is the dominance claim)",
    "FIXED / NOT ADOPTED": "the lever is attributed but A cannot replace DAG-OACC: B (the lower dose) under this table, with its own PI go",
    "PARTIAL / ADOPTED": "adopt A as the base; the crawl rows that still fail R-CRAWL go to NEXT-LAUNCH on A's states",
    "PARTIAL / NOT ADOPTED": "the lever is attributed but A cannot replace DAG-OACC: B (the lower dose) under this table, with its own PI go",
    "PARTIAL / ONE SEED": "not adopted and never called an effect; NEXT-LAUNCH on DAG-OACC states",
    "NO EFFECT": "the along-track weight on launch samples is not the crawl's lever at this dose; NEXT-LAUNCH on DAG-OACC states",
}
NEXT_B = dict(NEXT_A)
NEXT_B.update({
    "HARMS": "B is not adopted; the launch weight cannot be dosed under the guards (A and B both HARMS); NEXT-LAUNCH on DAG-OACC states",
    "FIXED / ADOPTED": "adopt B as the base; Read A re-run on B (DOMINANT there is the dominance claim)",
    "PARTIAL / ADOPTED": NEXT_A["PARTIAL / ADOPTED"].replace("A's", "B's").replace("adopt A", "adopt B"),
    "FIXED / NOT ADOPTED": "the lever is attributed but B cannot replace DAG-OACC either (A and B both fail adoption); NEXT-LAUNCH on DAG-OACC states",
    "PARTIAL / NOT ADOPTED": "the lever is attributed but B cannot replace DAG-OACC either (A and B both fail adoption); NEXT-LAUNCH on DAG-OACC states",
})

count = Counter()
n_cells = 0
dag2_proj = {}
for bits in itertools.product((True, False), repeat=1 + 10 + 8):
    ctrl = bits[0]
    gb = bits[1:11]
    held = {g: {s: gb[2 * i + s] for s in SEEDS} for i, g in enumerate(GUARDS)}
    rc, rm, keep, nobase = bits[11:13], bits[13:15], bits[15:17], bits[17:19]
    r = outcome_rows(ctrl, held, rc, rm, keep, nobase)
    hit = [k for k, v in r.items() if v]
    n_cells += 1
    if len(hit) != 1:
        fails.append(f"cell {bits}: matches {hit}")
        continue
    row = hit[0]
    count[row] += 1
    # semantic pins, computed independently of outcome_rows: a guard failing at EITHER seed is HARMS; a row that adopts
    # needs R-MOVE at both seeds AND R-KEEP and R-NOBASE at both seeds
    any_guard_fails = any(not held[g][s] for g in GUARDS for s in SEEDS)
    check((row == "HARMS") == (ctrl and any_guard_fails), f"HARMS iff a guard fails at >= 1 seed: {bits}")
    adopts = row.endswith("/ ADOPTED")
    check(adopts == (row in ADOPTING), f"an adopting row outside the registered set: {row}")
    # 3. adoption only with R-MOVE at both seeds
    if adopts:
        check(all(rm) and all(keep) and all(nobase), f"adoption without R-MOVE / R-KEEP / R-NOBASE at both seeds: {bits}")
    if ctrl and not any_guard_fails and all(rm) and all(keep) and all(nobase):
        check(adopts, f"a guarded cell with R-MOVE, R-KEEP and R-NOBASE at both seeds that does not adopt: {bits}")
    # 4. projection on the DAGGER2 cells (controls pass, every guard held)
    if ctrl and all(gb):
        proj = "FIXED" if row.startswith("FIXED") else "PARTIAL" if row.startswith("PARTIAL") else row
        key = (int(rc[0]), int(rc[1]), int(rm[0]), int(rm[1]))
        prev = dag2_proj.setdefault(key, proj)
        check(prev == proj, f"projection not a function of (R-CRAWL, R-MOVE) at {key}: {prev} vs {proj}")

check(n_cells == 2 ** 19, f"cell count {n_cells}")
check(set(count) == set(ROWS), f"a row is unreachable: {set(ROWS) - set(count)}")
check(set(NEXT_A) == set(ROWS) and set(NEXT_B) == set(ROWS), "a row has no committed next")


# the DAGGER2 Amendment-1 table, verbatim from raw/2026-10-27-dagger2-amend1/enumerate_cells.py ("new")
def dagger2_a1(c0, c1, m0, m1):
    cr, mv = (c0, c1), (m0, m1)
    both = [cr[i] and mv[i] for i in range(2)]
    fixed, pa, ne = all(both), all(mv) and not all(cr), not any(mv)
    return "FIXED" if fixed else "PARTIAL" if (pa or sum(mv) == 1) else "NO EFFECT" if ne else None


dag2_counts = Counter(dagger2_a1(*k) for k in dag2_proj)
moved_vs_dag2 = [k for k in dag2_proj if dagger2_a1(*k) != dag2_proj[k]]
check(len(dag2_proj) == 16, f"projection covers {len(dag2_proj)} of 16 DAGGER2 cells")
check(not moved_vs_dag2, f"cells that differ from DAGGER2 A1: {moved_vs_dag2}")


# ---------------------------------------------------------------- 5. the A -> B dose cascade (section 8)
NOT_ADOPTED_SENDS_TO_B = {"HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED"}


def cascade(a_row, b_trained, b_row):
    if a_row not in NOT_ADOPTED_SENDS_TO_B:
        return ("A", a_row, "B, if trained, is reported beside as dose-response and decides nothing")
    if not b_trained:
        return ("A", a_row, "queue B (the lower dose) through its gate, with its own PI go; its read is this table, unchanged")
    return ("B", b_row, NEXT_B[b_row])


casc = Counter()
for a_row, bt, b_row in itertools.product(ROWS, (True, False), ROWS):
    who, row, nxt = cascade(a_row, bt, b_row)
    check(bool(nxt), f"cascade has no next at {(a_row, bt, b_row)}")
    casc[(who, row)] += 1
check(sum(casc.values()) == len(ROWS) * 2 * len(ROWS), "cascade not total")
check(all(NEXT_A[r].count("B (the lower dose)") == 1 for r in NOT_ADOPTED_SENDS_TO_B), "every row that sends the programme to B names B in A's next")
check(all("B (the lower dose)" not in NEXT_A[r] for r in ROWS if r not in NOT_ADOPTED_SENDS_TO_B), "no other row names B")


# ---------------------------------------------------------------- 6. NEXT-LAUNCH (section 9)
SENDS_TO_NEXT = ("HARMS_B", "PARTIAL / ONE SEED", "NO EFFECT", "PARTIAL / ADOPTED")


def next_launch(mlaunch_moved, sender):
    """mlaunch_moved: M-LAUNCH (X - CTRL, mean v_own - v_GT at ticks 1-2 in the GT-launch windows) separated toward 0 at BOTH seeds. sender: the outcome row that sent the programme here."""
    if mlaunch_moved:
        return "LOOP-STATE: the plan's launch speed was repaired and the crawl rows did not follow; decompose the crawl by plan index in X's tick dumps (zero training, dev-box CPU), then PREREG the loop term it names"
    return "LAUNCH-INTENT-HI: the same lever at the half share (1 + W = (1 - f) / f), under this table, with its own PI go; if it also leaves M-LAUNCH in place, label weighting is not a lever on O's launch and every remaining lever is named blocked"


nl = {(m, s): next_launch(m, s) for m, s in itertools.product((True, False), SENDS_TO_NEXT)}
check(len(nl) == 8 and all(nl.values()), "NEXT-LAUNCH not total")
check(all(v.startswith("LOOP-STATE") for (m, s), v in nl.items() if m) and all(v.startswith("LAUNCH-INTENT-HI") for (m, s), v in nl.items() if not m), "NEXT-LAUNCH routing")
check(all("NEXT-LAUNCH" in NEXT_A[r] for r in ("PARTIAL / ONE SEED", "NO EFFECT", "PARTIAL / ADOPTED")), "the rows that send the programme on name NEXT-LAUNCH")
check(NEXT_B["HARMS"].count("NEXT-LAUNCH") == 1, "B's HARMS names NEXT-LAUNCH")


# launch routing (section 11): LAUNCH-INTENT launches when DAG-OACC is the base and no other lever has been adopted in front of it
def launch_route(base_is_dagoacc, other_adopted):
    if other_adopted:
        return "HOLD: the PREREG is re-based (amended) on the adopted base before any run, as DAG-ORC Amendment 1 did"
    return "LAUNCH as written" if base_is_dagoacc else "HOLD: the base is not DAG-OACC; the control would be invalid"


lr = {(b, o): launch_route(b, o) for b, o in itertools.product((True, False), (True, False))}
check(lr[(True, False)].startswith("LAUNCH") and all(v.startswith("HOLD") for k, v in lr.items() if k != (True, False)), "launch routing")


# ---------------------------------------------------------------- 7. the bar and the clear / fire predicates (UNCHANGED from DAG-ORC's enumerate_cells.py)
def bar(row, rf0):
    """v2 (MM_REDESIGN_2026-10-28 section 3 item 2): bar_r = 2 * max(F_r, RF0_r); F_r = F_ADE for ADE6, 0 for every other row. RF is not a term (it is reported beside)."""
    return 2.0 * max(F_ADE if row == "ADE6" else 0.0, abs(rf0))


def worse(point, lo, hi, higher_better):
    """separated-worse in the row's own direction"""
    return lo > 0 if not higher_better else hi < 0


def clears_better(point, lo, hi, b, higher_better=False):
    """R-MOVE per row: separated-better AND the improvement >= bar (inclusive)"""
    if higher_better:
        return lo > 0 and point >= b
    return hi < 0 and -point >= b


def fires_new(arm_p10, arm_ctl, b, higher_better=False):
    """R-NEW per domain row: arm - P10 separated-worse AND arm - control separated-worse with the worsening >= bar (inclusive). Each argument is (point, lo, hi)."""
    p, lo, hi = arm_ctl
    worsening = -p if higher_better else p
    return worse(*arm_p10, higher_better) and worse(p, lo, hi, higher_better) and worsening >= b


check(bar("ADE6", 0.01) == 2 * F_ADE, "ADE6 bar must not fall below 2F")
check(bar("ADE6", 0.2) == 0.4, "ADE6 bar follows RF0 when RF0 exceeds F")
check(bar("M1", 0.004) == 0.008, "non-ADE bar = 2 * RF0")
check(bar("M1", -0.004) == 0.008, "the bar takes |RF0|")
check(bar("M1", 0.0) == 0.0, "a floor of exactly 0 gives bar 0 (clears iff separated)")
RF0_CRAWL = {"yaw-rate MAE": 0.5535, "curvature MAE": 0.0263, "L2 curvature MAE (GT>=5.2)": 0.0068, "M1 crawl share": 0.0048}       # P0_RF0.json (sha256 1608d70b...), RF0 of the four crawl rows
RF_CRAWL = {"yaw-rate MAE": 15.8553, "curvature MAE": 0.1324, "L2 curvature MAE (GT>=5.2)": 0.0870, "M1 crawl share": 0.0388}         # RF_REP.json (sha256 64c9444d...): beside, never a bar term
GAP_CTRL_P10 = {"yaw-rate MAE": (3.4726, 8.6844), "curvature MAE": (0.0854, 0.1352), "L2 curvature MAE (GT>=5.2)": (0.0443, 0.0529), "M1 crawl share": (0.0568, 0.0604)}   # DAGORC_READ.json read_a.CTRL_P10 (sha256 70a7a3a4...), s0 / s1
CRAWL_BAR_V2 = {"yaw-rate MAE": 1.1070, "curvature MAE": 0.0526, "L2 curvature MAE (GT>=5.2)": 0.0136, "M1 crawl share": 0.0096}                # MM_REDESIGN_2026-10-28 section 3 item 2
for lab, rf0v in RF0_CRAWL.items():
    b2 = bar(lab, rf0v)
    check(round(b2, 4) == CRAWL_BAR_V2[lab], f"{lab}: the v2 crawl bar is {b2}, registered {CRAWL_BAR_V2[lab]}")
    check(all(b2 < g for g in GAP_CTRL_P10[lab]), f"{lab}: the v2 bar {b2} is not below the CTRL - P10 gap {GAP_CTRL_P10[lab]} at both seeds (the decisive condition would be unreachable)")
    v1 = 2.0 * max(rf0v, RF_CRAWL[lab])                                  # the v1 bar, shown to sit above the gap
    check(any(v1 > g for g in GAP_CTRL_P10[lab]), f"{lab}: the v1 bar {v1} was not above the gap at either seed (the registered defect is not reproduced)")
check(clears_better(-0.014, -0.02, -0.008, 0.014), "the bar is inclusive")
check(not clears_better(-0.0139, -0.02, -0.008, 0.014), "below the bar does not clear")
check(not clears_better(-0.05, -0.09, 0.001, 0.014), "not separated does not clear")
check(clears_better(+0.05, 0.02, 0.08, 0.04, higher_better=True), "direction: higher-is-better row clears upward")
check(fires_new((6.25, 3.27, 9.14), (7.0, 2.0, 12.0), 6.0), "fires: worse vs P10, worse vs control beyond the bar")
check(fires_new((6.25, 3.27, 9.14), (6.0, 2.0, 12.0), 6.0), "the fire bar is inclusive")
check(not fires_new((6.25, 3.27, 9.14), (5.9999, 2.0, 12.0), 6.0), "just below the fire bar does not fire")
check(not fires_new((6.25, 3.27, 9.14), (1.36, -1.0, 3.7), 0.0), "does not fire: arm - control not separated")
check(not fires_new((6.25, 3.27, 9.14), (5.0, 2.0, 8.0), 6.0), "does not fire: arm - control below the floor")
check(not fires_new((0.5, -0.2, 1.2), (7.0, 2.0, 12.0), 6.0), "does not fire: arm - P10 not separated-worse")

# ---------------------------------------------------------------- 8. the cell counts equal DAG-ORC's registered counts (raw/2026-10-27-dag-orc-draft/enumerate_cells.out)
ORC_COUNTS = {"REFUSED": 262144, "HARMS": 261888, "FIXED / ADOPTED": 1, "FIXED / NOT ADOPTED": 15, "PARTIAL / ADOPTED": 3, "PARTIAL / NOT ADOPTED": 45, "PARTIAL / ONE SEED": 128, "NO EFFECT": 64}
check(dict(count) == ORC_COUNTS, f"the per-row cell counts differ from DAG-ORC's: {dict(count)}")

# ---------------------------------------------------------------- 9. the dose rule and the K-SHARE band
F_MEAS = 49618 / 306566           # tools/measure_f.py, f_measure.json (MEASURED)


def dose_w(f, share):
    """W such that launch samples (weight 1 + W) hold `share` of the along-track weight when a fraction f of the GT-valid samples is launch: f w / (f w + 1 - f) = share  =>  w = share (1 - f) / ((1 - share) f)."""
    return share * (1.0 - f) / ((1.0 - share) * f) - 1.0


def share_of(f, w):
    return (1.0 + w) * f / ((1.0 + w) * f + 1.0 - f)


def k_share_ok(share, target, tol=0.02):
    return abs(share - target) <= tol + 1e-12


wa, wb, wh = (float(f"{dose_w(F_MEAS, s):.6g}") for s in (1 / 3, 1 / 4, 1 / 2))
check(wa == 1.58926 and wb == 0.726175 and wh == 4.17852, f"dose at the measured f: {wa}, {wb}, {wh}")
check(abs(share_of(F_MEAS, wa) - 1 / 3) < 1e-6 and abs(share_of(F_MEAS, wb) - 1 / 4) < 1e-6 and abs(share_of(F_MEAS, wh) - 1 / 2) < 1e-6, "the dose rule does not give its share")
check(abs((1 - F_MEAS) / (3 * F_MEAS) - 1 - dose_w(F_MEAS, 1 / 4)) < 1e-12, "the closed form (1 - f) / (3 f) equals the general rule at 1/4 (arm B)")
check(abs((1 - F_MEAS) / (2 * F_MEAS) - 1 - dose_w(F_MEAS, 1 / 3)) < 1e-12, "the closed form (1 - f) / (2 f) equals the general rule at 1/3")
check(abs(dose_w(F_MEAS, F_MEAS)) < 1e-12, "a share equal to f is the unweighted run: W = 0")
check(k_share_ok(1 / 3 + 0.02, 1 / 3) and k_share_ok(1 / 3 - 0.02, 1 / 3) and not k_share_ok(1 / 3 + 0.0201, 1 / 3) and not k_share_ok(0.0, 1 / 3), "K-SHARE band is inclusive")
for s_ in (0.3314508172724677, 0.33072924299512985, 0.3255864741422861, 0.3280925644385709):          # the sampler simulation at seeds 0, 1, 10, 11 (tools/measure_f.py)
    check(k_share_ok(s_, 1 / 3), f"simulated share {s_} outside K-SHARE")
check(F_MEAS + 0.05 < 1 / 4 < 1 / 3 - 0.05, "arm B at the share 1/4 sits strictly between the unweighted share f and A's 1/3, each by more than 0.05 (a real lower dose)")
check(abs(1 / 6 - F_MEAS) < 0.01, "arm B at the v1 share 1/6 sat within 0.01 of the UNWEIGHTED share f (nearly a no-op): the reason B is 1/4")
SIM_ROWS = {0: (191203, 30727), 1: (191396, 30674), 10: (191333, 30068), 11: (191391, 30367)}            # tools/measure_f.py sampler simulation (f_measure.json), rows / launch rows at trainer seeds 0, 1, 10, 11
SIM_SHARE = {}
for nm_, w_, tgt_ in (("A", wa, 1 / 3), ("B", wb, 1 / 4), ("HI", wh, 1 / 2)):
    for sd_, (r_, n_) in SIM_ROWS.items():
        sh_ = (1 + w_) * n_ / ((1 + w_) * n_ + r_ - n_)
        SIM_SHARE[(nm_, sd_)] = sh_
        check(k_share_ok(sh_, tgt_), f"simulated share of arm {nm_} at trainer seed {sd_} = {sh_:.4f} outside {tgt_} +- 0.02")
check(round(SIM_SHARE[("B", 0)], 4) == 0.2484 and round(SIM_SHARE[("HI", 11)], 4) == 0.4941, "the simulated shares quoted in the PREREG (B 0.2484 ... HI ... 0.4941)")

# ---------------------------------------------------------------- 10. v2: a seed's predicate is the AND over the seed's two pairs; the table is evaluated on the per-seed booleans
import random

PREDICATES = ("R-MOVE", "R-CRAWL", "R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW", "R-KEEP", "R-NOBASE")
GUARDS_AS_PREDICATES = GUARDS


def seed_bool(main_pair, replicate_pair):
    """MM_REDESIGN_2026-10-28 section 3 item 3: each condition holds at seed s only when it holds on BOTH pairs of that seed."""
    return bool(main_pair and replicate_pair)


and_table = {(a, b): seed_bool(a, b) for a in (False, True) for b in (False, True)}
check(and_table == {(False, False): False, (False, True): False, (True, False): False, (True, True): True}, "the per-seed AND is not the conjunction")
check(set(and_table.values()) == {False, True}, "a per-seed value is unreachable")


def row_from_pairs(ctrl, pair_bits):
    """pair_bits[pred][s] = (main, replicate): the 9 predicates x 2 seeds x 2 pairs. -> {row: bool} (exactly one True)."""
    seed_vals = {p: {s: seed_bool(*pair_bits[p][s]) for s in SEEDS} for p in PREDICATES}
    held = {g: seed_vals[g] for g in GUARDS}
    per_seed = lambda p: tuple(seed_vals[p][s] for s in SEEDS)      # noqa: E731  outcome_rows takes (s0, s1) tuples for these four
    return outcome_rows(ctrl, held, per_seed("R-CRAWL"), per_seed("R-MOVE"), per_seed("R-KEEP"), per_seed("R-NOBASE"))


rng = random.Random(20261028)
and_count = Counter()
N_AND = 200_000
for _ in range(N_AND):
    q = {p: (0.98 if p in GUARDS_AS_PREDICATES else 0.8) for p in PREDICATES}          # biased draws so that every row is reachable at 200,000 draws
    pb = {p: {s: (rng.random() < q[p], rng.random() < q[p]) for s in SEEDS} for p in PREDICATES}
    r_ = row_from_pairs(rng.random() < 0.9, pb)
    hit_ = [k for k, v in r_.items() if v]
    if len(hit_) != 1:
        fails.append(f"pair-level assignment matches {hit_}")
        continue
    and_count[hit_[0]] += 1
check(sum(and_count.values()) == N_AND, "a pair-level assignment is unplaced or double-placed")
check(set(and_count) == set(ROWS), f"the random pair-level draw reached only {sorted(and_count)}")
# a seed in which ONE pair misses is a miss
pb0 = {p: {s: (True, True) for s in SEEDS} for p in PREDICATES}
pb0["R-MOVE"][1] = (True, False)
check([k for k, v in row_from_pairs(True, pb0).items() if v] == ["PARTIAL / ONE SEED"], "R-MOVE true in the main pair at both seeds but false in one replicate pair is not ONE SEED")
pb1 = {p: {s: (True, True) for s in SEEDS} for p in PREDICATES}
pb1["R-ADE"][0] = (True, False)
check([k for k, v in row_from_pairs(True, pb1).items() if v] == ["HARMS"], "a guard failing in the replicate pair only is not HARMS")
pb2 = {p: {s: (True, True) for s in SEEDS} for p in PREDICATES}
check([k for k, v in row_from_pairs(True, pb2).items() if v] == ["FIXED / ADOPTED"], "all predicates true in both pairs at both seeds is not FIXED / ADOPTED")
AND_LINE = (f"a seed's predicate is the AND of its main-pair and replicate-pair values (all four combinations checked, both per-seed values reachable); "
            f"{N_AND:,} seeded random pair-level assignments (9 predicates x 2 seeds x 2 pairs, controls on or off) each matched exactly one of the 8 rows, and all 8 rows were reached; "
            f"a miss in either pair is a miss at the seed (three hand-made cases)")

# ---------------------------------------------------------------- report
print(f"cells enumerated: {n_cells:,} (controls 1 bit x 5 guards x 2 seeds x R-CRAWL / R-MOVE / R-KEEP / R-NOBASE x 2 seeds)")
print("cells per outcome row (each cell matches exactly one row):")
for k in ROWS:
    print(f"  {k:24s} {count[k]:>8,}")
print(f"per-row counts equal DAG-ORC's registered counts: {dict(count) == ORC_COUNTS}")
print(f"guarded cells (controls pass, every guard held): {sum(1 for _ in dag2_proj) * 16:,} = 16 (R-CRAWL, R-MOVE) x 16 (R-KEEP, R-NOBASE)")
print("projection of the guarded cells on (R-CRAWL s0, s1, R-MOVE s0, s1), adoption ignored:", dict(Counter(dag2_proj.values())))
print("PREREG_V8F_DAGGER2 Amendment 1 on the same 16 cells:                                 ", dict(dag2_counts))
print("cells that differ from DAGGER2 A1:", moved_vs_dag2)
print("adopting rows reachable only with R-MOVE at both seeds: checked on every cell")
print(f"dose cascade: {sum(casc.values())} (A row, B trained, B row) combinations, all with a next; final reader:")
for (who, row), n in sorted(casc.items()):
    print(f"  {who} {row:24s} {n:>4}")
print("NEXT-LAUNCH (M-LAUNCH moved at both seeds x the row that sent the programme here):")
for k, v in nl.items():
    print(f"  mlaunch_moved={k[0]!s:5s} from {k[1]:20s} -> {v.split(':')[0]}")
print("launch routing (base is DAG-OACC x another lever adopted in front):")
for k, v in lr.items():
    print(f"  base_is_dagoacc={k[0]!s:5s} other_adopted={k[1]!s:5s} -> {v.split(':')[0]}")
print(f"dose at the measured f = {F_MEAS:.6f}: W_A = {wa} (share 1/3), W_B = {wb} (share 1/4), W_HI = {wh} (share 1/2); share at W_A {share_of(F_MEAS, wa):.6f}, at W_B {share_of(F_MEAS, wb):.6f}; the UNWEIGHTED share is f = {F_MEAS:.6f}")
print("committed next, arm A:")
for k in ROWS:
    print(f"  {k:24s} {NEXT_A[k]}")
print("committed next, arm B (read only after A: HARMS or NOT ADOPTED):")
for k in ROWS:
    print(f"  {k:24s} {NEXT_B[k]}")
print("v2 bar: 2 * max(F, RF0); crawl bars", {k: round(bar(k, v), 4) for k, v in RF0_CRAWL.items()}, "each below the CTRL - P10 gap at both seeds; the v1 bars", {k: round(2.0 * max(v, RF_CRAWL[k]), 4) for k, v in RF0_CRAWL.items()}, "sit above it at >= 1 seed")
print("simulated realised shares (ESTIMATED, trainer seeds 0 / 1 / 10 / 11):", {nm: [round(SIM_SHARE[(nm, sd)], 4) for sd in SIM_ROWS] for nm in ("A", "B", "HI")})
print("per-seed composition:", AND_LINE)
print("pair-level random draw, rows reached:", dict(and_count))
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
