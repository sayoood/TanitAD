"""PREREG_V8F_DAG_ORC (DRAFT): prove that the outcome table, the dose cascade and NEXT-CRAWL are complete and exclusive.

python enumerate_cells.py > enumerate_cells.out     (exit 0 iff every check below passes)

What it proves, each over EVERY cell (no sampling):
  1. The 8 outcome rows of section 6, each coded as its own literal predicate, match EXACTLY ONE row in each of the
     2 x 2^18 = 524,288 cells: controls (1 bit) x 5 guards x 2 seeds (10 bits) x R-CRAWL, R-MOVE, R-KEEP, R-NOBASE x 2 seeds (8 bits).
  2. Every row has exactly one committed next, for arm A and for arm B (section 6 and 7).
  3. Adoption is reachable only with R-MOVE at BOTH seeds (H-ESTIM-SEED-1: a one-seed move adopts nothing).
  4. With adoption projected away, the (R-CRAWL, R-MOVE) x 2-seed guarded cells map exactly as PREREG_V8F_DAGGER2 Amendment 1
     (FIXED 1 / PARTIAL 11 / NO EFFECT 4): the floors and the adoption split move no DAGGER2 cell.
  5. The A -> B dose cascade (section 7) is total over A's 8 rows x B trained or not x B's 8 rows.
  6. NEXT-CRAWL (section 8) is total over the nav probe (2) x DAG-OACC's crawl read (3).
  7. The bar and clear / fire predicates of section 5 on hand-made edge cases (inclusive bar, direction, structural zero).
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


# ---------------------------------------------------------------- 1. the outcome rows, each a literal predicate
def outcome_rows(ctrl, held, rc, rm, keep, nobase):
    """held[g][s], rc[s], rm[s], keep[s], nobase[s]: booleans. Returns {row: bool} exactly as section 6 writes it."""
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
    "HARMS": "A is not adopted; read B (p 0.1) under this same table (train B first if the PI dropped it from the queue)",
    "FIXED / ADOPTED": "adopt A as the base; Read A re-run on A (DOMINANT there is the dominance claim); then A + DAG-OACC's lever if DAG-OACC is adopted",
    "FIXED / NOT ADOPTED": "the lever is attributed but A cannot replace DAG: DAG-ORC-SWAP (DAG2's argv + this perturbation, control DAG2) under its own PREREG",
    "PARTIAL / ADOPTED": "adopt A as the base; the crawl rows that still fail R-CRAWL go to NEXT-CRAWL on A's states",
    "PARTIAL / NOT ADOPTED": "the lever is attributed but A cannot replace DAG: DAG-ORC-SWAP under its own PREREG",
    "PARTIAL / ONE SEED": "not adopted and never called an effect; NEXT-CRAWL on DAG states",
    "NO EFFECT": "the own-state channel is eliminated for the crawl (DAgger iteration 2 and the perturbation); NEXT-CRAWL on DAG states",
}
NEXT_B = dict(NEXT_A)
NEXT_B.update({
    "HARMS": "B is not adopted; the perturbation cannot be dosed under the guards (0.3 and 0.1 both HARMS); NEXT-CRAWL on DAG states",
    "FIXED / ADOPTED": "adopt B as the base; Read A re-run on B (DOMINANT there is the dominance claim); then B + DAG-OACC's lever if DAG-OACC is adopted",
    "PARTIAL / ADOPTED": NEXT_A["PARTIAL / ADOPTED"].replace("A's", "B's").replace("adopt A", "adopt B"),
    "FIXED / NOT ADOPTED": NEXT_A["FIXED / NOT ADOPTED"].replace("A cannot", "B cannot"),
    "PARTIAL / NOT ADOPTED": NEXT_A["PARTIAL / NOT ADOPTED"].replace("A cannot", "B cannot"),
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


# ---------------------------------------------------------------- 5. the A -> B dose cascade (section 7)
def cascade(a_row, b_trained, b_row):
    if a_row != "HARMS":
        return ("A", a_row, "B, if trained, is reported beside as dose-response and decides nothing")
    if not b_trained:
        return ("A", "HARMS", "queue B (p 0.1) through its gate; its read is this table, unchanged")
    return ("B", b_row, NEXT_B[b_row])


casc = Counter()
for a_row, bt, b_row in itertools.product(ROWS, (True, False), ROWS):
    who, row, nxt = cascade(a_row, bt, b_row)
    check(bool(nxt), f"cascade has no next at {(a_row, bt, b_row)}")
    casc[(who, row)] += 1
check(sum(casc.values()) == len(ROWS) * 2 * len(ROWS), "cascade not total")


# ---------------------------------------------------------------- 6. NEXT-CRAWL (section 8)
def next_crawl(nav_moves, oacc):
    """nav_moves: every row of C separated-better for nav-logged - nav-deadreckon at both seeds (zero-GPU probe N0).
    oacc: DAG-OACC's read of DAG-OACC - DAG on the crawl rows C: 'moved' (every row separated-better at both seeds), 'not', 'unread'."""
    if nav_moves:
        return "ROUTE-EXTENT: PREREG the dead-reckoner fix (advance nav through the logged route's manoeuvres by own arc); its read separates route extent from the logged-nav timing leak"
    if oacc == "moved":
        return "COMPOSE: DAG-OACC's acceleration term on the base, read on Read A under its own PREREG"
    return "LAUNCH-INTENT: O's knots 1-12 along-track loss up-weighted on GT-launch TRAIN samples, one flag, DAG init, 3,000 steps, own PREREG"


nc = {(n, o): next_crawl(n, o) for n, o in itertools.product((True, False), ("moved", "not", "unread"))}
check(len(nc) == 6 and all(nc.values()), "NEXT-CRAWL not total")
EXPECT_NC = {(True, "moved"): "ROUTE-EXTENT", (True, "not"): "ROUTE-EXTENT", (True, "unread"): "ROUTE-EXTENT",
             (False, "moved"): "COMPOSE", (False, "not"): "LAUNCH-INTENT", (False, "unread"): "LAUNCH-INTENT"}
for k, v in EXPECT_NC.items():
    check(nc[k].startswith(v + ":"), f"NEXT-CRAWL {k} should be {v}, is {nc[k].split(':')[0]}")



# launch routing (section 10): N0 is zero-GPU and MAY be read before A's gate
def launch_route(n0_read_before_gate, nav_moves):
    if not n0_read_before_gate:
        return "LAUNCH as written; N0 runs inside NEXT-CRAWL if an outcome row sends it there"
    if nav_moves:
        return "HOLD: the route-extent PREREG goes first (the crawl rides on the nav channel, which this lever does not touch)"
    return "LAUNCH as written; NEXT-CRAWL starts from its LAUNCH-INTENT / COMPOSE branch (N0 already read)"


lr = {(r, n): launch_route(r, n) for r, n in itertools.product((True, False), (True, False))}
check(all(lr.values()) and lr[(True, True)].startswith("HOLD") and lr[(False, True)].startswith("LAUNCH")
      and lr[(True, False)].startswith("LAUNCH"), "launch routing")


# ---------------------------------------------------------------- 7. the bar and the clear / fire predicates (section 5)
def bar(row, rf0, rf):
    """bar_r = 2 * max(F_r, RF0_r, RF_r); F_r = F_ADE for ADE6, 0 for every other row (its row-equivalent is RF0_r)."""
    return 2.0 * max(F_ADE if row == "ADE6" else 0.0, abs(rf0), abs(rf))


def worse(point, lo, hi, higher_better):
    """separated-worse in the row's own direction"""
    return lo > 0 if not higher_better else hi < 0


def clears_better(point, lo, hi, b, higher_better=False):
    """R-MOVE per row: separated-better AND the improvement >= bar (inclusive)"""
    if higher_better:
        return lo > 0 and point >= b
    return hi < 0 and -point >= b


def fires_new(arm_p10, arm_ctl, b, higher_better=False):
    """R-NEW per domain row (DAG - P10 not separated-worse there): arm - P10 separated-worse AND arm - control separated-worse
    with the worsening >= bar (inclusive). Each argument is (point, lo, hi)."""
    p, lo, hi = arm_ctl
    worsening = -p if higher_better else p
    return worse(*arm_p10, higher_better) and worse(p, lo, hi, higher_better) and worsening >= b


check(bar("ADE6", 0.01, 0.02) == 2 * F_ADE, "ADE6 bar must not fall below 2F")
check(bar("M1", 0.004, 0.007) == 0.014, "non-ADE bar = 2 * max(RF0, RF)")
check(bar("M1", 0.0, 0.0) == 0.0, "two bit-identical floors give bar 0 (clears iff separated)")
check(clears_better(-0.014, -0.02, -0.008, 0.014), "the bar is inclusive")
check(not clears_better(-0.0139, -0.02, -0.008, 0.014), "below the bar does not clear")
check(not clears_better(-0.05, -0.09, 0.001, 0.014), "not separated does not clear")
check(clears_better(+0.05, 0.02, 0.08, 0.04, higher_better=True), "direction: higher-is-better row clears upward")
check(fires_new((6.25, 3.27, 9.14), (7.0, 2.0, 12.0), 6.0), "fires: worse vs P10, worse vs control beyond the bar")
check(not fires_new((6.25, 3.27, 9.14), (1.36, -1.0, 3.7), 0.0), "does not fire: arm - control not separated")
check(not fires_new((6.25, 3.27, 9.14), (5.0, 2.0, 8.0), 6.0), "does not fire: arm - control below the floor")
check(not fires_new((0.5, -0.2, 1.2), (7.0, 2.0, 12.0), 6.0), "does not fire: arm - P10 not separated-worse")

# ---------------------------------------------------------------- report
print(f"cells enumerated: {n_cells:,} (controls 1 bit x 5 guards x 2 seeds x R-CRAWL / R-MOVE / R-KEEP / R-NOBASE x 2 seeds)")
print("cells per outcome row (each cell matches exactly one row):")
for k in ROWS:
    print(f"  {k:24s} {count[k]:>8,}")
print(f"guarded cells (controls pass, every guard held): {sum(1 for _ in dag2_proj) * 16:,} = 16 (R-CRAWL, R-MOVE) x 16 (R-KEEP, R-NOBASE)")
print("projection of the guarded cells on (R-CRAWL s0, s1, R-MOVE s0, s1), adoption ignored:", dict(Counter(dag2_proj.values())))
print("PREREG_V8F_DAGGER2 Amendment 1 on the same 16 cells:                                 ", dict(dag2_counts))
print("cells that differ from DAGGER2 A1:", moved_vs_dag2)
print("adopting rows reachable only with R-MOVE at both seeds: checked on every cell")
print(f"dose cascade: {sum(casc.values())} (A row, B trained, B row) combinations, all with a next; final reader:")
for (who, row), n in sorted(casc.items()):
    print(f"  {who} {row:24s} {n:>4}")
print("NEXT-CRAWL (nav probe x DAG-OACC crawl read):")
for k, v in nc.items():
    print(f"  nav_moves={k[0]!s:5s} oacc={k[1]:6s} -> {v.split(':')[0]}")
print("launch routing (N0 read before A's gate x N0 nav moves; nav moves is moot when N0 is unread):")
for k, v in lr.items():
    print(f"  n0_read={k[0]!s:5s} nav_moves={k[1]!s:5s} -> {v.split(':')[0]}")
print("committed next, arm A:")
for k in ROWS:
    print(f"  {k:24s} {NEXT_A[k]}")
print("committed next, arm B (read only when A is HARMS):")
for k in ROWS:
    print(f"  {k:24s} {NEXT_B[k]}")
print("bar / clear / fire predicate checks: 11")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
