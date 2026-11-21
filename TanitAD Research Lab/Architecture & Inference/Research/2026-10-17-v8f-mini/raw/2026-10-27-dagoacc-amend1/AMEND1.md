
## Amendment 1 (A1-2026-10-27): before any DAG-OACC T1-kin rollout or value

Written 2026-10-27 by the Master Mind.

**State when written.**
* Both DAG-OACC runs are trained on box 2 and pulled to the dev box, with their T0 evals (`ZZV8FMINI-DONE` 21:51).
* No T1-kin rollout of DAG-OACC has run.
* No DAG-OACC metric value, T0 or T1-kin, has been opened by the Master Mind, the reader or the chain. The reader's dry run hashed the
  pulled files and did not read them.

**Why.** The reader's cell enumeration (`raw/2026-10-27-dagoacc-amend1/enumerate_cells_before.out`) shows two defects in §4. First, the table
leaves **255 of 524,288 cells** unplaced. Second, two clauses each admit two readings. This amendment fixes both before any row exists. Rows,
bars, controls, the order, the weight rule and W = 49.6969 do not change.

1. **The table is made complete.** LONGITUDINAL FIXED's condition reads: *PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not
   DOMINANT.*
   * The unplaced cells are cells where PRIMARY-ACC, KEEP and LON-CLEAR hold, no lateral or crawl row is separated-worse, and DOMINANT fails
     because D1 or D3 is short or a TACTICAL or STRATEGIC row is separated-worse.
   * In each of them the lever did its longitudinal job and the arm is not dominant, so they join LONGITUDINAL FIXED.
   * The result names, per seed, which of DOMINANT's clauses failed.
   * Clause (a) asked whether a lateral or crawl row had to be worse at one seed or at both. It then decides nothing and is dropped.
2. **DOMINANT needs the crawl rows to pass.** DOMINANT reads: *PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds; and Read
   A's crawl rows pass (L2 and M1 not separated-worse at either seed).*
   * Reason: this table's own LONGITUDINAL FIXED row names *"a lateral or crawl row still separated-worse"* as a state that is not dominant.
     Read A's D2 excludes M1, so as written a cell with M1 separated-worse reads DOMINANT before LONGITUDINAL FIXED is reached.
   * The programme's goal is dominant driving, and a launch crawl is a driving defect.
   * A cell with D1–D3 held and the crawl open therefore reads LONGITUDINAL FIXED, reported as *"D1–D3 held, crawl open"*.
   * Read A's own table (`PREREG_V8F_DAG_REREAD` §5) calls that state DOMINANT, CRAWL OPEN. This PREREG does not register a dominance claim
     while the crawl is open.
3. **LONGITUDINAL FIXED's committed next.** As written it named DAG2-OACC, *"as `PREREG_V8F_DAGGER2`'s FIXED / PARTIAL rows commit"*.
   * DAGGER2 read **HARMS** (`RESULT_V8F_DAGGER2.md`, landed at `d64750bd`, after this PREREG was written). HARMS is neither FIXED nor
     PARTIAL, so the condition of that route is not met.
   * The committed next reads: *the term is adopted and DAG-OACC becomes the base; next is the lever of the row still open.*
     * If the open rows are crawl rows (L2, curvature, yaw-rate, M1; the list `PREREG_V8F_DAG_REREAD` §5 uses), the lever is DAG-ORC
       (`PREREG_V8F_DAG_ORC.md`), re-based on DAG-OACC by an amendment of its own PREREG before any DAG-ORC data.
     * Otherwise, the failing family's lever, under its own PREREG.
4. **The reader codes this amendment.** Its enumeration must read **0 unplaced** cells, every row reached, and decide() equal to an
   independent table on every cell. The unamended outcome is computed beside, as a descriptive row that decides nothing.
