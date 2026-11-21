# PREREG_V8F_TSWAP: how much of the launch crawl does T's own-state decision carry? (Master Mind, 2026-11-07, written before any TSWAP code or run exists)

## Why
* TOWN (`PREREG_V8F_TOWN.md`, `707e29f3`; result `raw/2026-11-07-v8f-town-result/`, row H-V8F-TOWN-1) read MIXED. T's longitudinal decision does not follow the car's own speed deficit (the slope is not separated at either seed). T's decisions are worse on the model's own states: T-lon macro-F1 own minus logged is -0.0831 / -0.0702, separated-worse at both seeds.
* TOWN's card and result name this read as the next step: a zero-training read with O fed T's LOGGED-state output at T1-kin. If it removes the crawl excess, the share that T carries is measured.

## Arms (zero training; C0 s0 / s1, TOWN's checkpoints; eval-141, 2,399 windows; T1-kin m = 10, own executed kinematics, logged cameras; 4 threads, as TOWN)
* **OWN:** the pinned T1-kin tool, unchanged.
* **TSWAP:** at every replan, every T output that O reads is taken from T run on the LOGGED state at the same time index (TOWN's twin batch: the own batch with only TOWN's `SWAP_KEYS` replaced by the logged clip's values). O's own inputs, S, the executor and the rollout state are unchanged.
* The list of T outputs that O reads is fixed from source in `IMPL_CHOICES_TSWAP.md`, frozen with its sha256 before the first run.
* TSWAP feeds a privileged signal (the logged state) and is a diagnostic, never a deployable arm.

## Statistics (paired TSWAP - OWN per seed over the same windows; episode-cluster bootstrap, B 2,000, seed 0)
* All four families, with the pinned T1-kin rows. STRATEGIC PAUSED (PI item 25).
* Named rows: M1, the launch-crawl share (executed speed 3 to 8 m/s); ADE6.
* **Crawl excess** of an arm = its M1 minus P10's M1, on the same windows.
* **T share** = (M1(OWN) - M1(TSWAP)) / (M1(OWN) - M1(P10)), as a bootstrap ratio with its interval. It is reported only where OWN's excess is separated above 0.

## Controls (each must read its known value, or the read is INVALID)
1. **Wrapper identity:** OWN equals the bare pinned tool at 4 threads, bit for bit, on every record.
2. **Tick-0 identity:** at the first replan of every window the own state equals the logged state. TSWAP's T outputs and plan therefore equal OWN's bit for bit there.
3. **Mutant:** a TSWAP whose twin is not swapped (twin := own) reproduces OWN bit for bit on every record.
4. **Swap is live:** on later replans, the T outputs that O reads differ between the arms on more than 0 rows (receipt counts reported).

## Outcomes, committed now
1. **T-CARRIES:** TSWAP - OWN is separated-better on M1 at both seeds, AND TSWAP's M1 row against P10 is not separated-worse at either seed (the excess is removed).
   * Next: T trained on its own states, the DT2 / RECOVER recipe on T's labels, under its own card.
2. **T-PARTIAL:** TSWAP - OWN is separated-better on M1 at both seeds, but the excess remains at a seed.
   * The T share is reported. Next: T trained on its own states, AND the twin read on O, each under its own card.
3. **T-NOT:** anything else.
   * The crawl does not sit in T's own-state decision. Next: the twin read on O (O fed the logged ego history), under its own card.
4. **INVALID:** a control fails.
* Every family row is reported as measured, whichever outcome holds. A TSWAP row that is separated-worse on any family is named in the result.

## Scope and evidence
* Tier T1-kin. This is a mechanism read, not a driving claim. One training run per seed: the two seeds answer the training-run question. Inference is deterministic at a fixed thread count.

## Log (dated, append-only after the freeze)
* 2026-11-07 13:11:01 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 3642 bytes above this log, sha256 `fc775ca1d3ebaadc792d546e639d7dcc01506cf4f4bb939fe5526f808fd53cf0`, before any code, run or statistic of this card exists.
