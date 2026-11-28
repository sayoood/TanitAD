# PREREG_V8F_FLATDAG (DRAFT): does the hierarchy beat flat once BOTH get own-state supervision?

Drafted 2026-10-27 by the Master Mind. It was written **before** any row of `PREREG_V8F_DAG_REREAD.md` (`bdf8d561`) was seen, and is banked
in `raw/2026-10-27-flatdag-draft/`. Claim row `H-V8F-FLATDAG-1` (created when the draft is promoted).

**When it runs.** It runs if and only if `PREREG_V8F_DAG_REREAD` Read B-2 reads **ESTABLISHED**, which is that row's committed next. Under B-2
NOT ESTABLISHED it is parked. Box 2 needs the PI's go.

**The question.** B-2 compares DAG, which had a DAgger fine-tune, with FLAT, which had none. This arm gives FLAT the same treatment: its own
bank and the same recipe. The DAG − FLAT-DAG contrast then varies only the hierarchy's interface.

## The arms

* **FLAT-DAG** = `M2b_s{seed}_flat.pt` (sha256 `8496934a…` / `6fdfc9b0…`) fine-tuned 3,000 steps with DAG's recipe:
  * `--dagger-p 0.1 --dagger-bank <FLAT bank> --dagger-seed <seed> --ego-perturb-p 0 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2`;
  * FLAT's `--o-interface cut` is kept.
* **FLAT-DAG-0** = the same argv with `--dagger-p 0`.
* **The FLAT bank** = `PREREG_V8F_DAGGER` §1–2 verbatim, with FLAT_s{seed} as the generator (PLAIN, the same rollout configuration, the same
  4,512 TRAIN clips, ticks 1–2) and the same controls K-B0…K-B5 and COVERAGE ≥ 0.5.
* ⚠ **Build check.** The trainer's `--o-interface cut` and `--dagger-*` flags must compose. The launch gate's G-DVB has to show both the cut
  (zero gradient on `e_code`; O invariant to S and T) and the swap reached (share 0.1 ± 0.02) in the same run. Until that is shown, this
  arm is not launched (a launch blocker, CLAUDE.md 2026-09-26).

## Rows, all at T1-kin (DAG's decisive settings), paired episode-cluster bootstrap, B = 2,000, per seed

| row | definition | bar |
|---|---|---|
| **PRIMARY H-T1** | ADE6, DAG − FLAT-DAG | ≤ −2F = −0.2415 m, separated, at both seeds |
| H-T1-HEADS | R-TLAT / R-TLON, DAG − FLAT-DAG (T's heads exist in both; FLAT's O does not read them) | beside |
| D-FLAT | ADE6, FLAT-DAG − FLAT-DAG-0: does DAgger help flat too? | beside, with the same R-ADE / R-T0 guards as DAGGER |
| families | Read A's rows of `PREREG_V8F_DAG_REREAD` §2, DAG − FLAT-DAG | beside; none may be separated-worse for the claim to read "beats" |

## Outcomes

| outcome | condition | committed next |
|---|---|---|
| **HIERARCHY BEATS FLAT** | H-T1 at both seeds, and no family row separated-worse | the first hierarchy-over-flat claim of v8F, at T1-kin: register it and signal the PI |
| **NOT ESTABLISHED** | otherwise, at either seed | B-2's margin was DAgger's, not the hierarchy's. The interface stays the defect; next is MIXO's committed T-side arm, under its own PREREG |
| **FLAT BETTER** | DAG − FLAT-DAG separated-worse at ≥ 1 seed | report; the hierarchy costs at T1-kin; the PI decides the architecture |
| **REFUSED** | any control fails | fix and re-run |

## Queue and cost (ESTIMATED)

* **Box 2:** 2 banks + 4 runs, ~6–7 h (by analogy with DAGGER).
* **Dev-box CPU:** 8 T1-kin rollouts and the reader.
* **Code:** the composition of the two flags in the gate (above), the chain's re-parametrisation, and a reader proved by mutation.
