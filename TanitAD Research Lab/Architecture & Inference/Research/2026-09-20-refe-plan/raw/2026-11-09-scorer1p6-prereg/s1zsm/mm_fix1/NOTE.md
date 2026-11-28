# S1ZSM MM fix 1: G18 windows sized for speed-neutral pairs (2026-11-10, before any S1ZSM run)

**Change (speed-neutral path only; S1Z's path keeps 16 and 400):** with `--twin-speed-neutral`, G18's live share window is `TWIN_SN_PREFLIGHT_MICRO = 128` micro-batches (was 16) and the probe-batch search limit is `TWIN_SN_PROBE_SEARCH_MAX = 1000` (was 400). Six single-occurrence edits in `scorer_finetune.py`, each with its reversal appended to `s1zsm_layer.REVERSALS`; `peel()` still returns the tip trainer blob `6a16d35c1818422ff514e97c4b292c32dde81795` exactly (checked). Diff against the agent's version: `mm_fix1_vs_agent.patch`.

**Why (descriptive, navtrain FIT, `s1zsm/why/sn_risk.py`; Monte Carlo over random micro-batches of 8 sets):**

| chance a CORRECT run is refused | S1Z rule | speed-neutral rule |
|---|---|---|
| per-set share with a kept pair (NC, DAC, TTC) | 0.0133, 0.0240, 0.0736 | 0.0079, 0.0201, 0.0494 |
| live window of 16 micro-batches has a head at 0 | 0.22 | **0.40** |
| live window of 128 micro-batches has a head at 0 | 0.0000 | 0.0005 |
| probe search (all heads in one micro-batch) misses within 400 | 0.0002 | 0.012 |
| same within 1,000 | 0.0000 | 0.0000 |

FIT's NC share under the S1Z rule (0.0133) matches S1Z's measured training share (0.0138), so FIT is a fair proxy. The card (section 2) says "0 on any head is a blocker" and sets no window, so this is an implementation constant chosen before any run, not a bar change. Cost: about one minute of no-grad forwards per preflight run (main plus five mutants) on the L40. The probe search stops at the first hit (expected about 91 micro-batches).

**Test edit.** `test_s1zsm_preflight.py` section B pinned the old live window (16 micro-batches = 32 sets, 64 pairs, 96 band pairs). It now asserts `TWIN_SN_PREFLIGHT_MICRO == 128`, `TWIN_PREFLIGHT_MICRO == 16` and `micro_batches == 50`. The fixture loader holds 50 micro-batches of 2 sets, more than 16 and fewer than 128, so 50 proves the speed-neutral window is in force. It also asserts 100 sets, 200 pairs, kept F = kept S = 100 and 300 band pairs. S1Z's own preflight test still pins 16. The other test changes are the agent's (see `../HANDOFF.md`). Results: `test_results_mm.txt` (12 of 12 scripts pass).
