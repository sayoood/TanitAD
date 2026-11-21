## AMENDMENT 1: aggregate concordance of the search outputs (reported only)

**REPORTED ONLY. It is never gated and it changes no gate, bar, arm or verdict of `PREREG_L2_STEPB.md` (sha256 `c661588c952d1f8bf65474f9d6346cdb243eb8cf5e882101e040a9cd4f03172b`); Step B's analysis and verdict line stand exactly as sealed.**  Sealed text: `PREREG_L2_STEPB_AMENDMENT1.md` sha256 `a2e0c2fed1ebc46e9c85b7064070ada5c4fbf6525552010473b12a4126ca48f5`.

- **Why this section exists (INHERITED from the sealed amendment, not re-measured here):** `MM_RULING_L2_FRESH.md` / `PREREG_L2_FRESH.md` (sha256 `5006ed84...`) committed, before READ 1 was read, that if the paired dA3 upper CI were below 0 on both seeds the Step B amendment would report the aggregate concordance of the search outputs beside dPDMS; READ 1 fired that condition.  This section carries it out; it is a record, not a test.
- **Evidence class:** MEASURED (ours; the inputs and their sha256 are listed under Provenance).
- **Tier / stamp:** SEEN-SCENE (L1 amendment A1.3); the head is TOKEN-FRESH and LOG-SHARED on carve B (400 tokens disjoint from the 400 the head was fitted on, inside the SAME 24 logs: NOT a fresh-log test); SELECTION tier, label-only (not T1, not navtest); one training seed per head arm (H-ESTIM-SEED-1 open); the interval is 'another draw of LOGS' only; no capability claim
- **Estimator:** paired log-cluster bootstrap: whole LOGS resampled with the SAME draws for every quantity (stepA.boot_indices; B = 10,000, seed 20260927); a difference is the per-resample difference of the two ratios of sums.  It answers 'another draw of LOGS' and nothing else: not another head seed (H-ESTIM-SEED-1), not another search seed (the per-seed values are that spread).  (24 logs, B = 10,000, seed 20260927.)
- **What a pair is:** anchor = P2's pick (its BANKED label), candidate = the search output (its label); both scored by the ARM'S OWN scorer (6 heads, the pick's slot replaced inside the complete 64-set); an output identical to the pick contributes to no pair.  Terms are L1's `stepA.admission_terms`, unchanged (L1 code sha-pinned and re-verified).
- **Logits source:** `banked`.  Identity (passed): structural identities on the banked logits; model identity (re-score on CPU, same `score_fn` path, bit for bit) on L2s0: 16 tokens, L2s1: 16 tokens, P2HEAD: 16 tokens.
- **Definitions:** A3 = arm aggregate vs TRUE PDMS concordance (pairs with |dPDMS| >= 0.05; scorer tie = 0.5); A4 = of pairs whose aggregate rises >= 0.02: share whose true PDMS falls by >= Step A's 0.10; A5 = of the same pairs: share where true NC or DAC goes 1 -> 0.

### Pooled over the 3 search seeds: point [95 % CI] (pairs counted)

| arm | A3 concordance | A4 exploit rate | A5 NC/DAC-breach rate |
|---|---|---|---|
| L2-s0 | 0.7500 [0.0000, 1.0000] (n=12) | 0.0000 [0.0000, 0.0000] (n=37) | 0.0000 [0.0000, 0.0000] (n=37) |
| L2-s1 | 0.6250 [0.0000, 1.0000] (n=16) | 0.0526 [0.0000, 0.1829] (n=57) | 0.0000 [0.0000, 0.0000] (n=57) |
| P2HEAD | 1.0000 [1.0000, 1.0000] (n=14) | 0.0000 [0.0000, 0.0000] (n=50) | 0.0000 [0.0000, 0.0000] (n=50) |

n counts: A3 = pairs whose true PDMS differs by >= 0.05; A4 / A5 = pairs whose aggregate rises by >= 0.02.

### L2 minus P2HEAD, paired (pooled over the 3 search seeds): point [95 % CI]

| L2 arm | dA3 | dA4 | dA5 |
|---|---|---|---|
| L2-s0 - P2HEAD | -0.2500 [-1.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s1 - P2HEAD | -0.3750 [-1.0000, 0.0000] | 0.0526 [0.0000, 0.1829] | 0.0000 [0.0000, 0.0000] |

### Per search seed (the search-seed spread): point [95 % CI]

| arm | seed | A3 | A4 | A5 |
|---|---|---|---|---|
| L2-s0 | 20260927 | 0.7500 [0.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s0 | 20260928 | 0.7500 [0.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s0 | 20260929 | 0.7500 [0.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s1 | 20260927 | 0.6667 [0.0000, 1.0000] | 0.0526 [0.0000, 0.1818] | 0.0000 [0.0000, 0.0000] |
| L2-s1 | 20260928 | 0.7500 [0.0000, 1.0000] | 0.0526 [0.0000, 0.1818] | 0.0000 [0.0000, 0.0000] |
| L2-s1 | 20260929 | 0.5000 [0.0000, 1.0000] | 0.0526 [0.0000, 0.1818] | 0.0000 [0.0000, 0.0000] |
| P2HEAD | 20260927 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| P2HEAD | 20260928 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| P2HEAD | 20260929 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

Spread of the per-seed points (max - min over defined seeds): L2-s0 A3 0.0000, A4 0.0000, A5 0.0000; L2-s1 A3 0.2500, A4 0.0000, A5 0.0000; P2HEAD A3 0.0000, A4 0.0000, A5 0.0000.

### L2 minus P2HEAD, paired, per search seed: point [95 % CI]

| L2 arm | seed | dA3 | dA4 | dA5 |
|---|---|---|---|---|
| L2-s0 - P2HEAD | 20260927 | -0.2500 [-1.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s0 - P2HEAD | 20260928 | -0.2500 [-1.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s0 - P2HEAD | 20260929 | -0.2500 [-1.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| L2-s1 - P2HEAD | 20260927 | -0.3333 [-1.0000, 0.0000] | 0.0526 [0.0000, 0.1818] | 0.0000 [0.0000, 0.0000] |
| L2-s1 - P2HEAD | 20260928 | -0.2500 [-1.0000, 0.0000] | 0.0526 [0.0000, 0.1818] | 0.0000 [0.0000, 0.0000] |
| L2-s1 - P2HEAD | 20260929 | -0.5000 [-1.0000, 0.0000] | 0.0526 [0.0000, 0.1818] | 0.0000 [0.0000, 0.0000] |

### Pair counts

| arm | tokens x seeds | identical to the pick (excluded) | invalid anchor (dropped) | pairs | A3 counted | A3 concordant weight | rise pairs | rise and fall | rise and NC/DAC 1->0 |
|---|---|---|---|---|---|---|---|---|---|
| L2-s0 | 1200 | 868 | 0 | 332 | 12 | 9.0 | 37 | 0 | 0 |
| L2-s1 | 1200 | 818 | 0 | 382 | 16 | 10.0 | 57 | 3 | 0 |
| P2HEAD | 1200 | 882 | 0 | 318 | 14 | 14.0 | 50 | 0 | 0 |

### Provenance

- bundle `D:\Projects\TanitAD-artifacts\l2_fresh_stepA\B_token_disjoint\bundle\bundle.npz` sha256 `580b6cc395aa9074c75eb7a62fe99e8b493abd9ea5671be1a5765d8ae4fd673f` (400 tokens, 24 logs); carve `0629811ff5d0`; checkpoint `ea31db229621`; heads L2s0: `bded124e1322`, L2s1: `86f0b20f6ab0`, P2HEAD: `plain P2`.
- search files: L2s0 `711d60c6daf6`; L2s1 `015506523d53`; P2HEAD `390ec40b4721`.  labels: L2s0 `f6efacafa8a8`; L2s1 `3ec197ef38fd`; P2HEAD `f2677f7c1932`.
- script `amend1_concordance.py` sha256 `516c8c014cca`; L1 code stepA.py `05d182c7cf1b`, tts_cem.py `9dea96038d18`, real_scorer.py `05c4b4fb771a`, label_perturbations.py `9e2caca122d1`.

