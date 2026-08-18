# DUPLICATE DUMPS in `taniteval/results/` — pairs, canonical names, evidence

**Created 2026-08-18 (results-hygiene pass).** Machine-readable form:
[`dump_exclusions.json`](dump_exclusions.json) — **any census / ranking / correlation / mean
"over arms" must drop the excluded rows first.** No dump is deleted: both non-canonical files are
referenced by banked artifacts, and one is a designed experiment control.

## The two pairs — both are CASE (i) double-banks of one eval, NEITHER is a mislabel

Verdict grammar: (i) = one model banked under two names, every number under either name describes
the model it claims; (ii) = a phantom name whose eval never ran. **Both pairs are (i).** No number
ever published under any of these four names describes a different model than its label says.

### Pair 1 — `windows_refa-dynin-30k.pt` (CANONICAL) ≡ `windows_overfit_refa-dynin-30k.pt` (excluded)

- **MEASURED 2026-08-18:** `torch.equal` TRUE on every key (`pred/gt/cv/eid/speed/head_deg/wp_steps`).
  Files are NOT byte-identical (sha256 `f159bc60…` 96 190 B vs `22cecd30…` 96 278 B) — the delta is
  the torch-zip member prefix embedding each file's own basename (8 chars × 11 members = 88 B).
  Two separate `torch.save` calls of bit-identical tensors.
- **Mechanism (from source):** `taniteval/refa_overfit_driver.py:31-38` — the overfit-curve driver's
  30k point clones the canonical registry entry (`BASE = … key == "refa-dynin-30k"`) and evaluates
  the identical checkpoint `/root/models/refa-dynin-30k/ckpt.pt`, which is the canonical arm's own
  ckpt (`taniteval/taniteval/registry.py:110-112`). Same ckpt + same cached features + deterministic
  eval ⇒ bit-identical output banked twice (curve label vs registry key).
- **Canonical name:** `refa-dynin-30k` — the registry arm. The other three `overfit_refa-dynin-{5k,15k,20k}`
  dumps are distinct checkpoints and stay in every census.

### Pair 2 — `windows_refc-xl-30k.pt` (CANONICAL) ≈ `windows_refc-v12-identity.pt` (excluded)

- **MEASURED 2026-08-18:** all shared keys `torch.equal` EXCEPT `pred`: max |Δ| **7.62939e-06**
  over 881×4×2 (GPU re-run epsilon — two models as different as their names could not agree to 1e-5).
  `windows_refc-xl-30k.pt` additionally carries `method` (`refc anchored-diffusion decode
  (mode=diffusion, steps=2, 256 anchors, argmax-conf anchor trajectory, nav=follow)`, written by
  `taniteval/taniteval/refc_eval.py:191`); the identity dump has no `method` key (writer:
  `stack/scripts/refc_v12_eval.py:244`).
- **Mechanism (from the experiment's own report):** `refc-v12-identity` is the v1.2 learned-rescorer
  experiment's **zero-init / identity-selection control** — *"produced by `stack/scripts/refc_v12_eval.py`
  with a **zero-init** head"* whose full_set 0.47144 the report itself labels *"the frozen REF-C
  selection — the number to beat"*
  (`…/Benchmarks & Eval/Research/2026-07-20-refc-v12-learned-rescorer.md:33-38`; guard:
  *"identity-at-init holds at every K"*). Pod3's own result JSON confirms
  (`stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/refc-v12-identity.json`:
  `rescorer.head_step 0`, encoder `"frozen refc-xl-30k"`).
- **Canonical name:** `refc-xl-30k` — the registry arm / standard decode. **Keep the identity dump:**
  it is the paired same-fan control for `refc-v12` / `refc-v12-k16reg`; it is only excluded from
  censuses over *distinct arms*. (`refc-v12` itself is a DIFFERENT arm — same frozen decoder,
  different selection — and stays.)

## Discovery timeline (why this file exists)

| date | what happened |
|---|---|
| 2026-07-25 | `df32781a` banks all 27 dumps; jack blast-radius sweep publishes aggregates "over 27 arms" |
| 2026-07-26 | `DOC_CORRECTION_SWEEP.md:101-103` states BOTH pairs verbatim ("the same checkpoint under two keys … likewise duplicates") — prose only, nothing machine-readable |
| 2026-08-02 | LEADERBOARD §1b merges the refc pair into one ROW (`refc-v12-identity` / `refc-xl-30k`) but still counts it as 2 of "6 arms beat CTRV" |
| 2026-08-03 | sota-top3 `RESULTS.md:147-151` re-finds the refa pair by prediction hash ("26 unique prediction sets among 27 files"), flags "for whoever owns the bank" — the refc pair evades any hash check (ε-level differences) |
| 2026-08-18 | val40 distance-keeping panel re-derives BOTH from per-window values; this pass institutionalizes the exclusion |

**Root-cause class:** a duplication documented in prose but never encoded where censuses read —
every census re-derives its arm list from `glob("windows_*.pt")`, so a prose correction decays on
the next census. Same decay mechanism as the "2 of 36 features" count (pinned by test 2026-08-16
after four rots).

## What moves under deduplication (recomputed 2026-08-18 from the committed raw data)

| published claim | site(s) | deduplicated | verdict |
|---|---|---|---|
| jack bias "−6.67 %…+11.69 % over 27 arms" | CLAUDE.md, README.md:61, Paper:919/975, PROGRAM_OVERVIEW.md:282, MODEL_REGISTRY §0.3+§jack | range **UNCHANGED** (extremes `flagship-v16-ab-ft` +11.689 / `refc-xl` −6.669 are unique arms) | phrasing only: "27 dumps = 25 distinct arms" |
| "11 of 27 arms inflated, 16 deflated, none flat" | CLAUDE.md:37, Paper:975, MODEL_REGISTRY:2730 | **11 inflated, 14 deflated, none flat over 25 distinct arms** (both dup rows are deflated: −4.185 %, −2.914 %) | count correction |
| interval narrowing "1.107–3.100×, median 1.499× (27 arms)" | CLAUDE.md:40, Paper:924, PROGRAM_OVERVIEW:306, MODEL_REGISTRY:2721 | **UNCHANGED to 3 decimals** (dup rows 1.511 / 1.451 straddle the median; extremes unique) | phrasing only |
| jack paired contrasts | `jack_recompute.json["paired"]` | none of the 10 contrasts uses a duplicate name | unaffected |
| LEADERBOARD §1b "12 arms beat CV; 6 beat CTRV; 16 of 25 verdicts move" | `Benchmarks & Eval/LEADERBOARD.md:71-90` | **11 / 5 distinct arms**; the moved-verdict census counts 25 dump rows | count correction |
| sota-top3 E2c τ_b **0.7991** (Spearman 0.9209) over "22 comparable arms" | `…/2026-08-03-sota-top3-executed/RESULTS.md` §2.4 | **τ_b 0.7895, ρ 0.9173 over 20 distinct** — stays inside the pre-registered 0.7–0.9 band ⇒ **verdict INDETERMINATE unchanged** (its caveat (b) already named the refa pair) | quantified, no verdict change |
| "26 unique prediction sets among 27 files" | same report §2.4 | 26 unique hashes but **25 distinct models** — the refc pair differs at 7.6e-06 and evades hashing | precision note |
| file-census comments "27 committed dumps", "24 of 27 canonical eids", "3 affected, 24 clean" | `bench.py:197`, `runner.py:189`, `rollout.py:291/323`, `tests/test_eid_normalisation.py:39`, `dump_lead_join.py:57` | counts of FILES, still true as written | unaffected |

**No verdict, gate decision, ranking extreme, median, or sign changes anywhere.** The duplicates
were mid-pack, negative-bias rows; every load-bearing number survives. The corrections are counts
and phrasing ("arms" → "dumps / distinct arms").

*(Registry and CLAUDE.md phrasing fixes are routed via the 2026-08-18 results-hygiene report —
`MODEL_REGISTRY.md` is sibling-owned today and was not edited by this pass.)*
