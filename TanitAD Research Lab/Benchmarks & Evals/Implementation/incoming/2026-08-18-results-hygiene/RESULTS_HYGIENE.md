# RESULTS HYGIENE — the two duplicate dump pairs are DOUBLE-BANKS, not mislabels; no verdict moves, three counts do

- **Date:** 2026-08-18 · **Discipline:** Benchmarks & Eval · **Status:** deliverables staged; PENDING orchestrator triage (retraction draft §5, routed phrasing fixes §6)
- **Evidence class:** `MEASURED (ours, this clone)` unless marked. Compute: dev-box CPU only, 0 GPU.
- **Scope:** the three items escalated by `…/2026-08-18-val40-lead-join/VAL40_LEAD_JOIN.md` §8.

## 0. A premise correction first

The tasking said the pairs were "byte-identical per the panel". **MEASURED: they are NOT byte-identical
as files** — all four sha256 differ. The panel's actual claim (§5: "identical per-window values; same
underlying rollouts banked under two keys") is what holds:

| file | sha256 (head) | bytes |
|---|---|---:|
| `windows_refa-dynin-30k.pt` (CANONICAL) | `f159bc60c33a…` | 96 190 |
| `windows_overfit_refa-dynin-30k.pt` (excluded) | `22cecd302360…` | 96 278 |
| `windows_refc-xl-30k.pt` (CANONICAL) | `e38a9b9e6f98…` | 96 221 |
| `windows_refc-v12-identity.pt` (excluded) | `2be17fd44a8d…` | 96 223 |

Pair 1: **`torch.equal` TRUE on every key** — the byte delta is exactly the torch-zip member prefix
embedding each file's own basename (8 chars × 11 members = 88 B). Pair 2: all shared keys bit-equal
EXCEPT `pred`, **max |Δ| = 7.62939e-06** (GPU re-run epsilon), plus a `method` key only in
`refc-xl-30k`. *(A hash check can therefore never catch pair 2 — which is exactly how sota-top3's
2026-08-03 hash sweep found pair 1 and missed pair 2.)*

## 1. ITEM 1a — provenance verdicts: BOTH pairs are case (i), one model banked twice; NEITHER is case (ii)

**No phantom arm exists; every number ever quoted under any of the four names describes the model its
label claims.** The two names differ only in which harness banked the same eval.

**Pair 1 — `refa-dynin-30k` ≡ `overfit_refa-dynin-30k`.** The overfit-curve driver's 30k point
*clones the canonical registry entry and evaluates the canonical arm's own checkpoint*:
`taniteval/refa_overfit_driver.py:31-38` (`("refa-dynin-30k", "/root/models/refa-dynin-30k/ckpt.pt")`,
`BASE = … key == "refa-dynin-30k"`) vs `taniteval/taniteval/registry.py:110-112` (same ckpt path).
Same ckpt + same cached frozen-DINO features + deterministic eval ⇒ bit-identical tensors saved twice
(curve label vs registry key). **Canonical: `refa-dynin-30k`** (the registry arm). The 5k/15k/20k
overfit dumps are distinct checkpoints and stay.

**Pair 2 — `refc-xl-30k` ≡ `refc-v12-identity`.** The v1.2 learned-rescorer experiment's
**zero-init / identity-selection control** over the SAME frozen decode: its own report reads *"produced
by `stack/scripts/refc_v12_eval.py` with a **zero-init** head"* and labels its full_set 0.47144
*"the frozen REF-C selection — the number to beat"*
(`…/Benchmarks & Eval/Research/2026-07-20-refc-v12-learned-rescorer.md:33-38`; test guard
*"identity-at-init holds at every K"* ibid.:406). Pod3's own JSON confirms
(`stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/refc-v12-identity.json`:
`rescorer.head_step 0`, encoder `"frozen refc-xl-30k"`). Writers identified: `refc_eval.py:191`
(the `method`-carrying canonical dump) vs `stack/scripts/refc_v12_eval.py:244`. Two models as
different as their names could not agree to 1e-5 over 881×4×2; ε-level disagreement = the same
frozen decode re-rolled. **Canonical: `refc-xl-30k`** (registry arm / standard decode); the identity
dump is KEPT as the paired within-experiment control for `refc-v12`/`refc-v12-k16reg`.
*(`refc-v12` itself — learned selection, head_step 2499 — is a genuinely distinct arm and is untouched.)*

**Git provenance:** all four files entered in ONE commit `df32781a` (2026-07-25, the taniteval
banking commit); `--follow --diff-filter=A` shows no earlier/other history — the duplication
happened pod-side at bank time, not in git.

**Prior art (this was KNOWN and never institutionalized):**
`…/incoming/2026-07-26-doc-correction-sweep/DOC_CORRECTION_SWEEP.md:101-103` states both pairs
verbatim ("the same checkpoint under two keys … likewise duplicates"); LEADERBOARD §1b (2026-08-02)
merges the refc pair into one row but counts it as 2; sota-top3 `RESULTS.md:147-151` (2026-08-03)
re-finds pair 1 by hash and flags "for whoever owns the bank". Three discoveries, zero encoding —
every census re-derives its arm list from `glob("windows_*.pt")`.

## 2. ITEM 1b — blast radius (each aggregate recomputed from its own committed raw data)

| published claim | site(s) | deduplicated recompute | changes? |
|---|---|---|---|
| jack bias "−6.67 %…+11.69 % over **27 arms**" | CLAUDE.md:36; README.md:61; Paper/TANITAD_PAPER.md:919,975; PROGRAM_OVERVIEW.md:282; MODEL_REGISTRY §0.3 + :2722 | range **UNCHANGED** — extremes `flagship-v16-ab-ft` (+11.689) and `refc-xl` (−6.669) are unique arms | phrasing: "27 dumps = **25 distinct arms**" |
| "**11 of 27 arms inflated, 16 deflated**, none flat" | CLAUDE.md:37; Paper:975; MODEL_REGISTRY:2730 | **11 inflated, 14 deflated, none flat over 25 distinct arms** (both dup rows deflated: −4.185 %, −2.914 %) | **count correction** |
| narrowing "1.107–3.100×, **median 1.499×** (27 arms)" | CLAUDE.md:40; Paper:924; PROGRAM_OVERVIEW:306; MODEL_REGISTRY:2721 | min/max/median **UNCHANGED to 3 decimals** (dup rows 1.511/1.451 straddle the median) | phrasing only |
| jack paired contrasts (10) | `jack_recompute.json["paired"]` | no contrast uses a duplicate name | no |
| LEADERBOARD §1b "**12 arms beat CV; 6 beat CTRV**; 16 of 25 verdicts move; best margin +0.0890" | `Benchmarks & Eval/LEADERBOARD.md:71-90` | **11 / 5 distinct arms**; margins unchanged; the refc pair was already one ROW (counted as 2) | **count correction** |
| sota-top3 E2c **τ_b 0.7991** (ρ 0.9209), "22 comparable arms", verdict INDETERMINATE (band 0.7–0.9) | `…/2026-08-03-sota-top3-executed/RESULTS.md` §2.4 | reproduced 0.7991 exactly, then dedup **τ_b 0.7895, ρ 0.9173 over 20 distinct** — stays in band ⇒ **verdict unchanged** (report's caveat (b) already named pair 1) | quantified; no verdict change |
| "26 unique prediction sets among 27 files" | ibid. | 26 unique **hashes**, **25 distinct models** — pair 2 evades hashing at 7.6e-06 | precision note |
| file-census comments ("27 committed dumps", "24 of 27 canonical", "3 affected, 24 clean") | `bench.py:197`, `runner.py:189`, `rollout.py:291/323`, `tests/test_eid_normalisation.py:39`, `dump_lead_join.py:57` | counts of FILES — true as written | no |
| val40 DK panel (27-row table) | `…/2026-08-18-val40-lead-join/` §5 | already flags both pairs and orders dedup-before-census | no |

**Bottom line: NO verdict, gate decision, ranking extreme, median, interval, or sign changes
anywhere.** The material corrections are three counts (11/16→11/14; 12→11; 6→5) and the census
phrasing "arms"→"dumps / 25 distinct arms". Everything above is recomputed in this pass from
`jack_recompute.json` and `raw/item2_progress_and_predictivity.json` (committed raw data); nothing
needs re-running on GPU.

## 3. ITEM 1c — the fix (both files new, in `taniteval/results/`)

- **`DUPLICATES.md`** — pairs, canonical names, mechanism evidence (file:line), hashes, discovery
  timeline, and the corrected-census table above.
- **`dump_exclusions.json`** — machine-readable, schema
  `{excluded_name, canonical_name, reason, evidence, sha256}` (+ `canonical_sha256`, census note).
  **No such list existed** (three probes: taniteval grep for exclusion/dedup manifests; results-dir
  listing; the val40 panel's own `score_val40_dumps.py:127` globs `windows_*.pt` unfiltered).
  Census authors: subtract this list or state you count dumps. **No dump deleted.**

## 4. ITEMS 2–3 — the two surgical doc fixes

- **ITEM 2** (`…/2026-08-18-dump-lead-wiring/DUMP_LEAD_WIRING.md`): struck *"no obstacle join for
  the VAL corpus exists anywhere yet"* in place (original visible) and added a dated correction
  blockquote naming the 2026-08-04 npz lead block (`val40_lead_block.npz`, states 270/551/60), why
  the four probes (scoped to `attach_lead`-consumable agents-JSONL) never saw it, and that the
  conclusion still holds for the jsonl-join form. Absence-claim class noted.
- **ITEM 3** (`wp_steps` vs `path_steps` dense-dump footgun): ⚠️ note added to
  `taniteval/taniteval/dump_lead_join.py` `attach_lead` docstring (emitted `wp_steps` is never read
  by `four_families._distance_keeping`; dense path ⇒ shape REFUSAL unless
  `win["lead"]["path_steps"]=[4,9,14,19]`; the tempting wrong fixes — truncation, sparse path with
  dense dt — silently mis-scale TTC ∝1/dt). Matching pointer added to `rollout.py`'s dense-path
  module docstring (it exists, lines 10–43 — so nothing was put in DUPLICATES.md for this).
  **Docstrings only; zero logic changes.**
- **VERIFY:** `pytest taniteval/tests -q -k "dump_lead or lead"` → **74 passed, 0 failed** (4.4 s,
  this box, after all edits).

## 5. DRAFT retraction entry (text only — orchestrator assigns the C-number; NOT appended to the log)

> ## C1?? — TWO BANKED DUMP PAIRS ARE ONE MODEL EACH: "27 arms" censuses counted 25 distinct arms — and the duplication was documented 23 days before it was institutionalized (2026-08-18, results hygiene)
>
> **MEASURED:** `windows_refa-dynin-30k.pt` ≡ `windows_overfit_refa-dynin-30k.pt` (`torch.equal`
> every key — the overfit driver's 30k point evaluates the canonical arm's own ckpt,
> `refa_overfit_driver.py:31-38` vs `registry.py:110-112`) and `windows_refc-v12-identity.pt` ≈
> `windows_refc-xl-30k.pt` (max|Δpred| 7.6e-06 — the v1.2 zero-init/identity control IS the frozen
> refc-xl-30k selection per its own report). **Neither is a mislabel; no phantom arm; no number under
> any name describes a different model than its label.** Corrections: (a) jack sweep "over 27 arms"
> → **27 dumps = 25 distinct arms**; "11 inflated, 16 deflated" → **11 / 14 over distinct arms**;
> bias range −6.67…+11.69 %, narrowing 1.107–3.100×, median 1.499× all UNCHANGED (extremes unique,
> dup rows straddle the median; no paired contrast touched). (b) LEADERBOARD §1b "12 beat CV / 6
> beat CTRV" → **11 / 5 distinct**. (c) sota-top3 E2c τ_b 0.7991 → **0.7895 deduplicated** — stays
> in the pre-registered 0.7–0.9 band, verdict INDETERMINATE unchanged; its "26 unique prediction
> sets" were 26 hashes but 25 models (ε-duplicates evade hashing). **No verdict, gate, extreme, or
> sign changes anywhere.**
>
> **ROOT-CAUSE CLASS: a correction that lives only in prose decays — censuses re-derive their arm
> list from `glob("windows_*.pt")`, so a duplication documented in a report (DOC_CORRECTION_SWEEP
> 2026-07-26, verbatim: "the same checkpoint under two keys … likewise duplicates") was re-counted
> by every later census and re-discovered twice (2026-08-03 by hash — which structurally cannot see
> ε-duplicates — and 2026-08-18 by per-window values).** Same decay mechanism as the "2 of 36
> features" count (pinned by test 2026-08-16 after four rots). Fix: `taniteval/results/
> dump_exclusions.json` (machine-readable) + `DUPLICATES.md`; census code subtracts the list or
> declares it counts dumps.

## 6. Escalations (integration I cannot do myself)

1. **`MODEL_REGISTRY.md`** (sibling-owned today): §0.3 statistic row, :2722 ("across 27 arms"),
   :2730 ("11 of 27 … 16 deflated"), :2757-2760 ("27 arms recomputed", "10 of 27 positions") —
   apply the §2 phrasings; the rank-move census already carries DOC_CORRECTION_SWEEP's "9 of 27
   substantive" caveat.
2. **`CLAUDE.md` :36-40**, **`README.md` :61**, **`Paper/TANITAD_PAPER.md` :919/924/975**,
   **`PROGRAM_OVERVIEW.md` :282/306** — same three-count/phrasing fix (orchestrator owns these).
3. **`Benchmarks & Eval/LEADERBOARD.md` §1b** — "12/6" → "11/5 distinct arms" (one-line fix,
   whoever owns the leaderboard).
4. Future censuses (`ff_rescore.py` refuses duplicate NAMES only — `tools/ff_rescore.py:574`;
   nothing checks duplicate VALUES): wire `dump_exclusions.json` into the next census tool that
   walks `windows_*.pt`. Backlog-sized, not urgent — the panel's dedup note covers today.

## 7. Deliverable manifest

| artifact | status | location (repo clone `/c/Users/Admin/wt-tanitad-local`) |
|---|---|---|
| `taniteval/results/DUPLICATES.md` | NEW, staged | repo |
| `taniteval/results/dump_exclusions.json` | NEW, staged | repo |
| `taniteval/taniteval/dump_lead_join.py` | docstring note, staged | repo |
| `taniteval/taniteval/rollout.py` | docstring note, staged | repo |
| `…/incoming/2026-08-18-dump-lead-wiring/DUMP_LEAD_WIRING.md` | dated correction, staged | repo |
| this report | NEW, staged | repo |

Nothing lives on a pod, worktree, or scratchpad; no dump was modified or deleted; `MODEL_REGISTRY.md`
and the registry lint tests untouched; sibling's staged index entry undisturbed (verified post-stage).
