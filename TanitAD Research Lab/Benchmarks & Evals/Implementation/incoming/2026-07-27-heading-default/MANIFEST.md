# Deliverable manifest — `heading-default` (2026-07-27)

**HEAD at start:** `8ab5327` · **branch:** `agent/benchmarks-eval-20260721`
**STAGED, NEVER PUSHED.** I ran no `git commit`, no `git push`, no branch switch.
⚠️ **The orchestrator committed mid-session** (`2cc2526`, alongside a sibling's fleet-sync work),
sweeping in the Job-1 files that were staged at that moment. The remaining edits are staged in the
working tree. Nothing was lost and **nothing lives in only one place.**

⛔ **No pod was touched.** pod1 (training ~21,650/30,000), pod2 (small validation, cgroup
53.9/55.0 GB, armed eval chain) and pod3 / `tanitad-eval` (a sibling's IDM-steer 3-seed) were not
probed, not signalled, not logged into. Everything ran on the dev box (RTX 4060, ~6 min of GPU).
⛔ **Nothing pushed to HF.** ⛔ **`Mission Plan.md` and `PRE_REGISTRATION_IDMV2.md` were not opened.**
🔒 Counts only, no clip UUIDs. 🔒 Parity untouched — comma2k19 is a NON-PARITY corpus; no PhysicalAI
path, cache key or episode selection was modified, and `test_wheelbase_regime.py` is green.

---

## A. New artifacts

| # | artifact | where it lives | note |
|---|---|---|---|
| 1 | `HEADING_DEFAULT.md` | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-heading-default/` | **the deliverable** — the choice + reasoning, the guard with its demonstrated failure, the reproducibility pin, the re-score per corpus, stale-pending resolution, escalations |
| 2 | `MANIFEST.md` | same dir | this file |
| 3 | `code/heading_default_demo.py` | same dir | produces (4) by RUNNING the shipped guard — every refusal exercised on the input that must trigger it |
| 4 | `raw/heading_default_guard.json` | same dir | raw JSON for every §2/§3 number: the defect reproduced (61.78 → 0.0 rad/s), 7 refusals, the cache keys, the bit-identical pin |
| 5 | `code/rescore_idm_head_v1_comma.py` | same dir | the re-score. md5-pins both artifacts at run time, resumable, ~10 min cold / ~30 s warm on the dev box |
| 6 | `raw/idm_head_v1_comma_rescore.json` | same dir | raw JSON for every §5 number: both protocols × 4 channels with CIs, paired contrasts, controls, the refuted hypothesis, the measured mechanism, the admissibility diagnostic, the leak-probe episode ids |

## B. Code + tests (repo)

| # | file | change |
|---|---|---|
| 7 | `stack/tanitad/data/comma2k19.py` | ⭐ `DEFAULT_HEADING_MODE` → `HEADING_MODE_HOLD`; `LegacyHeadingRefused`, `LEGACY_HEADING_REASON`, `resolve_heading_mode()`, `label_params()`, `cache_build_params()`; heading kwargs on `actions_and_poses` / `build_episode` / `Comma2k19Dataset`; the wholly-stationary consequence documented on the repair itself |
| 8 | `stack/tanitad/train/train_worldmodel.py` | comma cache params via `cache_build_params`; resolved mode passed to `build_episode`; protocol printed into the run log; `--comma-heading-mode` / `--comma-legacy-heading-reason`; forwarded through `_build_datasets`, `train()`, and the `mix`/`realmix` recursions |
| 9 | `stack/tanitad/lake/ingest.py` | 🔴 **second-probe find:** `Comma2k19Ingestor.build_params` carried **no label regime**, yet `build_params_hash` is written into every lake record **and exported to HF** — a post-flip lake would have shared a hash with a pre-flip one. Now routed through `cache_build_params` |
| 10 | `stack/tests/test_comma_heading_regime.py` | **new, 23 tests** — flip, guard firing, cache-key separation, reproducibility pin, trainer end-to-end |
| 11 | `stack/tests/test_comma2k19.py` | the one assertion the flip invalidated, renamed and inverted; the stationary-no-op test now asserts its **consequence** |

## C. Stale-pending updates (superseded values preserved, with their dates)

| # | file | change |
|---|---|---|
| 12 | `repo:…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json` | ⭐ **the card a consumer reads.** Both `val_heldout_traindomain` yaw verdicts updated + a new `heldout_traindomain_comma_component_2026_07_27` block. **Purely additive: 18 insertions / 3 deletions, all three "deletions" being the lines appended to.** Re-parsed clean and every published value re-read after the edit — `/val` is byte-identical |
| 13 | `repo:…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` | §4 row 1 marked ACTIONED + new §4.1 with the result; §8 escalations **#1 CLOSED** and **#2 ACTIONED**. Original row text preserved |
| 14 | `repo:Project Steering/RETRACTION_LOG.md` | C5 row gains a forward pointer with the measurement and a **binding lesson** (a repair and an admissibility decision are different things). **Row not rewritten; `0.010` keeps its date** |

⛔ **Deliberately NOT touched:** the other 6 stale-pending substrate groups of the 46 (idm-proof,
own-dynamics-encoder, v1-encoder-char, idm-pipeline-derisk, idm-v2 arms, the `IDM_DIAGNOSIS` ceiling
family) — **named, not inferred**. No PhysicalAI number re-issued. **No repaired ceiling derived.**

## D. Not staged, on purpose

| artifact | where | why |
|---|---|---|
| `lat_cm40_69/` — 30 re-encoded latent files, 37 MB | `scratchpad:…/8fc25020-…/scratchpad/lat_cm40_69/` | binary latents, regenerable in ~90 s by (5) from artifacts pinned by md5. Staging them adds weight, not reproducibility. **Nothing unique lives here** — every number derived from them is in (6) |

## E. Suites

| suite | result | vs the brief's baseline |
|---|---|---|
| `stack/` `pytest -q` | **1557 passed, 12 skipped**, 2 warnings | baseline 1534/12 → **+23 = exactly the new test file**, **zero new skips** |
| `taniteval/` `pytest -q` | **663 passed, 0 skipped** | ⚠️ brief says 661. `taniteval/` is byte-identical to HEAD and this pass changed nothing in it — **HEAD `8ab5327` itself added `taniteval/tests/test_stack_guard.py` (+2 tests)**. Flagged, not papered over |

## F. 🔴 Integration this needs — escalated here AND in the report headline

1. **`hold_heading_through_standstill`'s `observable` mask is returned and never consumed.** It is
   the documented admissibility signal, and its absence let 84 undefined-label windows into a
   published-grade score. Needs a decision: extend the episode contract (`poses` is `[T,4]`, with
   nowhere to put it) or require every yaw-channel scorer to gate on the mask.
2. **Is the v3 anchor partly IN-TRAIN?** Two different comma caches ⇒ tag indices not comparable ⇒
   unknown. One command on the eval pod (`/root/idm2/manifest.json`); my half of the probe — the
   `episode_id` of all 70 relevant clips — is already in (6).
3. **"comma can test yaw" needs qualification.** Testable ≠ working: with a *defined* label the
   deployed head reads comma yaw **R² −0.288 / ρ 0.211 / nMedAE 2.36** on its own held-out clips.
4. **A comma corpus rebuild has NOT been done.** The flip changes what a *new* build produces; no
   existing cache was rebuilt, deliberately.
