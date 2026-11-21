# P6-HN-S implementation, part 1: checklist items 1, 2, 5, 6, 7 (all NEW files, tests green)

Tip `bae5a1d0e0257a1781e3ec7303b6d17198518c56`. Prereg `PREREG_P6HN_SCREEN.md` sha256 `9eb7cb2f…` (verified equal to the landed text before starting). No existing repo file was edited, so no base blob sha applies.
No real data, no GPU, no pod, no navtest. Items 3, 4, 8 (`scorer_finetune.py`, `gate_checks.py`, `launch_gate.py`, the chain) are NOT done, by the brief (a concurrent B8 change owns them).

## Done, with test tokens (MEASURED, ours; logs beside this file)

| item | file (PK-relative) | test | token |
|---|---|---|---|
| 1 | `refe/hn_bank.py` | `refe/test_hn_bank.py` | `ZZTEST_PASS 25  (1 unmutated suite + 24 RED mutations)` |
| 2 | `refe/ctx_cache.py` | `refe/test_ctx_cache.py` | `ZZTEST_PASS 15  (1 unmutated suite + 14 RED mutations)` |
| 5 | `eval/hn_screen_offsets_score.py` | in `eval/test_hn_screen_read.py` (checks `S_*`) | below |
| 6 | `eval/hn_screen_families.py` | in `eval/test_hn_screen_read.py` (checks `F_*`, real taniteval adapter) | below |
| 7 | `eval/hn_screen_read.py` | `eval/test_hn_screen_read.py` | `ZZTEST_PASS 18  (1 unmutated suite (incl. items 5, 6, CLI) + 17 RED mutations)` |

Run: `PYTHONIOENCODING=utf-8 python refe/test_hn_bank.py`, `... refe/test_ctx_cache.py`, `TANITEVAL_DIR=<dir holding adapters/ and taniteval/> python eval/test_hn_screen_read.py`
(the last defaults to `<repo>/taniteval`). Interpreter `venvs/tanitad`. Each mutation edits source (snippet must occur once) and must turn the named check RED; the unmutated suite must be green.

## What each file gives the next agent

- `hn_bank.py`: `build_bank / mint_bank / load_bank` (deterministic container + manifest, refuses another partition / label_version / fan fingerprint / changed bytes / missing key / other ckpt_step),
  `HNBank.batch(keys, ckpt_steps, subtypes)` -> `hn_xy [B,8,20,3]`, `hn_tg [B,8,6]`, `hn_m [B,8]` (the trainer's batch fields; `subtypes=(1,)` is the S-DAC bank), `HNBank.hn_block(arm)` (the `hn` data-event block),
  `hard_negative_rule` (ONE definition), `label_identity_check` (I1), `yield_report` (SY), `hn_attn_mask`, `score_set`, `hn_scores`, `hn_term` (the masked two-call term, shared by both trainers), `probe_fields` (raw fields for G14).
- `ctx_cache.py`: `trunk_fingerprint` (non-scorer tensors only, so a scorer-only fine-tune keeps the cache valid), `VisualCtxTap`, `build_units` (per-log units, resumable by CONTENT, lock once per unit via `landed_gpu_lock`),
  `finalize`, `CtxCache` (+ `verify`), `check_fan` (through the landed `rl_fan_dump.compare_sets`, both bounds), `cache_vs_live` (G15's raw field).
- `hn_screen_read.py`: `JointBoot`, `lookup`, `decide_zeros` (Z1-Z3, rule E), `cogate`, M1-M3, IC, `classify` (the section-11 table, first match), `read`, `main`.

## Findings to escalate (decide before items 3/4/8 and before landing the prereg amendment)

1. **The prereg's "65th candidate in its real set" is exact at ONE decoder layer only.** The masked form leaves fan rows unperturbed, so at depth > 1 an hn row reads the fan's UNPERTURBED per-layer states, which is not what a real 65-candidate call
   would give (there the fan rows would see the hn row at layer 1). I think this is the right semantics (it is the situation at selection, where no hn row exists), but the wording in `PREREG_P6HN_SCREEN.md` 2.4 and the amendment should say it.
   Tested against an independent per-layer oracle at depth 3 and against a real 65-candidate call at depth 1 (`T_masked_logits_equal_*`).
2. **The draft's duplicate-row identity** ("a bank row that duplicates fan row j adds exactly row j's share of bce_fan_B") **does not hold with set-level self-attention** (the duplicate is scored against the fan plus itself). It holds on a scorer with the
   self-attention neutralised, which is what `T_weight_1_over_BK6_and_duplicate_row_identity_without_self_attention` proves; gate G14's identity field must use that form (or the oracle), not the literal wording. Item 4's `judge_g14` must be written accordingly.
3. **Fan tolerance (I6).** `raw/rl_fan/atol_calibration` reports `CHOSEN_ATOL = null` (the rule's value 0.5 is taken by the chain from `RLS1_FANCMP_ATOL`) and round 2 `CHOSEN_MEAN_ATOL = 0.025`. `ctx_cache.FAN_ATOL = 0.5`, `FAN_MEAN_ATOL = 0.025` are INHERITED
   from those files and are NOT defaults of `check_fan`: the caller names them. The registered value of the max bound is UNVERIFIED; confirm before S2.
4. **Trajectory-level families are not paired.** The landed instrument (`taniteval` adapter -> `four_families_block`) exports aggregates and its own unpaired intervals, not per-window values. `hn_screen_families.py` therefore applies the landed
   `proxy_eval.fam_adverse` rule (non-overlap) to each seed and to the SEED-MEAN BLOCK (numeric leaves averaged). The PDM sub-scores (EP, TTC, DAC) and the tactical selection quality ARE paired on the joint draws, in `hn_screen_read.py`.
   Prereg section 10 says "paired"; this is a stated deviation for the trajectory-level components only (reported, not gating).
5. **Interpretations I made (each is one line to change):** (a) SY's `< 0.10` is judged on the POINT estimate of the share over EVERY bank scene (stationary and label-failed count as "no hard negative"); the interval is reported beside it.
   (b) hard negative = (NC == 0 or DAC == 0) and PDMS(offset) == 0 and PDMS(anchor) > 0; a PDMS = 0 caused by neither (e.g. DDC) is NOT one (`zero_other` is counted in the read). (c) offsets and anchor are rounded to the fan dump's
   precision (5 d.p. xy, 6 d.p. yaw) before BOTH labelling and storage, so the scorer is fed exactly the labelled numbers. (d) slot order of the 8 offsets is ascending delta.
6. **G15 bit-identity may depend on the micro-batch.** `build_units` defaults to batch 8 (the trainer's). Build the cache with the trainer's batch; on GPU, batch-dependent kernels can break bit equality (CPU fixture: independent of batch).
7. **Lock owner label.** The landed lock writes `OWNER = "aux_navtest_forward"`; `landed_gpu_lock` sets it to `ctx_cache` for the call and restores it.
8. **STRATEGIC** is stamped PAUSED (PI 2026-09-19, item 25) in the read and in the families output; never a pass, never omitted.

## Not done / UNVERIFIED

- Items 3, 4, 8 (see `HANDOFF.md`). Nothing here is wired into a trainer: the "built, tested, unreachable from its caller" rule applies to the wiring, which item 3's `test_hn_screen_wiring.py` must prove (hop trace, gradient table, the 7 mutations).
- Real-data loaders: the builder takes anchor dicts and a label function; reading the TargetBank, the RL-S1 manifests and the fan sets is S1 work. `hn_screen_offsets_score.py` takes the scorer and the fan provider as arguments; loading an arm's saved scorer tensors
  is `load_scorer_state` (strict, scorer modules only) and its on-disk key layout is UNVERIFIED.
- Nothing was run on a GPU; the `landed_gpu_lock` path was not exercised (the tests inject a fake lock).
- `ZT4` M2 reuses `zt4_baseline.twin_pairs` loaded from its landed file; the screen's M2 population (all sets with a pair) is the prereg's, ZT4's own population is reported per arm.
