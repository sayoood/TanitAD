# PRE-REGISTRATION ADDENDUM — `E-BEVHEAD-DATA-1` (lever L4): does adding 200 training clips lift the frozen-trunk BEV head?

**Date** 2026-09-13 (Europe/Berlin) · **Author** Architecture & Inference FlyWheel
**Status** written **before any head trained on the extra clips, and before the
`E-BEVHEAD-FROZEN-1` panel was SCORED on test.** The only outcome information available when
this was written is the panel's VALIDATION curves (quoted below) — which is exactly the
diagnosis that motivates the lever, and is why this is labelled a POST-HOC lever everywhere it
is reported.

## 0. The diagnosis, measured on VALIDATION only

`runs/main/log.jsonl` (frozen s32 tokens, 82 training clips): val AP (rule B) **0.4424 → 0.4548 →
0.4361 → 0.4207 → 0.4134 → 0.4044 → 0.4033** at steps 500…3,500, while training loss falls
**0.518 → 0.354**. The best checkpoint is step **1,000**. ⇒ the head **overfits** 82 clips within
the first sixth of its schedule. `runs/shuffled/log.jsonl` reads val AP **0.3636** at step 500
against a val per-cell prior of **0.362** — the zero-information arm behaves as it must.

## 1. The lever — exactly one thing changes

**Training rows only:** the 82 fit-split eval clips **plus 200 B1 TRAIN clips**
(`code/p6b_extra_corpus.py`, selection rule in `raw/p6b_extra_selection.json`: the v7.2 train
release md5 `0ff902130ce76886b8a925eceed9e3a5`, locally available, not an eval clip, not a
protected Qwen-Drive clip, sorted by sha12, first 200). Val (18 clips) and **test (34 clips) are
unchanged, row for row**; the arm is refused if its `rows.npz` test/val arrays differ from the
panel's.

Everything else is the `main` arm: frozen refcv5-v2 s32 tokens, the same head (2,284,993 params),
6,000 steps, batch 32, AdamW 3e-4, best-val checkpoint, rule-B mask.

⚠️ **Two differences in the extra rows, stated rather than hidden:**
1. their frames are REBUILT locally from mp4 (P6a: max 3 levels / mean 0.75 levels from the
   pod-built PNGs; not bit-identical). ⇒ **a token-level equivalence check on two eval clips
   gates the arm** (§3, G5);
2. their images are IN-SAMPLE for the trunk (refcv5-v2 trained on them for planning). The BEV
   target never reached the trunk, so this is not a label leak; it can make the extra tokens
   slightly more "familiar" than test tokens. Test stays 100 % trunk-held-out.

## 2. Arms

| arm | what |
|---|---|
| `data` | the lever, seed 0 |
| `data_s1` | the lever, seed 1 — its OWN training-variance floor |
| `data_shuffled` | the lever with (target, mask) deranged across ALL training rows — its own zero-information control |

(`prior`, `const` are recomputed on the enlarged training rows by the same evaluator; `pixel` is
NOT re-run for L4 — see B3 below.)

## 3. Committed criteria

Gates (instrument): **G1** `const` AP = test prevalence (1e-6). **G2** `data_shuffled` AP ≤
enlarged-`prior` AP + 0.02. **G5** the P6a frames give trunk tokens within **1 %** relative
mean-abs of the pod-frame tokens on two eval clips (`raw/p6c_token_equivalence.json`); if G5
fails, the arm is not read.

**`E-BEVHEAD-DATA-1` SUPPORTED iff:** `data − main` test AP ≥ **+0.03**, paired clip-cluster
bootstrap 95 % CI excluding 0, and ≥ **3 × max(F_main, |data − data_s1|)**.
**The pre-registered bars B1, B2, B4 of `E-BEVHEAD-FROZEN-1` are then re-read on `data`**, with
its own floor. **B3 (beyond pixels) is reported as NOT RE-TESTED** — the pixel floor was trained
on 82 clips and a fair B3 needs `pixel` on the same 282; that arm is named, not run, unless the
GPU budget allows.

**REFUTED iff** `data − main` ≤ +0.01 or its CI includes 0 ⇒ the head is not data-limited at
this scale ⇒ the next lever is capacity/inductive bias (L1–L3), not data.

---

## 4. ⛔ AMENDMENT A1 (2026-09-13, after G5 FAILED, before any L4 arm ran)

**G5 failed as pre-registered** (`raw/p6c_token_equivalence.json`): rebuilt-frame tokens differ from
the pod-frame tokens by **1.84 % / 1.73 %** relative mean-abs on two eval clips (bar ≤ 1 %), while the
wrong-row control reads **65.8 % / 56.6 %** — the comparison discriminates; the rebuild is simply not
close enough. ⇒ **the arm as designed in §1–3 is NOT run** (it would have mixed two frame
distributions across train and test).

**Cause, narrowed and not closed:** `code/p6e_decode_variants.py` (`raw/p6e_decode_variants.json`)
— all six PyAV interpolation variants and the ITU601 source colourspace give the IDENTICAL match
(27.09 % exactly equal, mean 0.772 levels); ITU709 is worse (mean 1.118, max 34). The stream is HEVC
yuv420p (bit-exact decoding by spec), and `stack/tanitad/data/calib.py` has had no code change since
the payloads were built (2026-08-30; `fdc9362` touched a comment only). ⇒ the residual is a
**library-version** difference on the pod (swscale YUV→RGB rounding and/or torch `grid_sample`) that
no exposed flag reproduces. Not pinned further.

**Amended design — the same question on ONE frame distribution:**

| arm | frames for train / val / test | train clips |
|---|---|---|
| `main_rb` | ALL rebuilt locally (the 139 join clips re-rendered from mp4) | 82 |
| `data_rb` | ALL rebuilt | 82 + 181 extra |
| `data_rb_s1` | ALL rebuilt, seed 1 | 82 + 181 |
| `data_rb_shuffled` | ALL rebuilt, targets deranged over all training rows | 82 + 181 |

Same split rows, same test clips (rebuilt), same head and schedule. **Criteria unchanged in form,
re-based on the same-distribution baseline:** `data_rb − main_rb` test AP ≥ **+0.03**, CI excluding 0,
≥ **3 × max(F_main, |data_rb − data_rb_s1|)**; G1/G2 as before on the rebuilt panel.
⭐ **An extra reading this buys for free:** `main_rb − main` (the same 82-clip arm on rebuilt vs pod
frames, paired on the same test rows) measures what a 1.8 % token shift does to test AP. It is
reported as a measurement, not a bar.
⚠️ Scope: every L4 number is on frames within ~1.8 % token distance of the trunk's training frames,
not on the pod-built frames themselves.
