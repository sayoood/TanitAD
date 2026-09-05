# RESULT — v7f instrument repair: P1 non-collapse, P2 drift (+ the L3 arm both point at)

**ArchInf FlyWheel · 2026-09-05/06 · 0 GPU on both hosts** (`guard_and_stack` REFUSES to run
with a visible CUDA device; `CUDA_VISIBLE_DEVICES=-1` before torch import). Thor stayed at
97–98 % on three sibling `refav1_lon` arms and the dev-box 4060 at 100 % / 7341 MiB
throughout; the only Thor traffic was **five `scp` checkpoint pulls, md5-verified on both
sides**, plus two `ls`. ⛔ **v7f was not trained.**
SPEC: `SPEC.md`. Predecessor: `…/2026-09-05-v7f-six-problems/STATUS_BOARD.md`.

---

## 0. THE ONE THING TO READ FIRST

⭐⭐⭐ **BOTH `UNVERIFIED-CLAIM` ROWS ARE NOW RULED, AND THEY GO OPPOSITE WAYS.**

| row | was | is now | why |
|---|---|---|---|
| **2 — non-collapse** | ⛔⛔ UNVERIFIED-CLAIM | ⚠️ **PARTIAL — the arms are fine, the CRITERION is retired** | the three T1-read arms, never measured, read **24.66 / 24.98 / 22.97** against `rdw8p30k`'s **25.58** on the identical 12 clips. **There is no admissible absolute bar** — and the gate that actually rejected `splitp30k` never used one. |
| **5 — drift** | ⛔⛔ UNVERIFIED-CLAIM | ⛔ **SOLVED AS A MEASUREMENT · THE EFFECT IS NULL** | the freeze arm's own arithmetic control **falls with its signal**: 0.3902 true vs **0.4487 control**, share **1.1488**, control higher on **8 of 8 directions**. Pre-registered branch, fired as written. |

⭐⭐ **AND THE CHEAPEST DECISIVE ITEM ON THE WHOLE BOARD IS DONE TOO.** P2's null points at
the question drift was proxying for — *does the predictor add anything over looking at now?* —
which is **L3**. Its one missing arm was `splitp30k`. It is now re-scored: **max |t| = 1.08
across all 9 cells against a null bar of |t| ≥ 2.9.** With three arms already re-scored and
none separating, **L3 is rulable for the first time, and NO ARM PASSES IT.**

⛔⛔ **A LIVE DEFECT NEITHER ROW ASKED ABOUT: the O6 collapse gate wired into
`train_v6_staged.py` RULES ON `effective_rank` AND NEVER CONSULTS `participation_ratio`** —
the exact statistic inversion C132 exists to prevent, on the code path a v7f run would take.
§4.

---

## 1. P1 — THE FOUR "FLOOR" VALUES ARE NOT FOUR READINGS OF ONE QUANTITY

Each was taken on a different **(corpus sample × pooling × ambient dimension)**. Re-derived
from source; every row names the file and line that produced it.

| value | what was pooled | corpus SAMPLE | n | d | top-8 | source |
|---|---|---|---|---|---|---|
| **5.756** | frozen DINOv3 ViT-L/16, `last_hidden_state[:, -640:].mean(1)` — one **frame-mean** vector | `physicalai-val-w120-256x640cyl` `sorted()[:12]` × 120 frames | 1440 | 1024 | 0.8387 | `code/floor_valclips.py:46,51` → `code/v7tiny_probe.py:78-128` |
| **8.56** | *nominally the same function* | ⛔ **TWELVE CLIPS THAT EXIST IN NO CACHE ON THIS BOX** — ids `001d413e-2 … 729c7c83-8` | 1440 | 1024 | 0.783 | `raw/v6F_v7tiny_rank_probe.txt:41`; **emitter script is not in the repo** (2 probes) |
| **20.228** (20.516 at full n) | same frame-mean, banked DINOv3 fields | `slotprobe-lead130`, **130 clips** / 5,617 frames, subsampled to n 1440 | 1440 | 1024 | — | `code/floor_reconcile.py` |
| **40.77** | ⛔ **`dino_pooled` CELLS — a flattened SPATIAL GRID** (`r["cells"].reshape(-1)`), not a frame mean | E-TRUNK-3 ladder cache | — | — | **0.348** | `E_TRUNK_3_LADDER.md:20`, `code/e_trunk3_ladder.py:63` |

The `top-8` column is the tell that these are different objects: **0.348 vs 0.78–0.84** is not
sampling noise, it is a different row unit.

### 1.1 ⭐ The 8.56-vs-5.756 gap is the SAME mechanism as the rest — not a fourth mystery

`floor_valclips.py`'s docstring says it runs on *"the SAME 12 val clips the 8.56 is sourced
to"*. **It does not, and nothing in the code ever checked.** It calls
`sorted(VAL.glob("*.v2ep.pt"))[:12]`; `rank_probe.txt:29-40` prints the twelve clips the 8.56
was actually taken on. **The two sets are disjoint — not one clip in common:**

```
sorted()[:12] here : 0002ad3a-5 0084596e-4 00ba3704-6 00ee8da5-f 010dbd68-0 014ee1b8-b
                     0192cb35-7 01acc9de-8 0207d830-b 0250cddb-7 026ef99a-2 028eff14-2
the 8.56 sample    : 001d413e-2 09376a99-6 12bf2d7f-7 1d29c402-e 26ceecfd-8 324f887a-6
                     3e7c570b-5 486043c8-5 5470ba35-1 5ee7eb9d-4 68b1a194-e 729c7c83-8
```

The shape settles it: a `sorted()[:12]` from any pool returns a **contiguous** prefix (all
`00…` here); the 8.56 ids are **spread across the whole id space**, so they were never a
`[:12]` of anything we hold. ⛔ **Channel-proven absence** (`M50`): searching all of
`tanitad-caches` for the twelve 8.56 prefixes returns **0, 0, 0** while the same-breath
positive controls return **1, 1, 3**.

⇒ **`H-RANK-24`'s "a fourth distinct value" should be read as `H-RANK-23` applied a third
time.** Same encoder, same instrument, same n, same d — **different clips**. The 3.51×
val-vs-lead spread and the 1.49× 5.756-vs-8.56 spread have one cause: **participation of a
frozen encoder measures the scene diversity of the SAMPLE at least as much as it measures the
representation.** Nothing about the floor family is unexplained any more.

### 1.2 ⛔ WHICH SINGLE VALUE IS THE ADMISSIBLE BAR? **NONE — and the programme already pins it**

I set out to argue **5.756** is the bar, because it is the one value measured on the arms' own
evaluation clips. **That is wrong, and `stack/tanitad/models/v6.py:1521-1567` already says
why** — a block written 2026-08-23 that reaches the same corpus conclusion independently and
adds the half I had missed:

> *"Note d differs across the things we routinely compare: `z_op` is **d=2048**, this DINOv3
> column is **d=1024**, so they are NOT directly comparable in either direction."*

`stack/tests/test_participation_floor_provenance.py` pins it — **6 passed**, re-run tonight —
including `test_the_measured_references_are_NOT_dimensionally_comparable_to_z_op`, whose
docstring is literally *"The trap on the other side: 'champ30k 6.489 > DINOv3 5.756' is also
invalid."* That is exactly the claim I was one step from making, and I record it against
myself.

⇒ **The admissible L1 statement is RELATIVE — between arms at matched corpus, matched n and
matched d.** There is no scalar floor; `O6_PARTICIPATION_FLOOR = 8.56` should be **retired
from the decision path**, not re-derived. The reconcile artifacts' *"do not fail any arm on
it"* is **CORRECT**, now for three independent reasons: corpus-specificity (`H-RANK-23`),
dimension mismatch (`v6.py:1541`), and the clip-disjointness proved in §1.1.

### 1.3 ⭐⭐ AND THE BAR UNDER DISPUTE WAS NEVER THE CRITERION ANY ARM WAS GATED ON

The board reads the criterion as `V7_RECIPE_AND_SCALEUP.md:266` — *"participation ≥ 8.56"* —
and concludes *"1 PASS, 8 FAIL"*. **The panel that produced those verdicts applies a different
rule.** `code/full_panel.py:26-28`, committed in advance:

> *"KILL-GATE (committed in advance): an arm is REJECTED if participation falls **below the
> rdw8 baseline** on BOTH held-out sets…"*

`:209-210` implements exactly that — `participation_val < base_val AND participation_heldout24
< base_held24`. **The string `8.56` appears nowhere in the gate.**

⇒ **`splitp30k`'s rejection is FLOOR-INDEPENDENT and it STANDS.** 6.383/7.629 against
`rdw8p30k`'s 25.583/26.965 falls on both sets whatever the floor is; `MODEL_REGISTRY.md:4332`
is accurate. ⛔ **The board's Row 2(b) — *"the bar cannot be reproduced ⇒ every verdict in (a)
is UNDETERMINED"* — does not hold for the kill-gate verdicts.** It holds only for the
absolute-floor verdicts (`champ30k`'s `H-RANK-13` FAIL), which §1.2 retires anyway.

### 1.4 ⛔ BUT THE RELATIVE GATE HAS ITS OWN DEFECT: **THE BASELINE IS WHOEVER IS LISTED FIRST**

`full_panel.py:72` `present = [a for a in ARMS if ckpt exists]`, `:205` `base = present[0]`,
`:207` `for arm in present[1:]`. Two consequences, both arithmetic from banked data, no new
compute:

1. **The verdict depends on list order.** `gateb_panel.json`'s order is
   `rdw8p30k, rdw8s30k, splitp30k` ⇒ base 25.583/26.965 ⇒ **2 arms REJECTED**. Reverse it so
   `splitp30k` (6.383/7.629) is first and **0 arms are rejected** — same three arms, same
   numbers, opposite verdict.
2. ⛔ **The first arm never receives a verdict at all — it is structurally unfalsifiable.**
   MEASURED: `o14_content_panel.json`'s arm order is `['o14fut30k', 'postrain30k']`, and its
   `kill_gate` block contains **only** `postrain30k`. **`o14fut30k` — the arm the panel exists
   to evaluate — was its own baseline.** ⇒ `E-DEC-67`'s *"rank gate HOLDS, O14 costs no rank"*
   is not wrong about the numbers (22.97/19.55 vs 22.37/20.58) but **no gate verdict was ever
   computed for that arm**, and the sentence reads as though one was.

⇒ **Fix (0 GPU, small): name the reference arm explicitly and record it in the artifact**, and
emit a verdict for the reference too (against itself it is trivially "not rejected", but its
absence is what makes the omission invisible).

### 1.5 ⭐ THE MEASUREMENT THE BOARD ASKED FOR: the never-measured arms, corpus-matched

`tools/l1_participation.py` → `raw/l1_participation.json`, `raw/l1_splitp30k.json`.
Corpus `physicalai-val-w120-256x640cyl` `sorted()[:12]` × 120 frames = **n 1440**, d 2048,
val-side, `spectrum_report → participation_ratio`. **Clip ids are banked in the artifact** —
because the whole confusion in §1.1 exists precisely because no artifact named its sample.

⭐ **CONTROL, and the panel is inadmissible without it:** `rdw8p30k` must reproduce
`gateb_panel.json`'s 25.583. **It reads 25.582 — rel_err 0.0000.** A second control landed
free: `splitp30k` reads **6.384** against its banked **6.383**.
⭐ **And a third, on Thor:** `val_rank_probe.py` reads a *different directory*
(`~/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl`, 35 clips) from `full_panel.py`
(`sp2/cache/…`, 24 clips) — so `champ30k`'s 6.499 might not have been corpus-matched at all.
**`ls` on Thor shows their `sorted()[:12]` are the SAME TWELVE CLIPS** (control: home listing
non-empty). ⇒ the two banked panels *are* mutually comparable, and the table below is one
corpus throughout.

| arm | participation (val) | top-8 | `effective_rank` ⛔ not the criterion | ratio to `rdw8p30k` | status before tonight |
|---|---|---|---|---|---|
| `rdw8p30k` **(control)** | **25.582** | 0.4311 | 43.98 | 1.000 | banked 25.583 ✅ reproduced |
| `emao14_30k_tauramp` | **24.975** | 0.4355 | 44.92 | 0.976 | ⭐ **never measured** |
| `emao14_30k` | **24.659** | 0.4369 | 45.40 | 0.964 | ⭐ **never measured** |
| `o14fut30k` | **22.969** | 0.4710 | 46.27 | 0.898 | banked 22.97 (as its own baseline, §1.4) |
| `postrain30k` | **22.372** | 0.4770 | 39.89 | 0.874 | banked 22.373 |
| `splitp30k` **(control)** | **6.384** | 0.8545 | 59.18 | 0.250 | banked 6.383 ✅ reproduced |
| `postrain30k_freeze` | **6.357** | 0.8568 | 58.74 | 0.248 | ⭐ **never measured** |

⭐⭐ **THE HEADLINE: all three T1-read arms sit at 0.90–0.98 of `rdw8p30k`.** The board's worry
— that the arms we actually evaluate might be collapsed and nobody had checked — is
**refuted**. Whatever is wrong with the T1 read (Row 3's ≈0-motion readout), **it is not L1
collapse**, and Row 3's diagnosis should not be looked for here.

⛔ **Two things this does NOT license.**
(a) **Rank is NECESSARY, NOT SUFFICIENT (`C131`).** These are the same arms beaten by a
zero-parameter `hold-v0` on speed by 10.2 m/s. A participation of 24.7 is positive evidence of
**nothing** about driving, and §5 below shows the same arms failing L3 outright.
(b) The two frozen/distilled arms (`splitp30k` 6.384, `postrain30k_freeze` 6.357) sit at ¼ of
the trainable arms — **freezing the encoder costs 3.5× of the participation**, the same
direction as Row 4(c2)'s measured trade (the freeze makes the action channel ~7× deader), now
visible on L1 too.

---

## 2. ⛔⛔ THE C132 INVERSION, MEASURED LIVE IN THIS PANEL

The two statistics **rank the arms in opposite orders**, on one table, tonight:

| | lowest | highest |
|---|---|---|
| `participation_ratio` (σ², **the criterion**) | `postrain30k_freeze` **6.357** | `rdw8p30k` **25.582** |
| `effective_rank` (σ, ⛔ **forbidden**) | `postrain30k` **39.89** | `postrain30k_freeze` **58.74** |

`postrain30k_freeze` holds **85.7 % of its energy in 8 directions** and is simultaneously the
**best-ranked arm by `effective_rank`**. C132 reproduced on real arms, not a synthetic
example — and the reason §3 matters.

---

## 3. ⛔⛔ LIVE DEFECT: THE O6 GATE RULES ON THE FORBIDDEN STATISTIC

`tanitad/models/v6.py::o6_rank_verdict` emits `participation_ratio` and `participation_pass`
as **reported fields**, then decides on `effective_rank` in **every** branch that sets `pass`:

* `:1829` `if er < floor:` → **FAIL** — `er` is `effective_rank`, `floor` is `O6_RANK_FLOOR`;
* `:1846-1875` the retention clause — ratio `er / er0` with `effective_rank_ci95` bounds.

`participation_ratio` appears in **no** `pass` branch. Its own docstring (`:1495-1519`)
requires the opposite — *"COLLAPSE is an energy question — decide on participation"* — and
`:1517` records that **`O6_RANK_FLOOR = 64` was calibrated on a synthetic α=2 power-law and
never against a real sample.**

⛔ **This is live, not archaeology:** `train_v6_staged.py:5148` and `:7347` call it as the
stage gate, so a v7f run would be gated this way.
⚠️ **Scope, stated honestly:** my `effective_rank` column is a val-side n=1440 reading; the
trainer's is an O4-weighted train-pooled stream at a different ceiling (`--spectrum-accum
43 × 48 → ceiling 2063`). I therefore do **NOT** claim "every arm fails clause 3". The claim
is narrower and sufficient: **the ruling statistic is the one the programme's own analysis
forbids, and the floor it rules against was never calibrated on real data.**

**Fix (0 GPU, small):** make `participation_ratio` a ruling clause against a **relative,
matched-reference** criterion; keep `effective_rank` reported; delete
`O6_PARTICIPATION_FLOOR` from the decision path. ⛔ Changing the semantics of a live gate
needs a pre-registration — **this is a PI decision and is NOT taken here.**

---

## 4. P2 — DRIFT: THE CONTROL SUBTRACTED, AND THE ROW RULED

`tools/p2_leak_rescore.py drift` (**unmodified estimator**; the only edit is five `CKPTS` path
registrations) → `raw/drift_freeze.json`. Corpus `physicalai-val130-heldout`, **80 clips /
7,680 rows, K 4, band [0,8)** — the identical selection as the banked `drift.json`.

⭐ **CONTROL FIRST: the rig reproduces the number under dispute.** `postrain30k_freeze`'s
`drift_r_true_minus_timeshuf` reads **0.3902** against the banked **0.3905**
(`freezedrift.json`). The instrument is faithful; what follows is not a different measurement.

| arm | true (r − timeshuf) | **endpoint-shuffled control** | **corrected = true − control** | t | share arithmetic |
|---|---|---|---|---|---|
| `rdw8p30k` | 0.6737 | 0.6727 | **+0.0010** | −0.43 | 1.0019 |
| `postrain30k` | 0.6697 | 0.6774 | **−0.0077** | −3.80 | 1.0154 |
| **`postrain30k_freeze`** ⭐ | **0.3902** | **0.4487** | **−0.0585** | **−4.72** | **1.1488** |

Per-direction, freeze arm — **the control is higher on 8 of 8**, mean −0.0580:

| dir | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| true | .4106 | .4662 | .4462 | .3593 | .3666 | .3479 | .4093 | .3191 |
| control | .4157 | .5588 | .4772 | .3898 | .3824 | .4551 | .5063 | .4042 |
| **Δ** | −.0051 | −.0926 | −.0310 | −.0305 | −.0158 | −.1072 | −.0970 | −.0851 |

### 4.1 The pre-registered branch, and which one fired

`SPEC.md` §2.7 committed both outcomes before the number was read:

* *"if the freeze arm's control **stays near 0.677** while its signal falls to ~0.39, the drift
  improvement is REAL and the row becomes SOLVED"* — **did not happen**;
* *"if the freeze arm's control **falls with the signal**, the improvement is arithmetic and
  the row is OPEN with a null effect"* — ⭐ **THIS FIRED.** The control fell 0.6774 → 0.4487.

⭐⭐ **AND THE ORDERING INVERTS.** On the raw metric the freeze arm looked **1.71× better**
than `postrain30k` (0.3905 vs 0.6674) — the programme's "3.3× effect". After subtracting its
own arithmetic control it is the **WORST of the three**, sitting **7.6× further below its
control** than `postrain30k` does. ⇒ **the freeze did not lower drift; it lowered the latent
geometry that the metric was reading, and the control tracks that change exactly.**

⇒ **`H-LEAK-3`'s verdict is confirmed ON THE ARM IT MATTERS FOR**, rather than inferred from
the other two: readiness decision **G.1 (accept P3 as met by `postrain30k_freeze`, 0.3905)
MUST NOT be adopted.** ⛔ **And the 8.6 h queue item #3 must not be run** — not merely because
the arm exists, but because **the metric it would be read on is null.**

### 4.2 ⚠️ Estimator, named — and which of the three variance questions it answers

`t_true_vs_endpoint_shuffled` is a **paired t over `TR − CT`**, 8 directions × 80 clips = 640
paired per-clip k-fold scores. ⚠️ **Those 640 are NOT independent — the 8 directions share the
same 80 clips**, so the t is **anti-conservative** (pseudo-replication, the `H-LEAK-2` family).
I therefore do not quote −4.72 as a decision-grade interval. **The verdict does not rest on
it:** the corrected effect is **negative on 8 of 8 directions** with mean −0.0580, and a
clip-clustered estimator can only widen an interval that is already wholly on the wrong side
of zero.

⭐ **Which variance?** This resamples **clips ⇒ "would another draw of EPISODES say this?"** —
not another training run (`H-ESTIM-SEED-1`), not another inference run. ⚠️ **But here that
limitation is unusually weak, and it is worth saying why:** the comparison is **within-arm** —
one checkpoint scored against a permutation of its own endpoints on the same rows — so
training and inference variance are **held fixed by construction**, not merely unmeasured. A
seed replicate would move `r_true` and `r_endpoint_shuffled` together.

---

## 5. ⭐⭐ RULE ZERO — THE NEXT LEVER, RUN IN THE SAME TURN: L3's MISSING ARM

A null is a waypoint. The question drift was *proxying for* — **does the predictor add
anything over simply looking at `z_t`?** — is **L3** (Row 6), and the board sizes its one
missing arm at ~11 min CPU + a 133 MB pull: **`splitp30k`, the deliberate-regression arm the
whole gate depends on**, with the standing note that *"until it is re-scored a PASS by any
other arm would mean nothing"*.

`tools/p2_leak_rescore.py l3 --arms splitp30k` → `raw/l3_splitp30k.json`. True
leave-one-clip-out, within-clip Pearson r per clip, paired clip-level t + clip bootstrap.
Corpus `physicalai-val130-heldout`: **124 labelled → 70 lead-matched → 24 used** (the banked
selection reproduced exactly).

| K | target | n | `z_t` r | `ẑ_GT` Δ vs `z_t` | t | CI95 | `ẑ_HOLD` Δ | **GT − HOLD** |
|---|---|---|---|---|---|---|---|---|
| 1 | `n_agents` | 24 | +0.0311 | −0.0155 | −0.66 | [−0.0590, +0.0308] | −0.0155 | +0.0000 |
| 1 | `lead_range_m` | 23 | +0.1391 | +0.0057 | +1.08 | [−0.0045, +0.0152] | +0.0057 | +0.0000 |
| 1 | `lead_closing` | 23 | +0.2173 | −0.0041 | −0.41 | [−0.0233, +0.0152] | −0.0041 | +0.0000 |
| 3 | `n_agents` | 24 | +0.0409 | −0.0161 | −0.51 | [−0.0803, +0.0413] | −0.0161 | +0.0000 |
| 3 | `lead_range_m` | 23 | +0.1375 | −0.0043 | −0.21 | [−0.0439, +0.0317] | −0.0043 | +0.0000 |
| 3 | `lead_closing` | 23 | +0.1545 | +0.0081 | +0.31 | [−0.0396, +0.0559] | +0.0081 | +0.0000 |
| 6 | `n_agents` | 24 | +0.0051 | +0.0412 | +0.74 | [−0.0600, +0.1556] | +0.0429 | −0.0017 |
| 6 | `lead_range_m` | 23 | +0.1322 | −0.0460 | −0.96 | [−0.1414, +0.0385] | −0.0464 | +0.0004 |
| 6 | `lead_closing` | 23 | +0.1170 | +0.0172 | +0.50 | [−0.0501, +0.0823] | +0.0177 | −0.0005 |

⭐⭐ **MAX |t| = 1.08 OVER ALL NINE CELLS, against the measured null bar |t| ≥ 2.9. Every CI
straddles zero.** `splitp30k` does **not** separate from `z_t` at any K, on any target.

⇒ **L3 IS RULABLE FOR THE FIRST TIME, AND THE RULING IS THAT NO ARM PASSES IT.** Three arms
(`rdw8p30k`, `postrain30k`, `postrain30k_freeze`) were already re-scored and none separated;
the fourth and last is now in and does not either. **The predictor adds nothing over looking at
now — on every arm the programme has, at every horizon, on every environment target.**

⚠️ ⭐ **AND A SECOND FACT FELL OUT THAT NOBODY ASKED FOR: `GT − HOLD` IS ±0.0017 AT WORST AND
EXACTLY 0.0000 IN SIX OF NINE CELLS.** Feeding the **true future actions** and **holding the
last observed action** produce the same rollout to four decimal places. ⇒ **Row 4's
action-deafness and Row 6's L3 null are the same fact seen twice**: a predictor that ignores
its action input has a rollout that is a fixed function of `z_t`, and therefore *cannot* add
anything over `z_t`. This is an independent corroboration of P2/`MM-E10` at the decodability
level, on the split arm, on a different corpus and instrument.

---

## 6. WHAT IS STILL OPEN, AND WHAT ACTUALLY NEEDS A GPU

* **Row 2's remaining gap is not a measurement, it is a DECISION.** Retiring
  `O6_PARTICIPATION_FLOOR` from `o6_rank_verdict` and making participation a ruling clause
  changes the semantics of a live gate ⇒ **pre-registration + PI**. Zero GPU when authorised.
* ⛔ **`postrain30k_freeze`'s and `emao14_*`'s `participation_heldout24` are NOT measured
  here** — the `v7tiny-heldout24-w120-256x640cyl` cache is not on this box. The relative
  kill-gate needs BOTH sets, so those arms have a **val-side reading only**. Closing it is one
  more cache pull + ~40 s/arm CPU. It does not change any verdict above, because an arm at or
  above baseline on val cannot be rejected by an `AND`.
* ⭐ **The one thing on the board that genuinely needs a GPU is unchanged and is Row 3:** the
  **≈0-motion readout** speed-scale probe, ~15 min on a checkpoint we already hold. It gates
  every T1 number, and if it is a scale defect every T1 number is recoverable by re-analysis
  with **zero retraining**. Both hosts were saturated all session; it is queued, not blocked.

---

## 7. DELIVERABLE MANIFEST

All paths under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/`.

| artifact | what it is |
|---|---|
| `SPEC.md` | pre-registration, both outcomes committed before the reads |
| `RESULT.md` | this file |
| `raw/l1_participation.json` | 6-arm corpus-matched L1 panel + clip ids + the `rdw8p30k` control |
| `raw/l1_splitp30k.json` | `splitp30k` L1 + the control again |
| `raw/drift_freeze.json` | ⭐ the endpoint-shuffled control on `postrain30k_freeze` |
| `raw/l3_splitp30k.json` | ⭐ the corrected L3 re-score of the deliberate-regression arm |
| `raw/*.log` | stdout of all four runs, including the geometry/param banners |
| `tools/l1_participation.py` | new: the corpus-matched L1 probe (0-GPU, control-gated) |
| `tools/p2_leak_rescore.py` | the audit tool + **five `CKPTS` path registrations only** (md5s in the comment) |
| `tools/{_stackresolve,rangeprobe_rff,panel_kfold,actdiv_pairs}.py` | its helpers, copied unmodified so the package runs standalone |

**Checkpoints pulled from Thor (md5-verified both sides), now under
`C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901\`:**
`splitp30k 4348cad2…` · `postrain30k_freeze 5f5e5c92…` · `emao14_30k 3a030ba2…` ·
`emao14_30k_tauramp a64aa48a…` · `o14fut30k 3e4a7443…`

⚠️ **Operational note for the next agent:** the G: mount went fully down mid-session
(`Invalid request code` on reads; a same-breath control file failed **with** the target,
proving a mount outage rather than an unreadable file) and then flapped. The working set was
mirrored to local disk with `robocopy /R:40 /W:3` and everything was authored locally and
copied in. ⚠️ **The session scratchpad `…/scratchpad/pkg/` is SHARED with a sibling agent
(the turn-asymmetry panel) and my `RESULT.md` was overwritten there once**; this package was
re-authored under `…/scratchpad/v7fir/` and the sibling's files were left untouched.
