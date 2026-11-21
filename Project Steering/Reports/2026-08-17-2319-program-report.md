# TanitAD program report — 2026-08-17 23:19 Europe/Berlin

⚠️ **CLOCK NOTE.** The cron slot that fired this is labelled *"Morning slot (07:57)"*; the wall
clock is **23:19 CEST / 21:19 UTC**. The report is dated and filed by **wall time**, not by the
slot label. All times below are **Europe/Berlin** unless stamped UTC; Thor's logs are UTC.

**Headline:** the world model is training healthily and *improving*, and the single largest
architectural claim of the last two days — *"the WM latent does not encode other agents"* — was
**WITHDRAWN by its own positive control**. What replaces it is narrower and more useful: the
information is present and weak, and the slot decoder cannot use it.

---

## 1. Fresh measurements

### v6F S-W 30k on Thor — MEASURED 2026-08-17 21:19 UTC

| | |
|---|---|
| step | **12,500 / 30,000 (41.7 %)** |
| marginal pace | **26.47 s/step** |
| projected finish | **≈5.36 days** (17,500 steps remaining) |
| log freshness | **1,171 s** — normal; `--log-every 50` at 26.5 s/step is one line per ~22 min |
| trainer | **PID 25477 ALIVE** (matched on `args`, then `kill -0`) |
| snapshot daemon | **PID 42229 ALIVE** — 4 snapshots (9250 / 10000 / 11250 / 12000) |
| disk | 366 GB free |

**gnorm-vs-loss health (n = 32 logged points):**

```
loss   last 1.434 · median 2.188 · min 1.335 · max 3.674
gnorm  median 568.8 · min 82.0 · max 1366.7
loss spikes > 2x median:  NONE
```

⇒ **HEALTHY, and improving.** Loss has fallen from a ~2.8 median earlier in the session to a
1.41–1.43 tail, with zero spikes above the 2× threshold. gnorm sits mid-band throughout — the
discriminator that separates a hard batch from instability, and it never fires.

⚠️ **Standing risk, unchanged: the run has NO AUTO-RESTART.** No supervisor is attached. If it
dies it stays dead until a human or a drumbeat notices. *(The absence is also what made
file-shipping to Thor safe today — the same fact cuts both ways.)*

### SAM3 perception backfill — census by CONTENT, never by file count (C77)

Read from the banked homogeneity manifest, not by re-listing files:

| | |
|---|---|
| clips | **201 / 201** |
| `UNIFIED` / `HOMOGENEOUS` | **true** |
| residual | **[]** (empty) |
| floors | one — **0.25** |
| schema versions | one — **v2** |
| `perception_engine_mixed` | **false** |

⇒ **The corpus is complete and homogeneous.** Every record carries the `liveness` control and zero
error entries; the earlier zero-split resolves to *legitimately empty scenes*, **0 dead-control
failures**. Published gated-public (`sam3_unified_201_v2/`, `fused_aug120_v3/`) with a 3/3 byte
round-trip against md5s committed to git.

### Local / streams

| | |
|---|---|
| repo | HEAD **`8bc4b69`**, **0 unpushed · 0 staged · 0 dirty** |
| commits since the last report (2026-08-16 01:30) | **24** |
| live streams | Thor training · snapshot daemon · 1 agent (latent linear-readout ladder) |
| dev-box GPU | RTX 4060, ~7.4 GB free — **a second usable GPU**, discovered this session |

---

## 2. Agent output and knowledge transfer

| stream | produced | state |
|---|---|---|
| SAM3 dtype fix + v2 extraction | corpus 0 → **2,496 → unified 201/201**; contours + oriented boxes + scene classes; conf floor an explicit **0.25** | ✅ integrated, published |
| aug120 re-fuse | all five banked defects cleared; `g_str`/`a_str` **201/201 identical** to the corrected labels via an independent code path | ✅ integrated |
| Lane-change deep review | PI ruling implemented: `LANE_TARGET`/`PREPARE_LANE_CHANGE` **80 → 0**; all 80 were route-following | ✅ integrated |
| Tactical label validation | Alpamayo sound (n=4,729, 40.62 % dual-axis); VLM longitudinal was a **constant**; the 2-of-3 vote was **one source counted twice** | ✅ integrated |
| F-18 agent-slot decoder | 3,207,445 params, inert at default, new `interp` isolation group + probed edge | ✅ merged, ⛔ **not trained** |
| F-16 band-seam instrument | fires on injected seams, **FPR 0.0 across 58 boundaries**; found 2 defects in its own statistic | ✅ merged, ⚠️ **no real numbers yet** |
| Train-corpus obstacle join | **2,308 eps / 433,040 frames / 12,122,129 boxes**, HF + 3-way md5 | ✅ integrated |
| Slot probe (parity) | negative at 5 checkpoints × 3 seeds; **killed its own "progressive discard" trajectory as fit noise** | ✅ superseded by the control below |
| **Probe positive control** | ⛔ **D1 WITHDRAWN** — the probe cannot read the answer when handed the answer | ✅ integrated, **highest-value result of the period** |
| w120val sign adjudication | the ⅔-garbage claim is an **instrument** defect (G1's cropper) — sign channel **released** | ✅ integrated |
| DIR_YAW re-read | do NOT change 0.15; it **aliases the training-label threshold**; the paper's κ-collapse is **confounded** | ✅ integrated, 🔶 paper action pending |
| S-T launch readiness + fixes | five launch defects found and fixed; **Thor synced** (14 files, real-import verified) | ✅ integrated |
| Latent linear ladder | running | 🔄 in flight |

**Rejected / not done, with reasons:** the w120 Alpamayo extraction was launched locally and
**stopped by PI decision** (it belongs on Thor, where training happens); the O-term gradient
cosines were **correctly not landed** (their reference direction would have been the *broken*
head's own loss); E4 was **pinned rather than fixed** (two legitimate repairs, both the PI's call).

---

## 3. Position vs the four edges — with an evidence grade each

### PLANNING (the hierarchy thesis) — 🟥 **WEAK, and honestly weaker than a week ago**
The three-layer claim still has no positive evidence at the tactical or strategic level.
`a_tac_lon` is corroborated on **0/201** clips; the strategic lane-change label was **≈78 % wrong**
and its gate is removed; `LANE_TARGET` no longer emits at all. The slot readout that was meant to
give the perception layer a measurable output **failed its own positive control**.
⇒ *Grade: the hierarchy is implemented and instrumented; it is not yet demonstrated.*

### EFFICIENCY (self-supervised WM, the core claim) — 🟨 **UNRESOLVED, and now measurable**
Training is healthy and loss is falling. The D1 withdrawal means we have **no valid negative
result** about what the latent encodes — but the linear-readout finding (**~1.8 m better than the
random null, r +0.159**) says the information is *present and weak*. ⛔ Nothing here touches the
self-supervised claim: every probe is a **frozen-trunk readout**, and labels never reach the
encoder.
⇒ *Grade: the claim is intact and untested; the ladder now running is the first instrument that
could grade it.*

> ⛔⛔ **CORRECTION 2026-08-18 (citation sweep) — THE SENTENCE ABOVE IS RETRACTED (C92, C97, C103,
> C107). It is kept visible because this is a dated report, not a live doc.**
> **Cite this block by its heading, never by line number.**
>
> *"~1.8 m better than the random null, r +0.159 ⇒ the information is present and weak"* was quoted
> **without a trivial-proxy control**, and the margin is an **EGO-SPEED PROXY**:
> * ⛔ **It does not shrink — it INVERTS.** At the *eval-optimal* alpha (cheating in the arm's own
>   favour) the true margin is **~0.02–0.07 m — a 25–90× overstatement — and no alpha anywhere
>   reaches a PASS.** On the repaired 3-seed read the latent is **+0.283 m WORSE** than the
>   matched-random null (per seed **+0.694 / −0.017 / +0.173** m — ⚠️ **the sign flips on seed 1**).
>   The arm wins on the **inner split** and loses on **held-out episodes** ⇒ **episode-level
>   overfitting, not agent geometry.**
> * ⛔ **`r +0.159` is a pre-repair value, and partialling ego speed out drives it NEGATIVE**:
>   **−0.0884** (3-seed mean; per seed −0.1065 / −0.0665 / −0.0922 — a **SPREAD, not a CI**).
> * ⭐ **Ego speed ALONE — one feature — beats the entire 2 048-dimension latent on lead gap**:
>   **K1 −1.5618 [−2.0229, −1.1363], separated PASS, guard OK, on all three seeds and both repair
>   routes** (MAE 3.5712, r² 0.4672) against the latent's **r² 0.0069** and a K1 that never passes.
>
> ⇒ ⭐ **The corrected grade is BETTER EVIDENCE, not a worse one.** *"The information is present and
> weak"* becomes *"the information the readout was finding was the ego-speed scalar the model is
> handed"* — a **measured, falsifiable statement** where the original was an artifact. **The ladder
> that was "now running" when this report was written has since reported, twice, and the finding it
> returned survives at three seeds on both routes.**
> **Re-quote from** `…/incoming/2026-08-18-ladder-3seed/LADDER_3SEED.md` **§6a** and
> `…/Benchmarks & Eval/Implementation/incoming/2026-08-18-citation-sweep/CITATION_SWEEP.md` **§1**.
> ⚠️ **PROVENANCE:** `MEASURED` · `v6F-SW-30k@11250` ⚠️ **early read, 37.5 %** · **T0-DIAGNOSTIC —
> never driving performance** · 130-clip lead-enriched probe pool, **70 eval clips** ⚠️ **NOT the
> 40-episode val set** · paired episode-cluster bootstrap, `n_boot 2000` · route A (`unpen`).

### SAFETY / SELF-KNOWLEDGE — 🟩 **STRONG, and the period's real output**
**Nine retraction classes** logged (C77, C79–C88), most found by instruments catching instruments:
a corpus-wide flattened RLE that summed to the right total; a criterion pinned at 0.5 by
construction that **prefers noise**; a "flaky" test that was deterministic on `PYTHONHASHSEED`; an
S-J group training a module its loss never reaches; a suite baseline that was a *collected* count;
a probe validated only by negative controls.
⇒ *Grade: the programme's ability to detect its own errors is its most developed capability.*

### DATA EFFICIENCY — 🟩 **GOOD, materially advanced this period**
Perception corpus unified and published; the train-corpus obstacle join exists (12.1 M boxes);
labels corrected and re-fused; the sign channel released with limits stated.
⚠️ **The gap is human validation**: no person has reviewed a single *tactical* label.
⇒ *Grade: the data layer is the healthiest part of the programme.*

---

## 4. Ordered next steps

1. **Schedule the P1/P3/P6 gate battery** for the moment the 30k lands — without it the S-W gate
   reads **INCONCLUSIVE**. GPU job, ~5.4 days out, Thor's emitters are now current. **Deadline.**
2. **Re-run D1 properly at 30k**: {oracle, latent, null} × `n_queries 16` × ≥3 seeds — **9 fits, no
   trunk compute**. This is what converts a withdrawn claim into a real one.
3. **Launch S-T** using `…/2026-08-17-st-launch-fixes/raw/st_launch_line_fixed.txt`,
   ⛔ **with `--v2-lru 64`** (chain default is 6).
4. **w120 Alpamayo extraction on Thor**, densest-chunks-first (top 50 ⇒ 1,317 clips / 65 GB).
   No GPU needed — it can in principle run *beside* S-T training; pilot ~10 chunks with `step_s`
   measured before/after rather than assuming.
5. **The tactical review sheet** — the only thing that can validate that label family.
6. Re-take the `tokens` → *"the loss is at the encoder"* conclusion with the repaired instrument.

---

## 5. Decisions required from Sayed — each with a default

| # | decision | default if you say nothing |
|---|---|---|
| **D-a** | **E4**: the S-T gate requires `sel_gap`, emitted only when `w_select > 0`, which needs a selector, which SEL-1 refuses ⇒ INCONCLUSIVE by construction, and it propagates to S-S. Two legitimate fixes. | **Leave pinned.** 3 tests hold it still; S-T runs and the gate reports INCONCLUSIVE honestly rather than being resolved in a direction nobody chose. |
| **D-b** | **The `a_str` abstain channel** is built, default-off, and **zero records use it**. | **Keep.** `a_str` genuinely has no abstain token; it is inert and a clean revert. |
| **D-c** | **`TANITAD_PAPER.md:1868`** — κ 0.253 → 0.0072 is **confounded** with a label-definition change (v2 gates curvature, v1 net yaw). Real, but not separable from banked data. | **Annotate, do not delete.** The number describes that arm; it may not be read as "the corpus caused a collapse of this size". |
| **D-d** | **Extraction/training concurrency on Thor** after 30k. | **Pilot, don't assume.** ~10 chunks alongside training with `step_s` before/after, scale only if provably unaffected. |
| **D-e** | **A supervisor for the live run** (no auto-restart for ~5.4 days). | **Do not add one mid-run.** Our own trap list says a supervisor sources its manifest once and will resurrect a finished run; the drumbeat already probes every iteration. |

---

## 6. Incidents — honestly

* ⛔ **I propagated a false test baseline into three agent briefs** ("3572 passed / 0 failed"), then
  **retracted a CORRECT claim** on one agent's uncorroborated correction — inside the retraction
  about doing exactly that. Three banked pytest artifacts answered it the whole time. **C82/C86.**
* ⛔ **My own commit `06b8782` caused the S-J defect**: appending `interp` to `MODULE_GROUPS`
  changed what S-J trains **without touching the line that declares S-J**. Tuple immutability
  protects against mutation, not meaning. **Two tests had hardened the defect into an assertion.**
* ⛔ **Three commits swept live agents' in-flight work** under unrelated titles. Our two git rules
  contradict each other under concurrency — the only commit form that works is the only one that
  sweeps. **C88**; the procedure is now "enumerate every stream in the message".
* ⚠️ **I relayed "13/13 launch-path files drift"** — **7 were a CRLF/LF artifact**. The conclusion
  survived (2 real drifts + 6 absences); the evidence for it was partly spurious.
* ⚠️ **I briefed "Colab T4 only"** for a job the dev box could run, manufacturing a resource
  decision out of my own omission — after having logged that exact blind spot the night before.
* ⚠️ **The w120 extraction was launched and stopped** on PI decision. No bandwidth burned.

---

## 7. Evidence classes

Every number above is **MEASURED** (ours, with the artifact path in the referenced package) except:
the ≈5.36-day projection (**ESTIMATED**, linear extrapolation of the marginal pace) and the
2.5 MB/s link figure underlying the w120 sizing (**MEASURED but n=1**, one timed 41.4 MB file —
not a rate).

**Tier stamps:** every probe result in this report is **T0-DIAGNOSTIC** (a WM diagnostic, never
driving performance). No T1 number is claimed.
