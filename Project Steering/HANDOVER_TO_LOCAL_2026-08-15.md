# Handover: cloud session → local session ("TanitAD project kickoff (fork)")

**Written 2026-08-15 by the cloud orchestrator at the PI's request**, at the close of the
cloud campaign that ran while the local session was unreachable. **This document is a MAP,
not a mirror**: per the standing rule, `Project Steering/MODEL_REGISTRY.md` and the raw eval
JSONs are the only quotable sources for model facts — every number here is either
(a) MEASURED by this session with its artifact named, or (b) a pointer to the document that
owns it. When this doc and a pointed-to document disagree, the pointed-to document wins.

---

## 0. Rehydrate in three commands

```
git fetch origin claude/tanitad-resumption-handoff-92zx39
git checkout claude/tanitad-resumption-handoff-92zx39      # or merge into your line
# then read, in order:
#   Project Steering/STOP_2026-08-15_RESUME_RUNBOOK.md     (live state + resume recipes)
#   this file                                              (the campaign map)
```

The branch is also **PR #2** on `sayoood/TanitAD` (draft). Working tree was clean at stop;
the full suite was certified on the exact stop-state tree: **2 804 passed / 0 failed**
(MEASURED 2026-08-15).

**HF artifact map** (all public + **gated-manual** per PI policy; nothing was deleted):

| repo | contents |
|---|---|
| `Sayood/tanitad-v6` (model) | `v6F-SW-30k/`: **step-6250 checkpoint** under two names (`ckpt.pt`, `ckpt_final_stop.pt`, byte-identical, md5 `01a0c5e8`), `config.json` (the exact-flags record a strict resume needs), metrics/train_log/train.out, fp16 weights-only snapshot; `pbattery_2k5/` + `pbattery_5k/` (battery + echo-control results); `ops/` (shipped module set) |
| `Sayood/tanitad-ph0-aug120` (dataset) | `batch_*` (201 Alpamayo clips VLM+SAM3-labelled at 120°) · `fused_w120val/` (600 fused PH1 records) · `bridged_w120train_2400/` (5.72 GB, 4 802 files — the formerly stranded corpus) · `epcache_oodval_290/` (34.02 GB, 291 files) · `w120val_600/` ego spine · `g1_evidence/` (the graded sign sheet + crops). All counts verified far-side at stop. |
| `Sayood/tanitad-alpamayo2-augmentation` (dataset) | `records.parquet` — **23 644 rows = 4 729 clips × 5 tasks** (trajectory / meta_action / auto_labeling / vqa / grounding_via_vqa, with a `quantisation` column recording the quantized-run arms), `selection_manifest.json` (4 800), `vqa_bank_500.json` (MEASURED by direct read 2026-08-14) |
| `Sayood/tanitad-physicalai-w120-256x640cyl` (dataset) | the w120 corpus: train 2 403 files / 85 GB, val 603 / 21.2 GB, epcache-256px 3 053 / 349.5 GB (MEASURED listing 2026-08-14) |

---

## 1. The campaign, phase by phase — where each piece of work lives

*(Chronology since early August. Phases A–C largely predate this orchestrator's context and
are pointed at their own records; D–K were executed and verified by this session.)*

**A. Alpamayo-2 Super: run, quantize, augment** — pod4 was the A2 pod (`a2venv`,
`models/Alpamayo2-Super`, `alpamayo2_repo`). Products: the augmentation record set above,
plus pod-side `a2_batch_out/` (alpamayo_gt, meta_action, retime arms, `alpamayo_vs_flagship`
comparison). → ⛔ **CORRECTED 2026-08-15: there are NO chronicle rows in `PROJECT_STATE.md`**
(`grep` for `a2venv` / `4,729` / `23,644` / `alpamayo_vs_flagship` returns zero; the only
`alpamayo` hits there are three Opponent-Analyzer rows about the 32 B competitor). **The
authoritative record is now `MODEL_REGISTRY.md` §11.1** (counts, cost, completeness holes,
license), with the strategy index at `DataEng/DATA_STRATEGY.md` **v2.0** — v1.0 was a month
stale and carried no Alpamayo content when this handover cited it. Weeklies in
`Project Steering/Progress Reports/`. The quantisation arms are recorded per-row in
`records.parquet` itself — quote from there, not from prose ⚠️ **and note the card disagrees
with the parquet: 4,729 clips / 23,644 rows MEASURED, not the card's 4,800 / 23,999.**

**B. World-model / planner research line** →
`…/incoming/2026-07-23-frozen-wm-learned-planner/` (frozen-WM planner, amortised-MPC, value
model experiments), `…/incoming/2026-08-06-mpc-planner-design/` (`MPC_WM_DESIGN.md`,
`DIFFUSION_MPC_SYNTHESIS.md`), `…/incoming/2026-08-07-…/DIFFUSION_PLANNER_COMPARISON.md`
and `JEPA_PHYSICS_SURVEY.md`. This is the research basis the v6 design cites.

**C. v1arch / v5f findings that shaped everything** → the registry §§ and:
`LEAK_v1arch_val_2026-08-05.md` (v1arch val leak), `…/2026-08-06-v1-defect-triage/`,
`V5F_ARCHITECTURE_REVIEW.md`, `V5F_DATA_WIRING_AUDIT.md`, `V58F_FUSION.md` + the
`PREREG_W*` family (the W-wedge experiments), `EVAL_DOCTRINE.md` (T0/T1/T2 tiers — born from
the measured action-echo), the four-metric-families rule (binding, in `CLAUDE.md`). The
headline defects that motivated v6: **the T1 action echo** (open-loop lateral skill that
vanished closed-loop), **~99 % of the oracle gap longitudinal**, the **5-way mixed softmax**
tactical defect, and the **nav-echo** (a route head scoring 1.0000 by bijecting its own
input).

**D. v6 design + new training strategy** → `HIERARCHICAL_WM_REDESIGN.md`,
`HIERARCHY_VOCABULARY.md` (the g_str/g_tac token sets, factored LAT×LON),
`V6_TRAINER_DESIGN.md` (staged S-W → S-T → S-S → S-J, X3/X5 isolation),
`V6_TRAINING_MEASURES.md` (O1–O6), `V6_SIZING.md`, `V6_SIZE_VS_FRONTIER.md` (frontier
comparison; config table A–F), `V6_GO_PACKAGE.md`, `PI_DECISIONS_2026-08-12.md`.
**What is actually training: config E** — ViT-5 encoder (768×12, registers, RoPE),
ModernCausalBlock predictor 1024×12, 256×640 **cylindrical 120°** 9-ch input, `o5_k=60`
(the 6 s rollout contract), 336.5 M params, `param_budget` raised to 350 M —
verified against the run's own `config.json` (MEASURED 2026-08-14).

**E. E-ENC + the S-W run** → `E_ENC_RESULT.md` (plain-ViT 384×8 beat plain-ViT 768×12 at
step 500; ⚠️ **the ViT-5-form width question is UNMEASURED** — the running encoder is a
different architecture than either arm), `V6_SW_RUN_RECORD.md`. Run history this session:
OOM at k=60 fixed by gradient-checkpointed rollout (exact full-chain gradient, not truncated
BPTT); later the gc flag was **split** (`--enc-grad-checkpoint` / `--rollout-grad-checkpoint`,
`auto` = byte-identical old behaviour) after MEASURING 42.8 % mean GPU util with 20 GB free;
two gradient-spike episodes (~3 450–3 850 peaking gnorm 354 076, and ~5 150), both
self-recovered; stopped cleanly at **step 6 300, last ckpt 6 250**.

**F. The P-battery ported to v6 and its first results** → `PBATTERY_V6_FIRST_RESULT.md`.
The port (`tanitad/eval/v6_probe_trunk.py`) took eight failing runs to diagnose — the
blocker was architectural (probes *built* a v5 WorldModel) and the fix is a thin four-item
adapter; geometry and causal window now come from the checkpoint itself, never from v5
defaults. **Two scientific results:** (1) ⛔ the P1 speed row is an **ECHO** — R²(ẑ,speed)
0.995 collapses to −0.72 when v0 is shuffled (the FiLM v0 channel; third member of the
echo family after nav-echo and the T1 action echo); the `--speed-echo-control` flag now
exists and must accompany every battery run. (2) ⭐ the encoded-latent curve is **moving**:
R²(enc, speed, k=10) −2.30 → −0.74 between steps 2 500 and 5 000 — still negative, right
direction; the 10 k milestone is armed via `stack/ops/pbattery_watcher.py`. **P3/P6 have
never produced a v6 verdict** (the 5 k run predated the V6Grounding shim; wired now).

**G. PH0 pipeline (VLM + SAM3) validated and run at scale** →
`PH0_PIPELINE_VALIDATION.md` (3 of 4 channels pass; **B3 VLM grounding measured 2/23 →
demoted to diagnostic**, SAM3 owns pixels), `PH0_COVERAGE_AUDIT.md`, `PREREG_PH0_VLM.md`.
Production: 600 w120-val clips fully labelled; then the **Alpamayo-at-120°** pass:
of 4 729 augmented clips, 257 have w120 caches, 56 were already done, and the **201
runnable were all processed and pushed** (`aug120_pipeline.py` — per-batch pull/process/
push/delete, bounded disk). ⛔ **4 472 clips have no w120 cache**; the source stores the
120° camera as **chunked zips + per-feature parquet chunks**, so the build path is
chunk-index → `v2_compressed.py build --only-clips` (scoped in the stop runbook, not started).

**H. PH1 fusion — the ego/VLM/SAM3/Alpamayo combination strategy, implemented** →
`PH1_FUSION_STRATEGY.md` + `stack/scripts/ph1_fuse.py` (12 tests). Jurisdiction not
averaging; versioned corroboration checks; 2-of-3 vocabulary voting emitting the REAL v6
tokens (imported, cannot drift); conflicts recorded, never merged; ego layer
labels-only/never inference-whitelisted; deterministic scenario line. First run over the
600: **175 corroborations, 41 conflicts, 56 with the Alpamayo layer** (MEASURED). ⚠️ The
aug120 batches are **not yet fused** — first follow-up item.

**I. G1 sign-OCR, reviewed under PI delegation** → `G1_RESULT.md` +
`G1_SIGN_OCR_GRADING_SHEET.md` + the evidence sheet in `g1_evidence/` on HF.
**0/31 verifiable at pipeline fidelity → gate stays CLOSED** (sign text remains
extraction-only, enforced as `pending_g1_gate` in every fused record), and the larger
finding: **~⅔ of SAM3's own best "traffic sign" crops contained no sign at all** (scores up
to 0.94) — the sign class needs a threshold study before the fusion treats it as
authoritative for signs. Route to closure: native-res crops from the source chunks (same
machinery as the 4 472 build).

**J. Ops findings with teeth** (each cost real time; all are documented in place):
the HF push loop that reported success while its 3.5 GB payload failed 100 % of cycles
with a truncated error (private-quota exhaustion; fix = **public+gated-manual policy** +
`stack/ops/hf_push_loop.py` which verifies **from the far side** every cycle);
`POD_HANDOVER_2026-08-13.md §4b/4c`; the v6-vs-v5 checkpoint-layout compat fixes
(`ckpt_compat`, `replay/arms`, tests); the Orbis 2 analysis (`ORBIS2_ANALYSIS.md` — nearest
published analogue, inverse parameter allocation, task-matched data target: we are **139×
under** it hours-per-M-param) and its integration into `V6_DATA_REQUIREMENT.md` (data, not
parameters, is the binding constraint; levers P0/P1 ranked).

**K. Clean stop** → `STOP_2026-08-15_RESUME_RUNBOOK.md`: snapshot-then-kill by explicit
PID, checkpoint verified loadable before the kill was trusted, every archive verified by
repo listing, and the **twelve-item missing list**. Nothing of value remains only on a pod.

---

## 2. The open work, in priority order (from the stop runbook, unchanged)

1. **Resume S-W** (step 6 250 → 30 k ≈ 4.8 days on one A40; exact recipe in the runbook,
   including the venv torch trap and the strict-load semantics). Re-arm both safety loops.
2. **10 k P-battery** (watcher banked; will produce the first P3/P6 verdict) — always with
   the echo control.
3. **Fuse the aug120 batches** (fuser exists; one loop).
4. **The 4 472-clip chunk-index build** — the single biggest data job; also unlocks the
   G1 native-res re-run.
5. BEV (P8) probe port to v6 · SAM3 sign-threshold study · E-ENC in ViT-5 form ·
   DataLoader/batch measurement (42.8 % util says headroom exists) · `batch_00184` SAM3
   redo (8 clips) · OOD-290 redo with the fixed bridge · tasks 12/13/16 · sitclf
   provenance probe.

## 3. What this handover deliberately does NOT do

It does not restate registry numbers (v1→v5.8f rows, W-wedge results, T1 tables) — those
live in `MODEL_REGISTRY.md` and their eval JSONs, and copying them here is how three
documented errors propagated in July. It also does not claim the phases A–C narrative from
memory: those sections are pointers to their own primary documents, written at the time.

**Binding rules unchanged and in force** — parity key, vision-only-at-inference,
goal/situation disjointness, four metric families, T-tier stamps, the echo test
(*"does an input at inference contain something the thing being measured also produces?"*),
never-idle, ≥5 streams. `CLAUDE.md` carries them all, including this campaign's additions.
