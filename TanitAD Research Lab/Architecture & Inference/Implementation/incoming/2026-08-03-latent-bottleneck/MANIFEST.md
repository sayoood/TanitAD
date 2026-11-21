# D-LATENT — deliverable manifest

**Stream** latent / world-model representation · **Date** 2026-08-03 · **0 pod GPU-h**, no training
pod touched (`tanitad-new` / `tanitad-pod4` untouched; Thor untouched). All compute on the dev box
RTX 4060 with `OMP_NUM_THREADS` set (the documented multi-arm trap).

## Artifacts and where they live

| artifact | repo path | in ONE place only? |
|---|---|---|
| **Main deliverable** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck/LATENT_BOTTLENECK.md` | no — staged in git |
| **Pre-registration** (both outcomes fixed in advance) | `Project Steering/PREREG_TEMPORAL_LATENT.md` | no — staged in git |
| ⭐ mechanism results | `…/2026-08-03-latent-bottleneck/results_mechanism.json` | no — staged |
| precision-ladder results | `…/2026-08-03-latent-bottleneck/results_precision_ladder.json` | no — staged |
| D-probe results (pass 2, corrected substrate; 35 arms) | `…/2026-08-03-latent-bottleneck/results_temporal_falsifier.json` | no — staged |
| ⭐ tables emitted FROM the JSON (nothing hand-transcribed) | `…/2026-08-03-latent-bottleneck/summarize.py`, `raw/summary_tables.txt` | no — staged |
| D-probe pass 1 (defective substrate, KEPT as evidence) | `…/2026-08-03-latent-bottleneck/raw/results_pass1_STACKAVG_INADMISSIBLE.json` | no — staged |
| approach-A cost measurement | `…/2026-08-03-latent-bottleneck/raw/temporal_kv_cost.json` | no — staged |
| runners | `…/2026-08-03-latent-bottleneck/run_mechanism.py`, `run_precision_ladder.py`, `run_temporal_falsifier.py`, `analyze_temporal_kv_cost.py` | no — staged |
| logs | `…/2026-08-03-latent-bottleneck/raw/run_log*.txt` | no — staged |
| **Instrument change** (reusable) | `stack/tanitad/eval/accel_probe.py` — added the `tdiff` / `abstdiff` adjacent-frame feature bases | no — staged |
| **Instrument test** | `stack/tests/test_accel_probe.py` — `test_adjacent_frame_bases` | no — staged |

### ⚠️ Lives in ONE place (dev box only) — derived caches, rebuildable from the staged runners

| file | size | rebuild |
|---|---|---|
| `C:/Users/Admin/tanitad-data/eval/dlatent_pixel_substrate.pt` | ~1.1 GB | `python run_temporal_falsifier.py --stage substrate` (≈95 s) |
| `C:/Users/Admin/tanitad-data/eval/dlatent_pixel_substrate_pass1.pt` | ~0.9 GB | pass-1 variant, kept for the documented defect; rebuildable by reverting `latest_frame` |

**Not staged and deliberately so:** the two `.pt` caches are derived data, deterministic from the
banked comma2k19 episode cache + the staged code, and too large for the repo. Neither is an input
that took real effort to produce (95 s each). The banked inputs they derive from —
`C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f/ep_*.pt` and
`…/idm_derived_accel_latents.pt` — pre-date this stream and are owned by the IDM stream.

## Reproduction

```bash
cd "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck"
OMP_NUM_THREADS=4 python run_mechanism.py        --n-boot 2000 --out results_mechanism.json
OMP_NUM_THREADS=4 python run_precision_ladder.py --n-boot 2000 --out results_precision_ladder.json
OMP_NUM_THREADS=6 python run_temporal_falsifier.py --stage all --n-boot 2000 \
    --out results_temporal_falsifier.json
OMP_NUM_THREADS=2 python analyze_temporal_kv_cost.py --out raw/temporal_kv_cost.json
```

## Escalations — these need someone else to act (repeated from §8, because a README is not a channel)

1. **`Project Steering/GATE_PROTOCOL.md`**: adopt the §6 latent screen as a pre-flight gate for any
   encoder-training authorisation. `dynenc-branchB` spent 40 k steps on a latent this screen would
   have rejected in minutes.
2. **`Project Steering/BACKLOG.md` B5** (frozen V-JEPA 2 video-pretrained encoder): top-ranked
   encoder experiment by this analysis; re-scope it to run the screen FIRST.
3. **`stack/scripts/idm_head.py` docstring** and
   `…/2026-08-03-idm-accel-recoverability/ACCEL_RECOVERABILITY.md` §4.7: the *"σ ≲ 0.1 m/s, ~47×"*
   precision requirement is estimator-specific (2-point centred difference). With the optimal
   9-point Savitzky-Golay derivative it is **σ ≲ 0.28 m/s, ~21×**.
4. **`Project Steering/BACKLOG.md` A7** (Delta-JEPA): second independent refutation — the true
   adjacent-frame difference bases are at the null and the direct-increment regression gives
   corr +0.0007.
5. ⭐⭐ **The APPEARANCE SHORTCUT (LATENT_BOTTLENECK.md §0.0, escalation 0) needs an owner.** A
   single static frame reads `speed` at 93 % of the full latent's accuracy and every pure-difference
   basis reads it at the null. If this holds off-highway it re-explains a large part of the
   longitudinal story, including the cross-rig collapse.
6. **`Project Steering/RETRACTION_LOG.md`**: the "single RGB frame" framing that this stream and the
   sitclf-temporal stream were both briefed with is factually wrong about our input
   (`in_channels=9`, D-015 3-frame sliding stack). Root-cause class: *an architectural claim
   inherited from a code READING rather than from a shape MEASUREMENT.*

> **⏹ Stale-blocker re-sweep 2026-08-16** (13 days on) — all six escalations re-probed at HEAD, CPU-only.
> **2 CLOSED · 4 STILL OPEN.**
>
> - **#1 — ✅ RE-CONFIRMED STILL TRUE.** `grep -cin "latent_screen" Project Steering/GATE_PROTOCOL.md`
>   → **0**. The instrument is real (`stack/tanitad/eval/latent_screen.py` + `stack/tests/test_latent_screen.py`)
>   but has **zero call sites outside its own test** — it is neither a protocol step nor wired into any
>   runnable path. The `dynenc-branchB` 40 k-step lesson this item cites is therefore still un-banked as
>   a gate. *(Same request, independently raised, at `…/2026-08-03-appearance-shortcut-audit/MANIFEST.md`
>   escalation #3 — two streams asked, nobody acted. That is the signature of a doc-buried merge request.)*
> - **#2 — ⚠️ STILL TRUE as written.** `Project Steering/BACKLOG.md:35` (B5, frozen V-JEPA 2) is unchanged
>   and carries **no "run the screen first" re-scope**. Partial credit only: `BACKLOG.md:77` (E8) now folds
>   "B5's frozen V-JEPA-2 control" into the E-ENC prereg — a different vehicle, not the requested re-scope.
> - **#3 — ✅ RE-CONFIRMED STILL TRUE.** `stack/scripts/idm_head.py:124-126` still states **"σ ≲ 0.1 m/s"**
>   and **"~47×"**; no Savitzky-Golay figure (σ ≲ 0.28 m/s, ~21×) appears anywhere in the file. A live
>   docstring still over-states the precision bar by ~2.2×.
> - **#4 — ⚠️ STILL TRUE.** `Project Steering/BACKLOG.md:22` (A7, Delta-JEPA) still reads as a live
>   *"HYPOTHESIS-class lead"* with no refutation note, despite two independent refutations.
> - **#5 — ⏹ SUBSTANTIALLY CLOSED, and the answer is the opposite of the fear.** The audit was run the
>   same day: `…/incoming/2026-08-03-appearance-shortcut-audit/APPEARANCE_SHORTCUT.md` §0.1 —
>   pre-registered **OUTCOME C (CORPUS-SPECIFIC)**: *"the still frame reads speed at the NULL on
>   PhysicalAI-AV"*, i.e. the shortcut is **true of comma2k19 highway and false of PhysicalAI-AV**, with
>   `results_p1_physicalai.json` as the artifact and thresholds fixed before any PhysicalAI number
>   existed. ⇒ it does **not** re-explain the longitudinal story off-highway. **Residual still open:**
>   that audit's own manifest records v5f and REF-C's ResNet trunk as *neither probed nor screened*, and
>   the cross-rig −2.4654 as *not attributed* — those need a GPU pass and still have no owner.
> - **#6 — ⏹ CLOSED.** The retraction was logged: `Project Steering/RETRACTION_LOG.md:2374`,
>   entry **`R-2026-08-03-latent`** — *"the PREMISE is FALSE, and the CONCLUSION it supported is
>   separately refuted"* — citing `refc.py:241 in_channels: int = 9` (:2389) and restated at :2420 and
>   :2500. Root-cause class recorded as asked.
>
> All verdicts MEASURED at HEAD. Swept by the 2026-08-16 stale-blocker sweep.

## What is NOT done

* ✅ **DONE** — the pre-registered D probe ran to a verdict: **OUTCOME V (VIDEO-LIMITED)**, with the
  admissibility gate met (6 hand-built arms separated on `speed`; 0 of 35 arms separated on
  `long_accel`). See `LATENT_BOTTLENECK.md` §0.0 and §4.4.
* ⛔ **NOT DONE — the appearance-shortcut audit (RANK 1 / escalation 0)** on PhysicalAI-AV, v5f and
  REF-C. This run measured the shortcut only on comma2k19 highway, where it is largest by
  construction. It is the highest-value follow-up in the document and it has **no owner**.
* The §6 screen has **not** been run on v5f, on REF-C's ResNet trunk, or on PhysicalAI-AV. Those are
  the arms the programme is actually deciding about.
* No retrain was launched and none is proposed here without the screen first.
