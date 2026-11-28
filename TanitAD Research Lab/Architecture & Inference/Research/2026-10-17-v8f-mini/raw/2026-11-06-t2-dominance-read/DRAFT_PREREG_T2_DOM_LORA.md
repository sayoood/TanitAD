# DRAFT PREREG_T2_DOM_LORA: the LoRA arm read for dominance at T2 (Master Mind, 2026-11-06 ~22:10; a DRAFT, frozen before any LoRA arm exists)

**The case.** `PREREG_T2_DOMINANCE.md` (frozen `38aa9df3`) read **NOT-DOMINANT** at both seeds on 48 of 62 scenes (`t2v8f/stage4/score/DOMINANCE_T2.md`, 2026-11-06 22:05).
* The failing term with the largest margin is the crawl. MEASURED, C0C − P10, paired scene-cluster bootstrap B 2,000, seed 0:
  * M1 crawl share: +0.4073 [+0.3180, +0.5002] at s0 and +0.4041 [+0.3033, +0.5027] at s1;
  * along-track MAE: +1.4164 m and +1.2962 m;
  * ADE6: +0.7893 m and +0.6399 m;
  * TAC-LON: separated-worse at both seeds.
* LATERAL is not the problem: yaw-rate, curvature, cross-track and lateral-acceleration MAE are separated-better at both seeds.
* The dominance card's section 5 (2) routes D2-on-M1 to the crawl card's lever. The crawl card read CRAWL-PERSISTS (a): BLIND is not separated from C0, so the frozen tokens do not carry the cue. Its committed next is the perception lever, `PREREG_V8F_LORA`, *"with this tier as its evaluation and this read as the case"*.

## The arm
* **LC_s** = the composite recipe of C0C applied to **L_s**, the LoRA arm of `PREREG_V8F_LORA` at seed s. Nothing else changes: the composite, the executor (playback), the agents replayed at log time, the renderer, and the starts.
* **Comparators:** P10 (stages 2 and 3, rows reused, the same starts) for the dominance read; C0C (stage 4, rows reused) for the mechanism row.
* **Scenes:** the 48 that survive rule R-SCN in every stage, never re-drawn and never replaced. A scene the renderer refuses in the new stage leaves every arm, by the same rule.

## The read
* **Dominance:** the pinned `read_t2_dominance.py` logic (`d6664718`) with C0C replaced by LC. It needs a re-pin that takes the arm name as an argument. The bars (D1 −0.2415 m, D2, D3), the 19 rows, the estimator and the outcome order are unchanged.
* **Mechanism row (decides nothing for dominance):** LC − C0C on M1 crawl share, per seed, paired over the same starts.
  * Does the perception lever lift the crawl at all? A separated fall is the mechanism.
  * Its size is read against |C0C_s1 − C0C_s0| = |+0.4041 − +0.4073| on this rig's seed gap.

## Open, decided before freezing (the PI's or the MM's, named)
1. **When the T2 stage runs.** The MM recommends: on L_s0 as soon as it exists, whatever the eval-141 verdict.
   * Why: the eval-141 PRIMARY is ADE6 at T1-class open loop, and the defect this read found is M1 at T2. An arm that fails ADE6 by 2F could still lift the crawl.
   * Cost: about 1.6–2.5 h of box 2 per seed (MEASURED stage-4 wall: 20:18 to 21:53 for 62 scenes × 4 cells).
2. **Outcomes, committed at freeze:** DOMINANT-T2 is the scale signal, with the video evidence (T2 renders of LC beside P10). NOT-DOMINANT runs section 5 of the dominance card again. A separated M1 fall without dominance names the next longitudinal lever.
3. **Seed:** both seeds, as the dominance card. If L_s1 does not exist (the LoRA card's stage rule stops at seed 0), the read is reported as seed 0 only and is not a counted DOMINANT.
