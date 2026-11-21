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
4. **The arm's post-training stages and its control (added 2026-11-07 02:31; decided by `PREREG_T2_ATTRIB.md`, frozen `f7d2cfa1`, before any of its cells existed).** C0 is M2b → DAG-OACC (3,000 steps) → recovery fine-tune (3,000 steps); L is M2b + LoRA. So "nothing else changes" needs the same two stages on L, and LC − C0C also carries the pod route. The attribution read fixes both before this card freezes: PT-ACTIVE ⇒ LC = composite(recft(dagoacc(L))) on L's merged-token cache with C0's argv and only the cache roots changed, and the control is the same pipeline on B′; PT-INERT ⇒ LC = composite(L). RT-MATTERS ⇒ the mechanism row is LC against the B′ control only; RT-WITHIN-FLOOR ⇒ LC − C0C stands beside it.

## Log (dated)
* 2026-11-07 00:42 (dev-box clock, from `date`), Master Mind. FOURTH PIN of `t2v8f/code/read_t2_dominance.py`: sha256 `7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9` (AGENT Q; third pin `d6664718` kept as `read_t2_dominance.pin3.py`). It adds `--arm NAME` (default C0C) and `--arm-stage N` (default 4), plus `--arm-result`, `--arm-root`, `--arm-exclusions`, `--arm-overlay` and `--arm-gate-log` for a stage beyond 4. The union of R-SCN exclusions over every stage read leaves every arm. **Identity, binding: with the defaults it reproduces the read of record exactly** (`logs/q_identity.json`, identical: true; decision, gates, both seeds' 19 rows, seed gap, r_scn and the md equal; only the script sha, the argv and the append-only card's whole-file sha differ, the frozen prefix intact). Tests: 14 new and 57 existing green; 80 of 80 mutations RED on the working copy. The full-suite close-out is in flight (`logs/Q_FINISHED.txt`). Entered before any stage-5 cell exists.
* 2026-11-07 02:31 (dev-box clock, from `date`), Master Mind. Open item 4 added: C0's two post-training stages and the pod route are confounds of LC − C0C. `PREREG_T2_ATTRIB.md` (frozen sha256 `f7d2cfa1dcef07bc4b8a90433325095822901677e8b1b37d7fc20d1770ae2895`, 02:28:30) reads them on box 2 with two stage-5 arms, MC (composite of plain M2b; launched ~02:30) and BPC (composite of B′), before L exists. Entered before any cell of either arm exists.
