# PREREG_T2_DOM_LORA: the LoRA arm read for dominance at T2 (Master Mind, 2026-11-07 06:59 dev-box clock, from `date`; frozen from `DRAFT_PREREG_T2_DOM_LORA.md` before any L checkpoint has been read, while L_s0 trains on the PI's GPU pod)

## 1. The case
* The composite C0C reads **NOT-DOMINANT** at both seeds on the 48 read scenes (`PREREG_T2_DOMINANCE.md`, frozen `38aa9df3`; `D-V8F-T2-DOM-1`). Its largest failing term is the crawl: M1 +0.4073 / +0.4041 against P10, paired scene-cluster bootstrap.
* The crawl card's committed next is the perception lever: the LoRA arm L (`AMENDMENT_V8F_LORA_4.md`, frozen `626aa1e5`), with T2 as its tier.
* Since the draft, three reads have fixed the arm and its gates:
  * the attribution card (`PREREG_T2_ATTRIB.md`, frozen `f7d2cfa1`) read PT-ACTIVE and RT-WITHIN-FLOOR;
  * the T1-kin BLIND read (`PREREG_T1_BLIND.md`, frozen `9619414a`) read VISION-USED, so the render gap binds at T2;
  * the render-alignment map read ALIGN-FAILS-MECH (`D-V8F-T2-RALIGN-1`).

## 2. The arm
* **LC_s** = composite(recft(dagoacc(L_s))). The two post-training stages run with C0's argv (`v8f-dagoacc/M2b_s{s}_dagoacc/argv.json`, `v8f-recover/M2b_s{s}_recft_C0/argv.json`) on L_s's merged-token cache, with only the cache roots and `--init-ckpt` changed. The composite is stage 4's `t2v8f_c0c.CompositePolicy`. Nothing else changes: executor (playback), agents replayed at log time, renderer, starts.
* **The B′ control, BC_s** = composite(recft(dagoacc(B′_s))): the same pipeline on B′, M2b retrained on the pod's re-encoded cache.
* **Scenes:** the 48 that survive rule R-SCN in every stage, never re-drawn. A scene the renderer refuses in a new stage leaves every arm.
* **Stage:** the stage-5 driver with LC support (`raw/2026-11-07-t2-lc-support/`), on box 2.

## 3. When it runs
* On L_s0 as soon as L_s0 and its two post-training stages exist, whatever L's eval-141 verdict. The eval-141 PRIMARY is ADE6 at T1-class open loop; the defect read here is M1 at T2.
* L_s1 is read if it exists, under `AMENDMENT_V8F_LORA_4.md`'s stage rule.

## 4. The reads
* **Dominance (deciding):**
  * With both seeds: pin 4, `read_t2_dominance.py` (`7fd65ffd`), `--arm LC --arm-stage 5`, each arm against P10. The bars (D1 −0.2415 m, D2, D3), the 19 rows, the estimator and the outcome order are those of `PREREG_T2_DOMINANCE.md`.
  * With seed 0 only: pin 5 (`56a91667`, `--arm-seeds 0`). Its outcome SEED0-ONLY is never a counted DOMINANT.
* **Mechanism rows (decide nothing for dominance):**
  * LC − C0C on every row, paired over the same starts, read against |C0C_s1 − C0C_s0|. RT-WITHIN-FLOOR makes this row admissible.
  * LC − BC beside it.
* **Token-use gates (decide whether any T2 number of LC is evidence about L's perception):**
  * **(a) T1-kin:** L-BLIND − L is separated on ADE6 or on M1, as `t1_blind/t1blind_read.py` reads it. L-BLIND is L with every scene token replaced by the pinned BLIND mean `cd6a7ab5`.
  * **(b) T2:** LC-BLIND − LC on the same starts is separated on ADE6 or on M1, by the paired scene-cluster bootstrap.
* **Beside:**
  * the four families for every arm (STRATEGIC PAUSED, PI item 25);
  * LC's renders beside P10's;
  * the weight-space prediction of gate (b), recorded under `PREREG_WSP_LORA.md` (frozen `0529fcfb`) before gate (b) is read.

## 5. Outcomes, committed now
1. **DOMINANT-T2 at both seeds:** the scale signal, reported to the PI with the video evidence. If gate (a) or (b) fails, the signal stands as a driving result, and the report says that it is not attributed to L's perception.
2. **SEED0-ONLY** (only L_s0 exists): reported as written, never counted. L_s1 is then the next compute item.
3. **NOT-DOMINANT:** section 5 of `PREREG_T2_DOMINANCE.md` runs again on LC's failing terms. A separated M1 fall in LC − C0C without dominance names the next longitudinal lever.
4. **Gate (b) fails** (whatever the outcome above): the render gap blocks L's perception at T2. The next step for L is the RENC recipe (`PREREG_T2_RENC.md`, frozen prefix `425d3496`), applied on top of L's merged encoder under that card's rules.
* **If RENC reads RE-MOVES or RE-DOMINANT before LC's stage runs:** the arm LC-RE (LC with E′ trained on top of L's merged encoder, same folds) is read beside LC with the same reads. A DOMINANT on LC-RE is counted only under the RENC card's replicate rule. LC stays the deciding arm of this card.

## 6. Scope
* Tier T2: camera rendered at the executed pose, agents replayed at log time, playback executor. Estimator: paired scene-cluster bootstrap, B 2,000, seed 0, answering another draw of scenes. Training variance is answered by two seeds and by the B′ control. Inference is deterministic.
* No DOMINANT exists today.

## 7. Log (dated, append-only after the freeze)
