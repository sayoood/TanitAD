# PREREG_T2_ATTRIB: what post-training and the pod route each do at T2 (Master Mind, 2026-11-07 02:28 dev-box clock, from `date`; written before any cell of either stage exists)

## Why this read exists
* The LoRA arm L is M2b plus LoRA, trained on the PI's GPU pod (RTX 4000 Ada) from frames rebuilt there (`AMENDMENT_V8F_LORA_4.md`, frozen `626aa1e5`).
* C0C, the comparator in `DRAFT_PREREG_T2_DOM_LORA.md`, is the composite of C0, and C0 is M2b → DAG-OACC (3,000 steps) → recovery fine-tune (3,000 steps) (`t2v8f/HANDOFF_A.md`; `v8f-dagoacc/M2b_s{0,1}_dagoacc/argv.json`, `v8f-recover/M2b_s{0,1}_recft_C0/argv.json`, each `--init-ckpt` the previous stage). C0 was trained on box 2's cache.
* So LC − C0C would carry three terms at once: LoRA, the two post-training stages, and the pod route (pod GPU tokens from pod-rebuilt frames, plus one more training run). The draft card's sentence "nothing else changes" does not hold for that row as written.
* Two stage-5 runs on box 2 separate the last two terms before L exists, so the LoRA card is frozen on measured facts.

## The arms (stage-5 driver `t2v8f/code/t2v8f_stage5.py`, banked `01095679`; the composite is stage 4's `t2v8f_c0c.CompositePolicy`, only the weights differ)
* **MC_s** = the composite of plain M2b_s (box 2 `D:\tanitad\runs\v8f_mini_full\M2b_s{0,1}\ckpt_last.pt`, sha256 `601cb435…c6f8` / `6104fb15…c2e6`, equal on the dev box and box 2, checked 2026-11-07 02:2x).
* **BPC_s** = the composite of B′_s, M2b retrained on the pod's re-encoded cache (`AMENDMENT_V8F_LORA_4.md` item 3; argv = M2b's with only the cache roots changed). Built when both B′ runs end (`summary.json` with `done: true`); its checkpoint sha256 is recorded in the log below before the stage launches.
* Order on box 2 (one job at a time): MC, then the B′ eval-141 read (F′, owned by the LoRA card), then BPC.
* Scenes: the 48 that survive rule R-SCN in every stage read; a scene refused in a new stage leaves every arm.

## The statistic and the estimator
* Rows: the pinned dominance read's 19 rows per seed (`t2v8f/code/read_t2_dominance.py`, pin 4 `7fd65ffd`, `--arm NAME --arm-stage 5`), each arm against P10 as written.
* Arm-against-arm differences: A − B per row, per seed, paired over the same starts, by the dominance card's estimator (paired scene-cluster bootstrap, B 2,000, seed 0). The instrument is `read_t2_pair.py`; it is admissible only after its identity holds: with A = C0C and B = P10 it must reproduce the read of record's 19 C0C − P10 rows at both seeds exactly. If it does not exist or fails its identity, the pair rows are NOT READ, named, and only the per-arm rows against P10 are reported.
* The rig's seed floor for each row: |X_s1 − X_s0| of the same arm, as the dominance card reads its seed gap.

## Outcomes, committed now
1. **The post-training term, PT = C0C − MC** (same device, same base runs; only DAG-OACC + recovery fine-tune differ).
   * **PT-ACTIVE** if any of the 19 rows is separated at both seeds. Then the LoRA arm read for dominance includes the same two stages: LC = composite(recft(dagoacc(L))), the stages run with C0's argv on L's merged-token cache with only the cache roots changed, and its control is the same pipeline on B′.
   * **PT-INERT** otherwise. Then LC = composite(L), and the post-trained arm is read beside, deciding nothing.
2. **The route term, RT = BPC − MC** (same pipeline; the pod route plus one training run differ).
   * **RT-WITHIN-FLOOR** if, on M1 crawl share, ADE6 and along-track MAE, RT is not separated at either seed, or |RT| is below that row's seed floor of MC at both seeds. Then LC − C0C is admissible as the mechanism row, beside LC − (the B′ control).
   * **RT-MATTERS** otherwise. Then the mechanism row is LC against the B′ control only, and LC − C0C is reported beside, deciding nothing.
3. **Any arm here that reads DOMINANT-T2** on the pinned read at both seeds is reported as written with its renders beside P10's. It is not counted as the scale signal until a replicate (same flags, a new training seed) reads DOMINANT too, because two arms are read here rather than one.
4. **Beside, deciding nothing:** the four families per arm (STRATEGIC PAUSED, PI item 25), MC and BPC against C0C on every row, and each arm's renders.

## Scope and evidence
* Tier T2 (camera rendered at the executed pose, agents replayed at log time, playback executor), RR beside. Training variance enters RT (B′ is one more training run) and is read against the seed floor; inference variance: none (the planners do not sample).
* Nothing here is a driving claim beyond the arms named. No DOMINANT exists today.

## Log (dated, append-only after the freeze)
* 2026-11-07 03:14 (dev-box clock, from `date`), Master Mind. B-prime done at both seeds (pod `summary.json` done: true, 19,164 steps, 4,512 clips, 306,566 windows, train sha256 `37f4f9ee...` equal to M2b's; argv sha16 `a2b68127d72210f0` / `c226fba964961dc5`). Pulled and sha-verified on the dev box (`lorapod_eval/bprime/pod_sha256.txt`). **BPC checkpoints, recorded before the BPC stage launches:** BPC_s0 sha256 `b9aaf33e458c4da7a4e38becae905fbdd20b35b7dccf8858c973791e7e61c1a7`, BPC_s1 `a4bcc4722a37cfada1ac5640bed900e47d29d2278632a889fb1be1fed9ebd3a9`. The stage-5 driver and cell bytes are the ones the MC package was built from.
* 2026-11-07 04:10 (dev-box clock, from `date`), Master Mind. **MC read; the post-training term is PT-ACTIVE.** MC stage 5 ran on box 2 02:29:42 to 04:06:23 (package tar `355217d5...`), pulled complete (1,027 files, 44,472,955 bytes equal on both sides). R-SCN: ATTRIBUTED, the same 14 scenes as stage 4 (`stage5/MC/EXCLUSIONS5.json`; a first build without the cell logs read UNATTRIBUTED and is kept aside). Pinned read (pin 4) on MC: NOT-DOMINANT at both seeds (s0 ADE6 against P10 +1.4552 m; `stage5/MC/score_dom/`). Pair read C0C - MC (`read_t2_pair.py`, `stage5/MC/pair_C0C_MC/PAIR_T2.md`), separated at BOTH seeds: ADE6 -0.6659 / -0.3064 m, along-track MAE -0.7109 / -0.3920, accel MAE -1.0108 / -1.2948, DK-HW -0.4326 / -0.3071, and **M1 crawl share +0.0580 / +0.1121 (C0C crawls MORE than MC)**. By section "Outcomes" 1: LC = composite(recft(dagoacc(L))) on L's merged-token cache with C0's argv and only the cache roots changed; its control is the same pipeline on B-prime. Beside, deciding nothing: MC's own seed floor on ADE6 is 0.5089, so the s1 ADE6 difference sits inside it; the rule as written reads separation, and it holds.
