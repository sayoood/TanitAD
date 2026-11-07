# PREREG_V8F_TOWN: does the tactical layer's speed decision follow the car's own speed on its own states? (Master Mind, 2026-11-07, written before any number of it exists; prompted by the PI asking for each layer's prediction quality)

## Why this card exists
* The failing T2 term is the crawl: C0C − P10 M1 crawl share +0.4073 / +0.4041, along-track +1.4164 / +1.2962 m (`DOMINANCE_T2.md`). Post-training adds +0.0580 / +0.1121 over the plain base (`PREREG_T2_ATTRIB.md`, PT-ACTIVE).
* The tactical code is the input O depends on most. With another episode's code O is 4.4 to 6.4 m worse at 6 s (`H-V8F-TAC-1`), and the true code would gain 1.05 to 1.30 m, mostly longitudinal.
* T-lon's largest single dependence is the ego history: +0.247 / +0.261 macro-F1 (`H-V8F-LISTEN-1`).
* Every T-layer number so far was read on LOGGED states (T0 / T1-class open loop). T's own decisions on the model's own states were never dumped (`H-V8F-T1-FAMILIES-1`: "the model's own T-code at T1 stays NOT COMPUTED").
* **The hypothesis (H-V8F-TOWN-1):** on its own states, T-lon reads the car's own slowness as a reason to stay slow. The deficit feeds back through T into O, which is a candidate mechanism for the crawl.

## Arms, data, tier
* C0 s0 / s1 (`v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt`), with the composite flags of the BLIND read (`--smooth sg72 --scene reindexed`).
* eval-141 (`sfresh_inputs/cache`, 2,399 windows, 141 episodes), T1-kin m = 10: own executed kinematics, logged cameras.
* The T1-kin tool of record (`v8f_t1_eval_ri.py`, sha256 `0417c838...`) gains a dump of T's outputs at every replan: lat / lon softmax and constraints. **Identity, binding:** with the dump on, every executed track equals the tool of record's bit for bit on the full run. If not, nothing below is read.
* **Twin replans:** at every own-state replan, T is also evaluated on the LOGGED state at the same time index (logged ego history, same frames, same nav). This changes nothing in the rollout.
* GT tactical labels are the training labels at that logged time (ego and future may be used for labels).

## The statistics (per seed; episode-cluster bootstrap over 141 episodes, B 2,000, seed 0)
* **Shift:** per replan, d = E_own[lon class index] − E_logged[lon class index], where the expectation is over T-lon's softmax and the classes are ordered from slowest to fastest. The order is taken from the label definition and recorded before data.
* **Deficit:** per replan, Δv = v_logged − v_own at the replan time (m/s).
* **FEEDBACK slope:** the slope of d on Δv over all replans with |Δv| ≥ 0.5 m/s. It is fitted per bootstrap draw and read with its interval.
* **Quality:** T-lat and T-lon macro-F1 against GT, own minus logged state (paired over the same replans), and the constraint RMSE own minus logged.
* **Beside, deciding nothing:** the share of own-state replans whose lon argmax is a slower class than GT's, against the same share on logged states. The S layer is not read here (STRATEGIC PAUSED, PI item 25).

## Controls (binding)
1. **Identity:** the dump run's executed tracks equal the tool of record's, max |Δ| exactly 0.0.
2. **Twin identity:** where the own state equals the logged state (the first replan of every window, before any action executes), d is exactly 0.0 and both softmaxes are equal.
3. **Deliberate regression:** a copy of C0 with T's ego-token input zeroed (`e_ego` weights and bias 0, inside T only). The slope must read not separated from 0 at both seeds. A slope that survives it means the probe is not reading T's use of ego.

## Outcomes, committed now
1. **T-FEEDBACK:** the slope is below 0 and separated at both seeds (a larger deficit moves T-lon to slower classes), AND T-lon macro-F1 own − logged is separated-worse at both seeds.
   * The crawl has a tactical feedback term.
   * Next lever: T trained on its own states. That is the DT2 / RECOVER recipe applied to T's labels, under its own card. Until then, a zero-training read: O fed T's LOGGED-state code at T1-kin. If that removes the crawl excess at T1-kin, the share T carries is measured.
2. **T-STABLE:** the slope is not separated at either seed, and T-lon F1 own − logged is not separated-worse at either seed.
   * T is not the crawl source.
   * The crawl lives in O's reading of its own ego state. Next: the same twin read on O (own plan against O fed the logged ego history).
3. **MIXED:** anything else, reported as written. The next step is named in the result.

## Scope
* Tier T1-kin for the own-state rows, T0 for the twin rows. This is a mechanism read, not a driving claim, and never pooled across tiers.

## Log (dated, append-only after the freeze)
* 2026-11-07 08:53:17 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 4676 bytes above this log, sha256 `707e29f388bac0cd3d0b077b5be07a93be29d82fdd7b06859070ba169bda56b2`, before any number of it exists.
