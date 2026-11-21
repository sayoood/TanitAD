# E2-TAIL: the E2 rule on the TAIL-EMA weights of K2DC and K0 -- verdict NO-GO

Pre-registered before any tail number: `../PREREG_E2TAIL_2026-11-16.md` (landed c39d9fd3). Tail weights: `stack/scripts/ema_tail_weights.py`
(landed e8a4631a): W_tail = (EMA_6000 - 0.81872 EMA_5000) / 0.18128 = the exact EMA-weighted mean of the raw weights at steps 5001-6000,
with no step-0 weights (as any EMA at E3's 73,280 steps). Zero training. Thor read tree `read_tree_9697fa4df004_k2dc` (stage A) and
`read_tree_9697fa4df004_k2dc_raw1c48ee8f` (stage B, roll 3816129936e8 with `R9E2_CKPT_NAME=model_ema_tail5k6k.pt`). DEV = eval139 even
half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only.

## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)

| arm | EMA (E2) | RAW (E2-RAW) | TAIL (this read) | default cut (decides nothing) |
|---|---|---|---|---|
| K2DC | 0.3940 | 0.4912 | **0.4850** [0.3996, 0.5632] | 0.4807 |
| K0 | 0.4776 | 0.4729 | **0.4665** [0.3852, 0.5375] | 0.7469 |

* **Clause (a) on tail weights: PASS** (point rule as pre-registered): +0.0185. Per-arm CIs
  (`r9_e1_eval.py` bootstrap B 2000 seed 0) overlap; the paired interval is below.
* **M-map: CONFIRMED** (bar: K2DC tail >= raw - 0.02 = 0.4712; read 0.4850).
* At the default cut K0 leads by a wide margin, as in E2-RAW (K0's fitted pedge thresholds are near zero): the heads sit at different
  operating points, which is why the design scores (a) at fitted thresholds.
* Paired diagnostic of clause (a) (`paired_diag/FLOOR.md`, decides nothing; K2DC tail - K0 tail at own FIT thresholds, paired episode-cluster bootstrap B 2000, 68 episodes, T0): primary 5-20 m +0.0185 [-0.0029, +0.0443] not separated; 0-5 m +0.0653 [+0.0159, +0.1196] SEPARATED; 5-10 m +0.0301 [+0.0007, +0.0653] SEPARATED; 10-20 m +0.0135 [-0.0085, +0.0391] not separated; box ap2m_all -0.0000 [-0.0077, +0.0080] not separated.
* Tail guard: every dump.json records ckpt_md5 = the tail file's md5 (K2DC 893c790e, K0 f0ae6f32) and step 6000 (`stageA/dump_headers/`).

## Stage B -- the read's verdict (quoted, unedited from `stageB/read/E2_VERDICT.md`)

# E2 verdict: NO-GO  (K2 vs K0)

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (b) box AP@2 m head output no NMS: K2 > K0 (point) | **FAIL** | T0 (label level) | 0,1 |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **PASS** | T1 self-action open loop (S6c-DEV) | - |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

GO iff K2 beats K0 in point estimate on the map primary, box AP@2 m and the planner's S6c sidewalk rate, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4).
NO-GO names clause (b) box AP@2 m head output no NMS: K2 > K0 (point) [FAIL, seeds ['0', '1']] -- next lever: box lever: B2 perspective aux / B3 quality-aware presence + DN / B4 queries (REFCV9_DESIGN 4.3), re-read AP@2 m
NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

## M-lat (decides the DIAGNOSIS, not the GO)

* K2DC seed-0 LAT cross-track mean on tail weights: **0.1160 m** (EMA 0.1040, raw 0.1163; bars <= 0.1080 P1, >= 0.1123 P2) -> **P2 (init pull)**.
* Paired K2DC - K0 LAT_cross_mae_m (T1, 70 ep): s0 +0.0126 [+0.0063, +0.0194] separated (K0 0.1034); s1 +0.0091 [+0.0024, +0.0166] separated (K2DC 0.1137, K0 0.1046).

## Next (Rule Zero)

* committed outcome 2 ((d) fails, M-lat P2): next arm = K2DC share target 0.02 (~20.5 Thor-h, needs the PI); the cheaper alternative beside it = a planner-only L2-SP anchor toward init (new code, own prereg). Also failing: (b) (no branch of section 5 names it; reported as written, see the clause rows). E2-WISE (PREREG_E2WISE_2026-11-17.md, landed ad9f1cbe before these lateral rows existed) is triggered: NO-GO with (d) failing.
* The original E2 and E2-RAW verdicts stand as written.

## Provenance

* Stage A files: pulled from Thor `runs/E2TAIL_K2DC/stageA/` 2026-11-17, sha256-equal (18 files). Rasters and perc_det.pkl not
  banked (re-creatable from the tail files + read tree). Stage B read files: byte-exact copies of the S6c driver's pull (sha256-verified
  against Thor `runs/E2TAIL_K2DC/stageB/`).
