# P6_SPEED_RESULT (H-P6-SPEED-1) -- tier T0, one checkpoint, deterministic planner

- checkpoint: `F:\TanitAD_p6read_work\ckpt\run\model_final.pt` sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`
- prereg sha256 `8e8aa3cac3762ae4351dc09ef8842945b2f8d6ff82aeb849ee22811b594c0e6a`

## VERDICT: **REFUTED**

- conditions: {'A_head_calibration': False, 'B_not_subset_driven': True, 'C_behaviour': False, 'pessimism_persists': True}
- sub-cases: []
- inputs: {'G_TTC': -0.19020963776756433, 'G_NC': -0.15737814281911489, 'S_TTC': -0.17973128062413773, 'S_NC': -0.18020372484658195, 'T_TTC': -0.05114761608968947, 'T_NC': -0.045015891717776785, 'share': 0.22355169210100873, 'dA_hi': -0.08354272223584894, 'dB_hi': -0.03665938770331686}

## Gates

| gate | PASS | detail |
|---|---|---|
| G6-PIN | True | `{}` |
| G6-ARGV | True | `{}` |
| G6-DISJOINT | True | `{"n_c3_logs": 123, "overlap_navtrain_yaml": [], "n_train_logs": 1075, "overlap_pod_train_logs": []}` |
| G6-0 | True | `{"n_equal": 1058, "n": 1058, "RED_argmin_disagree": 1058, "RED_flip_p3_disagree": 843}` |
| G6-1 | True | `{"n_within_tol": 1058, "n": 1058, "max_abs_m": 0.0, "RED_cyclic_shift_n_failing": 1058}` |
| G6-2 | True | `{"ego_bit_equal": 1058, "n": 1058, "cruise_formula_equal": 1058, "n_cruise_formula": 154, "n_cruise_file": 154, "RED_perturbed_ego_still_equal": 0}` |
| G6-3 | True | `{"n_bit_equal": 1058, "n": 1058, "max_abs_diff": 0.0, "RED_shift_y2m_n_differ": 742}` |
| G6-4 | True | `{"shape": [1058, 64], "all_finite": true, "all_valid": true, "max_abs_diff": 0.0, "RED_perturbed_EP_max_abs_diff": 0.004166666666666763}` |
| G6-FP32 | True | `{"n_values_in_32_64m": 16604, "share_on_0.25m_grid": 0.0, "bar": 0.01, "RED_snapped_n": 16780, "RED_snapped_share": 1.0}` |
| G6-REPRO | True | `{"max_abs_diff_vs_pinned_reference": 0.0, "legacy_vs_p1_diag_all_max_abs_diff": 0.0, "RED_swap_flips_every_gap_sign": true, "RED_argsort_of_argsort_tie_block_max_abs_diff": 116.0}` |
| G6-CKPT | True | `{"dump_ckpt_sha256": "968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1", "expected": "968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1"}` |

G6-ARGV quoted: `{"labeller": "online", "label_backend": "pdm", "nav_mode": "command", "meta_nav_mode": "command", "metric_cache": "/workspace/data/refe_navtrain/metric_cache", "yaw_wrap": "off", "epochs": 25, "argv": ["train.py", "--backbone", "vitl16", "--targets", "/workspace/data/refe_navtrain/train_grow_cmd_3c", "--images", "/workspace/data/navtrain_pixels", "--calib", "/workspace/data/refe_navtrain/train_grow/calib_table.json", "--metric-cache", "/workspace/data/refe_navtrain/metric_cache", "--scorer-sees-trunk", "--yaw-wrap", "off", "--nav-mode", "command", "--labeller", "online", "--label-workers", "6", "--label-chunks", "2", "--epochs", "25", "--epoch-unit", "scenes", "--batch", "4", "--accum", "64", "--lr", "2e-4", "--weight-decay", "0.01", "--amp", "bf16", "--score-w", "0.1", "--tf32", "--compile", "--workers", "2", "--log-every", "1", "--ckpt-every-min", "20", "--out", "/workspace/data/refe_runs/p6_paper_faithful", "--resume"], "dump_rule": "navsim_v1", "dump_n_outputs": 6, "dump_nav_mode": "command"}`

## Statistics (paired log-cluster bootstrap, B=10,000, seed 20260927, 95 % percentile)

| head | S_h (all CRUISE) | T_h | pred (paired) | true (paired) | G_h gap [CI] | n paired |
|---|---|---|---|---|---|---|
| NC | -0.180 (n=154) | -0.045 (n=49) | -0.202 | -0.045 | -0.157 [-0.239, -0.079] | 49 |
| DAC | -0.170 (n=154) | -0.025 (n=138) | -0.178 | -0.025 | -0.154 [-0.266, -0.050] | 138 |
| EP | +0.301 (n=154) | +0.443 (n=153) | +0.304 | +0.443 | -0.139 [-0.204, -0.087] | 153 |
| TTC | -0.180 (n=154) | -0.051 (n=66) | -0.241 | -0.051 | -0.190 [-0.272, -0.102] | 66 |

Weighted CRUISE slow-pick share P6: **0.224** [0.117, 0.348] (u400 0.412 [0.275, 0.569])
Delta vs u400 banked: {'point': -0.18888082662164668, 'lo': -0.31313225594463107, 'hi': -0.08354272223584894}
Delta vs u400 without lane: {'point': -0.1428571428571429, 'lo': -0.2667380999089826, 'hi': -0.03665938770331686}

## Descriptive (cannot change the verdict)

```
{
 "LONGITUDINAL": {
  "slow_share_weighted": {
   "point": 0.22355169210100873,
   "lo": 0.11718458337937954,
   "hi": 0.3482665362064991,
   "n_cruise_zero": 91,
   "n_cruise_random": 63,
   "share_zero": 0.25274725274725274,
   "share_random": 0.2222222222222222,
   "share_unweighted": 0.24025974025974026,
   "n_draws_used": 10000
  },
  "slow_share_unweighted": 0.24025974025974026,
  "overspeed_share_weighted": {
   "point": 0.5348432693237568,
   "lo": 0.3558287315649023,
   "hi": 0.6940726854268349
  },
  "overspeed_share_unweighted": 0.474025974025974,
  "plan_v4_minus_gt_v4_mps_cruise": [
   0.8415022847508983,
   -0.15212907325933347,
   1.6796205987465593
  ],
  "mean_plan_v4_over_v0_cruise": [
   1.1105746239170826,
   0.950659480297179,
   1.2508638693660519
  ],
  "u400_references": "plan v4 minus GT v4 -0.535 [-1.391, +0.221] m/s (refe_ego_probe_20261106/RESULT.md); mean plan v4/v0 0.934 [0.803, 1.054] on A0 (INHERITED)"
 },
 "LATERAL": {
  "ade_to_human_future_m_cruise": [
   1.7936869877557047,
   1.5494065287086296,
   1.9956274815223194
  ],
  "n_ade_masked_by_frame_control": 0,
  "abs_y_at_4s_m_cruise": [
   3.8668956584357597,
   3.179852425582709,
   4.478038467179789
  ],
  "dac_failure_rate_cruise": [
   0.09090909090909091,
   0.047611003861003864,
   0.1366906474820144
  ],
  "ddc_mean_cruise": [
   0.9805194805194806,
   0.9523809523809523,
   1.0
  ],
  "note": "what the proposals allow, not tracking; ADE masks tokens whose frame control exceeds 1 mm"
 },
 "SAFETY": {
  "nc_failure_rate_pick_cruise": [
   0.012987012987012988,
   0.0,
   0.03546099290780142
  ],
  "ttc_failure_rate_pick_cruise": [
   0.05194805194805195,
   0.02127284986514834,
   0.08823529411764706
  ],
  "changed_picks_vs_u400": "n/a: P6 and u400 propose different sets, an index comparison is meaningless; the lane-free re-pick count is in u400_without_lane_control"
 },
 "TACTICAL": {
  "status": "n/a",
  "n": 0,
  "reason": "no manoeuvre-decision label on these tokens; REFe emits no tactical output here"
 },
 "STRATEGIC": {
  "status": "PAUSED",
  "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md item 25; not a pass, not omitted"
 }
}
```

## Limits (prereg section 7)
- One checkpoint, deterministic planner: no claim about training or inference variance (H-ESTIM-SEED-1). HOLDS means this checkpoint has the property, not that online labelling produces it.
- P6 has no lane factor; the lane-free u400 control bounds that part.
- 154 CRUISE tokens over 51 logs.

## Ambiguities applied
- S_h is the mean of rho(pred, speed) over ALL CRUISE tokens where defined (card section 4). u400's -0.546 / -0.491 in the card's section 2 table are the PAIRED-subset values; the all-token S_h is -0.479 (TTC) and -0.443 (NC). Both are reported; (B) uses S_h as defined in section 4.
- Sub-cases N2 and N3 can overlap (A holds, B fails, C holds); every matching sub-case is listed. A not-proven outcome matching none of N1-N3 is labelled N-unnamed.
- G6-DISJOINT is evaluated against splits/navtrain.yaml (local) AND the pod's metric-cache log list (the logs of --targets); both must be disjoint.
- A gate that could not be evaluated (missing input) is RED, so the read ends in INSTRUMENT FAILURE rather than a partial verdict.
