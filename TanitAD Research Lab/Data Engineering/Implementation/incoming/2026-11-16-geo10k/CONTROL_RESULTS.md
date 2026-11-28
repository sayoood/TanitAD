# GEO10K control results (all MEASURED on the pod CPU, 2026-11-16; raw JSON in `labels/controls/` of the HF repo and in `F:/TanitAD-scratch/geo10k/controls/`)

Overlap of the 10,000 selected clips with the existing corpus: 0 of 10,000 (selection vs the 4,572-clip cache manifest, MEASURED). So the known-value control uses 20 existing-corpus clips (every 228th of the cache manifest, list in `control_clips20.txt`), run through the identical path (camera timestamps and egomotion re-fetched from NVIDIA, poses, clock, lead input, v9 build) and compared with the existing release rows (HF `Sayood/tanitad-refcv8-v9-labels`, md5 f63ece41..., 4,369 clips; all 20 controls are in it).

| step | control | result | regression arm (must go RED) |
|---|---|---|---|
| v2-manifest poses (g2) | 20 control clips vs `train4572_v2manifest.pt` | 20 of 20 bitwise equal, max abs diff 0.0 | control clip fed another clip's egomotion: max abs diff 167.6, RED |
| native tracks (g2) | 20 vs the existing `native_tracks.pt` | 20 of 20 equal | (same wrong-egomotion arm) |
| clip clock (g3a, `fit_clock` unedited) | 20 vs `refcv8_clip_clock_sidecar_ts33.jsonl` | 19 of 20 bitwise; the 20th differs by 4.8e-15 s in grid_start (1 ulp class); 0 refused | n/a |
| v9 labels, WITH records, lead, lc (ctlA) | 20 clips, 118 row fields | 66 fields bitwise equal on all 20 clips; 52 differ only at float noise, max abs 3.56e-12 (largest: `rcH_A30_psi`); 0 fields differ above 1e-9 | wrong-egomotion arm: 55 of 118 fields differ above 1e-9 (largest 2,400 m in `nav_lookahead_m`), RED |
| v9 labels, NO records, existing lead, no lc (ctlC) = the new-clip path | same 20 clips, 115 fields | 56 bitwise equal; 8 fields differ materially: `goal_y`, `goal_w`, `goal_lc_prov`, `vlm_has_cot`, `vlm_tl_probe_negative` (records) and `lc_frac`, `lc_measurable`, `lc_measurable_raw` (lc input); every other field is within 1e-9 of the release | n/a |
| v9 labels, NO records, NO lead, no lc (ctlB) | same 20 clips, 110 fields | 20 fields differ materially: the 8 above plus `lead_cov_s`, `lead_follow_frac`, `lon_allowed`, `lon_allowed_v7`, `lon_cls`, `lon_cls_computed`, `lon_d_reach_m`, `lon_t_reach_s`, `lon_t_start_s`, `lon_truncated`, `lon_v7id`, `lon_v_target_ms` | n/a |
| speed sidecar (g6) | `SPEED_BAND.v_hi_ms` from geometry (`cot=None`) vs the value stored in `s2_labels_v8_train.jsonl.gz` | 20 of 20 equal | a clip given another clip's value: RED |
| context fields (g5) | ALL 4,659 clips of `tanitad-context-labels-v1`: country, month, hour_of_day, period_of_day, sun_band_est, chunk | 4,659 of 4,659 equal in each of the 6 columns | hour of clip 0 shifted by 12: 1 mismatch, RED |

Reading of the table:
* The geometry path reproduces the existing labels to float noise (cross-environment BLAS class, the same family as the 1e-14 to 4e-4 noise already recorded for pod vs dev box labellers). ctlA is the control that says the egomotion lookup, poses, clock and builder are right.
* ctlC says what a clip WITHOUT a record loses: the VLM goal bits and the VLM flags, and the lane-change diagnostics. Nothing else.
* ctlB says the `lon_*` classes and the lead aggregates depend on the lead input, so the lead pass matters for the 10k.

NOT CONTROLLED (stated, not hidden):
* The lead pass itself: no obstacle join exists for the 20 existing-corpus controls, so the 10k lead file (9,559 clips with join rows) has no known-value control. Evidence it is sane: the join's own `t_s` agrees with the clip clock (median residual 0.0064 s at offset 0, 0.195 s at offset 2, 98.9 % of clips better at offset 0), and the trainer-side coverage of the lead head is in the table below. Treat the lead-derived fields (`lead_*`, `lon_*`) as UNVERIFIED against a known value until a control join is built.
* The 20-clip control used the 4,369-clip release (builder md5 128846cb, base 50efa52). The training release is the 4,572-clip file (md5 83e30bd3..., builder e6b2a7d4), which is not on the HF repo; the builder used here is e6b2a7d4 at cdbff971 (MEASURED, same md5 as the 4,572 manifest's builder).

## Reachability and coverage through the trainer's own loaders (g7, `reach_and_coverage.json`)
`v8f_mini_data.build_label_join` on the 10k, with the new v9 file, lead file and speed sidecar: 0 of 10,000 clips missing labels; `load_native_tracks` finds all 10,000. The first run of this check failed with `KeyError: 'lc_t_cross_s'`: the trainer reads two lane-change constraint cells that the builder emits only when an lc input exists. They are now filled with the builder's own no-crossing value (NaN, NaN, side 0; that is the value on 99.07 % of the existing release rows) by `g3d_fill.py`, and recorded in the manifest.

| share of rows with a v9 row | new 10k | existing 4,572 through the same code (4,369-clip release) |
|---|---|---|
| rows with a v9 row | 0.990 | 0.946 (203 clips have no row in the 4,369 release) |
| nav FOLLOW / TURN_L / TURN_R | 0.766 / 0.126 / 0.108 | 0.789 / 0.106 / 0.105 |
| RC-A30 valid / RC-A80 valid | 0.969 / 0.942 | 0.977 / 0.956 |
| lat_cls labelled / lon_cls labelled | 0.827 / 0.686 | 0.839 / 0.712 |
| lead_valid / lead gap finite | 0.424 / 0.390 | 0.402 / 0.384 |
| constraint cells finite | 0.410 | 0.414 |
| goal weight present on some bit | 0.929 | 0.951 |

Lead head: 9,559 clips with lead rows, 915,796 labelled rows, 268,824 with a lead within 40 m. 441 clips have no join rows (205 have no `obstacle.offline` parquet, 236 more have a parquet but no join row).

## Clock sources of the 10k (v9 `clock_src`): 9,636 sidecar, 45 pose_dt fallback, 319 nominal 0.1 s fallback. 364 clips were refused by the clock fit's own gates (K1 fit residual, K3 speed residual, fewer than 20 moving rows) and take the builder's fallback.
