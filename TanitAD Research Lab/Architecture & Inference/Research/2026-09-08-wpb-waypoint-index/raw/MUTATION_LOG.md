# WP-B mutation proof — `E-WP-INDEX-1`

`torch 2.11.0+cu128` · `python 3.13.5` · **n_guards = 12** · **n_mutations = 9** · dev-box CPU (RTX 4060 box, CPU path), float32.

## Baseline — the TRUE implementation

| guard | verdict |
|---|---|
| `G1_analytic_metric_literals` | **GREEN** |
| `G2_polar_literals_from_math` | **GREEN** |
| `G3_addressed_slot_moves_with_the_plan` | **GREEN** |
| `G4_attention_ratio_is_exp_of_the_gap` | **GREEN** |
| `G5_const_control_identical_across_anchors` | **GREEN** |
| `G6_shuffle_identity_and_batch1_refusal` | **GREEN** |
| `G7_detach_severs_the_graph` | **GREEN** |
| `G8_degenerate_scene_is_finite` | **GREEN** |
| `G9_radius_gate_row_and_padding` | **GREEN** |
| `G10_bias_shape_BH_N_M` | **GREEN** |
| `G11_output_layer_zero_init` | **GREEN** |
| `G12_first_waypoint_heading_from_origin` | **GREEN** |

## Mutations — each reintroduces a REAL defect

⛔ A guard that cannot go RED proves nothing. Every row below must name at least one RED guard.

| mutation | guards that went RED |
|---|---|
| `M1_index_by_array_position` | `G12_first_waypoint_heading_from_origin`, `G1_analytic_metric_literals`, `G2_polar_literals_from_math`, `G3_addressed_slot_moves_with_the_plan`, `G5_const_control_identical_across_anchors`, `G7_detach_severs_the_graph` |
| `M2_swap_x_and_y` | `G12_first_waypoint_heading_from_origin`, `G1_analytic_metric_literals`, `G2_polar_literals_from_math`, `G3_addressed_slot_moves_with_the_plan`, `G4_attention_ratio_is_exp_of_the_gap` |
| `M3_ignore_the_detach_flag` | `G7_detach_severs_the_graph` |
| `M4_transpose_the_bias` | `G10_bias_shape_BH_N_M`, `G4_attention_ratio_is_exp_of_the_gap`, `G5_const_control_identical_across_anchors` |
| `M5_nonzero_output_init` | `G11_output_layer_zero_init` |
| `M6_drop_the_origin_prepend` | `G12_first_waypoint_heading_from_origin` |
| `M7_radius_gate_ignores_padding` | `G9_radius_gate_row_and_padding` |
| `M8_shuffle_accepts_batch_one` | `G6_shuffle_identity_and_batch1_refusal` |
| `M9_no_denominator_clamp` | `G8_degenerate_scene_is_finite` |

## Verdict

**PASS** — baseline all-GREEN and every mutation turns at least one guard RED.
