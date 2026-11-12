# TASKVEC (H-V8F-PROOF-2, P2a), MEASURED, evidence `F:/TanitAD-scratch/v8proof/wsp/TASKVEC.json`

delta_G2 = G2_s0 - NULL_s0, delta_G = G_s0 - NULL_s0, nu = NULL_rep - NULL_s0 (trainable tensors only; 252 of 276; frozen 24 excluded). Weight-space diagnostic, one replicate pair, no interval.

| group | tensors | ||dG2||/||nu|| | ||dG||/||nu|| | cos(dG,dG2) | cos(dG2,nu) | ||dG||/||dG2|| |
|---|---|---|---|---|---|---|
| adapter | 6 | 0.299 | 0.292 | 0.514 | 0.135 | 0.974 |
| cond | 12 | 0.570 | 0.454 | 0.451 | 0.150 | 0.798 |
| S | 21 | 0.605 | 0.620 | 0.636 | 0.359 | 1.025 |
| T | 21 | 0.528 | 0.512 | 0.428 | 0.276 | 0.970 |
| O | 39 | 0.518 | 0.482 | 0.496 | 0.234 | 0.931 |
| head_S | 4 | 0.642 | 0.696 | 0.471 | 0.337 | 1.085 |
| head_T | 12 | 0.483 | 0.435 | 0.453 | 0.197 | 0.901 |
| head_O | 4 | 0.440 | 0.404 | 0.526 | 0.229 | 0.917 |
| WM | 133 | 0.111 | 0.111 | 0.481 | 0.056 | 0.999 |
| T_block(T+head_T) | 33 | 0.501 | 0.466 | 0.442 | 0.229 | 0.931 |
| ALL_trainable | 252 | 0.423 | 0.416 | 0.495 | 0.211 | 0.983 |

P2a primary (T-group = T + head_T): T-block ratio 0.501; largest other group 0.642; factor 0.780 (bar 1.5) -> **DIFFUSE**.
P2a card-literal (T-group = read_t only): ratio 0.528; largest other 0.642; factor 0.822 -> **DIFFUSE**.

Top 15 tensors by ||dG2||/||nu||: h_lon.bias 3.81; h_goal.bias 1.55; read_s.q 1.09; e_lim.bias 1.00; e_nav.bias 1.00; read_t.dec.layers.0.norm2.bias 0.96; read_t.dec.layers.0.norm2.weight 0.95; e_ego.bias 0.90; e_code.bias 0.88; read_s.dec.layers.0.norm1.bias 0.88; read_t.dec.layers.0.self_attn.out_proj.bias 0.87; trunk_t.2.bias 0.84; h_cons.bias 0.82; read_t.dec.layers.0.multihead_attn.in_proj_bias 0.80; read_t.dec.layers.0.norm1.bias 0.78.

Trainable tensors with ||nu|| = 0: 0; with ||dG2|| = 0: 0. Caveat: nu is one replicate (no spread); lever arms keep seed 0, nu changes the seed.
