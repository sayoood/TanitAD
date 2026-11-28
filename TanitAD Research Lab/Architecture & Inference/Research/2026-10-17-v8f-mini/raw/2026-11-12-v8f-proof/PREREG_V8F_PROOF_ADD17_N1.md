# PREREG_V8F_PROOF_ADD17_N1: tolerances of the controls x1, x2, x4 and the window matching (2026-11-14)

Written after the first start of `spatial/q2/ctx_roll.py` on G_s0 REFUSED at the window matching, before any rollout, control value or outcome number of the read existed.

Found (MEASURED, the file itself): the latents file of the pixel read (`LATENTS_GW_<ARM>.npz`) stores its arrays as float16; the per-frame state file (`latents/<ARM>.pt`) is float32.
Exact equality of the frame-0 state, as section 2 of the card writes it, is therefore not available, and the tolerances 1e-4 / 1e-5 / 1e-3 of x1, x2, x4 are below the rounding of the banked file.

Changed, and nothing else:
* Window matching: the nearest row of the state file, max abs difference at or below 4e-3 (the float16 rounding at the magnitude of the states), and every other row at or above 0.1; a window without such a unique row refuses the read. Rows that match the same window must hold identical sequences (difference at or below 1e-6).
* x1, x2, x4: tolerance 4e-3 for every comparison against an array of the banked file. x3 (float32 against float32) keeps 1e-5.
* The rollouts and the lead read use the float32 states of the state file, not the float16 copies.

Outcomes, rollouts, start frame, read tool and committed next steps of the card are unchanged.
