# refcv8 EMA-48,852 navtest read: PREPARATION drop (banked 2026-10-21 while the full roll runs on Thor)

This drop holds the launch code, the gates (G0-G3), the refusal mutation, the 20-token smoke (VALIDATION ONLY, never quoted),
the shipping scripts and their md5 checks, and the two known-value controls scored on Thor (STOP 61.8202; refcv7-50,400 LEGAL
70.2048, byte-identical to the banked Thor CSV). **No refcv8 navtest number is in this drop.** The roll started 2026-10-21
09:57:23 Berlin under the Thor GPU flock; results land in a separate drop.
Four files were EMPTY and are omitted (the lander refuses empty files): `raw/ship_verify/{bank,cache,cbanks,logs}_md5_diff.txt`.
Each is the output of `diff` between dev-box and Thor md5 lists, so EMPTY means NO DIFFERENCE (82/82, 12,152/12,152,
12,238/12,238, 147/147 files).
Pre-registered beside this read, before any refcv8 navtest score: R8-GATE-NAVTEST (0c5b5db).
