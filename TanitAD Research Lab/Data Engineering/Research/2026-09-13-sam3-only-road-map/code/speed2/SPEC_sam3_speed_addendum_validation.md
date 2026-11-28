# SPEC addendum 3 — validation clips for the fp16 measures (pre-registered 2026-09-14, before any validation-clip result)

PI, 2026-09-14: *"check the fp16 on a third clip"*, then *"Combined the proven measures and give me a rendered long video to confirm
quality. After this we will start the production for the whole training corpus"*.

Bars: UNCHANGED from `SPEC_sam3_speed_prereg.md` (N1 map agreement ≥ 0.990 · N2 IoU ≥ 0.95 per class with ≥ 500 cells ·
N3 PI-check deltas · N4 raster agreement ≥ 0.990), each arm against the EXACT reference arm on the same clip.

Clips (none processed by any SAM3 map arm before this addendum):
* 2 LiDAR clips in the approved format (native f-theta frames, LiDAR ground): 6924358fafe0, 0d90d20036a3 — all four bars; PI checks at frame 48.
* 8 production-format clips (native f-theta front frames, ground from the ego path, no LiDAR): 1f1f05ca011d, b5d9b91e6637, b975bf8ebf95, 41f10d46174e,
  26015e788849, f63e215a546a, c1dc66b6ae42, d672fc17a315 — N1, N2, N4; N3 is NOT COMPUTABLE (its curb reference needs LiDAR) and is reported as such.

Arms (streaming driver, every clip): `spdSA` exact reference (measures 1–4, 7–8; equal to today's pipeline by their approval) ·
`spdS4` = `spdF4a` flags · `spdS5` = `spdF5a` flags.

Committed in advance:
* `spdS5` passes every computable bar on all 10 clips → the fp16 image backbone is **promoted to the production default**
  (it passed the two test clips already) and the long video shows `spdS5`.
* `spdS5` fails any bar on any clip → `spdF4a` flags stay the production default and the video shows `spdS4`.
* `spdS4` fails any bar on any clip → the production configuration is NOT confirmed; report the clip and the bar before production.
