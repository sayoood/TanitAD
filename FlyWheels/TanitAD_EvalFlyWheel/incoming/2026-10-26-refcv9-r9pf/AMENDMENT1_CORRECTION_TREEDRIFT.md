# Correction to PREREG_REFCV9_INMODEL_PA_AMENDMENT1 s(a) (Master Mind, 2026-10-26 ~16:34 Berlin)

AMENDMENT1 (1c25a064) attributed the 70 / 70 calibration-window path difference (max 7.24 m) to planner-path CODE that changed between
fb47a58 and the tip. **That attribution is WRONG (H-R9-TREEDRIFT-1, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-refcv9-treedrift/TREEDRIFT_REPORT.md`).** All 8 trees from fb47a58 to the
tip are bit-identical on the same 70 windows; the difference is the CUDA Philox stream POSITION (offset +4 per forward): a 70-window
roll draws different sampler noise than the same windows did inside the bank's 10,456-window roll. Injecting the bank's stream
position reproduces the banked plans bit for bit (0 / 70 differ); a +1-position control breaks it (70 / 70, 7.27 m).

AMENDMENT1's CHANGES stand: dropping K-OFF's 'equals the banked plan' clause was right for the right practical reason (a subset roll
cannot reproduce a full roll window by window); only the stated mechanism is corrected. The sealed file is not edited.
