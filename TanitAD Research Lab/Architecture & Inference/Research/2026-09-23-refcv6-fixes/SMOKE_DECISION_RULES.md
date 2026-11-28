# refcv6 launch -- smoke decision rules, fixed 2026-09-23 ~09:15 BEFORE any smoke number existed

Smokes (Thor, commit 4ca61688, the launch line with EVAL=1 and the train+eval joins, 40 steps,
log every step): S1 = batch 4, conflict probe every step; S2 = batch 4, every 10th step (same batch,
so the probe's cost is isolated); S3 = batch 8, every 10th step (memory and throughput scaling).

* Conflict probe cadence: every step if it adds <= 25 % to the step time (S1 vs S2); otherwise every
  10th step. The detector stays ON either way (SPEC R2) and config.json stamps the cadence.
* Batch: the largest of {4, 8, 12, 16} whose measured peak `cuda_max_mem_gb` (the only admissible
  device-memory probe on Thor) is <= 60 % of MemAvailable at launch; among those that fit, the
  highest samples/s. Batch sizes not smoked are extrapolated from S1/S3 only if linear in B, and
  the extrapolation is stated.
* Steps: ceil(805,680 / B) rounded up to a multiple of 50 -- the budget is fixed in SAMPLES (the
  prereg's full = 40,284 x 20), so the batch decision cannot move it.
* s/step is read from metrics.jsonl `elapsed_s` between the 10th and the last logged step (the first
  steps carry warm-up); an eval inside that span is stated.
