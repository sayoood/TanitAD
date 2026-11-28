# AMENDMENT 7b to STOPPERC Amendment 7 (Master Mind, 2026-11-16, written before any C0 shard is extracted on box 2)

**What changes.** Control C0's shard-manifest clause only. Amendment 7 §4 C0 reads: *"shard manifests equal `C:/Users/Admin/s0p_state/shard_{0..3}.verify.json`"*.

**Why the literal clause cannot hold (MEASURED by the Q5C0 build agent, read-only, from box 2's four S0-P shard logs copied to `C:/Users/Admin/q5c0/ref/sh{0..3}.log`):**
* each banked `manifest_sha256` covers that shard's `meta.json`;
* `meta.json` records the shard's wall time `elapsed_s` (161.4, 156.6, 160.7, 159.4 s for shards 0-3);
* so a bit-exact re-extraction writes a different `meta.json` and a different manifest hash. The literal comparison is false even for a perfect reproduction: a control that cannot pass decides nothing.

**New C0 shard clause.** For each of the 4 S1 shards:
* (a) every data file in the shard is byte-equal to the banked one (its sha256 equals the banked manifest's entry for that file);
* (b) the new `meta.json` equals the banked `meta.json` in every key except `elapsed_s`;
* (c) the manifest rebuilt from the new data files plus the BANKED `meta.json` hashes to the banked `manifest_sha256`.
A miss on (a), (b) or (c) on any shard refuses the read. The literal comparison (new manifest == banked) is printed beside, expected false.

**Unchanged:** the S1 sample sha256 (`843e4017...`), the eval npz sha256 (`0c25fe72...fb39`; already MEASURED equal on 2026-11-15, `F:/TanitAD-scratch/q5_416/run/c0/s0p_eval.npz`), the dev-box refit of E|linear and V40+E|linear bitwise against the banked S0-P scores (it runs on the pulled S1 shards before the read, the banked scores are in the repo under `raw/2026-11-11-v8f-stopperc/s0p/`), C4, every other control, the outcomes, the samples and the estimator.

**State at writing:** no C0 shard has been extracted; no 416 probe score exists.
