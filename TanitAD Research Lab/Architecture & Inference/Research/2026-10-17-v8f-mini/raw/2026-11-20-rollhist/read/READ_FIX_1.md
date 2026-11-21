# READ FIX 1 (MM, 2026-11-21T05:35:05Z): the checkpoint pull reached the wrong pod

* The provisional seed-0 read (RH_SEEDS=0, launched 05:27:23Z) passed blobcheck and the reader selftests (ALL_PASS). It then stopped at `pull_RH_s0` with "no pod sha256 for RH_s0".
* Cause: `rh_read_chain.sh` pulled through `C:/Users/Admin/v8pod/pssh.sh` / `pcat.sh`, which read `v8pod/pod_direct.txt` (the v8proof pod). The RH/B0 checkpoints are on the LoRA pod (`lorapod/pod_direct.txt`). The two address files differ (compared by hash only).
* Fix: new helpers `C:/Users/Admin/lorapod/pssh_cmd.sh` and `C:/Users/Admin/lorapod/pcat.sh`, with the same interface, sourcing `podenv.sh`. Line 24 of `rh_read_chain.sh` points at them.
  * sha256 a5dc9763 → a2a4e24f; the copy is kept as `rh_read_chain.sh.bv8pod`.
  * Check: through the new helper, the pod returns the RH_s0 sha256 351bb5f26ed0e156..., equal to the pod's own `sha256sum`.
* No value had been read. The readers, cells, bars and steps are unchanged. The pull still checks every checkpoint against the pod's own sha256.
* `DRY_CHECK.json` / `MANIFEST_SHA256.txt` (04:18Z) record the old sha of this one script. Refresh both at the next dry check.
