# HANDOFF_RHK (build agent, 2026-11-21): RHK = RH argv + a bank generated under readout K; STAGED, nothing run on the pod, nothing committed

State: the build is complete. Nothing is staged in git, nothing was written on the pod (only `pssh` reads: `rh_bank.sh`, tree greps, sha256sums, logs).

Key fact: readout K is `R0KHb20` = `--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2`, trainer token **`causal_g0x`** (read/rh_common.py:29, `readout_token`). RHK argv = RH argv with only `--dagger-bank` (bank_k/sK), `--dagger-bank-sha256`, `--dagger-bank-readout causal_g0x`, `--out`; dose 0.1 literal.

Next steps (MM):
1. Read and freeze `DRAFT_AMENDMENT_8.md` (write its sha256 beside it). Cost ESTIMATED about 4.3 pod-h (banks 1.9 h x 2, arms 0.27 h x 2): needs the PI's second go with that cost.
2. Install pod files per `DRY_MINT_PLAN.md` A0, check `POD_FILES_SHA256.txt`, run phases A1 to A3. Phase B is either the chain's own mint at the arm step or `RHK_BANK_ONLY=1`.
3. `bash /root/rollhist/launch_k.sh`; after both RHK arms show ARM_DONE run `read/rh_read_k_chain.cmd` (WMI launch).

Not proved: the K bank itself (none exists; K-B5 under gain 0 first runs at the bank step); G-DAGGER on the real K bank; C-BANK / C-RH-BANKGEN on a real bank. The tests use SYNTHETIC K banks and a synthetic K sha over the real RH/B0 argv.

Design notes: (i) stock `rollhist_pair_check.py` hard-codes `causal_g1x`: `rhk_pair.py` and the checker swap that one token in a copy (stock file untouched). (ii) `rhk_gate_record.py` = pod `rh_gate_record.py` + 3 anchored edits (`patch_rhk_gate.py`): bank_k paths, PILOT_RH.json is the RH chain's code-level pilot (no K pilot exists; the full-bank check `rhk_bank_check.py` is K's own), K token only. (iii) `RHK_BANK_ONLY=1` in the chain builds banks only.

Edited in `read/` (each has a `.bk9` copy): rh_common.py, rh_roll.py, rh_read.py, dry_check.py (patch scripts `patch_read_k.py`, `patch_dry_k.py` here). New: rh_read_k_chain.sh/.cmd. Default and RHp3 reads proved unchanged by dry_check (rh_common vs .bk9 and .bp03).

To rerun the checks: `mkdir build/b9rhk/tmp`, then `bash run_checks.sh; bash run_dry.sh` (one python at a time, commit gate inside).

Deletions (own temp, by literal path, re-creatable): see `deletions_k.log` (b9rhk/tmp 78,237 KiB test scratch, 7 `__pycache__` dirs about 1 MiB). Nothing of the pod, `stack/` or `read/out/` was deleted.
