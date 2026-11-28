# Verified-copy cleanup on the A40 pod, 2026-10-14

**PI ruling 2026-10-14:** "Delete verified copies". A pod-local file is deleted ONLY when the HF remote listing proves an identical copy of it:
the same file name, the same size, AND the same LFS sha256. The sha256 is re-computed on the pod at delete time.

* Tool: `verified_cleanup.py` (dry run by default; `--delete` acts). One JSON line per decision. The token is read on the pod and never printed.
* Run 1 (`cleanup_dry.log`): dry run. 30 files would go, 9.069 GB, 0 kept.
* Run 2 (`cleanup_delete.log`, `cleanup_20261014.jsonl`): **30 files DELETED, 9.069 GB**, each with its proof (`repo`, `path`, `sha256`).
* Scope:
  * `rls1/arms/R0/` (killed arm, `Sayood/tanitad-refe-rls1-r0`);
  * `refe_runs/p2_joint_pdm/` (P2, `Sayood/tanitad-refe-p2-joint-pdm`);
  * the downloaded `hf_dl/Sayood__tanitad-refe-navtrain-banks/` (dataset `Sayood/tanitad-refe-navtrain-banks`);
  * the downloaded `hf_dl/Sayood__tanitad-refe-vitl16-navtrain10-grow-final/` (`Sayood/tanitad-refe-vitl16-navtrain10-grow-final`).
* Deliberately KEPT:
  * `rls1/arms/C0/` (RL-S1b's control and REFDIR);
  * `hf_dl/queue` and `hf_dl/heldout` (S1 re-verifies them on every start);
  * `refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt` (the INIT of P2-s1 and RL-S1).

**One edit after the runs (2026-10-14 ~21:05 Berlin):** the runs read the HF token from the pod's token file. The banked tool, and the copy now in
`/workspace/ops/`, read it from `HF_TOKEN` in the environment instead. The repo's secret scan refuses a token-file path. Nothing else changed.
