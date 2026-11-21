# RL-S1 fan-equivalence gate on the A40 pod: the pod's own floor (MEASURED 2026-10-13 15:35 Berlin)

Stage `fancmp` of `refe/rls1_pod.sh` (shipped at `a120418`), run on the A40 pod right after P2.
- **Dumper:** `rl_fan_dump.py --compare-sets --compare-atol 0.5 --compare-mean-atol 0.025`, run on the pinned 512-scene slice (`splits/rl_fancmp_keys.tsv`, sha256 `4c613ff0…`).
- **Checkpoint:** `model_final`, step 10,075, sha256 `4246a320…`.
- **Reference:** the main run's banked step-10,075 sets on the same pod (`/workspace/data/refe_navtrain/onpolicy/sets`).

| quantity | pod (A40) | dev box (RTX 4060), `RESULT_IDENTITY.md` check B |
|---|---|---|
| sets bit-identical to the bank | **512 / 512** | 0 / 1,365 |
| max \|Δ\| (x, y, yaw) | **0.0** | 0.25 |
| per-set max \|Δ\| quantiles p50 / p90 / p99 / max | **0 / 0 / 0 / 0** | median per-set max 0.125 |
| per-set mean \|Δ\| over x, y: p50 / p90 / p99 / max | **0 / 0 / 0 / 0** | – (round 2 chose 0.025 from it) |
| dumper rate | 0.254 s/set | 0.520 s/set |

**What it says.**
- On the GPU type and batching that produced the bank, the dumper reproduces the main run's sets **exactly**.
- The batch- and hardware-dependence measured on the dev box (`RESULT_IDENTITY.md`: bf16 forward, up to 0.25 absolute) does **not** occur here.
- The pinned bounds (0.5 catastrophe, 0.025 mean; runbook D-RLS1-5) were therefore never approached. They remain the right gate for a pod whose GPU or batching differs from the bank's.
- This floor answers "does this pod's dumper reproduce the bank?" only. It says nothing about training or inference variance of the RL arms.

Files: `rl_fan_dump_fancmp_s0of1.manifest.json` (its `compare_sets` block) and `fancmp.log`, copied from `/workspace/data/rls1/` on the pod. The 19 MB set file stays on the pod.
