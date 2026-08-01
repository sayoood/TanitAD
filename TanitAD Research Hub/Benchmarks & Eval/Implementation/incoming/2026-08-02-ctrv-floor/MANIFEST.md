# MANIFEST — 2026-08-02-ctrv-floor (Benchmarks & Eval)

Every artifact and where it lives (AGENT_OPERATING_STANDARD rule 2).

## In the repo (staged on `agent/benchmarks-eval-20260802`)

| path | what |
|---|---|
| `PREREGISTRATION.md` | pre-registered before the driver ran; both outcomes + INSTRUMENT-FAIL branch |
| `ctrv_floor.py` | the floor + window enumeration + `verify_alignment` precondition gate |
| `tests/test_ctrv_floor.py` | 11 analytic-ground-truth sanity tests (G-B2) — `pytest tests` standalone |
| `run_ctrv_readjudication.py` | the 27-arm sweep driver (CPU-only, no GPU, no checkpoint) |
| `verify_patched_block.py` | applies the patch to a throwaway copy and validates the block end-to-end |
| `proposed_ctrv_floor.patch` | the two-file change; `git apply --check` clean against `4978a82` |
| `INTAKE.md` | triage package: what / why / evidence / target / risk / rollback / escalations |
| `raw/ctrv_readjudication.json` | 1.3 MB — every arm, floor, stratum, paired interval and verdict flip |
| `raw/ctrv_run.log` | the run log (27 arms, 372.9 s) |
| `../../Research/2026-08-02-ctrv-floor-readjudication.md` | the research note |

## On the eval pod (`tanitad-eval`, `/workspace/`) — reproducible, not sole copies

| path | what |
|---|---|
| `/workspace/ctrv_floor.py`, `/workspace/run_ctrv_readjudication.py` | copies of the above |
| `/workspace/verify_patched_block.py`, `/workspace/proposed_ctrv_floor.patch` | copies of the above |
| `/workspace/ctrv_readjudication.json`, `/workspace/ctrv_run.log` | **banked into `raw/` above** |
| `/workspace/_ctrv_patchtest/` | throwaway patched copy of taniteval — **deleted after validation** |

⛔ **Nothing is stranded.** Both raw artifacts are in the repo; the pod copies are conveniences.
The pod's own `taniteval` checkout was **not modified** — the patch was applied only to a throwaway
copy under `/workspace/_ctrv_patchtest/`, which was removed.

## Reproduce

```
cd /workspace && OMP_NUM_THREADS=6 \
  PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval \
  python3 run_ctrv_readjudication.py --out /workspace/ctrv_readjudication.json
```
