# MM-E19 — the ONE-COMMAND read (k=60 horizon arm at 30,000)

**Prereg (governs, verbatim):** `Project Steering/PREREG_MM_E19_K60_HORIZON.md`
**Prepared:** 2026-09-01, Benchmarks & Evals FlyWheel. Dry-run proven end-to-end
against the interim step-17,500 checkpoint (`raw/interim_dryrun_step17500.json`
— INTERIM-DRYRUN, NEVER the read; latentmotion at n=12 for the pipeline proof,
actdiv at its banked n=24, CPU because the dev-box GPU sat at 100 % util).
⛔ **Read its `spike_arrival` block before quoting any regime bound: the full
18.6k log REFUTES the prereg's "bounded ~4,000–14,000" — window 16,000–18,000
is the worst of the run (8 spikes, rate 4.0/1000, max 1.24e10 at step 18,000).**

⛔ **Division of labour, per the brief:** this pipeline PREPARES THE NUMBERS.
Every criterion verdict field in the emitted JSON is EMPTY. The ORCHESTRATOR
applies the pre-registered criteria (§3 table, §3b both-bands rule, §3c
attribution gate, §4 anti-gates) and writes the verdicts. Nothing here says
PASS/FAIL.

## Step 1 — pull the final checkpoint (read-only ssh; trainer must be DONE)

```bash
# snapshot on Thor first so scp never reads a file mid-write:
ssh -n tanitad-thor-wifi 'cp /home/nvidia/v7tiny/k60clip05p30k/ckpt.pt /tmp/mm_e19_ckpt_pull.pt && \
  cp /home/nvidia/v7tiny/k60clip05p30k/config.json /tmp/mm_e19_config_pull.json && \
  cp /home/nvidia/v7tiny/k60clip05p30k/train_log.jsonl /tmp/mm_e19_trainlog_pull.jsonl && \
  md5sum /tmp/mm_e19_*'
scp tanitad-thor-wifi:/tmp/mm_e19_ckpt_pull.pt      <pull>/ckpt.pt
scp tanitad-thor-wifi:/tmp/mm_e19_config_pull.json  <pull>/config.json
scp tanitad-thor-wifi:/tmp/mm_e19_trainlog_pull.jsonl <pull>/train_log.jsonl
# verify the three local md5s against the Thor-side ones before proceeding.
```

## Step 2 — THE ONE COMMAND

Run from a LOCAL copy of the probes dir (G: cannot RUN code; the proven run dir
is `C:\Users\Admin\tanitad-wt\mm-e19-probes\`):

```bash
OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
C:/Users/Admin/venvs/tanitad/Scripts/python.exe mm_e19_read.py \
  --ckpt <pull>/ckpt.pt --config <pull>/config.json \
  --train-log <pull>/train_log.jsonl \
  --device cuda        # ⛔ cuda ONLY if nvidia-smi shows <20% util; else cpu
```

Defaults already point at the verified assets
(`C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901` — md5-verified copies of
the E-DEC-59 corpus, the incumbent `postrain30k` ckpt, and the first-24 clips of
Thor's actdiv val corpus) and the verified stack
(`C:\Users\Admin\tanitad-wt\stack` — see `_stackresolve.py` header for the
Thor-parity evidence). It emits ONE JSON: `mm_e19_out/mm_e19_read_step30000.json`
containing

* **actdiv** h1 `ratio_action_over_scene` for `k60clip05p30k` AND `postrain30k`
  (same device, same 24 clips, same instrument) — the PRIMARY ≥10× input;
  banked Thor incumbent reference 0.005947 is embedded;
* **latentmotion** ego-marginal on BOTH bands `0:8` and `8:16` at BOTH probe
  horizons k=4 and k=60, both arms — §3b's admissibility requirement;
* **norms** — the §3 secondary read (`‖to_scale_shift‖`, `‖act_emb.*‖`) for both
  ckpts, with the prereg's incumbent reference values as a known-value control,
  plus the ONE-VARIABLE args diff (expect `o5_k`, `clip`, paths);
* **spike_arrival** — the §3c instrument recomputed over the FULL 30k log
  (per-2,000-step windows, exact spike steps); quote THESE regime bounds,
  not the interim ~4,000–14,000;
* **caveat_MUST_TRAVEL** and **criteria_FOR_ORCHESTRATOR_verdicts_EMPTY**.

## Step 3 — bank

Copy the merged JSON into this directory's `raw/` and stage it. The interim
dry-run artifact is already here as `raw/interim_dryrun_step18600.json`.

## What the orchestrator must still bring (not producible by this pipeline)

1. ⛔ **HORIZON-WORKS needs the clip-0.5 k=8 control** (prereg §3c): this arm is
   `o5_k 60 + clip 0.5` — two changes from the incumbent. The control (~10 h at
   1.21 s/step) has NOT been run; a ≥10× result without it is HORIZON-PARTIAL
   attribution at best.
2. The T0 prediction-quality comparison respecting the §4 anti-gates (o5_loss
   is NOT comparable across k).
3. The seed-band caveat (§5): single seed; sub-band magnitudes are unresolved,
   not null.

## Instrument provenance

* `latentmotion.py` — E-DEC-59, MM-E19-revised: `--k` parameterised (default 4
  reproduces the banked invocation), MM-C12 stack preflight + refusal, tree
  stamped. **K=4 reproduction, MEASURED 2026-09-01 (CPU, n=80, same corpus +
  ckpt bytes as banked, `raw/repro_k4_rdw8p30k_cpu.json`):** constant control
  +0.0000/t 0.00 EXACT; ego r +0.0073 EXACT (t 2.75 vs 2.78); drift r 0.6714
  vs 0.6718; joint 0.6709 vs 0.6712; marginal −0.0005 (t −0.39) vs −0.0006
  (t −0.48); n/steps/rows exact. Residual ≤5e-4 deltas are device-shaped: the
  banked run was GPU and the GPU was occupied (100 % util) for this rerun.
  ⚠️ **Byte-identity on the SAME device is therefore UNPROVEN — settle it when
  the GPU frees:**

  ```bash
  OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe latentmotion.py \
    --arms rdw8p30k --assets C:/Users/Admin/tanitad-caches/mm-e19-assets-20260901 \
    --stack C:/Users/Admin/tanitad-wt/stack --device cuda \
    --out repro_k4_rdw8p30k_gpu.json     # expect the banked table verbatim
  ```
* ⚠️ **Prereg §3 attribution flag (secondary read):** the quoted incumbent
  norms 2.97/3.06/5.44 (act_emb2 9.5694→9.4015) are **o1ctrl30k's**, per
  MM-E18's own artifact (`raw/msg_e18.txt`) — VERIFIED by reproducing that row
  to all four decimals from `o1ctrl30k/ckpt_step30000.pt` (md5 `eeb12bd11269…`)
  with the runner's extractor (Frobenius of `.weight`). The ACTUAL incumbent
  `postrain30k` (md5 `a58585883c27…`) reads to_scale_shift
  **3.8301/4.3603/4.2527**, act_emb.2.weight **9.6288**. The read JSON carries
  both; the orchestrator decides which baseline §3 binds to.
* `actdiv_local.py` — dev-box port of `actdiv_thor.py` (md5 `371d4c13…` on Thor
  AND in the session bank — now also banked beside this file); compute section
  verbatim. The interim arm ckpt is preserved at
  `<assets>/v7tiny_k60clip05p30k/ckpt_step17500_INTERIM.pt` (the runner reads
  only `ckpt.pt`, so this cannot pollute a census) — with the 30k final that
  gives the act_emb decline direction two points at read time.
* `mm_e19_read.py` — the runner; `test_latentmotion_params.py` — 10/10 tests
  (K reaches the computation; refusal fires; spike parser reads known values).
