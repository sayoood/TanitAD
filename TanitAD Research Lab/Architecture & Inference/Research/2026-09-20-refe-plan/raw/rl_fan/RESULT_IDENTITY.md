# RL-S1 frozen-fan identity on the dev-box GPU (MEASURED 2026-10-12, RTX 4060)

Instrument: `eval/rl_fan_identity.py` (run 19:59-20:14 Berlin), output `identity_devbox.json`. Follow-up probe: `old_dumper_batch_probe.py`, output
`old_batch_probe.json`. Consumer proof: `eval/rl_fan_labeller_proof.py`, output `labeller_proof_dumped.json`.

## Verdict against the committed bars (`HANDOFF_IDENTITY.md`)

| check | committed bar | result |
|---|---|---|
| **A**: `rl_fan_dump.py` (model_final) vs the old `onpolicy_dump.py` (ckpt_last@10075), same GPU, 500 keys | EXACT | **PASS**: 500/500 sets bit-identical, 1,920,000 values, max delta 0.0 |
| **C**: invariance to batch size 1 / 2 / 4, shifted batch grouping, `--workers 2`, subset vs full run | EXACT | **FAIL**: batch 1 and 2 vs 4 differ on 24/24 sets; shifted grouping on 3/23; workers 2 and subset-vs-full are exact. Max abs delta 0.25 |
| **B**: dev box vs the A40's banked step-10,075 sets, 1,365 keys | measured, magnitude reported | 0/1,365 bit-identical; 3,195,745 of 5,241,600 values differ; **max abs delta 0.25**, median per-set max 0.125. The old dumper shows the same (B2: 500 keys, max 0.25) |

The instrument printed `ZZFANID_A_OR_C_NOT_EXACT`. C failed its bar, and that is reported as written.

## Diagnosis: the batch dependence belongs to the trainer's forward pass, not to the new dumper

`old_batch_probe.json`, same GPU, the same 24 keys:

| comparison | bit-identical | values different | max abs delta |
|---|---|---|---|
| old dumper, batch 1 vs batch 4 | 0 / 24 | **59,156** | 0.25 |
| new dumper, batch 1 vs batch 4 (C above) | 0 / 24 | **59,156** | 0.25 |
| new vs old, both at batch 1 | **24 / 24** | 0 | 0.0 |
| new vs old, both at batch 4 | **24 / 24** | 0 | 0.0 |

The original pipeline is batch-variant by exactly the same 59,156 values, and the new dumper reproduces it bit for bit at every batch size tried.
So C's bar cannot be met by the dumper on this hardware without changing the forward pass's numerics, which would break A, the identity
with the pipeline that produced the main run's sets. The two committed bars are incompatible here; A is the one that answers the identity
question.

Deltas are absolute, not relative: in bf16 ulps of the value they reach 508.9 (near-zero coordinates make ulp ratios explode, and the
stored values are not bf16-exact), so a ULP tolerance is the wrong unit for any gate on these files.

## Consequence for RL-S1

- The chain (`refe/rls1_pod.sh` stage 6) dumps at one fixed batching and labels and trains on its own fans. It never compares fans
  across batchings or machines, so the batch dependence cannot make its data inconsistent: every label is computed on the fan it belongs to.
- The pod gate sketched in `eval/HANDOFF_RL_S1.md` section 6 (a slice of banked keys, `--compare-sets`, bit-identical or exit 5) would
  REFUSE a correct dumper: a different batch grouping flips some sets by a bf16 step even on the same GPU. It is not in the chain,
  runbook or prereg. A sound replacement needs an absolute tolerance, justified by the floors above (cross-GPU max 0.25) and by
  deliberate-regression arms that must exceed it.
- Consumer proof on genuinely dumped records: `ZZLABPROOF_PASS`. 6 sets went through `onpolicy_relabel_pdm.py` and `rl_terms_label.py`
  unmodified, then through `train.OnPolicyBank`, `scorer_finetune.load_rl_terms` and the RL-val loaders; 6/6 everywhere, 0 bad.
- Dev-box rate: 0.520 s/set total, 0.420 s/set forward, peak GPU 2.27 GB (RTX 4060, torch 2.7.1+cu128).
