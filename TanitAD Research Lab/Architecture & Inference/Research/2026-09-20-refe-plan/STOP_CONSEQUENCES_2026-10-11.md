# What the weekly-limit stop and the pod stop cost the REFe stream (2026-10-11)

> Written 2026-10-11 07:45 UTC (09:45 Berlin) from `HANDOFF_2026-10-11.md` and the files it names, plus fresh probes.
> Every claim carries its evidence class. Times are UTC unless labelled Berlin (UTC+2).

## The two stops

| event | when | evidence |
|---|---|---|
| Weekly API limit: the orchestrating session stopped | after the last handoff state, 2026-10-05 ~09:05 | INHERITED (`HANDOFF_2026-10-11.md`) |
| The A40 pod is unreachable (ssh rc 255, TCP connect timeout) | probed 2026-10-11 ~07:26; **the stop time is unknown** | MEASURED (probe this session) |

The pod-side chains ran without a session, so the API stop alone did not stop SFT-4 or P2. What decides their fate is
**when the pod stopped**, which nothing on our side records. ⇒ Everything below about "did it finish" is conditional.

## State of each run at the last known point

### SFT-4 (scorer-only, PDM targets + lane output; `eval/PREREG_SFT4.md`)
- **At 09:05 Oct 5:** update ~1,000 of 2,335, at ~14.4 s per update (INHERITED, handoff). It trained on the
  **146,185** PDM labels written at arming, 84 % of 174,912 (Amendment 4, MEASURED at arming and landed).
- **Held-out read at update 800** (PDM truth, 3,137 sets / 24 logs): arm B lane rule **+2.27 [+0.88, +3.62]**,
  arm A v1 **+2.27 [+1.10, +3.33]**. INHERITED: the only source is the handoff. ⚠️ **No SFT-4 `sft.log` or
  `chain.log` was ever banked** (the repo holds only `raw/2026-10-04-sft4-smoke/` and
  `raw/2026-10-04-sft4-stage2-identity/`), so these two numbers cannot be re-derived.
- **If the pod lived:** it would have finished at ~14:25 UTC Oct 5 (ESTIMATED: 1,335 updates × 14.4 s ≈ 5.3 h, plus
  three held-out reads).
- **The registered stage-1 read (at the final update) was never looked at.** Even if it exists, it is on the pod.

### P2 (joint fine-tune, scorer un-detached, PDM targets; `eval/PREREG_P2.md`)
- **At 09:04 Oct 5:** the launch gate (G1–G4 + preflight) was running; the training chain was armed behind SFT-4
  (INHERITED, handoff).
- **If the pod lived and the gate passed:** ~1,209 steps at ~60 s ≈ 20 h, i.e. ~14:30 Oct 5 → ~10:30 Oct 6
  (ESTIMATED from the prereg's own rate).
- No gate result, no P2 log and no P2 checkpoint ever left the pod (MEASURED: none in the repo, on the dev box or on HF).

### The PDM relabel (pod CPU)
- ~28,727 training sets remained at 05:06 Oct 5 at ~258 / min ⇒ complete ~07:00–07:30 Oct 5 (ESTIMATED).
  Irrelevant to recovery: the rows themselves lived only on the pod.

## What survives, and where

| where | what | evidence |
|---|---|---|
| **HF** `Sayood/tanitad-refe-vitl16-navtrain10-grow-final` (model, private) | `model_final`, `ckpt_last`, snapshots 001–027, config, metrics, logs, `CHECKPOINTS_MD5` — 74 files, 4,578,323,291 B, uploaded 2026-10-05 00:30 | MEASURED (HF API listing this session) |
| **HF** `Sayood/tanitad-refe-navtrain-banks` (dataset, private) | the training banks: `onpolicy_queue`, `onpolicy_sets_logs_full`, `refe_grow_test`, `refe_heldout_v5`, `refe_navtrain_banks_rest`, `refe_train_grow`, `workspace_scripts_logs` — 15 files, 8,095,637,319 B, 00:16 | MEASURED (HF API listing) |
| **Dev box** `E:/Projects/TanitAD-artifacts/refe-*` | the final run (`model_final` md5 `b54773f8…`), the on-policy bundle, the pod wind-down bundle | MEASURED (`ls` + md5) |
| **Dev box** `E:/Projects/TanitAD/data/refe_runs_eval/` | `model_final`, `lane7_identity`, snapshots 001–020, `sft3/model_sft3_A.pt`, and **`sft4_peek/`** | MEASURED |
| ⭐ **Dev box** `…/sft4_peek/model_sft4_B_u400.pt` | **the only SFT-4 checkpoint that survives anywhere off the pod**: arm B at update 400, 1,288,489,838 B, md5 `8e382eeb…`, plus `scorer_states_u400.pt` (md5 `4dded9b1…`) | MEASURED |
| **Repo** | every result doc and prereg; SFT-1 and SFT-3 raw logs; the SFT-4 smoke and identity gates; the exploratory u400 peek (`raw/2026-10-05-sft-stage2/sft4_peek_u400_B/`, +4.24 [+1.36, +7.39] on W3's 200) | MEASURED (tip `6949265a`) |

**The pod itself:** UNKNOWN. The handoff records **no persistent network volume**. Per RunPod's documentation
(INHERITED, not re-verified for this pod), a *Stopped* pod keeps its volume disk and a *Terminated* one does not. That
single fact decides whether the PDM labels, SFT-4's final checkpoints and P2's gate survive. ⇒ **Cheapest recovery
step, a PI action:** check in the RunPod console whether the pod is Stopped or Terminated. If Stopped and restartable,
copy off `/workspace/data/refe_sft2/{train_pdm,heldout_pdm}`, `/workspace/data/refe_sft4/` and
`/workspace/data/refe_p2/` **before** anything else runs on it. (No RunPod API key is held on the dev box, so this
cannot be checked from here.)

## What is lost unless the pod's disk survives

| item | redo cost | evidence for the cost |
|---|---|---|
| PDM labels: `train_pdm` (177,836 rows) and `heldout_pdm` | ~11.3 h CPU for 174,912 sets at ~258 / min — but that rate needs the navtrain DBs **and** the pod's sensor blobs | ESTIMATED |
| Lane relabel rows (≥ 2,109) | small; subset only | INHERITED (Amendment 4) |
| SFT-4 beyond update 400: both final checkpoints, the u800 checkpoints, `sft.log`, `chain.log` | ~9.3 h GPU (2,335 × 14.4 s), after the labels | ESTIMATED |
| P2: gate, run, checkpoints | gate + ~20 h GPU | ESTIMATED (prereg rate) |
| SFT-1's `scorer_states_last.pt` (u2,400) | none needed: SFT-1 is closed NOT PASSED | MEASURED (`RESULT_SFT1.md`) |
| navtrain DBs and sensor pixels | public, re-fetchable; size 164 GB (handoff) vs 449.0 GB (memory) — INHERITED both, to be measured on re-fetch | INHERITED |

## What has to be redone, in order

1. **PI:** check Stopped vs Terminated. If the disk survives, copy the three directories above off it first; most of the
   list below then disappears.
2. **PI spend:** provision a pod (GPU for SFT-4/P2; PI decision (c) asks for a CPU pod for labelling).
3. Re-fetch navtrain; restore the banks from HF (pod-side download — the dev box's 1.2 MB/s uplink rules out relaying
   8 GB through it).
4. Re-run the PDM relabel (~11.3 h CPU).
5. Re-run SFT-4 from step 0 on the **complete** labels (~9.3 h). ⚠️ This is a NEW arm, not a resume: the lost run saw
   84 % of the labels (Amendment 4). Its prereg must say so.
6. P2: gate, then ~20 h.

## The lesson (the root-cause class, for RETRACTION_LOG / TRAPS)

**Pod-only outputs with no automatic egress.** The final run's model was pushed to HF the moment it finished; the
fine-tunes that followed were not, and neither were their logs. A chain that writes its result only to an ephemeral
pod has produced nothing until something copies it off. ⇒ **Every pod chain ends with an HF push of its checkpoints
and logs, and its success marker is the remote listing, not the local file.** Same family as "an artifact on one disk
is not done" (CLAUDE.md operating standard 3).

## The 0-pod lever, run now (Rule Zero)

The one SFT-4 artifact that survives can still be read on the registered navtest stage-2 protocol, on the dev box, in
~30 min: **arm B at update 400 on SPEC_NAVTEST Amendment 5's 923 confirmation tokens**. It is a deviation and is
registered as such before the read (`eval/PREREG_SFT4.md`, Amendment 5): it reads a non-final checkpoint of one arm, and
the claim it can support is about **that checkpoint**, not the SFT-4 recipe. Result: `eval/RESULT_SFT4.md`.
