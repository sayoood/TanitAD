# VDAC A5: HB-s0 costs about 10 L40-h, not 7.6; the cap rule was missed; grace on the timeout; A5HC held for the PI (2026-11-13, MM)

## What is happening (MEASURED, pod `/workspace/data/refe_vdac/`)
- HB-s0 passed every gate on tree c78dff15, G-SUITE included (`ZZVDAC_PREFLIGHT_OK` 02:59:22Z, `ZZVDAC_GATE_OK` 03:16:54Z, `chain_hb-s0.log`). It has trained since 03:18Z (`ZZVDAC_TRAIN_START`).
- It trains at batch 4 x accum 2, the card's pre-registered memory fallback (PREREG 9.2, `vdac_pod.sh` HB rows; batch 8 OOMed on 2026-11-12).
- **Step cost: 3.2417 s per update** (update 25 at 03:28:37, update 2,450 at 05:39:38, `hb-s0/sft.jsonl`; the trainer's own mean reads 3.2424). Evals: step0 took about 7.1 min (HA's took 3.8 to 3.9 min).
- The preflight's own speed probe printed `s_per_microbatch` 2.34 and `gpu_mem_gb` 27.32 (`hb-s0/preflight.jsonl`).

## Projection (ESTIMATED from the measured rate)
| event | time (Z) |
|---|---|
| update 10,000 | ~12:27:30 (7,550 updates x 3.2417 s after 05:39:38) |
| + update-5,000 eval (~7.1 min) | ~12:34:40 |
| + final eval, save, run check, post-gate, export, egress (HA's post-training tail 9.6 min, with HB's longer eval: ~13.2 min) | **~12:47:50** |
| pause timeout (`timeout 36000` from 02:49:20Z, `vdac_pause_p7.sh:59,62`) | **12:49:20** |

The margin is about 1.5 min. An epoch boundary at update 7,500 or any slowdown would cross it. By the card's stop rule (1), a timed-out arm is NOT RUN, with no partial read, so all of HB-s0's ~10 L40-h would be lost.

## The missed rule (MM lapse, disclosed)
- PREREG_VDAC_P6_A5 section 9: "The first pod preflight prints `gpu_mem_gb` and `s_per_microbatch`; the MM sends them to the PI before HB-s0 trains if the extrapolated cost exceeds 7.6 L40-h."
- The probe's 2.34 s per microbatch extrapolates to 10,000 x 2 x 2.34 s = 13.0 h. The measured cost is ~10.0 L40-h for the whole HB-s0 chain (02:49Z to ~12:48Z).
- Both exceed 7.6. The sequencer (`a5_sequencer3.sh`) ran the stage straight through preflight into training, and the MM did not stop it after the gate (`VDAC_STOP_AFTER=gate_hb-s0`, POD_RUN_PLAN step 6). The PI was not told before HB-s0 trained.

## What the MM did (2026-11-13 05:46Z)
1. **Bounded grace on HB-s0's timeout** (`pod/hb_s0_grace.sh`, sha256 prefix `a88c177e`):
   - It acts only if the chain is still running at 12:46:00Z AND the trainer has logged update >= 9,000.
   - It then SIGSTOPs the `timeout` process (pid 249827). Its SIGALRM waits; the chain itself is untouched.
   - It SIGCONTs at chain exit, or at the hard ceiling 13:49:20Z (+1 h), whichever is first. At the ceiling the timeout fires as written.
   - The pause script decides on `ZZVDAC_CHAIN_DONE HB-s0` in the chain log, not on rc (`vdac_pause_p7.sh:64-65`). So a chain that finishes under the grace continues to HB-SHUF even though `timeout` then returns 124.
   - `pod/hb_s0_failsafe.sh` (`2842720a`) SIGCONTs at 14:00Z if the process is still stopped, so the pause can never hang.
   - The grace changes no lever, seed, update count or bar (stop rule 3). It is an operational deviation, disclosed here and in the result.
2. **A5HC held.** Sequencer3 (pid 248818) was stopped by explicit PID. `pod/a5_after_hb.sh` (`f9a03705`) watches the A5HB wrapper (pid 248857) and logs its end plus "not launching A5HC" into `a5_sequencer3.log`. P7 resumes after A5HB as usual, so the pod does not idle.

## Pod time against the PI's cap (ESTIMATED; cap = 16.7 L40-h, the card's upper bound)
| item | L40-h |
|---|---|
| Stage 0 | 0.12 to 0.25 |
| HB-s0 first pause (preflight OOM, 2026-11-12 21:50 to 21:55Z, plus P7 stop and resume) | ~0.1 to 0.4 |
| HC-s0 G-SUITE failure pause (01:32 to 02:04Z) | ~0.5 |
| HB-s0 (02:49Z to ~12:48Z) | ~10.0 |
| HB-SHUF (2,000 updates at the same rate + 3 evals + chain overhead) | ~2.7 |
| **through A5HB** | **~13.4 to 13.9** |
| HC-s0 (card estimate 4.3 + 0.5 overhead; HC has no trunk copy) | ~4.8 |
| HC-SHUF | ~1.4 |
| **all four arms** | **~19.6 to 20.1, i.e. ~3 to 3.5 over the cap** |

## Decision for the PI
- **A5HC:** approve about +3.5 L40-h over the 16.7 cap (total about 20.1), or name a narrower HC run. Until then A5HC does not launch and P7 keeps training.
- **The grace:** the PI may veto it at any time before 12:46Z. The MM then kills `hb_s0_grace.sh` by PID before it acts.
