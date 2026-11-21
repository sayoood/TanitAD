# S1 per-arm gate gap: attribution and two levers (report)

Written by the gate-gap agent (Sonnet) on 2026-10-15. The agent's harness refused the file write, so the Master Mind banked it from the hand-back.
Full figures: `RESULTS.json`. **Status: NOT APPLIED.** The patches wait for the S1 G12 fix, so both land in one S1 re-ship.

## Attribution of N-s0's 24.8 min GPU-idle gap (GPU_CLEAR 19:10:38Z to TRAIN_START 19:35:25Z)

| share | what | evidence |
|---|---|---|
| **73 %** | the same tree + data re-measure, run **five** times: mint, 3 x `token_ok`, manifest | MEASURED split, from pod timestamps |
| 23 % | G-SUITE | MEASURED |
| 4 % | the GPU wait | MEASURED |

Inside one re-measure, the dir listing of `navtrain_pixels` (**413,184 files**, `dir-listing-v1`: relpath + size of every file) is at least 97 % of the time.
That share is INFERRED: the pod logged no per-phase times.

## Levers

| lever | change | effect | evidence |
|---|---|---|---|
| **B** | `s1aux_pod.sh`: the `:403` verify runs only on the RESUMED path. The post-mint verify, the BINDING verify before the trainer, the manifest row and the PREREG order are unchanged | **3.4-3.8 min per arm** | MEASURED per-verify cost |
| **A** | threaded stat listing, string ops, `readinto` hashing (`launch_gate.py`, `gate_checks.py`, `gate_token.py`); **digests bit-identical**, with a frozen pre-patch copy as the oracle | dev box: listing **10x**, `data_manifest` **9.4x** | MEASURED on the dev box |
| A + B | | **4.9-17.5 min per arm**, depending on how many metadata requests the pod volume serves at once | ESTIMATED; the pod speedup is UNMEASURED |

## Suites (patched scratch copy; the counts on other chains are unchanged)

| suite | base | patched |
|---|---|---|
| `test_s1aux_chain.py` | 361 | **383** |
| `test_launch_gate.py --static` | 637 | **658** |
| `test_launch_gate.py --static --no-banked` | 623 | 644 |

- No skips anywhere.
- The other chains' suites read identical outcomes on base and patched: p2r 139, p2rep 350, p6 341, rl_s1b 18, aux_s1_wiring 20, rl_reward 31, rl_epo 42.
- `test_rls1_chain.py` fails identically on both. The cause is the scratch layout (no `MODEL_REGISTRY.md` above the package), so it is not a regression.

**Mutations: all 15 go RED in the intended suite.** These include:
- deleting the binding pre-trainer verify, which the chain test catches as `TRAIN_NO_TOKEN_CHECK`;
- digest-altering hashing changes, which the identity test catches.

See `mutation_results.json`.

## To land (Master Mind)

1. Apply the six `PATCH_*.diff` files to `refe/`, together with the G12 fix.
2. Re-run the suites.
3. `launch_gate.py ship-manifest`, then re-ship.
4. Run `bench_fingerprint.py --token <PASS json>` on the pod to MEASURE lever A there.

**Not done** (`RESULTS.json`): running G-SUITE beside the preflight, and prefetching the next arm's gate during the current arm's training. Each needs a
design or a PI decision, because it adds load beside a training arm.
