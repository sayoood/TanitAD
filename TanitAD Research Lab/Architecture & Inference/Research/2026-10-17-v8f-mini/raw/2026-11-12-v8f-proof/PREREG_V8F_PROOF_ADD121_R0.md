# PREREG_V8F_PROOF_ADD121_R0 — note to card ADD121 (REPL10K): control r0, the two new models

Written before the queue of the card is started and before any rollout of a new model exists. The card (e87dc5b05f6848b5) and its notes N1 and N2 are unchanged.

## 1. The models

Both models are stage A of the 10k line, trained by the other session on its pod and delivered to the fork pod under `/workspace/v8proof/integ10k/ckpt/<model>/A/` with their arm record and a copy of their gate token.

| model | `ckpt_last.pt` sha256 | arm record sha256 (first 16) | gate token copy sha256 (first 16) | argv sha256 (first 16) |
|---|---|---|---|---|
| `UE_s2` | 6d4345258e07dee15b1c5171268182967189df1cdc13b89b81d26aae71c3f327 | 537a9a9e91a57bc8 | 9dc5796e4a0fb71f | fc1da77086a2ccd4 |
| `UE_s3` | 8f5eb364ac374e258b423bf90b06844628d44595678d83cc9906f3b2b3c04b39 | 78c9ec367c125ffb | cccaf368787a35c7 | 19d8e873f8a15fec |

## 2. What was verified on the fork pod (MEASURED here, `repl/models/r0_check.py`)

* The sha256 of each checkpoint on the fork pod equals the one its arm record names.
* Both arm records name the code commit cdbff971fa6dad4d20b937fe8abda6cb9f622adb, checkpoint step 61,568 and a passed self-check of declared against built values.
* Each gate token copy lists 18 checks, all passed.
* The argument lists of the two records have 35 entries each and differ from each other in the seed and the output path only.

## 3. What is taken from the other session (INHERITED, its messages of 2026-11-23, not re-verified here)

* The argument list of each model is that of `UE_s0` with the seed and the output path as the only differences (2 of 35 entries), enforced by its checker when the tokens were minted. The arm records of `UE_s0` and `UE_s1` are not on the fork pod, so this comparison could not be repeated here; the comparison of the two new models with each other (section 2) is in line with it.
* The data manifests are those of `UE_s0` and `UE_s1` (the same tree, trainer and data).
* The checkpoints are equal on its pod, on the dev box and in its private store.

## 4. Verdict of control r0

Both models are admitted: checkpoint hashes equal their records, a launch-gate pass is recorded for each, and the recipe is that of `UE_s0` and `UE_s1` with the seed as the only difference (the part of section 3 on the word of the session that trained them). The record for the reader is `/workspace/v8proof/repl/models/REPL_MODELS_RECORD.json`; it carries the hash of this note.
