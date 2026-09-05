# Master Mind decisions, 2026-09-05

## M1. The pod's stale `train_v6_staged.py` ships at the NEXT v7 launch preflight — never mid-run

From the pod-currency audit (`…/2026-09-04-pod-currency-audit/POD_CURRENCY_AUDIT.md` §2 rank 2,
commit `94ceaf4`): `tanitad-refcv3:/workspace/TanitAD/stack/scripts/train_v6_staged.py` is
**113 KB (~19 %) behind HEAD** and lacks the `--bptt-truncate` wiring. A v7 launch from that box
would run the O5 rollout with the unbounded BPTT chain — the MEASURED `PREREG_MM_E19` failure
(gnorm 2.1e9, run killed at step 9,000) — and would fail *silently*, because the box is
stale-but-internally-consistent. The escalation was correct: this file must not be shipped while
`refcv4b` trains (a supervisor relaunch imports whatever is on disk), and it must not be shipped
in isolation (its compatibility with a future v7 config cannot be verified from here).

⇒ **Decision:** the v7 launch runbook gains a mandatory step: run
`stack/scripts/pod_currency_audit.py` against the target pod, ship `train_v6_staged.py` and
`tanitad/models/metric_dynamics.py` as part of that preflight, verify by md5 at both ends and by a
real `import` on the pod, and grep-verify `--bptt-truncate` is present before any launch. Owner:
whoever launches the next v7 arm. Not before refcv4b finishes.

## M2. `metric_dynamics.py` on the pod stays as it is until M1

Imported but never called by the live trainer; not a pure addition (38 pod-only lines);
default-equivalence only INHERITED from a docstring and a test not run against this pod. Zero
benefit to the live run, non-zero relaunch risk. Ships with M1.

## M3. The ego-dropout burden is NOT a live-run change

`D-REFCV4B-EGODROP2` (commit `ef0d5ec`): real, longitudinal, shrinking on its own. The fix
(`H-EGODROP-PRED`, roll the withheld bank at the model's own predicted speed) is pre-registered
for refcv5 on the v7-tiny ladder with a shuffle-the-prediction control. The live run is read
again on its final checkpoint.

## M4. Never run two `kb_add` writers concurrently

`library.json` was found at 0 bytes after the API-limit deaths — a rewrite interrupted mid-flight.
Restored from HEAD and re-verified (`7d1f48f`). Every literature brief now carries the rule, and a
0-byte `library.json` is to be read as the signature of an interrupted writer, never as an empty
library.
