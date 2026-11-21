# VALIDATION — tanitad-v7-training-corpus (pre-handover gate)

**Verdict: PASS** · corpus `a48251e89c7a8603` · 4,719 clips

| check | result |
|---|---|
| labels rows / exclusion violations | 4,719 / 0 |
| join labels↔camera | 4719/4719 |
| join labels↔egomotion | 4719/4719 |
| join labels↔cy (rig split {'B': 2723, 'A': 1996}) | 4719/4719 |
| join labels↔alpamayo | 4719/4719 |
| camera decode probe (first+last frame non-zero, 20±3 s) | **4719/4719** |
| camera failures (see validation_failures.json) | 0 |

Every camera file sha256-recorded in `camera_sha256.json`. Failing clips go to the
MANIFEST `exclusions` ledger with their reasons — never silently skipped downstream.
