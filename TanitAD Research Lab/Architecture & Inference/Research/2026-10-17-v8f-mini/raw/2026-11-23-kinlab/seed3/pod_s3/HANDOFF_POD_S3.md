# HANDOFF_POD_S3 (2026-11-23): SEED3 pod kit FINISHED, shipped, prep all green. Nothing launched; no lock, no marker taken.

**Launch (MM only, after the PI go + cost):** `pssh 'bash /root/rollhist/kit_s3/launch_s3.sh'` under timeout. Log `/root/rollhist/rh_s3_chain.log`, last line `ZZ ROLLHIST-S3 DONE ZZ` or `FAIL <why>`. The chain logs `LOCK-START epoch=`: return line 4.0 pod-h = 14,400 s after it.

**Facts** (`FACTS_POD_S3.md`): B0_s2 / RH_s2 train at **97fe0a56** on `tree_cdb` (Amendment 1, 0d1d9226); KI_s2 at 4097a89e on `tree_ki`. UE_s2.A sha 6d434525 = card. Pod free disk 46 GB (bank gate 30 GB, arms 20).

**Tests** (`TESTS_S3.md`): argv 34, gate-record 59, bank-check 12, chain DRY test (walk + 14 mutants refused), marker watcher (run alone: PASS, no hang). **Mutants 59/59 RED** (first run 45/59: CRLF-blind runner + redundant-check survivors; fixed in tests, no kit source changed).
`U_s2.B.tmpl.json` differs from U_s1 only at --seed, --out, --dagger-seed (same 3 lines as U_s0 to U_s1). Rendered B0_s2 / RH_s2 / KI_s2 differ from s0 only by the card's substitutions (+ `--dagger-base logged` for KI).

**Ship:** `/root/rollhist/kit_s3/` (17 files), `sha256sum -c SHA256_S3.txt` 16/16 OK; both shared pin files match the pod (17 files). Blob checks: tree_cdb 3408/3408 bad 0, tree_ki 3425/3425 bad 0.

**B0_s2 dry mint (CPU, 25 s selfcheck):** `declared_vs_built` PASS, `live` pass, all probes true; pair check PASS; gate-record mint rc 0, verify rc 0; study-side dry mint rc 0 (TOKEN-WRITTEN). Evidence `dry_evidence/`, pod `kit_s3/dry/`.
**Dry walk:** `DRY=1 RH_READOUT=causal_g1x bash kit_s3/rh_s3_chain.sh` ends `ZZ ROLLHIST-S3 DRY PASS ZZ`, wrote nothing.

**Projected lock (ESTIMATED from MEASURED walls):** bank 6,843 s (1 h 54 min) + 3 arms x ~0.26 pod-h = about 2.7 pod-h; return line 4.0 pod-h.
Design notes: B0_s2 trains before the bank (pair check uses `runs/B0_s2/pair_rh_skeleton.json`); zerograd is report-only as in s0/s1 (chainlib post is the fatal gate). Real-data `live` for RH_s2 / KI_s2 is only exercised after the bank exists.
