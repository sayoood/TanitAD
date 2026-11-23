# HANDOFF_READ_S3
State: code built and tested; nothing run on s2 (no checkpoint). Files: CHANGED_FILES_READ_S3.txt, FACTS_READ_S3.md.
Next (after ARM_DONE of B0_s2, RH_s2, KI_s2 on the pod):
1. Write out/read_s3/LEVER_ACTS_S3.json (keys in s3_name.LEVER_KEYS, all true) from the pod gate/zerograd/post/bank/dagger_probe records.
2. Launch via WMI (CreateFlags 0x08000200, stdin NUL): rh_read_s3_retry.cmd. ~2 h CPU for 18 rollouts + T0 evals. Resumable (out/steps_s3). Log out/rh_read_s3_chain.log; end line "ZZ SEED3-READ DONE ZZ".
3. Result: out/read_s3/RESULT_SEED3.json (name + every input). If LEVER_ACTS was written late: rm out/steps_s3/name.ok and re-run the chain.
Dry walk anytime: RH_S3_DRY=1 bash rh_read_s3_chain.sh. Never touched: seed3/pod_s3, the frozen readers.
