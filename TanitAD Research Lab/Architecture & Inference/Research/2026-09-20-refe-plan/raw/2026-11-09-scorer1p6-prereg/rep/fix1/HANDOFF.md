# rep fix1 (2026-11-10, Master Mind): test_rep_gate.py failed G-SUITE on the pod

**What happened (MEASURED, pod 2026-11-10).** The first replicate pause stopped P7 at its 04:50:45Z checkpoint. C0r2 passed precheck, inputs, G-EGR, dd and its preflight (04:55:49Z), then the launch gate refused it at 04:58:28Z: `G-SUITE FAIL test_rep_gate.py: rc 1, 'ZZTEST_PASS' NOT seen`. No token was minted and nothing trained. S1E1 would have hit the same gate, so the MM stopped its chain by PID (121816 121783 121567 121566) during its preflight. P7 resumed at 05:02:43Z (pid 122104), about 12 min of P7 lost.

**Cause.** The test's "no calibrate file" case ran the pod script with `REP_WS=/workspace`. Off-pod that path does not exist, so the case held. On the pod, `/workspace` holds S1E's real calibrate file, so S1E1's argv carried `--pair-w 0.10810585693683655` while `rep_argv(arm)` was called without a lambda: `AssertionError: ('S1E1', {('--pair-w', ('0.10810585693683655',))})`. A test that depends on the machine it runs on: it passed where the data was absent and failed where the gate runs.

**Fix.** That case now uses an empty temp workspace (the same pattern as the next block in the file), mapping the temp path back to `/workspace` before comparing. No pod script, launch_gate.py, profile or trainer changed.

**Proof (MEASURED on the pod, a throwaway copy of /workspace/refe-plan-rep, CPU, P7 training):** original file rc 1 with the same AssertionError; fixed file rc 0, `ZZTEST_PASS`, R3 digests unmoved (refe_rep_p6 dd348f44..., refe_rep_s1z_p6 2356afd4...). The full G-SUITE was not run while P7 trained; the next pause runs it inside the gate.

Also changed: `rep/post_rep.py` takes the pause start (T0) as its first argument.
