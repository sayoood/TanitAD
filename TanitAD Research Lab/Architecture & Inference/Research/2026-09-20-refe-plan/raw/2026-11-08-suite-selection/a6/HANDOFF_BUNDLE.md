# HANDOFF BUNDLE-A6 (2026-11-08)
State: chain launched 10:31 Berlin, immune to Ctrl-C (python launcher `a6/launch_detached_a6.py`, CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW, started via WMI pid 4136). After ~3 min: past prepare (27 tokens, 24 logs, 755 members, none missing), a0 forward running, extract running (32 archives), no ^C in log.
PIDs (at check): chain cmd 6780; a0 forward cmd 40012 / python 16896; extract python 18572. 
Log: F:/TanitAD_tsf_bundle2_20261107/work_build_a6_chain.log ; a0 out: F:/TanitAD_tsf_bundle2_20261107/a0_forward_a6(.out)
Sentinels: success = `CHAIN_DONE`; failure = `CHAIN_FAIL assemble` (new). Also a0_forward_a6/DONE.
Outputs: F:/TanitAD_tsf_bundle2_20261107/bundle_a6, work_a6. Expected 20-40 min total. If it dies, relaunch is resumable: `python a6/launch_detached_a6.py` (extract skips existing members).
Files: a6/build_bundle_a6.py (path-only copy; role label 'a3' kept as-is), a6/chain_a6.cmd, a6/sky_prescreen.py (copied from a3, needed by `sky` via HERE). a3/ untouched. Note MAP_FIELDS_A6.json is NOT consumed by the builder; apply it downstream.
Removable after bundle_a6 verifies: F:/TanitAD_tsf_bundle2_20261107/bundle_a3, work_a3 (not deleted by me).
Scratch: none created.
