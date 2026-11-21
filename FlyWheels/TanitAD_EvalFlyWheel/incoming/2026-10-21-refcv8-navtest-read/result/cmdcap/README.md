# G_CMDCAP on navtest (R8-GATE-NAVTEST AMENDMENT 1, registered 9cd2347 before any navtest score): WORKS

R8PATH_MINPROG (refcv8's R8_LEGAL path driven at min(R8, prior) progress) built on the navtest seams by
`build_lever_seams_navtest.py` (K1-K6 controls pass: it reproduces the banked LEGAL / RT / PRIOR seams), scored on Thor
(8 shard jobs, 0 failed), then G_CMDCAP = PRIOR on straight commands, R8PATH_MINPROG on turns. Result in
`R8_GATE_NAVTEST_3gates.json`. Only the two MINPROG seams are banked here; the other 16 builder arms (not registered for navtest)
were built locally and never scored. Per-token scorer rows are OFF-BANK (`OFFBANK.sha256`; dev box
`C:/Users/Admin/refcv8_pf/navtest_read/raw/cmdcap/scores_cmdcap/`, Thor `/home/nvidia/r8ns/navtest_read/full/scores_cmdcap/`).
