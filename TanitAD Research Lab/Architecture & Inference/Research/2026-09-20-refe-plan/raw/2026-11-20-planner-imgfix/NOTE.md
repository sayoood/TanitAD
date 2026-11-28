# HOLD planner_imgfix (MM 2026-11-19) - land BETWEEN campaigns only

What: refe/planner.py REFePlanner._image_for: one retry on a frame-load failure, and a NAMED miss reason
(frames.miss_reason = "frame load failed after 2 attempts (<member>): <ExcType>: <msg>"; imread None also named).
Why: P5-C read attempt 1 (2026-11-19) lost 1 of 12,146 tokens to a transient load failure under low commit;
the bare except left the miss reason EMPTY. The retry alone would have saved that 5 h pass.

Base: mirror tip d46f37f2, planner.py blob 6fb6598a2ecf230153a4d7adfb790afc30728070
sha256 old 91437cd09362ff872a861c6166b3c00eee98ce9b98c4e85b6a95ca133e29e211
sha256 new d4b2fe9afa0179df6c9dd971fd3bfd21037322d0b2fe5fb0c04568fb37d4568b  (LF, utf-8; built by imgfix_patch.py)

Test: land/<PK>/refe/test_planner_image_retry.py (new). MEASURED 2026-11-19 under driverl-eval venv:
ZZTEST_PASS 21, mutations "no retry" and "unnamed miss" RED; the UNPATCHED tip planner goes RED at T1
("got None ('')" = the empty reason seen in the read).

Re-pin (31 files hold the old sha256, 17 the blob/md5): ONLY the live operational ones move:
- PK/SHIP_MANIFEST.json, PK/refe/p7_ship_list.json (use its writer if one exists), PK/refe/test_ego_yaw.py (md5 f8a01699...)
- TF chain templates n4/chain/N4_PINS.template.json, n5_l1/chain/N5, n6_tol/chain/N6, n7/chain/N7 (only if no TF chain is mid-run; ask the owner)
- every profile fingerprint / count pin the p7_ship_list change moves (memory pin-edit-moves-profile-fingerprints): run all ~69 launch_gate-reading tests serially.
- NOT the banked raw/ artifacts (historical; they record what ran).

Land conditions: no read, gate or chain is bound to the old pin at land time. The P5-C read runs from the fixed export
F:/TanitAD_p5cread_tip (unaffected); the pod field chain runs package code_10848403 (unaffected). Check TF/N-chains.
