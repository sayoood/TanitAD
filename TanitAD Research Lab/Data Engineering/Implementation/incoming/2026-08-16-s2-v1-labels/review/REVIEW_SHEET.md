# S2 v1 label review sheet — aug120 sample (n=25 of 201)

> ⭐ **REVIEW IN `VISUAL_REVIEW.html` (same directory) INSTEAD** — the visual sheet (2026-08-16)
> renders 43 clips with camera frames + BEV + the emitted goal drawn, and carries verdict
> widgets with JSON export. This text sheet is kept as the original row dump; the visual sheet
> covers all 19 LANE_TARGET (this one shows 4) plus the ROUTE_TO disposition and the 4
> excluded val records. Renderer: `code/s2_visual_review.py`.

Every row: what the geometry measured, what the VLM said, what the label became, and what the video should show if the label is right. Labels: `labels/s2_labels_aug120.jsonl`; per-clip Engine A: `labels/engine_a_aug120.jsonl`; selection is deterministic (stratified over decision classes, edge cases by name).

Sample composition: TURN_LEFT 9, STOP_AT 4, LANE_TARGET 4, FOLLOW_MAIN_ROAD 4, TURN_RIGHT 3, NONE_ABSTAIN 1.

## 1. `8dc5d14d-ab19-43a4-843d-12fdd28173a8`

* scene: day, snow, urban 2-lane; ego 6.8 m/s steady/turning_left; no agents
* Engine A (hindsight): route `turn_left` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-5.27 m/s
* VLM said: goal `turn_left` · actions reduce_to(none), prepare_lane_change(left), hold_corridor(left)
* **label: g_str `TURN_LEFT` (—) · a_str `REDUCE_TO` (arg0=1.49, within_m=7.0)**
* check: video should show a left turn starting ~? m ahead

## 2. `079707d3-c6ac-4b41-9b61-1b98b298ecc9`

* scene: day, snow, urban 1-lane; ego 5.5 m/s steady/turning_left; no agents
* Engine A (hindsight): route `u_turn` valid=True dyaw=2.985 rad dist=0.0 m · stops=False net_dv=3.88 m/s
* VLM said: goal `route_to` · actions prepare_exit(left), reduce_to(none)
* **label: g_str `TURN_LEFT` (arg0=0.0) · a_str `HOLD_CORRIDOR` (at_arc_m=81.0)** · **ROUTE_TO remapped** (G1 gated)
* check: video should show a left turn starting ~0.0 m ahead

## 3. `01b24287-0026-4e07-89a5-8767cd2c1a7b`

* scene: day, snow, urban 3-lane; ego 0.7 m/s braking/straight; 3 bus, 36 car, 7 pedestrian, 35 traffic light, 15 traffic sign, 3 truck
* Engine A (hindsight): route `turn_left` valid=True dyaw=1.68 rad dist=27.3 m · stops=False net_dv=7.71 m/s
* VLM said: goal `turn_left` · actions prepare_lane_change(left), reduce_to(left)
* **label: g_str `TURN_LEFT` (arg0=27.322) · a_str `HOLD_CORRIDOR` (at_arc_m=69.4)**
* check: video should show a left turn starting ~27.322 m ahead

## 4. `01bee851-1dc5-4162-9428-ef35670065ce`

* scene: night, clear, urban 2-lane; ego 5.7 m/s braking/straight; 15 car, 2 traffic light, 3 traffic sign, 1 truck
* Engine A (hindsight): route `turn_left` valid=True dyaw=1.374 rad dist=2.9 m · stops=False net_dv=-4.14 m/s
* VLM said: goal `turn_left` · actions prepare_lane_change(left), reduce_to(left)
* **label: g_str `TURN_LEFT` (arg0=2.859) · a_str `REDUCE_TO` (arg0=1.6, within_m=5.7)**
* check: video should show a left turn starting ~2.859 m ahead

## 5. `2d6a15f2-1865-49f2-ab19-a1b8a1953983`

* scene: day, rain, urban 1-lane; ego 8.0 m/s steady/turning_right; no agents
* Engine A (hindsight): route `turn_left` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=1.04 m/s
* VLM said: goal `follow_main_road` · actions prepare_lane_change(left)
* **label: g_str `TURN_LEFT` (—) · a_str `HOLD_CORRIDOR` (at_arc_m=98.2)**
* check: video should show a left turn starting ~? m ahead

## 6. `559b4123-1e3c-4d57-9c08-79df011ae4b3`

* scene: day, clear, urban 1-lane; ego 5.0 m/s accelerating/straight; no agents
* Engine A (hindsight): route `turn_left` valid=True dyaw=0.266 rad dist=34.2 m · stops=False net_dv=-3.64 m/s
* VLM said: goal `follow_main_road` · actions reduce_to(none), hold_corridor(none)
* **label: g_str `TURN_LEFT` (arg0=34.238) · a_str `REDUCE_TO` (arg0=1.08, within_m=11.1)**
* check: video should show a left turn starting ~34.238 m ahead

## 7. `5c8bdfdc-2fa2-44b7-829f-46d3e28f8b41`

* scene: night, snow, urban 2-lane; ego 4.2 m/s steady/turning_right; no agents
* Engine A (hindsight): route `turn_right` valid=True dyaw=-1.065 rad dist=0.0 m · stops=False net_dv=5.72 m/s
* VLM said: goal `follow_main_road` · actions prepare_lane_change(right), hold_corridor(right)
* **label: g_str `TURN_RIGHT` (arg0=0.0) · a_str `HOLD_CORRIDOR` (at_arc_m=99.5)**
* check: video should show a right turn starting ~0.0 m ahead

## 8. `0089a096-68be-40df-8097-780bf1ae1c19`

* scene: night, clear, urban 2-lane; ego 6.0 m/s accelerating/turning_left; no agents
* Engine A (hindsight): route `turn_left` valid=True dyaw=-0.275 rad dist=50.0 m · stops=False net_dv=0.06 m/s
* VLM said: goal `route_to` · actions prepare_lane_change(left)
* **label: g_str `TURN_LEFT` (arg0=49.997) · a_str `HOLD_CORRIDOR` (at_arc_m=67.9)** · **ROUTE_TO remapped** (G1 gated)
* check: video should show a left turn starting ~49.997 m ahead

## 9. `14bc3af3-7764-4ab0-96d1-06b304feea33`

* scene: night, clear, highway 2-lane; ego 6.3 m/s braking/turning_right; no agents
* Engine A (hindsight): route `turn_left` valid=True dyaw=1.62 rad dist=4.8 m · stops=False net_dv=10.19 m/s
* VLM said: goal `route_to` · actions prepare_lane_change(left)
* **label: g_str `TURN_LEFT` (arg0=4.776) · a_str `HOLD_CORRIDOR` (at_arc_m=111.9)** · **ROUTE_TO remapped** (G1 gated)
* check: video should show a left turn starting ~4.776 m ahead

## 10. `00d05901-ed0a-4a43-adca-bdab70d30bfa`

* scene: night, clear, urban 2-lane; ego 4.1 m/s braking/turning_right; no agents
* Engine A (hindsight): route `turn_right` valid=True dyaw=-1.099 rad dist=0.0 m · stops=False net_dv=4.56 m/s
* VLM said: goal `route_to` · actions prepare_lane_change(right)
* **label: g_str `TURN_RIGHT` (arg0=0.0) · a_str `HOLD_CORRIDOR` (at_arc_m=70.3)** · **ROUTE_TO remapped** (G1 gated)
* check: video should show a right turn starting ~0.0 m ahead

## 11. `08fc0af2-ffeb-4a7c-9166-8f052f1a9b94`

* scene: night, clear, urban 2-lane; ego 0.7 m/s steady/straight; 3 car, 2 traffic light, 6 traffic sign
* Engine A (hindsight): route `turn_right` valid=True dyaw=-1.463 rad dist=0.6 m · stops=False net_dv=14.11 m/s
* VLM said: goal `route_to` · actions prepare_stop(none), resume_cruise(none), hold_corridor(none)
* **label: g_str `TURN_RIGHT` (arg0=0.62) · a_str `HOLD_CORRIDOR` (at_arc_m=92.1)** · **ROUTE_TO remapped** (G1 gated)
* check: video should show a right turn starting ~0.62 m ahead

## 12. `f0209b60-4dd6-4a26-85a7-645804e5365f`

* scene: day, rain, highway 2-lane; ego 20.2 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-2.03 m/s
* VLM said: goal `route_to` · actions hold_corridor(none)
* **label: g_str `NONE_ABSTAIN` (—) · a_str `HOLD_CORRIDOR` (at_arc_m=227.7)** · reason: vlm route_to unverifiable (G1 closed, evidence sign unchecked kind) and no junction geometry to remap to
* check: abstain: verify neither a turn nor a plain follow fits

## 13. `0330480a-9c2c-44e1-bb2a-f0f03f13eb95`

* scene: day, snow, urban 1-lane; ego 5.5 m/s braking/straight; 1 bus, 22 car, 5 pedestrian, 9 traffic sign, 3 truck
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=True net_dv=-5.38 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `STOP_AT` (arg0=20.002) · a_str `PREPARE_STOP` (within_m=0.0)**
* check: ego should come to a stop ~20.002 m ahead (red light / queue / sign)

## 14. `21dd9ee8-67ed-4865-b2b0-a8bcd83b31e3`

* scene: night, clear, urban 1-lane; ego 10.2 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=True net_dv=-10.23 m/s
* VLM said: goal `follow_main_road` · actions reduce_to(none), hold_corridor(none)
* **label: g_str `STOP_AT` (arg0=42.469) · a_str `PREPARE_STOP` (within_m=20.4)**
* check: ego should come to a stop ~42.469 m ahead (red light / queue / sign)

## 15. `12bb97af-e850-4ecb-86a9-6e6d0e517ae9`

* scene: day, snow, urban 3-lane; ego 5.3 m/s braking/straight; no agents
* Engine A (hindsight): route `unknown` valid=False dyaw=0.0 rad dist=None m · stops=True net_dv=-5.33 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `STOP_AT` (arg0=9.441) · a_str `PREPARE_STOP` (within_m=0.0)**
* check: ego should come to a stop ~9.441 m ahead (red light / queue / sign)

## 16. `15a65b76-a2f0-4413-bde6-695052f45b9d`

* scene: day, clear, highway 3-lane; ego 5.3 m/s steady/straight; no agents
* Engine A (hindsight): route `unknown` valid=False dyaw=0.0 rad dist=None m · stops=True net_dv=-5.1 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `STOP_AT` (arg0=16.586) · a_str `PREPARE_STOP` (within_m=36.3)**
* check: ego should come to a stop ~16.586 m ahead (red light / queue / sign)

## 17. `03ba450b-121a-483b-aa48-6d9097c308de`

* scene: day, snow, rural 1-lane; ego 13.7 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=0.17 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `LANE_TARGET` (arg0=1.0, arg1=0.0) · a_str `PREPARE_LANE_CHANGE` (arg0=1.0, within_m=0.0)**
* check: ⚠️ VLM did NOT flag this lane change (0/19 corroborated) — check a real ~3.63 m lateral move vs a curving road

## 18. `24b6948f-4206-4ef1-85ec-ff344b400635`

* scene: day, rain, highway 2-lane; ego 18.6 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-0.65 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `LANE_TARGET` (arg0=1.0, arg1=0.0) · a_str `PREPARE_LANE_CHANGE` (arg0=1.0, within_m=0.0)**
* check: ⚠️ VLM did NOT flag this lane change (0/19 corroborated) — check a real ~7.08 m lateral move vs a curving road

## 19. `2bc52cd7-0117-4f5c-a7d9-c301a70827f2`

* scene: day, clear, rural 2-lane; ego 16.5 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-0.07 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `LANE_TARGET` (arg0=1.0, arg1=0.0) · a_str `PREPARE_LANE_CHANGE` (arg0=1.0, within_m=0.0)**
* check: ⚠️ VLM did NOT flag this lane change (0/19 corroborated) — check a real ~4.67 m lateral move vs a curving road

## 20. `31308c82-c53a-427a-9201-8520832b51eb`

* scene: day, rain, highway 2-lane; ego 20.5 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-0.24 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `LANE_TARGET` (arg0=-1.0, arg1=20.6) · a_str `PREPARE_LANE_CHANGE` (arg0=-1.0, within_m=20.6)**
* check: ⚠️ VLM did NOT flag this lane change (0/19 corroborated) — check a real ~-4.45 m lateral move vs a curving road

## 21. `01fbc807-3183-4118-ab5a-3e38224d3108`

* scene: day, clear, highway 1-lane; ego 29.3 m/s steady/straight; 1 car, 8 traffic sign
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-10.82 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `FOLLOW_MAIN_ROAD` (—) · a_str `REDUCE_TO` (arg0=18.44, within_m=0.0)**
* check: ego should simply follow the corridor

## 22. `029ae2b0-f063-4c0d-83f3-8d8781bd69a3`

* scene: night, clear, urban 1-lane; ego 9.5 m/s steady/straight; no agents
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-5.49 m/s
* VLM said: goal `follow_main_road` · actions reduce_to(none), prepare_stop(none)
* **label: g_str `FOLLOW_MAIN_ROAD` (—) · a_str `REDUCE_TO` (arg0=0.9, within_m=0.0)**
* check: ego should simply follow the corridor

## 23. `0cd622e0-5a5f-4c12-b1d6-77af7c37e585`

* scene: day, clear, rural 1-lane; ego 20.5 m/s steady/straight; no agents
* Engine A (hindsight): route `unknown` valid=False dyaw=0.0 rad dist=None m · stops=False net_dv=-2.01 m/s
* VLM said: goal `follow_main_road` · actions hold_corridor(none)
* **label: g_str `FOLLOW_MAIN_ROAD` (—) · a_str `HOLD_CORRIDOR` (at_arc_m=229.6)**
* check: ego should simply follow the corridor

## 24. `15ac5a34-91ed-4c12-8a9f-eaee12b0b588`

* scene: night, clear, urban 2-lane; ego 0.0 m/s stopped/straight; no agents
* Engine A (hindsight): route `turn_left` valid=True dyaw=1.619 rad dist=8.0 m · stops=True net_dv=5.49 m/s
* VLM said: goal `turn_left` · actions prepare_lane_change(left)
* **label: g_str `TURN_LEFT` (arg0=8.036) · a_str `RESUME_CRUISE` (arg0=5.49)**
* check: video should show a left turn starting ~8.036 m ahead

## 25. `13141fac-c31b-4123-a3d6-68367304e1e1`

* scene: day, clear, urban 1-lane; ego 8.6 m/s steady/straight; 1 bus, 10 car, 3 pedestrian, 22 traffic sign, 1 truck
* Engine A (hindsight): route `follow` valid=True dyaw=0.0 rad dist=None m · stops=False net_dv=-6.12 m/s
* VLM said: goal `follow_main_road` · actions reduce_to(none), hold_corridor(none)
* **label: g_str `FOLLOW_MAIN_ROAD` (—) · a_str `REDUCE_TO` (arg0=0.32, within_m=0.0)**
* check: ego should simply follow the corridor

---

## The 4 excluded w120val records (item 5)

Triple-empty (VLM/SAM3/Alpamayo absent, ego_state null); their fused `NONE_ABSTAIN` was a default-of-absence. Excluded from the label set with reasons — Engine A geometry is banked for all 4 and shown here (what a re-fuse would recover):

* `1d4dcb4e-5117-4e84-9eac-59690879c7d6` — geometry route: `follow` — triple-empty record: VLM/SAM3/Alpamayo layers all absent, ego_state null — its f…
* `a26a627a-caf4-4f23-a02c-9a4e558fc867` — geometry route: `follow` — triple-empty record: VLM/SAM3/Alpamayo layers all absent, ego_state null — its f…
* `b02c28ce-e2c7-4f37-86f6-9888d519fe43` — geometry route: `follow` — triple-empty record: VLM/SAM3/Alpamayo layers all absent, ego_state null — its f…
* `b0388541-b7de-465d-8411-998cf5881bee` — geometry route: `follow` — triple-empty record: VLM/SAM3/Alpamayo layers all absent, ego_state null — its f…
