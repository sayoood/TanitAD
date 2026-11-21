# RESULT, PREREG_REWARD_2 (M-GYM-6, 2026-11-09, Master Mind)

Tier: simulator validity only. No model was evaluated. Seed 20261111, 160 states reached (below the 240 cap), 2 ladder build failures counted and listed (28, 80). Raw: `raw/reward2/results_v6.json`.

## Verdict as written
**B3 FAILED.** 41 eligible, 35 KNOWN. Not KNOWN: 51, 61, 70, 107, 121, 150. Four of the six are at the 4.47 m/s limit (51, 61, 107, 121), where a slow rung is a large share of the limit. In 70 and 150 the best rung is below 1.0x.

**B3-F, read per Amendment 1 (pre-simulation free-road class): FAILED.** 18 eligible, 16 KNOWN, not KNOWN 61 and 121. The card's branch: "B3-F has a state not KNOWN: the reward still pays a slow rung on free road. Next lever: the P1 pace term as its own freeze, after re-scoring the PDM, B1 and B4 arms on stored records and writing its 0.9x and 1.0x tolerance in advance." `B3_F_rollout_any_rung` is 9 of 9 and is labelled: it never decides a branch.

## Retained checks
B1 PASS. B2 PASS (45 of 45 real flagged by both; distinct real-collision states 12 and 17). B4: known except `always_fastest_log` on one cell (fail count 34 against PDM 35). B5 GO. B6 UNDETERMINED (no all-arms off-drivable state). Determinism bit-identical.

## Families
LONGITUDINAL only as reward terms and hard events. LATERAL only off-drivable and wrong-way events. TACTICAL n/a. STRATEGIC PAUSED (PI 2026-09-19, ITEM 25), not a pass.

## Departures, stated
`presim_v6.py` and `tests/test_v6.py` were re-pinned after the draw and before the first rollout: import path and test harness only, the rule did not change. The chain's abort text "seed 20261111 UNDRAWN" is fixed wording and was wrong for attempts 2 and 3. The draw was made once (attempt 2) and reused.

## Next
The pace term under its own freeze is the card's next lever. B1 rollout policy iteration stays blocked until the simulator is valid. On 2026-11-09 the PI moved the focus to optimising the P6 scorer; closed-loop reinforcement learning is one of its families and depends on this.
