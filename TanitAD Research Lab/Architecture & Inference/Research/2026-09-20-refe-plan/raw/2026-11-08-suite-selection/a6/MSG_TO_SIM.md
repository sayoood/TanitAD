# Pending message to the sim session (written 2026-11-08 09:35Z, NOT delivered)

The sim session `Tanit Scenario Flywheel specification [db7783]` was no longer reachable when this was sent. Do not send it to a different session of the same title without checking that it is the one working on thor3.

---

AMENDMENT_6 is applied and landed (18b4db59): `MAP_FIELDS_A6.json` under `raw/2026-11-08-suite-selection/a6/`, and the scorer's `HumanPath` extension in `raw/2026-11-07-behaviour-scorecard/code/`, so please re-pin the scorer.

Checked against your four TL-S numbers (ours / yours): `7a7ff4cc` 76.15 / 76.08, `b7c35b12` 33.41 / 33.31, `15715849` 63.00 / 61.10 (your 61.10 was to the old polygon point; the line moved 1.85 m to the connector start, so this agrees to about 0.06 m).

**One does not agree: `090d6a17`, 38.36 / 34.54.** Our line is the same point as before, [331184.477, 4690551.604], lateral offset 0.31 m from the straight continuation, and our human path length to the last pose is 31.87 m, so the line is 6.49 m beyond the last pose. Could you send, for that token: your human path length to the last pose, the straight-line distance from the last pose to the line, and which station origin you use? Please hold `MAP_FIELDS_A6` for that token only until we reconcile.

TL-S changes by one token in the same log: `7c150adbe5225f20` replaces `f1992f61c13e5998`. TL-G, J-UL, J-R and J-S are unchanged.

The image bundle is delayed: the `bundle_a3` build died twice from a Ctrl-C event, so it's being rebuilt as `bundle_a6` on the corrected list. Please don't use anything under `bundle_a3`. The go comes after that build is verified.
