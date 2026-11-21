# GOALS — Tools & DevEnv

D-029 standing goals: 1–3 concrete, measurable objectives with a target number and a
deadline. Each run advances one with a *measured* step. **A goal with no movement for two
runs is escalated in STATE, not silently carried.**

Created 2026-07-21 — this file was missing for the whole life of the discipline, which is
itself the finding: three runs of tooling work with no standing target to hold it to.

---

## G1 — No silent fleet failure survives one probe cycle
**Target:** every dead/frozen/idle trainer on the 4-pod fleet is detected by
`tools/fleet_probe.py` **within one 6-hour cycle**, with **zero false GREENs** across a
4-week window. **Deadline:** W35 (2026-08-08).

| date | measured step | status |
|---|---|---|
| 2026-07-21 | Probe built + wired into `fleet-status`. Live: 4 hosts in **9.7–11.3 s**; caught pod2 idle (RED) and pod3 unverifiable (AMBER); 20 falsifiers incl. the direct no-false-GREEN regression test. | **on track** |

**Gap to target:** nothing runs it automatically — it is still a discipline. Next step is
the session/cron hook (shared with `ci_gate`, backlog P0#2). Until then the "within one
cycle" half of the target is *unproven*.
**Falsifier:** if a dead trainer is found by a human before the probe reports it, G1 fails
for that cycle and the miss is recorded in this table.

---

## G2 — Every viz artifact this project ships is verified, not asserted
**Target:** a `.rrd` produced by the standard replay path is **byte-verified non-stub**
(within 5 % of the single-sink baseline for identical input) and visually verified through
the Rerun 0.34.1 Viewer-MCP, by **W35**.

| date | measured step | status |
|---|---|---|
| 2026-07-21 | Baseline measured: **52,966 B/window** (200 win × 3 arms × 256², jpeg85, 299 win/s). Dual-sink path found to produce a **3,196 B stub — 3,314× loss**; guard shipped via intake. Documented tee fix **deadlocks** (negative result). | **at risk** |

**Gap to target:** (a) the two-`RecordingStream` tee is unwritten; (b) the Viewer-MCP has
not been wired into any agent tool list, so "verified" is still "asserted"; (c) measured on
synthetic records, not a real `ep_*.pt` episode.
**Falsifier:** dual-sink `.rrd` must land within 5 % of single-sink for identical input.

---

## G3 — Zero stranded deliverables at session end, fleet-wide
**Target:** `tools/session_guard.py` reports **0 uncommitted hub deliverables and 0
uncommitted `stack/` source** on the shared tree at the end of any agent's session, for
**two consecutive weeks**. **Deadline:** W36 (2026-08-15).

| date | measured step | status |
|---|---|---|
| 2026-07-18 | Guard shipped. Baseline: 5 hub files / 9 branches / 5 stale INTAKEs. | |
| 2026-07-20 | Source check added. **Got worse**: 30 hub files, **40 `stack/` paths (22 untracked)**, 11 branches, 8 stale INTAKEs. | **escalated** |
| 2026-07-21 | See STATE — re-measured this run. | |

**Gap to target:** the guard is advisory; nothing blocks a session that ignores it, and the
debt is created by six agents but owned by none. This is an orchestrator-policy gap, not a
tooling gap — flagged as such rather than answered with more tooling.
