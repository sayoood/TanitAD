# RESULT — E-EGO-LIT-1: how successful planners handle ego-state input, ego dropout and speed-conditioned vocabularies — and what REF-C's origin paper (DiffusionDrive v1) actually does in code

*Architecture & Inference FlyWheel · Research Lab literature stream · 2026-09-05. **0 GPU.**
Research Lab agent; no sub-agents spawned (predecessor's four died unbanked).*

`status: IN PROGRESS — started 2026-09-05 04:05 (Europe/Berlin) — done: orientation (prior audit
7c67b3d + sibling E-REFC-EGO-1 read, library.json checked: 20 of the 25 target primaries already
banked) / next: §2 DiffusionDrive v1 code read at 9b52ed0, then §1 per-planner table, then §3
recommendation + pre-registration. Intermediate states committed via mm_commit.py.`

**PI question (verbatim):** *"do research how it is done in typical similar successful works and
the original paper of refc."*

**Tier stamp.** Every number in §1–§2 is **PUBLISHED (primary, read from a banked PDF or from
pinned source code at a named commit)** about other people's models on other people's
benchmarks; none is a TanitAD result. TanitAD numbers appear only in §3 and carry their
artifact path and tier (`D-REFCV4B-EGODROP1`, `H-ECHO-8`, `EGODROP_9500.json`).

**Builds on (cite, do not repeat):**
- `…/2026-09-05-refc-vs-diffusiondrive-audit/` (commit `7c67b3d`): DD's decoder conditioning
  as re-implemented by REF-C — status vector = command 4 + velocity 2 + acceleration 2 through
  the ego planning query; k-means anchors in metres, 20 on NAVSIM.
- `…/2026-09-03-refc-ego-inputs-and-anti-echo/RESULT.md` (E-REFC-EGO-1): TCP has no ego
  dropout; PlanTF SDE numbers; DRAMA 0.835→0.848; PLUTO +2.60; BEV-Planner/PARA-Drive echo
  asymmetry; CARLA-TransFuser Tab. 10 (velocity in the backbone: DS −11.33).

(sections follow as they are finished)
