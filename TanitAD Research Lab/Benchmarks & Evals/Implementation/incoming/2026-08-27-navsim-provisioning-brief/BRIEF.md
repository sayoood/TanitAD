# COMMISSION — NAVSIM provisioning scout (Eval FlyWheel)

**Commissioned by** the Master Mind on **PI approval, 2026-08-27** (*"prepare and
do what necessary for NAVSIM, ask the EvalFlyWheel for it"*). **Priority** normal —
runs in the FlyWheel's next slots, does not preempt live training.

⛔ **This brief carries the `Project Steering/AGENT_OPERATING_STANDARD.md` preamble
by reference — its three binding rules apply verbatim: (1) STAGE, NEVER PUSH;
(2) end with a DELIVERABLE MANIFEST naming where every artifact lives;
(3) ESCALATE integration, never bury a "please merge" in a doc.**

## Why (context the agent needs)

Stage D of the v7r plan proves hierarchy dominance against REF-C/REF-D on our
parity corpus — but the JEPA-driving literature we position against proves itself
on **NAVSIM** (Drive-JEPA 93.3 PDMS v1 / 87.8 EPDMS v2; Latent-WAM 89.3 EPDMS v2 —
primaries banked in the Library, keys 74–78). An external yardstick beside our
parity numbers is what makes the dominance claim land outside the programme.
**The PI has approved preparation now; the actual dataset download/provisioning
executes only on a later explicit go.**

## Scope — ZERO-COST scouting first, spend nothing

1. **Harness:** the `navsim` repo — pin the exact commit/version current SOTA
   tables use; what its Agent interface requires (inputs per frame, output
   trajectory format, ego-status fields); licence check (research use).
2. **Data:** which split the headline numbers use (`navtest` for PDMS; v2/EPDMS
   split); **exact download size per split**, hosting (HF? their servers?), and
   whether a mini split exists that produces comparable-in-kind numbers for a
   smoke run. ⛔ Report sizes; do NOT download anything beyond metadata/docs.
3. **The wrap:** a one-page mapping from our stack's interface (256×640
   cylindrical front camera, `[ω, a_long, v]` state, (a,κ)-unicycle outputs) to
   their agent API — name every mismatch (their camera set, their coordinate
   conventions, their 4 s horizon vs ours) and the cheapest adapter for each.
   ⚠️ Flag anything that would make our number non-comparable to published rows
   (sensor set differences especially — many entries use multi-camera + LiDAR;
   a front-camera-only entry must be labelled as such, never hidden).
4. **Compute estimate:** eval wall-time per split on (a) Thor, (b) HF Pro GPU
   (quota ceiling binds — estimate only, no jobs), (c) dev box.
5. **PRIMARY SOURCES:** bank the NAVSIM paper + any cited eval-protocol doc via
   `python tools/kb_add.py <id> --tag navsim --cited-by <this dir>/RESULT.md`.

## Deliverable

`RESULT.md` in this directory: the five sections above, each MEASURED (from the
repo/docs) or PUBLISHED (banked), a **go/no-go cost table** for the PI (disk, download,
eval-hours per split), and the deliverable manifest. **No downloads, no jobs, no
spend — scouting only.** The provisioning decision returns to the PI with your
numbers.
