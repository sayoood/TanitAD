# P6 D7 refusal: the lane-rank twins were never on their own route, for two independent reasons

**Master Mind, 2026-10-16 ~08:15 Berlin.**
- P6's chain REFUSED at stage D7 at 05:51:27Z with `ZZD7_DISTINCT_BAD k=1 sampled=40 differ=0 frac=0.0`: every sampled rank-1 twin's metric-cache parts carried the logged route's centreline.
- **The gate did its job.** No wrong-route PDM label reached training, and the A40 was unaffected (P2-s1 trains; P6's GPU stages are queued behind R0c and S1c).

## MEASURED (`d7_probe.py`, `d7_probe.out`; pod; one twin, `2021.06.23.18.23.38_veh-26_00663_01217 / 88362ada700b543d`)
| hypothesis | probe | result |
|---|---|---|
| **H1** — the cache build's `spawn` pool workers never get the route patch | a spawned worker with `DRIVERL_EVAL_ROUTE_LANE_RANK=1` | `installed: false`. Its sitecustomize is the SYSTEM `/usr/lib/python3.11/sitecustomize.py`. `p6_pod.sh`'s comment "reaches the spawned pool workers through the venv's sitecustomize" is false. |
| **H2** — PDM-Closed never consults `_choose_route_edge` | in-process, patch applied by `route_rank.apply(1)` (which asserts `__wrapped__`), with a counting wrapper | **0 calls** during `metric_cache_parts(sc)`. The route digest `92578972…` is **identical** to an unpatched rank-0 build in a fresh process. |

⇒ **H2 is decisive.** NAVSIM's PDM-Closed picks its own starting lane and runs its own centreline search; the DriveRL route builder is not on that path. Installing the patch in the workers would still produce identical twins.

## Root-cause class
**"Built, tested, and unreachable from its caller"** (CLAUDE.md, now ≥ 6 instances), in two layers at once:
1. the patch was verified in the PARENT process, while the work ran in spawned children;
2. the patched function is not on PDM-Closed's code path at all.

`route_rank.apply`'s assertion (`_choose_route_edge` is wrapped) is true and proves nothing about the parts. A success flag was checked on an object the computation never touches.

SPEC_P6 §2 D7 had flagged this as UNVERIFIED and declared both the test and the fallback. The test is what caught it.

## Next lever (running)
A PDM-side rank-k patch: PDM-Closed's starting lane becomes DriveRL's rank-k lane of the same first roadblock (the same `_route_edge_anchor_score` ordering). It is installed in EVERY pool worker by an asserting `Pool(initializer=…)`. Verification on the pod:
- the SPEC test (first centreline lane == DriveRL's rank-k lane);
- distinctness;
- the twin target is closer to the twin centreline;
- **rank-0 parts bit-identical** to the existing cache;
- a no-initializer mutation that must FAIL.

**Deadline default:** if the fix is not verified before the A40 frees for P6, P6 relaunches with `P6_D7_FALLBACK=1`, the SPEC's declared fallback (twins train with trajectory loss only). The deviation from the paper is then reported.
