
<!-- CLS-WEIGHT-VECTOR-NEEDS-THE-TRAIN-JOIN-MEASURED-2026-09-22 -->

### ⛔ 2026-09-22 — `H-BOXCLS-1`'s weight vector CANNOT be proxied from a held-out split: MEASURED sampling-sensitive to **2.26×** on a 62-clip draw ⇒ the TRAIN agent join is genuinely required

MEASURED by me, CPU only, read-only, no GPU
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-cls-weight-train-join/`).

`58ee9b5` landed the class-weighted `cls` capability and named what was still owed: the
**train-split** inverse frequencies and their stamped vector. This is that work, and it ends in a
blocker — established rather than assumed.

**Two probes say no TRAIN agent join exists anywhere I can reach.** Dev box:
`D:/…/b1-agent-join-3d-20260917/` holds only `b1eval_agents_3d.jsonl.xz`. Thor: nothing matching
`*agents*3d*.jsonl*` under `/home/nvidia` to depth 4. Building one is a **pod-side** step over the
2,376-episode parity corpus (`build_obstacle_join.py`), i.e. compute the PI provisions.

⛔ **And the eval census is not a substitute by default:** setting a TRAINING hyper-parameter from
the split the arm is SCORED on is precisely the leakage the probe discipline forbids.

⭐ **SO I RAN THE CHEAPEST EXPERIMENT THAT COULD HAVE REMOVED THE BLOCKER.** The eval corpus is
already cut into two **disjoint** clip-level halves (62/62, `a0_clean124_split.py`). If a class
distribution were stable under a 62-clip draw, a held-out split would be a defensible proxy and the
disagreement would BOUND its cost. Computed on the full 3-D join — 139 clips, 905,512 agents, all
10 classes present in **both** halves:

| class | halfA n | halfB n | wA | wB | ratio |
|---|---|---|---|---|---|
| **heavy_truck** | 10,056 | 7,000 | 0.08053 | 0.18189 | **0.443** |
| protruding_object | 1,577 | 1,450 | 0.51348 | 0.87807 | 0.585 |
| other_vehicle | 1,502 | 1,431 | 0.53912 | 0.88973 | 0.606 |
| automobile | 283,417 | 296,717 | 0.00286 | 0.00429 | 0.666 |
| stroller | 230 | 475 | 3.52072 | 2.68043 | 1.313 |
| trailer | 3,276 | 3,969 | 0.24718 | 0.32079 | 0.770 |
| rider | 8,840 | 12,449 | 0.09160 | 0.10227 | 0.896 |
| person | 59,575 | 98,562 | 0.01359 | 0.01292 | 1.052 |
| bus | 3,170 | 4,789 | 0.25545 | 0.26586 | 0.961 |
| animal | 171 | 273 | 4.73547 | 4.66375 | 1.015 |

⇒ **max disagreement 2.2587× (`heavy_truck`).** The weights compared are inverse frequencies
**normalised to mean 1**, on purpose: an unnormalised comparison would report agreement that is
really a shared constant, since the loss's own denominator absorbs scale (`58ee9b5`).

⛔ **VERDICT: a class weight vector is SAMPLING-SENSITIVE at 62 clips.** A proxy vector would be a
number with no evidence class, and it would be wrong by up to 2.26× on a class the arm exists to
rescue. ⇒ **the TRAIN join is genuinely required**, for two independent reasons now: leakage, and
sampling sensitivity *within* a split.

⚠️ **Scope, stated:** halfA and halfB are BOTH eval. This measures whether a 62-clip sample pins the
class mix — it says **nothing** about whether eval resembles train, and no claim here does.

⛔⛔ **TWO DEFECTS IN MY OWN PROBE, BOTH CAUGHT BY A COUNTER I HAPPENED TO REPORT.**
(1) The join carries a full **UUID** `clip_id` while the split is keyed on **sha12**; the first pass
left **ALL 905,512 agents unassigned** and would have printed an empty census as a clean result. It
was visible only because `n_agents_unassigned_to_a_half` was reported. The hash now comes from the
programme's own `s1_pass.sha12`, never a re-implementation.
(2) My first repair asserted **zero** unassigned — too strict, because the join has **139** clips
and the clean split **124**; the 15 outside are the stamp's own exclusions (11 inside parity TRAIN,
4 with no SAM3 map, 106,514 agents). The guard now asserts the **structure** (139 seen, 15 outside),
which still catches a wrong key while not lying about a correct one.
⭐ **A census with no "what did not match" counter cannot distinguish an empty result from a broken
join.** That counter was the whole difference between a finding and a fabrication.

⇒ **`H-BOXCLS-1` now has TWO blockers, not one:** the PI's GPU call for the arm itself, **and** a
pod-side TRAIN agent-join build without which no admissible weight vector exists. The second is new
information for that decision and is why it is recorded before the arm is proposed again.
