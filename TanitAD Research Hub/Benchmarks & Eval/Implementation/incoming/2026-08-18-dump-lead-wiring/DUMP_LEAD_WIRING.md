# DUMP-LEAD WIRING — `win["lead"]` for BANKED tier-0 dumps (2026-08-18)

The work item `driving.py`'s retired refusal names ("These tier-0 dumps carry no `win['lead']`
yet, so here it is an unwired WORK ITEM, not an absence") now has its wiring. **The wiring is
landed, tested on synthetic AND real labels, and MEASURED-validated against two real banked
dumps. What it cannot yet produce is a distance-keeping number for any banked arm — because no
obstacle join for the VAL corpus exists anywhere yet (four probes below), and every banked dump
is val40.**

## What landed

| artifact | where |
|---|---|
| `taniteval/taniteval/dump_lead_join.py` | the wiring: agents-jsonl join + episode poses + banked dump → `win["lead"]` (lead_source block shape, dump row order) + per-episode coverage with status/reason — never a silent drop |
| `taniteval/tests/test_dump_lead_join.py` | 13 tests: hand-computable happy path, `four_families._distance_keeping` end-to-end on the emitted block, six refusal paths (NO_JOIN / NO_RECORD / SPEED_MISMATCH / GRID_MISMATCH / AMBIGUOUS_PREFIX / labels-ended≠road-clear), reader/probe units, and a REAL-DATA end-to-end on a lead130 clip (skipif off this host) |
| `dump_lead_validation.json` (this dir) | the MEASURED validation + all four coverage probes |

Division of labour it slots into: `stack/scripts/build_obstacle_join.py` (pod-side, raw
parquet → jsonl join, registration banked into `t_s`) → **this module** (dev-box, jsonl +
poses → `win["lead"]`) → `four_families.longitudinal(..., lead=win["lead"])` →
`lead_metrics.distance_keeping[_by_speed]` (admitted by D-LEAD-1). No code duplicated: lead
selection/frame chain stays in `lead_source`, the metric stays in `lead_metrics`,
`taniteval/tools/build_lead_block.py` (the raw-parquet pod path) is untouched.

## MEASURED — the wiring is proven on the real substrate (evidence: `dump_lead_validation.json`)

* **Alignment proof, label-free, both dumps, all 40 episodes each:**
  `max |win["speed"] − poses[origin,3]| = 0.0 m/s` (bit-exact) on `windows_flagship-30k.pt`
  (canonical eids) **and** `windows_flagship-v4.2-step4000.pt` (packed eids →
  `rollout.load_windows` normalisation exercised). `collect` persists speed as
  `ep.poses[origin, 3]` (rollout.py:203), so 0.0 proves eid→episode mapping, window grid
  (`window_last_indices`), and row order simultaneously. Episode poses: the sha256-verified
  poses-only val40 view (verified against `…/2026-07-26-s3-decision-grade/artifacts/
  manifest_EVALPOD_val40.json`; durable copy now at
  `C:/Users/Admin/tanitad-caches/val40-poses-20260818/`).
* **Coverage outcome:** 40/40 episodes `NO_JOIN` on both dumps; 881/881 windows `NO_LABEL`,
  each episode carrying its reason. No window silently dropped, none counted as free flow.
* **Real-label end-to-end (in the committed test suite, runs on this host):** lead130 clip
  `0045da77…` — v2ep poses (n_stack front-trim) + the md5-manifested lead130 join → identity
  roundtrip of a cuboid through rig→world→frame at 0.06 m tolerance, attach `OK`,
  LEAD windows present, `gap0 ∈ [0, 80] m`, `_distance_keeping` returns `OK` with the
  stratified block. Tests: **13 passed** (`test_dump_lead_join.py`), plus the lead subset +
  four-families + eid/driving neighbours: **112 + 43 passed**, 0 failed.
* **Defect found and fixed by the hand-computable fixture (MEASURED):** query times rebuilt
  from the affine grid fit can land ~5e-5 s below a row's ROUNDED `t_s`; the causal
  last-sample rule then returns the previous frame — a systematically one-frame-stale lead
  (~v·0.1 s displacement) on ~half the queries. Fixed with `QUERY_EPS_S = 2e-3 s` (10× the
  rounding band, 50× under the grid step); the fixture pins it.

## Coverage — why no banked arm gets a distance-keeping number today (four probes)

| probe | artifact | result |
|---|---|---|
| A | val40 id4 set × the lead130 join **on disk** (`lead130_agents.jsonl`, 130 clips) | **∅** |
| B | val40 id4 set × the FULL 2,308-clip train join census (`p2_lead_census.json` — independent artifact) | **∅** |
| C | id convention: v2ep record `0045da77…` carries `episode_id 808465461 == big-endian(b"0045")` — packed id IS the clip UUID's first 4 chars, so prefix probes are sound (an empty prefix intersection implies an empty true intersection) | **MATCH** |
| D | structural: lead130 is sampled from the full TRAIN join (`physicalai-train-e438721ae894`); the dumps' corpus is `physicalai-val-0c5f7dac3b11` (`p3_selection.json`, dump manifests) | disjoint by construction |

⇒ **The brief's deliverable 3 (distance-keeping CI on a banked dump over lead130-covered
windows) is impossible with any label material that exists today** — not an instrument gap, a
label-coverage gap. Per the brief, stopped at deliverable 2 + this evidence. No interval is
quoted in this report because no scoreable window exists; when they do, the block emits
`episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap` only (`lead_metrics._agg`),
and the tier stamp for these teacher-forced dumps will be **T0 — WM diagnostic, never driving
performance**.

## Reproduce

```
# tests (dev box; PYTHONPATH pins the healthy clone tree, stack_guard verifies it)
cd /c/Users/Admin/wt-tanitad-local/taniteval
PYTHONUTF8=1 PYTHONPATH="C:/Users/Admin/wt-tanitad-local/taniteval;C:/Users/Admin/wt-tanitad-local/stack;C:/Users/Admin/wt-tanitad-local/stack/scripts" \
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_dump_lead_join.py

# the validation + probes artifact
python <scratchpad>/dump_lead_validation.py            # wrote dump_lead_validation.json (this dir)

# the wiring itself, once a val join exists:
python -m taniteval.dump_lead_join --windows taniteval/results/windows_<key>.pt \
    --agents <val40 agents.jsonl[.xz]> --epdir <val40 ep_*.pt dir> --out lead_<key>.pt
```

## What remains (the actual gap, in order)

1. **Build the VAL-corpus jsonl join** — 40 clips of `physicalai-val-0c5f7dac3b11` through the
   existing `stack/scripts/build_obstacle_join.py`, wherever the `egomotion` +
   `obstacle.offline` parquet zips are reachable (pod-side; the val40 camera/label chunks are
   NOT on this box). Output is a few MB of jsonl — trivially shippable. It must also record the
   **clip UUIDs in val-list order** (the box only holds 4-char id prefixes; `attach_lead`
   accepts a `--clip-map` and refuses ambiguous prefixes).
2. Run the CLI over all 27 banked dumps → `win["lead"]` per arm → `four_families` distance-
   keeping with paired episode-cluster bootstraps vs the CV/hold-v0/CTRV floors, stamped T0.
3. Optional: a train-side arm dump (none exists — all 27 banked dumps are val40) would let the
   lead130 join score an arm TODAY; that requires a checkpoint rollout over train clips, i.e.
   GPU work, and was out of this brief's scope.

## Evidence classes

MEASURED: everything in `dump_lead_validation.json`; the test outcomes; the QUERY_EPS_S
defect. INHERITED (named source, not re-verified here): D-LEAD-1 admission numbers
(four_families.py docstring); the sha256 poses verification (`val40_hf_poses_verify.json`,
prior session, re-used byte-identical). The G: mount died mid-task (orchestrator-confirmed);
everything above lives in the local clone `/c/Users/Admin/wt-tanitad-local`.
