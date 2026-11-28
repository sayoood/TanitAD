# REF-C v3 — TRAINING READINESS ON THE v7.2 / B1 CORPUS

**Date:** 2026-09-01 · **Agent:** Architecture & Inference FlyWheel (branch `agent/arch-inf-20260803`)
**PI directive:** 2026-09-01 — *"refav1 and refcv3 should be also prepared for training"* (four
reference arms same-corpus comparable on B1).
**Raw evidence:** `raw/` beside this file (probe scripts + verbatim outputs). Every claim carries
its evidence class; MEASURED numbers name the artifact.

**HEADLINE: REF-C v3 now trains end-to-end off the B1 `*.v2ep.pt` epcache at its native
256x640 cylindrical geometry — MEASURED by a real 2-step run on pulled B1 clips
(`raw/b1format_v2cache_2step_smoke_and_anchors.txt`). Three blockers were found and fixed
(minimal diffs, §3). No derived frame/feature cache is needed — the arm trains straight off
the epcache (§5); its one derived artifact is an 8 KB anchor vocabulary. Launch remains
blocked on compute placement, a prereg amendment, and an eval-side adapter (§7).**

---

## 1. What the arm IS — constructed and counted, not quoted

Architecture (design: `…/2026-08-18-refc-v3-design/REFC_V3_DESIGN.md`, edges E1–E12;
module `stack/tanitad/refs/refc_v3.py`): an UNMODIFIED `RefCModel` core (ResNet-34-style
trunk, anchored-truncated-diffusion trajectory decoder, 128 anchors, LAW/route/factored-tactical
aux) + a gated goal cascade — `PhiTac` causal-TCN tactical state, 195-param strategic goal head
off the `StrategicCtx` GRU, zero-init FiLM conditioning, factored lat/lon z_tac heads, tactical
geometric goals at {2,4,6} s, and `GoalDistanceScorer` selection behind a zero-init `goal_gate`
with the `refc_select` seam clamp. `hier=False` is a transparent wrapper around the core
(state_dict under `core.`).

**Parameter counts — MEASURED 2026-09-01 by building every rung and counting
(`raw/build_probe_params_nonsquare.json`, `param_breakdown_v3`):**

| rung | arm | core | hierarchy | TOTAL |
|---|---|---|---|---|
| small (bw 64) | hier | 60,882,074 | 2,053,475 † | **62,935,549** † |
| small | flat | 60,389,402 | — | **60,389,402** |
| xl (bw 124) | hier | 215,589,550 | 2,176,355 † | **217,765,905** † |
| xl | flat | 215,096,878 | — | **215,096,878** |

† at the module DEFAULT `tac_vocab_version="v7.0"` (8x8 heads, PI mandate 2026-08-27). At the
**kin3 pin this trainer now enforces (§3.2)** the registered numbers hold exactly:
**small/hier 62,930,419 · xl/hier 217,760,775** — MEASURED in the post-fix preflights
(`raw/preflights_kin3_256x256_and_256x640.txt`, `tac_heads 9,234`). The v7.0-vs-kin3 head cost
is exactly +5,130 at both rungs, already pinned by
`tests/test_refc_v3.py::test_the_two_size_rungs_build_and_measure`.

**Dominance delta — MEASURED, derived not asserted:** `config_delta(H, F)` =
`{hier, core.graft_target_latent}` at both rungs = the registered lever set (C122 gate green;
the trainer refuses any other delta at build).

**D-008 context (INHERITED from the scale-matrix suite, re-run green today):** the incumbent
"thin" cascade clears D-008's ≥250 M in NO cell; only xl+`aligned` (the 2026-08-21 review's
tier) reaches it. The trainer below trains the THIN incumbent — which rung/tier a B1 campaign
runs is a PI/orchestrator decision, not settled here. `--size {small,base,xl}` moves the
encoder only.

## 2. The B1 corpus and what the trainer expects

**B1 cache (INHERITED from `Project Steering/SPEC_V7_LABEL_TRAINER_WIRING.md`, measured there
2026-08-30):** `/home/nvidia/data/physicalai-b1-w120-256x640cyl` on **Thor** — **4,713
episodes / 177,998,547,213 B (~178 GB)**, val40-clean by digest (0 of 4,713), geometry
`requested_hfov 120.0 / achieved 120.0 / f_eff 305.5775`. Sibling eval cache
`physicalai-b1-EVAL6-w120-256x640cyl` (6 eps) beside it.

**B1 payload format — MEASURED on a pulled sample (24 clips,
`C:/Users/Admin/refav1_probe/eps/`; `raw/b1_sample_probe.json`):** `*.v2ep.pt`, codec **png**
(magic 89-50-4E-47), `n_stack 3`, `image_h/w 256/640`, frame dict
`{256, 640, f_ref 305.5774907, projection: cylindrical}`, poses `[T,4]` f32, actions `[T,2]`
f32. Through `build_v2_providers` the episode surface is `frames [T-2, 9, 256, 640] u8`
(lazy slice-decode, non-zero content verified: mean 26.4), `poses/actions [T-2, ·]` —
**exactly the `ToyEpisode` contract the REF-C window datasets consume.**

**What the trainer expected BEFORE today (MEASURED audit):** `refc_v3_train.py` had exactly two
sources — `--data-root` (raw `ep_*.pt` via `refb_train.load_cached_episodes`, parity-guarded)
and `--synth-episodes` (CI-only). **No v2 path existed** in `refc_v3_train.py` or
`refc_train.py` (the `--v2-cache` machinery lives in `train_flagship4b.py` and friends).
And the model could not be BUILT at 256x640 at all (§3.1).

## 3. Blockers found and fixed — each a minimal diff, none a redesign

### 3.1 Non-square construction crash (`refc.py:974`) — FIXED
MEASURED (`raw/build_probe_params_nonsquare.json` documents the post-fix build; the pre-fix
crash is in this package's audit trail): building any REF-C config with `image_width=640` died
at construction — `ValueError: REF-C feature map is 8x20 (non-square) — this caller still
reads the scalar 'grid'` — because `ResNetEncoder.__init__` eagerly read the
deliberately-raising `cfg.grid` scalar. Every in-repo consumer of the map is shape-agnostic
(`flatten(2)` / `grid_shape`; the imagination field — the one square reshape — is OFF in the
v3 core config). **Fix:** the eager read became a read-time `grid` property with the identical
raise; `grid_shape` is stored. Square behavior is byte-identical (`tests/test_refc.py` asserts
`encoder.grid == 8`; suite green §6). **After the fix the FULL v3 forward runs at 256x640** —
fan, selection, goals all emitted (same artifact, `nonsquare_forward` block).

### 3.2 The kin3/v7.0 vocabulary wiring gap — FIXED (found by this audit)
`RefCV3Config.tac_vocab_version` defaults to `"v7.0"` (8x8, PI vocab mandate) and its own field
comment says *"the kinematic trainer passes 'kin3' explicitly"* — **but the trainer never did**,
and its width refusal checked only the CORE's head. MEASURED pre-fix
(`raw/preflight_PRE-FIX_v7vocab_default.txt`): the preflight **PASSED** while building 8-wide
z_tac heads (`tac_heads 14,364`, `lat_tac`/`lon_tac` init-CE ≈ ln 8) that `compute_losses_v3`
supervised with 3-class kinematic labels — the exact *"train silently WRONG classes"* case its
comment names — and `derive_man5_logprobs` (a [B,3]x[B,3] contract) read v7.0's
LANE_CHANGE_L/R slots as turn_left/right into the H19 prior. **Fix:** `_pin_trainer_cfg` pins
`"kin3"` on every config this trainer builds (both arms, all paths, delta pair included), and
the loud width refusal now also covers `lat_logits_tac`/`lon_logits_tac`. The v7.0 head is the
go-forward space and NEEDS v7 labels, which are not deliverable today
(`SPEC_V7_LABEL_TRAINER_WIRING.md`: the S2 label blobs are on no reachable path) — so this is
a pin, not a flag. `tests/test_refc_v3_lan_preflight.py` builds its models through the same
pin now (`_smoke_cfg`), since it exercises the trainer's loss surface.

### 3.3 No v2/B1 data path — FIXED (flagship recipe, transplanted)
`refc_v3_train.py` gains `--v2-cache` (+ `--v2-lru`, default 6 — B1 payloads are ~34 MB/clip;
the old "2–4 MB" sizing is 8–17x low, V5F_SIGKILL lesson, and under shuffle the LRU hit rate
is ~lru/n_clips so big values buy RAM pressure, not throughput), `--require-parity`, and
`--image-hw H W` (builds the encoder at the corpus geometry; params UNCHANGED — fully-conv
trunk — only compute scales). Structure mirrors `train_flagship4b.py` verbatim: membership
guard (`assert_v2_parity_cache`) before any GPU work, lazy LRU-bounded providers,
provider-vs-guard count cross-check. Added: a fail-loud **geometry assertion against the
EPISODES** (a 256x640 corpus fed to a 256x256 build would run — conv is size-agnostic — and
silently train a model whose config lies about its input), the launch-line-P4 episode/window
count print, and config.json now records `image_hw`, `tac_vocab_version`, `v2_cache`,
`v2_parity`. `build_refc_anchors.py` gains the same `--v2-cache` (poses-only manifest scan —
no frame decode).

## 4. The smoke — actual outputs (all MEASURED today, dev box, CPU only)

* **Preflight, hier, kin3, 256x256** — PASS, exit 0. `total 62,930,419` (registered number
  restored), freeze-history gate pass (`pooled_rel_move 0.0` — the built-in negative control
  bit-identical; history grads nonzero at all 8 slots), E11 v0-vs-frames audit clean, finite
  loss step. (`raw/preflights_kin3_256x256_and_256x640.txt`)
* **Preflight, hier, `--image-hw 256 640`** (the B1 geometry) — **PASS, exit 0**, same gates
  at full size. Flat arm at 256x640 — PASS, `60,389,402`.
* **End-to-end trainer smoke on real B1-format clips** — `--arm hier --v2-cache
  <24 pulled B1 clips> --image-hw 256 640 --steps 2 --batch 2`: NON-PARITY loud line fired
  (B1 unregistered, §7.3), `24 episodes -> 4,102 windows (window 8, max_horizon 20, image_hw
  (256, 640), tac_vocab kin3)`, two finite steps (loss 123.28 → 116.79; route CE live at 0.18
  on step 1; `slot_valid_frac` 0.81/1.0 — the masked 6 s horizon working on real clips), ckpt
  + `summary.json {"done": true}` written. **~5.6 s/step at batch 2 on dev-box CPU including
  PNG decode** (`raw/smoke_metrics.jsonl` — a functional number, NOT a launch-device
  throughput). Durable record verified: `raw/smoke_config.json` carries image_hw [256,640],
  tac_vocab kin3, the v2_cache path, `v2_parity.parity false`, the pinned delta, and
  `param_breakdown.total 62,930,419`.
* **Anchor vocabulary off the v2 cache** — `[20, 8, 2]` FPS anchors from a 3,334-window pool,
  poses-only. (`raw/b1format_v2cache_2step_smoke_and_anchors.txt`)
* **Tests:** §6.

## 5. Derived artifacts — "does it need a cache like refav1's DINOv3 cache?"

**No. REF-C v3 trains STRAIGHT OFF the B1 epcache — that is the arm's advantage.** Frames are
lazily slice-decoded per window by the v2 providers; every label (waypoints, factored kin3
tactical, v2.1 route, E4.1 goals, LAW target) is derived at runtime from poses/frames already
in the cache. The ONE derived artifact is the **6 s anchor vocabulary**: 128 anchors × 8 slots
× 2 f32 ≈ **8 KB** on disk, built by `build_refc_anchors.py --v2-cache` from a poses-only
manifest scan — MEASURED seconds for 24 clips with the manifest present; the one-off manifest
build over 4,713 clips is a metadata-only mmap pass (ESTIMATED minutes on Thor; the B1 cache
may already carry `_v2manifest.pt` from other trainers, making it instant).

## 6. Tests

`pytest -q -k "refc or anchor"` from the mirror (`C:\Users\Admin\tanitad-wt\stack`), post-fix:
**321 passed, 18 skipped, exit 0** (baseline pre-edit `-k refc`: 189 passed, 4 skipped — the
wider `-k` additionally selects every anchor-named test across the suite, all green, so the
edited surface is covered from both sides). One transitional failure set
was found and fixed in the same turn: the new z_tac width refusal (§3.2) correctly fired on
`test_refc_v3_lan_preflight.py`'s raw-default smoke configs — 6 tests updated to build through
the trainer's own kin3 pin (`_smoke_cfg`), then 11/11 pass in that file. Skips are
pre-existing environment-conditional skips, untouched.

## 7. What remains blocked, and by what

| # | item | blocked by | owner |
|---|---|---|---|
| 1 | **Compute placement** | B1 (178 GB) exists ONLY on Thor; Thor's GPU is owned tonight by H-REFAV1-MOTION + a cache build, and ⛔ never add load to a training box. Options: (a) run on Thor when idle — zero transfer; (b) ship the cache to an idle pod A40 — pod-direct ssh ESTIMATED ~1.2 h at the C56 42 MB/s (probe the direct mapping first: it can be dead while the pod is healthy), or HF relay at ~118 MB/s (⛔ HF quota is a hard ceiling — check remaining quota BEFORE; never escalate spend autonomously) | PI / orchestrator |
| 2 | **Prereg amendment** | `PREREG_REFC_V3.md` §8 pins train = `physicalai-train-e438721ae894` and eval = canonical val40/881; §7's cost row is a 256x256/old-corpus measurement. A B1 campaign amends §8 (corpus; val choice — EVAL6? the registered val40 256x640cyl v2 sibling?), §7 (cost re-derived per §8 below), logged in §10 BEFORE any read — legitimate now, nothing has trained | orchestrator (prereg discipline) |
| 3 | **B1 parity registration** | `parity_manifest.json` carries NO `physicalai-b1-*` key (verified by hydrated Select-String; the registered keys are exactly the four e438721ae894 / 0c5f7dac3b11 family keys). Until registered (`register_v2_sibling.py`-class run WHERE the cache lives, Thor), every arm trains under the loud NON-PARITY line and `--require-parity` cannot enforce the four-arms-same-corpus guarantee | Data FlyWheel / orchestrator |
| 4 | **Eval adapter for v3** | NO `refc_v3`/`RefCV3` reference exists anywhere under `taniteval/` (two-probe absence: rg over `taniteval/taniteval/` + PowerShell Select-String over all of `taniteval/`). The FLAT arm is evaluable today (its state_dict is the core under `core.`; strip into a plain `RefCModel` for `refc_eval.collect`). The HIER arm needs a small constructor+forward adapter so the goal-selected `traj` is what gets scored (its forward already emits the same keys plus the v3 extras; `V3_HORIZONS[:4]` = the shared 5/10/15/20 WP_STEPS by design). Needed before the 5 k read, NOT before launch | Benchmarks & Evals FlyWheel |
| 5 | **v7-vocab arm** | S2 label artifacts (`s2_labels_v7.2_train.jsonl.gz` + clip indexes) are on no path the trainer can reach (SPEC_V7_LABEL_TRAINER_WIRING.md). kin3 is what today's B1 launch supervises; the v7.0-head arm follows the label manifest | Data FlyWheel |

## 8. Cost — ESTIMATED, with the basis named, and the re-time rule applied

The registered cost line (7–9 h A40 / 30 k, small) is a 256x256, 2,376-ep, A40 measurement —
**it does NOT port to B1** (corpus 4,713 eps ≈ 806 k windows, ESTIMATED from the measured
170.9 windows/clip × 4,713; geometry 2.5× conv area; possibly a different device). Per the
2026-08-29 corollary (*"when you correct the artifact, RE-RUN the cost — never port the old
timing onto the new format"*), the launch sequence below includes a **timed 50-step smoke on
the launch device** as the cost instrument; until it runs, the only honest statement is:
ESTIMATED ~2–2.5× the registered per-step cost from the conv-area ratio, i.e. ~15–22 h
A40-class per 30 k run, wide error bars. Steps (30 k) and batch (20) are the registered knobs;
30 k × 20 over ~806 k windows ≈ 0.74 epochs — still under one epoch, as on the old corpus.

## 9. THE EXACT B1 LAUNCH SEQUENCE (for the orchestrator — flags spelled out)

On the box that holds the cache (Thor paths shown; substitute the pod cache root after a
transfer). `<STACK>` = that box's `stack/`; every command needs `PYTHONPATH=<STACK>` (the
ModuleNotFound / editable-install trap) and `OMP_NUM_THREADS=6` (torch's ~113 threads/proc).
`<EXP>` = the experiments dir.

```bash
# 0. once — the anchor vocabulary (poses-only, minutes, 8 KB output):
PYTHONPATH=<STACK> OMP_NUM_THREADS=6 python3 <STACK>/scripts/build_refc_anchors.py \
  --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --out <EXP>/refc_v3_anchors_b1_6s_128.pt \
  --n-anchors 128 --horizons 5,10,15,20,30,40,50,60 --max-pool 200000 --seed 0

# 1. preflight AT THE LAUNCH GEOMETRY (both arms; any FAIL stops):
PYTHONPATH=<STACK> python3 <STACK>/scripts/refc_v3_train.py --arm hier --preflight \
  --image-hw 256 640 --out /tmp/v3pf
PYTHONPATH=<STACK> python3 <STACK>/scripts/refc_v3_train.py --arm flat --preflight \
  --image-hw 256 640 --out /tmp/v3pf

# 2. timed cost smoke (the §8 instrument; also the §7.1 throughput read):
PYTHONPATH=<STACK> OMP_NUM_THREADS=6 python3 <STACK>/scripts/refc_v3_train.py \
  --arm hier --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --image-hw 256 640 --out <EXP>/v3-b1-timing-smoke --steps 50 --batch 20 \
  --v2-lru 6 --workers 6 --log-every 10 \
  --anchors <EXP>/refc_v3_anchors_b1_6s_128.pt

# 3. the arms (seed-0 pair first; seeds 1-2 only after the prereg §6.4 5 k gates):
PYTHONPATH=<STACK> OMP_NUM_THREADS=6 nohup python3 <STACK>/scripts/refc_v3_train.py \
  --arm hier --seed 0 --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --image-hw 256 640 --out <EXP>/refc-v3-b1-hier-s0-30k --steps 30000 \
  --mode diffusion --batch 20 --v2-lru 6 --workers 6 --goal-str \
  --anchors <EXP>/refc_v3_anchors_b1_6s_128.pt \
  > <EXP>/refc-v3-b1-hier-s0-30k/train.out 2>&1 < /dev/null &
# flat: same line with --arm flat, no --goal-str, a -flat- out dir.
```

Notes that travel with the lines: `--goal-str` is hier-only by construction (the flat arm has
no goal heads); kill only by explicit PID; `< /dev/null` guards the nested-ssh stdin trap; the
trainer writes its own `summary.json` done-marker (the v5f supervisor-resurrection lesson);
after registration lands (§7.3) add `--require-parity` to every line — that flag is what turns
"same corpus" from a promise into a refusal.

## 10. Pre-launch gates that apply

1. **The trainer's own `--preflight` at the launch geometry** (§9 step 1): C122 delta pin →
   registered lever set; C115 freeze-history gate (H arm moves, `pooled` bit-identical); E11
   v0-vs-frames intervention audit; finite full-size loss step. Any FAIL = do not launch
   (the prereg's OUTCOME-V machinery).
2. **Prereg §6.4 sanity gates at 5 k** (seed-0 pair) before seeds 1–2: finite curves,
   freeze-history re-pass on the launch build, intervention audit clean, seam telemetry not
   saturated.
3. **`Project Steering/GATE_PROTOCOL.md`** governs any mid-run restart/continue verdict
   (`stack/scripts/run_gate.py`): the verdict names its horizon and n; co-primary
   corridor-departure at a pre-registered K — never a bare exponent, never `ade_0_2s` alone.
4. **Box-state preflight** (each item has cost hours before): disk by real `dd` write, never
   `df`; code by CONTENT grep on the running box (e.g. `grep -c "grid_shape = cfg.grid_shape"
   <STACK>/tanitad/refs/refc.py` ≥ 1 — a stale checkout resurrects the §3.1 crash); import
   probe `python3 -c "from tanitad.refs import refc_v3; print(refc_v3.V3_HORIZONS)"`; GPU
   idle + no training process on the box; `OMP_NUM_THREADS=6` exported.
5. **Eval reads** stay governed by the four-family rule (ADE alone is an incomplete result),
   the paired episode-cluster bootstrap, the σ ≤ 0.8 m @ 2 s selection admission with its two
   controls, and the T0/T1 tier stamp — all already committed in `PREREG_REFC_V3.md` §4–6.

## 11. DO NOT LAUNCH

⛔ **This deliverable makes REF-C v3 LAUNCHABLE on B1; it does not launch it, and nothing here
is authority to.** Launching is the orchestrator's call, after: (1) compute placement (§7.1 —
Thor is NOT idle tonight), (2) the prereg §10 amendment for the B1 corpus/val/cost (§7.2),
(3) the timed cost smoke (§9 step 2). The dev-box GPU was not touched for any of this
(CPU-only smokes, per the operating constraint).

---

### Deliverables of this package
* `stack/tanitad/refs/refc.py` — §3.1 fix (read-time `grid` property, `grid_shape` attr).
* `stack/scripts/refc_v3_train.py` — §3.2 kin3 pin + z_tac width refusal; §3.3 `--v2-cache` /
  `--v2-lru` / `--require-parity` / `--image-hw`, geometry assertion, P4 count print,
  config.json provenance fields.
* `stack/scripts/build_refc_anchors.py` — §3.3 poses-only `--v2-cache` source.
* `stack/tests/test_refc_v3_lan_preflight.py` — §6 (`_smoke_cfg` kin3 pin at the trainer-loss
  call sites).
* This file + `raw/` (probe scripts, pre-fix and post-fix preflight outputs, the B1-format
  2-step smoke config/metrics, the sample-payload probe).

*Written by the Architecture & Inference FlyWheel agent; staged, never committed (agent
operating standard).*
