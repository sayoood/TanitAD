# E1b — substrate verification (P0 gate) — all MEASURED

`2026-07-25` · `tanitad-pod3` (A40, idle: 0 MiB used / 46068 MiB, 0 % util) ·
read-only probe, zero training. Artifact: `tanitad-pod3:/workspace/e1b/probe_substrate.json`
(reproduce: `e1b_probe_substrate.py`). Every row below is `MEASURED` on the pod.

## 1. Episode-id DISJOINTNESS (the leak gate) — PASS

Method: `str(load_episode(ep, mmap=True).episode_id)` for **every** `ep_*.pt` in each cache;
set intersection at the byte level.

| cache | role | files | distinct ids | ∩ with the other |
|---|---|---:|---:|---:|
| `…/pai_epcache/physicalai-train-e438721ae894` | mining source (fine-tune) | 2376 | **2342** | — |
| `…/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6` | held-out EVAL (E1a's set) | 44 | **44** | — |
| **intersection** | | | | **0 → DISJOINT = True** |

⇒ Fine-tuning states are mined from episodes **provably disjoint** from the eval set. The trainer
re-checks this at startup (`--assert-disjoint-heldout`) and refuses to run on any overlap. The
leaky split `physicalai-val-f1b378f295ae` (78.5 % into parity train, per E1a §1.1) is used **nowhere**.
*(The 2376→2342 gap = a few parity episodes share an `episode_id`; irrelevant to the eval-set
disjointness, which is 0.)*

## 2. REF-C base checkpoint — loads STRICT, runs

| field | value |
|---|---|
| path | `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt` (1.25 GB) |
| step | **29999** (final); milestones `ckpt_step{5000,15000,20000}.pt` present |
| `load_state_dict(strict)` | **0 missing / 0 unexpected keys** |
| anchors buffer | **`[128, 4, 2]`** (base preset = 128 anchors, confirms E1a; XL=256) |
| params | **104,191,577** (matches `MODEL_REGISTRY.md` §4 base) |
| config.json | base_width 88, d384/4L, `diffusion_steps 2`, hierarchy+graft_maneuver on, imagination off — `refc_config()` |
| forward | `model(frames, nav_cmd=None, v0, steps=2)` → `traj [1,4,2]` finite, `anchor_logits [1,128]`, `anchor_traj [1,128,4,2]` |

## 3. Caches & harness

| item | value |
|---|---|
| parity-train episodes | **2376** in `physicalai-train-e438721ae894` (the sacred parity corpus; skip-hash `f09e44db`) |
| parity-train episode length T | ∈ **[197, 199]**, median 199 (sample n=119) → ~1 window/episode at K=185 (mining lever = episode count) |
| held-out eval | 44 `ep_*.pt` (+ `DONE`), T ∈ [190, 199] |
| E1a harness | `tanitad-pod3:/workspace/e1a_e2a/e1a_horizon.py` (+ `lowood_flagship_ci.json`, vendored `taniteval_ci.py`) — reused VERBATIM by E1b |
| stack / venv | `/workspace/TanitAD/stack` present; `/workspace/venv/bin/python` (pods run this, not `python3` — the `ps -C` trap) |
| rollout cost | E1a K=185 = **711 s / 43 windows ≈ 16.5 s/window** (`e1a_heldout44_K185.log`) → 400-episode mine ≈ 1.7 h |

## 4. Verdict

**P0 PASS.** REF-C base loads and runs; the mining source is byte-level disjoint from the eval set;
parity is intact (2376). Cleared to mine → CL-SFT → (later) paired eval. No P0 blocker.

## 5. Stranding found (escalation — see report)

Checked at two locations (operating standard: absence at one location is not absence). Result:

- **Committed already** (NOT stranded): the E2a dependencies `perturb.py` + `recovery_probe.py`
  (under `…/incoming/2026-07-23-refc-planner-closedloop/`) and `lowood_flagship_ci.json` (under
  `…/Benchmarks & Eval/…/2026-07-23-lower-ood-closedloop-source/`).
- **STRANDED on pod3 only:** the two E1a/E2a **driver** scripts `e1a_horizon.py` and
  `e2a_localize.py` — the instruments that produced the committed E1a/E2a result JSONs. Commit
  `2d6589b` staged the JSONs + `E1a_E2a_RESULTS.md` but **not** the code that made them (every E1a
  headline is currently un-reproducible from the repo). This is the "good work stranded on one pod"
  failure the operating standard targets.
- **Rescued by E1b:** `e1a_horizon.py` is now staged at `scripts/e1a_horizon.py` (E1b reuses it).
  **`e2a_localize.py` still lives only on `tanitad-pod3:/workspace/e1a_e2a/`** — needs homing into
  the E1a results dir by Sayed / the E1a author.
