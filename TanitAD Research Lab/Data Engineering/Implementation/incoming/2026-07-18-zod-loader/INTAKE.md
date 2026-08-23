# INTAKE — ZOD loader (CC-BY-SA-4.0, the #1 owned real-urban ingest)

- **Package:** `Data Engineering/Implementation/incoming/2026-07-18-zod-loader/`
- **Author agent / date:** Data Engineering agent (Tuesday), 2026-07-18
- **Proposed target:** `stack/tanitad/data/zod.py` (sibling of `cosmos_drive.py` /
  `pandaset.py`) + `stack/tests/test_zod.py`
- **Hypothesis / WP served:** OWN_DATASET_PLAN §7 ingest #1 / FLEET_REVIEW 2026-07-17
  P0 #1 (corpus diversity — the pending data-side gap) / H4 arm-B (EU/night/winter) /
  H7 (data flywheel, real-CAN anchor #2)

> ⏹ **PARTIALLY CLOSED 2026-08-16 — the "P0 #1 / pending data-side gap" framing is SUPERSEDED.**
> ZOD is no longer the #1 owned real-urban unlock. **Argoverse 2 landed instead** and its adapter is
> integrated: `stack/tanitad/data/argoverse2.py` (MEASURED — file present in the tip working tree;
> registered in `stack/tanitad/lake/schema.py`). The re-scope was decided by the 2026-07-26 ingest
> package, whose commit reads verbatim *"⛔ ZOD HAS NO LANE GRAPH — my recommendation was wrong and
> the PI approved on it. AV2 1000/1000 DECISION-GRADE. Overture is the real answer."*
> Evidence (MEASURED): commit `6016736`; `…/Data Engineering/…/incoming/2026-07-26-av2-zod-ingest/AV2_ZOD_INGEST.md`
> §4a (three independent probes: the ZOD devkit's `zod/anno/lane.py` has **no successor / predecessor /
> neighbour / topology field on any class**, and no map or graph module at all).
> ⚠️ **This does NOT retract the loader.** ZOD stays valuable in the §5c role (ZOD imagery + Overture's
> routable graph, both `owned-safe` share-alike). It is the *priority* that moved, not the geometry.
> Swept by the 2026-08-16 stale-blocker sweep.

## What & why (≤10 lines)

A contract-clean loader for **ZOD** (Zenseact Open Dataset) — the FLEET_REVIEW #1
unlock and OWN_DATASET_PLAN's headline owned real-urban corpus: **CC-BY-SA-4.0**,
14 European countries, day/night/seasons/weather, **real CAN steering + OxTS RT3000
ego-motion** — the diversity the current 74%-straight day-only mix lacks (the enabling
condition of the ego-status shortcut). **Key reuse result:** ZOD's front camera is a
**Kannala-Brandt fisheye**, and KB's radius `r(θ)=f(θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹)` is EXACTLY
`calib.FThetaIntrinsics.poly` (odd-power) — so `kb_to_ftheta` reuses the proven f-theta
crop path with **zero new geometry math**. OxTS heading drives yaw directly (offset-free,
unlike PandaSet's camera-heading fallback) → the tested `cosmos_drive.poses_to_signals`.
`CORPUS_META` byte-identical to comma2k19 (I7 → admissible in the D-010 mix). CC-BY-SA →
a SEPARATE shard (ShareAlike firewall), never co-mingled with the permissive core.
Note: `Research/2026-07-18-zod-loader-and-geometry-falsifier.md`.

## Evidence & tests

- Tests: `tests/test_zod.py` — **19 passed (2.1 s)** on author machine (venv
  `C:/Users/Admin/venvs/tanitad`, torch 2.11, py3.13). Zero real bytes / zero Pillow
  (decode + OxTS injected). Covers: the exact KB↔f-theta poly identity, the geometry
  falsifier (below) + a narrow-FOV witness, fail-loud `_canonicalize`, OxTS-heading
  arc steer recovery / straight-line / low-speed guard / standstill heading,
  WGS84→ENU, CAN steer-ratio recovery, `assert_contract(channels=9)`, I7 fingerprint
  == comma2k19, I3 sequence split, mix admissibility (`MixedWindowDataset`), episode-id.
- **MEASURED geometry falsifier (grounded on the published spec — 120° HFOV,
  3848×2168, equidistant KB; the exact per-drive KB is access-gated):**
  ZOD front → **f_eff = 266.0 px, observed_frac = 1.00, drop_in = True**
  (crop side 1648 px, fully inside frame). **Robust to the real KB coeffs:** f_px=1780
  + realistic k1=−0.05/k2=0.007 → still **266.0 / 1.00** — the FOV alone decides it.
  Narrow-40°-HFOV witness → observed_frac 0.34, f_eff 642 → drop_in False (the gate is
  not vacuous). **Falsifier PASS: ZOD is geometrically unblocked** (contrast PandaSet,
  height-bound at f_eff 467). No escalation on geometry — the falsifier did not trip.
- **Runnable real-bytes job card** (`zod_pilot_jobcard.md`, M-1.3/M-3): access request →
  5-drive ZOD-mini fetch → real KB from `calibration.json` → `verify_real_clip` +
  epcache precompute → push. Blocked only on dataset ACCESS (escalated).

> ✅ **RE-CONFIRMED STILL TRUE 2026-08-16 — ZOD access is STILL not granted, and the loader is STILL
> not integrated.** Probed three ways: (1) `stack/tanitad/data/zod.py` is **absent** from the tip
> working tree (`stack/tanitad/data/` holds `argoverse2 / calib / comma2k19 / cosmos_drive / l2d /
> nuscenes / pandaset-less …`, no `zod.py`); (2) `git log --all -- stack/tanitad/data/zod.py` returns
> **nothing** — it was never committed on any branch; (3) `grep -ri zod` over `stack/` hits only the
> *license registry* (`lake/schema.py:108` `SourceLicense("owned-safe","CC-BY-SA-4.0",share_alike=True)`,
> `lake/license_guard.py:30`, `data/calib.py:702`) — the licence row exists, the loader does not.
> The ACCESS step itself was **independently re-probed on 2026-07-26** and remains a human application:
> `…/incoming/2026-07-26-av2-zod-ingest/AV2_ZOD_INGEST.md` §4c — *"The download URL is the credential"*,
> obtained by submitting the request form, and §5b still records *"it still requires the ZOD access
> step (§4c)"*. That probe explicitly refuses form submission / account creation / Terms acceptance
> (`evidence/access_probes_2026-07-26.json` → `hard_constraint`), so the three HTTP-200 probes there
> read the licence and README **only** — they are **not** an access grant.
> ⚠️ §4d flags one unresolved lead a PI could still take: a **second official ZOD channel** that may
> need no human application (evidence class **PUBLISHED, not MEASURED**). Until someone probes it, the
> escalation stands.
> Swept by the 2026-08-16 stale-blocker sweep.

## Risk & rollback

- **Not blocked by geometry** (unlike PandaSet). The remaining unknowns are BYTE-LEVEL,
  pinned on real bytes by the job card, not blockers to integration: (a) the exact ZOD
  Sequences/Drives frame layout + camera↔OxTS 100→10 Hz timestamp alignment (the paper
  says 10 Hz front cam; the SDK README says "3 consecutive frames @30 Hz" per sequence —
  decode + oxts IO are INJECTABLE so either resolves without touching the math, the
  Cosmos/PandaSet precedent); (b) the real per-drive KB coeffs (drop in unchanged via
  `kb_to_ftheta`); (c) ZOD `steering angle` units/ratio (primary steer is OxTS-derived,
  ratio-free; CAN steer is a cross-check via `can_steer_ratio`).
- **Representative-calibration honesty (P8):** `ZOD_FRONT_REPR` is grounded on the
  PUBLISHED FOV/resolution, not guessed, and the falsifier verdict is shown robust to
  the real KB — but it is NOT the real per-drive calibration. Integrate as a
  ready-loader; the real-bytes `verify_real_clip` numbers land via the job card when
  access is granted. The unit-tested pure code (geometry, signals, contract) is final.
- **License firewall:** CC-BY-SA is COPYLEFT. `LICENSE="CC-BY-SA-4.0"`,
  `DATA_TAG="data:zod"` → the orchestrator/lake must keep ZOD in a SEPARATE shard
  (`license_guard` ShareAlike firewall) and never merge it into a permissive/proprietary
  `tanitad-own-core` file. Privacy: ZOD is de-identified; its notice must travel.
- Blast radius if integrated: additive — one new `stack/tanitad/data/zod.py` (+ its test);
  imports existing `calib` / `cosmos_drive` / `comma2k19` / `_contract` / `mixing`; no
  change to any existing module. Rollback: delete the module + test; nothing depends on it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
