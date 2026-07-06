# TanitAD — PROJECT STATE

> **This is the single entry point for every new working session (human or agent).**
> Read this file first. It tells you where the project stands, what is decided, and what to do next.
> Update it at the end of every working session (see `Project Steering/CONTINUATION_PROTOCOL.md`).

- **Last update:** 2026-07-13 (session: Tools&DevEnv weekly agent — MetaDrive front-camera RGB sim arm)
- **Current phase:** Phase 0 — foundation & first edge proofs
- **Constitution:** `Project Steering/Mission Plan.md` (owned by Sayed, never edited by agents)
- **Final evaluation date (P7):** 2026-10-05

---

## 1. Where we are (one paragraph)

Project kicked off 2026-07-05. Prior-experiment repos (ALPS-4B, 4B-HRM, ACRE, RSRA-4B) analyzed; initial
deep research across H0–H15 done and documented; master plan and Phase 0 plan written; the `stack/`
Python package scaffolded with the 4B world-model skeleton, instrument checks (I1–I4), and a smoke test
that runs on the RTX 4060; research hub concept written and weekly agents defined; repo pushed to
https://github.com/sayoood/TanitAD.

## 2. Key documents (read in this order)

| Priority | Document | Purpose |
|---|---|---|
| 1 | `Project Steering/Mission Plan.md` | Constitution: vision, goals, hypotheses H0–H15, principles P1–P8 |
| 2 | `PROJECT_STATE.md` (this file) | Current truth |
| 3 | `DECISIONS.md` | Decision log (ADR style) — what is decided and why |
| 4 | `Project Steering/Master Plan.md` | Refined execution plan, phases 0/1/2, gates |
| 5 | `Project Steering/Phase 0 Plan.md` | Detailed current-phase plan, week by week |
| 6 | `TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` | Research baseline across all hypotheses |
| 7 | `Project Steering/CONTINUATION_PROTOCOL.md` | How to resume/continue work across sessions |
| 8 | `stack/README.md` | The implementation: how to run, test, train |

## 3. Current focus (Phase 0)

The active week-by-week schedule lives in `Project Steering/Phase 0 Plan.md` §6.
Summary of the immediate next actions:

- [x] W1: smoke training runs on RTX 4060 (`stack/experiments/p0-s000-kickoff-smoke`: loss 3.34→1.89,
      I2 pass 3.6e-7, I4=6.17 — untrained baseline, correctly blocking predictive claims)
- [x] W1: MetaDrive wrapper (WP2) — contract-identical adapter merged + CI-green (`stack/tanitad/data/metadrive_env.py`);
      live rollout needs a supervised source-install of MetaDrive (PyPI pkg no-go on py3.13; see Tools&DevEnv STATE)
- [x] W1: I1–I4 instrument checks implemented + in test suite (10/10 tests pass)
- [x] W1: D-008 executed — TanitAD-4B-M at **261.1 M params** instantiated (budget enforced by test);
      H15 ImaginationField (advection + refine + epistemic σ) wired into training; D9 gate defined;
      exact Phase-0 data spec in Phase 0 Plan §2.2; 24 tests green + 1 sim-skip
- [x] W1: D-009 executed — real camera data first: comma2k19 loader (`stack/tanitad/data/comma2k19.py`,
      HF mirror, route-level splits, real CAN actions), `base250cam` config, Windows-safe extractor;
      Chunk_1 (8.7 GB) downloading locally; 36 tests green
- [x] W1: comma2k19 loader (D-009) **real-data-validated** (DataEng agent): `av` decodes real HEVC →
      [200,3,256,256]@~105 fps, A8 real≈0.053; contract reconciled into shared `_contract` (channels=1|6|None);
      comma2k19 exported + `test_comma2k19_contract.py`; data card + H7 note; 40 tests green
- [ ] W1: **Sayed**: start the A40 pod per `stack/RUNPOD_RUNBOOK.md` (run p0-sB01, comma2k19, planned $25)
- [ ] W1: first local real-camera run on 4060 once Chunk_1 lands (p0-sB00 pipe-proof)
- [x] W2: D-010 **sim arm unblocked** (Tools&DevEnv): MetaDrive front-camera RGB path renders 6ch/256
      2-frame stacks (comma2k19-identical) + perturbation policy + occluder/blocked-route scenarios —
      intake pkg `Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`
      (17 tests, 0 new deps), pending orchestrator triage. The merged 1ch BEV adapter is (correctly)
      rejected by `MixedWindowDataset` — proven by test.
- [ ] W2: **Sayed/supervised** MetaDrive source-install (~10 min) — last blocker for LIVE sim episodes
      (D5/D6 closed-loop) + the A8/BGR render check (Tools&DevEnv STATE handoff)
- [ ] W2: Stage-0 bake-off (residual+change-weighted vs MSE; grid readout vs pooling; probe_imag vs probe_real)
- [ ] W2–3: D1–D3 gates measured (see Phase 0 Plan §4)

## 4. Open questions / blocked items

- ~~NVIDIA PhysicalAI-AV license review~~ **RESOLVED 2026-07-07 (DataEng):** real AV sets = NVIDIA
  internal-dev-only/confidential/12-mo → **excluded from public claims**; comma2k19 (MIT) is the public
  corpus; Cosmos-Drive-Dreams (CC-BY-4.0) is the publicly-safe AV asset. Internal use needs Sayed+legal
  sign-off. See `TanitAD Research Hub/Data Engineering/Research/2026-07-07-physicalai-av-license-review.md`.
- RunPod budget approval per training stage (owner: Sayed; default budget in Master Plan §7)

## 5. Session log (newest first, keep last ~15 entries)

| Date | Session | What happened | Artifacts |
|---|---|---|---|
| 2026-07-13 | Tools&DevEnv agent | D-010 sim arm unblocked: MetaDrive front-camera RGB path (6ch/256 2-frame stacks, comma2k19-identical geometry/alignment) + scripted perturbation policy (off-expert coverage) + occluder(H15/D9)/blocked-route(D5/D6) scenario configs; proved the old 1ch BEV adapter is rejected by `MixedWindowDataset`. Intake pkg (17 tests, 0 new deps, import 1.38 s); live rollout still gated on supervised MetaDrive source-install. KB: Alpamayo 2 Super (32B) + OmniDreams = Phase-1 watch, reinforce P5/C2 | `.../Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`, `.../Tools&DevEnv/Research/2026-07-13-*.md` |
| 2026-07-07 | Data Engineering agent | comma2k19 (D-009) real-data-validated: `av` decodes real HEVC @~105 fps, A8 real≈0.053 → change-weighting justified; forked contract reconciled into shared `_contract` (channels param) + `test_comma2k19_contract.py`; comma2k19 exported; Windows `\|` confirmed (handled by `extract_comma2k19.py`); H7 deltas (LAOF, Sensorimotor-WM); data card + note | `.../Data Engineering/Research/2026-07-07-*.md`, `stack/tanitad/data/_contract.py`, `stack/tests/test_comma2k19_contract.py` |
| 2026-07-06 | D-008 scale-up | Model scaled to 261 M (measured per-component budget in Phase 0 Plan §2.1); H15 imagination in Phase 0 (module + losses + D9); exact data spec; RunPod runbook + ledger row; local 261 M pipe-check run | `stack/tanitad/models/imagination.py`, `stack/RUNPOD_RUNBOOK.md`, DECISIONS D-008 |
| 2026-07-06 | Tools&DevEnv agent | WP2 MetaDrive→toy-contract wrapper (17✓/1skip); MetaDrive install verdict (PyPI no-go py3.13, source is GO); AlpaSim/AlpaGym = Phase-1 cloud (40–60 GB VRAM); Rerun.io picked for viz | `stack/tanitad/data/metadrive_env.py`, `TanitAD Research Hub/Tools&DevEnv/Research/2026-07-06-*.md` |
| 2026-07-05 | Kickoff | Repo analysis, initial research, plans, stack scaffold, hub setup, first push | see §2 table |
