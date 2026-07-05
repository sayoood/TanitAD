# TanitAD

**Mission: build the AD stack that beats the best — hierarchical latent world models (4B architecture),
orders of magnitude less data, inference-efficient on Orin/Thor-class hardware, inherently safe and
aligned with the 2026 UN ADS regulation.**

> Constitution: [`Project Steering/Mission Plan.md`](Project%20Steering/Mission%20Plan.md) — owned by
> Sayed, agents never edit it. Final evaluation: 2026-10-05.

## Start here (any session, human or agent)

1. [`PROJECT_STATE.md`](PROJECT_STATE.md) — current truth: phase, focus, next actions, session log.
2. [`DECISIONS.md`](DECISIONS.md) — why things are the way they are (ADR log).
3. [`Project Steering/CONTINUATION_PROTOCOL.md`](Project%20Steering/CONTINUATION_PROTOCOL.md) — the
   session rituals that make this project resumable at any point.

## Map

| Folder | Content |
|---|---|
| `Project Steering/` | Constitution, Master Plan, Phase 0 Plan, continuation protocol, progress reports |
| `stack/` | The runnable implementation: `tanitad` Python package (4B world model, instruments, training) |
| `TanitAD Research Hub/` | Research baseline, hypothesis ledger, weekly agent definitions, per-discipline knowledge bases |
| `Data engineering/`, `Architecture/`, `Benchmarks & Eval/`, `Tools&DevEnv/` | Discipline work folders |
| `Ressources/` | UN ADS regulation, Deep Think analyses, reference designs |

## The edge, in one table

| Axis | Incumbents | TanitAD target |
|---|---|---|
| Params | 0.3–10 B | **10–100 M** |
| Training data | 1000s h + labels / internet-scale pretraining | **tens of hours, zero perception labels** |
| Planning | pixels/diffusion/CEM (seconds–minutes) | **latent imagine-and-select (milliseconds)** |
| Hierarchy | flat or 2-level | **strategic / tactical / operative / fallback (4B)** |
| Self-knowledge | none | **imagination-error monitoring, regulation-ready (ISMR/DSSAD)** |

Research grounding: [`TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md`](TanitAD%20Research%20Hub/INITIAL_RESEARCH_SYNTHESIS.md).

## Quick start (dev machine)

```powershell
C:\Users\Admin\venvs\tanitad\Scripts\Activate.ps1
cd stack
pip install -e .[dev]
pytest
python -m tanitad.train.train_worldmodel --smoke
```
