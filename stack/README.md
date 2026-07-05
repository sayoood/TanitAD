# TanitAD `stack/` — the runnable 4B world-model implementation

Phase 0 scaffold of the TanitAD autonomous-driving stack. Design provenance:
`../Project Steering/Phase 0 Plan.md` §2.1 (decided architecture) and the validated
assets A1–A10 from ALPS-4B (`docs/AD_TRANSFER_RESEARCH.md` there).

## Layout

```
tanitad/
  config.py                 all configs (dataclasses); every run serializes its config
  models/
    sigreg.py               LeJEPA SIGReg (Epps–Pulley, sliced) — collapse prevention
    encoder.py              ViT encoder, batch-free norms (I2-safe by construction)
    predictor.py            operative predictor: causal, FiLM action-conditioned,
                            residual + change-weighted + multi-horizon (A4, H5)
    inverse_dynamics.py     (z_t, z_t+1) -> action grounding head (A5, seed of H7 IDM)
    readout.py              spatial grid readout (A7) + frozen ridge probes with
                            imagination calibration (A3)
    kinematic.py            differentiable bicycle rollout + Kamm-circle penalty (H14)
    fourbrain.py            WorldModel + TacticalSelector (imagine-and-select) +
                            StrategicGraph + FallbackMonitor (H1, H11)
  data/toy_driving.py       zero-dep ego-centric BEV driving toy (consequence-dominant, A8)
  instruments/checks.py     I1–I4 instrument doctrine (D-004) — mandatory rows
  train/train_worldmodel.py Stage-A SSL training loop with experiment records
tests/                      pytest suite incl. the I2 CI tripwire and smoke train
```

## Setup (local dev machine, RTX 4060)

```powershell
# one-time venv (kept OFF Google Drive on purpose)
C:\Users\Admin\venvs\tanitad\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run

```powershell
pytest                                             # full test suite (CPU ok)
python -m tanitad.train.train_worldmodel --smoke   # ~1 min sanity train
python -m tanitad.train.train_worldmodel --steps 2000 --episodes 200 --out experiments\p0-sA01-toy
```

Every run writes `config.json`, `metrics.json` (with I2/I4 instrument rows) and
`model.pt` to its experiment folder. A run without instrument rows does not
exist for decision-making (D-004).

## Next (WP2/WP3, see Phase 0 Plan §3)

1. MetaDrive wrapper exposing the same episode contract as `toy_driving.py`
   (`pip install -e .[sim]`), BEV first, then front camera.
2. `diagnose_control`-style gate runner: D1 (probe ADE), D2 (calibrated
   maneuver ranking with I1 oracle row), D3 (imagined-ADE vs oracle-ADE).
3. Bake-offs: change-weighted vs plain MSE vs ego-compensated loss; grid vs
   pooled readout; probe_imag vs probe_real — one lever per run.
