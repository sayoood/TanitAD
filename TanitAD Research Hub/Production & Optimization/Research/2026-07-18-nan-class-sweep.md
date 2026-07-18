# NaN-class sweep (fleet directive 2026-07-17 idea #6; run 2026-07-18)

> Generalize the logvar-clamp fix: audit every unbounded `exp`/`log`/`sqrt`/`div`
> on a *learned* (gradient-carrying) output. Local, RTX-4060/CPU, $0.

## Method
Two grep passes over `tanitad/models`, `tanitad/train`, `tanitad/refs` for:
`exp(` / `log(` / `sqrt` / `rsqrt` / `pow(-` / division by `.norm()/.sum()/.std()`
without a `clamp`/`eps`, plus `acos/asin` domain and `atan2` origin. Every hit
inspected for (a) gradient flow and (b) can the argument reach the singular point.

## Findings

| site | pattern | verdict |
|---|---|---|
| `models/kinematic.py:43` | `(a_lon²+a_lat²).sqrt()` → `relu` | **REAL (fixed)** — sqrt'(0)=inf × relu'=0 = **0·inf=NaN** at zero total accel (a coasting/feasible control). The classic sqrt-relu trap. |
| `models/imagination.py:137` | `exp(-logvar)` | already fixed (logvar clamp [-10,10], 2026-07-17) |
| `models/sigreg.py:39,40` | `exp(-0.5·b²·x²)` | SAFE — argument ≤ 0, so `exp ∈ (0,1]`, cannot overflow |
| `models/fourbrain.py:298` | `exp(-(p·p.log()).sum())` (erank) | SAFE — `p` is `.clamp_min(1e-12)` before `log` |
| `train/train_worldmodel.py:369` | same erank | SAFE — same `.clamp_min(1e-12)` |
| `refs/refb.py:369` | `d2.sqrt()` (Mahalanobis OOD) | SAFE — `z.detach()`, no gradient through the sqrt |
| `data/physicalai.py:190` | `asin(...)` | SAFE — argument `max(-1,min(1,·))` clamped to domain |

## Fix
`kamm_circle_violation` (H14 friction-circle grounding): `.clamp_min(1e-12)` inside
the sqrt. Behaviour-preserving in the feasible range (forward value unchanged);
bounds the backward magnitude so a near-zero-accel control never NaNs the grad.
Witness test `tests/test_kinematic_nan.py` (4 tests): reproduces the NaN on the
unclamped path, pins finite grad at/around zero accel + forward invariance.
Note: H14 friction loss is defined but **not yet wired into `flagship_loss`** —
this was a *latent* trap that would have fired the day H14 is activated.

## Verdict
The codebase was already well-guarded (5 of 7 candidate sites safe by prior
clamps; the logvar site fixed 2026-07-17). One genuine latent trap found + fixed.
The silent-run-death (F-5/6/7) class for these operators is now closed with
witness coverage. Cost: 4060/CPU, ~20 min, $0.
