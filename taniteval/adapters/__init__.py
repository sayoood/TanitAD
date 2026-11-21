"""Dataset adapters for TanitEval.

The whole scoring stack below ``taniteval/tools/ff_rescore.py:150 load_dump`` is
**dataset-agnostic already** — it consumes a plain dict of tensors and nothing
under ``four_families.py``, ``ci.py``, ``lead_metrics.py`` or ``selgap.py`` knows
what PhysicalAI is (audit section C.5). An adapter's whole job is therefore to
produce the ``win`` dict, and to REFUSE — loudly, with a reason and an ``n`` —
every family the source dataset cannot supply.

Modules
-------
``navsim``     NavSim v1 (PDMS) / v2 (EPDMS) scenes -> ``win`` + the TanitEval artifact.
               ⛔ Reads ``score``, never ``pdm_score``.
``navsim_ci``  the NavSim interval: a LOG-CLUSTER bootstrap (clusters = the OpenScene
               ``log_name``, n >= 8) of the DEVKIT'S OWN aggregate. Pre-registered
               2026-09-19 in ``FlyWheels/TanitAD_EvalFlyWheel/incoming/
               2026-09-19-navsim-estimator-and-route-leak/SPEC.md``; ⛔ scene-token,
               mapping-key and episode-cluster intervals are refused.
"""
__all__ = ["navsim", "navsim_ci"]
