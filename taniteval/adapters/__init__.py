"""Dataset adapters for TanitEval.

The whole scoring stack below ``taniteval/tools/ff_rescore.py:150 load_dump`` is
**dataset-agnostic already** — it consumes a plain dict of tensors and nothing
under ``four_families.py``, ``ci.py``, ``lead_metrics.py`` or ``selgap.py`` knows
what PhysicalAI is (audit section C.5). An adapter's whole job is therefore to
produce the ``win`` dict, and to REFUSE — loudly, with a reason and an ``n`` —
every family the source dataset cannot supply.

Modules
-------
``navsim``  NavSim v1 (PDMS) / v2 (EPDMS) scenes -> ``win``.
            ⛔ Reads ``score``, never ``pdm_score``.
            ⛔ Emits no confidence interval: NavSim's cluster unit is unsettled.
"""
__all__ = ["navsim"]
