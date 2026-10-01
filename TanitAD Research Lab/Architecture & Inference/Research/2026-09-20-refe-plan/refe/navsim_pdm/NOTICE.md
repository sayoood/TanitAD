# Vendored from autonomousvision/navsim @ 3e8291bfa89ff247231e0227778840cd0a036896 (Apache License 2.0)

Copied VERBATIM from the navsim tree the dev box scores NAVSIM v1 with (the harness W3 / SPEC_NAVTEST use), so the
on-policy labeller can simulate a proposal EXACTLY as `navsim/evaluate/pdm_score.py` does before its drivable-area
check (LQR tracker + kinematic bicycle). The ONLY change: `navsim.planning.simulation.planner.pdm_planner.{simulation,utils}.`
import prefixes rewritten to `navsim_pdm.`. Source sha256 (first 16 hex of the ORIGINAL file):

- pdm_simulator.py  <- navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py  sha256 ae07b533ada10bc3  (3 import paths rewritten)
- batch_lqr.py  <- navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py  sha256 03c4d1b781edfba6  (3 import paths rewritten)
- batch_lqr_utils.py  <- navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr_utils.py  sha256 a140e4132c6ec601  (1 import paths rewritten)
- batch_kinematic_bicycle.py  <- navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py  sha256 253beb3c1e7005a8  (1 import paths rewritten)
- pdm_array_representation.py  <- navsim/planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py  sha256 e5102697bf941d2f  (2 import paths rewritten)
- pdm_enums.py  <- navsim/planning/simulation/planner/pdm_planner/utils/pdm_enums.py  sha256 827e7338b8e55368  (0 import paths rewritten)
- pdm_geometry_utils.py  <- navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py  sha256 b83b8b8e187e31b3  (1 import paths rewritten)
- pdm_comfort_metrics.py  <- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py  sha256 b2131da3b66d217b  (1 import path rewritten; added 2026-09-26 for NAVSIM-faithful comfort labels, PI decision option 2)

The full license text is in LICENSE (unchanged).
