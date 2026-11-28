# MM check of the paired tool (2026-10-25)

Independent re-derivation of one strategic cell (S's 50 m checkpoint position error, X = M2b-mixo_s0 minus B = M2b_s0) with a hand-written
episode-cluster bootstrap (`check_pairfam.py` beside this file, 2,000 resamples, its own RNG), outside `taniteval.ci` and outside the tool:
n 2,319, 137 episodes, delta +0.0245 m (X 8.0368, B 8.0122), 95 % [-0.1468, +0.2039]. The tool: +0.0245 [-0.1547, +0.2015], n 2,319 / 137.
Point and n identical; interval endpoints agree to resampling noise. `pytest stack/tests/test_v8f_paired_families.py`: 24 passed in 35.37s.
