# MRATE build, banked 2026-11-14 (not integrated)

Banked by the Master Mind so that the MRATE implementation is not held on one disk only. The working tree `motion_c/theory/mrate/tree/` on E: is a `git archive` of `1c0a348c` (stack, taniteval) plus the five files under `tree/stack/` here; `*.orig_tip` are the three stack files as they were at `1c0a348c`, and `patch_*.py` reproduce each diff on a fresh tip (`HANDOFF_MRATE.md`).

Status: built and tested on 2026-11-07, NOT landed into `stack/`, NOT run. MRATE on the 0.2/0.4/0.8 s ladder is not launched: the 2026-11-08 horizon ruling rejects that ladder, so MRATE needs a new card on the current ladder before any run. These files sit outside `stack/` on purpose; nothing imports them and no test collects them.
