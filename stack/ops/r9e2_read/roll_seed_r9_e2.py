"""roll_seed_r9_e2.py -- the battery's ONE-seed roll child (roll_seed_r8_pf.py) with the refcv9 staging proxy installed.  NEW file (E2-ROLL, 2026-11-04).

``run_battery_r9_e2.py`` swaps this in for ``roll_seed_r8_pf.py`` (same argv).  Before ``roll_seed_r8_pf`` is imported the env pins are checked, ``r8_repoint``
is pointed at R9E2_REMAP, and a post-import hook rebinds ``r7_roll.Refcv6Proxy`` (see ``run_battery_r9_e2.install_staging``).  After the roll the record
(``--out-json``) gains an ``r9e2_stage`` census; exit 3 if NO forward was staged.  UNVERIFIED end to end (never run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_battery_r9_e2 as R9  # noqa: E402


def main() -> int:
    pins = R9.require_env()
    R9.pin_repoint()
    R9.child_stage_hook()
    sys.path.insert(0, pins["R9E2_PF_CODE"])
    import roll_seed_r8_pf as PF
    rc = PF.main()
    a = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    st = getattr(sys.modules.get("r7_roll"), "R9E2_STAGER", None)
    if st is None:
        raise SystemExit("[r9e2-child] the staging proxy was never installed (r7_roll not imported through the hook)")
    rec = json.load(open(a["--out-json"], encoding="utf-8"))
    rec["r9e2_stage"] = st.census()
    json.dump(rec, open(a["--out-json"], "w", encoding="utf-8"), indent=1, default=str)
    print(f"[r9e2-child] staged {st.n_forwards} forwards {st.census()['n_stage_calls']} built={st.census()['mechanisms_built']}", flush=True)
    return rc if rc else (0 if st.n_forwards else 3)


if __name__ == "__main__":
    sys.exit(main())
