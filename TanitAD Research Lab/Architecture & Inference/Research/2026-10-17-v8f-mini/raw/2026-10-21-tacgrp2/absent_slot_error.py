"""Post-hoc diagnostic (2026-10-21) on PREREG_V8F_TAC Amendment 2's saved codes: how much of T's constraint error sits in ABSENT slots?

  python absent_slot_error.py <eval dir with the Amendment 2 runs> <path to v8f_mini.py>

T's own constraint values come from `all-<g>` (GT everywhere except g); the GT from `all-goal` (every constraint group GT). A slot is
ABSENT where the GT code is exactly 0 (`_code_gt` writes nan_to_num(..., 0)); windows with no v9 row are excluded. Deciding nothing:
it motivated PREREG_V8F_CZ, which pre-registers its own gate (stage 1) before any run."""
import importlib.util
import sys

import numpy as np

E, MP = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("v8f_mini_for_absent", MP)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)
C0 = M.N_GOAL + M.N_LAT + M.N_LON
for c in ("M1_s0", "M1_s1", "M2b_s0", "M2b_s1"):
    gt = np.load(f"{E}/eval_{c}-tacorc-all-goal.npz")
    ok = ~gt["tac_oracle_fallback"]
    G = gt["tac_code"][ok]
    print(f"{c}: {int(ok.sum())} windows with a v9 row")
    for g in ("latcons", "lonspeed", "stoplead"):
        t = np.load(f"{E}/eval_{c}-tacorc-all-{g}.npz")["tac_code"][ok]
        idx = list(M.CODE_GROUPS[g])
        e = t[:, idx] - G[:, idx]
        ab = G[:, idx] == 0.0
        print(f"  {g:9s} absent {ab.mean():.2f} of slots | RMSE present {np.sqrt((e[~ab] ** 2).mean()):.2f} SD, absent "
              f"{np.sqrt((e[ab] ** 2).mean()):.2f} SD | absent share of squared error {(e[ab] ** 2).sum() / (e ** 2).sum():.2f}")
        for k, i in enumerate(idx):
            a = ab[:, k]
            rp = np.sqrt((e[~a, k] ** 2).mean()) if (~a).any() else float("nan")
            ra = np.sqrt((e[a, k] ** 2).mean()) if a.any() else float("nan")
            print(f"      {M.CONS_NAMES[i - C0]:18s} absent {a.mean():.2f}  RMSE present {rp:.2f}  absent {ra:.2f}")
