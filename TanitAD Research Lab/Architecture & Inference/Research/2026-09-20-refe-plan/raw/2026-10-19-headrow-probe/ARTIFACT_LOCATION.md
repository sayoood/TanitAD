# Where the HEADROW-PROBE bulk outputs live (not in git)

| artifact | sha256 | location |
|---|---|---|
| `raw/f10/oof_proposals_f10.npz` (86,733,457 B; the per-proposal OOF export used by FASTNEG-CENSUS) | `fc86c4c07444e3b94f7f8d04846410b88988abf6ab10bf3be55ab2f7f8785e8e` | `D:/Projects/TanitAD-artifacts/headrow_probe_2026-10-19/oof_proposals_f10.npz` |
| `run/f10/` control tables (oof, c2, cpos, c2x, c2y, c2z; 147 files, 422 MB) | per-file list `run_f10_SHA256.txt` beside them (sha256 of the list: `c4f4782371f9ac8e236e7e0311cd956b59cbf50ac95ab36586d6a6177eb25eb4`) | `D:/Projects/TanitAD-artifacts/headrow_probe_2026-10-19/run_f10/` |

Copied 2026-10-19 by the Master Mind; both copies were verified by sha256 against the dev-box originals. They are regenerable by `code/` from S0's bank (`srttc_s0`). ⚠️ D: is an EXTERNAL drive; this is a single location.
