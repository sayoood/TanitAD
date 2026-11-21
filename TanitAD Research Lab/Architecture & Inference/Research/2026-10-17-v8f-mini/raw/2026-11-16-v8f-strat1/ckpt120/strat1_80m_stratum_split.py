"""strat1_80m_stratum_split.py -- where does STRAT1's S-PARTIAL shortfall at 80 m sit? (read-only re-read of the banked STRAT1 T0
passes; no new model run, no bar; a BESIDE diagnostic for the CKPT120 card's lever ranking).

For each banked arm (T0 at the window's first tick, real / donor / zero nav, `read/out/t0/eval_STRAT1-<arm>_<var>.npz`) it scores
S_CKPT at 80 m with the reader's rule (strategic_cells.ckpt_cell: side from the navtok class, d_text from navtext eval_nav_d,
COMPLETE_M 20, theta 25 / SLIGHT 10, join rc_ok) and splits the scorable windows by whether the TEXT nav token is on at the window
(d <= max(30, 6 v): the side reaches the model) or off (the model sees FOLLOW with a distance). C0's own input is the v9 token,
so its split is by the text token too and is labelled as such.
"""
from __future__ import annotations

import json
import sys

import numpy as np

RD = "F:/TanitAD-scratch/strat1/read"
sys.path.insert(0, RD)
import strategic_cells as SC  # noqa: E402

ARMS = ("C0_s0", "C0_s1", "N1_s0", "N1_s1", "S1_s0", "S1_s1", "C0_rep")
VARS = ("real", "donor", "zero")


def main(out):
    J = np.load(f"{RD}/inputs/join_eval141.npz")
    NT = np.load(f"{RD}/inputs/navtok_table.npz", allow_pickle=True)
    NX = np.load(f"{RD}/inputs/navtext_nav.npz", allow_pickle=True)
    G = json.load(open(f"{RD}/inputs/eval_grid_windows.json"))
    clips = list(J["clip"])
    ci, t = [], []
    for c in G["order"]:
        for tt in G["windows"][c]:
            ci.append(clips.index(c))
            t.append(int(tt))
    ci, t = np.array(ci), np.array(t)
    r = J["off"][ci] + t
    sd, th = SC.side_table(NT["vocab"])
    cls = NT["eval_cls_id"][r].astype(int)
    tok = NT["eval_flag"][r] == 0
    side = np.where(tok, sd[cls], 0).astype(np.int8)
    theta = np.where(tok, th[cls], np.nan)
    d = np.where(side != 0, NX["eval_nav_d"][r].astype(np.float64), np.nan)
    on = (NX["eval_nav"][r, 1] + NX["eval_nav"][r, 2]) > 0
    rc_ok = J["rc_ok"][r].astype(bool)
    rep = {"tool": "strat1_80m_stratum_split.py", "evidence_class": "MEASURED (re-read of the banked STRAT1 T0 passes; beside, no bar)",
           "tier": "T0 at the window's first tick", "arc_m": 80, "arms": {}}
    for arm in ARMS:
        row = {}
        for v in VARS:
            z = np.load(f"{RD}/out/t0/eval_STRAT1-{arm}_{v}.npz", allow_pickle=True)
            if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
                raise RuntimeError(f"{arm} {v}: windows differ from the grid")
            cell = SC.ckpt_cell(np.asarray(z["ckpt"], np.float64)[..., 2], (30, 50, 80), side, theta, d, tok, rc_ok)[:, 2]
            sc = np.isfinite(cell)
            st = {}
            for name, m in (("all", sc), ("text_token_on", sc & on), ("text_token_off", sc & ~on)):
                s = side[m]
                maj = 1 if (s == 1).sum() >= (s == -1).sum() else -1
                st[name] = {"n": int(m.sum()), "n_episodes": int(len(np.unique(ci[m]))), "level": round(float(cell[m].mean()), 4) if m.any() else None,
                            "best_constant": round(float((s == maj).mean()), 4) if m.any() else None,
                            "level_left": round(float(cell[m & (side == 1)].mean()), 4) if (m & (side == 1)).any() else None,
                            "level_right": round(float(cell[m & (side == -1)].mean()), 4) if (m & (side == -1)).any() else None}
            row[v] = st
        rep["arms"][arm] = row
        print(arm, json.dumps({v: {k: (row[v][k]["n"], row[v][k]["level"], row[v][k]["best_constant"]) for k in row[v]} for v in VARS}), flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
