"""D0b (descriptive, post-hoc, NOT pre-registered): the X2 baseline -- how often does refcv7's pick follow the residual
prior's lateral side against GT? Banked eval fans (route package capture), CPU only. Writes raw/d0b_prior_follow.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
                            r"2026-10-04-refcv7-route-following/code")))
import route_metrics as rm  # noqa: E402

out = {"_what": "descriptive post-hoc baseline for X2; prior side = rm.dir_class(terminal heading of prior_path)"}
for tag in ("eval_s0g", "eval_s1", "train_s0"):
    z = np.load(f"D:/refcv7_route_bin/2026-10-04/{tag}.npz", allow_pickle=True)
    cls, th = rm.gt_class(z["gt"], z["gt_valid"])
    tgt = rm.dir_class(th)
    pside = rm.dir_class(rm.terminal_heading(z["prior_path"]))
    W = len(z["sel_idx"])
    pick = rm.dir_class(rm.terminal_heading(z["fan"][np.arange(W), z["sel_idx"]]))
    turn = np.isin(cls, ["turnL", "turnR"])
    strt = cls == "straight"
    dis = turn & (pside != tgt)
    r = {"n_turn": int(turn.sum()),
         "prior_dir_correct_turn": round(float((pside == tgt)[turn].mean()), 4),
         "n_turn_prior_disagrees_gt": int(dis.sum()),
         "pick_follows_prior_when_prior_wrong_turn": round(float((pick == pside)[dis].mean()), 4) if dis.any() else None,
         "pick_correct_when_prior_wrong_turn": round(float((pick == tgt)[dis].mean()), 4) if dis.any() else None,
         "pick_correct_when_prior_right_turn": round(float((pick == tgt)[turn & (pside == tgt)].mean()), 4),
         "wrong_picks_turn": int((turn & (pick != tgt)).sum()),
         "wrong_picks_turn_with_prior_side": int((turn & (pick != tgt) & (pick == pside)).sum()),
         "straight_prior_nonstraight": round(float((pside != 0)[strt].mean()), 4),
         "straight_pick_follows_nonstraight_prior": round(float((pick == pside)[strt & (pside != 0)].mean()), 4)
         if (strt & (pside != 0)).any() else None}
    out[tag] = r
(PKG / "raw" / "d0b_prior_follow.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
