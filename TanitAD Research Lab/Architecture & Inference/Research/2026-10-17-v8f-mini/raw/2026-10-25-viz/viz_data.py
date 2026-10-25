"""viz_data.py -- per-cache-step model readout of the 4 selected clips (CPU), for the video and the key-frame figure.

For every cache step t = 3 .. T-31 (NOW; 0.2 s apart; every step has a GT 6 s future) of a clip it runs the hooked forward (exact mode) of one seed
with the PHYSICAL TICK SCHEDULE (sample-and-hold, declared):
    S ticks every 5 cache steps (1 Hz):      a_s = t mod 5            (S's tick frame = t - a_s)
    T ticks every 2.5 steps (2 Hz):          tick steps floor(2.5 m); a_t = t - (last T tick <= t)   in {0,1,2}
    O runs at every step (NOW).
(the banked eval cycles the ages  a_t = i % 3,  a_s = a_t + (i // 3) % 5  per window instead; both lie inside the trained range 0..2 / 0..6;
 `item()` clamps a tick to a frame that has `hist` real frames behind it.)  Also runs `forward_diag` with the GT code ("T0 ORACLE, diagnostic only")
where the tick row has a v9 label.  Everything is T0 (open loop, teacher-forced on the true past; no action feedback).

  python viz_data.py --seed M2b_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_extract import *  # noqa: E402,F401,F403

VIZ_CACHE = f"{WORK}/cache_viz"
VIZ_DIR = f"{WORK}/viz"


def tick_ages(t: int):
    ticks = sorted({int(np.floor(2.5 * m)) for m in range(0, 80)})
    return t - max(x for x in ticks if x <= t), t % 5


def run_clip(model, ew: EvalWorld, clip: str, seed: str):
    ci = ew.clips.index(clip)
    T = len(ew.store.get(clip)["pose"])
    ts = [t for t in range(3, T - 30) if ew.store.get(clip)["gt_ok"][t]]
    rows = []
    for t in ts:
        a_t, a_s = tick_ages(t)
        it = ew.ds.item(ci, t, a_t, a_s)
        b = D.collate([it])
        out, att = forward_tapped(model, b)
        has = bool(b["lab_t"]["has"][0])
        code_gt = model._code_gt(b["lab_t"]).to(out["code"].dtype)
        out_or = forward_plain(model, b, code_override=torch.where(b["lab_t"]["has"][:, None], code_gt, out["code"])) if has else None
        r = {"t": t, "a_t": int(b["a_t"][0]), "a_s": int(b["a_s"][0]), "has_label": has,
             "gt": b["gt"][0].numpy(), "traj30": out["traj30"][0].numpy(), "traj30_oracle": (out_or["traj30"][0].numpy() if has else None),
             "ck_now": M.transform_ckpt(out["ckpt"], b["pose_s"], b["pose_now"])[0].numpy(),
             "goal_prob": out["goal_prob"][0].numpy(), "lat_p": torch.softmax(out["lat_logits"], -1)[0].numpy(),
             "lon_p": torch.softmax(out["lon_logits"], -1)[0].numpy(), "cons_pred": out["cons_pred"][0].numpy(),
             "v": float(it["kin3"][0]), "limit": b["limit"][0].numpy(), "limit_ok": bool(b["limit_ok"][0]),
             "nav": b["nav"][0, -1].numpy()}
        L = b["lab_t"]
        r["gt_lat_cls"], r["gt_lon_cls"] = (int(L["lat_cls"][0]), int(L["lon_cls"][0])) if has else (-1, -1)
        r["gt_cons"] = L["cons"][0].numpy() if has else np.full(M.N_CONS, np.nan, np.float32)
        r["gt_goal_y"] = L["goal_y"][0].numpy() if has else np.zeros(M.N_GOAL, np.float32)
        r["gt_goal_w"] = L["goal_w"][0].numpy() if has else np.zeros(M.N_GOAL, np.float32)
        r["ade6"] = float(ade6_of(r["traj30"][None], r["gt"][None])[0])
        r["ade6_oracle"] = float(ade6_of(r["traj30_oracle"][None], r["gt"][None])[0]) if has else float("nan")
        for lv in LEVELS:
            m = level_maps(att, lv)
            r[f"cond_{lv}"], r[f"scene_{lv}"] = m["cond"][0].numpy(), m["scene"][0].numpy()          # [nc], [4, 40]
            r[f"layers_{lv}"] = np.stack([split_weights(att[k], lv)["scene"][0].numpy() for k in sorted(k for k in att if k[0] == lv)])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="M2b_s0")
    ap.add_argument("--clips", default=None)
    a = ap.parse_args()
    sel = json.load(open(f"{OUT}/selection.json"))
    clips = a.clips.split(",") if a.clips else [v["clip"] for k, v in sel.items() if not k.startswith("_")]
    model, cfg = load_model(a.seed)
    ew = EvalWorld(cfg, lru=4, cache=VIZ_CACHE)
    os.makedirs(VIZ_DIR, exist_ok=True)
    for clip in clips:
        t0 = time.time()
        rows = run_clip(model, ew, clip, a.seed)
        keys = rows[0].keys()
        pack = {}
        for k in keys:
            vals = [r[k] for r in rows]
            if vals[0] is None:
                continue
            if k == "traj30_oracle":
                vals = [v if v is not None else np.full((30, 2), np.nan, np.float32) for v in vals]
            pack[k] = np.array(vals)
        np.savez_compressed(f"{VIZ_DIR}/{a.seed}__{clip}.npz", **pack)
        print(clip, len(rows), "steps", f"{time.time() - t0:.0f}s", "ade6 mean %.2f" % pack["ade6"].mean(), flush=True)


if __name__ == "__main__":
    main()
