"""dump_heads_r8.py -- bank refcv8's OWN map-head / box-head / tactical / fan outputs per eval window (for the video's MAP and BEV-head panels).

WHY: the banked battery dumps carry only trajectories, tactical logits, the selected index and the nav / label context.  The 100 x 60 m MAP-HIRES logits and
the BOX3D slots (presence / class / box / yaw / rates) were never written, so the videos show GT only and say "pending GPU dump".  This script re-runs the
SAME battery roll (identical loader, windows, proxy inputs, pose clock) and captures the extra outputs of the MAIN nav_true forward.  It does NOT change what
the battery computes: the proxy's call is wrapped, not replaced, and the capture happens after the original call returns.

    # smoke (CPU, 1 window, no GPU):
    python dump_heads_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --out-dir E:/refcv8_eval_kit/videos/heads_smoke --episodes 114 --max-windows 1 --device cpu
    # GPU (the real run; the dev-box GPU lock MUST be held by the caller -- this script never takes or touches it):
    python dump_heads_r8.py --ckpt <ckpt_ema_48852.pt> --config E:/refcv8_eval_kit/ckpt/config.json --out-dir <dir> --episodes all --device cuda

OUTPUT   <out-dir>/heads_ep###.npz   (episode = SHORT clip index; window axis sorted by ws; arrays aligned to the battery's windows)
    ws [n]                      window origin row                      v0 [n]  fed speed (control: must equal the banked dump's v0)
    map_class [n, X, Y] uint8   decided class code 0..7 under the model's own rule (prior_corrected, class weights from --class-weights); X,Y = out_hw (400 x 240
                                at 0.25 m: x 0..100 m ahead, y +30..-30 m: row 0 = x 0, col 0 = y -30 m i.e. RIGHT -- the SAME grid as the GT cart_frac / 5)
    map_conf  [n, X, Y] uint8   decided-class posterior x 255          map_valid [n, X, Y] bool  lift_valid (cell seen by the camera lift)
    map_logits [n, 8, X, Y] f16 ONLY with --save-logits (6 MB / window)
    slot_<k> [n, Nq, ...]       every tensor of out['agent_slots'] (presence_logit, cls_logits, box, yaw..., box3d [x y z l w h yaw] when the 3-D head is on)
    out_<k>  [n, ...]           every other per-sample output tensor <= 150k elements of the main forward (tacv6_*_logits, lat/lon_logits_tac, traj, sel_idx,
                                anchor_traj = THE FAN, anchor_logits, and whichever allocation tensors the model emits)
    keys.json                   what was captured, skipped (too large) and absent -- so the FIRST GPU run settles whether the allocation is in `out`.

⛔ UNVERIFIED until the smoke has run: that the captured row is the nav_true row (asserted below via conds_fed), and the allocation keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BATT_PKG = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery")
KIT8 = Path("E:/refcv8_eval_kit")
WINDOWS_S2 = KIT8 / "windows/s2_windows.json"
CLASS_WEIGHTS = "E:/refcv6_eval_kit/data/refcv7/map_hires_class_weights_train_100x30.json"
BATTERY_SUMMARY = "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/battery_summary.json"
MAX_ELEMS_PER_SAMPLE = 150_000
WS_OFFSET = 7                       # banked ws = dataset window start t + (w - 1), w = 8 (MEASURED: smoke t=2 <-> banked ws 9, v0 equal to 1e-6)
BANKED_TREE = Path("C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc")

sys.path.insert(0, str(BATT_PKG / "code"))
os.environ.setdefault("R8_NO_POSE_SYNC", "1")      # the standalone pose clock == the banked pass-1 `unc` trees (the videos' dumps)
import r8_battery_patch as BP  # noqa: E402

BP.apply_child()                                    # loader re-pointing + (optional) pose-sync, exactly as roll_seed_r8.py does
import torch  # noqa: E402
import r7_roll as RR  # noqa: E402  (the refcv7 battery's roll: proxy + patches; on sys.path via r8_repoint)
import run_battery_r7 as RB  # noqa: E402

CAP: list = []            # one dict per window, in roll order
KEYS = {"captured": {}, "skipped_large": {}, "agent_slots": {}}


def _to_np(t, dtype=None):
    a = t.detach().float().cpu().numpy() if t.is_floating_point() else t.detach().cpu().numpy()
    return a.astype(dtype) if dtype is not None else a


def _walk(prefix, d, row, rec, skip=()):
    """tensors of a (possibly nested) output dict with a batch row -> rec[prefix + key]; large ones are listed in KEYS['skipped_large']."""
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            _walk(f"{prefix}{k}_", v, row, rec, skip)
        elif torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] > row:
            n = int(v[row].numel())
            if n <= MAX_ELEMS_PER_SAMPLE:
                rec[f"{prefix}{k}"] = _to_np(v[row])
                KEYS["captured"][f"{prefix}{k}"] = list(v[row].shape)
            else:
                KEYS["skipped_large"][f"{prefix}{k}"] = list(v[row].shape)


def capture(out: dict, row: int, key, class_w, save_logits: bool) -> dict:
    """key = ctx.index = (clip index, window START row t); the banked `ws` is t + 7 (pose_last = poses[t + w - 1], w = 8) -- controlled against the banked v0 in main()."""
    rec = {"key": tuple(int(x) for x in key)}
    perc = out.get("perception") if isinstance(out.get("perception"), dict) else {}
    # (1) map head: the 10 cm logits ride out on out['perception'] (refc_v3.py:2192), NOT at the top level
    mh_logits = perc.get("map_hires_logits", out.get("map_hires_logits"))
    mh_valid = perc.get("map_hires_lift_valid", out.get("map_hires_lift_valid"))
    if mh_logits is not None:
        from tanitad.models import map_head_hires as MH
        lg = mh_logits[row:row + 1].float()
        code = MH.decide(lg, "prior_corrected", class_w)                       # [1, X, Y] -- the model's own decision rule
        s = lg - torch.log(class_w.to(lg.device, torch.float32)).view(1, -1, 1, 1)
        p = torch.softmax(s, dim=1).gather(1, code[:, None]).squeeze(1)        # posterior of the decided class under the corrected scores
        rec["map_class"] = code[0].to(torch.uint8).cpu().numpy()
        rec["map_conf"] = (p[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        if mh_valid is not None:
            rec["map_valid"] = mh_valid[row].detach().cpu().numpy().astype(bool)
        if save_logits:
            rec["map_logits"] = lg[0].half().cpu().numpy()
        KEYS["captured"]["perception.map_hires_logits"] = list(lg.shape[1:])
    # (2) every other tensor under out['perception'] (the perception branch's own agent / box outputs, if any)
    _walk("perc_", perc, row, rec, skip=("map_hires_logits", "map_hires_lift_valid", "map_hires_bev"))
    # (3) the agent-slot decoder
    sl = out.get("agent_slots")
    if isinstance(sl, dict):
        for k, v in sl.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] > row:
                rec[f"slot_{k}"] = _to_np(v[row])
                KEYS["agent_slots"][k] = list(v[row].shape)
    # (4) everything else per-sample and small: tactical logits, the FAN (anchor_traj), selection, the allocation (r8_*)
    _walk("out_", {k: v for k, v in out.items() if k not in ("agent_slots", "perception", "map_hires_logits", "map_hires_lift_valid", "map_hires_bev")}, row, rec)
    return rec


def install_capture(class_w, save_logits):
    orig = RR.Refcv6Proxy.__call__

    def wrapped(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        out = orig(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)
        if nav_cmd is not None:                                                # the main batched call: one row per nav condition
            conds = list(self.__dict__["conds_fed"])
            assert "nav_true" in conds, f"conds_fed {conds} has no nav_true row"
            row = conds.index("nav_true")
            r = capture(out, row, self.ctx.index, class_w, save_logits)
            r["v0"] = float(v0.reshape(-1)[row]) if v0 is not None else float("nan")
            CAP.append(r)
        return out
    RR.Refcv6Proxy.__call__ = wrapped


def sha12(cid):
    return hashlib.sha256(cid.encode()).hexdigest()[:12]


def restrict_windows(episodes, max_windows):
    by = json.load(open(BATTERY_SUMMARY, encoding="utf-8"))["stages"]["roll_s0"]["extras"]["clip_sha12_by_index"]
    inv = {v: int(k) for k, v in by.items()}
    wins = json.load(open(WINDOWS_S2, encoding="utf-8"))
    out = {}
    for cid, ws in wins.items():
        ci = inv[sha12(cid)]
        if episodes is None or ci in episodes:
            out[cid] = list(ws)[:max_windows] if max_windows else list(ws)
    return out, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episodes", default="all", help="'all' or comma-separated SHORT clip indices")
    ap.add_argument("--max-windows", type=int, default=0, help="first N windows per episode (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-logits", action="store_true")
    ap.add_argument("--class-weights", default=CLASS_WEIGHTS)
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true", help="CPU: trunk bf16/NHWC off (the battery's own CPU-smoke switch)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    eps = None if a.episodes == "all" else {int(x) for x in a.episodes.split(",")}
    wins, inv = restrict_windows(eps, a.max_windows)
    wjson = out_dir / "_windows_subset.json"          # holds clip UUIDs: a WORK file, never bank it
    json.dump(wins, open(wjson, "w", encoding="utf-8"))
    print(f"[dump_heads] {len(wins)} episodes, {sum(len(v) for v in wins.values())} windows, device {a.device}", flush=True)

    from tanitad.models import map_head_hires as MH
    cw = MH.load_class_weights(a.class_weights)
    class_w = cw[0] if isinstance(cw, tuple) else cw
    class_w = torch.as_tensor(class_w, dtype=torch.float32).reshape(-1)
    assert class_w.numel() == 8, f"class weights {tuple(class_w.shape)} != 8"
    install_capture(class_w, a.save_logits)

    if a.smoke_cpu_fp32_trunk or a.device == "cpu":
        from tanitad.eval import refcv7_loader as L
        _orig = L.build_model

        def _fp32(*aa, **kk):
            m, c, ar, r = _orig(*aa, **kk)
            for mod in m.modules():
                lv = getattr(mod, "memory_levers", None)
                if isinstance(lv, dict) and "bf16" in lv:
                    lv["bf16"] = False
                    lv["channels_last"] = False
            return m, c, ar, r
        L.build_model = _fp32

    t0 = time.time()
    rec = RB.roll_one(a.ckpt, a.config, a.seed, str(out_dir / "roll_dump"), wins, None, device=a.device, full=False)
    wall = time.time() - t0
    print(f"[dump_heads] roll done in {wall:.0f}s, captured {len(CAP)} windows", flush=True)

    # ---- group by clip index, sort by ws, write npz
    byc: dict = {}
    for r in CAP:
        byc.setdefault(r["key"][0], []).append(r)
    written = []
    for ci, rows in sorted(byc.items()):
        rows.sort(key=lambda r: r["key"][1])
        d = {"ws": np.array([r["key"][1] + WS_OFFSET for r in rows], np.int64), "v0": np.array([r["v0"] for r in rows], np.float64)}
        for k in rows[0]:
            if k in ("key", "v0"):
                continue
            d[k] = np.stack([r[k] for r in rows])
        p = out_dir / f"heads_ep{ci:03d}.npz"
        np.savez_compressed(p, **d)
        written.append(str(p))
    # ---- control: the captured window IS the banked window (v0 equal) -- a mis-keyed capture would silently paint another window's map on a video
    ctl = {"checked": 0, "max_abs_v0_diff": None, "traj_max_abs_diff_m_vs_banked_nav_true": None, "note": "traj differs by sampler draw / device; informational only"}
    ex_p = BANKED_TREE / "dump_s0/refcv6_extras.npz"
    if ex_p.exists():
        ex = np.load(ex_p)
        key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        dv, dt = [], []
        for ci, rows in byc.items():
            for r in rows:
                i = key.get((ci, r["key"][1] + WS_OFFSET))
                if i is None:
                    raise SystemExit(f"[dump_heads] CONTROL FAILED: window ({ci}, ws {r['key'][1] + WS_OFFSET}) is not in the banked dump")
                dv.append(abs(r["v0"] - float(ex["v0"][i])))
                if "out_traj" in r:
                    dt.append(float(np.abs(r["out_traj"] - ex["nav_true.traj"][i]).max()))
        ctl.update(checked=len(dv), max_abs_v0_diff=max(dv) if dv else None, traj_max_abs_diff_m_vs_banked_nav_true=max(dt) if dt else None)
        if dv and max(dv) > 1e-3:
            raise SystemExit(f"[dump_heads] CONTROL FAILED: captured v0 differs from the banked window's v0 by {max(dv):.4g} (ws offset wrong?)")
    meta = {"tool": "dump_heads_r8.py", "control_vs_banked": ctl, "ckpt": a.ckpt, "ckpt_md5": hashlib.md5(open(a.ckpt, "rb").read()).hexdigest() if os.path.getsize(a.ckpt) < 4 * 2**30 else "skipped",
            "device": a.device, "seed": a.seed, "episodes": sorted(byc), "n_windows": len(CAP), "wall_s": round(wall, 1),
            "keys": KEYS, "class_weights": a.class_weights, "decision_rule": "prior_corrected",
            "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
            "pose_clock": "standalone (R8_NO_POSE_SYNC=1) == the banked pass-1 trees", "written": written,
            "roll_record_wall_s": rec.get("wall_s")}
    (out_dir / "keys.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("episodes", "n_windows", "wall_s", "keys")}, indent=1, default=str))


if __name__ == "__main__":
    main()
