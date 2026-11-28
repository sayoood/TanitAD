"""r8oracle_roll.py -- ONE inference-seed roll of the R8-ORACLE (PREREG_R8_ORACLE.md, sealed): refcv8 EMA-48,852's tactical layer
forced to GT through `RefCV3Model.set_r8_force`, on the SHIP-R8 battery's windows, per-frame nav feed.  T0 (GT at inference).

NEW file (EvalFlyWheel 2026-10-21, r8oracle).  It is a WRAPPER; no shared file is edited or copied:
  * `roll_seed_r8_pf.py` (hold_r8pf, the per-frame battery child) is IMPORTED, which runs its module-level set-up exactly as the
    battery does (overlay sha check, `r8_battery_patch.apply_child()`, the nav-feed overlay into `r7_roll`, `--nav-feed perframe_full`),
    and its `main()` is called with the battery child's own argv (`--os-only`), so the OWN arm IS the battery's roll and writes the
    battery's dump layout.
  * `r7_roll.Refcv6Proxy` (the battery's model proxy) is SUBCLASSED (`OracleProxy`) and re-bound on the module before the roll builds
    it (the same mechanism `force_lon_reroll_r8.py` uses).  `OracleProxy.__call__` first runs the battery's own `__call__` (= OWN, the
    battery's record), then, for each forced arm, re-runs the model on the SAME inputs from the SAME pre-call global RNG state inside
    `torch.random.fork_rng` (common random numbers; the battery's sampler stream continues exactly where OWN left it).
  * A read-only SPY wraps the model instance's `_r8_apply_force` (instance attribute, the bound original is called unchanged) and records
    the posterior argmax and the constraint values the planner actually received: the C4 readback.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6 \
      python -X utf8 r8oracle_roll.py --oracle-out <dir>/oracle_s1.npz -- \
        --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 1 \
        --dump-dir <dir>/dump_s1 --windows-json <battery>/windows_s1.json --device cuda --out-json <dir>/roll_s1.json --os-only
    (everything after `--` is the battery child's argv, passed through unchanged; add --smoke-cpu-fp32-trunk --device cpu for a smoke)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG = ROOT / "PREREG_R8_ORACLE.md"
SEAL = ROOT / "PREREG_R8_ORACLE.sha256"
PF_CODE = Path(os.environ.get("R8O_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
ARMS_ALL = ("GT_CONS", "GT_HYP", "GT_BOTH", "GT_SPEED_ONLY", "OWN_ARGMAX", "GT_HYP_SWAP", "GT_BOTH_SWAP", "SELF_NAN")
AMEND = ROOT / "PREREG_R8_ORACLE.AMENDMENT-1.md"          # adds GT_HYP_SWAP / GT_BOTH_SWAP (beside)
AMEND_SEAL = ROOT / "PREREG_R8_ORACLE.AMENDMENT-1.sha256"
CONTROL_ARMS = ("SELF_NAN",)
CONTROL_EVERY = 10                     # SELF_NAN on clip index % 10 == 0 (prereg: every 10th episode, 14 of 139)
N_SLOT = 8
MUTATION_SELFNAN_MIN = int(os.environ.get("R8O_MUTATE_SELFNAN_MIN", "0") or 0)   # 0 = no mutation (every real run)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_seal() -> str:
    for doc, seal in ((PREREG, SEAL), (AMEND, AMEND_SEAL)):
        want = seal.read_text(encoding="utf-8").split()[0].strip()
        got = sha256_file(doc)
        if len(want) != 64 or len(got) != 64 or got != want:
            raise SystemExit(f"[r8oracle] {doc.name} sha256 {got!r} != sealed {want!r}: refusing to roll")
    return sha256_file(PREREG) + "+" + sha256_file(AMEND)


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[r8oracle] usage: r8oracle_roll.py [oracle opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-out", required=True, help="npz with every arm's per-window record")
    ap.add_argument("--arms", default=",".join(ARMS_ALL))
    ap.add_argument("--save-every", type=int, default=500)
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    arms = tuple(x for x in a.arms.split(",") if x)
    bad = [x for x in arms if x not in ARMS_ALL]
    if bad:
        raise SystemExit(f"[r8oracle] unknown arms {bad}")
    return a, arms, child


A, ARMS, CHILD = parse()
SEAL_SHA = check_seal()
if "--os-only" not in CHILD:
    raise SystemExit("[r8oracle] the oracle rolls the battery's --os-only child (prereg section 2); --os-only missing")
# ---- the battery child, imported: its module-level set-up runs exactly as in the battery ------------------------------------------
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402  (overlay check, apply_child, nav-feed overlay, R.main + --nav-feed; READ-ONLY)
import torch  # noqa: E402

RR = PF.RR
_Base = RR.Refcv6Proxy
STATE = {"r3": None, "r8c": None, "rbl": None, "n_calls": 0, "n_windows": 0, "t0": time.time(), "min_alloc": None,
         "horizons": None, "spy": [], "records": [], "n_navnull": 0}


def _np(x):
    return x.detach().float().cpu().numpy()


def locate_r3(model):
    from tanitad.refs import refc_v3 as V3
    mods = [m for m in model.modules() if isinstance(m, V3.RefCV3Model)] if hasattr(model, "modules") else []
    if len(mods) != 1:
        raise SystemExit(f"[r8oracle] expected exactly one RefCV3Model inside the proxy's model, found {len(mods)}")
    r3 = mods[0]
    if not bool(getattr(r3, "r8_enabled", False)):
        raise SystemExit("[r8oracle] the model has no refcv8 seams")
    from tanitad.refs import refcv8_conditioning as r8c
    import refb_labels as rbl
    STATE.update(r3=r3, r8c=r8c, rbl=rbl, min_alloc=int(r3.cfg.refcv8.alloc_min_each),
                 horizons=tuple(int(h) for h in r3.cfg.core.trajectory.horizons))
    if STATE["min_alloc"] != 2 or int(r3.cfg.refcv8.n_alloc) != 32:
        raise SystemExit(f"[r8oracle] config drifted from the prereg: alloc_min_each {STATE['min_alloc']} n_alloc {r3.cfg.refcv8.n_alloc}")
    orig = r3._r8_apply_force                                  # the BOUND original; called unchanged

    def spy(out, b, dev):
        sw = STATE.pop("swap", None)                           # AMENDMENT 1: one-shot logit-column swap (own argmax <-> GT)
        if sw is not None:
            out = dict(out)                                    # a COPY: the loss cache already holds the raw decoder outputs
            for key, ids, g in (("lat_logits", r8c.LAT3_V7_IDS, sw[0]), ("lon_logits", r8c.LON6_V7_IDS, sw[1])):
                if g is None or int(g) < 0:
                    continue
                x = out[key].clone()
                own = x[:, list(ids)].argmax(-1)
                for r in range(x.shape[0]):
                    ia, ig = int(ids[int(own[r])]), int(ids[int(g)])
                    if ia != ig:
                        tmp = x[r, ia].clone()
                        x[r, ia] = x[r, ig]
                        x[r, ig] = tmp
                out[key] = x
            STATE["swap_used"] = STATE.get("swap_used", 0) + 1
        o2, fd = orig(out, b, dev)
        lp_l, lp_o = r8c.hypothesis_posteriors(o2["lat_logits"], o2["lon_logits"])
        hl, ho = lp_l.argmax(-1), lp_o.argmax(-1)
        ar = torch.arange(b, device=dev)
        rl = torch.as_tensor(r8c.LAT3_V7_IDS, device=dev)[hl]
        ro = torch.as_tensor(r8c.LON6_V7_IDS, device=dev)[ho]
        cf = torch.cat([o2["cons_lat"][ar, rl], o2["cons_lon"][ar, ro]], -1)
        STATE["spy"].append({"hl": int(hl[0]), "ho": int(ho[0]), "cons_fed": _np(cf[0]).astype(np.float32),
                             "force_dict": fd is not None, "force_min": (int(fd["min"]) if fd is not None else -1)})
        return o2, fd
    r3._r8_apply_force = spy                                   # instance attribute: read-only wrapper, the class is untouched
    print(f"[r8oracle] spy installed on {type(r3).__name__}; min_alloc={STATE['min_alloc']} horizons={STATE['horizons']}", flush=True)


def gt_targets(it) -> dict:
    """The TRAINER's literal target block (refc_v3_train.py:5419-5420 + refcv8_train.py:541-551) on this window's item."""
    r8c, rbl, hor = STATE["r8c"], STATE["rbl"], STATE["horizons"]
    pl = it["pose_last"].float().cpu()[None]
    fu = it["future_poses_ext"].float().cpu()[None]
    fv = it["future_valid_ext"].cpu()[None].to(torch.bool)
    tt = rbl.waypoint_targets(pl, fu, hor)
    svv = torch.stack([fv[:, h - 1] for h in hor], dim=1)
    v0 = pl[:, 3]
    slot_t = [float(h) * 0.1 for h in hor]
    gl, go = r8c.tag_paths(tt[:, None], v0, slot_t)
    gl, go = gl[:, 0], go[:, 0]
    ok = svv[:, -1].to(torch.bool)
    gl = torch.where(ok, gl, torch.full_like(gl, -1))
    go = torch.where(ok, go, torch.full_like(go, -1))
    ct = r8c.constraint_targets(tt, svv, pl, fu, fv)
    lt, lv, ot, ov = r8c.normalise_constraints(ct)
    cons = torch.cat([lt, ot], -1)[0].to(torch.float32)
    valid = torch.cat([lv.to(torch.bool) & (gl >= 0)[:, None], ov.to(torch.bool) & (go >= 0)[:, None]], -1)[0]
    cf = torch.where(valid, cons, torch.full_like(cons, float("nan")))
    return {"gl": int(gl[0]), "go": int(go[0]), "cons_force": cf.numpy().astype(np.float32), "cons_raw": torch.nan_to_num(cons).numpy(),
            "valid": valid.numpy()}


def force_spec(arm: str, gt: dict, own_hl: int, own_ho: int):
    gl, go, c = gt["gl"], gt["go"], gt["cons_force"]
    ma = STATE["min_alloc"]
    if arm == "GT_CONS":
        return dict(cons=c, min_alloc=ma) if np.isfinite(c).any() else None
    if arm == "GT_HYP":
        if gl < 0 and go < 0:
            return None
        return dict(lat3=(gl if gl >= 0 else None), lon6=(go if go >= 0 else None))
    if arm == "GT_BOTH":
        if gl < 0 and go < 0 and not np.isfinite(c).any():
            return None
        return dict(lat3=(gl if gl >= 0 else None), lon6=(go if go >= 0 else None),
                    cons=(c if np.isfinite(c).any() else None), min_alloc=ma)
    if arm == "GT_SPEED_ONLY":
        if not np.isfinite(c[3]):
            return None
        return dict(cons=np.array([np.nan, np.nan, np.nan, c[3]], dtype=np.float32), min_alloc=ma)
    if arm == "OWN_ARGMAX":
        return dict(lat3=int(own_hl), lon6=int(own_ho))
    if arm == "SELF_NAN":
        # MUTATION_SELFNAN_MIN (diagnostic only; default unset = the prereg's min_alloc): set it to the seam default 32 and C3 must go RED
        return dict(cons=np.full(4, np.nan, dtype=np.float32), min_alloc=(MUTATION_SELFNAN_MIN if MUTATION_SELFNAN_MIN else ma))
    if arm == "GT_HYP_SWAP":
        return None if (gl < 0 and go < 0) else dict(swap=(gl, go))
    if arm == "GT_BOTH_SWAP":
        if gl < 0 and go < 0 and not np.isfinite(c).any():
            return None
        return dict(swap=(gl, go), cons=(c if np.isfinite(c).any() else None), min_alloc=ma)
    raise ValueError(arm)


def save(final: bool, extra: dict | None = None):
    recs = STATE["records"]
    if not recs:
        return
    out = {"clip_index": np.array([r["ci"] for r in recs], np.int64), "ws": np.array([r["ws"] for r in recs], np.int64),
           "gt_gl": np.array([r["gt"]["gl"] for r in recs], np.int64), "gt_go": np.array([r["gt"]["go"] for r in recs], np.int64),
           "gt_cons_force": np.stack([r["gt"]["cons_force"] for r in recs]), "gt_cons_raw": np.stack([r["gt"]["cons_raw"] for r in recs]),
           "gt_valid": np.stack([r["gt"]["valid"] for r in recs])}
    for arm in ("OWN",) + ARMS:
        has = np.array([arm in r["arms"] for r in recs])
        out[f"has_{arm}"] = has
        nan8 = np.full((N_SLOT, 2), np.nan, np.float32)
        out[f"traj_{arm}"] = np.stack([r["arms"][arm]["traj"] if arm in r["arms"] else nan8 for r in recs]).astype(np.float32)
        out[f"sel_{arm}"] = np.array([r["arms"][arm]["sel"] if arm in r["arms"] else -1 for r in recs], np.int64)
        out[f"forced_{arm}"] = np.array([bool(r["arms"][arm]["forced"]) if arm in r["arms"] else False for r in recs])
        out[f"spy_hl_{arm}"] = np.array([r["arms"][arm]["spy"]["hl"] if arm in r["arms"] else -1 for r in recs], np.int64)
        out[f"spy_ho_{arm}"] = np.array([r["arms"][arm]["spy"]["ho"] if arm in r["arms"] else -1 for r in recs], np.int64)
        out[f"spy_cons_{arm}"] = np.stack([r["arms"][arm]["spy"]["cons_fed"] if arm in r["arms"] else np.full(4, np.nan, np.float32)
                                          for r in recs]).astype(np.float32)
        out[f"spy_fd_{arm}"] = np.array([bool(r["arms"][arm]["spy"]["force_dict"]) if arm in r["arms"] else False for r in recs])
        out[f"spy_fmin_{arm}"] = np.array([r["arms"][arm]["spy"]["force_min"] if arm in r["arms"] else -9 for r in recs], np.int64)
        out[f"prior_maxdiff_{arm}"] = np.array([r["arms"][arm]["prior_maxdiff"] if arm in r["arms"] else np.nan for r in recs], np.float64)
        out[f"wall_{arm}"] = np.array([r["arms"][arm]["wall"] if arm in r["arms"] else np.nan for r in recs], np.float64)
    out["prior_OWN"] = np.stack([r["prior"] for r in recs]).astype(np.float32)
    meta = {"tool": "r8oracle_roll.py", "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__), "arms": list(ARMS),
            "child_argv": CHILD, "final": bool(final), "n_windows": len(recs), "n_navnull_calls": STATE["n_navnull"],
            "min_alloc": STATE["min_alloc"], "horizons": STATE["horizons"], "control_every": CONTROL_EVERY,
            "MUTATION_SELFNAN_MIN": MUTATION_SELFNAN_MIN,
            "wall_s": round(time.time() - STATE["t0"], 1), "saved": time.strftime("%FT%T%z"), **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.oracle_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


class OracleProxy(_Base):
    """The battery's proxy + the forced arms (prereg section 2).  OWN = the battery's own `__call__` (its record, its dump)."""

    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        model = d["model"]
        if STATE["r3"] is None:
            locate_r3(model)
        if nav_cmd is None:                                    # not expected under --os-only; passed through unforced
            STATE["n_navnull"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        r3 = STATE["r3"]
        n, dev = int(frames.shape[0]), frames.device
        devs = [dev] if dev.type == "cuda" else []
        st_cpu = torch.get_rng_state()
        st_cuda = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
        k0 = len(STATE["spy"])
        t_w = time.time()
        out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        w_own = time.time() - t_w
        if len(STATE["spy"]) != k0 + 1:
            raise RuntimeError(f"[r8oracle] the OWN call ran the r8 force site {len(STATE['spy']) - k0} times (expected 1)")
        sp_own = STATE["spy"][-1]
        key = tuple(int(x) for x in self.ctx.index)
        it = self.ctx.item
        W = int(getattr(r3.cfg.core, "window", 8))
        if W != 8:
            raise SystemExit(f"[r8oracle] cfg.core.window {W} != 8: the (clip, ws) key rule would drift")
        gt = gt_targets(it)
        prior_own = (_np(out["residual_prior_path"][0]) if "residual_prior_path" in out and torch.is_tensor(out["residual_prior_path"])
                     else np.full((N_SLOT, 2), np.nan, np.float32))
        rec = {"ci": key[0], "ws": key[1] + W - 1, "gt": gt, "prior": prior_own,
               "arms": {"OWN": {"traj": _np(out["traj"][0]), "sel": int(out["sel_idx"][0]), "forced": False, "spy": sp_own,
                                "prior_maxdiff": 0.0, "wall": round(w_own, 3)}}}
        for arm in ARMS:
            if arm in CONTROL_ARMS and key[0] % CONTROL_EVERY != 0:
                continue
            spec = force_spec(arm, gt, sp_own["hl"], sp_own["ho"])
            if spec is None:                                   # no GT for this window: the arm IS OWN by construction (counted)
                rec["arms"][arm] = dict(rec["arms"]["OWN"], forced=False, wall=0.0)
                continue
            t_a = time.time()
            with torch.random.fork_rng(devices=devs):
                torch.set_rng_state(st_cpu)
                if st_cuda is not None:
                    torch.cuda.set_rng_state(st_cuda, dev)
                kw2 = dict(kw)
                kw2.update(self._inputs(n, dev))               # ego_poses is one-shot: re-supplied for every call
                fz = dict(spec)
                swap = fz.pop("swap", None)
                if fz.get("cons") is not None:
                    fz["cons"] = torch.as_tensor(np.asarray(fz["cons"], np.float32), device=dev)[None].expand(n, 4).contiguous()
                if any(fz.get(k) is not None for k in ("lat3", "lon6", "cons")):
                    r3.set_r8_force(**fz)
                if swap is not None:
                    STATE["swap"] = swap
                k1 = len(STATE["spy"])
                with torch.no_grad():
                    o = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw2)
                if getattr(r3, "_r8_force", None) is not None:
                    raise RuntimeError(f"[r8oracle] arm {arm}: the force was NOT consumed by the forward (window {key})")   # C5
                if len(STATE["spy"]) != k1 + 1:
                    raise RuntimeError(f"[r8oracle] arm {arm}: force site ran {len(STATE['spy']) - k1} times")
                if "swap" in STATE:
                    raise RuntimeError(f"[r8oracle] arm {arm}: the logit swap was NOT consumed by the forward (window {key})")
            pr = (_np(o["residual_prior_path"][0]) if "residual_prior_path" in o and torch.is_tensor(o["residual_prior_path"]) else None)
            pmd = float(np.nanmax(np.abs(pr - prior_own))) if pr is not None else float("nan")
            rec["arms"][arm] = {"traj": _np(o["traj"][0]), "sel": int(o["sel_idx"][0]), "forced": True, "spy": STATE["spy"][-1],
                                "prior_maxdiff": pmd, "wall": round(time.time() - t_a, 3)}
            del o
        STATE["spy"].clear()
        STATE["records"].append(rec)
        STATE["n_windows"] += 1
        if STATE["n_windows"] % int(A.save_every) == 0:
            save(False)
            el = time.time() - STATE["t0"]
            print(f"ZZR8OPROGZZ {STATE['n_windows']} windows elapsed={el:.0f}s last_window_own={w_own:.2f}s "
                  f"forced={sum(v['wall'] for k, v in rec['arms'].items() if k != 'OWN'):.2f}s", flush=True)
        return out


def main() -> int:
    RR.Refcv6Proxy = OracleProxy                               # install_patches' load_model resolves the name at call time
    print(f"[r8oracle] prereg sha256 {SEAL_SHA[:16]}... arms={ARMS} child={' '.join(CHILD)}", flush=True)
    t0 = time.time()
    rc = PF.main()
    if STATE["n_windows"] == 0:
        print("ZZR8OBADZZ the OracleProxy never ran: the roll did not use the re-bound proxy class", flush=True)
        return 4
    walls = {arm: float(np.nansum([r["arms"][arm]["wall"] for r in STATE["records"] if arm in r["arms"]])) for arm in ("OWN",) + ARMS}
    save(True, {"pf_main_rc": rc, "wall_by_arm_s": walls, "roll_wall_s": round(time.time() - t0, 1)})
    print(f"[r8oracle] windows={STATE['n_windows']} rc={rc} wall_by_arm_s={json.dumps({k: round(v, 1) for k, v in walls.items()})} "
          f"{'ZZR8OOKZZ' if rc == 0 else 'ZZR8OBADZZ'}", flush=True)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
