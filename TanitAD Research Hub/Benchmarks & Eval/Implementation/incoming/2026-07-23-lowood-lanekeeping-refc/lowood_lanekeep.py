"""P1+P2+P3 — the FIRST absolute low-OOD closed-loop LANE-KEEPING comparison of
the two main arms (flagship v1 vs REF-C base), on real-footage-in-the-loop.

Extends 2026-07-23-lower-ood-closedloop-source/lowood_closedloop.py. The design of
the loop is verbatim that harness (which is itself a C6-clean minimal edit of
taniteval/closedloop.py): KEEP the deployed planner + controller (strategic ->
tactical -> 0.5 s pure-pursuit waypoint -> (steer, accel) [wp_to_control] ->
kinematic bicycle); the observation is ALWAYS a real recorded window, arc-length
re-indexed by the ego's on-policy progress and warped by the residual on-policy
(dlat, dpsi). The loop's OWN deviation drives the OOD; longitudinal OOD ~= 0 by
construction (real frame at the right arc-length).

WHAT THIS FILE ADDS
  P1  corridor_departure_rate — the LANE-KEEPING proxy that IS measurable at low
      OOD. XTE (cross-track error) == the harness's signed lateral offset |dlat|.
      corridor_departure_rate := fraction of on-policy steps whose |XTE| exceeds a
      lane HALF-width threshold. Primary threshold 1.75 m (a 3.5 m lane; see
      THRESHOLD note below). Also emits peak XTE and time-to-departure (TTD).
  P2  the REF-C arm (base 104.2M). REF-C's DEPLOYED anchored-diffusion head is a
      drop-in: model(fw, nav_cmd=None, v0, steps=2)["traj"][:, 0] is the 0.5 s
      ego-frame lookahead the SAME pure-pursuit controller consumes. It sees the
      EXACT SAME warped real-footage window the flagship sees (the val cache is the
      canonical phase-0 f-theta stack both arms trained on -> the f-theta canonical
      input contract is already met; no re-canonicalization, cf. refc_driver.py
      which re-canonicalizes ONLY because NuRec hands it raw native f-theta).
  P3  both arms on the 40-ep clean val (physicalai-val-0c5f7dac3b11) in ONE process
      on the IDENTICAL windows -> episode-cluster bootstrap CIs (taniteval/ci.py)
      per arm + a PAIRED flagship-vs-REF-C bootstrap (same windows each draw).

*** HONEST FRAME — do not overclaim. *** This measures LANE-KEEPING / on-policy
drift at low OOD. It is STRUCTURALLY UNABLE to measure off-road departure or
collision: the real-footage source is map-free and agent-free, so it can only emit
deviation from the recorded corridor. A true off-road / collision rate needs a
map + reactive agents + a low-OOD renderer = AlpaSim (whose reconstructions are the
~3.2x OOD this whole source exists to escape; RETRACTION_LOG 07-22/07-23). The
"corridor" here is the recorded ego path +/- a lane half-width, NOT a mapped lane.

THRESHOLD (cited). A lane half-width of 1.75 m corresponds to a 3.5 m lane — the
common design width for US arterials / German Autobahn lanes (US Interstate 3.6 m;
German RAA Autobahn 3.5-3.75 m; urban lanes 3.0-3.5 m). |XTE| > 1.75 m therefore
means the ego REFERENCE POINT has crossed from the lane centre to the lane edge —
a genuine lane departure. We also report a stricter 1.0 m (with a ~1.8 m-wide
vehicle centred, ~0.85 m of edge clearance, so |XTE| > 1.0 m already puts a wheel
over the line) and a looser 2.5 m (well into an adjacent lane / off a narrow road),
so the departure rate is read as a curve, not a single knife-edge.

tick-0 self-check: at k=0 the ego is on-path -> dlat == 0 -> |XTE| == 0 -> no
threshold is exceeded -> corridor_departure at tick-0 is EXACTLY 0 for every arm
(asserted). Same invariant the parent harness checks with max_lat_m == 0.

Reproduce (eval pod tanitad-eval, GPU free, gpu_lock held):
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    python3 lowood_lanekeep.py \
      --flagship-ckpt /root/models/flagship-30k/ckpt.pt \
      --refc-ckpt /root/models/refc-base-30k/ckpt.pt --refc-preset base \
      --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 --episodes 40 \
      --p1-json lowood_flagship_ci.json \
      --out lowood_lanekeep_40ep.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Portable across pod layouts (eval pod: /root/TanitAD/stack; pod1: /workspace/...).
for _p in ("/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
           "/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace", "/root/taniteval", "/root/TanitAD/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from driving_diagnostic import (WP_STEPS, gt_ego_waypoints,  # noqa: E402
                                net_heading_change_deg)
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,  # noqa: E402
                               refc_xl_config)
from taniteval import ci as _ci  # noqa: E402

# ---- loop constants (identical to lowood_closedloop.py / closedloop.py) ------
W = 8
K = max(WP_STEPS)                 # 20 = 2 s
DT = 0.1
WHEELBASE = 2.7
LOOKAHEAD_STEP = 5                # 0.5 s pure-pursuit target
LD2_FLOOR = 0.25
STEER_CLAMP = 0.05
ACCEL_CLAMP = 3.0
SPEED_TC = 0.5
WP_IDX = [k - 1 for k in WP_STEPS]

# ---- inlined geometry (verbatim from lowood_probe.py, so this harness is
#      self-contained and depends only on the on-pod stack) --------------------
F_EFF = 266.0            # canonical pinhole focal (px) of the 256x256 phase-0 cache
CXY = 128.0             # principal point (centred)
SPEED_SCALE = 10.0      # flagship_losses.py: v0 = pose_last[:,3] / 10

_REFC_PRESETS = {"base": refc_config, "small": refc_small_config,
                 "xl": refc_xl_config}


def sampling_homography(dlat_m, dyaw_deg, h_cam, pitch_deg, f=F_EFF, c=CXY):
    """cam2(offset)->cam1(real) sampling homography for grid_sample. dlat_m to the
    RIGHT, dyaw_deg to the LEFT; Delta=0 -> identity. (lowood_probe.py verbatim.)"""
    Kk = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=torch.float64)
    Ki = torch.linalg.inv(Kk)
    p = math.radians(pitch_deg)
    n = torch.tensor([0.0, math.cos(p), math.sin(p)], dtype=torch.float64)
    d = float(h_cam)
    psi = math.radians(dyaw_deg)
    Ry = torch.tensor([[math.cos(-psi), 0, math.sin(-psi)],
                       [0, 1.0, 0],
                       [-math.sin(-psi), 0, math.cos(-psi)]], dtype=torch.float64)
    Cc = torch.tensor([dlat_m, 0.0, 0.0], dtype=torch.float64)
    t = -(Ry @ Cc)
    H_1to2 = Kk @ (Ry + torch.outer(t, n) / d) @ Ki
    return torch.linalg.inv(H_1to2)


def build_world_from_ckpt(cfg, ck):
    """flagship: infer trained operative action_dim from act_emb, wire predictor/
    tactical_pred, strict-load. (lowood_probe.py verbatim.)"""
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    w = sd.get("predictor.act_emb.0.weight")
    action_dim = int(w.shape[1]) if torch.is_tensor(w) and w.ndim == 2 else 2
    if cfg.predictor.action_dim != action_dim:
        cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=action_dim)
        if getattr(cfg, "tactical_pred", None) is not None:
            cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred,
                                                    action_dim=action_dim)
        if hasattr(cfg, "speed_input"):
            cfg.speed_input = (action_dim == 3)
    world = WorldModel(cfg)
    world.load_state_dict(sd)
    return world, action_dim >= 3


def _apply_overrides(cfg, d):
    """Push a run's config.json cfg dump onto a preset dataclass (loaders.py /
    refc_v12_cache.load_frozen convention) so every gated graft is built at the
    trained shape and the state_dict loads STRICT."""
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply_overrides(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


def load_refc(ckpt, preset, device):
    """Minimal inline of refc_v12_cache.load_frozen (avoids the refb_train /
    refc_rescorer import chain): preset cfg + config.json overrides -> RefCModel ->
    STRICT load ck['model'] -> eval, frozen. Anchors travel in the ckpt buffer."""
    cfg = _REFC_PRESETS[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    assert not cfg.refc1, "refc1 ckpt: horizons are path checkpoints, not time"
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])            # STRICT
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, int(ck.get("step", -1))


def wp_to_control(w_look, v):
    """closedloop.py verbatim: 0.5 s ego-frame waypoint + speed -> (steer, accel)."""
    x, y = w_look[:, 0], w_look[:, 1]
    ld2 = (x * x + y * y).clamp_min(LD2_FLOOR)
    kappa = 2.0 * y / ld2
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    v_target = x / (LOOKAHEAD_STEP * DT)
    accel = ((v_target - v) / SPEED_TC).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    return steer, accel


def warp_batch(fw, Hs):
    """fw [b,W,C,Hh,Ww] in [0,1]; Hs [b,3,3] per-window cam2->cam1 sampling
    homography. Per-window H, border-replicate bilinear. (lowood_closedloop.py.)"""
    b, Wn, C, Hh, Ww = fw.shape
    dev = fw.device
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64, device=dev),
                            torch.arange(Ww, dtype=torch.float64, device=dev),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T
    src = Hs.to(dev).to(torch.float64) @ P
    su = (src[:, 0] / src[:, 2]).reshape(b, Hh, Ww)
    sv = (src[:, 1] / src[:, 2]).reshape(b, Hh, Ww)
    gx = 2.0 * su / (Ww - 1) - 1.0
    gy = 2.0 * sv / (Hh - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    grid = grid[:, None].expand(-1, Wn, -1, -1, -1).reshape(b * Wn, Hh, Ww, 2).float()
    out = F.grid_sample(fw.reshape(b * Wn, C, Hh, Ww), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(b, Wn, C, Hh, Ww)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class OODMap:
    """Interpolate the P1 MEASURED flagship envelope: |dlat|,|dpsi| -> ADE ratio.
    The envelope characterises the SOURCE's observation-OOD (flagship-measured);
    applied to EACH arm's OWN on-policy deviations. (lowood_closedloop.py.)"""
    def __init__(self, ci_json):
        d = json.loads(Path(ci_json).read_text())
        self.base = d["baseline_real_frames"]["mean"]
        self.lat_x = np.array([r["amount"] for r in d["conditions"]["lat"]])
        self.lat_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["lat"]])
        self.yaw_x = np.array([r["amount"] for r in d["conditions"]["yaw"]])
        self.yaw_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["yaw"]])

    def ratio(self, dlat_abs, dpsi_abs_deg):
        al = float(np.interp(dlat_abs, self.lat_x, self.lat_y))
        ay = float(np.interp(dpsi_abs_deg, self.yaw_x, self.yaw_y))
        ex_l = max(0.0, (al - self.base) / self.base)
        ex_y = max(0.0, (ay - self.base) / self.base)
        return 1.0 + ex_l + ex_y


def plan_lookahead(arch, model, fw, ev, nav, device):
    """Return each arm's 0.5 s ego-frame lookahead waypoint [b,2] on CPU, from the
    SAME warped real-footage window fw. Deployed planner head of each arm.
      flagship: encode_window -> strategic_policy(nav=follow) -> tactical_policy ->
                waypoints[LOOKAHEAD_STEP]   (== 0.5 s)
      refc:     model(fw, nav_cmd=None(->follow), v0=ev, steps=2)["traj"][:, 0]
                (horizon 5 == 0.5 s; v0 raw m/s, model scales /10 internally)."""
    if arch == "flagship":
        states = model.encode_window(fw)
        ctx = model.strategic_policy(states, nav)["ctx"]
        wp = model.tactical_policy(states, ctx)["waypoints"]
        return wp[LOOKAHEAD_STEP].cpu()
    out = model(fw, nav_cmd=None, v0=ev.to(device), steps=2)
    return out["traj"][:, 0].cpu()


@torch.no_grad()
def cl_realfootage(arch, model, episodes, device, ood, stride=8, batch=16,
                   max_windows=None):
    """Real-footage-in-the-loop closed loop over every stride-window, for one arm.
    Returns per-window arrays (identical windows/order across arms for a fixed
    stride/batch)."""
    navfollow_cache = {}
    rows = {k: [] for k in ("de", "peak_lat", "mean_lat", "peak_yaw", "mean_yaw",
                            "ood_mean", "ood_peak", "head_deg", "speed", "eid",
                            "lat_traj", "yaw_traj", "tick0_lat", "tick0_yaw")}
    n_done = 0
    for ep_i, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - W - K, stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            t0 = torch.tensor(ch)
            last = t0 + W - 1
            idx = last[:, None] + torch.arange(0, K + 1)[None]
            Pxy = poses[idx][..., :2]
            Pyaw = poses[idx][..., 2]
            oyaw = poses[last, 2]
            oxy = poses[last, :2]
            ex = poses[last, 0].clone(); ey = poses[last, 1].clone()
            eyaw = poses[last, 2].clone(); ev = poses[last, 3].clone()
            nav = navfollow_cache.get(b)
            if nav is None:
                nav = torch.zeros(b, dtype=torch.long, device=device)
                navfollow_cache[b] = nav
            ego_ego = torch.zeros(b, K, 2)
            lat_t = torch.zeros(b, K); yaw_t = torch.zeros(b, K)
            ar = torch.arange(b)
            for k in range(K):
                d = (Pxy - torch.stack([ex, ey], -1)[:, None]).norm(dim=-1)
                mstar = d.argmin(dim=1)
                pref = Pxy[ar, mstar]; yref = Pyaw[ar, mstar]
                dx = ex - pref[:, 0]; dy = ey - pref[:, 1]
                dlat = -torch.sin(yref) * dx + torch.cos(yref) * dy   # left +, metres = signed XTE
                dpsi = _wrap(eyaw - yref)
                lat_t[:, k] = dlat; yaw_t[:, k] = dpsi
                wins = []
                for i in range(b):
                    s = int(t0[i] + mstar[i])
                    wins.append(fr[s:s + W])
                fw = torch.stack(wins).to(device)
                Hs = torch.stack([
                    sampling_homography(float(dlat[i]),
                                        float(math.degrees(dpsi[i])), 1.5, 0.0)
                    for i in range(b)])
                fw = warp_batch(fw, Hs)
                w_look = plan_lookahead(arch, model, fw, ev, nav, device)
                steer, accel = wp_to_control(w_look, ev)
                ex = ex + ev * torch.cos(eyaw) * DT
                ey = ey + ev * torch.sin(eyaw) * DT
                eyaw = eyaw + ev / WHEELBASE * torch.tan(steer) * DT
                ev = (ev + accel * DT).clamp_min(0.0)
                wdx = ex - oxy[:, 0]; wdy = ey - oxy[:, 1]
                xf = torch.cos(oyaw) * wdx + torch.sin(oyaw) * wdy
                yl = -torch.sin(oyaw) * wdx + torch.cos(oyaw) * wdy
                ego_ego[:, k, 0] = xf.cpu(); ego_ego[:, k, 1] = yl.cpu()
            pred = ego_ego[:, WP_IDX]
            gt = gt_ego_waypoints(poses, last)
            de = torch.linalg.norm(pred - gt, dim=-1)
            lat_abs = lat_t.abs(); yaw_abs_deg = yaw_t.abs() * 180 / math.pi
            ood_mean = torch.zeros(b); ood_peak = torch.zeros(b)
            if ood is not None:
                for i in range(b):
                    rr = [ood.ratio(float(lat_abs[i, k]), float(yaw_abs_deg[i, k]))
                          for k in range(K)]
                    ood_mean[i] = float(np.mean(rr)); ood_peak[i] = float(np.max(rr))
            hd = net_heading_change_deg(poses, last)
            rows["de"].append(de)
            rows["peak_lat"].append(lat_abs.max(1).values)
            rows["mean_lat"].append(lat_abs.mean(1))
            rows["peak_yaw"].append(yaw_abs_deg.max(1).values)
            rows["mean_yaw"].append(yaw_abs_deg.mean(1))
            rows["ood_mean"].append(ood_mean); rows["ood_peak"].append(ood_peak)
            rows["head_deg"].append(hd if torch.is_tensor(hd) else torch.tensor(hd))
            rows["speed"].append(poses[last, 3])
            rows["tick0_lat"].append(lat_abs[:, 0]); rows["tick0_yaw"].append(yaw_abs_deg[:, 0])
            rows["lat_traj"].append(lat_abs); rows["yaw_traj"].append(yaw_abs_deg)
            rows["eid"].extend([str(ep_i)] * b)
            n_done += b
            if max_windows and n_done >= max_windows:
                break
        if max_windows and n_done >= max_windows:
            break
    out = {}
    for k, v in rows.items():
        out[k] = v if k == "eid" else torch.cat(v)
    out["ade"] = out["de"].mean(1)            # [N] closed-loop ADE 0-2s
    return out


# --------------------------------------------------------------------------- #
# Corridor / lane-keeping metric (P1)                                         #
# --------------------------------------------------------------------------- #
def corridor_stats(lat_abs, thr, dt=DT):
    """lat_abs [N,K] per-window per-step |XTE| -> per-window:
       dep_rate [N]  fraction of the K on-policy steps with |XTE| > thr,
       ever    [N]   1 if the window EVER leaves the corridor (== peak XTE > thr),
       ttd     [N]   time (s) to first departure; NaN if it never departs."""
    over = lat_abs > thr
    dep_rate = over.float().mean(1)
    has = over.any(1)
    first_idx = over.float().argmax(1).float()        # first True index (0 if none)
    ttd = torch.where(has, first_idx * dt,
                      torch.full_like(dep_rate, float("nan")))
    return dep_rate, has.float(), ttd


def _boot(x, eid, reduce="mean"):
    return _ci.episode_cluster_bootstrap(np.asarray(x, float), eid, reduce=reduce,
                                         n_boot=2000)


def summarize(pw, thresholds, primary, junction_deg):
    """Per-arm bootstrapped block, overall + junction/longitudinal strata."""
    eid = pw["eid"]
    ade = pw["ade"].numpy()
    lat_abs = pw["lat_traj"]                    # [N,K]
    peak_xte = pw["peak_lat"].numpy()
    hd = pw["head_deg"].numpy(); spd = pw["speed"].numpy()
    dep = {t: corridor_stats(lat_abs, t)[0].numpy() for t in thresholds}
    ever = {t: corridor_stats(lat_abs, t)[1].numpy() for t in thresholds}
    ttd_p = corridor_stats(lat_abs, primary)[2].numpy()

    junc = np.abs(hd) >= junction_deg
    long_ = (~junc) & (spd >= np.median(spd))

    def blk(mask):
        m = np.flatnonzero(mask)
        if not len(m):
            return None
        e = [eid[i] for i in m]
        ttd_m = ttd_p[m]
        dep_wins = ttd_m[~np.isnan(ttd_m)]
        out = {
            "n_windows": int(len(m)),
            "n_episodes": int(len(set(e))),
            "closed_ade2s_m": _boot(ade[m], e),
            "corridor_departure_rate": _boot(dep[primary][m], e),
            "corridor_departure_rate_by_threshold_m": {
                f"{t:g}": _boot(dep[t][m], e) for t in thresholds},
            "window_departure_rate": _boot(ever[primary][m], e),
            "window_departure_rate_by_threshold_m": {
                f"{t:g}": round(float(ever[t][m].mean()), 4) for t in thresholds},
            "peak_xte_m": _boot(peak_xte[m], e),
            "time_to_departure_s": {
                "primary_threshold_m": primary,
                "mean_among_departing": (round(float(dep_wins.mean()), 4)
                                         if len(dep_wins) else None),
                "median_among_departing": (round(float(np.median(dep_wins)), 4)
                                           if len(dep_wins) else None),
                "n_departing_windows": int(len(dep_wins)),
                "frac_windows_departing": round(float((~np.isnan(ttd_m)).mean()), 4),
            },
        }
        if pw["ood_peak"].abs().sum() > 0:
            out["ood_peak_ratio"] = _boot(pw["ood_peak"].numpy()[m], e)
            out["ood_mean_ratio"] = _boot(pw["ood_mean"].numpy()[m], e)
        return out

    return {"overall": blk(np.ones(len(eid), bool)),
            "junction": blk(junc), "longitudinal": blk(long_),
            "junction_deg_threshold": junction_deg}


def paired(a_pw, b_pw, primary, junction_deg):
    """PAIRED flagship(a) - refc(b) bootstrap on the SAME windows each draw. Both
    arms scored on the identical window set (asserted eid-aligned) so the shared
    per-window difficulty cancels. Negative delta => a (flagship) is LOWER/better."""
    assert a_pw["eid"] == b_pw["eid"], "arms not window-aligned"
    eid = a_pw["eid"]
    hd = a_pw["head_deg"].numpy(); spd = a_pw["speed"].numpy()
    junc = np.abs(hd) >= junction_deg
    long_ = (~junc) & (spd >= np.median(spd))
    a_ade, b_ade = a_pw["ade"].numpy(), b_pw["ade"].numpy()
    a_dep = corridor_stats(a_pw["lat_traj"], primary)[0].numpy()
    b_dep = corridor_stats(b_pw["lat_traj"], primary)[0].numpy()
    a_pk, b_pk = a_pw["peak_lat"].numpy(), b_pw["peak_lat"].numpy()

    def pblk(mask):
        m = np.flatnonzero(mask)
        if not len(m):
            return None
        e = [eid[i] for i in m]
        P = _ci.paired_episode_cluster_bootstrap
        return {
            "n_windows": int(len(m)), "n_episodes": int(len(set(e))),
            "delta_closed_ade2s_m_flagship_minus_refc":
                P(a_ade[m], b_ade[m], e, n_boot=2000),
            "delta_corridor_departure_rate_flagship_minus_refc":
                P(a_dep[m], b_dep[m], e, n_boot=2000),
            "delta_peak_xte_m_flagship_minus_refc":
                P(a_pk[m], b_pk[m], e, n_boot=2000),
        }
    return {"note": "negative delta => flagship LOWER (better) than REF-C; "
                    "'separated' == CI excludes 0 (decision predicate)",
            "overall": pblk(np.ones(len(eid), bool)),
            "junction": pblk(junc), "longitudinal": pblk(long_)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagship-ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--refc-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base", choices=("base", "small", "xl"))
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--p1-json", default="",
                    help="P1 flagship OOD envelope (optional). Empty -> skip OOD ratio.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75,
                    help="primary lane HALF-width (m) for corridor_departure_rate")
    ap.add_argument("--corridor-grid", default="1.0,1.75,2.5")
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--arms", default="flagship,refc",
                    help="which arms to run (comma). Default both.")
    ap.add_argument("--max-windows", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = [float(x) for x in args.corridor_grid.split(",") if x != ""]
    primary = args.corridor_halfwidth
    assert primary in thresholds, f"primary {primary} must be in grid {thresholds}"
    ood = OODMap(args.p1_json) if args.p1_json and Path(args.p1_json).exists() else None
    print(f"[lanekeep] dev={device} thresholds={thresholds} primary={primary} "
          f"OOD_envelope={'yes' if ood else 'none'}", flush=True)

    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    print(f"[lanekeep] {len(episodes)} val episodes from {args.val_dir}", flush=True)
    arms = [a for a in args.arms.split(",") if a]

    per_arm = {}
    arm_meta = {}
    with strict_numerics():
        for arch in arms:
            if arch == "flagship":
                ck = torch.load(args.flagship_ckpt, map_location="cpu",
                                weights_only=True)
                model, speed_input = build_world_from_ckpt(flagship4b_config(), ck)
                model = model.to(device).eval()
                step = int(ck.get("step", -1))
                arm_meta[arch] = {"ckpt": args.flagship_ckpt, "step": step,
                                  "speed_input": speed_input, "params_head": "strategic->tactical"}
            else:
                model, step = load_refc(args.refc_ckpt, args.refc_preset, device)
                arm_meta[arch] = {"ckpt": args.refc_ckpt, "step": step,
                                  "preset": args.refc_preset,
                                  "decode": "model(fw,nav_cmd=None,v0,steps=2)['traj'][:,0]"}
            print(f"[lanekeep] {arch} ckpt step {arm_meta[arch]['step']} "
                  f"loaded on {device}", flush=True)
            pw = cl_realfootage(arch, model, episodes, device, ood,
                                stride=args.stride, batch=args.batch,
                                max_windows=(args.max_windows or None))
            # tick-0 self-check: on-path -> XTE == 0 -> zero corridor departures.
            t0lat = float(pw["tick0_lat"].max())
            t0_dep = int((pw["tick0_lat"] > min(thresholds)).sum())
            assert t0lat < 1e-6 and t0_dep == 0, \
                f"{arch} tick-0 self-check FAILED: max_lat={t0lat} n_dep={t0_dep}"
            per_arm[arch] = pw
            print(f"[lanekeep] {arch} n_win={pw['ade'].shape[0]} "
                  f"n_ep={len(set(pw['eid']))} tick0_lat={t0lat:.2e} (dep={t0_dep}) "
                  f"closed_ade2s={float(pw['ade'].mean()):.3f} "
                  f"peak_xte={float(pw['peak_lat'].mean()):.3f}m", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    res = {
        "_design": "real-footage-in-the-loop low-OOD closed-loop LANE-KEEPING "
                   "(corridor = recorded path +/- lane half-width). Deployed planner "
                   "+ pure-pursuit + bicycle; obs = arc-length re-indexed REAL window "
                   "warped by on-policy (dlat,dpsi). Both arms, identical windows.",
        "_honest_frame": "Measures LANE-KEEPING / on-policy drift at low OOD; "
                         "STRUCTURALLY NOT off-road/collision (map-free, agent-free "
                         "source). A real off-road/collision rate needs AlpaSim "
                         "(the ~3.2x-OOD renderer this source escapes).",
        "corridor_threshold_m": {"primary_half_width": primary, "grid": thresholds,
            "citation": "1.75 m = half of a 3.5 m lane (US arterial / DE Autobahn "
                        "class); 1.0 m ~ wheel-over-line for a ~1.8 m vehicle; "
                        "2.5 m ~ into the adjacent lane."},
        "val_dir": args.val_dir, "n_episodes_requested": args.episodes,
        "stride": args.stride, "junction_deg_threshold": args.junction_deg,
        "ood_envelope": (args.p1_json if ood else None),
        "ood_envelope_note": ("flagship-measured source-OOD envelope, applied to "
                              "EACH arm's own on-policy deviations" if ood else
                              "no OOD envelope supplied; OOD ratio omitted"),
        "arms": arm_meta,
        "results": {a: summarize(per_arm[a], thresholds, primary, args.junction_deg)
                    for a in arms},
    }
    if "flagship" in per_arm and "refc" in per_arm:
        res["paired_flagship_vs_refc"] = paired(per_arm["flagship"], per_arm["refc"],
                                                primary, args.junction_deg)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    for a in arms:
        o = res["results"][a]["overall"]
        cdr = o["corridor_departure_rate"]
        print(f"[lanekeep] {a:8s} overall: closed_ade2s="
              f"{o['closed_ade2s_m']['mean']:.3f}[{o['closed_ade2s_m']['lo']:.3f},"
              f"{o['closed_ade2s_m']['hi']:.3f}] corridor_departure_rate@{primary}m="
              f"{cdr['mean']:.4f}[{cdr['lo']:.4f},{cdr['hi']:.4f}] "
              f"peak_xte={o['peak_xte_m']['mean']:.3f}m "
              f"win_dep={o['window_departure_rate']['mean']:.3f}", flush=True)
    if "paired_flagship_vs_refc" in res:
        p = res["paired_flagship_vs_refc"]["overall"]
        for key in ("delta_closed_ade2s_m_flagship_minus_refc",
                    "delta_corridor_departure_rate_flagship_minus_refc"):
            d = p[key]
            print(f"[lanekeep] PAIRED {key}: {d['delta']:+.4f} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}] separated={d['separated']}",
                  flush=True)
    print(f"[lanekeep] wrote {args.out}", flush=True)
    print("LOWOOD_LANEKEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
