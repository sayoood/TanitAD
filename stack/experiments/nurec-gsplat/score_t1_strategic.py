#!/usr/bin/env python3
"""Score the STRATEGIC option-set family for flagship-v1 and REF-C across MANY T1 scenes.

WHY THIS IS OPEN-LOOP ON THE LOGGED TRACK, AND WHY THAT IS THE RIGHT EXPERIMENT
-------------------------------------------------------------------------------
The one previous run of this family (``results/closedloop_strategic_7c72937c.json``)
drove the policies closed-loop inside a rendered NuRec scene. That has **one scene**, so
the episode-cluster bootstrap had ONE cluster and every interval came back
``CI_NOT_ADMISSIBLE``. Widening it closed-loop means ``volume.nurec`` (~600 MB of
gaussians) per scene plus a render loop — and it would *still* answer a different
question, because the rollout drifts off the logged track and the labels are indexed by
**clipgt pose**.

Here the ego is placed on its LOGGED poses and asked, at each pose on a junction
approach: *given the real camera observations up to now, which continuation do you
choose?* That is exactly what the option-set label scores, and it buys three things:

* **the join is exact** — no fitted pose offset, because the tick IS the labelled pose;
* **the observations are the REAL recorded 4K camera**, not a reconstruction. On these
  scenes the gsplat render is measured at **3.21x OOD** for REF-C, so real footage is
  *more* in-distribution, not less;
* **cost collapses**: ~40 MB of mp4 + ~2 MB of Range-fetched members per scene, against
  a ~1.8-2.0 GB usdz. The 141-scene label sweep MEASURED **310 MB against 160 GB** of
  archives not downloaded.

It is a DIFFERENT experiment from the closed-loop panel and is reported as such. It
cannot see closed-loop instability; it can see whether the strategic head chooses.

⛔ THE CONFOUND THIS SCRIPT EXISTS TO CONTROL: THE NAV COMMAND IS AN ORACLE
---------------------------------------------------------------------------
``closedloop_drive.nav_from_route`` (:348) feeds the policy
``refb_labels.nav_command_v21(gt_poses, i)`` — a command derived from **the ego's own
logged future**. Both deployed heads consume it: ``flagship-v1`` as
``strategic_policy(states, nav)``, ``refc-base`` as ``model(fw, nav_cmd=navt, ...)``.
Scoring ``s_route_logits`` against the branch the ego took while handing the model a
label-derived summary of that same branch can be **pure pass-through**, and a
pass-through would print as perfect strategic skill.

So every arm is run under THREE nav conditions and against a nav-only baseline:

=================  =========================================================
condition          nav_cmd at every pose
=================  =========================================================
``navFOLLOW``      constant ``0`` (FOLLOW) — the honest deployable setting.
                   **This is the headline.** Any signal must come from pixels.
``navORACLE``      ``nav_command_v21`` from the logged future — what the
                   closed-loop panel does, reproduced so the two are comparable.
``navSHUFFLED``    the oracle nav of a DIFFERENT event (seeded permutation) —
                   keeps the marginal, destroys the nav->route link.
=================  =========================================================

``NAV_ECHO`` is a baseline arm with **no image at all**: it maps the oracle nav command
straight to a route class. An arm in ``navORACLE`` that does not beat ``NAV_ECHO`` has
demonstrated nothing about strategy. ``nav_sensitivity`` = navORACLE - navSHUFFLED
measures the pass-through directly.

Together with :func:`taniteval.strategic_optionset.discrimination_control` (a constant
predictor must not score well — the defect that made ``route_head_eq_logged = 1.0000``
meaningless) that is the degeneracy control the brief demands, in two independent forms:
**a constant predictor** and **an input-echo predictor**.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CAM = "camera_front_wide_120fov"
WINDOW = 8            # closedloop_drive.WINDOW
STACK = 3             # closedloop_drive.STACK
NEED = WINDOW + STACK - 1          # 10 native 10 Hz frames -> [8,9,256,256]
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
RELEASE = "sample_set/26.04_release"

#: ``closedloop_drive.NAV_NAMES`` = ("follow","left","right","straight") ->
#: ``strategic_gt.ROUTE`` = {0:LEFT, 1:STRAIGHT, 2:RIGHT, 3:UTURN}.
#: FOLLOW has no route meaning; the echo baseline answers STRAIGHT for it, which is the
#: kindest reading (it is also the most common class, so the baseline is STRONG).
NAV_TO_ROUTE = {0: 1, 1: 0, 2: 2, 3: 1}
NAV_NAMES = ("follow", "left", "right", "straight")


def _taniteval():
    for cand in (Path("/home/nvidia/tv"), *HERE.parents):
        if (cand / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(cand))
            break
        if (cand / "taniteval" / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(cand / "taniteval"))
            break
    from taniteval import strategic_optionset as SO
    return SO


# --------------------------------------------------------------------------- #
# scene assets                                                                 #
# --------------------------------------------------------------------------- #
def fetch_mp4(sid: str, dest: Path, timeout=600) -> dict:
    """The reference camera video — a SEPARATE ~40 MB file, not inside the usdz."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return {"cached": True, "bytes": dest.stat().st_size}
    url = f"{RESOLVE}{RELEASE}/{sid}/{CAM}.mp4"
    h = {"User-Agent": "tanitad-strategic/1"}
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        h["Authorization"] = "Bearer " + tok
    t0 = time.time()
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                timeout=timeout) as r, tmp.open("wb") as fh:
        n = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            n += len(b)
    tmp.replace(dest)
    return {"cached": False, "bytes": n, "seconds": round(time.time() - t0, 1),
            "MBps": round(n / 1e6 / max(time.time() - t0, 1e-9), 1)}


def pose_track(members: Path):
    """``(xy, yaw, ts_us)`` from the clipgt egomotion table — the LABELLED track."""
    import pyarrow.parquet as pq
    t = pq.read_table(members / "clipgt" / "egomotion_estimate.parquet")
    keys = t.column("key").to_pylist()
    ego = t.column("egomotion_estimate").to_pylist()
    ts = np.array([k["timestamp_micros"] for k in keys], np.int64)
    P = np.array([[d["location"]["x"], d["location"]["y"]] for d in ego], float)
    Q = np.array([[d["orientation"]["w"], d["orientation"]["x"],
                   d["orientation"]["y"], d["orientation"]["z"]] for d in ego], float)
    w, x, y, z = Q.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return P, yaw, ts


def gtp_xyv(P, yaw, ts):
    """[T,4] (x, y, yaw, v) — the exact format ``refb_labels`` wants."""
    dt = np.diff(ts) / 1e6
    d = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(dt, 1e-6)
    v = np.concatenate([d, d[-1:]])
    return np.stack([P[:, 0], P[:, 1], yaw, v], 1)


def canon_frames_for_poses(mp4: Path, members: Path, poses_needed, intr, torch,
                           crop_fn, max_pose: int):
    """``{pose -> uint8[3,256,256]}`` — each needed frame canonicalized exactly ONCE.

    ⚠️ ``ftheta_crop_resize``'s crop box depends only on ``(intr, H, W)``, never on the
    frame's content (calib.py:541), so per-frame canonicalization is bit-identical to
    canonicalizing the batch. That is what makes a rolling window affordable: the native
    frames are **3840x2160**, and holding 202 of them would be 5 GB.
    """
    import cv2
    from nurec_loader import RigTrajectories

    rig = RigTrajectories(members / "rig_trajectories.json")
    n_cf = rig.n_frames(CAM)
    ts_cam = np.array([rig.frame_timestamps_us(CAM, i)[0] for i in range(n_cf)], np.int64)
    _, _, ts_pose = pose_track(members)
    j = np.abs(ts_cam[None, :] - ts_pose[:, None]).argmin(1)
    resid = np.abs(ts_cam[j] - ts_pose)

    want = {int(j[p]): p for p in sorted(poses_needed) if p <= max_pose}
    cap = cv2.VideoCapture(str(mp4))
    out, k = {}, 0
    last = max(want) if want else -1
    while k <= last:
        if k in want:
            ok, fr = cap.retrieve() if cap.grab() else (False, None)
            if not ok:
                break
            v = torch.from_numpy(fr[:, :, ::-1].copy()).permute(2, 0, 1)[None]  # BGR->RGB
            out[want[k]] = crop_fn(v, intr, 256, center="principal")[0].cpu()
        else:
            if not cap.grab():
                break
        k += 1
    cap.release()
    return out, {"n_camera_frames": n_cf,
                 "pose_to_frame_stride_median": float(np.median(np.diff(j))),
                 "pose_frame_residual_us_median": int(np.median(resid)),
                 "pose_frame_residual_us_max": int(resid.max()),
                 "n_frames_canonicalized": len(out)}


# --------------------------------------------------------------------------- #
# arms                                                                         #
# --------------------------------------------------------------------------- #
class Arm:
    """A deployed policy, asked only for its route logits."""

    def __init__(self, name, ckpt, kind, device="cuda"):
        import torch
        self.torch, self.name, self.kind = torch, name, kind
        self.device = device if torch.cuda.is_available() else "cpu"
        if kind == "flagship":
            from tanitad.config import flagship4b_config
            from tanitad.models.fourbrain import WorldModel
            cfg = flagship4b_config()
            object.__setattr__(cfg.predictor, "action_dim", 3)
            if getattr(cfg, "tactical_pred", None) is not None:
                object.__setattr__(cfg.tactical_pred, "action_dim", 3)
            self.model = WorldModel(cfg)
            ck = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ck["model"])
            self.model = self.model.to(self.device).eval()
            self.step = ck.get("step")
        elif kind == "refc":
            from refc_v12_cache import load_frozen
            self.model, self.cfg, self.step = load_frozen(ckpt, "base", None, self.device)
        else:
            raise ValueError(kind)

    def route_logits_nav_sweep(self, fw, v0, navs=(0, 1, 2, 3)):
        """-> ``({nav: logits}, resolved_key)`` — the SAME observation under EVERY nav.

        ⛔ **A permutation control is not enough and this replaces one.** Most T1 scenes
        carry exactly ONE decision event, so a within-scene shuffle is the identity and
        the control silently measures nothing (MEASURED on the first smoke run). Sweeping
        the nav input over its whole vocabulary at a FIXED observation measures the
        pass-through directly: if the argmax moves with ``nav`` the head is echoing its
        own input, and any accuracy scored under an ORACLE nav is that echo, not strategy.
        A shuffled-nav condition is then derived offline from this same sweep, permuting
        across ALL events in the run rather than within one clip.

        The KEY MATTERS: MEASURED, flagship-v1 writes ``s_route_logits`` and refc-base
        writes ``route_logits`` — reading only the first is what made REF-C's route head
        invisible to ``cl_metrics.py:176``.
        """
        torch = self.torch
        out, key = {}, None
        with torch.no_grad():
            states = self.model.encode_window(fw) if self.kind == "flagship" else None
            for nav in navs:
                navt = torch.tensor([int(nav)], dtype=torch.long, device=self.device)
                if self.kind == "flagship":
                    d = {("s_" + k): v
                         for k, v in self.model.strategic_policy(states, navt).items()}
                else:
                    v0t = torch.tensor([float(v0)], dtype=torch.float32,
                                       device=self.device)
                    d = self.model(fw, nav_cmd=navt, v0=v0t, steps=2)
                for k in ("s_route_logits", "route_logits", "route_head"):
                    v = d.get(k)
                    if v is not None and hasattr(v, "shape") and v.ndim == 2:
                        out[int(nav)] = [round(float(x), 5)
                                         for x in v[0].float().cpu().numpy()]
                        key = k
                        break
        return out, key


def build_window(canon, pose, torch, stack_fn, device):
    """poses ``pose-9..pose`` -> ``[1,8,9,256,256]`` float in [0,1]."""
    idx = [pose - (NEED - 1) + i for i in range(NEED)]
    if any(i not in canon for i in idx):
        return None
    vid = torch.stack([canon[i] for i in idx])            # [10,3,256,256] uint8
    st = stack_fn(vid, STACK)                             # [8,9,256,256]
    fw = st[-WINDOW:][None].to(device).float().div_(255.0)
    if tuple(fw.shape) != (1, WINDOW, 9, 256, 256):
        raise RuntimeError(f"raster assertion failed: {tuple(fw.shape)}")
    return fw


# --------------------------------------------------------------------------- #
def run_scene(sid, label, arms, root, members_root, torch, crop_fn, stack_fn,
              nav_fn, max_poses_per_event=None):
    """-> ``{condition -> {arm -> [ticks]}}`` plus a provenance block."""
    members = members_root / sid
    mp4 = root / sid / f"{CAM}.mp4"
    dl = fetch_mp4(sid, mp4)

    per_pose = label["per_pose"]
    adm = [p for p in per_pose if p.get("admissible")]
    # a window needs 9 poses of history
    adm = [p for p in adm if p["pose"] >= NEED - 1]
    if max_poses_per_event:
        keep, seen = [], {}
        for p in sorted(adm, key=lambda q: q["dist_to_decision_point_m"]):
            c = seen.get(p["event_id"], 0)
            if c < max_poses_per_event:
                keep.append(p)
                seen[p["event_id"]] = c + 1
        adm = sorted(keep, key=lambda q: q["pose"])
    if not adm:
        return None, {"scene": sid, "skip": "no admissible pose with enough history"}

    need = set()
    for p in adm:
        need.update(range(p["pose"] - (NEED - 1), p["pose"] + 1))
    P, yaw, ts = pose_track(members)
    canon, meta = canon_frames_for_poses(mp4, members, need, arms["_intr"], torch,
                                         crop_fn, max_pose=len(P) - 1)
    gtp = gtp_xyv(P, yaw, ts)
    nav_oracle = {p["pose"]: nav_fn(gtp, p["pose"]) for p in adm}

    rows, n_skipped, keys = [], 0, {}
    for p in adm:
        pose = p["pose"]
        fw = build_window(canon, pose, torch, stack_fn, arms["_device"])
        if fw is None:
            n_skipped += 1
            continue
        v0 = float(gtp[pose, 3])
        row = {"i_gt": pose, "event_id": p["event_id"],
               "dist_m": p["dist_to_decision_point_m"],
               "nav_oracle": int(nav_oracle[pose]), "v0": round(v0, 3), "sweep": {}}
        for aname, arm in arms.items():
            if aname.startswith("_"):
                continue
            sw, key = arm.route_logits_nav_sweep(fw, v0)
            row["sweep"][aname] = sw
            keys[aname] = key
        rows.append(row)

    prov = {"scene": sid, "download": dl,
            "n_admissible_poses_scored": len(rows),
            "n_poses_skipped_no_history": n_skipped,
            "route_logit_key_resolved": keys,
            "nav_oracle_distribution": {NAV_NAMES[k]: int(sum(
                1 for v in nav_oracle.values() if v == k)) for k in range(4)},
            **meta}
    return rows, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(HERE / "results" / "strategic_gt_t1"))
    ap.add_argument("--members-root", default=str(HERE / "scene_members"))
    ap.add_argument("--scene-root",
                    default="/home/nvidia/nurec_scenes/sample_set/26.04_release")
    ap.add_argument("--flagship", default="/home/nvidia/models/flagship-v1-speedjerk/ckpt.pt")
    ap.add_argument("--refc", default="/home/nvidia/models/refc-base/ckpt.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None,
                    help="per-scene tick JSONs. RESUMABLE: a scene already present is "
                         "skipped. Default <out>.parts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-poses-per-event", type=int, default=25)
    ap.add_argument("--stack", default="/home/nvidia/tanitad_cl/stack")
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    sys.path.insert(0, str(Path(a.stack) / "scripts"))
    import torch
    from tanitad.data.calib import F_REF, FThetaIntrinsics, ftheta_crop_resize
    from tanitad.data.comma2k19 import stack_frames
    from refb_labels import nav_command_v21
    from nurec_loader import RigTrajectories

    SO = _taniteval()
    reports = SO.load_label_reports(a.labels)
    scenes = [s for s in reports if not s.startswith("_")]
    with_branch = [s for s in scenes
                   if any(e.get("SCOREABLE") for e in reports[s].get("events", []))]
    with_branch.sort()
    if a.limit:
        with_branch = with_branch[:a.limit]
    print(f"{len(scenes)} admissible labels, {len(with_branch)} with a SCOREABLE branch",
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = {"flagship-v1": Arm("flagship-v1", a.flagship, "flagship", device),
            "refc-base": Arm("refc-base", a.refc, "refc", device),
            "_device": device}

    def nav_fn(gtp, i):
        t = torch.from_numpy(gtp).float()
        try:
            nav, valid = nav_command_v21(t, int(i))
            if valid:
                return int(nav)
        except Exception:                                     # noqa: BLE001
            pass
        try:
            h = int(min(60, max(10, gtp.shape[0] - int(i) - 2)))
            nav, _ = nav_command_v21(t, int(i), horizon_steps=h, min_steps=10)
            return int(nav)
        except Exception:                                     # noqa: BLE001
            return 0

    out = {"tool": "score_t1_strategic.py", "evidence_class": "MEASURED",
           "mode": "OPEN-LOOP on the LOGGED clipgt track, REAL 4K reference camera",
           "arms": {k: {"ckpt": (a.flagship if k == "flagship-v1" else a.refc),
                        "step": getattr(v, "step", None)}
                    for k, v in arms.items() if not k.startswith("_")},
           "scenes": {}, "errors": []}

    # ⛔ PER-SCENE CHECKPOINTS. Written because this run was lost once: the host rebooted
    # ~22 scenes in, and because the ticks were only serialised at the END, ~18 minutes of
    # GPU work vanished with the process. A scene already on disk is now SKIPPED, so a
    # restart resumes instead of repeating. (The mp4 cache lives under /home/nvidia — the
    # local nvme — precisely because Thor's /tmp is tmpfs and does not survive a reboot.)
    ck = Path(a.ckpt_dir or (a.out + ".parts"))
    ck.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_ticks = {}
    for p in sorted(ck.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            all_ticks[d["scene"]] = d["rows"]
            out["scenes"][d["scene"]] = d["prov"]
        except Exception:                                     # noqa: BLE001, S110
            pass
    if all_ticks:
        print(f"resuming: {len(all_ticks)} scenes already checkpointed in {ck}", flush=True)
    for i, sid in enumerate(with_branch, 1):
        if sid in all_ticks:
            continue
        try:
            members = Path(a.members_root) / sid
            rig = RigTrajectories(members / "rig_trajectories.json")
            c = rig.camera(CAM)
            arms["_intr"] = FThetaIntrinsics(
                poly=tuple(c.angle_to_pixeldist_poly), cx=c.cx, cy=c.cy,
                width=c.width, height=c.height, per_clip=True)
            tk, prov = run_scene(sid, reports[sid], arms, Path(a.scene_root),
                                 Path(a.members_root), torch, ftheta_crop_resize,
                                 stack_frames, nav_fn, a.max_poses_per_event)
            if tk is None:
                out["errors"].append(prov)
                continue
            f_eff = float(ftheta_crop_resize.last_f_eff)
            prov["f_eff"] = round(f_eff, 2)
            prov["f_eff_ok"] = bool(abs(f_eff - F_REF) < 8.0)
            if not prov["f_eff_ok"]:
                out["errors"].append({"scene": sid,
                                      "skip": f"f_eff {f_eff:.2f} != F_REF {F_REF}"})
                continue
            all_ticks[sid] = tk
            out["scenes"][sid] = prov
            (ck / f"{sid}.json").write_text(json.dumps(
                {"scene": sid, "prov": prov, "rows": tk}, default=str))
            print(f"  [{i}/{len(with_branch)}] {sid[:8]} "
                  f"poses={prov['n_admissible_poses_scored']} f_eff={f_eff:.1f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:                                # noqa: BLE001
            out["errors"].append({"scene": sid, "error": f"{type(e).__name__}: {e}",
                                  "tb": traceback.format_exc()[-800:]})
            print(f"  [{i}] {sid[:8]} ERROR {type(e).__name__}: {e}", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    out["n_scenes_with_ticks"] = len(all_ticks)
    Path(a.out).write_text(json.dumps({"provenance": out, "ticks": all_ticks},
                                      indent=1, default=str))
    print(f"wrote {a.out}  ({len(all_ticks)} scenes, {out['seconds']}s, "
          f"{len(out['errors'])} errors)")


if __name__ == "__main__":
    main()
