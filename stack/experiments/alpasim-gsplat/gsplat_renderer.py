"""gsplat-backed NuRec renderer — the render core behind our AlpaSim sensorsim service.

WHY THIS EXISTS
---------------
AlpaSim's stock renderer is NVIDIA's closed `nre-ga` container, which is amd64-only.
`stack/experiments/nurec-gsplat/FINDINGS.md` established (MEASURED 2026-08-02/03, Thor
aarch64 sm_110) that `volume.nurec` is gzip+MessagePack and that gsplat 1.5.3 reproduces
the scene, validated by a **gradient-NCC negative control** (correct frame 0.3802 vs best
wrong 0.2110). This module turns that one-shot probe into a *reusable, GPU-resident*
renderer that can be driven from an arbitrary camera pose — which is what a closed loop
needs and what a one-shot probe is not.

DESIGN
------
* Scene tensors are uploaded to the GPU **once** and reused for every frame. The probe
  rebuilt them per frame (numpy -> GPU) which dominated its wall clock.
* **Front camera only** (the PI's explicit steer). The rig has 6 cameras; rendering the
  others is the dominant cost and buys nothing for a front-camera policy.
* The camera model is the scene's own **f-theta** (gsplat has native ftheta support —
  `FThetaCameraDistortionParameters`), so frames are the same projection TanitAD's
  `ftheta_crop_resize` canonicalization expects. No pinhole approximation anywhere.
* Appearance basis defaults to `f0` (the leading Fourier feature) — the basis the
  validated run used. With `f0` the SH block is time-invariant, so it is computed once.

⛔ NOT DONE HERE (documented, not hidden):
* Sky env-map compositing is OFF. MEASURED in FINDINGS: `--sky` floods the ~49 % of
  pixels no gaussian covers and makes the render *worse*.
* Per-frame ISP is not applied — REFUTED as a residual source (`per_frame_ppisp_enabled`
  is false; total effect 0.18 %).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

CAM_FRONT = "camera_front_wide_120fov"


# --------------------------------------------------------------------------------- #
# small SE(3) helpers (numpy, float64)                                               #
# --------------------------------------------------------------------------------- #
def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def translate(x: float, y: float, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def yaw_of(T: np.ndarray) -> float:
    """Yaw of a rig->world transform (rig +x is forward)."""
    return math.atan2(T[1, 0], T[0, 0])


# --------------------------------------------------------------------------------- #
# dynamic actor tracks (from the USDZ's sequence_tracks.json)                        #
# --------------------------------------------------------------------------------- #
class ActorTracks:
    """Per-track world poses of the scene's annotated agents.

    `sequence_tracks.json` (extracted from the scene USDZ) carries, per chunk:
      tracks_data.tracks_id       [n_tracks] string ids
      tracks_data.tracks_poses    [n_tracks][n_pose][7]  (x,y,z, qx,qy,qz,qw)
    plus (optionally) per-track timestamps and cuboid dimensions. Everything is read
    defensively: a missing optional key degrades a feature, never the whole loader.
    """

    def __init__(self, path: str | Path):
        d = json.load(open(path))
        chunk = d[next(iter(d))]
        td = chunk["tracks_data"]
        self.raw_chunk_keys = sorted(chunk.keys())
        self.track_keys = sorted(td.keys())
        self.ids: list[str] = [str(t) for t in td["tracks_id"]]
        self.poses: list[np.ndarray] = [np.asarray(p, np.float64) for p in td["tracks_poses"]]
        # optional metadata (names differ between releases -> probe, never assume)
        self.timestamps = None
        for k in ("tracks_timestamps_us", "tracks_timestamps", "tracks_poses_timestamps"):
            if k in td:
                self.timestamps = [np.asarray(t, np.float64) for t in td[k]]
                break
        self.dims = None
        for k in ("tracks_dimensions", "tracks_size", "tracks_bbox_dimensions", "tracks_dims"):
            if k in td:
                self.dims = [np.asarray(x, np.float64) for x in td[k]]
                break
        self.labels = None
        for k in ("tracks_label_class", "tracks_label", "tracks_labels", "tracks_type"):
            if k in td:
                self.labels = [str(x) for x in td[k]]
                break

    def time_range(self, i: int):
        if self.timestamps is None or self.timestamps[i].size == 0:
            return None
        return float(self.timestamps[i][0]), float(self.timestamps[i][-1])

    def pose_at_time(self, i: int, t_us: float, tol_us: float = 1.5e5):
        """4x4 world pose of track `i` at wall time `t_us`, or None if the track is not
        alive then. Nearest annotation within `tol_us` (annotations are ~10 Hz)."""
        if self.timestamps is None:
            return None
        ts = self.timestamps[i]
        if ts.size == 0:
            return None
        if t_us < ts[0] - tol_us or t_us > ts[-1] + tol_us:
            return None
        k = int(np.argmin(np.abs(ts - float(t_us))))
        if abs(ts[k] - float(t_us)) > tol_us:
            return None
        return self.pose_at_index(i, k)

    def __len__(self) -> int:
        return len(self.ids)

    def pose_at(self, i: int, frac: float) -> Optional[np.ndarray]:
        """4x4 world pose of track `i` at normalised time `frac` in [0,1] of its own
        pose sequence. Returns None if the track has no poses."""
        p = self.poses[i]
        if p.size == 0:
            return None
        n = p.shape[0]
        idx = int(round(float(np.clip(frac, 0.0, 1.0)) * (n - 1)))
        row = p[idx]
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_R([row[6], row[3], row[4], row[5]])
        T[:3, 3] = row[:3]
        return T

    def pose_at_index(self, i: int, k: int) -> Optional[np.ndarray]:
        p = self.poses[i]
        if p.size == 0:
            return None
        k = int(np.clip(k, 0, p.shape[0] - 1))
        row = p[k]
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_R([row[6], row[3], row[4], row[5]])
        T[:3, 3] = row[:3]
        return T


# --------------------------------------------------------------------------------- #
# the renderer                                                                       #
# --------------------------------------------------------------------------------- #
class NuRecGsplatRenderer:
    """GPU-resident gaussian scene + f-theta front camera.

    Usage:
        r = NuRecGsplatRenderer(scene_dir)
        img = r.render(cam_to_nre)          # HxWx3 uint8, native f-theta
    """

    def __init__(self, scene_dir: str | Path, cam: str = CAM_FRONT,
                 layers: Sequence[str] = ("background", "road"),
                 quat_layout: str = "wxyz", basis: str = "f0",
                 device: str = "cuda", loader_dir: str | None = None,
                 verbose: bool = True):
        import sys
        if loader_dir:
            sys.path.insert(0, str(loader_dir))
        import torch
        from nurec_loader import (NuRecScene, RigTrajectories, K_for,  # noqa
                                  ftheta_coeffs_for, read_volume_nurec)

        self.torch = torch
        self.device = device
        self.scene_dir = Path(scene_dir)
        self.cam_name = cam
        self.layers = list(layers)
        self.basis_kind = basis
        self._verbose = verbose

        t0 = time.time()
        nre = read_volume_nurec(self.scene_dir / "volume.nurec")
        self.scene = NuRecScene(nre, quat_layout=quat_layout)
        self.rig = RigTrajectories(self.scene_dir / "rig_trajectories.json")
        self.cam = self.rig.camera(cam)
        self._K_for, self._ftheta_coeffs_for = K_for, ftheta_coeffs_for
        self.load_s = time.time() - t0

        ex = self.scene.sd[".gaussians_nodes.background.time_embed._extra_state"]
        self.t_lo = int(ex["timestamps_us_min"])
        self.t_hi = int(ex["timestamps_us_max"])

        # ---- upload every layer's activated gaussians ONCE -------------------------
        means, quats, scales, opac, sh = [], [], [], [], []
        self.layer_counts = {}
        for L in self.layers:
            b = self._basis(L, tau=0.0)
            g = self.scene.gaussians(L, time_basis=b)
            self.layer_counts[L] = int(g.means.shape[0])
            means.append(g.means); quats.append(g.quats); scales.append(g.scales)
            opac.append(g.opacities); sh.append(g.sh)
        cat = lambda xs: torch.from_numpy(np.concatenate(xs)).to(device)
        self.means, self.quats = cat(means), cat(quats)
        self.scales, self.opac = cat(scales), cat(opac)
        self.sh = cat(sh)
        self.n_gauss = int(self.means.shape[0])

        # slots for optionally-attached dynamic actors (see attach_actors)
        self._actor = None
        self.build_s = time.time() - t0
        if verbose:
            print(f"[rend] scene ready: {self.n_gauss} gaussians "
                  f"{self.layer_counts} in {self.build_s:.1f}s", flush=True)

    # -- appearance basis ----------------------------------------------------------
    def _basis(self, layer: str, tau: float) -> np.ndarray:
        F = self.scene.fourier_dim(layer)
        b = np.zeros(F, np.float32)
        if self.basis_kind == "f0":
            b[0] = 1.0
        elif self.basis_kind == "poly":
            for i in range(F):
                b[i] = tau ** i
        else:
            raise ValueError(f"unsupported basis {self.basis_kind!r}")
        return b

    # -- geometry ------------------------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.cam.width)

    @property
    def height(self) -> int:
        return int(self.cam.height)

    def n_frames(self) -> int:
        return self.rig.n_frames(self.cam_name)

    def gt_cam_to_nre(self, frame: int) -> np.ndarray:
        return self.rig.cam_to_nre(self.cam_name, frame, shutter=1)

    def gt_rig_to_world(self, frame: int) -> np.ndarray:
        return self.rig.T_rig_world(self.cam_name, frame, shutter=1)

    def rig_to_cam_to_nre(self, T_rig_world: np.ndarray) -> np.ndarray:
        """rig->world  =>  camera(optical)->NRE-world, the matrix gsplat inverts."""
        return self.rig.world_to_nre @ T_rig_world @ self.cam.T_sensor_rig

    def frame_timestamps_us(self, frame: int):
        return self.rig.frame_timestamps_us(self.cam_name, frame)

    def tau_of_us(self, ts_us: float) -> float:
        return (float(ts_us) - self.t_lo) / float(self.t_hi - self.t_lo)

    # -- dynamic actors ------------------------------------------------------------
    def attach_actors(self, tracks: "ActorTracks", cuboid_to_track: dict[int, int],
                      layer: str = "dynamic_rigids"):
        """Upload a dynamic layer whose gaussians are re-posed per frame.

        `cuboid_to_track` maps the layer's `gaussian_cuboid_ids` values onto indices
        into `tracks`. Placement is FALSIFIED by `falsify_actors` before use — the
        mapping is a hypothesis until a negative control says otherwise.
        """
        import torch
        g = self.scene.gaussians(layer, time_basis=self._basis(layer, 0.0))
        cid = np.frombuffer(
            self.scene.sd[f".gaussians_nodes.{layer}.gaussian_cuboid_ids"],
            dtype=np.int32).copy()
        # the loader drops non-finite gaussians; re-derive the same mask so the
        # cuboid ids stay aligned with the surviving rows.
        raw = self.scene.raw(layer)
        keep = self._finite_mask(raw)
        cid = cid[keep]
        assert cid.shape[0] == g.means.shape[0], (cid.shape, g.means.shape)
        self._actor = dict(
            means0=torch.from_numpy(g.means).to(self.device),
            quats0=torch.from_numpy(g.quats).to(self.device),
            scales=torch.from_numpy(g.scales).to(self.device),
            opac=torch.from_numpy(g.opacities).to(self.device),
            sh=torch.from_numpy(g.sh).to(self.device),
            cid=torch.from_numpy(cid.astype(np.int64)).to(self.device),
            uniq=sorted(set(int(c) for c in np.unique(cid))),
            tracks=tracks, map=dict(cuboid_to_track), layer=layer)
        return self._actor

    @staticmethod
    def _finite_mask(raw) -> np.ndarray:
        q = raw["rotations"]
        qn = np.linalg.norm(q, axis=1)
        scales = np.exp(raw["log_scales"])
        opac = 1.0 / (1.0 + np.exp(-raw["density_logits"].astype(np.float64)))
        sh = np.concatenate([raw["albedo"][:, :1, :], raw["specular"]], axis=1)
        return (np.isfinite(raw["positions"]).all(1) & np.isfinite(q).all(1)
                & np.isfinite(scales).all(1) & np.isfinite(opac)
                & np.isfinite(sh).all(axis=(1, 2)) & (qn > 1e-6))

    # -- the render ----------------------------------------------------------------
    def render(self, cam_to_nre: np.ndarray, width: int | None = None,
               height: int | None = None, tau: float | None = None,
               actor_frac: float | None = None, actor_time_us: float | None = None,
               near: float = 0.05, far: float = 2000.0):
        """Render one f-theta frame. Returns (uint8 HxWx3, alpha HxW float32, ms)."""
        import torch
        from gsplat import rasterization
        W = int(width or self.width)
        H = int(height or self.height)
        K = self._K_for(self.cam).copy()
        if W != self.width or H != self.height:
            raise ValueError(
                f"raster {W}x{H} != the camera's calibrated {self.width}x{self.height}; "
                "the f-theta polynomial is in NATIVE pixels, so a different raster "
                "silently changes the projection. Render native, resize afterwards.")
        Kt = torch.from_numpy(K[None]).to(self.device)
        vm = torch.from_numpy(np.linalg.inv(cam_to_nre)[None].astype(np.float32)).to(self.device)

        means, quats, scales, opac, sh = (self.means, self.quats, self.scales,
                                          self.opac, self.sh)
        self.last_actors_rendered = 0
        if self._actor is not None and (actor_time_us is not None or actor_frac is not None):
            am, aq, keep = self._posed_actors(actor_frac, actor_time_us)
            if am is not None:
                # keep the five arrays consistently subset — an actor that is not alive
                # at this timestamp must not be drawn at its canonical (origin) pose.
                means = torch.cat([means, am]); quats = torch.cat([quats, aq])
                scales = torch.cat([scales, self._actor["scales"][keep]])
                opac = torch.cat([opac, self._actor["opac"][keep]])
                sh = torch.cat([sh, self._actor["sh"][keep]])
                self.last_actors_rendered = int(keep.sum())

        torch.cuda.synchronize()
        t1 = time.time()
        colors, alphas, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=sh,
            viewmats=vm, Ks=Kt, width=W, height=H, sh_degree=3, packed=False,
            with_ut=True, with_eval3d=True, camera_model="ftheta",
            ftheta_coeffs=self._ftheta_coeffs_for(self.cam),
            near_plane=near, far_plane=far,
        )
        torch.cuda.synchronize()
        ms = (time.time() - t1) * 1000.0
        img = (colors[0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        alpha = alphas[0, ..., 0].float().cpu().numpy()
        return img, alpha, ms

    def _posed_actors(self, frac: float | None = None, t_us: float | None = None):
        """Apply each track's world pose at wall time `t_us` (preferred) or normalised
        `frac` to its gaussians, and express the result in the NRE frame."""
        import torch
        A = self._actor
        tracks, cmap = A["tracks"], A["map"]
        w2n = self.rig.world_to_nre
        R_all = torch.zeros((max(A["uniq"]) + 1, 3, 3), device=self.device)
        t_all = torch.zeros((max(A["uniq"]) + 1, 3), device=self.device)
        vis = torch.zeros((max(A["uniq"]) + 1,), dtype=torch.bool, device=self.device)
        any_ok = False
        for c in A["uniq"]:
            ti = cmap.get(int(c))
            if ti is None:
                continue
            T = (tracks.pose_at_time(ti, t_us) if t_us is not None
                 else tracks.pose_at(ti, frac))
            if T is None:
                continue
            Tn = w2n @ T
            R_all[c] = torch.from_numpy(Tn[:3, :3].astype(np.float32)).to(self.device)
            t_all[c] = torch.from_numpy(Tn[:3, 3].astype(np.float32)).to(self.device)
            vis[c] = True
            any_ok = True
        if not any_ok:
            return None, None, None
        cid = A["cid"]
        keep = vis[cid]
        R = R_all[cid]                                   # [N,3,3]
        m = torch.einsum("nij,nj->ni", R, A["means0"]) + t_all[cid]
        q = _quat_mul(_R_to_quat(R), A["quats0"])
        return m[keep], q[keep], keep


def _R_to_quat(R):
    """[N,3,3] -> [N,4] wxyz (branchless, numerically safe for proper rotations)."""
    import torch
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    t = 1.0 + m00 + m11 + m22
    w = torch.sqrt(t.clamp_min(1e-8)) / 2.0
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w)
    q = torch.stack([w, x, y, z], -1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_mul(a, b):
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


# --------------------------------------------------------------------------------- #
# falsifiers                                                                         #
# --------------------------------------------------------------------------------- #
def grad_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC of image gradients. FINDINGS: PSNR and plain NCC are RETRACTED on this
    night clip (both rank a WRONG frame first); grad-NCC picks the correct frame."""
    import cv2
    la = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ga = np.hypot(cv2.Sobel(la, cv2.CV_32F, 1, 0, 3), cv2.Sobel(la, cv2.CV_32F, 0, 1, 3))
    gb = np.hypot(cv2.Sobel(lb, cv2.CV_32F, 1, 0, 3), cv2.Sobel(lb, cv2.CV_32F, 0, 1, 3))
    ga = ga.ravel() - ga.mean(); gb = gb.ravel() - gb.mean()
    return float(ga @ gb / max(np.linalg.norm(ga) * np.linalg.norm(gb), 1e-12))


def read_ref_frame(mp4: str, frame: int, size_wh) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame} of {mp4}")
    img = img[:, :, ::-1]
    if (img.shape[1], img.shape[0]) != tuple(size_wh):
        img = cv2.resize(img, tuple(size_wh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)
