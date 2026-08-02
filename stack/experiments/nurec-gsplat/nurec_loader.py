"""
nurec_loader.py -- read a NVIDIA NuRec ``volume.nurec`` gaussian scene and hand it
to gsplat.

Everything in here was derived by probing the file itself (see FINDINGS.md for the
evidence behind each decision). Nothing is inherited from documentation.

FILE FORMAT (MEASURED on
  00040136-e651-4abd-991d-0655ccda9430.usdz : volume.nurec)

  volume.nurec = gzip  ->  MessagePack  ->  {"nre_data": {...}}
  nre_data has exactly 4 keys: version, model, config, state_dict.

  config  = the training config (this is where the ACTIVATIONS are declared).
  state_dict = 258 entries, values are RAW LITTLE-ENDIAN float16 BYTES with no
               header. There is no dtype/shape sibling key: the component count is
               derived from ``len(bytes) / 2 / N`` and N from the positions array.
               Payload is INLINE -- the ``clipgt-*.zarr.itar`` shard named in
               data_info.json is the *source* clip data, not the gaussians.

  Per layer L in {background, road, dynamic_rigids, dynamic_deformables}:
    .gaussians_nodes.L.positions          fp16 [N,3]     metres, NRE frame
    .gaussians_nodes.L.rotations          fp16 [N,4]     quaternion, UNNORMALISED
    .gaussians_nodes.L.scales             fp16 [N,3]     LOG scale
    .gaussians_nodes.L.densities          fp16 [N]       LOGIT opacity
    .gaussians_nodes.L.features_albedo    fp16 [N,F,3]   SH band-0 (DC), F = fourier_features_dim
    .gaussians_nodes.L.features_specular  fp16 [N,15,3]  SH bands 1..3
    .gaussians_nodes.L.camera_extra_signal fp16 [N,20]   per-gaussian semantic logits
"""

from __future__ import annotations

import gzip
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import msgpack
except ImportError as e:  # pragma: no cover
    raise SystemExit("msgpack is required: pip install msgpack") from e


LAYERS = ("background", "road", "dynamic_rigids", "dynamic_deformables")

# Band-0 spherical-harmonic constant. 3DGS stores DC such that
#   colour = 0.5 + C0 * dc            (gsplat applies the +0.5 internally)
SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------------------
def read_volume_nurec(path: str | Path) -> dict:
    """gunzip + msgpack-decode a volume.nurec. Returns the ``nre_data`` dict."""
    path = Path(path)
    with gzip.open(path, "rb") as f:
        raw = f.read()
    obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    return obj["nre_data"]


def extract_from_usdz(usdz: str | Path, names, outdir: str | Path) -> Dict[str, Path]:
    """Extract selected members of a .usdz (a STORED zip) to ``outdir``."""
    usdz, outdir = Path(usdz), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    with zipfile.ZipFile(usdz) as z:
        for n in names:
            z.extract(n, outdir)
            out[n] = outdir / n
    return out


# --------------------------------------------------------------------------------------
# gaussians
# --------------------------------------------------------------------------------------
@dataclass
class LayerGaussians:
    """Activated, render-ready gaussians for one layer, in the NRE world frame."""

    name: str
    means: np.ndarray  # [N,3] float32, metres
    quats: np.ndarray  # [N,4] float32, WXYZ, unit norm  (gsplat convention)
    scales: np.ndarray  # [N,3] float32, metres  (exp of stored log-scale)
    opacities: np.ndarray  # [N]   float32, sigmoid of stored logit
    sh: np.ndarray  # [N,16,3] float32, band 0..3; band0 = DC at the requested time
    semantics: Optional[np.ndarray] = None  # [N,20] float32 logits
    n_dropped: int = 0
    drop_reason: str = ""


class NuRecScene:
    def __init__(self, nre_data: dict, quat_layout: str = "wxyz"):
        self.nre = nre_data
        self.cfg = nre_data["config"]
        self.sd = nre_data["state_dict"]
        self.version = nre_data["version"]
        if quat_layout not in ("wxyz", "xyzw"):
            raise ValueError(quat_layout)
        self.quat_layout = quat_layout

    # -- raw access -------------------------------------------------------------------
    def _f16(self, key: str) -> np.ndarray:
        return np.frombuffer(self.sd[key], dtype=np.float16).astype(np.float32)

    def shape_of(self, key: str):
        """The file DOES carry an explicit ``<key>.shape`` list for every array
        (126 of the 258 state_dict entries are shapes). Prefer it; fall back to the
        byte-count derivation only if it is missing."""
        return self.sd.get(key + ".shape")

    def n_gaussians(self, layer: str) -> int:
        key = f".gaussians_nodes.{layer}.positions"
        shp = self.shape_of(key)
        b = len(self.sd[key])
        if b % 6:
            raise ValueError(f"{layer}: positions length {b} is not a multiple of 6")
        n_derived = b // 6
        if shp is not None:
            n = int(shp[0])
            if n != n_derived:
                raise ValueError(f"{layer}: declared N={n} but bytes imply {n_derived}")
            return n
        return n_derived

    def fourier_dim(self, layer: str) -> int:
        return self.cfg["layers"][layer]["fourier_features_dim"]

    def verify_shapes(self, layer: str) -> Dict[str, tuple]:
        """Cross-check every declared shape against the raw byte count under fp16.
        Raises if any array disagrees -- this is what proves the dtype is float16."""
        p = f".gaussians_nodes.{layer}."
        out = {}
        for a in ("positions", "rotations", "scales", "densities", "features_albedo",
                  "features_specular", "camera_extra_signal"):
            k = p + a
            if k not in self.sd:
                continue
            shp = self.shape_of(k)
            nbytes = len(self.sd[k])
            if shp is None:
                out[a] = (None, nbytes)
                continue
            n_elem = int(np.prod(shp)) if len(shp) else 1
            if n_elem * 2 != nbytes:
                raise ValueError(
                    f"{k}: shape {shp} = {n_elem} elems, but {nbytes} bytes "
                    f"({nbytes / max(n_elem,1):.3f} B/elem) -- dtype is NOT float16"
                )
            out[a] = (tuple(shp), nbytes)
        return out

    def raw(self, layer: str) -> Dict[str, np.ndarray]:
        """Un-activated arrays, reshaped. This is the ground truth of the file."""
        n = self.n_gaussians(layer)
        p = f".gaussians_nodes.{layer}."
        out = {
            "positions": self._f16(p + "positions").reshape(n, 3),
            "rotations": self._f16(p + "rotations").reshape(n, 4),
            "log_scales": self._f16(p + "scales").reshape(n, 3),
            "density_logits": self._f16(p + "densities").reshape(n),
            # [N, F, 3] -- coefficient-major, RGB innermost (MEASURED, see FINDINGS)
            "albedo": self._f16(p + "features_albedo").reshape(n, self.fourier_dim(layer), 3),
            "specular": self._f16(p + "features_specular").reshape(n, 15, 3),
        }
        ces = self.sd.get(p + "camera_extra_signal", b"")
        if len(ces):
            out["semantics"] = self._f16(p + "camera_extra_signal").reshape(n, -1)
        return out

    # -- activations ------------------------------------------------------------------
    def activations(self, layer: str) -> Dict[str, str]:
        """What the FILE declares (config), not what we assume."""
        lc = self.cfg["layers"][layer]
        return {
            "density_activation": lc["density_activation"],
            "scale_activation": lc["scale_activation"],
            "rotation_activation": lc["rotation_activation"],
            "radiance_sph_degree": lc["particle"]["radiance_sph_degree"],
            "radiance_sph_O0": lc["particle"]["radiance_sph_O0"],
            "fourier_features_dim": lc["fourier_features_dim"],
        }

    def _dc_at(self, albedo: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """albedo [N,F,3] x basis [F] -> [N,3]."""
        if basis.shape[0] != albedo.shape[1]:
            raise ValueError(f"basis dim {basis.shape[0]} != F {albedo.shape[1]}")
        return np.einsum("nfc,f->nc", albedo, basis.astype(np.float32))

    def gaussians(
        self,
        layer: str,
        time_basis: Optional[np.ndarray] = None,
        opacity_floor: float = 0.0,
        with_semantics: bool = False,
    ) -> LayerGaussians:
        """Fully activated gaussians ready for gsplat.

        ``time_basis`` : [F] weights over the Fourier/appearance features. Default is
        [1,0,0,...] i.e. the leading feature only.
        """
        r = self.raw(layer)
        n = r["positions"].shape[0]
        F = r["albedo"].shape[1]
        if time_basis is None:
            time_basis = np.zeros(F, np.float32)
            time_basis[0] = 1.0

        act = self.activations(layer)
        if act["density_activation"] != "sigmoid":
            raise NotImplementedError(f"density_activation={act['density_activation']!r}")
        if act["scale_activation"] != "exp":
            raise NotImplementedError(f"scale_activation={act['scale_activation']!r}")
        if act["rotation_activation"] != "normalize":
            raise NotImplementedError(f"rotation_activation={act['rotation_activation']!r}")

        means = r["positions"]
        q = r["rotations"]
        if self.quat_layout == "xyzw":
            q = q[:, [3, 0, 1, 2]]
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        quats = q / np.maximum(qn, 1e-12)

        scales = np.exp(r["log_scales"])
        opac = 1.0 / (1.0 + np.exp(-r["density_logits"].astype(np.float64)))
        opac = opac.astype(np.float32)

        dc = self._dc_at(r["albedo"], time_basis)  # [N,3]
        sh = np.concatenate([dc[:, None, :], r["specular"]], axis=1)  # [N,16,3]

        # ---- guard: the file contains non-finite fp16 values (positions overflow) ----
        finite = (
            np.isfinite(means).all(1)
            & np.isfinite(quats).all(1)
            & np.isfinite(scales).all(1)
            & np.isfinite(opac)
            & np.isfinite(sh).all(axis=(1, 2))
            & (qn[:, 0] > 1e-6)
        )
        keep = finite
        reason = "non-finite fp16 (position overflow) or degenerate quaternion"
        if opacity_floor > 0:
            keep = keep & (opac > opacity_floor)
            reason += f"; opacity<= {opacity_floor}"
        n_dropped = int(n - keep.sum())

        sem = r.get("semantics")
        return LayerGaussians(
            name=layer,
            means=np.ascontiguousarray(means[keep]),
            quats=np.ascontiguousarray(quats[keep]),
            scales=np.ascontiguousarray(scales[keep]),
            opacities=np.ascontiguousarray(opac[keep]),
            sh=np.ascontiguousarray(sh[keep]),
            semantics=(np.ascontiguousarray(sem[keep]) if (with_semantics and sem is not None) else None),
            n_dropped=n_dropped,
            drop_reason=reason,
        )

    # -- sky --------------------------------------------------------------------------
    def sky_cubemap(self) -> Optional[np.ndarray]:
        """[6, H, W, 3] float32 environment cubemap, or None if absent."""
        k = ".background.textures"
        if k not in self.sd:
            return None
        bg = self.cfg["background"]
        h, w = bg["height"], bg["width"]
        a = self._f16(k)
        want = 6 * h * w * 3
        if a.size != want:
            raise ValueError(f"cubemap size {a.size} != 6*{h}*{w}*3 = {want}")
        return a.reshape(6, h, w, 3)


# --------------------------------------------------------------------------------------
# quaternion helpers / self-test
# --------------------------------------------------------------------------------------
def quat_to_R(q: np.ndarray) -> np.ndarray:
    """[N,4] WXYZ unit quaternions -> [N,3,3] rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def gaussian_normals(quats_wxyz: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Surface normal of each gaussian = the axis with the SMALLEST scale."""
    R = quat_to_R(quats_wxyz)
    k = np.argmin(scales, axis=1)
    return R[np.arange(R.shape[0]), :, k]


def quat_layout_selftest(scene: NuRecScene, layer: str = "road", patch: float = 8.0,
                         n_patches: int = 40, seed: int = 0) -> Dict[str, float]:
    """Decide WXYZ vs XYZW WITHOUT using any reference image.

    The road layer is a surface. Its local normal can be estimated from the POSITIONS
    alone (PCA over a small xy patch) -- that estimate is independent of the quaternion
    convention. The convention whose gaussian normals agree with the point-cloud normal
    is the right one.

    NOTE: the mean |dot| form of this test is NEARLY DEGENERATE, because road quaternions
    are dominated by one component -- WXYZ reads that as ~identity and XYZW as ~180 deg
    about X, and BOTH leave the normal vertical (a symmetric disc has no up/down). The
    discriminating signal is the TILT: the horizontal components of the normal. Under
    WXYZ the y-tilt is built from stored component 1, under XYZW from component 3 -- two
    unrelated numbers. So we correlate the predicted tilt against the point-cloud tilt;
    the right convention correlates, the wrong one does not.
    """
    rng = np.random.default_rng(seed)
    r = scene.raw(layer)
    P, S = r["positions"], np.exp(r["log_scales"])
    Q = r["rotations"]
    Qw = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    Qx = Q[:, [3, 0, 1, 2]]
    Qx = Qx / np.maximum(np.linalg.norm(Qx, axis=1, keepdims=True), 1e-12)

    idx = rng.choice(P.shape[0], size=n_patches, replace=False)
    agree = {"wxyz": [], "xyzw": []}
    tilt_pc, tilt_q = [], {"wxyz": [], "xyzw": []}
    for i in idx:
        c = P[i]
        m = (np.abs(P[:, 0] - c[0]) < patch) & (np.abs(P[:, 1] - c[1]) < patch)
        if m.sum() < 200:
            continue
        pts = P[m]
        pts = pts - pts.mean(0)
        # smallest principal direction of the patch = plane normal
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        n_pc = vt[2]
        n_pc = n_pc * np.sign(n_pc[2] if n_pc[2] != 0 else 1.0)  # point up
        for lab, QQ in (("wxyz", Qw), ("xyzw", Qx)):
            nq = gaussian_normals(QQ[m], S[m])
            agree[lab].append(float(np.abs(nq @ n_pc).mean()))
            nq = nq * np.sign(nq[:, 2:3] + 1e-12)  # disambiguate the sign: point up
            tilt_q[lab].append(nq[:, :2].mean(0))
        tilt_pc.append(n_pc[:2])

    out = {f"absdot_{k}": float(np.mean(v)) for k, v in agree.items() if v}
    tp = np.array(tilt_pc)
    for lab in ("wxyz", "xyzw"):
        tq = np.array(tilt_q[lab])
        r_ = [float(np.corrcoef(tp[:, j], tq[:, j])[0, 1]) for j in range(2)]
        out[f"tiltcorr_{lab}"] = float(np.mean(r_))
        out[f"tiltcorr_{lab}_x"], out[f"tiltcorr_{lab}_y"] = r_[0], r_[1]
    out["n_patches_used"] = float(len(tilt_pc))
    out["winner"] = 1.0 if out["tiltcorr_wxyz"] >= out["tiltcorr_xyzw"] else 0.0
    return out


# --------------------------------------------------------------------------------------
# cameras / poses
# --------------------------------------------------------------------------------------
@dataclass
class FThetaCamera:
    name: str
    width: int
    height: int
    cx: float
    cy: float
    angle_to_pixeldist_poly: Tuple[float, ...]
    pixeldist_to_angle_poly: Tuple[float, ...]
    reference_poly: str
    max_angle: float
    linear_cde: Tuple[float, float, float]
    shutter_type: str
    T_sensor_rig: np.ndarray  # 4x4, sensor -> rig


class RigTrajectories:
    """Reader for rig_trajectories.json.

    NAMING (MEASURED, not assumed): ``T_A_B`` in this file means **A -> B**.
      * ``T_sensor_rig``  maps camera/optical -> rig
      * ``T_rig_worlds``  maps rig -> world
    Verified by putting the rig on the road surface at every frame; the inverse
    reading puts it hundreds of metres off the road. See FINDINGS.md.
    """

    def __init__(self, path: str | Path):
        self.d = json.load(open(path))
        self.entry = self.d["rig_trajectories"][0]
        self.clip = self.entry["sequence_id"]
        self.world_to_nre = np.array(self.d["world_to_nre"]["matrix"], np.float64)

    def _key(self, cam: str) -> str:
        return cam if "@" in cam else f"{cam}@{self.clip}"

    def camera_names(self):
        return sorted(k.split("@")[0] for k in self.d["camera_calibrations"])

    def camera(self, cam: str) -> FThetaCamera:
        c = self.d["camera_calibrations"][self._key(cam)]
        cm = c["camera_model"]
        if cm["type"] != "ftheta":
            raise NotImplementedError(f"camera_model.type={cm['type']!r}")
        p = cm["parameters"]
        return FThetaCamera(
            name=cam.split("@")[0],
            width=int(p["resolution"][0]),
            height=int(p["resolution"][1]),
            cx=float(p["principal_point"][0]),
            cy=float(p["principal_point"][1]),
            angle_to_pixeldist_poly=tuple(float(x) for x in p["angle_to_pixeldist_poly"]),
            pixeldist_to_angle_poly=tuple(float(x) for x in p["pixeldist_to_angle_poly"]),
            reference_poly=p["reference_poly"],
            max_angle=float(p["max_angle"]),
            linear_cde=tuple(float(x) for x in p["linear_cde"]),
            shutter_type=p["shutter_type"],
            T_sensor_rig=np.array(c["T_sensor_rig"], np.float64),
        )

    def n_frames(self, cam: str) -> int:
        return len(self.entry["cameras_frame_T_rig_worlds"][self._key(cam)])

    def frame_timestamps_us(self, cam: str, frame: int) -> Tuple[int, int]:
        """(shutter_start_us, shutter_end_us) for this frame."""
        t = self.entry["cameras_frame_timestamps_us"][self._key(cam)][frame]
        return int(t[0]), int(t[1])

    def T_rig_world(self, cam: str, frame: int, shutter: int = 1) -> np.ndarray:
        """rig -> world at shutter start (0) or shutter end (1)."""
        return np.array(self.entry["cameras_frame_T_rig_worlds"][self._key(cam)][frame][shutter], np.float64)

    def cam_to_nre(self, cam: str, frame: int, shutter: int = 1) -> np.ndarray:
        """camera(optical, x-right y-down z-forward) -> NRE world, 4x4."""
        c = self.camera(cam)
        return self.world_to_nre @ self.T_rig_world(cam, frame, shutter) @ c.T_sensor_rig

    def viewmat(self, cam: str, frame: int, shutter: int = 1) -> np.ndarray:
        """gsplat viewmat = world(NRE) -> camera."""
        return np.linalg.inv(self.cam_to_nre(cam, frame, shutter))


def K_for(cam: FThetaCamera) -> np.ndarray:
    """gsplat reads ONLY Ks[0,2] and Ks[1,2] (=cx,cy) for the ftheta model; the radial
    scale comes from the polynomial. fx/fy are set to the equivalent focal so that any
    incidental use is sane."""
    f = cam.angle_to_pixeldist_poly[1]
    return np.array([[f, 0.0, cam.cx], [0.0, f, cam.cy], [0.0, 0.0, 1.0]], np.float32)


def ftheta_coeffs_for(cam: FThetaCamera):
    """Build gsplat's FThetaCameraDistortionParameters straight from the JSON."""
    from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType

    return FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType[cam.reference_poly],
        pixeldist_to_angle_poly=tuple(cam.pixeldist_to_angle_poly),
        angle_to_pixeldist_poly=tuple(cam.angle_to_pixeldist_poly),
        max_angle=cam.max_angle,
        linear_cde=tuple(cam.linear_cde),
    )


# --------------------------------------------------------------------------------------
# sky
# --------------------------------------------------------------------------------------
def sample_cubemap(cube: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Nearest-neighbour cubemap lookup.

    ``cube`` [6,H,W,3]; ``dirs`` [...,3] in the SAME frame the cubemap was authored in.
    Face order assumed +X,-X,+Y,-Y,+Z,-Z (the OpenGL/USD order).
    """
    d = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.zeros(x.shape, np.int64)
    u = np.zeros(x.shape, np.float32)
    v = np.zeros(x.shape, np.float32)
    ma = np.maximum(np.maximum(ax, ay), az)

    m = (ax >= ay) & (ax >= az)
    face[m & (x > 0)], face[m & (x <= 0)] = 0, 1
    u[m] = np.where(x[m] > 0, -z[m], z[m]); v[m] = -y[m]
    m = (ay > ax) & (ay >= az)
    face[m & (y > 0)], face[m & (y <= 0)] = 2, 3
    u[m] = x[m]; v[m] = np.where(y[m] > 0, z[m], -z[m])
    m = (az > ax) & (az > ay)
    face[m & (z > 0)], face[m & (z <= 0)] = 4, 5
    u[m] = np.where(z[m] > 0, x[m], -x[m]); v[m] = -y[m]

    ma = np.maximum(ma, 1e-12)
    s = np.clip((u / ma + 1) * 0.5, 0, 1)
    t = np.clip((v / ma + 1) * 0.5, 0, 1)
    H, W = cube.shape[1], cube.shape[2]
    px = np.clip((s * W).astype(np.int64), 0, W - 1)
    py = np.clip((t * H).astype(np.int64), 0, H - 1)
    return cube[face, py, px]
