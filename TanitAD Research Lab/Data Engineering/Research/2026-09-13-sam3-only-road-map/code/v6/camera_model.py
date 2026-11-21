"""One camera model for every projection in the SAM3 map pipeline: pinhole (the virtual nuPlan views) or f-theta (the
native PhysicalAI cameras), with the ground as a SMOOTH surface (ground_surface.py).

Why: the front-only pipeline cropped the 120 deg front camera to the 63.7 deg virtual pinhole view Qwen-Drive needs
(fx 1545 @ 1920 vs the native f-theta focal 932.7 px -> a 1.66x interpolated upsampling), and every projection read the
ground height per 2 m cell, which put seams through painted arrows. Measured on the PI's frame: smooth ground doubles
the frame-to-frame gradient agreement (0.060 -> 0.108) and restores the arrow shapes.

Camera frame: x right, y down, z forward. R maps camera -> rig, t is the camera centre in the rig frame.
"""
import math
import numpy as np
import ground_surface as GS


class Camera:
    def __init__(self, R, t, K=None, ftheta=None, width=1920, height=1080):
        self.R = np.asarray(R, np.float64); self.t = np.asarray(t, np.float64)
        self.W, self.H = int(width), int(height)
        self.K = None if K is None else np.asarray(K, np.float64)
        self.ft = ftheta                         # dict(poly=(p0..p4), cx, cy) for f-theta
        if self.ft is not None:
            th = np.linspace(0.0, 2.2, 4001)
            r = self._r_of_theta(th)
            keep = np.r_[True, np.diff(r) > 0]               # monotonic part of the polynomial
            last = int(np.argmax(~keep)) if (~keep).any() else len(th)
            self._th_tab, self._r_tab = th[:last], r[:last]

    @classmethod
    def from_calib(cls, c, i, img_w=1920, img_h=1080):
        """calib.npz of the virtual views (pinhole) or of build_front_seq_native.py (f-theta)."""
        if "ftheta_poly" in c.files:
            ft = {"poly": tuple(float(p) for p in c["ftheta_poly"][i]), "cx": float(c["ftheta_cx"][i]), "cy": float(c["ftheta_cy"][i])}
            return cls(c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i], ftheta=ft, width=img_w, height=img_h)
        return cls(c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i], K=c["cam_intrinsic"][i], width=img_w, height=img_h)

    def _r_of_theta(self, th):
        out = np.zeros_like(th, dtype=np.float64)
        for c in reversed(self.ft["poly"]):
            out = out * th + c
        return out

    # ---- camera-frame points -> pixels
    def project_cam(self, Pc):
        x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
        if self.ft is None:
            zs = np.where(z > 1e-6, z, 1.0)
            u = self.K[0, 0] * x / zs + self.K[0, 2]; v = self.K[1, 1] * y / zs + self.K[1, 2]
            ok = z > 0.3
        else:
            rho = np.hypot(x, y); th = np.arctan2(rho, z)
            ok = (th <= self._th_tab[-1]) & (z > 0.05)           # inside the polynomial's monotonic range, in front
            r = self._r_of_theta(th)
            k = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
            u = self.ft["cx"] + x * k; v = self.ft["cy"] + y * k
        ok &= (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, ok

    def project_rig(self, P):
        return self.project_cam((P - self.t) @ self.R)

    # ---- pixels -> unit rays in the rig frame
    def rays_rig(self, u, v):
        if self.ft is None:
            d = np.c_[(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones(len(u))]
        else:
            du, dv = u - self.ft["cx"], v - self.ft["cy"]
            r = np.hypot(du, dv)
            th = np.interp(r, self._r_tab, self._th_tab, right=np.nan)
            s = np.where(r > 1e-9, np.sin(th) / np.maximum(r, 1e-9), 0.0)
            d = np.c_[du * s, dv * s, np.cos(th)]
        return d @ self.R.T

    def lift(self, mask, sgrid, stride=2, max_xy=(30.0, 15.0)):
        """Pixels of `mask` (H x W bool, sampled at (stride i, stride j) with centres +stride/2) -> ground points (N, 2) in
        the rig frame, intersected with the SMOOTH ground (3 fixed-point iterations), and their camera ranges."""
        vv, uu = np.nonzero(mask[::stride, ::stride])
        if not len(uu):
            return np.zeros((0, 2)), np.zeros(0)
        u = uu * stride + stride / 2.0; v = vv * stride + stride / 2.0
        d = self.rays_rig(u, v)
        good = np.isfinite(d).all(axis=1) & (d[:, 2] < -1e-3)
        d = d[good]
        z = np.full(len(d), float(np.median(sgrid)))
        for _ in range(3):
            s = (z - self.t[2]) / d[:, 2]
            xy = self.t[:2] + d[:, :2] * s[:, None]
            z = GS.height(xy, sgrid)
        keep = (s > 0) & (np.abs(xy[:, 0]) <= max_xy[0]) & (np.abs(xy[:, 1]) <= max_xy[1])
        xy = xy[keep]
        return xy, np.hypot(xy[:, 0] - self.t[0], xy[:, 1] - self.t[1])


def self_test():
    """Round trip rig ground point -> pixel -> ray -> ground point, both models; must be ~0."""
    rng = np.random.default_rng(0)
    sg = np.zeros((50, 50))
    R_v = np.array([[0.0, -0.0175, 0.9998], [-1.0, 0.0, 0.0], [0.0, -0.9998, -0.0175]])
    for name, cam in (("pinhole", Camera(R_v, [1.5, 0.0, 1.35], K=[[1545, 0, 960], [0, 1545, 560], [0, 0, 1]])),
                      ("ftheta", Camera(R_v, [1.5, 0.0, 1.28], ftheta={"poly": (0.0, 932.72, 22.945, -58.501, 16.507), "cx": 960.0, "cy": 540.0}))):
        P = np.c_[rng.uniform(4, 30, 2000), rng.uniform(-12, 12, 2000), np.zeros(2000)]
        u, v, ok = cam.project_rig(P)
        d = cam.rays_rig(u[ok], v[ok])
        s = (0.0 - cam.t[2]) / d[:, 2]
        back = cam.t[:2] + d[:, :2] * s[:, None]
        err = np.hypot(*(back - P[ok, :2]).T)
        print(f"self-test {name}: n {int(ok.sum())} visible of 2000, round-trip max error {err.max():.2e} m")


if __name__ == "__main__":
    self_test()
