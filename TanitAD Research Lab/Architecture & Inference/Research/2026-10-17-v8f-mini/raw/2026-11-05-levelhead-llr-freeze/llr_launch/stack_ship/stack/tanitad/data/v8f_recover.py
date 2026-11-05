"""PREREG_RECOVER_FT (H-RECOVER-FT-1): the recovery perturbation of O's training batch -- a deviation drawn from a TRAIN own-state bank, applied to the ego state, and the GT future re-expressed from the
deviated frame with a rejoin blend. Rules live in ``PREREG_RECOVER_FT_FROZEN.md`` sections 3-5; nothing here is a free parameter (the flags carry the registered values).

WHAT IT DOES. A selected sample (probability ``p``, ONE deviation record per sample) has its ego state offset by (dv, da, dom, dy, dpsi):
  kinematic channels (v, a, omega) on every real history frame: v' = max(v + dv, 0), a' = a + da, omega' = omega + dom   (ORC's convention: a constant offset over the window);
  REBUILT  = ``D.EGO_PERTURB_REBUILT`` (p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav) PLUS ``pose_now`` (the anchor moves by the lateral and heading offset);
  the pose history moves by the speed-induced displacement (-dv x age x DT along the deviated heading) and, on a linear ramp (1 at NOW, 0 at ``RAMP_STEPS`` = 6 steps = ``max_age_s`` back), by (dy, dpsi);
  ``lab_s.rc`` is re-expressed by the frame change the DAgger swap uses; the images, ``act``, the hold ages, the speed-limit bucket and the speed-profile labels are the SAME objects.
  TARGET ``gt`` [B, 30, 2]: the logged future in the deviated frame, ``g' = Rot(-dpsi)(g - (0, dy))``; ``rejoin`` blends the first ``rejoin_steps`` knots by smoothstep from the logged future ``gt`` read as if driven from the
  deviated pose onto ``g'`` (AMENDMENT_1; knot i weight ``smoothstep((i + 1) / K)``; zero deviation gives ``gt`` exactly), ``logged`` is the pure frame change.
THE LAW: ``empirical`` draws one record uniformly from a :class:`RecoverDevTable` (own-minus-logged deviations of a TRAIN own-state bank); ``apriori`` draws the disclosed fixed law (section 3 of the card).
THE SHAPE (``dev_shape``): ``constant`` (default) offsets all F history frames by the ONE (dv, da, dom) of the record, as above. ``pertick`` offsets history frame k by the record's own per-tick deviation (dv_t[k], da_t[k],
dom_t[k]), the own-minus-logged difference at that clock row, so the history carries the shape the deviation grew with. It needs the empirical law and a table built with ``per_tick=True`` (columns ``dv_t, da_t, dom_t``
float64 [N, F], oldest row first, row F - 1 = NOW = the constant law's value, bit for bit). v' = max(v + dv_t, 0) per frame, so the speed change ``dv_eff_t = v' - v`` is read per frame; the history row at age q moves by
-(sum over ages 0 .. q - 1 of dv_eff) x DT along the deviated heading, its yaw by -(sum over the same ages of dom) x DT, and nav channel 3 by the same cumulative displacement / 100. dy, dpsi, the NOW kinematics, the
target and the draw (same stream, same record index) are the ``constant`` shape's.
DETERMINISM: a private generator seeded from a domain-separated hash of (seed, step); the global RNG is never touched. ``p == 0`` returns the SAME batch, no draw, no copy.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from tanitad.data import v8f_mini_data as D

RECOVER_DEFAULTS = {"p": 0.0, "dev_law": "empirical", "target": "rejoin", "rejoin_steps": 10, "seed": 0, "dev_shape": "constant"}
DEV_LAWS = ("empirical", "apriori")
DEV_SHAPES = ("constant", "pertick")
TARGETS = ("rejoin", "logged")
RAMP_STEPS = 6                       # = max_age_s: the lateral / heading offset develops over the S hold period
#: the batch tensors this lever rebuilds / re-expresses; ``pose_now`` and ``gt`` are the two it adds to ORC's partition
RECOVER_REBUILT = D.EGO_PERTURB_REBUILT + ("pose_now",)
RECOVER_TARGETS = ("gt", "lab_s.rc")
APRIORI = {"dv": (-3.0, 0.0), "da": (-1.0, 0.5), "dy_std": 0.5, "dpsi_std_deg": 3.0, "dom_std_deg_s": 2.0}   # the card's a-priori law, fixed before any data
TABLE_FIELDS = ("dv", "da", "dom", "dy", "dpsi")
PER_TICK_FIELDS = ("dv_t", "da_t", "dom_t")          # the optional per-tick columns [N, F], oldest row first (row F - 1 = NOW)


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _om_profile(psi, dt):
    """Yaw rate per history frame from a heading history ``psi`` [n, F]: wrap(psi[k] - psi[k - 1]) / dt for k >= 1 and row 0 := row 1 (the first frame has no earlier heading). The last column is the NOW yaw rate."""
    om = np.empty_like(psi)
    om[:, 1:] = _wrap(psi[:, 1:] - psi[:, :-1]) / dt
    om[:, 0] = om[:, 1]
    return om


class RecoverDevTable:
    """The empirical deviation table: ``dv, da, dom, dy, dpsi`` float64 [N] (own minus logged at the SAME clock row j), its provenance (``source_sha256`` = the bank manifest's) and a content sha256."""

    def __init__(self, cols: dict, *, source_sha256: str | None = None, path: str | None = None, counts: dict | None = None):
        miss = [k for k in TABLE_FIELDS if k not in cols]
        if miss:
            raise ValueError(f"deviation table lacks {miss}")
        self.cols = {k: np.ascontiguousarray(np.asarray(cols[k], np.float64)) for k in TABLE_FIELDS}
        n = len(self.cols["dv"])
        if any(len(v) != n for v in self.cols.values()):
            raise ValueError("deviation table columns disagree on the record count")
        if not all(np.isfinite(v).all() for v in self.cols.values()):
            raise ValueError("deviation table holds a non-finite entry")
        self.cols_t = None                                       # the optional per-tick columns (``from_dagger_bank(per_tick=True)``); None keeps the table, its counts and its hash exactly the constant table's
        have = [k for k in PER_TICK_FIELDS if k in cols]
        if have:
            if len(have) != len(PER_TICK_FIELDS):
                raise ValueError(f"per-tick columns are all or none: got {have} of {list(PER_TICK_FIELDS)}")
            ct = {k: np.ascontiguousarray(np.asarray(cols[k], np.float64)) for k in PER_TICK_FIELDS}
            if any(v.ndim != 2 or v.shape != ct["dv_t"].shape for v in ct.values()) or ct["dv_t"].shape[0] != n:
                raise ValueError(f"per-tick columns {[v.shape for v in ct.values()]} disagree with each other or with the {n} records")
            if not all(np.isfinite(v).all() for v in ct.values()):
                raise ValueError("deviation table holds a non-finite per-tick entry")
            self.cols_t = ct
        self.source_sha256, self.path, self.counts = source_sha256, path, dict(counts or {})

    @property
    def has_per_tick(self) -> bool:
        return self.cols_t is not None

    @property
    def ticks(self):
        """F, the number of history frames the per-tick columns span (None without them)."""
        return None if self.cols_t is None else int(self.cols_t["dv_t"].shape[1])

    def __len__(self):
        return len(self.cols["dv"])

    def content_sha256(self) -> str:
        h = hashlib.sha256()
        for k in TABLE_FIELDS:
            a = self.cols[k]
            h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        if self.cols_t is not None:                                # the per-tick columns are part of the content only when present
            for k in PER_TICK_FIELDS:
                a = self.cols_t[k]
                h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        return h.hexdigest()

    def quantiles(self, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), abs_qs=(0.5, 0.9, 0.95, 0.99)) -> dict:
        """Per column: the signed quantiles and the quantiles of |x| (the dose a draw delivers), float64, deterministic (``np.quantile`` linear)."""
        if not len(self):
            return {}
        return {k: {"signed": {f"q{int(round(q * 100)):02d}": float(np.quantile(v, q)) for q in qs}, "abs": {f"q{int(round(q * 100)):02d}": float(np.quantile(np.abs(v), q)) for q in abs_qs}} for k, v in self.cols.items()}

    def band_coverage(self, logged_v, lo: float = 3.0, hi: float = 8.0) -> dict:
        """Coverage of the ``[lo, hi]`` m/s band where the M1 crawl excess lives, by the LOGGED speed of each table record (``logged_v`` aligned with the rows, as ``from_dagger_bank`` returns it): the count and share of
        records in the band, and the deviation quantiles of dv, dy, dpsi inside it. The draw is MARGINAL (independent of the sample's own speed), so this is the table's share of band-like states, not a conditional law."""
        v = np.asarray(logged_v, np.float64)
        if v.shape != (len(self),):
            raise ValueError(f"logged_v {v.shape} is not aligned with the {len(self)} table rows")
        m = (v >= lo) & (v <= hi)
        out = {"band_mps": [lo, hi], "n": int(len(self)), "n_in_band": int(m.sum()), "share_in_band": float(m.mean()) if len(self) else None}
        if m.any():
            out["in_band"] = {k: {"mean": float(self.cols[k][m].mean()), **{f"q{int(round(q * 100)):02d}": float(np.quantile(self.cols[k][m], q)) for q in (0.05, 0.5, 0.95)}} for k in ("dv", "dy", "dpsi")}
        return out

    def summary(self, store_v=None) -> dict:
        s = {"n": len(self), "content_sha256": self.content_sha256(), "source_sha256": self.source_sha256, **{f"mean_{k}": float(v.mean()) for k, v in self.cols.items() if len(v)},
             **{f"mean_abs_{k}": float(np.abs(v).mean()) for k, v in self.cols.items() if len(v)}, "counts": self.counts}
        if store_v is not None and len(store_v):
            s["share_logged_v_3_8"] = float(((store_v >= 3.0) & (store_v <= 8.0)).mean())
        if self.cols_t is not None and len(self):
            s["per_tick"] = {"ticks": self.ticks, **{f"mean_abs_{k}": [float(x) for x in np.abs(v).mean(0)] for k, v in self.cols_t.items()}}
        return s

    @classmethod
    def from_arrays(cls, own: dict, log: dict, **kw):
        """``own`` / ``log``: dicts of float64 [N] ``v, a, om, x, y, psi`` (the own state and the LOGGED state of the same clip at the same row j). dy = the own offset in the logged frame, left positive."""
        dx, dyw = own["x"] - log["x"], own["y"] - log["y"]
        dy = -np.sin(log["psi"]) * dx + np.cos(log["psi"]) * dyw
        return cls({"dv": own["v"] - log["v"], "da": own["a"] - log["a"], "dom": own["om"] - log["om"], "dy": dy, "dpsi": _wrap(own["psi"] - log["psi"])}, **kw)

    @classmethod
    def from_dagger_bank(cls, bank, store, J, *, dt: float = D.DT, per_tick: bool = False):
        """Build the table from a loaded :class:`DaggerBank` (reason == ok records) and the trainer's own ClipStore ``store`` + join ``J`` (the LOGGED state at the record's clip and row j). A record whose logged
        row j or j - 1 is not a real kinematic row is dropped and COUNTED. Needs the TRAIN data (box 2); the unit tests drive it with the synthetic world.

        ``per_tick=True`` ALSO builds ``dv_t, da_t, dom_t`` [N, F] (F = the bank's hmax, oldest row first, row F - 1 = the clock row j): the own value at each of the history rows j - F + 1 .. j (v, a from the bank's ``p10f``
        channels 0 and 1, the NOW row from the bank's float64 ``kin3`` so that the NOW column equals ``dv, da`` bit for bit; yaw rate = wrap(psi[k] - psi[k - 1]) / dt on the bank's pose, row 0 := row 1) minus the LOGGED value at
        the same clip rows (``kin3`` columns 0 and 1; the same yaw-difference rule on the logged pose, which needs rows j - F .. j, i.e. j >= F). A record with a non-finite per-tick entry, or j < F, is dropped and COUNTED
        (``dropped_per_tick_row_unreal``). With ``per_tick=False`` the table, its counts and its hash are exactly the earlier table's."""
        a = bank.a
        ok = np.nonzero(np.asarray(a["reason"]) == 0)[0]
        own_v, own_a = np.asarray(a["kin3"])[ok, 0], np.asarray(a["kin3"])[ok, 1]
        P = np.asarray(a["pose"], np.float64)[ok]                                      # [n, hmax, 3] own world pose history, last row = j
        own_om = _wrap(P[:, -1, 2] - P[:, -2, 2]) / dt
        ci, j = bank.ci[ok], np.asarray(a["j"])[ok].astype(np.int64)
        cols = {k: np.full(len(ok), np.nan) for k in ("v", "a", "om", "x", "y", "psi")}
        names = J["clip"].tolist()
        read = getattr(store, "get_kin", None) or store.get          # the table reads only ``pose`` and ``kin3``: ``ClipStore.get_kin`` reads just those (bit-identical to ``get``'s; the image-feature grid is not read); a store without it keeps ``get``
        for c in np.unique(ci):
            d = read(str(names[int(c)]))
            m = np.nonzero(ci == c)[0]
            jj = j[m]
            good = (jj >= 2) & (jj < len(d["pose"])) & np.isfinite(np.asarray(d["kin3"])[np.clip(jj, 0, len(d["pose"]) - 1)]).all(1)
            m, jj = m[good], jj[good]
            pg = np.asarray(d["pose"], np.float64)
            cols["v"][m], cols["a"][m] = np.asarray(d["kin3"])[jj, 0], np.asarray(d["kin3"])[jj, 1]
            cols["om"][m] = _wrap(pg[jj, 2] - pg[jj - 1, 2]) / dt
            cols["x"][m], cols["y"][m], cols["psi"][m] = pg[jj, 0], pg[jj, 1], pg[jj, 2]
        keep = np.isfinite(np.stack(list(cols.values()) + [own_v, own_a, own_om, P[:, -1, 0], P[:, -1, 1], P[:, -1, 2]], 1)).all(1)
        if per_tick:
            return cls._with_per_tick(bank, read, names, ok, ci, j, P, (own_v, own_a, own_om), cols, keep, dt)
        own = {"v": own_v[keep], "a": own_a[keep], "om": own_om[keep], "x": P[keep, -1, 0], "y": P[keep, -1, 1], "psi": P[keep, -1, 2]}
        log = {k: v[keep] for k, v in cols.items()}
        counts = {"records_ok": int(len(ok)), "kept": int(keep.sum()), "dropped_logged_row_unreal": int((~keep).sum())}
        return cls.from_arrays(own, log, source_sha256=getattr(bank, "sha256", None), path=getattr(bank, "path", None), counts=counts), log["v"]

    @classmethod
    def _with_per_tick(cls, bank, read, names, ok, ci, j, P, own_now, cols, keep, dt):
        """The ``per_tick=True`` half of ``from_dagger_bank``: the five NOW columns of the records both rules keep, plus the [N, F] profiles (see ``from_dagger_bank``)."""
        own_v, own_a, own_om = own_now
        F, n = int(P.shape[1]), len(ok)
        if F < 2:
            raise ValueError(f"per-tick columns need at least 2 history frames; the bank holds {F}")
        p10f = np.asarray(bank.a["p10f"])[ok]                                          # [n, F, 8] own P10 features, NaN before the clip start
        own_vt, own_at = p10f[:, :, 0].astype(np.float64), p10f[:, :, 1].astype(np.float64)
        own_vt[:, -1], own_at[:, -1] = own_v, own_a                                    # the NOW column is the bank's float64 kin3 (the constant law's operands), so dv_t[:, -1] == dv bit for bit
        own_omt = _om_profile(P[:, :, 2], dt)
        log_vt, log_at, log_omt = (np.full((n, F), np.nan) for _ in range(3))
        for c in np.unique(ci):
            m = np.nonzero(ci == c)[0]
            d = read(str(names[int(c)]))
            kin, pg = np.asarray(d["kin3"]), np.asarray(d["pose"], np.float64)
            jj = j[m]
            sub = (jj >= F) & (jj < len(pg)) & (jj < len(kin))                         # the clip rows j - F .. j must exist
            m, jj = m[sub], jj[sub]
            if not len(m):
                continue
            rows = jj[:, None] - (F - 1) + np.arange(F)[None]                          # [s, F] the clip rows j - F + 1 .. j
            log_vt[m], log_at[m] = kin[rows, 0], kin[rows, 1]
            om = _om_profile(pg[rows, 2], dt)
            om[~np.isfinite(pg[jj - F, 2])] = np.nan                                   # row j - F must be real too
            log_omt[m] = om
        fin = keep & np.isfinite(np.concatenate([own_vt, own_at, own_omt, log_vt, log_at, log_omt], 1)).all(1)
        own = {"v": own_v[fin], "a": own_a[fin], "om": own_om[fin], "x": P[fin, -1, 0], "y": P[fin, -1, 1], "psi": P[fin, -1, 2]}
        log = {k: v[fin] for k, v in cols.items()}
        counts = {"records_ok": int(n), "kept": int(fin.sum()), "dropped_logged_row_unreal": int((~keep).sum()), "dropped_per_tick_row_unreal": int((keep & ~fin).sum())}
        now = cls.from_arrays(own, log, source_sha256=getattr(bank, "sha256", None), path=getattr(bank, "path", None), counts=counts)
        prof = {"dv_t": (own_vt - log_vt)[fin], "da_t": (own_at - log_at)[fin], "dom_t": (own_omt - log_omt)[fin]}
        return cls({**now.cols, **prof}, source_sha256=now.source_sha256, path=now.path, counts=counts), log["v"]


def rejoin_target(gt, dy, dpsi, *, mode: str = "rejoin", steps: int = 10):
    """The GT future ``gt`` [N, 30, 2] (NOW frame) in the deviated frame, optionally blended (AMENDMENT_1). float64 in, float64 out.

    ``g' = Rot(-dpsi)(gt - (0, dy))`` is the pure frame change (``mode='logged'``). ``mode='rejoin'`` blends the first ``steps`` knots from the logged future ``gt`` read as if driven from the deviated pose
    onto ``g'``: ``target_i = (1 - w_i) gt_i + w_i g'_i`` with ``w_i = smoothstep((i + 1) / K)``, and ``target_i = g'_i`` for ``i >= K``. Written as ``gt + w (g' - gt)`` (the same algebra) so that zero
    deviation is EXACTLY ``gt`` (``g' - gt`` is exactly 0.0, and 0.0 x w + gt is ``gt``): the binding known answer of AMENDMENT_1. Knots with ``w == 1`` are ``g'`` bit for bit."""
    import torch
    c, s = torch.cos(dpsi)[:, None], torch.sin(dpsi)[:, None]
    gx, gy = gt[..., 0], gt[..., 1] - dy[:, None]
    gp = torch.stack([c * gx + s * gy, -s * gx + c * gy], -1)                          # Rot(-dpsi)
    if mode == "logged":
        return gp
    K = int(steps)
    w = torch.ones(gt.shape[1], dtype=gt.dtype, device=gt.device)
    x = (torch.arange(K, dtype=gt.dtype, device=gt.device) + 1.0) / K
    w[:K] = smoothstep(x)
    return torch.where((w >= 1.0)[None, :, None], gp, gt + w[None, :, None] * (gp - gt))


class RecoverPerturb:
    PROBE_STEP = -1

    def __init__(self, p=0.0, dev_law="empirical", table: RecoverDevTable | None = None, target="rejoin", rejoin_steps=10, seed=0, kin_mode="yaw", bank_path=None, bank_sha256=None, dev_shape="constant"):
        self.p, self.dev_law, self.table, self.target = float(p), str(dev_law), table, str(target)
        self.dev_shape = str(dev_shape)
        self.rejoin_steps, self.seed, self.kin_mode = int(rejoin_steps), int(seed), str(kin_mode)
        self.bank_path, self.bank_sha256 = (None if bank_path is None else str(bank_path)), (None if bank_sha256 is None else str(bank_sha256).lower())
        self.last = None

    @property
    def active(self) -> bool:
        return self.p > 0.0

    def built_levers(self) -> dict:
        """What the BUILT perturber applies (the G-DVB 'built' side): the bank sha256 is the LOADED table's source, not the declared string."""
        return {"recover_p": self.p, "recover_dev_law": self.dev_law, "recover_dev_shape": self.dev_shape, "recover_dev_bank": self.bank_path,
                "recover_dev_bank_sha256": (None if self.table is None else (None if self.table.source_sha256 is None else str(self.table.source_sha256).lower())),
                "recover_target": self.target, "recover_rejoin_steps": self.rejoin_steps, "recover_seed": self.seed, "kin_mode": self.kin_mode,
                "rebuilt": list(RECOVER_REBUILT), "retargeted": list(RECOVER_TARGETS)}

    def problems(self) -> list:
        bad, fin = [], math.isfinite
        if not (fin(self.p) and 0.0 <= self.p <= 1.0):
            bad.append(f"recover_p {self.p!r} is not in [0, 1]")
        if self.dev_law not in DEV_LAWS:
            bad.append(f"recover_dev_law {self.dev_law!r} is not one of {DEV_LAWS}")
        if self.dev_shape not in DEV_SHAPES:
            bad.append(f"recover_dev_shape {self.dev_shape!r} is not one of {DEV_SHAPES}")
        if self.target not in TARGETS:
            bad.append(f"recover_target {self.target!r} is not one of {TARGETS}")
        if not 1 <= self.rejoin_steps <= 30:
            bad.append(f"recover_rejoin_steps {self.rejoin_steps!r} is not in [1, 30]")
        if self.seed < 0:
            bad.append(f"recover_seed {self.seed!r} is negative")
        if self.active:
            if self.kin_mode != "yaw":
                bad.append(f"recover_p {self.p} needs --kin-mode yaw; got {self.kin_mode!r}")
            if self.dev_law == "empirical" and (self.table is None or len(self.table) == 0):
                bad.append("recover_p > 0 with the empirical law but no (or an empty) deviation table: a lever that is declared and has nothing to draw")
            if self.dev_law == "apriori" and self.table is not None:
                bad.append("recover_dev_law apriori with a deviation bank declared: the bank would do nothing")
            if self.dev_law == "empirical" and self.bank_sha256 is None:
                bad.append("recover_p > 0 with the empirical law needs --recover-dev-bank-sha256")
            if self.dev_shape == "pertick":
                if self.dev_law == "apriori":
                    bad.append("recover_dev_shape pertick with the apriori law: the a-priori law has no per-tick profile to draw")
                if self.table is not None and not self.table.has_per_tick:
                    bad.append("recover_dev_shape pertick with a deviation table that has no per-tick columns (dv_t, da_t, dom_t): build it with per_tick=True")
        else:
            lv, dfl = self.built_levers(), RECOVER_DEFAULTS
            moved = [k for k, d in (("recover_dev_law", dfl["dev_law"]), ("recover_dev_shape", dfl["dev_shape"]), ("recover_target", dfl["target"]), ("recover_rejoin_steps", dfl["rejoin_steps"]), ("recover_seed", dfl["seed"])) if lv[k] != d]
            moved += [k for k, v in (("recover_dev_bank", self.bank_path), ("recover_dev_bank_sha256", self.bank_sha256)) if v is not None] + (["table"] if self.table is not None else [])
            if moved:
                bad.append(f"{moved} are set while recover_p is 0: a lever that is declared and does nothing")
        return bad

    def eligible(self, batch: dict):
        import torch
        return (batch["t"] >= 1) & batch["frame_ok"][:, -1] & torch.isfinite(batch["kin3"]).all(-1)

    def _seed_of(self, step: int) -> int:
        h = hashlib.sha256(f"recover/{self.seed}/{int(step)}".encode("ascii")).digest()
        return int.from_bytes(h[:8], "little") % (1 << 62)

    def _draw(self, B, g):
        import torch
        u = torch.rand(B, generator=g, dtype=torch.float64)
        r = torch.rand(B, generator=g, dtype=torch.float64)
        z = torch.randn(3, B, generator=g, dtype=torch.float64)
        w = torch.rand(2, B, generator=g, dtype=torch.float64)                            # fixed draw order: both laws consume the same stream
        if self.dev_law == "empirical":
            i = torch.clamp((r * len(self.table)).long(), max=len(self.table) - 1).numpy()
            dev = [torch.from_numpy(self.table.cols[k][i]) for k in TABLE_FIELDS]
            if self.dev_shape == "pertick":                                               # the same record index; dv, da, dom become the record's [B, F] profiles (dy, dpsi stay the NOW row's)
                dev[:3] = [torch.from_numpy(self.table.cols_t[k][i]) for k in PER_TICK_FIELDS]
        else:
            A = APRIORI
            dev = [A["dv"][0] + (A["dv"][1] - A["dv"][0]) * w[0], A["da"][0] + (A["da"][1] - A["da"][0]) * w[1], z[2] * math.radians(A["dom_std_deg_s"]), A["dy_std"] * z[0], math.radians(A["dpsi_std_deg"]) * z[1]]
        return u, dev

    def apply(self, batch: dict, step: int, *, p=None) -> dict:
        """The perturbed + re-targeted batch (a shallow copy), or ``batch`` itself when the lever is off. ``p`` overrides the probability for this call only (the gate's probes)."""
        import torch
        p = self.p if p is None else float(p)
        if not p > 0.0:
            elig = self.eligible(batch)
            self.last = {"step": int(step), "p": 0.0, "n_rows": int(elig.numel()), "n_eligible": int(elig.sum()), "n_selected": 0}
            return batch
        saved, self.p = self.p, p
        try:
            bad = self.problems()
        finally:
            self.p = saved
        if bad:
            raise ValueError("recover perturbation REFUSED: " + "; ".join(bad))
        if self.dev_shape == "pertick" and self.table.ticks != int(batch["p10f"].shape[1]):
            raise ValueError(f"recover perturbation REFUSED: the per-tick table spans {self.table.ticks} history frames but the batch holds {int(batch['p10f'].shape[1])}: a table of another hmax")
        B = int(batch["p10f"].shape[0])
        dev0 = batch["p10f"].device
        g = torch.Generator().manual_seed(self._seed_of(step))
        u, (dv, da, dom, dy, dpsi) = self._draw(B, g)
        elig = self.eligible(batch)
        sel = (u < p).to(dev0) & elig
        rows = sel.nonzero(as_tuple=True)[0]
        now = [x[:, -1] if x.dim() == 2 else x for x in (dv, da, dom, dy, dpsi)]       # the NOW-row deviation of each sample: the constant shape's own values
        self.last = {"step": int(step), "p": p, "n_rows": B, "n_eligible": int(elig.sum()), "n_selected": int(rows.numel()), "sel": sel.detach().cpu().clone(),
                     "dev": {k: torch.where(sel.cpu(), v, torch.zeros_like(v)) for k, v in zip(TABLE_FIELDS, now)}}
        if self.dev_shape == "pertick":
            self.last["dev_t"] = {k: torch.where(sel.cpu()[:, None], v, torch.zeros_like(v)) for k, v in zip(PER_TICK_FIELDS, (dv, da, dom))}
        out = dict(batch)
        if rows.numel():
            r = rows.cpu()
            out.update(self._rebuilt(batch, rows, dv[r].to(dev0), da[r].to(dev0), dom[r].to(dev0), dy[r].to(dev0), dpsi[r].to(dev0)))      # the ONE line that applies the perturbation
        return out

    def _rebuilt(self, batch, rows, dv, da, dom, dy, dpsi) -> dict:
        if self.dev_shape == "pertick":
            return self._rebuilt_pertick(batch, rows, dv, da, dom, dy, dpsi)
        import torch
        from tanitad.models.kinematic_prior import _pose_kappa0
        from tanitad.data.v8f_dagger import rc_frame_change
        f64 = torch.float64
        p10f, pose = batch["p10f"], batch["pose"]
        F, dev, N = int(p10f.shape[1]), p10f.device, int(rows.numel())
        vm = D.EgoPerturb._hist_valid(batch, rows)
        P = p10f[rows]
        v, a, om = P[..., 0].to(f64), P[..., 1].to(f64), batch["act"][rows][..., 1].to(f64)
        v2 = torch.clamp(v + dv[:, None], min=0.0)
        a2, om2 = a + da[:, None], om + dom[:, None]
        dv_eff = (v2 - v)[:, -1]                                                                   # the speed change at NOW after the v >= 0 clamp
        kin2 = torch.stack([v2, a2, _pose_kappa0(om2, v2)], -1)
        feat2 = torch.from_numpy(D._fk().kin_features(kin2.reshape(-1, 3).cpu().numpy())).reshape(N, F, 8).to(device=dev, dtype=p10f.dtype)
        kin_now = kin2[:, -1]
        new = {k: batch[k].clone() for k in ("p10f", "kin3", "p10_t", "base", "pose", "pose_t", "pose_s", "nav", "pose_now", "gt")}
        new["p10f"][rows] = torch.where(vm[..., None], feat2, P)
        new["kin3"][rows] = kin_now.to(batch["kin3"].dtype)
        new["p10_t"][rows] = feat2[:, -1]
        new["base"][rows] = torch.from_numpy(D.kdx_base(kin_now.cpu().numpy(), D.KIN_UNITS["yaw"])).to(device=dev, dtype=batch["base"].dtype)
        Pz = pose[rows].to(f64)
        age = (F - 1 - torch.arange(F, device=dev)).to(f64)[None]                                  # [1, F] steps behind NOW
        ramp = torch.clamp(1.0 - age / RAMP_STEPS, 0.0, 1.0)
        psi_dev = Pz[:, -1, 2] + dpsi                                                              # the deviated heading at NOW
        disp = -(dv_eff[:, None] * age * D.DT)                                                     # speed-induced displacement of the history rows
        off = torch.stack([-torch.sin(Pz[:, -1, 2]) * dy, torch.cos(Pz[:, -1, 2]) * dy], -1)      # the lateral offset in the world frame
        xy2 = Pz[..., :2] + disp[..., None] * torch.stack([torch.cos(psi_dev), torch.sin(psi_dev)], -1)[:, None] + ramp[..., None] * off[:, None]
        yaw2 = Pz[..., 2] + ramp * dpsi[:, None] - dom[:, None] * age * D.DT
        pose2 = torch.where(vm[..., None], torch.cat([xy2, yaw2[..., None]], -1), Pz).to(pose.dtype)
        new["pose"][rows] = pose2
        k = torch.arange(N, device=dev)
        new["pose_t"][rows] = pose2[k, F - 1 - batch["a_t"][rows]]
        new["pose_s"][rows] = pose2[k, F - 1 - batch["a_s"][rows]]
        new["pose_now"][rows] = pose2[:, -1]
        nav_r, nav_ok = batch["nav"][rows], D.EgoPerturb._nav_masks(batch, rows)[0]
        now3 = nav_r[:, -1:, 3].to(f64)
        nav3 = torch.maximum(now3, nav_r[..., 3].to(f64) + dv_eff[:, None] * age * D.DT / 100.0).clamp(0.0, 3.0)   # the distance behind NOW changes by dv x age x DT
        new["nav"][rows, :, 3] = torch.where(nav_ok, nav3.to(nav_r.dtype), nav_r[..., 3])
        new["gt"][rows] = rejoin_target(batch["gt"][rows].to(f64), dy, dpsi, mode=self.target, steps=self.rejoin_steps).to(batch["gt"].dtype)
        lab_s = dict(batch["lab_s"])
        rc = lab_s["rc"].clone()
        rc[rows] = rc_frame_change(rc[rows], lab_s["rc_ok"][rows], batch["pose_s"][rows], new["pose_s"][rows])             # the S frame moved: the route checkpoints are the same world points
        lab_s["rc"] = rc
        new["lab_s"] = lab_s
        return new

    def _rebuilt_pertick(self, batch, rows, dv, da, dom, dy, dpsi) -> dict:
        """The ``pertick`` shape of ``_rebuilt``: ``dv, da, dom`` are the records' [N, F] profiles (oldest row first, row F - 1 = NOW), ``dy, dpsi`` [N] the NOW row's. Every tensor it writes is ``_rebuilt``'s, with
        the per-frame deviation in place of the one constant: the history row at age q moves by -(sum over ages 0 .. q - 1 of dv_eff) x DT, its yaw by -(sum over the same ages of dom) x DT, nav channel 3 by the same cumulative displacement / 100."""
        import torch
        from tanitad.models.kinematic_prior import _pose_kappa0
        from tanitad.data.v8f_dagger import rc_frame_change
        f64 = torch.float64
        p10f, pose = batch["p10f"], batch["pose"]
        F, dev, N = int(p10f.shape[1]), p10f.device, int(rows.numel())
        vm = D.EgoPerturb._hist_valid(batch, rows)
        P = p10f[rows]
        v, a, om = P[..., 0].to(f64), P[..., 1].to(f64), batch["act"][rows][..., 1].to(f64)
        dv_t, da_t, dom_t = dv, da, dom                                                            # PERTICK-PROFILE-INTAKE: the [N, F] profiles of the drawn records
        v2 = torch.clamp(v + dv_t, min=0.0)
        a2, om2 = a + da_t, om + dom_t
        dv_eff_t = v2 - v                                                                          # the speed change of every history frame after the v >= 0 clamp
        kin2 = torch.stack([v2, a2, _pose_kappa0(om2, v2)], -1)
        feat2 = torch.from_numpy(D._fk().kin_features(kin2.reshape(-1, 3).cpu().numpy())).reshape(N, F, 8).to(device=dev, dtype=p10f.dtype)
        kin_now = kin2[:, -1]
        new = {k: batch[k].clone() for k in ("p10f", "kin3", "p10_t", "base", "pose", "pose_t", "pose_s", "nav", "pose_now", "gt")}
        new["p10f"][rows] = torch.where(vm[..., None], feat2, P)
        new["kin3"][rows] = kin_now.to(batch["kin3"].dtype)
        new["p10_t"][rows] = feat2[:, -1]
        new["base"][rows] = torch.from_numpy(D.kdx_base(kin_now.cpu().numpy(), D.KIN_UNITS["yaw"])).to(device=dev, dtype=batch["base"].dtype)
        Pz = pose[rows].to(f64)
        age = (F - 1 - torch.arange(F, device=dev)).to(f64)[None]                                  # [1, F] steps behind NOW
        ramp = torch.clamp(1.0 - age / RAMP_STEPS, 0.0, 1.0)

        def behind(x):                                                                             # [N, F] frame k -> the sum of x over the real frames strictly newer than k (ages 0 .. age(k) - 1)
            by_age = torch.where(vm, x, torch.zeros_like(x)).flip(1)
            return torch.cat([torch.zeros_like(by_age[:, :1]), torch.cumsum(by_age, 1)[:, :-1]], 1).flip(1)
        cum_dv, cum_dom = behind(dv_eff_t), behind(dom_t)
        psi_dev = Pz[:, -1, 2] + dpsi                                                              # the deviated heading at NOW
        disp = -(cum_dv * D.DT)                                                                    # speed-induced displacement of the history rows
        off = torch.stack([-torch.sin(Pz[:, -1, 2]) * dy, torch.cos(Pz[:, -1, 2]) * dy], -1)      # the lateral offset in the world frame
        xy2 = Pz[..., :2] + disp[..., None] * torch.stack([torch.cos(psi_dev), torch.sin(psi_dev)], -1)[:, None] + ramp[..., None] * off[:, None]
        yaw2 = Pz[..., 2] + ramp * dpsi[:, None] - cum_dom * D.DT
        pose2 = torch.where(vm[..., None], torch.cat([xy2, yaw2[..., None]], -1), Pz).to(pose.dtype)
        new["pose"][rows] = pose2
        k = torch.arange(N, device=dev)
        new["pose_t"][rows] = pose2[k, F - 1 - batch["a_t"][rows]]
        new["pose_s"][rows] = pose2[k, F - 1 - batch["a_s"][rows]]
        new["pose_now"][rows] = pose2[:, -1]
        nav_r, nav_ok = batch["nav"][rows], D.EgoPerturb._nav_masks(batch, rows)[0]
        now3 = nav_r[:, -1:, 3].to(f64)
        nav3 = torch.maximum(now3, nav_r[..., 3].to(f64) + cum_dv * D.DT / 100.0).clamp(0.0, 3.0)  # the distance behind NOW changes by the cumulative speed change x DT
        new["nav"][rows, :, 3] = torch.where(nav_ok, nav3.to(nav_r.dtype), nav_r[..., 3])
        new["gt"][rows] = rejoin_target(batch["gt"][rows].to(f64), dy, dpsi, mode=self.target, steps=self.rejoin_steps).to(batch["gt"].dtype)
        lab_s = dict(batch["lab_s"])
        rc = lab_s["rc"].clone()
        rc[rows] = rc_frame_change(rc[rows], lab_s["rc_ok"][rows], batch["pose_s"][rows], new["pose_s"][rows])             # the S frame moved: the route checkpoints are the same world points
        lab_s["rc"] = rc
        new["lab_s"] = lab_s
        return new
