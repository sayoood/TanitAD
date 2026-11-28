"""E1 -- `V8FMiniT2Policy`: v8F-mini behind the T2-A kit's policy interface (the T2 re-perception rollout).

It builds the model's inference items EXACTLY as the T1-kin tool (`v8f_t1_eval_ri_lf6.py`, floor flags off) does, from a stream:
  * ego history: the 10 Hz track (x, y, yaw, v) -- the LOGGED rows up to NOW at `set_t0`, then the EXECUTED rows the harness
    hands back through `set_history`. kin3 / yaw rate / p10f / base / pose come from `D.yaw_kin_from_track`,
    `D.ego_channels_native`, `fk.kin_features`, `D.kdx_base`: the programme's one spelling, CALLED, not re-implemented.
  * nav: rows for the steps <= t0, then dead-reckoned from the OWN arc length (the tool's `--nav deadreckon`).
  * speed bucket: the CAUSAL bucket (max measured ego speed over the last 21 cache rows, `speed_max_bin`), never the logged future.
  * camera: a deque of the last `hmax` pooled-token rows on the 0.2 s grid (fp8 -> fp16 as the cache reader casts them). Banked
    tokens can be pushed directly (`push_tokens`, the reproduction gate) or the in-loop encoder fills them from frames.
It runs `forward_infer` and returns the 6 s plan (30 knots, float64, ego frame at NOW) plus the same forward's P10 prior.

TIER: this is the T2 re-perception rollout's policy. It is not T1; it never reads a logged value after NOW.
INFERENCE READS ONLY `INFERENCE_INPUT_KEYS`: the batch handed to the model is exactly `MODEL_KEYS` (the tool's allowlist) and the
keys `forward_infer` reports reading are checked against `INFERENCE_INPUT_KEYS` on every call (`KeyGuardError`).
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from collections import deque

import numpy as np

TREE_DEFAULT = "F:/TanitAD-scratch/t2v8f/tree"          # `git archive 24b97ba9 stack/tanitad taniteval/taniteval stack/scripts/v8f_t1_eval.py`
DT = 0.2
H = 30
CAUSAL_STEPS = 20
NO_READ = frozenset({"act", "pose"})                      # in INFERENCE_INPUT_KEYS, never read by forward_infer (the tool drops them)
NAV_FROM_HARNESS = {0: 0, 1: 1, 2: 2, 3: 0}               # kit NAV_NAMES (follow, left, right, straight) -> v8F token (UNVERIFIED mapping; straight collapses to follow)


class KeyGuardError(RuntimeError):
    pass


def bootstrap(tree: str | None = None):
    """Put the exported tip tree on sys.path FIRST and assert that `tanitad` resolves inside it (never the G: editable install)."""
    tree = os.path.abspath(tree or os.environ.get("T2V8F_TREE") or TREE_DEFAULT).replace("\\", "/")
    if tree.upper().startswith("G:"):
        raise RuntimeError("REFUSED: a G: tree")
    for p in (tree + "/taniteval", tree + "/stack"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    if not os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(tree):
        raise RuntimeError(f"tanitad resolved outside the tree: {tanitad.__file__}")
    return tree


def sha256_file(path, n: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def ages_for(step: int, schedule: str, k_t: int = 3, k_s: int = 6):
    """The tool's `ages_for` (sample-and-hold ages in cache steps since t0)."""
    if schedule == "fresh":
        return 0, 0
    return step % k_t, step % k_s


def causal_bucket(v, j: int, steps: int | None, speed_max_bin):
    """The tool's `causal_bucket`, verbatim: speed_max_bin of the max finite speed over rows [j - steps, j]. Reads nothing after row j."""
    lo = 0 if steps is None else max(0, j - steps)
    w = np.asarray(v[lo:j + 1], np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.zeros(4, np.float32), False
    i, _ = speed_max_bin(float(max(w.max(), 0.0)))
    oh = np.zeros(4, np.float32)
    oh[i] = 1.0
    return oh, True


class V8FMiniT2Policy:
    """Kit policy interface: `window`, `consumes_nav_known`, `canon_mode`, `canon_provenance`, `set_t0`, `set_history`, `plan`."""
    name = "v8f_mini"
    consumes_nav_known = False
    canon_mode = None                  # the in-loop encoder owns the canon (`canon_provenance` is filled from it)
    ARMS = ("model", "p10")

    def __init__(self, ckpt: str, *, encoder=None, hacc_weights=None, arm: str = "model", schedule: str = "fresh",
                 limit_mode: str = "causal", tree: str | None = None, device: str = "cpu"):
        import torch
        self.tree = bootstrap(tree)
        from tanitad.data import v8f_mini_data as D
        from tanitad.models import v8f_mini as M
        from tanitad.refs.refcv6_max_speed import speed_max_bin
        from taniteval import floors_kin as fk
        self.torch, self.D, self.M, self.fk, self.speed_max_bin = torch, D, M, fk, speed_max_bin
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}")
        if limit_mode not in ("causal", "off"):
            raise ValueError("limit_mode must be 'causal' (the T1 statement of record) or 'off'")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = M.V8FMiniConfig(**ck["cfg"])
        if self.cfg.kin_mode != "yaw":
            raise RuntimeError("REFUSED: a steer-mode checkpoint reads the recorded steering angle (PI ruling D-PI-YAWRATE-1)")
        self.model = M.V8FMini(self.cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        self.model.load_state_dict(ck["model"])                              # strict
        if hacc_weights is not None:
            self.model.load_hacc_weights(hacc_weights)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.ckpt, self.ckpt_sha256, self.arm, self.schedule, self.limit_mode = str(ckpt), None, arm, schedule, limit_mode
        self.encoder = encoder
        self.hmax, self.hist = int(self.cfg.hmax), int(self.cfg.hist)
        self.window_start = D.effective_window_start(self.hist, self.cfg.window_start)
        self.model_keys = frozenset(M.INFERENCE_INPUT_KEYS) - NO_READ
        self.window = 2 * (self.hist + self.cfg.max_age_s - 1) + 1 - 2        # kit: need = window + STACK(3) - 1 raw frames = the newest `hist + max_age_s` grid frames
        self.canon_provenance = {"mode": "in_loop_dinov3L_pool4x10_fp8", "encoder": getattr(encoder, "snapshot", None)}
        self.prov = {"ckpt": self.ckpt, "hacc": bool(self.model.hacc_active), "arm": arm, "schedule": schedule, "limit_mode": limit_mode,
                     "hist": self.hist, "hmax": self.hmax, "tier": "T2 re-perception rollout (nothing logged is read after NOW)"}
        self.reset()

    # ------------------------------------------------------------------------------------------------ state
    def reset(self):
        self.track = np.zeros((0, 4), np.float64)        # raw 10 Hz rows (x, y, yaw, v); the LAST row is NOW; parity-trimmed so NOW is an even index
        self.t0_step = None
        self.nav0 = None
        self.nav_log = None                              # logged/constant nav rows for steps <= t0 (None -> constant nav0)
        self.tok = {}                                    # step -> fp16 [40, 1024]
        self.raw_off = 0
        self.ticks = 0
        self.t0_info = None

    def set_t0(self, gt_T, f0, nav_rows=None, nav_cmd=None):
        """Called once per rollout. `gt_T` are the LOGGED rig poses; ONLY rows <= f0 are copied (the logged past of the observation
        window); nothing after f0 is ever stored. Seeds the track the way `WindowState` does (logged rows <= t0)."""
        from_h = np.stack([np.array([T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]), 0.0]) for T in gt_T[:int(f0) + 1]])
        d = np.linalg.norm(np.diff(from_h[:, :2], axis=0), axis=1) / 0.1
        from_h[:-1, 3] = d
        from_h[-1, 3] = d[-1] if len(d) else 0.0
        self.seed_track(from_h, nav_rows=nav_rows)

    def seed_track(self, rows_xyyawv, nav_rows=None):
        """The logged past: rows (x, y, yaw, v) at 10 Hz up to NOW. A single leading row is dropped when needed so NOW is an even index."""
        self.reset()
        r = np.asarray(rows_xyyawv, np.float64)
        self.raw_off = (len(r) - 1) % 2
        self.track = r[self.raw_off:].copy()
        self.t0_step = (len(self.track) - 1) // 2
        self.nav_log = None if nav_rows is None else np.asarray(nav_rows, np.float32)[:self.t0_step + 1].copy()
        self.t0_info = {"t0_step": self.t0_step, "rows": len(self.track), "parity_trim": self.raw_off}

    def set_history(self, h, n_new: int | None = None):
        """The harness' executed history (last 16 rows, NOW last). New rows are merged onto the policy's own track by a STRICT overlap check.
        `n_new` (E3 passes it) states how many rows are new; without it the overlap must be unambiguous: a stationary history
        matches at several shifts, and inventing the shift would invent time, so that case is refused."""
        h = np.asarray(h, np.float64)
        if self.t0_step is None:
            raise RuntimeError("set_history before set_t0/seed_track")
        ok = [s for s in range(len(h)) if len(h) - s <= len(self.track)
              and np.array_equal(h[:len(h) - s], self.track[len(self.track) - (len(h) - s):])]
        if n_new is not None:
            if int(n_new) not in ok:
                raise RuntimeError(f"set_history: n_new={n_new} is not consistent with the overlap (valid shifts {ok})")
            s = int(n_new)
        elif len(ok) == 1:
            s = ok[0]
        else:
            raise RuntimeError(f"set_history: {'no overlap' if not ok else 'ambiguous overlap ' + str(ok)}; pass n_new (refusing to invent rows)")
        if s:
            self.track = np.vstack([self.track, h[len(h) - s:]])
        self._odd = bool((len(self.track) - 1) % 2)

    def set_nav(self, nav_cmd=None, d_next_m=None, row=None):
        """The route input at t0: a 5-vector (one-hot token(3), d_next/100 clipped to [0, 3], valid). `row` wins; else built from the harness' discrete command."""
        if row is not None:
            self.nav0 = np.asarray(row, np.float32).copy()
            return
        tok = NAV_FROM_HARNESS[int(nav_cmd)]
        r = np.zeros(5, np.float32)
        r[tok] = 1.0
        if d_next_m is not None and tok > 0:
            r[3], r[4] = float(np.clip(d_next_m / 100.0, 0.0, 3.0)), 1.0
        self.nav0 = r

    def push_tokens(self, step: int, tok_fp8_or_fp16):
        """One 0.2 s grid frame's pooled tokens (fp8 from the encoder or a cache; stored as fp16 exactly as `ClipStore.get` casts them)."""
        t = self.torch.as_tensor(tok_fp8_or_fp16)
        self.tok[int(step)] = t.to(self.torch.float16)
        for k in [k for k in self.tok if k < int(step) - self.hmax - 8]:
            del self.tok[k]

    # ------------------------------------------------------------------------------------------------ the item
    def _steps(self):
        D, fk = self.D, self.fk
        tr = self.track
        kin, om = D.yaw_kin_from_track(tr)
        p10 = np.full((len(kin), 8), np.nan, np.float32)
        p10[1:] = fk.kin_features(kin[1:]).astype(np.float32)
        return {"kin3": kin, "p10f": p10, "base": D.kdx_base(kin, D.KIN_UNITS["yaw"]), "pose": tr[0::2], "v": tr[0::2, 3]}

    def _nav_rows(self, st, j, idc):
        """Rows for steps <= t0: logged/constant; after t0: nav0 with the countdown decremented by the OWN arc length (the tool's deadreckon)."""
        rows = np.zeros((len(idc), 5), np.float32)
        arc = [0.0]                                     # python floats added in order, as the tool's WindowState.arc (float32 row arithmetic must see a weak python float)
        for q in range(self.t0_step + 1, j + 1):
            arc.append(arc[-1] + float(np.linalg.norm(st["pose"][q, :2] - st["pose"][q - 1, :2])))
        for n, s in enumerate(idc):
            s = int(s)
            if s <= self.t0_step:
                rows[n] = self.nav_log[s] if self.nav_log is not None else self.nav0
            else:
                r = self.nav0.copy()
                if r[4] > 0:
                    r[3] = np.clip(r[3] - arc[s - self.t0_step] / 100.0, 0.0, 3.0)
                rows[n] = r
        return rows

    def build_batch(self, a_t: int = 0, a_s: int = 0) -> dict:
        """The MODEL_KEYS batch (B = 1) at NOW: `V8FMiniWindows.item` + the tool's `build_item` overrides + `collate`, for the keys forward_infer reads."""
        torch, np_ = self.torch, np
        if self.t0_step is None or self.nav0 is None:
            raise RuntimeError("set_t0/seed_track and set_nav must run before plan()")
        st = self._steps()
        T = len(st["pose"])
        j = T - 1
        idx = np.arange(j - self.hmax + 1, j + 1)
        idc = np.clip(idx, 0, T - 1)
        okf = idx >= 0
        okf &= np.isfinite(st["p10f"][idc]).all(1) | (idx < 1)
        have = np.array([int(s) in self.tok for s in idc])
        room = max(j - self.window_start, 0)
        a_t = int(min(a_t, room, self.cfg.max_age_t))
        a_s = int(max(a_t, min(a_s, room, self.cfg.max_age_s)))
        need = {int(s) for s in (idc[-(self.hist + a_s):])}
        miss = sorted(s for s in need if s not in self.tok)
        if miss:
            raise RuntimeError(f"no camera tokens for grid steps {miss}: the model would read frames that do not exist (push_tokens / the encoder did not run)")
        okf &= have
        feats = np.stack([(self.tok[int(s)].numpy() if int(s) in self.tok else np.zeros((self.cfg.n_tok, self.cfg.d_feat), np.float16)) for s in idc])
        if self.limit_mode == "causal":
            lim, lim_ok = causal_bucket(st["v"], j, CAUSAL_STEPS, self.speed_max_bin)
        else:
            lim, lim_ok = np.zeros(4, np.float32), False
        f32, T_ = np.float32, torch.as_tensor
        pose = st["pose"]
        b = {"feats": T_(feats, dtype=torch.float16)[None], "frame_ok": T_(okf, dtype=torch.bool)[None],
             "p10f": T_(np.nan_to_num(st["p10f"][idc], nan=0.0), dtype=torch.float32)[None],
             "nav": T_(self._nav_rows(st, j, idc), dtype=torch.float32)[None],
             "limit": T_(lim, dtype=torch.float32)[None], "limit_ok": T_([bool(lim_ok)], dtype=torch.bool),
             "base": T_(st["base"][j], dtype=torch.float32)[None],
             "a_t": T_([a_t], dtype=torch.long), "a_s": T_([a_s], dtype=torch.long),
             "pose_now": T_(pose[j][:3].astype(f32))[None], "pose_t": T_(pose[j - a_t][:3].astype(f32))[None],
             "pose_s": T_(pose[j - a_s][:3].astype(f32))[None]}
        return b

    # ------------------------------------------------------------------------------------------------ plan
    def _guard(self, batch_keys, keys_read):
        extra_in = set(batch_keys) - self.M.INFERENCE_INPUT_KEYS
        if extra_in or set(batch_keys) != set(self.model_keys):
            raise KeyGuardError(f"batch keys differ from MODEL_KEYS: extra {sorted(extra_in)}, missing {sorted(set(self.model_keys) - set(batch_keys))}")
        bad = set(keys_read) - self.M.INFERENCE_INPUT_KEYS
        if bad:
            raise KeyGuardError(f"forward_infer READ keys outside INFERENCE_INPUT_KEYS: {sorted(bad)}")

    def forward_batch(self, batch: dict):
        """batch -> (plan [30,2] f64 ego frame at NOW, extra). Arm 'p10' returns the SAME forward's prior; HACC (if set) acts inside forward_infer."""
        torch = self.torch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model.forward_infer(batch)
        self._guard(batch.keys(), out["_keys_read"])
        prior = out["p10"].double().cpu().numpy()[0]
        plan = out["traj30"].float().cpu().numpy()[0]
        raw_model = plan
        res = prior if self.arm == "p10" else plan
        extra = {"arm": self.arm, "keys_read": sorted(out["_keys_read"]), "p10_prior": prior.tolist(), "hacc_active": bool(self.model.hacc_active),
                 "limit_bucket": int(batch["limit"][0].argmax()) if bool(batch["limit_ok"][0]) else None,
                 "a_t": int(batch["a_t"][0]), "a_s": int(batch["a_s"][0]), "v0_ms": float(batch["p10f"][0, -1, 0]),
                 "tau_o": None if "traj30_o" not in out else out["traj30_o"].double().cpu().numpy()[0].tolist(),
                 # ADDITIVE (E5b, S-POS): S's three route checkpoints (x, y, psi) in S's OWN tick frame, and that frame's world pose (float32, as the model was fed).
                 # Exact float32 -> float64 widening, so a JSON round trip returns the same float32 values. No existing key or value changes.
                 "s_ckpt": out["ckpt"].double().cpu().numpy()[0].tolist(), "s_pose": batch["pose_s"].double().cpu().numpy()[0].tolist()}
        return np.ascontiguousarray(res, np.float64), extra

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        """Kit interface. `frames`: the harness' last raw frames, NEWEST LAST; frames[-1] is NOW, frames[-1-2k] is grid step j-k."""
        if getattr(self, "_odd", False):
            raise RuntimeError("plan() at a half-step: v8F replans on the 0.2 s grid (an even number of raw ticks after t0)")
        if self.nav0 is None:
            self.set_nav(nav_cmd=nav_cmd)
        j = (len(self.track) - 1) // 2
        if abs(float(self.track[-1, 3]) - float(v0)) > 1e-6:
            raise RuntimeError(f"history v={float(self.track[-1, 3])} != harness v0={float(v0)}")
        if self.encoder is not None and frames is not None:
            for k in range(self.hist + self.cfg.max_age_s):
                s = j - k
                if s in self.tok or s < 0 or len(frames) < 1 + 2 * k:
                    continue
                self.push_tokens(s, self.encoder.tokens(frames[-1 - 2 * k], intr))
        a_t, a_s = ages_for(j - self.t0_step, self.schedule)
        traj, extra = self.forward_batch(self.build_batch(a_t, a_s))
        self.ticks += 1
        extra.update({"step": j, "steps_since_t0": j - self.t0_step})
        return traj, extra
