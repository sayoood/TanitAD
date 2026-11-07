#!/usr/bin/env python3
"""EVAL / DEPLOY SIDE OF WP YAW: feed a yaw-trained REFe checkpoint the SAME yaw rate it was trained on (PLAN: TanitAD-artifacts/yawrate_wp/PLAN.md).

A checkpoint trained with `train.py --ego-yaw-table` says so in its meta (`ego_yaw = "nuplan_angular_rate_z"`, `ego_yaw_scale`, `ego_yaw_table_sha256`, `ego_yaw_slot = 4`).  It was trained on
ego[4] = ego_yaw_scale x (rad/s); the shim multiplies by the META's scale (ego_yaw_table.YAW_SCALE at training time, MM 2026-11-06) and REFUSES a yaw checkpoint without one.  Such a model reads a REAL
ego[4]; the planner (`planner.py::_ego_vec`) fills ego[4] from `EgoState.dynamic_car_state.angular_velocity`, which for every DB-built EgoState (the NAVSIM navtest seam, the bank's
scenario builder) is 0.0 -- the very defect the table fixes at training.  Evaluating a yaw-trained checkpoint through the unmodified planner would therefore feed it a yaw of 0.0 and
measure a TRAIN/EVAL MISMATCH, not the model.

THIS MODULE wraps the MODEL'S forward (never the planner, never any file pinned by eval/HARNESS_PINS.json): for a checkpoint whose meta says `ego_yaw`, the wrapper puts the token's table
yaw into ego[..., 4] before the model sees it.  For any other checkpoint `from_checkpoint` returns None and NOTHING is wrapped (bit-identical).

    shim = EgoYawShim.from_checkpoint(ckpt, "ego_yaw_navtest.json.gz")      # None for a non-yaw checkpoint
    shim.install(planner.model, token_source=lambda: planner._scenario._initial_lidar_token)
    # or, for a script you must not edit (the navtest seam):
    python ego_yaw.py --table ego_yaw_navtest.json.gz -- ../eval/refe_navtest_seam.py --ckpt ... --out ...

FAILS LOUD, never silent: no token available at a forward, a token absent from the table, a table that is not an ego-yaw table, a checkpoint that wants yaw with no table given, and (strict,
the default) an incoming ego[4] that is already non-zero -- two sources for one slot.
SCOPE: this is for OPEN-LOOP, per-token evaluation (the table is keyed by lidar_pc token).  In a closed-loop sim the simulated ego's own angular_velocity is already live and non-zero, and
the planner feeds it as is; do not install the shim there.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ego_yaw_table as EYT  # noqa: E402


def checkpoint_meta(path: str) -> dict:
    """The checkpoint's meta only (memory-mapped where torch allows: the dev box is RAM-bound)."""
    import torch
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        sd = torch.load(path, map_location="cpu", weights_only=False)
    return (sd.get("meta") or {}) if isinstance(sd, dict) else {}


def wants_yaw(meta: dict) -> bool:
    return (meta or {}).get("ego_yaw") == EYT.YAW_TAG


class EgoYawShim:
    def __init__(self, table: dict, table_sha256: str, scale: float, *, train_table_sha256: str | None = None, slot: int = EYT.EGO_YAW_SLOT, allow_incoming: bool = False):
        # `scale` has NO default on purpose: the number lives in ego_yaw_table.YAW_SCALE (the trainer's source) and in the checkpoint's meta (what the model was trained on) -- never re-typed here
        if not (isinstance(scale, (int, float)) and scale == scale and 0.0 < abs(float(scale)) < float("inf")):
            raise RuntimeError(f"EgoYawShim: the yaw scale must be a finite non-zero number, got {scale!r}")
        self.table, self.table_sha256, self.train_table_sha256 = table, table_sha256, train_table_sha256
        self.scale = float(scale)
        self.slot, self.allow_incoming = int(slot), bool(allow_incoming)
        self._token = None
        self._source = None
        self._orig = None
        self._model = None
        self.n_calls = self.n_rows = self.n_incoming_nonzero = 0

    # ---- construction
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, table_path: str | None, *, meta: dict | None = None, allow_incoming: bool = False):
        """None when the checkpoint was NOT trained with a real yaw rate (the model call stays untouched)."""
        meta = checkpoint_meta(ckpt_path) if meta is None else meta
        if not wants_yaw(meta):
            return None
        if not table_path:
            raise RuntimeError(f"{ckpt_path}: trained with ego_yaw={meta.get('ego_yaw')!r} (train table sha256 {str(meta.get('ego_yaw_table_sha256'))[:12]}); evaluating it needs the "
                               f"yaw table of the EVAL tokens (ego_yaw_table.py build --split navtest) -- a yaw of 0.0 would measure a train/eval mismatch, not the model.")
        if "ego_yaw_scale" not in meta:
            raise RuntimeError(f"{ckpt_path}: meta says ego_yaw={meta.get('ego_yaw')!r} but carries NO `ego_yaw_scale`: the multiplier the model was trained with is unknown, and feeding raw rad/s "
                               f"(or any guess) is a train/eval mismatch. REFUSED.")
        table, tmeta, sha = EYT.load(table_path)
        if tmeta.get("field") != EYT.FIELD:
            raise RuntimeError(f"{table_path}: table field {tmeta.get('field')!r} is not {EYT.FIELD!r}, which the checkpoint was trained on")
        if int(meta.get("ego_yaw_slot", EYT.EGO_YAW_SLOT)) != EYT.EGO_YAW_SLOT:
            raise RuntimeError(f"{ckpt_path}: ego_yaw_slot {meta.get('ego_yaw_slot')} != {EYT.EGO_YAW_SLOT}")
        return cls(table, sha, meta["ego_yaw_scale"], train_table_sha256=meta.get("ego_yaw_table_sha256"), slot=EYT.EGO_YAW_SLOT, allow_incoming=allow_incoming)

    # ---- the token of the NEXT forward
    def set_token(self, token) -> None:
        """A lidar_pc token, or a sequence of them (one per batch row)."""
        self._token = token

    def clear(self) -> None:
        self._token = None

    @contextlib.contextmanager
    def token(self, tok):
        prev = self._token
        self._token = tok
        try:
            yield self
        finally:
            self._token = prev

    def _current(self):
        t = self._source() if self._source is not None else None
        if t is None:
            t = self._token
        if t is None:
            raise RuntimeError("EgoYawShim: a forward with NO token set (no set_token / token() context / token_source result). A yaw-trained checkpoint cannot be evaluated on a "
                               "frame whose token is unknown -- REFUSED rather than feed it a yaw of 0.0.")
        return [t] if isinstance(t, str) else list(t)

    # ---- install / remove
    def install(self, model, token_source=None):
        """Wrap `model.forward` (the instance's, not the class's).  `token_source()` (optional) returns the token(s) of the CURRENT forward; it wins over set_token()."""
        if self._orig is not None:
            raise RuntimeError("EgoYawShim already installed")
        self._model, self._source, self._orig = model, token_source, model.forward
        orig = self._orig

        def forward(img, ego, goal, *a, **k):
            return orig(img, self.apply(ego), goal, *a, **k)
        model.forward = forward
        return self

    def remove(self) -> None:
        if self._orig is not None and self._model is not None:
            try:
                del self._model.forward                      # drop the instance attribute: the class's forward is back
            except AttributeError:
                self._model.forward = self._orig
        self._orig = self._model = None

    # ---- the override itself
    def apply(self, ego):
        """ego [B, D] -> a COPY with ego[:, slot] = scale x table[token] (float32 of the float64 product -- the SAME rounding as the loader's `ego_t[4] = YAW_SCALE * table[tok]`).
        Everything else is the input, bit for bit."""
        import torch
        toks = self._current()
        if ego.shape[-1] <= self.slot:
            raise RuntimeError(f"EgoYawShim: ego is {ego.shape[-1]}-D, the yaw slot is {self.slot}")
        if len(toks) != ego.shape[0]:
            if len(toks) == 1 and ego.shape[0] >= 1:
                toks = toks * ego.shape[0]
            else:
                raise RuntimeError(f"EgoYawShim: {len(toks)} token(s) for a batch of {ego.shape[0]}")
        miss = [t for t in toks if t not in self.table]
        if miss:
            raise RuntimeError(f"EgoYawShim: token(s) {miss[:3]} are NOT in the yaw table (sha256 {self.table_sha256[:12]}): REFUSED rather than feed a yaw of 0.0.")
        inc = float(ego[..., self.slot].detach().abs().max()) if ego.numel() else 0.0
        if inc != 0.0:
            self.n_incoming_nonzero += 1
            if not self.allow_incoming:
                raise RuntimeError(f"EgoYawShim: the incoming ego[{self.slot}] is already non-zero (max |.| {inc:.3g}): two sources for one slot. REFUSED (allow_incoming=True to replace).")
        out = ego.clone()
        out[..., self.slot] = torch.tensor([self.scale * self.table[t] for t in toks], dtype=out.dtype, device=out.device)
        self.n_calls += 1
        self.n_rows += len(toks)
        return out

    def report(self) -> dict:
        return {"table_sha256": self.table_sha256, "train_table_sha256": self.train_table_sha256, "scale": self.scale, "n_calls": self.n_calls, "n_rows": self.n_rows,
                "n_incoming_nonzero": self.n_incoming_nonzero, "slot": self.slot, "tokens_in_table": len(self.table)}


def _scenario_token(planner):
    sc = getattr(planner, "_scenario", None)
    return getattr(sc, "_initial_lidar_token", None) if sc is not None else None


def attach_to_planner(planner, checkpoint: str | None, table_path: str | None, *, allow_incoming: bool = False):
    """Install the shim on a REFePlanner's model, token read from `planner._scenario._initial_lidar_token` (the navtest seam sets `_scenario` before every `infer`).  None for a non-yaw checkpoint."""
    if not checkpoint:
        return None
    shim = EgoYawShim.from_checkpoint(checkpoint, table_path, allow_incoming=allow_incoming)
    if shim is None:
        return None
    shim.install(planner.model, token_source=lambda: _scenario_token(planner))
    planner.ego_yaw_shim = shim
    return shim


def patch_planner_class(table_path: str | None, *, allow_incoming: bool = False):
    """Make every `REFePlanner(...)` built from now on attach the shim when its checkpoint wants one -- for scripts that construct the planner themselves (the seam) and must not be edited.
    Returns an undo function."""
    import planner as PL
    orig = PL.REFePlanner.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        ck = k.get("checkpoint", a[0] if a else None)
        attach_to_planner(self, ck, table_path, allow_incoming=allow_incoming)
    PL.REFePlanner.__init__ = init

    def undo():
        PL.REFePlanner.__init__ = orig
    return undo


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: ego_yaw.py --table <eval table> [--allow-incoming] -- <script.py> [args...]")
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None)
    ap.add_argument("--allow-incoming", action="store_true")
    a = ap.parse_args(argv[:i])
    script, rest = argv[i + 1], argv[i + 2:]
    patch_planner_class(a.table, allow_incoming=a.allow_incoming)
    sys.argv = [script] + rest
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
