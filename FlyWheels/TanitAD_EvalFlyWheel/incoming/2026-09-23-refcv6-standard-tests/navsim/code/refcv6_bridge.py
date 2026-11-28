#!/usr/bin/env python3
"""TanitAD side of the NavSim seam for refcv6 (RUN IN THE TANITAD VENV).

    NavSim export (devkit AgentInput per scorer token) ──► DECLARE (arm) ──► refcv6 inputs:
      frames      416x1024 cylindrical 3-camera stitch (frames416.py), 10 Hz slots <- 2 Hz frames
      nav         NavSim driving_command -> the v7 token (follow / left / right)
      v0          |velocity| at t0 (the core's measured initial state, PI ruling 2026-09-02)
      ego history 8 x (x, y, yaw, v) at 10 Hz over t0-0.7 ... t0, LINEAR in time between the
                  2 Hz states at t0-1.0 / -0.5 / 0 (the GRU's channels v, dv/dt, dyaw/dt follow)
      max speed   the posted limit of the ego's lane at t0 (nuPlan map; export_speed_limits.py),
                  valid=0 ("no ceiling known" = the model's own all-zero one-hot) where absent
      lift        the stitch's virtual camera in refcv6's rig frame (rig6.py)
    refcv6 (DDIM, 117 v0-conditioned anchors) ──► traj [8 knots, 2] ──► NavSim poses [8, 3]

⛔ ENFORCEMENT (gate ``navsim.ego_enforcement``) lives in ONE place: :func:`declare6`. It COPIES
only the EgoStatus fields an arm declares into a fresh dict; every input function below takes that
dict, never the export. ``tests/test_inputs6.py`` randomises every UNDECLARED field and requires
byte-identical model inputs, and mutates a DECLARED one and requires them to change.

Every model fact is READ from the checkpoint's ``config.json`` (rebuilt through the trainer's own
``build_parser`` + ``_pin_trainer_cfg`` by ``taniteval/tools/refcv3_arm.load_model``, strict
load) and ASSERTED here: window 8, knots (5..60) x 0.1 s, image 416x1024, 117 anchors.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frames416 as F4  # noqa: E402  (repo bootstrap: REPO/stack first on sys.path)

REPO = F4.REPO
B2 = F4._load("e2_navsim_bridge", os.path.join(F4.E2_CODE, "tanitad_navsim_bridge.py"))
RefusedInput = B2.RefusedInput

# --------------------------------------------------------------------------- #
# constants — each ASSERTED against its source at load time                     #
# --------------------------------------------------------------------------- #
KNOT_T_S = B2.KNOT_T_S                   # (0.5, 1, 1.5, 2, 3, 4, 5, 6) — refcv6 horizons x 0.1 s
NAVSIM_T_S = B2.NAVSIM_T_S               # (0.5 ... 4.0)
DT_FRAME_S = 0.1                         # the v2ep provider grid; the ego-history encoder's dt
N_STACK = 3                              # D-015
WINDOW = 8                               # config.json window / the ego-history steps
HIST_T_S = tuple(-(WINDOW - 1 - k) * DT_FRAME_S for k in range(WINDOW))   # -0.7 ... 0.0
EQUALIZE_BOTTOM_ROWS = 43                # argv --equalize-bottom-rows (the LIFT mask; see rig6)
#: the NavSim one-hot index -> refb.NAV_COMMANDS name — IMPORTED from E2 (``NAVSIM_CMD_TO_NAV_NAME``,
#: index order verified by E2's control K9 on the logs: 0 left, 1 straight, 2 right, 3 unknown).
NAVSIM_CMD_TO_NAV_NAME = B2.NAVSIM_CMD_TO_NAV_NAME
#: the 2 Hz history states the 10 Hz window is built from: indices into the 4-frame export list
HIST_SRC_IDX = (1, 2, 3)                 # t0-1.0, t0-0.5, t0

_HIST_DECL = tuple(f"{f}[{i}]" for f in ("ego_pose", "ego_velocity") for i in HIST_SRC_IDX)
ARMS6 = {
    "R6_A1": {
        "declared": _HIST_DECL + ("driving_command[3]",),
        "frames": "ST", "nav": "cmd", "vmax": "map", "seed": 0,
        "meaning": ("PRIMARY: frames (416x1024 stitch, static-t0 history) + v0 + 10 Hz ego "
                    "history + NavSim command -> v7 nav token + map posted limit")},
    "R6_A1_s1": {
        "declared": _HIST_DECL + ("driving_command[3]",),
        "frames": "ST", "nav": "cmd", "vmax": "map", "seed": 1,
        "meaning": "R6_A1 with a DIFFERENT inference seed: the sampler's run-to-run floor"},
    "R6_A1NT": {
        "declared": _HIST_DECL + ("driving_command[3]",),
        "frames": "NT", "nav": "cmd", "vmax": "map", "seed": 0,
        "meaning": "R6_A1 with the NEAREST-TIME frame history (declared sensitivity)"},
    "R6_BLIND": {
        "declared": _HIST_DECL + ("driving_command[3]",),
        "frames": "BLIND", "nav": "cmd", "vmax": "map", "seed": 0,
        "meaning": "frames-blind (the registered deliberate regression): one constant grey"},
    "R6_NAVOFF": {
        "declared": _HIST_DECL,
        "frames": "ST", "nav": None, "vmax": "map", "seed": 0,
        "meaning": "nav withheld: nav_cmd=None (core -> follow; tactical one-hot all-zero)"},
    "R6_VMAXOFF": {
        "declared": _HIST_DECL + ("driving_command[3]",),
        "frames": "ST", "nav": "cmd", "vmax": "off", "seed": 0,
        "meaning": "max speed withheld: v_max_valid=0 on EVERY scene (all-zero one-hot)"},
    # ⛔ SPEC §12 (amendment A4): PRIVILEGED DIAGNOSTIC — never a result, never in a bar.
    "R6_VMAXORACLE": {
        "declared": _HIST_DECL + ("driving_command[3]",),
        "frames": "ST", "nav": "cmd", "vmax": "oracle", "seed": 0,
        "meaning": ("PRIVILEGED DIAGNOSTIC: the max-speed one-hot from the human's OWN realised "
                    "max |v| over [t0+2, t0+6] s (the channel's TRAINING definition, "
                    "code/vmax_oracle.py) — prices the train/deploy mismatch; never a result")},
}
FIELDS = ("ego_pose", "ego_velocity", "ego_acceleration", "driving_command")


# --------------------------------------------------------------------------- #
# 1. DECLARE — the enforcement point                                            #
# --------------------------------------------------------------------------- #
def declare6(ego_statuses: list, arm: str) -> dict:
    """Copy ONLY ``ARMS6[arm]['declared']`` out of the 4-frame EgoStatus list (oldest first,
    index 3 = t0). The returned dict is the COMPLETE ego/route information any input function
    below may read."""
    if len(ego_statuses) != 4:
        raise RefusedInput(f"expected 4 history EgoStatus, got {len(ego_statuses)}")
    spec = ARMS6[arm]
    out = {"_arm": arm, "_declared": list(spec["declared"])}
    for name in spec["declared"]:
        field, idx = name[:-1].split("[")
        out[name] = [float(x) for x in ego_statuses[int(idx)][field]]
    return out


def speed_of(vxy) -> float:
    """``v = hypot(vx, vy)`` — E2's ``ego_block`` / ``physicalai.signals_at`` convention."""
    return float(math.hypot(float(vxy[0]), float(vxy[1])))


def v0_of(decl: dict) -> float:
    return speed_of(decl["ego_velocity[3]"])


def ego_history_poses(decl: dict, times_s: list) -> np.ndarray:
    """``[WINDOW, 4]`` = (x, y, yaw, v) at ``HIST_T_S`` (t0-0.7 ... t0, 10 Hz), oldest first.

    NavSim holds the ego at 2 Hz; refcv6's encoder (``ego_channels_from_poses``) reads 10 Hz and
    takes BACKWARD differences (dv/dt, dyaw/dt) at ``dt = 0.1``. The window is filled by LINEAR
    interpolation IN TIME between the declared 2 Hz states (t0-1.0, t0-0.5, t0), so within each
    0.5 s segment the encoder sees that segment's mean acceleration / yaw rate — the honest
    content of 2 Hz data, and no sample later than t0 (none is declared, none exists here).
    ``yaw`` is unwrapped across the samples before interpolation (a heading crossing +-pi would
    otherwise read as a ~60 rad/s yaw rate). ``times_s`` are the export's frame times relative to
    t0 — normally -1.5, -1.0, -0.5, 0; a log that DROPPED a frame is interpolated at its actual
    times under the guards below (SPEC amendment A1), anything else is REFUSED.
    """
    ts = [float(times_s[i]) for i in HIST_SRC_IDX]
    # ⚠️ SPEC AMENDMENT A1 (2026-09-24, before any navhard/navtest score; no warmup scene is
    # affected — all 220 are on the grid): a few logs DROP a 2 Hz frame (MEASURED: navhard 2/5,912,
    # navtest 15/12,146 carry e.g. (-2.0, -1.5, -0.5, 0) or (-2.0, -1.5, -1.0, 0)). The rule is
    # unchanged — linear interpolation between the declared states — applied at their ACTUAL
    # times, with the guards that keep it an interpolation: t0 == 0, strictly increasing, the
    # oldest used state at or before the window's first slot (never an extrapolation), and no gap
    # over 1.0 s. Anything else is REFUSED.
    if abs(ts[-1]) > 2e-3 or not (ts[0] < ts[1] < ts[2]):
        raise RefusedInput(f"history frame times {ts} are not increasing to t0 = 0")
    if ts[0] > HIST_T_S[0] + 2e-3:
        raise RefusedInput(f"history frame times {ts}: the oldest used state is after the "
                           f"window's first slot {HIST_T_S[0]} s — that would be extrapolation")
    if max(b - a for a, b in zip(ts, ts[1:])) > 1.0 + 2e-3:
        raise RefusedInput(f"history frame times {ts}: a gap over 1.0 s")
    xy = np.array([decl[f"ego_pose[{i}]"][:2] for i in HIST_SRC_IDX], dtype=np.float64)
    yaw = np.unwrap(np.array([decl[f"ego_pose[{i}]"][2] for i in HIST_SRC_IDX],
                             dtype=np.float64))
    v = np.array([speed_of(decl[f"ego_velocity[{i}]"]) for i in HIST_SRC_IDX], dtype=np.float64)
    q = np.asarray(HIST_T_S, dtype=np.float64)
    out = np.stack([np.interp(q, ts, xy[:, 0]), np.interp(q, ts, xy[:, 1]),
                    np.interp(q, ts, yaw), np.interp(q, ts, v)], axis=-1)
    return out.astype(np.float32)


def nav_input(decl: dict, arm: str) -> dict:
    """``{"nav_index": int | None, "name", "navsim_argmax"}``. The NavSim one-hot is mapped with
    E2's ``nav_index_from_command`` (IMPORTED; it REFUSES a non-one-hot vector). ``None`` for an
    arm that withholds nav (the model's own nav-ZERO intervention)."""
    if ARMS6[arm]["nav"] != "cmd":
        return {"nav_index": None, "name": None, "navsim_argmax": None}
    idx, name, k = B2.nav_index_from_command(decl["driving_command[3]"])
    return {"nav_index": int(idx), "name": name, "navsim_argmax": int(k)}


def max_speed_input(speed_rec: dict | None, arm: str) -> dict:
    """``(v_max_ms, v_max_valid)`` for the model's 4-way one-hot. The RAW m/s goes in and the
    model applies its containing-window ladder ONCE (``MaxSpeedOneHotEncoder``); an unknown
    ceiling is ``valid = 0`` -> an ALL-ZERO one-hot (the model's contract: "Pass v_max_valid=0 for
    'no limit known'"; X15 — "unknown" and "30 km/h" are different inputs)."""
    src = ARMS6[arm]["vmax"]
    if src == "oracle" and speed_rec is not None and not str(speed_rec.get("why", "")).startswith("ORACLE"):
        # the oracle arm must be fed the ORACLE record (code/vmax_oracle.py), never the map's
        raise RefusedInput(f"{arm}: fed a non-oracle max-speed record ({speed_rec.get('why')!r})")
    if src == "off" or speed_rec is None or speed_rec.get("status") != "limit":
        why = ("withheld by the arm" if src == "off"
               else f"{src}: {None if speed_rec is None else speed_rec.get('status')}")
        return {"v_max_ms": 0.0, "v_max_valid": 0.0, "why": why}
    v = float(speed_rec["speed_limit_mps"])
    # a map limit must be > 0; an ORACLE max may be exactly 0 (a human who never moved), which the
    # containing-window ladder puts in the lowest bin (30 km/h, valid) — exactly as in training
    if not (math.isfinite(v) and (v > 0 or (src == "oracle" and v >= 0))):
        raise RefusedInput(f"non-positive {src} limit {v}")
    from tanitad.refs.refcv6_max_speed import speed_max_bin, SPEED_MAX_STEPS_KMH_V6
    b, over = speed_max_bin(v)
    return {"v_max_ms": v, "v_max_valid": 1.0,
            "why": "map posted limit" if src == "map" else speed_rec["why"],
            "bin_kmh": SPEED_MAX_STEPS_KMH_V6[b], "over_ceiling": bool(over)}


# --------------------------------------------------------------------------- #
# 2. FRAMES                                                                     #
# --------------------------------------------------------------------------- #
class Bank416:
    """A ``frames416.py`` bank: ``frames/<key>.npy`` u8 ``[K, 416, 1024, 3]`` (the LAST K
    history frames, index -1 = t0), sha-VERIFIED against the provenance before use."""

    def __init__(self, root: str):
        import pandas as pd
        self.root = root
        p = pd.read_parquet(os.path.join(root, "frames_provenance.parquet"))
        self.prov = {r.scene_token: r for r in p.itertuples()}
        self.report = json.load(open(os.path.join(root, "BUILD_REPORT.json"), encoding="utf-8"))
        fr = self.report["frame"]
        if (fr["h"], fr["w"], round(fr["f_ref"], 9), fr["projection"]) != (
                416, 1024, round(F4.FRAME_416x1024.f_ref, 9), "cylindrical"):
            raise RefusedInput(f"bank {root} frame {fr} is not FRAME_416x1024")
        self.rigs = self.report["rigs"]
        self.keep = int(self.report["keep"])
        self.mean_px = float(np.mean([r.mean_px for r in self.prov.values()]))

    def load(self, key: str) -> tuple:
        a = np.load(os.path.join(self.root, "frames", f"{key}.npy"))
        sha = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        r = self.prov.get(key)
        if r is None or sha != r.sha256:
            raise RefusedInput(f"{key}: bank sha {sha} != provenance {None if r is None else r.sha256}")
        if a.dtype != np.uint8 or a.shape[1:] != (416, 1024, 3) or a.shape[0] != self.keep:
            raise RefusedInput(f"{key}: bank is {a.dtype}{a.shape}")
        if float(a.mean()) < 1.0:
            raise RefusedInput(f"{key}: bank mean {a.mean():.3f} — an all-black bank")
        return a, r.rig_key, sha


class BankNavtest416:
    """W3's navtest bank layout at 416 x 1024 (``build_navtest416.py``): ``index.json`` maps a
    token to ``(frames_sNN.npy, rows)`` — the 4 history frames stored ONCE per unique frame —
    and records the per-token sha256[:16] of the stacked ``[4, 416, 1024, 3]`` array, which is
    re-checked on every load. ``keep`` is 4 (all history frames), so ST and NT are both served."""

    def __init__(self, root: str):
        self.root = root
        self.index = json.load(open(os.path.join(root, "index.json"), encoding="utf-8"))
        fr = self.index["frame"]
        if (fr["h"], fr["w"], round(fr["f_ref"], 9), fr["projection"]) != (
                416, 1024, round(F4.FRAME_416x1024.f_ref, 9), "cylindrical"):
            raise RefusedInput(f"navtest bank {root} frame {fr} is not FRAME_416x1024")
        self.prov = self.index["tokens"]
        self.keep = 4
        self.mean_px = float(np.mean([r["mean_px"] for r in self.prov.values()]))
        self._mm: dict = {}

    def load(self, key: str) -> tuple:
        r = self.prov.get(key)
        if r is None:
            raise RefusedInput(f"{key}: not in the navtest bank")
        if r["file"] not in self._mm:
            self._mm = {r["file"]: np.load(os.path.join(self.root, r["file"]), mmap_mode="r")}
        a = np.ascontiguousarray(self._mm[r["file"]][r["rows"]])
        sha = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        if sha != r["sha256"]:
            raise RefusedInput(f"{key}: bank sha {sha} != index {r['sha256']}")
        if a.dtype != np.uint8 or a.shape != (4, 416, 1024, 3) or float(a.mean()) < 1.0:
            raise RefusedInput(f"{key}: bank is {a.dtype}{a.shape} mean {a.mean():.3f}")
        return a, r["rig_key"], sha


def slot_sources6(times_s: list, construction: str) -> list:
    """NavSim frame index (0..3, 3 = t0) for each of the ``WINDOW + N_STACK - 1`` = 10 raw 10 Hz
    slots, oldest first — E2's ``slot_sources`` (IMPORTED) with refcv6's window."""
    if construction == "BLIND":
        construction = "ST"
    return B2.slot_sources(times_s, construction, WINDOW)


def pack_rows(bank_frames: np.ndarray, sources: list, blind_grey: int | None = None):
    """``[K,H,W,3]`` (the LAST K history frames) + NavSim slot sources -> ``[8, 9, H, W]`` u8 via
    E2's ``pack_frames`` (the trainer's own ``stack_frames``, D-015 oldest first)."""
    k = bank_frames.shape[0]
    off = 4 - k
    local = [s - off for s in sources]
    if min(local) < 0:
        raise RefusedInput(f"slot sources {sources} need frames the bank (keep={k}) did not store")
    return B2.pack_frames(bank_frames, local, blind_grey)


# --------------------------------------------------------------------------- #
# 3. MODEL                                                                      #
# --------------------------------------------------------------------------- #
#: argv entries that name files on Thor and are read at REBUILD time (the rig table for the
#: agent-loss camera, the anchor artifact). The data/label paths are never read at rebuild.
THOR_PATHS = {
    "/home/nvidia/data/refcv6_train_eval139_extrinsics.json": "refcv6_train_eval139_extrinsics.json",
    "/home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt": "refc_anchors_6s_v0cond_alat_117.pt",
}
AUX_DEFAULT = "D:/refcv6_eval_kit/navsim_aux"


def rebuilt_config(config_path: str, aux_dir: str = AUX_DEFAULT) -> tuple[str, dict]:
    """A copy of the run's ``config.json`` whose argv (a) drops ``--trunk-compile`` (no Triton on
    Windows; compile is a speed lever whose function the run's own test pins as identical) and
    (b) points the two rebuild-time Thor paths at their md5-verified local copies. Nothing else
    changes; the edits are returned so the manifest records them."""
    cfg = json.load(open(config_path, encoding="utf-8"))
    argv0 = list(cfg["argv"])
    argv, edits = [], []
    for a in argv0:
        if a == "--trunk-compile":
            edits.append("dropped --trunk-compile (no Triton on Windows)")
            continue
        if a in THOR_PATHS:
            loc = os.path.join(aux_dir, THOR_PATHS[a]).replace("\\", "/")
            if not os.path.isfile(loc):
                raise RefusedInput(f"rebuild needs {a}; local copy {loc} missing")
            edits.append(f"{a} -> {loc}")
            argv.append(loc)
            continue
        argv.append(a)
    cfg["argv"] = argv
    fd, tmp = tempfile.mkstemp(prefix="refcv6_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return tmp, {"edits": edits, "n_argv": len(argv0)}


def load_refcv6(ckpt: str, config: str, device: str = "cpu", precision: str = "auto",
                aux_dir: str = AUX_DEFAULT):
    """``(model, cfg, targs, prov, arm_module, meta)`` — rebuilt by the landed eval's own
    ``refcv3_arm.load_model`` (trainer parser + ``_pin_trainer_cfg``, config cross-check incl.
    ``param_breakdown``, perception branch rebuilt, STRICT load), then asserted.

    ``precision``: ``as_trained`` keeps the trunk's recorded levers (bf16 autocast + NHWC — what
    the run and its in-run eval compute); ``fp32`` turns those two off (the same weights, the
    backbone in fp32: ~2 %/5 % feature deviation, MEASURED by the run's own §9 table — used on
    CPU, where bf16 is emulated and 10x slower, MEASURED here 98 s vs 9 s). ``auto`` =
    as_trained on CUDA, fp32 on CPU. The choice is returned in ``meta`` and stamped on every row.
    """
    ra = F4._load("refcv3_arm_for_refcv6", os.path.join(REPO, "taniteval", "tools",
                                                        "refcv3_arm.py"))
    tmp, edits = rebuilt_config(config, aux_dir)
    # ⛔⛔ THE ANCHOR UNITS (KL, MEASURED 2026-09-24). `refc_v3_train.train()` calls
    # `_read_anchor_artifact(args)` BEFORE `_pin_trainer_cfg` (`:6726`): it reads the file's DECLARED
    # `control_units` ('alat' for refc_anchors_6s_v0cond_alat_117.pt) into
    # `args.anchor_control_units` and its derivation constants into `args._anchor_artifact_meta`,
    # which the pin then adopts. `refcv3_arm.rebuild_config` parses argv and pins WITHOUT that read,
    # and this run's argv carries no `--anchor-control-units` (the FILE declares it) — so the rebuilt
    # decoder defaulted to 'kappa' and rolled a lateral-ACCELERATION vocabulary as CURVATURE (the
    # anchor bank differed by up to 108 m from the trainer-exact build; every plan changed). The
    # CLAUDE.md "units" trap, caught only by the loader cross-check against the battery stream's
    # trainer-exact loader. Fixed by replaying train()'s order: the artifact read runs first.
    tr = ra.trainer()
    orig_pin = tr._pin_trainer_cfg

    def _pin_like_train(cfg0, args):
        if getattr(args, "anchors", None) and getattr(args, "anchor_control_units", None) is None:
            tr._read_anchor_artifact(args)
        return orig_pin(cfg0, args)

    tr._pin_trainer_cfg = _pin_like_train
    try:
        model, cfg, targs, prov = ra.load_model(ckpt, tmp, device, False)
    finally:
        tr._pin_trainer_cfg = orig_pin
        try:
            os.remove(tmp)
        except OSError:
            pass
    units = str(getattr(model.core.decoder, "anchor_control_units", "?"))
    if units != "alat":
        raise RefusedInput(f"decoder anchor_control_units {units!r} != the artifact's 'alat'")
    hz = tuple(round(h * DT_FRAME_S, 6) for h in prov["horizons"])
    if hz != KNOT_T_S:
        raise RefusedInput(f"model slots {hz} != KNOT_T_S {KNOT_T_S}")
    if int(prov["window"]) != WINDOW:
        raise RefusedInput(f"window {prov['window']} != {WINDOW}")
    if tuple(cfg.core.encoder.image_hw()) != (416, 1024):
        raise RefusedInput(f"image_hw {cfg.core.encoder.image_hw()} != (416, 1024)")
    if int(prov["n_anchors"]) != 117:
        raise RefusedInput(f"n_anchors {prov['n_anchors']} != 117")
    if model.core.ego_hist is None or not bool(getattr(cfg, "max_speed_onehot_v6", False)):
        raise RefusedInput("the build lacks the ego-history encoder or the v6 max-speed one-hot")
    if getattr(model, "_perception", None) is None or model._perception.lift is None:
        raise RefusedInput("the build has no BEV lift — not the refcv6 launch configuration")
    enc = model.core.encoder
    if precision == "auto":
        precision = "as_trained" if str(device).startswith("cuda") else "fp32"
    levers0 = dict(enc.memory_levers)
    if precision == "fp32":
        enc.memory_levers["bf16"] = False
        enc.memory_levers["channels_last"] = False
    elif precision != "as_trained":
        raise RefusedInput(f"precision {precision!r}")
    model._lift_bank = None          # PhysicalAI's per-clip bank is never used on NavSim scenes
    meta = {"precision": precision, "trunk_levers_recorded": levers0,
            "anchor_control_units": units,
            "anchor_units_fix": ("train()'s _read_anchor_artifact replayed before _pin_trainer_cfg "
                                 "(refcv3_arm.rebuild_config skips it)"),
            "trunk_levers_eval": dict(enc.memory_levers),
            "trunk_equalize_bottom_rows_BUILT": int(getattr(enc.cfg, "equalize_bottom_rows", 0)),
            "config_edits": edits, "device": str(device)}
    return model, cfg, targs, prov, ra, meta


def exact_dedup(enc) -> None:
    """EVAL-ONLY speed lever: compute each EXACTLY-EQUAL frame of a call ONCE (the trunk's own
    ``_backbone_dedup`` shares frames only between ADJACENT rows, so a static-history window —
    every raw slot the same t0 frame — still costs 10 backbone passes). With BN frozen a frame's
    features depend on that frame alone, so this is the same function (the trunk's own docstring
    argument); ``tests/test_exact_dedup.py`` MEASURES the difference against the native path on
    real scenes. Recorded on the trunk as ``memory_levers['eval_exact_dedup']``."""
    import torch
    import types

    def _dd(self, per):
        n, k = int(per.shape[0]), int(per.shape[1])
        flat = per.reshape(n * k, *per.shape[2:])
        reps, slot = [], []
        for i in range(n * k):
            for ri, r in enumerate(reps):
                if torch.equal(flat[i], flat[r]):
                    slot.append(ri)
                    break
            else:
                reps.append(i)
                slot.append(len(reps) - 1)
        f16u, f32u = self._backbone(flat[torch.tensor(reps, device=per.device)].contiguous())
        idx = torch.tensor(slot, device=f16u.device)
        self.last_dedup = (n * k, len(reps))
        return f16u.index_select(0, idx), f32u.index_select(0, idx)

    enc._backbone_dedup = types.MethodType(_dd, enc)
    enc.memory_levers["eval_exact_dedup"] = True


def scene_seed(base: int, token: str) -> int:
    """Per-scene inference seed: COMMON RANDOM NUMBERS across arms (every arm draws the same
    sampler eps for a scene), deterministic, independent of batch order."""
    return int(hashlib.sha256(f"{int(base)}:{token}".encode()).hexdigest()[:8], 16)


def run_model6(model, arm_mod, rows_u8, decl: dict, arm: str, nav: dict, vmax: dict,
               hist_poses: np.ndarray, grid, valid, steps: int, seed: int, device: str) -> dict:
    """One scene through refcv6's forward with EXACTLY the in-run eval's kwargs
    (``compute_losses_v3``): frames, nav_cmd, v0, steps, ego_state=None (not injected in this
    build), v_max_ms/v_max_valid, perception_grid/valid; the ego window via ``set_ego_window``."""
    import torch
    fr = arm_mod.trainer().frames_to_device(torch.from_numpy(rows_u8)[None], device)
    v0 = v0_of(decl)
    kw = {"nav_cmd": (None if nav["nav_index"] is None
                      else torch.tensor([nav["nav_index"]], dtype=torch.long, device=device)),
          "v0": torch.tensor([v0], dtype=torch.float32, device=device),
          "steps": int(steps),
          "v_max_ms": torch.tensor([vmax["v_max_ms"]], dtype=torch.float32, device=device),
          "v_max_valid": torch.tensor([vmax["v_max_valid"]], dtype=torch.float32, device=device),
          "perception_grid": grid[None].to(device), "perception_valid": valid[None].to(device)}
    model.core.set_ego_window(torch.from_numpy(hist_poses)[None].to(device), WINDOW)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        out = model(fr, **kw)
    traj = out["traj"].float()[0].cpu().numpy().astype(np.float64)
    diag = {"sel_idx": int(out["sel_idx"][0]),
            "tacv6_lat_argmax": int(out["tacv6_lat_logits"][0].argmax(-1)),
            "tacv6_lon_argmax": int(out["tacv6_lon_logits"][0].argmax(-1)),
            "max_speed_injected": bool(out.get("max_speed_injected", False)),
            "nav_injected": bool(out.get("nav_injected", False)),
            "dedup": list(getattr(model.core.encoder, "last_dedup", None) or [])}
    return {"traj": traj, "v0": v0, "diag": diag}


def knots_to_navsim(knots_xy: np.ndarray) -> np.ndarray:
    """E2's conversion, IMPORTED unchanged: a C2 not-a-knot cubic spline in time through the
    origin and the 8 knots (0.5 ... 6 s), evaluated at NavSim's 0.5 ... 4.0 s (knots reproduced;
    2.5 / 3.5 s interpolated; the 5 s / 6 s knots shape the spline but are not emitted); heading
    = the spline tangent, held below 1.0 m/s (E2 SPEC amendment 1)."""
    return B2.knots_to_navsim(knots_xy)


def json_dump(obj, path: str) -> None:
    B2.json_dump(obj, path)
