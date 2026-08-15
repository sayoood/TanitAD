"""P8 — frozen-trunk BEV occupancy readout (WM_PHYSICS_PROOF.md P8, PI ask 2026-08-10).

Trains a small decoder ``z -> BEV occupancy raster`` on the FROZEN world-model trunk
(the §1.10 latents-only discipline: the trunk is NEVER updated, so the readout measures
what the latent ALREADY carries), then applies the SAME decoder to PREDICTED latents
ẑ_{t+k} — ``decode(ẑ_{t+k})`` vs the GT raster at t+k is the picture of what the WM
believes the world will look like.

TRAINING TARGET (one lever, stated):
  * The BCE loss is computed ONLY on the ENCODED-latent pair — ``decode(z_t)`` vs the
    GT raster at the window's present frame t. The PREDICTED-latent path is EVAL-ONLY:
    it measures what the predictor RETAINS of the scene. Training the decoder on
    predicted latents would let it compensate for predictor blur and the probe would
    stop answering "does the predicted latent still carry the scene" — the exact
    question P8 exists to answer.

PRE-REGISTERED GATE (WM_PHYSICS_PROOF.md P8, committed before any number):
  (a) IoU(decode(ẑ_{t+k})) >= 0.8 x IoU(decode(z_{t+k})) at k=10 — prediction retains
      the scene, not just ego. Written to ``<out>/p8_gate.json``.
  Gate (b) of the doc (occluded-agent POSITIONAL error < 2x visible) is a per-agent
  probe belonging to the P4-join harness; this trainer reports the visible/occluded
  CELL-RECALL split when (and only when) the join file carries occlusion flags, and
  says "n/a + reason" when it does not (`obstacle.offline` has no native flag).

DATA PATH — honest note (verified from source, not assumed):
  The episode corpus does NOT carry agent tracks — the episode contract is
  frames/actions/poses/episode_id/maneuvers only (tanitad/data/_contract.py:8-12,
  physicalai.build_episode -> ToyEpisode, v2_dataset.LazyV2Episode.__slots__). The
  raster therefore reaches this trainer through ``--raster-source``:
    * ``join-file`` (the ONLY working source today): a pod-built jsonl of
      ``{"clip_id": str, "frame_idx": int, "agents": [{cx, cy, yaw, l, w[, occ]}]}``
      with agents in the EGO frame at that frame (+x fwd, +y left —
      refb_labels.ego_frame's convention, scripts/refb_labels.py:86-90) and
      ``frame_idx`` in EPISODE index space (the post-n_stack-trim space that
      ``taniteval.lead_source.register_poses_to_time`` fits its time grid over —
      NOT the raw clip frame number). Building that file is the POD-SIDE join step
      (clip_id + registered frame time -> `obstacle.offline` rows ->
      ``tanitad.data.bev_raster.agents_at_time``); this trainer does not fake it.
      A frame ABSENT from the file is NO_LABEL and is skipped+counted — never read
      as empty road (2026-08-03-obstacle-offline-join §4). An empty ``agents`` list
      IS a label: road clear.
    * ``episode``: reserved for a future corpus rebuild that attaches per-frame
      agent tracks to episodes; today it REFUSES loudly (no provider has ``.agents``).

SEAMS (imported, never re-implemented):
  * corpus/parity/geometry: ``train_v58f_unicycle_head.build_train_episodes`` +
    ``eval_flagship_v4.build_v2_val_episodes`` / ``resolve_eval_frames`` / ``_eval_cfg``
    / ``_plan`` (the W4-family loader seams, byte-identical guards);
  * frozen trunk: ``eval_flagship_v4.load_v1_from_ck`` (MODE A: model + grounding, no
    planner head needed for a WM readout);
  * predictor roll: ``tanitad.models.metric_dynamics.rollout_transitions``
    (metric_dynamics.py:247-266) — the SAME roll the imagination/canary family uses
    (``train_flagship_v4.canary_rollout`` rolls via its unit-pinned twin
    ``rollout_decode``; ``rollout_transitions`` is the latents-out variant, byte-identical
    roll, which is exactly what P8 decodes). ``trans[k-1][1]`` IS ẑ_{t+k};
  * action lift: the 3-channel speed-append pattern of ``canary_rollout``
    (train_flagship_v4.py:578-580, SPEED_SCALE contract).

TIER/DISCIPLINE STAMP: P8 is a T0-DIAGNOSTIC by design (it interrogates
representations, like P1/P2/P4 — WM_PHYSICS_PROOF.md "Discipline"), NEVER a driving-
performance number; it is one probe row of the WM-physics battery, not a standalone
driving eval, so no ADE and no four-family table is produced here. Plain corpus-grid
means; the decision-grade interval for any registry claim is the episode-cluster
bootstrap (taniteval/ci.py) — run it pod-side before publishing.

⚠️ POD-SIDE ONLY for the full path (GPU + v5f/v4 checkpoint + v2 corpora + the join
file). Runnable here: ``python -m py_compile`` and the CPU tests
(``stack/tests/test_p8.py``: rasteriser exactness, decoder shapes/params band,
gate-JSON both branches, join-file reader roundtrip).

Usage (pod; PYTHONPATH=/workspace/TanitAD/stack):

  python3 scripts/train_p8_occupancy.py \
      --ckpt /workspace/experiments/flagship-v5f-.../ckpt_step30000.pt \
      --v2-cache  /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --raster-source join-file --join-file /workspace/data/p8_join/agents.jsonl \
      --out /workspace/experiments/p8-occupancy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tanitad.data.bev_raster import (BEVGrid, GRID_DEFAULT,  # noqa: E402
                                     agents_to_array, rasterize)

KS_DEFAULT = (5, 10, 15, 20)          # 0.5/1/1.5/2 s @10 Hz — the WP_STEPS grid
GATE_K = 10                           # gate (a) horizon (WM_PHYSICS_PROOF.md P8)
GATE_RETENTION = 0.8                  # IoU(ẑ) >= 0.8 x IoU(z) at k=GATE_K
PARAM_BAND = (500_000, 2_000_000)     # "~1M params" — asserted, not hoped


# ============================================================================
# decoder — 2 conv-transpose layers from the flat latent (~1M params @ S=2048)
# ============================================================================
class BEVOccupancyHead(nn.Module):
    """``z [B, S] -> occupancy logits [B, nx, ny]`` (nx, ny = grid.shape, 120x64).

    Linear(S -> ch0 * nx/8 * ny/8) reshaped to a [ch0, 15, 8] seed map, then TWO
    ConvTranspose2d layers: (k4, s4) -> [ch1, 60, 32], GELU, (k4, s2, p1) ->
    [1, 120, 64]. At the flagship state_dim 2048 with the default widths the head
    is ~0.985 M params — inside :data:`PARAM_BAND`, which ``enforce_band=True``
    (the trainer's setting) asserts at construction. Tests may pass
    ``enforce_band=False`` to probe shapes at toy widths.
    """

    def __init__(self, state_dim: int, grid: BEVGrid = GRID_DEFAULT,
                 ch0: int = 4, ch1: int = 16, enforce_band: bool = True):
        super().__init__()
        nx, ny = grid.shape
        if nx % 8 or ny % 8:
            raise ValueError(f"grid shape {grid.shape} must be divisible by 8 "
                             f"(two conv-transpose ups: x4 then x2)")
        self.grid = grid
        self.state_dim = int(state_dim)
        self.h0, self.w0 = nx // 8, ny // 8
        self.ch0 = int(ch0)
        self.fc = nn.Linear(state_dim, ch0 * self.h0 * self.w0)
        self.up1 = nn.ConvTranspose2d(ch0, ch1, kernel_size=4, stride=4)
        self.act = nn.GELU()
        self.up2 = nn.ConvTranspose2d(ch1, 1, kernel_size=4, stride=2, padding=1)
        n = self.n_params
        if enforce_band and not (PARAM_BAND[0] <= n <= PARAM_BAND[1]):
            raise ValueError(
                f"BEVOccupancyHead has {n:,} params — outside the pre-registered "
                f"~1M band {PARAM_BAND} (state_dim={state_dim}, ch0={ch0}, "
                f"ch1={ch1}). A bigger head would stop measuring what the LATENT "
                f"carries; pass enforce_band=False only in shape tests.")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.state_dim:
            raise ValueError(f"latent dim {z.shape[-1]} != {self.state_dim}")
        x = self.fc(z).reshape(z.shape[0], self.ch0, self.h0, self.w0)
        x = self.up2(self.act(self.up1(x)))
        return x.squeeze(1)                                  # [B, nx, ny] logits


# ============================================================================
# episode identity — the two id schemes a provider can carry
# ============================================================================
def episode_uid_of_clip(clip_id: str) -> int:
    """The collision-free 63-bit provider id for ``clip_id``.

    Delegates to ``tanitad.data.v2_dataset.stable_episode_id`` (the canonical
    definition) when importable; falls back to the SAME two-line formula
    (blake2b digest_size=8 >> 1, v2_dataset.py:95-96) on hosts without
    torchvision (v2_dataset imports torchvision.io at module level). The two are
    pinned equal in tests/test_p8.py whenever both are importable."""
    try:
        from tanitad.data.v2_dataset import stable_episode_id
        return stable_episode_id(clip_id)
    except ImportError:
        import hashlib
        return int.from_bytes(
            hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(),
            "big") >> 1


def legacy_episode_id_of_clip(clip_id: str) -> int:
    """The 16-bit id baked into payloads/epcaches: first 4 BYTES of the clip id
    (physicalai.py:740; v2 payloads store the same). COLLIDES (~1.4-6.8 % of
    clips — v2_dataset.stable_episode_id docstring), so lookups through this id
    must refuse ambiguous keys rather than guess."""
    return int.from_bytes(clip_id.encode()[:4].ljust(4, b"\0"), "big")


# ============================================================================
# raster sources — the --raster-source seam
# ============================================================================
class JoinFileReader:
    """Reader for the pod-built join file (jsonl; one line per LABELLED frame).

    Line schema: ``{"clip_id": str, "frame_idx": int, "agents": [
    {"cx": f, "cy": f, "yaw": f, "l": f, "w": f[, "occ": 0|1]}]}`` — agents in
    the EGO frame of that frame (+x fwd, +y left), ``frame_idx`` in EPISODE index
    space. Absent (clip, frame) = NO_LABEL -> :meth:`lookup` returns ``None``
    (skip + count, never "road clear"); an empty agents list IS a label (clear).

    Lookup keys: the collision-free 63-bit uid (:func:`episode_uid_of_clip`, the
    id v2 providers carry under the default ``stable_ids=True``) AND the legacy
    16-bit payload id — ambiguous legacy ids (two clips, same first 4 bytes) are
    detected at load and REFUSED at lookup instead of silently joining the wrong
    clip's agents.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self._by_clip: dict[tuple[str, int], np.ndarray] = {}
        self._cls_by_clip: dict[tuple[str, int], np.ndarray] = {}
        self._clip_of_uid: dict[int, str] = {}
        self._clip_of_legacy: dict[int, str] = {}
        self._ambiguous_legacy: set[int] = set()
        self.has_occlusion_flags = False
        self.has_classes = False
        seen_clips: set[str] = set()
        n_lines = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cid = str(rec["clip_id"])
                    fi = int(rec["frame_idx"])
                    ag = agents_to_array(rec["agents"])
                except (KeyError, TypeError, ValueError) as ex:
                    raise ValueError(f"{self.path}:{ln}: bad join record "
                                     f"({ex!r})") from ex
                key = (cid, fi)
                if key in self._by_clip:
                    raise ValueError(f"{self.path}:{ln}: duplicate record for "
                                     f"clip {cid!r} frame {fi} — the join "
                                     f"builder emitted this frame twice")
                self._by_clip[key] = ag
                # per-agent label_class ("cls" in the 2026-08 join schema) —
                # kept ALIGNED with the [A, 6] rows so consumers (the P1
                # lead-gap probe) can filter candidates by class. Older join
                # files without the field stay loadable: has_classes False,
                # lookup_classes -> None, behaviour unchanged.
                if isinstance(rec["agents"], list) and any(
                        isinstance(d, dict) and "cls" in d
                        for d in rec["agents"]):
                    self._cls_by_clip[key] = np.asarray(
                        [str(d.get("cls", "")) for d in rec["agents"]],
                        dtype=object)
                    self.has_classes = True
                if (ag.shape[0] > 0) and (ag[:, 5] >= 0.0).any():
                    self.has_occlusion_flags = True
                if cid not in seen_clips:
                    seen_clips.add(cid)
                    self._clip_of_uid[episode_uid_of_clip(cid)] = cid
                    leg = legacy_episode_id_of_clip(cid)
                    if leg in self._clip_of_legacy \
                            and self._clip_of_legacy[leg] != cid:
                        self._ambiguous_legacy.add(leg)
                    else:
                        self._clip_of_legacy[leg] = cid
                n_lines += 1
        self.n_records = n_lines
        self.n_clips = len(self._clip_of_uid)
        if n_lines == 0:
            raise ValueError(f"{self.path}: empty join file")

    def _clip_of(self, episode_id: int) -> str | None:
        cid = self._clip_of_uid.get(int(episode_id))
        if cid is not None:
            return cid
        if int(episode_id) in self._ambiguous_legacy:
            raise RuntimeError(
                f"episode_id {episode_id} is a LEGACY 16-bit id shared by "
                f"multiple clips in this join file — joining would guess. Use "
                f"providers with stable ids (build_v2_providers default) or "
                f"disambiguate the join file.")
        return self._clip_of_legacy.get(int(episode_id))

    def covers_episode(self, episode_id: int) -> bool:
        return self._clip_of(episode_id) is not None

    def lookup(self, episode_id: int, frame_idx: int) -> np.ndarray | None:
        """Agents ``[A, 6]`` at (episode, frame), or ``None`` = NO_LABEL."""
        cid = self._clip_of(episode_id)
        if cid is None:
            return None
        return self._by_clip.get((cid, int(frame_idx)))

    def lookup_classes(self, episode_id: int,
                       frame_idx: int) -> np.ndarray | None:
        """Per-agent ``label_class`` strings aligned with :meth:`lookup`'s rows,
        or ``None`` when the record (or the whole file) carries no classes."""
        cid = self._clip_of(episode_id)
        if cid is None:
            return None
        return self._cls_by_clip.get((cid, int(frame_idx)))

    def raster(self, episode_id: int, frame_idx: int,
               grid: BEVGrid = GRID_DEFAULT,
               subset: str = "all") -> np.ndarray | None:
        """Occupancy raster at (episode, frame) or None. ``subset``: ``all`` |
        ``visible`` (occ==0) | ``occluded`` (occ==1); subsets need flags."""
        ag = self.lookup(episode_id, frame_idx)
        if ag is None:
            return None
        if subset != "all":
            if not self.has_occlusion_flags:
                raise RuntimeError("join file carries no occlusion flags — "
                                   f"subset={subset!r} is not derivable")
            want = 0.0 if subset == "visible" else 1.0
            ag = ag[ag[:, 5] == want]
        return rasterize(ag, grid=grid)


class EpisodeCarriedSource:
    """``--raster-source episode``: read per-frame agents off the providers.

    RESERVED for a future corpus rebuild. Today NO corpus carries agent tracks
    (episode contract: frames/actions/poses/episode_id/maneuvers —
    _contract.py:8-12; `obstacle.offline` was never ingested at build time), so
    construction REFUSES with the pod-side alternative rather than pretending.
    """

    def __init__(self, episodes):
        carried = [ep for ep in episodes if getattr(ep, "agents", None) is not None]
        if not carried:
            raise SystemExit(
                "[p8] --raster-source episode: no provider carries `.agents` — "
                "the episode corpus does NOT include agent tracks (the episode "
                "contract is frames/actions/poses/episode_id/maneuvers only; "
                "`obstacle.offline` is not ingested at build time). Build the "
                "pod-side join (clip_id + registered frame index -> "
                "bev_raster.agents_at_time) and pass "
                "--raster-source join-file --join-file <agents.jsonl>.")
        self._of = {int(ep.episode_id): ep for ep in carried}
        self.has_occlusion_flags = False
        self.n_records = sum(len(ep.agents) for ep in carried)
        self.n_clips = len(carried)

    def covers_episode(self, episode_id: int) -> bool:
        return int(episode_id) in self._of

    def lookup(self, episode_id: int, frame_idx: int) -> np.ndarray | None:
        ep = self._of.get(int(episode_id))
        if ep is None or frame_idx >= len(ep.agents):
            return None
        ag = ep.agents[int(frame_idx)]
        return None if ag is None else agents_to_array(ag)

    def raster(self, episode_id: int, frame_idx: int,
               grid: BEVGrid = GRID_DEFAULT,
               subset: str = "all") -> np.ndarray | None:
        if subset != "all":
            raise RuntimeError("episode-carried source has no occlusion flags")
        ag = self.lookup(episode_id, frame_idx)
        return None if ag is None else rasterize(ag, grid=grid)


def build_raster_source(a, episodes):
    if a.raster_source == "join-file":
        if not a.join_file:
            raise SystemExit("[p8] --raster-source join-file needs --join-file")
        return JoinFileReader(a.join_file)
    return EpisodeCarriedSource(episodes)


# ============================================================================
# metrics + gate (pure, CPU-testable)
# ============================================================================
def iou_at_05(logits: Tensor, target: Tensor) -> Tensor:
    """Per-window IoU at threshold 0.5 (logit > 0). ``[B, nx, ny]`` -> ``[B]``.

    Empty-union windows (no GT and no predicted occupancy) are NaN — excluded
    from means rather than scored 1.0, so an all-empty road cannot inflate the
    gate; the n that survives travels with every mean."""
    pred = (logits > 0.0).float()
    tgt = (target > 0.5).float()
    inter = (pred * tgt).sum(dim=(-2, -1))
    union = ((pred + tgt) > 0.5).float().sum(dim=(-2, -1))
    return torch.where(union > 0, inter / union.clamp_min(1.0),
                       torch.full_like(union, float("nan")))


#: mini-eval operating-point sweep (attempt-2 fix, 2026-08-11): attempt 1
#: measured IoU only at sigmoid 0.5 on an all-empty-collapsed readout and got
#: 0.0003 vs 0.0001 — a ratio of noise. The sweep finds the readout's actual
#: operating point; tau* is chosen on the ENCODED arm, which makes the
#: pred/enc >= 0.8 retention gate strictly HARDER, never easier.
TAU_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def iou_at_tau(logits: Tensor, target: Tensor, tau: float) -> Tensor:
    """Per-window IoU at sigmoid threshold ``tau``. ``[B, nx, ny]`` -> ``[B]``."""
    pred = (torch.sigmoid(logits) > tau).float()
    tgt = (target > 0.5).float()
    inter = (pred * tgt).sum(dim=(-2, -1))
    union = ((pred + tgt) > 0.5).float().sum(dim=(-2, -1))
    return torch.where(union > 0, inter / union.clamp_min(1.0),
                       torch.full_like(union, float("nan")))


def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    """Mean soft-Dice loss — the imbalance-robust overlap objective (attempt-2
    fix): an all-empty prediction scores ~1.0 here regardless of how rare the
    positive class is, which is exactly the collapse BCE alone rewarded."""
    p = torch.sigmoid(logits)
    tgt = (target > 0.5).float()
    inter = (p * tgt).sum(dim=(-2, -1))
    denom = p.sum(dim=(-2, -1)) + tgt.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def cell_recall(logits: Tensor, subset_target: Tensor,
                tau: float = 0.5) -> Tensor:
    """Fraction of subset-GT cells the decoded raster marks occupied ``[B]``;
    NaN where the subset is empty. (The visible/occluded split metric: recall
    against a SUBSET raster is well-defined where IoU is confounded — the
    decoder legitimately predicts the other subset's cells too.)"""
    pred = (torch.sigmoid(logits) > tau).float()
    tgt = (subset_target > 0.5).float()
    n = tgt.sum(dim=(-2, -1))
    hit = (pred * tgt).sum(dim=(-2, -1))
    return torch.where(n > 0, hit / n.clamp_min(1.0),
                       torch.full_like(n, float("nan")))


def _mean_n(vals: list[float]) -> tuple[float | None, int]:
    v = [x for x in vals if x == x]                       # drop NaN
    return (float(np.mean(v)) if v else None), len(v)


def p8_gate_dict(per_k: dict, gate_k: int = GATE_K,
                 retention: float = GATE_RETENTION) -> dict:
    """Gate (a) verdict from the per-k mini-eval rows (pure; both branches
    tested CPU-side). ``per_k[k]`` needs ``iou_enc``/``iou_pred`` (means, None
    when n=0) and ``n_enc``/``n_pred``.

    Not-computable (missing k, n=0, or iou_enc == 0 — retention against a
    failed readout is undefined) yields ``pass: None``, never a fake verdict.
    """
    row = per_k.get(gate_k, per_k.get(str(gate_k)))
    gate = {"rule": f"IoU(decode(z_hat_t+k)) >= {retention} x "
                    f"IoU(decode(z_t+k encoded)) at k={gate_k}",
            "k": gate_k, "retention": retention}
    reason = None
    if row is None:
        reason = f"k={gate_k} not evaluated"
    elif not row.get("n_enc") or row.get("iou_enc") is None:
        reason = f"no encoded-path windows at k={gate_k} (n_enc=0)"
    elif not row.get("n_pred") or row.get("iou_pred") is None:
        reason = f"no predicted-path windows at k={gate_k} (n_pred=0)"
    elif row["iou_enc"] <= 0.0:
        reason = ("encoded IoU is 0 — the readout itself failed; retention "
                  "is undefined (fix the readout before gating the predictor)")
    if reason is not None:
        gate.update(computable=False, reason=reason, ratio=None,
                    iou_enc=None if row is None else row.get("iou_enc"),
                    iou_pred=None if row is None else row.get("iou_pred"),
                    **{"pass": None})
        return {"gate_a": gate, "PASS": None}
    ratio = float(row["iou_pred"]) / float(row["iou_enc"])
    ok = bool(ratio >= retention)
    gate.update(computable=True, reason=None,
                iou_enc=float(row["iou_enc"]), iou_pred=float(row["iou_pred"]),
                n_enc=int(row["n_enc"]), n_pred=int(row["n_pred"]),
                ratio=round(ratio, 6), **{"pass": ok})
    return {"gate_a": gate, "PASS": ok}


# ============================================================================
# frozen forward — encode + the imagination/canary-family predictor roll
# ============================================================================
def lift_actions3(aw2: Tensor, fa2: Tensor, v0: Tensor) -> tuple[Tensor, Tensor]:
    """2-channel (steer, accel) -> the 3-channel speed-append format the
    predictor trains with — the exact ``canary_rollout`` pattern
    (train_flagship_v4.py:578-580; SPEED_SCALE contract)."""
    from tanitad.models.flagship_v15 import SPEED_SCALE
    vch = (v0 / SPEED_SCALE)[:, None, None]
    return (torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], dim=-1),
            torch.cat([fa2, vch.expand(-1, fa2.shape[1], -1)], dim=-1))


@torch.no_grad()
def p8_latents(world, batch: dict, ks: tuple[int, ...], *, amp_on: bool,
               want_pred: bool, want_enc_k: bool):
    """Frozen-trunk latents for one batch: ``(z_t, {k: z_enc_k}, {k: z_hat_k})``.

    * ``z_t`` — encoded latent of the present frame (frames[:, -1]); identical to
      ``encode_window(frames)[:, -1]`` because ``encode_window`` is per-frame
      (fourbrain.py:474-478) — encoded HERE via one ``encode_window`` over
      frames(+futures) so every window/roll below reuses the same pass.
    * ``z_enc_k`` — latent of the TRUE frame at t+k, i.e. the window shifted k
      steps: with per-frame encoding, ``states_all[:, k+W-1]`` of the once-encoded
      ``[frames | future_frames]`` sequence (bit-identical to re-encoding the
      shifted window, no second encoder pass).
    * ``z_hat_k`` — the PREDICTED latent: ``rollout_transitions`` (the
      imagination/canary-family roll, metric_dynamics.py:247-266) under TRUE
      actions, lifted to 3 channels; ``trans[k-1][1]`` is ẑ_{t+k}.
    """
    frames = batch["frames"]
    fut = batch["future_frames"]
    k_max = max(ks) if ks else 0
    w = frames.shape[1]
    if want_enc_k and fut.shape[1] < k_max:
        raise ValueError(f"future_frames covers {fut.shape[1]} < k_max={k_max}")
    dev_type = frames.device.type
    with torch.autocast(dev_type, dtype=torch.bfloat16,
                        enabled=amp_on and dev_type == "cuda"):
        if want_enc_k or want_pred:
            seq = torch.cat([frames, fut[:, :k_max]], dim=1) if want_enc_k \
                else frames
            states_all = world.encode_window(seq)            # [B, W(+k), S]
            z_t = states_all[:, w - 1]
        else:
            z_t = world.encode_window(frames[:, -1:])[:, 0]
        z_enc = {k: states_all[:, w - 1 + k] for k in ks} if want_enc_k else {}
        z_hat = {}
        if want_pred:
            from tanitad.models.metric_dynamics import rollout_transitions
            v0 = batch["pose_last"][:, 3].float()
            aw3, fa3 = lift_actions3(batch["actions"].float(),
                                     batch["future_actions"].float(), v0)
            trans = rollout_transitions(world.predictor, states_all[:, :w],
                                        aw3, fa3, k_max)
            z_hat = {k: trans[k - 1][1] for k in ks}
    return (z_t.float(), {k: v.float() for k, v in z_enc.items()},
            {k: v.float() for k, v in z_hat.items()})


# ============================================================================
# raster plumbing — window -> (episode uid, present frame) -> GT raster
# ============================================================================
def window_frame(ds, i: int) -> tuple[int, int]:
    """Dataset window i -> (episode uid, present frame index). Present =
    ``t + window - 1`` — the same origin rule as the canonical 881-window grid
    (rollout.collect: origin start+7 at window 8; join doc §5)."""
    e_i, t = ds.index[i]
    return int(ds.episodes[e_i].episode_id), t + ds.window - 1


def batch_rasters(ds, idx: list[int], source, k: int,
                  grid: BEVGrid, subset: str = "all"
                  ) -> tuple[Tensor | None, list[int], int]:
    """GT rasters at present+k for the windows ``idx``. Returns
    ``(rasters [n, nx, ny] | None, kept_positions, n_no_label)`` — NO_LABEL
    windows are DROPPED AND COUNTED, never rasterised as empty road."""
    keep, mats = [], []
    for pos, i in enumerate(idx):
        eid, pf = window_frame(ds, i)
        r = source.raster(eid, pf + k, grid=grid, subset=subset)
        if r is None:
            continue
        keep.append(pos)
        mats.append(torch.from_numpy(r))
    if not keep:
        return None, [], len(idx)
    return torch.stack(mats), keep, len(idx) - len(keep)


def covered_indices(ds, source, k: int = 0) -> list[int]:
    """Window indices whose present(+k) frame the source labels (train filter)."""
    out = []
    for i in range(len(ds.index)):
        eid, pf = window_frame(ds, i)
        if source.lookup(eid, pf + k) is not None:
            out.append(i)
    return out


def make_covered_sampler(ds, covered: list[int], eps_per_batch: int,
                         rng: random.Random):
    """Episode-grouped sampler over the COVERED windows only — the
    ``train_v58f_unicycle_head.make_sampler`` I/O shape (few episodes x many
    windows; ~8x fewer cold MooseFS payload loads), restricted to labelled
    windows so a batch never wastes a decode on NO_LABEL."""
    ep2idx: dict[int, list[int]] = {}
    for i in covered:
        e_i, _t = ds.index[i]
        ep2idx.setdefault(e_i, []).append(i)
    ep_ids = list(ep2idx)
    if not ep_ids:
        raise SystemExit("[p8] no covered windows to sample")

    def sample(bs: int) -> list[int]:
        chosen = [ep_ids[rng.randrange(len(ep_ids))]
                  for _ in range(min(eps_per_batch, len(ep_ids)))]
        out = []
        gi = 0
        while len(out) < bs:
            pool = ep2idx[chosen[gi % len(chosen)]]
            out.append(pool[rng.randrange(len(pool))])
            gi += 1
        return out

    return sample


# ============================================================================
# mini-eval — per-k IoU for encoded vs predicted latents (+ occlusion split)
# ============================================================================
@torch.no_grad()
def mini_eval(world, head, ds_val, source, device, *, ks: tuple[int, ...],
              grid: BEVGrid, amp_on: bool, episodes: int = 40, stride: int = 8,
              batch: int = 16) -> dict:
    """IoU@0.5 of ``decode(z_enc_{t+k})`` and ``decode(z_hat_{t+k})`` vs the GT
    raster at t+k, per k, over the eval-default window grid (e < episodes,
    t % stride == 0 — the same rule as canary_rollout / W4). k=0 row = the
    readout itself (train metric on val). Visible/occluded CELL-RECALL split
    only when the join carries flags; else n/a with the reason."""
    from torch.utils.data import default_collate

    from train_flagship_v4 import _to_device
    head.eval()
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        raise SystemExit("[p8] mini-eval selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    ks_all = tuple(sorted(set(ks)))
    acc = {k: {p: {t: [] for t in TAU_GRID} for p in ("enc", "pred")}
           for k in ks_all}
    acc0: dict[float, list[float]] = {t: [] for t in TAU_GRID}
    occ_acc = {k: {p: {s: {t: [] for t in TAU_GRID}
                       for s in ("visible", "occluded")}
                   for p in ("enc", "pred")} for k in ks_all} \
        if source.has_occlusion_flags else None
    n_grid = 0
    no_label = {k: 0 for k in (0,) + ks_all}
    t0 = time.time()
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        n_grid += len(idx)
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        z_t, z_enc, z_hat = p8_latents(world, b, ks_all, amp_on=amp_on,
                                       want_pred=True, want_enc_k=True)
        r0, keep0, miss0 = batch_rasters(ds_val, idx, source, 0, grid)
        no_label[0] += miss0
        if r0 is not None:
            l0 = head(z_t[keep0])
            r0d = r0.to(device)
            for t in TAU_GRID:
                acc0[t].extend(iou_at_tau(l0, r0d, t).cpu().tolist())
        for k in ks_all:
            rk, keep, miss = batch_rasters(ds_val, idx, source, k, grid)
            no_label[k] += miss
            if rk is None:
                continue
            rk = rk.to(device)
            log_enc = head(z_enc[k][keep])
            log_pred = head(z_hat[k][keep])
            for t in TAU_GRID:
                acc[k]["enc"][t].extend(iou_at_tau(log_enc, rk, t).cpu().tolist())
                acc[k]["pred"][t].extend(iou_at_tau(log_pred, rk, t).cpu().tolist())
            if occ_acc is not None:
                for s in ("visible", "occluded"):
                    rs, ks_keep, _m = batch_rasters(ds_val, idx, source, k,
                                                    grid, subset=s)
                    if rs is None:
                        continue
                    rs = rs.to(device)
                    sub = [keep.index(p) for p in ks_keep if p in keep]
                    rows = [ks_keep.index(keep[j]) for j in sub]
                    if not sub:
                        continue
                    for t in TAU_GRID:
                        occ_acc[k]["enc"][s][t].extend(cell_recall(
                            log_enc[sub], rs[rows], tau=t).cpu().tolist())
                        occ_acc[k]["pred"][s][t].extend(cell_recall(
                            log_pred[sub], rs[rows], tau=t).cpu().tolist())
    head.train()
    # tau* on the ENCODED arm, pooled over all ks (one operating point for the
    # readout; choosing on enc makes the pred/enc retention gate harder).
    pooled_enc = {t: [x for k in ks_all for x in acc[k]["enc"][t]]
                  for t in TAU_GRID}
    tau_means = {t: _mean_n(pooled_enc[t])[0] for t in TAU_GRID}
    tau_star = max(TAU_GRID,
                   key=lambda t: (-1.0 if tau_means[t] is None
                                  else tau_means[t]))
    per_k = {}
    for k in ks_all:
        me, ne = _mean_n(acc[k]["enc"][tau_star])
        mp, np_ = _mean_n(acc[k]["pred"][tau_star])
        me5, _ = _mean_n(acc[k]["enc"][0.5])
        mp5, _ = _mean_n(acc[k]["pred"][0.5])
        per_k[k] = {"iou_enc": me, "n_enc": ne, "iou_pred": mp, "n_pred": np_,
                    "iou_enc_at_05": me5, "iou_pred_at_05": mp5,
                    "n_no_label": no_label[k]}
    m0, n0 = _mean_n(acc0[tau_star])
    if occ_acc is not None:
        split = {"available": True, "metric": "cell recall vs subset raster "
                 "(IoU vs a subset is confounded — the decoder legitimately "
                 "predicts the complement subset's cells)"}
        for k in ks_all:
            row = {}
            for p in ("enc", "pred"):
                for s in ("visible", "occluded"):
                    mv, nv = _mean_n(occ_acc[k][p][s][tau_star])
                    row[f"recall_{s}_{p}"] = mv
                    row[f"n_{s}_{p}"] = nv
            split[str(k)] = row
    else:
        split = {"available": False,
                 "reason": "join file carries no occlusion flags — "
                           "`obstacle.offline` has no native visibility field; "
                           "the flags come from the pod-side P4 join, which was "
                           "not provided here"}
    return {"iou_k0_readout": m0, "n_k0": n0, "per_k": per_k,
            "tau_star": tau_star,
            "tau_rule": "argmax mean ENCODED-arm IoU pooled over ks (harder "
                        "for the pred/enc retention gate, never easier)",
            "tau_sweep_enc_pooled": {str(t): tau_means[t] for t in TAU_GRID},
            "visible_occluded_split": split,
            "grid_rule": {"episodes": episodes, "stride": stride,
                          "batch": batch, "n_grid_windows": n_grid},
            "wallclock_s": round(time.time() - t0, 1)}


# ============================================================================
# main (POD-SIDE: GPU + checkpoint + v2 corpora + join file)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("train_p8_occupancy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint with model+grounding keys (v1/v4/v5f all "
                         "qualify — MODE A load, planner head unused)")
    # corpus seams — byte-identical to the W4 arg surface
    ap.add_argument("--v2-cache", required=True, nargs="+",
                    help="v2 compressed TRAIN split dir(s) — the canonical "
                         "physicalai-train-e438721ae894 build")
    ap.add_argument("--v2-val-cache", required=True, nargs="+")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # raster source
    ap.add_argument("--raster-source", choices=("join-file", "episode"),
                    default="join-file",
                    help="'join-file' (the working path): pod-built jsonl of "
                         "{clip_id, frame_idx, agents}; 'episode' refuses today "
                         "(corpus carries no agent tracks) — reserved for a "
                         "rebuild that attaches them")
    ap.add_argument("--join-file", default=None,
                    help="agents jsonl covering train+val clips (one file; "
                         "clips resolve per episode id)")
    # training
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eps-per-batch", type=int, default=4)
    ap.add_argument("--pos-weight", default="auto",
                    help="BCE positive-class weight: 'auto' (DEFAULT since the "
                         "2026-08-11 attempt-2 fix: neg/pos ratio measured on "
                         "a sample of training rasters, capped at 200 — "
                         "attempt 1 ran at 1.0 and collapsed to all-empty) "
                         "or an explicit float")
    ap.add_argument("--w-dice", type=float, default=0.5,
                    help="weight of the soft-Dice term added to BCE (0 "
                         "disables; the imbalance-robust overlap objective — "
                         "attempt-2 fix)")
    ap.add_argument("--pos-weight-sample", type=int, default=256,
                    help="labelled train windows sampled for the 'auto' "
                         "pos-weight measurement")
    ap.add_argument("--ch0", type=int, default=4)
    ap.add_argument("--ch1", type=int, default=16)
    ap.add_argument("--ks", default="5,10,15,20",
                    help="prediction horizons for the eval (0.1 s ticks)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    # mini-eval grid (eval defaults)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[p8] WARNING: cuda unavailable, falling back to cpu", flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    ks = tuple(int(x) for x in str(a.ks).split(",") if x)
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v1_from_ck, resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_flagship_v4 import _to_device
    from train_v58f_unicycle_head import build_train_episodes, module_md5

    # ---- geometry FIRST (the W4/eval seam, not re-resolved here) ------------
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg,
                                                   label="train_p8_occupancy")
    plan = _plan(cfg)
    if plan.max_horizon < max(ks):
        raise SystemExit(f"[p8] --ks max {max(ks)} > plan.max_horizon "
                         f"{plan.max_horizon} — future_frames cannot cover it")

    # ---- frozen trunk: MODE A (model + grounding; no planner head) ----------
    print(f"[p8] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, _grounding, base_step = load_v1_from_ck(ck, device,
                                                   frame=model_frame)
    del ck
    assert not any(p.requires_grad for p in world.parameters())
    md5_before = module_md5(world)
    print(f"[p8] trunk frozen · base step {base_step} · "
          f"state_dim {world.state_dim} · md5 {md5_before[:12]}", flush=True)

    # ---- data (the W4-family loader seams, imported) ------------------------
    train_eps, train_prov = build_train_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    # FlagshipWindowDataset (v1 keys) — P8 reads frames/future_frames/actions/
    # future_actions/pose_last only; the v4 label mint (FlagshipV4Dataset) feeds
    # a planner head P8 does not run, so it is deliberately not paid for.
    ds_train = FlagshipWindowDataset(train_eps, window=cfg.predictor.window,
                                     max_horizon=plan.max_horizon,
                                     maneuver_h=plan.maneuver_h,
                                     channels=cfg.encoder.in_channels)
    ds_val = FlagshipWindowDataset(val_eps, window=cfg.predictor.window,
                                   max_horizon=plan.max_horizon,
                                   maneuver_h=plan.maneuver_h,
                                   channels=cfg.encoder.in_channels)
    print(f"[p8] train {len(train_eps)} eps / {len(ds_train)} windows; "
          f"val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)

    # ---- raster source + coverage census (NO_LABEL is counted, never zeroed) -
    source = build_raster_source(a, train_eps)
    covered = covered_indices(ds_train, source, k=0)
    # episode-disjoint guard: when --v2-cache IS the val corpus, the mini-eval
    # grid (episodes < a.episodes in provider order) must never train the
    # decoder. Harmless when the corpora differ (train-corpus episode indices
    # also start at 0, but their labels come from a disjoint clip set).
    if any(c in a.v2_cache for c in a.v2_val_cache):
        n_before = len(covered)
        covered = [i for i in covered if ds_train.index[i][0] >= a.episodes]
        print(f"[p8] eval-grid exclusion: dropped {n_before - len(covered)} "
              f"windows of episodes < {a.episodes} from TRAINING "
              f"(episode-disjoint from the mini-eval grid)", flush=True)
    n_cov_eps = len({ds_train.index[i][0] for i in covered})
    print(f"[p8] raster source: {a.raster_source} ({source.n_records} records, "
          f"{source.n_clips} clips, occlusion_flags="
          f"{source.has_occlusion_flags})", flush=True)
    print(f"[p8] coverage: {len(covered)}/{len(ds_train)} train windows "
          f"labelled ({n_cov_eps} episodes); "
          f"{len(ds_train) - len(covered)} NO_LABEL (skipped+counted)",
          flush=True)
    if not covered:
        raise SystemExit(
            "[p8] the join file labels 0 train windows — its clip_ids/frame "
            "indices do not match this corpus (frame_idx must be EPISODE index "
            "space; ids must be the providers' clip ids). Refusing to train on "
            "nothing.")
    sample = make_covered_sampler(ds_train, covered, a.eps_per_batch, rng)

    # ---- the readout head (the ONLY trainable module; band asserted) --------
    head = BEVOccupancyHead(world.state_dim, grid=GRID_DEFAULT,
                            ch0=a.ch0, ch1=a.ch1, enforce_band=True).to(device)
    n_par = head.n_params
    print(f"[p8] BEVOccupancyHead {tuple(GRID_DEFAULT.shape)} "
          f"({n_par/1e6:.3f} M trainable; band {PARAM_BAND}; frozen everything "
          f"else)", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    # ---- pos-weight: measured from the data unless overridden ---------------
    if str(a.pos_weight).lower() == "auto":
        smp = rng.sample(covered, k=min(a.pos_weight_sample, len(covered)))
        n_pos = n_tot = 0
        for j0 in range(0, len(smp), 64):
            r, _keep, _m = batch_rasters(ds_train, smp[j0:j0 + 64], source, 0,
                                         GRID_DEFAULT)
            if r is None:
                continue
            n_pos += int((r > 0.5).sum())
            n_tot += int(r.numel())
        if n_pos == 0:
            raise SystemExit("[p8] pos-weight auto: sampled rasters contain "
                             "ZERO occupied cells — the join/grid is broken, "
                             "refusing to train on empties")
        pos_weight = min(200.0, (n_tot - n_pos) / n_pos)
        print(f"[p8] pos-weight AUTO: occupied {n_pos}/{n_tot} cells "
              f"({100.0 * n_pos / n_tot:.3f} %) over {len(smp)} sampled "
              f"windows -> pos_weight {pos_weight:.1f} (cap 200)", flush=True)
    else:
        pos_weight = float(a.pos_weight)
        print(f"[p8] pos-weight EXPLICIT: {pos_weight}", flush=True)
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    log_path = os.path.join(a.out, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({
        "run": "p8-bev-occupancy-readout", "args": vars(a),
        "pos_weight_effective": pos_weight, "w_dice": a.w_dice,
        "base_ckpt": a.ckpt, "base_step": base_step,
        "trunk_md5": md5_before, "n_trainable": n_par,
        "grid": {"x_fwd_m": GRID_DEFAULT.x_fwd_m,
                 "y_half_m": GRID_DEFAULT.y_half_m,
                 "cell_m": GRID_DEFAULT.cell_m,
                 "shape": list(GRID_DEFAULT.shape)},
        "coverage": {"train_windows": len(ds_train),
                     "train_windows_labelled": len(covered),
                     "train_episodes_labelled": n_cov_eps},
        "train_parity": {"n_dirs": len(a.v2_cache)},
        "_evidence_class": "MEASURED (ours; artifact = this log)"}) + "\n")
    fh.flush()

    history: list[dict] = []
    acc = {"loss": 0.0, "iou": 0.0, "n": 0, "n_iou": 0}
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = sample(a.bs)
        b = _to_device(default_collate([ds_train[i] for i in idx]), device)
        # ENCODED-latent training pair only (predicted path is EVAL-ONLY — see
        # module docstring). z_t under no_grad: frozen-trunk means the latent is
        # a constant input to the readout.
        z_t, _, _ = p8_latents(world, b, (), amp_on=amp_on,
                               want_pred=False, want_enc_k=False)
        tgt, keep, _miss = batch_rasters(ds_train, idx, source, 0, GRID_DEFAULT)
        if tgt is None:                       # all-NO_LABEL batch (rare: the
            continue                          # sampler draws from covered only)
        tgt = tgt.to(device)
        logits = head(z_t[keep])
        loss = bce(logits, tgt)
        if a.w_dice > 0.0:
            loss = loss + a.w_dice * soft_dice_loss(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()

        bs = tgt.shape[0]
        with torch.no_grad():
            v = iou_at_05(logits, tgt)
            vok = v[torch.isfinite(v)]
        acc["loss"] += float(loss.detach()) * bs
        acc["iou"] += float(vok.sum()) if vok.numel() else 0.0
        acc["n"] += bs
        acc["n_iou"] += int(vok.numel())

        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(float(loss.detach()), 5),
                   "batch_iou": round(float(vok.mean()), 4) if vok.numel()
                   else None,
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.save_every == 0:
            row = {"step": step,
                   "loss": round(acc["loss"] / max(acc["n"], 1), 5),
                   "train_iou": round(acc["iou"] / max(acc["n_iou"], 1), 5),
                   "n_windows": acc["n"],
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            acc = {"loss": 0.0, "iou": 0.0, "n": 0, "n_iou": 0}
            with open(os.path.join(a.out, "metrics.json"), "w") as mf:
                json.dump({"history": history, "args": vars(a),
                           "base_step": base_step,
                           "_read": "train_iou is a TRAIN-batch running mean; "
                                    "the gate numbers are the held-out "
                                    "mini-eval in p8_gate.json",
                           "_evidence_class": "MEASURED (ours)"}, mf, indent=1)
            torch.save({"head": head.state_dict(),
                        "state_dim": world.state_dim,
                        "grid": {"x_fwd_m": GRID_DEFAULT.x_fwd_m,
                                 "y_half_m": GRID_DEFAULT.y_half_m,
                                 "cell_m": GRID_DEFAULT.cell_m},
                        "ch0": a.ch0, "ch1": a.ch1,
                        "step": step, "args": vars(a), "base_ckpt": a.ckpt,
                        "base_step": base_step},
                       os.path.join(a.out, "p8_head.pt"))
            fh.write(json.dumps({"per500": row}) + "\n")
            fh.flush()
            print(f"[p8 @{step}] {row}", flush=True)

    # ---- frozen proof + the pre-registered gate -----------------------------
    md5_after = module_md5(world)
    ev = mini_eval(world, head, ds_val, source, device, ks=ks,
                   grid=GRID_DEFAULT, amp_on=amp_on, episodes=a.episodes,
                   stride=a.stride, batch=a.eval_batch)
    gate = p8_gate_dict(ev["per_k"])
    summary = {
        "probe": "P8 BEV occupancy readout (WM_PHYSICS_PROOF.md P8)",
        "tier": "T0-diagnostic (representation probe — NEVER a driving-"
                "performance number; one row of the WM-physics battery, not a "
                "standalone driving eval)",
        "mini_eval": ev,
        **gate,
        "head": {"n_trainable": n_par, "param_band": list(PARAM_BAND),
                 "ch0": a.ch0, "ch1": a.ch1,
                 "grid_shape": list(GRID_DEFAULT.shape)},
        "raster_source": {"kind": a.raster_source,
                          "path": a.join_file,
                          "n_records": source.n_records,
                          "n_clips": source.n_clips,
                          "occlusion_flags": source.has_occlusion_flags},
        "training_target": "BCE on decode(z_t) vs GT raster at t — ENCODED "
                           "latents only; predicted path eval-only by design",
        "trunk_frozen_proof": {"md5_before": md5_before,
                               "md5_after": md5_after,
                               "identical": md5_before == md5_after},
        "steps": a.steps, "base_ckpt": a.ckpt, "base_step": base_step,
        "wall_s": round(time.time() - t0, 1),
        "_estimator_note": ("plain corpus-grid means over the eval-default "
                            "window grid; the DECISION-grade interval for any "
                            "registry claim is the episode-cluster bootstrap "
                            "(taniteval/ci.py) — run it before publishing"),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }
    with open(os.path.join(a.out, "p8_gate.json"), "w") as gf:
        json.dump(summary, gf, indent=1)
    fh.write(json.dumps({"summary": summary}) + "\n")
    fh.close()
    print(f"\n[P8 SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    if not summary["trunk_frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK CHANGED DURING TRAINING — run invalid")
    verdict = ("PASS" if gate["PASS"] else
               "FAIL" if gate["PASS"] is not None else "NOT COMPUTABLE")
    print(f"[P8 GATE] {verdict} ({json.dumps(gate['gate_a'])})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
