"""Arm adapters: the three architecture arms behind one replay interface.

Every arm consumes the engine's :class:`~tanitad.replay.engine.WindowBatch`
and returns one :class:`~tanitad.replay.engine.ArmOutput` per window:

* :class:`MainArm`  — the latent world model (WorldModel ckpt): encoder state
  -> ridge-probe waypoint decode (D1 pattern), predictor imagination decoded
  by A3-calibrated per-horizon probes (the imagination fan), inverse-dynamics
  action readout, imag_rel self-monitor signal (A9), H15 belief sigma.
* :class:`RefAArm`  — frozen-DINO reference: ONLINE DINOv2-B/14 tokenization
  (scripts/dino_precompute.py pattern) -> adapter states -> the SAME decode
  path as MainArm (the comparison isolates the encoder axis).
* :class:`RefBArm`  — E2E reference: direct tactical waypoints, maneuver
  distribution, operative action + 0.5 s sequence, confidence and
  feature-OOD fallback signals (what a no-world-model stack still has).

Probe doctrine: waypoint probes are fitted on a HELD-OUT fit split of the
replay corpus (episode-level disjoint, scripts/evaluate_checkpoint.py
pattern); imagination probes are fitted ON imagined latents (A3 calibrated
decode). Latency timing covers the arm's control path only (encode ->
predict -> decode); ground-truth-dependent diagnostics (imag_rel scoring,
maneuver pseudo-labels) run outside the timer.
"""

from __future__ import annotations

import importlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.models.readout import RidgeProbe
from tanitad.replay.engine import (WAYPOINT_STEPS, ArmOutput, LatencyTimer,
                                   ReplayEngine, ReplayEpisode, WindowBatch,
                                   future_frames_at, gt_disp_at)

# One color per arm, used EVERYWHERE (rerun entities, README, reports) so a
# viewer never has to re-learn the mapping. GT green / main blue matches
# scripts/viz_trajectory_fan.py.
ARM_COLORS: dict[str, tuple[int, int, int]] = {
    "gt": (44, 160, 44),        # green  — ground truth
    "main": (31, 119, 180),     # blue   — main world model
    "refa": (148, 103, 189),    # purple — REF-A frozen-DINO
    "refb": (255, 127, 14),     # orange — REF-B E2E
}


def _script_module(name: str):
    """Import a ``stack/scripts`` module (they are not a package).

    The replay arms reuse the pinned pseudo-label/config code in
    ``scripts/refb_labels.py`` and ``scripts/refa_train.py`` instead of
    duplicating thresholds. Mirrors how the test suite imports scripts."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        if not (scripts / f"{name}.py").exists():
            raise
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        return importlib.import_module(name)


def load_checkpoint_state(path: str | Path) -> tuple[dict, int]:
    """Load a checkpoint -> (state_dict, step). Accepts both the trainer
    format ``{"model": sd, "step": n, ...}`` and a bare state dict."""
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"], int(ck.get("step", -1))
    return ck, -1


# --------------------------------------------------------------------------
# Latent arms (main / REF-A): shared probe-decode machinery
# --------------------------------------------------------------------------

class _LatentArm:
    """Shared logic for arms that decode a latent state via frozen probes.

    Subclasses provide ``_states(batch)`` (control path: window -> latent
    states [B, w, S]) and ``_future_state(batch, k)`` (diagnostics: encoding
    of the TRUE frame at anchor+k), plus ``inv_dyn`` and a ``_predict``
    callable. Everything else — probe fitting on the fit split, A3 imagined
    probes, imag_rel scoring, latency — lives here once.
    """

    name: str = "latent"
    requires_fit = True

    def _init_latent(self, predict: Callable[[Tensor, Tensor], dict],
                     inv_dyn, window: int, horizons: tuple[int, ...],
                     ckpt: str, step: int, probe_alpha: float,
                     compute_imag_rel: bool) -> None:
        if 1 not in horizons:
            raise ValueError(
                f"{self.name}: predictor horizons {horizons} lack k=1 — the "
                f"inverse-dynamics action readout needs the 1-step head")
        self._predict = predict
        self.inv_dyn = inv_dyn
        self.window = int(window)
        self.horizons = tuple(int(k) for k in horizons)
        self.needs_ahead = max(max(WAYPOINT_STEPS), max(self.horizons))
        self.ckpt = str(ckpt)
        self.step = step
        self.probe_alpha = probe_alpha
        self.compute_imag_rel = compute_imag_rel
        self.engine: ReplayEngine | None = None
        self.probe_wp: RidgeProbe | None = None
        self.probe_imag: dict[int, RidgeProbe] = {}
        self.fit_report: dict = {}

    @property
    def _eng(self) -> ReplayEngine:
        if self.engine is None:
            raise RuntimeError(
                f"arm {self.name!r} is not bound to an engine — construct a "
                f"ReplayEngine with this arm before calling run_batch")
        return self.engine

    # subclass surface ------------------------------------------------------
    def _states(self, batch: WindowBatch) -> Tensor:
        raise NotImplementedError

    def _future_state(self, batch: WindowBatch, k: int) -> Tensor:
        raise NotImplementedError

    def _sigma(self, batch: WindowBatch) -> Tensor | None:
        return None                                   # main-arm only (H15)

    # probe fitting -----------------------------------------------------------
    @torch.no_grad()
    def prepare(self, engine: ReplayEngine,
                fit_reps: Sequence[ReplayEpisode]) -> None:
        """Fit the waypoint probe (encoder states -> D1 waypoints) and the
        per-horizon A3 imagination probes (imagined latents -> realized ego
        displacement) on the held-out fit split."""
        self.engine = engine
        states_l: list[Tensor] = []
        wp_l: list[Tensor] = []
        imag_l: dict[int, list[Tensor]] = {k: [] for k in self.horizons}
        disp_l: dict[int, list[Tensor]] = {k: [] for k in self.horizons}
        n = 0
        for batch in engine.iter_batches(fit_reps, stride=engine.fit_stride):
            actions = batch.actions[:, -self.window:]
            with engine.autocast():
                states = self._states(batch)
                preds = self._predict(states, actions)
            states_l.append(states[:, -1].float().cpu())
            wp_l.append(batch.gt_waypoints)
            for k in self.horizons:
                imag_l[k].append(preds[k].float().cpu())
                disp_l[k].append(gt_disp_at(batch, k))
            n += len(batch)
            if n >= engine.max_fit_windows:
                break
        if n < engine.min_fit_windows:
            raise ValueError(
                f"{self.name}: only {n} fit windows collected "
                f"(< {engine.min_fit_windows}) — probes would be meaningless;"
                f" add fit episodes or lower --fit-stride")
        S = torch.cat(states_l)
        WP = torch.cat(wp_l)
        self.probe_wp = RidgeProbe(self.probe_alpha).fit(S, WP.flatten(1))
        self.fit_report = {
            "fit_windows": n,
            "probe_r2_wp": round(self.probe_wp.r2(S, WP.flatten(1)), 4),
        }
        for k in self.horizons:
            Z, D = torch.cat(imag_l[k]), torch.cat(disp_l[k])
            self.probe_imag[k] = RidgeProbe(self.probe_alpha).fit(Z, D)
            self.fit_report[f"probe_r2_imag_k{k}"] = round(
                self.probe_imag[k].r2(Z, D), 4)

    # replay ------------------------------------------------------------------
    @torch.no_grad()
    def run_batch(self, batch: WindowBatch) -> list[ArmOutput]:
        if self.probe_wp is None:
            raise RuntimeError(
                f"arm {self.name!r} has no fitted probes — call "
                f"engine.prepare(fit_episodes) before engine.run()")
        eng = self._eng
        B = len(batch)
        actions = batch.actions[:, -self.window:]
        with LatencyTimer(eng) as lt:
            with eng.autocast():
                states = self._states(batch)
                z_t = states[:, -1]
                preds = self._predict(states, actions)
                act = self.inv_dyn(z_t, preds[1])
            z_cpu = z_t.float().cpu()
            wp = self.probe_wp.predict(z_cpu).reshape(
                B, len(WAYPOINT_STEPS), 2).numpy()
            imag_traj = {k: self.probe_imag[k].predict(
                preds[k].float().cpu()).numpy() for k in self.horizons}
        per_win_ms = lt.ms / B

        # -- diagnostics (need ground truth; outside the latency timer) ------
        imag_rel: dict[int, Tensor] | None = None
        if self.compute_imag_rel:
            imag_rel = {}
            for k in self.horizons:
                with eng.autocast():
                    z_true = self._future_state(batch, k).float()
                scale = (z_true - z_t.float()).norm(dim=-1).clamp_min(1e-8)
                imag_rel[k] = ((preds[k].float() - z_true).norm(dim=-1)
                               / scale).cpu()
        sigma = self._sigma(batch)
        act_np = act.float().cpu().numpy()

        outs = []
        for j in range(B):
            outs.append(ArmOutput(
                latency_ms=per_win_ms,
                waypoints=wp[j],
                waypoint_steps=WAYPOINT_STEPS,
                action=act_np[j],
                imag_rel=({k: float(v[j]) for k, v in imag_rel.items()}
                          if imag_rel is not None else None),
                imag_traj={k: v[j] for k, v in imag_traj.items()},
                sigma=(float(sigma[j]) if sigma is not None else None),
            ))
        return outs

    def describe(self) -> dict:
        return {"ckpt": self.ckpt, "ckpt_step": self.step,
                "window": self.window, "horizons": list(self.horizons),
                "fit": dict(self.fit_report)}


class MainArm(_LatentArm):
    """The main latent world model (tanitad.models.fourbrain.WorldModel).

    Control path per window: ``encode_window`` -> last state -> waypoint
    probe decode + ``imagine`` -> per-horizon A3 probe decode (imagination
    fan) + inverse-dynamics action readout. Diagnostics: imag_rel per horizon
    (A9 self-monitor, per-sample persistence-normalized) and the H15 mean
    belief sigma of the anchor frame (one extra encoder pass; disable with
    ``sigma=False`` if that cost matters).
    """

    name = "main"

    def __init__(self, ckpt: str | Path, cfg=None, device: str = "cpu",
                 probe_alpha: float = 10.0, compute_imag_rel: bool = True,
                 sigma: bool = True):
        from tanitad.config import base250cam_config
        from tanitad.models.fourbrain import WorldModel
        cfg = cfg if cfg is not None else base250cam_config()
        world = WorldModel(cfg)
        sd, step = load_checkpoint_state(ckpt)
        world.load_state_dict(sd)
        self.world = world.to(device).eval()
        self.device = device
        self.compute_sigma = bool(sigma and self.world.imagination is not None)
        self._init_latent(
            predict=self.world.imagine, inv_dyn=self.world.inv_dyn,
            window=cfg.predictor.window, horizons=cfg.predictor.horizons,
            ckpt=ckpt, step=step, probe_alpha=probe_alpha,
            compute_imag_rel=compute_imag_rel)

    def _states(self, batch: WindowBatch) -> Tensor:
        return self.world.encode_window(batch.frames[:, -self.window:])

    def _future_state(self, batch: WindowBatch, k: int) -> Tensor:
        return self.world.encode(future_frames_at(batch, k))

    @torch.no_grad()
    def _sigma(self, batch: WindowBatch) -> Tensor | None:
        """H15 mean belief sigma over the anchor frame's token grid (all
        cells visible): exp(logvar/2) averaged per sample."""
        if not self.compute_sigma:
            return None
        with self._eng.autocast():
            tokens = self.world.encode_tokens(batch.frames[:, -1])
            vis = torch.ones(tokens.shape[:2], device=tokens.device)
            _, logvar = self.world.imagination(tokens, vis)
        return (0.5 * logvar.float().clamp(-10.0, 10.0)).exp().mean(-1).cpu()

    def describe(self) -> dict:
        d = super().describe()
        d["h15_sigma"] = self.compute_sigma
        return d


# --------------------------------------------------------------------------
# REF-A: online DINO tokenization + adapter states
# --------------------------------------------------------------------------

class DinoV2Tokenizer:
    """Online DINOv2-B/14 via torch.hub (scripts/dino_precompute.py fallback
    path, which is what the accepted REF-A run used): latest 3 channels of
    each frame stack -> resize 224 -> ImageNet-normalize -> 16x16 fp16 token
    grid [*, 256, 768]. Needs network access on first load (cached by
    torch.hub afterwards)."""

    n_tokens, d = 256, 768

    def __init__(self, device: str = "cpu", size: int = 224, batch: int = 32):
        self.device, self.size, self.batch = device, size, batch
        self.model = torch.hub.load("facebookresearch/dinov2",
                                    "dinov2_vitb14").to(device).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406],
                                  device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225],
                                 device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> Tensor:
        """frames [M, C, H, W] float [0,1] -> tokens [M, 256, 768] fp16."""
        if frames.shape[1] < 3:
            raise ValueError(
                f"DINO tokenizer needs RGB frame stacks (C >= 3), got "
                f"C={frames.shape[1]} — REF-A replays camera corpora only")
        x = frames[:, -3:].to(self.device)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        toks = [self.model.get_intermediate_layers(x[i:i + self.batch], n=1)[0]
                for i in range(0, x.shape[0], self.batch)]
        return torch.cat(toks).half()


class ToyTokenizer:
    """Deterministic pixel tokenizer for tests/demos — NOT a DINO substitute.

    Adaptive-avg-pools the frame onto a sqrt(n_tokens) grid (channel-mean)
    and lifts each cell value through a fixed seeded direction vector plus a
    fixed per-token embedding, so tokens stay spatially faithful (token n
    sees exactly patch n — grid adapters remain meaningful) and fully
    reproducible. Emits fp16 like the real tokenizer.
    """

    def __init__(self, n_tokens: int = 256, d: int = 768, seed: int = 0):
        g = int(n_tokens ** 0.5)
        if g * g != n_tokens:
            raise ValueError(f"n_tokens must be square, got {n_tokens}")
        self.grid, self.n_tokens, self.d = g, n_tokens, d
        gen = torch.Generator().manual_seed(seed)
        self._dir = torch.randn(d, generator=gen)
        self._pos = torch.randn(n_tokens, d, generator=gen) * 0.05

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> Tensor:
        """frames [M, C, H, W] float [0,1] -> tokens [M, N, d] fp16."""
        dev = frames.device
        pix = F.adaptive_avg_pool2d(frames, (self.grid, self.grid)
                                    ).mean(1).flatten(1)          # [M, N]
        toks = pix.unsqueeze(-1) * self._dir.to(dev) + self._pos.to(dev)
        return toks.half()


class RefAArm(_LatentArm):
    """REF-A frozen-DINO arm: online tokenization -> RefAModel adapter states
    -> the shared latent decode path (probes, inverse dynamics, imag_rel).

    The checkpoint MUST carry fitted standardizer stats (they travel in every
    refa_train checkpoint); an unfitted standardizer raises. Per-frame token
    grids are LRU-cached (overlapping windows re-tokenize nothing while the
    replay walks an episode); online tokenization is billed to the arm's
    latency because it IS part of this architecture's inference cost.
    """

    name = "refa"

    def __init__(self, ckpt: str | Path, adapter: str = "grid",
                 pred_cfg=None, device: str = "cpu",
                 tokenizer: Callable[[Tensor], Tensor] | None = None,
                 d_dino: int = 768, n_tokens: int = 256,
                 probe_alpha: float = 10.0, compute_imag_rel: bool = True,
                 cache_frames: int = 512, **model_kw):
        from tanitad.refs.refa import RefAModel
        model = RefAModel(pred_cfg=pred_cfg, d_dino=d_dino,
                          adapter_kind=adapter, n_tokens=n_tokens, **model_kw)
        sd, step = load_checkpoint_state(ckpt)
        model.load_state_dict(sd)
        if not bool(model.standardizer.fitted):
            raise RuntimeError(
                f"REF-A checkpoint {ckpt} has UNFITTED standardizer stats — "
                f"replay refuses to guess feature statistics (spec item 1)")
        self.model = model.to(device).eval()
        self.device = device
        self.adapter_kind = adapter
        self._tokenizer = tokenizer          # None -> lazy DINOv2 on first use
        self._cache: OrderedDict[tuple, Tensor] = OrderedDict()
        self._cache_max = int(cache_frames)
        self._init_latent(
            predict=self.model.predict, inv_dyn=self.model.inv_dyn,
            window=self.model.pred_cfg.window,
            horizons=self.model.pred_cfg.horizons,
            ckpt=ckpt, step=step, probe_alpha=probe_alpha,
            compute_imag_rel=compute_imag_rel)

    @property
    def tokenizer(self) -> Callable[[Tensor], Tensor]:
        if self._tokenizer is None:
            self._tokenizer = DinoV2Tokenizer(device=self.device)
        return self._tokenizer

    # -- per-frame token cache -------------------------------------------------
    def _tokens_for(self, keys: list[tuple], frames: list[Tensor]) -> None:
        """Tokenize the frames whose keys are not cached; insert (LRU)."""
        missing = [(k, f) for k, f in zip(keys, frames) if k not in self._cache]
        seen: dict[tuple, Tensor] = {}
        for k, f in missing:
            seen.setdefault(k, f)
        if seen:
            toks = self.tokenizer(torch.stack(list(seen.values())))
            for k, t in zip(seen.keys(), toks):
                self._cache[k] = t
        for k in keys:                                  # LRU refresh + evict
            self._cache.move_to_end(k)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _window_tokens(self, batch: WindowBatch) -> Tensor:
        """Token grids for every window frame: [B, w, N, D] fp16."""
        w = self.window
        keys, frames = [], []
        for j, ref in enumerate(batch.refs):
            for i in range(w):
                keys.append((ref.corpus, ref.episode_id, ref.last - w + 1 + i))
                frames.append(batch.frames[j, -w + i])
        self._tokens_for(keys, frames)
        it = iter(keys)
        return torch.stack([
            torch.stack([self._cache[next(it)] for _ in range(w)])
            for _ in range(len(batch))])

    def _states(self, batch: WindowBatch) -> Tensor:
        return self.model.encode_window(self._window_tokens(batch))

    def _future_state(self, batch: WindowBatch, k: int) -> Tensor:
        keys = [(ref.corpus, ref.episode_id, ref.last + k)
                for ref in batch.refs]
        frames = list(future_frames_at(batch, k))
        self._tokens_for(keys, frames)
        feats = torch.stack([self._cache[key] for key in keys])
        return self.model.encode(feats)

    def describe(self) -> dict:
        d = super().describe()
        d["adapter"] = self.adapter_kind
        d["tokenizer"] = type(self._tokenizer).__name__ \
            if self._tokenizer is not None else "DinoV2Tokenizer(lazy)"
        return d


# --------------------------------------------------------------------------
# REF-B: direct heads, no probes
# --------------------------------------------------------------------------

class RefBArm:
    """REF-B E2E arm (tanitad.refs.refb.RefBModel): tactical waypoints,
    maneuver distribution, operative action + 0.5 s sequence, confidence
    prediction and feature-OOD score — all direct heads, nothing to fit.

    The strategic nav command is DERIVED per window from route-scale future
    heading (scripts/refb_labels.nav_command) — it is the navigator's input
    to the model, not a prediction, so it is computed outside the latency
    timer. Maneuver ground truth uses the pinned pseudo-label derivation
    (refb_labels.window_maneuver_labels).
    """

    name = "refb"
    requires_fit = False

    def __init__(self, ckpt: str | Path, cfg=None, device: str = "cpu"):
        from tanitad.refs.refb import RefBModel, refb_config
        self._labels = _script_module("refb_labels")
        cfg = cfg if cfg is not None else refb_config()
        if tuple(cfg.tactical.waypoint_horizons) != WAYPOINT_STEPS:
            raise ValueError(
                f"REF-B tactical horizons {cfg.tactical.waypoint_horizons} "
                f"!= replay WAYPOINT_STEPS {WAYPOINT_STEPS} — the ADE "
                f"comparison would be apples-to-oranges")
        model = RefBModel(cfg)
        sd, step = load_checkpoint_state(ckpt)
        model.load_state_dict(sd)
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.ckpt, self.step, self.device = str(ckpt), step, device
        self.window = int(cfg.window)
        self.needs_ahead = max(max(WAYPOINT_STEPS),
                               int(self._labels.LABEL_HORIZON),
                               int(cfg.operative.action_seq) - 1)
        self.engine: ReplayEngine | None = None

    @property
    def _eng(self) -> ReplayEngine:
        if self.engine is None:
            raise RuntimeError(
                "arm 'refb' is not bound to an engine — construct a "
                "ReplayEngine with this arm before calling run_batch")
        return self.engine

    def prepare(self, engine: ReplayEngine,
                fit_reps: Sequence[ReplayEpisode]) -> None:
        """Nothing to fit — direct heads (protocol compliance)."""
        self.engine = engine

    @torch.no_grad()
    def run_batch(self, batch: WindowBatch) -> list[ArmOutput]:
        eng = self._eng
        B = len(batch)
        frames = batch.frames[:, -self.window:]
        # Navigator input (given, not predicted): derived route command.
        nav = [self._labels.nav_command(ref.episode.poses, ref.last)[0]
               for ref in batch.refs]
        nav_t = torch.tensor(nav, dtype=torch.long, device=frames.device)

        with LatencyTimer(eng) as lt:
            with eng.autocast():
                out = self.model(frames, nav_t)
                wp = torch.stack([out["waypoints"][k]
                                  for k in WAYPOINT_STEPS], dim=1)
                probs = torch.softmax(out["maneuver_logits"].float(), dim=-1)
                ood = self.model.ood.score(out["states"][:, -1].float())
        per_win_ms = lt.ms / B

        # Diagnostics (ground truth; outside the timer): maneuver labels.
        H = int(self._labels.LABEL_HORIZON)
        pose_last = torch.stack([ref.episode.poses[ref.last]
                                 for ref in batch.refs])
        fut_poses = torch.stack(
            [ref.episode.poses[ref.last + 1: ref.last + 1 + H]
             for ref in batch.refs])
        man_gt = self._labels.window_maneuver_labels(pose_last, fut_poses)

        wp_np = wp.float().cpu().numpy()
        seq_np = out["action_seq"].float().cpu().numpy()
        probs_np = probs.cpu().numpy()
        conf = out["conf_pred"].float().cpu()
        outs = []
        for j in range(B):
            outs.append(ArmOutput(
                latency_ms=per_win_ms,
                waypoints=wp_np[j],
                waypoint_steps=WAYPOINT_STEPS,
                action=seq_np[j, 0],
                action_seq=seq_np[j],
                maneuver_probs=probs_np[j],
                maneuver_gt=int(man_gt[j]),
                nav_cmd=int(nav[j]),
                conf=float(conf[j]),
                ood=float(ood[j]),
            ))
        return outs

    def describe(self) -> dict:
        return {"ckpt": self.ckpt, "ckpt_step": self.step,
                "window": self.window,
                "waypoint_horizons": list(self.cfg.tactical.waypoint_horizons),
                "action_seq": int(self.cfg.operative.action_seq)}
