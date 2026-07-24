"""Multi-domain trainer for OUR rig-robust dynamics-estimation encoder.

Design + rationale: `…/Architecture & Inference/Implementation/incoming/
2026-07-22-own-dynamics-encoder/DESIGN.md`. Model: `tanitad.models.dynamics_encoder`.
Launch config + go/no-go: the sibling `LAUNCH_PLAN.md`.

This file owns the TRAINING SCAFFOLDING:
  * a MULTI-DOMAIN window dataset that mixes rigs/corpora with PER-CLIP camera
    parameters and applies GEOMETRY (extrinsics) domain-randomisation, keeping
    (frames, camera-params) consistent — the rig-robustness apparatus;
  * the combined-objective training loop (delegates to
    ``DynamicsEncoderModel.training_step``);
  * ``--smoke``: a self-contained CPU run on synthetic multi-domain episodes that
    proves the pipeline is finite, differentiable, fits, mixes domains, and stays
    inside the sub-300M envelope. This is what `test_dynamics_encoder.py` asserts.

DO NOT launch the multi-GPU-day run from here yet — that needs the multi-rig
co-train verdict (`ae75b7c`) + Sayed's go (LAUNCH_PLAN §go/no-go). The real
data path (`build_domains_from_caches`) is wired and import-clean but is only
exercised at launch; the smoke path needs no caches, no GPU, no parity contact.

Parity firewall: this is a SIDE model. It never reads the WM parity key
`e438721ae894` / skip-hash `f09e44db` as truth and never re-selects parity
episodes; its splits are by rig / by corpus (orthogonal to the WM selection).

Usage:
  # CPU smoke (no caches, no GPU) — the pipeline proof:
  PYTHONPATH=stack python scripts/train_dynamics_encoder.py --smoke

  # launch (pod1/pod3, NON-training pod, under gpu_lock.sh acquire dyn-encoder):
  PYTHONPATH=/workspace/TanitAD/stack python scripts/train_dynamics_encoder.py \
    --pai-cache ... --pai-rig-table ... --comma-cache ... --l2d-cache ... \
    --warm-start /workspace/tmp/flagship/ckpt.pt --out experiments/dyn-encoder
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
import idm_head as ih  # noqa: E402
import run_idm_proof as R  # noqa: E402  (shared ep loader for the shard loader)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # stack/
from tanitad.models.dynamics_encoder import (  # noqa: E402
    CAM_PARAM_NAMES, DynEncConfig, DynamicsEncoderModel,
    dynamics_encoder_smoke_config, normalize_cam_params)
from tanitad.models.metric_dynamics import relative_ego_pose  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# camera-params: physical vector + known mask per domain                      #
# --------------------------------------------------------------------------- #
def cam_vec(f_eff=266.0, cx=128.0, cy=128.0, pitch=0.0, height=1.4, roll=0.0,
            k1=0.0, is_fisheye=0.0, known=(1, 1, 1, 1, 1, 1, 1, 1)
            ) -> tuple[Tensor, Tensor]:
    """Physical camera vector in the CAM_GROUPS order (intrinsics | extrinsics |
    distortion) + a per-parameter known/unknown mask."""
    raw = torch.tensor([f_eff, cx, cy, pitch, height, roll, k1, is_fisheye],
                       dtype=torch.float32)
    return raw, torch.tensor(known, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# geometry (extrinsics) domain-randomisation — the rig-robustness apparatus.  #
# A vertical image shift of `dv` px ≈ a small camera-pitch rotation; we apply  #
# it to the frames AND reflect it in the camera params so the pair stays       #
# consistent, forcing the encoder to USE the camera input (breaks the implicit #
# one-rig binding that collapsed cross-rig — results_regate.json). The proper  #
# training homography (pitch+height, principal-point re-crop) is in DESIGN §3; #
# this consistent proxy is what the smoke validates.                           #
# --------------------------------------------------------------------------- #
def geom_augment(frames: Tensor, cam_raw: Tensor, max_dv: int, gen: torch.Generator
                 ) -> tuple[Tensor, Tensor]:
    """frames [W,C,H,W'], cam_raw [6] -> (shifted frames, updated cam_raw)."""
    if max_dv <= 0:
        return frames, cam_raw
    dv = int(torch.randint(-max_dv, max_dv + 1, (1,), generator=gen).item())
    if dv == 0:
        return frames, cam_raw
    f = torch.roll(frames, shifts=dv, dims=2)
    if dv > 0:
        f[:, :, :dv] = 0
    else:
        f[:, :, dv:] = 0
    cam = cam_raw.clone()
    cam[2] = cam[2] + dv                                     # cy shifts with content
    cam[3] = cam[3] + math.atan2(float(dv), float(cam[0].clamp_min(1.0)))  # pitch
    return f, cam


# --------------------------------------------------------------------------- #
# multi-domain window dataset                                                 #
# --------------------------------------------------------------------------- #
class MultiDomainWindowDataset:
    """Mixes windows from several domains (rigs / corpora), each carrying its own
    camera params. Sampling is domain-balanced by window count by default so a
    small domain is not swamped. Emits batches ready for ``training_step``."""

    def __init__(self, domains: list[dict], k: int, stride: int = 2,
                 max_dv: int = 0, seed: int = 0):
        self.domains = domains          # each: {name, eps:[{frames,poses,actions}],
        #                                          cam_raw[6], cam_known[6]}
        self.k = k
        self.max_dv = max_dv
        self.gen = torch.Generator().manual_seed(seed)
        # per-domain window index: (d_i, e_i, center_t)
        self.index_by_domain: list[list[tuple[int, int, int]]] = []
        for d_i, dom in enumerate(domains):
            idx = []
            for e_i, ep in enumerate(dom["eps"]):
                T = ep["frames"].shape[0]
                for t in ih.valid_centers(T, k, ih.DEFAULT_HORIZONS, stride).tolist():
                    idx.append((d_i, e_i, t))
            self.index_by_domain.append(idx)
        self.n_windows = sum(len(x) for x in self.index_by_domain)
        if self.n_windows == 0:
            raise RuntimeError("no windows built — episodes too short for the window")

    def domain_window_counts(self) -> dict[str, int]:
        return {self.domains[i]["name"]: len(self.index_by_domain[i])
                for i in range(len(self.domains))}

    def _one(self, d_i: int, e_i: int, t: int) -> dict:
        dom = self.domains[d_i]
        ep = dom["eps"][e_i]
        k = self.k
        frames = ep["frames"][t - k:t + k + 1]                  # [W,C,H,W']
        # per-CLIP camera params when present (real run: per-clip cy), else the
        # per-DOMAIN default (smoke). geom_augment perturbs the raw before norm.
        cam_raw0 = ep.get("cam_raw", dom["cam_raw"])
        cam_known = ep.get("cam_known", dom["cam_known"])
        frames, cam_raw = geom_augment(frames.clone(), cam_raw0,
                                       self.max_dv, self.gen)
        cam12 = normalize_cam_params(cam_raw, cam_known)
        poses, actions = ep["poses"], ep["actions"]
        tt = torch.tensor([t])
        return {
            "frames": frames,
            "cam": cam12,
            "actions": actions[t - k:t + k + 1].float(),
            "poses": poses[t - k:t + k + 1].float(),
            "scal_tgt": ih.scalar_targets_at(poses, actions, tt)[0],
            "traj_tgt": ih.traj_targets_at(poses, tt)[0],
            "step_tgt": relative_ego_pose(poses[t].float(), poses[t + 1].float()),
            "domain": d_i,
        }

    def sample_batch(self, batch: int) -> dict:
        """Domain-balanced batch: round-robin over domains, uniform within."""
        picks: list[tuple[int, int, int]] = []
        d_order = torch.randperm(len(self.domains), generator=self.gen).tolist()
        di = 0
        while len(picks) < batch:
            d_i = d_order[di % len(d_order)]
            idx = self.index_by_domain[d_i]
            if idx:
                j = int(torch.randint(len(idx), (1,), generator=self.gen).item())
                picks.append(idx[j])
            di += 1
        items = [self._one(*p) for p in picks]
        out = {
            "frames": torch.stack([x["frames"] for x in items]),
            "cam": torch.stack([x["cam"] for x in items]),
            "actions": torch.stack([x["actions"] for x in items]),
            "poses": torch.stack([x["poses"] for x in items]),
            "scal_tgt": torch.stack([x["scal_tgt"] for x in items]),
            "traj_tgt": torch.stack([x["traj_tgt"] for x in items]),
            "step_tgt": torch.stack([x["step_tgt"] for x in items]),
            "domain": torch.tensor([x["domain"] for x in items]),
        }
        return out


# --------------------------------------------------------------------------- #
# real data path (launch-time; import-clean, not exercised by the smoke)      #
# --------------------------------------------------------------------------- #
def build_domains_from_caches(args) -> list[dict]:
    """Assemble the launch multi-domain mix from the SAME ep caches the IDM proof
    used (via run_idm_proof._load_ep). PhysicalAI split into rig-A / rig-B by the
    rig table; comma2k19 and L2D added as further domains. Camera params per domain
    from calib (PhysicalAI/comma) or ESTIMATED+flagged (L2D ships no intrinsics)."""
    import run_idm_proof as R                                     # noqa: E402
    doms: list[dict] = []

    def load(paths: list[str]) -> list[dict]:
        eps = []
        for p in paths:
            d = R._load_ep(p)
            eps.append({"frames": d["frames_u8"], "poses": d["poses"].float(),
                        "actions": d["actions"].float()})
        return eps

    if args.pai_cache and args.pai_rig_table:
        rig = json.loads(Path(args.pai_rig_table).read_text())
        a, b = R.select_episodes(rig, args.pai_cache, args.cap, args.cap)
        # PhysicalAI is f-theta FISHEYE (is_fisheye=1); rig-A/rig-B differ in
        # extrinsics (cy 543 vs 755 -> camera pitch/height) — the measured collapse.
        raw_a, kn = cam_vec(pitch=+0.02, height=1.30, is_fisheye=1.0)
        raw_b, _ = cam_vec(pitch=-0.03, height=1.60, is_fisheye=1.0)
        doms.append({"name": "pai_rigA", "eps": load([p for _t, p in a]),
                     "cam_raw": raw_a, "cam_known": kn})
        doms.append({"name": "pai_rigB", "eps": load([p for _t, p in b]),
                     "cam_raw": raw_b, "cam_known": kn})
    if args.comma_cache:
        paths = [str(p) for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))]
        raw, kn = cam_vec(pitch=+0.01, height=1.20, is_fisheye=0.0)  # rectilinear
        doms.append({"name": "comma2k19", "eps": load(paths[:args.cap]),
                     "cam_raw": raw, "cam_known": kn})
    if args.l2d_cache:
        paths = [str(p) for p in sorted(Path(args.l2d_cache).glob("ep_*.pt"))]
        # L2D ships NO intrinsics -> f_eff/cx/cy + distortion UNKNOWN (mask 0);
        # extrinsics_RDF gives pitch/height. "unknown intrinsics" is a first-class
        # input pattern (the GAIA-2 mask lets the encoder fall back gracefully).
        raw, _ = cam_vec(pitch=0.0, height=1.50)
        kn = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0], dtype=torch.float32)
        doms.append({"name": "l2d", "eps": load(paths[:args.cap]),
                     "cam_raw": raw, "cam_known": kn})
    if not doms:
        raise SystemExit("no domains — pass at least one cache (or use --smoke)")
    return doms


def maybe_warm_start(model: DynamicsEncoderModel, ckpt_path: str) -> dict:
    """PASS-branch warm-start: load flagship-v1 encoder+readout into the
    camera-conditioned encoder (cam_film is zero-init => identical forward at
    start). Returns a report; strict on the two submodules, tolerant of the rest."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    ro_sd = {k[len("readout."):]: v for k, v in sd.items() if k.startswith("readout.")}
    me = model.encoder.enc.load_state_dict(enc_sd, strict=True)
    mr = model.encoder.readout.load_state_dict(ro_sd, strict=True)
    return {"enc_keys": len(enc_sd), "ro_keys": len(ro_sd),
            "enc_missing": list(me.missing_keys), "ro_missing": list(mr.missing_keys)}


# --------------------------------------------------------------------------- #
# MEMORY-SAFE real-run data: per-clip camera specs + a buffered SHARD loader   #
# (pod cgroup ~46 GB, clips 117 MB) — never all frames resident.               #
# --------------------------------------------------------------------------- #
def clip_cam_raw(cy: float, is_fisheye: float, pitch: float = 0.0,
                 height: float = 1.4) -> tuple[Tensor, Tensor]:
    """Per-clip physical camera vector: real principal-point cy (the rig signal:
    rig-A~542 / rig-B~753) + the fisheye flag (PhysicalAI f-theta vs comma
    rectilinear). All-known mask."""
    raw = torch.tensor([266.0, 128.0, float(cy), pitch, height, 0.0, 0.0,
                        float(is_fisheye)], dtype=torch.float32)
    return raw, torch.ones(8, dtype=torch.float32)


def build_clip_specs(args, rig_table: dict) -> list[dict]:
    """PER-CLIP specs {domain, path, cam_raw, cam_known} — NO frames loaded. The
    multi-rig corpus: PhysicalAI rig-A + rig-B (per-clip cy, fisheye) + comma2k19
    (rectilinear). Parity firewall: rig/corpus splits only, never the parity key."""
    specs: list[dict] = []
    for idx in sorted(int(i) for i in rig_table):
        p = Path(args.pai_cache) / f"ep_{idx:05d}.pt"
        if not p.exists():
            continue
        e = rig_table[str(idx)]
        cy = e.get("cy") or (542.0 if e["rig"] == "a" else 753.0)
        pitch = 0.02 if e["rig"] == "a" else -0.03      # coarse per-rig extrinsic
        raw, kn = clip_cam_raw(cy, 1.0, pitch=pitch,
                               height=1.30 if e["rig"] == "a" else 1.60)
        specs.append({"domain": f"pai_{e['rig']}", "path": str(p),
                      "cam_raw": raw, "cam_known": kn})
    if args.comma_cache:
        raw, kn = clip_cam_raw(128.0, 0.0, pitch=0.01, height=1.20)
        for p in sorted(Path(args.comma_cache).glob("ep_*.pt")):
            specs.append({"domain": "comma", "path": str(p),
                          "cam_raw": raw.clone(), "cam_known": kn.clone()})
    return specs


class ShardLoader:
    """Holds ~`n_resident` clips resident (balanced across domains), samples windows
    from them for `steps_per_shard` steps, then rotates in a fresh shard. Bounds RAM
    at n_resident x 117 MB. Emits the exact batch dict `training_step` expects, by
    delegating to a transient `MultiDomainWindowDataset` over the resident shard."""

    def __init__(self, specs: list[dict], k: int, stride: int, n_resident: int,
                 steps_per_shard: int, max_dv: int, seed: int):
        self.specs = specs
        self.k, self.stride, self.n_res = k, stride, n_resident
        self.sps, self.max_dv = steps_per_shard, max_dv
        self.by_dom: dict[str, list[dict]] = {}
        for s in specs:
            self.by_dom.setdefault(s["domain"], []).append(s)
        self.gen = torch.Generator().manual_seed(seed)
        self._step = 0
        self._reload()

    def _reload(self) -> None:
        # round-robin across domains so every shard is multi-rig
        doms = sorted(self.by_dom)
        picks: list[dict] = []
        di = 0
        while len(picks) < min(self.n_res, len(self.specs)):
            pool = self.by_dom[doms[di % len(doms)]]
            picks.append(pool[int(torch.randint(len(pool), (1,),
                                                 generator=self.gen).item())])
            di += 1
        # group picks into MultiDomainWindowDataset "domains" (per-clip cam on eps)
        grouped: dict[str, list[dict]] = {}
        for s in picks:
            d = R._load_ep(s["path"])
            grouped.setdefault(s["domain"], []).append(
                {"frames": d["frames_u8"], "poses": d["poses"].float(),
                 "actions": d["actions"].float(),
                 "cam_raw": s["cam_raw"], "cam_known": s["cam_known"]})
        domains = [{"name": name, "eps": eps,
                    "cam_raw": eps[0]["cam_raw"], "cam_known": eps[0]["cam_known"]}
                   for name, eps in grouped.items()]
        self.ds = MultiDomainWindowDataset(domains, k=self.k, stride=self.stride,
                                           max_dv=self.max_dv,
                                           seed=int(self.gen.initial_seed()) + self._step)
        self.resident_domains = {name: len(eps) for name, eps in grouped.items()}

    def sample_batch(self, batch: int) -> dict:
        if self._step > 0 and self._step % self.sps == 0:
            self._reload()
        self._step += 1
        return self.ds.sample_batch(batch)


# --------------------------------------------------------------------------- #
# atomic checkpoint / resume + md5                                            #
# --------------------------------------------------------------------------- #
def md5_file(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_ckpt(model, opt, step: int, cfg: DynEncConfig, path: str) -> str:
    """Atomic: write .tmp then rename, so a crash mid-save cannot corrupt the ckpt
    the supervisor resumes from."""
    import os
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": cfg.__dict__}, tmp)
    os.replace(tmp, path)
    return md5_file(path)


def load_ckpt(model, opt, path: str, device: str) -> int:
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    return int(ck.get("step", 0))


# --------------------------------------------------------------------------- #
# training loop                                                               #
# --------------------------------------------------------------------------- #
def train(model: DynamicsEncoderModel, ds, *, steps: int, batch: int, lr: float,
          wd: float, device: str, warmup: int = 50, log_every: int = 50,
          out_dir: str | None = None, resume: bool = False, ckpt_every: int = 1000,
          milestone_step: int | None = None) -> list[dict]:
    """Loader-agnostic (`ds` just needs `.sample_batch`). When ``out_dir`` is set:
    atomic checkpoint every ``ckpt_every`` steps to ``out_dir/ckpt.pt`` (durable,
    the supervisor resumes from it), a one-time milestone copy at ``milestone_step``,
    and resume from the latest ckpt when ``resume``. ``out_dir=None`` => the smoke
    path (no I/O), byte-compatible with the CPU test."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    start = 0
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        latest = str(Path(out_dir) / "ckpt.pt")
        if resume and Path(latest).exists():
            start = load_ckpt(model, opt, latest, device)
            log(f"RESUMED from {latest} at step {start} (md5 {md5_file(latest)})")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    for _ in range(start):
        sched.step()                                    # fast-forward schedule
    history = []
    for step in range(start, steps):
        b = ds.sample_batch(batch)
        b = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in b.items()}
        loss, logs = model.training_step(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        logs["step"] = step
        history.append(logs)
        if step % log_every == 0 or step == steps - 1:
            extra = (f" resident={ds.resident_domains}"
                     if hasattr(ds, "resident_domains") else "")
            log(f"step {step}/{steps} " + " ".join(
                f"{k}={logs[k]:.4f}" for k in
                ("total", "idm", "fwd", "mask", "ground", "sigreg")) + extra)
        if out_dir is not None and (step + 1) % ckpt_every == 0:
            m = save_ckpt(model, opt, step + 1, model.cfg, latest)
            log(f"CKPT step {step + 1} -> {latest} md5 {m}")
        if out_dir is not None and milestone_step is not None \
                and step + 1 == milestone_step:
            import shutil
            mpath = str(Path(out_dir) / f"milestone_step{step + 1}.pt")
            save_ckpt(model, opt, step + 1, model.cfg, latest)
            shutil.copyfile(latest, mpath)
            log(f"MILESTONE step {step + 1} -> {mpath} md5 {md5_file(mpath)}")
    return history


# --------------------------------------------------------------------------- #
# synthetic multi-domain episodes for the smoke                               #
# --------------------------------------------------------------------------- #
def _synthetic_domain(name: str, n_eps: int, T: int, cfg: DynEncConfig,
                      cam_raw: Tensor, cam_known: Tensor, seed0: int) -> dict:
    """Frames whose CONTENT encodes the kinematics (so the supervised IDM +
    grounding can actually fit and the loss must fall), plus distinct camera
    params per domain. Bounded hidden state -> shared distribution across eps."""
    C, S = cfg.in_channels, cfg.image_size
    eps = []
    for s in range(n_eps):
        z, poses, actions = ih._synthetic_episode(T, 8, seed0 + s)
        v = poses[:, 3]                                          # speed
        yaw = poses[:, 2]
        # paint the signal into the frame: channel-0 brightness ~ speed, a
        # horizontal band whose row ~ yaw. Deterministic, cheap, learnable.
        frames = torch.zeros(T, C, S, S)
        frames[:, 0] = ((v - 4.0) / 8.0).clamp(0, 1)[:, None, None]
        row = ((yaw * 6.0 + 0.5) * S).long().clamp(0, S - 1)
        for t in range(T):
            frames[t, 1, row[t]] = 1.0
        frames = (frames + 0.02 * torch.randn(T, C, S, S)).clamp(0, 1)
        eps.append({"frames": (frames * 255).to(torch.uint8),
                    "poses": poses, "actions": actions})
    return {"name": name, "eps": eps, "cam_raw": cam_raw, "cam_known": cam_known}


def build_smoke_domains(cfg: DynEncConfig) -> list[dict]:
    """Three domains with DISTINCT camera geometry + one with unknown intrinsics —
    proves per-domain conditioning, domain mixing, and the unknown-rig fallback."""
    a_raw, kn = cam_vec(pitch=+0.02, height=1.30, is_fisheye=1.0)   # rig-A-like
    b_raw, _ = cam_vec(pitch=-0.03, height=1.60, is_fisheye=1.0)    # rig-B-like
    c_raw, _ = cam_vec(f_eff=240.0, pitch=+0.01, height=1.20)       # comma-like
    l_known = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0], dtype=torch.float32)  # L2D
    l_raw, _ = cam_vec(pitch=0.0, height=1.50)
    return [
        _synthetic_domain("rigA", 6, 40, cfg, a_raw, kn, 0),
        _synthetic_domain("rigB", 6, 40, cfg, b_raw, kn, 100),
        _synthetic_domain("comma", 6, 40, cfg, c_raw, kn, 200),
        _synthetic_domain("l2d_unk", 4, 40, cfg, l_raw, l_known, 300),
    ]


def smoke() -> dict:
    """Self-contained CPU pipeline proof. Returns a report dict (asserts inside)."""
    torch.manual_seed(0)
    cfg = dynamics_encoder_smoke_config()
    model = DynamicsEncoderModel(cfg)
    dep, tot = model.deployable_params(), model.total_params()
    log(f"params: deployable(enc+readout+idm)={dep/1e6:.3f}M  "
        f"total(+aux)={tot/1e6:.3f}M  state_dim={model.state_dim}")

    domains = build_smoke_domains(cfg)
    ds = MultiDomainWindowDataset(domains, k=cfg.window // 2, stride=2, max_dv=3,
                                  seed=0)
    counts = ds.domain_window_counts()
    log(f"multi-domain windows: {counts} (total {ds.n_windows})")
    assert len([c for c in counts.values() if c > 0]) >= 2, "need >=2 live domains"

    # one batch: shapes + domain mixing + finite/differentiable
    b = ds.sample_batch(12)
    assert b["frames"].shape == (12, cfg.window, cfg.in_channels,
                                 cfg.image_size, cfg.image_size)
    assert b["cam"].shape == (12, 2 * len(CAM_PARAM_NAMES))
    n_dom_in_batch = int(b["domain"].unique().numel())
    log(f"batch mixes {n_dom_in_batch} domains; cam[0]={b['cam'][0].tolist()}")
    assert n_dom_in_batch >= 2, "batch did not mix domains"

    loss, logs = model.training_step(b)
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    loss.backward()
    gnorm = sum(float(p.grad.norm()) for p in model.parameters()
                if p.grad is not None)
    assert math.isfinite(gnorm) and gnorm > 0, "no/NaN gradient"
    # every sub-objective finite
    for key in ("idm", "fwd", "mask", "ground", "sigreg"):
        assert math.isfinite(logs[key]), f"non-finite {key}"
    log("forward/backward OK; sub-losses " +
        " ".join(f"{k}={logs[k]:.4f}" for k in ("idm", "fwd", "mask", "ground",
                                                "sigreg")))

    # camera conditioning is LIVE (different rigs -> different latent) once FiLM
    # is non-identity. Build a bite-enabled encoder and check it responds.
    import dataclasses
    live = DynamicsEncoderModel(dataclasses.replace(cfg, cam_inject_zero_init=False))
    with torch.no_grad():
        z_a = live.encoder.encode_window(b["frames"][:2], b["cam"][:2])
        cam_other = b["cam"][:2].clone()
        cam_other[:, 3] += 1.0                              # perturb pitch feature
        z_b = live.encoder.encode_window(b["frames"][:2], cam_other)
        cam_delta = float((z_a - z_b).abs().mean())
    log(f"camera-conditioning response (mean|dz| on pitch change) = {cam_delta:.4e}")
    assert cam_delta > 1e-6, "camera params do not affect the latent (FiLM dead)"

    # short fit: the combined loss must fall (driven by the supervised terms,
    # which have real targets and cannot collapse)
    ds2 = MultiDomainWindowDataset(domains, k=cfg.window // 2, stride=2, max_dv=3,
                                   seed=1)
    hist = train(model, ds2, steps=40, batch=12, lr=3e-3, wd=0.01, device="cpu",
                 warmup=10, log_every=10)
    first = sum(h["total"] for h in hist[:5]) / 5
    last = sum(h["total"] for h in hist[-5:]) / 5
    first_idm = sum(h["idm"] for h in hist[:5]) / 5
    last_idm = sum(h["idm"] for h in hist[-5:]) / 5
    log(f"fit: total {first:.4f} -> {last:.4f} ; idm {first_idm:.4f} -> {last_idm:.4f}")
    assert last < first, f"combined loss did not fall ({first:.4f} -> {last:.4f})"
    assert last_idm < first_idm, "supervised IDM loss did not fall"
    assert dep < 300e6, f"deployable {dep/1e6:.1f}M exceeds the sub-300M envelope"

    report = {
        "deployable_params_M": round(dep / 1e6, 3),
        "total_params_M": round(tot / 1e6, 3),
        "state_dim": model.state_dim,
        "domain_windows": counts,
        "batch_domains": n_dom_in_batch,
        "grad_norm": round(gnorm, 3),
        "cam_conditioning_response": cam_delta,
        "sub_losses_first_batch": {k: round(logs[k], 4)
                                   for k in ("idm", "fwd", "mask", "ground", "sigreg")},
        "fit_total": [round(first, 4), round(last, 4)],
        "fit_idm": [round(first_idm, 4), round(last_idm, 4)],
        "PASS": True,
    }
    log("SMOKE PASS " + json.dumps(report))
    return report


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run the self-contained CPU pipeline proof and exit")
    ap.add_argument("--smoke-out", default=None, help="write the smoke report JSON")
    # launch args (not used by --smoke)
    ap.add_argument("--pai-cache", default=None)
    ap.add_argument("--pai-rig-table", default=None)
    ap.add_argument("--comma-cache", default=None)
    ap.add_argument("--l2d-cache", default=None)
    ap.add_argument("--warm-start", default=None,
                    help="flagship-v1 ckpt to warm-start encoder+readout (PASS branch)")
    ap.add_argument("--cap", type=int, default=400, help="episodes/domain cap")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--max-dv", type=int, default=12, help="geom-aug vertical jitter px")
    ap.add_argument("--out", default="experiments/dyn-encoder")
    ap.add_argument("--seed", type=int, default=0)
    # Branch B (memory-safe multi-day run) controls
    ap.add_argument("--n-resident", type=int, default=48,
                    help="clips held in RAM per shard (46 GB cgroup: 48 ~ 5.6 GB)")
    ap.add_argument("--steps-per-shard", type=int, default=200,
                    help="training steps before rotating the resident shard")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--milestone", type=int, default=2000,
                    help="step at which to save the first milestone ckpt")
    ap.add_argument("--resume", action="store_true",
                    help="resume from out/ckpt.pt if present (supervisor auto-restart)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient-checkpoint the encoder (GPU-memory lever)")
    args = ap.parse_args()

    if args.smoke:
        rep = smoke()
        if args.smoke_out:
            Path(args.smoke_out).write_text(json.dumps(rep, indent=2))
            log(f"WROTE {args.smoke_out}")
        return

    # ---- Branch B: from-scratch camera-conditioned video-SSL encoder ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cfg = DynEncConfig(grad_checkpoint=args.grad_checkpoint)   # all-block cond, sub-300M
    model = DynamicsEncoderModel(cfg)
    log(f"model: deployable {model.deployable_params()/1e6:.1f}M / total "
        f"{model.total_params()/1e6:.1f}M / state_dim {model.state_dim} / "
        f"grad_checkpoint {cfg.grad_checkpoint}")
    if args.warm_start:                                       # Branch A only (default off)
        log(f"warm-start from flagship-v1: {maybe_warm_start(model, args.warm_start)}")
    else:
        log("FROM SCRATCH (Branch B — conditioning learned jointly with the encoder)")

    rig_table = json.loads(Path(args.pai_rig_table).read_text())
    specs = build_clip_specs(args, rig_table)
    from collections import Counter
    log(f"corpus: {len(specs)} clips {dict(Counter(s['domain'] for s in specs))} "
        f"(SIDE model — rig/corpus splits only, never the parity key)")
    loader = ShardLoader(specs, k=cfg.window // 2, stride=2,
                         n_resident=args.n_resident,
                         steps_per_shard=args.steps_per_shard, max_dv=args.max_dv,
                         seed=args.seed)
    log(f"shard loader: {args.n_resident} clips resident, rotate every "
        f"{args.steps_per_shard} steps; first shard {loader.resident_domains}")

    # fixed scalar standardiser from a corpus sample (stable IDM loss; resume
    # restores it from the ckpt buffer so this only matters on a fresh start)
    samp = torch.cat([loader.sample_batch(args.batch)["scal_tgt"] for _ in range(20)])
    model.set_standardizer(samp.mean(0), samp.std(0))
    log(f"standardizer (speed,yaw,steer,accel): mean={[round(x,3) for x in model.std_mean.tolist()]} "
        f"std={[round(x,3) for x in model.std_std.tolist()]}")

    hist = train(model, loader, steps=args.steps, batch=args.batch, lr=args.lr,
                 wd=args.wd, device=device, warmup=1000, log_every=25,
                 out_dir=args.out, resume=args.resume, ckpt_every=args.ckpt_every,
                 milestone_step=args.milestone)
    final = str(Path(args.out) / "ckpt.pt")
    save_ckpt(model, torch.optim.AdamW(model.parameters()), args.steps, cfg, final)
    Path(args.out, "history.json").write_text(json.dumps(hist))
    log(f"DONE -> {args.out} (final md5 {md5_file(final)})")


if __name__ == "__main__":
    main()
