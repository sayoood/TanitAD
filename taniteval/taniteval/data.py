"""TanitEval — data layer.

Loads raw-frame validation episodes (canonical epcache format) and prepares the
per-architecture model inputs:
  * flagship / refb : raw frame stacks  [T, 9, 256, 256] uint8
  * refa-*          : frozen-encoder token features [T, 256, d] fp16, computed
                      on the fly with the EXACT extraction the training features
                      used (dinov2: torch.hub get_intermediate_layers @224;
                      ijepa: transformers last_hidden_state @224) and cached to
                      disk so repeat runs are free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/TanitAD/stack")
from tanitad.data.mixing import load_episode  # noqa: E402

FEATCACHE = Path("/root/featcache")


def list_val_episodes(val_dir: str, n: int | None = None):
    files = sorted(Path(val_dir).glob("ep_*.pt"))
    return files[:n] if n else files


def load_raw(files):
    """Episodes with raw frames (flagship / refb path)."""
    return [load_episode(str(f), mmap=True) for f in files]


class FeatEp:
    """Episode view exposing frozen-encoder features as .feats (gate EpWrap style)."""

    def __init__(self, feats, ep, eid):
        self.feats = feats                       # [T, 256, d] fp16 (cpu)
        self.actions = ep.actions.float()
        self.poses = ep.poses.float()
        self.episode_id = eid


def _imagenet(latest, size):
    if size != latest.shape[-1]:
        latest = F.interpolate(latest, size=(size, size), mode="bilinear",
                               align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (latest - mean) / std


class FrozenEncoder:
    """Lazy frozen encoder matching the training feature extraction exactly."""

    def __init__(self, kind: str, device="cuda"):
        assert kind in ("dinov2", "ijepa")
        self.kind, self.device, self._m = kind, device, None

    def _model(self):
        if self._m is None:
            if self.kind == "dinov2":
                m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            else:
                from transformers import IJepaModel
                m = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")
            self._m = m.to(self.device).eval()
        return self._m

    @torch.no_grad()
    def encode_episode(self, ep, batch=16):
        latest = ep.frames[:, -3:].float().div(255.0)      # current RGB [T,3,S,S]
        latest = _imagenet(latest, 224)
        m, toks = self._model(), []
        for i in range(0, latest.shape[0], batch):
            x = latest[i:i + batch].to(self.device)
            if self.kind == "dinov2":
                out = m.get_intermediate_layers(x, n=1)[0]          # [B,256,768]
            else:
                out = m(pixel_values=x).last_hidden_state           # [B,256,1280]
            toks.append(out.half().cpu())
        return torch.cat(toks)

    def free(self):
        self._m = None
        torch.cuda.empty_cache()


def load_features(files, kind: str, device="cuda", verbose=True):
    """FeatEps for the given val files, via the disk cache when possible."""
    cache = FEATCACHE / kind
    enc, out = FrozenEncoder(kind, device), []
    for i, f in enumerate(files):
        # Namespace the cache by the corpus dir: ep_*.pt filenames COLLIDE
        # across corpora (physicalai/comma/cosmos all start at ep_00000.pt),
        # so a bare-filename key would serve physicalai features for comma/
        # cosmos frames. Keying on the parent (val-root) dir disambiguates.
        cf = cache / f.parent.name / f.name
        cf.parent.mkdir(parents=True, exist_ok=True)
        ep = load_episode(str(f), mmap=True)
        if cf.exists():
            feats = torch.load(cf, map_location="cpu", mmap=True,
                               weights_only=True)
        else:
            feats = enc.encode_episode(ep)
            torch.save(feats, cf)
            if verbose and i % 10 == 0:
                print(f"[feat:{kind}] encoded {i}/{len(files)}", flush=True)
        out.append(FeatEp(feats, ep, i))
    enc.free()
    return out


class RawEp:
    """Raw-frame episode view for frame-input models (.feats = frames)."""

    def __init__(self, ep, eid):
        self.feats = ep.frames                   # [T, 9, S, S] uint8 (mmap)
        self.actions = ep.actions.float()
        self.poses = ep.poses.float()
        self.episode_id = eid


def load_frames(files):
    return [RawEp(load_episode(str(f), mmap=True), i)
            for i, f in enumerate(files)]
