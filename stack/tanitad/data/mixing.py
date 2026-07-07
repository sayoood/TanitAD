"""Real+sim co-training mix (D-010).

Role separation (see DECISIONS.md D-010):
- REAL data (comma2k19, PhysicalAI-AV) owns representation learning and all
  public open-loop numbers (D1-D3 report on real held-out routes).
- SIM data (MetaDrive front-camera RGB, same episode contract) owns what logs
  can never provide: off-expert action-consequence coverage (perturbed/
  exploration policies), scripted occluders (H15/D9 object-level), blocked
  routes (D5/D6), collisions/near-misses, and closed-loop evaluation.

MixedWindowDataset interleaves any window datasets with identical item
contracts at a fixed ratio, deterministically (seeded), and tags every item
with its source domain (0 = first source = real by convention). The domain
tag lets the encoder receive a domain embedding later and lets eval slice
metrics per domain. The real-vs-mixed comparison is a MANDATORY bake-off
(one lever per run): sim share earns its place only if real-data gates do
not regress.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tanitad.data.toy_driving import ToyEpisode


def save_episode(ep: ToyEpisode, path: str) -> None:
    """Persist an episode — frames stored uint8 to keep files small
    (accepts uint8 [0,255] or float [0,1] frames)."""
    if ep.frames.dtype == torch.uint8:
        u8 = ep.frames
    else:
        u8 = (ep.frames.clamp(0, 1) * 255).to(torch.uint8)
    torch.save({
        "frames_u8": u8,
        "actions": ep.actions,
        "poses": ep.poses,
        "episode_id": ep.episode_id,
    }, path)


def load_episode(path: str, mmap: bool = False) -> ToyEpisode:
    """Loads with uint8 frames (memory layout); window datasets convert.

    mmap=True (F-7): tensors stay disk-backed — the kernel pages frames in on
    access and reclaims under pressure, so a 62 GB container can train on a
    500-episode corpus that would need ~135 GB resident. Window __getitem__
    copies only its slice."""
    d = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    return ToyEpisode(frames=d["frames_u8"],
                      actions=d["actions"], poses=d["poses"],
                      episode_id=int(d["episode_id"]))


class MixedWindowDataset(torch.utils.data.Dataset):
    """Deterministic ratio-mix of window datasets with identical contracts.

    sources: list of (dataset, weight). Weights are normalized; an epoch has
    `length` items (default: sum of source lengths). Item = source item plus
    `domain` (source index). Fails fast if the tensor contracts differ.
    """

    def __init__(self, sources: list[tuple[torch.utils.data.Dataset, float]],
                 length: int | None = None, seed: int = 0):
        assert sources and all(len(ds) > 0 for ds, _ in sources), \
            "every mix source must be non-empty"
        self.sources = [ds for ds, _ in sources]
        w = torch.tensor([max(0.0, float(wt)) for _, wt in sources])
        assert w.sum() > 0, "at least one positive weight required"
        self.weights = w / w.sum()
        self.length = length or sum(len(ds) for ds, _ in sources)
        g = torch.Generator().manual_seed(seed)
        self._src = torch.multinomial(self.weights, self.length,
                                      replacement=True, generator=g)
        self._item = torch.stack([
            torch.randint(len(self.sources[int(s)]), (1,), generator=g)[0]
            for s in self._src])
        self._check_contract()

    def _check_contract(self) -> None:
        ref = self.sources[0][0]
        for i, ds in enumerate(self.sources[1:], start=1):
            item = ds[0]
            for key in ("frames", "actions", "future_frames"):
                assert item[key].shape[1:] == ref[key].shape[1:], (
                    f"mix source {i} contract mismatch on '{key}': "
                    f"{tuple(item[key].shape[1:])} vs {tuple(ref[key].shape[1:])}"
                    " — sim episodes must be rendered at the SAME size/channels"
                    " as the real data (front-camera RGB 2-frame stacks)")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        s = int(self._src[i])
        item = dict(self.sources[s][int(self._item[i]) % len(self.sources[s])])
        item["domain"] = s
        return item

    def mix_report(self) -> dict:
        counts = torch.bincount(self._src, minlength=len(self.sources))
        return {f"domain_{i}_frac": float(c) / self.length
                for i, c in enumerate(counts)}
