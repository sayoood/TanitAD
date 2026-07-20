"""Stage 2 (dedup pass) — TWO independent dedup passes (TanitDataSet rev-3 §7.2.5).

Pass 1 — perceptual-hash near-dup, WITHIN source, cheap first: an aHash per clip
keyframe → LSH candidate banding → union-find collapse of near-identical clips
(re-uploads, overlapping comma segments). One clip is the ``is_exemplar``; the
rest carry its ``dedup_cluster_id``.

Pass 2 — GPS/time overlap, CROSS source: a ``geo_cell × time_bucket`` index
collapses TRUE content-dups (same drive, same moment, re-hosted — the comma2k19
GitHub-vs-HF case pHash-within-source misses). The load-bearing rule: same road at
a DIFFERENT time is a wanted multi-traversal (Open-MARS-style world-model
consistency signal) — it is KEPT and tagged ``multi_traversal``, never collapsed.

Pure functions over tensors + plain metadata (no cv2, no I/O). aHash is a fixed
8×8 average-hash, popcount via Python ``int.bit_count`` — CPU, dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.lake.filtering import _latest_luma

# =========================================================================== #
# Perceptual hash (aHash) — 64-bit                                             #
# =========================================================================== #
AHASH_SIDE = 8                # 8x8 = 64-bit hash
NEAR_DUP_HAMMING = 6          # <= this many differing bits = a near-duplicate


def ahash_luma(luma2d: Tensor) -> int:
    """8×8 average-hash of a single luma image ``[H,W]`` (or ``[1,H,W]``) → 64-bit
    int: adaptive-avg-pool to 8×8, then bit = (cell >= grid mean)."""
    x = luma2d.float()
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    small = F.adaptive_avg_pool2d(x, (AHASH_SIDE, AHASH_SIDE)).view(-1)
    bits = (small >= small.mean()).to(torch.int64)
    h = 0
    for b in bits.tolist():
        h = (h << 1) | int(b)
    return h


def clip_phash(frames: Tensor) -> int:
    """A clip's perceptual hash: aHash of the latest RGB frame at the clip's middle
    timestep (a stable, motion-representative keyframe)."""
    luma = _latest_luma(frames)                        # [T,1,H,W]
    mid = luma[luma.shape[0] // 2, 0]                  # [H,W]
    return ahash_luma(mid)


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes (popcount of XOR)."""
    return int(a ^ b).bit_count()


# =========================================================================== #
# union-find (shared by both passes)                                           #
# =========================================================================== #
class _UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)          # deterministic: min id is root


def _cluster_id(root) -> str:
    return f"c{root}"


# =========================================================================== #
# Pass 1 — pHash near-dup WITHIN source (LSH banding + union-find)              #
# =========================================================================== #
LSH_BANDS = 8                 # split the 64-bit hash into 8 bands of 8 bits


def _lsh_bands(h: int, n_bands: int = LSH_BANDS) -> list[tuple[int, int]]:
    """(band_index, band_value) list — items sharing any band are dup CANDIDATES,
    so the pairwise Hamming check runs only within candidate buckets (not O(n²))."""
    w = 64 // n_bands
    mask = (1 << w) - 1
    return [(bi, (h >> (bi * w)) & mask) for bi in range(n_bands)]


def dedup_within_source(items: list[dict], radius: int = NEAR_DUP_HAMMING
                        ) -> dict:
    """items = ``[{id, source, phash}]`` → ``{id: {dedup_cluster_id, is_exemplar,
    phash}}``. Near-dups collapse ONLY within the same source. Lowest id in a
    cluster is the exemplar (deterministic)."""
    by_source: dict[str, list[dict]] = {}
    for it in items:
        by_source.setdefault(it["source"], []).append(it)

    out: dict = {}
    for source, its in by_source.items():
        ids = [it["id"] for it in its]
        uf = _UF(ids)
        buckets: dict[tuple[int, int], list[dict]] = {}
        for it in its:
            for key in _lsh_bands(int(it["phash"])):
                buckets.setdefault(key, []).append(it)
        for cand in buckets.values():
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    if hamming(int(cand[i]["phash"]), int(cand[j]["phash"])) <= radius:
                        uf.union(cand[i]["id"], cand[j]["id"])
        roots = {i: uf.find(i) for i in ids}
        exemplars = set()
        for root in set(roots.values()):
            exemplars.add(min(i for i in ids if roots[i] == root))
        for it in its:
            out[it["id"]] = {
                "dedup_cluster_id": _cluster_id(roots[it["id"]]),
                "is_exemplar": it["id"] in exemplars,
                "phash": int(it["phash"]),
            }
    return out


# =========================================================================== #
# Pass 2 — GPS/time overlap CROSS source (keep multi-traversal)                 #
# =========================================================================== #
GEO_CELL_DEG = 0.001          # ~100 m grid (lat degree ~111 km)
TIME_BUCKET_S = 5.0           # same place within this window = the same drive


def geo_cell(lat: float, lon: float, cell_deg: float = GEO_CELL_DEG) -> str:
    """Quantize (lat, lon) to a grid-cell id. The unit of the cross-source GPS
    index; ``None`` lat/lon → ``''`` (no-GPS clip, excluded from pass 2)."""
    if lat is None or lon is None:
        return ""
    return f"{round(float(lat) / cell_deg)}_{round(float(lon) / cell_deg)}"


def dedup_cross_source(items: list[dict], time_bucket_s: float = TIME_BUCKET_S
                       ) -> dict:
    """items = ``[{id, geo_cell, t}]`` (t = unix seconds; ``geo_cell=''`` = no GPS)
    → ``{id: {cross_cluster_id, is_exemplar, multi_traversal}}``.

    Collapse rule: same ``geo_cell`` AND same time bucket = one physical drive
    (re-host) → union, keep one exemplar. Same ``geo_cell`` but a DIFFERENT time
    bucket = a re-traversal of the same road → KEPT, tagged ``multi_traversal``
    (a wanted signal, not a dup). No-GPS clips are singletons, always exemplars."""
    gps = [it for it in items if it.get("geo_cell")]
    nogps = [it for it in items if not it.get("geo_cell")]

    ids = [it["id"] for it in gps]
    uf = _UF(ids)
    st_bucket: dict[tuple[str, int], list] = {}     # (geo_cell, time_bucket)
    cells: dict[str, set] = {}                      # geo_cell -> {time_buckets}
    for it in gps:
        tb = int(float(it["t"]) // time_bucket_s)
        st_bucket.setdefault((it["geo_cell"], tb), []).append(it["id"])
        cells.setdefault(it["geo_cell"], set()).add(tb)
    for group in st_bucket.values():                # same place+time -> re-host dup
        for other in group[1:]:
            uf.union(group[0], other)

    out: dict = {}
    if ids:
        roots = {i: uf.find(i) for i in ids}
        exemplars = set()
        for root in set(roots.values()):
            exemplars.add(min(i for i in ids if roots[i] == root))
        for it in gps:
            multi = len(cells[it["geo_cell"]]) > 1   # same road, >1 distinct time
            out[it["id"]] = {
                "cross_cluster_id": _cluster_id(roots[it["id"]]),
                "is_exemplar": it["id"] in exemplars,
                "multi_traversal": bool(multi),
            }
    for it in nogps:
        out[it["id"]] = {"cross_cluster_id": f"nogps:{it['id']}",
                         "is_exemplar": True, "multi_traversal": False}
    return out


# =========================================================================== #
# The two-pass driver                                                          #
# =========================================================================== #
@dataclass
class DedupVerdict:
    phash: int
    dedup_cluster_id: str
    geo_cell: str
    is_exemplar: bool                # exemplar in BOTH passes (survives selection)
    within_exemplar: bool
    cross_exemplar: bool
    multi_traversal: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"phash": self.phash, "dedup_cluster_id": self.dedup_cluster_id,
                "geo_cell": self.geo_cell, "is_exemplar": self.is_exemplar,
                "within_exemplar": self.within_exemplar,
                "cross_exemplar": self.cross_exemplar,
                "multi_traversal": self.multi_traversal}


def two_pass_dedup(items: list[dict], radius: int = NEAR_DUP_HAMMING
                   ) -> dict[object, DedupVerdict]:
    """Run both passes and merge → ``{id: DedupVerdict}``.

    ``items`` = ``[{id, source, phash, geo_cell?, t?}]`` (``geo_cell``/``t`` optional
    — absent = no-GPS clip). A clip is kept (``is_exemplar``) only if it is the
    exemplar of BOTH its perceptual cluster AND its GPS/time cluster; a
    multi-traversal is never suppressed by pass 2 (it lands in its own time bucket)."""
    within = dedup_within_source(items, radius=radius)
    cross = dedup_cross_source(
        [{"id": it["id"], "geo_cell": it.get("geo_cell", ""), "t": it.get("t", 0.0)}
         for it in items])
    out: dict[object, DedupVerdict] = {}
    for it in items:
        i = it["id"]
        w, c = within[i], cross[i]
        out[i] = DedupVerdict(
            phash=w["phash"],
            dedup_cluster_id=w["dedup_cluster_id"],
            geo_cell=it.get("geo_cell", ""),
            within_exemplar=w["is_exemplar"],
            cross_exemplar=c["is_exemplar"],
            is_exemplar=bool(w["is_exemplar"] and c["is_exemplar"]),
            multi_traversal=c["multi_traversal"],
        )
    return out
