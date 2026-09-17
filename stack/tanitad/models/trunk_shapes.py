"""refcv6: the trunk's feature-map SHAPE and CHANNEL COUNT, read at runtime.

⛔ **PI, 2026-09-16 (after `SPEC_REFCV6_V2.md` was written):** the input becomes
**256 x 1024** cylindrical at the same 120 deg field, and the trunk becomes
**resnet101** with **resnet34** as a comparison arm. Two consequences, and this
module exists so that neither can be written as a literal anywhere:

1. **The stride-16 map is 16 x 64 = 1,024 cells** (1.875 deg per column), not
   16 x 40 = 640 at 3.0 deg. Our own oracle study measured that cutting azimuth
   alone costs **57 % of AP**, so the perception branch is the direct
   beneficiary and must not be the thing that pins the old width.
2. **The channel count is a parameter, not a constant.** MEASURED from
   ``timm.feature_info`` 2026-09-16 (``pretrained=False``, so no download):

   | trunk | stride 16 | stride 32 |
   |---|---|---|
   | ``resnet34`` | **256** | 512 |
   | ``resnet101`` | **1024** | 2048 |

   Both arms must build. A head that hard-codes 256 silently mis-projects a
   resnet101 trunk -- or, worse, happens to run because someone inserted a
   1x1 "adapter" and the ablation then measures the adapter.

⭐ **The SAM3 label grid does NOT move.** It is 120 x 64 cells at 0.5 m in the
RIG frame -- metric, not pixel -- so it is unchanged by the image resolution
(:data:`semantic_map_gt.CART_SHAPE`). The lift's job is simply better sampled.
:func:`assert_label_grid_unmoved` states that in code so a future resolution
change cannot quietly drag the label with it.

## Use

    spec = TrunkSpec.from_timm("resnet101", FRAME_256x1024)
    spec.perception          # FeatureMap(stride=16, channels=1024, hw=(16, 64))
    spec.planner             # FeatureMap(stride=32, channels=2048, hw=(8, 32))

or, without timm (a stand-in trunk, a test, a checkpoint whose backbone is
already built)::

    spec = TrunkSpec.from_channels({16: 1024, 32: 2048}, FRAME_256x1024)
"""
from __future__ import annotations

from dataclasses import dataclass

from tanitad.data.calib import PHYSICALAI_WIDE120_256x640, CanonicalFrame
from tanitad.data.semantic_map_gt import CART_SHAPE

__all__ = [
    "FRAME_256x640", "FRAME_256x1024", "FRAME_416x1024",
    "PERCEPTION_STRIDE", "PLANNER_STRIDE",
    "FeatureMap", "TrunkSpec", "feature_hw", "assert_label_grid_unmoved",
    "frame_for_width",
]

#: the cache the real-data checks of 2026-09-16 ran against.
FRAME_256x640: CanonicalFrame = PHYSICALAI_WIDE120_256x640

#: ⭐ the PI's 2026-09-16 geometry. ``f_ref`` is the 640 frame's scaled by
#: 1024/640 -- ``305.5774907364391 * 1.6 = 488.92398517830253`` -- which holds
#: the field at EXACTLY 120.0000 deg, the same field the rig actually has.
#: ⚠️ The PI wrote "f_ref 488.92"; that rounding gives 120.0010 deg, a 0.0010 deg
#: (0.0085 px at the frame edge) error. The exact value is used here, and the
#: difference is recorded rather than silently adopted in either direction.
FRAME_256x1024: CanonicalFrame = CanonicalFrame(
    height=256, width=1024, f_ref=PHYSICALAI_WIDE120_256x640.f_ref * 1024 / 640,
    projection="cylindrical")

#: ``SPEC_REFCV6_V2.md`` §2: perception reads stride 16, the planner stride 32.
PERCEPTION_STRIDE: int = 16
PLANNER_STRIDE: int = 32


def frame_for_width(width: int, height: int = 256) -> CanonicalFrame:
    """The 120 deg cylindrical frame of a given width, ``f_ref`` scaled so the
    FIELD is invariant. ⛔ The field is a property of the LENS; changing the
    width without scaling ``f_ref`` silently changes what the camera sees."""
    if int(width) < 1 or int(height) < 1:
        raise ValueError(f"frame must be positive, got {height}x{width}")
    return CanonicalFrame(
        height=int(height), width=int(width),
        f_ref=PHYSICALAI_WIDE120_256x640.f_ref * int(width) / 640,
        projection="cylindrical")


#: ⭐ the PI's 2026-09-17 geometry, replacing 256x1024. ⛔ DERIVED, never typed:
#: ``frame_for_width`` is the one spelling of this camera constant, and because
#: 416x1024 keeps the WIDTH it keeps ``f_ref`` EXACTLY -- a cylindrical frame's
#: ``f_ref`` is a horizontal quantity (the column is linear in azimuth,
#: ``az_max = (W/2)/f_ref``), so only the VERTICAL field changes: 29.3414 deg at
#: 256 rows -> 46.0921 deg at 416. The builder's own manifest reports the same
#: 46.09213171161337, independently computed, which is the cross-check.
#: ⚠️ 408x1024 -- the height first authorised -- is UNBUILDABLE: ``408 % 32 = 24``,
#: so the stride-32 map would be mis-sized (the trunk declares 12 rows while a
#: stride-32 CNN emits 13) and ``timm_trunk.py:216`` refuses it. ``416 % 32 = 0``
#: gives exactly 13 rows and slightly EXCEEDS the 256x640 reference field
#: (45.4556 deg), so no vertical field is given up to gain the alignment.
FRAME_416x1024: CanonicalFrame = frame_for_width(1024, 416)


def feature_hw(frame: CanonicalFrame, stride: int) -> tuple[int, int]:
    """``(rows, cols)`` of a stride-``stride`` feature map of ``frame``.

    Refuses a frame the stride does not tile: a map that does not tile the frame
    has no exact pixel-to-feature-index map, and
    :func:`bev_lift.pixel_to_feature_grid` says so too.
    """
    s = int(stride)
    if s < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if frame.height % s or frame.width % s:
        raise ValueError(
            f"{frame.height}x{frame.width} is not divisible by stride {s}: the "
            f"feature map would not tile the frame")
    return (frame.height // s, frame.width // s)


@dataclass(frozen=True)
class FeatureMap:
    """One stride of the trunk: its channel count and its map shape."""

    stride: int
    channels: int
    hw: tuple[int, int]

    @property
    def n_tokens(self) -> int:
        return self.hw[0] * self.hw[1]

    @property
    def deg_per_column(self) -> float:
        """Azimuth subtended by one feature column -- the quantity the oracle
        study found load-bearing (57 % of AP)."""
        import math
        return math.degrees(float(self.stride) / _F_REF_OF[self]) \
            if self in _F_REF_OF else float("nan")

    def as_dict(self) -> dict:
        return {"stride": int(self.stride), "channels": int(self.channels),
                "hw": list(self.hw), "n_tokens": self.n_tokens}


_F_REF_OF: dict = {}          # filled by TrunkSpec so deg_per_column is exact


@dataclass(frozen=True)
class TrunkSpec:
    """The trunk's perception and planner feature maps, for ONE frame."""

    frame: CanonicalFrame
    perception: FeatureMap
    planner: FeatureMap
    source: str

    @classmethod
    def from_channels(cls, channels_by_stride: dict, frame: CanonicalFrame,
                      source: str = "explicit") -> "TrunkSpec":
        got = {int(k): int(v) for k, v in channels_by_stride.items()}
        for s in (PERCEPTION_STRIDE, PLANNER_STRIDE):
            if s not in got:
                raise ValueError(
                    f"no channel count for stride {s}; got strides "
                    f"{sorted(got)}. refcv6 reads stride {PERCEPTION_STRIDE} "
                    f"for perception and {PLANNER_STRIDE} for the planner "
                    f"(SPEC_REFCV6_V2.md §2)")
        maps = {}
        for s in (PERCEPTION_STRIDE, PLANNER_STRIDE):
            fm = FeatureMap(s, got[s], feature_hw(frame, s))
            _F_REF_OF[fm] = float(frame.f_ref)
            maps[s] = fm
        return cls(frame=frame, perception=maps[PERCEPTION_STRIDE],
                   planner=maps[PLANNER_STRIDE], source=str(source))

    @classmethod
    def from_timm(cls, model_or_name, frame: CanonicalFrame) -> "TrunkSpec":
        """Read the channel counts from ``feature_info`` -- never a literal.

        ``model_or_name``: a timm model with ``feature_info`` (as
        ``features_only=True`` builds), or a name to construct
        ``pretrained=False`` purely to read its shape (no download, no weights).
        """
        m = model_or_name
        name = getattr(m, "default_cfg", {}).get("architecture") if not isinstance(m, str) else m
        if isinstance(m, str):
            import timm
            m = timm.create_model(m, pretrained=False, features_only=True)
        fi = getattr(m, "feature_info", None)
        if fi is None:
            raise ValueError(
                f"{type(m).__name__} has no `feature_info`; build the backbone "
                f"with timm `features_only=True`, or pass the channel counts "
                f"explicitly via `from_channels` -- this module refuses to guess "
                f"a channel count (PI 2026-09-16)")
        red = list(fi.reduction())
        ch = list(fi.channels())
        by_stride = {int(r): int(c) for r, c in zip(red, ch)}
        return cls.from_channels(by_stride, frame,
                                 source=f"timm:{name or 'model'}")

    def as_dict(self) -> dict:
        return {"frame": {"height": self.frame.height, "width": self.frame.width,
                          "f_ref": float(self.frame.f_ref),
                          "projection": self.frame.projection,
                          "hfov_deg": round(self.frame.hfov_deg(), 4)
                          if hasattr(self.frame, "hfov_deg") else None},
                "perception": self.perception.as_dict(),
                "planner": self.planner.as_dict(),
                "source": self.source}


def assert_label_grid_unmoved(shape=CART_SHAPE) -> tuple[int, int]:
    """⛔ The SAM3 label grid is METRIC and must not move with image resolution.

    Raises if ``semantic_map_gt.CART_SHAPE`` is no longer 120 x 64. Called by the
    guard set so that a future resolution change cannot drag the label with it
    -- the head predicts on the label's own cells, and if the cells move, the
    prediction and the label are quietly on different grids.
    """
    if tuple(shape) != (120, 64):
        raise RuntimeError(
            f"the SAM3 label grid is {tuple(shape)}, not (120, 64). It is a "
            f"METRIC grid (0.5 m cells in the rig frame) and does not move with "
            f"the image resolution: if this really changed, every lift geometry "
            f"and every banked map file has to be rebuilt, which is not a "
            f"silent edit")
    return tuple(shape)


if __name__ == "__main__":
    import json
    for frame, tag in ((FRAME_256x640, "256x640 (the 2026-09-16 cache)"),
                       (FRAME_256x1024, "256x1024 (PI 2026-09-16)")):
        print(f"\n=== {tag} ===")
        for name in ("resnet34", "resnet101"):
            sp = TrunkSpec.from_timm(name, frame)
            p, q = sp.perception, sp.planner
            print(f"  {name:10s} perception s{p.stride} {p.channels:5d}ch "
                  f"{p.hw[0]}x{p.hw[1]} = {p.n_tokens:5d} tokens   |   "
                  f"planner s{q.stride} {q.channels:5d}ch {q.hw[0]}x{q.hw[1]}")
        import math
        deg = math.degrees(16.0 / frame.f_ref)
        print(f"  stride-16 column subtends {deg:.3f} deg")
    print("\nlabel grid:", assert_label_grid_unmoved())
    _ = json
