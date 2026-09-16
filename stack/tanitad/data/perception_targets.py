"""refcv6: the loader / trainer hooks for the MAP and BOX targets.

``SPEC_REFCV6_V2.md`` §6 and §7 put two supervised heads on the trunk. This
module is everything between the files on disk and the two losses:

* :class:`MapGTStore` -- resolve, open and cache one clip's SAM3 GT, under
  EITHER the canonical ``<root>/semantic_maps/gt/<clip_id>.sam3mapgt.npz``
  layout or the dev-box's flat, **sha12-named** directory. It hands both to
  ``semantic_map_gt.open_path``, so the schema / identity / frame-axis checks
  have exactly one spelling.
* :func:`require_map_coverage` -- ⛔ **REFUSES a run** when fewer than
  ``min_frac`` (default **0.90**, the brief's floor) of the train windows have a
  map frame. Not a warning: a trainer that starts with 40 % coverage produces a
  map head trained on a biased subset, and nothing downstream can tell.
* :func:`assert_frame_alignment` -- the label's ``t_img_us`` must equal the
  window's own frame timestamp. Delegates to
  ``ClipMapGT.check_times`` (1 ms) and adds the WINDOW -> RAW FRAME step, which
  is where the off-by-``n_stack`` error lives.
* :func:`collate_map_targets` / :func:`collate_box_targets` -- the batched
  tensors :func:`bev_encoder.map_soft_ce` and :func:`box3d_head.box3d_set_loss`
  consume.

## The three index spaces, and why the hook takes the window

There are three, and confusing two of them is the defect this module exists to
make impossible:

===========================  ==========================================
``window`` / stacked row j   what the trainer iterates
``raw`` v2ep frame index     ``j + n_stack - 1`` -- the CURRENT frame of
                             the stack (``v2_dataset.py:115-117``,
                             ``semantic_map_gt.raw_frame_index``); this is
                             the label axis
``cam_frame_idx``            the camera frame the SAM3 pixels came from;
                             stored in the GT, never computed here
===========================  ==========================================

:meth:`MapGTStore.frames_for_windows` takes WINDOW indices and an ``n_stack``
and does the conversion once, in one place.

## What is counted, and how

:func:`require_map_coverage` reports four states per window, never two:
``ok`` (a GT file that validates AND a frame in range), ``no_file``,
``frame_out_of_range``, ``inconclusive`` (any I/O or schema failure). The
fraction is exact only when ``n_inconclusive == 0``; otherwise it is a LOWER
BOUND and the verdict says so -- the ``semantic_map_gt.coverage`` rule.

Clip ids stay out of every returned record: sha12 only.
"""
from __future__ import annotations

import errno
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tanitad.data.semantic_map_gt import (
    CART_SHAPE, GT_SUBDIR, N_CHANNELS, TIME_TOL_US, ClipMapGT, MapFrames,
    SemanticMapGTError, TimeMisalignment, gt_path, open_path, raw_frame_index,
    sha12,
)

__all__ = [
    "GT_SUFFIX", "MIN_MAP_COVERAGE", "MapGTStore", "WindowMapTargets",
    "require_map_coverage", "MapCoverageTooLow", "assert_frame_alignment",
    "collate_map_targets", "collate_box_targets",
]

GT_SUFFIX = ".sam3mapgt.npz"
#: the brief's floor: *"a coverage check that REFUSES a run when < 90 % of train
#: windows have a map frame"*.
MIN_MAP_COVERAGE: float = 0.90


class MapCoverageTooLow(RuntimeError):
    """Raised BEFORE a run when map coverage is under the floor."""


# --------------------------------------------------------------------------- #
# the store                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class MapGTStore:
    """Resolve + cache the SAM3 GT of many clips.

    ``root``: a directory. Three layouts are probed, in this order, and the one
    that HITS is recorded per clip in :attr:`layout_of`:

    1. ``<root>/semantic_maps/gt/<clip_id>.sam3mapgt.npz``  (canonical)
    2. ``<root>/<clip_id>.sam3mapgt.npz``                   (flat by clip id)
    3. ``<root>/<sha12>.sam3mapgt.npz``                     (flat by sha12 --
       what the 2026-09-16 Thor copy of the 135 eval maps is)

    ⚠️ Layout 3 identifies the file by the SAME sha12 the file stores inside
    ``meta.source.clip_sha12``, so a name/content mismatch is still caught:
    ``open_path`` re-derives the identity from ``clip_id`` and refuses a file
    that belongs to another clip. The file NAME is a hint; the CONTENT is the
    identity.

    ``max_open``: how many decompressed ``cart_frac`` arrays to keep (each ~14 MB
    for 201 frames). Least-recently-used is evicted. ⛔ Dev-box default is small
    on purpose -- 135 clips fully resident is ~1.9 GB.
    """

    root: Path
    max_open: int = 4
    _open: dict = field(default_factory=dict, repr=False)
    _order: list = field(default_factory=list, repr=False)
    layout_of: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if int(self.max_open) < 1:
            raise ValueError(f"max_open must be >= 1, got {self.max_open}")

    # ---- resolution ------------------------------------------------------- #
    def candidate_paths(self, clip_id: str) -> list:
        s12 = sha12(clip_id)
        return [gt_path(self.root, clip_id),
                self.root / f"{clip_id}{GT_SUFFIX}",
                self.root / f"{s12}{GT_SUFFIX}"]

    def resolve(self, clip_id: str) -> Path | None:
        """The first existing candidate, or ``None``. Records the layout."""
        for i, p in enumerate(self.candidate_paths(clip_id)):
            try:
                if p.is_file():
                    self.layout_of[sha12(clip_id)] = ("canonical", "flat_clip_id",
                                                      "flat_sha12")[i]
                    return p
            except OSError:
                continue
        return None

    # ---- opening ---------------------------------------------------------- #
    def open(self, clip_id: str) -> ClipMapGT:
        s12 = sha12(clip_id)
        if s12 in self._open:
            self._order.remove(s12)
            self._order.append(s12)
            return self._open[s12]
        path = self.resolve(clip_id)
        if path is None:
            raise FileNotFoundError(
                errno.ENOENT, f"no SAM3 map GT for clip sha12 {s12} under "
                              f"{self.root} (probed canonical "
                              f"{'/'.join(GT_SUBDIR)}/, flat clip-id, flat sha12)",
                str(self.root))
        gt = open_path(path, clip_id)
        self._open[s12] = gt
        self._order.append(s12)
        while len(self._order) > int(self.max_open):
            self._open.pop(self._order.pop(0), None)
        return gt

    def close(self) -> None:
        self._open.clear()
        self._order.clear()

    # ---- the window -> raw-frame step, in ONE place ----------------------- #
    @staticmethod
    def raw_frames(window_idx, n_stack: int = 3) -> np.ndarray:
        """Window (stacked-row) indices -> the RAW v2ep frames whose labels they
        carry. A thin, named wrapper over ``semantic_map_gt.raw_frame_index`` so
        no call site writes ``+ n_stack - 1`` itself."""
        return raw_frame_index(window_idx, n_stack)

    def frames_for_windows(self, clip_id: str, window_idx, *, n_stack: int = 3,
                           frame_t_us=None, tol_us: float = TIME_TOL_US
                           ) -> MapFrames:
        """Labels for WINDOW indices of one clip, alignment-checked.

        ``frame_t_us``: the episode's own timestamps for those windows' CURRENT
        frames. When given (it should be, in a trainer) the 1 ms assertion runs;
        when absent the returned :class:`MapFrames` still carries ``t_img_us``
        so the caller can check it later, and :func:`assert_frame_alignment`
        exists for exactly that.
        """
        gt = self.open(clip_id)
        raw = self.raw_frames(window_idx, n_stack)
        return gt.read(raw, frame_t_us, tol_us=tol_us)


@dataclass(frozen=True)
class WindowMapTargets:
    """A batch of map targets, ready for :func:`bev_encoder.map_soft_ce`."""

    frac: torch.Tensor        # [B, 9, 120, 64] float32 in [0, 1]
    seen: torch.Tensor        # [B, 120, 64] bool
    t_img_us: torch.Tensor    # [B] int64
    raw_frame: torch.Tensor   # [B] int64
    clip_sha12: tuple         # [B] identity, sha12 only


# --------------------------------------------------------------------------- #
# coverage -- the refusal                                                      #
# --------------------------------------------------------------------------- #
def require_map_coverage(windows, store: MapGTStore, *, n_stack: int = 3,
                         min_frac: float = MIN_MAP_COVERAGE,
                         raise_on_low: bool = True) -> dict:
    """⛔ Refuse a run whose train windows are not ``min_frac`` map-covered.

    ``windows``: an iterable of ``(clip_id, window_idx)`` pairs -- the trainer's
    own index, so this measures the windows that will actually be drawn, not the
    clips that happen to have files.

    Returns the report; raises :class:`MapCoverageTooLow` when
    ``frac_ok < min_frac`` and ``raise_on_low`` (the default). ⚠️ The comparison
    uses ``frac_ok``, the LOWER bound -- an inconclusive window counts against
    the run, because "we could not tell" is not coverage.
    """
    pairs = [(str(c), int(i)) for c, i in windows]
    if not pairs:
        raise ValueError("no windows: coverage of an empty set is not 1.0, it "
                         "is undefined")
    states = {"ok": 0, "no_file": 0, "frame_out_of_range": 0, "inconclusive": 0}
    per_clip: dict = {}
    #: sha12 -> n_frames, or a FAILURE STATE string. ⛔ Caching a sentinel
    #: ``n_frames = -1`` here instead was a real defect, caught by the guard's
    #: own mutation test on 2026-09-16: a clip with NO FILE then reported its
    #: first window as ``no_file`` and every later one as
    #: ``frame_out_of_range``, because ``0 <= raw < -1`` is false. The count was
    #: still right in total and wrong in every column -- exactly the kind of
    #: diagnostic that sends the next reader to the wrong cause.
    clip_state: dict = {}
    reasons: dict = {}
    for cid, wi in pairs:
        s12 = sha12(cid)
        rec = per_clip.setdefault(s12, dict(states))
        if s12 not in clip_state:
            try:
                clip_state[s12] = int(store.open(cid).n_frames)
            except FileNotFoundError as e:
                clip_state[s12] = ("no_file" if e.errno == errno.ENOENT
                                   else "inconclusive")
                reasons.setdefault(s12, f"FileNotFoundError errno {e.errno}")
            except (SemanticMapGTError, OSError, ValueError) as e:
                clip_state[s12] = "inconclusive"
                reasons.setdefault(s12, f"{type(e).__name__}: "
                                        f"{str(e).replace(cid, '<clip>')[:200]}")
        st = clip_state[s12]
        if isinstance(st, str):
            key = st
        else:
            raw = int(raw_frame_index(wi, n_stack))
            key = "ok" if 0 <= raw < st else "frame_out_of_range"
        states[key] += 1
        rec[key] += 1
    n = len(pairs)
    frac_ok = states["ok"] / n
    frac_upper = (states["ok"] + states["inconclusive"]) / n
    rep = {
        "n_windows": n, "n_clips": len(per_clip), "n_stack": int(n_stack),
        **{f"n_{k}": v for k, v in states.items()},
        "frac_ok": frac_ok, "frac_ok_upper": frac_upper,
        "min_frac": float(min_frac),
        "verdict": ("INCONCLUSIVE" if states["inconclusive"]
                    else ("PASS" if frac_ok >= float(min_frac) else "FAIL")),
        "per_clip": per_clip, "reasons": reasons,
        "layout_of": dict(store.layout_of),
    }
    if raise_on_low and frac_ok < float(min_frac):
        raise MapCoverageTooLow(
            f"SAM3 map coverage {frac_ok:.4f} of {n} train windows is below the "
            f"floor {float(min_frac):.2f} "
            f"(ok {states['ok']}, no_file {states['no_file']}, "
            f"out_of_range {states['frame_out_of_range']}, inconclusive "
            f"{states['inconclusive']} across {len(per_clip)} clips). ⛔ The run "
            f"is refused BEFORE the GPU: a map head trained on a biased subset "
            f"of windows cannot be told apart downstream from one trained on "
            f"all of them.")
    return rep


# --------------------------------------------------------------------------- #
# alignment                                                                    #
# --------------------------------------------------------------------------- #
def assert_frame_alignment(store: MapGTStore, clip_id: str, window_idx,
                           frame_t_us, *, n_stack: int = 3,
                           tol_us: float = TIME_TOL_US) -> dict:
    """⛔ The label's ``t_img_us`` must equal the window's own frame timestamp.

    ``window_idx`` are the trainer's stacked-row indices and ``frame_t_us`` the
    episode timestamps of those windows' CURRENT frames. Raises
    :class:`semantic_map_gt.TimeMisalignment` on any window off by more than
    ``tol_us`` (1 ms). Returns ``{"worst_abs_us", "n", "raw_frame"}``.

    ⚠️ The ``+ (n_stack - 1)`` is applied HERE and nowhere else in a trainer: a
    caller that passes RAW frames as ``window_idx`` would silently label every
    window with a frame 2 steps early, and at 10 Hz and 30 km/h that is 1.7 m of
    map shift -- inside the tolerance of nothing and visible in no metric.
    """
    gt = store.open(clip_id)
    raw = raw_frame_index(window_idx, n_stack)
    t = np.atleast_1d(np.asarray(frame_t_us, dtype=np.float64))
    if t.shape != np.atleast_1d(raw).shape:
        raise TimeMisalignment(
            f"[{sha12(clip_id)}] {t.size} frame timestamps for "
            f"{np.atleast_1d(raw).size} windows")
    worst = gt.check_times(raw, t, tol_us)
    return {"worst_abs_us": float(worst), "n": int(t.size),
            "raw_frame": np.atleast_1d(raw).astype(np.int64),
            "clip_sha12": gt.clip_sha12}


# --------------------------------------------------------------------------- #
# collation                                                                    #
# --------------------------------------------------------------------------- #
def collate_map_targets(store: MapGTStore, windows, *, n_stack: int = 3,
                        frame_t_us=None, device=None,
                        dtype=torch.float32) -> WindowMapTargets:
    """``[(clip_id, window_idx), ...]`` -> a batched :class:`WindowMapTargets`.

    ``frame_t_us``: optional per-window timestamps; when given, EVERY window is
    alignment-asserted before a single label is returned (fail loud, before the
    batch reaches a loss).
    """
    pairs = [(str(c), int(i)) for c, i in windows]
    if not pairs:
        raise ValueError("collate_map_targets: empty batch")
    ts = None if frame_t_us is None else np.asarray(frame_t_us, dtype=np.float64)
    if ts is not None and ts.size != len(pairs):
        raise ValueError(f"{ts.size} timestamps for {len(pairs)} windows")
    frac = torch.empty((len(pairs), N_CHANNELS) + CART_SHAPE, dtype=dtype)
    seen = torch.empty((len(pairs),) + CART_SHAPE, dtype=torch.bool)
    t_img = torch.empty(len(pairs), dtype=torch.int64)
    raws = torch.empty(len(pairs), dtype=torch.int64)
    s12s = []
    for k, (cid, wi) in enumerate(pairs):
        mf = store.frames_for_windows(
            cid, [wi], n_stack=n_stack,
            frame_t_us=None if ts is None else [float(ts[k])])
        frac[k] = torch.from_numpy(mf.cart[0]).to(dtype)
        seen[k] = torch.from_numpy(mf.seen[0])
        t_img[k] = int(mf.t_img_us[0])
        raws[k] = int(mf.frame_idx[0])
        s12s.append(sha12(cid))
    if device is not None:
        frac, seen, t_img, raws = (frac.to(device), seen.to(device),
                                   t_img.to(device), raws.to(device))
    return WindowMapTargets(frac=frac, seen=seen, t_img_us=t_img,
                            raw_frame=raws, clip_sha12=tuple(s12s))


def collate_box_targets(per_frame, *, n_pad: int | None = None, device=None,
                        dtype=torch.float32) -> dict:
    """Stack per-frame target dicts (``agent_slots.targets_from_join`` output,
    optionally widened by ``box3d_head.zh_targets``) into one batch.

    ``n_pad`` defaults to the largest agent count in the batch. Every key is
    padded to it; ``valid`` (and ``zh_mask`` when present) carry the padding, so
    nothing downstream has to know the per-frame counts.

    ⛔ A frame ABSENT from the join must not appear here at all -- that is
    NO_LABEL, not an empty set (``targets_from_join``'s own refusal). This
    function stacks what it is given and does not manufacture a row.
    """
    tgts = list(per_frame)
    if not tgts:
        raise ValueError("collate_box_targets: empty batch")
    widths = [int(t["box"].shape[1]) for t in tgts]
    pad = int(n_pad if n_pad is not None else max(widths))
    if pad < max(widths):
        raise ValueError(f"n_pad {pad} < largest frame ({max(widths)} agents)")
    keys_f = ["box", "yaw", "occ", "rates"]
    has_zh = all("zh_mask" in t for t in tgts)
    if has_zh:
        keys_f += ["cz", "h"]
    out: dict = {}
    for k in keys_f:
        ref = tgts[0][k]
        shape = (len(tgts), pad) + tuple(ref.shape[2:])
        fill = -1.0 if k == "occ" else 0.0
        buf = torch.full(shape, fill, dtype=dtype, device=device)
        for i, t in enumerate(tgts):
            w = int(t[k].shape[1])
            if w:
                buf[i, :w] = t[k][0].to(dtype=dtype, device=device)
        out[k] = buf
    for k, dt, fill in (("cls", torch.long, -1), ("valid", torch.bool, False),
                        ("rates_mask", torch.bool, False),
                        ("zh_mask", torch.bool, False)):
        if k == "zh_mask" and not has_zh:
            continue
        buf = torch.full((len(tgts), pad), fill, dtype=dt, device=device)
        for i, t in enumerate(tgts):
            w = int(t[k].shape[1])
            if w:
                buf[i, :w] = t[k][0].to(device=device)
        out[k] = buf
    if not has_zh:                      # the honest default: no 3-D labels
        out["cz"] = torch.zeros((len(tgts), pad), dtype=dtype, device=device)
        out["h"] = torch.zeros((len(tgts), pad), dtype=dtype, device=device)
        out["zh_mask"] = torch.zeros((len(tgts), pad), dtype=torch.bool,
                                     device=device)
    return out


def _scan_flat_store(root) -> dict:
    """What a flat, sha12-named directory holds -- a probe, not a coverage
    claim: sha12 -> path, and the count. Used by the real-data checks."""
    root = Path(root)
    out = {}
    if not root.is_dir():
        os.stat(root)
        return out
    for p in sorted(root.glob(f"*{GT_SUFFIX}")):
        out[p.name[:-len(GT_SUFFIX)]] = p
    return out


def coverage_fraction(rep: dict) -> float:
    """``frac_ok``, or NaN for an empty report -- so a caller never divides."""
    n = int(rep.get("n_windows", 0))
    return float(rep["frac_ok"]) if n else math.nan
