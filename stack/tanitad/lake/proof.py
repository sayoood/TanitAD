"""Acceptance gate: byte-equivalence of lake windows vs the current contract.

The Phase-A gate (spec §7 Phase A, task point 4): prove that
``LakeWindowDataset`` yields windows BIT-IDENTICAL to ``EpisodeWindowDataset``
over the same episodes — frames/actions/future_*/poses equal to the byte, and
the CORPUS_META / I7 identity (channels/size/f_eff) preserved. This module is the
reusable primitive; ``scripts/lake_byteproof.py`` and the test both call it.
"""

from __future__ import annotations

from typing import Any

import torch

WINDOW_KEYS = ("frames", "actions", "future_frames", "future_actions",
               "future_poses", "pose_last")


def _tensor_equal(a, b) -> bool:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return a == b
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def assert_datasets_bit_identical(ds_ref, ds_lake, keys=WINDOW_KEYS,
                                  max_report: int = 5) -> dict[str, Any]:
    """Assert two window datasets are byte-identical; return a proof summary.

    Compares length, then every window's tensors (exact ``torch.equal`` — same
    shape, dtype, and bytes) plus ``episode_id``. Raises ``AssertionError`` on
    the first mismatch with a precise locator.
    """
    n_ref, n_lake = len(ds_ref), len(ds_lake)
    assert n_ref == n_lake, (
        f"window count differs: ref={n_ref} lake={n_lake}")
    assert n_ref > 0, "no windows to compare — empty datasets"

    checksum = 0
    for i in range(n_ref):
        a, b = ds_ref[i], ds_lake[i]
        if int(a["episode_id"]) != int(b["episode_id"]):
            raise AssertionError(
                f"window {i}: episode_id {a['episode_id']} != {b['episode_id']} "
                f"(episode ordering diverged)")
        for k in keys:
            if k not in a and k not in b:
                continue
            if not _tensor_equal(a[k], b[k]):
                ta, tb = a.get(k), b.get(k)
                sa = tuple(ta.shape) if torch.is_tensor(ta) else ta
                sb = tuple(tb.shape) if torch.is_tensor(tb) else tb
                raise AssertionError(
                    f"window {i} key {k!r} DIFFERS: ref{sa} vs lake{sb}"
                    + (f"; max|diff|={float((ta-tb).abs().max()):.3e}"
                       if torch.is_tensor(ta) and torch.is_tensor(tb)
                       and ta.shape == tb.shape else ""))
        # cheap running checksum over the frames for the report
        f = a["frames"]
        checksum = (checksum + int(f.reshape(-1)[::997].to(torch.float64).sum()
                                   .item())) & 0xFFFFFFFF

    ref0 = ds_ref[0]
    return {
        "windows": n_ref,
        "keys_compared": [k for k in keys if k in ref0],
        "frame_shape": tuple(ref0["frames"].shape),
        "frame_dtype": str(ref0["frames"].dtype),
        "channels": int(ref0["frames"].shape[1]),
        "image_size": int(ref0["frames"].shape[-1]),
        "frames_checksum": checksum,
        "bit_identical": True,
    }


def assert_corpus_meta_identity(members: list[dict], expected) -> dict[str, Any]:
    """Assert the lake preserves the CORPUS_META / I7 fields (channels, size,
    f_eff, hz) recorded in the catalog — the task-identity fingerprint holds."""
    import pyarrow.dataset as pads  # noqa: F401  (kept for API parity)
    got = {"channels": set(), "image_size": set(), "f_eff_px": set(),
           "hz": set()}
    for m in members:
        # members from resolve_members are minimal; the caller passes catalog rows
        for k in got:
            if k in m:
                got[k].add(m[k])
    checks = {}
    for k, exp in (("channels", expected["channels"]),
                   ("image_size", expected["image_size"]),
                   ("f_eff_px", expected["f_eff_px"]), ("hz", expected["hz"])):
        vals = got.get(k, set())
        if vals:
            assert vals == {exp}, f"CORPUS_META {k}: lake has {vals}, expected {exp}"
            checks[k] = exp
    return checks
