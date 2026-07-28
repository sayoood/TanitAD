"""⛔ THE INERT-BUFFER EXEMPTION MUST NOT BECOME A GENERAL ``strict=False``.

THE DEFECT (MEASURED 2026-07-28, on the real 30k gate)
------------------------------------------------------
The completed ``flagship-v4-fromscratch`` arm (step 29,999) could not be loaded by
HEAD's ``eval_flagship_v4.py`` at all:

  RuntimeError: Missing key(s) in state_dict for FlagshipV4Head:
    "imag_pos", "src_embed", "imag_proj.weight", "imag_proj.bias",
    "vision_rank_proj.basis_loaded"

FOUR of those five were **self-inflicted**: the ckpt sat at ``v4fs_ckpt.pt`` with its
config at ``v4fs_config.json``, so the loader's ``<ckpt-dir>/config.json`` auto-detect
missed it and built the head from CURRENT defaults — which enable imagination. The
run's own config says ``"cond_imagination": false``. Passing ``--head-config`` removed
all four.

The FIFTH is real and benign: ``vision_rank_proj.basis_loaded`` is a scalar bool buffer
(``vision_rank.py:155``) registered AFTER this arm trained. ``VisionRankProj.forward``
reads only ``is_raw``, ``mu`` and ``proj`` — the flag cannot move a number.

⭐ THE RED HALF is the point of this file. Exempting one inert buffer is safe; degrading
to ``strict=False`` is not, because it would swallow the imagination mismatch above and
silently evaluate a DIFFERENT ARCHITECTURE than the checkpoint. A gate that cannot fail
on that is cover, not a guard.
"""
from __future__ import annotations

import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "scripts" / "eval_flagship_v4.py").read_text(encoding="utf-8")


def test_the_inert_set_contains_ONLY_the_provenance_flag() -> None:
    """If someone widens this set, they must come through this test first."""
    m = re.search(r"_INERT_BUFFERS\s*=\s*\{([^}]*)\}", SRC)
    assert m, "the inert-buffer allowlist vanished from eval_flagship_v4.py"
    entries = {e.strip().strip('"\'') for e in m.group(1).split(",") if e.strip()}
    assert entries == {"vision_rank_proj.basis_loaded"}, (
        f"inert-buffer allowlist widened to {entries}. Every entry must be a buffer "
        "that is provably NOT read in forward(); add the proof to the docstring.")


def test_a_missing_LEARNED_tensor_still_raises() -> None:
    """The head load must still reject anything outside the inert set."""
    assert "_hard_missing = [k for k in _missing if k not in _INERT_BUFFERS]" in SRC
    assert "if _hard_missing or _unexpected:" in SRC
    assert "raise RuntimeError(" in SRC.split("_hard_missing or _unexpected")[1][:400], (
        "a hard missing/unexpected key must RAISE, not warn")


def test_unexpected_keys_are_also_fatal() -> None:
    """``strict=False`` silently tolerates EXTRA keys too; that must stay fatal.

    Anchored on the whole source, not on a split around ``_INERT_BUFFERS`` — that name
    occurs twice, so splitting on it truncates the segment before the enforcement line
    and the assertion fails against correct code. (It did, on first run.)
    """
    assert "_unexpected = head.load_state_dict(" in SRC, (
        "unexpected keys are not captured from load_state_dict at all")
    assert "if _hard_missing or _unexpected:" in SRC, (
        "unexpected keys are captured but never enforced")


def test_the_error_names_the_head_config_remedy() -> None:
    """The four imagination keys cost a full eval round-trip to diagnose.

    The message must point at ``--head-config`` so the next person does not repeat it.
    """
    assert "--head-config" in SRC.split("_hard_missing or _unexpected")[1][:900], (
        "the failure message must name the --head-config remedy")


def test_load_state_dict_is_not_bare_strict_false_anywhere_for_the_head() -> None:
    """No unguarded ``head.load_state_dict(..., strict=False)`` may exist."""
    for line in SRC.splitlines():
        s = line.strip()
        if s.startswith("head.load_state_dict") and "strict=False" in s:
            assert "_missing, _unexpected" in s or "_missing," in s, (
                f"unguarded non-strict head load: {s}")
