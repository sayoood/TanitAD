"""Step-stamped, ATOMIC checkpoint saving.

⛔ THE DEFECT: `--save-every` overwrote `ckpt.pt`. MEASURED across all three
banked 30k arms — each directory holds exactly ONE checkpoint, overwritten 12
times. A val curve can prove step 12,300 was the best model in the run and that
model no longer exists. Selection needs ARTIFACTS, not just a signal.
"""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train_v6_staged as T  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(2, 2)


def _save(tmp_path, step, keep_step=True):
    m = _Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    T._save_ckpt(tmp_path / "ckpt.pt", stack=m, opt=opt, step=step,
                 cfg_json={"x": 1}, keep_step=keep_step)


def test_a_step_stamped_copy_is_kept(tmp_path):
    _save(tmp_path, 2500)
    assert (tmp_path / "ckpt.pt").exists()
    assert (tmp_path / "ckpt_step2500.pt").exists(), (
        "without this, a 30k run leaves ONE checkpoint and the trajectory is "
        "unrecoverable")


def test_successive_saves_do_NOT_overwrite_each_other(tmp_path):
    """⭐ THE WHOLE POINT: 12 saves must leave 12 selectable artifacts."""
    for s in (2500, 5000, 7500):
        _save(tmp_path, s)
    stamped = sorted(p.name for p in tmp_path.glob("ckpt_step*.pt"))
    assert stamped == ["ckpt_step2500.pt", "ckpt_step5000.pt", "ckpt_step7500.pt"]


def test_the_latest_is_still_written_for_resume(tmp_path):
    _save(tmp_path, 2500)
    _save(tmp_path, 5000)
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["step"] == 5000


def test_each_stamped_copy_carries_ITS_OWN_step(tmp_path):
    for s in (2500, 5000):
        _save(tmp_path, s)
    for s in (2500, 5000):
        ck = torch.load(tmp_path / f"ckpt_step{s}.pt", weights_only=False)
        assert ck["step"] == s, "a stamped copy holding the wrong step is worse than none"


def test_no_TEMP_FILES_survive_a_successful_save(tmp_path):
    """⛔ ATOMIC BY RENAME. A mid-write copy yields a TORN CHECKPOINT THAT LOADS
    AND IS WRONG — worse than a missing one. Polling for a stable size only
    narrows the window; os.replace closes it, and leaves no debris."""
    _save(tmp_path, 2500)
    assert not list(tmp_path.glob("*.tmp")), "temp files must be renamed away"


def test_the_optout_exists_and_ACTUALLY_SUPPRESSES(tmp_path):
    """⚠️ The flag must change BEHAVIOUR, not merely parse — the lesson from an
    escape hatch that named itself and refused anyway."""
    _save(tmp_path, 2500, keep_step=False)
    assert (tmp_path / "ckpt.pt").exists()
    assert not list(tmp_path.glob("ckpt_step*.pt")), "opt-out must suppress"


def test_keeping_step_ckpts_is_the_DEFAULT():
    """⚠️ Deliberately opposite to nav_cond's default-False, and the distinction
    is the point: a default that changes ARTIFACTS is not a default that changes
    ARCHITECTURE. This alters no model, no loss, no RNG draw and no
    comparability — it only adds files."""
    import inspect
    sig = inspect.signature(T._save_ckpt)
    assert sig.parameters["keep_step"].default is True
    ap = T.build_parser()
    a = ap.parse_args(["--stage", "S-W", "--out", "x", "--v2-cache", "y"])
    assert getattr(a, "no_step_ckpts", False) is False
