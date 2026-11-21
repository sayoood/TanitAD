"""E2-K2SAFE / H4 -- the camera-blind roll (W4) of a K2 built with ``--r9-bev-warp`` (E1 arm T1 / T1s).

Hazard (E2-ROLL SHIP_LIST H4): ``V3Dataset.__getitem__`` adds ``bw_frames / bw_pose_now / bw_pose_past / bw_has`` (refc_v3_train.py:4905-4907, via
``bev_warp_flag.item_fields``) for a model built with the temporal BEV warp.  ``camblind_r9.CENSUS`` does not know them and is FAIL-CLOSED, so a T1 K2's blind roll
REFUSES at window 0.  The wrong repair is to class ``bw_frames`` as non-camera: it is the OWN past camera pixels, and the warp feeds them to the trunk, so the "blind"
arm would read its own scene (REFCV9_DESIGN W4: "every camera-derived input from the donor").  The right repair (``camblind_roll_r9_e2.patch_camblind``): ``bw_frames``
CAMERA -> the DONOR episode's frames at the donor window's NOW row and the same lag rows; ``bw_pose_now / bw_pose_past`` EGO and ``bw_has`` META -> the window's OWN.

Everything imports the REAL modules (``camblind_r9``, ``bev_warp_flag.item_fields``, the launcher); the only fakes are the dataset and its episodes.  RED arms:
  * the STOCK instrument refuses a T1 item at window 0 (the H4 failure, reproduced);
  * the NAIVE repair (``bw_frames`` classed EGO / META) runs and LEAKS: its blind ``bw_frames`` equal the window's own on every window;
  * control (c)'s digest of the NON-camera inputs differs between the arms without the extension (it would hash ``bw_frames``) and is equal with it;
  * a donor fn that took the OWN episode / the T1 shuffle control's episode, a zero frame for a missing donor row, or a FUTURE row, each fails the pinned expectations.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
R9DIR = ROOT / "ops" / "r9e2_read"
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests"), str(R9DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.eval import camblind_r9 as CB  # noqa: E402
from tanitad.train import bev_warp_flag as BWF  # noqa: E402

import camblind_roll_r9_e2 as K2  # noqa: E402

LAGS = (2, 5)
T_BY_E = (20, 9, 9, 9)          # e0 is long (own history exists at t >= 4); every donor is short (its row - 5 < 0): the clamp case is guaranteed for e0's windows


@pytest.fixture(autouse=True)
def _restore_imports():
    """K2.main() installs an import hook and a fake r7_roll: nothing may leak into the next test."""
    mp, path, mods = list(sys.meta_path), list(sys.path), set(sys.modules)
    yield
    sys.meta_path[:] = mp
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        sys.modules.pop(m, None)


class _Ep:
    def __init__(self, e, T):
        g = torch.Generator().manual_seed(100 + e)
        self.frames = torch.randint(1, 256, (T, 9, 4, 4), generator=g, dtype=torch.uint8)      # 3 stacked RGB frames per row; >= 1: a zero frame is never a real one
        self.actions, self.poses = torch.randn(T, 2, generator=g), torch.randn(T, 4, generator=g)
        self.episode_id = 1000 + e


class _T1Eval:
    """r7_roll's Refcv6EvalWindows reduced to the window dict + the V3Dataset-style tail, INCLUDING the T1 hook (refc_v3_train.py:4905-4907) calling the REAL item_fields."""

    def __init__(self, t1=True, shuffle=False, window=3, max_horizon=2):
        self.episodes = [_Ep(e, T) for e, T in enumerate(T_BY_E)]
        self.window, self.max_horizon = window, max_horizon
        self.index = [(e, t) for e, ep in enumerate(self.episodes) for t in range(0, ep.frames.shape[0] - window - max_horizon, 2)]
        if t1:
            self._r9_bw = BWF.ItemSource(LAGS, shuffle, len(self.episodes))

    def _window_u8(self, i):
        e_i, t = self.index[i]
        ep, w = self.episodes[e_i], self.window
        return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w], "future_frames": torch.zeros(0, dtype=torch.uint8),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon], "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}

    def __getitem__(self, i):
        item = self._window_u8(i)
        e_i, t = self.index[i]
        item["nav_cmd"] = torch.tensor(e_i % 3)
        item["map_fine"] = torch.full((2, 2), e_i, dtype=torch.uint8)
        if getattr(self, "_r9_bw", None) is not None:
            item.update(BWF.item_fields(self.episodes, e_i, t + self.window - 1, self._r9_bw))
        return item


def _blind(census=None, **kw):
    """-> (instance, dig, drange): the blind class over _T1Eval, from the module's (possibly patched) make_camblind_dataset_cls."""
    dig, dr = {}, {}
    extra = {} if census is None else {"census": census}
    cls = CB.make_camblind_dataset_cls(_T1Eval, camera="derange", dig=dig, drange=dr, own_every=1, **extra, **kw)
    return cls(), dig, dr


def _expected_donor_frames(base, e_d, t_d):
    row_d = t_d + base.window - 1
    return torch.stack([base.episodes[e_d].frames[max(row_d - r, 0)][-3:] for r in LAGS])


# =========================================================================== #
# census rows                                                                  #
# =========================================================================== #
def test_the_extension_is_exactly_the_four_bw_rows_well_formed_with_resolving_pins():
    ext = K2.k2safe_census(CB)
    assert ext[:len(CB.CENSUS)] == CB.CENSUS and [e.name for e in ext[len(CB.CENSUS):]] == list(K2.BW_KEYS)
    CB._check_census(ext)
    assert K2.k2safe_census(CB) == ext                                                           # deterministic
    by = {e.name: e for e in ext if e.kind == "item"}
    assert (by["bw_frames"].cls, by["bw_frames"].pixel_derived) == (CB.CAMERA, True)
    assert (by["bw_pose_now"].cls, by["bw_pose_past"].cls, by["bw_has"].cls) == (CB.EGO, CB.EGO, CB.META)
    assert not any(by[k].pixel_derived for k in ("bw_pose_now", "bw_pose_past", "bw_has"))
    res = {e.name: (loc, n) for e, loc, n in CB.verify_census(Path(BWF.__file__).resolve().parents[2], ext[len(CB.CENSUS):])}
    assert all(n >= 1 and loc.startswith("tanitad/train/bev_warp_flag.py:") for loc, n in res.values()), res
    print("\n[bw census pins]", {k: v[0] for k, v in res.items()})
    # RED: a stale pin is read as n == 0
    e = by["bw_frames"]
    stale = CB.Entry(e.name, e.kind, e.cls, e.pixel_derived, e.reaches, e.src, "NOT IN THE SOURCE", "", "")
    assert CB.verify_census(Path(BWF.__file__).resolve().parents[2], (stale,))[0][2] == 0


def test_the_extension_covers_every_key_item_fields_emits():
    """The census keys == the REAL item_fields' keys (a fifth key added to item_fields would make this RED)."""
    ep = [_Ep(0, 12)]
    got = set(BWF.item_fields(ep, 0, 8, BWF.ItemSource(LAGS, False, 1)))
    assert got == set(K2.BW_KEYS) == set(BWF.BATCH_KEYS)
    assert CB.camera_keys("item", K2.k2safe_census(CB)) == ("frames", "future_frames", "bw_frames")


# =========================================================================== #
# RED: the stock instrument refuses (H4) and the naive repair leaks            #
# =========================================================================== #
def test_RED_the_stock_instrument_refuses_a_T1_item_at_window_0_and_a_non_T1_item_passes():
    for camera in ("derange", "identity"):
        cls = CB.make_camblind_dataset_cls(_T1Eval, camera=camera)
        with pytest.raises(CB.UnclassifiedKey, match="bw_frames"):
            cls()[0]
    cls = CB.make_camblind_dataset_cls(_NoT1, camera="derange")
    assert not any(k in cls()[0] for k in K2.BW_KEYS)


class _NoT1(_T1Eval):
    def __init__(self):
        super().__init__(t1=False)


def _own_bw_equal_count(ds, base):
    return sum(1 for i in range(len(base.index)) if torch.equal(ds[i]["bw_frames"], base[i]["bw_frames"]))


def test_RED_the_naive_repair_classing_bw_frames_non_camera_runs_and_LEAKS_the_own_past_frames():
    base = _T1Eval()
    naive = CB.CENSUS + (CB.Entry("bw_frames", "item", CB.META, False, True, "synthetic", "", "", "NAIVE: 'a history tensor, not the window'"),
                         CB.Entry("bw_pose_now", "item", CB.EGO, False, True, "synthetic", "", "", ""),
                         CB.Entry("bw_pose_past", "item", CB.EGO, False, True, "synthetic", "", "", ""),
                         CB.Entry("bw_has", "item", CB.META, False, True, "synthetic", "", "", ""))
    ds, dig, _ = _blind(census=naive)
    n = len(base.index)
    assert _own_bw_equal_count(ds, base) == n                                                      # the leak: the blind arm's past frames ARE its own, on every window
    # ... while its CURRENT pixels are the donor's (the naive instrument LOOKS blind): the leak is invisible to the stock controls
    e_i, t = base.index[0]
    e_d, t_d = dig[(e_i, t)]["donor"]
    assert e_d != e_i and torch.equal(ds[0]["frames"], base.episodes[e_d].frames[t_d:t_d + base.window])


# =========================================================================== #
# GREEN: the K2SAFE instrument                                                 #
# =========================================================================== #
@pytest.fixture
def patched():
    census, undo = K2.patch_camblind(CB)
    try:
        yield census
    finally:
        undo()


def test_GREEN_bw_frames_come_from_the_donor_and_the_other_bw_keys_pass_through(patched):
    base = _T1Eval()
    ds, dig, dr = _blind()
    n_clamped = n_valid = 0
    for i in range(len(base.index)):
        b, o = ds[i], base[i]
        e_i, t = base.index[i]
        e_d, t_d = dig[(e_i, t)]["donor"]
        assert e_d != e_i
        row_d = t_d + base.window - 1
        # bw_frames: the DONOR episode's frames, never the window's own
        assert b["bw_frames"].shape == o["bw_frames"].shape and b["bw_frames"].dtype == o["bw_frames"].dtype == torch.uint8
        assert not torch.equal(b["bw_frames"], o["bw_frames"])
        assert torch.equal(b["bw_frames"], _expected_donor_frames(base, e_d, t_d))
        for p, r in enumerate(LAGS):
            if row_d - r >= 0:                                                                     # where the trainer's own builder has a donor frame, it is that frame
                n_valid += 1
                plain = BWF.item_fields(base.episodes, e_d, row_d, BWF.ItemSource(LAGS, False, len(base.episodes)))
                assert torch.equal(b["bw_frames"][p], plain["bw_frames"][p])
            else:
                n_clamped += 1
                assert torch.equal(b["bw_frames"][p], base.episodes[e_d].frames[0][-3:])           # clamped to the donor's first frame: a REAL donor frame, never zeros
                assert b["bw_frames"][p].any()
        # the other bw_* keys are the window's OWN, untouched
        for k in ("bw_pose_now", "bw_pose_past", "bw_has"):
            assert torch.equal(b[k], o[k]), k
        # and every other key, as in the stock instrument
        for k in o:
            if k not in CB.camera_keys("item"):
                assert torch.equal(b[k], o[k]) if torch.is_tensor(o[k]) else b[k] == o[k], k
        assert torch.equal(b["frames"], base.episodes[e_d].frames[t_d:t_d + base.window])           # the current pixels are still the donor window's
    assert n_clamped > 0 and n_valid > 0                                                           # both branches were exercised
    own_true_clamped = [i for i in range(len(base.index)) if base[i]["bw_has"].all() and base.index[i][0] == 0]
    assert own_true_clamped, "no window with own history (has True) and a short donor: the clamp case is untested"


def test_GREEN_zero_leak_on_every_window_and_the_donor_fn_is_the_only_new_reader(patched):
    base = _T1Eval()
    ds, _, _ = _blind()
    assert _own_bw_equal_count(ds, base) == 0
    assert ds.__class__.__mro__[1] is _T1Eval and CB.make_camblind_dataset_cls.__name__ == "make_camblind_dataset_cls"


def test_GREEN_the_script_donor_fn_is_overridden_it_would_re_enter_the_blind_window_u8(patched):
    """camblind_roll_r9.py passes ``donor_item_fn=lambda ds, j: V3Dataset.__getitem__(ds, j)``; the wrapper must not use it (it re-enters the BLIND ``_window_u8``)."""
    def boom(ds, j):
        raise AssertionError("the script's donor_item_fn was called")
    ds, dig, _ = _blind(donor_item_fn=boom)
    base = _T1Eval()
    assert not torch.equal(ds[3]["bw_frames"], base[3]["bw_frames"])


def test_GREEN_control_c_digest_of_the_non_camera_inputs_is_equal_between_the_arms_only_with_the_extension(patched):
    base = _T1Eval()
    ident_cls = CB.make_camblind_dataset_cls(_T1Eval, camera="identity")
    ident, (blind, _, _) = ident_cls(), _blind()
    nav, v0, es = torch.tensor([1]), torch.tensor([2.0]), torch.zeros(1, 5)
    for i in range(len(base.index)):
        a, b = ident[i], blind[i]
        assert CB.other_inputs_digest(a, nav, v0, es) == CB.other_inputs_digest(b, nav, v0, es)               # the patched module function: bw_frames is CAMERA, excluded
        assert CB.other_inputs_digest(a, nav, v0, es, census=CB.CENSUS) != CB.other_inputs_digest(b, nav, v0, es, census=CB.CENSUS)   # RED: stock census hashes bw_frames
        assert CB.other_inputs_digest(a, nav, v0, es, census=K2.k2safe_census(CB)) == CB.other_inputs_digest(b, nav, v0, es, census=K2.k2safe_census(CB))


def test_GREEN_identity_arm_is_the_batterys_own_item_with_a_T1_K2(patched):
    base = _T1Eval()
    ds = CB.make_camblind_dataset_cls(_T1Eval, camera="identity")()
    for i in range(len(base.index)):
        a, o = ds[i], base[i]
        assert a.keys() == o.keys() and all(torch.equal(a[k], o[k]) for k in o if torch.is_tensor(o[k]))


def test_a_non_T1_K2_is_unchanged_by_the_extension(patched):
    base = _NoT1()
    cls = CB.make_camblind_dataset_cls(_NoT1, camera="derange", own_every=1)
    d2 = cls()
    for i in range(len(base.index)):
        item, o = d2[i], base[i]
        assert item.keys() == o.keys() and not any(k in item for k in K2.BW_KEYS)
        e_i, t = base.index[i]
        assert torch.equal(item["frames"], base.episodes[type(d2)._dw[(e_i, t)][0]].frames[type(d2)._dw[(e_i, t)][1]:][:base.window])
    assert K2.bw_donor_item_fn(types.SimpleNamespace(index=None, episodes=None), 0) == {}         # no _r9_bw: returns before touching anything


def test_the_donor_fn_is_past_only_never_shuffled_and_never_zero():
    seen = []

    class Frames:
        def __init__(self, tag):
            self.tag = tag

        def __getitem__(self, r):
            seen.append((self.tag, int(r)))
            return torch.full((9, 4, 4), 7 + self.tag, dtype=torch.uint8)

    eps = [types.SimpleNamespace(frames=Frames(e)) for e in range(4)]
    for shuffle in (False, True):
        ds = types.SimpleNamespace(index=[(2, 6), (3, 0)], episodes=eps, window=3, _r9_bw=BWF.ItemSource(LAGS, shuffle, 4))
        seen.clear()
        out = K2.bw_donor_item_fn(ds, 0)["bw_frames"]
        assert {tag for tag, _ in seen} == {2}                                                     # the donor episode itself, even for the T1 shuffle control's source
        assert sorted(r for _, r in seen) == [3, 6] and max(r for _, r in seen) <= 6 + 3 - 1       # row_d = t_d + w - 1 = 8; rows 8 - 5, 8 - 2: PAST rows only
        assert out.shape == (2, 3, 4, 4) and bool((out == 9).all())
        seen.clear()
        out = K2.bw_donor_item_fn(ds, 1)["bw_frames"]                                              # row_d = 2: rows 0 and -3 -> clamped to 0 (a real frame, not zeros)
        assert sorted(r for _, r in seen) == [0, 0] and bool((out == 10).all())


# =========================================================================== #
# the patch's mechanics                                                        #
# =========================================================================== #
def test_the_patch_rebinds_exactly_four_entry_points_handles_positional_census_and_undoes():
    stock = {n: getattr(CB, n) for n in K2.PATCHED}
    sha0, keys0 = CB.census_sha256(), CB.camera_keys("item")
    census, undo = K2.patch_camblind(CB)
    try:
        assert all(getattr(CB, n) is not stock[n] for n in K2.PATCHED)
        assert CB.camera_keys("item") == keys0 + ("bw_frames",) and CB.census_sha256() != sha0
        assert CB.census_sha256() == stock["census_sha256"](census) and CB.camera_keys("item", CB.CENSUS) == keys0       # CB's own internal calls pass census positionally
        assert CB.census_sha256(CB.CENSUS) == sha0
        assert [n for n in dir(CB) if getattr(getattr(CB, n), "__wrapped__", None) is not None] == sorted(K2.PATCHED)
    finally:
        undo()
        undo()                                                                                     # idempotent
    assert all(getattr(CB, n) is stock[n] for n in K2.PATCHED) and CB.census_sha256() == sha0 and CB.camera_keys("item") == keys0


def test_RED_a_census_with_another_camera_key_is_refused_the_donor_fn_knows_only_bw_frames():
    ext = K2.k2safe_census(CB) + (CB.Entry("bev_cache", "item", CB.CAMERA, True, True, "synthetic", "", "", ""),)
    with pytest.raises(SystemExit, match="bev_cache"):
        K2.patch_camblind(CB, ext)
    with pytest.raises(SystemExit, match=r"beyond the window pixels"):
        K2.patch_camblind(CB, CB.CENSUS)                                                           # no bw_frames at all: refused too (the T1 K2 would die at window 0)


# =========================================================================== #
# through main(): the launcher installs the extension for the roll and removes it after      #
# =========================================================================== #
_FAKE_SCRIPT = '''
import json, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_roll
import torch
from tanitad.eval import camblind_r9 as CB
import test_r9e2_k2safe as T
rec = {"camera_keys": list(CB.camera_keys("item")), "sha": CB.census_sha256()}
def script_donor(ds, j):          # camblind_roll_r9.py:121 -- must NOT be used
    raise AssertionError("the script's donor_item_fn was used")
cls = CB.make_camblind_dataset_cls(T._T1Eval, camera="derange", dig={}, drange={}, own_every=25, donor_item_fn=script_donor)
ds = cls()
base = T._T1Eval()
rec["leaks"] = sum(1 for i in range(len(base.index)) if torch.equal(ds[i]["bw_frames"], base[i]["bw_frames"]))
rec["n"] = len(base.index)
ctx = types.SimpleNamespace(item={"pose_hist": torch.zeros(4, 3)}, index=(0, 1))
r7_roll.R9E2_STAGER.stage(types.SimpleNamespace(), ctx, torch.zeros(2, 3))
Path(sys.argv[0]).with_suffix(".json").write_text(json.dumps(rec))
sys.exit(0)
'''


def _fake_tree(tmp_path):
    (tmp_path / "r7_roll.py").write_text(
        "import types\nclass Refcv6Proxy:\n    def __init__(self, model, ctx, *a, **k):\n        self.__dict__['model'] = model\n"
        "L = types.SimpleNamespace(build_eval_dataset=lambda *a, **k: (types.SimpleNamespace(episodes=[]), None, {}))\n", encoding="utf-8")
    (tmp_path / "fake_camblind.py").write_text(_FAKE_SCRIPT, encoding="utf-8")
    return tmp_path / "fake_camblind.py"


def test_main_runs_the_script_under_the_extended_census_and_removes_it_afterwards(monkeypatch, tmp_path):
    import json
    script = _fake_tree(tmp_path)
    for k in ("R8_LAUNCH_TREE", "PF_R8_CODE", "R8_EVAL_KIT", "L1_PF_CODE", "L1_R8_CODE"):
        monkeypatch.setenv(k, str(tmp_path / k))
    monkeypatch.setitem(sys.modules, "test_r9e2_k2safe", sys.modules[__name__])
    stock_keys, stock_sha = CB.camera_keys("item"), CB.census_sha256()
    monkeypatch.setattr(sys, "argv", ["x", "--camblind-script", str(script)])
    with pytest.raises(SystemExit) as e:
        K2.main()
    assert e.value.code in (0, None)
    rec = json.loads(script.with_suffix(".json").read_text())
    assert rec["camera_keys"] == list(stock_keys) + ["bw_frames"] and rec["sha"] != stock_sha          # the SCRIPT saw the extended census
    assert rec["leaks"] == 0 and rec["n"] > 0                                                          # ... and its call (with the script's own donor_item_fn) swapped bw_frames
    assert CB.camera_keys("item") == stock_keys and CB.census_sha256() == stock_sha                    # undone after the roll


def test_RED_main_refuses_when_tanitad_cannot_be_imported(monkeypatch, tmp_path):
    script = _fake_tree(tmp_path)
    for k in ("R8_LAUNCH_TREE", "PF_R8_CODE", "R8_EVAL_KIT", "L1_PF_CODE", "L1_R8_CODE"):
        monkeypatch.setenv(k, str(tmp_path / k))
    import tanitad.eval as _ev
    monkeypatch.delattr(_ev, "camblind_r9")                                                            # `from tanitad.eval import camblind_r9` checks the package attribute first
    monkeypatch.setitem(sys.modules, "tanitad.eval.camblind_r9", None)                                 # ... then imports the submodule: ImportError
    monkeypatch.setattr(sys, "argv", ["x", "--camblind-script", str(script)])
    with pytest.raises(SystemExit, match="cannot import tanitad.eval.camblind_r9"):
        K2.main()
