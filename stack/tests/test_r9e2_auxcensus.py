"""E2-AUXCENSUS -- the camera-blind census (W4) and the C4 box-aux TARGET keys (``p3_*`` = B2, ``b2_*`` = B2c).

``V3Dataset.__getitem__`` adds them (refc_v3_train.py:4902-4904, ``persp3d_aux_flag.item_targets``) iff ``ds._r9_p3`` is set.  ``camblind_r9.CENSUS`` is FAIL-CLOSED and does not know
them, so a dataset that carries them is REFUSED at window 0.  Classification (from source, HANDOFF.md): TRAINING-ONLY targets built from the GT cuboids + the rig camera / BEV grid,
never from ``item["frames"]``, read only by ``loss_term`` (``model.training``) => LABEL (the two ``*_build_s`` wall-clock telemetry keys: META), passed through UNTOUCHED.

REACH (honest scope): the stock roll's eval dataset is NOT given ``_r9_p3`` today (only ``_bw.attach_dataset`` reaches ``e_ds``) -- the extension is defence in depth, pinned below.

Everything imports the REAL modules (``camblind_r9``, ``persp3d_aux_flag.item_targets``, the launcher); the only fakes are the dataset and its episodes.  RED arms:
  * the STOCK instrument refuses a B2 item and a B2c item at window 0;
  * a WRONG classification (an aux key classed CAMERA) is refused by ``patch_camblind`` (the donor fn knows only ``bw_frames``), so it can never silently swap / leak;
  * the aux targets are a function of the cuboids ONLY: perturbing the frames moves them by EXACTLY 0, perturbing a cuboid moves them (the control that proves the probe works);
  * control (c): with the keys admitted as plain keys the digest of the non-camera inputs differs run to run (wall-clock ``*_build_s``) -- the patch drops exactly those two keys.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests"), str(ROOT / "ops" / "r9e2_read")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.bev_raster import GRID_DEFAULT  # noqa: E402
from tanitad.data.calib import CanonicalFrame  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.eval import camblind_r9 as CB  # noqa: E402
from tanitad.train import persp3d_aux_flag as F  # noqa: E402

import camblind_roll_r9_e2 as K2  # noqa: E402

STACK = Path(CB.__file__).resolve().parents[2]           # <tree>/stack, whichever tree supplies tanitad
A = 4


def _cam():
    return RigCamera.nominal(CanonicalFrame.from_hfov(120.0, 416, 1024, "cylindrical"), height_m=1.5, x_m=0.0)


def _src(kind):
    return F.TargetSource("persp", cam=_cam(), eq=0) if kind == "persp" else F.TargetSource("bev2", grid=GRID_DEFAULT)


def _agents(e, x=20.0, y=3.0):
    """The GT agent block of one item (what V3Dataset.__getitem__ holds when item_targets runs)."""
    box = torch.zeros(A, 4)
    box[0] = torch.tensor([x + e, y, 4.5, 1.9])
    valid = torch.zeros(A, dtype=torch.bool)
    valid[0] = True
    return {"agent_ep": torch.tensor(1000 + e, dtype=torch.long), "agent_box": box, "agent_yaw": torch.zeros(A), "agent_valid": valid,
            "agent_cz": torch.full((A,), 0.75), "agent_h": torch.full((A,), 1.5), "agent_zh_mask": valid.clone(), "agent_label": torch.tensor(True)}


class _Ep:
    def __init__(self, e, T=12):
        g = torch.Generator().manual_seed(7 + e)
        self.frames = torch.randint(1, 256, (T, 9, 4, 4), generator=g, dtype=torch.uint8)
        self.actions, self.poses = torch.randn(T, 2, generator=g), torch.randn(T, 4, generator=g)
        self.episode_id = 1000 + e


class _AuxEval:
    """r7_roll's Refcv6EvalWindows reduced to the window dict + the V3Dataset tail INCLUDING the C4 hook (refc_v3_train.py:4902-4904) calling the REAL ``item_targets``."""
    kind = "persp"

    def __init__(self):
        self.episodes = [_Ep(e) for e in range(3)]
        self.window, self.max_horizon = 3, 2
        self.index = [(e, t) for e in range(3) for t in range(0, 6, 2)]
        self._r9_p3 = _src(self.kind)

    def _window_u8(self, i):
        e_i, t = self.index[i]
        ep, w = self.episodes[e_i], self.window
        return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w], "future_frames": torch.zeros(0, dtype=torch.uint8),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon], "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}

    def __getitem__(self, i):
        item = self._window_u8(i)
        item.update(_agents(self.index[i][0]))
        item.update(F.item_targets(item, self._r9_p3))
        return item


class _Bev2Eval(_AuxEval):
    kind = "bev2"


@pytest.fixture
def patched():
    census, undo = K2.patch_camblind(CB)
    try:
        yield census
    finally:
        undo()


def _real_keys(kind):
    return set(F.item_targets(_agents(0), _src(kind)))


# =========================================================================== #
# census rows                                                                  #
# =========================================================================== #
def test_the_aux_keys_are_exactly_what_item_targets_emits():
    assert _real_keys("persp") == set(K2.P3_KEYS) and len(K2.P3_KEYS) == 14
    assert _real_keys("bev2") == set(K2.B2_KEYS) and len(K2.B2_KEYS) == 7
    assert set(F.BATCH_KEYS_P + F.CENSUS_KEYS_P) == set(K2.P3_KEYS) and set(F.BATCH_KEYS_B + F.CENSUS_KEYS_B) == set(K2.B2_KEYS)


def test_the_rows_are_label_or_meta_never_camera_and_every_pin_resolves():
    ent = K2.aux_entries(CB)
    assert [e.name for e in ent] == list(K2.AUX_KEYS) and len(ent) == 21
    for e in ent:
        assert e.kind == "item" and not e.pixel_derived and not e.reaches
        assert e.cls == (CB.META if e.name in K2.AUX_TELEMETRY else CB.LABEL), e.name
    res = {e.name: (loc, n) for e, loc, n in CB.verify_census(STACK, ent)}
    assert all(n >= 1 and loc.startswith(K2.AUX_SRC + ":") for loc, n in res.values()), res
    assert res["p3_heat8"][0] == res["p3_own16"][0] and res["b2_ign"][0] != res["b2_heat"][0]
    stale = dataclasses.replace(ent[0], needle='out["p3_no_such_key"] =')                       # RED: a stale pin reads 0
    assert CB.verify_census(STACK, (stale,))[0][2] == 0


def test_auxcensus_is_k2safe_plus_the_aux_rows_and_k2safe_census_is_unchanged():
    k2s, ext = K2.k2safe_census(CB), K2.auxcensus(CB)
    assert k2s[:len(CB.CENSUS)] == CB.CENSUS and [e.name for e in k2s[len(CB.CENSUS):]] == list(K2.BW_KEYS)
    assert ext[:len(k2s)] == k2s and [e.name for e in ext[len(k2s):]] == list(K2.AUX_KEYS) and K2.auxcensus(CB) == ext
    assert CB.camera_keys("item", ext) == ("frames", "future_frames", "bw_frames")             # no aux key is a camera key
    CB._check_census(ext)


# =========================================================================== #
# RED: the stock instrument, a wrong classification                            #
# =========================================================================== #
@pytest.mark.parametrize("cls,keys", [(_AuxEval, K2.P3_KEYS), (_Bev2Eval, K2.B2_KEYS)])
def test_RED_the_stock_instrument_refuses_a_C4_item_at_window_0(cls, keys):
    ds = CB.make_camblind_dataset_cls(cls, camera="derange", own_every=1)()
    with pytest.raises(CB.UnclassifiedKey) as e:
        ds[0]
    assert all(k in str(e.value) for k in keys)                                                 # it names every aux key, not just the first
    ident = CB.make_camblind_dataset_cls(cls, camera="identity")()
    with pytest.raises(CB.UnclassifiedKey):
        ident[0]


@pytest.mark.parametrize("bad", ["p3_heat8", "b2_heat", "p3_n_boxes"])
def test_RED_a_wrong_classification_aux_key_as_CAMERA_is_refused_by_the_patch(bad):
    ext = tuple(dataclasses.replace(e, cls=CB.CAMERA, pixel_derived=True, reaches=True) if e.name == bad else e for e in K2.auxcensus(CB))
    with pytest.raises(SystemExit, match=bad):
        K2.patch_camblind(CB, census=ext)
    assert CB.camera_keys("item") == ("frames", "future_frames")                                # nothing was rebound


# =========================================================================== #
# GREEN: passes through untouched, zero leak, pixel-independence               #
# =========================================================================== #
@pytest.mark.parametrize("cls,keys", [(_AuxEval, K2.P3_KEYS), (_Bev2Eval, K2.B2_KEYS)])
def test_GREEN_the_patched_blind_roll_runs_swaps_only_the_pixels_and_keeps_every_aux_key_the_windows_own(patched, cls, keys):
    dig, dr = {}, {}
    ds = CB.make_camblind_dataset_cls(cls, camera="derange", dig=dig, drange=dr, own_every=1)()
    base = cls()
    n_frames_swapped = 0
    for i in range(len(base.index)):
        got, own = ds[i], base[i]
        for k in keys:
            if k in K2.AUX_TELEMETRY:
                assert k in got
            else:
                assert torch.equal(got[k], own[k]), (i, k)                                       # OWN target, untouched
        assert not torch.equal(got["frames"], own["frames"])                                     # the camera IS swapped
        n_frames_swapped += 1
    assert n_frames_swapped == len(base.index) and dr["fixed_points"] == 0
    assert set(dr["camera_keys_swapped"]) == {"frames", "future_frames", "bw_frames"}


def test_the_aux_targets_are_a_function_of_the_cuboids_only_frames_move_them_by_exactly_zero():
    items = []
    for kind in ("persp", "bev2"):
        a = _agents(1)
        a["frames"] = torch.randint(0, 256, (3, 9, 4, 4), dtype=torch.uint8)
        b = dict(a, frames=torch.randint(0, 256, (3, 9, 4, 4), dtype=torch.uint8))               # DIFFERENT pixels, same cuboids
        c = _agents(1, x=40.0, y=-6.0)                                                           # SAME pixels, a MOVED cuboid (the control)
        c["frames"] = a["frames"]
        ra, rb, rc = (F.item_targets(x, _src(kind)) for x in (a, b, c))
        moved = 0
        for k in ra:
            if k in K2.AUX_TELEMETRY:
                continue
            assert torch.equal(ra[k], rb[k]), (kind, k)                                          # pixels: exactly 0.0
            moved += int(not torch.equal(ra[k], rc[k]))
        assert moved > 0, kind                                                                   # the probe CAN read a change: a cuboid moves the targets
        items.append(moved)
    assert all(m >= 1 for m in items)


# =========================================================================== #
# control (c): the digest and the telemetry keys                               #
# =========================================================================== #
def _it(build_s, n_boxes=1):
    d = {"actions": torch.zeros(2), "agent_box": torch.ones(2, 4), "p3_n_boxes": torch.tensor(n_boxes), "p3_build_s": torch.tensor(build_s, dtype=torch.float64),
         "b2_build_s": torch.tensor(build_s, dtype=torch.float64)}
    return d


def test_RED_control_c_stock_digest_differs_on_wall_clock_telemetry_and_the_patch_drops_exactly_those_keys(patched):
    nav, v0, es = torch.tensor([1]), torch.tensor([3.0]), torch.zeros(1, 5)
    stock = CB.other_inputs_digest.__wrapped__
    a, b = _it(0.01), _it(0.93)
    assert stock(a, nav, v0, es, census=patched) != stock(b, nav, v0, es, census=patched)       # RED: without the drop, control (c) would fail on timing alone
    assert CB.other_inputs_digest(a, nav, v0, es) == CB.other_inputs_digest(b, nav, v0, es)      # patched: equal
    assert CB.other_inputs_digest(a, nav, v0, es) != CB.other_inputs_digest(_it(0.01, n_boxes=2), nav, v0, es)   # NOT over-broad: a real target change still moves it
    assert CB.other_inputs_digest(a, nav, v0, es, census=patched) == CB.other_inputs_digest(b, nav, v0, es, census=patched)
    clean = {k: v for k, v in a.items() if k not in K2.AUX_TELEMETRY}
    assert CB.other_inputs_digest(clean, nav, v0, es) == stock(clean, nav, v0, es, census=patched)   # no telemetry key: byte-identical to the stock digest
    assert a["p3_build_s"].item() == 0.01 and "p3_build_s" in a                                  # the caller's item is not mutated


def test_the_patch_undoes_and_the_stock_census_is_never_edited(patched):
    sha_patched = CB.census_sha256()
    assert sha_patched != CB.census_sha256(CB.CENSUS) and "p3_heat8" not in {e.name for e in CB.CENSUS}
    assert "p3_heat8" in {e.name for e in patched} and len(patched) == len(CB.CENSUS) + 4 + 21


def test_undo_restores_the_four_entry_points_and_the_digest():
    stock = {n: getattr(CB, n) for n in K2.PATCHED}
    sha0 = CB.census_sha256()
    _c, undo = K2.patch_camblind(CB)
    undo()
    undo()
    assert all(getattr(CB, n) is stock[n] for n in K2.PATCHED) and CB.census_sha256() == sha0


# =========================================================================== #
# reach pin: why this is defence in depth today                                #
# =========================================================================== #
def _src_text(rel):
    return (STACK / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_reach_pin_no_eval_dataset_path_attaches_the_C4_targets_today():
    """If this goes RED an eval path now attaches ``_r9_p3``: the extension is LIVE, re-read the classification (HANDOFF.md D1) before a B2 / B2c K2 is rolled."""
    tr = _src_text("scripts/refc_v3_train.py")
    assert tr.count("_p3f.attach_dataset(") == 1 and "_p3f.attach_dataset(ds, model, args)" in tr and "_p3f.attach_dataset(e_ds" not in tr
    assert "_bw.attach_dataset(e_ds, model, args)" in tr                                         # CONTROL: the probe finds the one aux attach the eval ds DOES get
    for rel in ("tanitad/eval/refcv6_loader.py", "tanitad/eval/refcv7_loader.py"):
        txt = _src_text(rel)
        assert "_r9_p3" not in txt and "_p3f.attach_dataset" not in txt and "persp3d_aux_flag.attach_dataset" not in txt, rel   # (the HEAD attach `tr._p3f.attach(model, args)` is build-time and legitimate)
    assert "tr._bw.attach_dataset(e_ds, model, args)" in _src_text("tanitad/eval/refcv7_loader.py")   # CONTROL (same breath)
