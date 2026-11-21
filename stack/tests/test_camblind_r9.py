"""refcv9 C9 -- the camera-blind instrument (W4) swaps ALL camera-derived inputs, proven by a SENSITIVITY IDENTITY.

REFCV9_DESIGN 2.7 / 5.2 W4 / 4.2 C9.  Four layers, each with RED arms:

  A. the CENSUS: every dataset-item key, every model-forward input and every path-valued argv flag (r9e1_*.argv.json + the refcv8 full argv) is
     classified CAMERA / EGO / NAV / LABEL / CALIB / META with a source pin that resolves in the tree; the census is checked against the LIVE
     forward signature and the live dataset sources (a new key breaks the test until it is classified).
  B. the INSTRUMENT: ``swap_item`` (every CAMERA key from the donor, everything else the window's OWN object; fail-closed on unknown keys; refuses to
     swap an ego / nav / label / calibration key), the derangement (verbatim from anticheat_common, pinned), controls (a)-(c), the roll-side class.
  C. the IDENTITY on a tiny REAL ``RefCV3Model`` built WITH C8's wires on (banded map head + critic + map channels + box tokens): perturb the window's OWN
     key (pixels, a synthetic offline sidecar, every label ...) and measure the change of every planner input and of the planner output.
        correct instrument -> EXACTLY 0.0 for every pixel-derived key;  mutation "swap pixels only" -> > 0 wherever an offline camera tensor is read;
        control: perturbing the DONOR's camera moves the planner.
  D. the adapter for the anticheat battery compiles and routes through this module.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.eval import camblind_r9 as CB  # noqa: E402

AC_ORIG = Path("C:/Users/Admin/qland/hold_anticheat_r8/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat/code/anticheat_common.py")
AC_ORIG_SHA256 = "bf084c919b4bd28928c47c95860be10070cb4e5311091ee33bc404b44e0678b5"
DERANGE_PINS = {(100, 0): "f21aa2de0600814cba88ca4036f8c95c9757dacdf7d933d4ace872c40543e23b", (40, 0): "a4b6a8cb9ba624acb0fee9b8ddaf5d41e3681eed492a8eb4605da7f6bf057324",
                (40, 1): "55e42678ca6c40da26a6fe19728630c4de74335ec05355cd041ae49f2d508b60", (10, 0): "f57dfe5de20005aa651e5bf57752a9335f97770fc29538627bb7a69b3d1c9bb7"}
ADAPTER = ROOT.parent / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-11-03-refcv9-camblind/code/camblind_roll_r9.py"


def _src(rel):
    return (ROOT / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


# =========================================================================== #
# A. the census                                                                #
# =========================================================================== #
def test_the_census_is_well_formed_and_every_source_pin_resolves():
    CB._check_census()
    res = CB.verify_census(ROOT)
    stale = [(e.kind, e.name, e.src, e.needle) for e, loc, n in res if n == 0]
    assert not stale, "stale census pins: %s" % stale[:5]
    print("\n[census file:line]\n" + "\n".join("  %-12s %-24s %-7s %s" % (e.kind, e.name, e.cls, loc) for e, loc, n in res))
    # RED: a pin whose needle is gone reads n == 0
    e0 = CB.CENSUS[0]
    bad = CB.Entry(e0.name, e0.kind, e0.cls, e0.pixel_derived, e0.reaches, e0.src, "THIS TEXT IS NOT IN THE SOURCE", e0.after, "")
    assert CB.locate(bad, ROOT) == (0, 0)


def test_the_camera_set_and_the_offline_pixel_derived_set_are_exactly_what_the_census_says():
    assert CB.camera_keys("item") == ("frames", "future_frames")
    assert CB.camera_keys("model_input") == ("frames",)
    assert CB.camera_keys("argv") == ("--v2-cache", "--eval-cache")
    px = {(e.kind, e.name): e.cls for e in CB.CENSUS if e.pixel_derived}
    # the ONLY pixel-derived things in the census: the pixel store + frames (CAMERA, swapped) and the SAM3 map targets (LABEL: offline, pixel-derived, a TARGET)
    assert {k for k, c in px.items() if c != CB.CAMERA} == {("item", n) for n in ("map_frac", "map_seen", "map_label", "map_raw_frame", "map_fine", "map_fine_label")} | {("argv", "--map-gt-root")}
    assert all(c == CB.LABEL for k, c in px.items() if c != CB.CAMERA)
    # THE CENSUS CONCLUSION: no offline camera tensor reaches the roll's forward except through the swapped pixel store
    leaks = [e.name for e in CB.CENSUS if e.pixel_derived and e.reaches and e.cls != CB.CAMERA]
    assert leaks == [], leaks
    # and every LABEL, by the census, does not reach the forward
    assert all(not e.reaches for e in CB.CENSUS if e.cls == CB.LABEL)
    by = {}
    for e in CB.CENSUS:
        by.setdefault((e.kind, e.cls), []).append(e.name)
    print("\n[census counts] " + ", ".join("%s/%s=%d" % (k, c, len(v)) for (k, c), v in sorted(by.items())))


def test_the_model_input_census_equals_the_live_forward_signature():
    pytest.importorskip("timm")
    from tanitad.refs import refc_v3 as v3
    live = [p for p in inspect.signature(v3.RefCV3Model.forward).parameters if p != "self"]
    assert live == [e.name for e in CB.CENSUS if e.kind == "model_input"], "a forward kwarg was added / removed: classify it in CB.CENSUS"
    # RED: an unknown kwarg is refused by classify()
    with pytest.raises(CB.UnclassifiedKey):
        CB.classify("precomputed_bev", "model_input")


def _source_item_keys():
    keys = set()

    def scan(rel, classes):
        for cn in ast.walk(ast.parse(_src(rel))):
            if not (isinstance(cn, ast.ClassDef) and cn.name in classes):
                continue
            for fn in [n for n in ast.walk(cn) if isinstance(n, ast.FunctionDef)]:
                for n in ast.walk(fn):
                    if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id == "item" \
                            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                        keys.add(n.slice.value)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "item" for t in n.targets) and isinstance(n.value, ast.Dict):
                        keys.update(k.value for k in n.value.keys if isinstance(k, ast.Constant))
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict) and fn.name in ("_window_u8", "_agent_future_item", "_map_item", "_map_fine_item", "_vis1_item"):
                        keys.update(k.value for k in n.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    scan("scripts/refb_train.py", {"FailLoudWindowDataset"})
    scan("scripts/refc_train.py", {"RouteV21Dataset"})
    scan("scripts/refc_v3_train.py", {"V3Dataset"})
    return keys


def test_every_dataset_item_key_in_the_live_sources_is_classified():
    src_keys = _source_item_keys()
    assert len(src_keys) > 45, len(src_keys)          # the AST walk is not vacuous (55 on the tip + C8 + C3)
    print("\n[dataset-item keys found in the live sources] %d, all classified" % len(src_keys))
    missing = sorted(k for k in src_keys if k not in CB.entries("item"))
    assert not missing, "dataset-item keys the census does not know (each could be an offline camera tensor): %s" % missing


def _path_valued_flags(argv):
    out, cur = {}, None
    for t in argv:
        if t.startswith("--"):
            cur = t
            out[cur] = []
        elif cur is not None:
            out[cur].append(t)
    return {f: v for f, v in out.items() if any(("/" in s or s.endswith((".json", ".npz", ".pt", ".jsonl", ".xz", ".gz"))) and not s.replace(".", "", 1).isdigit() for s in v)}


def test_every_path_valued_flag_of_the_refcv9_and_refcv8_argvs_is_classified():
    files = sorted((ROOT / "ops/runs.d").glob("r9e1_*.argv.json")) + [ROOT / "ops/runs.d/refcv8-r101-s0.argv.json"]
    assert len(files) >= 6
    seen = {}
    for f in files:
        for flag in _path_valued_flags(json.loads(f.read_text(encoding="utf-8"))["argv"]):
            seen.setdefault(flag, []).append(f.name)
    unclassified = sorted(f for f in seen if f not in CB.entries("argv") and f != "--out")        # --out is the OUTPUT directory
    assert not unclassified, unclassified
    print("\n[path-valued argv flags -> class]\n" + "\n".join("  %-30s %-7s %d argvs" % (f, CB.classify(f, "argv").cls, len(v)) for f, v in sorted(seen.items()) if f != "--out"))
    # the only camera-class flags are the pixel stores; the flags the brief names:
    assert CB.classify("--vis1-sidecar", "argv").cls == CB.LABEL and not CB.classify("--vis1-sidecar", "argv").pixel_derived
    assert CB.classify("--v2-cache", "argv").cls == CB.CAMERA and CB.classify("--eval-cache", "argv").cls == CB.CAMERA
    with pytest.raises(CB.UnclassifiedKey):
        CB.classify("--bev-feature-cache", "argv")


# =========================================================================== #
# B. the instrument                                                            #
# =========================================================================== #
def _item(seed):
    g = torch.Generator().manual_seed(seed)
    return {"frames": torch.randint(0, 256, (3, 9, 4, 4), generator=g, dtype=torch.uint8), "future_frames": torch.zeros(0, dtype=torch.uint8),
            "actions": torch.randn(3, 2, generator=g), "pose_last": torch.randn(4, generator=g), "pose_hist": torch.randn(3, 4, generator=g),
            "nav_cmd": torch.tensor(seed % 3), "nav_valid": torch.tensor(True), "future_poses_ext": torch.randn(6, 4, generator=g),
            "map_fine": torch.randint(0, 8, (5, 5), generator=g, dtype=torch.uint8), "episode_id": seed}


def test_swap_item_takes_every_camera_key_from_the_donor_and_leaves_everything_else_the_windows_own_object():
    own, donor = _item(1), _item(2)
    fed = CB.swap_item(own, donor)
    assert fed["frames"] is donor["frames"] and fed["future_frames"] is donor["future_frames"]
    for k in own:
        if k not in CB.camera_keys():
            assert fed[k] is own[k], k
    assert own["frames"] is not donor["frames"] and not torch.equal(own["frames"], donor["frames"])


def test_swap_item_is_fail_closed_and_refuses_to_swap_ego_nav_label_calibration():
    own, donor = _item(1), _item(2)
    with pytest.raises(CB.UnclassifiedKey, match="precomputed_bev"):
        CB.swap_item({**own, "precomputed_bev": torch.zeros(2)}, donor)
    for k in ("pose_last", "pose_hist", "actions", "nav_cmd", "future_poses_ext", "map_fine", "episode_id"):       # EGO, EGO, EGO, NAV, LABEL, LABEL(pixel-derived), META
        with pytest.raises(CB.SwapRefused, match=k):
            CB.swap_item(own, donor, swap_keys=("frames", k))
    d2 = dict(donor)
    del d2["frames"]
    with pytest.raises(KeyError):
        CB.swap_item(own, d2)


def test_the_derangement_and_donor_rule_are_the_W4_ones_bit_for_bit():
    for (n, seed), h in DERANGE_PINS.items():
        pi = CB.sattolo_derangement(range(n), seed)
        assert hashlib.sha256("".join(f"{e}:{d}," for e, d in sorted(pi.items())).encode()).hexdigest() == h, (n, seed)
        assert all(pi[e] != e for e in pi) and sorted(pi.values()) == list(range(n))
    if AC_ORIG.exists():                                       # the original, when the hold is reachable
        assert hashlib.sha256(AC_ORIG.read_bytes()).hexdigest() == AC_ORIG_SHA256
        spec = importlib.util.spec_from_file_location("ac_orig", AC_ORIG)
        ac = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ac)
        idx = [(e, t) for e in range(6) for t in range(0, 3 * (e + 1), 3)]
        pi = ac.sattolo_derangement(range(6), 0)
        assert CB.sattolo_derangement(range(6), 0) == pi and CB.donor_windows(idx, pi) == ac.donor_windows(idx, pi)
        assert CB.digest_bytes(b"abc") == ac.digest_bytes(b"abc")
    with pytest.raises(ValueError):
        CB.sattolo_derangement([3], 0)


def test_controls_a_b_c_carried_over_pass_on_the_right_inputs_and_go_RED_on_the_wrong_ones():
    rng = np.random.default_rng(0)
    base = {"e_i": np.arange(5), "t": np.arange(5) * 3, "traj_own": rng.normal(size=(5, 4, 2)).astype(np.float32), "z_tac": rng.normal(size=(5, 7)).astype(np.float32),
            "sel_own": np.arange(5)}
    new = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    assert CB.check_bank_equal(base, new)["PASS"]                               # (a) identity roll reproduces the base bank bit for bit
    new["traj_own"][2, 1, 0] = np.nextafter(new["traj_own"][2, 1, 0], np.float32(9.0))
    assert not CB.check_bank_equal(base, new)["PASS"]                           # RED: ONE ULP off -> the identity roll is not bit-equal
    new2 = {**base, "t": base["t"] + 1}
    assert not CB.check_bank_equal(base, new2)["PASS"]                          # RED: other windows
    fed, ref = np.arange(1, 6, dtype=np.uint64), np.arange(11, 16, dtype=np.uint64)
    assert CB.check_changed_fraction(fed, ref)["PASS"]                          # (b)
    ref2 = ref.copy()
    ref2[3] = fed[3]
    assert not CB.check_changed_fraction(fed, ref2)["PASS"] and not CB.check_changed_fraction(np.zeros(5, np.uint64), ref)["PASS"]
    assert CB.check_ego_equal(fed, fed.copy())["PASS"] and not CB.check_ego_equal(fed, ref)["PASS"]       # (c)
    assert not CB.check_ego_equal(np.zeros(3, np.uint64), np.zeros(3, np.uint64))["PASS"]


def _original_other_inputs_digest(item, nav_cmd, v0, ego_state):
    """camblind_roll._other_inputs_digest, verbatim."""
    h = hashlib.blake2b(digest_size=8)
    for k in sorted(item):
        if k in ("frames", "future_frames"):
            continue
        v = item[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(str(tuple(v.shape)).encode() + str(v.dtype).encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy()).data)
        elif isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).data)
        else:
            h.update(repr(v).encode())
    for nm, v in (("nav_cmd", nav_cmd), ("v0", v0), ("ego_state", ego_state)):
        h.update(nm.encode())
        if v is None:
            h.update(b"None")
        elif torch.is_tensor(v):
            h.update(np.ascontiguousarray(v.detach().cpu().numpy()).data)
        else:
            h.update(repr(v).encode())
    return int.from_bytes(h.digest(), "little")


def test_control_c_digest_is_byte_identical_to_the_W4_one_and_excludes_exactly_the_camera_keys():
    it = _item(5)
    nav, v0, es = torch.tensor([1]), torch.tensor([3.5]), torch.randn(1, 5)
    assert CB.other_inputs_digest(it, nav, v0, es) == _original_other_inputs_digest(it, nav, v0, es)
    assert CB.other_inputs_digest({**it, "frames": torch.zeros(1)}, nav, v0, es) == CB.other_inputs_digest(it, nav, v0, es)         # camera keys excluded
    assert CB.other_inputs_digest({**it, "pose_last": it["pose_last"] + 1e-6}, nav, v0, es) != CB.other_inputs_digest(it, nav, v0, es)  # an ego key is in
    # the generalisation: a census CAMERA key beyond the pixels is excluded too (it is swapped, so it must not be 'equal' evidence)
    ext = CB.CENSUS + (CB.Entry("bev_sidecar", "item", CB.CAMERA, True, True, "synthetic", "", "", ""),)
    a = {**it, "bev_sidecar": torch.zeros(2)}
    b = {**it, "bev_sidecar": torch.ones(2)}
    assert CB.other_inputs_digest(a, nav, v0, es, census=ext) == CB.other_inputs_digest(b, nav, v0, es, census=ext)
    assert CB.other_inputs_digest(a, nav, v0, es) != CB.other_inputs_digest(b, nav, v0, es)         # under the BASE census it is a non-camera (digested) key


class _Ep:
    def __init__(self, e, T=12):
        g = torch.Generator().manual_seed(100 + e)
        self.frames = torch.randint(0, 256, (T, 9, 4, 4), generator=g, dtype=torch.uint8)
        self.actions, self.poses = torch.randn(T, 2, generator=g), torch.randn(T, 4, generator=g)
        self.episode_id = 1000 + e


class _FakeEval:
    """r7_roll.make_dataset_cls's Refcv6EvalWindows reduced to what the instrument touches (window dict + a V3Dataset-style label tail)."""
    extra_item = None

    def __init__(self, n_eps=4, window=3, max_horizon=2):
        self.episodes = [_Ep(e) for e in range(n_eps)]
        self.window, self.max_horizon = window, max_horizon
        self.index = [(e, t) for e, ep in enumerate(self.episodes) for t in range(0, ep.frames.shape[0] - window - max_horizon, 2)]

    def _window_u8(self, i):
        e_i, t = self.index[i]
        ep, w = self.episodes[e_i], self.window
        return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w], "future_frames": torch.zeros(0, dtype=torch.uint8),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon], "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}

    def __getitem__(self, i):
        item = self._window_u8(i)
        item["nav_cmd"] = torch.tensor(self.index[i][0] % 3)
        item["map_fine"] = torch.full((2, 2), self.index[i][0], dtype=torch.uint8)
        if self.extra_item:
            item.update(self.extra_item(self, i))
        return item


def test_the_roll_class_identity_is_the_batterys_own_window_and_derange_swaps_exactly_the_camera_keys():
    base = _FakeEval()
    dig_i, dig_d, dr = {}, {}, {}
    Ident = CB.make_camblind_dataset_cls(_FakeEval, camera="identity", dig=dig_i)
    Blind = CB.make_camblind_dataset_cls(_FakeEval, camera="derange", dig=dig_d, drange=dr, own_every=1)
    di, db = Ident(), Blind()
    for i in range(len(base.index)):
        a, b, o = di[i], db[i], base[i]
        assert all(torch.equal(a[k], o[k]) for k in o if torch.is_tensor(o[k])) and a.keys() == o.keys()                 # control (a): identity == the battery's own item
        e_i, t = base.index[i]
        e_d, t_d = dig_d[(e_i, t)]["donor"]
        assert e_d != e_i
        assert torch.equal(b["frames"], base.episodes[e_d].frames[t_d:t_d + base.window])                                 # frames = the DONOR window's
        assert not torch.equal(b["frames"], o["frames"])
        for k in o:                                                                                                       # EVERY other key = the window's own
            if k not in CB.camera_keys():
                assert torch.equal(b[k], o[k]) if torch.is_tensor(o[k]) else b[k] == o[k], k
        assert CB.check_changed_fraction([dig_d[(e_i, t)]["fed"]], [dig_i[(e_i, t)]["fed"]])["PASS"]
        assert dig_d[(e_i, t)]["own"] == dig_i[(e_i, t)]["fed"]                                                           # control (b): own digest self-contained
    assert dr["fixed_points"] == 0 and dr["each_episode_a_donor_once"] and dr["camera_keys_swapped"] == ["frames", "future_frames"]
    assert dr["mapping_sha256"] == hashlib.sha256("".join(f"{e}:{d}," for e, d in sorted(CB.sattolo_derangement(range(4), 0).items())).encode()).hexdigest()


def test_the_roll_class_refuses_an_unclassified_key_and_a_declared_camera_key_it_cannot_swap():
    class Leaky(_FakeEval):
        extra_item = staticmethod(lambda self, i: {"bev_cache": torch.zeros(2)})
    with pytest.raises(CB.UnclassifiedKey, match="bev_cache"):
        CB.make_camblind_dataset_cls(Leaky, camera="derange")()[0]
    with pytest.raises(CB.UnclassifiedKey, match="bev_cache"):
        CB.make_camblind_dataset_cls(Leaky, camera="identity")()[0]               # identity refuses too: the same fail-closed gate
    ext = CB.CENSUS + (CB.Entry("bev_cache", "item", CB.CAMERA, True, True, "synthetic", "", "", ""),)
    with pytest.raises(CB.CensusGap, match="bev_cache"):
        CB.make_camblind_dataset_cls(Leaky, camera="derange", census=ext)         # a declared CAMERA key with no way to swap it: refuse to build
    Blind = CB.make_camblind_dataset_cls(Leaky, camera="derange", census=ext, drange={}, donor_item_fn=lambda ds, j: {"bev_cache": torch.full((2,), float(j))})
    ds = Blind()
    for i in range(len(ds.index)):
        item = ds[i]                                                    # first: builds the derangement map
        e_i, t = ds.index[i]
        e_d, t_d = type(ds)._dw[(e_i, t)]
        assert torch.equal(item["bev_cache"], torch.full((2,), float(type(ds)._row[(e_d, t_d)])))      # the extra camera tensor comes from the DONOR row
        assert torch.equal(item["frames"], ds.episodes[e_d].frames[t_d:t_d + ds.window])                # and the pixels are still the donor window's, not the donor's donor


# =========================================================================== #
# C. the identity on a tiny real model with C8's wires on                      #
# =========================================================================== #
SIDECARS = ("bev_sidecar", "box_sidecar")
SYN = CB.CENSUS + tuple(CB.Entry(n, "item", CB.CAMERA, True, True, "synthetic", "", "", "synthetic offline camera tensor (a precomputed BEV feature / box-score cache)") for n in SIDECARS)
TAPS = {"trunk", "map_chan_1m", "box_slots", "bev_pool", "box_tokens", "critic_map", "plan", "out_all"}


@contextlib.contextmanager
def _ports(model, batch):
    """TEST FIXTURE: stands in for a precomputed camera-derived tensor that reaches the forward from OUTSIDE the pixels -- a BEV-feature cache added to the
    planner BEV pool, and a box-score cache added to the box head's presence logit.  The real census finds none; this is what lets the mutation go RED."""
    hs = []
    if "bev_sidecar" in batch:
        s = batch["bev_sidecar"]
        hs.append(model._perception.bev_pool.register_forward_hook(lambda m, i, o: o + s.to(o.dtype)))
    if "box_sidecar" in batch:
        s2 = batch["box_sidecar"]

        def f(m, i, o):
            o = dict(o)
            bs = dict(o["box_slots"])
            bs["presence_logit"] = bs["presence_logit"] + s2.to(bs["presence_logit"].dtype)
            o["box_slots"] = bs
            return o
        hs.append(model._perception.register_forward_hook(f))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("timm")
    return _build_rig()


def _build_rig():
    """the tiny REAL RefCV3Model with C8's wires ON (test_plan_wires_r9_model's rig), a fixed batch, two synthetic offline sidecars and the donor batch."""
    import test_plan_wires_r9_model as PWM
    cfg, model, _ = PWM._rig()
    PWM._arm_nonzero(model)                          # the wires' weights non-zero: a zero gate would hide a leak behind itself
    batch = PWM._batch(cfg)
    B = int(batch["frames"].shape[0])
    g = torch.Generator().manual_seed(7)
    rec, hooked = CB.run_inference(model, batch)
    batch["bev_sidecar"] = 0.2 * torch.randn(tuple(rec["bev_pool"][0].shape), generator=g)
    batch["box_sidecar"] = 1.0 * torch.randn(tuple(rec["box_slots"][0].shape[:2]), generator=g)
    pi = CB.sattolo_derangement(range(B), 0)
    idx = torch.tensor([pi[r] for r in range(B)])
    donor = {k: (v[idx] if torch.is_tensor(v) and v.shape[:1] == (B,) else v) for k, v in batch.items()}
    keys = [k for k in batch if k != "map_ep"]        # map_ep is the rig-bank LOOKUP key (one clip in this bank): a CALIB id, not data
    return {"model": model, "batch": batch, "donor": donor, "keys": keys, "hooked": hooked, "rec": rec}


CORRECT = lambda o, d: CB.swap_item(o, d, census=SYN)  # noqa: E731
PIXELS_ONLY = lambda o, d: CB.swap_item(o, d, census=SYN, swap_keys=("frames", "future_frames"))  # noqa: E731


def _fmt(rows, title):
    out = ["", title, "  %-17s %-6s %-5s %-12s %-12s  %s" % ("key", "class", "px", "d(plan)", "d(any tap)", "taps that moved")]
    for r in rows:
        out.append("  %-17s %-6s %-5s %-12.6g %-12.6g  %s" % (r["key"], r["cls"], "yes" if r["pixel_derived"] else "no", r["max_plan"], r["max_all"],
                                                               ",".join(k for k, v in r["taps"].items() if v["max_abs"] > 0)))
    return "\n".join(out)


def test_the_probe_is_alive_every_tap_fires_and_the_forward_is_deterministic(rig):
    assert rig["hooked"] >= TAPS
    for tap in TAPS:
        assert len(rig["rec"].get(tap, [])) >= 1, tap
    again, _ = CB.run_inference(rig["model"], rig["batch"])
    assert all(v["bit_equal"] and v["max_abs"] == 0.0 for v in CB.diff_taps(rig["rec"], again).values())
    # the wires are ON in this model: the box tokens the planner consumed and the critic's banded map read exist
    assert rig["rec"]["box_tokens"][0].shape[1] == 32 and rig["rec"]["critic_map"][0].shape[-1] == 5 and rig["rec"]["map_chan_1m"][0].shape[1] == 5


def test_IDENTITY_correct_instrument_every_pixel_derived_key_reads_exactly_0_and_only_legitimate_inputs_move_the_planner(rig):
    ref, rows, hooked = CB.identity_sweep(rig["model"], rig["batch"], rig["donor"], instrument=CORRECT, keys=rig["keys"], ports=_ports, census=SYN)
    print(_fmt(rows, "[IDENTITY: correct instrument (every CAMERA key from the donor)]"))
    assert CB.violations(rows) == []
    byk = {r["key"]: r for r in rows}
    px = [r for r in rows if r["pixel_derived"]]
    assert {r["key"] for r in px} >= {"frames", "future_frames", "map_fine", "map_fine_label", "bev_sidecar", "box_sidecar"}
    for r in px:
        assert r["max_all"] == 0.0 and all(v["bit_equal"] and v["max_abs"] == 0.0 for v in r["taps"].values()), r["key"]
    for r in rows:                       # every key that moves the planner is a legitimate non-camera input
        if r["max_all"] > 0.0:
            assert r["cls"] in CB.LEGIT_INPUT, r["key"]
        if r["cls"] == CB.LABEL:
            assert r["max_plan"] == 0.0, r["key"]
    # the probe is not dead: the ego and the nav command DO move the planner (they are inputs and the instrument keeps them the window's own)
    assert byk["pose_last"]["max_plan"] > 0.0 and byk["nav_cmd"]["max_plan"] > 0.0 and byk["pose_hist"]["max_plan"] > 0.0


def test_MUTATION_swap_pixels_only_is_RED_wherever_an_offline_camera_tensor_exists(rig):
    ref, rows, _ = CB.identity_sweep(rig["model"], rig["batch"], rig["donor"], instrument=PIXELS_ONLY, keys=rig["keys"], ports=_ports, census=SYN)
    print(_fmt(rows, "[MUTATION: swap pixels only]"))
    bad = CB.violations(rows)
    assert len(bad) == 2 and any(b.startswith("bev_sidecar") for b in bad) and any(b.startswith("box_sidecar") for b in bad), bad
    byk = {r["key"]: r for r in rows}
    for k in SIDECARS:                                                   # non-zero on the planner OUTPUT and on the planner inputs it feeds
        assert byk[k]["max_plan"] > 0.0 and byk[k]["max_all"] > 0.0, k
    assert byk["bev_sidecar"]["taps"]["bev_pool"]["max_abs"] > 0.0 and byk["bev_sidecar"]["taps"]["box_tokens"]["max_abs"] > 0.0
    assert byk["box_sidecar"]["taps"]["box_slots"]["max_abs"] > 0.0 and byk["box_sidecar"]["taps"]["box_tokens"]["max_abs"] > 0.0
    assert byk["frames"]["max_all"] == 0.0                              # the pixels themselves are swapped by this mutation too


def test_CONTROL_perturbing_the_DONORS_camera_moves_the_planner_under_the_instrument(rig):
    rows = CB.donor_sweep(rig["model"], rig["batch"], rig["donor"], instrument=CORRECT, keys=["frames", *SIDECARS], ports=_ports, census=SYN)
    print("\n[DONOR control] " + "; ".join("%s d(plan)=%.6g d(any)=%.6g" % (r["key"], r["max_plan"], r["max_all"]) for r in rows))
    assert all(r["max_plan"] > 0.0 for r in rows), rows
    byf = {r["key"]: r for r in rows}["frames"]
    assert all(byf["taps"][t]["max_abs"] > 0.0 for t in ("trunk", "bev_pool", "map_chan_1m", "box_slots", "critic_map", "plan"))   # the donor's pixels reach every wire


def test_RED_dropping_one_census_key_from_the_swap_is_named(rig):
    keys = ["frames", *SIDECARS, "map_fine"]
    for dropped in ("frames", "bev_sidecar", "box_sidecar"):
        sw = tuple(k for k in ("frames", "future_frames", *SIDECARS) if k != dropped)
        _, rows, _ = CB.identity_sweep(rig["model"], rig["batch"], rig["donor"], keys=keys, ports=_ports, census=SYN,
                                       instrument=lambda o, d, sw=sw: CB.swap_item(o, d, census=SYN, swap_keys=sw))
        bad = CB.violations(rows)
        assert len(bad) == 1 and bad[0].startswith(dropped), (dropped, bad)


def test_RED_swapping_an_ego_or_nav_key_is_refused(rig):
    for k in ("pose_last", "pose_hist", "nav_cmd", "actions"):
        with pytest.raises(CB.SwapRefused, match=k):
            CB.swap_item(rig["batch"], rig["donor"], census=SYN, swap_keys=("frames", k))


def test_RED_a_feed_that_lets_a_LABEL_reach_the_planner_is_named(rig, monkeypatch):
    orig = CB.roll_feed

    def leaky(model, batch):                                           # the nav command read from the route TARGET (a label) instead of nav_cmd
        kw = orig(model, batch)
        kw["nav_cmd"] = batch["route_target"].clamp(0, 2)
        return kw
    monkeypatch.setattr(CB, "roll_feed", leaky)
    _, rows, _ = CB.identity_sweep(rig["model"], rig["batch"], rig["donor"], instrument=CORRECT, keys=["frames", "route_target", "nav_cmd", "map_fine"], ports=_ports, census=SYN)
    bad = CB.violations(rows)
    assert len(bad) == 1 and bad[0].startswith("route_target") and "LABEL reaches the planner output" in bad[0], bad


def test_RED_a_pixel_derived_label_read_by_the_forward_is_named(rig):
    """The SAM3 map target (offline, pixel-derived) is a LABEL the forward never reads.  If a feed DID read it, the identity names it."""
    model = rig["model"]
    batch = dict(rig["batch"])

    @contextlib.contextmanager
    def reads_map_fine(m, b):
        s = b["map_fine"].float().mean() * 0.01
        h = m._perception.bev_pool.register_forward_hook(lambda mod, i, o: o + s)
        try:
            yield
        finally:
            h.remove()
    _, rows, _ = CB.identity_sweep(model, batch, rig["donor"], instrument=CORRECT, keys=["frames", "map_fine"], ports=reads_map_fine, census=SYN)
    bad = CB.violations(rows)
    assert len(bad) == 1 and bad[0].startswith("map_fine") and "pixel-derived" in bad[0], bad


# =========================================================================== #
# D. the adapter                                                               #
# =========================================================================== #
@pytest.mark.skipif(not ADAPTER.exists(), reason="the adapter lives beside the anticheat package (FlyWheels/, not in the stack-only trees)")
def test_the_adapter_compiles_and_routes_the_dataset_class_and_the_digest_through_this_module():
    src = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    compile(src, str(ADAPTER), "exec")
    assert "from tanitad.eval import camblind_r9 as CB" in src
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls.count("CB.make_camblind_dataset_cls") == 1 and calls.count("CB.other_inputs_digest") == 1 and calls.count("CB.census_sha256") == 1, calls
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "AC"]       # no leftover use of the old pixels-only helper module
    # the pixels-only wrapper is GONE: no class defined inside the adapter overrides `_window_u8`
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_window_u8"]
    assert "--camera" in src and "identity" in src and "derange" in src
