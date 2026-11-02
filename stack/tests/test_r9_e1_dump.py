"""r9_e1_dump.py pin + known-answer tests (refcv9 E1, BRIEF_E1_DUMP). Literal expectations.

* The ORIGINALS the dump copies (PERC8's ``perc_pass_r7.py`` dataset class / tensor picker, ``g0_refcv7.py``'s future-frame helpers,
  MAPSEL's ``mapsel_roll.py::own_planes``) are embedded VERBATIM below with the sha256 of the file they were cut from, and the copies
  in ``scripts/r9_e1_dump.py`` are held TEXT-EQUAL to them. ``own_planes`` is also held equal in BEHAVIOUR (the original exec'd).
* The dump's output is read back by ``r9_e1_eval.py`` (the consumer): GT-as-prediction => AP@2 m 1.0 / near-field precision 1.0.
* A stub loader drives ``run_dump`` end to end (window selection, forward capture, records, rasters, meta, the future-frame control
  and its refusal), for a full model AND a rig.
"""
import hashlib
import inspect
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
import r9_e1_dump as D                                    # noqa: E402
import r9_e1_eval as E                                    # noqa: E402

from tanitad.eval import detection_metrics as dm          # noqa: E402

ORIG = {
    'make_perc_dataset_cls': ('perc_pass_r7.py', '0309c5f5668cbde3d82c87ba1826c0e8205bb7509da340a3eddf7b74ef515201', 'def make_perc_dataset_cls(tr, law_ahead: int):\n    """The trainer\'s V3Dataset; the future frames are ZEROS (never decoded). See the module docstring."""\n    class PercWindows(tr.V3Dataset):\n        def _window_u8(self, i: int) -> dict:          # refb_train.py:220-231, future frames zeroed\n            e_i, t = self.index[i]\n            ep = self.episodes[e_i]\n            w = self.window\n            fr = ep.frames[t:t + w]\n            return {\n                "frames": fr,\n                "actions": ep.actions[t:t + w],\n                "future_frames": torch.zeros((law_ahead,) + tuple(fr.shape[1:]), dtype=fr.dtype),\n                "future_actions": ep.actions[t + w:t + w + self.max_horizon],\n                "future_poses": ep.poses[t + w:t + w + self.max_horizon],\n                "pose_last": ep.poses[t + w - 1],\n                "episode_id": ep.episode_id,\n            }\n    return PercWindows'),
    '_perc_tensors': ('perc_pass_r7.py', '0309c5f5668cbde3d82c87ba1826c0e8205bb7509da340a3eddf7b74ef515201', 'def _perc_tensors(out) -> dict:\n    p = out.get("perception") or {}\n    d = {}\n    if p.get("map_hires_logits") is not None:\n        d["map_hires_logits"] = p["map_hires_logits"]\n    bs = p.get("box_slots") or {}\n    for k in ("presence_logit", "box", "cls_logits", "cz"):\n        if torch.is_tensor(bs.get(k)):\n            d[f"box_slots.{k}"] = bs[k]\n    return d'),
    'LawOnly': ('g0_refcv7.py', '40062e7afab1c92d0dd217e9bf9880d015d8f0a32ad98cfd3bb8ef17f8416d34', 'class LawOnly:\n    def __init__(self, u8_frame, law_idx):\n        self.u8 = u8_frame\n        self.law_idx = int(law_idx)'),
    'LawOnlyDevice': ('g0_refcv7.py', '40062e7afab1c92d0dd217e9bf9880d015d8f0a32ad98cfd3bb8ef17f8416d34', 'class LawOnlyDevice:\n    def __init__(self, dev_frame, law_idx):\n        self.x = dev_frame\n        self.law_idx = law_idx\n\n    def __getitem__(self, key):\n        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)\n                and int(key[1]) == self.law_idx):\n            raise KeyError(f"LawOnlyDevice holds only [:, {self.law_idx}], asked {key}")\n        return self.x'),
    'patch_frames_to_device': ('g0_refcv7.py', '40062e7afab1c92d0dd217e9bf9880d015d8f0a32ad98cfd3bb8ef17f8416d34', 'def patch_frames_to_device(tr):\n    if "f2d" in _PATCHED:\n        return\n    orig = tr.frames_to_device\n\n    def f(x, device):\n        if isinstance(x, LawOnly):\n            return LawOnlyDevice(orig(x.u8, device), x.law_idx)\n        return orig(x, device)\n    tr.frames_to_device = f\n    _PATCHED["f2d"] = orig'),
    'collate': ('g0_refcv7.py', '40062e7afab1c92d0dd217e9bf9880d015d8f0a32ad98cfd3bb8ef17f8416d34', 'def collate(e_ds, idx_list, law_idx):\n    import torch.utils.data as tud\n    items = [e_ds[i] for i in idx_list]\n    eb = tud.default_collate(items)\n    ff = eb["future_frames"]\n    eb["future_frames"] = LawOnly(ff[:, law_idx].clone(), law_idx)\n    del ff\n    return eb'),
    'own_planes': ('mapsel_roll.py', 'e075b8ed5204cad1b96b94add8ea9d523cb798850dbe182d75f85862fb16a3c9', 'def own_planes(model, logits_r):\n    """logits_r [1, 8, H, W] -> (pdrv_u8, pedge_u8, psw_u8, code_u8) numpy [H, W], under the model\'s DECLARED rule: p^ = softmax(z - log w) (w = the loss\'s frozen class weights),\n    code = argmax(z - log w) (== map_head_hires.decide(\'prior_corrected\', w), asserted on a sample), pdrv = sum of p^ over the DRIVABLE codes."""\n    from tanitad.models import map_head_hires as MH\n    cw = model._map_hires_class_weight\n    rule = str(model._map_hires.cfg.decision_rule)\n    ST["c"]["rule_seen"].add(rule)\n    if rule != "prior_corrected" or cw is None:\n        raise RuntimeError(f"[mapsel] the declared decision rule is {rule!r} / class weights {\'absent\' if cw is None else \'present\'}: this arm is defined for prior_corrected + weights")\n    z = logits_r.detach().float()\n    s = z - MH._decision_logw(cw, z).view(1, -1, 1, 1)\n    p = torch.softmax(s, 1)[0]\n    code = s.argmax(1)[0]\n    if ST["c"]["decide_checked"] < 25 or ST["c"]["calls"] % 100 == 0:\n        ref = MH.decide(z, rule, cw)[0]\n        ST["c"]["decide_checked"] += 1\n        ST["c"]["decide_mismatch_cells"] += int((ref != code).sum())\n    pdrv = p[list(DRV_CODES)].sum(0)\n    q = lambda t: t.mul(255.0).round().clamp_(0, 255).to(torch.uint8).cpu().numpy()   # noqa: E731\n    return q(pdrv), q(p[EDGE_CODE]), q(p[SIDEWALK_CODE]), code.to(torch.uint8).cpu().numpy()'),
}


def _sha12(c):
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------------- the pins
def test_verbatim_copies_are_text_equal_to_the_banked_originals():
    assert set(ORIG) == {"make_perc_dataset_cls", "_perc_tensors", "LawOnly", "LawOnlyDevice", "patch_frames_to_device",
                         "collate", "own_planes"}
    for name, (fname, sha, src) in ORIG.items():
        assert len(sha) == 64 and fname.endswith(".py")
        mine = inspect.getsource(getattr(D, name)).replace("\r\n", "\n")
        assert mine.strip() == src.strip(), f"{name} drifted from {fname}"


def test_constants_match_the_mapsel_originals():
    assert D.DRV_CODES == (1, 2, 3, 4, 6) and D.EDGE_CODE == 5 and D.SIDEWALK_CODE == 7


class _Cfg:
    decision_rule = "prior_corrected"


class _Branch:
    cfg = _Cfg()


class _Model:
    def __init__(self, w, rule="prior_corrected"):
        self._map_hires = _Branch()
        self._map_hires.cfg = type("C", (), {"decision_rule": rule})()
        self._map_hires_class_weight = w


W8 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5])


def test_own_planes_equals_the_original_exec_on_random_logits():
    orig_src = ORIG["own_planes"][2]
    ns = {"torch": torch, "DRV_CODES": (1, 2, 3, 4, 6), "EDGE_CODE": 5, "SIDEWALK_CODE": 7,
          "ST": {"c": {"calls": 0, "rule_seen": set(), "decide_checked": 0, "decide_mismatch_cells": 0}}}
    exec(compile(orig_src, "mapsel_roll.own_planes", "exec"), ns)
    g = torch.Generator().manual_seed(3)
    for _ in range(4):
        z = torch.randn(1, 8, 40, 30, generator=g) * 3.0
        m = _Model(W8)
        mine = D.own_planes(m, z)
        ref = ns["own_planes"](m, z)
        assert len(mine) == 4
        for a, b in zip(mine, ref):
            assert a.dtype == np.uint8 and a.shape == (40, 30)
            np.testing.assert_array_equal(a, b)
    assert D.ST["c"]["decide_mismatch_cells"] == 0           # the argmax IS map_head_hires.decide('prior_corrected')


def test_own_planes_known_answers():
    def planes(cls, w=W8):
        z = torch.full((1, 8, 6, 5), -30.0)
        z[:, cls] = 30.0
        return D.own_planes(_Model(w), z)
    pdrv, pedge, psw, code = planes(7)                         # strongly sidewalk
    assert (psw == 255).all() and (pdrv == 0).all() and (pedge == 0).all() and (code == 7).all()
    pdrv, pedge, psw, code = planes(2)                         # drivable (lane)
    assert (pdrv == 255).all() and (psw == 0).all() and (code == 2).all()
    pdrv, pedge, psw, code = planes(5)                         # edge: NOT a drivable code
    assert (pedge == 255).all() and (pdrv == 0).all() and (code == 5).all()
    # the prior correction is APPLIED: all-equal logits, non-uniform weights => the class with the SMALLEST weight wins (-log w largest)
    z0 = torch.zeros(1, 8, 4, 4)
    assert (D.own_planes(_Model(W8), z0)[3] == 7).all()
    assert (D.own_planes(_Model(torch.ones(8)), z0)[3] == 0).all()   # uniform weights: argmax of a tie is class 0
    with pytest.raises(RuntimeError):                          # a rule that is not prior_corrected, or no weights, REFUSES
        D.own_planes(_Model(W8, rule="raw"), z0)
    with pytest.raises(RuntimeError):
        D.own_planes(_Model(None), z0)


# ---------------------------------------------------------------------------------------------------------------- formats read by the evaluator
def _pack(gt_xy, gt_cls=None, n_slots=20):
    """GT-as-prediction (A2's fixture): slot i sits on GT i, logit +8; the rest far away at logit -8."""
    g = len(gt_xy)
    gt_xy = np.asarray(gt_xy, np.float32).reshape(-1, 2)
    gc = np.zeros(g, np.int16) if gt_cls is None else np.asarray(gt_cls, np.int16)
    xy = [gt_xy[i] for i in range(g)]
    logit = [8.0] * g
    cls = [gc[i] for i in range(g)]
    while len(xy) < n_slots:
        xy.append(np.array([500.0 + len(xy), 500.0], np.float32)); logit.append(-8.0); cls.append(0)
    n = len(xy)
    return {"ep": None, "logit": np.array(logit, np.float32), "xy": np.array(xy, np.float32), "cls": np.array(cls, np.int16),
            "cls_corr": np.array(cls, np.int16), "matched": np.zeros(n, bool), "exempt": np.zeros(n, bool),
            "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0),
            "gt_xy": gt_xy, "gt_cls": gc, "pos": np.ones(g, bool), "ign": np.zeros(g, bool), "hidden": np.zeros(g, bool)}


GT = [[5.0, 1.0], [12.0, -2.0], [25.0, 3.0], [33.0, 0.0], [47.0, 2.0], [55.0, -1.0]]


def test_records_and_packs_are_read_by_the_evaluator_gt_as_prediction(tmp_path):
    det = []
    for c in range(3):
        for ws in (7, 12):
            el = {"_det_pack_box3d": [_pack(GT, [0, 1, 0, 2, 0, 1])], "_det_pack_agent": [_pack(GT)], "loss": 1.0}
            rec = D.make_record("%012x" % (c + 1), ws, el, dm.HEADS)
            assert set(rec) == {"sha12", "ws", "box3d", "agent"}
            det.append(rec)
    assert D.make_record("a" * 12, 3, {"loss": 1.0}, dm.HEADS) == {"sha12": "a" * 12, "ws": 3}      # no pack -> no head key
    p = tmp_path / "perc_det.pkl"
    sha = D.write_packs(p, det)
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()
    recs = E.load_records(p)
    assert len(recs) == 6 and recs[0][0] == "000000000001" and recs[0][1] == 7
    keys, items = E.build_clip_items(recs, (1.0, 2.0), 0.5, None)
    cl, ids = E.pool_by_clip(keys, items)
    s = E.box_stats(cl, (1.0, 2.0))
    assert s["ap2m_all"] == 1.0 and s["ap1m_all"] == 1.0 and s["n_pos"] == 36.0 and len(ids) == 3


def _rast_planes(rng, sidewalk_corridor=False):
    pdrv = (rng.random((1000, 600)) * 255).astype(np.uint8)
    pedge = (rng.random((1000, 600)) * 255).astype(np.uint8)
    psw = np.zeros((1000, 600), np.uint8)
    code = rng.integers(0, 8, (1000, 600)).astype(np.uint8)
    gt = np.full((1000, 600), 1, np.uint8)
    if sidewalk_corridor:
        gt[20:180, 280:320] = 7
        psw[20:180, 280:320] = 255                              # flags exactly the GT sidewalk: precision = recall = 1
    return (pdrv, pedge, psw, code), gt


def test_rasters_have_mapsel_layout_and_are_read_by_the_nearfield_reader(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "rast"
    planes, gt = _rast_planes(rng, True)
    n = D.write_raster(d, 3, 17, planes, gt)
    D.write_raster(d, 3, 22, planes, None)                       # a window with no 10 cm label: no `gt` plane
    assert sorted(os.listdir(d)) == ["e003_w0017.npz", "e003_w0022.npz"] and n == os.path.getsize(d / "e003_w0017.npz")
    z = np.load(d / "e003_w0017.npz")
    assert sorted(z.files) == ["code", "gt", "pdrv", "pedge", "psw"] and all(z[k].dtype == np.uint8 and z[k].shape == (1000, 600) for k in z.files)
    assert sorted(np.load(d / "e003_w0022.npz").files) == ["code", "pdrv", "pedge", "psw"]
    assert D.raster_name(0, 7) == "e000_w0007.npz" and D.raster_name(137, 1234) == "e137_w1234.npz"
    names, acc, gtcls = E.nearfield_per_file(sorted(str(p) for p in d.glob("*.npz")))
    assert names == ["e003_w0017.npz"]                           # the label-less file is skipped, as in the original script
    tab = E.nearfield_table(acc, gtcls)
    assert tab["0-5"]["n_flags"] == 0.0 or tab["0-5"]["precision_exact"] == 1.0
    assert tab["5-10"]["precision_exact"] == 1.0 and tab["5-10"]["recall"] == 1.0 and tab["10-20"]["precision_exact"] == 1.0


# ---------------------------------------------------------------------------------------------------------------- windows
def _grid_and_split(n_clips=6, per=4):
    clips = ["clip-%02d" % i for i in range(n_clips)]
    grid = {c: [7 + 5 * k for k in range(per)] for c in clips}
    sh = [_sha12(c) for c in clips]
    dev, hold = sh[0::2], sh[1::2]
    mk = lambda lst: {"n_episodes": len(lst), "n_windows_s2": per * len(lst), "sha12": sorted(lst),           # noqa: E731
                      "windows_per_episode": {s: [per, per - 1] for s in sorted(lst)}}
    return clips, grid, {"tool": "e0i_split.py", "DEV": mk(dev), "HOLD": mk(hold), "inputs": {"s2_windows_json_sha256": "f" * 64}}


def test_split_list_loader_reads_the_real_e0i_shape(tmp_path):
    clips, grid, split = _grid_and_split()
    p = tmp_path / "split.json"
    p.write_text(json.dumps(split), encoding="utf-8")
    d, h = E.load_window_list(p, "dev"), E.load_window_list(p, "hold")
    assert d == split["DEV"]["sha12"] and h == split["HOLD"]["sha12"] and not set(d) & set(h) and len(d) + len(h) == 6


def test_select_clips_and_split_checks(tmp_path):
    clips, grid, split = _grid_and_split()
    p = tmp_path / "split.json"
    p.write_text(json.dumps(split), encoding="utf-8")
    sel, sha, rec = D.select_clips(grid, _sha12, p, "dev")
    assert sorted(_sha12(c) for c in sel) == split["DEV"]["sha12"] == sorted(sha) and rec["set"] == "DEV" and rec["n_clips"] == 3
    allc, allsha, rec_all = D.select_clips(grid, _sha12)
    assert len(allc) == 6 and rec_all["set"] == "ALL"
    counts = {_sha12(c): len(v) for c, v in grid.items()}
    ok = D.check_split_against_grid(split, "dev", "f" * 64, counts)
    assert ok["checked"] is True
    with pytest.raises(SystemExit):                              # a grid that is not the one the split was cut from
        D.check_split_against_grid(split, "dev", "0" * 64, counts)
    bad = dict(counts)
    bad[split["DEV"]["sha12"][0]] = 99
    with pytest.raises(SystemExit):                              # a window count that differs
        D.check_split_against_grid(split, "dev", "f" * 64, bad)
    gone = {k: v for k, v in grid.items() if _sha12(k) != split["DEV"]["sha12"][0]}
    with pytest.raises(SystemExit):                              # a listed clip the grid lacks
        D.select_clips(gone, _sha12, p, "dev")


def test_window_order_follows_manifest_order_and_refuses_a_missing_window():
    clips, grid, _ = _grid_and_split(4, 3)
    ci = {c: i for i, c in enumerate(reversed(clips))}           # manifest order is NOT sorted order
    W = 5
    pos = {(ci[c], ws - (W - 1)): 100 * ci[c] + k for c in clips for k, ws in enumerate(grid[c])}
    order = D.window_order(grid, clips, ci, pos, W)
    assert [ci[c] for c, _w, _j in order] == sorted(ci[c] for c in clips for _ in grid[c])
    assert [j for _c, _w, j in order][:3] == [k for k in range(3)] and len(order) == 12
    d1 = D.order_digest(order, _sha12)
    assert d1 == D.order_digest(list(order), _sha12) and d1 != D.order_digest(order[::-1], _sha12)
    del pos[(ci[clips[1]], grid[clips[1]][1] - (W - 1))]
    with pytest.raises(SystemExit):
        D.window_order(grid, clips, ci, pos, W)


def test_rig_step0_checkpoint_keeps_exactly_the_built_roots(tmp_path):
    src = tmp_path / "ema.pt"
    sd = {k: torch.zeros(2) for k in ("core.encoder.net.w", "core.agent_head.q", "_perception.a", "_map_hires.b", "core.decoder.z",
                                      "scorer.k", "phi_tac.x", "core.lat_log_prior")}
    torch.save({"model": sd, "step": 48852, "kind": "ema_model_only"}, src)
    rec = D.make_rig_step0_ckpt(src, tmp_path / "rig.pt")
    out = torch.load(tmp_path / "rig.pt", weights_only=False)
    assert sorted(out["model"]) == ["_map_hires.b", "_perception.a", "core.agent_head.q", "core.encoder.net.w"]
    assert rec["n_kept"] == 4 and rec["n_dropped"] == 4 and out["step"] == 48852 and out["kind"] == "rig_step0_from_ema_model_only"
    torch.save({"model": {"weird.key": torch.zeros(1)}, "step": 1}, src)
    with pytest.raises(SystemExit):                              # neither built nor skipped by the rig: REFUSED, never silently kept
        D.make_rig_step0_ckpt(src, tmp_path / "rig2.pt")


# ---------------------------------------------------------------------------------------------------------------- end to end (stub loader)
class _Ep:
    def __init__(self, k, T):
        self.frames = torch.randint(0, 255, (T, 1, 8, 8), dtype=torch.uint8)
        self.actions = torch.zeros(T, 2)
        self.poses = torch.zeros(T, 3)
        self.episode_id = 1000 + k


def _install_stub(monkeypatch, tmp_path, rig, leak=False, label_every=2):
    clips, grid, split = _grid_and_split(4, 4)
    W, LA = 5, 3
    T = 60
    eps = [_Ep(k, T) for k in range(4)]
    manifest_ids = list(reversed(clips))                          # eval manifest order
    tr = types.SimpleNamespace(LAW_AHEAD=LA, calls=[])

    class V3Dataset:
        def __init__(self, episodes, window=W, max_horizon=20, channels=1):
            self.episodes, self.window, self.max_horizon = episodes, window, max_horizon
            self.index = [(e, t) for e in range(len(episodes)) for t in range(0, T - W - max_horizon + 1, 1)]
            self.u8_frames = True

        def __len__(self):
            return len(self.index)

        def _window_u8(self, i):
            raise AssertionError("the subclass overrides this")

        def __getitem__(self, i):
            item = self._window_u8(i)
            e_i, t = self.index[i]
            item["map_fine"] = torch.full((16, 12), 1 + (e_i % 2), dtype=torch.uint8)
            item["map_fine_label"] = torch.tensor(bool((t % label_every) == 0))
            return item

    tr.V3Dataset = V3Dataset
    tr.frames_to_device = lambda x, device: x.float() / 255.0
    tr._det_metrics = dm

    def compute_losses_v3(model, eb, device, mode="diffusion", ablate_frames=False):
        frames = tr.frames_to_device(eb["frames"], device)
        fut = tr.frames_to_device(eb["future_frames"], device)[:, LA - 1]
        out = model(frames, future=fut if leak else None)
        el = {"loss": torch.zeros(()), "_det_pack_box3d": [_pack(GT, [0, 1, 0, 2, 0, 1])], "_det_pack_agent": [_pack(GT)]}
        if not rig:
            el["law"] = fut.float().mean()                         # the full model's law term READS the future frame
        return el
    tr.compute_losses_v3 = compute_losses_v3

    class Core:
        _perc_rig = rig

    class Model:
        core = Core()
        _map_hires = _Branch()
        _map_hires_class_weight = W8

        def forward(self, frames, future=None):
            seed = int(frames.double().sum().item() * 1000) % 100000
            g = torch.Generator().manual_seed(seed)
            lg = torch.randn(frames.shape[0], 8, 16, 12, generator=g) * 2.0
            if future is not None:
                lg = lg + future.float().mean()
            return {"perception": {"map_hires_logits": lg, "box_slots": {"presence_logit": lg[:, 0, 0, :4]}}}

        def __call__(self, *a, **k):
            return self.forward(*a, **k)
    model = Model()
    args = types.SimpleNamespace(eval_cache="EVAL", eval_labels=None, pose_sync_sidecar="/x/sync.jsonl", mode="diffusion", ablate_frames=False)
    L = types.ModuleType("tanitad.eval.refcv7_loader")
    L.PATH_REMAP = {"--eval-labels": "/kit/s2_labels_v8_eval.jsonl.gz"}
    L.bootstrap = lambda: None
    L.trainer = lambda: tr
    L.load_config = lambda p: json.load(open(p))
    L.sha12 = _sha12
    L.md5_file = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    seen = {}

    def build_model(config, ckpt, device):
        seen["ckpt"] = ckpt
        cfg = types.SimpleNamespace(core=types.SimpleNamespace(window=W))
        return model, cfg, args, {"state_dict": {"step": 48852}, "departures": ["stub"], "declared_vs_built": {"mismatches": []},
                                  "param_breakdown": {"equal": False}}
    L.build_model = build_model

    def build_eval_dataset(model_, cfg, args_, config, *, with_perception_targets=True, dataset_cls=None):
        seen["args_pose_sync"] = args_.pose_sync_sidecar
        seen["args_eval_labels"] = args_.eval_labels
        seen["with_targets"] = with_perception_targets
        assert issubclass(dataset_cls, V3Dataset)
        return dataset_cls(eps, window=W), eps, {"pose_sync": None}
    L.build_eval_dataset = build_eval_dataset
    import tanitad.eval as te
    monkeypatch.setitem(sys.modules, "tanitad.eval.refcv7_loader", L)
    monkeypatch.setattr(te, "refcv7_loader", L, raising=False)
    import tanitad.data.v2_dataset as v2
    monkeypatch.setattr(v2, "load_or_build_manifest", lambda cache, verbose=False: {"clip_id": manifest_ids})
    D._PATCHED.clear()
    D.ST["c"].update({"calls": 0, "rule_seen": set(), "decide_checked": 0, "decide_mismatch_cells": 0})
    grid_p = tmp_path / "s2.json"
    grid_p.write_text(json.dumps(grid), encoding="utf-8")
    split["inputs"]["s2_windows_json_sha256"] = hashlib.sha256(grid_p.read_bytes()).hexdigest()
    split_p = tmp_path / "split.json"
    split_p.write_text(json.dumps(split), encoding="utf-8")
    cfg_p = tmp_path / "config.json"
    cfg_p.write_text(json.dumps({"argv": []}), encoding="utf-8")
    ck_p = tmp_path / "ema.pt"
    torch.save({"model": {"core.encoder.w": torch.zeros(1)}, "step": 48852, "kind": "ema_model_only"}, ck_p)
    return types.SimpleNamespace(grid=grid, split=split, grid_p=grid_p, split_p=split_p, cfg_p=cfg_p, ck_p=ck_p, seen=seen, clips=clips)


def _argv(st, out, *extra, split="dev"):
    return ["--ckpt", str(st.ck_p), "--config", str(st.cfg_p), "--windows", str(st.split_p), "--split", split, "--s2-windows", str(st.grid_p),
            "--out", str(out), "--device", "cpu", *extra]


@pytest.mark.parametrize("rig", [False, True])
def test_run_dump_end_to_end_formats_controls_and_privacy(monkeypatch, tmp_path, rig):
    st = _install_stub(monkeypatch, tmp_path, rig)
    out = tmp_path / ("rig" if rig else "full")
    assert D.main(_argv(st, out)) == 0
    meta = json.loads((out / "dump.json").read_text(encoding="utf-8"))
    dev = set(st.split["DEV"]["sha12"])
    det = pickle.loads((out / "perc_det.pkl").read_bytes())
    assert meta["rig"] is rig and meta["n_clips"] == 2 and meta["n_windows"] == len(det) == 8
    assert {r["sha12"] for r in det} == dev and all(set(r) == {"sha12", "ws", "box3d", "agent"} for r in det)
    assert sorted((r["sha12"], r["ws"]) for r in det) == sorted((_sha12(c), w) for c in st.clips if _sha12(c) in dev for w in st.grid[c])
    # the consumer reads it: GT-as-prediction => AP 1.0
    recs = E.load_records(out / "perc_det.pkl", dev)
    keys, items = E.build_clip_items(recs, (1.0, 2.0), 0.5, None)
    s = E.box_stats(E.pool_by_clip(keys, items)[0], (1.0, 2.0))
    assert s["ap2m_all"] == 1.0
    # rasters: one per window, MAPSEL names (e_i = manifest index), `gt` only where the label is true
    files = sorted(os.listdir(out / "rast"))
    assert len(files) == 8 and all(f.startswith("e00") and f.endswith(".npz") and ".tmp" not in f for f in files)
    n_gt = sum("gt" in np.load(out / "rast" / f).files for f in files)
    assert meta["n_rasters"] == 8 and meta["n_rasters_with_gt"] == n_gt and 0 < n_gt < 8
    assert all(np.load(out / "rast" / f)["pdrv"].dtype == np.uint8 for f in files)
    # the clock / labels / control records
    assert st.seen["args_pose_sync"] is None and meta["pose_sync"] == "off" and any("NOT enabled" in x for x in meta["departures"])
    assert st.seen["args_eval_labels"] == "/kit/s2_labels_v8_eval.jsonl.gz" and any("--eval-labels" in x for x in meta["departures"])
    assert st.seen["with_targets"] is True
    assert meta["future_frame_control"]["pass"] is True and len(meta["future_frame_control"]["windows"]) == 2
    w0 = meta["future_frame_control"]["windows"][0]
    assert w0["has_law_term"] is (not rig) and (w0["law_differs"] is True if not rig else w0["law_differs"] is None)
    assert meta["weights_are_ema"] is True and meta["ckpt_md5"] and meta["packs"]["sha256"] and meta["map_decision"]["decide_mismatch_cells"] == 0
    assert meta["split"]["set"] == "DEV" and meta["split"]["check"]["checked"] is True
    # privacy: no raw clip id anywhere in the output
    for p in [out / "dump.json", out / "perc_det.pkl"]:
        assert b"clip-" not in p.read_bytes(), p
    # --pose-sync on keeps the run's clock
    (tmp_path / "b").mkdir()
    st2 = _install_stub(monkeypatch, tmp_path / "b", rig)
    assert D.main(_argv(st2, tmp_path / "on", "--pose-sync", "on", "--every", "2")) == 0
    assert st2.seen["args_pose_sync"] == "/x/sync.jsonl"
    m2 = json.loads((tmp_path / "on" / "dump.json").read_text(encoding="utf-8"))
    assert m2["n_windows"] == 4 and m2["SMOKE_RESTRICTED"] is True and m2["pose_sync"] == "on"


def test_run_dump_refuses_when_the_future_frame_leaks_into_perception(monkeypatch, tmp_path):
    st = _install_stub(monkeypatch, tmp_path, rig=False, leak=True)
    with pytest.raises(SystemExit):
        D.main(_argv(st, tmp_path / "leaky"))
    assert (tmp_path / "leaky" / "dump.json.REFUSED").is_file() and not (tmp_path / "leaky" / "perc_det.pkl").exists()


def test_require_ema_refuses_a_raw_checkpoint(monkeypatch, tmp_path):
    st = _install_stub(monkeypatch, tmp_path, rig=False)
    torch.save({"model": {"core.encoder.w": torch.zeros(1)}, "step": 5, "ema": {"schema": 1}}, st.ck_p)
    with pytest.raises(SystemExit):
        D.main(_argv(st, tmp_path / "raw", "--require-ema"))
    assert D.main(_argv(st, tmp_path / "raw2")) == 0
    m = json.loads((tmp_path / "raw2" / "dump.json").read_text(encoding="utf-8"))
    assert m["weights_are_ema"] is False and "raw weights" in m["ckpt_kind"]


def test_rig_step0_flag_builds_from_a_full_ema(monkeypatch, tmp_path):
    st = _install_stub(monkeypatch, tmp_path, rig=True)
    torch.save({"model": {"core.encoder.w": torch.zeros(1), "core.decoder.z": torch.zeros(1)}, "step": 48852, "kind": "ema_model_only"}, st.ck_p)
    assert D.main(_argv(st, tmp_path / "s0", "--rig-step0-from", str(st.ck_p))) == 0
    m = json.loads((tmp_path / "s0" / "dump.json").read_text(encoding="utf-8"))
    assert m["rig_step0"]["n_kept"] == 1 and m["rig_step0"]["n_dropped"] == 1 and st.seen["ckpt"].endswith("rig_step0_ckpt.pt")
    assert m["weights_are_ema"] is True and m["step"] == 48852


# ---------------------------------------------------------------------------------------------------------------- compare
def test_compare_packs_identical_perturbed_and_missing(tmp_path):
    def mk(delta=0.0, drop=0):
        det = []
        for c in range(3):
            for ws in (7, 12):
                pk = _pack(GT, [0, 1, 0, 2, 0, 1])
                pk["logit"] = pk["logit"] + np.float32(delta if (c, ws) == (1, 12) else 0.0)
                det.append({"sha12": "%012x" % (c + 1), "ws": ws, "box3d": [pk]})
        return det[drop:]
    r = D.compare_packs(mk(), mk())
    assert r["n_common"] == 6 and r["n_windows_bit_identical"] == 6 and all(v["max_abs_diff"] == 0.0 for v in r["keys"].values())
    r = D.compare_packs(mk(), mk(delta=1e-3))
    assert r["n_windows_bit_identical"] == 5 and r["keys"]["logit"]["n_windows_differing"] == 1
    assert 9e-4 < r["keys"]["logit"]["max_abs_diff"] < 1.1e-3 and r["keys"]["gt_xy"]["max_abs_diff"] == 0.0
    r = D.compare_packs(mk(), mk(drop=2))
    assert r["n_common"] == 4 and r["only_a"] == 2 and r["only_b"] == 0
    b = mk()
    b[0]["box3d"][0]["xy"] = b[0]["box3d"][0]["xy"][:5]
    assert D.compare_packs(mk(), b)["keys"]["xy"]["n_shape_mismatch"] == 1


def test_compare_rasters_tolerance_rule(tmp_path):
    rng = np.random.default_rng(5)
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(6):
        planes, gt = _rast_planes(rng, True)
        D.write_raster(a, 0, 7 + 5 * i, planes, gt)
        D.write_raster(b, 0, 7 + 5 * i, planes, gt)
    r = D.compare_rasters(a, b)
    assert r["n_common"] == 6 and r["pass"] is True and all(v["max_abs_diff"] == 0 for v in r["planes"].values())
    assert D.compare_rasters(a, b, n=3)["n_common"] == 3
    z = dict(np.load(b / "e000_w0007.npz"))
    z["pdrv"] = z["pdrv"].copy()
    z["pdrv"][10, 10] = np.uint8((int(z["pdrv"][10, 10]) + 1) % 256)             # one count of 255 on one cell: float noise, inside tol 2
    np.savez_compressed(b / "e000_w0007.npz", **z)
    r = D.compare_rasters(a, b)
    assert r["pass"] is True and r["planes"]["pdrv"]["max_abs_diff"] <= 2 and r["planes"]["pdrv"]["n_cells_differing"] == 1
    z["pdrv"][10, 11] = np.uint8((int(z["pdrv"][10, 11]) + 5) % 256)             # 5 counts: beyond the declared float noise
    np.savez_compressed(b / "e000_w0007.npz", **z)
    r = D.compare_rasters(a, b)
    assert r["pass"] is False and r["planes"]["pdrv"]["max_abs_diff"] >= 5 and D.compare_rasters(a, b, tol_prob=8)["pass"] is True
    z["pdrv"] = np.load(a / "e000_w0007.npz")["pdrv"]
    z["code"] = z["code"].copy()
    z["code"][3, 3] = (int(z["code"][3, 3]) + 1) % 8                              # a class code differs: a different forward
    np.savez_compressed(b / "e000_w0007.npz", **z)
    assert D.compare_rasters(a, b)["pass"] is False
    z2 = dict(z)
    z2.pop("gt")
    np.savez_compressed(b / "e000_w0007.npz", **z2)
    r = D.compare_rasters(a, b)
    assert r["files_with_missing_plane"] == 1 and r["pass"] is False


def test_compare_cli_writes_json(tmp_path):
    det = [{"sha12": "a" * 12, "ws": 7, "box3d": [_pack(GT)]}]
    for n in ("a.pkl", "b.pkl"):
        (tmp_path / n).write_bytes(pickle.dumps(det))
    assert D.main(["compare", "--packs-a", str(tmp_path / "a.pkl"), "--packs-b", str(tmp_path / "b.pkl"), "--out", str(tmp_path / "c.json")]) == 0
    r = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert r["packs"]["n_windows_bit_identical"] == 1 and r["packs_a"]["sha256"] == r["packs_b"]["sha256"]
