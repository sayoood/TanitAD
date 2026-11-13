"""PREREG_V8F_MRATE, the trainer half: ``--wm-k`` and ``--wm-only`` (every case runs the REAL trainer ``v8f_mini_train.main`` on its synthetic world with tiny dimensions).

* G-DVB: both flags have an entry (``FLAG_CHECKS``), are read back from the BUILT model (``bad_wm_only``) and refuse a lie (a lever on the frozen readers, ``--w-wm 0``, no ``--init-ckpt``, a checkpoint of another shape, a missing freeze).
* The start state (the Master Mind's reading of 2026-11-07 13:09): hmax is not compared; every non-WM tensor of the checkpoint loads strictly and bit-equal; NO checkpoint WM tensor is loaded; the WM is re-initialised under the run seed.
* G-LIVE (control 3): every WM term finite, every WM parameter group non-zero gradient, adapter / readers / heads / planner EXACTLY 0.0; ``w_wm`` 0 FAILS; deleting the adapter ``.detach()`` FAILS.
* Planner identity (control 1): ``forward_infer`` is bit-equal before and after 3 WM-only steps, and after a whole run; a run that trains the planner too FAILS the probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
WMO = ["--hmax", "28", "--wm-only"]


@pytest.fixture(scope="module")
def init(tmp_path_factory):
    root = tmp_path_factory.mktemp("mrate")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    ck = root / "m2b" / "ckpt_last.pt"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    assert blob["cfg"]["hmax"] == 16 and blob["cfg"].get("wm_k", 1) == 1
    return {"root": root, "ck": ck, "blob": blob}


def gate(init, name, extra=(), ck=True):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *BASE, *(["--init-ckpt", str(init["ck"])] if ck else []), *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    return rc


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


def test_both_flags_have_a_declared_vs_built_entry():
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert {"wm_k", "wm_only"} <= dests and dests == set(TR.FLAG_CHECKS)
    assert TR.FLAG_CHECKS["wm_k"] == "lever:wm_k" and TR.FLAG_CHECKS["wm_only"] == "lever:wm_only"
    a = TR.build_parser().parse_args(["--out", "x"])
    assert (a.wm_k, a.wm_only) == (1, False)                                   # the defaults are the earlier behaviour


@pytest.mark.parametrize("k", [1, 3])
def test_the_gate_passes_for_w0_and_w1_and_reads_the_levers_back(init, k):
    name = f"g_k{k}"
    assert gate(init, name, [*WMO, "--wm-k", str(k)]) == 0
    sc = sc_of(init, name)
    lv = sc["levers"]
    assert lv.get("wm_k", 1) == k and ("wm_k_levels" in lv) == (k > 1)
    wo = lv["wm_only"]
    assert wo["n_trainable"] == wo["n_wm"] > 0 and wo["init"]["wm_loaded_from_ckpt"] == 0 and wo["init"]["wm_params_equal_to_ckpt"] == 0 and wo["init"]["wm_reinit_seed"] == 5
    assert wo["init"]["n_mismatch_after_load"] == 0 and wo["init"]["hmax_ckpt"] == 16 and wo["init"]["hmax_built"] == 28
    live = sc["live"]
    assert live["pass"] and live["control_nonwm_grad_sum"] == 0.0 and live["control_adapter_grad_sum"] == 0.0 and not live["missing_terms"] and not live["dead_wm_groups"]
    assert (("wm.enc1" in live["wm_groups_grad"]) == (k > 1)) and all(g > 0.0 for g in live["wm_groups_grad"].values())
    assert sc["planner_identity_probe"]["pass"] and sc["planner_identity_probe"]["n_equal"] == sc["planner_identity_probe"]["n_tensors"] > 5
    assert sc["code_mix_probe"] == {"skipped": "wm_only"}


def test_the_gate_refuses_every_lie(init):
    def refused(name, extra, ck=True, key="ZZDVB-REFUSEDZZ"):
        rc = gate(init, name, extra, ck)
        assert isinstance(rc, str) and key in rc, (name, rc)
        return rc
    assert "does nothing" in refused("r_w0", [*WMO, "--w-wm", "0"])
    assert "without --init-ckpt" in refused("r_nockpt", [*WMO], ck=False)
    assert "--w-acc" in refused("r_wacc", [*WMO, "--w-acc", "0.5"])
    assert "--ego-perturb-p" in refused("r_ego", [*WMO, "--ego-perturb-p", "0.3"])
    assert "o_interface" in refused("r_oif", [*WMO, "--o-interface", "cut"])
    refused("r_notM2a", [*WMO, "--arm", "M2a"])                                # the arm is M2b's: another arm's gate is refused (config mismatch with the checkpoint, and the arm clause)
    assert "wm: 'b' != 'off'" in refused("r_off", [*WMO, "--wm", "off"])
    assert "hmax" in refused("r_hmax_full", ["--hmax", "28"])                  # WITHOUT --wm-only the hmax comparison is still on (the earlier loader)
    with pytest.raises(ValueError, match="wm_k"):                              # a multi-frame WM with no WM is a config error, not a silent no-op
        TR.make_config(TR.build_parser().parse_args(["--out", "x", "--arm", "M1", "--wm-k", "3"]))


def test_the_gate_refuses_a_checkpoint_of_another_shape_and_a_missing_freeze(init, monkeypatch):
    blob = torch.load(init["ck"], map_location="cpu", weights_only=False)
    blob["cfg"]["wm_dim"] = 64
    bad = init["root"] / "bad_wm_dim.pt"
    torch.save(blob, bad)
    rc = None
    try:
        TR.main(["--out", str(init["root"] / "r_shape"), *BASE, *WMO, "--init-ckpt", str(bad), "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    assert isinstance(rc, str) and "wm_dim" in rc, rc
    monkeypatch.setattr(TR, "freeze_all_but_wm", lambda model: None)           # the freeze line deleted: the readers stay trainable -> the gate names the stray parameters
    rc = gate(init, "r_nofreeze", WMO)
    assert isinstance(rc, str) and "trainable parameters are not exactly the WM's" in rc, rc


# ------------------------------------------------------------------------------------------------ G-LIVE and planner identity on a built model
def tiny(k=3, **kw):
    world = None
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=28, kin_mode="yaw", wm_k=k, **kw)
    return cfg


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=28)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    return {"J": J, "batch": D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])}


def built(world, k=3, **kw):
    cfg = tiny(k, **kw)
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(0)
    m = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    TR.freeze_all_but_wm(m)
    return m, cfg


@pytest.mark.parametrize("k", [1, 3])
def test_wm_only_live_check_passes_and_the_controls_read_exactly_zero(world, k):
    m, cfg = built(world, k)
    res = TR.wm_only_live_check(m, world["batch"], cfg)
    assert res["pass"] and res["control_nonwm_grad_sum"] == 0.0 and res["control_adapter_grad_sum"] == 0.0
    assert res["n_wm_params"] > 0 and res["n_nonwm_params"] > 0 and not res["missing_terms"]
    assert all(np.isfinite(v) for v in res["terms"].values()) and res["terms"]["total"] == pytest.approx(cfg.w_wm * res["terms"]["wm"])


def test_wm_only_live_check_fails_with_w_wm_zero(world):
    m, cfg = built(world, 3, w_wm=0.0)
    with pytest.raises(AssertionError, match="G-LIVE"):
        TR.wm_only_live_check(m, world["batch"], cfg)


def test_wm_only_live_check_fails_when_the_adapter_detach_is_deleted(world, monkeypatch):
    m, cfg = built(world, 3)

    def no_detach(self, batch):                                                # wm_only_loss with `.detach()` deleted: the adapter reaches the WM loss
        wm = self.wm(self._encode(batch), batch["act"])
        T = {"wm": wm["total"], **{f"wm/{k}": v for k, v in wm.items() if k != "total"}}
        T["total"] = self.cfg.w_wm * wm["total"]
        return {"terms": T}
    monkeypatch.setattr(M.V8FMini, "wm_only_loss", no_detach)
    with pytest.raises(AssertionError, match="G-LIVE") as e:
        TR.wm_only_live_check(m, world["batch"], cfg)
    assert "adapter" in str(e.value) or "nonwm_with_grad" in str(e.value)


def test_planner_identity_holds_and_fails_for_a_run_that_trains_the_planner(world, monkeypatch):
    m, cfg = built(world, 3)
    res = TR.planner_identity_probe(m, world["batch"], cfg)
    assert res["pass"] and res["n_equal"] == res["n_tensors"] > 5 and res["wm_params_moved"] > 0
    m2, cfg2 = built(world, 3)
    for p in m2.parameters():
        p.requires_grad_(True)                                                 # the freeze forgotten ...

    def with_planner(self, batch):                                             # ... and the planner's loss in the objective
        T = self.loss(batch, p_gt=0.5)["terms"]
        return {"terms": {"total": T["total"]}}
    monkeypatch.setattr(M.V8FMini, "wm_only_loss", with_planner)
    with pytest.raises(AssertionError, match="planner identity failed"):
        TR.planner_identity_probe(m2, world["batch"], cfg2)


# ------------------------------------------------------------------------------------------------ a whole run
@pytest.mark.parametrize("k", [1, 3])
def test_a_wm_only_run_trains_only_the_wm_leaves_the_planner_bit_equal_and_resumes(init, monkeypatch, k):
    common = [*BASE, *WMO, "--wm-k", str(k), "--init-ckpt", str(init["ck"]), "--steps", "6", "--log-every", "2", "--ckpt-every", "3"]
    full = init["root"] / f"run_k{k}"
    assert TR.main(["--out", str(full), *common]) == 0
    s = json.loads((full / "summary.json").read_text(encoding="utf-8"))
    assert s["done"] is True and s["steps"] == 6 and s["wm_only"] is True and s["wm_k"] == k
    ck0, ck1 = init["blob"], torch.load(full / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck1["wm_only"] is True and ck1["cfg"]["wm_k"] == k and ck1["cfg"]["hmax"] == 28
    a, b = ck0["model"], ck1["model"]
    rest = [n for n in a if not n.startswith("wm.")]
    assert rest and all(torch.equal(a[n], b[n]) for n in rest), "a non-WM tensor moved"                         # the planner, readers, heads, adapter: bit-equal to M2b's
    assert ("wm.enc1.net.0.weight" in b) == (k > 1) and tuple(b["wm.stack.encoders.0.net.0.weight"].shape) == (128, 128 * k)
    torch.manual_seed(5)
    fresh = M.WMAux(M.V8FMiniConfig(**ck1["cfg"]))                            # the init the run started the WM from (seed 5): the run moved it
    fs = fresh.state_dict()
    moved = [n for n, q in fs.items() if q.dtype.is_floating_point and f"wm.{n}" in b and not torch.equal(q, b[f"wm.{n}"])]
    assert any("predictors" in n for n in moved) and (k == 1 or any(n.startswith("enc1.") for n in moved))
    log = [json.loads(ln) for ln in (full / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and all(np.isfinite(r["wm"]) and "ade" not in r for r in log) and log[-1]["step"] == 6
    # crash at step 4, resume from the step-3 checkpoint: the SAME run
    res = init["root"] / f"res_k{k}"
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "4")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(res), *common])
    assert not (res / "summary.json").exists()
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(["--out", str(res), *common]) == 0
    r1 = torch.load(res / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    for n in b:
        assert torch.allclose(b[n].float(), r1[n].float(), atol=1e-5), f"resume diverged at {n}"


def test_wm_k_reaches_a_full_m2b_run_through_the_loss(init):
    """`--wm-k 3` WITHOUT --wm-only: the multi-frame WM is built and trained inside the ordinary M2b loss (reachability of the flag through `model.loss`)."""
    out = init["root"] / "full_k3"
    assert TR.main(["--out", str(out), *BASE, "--wm-k", "3", "--steps", "2", "--log-every", "1", "--ckpt-every", "100"]) == 0
    sd = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert sd["cfg"]["wm_k"] == 3 and "wm.enc1.net.0.weight" in sd["model"] and "wm_only" not in sd
    assert json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))["levers"]["wm_k"] == 3
