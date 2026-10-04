"""Control-flow test of `g0_refcv7.main()` with the SPEC A7 additions (A7.2 verdict + M5, A7.3 packs, A7.4 reports), on
CPU with the FAKE trainer / loader of `test_g0_main_a6_flow.py`, extended with two fake slot heads that carry the
REAL channel layout.

    REFCV6_REPO=D:/Projects/TanitAD PYTHONPATH=D:/Projects/TanitAD/stack python -m pytest -q test_g0_main_a7_flow.py

Why it exists: a refcv8 G0 runs unattended for hours; a crash in the new code paths (the M5 swap and restore, the pack
capture, the A7 verdict, the registration gate) would cost the artifact. This drives the REAL `main()` end to end --
argument parsing, 24 seeds, the A6 arm, the M5 mutation, the pack capture on seed 0 and fp32_s0, the verdicts, the JSON,
the sibling `.npz` -- and asserts on the WRITTEN ARTIFACTS, never on a return code.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_g0_main_a6_flow import _run, harness  # noqa: E402,F401  (the fixture and the runner, not the tests)


def _wire_detection(G, model, tmp_path, met):
    """Give the fake model two slot heads with the real SLOT layout (`heavy_truck` wins every slot), make the fake
    trainer emit one detection pack per window and the fake `summarise` turn packs into two per-class cells per head
    (the fraction of slots whose argmax is bus / heavy_truck; support n = 3), and rewrite the in-run row from it."""
    from tanitad.models import agent_slots as A
    s = A.SLOT_SLICES
    tr = G.L.trainer()

    def head():
        m = torch.nn.Module()
        m.head = torch.nn.Linear(3, A.SLOT_WIDTH)
        with torch.no_grad():
            m.head.weight.zero_()
            m.head.bias.zero_()
            m.head.bias[s["cls"].start + 1] = 5.0                          # class index 1 = heavy_truck
        return m
    model.core = SimpleNamespace(agent_head=head())
    model._perception = SimpleNamespace(box_dec=head())

    def summarise(packs, hd):
        out = {f"eval_{hd}_n_windows": float(len(packs))}
        for cname, ci in (("bus", 2), ("heavy_truck", 1)):
            frac = sum(int((p["cls"] == ci).sum()) for p in packs) / (4.0 * len(packs))
            for thr in ("1", "2"):
                out[f"eval_{hd}_det_ap{thr}_{cname}_0_20"] = frac
            out[f"eval_{hd}_det_npos_{cname}_0_20"] = 3.0
        return out
    tr._det_metrics = SimpleNamespace(HEADS=("agent", "box3d"), summarise=summarise)
    orig = tr.compute_losses_v3

    def wrapped(m, eb, device, mode=None, ablate_frames=False):
        el = orig(m, eb, device, mode=mode, ablate_frames=ablate_frames)
        B = eb["x"].shape[0]
        for hd, h in (("agent", m.core.agent_head), ("box3d", m._perception.box_dec)):
            with torch.no_grad():
                raw = h.head(torch.ones(B, 4, 3))
            cls = raw[..., s["cls"]].argmax(-1)
            el[f"_det_pack_{hd}"] = [
                {"ep": 7000 + i, "logit": raw[i, :, 0].numpy(), "xy": np.zeros((4, 2), np.float32),
                 "cls": cls[i].numpy().astype(np.int16), "gt_xy": np.zeros((1, 2), np.float32),
                 "gt_cls": np.zeros(1, np.int16), "pos": np.ones(1, bool), "ign": np.zeros(1, bool)}
                for i in range(B)]
        return el
    tr.compute_losses_v3 = wrapped
    # the in-run row, built the way the fixture builds it, now including the detection cells
    items = G.L.build_eval_dataset(model, None, None, {})[0]
    perm = G.L.inrun_eval_perm(items, 2, 4)
    acc, packs = {}, {"agent": [], "box3d": []}
    for b in range(2):
        eb = G.collate(items, perm[b * 4:(b + 1) * 4], 1)
        with torch.no_grad():
            el = wrapped(model, eb, "cpu")
        for k, v in el.items():
            if k.startswith("_det_pack_"):
                packs[k[len("_det_pack_"):]].extend(v)
            else:
                acc[k] = acc.get(k, 0.0) + float(v)
    inrun = {f"eval_{k}": round(v / 2, 5) for k, v in acc.items()}
    for hd, pk in packs.items():
        inrun.update({k: round(float(v), 5) for k, v in summarise(pk, hd).items()})
    inrun["eval_tacv6_goal_conf_bce"] = round(inrun["eval_tacv6_goal_conf_bce"] * 1.004, 5)
    met.write_text(json.dumps({"step": 100, "eval_loss": 1.0, **inrun}) + "\n", encoding="utf-8")
    return tr


@pytest.fixture
def det(harness, monkeypatch, tmp_path):
    G, tmp, ck, cfg, met = harness
    monkeypatch.delenv("REFCV7_G0_BATCH_CACHE", raising=False)
    monkeypatch.setattr(G, "HERE", tmp_path / "code")          # the default earlier-G0 discovery must not touch the real raw/
    model = G.L.build_model({}, "x", "cpu")[0]
    _wire_detection(G, model, tmp, met)
    return G, tmp, ck, cfg, met, model


def _register(tmp_path, G, monkeypatch, a6=True, a7=True):
    past = time.time() - 3600
    for flag, name, attr in ((a6, "SPEC_SHA256_AMENDMENT_A6.txt", "A6_REGISTRATION"),
                             (a7, "SPEC_SHA256_AMENDMENT_A7.txt", "A7_REGISTRATION")):
        p = tmp_path / name
        if flag:
            p.write_text("sha256 ...", encoding="utf-8")
            os.utime(p, (past, past))
        monkeypatch.setattr(G, attr, p)


def test_main_runs_M5_banks_the_packs_and_reports_A7_as_not_the_gate(det, monkeypatch):
    G, tmp, ck, cfg, met, model = det
    w0 = [h.head.weight.detach().clone() for h in (model.core.agent_head, model._perception.box_dec)]
    out = tmp / "g0.json"
    rec = _run(G, tmp, ck, cfg, met, out, extra=("--mutations", "m5"))      # last --mutations wins over _run's ""
    # ---- M5: ran, swapped, restored bit-exactly, and the heads are the heads we started with ----
    m5 = rec["mutations"]["m5"]
    assert m5["restoration_bit_exact"] is True and m5["swap_took_effect"] is True
    assert m5["swap"]["classes"] == {"bus": 2, "heavy_truck": 1} and m5["swap"]["param_rows"] == [3, 2]
    s0 = rec["by_seed"]["0"]["row"]
    assert (s0["eval_agent_det_ap1_bus_0_20"], s0["eval_agent_det_ap1_heavy_truck_0_20"]) == (0.0, 1.0)
    assert (m5["row"]["eval_agent_det_ap1_bus_0_20"], m5["row"]["eval_agent_det_ap1_heavy_truck_0_20"]) == (1.0, 0.0)
    assert all(torch.equal(h.head.weight, w) for h, w in zip((model.core.agent_head, model._perception.box_dec), w0))
    # ---- A7: computed, REPORTED, and not the gate (A6 / A7 are DRAFT in this harness) ----
    v7 = rec["verdict_A7"]
    assert v7["amendment"] == "A7" and v7["registration"]["registered"] is False
    assert rec["verdict"]["amendment"] == "A5"
    g = v7["a7_lowsupport"]
    assert (g["n_members"], g["N_in"], g["N_num"], g["bound"], g["status"]) == (8, 0, 0, 5, "PASS")
    # 2 heads x 2 classes x 2 thresholds = 8 members, each moved by M5 (bus 0 -> 1, heavy_truck 1 -> 0): 8 > 5
    assert (v7["m5"]["status"], v7["m5"]["N_M5"], v7["m5"]["bound"]) == ("DETECTED", 8, 5)
    assert v7["mutation_detection"]["m5"] == {"detected": True, "n_terms_out": 8, "how": "DETECTED"}
    # ---- the reporting order in the artifact: as registered -> A2 -> A5 -> A6 -> A7 ----
    keys = list(rec)
    order = [keys.index(k) for k in ("verdict_as_registered", "verdict_A2", "verdict_A5", "verdict_A6", "verdict_A7")]
    assert order == sorted(order)
    # ---- A7.3: the packs of seed 0 and fp32_s0, both heads, all 8 windows, in the sibling npz ----
    pk = rec["a7"]["packs"]
    assert pk["status"] == "ALL BANKED" and pk["gaps"] == []
    assert {c: v["n_packs"] for c, v in pk["arms"].items()} == {"s0.agent": 8, "s0.box3d": 8, "fp32_s0.agent": 8,
                                                                "fp32_s0.box3d": 8}
    npz = tmp / "g0.packs.npz"
    assert pk["npz"]["path"] == str(npz) and npz.exists()
    back = G.unpack_detection_packs(npz, "fp32_s0", "box3d")
    assert len(back) == 8 and all((b["cls"] == 1).all() for b in back)
    assert [b["batch"] for b in back] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(b["ep_sha12"] not in ("", "7000") and len(b["ep_sha12"]) == 12 for b in back)
    # ---- A7.4: a diagnostic, never gating; this fake run has a constant `traj`, so the F test is not computable ----
    assert rec["a7"]["gating"] is False
    assert rec["a7"]["seed_group"]["status"] == "NOT COMPUTABLE"
    assert rec["a7"]["seed_draw_correlation"]["status"] == "NO EARLIER G0 (comparable) FOUND"
    assert "ONE draw" in rec["a7"]["inference_seed_floor"]


def test_the_DEFAULT_mutation_list_carries_M5_and_a_probe_that_cannot_run_is_recorded_not_fatal(det, monkeypatch):
    """An A7 G0 launched without `--mutations` (the battery's own command line) must still run M5. The fake model has no
    `_map_hires` / `core.decoder`, so M2 and M4 RAISE -- recorded as such (a loud failure is a detection), never fatal."""
    G, tmp, ck, cfg, met, model = det
    argv = ["g0_refcv7.py", "--ckpt", str(ck), "--config", str(cfg), "--metrics", str(met),
            "--seeds", ",".join(str(i) for i in range(24)), "--diagnostic-arms", "", "--skip-wrapper-control",
            "--micro", "1,3", "--out", str(tmp / "g0d.json")]
    old, sys.argv = sys.argv, argv
    try:
        G.main()
    finally:
        sys.argv = old
    rec = json.load(open(tmp / "g0d.json", encoding="utf-8"))
    assert list(rec["mutations"]) == ["m1", "m2", "m4", "m5"]
    assert "raised" in rec["mutations"]["m2"] and "raised" in rec["mutations"]["m4"]
    assert rec["mutations"]["m5"]["restoration_bit_exact"] is True and "row" in rec["mutations"]["m5"]


def test_main_gates_on_A7_when_registered_before_start_and_reports_A6_beside_it(det, monkeypatch):
    G, tmp, ck, cfg, met, model = det
    _register(tmp, G, monkeypatch)
    rec = _run(G, tmp, ck, cfg, met, tmp / "g0r.json", extra=("--mutations", "m5"))
    assert rec["verdict"]["amendment"] == "A7" and rec["verdict"]["registration"]["registered"] is True
    assert rec["verdict_A6"]["amendment"] == "A6" and rec["verdict_A5"]["amendment"] == "A5"
    assert rec["verdict"]["m5"]["status"] == "DETECTED"


def test_main_A7_registered_but_A6_not_is_not_the_gate_A7_needs_A6(det, monkeypatch):
    G, tmp, ck, cfg, met, model = det
    _register(tmp, G, monkeypatch, a6=False, a7=True)
    rec = _run(G, tmp, ck, cfg, met, tmp / "g0n.json", extra=("--mutations", "m5"))
    assert rec["verdict_A7"]["registration"]["registered"] is False
    assert "A6 is not registered" in rec["verdict_A7"]["registration"]["why"]
    assert rec["verdict"]["amendment"] == "A5"


def test_main_A7_registered_without_the_numerics_arm_FAILS_closed_and_reports_the_missing_packs(det, monkeypatch):
    G, tmp, ck, cfg, met, model = det
    _register(tmp, G, monkeypatch)
    rec = _run(G, tmp, ck, cfg, met, tmp / "g0f.json", extra=("--mutations", "m5", "--no-a6"))
    assert "verdict_A6" not in rec
    v = rec["verdict"]
    assert v["amendment"] == "A7" and v["G0"] == "FAIL"
    assert v["a7_lowsupport"]["status"] == "NOT EVALUABLE"
    assert any(r.startswith("A7.2 DISCRETE-SMALL-N guard NOT EVALUABLE") for r in v["reasons"])
    # A7.3: the fp32_s0 packs are missing and the artifact SAYS so, with the reason
    pk = rec["a7"]["packs"]
    assert pk["status"] == "GAPS: fp32_s0.agent=MISSING, fp32_s0.box3d=MISSING"
    assert pk["arms"]["fp32_s0.agent"]["why"] == "--no-a6: the fp32_s0 arm did not run"
    assert pk["arms"]["s0.agent"]["status"] == "BANKED"


def test_main_a_raising_A7_judge_is_an_ERROR_gate_when_registered_and_a_note_when_not(det, monkeypatch):
    G, tmp, ck, cfg, met, model = det
    real = G.judge

    def boom(inrun, by_seed, rec, amend=None):
        if amend == "A7":
            raise RuntimeError("synthetic A7 defect")
        return real(inrun, by_seed, rec, amend=amend)
    monkeypatch.setattr(G, "judge", boom)
    rec = _run(G, tmp, ck, cfg, met, tmp / "g0e0.json", extra=("--mutations", "m5"))
    assert rec["verdict_A7"]["G0"] == "ERROR" and rec["verdict"]["amendment"] == "A5"
    _register(tmp, G, monkeypatch)
    rec = _run(G, tmp, ck, cfg, met, tmp / "g0e1.json", extra=("--mutations", "m5"))
    assert rec["verdict"]["G0"] == "ERROR" and rec["verdict"]["amendment"] == "A7"      # fail closed, never PASS
    assert rec["verdict"]["reasons"] == ["A7 judge raised RuntimeError: synthetic A7 defect"]


def test_main_earlier_g0_is_correlated_when_given(det, monkeypatch):
    G, tmp, ck, cfg, met, model = det
    first = _run(G, tmp, ck, cfg, met, tmp / "g0a.json", extra=("--mutations", ""))
    earlier = tmp / "earlier_g0.json"
    earlier.write_text(json.dumps({"step": 1, "ckpt_md5": "another", "perm_sha256": first["perm_sha256"],
                                   "by_seed": {str(s): {"per_batch": [{"traj": 0.5}, {"traj": 0.5}]}
                                               for s in range(24)}}), encoding="utf-8")
    rec = _run(G, tmp, ck, cfg, met, tmp / "g0b.json", extra=("--mutations", "", "--earlier-g0", str(earlier)))
    c = rec["a7"]["seed_draw_correlation"]
    assert c["status"] == "COMPUTED" and c["gating"] is False
    assert (c["pairs"][0]["earlier_step"], c["pairs"][0]["n_cells"], c["pairs"][0]["r_cell"]) == (1, 48, None)
    assert "verdict_A7" in rec and rec["a7"]["packs"]["status"] == "ALL BANKED"
