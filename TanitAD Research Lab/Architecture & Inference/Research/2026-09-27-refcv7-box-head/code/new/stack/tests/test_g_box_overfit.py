"""G-BOX-OVERFIT harness (SPEC_REFCV7 A9 / A10 §15.1): the prereg literals, the verdict logic, the controls, and a
TOY end-to-end that proves the loop can tell a head that memorises from one that cannot (the must-fail arm).

The real run needs Thor (the 16 frames' payloads, the sidecar, the GPU); these tests pin everything that does not.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _gbo():
    spec = importlib.util.spec_from_file_location("g_box_overfit_t", str(ROOT / "scripts" / "g_box_overfit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _gbo()


def test_the_prereg_literals_are_A10s():
    assert (G.N_FRAMES, G.N_POS, G.N_IGN) == (16, 113, 77)
    assert (G.STEPS, G.BATCH, G.LR, G.LOG_EVERY, G.SEED) == (2000, 4, 2e-4, 100, 0)
    assert G.BARS == {"ap2m": 0.90, "prec": 0.90, "rec": 0.90, "count_rel": 0.10, "centre_p50_m": 0.30,
                      "size_p50_m": 0.30, "z_p50_m": 0.15, "cls_acc": 0.90, "presence_ratio": 0.25}
    assert G.PREREG_MD5 == "594c71196cc5bbd527fee40b2cb0e3f1" and G.FRAMESET_MD5 == "b291404c36f83c3e397e3b90367e8e7b"
    assert G.MUST_FAIL == {"memory_zeros": ("1",), "presence_w0": ("2", "3")}


def test_the_schedule_is_seeded_batches_of_4_covering_all_16_every_pass():
    s = G.batch_schedule()
    assert len(s) == 2000 and all(len(b) == 4 for b in s)
    for p in range(0, 2000, 4):
        assert sorted(sum(s[p:p + 4], [])) == list(range(16))
    assert s == G.batch_schedule() and s != G.batch_schedule(seed=1)


def _score(**kw):
    base = {"ap2m": 0.95, "prec": 0.95, "rec": 0.95, "n_conf": 113.0, "centre_p50_m": 0.1, "size_p50_m": 0.1,
            "z_p50_m": 0.05, "cls_acc": 0.95}
    base.update(kw)
    return base


def test_criteria_at_the_bars_and_nan_never_passes():
    assert G.criteria(_score(), presence_step0=1.0, presence_end=0.2, finite=True)["PASS"]
    assert G.criteria(_score(ap2m=0.90), presence_step0=1.0, presence_end=0.25, finite=True)["PASS"]
    for bad, k in ((dict(ap2m=0.899), "1"), (dict(rec=0.89), "2"), (dict(n_conf=125.0), "3"),
                   (dict(z_p50_m=0.16), "4"), (dict(cls_acc=float("nan")), "5")):
        c = G.criteria(_score(**bad), presence_step0=1.0, presence_end=0.2, finite=True)
        assert not c[k]["pass"] and not c["PASS"], k
    assert not G.criteria(_score(), presence_step0=1.0, presence_end=0.26, finite=True)["6"]["pass"]
    assert not G.criteria(_score(), presence_step0=1.0, presence_end=0.1, finite=False)["6"]["pass"]
    # the count window is |x - 113| / 113 <= 0.10: 102 and 124 pass, 101 and 125 fail
    assert G.criteria(_score(n_conf=102.0), presence_step0=1, presence_end=0, finite=True)["3"]["pass"]
    assert not G.criteria(_score(n_conf=101.0), presence_step0=1, presence_end=0, finite=True)["3"]["pass"]


def test_the_verdicts_and_VOID():
    ok = G.criteria(_score(), presence_step0=1.0, presence_end=0.2, finite=True)
    bad1 = G.criteria(_score(ap2m=0.3), presence_step0=1.0, presence_end=0.2, finite=True)
    assert G.arm_verdict("main", ok) == "PASS" and G.arm_verdict("main", bad1) == "FAIL"
    assert G.arm_verdict("memory_zeros", bad1) == "failed as required"
    assert G.arm_verdict("memory_zeros", ok).startswith("VOID")
    only3 = G.criteria(_score(n_conf=10.0), presence_step0=1.0, presence_end=0.2, finite=True)
    assert G.arm_verdict("presence_w0", only3).startswith("VOID")          # must fail 2 AND 3
    both = G.criteria(_score(n_conf=0.0, rec=0.0), presence_step0=1.0, presence_end=0.2, finite=True)
    assert G.arm_verdict("presence_w0", both) == "failed as required"


def _synthetic_packs(counts=(7,) * 15 + (8,)):
    """16 frames whose positives total 113 (15 x 7 + 8), one IGNORE row each, 30 slots."""
    rng = np.random.default_rng(0)
    out = []
    for c in counts:
        gt = np.stack([rng.uniform(3, 58, c + 1), rng.uniform(-14, 14, c + 1)], 1).astype(np.float32)
        n = 30
        out.append({"ep": 0, "logit": rng.normal(size=n).astype(np.float32),
                    "xy": rng.uniform(0, 60, (n, 2)).astype(np.float32), "cls": np.zeros(n, np.int16),
                    "cls_corr": np.zeros(n, np.int16), "matched": np.arange(n) < c, "exempt": np.zeros(n, bool),
                    "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0),
                    "gt_xy": gt, "gt_cls": np.zeros(c + 1, np.int16),
                    "pos": np.array([True] * c + [False]), "ign": np.array([False] * c + [True]),
                    "hidden": np.zeros(c + 1, bool)})
    return out


def test_controls_C1_reads_exactly_113_and_C2_reads_half():
    r = G.control_c1_c2(_synthetic_packs())
    assert r["C1"]["pass"] and r["C1"]["n_conf"] == 113.0 and r["C1"]["ap2m"] == 1.0
    assert r["C2"]["pass"] and r["C2"]["auroc_matched"] == 0.5
    # RED arm: one positive short -> C1 must fail on the count
    assert not G.control_c1_c2(_synthetic_packs((7,) * 16))["C1"]["pass"]


def test_control_C3_needs_per_frame_equality_totals_and_the_md5():
    fs = {"frames": [{"sha12": f"{i:012x}", "t": 21, "positive": 7, "ignore": 1, "dropped": 3} for i in range(15)]
          + [{"sha12": "f" * 12, "t": 64, "positive": 8, "ignore": 16, "dropped": 1}]}
    per = [(f["sha12"], f["t"], f["positive"], f["ignore"] + f["dropped"]) for f in fs["frames"]]
    ok = G.control_c3(per, fs, G.FRAMESET_MD5)
    assert ok["pass"] and ok["totals"] == [113, 77]
    assert not G.control_c3(per, fs, "0" * 32)["pass"]                        # wrong frame-set file
    per_bad = list(per)
    per_bad[0] = (per[0][0], per[0][1], per[0][2], per[0][3] - 3)              # the audit's DROPPED left out
    assert not G.control_c3(per_bad, fs, G.FRAMESET_MD5)["pass"]


def test_gpu_shared_with_names_a_concurrent_harness_and_never_itself(tmp_path):
    """``gpu_shared_with`` must list a SECOND harness process (the early MAIN and +R6 arms overlapped on Thor, and the
    first filter dropped every ``g_box_overfit.py`` line, so neither record named the other) and never its own pid."""
    def proc(pid, cmd):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
    proc(100, "/venv/bin/python tree/stack/scripts/g_box_overfit.py --launch-argv a.json --arms main")
    proc(200, "/venv/bin/python tree_r6/stack/scripts/g_box_overfit.py --launch-argv a_r6.json --arms main")
    proc(300, "/venv/bin/python /home/x/gmo_early_0327/gmo_early_runner.py code out")
    proc(400, "/venv/bin/python -m http.server")
    proc(500, "bash run_gbo.sh g_box_overfit.py")
    (tmp_path / "self").mkdir()
    got = G.other_gpu_jobs(proc_root=str(tmp_path), me=100)
    pids = [x.split(":")[0] for x in got]
    assert pids == ["pid 200", "pid 300"], got
    assert "g_box_overfit.py --launch-argv a_r6.json" in got[0]
    # RED arm: the first build's filter ("g_box_overfit.py" not in cmd) misses the concurrent harness
    legacy = [x for x in got if "g_box_overfit.py" not in x]
    assert [x.split(":")[0] for x in legacy] == ["pid 300"]
    assert G.other_gpu_jobs(proc_root=str(tmp_path / "absent"), me=1) == [f"<unreadable {tmp_path / 'absent'}>"]


def test_launch_argv_reads_the_canonical_object_and_a_plain_list(tmp_path):
    """A11: ``--launch-argv`` takes ``stack/ops/runs.d/<run>.argv.json`` (an object with "argv") and a plain list;
    both give the same argv and the same compact-JSON sha256 (the gate's form)."""
    import hashlib
    import json
    argv = ["--arm", "flat", "--out", "/x/y", "--trunk-compile", "--agent-join", "/data/a & b.jsonl"]
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps(argv), encoding="utf-8")
    obj = tmp_path / "refcv7-r101-s0.argv.json"
    obj.write_text(json.dumps({"run": "refcv7-r101-s0", "argv": argv, "note": "canonical"}, indent=2),
                   encoding="utf-8")
    a1, r1 = G.load_launch_argv(lst)
    a2, r2 = G.load_launch_argv(obj)
    assert a1 == a2 == argv
    want = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert r1["launch_argv_sha256"] == r2["launch_argv_sha256"] == want
    assert (r1["launch_argv_shape"], r2["launch_argv_shape"]) == ("list", "object['argv']")
    assert r1["launch_argv_file_sha256"] != r2["launch_argv_file_sha256"]
    # RED arm: the first build json-loaded the file and used it AS the argv -- on the canonical object that hashes
    # the whole object, and `"--trunk-compile" in <dict>` tests the KEYS, so it read "no compile"
    legacy = json.loads(obj.read_text(encoding="utf-8"))
    assert hashlib.sha256(json.dumps(legacy).encode()).hexdigest() != want
    assert "--trunk-compile" not in legacy and "--trunk-compile" in a2
    for bad in ({"args": argv}, {"argv": "--arm flat"}, {"argv": ["--arm", 3]}, "--arm flat"):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit):
            G.load_launch_argv(p)


def test_trunk_compile_is_dropped_for_the_build_and_declared():
    b, r = G.harness_build_argv(["--arm", "flat", "--trunk-compile", "--seed", "0"])
    assert b == ["--arm", "flat", "--seed", "0"]
    assert r["trunk_compile"] is False and r["trunk_compile_in_launch_argv"] is True
    assert "harness" in r["trunk_compile_why"]
    b2, r2 = G.harness_build_argv(b)
    assert b2 == b and r2["trunk_compile_in_launch_argv"] is False and r2["trunk_compile"] is False


def test_the_vendored_loader_is_the_audit_blob_byte_for_byte():
    """A11: the loader the harness imports is the audit's file with ONLY the provenance block added."""
    src = (ROOT / "tanitad" / "eval" / "refcv6_loader.py").read_bytes().decode("utf-8")
    assert G.vendored_audit_blob(src) == G.AUDIT_LOADER_BLOB == "11808258cd5c90647fae03f5859774b5541e9300"
    # RED arms: one character of the vendored code changed; the provenance block removed
    mut = src.replace("max_horizon=20", "max_horizon=21")
    assert mut != src and G.vendored_audit_blob(mut) != G.AUDIT_LOADER_BLOB
    with pytest.raises(SystemExit):
        G.vendored_audit_blob(src.split(G.VENDOR_END, 1)[1])


def test_the_harness_imports_the_vendored_loader_against_this_tree(monkeypatch, tmp_path):
    """The loader comes from stack/ (never ``--audit-dir``/code/), resolves THIS tree, and a copy imported earlier
    against another repo is refused (the module reads REFCV6_REPO at import)."""
    assert '"code" / "refcv6_loader.py"' not in (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")
    monkeypatch.setenv("REFCV6_REPO", str(tmp_path))
    monkeypatch.delitem(sys.modules, G.LOADER_MODULE, raising=False)
    try:
        L, info = G.load_vendored_loader()
        assert Path(info["file"]).resolve() == (ROOT / "tanitad" / "eval" / "refcv6_loader.py").resolve()
        assert info["audit_blob"] == G.AUDIT_LOADER_BLOB and Path(str(L.STACK)).resolve() == ROOT.resolve()
        # RED arm: the module already imported against another repo
        sys.modules.pop(G.LOADER_MODULE, None)
        monkeypatch.setenv("REFCV6_REPO", str(tmp_path))
        importlib.import_module(G.LOADER_MODULE)
        with pytest.raises(SystemExit, match="resolved the repo"):
            G.load_vendored_loader()
    finally:
        sys.modules.pop(G.LOADER_MODULE, None)


# --------------------------------------------------------------------------------------------------------- #
# the toy end-to-end: the loop separates a memorising head from the must-fail arm                          #
# --------------------------------------------------------------------------------------------------------- #
class _Toy:
    """16 frames x 3 targets, a 3-query 3-D slot decoder on frame-specific random memory.

    ⚠️ AS MANY QUERIES AS TARGETS, deliberately. MEASURED while building this test (raw/toy_presence_probe.log): with
    12 queries for 3-7 targets the presence of EVERY slot sat at the base-rate optimum (p_matched == p_unmatched
    ~0.30 focal, ~0.77-0.93 BCE) for 4,000 steps -- the Hungarian assignment never specialised the slots -- so a toy
    with surplus queries cannot tell the LOOP's correctness from DETR's slow query specialisation. With Q = A every
    slot is always matched, presence is trivial, and what remains is exactly what this test is for: the loop
    memorises the geometry, and the memory_zeros arm (one input for every frame) cannot."""

    def __init__(self):
        from tanitad.models import box3d_head as B3
        g = torch.Generator().manual_seed(1)
        self.B3 = B3
        self.mem = torch.randn(16, 8, 16, generator=g)
        A = 3
        box = torch.zeros(16, A, 4)
        box[..., 0] = torch.rand(16, A, generator=g) * 50 + 5
        box[..., 1] = torch.rand(16, A, generator=g) * 20 - 10
        box[..., 2] = 4.5
        box[..., 3] = 1.9
        v = torch.ones(16, A, dtype=torch.bool)
        self.tgt = {"box": box, "yaw": torch.zeros(16, A), "cls": torch.zeros(16, A, dtype=torch.long), "valid": v,
                    "occ": torch.full((16, A), -1.0), "rates": torch.zeros(16, A, 3),
                    "rates_mask": torch.zeros(16, A, dtype=torch.bool), "cz": torch.full((16, A), 0.8),
                    "h": torch.full((16, A), 1.6), "zh_mask": v.clone()}
        self.vis = {"n_full": torch.full((16, A), 1000, dtype=torch.int32),
                    "n_vis": torch.full((16, A), 900, dtype=torch.int32), "known": v.clone()}

    def setup(self, arm, seed):
        torch.manual_seed(seed)
        self.arm = arm
        self.dec = self.B3.Box3DSlotDecoder(16, 8, n_queries=3, d_model=64, depth=3, n_heads=4, enforce_band=False,
                                            presence_prior=0.01)
        self.dec.deep_supervision = True

    def params(self, arm):
        return list(self.dec.parameters())

    def optimizer(self, arm, params, lr=None):
        """A TOY optimiser (the toy is not the launch path): AdamW at the test's lr, no clip."""
        opt = torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=0.0)
        return opt, G.optimizer_spec(opt, None)

    def clip(self):
        raise AssertionError("the toy's spec carries no clip; run_arm must not call clip()")

    def _mem(self, idx):
        m = self.mem[idx]
        return torch.zeros_like(m) if self.arm == "memory_zeros" else m

    def _sub(self, idx):
        return ({k: v[idx] for k, v in self.tgt.items()}, {k: v[idx] for k, v in self.vis.items()})

    def loss(self, idx):
        from tanitad.models.slot_presence import refined_box3d_losses
        t, v = self._sub(idx)
        return refined_box3d_losses(self.dec(self._mem(idx)), t, presence_loss="focal", vis1=True, vis=v)["total"]

    def evaluate(self):
        from tanitad.eval.detection_metrics import window_packs
        from tanitad.models.slot_presence import refined_box3d_losses
        idx = list(range(16))
        with torch.no_grad():
            pred = self.dec(self._mem(idx))
            t, v = self._sub(idx)
            pres = float(refined_box3d_losses(pred, t, presence_loss="focal", vis1=True, vis=v)["loss_presence"])
            packs = window_packs(pred, t, v, presence_cost="focal")
        return packs, pres

    def teardown(self):
        pass


_T = None


def _trainer():
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location("refc_v3_train_for_gbo_test",
                                                      str(ROOT / "scripts" / "refc_v3_train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T = mod
    return _T


class _TinyLaunchModel(torch.nn.Module):
    """``core.encoder`` (the trunk prefix DD's groups split on) + head tensors."""

    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.encoder = torch.nn.Linear(4, 4)
        self.core.decoder = torch.nn.Linear(4, 2)
        self.box = torch.nn.Linear(2, 3)


def _launch_args(*extra):
    return _trainer().build_parser().parse_args(["--arm", "hier", "--size", "tiny", "--out", "X", "--opt", "dd",
                                                 "--lr", "1e-4", *extra])


def test_A13_the_harness_optimizer_EQUALS_what_the_trainer_builds_from_the_launch_argv():
    """SPEC_REFCV7 §18: the harness's groups, lrs, weight decay and clip EQUAL the trainer's own build from the
    launch argv's optimiser tokens (``--opt dd --lr 1e-4``, defaults for the rest); the constant-2e-4 config of the
    early runs must FAIL that equality (red arm)."""
    T = _trainer()
    m = _TinyLaunchModel()
    args = _launch_args()
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    want = G.optimizer_spec(T.build_optimizer(m, args), G.trainer_clip_literal(src))
    opt, got = G.launch_optimizer(T, m, args)
    assert G.spec_mismatches(got, want) == []
    # the values, as LITERALS (never an expression over the code under test)
    assert got["class"] == "AdamW" and got["clip"] == 10.0
    assert sorted(g["lr"] for g in got["groups"]) == [5e-05, 1e-04]
    assert [g["weight_decay"] for g in got["groups"]] == [1e-04, 1e-04]
    enc = [g for g in got["groups"] if g["lr"] == 5e-05][0]
    assert enc["param_ids"] == sorted(id(q) for q in m.core.encoder.parameters())
    # RED arm: the early runs' optimiser (every tensor at 2e-4, weight decay 0, no clip)
    old = G.optimizer_spec(torch.optim.AdamW(list(m.parameters()), lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
                                             weight_decay=0.0), None)
    mm = G.spec_mismatches(old, want)
    assert mm and any(x.startswith("clip") for x in mm) and any(x.startswith("groups") for x in mm), mm
    # and a one-group launch-lr spelling still fails on the partition
    one = G.optimizer_spec(torch.optim.AdamW(list(m.parameters()), lr=1e-4, weight_decay=1e-4), 10.0)
    assert G.spec_mismatches(one, want)


def test_A13_the_clip_literal_is_read_from_the_trainer_source():
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert G.trainer_clip_literal(src) == 10.0 == G.CLIP_NORM
    # RED arms: the trainer's clip changes -> the harness constant no longer matches; two clip sites -> refused
    call = "clip_grad_norm_(model.parameters(), 10.0)"
    assert src.count(call) == 1, "the anchor moved -- re-point the red arm"
    assert G.trainer_clip_literal(src.replace(call, "clip_grad_norm_(model.parameters(), 1.0)")) == 1.0
    with pytest.raises(SystemExit):
        G.trainer_clip_literal(src.replace(call, call + "; torch.nn.utils." + call))


def test_A13_run_arm_uses_the_adapters_optimizer_and_clips_before_every_step():
    calls = {"opt": 0, "clip": 0, "step": 0}

    class _A:
        def setup(self, arm, seed):
            self.w = torch.nn.Parameter(torch.ones(2))

        def params(self, arm):
            return [self.w]

        def optimizer(self, arm, params, lr=None):
            calls["opt"] += 1
            opt = torch.optim.SGD(params, lr=0.1)
            _step = opt.step

            def step(*a, **k):
                calls["step"] += 1
                assert calls["clip"] == calls["step"], "clip must run BEFORE the step it guards"
                return _step(*a, **k)
            opt.step = step
            return opt, G.optimizer_spec(opt, 10.0)

        def clip(self):
            calls["clip"] += 1

        def loss(self, idx):
            return (self.w ** 2).sum()

        def evaluate(self):
            return _synthetic_packs(), 1.0

        def teardown(self):
            pass
    res = G.run_arm(_A(), "main", steps=3, log_every=3)
    assert calls == {"opt": 1, "clip": 3, "step": 3}
    assert res["optimizer"]["clip"] == 10.0 and "param_ids" not in res["optimizer"]["groups"][0]


def test_A13_the_real_adapter_REFUSES_an_lr_override():
    import types
    fake = types.SimpleNamespace(model=torch.nn.Linear(2, 2), tr=None, args=None)
    with pytest.raises(SystemExit):
        G.TrainerAdapter.optimizer(fake, "main", [], lr=1e-3)


def test_A14_keep_share_and_the_one_frame_verdict_literals():
    assert G.ONE_FRAME_STEPS == 500 and G.ONE_FRAME_EVERY == 25
    assert G.ONE_FRAME_BARS == {"p_matched_median": 0.50, "assign_keep_final100": 0.80}
    assert G.keep_share(None, [{0: 1}]) is None and G.keep_share([{0: 1}], None) is None
    assert G.keep_share([{0: 5, 1: 7}], [{0: 5, 1: 8}]) == 0.5
    rows = [{"step": s, "keep": (0.8 if s > 400 else 0.0)} for s in range(25, 501, 25)]
    v = G.one_frame_verdict(rows, 0.5)
    assert v["PASS"] and v["ii_windows"] == 4 and v["ii_keep_final100"] == 0.8
    # RED arms: one bar missed at a time, and NaN never passes
    assert not G.one_frame_verdict(rows, 0.4999)["PASS"]
    rows_low = [dict(r, keep=0.79) if r["step"] > 400 else r for r in rows]
    assert not G.one_frame_verdict(rows_low, 0.9)["ii_pass"]
    assert not G.one_frame_verdict(rows, float("nan"))["i_pass"]
    assert not G.one_frame_verdict([dict(r, keep=None) for r in rows], 0.9)["ii_pass"]
    # A14.1: (ii) reads the ANCHOR key under HQS -- a churning slot index with a stable cell passes on the cell
    rows_hqs = [dict(r, keep=0.1, anchor_keep=(0.95 if r["step"] > 400 else 0.0)) for r in rows]
    assert not G.one_frame_verdict(rows_hqs, 0.7)["ii_pass"]
    v_cell = G.one_frame_verdict(rows_hqs, 0.7, key="anchor_keep")
    assert v_cell["ii_pass"] and v_cell["ii_key"] == "anchor_keep" and v_cell["PASS"]
    assert G.one_frame_index([("a", 1, 3, 0), ("b", 2, 13, 1), ("c", 3, 13, 0)]) == 1


class _ToyOneFrame(_Toy):
    """The toy with the one-frame protocol's seams: items / per_frame, a ``tr._perc.box3d_loss_row`` through which
    the loss runs (so the harness can capture the assignment), the A13-shaped optimiser, and a clip."""

    def __init__(self, route_through_row=True):
        super().__init__()
        import types
        from tanitad.models.slot_presence import refined_box3d_losses
        self.items = list(range(16))
        self.per_frame = [(f"{i:012x}", i, 3 if i != 5 else 4, 0) for i in range(16)]
        self.route = route_through_row
        self.tr = types.SimpleNamespace(_perc=types.SimpleNamespace(
            box3d_loss_row=lambda slots, tgt, **kw: {"loss": refined_box3d_losses(
                slots, tgt, presence_loss="focal", vis1=True, vis=kw["vis"])["total"]}))

    def optimizer(self, arm, params, lr=None):
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)
        return opt, G.optimizer_spec(opt, 10.0)

    def clip(self):
        torch.nn.utils.clip_grad_norm_(self.dec.parameters(), 10.0)

    def loss(self, idx):
        from tanitad.models.slot_presence import refined_box3d_losses
        real = [self.items[i] for i in idx]
        t, v = self._sub(real)
        pred = self.dec(self._mem(real))
        if self.route:
            return self.tr._perc.box3d_loss_row(pred, t, vis=v)["loss"]
        return refined_box3d_losses(pred, t, presence_loss="focal", vis1=True, vis=v)["total"]

    def evaluate(self):
        from tanitad.eval.detection_metrics import window_packs
        from tanitad.models.slot_presence import refined_box3d_losses
        real = list(self.items)
        with torch.no_grad():
            pred = self.dec(self._mem(real))
            t, v = self._sub(real)
            pres = float(refined_box3d_losses(pred, t, presence_loss="focal", vis1=True, vis=v)["loss_presence"])
            packs = window_packs(pred, t, v, presence_cost="focal")
        return packs, pres


def test_A14_run_one_frame_passes_a_memorising_toy_and_an_uncaptured_assignment_cannot():
    toy = _ToyOneFrame()
    r = G.run_one_frame(toy, "main", steps=200, every=25)
    assert r["frame_index"] == 5 and len(r["rows"]) == 8
    v = G.one_frame_verdict(r["rows"], r["rows"][-1]["p_matched_median"], steps=200, every=25)
    assert v["ii_windows"] == 4 and v["ii_keep_final100"] == 1.0, v
    assert v["i_p_matched_median"] > 0.5 and v["PASS"], v
    assert toy.items == list(range(16))                      # the frame list is restored
    # RED arm: a loss that bypasses box3d_loss_row leaves no captured assignment -> (ii) cannot pass
    r2 = G.run_one_frame(_ToyOneFrame(route_through_row=False), "main", steps=100, every=25)
    assert all(x["keep"] is None for x in r2["rows"])
    assert not G.one_frame_verdict(r2["rows"], 0.9, steps=100, every=25)["ii_pass"]


def test_TOY_the_loop_memorises_and_the_memory_zeros_arm_cannot():
    toy = _Toy()
    main = G.run_arm(toy, "main", steps=1200, log_every=600, lr=1e-3)
    zero = G.run_arm(toy, "memory_zeros", steps=1200, log_every=600, lr=1e-3)
    m, z = main["rows"][-1], zero["rows"][-1]
    assert m["ap2m"] > 0.8 and m["rec"] > 0.8, m
    assert z["ap2m"] < 0.5, z
    assert main["rows"][-1]["presence"] < main["rows"][0]["presence"]
    assert zero["verdict"] == "failed as required"
