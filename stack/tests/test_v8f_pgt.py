"""PREREG_V8F_PGT: the stack trained against its own upstream outputs, by the scheduled-sampling floor (`--p-gt-start 0.5 --p-gt-end 0.0`).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): `p_gt` has existed since v8F-mini's first
commit and the M2b arm used it (1.0 -> 0.2), but nothing pinned that the flag the PGT arm moves REACHES the two masks that read it. Pinned
here, hop by hop, as identities on the synthetic batch the other v8F-mini tests use:

  argv --p-gt-start/--p-gt-end -> cfg (make_config) -> built_levers / selfcheck / summary -> p_gt_schedule -> model.loss(b, p_gt=...)
  -> `_forward`'s masks: S -> T / S -> O (the checkpoint) and T -> O (the code).

  * p_gt = 0 : the code O receives equals T's PREDICTED code EXACTLY (torch.equal), and the checkpoint T and O receive equals S's own.
  * p_gt = 1 : the code O receives equals the GT code wherever the label exists, and T's code elsewhere; the checkpoint equals the GT
               checkpoint wherever the label is valid, S's own elsewhere.
  * p_gt in between (0.1 -- BELOW M2b's floor 0.2, so a hard-coded floor is visible -- and 0.5): the draws are replayed from the same
               generator, so the mask is the exact Bernoulli(p_gt) draw, in the documented order (S first, then T).
  * the trainer loop: a spy on `_forward` records the p_gt the masks actually received on every training step of a real `TR.main` run, and
               train_log.jsonl records the same numbers; the control run (M2b's defaults) reads the M2b schedule.

Each identity has a source mutation that turns it red: `raw/2026-10-22-pgt/mutate_pgt.py` (outputs beside it).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.models import v8f_mini as M  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
PGT_R = PKG / "code" / "v8f_pgt_read.py"
QUEUE = PKG / "code" / "v8f_mini_box2_queue.cmd"
CHAIN = PKG / "code" / "box2_ops" / "chain_v8f_pgt.sh"
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_pgt", Path(__file__).resolve().parent / "test_v8f_mini.py")
world, make_batch, tiny_model = _T.world, _T.make_batch, _T.tiny_model
TR = _T.TR
torch.set_num_threads(1)

PGT_ARGV = ["--arm", "M2b", "--p-gt-start", "0.5", "--p-gt-end", "0.0"]


# ---- the batch: mixed label availability, so "wherever the label exists" and "elsewhere" are both non-empty ---------------------------------
def _mixed_batch(world, n=16, seed=1):
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    torch.manual_seed(0)
    model, cfg = tiny_model(J, wm="off")
    b, _, _ = make_batch(world, n=n, seed=seed, hmax=cfg.hmax)
    B = b["feats"].shape[0]
    i = torch.arange(B)
    b["lab_t"]["has"] = (i % 3 != 0)                                           # T's label: 2 of 3 windows
    b["lab_s"]["has"] = (i % 4 != 1)                                           # S's label: 3 of 4 windows
    g = torch.Generator().manual_seed(123)
    b["lab_s"]["rc_ok"] = torch.rand(B, 3, generator=g) > 0.35                 # per-checkpoint validity
    return model, b


def _spy(model):
    """Record what T and O actually RECEIVE: T's checkpoint, O's checkpoint and O's code (the arguments of run_t / run_o), and T's own
    predicted code IN THAT SAME CALL (`code_t`): T's prediction depends on the checkpoint it read, so "T's code elsewhere" at p_gt > 0 is
    the code T predicted from the checkpoint it was actually given, not the one a separate p_gt = 0 forward would give."""
    cap = {}
    t0, o0 = model.run_t, model.run_o

    def run_t(bb, ad, last, ck_t, ck_ok):
        cap["ck_t"] = ck_t.detach().clone()
        tout = t0(bb, ad, last, ck_t, ck_ok)
        cap["code_t"] = model._code_pred(tout).detach().clone()
        return tout

    def run_o(bb, ad, last, ck_o, ck_ok, code):
        cap["ck_o"], cap["code_o"] = ck_o.detach().clone(), code.detach().clone()
        return o0(bb, ad, last, ck_o, ck_ok, code)
    model.run_t, model.run_o = run_t, run_o
    return cap


def _own_outputs(model, b):
    """S's own checkpoint and T's own predicted code on this batch: the label-free path (`_forward(b, 0.0, False)` -- what forward_infer
    calls), in the SAME mode (train, grad on) as the call under test. Not eval + no_grad: PyTorch's inference fast path changes the
    last bits of a transformer layer, and an identity is only an identity under identical numerics."""
    model.train()
    o = model._forward(b, 0.0, False)
    return o["ckpt"].detach().clone(), o["code"].detach().clone()


# ---- 1. the schedule -----------------------------------------------------------------------------------------------------------------------
def test_p_gt_schedule_values_and_the_dose_arithmetic():
    f = M.p_gt_schedule
    # PGT: 0.5 -> 0.0 over `total` steps; the trainer evaluates step = 0 .. total-1
    assert f(0, 1000, 0.5, 0.0) == 0.5
    assert f(500, 1000, 0.5, 0.0) == pytest.approx(0.25, abs=1e-12)
    assert f(1000, 1000, 0.5, 0.0) == 0.0 and f(5000, 1000, 0.5, 0.0) == 0.0, "clamped at the end value, never below it"
    assert f(999, 1000, 0.5, 0.0) == pytest.approx(0.5 / 1000, abs=1e-12), "the last step the loop evaluates"
    assert f(-3, 1000, 0.5, 0.0) == 0.5
    # M2b (the defaults) is unchanged: 1.0 -> 0.2
    assert f(0, 1000) == 1.0 and f(500, 1000) == pytest.approx(0.6) and f(1000, 1000) == pytest.approx(0.2)
    # the dose arithmetic PREREG_V8F_PGT commits (share of draws on which a layer reads its OWN upstream output = 1 - p_gt), at the 19,164 steps
    total = 19164

    def own(p0, p1, lo=0.0):
        ps = np.array([f(s, total, p0, p1) for s in range(total)])
        k = int(lo * total)
        return 1.0 - ps.mean(), 1.0 - ps[k:].mean()
    run, q4 = own(1.0, 0.2, 0.75)
    assert (round(run, 2), round(q4, 2)) == (0.40, 0.70), "M2b: 40 % over the run, 70 % in the last quarter"
    run, q4 = own(1.0, 0.0, 0.75)
    assert (round(run, 3), round(q4, 3)) == (0.5, 0.875), "the floor alone: 50 % / 87.5 %"
    run, q4 = own(0.5, 0.0, 0.75)
    assert (round(run, 2), round(q4, 4)) == (0.75, 0.9375), "PGT: 75 % over the run, about 94 % in the last quarter"
    assert f(0, total, 0.5, 0.0) == 0.5, "a warm start: at step 0 a layer still reads GT half of the time"


# ---- 2. the masks: p = 0 and p = 1 as identities --------------------------------------------------------------------------------------------
def test_o_reads_t_code_exactly_at_p0_and_the_gt_code_where_labelled_at_p1(world):
    model, b = _mixed_batch(world)
    ck_own, code_own = _own_outputs(model, b)
    has = b["lab_t"]["has"]
    gt_code = model._code_gt(b["lab_t"])
    assert has.any() and (~has).any(), "the batch must have windows with and without a T label"
    assert (gt_code[has] - code_own[has]).abs().max() > 1e-3, "non-vacuous: the GT code differs from T's own on labelled windows"
    cap = _spy(model)
    # p = 0: the code O receives IS T's predicted code, bit for bit, on every window (and it is the code a label-free forward gives)
    out0 = model(b, p_gt=0.0, gen=torch.Generator().manual_seed(7))
    assert torch.equal(cap["code_o"], cap["code_t"]), "p_gt = 0: O's code input != T's predicted code"
    assert torch.equal(cap["code_o"], code_own), "p_gt = 0: O's code input != the label-free forward's code"
    assert torch.equal(out0["code"], cap["code_t"])
    # p = 1: the GT code wherever the label exists, T's own code elsewhere (a window with no label has no GT code to read)
    out1 = model(b, p_gt=1.0, gen=torch.Generator().manual_seed(7))
    assert torch.equal(cap["code_o"][has], gt_code[has]), "p_gt = 1: O's code input != the GT code on labelled windows"
    assert torch.equal(cap["code_o"][~has], cap["code_t"][~has]), "p_gt = 1: an unlabelled window must keep T's own code"
    assert (cap["code_o"][has] - cap["code_t"][has]).abs().max() > 1e-3, "non-vacuous at p = 1"
    assert torch.equal(out1["code"], cap["code_o"])


def test_checkpoint_interfaces_s_to_t_and_s_to_o_at_p0_and_p1(world):
    """The same draw governs the checkpoint T reads and the checkpoint O reads (one mask, `use`), so the dose acts at THREE reads, not two."""
    model, b = _mixed_batch(world)
    ck_own, _ = _own_outputs(model, b)
    ls = b["lab_s"]
    use = ls["rc_ok"] & ls["has"][:, None]
    assert use.any() and (~use).any()
    cap = _spy(model)
    model(b, p_gt=0.0, gen=torch.Generator().manual_seed(7))
    for key, dst in (("ck_t", "pose_t"), ("ck_o", "pose_now")):
        assert torch.equal(cap[key], M.transform_ckpt(ck_own, b["pose_s"], b[dst])), f"p_gt = 0: {key} != S's own checkpoint"
    model(b, p_gt=1.0, gen=torch.Generator().manual_seed(7))
    for key, dst in (("ck_t", "pose_t"), ("ck_o", "pose_now")):
        gt_k = M.transform_ckpt(ls["rc"], b["pose_s"], b[dst])
        own_k = M.transform_ckpt(ck_own, b["pose_s"], b[dst])
        assert torch.equal(cap[key][use], gt_k[use]), f"p_gt = 1: {key} != the GT checkpoint where the label is valid"
        assert torch.equal(cap[key][~use], own_k[~use]), f"p_gt = 1: {key} must keep S's own checkpoint where the label is not valid"


# ---- 3. in between: the exact Bernoulli draw, in the documented order ---------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_mid_p_is_the_exact_bernoulli_draw_s_first_then_t(world, p):
    """p = 0.1 sits BELOW M2b's floor 0.2: a floor hard-coded at M2b's value is only visible here."""
    model, b = _mixed_batch(world, n=40)
    ck_own, code_own = _own_outputs(model, b)
    B = b["feats"].shape[0]
    for seed in range(11, 200):                                                  # the first generator seed whose draw is non-trivial at this p
        g = torch.Generator().manual_seed(seed)
        r_s, r_t = torch.rand(B, generator=g), torch.rand(B, generator=g)       # the order in `_forward`: S's draw, then T's
        use_t = (r_t < p) & b["lab_t"]["has"]
        use_s = (r_s < p)[:, None] & b["lab_s"]["rc_ok"] & b["lab_s"]["has"][:, None]
        if use_t.sum() >= 2 and (~use_t & b["lab_t"]["has"]).sum() >= 2 and use_s.sum() >= 2 and (~use_s).sum() >= 2:
            break
    else:
        pytest.fail("no non-trivial draw found")
    cap = _spy(model)
    model(b, p_gt=p, gen=torch.Generator().manual_seed(seed))
    gt_code = model._code_gt(b["lab_t"])
    assert torch.equal(cap["code_o"][use_t], gt_code[use_t]) and torch.equal(cap["code_o"][~use_t], cap["code_t"][~use_t]), "T -> O: not the Bernoulli(p) draw"
    for key, dst in (("ck_t", "pose_t"), ("ck_o", "pose_now")):
        gt_k = M.transform_ckpt(b["lab_s"]["rc"], b["pose_s"], b[dst])
        own_k = M.transform_ckpt(ck_own, b["pose_s"], b[dst])
        assert torch.equal(cap[key][use_s], gt_k[use_s]) and torch.equal(cap[key][~use_s], own_k[~use_s]), f"S -> {key[-1].upper()}: not the Bernoulli(p) draw"


# ---- 4. argv -> cfg -> built -> the gate --------------------------------------------------------------------------------------------------
def test_argv_reaches_cfg_and_the_built_record_and_a_lie_is_refused(world):
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    a = _T._args(PGT_ARGV)
    cfg = TR.make_config(a)
    assert (cfg.p_gt_start, cfg.p_gt_end) == (0.5, 0.0), "the flags reach the config"
    d = TR.make_config(_T._args(["--arm", "M2b"]))
    assert (d.p_gt_start, d.p_gt_end) == (1.0, 0.2), "M2b's defaults: the PGT arm moves two flags, nothing else"
    # nothing else moves: every other config field equals M2b's
    assert {k: v for k, v in cfg.__dict__.items() if k not in ("p_gt_start", "p_gt_end")} == \
        {k: v for k, v in d.__dict__.items() if k not in ("p_gt_start", "p_gt_end")}
    cfg, model, opt, info = _T._built(a, J)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    bl = model.built_levers()
    assert (bl["p_gt_start"], bl["p_gt_end"]) == (0.5, 0.0), "selfcheck.json / summary.json record the BUILT schedule"
    assert model.cfg is cfg or (model.cfg.p_gt_start, model.cfg.p_gt_end) == (0.5, 0.0)
    # declared 0.5 -> 0.0 but built M2b's 1.0 -> 0.2 : each half is refused on its own
    for over, key in ((dict(p_gt_start=1.0), "p_gt_start"), (dict(p_gt_end=0.2), "p_gt_end")):
        cfg, model, opt, info = _T._built(a, J, **over)
        assert any(key in m for m in TR.declared_vs_built(a, cfg, model, opt, info)), key
    assert TR.FLAG_CHECKS.get("p_gt_start") == "cfg:p_gt_start" and TR.FLAG_CHECKS.get("p_gt_end") == "cfg:p_gt_end"
    assert "M2b-pgt" not in TR.ARM_SPECS, "M2b-pgt is M2b + two flags; the arm table is unchanged"
    # the M2b control records M2b's schedule
    m2b_cfg, m2b_model, _, _ = _T._built(_T._args(["--arm", "M2b"]), J)
    assert (m2b_model.built_levers()["p_gt_start"], m2b_model.built_levers()["p_gt_end"]) == (1.0, 0.2)


# ---- 5. the train loop: the schedule the masks actually receive -----------------------------------------------------------------------------
def _train(out, extra, monkeypatch):
    seen = []
    orig = M.V8FMini._forward

    def spy(self, b, p_gt, use_labels, gen=None, code_override=None):
        if use_labels:
            seen.append(p_gt)
        return orig(self, b, p_gt, use_labels, gen, code_override) if code_override is not None else orig(self, b, p_gt, use_labels, gen)
    monkeypatch.setattr(M.V8FMini, "_forward", spy)
    common = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "8", "--log-every", "1", "--warmup", "2",
              "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5"]
    assert TR.main(["--out", str(out), *common, *extra]) == 0
    monkeypatch.setattr(M.V8FMini, "_forward", orig)
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return seen, log, json.load(open(out / "selfcheck.json", encoding="utf-8")), json.load(open(out / "summary.json", encoding="utf-8"))


def test_the_train_loop_hands_the_schedule_to_the_masks(tmp_path, monkeypatch):
    seen, log, sc, sm = _train(tmp_path / "pgt", PGT_ARGV, monkeypatch)
    want = [M.p_gt_schedule(k, 8, 0.5, 0.0) for k in range(8)]
    assert seen[-8:] == want, "the p_gt `_forward` (the masks) received on the 8 training steps is not the 0.5 -> 0.0 schedule"
    assert [r["p_gt"] for r in log] == want and [r["step"] for r in log] == list(range(1, 9)), "train_log.jsonl: step k logs schedule(k - 1)"
    assert want[0] == 0.5 and want[-1] == pytest.approx(0.5 / 8)
    for rec in (sc["levers"], sm["levers"]):
        assert (rec["p_gt_start"], rec["p_gt_end"]) == (0.5, 0.0), "selfcheck.json and summary.json record the BUILT schedule"
    assert sc["declared_vs_built"] == "PASS" and sm["arm"] == "M2b"
    # CONTROL: M2b's defaults, same loop, must read the M2b schedule -- the spy is not a constant
    seen2, log2, sc2, _ = _train(tmp_path / "m2b", ["--arm", "M2b"], monkeypatch)
    want2 = [M.p_gt_schedule(k, 8) for k in range(8)]
    assert seen2[-8:] == want2 and [r["p_gt"] for r in log2] == want2 and want2 != want
    assert (sc2["levers"]["p_gt_start"], sc2["levers"]["p_gt_end"]) == (1.0, 0.2)


# =================================================================================================================================================
# THE READER (code/v8f_pgt_read.py): a synthetic straight-road world whose answers are analytic (test_v8f_dk.World), real files on disk, the real
# `main`. The banked premises (F, LAMBDA, the S5 excess) are the REAL banked M2b values and cannot hold on synthetic data: the tests that are not
# about them replace `known_values`; one test proves the real one refuses; the real-data proofs (raw/2026-10-22-pgt) run it unpatched.
# =================================================================================================================================================
needs_pkg = pytest.mark.skipif(not PGT_R.exists(), reason="the v8F-mini package is not in this checkout")
R = _load("v8f_pgt_read_under_test", PGT_R) if PGT_R.exists() else None          # importable without the package (the lander's stack-only probe)
_DK = _load("test_v8f_dk_for_pgt", Path(__file__).resolve().parent / "test_v8f_dk.py") if PGT_R.exists() else None
C0, NC = M.N_GOAL + M.N_LAT + M.N_LON, len(M.CONS_NAMES)
LATCONS = [C0 + i for i in range(12)]                                            # CODE_GROUPS["latcons"] (pinned against the model below)
M2B_ARGV = ["--arm", "M2b", "--seed", "{s}", "--grid", "pool4x10", "--out", "D:\\runs\\M2b_s{s}", "--kin-mode", "yaw", "--exclude", "540:600",
            "--batch", "64", "--steps-mult", "4", "--device", "cuda"]


def _schedule(step, total, p0, p1):                                              # independent of the model's p_gt_schedule
    return p0 + (p1 - p0) * min(max(step / total, 0.0), 1.0)


def _pgt_files(d, per_seed=None, sp_m2b=11.0, sp_pgt=10.4, off_m2b=1.0, d_ade=-0.7, gap=0.65, lam_m2b=0.27, lam_pgt=0.10, err_m2b=0.6, err_pgt=0.5,
               steps_pgt=19164, argv_pgt=("0.5", "0.0"), built=(0.5, 0.0), sc="PASS", m2b_flag=False, m2b_built=None, extra_argv=(), drop=None,
               hold=HOLD, clip="t" * 64, clip_pgt=None, done=True, n_clips=4512, argv_arm="M2b", seeds=(0, 1), gt_shift=0.0,
               log_shift=0.0, no_log=False, no_built=False, states=None, lead_shift_noise=5.0, old_m2b_orc=True, shuffle_eid=False,
               orc_mode="oracle", lat_gt_wrong=False, abs_mode="absmask", cure=0,
               sm_arm="M2b", sm_seed_shift=0, built_sm=None, built_sc=None, no_built_only=None, empty_log=False, omit=None, abs_no_code=False, block_k=None):
    """Synthetic eval + runs for M2b (banked) and M2b-pgt, both seeds. The knobs: `sp` plan speed -> the low-TTC tail, `lam` the 2 s forward shift of the
    all-latcons plan -> the leak, `err` T's present-slot error (all per arm); `d_ade` the target ADE6 difference M2b-pgt - M2b (the PGT plan's lateral
    deviation after 2 s is SOLVED for it, so ADE6 is steered independently of the speed that sets the tail); `gap` M2b's seed-1 offset (-> F).
    `per_seed={1: {"pgt_d_ade": 0.8}}` overrides one knob of one arm at one seed."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    w = _DK.World(d, states=states, **({} if block_k is None else dict(block_k=block_k)))
    N, eid, gt = w.n, w.eid, w.gt
    lead = np.array([s == "LEAD" for s in w.states])
    rng = np.random.default_rng(5)
    g = np.zeros((N, M.CODE_DIM))
    g[:, :C0] = rng.normal(0, 1, (N, C0))
    present = rng.random((N, NC)) < 0.45
    g[:, C0:] = np.where(present, rng.normal(0, 1, (N, NC)), 0.0)
    fb = np.zeros(N, bool)
    fb[:5] = True
    ev, ru = d / "eval", d / "runs"
    ev.mkdir(exist_ok=True)
    ru.mkdir(exist_ok=True)

    def ade(p):
        return np.linalg.norm(p.astype(np.float32).astype(np.float64) - gt, axis=-1).mean(1)

    def std_plan(sp, off, noise):
        p = _DK.path_at(np.full(N, sp), N)
        p[:, 10:, 1] += (off * (1.0 + 0.05 * noise))[:, None]        # a lateral deviation BEYOND 2 s: ADE6 moves, the K = 10 lead metrics do not
        return p
    knobs = dict(sp=(sp_m2b, sp_pgt), lam=(lam_m2b, lam_pgt), err=(err_m2b, err_pgt), d_ade=(None, d_ade))

    def solve_off(sp, nz, target):                                              # the lateral offset whose mean ADE6 is `target` (monotone in the offset)
        lo, hi = 0.0, 40.0
        if ade(std_plan(sp, lo, nz)).mean() >= target:
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ade(std_plan(sp, mid, nz)).mean() < target else (lo, mid)
        return 0.5 * (lo + hi)
    for arm_i, arm in enumerate(("M2b", "M2b-pgt")):
        for s in (0, 1):
            if arm == "M2b-pgt" and s not in seeds:
                continue
            kn = {k: v[arm_i] for k, v in knobs.items()}
            prefix = "pgt_" if arm_i else "m2b_"
            for k, v in ((per_seed or {}).get(s) or {}).items():
                if k.startswith(prefix):
                    kn[k[len(prefix):]] = v
            tag = f"{arm}_s{s}"
            is_p = arm == "M2b-pgt"
            nz = np.random.default_rng(100 + s).normal(0, 1, N)                 # the SAME per-window noise for both arms: a paired design
            if is_p:
                m2b_sp = ((per_seed or {}).get(s) or {}).get("m2b_sp", sp_m2b)
                ref = ade(std_plan(m2b_sp, off_m2b + gap * s, nz)).mean()
                pred = std_plan(kn["sp"], solve_off(kn["sp"], nz, ref + (kn["d_ade"] if kn["d_ade"] is not None else 0.0)), nz)
                pred[:cure, :10, 1] += 5.0                               # these windows: the PGT arm swerves 5 m clear of the lead (LEAD_LAT_M 2.0): no admissible step, min-TTC undefined
            else:
                pred = std_plan(kn["sp"], off_m2b + gap * s, nz)
            orc = _DK.path_at(np.full(N, 10.0), N)
            lam_w = kn["lam"] * (1.0 + 0.3 * np.random.default_rng(7 + s).normal(0, 1, N))
            lam_w = np.where(lead, lam_w, lead_shift_noise if is_p else 0.0)     # NON-lead windows of the PGT arm carry a huge shift that must never be read
            lat = orc.copy()
            lat[:, :, 0] += lam_w[:, None] * np.minimum(1.0, (np.arange(30) + 1) / 10.0)[None]
            ev_eid = eid[::-1].copy() if (shuffle_eid and is_p and s == 0) else eid
            oeid = eid[::-1].copy() if (shuffle_eid == 'world' or (shuffle_eid == 'all' and is_p and s == 0)) else eid       # the eid of every OTHER file
            ev_eid = oeid if shuffle_eid in ('world', 'all') else ev_eid
            if drop != tag:
                std = dict(pred=pred.astype(np.float32), ade6=ade(pred), eid=ev_eid)
                if is_p and omit:
                    std.pop(omit)
                np.savez(ev / f"eval_{tag}.npz", **std)
                json.dump({"raw": {"traj": {"ade6": float(ade(pred).mean())}, "tactical": {"acc": 0.5}, "strategic": {"acc": 0.5}, "families": {}}},
                          open(ev / f"eval_{tag}.json", "w", encoding="utf-8"))
            tc = g.copy()
            tc[:, C0:] = np.where(present, g[:, C0:] + np.random.default_rng(31 + s + arm_i).normal(0, kn["err"], (N, NC)), 0.0)
            gg = g.copy()
            if is_p:
                gg[:, C0:] += gt_shift * present
            tabs = dict(pred=pred.astype(np.float32), ade6=ade(pred), eid=oeid, tac_code=tc, tac_oracle_fallback=fb)
            if is_p and abs_no_code:
                tabs.pop("tac_code")
            np.savez(ev / f"eval_{tag}-tacabs.npz", **tabs)
            json.dump({"tac_manipulation": {"mode": abs_mode}}, open(ev / f"eval_{tag}-tacabs.json", "w", encoding="utf-8"))
            if arm == "M2b" and old_m2b_orc:                                      # the banked 2026-10-20 format: no tac_code (WC Amendment 1)
                np.savez(ev / f"eval_{tag}-tacorc.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=oeid, tac_oracle_fallback=fb)
                ag = gg.copy()
                ag[:, :22] = np.random.default_rng(99).normal(0, 1, (N, 22))
                np.savez(ev / f"eval_{tag}-tacorc-all-goal.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=oeid, tac_code=ag, tac_oracle_fallback=fb)
            else:
                np.savez(ev / f"eval_{tag}-tacorc.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=oeid, tac_code=gg, tac_oracle_fallback=fb)
            json.dump({"tac_manipulation": {"mode": orc_mode}}, open(ev / f"eval_{tag}-tacorc.json", "w", encoding="utf-8"))
            lc = gg.copy()
            lc[:, LATCONS] = tc[:, LATCONS] + 0.01                               # T's own latcons in place of the GT ones, GT elsewhere
            if lat_gt_wrong:
                lc[:, C0 + 14] += 1.0                                            # a non-latcons constraint slot that is NOT the GT
            np.savez(ev / f"eval_{tag}-tacorc-all-latcons.npz", pred=lat.astype(np.float32), ade6=ade(lat), eid=oeid, tac_code=lc, tac_oracle_fallback=fb)
            json.dump({"tac_manipulation": {"mode": "oracle"}}, open(ev / f"eval_{tag}-tacorc-all-latcons.json", "w", encoding="utf-8"))
            # ---- the run
            rd = ru / tag
            rd.mkdir(parents=True, exist_ok=True)
            steps = steps_pgt if is_p else 19164
            lev_by = {}
            for where in ("summary", "selfcheck"):
                b0, b1 = ((built_sm if where == "summary" else built_sc) or built) if is_p else (m2b_built or (None, None))
                lev_by[where] = {"wm": "b", "w": {"w_pen": 0.5}}
                if (b0 is not None) and not (is_p and (no_built or no_built_only == where)):
                    lev_by[where].update(p_gt_start=b0, p_gt_end=b1)
            json.dump({"arm": sm_arm if is_p else "M2b", "seed": s + (sm_seed_shift if is_p else 0), "steps": steps, "done": done, "n_clips": n_clips,
                       "train_sha256": (clip_pgt or clip) if is_p else clip, "excluded_sha256": hold, "levers": lev_by["summary"]},
                      open(rd / "summary.json", "w", encoding="utf-8"))
            av = [x.replace("{s}", str(s)) for x in M2B_ARGV]
            if is_p:
                av[1] = argv_arm
                av += ["--p-gt-start", argv_pgt[0], "--p-gt-end", argv_pgt[1], *extra_argv]
                av[av.index("--out") + 1] = f"D:\\runs\\M2b-pgt_s{s}"
            elif m2b_flag:
                av += ["--p-gt-end", "0.0"]
            json.dump({"argv": av, "sha": "x"}, open(rd / "argv.json", "w", encoding="utf-8"))
            json.dump({"declared_vs_built": sc if is_p else "PASS", "levers": lev_by["selfcheck"]}, open(rd / "selfcheck.json", "w", encoding="utf-8"))
            p0, p1 = (0.5, 0.0) if is_p else (1.0, 0.2)
            if empty_log and is_p:
                (rd / "train_log.jsonl").write_text("", encoding="utf-8")
            elif not (no_log and is_p):
                with open(rd / "train_log.jsonl", "w", encoding="utf-8") as f:
                    for st in range(957, steps + 1, 957):
                        f.write(json.dumps({"step": st, "p_gt": _schedule(st - 1, steps, p0, p1) + (log_shift if is_p else 0.0), "ade": 1.0}) + "\n")
    np.save(d / "p10.npy", _DK.path_at(np.full(N, 10.0), N))
    w.write_gt()
    w.write_block()
    return {"d": d, "w": w, "eval": ev, "runs": ru, "lead": lead,
            "args": ["--eval-dir", str(ev), "--runs-dir", str(ru), "--out", str(d / "pgt.json"), "--n-boot", "300", "--dk-n-boot", "60", "--no-families",
                     "--no-k30", "--lead-block", str(d / "lead.npz"), "--p10-pred", str(d / "p10.npy"), "--gt-npz", str(d / "gt.npz")]}


def _go(f, stage="final", extra=()):
    rc = R.main(["--stage", stage, *f["args"], *extra])
    out = f["d"] / "pgt.json"
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


@pytest.fixture
def no_known(monkeypatch):
    """The banked premises (F 0.1207, LAMBDA +0.2678 / +0.1667, S5 excess +4.42 / +2.50 pp) are the real banked M2b values; a synthetic world cannot
    hold them. The tests of everything else replace the check; `test_reader_identity_control_known_values_and_check_inputs` and the real-data proofs
    run the real one."""
    monkeypatch.setattr(R, "known_values", lambda *a, **k: {"checked": False, "why": "synthetic test input"})


@needs_pkg
def test_reader_layout_and_constants_match_the_model_and_the_trainer():
    DK, WC = R._modules()
    assert (WC.C0, WC.N_CONS) == (C0, NC) == R._WC_LAYOUT and WC.GROUPS["latcons"] == tuple(range(12))
    assert [C0 + i for i in WC.GROUPS["latcons"]] == list(M.CODE_GROUPS["latcons"]) == LATCONS
    assert (R.PGT_START, R.PGT_END, R.M2B_START, R.M2B_END) == (0.5, 0.0, 1.0, 0.2)
    d = TR.build_parser().parse_args(["--out", "x"])
    assert (d.p_gt_start, d.p_gt_end) == (R.M2B_START, R.M2B_END), "the reader's M2b dose is the trainer's default"
    assert R.IDX_2S == 9 and DK.DT == 0.2 and DK.DT * (R.IDX_2S + 1) == pytest.approx(2.0), "pred[:, 9] is t + 2 s"
    assert R.HOLDOUT_SHA == WC.HOLDOUT_SHA and R.N_TRAIN == WC.N_TRAIN
    # the PREREG's banked premises (F 0.1207 m; M2b's leak +0.268 / +0.167 m; its S5 excess over P10 +4.4 / +2.5 pp), to the PREREG's own rounding
    assert R.KNOWN["F"] == 0.1207 and R.KNOWN["lambda"] == pytest.approx({0: 0.268, 1: 0.167}, abs=5e-4) and R.KNOWN["excess"] == pytest.approx({0: 0.044, 1: 0.025}, abs=5e-4)
    assert R.schedule(5, 10, 0.5, 0.0) == pytest.approx(0.25) and R.schedule(0, 10, 1.0, 0.2) == 1.0 and R.schedule(99, 10, 0.5, 0.0) == 0.0


@needs_pkg
def test_reader_decision_table_final_stage(tmp_path, no_known):
    rc, r = _go(_pgt_files(tmp_path / "ok"))
    assert rc == 0 and r["VERDICT"] == "SUCCESS" and r["mechanism"] is True and r["DECISION"] is None, r["reading"]
    f = r["F"]
    d = np.load(tmp_path / "ok" / "eval" / "eval_M2b_s0.npz")["ade6"].mean() - np.load(tmp_path / "ok" / "eval" / "eval_M2b_s1.npz")["ade6"].mean()
    assert f == pytest.approx(abs(d), abs=1e-12) and r["two_F"] == pytest.approx(2 * f), "F is the banked M2b pair's gap, from BOTH evals"
    for s in ("s0", "s1"):
        row = r["rows"][s]
        assert row["primary"]["point"] < -2 * f and row["primary"]["separated"] and row["judge"]["passed"]
        assert row["mleak"]["delta"]["point"] < 0 and row["mdk"]["delta"]["point"] < 0
        assert row["judge"]["mleak"]["met"] and row["judge"]["mdk"]["met"]
        assert row["beside"]["present_slot"]["rmse"]["M2b-pgt"] < row["beside"]["present_slot"]["rmse"]["M2b"]
    # PRIMARY beyond 0 but under 2F, mechanism met -> FAIL-MECHANISM-OK, and the committed next lever is named
    rc, r = _go(_pgt_files(tmp_path / "small", d_ade=-0.12))
    assert rc == 0 and r["VERDICT"] == "FAIL-MECHANISM-OK" and "pool8x20" in r["next"] and r["mechanism"] is True
    assert r["rows"]["s0"]["primary"]["separated"] and 0 < abs(r["rows"]["s0"]["primary"]["point"]) < r["two_F"], "beyond 0, under 2F"
    # neither the leak nor the tail halved -> FAIL-NO-MECHANISM, whatever the ADE did
    rc, r = _go(_pgt_files(tmp_path / "nomech", d_ade=-0.12, lam_pgt=0.25, sp_pgt=10.95))
    assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM" and "O's longitudinal head" in r["next"] and r["mechanism"] is False
    # driving better beyond 2F at both seeds is SUCCESS even where the mechanism read is not met (the table: SUCCESS = both seeds pass)
    rc, r = _go(_pgt_files(tmp_path / "succ_nomech", lam_pgt=0.25))
    assert rc == 0 and r["VERDICT"] == "SUCCESS" and r["mechanism"] is False
    # worse driving is never a pass
    rc, r = _go(_pgt_files(tmp_path / "worse", d_ade=0.8))
    assert rc == 0 and r["VERDICT"] != "SUCCESS" and not r["rows"]["s0"]["judge"]["passed"]
    # one seed is not enough
    rc, r = _go(_pgt_files(tmp_path / "oneseed", per_seed={1: {"pgt_d_ade": 0.8}}))
    assert rc == 0 and r["VERDICT"] != "SUCCESS" and r["rows"]["s0"]["judge"]["passed"] and not r["rows"]["s1"]["judge"]["passed"]


@needs_pkg
def test_reader_mechanism_needs_both_symptoms_and_the_size_bar_is_half(tmp_path, no_known):
    # LEAK met, TAIL not: the mechanism is NOT met
    rc, r = _go(_pgt_files(tmp_path / "tail_missing", d_ade=-0.12, sp_pgt=10.95))
    j = r["rows"]["s0"]["judge"]
    assert rc == 0 and j["mleak"]["met"] and not j["mdk"]["met"] and j["mechanism"] is False and r["VERDICT"] == "FAIL-NO-MECHANISM"
    rc, r = _go(_pgt_files(tmp_path / "leak_missing", d_ade=-0.12, lam_pgt=0.25))
    j = r["rows"]["s0"]["judge"]
    assert rc == 0 and not j["mleak"]["met"] and j["mdk"]["met"] and j["mechanism"] is False
    # the size bar is HALF of M2b's symptom: a separated reduction just under half does not count, just over does
    rc, r = _go(_pgt_files(tmp_path / "under", lam_pgt=0.27 * 0.56))
    m = r["rows"]["s0"]["mleak"]["delta"]
    assert m["separated"] and m["point"] < 0 and abs(m["point"]) < 0.5 * r["rows"]["s0"]["base"]["lambda_m2b_m"] and not r["rows"]["s0"]["judge"]["mleak"]["met"]
    rc, r = _go(_pgt_files(tmp_path / "over", lam_pgt=0.27 * 0.44))
    m = r["rows"]["s0"]["mleak"]["delta"]
    assert abs(m["point"]) >= 0.5 * r["rows"]["s0"]["base"]["lambda_m2b_m"] and r["rows"]["s0"]["judge"]["mleak"]["met"]
    # the same bar for the tail: a reduction under half of M2b's excess over P10 does not count
    rc, r = _go(_pgt_files(tmp_path / "tail_under", sp_pgt=10.9))
    m = r["rows"]["s0"]["mdk"]["delta"]
    assert m["point"] < 0 and abs(m["point"]) < 0.5 * r["rows"]["s0"]["base"]["excess_p10"] and not r["rows"]["s0"]["judge"]["mdk"]["met"]
    # a symptom M2b does not have leaves nothing to halve: NOT-APPLICABLE, counted as not met, said so
    rc, r = _go(_pgt_files(tmp_path / "noleak", lam_m2b=0.0, lam_pgt=0.0))
    j = r["rows"]["s0"]["judge"]
    assert rc == 0 and j["mleak"]["status"] == "NOT-APPLICABLE" and j["mleak"]["met"] is False and "no symptom to halve" in j["mleak"]["reason"]


@needs_pkg
def test_reader_lambda_reads_only_lead_windows_and_is_the_2s_along_track_difference(tmp_path, no_known):
    states = {0: "NO_LEAD", 1: "NOT_STRAIGHT", 2: "NO_LABEL", 3: "NO_LABEL", 7: "NO_LEAD", 20: "NO_LEAD"}
    f = _pgt_files(tmp_path / "nl", states=states)
    rc, r = _go(f)
    assert rc == 0 and r["n_lead_windows_k10"] == int(f["lead"].sum()) == 72 - len(states)
    # the PGT arm's NON-lead windows carry a +5 m shift: reading them would turn the leak POSITIVE; the LEAD windows say it fell
    assert r["rows"]["s0"]["mleak"]["delta"]["point"] < 0 and r["rows"]["s0"]["mleak"]["lambda_a_m"] < r["rows"]["s0"]["mleak"]["lambda_b_m"]
    # LAMBDA by hand: along-track(2 s) of all-latcons minus that of tacorc, on the LEAD windows
    ev = f["eval"]
    for arm, key in (("M2b-pgt", "lambda_a_m"), ("M2b", "lambda_b_m")):
        lat = np.load(ev / f"eval_{arm}_s0-tacorc-all-latcons.npz")["pred"][:, 9, 0].astype(np.float64)
        orc = np.load(ev / f"eval_{arm}_s0-tacorc.npz")["pred"][:, 9, 0].astype(np.float64)
        assert r["rows"]["s0"]["mleak"][key] == pytest.approx(float((lat - orc)[f["lead"]].mean()), abs=1e-9), arm


@needs_pkg
def test_reader_mdk_is_s5_of_the_k10_lead_windows_and_the_bar_is_half_of_the_p10_excess(tmp_path, no_known):
    f = _pgt_files(tmp_path / "s5")
    rc, r = _go(f)
    g0 = f["w"].g0

    def s5(v):      # analytic: ego at v, lead at 8 m/s, K = 10 -> min-TTC = (g0 + 0.2 (8 - v) 9) / (v - 8); S5 = share with TTC < 5 s
        return float(((g0 + 0.2 * (8.0 - v) * 9) / (v - 8.0) < 5.0).mean())
    row = r["rows"]["s0"]
    assert row["mdk"]["delta"]["share_a"] == pytest.approx(s5(10.4), abs=1e-9) and row["mdk"]["delta"]["share_b"] == pytest.approx(s5(11.0), abs=1e-9)
    assert row["mdk"]["delta"]["point"] == pytest.approx(s5(10.4) - s5(11.0), abs=1e-9)
    assert row["base"]["excess_p10"] == pytest.approx(s5(11.0) - s5(10.0), abs=1e-9), "M2b's excess over P10 (P10 plans at the human's 10 m/s)"
    assert row["judge"]["mdk"]["half_of_m2b"] == pytest.approx(0.5 * (s5(11.0) - s5(10.0)), abs=1e-9)
    # the distance-keeping family rides beside (deciding nothing): headway / time-gap / TTC / LEAD-VIOL with its progress control, vs M2b and vs P10
    dk = row["beside"]["distance_keeping_k10"]
    assert {"headway_min_m", "time_gap_min_s", "min_ttc_s"} <= set(dk["pair_vs_m2b"]["dk"]) and dk["pair_vs_m2b"]["lead_viol"]["at_risk"] is not None
    assert "progress" in dk["arms"]["M2b-pgt_s0"]["lead_viol"] and dk["pair_vs_p10"] is not None


@needs_pkg
def test_reader_stage_s0_continues_or_stops_on_seed_0_alone(tmp_path, no_known):
    rc, r = _go(_pgt_files(tmp_path / "go", seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "CONTINUE" and r["VERDICT"] is None and "train seed 1" in r["next"] and list(r["rows"]) == ["s0"], r["reading"]
    assert r["two_F"] > 0, "F still comes from BOTH banked M2b evals at stage s0"
    rc, r = _go(_pgt_files(tmp_path / "stop_mech", d_ade=-0.12, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-MECHANISM-OK" and "pool8x20" in r["next"]
    rc, r = _go(_pgt_files(tmp_path / "stop_nomech", d_ade=-0.12, lam_pgt=0.25, sp_pgt=10.95, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-NO-MECHANISM" and "O's longitudinal head" in r["next"]
    rc, r = _go(_pgt_files(tmp_path / "stop_worse", d_ade=0.8, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP"
    # seed 1 is not read at s0 even if its files are present
    assert "s1" not in _go(_pgt_files(tmp_path / "both", seeds=(0, 1)), "s0")[1]["rows"]


@needs_pkg
def test_reader_deliberate_regression_forward_shift_at_2s_reads_worse_separated(tmp_path, no_known):
    """A synthetic M2b-pgt whose plans are M2b's PLUS a forward shift at 2 s: the leak grows and the low-TTC tail grows, both separated."""
    rc, r = _go(_pgt_files(tmp_path / "reg", sp_pgt=11.0 + 0.6, lam_pgt=0.27 + 0.5, d_ade=0.0))
    for s in ("s0", "s1"):
        row = r["rows"][s]
        assert row["mleak"]["delta"]["point"] > 0 and row["mleak"]["delta"]["separated"], "M-LEAK worse, separated"
        assert row["mdk"]["delta"]["point"] > 0 and row["mdk"]["delta"]["separated"], "M-DK worse, separated"
        assert not row["judge"]["mleak"]["met"] and not row["judge"]["mdk"]["met"] and not row["judge"]["passed"]
    assert r["VERDICT"] == "FAIL-NO-MECHANISM" and r["mechanism"] is False


@needs_pkg
def test_reader_identity_control_known_values_and_check_inputs(tmp_path, monkeypatch, capsys):
    f = _pgt_files(tmp_path / "ident")
    # on synthetic data the REAL banked premises do not hold: the real known_values check refuses
    assert _go(f)[0] == 2
    assert "does not reproduce the PREREG's known values" in capsys.readouterr().err
    monkeypatch.setattr(R, "known_values", lambda *a, **k: {"checked": False})
    rc, r = _go(f)
    assert rc == 0
    ic = r["identity_control"]["s0"]
    for k in ("primary", "mleak", "mdk", "mdk_alt", "present_mse", "oracle_ade6"):
        assert ic[k]["delta"] == 0.0 and ic[k]["lo"] == 0.0 and ic[k]["hi"] == 0.0, k
    # a broken paired instrument (noise for identical inputs) is caught before any verdict is printed
    import taniteval.ci as CI
    real = CI.paired_episode_cluster_bootstrap

    def noisy(a, b, eid, n_boot=2000, seed=0, **kw):
        out = real(a, b, eid, n_boot, seed, **kw)
        return {**out, "delta": 1e-3, "lo": 5e-4, "hi": 2e-3}
    monkeypatch.setattr(CI, "paired_episode_cluster_bootstrap", noisy)
    assert _go(f)[0] == 2 and "IDENTITY CONTROL FAILED" in capsys.readouterr().err
    monkeypatch.setattr(CI, "paired_episode_cluster_bootstrap", real)
    # --check-inputs validates the banked M2b side ALONE (no M2b-pgt file needed) and prints OK
    out = f["d"] / "chk.json"
    base = ["--check-inputs", "--eval-dir", f["args"][1], "--runs-dir", f["args"][3], "--out", str(out), *f["args"][6:]]
    assert R.main(base) == 0 and "PGT INPUTS OK" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["identity_control"]["s0"]["mdk"]["delta"] == 0.0
    ru = Path(f["args"][3]) / "M2b_s1" / "summary.json"
    keep = ru.read_text(encoding="utf-8")
    ru.write_text("{", encoding="utf-8")
    assert R.main(base) == 2, "a corrupt banked run stops the chain while box 2 is still idle"
    ru.write_text(keep, encoding="utf-8")
    (Path(f["args"][1]) / "eval_M2b_s1-tacorc-all-latcons.npz").rename(Path(f["args"][1]) / "x.npz")
    assert R.main(base) == 2 and "eval_M2b_s1-tacorc-all-latcons.npz is missing" in capsys.readouterr().err


@needs_pkg
def test_reader_refusals(tmp_path, capsys, no_known):
    cases = (("miss", dict(drop="M2b-pgt_s1"), "eval_M2b-pgt_s1.npz is missing"),
             ("clip", dict(clip_pgt="u" * 64), "did not train on one clip list"),
             ("hold", dict(hold="0" * 64), "did not withhold the RA-1 holdout"),
             ("nclips", dict(n_clips=4000), "n_clips"),
             ("notdone", dict(done=False), "a run is not done"),
             ("steps", dict(steps_pgt=9000), "different step counts"),
             ("gate", dict(sc="FAIL"), "declared_vs_built is not PASS"),
             ("argvstart", dict(argv_pgt=("0.4", "0.0")), "argv --p-gt-start 0.4, expected 0.5"),
             ("argvend", dict(argv_pgt=("0.5", "0.1")), "argv --p-gt-end 0.1, expected 0.0"),
             ("builtstart", dict(built=(1.0, 0.0)), "was built with p_gt 1.0 -> 0.0"),
             ("builtend", dict(built=(0.5, 0.2)), "was built with p_gt 0.5 -> 0.2"),
             ("nobuilt", dict(no_built=True), "record no built p_gt_start / p_gt_end"),
             ("m2bflag", dict(m2b_flag=True), "M2b_s0: argv carries --p-gt-end"),
             ("m2bbuilt", dict(m2b_built=(0.5, 0.0)), "M2b_s0 was built with p_gt 0.5 -> 0.0"),
             ("arm", dict(argv_arm="M1"), "argv --arm M1, expected M2b"),
             ("otherlever", dict(extra_argv=("--w-cons", "4.0")), "not one lever"),
             ("badlog", dict(log_shift=0.03), "logged p_gt differs from the declared schedule"),
             ("nolog", dict(no_log=True), "train_log.jsonl is missing"),
             ("grid", dict(shuffle_eid=True), "eval_M2b-pgt_s0-tacabs.npz is not on the same eval grid as eval_M2b-pgt_s0.npz (eid differs)"),
             ("gridarms", dict(shuffle_eid="all"), "eval_M2b-pgt_s0.npz is not on the same eval grid as the others (eid differs)"),
             ("gridworld", dict(shuffle_eid="world"), "the rebuilt eval grid (eid) is not the evals' grid"),
             ("blockk", dict(block_k=5), "the lead block's K = 5 < 10"),
             ("smarm", dict(sm_arm="M1"), "M2b-pgt_s0/summary.json says M1 seed 0, expected M2b seed 0"),
             ("smseed", dict(sm_seed_shift=1), "M2b-pgt_s0/summary.json says M2b seed 1, expected M2b seed 0"),
             ("builtsc", dict(built_sc=(1.0, 0.2)), "was built with p_gt 1.0 -> 0.2 (selfcheck.json)"),
             ("builtsm", dict(built_sm=(1.0, 0.2)), "was built with p_gt 1.0 -> 0.2 (summary.json)"),
             ("nobuiltsc", dict(no_built_only="selfcheck"), "selfcheck.json levers record no built"),
             ("nobuiltsm", dict(no_built_only="summary"), "summary.json levers record no built"),
             ("emptylog", dict(empty_log=True), "M2b-pgt_s0/train_log.jsonl is empty"),
             ("noade", dict(omit="ade6"), "eval_M2b-pgt_s0.npz carries no ade6"),
             ("nopred", dict(omit="pred"), "eval_M2b-pgt_s0.npz carries no pred"),
             ("abscode", dict(abs_no_code=True), "eval_M2b-pgt_s0-tacabs.npz has no tac_code"),
             ("gt", dict(gt_shift=0.5), "the GT constraint block differs"),
             ("latgt", dict(lat_gt_wrong=True), "not the all-latcons oracle"),
             ("orcmode", dict(orc_mode="absmask"), "is not an oracle run"),
             ("absmode", dict(abs_mode="oracle"), "is not an absmask run"))
    for name, kw, msg in cases:
        assert _go(_pgt_files(tmp_path / name, **kw))[0] == 2, name
        assert msg in capsys.readouterr().err, name
    f = _pgt_files(tmp_path / "f", drop="M2b_s1", seeds=(0,))
    assert _go(f, "s0")[0] == 2 and "eval_M2b_s1.npz is missing" in capsys.readouterr().err, "M2b's seed floor needs both banked seeds"
    f = _pgt_files(tmp_path / "gone")
    (f["eval"] / "eval_M2b-pgt_s0-tacorc.npz").rename(f["eval"] / "gone.npz")
    assert _go(f)[0] == 2 and "eval_M2b-pgt_s0-tacorc.npz is missing" in capsys.readouterr().err


@needs_pkg
def test_reader_primary_bar_is_two_F_needs_a_separated_interval_and_every_bar_reads_its_own_seed(tmp_path, no_known, monkeypatch):
    F = _go(_pgt_files(tmp_path / "f"))[1]["F"]
    assert F > 0.05, "a synthetic world whose seed floor is not degenerate"
    # PRIMARY: |delta| > 2F exactly. 1.9 F is a separated improvement that is NOT beyond the floor; 2.1 F and 2.9 F are
    for name, mult, ok in (("u19", 1.9, False), ("o21", 2.1, True), ("o29", 2.9, True)):
        rc, r = _go(_pgt_files(tmp_path / name, d_ade=-mult * F))
        for s in ("s0", "s1"):
            p = r["rows"][s]["primary"]
            assert p["separated"] and p["point"] == pytest.approx(-mult * F, abs=2e-3), (name, s, p["point"])
            assert r["rows"][s]["judge"]["passed"] is ok, (name, s)
        assert r["VERDICT"] == ("SUCCESS" if ok else "FAIL-MECHANISM-OK"), (name, r["reading"])
    # a separated interval is NECESSARY at all three bars: with every interval forced to straddle zero, nothing passes, whatever the point estimates
    import taniteval.ci as CI
    real = CI.paired_episode_cluster_bootstrap

    def unseparated(a, b, eid, n_boot=2000, seed=0, **kw):
        return {**real(a, b, eid, n_boot, seed, **kw), "separated": False}
    monkeypatch.setattr(CI, "paired_episode_cluster_bootstrap", unseparated)
    rc, r = _go(_pgt_files(tmp_path / "unsep"))
    assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM" and r["mechanism"] is False, r["reading"]
    for s in ("s0", "s1"):
        j = r["rows"][s]["judge"]
        assert r["rows"][s]["primary"]["point"] < -2 * F and not j["passed"] and not j["mleak"]["met"] and not j["mdk"]["met"], s
    monkeypatch.setattr(CI, "paired_episode_cluster_bootstrap", real)
    # the mechanism is read at EVERY seed: met at seed 0, the leak not halved at seed 1 -> not met, and the verdict names the structural lever
    rc, r = _go(_pgt_files(tmp_path / "mech_s0_only", d_ade=-0.12, per_seed={1: {"pgt_lam": 0.25}}))
    assert rc == 0 and r["rows"]["s0"]["judge"]["mechanism"] is True and r["rows"]["s1"]["judge"]["mechanism"] is False
    assert r["mechanism"] is False and r["VERDICT"] == "FAIL-NO-MECHANISM", r["reading"]
    # each seed's bars use ITS OWN M2b reference: seed 1's M2b leak is 0.20 (half 0.10), seed 0's 0.27 (half 0.135); a 0.12 reduction at seed 1 meets
    # seed 1's bar and would miss seed 0's
    rc, r = _go(_pgt_files(tmp_path / "ref_s1", per_seed={1: {"m2b_lam": 0.2, "pgt_lam": 0.08}}))
    j0, j1 = r["rows"]["s0"]["judge"], r["rows"]["s1"]["judge"]
    assert rc == 0 and j1["mleak"]["met"] and abs(r["rows"]["s1"]["mleak"]["delta"]["point"]) < 0.5 * r["rows"]["s0"]["base"]["lambda_m2b_m"]
    for s, j in (("s0", j0), ("s1", j1)):
        assert j["mleak"]["ref"] == r["rows"][s]["base"]["lambda_m2b_m"] and j["mdk"]["ref"] == r["rows"][s]["base"]["excess_p10"], s
    assert j0["mleak"]["ref"] != j1["mleak"]["ref"]


@needs_pkg
def test_reader_mdk_pairs_the_windows_where_both_arms_close_and_reports_the_rest_beside_it(tmp_path, no_known):
    """The deciding S5 pairs the LEAD windows in which the lead is admissible for BOTH arms (min-TTC finite: the convention that reproduces the PREREG's
    banked premises); a window in which M2b-pgt's path never has the lead in its corridor leaves the pair. The count and the all-LEAD contrast ride
    beside it, deciding nothing."""
    f = _pgt_files(tmp_path / "cure", cure=10, states={0: "NO_LEAD", 15: "NO_LEAD"})
    rc, r = _go(f)
    g0 = f["w"].g0
    n = len(g0)
    lead = f["lead"]

    def at_risk(v):                                                     # the analytic TTC < 5 s indicator of an ego plan at v m/s behind the 8 m/s lead
        return (g0 + 0.2 * (8.0 - v) * 9) / (v - 8.0) < 5.0
    cured = np.arange(n) < 10                                           # windows 0 .. 9: M2b-pgt swerves clear of the lead, min-TTC undefined
    row = r["rows"]["s0"]
    d = row["mdk"]["delta"]
    both = lead & ~cured                                                # window 0 is cured AND not a lead window: counted once
    assert d["n_lead"] == int(lead.sum()) == n - 2 and d["n_used"] == int(both.sum()) and d["n_dropped_no_lead_in_path"] == int((lead & cured).sum()) == 9
    assert d["share_a"] == pytest.approx(float(at_risk(10.4)[both].mean())) and d["share_b"] == pytest.approx(float(at_risk(11.0)[both].mean()))
    assert d["point"] == pytest.approx(float(at_risk(10.4)[both].mean() - at_risk(11.0)[both].mean()), abs=1e-9)
    alt = d["alt_all_lead"]                                             # every LEAD window; the cured ones are NOT at risk; the non-lead ones are out
    sa, sb = float((at_risk(10.4) & ~cured)[lead].mean()), float(at_risk(11.0)[lead].mean())
    assert alt["point"] == pytest.approx(sa - sb, abs=1e-9) and alt["point"] != pytest.approx(d["point"], abs=1e-6)
    # the baseline's own excess carries the same companion (M2b and P10 are both closing everywhere here: nothing dropped)
    ex = row["base"]["excess_ci"]
    assert ex["n_dropped_no_lead_in_path"] == 0 and ex["alt_all_lead"]["point"] == pytest.approx(ex["point"], abs=1e-9)
    # M2b's baseline LAMBDA is read over the LEAD windows, and its interval is the interval OF that contrast
    ev = f["eval"]
    lam_b = np.load(ev / "eval_M2b_s0-tacorc-all-latcons.npz")["pred"][:, 9, 0].astype(np.float64) - np.load(ev / "eval_M2b_s0-tacorc.npz")["pred"][:, 9, 0].astype(np.float64)
    assert row["base"]["lambda_m2b_m"] == pytest.approx(float(lam_b[lead].mean()), abs=1e-9) and row["base"]["n_lead"] == int(lead.sum())
    assert row["base"]["lambda_ci"]["point"] == pytest.approx(float(lam_b[lead].mean()), abs=1e-9)
    assert not np.isclose(lam_b[lead].mean(), lam_b.mean()), "the world must tell LEAD windows from all windows"


@needs_pkg
def test_reader_known_values_are_checked_term_by_term_with_their_tolerance_and_the_identity_control_checks_the_bounds(tmp_path, monkeypatch, capsys):
    f = _pgt_files(tmp_path / "kv")
    with monkeypatch.context() as m:
        m.setattr(R, "known_values", lambda *a, **k: {"checked": False})
        rc, r = _go(f)
    assert rc == 0
    lam = {s: r["rows"][f"s{s}"]["base"]["lambda_m2b_m"] for s in (0, 1)}
    exc = {s: r["rows"][f"s{s}"]["base"]["excess_p10"] for s in (0, 1)}
    F = r["F"]
    assert R.KNOWN_TOL == {"lambda": 6e-4, "excess": 6e-5, "F": 6e-5}, "the tolerances are the ones the banked premises were checked at"

    def known(dlam=(0.0, 0.0), dexc=(0.0, 0.0), dF=0.0):
        return {"lambda": {s: lam[s] + dlam[s] for s in (0, 1)}, "excess": {s: exc[s] + dexc[s] for s in (0, 1)}, "F": F + dF}
    monkeypatch.setattr(R, "KNOWN", known())
    rc, r = _go(f)
    assert rc == 0 and r["known_values"]["checked"] is True
    # a perturbation of ONE term inside its tolerance passes, just outside refuses naming that term (so no term is checked only through another)
    for kw, ok_, bad_, msg in (("dlam", (0.0, 5e-4), (0.0, 7e-4), "M2b_s1 lambda reads"), ("dlam", (5e-4, 0.0), (7e-4, 0.0), "M2b_s0 lambda reads"),
                               ("dexc", (5e-5, 0.0), (7e-5, 0.0), "M2b_s0 excess reads"), ("dexc", (0.0, 5e-5), (0.0, 7e-5), "M2b_s1 excess reads"),
                               ("dF", 5e-5, 7e-5, "F reads")):
        monkeypatch.setattr(R, "KNOWN", known(**{kw: ok_}))
        assert _go(f)[0] == 0, (kw, ok_)
        monkeypatch.setattr(R, "KNOWN", known(**{kw: bad_}))
        capsys.readouterr()
        assert _go(f)[0] == 2 and msg in capsys.readouterr().err, (kw, bad_)
    # the identity control reads delta, point AND both bounds, on this reader's OWN contrast (the DK reader runs a control of its own first, on the same
    # message, so the bootstrap is NOT patched here: only R.paired is, and only for identical inputs)
    monkeypatch.setattr(R, "known_values", lambda *a, **k: {"checked": False})
    real_paired = R.paired
    for field, val in (("delta", 1e-3), ("lo", -1e-3), ("hi", 1e-3), ("point", 1e-3)):
        def broken(pf, a, b, eid, n_boot, _f=field, _v=val):
            r = real_paired(pf, a, b, eid, n_boot)
            return {**r, _f: _v} if np.array_equal(a, b) else r
        monkeypatch.setattr(R, "paired", broken)
        capsys.readouterr()
        assert _go(f)[0] == 2, field
        err = capsys.readouterr().err
        assert "IDENTITY CONTROL FAILED: M2b_s0 - M2b_s0" in err and "on primary" in err and "headway" not in err, (field, err)
    monkeypatch.setattr(R, "paired", real_paired)
    assert _go(f)[0] == 0


@needs_pkg
def test_reader_check_inputs_refuses_banked_gt_codes_that_differ_between_the_two_m2b_seeds(tmp_path, capsys):
    """--check-inputs is the ONLY caller of the GT-code check on the banked side (the stage reads call it again through the present-slot machinery)."""
    f = _pgt_files(tmp_path / "gtseeds")
    chk = ["--check-inputs", "--eval-dir", f["args"][1], "--runs-dir", f["args"][3], "--out", str(tmp_path / "chk.json"), *f["args"][6:]]
    p = Path(f["args"][1]) / "eval_M2b_s1-tacorc-all-goal.npz"                   # the banked M2b oracle's GT source (old format: no tac_code in -tacorc)
    z = dict(np.load(p))
    z["tac_code"] = z["tac_code"].copy()
    z["tac_code"][:, C0 + 3] += 1.0                                              # one GT constraint slot differs from seed 0's
    np.savez(p, **z)
    assert R.main(chk) == 2 and "the GT constraint block differs" in capsys.readouterr().err


# =================================================================================================================================================
# THE QUEUE STAGES (gatepgt / trainpgt / evalpgt) AND THE CHAIN
# The WC chain's lesson: a dry run that only PRINTS proved nothing -- the mutation proof found a real line that could drop the lever under a green
# dry run. Here every stage builds ONE command string per command; the dry run echoes it, the real run executes it; three tests pin that:
#   (static) the trainer / eval appear only inside the `set` that defines the string, and the string is executed exactly once;
#   (dry)    the printed lines carry the lever;
#   (real)   the REAL (non-dry) stages are run against FAKE trainer / eval scripts that record the argv they were handed, and that argv equals the
#            printed line's argv, token for token.
# =================================================================================================================================================
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")


@needs_queue
def test_queue_pgt_dose_is_defined_once_and_the_stages_read_it():
    q = _queue_text()
    assert q.count('\nset "PGTS=0.5"\n') == 1 and q.count('\nset "PGTE=0.0"\n') == 1, "the dose lives once, in the queue"
    assert len(re.findall(r'^set "PGT[SE]=', q, re.M)) == 2
    for label in ("gatepgt", "trainpgt", "evalpgt"):
        body = "\n".join(_code(_block(label)))
        assert "p-gt-start 0.5" not in body and "p-gt-end 0.0" not in body, f"{label}: a literal copy of the dose"
    assert "p-gt-start 0.5" not in q and "p-gt-end 0.0" not in q.replace("rem ", "")
    assert 'if /i "%~1"=="gatepgt" goto :gatepgt' in q and 'if /i "%~1"=="trainpgt" goto :trainpgt' in q and 'if /i "%~1"=="evalpgt" goto :evalpgt' in q
    # the dose is M2b-pgt's and NOT M2b's: the queue's `train M2b` line carries no p-gt flag
    assert "p-gt" not in "\n".join(_code(_block("train")))
    # M2b's dose is the trainer's default, the reader's constants are the PREREG's: the queue is the single definition, the reader the independent check
    if PGT_R.exists():
        assert (R.PGT_START, R.PGT_END) == (float(re.search(r'^set "PGTS=(.*)"$', q, re.M).group(1)), float(re.search(r'^set "PGTE=(.*)"$', q, re.M).group(1)))


@needs_queue
def test_queue_pgt_runs_the_command_it_prints():
    """A dry run can only print: the real line is checked statically -- the trainer / eval appear ONCE per command, inside the one string, and the
    real run executes that string."""
    for label, var, lever in (("gatepgt", "GP", True), ("trainpgt", "TP", True)):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" --arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% ') for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
    code = _code(_block("evalpgt"))
    assert [x for x in code if "%EVAL%" in x] == [x for x in code if re.match(r'set "E[0-3]=', x)], "the eval script appears only inside the four strings"
    assert len([x for x in code if re.match(r'set "E[0-3]=', x)]) == 4
    assert "if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)" in code
    for k in range(4):
        assert sum(x.strip().startswith(f"%E{k}% >> ") for x in code) == 1, f"E{k} is executed exactly once"
    e = {k: next(x for x in code if x.startswith(f'set "E{k}=')) for k in range(4)}
    assert "--project --families" in e[0] and "--tac" not in e[0]
    assert "--tac absmask" in e[1] and "--tac oracle" in e[2] and "--tac-groups" not in e[2]
    assert "--tac oracle --tac-groups all-latcons --tac-alpha 1" in e[3]
    for k in range(4):
        assert "%EC% %ER%" in e[k] and "--ckpt" in e[k]


def _argv_of(line):
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', line.strip())]


def _dry(args, env_extra=None, clean_paths=False):
    env = dict(os.environ, V8F_DRYRUN="1")
    if clean_paths:
        for k in ("REPO", "RUNSM", "CACHEFULL", "V9", "PY"):
            env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]


@needs_queue
@win
def test_queue_pgt_stages_dry(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    rc, ln = _dry("gatepgt", env)
    assert rc == 0 and len(ln) == 1 and "--arm M2b --p-gt-start 0.5 --p-gt-end 0.0" in ln[0] and "--selfcheck-only" in ln[0] and "gate_M2b-pgt" in ln[0], ln
    rc, ln = _dry("trainpgt 1", env)
    assert rc == 0 and len(ln) == 1 and "--arm M2b --p-gt-start 0.5 --p-gt-end 0.0 --seed 1" in ln[0] and "M2b-pgt_s1" in ln[0], ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0] and "--batch 64" in ln[0], ln
    assert "--w-cons" not in ln[0] and "--w-tac" not in ln[0], "ONE lever: no other weight on the line"
    assert _dry("trainpgt", env)[0] == 2
    rc, ln = _dry("evalpgt 0", env)
    assert rc == 0 and len(ln) == 4, ln
    assert "--tag M2b-pgt_s0 " in ln[0] and "--project --families" in ln[0], ln
    assert "--tag M2b-pgt_s0-tacabs " in ln[1] and "--tac absmask" in ln[1], ln
    assert "--tag M2b-pgt_s0-tacorc " in ln[2] and "--tac oracle" in ln[2] and "--tac-groups" not in ln[2], ln
    assert "--tag M2b-pgt_s0-tacorc-all-latcons " in ln[3] and "--tac oracle --tac-groups all-latcons --tac-alpha 1" in ln[3], ln
    assert _dry("evalpgt", env)[0] == 2
    # M2b's own train line is untouched
    rc, ln = _dry("train M2b 0", env)
    assert rc == 0 and "p-gt" not in ln[0]

    # ONE lever: M2b-pgt's gate and train lines are M2b's own lines (the same stage, the same grid, the same flags) plus the two lever flags -- and the
    # output directory, which is the only other thing that differs
    def minus_lever(line):
        t, out, i = _argv_of(line), [], 0
        while i < len(t):
            if t[i] in ("--p-gt-start", "--p-gt-end", "--out"):
                i += 2
                continue
            out.append(t[i])
            i += 1
        return out
    assert minus_lever(_dry("trainpgt 1", env)[1][0]) == minus_lever(_dry("train M2b 1", env)[1][0]) and "--grid" in _dry("trainpgt 1", env)[1][0]
    assert "--grid pool4x10" in _dry("trainpgt 1", env)[1][0]
    assert minus_lever(_dry("gatepgt", env)[1][0]) == minus_lever(_dry("gatearm M2b", env)[1][0])
    for k, line in enumerate(_dry("evalpgt 0", env)[1]):                    # every eval reads THIS arm's last checkpoint, never M2b's
        assert f"M2b-pgt_s0{os.sep}ckpt_last.pt" in line and "M2b_s0" not in line.replace("M2b-pgt_s0", ""), (k, line)


def _fake_tree(tmp):
    """A repo whose trainer / eval are FAKES that record the argv they are handed (and make the files the next stage waits for)."""
    repo = tmp / "repo"
    sc = repo / "stack" / "scripts"
    sc.mkdir(parents=True)
    rec = tmp / "argv.jsonl"
    (sc / "v8f_mini_train.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'train', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; out = a[a.index('--out') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "if os.environ.get('FAKE_TRAIN') == 'fail':\n"
        "    sys.exit(3)\n"
        "if '--selfcheck-only' not in a and os.environ.get('FAKE_TRAIN') != 'nosummary':\n"
        "    open(os.path.join(out, 'summary.json'), 'w').write('{}')\n", encoding="utf-8")
    (sc / "v8f_mini_eval.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'eval', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; d = a[a.index('--out-dir') + 1]; t = a[a.index('--tag') + 1]\n"
        "if os.environ.get('FAKE_FAIL_TAG') == t:\n"
        "    sys.exit(1)\n"
        "os.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'eval_' + t + '.npz'), 'wb').write(b'x')\n", encoding="utf-8")
    hj = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"p")
    (hj / "p10_eval_pred6.npy").write_bytes(b"p")
    cf = tmp / "cf"
    for sub in ("train", "eval"):
        (cf / sub).mkdir(parents=True)
        (cf / sub / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (cf / "native_tracks.pt").write_bytes(b"t")
    (tmp / "runsm").mkdir()
    return repo, rec


@needs_queue
@win
def test_queue_pgt_real_lines_execute_exactly_the_printed_commands(tmp_path):
    repo, rec = _fake_tree(tmp_path)
    env = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    env.pop("V8F_DRYRUN", None)

    def real(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=120, encoding="utf-8")
        assert r.returncode == 0, (args, r.stdout, r.stderr)
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        denv = dict(env, V8F_DRYRUN="1")
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=denv, capture_output=True, text=True, timeout=60, encoding="utf-8")
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]       # drop python.exe and the script: sys.argv[1:]
    want_gate, want_train, want_eval = printed("gatepgt"), printed("trainpgt 1"), printed("evalpgt 1")      # BEFORE the runs: a finished stage prints [skip]
    assert len(want_gate) == 1 and len(want_train) == 1 and len(want_eval) == 4
    got = real("gatepgt")
    assert [g["argv"] for g in got] == want_gate and got[-1]["who"] == "train" and "--selfcheck-only" in got[-1]["argv"]
    n = len(got)
    got = real("trainpgt 1")[n:]
    assert [g["argv"] for g in got] == want_train and len(got) == 1, "the REAL trainpgt line hands the trainer the printed command"
    a = got[0]["argv"]
    assert a[a.index("--arm") + 1] == "M2b" and a[a.index("--p-gt-start") + 1] == "0.5" and a[a.index("--p-gt-end") + 1] == "0.0" and a[a.index("--seed") + 1] == "1"
    n += 1
    got = real("evalpgt 1")[n:]
    assert [g["argv"] for g in got] == want_eval and len(got) == 4 and all(g["who"] == "eval" for g in got), "the REAL evalpgt lines hand the eval the printed commands"
    tags = [g["argv"][g["argv"].index("--tag") + 1] for g in got]
    assert tags == ["M2b-pgt_s1", "M2b-pgt_s1-tacabs", "M2b-pgt_s1-tacorc", "M2b-pgt_s1-tacorc-all-latcons"]
    # a finished stage is skipped, not re-run
    n += 4
    assert len(real("evalpgt 1")) == n and len(real("trainpgt 1")) == n


@needs_queue
@win
def test_queue_pgt_real_stages_notice_failures_and_skip_exactly_what_is_finished(tmp_path):
    repo, rec = _fake_tree(tmp_path)
    runsm = tmp_path / "runsm"
    ev = runsm / "eval"
    base = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(runsm), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    base.pop("V8F_DRYRUN", None)

    def stage(args, **extra):
        env = dict(base, **extra)
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=120, encoding="utf-8")
        return r.returncode, r.stdout

    def evals():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines() if '"eval"' in x] if rec.exists() else []
    # a gate or a trainer that fails is a FAILED stage (rc 1), not a silent success
    rc, out = stage("gatepgt", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] gate M2b-pgt" in out
    rc, out = stage("trainpgt 1", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] train M2b-pgt_s1 - see" in out
    # a trainer that exits 0 WITHOUT writing summary.json (a crash that swallowed its exit code) is a failed stage too
    rc, out = stage("trainpgt 1", FAKE_TRAIN="nosummary")
    assert rc == 1 and "exited 0 without summary.json" in out
    # evaluation refuses an arm that has not finished training
    rc, out = stage("evalpgt 1")
    assert rc == 2 and "has not finished training" in out and not evals()
    # a finished run: each eval that fails fails the stage, naming the eval that failed
    assert stage("trainpgt 1")[0] == 0
    for tag in ("M2b-pgt_s1", "M2b-pgt_s1-tacabs", "M2b-pgt_s1-tacorc", "M2b-pgt_s1-tacorc-all-latcons"):
        rc, out = stage("evalpgt 1", FAKE_FAIL_TAG=tag)
        assert rc == 1 and f"[FAIL] eval {tag} - see" in out, (tag, rc, out)
    # partial resume: only the evals whose npz is missing run, and a finished eval skips ITSELF only (each of the four, alone and in pairs)
    ev.mkdir(exist_ok=True)
    tags = ["M2b-pgt_s1", "M2b-pgt_s1-tacabs", "M2b-pgt_s1-tacorc", "M2b-pgt_s1-tacorc-all-latcons"]
    for n, done in enumerate(([0, 2], [1], [0], [2], [3], [0, 1, 2], [1, 3])):
        for f in list(ev.glob("eval_M2b-pgt_s1*")):
            f.rename(f.with_name(f"old{n}_" + f.name))
        rec.write_text("", encoding="utf-8")
        for k in done:
            (ev / f"eval_{tags[k]}.npz").write_bytes(b"x")
        assert stage("evalpgt 1")[0] == 0, done
        ran = [g["argv"][g["argv"].index("--tag") + 1] for g in evals()]
        assert ran == [tg for k, tg in enumerate(tags) if k not in done], (done, ran)


# ---- the chain ------------------------------------------------------------------------------------------------------------------------------
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")


@needs_chain
def test_chain_pgt_static_ordering_and_the_lever():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    # no second copy of the dose; it is parsed from the tip's queue, and checked there to be defined exactly once
    assert "p-gt-start 0.5" not in c and not re.search(r"\bPGTS=0\.5\b", c) and "PGTS=$(sed -n" in c and "PGTE=$(sed -n" in c
    assert "defines the dose more than once" in c
    # the order: refresh -> check-inputs on the REAL banked side -> telemetry/guard -> gate -> seed 0 -> read s0 -> (CONTINUE) seed 1 -> final read
    order = ['v8f_pgt_read.py" --check-inputs', "queue gatepgt 120 gatepgt", "runone 0", "pgtread s0 pgt_read_s0.json", "runone 1", "pgtread final pgt_read.json"]
    idx = [c.index(x) for x in order]
    assert idx == sorted(idx), dict(zip(order, idx))
    assert c.index("v8f_guard.sh >") > c.index('v8f_pgt_read.py" --check-inputs') and c.index("queue gatepgt") > c.index("v8f_guard.sh >")
    assert 'queue "t_M2b-pgt_s$s" 1500 trainpgt $s' in c and 'queue "e_M2b-pgt_s$s" 600 evalpgt $s' in c
    assert '[ "$(jget pgt_read_s0.json DECISION)" = CONTINUE ] || finish' in c, "a STOP at seed 0 is a committed outcome (DONE), not a crash"
    # the gate assertions read the BUILT schedule from selfcheck.json and refuse M2b's own
    assert 'lv.get("p_gt_start") != p0 or lv.get("p_gt_end") != p1' in c and "(1.0, 0.2)" in c and "the lever did not move" in c
    assert '(w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5)' in c, "a second lever (the WC weights) must not ride along"
    for s in ("train_log.jsonl", "ckpt_last.pt", "selfcheck.json", "summary.json", "argv.json"):
        assert s in c
    for x in ("-tacabs", "-tacorc", "-tacorc-all-latcons"):
        assert x in c
    # the banked M2b side that M-LEAK / M-DK / the P10 guard need is staged BEFORE any training
    for f in ("-tacorc-all-latcons.npz", "-tacorc-all-goal.npz", "floors_perwindow.npz", "-tacabs.npz", "-tacorc.npz"):
        assert f in c[: c.index("queue gatepgt")], f
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert '"$B1"' in arch and "b1_eval_lead_block.npz" in c, "the reader's K = 10 lead block rides in the archive"
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"'))
    # markers + the guard + the predecessor
    assert "ZZV8FMINI-STOP" in c and "ZZV8FMINI-DONE" in c and "chain_v8f_pgt.out" in c and "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_wc.out}" in c
    assert 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c


@needs_chain
def test_chain_pgt_syntax_and_dry_run_prints_the_queue_lines(tmp_path):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), TMPDIR=tmp_path.as_posix())
    env.pop("V8F_DRYRUN", None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert [x for x in out if x.startswith("== ")] == ["== gatepgt", "== trainpgt 0", "== trainpgt 1", "== evalpgt 0", "== evalpgt 1"]
    # the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths (the chain ships the stage name; the queue runs that string)
    want = []
    for st in ("gatepgt", "trainpgt 0", "trainpgt 1", "evalpgt 0", "evalpgt 1"):
        want += _dry(st, clean_paths=True)[1]
    assert [x for x in out if ".py" in x] == want and len(want) == 1 + 1 + 1 + 4 + 4
    train0 = next(x for x in out if "--seed 0" in x and "v8f_mini_train.py" in x)
    assert "--arm M2b --p-gt-start 0.5 --p-gt-end 0.0 --seed 0" in train0 and "D:\\tanitad\\repo" in train0


def _gate_assertions():
    """The python block the chain runs on the PULLED gate selfcheck.json (its heredoc), as a program."""
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"gate M2b-pgt assertions\"\n(.*?)\nEOF\n", c, re.S)
    assert m, "the chain's gate assertions are a python heredoc whose failure stops the chain"
    return m.group(1)


@needs_chain
def test_chain_pgt_gate_assertions_pass_the_pgt_gate_and_refuse_every_way_it_can_be_wrong(tmp_path):
    """The chain's gate block is RUN against a synthetic selfcheck.json: a good M2b-pgt gate passes; each of 20-odd single-field corruptions exits 1 and names
    the broken field (so the dose, the clip list, the holdout, the weights and the step count cannot drift under a green gate)."""
    import copy
    body = _gate_assertions()
    hold, m2b_sha = "f6a1dee0" + "0" * 56, "c" * 64
    good = {"declared_vs_built": "PASS", "live": {"pass": True, "control_prior_grad_sum": 0.0}, "steps_total": 19164,
            "info": {"n_clips": 4512, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "excluded_sha256": hold, "train_sha256": m2b_sha, "kin_mode": "yaw"},
            "levers": {"wm": "b", "navonly": False, "consistency": True, "cons_absent": "mask", "p_gt_start": 0.5, "p_gt_end": 0.0,
                       "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 1.0, "w_tac": 1.0, "w_cons": 0.5}}}
    m2b = tmp_path / "m2b_summary.json"
    m2b.write_text(json.dumps({"train_sha256": m2b_sha, "levers": {"w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 1.0}}}), encoding="utf-8")

    def run(mod=None, p=("0.5", "0.0")):
        sc = copy.deepcopy(good)
        if mod:
            mod(sc)
        f = tmp_path / "sc.json"
        f.write_text(json.dumps(sc), encoding="utf-8")
        r = subprocess.run([sys.executable, "-", str(f), hold, "19164", str(m2b), *p], input=body, capture_output=True, text=True, encoding="utf-8", timeout=60)
        return r.returncode, r.stdout + r.stderr
    rc, out = run()
    assert rc == 0 and "PASS" in out and "built p_gt 0.5 -> 0.0" in out, out
    L = lambda k, v: (lambda s: s["levers"].__setitem__(k, v))                   # noqa: E731
    W = lambda k, v: (lambda s: s["levers"]["w"].__setitem__(k, v))              # noqa: E731
    I = lambda k, v: (lambda s: s["info"].__setitem__(k, v))                     # noqa: E731
    cases = (("declared_vs_built", lambda s: s.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built"),
             ("live", lambda s: s["live"].__setitem__("pass", False), "live"),
             ("control grad", lambda s: s["live"].__setitem__("control_prior_grad_sum", 1.0), "live"),
             ("n_clips", I("n_clips", 4000), "clips / holdout counts"), ("n_excluded", I("n_excluded", 59), "clips / holdout counts"),
             ("n_excluded_declared", I("n_excluded_declared", 59), "clips / holdout counts"), ("excluded_in_train", I("excluded_in_train", 1), "clips / holdout counts"),
             ("holdout", I("excluded_sha256", "0" * 64), "not the RA-1 holdout"), ("train list", I("train_sha256", "d" * 64), "TRAIN list"),
             ("kin_mode", I("kin_mode", "steer"), "kin_mode"), ("wm", L("wm", "a"), "levers"), ("navonly", L("navonly", True), "levers"),
             ("consistency", L("consistency", False), "levers"), ("cons_absent", L("cons_absent", "zero"), "levers"),
             ("p_gt start", L("p_gt_start", 0.4), "built p_gt 0.4 -> 0.0"), ("p_gt end", L("p_gt_end", 0.1), "built p_gt 0.5 -> 0.1"),
             ("p_gt missing", lambda s: (s["levers"].pop("p_gt_start"), s["levers"].pop("p_gt_end")), "built p_gt None -> None"),
             ("w_pen", W("w_pen", 0.6), "w_pen"), ("w_pen_self", W("w_pen_self", 0.3), "w_pen_self"), ("w_wm", W("w_wm", 2.0), "w_wm"),
             ("w_tac", W("w_tac", 2.0), "w_tac / w_cons"), ("w_cons", W("w_cons", 1.0), "w_tac / w_cons"), ("steps", lambda s: s.__setitem__("steps_total", 9000), "steps 9000"))
    for name, mod, msg in cases:
        rc, out = run(mod)
        assert rc == 1 and msg in out, (name, rc, out)
    # the lever itself: M2b's own schedule is refused twice over (it is not the dose, and it means the flags did not move anything)
    rc, out = run(lambda s: s["levers"].update(p_gt_start=1.0, p_gt_end=0.2))
    assert rc == 1 and "built p_gt 1.0 -> 0.2" in out and "the lever did not move" in out, out
    # the expected dose comes from the arguments (the queue's PGTS / PGTE), not from the block: a different dose is checked against what was built
    rc, out = run(p=("0.6", "0.0"))
    assert rc == 1 and "built p_gt 0.5 -> 0.0, expected 0.6 -> 0.0" in out, out
    rc, out = run(L("p_gt_start", 0.6), p=("0.6", "0.0"))
    assert rc == 0, out


@needs_chain
def test_chain_pgt_load_bearing_lines_are_present():
    """The chain needs box 2 and ssh, so it cannot be run here: every line whose loss would let a wrong run through is pinned (a mutation of any of them is
    caught by this test or by the gate-assertion test above). Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""PGTS=$(sed -n 's/^set "PGTS=\(.*\)"$/\1/p' $QT); PGTE=$(sed -n 's/^set "PGTE=\(.*\)"$/\1/p' $QT)""",
        r"""[ -n "$PGTS" ] && [ -n "$PGTE" ] || stop "the tip's queue defines no PGTS / PGTE""",
        r"""[ "$(grep -c '^set "PGTS=' $QT)" = 1 ] && [ "$(grep -c '^set "PGTE=' $QT)" = 1 ] || stop "the tip's queue defines the dose more than once""",
        r"""for st in gatepgt trainpgt evalpgt; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start""",
        r"""grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""|| stop "the reader cannot use the banked M2b side (check-inputs)""",
        r"""$PY "$RD/v8f_pgt_read.py" --check-inputs --eval-dir""",
        r"""|| stop "pgt read $1""",
        r"""$PY "$RD/v8f_pgt_read.py" --stage $1 --eval-dir""",
        r"""for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "M2b-pgt_s$s/$f" "$E/runs/M2b-pgt_s$s"; done""",
        r"""for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull "eval/eval_M2b-pgt_s$s$f.$x" "$E/eval"; done; done""",
        r"""$HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $L/v8f_mini/runs/M2b_s0/summary.json)" $PGTS $PGTE <<'EOF' || stop "gate M2b-pgt assertions""",
        r"""stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do""",
        r"""grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped""",
        r"""cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo""",
        r"""for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done""",
        r"""for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done""",
        r"""cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz""",
        r"""for st in gatepgt "trainpgt 0" "trainpgt 1" "evalpgt 0" "evalpgt 1"; do""",
        r"""call "%s" %s\r\n' "$(cygpath -w "$QF")" "$st" > "$W""",
        r"""queue gatepgt 120 gatepgt""",
        r"""pull "gate_M2b-pgt/selfcheck.json" "$E/gate_M2b-pgt""",
        r"""[ "$(jget pgt_read_s0.json DECISION)" = CONTINUE ] || finish "$G1 -- seed 1 not trained (committed)""",
        r"""finish "$G1 | $(jget pgt_read.json reading)""")
    for ln in lines:
        assert ln in c, f"missing from the chain: {ln!r}"
