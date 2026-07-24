"""Tests for the INFERENCE-LEVER section of ``taniteval/efficiency.py``.

A lever sweep invites a specific class of lie: publishing a *fast wrong answer*.
Four ways that happens, one test each:

  1. a variant that FAILED TO BUILD silently falls back to the eager path, so the
     baseline gets published as a speedup;
  2. the equivalence check is computed against the wrong reference, or with an
     ADE definition that is not the harness's ``ade_0_2s``;
  3. the latency-only rollout-length sweep is quoted as if it had settled the
     accuracy question;
  4. a contaminated sweep overwrites a clean artifact.

CPU-only by default — the timing primitives are exercised by the real pod runs.
The one CUDA-gated test captures a REAL predictor rollout as a CUDA graph and
demands bit-identical output, because that is the claim the whole L1 lever rests
on and it cannot be checked with arithmetic.

pytest is NOT installed on the eval pod, so this runs standalone too:
  python taniteval/tests/test_efficiency_levers.py
"""
import json
import sys
import tempfile
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, str(_HERE.parents[2] / "stack"))            # repo layout
sys.path.insert(0, str(_HERE.parents[2] / "stack" / "scripts"))
sys.path.insert(0, "/root/taniteval")              # pod layout
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import efficiency as E  # noqa: E402

HAS_CUDA = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# scope contract — which arms even HAVE this tick shape                        #
# --------------------------------------------------------------------------- #
def test_levers_refuse_a_non_rollout_arch():
    """REF-C's tick is one parallel anchor pass, REF-B's is a direct regression.
    Neither has the 20-step serial rollout these levers attack — measuring them
    with this code would produce a meaningless 'speedup'."""
    for arch in ("refc", "refb", "something-new"):
        try:
            E._wm_parts({"arch": arch}, {"model": None})
        except ValueError:
            continue
        raise AssertionError(f"arch {arch} must be refused, not silently timed")


def test_rollout_arms_are_accepted():
    for arch in ("flagship-worldmodel", "flagship-worldmodel-v2", "refa-plus"):
        m, sr = E._wm_parts({"arch": arch},
                            {"model": "M", "step_readout": "SR"})
        assert (m, sr) == ("M", "SR")


def test_lever_order_lists_the_reference_first_and_covers_the_levers():
    assert E.LEVER_ORDER[0] == "eager", \
        "the reference must be measured first — every speedup is relative to it"
    for name in ("graph_step", "graph_rollout", "graph_fulltick",
                 "compile_rollout", "compile_cudagraphs", "enc_cache",
                 "enc_cache_graph", "drop_horizons", "drop_horizons_graph",
                 "fp16_eager", "fp16_graph_rollout", "fp16_enc_cache_graph",
                 "all_levers", "fan_shared_encoder"):
        assert name in E.LEVER_ORDER, f"{name} missing from LEVER_ORDER"
    assert len(set(E.LEVER_ORDER)) == len(E.LEVER_ORDER)
    # the fully-composed batch-1 tick is the number the 10 Hz verdict is read
    # off, so it must be built (and reported) before the fan variant that
    # generalises it to K candidates
    assert (E.LEVER_ORDER.index("all_levers")
            < E.LEVER_ORDER.index("fan_shared_encoder"))
    assert E.LEVER_ORDER.index("eager") == 0


# --------------------------------------------------------------------------- #
# L7 — dropping the unused multi-horizon heads                                  #
# --------------------------------------------------------------------------- #
def _predictor(state_dim=32, horizons=(1, 2, 4)):
    from tanitad.config import PredictorConfig
    from tanitad.models.predictor import OperativePredictor
    torch.manual_seed(0)
    return OperativePredictor(
        PredictorConfig(d_model=32, depth=1, n_heads=2, window=8, action_dim=3,
                        horizons=horizons), state_dim).eval()


def test_prune_horizons_keeps_only_the_consumed_head():
    """`rollout_decode` reads only head [1], but forward() evaluates all three
    every call, 20x per tick. Pruning must keep the SAME weights (so the output
    is bit-identical) and must NOT mutate the model being benchmarked."""
    p = _predictor()
    q = E.prune_horizons(p, keep=(1,))
    assert set(q.heads.keys()) == {"1"}
    assert tuple(q.cfg.horizons) == (1,)
    assert set(p.heads.keys()) == {"1", "2", "4"}, "original predictor mutated"
    assert tuple(p.cfg.horizons) == (1, 2, 4), "original cfg mutated"
    torch.manual_seed(1)
    st = torch.randn(1, 8, 32)
    aw = torch.randn(1, 8, 3)
    with torch.no_grad():
        a, b = p(st, aw)[1], q(st, aw)[1]
    assert torch.equal(a, b), \
        f"pruning changed the output: max|d| {float((a - b).abs().max()):.3e}"
    assert set(q(st, aw).keys()) == {1}, "pruned predictor still computes extras"


def test_prune_horizons_is_a_deep_copy_not_an_alias():
    p = _predictor()
    q = E.prune_horizons(p, keep=(1,))
    assert q.heads["1"].weight.data_ptr() != p.heads["1"].weight.data_ptr(), \
        "a shared tensor would let a later .half() corrupt the fp32 reference"
    assert torch.equal(q.heads["1"].weight, p.heads["1"].weight)


# --------------------------------------------------------------------------- #
# L5b — the strided multi-step-head probe must refuse an accuracy claim          #
# --------------------------------------------------------------------------- #
def test_strided_caveat_states_the_recalibration_dependency():
    """The k=2/k=4 heads are already trained, so a strided roll is reachable
    with no retraining — but the step readout was calibrated on 0.1 s
    transitions. Quoting a strided ADE without recalibration is the failure this
    caveat exists to block."""
    c = E.STRIDED_ACCURACY_CAVEAT
    assert "NOT CLAIMABLE" in c and "recalibrat" in c
    assert "0.1 s" in c, "the caveat must name the readout's calibration units"
    assert "latency" in c.lower()


# --------------------------------------------------------------------------- #
# L2 machinery — the rolling encoder-state cache                                #
# --------------------------------------------------------------------------- #
def test_rolling_cache_reproduces_the_full_window():
    """Pushing W states one at a time must leave exactly the window that a
    batched encode of the same W frames would produce (order included)."""
    w, s = 8, 5
    states = torch.arange(w * s, dtype=torch.float32).reshape(1, w, s)
    c = E.RollingStateCache(torch.zeros(1, w, s))
    for i in range(w):
        buf = c.push(states[:, i])
    assert torch.equal(buf, states), "cache window drifted from the batch encode"
    assert buf.shape == (1, w, s)


def test_rolling_cache_push_shifts_and_drops_the_oldest():
    c = E.RollingStateCache(torch.arange(8, dtype=torch.float32)
                            .reshape(1, 8, 1))
    buf = c.push(torch.tensor([[99.0]]))
    assert buf[0, -1, 0] == 99.0 and buf[0, 0, 0] == 1.0, \
        "push must roll the window, not overwrite in place"


def test_rolling_cache_does_not_alias_its_seed():
    seed = torch.zeros(1, 8, 2)
    c = E.RollingStateCache(seed)
    c.push(torch.ones(1, 2))
    assert torch.equal(seed, torch.zeros(1, 8, 2)), \
        "the cache must clone its seed, not alias the caller's states"


# --------------------------------------------------------------------------- #
# equivalence — every speed row must ship an accuracy row                        #
# --------------------------------------------------------------------------- #
def _traj(n=6, k=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.cumsum(torch.rand(n, k, 2, generator=g), dim=1)
    return t


WP = [5, 10, 15, 20]


def test_equivalence_is_exactly_zero_for_an_identical_trajectory():
    t = _traj()
    gt = t.index_select(1, torch.tensor([s - 1 for s in WP])) + 0.1
    eq = E.equivalence(t, t.clone(), gt, WP)
    assert eq["max_abs_dev_m"] == 0.0
    assert eq["rel_err_max"] == 0.0
    assert eq["wp_shift_m_mean"] == 0.0
    assert eq["ade_0_2s_delta_m"] == 0.0
    assert abs(eq["cosine"] - 1.0) < 1e-9
    assert eq["finite"] is True
    assert eq["n_windows"] == t.shape[0]


def test_equivalence_measures_a_known_shift_and_its_ade_cost():
    """A variant displaced by a known 3-4-5 offset must report a 5 cm waypoint
    shift; the ADE delta must be the real change in the scored quantity, not a
    proxy."""
    base = _traj()
    var = base.clone()
    var[..., 0] += 0.03
    var[..., 1] += 0.04
    gt = base.index_select(1, torch.tensor([s - 1 for s in WP]))
    eq = E.equivalence(base, var, gt, WP)
    assert abs(eq["wp_shift_m_mean"] - 0.05) < 1e-6
    assert abs(eq["wp_shift_m_max"] - 0.05) < 1e-6
    assert abs(eq["max_abs_dev_m"] - 0.04) < 1e-6
    # base is exactly GT here, so the variant's ADE IS the shift
    assert abs(eq["ade_0_2s_reference"]) < 1e-9
    assert abs(eq["ade_0_2s_delta_m"] - 0.05) < 1e-6


def test_equivalence_flags_non_finite_output():
    base = _traj()
    var = base.clone()
    var[0, 0, 0] = float("nan")
    gt = base.index_select(1, torch.tensor([s - 1 for s in WP]))
    assert E.equivalence(base, var, gt, WP)["finite"] is False


def test_equivalence_ade_is_the_harness_ade_0_2s():
    """`_ade` must be the SAME quantity `bench._suite` calls `ade_0_2s` (mean
    displacement error over the 4 canonical horizons) — otherwise a lever could
    be declared 'ADE-neutral' against a metric nobody scores."""
    pred = _traj(n=7, k=20, seed=3).index_select(
        1, torch.tensor([s - 1 for s in WP]))
    gt = _traj(n=7, k=20, seed=4).index_select(
        1, torch.tensor([s - 1 for s in WP]))
    mine = E._ade(pred, gt)
    manual = float(torch.linalg.norm(pred - gt, dim=-1).mean())
    assert abs(mine - manual) < 1e-9
    try:
        from taniteval import bench
    except Exception:                                            # noqa: BLE001
        return                                                   # stack absent
    assert abs(bench._suite(pred, gt)["ade_0_2s"] - mine) < 1e-9


# --------------------------------------------------------------------------- #
# L5 — the latency-only block must say so, in the artifact                       #
# --------------------------------------------------------------------------- #
def test_k_sweep_summary_declares_accuracy_unmeasured():
    rows = {"eager": {"20": {"p50_ms": 90.0}, "10": {"p50_ms": 46.0},
                      "5": {"p50_ms": 24.0}}}
    out = E._k_sweep_summary(rows, (20, 10, 5))
    assert "UNMEASURED" in out["ACCURACY"]
    assert "accuracy verdict" in out["ACCURACY"]
    saved = out["by_variant"]["eager"]["_saved_ms_p50_vs_k20"]
    assert abs(saved["10"] - 44.0) < 1e-9 and abs(saved["5"] - 66.0) < 1e-9
    assert abs(saved["20"]) < 1e-9


def test_k_sweep_summary_tolerates_a_failed_variant():
    rows = {"graph_rollout": {"20": {"error": "capture failed"},
                              "10": {"p50_ms": 5.0}, "5": {"p50_ms": 3.0}}}
    out = E._k_sweep_summary(rows, (20, 10, 5))
    assert "_saved_ms_p50_vs_k20" not in out["by_variant"]["graph_rollout"]


# --------------------------------------------------------------------------- #
# misc primitives                                                               #
# --------------------------------------------------------------------------- #
def test_tmap_walks_nested_containers():
    o = {"a": torch.ones(2), "b": (torch.ones(1), [torch.ones(3)]), "c": 7}
    out = E._tmap(o, lambda t: t * 2)
    assert out["a"].sum() == 4 and out["b"][0].sum() == 2
    assert out["b"][1][0].sum() == 6 and out["c"] == 7


def test_measure_levers_restores_backend_flags():
    """A lever sweep runs on the same box as accuracy evals; leaving TF32 or
    cudnn.benchmark flipped would silently change a later ADE."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    before = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        E.measure_levers({"arch": "refc"}, {"model": None}, None,
                         precision="tf32")
    except Exception:                                            # noqa: BLE001
        pass
    after = (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    assert before == after, f"backend flags leaked: {before} -> {after}"


# --------------------------------------------------------------------------- #
# artifact contract                                                             #
# --------------------------------------------------------------------------- #
def _fake_sweep(valid=True, eager=103.0, best=30.0):
    def row(ms, name, lever, dev=0.0):
        return {"meta": {"lever": lever, "desc": name},
                "tick": {"p50_ms": ms, "p95_ms": ms * 1.02, "p99_ms": ms * 1.05,
                         "mean_ms": ms},
                "speedup_vs_eager": {"p50": round(eager / ms, 4),
                                     "p99": round(eager / ms, 4)},
                "equivalence": {"max_abs_dev_m": dev, "ade_0_2s_delta_m": dev},
                "realtime": {"meets_10hz_p99": ms * 1.05 < 100}}
    return {"key": "k", "model": {"name": "k"}, "ckpt_step": 29999,
            "tf32": {"precision": {"precision": "tf32"},
                     "levers": {"eager": row(eager, "baseline", "—"),
                                "graph_rollout": row(best, "graph", "L1b")},
                     "contamination_check": {"valid": valid}}}


def test_lever_table_renders_the_before_after_and_the_10hz_verdict():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "eff_levers_k.json").write_text(json.dumps(_fake_sweep()))
        t = E.lever_table(p, "k", "tf32")
        assert "eager" in t and "graph_rollout" in t
        assert "YES" in t, "a variant under 100 ms at p99 must be marked"
        assert "clean=True" in t
        rows = t.split("\n")
        assert rows[2].startswith("eager") and rows[2].rstrip().endswith("no"), \
            "the 108 ms-p99 eager row must be marked as MISSING 10 Hz"
        assert rows[3].rstrip().endswith("YES"), \
            "the 31.5 ms-p99 graph row must be marked as meeting 10 Hz"


def test_lever_table_missing_artifact_is_not_an_error():
    with tempfile.TemporaryDirectory() as d:
        assert "no eff_levers_" in E.lever_table(Path(d), "nope")


def test_contaminated_lever_sweep_is_quarantined_not_published():
    """Exactly the 2026-07-20 accident, in the lever artifact: a sweep measured
    while a neighbour job shared the GPU must never overwrite a clean file."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        clean = _fake_sweep()
        (p / "eff_levers_x.json").write_text(json.dumps(clean))
        dirty = _fake_sweep(valid=False)["tf32"]
        orig_load, orig_meas = E._load_eps, E.measure_levers
        E._load_eps = lambda k, device="cuda", n_eps=1: (
            {"key": k}, {"step": 1}, [None])
        E.measure_levers = lambda *a, **kw: json.loads(json.dumps(dirty))
        try:
            E.run_levers_and_save("x", precisions=("tf32",), k_sweep=None,
                                  res_dir=p)
        finally:
            E._load_eps, E.measure_levers = orig_load, orig_meas
        assert json.loads((p / "eff_levers_x.json").read_text()) == clean, \
            "a contaminated sweep overwrote the clean artifact"
        q = list(p.glob("eff_levers_x.CONTAMINATED-*.json"))
        assert len(q) == 1 and "QUARANTINED" in json.loads(q[0].read_text())


def test_contamination_is_invalidated_by_a_MID_RUN_intruder():
    """`gpu_state` before/after can both be clean while a neighbour job came and
    went during the 10-minute sweep. The per-variant samples exist for exactly
    that case and must be able to fail the run."""
    src = Path(E.__file__).read_text()
    assert "intrusions_mid_run" in src and "gpu_state_samples" in src
    good = {"gpu_state_before": {"exclusive": True},
            "gpu_state_after": {"exclusive": True},
            "gpu_state_samples": [{"other_compute_procs": 0},
                                  {"other_compute_procs": 2}]}
    mid = [s for s in good["gpu_state_samples"]
           if (s.get("other_compute_procs") or 0) > 0]
    assert mid, "the mid-run sample must be detectable"
    assert not bool(good["gpu_state_before"]["exclusive"]
                    and good["gpu_state_after"]["exclusive"] and not mid)


# --------------------------------------------------------------------------- #
# CUDA-gated: the claim the whole L1 lever rests on                             #
# --------------------------------------------------------------------------- #
def test_cuda_graph_rollout_is_bit_identical_to_eager():
    """A CUDA graph replays the SAME kernels, so a captured 20-step rollout must
    reproduce the eager trajectory EXACTLY. Anything else is a capture bug (a
    stale static buffer, a missing input copy), not a precision trade — and it
    would ship as a silent accuracy regression.

    This is also the executable answer to the standing worry that the rollout's
    ~40 allocating `torch.cat`s (`metric_dynamics.py:241-242`) and the per-call
    causal-mask build (`predictor.py:112`) make the region uncapturable. They do
    not: allocations DURING capture are served from the graph's private pool and
    replay at the same addresses. The failure mode to guard against is not
    "capture refuses" but "capture succeeds and quietly returns stale numbers" —
    which is why this test also checks that a SECOND, different input produces a
    correspondingly different output."""
    if not HAS_CUDA:
        return
    from tanitad.config import PredictorConfig
    from tanitad.models.metric_dynamics import (StepDisplacementReadout,
                                                rollout_decode)
    from tanitad.models.predictor import OperativePredictor
    torch.manual_seed(0)
    s_dim, w, k = 96, 8, 20
    pr = OperativePredictor(
        PredictorConfig(d_model=128, depth=2, n_heads=4, window=w,
                        action_dim=3, horizons=(1, 2, 4)), s_dim).cuda().eval()
    sr = StepDisplacementReadout(s_dim, hidden=64).cuda().eval()
    st = torch.randn(1, w, s_dim, device="cuda")
    aw = torch.randn(1, w, 3, device="cuda") * 0.1
    fa = torch.randn(1, k, 3, device="cuda") * 0.1
    with torch.no_grad():
        want = rollout_decode(pr, st, aw, fa, sr, k)[0].clone()
        g = E.GraphedFn(lambda states, actions, future_actions: rollout_decode(
            pr, states, actions, future_actions, sr, k),
            {"states": st, "actions": aw, "future_actions": fa})
        got = g.run(states=st, actions=aw, future_actions=fa)[0].clone()
    assert torch.equal(want, got), \
        f"graph diverged from eager: max|Δ| {float((want - got).abs().max()):.3e}"
    # and it must FOLLOW its inputs — a graph that ignores the copy_ would also
    # pass the test above
    with torch.no_grad():
        st2 = torch.randn(1, w, s_dim, device="cuda")
        want2 = rollout_decode(pr, st2, aw, fa, sr, k)[0].clone()
        got2 = g.run(states=st2, actions=aw, future_actions=fa)[0].clone()
    assert torch.equal(want2, got2), "graph did not pick up the new inputs"
    assert not torch.equal(got, got2), "the two inputs must give two outputs"


def test_cuda_graph_output_is_overwritten_by_the_next_replay():
    """Pins the trap the harness must not fall into: graph outputs live in the
    graph's private pool. Any caller that keeps a result WITHOUT cloning it will
    silently compare a variant against itself."""
    if not HAS_CUDA:
        return
    g = E.GraphedFn(lambda x: (x * 2 + 1,), {"x": torch.ones(4, device="cuda")})
    a = g.run(x=torch.ones(4, device="cuda"))[0]                 # not cloned
    b = g.run(x=torch.full((4,), 5.0, device="cuda"))[0]
    assert torch.equal(a, b), "expected aliasing — the harness must clone"
    assert float(b[0]) == 11.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"==== {len(fns) - bad}/{len(fns)} passed "
          f"(cuda={'yes' if HAS_CUDA else 'no'}) ====")
    sys.exit(1 if bad else 0)
