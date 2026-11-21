"""F3 -- a v6 resume is EXACT: a run interrupted after its step-N checkpoint and resumed with the SAME argv
draws the batches, and ends in the bit-identical state, of the uninterrupted run (CPU).

MEASURED defects this pins:
  * F3 (v7f launch gate, G-CKPT rehearsal, 2026-09-27, `…/2026-09-27-v7f-refav1-state/v7f_gate/RESULT.md`):
    `train()` re-seeded torch / random / the sampler rng / the shared generator from --seed at EVERY launch and
    `_save_ckpt` wrote only {stack, opt, step, config}, so a resumed run drew the uninterrupted run's FIRST
    batches;
  * F3b (found while fixing F3, 2026-09-27): the cosine LR schedule was replayed ON TOP of the optimiser's
    loaded (already decayed) `lr`; `CosineAnnealingLR` is recursive, so the whole resumed remainder ran at
    `(1 + cos(pi*n/T)) / 2` of its schedule (0.34549 on this rig's resume at 3 of 5).

Protocol (the only one under which "resume == uninterrupted" is a well-posed claim): run A is launched with the
FINAL --steps N+E and --save-every N, and dies right after its step-N checkpoint (a `_save_ckpt` wrapper raises
after the save -- the SIGKILL-after-checkpoint case); run B resumes A's directory with the SAME argv; run C is the
same argv uninterrupted. ⚠️ Resuming a run that FINISHED at --steps N with --steps N+E is a different experiment:
the cosine schedule is a function of --steps, so no resume fix can make it equal an uninterrupted N+E run (this
is why the launch gate's G-CKPT protocol was changed in the same patch).

Every expectation is a LITERAL (the draws were MEASURED once on this synthetic corpus and are pinned); each
regression arm re-introduces one MEASURED defect and must go RED.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]                    # stack/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

import train_v6_staged as T  # noqa: E402

N, E = 3, 2                    # checkpoint (and interrupt) at step N, then E more steps
BATCH = 2


class _Interrupt(BaseException):
    """Raised by the `_save_ckpt` wrapper right after the step-N save: the process dies there."""


def _episodes(a, *, n_eps: int = 3, seed: int = 20260927) -> list:
    """Seeded ToyEpisodes with the trainer's episode contract, from a DEDICATED generator (the run's own
    streams are not touched): uint8 frames at the encoder's frame, a unicycle pose track, (steer, accel)."""
    from tanitad.data.toy_driving import ToyEpisode
    g = torch.Generator().manual_seed(seed)
    c, h, w = int(a.in_channels), int(a.frame_h), int(a.frame_w)
    need = max(int(getattr(a, "plan_steps", 60) or 60), int(a.o1_k), int(a.o5_k),
               int(getattr(a, "max_horizon", 0) or 0), 20, int(getattr(a, "s1_multi_k", 2)) * 20)
    n_t = int(a.window) + need + 24
    dt = float(getattr(a, "dt", 0.1) or 0.1)
    eps = []
    for e in range(n_eps):
        v = 6.0 + 4.0 * float(torch.rand((), generator=g))
        yaw = x = y = 0.0
        poses, acts = [], []
        for t in range(n_t):
            steer = 0.02 * math.sin(0.1 * t + e)
            acc = 0.5 * math.cos(0.07 * t + 2 * e)
            poses.append((x, y, yaw, v))
            acts.append((steer, acc))
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += v * steer / 2.9 * dt
            v = max(0.0, v + acc * dt)
        frames = (torch.rand(n_t, c, h, w, generator=g) * 255).to(torch.uint8)
        eps.append(ToyEpisode(frames=frames, actions=torch.tensor(acts, dtype=torch.float32),
                              poses=torch.tensor(poses, dtype=torch.float32), episode_id=1_000_000 + e))
    return eps


def _argv(out: Path, cache: Path, *extra: str, steps: int = N + E, save_every: int = N) -> list[str]:
    """A TINY S-W launch on CPU (the launch gate's rehearsal overlay)."""
    return ["--stage", "S-W", "--out", str(out), "--steps", str(steps), "--batch", str(BATCH),
            "--lr", "0.0001", "--newest-frame-only", "--in-channels", "3", "--enc-dim", "32",
            "--enc-depth", "1", "--enc-heads", "2", "--frame-h", "64", "--frame-w", "128",
            "--horizons", "1", "--strategic-off", "--goal-multilabel", "--save-every", str(save_every),
            "--log-every", "1", "--patch", "16", "--pred-dim", "32", "--pred-depth", "1",
            "--pred-heads", "2", "--readout-dim", "8", "--d-tac", "32", "--d-str", "16",
            "--d-goal-embed", "16", "--adapter-hidden", "32", "--f-hidden-tac", "32",
            "--f-hidden-str", "32", "--window", "3", "--eps-per-batch", "1", "--device", "cpu",
            "--v2-cache", str(cache), *extra]


#: the synthetic nav label source (the launch gate's rehearsal pattern): one record per synthetic clip, served
#: through the trainer's REAL join (`assert_cache_join`) and `NavEmitter`
NAV_LABELS = "__f3_test_synthetic_nav__/nav.jsonl.gz"
NAV = ("--nav-cond", "--nav-labels", NAV_LABELS)
NO_NAV = ("--i-know-this-arm-predates-nav",)
T3_SCORES = "__f3_test_synthetic_t3__.pt"
#: T3 with a curriculum so shallow that round(alpha, 3) never moves: the live weights stay the ones computed at
#: step 1's UNROUNDED alpha -- exactly the state a resume must rebuild rather than recompute
T3 = ("--o4-alpha", "0", "--t3-scores", T3_SCORES, "--t3-alpha-start", "1.0", "--t3-alpha-end", "1.0004",
      "--t3-warmup-frac", "1.0")
SPECTRUM = ("--spectrum-every", "2", "--spectrum-accum", "2", "--spectrum-ci-reps", "4")
#: every rolling buffer and dedicated generator at once: the SIGReg row bank, the O6 pool + spec_gen (seed+7), X4's
#: per-layer pools + generator (seed+11), the observer monitor
ALL_MONITORS = SPECTRUM + ("--sigreg-accum", "3", "--x4-spectrum-layers", "tac,str", "--obs-monitor-every", "2")

#: the regression arms, each re-introducing ONE measured defect through a seam of the REAL trainer
DROPS = ("all", "loop_state", "t3", "lr_replay_on_loaded_lr")


def _clip_ids(n: int = 3) -> list[str]:
    return [f"f3-test-clip-{i:04d}" for i in range(n)]


def _digest(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def run_trainer(argv: list[str], *, module=None, interrupt_at: int | None = None,
                drop: str | None = None) -> dict:
    """Run the REAL `main(argv)` on the synthetic corpus; record every drawn window (ep_idx, t_last) and a
    digest of the O4/T3 sampler's weights at every draw. `interrupt_at`: die right after that step's
    checkpoint. `drop`: a REGRESSION ARM (see DROPS)."""
    import types

    import tanitad.data.v7_labels as V7L
    import train_flagship4b as TF4
    import train_v58f_unicycle_head as tv58
    from tanitad.models import v6 as V6
    M = module or T
    draws: list[tuple[int, int]] = []
    wdig: list[str] = []
    saved: list[int] = []
    o_get, o_build, o_save = TF4.FlagshipWindowDataset.__getitem__, tv58.build_train_episodes, M._save_ckpt
    # (getattr: a PRE-F3 trainer module -- the before/after probes -- has neither seam)
    o_apply, o_replay = getattr(M, "apply_resume_state", None), getattr(M, "replay_lr_schedule", None)
    o_t3 = M.load_t3_scores
    o_join, o_labels, o_call = M.join_clip_ids, V7L.load_v7_labels, V6.InteractionSampler.__call__

    def get(self, i):
        item = o_get(self, i)
        draws.append((int(item["ep_idx"]), int(item["t_last"])))
        return item

    def call(self, bs):
        wdig.append(_digest(self.weights))
        return o_call(self, bs)

    def save(path, **kw):
        o_save(path, **kw)
        saved.append(int(kw["step"]))
        if interrupt_at is not None and int(kw["step"]) == interrupt_at:
            raise _Interrupt()

    def labels(path, *aa, **kk):
        if not str(path).replace("\\", "/").endswith(NAV_LABELS):
            return o_labels(path, *aa, **kk)
        toks = tuple(V7L.NAV_TOKENS)
        recs = [types.SimpleNamespace(clip_id=c, _oracle={"nav_command": {
                    "token": toks[i % len(toks)],
                    "args": {"distance_m": 40.0 + 15.0 * i, "time_s": 4.0 + 2.0 * i}}})
                for i, c in enumerate(_clip_ids())]
        return recs, V7L.LabelManifest(path=str(path), md5="f3-test-synthetic", n_records=len(recs),
                                       schema_version="f3-test-synthetic", vocab="v7",
                                       allow_oracle_nav=bool(kk.get("allow_oracle_nav", False)))

    def t3_scores(path, *, n_windows):
        g = torch.Generator().manual_seed(7)
        return torch.rand(n_windows, generator=g), {"test": "synthetic T3 scores (seeded)"}

    def apply_arm(state, **kw):
        if drop == "all":                            # the pre-F3 resume: every stream restarts from --seed
            return {"t3_alpha_applied": None, "t3_prog_applied": None, "monitors": {}, "applied": []}
        if drop == "loop_state":                     # streams restored, the rolling buffers NOT
            state = dict(state, loop_state={})
        if drop == "t3":                             # T3 weights recomputed at the resume step's prog
            state = dict(state, data_pos=dict(state["data_pos"], t3_alpha_applied=None,
                                              t3_prog_applied=None))
        return o_apply(state, **kw)

    def replay_arm(opt, sched, n):                   # F3b: the replay on top of the LOADED lr
        for _ in range(int(n)):
            sched.step()

    TF4.FlagshipWindowDataset.__getitem__ = get
    V6.InteractionSampler.__call__ = call
    tv58.build_train_episodes = lambda a, *, cache_frame, train_frame: (_episodes(a), {"test": "synthetic"})
    M.join_clip_ids = lambda a, cache_dir: _clip_ids()
    V7L.load_v7_labels = labels
    M.load_t3_scores = t3_scores
    M._save_ckpt = save
    if drop in ("all", "loop_state", "t3"):
        M.apply_resume_state = apply_arm
    if drop == "lr_replay_on_loaded_lr":
        M.replay_lr_schedule = replay_arm
    rc = None
    try:
        rc = M.main(list(argv))
    except _Interrupt:
        rc = "interrupted"
    finally:
        TF4.FlagshipWindowDataset.__getitem__ = o_get
        V6.InteractionSampler.__call__ = o_call
        tv58.build_train_episodes = o_build
        M._save_ckpt = o_save
        if o_apply is not None:
            M.apply_resume_state = o_apply
        if o_replay is not None:
            M.replay_lr_schedule = o_replay
        M.load_t3_scores = o_t3
        M.join_clip_ids = o_join
        V7L.load_v7_labels = o_labels
    return {"rc": rc, "draws": draws, "weights": wdig, "saved": saved}


def _load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


def _rows(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def _train_rows(run_dir: Path) -> dict[int, dict]:
    """step -> the training row (a resumed dir holds A's rows, then B's -- the later one wins)."""
    return {int(r["step"]): r for r in _rows(run_dir) if "loss" in r and "step" in r and "run_start" not in r}


def _spectrum_rows(run_dir: Path) -> dict[int, dict]:
    return {int(r["step"]): r for r in _rows(run_dir) if "spectrum" in r}


def _tensor_diff(a: dict, b: dict) -> list[str]:
    ka, kb = set(a), set(b)
    bad = sorted(str(k) for k in ka ^ kb)
    for k in sorted(ka & kb, key=str):
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            if not (torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                bad.append(str(k))
        elif isinstance(x, dict):
            bad += [f"{k}.{s}" for s in _tensor_diff(x, y if isinstance(y, dict) else {})]
        elif x != y:
            bad.append(str(k))
    return bad


def judge(a_run: dict, b_run: dict, c_run: dict, b_dir: Path, c_dir: Path) -> tuple[list[str], dict]:
    """The exact-resume verdict: [] iff B == C on draws, sampler weights, per-step loss AND lr, the spectrum
    records, and the final model and optimiser state."""
    reasons: list[str] = []
    det = {"draws_resumed": b_run["draws"], "draws_uninterrupted_same_steps": c_run["draws"][N * BATCH:],
           "draws_uninterrupted_first_steps": c_run["draws"][:E * BATCH],
           "weights_resumed": b_run["weights"], "weights_uninterrupted_same_steps": c_run["weights"][N:]}
    if a_run["rc"] != "interrupted" or a_run["saved"] != [N]:
        reasons.append(f"run A was not interrupted right after its step-{N} save: {a_run['rc']} {a_run['saved']}")
    if b_run["rc"] != 0 or c_run["rc"] != 0:
        reasons.append(f"B/C did not complete: {b_run['rc']} / {c_run['rc']}")
    if det["draws_resumed"] != det["draws_uninterrupted_same_steps"]:
        reasons.append("draws: the resumed run did not draw the uninterrupted run's batches at the same steps")
    if det["weights_resumed"] != det["weights_uninterrupted_same_steps"]:
        reasons.append("sampler weights: the resumed run sampled under different O4/T3 weights")
    rb, rc_ = _train_rows(b_dir), _train_rows(c_dir)
    steps = range(N + 1, N + E + 1)
    det["loss_resumed"] = {s: (rb.get(s) or {}).get("loss") for s in steps}
    det["loss_uninterrupted"] = {s: (rc_.get(s) or {}).get("loss") for s in steps}
    det["lr_resumed"] = {s: (rb.get(s) or {}).get("lr") for s in steps}
    det["lr_uninterrupted"] = {s: (rc_.get(s) or {}).get("lr") for s in steps}
    if det["loss_resumed"] != det["loss_uninterrupted"]:
        reasons.append("losses: the resumed steps' losses differ from the uninterrupted run's")
    if det["lr_resumed"] != det["lr_uninterrupted"]:
        reasons.append("lr: the resumed steps' learning rates differ from the uninterrupted run's")
    sb, sc = _spectrum_rows(b_dir), _spectrum_rows(c_dir)
    det["spectrum_resumed"] = {s: sb[s] for s in steps if s in sb}
    det["spectrum_uninterrupted"] = {s: sc[s] for s in steps if s in sc}
    if det["spectrum_resumed"] != det["spectrum_uninterrupted"]:
        reasons.append("spectrum: the resumed run's O6 spectrum records differ from the uninterrupted run's")
    ckb, ckc = _load(b_dir / "ckpt.pt"), _load(c_dir / "ckpt.pt")
    det["final_step"] = (ckb.get("step"), ckc.get("step"))
    det["model_differ"] = _tensor_diff(ckb["stack"], ckc["stack"])
    det["opt_state_differ"] = _tensor_diff(ckb["opt"]["state"], ckc["opt"]["state"])
    det["opt_groups_equal"] = ckb["opt"]["param_groups"] == ckc["opt"]["param_groups"]
    det["n_model_tensors"] = len(ckc["stack"])
    if det["final_step"] != (N + E, N + E):
        reasons.append(f"final steps {det['final_step']}")
    if det["model_differ"]:
        reasons.append(f"model: {len(det['model_differ'])} tensors differ, e.g. {det['model_differ'][:3]}")
    if det["opt_state_differ"]:
        reasons.append(f"optimiser: {len(det['opt_state_differ'])} state entries differ")
    if not det["opt_groups_equal"]:
        reasons.append("optimiser: param_groups (incl. lr) differ")
    return reasons, det


def abc(tmp: Path, *extra: str, module=None, drop: str | None = None) -> tuple[list[str], dict]:
    """A (interrupted at N) -> B (resume, same argv) vs C (uninterrupted, same argv)."""
    cache = tmp / "empty_v2cache"
    cache.mkdir(exist_ok=True)
    (tmp / T3_SCORES).write_bytes(b"synthetic -- load_t3_scores is replaced by the test")
    extra = tuple(str(tmp / T3_SCORES) if x == T3_SCORES else x for x in extra)
    a_dir, c_dir = tmp / "interrupted", tmp / "uninterrupted"
    a = run_trainer(_argv(a_dir, cache, *extra), module=module, interrupt_at=N)
    b = run_trainer(_argv(a_dir, cache, *extra), module=module, drop=drop)
    c = run_trainer(_argv(c_dir, cache, *extra), module=module)
    reasons, det = judge(a, b, c, a_dir, c_dir)
    det["config_resume_rng"] = json.loads((a_dir / "config.json").read_text(encoding="utf-8")).get("resume_rng")
    det["a_dir"], det["c_dir"] = a_dir, c_dir
    return reasons, det


# ----------------------------------------------------------------------------------------------------------------
# MEASURED 2026-09-27 on this synthetic corpus (3 episodes, --batch 2, --eps-per-batch 1, default --seed) with the
# F3 trainer. (ep_idx, t_last) per drawn window.
# ----------------------------------------------------------------------------------------------------------------
#: the v7F path (nav on, default --o4-alpha 1.0 = InteractionSampler on the shared `gen`): steps 4-5 ...
O4_STEPS_4_5 = [(0, 47), (0, 44), (0, 56), (0, 30)]
#: ... and steps 1-2 -- what the pre-F3 resume RE-DREW (MEASURED on the no-restore arm)
O4_STEPS_1_2 = [(1, 47), (1, 6), (0, 36), (0, 59)]
#: the uniform sampler (--o4-alpha 0 = make_sampler on the python `rng`): steps 4-5 and 1-2
UNIFORM_STEPS_4_5 = [(2, 29), (2, 19), (1, 19), (1, 14)]
UNIFORM_STEPS_1_2 = [(1, 55), (1, 7), (1, 64), (1, 53)]
#: (1 + cos(pi * 3/5)) / 2 -- the factor F3b scaled the resumed remainder's lr by, on this rig
F3B_LR_FACTOR = 0.3454915028125263


@pytest.fixture(scope="module")
def v7f_positive(tmp_path_factory):
    return abc(tmp_path_factory.mktemp("f3_nav_o4"), *NAV)


def test_POSITIVE_v7f_path_resume_equals_uninterrupted_draws_losses_and_bit_identical_state(v7f_positive):
    reasons, det = v7f_positive
    assert reasons == []
    assert det["draws_resumed"] == O4_STEPS_4_5                  # the uninterrupted run's steps 4-5 ...
    assert det["draws_uninterrupted_same_steps"] == O4_STEPS_4_5
    assert det["draws_uninterrupted_first_steps"] == O4_STEPS_1_2   # ... NOT its first batches
    assert det["model_differ"] == [] and det["opt_state_differ"] == [] and det["opt_groups_equal"] is True
    assert det["final_step"] == (5, 5) and det["n_model_tensors"] > 0
    assert det["config_resume_rng"]["mode"] == "restored"
    assert det["config_resume_rng"]["saved_step"] == 3
    assert det["config_resume_rng"]["stream_mismatch"] == {}


def test_POSITIVE_the_checkpoint_carries_every_stream_and_the_data_position(v7f_positive):
    _r, det = v7f_positive
    ck = _load(det["a_dir"] / "ckpt_step3.pt")
    assert sorted(ck) == ["config", "data_pos", "loop_state", "opt", "rng_state", "stack", "step"]
    rs = ck["rng_state"]
    assert rs["schema"] == "v6-exact-resume/1"
    assert sorted(k for k, v in rs.items() if v is not None) == [
        "gen", "numpy", "python_random", "sampler_rng", "schema", "torch_cpu"]  # CPU run: no CUDA state
    assert ck["data_pos"]["step"] == 3 and ck["data_pos"]["next_step"] == 4
    assert ck["data_pos"]["stream"]["n_windows"] > 0 and ck["data_pos"]["stream"]["sampler"] == \
        "InteractionSampler"


@pytest.mark.parametrize("name,extra,literal", [
    ("nonav_default_path", NO_NAV, O4_STEPS_4_5),
    ("uniform_sampler_python_rng", NAV + ("--o4-alpha", "0"), UNIFORM_STEPS_4_5),
    ("sigreg_row_bank", NAV + ("--sigreg-accum", "3"), O4_STEPS_4_5),
    ("spectrum_monitors", NAV + SPECTRUM, O4_STEPS_4_5),
    ("t3_curriculum", NAV + T3, None),
    ("all_monitors_together", NAV + ALL_MONITORS, O4_STEPS_4_5),
])
def test_POSITIVE_every_stream_and_buffer_config_resumes_exactly(tmp_path, name, extra, literal):
    reasons, det = abc(tmp_path, *extra)
    assert reasons == [], name
    if literal is not None:
        assert det["draws_resumed"] == literal
    assert det["model_differ"] == [] and det["opt_state_differ"] == []
    if name == "spectrum_monitors":
        # the step-4 emission pools steps 3 AND 4 -- step 3's rows came through the checkpoint
        assert det["spectrum_resumed"][4]["spectrum_pooled"]["pooled_steps"] == 2
    if name == "t3_curriculum":
        assert len(det["weights_resumed"]) == E and det["weights_resumed"] == det["weights_uninterrupted_same_steps"]


# ---------------------------------------------------------------------------------------------------------- arms
def test_ARM_no_restore_REPLAYS_the_first_batches_and_FAILS(tmp_path):
    """The pre-F3 resume, re-introduced through the real trainer's seam: every stream restarts from --seed."""
    reasons, det = abc(tmp_path, *NAV, drop="all")
    r = " || ".join(reasons)
    assert "draws:" in r and "model:" in r
    assert det["draws_resumed"] == O4_STEPS_1_2                     # the MEASURED defect, literally
    assert det["draws_uninterrupted_same_steps"] == O4_STEPS_4_5


def test_ARM_uniform_sampler_no_restore_replays_the_python_rng(tmp_path):
    reasons, det = abc(tmp_path, *NAV, "--o4-alpha", "0", drop="all")
    assert "draws:" in " || ".join(reasons)
    assert det["draws_resumed"] == UNIFORM_STEPS_1_2


def test_ARM_F3b_lr_replayed_on_the_loaded_lr_FAILS_with_the_predicted_factor(tmp_path):
    """F3b re-introduced: the draws are right (F3 holds) but the lr of every resumed step is scaled by
    (1 + cos(pi*n/T)) / 2 -- MEASURED 0.34549 at n=3, T=5."""
    reasons, det = abc(tmp_path, *NAV, drop="lr_replay_on_loaded_lr")
    r = " || ".join(reasons)
    assert "lr:" in r and "model:" in r and "draws:" not in r
    ratio = det["lr_resumed"][4] / det["lr_uninterrupted"][4]
    assert abs(ratio - F3B_LR_FACTOR) < 1e-9


def test_ARM_rolling_buffers_dropped_moves_the_SIGReg_loss(tmp_path):
    reasons, det = abc(tmp_path, *NAV, "--sigreg-accum", "3", drop="loop_state")
    r = " || ".join(reasons)
    assert "losses:" in r and "draws:" not in r                    # the bank enters the O6 LOSS
    assert det["loss_resumed"][4] != det["loss_uninterrupted"][4]


def test_ARM_rolling_buffers_dropped_moves_the_pooled_spectrum(tmp_path):
    reasons, det = abc(tmp_path, *NAV, *SPECTRUM, drop="loop_state")
    assert "spectrum:" in " || ".join(reasons)
    assert det["spectrum_resumed"][4]["spectrum_pooled"]["pooled_steps"] == 1
    assert det["spectrum_uninterrupted"][4]["spectrum_pooled"]["pooled_steps"] == 2


def test_ARM_t3_weights_recomputed_at_the_resume_step_FAILS(tmp_path):
    reasons, det = abc(tmp_path, *NAV, *T3, drop="t3")
    assert "sampler weights:" in " || ".join(reasons)
    assert det["weights_resumed"][0] != det["weights_uninterrupted_same_steps"][0]


def test_ARM_SOURCE_mutation_deleting_the_restore_call_goes_RED(tmp_path):
    """The restore call REMOVED from the trainer SOURCE (compiled under its own module name): the positive
    protocol must fail on the draws -- a guard that survives a refactor of the seam the other arms patch."""
    src = (ROOT / "scripts" / "train_v6_staged.py").read_text(encoding="utf-8")
    call = "_rl = apply_resume_state(resume_state, device=device, rng=rng, gen=gen,"
    assert src.count(call) == 1
    mutated = src.replace(call, "_rl = (lambda *_a, **_k: {'t3_alpha_applied': None, 't3_prog_applied': None, "
                                "'monitors': {}, 'applied': []})(resume_state, device=device, rng=rng, gen=gen,")
    assert mutated.count("apply_resume_state(resume_state") == 0
    p = Path(tempfile.mkdtemp()) / "train_v6_staged_f3_mutant.py"
    p.write_text(mutated, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("train_v6_staged_f3_mutant", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_v6_staged_f3_mutant"] = mod
    spec.loader.exec_module(mod)
    reasons, det = abc(tmp_path, *NAV, module=mod)
    assert "draws:" in " || ".join(reasons)
    assert det["draws_resumed"] == O4_STEPS_1_2


# ------------------------------------------------------------------------------------------------ legacy + fresh
def test_LEGACY_checkpoint_without_the_payload_resumes_LOUDLY_and_records_absent_replayed(tmp_path, capsys):
    """A checkpoint written before F3 must still resume (refusing would strand every banked v6 ckpt), with a
    loud warning and `resume_rng.mode = "absent-replayed"` in config.json; the stream replays (the MEASURED
    defect, now declared); the lr is still exact (F3b's fix does not need the payload); and the NEXT
    checkpoint carries the payload."""
    cache = tmp_path / "empty_v2cache"
    cache.mkdir()
    d = tmp_path / "legacy"
    run_trainer(_argv(d, cache, *NAV), interrupt_at=N)
    ck = _load(d / "ckpt.pt")
    for k in T.RESUME_STATE_KEYS:
        ck.pop(k)
    torch.save(ck, d / "ckpt.pt")
    assert sorted(_load(d / "ckpt.pt")) == ["config", "opt", "stack", "step"]      # the pre-F3 layout
    capsys.readouterr()
    b = run_trainer(_argv(d, cache, *NAV))
    out = capsys.readouterr().out
    assert b["rc"] == 0
    assert "THIS RESUME REPLAYS THE DATA/RNG STREAM" in out
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["resume_rng"]["mode"] == "absent-replayed"
    assert b["draws"] == O4_STEPS_1_2                              # replayed -- and declared
    rows = _train_rows(d)
    lr4 = rows[4]["lr"]
    c = tmp_path / "ref"
    run_trainer(_argv(c, cache, *NAV))
    assert lr4 == _train_rows(c)[4]["lr"]                          # F3b is fixed without the payload
    fin = _load(d / "ckpt.pt")
    assert fin["step"] == 5 and set(T.RESUME_STATE_KEYS) <= set(fin)
    assert fin["config"]["resume_rng"]["mode"] == "absent-replayed"   # the record rides the checkpoint too


def test_FRESH_run_numerics_are_untouched_by_the_capture(tmp_path):
    """The capture is a pure read: a fresh run that checkpoints (and captures) at EVERY step is bit-identical,
    loss by loss and tensor by tensor, to one that captures only at its last step -- and a fresh run's
    config.json carries no resume record."""
    cache = tmp_path / "empty_v2cache"
    cache.mkdir()
    d1, d2 = tmp_path / "every_step", tmp_path / "last_step_only"
    r1 = run_trainer(_argv(d1, cache, *NAV, save_every=1))
    r2 = run_trainer(_argv(d2, cache, *NAV, save_every=1000))
    assert r1["saved"] == [1, 2, 3, 4, 5] and r2["saved"] == [5]
    assert r1["draws"] == r2["draws"]
    l1 = {s: r["loss"] for s, r in _train_rows(d1).items()}
    l2 = {s: r["loss"] for s, r in _train_rows(d2).items()}
    assert l1 == l2 and sorted(l1) == [1, 2, 3, 4, 5]
    c1, c2 = _load(d1 / "ckpt.pt"), _load(d2 / "ckpt.pt")
    assert _tensor_diff(c1["stack"], c2["stack"]) == [] and _tensor_diff(c1["opt"]["state"], c2["opt"]["state"]) == []
    assert torch.equal(c1["rng_state"]["torch_cpu"], c2["rng_state"]["torch_cpu"])
    assert "resume_rng" not in json.loads((d1 / "config.json").read_text(encoding="utf-8"))


def test_a_changed_stream_geometry_is_RECORDED_not_hidden(tmp_path):
    cache = tmp_path / "empty_v2cache"
    cache.mkdir()
    d = tmp_path / "run"
    run_trainer(_argv(d, cache, *NAV), interrupt_at=N)
    b = run_trainer(_argv(d, cache, *NAV, "--eps-per-batch", "2"))
    assert b["rc"] == 0
    rec = json.loads((d / "config.json").read_text(encoding="utf-8"))["resume_rng"]
    assert rec["mode"] == "restored-stream-mismatch"
    assert rec["stream_mismatch"] == {"eps_per_batch": [1, 2]}


def test_the_payload_needs_NO_unpickling_global(tmp_path):
    """`weights_only=True` loaders (eval scripts, snapshots) must see the allowlist they saw before F3: the payload is
    tensors + primitives only, with every buffer and generator populated. MEASURED: a PRE-F3 v6 checkpoint already
    needs `torch.torch_version.TorchVersion` (the config's provenance); a post-F3 one needs that and nothing more."""
    from torch.torch_version import TorchVersion
    cache = tmp_path / "empty_v2cache"
    cache.mkdir()
    d = tmp_path / "run"
    assert run_trainer(_argv(d, cache, *NAV, *ALL_MONITORS))["rc"] == 0
    ck = _load(d / "ckpt.pt")
    ls, rs = ck["loop_state"], ck["rng_state"]
    assert len(ls["sigreg_bank"]) == 2 and ls["spec_acc"] and sorted(ls["x4_acc"]) == ["str", "tac"]
    assert ls["obs_mon"] is not None and rs["spec_gen"] is not None and rs["x4_gen"] is not None
    p = tmp_path / "payload_only.pt"
    torch.save({k: ck[k] for k in T.RESUME_STATE_KEYS}, p)
    torch.load(p, map_location="cpu", weights_only=True)                      # NO allowlist at all
    with torch.serialization.safe_globals([TorchVersion]):
        torch.load(d / "ckpt.pt", map_location="cpu", weights_only=True)      # the pre-F3 allowlist


def test_save_ckpt_refuses_an_extra_key_that_would_overwrite_its_own(tmp_path):
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters())
    with pytest.raises(ValueError, match="would overwrite"):
        T._save_ckpt(tmp_path / "ckpt.pt", stack=m, opt=opt, step=1, cfg_json={"stage": "S-W"},
                     extra={"step": 99})
    T._save_ckpt(tmp_path / "ckpt.pt", stack=m, opt=opt, step=1, cfg_json={"stage": "S-W"})
    assert sorted(_load(tmp_path / "ckpt.pt")) == ["config", "opt", "stack", "step"]   # no extra: unchanged
