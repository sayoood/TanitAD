"""PREREG_V8F_SPEEDAVG: the instruments (driver, reader, dev-box runner), BUILT AND TESTED, NOT RUN on the real arm. CPU only, < 3 min.

What is pinned, and the deliberate-regression arm of each part:
  (A) the pure pieces       the five speed cells are the PREREG's (by name, index for index against the model's partition); mean_cells / replace_cells / source_code / partner_of at known values;
                            averaging a code with itself is the identity BIT FOR BIT.  Arms: a moved cell is refused.
  (B) the hook              on two tiny models: `self` reproduces `forward_infer` exactly, `avg` writes the MEAN into exactly the five cells and leaves every other cell alone, `gt` is the eval's own
                            oracle code cell for cell, and the wrapped `forward_diag` is restored afterwards (also when the eval raises).
  (C) end to end            the real `v8f_mini_eval.py run` on the synthetic world, two checkpoints, four modes each: controls 1-3 PASS, the averaged cells are the mean, tactical / strategic equal X_s.
                            ARMS (a corrupted input or a broken driver, each must turn ONE control red): self != own, gt != KINDX's speed oracle, own != banked, a hook that touches a non-speed cell, an average
                            that moves nothing, the two seeds averaging different pairs; and the real driver with `replace_cells` averaging ONE cell (control 2 red) and with the seeds swapped in `self` (control 1 red).
  (D) the committed table   every outcome row, the boundaries (exactly -F), the separation test, SECONDARY (each of speed MAE / accel MAE / three distance-keeping metrics / LEAD-VIOL), the PARTIAL ambiguity
                            flag, CAPTURE, the descriptives at known values; the table's text equals `Project Steering/PREREG_V8F_SPEEDAVG.md` cell for cell.
  (E) refusals              a limited (smoke) run, a wrong checkpoint sha, a mislabelled mode, a missing speedavg block, a failed control (rc 2, no outcome file).
  (F) the runner            syntax, `--dry-run` (every stage, the read-only box-2 pull, the markers; writes nothing), load-bearing lines, the guard's regex matches the driver, KINDX's argv flags.
No test launches anything on box 2, writes into box2_ship or touches a process it did not start."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.models import v8f_mini as M  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
DRIVER = REPO / "stack" / "scripts" / "v8f_mini_eval_speedavg.py"
KX_DRIVER = REPO / "stack" / "scripts" / "v8f_mini_eval_kindx.py"
READER = CODE / "v8f_speedavg_read.py"
RUNNER = CODE / "devbox_ops" / "run_speedavg.sh"
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_SPEEDAVG.md"
def _find_bash():
    """Git-for-Windows bash (the runner is a Git-bash script); `bash` on PATH may be the WSL launcher, which is not it."""
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and os.path.exists(c):
            try:
                r = subprocess.run([c, "-c", "echo ok; command -v df >/dev/null && echo df"], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.stdout.split() == ["ok", "df"]:
                return c
    return None


BASH = _find_bash()
bash = pytest.mark.skipif(BASH is None, reason="needs Git bash")
torch.set_num_threads(1)
os.environ["SPEEDAVG_THREADS"] = "1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_speedavg", Path(__file__).resolve().parent / "test_v8f_mini.py")
world, make_batch, tiny_model = _T.world, _T.make_batch, _T.tiny_model
SA = _load("v8f_mini_eval_speedavg_for_test", DRIVER)
# the lander imports this file from a stack-only tree: the package code dir is absent there, so load it only when present and skip otherwise
RD = _load("v8f_speedavg_read_for_test", READER) if READER.exists() else None
pytestmark = pytest.mark.skipif(RD is None, reason="the v8F-mini package code dir is absent (a stack-only tree)")
KX = _load("v8f_mini_eval_kindx_for_test", KX_DRIVER)
FIVE = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")


# ============================================================================================== (A) the pure pieces
def test_speed_idx_is_the_preregs_five_cells_and_a_moved_cell_is_refused(monkeypatch):
    assert tuple(SA.SPEED_CELLS) == FIVE, "the driver's cell set is the PREREG's five, by name and in order"
    idx = SA.speed_idx(M)
    c0 = M.N_GOAL + M.N_LAT + M.N_LON
    assert idx == sorted(c0 + M.CONS_NAMES.index(n) for n in FIVE) and len(idx) == 5
    assert idx == sorted(M.CODE_KINDS["speed"]) == list(M.code_group_index("speed"))
    bad = dict(M.CODE_KINDS)
    bad["speed"] = tuple(idx[:4]) + (c0 + M.CI["turn_dyaw_deg"],)           # v_hi_ms replaced by an angle cell
    monkeypatch.setattr(M, "CODE_KINDS", bad)
    with pytest.raises(SystemExit, match="REFUSED"):
        SA.speed_idx(M)


def test_mean_cells_and_replace_cells_known_values():
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b = torch.tensor([[3.0, 6.0, 7.0, 8.0]])
    assert SA.mean_cells(a, b).tolist() == [[2.0, 4.0, 5.0, 6.0]]
    x = torch.randn(64, 60) * 7.3
    assert torch.equal(SA.mean_cells(x, x), x), "averaging a code with itself must be the identity BIT FOR BIT (control 1 rests on it)"
    out = SA.replace_cells(a, b, [1, 3])
    assert out.tolist() == [[1.0, 6.0, 3.0, 8.0]] and a.tolist() == [[1.0, 2.0, 3.0, 4.0]], "replace_cells changes only its cells and never its input"


def test_source_code_and_partner_of():
    own, part = torch.randn(4, 60), torch.randn(4, 60)
    gt = torch.randn(4, 60)
    has = torch.tensor([True, False, True, False])
    assert torch.equal(SA.source_code("gt", own, None, gt, has), torch.where(has[:, None], gt, own))
    assert torch.equal(SA.source_code("avg", own, part, gt, has), (own + part) * 0.5)
    assert torch.equal(SA.source_code("self", own, own, gt, has), own)
    with pytest.raises(ValueError):
        SA.source_code("own", own, part, gt, has)
    a, b = object(), object()
    assert SA.partner_of("self", a, b) is a and SA.partner_of("avg", a, b) is b and SA.partner_of("own", a, b) is None and SA.partner_of("gt", a, b) is None


def test_the_eval_argv_is_kindxs_and_the_tag_and_guard_regex():
    paths = {"v9_eval": "v9", "lead_eval": "ld", "speed_eval": "sp", "p10": "p", "eval_dir": "ed", "cache": "c", "tracks": "t", "eval_grid": None, "eval_kin": None, "eval_join": None}
    own, orc = KX.eval_argv("own", "CK", "T", paths), KX.eval_argv("speed", "CK", "T", paths)
    mine_own, mine_orc = SA.e_argv("own", "CK", "T", paths), SA.e_argv("avg", "CK", "T", paths)
    assert mine_own == own[:-2] and own[-2:] == ["--project", "--families"], "own: KINDX's standard eval without --project / --families"
    assert mine_orc == orc and SA.e_argv("self", "CK", "T", paths) == orc and SA.e_argv("gt", "CK", "T", paths) == orc, "self / gt / avg ride KINDX's `--tac oracle --tac-groups speed --tac-alpha 1`"
    assert SA.tag_of("avg", 1) == "M2b-mixo_s1-speedavg-avg" and SA.tag_of("own", 0) == "M2b-mixo_s0-speedavg-own"
    regex = re.search(r"-match '([^']*v8f_mini_eval[^']*)'", GUARD.read_text(encoding="utf-8")).group(1)
    assert re.search(regex, "python.exe -I stack/scripts/v8f_mini_eval_speedavg.py evals --seed 0"), "the thermal / time guard that watches v8f_mini_eval must watch this driver"


def test_pair_violations_on_checkpoints(world, tmp_path):
    _, J, _ = make_batch(world)
    torch.manual_seed(1)
    m0, cfg = tiny_model(J)
    torch.manual_seed(2)
    m1, _ = tiny_model(J)
    ck = lambda m: {"cfg": dict(vars(cfg)), "grid": "pool4x10", "model": {k: v.clone() for k, v in m.state_dict().items()}}        # noqa: E731
    assert SA.pair_violations(ck(m0), ck(m1)) == []
    assert any("identical weights" in x for x in SA.pair_violations(ck(m0), ck(m0)))
    c1 = ck(m1)
    c1["model"]["cons_sd"] = c1["model"]["cons_sd"] * 1.0001
    assert any("cons_sd" in x for x in SA.pair_violations(ck(m0), c1)), "a different standardisation = a different code space"
    c2 = ck(m1)
    c2["cfg"]["hist"] = 3
    assert any("configs differ" in x for x in SA.pair_violations(ck(m0), c2))
    c3 = ck(m1)
    c3["model"].pop("pos")
    assert any("keys or shapes" in x for x in SA.pair_violations(ck(m0), c3))


def test_install_limit_cuts_the_windows_and_every_per_window_array_and_undoes():
    n_full = 10
    X = {"eid": np.arange(n_full), "gt": np.zeros((n_full, 30, 2)), "scalar": 3, "short": np.zeros(4)}
    E = SimpleNamespace(D=SimpleNamespace(eval_grid_windows=lambda g, J: list(range(n_full))), ctx_from_args=lambda a, k=None: ("grid", "J", dict(X)))
    orig_w, orig_c = E.D.eval_grid_windows, E.ctx_from_args
    undo = SA.install_limit(E, 3)
    assert E.D.eval_grid_windows("g", "J") == [0, 1, 2]
    _, _, X2 = E.ctx_from_args(None)
    assert X2["eid"].shape == (3,) and X2["gt"].shape == (3, 30, 2) and X2["scalar"] == 3 and X2["short"].shape == (4,)
    undo()
    assert E.D.eval_grid_windows is orig_w and E.ctx_from_args is orig_c


# ============================================================================================== (B) the hook on two tiny models
def _sens(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(model.head_o.parameters()) + list(model.e_code.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.3)
    return model.eval()


def test_hook_modes_on_tiny_models(world):
    b, J, _ = make_batch(world)
    torch.manual_seed(11)
    m0 = _sens(tiny_model(J)[0], 1)
    torch.manual_seed(12)
    m1 = _sens(tiny_model(J)[0], 2)
    idx = SA.speed_idx(M)
    non = [i for i in range(M.CODE_DIM) if i not in idx]
    orig = M.V8FMini.forward_diag
    with torch.no_grad():
        own = m0.forward_infer(b)
        c1 = m1.forward_infer(b)["code"]
    assert (own["code"][:, idx] - c1[:, idx]).abs().max() > 1e-3, "the two models' speed cells must differ, or the test below means nothing"

    def run(mode, other=None, override=None):
        h = SA.SpeedAvgHook(M, mode, other, idx).install()
        try:
            with torch.no_grad():
                o = m0.forward_diag(b, own["code"] if override is None else override)
        finally:
            h.restore()
        assert M.V8FMini.forward_diag is orig, "restore() puts the original forward_diag back (checked after EVERY mode, not only at the end)"
        return o, h
    o_self, h_self = run("self")
    assert torch.equal(o_self["traj30"], own["traj30"]) and torch.equal(o_self["code"], own["code"]), "CONTROL 1 at the model level: X averaged with itself changes nothing, bit for bit"
    o_avg, h_avg = run("avg", m1)
    expect = own["code"].clone()
    expect[:, idx] = (own["code"][:, idx] + c1[:, idx]) * 0.5
    assert torch.equal(o_avg["code"], expect), "avg writes the MEAN into exactly the five speed cells"
    assert torch.equal(o_avg["code"][:, non], own["code"][:, non]), "every other cell is X_s's own"
    assert (o_avg["traj30"] - own["traj30"]).abs().max() > 1e-4, "O is code-sensitive here, so the average moves its plan"
    assert h_avg.stats()["nonspeed_max_abs_dev"] == 0.0 and h_avg.stats()["oracle_code_max_abs_dev"] is None
    cells = h_avg.cells()
    assert np.array_equal(cells["used_std"], (cells["own_std"] + cells["partner_std"]) * np.float32(0.5)) and cells["own_std"].shape == (len(b["a_t"]), 5)
    # gt: the eval's own oracle code for `--tac-groups speed` (what v8f_mini_eval builds), cell for cell
    has = b["lab_t"]["has"].bool()
    code = torch.where(has[:, None], m0._code_gt(b["lab_t"]).to(own["code"].dtype), own["code"])
    sel = torch.zeros(M.CODE_DIM, dtype=torch.bool)
    sel[idx] = True
    oracle = torch.where(sel[None, :], code, own["code"])
    o_gt, h_gt = run("gt", None, oracle)
    assert torch.equal(o_gt["code"], oracle) and h_gt.stats()["oracle_code_max_abs_dev"] == 0.0 and h_gt.stats()["nonspeed_max_abs_dev"] == 0.0
    with torch.no_grad():
        ref = m0.forward_diag(b, oracle)
    assert torch.equal(o_gt["traj30"], ref["traj30"]), "CONTROL 2 at the model level: the GT in the five speed cells is KINDX's speed-only oracle"
    gtc = h_gt.cells()["gt_phys"]
    assert gtc.shape == (len(b["a_t"]), 5)
    assert M.V8FMini.forward_diag is orig, "the wrapped forward_diag is restored after every eval"


# ============================================================================================== (C) end to end on the synthetic world
def _ckpt(tmp, seed, like=None):
    """A tiny trained checkpoint. `like`: another checkpoint whose constraint standardisation (cons_mu / cons_sd) this one takes over: the real pair (MIXO's two seeds) trains on ONE train list, so its
    standardisation is bitwise equal; the synthetic trainer builds a different world per seed, so the stand-in pair is made to share it (the driver REFUSES a pair that does not: tested below)."""
    TR = _load(f"v8f_mini_train_for_speedavg_{seed}", REPO / "stack" / "scripts" / "v8f_mini_train.py")
    out = tmp / f"run{seed}"
    assert TR.main(["--out", str(out), "--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "4", "--log-every", "2", "--warmup", "1", "--ckpt-every", "4", "--threads", "1",
                    "--seed", str(seed)]) == 0
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    g = torch.Generator().manual_seed(1)
    for k, v in ck["model"].items():
        if k.startswith("head_o.") or k.startswith("e_code."):
            ck["model"][k] = v + torch.randn(v.shape, generator=g).to(v.device) * 0.3
    if like is not None:
        ref = torch.load(like, weights_only=False)["model"]
        for k in ("cons_mu", "cons_sd"):
            ck["model"][k] = ref[k].clone()
    p = tmp / f"ck_s{seed}.pt"
    torch.save(ck, p)
    return p


def _eval_args(world):
    root = Path(world["root"])
    return ["--eval-cache", str(root), "--tracks", str(root / "native_tracks.pt"), "--v9-eval", "x", "--lead-eval", "x", "--speed-eval", "x", "--p10", "x",
            "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"), "--n-boot", "20", "--batch", "16", "--lru", "4"]


def _driver(world, ck, sha, seed, evdir, modes=None):
    argv = ["evals", "--seed", str(seed), "--x0", str(ck[0]), "--x1", str(ck[1]), "--expect-sha0", sha[0], "--expect-sha1", sha[1], "--eval-dir", str(evdir), *_eval_args(world)]
    if modes:
        argv += ["--modes", modes]
    return SA.main(argv)


@pytest.fixture(scope="module")
def e2e(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sa")
    EV = _load("v8f_mini_eval_for_speedavg_ref", REPO / "stack" / "scripts" / "v8f_mini_eval.py")
    ck = {0: _ckpt(tmp, 3)}
    ck[1] = _ckpt(tmp, 4, like=ck[0])
    sha = {0: SA.sha256_file(str(ck[0])), 1: SA.sha256_file(str(ck[1]))}
    assert sha[0] != sha[1]
    root = Path(world["root"])
    kx = {}
    for s in (0, 1):                                                                      # KINDX's own and speed-only-oracle evals of X_s: the references of controls 2 and 3
        kx[s] = tmp / f"kx{s}"
        for nm, extra in (("own", ["--tac", "pred"]), ("only-speed", ["--tac", "oracle", "--tac-groups", "speed", "--tac-alpha", "1"])):
            assert EV.main(["run", "--ckpt", str(ck[s]), "--tag", f"M2b-mixo_s{s}-kindx-{nm}", "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
                            "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(kx[s]), "--tracks", str(root / "native_tracks.pt"),
                            "--n-boot", "20", "--batch", "16", "--lru", "4", *extra]) == 0
    ev = tmp / "ev"
    for s in (0, 1):
        assert _driver(world, ck, (sha[0], sha[1]), s, ev) == 0
    A, K, eid = RD.load_all(str(ev), {0: str(kx[0]), 1: str(kx[1])}, sha[0], sha[1], allow_smoke=True)
    return {"tmp": tmp, "ck": ck, "sha": sha, "kx": kx, "ev": ev, "A": A, "K": K, "eid": eid, "pf": RD.pf_default()}


def test_e2e_controls_pass_and_the_average_is_live(e2e):
    A, K, eid, pf = e2e["A"], e2e["K"], e2e["eid"], e2e["pf"]
    ctl = RD.controls(A, K, eid, pf, n_boot=30)
    assert ctl["PASS"], json.dumps(ctl, indent=1, default=str)[:3000]
    for s in (0, 1):
        r = ctl["seeds"][f"s{s}"]
        assert r["control_1_self_average_is_identity"]["ade6_max_abs_dev_m"] == 0.0 and r["control_1_self_average_is_identity"]["pred_equal"]
        assert r["control_2_gt_equals_kindx_speed_oracle"]["ade6_max_abs_dev_m"] <= 1e-4 and r["control_2_gt_equals_kindx_speed_oracle"]["driver_code_max_abs_dev"] == 0.0
        assert max(r["control_3_own_equals_banked"]["ade6_max_abs_dev_m"].values()) <= 1e-4
        assert all(v for k, v in r["guards"].items() if isinstance(v, bool))
        assert np.abs(A[s]["avg"]["pred"] - A[s]["own"]["pred"]).max() > 0, "the average moves O's plan on some window"
        assert A[s]["avg"]["json"]["tac"] == "speedavg-avg" and A[s]["avg"]["json"]["tac_manipulation"]["mode"] == "speedavg-avg", "the carrier's `oracle` label is relabelled"
    assert ctl["cross_seed_same_pair"]["PASS"]
    assert not np.array_equal(A[0]["avg"]["pred"], A[0]["own"]["pred"])


def test_e2e_averaged_cells_are_the_mean_of_the_two_seeds_and_s_and_t_are_untouched(e2e):
    A = e2e["A"]
    c0, c1 = A[0]["avg"]["cells"], A[1]["avg"]["cells"]
    assert np.array_equal(c0["used_std"], (c0["own_std"] + c1["own_std"]) * np.float32(0.5)), "seed 0's averaged cells = (X_0's + X_1's) / 2"
    assert np.max(np.abs(c0["used_std"] - c1["used_std"])) <= 1e-5, "both seeds averaged the SAME pair"
    for s in (0, 1):
        assert RD.same_json(A[s]["own"]["json"]["raw"]["strategic"], A[s]["avg"]["json"]["raw"]["strategic"]), "S is X_s's own"
        assert RD.same_json(A[s]["own"]["json"]["raw"]["tactical"], A[s]["avg"]["json"]["raw"]["tactical"]), "T's decisions are X_s's own"
        assert A[s]["avg"]["block"]["hook"]["nonspeed_max_abs_dev"] == 0.0
        assert A[s]["gt"]["block"]["hook"]["oracle_code_max_abs_dev"] == 0.0
        assert A[s]["avg"]["block"]["tier"] == "T1-class open loop (own held outputs; no recorded future action)" and "T0" in A[s]["gt"]["block"]["tier"]


def test_e2e_the_driver_is_resumable_and_idempotent(e2e, world):
    ev = e2e["ev"]
    before = {p.name: p.stat().st_mtime_ns for p in ev.glob("*")}
    assert _driver(world, e2e["ck"], (e2e["sha"][0], e2e["sha"][1]), 0, ev) == 0
    assert {p.name: p.stat().st_mtime_ns for p in ev.glob("*")} == before, "a finished mode is skipped, nothing is rewritten"


def test_e2e_the_driver_undoes_its_hook_and_its_limit_when_the_eval_raises(e2e, world, tmp_path):
    class Boom(Exception):
        pass

    def boom(argv):
        raise Boom()
    E = SimpleNamespace(D=SimpleNamespace(eval_grid_windows=lambda g, J: []), ctx_from_args=lambda a, k=None: (0, 0, {}), main=boom)
    orig_w, orig_c, orig_d = E.D.eval_grid_windows, E.ctx_from_args, M.V8FMini.forward_diag
    ck, sha = e2e["ck"], e2e["sha"]
    a = SA.build_parser().parse_args(["evals", "--seed", "0", "--x0", str(ck[0]), "--x1", str(ck[1]), "--expect-sha0", sha[0], "--expect-sha1", sha[1], "--eval-dir", str(tmp_path / "boom"),
                                      *_eval_args(world), "--modes", "avg", "--limit", "3"])
    with pytest.raises(Boom):
        SA.run_evals(a, M=M, E=E, log=lambda *_: None)
    assert M.V8FMini.forward_diag is orig_d and E.D.eval_grid_windows is orig_w and E.ctx_from_args is orig_c, "the hook and the limit are undone in a finally"


def test_e2e_driver_refuses_a_wrong_sha_the_same_file_twice_and_a_missing_dry_run_needs_nothing(e2e, world, tmp_path):
    ck, sha = e2e["ck"], e2e["sha"]
    with pytest.raises(SystemExit, match="REFUSED"):
        _driver(world, ck, ("0" * 64, sha[1]), 0, tmp_path / "e1", "own")
    other = _ckpt(tmp_path, 5)                                                       # a checkpoint with ITS OWN standardisation: a pair whose code spaces differ is refused
    with pytest.raises(SystemExit, match="cons_mu|cons_sd"):
        _driver(world, {0: ck[0], 1: other}, (sha[0], SA.sha256_file(str(other))), 0, tmp_path / "e1b", "own")
    with pytest.raises(SystemExit, match="same file|REFUSED"):
        SA.main(["evals", "--seed", "0", "--x0", str(ck[0]), "--x1", str(ck[0]), "--expect-sha0", sha[0], "--expect-sha1", sha[0], "--eval-dir", str(tmp_path / "e2"), *_eval_args(world)])
    with pytest.raises(SystemExit, match="REFUSED"):
        SA.main(["evals", "--seed", "0", "--x0", str(ck[0]), "--x1", str(ck[1]), "--eval-dir", str(tmp_path / "e3"), *_eval_args(world)])        # no expected shas
    # --dry-run prints the argv of each mode and touches nothing (it works on files that do not exist)
    assert SA.main(["evals", "--seed", "1", "--x0", "nope0", "--x1", "nope1", "--eval-dir", str(tmp_path / "e4"), *_eval_args(world), "--dry-run"]) == 0
    assert not (tmp_path / "e4").exists()


# ---- the deliberate-regression arms: a corrupted input must turn exactly the right control red
def _ctl(e2e, mutate=None):
    A, K = copy.deepcopy(e2e["A"]), copy.deepcopy(e2e["K"])
    if mutate:
        mutate(A, K)
    return RD.controls(A, K, e2e["eid"], e2e["pf"], n_boot=30)


def _only_red(ctl, s, which):
    """`which` is the one control (a key of the seed's block) that must be red; every other control of that seed stays green."""
    r = ctl["seeds"][f"s{s}"]
    assert not ctl["PASS"] and not r["PASS"]
    for k in ("control_1_self_average_is_identity", "control_2_gt_equals_kindx_speed_oracle", "control_3_own_equals_banked"):
        assert r[k]["PASS"] is (k != which), (k, r[k])


def test_arm_control_1_self_average_that_is_not_identity(e2e):
    def mut(A, K):
        A[0]["self"]["ade6"][0] += 1e-6                                          # control 1 is EXACT (0.0): this is 100x below the bars of controls 2 and 3
    c = _ctl(e2e, mut)
    assert not c["seeds"]["s0"]["control_1_self_average_is_identity"]["PASS"] and c["seeds"]["s0"]["control_2_gt_equals_kindx_speed_oracle"]["PASS"] and not c["PASS"]

    def mut2(A, K):
        A[1]["self"]["pred"] = A[1]["self"]["pred"] + 1e-6                       # an ADE-invisible change to the plan is still not the identity
    assert not _ctl(e2e, mut2)["seeds"]["s1"]["control_1_self_average_is_identity"]["PASS"]


def test_arm_control_2_gt_that_is_not_kindxs_speed_oracle(e2e):
    def mut(A, K):
        A[0]["gt"]["ade6"][1] += 2e-4
    _only_red(_ctl(e2e, mut), 0, "control_2_gt_equals_kindx_speed_oracle")

    def mut2(A, K):
        A[1]["gt"]["block"]["hook"]["oracle_code_max_abs_dev"] = 1e-3             # the driver's own code-level comparison with the eval's oracle code
    _only_red(_ctl(e2e, mut2), 1, "control_2_gt_equals_kindx_speed_oracle")

    def mut3(A, K):
        A[0]["gt"]["ade6"] = A[0]["own"]["ade6"].copy()                             # "GT" that is T's own code: reproduces nothing of KINDX's oracle
    c = _ctl(e2e, mut3)
    assert not c["seeds"]["s0"]["control_2_gt_equals_kindx_speed_oracle"]["PASS"], "the oracle must differ from own on this world, or control 2 is blind here"


def test_arm_control_3_own_that_is_not_the_banked_eval(e2e):
    def mut(A, K):
        K[0]["own"]["ade6"][2] += 2e-4
    c = _ctl(e2e, mut)
    assert not c["seeds"]["s0"]["control_3_own_equals_banked"]["PASS"] and c["seeds"]["s0"]["control_1_self_average_is_identity"]["PASS"]
    assert not c["seeds"]["s0"]["PASS"] and not c["PASS"], "a red control 3 fails the seed and the whole control block"
    banked = {0: {"ade6": e2e["K"][0]["own"]["ade6"] + 1.0}}
    c2 = RD.controls(e2e["A"], e2e["K"], e2e["eid"], e2e["pf"], banked=banked, n_boot=30)
    assert not c2["seeds"]["s0"]["control_3_own_equals_banked"]["PASS"], "the banked STANDARD eval is a second reference"


@pytest.mark.parametrize("guard,mut", [
    ("hook_leaves_nonspeed_cells_alone", lambda A, K: A[0]["avg"]["block"]["hook"].__setitem__("nonspeed_max_abs_dev", 1e-3)),
    ("tactical_and_strategic_equal_to_X_s", lambda A, K: A[0]["avg"]["json"]["raw"]["strategic"].__setitem__("n_valid", [0, 0, 0])),
    ("avg_cells_are_the_mean_of_the_two_seeds_cells", lambda A, K: A[0]["avg"]["cells"]["used_std"].__iadd__(1e-3)),
    ("self_cells_equal_own_cells", lambda A, K: A[1]["self"]["cells"]["partner_std"].__iadd__(1e-3)),
    ("avg_is_live_it_moves_the_plan", lambda A, K: A[1]["avg"].__setitem__("pred", A[1]["own"]["pred"].copy())),
])
def test_arm_each_guard_goes_red_on_its_own_corruption(e2e, guard, mut):
    c = _ctl(e2e, mut)
    s = "s1" if guard in ("self_cells_equal_own_cells", "avg_is_live_it_moves_the_plan") else "s0"
    assert c["seeds"][s]["guards"][guard] is False and not c["seeds"][s]["guards_PASS"] and not c["PASS"], guard


def test_arm_cross_seed_the_two_avg_arms_must_average_the_same_pair(e2e):
    def mut(A, K):
        A[1]["avg"]["cells"]["partner_std"] = A[1]["avg"]["cells"]["own_std"].copy()          # seed 1 averaged X_1 with itself
    c = _ctl(e2e, mut)
    assert not c["cross_seed_same_pair"]["PASS"] and not c["PASS"]

    def mut2(A, K):
        A[0]["avg"]["cells"]["used_std"] = A[0]["avg"]["cells"]["own_std"].copy()
    assert not _ctl(e2e, mut2)["cross_seed_same_pair"]["PASS"]


def test_arm_the_real_driver_averaging_one_cell_is_caught_by_control_2(e2e, world, monkeypatch):
    idx = SA.speed_idx(M)

    def one_cell(code, src, ix):
        out = code.clone()
        out[:, ix[:1]] = src[:, ix[:1]]
        return out
    monkeypatch.setattr(SA, "replace_cells", one_cell)
    d = e2e["tmp"] / "mut_onecell"
    _driver(world, e2e["ck"], (e2e["sha"][0], e2e["sha"][1]), 0, d, "gt")
    monkeypatch.undo()
    E = RD.load_eval(str(d), "M2b-mixo_s0-speedavg-gt", need_cells=True)
    E["block"] = RD.check_arm(E, 0, "gt", e2e["sha"][0], e2e["sha"][1], True)
    assert len(idx) == 5 and E["block"]["hook"]["oracle_code_max_abs_dev"] > 0, "the driver's own code-level check sees the missing cells"

    def mut(A, K):
        A[0]["gt"] = E
    c = _ctl(e2e, mut)
    r = c["seeds"]["s0"]["control_2_gt_equals_kindx_speed_oracle"]
    assert not r["PASS"] and r["ade6_max_abs_dev_m"] > 1e-4, "CONTROL 2 catches an average (here: a GT replacement) that covers one cell of five"
    assert c["seeds"]["s0"]["control_1_self_average_is_identity"]["PASS"], "control 1 is blind to it: mean(a, a) = a for any number of cells; control 2 is the one that sees it"


def test_arm_the_real_driver_with_the_seeds_swapped_in_self_is_caught_by_control_1(e2e, world, monkeypatch):
    monkeypatch.setattr(SA, "partner_of", lambda mode, model, other: other if mode in ("self", "avg") else None)         # `self` averages with the OTHER seed
    d = e2e["tmp"] / "mut_swap"
    _driver(world, e2e["ck"], (e2e["sha"][0], e2e["sha"][1]), 0, d, "self,avg")
    monkeypatch.undo()
    E = RD.load_eval(str(d), "M2b-mixo_s0-speedavg-self", need_cells=True)
    E["block"] = RD.check_arm(E, 0, "self", e2e["sha"][0], e2e["sha"][1], True)

    def mut(A, K):
        A[0]["self"] = E
    c = _ctl(e2e, mut)
    r = c["seeds"]["s0"]["control_1_self_average_is_identity"]
    assert not r["PASS"] and r["ade6_max_abs_dev_m"] > 0.0, "CONTROL 1 catches a `self` that was averaged with the other seed"
    assert c["seeds"]["s0"]["control_2_gt_equals_kindx_speed_oracle"]["PASS"] and c["seeds"]["s0"]["control_3_own_equals_banked"]["PASS"]


# ============================================================================================== (D) the committed table
def _g(point, sep=True):
    """A paired block: separated -> the CI sits on one side of 0; not separated -> it straddles 0."""
    if sep:
        lo, hi = (point - 0.04, min(point + 0.04, -1e-4)) if point < 0 else (max(point - 0.04, 1e-4), point + 0.04)
    else:
        lo, hi = min(point - 0.05, -0.01), max(point + 0.05, 0.01)
    return {"point": point, "delta": round(point, 4), "lo": lo, "hi": hi, "separated": sep}


F = 0.1207


@pytest.mark.parametrize("p0,p1,sec,want", [
    (_g(-0.30), _g(-0.20), (True, True), "VARIANCE PAYS"),
    (_g(-F), _g(-F), (True, True), "VARIANCE PAYS"),                                       # exactly -F is "<= -F"
    (_g(-0.1206), _g(-0.30), (True, True), "PARTIAL"),                                     # one seed a hair above -F: the bar "at BOTH seeds" is missed
    (_g(-0.05), _g(-0.06), (True, True), "PARTIAL"),
    (_g(-0.30), _g(-0.05), (True, True), "PARTIAL"),                                       # the pair the row is silent on (interpretation 3)
    (_g(-0.30), _g(-0.20), (False, True), "MIXED"),                                        # PRIMARY met at both, SECONDARY separated-worse at seed 0
    (_g(-0.30), _g(-0.20), (True, False), "MIXED"),
    (_g(-0.50, sep=False), _g(-0.20), (True, True), "MIXED"),                              # one seed separated, one not
    (_g(-0.50, sep=False), _g(-0.60, sep=False), (True, True), "BIAS, NOT VARIANCE"),      # a big point that is not separated is no evidence: dropping the separation test would call this VARIANCE PAYS
    (_g(0.02, sep=False), _g(-0.01, sep=False), (True, True), "BIAS, NOT VARIANCE"),
    (_g(0.20), _g(0.30), (True, True), "MIXED"),                                           # separated WORSE at both: not "not separated", so not BIAS
    (_g(0.20), _g(0.02, sep=False), (True, True), "MIXED"),
    (_g(-0.30), _g(0.02, sep=False), (True, True), "MIXED"),
])
def test_the_outcome_table(p0, p1, sec, want):
    d = RD.decide({0: p0, 1: p1}, {0: sec[0], 1: sec[1]}, F)
    assert d["outcome"] == want, d


def test_partial_flags_the_reading_dependence_only_where_it_exists():
    d = RD.decide({0: _g(-0.30), 1: _g(-0.05)}, {0: True, 1: True}, F)
    assert d["outcome"] == "PARTIAL" and d["partial_reading_dependent"] and d["seeds_above_minus_F"] == ["s1"]
    d2 = RD.decide({0: _g(-0.05), 1: _g(-0.06)}, {0: True, 1: True}, F)
    assert d2["outcome"] == "PARTIAL" and not d2["partial_reading_dependent"]


def test_primary_and_secondary_predicates():
    assert RD.primary_met(_g(-0.1207)) and not RD.primary_met(_g(-0.12069)) and not RD.primary_met(_g(-0.5, sep=False)) and not RD.primary_met(_g(0.5))
    assert RD.separated_worse({"lo": 0.01, "hi": 0.2, "delta": 0.1}) and not RD.separated_worse({"lo": -0.01, "hi": 0.2, "delta": 0.1, "separated": False})
    assert not RD.separated_worse({"lo": -0.2, "hi": -0.01, "delta": -0.1, "separated": True}), "separated BETTER is not worse"
    assert RD.F_BAR == 0.1207


def _fam(**over):
    blk = lambda lo, hi: {"delta": (lo + hi) / 2, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}            # noqa: E731
    ok = (-0.05, 0.05)
    d = {"speed_mae_mps": blk(*ok), "accel_mae_mps2": blk(*ok)}
    d.update(over.get("lon", {}))
    return {"longitudinal": d, "lateral": {"heading_mae_deg": blk(*ok)}}


def _pair(**over):
    blk = lambda lo, hi: {"delta": (lo + hi) / 2, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}            # noqa: E731
    ok = (-0.05, 0.05)
    ae = {k: blk(*ok) for k in RD.DK_KEYS}
    ae.update(over.get("ae", {}))
    return {"abs_error_vs_human": ae, "lead_viol": {"all_lead_windows": over.get("lv", blk(*ok))}}


def test_secondary_each_of_the_six_metrics_can_fail_it():
    bad = (0.01, 0.2)
    assert RD.secondary_of(_fam(), _pair())["not_separated_worse"]
    cases = [("speed MAE", _fam(lon={"speed_mae_mps": {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}}), _pair()),
             ("accel MAE", _fam(lon={"accel_mae_mps2": {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}}), _pair()),
             ("headway", _fam(), _pair(ae={"headway_min_m": {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}})),
             ("time gap", _fam(), _pair(ae={"time_gap_min_s": {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}})),
             ("min TTC", _fam(), _pair(ae={"min_ttc_s": {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}})),
             ("LEAD-VIOL", _fam(), _pair(lv={"delta": 0.1, "lo": bad[0], "hi": bad[1], "separated": True}))]
    for nm, f, p in cases:
        assert not RD.secondary_of(f, p)["not_separated_worse"], nm
    better = _fam(lon={"speed_mae_mps": {"delta": -0.3, "lo": -0.4, "hi": -0.2, "separated": True}})
    assert RD.secondary_of(better, _pair())["not_separated_worse"], "separated BETTER passes"
    for f, p in ((_fam(lon={"speed_mae_mps": {"status": "UNAVAILABLE"}}), _pair()), (_fam(), {"abs_error_vs_human": {}, "lead_viol": {}}), (_fam(), None)):
        with pytest.raises(SystemExit) as e:
            RD.secondary_of(f, p)
        assert e.value.code == 2, "a missing family metric is refused, never dropped"


def test_capture_known_values_and_the_not_defined_case():
    eid = np.repeat(np.arange(6), 5)
    own, avg = np.full(30, 2.0), np.full(30, 1.5)
    c = RD.capture(own, avg, np.full(30, 2.0), np.full(30, 1.0), eid, n_boot=40)
    assert abs(c["point"] - 0.5) < 1e-12 and abs(c["recovered_m"] - 0.5) < 1e-12 and abs(c["gain_speed_m"] - 1.0) < 1e-12
    worse = RD.capture(own, np.full(30, 2.5), np.full(30, 2.0), np.full(30, 1.0), eid, n_boot=40)
    assert worse["point"] < 0, "an average that HURTS has a negative capture"
    assert RD.capture(own, avg, np.full(30, 1.0), np.full(30, 2.0), eid, n_boot=40)["status"] == "NOT-DEFINED"


def test_descriptives_known_values():
    rng = np.random.default_rng(0)
    n = 400
    mu, sd = np.array([8.0, 7.0, 6.0, 5.0, 12.0]), np.array([3.0, 2.0, 2.5, 2.0, 4.0])
    gt = mu + rng.normal(size=(n, 5)) * sd
    e0, e1 = rng.normal(size=(n, 5)) * 0.8, rng.normal(size=(n, 5)) * 0.8
    to_std = lambda x: ((x - mu) / sd).astype(np.float32)                                                 # noqa: E731
    c0 = {"own_std": to_std(gt + e0), "used_std": to_std(gt + 0.5 * (e0 + e1)), "gt_phys": gt.astype(np.float32), "has": np.ones(n, bool), "cons_mu": mu.astype(np.float32), "cons_sd": sd.astype(np.float32)}
    c1 = {"own_std": to_std(gt + e1)}
    d = RD.descriptives(c0, c1)
    r0 = d["per_cell"]["v_a_ms"]
    assert abs(r0["rmse_to_gt_X0"] - np.sqrt((e0[:, 1] ** 2).mean())) < 1e-3 and abs(r0["rmse_to_gt_averaged"] - np.sqrt((((e0 + e1) / 2)[:, 1] ** 2).mean())) < 1e-3
    assert abs(r0["corr_error_X0_X1"] - np.corrcoef(e0[:, 1], e1[:, 1])[0, 1]) < 1e-3
    assert r0["rmse_ratio_measured"] < 0.85 and abs(r0["rmse_ratio_if_pure_variance"] - np.sqrt((1 + r0["corr_error_X0_X1"]) / 2)) < 1e-9, "independent errors: the average removes ~ 1/sqrt(2)"
    shared = dict(c0, used_std=to_std(gt + e0))
    d2 = RD.descriptives(shared, {"own_std": to_std(gt + e0)})
    assert abs(d2["per_cell"]["v_a_ms"]["rmse_ratio_measured"] - 1.0) < 1e-3 and abs(d2["per_cell"]["v_a_ms"]["rmse_ratio_if_pure_variance"] - 1.0) < 1e-9, "identical errors: bias-like, nothing to remove"


def test_the_tables_text_is_the_preregs_text():
    txt = PREREG.read_text(encoding="utf-8")
    for o, (cond, nxt) in RD.TABLE.items():
        assert f"**{o}** | {cond} | {nxt} |" in txt, o
    assert tuple(RD.TABLE) == RD.OUTCOMES == ("VARIANCE PAYS", "PARTIAL", "BIAS, NOT VARIANCE", "MIXED")
    assert "F = 0.1207 m" in txt and RD.F_BAR == 0.1207 and RD.B_BOOT == 2000 and (RD.N_WINDOWS, RD.N_EPISODES) == (2399, 141)
    for n in FIVE:
        assert f"`{n}`" in txt
    assert tuple(SA.SPEED_CELLS) == tuple(RD.SPEED_CELLS) == FIVE
    assert "(Δ = 0.0 per window)" in txt and "≤ 1e-4 m" in txt, "the controls' bars are the PREREG's"
    assert RD.TOL_M == 1e-4
    assert "| own-code ADE6, SPEEDAVG_s − X_s | **≤ −F and separated, at BOTH seeds** |" in txt


# ============================================================================================== (E) refusals, the result block, the output
def _stub_fams(e2e, bad=None):
    blk = lambda lo, hi: {"delta": (lo + hi) / 2, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}            # noqa: E731
    out = {}
    for s in (0, 1):
        fam = _fam()
        pair = _pair(**(bad or {}))
        dk = {"arms": {f"M2b-mixo_s{s}-speedavg-avg": {"stub": True}}, "pairs": {}}
        out[s] = (fam, dk, pair)
    return out


def test_build_result_block_and_output_text(e2e):
    A, K, eid, pf = e2e["A"], e2e["K"], e2e["eid"], e2e["pf"]
    ctl = RD.controls(A, K, eid, pf, n_boot=30)
    fams = _stub_fams(e2e)
    res = RD.build_result(A, K, eid, ctl, pf, fams, n_boot=60)
    assert res["OUTCOME"] in RD.OUTCOMES and res["committed_next"] == RD.TABLE[res["OUTCOME"]][1]
    assert res["tier"] == "T1-class open loop (own held outputs; no recorded future action)"
    assert sum(v["this_read"] for v in res["outcome_table_verbatim"].values()) == 1 and list(res["outcome_table_verbatim"]) == list(RD.OUTCOMES)
    assert res["PRIMARY"]["s0"]["point"] == pytest.approx(float(A[0]["avg"]["ade6"].mean() - A[0]["own"]["ade6"].mean()), abs=1e-12)
    assert set(res["CAPTURE"]) == {"s0", "s1"} and res["controls"]["PASS"]
    res["beside"] = RD.beside_block(A, K, eid, fams, pf, n_boot=60)
    txt = json.dumps(res, default=str) + RD.md_of(res)
    assert "paused" not in txt.lower(), "STRATEGIC is in force for v8F: the word never appears"
    assert "T0-open-loop" not in txt, "the eval json's own tier string is a known mislabel and is not copied"
    for s in (0, 1):
        b = res["beside"]["families"][f"s{s}"]
        assert b["STRATEGIC"]["equal_to_X_s"] is True and "in force for v8F" in b["STRATEGIC"]["status"] and b["TACTICAL"]["equal_to_X_s"] is True
        assert set(b) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}, "all four families"
    p = e2e["tmp"] / "R.json"
    RD._dump(str(p), res)
    assert json.loads(p.read_text(encoding="utf-8"))["OUTCOME"] == res["OUTCOME"]
    for f in (READER, DRIVER, RUNNER):
        assert "paused" not in f.read_text(encoding="utf-8").lower(), f.name


def _copy_ev(e2e, name):
    d = e2e["tmp"] / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(e2e["ev"], d)
    return d


def _patch_json(d, tag, fn):
    p = d / f"eval_{tag}.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    fn(j)
    p.write_text(json.dumps(j), encoding="utf-8")


def _load_from(e2e, d, smoke=True):
    return RD.load_all(str(d), {0: str(e2e["kx"][0]), 1: str(e2e["kx"][1])}, e2e["sha"][0], e2e["sha"][1], allow_smoke=smoke)


def test_refusals_of_load_all(e2e):
    sha = e2e["sha"]
    d = _copy_ev(e2e, "r1")
    _patch_json(d, "M2b-mixo_s0-speedavg-avg", lambda j: j["speedavg"].__setitem__("limit", 48))
    with pytest.raises(SystemExit) as e:
        _load_from(e2e, d, smoke=False)
    assert e.value.code == 2
    _load_from(e2e, d, smoke=True)                                                    # a smoke may load a limited run (and never reads an outcome from it)
    d = _copy_ev(e2e, "r2")
    _patch_json(d, "M2b-mixo_s1-speedavg-gt", lambda j: j["speedavg"].__setitem__("x1_sha256", "f" * 64))
    with pytest.raises(SystemExit) as e:
        _load_from(e2e, d)
    assert e.value.code == 2
    d = _copy_ev(e2e, "r3")
    _patch_json(d, "M2b-mixo_s0-speedavg-self", lambda j: j["speedavg"].__setitem__("mode", "avg"))
    with pytest.raises(SystemExit):
        _load_from(e2e, d)
    d = _copy_ev(e2e, "r4")
    _patch_json(d, "M2b-mixo_s0-speedavg-avg", lambda j: j.pop("speedavg"))
    with pytest.raises(SystemExit):
        _load_from(e2e, d)
    d = _copy_ev(e2e, "r5")
    _patch_json(d, "M2b-mixo_s1-speedavg-avg", lambda j: j.__setitem__("tac", "oracle"))                 # the carrier label was not relabelled
    with pytest.raises(SystemExit):
        _load_from(e2e, d)
    d = _copy_ev(e2e, "r6")
    os.remove(d / "speedavg_cells_M2b-mixo_s0-speedavg-avg.npz")
    with pytest.raises(SystemExit):
        _load_from(e2e, d)
    d = _copy_ev(e2e, "r7")
    _patch_json(d, "M2b-mixo_s0-speedavg-own", lambda j: j.__setitem__("ckpt", "elsewhere/ckpt_last.pt"))
    with pytest.raises(SystemExit):
        _load_from(e2e, d)
    with pytest.raises(SystemExit) as e:
        _load_from(e2e, e2e["ev"], smoke=False)
    assert e.value.code == 2, "the committed 2,399 windows over 141 episodes"


def test_check_arm_refuses_a_limited_run_on_its_own(e2e, capsys):
    sha = e2e["sha"]
    E = copy.deepcopy(e2e["A"][0]["avg"])
    assert RD.check_arm(E, 0, "avg", sha[0], sha[1]) is E["json"]["speedavg"], "the unmodified arm is accepted"
    E["json"]["speedavg"]["limit"] = 48
    with pytest.raises(SystemExit) as e:
        RD.check_arm(E, 0, "avg", sha[0], sha[1], allow_smoke=False)
    assert e.value.code == 2 and "LIMITED run" in capsys.readouterr().err, "a limited run is a smoke, never a result (isolated: load_all's window-count refusal is not in play here)"
    RD.check_arm(E, 0, "avg", sha[0], sha[1], allow_smoke=True)
    for bad, why in (((sha[1], sha[0]), "the two checkpoints' sha256 in the wrong order"),):
        E2 = copy.deepcopy(e2e["A"][1]["gt"])
        with pytest.raises(SystemExit) as e:
            RD.check_arm(E2, 1, "gt", *bad)
        assert e.value.code == 2 and "expected" in capsys.readouterr().err, why
    E3 = copy.deepcopy(e2e["A"][0]["self"])
    with pytest.raises(SystemExit) as e:
        RD.check_arm(E3, 0, "avg", sha[0], sha[1])
    assert e.value.code == 2 and "expected" in capsys.readouterr().err, "an eval that is the self arm cannot be passed off as the avg arm"


def _cli(e2e, ev, out, *extra):
    return RD.main(["--eval-dir", str(ev), "--kindx-dir-s0", str(e2e["kx"][0]), "--kindx-dir-s1", str(e2e["kx"][1]), "--expect-sha0", e2e["sha"][0], "--expect-sha1", e2e["sha"][1],
                    "--out", str(out), "--n-boot", "30", *extra])


def test_cli_smoke_passes_and_a_failed_control_is_rc_2_with_no_outcome(e2e):
    n = int(e2e["eid"].size)
    out = e2e["tmp"] / "cli" / "SMOKE_READ.json"
    out.parent.mkdir(exist_ok=True)
    assert _cli(e2e, e2e["ev"], out, "--smoke-first-n", str(n)) == 0
    assert json.loads((out.parent / "SMOKE_READ_controls.json").read_text(encoding="utf-8"))["PASS"] is True and not out.exists(), "a smoke reads no outcome"
    with pytest.raises(SystemExit) as e:
        _cli(e2e, e2e["ev"], out, "--smoke-first-n", str(n + 1))
    assert e.value.code == 2
    d = _copy_ev(e2e, "cli_bad")
    z = dict(np.load(d / "eval_M2b-mixo_s0-speedavg-self.npz"))
    z["ade6"] = z["ade6"] + 1e-3
    np.savez_compressed(d / "eval_M2b-mixo_s0-speedavg-self.npz", **z)
    out2 = e2e["tmp"] / "cli" / "BAD_READ.json"
    with pytest.raises(SystemExit) as e:
        _cli(e2e, d, out2, "--smoke-first-n", str(n))
    assert e.value.code == 2 and not out2.exists(), "a failed control: rc 2 and NO outcome file"
    assert json.loads((out2.parent / "BAD_READ_controls.json").read_text(encoding="utf-8"))["seeds"]["s0"]["control_1_self_average_is_identity"]["PASS"] is False


# ============================================================================================== (F) the runner
@bash
def test_runner_syntax_dry_run_and_load_bearing_lines(tmp_path):
    src = RUNNER.read_text(encoding="utf-8")
    assert subprocess.run([BASH, "-n", str(RUNNER)], capture_output=True).returncode == 0
    out_dir = tmp_path / "never"
    env = dict(os.environ, SPEEDAVG_OUT=str(out_dir).replace("\\", "/"), SPEEDAVG_B2=str(tmp_path / "nobox").replace("\\", "/"), SPEEDAVG_PY=sys.executable.replace("\\", "/"))
    r = subprocess.run([BASH, str(RUNNER), "--dry-run"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    o = r.stdout
    for need in ("== check", "== fetch", "== evals seed 0 (own, self, gt, avg)", "== evals seed 1 (own, self, gt, avg)", "== read", "--check-inputs", "ZZSPEEDAVG-DONE", "ZZSPEEDAVG-STOP",
                 "scp -q -o BatchMode=yes", "tanitad-box2:D:/tanitad/runs/v8f_mini_full/M2b_s1_mixo/ckpt_last.pt", "tanitad-box2:D:/tanitad/runs/v8f_mini_full/M2b_s0_mixo/ckpt_last.pt",
                 "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", "speedavg-own", "speedavg-self", "speedavg-gt", "speedavg-avg", "--tac oracle --tac-groups speed --tac-alpha 1", "CPU only"):
        assert need in o, need
    assert not out_dir.exists(), "a dry run writes nothing"
    for need in ("CUDA_VISIBLE_DEVICES=-1", "SPEEDAVG_THREADS", 'MIN_FREE_KB=2097152', "ZZSPEEDAVG-STOP", "ZZSPEEDAVG-DONE", "BatchMode=yes", "sha256 of $dst is $got, expected $want",
                 "OUTCOME CARRIER", "'speed' in", "--check-inputs", "[ \"$free\" -ge \"$MIN_FREE_KB\" ]"):
        assert need in src, need
    assert "kill" not in src and "taskkill" not in src and "git add" not in src and "git commit" not in src and "git push" not in src and " rm " not in src, "the runner kills, stages and removes nothing"
    # box 2 is touched READ-ONLY: every `$SCP` is a pull (the remote is the SOURCE), every `$SSH` call is a hash
    scp_lines = [ln for ln in src.splitlines() if "$SCP " in ln and not ln.lstrip().startswith("#") and "SCP=" not in ln]
    ssh_lines = [ln for ln in src.splitlines() if "$SSH " in ln and not ln.lstrip().startswith("#") and "SSH=" not in ln]
    assert scp_lines and ssh_lines
    for ln in scp_lines:
        if ln.lstrip().startswith("echo"):
            continue
        assert re.search(r'\$SCP\s+"tanitad-box2:[^"]+"\s+"\$dst"', ln), ln
    for ln in ssh_lines:
        assert "Get-FileHash" in ln and not re.search(r"Remove|Stop-|Set-|New-Item|Move|Copy|del|rm|Start-Process|Invoke-", ln), ln
    # stage order of `all`
    m = re.search(r"all\)\s+(stage_check; stage_fetch; stage_evals 0; stage_evals 1; stage_read)", src)
    assert m, "all = check, fetch, evals 0, evals 1, read"


def _fwd(p):
    return str(p).replace("\\", "/")


def _runner_world(tmp_path, read_s1=None, **extra):
    """Stub inputs for `check` (the files only have to EXIST: the check stage tests presence, the evals are never launched) and an env that points every default at them."""
    sf = tmp_path / "sf"
    (sf / "cache").mkdir(parents=True)
    for rel in ("inputs/native_tracks.pt", "inputs/p10/p10_prior.pt", "inputs/v9/lead_eval139.npz"):
        q = sf / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(b"stub")
    kx0, kx1 = tmp_path / "kx0", tmp_path / "kx1"
    kx0.mkdir()
    kx1.mkdir()
    for nm in ("own", "only-speed"):
        (kx0 / f"eval_M2b-mixo_s0-kindx-{nm}.npz").write_bytes(b"stub")
    if read_s1 is not None:
        (kx1 / "READ_s1.json").write_text(json.dumps(read_s1), encoding="utf-8")
    out = tmp_path / "o"
    out.mkdir()
    env = dict(os.environ, SPEEDAVG_OUT=_fwd(out), SPEEDAVG_SF=_fwd(sf), SPEEDAVG_KINDX_S0=_fwd(kx0), SPEEDAVG_KX1DIR=_fwd(kx1), SPEEDAVG_KINDX_S1=_fwd(kx1),
               SPEEDAVG_READ_S1=_fwd(kx1 / "READ_s1.json"), SPEEDAVG_B2=_fwd(tmp_path / "nobox"), SPEEDAVG_PY=_fwd(sys.executable), SPEEDAVG_THREADS="1")
    env.pop("SPEEDAVG_SKIP_CONDITION", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env, out


def _runner(env, stage):
    return subprocess.run([BASH, str(RUNNER), stage], capture_output=True, text=True, env=env, timeout=120)


@bash
@pytest.mark.parametrize("read_s1,msg", [
    (None, "does not exist: this arm is CONDITIONAL"),                                              # KINDX's seed-1 read is absent
    ({"OUTCOME": "NOT A CARRIER", "carriers": []}, "the PREREG's condition is not met"),            # present, speed not a carrier
    ({"OUTCOME": "CARRIER", "carriers": ["tactical"]}, "the PREREG's condition is not met"),        # a carrier, but not SPEED
    ({"OUTCOME": "NOT A CARRIER", "carriers": ["speed"]}, "the PREREG's condition is not met"),     # speed listed, outcome is not CARRIER
])
def test_runner_refuses_when_kindx_did_not_make_speed_a_carrier(tmp_path, read_s1, msg):
    env, _ = _runner_world(tmp_path, read_s1)
    r = _runner(env, "check")
    assert r.returncode == 3, (r.stdout[-1500:], r.stderr[-800:])
    assert any("ZZSPEEDAVG-STOP" in ln and msg in ln for ln in r.stdout.splitlines()), "the STOP marker line itself carries the reason (a message printed without stopping does not count)"
    assert "CHECK OK" not in r.stdout


@bash
def test_runner_passes_the_condition_only_for_a_speed_carrier_and_the_rehearsal_switch_is_explicit(tmp_path):
    env, _ = _runner_world(tmp_path, {"OUTCOME": "CARRIER", "carriers": ["tactical", "speed"]})
    r = _runner(env, "check")                                                                        # the condition holds; the NEXT requirement (KINDX's s1 evals) is what stops it
    assert "KINDX s1 OUTCOME CARRIER" in r.stdout and "condition is not met" not in r.stdout and "does not exist" not in r.stdout, r.stdout[-1500:]
    assert r.returncode == 3 and "seed-1 evals (own, only-speed) are not in" in r.stdout
    env2, _ = _runner_world(tmp_path / "w2")
    r2 = _runner(dict(env2, SPEEDAVG_SKIP_CONDITION="1"), "check")                                   # an absent read is let through ONLY by the explicit rehearsal switch
    assert "does not exist" not in r2.stdout and "seed-1 evals (own, only-speed) are not in" in r2.stdout, r2.stdout[-1500:]


@bash
def test_runner_fetch_stage_stops_on_a_wrong_sha_and_on_the_same_checkpoint_twice(tmp_path):
    import hashlib
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"checkpoint-a" * 50)
    b.write_bytes(b"checkpoint-b" * 50)
    ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
    env, out = _runner_world(tmp_path / "w", {"OUTCOME": "CARRIER", "carriers": ["speed"]}, SPEEDAVG_X0=_fwd(a), SPEEDAVG_X1=_fwd(b), SPEEDAVG_X0_SHA=ha, SPEEDAVG_X1_SHA=hb)
    ok = _runner(env, "fetch")
    assert ok.returncode == 0 and "fetch OK" in ok.stdout and (out / "ckpt_paths.txt").read_text().splitlines() == [_fwd(a), _fwd(b)], (ok.stdout[-1200:], ok.stderr[-600:])
    assert "pull (read-only)" not in ok.stdout, "checkpoints that are present are not pulled from box 2"
    bad = _runner(dict(env, SPEEDAVG_X1_SHA="0" * 64), "fetch")                                      # the X_s1 on disk is not the one its sha256 names
    assert bad.returncode == 3 and "ZZSPEEDAVG-STOP" in bad.stdout and "sha256 of" in bad.stdout and "expected " + "0" * 64 in bad.stdout, bad.stdout[-1200:]
    c = tmp_path / "c.pt"
    c.write_bytes(a.read_bytes())                                                                    # a second COPY of X_s0 offered as X_s1: no second seed
    same = _runner(dict(env, SPEEDAVG_X1=_fwd(c), SPEEDAVG_X1_SHA=ha), "fetch")
    assert same.returncode == 3 and "the same file: no second seed" in same.stdout, same.stdout[-1200:]
