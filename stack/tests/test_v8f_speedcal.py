"""PREREG_V8F_SPEEDCAL: the instruments (driver, reader, dev-box runner), BUILT AND TESTED. CPU only, a few minutes.

What is pinned, and the deliberate-regression arm of each part:
  (A) the pure pieces       the closed-form masked least squares at known values and against numpy `lstsq`; the mask (a window without a v9 row or without a finite GT for a cell is out); the permuted-target
                            construction (a fixed-seed permutation of the masked targets, slope ~ 0); the map in the standardised cell O reads is the PREREG's c' = a c + b in m/s, and the identity map is the
                            identity BIT FOR BIT.  Arms: a dropped mask, a map in the wrong units, an identity that is not exact.
  (B) the hook              on a tiny model: `identity` reproduces `forward_infer` exactly, `cal` writes the affine map into exactly the five speed cells and leaves every other cell alone, `gt` is SPEEDAVG's own
                            gt hook cell for cell, and the wrapped `forward_diag` is restored (also when the eval raises).
  (C) the fit               on the synthetic world's TRAIN split: the window list is the trainer's, the fit arrays are what the controls re-fit, HBLEND's window list is CHECKED, nothing of the eval enters.
  (D) end to end            the real `v8f_mini_eval.py run` on the synthetic world, two checkpoints, fit + six modes each: controls 1-4 PASS.
                            ARMS (a corrupted input or a broken driver, each must turn ONE control red): identity that is not exact; a fitted map off by 1e-8 from lstsq; a permuted-target map that is not a
                            permutation; fit clips that overlap the eval clips; an applied map that is not the fit json's; a GT arm that is not KINDX's speed oracle; a permuted arm that IS separated-better.
  (E) the committed table   every outcome row, the boundaries (exactly -F), the partition (every combination of the inputs lands in exactly one row, in the table's order), the CEILING routing, SECONDARY, CAPTURE;
                            the table's text equals `Project Steering/PREREG_V8F_SPEEDCAL.md` cell for cell.
  (F) refusals              a limited (smoke) run, a wrong checkpoint sha, a mislabelled mode, a missing speedcal block, a limited fit, a failed control (rc 2, no outcome file).
  (G) the runner            syntax, `--dry-run` (every stage, the markers, the substitute control; writes nothing), load-bearing lines, a STOP banks what exists, the guard's regex matches the driver.
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

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
DRIVER = REPO / "stack" / "scripts" / "v8f_mini_eval_speedcal.py"
READER = CODE / "v8f_speedcal_read.py"
RUNNER = CODE / "devbox_ops" / "run_speedcal.sh"
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_SPEEDCAL.md"


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
os.environ["SPEEDCAL_THREADS"] = "1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_speedcal", Path(__file__).resolve().parent / "test_v8f_mini.py")
world, make_batch, tiny_model = _T.world, _T.make_batch, _T.tiny_model
SC = _load("v8f_mini_eval_speedcal_for_test", DRIVER)
SA = SC.SA
# the lander imports this file from a stack-only tree: the package code dir is absent there, so load it only when present and skip otherwise
RD = _load("v8f_speedcal_read_for_test", READER) if (READER.exists() and (CODE / "v8f_speedavg_read.py").exists()) else None
pytestmark = pytest.mark.skipif(RD is None, reason="the v8F-mini package code dir is absent (a stack-only tree)")
FIVE = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
F = 0.1207


# ============================================================================================== (A) the pure pieces
FORBIDDEN = "pau" + "sed"                       # the word the STRATEGIC family must never carry for v8F (built from halves so this file does not carry it either)

def test_the_cells_are_the_preregs_five_and_the_modes_and_tags():
    assert tuple(SC.SPEED_CELLS) == FIVE == tuple(RD.SPEED_CELLS) == tuple(SA.SPEED_CELLS), "the same five cells as SPEEDAVG, by name and in order"
    assert SC.MODES == RD.MODES == ("own", "identity", "gt", "cal", "perm", "ceiling")
    assert SC.tag_of("cal", 1) == "M2b-mixo_s1-speedcal-cal" and SC.tag_of("own", 0, "b") == "M2b_s0-speedcal-own" and SC.fit_tag(0) == "M2b-mixo_s0-speedcal-fit"
    idx = SA.speed_idx(M)
    assert len(idx) == 5 and idx == sorted(M.CODE_KINDS["speed"])
    assert SC.window_ages(0) == (0, 0) and SC.window_ages(4) == (1, 2) and SC.window_ages(17) == (2, 2 + 5 % 5), "the eval's own age schedule: a_t = i % 3, a_s = a_t + (i // 3) % 5"
    assert SC.PERM_SEED == 20261025 and RD.F_BAR == F and RD.B_BOOT == 2000 and (RD.N_WINDOWS, RD.N_EPISODES) == (2399, 141)


def test_masked_fit_known_values_and_it_is_numpys_lstsq():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 99.0])
    y = 2.0 * x + 1.0
    m = np.array([1, 1, 1, 1, 1, 0], bool)
    y_out = y.copy()
    y_out[5] = -500.0                                                     # the masked-out window is a wild outlier: it must not move the fit
    a, b, n = SC.masked_fit(x, y_out, m)
    assert abs(a - 2.0) < 1e-12 and abs(b - 1.0) < 1e-12 and n == 5
    rng = np.random.default_rng(1)
    x = rng.normal(10, 6, 500)
    y = 0.9 * x + 1.7 + rng.normal(0, 2, 500)
    m = rng.random(500) < 0.8
    a, b, n = SC.masked_fit(x, y, m)
    ar, br = RD.lstsq_map(x, y, m)
    assert abs(a - ar) <= 1e-9 and abs(b - br) <= 1e-9 and n == int(m.sum()), "control 2's reference: the closed form IS numpy lstsq on the masked arrays"
    with pytest.raises(ValueError):
        SC.masked_fit(x, y, np.zeros(500, bool))
    with pytest.raises(ValueError):
        SC.masked_fit(np.full(10, 3.0), np.arange(10.0), np.ones(10, bool))            # no variance in the prediction: the slope is undefined


def test_the_mask_drops_windows_without_a_v9_row_or_a_finite_gt():
    x = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, np.nan], [1.0, 2.0]])
    y = np.array([[1.0, 2.0], [np.nan, 2.0], [1.0, 2.0], [1.0, 2.0]])
    has = np.array([True, True, True, False])
    m = SC.fit_mask(x, y, has)
    assert m.tolist() == [[True, True], [False, True], [True, False], [False, False]], "has & finite GT & finite prediction, per cell"
    rng = np.random.default_rng(0)
    xx = rng.normal(10, 5, (300, 5))
    yy = 0.8 * xx + 2 + rng.normal(0, 1, (300, 5))
    yy[::7, 2] = np.nan
    hh = rng.random(300) < 0.9
    Fm = SC.fit_cells(xx, yy, hh, min_n=10)
    for c in range(5):
        ar, br = RD.lstsq_map(xx[:, c], yy[:, c], hh & np.isfinite(yy[:, c]))
        assert abs(Fm["a"][c] - ar) <= 1e-9 and abs(Fm["b"][c] - br) <= 1e-9 and Fm["n"][c] == int((hh & np.isfinite(yy[:, c])).sum())
        assert Fm["rmse_after"][c] <= Fm["rmse_before"][c] + 1e-12, "the least-squares map cannot raise the in-sample RMSE"
    with pytest.raises(ValueError, match="refused"):
        SC.fit_cells(xx[:5], yy[:5], hh[:5], min_n=10)


def test_permuted_targets_are_a_fixed_seed_permutation_of_the_masked_targets_and_collapse_the_slope():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(10, 6, (n, 5))
    y = x * 1.0 + rng.normal(0, 1, (n, 5))
    has = np.ones(n, bool)
    y[::9, 1] = np.nan
    mask = SC.fit_mask(x, y, has)
    yp = SC.permute_targets(y, mask, SC.PERM_SEED)
    assert np.array_equal(yp, SC.permute_targets(y, mask, SC.PERM_SEED), equal_nan=True), "deterministic for the declared seed"
    assert not np.array_equal(yp, SC.permute_targets(y, mask, SC.PERM_SEED + 1), equal_nan=True)
    for c in range(5):
        assert np.array_equal(np.sort(yp[mask[:, c], c]), np.sort(y[mask[:, c], c])), "a permutation of the masked targets: the marginal is kept"
        assert not np.array_equal(yp[mask[:, c], c], y[mask[:, c], c])
        assert np.array_equal(yp[~mask[:, c], c], y[~mask[:, c], c], equal_nan=True), "unmasked entries untouched"
    Fp = SC.fit_cells(x, yp, has, min_n=10)
    assert max(abs(a) for a in Fp["a"]) < 0.1, "the pairing with T's prediction is destroyed, so the map collapses toward the mean speed (slope ~ 0)"
    assert all(abs(b - y[mask[:, c], c].mean()) < 1.5 for c, b in enumerate(Fp["b"]))


def test_calibration_in_the_standardised_cell_is_the_preregs_affine_map_and_the_identity_is_exact():
    rng = np.random.default_rng(2)
    mu = np.array([11.0, 10.8, 11.3, 9.5, 12.1])
    sd = np.array([7.8, 7.7, 7.7, 8.0, 7.4])
    std = rng.normal(0, 1.2, (64, 5)).astype(np.float32)
    a = np.array([1.011, 1.099, 0.997, 1.081, 1.048])
    b = np.array([-0.955, 0.133, -1.122, -0.864, -0.246])
    out = SC.calibrate_std_np(std, a, b, mu, sd)
    assert out.dtype == np.float32
    phys = SC.std_to_phys(std, mu, sd)
    want = ((a * phys + b) - mu) / sd                                                  # the PREREG: c' = a c + b in m/s, then re-standardised
    assert np.abs(out - want).max() < 2e-6
    assert np.array_equal(RD.calibrate_std_np(std, a, b, mu, sd), out), "the reader restates the formula independently and agrees bit for bit"
    ident = SC.calibrate_std_np(std, np.ones(5), np.zeros(5), mu, sd)
    assert np.array_equal(ident, std), "a = 1, b = 0 is the identity BIT FOR BIT (control 1 rests on it)"
    assert not np.array_equal(SC.calibrate_std_np(std, np.ones(5), np.full(5, 1e-3), mu, sd), std)


def test_the_eval_argv_the_tag_and_the_guard_regex():
    paths = {"v9_eval": "v9", "lead_eval": "ld", "speed_eval": "sp", "p10": "p", "eval_dir": "ed", "cache": "c", "tracks": "t", "eval_grid": None, "eval_kin": None, "eval_join": None}
    own, orc = SA.e_argv("own", "CK", "T", paths), SA.e_argv("gt", "CK", "T", paths)
    assert SC.e_argv("own", "CK", "T", paths) == own and "--tac" in own and own[own.index("--tac") + 1] == "pred"
    for m in ("identity", "gt", "cal", "perm", "ceiling"):
        assert SC.e_argv(m, "CK", "T", paths) == orc, "every hooked mode rides KINDX's `--tac oracle --tac-groups speed --tac-alpha 1` carrier"
    assert orc[-6:] == ["--tac", "oracle", "--tac-groups", "speed", "--tac-alpha", "1"]
    ns = type("A", (), {"n_boot": 2000, "batch": 64, "lru": 32, "device": "cpu"})()
    assert SC.extra_of(ns)[-1] == "--families" and "--families" not in SC.extra_of(type("A", (), {"n_boot": 1, "batch": 1, "lru": 1, "device": "cpu", "no_families": True})())
    with pytest.raises(SystemExit):
        SC.e_argv("avg", "CK", "T", paths)
    regex = re.search(r"-match '([^']*v8f_mini_eval[^']*)'", GUARD.read_text(encoding="utf-8")).group(1)
    assert re.search(regex, "python.exe -I stack/scripts/v8f_mini_eval_speedcal.py evals --seed 0"), "the thermal / time guard that watches v8f_mini_eval must watch this driver"


# ============================================================================================== (B) the hook on a tiny model
def _sens(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(model.head_o.parameters()) + list(model.e_code.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.3)
    return model.eval()


def test_hook_modes_on_a_tiny_model(world):
    b, J, _ = make_batch(world)
    torch.manual_seed(11)
    m0 = _sens(tiny_model(J)[0], 1)
    idx = SA.speed_idx(M)
    ci = [i - (M.N_GOAL + M.N_LAT + M.N_LON) for i in idx]
    non = [i for i in range(M.CODE_DIM) if i not in idx]
    orig = M.V8FMini.forward_diag
    with torch.no_grad():
        own = m0.forward_infer(b)
    a = np.array([1.2, 0.8, 1.05, 0.9, 1.1])
    bb = np.array([0.5, -0.4, 0.2, 0.0, -0.3])

    def run(mode, mp=None, override=None):
        h = SC.SpeedCalHook(M, mode, idx, *(mp or (None, None))).install()
        try:
            with torch.no_grad():
                o = m0.forward_diag(b, own["code"] if override is None else override)
        finally:
            h.restore()
        assert M.V8FMini.forward_diag is orig, "restore() puts the original forward_diag back (checked after EVERY mode)"
        return o, h
    o_id, h_id = run("identity", (np.ones(5), np.zeros(5)))
    assert torch.equal(o_id["traj30"], own["traj30"]) and torch.equal(o_id["code"], own["code"]), "CONTROL 1 at the model level: the identity map changes nothing, bit for bit"
    o_cal, h_cal = run("cal", (a, bb))
    mu, sd = m0.cons_mu[ci].double().numpy(), m0.cons_sd[ci].double().numpy()
    expect = own["code"].clone()
    expect[:, idx] = torch.from_numpy(SC.calibrate_std_np(own["code"][:, idx].numpy(), a, bb, mu, sd))
    assert torch.equal(o_cal["code"], expect), "cal writes the affine map of T's OWN cells into exactly the five speed cells"
    assert torch.equal(o_cal["code"][:, non], own["code"][:, non]), "every other cell is X_s's own"
    assert (o_cal["traj30"] - own["traj30"]).abs().max() > 1e-4, "O is code-sensitive here, so the map moves its plan"
    assert h_cal.stats()["nonspeed_max_abs_dev"] == 0.0 and h_cal.stats()["oracle_code_max_abs_dev"] is None
    cells = h_cal.cells()
    assert np.array_equal(cells["own_std"], own["code"][:, idx].numpy()) and np.array_equal(cells["used_std"], expect[:, idx].numpy()) and np.isnan(cells["partner_std"]).all()
    # gt: the eval's own oracle code for `--tac-groups speed`, and SPEEDAVG's own gt hook, cell for cell
    has = b["lab_t"]["has"].bool()
    code = torch.where(has[:, None], m0._code_gt(b["lab_t"]).to(own["code"].dtype), own["code"])
    sel = torch.zeros(M.CODE_DIM, dtype=torch.bool)
    sel[idx] = True
    oracle = torch.where(sel[None, :], code, own["code"])
    o_gt, h_gt = run("gt", None, oracle)
    assert torch.equal(o_gt["code"], oracle) and h_gt.stats()["oracle_code_max_abs_dev"] == 0.0 and h_gt.stats()["nonspeed_max_abs_dev"] == 0.0
    hs = SA.SpeedAvgHook(M, "gt", None, idx).install()
    try:
        with torch.no_grad():
            m0.forward_diag(b, oracle)
    finally:
        hs.restore()
    assert np.array_equal(hs.cells()["used_std"], h_gt.cells()["used_std"]) and np.array_equal(hs.cells()["gt_phys"], h_gt.cells()["gt_phys"], equal_nan=True), "the gt arm IS SPEEDAVG's gt hook"
    with pytest.raises(ValueError):
        SC.SpeedCalHook(M, "cal", idx, None, None)
    assert M.V8FMini.forward_diag is orig


# ============================================================================================== (C) the synthetic world, the fit and the evals
@pytest.fixture(scope="module")
def sworld(world):
    """The synthetic world with the five speed cells filled in the TRAIN and the EVAL joins (the stock synthetic join carries only v_hi_ms)."""
    root = Path(world["root"])
    rng = np.random.default_rng(5)
    for split in ("train", "eval"):
        J = D.load_join(root / f"join_{split}.npz")
        cons = J["cons"].copy()
        n = len(cons)
        base = 8 + 3 * np.sin(np.arange(n) / 9.0)
        for q, nm in enumerate(FIVE):
            col = base * (0.8 + 0.1 * q) + rng.normal(0, 0.5, n)
            col[rng.random(n) < 0.08] = np.nan
            cons[:, D.CONS_IDX[nm]] = col
        J["cons"] = cons
        D.save_join(J, root / f"join_{split}_speed.npz")
    import hashlib
    clips = list(world["names"]["train"])
    (root / "fit_clips.json").write_text(json.dumps({"clips": clips, "sha256_newline_joined": hashlib.sha256("\n".join(clips).encode()).hexdigest()}), encoding="utf-8")
    return {"root": root}


def _ckpt(tmp, seed):
    """A tiny trained checkpoint (the synthetic trainer), its O made code-sensitive so a changed speed cell moves the plan."""
    TR = _load(f"v8f_mini_train_for_speedcal_{seed}", REPO / "stack" / "scripts" / "v8f_mini_train.py")
    out = tmp / f"run{seed}"
    assert TR.main(["--out", str(out), "--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "4", "--log-every", "2", "--warmup", "1", "--ckpt-every", "4", "--threads", "1",
                    "--seed", str(seed)]) == 0
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    g = torch.Generator().manual_seed(1)
    for k, v in ck["model"].items():
        if k.startswith("head_o.") or k.startswith("e_code."):
            ck["model"][k] = v + torch.randn(v.shape, generator=g).to(v.device) * 0.3
    p = tmp / f"ck_s{seed}.pt"
    torch.save(ck, p)
    return p


def _eval_args(sw):
    root = sw["root"]
    return ["--eval-cache", str(root), "--tracks", str(root / "native_tracks.pt"), "--v9-eval", "x", "--lead-eval", "x", "--speed-eval", "x", "--p10", "x",
            "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval_speed.npz"), "--n-boot", "20", "--batch", "16", "--lru", "4"]


def _fit(sw, ck, sha, seed, outdir, *extra):
    root = sw["root"]
    return SC.main(["fit", "--seed", str(seed), "--x", str(ck), "--expect-sha", sha, "--fit-clips", str(root / "fit_clips.json"), "--fit-cache", str(root),
                    "--fit-join", str(root / "join_train_speed.npz"), "--tracks", str(root / "native_tracks.pt"), "--out-dir", str(outdir), "--batch", "16", "--lru", "4", "--min-fit-n", "5", *extra])


def _evals(sw, ck, sha, seed, evdir, fitdir, modes=None, *extra):
    argv = ["evals", "--seed", str(seed), "--x", str(ck), "--expect-sha", sha, "--fit-json", str(Path(fitdir) / f"fit_M2b-mixo_s{seed}-speedcal-fit.json"), "--eval-dir", str(evdir),
            "--min-fit-n", "5", *_eval_args(sw), *extra]
    if modes:
        argv += ["--modes", modes]
    return SC.main(argv)


@pytest.fixture(scope="module")
def e2e(world, sworld, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sc")
    EV = _load("v8f_mini_eval_for_speedcal_ref", REPO / "stack" / "scripts" / "v8f_mini_eval.py")
    root = sworld["root"]
    ck = {s: _ckpt(tmp, 3 + s) for s in (0, 1)}
    sha = {s: SA.sha256_file(str(ck[s])) for s in (0, 1)}
    assert sha[0] != sha[1]
    kx = {}
    for s in (0, 1):                                                                      # KINDX's own and speed-only-oracle evals of X_s: the references of control 3 and the own guard
        kx[s] = tmp / f"kx{s}"
        for nm, extra in (("own", ["--tac", "pred"]), ("only-speed", ["--tac", "oracle", "--tac-groups", "speed", "--tac-alpha", "1"])):
            assert EV.main(["run", "--ckpt", str(ck[s]), "--tag", f"M2b-mixo_s{s}-kindx-{nm}", "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
                            "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval_speed.npz"), "--out-dir", str(kx[s]), "--tracks", str(root / "native_tracks.pt"),
                            "--n-boot", "20", "--batch", "16", "--lru", "4", *extra]) == 0
    fitdir, ev = tmp / "fit", tmp / "ev"
    for s in (0, 1):
        assert _fit(sworld, ck[s], sha[s], s, fitdir) == 0
        assert _evals(sworld, ck[s], sha[s], s, ev, fitdir) == 0
    A, Fit, K, eid = RD.load_all(str(ev), str(fitdir), {0: str(kx[0]), 1: str(kx[1])}, sha[0], sha[1], allow_smoke=True)
    return {"tmp": tmp, "ck": ck, "sha": sha, "kx": kx, "ev": ev, "fit": fitdir, "A": A, "Fit": Fit, "K": K, "eid": eid, "pf": RD.pf_default(), "world": world, "sw": sworld,
            "eval_clips": list(world["names"]["eval"])}


def test_fit_arrays_are_the_trainers_windows_and_what_the_controls_refit(e2e, world):
    for s in (0, 1):
        fj, z = e2e["Fit"][s]["json"], e2e["Fit"][s]["cells"]
        n = int(z["has"].size)
        assert fj["schema"] == "v8f-speedcal-fit-v1" and fj["seed"] == s and fj["x_sha256"] == e2e["sha"][s] and fj["fit_set"]["n_windows"] == n and fj["limit"] is None
        assert z["own_std"].shape == (n, 5) and z["gt_phys"].shape == (n, 5) and z["perm_gt_phys"].shape == (n, 5)
        assert fj["fit_set"]["clips"] == list(world["names"]["train"]) and not (set(fj["fit_set"]["clips"]) & set(world["names"]["eval"])), "the fit clips are TRAIN clips; no eval clip enters"
        assert [fj["speed_cells"], fj["speed_idx"]] == [list(FIVE), SA.speed_idx(M)]
        # the trainer's own window list: every clip's windows t in [hist-1, T-31], at the eval's age schedule
        ci = np.asarray(z["ci"])
        assert set(ci.tolist()) == set(range(len(world["names"]["train"])))
        for i in range(0, n, 17):
            at, as_ = SC.window_ages(i)
            room = max(int(z["t"][i]) - 3, 0)
            at_c = min(at, room, 2)
            assert int(z["a_t"][i]) == at_c and int(z["a_s"][i]) == max(at_c, min(as_, room, 6))
        # control 2's identity in miniature: the json map IS lstsq on the saved arrays
        mask = np.asarray(z["mask"], bool)
        assert fj["perm"]["seed"] == SC.PERM_SEED == 20261025 and np.array_equal(
            np.asarray(z["perm_gt_phys"]), SC.permute_targets(np.asarray(z["gt_phys"], np.float64), mask, 20261025).astype(np.float32), equal_nan=True), \
            "control 4's permutation is the COMMITTED fixed-seed one, as the fit actually used it"
        x = SC.std_to_phys(z["own_std"], z["cons_mu"], z["cons_sd"])
        for c in range(5):
            ar, br = RD.lstsq_map(x[:, c], np.asarray(z["gt_phys"], np.float64)[:, c], mask[:, c])
            assert abs(fj["map"]["a"][c] - ar) <= 1e-9 and abs(fj["map"]["b"][c] - br) <= 1e-9 and fj["map"]["n"][c] == int(mask[:, c].sum())
        assert (np.asarray(z["has"]) & np.isfinite(np.asarray(z["gt_phys"])).all(1)).sum() > 0 and (~mask).any(), "the synthetic world has windows with holes, so the mask is exercised"


def test_fit_checks_hblends_window_list_and_refuses_a_different_one(e2e, sworld, tmp_path):
    s = 0
    root = sworld["root"]
    z = e2e["Fit"][s]["cells"]
    hb = tmp_path / "hb.npz"
    np.savez(hb, ci=z["ci"], t=z["t"], a_t=z["a_t"], a_s=z["a_s"])
    out = tmp_path / "fit_ok"
    assert _fit(sworld, e2e["ck"][s], e2e["sha"][s], s, out, "--hblend-windows", str(hb)) == 0
    fj = json.loads((out / "fit_M2b-mixo_s0-speedcal-fit.json").read_text(encoding="utf-8"))
    assert fj["fit_set"]["hblend_windows"]["equal"] is True and fj["fit_set"]["hblend_windows"]["n_compared"] == int(z["has"].size)
    assert fj["map"] == e2e["Fit"][s]["json"]["map"], "the fit is deterministic"
    bad = {k: np.asarray(z[k]).copy() for k in ("ci", "t", "a_t", "a_s")}
    bad["t"][3] += 1
    np.savez(hb, **bad)
    with pytest.raises(SystemExit, match="differ from HBLEND"):
        _fit(sworld, e2e["ck"][s], e2e["sha"][s], s, tmp_path / "fit_bad", "--hblend-windows", str(hb))
    with pytest.raises(SystemExit, match="REFUSED"):
        _fit(sworld, e2e["ck"][s], "0" * 64, s, tmp_path / "fit_sha")
    cj = tmp_path / "clips_bad.json"
    cj.write_text(json.dumps({"clips": list(sworld_clips(root)), "sha256_newline_joined": "00" * 32}), encoding="utf-8")
    with pytest.raises(SystemExit, match="hashes to"):
        SC.main(["fit", "--seed", "0", "--x", str(e2e["ck"][s]), "--expect-sha", e2e["sha"][s], "--fit-clips", str(cj), "--fit-cache", str(root), "--fit-join", str(root / "join_train_speed.npz"),
                 "--tracks", str(root / "native_tracks.pt"), "--out-dir", str(tmp_path / "fit_cj"), "--min-fit-n", "5"])


def sworld_clips(root):
    return json.loads((Path(root) / "fit_clips.json").read_text(encoding="utf-8"))["clips"]


def test_evals_are_resumable_idempotent_and_refuse_a_wrong_sha_or_a_wrong_order(e2e, sworld, tmp_path):
    s = 0
    ev = tmp_path / "ev"
    assert _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, ev, e2e["fit"], "own,identity") == 0
    t0 = {p.name: p.stat().st_mtime_ns for p in ev.glob("*.npz")}
    assert _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, ev, e2e["fit"], "own,identity") == 0
    assert {p.name: p.stat().st_mtime_ns for p in ev.glob("*.npz")} == t0, "a finished mode is skipped"
    with pytest.raises(SystemExit, match="REFUSED"):
        _evals(sworld, e2e["ck"][s], "0" * 64, s, tmp_path / "ev2", e2e["fit"], "own")
    with pytest.raises(SystemExit, match="order"):
        _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, tmp_path / "ev3", e2e["fit"], "cal,own")
    with pytest.raises(SystemExit, match="identity first"):
        _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, tmp_path / "ev4", e2e["fit"], "own,ceiling")
    with pytest.raises(SystemExit, match="not available"):
        _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, tmp_path / "ev5", e2e["fit"], "own,perm", "--target", "b")
    # a fit of ANOTHER seed / another checkpoint is refused
    with pytest.raises(SystemExit, match="REFUSED"):
        SC.main(["evals", "--seed", "1", "--x", str(e2e["ck"][1]), "--expect-sha", e2e["sha"][1], "--fit-json", str(e2e["fit"] / "fit_M2b-mixo_s0-speedcal-fit.json"), "--eval-dir", str(tmp_path / "ev6"),
                 "--modes", "own,cal", *_eval_args(sworld)])
    with pytest.raises(SystemExit, match="--fit-json is required"):
        SC.main(["evals", "--seed", "0", "--x", str(e2e["ck"][0]), "--expect-sha", e2e["sha"][0], "--eval-dir", str(tmp_path / "ev7"), "--modes", "own,cal", *_eval_args(sworld)])


def test_the_evals_refuse_a_limited_fit_and_a_fit_on_other_cells(e2e, sworld, tmp_path):
    src = json.loads((e2e["fit"] / "fit_M2b-mixo_s0-speedcal-fit.json").read_text(encoding="utf-8"))
    for key, val, msg in (("limit", 100, "LIMITED"), ("speed_idx", [1, 2, 3, 4, 5], "fitted on cells"), ("x_sha256", "0" * 64, "the fit was made on checkpoint")):
        fd = tmp_path / f"fit_{key}"
        fd.mkdir()
        (fd / "fit_M2b-mixo_s0-speedcal-fit.json").write_text(json.dumps({**src, key: val}), encoding="utf-8")
        with pytest.raises(SystemExit, match=msg):
            _evals(sworld, e2e["ck"][0], e2e["sha"][0], 0, tmp_path / f"ev_{key}", fd, "own,cal")


def test_the_driver_undoes_its_hook_and_its_limit_when_the_eval_raises(e2e, sworld, tmp_path, monkeypatch):
    orig = M.V8FMini.forward_diag
    EVM = SA._load_modules()[1]
    orig_w, orig_c = EVM.D.eval_grid_windows, EVM.ctx_from_args

    def boom(argv):
        raise RuntimeError("boom")
    monkeypatch.setattr(EVM, "main", boom)
    monkeypatch.setattr(SA, "_load_modules", lambda: (M, EVM))
    with pytest.raises(RuntimeError):
        _evals(sworld, e2e["ck"][0], e2e["sha"][0], 0, tmp_path / "evx", e2e["fit"], "identity", "--limit", "5")
    assert M.V8FMini.forward_diag is orig and EVM.D.eval_grid_windows is orig_w and EVM.ctx_from_args is orig_c


# ============================================================================================== (D) end to end: the controls and their arms
def test_e2e_controls_pass_and_the_arm_is_live(e2e):
    A, Fit, K, eid, pf = e2e["A"], e2e["Fit"], e2e["K"], e2e["eid"], e2e["pf"]
    ctl = RD.controls(A, Fit, K, eid, pf, n_boot=30, eval_clips=e2e["eval_clips"])
    assert ctl["PASS"], json.dumps(ctl, indent=1, default=str)[:14000]
    for s in (0, 1):
        r = ctl["seeds"][f"s{s}"]
        c1 = r["control_1_identity_map_is_identity"]
        assert c1["ade6_max_abs_dev_m"] == 0.0 and c1["pred_equal"] and c1["map_is_identity"] and c1["cells_unchanged"]
        assert r["control_2_fitted_map_is_lstsq"]["max_abs_dev"] <= 1e-9 and r["control_2_fitted_map_is_lstsq"]["fit_clips_disjoint_from_eval_clips"]
        assert r["control_3_gt_equals_kindx_speed_oracle"]["ade6_max_abs_dev_m"] <= 1e-4 and r["control_3_gt_equals_kindx_speed_oracle"]["driver_code_max_abs_dev"] == 0.0
        assert r["control_4_permuted_map_is_not_separated_better"]["PASS"]
        assert all(v for k, v in r["guards"].items() if isinstance(v, bool)) and all(r["guards"]["used_cells_are_the_map_applied_to_own_cells"].values())
        assert np.abs(A[s]["cal"]["pred"] - A[s]["own"]["pred"]).max() > 0, "the calibration moves O's plan on some window"
        assert A[s]["cal"]["json"]["tac"] == "speedcal-cal" and A[s]["cal"]["json"]["tac_manipulation"]["mode"] == "speedcal-cal", "the carrier's `oracle` label is relabelled"
        assert A[s]["cal"]["json"]["raw"].get("families"), "the evals carry `families` (the paired-families tool's reproduction control reads them)"
        assert A[s]["cal"]["block"]["map"]["a"] == Fit[s]["json"]["map"]["a"]


def _ctl(e2e, mutate=None):
    A, Fit, K = copy.deepcopy(e2e["A"]), copy.deepcopy(e2e["Fit"]), copy.deepcopy(e2e["K"])
    if mutate:
        mutate(A, Fit, K)
    return RD.controls(A, Fit, K, e2e["eid"], e2e["pf"], n_boot=30, eval_clips=e2e["eval_clips"])


CTL = ("control_1_identity_map_is_identity", "control_2_fitted_map_is_lstsq", "control_3_gt_equals_kindx_speed_oracle", "control_4_permuted_map_is_not_separated_better")


def _only_red(ctl, s, which):
    """`which` is the one control (a key of the seed's block) that must be red; every other control of that seed stays green."""
    r = ctl["seeds"][f"s{s}"]
    assert not ctl["PASS"] and not r["PASS"]
    for k in CTL:
        assert r[k]["PASS"] is (k != which), (k, r[k])


def test_arm_control_1_an_identity_that_is_not_the_identity(e2e):
    def mut(A, Fit, K):
        A[0]["identity"]["ade6"][0] += 1e-6                                       # control 1 is EXACT (0.0): this is 100x below the bar of control 3
    _only_red(_ctl(e2e, mut), 0, CTL[0])

    def mut2(A, Fit, K):
        A[1]["identity"]["pred"] = A[1]["identity"]["pred"] + 1e-6                  # an ADE-invisible change to the plan is still not the identity
    _only_red(_ctl(e2e, mut2), 1, CTL[0])

    def mut3(A, Fit, K):
        A[0]["identity"]["block"]["map"]["a"][2] = 1.0000001                         # the map the eval applied was not a = 1
    c = _ctl(e2e, mut3)
    assert not c["seeds"]["s0"][CTL[0]]["PASS"] and not c["seeds"]["s0"][CTL[0]]["map_is_identity"]

    def mut4(A, Fit, K):
        A[1]["identity"]["cells"]["used_std"] = A[1]["identity"]["cells"]["used_std"] + np.float32(1e-6)
    c4 = _ctl(e2e, mut4)
    assert not c4["seeds"]["s1"][CTL[0]]["cells_unchanged"] and not c4["PASS"]


def test_arm_control_2_a_fitted_map_that_is_not_lstsq(e2e):
    def mut(A, Fit, K):
        Fit[0]["json"]["map"]["a"][0] += 1e-8                                       # 1e-8 off numpy lstsq: above the 1e-9 bar
    _only_red(_ctl(e2e, mut), 0, CTL[1])

    def mut_b(A, Fit, K):
        Fit[1]["json"]["map"]["b"][3] += 1e-8
    _only_red(_ctl(e2e, mut_b), 1, CTL[1])

    def mut2(A, Fit, K):
        Fit[0]["json"]["perm"]["a"][1] += 1e-8                                      # the permuted-target map is held to lstsq too
    _only_red(_ctl(e2e, mut2), 0, CTL[1])

    def mut3(A, Fit, K):
        Fit[0]["cells"]["gt_phys"] = Fit[0]["cells"]["gt_phys"].copy()
        Fit[0]["cells"]["gt_phys"][np.flatnonzero(Fit[0]["cells"]["mask"][:, 0])[0], 0] += 5.0     # the training arrays moved after the fit
    _only_red(_ctl(e2e, mut3), 0, CTL[1])

    def mut4(A, Fit, K):
        Fit[0]["json"]["map"]["n"][0] += 1
    _only_red(_ctl(e2e, mut4), 0, CTL[1])

    def mut5(A, Fit, K):
        Fit[1]["cells"]["perm_gt_phys"] = Fit[1]["cells"]["gt_phys"].copy()         # the "permuted" targets are the true ones: not a permutation
    c = _ctl(e2e, mut5)
    assert not c["seeds"]["s1"][CTL[1]]["PASS"] and not c["PASS"]
    assert not all(v["perm_is_a_nonidentity_permutation_of_the_targets"] for v in c["seeds"]["s1"][CTL[1]]["cells"].values())

    def mut6(A, Fit, K):
        Fit[0]["json"]["fit_set"]["clips"] = list(Fit[0]["json"]["fit_set"]["clips"]) + [e2e["eval_clips"][0]]      # an EVAL clip in the fit set
    c6 = _ctl(e2e, mut6)
    assert c6["seeds"]["s0"][CTL[1]]["fit_clips_disjoint_from_eval_clips"] is False and not c6["PASS"], "no eval window may enter the fit"

    def mut7(A, Fit, K):
        A[0]["cal"]["block"]["map"]["a"][1] += 1e-6                                 # the map the cal eval APPLIED is not the fit json's
    c7 = _ctl(e2e, mut7)
    assert c7["seeds"]["s0"][CTL[1]]["cal_eval_applied_the_fit_json_map"] is False and not c7["PASS"]

    def mut8(A, Fit, K):
        A[1]["ceiling"]["block"]["map"]["b"][2] += 1e-6                             # the CEILING map is not lstsq on the eval cells
    c8 = _ctl(e2e, mut8)
    assert c8["seeds"]["s1"][CTL[1]]["ceiling_map_vs_lstsq_on_eval_cells_max_abs_dev"] > 1e-9 and not c8["PASS"]

    def mut9(A, Fit, K):
        Fit[0]["json"]["fit_set"]["clips"] = []
    assert _ctl(e2e, mut9)["seeds"]["s0"][CTL[1]]["fit_clips_disjoint_from_eval_clips"] is False, "an empty clip list proves nothing"


def test_arm_control_3_a_gt_arm_that_is_not_kindxs_speed_oracle(e2e):
    def mut(A, Fit, K):
        A[0]["gt"]["ade6"][1] += 2e-4
    _only_red(_ctl(e2e, mut), 0, CTL[2])

    def mut2(A, Fit, K):
        A[1]["gt"]["block"]["hook"]["oracle_code_max_abs_dev"] = 1e-3
    _only_red(_ctl(e2e, mut2), 1, CTL[2])

    def mut3(A, Fit, K):
        A[0]["gt"]["ade6"] = A[0]["own"]["ade6"].copy()
    assert not _ctl(e2e, mut3)["seeds"]["s0"][CTL[2]]["PASS"], "the oracle must differ from own on this world, or control 3 is blind here"


def test_arm_control_4_a_permuted_map_that_is_separated_better_and_one_that_is_merely_worse(e2e):
    def mut(A, Fit, K):
        A[0]["perm"]["ade6"] = A[0]["own"]["ade6"] - 1.0                              # the deliberate regression IMPROVES on X_s: the instrument is broken
    _only_red(_ctl(e2e, mut), 0, CTL[3])

    def mut2(A, Fit, K):
        A[1]["perm"]["ade6"] = A[1]["own"]["ade6"] + 1.0                              # separated WORSE is what a collapsed map should do: green
    c = _ctl(e2e, mut2)
    assert c["seeds"]["s1"][CTL[3]]["PASS"] and c["seeds"]["s1"][CTL[3]]["separated"] is True and c["PASS"]

    def mut3(A, Fit, K):
        A[0]["perm"]["ade6"] = A[0]["own"]["ade6"].copy()                             # identical to X_s: delta 0, not separated, not better: green
    assert _ctl(e2e, mut3)["seeds"]["s0"][CTL[3]]["PASS"]

    def mut4(A, Fit, K):
        d = np.where(np.arange(A[1]["own"]["ade6"].size) % 2 == 0, 0.5, -0.5) - 1e-3        # a hair BETTER on average, with a spread that keeps the CI around 0
        A[1]["perm"]["ade6"] = A[1]["own"]["ade6"] + d
    c4 = _ctl(e2e, mut4)
    assert c4["seeds"]["s1"][CTL[3]]["paired_delta"][0] < 0 and c4["seeds"]["s1"][CTL[3]]["separated"] is False and c4["seeds"]["s1"][CTL[3]]["PASS"], "better but NOT separated is not 'separated-better'"


@pytest.mark.parametrize("guard,mut,seed", [
    ("own_reproduces_the_references", lambda A, Fit, K: K[0]["own"]["ade6"].__iadd__(2e-4), "s0"),
    ("hook_leaves_nonspeed_cells_alone", lambda A, Fit, K: A[0]["cal"]["block"]["hook"].__setitem__("nonspeed_max_abs_dev", 1e-3), "s0"),
    ("tactical_and_strategic_equal_to_X_s", lambda A, Fit, K: A[0]["cal"]["json"]["raw"]["strategic"].__setitem__("n_valid", [0, 0, 0]), "s0"),
    ("every_mode_saw_the_same_own_cells", lambda A, Fit, K: A[1]["cal"]["cells"]["own_std"].__iadd__(1e-3), "s1"),
    ("gt_falls_back_to_own_where_no_v9_row", lambda A, Fit, K: A[1]["gt"]["cells"]["has"].__setitem__(slice(0, 3), False), "s1"),
    ("cal_is_live_it_moves_the_cells_and_the_plan", lambda A, Fit, K: A[1]["cal"].__setitem__("pred", A[1]["own"]["pred"].copy()), "s1"),
])
def test_arm_each_guard_goes_red_on_its_own_corruption(e2e, guard, mut, seed):
    c = _ctl(e2e, mut)
    r = c["seeds"][seed]
    assert r["guards"][guard] is False and not r["guards_PASS"] and not c["PASS"], guard


def test_arm_the_calibrated_cells_must_be_the_map_applied_to_the_own_cells(e2e):
    def mut(A, Fit, K):
        A[0]["cal"]["cells"]["used_std"] = A[0]["cal"]["cells"]["used_std"] + np.float32(1e-4)
    c = _ctl(e2e, mut)
    assert c["seeds"]["s0"]["guards"]["used_cells_are_the_map_applied_to_own_cells"]["cal"] is False and not c["seeds"]["s0"]["guards_PASS"] and not c["PASS"]

    def mut2(A, Fit, K):
        A[1]["perm"]["cells"]["used_std"] = A[1]["perm"]["cells"]["own_std"].copy()
    assert not _ctl(e2e, mut2)["PASS"]


def test_arm_the_real_driver_with_a_wrong_formula_or_an_inexact_identity_is_caught(e2e, sworld, tmp_path, monkeypatch):
    """Two broken DRIVERS (not corrupted outputs): the map written in the wrong units, and an identity that adds float noise. The reader's controls must go red on the evals they produce."""
    s = 0
    fit = e2e["fit"]

    def wrong_units(std, a, b, mu, sd):                                               # b added in standardised units: forgets mu and sd
        return (np.asarray(a) * np.asarray(std, np.float64) + np.asarray(b)).astype(np.asarray(std).dtype)
    monkeypatch.setattr(SC, "calibrate_std_np", wrong_units)
    d = tmp_path / "mut_units"
    _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, d, fit, "own,identity,cal")
    monkeypatch.undo()
    E = RD.load_eval(str(d), "M2b-mixo_s0-speedcal-cal", need_cells=True)
    E["block"] = RD.check_arm(E, 0, "cal", e2e["sha"][0], True)

    def mut(A, Fit, K):
        A[0]["cal"] = E
    c = _ctl(e2e, mut)
    assert c["seeds"]["s0"]["guards"]["used_cells_are_the_map_applied_to_own_cells"]["cal"] is False and not c["PASS"], "a map in the wrong units is not the PREREG's c' = a c + b"

    orig_cal = SC.calibrate_std_np

    def noisy(std, a, b, mu, sd):
        out = orig_cal(std, a, b, mu, sd)
        return (out + np.float32(1e-6)).astype(out.dtype)
    monkeypatch.setattr(SC, "calibrate_std_np", noisy)
    d2 = tmp_path / "mut_noisy"
    _evals(sworld, e2e["ck"][s], e2e["sha"][s], s, d2, fit, "own,identity")
    monkeypatch.undo()
    Ei = RD.load_eval(str(d2), "M2b-mixo_s0-speedcal-identity", need_cells=True)
    Ei["block"] = RD.check_arm(Ei, 0, "identity", e2e["sha"][0], True)

    def mut2(A, Fit, K):
        A[0]["identity"] = Ei
    c2 = _ctl(e2e, mut2)
    assert not c2["seeds"]["s0"][CTL[0]]["PASS"] and not c2["seeds"]["s0"][CTL[0]]["cells_unchanged"], "control 1 catches an identity that is not exact"


# ============================================================================================== (E) the committed table
def _g(point, sep=True):
    """A paired block: separated -> the CI sits on one side of 0; not separated -> it straddles 0."""
    if sep:
        lo, hi = (point - 0.04, min(point + 0.04, -1e-4)) if point < 0 else (max(point - 0.04, 1e-4), point + 0.04)
    else:
        lo, hi = min(point - 0.05, -0.01), max(point + 0.05, 0.01)
    return {"point": point, "delta": round(point, 4), "lo": lo, "hi": hi, "separated": sep}


def _dec(p0, p1, c0, c1, sec=(True, True)):
    return RD.decide({0: p0, 1: p1}, {0: c0, 1: c1}, {0: sec[0], 1: sec[1]}, F)["outcome"]


NS = (_g(0.01, sep=False), _g(-0.02, sep=False))


@pytest.mark.parametrize("p0,p1,c0,c1,sec,want", [
    (_g(-0.30), _g(-0.20), _g(-0.4), _g(-0.4), (True, True), "HEAD BIAS PAYS"),
    (_g(-F), _g(-F), _g(-0.4), _g(-0.4), (True, True), "HEAD BIAS PAYS"),                       # exactly -F is "<= -F"
    (_g(-0.30), _g(-0.20), _g(0.0), _g(0.0), (True, True), "HEAD BIAS PAYS"),                   # the CEILING does not enter this row
    (_g(-0.1206), _g(-0.30), _g(-0.4), _g(-0.4), (True, True), "PARTIAL BIAS"),                 # one seed a hair above -F
    (_g(-0.05), _g(-0.06), _g(-0.4), _g(-0.4), (True, True), "PARTIAL BIAS"),
    (_g(-0.30), _g(-0.05), _g(-0.4), _g(-0.4), (True, True), "PARTIAL BIAS"),
    (_g(-0.05), _g(-0.06), _g(-0.4), _g(-0.4), (False, False), "PARTIAL BIAS"),                 # the PARTIAL row names no SECONDARY
    (_g(-0.30), _g(-0.20), _g(-0.4), _g(-0.4), (False, True), "MIXED"),                         # PRIMARY met at both, SECONDARY separated-worse at seed 0
    (_g(-0.30), _g(-0.20), _g(-0.4), _g(-0.4), (True, False), "MIXED"),
    (_g(-0.50, sep=False), _g(-0.20), _g(-0.4), _g(-0.4), (True, True), "MIXED"),               # one seed separated, one not
    (_g(-0.50, sep=False), _g(-0.60, sep=False), _g(-0.4), _g(-0.4), (True, True), "MAP DOES NOT TRANSFER"),      # a big point that is not separated is no evidence
    (_g(0.02, sep=False), _g(-0.01, sep=False), _g(-F), _g(-0.5), (True, True), "MAP DOES NOT TRANSFER"),         # CEILING exactly -F at one seed, below at the other
    (_g(0.02, sep=False), _g(-0.01, sep=False), _g(-0.1206), _g(-0.5), (True, True), "INFORMATION-LIMITED"),     # CEILING a hair above -F at EITHER seed
    (_g(0.02, sep=False), _g(-0.01, sep=False), _g(-0.5), _g(-0.05), (True, True), "INFORMATION-LIMITED"),
    (_g(0.02, sep=False), _g(-0.01, sep=False), _g(0.0), _g(0.0), (True, True), "INFORMATION-LIMITED"),
    (_g(0.02, sep=False), _g(-0.01, sep=False), _g(-0.5), _g(-0.5), (False, False), "MAP DOES NOT TRANSFER"),    # the "no" rows name no SECONDARY
    (_g(0.20), _g(0.30), _g(-0.5), _g(-0.5), (True, True), "MIXED"),                            # separated WORSE at both: not "not separated"
    (_g(0.20), _g(0.02, sep=False), _g(-0.5), _g(-0.5), (True, True), "MIXED"),
    (_g(-0.30), _g(0.02, sep=False), _g(-0.5), _g(-0.5), (True, True), "MIXED"),
])
def test_the_outcome_table(p0, p1, c0, c1, sec, want):
    assert _dec(p0, p1, c0, c1, sec) == want


def test_the_ceiling_routing_is_the_only_thing_that_splits_the_two_no_rows_and_it_is_not_inverted():
    both_le = _dec(*NS, _g(-0.3), _g(-0.2))
    one_gt = _dec(*NS, _g(-0.3), _g(-0.02))
    assert (both_le, one_gt) == ("MAP DOES NOT TRANSFER", "INFORMATION-LIMITED")
    d = RD.decide({0: NS[0], 1: NS[1]}, {0: _g(-0.3), 1: _g(-0.2)}, {0: True, 1: True}, F)
    assert d["ceiling_le_minus_F"] == {"s0": True, "s1": True} and d["primary_separated"] == {"s0": False, "s1": False}


def test_the_rows_are_a_partition_in_the_tables_order():
    """Every combination of (PRIMARY state x CEILING <= -F x SECONDARY) per seed lands in EXACTLY one row, and the row `decide` picks is the first whose condition (written here from the PREREG's text,
    independently of `decide`) holds."""
    import itertools
    prim_states = {"met": _g(-0.30), "neg_sep_above_F": _g(-0.05), "ns": _g(0.01, sep=False), "worse": _g(0.2)}
    ceil_states = {"le": _g(-0.3), "gt": _g(-0.01)}
    seen = set()
    for (n0, p0), (n1, p1), (c0n, c0), (c1n, c1), s0, s1 in itertools.product(prim_states.items(), prim_states.items(), ceil_states.items(), ceil_states.items(), (True, False), (True, False)):
        met = [n0 == "met", n1 == "met"]
        negsep = [n0 in ("met", "neg_sep_above_F"), n1 in ("met", "neg_sep_above_F")]
        sep = [n0 != "ns", n1 != "ns"]
        cle = [c0n == "le", c1n == "le"]
        rows = {"HEAD BIAS PAYS": all(met) and s0 and s1,
                "PARTIAL BIAS": all(negsep) and not all(met),
                "MAP DOES NOT TRANSFER": (not any(sep)) and all(cle),
                "INFORMATION-LIMITED": (not any(sep)) and not all(cle)}
        hit = [r for r, ok in rows.items() if ok]
        assert len(hit) <= 1, (n0, n1, c0n, c1n, s0, s1, hit)                          # mutually exclusive
        want = hit[0] if hit else "MIXED"
        assert _dec(p0, p1, c0, c1, (s0, s1)) == want, (n0, n1, c0n, c1n, s0, s1)
        seen.add(want)
    assert seen == set(RD.OUTCOMES), "every row is reachable"


def test_predicates_and_constants():
    assert RD.primary_met(_g(-0.1207)) and not RD.primary_met(_g(-0.12069)) and not RD.primary_met(_g(-0.5, sep=False)) and not RD.primary_met(_g(0.5))
    assert RD.ceiling_le(_g(-0.1207)) and not RD.ceiling_le(_g(-0.12069)) and RD.ceiling_le(_g(-0.5, sep=False)), "CEILING <= -F is the unrounded point; no separation is required"
    assert RD.separated_worse({"lo": 0.01, "hi": 0.2, "delta": 0.1}) and not RD.separated_worse({"lo": -0.01, "hi": 0.2, "delta": 0.1, "separated": False})
    assert not RD.separated_worse({"lo": -0.2, "hi": -0.01, "delta": -0.1, "separated": True}), "separated BETTER is not worse"
    assert RD.F_BAR == 0.1207 and RD.TOL_M == 1e-4 and RD.TOL_LSTSQ == 1e-9


def _fam(**over):
    blk = lambda lo, hi: {"delta": (lo + hi) / 2, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}            # noqa: E731
    ok = (-0.05, 0.05)
    d = {"speed_mae_mps": blk(*ok), "accel_mae_mps2": blk(*ok)}
    d.update(over.get("lon", {}))
    return {"longitudinal": d, "lateral": {"heading_mae_deg": blk(*ok)}}


def _pair(**over):
    blk = lambda lo, hi: {"delta": (lo + hi) / 2, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}            # noqa: E731
    ok = (-0.05, 0.05)
    ae = {k: blk(*ok) for k in RD.RDA.DK_KEYS}
    ae.update(over.get("ae", {}))
    return {"abs_error_vs_human": ae, "lead_viol": {"all_lead_windows": over.get("lv", blk(*ok))}}


def test_secondary_is_speedavgs_and_each_of_the_six_metrics_can_fail_it():
    assert RD.secondary_of is RD.RDA.secondary_of and RD.capture is RD.RDA.capture, "imported, not copied"
    assert RD.secondary_of(_fam(), _pair())["not_separated_worse"]
    bad = {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}
    cases = [("speed MAE", _fam(lon={"speed_mae_mps": bad}), _pair()), ("accel MAE", _fam(lon={"accel_mae_mps2": bad}), _pair()),
             ("headway", _fam(), _pair(ae={"headway_min_m": bad})), ("time gap", _fam(), _pair(ae={"time_gap_min_s": bad})), ("min TTC", _fam(), _pair(ae={"min_ttc_s": bad})),
             ("LEAD-VIOL", _fam(), _pair(lv=bad))]
    for nm, f, p in cases:
        assert not RD.secondary_of(f, p)["not_separated_worse"], nm
    for f, p in ((_fam(lon={"speed_mae_mps": {"status": "UNAVAILABLE"}}), _pair()), (_fam(), None)):
        with pytest.raises(SystemExit) as e:
            RD.secondary_of(f, p)
        assert e.value.code == 2, "a missing family metric is refused, never dropped"


def test_the_tables_text_is_the_preregs_text():
    if not PREREG.exists():
        pytest.skip("Project Steering/PREREG_V8F_SPEEDCAL.md is absent (a stack-only tree)")
    txt = PREREG.read_text(encoding="utf-8")
    for o, (cond, nxt) in RD.TABLE.items():
        assert f"**{o}** | {cond} | {nxt} |" in txt, o
    assert tuple(RD.TABLE) == RD.OUTCOMES == ("HEAD BIAS PAYS", "PARTIAL BIAS", "MAP DOES NOT TRANSFER", "INFORMATION-LIMITED", "MIXED")
    order = [txt.index(f"**{o}** |") for o in RD.OUTCOMES]
    assert order == sorted(order), "the rows are in the PREREG's order"
    assert "F = 0.1207 m" in txt and RD.F_BAR == 0.1207 and RD.B_BOOT == 2000 and (RD.N_WINDOWS, RD.N_EPISODES) == (2399, 141)
    for n in FIVE:
        assert f"`{n}`" in txt
    assert "| **PRIMARY** | own-code ADE6, SPEEDCAL_s − X_s | **≤ −F and separated, at BOTH seeds** |" in txt
    assert "to 1e-9" in txt and "Δ = 0.0" in txt and "≤ 1e-4 m" in txt and "must NOT be separated-better" in txt, "the controls' bars are the PREREG's"
    assert "**No eval window enters the fit.**" in txt and "HBLEND" in txt and "141 training clips" in txt
    assert RD.TOL_LSTSQ == 1e-9 and RD.TOL_M == 1e-4


def test_cell_rmse_known_values_and_maps():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 3, (50, 5))
    y = x + 2.0
    r = RD.cell_rmse(x, y, np.ones(50, bool))
    assert all(abs(v["rmse"] - 2.0) < 1e-12 and v["n"] == 50 for v in r.values())
    y[:10, 0] = np.nan
    assert RD.cell_rmse(x, y, np.ones(50, bool))[FIVE[0]]["n"] == 40
    assert RD.cell_rmse(x, y, np.zeros(50, bool))[FIVE[1]]["rmse"] is None


# ============================================================================================== (F) the result block, refusals, the CLI
def _stub_fams(e2e):
    out = {}
    for s in (0, 1):
        out[s] = (_fam(), {"arms": {f"M2b-mixo_s{s}-speedcal-cal": {"stub": True}}, "pairs": {}}, _pair())
    return out


def test_build_result_block_and_output_text(e2e, tmp_path):
    A, Fit, K, eid, pf = e2e["A"], e2e["Fit"], e2e["K"], e2e["eid"], e2e["pf"]
    ctl = RD.controls(A, Fit, K, eid, pf, n_boot=30, eval_clips=e2e["eval_clips"])
    fams = _stub_fams(e2e)
    res = RD.build_result(A, Fit, K, eid, ctl, pf, fams, n_boot=60)
    assert res["OUTCOME"] in RD.OUTCOMES and res["committed_next"] == RD.TABLE[res["OUTCOME"]][1]
    assert res["tier"] == "T1-class open loop (own held outputs; no recorded future action)" and res["CEILING"]["tier"].startswith("T0")
    assert sum(v["this_read"] for v in res["outcome_table_verbatim"].values()) == 1 and list(res["outcome_table_verbatim"]) == list(RD.OUTCOMES)
    for s in (0, 1):
        assert res["PRIMARY"][f"s{s}"]["point"] == pytest.approx(float(A[s]["cal"]["ade6"].mean() - A[s]["own"]["ade6"].mean()), abs=1e-12)
        assert res["CEILING"][f"s{s}"]["point"] == pytest.approx(float(A[s]["ceiling"]["ade6"].mean() - A[s]["own"]["ade6"].mean()), abs=1e-12)
    assert set(res["CAPTURE"]) == {"s0", "s1"} and res["controls"]["PASS"]
    res["beside"] = RD.beside_block(A, Fit, K, eid, fams, pf, n_boot=60, pf_dir=str(tmp_path / "nopairfam"))
    txt = json.dumps(res, default=str) + RD.md_of(res)
    assert FORBIDDEN not in txt.lower(), "STRATEGIC is in force for v8F: the word never appears"
    assert "T0-open-loop" not in txt, "the eval json's own tier string is a known mislabel and is not copied"
    for s in (0, 1):
        b = res["beside"]["families"][f"s{s}"]
        assert set(b) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}, "all four families"
        assert b["STRATEGIC"]["equal_to_X_s"] is True and "in force for v8F" in b["STRATEGIC"]["status"] and b["TACTICAL"]["heads_equal_to_X_s"] is True
        assert b["TACTICAL"]["paired"]["status"].startswith("NOT RUN"), "an absent paired-families output is reported, never silently dropped"
        pc = res["beside"]["per_cell"][f"s{s}"]
        assert set(pc) == {"SPEEDCAL", "perm_control", "ceiling_T0"} and set(pc["SPEEDCAL"]) == set(FIVE)
        assert all(k in pc["SPEEDCAL"][FIVE[0]] for k in ("a", "b", "train_rmse_before", "train_rmse_after", "eval_rmse_before", "eval_rmse_after"))
        assert pc["SPEEDCAL"][FIVE[0]]["a"] == Fit[s]["json"]["map"]["a"][0]
    assert res["beside"]["baseline_B"]["status"].startswith("NOT RUN")
    p = tmp_path / "R.json"
    RD._dump(str(p), res)
    assert json.loads(p.read_text(encoding="utf-8"))["OUTCOME"] == res["OUTCOME"]
    for f in (READER, DRIVER, RUNNER):
        assert FORBIDDEN not in f.read_text(encoding="utf-8").lower(), f.name


def test_the_result_block_wires_secondary_capture_and_the_ceiling(e2e):
    A, Fit, K, eid, pf = e2e["A"], e2e["Fit"], e2e["K"], e2e["eid"], e2e["pf"]
    ctl = RD.controls(A, Fit, K, eid, pf, n_boot=30, eval_clips=e2e["eval_clips"])
    bad = {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}
    fams = _stub_fams(e2e)
    fams[0] = (_fam(lon={"speed_mae_mps": bad}), fams[0][1], fams[0][2])                  # seed 0: speed MAE separated-worse
    res = RD.build_result(A, Fit, K, eid, ctl, pf, fams, n_boot=60)
    assert res["SECONDARY"]["s0"]["not_separated_worse"] is False and res["SECONDARY"]["s1"]["not_separated_worse"] is True
    assert res["decision"]["secondary_not_separated_worse"] == {"s0": False, "s1": True}
    for s in (0, 1):
        cap = res["CAPTURE"][f"s{s}"]
        assert cap["recovered_m"] == pytest.approx(float(A[s]["own"]["ade6"].mean() - A[s]["cal"]["ade6"].mean()), abs=1e-12), "CAPTURE = (ADE6 X_s - SPEEDCAL_s) / gain_speed(X_s)"
        assert cap["driver_gain_speed_m"] == pytest.approx(float(A[s]["own"]["ade6"].mean() - A[s]["gt"]["ade6"].mean()), abs=1e-12)
        assert res["ade6_m"][f"s{s}"]["perm_control"] == pytest.approx(float(A[s]["perm"]["ade6"].mean())) and res["ade6_m"][f"s{s}"]["ceiling_T0"] == pytest.approx(float(A[s]["ceiling"]["ade6"].mean()))
        assert res["decision"]["ceiling_le_minus_F"][f"s{s}"] == (res["CEILING"][f"s{s}"]["point"] <= -F)


def _pf_json(repro_failed=0, n_checked=119, ok=True, delta=3e-7):
    ctl = [{"name": "identity", "ok": True}] + [{"arm": a, "n_checked": n_checked, "n_failed": repro_failed, "ok": repro_failed == 0, "max_abs_diff": 0.0} for a in ("A", "B")]
    cell = {"status": "OK", "delta": delta, "lo": 0.0, "hi": 0.0, "separated": False}
    return {"controls_ok": ok, "controls": ctl, "summary": {"tactical": {"separated": []}, "strategic": {"separated": []}}, "tactical": {"x": cell}, "strategic": {"y": cell}, "n_windows": 2399, "n_episodes": 141}


def test_the_pairfam_substitute_control_and_the_instrument_gap(tmp_path):
    d = tmp_path / "pf"
    d.mkdir()
    (d / "pairfam_speedcal_s0.json").write_text(json.dumps(_pf_json(repro_failed=22, n_checked=79, ok=False)), encoding="utf-8")
    (d / "subctl_own_vs_banked_s0.json").write_text(json.dumps(_pf_json()), encoding="utf-8")
    blk = RD.pairfam_block(str(d), 0)
    assert blk["controls_ok"] is False and blk["instrument_gap"] and blk["instrument_gap"][0]["n_failed"] == 22, "a native reproduction failure is reported as an INSTRUMENT GAP, not a pass"
    assert blk["substitute_control"]["PASS"] is True and blk["substitute_control"]["paired_max_abs_delta"] <= 1e-6
    for name, j in (("118", _pf_json(n_checked=118)), ("failed", _pf_json(repro_failed=1)), ("notok", _pf_json(ok=False)), ("bigdelta", _pf_json(delta=2e-6))):
        (d / "subctl_own_vs_banked_s0.json").write_text(json.dumps(j), encoding="utf-8")
        assert RD.substitute_control(str(d), 0)["PASS"] is False, name
    assert RD.substitute_control(str(d), 1)["PASS"] is False and RD.substitute_control(str(d), 1)["status"].startswith("NOT RUN")
    (d / "pairfam_speedcal_s1.json").write_text(json.dumps(_pf_json()), encoding="utf-8")
    assert RD.pairfam_block(str(d), 1)["instrument_gap"] is None


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


def _load_from(e2e, d, fit=None, smoke=True):
    return RD.load_all(str(d), str(fit or e2e["fit"]), {0: str(e2e["kx"][0]), 1: str(e2e["kx"][1])}, e2e["sha"][0], e2e["sha"][1], allow_smoke=smoke)


def _refused(capsys, why, e2e, d, fit=None, smoke=True):
    """The refusal must be the NAMED one: the synthetic grid is also refused (it is not the committed 2,399 x 141), so a bare `SystemExit 2` would pass even with the guard under test removed."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        _load_from(e2e, d, fit=fit, smoke=smoke)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert why in err, f"refused, but not for {why!r}: {err[-400:]}"


def test_refusals_of_load_all(e2e, capsys):
    d = _copy_ev(e2e, "r1")
    _patch_json(d, "M2b-mixo_s0-speedcal-cal", lambda j: j["speedcal"].__setitem__("limit", 48))
    _refused(capsys, "LIMITED run", e2e, d, smoke=False)                                      # a LIMITED (smoke) run is refused
    _load_from(e2e, d, smoke=True)
    d = _copy_ev(e2e, "r2")
    _patch_json(d, "M2b-mixo_s1-speedcal-perm", lambda j: j["speedcal"].__setitem__("x_sha256", "0" * 64))
    _refused(capsys, "scored checkpoint", e2e, d)                                              # a wrong checkpoint sha256
    d = _copy_ev(e2e, "r3")
    _patch_json(d, "M2b-mixo_s0-speedcal-cal", lambda j: j["speedcal"].__setitem__("mode", "perm"))
    _refused(capsys, "says mode/seed/tag/target", e2e, d)                                      # a mislabelled mode
    d = _copy_ev(e2e, "r4")
    _patch_json(d, "M2b-mixo_s0-speedcal-gt", lambda j: j.pop("speedcal"))
    _refused(capsys, "carries no speedcal block", e2e, d)
    d = _copy_ev(e2e, "r5")
    _patch_json(d, "M2b-mixo_s0-speedcal-cal", lambda j: j.__setitem__("tac", "oracle"))
    _refused(capsys, "relabelled", e2e, d)                                                     # the carrier label was not relabelled
    d = _copy_ev(e2e, "r6")
    (d / "speedcal_cells_M2b-mixo_s1-speedcal-cal.npz").unlink()
    _refused(capsys, "speedcal_cells_M2b-mixo_s1-speedcal-cal.npz is missing", e2e, d)       # a missing cells dump
    d = _copy_ev(e2e, "r7")
    (d / "ceiling_map_M2b-mixo_s0-speedcal-ceiling.json").unlink()
    _refused(capsys, "ceiling_map_M2b-mixo_s0-speedcal-ceiling.json is missing", e2e, d)     # a missing ceiling map
    d = _copy_ev(e2e, "r8")
    z = dict(np.load(d / "eval_M2b-mixo_s1-speedcal-cal.npz"))
    z["eid"] = z["eid"][::-1].copy()
    np.savez_compressed(d / "eval_M2b-mixo_s1-speedcal-cal.npz", **z)
    _refused(capsys, "window list", e2e, d)                                                    # a window list that moved
    # a LIMITED fit and a fit of another checkpoint: `load_fit` itself, then through `load_all`
    fd = e2e["tmp"] / "fit_lim"
    shutil.copytree(e2e["fit"], fd, dirs_exist_ok=True)
    pj = fd / "fit_M2b-mixo_s0-speedcal-fit.json"
    j = json.loads(pj.read_text(encoding="utf-8"))
    j["limit"] = 100
    pj.write_text(json.dumps(j), encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        RD.load_fit(str(fd), 0, e2e["sha"][0], False)
    assert e.value.code == 2 and "LIMITED (smoke) fit" in capsys.readouterr().err, "a LIMITED fit is never a result"
    assert RD.load_fit(str(fd), 0, e2e["sha"][0], True)["json"]["limit"] == 100, "the smoke switch admits it"
    _refused(capsys, "LIMITED (smoke) fit", e2e, e2e["ev"], fit=fd, smoke=False)
    j["limit"] = None
    j["x_sha256"] = "1" * 64
    pj.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        RD.load_fit(str(fd), 0, e2e["sha"][0], True)
    assert e.value.code == 2 and "was fitted on checkpoint" in capsys.readouterr().err, "a fit made on another checkpoint"
    _refused(capsys, "was fitted on checkpoint", e2e, e2e["ev"], fit=fd)


def test_the_committed_window_counts_are_refused_when_wrong(e2e):
    with pytest.raises(SystemExit) as e:
        _load_from(e2e, e2e["ev"], smoke=False)
    assert e.value.code == 2, "the synthetic grid is not the committed 2,399 x 141"


def _cli(e2e, ev, out, *extra, fit=None):
    return RD.main(["--eval-dir", str(ev), "--fit-dir", str(fit or e2e["fit"]), "--kindx-dir-s0", str(e2e["kx"][0]), "--kindx-dir-s1", str(e2e["kx"][1]), "--expect-sha0", e2e["sha"][0],
                    "--expect-sha1", e2e["sha"][1], "--eval-grid", str(e2e["sw"]["root"] / "eval_grid_windows.json"), "--out", str(out), "--n-boot", "30", *extra])


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
    z = dict(np.load(d / "eval_M2b-mixo_s0-speedcal-identity.npz"))
    z["ade6"] = z["ade6"] + 1e-3
    np.savez_compressed(d / "eval_M2b-mixo_s0-speedcal-identity.npz", **z)
    out2 = e2e["tmp"] / "cli" / "BAD_READ.json"
    with pytest.raises(SystemExit) as e:
        _cli(e2e, d, out2, "--smoke-first-n", str(n))
    assert e.value.code == 2 and not out2.exists(), "a failed control: rc 2 and NO outcome file"
    assert json.loads((out2.parent / "BAD_READ_controls.json").read_text(encoding="utf-8"))["seeds"]["s0"][CTL[0]]["PASS"] is False


def test_the_reader_refuses_the_same_sha_twice(e2e):
    with pytest.raises(SystemExit) as e:
        RD.main(["--eval-dir", str(e2e["ev"]), "--fit-dir", str(e2e["fit"]), "--kindx-dir-s0", str(e2e["kx"][0]), "--kindx-dir-s1", str(e2e["kx"][1]), "--expect-sha0", e2e["sha"][0],
                 "--expect-sha1", e2e["sha"][0], "--out", str(e2e["tmp"] / "x.json")])
    assert e.value.code == 2


# ============================================================================================== (G) the runner
def _fwd(p):
    return str(p).replace("\\", "/")


@bash
def test_runner_syntax_dry_run_and_load_bearing_lines(tmp_path):
    src = RUNNER.read_text(encoding="utf-8")
    assert "\r" not in src
    assert subprocess.run([BASH, "-n", str(RUNNER)], capture_output=True).returncode == 0
    out_dir = tmp_path / "never"
    env = dict(os.environ, SPEEDCAL_OUT=_fwd(out_dir), SPEEDCAL_B2=_fwd(tmp_path / "nobox"), SPEEDCAL_PY=_fwd(sys.executable), SPEEDCAL_BANK=_fwd(tmp_path / "nobank"))
    r = subprocess.run([BASH, str(RUNNER), "--dry-run"], capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    o = r.stdout
    for need in ("== check", "== fit seed 0", "== fit seed 1", "== evals seed 0 (own, identity, gt, cal, perm, ceiling)", "== evals seed 1 (own, identity, gt, cal, perm, ceiling)", "== evalsB seed 0",
                 "== pairfam seed 0", "substitute control", "== read", "== bank", "--check-inputs", "ZZSPEEDCAL-DONE", "ZZSPEEDCAL-STOP", "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d",
                 "speedcal-own", "speedcal-identity", "speedcal-gt", "speedcal-cal", "speedcal-perm", "speedcal-ceiling", "--tac oracle --tac-groups speed --tac-alpha 1", "--families", "CPU only",
                 "--hblend-windows", "fit_windows_M2b_s0.npz", "fit_windows_M2b_s1.npz", "--a-json", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"):
        assert need in o, need
    assert re.search(r"fit --seed 0 --x \S*M2b_s0_mixo/ckpt_last\.pt --expect-sha c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", o), "seed 0 is X_s0 (the committed sha)"
    assert re.search(r"fit --seed 1 --x \S*M2b_s1_mixo/ckpt_last\.pt ", o), "seed 1 is X_s1"
    assert re.search(r"evals --seed 1 --x \S*M2b_s1_mixo/ckpt_last\.pt ", o) and re.search(r"evals --seed 0 --x \S*M2b_s0_mixo/ckpt_last\.pt ", o)
    for sd in (0, 1):
        assert re.search(rf"--a \S*eval_M2b-mixo_s{sd}-speedcal-cal\.npz --b \S*eval_M2b-mixo_s{sd}-speedcal-own\.npz", o), "pairfam: SPEEDCAL - X_s"
        m = re.search(rf"substitute control: \S+ -I \S+ --a \S*eval_M2b-mixo_s{sd}-speedcal-own\.npz --a-json (\S+)\.json --b (\S+)\.npz --b-json (\S+)\.json", o)
        assert m and m.group(1) == m.group(2) == m.group(3), "the substitute control: our own npz against the banked npz, BOTH read against the banked eval json"
    assert "eval_M2b-mixo_s0.json" in o and "eval_M2b-mixo_s1-kindx-own.json" in o, "the banked X_0 / X_1 evals the coordinator named"
    assert not out_dir.exists() and not (tmp_path / "nobank").exists(), "a dry run writes nothing"
    for need in ("CUDA_VISIBLE_DEVICES=-1", "SPEEDCAL_THREADS", "MIN_FREE_KB=4194304", "ZZSPEEDCAL-STOP", "ZZSPEEDCAL-DONE", "[ \"$free\" -ge \"$MIN_FREE_KB\" ]", "--check-inputs", "bank_files",
                 "X_s0 and X_s1 are the same file: no second seed", "sha256 of $X0 is $g0, expected $X0_SHA"):
        assert need in src, need
    assert "kill" not in src and "taskkill" not in src and "git add" not in src and "git commit" not in src and "git push" not in src and " rm " not in src, "the runner kills, stages and removes nothing"
    assert "scp " not in src and "ssh " not in src and "tanitad-box2" not in src, "box 2 is not touched at all: the checkpoints are read from the dev box"
    assert '--a "$OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz" --a-json "$pb.json" --b "$pb.npz" --b-json "$pb.json"' in src, "the substitute control reads BOTH arms against the banked json"
    assert '--a "$OUT/eval/eval_M2b-mixo_s$s-speedcal-cal.npz" --b "$OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz"' in src
    m = re.search(r"all\)\s+(stage_check; stage_fit 0; stage_fit 1; stage_evals 0; stage_evals 1; stage_evalsB; stage_pairfam; stage_read)", src)
    assert m, "all = check, fit 0, fit 1, evals 0, evals 1, baseline, pairfam, read (then bank)"


@bash
def test_runner_stops_with_the_marker_on_a_missing_input_and_still_banks_what_exists(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "some_result.txt").write_text("evidence", encoding="utf-8")
    bank = tmp_path / "bank"
    env = dict(os.environ, SPEEDCAL_OUT=_fwd(out), SPEEDCAL_B2=_fwd(tmp_path / "nobox"), SPEEDCAL_SC=_fwd(tmp_path / "nosc"), SPEEDCAL_PY=_fwd(sys.executable), SPEEDCAL_BANK=_fwd(bank))
    r = subprocess.run([BASH, str(RUNNER), "check"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 3 and "ZZSPEEDCAL-STOP" in r.stdout and ("missing input" in r.stdout or "less than 4 GB" in r.stdout), r.stdout[-1500:]
    assert (bank / "some_result.txt").exists() and (bank / "some_result.txt").read_text(encoding="utf-8") == "evidence", "a STOP still banks what exists: a failed control is evidence"
    r2 = subprocess.run([BASH, str(RUNNER), "nonsense"], capture_output=True, text=True, env=env, timeout=60)
    assert r2.returncode == 2


def test_bank_survives_a_destination_path_longer_than_windows_max_path(tmp_path):
    """The real bank lives under a deep tree (> 260 characters); a plain `shutil.copy2` raised FileNotFoundError there and the run's STOP/DONE could not bank."""
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "sub" / "r.txt").write_text("evidence", encoding="utf-8")
    deep = tmp_path
    for i in range(5):
        deep = deep / ("d" * 60 + str(i))
    bank = deep / "bank"
    assert len(str(bank)) > 300
    env = dict(os.environ, SPEEDCAL_OUT=_fwd(out), SPEEDCAL_B2=_fwd(tmp_path / "nobox"), SPEEDCAL_SC=_fwd(tmp_path / "nosc"), SPEEDCAL_PY=_fwd(sys.executable), SPEEDCAL_BANK=_fwd(bank))
    r = subprocess.run([BASH, str(RUNNER), "bank"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "banked 1 files" in r.stdout, (r.stdout[-800:], r.stderr[-800:])
    got = str(bank / "sub" / "r.txt")
    got = ("\\\\?\\" + os.path.abspath(got)) if os.name == "nt" else got
    try:
        assert os.path.exists(got) and open(got, encoding="utf-8").read() == "evidence"
    finally:                                                    # a tree past MAX_PATH is not removable by pytest's own cleanup: remove it with the extended-length prefix
        if os.name == "nt":
            shutil.rmtree("\\\\?\\" + os.path.abspath(str(tmp_path / ("d" * 60 + "0"))), ignore_errors=True)
