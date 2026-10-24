"""PREREG_V8F_DSCALE: half the unique training clips at fixed steps (`--data-frac 0.5`, ONE lever, seed 0 only).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): the trainer has had `--data-frac` since v8F-mini's first commit, a G-DVB entry for it
(`data_frac -> data:n_clips`) and a test that the gate REFUSES a fraction that was not applied. Nothing pinned that a run launched with `--data-frac 0.5` TRAINS ON THE HALF: that the flag reaches
`select_train_clips`, that the window list the batches are drawn from is the half's, and -- the property the PREREG's whole arithmetic rests on -- that the STEP BUDGET stays B's, because
`S = ceil(n_windows_all_clips / batch)` counts ALL clips. Pinned here, hop by hop, on a synthetic world (and on the real clip list: `raw/2026-10-25-dscale/measure_dscale.{py,json,out}`):

  REACH   argv --data-frac 0.5 -> select_train_clips (the first round(0.5 x N) in sha12 order, nested above 0.25) -> the clips build_data keeps (n_clips, train_sha256) -> the window list
          (n_windows) -> the (clip, t) items the training loop's batches are built from (a spy on V8FMiniWindows.item: every item belongs to the half) -> summary.json {data_frac, n_clips}
  BUDGET  S and the step count are B's (the same on the full and the half run) and Q therefore sees each of its windows ~2x as often
  EVAL    no eval path reads `data_frac` (a static scan of the whole stack, and the model config cannot carry it): the same eval grid and the same four evals as B's
  QUEUE   `gatedscale` / `traindscale` / `evaldscale`: CRLF, dispatched, ONE lever (Q's argv = B's + `--data-frac 0.5`), what is PRINTED is what RUNS, every other stage byte-identical
  CHAIN   the order, the dry run, the gate assertions (RUN against synthetic selfchecks: the half's clips, the all-clips budget, the model = B's), no seed-1 stage, HIST's guards
  READER  D-T = r = RMSE(Q)/RMSE(B) - 1 of T's slot-pooled present-slot latcons RMSE: +6 % -> DATA-LIMITED, +1 % -> NOT DATA-LIMITED, -6 % -> EPOCH-LIMITED; Q against itself reads +0 [0, 0];
          every refusal of the PREREG (rc 2)

Each identity has a source mutation that turns it red: `raw/2026-10-25-dscale/mutate_dscale.py` (output beside it). Everything that needs the v8F-mini research package (reader, queue, chain) is below
the reachability block and is SKIPPED, not failing, in a checkout that has only `stack/` (the lander's probe).
Test names carry their section (`reach` / `reader` / `queue` / `chain`) so the mutation harness can run the section a mutant belongs to.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_dscale", Path(__file__).resolve().parent / "test_v8f_mini.py")
TR = _T.TR
torch.set_num_threads(1)
STEPS_B, S_B, WINDOWS_B, N_B, N_HALF = 19164, 4791, 306566, 4512, 2256       # B's banked record (raw/full/runs/M2b_s0/summary.json) and the PREREG's round(0.5 x 4512)
BANKED_M2B_TOTAL = 7214495
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


# =================================================================================================================================================
# REACH: --data-frac 0.5 -> the clips, the windows, the batches; the budget is B's
# =================================================================================================================================================
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_dscale")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def _train(out, extra, steps=0):
    """A real `TR.main` run on the synthetic world, with a spy on `V8FMiniWindows.item` recording the (clip index) of every window the loop draws."""
    cap = []
    orig = D.V8FMiniWindows.item

    def spy(self, ci, *a, **k):
        cap.append(int(ci))
        return orig(self, ci, *a, **k)
    D.V8FMiniWindows.item = spy
    common = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "2", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
    if steps:
        common += ["--steps", str(steps)]
    try:
        assert TR.main(["--out", str(out), *common, *extra]) == 0
    finally:
        D.V8FMiniWindows.item = orig
    return {"items": cap, "root": Path(out) / "_synthetic", "sc": json.load(open(Path(out) / "selfcheck.json", encoding="utf-8")),
            "sm": json.load(open(Path(out) / "summary.json", encoding="utf-8")), "out": Path(out)}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("dscale_runs")
    return {f: _train(base / f"f{int(100 * f)}", [] if f == 1.0 else ["--data-frac", str(f)]) for f in (1.0, 0.5)}


def _sha(ids):
    return hashlib.sha256(chr(10).join(ids).encode("utf-8")).hexdigest()


def test_reach_the_flag_selects_the_first_half_of_the_clips_in_sha12_order_and_the_subsets_nest():
    names = [f"syn-{i:03d}" for i in range(100)]
    order = sorted(names, key=D.sha12)
    assert D.select_train_clips(names, 0.5) == order[:50] and D.select_train_clips(names, 1.0) == order
    for frac, n in ((0.25, 25), (0.5, 50), (1.0, 100)):
        assert len(D.select_train_clips(names, frac)) == n
    q, h, f = (set(D.select_train_clips(names, x)) for x in (0.25, 0.5, 1.0))
    assert q < h < f, "the subsets are NESTED: 25 % < 50 % < 100 %"
    assert set(D.select_train_clips(list(reversed(names)), 0.5)) == set(order[:50]), "the half does not depend on the manifest's order"
    six = [f"clip-{i:03d}" for i in range(6)]
    assert [len(D.select_train_clips(six, x)) for x in (0.25, 0.5, 1.0)] == [2, 3, 6], "round(0.25 x 6) = 2 (round half to even), not the truncated 1"
    four_five_twelve = [f"c{i}" for i in range(N_B)]
    assert len(D.select_train_clips(four_five_twelve, 0.5)) == N_HALF == round(0.5 * N_B)


def test_reach_the_trainer_hands_the_flag_to_select_train_clips_and_sizes_the_budget_from_all_clips_static():
    src = (REPO / "stack" / "scripts" / "v8f_mini_train.py").read_text(encoding="utf-8")
    assert "clips = D.select_train_clips(clips_all, a.data_frac)" in src, "the flag must reach select_train_clips (the mutant drops it)"
    assert 'allw = D.train_windows(store, J, clips_all, hist=cfg.hist, start=cfg.window_start)' in src, "the budget's window list is built from ALL clips"
    assert 'wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)' in src, "the training window list is built from the kept clips"
    assert 'S = math.ceil(info["n_windows_all_clips"] / a.batch)' in src and "total = a.steps or S * a.steps_mult" in src, "S counts ALL clips: the steps do not follow the fraction"
    assert 'sampler = D.ChunkedSampler(wins, a.batch' in src, "the batches are drawn from the kept clips' windows"
    assert TR.FLAG_CHECKS["data_frac"] == "data:n_clips", "the launch gate reads the built clip count back"
    assert any(act.dest == "data_frac" and act.default == 1.0 and "--data-frac" in act.option_strings for act in TR.build_parser()._actions), "the flag exists and defaults to the full data"


def test_reach_the_half_run_trains_on_the_half_and_every_batch_item_belongs_to_it(world, runs):
    names = world["names"]["train"]
    half = sorted(names, key=D.sha12)[:3]
    J = D.load_join(Path(runs[0.5]["root"]) / "join_train.npz")
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    ok = {ci_of[c] for c in half}
    q, b = runs[0.5], runs[1.0]
    assert q["sm"]["data_frac"] == 0.5 and b["sm"]["data_frac"] == 1.0
    assert q["sm"]["n_clips"] == 3 and b["sm"]["n_clips"] == 6 and q["sc"]["info"]["n_clips_all"] == 6 and q["sc"]["info"]["n_clips_expected"] == 3
    assert q["sm"]["train_sha256"] == _sha(sorted(half, key=D.sha12)) != b["sm"]["train_sha256"] == _sha(sorted(names, key=D.sha12))
    assert len(q["items"]) > 8 and set(q["items"]) <= ok, "every window the loop draws belongs to the half's clips"
    assert set(q["items"]) == ok, "and the loop reaches all of the half"
    assert set(b["items"]) - ok, "CONTROL: the full run draws windows of clips outside the half (so the spy would see a leak)"
    assert q["sm"]["n_windows"] == sum(len(range(3, int(J["n"][ci_of[c]]) - D.HORIZON)) for c in half) < b["sm"]["n_windows"]


def test_reach_the_step_budget_is_b_s_because_s_counts_all_clips(runs):
    q, b = runs[0.5], runs[1.0]
    assert q["sc"]["info"]["n_windows_all_clips"] == b["sc"]["info"]["n_windows_all_clips"] == b["sm"]["n_windows"]
    assert q["sm"]["steps"] == b["sm"]["steps"] == q["sc"]["steps_total"] == b["sc"]["steps_total"]
    assert q["sm"]["S"] == b["sm"]["S"] == math.ceil(b["sm"]["n_windows"] / 8) and q["sm"]["steps_mult"] == b["sm"]["steps_mult"] == 1
    assert q["sm"]["steps"] == q["sm"]["S"] * q["sm"]["steps_mult"]
    passes = lambda r: r["sm"]["steps"] * 8 / r["sm"]["n_windows"]      # noqa: E731
    assert passes(q) / passes(b) == pytest.approx(b["sm"]["n_windows"] / q["sm"]["n_windows"]) == pytest.approx(2.0), "Q sees each of its windows twice as often as B"
    assert len(q["items"]) == len(b["items"]), "the same number of windows is drawn"


def test_reach_no_eval_path_reads_data_frac():
    """A static scan of the whole stack (tests excluded): the only file that mentions the flag is the trainer, and the model config -- the one thing a checkpoint hands the evaluators -- cannot carry it."""
    hits = []
    for root in (REPO / "stack", REPO / "taniteval"):
        for f in root.rglob("*.py"):
            if "tests" in f.relative_to(REPO).parts or "__pycache__" in f.parts or ".claude" in f.parts:
                continue
            t = f.read_text(encoding="utf-8", errors="replace")
            if "data_frac" in t or "data-frac" in t:
                hits.append(f.relative_to(REPO).as_posix())
    assert hits == ["stack/scripts/v8f_mini_train.py"], hits
    assert "data_frac" not in M.V8FMiniConfig.__dataclass_fields__ and "data_frac" not in M.V8FMiniConfig().__dict__
    for ev in ("v8f_mini_eval.py", "v8f_t1_eval.py"):
        src = (REPO / "stack" / "scripts" / ev).read_text(encoding="utf-8")
        assert "data_frac" not in src and "data-frac" not in src and "select_train_clips" not in src, f"{ev}: the eval path reads the training fraction"
    import v8f_mini_eval as EV
    assert not any("data" in a.dest and "frac" in a.dest for a in EV.build_parser()._actions) if hasattr(EV, "build_parser") else True


# =================================================================================================================================================
# THE REAL CLIP LIST (skipped where the file is not on disk): the half's identity on the clips the trainer reads
# =================================================================================================================================================
REAL_V2 = Path("C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt")
CODE = PKG / "code"
READ_P = CODE / "v8f_dscale_read.py"
HIST_P = CODE / "v8f_hist_read.py"
GRID_P = CODE / "v8f_grid_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_dscale.sh"
CHAIN_HIST = CODE / "box2_ops" / "chain_v8f_hist.sh"
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
BANKED = PKG / "raw" / "full" / "runs"
PREREG = REPO / "Project Steering" / "PREREG_V8F_DSCALE.md"
needs_pkg = pytest.mark.skipif(not (READ_P.exists() and BANKED.exists()), reason="the v8F-mini package is not in this checkout")
R = _load("v8f_dscale_read_under_test", READ_P) if READ_P.exists() else None            # importable without the package (the lander's stack-only probe)
_DK = _load("test_v8f_dk_for_dscale", Path(__file__).resolve().parent / "test_v8f_dk.py") if READ_P.exists() else None
C0, NC = M.N_GOAL + M.N_LAT + M.N_LON, len(M.CONS_NAMES)
LATCONS = [C0 + i for i in range(12)]


@pytest.mark.skipif(not REAL_V2.exists() or not BANKED.exists() or R is None, reason="the real train clip list (train4572_v2manifest.pt) is not on this disk")
def test_reach_the_real_clip_list_reproduces_b_and_gives_the_half():
    """On the REAL 4,572 clip ids: after the trainer's own `--exclude 540:600`, the sorted list and the excluded ids hash to B's RECORDED train_sha256 / excluded_sha256, so the half the reader
    derives is the half the trainer keeps (frame counts derived: n_frames = ceil(T_out / 2) + 1 reproduces B's 306,566 windows exactly)."""
    o = torch.load(REAL_V2, map_location="cpu", weights_only=True)
    ids, nf = list(o["clip_id"]), [(int(t) + 1) // 2 + 1 for t in o["T_out"]]
    exp = R.expected_from_manifest(list(zip(ids, nf)), "540:600")
    rsm = json.loads((BANKED / "M2b_s0" / "summary.json").read_text(encoding="utf-8"))
    assert (exp["n_manifest"], exp["n_b"], exp["n_half"]) == (4572, N_B, N_HALF)
    assert exp["b_sha256"] == rsm["train_sha256"] and exp["excluded_sha256"] == rsm["excluded_sha256"] == HOLD
    assert exp["windows_b"] == rsm["n_windows"] == WINDOWS_B and math.ceil(exp["windows_b"] / 64) == rsm["S"] == S_B and rsm["S"] * rsm["steps_mult"] == rsm["steps"] == STEPS_B
    order = sorted([c for c in ids if c not in set(sorted(ids, key=D.sha12)[540:600])], key=D.sha12)
    assert exp["half"] == order[:N_HALF] and exp["half_sha256"] == _sha(order[:N_HALF]) and exp["first_sha12"] == D.sha12(order[0]) and exp["last_sha12"] == D.sha12(order[N_HALF - 1])
    assert 0.49 < exp["windows_q"] / exp["windows_b"] < 0.51, "a half of the clips holds about half of the windows"


# =================================================================================================================================================
# THE READER (code/v8f_dscale_read.py)
# =================================================================================================================================================
def _ej(path, fn):
    p = Path(path)
    j = json.loads(p.read_text(encoding="utf-8"))
    r = fn(j)
    p.write_text(json.dumps(j if r is None else r), encoding="utf-8")


def _b_real(s):
    d = BANKED / f"M2b_s{s}"
    return ({n: json.loads((d / n).read_text(encoding="utf-8")) for n in ("summary.json", "argv.json", "selfcheck.json")}, (d / "train_log.jsonl").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _syn():
    """A SYNTHETIC cache manifest with B's real shape: 4,572 clips whose frame counts make B's 4,512-clip list hold EXACTLY the banked 306,566 windows (68 x 4,262 + 67 x 250); the order of the file
    is shuffled so nothing depends on it. The reader's list arithmetic is computed here INDEPENDENTLY (plain sorted / slice / hash)."""
    ids = [f"syn-clip-{i:05d}" for i in range(4572)]
    ex = sorted(ids, key=D.sha12)[540:600]
    exs = set(ex)
    order = sorted([c for c in ids if c not in exs], key=D.sha12)
    assert len(order) == N_B
    nf = {c: 100 for c in ids}
    for k, c in enumerate(order):
        nf[c] = 101 if k < 4262 else 100
    half = order[:N_HALF]
    shuf = list(ids)
    random.Random(1).shuffle(shuf)
    lines = "".join(json.dumps({"clip": c, "n_frames": nf[c]}) + "\n" for c in shuf)
    w = lambda cs: sum(nf[c] - 33 for c in cs)       # noqa: E731    len(range(3, n - 30)) = n - 33
    assert w(order) == WINDOWS_B
    return {"ids": ids, "nf": nf, "order": order, "half": half, "lines": lines, "b_sha": _sha(order), "ex_sha": _sha(ex), "half_sha": _sha(half), "windows_q": w(half), "last_half_sha": _sha(order[N_HALF:])}


@pytest.fixture
def syn_list(monkeypatch):
    """The synthetic clip list stands in for the real one: its holdout is the 'RA-1 holdout' of this world."""
    monkeypatch.setattr(R.PGT, "HOLDOUT_SHA", _syn()["ex_sha"])


@pytest.fixture
def no_known(monkeypatch, syn_list):
    """The banked premises (F 0.1207, LAMBDA +0.2678 / +0.1667, S5 excess) are the REAL banked M2b values; a synthetic world cannot hold them. The tests of everything else replace the check;
    `test_reader_known_values_are_checked...` runs the real one."""
    monkeypatch.setattr(R.PGT, "known_values", lambda *a, **k: {"checked": False, "why": "synthetic test input"})


def _dworld(d, *, d_ade=0.0, err_b=0.6, err_q=0.6, sp_b=11.0, sp_q=11.0, gap_b=0.65, off_b=1.0, lam_b=0.27, lam_q=0.27, cons_q=1.0, cons_q_head=1.0, with_q=True, same_as_b=False,
            tail_only_b=False):
    """Synthetic evals + runs for B (M2b, both seeds, the REAL banked run records with the synthetic clip list's hashes) and Q = B + --data-frac 0.5 (seed 0). Knobs: `d_ade` = ADE6(Q) - ADE6(B)
    (the lateral offset beyond 2 s is SOLVED for it), `err_*` = T's present-slot error (sd of the noise on every present constraint slot, the SAME draws for both arms: a paired design, so
    RMSE(Q)/RMSE(B) = err_q/err_b EXACTLY on every group), `cons_q` / `cons_q_head` scale Q's logged training `cons` over the last 10 % of steps / before it, `same_as_b` = Q's evals ARE B's arrays."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    syn = _syn()
    (d / "manifest_train.jsonl").write_text(syn["lines"], encoding="utf-8")
    w = _DK.World(d)
    N, eid, gt = w.n, w.eid, w.gt
    lead = np.array([s == "LEAD" for s in w.states])
    rng = np.random.default_rng(5)
    g = np.zeros((N, M.CODE_DIM))
    g[:, :C0] = rng.normal(0, 1, (N, C0))
    present = rng.random((N, NC)) < 0.45
    z_all = np.random.default_rng(31).normal(0, 1, (N, NC))
    g[:, C0:] = np.where(present, rng.normal(0, 1, (N, NC)), 0.0)
    fb = np.zeros(N, bool)
    fb[:5] = True
    ev, ru = d / "eval", d / "runs"
    ev.mkdir(exist_ok=True)
    ru.mkdir(exist_ok=True)

    def ade(p):
        return np.linalg.norm(p.astype(np.float32).astype(np.float64) - gt, axis=-1).mean(1)

    def std_plan(sp, off, nz):
        p = _DK.path_at(np.full(N, sp), N)
        p[:, 10:, 1] += (off * (1.0 + 0.05 * nz))[:, None]
        return p

    def solve_off(sp, nz, target):
        lo, hi = 0.0, 40.0
        if ade(std_plan(sp, lo, nz)).mean() >= target:
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ade(std_plan(sp, mid, nz)).mean() < target else (lo, mid)
        return 0.5 * (lo + hi)
    ade_b, keep = {}, {}
    for s in (0, 1):
        nz = np.random.default_rng(100 + s).normal(0, 1, N)
        for who in ("B", "Q"):
            if who == "Q" and (s != 0 or not with_q):
                continue
            is_q = who == "Q"
            tag = "M2b-df50_s0" if is_q else f"M2b_s{s}"
            if is_q and same_as_b:
                pred, lam_w, tc = keep[s]
            else:
                pred = std_plan(sp_b, off_b + gap_b * s, nz) if not is_q else std_plan(sp_q, solve_off(sp_q, nz, ade_b[s] + d_ade), nz)
                lam_w = (lam_q if is_q else lam_b) * (1.0 + 0.3 * np.random.default_rng(7 + s).normal(0, 1, N))
                tc = g.copy()
                tc[:, C0:] = np.where(present, g[:, C0:] + (err_q if is_q else err_b) * z_all, 0.0)
            if not is_q:
                ade_b[s] = ade(pred).mean()
                keep[s] = (pred, lam_w, tc)
            orc = _DK.path_at(np.full(N, 10.0), N)
            lat = orc.copy()
            lat[:, :, 0] += np.where(lead, lam_w, 0.0)[:, None] * np.minimum(1.0, (np.arange(30) + 1) / 10.0)[None]
            ckpt = "D:\\runs\\M2b_s0_df50\\ckpt_last.pt" if is_q else None
            np.savez(ev / f"eval_{tag}.npz", pred=pred.astype(np.float32), ade6=ade(pred), eid=eid)
            ej = {"raw": {"traj": {"ade6": float(ade(pred).mean())}, "tactical": {"acc": 0.5}, "strategic": {"acc": 0.5}, "families": {}}}
            if ckpt:
                ej["ckpt"] = ckpt
            (ev / f"eval_{tag}.json").write_text(json.dumps(ej), encoding="utf-8")
            np.savez(ev / f"eval_{tag}-tacabs.npz", pred=pred.astype(np.float32), ade6=ade(pred), eid=eid, tac_code=tc, tac_oracle_fallback=fb)
            (ev / f"eval_{tag}-tacabs.json").write_text(json.dumps({"tac_manipulation": {"mode": "absmask"}, **({"ckpt": ckpt} if ckpt else {})}), encoding="utf-8")
            if not is_q:                                                              # the banked 2026-10-20 format: no tac_code (WC Amendment 1)
                np.savez(ev / f"eval_{tag}-tacorc.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=eid, tac_oracle_fallback=fb)
                ag = g.copy()
                ag[:, :22] = np.random.default_rng(99).normal(0, 1, (N, 22))
                np.savez(ev / f"eval_{tag}-tacorc-all-goal.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=eid, tac_code=ag, tac_oracle_fallback=fb)
            else:
                np.savez(ev / f"eval_{tag}-tacorc.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=eid, tac_code=g, tac_oracle_fallback=fb)
            (ev / f"eval_{tag}-tacorc.json").write_text(json.dumps({"tac_manipulation": {"mode": "oracle"}, **({"ckpt": ckpt} if ckpt else {})}), encoding="utf-8")
            lc = g.copy()
            lc[:, LATCONS] = tc[:, LATCONS] + 0.01
            np.savez(ev / f"eval_{tag}-tacorc-all-latcons.npz", pred=lat.astype(np.float32), ade6=ade(lat), eid=eid, tac_code=lc, tac_oracle_fallback=fb)
            (ev / f"eval_{tag}-tacorc-all-latcons.json").write_text(json.dumps({"tac_manipulation": {"mode": "oracle"}, **({"ckpt": ckpt} if ckpt else {})}), encoding="utf-8")
            # ---- the run: B's records are the REAL banked ones (with the synthetic clip list's hashes); Q's are B's with the lever and the build facts changed
            rd = ru / ("M2b_s0_df50" if is_q else f"M2b_s{s}")
            rd.mkdir(parents=True, exist_ok=True)
            real, log = _b_real(s)
            real["summary.json"].update(train_sha256=syn["b_sha"], excluded_sha256=syn["ex_sha"])             # the synthetic clip list's hashes stand in for the real list's
            real["selfcheck.json"]["info"].update(train_sha256=syn["b_sha"], excluded_sha256=syn["ex_sha"])
            if not is_q:
                for n_, j in real.items():
                    (rd / n_).write_text(json.dumps(j), encoding="utf-8")
                (rd / "train_log.jsonl").write_text(log, encoding="utf-8")
                continue
            bl = real["summary.json"]["levers"]
            argv = list(real["argv.json"]["argv"])
            at = argv.index("--out")
            argv[at:at] = ["--data-frac", "0.5"]                                      # B's argv + --data-frac 0.5 and the run dir
            argv[argv.index("--out") + 1] = "D:\\runs\\M2b_s0_df50"
            lev = {**bl, "cons_absent": "mask", "p_gt_start": 1.0, "p_gt_end": 0.2, "residual_zero_init": False, "hist": 4, "hmax": 16, "window_start": 3,
                   "w": {**bl["w"], "w_tac": 1.0, "w_cons": 0.5}}
            sha = "dfM2bq0"
            sm = {**real["summary.json"], "data_frac": 0.5, "n_clips": N_HALF, "n_windows": syn["windows_q"], "train_sha256": syn["half_sha"], "argv_sha": sha,
                  "seconds": real["summary.json"]["seconds"] * 0.95, "levers": lev}
            (rd / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
            (rd / "argv.json").write_text(json.dumps({"argv": argv, "sha": sha}), encoding="utf-8")
            sc = {**real["selfcheck.json"], "declared_vs_built": "PASS", "argv_sha": sha, "levers": dict(lev, residual_zero_init=True),
                  "info": {**real["selfcheck.json"]["info"], "n_clips": N_HALF, "n_clips_expected": N_HALF, "n_clips_all": N_B, "n_windows": syn["windows_q"], "train_sha256": syn["half_sha"],
                           "window_start": 3, "window_t_min": 3, "ds_window_start": 3}}
            (rd / "selfcheck.json").write_text(json.dumps(sc), encoding="utf-8")
            rows = [json.loads(x) for x in log.splitlines() if x.strip()]
            cut = 0.9 * sm["steps"]
            for r_ in rows:
                r_["cons"] = float(r_["cons"]) * (cons_q if r_["step"] > cut else cons_q_head)
            (rd / "train_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            torch.save({"model": {"time": torch.zeros(4, 256), "pos": torch.zeros(40, 4)}, "cfg": {"hist": 4, "hmax": 16, "window_start": -1, "d": 256, "n_tok": 40}, "grid": "pool4x10",
                        "argv_sha": sha, "step": sm["steps"], "arm": "M2b", "seed": 0}, rd / "ckpt_last.pt")
    np.save(d / "p10.npy", _DK.path_at(np.full(N, 10.0), N))
    w.write_gt()
    w.write_block()
    return {"d": d, "w": w, "eval": ev, "runs": ru, "lead": lead,
            "args": ["--eval-dir", str(ev), "--runs-dir", str(ru), "--out", str(d / "dscale.json"), "--train-manifest", str(d / "manifest_train.jsonl"), "--n-boot", "300", "--dk-n-boot", "60",
                     "--no-families", "--no-k30", "--lead-block", str(d / "lead.npz"), "--p10-pred", str(d / "p10.npy"), "--gt-npz", str(d / "gt.npz")]}


def _go(f, extra=()):
    rc = R.main([*f["args"], *extra])
    out = f["d"] / "dscale.json"
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


# ---- the committed decision table --------------------------------------------------------------------------------------------------------------
@needs_pkg
def test_reader_decide_is_the_preregs_table_on_every_boundary():
    sep = {"separated": True, "point": 1.0}
    Q = lambda **k: dict(sep, **k)         # noqa: E731
    assert R.BAR == 0.05
    assert R.decide(0.05, Q()) == "DATA-LIMITED" and R.decide(0.06, Q()) == "DATA-LIMITED" and R.decide(0.5, Q()) == "DATA-LIMITED"
    assert R.decide(0.0499999, Q()) == "NOT DATA-LIMITED" and R.decide(0.01, Q()) == "NOT DATA-LIMITED" and R.decide(0.0, Q()) == "NOT DATA-LIMITED"
    assert R.decide(-0.0499999, Q(point=-1.0)) == "NOT DATA-LIMITED" and R.decide(-0.01, Q(point=-1.0)) == "NOT DATA-LIMITED"
    assert R.decide(-0.05, Q(point=-1.0)) == "EPOCH-LIMITED" and R.decide(-0.06, Q(point=-1.0)) == "EPOCH-LIMITED" and R.decide(-0.9, Q(point=-1.0)) == "EPOCH-LIMITED"
    # "or the MSE delta not separated": a size beyond the bar without separation is NOT DATA-LIMITED, in either direction
    assert R.decide(0.06, Q(separated=False)) == "NOT DATA-LIMITED" and R.decide(0.30, Q(separated=False)) == "NOT DATA-LIMITED"
    assert R.decide(-0.06, Q(separated=False, point=-1.0)) == "NOT DATA-LIMITED" and R.decide(-0.30, Q(separated=False, point=-1.0)) == "NOT DATA-LIMITED"
    # a separated delta AGAINST r's direction does not support r
    assert R.decide(0.06, Q(point=-1.0)) == "NOT DATA-LIMITED" and R.decide(-0.06, Q(point=1.0)) == "NOT DATA-LIMITED"
    assert {R.decide(r, Q(point=math.copysign(1.0, r))) for r in (-0.0499, 0.0, 0.0499)} == {"NOT DATA-LIMITED"}


@needs_pkg
def test_reader_dt_block_decides_on_the_slot_pooled_rmse_and_reports_the_window_weighted_one_beside_it():
    """The PREREG's r is the SLOT-POOLED RMSE ratio. When the window-weighted definition would read on the other side of the bar, the outcome is still the slot-pooled one, the other is reported, and
    `definitions_agree` says False. A gap between M-LAT's fall and r is a refusal, never a decision."""
    def ml(rq, rb, wq, wb, sep=True, point=1.0):
        return {"rmse_slot": {"G": rq, "B": rb}, "rmse_window": {"G": wq, "B": wb}, "fall": -(rq / rb - 1.0), "mse_delta": {"separated": sep, "point": point},
                "n_windows_present": 7, "n_slots_present": 3}
    a = R.dt_block(ml(1.06, 1.0, 1.03, 1.0))                   # slot-pooled +6 % (DATA-LIMITED), window-weighted +3 % (not)
    assert abs(a["r"] - 0.06) < 1e-12 and abs(a["r_window"] - 0.03) < 1e-12
    assert a["outcome"] == "DATA-LIMITED" and a["outcome_window_definition"] == "NOT DATA-LIMITED" and a["definitions_agree"] is False
    b = R.dt_block(ml(1.03, 1.0, 1.06, 1.0))                   # the other way round: the deciding definition is the slot-pooled one
    assert b["outcome"] == "NOT DATA-LIMITED" and b["outcome_window_definition"] == "DATA-LIMITED" and b["definitions_agree"] is False
    c = R.dt_block(ml(0.94, 1.0, 0.99, 1.0, point=-1.0))
    assert c["outcome"] == "EPOCH-LIMITED" and c["outcome_window_definition"] == "NOT DATA-LIMITED"
    broken = ml(1.06, 1.0, 1.06, 1.0)
    broken["fall"] = 0.0                                       # the instrument's fall and r disagree
    with pytest.raises(SystemExit) as e:
        R.dt_block(broken)
    assert e.value.code == 2


@needs_pkg
def test_reader_decision_table_plus_6_plus_1_minus_6_percent(tmp_path, no_known):
    """r = +6 % -> DATA-LIMITED, +1 % -> NOT DATA-LIMITED, -6 % -> EPOCH-LIMITED, each with the committed next-step cell verbatim, on a world where T's error is the only thing that moved."""
    for tag, err_q, want in (("up6", 0.6 * 1.06, "DATA-LIMITED"), ("up1", 0.6 * 1.01, "NOT DATA-LIMITED"), ("dn6", 0.6 * 0.94, "EPOCH-LIMITED")):
        rc, r = _go(_dworld(tmp_path / tag, err_q=err_q))
        assert rc == 0 and r["DECISION"] == want and r["VERDICT"] == want and r["base"] == "M2b" and r["arm"] == "M2b-df50" and r["seed"] == 0, (tag, r and r["reading"])
        assert R.NEXT_TEXT[want] in r["next"] and r["next"] == R.NEXT[want]
        dt = r["rows"]["s0"]["dt"]
        assert dt["r"] == pytest.approx(err_q / 0.6 - 1.0, abs=1e-9) and dt["outcome"] == want and dt["definitions_agree"] is True and dt["bar"] == 0.05
        assert dt["rmse_slot"]["Q"] / dt["rmse_slot"]["B"] - 1.0 == pytest.approx(dt["r"], abs=1e-12)
        if want != "NOT DATA-LIMITED" or tag == "up1":
            assert dt["mse_delta"]["separated"] is True, "a uniform 1 % shift of every window is still a separated paired delta -- it is the SIZE bar that keeps +1 % out"
        assert set(r["rows"]) == {"s0"}, "seed 0 only"
    # the table is a commitment about SIZE and SEPARATION: a Q that is the same as B reads NOT DATA-LIMITED with a delta that is not separated
    rc, r = _go(_dworld(tmp_path / "flat", err_q=0.6))
    assert rc == 0 and r["DECISION"] == "NOT DATA-LIMITED" and r["rows"]["s0"]["dt"]["mse_delta"]["separated"] is False


@needs_pkg
@pytest.mark.skipif(not PREREG.exists(), reason="PREREG_V8F_DSCALE.md is not in this checkout")
def test_reader_the_committed_next_steps_and_bars_are_verbatim_the_preregs():
    text = PREREG.read_text(encoding="utf-8")
    rows = {ln.split("|")[1].strip().strip("*"): [c.strip() for c in ln.strip().strip("|").split("|")] for ln in text.splitlines() if ln.startswith("| **")}
    assert set(rows) == {"DATA-LIMITED", "NOT DATA-LIMITED", "EPOCH-LIMITED"} and set(R.NEXT_TEXT) == set(rows)
    for k, cells in rows.items():
        assert R.NEXT_TEXT[k] == cells[2], f"{k}: the next step is the PREREG's cell, character for character"
        assert R.NEXT_TEXT[k] in R.NEXT[k]
    assert rows["DATA-LIMITED"][1] == "r ≥ +5 %, MSE Δ separated" and rows["EPOCH-LIMITED"][1] == "r ≤ −5 %, MSE Δ separated" and rows["NOT DATA-LIMITED"][1] == "−5 % < r < +5 %, or the MSE Δ not separated"
    assert "B = 2,000" in text and R.B == 2000 and "19,164" in text and R.STEPS_B == 19164 and "round(0.5 × N_B)" in text and "`--data-frac 0.5`" in text
    assert "**Seed 0 only.**" in text and "TRAINING constraint loss over the last 10 % of steps" in text and R.TAIL == 0.10
    assert "T's slot-pooled present-slot latcons RMSE" in text and "HIST's M-LAT" in text


@needs_pkg
def test_reader_q_against_itself_reads_exactly_zero_on_every_contrast(tmp_path, no_known):
    """The identity control at the level of the whole read: Q's evals ARE B's arrays, so every paired cell reads exactly +0 [0, 0], r = 0 and the outcome is NOT DATA-LIMITED."""
    rc, r = _go(_dworld(tmp_path / "same", same_as_b=True))
    assert rc == 0 and r["DECISION"] == "NOT DATA-LIMITED"
    zero = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "point": 0.0}
    row = r["rows"]["s0"]
    dt = row["dt"]
    assert dt["r"] == 0.0 and dt["r_window"] == 0.0 and {k: dt["mse_delta"][k] for k in zero} == zero and dt["mse_delta"]["separated"] is False
    assert {k: row["primary"][k] for k in zero} == zero and {k: row["mleak"]["delta"][k] for k in zero} == zero and {k: row["mdk"]["delta"][k] for k in zero} == zero
    be = row["beside"]
    assert {k: be["oracle_ade6"]["delta"][k] for k in zero} == zero and {k: be["present_slot"]["mse_delta"][k] for k in zero} == zero
    for key in ("timing4", "geometry8"):
        assert {k: be["latcons_slots"]["sets"][key]["mse_delta"][k] for k in zero} == zero and be["latcons_slots"]["sets"][key]["fall"] == 0.0
    for name, g in be["groups"].items():
        assert g["r"] == 0.0 and {k: g["mse_delta"][k] for k in zero} == zero, name
    ic = r["identity_control"]["s0"]
    assert ic["dt"] == {"r": 0.0, "r_window": 0.0, "outcome": "NOT DATA-LIMITED"} and {k: ic["mlat_mse"][k] for k in zero} == zero


@needs_pkg
def test_reader_f_is_b_s_own_seed_floor_and_the_ade6_read_decides_nothing(tmp_path, no_known):
    f = _dworld(tmp_path / "w", d_ade=0.9, err_q=0.6)
    rc, r = _go(f)
    ev = f["eval"]
    d = np.load(ev / "eval_M2b_s0.npz")["ade6"].mean() - np.load(ev / "eval_M2b_s1.npz")["ade6"].mean()
    assert rc == 0 and r["F"] == pytest.approx(abs(d), abs=1e-12) and r["two_F"] == pytest.approx(2 * r["F"])
    ad = r["rows"]["s0"]["beside"]["ade6"]
    assert ad["Q"] - ad["B"] == pytest.approx(0.9, abs=1e-3) and ad["delta"]["separated"] and ad["delta"]["point"] > 2 * r["F"]
    assert ad["separated_and_beyond_2F_if_it_were_deciding"] is False, "a WORSE ADE6 is not a pass of anything, and it decides nothing"
    assert r["DECISION"] == "NOT DATA-LIMITED", "ADE6 does not enter D-T"
    rc, r2 = _go(_dworld(tmp_path / "w2", d_ade=-0.9, err_q=0.6 * 1.06, sp_q=10.0))             # a Q that plans the GT speed can be 0.9 m better: it is still DATA-LIMITED on T
    assert rc == 0 and r2["DECISION"] == "DATA-LIMITED" and r2["rows"]["s0"]["beside"]["ade6"]["separated_and_beyond_2F_if_it_were_deciding"] is True


@needs_pkg
def test_reader_reports_the_beside_reads_and_the_training_loss_tail(tmp_path, no_known):
    f = _dworld(tmp_path / "w", err_q=0.6 * 1.06, cons_q=0.8, cons_q_head=3.0)
    rc, r = _go(f)
    assert rc == 0
    row = r["rows"]["s0"]
    be = row["beside"]
    # T's TRAINING constraint loss over the last 10 % of steps, Q against B: only the tail enters (the head was scaled 3x and must not)
    tc = be["train_constraint_loss"]
    assert tc["Q"]["status"] == tc["B"]["status"] == "OK" and tc["Q"]["n_records"] == tc["B"]["n_records"] > 30 and tc["Q"]["first_step"] > 0.9 * 19164
    assert tc["Q_over_B_minus_1"] == pytest.approx(-0.2, abs=1e-9) and tc["Q"]["cons_mean"] == pytest.approx(0.8 * tc["B"]["cons_mean"])
    assert "last 10 %" in tc["window"]
    sig = be["generalisation_signature"]
    assert sig["train_cons_fell"] is True and sig["eval_rmse_rose"] is True, "a generalisation-limited T fits its smaller set better at worse eval"
    # per group (latcons, lonspeed, stoplead) and per latcons slot
    assert set(be["groups"]) == {"latcons", "lonspeed", "stoplead"} and all(g["r"] == pytest.approx(0.06, abs=1e-9) for g in be["groups"].values())
    assert set(be["present_slot"]["rmse_group"]["B"]) == {"latcons", "lonspeed", "stoplead"}
    sl = be["latcons_slots"]
    assert list(sl["slots"]) == list(D.LAT_CONSTRAINTS) and len(sl["slots"]) == 12 and all(set(v["rmse"]) == {"B", "Q"} for v in sl["slots"].values())
    assert all(v["fall"] == pytest.approx(-0.06, abs=1e-9) for v in sl["slots"].values() if v["fall"] is not None)
    # the leak LAMBDA, the GT-code oracle ADE6, parameters, pipeline, data facts
    assert row["mleak"]["lambda_b_m"] > 0 and be["oracle_ade6"]["B"] == pytest.approx(be["oracle_ade6"]["Q"])
    assert be["parameters"]["n_params_trainable"] == {"B": BANKED_M2B_TOTAL, "Q": BANKED_M2B_TOTAL}
    assert be["pipeline"]["Q"]["steps"] == be["pipeline"]["B"]["steps"] == STEPS_B and be["pipeline"]["Q"]["n_windows"] == _syn()["windows_q"] < be["pipeline"]["B"]["n_windows"] == WINDOWS_B
    assert be["built"] == {"n_clips": N_HALF, "n_clips_all": N_B, "data_frac": 0.5, "first_sha12": D.sha12(_syn()["half"][0]), "last_sha12": D.sha12(_syn()["half"][-1]),
                           "train_sha256": _syn()["half_sha"], "hist": 4, "hmax": 16, "window_start": -1}
    assert be["data"]["n_clips"] == {"B": N_B, "Q": N_HALF} and be["data"]["steps"] == {"B": STEPS_B, "Q": STEPS_B}
    assert r["strategic_family"].startswith("NOT PAUSED") and r["tier"].startswith("T1-class open loop") and "answers 'another draw of EPISODES' only" in r["estimator"] and r["estimator"].count("B=300") == 1
    assert r["manifest"]["n_half"] == N_HALF and r["manifest"]["half_sha256"] == _syn()["half_sha"] and "half" not in r["manifest"]
    assert "ADE6 Q - B" in r["reading"] and f"{N_HALF} clips" in r["reading"] and "r +6.00%" in r["reading"]
    # a banked B without its train log: the beside read says so, nothing is invented, and the verdict stands
    (f["runs"] / "M2b_s0" / "train_log.jsonl").unlink()
    rc, r3 = _go(f)
    assert rc == 0 and r3["rows"]["s0"]["beside"]["train_constraint_loss"]["B"]["status"] == "NOT-AVAILABLE" and "Q_over_B_minus_1" not in r3["rows"]["s0"]["beside"]["train_constraint_loss"]
    assert r3["DECISION"] == r["DECISION"]


# ---- the refusals: every precondition of a Q-vs-B read is a refusal, never a quiet pass ---------------------------------------------------------
def _qr(f):
    return f["runs"] / "M2b_s0_df50"


def _br(f, s=0):
    return f["runs"] / f"M2b_s{s}"


def _qev(f, suf=""):
    return f["eval"] / f"eval_M2b-df50_s0{suf}"


def _argv_edit(path, fn):
    def go(j):
        j["argv"] = fn(list(j["argv"]))
    _ej(path, go)


def _set_flag(argv, flag, val):
    argv[argv.index(flag) + 1] = val
    return argv


def _ck_edit(f, fn):
    p = _qr(f) / "ckpt_last.pt"
    o = torch.load(p, weights_only=False)
    fn(o)
    torch.save(o, p)


def _lev(path, fn):
    def go(j):
        fn(j["levers"])
    _ej(path, go)


def _info(path, **kv):
    _ej(path, lambda j: j["info"].update(kv))


def _npz_edit(path, fn):
    z = dict(np.load(path))
    fn(z)
    np.savez(path, **z)


def _log_edit(f, fn):
    p = _qr(f) / "train_log.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    fn(rows)
    p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def _shift_eid(z):
    z["eid"] = np.asarray(z["eid"]).copy()
    z["eid"][0] += 1


def _gt_codes_off(z):
    z["tac_code"] = np.asarray(z["tac_code"]).copy()
    z["tac_code"][10:40, C0:] += 1.0


def _not_the_latcons_oracle(z):
    z["tac_code"] = np.asarray(z["tac_code"]).copy()
    z["tac_code"][10:40, C0 + 12:] += 1.0


def _fewer_steps(f, n):
    """A run that stopped at `n` steps, with its own consistent log (so the step-count check, not the log check, is what must refuse)."""
    _ej(_qr(f) / "summary.json", lambda j: j.update(steps=n))
    _log_edit(f, lambda r: r.__setitem__(slice(None), [dict(x, p_gt=1.0 + (0.2 - 1.0) * min(max((x["step"] - 1) / n, 0.0), 1.0)) for x in r if x["step"] <= n]))


def _del_w(key):
    def go(l):
        l["w"].pop(key)
    return go


MUT = {                                # name -> (mutator(f), a message fragment)
    "q summary missing": (lambda f: (_qr(f) / "summary.json").unlink(), "is missing"),
    "q checkpoint missing": (lambda f: (_qr(f) / "ckpt_last.pt").unlink(), "is missing"),
    "q train log missing": (lambda f: (_qr(f) / "train_log.jsonl").unlink(), "is missing"),
    "q not done": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(done=False)), "is not done"),
    "q is another seed": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(seed=1)), "expected M2b seed 0"),
    "q withheld nothing": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(excluded_sha256="0" * 64)), "did not withhold"),
    # ---- the argv: B's plus --data-frac 0.5
    "q argv has no --data-frac": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: [x for x in a if x not in ("--data-frac", "0.5")]), "argv --data-frac None, expected 0.5"),
    "q argv says --data-frac 0.25": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: _set_flag(a, "--data-frac", "0.25")), "argv --data-frac 0.25, expected 0.5"),
    "q argv says --data-frac 1.0": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: _set_flag(a, "--data-frac", "1.0")), "argv --data-frac 1.0, expected 0.5"),
    "q argv has a second lever": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: a + ["--w-wm", "0.2"]), "not one lever"),
    "q argv moved the steps": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: a + ["--steps", "9582"]), "not one lever"),
    "q argv moved the grid": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: _set_flag(a, "--grid", "pool8x20")), "not one lever"),
    "q argv moved the history": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: a + ["--hist", "10", "--window-start", "3"]), "not one lever"),
    "q argv lost a flag": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: [x for x in a if x not in ("--kin-mode", "yaw")]), "not one lever"),
    "q argv is another arm": (lambda f: _argv_edit(_qr(f) / "argv.json", lambda a: _set_flag(a, "--arm", "M2")), "argv --arm"),
    "b argv names --data-frac 0.5": (lambda f: _argv_edit(_br(f) / "argv.json", lambda a: a + ["--data-frac", "0.5"]), "B is the full-data run"),
    "b summary says data_frac 0.5": (lambda f: _ej(_br(f) / "summary.json", lambda j: j.update(data_frac=0.5)), "B is not a full-data run"),
    # ---- the data: round(0.5 x N_B) clips, the FIRST half of B's list in sha12 order
    "q summary says data_frac 1.0": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(data_frac=1.0)), "summary.json data_frac 1.0"),
    "q trained the whole list (4512 clips)": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(n_clips=4512)), "trained on 4512 clips, round(0.5 x N_B"),
    "q trained 2255 clips": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(n_clips=2255)), "trained on 2255 clips"),
    "q trained 2257 clips": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(n_clips=2257)), "trained on 2257 clips"),
    "q selfcheck built the whole list": (lambda f: _info(_qr(f) / "selfcheck.json", n_clips=4512), "selfcheck.json built 4512 clips"),
    "q selfcheck n_clips_all is not N_B": (lambda f: _info(_qr(f) / "selfcheck.json", n_clips_all=4000), "selfcheck.json built"),
    "q selfcheck declared another count": (lambda f: _info(_qr(f) / "selfcheck.json", n_clips_expected=2000), "selfcheck.json built"),
    "q clips are another list": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(train_sha256="u" * 64)), "not the first 2256 of B's list in sha12 order"),
    "q clips are the LAST half": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(train_sha256=_syn()["last_half_sha"])), "not the first 2256 of B's list in sha12 order"),
    "q clips are B's whole list": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(train_sha256=_syn()["b_sha"])), "trained on B's whole clip list"),
    "q selfcheck train list differs": (lambda f: _info(_qr(f) / "selfcheck.json", train_sha256="u" * 64), "not the first 2256 of B's list in sha12 order"),
    # ---- the budget: B's 19,164 (S counts ALL clips)
    "q trained fewer steps": (lambda f: _fewer_steps(f, 9000), "trained for 9000 steps"),
    "q trained half the steps (S sized from the half)": (lambda f: _fewer_steps(f, 9582), "trained for 9582 steps, M2b_s0 for 19164"),
    "q S differs": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(S=2396)), "S 2396"),
    "q steps-mult differs": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(steps_mult=2)), "steps-mult 2"),
    "q budget sized from the half's windows": (lambda f: _info(_qr(f) / "selfcheck.json", n_windows_all_clips=_syn()["windows_q"]), "the budget was sized from"),
    "q selfcheck steps_total is the half's": (lambda f: _ej(_qr(f) / "selfcheck.json", lambda j: j.update(steps_total=9582)), "steps_total 9582"),
    "q trained on all of B's windows": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(n_windows=WINDOWS_B)), "trained on 306566 windows, the half's clips give"),
    "q trained on another number of windows": (lambda f: _ej(_qr(f) / "summary.json", lambda j: j.update(n_windows=_syn()["windows_q"] + 1)), "the half's clips give"),
    # ---- the launch gate
    "q launch gate not PASS": (lambda f: _ej(_qr(f) / "selfcheck.json", lambda j: j.update(declared_vs_built="FAIL")), "declared_vs_built is not PASS"),
    "q G-LIVE not pass": (lambda f: _ej(_qr(f) / "selfcheck.json", lambda j: j["live"].update(**{"pass": False})), "G-LIVE is not a pass"),
    "q frozen-prior control not exactly zero": (lambda f: _ej(_qr(f) / "selfcheck.json", lambda j: j["live"].update(control_prior_grad_sum=1e-9)), "G-LIVE is not a pass"),
    "q selfcheck from another run": (lambda f: _ej(_qr(f) / "selfcheck.json", lambda j: j.update(argv_sha="zzz")), "is not this run's"),
    # ---- the model and the losses are B's
    "q built another weight": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l["w"].update(w_pen=0.9)), "w.w_pen"),
    "q built another w_tac": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l["w"].update(w_tac=2.0)), "w.w_tac"),
    "q built another w_cons": (lambda f: _lev(_qr(f) / "selfcheck.json", lambda l: l["w"].update(w_cons=1.0)), "w.w_cons"),
    "q lost a weight key B recorded": (lambda f: _lev(_qr(f) / "summary.json", _del_w("w_pen_self")), "w.w_pen_self"),
    "q built another mask": (lambda f: _lev(_qr(f) / "selfcheck.json", lambda l: l.update(wm="a")), "built levers"),
    "q has one parameter too many": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l.update(n_params_trainable=l["n_params_trainable"] + 1)), "parameter count is not B's"),
    "q adapter group differs": (lambda f: _lev(_qr(f) / "selfcheck.json", lambda l: l["params"].update(adapter=l["params"]["adapter"] + 256)), "parameter count is not B's"),
    "q grew a second group": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l["params"].update(T=l["params"]["T"] + 5)), "group T"),
    "q built hist 10": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l.update(hist=10)), "built hist 10, expected 4"),
    "q records no built hist": (lambda f: _lev(_qr(f) / "selfcheck.json", lambda l: l.pop("hist")), "record no built hist"),
    "q built hmax 10": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l.update(hmax=10)), "built hmax 10, expected 16"),
    "q built window_start 9": (lambda f: _lev(_qr(f) / "selfcheck.json", lambda l: l.update(window_start=9)), "built window_start 9, expected 3"),
    "q built another p_gt": (lambda f: _lev(_qr(f) / "summary.json", lambda l: l.update(p_gt_start=0.7)), "was built with p_gt"),
    "q records no built p_gt": (lambda f: _lev(_qr(f) / "selfcheck.json", lambda l: l.pop("p_gt_end")), "record no built p_gt"),
    "q checkpoint built at hist 10": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(hist=10)), "BUILT at hist 10"),
    "q checkpoint time table has 10 rows": (lambda f: _ck_edit(f, lambda o: o["model"].update(time=torch.zeros(10, 256))), "10-row time table"),
    "q checkpoint built another hmax": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(hmax=10)), "BUILT with hmax 10, B's is 16"),
    "q checkpoint built window_start 3": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(window_start=3)), "BUILT with window_start 3"),
    "q checkpoint is an 8x20 checkpoint": (lambda f: _ck_edit(f, lambda o: o.update(grid="pool8x20")), "are not B's"),
    "q checkpoint from another argv": (lambda f: _ck_edit(f, lambda o: o.update(argv_sha="other")), "another argv"),
    "q checkpoint at another step": (lambda f: _ck_edit(f, lambda o: o.update(step=100)), "checkpoint is at step 100"),
    "q checkpoint of another seed": (lambda f: _ck_edit(f, lambda o: o.update(seed=1)), "checkpoint says arm"),
    "q train log carries the wrong p_gt": (lambda f: _log_edit(f, lambda r: r[3].update(p_gt=0.99)), "p_gt"),
    # ---- the evals
    "q eval scored B's checkpoint": (lambda f: _ej(_qev(f, ".json"), lambda j: j.update(ckpt=str(_br(f) / "ckpt_last.pt"))), "scored"),
    "q oracle eval scored no checkpoint": (lambda f: _ej(_qev(f, "-tacorc.json"), lambda j: j.pop("ckpt") and None), "scored None"),
    "q eval is on another eval grid": (lambda f: _npz_edit(_qev(f, ".npz"), _shift_eid), "eid"),
    "q evals are on another eval grid than B's": (lambda f: [_npz_edit(_qev(f, suf), _shift_eid) for suf in (".npz", "-tacabs.npz", "-tacorc.npz", "-tacorc-all-latcons.npz")], "not on the same eval grid as M2b's"),
    "q oracle eval has other GT codes": (lambda f: _npz_edit(_qev(f, "-tacorc.npz"), _gt_codes_off), "the GT constraint block differs"),
    "q all-latcons oracle is not GT outside latcons": (lambda f: _npz_edit(_qev(f, "-tacorc-all-latcons.npz"), _not_the_latcons_oracle), "not the all-latcons oracle"),
    "q eval missing": (lambda f: _qev(f, "-tacorc-all-latcons.npz").unlink(), "is missing"),
}


@needs_pkg
@pytest.mark.parametrize("name", list(MUT))
def test_reader_refuses_rc2_with_the_reason(tmp_path, no_known, capsys, name):
    mut, frag = MUT[name]
    f = _dworld(tmp_path / "w")
    mut(f)
    capsys.readouterr()
    rc = R.main(f["args"])
    err = capsys.readouterr().err
    assert rc == 2 and frag in err, f"{name}: rc {rc}; stderr: {err[-400:]}"
    assert not (f["d"] / "dscale.json").exists(), "a refused read must leave no result file for the next step to find"


@needs_pkg
def test_reader_the_world_without_a_mutation_reads_clean(tmp_path, no_known, capsys):
    """The control of the refusal table: the very same world, unmutated, is read (rc 0) -- so each refusal above is the mutation's, not the world's."""
    assert R.main(_dworld(tmp_path / "w")["args"]) == 0


@needs_pkg
def test_reader_refuses_a_manifest_that_is_not_b_s_list(tmp_path, no_known, capsys):
    syn = _syn()
    variants = {
        "a clip dropped": (lambda ls: ls[:100] + ls[101:], "not the list B trained on"),
        "a clip renamed": (lambda ls: [ls[0].replace("syn-clip-", "syn-clop-")] + ls[1:], "not the list B trained on"),
        "frame counts that do not give B's windows": (lambda ls: [json.dumps({"clip": json.loads(x)["clip"], "n_frames": json.loads(x)["n_frames"] + 1}) + "\n" for x in ls], "windows for B"),
    }
    for name, (fn, frag) in variants.items():
        f = _dworld(tmp_path / name.replace(" ", "_"))
        lines = syn["lines"].splitlines(keepends=True)
        (f["d"] / "manifest_train.jsonl").write_text("".join(fn(lines)), encoding="utf-8")
        capsys.readouterr()
        rc = R.main(f["args"])
        err = capsys.readouterr().err
        assert rc == 2 and frag in err, (name, rc, err[-300:])
    f = _dworld(tmp_path / "empty")
    (f["d"] / "manifest_train.jsonl").write_text("", encoding="utf-8")
    capsys.readouterr()
    assert R.main(f["args"]) == 2 and "is empty" in capsys.readouterr().err
    f = _dworld(tmp_path / "gone")
    (f["d"] / "manifest_train.jsonl").unlink()
    capsys.readouterr()
    assert R.main(f["args"]) == 2 and "cannot read the train manifest" in capsys.readouterr().err


@needs_pkg
def test_reader_the_manifest_without_frame_counts_still_checks_the_clips_and_the_window_ordering(tmp_path, no_known, capsys):
    """A manifest of plain ids (no n_frames) cannot give window counts: the list identity is still checked exactly, and the half's windows are only required to be fewer than B's."""
    syn = _syn()
    ids_only = "".join(json.dumps({"clip": c}) + chr(10) for c in syn["ids"])
    p = tmp_path / "ids.jsonl"
    p.write_text(ids_only, encoding="utf-8")
    parsed = R.read_manifest(p)
    exp = R.expected_from_manifest(parsed, "540:600")
    assert exp["windows_b"] is None and exp["windows_q"] is None and exp["frame_counts"] is False and exp["half_sha256"] == syn["half_sha"] and exp["b_sha256"] == syn["b_sha"]
    f = _dworld(tmp_path / "w")
    (f["d"] / "manifest_train.jsonl").write_text(ids_only, encoding="utf-8")
    assert R.main(f["args"]) == 0, "the clip list is proven exactly; the window count is only bounded"
    _ej(_qr(f) / "summary.json", lambda j: j.update(n_windows=WINDOWS_B))
    capsys.readouterr()
    assert R.main(f["args"]) == 2 and "a half of the clips must give fewer" in capsys.readouterr().err


@needs_pkg
def test_reader_missing_inputs_are_rc2_not_a_traceback(tmp_path, no_known, capsys):
    assert R.main(["--eval-dir", str(tmp_path), "--runs-dir", str(tmp_path), "--out", str(tmp_path / "o.json"), "--train-manifest", str(tmp_path / "m.jsonl")]) == 2
    assert "is missing" in capsys.readouterr().err
    assert R.main(["--eval-dir", str(tmp_path), "--runs-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 2, "argparse: the manifest is required"
    assert R.main([]) == 2


@needs_pkg
def test_reader_refuses_a_b_that_is_not_the_banked_m2b(tmp_path, no_known, capsys):
    f = _dworld(tmp_path / "w")
    _argv_edit(_br(f) / "argv.json", lambda a: a + ["--p-gt-start", "0.5", "--p-gt-end", "0.0"])
    capsys.readouterr()
    assert R.main(f["args"]) == 2 and "REFUSED" in capsys.readouterr().err
    f = _dworld(tmp_path / "w2")
    _ej(_br(f, 1) / "summary.json", lambda j: j.update(steps=9582))
    capsys.readouterr()
    assert R.main(f["args"]) == 2 and "REFUSED" in capsys.readouterr().err


@needs_pkg
def test_reader_check_inputs_reads_b_alone_and_the_identity_control_is_exactly_zero(tmp_path, no_known, capsys):
    f = _dworld(tmp_path / "c", with_q=False)                                  # no Q run exists yet: this is the check that runs BEFORE training
    assert R.main(["--check-inputs", *f["args"]]) == 0
    out = capsys.readouterr().out
    assert "DSCALE INPUTS OK" in out and "identity control exactly +0 [0, 0]" in out and "b_hist 4" in out and "b_hmax 16" in out
    assert f"Q must be {N_HALF} clips" in out and D.sha12(_syn()["half"][0]) in out and D.sha12(_syn()["half"][-1]) in out
    j = json.loads((f["d"] / "dscale.json").read_text(encoding="utf-8"))
    assert j["base"] == "M2b" and j["F"] > 0 and j["b_hmax"] == 16 and j["b_hist"] == 4 and j["b_steps"] == {"0": STEPS_B, "1": STEPS_B} and j["b_n_windows"] == {"0": WINDOWS_B, "1": WINDOWS_B}
    eq = j["expected_q"]
    assert (eq["n_clips"], eq["n_clips_all"], eq["steps"], eq["S"], eq["n_windows_all_clips"], eq["n_windows_q"]) == (N_HALF, N_B, STEPS_B, S_B, WINDOWS_B, _syn()["windows_q"])
    assert eq["train_sha256"] == _syn()["half_sha"] and eq["first_sha12"] == D.sha12(_syn()["half"][0]) and eq["last_sha12"] == D.sha12(_syn()["half"][-1])
    zero = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "point": 0.0}
    for s in ("s0", "s1"):
        cells = j["identity_control"][s]
        assert cells["mlat_mse"] == zero and cells["slots_timing4_mse"] == zero and cells["slots_geometry8_mse"] == zero
        assert cells["dt"] == {"r": 0.0, "r_window": 0.0, "outcome": "NOT DATA-LIMITED"} and cells["latcons_rmse_slot"] > 0
    assert j["window_counts"]["asserted_against_prereg"] is False                          # the synthetic grid is 72 windows
    assert j["b_train_constraint_loss_tail"]["status"] == "OK" and j["b_train_constraint_loss_tail"]["n_records"] > 30
    # and the manifest that is not B's list is refused BEFORE any training
    (f["d"] / "manifest_train.jsonl").write_text(_syn()["lines"].replace("syn-clip-00007", "syn-clop-00007"), encoding="utf-8")
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2 and "not the list B trained on" in capsys.readouterr().err


@needs_pkg
def test_reader_check_inputs_breaks_loudly_when_the_paired_instrument_is_broken(tmp_path, no_known, capsys, monkeypatch):
    """The identity control has teeth: a paired estimator that does not return exactly 0 on B-against-B stops the check (and the read)."""
    f = _dworld(tmp_path / "c", with_q=False)
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        if np.array_equal(np.asarray(a), np.asarray(b)):
            o = dict(o, delta=o["delta"] + 1e-9)
        return o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


@needs_pkg
def test_reader_check_inputs_stops_when_d_t_itself_does_not_read_exactly_zero_on_b_against_b(tmp_path, no_known, capsys, monkeypatch):
    """The identity control reaches D-T too: if the decision rule called a B-against-B contrast DATA-LIMITED (r = 0 exactly), the check refuses naming D-T, and the read never starts."""
    f = _dworld(tmp_path / "d", with_q=False)
    monkeypatch.setattr(R, "decide", lambda r, d, bar=R.BAR: "DATA-LIMITED")
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    err = capsys.readouterr().err
    assert "IDENTITY CONTROL FAILED" in err and "D-T" in err and "NOT DATA-LIMITED" in err


@needs_pkg
def test_reader_turns_an_inherited_refusal_into_rc2_not_a_traceback(tmp_path, no_known, capsys):
    """v8f_dk_read's own `Refusal` (raised by the lead-block join when a window has no row) is not a SystemExit: the reader must still exit rc 2 with the reason."""
    f = _dworld(tmp_path / "w")

    def odd(z):
        z["frame"] = np.asarray(z["frame"]).copy()
        z["frame"][0] += 1
    _npz_edit(f["d"] / "gt.npz", odd)
    capsys.readouterr()
    assert R.main(f["args"]) == 2
    err = capsys.readouterr().err
    assert "DSCALE READ REFUSED" in err and "no row in the lead block" in err


@needs_pkg
def test_reader_known_values_are_checked_and_refuse_a_world_that_does_not_hold_them(tmp_path, syn_list, capsys):
    """The PREREG's premises (F 0.1207, LAMBDA, S5 excess) are the banked M2b evals' values: a synthetic world cannot reproduce them, so the real check REFUSES it."""
    f = _dworld(tmp_path / "m", with_q=False)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "known values" in capsys.readouterr().err


@needs_pkg
def test_reader_constants_match_the_trainer_the_data_module_and_the_hist_reader():
    assert R.sha12("syn-clip-00001") == D.sha12("syn-clip-00001") and R.HORIZON == D.HORIZON and R.B_START == D.effective_window_start(R.B_HIST, None)
    assert R.DF_FLAG == "--data-frac" and any(a.dest == "data_frac" and R.DF_FLAG in a.option_strings for a in TR.build_parser()._actions)
    assert R.half_n(4512) == 2256 == D.select_train_clips([f"c{i}" for i in range(4512)], R.FRAC).__len__() and R.half_n(6) == 3 and R.half_n(1, 0.25) == 1 and R.half_n(100, 1.0) == 100
    assert R.half_n(6, 0.25) == 2 == D.select_train_clips([f"c{i}" for i in range(6)], 0.25).__len__(), "round, not floor: 6 x 0.25 = 1.5 -> 2 in the trainer and in the reader"
    assert R.PGT.N_TRAIN == N_B and R.STEPS_B == STEPS_B and R.PGT.M2B_START == 1.0 and R.PGT.M2B_END == 0.2 and R.HR.B_GRID == "pool4x10" and R.HR.N_TOK_B == 40
    assert R.SUF == "_df50" and R.ARM_Q == "M2b-df50" and R.q_dir(0) == "M2b_s0_df50" and R.EVAL_SUFFIXES == ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
    assert R.w_diff is R.GR.w_diff and R.B_UNRECORDED_W is R.GR.B_UNRECORDED_W and R.HR.PGT is R.PGT and R.GR.PGT is R.PGT, "one set of IMPORTED instruments, never copies"
    real, _ = _b_real(0)
    assert real["summary.json"]["steps"] == STEPS_B and real["summary.json"]["S"] == S_B and real["summary.json"]["n_windows"] == WINDOWS_B and real["summary.json"]["n_clips"] == N_B
    assert R.windows_of([("a", 100), ("b", 40)], ["a"]) == 67 and R.windows_of([("a", 100), ("b", 33)], ["b"]) == 0 and R.windows_of([("a", 100), ("b", 40)], ["b"]) == 7 and R.windows_of([("a", None)], ["a"]) is None


@needs_pkg
def test_reader_every_refusal_of_the_prereg_is_covered_by_a_case_above():
    """The PREREG's five refusals, each mapped to the cases of MUT that exercise it (the mapping is checked, so a refusal cannot lose its test silently)."""
    text = PREREG.read_text(encoding="utf-8") if PREREG.exists() else ""
    cases = {"argv": ("q argv has no --data-frac", "q argv has a second lever", "b argv names --data-frac 0.5"),
             "clips": ("q trained the whole list (4512 clips)", "q trained 2255 clips", "q clips are another list", "q clips are the LAST half"),
             "steps": ("q trained half the steps (S sized from the half)", "q trained fewer steps"),
             "model": ("q has one parameter too many", "q built another weight", "q built another w_cons"),
             "evals": ("q evals are on another eval grid than B's", "q oracle eval has other GT codes")}
    for what, names in cases.items():
        assert all(n in MUT for n in names), what
    if text:
        assert "Q's argv is not B's plus `--data-frac 0.5`" in text and "The step count is not B's 19,164" in text and "Evals are not on one grid, or the GT codes differ" in text
        assert "A parameter count or any loss weight (`w_diff`) differs from B's" in text and "or the clips are not the first half of B's list in sha12 order" in text


# ---- module-level guard (the lander imports test files from a stack-only tree) ------------------------------------------------------------------------
def test_reach_module_level_loads_of_the_package_are_guarded():
    """Every module-level load of a research-package file sits behind an existence guard, so importing THIS file in a tree that has only `stack/` cannot raise."""
    src = Path(__file__).read_text(encoding="utf-8")
    for name in ("R = _load(\"v8f_dscale_read_under_test\", READ_P) if READ_P.exists() else None",
                 "_DK = _load(\"test_v8f_dk_for_dscale\", Path(__file__).resolve().parent / \"test_v8f_dk.py\") if READ_P.exists() else None"):
        assert name in src, name
    top = [ln for ln in src.splitlines() if ln.startswith(("R = ", "_DK = ", "CC = "))]
    assert top and all(" if " in ln and ln.split("  #")[0].rstrip().endswith("else None") for ln in top), top


# =================================================================================================================================================
# THE QUEUE STAGES (code/v8f_mini_box2_queue.cmd: gatedscale / traindscale / evaldscale): what is PRINTED is what RUNS, one lever
# =================================================================================================================================================
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
DSTAGES = ("gatedscale", "traindscale", "evaldscale")
# every label of the queue as it stood at tip 343f366d, with the sha256[:16] of its text ("\n" + the lines up to the next label): the DSCALE edit may ADD stages and extend the usage line, never touch these
PRE_EXISTING = {
    "d_train": "653a25ac6e43be79", "d_eval": "b64a718bcf22e4ca", "prep": "8f8749398e785a74", "gate": "01fa4b4d415d3057", "train": "c3867e94dae8b06b", "eval": "8d3f96a2ec519c6d", "tac": "8357d58b9565d87f",
    "tacone": "d05a9be75c4de2bd", "tacgrp": "7cae083fcfb5a237", "tacgone": "cdbb6e94327d04af", "tacgrp2": "efb5d1a2944b99fa", "tacgtwo": "14ccef2e21c86f83", "tacabs": "60aa7130ee682938",
    "tacabsarm": "0ed52c566b9de578", "gatewc": "487c836142548e93", "trainwc": "5565fa65d285f47f", "evalwc": "ccb014965cc92aa5", "gatepgt": "3ca89871f8da1c43", "trainpgt": "49fe41ddd1622379",
    "evalpgt": "a04620ebdcfbb068", "evalpgt_1": "effe5bbee55d4771", "evalpgt_2": "33cdd72c074d8c6a", "evalpgt_3": "a7e30c81d582343c", "evalpgt_4": "22636d9d007850fa", "gridbase": "56299fbe24cec6f7",
    "gategrid": "e3250ab756539587", "traingrid": "87cf143655b55573", "evalgrid": "399406480951bc1d", "evalgrid_1": "74aaef5e39bbaca5", "evalgrid_2": "f242aa25ecd51bd6", "evalgrid_3": "210db6bc5c411a0c",
    "evalgrid_4": "22636d9d007850fa", "gatehist": "93a341e35cc9a0b1", "trainhist": "344fcc45b1624ead", "evalhist": "413108b37c6fe068", "evalhist_1": "03081f7da50e8739", "evalhist_2": "168db150abcf3ce1",
    "evalhist_3": "d9d586a4304f2d24", "evalhist_4": "22636d9d007850fa", "gatearm": "b79d2a733b86fe61", "floors": "2b7f345ee3e86bcc", "compare": "b41e1a1c859c59c3"}


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


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


def _without(line, drop):
    t, out, i = _argv_of(line), [], 0
    while i < len(t):
        if t[i] in drop:
            i += 2
            continue
        out.append(t[i])
        i += 1
    return out


@needs_queue
def test_queue_dscale_the_queue_is_crlf_and_the_stages_are_dispatched():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 500, "a .cmd with bare LF endings misparses labels and `exit /b`"
    assert raw.count(b"\r") == raw.count(b"\r\n"), "no bare CR"
    q = _queue_text()
    for st in DSTAGES:
        assert f'if /i "%~1"=="{st}" goto :{st}' in q, "a stage nobody dispatches never runs"
        assert q.count(f"\n:{st}\n") == 1
    assert "^|gatehist^|trainhist SEED^|evalhist SEED^|gatedscale^|traindscale SEED^|evaldscale SEED" in q, "the usage line names the new stages (HIST's still first)"
    for label in DSTAGES:
        body = "\n".join(_code(_block(label)))
        for no in ("p-gt", "8x20", "--w-", "--hmax", "--hist", "--window-start", "--steps ", "--max-clips", "h10"):
            assert no not in body, f"{label}: the lever is --data-frac 0.5 and nothing else ({no!r})"


@needs_queue
def test_queue_dscale_every_existing_stage_is_byte_identical():
    """The edit only ADDS stages: the text of every label that existed at tip 343f366d hashes as it did, the dispatch of each old stage is still there, and none mentions the new lever."""
    q = _queue_text()
    parts = re.split(r"(?m)^(:[A-Za-z0-9_]+)$", q)
    blocks = {parts[i][1:]: parts[i + 1] for i in range(1, len(parts), 2)}
    for label, h in PRE_EXISTING.items():
        assert label in blocks, f"the stage {label} was removed"
        assert hashlib.sha256(("\n" + blocks[label]).encode()).hexdigest()[:16] == h or hashlib.sha256(blocks[label].encode()).hexdigest()[:16] == h, f"the stage {label} was edited"
        assert "data-frac" not in blocks[label] and "df50" not in blocks[label], label
    for st in ("gate", "train", "eval", "floors", "compare", "tac", "tacgrp", "tacgrp2", "tacabs", "gatearm", "tacabsarm", "gatewc", "trainwc", "evalwc", "gatepgt", "trainpgt", "evalpgt",
               "gategrid", "traingrid", "evalgrid", "gatehist", "trainhist", "evalhist"):
        assert f'if /i "%~1"=="{st}" goto :' in q, st


@needs_queue
def test_queue_dscale_runs_the_command_it_prints_static():
    """A dry run can only print: the real line is checked statically -- the trainer / eval appear ONCE per command, inside the one string, and the real run executes that string."""
    for label, var, pre in (("gatedscale", "GD", '--arm M2b --data-frac 0.5 --out "%RUNSM%\\gate_M2b-df50" '),
                            ("traindscale", "TD", '--arm M2b --seed %~2 --grid pool4x10 --data-frac 0.5 --out "%OUT%" ')):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" ' + pre) for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
        assert sum(x.count("--data-frac") for x in code if x.startswith(f'set "{var}=')) == 1, f"{label}: --data-frac appears once, in the one string"
        assert sum(x.count("0.5") for x in code if x.startswith(f'set "{var}=')) == 1
    code = _code(_block("evaldscale"))
    assert [x for x in code if "%EVAL%" in x] == [x for x in code if re.match(r'set "E[0-3]=', x)], "the eval script appears only inside the four strings"
    assert len([x for x in code if re.match(r'set "E[0-3]=', x)]) == 4
    assert "if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)" in code
    for k in range(4):
        assert sum(x.strip().startswith(f"%E{k}% >> ") for x in code) == 1, f"E{k} is executed exactly once"
    e = {k: next(x for x in code if x.startswith(f'set "E{k}=')) for k in range(4)}
    assert "--project --families" in e[0] and "--tac" not in e[0] and "--tac absmask" in e[1] and "--tac oracle" in e[2] and "--tac-groups" not in e[2]
    assert "--tac oracle --tac-groups all-latcons --tac-alpha 1" in e[3]
    assert 'set "OUT=%RUNSM%\\M2b_s%~2_df50"' in code and 'set "TG=M2b-df50_s%~2"' in code, "the eval reads Q's own run dir"
    for k in range(4):
        assert '--ckpt "%OUT%\\ckpt_last.pt"' in e[k] and "%EC% %ER%" in e[k] and "--data-frac" not in e[k] and "--grid" not in e[k] and "--hist" not in e[k], \
            "the eval reads the model FROM the checkpoint; no data flag exists on the eval line"
    for label in DSTAGES:
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block"
        assert sum(1 for x in body if x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)")) >= (1 if label != "evaldscale" else 4)
    assert 'if exist "%OUT%\\summary.json" (echo [skip] M2b_s%~2_df50 & exit /b 0)' in _code(_block("traindscale")), "a finished Q run is not re-trained"


@needs_queue
@win
def test_queue_dscale_stages_dry_one_lever(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    rc, ln = _dry("gatedscale", env)
    assert rc == 0 and len(ln) == 1 and "--selfcheck-only" in ln[0] and "gate_M2b-df50" in ln[0], ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm"):t.index("--arm") + 4] == ["--arm", "M2b", "--data-frac", "0.5"] and t.count("--data-frac") == 1, ln
    rc, ln = _dry("traindscale 0", env)
    assert rc == 0 and len(ln) == 1, ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm") + 1] == "M2b" and t[t.index("--seed") + 1] == "0" and t[t.index("--data-frac") + 1] == "0.5" and t.count("--data-frac") == 1, ln
    assert t[t.index("--grid") + 1] == "pool4x10" and t.count("--grid") == 1 and t[t.index("--out") + 1].endswith("M2b_s0_df50"), ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0] and "--batch 64" in ln[0], ln
    for no in ("--w-cons", "--w-tac", "--w-wm", "--hmax", "--hist", "--window-start", "--steps ", "--max-clips", "--p-gt"):
        assert no not in ln[0], f"ONE lever: {no} is not on the line"
    rc, ev = _dry("evaldscale 0", env)
    assert rc == 0 and len(ev) == 4, ev
    for k, (suffix, frag) in enumerate((("", "--project --families"), ("-tacabs", "--tac absmask"), ("-tacorc", "--tac oracle"),
                                        ("-tacorc-all-latcons", "--tac oracle --tac-groups all-latcons --tac-alpha 1"))):
        assert f"--tag M2b-df50_s0{suffix} " in ev[k] + " " and frag in ev[k], (k, ev[k])
        assert f"M2b_s0_df50{os.sep}ckpt_last.pt" in ev[k], "every eval reads THIS run's last checkpoint"
        assert f"{os.sep}M2b_s0{os.sep}" not in ev[k] and "--data-frac" not in ev[k], "never B's checkpoint, and no data flag"
    # ONE lever, from the queue's own lines: Q's gate / train line is B's line with --data-frac 0.5 (and the output dir) added and NOTHING else changed
    b_train = _dry("train M2b 0", env)[1][0]
    assert "--data-frac" not in b_train
    assert _without(b_train, {"--out"}) == _without(_dry("traindscale 0", env)[1][0], {"--data-frac", "--out"}), "Q's train line = B's + --data-frac 0.5"
    b_gate = _dry("gatearm M2b", env)[1][0]
    assert _without(b_gate, {"--out"}) == _without(_dry("gatedscale", env)[1][0], {"--data-frac", "--out"}), "Q's gate line = B's gate line + --data-frac 0.5"
    # and Q's train argv differs from HIST's by exactly the lever swap: `--hist 10 --window-start 3` out, `--data-frac 0.5` in
    h_train = _dry("trainhist 0", env)[1][0]
    assert _without(h_train, {"--hist", "--window-start", "--out"}) == _without(_dry("traindscale 0", env)[1][0], {"--data-frac", "--out"})


@needs_queue
@win
def test_queue_dscale_refuses_missing_arguments(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("traindscale", "evaldscale"):
        rc, ln = _dry(args, env)
        assert rc == 2 and ln == [], (args, rc, ln)


@needs_queue
@win
def test_queue_dscale_leaves_every_other_stage_untouched(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("train M2b 0", "trainpgt 0", "trainwc 0", "gatearm M2b", "gatepgt", "gategrid M2b", "traingrid 0 M2b", "evalgrid 0 M2b", "gatehist", "trainhist 0", "evalhist 0"):
        rc, ln = _dry(args, env)
        assert rc == 0 and len(ln) >= 1 and all("--data-frac" not in x and "df50" not in x for x in ln), (args, ln)


@pytest.fixture
def qtmp(tmp_path):
    """The fake tree mirrors the queue's long repo-relative path (HJRAW), and Windows refuses a directory over 248 characters: under a long --basetemp (a scratchpad) the tree goes to the system temp."""
    if len(str(tmp_path)) <= 60:
        yield tmp_path
        return
    d = Path(tempfile.mkdtemp(prefix="dsq"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
def test_queue_dscale_real_lines_execute_exactly_the_printed_commands(qtmp):
    repo, rec = _fake_tree(qtmp)
    env = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(qtmp / "runsm"), CACHEFULL=str(qtmp / "cf"), V9=str(qtmp / "v9"))
    env.pop("V8F_DRYRUN", None)

    def real(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=120, encoding="utf-8")
        assert r.returncode == 0, (args, r.stdout, r.stderr)
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8")
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]       # drop python.exe and the script: sys.argv[1:]
    want_gate, want_train, want_eval = printed("gatedscale"), printed("traindscale 0"), printed("evaldscale 0")        # BEFORE the runs: a finished stage prints [skip]
    assert len(want_gate) == 1 and len(want_train) == 1 and len(want_eval) == 4
    got = real("gatedscale")
    assert [g["argv"] for g in got] == want_gate and got[-1]["who"] == "train" and "--selfcheck-only" in got[-1]["argv"]
    n = len(got)
    got = real("traindscale 0")[n:]
    assert [g["argv"] for g in got] == want_train and len(got) == 1, "the REAL traindscale line hands the trainer the printed command"
    a = got[0]["argv"]
    assert a[a.index("--data-frac") + 1] == "0.5" and a.count("--data-frac") == 1 and a[a.index("--arm") + 1] == "M2b" and a[a.index("--seed") + 1] == "0" and a[a.index("--grid") + 1] == "pool4x10"
    assert (qtmp / "runsm" / "M2b_s0_df50" / "summary.json").exists() and not (qtmp / "runsm" / "M2b_s0").exists(), "Q has its own run dir; B's is never touched"
    n += 1
    got = real("evaldscale 0")[n:]
    assert [g["argv"] for g in got] == want_eval and len(got) == 4 and all(g["who"] == "eval" for g in got), "the REAL evaldscale lines hand the eval the printed commands"
    tags = [g["argv"][g["argv"].index("--tag") + 1] for g in got]
    assert tags == ["M2b-df50_s0", "M2b-df50_s0-tacabs", "M2b-df50_s0-tacorc", "M2b-df50_s0-tacorc-all-latcons"]
    n += 4                                              # a finished stage is skipped, not re-run
    assert len(real("evaldscale 0")) == n and len(real("traindscale 0")) == n


@needs_queue
@win
def test_queue_dscale_real_stages_notice_failures_and_skip_exactly_what_is_finished(qtmp):
    repo, rec = _fake_tree(qtmp)
    runsm = qtmp / "runsm"
    ev = runsm / "eval"
    base = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(runsm), CACHEFULL=str(qtmp / "cf"), V9=str(qtmp / "v9"))
    base.pop("V8F_DRYRUN", None)

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(base, **extra), capture_output=True, text=True, timeout=120, encoding="utf-8")
        return r.returncode, r.stdout

    def evals():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines() if '"eval"' in x] if rec.exists() else []
    rc, out = stage("gatedscale", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] gate M2b-df50" in out
    rc, out = stage("traindscale 0", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] train M2b_s0_df50 - see" in out
    rc, out = stage("traindscale 0", FAKE_TRAIN="nosummary")                     # a trainer that exits 0 WITHOUT a summary (a crash that swallowed its code)
    assert rc == 1 and "exited 0 without summary.json" in out
    rc, out = stage("evaldscale 0")                                              # evaluation refuses a run that has not finished training
    assert rc == 2 and "has not finished training" in out and not evals()
    assert stage("traindscale 0")[0] == 0
    rc, out = stage("traindscale 0")
    assert rc == 0 and "[skip] M2b_s0_df50" in out, "a finished run is not re-trained"
    tags = ["M2b-df50_s0", "M2b-df50_s0-tacabs", "M2b-df50_s0-tacorc", "M2b-df50_s0-tacorc-all-latcons"]
    for tag in tags:
        rc, out = stage("evaldscale 0", FAKE_FAIL_TAG=tag)
        assert rc == 1 and f"[FAIL] eval {tag} - see" in out, (tag, rc, out)
        for f in list(ev.glob("eval_M2b-df50_s0*.npz")):
            f.unlink()
    ev.mkdir(exist_ok=True)
    for n, done in enumerate(([0, 2], [1], [0], [2], [3], [0, 1, 2], [1, 3])):
        for f in list(ev.glob("eval_M2b-df50_s0*")):
            f.rename(f.with_name(f"old{n}_" + f.name))
        rec.write_text("", encoding="utf-8")
        for k in done:
            (ev / f"eval_{tags[k]}.npz").write_bytes(b"x")
        assert stage("evaldscale 0")[0] == 0, done
        ran = [g["argv"][g["argv"].index("--tag") + 1] for g in evals()]
        assert ran == [tg for k, tg in enumerate(tags) if k not in done], (done, ran)


# =================================================================================================================================================
# THE CHAIN (code/box2_ops/chain_v8f_dscale.sh): needs box 2 and ssh, so it cannot RUN here -- its order, its gate assertions (RUN against synthetic selfchecks),
# its --dry-run (== the queue's own dry-run lines) and every load-bearing line are pinned
# =================================================================================================================================================
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")


@needs_chain
def test_chain_dscale_static_ordering_the_wait_and_no_base_decision():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    code = "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))
    assert "decide-base" not in code and "PGTD" not in code and "--pgt-dir" not in code and "--base" not in code and "BASE" not in code, "B = M2b is fixed in the PREREG: nothing is decided at run time"
    assert "runone 1" not in code and "trainhist" not in code and "evalhist" not in code and "--stage" not in code and "_h10" not in code, "seed 0 only: no seed-1 stage, no HIST stage"
    assert code.count("runone 0") == 1 and "traindscale $s" in code and "evaldscale $s" in code
    # the order: predecessor -> refresh -> manifest -> check-inputs on the REAL B side -> guard -> gate -> seed 0 -> read -> DONE
    order = ["waiting for the predecessor chain", "fetch -q origin", 'stop "pull box 2\'s cache train manifest"', 'v8f_dscale_read.py" --check-inputs', "v8f_guard.sh > ", "queue gatedscale 120 gatedscale",
             "runone 0\n", "dscaleread dscale_read.json", 'finish "$D1 |']
    idx = [c.index(x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    assert 'queue "t_$g" $TRAIN_POLLS traindscale $s' in c and 'queue "e_$g" $EVAL_POLLS evaldscale $s' in c
    assert c.index('v8f_dscale_read.py" --check-inputs') < c.index("launch telemetry_long") < c.index("queue gatedscale 120"), "B's side is refused while box 2 is still idle and untouched"
    # the predecessor is PREREG_V8F_HIST's chain, by default; its markers are the shared ones, in THIS chain's own log
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_hist.out}" in c and "ZZV8FMINI-STOP" in c and "ZZV8FMINI-DONE" in c and "chain_v8f_dscale.out" in c
    assert 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c
    # the training wait covers the longest run the guard lets live: TRAIN_POLLS x 30 s > V8F_MAX_RUN_H (default) hours
    polls = int(re.search(r"^TRAIN_POLLS=(\d+)", c, re.M).group(1))
    maxh = float(re.search(r"V8F_MAX_RUN_H:-(\d+)", GUARD.read_text(encoding="utf-8")).group(1))
    assert polls == 1320 and polls * 30 / 3600 > maxh, f"a run the guard lets live ({maxh} h) outlasts the wait ({polls * 30 / 3600} h)"
    for x in ("$L/v8f_mini/runs/M2b_s$s/$f", "$L/v8f_cz/eval/$f", "$L/v8f_tacgrp2/eval/$f", "floors_perwindow.npz"):
        assert x in c, x
    assert c.index("floors_perwindow.npz") < c.index('v8f_dscale_read.py" --check-inputs') < c.index("queue gatedscale 120")
    assert "for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull" in c
    assert 'for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull "eval/eval_$tg$f.$x"' in c
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"', '"$B1"')), "the reader and the lead block ride in the archive"
    for f in ("v8f_dscale_read.py", "v8f_hist_read.py", "v8f_grid_read.py", "v8f_pgt_read.py", "v8f_dk_read.py", "v8f_wc_read.py", "v8f_tac_groups_read.py", "v8f_mini_full_report.py"):
        assert f in c, f"the chain insists the tip has {f}"
    g = GUARD.read_text(encoding="utf-8")
    assert "ZZV8FMINI" in g and "(DONE|STOP)" in g


@needs_chain
@pytest.mark.skipif(not CHAIN_HIST.exists(), reason="chain_v8f_hist.sh is not in this checkout")
def test_chain_dscale_keeps_hist_s_guards_unchanged():
    """Every guard line of the HIST chain that carries no HIST-specific name is in this chain verbatim: the busy marker, the temperature guard, the STOP/DONE markers, the polling and the timeouts."""
    h, c = CHAIN_HIST.read_text(encoding="utf-8").splitlines(), CHAIN.read_text(encoding="utf-8")
    keys = ("BOX2_BUSY", "v8f_guard.sh", "fresh telemetry", "ZZV8FMINI-STOP", "ZZV8FMINI-DONE", "TRAIN_POLLS", "waitjob.sh", "ZZGUARD-STOP", "ZZJOB-END", "telemetry", "BOOT0", "sleep 300", "--hist")
    guard_lines = [x.strip() for x in h if not x.lstrip().startswith("#") and any(k in x for k in keys[:-1]) and not any(w in x.lower() for w in ("hist", "grid", "chain_v8f"))]
    assert len(guard_lines) >= 15
    missing = [x for x in guard_lines if x not in c]
    assert not missing, f"HIST's guard lines missing from the DSCALE chain: {missing}"


@needs_chain
def test_chain_dscale_syntax_and_dry_run_prints_the_queue_lines(tmp_path):
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
    stages = ["gatedscale", "traindscale 0", "evaldscale 0"]
    assert [x for x in out if x.startswith("== ")] == [f"== {s}" for s in stages], "seed 0 only: no seed-1 stage is printed"
    want = []                                               # the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths
    for st in stages:
        want += _dry(st, clean_paths=True)[1]
    assert [x for x in out if ".py" in x] == want and len(want) == 1 + 1 + 4
    t0 = next(x for x in out if "--seed 0" in x and "v8f_mini_train.py" in x)
    assert "--arm M2b --seed 0 --grid pool4x10 --data-frac 0.5 --out" in t0 and "D:\\tanitad\\repo" in t0 and "M2b_s0_df50" in t0
    g0 = next(x for x in out if "--selfcheck-only" in x)
    assert "--arm M2b --data-frac 0.5 --out" in g0 and "gate_M2b-df50" in g0


def _gate_assertions():
    """The python block the chain runs on the PULLED gate selfcheck.json (its heredoc), as a program."""
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"gate Q assertions\"\n(.*?)\nEOF\n", c, re.S)
    assert m, "the chain's gate assertions are a python heredoc whose failure stops the chain"
    return m.group(1)


GLINE_OK = '"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_train.py" --arm M2b --data-frac 0.5 --out "D:\\tanitad\\runs\\v8f_mini_full\\gate_M2b-df50" --kin-mode yaw --selfcheck-only'


def _gate_run(tmp_path, mod=None, line=GLINE_OK, steps_arg="19164", windows_arg="306566", manifest_mod=None):
    import copy
    body = _gate_assertions()
    syn = _syn()
    real, _ = _b_real(0)
    bsum = dict(real["summary.json"], train_sha256=syn["b_sha"], excluded_sha256=syn["ex_sha"])
    bl = bsum["levers"]
    good = {"declared_vs_built": "PASS", "live": {"pass": True, "control_prior_grad_sum": 0.0}, "steps_total": STEPS_B, "S_steps_per_epoch_100pct": S_B,
            "info": {"n_clips": N_HALF, "n_clips_expected": N_HALF, "n_clips_all": N_B, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "excluded_sha256": syn["ex_sha"],
                     "train_sha256": syn["half_sha"], "kin_mode": "yaw", "lru": 174, "n_windows_all_clips": WINDOWS_B, "n_windows": syn["windows_q"]},
            "levers": {"wm": "b", "navonly": False, "consistency": True, "cons_absent": "mask", "p_gt_start": 1.0, "p_gt_end": 0.2, "hist": 4, "hmax": 16, "window_start": 3,
                       "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5}, "params": dict(bl["params"]), "n_params_trainable": bl["n_params_trainable"]}}
    sc = copy.deepcopy(good)
    if mod:
        mod(sc)
    (tmp_path / "bsum.json").write_text(json.dumps(bsum), encoding="utf-8")
    f = tmp_path / "sc.json"
    f.write_text(json.dumps(sc), encoding="utf-8")
    man = tmp_path / "man.jsonl"
    lines = syn["lines"].splitlines(keepends=True)
    man.write_text("".join(manifest_mod(lines) if manifest_mod else lines), encoding="utf-8")
    r = subprocess.run([sys.executable, "-", str(f), syn["ex_sha"], steps_arg, str(tmp_path / "bsum.json"), line, windows_arg, str(man)], input=body, capture_output=True, text=True, encoding="utf-8", timeout=120)
    return r.returncode, r.stdout + r.stderr


@needs_chain
@needs_pkg
def test_chain_dscale_gate_assertions_pass_the_q_gate_and_refuse_every_way_it_can_be_wrong(tmp_path):
    """The chain's gate block is RUN against a synthetic selfcheck.json built from B's REAL banked record: a good Q gate passes; each single-field corruption exits 1 and names the broken field (the
    clip count, the half's identity, the all-clips budget and the model cannot drift under a green gate)."""
    syn = _syn()
    rc, out = _gate_run(tmp_path)
    assert rc == 0 and "PASS" in out and "built fraction 0.5" in out and f"params {BANKED_M2B_TOTAL}" in out and f"windows trained {syn['windows_q']} of {WINDOWS_B}" in out, out
    assert f"clips {N_HALF} of {N_B}" in out and "built hist/hmax/window_start 4 16 3" in out and D.sha12(syn["half"][0]) in out and D.sha12(syn["half"][-1]) in out, out
    L = lambda k, v: (lambda s: s["levers"].__setitem__(k, v))                   # noqa: E731
    W = lambda k, v: (lambda s: s["levers"]["w"].__setitem__(k, v))              # noqa: E731
    I = lambda k, v: (lambda s: s["info"].__setitem__(k, v))                     # noqa: E731
    pb = _b_real(0)[0]["summary.json"]["levers"]["params"]
    cases = (("declared_vs_built", lambda s: s.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built"),
             ("live", lambda s: s["live"].__setitem__("pass", False), "live"),
             ("control grad", lambda s: s["live"].__setitem__("control_prior_grad_sum", 1.0), "live"),
             # ---- the data: the half, its identity, the holdout
             ("--data-frac did not reach the data (the whole list)", lambda s: (s["info"].update(n_clips=N_B, train_sha256=_b_syn_sha())), "clips built / declared / of"),
             ("2255 clips", I("n_clips", 2255), "clips built / declared / of"), ("2257 clips", I("n_clips", 2257), "clips built / declared / of"),
             ("n_clips_all is not N_B", I("n_clips_all", 4000), "clips built / declared / of"), ("n_clips_expected", I("n_clips_expected", 2000), "clips built / declared / of"),
             ("built fraction is not 0.5 (an N_B that is not B's)", lambda s: s["info"].update(n_clips_all=4513, n_clips_expected=N_HALF), "built data fraction"),
             ("n_excluded", I("n_excluded", 59), "holdout counts"), ("excluded_in_train", I("excluded_in_train", 1), "holdout counts"), ("holdout", I("excluded_sha256", "0" * 64), "not the RA-1 holdout"),
             ("clips are another list", I("train_sha256", "d" * 64), "is not the first 2256 of B's list in sha12 order"),
             ("clips are the LAST half", I("train_sha256", syn["last_half_sha"]), "is not the first 2256 of B's list in sha12 order"),
             ("train list is B's whole list", I("train_sha256", syn["b_sha"]), "--data-frac did not reach select_train_clips"),
             ("kin_mode", I("kin_mode", "steer"), "kin_mode"), ("lru unrecorded", lambda s: s["info"].pop("lru"), "info.lru"),
             # ---- the levers and the schedule are B's
             ("wm", L("wm", "a"), "levers"), ("navonly", L("navonly", True), "levers"), ("consistency", L("consistency", False), "levers"), ("cons_absent", L("cons_absent", "zero"), "levers"),
             ("p_gt start", L("p_gt_start", 0.5), "built p_gt"), ("p_gt end", L("p_gt_end", 0.0), "built p_gt"),
             ("p_gt missing", lambda s: (s["levers"].pop("p_gt_start"), s["levers"].pop("p_gt_end")) and None, "built p_gt None -> None"),
             ("w_pen", W("w_pen", 0.6), "w_pen"), ("w_pen_self", W("w_pen_self", 0.3), "w_pen_self"), ("w_wm", W("w_wm", 2.0), "w_wm"),
             ("w_tac", W("w_tac", 2.0), "w_tac / w_cons"), ("w_cons", W("w_cons", 1.0), "w_tac / w_cons"),
             # ---- the budget: S counts ALL clips, so the steps are B's whatever the fraction
             ("steps are the half's (S sized from the half)", lambda s: s.__setitem__("steps_total", 9582), "steps 9582 != B's 19164"),
             ("steps 9000", lambda s: s.__setitem__("steps_total", 9000), "steps 9000"),
             ("S is the half's", lambda s: s.__setitem__("S_steps_per_epoch_100pct", 2396), "S 2396 != B's 4791"),
             ("the budget was sized from the half's windows", I("n_windows_all_clips", syn["windows_q"]), "the budget is not sized from B's window set"),
             ("n_windows_all_clips is not the PREREG's number", I("n_windows_all_clips", WINDOWS_B + 1), "the budget is not sized from B's window set"),
             ("the half trains all windows", I("n_windows", WINDOWS_B), "the half trains"), ("the half trains the wrong number of windows", I("n_windows", syn["windows_q"] + 1), "the half trains"),
             # ---- the model is B's
             ("built hist 10", L("hist", 10), "built hist 10 != 4"), ("built hist unrecorded", lambda s: s["levers"].pop("hist"), "built hist None != 4"),
             ("built hmax 10", L("hmax", 10), "built hmax 10 != 16"), ("built window_start 9", L("window_start", 9), "built window_start 9 != 3"),
             ("one parameter too many", L("n_params_trainable", BANKED_M2B_TOTAL + 1), "n_params_trainable"),
             ("one parameter too few", L("n_params_trainable", BANKED_M2B_TOTAL - 1), "n_params_trainable"),
             ("a group grew", lambda s: s["levers"]["params"].__setitem__("T", pb["T"] + 5), "parameter group T"),
             ("the adapter grew (a hist-10 build)", lambda s: s["levers"]["params"].__setitem__("adapter", pb["adapter"] + 1536), "parameter group adapter"),
             ("params unrecorded", lambda s: s["levers"].pop("params"), ""))
    for name, mod, msg in cases:
        rc, out = _gate_run(tmp_path, mod)
        assert rc != 0 and msg in out, (name, rc, out)
    # the lever, from the queue's own printed line
    for name, line, msg in (("no --data-frac on the line", GLINE_OK.replace(" --data-frac 0.5", ""), "--data-frac 0.5 exactly once"),
                            ("--data-frac 0.25", GLINE_OK.replace("--data-frac 0.5", "--data-frac 0.25"), "--data-frac 0.5 exactly once"),
                            ("--data-frac twice", GLINE_OK.replace("--data-frac 0.5", "--data-frac 0.5 --data-frac 0.5"), "--data-frac 0.5 exactly once"),
                            ("--steps set", GLINE_OK + " --steps 9582", "sets --steps"), ("--max-clips set", GLINE_OK + " --max-clips 100", "sets --max-clips"),
                            ("--hist set", GLINE_OK + " --hist 10", "sets --hist"), ("--window-start set", GLINE_OK + " --window-start 3", "sets --window-start"),
                            ("--hmax set", GLINE_OK + " --hmax 16", "sets --hmax"), ("--w-cons set", GLINE_OK + " --w-cons 1.0", "sets --w-cons"),
                            ("--p-gt-start set", GLINE_OK + " --p-gt-start 0.5", "sets --p-gt-start"), ("a TRAIN line, not a gate line", GLINE_OK.replace(" --selfcheck-only", ""), "not a --selfcheck-only run")):
        rc, out = _gate_run(tmp_path, line=line)
        assert rc == 1 and msg in out, (name, rc, out)
    # the manifest the half is derived from must BE B's list
    rc, out = _gate_run(tmp_path, manifest_mod=lambda ls: ls[:100] + ls[101:])
    assert rc == 1 and "not the list B trained on" in out, out
    rc, out = _gate_run(tmp_path, manifest_mod=lambda ls: [json.dumps({"clip": json.loads(x)["clip"], "n_frames": json.loads(x)["n_frames"] + 1}) + "\n" for x in ls])
    assert rc == 1 and "frame counts give" in out and "windows for B" in out, out
    # the PREREG's own numbers are asserted too (a wrong staging of B's record or of the constants cannot hide)
    rc, out = _gate_run(tmp_path, steps_arg="19000")
    assert rc == 1 and "steps 19164 != B's 19000" in out, out
    rc, out = _gate_run(tmp_path, windows_arg="306565")
    assert rc == 1 and "the PREREG's 306565" in out and "sizes S from the windows of ALL clips" in out, out


def _b_syn_sha():
    return _syn()["b_sha"]


@needs_chain
@needs_pkg
def test_chain_dscale_gate_stops_on_a_budget_sized_from_the_half(tmp_path):
    """The PREREG's arithmetic needs S to count ALL clips (Q trains 19,164 steps and sees each window ~2x as often). A trainer that sized S from the half would train 9,582 steps and make
    B - Q a different experiment ('half the data AND half the steps'); the gate STOPS the chain on it, naming the cause, before a GPU hour is spent."""
    syn = _syn()

    def half_budget(s):
        s["steps_total"] = 9582
        s["S_steps_per_epoch_100pct"] = 2396
        s["info"]["n_windows_all_clips"] = syn["windows_q"]
    rc, out = _gate_run(tmp_path, half_budget)
    assert rc == 1 and "steps 9582 != B's 19164" in out and "the budget is not sized from B's window set" in out and "sizes S from the windows of ALL clips" in out and "S 2396 != B's 4791" in out, out
    assert _gate_run(tmp_path)[0] == 0, "and a trainer that sizes S from all clips passes the gate as the PREREG states it"


@needs_chain
def test_chain_dscale_finish_releases_the_box_before_it_announces_done():
    """The next chain waits for DONE and then refuses on a leftover BOX2_BUSY.txt: finish() must remove the marker and stop the telemetry BEFORE it prints ZZV8FMINI-DONE."""
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"\nfinish\(\) \{.*?\n\}\n", c, re.S)
    assert m, "no finish() function in the chain"
    body = m.group(0)
    for need in ("telemetry.stop", "rm -f $B/BOX2_BUSY.txt", 'echo "" > $CUR', "ZZV8FMINI-DONE", "exit 0"):
        assert need in body, f"finish() lost {need!r}"
    assert body.index("rm -f $B/BOX2_BUSY.txt") < body.index("ZZV8FMINI-DONE") < body.index("exit 0")


@needs_chain
def test_chain_dscale_load_bearing_lines_are_present():
    """Every line whose loss would let a wrong run through is pinned. Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""for st in gatedscale traindscale evaldscale; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start""",
        r"""grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end""",
        r"""grep -q '"window_start": cfg.window_start_effective' || stop "the tip's built_levers does not record window_start""",
        r"""grep -q -- '"--data-frac"' || stop "the tip's trainer has no --data-frac flag""",
        r"""grep -q '"data_frac": "data:n_clips"' || stop "the tip's trainer has no G-DVB entry for data_frac""",
        r"""grep -q 'select_train_clips(clips_all, a.data_frac)' || stop "the tip's trainer does not hand --data-frac to select_train_clips""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""$SCP tanitad-box2:D:/tanitad/cache_full/train/manifest.jsonl $E/manifest_train.jsonl || stop "pull box 2's cache train manifest""",
        r"""|| stop "the reader cannot use the M2b side (check-inputs)""",
        r"""|| stop "dscale read""",
        r"""$PY "$RD/v8f_dscale_read.py" --eval-dir""",
        r"""--train-manifest "$(cygpath -m $E/manifest_train.jsonl)" """,
        r"""GLINE=$(dry_stage "$QC" gatedscale | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gatedscale line""",
        r"""$HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS "$(cygpath -m $E/manifest_train.jsonl)" <<'EOF' || stop "gate Q assertions""",
        r"""pull "gate_M2b-df50/selfcheck.json" "$E/gate_M2b-df50""",
        r"""stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do""",
        r"""grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped""",
        r"""cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo""",
        r"""for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done""",
        r"""cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz""",
        r"""finish "$D1 | $(jget dscale_read.json next)""",
        r"""g="M2b_s${1}_df50" tg="M2b-df50_s${1}""",
        r"""grep ZZGUARD-STOP $L/v8fdscale_guard.out 2>/dev/null && stop "guard stop""",
        r"""V8F_CHAIN_OUT=$L/chain_v8f_dscale.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh""",
        r"""[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry""",
        r"""M2B_STEPS=19164""",
        r"""M2B_WINDOWS=306566""",
        r"""TRAIN_POLLS=1320; EVAL_POLLS=600""",
        r"""CUR=$L/v8fdscale_current.txt""",
        r"""if toks.count("--data-frac") != 1 or toks[toks.index("--data-frac") + 1] != "0.5":""",
        r"""if sc["steps_total"] != steps: bad.append""",
        r"""if i["n_clips"] / i["n_clips_all"] != 0.5:""",
        r"""if i["train_sha256"] != shal(half):""")
    for ln in lines:
        assert ln in c, f"missing from the chain: {ln!r}"
