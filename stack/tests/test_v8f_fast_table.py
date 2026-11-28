"""RECOVER deviation table, FAST READ: ``ClipStore.get_kin`` (``tanitad/data/v8f_mini_data.py``) + its one caller ``RecoverDevTable.from_dagger_bank`` (``tanitad/data/v8f_recover.py``).

WHY. The table needs two arrays per clip -- ``pose`` (``pose_grid`` as float64) and ``kin3`` -- but ``ClipStore.get`` did a FULL ``torch.load`` of every TRAIN clip file, image-feature grid included, from a USB HDD: ~42-45 min, TWICE per
recovery-family run (the launch gate's ``--selfcheck-only`` and the training start). ``get_kin`` memory-maps the file and touches only what the table reads. The CONTRACT this file pins: the table's columns, order, dtype, counts,
``log_v`` and ``content_sha256()`` are BIT-IDENTICAL to the ``get`` path's. It is an I/O path only: no flag, no lever, no number moves.

WHAT IS PROVEN, hop by hop (the class "built, tested, and unreachable from its caller", CLAUDE.md 2026-09-10/11):
  1. per clip, ``get_kin`` == ``get`` for ``pose`` / ``kin3``: dtype, shape and BYTES (NaN rows included), in the yaw AND the steer mode;
  2. ``from_dagger_bank`` USES it (a counter on the full ``get`` reads exactly 0 on the fast path and > 0 on a ``get``-only store), and the table it builds equals the ``get`` path's column for column;
  3. the TRAINER's ``make_recover`` reaches it, on the generator-written bank of the trainer's own synthetic world;
  4. the refusals ``get`` has (no track / a foreign track / a missing file) are kept; a non-mmap-loadable file falls back, counted, bit-identical; the mapping does not outlive the call;
  5. REAL clip files, when a real cache is on this machine (skipped, loudly, when absent -- the real train cache lives on box 2).
Each deliberate-regression arm breaks ONE thing and requires the identity check to go RED.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_recover as R  # noqa: E402

torch.set_num_threads(1)
RECOVER_P = REPO / "stack" / "tanitad" / "data" / "v8f_recover.py"
SYN_T = 70


# ------------------------------------------------------------------------------------------------ the world, a bank, two stores
def make_world(root, kin_mode, nan_row=True):
    """The trainer's synthetic world (the exact S4/v9 on-disk formats). With ``nan_row`` one clip's ``kin3`` row 7 (steer) / native-track frames 14-15 (yaw) are NaN, so the table's 'logged row is not real' drop is live."""
    D.make_synthetic_world(root, n_train=6, n_eval=2, T=SYN_T, seed=3)
    names = sorted(p.stem for p in (Path(root) / "train" / "clips").glob("*.pt"))
    tracks = D.load_native_tracks(Path(root) / "native_tracks.pt")
    if nan_row:
        f = Path(root) / "train" / "clips" / f"{names[1]}.pt"
        o = torch.load(f, map_location="cpu", weights_only=True)
        o["kin3"][7] = float("nan")
        torch.save(o, f)
        tracks[names[1]] = tracks[names[1]].copy()
        tracks[names[1]][14:16] = np.nan
    return names, (tracks if kin_mode == "yaw" else None)


class FakeBank:
    """What ``from_dagger_bank`` reads of a ``DaggerBank``: ``a`` (reason, kin3, pose, j), ``ci``, ``sha256``, ``path``."""

    def __init__(self, names, seed=0, per_clip=24, hmax=10):
        rng = np.random.default_rng(seed)
        ci, j = [], []
        for c in range(len(names)):
            ci += [c] * per_clip
            j += [0, 1, 2, 3, 7, 8, SYN_T - 1, SYN_T, SYN_T + 4] + list(rng.integers(2, SYN_T, per_clip - 9))      # the edges: j < 2, the NaN row, j >= T
        n = len(ci)
        self.ci = np.asarray(ci, np.int64)
        self.a = {"reason": np.where(rng.random(n) < 0.15, 1, 0), "j": np.asarray(j, np.int64), "kin3": rng.normal(6.0, 2.0, (n, 3)),
                  "pose": np.cumsum(rng.normal(0.0, 0.5, (n, hmax, 3)), 1)}
        self.sha256, self.path = "ab" * 32, "fake_bank"


class Spy(D.ClipStore):
    """The real ClipStore, counting full reads and fast reads."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_get = self.n_kin = 0

    def get(self, clip):
        self.n_get += 1
        return super().get(clip)

    def get_kin(self, clip):
        self.n_kin += 1
        return super().get_kin(clip)


class GetOnly:
    """A store with ONLY ``get`` -- the slow reference path (today's table build; also the shape of a unit-test store)."""

    def __init__(self, store):
        self._s, self.n_get = store, 0

    def get(self, clip):
        self.n_get += 1
        return self._s.get(clip)


def _bytes_equal(x, y, what):
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype, f"{what}: dtype {x.dtype} != {y.dtype}"
    assert x.shape == y.shape, f"{what}: shape {x.shape} != {y.shape}"
    assert x.tobytes() == y.tobytes(), f"{what}: bytes differ"


def assert_identity(root, names, tracks, kin_mode):
    """THE identity check (mutated below): per clip ``get_kin`` == ``get``, and the table + ``log_v`` through the caller equal the ``get``-only path's, bit for bit."""
    s_ref, s_fast = D.ClipStore(root, "train", lru=2, kin_mode=kin_mode, tracks=tracks), D.ClipStore(root, "train", lru=2, kin_mode=kin_mode, tracks=tracks)
    for c in names:
        g, k = s_ref.get(c), s_fast.get_kin(c)
        assert set(k) == {"pose", "kin3"}
        _bytes_equal(g["pose"], k["pose"], f"{c} pose")
        _bytes_equal(g["kin3"], k["kin3"], f"{c} kin3")
    J, bank = {"clip": np.asarray(names)}, FakeBank(names)
    slow = GetOnly(D.ClipStore(root, "train", lru=2, kin_mode=kin_mode, tracks=tracks))
    fast = Spy(root, "train", lru=2, kin_mode=kin_mode, tracks=tracks)
    t_ref, v_ref = R.RecoverDevTable.from_dagger_bank(bank, slow, J)
    t_fast, v_fast = R.RecoverDevTable.from_dagger_bank(bank, fast, J)
    assert slow.n_get == len(names) and fast.n_kin == len(names), (slow.n_get, fast.n_kin)
    assert fast.n_get == 0, "the fast table build fell back to the full per-clip read"
    assert len(t_fast) == len(t_ref) > 0 and t_fast.counts == t_ref.counts
    assert t_ref.counts["dropped_logged_row_unreal"] > 0, "the fixture must exercise the drop mask"
    for k in R.TABLE_FIELDS:
        _bytes_equal(t_ref.cols[k], t_fast.cols[k], f"table.{k}")
    _bytes_equal(v_ref, v_fast, "log_v")
    assert t_ref.content_sha256() == t_fast.content_sha256()
    return t_fast


# ------------------------------------------------------------------------------------------------ 1-2. identity, per clip and through the caller
@pytest.mark.parametrize("kin_mode", ["yaw", "steer"])
def test_get_kin_equals_get_per_clip_and_the_table_is_bit_identical_through_from_dagger_bank(tmp_path, kin_mode):
    names, tracks = make_world(tmp_path, kin_mode)
    t = assert_identity(tmp_path, names, tracks, kin_mode)
    assert t.content_sha256() and 0 < len(t) < len(FakeBank(names).ci)


@pytest.mark.parametrize("kin_mode", ["yaw", "steer"])
def test_get_kin_on_a_clip_already_in_the_lru_answers_from_it_and_does_not_fill_the_lru(tmp_path, kin_mode):
    names, tracks = make_world(tmp_path, kin_mode, nan_row=False)
    s = D.ClipStore(tmp_path, "train", lru=4, kin_mode=kin_mode, tracks=tracks)
    k0 = s.get_kin(names[0])
    assert len(s._c) == 0, "get_kin must not enter the LRU (it holds no feature grid)"
    g = s.get(names[0])
    k1 = s.get_kin(names[0])
    assert k1["pose"] is g["pose"] and k1["kin3"] is g["kin3"]
    _bytes_equal(k0["pose"], k1["pose"], "pose"), _bytes_equal(k0["kin3"], k1["kin3"], "kin3")


def test_get_kin_uses_mmap_and_does_not_hold_the_file_open(tmp_path, monkeypatch):
    names, tracks = make_world(tmp_path, "yaw", nan_row=False)
    s = D.ClipStore(tmp_path, "train", kin_mode="yaw", tracks=tracks)
    seen = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (seen.append(dict(k)), real(*a, **k))[1])
    s.get_kin(names[0])
    assert seen and seen[0].get("mmap") is True and seen[0].get("weights_only") is True and s.kin_fallbacks == 0
    monkeypatch.undo()
    f = tmp_path / "train" / "clips" / f"{names[0]}.pt"
    f.unlink()                                       # Windows refuses to delete a file that is still mapped
    assert not f.exists()


# ------------------------------------------------------------------------------------------------ 4. refusals kept, fallback counted
def test_the_refusals_get_has_are_kept(tmp_path):
    names, tracks = make_world(tmp_path, "yaw", nan_row=False)
    s = D.ClipStore(tmp_path, "train", kin_mode="yaw", tracks={k: v for k, v in tracks.items() if k != names[0]})
    with pytest.raises(KeyError, match="no native pose track"):
        s.get_kin(names[0])
    with pytest.raises(KeyError, match="no native pose track"):
        s.get(names[0])
    bad = dict(tracks)
    bad[names[2]] = bad[names[2]] + 1.0                                      # a foreign track: the even rows no longer reproduce pose_grid
    s = D.ClipStore(tmp_path, "train", kin_mode="yaw", tracks=bad)
    with pytest.raises(ValueError, match="disagrees with the cache pose_grid"):
        s.get_kin(names[2])
    with pytest.raises(ValueError, match="disagrees with the cache pose_grid"):
        s.get(names[2])
    with pytest.raises(FileNotFoundError):
        s.get_kin("no-such-clip")
    assert s.kin_fallbacks == 0


@pytest.mark.parametrize("exc", [RuntimeError("mmap can only be used with files saved with `torch.save(_use_new_zipfile_serialization=True)`"), TypeError("load() got an unexpected keyword argument 'mmap'")])
@pytest.mark.parametrize("kin_mode", ["yaw", "steer"])
def test_a_file_mmap_cannot_read_falls_back_counted_and_bit_identical(tmp_path, monkeypatch, kin_mode, exc):
    """(torch 2.11 cannot ``get()`` a legacy-format file under ``weights_only=True`` either, so the refusal is simulated at ``torch.load``: the exact exceptions an old torch / a non-zip file raise for ``mmap=True``.)"""
    names, tracks = make_world(tmp_path, kin_mode, nan_row=False)
    s = D.ClipStore(tmp_path, "train", kin_mode=kin_mode, tracks=tracks)
    r = D.ClipStore(tmp_path, "train", kin_mode=kin_mode, tracks=tracks)
    real = torch.load

    def load(*a, **k):
        if k.get("mmap"):
            raise exc
        return real(*a, **k)
    monkeypatch.setattr(torch, "load", load)
    g, k = r.get(names[3]), s.get_kin(names[3])
    assert s.kin_fallbacks == 1
    _bytes_equal(g["pose"], k["pose"], "pose"), _bytes_equal(g["kin3"], k["kin3"], "kin3")
    s.get_kin(names[0])
    assert s.kin_fallbacks == 2                                              # counted per call, never silent


def test_a_store_without_get_kin_keeps_working_with_from_dagger_bank(tmp_path):
    names, tracks = make_world(tmp_path, "yaw", nan_row=False)
    g = GetOnly(D.ClipStore(tmp_path, "train", kin_mode="yaw", tracks=tracks))
    t, _ = R.RecoverDevTable.from_dagger_bank(FakeBank(names), g, {"clip": np.asarray(names)})
    assert g.n_get == len(names) and len(t) > 0


# ------------------------------------------------------------------------------------------------ deliberate-regression arms: the identity check must go RED
def _broken(fn):
    """A ``ClipStore.get_kin`` whose result is passed through ``fn(pose, kin3) -> (pose, kin3)``."""
    orig = D.ClipStore.get_kin

    def get_kin(self, clip):
        d = orig(self, clip)
        pose, kin3 = fn(d["pose"], d["kin3"])
        return {"pose": pose, "kin3": kin3}
    return get_kin


MUTANTS = {
    "pose_float32": lambda p, k: (p.astype(np.float32), k),                                          # the wrong cast (the brief's example)
    "kin3_float32_rounded": lambda p, k: (p, k.astype(np.float32).astype(np.float64)),                # right dtype, values rounded through float32
    "kin3_shifted_one_row": lambda p, k: (p, np.concatenate([k[1:], k[-1:]])),                       # off by one cache step
    "pose_wrong_key": lambda p, k: (p[:, [1, 0, 2, 3]], k),                                          # x/y swapped: the 'wrong key' family
}


# (steer, kin3_float32_rounded) is an EQUIVALENT mutant, measured: the steer-mode kin3 is STORED float32, so rounding it through float32 is the identity and cannot be told apart (the yaw-mode kin3 is float64 math).
@pytest.mark.parametrize("kin_mode,name", [(m, n) for m in ("yaw", "steer") for n in sorted(MUTANTS) if (m, n) != ("steer", "kin3_float32_rounded")])
def test_MUTATION_a_broken_get_kin_turns_the_identity_check_red(tmp_path, monkeypatch, kin_mode, name):
    names, tracks = make_world(tmp_path, kin_mode)
    assert_identity(tmp_path, names, tracks, kin_mode)                                              # control: unbroken -> green
    monkeypatch.setattr(D.ClipStore, "get_kin", _broken(MUTANTS[name]))
    with pytest.raises(AssertionError):
        assert_identity(tmp_path, names, tracks, kin_mode)


def test_MUTATION_the_yaw_mode_reading_the_recorded_steer_kin3_instead_of_the_track_turns_it_red(tmp_path, monkeypatch):
    names, tracks = make_world(tmp_path, "yaw")
    assert_identity(tmp_path, names, tracks, "yaw")

    def steer_kin3(self, clip):                                                                       # the yaw mode must read the NATIVE TRACK's kin3, never the cache's recorded-steer kin3
        o = torch.load(self.dir / f"{clip}.pt", map_location="cpu", weights_only=True)
        return {"pose": o["pose_grid"].numpy().astype(np.float64), "kin3": o["kin3"].numpy().astype(np.float64)}
    monkeypatch.setattr(D.ClipStore, "get_kin", steer_kin3)
    with pytest.raises(AssertionError):
        assert_identity(tmp_path, names, tracks, "yaw")


def _recover_source_without_get_kin():
    src = RECOVER_P.read_text(encoding="utf-8")
    old = 'read = getattr(store, "get_kin", None) or store.get'
    assert src.count(old) == 1, "the one-line dispatch moved: update this mutation"
    m = types.ModuleType("v8f_recover_noget_kin")
    m.__file__ = str(RECOVER_P)
    exec(compile(src.replace(old, "read = store.get"), str(RECOVER_P), "exec"), m.__dict__)
    return m


def test_MUTATION_a_from_dagger_bank_that_ignores_get_kin_is_caught_by_the_full_read_counter(tmp_path):
    names, tracks = make_world(tmp_path, "yaw", nan_row=False)
    J, bank = {"clip": np.asarray(names)}, FakeBank(names)
    ok = Spy(tmp_path, "train", kin_mode="yaw", tracks=tracks)
    R.RecoverDevTable.from_dagger_bank(bank, ok, J)
    assert ok.n_get == 0 and ok.n_kin == len(names)                                                   # control
    bad = Spy(tmp_path, "train", kin_mode="yaw", tracks=tracks)
    t_bad, _ = _recover_source_without_get_kin().RecoverDevTable.from_dagger_bank(bank, bad, J)
    assert bad.n_get == len(names) and bad.n_kin == 0                                                 # the counter reads the regression
    t_ok, _ = R.RecoverDevTable.from_dagger_bank(bank, Spy(tmp_path, "train", kin_mode="yaw", tracks=tracks), J)
    assert t_bad.content_sha256() == t_ok.content_sha256()                                            # ... while the TABLE is unchanged: only the counter can see an unreachable fast path


# ------------------------------------------------------------------------------------------------ 3. the trainer reaches it (generator-written bank, the trainer's own synthetic world)
from test_v8f_dagger_train import pack  # noqa: E402,F401  (the fixture: trainer-written checkpoint, the DATA generator's bank on the synthetic world)
from test_v8f_recover_wiring import REC, parts  # noqa: E402


def test_the_trainers_make_recover_builds_its_table_through_get_kin_and_equals_the_get_path(pack, monkeypatch):
    import v8f_mini_train as TR
    a, cfg, model, opt, info, rcv, J, wins, clips, store = parts(pack, REC(pack))                    # the trainer as it runs
    tracks = D.load_native_tracks(pack.world / "native_tracks.pt")
    fast = Spy(pack.world, "train", lru=8, kin_mode="yaw", tracks=tracks)
    rcv_fast = TR.make_recover(a, J, wins, fast, clips, {})
    assert fast.n_get == 0 and fast.n_kin > 0 and fast.kin_fallbacks == 0, (fast.n_get, fast.n_kin)  # reachability: argparse -> make_recover -> load_recover_table -> from_dagger_bank -> get_kin
    slow = GetOnly(D.ClipStore(pack.world, "train", lru=8, kin_mode="yaw", tracks=tracks))
    rcv_slow = TR.make_recover(a, J, wins, slow, clips, {})
    assert slow.n_get > 0
    assert rcv_fast.table.content_sha256() == rcv_slow.table.content_sha256() == rcv.table.content_sha256() and len(rcv_fast.table) > 0
    for k in R.TABLE_FIELDS:
        _bytes_equal(rcv_slow.table.cols[k], rcv_fast.table.cols[k], f"trainer table.{k}")


# ------------------------------------------------------------------------------------------------ 5. REAL clip files (skipped when absent: the real train cache lives on box 2)
REAL_CANDIDATES = ("D:/tanitad/cache_full", "E:/tanitad/cache_full", "F:/tanitad/cache_full")


def _real_cache():
    root = os.environ.get("TANITAD_V8F_REAL_CACHE") or next((c for c in REAL_CANDIDATES if (Path(c) / "eval" / "clips").is_dir() or (Path(c) / "train" / "clips").is_dir()), None)
    if not root:
        return None
    split = os.environ.get("TANITAD_V8F_REAL_SPLIT") or ("eval" if (Path(root) / "eval" / "clips").is_dir() else "train")
    return Path(root), split


@pytest.mark.skipif(_real_cache() is None, reason="no real v8F clip cache on this machine (set TANITAD_V8F_REAL_CACHE=<dir holding <split>/clips>; the real train cache is D:/tanitad/cache_full on box 2)")
def test_REAL_files_get_kin_equals_get_bit_for_bit_on_at_least_50_clips_and_time_both():
    root, split = _real_cache()
    clips = sorted(p.stem for p in (root / split / "clips").glob("*.pt"))
    n = min(len(clips), int(os.environ.get("TANITAD_V8F_REAL_N", "50")))
    assert n >= 50 or os.environ.get("TANITAD_V8F_REAL_N"), f"only {len(clips)} real clips in {root / split}: the identity test wants >= 50"
    tp = os.environ.get("TANITAD_V8F_REAL_TRACKS") or str(root / "native_tracks.pt")
    mode, tracks = ("yaw", D.load_native_tracks(tp)) if Path(tp).is_file() else ("steer", None)
    grid = os.environ.get("TANITAD_V8F_REAL_GRID", "pool4x10")
    # Timing protocol: each clip is read by BOTH paths (for the bit-identity), but only the FIRST read of a clip is timed -- the second finds the file in the OS cache. The first reader alternates (even index:
    # get_kin first, odd index: get first), so each path is timed cold on half of the clips, interleaved so disk drift cancels. Both paths' results are compared on every clip.
    s_get, s_kin = D.ClipStore(root, split, lru=1, grid=grid, kin_mode=mode, tracks=tracks), D.ClipStore(root, split, lru=1, grid=grid, kin_mode=mode, tracks=tracks)
    t_get, t_kin, rows = [], [], []
    for i, c in enumerate(clips[:n]):
        order = ("kin", "get") if i % 2 == 0 else ("get", "kin")
        res = {}
        for j, w in enumerate(order):
            t0 = time.perf_counter()
            res[w] = s_kin.get_kin(c) if w == "kin" else s_get.get(c)
            dt = time.perf_counter() - t0
            if j == 0:
                (t_kin if w == "kin" else t_get).append(dt)
                rows.append({"clip": c, "bytes": (root / split / "clips" / f"{c}.pt").stat().st_size, "first_reader": w, "first_read_s": dt})
        _bytes_equal(res["get"]["pose"], res["kin"]["pose"], f"{c} pose")
        _bytes_equal(res["get"]["kin3"], res["kin"]["kin3"], f"{c} kin3")
    summ = {"cache": str(root), "split": split, "kin_mode": mode, "grid": grid, "n": n, "kin_fallbacks": s_kin.kin_fallbacks, "n_timed_get": len(t_get), "n_timed_get_kin": len(t_kin), "get_s_mean": float(np.mean(t_get)), "get_kin_s_mean": float(np.mean(t_kin)),
            "get_s_median": float(np.median(t_get)), "get_kin_s_median": float(np.median(t_kin)), "mean_speedup": float(np.mean(t_get) / np.mean(t_kin)), "rows": rows}
    out = os.environ.get("TANITAD_FTAB_TIMING_OUT")
    if out:
        Path(out).write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\nREAL-FILE TIMING", {k: v for k, v in summ.items() if k != "rows"})
    assert s_kin.kin_fallbacks == 0, "real clip files were not mmap-loadable: the table build would NOT be fast on them"
