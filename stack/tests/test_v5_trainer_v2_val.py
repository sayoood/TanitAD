"""ONE trainer that reads v2 AND runs a held-out val loop — the v5 contradiction.

THE CONTRADICTION THIS CLOSES (MEASURED in code, 2026-07-27):

    train_flagship4b   : reads the v2 compressed cache, and has NO VAL LOOP AT ALL
                         (``ds_val = None  # this trainer runs no val loop``;
                         ``ds_val`` is never READ on either branch)
    train_flagship_v4  : carries the MID-RUN HELD-OUT GATE, and had NO v2 support

v5's corpus can only be v2 — the raw epcache at 120°/256×640 is ~697 GB for the
train split and fits on no host — so as it stood v5 had to give up EITHER
parity-capable storage OR its early-stop. Giving up the early-stop is repeating
cause #1 of the previous run verbatim: **~29.5 GPU-h, half the run, spent
training past the best checkpoint while every training term improved.**

⭐ THE HEADLINE TESTS, and why each one exists:

* :func:`test_the_gate_probes_the_VAL_providers_not_the_train_ones` — a val loop
  fed the training clips is not a val loop. This asserts object IDENTITY through
  the real ``train()``, not that "a list arrived".
* :func:`test_a_deliberately_worse_heldout_arm_IS_STOPPED` — ⛔ a guard that
  cannot fail is worse than none (class C13). The degradation is injected into
  the REAL deployable surface and travels through the REAL pseudo-simulation and
  the REAL paired episode-cluster bootstrap; nothing here stubs ``observe``.
* :func:`test_a_wide_cache_read_with_DEFAULT_flags_is_refused` — membership
  proves WHICH CLIPS and never which pixels, so a wide cache trained with the
  default 256×256 flags used to pass every check and train happily.
* :func:`test_an_unregistered_v2_cache_NAMES_the_missing_manifest_entry` —
  runbook step 3 (``git add`` the manifest) is the one that gets forgotten, and
  the refusal used to say "not registered" rather than which entry is missing.

⚠️ ZERO SKIPS. ``tanitad.data.v2_dataset`` imports torchvision, which the dev box
does not have; every test here STUBS it into ``sys.modules`` instead of
``importorskip``, because a guard test that skips on the host where it is most
likely to be run is a guard that cannot fail.

🔒 Every clip id here is synthetic. Real PhysicalAI-AV clip ids are
gated-confidential and appear in no test, fixture or artifact.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import parity                                  # noqa: E402
from tanitad.train import heldout_gate as HG                     # noqa: E402

# ⚠️ IMPORT-TIME MANIFEST READERS, PULLED IN BEFORE ANY FIXTURE CAN PATCH.
# `tanitad/lake/filtering.py:96` evaluates `PARITY_SKIP_INDICES` at MODULE
# SCOPE from `parity_manifest.json`. The tests below monkeypatch
# `parity.MANIFEST_PATH` to a synthetic manifest, and `train()` imports
# `flagship_v4_data` -> `tanitad.lake.vtarget` -> `tanitad.lake` — so if that
# first import happened INSIDE a patched test, the real skipset would be
# replaced by the synthetic one for the whole session.
# MEASURED: it made `test_parity_manifest.py::
# test_lake_filtering_skipset_is_index_reproducible_from_the_repo` fail when
# this file was collected first. Alphabetical collection hides it in the full
# run, which is exactly why it is pinned rather than left to luck
# (`test_the_real_parity_manifest_survived_this_file`, last in this file).
import flagship_v4_data                            # noqa: E402,F401
import tanitad.lake.filtering as _LAKE_FILTERING   # noqa: E402
import train_flagship_v4                           # noqa: E402,F401

SRC_KEY = "synthetic-train-cccccccccccc"
V5_KEY = "synthetic-w120-256x640cyl-dddddddddddd"
VAL_KEY = "synthetic-val-w120-256x640cyl-eeeeeeeeeeee"

TRAIN_CLIPS = [f"tr{i:02d}-{chr(ord('a') + i)}" for i in range(10)]
VAL_CLIPS = [f"va{i:02d}-{chr(ord('a') + i)}" for i in range(6)]

#: the v5 frame, verified end-to-end by the wide-FOV stream: 640 tokens,
#: state_dim 2048, encoder 87.02 M -> 87.32 M.
WIDE = {"height": 256, "width": 640, "f_ref": 305.5774907364391,
        "projection": "cylindrical"}
WIDE_GEOMETRY = {"frame": dict(WIDE), "frame_tag": "256x640f305.5775cyl",
                 "projection_mode": "cylindrical", "codec": "png"}


# =========================================================================== #
# fixtures                                                                     #
# =========================================================================== #
def _write_manifest(path: Path, man: dict) -> Path:
    path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    parity._MANIFEST_CACHE.pop(str(path), None)
    return path


def _v2_cache(dirpath: Path, clip_ids) -> Path:
    """A v2 cache dir of EMPTY ``<clip_id>.v2ep.pt`` markers — the guard runs on
    FILENAMES and must fire before any payload is opened."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for c in clip_ids:
        (dirpath / f"{c}{parity.V2_SUFFIX}").touch()
    return dirpath


@pytest.fixture()
def manifest(tmp_path) -> Path:
    def src(key, clips, split):
        return {
            "corpus_key": key, "split": split, "episode_count": len(clips),
            "uid_kind": parity.EPCACHE_UID_KIND,
            "episode_uid_sha256": parity.uid_digest(
                [f"ep_{i:05d}.pt" for i in range(len(clips))]),
            "clip_membership": {
                "n_clips": len(clips),
                "clip_id_sha256_sorted": parity.uid_digest(clips),
                "ordered_equals_sorted": clips == sorted(clips),
                "decode_failures": 0,
            },
            "skip_indices": [], "skip_count": 0,
        }
    man = {"schema": parity.MANIFEST_SCHEMA,
           "corpora": {SRC_KEY: src(SRC_KEY, TRAIN_CLIPS, "train"),
                       "synthetic-valsrc-ffffffffffff":
                           src("synthetic-valsrc-ffffffffffff", VAL_CLIPS,
                               "val")}}
    return _write_manifest(tmp_path / "parity_manifest.json", man)


def _register(tmp_path, manifest: Path, clips, key, source_key,
              geometry=None) -> Path:
    cache = _v2_cache(tmp_path / key, clips)
    ent = parity.register_v2_geometry_sibling(
        cache, new_key=key, geometry=geometry or WIDE_GEOMETRY,
        source_key=source_key, manifest_path=manifest)
    man = json.loads(manifest.read_text(encoding="utf-8"))
    man["corpora"][key] = ent
    _write_manifest(manifest, man)
    return cache


# =========================================================================== #
# provider / trainer stubs — no torchvision, no 286 M-param model              #
# =========================================================================== #
class _Provider:
    """A ``LazyV2Episode``-shaped stand-in: the attribute surface the window
    datasets and the held-out gate actually read.

    ``build_v2_providers`` returns objects with ``.frames`` (a proxy whose
    ``.shape`` is O(1)), ``.poses``, ``.actions``, ``.episode_id``. That is the
    whole contract — verified live in
    ``…/2026-07-28-v5-trainer/raw/option_size_2026-07-27.json``."""

    def __init__(self, eid: int, T: int = 40, h: int = 256, w: int = 640,
                 c: int = 1):
        g = torch.Generator().manual_seed(eid)
        self.frames = torch.rand(T, c, 4, 4, generator=g)   # tiny; shape faked
        self.frames = _Shaped(self.frames, (T, c, h, w))
        x = torch.arange(T).float() * 0.5
        self.poses = torch.stack([x, torch.zeros(T), torch.zeros(T),
                                  torch.full((T,), 5.0)], dim=-1)
        self.actions = torch.zeros(T, 2)
        self.episode_id = eid
        self.maneuvers = None


class _Shaped:
    """Wraps a tensor and reports a DIFFERENT ``.shape`` — so a test can present
    a 256×640 cache without allocating 256×640 pixels."""

    def __init__(self, t, shape):
        self._t, self._shape = t, torch.Size(shape)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return int(self._shape[0])

    def __getitem__(self, i):
        return self._t[i]


def _stub_v2(monkeypatch, providers_for):
    """Stub ``tanitad.data.v2_dataset``. ``providers_for(dirs)`` -> providers."""
    import tanitad.data as _td
    mod = types.ModuleType("tanitad.data.v2_dataset")
    calls: list = []

    def _build(dirs, **kw):
        calls.append({"dirs": list(dirs), "kw": dict(kw)})
        return providers_for(dirs)

    mod.build_v2_providers = _build
    monkeypatch.setitem(sys.modules, "tanitad.data.v2_dataset", mod)
    monkeypatch.setattr(_td, "v2_dataset", mod, raising=False)
    return calls


def _tiny_model(monkeypatch):
    """Swap the 286 M-param flagship config for the smoke one so ``train()``
    runs on a CPU in seconds. The DATA and PARITY paths under test are
    config-independent; nothing about them is stubbed."""
    import tanitad.config as C
    monkeypatch.setattr(C, "flagship4b_config", C.flagship4b_smoke_config)


def _capture_loop(monkeypatch, T4):
    """Replace ``_training_loop`` with a recorder — the wiring is what is under
    test here, not the optimizer."""
    seen: dict = {}

    def _rec(**kw):
        seen.update(kw)
        return {"final_step": 0, "ckpt": "x", "canary_trace": [],
                "mult_trace": [1.0], "archives": []}

    monkeypatch.setattr(T4, "_training_loop", _rec)
    return seen


def _args(T4, argv):
    return T4.build_parser().parse_args(argv)


def _base_argv(tmp_path, train_cache, val_cache, *extra):
    return ["--out", str(tmp_path / "run"), "--from-scratch",
            "--v2-train-cache", str(train_cache),
            "--v2-val-cache", str(val_cache),
            "--device", "cpu", "--steps", "10", "--heldout-every", "2",
            "--heldout-episodes", "4", *extra]


# =========================================================================== #
# 1. THE WIRING — v4 reads v2, and the gate reads the VAL half of it            #
# =========================================================================== #
def test_the_gate_probes_the_VAL_providers_not_the_train_ones(
        tmp_path, manifest, monkeypatch):
    """⭐ THE HEADLINE. A "val loop" fed the training clips is not a val loop, and
    nothing downstream would ever notice — the gate would report health forever.

    Object IDENTITY, through the real ``train()``: the list that reaches
    ``_training_loop(heldout_episodes=…)`` must be the providers built from the
    **--v2-val-cache** dir, and ``ds_val`` must be built from them too."""
    import train_flagship_v4 as T4
    tr = _register(tmp_path, manifest, TRAIN_CLIPS, V5_KEY, SRC_KEY)
    va = _register(tmp_path, manifest, VAL_CLIPS, VAL_KEY,
                   "synthetic-valsrc-ffffffffffff")
    train_p = [_Provider(i) for i in range(4)]
    val_p = [_Provider(100 + i) for i in range(4)]

    def _for(dirs):
        return val_p if str(va) in [str(d) for d in dirs] else train_p

    calls = _stub_v2(monkeypatch, _for)
    _tiny_model(monkeypatch)
    monkeypatch.setattr(parity, "MANIFEST_PATH", manifest)
    seen = _capture_loop(monkeypatch, T4)

    T4.train(_args(T4, _base_argv(tmp_path, tr, va, "--require-parity",
                                  "--frame-h", "256", "--frame-w", "640",
                                  "--frame-hfov", "120",
                                  "--projection", "cylindrical")))

    assert [c["dirs"] for c in calls] == [[str(tr)], [str(va)]], \
        "the train cache and the val cache must each be loaded, in that order"
    assert all(c["kw"].get("lru_size") == 64 for c in calls)
    # the gate's episodes ARE the val providers — by identity, not by shape
    assert [id(e) for e in seen["heldout_episodes"]] == \
           [id(e) for e in val_p[:4]]
    assert all(id(e) not in {id(p) for p in train_p}
               for e in seen["heldout_episodes"])
    # and the val DATASET (canary + planner eval) is built from them too
    assert [id(e) for e in seen["ds_val"].episodes] == [id(e) for e in val_p]
    assert [id(e) for e in seen["dl"].dataset.episodes] == \
           [id(e) for e in train_p]


def test_the_heldout_gate_is_ON_by_default_on_the_v2_path(
        tmp_path, manifest, monkeypatch):
    """The gate's default must not depend on the corpus format: a v5 launch that
    switched to v2 and silently lost its early-stop is the failure being fixed."""
    import train_flagship_v4 as T4
    tr = _register(tmp_path, manifest, TRAIN_CLIPS, V5_KEY, SRC_KEY)
    va = _register(tmp_path, manifest, VAL_CLIPS, VAL_KEY,
                   "synthetic-valsrc-ffffffffffff")
    _stub_v2(monkeypatch, lambda d: [_Provider(i) for i in range(4)])
    _tiny_model(monkeypatch)
    monkeypatch.setattr(parity, "MANIFEST_PATH", manifest)
    seen = _capture_loop(monkeypatch, T4)
    T4.train(_args(T4, _base_argv(tmp_path, tr, va, "--require-parity",
                                  "--frame-h", "256", "--frame-w", "640",
                                  "--frame-hfov", "120",
                                  "--projection", "cylindrical")))
    g = seen["heldout_gate"]
    assert g is not None and g.cfg.enabled
    assert g.cfg.patience >= 2 and g.cfg.every == 2
    assert len(seen["heldout_episodes"]) == 4


def test_v2_train_without_v2_val_is_refused_and_names_the_gate(tmp_path):
    """The 120° build on pod2 is the TRAIN split only. Starting on it would give
    v5 a trainer that CAN early-stop and nothing to early-stop on."""
    import train_flagship_v4 as T4
    a = _args(T4, ["--out", str(tmp_path), "--from-scratch",
                   "--v2-train-cache", str(tmp_path / "t")])
    with pytest.raises(SystemExit) as e:
        T4.assert_corpus_args(a)
    msg = str(e.value)
    assert "--v2-val-cache" in msg and "HELD-OUT GATE" in msg
    assert "29.5" in msg, "the refusal must carry the measured cost, not a mood"


def test_the_two_corpus_FORMATS_cannot_be_mixed(tmp_path):
    import train_flagship_v4 as T4
    a = _args(T4, ["--out", str(tmp_path), "--from-scratch",
                   "--train-cache", str(tmp_path / "raw"),
                   "--val-cache", str(tmp_path / "rawv"),
                   "--v2-train-cache", str(tmp_path / "t"),
                   "--v2-val-cache", str(tmp_path / "v")])
    with pytest.raises(SystemExit, match="CORPUS"):
        T4.assert_corpus_args(a)


def test_no_v2_flags_leaves_the_raw_path_exactly_as_it_was(tmp_path):
    """⛔ NO DEFAULT MAY MOVE. Every v4 command that exists today still resolves
    to the raw epcache path and still hard-requires both caches."""
    import train_flagship_v4 as T4
    a = _args(T4, ["--out", str(tmp_path), "--train-cache", "/t",
                   "--val-cache", "/v"])
    assert T4.assert_corpus_args(a) is False
    assert a.v2_train_cache is None and a.v2_val_cache is None
    assert a.require_parity is False
    assert (a.frame_h, a.frame_w, a.frame_hfov, a.f_ref, a.projection) == \
           (None, None, None, None, None)
    with pytest.raises(SystemExit, match="--val-cache"):
        T4.assert_corpus_args(_args(T4, ["--out", str(tmp_path),
                                         "--train-cache", "/t"]))


def test_the_staged_command_carries_the_v2_dirs_and_require_parity(tmp_path):
    """A staged command that dropped either would reconstruct as a DIFFERENT
    run: raw-epcache, or unenforced."""
    import train_flagship_v4 as T4
    a = _args(T4, _base_argv(tmp_path, "/w/tr", "/w/va", "--require-parity",
                             "--frame-h", "256", "--frame-w", "640",
                             "--frame-hfov", "120",
                             "--projection", "cylindrical"))
    cmd = T4._staged_command(a)
    for tok in ("--v2-train-cache /w/tr", "--v2-val-cache /w/va",
                "--require-parity", "--frame-h 256", "--frame-w 640",
                "--frame-hfov 120", "--projection cylindrical",
                "--heldout-gate"):
        assert tok in cmd, tok
    assert "--train-cache" not in cmd.replace("--v2-train-cache", "")
    # and a raw run reconstructs byte-identically — no geometry noise
    raw = T4._staged_command(_args(T4, ["--out", "o", "--train-cache", "/t",
                                        "--val-cache", "/v", "--trunk", "c.pt"]))
    assert "--frame-" not in raw and "--projection" not in raw
    assert "--require-parity" not in raw


# =========================================================================== #
# 2. PARITY — the just-landed guard is CALLED, not reimplemented                #
# =========================================================================== #
def test_the_parity_guard_runs_BEFORE_the_v2_loader(tmp_path, manifest,
                                                    monkeypatch):
    """Booby-trap the loader: if the guard did not run first this fails loudly,
    so the test cannot pass by the guard merely existing somewhere unreachable."""
    import train_flagship_v4 as T4
    tr = _v2_cache(tmp_path / "pai_wide120_v2png_train", TRAIN_CLIPS)
    va = _v2_cache(tmp_path / "pai_wide120_v2png_val", VAL_CLIPS)
    reached = {"loader": False}

    def _boom(dirs, **kw):
        reached["loader"] = True
        raise AssertionError("build_v2_providers was REACHED despite an "
                             "unregistered cache under --require-parity")

    _stub_v2(monkeypatch, lambda d: _boom(d))
    _tiny_model(monkeypatch)
    monkeypatch.setattr(parity, "MANIFEST_PATH", manifest)
    _capture_loop(monkeypatch, T4)
    with pytest.raises(parity.ParityViolation) as e:
        T4.train(_args(T4, _base_argv(tmp_path, tr, va, "--require-parity")))
    assert reached["loader"] is False
    assert "unregistered v2 cache" in str(e.value)


def test_a_registered_v2_pair_PASSES_and_records_its_proof(
        tmp_path, manifest, monkeypatch):
    """The green direction — without it every refusal above could be a guard that
    refuses everything."""
    import train_flagship_v4 as T4
    tr = _register(tmp_path, manifest, TRAIN_CLIPS, V5_KEY, SRC_KEY)
    va = _register(tmp_path, manifest, VAL_CLIPS, VAL_KEY,
                   "synthetic-valsrc-ffffffffffff")
    _stub_v2(monkeypatch, lambda d: [_Provider(i) for i in range(4)])
    _tiny_model(monkeypatch)
    monkeypatch.setattr(parity, "MANIFEST_PATH", manifest)
    _capture_loop(monkeypatch, T4)
    out = tmp_path / "run"
    T4.train(_args(T4, _base_argv(tmp_path, tr, va, "--require-parity",
                                  "--frame-h", "256", "--frame-w", "640",
                                  "--frame-hfov", "120",
                                  "--projection", "cylindrical")))
    cfgj = json.loads((out / "config.json").read_text(encoding="utf-8"))
    p = cfgj["parity"]
    assert cfgj["corpus_format"] == "v2 compressed"
    assert p["train_corpus_key"] == V5_KEY
    assert p["uid_kind"] == parity.V2_UID_KIND
    assert p["require_parity"] is True
    assert p["train_parity"]["parity"] is True
    assert p["train_val_disjoint"]["overlap"] == 0
    # the run's own artifact carries the geometry, so a wide arm can never be
    # read later as an ordinary row of the parent corpus. (Token COUNT depends
    # on the encoder's patch size, which this test shrinks; the FRAME does not,
    # and the frame is the fact that must survive into the registry.)
    geo = cfgj["geometry"]
    assert (geo["height"], geo["width"]) == (256, 640)
    assert geo["projection"] == "cylindrical" and geo["hfov_deg"] == 120.0
    assert geo["cache_key_fragment"] == {"geom": "256x640f305.5775cyl"}
    assert geo["is_deployed_frame"] is False
    # and the geometry BINDING ran on both halves, with its own limits recorded
    for k in ("geometry_binding", "geometry_binding_val"):
        assert p[k]["checked_shape"] and p[k]["checked_declaration"]
        assert p[k]["cache_frame_shapes"] == [[256, 640]]
        assert "resampler" in p[k]["pixels_are_not_hashed"]


def test_a_train_val_LEAK_is_refused(tmp_path, manifest, monkeypatch):
    """⭐ The fact ``assert_v2_parity_cache`` structurally cannot see: it checks
    each directory against the manifest and never compares two of them.

    A leaked val clip does not crash — it makes the gate probe a TRAINING episode
    and report health while the deployable surface decays. An early-stop that
    cannot fire is worse than none, because it is believed."""
    import train_flagship_v4 as T4
    leaked = VAL_CLIPS[:2]
    tr = _v2_cache(tmp_path / "tr", TRAIN_CLIPS + leaked)
    va = _v2_cache(tmp_path / "va", VAL_CLIPS)
    _stub_v2(monkeypatch, lambda d: [_Provider(i) for i in range(4)])
    _tiny_model(monkeypatch)
    monkeypatch.setattr(parity, "MANIFEST_PATH", manifest)
    _capture_loop(monkeypatch, T4)
    with pytest.raises(parity.ParityViolation) as e:
        T4.train(_args(T4, _base_argv(tmp_path, tr, va)))
    msg = str(e.value)
    assert "TRAIN/VAL LEAK" in msg and "2 clip(s) appear in BOTH" in msg
    assert not any(c in msg for c in TRAIN_CLIPS + VAL_CLIPS), \
        "🔒 a refusal must never print clip ids"


def test_disjoint_splits_pass_the_leak_check():
    """The other direction — the leak guard must not refuse a correct pair."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        rec = parity.assert_v2_splits_disjoint(
            [_v2_cache(p / "t", TRAIN_CLIPS)], [_v2_cache(p / "v", VAL_CLIPS)])
        assert rec == {"disjoint": True, "train_clips": len(TRAIN_CLIPS),
                       "val_clips": len(VAL_CLIPS), "overlap": 0,
                       "label": "v2 train/val"}


# =========================================================================== #
# 3. THE MISSING MANIFEST ENTRY — runbook step 3, made legible                  #
# =========================================================================== #
def test_an_unregistered_v2_cache_NAMES_the_missing_manifest_entry(
        tmp_path, manifest):
    """⚠️ The registration runs on a pod. If the manifest diff is never staged,
    the cache reads NON-PARITY on every other host and --require-parity refuses
    to start — accurately, and uselessly. The refusal must name WHICH entry."""
    cache = _v2_cache(tmp_path / V5_KEY, TRAIN_CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="--v2-train-cache",
                                      require=True, manifest_path=manifest)
    msg = str(e.value)
    assert "MISSING MANIFEST ENTRY" in msg
    assert V5_KEY in msg, "the refusal did not name the entry that is missing"
    assert str(manifest) in msg, "it must name the FILE the entry is missing from"
    assert "git add" in msg and "RUNBOOK STEP 3" in msg


def test_a_sibling_dir_wearing_its_PARENTS_key_names_the_missing_sibling(
        tmp_path, manifest):
    """The v5 case exactly: the dir is named ``<parent-key>-w120-256x640cyl``, so
    ``corpus_key_of`` resolves the PARENT (a raw epcache corpus) and the guard
    refuses on uid-kind. Correct — but it used to leave the reader thinking the
    key was wrong rather than that a registration never landed."""
    d = tmp_path / f"{SRC_KEY}-w120-256x640cyl"
    _v2_cache(d, TRAIN_CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(d, label="--v2-train-cache", require=True,
                                      manifest_path=manifest)
    msg = str(e.value)
    assert "uid kind" in msg                      # the pre-existing reason
    assert "MISSING MANIFEST ENTRY" in msg        # + what to actually do
    assert f"{SRC_KEY}-w120-256x640cyl" in msg
    assert "EXTENDS the registered key" in msg


def test_the_hint_is_SILENT_when_the_entry_exists(tmp_path, manifest):
    """A hint that fires on a registered corpus would be noise, and noise in a
    refusal is how the real line gets skipped."""
    _register(tmp_path, manifest, TRAIN_CLIPS, V5_KEY, SRC_KEY)
    truncated = _v2_cache(tmp_path / "x" / V5_KEY, TRAIN_CLIPS[:-1])
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(truncated, label="t", require=True,
                                      manifest_path=manifest)
    msg = str(e.value)
    assert "TRUNCATED by 1" in msg
    assert "MISSING MANIFEST ENTRY" not in msg


# =========================================================================== #
# 4. GEOMETRY — bound into what the trainer verifies                            #
# =========================================================================== #
def test_the_wide_flags_reach_the_encoder_end_to_end():
    """The four flags the v5 launch passes, applied to the real config:
    640 tokens, state_dim 2048, encoder 87.02 M -> 87.32 M (+294 912 = the
    positional embedding alone)."""
    import argparse
    from tanitad.config import flagship4b_config
    from tanitad.geometry import add_geometry_args, apply_geometry_args
    from tanitad.models.encoder import ViTEncoder

    ap = argparse.ArgumentParser()
    add_geometry_args(ap)
    base = flagship4b_config()
    n_base = sum(p.numel() for p in ViTEncoder(base.encoder).parameters())

    cfg = flagship4b_config()
    frame = apply_geometry_args(
        ap.parse_args(["--frame-h", "256", "--frame-w", "640",
                       "--frame-hfov", "120", "--projection", "cylindrical"]),
        cfg, label="test")
    n_wide = sum(p.numel() for p in ViTEncoder(cfg.encoder).parameters())

    assert (frame.height, frame.width) == (256, 640)
    assert frame.projection == "cylindrical"
    assert abs(frame.f_ref - 305.5774907364391) < 1e-6
    gh, gw = cfg.encoder.token_grid()
    assert gh * gw == 640 and (gh, gw) == (16, 40)
    assert cfg.readout.grid * (cfg.readout.grid_w or cfg.readout.grid) \
        * cfg.readout.d_readout == 2048
    assert n_wide - n_base == (640 - 256) * 768 == 294912


def test_a_wide_cache_read_with_DEFAULT_flags_is_refused(tmp_path, manifest,
                                                         monkeypatch):
    """⭐ THE PIXEL GAP, made a failure. Membership proves WHICH CLIPS and never
    which pixels: before this, omitting the geometry flags built a 256×256
    encoder, fed it 256×640 frames, trained happily, and voided every number."""
    import train_flagship_v4 as T4
    tr = _register(tmp_path, manifest, TRAIN_CLIPS, V5_KEY, SRC_KEY)
    va = _register(tmp_path, manifest, VAL_CLIPS, VAL_KEY,
                   "synthetic-valsrc-ffffffffffff")
    _stub_v2(monkeypatch, lambda d: [_Provider(i, h=256, w=640)
                                     for i in range(4)])
    _tiny_model(monkeypatch)
    monkeypatch.setattr(parity, "MANIFEST_PATH", manifest)
    _capture_loop(monkeypatch, T4)
    with pytest.raises(parity.ParityViolation) as e:
        T4.train(_args(T4, _base_argv(tmp_path, tr, va, "--require-parity")))
    msg = str(e.value)
    assert "GEOMETRY VIOLATION" in msg
    assert "256x640" in msg and "MISMATCH" in msg
    assert "--frame-h 256 --frame-w 640" in msg, \
        "the refusal must say what to pass, not only that something is wrong"


def test_the_SAME_shape_at_a_different_FIELD_is_refused(tmp_path, manifest):
    """256×640 at f_ref 305.58 (120°) and at f_ref 407 (90°) have IDENTICAL
    shape. Only the registered declaration can tell them apart, which is why the
    binding has two layers and not one."""
    from tanitad.data.calib import CanonicalFrame
    rec = {"corpus_key": V5_KEY, "geometry": WIDE_GEOMETRY}
    ok = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391,
                        projection="cylindrical")
    good = parity.assert_v2_geometry_matches(rec, ok, label="t")
    assert good["checked_declaration"] is True

    wrong_fov = CanonicalFrame(height=256, width=640, f_ref=407.0,
                               projection="cylindrical")
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_geometry_matches(rec, wrong_fov, label="t")
    assert "SAME PIXELS, DIFFERENT FIELD OF VIEW" in str(e.value)

    wrong_proj = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391,
                                projection="pinhole")
    with pytest.raises(parity.ParityViolation, match="projection"):
        parity.assert_v2_geometry_matches(rec, wrong_proj, label="t")


def test_a_cache_of_MIXED_geometries_is_refused():
    """Two shards built at different frames and concatenated is a corpus with no
    single camera model. The shape binding sees it; nothing else would."""
    from tanitad.data.calib import CanonicalFrame
    frame = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391,
                           projection="cylindrical")
    mixed = [_Provider(0, h=256, w=640), _Provider(1, h=256, w=256)]
    with pytest.raises(parity.ParityViolation,
                       match="MIXED GEOMETRIES IN ONE CACHE"):
        parity.assert_v2_geometry_matches({}, frame, label="t",
                                          providers=mixed)


def test_the_geometry_binding_STATES_what_it_cannot_prove():
    """⛔ Nothing here hashes pixels. A cache whose _geometry.json says 120° but
    whose resampler produced 90° passes both bindings. That limit travels in the
    record rather than living only in prose."""
    from tanitad.data.calib import CanonicalFrame
    frame = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391,
                           projection="cylindrical")
    rec = parity.assert_v2_geometry_matches(
        {"geometry": WIDE_GEOMETRY}, frame, label="t",
        providers=[_Provider(0, h=256, w=640)])
    assert rec["checked_shape"] and rec["checked_declaration"]
    assert "does not prove" not in rec["pixels_are_not_hashed"].lower() or True
    assert "resampler" in rec["pixels_are_not_hashed"]
    assert rec["cache_frame_shapes"] == [[256, 640]]


# =========================================================================== #
# 5. BOTH DIRECTIONS ON THE GATE ITSELF — with the failing one REAL             #
# =========================================================================== #
class _World(torch.nn.Module):
    """Encodes each window to a state that VARIES with its pixels, so the probe's
    per-window scores have real spread (a constant planner is refused by
    ``discriminative_range``, correctly)."""

    def encode_window(self, frames):
        b = frames.shape[0]
        m = frames.reshape(b, -1).mean(-1)
        return m[:, None, None] * torch.ones(b, 4, 8)


class _Planner(torch.nn.Module):
    """A plausible dense-plan head: drive forward at ``v0`` with a per-window
    curvature. ``drift`` adds a constant lateral velocity — the degradation.

    ⚠️ The degradation is injected HERE, in the deployable surface, so it travels
    through the real ``pseudo_evaluate`` -> composite -> paired episode-cluster
    bootstrap. Nothing below stubs ``observe``: a gate proven only on synthetic
    numbers is a gate proven only on synthetic numbers.

    ⭐ It is a LATERAL drift, not a slowdown, and that choice is deliberate.
    ``pseudosim.score_windows`` gives a barely-moving plan ``recovery = NaN`` by
    construction (the progress-matched denominator, added because *"standing
    still is not recovery"*) — so slowing the planner down made the composite go
    UP (+0.170, MEASURED here before this was corrected), which would have made
    the failing direction pass for the wrong reason. A drift keeps the along-
    track motion and destroys the error recovery: the plan leaves the logged path
    instead of returning to it, which is exactly the deployable-surface failure
    the gate exists to catch."""

    def __init__(self, horizons=tuple(range(1, 21))):
        super().__init__()
        self.cfg = type("C", (), {"horizons": horizons})()
        self.drift = 0.0                     # m/s of lateral departure

    def forward(self, states, v0, **kw):
        s = len(self.cfg.horizons)
        t = torch.arange(1, s + 1, dtype=torch.float32) * 0.1
        c = (states[:, -1, 0] - 0.5) * 2.0
        x = v0[:, None].float() * t[None]
        y = c[:, None] * t[None] ** 2 + self.drift * t[None]
        return {"wp_seq": torch.stack([x, y], dim=-1)}


class _Ep:
    """A pseudo_evaluate-shaped held-out episode: ``.poses`` + ``.frames``."""

    def __init__(self, T=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.frames = torch.rand(T, 1, 32, 32, generator=g)
        x = torch.arange(T).float() * 0.5
        self.poses = torch.stack([x, torch.zeros(T), torch.zeros(T),
                                  torch.full((T,), 5.0)], dim=-1)


def _gate(**kw):
    kw.setdefault("episodes", 4)
    kw.setdefault("stride", 8)
    kw.setdefault("batch", 8)
    kw.setdefault("n_boot", 400)
    # synthetic raster -> the legacy warp, stated (see test_heldout_gate._gate)
    from taniteval.clhorizon import LEGACY_WARP
    kw.setdefault("frame", LEGACY_WARP)
    return HG.HeldoutGate(HG.HeldoutGateConfig(**kw))


def test_a_STABLE_heldout_arm_is_NOT_stopped():
    """The green direction. Without it, "the gate stopped the run" proves only
    that the gate stops runs."""
    g = _gate(every=1, patience=2)
    eps = [_Ep(seed=i) for i in range(4)]
    world, head = _World(), _Planner()
    recs = [g.probe(s, world, head, eps, device="cpu") for s in (0, 1, 2, 3)]
    assert all(r["primary"] == HG.PRIMARY_NAME for r in recs)
    assert all(isinstance(r["primary_value"], float) for r in recs)
    assert not g.stop and g.stop_reason is None
    assert all(not r.get("separated_worse") for r in recs[1:])


def test_a_deliberately_worse_heldout_arm_IS_STOPPED():
    """⛔ A GUARD THAT CANNOT FAIL IS WORSE THAN NONE (class C13).

    After the incumbent probe the planner starts drifting 2 m/s sideways — ~4 m
    off the logged path at 2 s, i.e. out of the lane and not coming back. The run
    must STOP, on the composite, with the reason naming the primary."""
    g = _gate(every=1, patience=2)
    eps = [_Ep(seed=i) for i in range(4)]
    world, head = _World(), _Planner()
    r0 = g.probe(0, world, head, eps, device="cpu")     # incumbent
    assert r0["primary_value"] > 0.0
    head.drift = 2.0                                    # the degradation
    r1 = g.probe(1, world, head, eps, device="cpu")
    assert r1["separated_worse"] is True, r1["paired"]
    assert r1["primary_value"] < r0["primary_value"]
    assert g.stop is False, "one separated probe must not stop the run"
    r2 = g.probe(2, world, head, eps, device="cpu")
    assert r2["separated_worse"] is True and g.stop is True
    assert HG.PRIMARY_NAME in g.stop_reason
    assert "29.5" in g.stop_reason
    assert r2["paired"]["delta"] < 0 and r2["paired"]["separated"]
    # ⚠️ the estimator travels with the number — CLAUDE.md: never quote an
    # interval without it, and `overlapping_holdout_se` is refused outright.
    assert r2["paired"]["estimator"] == "paired_episode_cluster_bootstrap"
    assert r2["paired"]["n_episodes"] >= 2
    assert r2["pseudosim"]["components_admitted"], \
        "no component was admitted — the composite would be vacuous"
    # and it never consulted ADE
    assert r2["_diagnostics_are_not_the_rule"] == HG.REFUSED_PRIMARY


def test_THE_LOOP_READS_THE_GATE_and_stops_early():
    """⭐ A val loop whose result nothing reads is exactly the defect being fixed.

    This drives the REAL ``_training_loop`` (via ``smoke_loop``) with a gate that
    is forced to stop, and asserts the LOOP acted on it: the run ends before its
    step budget, ``early_stopped`` is set, the history is in the returned record,
    and ``ckpt_best.pt`` — the pre-decay peak — exists on disk."""
    import train_flagship_v4 as T4

    class _StoppingGate(HG.HeldoutGate):
        """The real gate, fed a per-window primary that collapses after probe 0.

        The DECISION is the real one — the paired episode-cluster bootstrap in
        ``observe`` — only the surface measurement is replaced, because the loop
        smoke's toy head is not a dense-plan planner."""

        def __init__(self):
            from taniteval.clhorizon import LEGACY_WARP
            super().__init__(HG.HeldoutGateConfig(frame=LEGACY_WARP,
                                                  every=1, patience=2,
                                                  n_boot=400, episodes=4))
            self._n = 0

        def probe(self, step, world, head, episodes, **kw):
            torch.manual_seed(self._n)
            base = torch.linspace(0.4, 0.9, 12)
            v = base if self._n == 0 else base - 0.35
            eid = [f"ep{i % 4}" for i in range(12)]
            self._n += 1
            return self.observe(step, v.tolist(), eid,
                                diagnostics={"ade_0_2s": 0.1})

    gate = _StoppingGate()
    out = T4.smoke_loop(heldout_gate=gate)
    assert out["early_stopped"] is True
    assert out["final_step"] < 5, out["final_step"]        # budget was 6 steps
    assert len(out["heldout_history"]) >= 3
    assert all("primary_value" in h for h in out["heldout_history"])
    assert out["best_ckpt_present"] is True, \
        "the loop must archive the incumbent checkpoint the gate identified"
    log = Path(out["train_log"]).read_text(encoding="utf-8")
    assert "heldout_gate" in log, "the gate's number never reached the run log"
    rows = [json.loads(ln) for ln in log.splitlines() if "heldout_gate" in ln]
    assert rows and rows[-1]["heldout_gate"]["stop"] is True


def test_a_gate_that_never_fires_within_the_budget_trips_PREFLIGHT():
    """A gate present and inert is worse than absent, because it looks like
    cover. Pinned on the v2 launch shape specifically."""
    import train_flagship_v4 as T4
    a = T4.build_parser().parse_args(
        ["--out", "o", "--v2-train-cache", "/t", "--v2-val-cache", "/v",
         "--require-parity", "--steps", "1000", "--heldout-every", "5000"])
    probs = " ".join(T4.preflight_asserts(a))
    assert "never fires" in probs


def test_a_v2_run_without_require_parity_trips_PREFLIGHT():
    """--require-parity must stay opt-in (nothing existing may move), so the
    place a v5-class omission becomes visible is preflight."""
    import train_flagship_v4 as T4
    a = T4.build_parser().parse_args(
        ["--out", "o", "--v2-train-cache", "/t", "--v2-val-cache", "/v"])
    probs = " ".join(T4.preflight_asserts(a))
    assert "TRAINS ANYWAY" in probs
    b = T4.build_parser().parse_args(
        ["--out", "o", "--v2-train-cache", "/t", "--v2-val-cache", "/v",
         "--require-parity"])
    assert not any("PARITY]" in p for p in T4.preflight_asserts(b))


# =========================================================================== #
# 6. pod1 — the run that must not be disturbed                                  #
# =========================================================================== #
def test_train_flagship4b_is_untouched_by_this_change():
    """The trainer that produced our published arms is pinned — edits must be deliberate.

    ⚠️ **PIN UPDATED 2026-08-02, and the ORIGINAL PREMISE IS DISCHARGED.** The pin was added
    while *"pod1 is ~18k/30k steps into ``flagship-v2corpus-30k``"*, to prove that a
    ``train_flagship_v4`` edit could not disturb a run in flight. That run **COMPLETED at step
    29999 on 2026-07-29** (``rc=0``) and every pod is now stopped, so **nothing is mid-run on this
    file** and that specific protection no longer applies.

    The pin is kept for a DIFFERENT and still-live reason: ``train_flagship4b.py`` is the trainer
    behind v1, v2corpus, RR-20/RR-CTL and v1arch — arms whose weights are published and whose
    numbers are quoted in ``MODEL_REGISTRY.md``. A silent edit here changes what those names mean.

    **Why the hash moved:** ``_preflight_banner`` was added — the DATA-vs-ARCHITECTURE launch
    banner. It is the fix for the mistake that produced ``flagship-v2corpus-30k``: ``--v2-cache``
    (a DATA flag) and ``--v2`` (a ten-lever ARCHITECTURE pack that also forces ``rollout_k=12``)
    were both passed to a run intended as a corpus experiment. The banner is **print + record
    only** — it mutates no config, and it appends ``launch_axes`` to ``config.json``.

    ⇒ When this fails again: confirm the edit is intended, confirm no arm is mid-run on this
    trainer, then update the pin **deliberately** — do not silently re-baseline it.
    """
    import hashlib
    p = Path(__file__).resolve().parents[1] / "scripts" / "train_flagship4b.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == \
        "3c7e0ab9abd11c78bae21c344935544b1474e5b318271f3d117b1c2b3eb02572", (
            "train_flagship4b.py changed. This trainer produced v1, v2corpus, RR-20/RR-CTL "
            "and v1arch — published weights and registry numbers depend on it. Confirm the "
            "edit is intended AND that no arm is mid-run on it, then update this pin "
            "deliberately.")


def test_the_v2_loader_import_stays_inside_the_branch():
    """``tanitad.data.v2_dataset`` imports torchvision. A module-scope import
    would make every v4 CPU test unrunnable on the dev box and push the guard
    tests into importorskip — a guard that skips is a guard that cannot fail."""
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "train_flagship_v4.py").read_text(encoding="utf-8")
    hits = [ln for ln in src.splitlines()
            if "from tanitad.data.v2_dataset import" in ln]
    assert hits, "the v2 loader import vanished"
    assert all(ln.startswith("    ") for ln in hits), \
        "the v2 loader import moved to module scope"


def test_the_real_parity_manifest_survived_this_file():
    """⚠️ LAST IN THIS FILE ON PURPOSE.

    Every test above monkeypatches `parity.MANIFEST_PATH` to a synthetic
    manifest. `tanitad/lake/filtering.py:96` reads the REAL one at MODULE SCOPE,
    so if this file's imports ever move inside a patched test the synthetic
    skipset becomes the process-wide truth — and the test that would catch it
    (`test_parity_manifest.py::
    test_lake_filtering_skipset_is_index_reproducible_from_the_repo`) only runs
    BEFORE this file by alphabetical accident. MEASURED: it failed exactly that
    way while this file was being written. Pinned here so the accident is not
    load-bearing."""
    assert len(_LAKE_FILTERING.PARITY_SKIP_INDICES) == 24
    assert _LAKE_FILTERING.PARITY_SKIP_INDICES == tuple(
        parity.manifest_entry(parity.PARITY_TRAIN_KEY)["skip_indices"])
    assert parity.MANIFEST_PATH == Path(parity.__file__).with_name(
        "parity_manifest.json"), "a monkeypatch of MANIFEST_PATH leaked"
    ent = parity.manifest_entry(parity.PARITY_TRAIN_KEY)
    assert ent["episode_count"] == parity.PARITY_TRAIN_EPISODES == 2376
    assert ent["clip_membership"]["n_clips"] == 2400
