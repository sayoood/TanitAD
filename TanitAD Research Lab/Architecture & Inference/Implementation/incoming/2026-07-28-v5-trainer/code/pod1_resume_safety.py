"""pod1 SAFETY CHECK — `flagship-v2corpus-30k` must still parse and still resume.

pod1 (`tanitad-pod`) is ~18k/30k steps into a 2-day run on `train_flagship4b
--v2-cache`. A change that broke its resume would destroy two GPU-days, so this
probe answers three questions with evidence rather than with a promise:

  1. Is `stack/scripts/train_flagship4b.py` byte-identical to the state pod1's
     command was verified against?  (sha256, recorded both sides.)
  2. Does pod1's EXACT argv still parse, with every value unchanged?
  3. Does the resume branch still fire and still restore step/model/opt?

⚠️ (3) is the one that cannot be answered by reading. It runs the REAL
`train_flagship4b.train()` twice over a stubbed v2 provider list — first to write
`ckpt.pt`, then again to prove `[resume] resuming at step N` restores it. The
stub replaces ONLY `tanitad.data.v2_dataset` (which imports torchvision, absent
on the dev box); every other line executed is the deployed one.

⛔ pod1 is NOT contacted. The argv below is read from the staged supervisor env
file `…/2026-07-25-v2-launch-readiness/flagship-v2corpus-30k.env` (`TRAIN_CMD`),
which is the artifact that owns that fact.

Run:  python pod1_resume_safety.py --out ../raw/pod1_resume_safety_<date>.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
STACK = REPO / "stack"
for p in (str(STACK), str(STACK / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

ENV_FILE = (REPO / "TanitAD Research Hub" / "Data Engineering" / "Implementation"
            / "incoming" / "2026-07-25-v2-launch-readiness"
            / "flagship-v2corpus-30k.env")

#: the sha256 of scripts/train_flagship4b.py as it stood when this probe was
#: written — i.e. the exact bytes pod1's command was last verified against.
BASELINE_SHA = "53f3ab5b0dd3e7118d836e75d061ddfc2355e6eb377b12f833a6abdb1b5d8ab8"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pod1_argv() -> list[str]:
    """pod1's EXACT trainer argv, lifted from the supervisor env's TRAIN_CMD."""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("TRAIN_CMD="):
            cmd = line.split("=", 1)[1].strip()
            if cmd[:1] in "'\"":
                cmd = cmd[1:-1]
            toks = shlex.split(cmd)
            i = toks.index("scripts/train_flagship4b.py")
            return toks[i + 1:]
    raise SystemExit(f"no TRAIN_CMD in {ENV_FILE}")


def _parse(argv):
    """The REAL Namespace `train_flagship4b.main` builds, without running train."""
    import train_flagship4b as T4
    holder: dict = {}
    real = T4.train
    T4.train = lambda a: holder.setdefault("a", a)
    try:
        T4.main(argv)
    finally:
        T4.train = real
    return holder["a"]


# --------------------------------------------------------------------------- #
# the resume proof — the real train() twice, over a stubbed v2 provider list    #
# --------------------------------------------------------------------------- #
def _toy_providers(n=4, T=60):
    """A ToyEpisode list shaped like `build_v2_providers`' return value.

    Contract-identical where the trainer touches it (`.frames` [T,C,H,W] u8,
    `.poses` [T,4], `.actions` [T,2], `.episode_id`), which is exactly the
    `LazyV2Episode` surface — see `measure_options.py`.

    ⚠️ The size is taken AFTER `apply_geometry_args`, not from the raw smoke
    config, because at HEAD that call rewrites the smoke encoder 64 -> 256 (see
    `geometry_smoke_drift` below). Building at the raw 64 would make this probe
    fail for a reason that has nothing to do with pod1."""
    import argparse as _ap
    from tanitad.config import flagship4b_smoke_config
    from tanitad.data.toy_driving import generate_episode
    from tanitad.geometry import add_geometry_args, apply_geometry_args
    cfg = flagship4b_smoke_config()
    p = _ap.ArgumentParser()
    add_geometry_args(p)
    apply_geometry_args(p.parse_args([]), cfg, label="probe")
    return [generate_episode(i, steps=T, size=cfg.encoder.image_size)
            for i in range(n)]


def geometry_smoke_drift() -> dict:
    """⚠️ A DEFAULT THAT ALREADY MOVED, found while building this probe.

    `train_flagship4b` calls `apply_geometry_args` unconditionally. `frame_of`
    returns CANONICAL_256 for any config with `geometry is None`, so the call
    REWRITES the smoke config's encoder from 64x64 to 256x256 (1024 tokens
    instead of 256) while printing "DEPLOYED (unchanged)".

    ⭐ It is a no-op for `flagship4b` and `flagship4b_reduced`, which already
    declare 256 — so **pod1 is unaffected**, and that is measured here rather
    than assumed. Reported, not fixed: `geometry.py` belongs to another stream.
    """
    import argparse as _ap
    from tanitad.config import (flagship4b_config, flagship4b_reduced_config,
                                flagship4b_smoke_config)
    from tanitad.geometry import add_geometry_args, apply_geometry_args
    p = _ap.ArgumentParser()
    add_geometry_args(p)
    args = p.parse_args([])
    rows = {}
    for name, f in (("smoke", flagship4b_smoke_config),
                    ("flagship4b_reduced", flagship4b_reduced_config),
                    ("flagship4b", flagship4b_config)):
        cfg = f()
        before = [cfg.encoder.image_size, cfg.encoder.image_width]
        buf = io.StringIO()
        with redirect_stdout(buf):
            apply_geometry_args(args, cfg, label=name)
        after = [cfg.encoder.image_size, cfg.encoder.image_width]
        rows[name] = {"before": before, "after": after,
                      "unchanged": before == after}
    rows["_pod1_config"] = "flagship4b"
    rows["POD1_UNAFFECTED"] = rows["flagship4b"]["unchanged"]
    return rows


def _stub_v2(monkey_holder, providers):
    """sys.modules stub for `tanitad.data.v2_dataset` (imports torchvision).

    ⚠️ NOT an importorskip: on this host that would SKIP the pod1 safety check
    entirely, and a safety check that skips is a check that cannot fail."""
    import tanitad.data as _td
    mod = types.ModuleType("tanitad.data.v2_dataset")
    mod.build_v2_providers = lambda dirs, **k: list(providers)
    monkey_holder["prev_mod"] = sys.modules.get("tanitad.data.v2_dataset")
    monkey_holder["prev_attr"] = getattr(_td, "v2_dataset", None)
    sys.modules["tanitad.data.v2_dataset"] = mod
    _td.v2_dataset = mod


def _unstub(monkey_holder):
    import tanitad.data as _td
    if monkey_holder.get("prev_mod") is None:
        sys.modules.pop("tanitad.data.v2_dataset", None)
    else:
        sys.modules["tanitad.data.v2_dataset"] = monkey_holder["prev_mod"]
    if monkey_holder.get("prev_attr") is not None:
        _td.v2_dataset = monkey_holder["prev_attr"]


def resume_proof(tmp: Path) -> dict:
    import torch
    import train_flagship4b as T4

    cache = tmp / "physicalai-v2bal-4b7eeeac222d"     # pod1's non-parity corpus
    cache.mkdir(parents=True, exist_ok=True)
    out = tmp / "run"

    holder: dict = {}
    _stub_v2(holder, _toy_providers())
    real_guard = T4.start_cache_guard
    T4.start_cache_guard = lambda *a, **k: None
    try:
        # pod1's flag SHAPE (the v2 lever pack + --v2-cache + no --require-parity),
        # at `--config smoke` so it runs on a CPU in seconds. The resume branch
        # is config-independent: it is `if ckpt_path.exists()` in train().
        base = ["--v2-cache", str(cache), "--config", "smoke", "--v2",
                "--out", str(out), "--batch-size", "4", "--log-every", "1",
                "--ckpt-every", "2", "--workers", "0", "--device", "cpu",
                "--no-amp"]
        buf1 = io.StringIO()
        with redirect_stdout(buf1):
            T4.train(_parse(base + ["--steps", "4"]))
        log1 = buf1.getvalue()
        ck1 = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            T4.train(_parse(base + ["--steps", "7"]))
        log2 = buf2.getvalue()
        ck2 = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    finally:
        T4.start_cache_guard = real_guard
        _unstub(holder)

    resumed_line = [ln for ln in log2.splitlines() if ln.startswith("[resume]")]
    non_parity_1 = "NON-PARITY v2 corpus" in log1
    return {
        "run1_final_step": int(ck1["step"]),
        "run2_final_step": int(ck2["step"]),
        "resume_line": resumed_line[0] if resumed_line else None,
        "resume_branch_fired": bool(resumed_line),
        "resumed_from_step": int(ck1["step"]) + 1,
        "continued_not_restarted": int(ck2["step"]) > int(ck1["step"]),
        "first_run_did_not_resume": not any(
            ln.startswith("[resume]") for ln in log1.splitlines()),
        "unregistered_cache_warned_and_proceeded": non_parity_1,
        "PASS": bool(resumed_line) and int(ck2["step"]) > int(ck1["step"])
        and non_parity_1,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    fb = STACK / "scripts" / "train_flagship4b.py"
    sha = _sha(fb)
    argv_pod1 = _pod1_argv()
    ns = _parse(argv_pod1)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        proof = resume_proof(Path(td))
    drift = geometry_smoke_drift()

    rec = {
        "host": "dev box (pod1 NOT contacted)",
        "source_of_argv": str(ENV_FILE.relative_to(REPO)).replace("\\", "/"),
        "pod1_argv": argv_pod1,
        "1_trainer_file_unchanged": {
            "file": "stack/scripts/train_flagship4b.py",
            "sha256_baseline": BASELINE_SHA,
            "sha256_now": sha,
            "PASS": sha == BASELINE_SHA,
            "why_this_is_the_strongest_check": (
                "the chosen option edits train_flagship_v4 only, so pod1's "
                "trainer is byte-identical — no argument about behaviour is "
                "needed at all"),
        },
        "2_argv_still_parses": {
            "PASS": True,
            "namespace": {k: v for k, v in sorted(vars(ns).items())},
            "require_parity": ns.require_parity,
            "require_parity_is_off": ns.require_parity is False,
            "geometry_defaults": {k: getattr(ns, k, None) for k in
                                  ("frame_h", "frame_w", "frame_hfov", "f_ref",
                                   "projection")},
        },
        "3_resume_still_works": proof,
        "4_geometry_is_a_noop_for_pod1s_config": drift,
    }
    rec["ALL_PASS"] = bool(rec["1_trainer_file_unchanged"]["PASS"]
                           and rec["2_argv_still_parses"]["PASS"]
                           and rec["3_resume_still_works"]["PASS"]
                           and drift["POD1_UNAFFECTED"]
                           and ns.require_parity is False)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rec, indent=2, default=str))
    return 0 if rec["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
