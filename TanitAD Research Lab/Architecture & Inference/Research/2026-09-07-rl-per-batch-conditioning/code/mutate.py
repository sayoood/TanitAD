# -*- coding: utf-8 -*-
"""THE MUTATION PROOF for the per-batch conditioning contract.

Each mutation reintroduces ONE real defect in the REAL file and requires a RED; the
controls require a GREEN, because a check that always fails proves nothing either.

THE DEFECT IS TEMPORAL, so the proof is temporal. M1 restores the pre-change function
body VERBATIM -- the ``checked = {"done": False}`` latch and the two recording lines
that reference ``contract``, because reverting only the head would leave a NameError
and a RED caused by a NameError proves nothing about the defect. M1b then isolates the
once-only SEMANTICS alone, with every other line identical.

Both are proved twice: the suite must go RED, and a DIRECT probe drives the SAME
sample_fn with a good batch and then a bad one and prints what actually happens -- under
the live code a refusal on batch #2, under the restored latch a well-formed fan handed a
``v0=None`` forward, silently.

CPU-ONLY. Zero GPU: refcv5-v2 is live on the A40.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(r"C:\Users\Admin\tanitad-perbatch")
STACK = ROOT / "stack"
TARGET = STACK / "tanitad" / "rl" / "refc_adapter.py"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"

SUITES = ["tests/test_rl_per_batch_conditioning.py",
          "tests/test_rl_refc_adapter_robust.py",
          "tests/test_rl_channel_value_flow.py",
          "tests/test_rl_forward_keys_cover_signature.py",
          "tests/test_rl_channel_guard.py"]

# a crash must never leave the file mutated
_SAFE = ROOT / "refc_adapter.PRISTINE.py"
if not _SAFE.exists():
    _SAFE.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
PRISTINE = _SAFE.read_text(encoding="utf-8")
TARGET.write_text(PRISTINE, encoding="utf-8")

# --------------------------------------------------------------------------------------
LIVE_HEAD = (
    "    contract = ConditioningContract(model) if strict_conditioning else None\n"
    "\n"
    "    def sample_fn(batch, cfg_in: PostTrainConfig):\n"
    "        if contract is not None:\n"
    "            contract.check(batch)\n")

ONCE_ONLY_HEAD = (
    '    checked = {"done": False}\n'
    "\n"
    "    def sample_fn(batch, cfg_in: PostTrainConfig):\n"
    '        if strict_conditioning and not checked["done"]:\n'
    "            assert_conditioning(model, batch)\n"
    '            checked["done"] = True\n')

RECORD_TAIL = (
    '    # \u2b50 "it must be typed, and it is RECORDED" \u2014 now true of the object, '
    "not just of\n"
    "    # the docstring. A preflight/run record reads these instead of trusting an "
    "argv.\n"
    "    sample_fn.strict_conditioning = bool(strict_conditioning)\n"
    "    sample_fn.conditioning_contract = contract\n"
    "    return sample_fn\n")

RESOLVE_BLOCK = (
    '        cfg = getattr(self._model, "cfg", None)\n'
    "        if cfg is not self._cfg:\n"
    "            # raises ConditioningError on a missing .cfg or an unresolvable "
    "predicate,\n"
    "            # and does NOT record the cfg, so a declaration bug stays loud\n"
    "            self._req = conditioning_requirements(self._model)\n"
    "            self._cfg = cfg\n")

RESOLVE_MUTATED = (
    '        cfg = getattr(self._model, "cfg", None)\n'
    "        if cfg is not self._cfg:\n"
    "            self._cfg = cfg\n"
    "            self._req = conditioning_requirements(self._model)\n")

CHECK_CALL = ("        if contract is not None:\n"
              "            contract.check(batch)")
CHECK_ONCE = ("        if contract is not None and contract.batches == 0:\n"
              "            contract.check(batch)")

MUTATIONS = [
    ("M1", "THE DEFECT RESTORED, faithfully: the pre-change body verbatim",
     [(LIVE_HEAD, ONCE_ONLY_HEAD), (RECORD_TAIL, "    return sample_fn\n")], "RED"),

    ("M1b", "ONCE-ONLY SEMANTICS ALONE: every other line identical",
     [(CHECK_CALL, CHECK_ONCE)], "RED"),

    ("M2", "the guard becomes a latch in disguise: batches 2..N skip the check",
     [("        self.batches += 1\n        missing = _missing_channels",
       "        self.batches += 1\n        if self.batches > 1:\n"
       "            return self._req\n        missing = _missing_channels")], "RED"),

    ("M3", "OVER-assertion: every declared channel required regardless of the config",
     [("            self._required = tuple(k for k, needed in self._req.items() "
       "if needed)",
       "            self._required = tuple(self._req)")], "RED"),

    ("M4", "the cache swallows a declaration bug: cfg recorded BEFORE the resolve",
     [(RESOLVE_BLOCK, RESOLVE_MUTATED)], "RED"),

    ("M5", "the escape hatch is broken: strict_conditioning=False still asserts",
     [("    contract = ConditioningContract(model) if strict_conditioning else None",
       "    contract = ConditioningContract(model)")], "RED"),

    ("M6", "VACUITY: the check refuses every batch",
     [("        missing = _missing_channels(self._required, batch)",
       "        missing = list(self._required) or ['__always__']")], "RED"),

    ("M7", "the VALUE is not checked, only the KEY's presence is",
     [("    return [k for k in required if batch.get(k) is None]",
       "    return [k for k in required if k not in batch]")], "RED"),
]

# --------------------------------------------------------------------------------------
# the direct temporal probe: ONE sample_fn, a GOOD batch then a BAD one
# --------------------------------------------------------------------------------------
PROBE = r'''
import sys, torch
sys.path.insert(0, r"C:\Users\Admin\tanitad-perbatch\stack")
from tanitad.rl import refc_adapter as A
from tanitad.rl.config import PostTrainConfig
from tanitad.refs.refc_v3 import RefCV3Config

cfg = RefCV3Config()
cfg.ego_state_inject = True
cfg.core.anchors.v0_conditioned = True          # refcv4b / refcv5 argv

class M:
    def __init__(self):
        self.cfg = cfg
        self.calls = []
    def __call__(self, frames, **kw):
        self.calls.append(kw)
        z = torch.zeros(1, 2, 4, 2)
        return {"anchor_traj": z, "offset": z.clone()}

good = {"frames": torch.zeros(1, 4, 1, 8, 8), "v0": torch.full((1,), 7.5),
        "ego_state": torch.zeros(1, 5)}
bad = {k: v for k, v in good.items() if k != "v0"}

m = M()
fn = A.make_refc_sample_fn(m, PostTrainConfig())
t1, _, _ = fn(good, PostTrainConfig())
print("  BATCH 1 (v0 present): PASSED, fan %s, forward got v0=%.1f"
      % (tuple(t1.shape), float(m.calls[-1]["v0"][0])))
try:
    traj, logp, ctx = fn(bad, PostTrainConfig())
    print("  BATCH 2 (v0 DROPPED): NO REFUSAL. fan %s logp %s all-finite=%s"
          % (tuple(traj.shape), tuple(logp.shape), bool(torch.isfinite(traj).all())))
    print("  the forward was handed v0=%r -> refc.py:3097 v_ms=None -> refc.py:1722 "
          "every anchor rolled at the 10 m/s reference" % (m.calls[-1]["v0"],))
    print("  forward call count = %d (the defective batch WAS sampled from)"
          % len(m.calls))
    print("  => A WELL-FORMED FAN FROM A POLICY THAT NEVER EXISTED, SILENTLY.")
except A.ConditioningError as e:
    print("  BATCH 2 (v0 DROPPED): REFUSED -> %s" % str(e).splitlines()[0][:160])
    print("  forward call count = %d (the bad batch never reached it)" % len(m.calls))
'''


def _env():
    return {**os.environ, "PYTHONPATH": str(STACK), "CUDA_VISIBLE_DEVICES": "",
            "PYTHONIOENCODING": "utf-8"}


def _run_suites():
    p = subprocess.run([PY, "-m", "pytest", "-q", *SUITES], cwd=STACK,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=_env())
    lines = [ln for ln in (p.stdout or "").strip().splitlines() if ln.strip()]
    return ("GREEN" if p.returncode == 0 else "RED"), (lines[-1] if lines else "?")


def _run_probe():
    p = subprocess.run([PY, "-c", PROBE], cwd=STACK, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_env())
    return ((p.stdout or "") + (p.stderr or "")).rstrip()


def main() -> int:
    rows = []
    print("=" * 92)
    print("CONTROL, unmutated")
    got, tail = _run_suites()
    rows.append(("-", "CONTROL, unmutated", "GREEN", got, tail))
    print(f"  {got}  {tail}")
    print("\nTEMPORAL PROBE against the LIVE per-batch code:")
    print(_run_probe())

    for mid, desc, edits, want in MUTATIONS:
        print("=" * 92)
        print(f"{mid}  {desc}")
        text = PRISTINE
        for old, new in edits:
            assert old in text, f"{mid}: anchor not found -- the file moved:\n{old!r}"
            text = text.replace(old, new, 1)
        assert text != PRISTINE, f"{mid}: mutation was a no-op"
        TARGET.write_text(text, encoding="utf-8")
        got, tail = _run_suites()
        rows.append((mid, desc, want, got, tail))
        print(f"  expected {want}  got {got}   {tail}")
        if mid in ("M1", "M1b"):
            print(f"\n  TEMPORAL PROBE against {mid} (once-only restored):")
            print(_run_probe())
        TARGET.write_text(PRISTINE, encoding="utf-8")

    print("=" * 92)
    print("CONTROL, after restore")
    got, tail = _run_suites()
    rows.append(("-", "CONTROL, after restore", "GREEN", got, tail))
    print(f"  {got}  {tail}")

    print("\n" + "=" * 92)
    print(f"{'#':5} {'want':6} {'got':6} {'ok':4} mutation")
    ok = 0
    for mid, desc, want, got, tail in rows:
        good = want == got
        ok += good
        print(f"{mid:5} {want:6} {got:6} {'YES' if good else 'NO':4} {desc}  [{tail}]")
    print(f"\n{ok}/{len(rows)} expectations met")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
