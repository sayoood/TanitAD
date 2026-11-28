# -*- coding: utf-8 -*-
"""THE BIT-IDENTICAL OFF PROOF for --max-speed-input (E16), with its control.

WHAT IS COMPARED. The PRE-PATCH `refc_v3.py` (extracted from git HEAD^ for this
run) and the PATCHED one, both built on the SAME fixed seed with the flag at its
default, then run forward on the SAME frames. Every state_dict tensor and every
output tensor must be bit-for-bit equal. This is not "the tests still pass": it
is the live-run obligation, because a 40,284-step training resumes through this
file.

WHAT THE MUTATION CONTROL PROVES, AND WHAT IT DOES NOT.
  * IT PROVES the comparator has TEETH -- that this equality CAN go red. An
    assertion that cannot fail measures nothing.
  * IT DOES NOT PROVE that a zero-init survived a mutation, and this script does
    not claim it. With the flag OFF **nothing is constructed and nothing is
    fed**, so the identity is STRUCTURAL, not a cancellation: there is no
    zero-init doing work here to be mutated. The zero-init claim belongs to the
    ON path (emission bit-inert at step 0) and is tested separately.

ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys

import torch

REPO_RUN = "C:/Users/Admin/tanitad-wt"          # the runnable copy
REPO_GIT = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
sys.path.insert(0, os.path.join(REPO_RUN, "stack"))
HERE = os.path.dirname(os.path.abspath(__file__))
PATCHED = os.path.join(REPO_RUN, "stack", "tanitad", "refs", "refc_v3.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_head(dst: str) -> str:
    """The PRE-PATCH file, from git itself -- never a copy I made earlier."""
    out = subprocess.run(
        ["git", "-C", REPO_GIT, "show", "HEAD:stack/tanitad/refs/refc_v3.py"],
        capture_output=True, check=True)
    io.open(dst, "wb").write(out.stdout)
    return dst


def build_and_run(mod, seed: int = 0):
    """Same seed, same frames, same call -- on whichever module is handed in."""
    torch.manual_seed(seed)
    cfg = mod.refc_v3_smoke_config(hier=True)
    torch.manual_seed(seed)
    m = mod.RefCV3Model(cfg).eval()
    g = torch.Generator().manual_seed(1234)
    h, w = cfg.core.encoder.image_hw()
    f = torch.rand(2, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                   generator=g)
    with torch.no_grad():
        out = m(f, nav_cmd=torch.tensor([0, 1]), v0=torch.tensor([3.0, 7.0]),
                steps=2)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    ot = {k: v.detach().clone() for k, v in out.items()
          if isinstance(v, torch.Tensor)}
    return sd, ot


def compare(a, b, what):
    ka, kb = set(a), set(b)
    if ka != kb:
        return False, (f"{what}: KEY SET DIFFERS "
                       f"(+{sorted(kb - ka)[:6]} -{sorted(ka - kb)[:6]})")
    for k in sorted(ka):
        if a[k].shape != b[k].shape:
            return False, f"{what}: {k} shape {a[k].shape} vs {b[k].shape}"
        if not torch.equal(a[k], b[k]):
            return False, f"{what}: {k} NOT bit-identical"
    return True, f"{what}: {len(ka)} tensors bit-identical"


def run_pair(patched_path, tag):
    head_path = extract_head(os.path.join(HERE, "_refc_v3_HEAD.py"))
    head = load(f"v3_head_{tag}", head_path)
    new = load(f"v3_new_{tag}", patched_path)
    sd_h, ot_h = build_and_run(head)
    sd_n, ot_n = build_and_run(new)
    ok1, m1 = compare(sd_h, sd_n, "state_dict")
    ok2, m2 = compare(ot_h, ot_n, "forward outputs")
    return (ok1 and ok2), [m1, m2]


print("=" * 78)
print("BASELINE -- the landed patch, flag at its default (OFF)")
print("=" * 78)
ok, msgs = run_pair(PATCHED, "base")
for m in msgs:
    print("   " + m)
print("   RESULT:", "BIT-IDENTICAL" if ok else "DIFFERS")
baseline_ok = ok

print()
print("=" * 78)
print("MUTATION CONTROL -- the conditioner built UNCONDITIONALLY")
print("(exactly the D-ROLL-1 construction that made every banked v7.0")
print(" checkpoint unrollable). The comparison MUST go red.")
print("=" * 78)
src = io.open(PATCHED, encoding="utf-8").read()
NEEDLE = "        if cfg.max_speed_input:\n" \
         "            self.max_speed_cond = msi.MaxSpeedConditioner(\n"
assert src.count(NEEDLE) == 1, "mutation anchor not unique -- refusing"
mutated = src.replace(
    NEEDLE,
    "        if True:   # <-- MUTATION: the D-ROLL-1 unconditional build\n"
    "            self.max_speed_cond = msi.MaxSpeedConditioner(\n", 1)
mut_path = os.path.join(HERE, "_refc_v3_MUTATED.py")
io.open(mut_path, "w", encoding="utf-8", newline="").write(mutated)
ok_m, msgs_m = run_pair(mut_path, "mut")
for m in msgs_m:
    print("   " + m)
print("   RESULT:", "BIT-IDENTICAL (BAD -- the check has no teeth)"
      if ok_m else "DIFFERS (GOOD -- the check can fail)")

print()
print("=" * 78)
verdict = baseline_ok and not ok_m
print("OFF-PARITY PROOF:", "PASS" if verdict else "FAIL")
print("  baseline bit-identical :", baseline_ok)
print("  mutation goes red      :", not ok_m)
print()
print("  WHAT THE MUTATION PROVES: the comparator has teeth -- this equality")
print("  CAN fail, so the baseline PASS is a measurement.")
print("  WHAT IT DOES NOT PROVE: that a zero-init survived a mutation. With the")
print("  flag OFF nothing is constructed and nothing is fed, so the identity is")
print("  STRUCTURAL, not a cancellation. Claiming otherwise would be a false")
print("  strength claim.")
for p in (os.path.join(HERE, "_refc_v3_MUTATED.py"),
          os.path.join(HERE, "_refc_v3_HEAD.py")):
    try:
        os.remove(p)
    except OSError:
        pass
sys.exit(0 if verdict else 1)
