"""Source-level MUTATION check of the new tests (deliberate regressions that must go RED).

For each mutant: copy the patched source into `tree_patched`, edit ONE anchor (EOL-aware, must match exactly once),
run the two new test files, record failed/passed, then RESTORE the file byte-for-byte from `out/` (sha256 checked).
A mutant that leaves the tests GREEN is a hole in the tests and is reported as SURVIVED.
One pytest process at a time.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _edit as E  # noqa: E402

PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TESTS = ["tests/test_refcv8_plan_vmax_clamp.py", "tests/test_refcv8_v7f_equivalence.py"]
RAW = E.OUT / "Project Steering/audits/2026-10-18-refcv8-v7f-fixes/raw"

# (id, file, old, new)
MUTANTS = [
    ("R1-M1 E9 clamp call removed", "stack/tanitad/refs/refc_v3.py",
     "            traj, _cd = v6sel.cap_plan_to_limit(\n                traj, _pl, horizons=_dec.anchor_horizons, tick_s=_dec.anchor_dt, v0=v0,\n"
     "                brake_ms2=float(_dec.sel.accel_max))\n            out.update(_cd)\n",
     "            _cd = {}\n"),
    ("R1-M2 limit export removed", "stack/tanitad/refs/refc_v3.py",
     '                cache["plan_v_lim_ms"] = v_lim.detach()          # the E9 block clamps the EMITTED plan to it\n',
     "                pass\n"),
    ("R1-M3 inference-only guard removed", "stack/tanitad/refs/refc_v3.py",
     'if bool(getattr(self.cfg, "plan_vmax_clamp", False)) and not self.training:',
     'if bool(getattr(self.cfg, "plan_vmax_clamp", False)):'),
    ("R1-M4 trainer pin does not set the cfg", "stack/scripts/refc_v3_train.py",
     "        cfg.plan_vmax_clamp = True\n", "        pass\n"),
    ("R1-M5 G-DVB entry made a no-op", "stack/tanitad/train/declared_vs_built.py",
     'lambda a: bool(_a(a, "plan_vmax_clamp", False)),', "lambda a: None,"),
    ("R1-M6 no-channel build guard removed", "stack/tanitad/refs/refc_v3.py",
     'if not getattr(cfg, "max_speed_onehot_v6", False):\n                raise ValueError(\n                    "[refcv8] plan_vmax_clamp without',
     'if False:\n                raise ValueError(\n                    "[refcv8] plan_vmax_clamp without'),
    ("R1-M7 typed dt=0.1 instead of the derived grid", "stack/tanitad/refs/refcv6_selection.py",
     "    dt = torch.tensor(dts, device=traj.device, dtype=traj.dtype)                      # [S]\n",
     "    dt = torch.full((len(dts),), 0.1, device=traj.device, dtype=traj.dtype)\n"),
    ("R3-M1 tactical behaviour loss dropped from the total", "stack/scripts/refc_v3_train.py",
     "        loss = loss + _w_t6 * _t6_loss\n", "        loss = loss + 0.0 * _t6_loss\n"),
    ("R4-M1 r8 feed (tactical posterior -> planner) severed", "stack/tanitad/refs/refc_v3.py",
     '                **({} if r8_feed is None else {"r8": r8_feed}),\n', "                **({}),\n"),
    ("R4-M2 tac8 prior port severed", "stack/tanitad/refs/refc.py",
     "            terms.append(_tile(self.tac8_lat_to_anchor(tac_lat_prior)))\n", "            pass\n"),
    ("R6-M1 strategic ctx handed to the decoder again", "stack/tanitad/refs/refc.py",
     "        ctx_dec = None if self.cfg.no_strategic else ctx\n", "        ctx_dec = ctx\n"),
    ("R2-M1 nav token dropped before the forward", "stack/scripts/refc_v3_train.py",
     "    out = _fwd(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan,",
     "    out = _fwd(frames, nav_cmd=None, v0=v0, steps=steps, lan=lan,"),
    ("R2-M2 nav -> tactical port removed", "stack/tanitad/refs/refc_v3.py",
     "                z_tac_raw = z_tac_raw + nav_t                     # -> tactical\n", "                pass\n"),
]


def run_tests() -> dict:
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", "-x", *TESTS],
                       cwd=str(E.DST_TREE / "stack"), capture_output=True, text=True, timeout=900)
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-4:]
    failed = [ln.split("::", 1)[-1][:90] for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
    return {"rc": p.returncode, "tail": tail[-1] if tail else "", "first_failed": failed[:3]}


def main(only=None) -> None:
    rows = []
    base = run_tests()
    rows.append({"id": "BASELINE (unmutated patched tree)", **base})
    print(rows[-1], flush=True)
    for mid, rel, old, new in MUTANTS:
        if only and not mid.startswith(only):
            continue
        good = (E.DST_TREE / rel).read_bytes()
        sha = hashlib.sha256(good).hexdigest()
        target = E.DST_TREE / rel
        try:
            target.write_bytes(E.edit(good, old, new, mode="replace"))
            r = run_tests()
        finally:
            target.write_bytes(good)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == sha, "restore failed"
        r["verdict"] = "KILLED" if r["rc"] != 0 else "SURVIVED"
        rows.append({"id": mid, **r})
        print(rows[-1], flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "mutation_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
