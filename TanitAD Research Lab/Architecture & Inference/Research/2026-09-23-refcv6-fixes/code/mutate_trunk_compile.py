"""MUTATION PROOF for `test_trunk_compile.py` (--trunk-compile).

Anchors are WHOLE LINES (or a WINDOW of consecutive whole lines), matched by equality.

  C1  STAMPED BUT NOT APPLIED: `_backbone` always runs the eager `self.net`
  C2  the compiled wrapper is REGISTERED as a submodule (state_dict keys change)
  C3  the backend knob is ignored (always Inductor)
  C4  the trunk never records the backend in memory_levers
  C5  the trainer does not pin --trunk-compile into the config
  C6  build_encoder does not pass compile_backbone to the trunk
  C7  config.json stamps a constant
  C8  the refc-trunk refusal ignores --trunk-compile
  C9  the donated-buffer switch is removed (S11's step-1 crash on Thor)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path("D:/Projects/TanitAD")
TRAIN = REPO / "stack" / "scripts" / "refc_v3_train.py"
TESTS = REPO / "stack" / "tests" / "test_trunk_compile.py"
TT = REPO / "stack" / "tanitad" / "models" / "timm_trunk.py"
RC = REPO / "stack" / "tanitad" / "refs" / "refc.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT = pathlib.Path("C:/Users/Admin/qland/work/f_dedup/mutation_proof_trunk_compile.json")

# chr() on purpose: a heredoc eats one backslash level; there is no escape here to eat.
EOL_CHARS = chr(13) + chr(10)

MUTATIONS = [
    ("C1_STAMPED_but_NOT_APPLIED", TT,
     '        net = self._net_fn if getattr(self, "_net_fn", None) is not None else self.net',
     '        net = self.net'),
    ("C2_wrapper_registered_as_a_submodule", TT,
     '            object.__setattr__(self, "_net_fn", torch.compile(self.net, backend=_be))',
     '            self._net_fn = torch.compile(self.net, backend=_be)'),
    ("C3_backend_knob_ignored", TT,
     '            object.__setattr__(self, "_net_fn", torch.compile(self.net, backend=_be))',
     '            object.__setattr__(self, "_net_fn", torch.compile(self.net))'),
    ("C4_backend_never_recorded", TT,
     '            self.memory_levers["compile"] = _be',
     '            pass'),
    ("C5_trainer_does_not_pin_the_flag", TRAIN,
     '    cfg.core.encoder.trunk_compile = bool(getattr(args, "trunk_compile", False))',
     '    cfg.core.encoder.trunk_compile = False'),
    ("C6_build_encoder_drops_the_flag", RC,
     '            compile_backbone=bool(getattr(cfg, "trunk_compile", False)))',
     '            compile_backbone=False)'),
    ("C7_stamp_is_a_constant", TRAIN,
     '        "trunk_compile": bool(getattr(core.encoder, "trunk_compile", False)),',
     '        "trunk_compile": False,'),
    ("C8_refc_refusal_ignores_compile", TRAIN,
     '            or cfg.core.encoder.trunk_compile) and _trunk != "timm":',
     '            or False) and _trunk != "timm":'),
    ("C9_donated_buffer_switch_removed", TT,
     '            _fconfig.donated_buffer = False',
     '            pass'),
]


def run_tests():
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        env={**{k: v for k, v in os.environ.items()
                if k in ("USERNAME", "USERPROFILE", "HOME", "HOMEDRIVE",
                         "HOMEPATH", "TEMP", "TMP", "SYSTEMROOT", "COMSPEC")},
             "PYTHONPATH": str(REPO / "stack"), "PATH": os.environ["PATH"],
             "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2",
             "CUDA_VISIBLE_DEVICES": ""})
    out = (r.stdout or "") + (r.stderr or "")
    failed, seen = [], set()
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("FAILED") and "::" in s:
            nm = s.split("::")[-1].split()[0].strip()
            if nm not in seen:
                seen.add(nm)
                failed.append(nm)
    return r.returncode, failed, (not out.strip()), out[-500:]


def _write(res):
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    files = {TRAIN, TESTS, TT, RC}
    orig = {p: p.read_bytes() for p in files}
    md5 = {p: hashlib.md5(b).hexdigest() for p, b in orig.items()}
    res = {"_what": "mutation proof that the --trunk-compile tests can FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "targets": {str(p.relative_to(REPO)): m for p, m in md5.items()}, "arms": []}
    try:
        rc, failed, dead, tail = run_tests()
        res["baseline"] = {"rc": rc, "failed": failed}
        print(f"BASELINE rc={rc} failed={failed}")
        if dead:
            print("ZZABORT baseline produced NO OUTPUT -- INCONCLUSIVE, not green")
            return 6
        if rc != 0:
            print("ZZABORT baseline not green\n" + tail)
            return 3
        for name, target, old, new in MUTATIONS:
            txt = orig[target].decode("utf-8")
            lines = txt.splitlines(keepends=True)
            win = (old,) if isinstance(old, str) else tuple(old)
            bare = [ln.rstrip(EOL_CHARS) for ln in lines]
            # a WINDOW of consecutive whole lines; the LAST line of the window is replaced
            hits = [i + len(win) - 1 for i in range(len(bare) - len(win) + 1)
                    if tuple(bare[i:i + len(win)]) == win]
            if len(hits) != 1:
                print(f"ZZABORT {name}: anchor matches {len(hits)} LINES in "
                      f"{target.name} -- the proof is INVALID, not weak")
                res["_VERDICT"] = "INVALID -- an arm could not be applied."
                res["arms"].append({"arm": name, "error": "anchor not unique",
                                    "occurrences": len(hits)})
                _write(res)
                return 5
            try:
                eol = lines[hits[0]][len(lines[hits[0]].rstrip(EOL_CHARS)):]
                lines[hits[0]] = new + eol
                target.write_bytes("".join(lines).encode("utf-8"))
                rc_m, failed_m, dead_m, tail_m = run_tests()
            finally:
                target.write_bytes(orig[target])
            caught = bool(rc_m != 0 and failed_m and not dead_m)
            print(f"  {name:<44} rc={rc_m} RED={caught} caught_by={failed_m[:3]}")
            res["arms"].append({"arm": name, "file": str(target.relative_to(REPO)),
                                "rc": rc_m, "went_RED": caught, "caught_by": failed_m,
                                "tail": None if caught else tail_m})
    finally:
        for p, b in orig.items():
            p.write_bytes(b)
    back = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in files}
    res["restored_ok"] = all(back[p] == md5[p] for p in files)
    assert res["restored_ok"], "ZZABORT targets NOT restored"
    rc_f, failed_f, _, _ = run_tests()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f}
    n = sum(1 for a in res["arms"] if a.get("went_RED"))
    res["arms_caught"], res["arms_total"] = n, len(MUTATIONS)
    res["_VERDICT"] = (
        f"MUTATION-PROVEN -- all {n}/{len(MUTATIONS)} arms RED, including the compile "
        "stamped but not applied and the wrapper registered as a submodule; all files restored "
        "byte-identical and the final clean run is green."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    _write(res)
    print(res["_VERDICT"])
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
