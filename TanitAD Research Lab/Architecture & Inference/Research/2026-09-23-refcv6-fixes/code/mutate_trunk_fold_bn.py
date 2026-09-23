"""MUTATION PROOF for the frozen-BN FOLD tests in `test_trunk_speed_levers.py`.

Anchors are WHOLE LINES, matched by equality over lines (CRLF- and indent-safe).

  F1  the fold drops the running-mean term              (a DIFFERENT function)
  F2  the fold drops eps                                (the subtlest wrong fold: 2.2e-4 rel)
  F3  the BN is NOT replaced -> applied twice          (a DIFFERENT function)
  F4  the conv is NOT folded but the BN is removed      (BN silently dropped)
  F5  STAMPED BUT NOT APPLIED: the right count recorded, the net left unfolded
  F6  the trunk accepts fold_bn on a BN that trains
  F7  the trainer accepts --trunk-fold-bn without --trunk-frozen-bn
  F8  the trainer does not pin --trunk-fold-bn into the config
  F9  build_encoder does not pass fold_bn to the trunk
  F10 config.json stamps a constant instead of the config value
  F11 the refc-trunk refusal ignores --trunk-fold-bn
  F12 the shortcut (downsample) BNs are silently left unfolded
  F13 config.json reads the levers off the WRONG module (the record goes blind)
  F14 config.json drops the built-trunk record
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
TESTS = REPO / "stack" / "tests" / "test_trunk_speed_levers.py"
TT = REPO / "stack" / "tanitad" / "models" / "timm_trunk.py"
RC = REPO / "stack" / "tanitad" / "refs" / "refc.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT = pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_trunk_fold_bn.json")

# chr() on purpose: a heredoc eats one backslash level; there is no escape here to eat.
EOL_CHARS = chr(13) + chr(10)

MUTATIONS = [
    ("F1_fold_drops_the_running_mean", TT,
     '            bias = _b.bias - _b.running_mean * s',
     '            bias = _b.bias + 0.0 * s'),
    ("F2_fold_drops_eps", TT,
     '            s = _b.weight / torch.sqrt(_b.running_var + _b.eps)',
     '            s = _b.weight / torch.sqrt(_b.running_var)'),
    ("F3_bn_not_replaced_applied_twice", TT,
     '        bn.forward = (lambda x: x)   # noqa: E731 -- its affine now lives in the conv',
     '        pass'),
    ("F4_conv_not_folded_bn_dropped", TT,
     '        conv.forward = _fwd',
     '        pass'),
    ("F5_STAMPED_but_NOT_APPLIED", TT,
     '            self.memory_levers["bn_folded"] = _fold_frozen_bn_(self.net)',
     '            self.memory_levers["bn_folded"] = len(_conv_bn_pairs(self.net))'),
    # the same `if not ...frozen_bn` line guards chunk_ckpt too: anchor on the WINDOW
    ("F6_trunk_folds_a_training_bn", TT,
     ('        if bool(getattr(self.cfg, "fold_bn", False)):',
      '            if not self.memory_levers["frozen_bn"]:'),
     '            if False:'),
    ("F7_trainer_accepts_fold_without_frozen", TRAIN,
     '    if cfg.core.encoder.trunk_fold_bn and not cfg.core.encoder.trunk_frozen_bn:',
     '    if False:'),
    ("F8_trainer_does_not_pin_the_flag", TRAIN,
     '    cfg.core.encoder.trunk_fold_bn = bool(getattr(args, "trunk_fold_bn", False))',
     '    cfg.core.encoder.trunk_fold_bn = False'),
    ("F9_build_encoder_drops_the_flag", RC,
     '            fold_bn=bool(getattr(cfg, "trunk_fold_bn", False)))',
     '            fold_bn=False)'),
    ("F10_stamp_is_a_constant", TRAIN,
     '        "trunk_fold_bn": bool(getattr(core.encoder, "trunk_fold_bn", False)),',
     '        "trunk_fold_bn": False,'),
    ("F11_refc_refusal_ignores_fold", TRAIN,
     '            or cfg.core.encoder.trunk_fold_bn) and _trunk != "timm":',
     '            or False) and _trunk != "timm":'),
    ("F12_shortcut_bns_left_unfolded", TT,
     '        if isinstance(mod, nn.Sequential):',
     '        if False:'),
    ("F13_record_reads_the_wrong_module", TRAIN,
     '    enc = getattr(getattr(model, "core", None), "encoder", None)',
     '    enc = getattr(model, "encoder", None)'),
    ("F14_record_dropped", TRAIN,
     '        "trunk_memory_levers": _trunk_levers_built(model),',
     '        "trunk_memory_levers": None,'),
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
    res = {"_what": "mutation proof that the frozen-BN FOLD tests can FAIL",
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
        f"MUTATION-PROVEN -- all {n}/{len(MUTATIONS)} arms RED, including the fold "
        "stamped but not applied and the eps-only wrong fold; all files restored "
        "byte-identical and the final clean run is green."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    _write(res)
    print(res["_VERDICT"])
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
