"""MUTATION PROOF for `test_trunk_dedup_frames.py` (--trunk-dedup-frames).

Anchors are WHOLE LINES (or a WINDOW of consecutive whole lines), matched by equality.

  D1  the overlap check always says "same" (non-overlapping frames get reused)
  D2  the overlap check compares the WRONG frames (never dedups)
  D3  a reused row maps its first K-1 slots to the wrong earlier frames
  D4  a reused row never computes its NEWEST frame
  D5  STAMPED BUT NOT APPLIED: forward_features never takes the dedup path
  D6  the trunk dedups with a BatchNorm that trains
  D7  the trainer accepts --trunk-dedup-frames without --trunk-frozen-bn
  D8  the trainer does not pin the flag into the config
  D9  build_encoder does not pass dedup_frames to the trunk
  D10 config.json stamps a constant
  D11 the run log never resets its counter (rows accumulate)
  D12 the run log reads the LAST call instead of the step's calls
  D13 the gather order is reversed within each row
  D14 the refc-trunk refusal ignores --trunk-dedup-frames
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
TESTS = REPO / "stack" / "tests" / "test_trunk_dedup_frames.py"
TT = REPO / "stack" / "tanitad" / "models" / "timm_trunk.py"
RC = REPO / "stack" / "tanitad" / "refs" / "refc.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT = pathlib.Path("C:/Users/Admin/qland/work/f_dedup/mutation_proof_trunk_dedup_frames.json")

# chr() on purpose: a heredoc eats one backslash level; there is no escape here to eat.
EOL_CHARS = chr(13) + chr(10)

MUTATIONS = [
    ("D1_overlap_check_always_same", TT,
     '        same = ((per[1:, :k - 1] == per[:-1, 1:]).flatten(1).all(dim=1).tolist()',
     '        same = ([True] * (n - 1)'),
    ("D2_overlap_check_wrong_frames", TT,
     '        same = ((per[1:, :k - 1] == per[:-1, 1:]).flatten(1).all(dim=1).tolist()',
     '        same = ((per[1:, 1:] == per[:-1, :k - 1]).flatten(1).all(dim=1).tolist()'),
    ("D3_reused_slots_point_at_the_wrong_frames", TT,
     '                slot[i][:k - 1] = slot[i - 1][1:]',
     '                slot[i][:k - 1] = slot[i - 1][:k - 1]'),
    ("D4_newest_frame_never_computed", TT,
     '                first = k - 1',
     '                first = k'),
    ("D5_STAMPED_but_NOT_APPLIED", TT,
     '            if self.memory_levers.get("dedup_frames"):',
     '            if False:'),
    ("D6_trunk_dedups_a_training_bn", TT,
     ('        if bool(getattr(self.cfg, "dedup_frames", False)):',
      '            if not self.memory_levers["frozen_bn"]:'),
     '            if False:'),
    ("D7_trainer_accepts_dedup_without_frozen", TRAIN,
     '    if cfg.core.encoder.trunk_dedup_frames and not cfg.core.encoder.trunk_frozen_bn:',
     '    if False:'),
    ("D8_trainer_does_not_pin_the_flag", TRAIN,
     '    cfg.core.encoder.trunk_dedup_frames = bool(getattr(args, "trunk_dedup_frames", False))',
     '    cfg.core.encoder.trunk_dedup_frames = False'),
    ("D9_build_encoder_drops_the_flag", RC,
     '            dedup_frames=bool(getattr(cfg, "trunk_dedup_frames", False)),',
     '            dedup_frames=False,'),
    ("D10_stamp_is_a_constant", TRAIN,
     '        "trunk_dedup_frames": bool(getattr(core.encoder, "trunk_dedup_frames", False)),',
     '        "trunk_dedup_frames": False,'),
    ("D11_log_counter_never_reset", TRAIN,
     '                _enc.dedup_counts = [0, 0]',
     '                pass'),
    ("D12_log_reads_the_last_call_only", TRAIN,
     '            _ddc = getattr(_enc, "dedup_counts", None)',
     '            _ddc = getattr(_enc, "last_dedup", None)'),
    ("D13_gather_order_reversed_within_a_row", TT,
     '        flat = torch.tensor([u for r in slot for u in r], device=f16u.device)',
     '        flat = torch.tensor([u for r in slot for u in r[::-1]], device=f16u.device)'),
    ("D14_refc_refusal_ignores_dedup", TRAIN,
     '            or cfg.core.encoder.trunk_dedup_frames',
     '            or False'),
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
    res = {"_what": "mutation proof that the --trunk-dedup-frames tests can FAIL",
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
        f"MUTATION-PROVEN -- all {n}/{len(MUTATIONS)} arms RED, including the dedup "
        "stamped but not applied and the slot map off by one; all files restored "
        "byte-identical and the final clean run is green."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    _write(res)
    print(res["_VERDICT"])
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
