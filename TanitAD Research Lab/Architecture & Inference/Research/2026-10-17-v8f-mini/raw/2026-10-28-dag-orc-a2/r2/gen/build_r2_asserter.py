#!/usr/bin/env python3
"""Derive box2_ship/gate_dagoaccorc_r2_assert.py from r1's gate asserter (exact-count insertions only; r1's file, installed as C:/Users/Admin/box2_ship/gate_dagoaccorc_assert.py, is never touched):

  ADDED (r2), from the Master Mind's specification of the landed trainer fix (E:/Projects/TanitAD-artifacts/dagland/compose_fix/stage: v8f_mini_data.py `EGO_PERTURB_COMPOSED_STREAM = 1`; the trainer records
  `levers["ego_perturb_stream"] = comp.ego.stream`, v8f_mini_train.py:1739, in selfcheck.json AND summary.json):
    ego_stream_problems(lv, p)   the BUILT perturbation stream in the gate selfcheck: levers.ego_perturb_stream is EXACTLY the int 1 on A (domain-separated from the swap's stream) and EXACTLY the int 0 on REP (the original
                                 generator). ABSENT is a failure on both arms; a bool (True == 1) or a float is not an int. Called from ego_problems, so `check` and `token` both refuse.
    the launch-gate token also records ego_perturb_stream.
  UNCHANGED: every other clause (finite / wacc_problems / swap_problems / the rest of ego_problems / expected_from_ref / check / write_token), byte for byte.
    python build_r2_asserter.py          writes ../box2_ship/gate_dagoaccorc_r2_assert.py (LF)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = Path("E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/box2_ship/gate_dagoaccorc_assert.py")
OUT = HERE.parent / "box2_ship" / "gate_dagoaccorc_r2_assert.py"


def rep(t: str, old: str, new: str, n: int = 1) -> str:
    assert t.count(old) == n, (old[:80], t.count(old))
    return t.replace(old, new)


STREAM_CONST = '''SWAP_P = 0.1
#: r2: the perturbation's generator stream the landed trainer fix BUILDS (levers.ego_perturb_stream): 1 = EGO_PERTURB_COMPOSED_STREAM (domain-separated from the swap's stream) on A, 0 = the original generator on REP.
EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 0}
'''

STREAM_FUNC = '''def ego_stream_problems(lv: dict, p: float) -> list:
    """r2: the BUILT perturbation stream (levers.ego_perturb_stream) is EXACTLY the int 1 on A and EXACTLY the int 0 on REP. Absent is a failure on both; True / 1.0 are not the int 1 (type check)."""
    want = EGO_STREAM["dagoaccorc" if p > 0.0 else "dagoaccorcrep"]
    got = lv.get("ego_perturb_stream", "<absent>")
    if type(got) is not int or got != want:
        return [f"built ego_perturb_stream {got!r} on {'A' if p > 0.0 else 'REP'} (must be exactly the int {want}: the perturbation's generator is {'domain-separated from the swap' if p > 0.0 else 'the original stream'})"]
    return []


def ego_problems(sc: dict, arm: str, p: float, seed: int) -> list:
'''


def derive(src: str) -> str:
    t = src
    t = rep(t, '"""PREREG_V8F_DAG_ORC on box 2, RE-BASED ON DAG-OACC by Amendment 1', '"""[r2 of the DAG-ORC chain: + ego_stream_problems] PREREG_V8F_DAG_ORC on box 2, RE-BASED ON DAG-OACC by Amendment 1')
    t = rep(t, "SWAP_P = 0.1\n", STREAM_CONST)
    t = rep(t, "def ego_problems(sc: dict, arm: str, p: float, seed: int) -> list:\n", STREAM_FUNC)
    t = rep(t, '        bad.append(f"built ego_perturb_with_dagger {wd!r} on REP (the flag must be absent)")\n    return bad\n',
            '        bad.append(f"built ego_perturb_with_dagger {wd!r} on REP (the flag must be absent)")\n    bad += ego_stream_problems(lv, p)      # r2: the separated perturbation stream was BUILT\n    return bad\n')
    t = rep(t, '"ego_perturb_with_dagger": ARMS[a.arm] > 0.0,', '"ego_perturb_with_dagger": ARMS[a.arm] > 0.0, "ego_perturb_stream": EGO_STREAM[a.arm],')
    return t


def main() -> int:
    t = derive(R1.read_text(encoding="utf-8"))
    assert "\r" not in t
    OUT.write_bytes(t.encode("utf-8"))
    print("wrote", OUT, len(t), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
