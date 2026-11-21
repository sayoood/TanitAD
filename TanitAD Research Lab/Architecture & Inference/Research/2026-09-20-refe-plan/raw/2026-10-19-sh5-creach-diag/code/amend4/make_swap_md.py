"""Write SWAP.md for the Amendment-2 install (checker/tooling; not pinned).  Reads, never modifies, the live package `par18/sh5`; writes only `par18/sh5_a2/SWAP.md`.

    python make_swap_md.py [--live C:/Users/Admin/par18/sh5] [--copy C:/Users/Admin/par18/sh5_a2] [--base-tests D:/.../tests_base.json]

Run it AFTER `run_tests.py --out work/tests_summary_amend2.json` has finished and `build_package.py` has re-staged `out/` (HANDOFF.md lists the order).
"""
import argparse
import hashlib
import json
import os

BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-sh5-dac-prox"
SKIP = ("run", "out", "__pycache__")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def walk(root, skip=SKIP):
    out = {}
    for d, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x not in skip]
        for f in fn:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace(chr(92), "/")] = p
    return out


def counts(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return {k: v["ran"] for k, v in d.items() if isinstance(v, dict) and "ran" in v}, bool(d.get("all_passed")), {k: v["seconds"] for k, v in d.items() if isinstance(v, dict) and "seconds" in v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="C:/Users/Admin/par18/sh5")
    ap.add_argument("--copy", default="C:/Users/Admin/par18/sh5_a2")
    ap.add_argument("--base-tests", default="D:/Projects/TanitAD-artifacts/sh5_a2_2026-10-19/tests_base.json")
    a = ap.parse_args()
    L, C = walk(a.live), walk(a.copy)
    for k in ("SWAP.md", "HANDOFF.md", "work/FINISHED.txt", "work/finish_a2.py", "work/finish_a2.log"):          # working notes of this agent, not part of the package
        C.pop(k, None)
    new = sorted(k for k in C if k not in L)
    chg = sorted(k for k in C if k in L and sha(L[k]) != sha(C[k]))
    gone = sorted(k for k in L if k not in C)
    swap = [(k, "NEW" if k in new else "CHANGED") for k in sorted(set(new) | set(chg))]
    oL, oC = walk(f"{a.live}/out/{BASE}", skip=()), walk(f"{a.copy}/out/{BASE}", skip=())
    o_new = sorted(k for k in oC if k not in oL)
    o_chg = sorted(k for k in oC if k in oL and sha(oL[k]) != sha(oC[k]))
    o_same = sorted(k for k in oC if k in oL and sha(oL[k]) == sha(oC[k]))
    o_gone = sorted(k for k in oL if k not in oC)
    base, mine = counts(a.base_tests), counts(f"{a.copy}/work/tests_summary_amend2.json")
    lines = ["# SWAP: install Amendment 2 (`par18/sh5_a2` -> `par18/sh5`)", "",
             "Built by the SH5 Amendment-2 agent in a COPY; **nothing under `par18/sh5` was touched** (the live ctx build keeps running from it). The Master Mind installs this AFTER the ctx build has exited "
             "(`run/ctx_local.log` shows `ZZCTX_OK` / `ZZGPURELEASED`).", "",
             "## 0. State of the copy", "",
             f"- guard on the copy: `python -B -c \"import sys; sys.path.insert(0,'code'); import sh5_guard as G; print(G.require_sealed())\"` from `par18/sh5_a2` -> pins 16, amendments [1, 2] (proof against the old code: `work/amend2_guard_proof.json`, 8 cases, all as expected).",
             "- Amendment 2 sealed 2026-10-19T14:43:27+02:00, text sha256 `6c667c5164f641a7a9998b95db2711de50d9386d22269ee8d5a4cdee72328a9e`. Pinned files re-pinned: `code/sh5_lib.py`, `code/test_sh5_lib.py` (the other 14 keep their pins).", "",
             "## 1. Copy these files (relative to `par18/sh5_a2` -> `par18/sh5`), all together, before running any stage", "",
             "A partial copy is safe in the sense that the guard REFUSES every number-producing stage until the pins, the amendment text and its pin file agree; it is not safe to start a stage in between.", "",
             "| file | action | sha256 | bytes |", "|---|---|---|---|"]
    for k, act in swap:
        lines.append(f"| `{k}` | {act} | `{sha(C[k])}` | {os.path.getsize(C[k])} |")
    lines += ["", "Pinned (the guard checks these): `code/sh5_lib.py`, `code/test_sh5_lib.py`, `PREREG_SH5_DAC_PROX_AMENDMENT2.md`, `PREREG_SH5_DAC_PROX_AMENDMENT2.sha256`, `CODE_PINS_AMENDMENT2.json`. "
              "Not pinned (tooling and evidence): the rest. `AMENDMENT2.draft.md` is the draft the seal tool filled in (the sealed text is `PREREG_SH5_DAC_PROX_AMENDMENT2.md`); copy it for symmetry with `AMENDMENT1.draft.md`, or skip it. "
              "`work/run_tests_amend2.log` and `work/tests_summary_amend2.json` are the evidence behind the `red_table` gate (`READY.md` Q5 now points `sh5_gates.py --tests` at it).", "",
             f"Not in the list and not to be copied: `run/`, `out/` (re-staged, section 3), any `__pycache__`. Files that exist in the live package and not in the copy: {gone if gone else 'none'}.", "",
             "Bash form:", "```", "cd /c/Users/Admin/par18", "for f in " + " ".join(f'"{k}"' for k, _ in swap) + "; do cp \"sh5_a2/$f\" \"sh5/$f\"; done", "```", "",
             "## 2. Optional, on D: (the pre-amendment copies the guard must refuse)", "",
             "`D:/Projects/TanitAD-artifacts/sh5_a2_2026-10-19/amend1_state/{sh5_lib.py,test_sh5_lib.py}` (sha256 `08781b86b7a21d016f29dd47ee82d029a7f50ec3d4070d10f33c5610abdada9e`, `f0ebf31f19d056a38a4788b34319c2514715b79c9817dd5b82b5650cd3778f7f` = the Amendment-1 pins). "
             "They are already recorded in the staged `raw/ARTIFACTS_D.json` under `sealed_state_amend1/`. Copy them into `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/sealed_state/` if you want them beside the other sealed-state copies; nothing reads them.", "",
             "## 3. After the copy: verify, then re-stage", "",
             "1. `cd par18/sh5 && python -B -c \"import sys; sys.path.insert(0,'code'); import sh5_guard as G; print(G.require_sealed())\"` must print `{'prereg': '687e08c8...', 'pins': 16, 'amendments': [1, 2]}`.",
             "2. `sha256sum` each copied file against section 1.",
             "3. Optional fast tests (CPU, `CUDA_VISIBLE_DEVICES=-1`, ~15 s): `test_sh5_lib.py test_sh5_branch.py test_sh5_ctx.py test_sh5_guard.py` from `par18/sh5/code`.",
             "4. `python code/build_package.py` from `par18/sh5` re-stages `out/` (its `ROOT` is now derived from its own location; the OLD `build_package.py` hardcoded `par18/sh5` and `rmtree`d `ROOT/out`, so never run the old one from the copy). "
             "The result is byte-identical to the copy's `out/` (section 4) except that `paths.json` holds `par18/sh5/out/...` paths.", "",
             "## 4. `out/` staging: what changes", "",
             f"`par18/sh5_a2/out/` was built by `code/build_package.py` in the copy; its `paths.json` ([repo path, absolute source in `sh5_a2/out`, \"raw\"]) is valid as it stands, so the lander can take `par18/sh5_a2/out/paths.json` directly, or you can re-stage from `par18/sh5` (section 3.4). "
             f"Entries under `{BASE}/` (live `sh5/out` = {len(oL)} files incl. MANIFEST; copy `sh5_a2/out` = {len(oC)}):", "",
             f"- **NEW ({len(o_new)})**: " + ", ".join(f"`{k}`" for k in o_new),
             f"- **CHANGED ({len(o_chg)})**: " + ", ".join(f"`{k}`" for k in o_chg),
             f"- unchanged ({len(o_same)}); removed ({len(o_gone)}): " + (", ".join(f"`{k}`" for k in o_gone) if o_gone else "none"), "",
             "| `out/` entry (under the package folder) | status | sha256 now |", "|---|---|---|"]
    for k in o_new + o_chg:
        lines.append(f"| `{k}` | {'NEW' if k in o_new else 'CHANGED'} | `{sha(oC[k])}` |")
    lines += ["", "Line endings: `build_package.py` normalises CRLF to LF in the STAGED copies of `.py` / `.md` / `.json` / `.log` files, so a staged sha256 can differ from the working-copy sha256 in section 1 when the working file has CRLF "
              "(here `code/sh5_amend2_power.py` and `work/run_tests_amend2.log`; the `power_script_sha256` recorded inside `raw/amend2_power.json` is of the CRLF working-copy file, content identical). The pinned files and the amendment text are LF and hash identically in both places.",
              "", "## 5. Tests: base vs mine (CPU, `CUDA_VISIBLE_DEVICES=-1`)", ""]
    if base:
        lines.append(f"- BASE (a hash-identical copy of the live package, run on D:, `{a.base_tests}`): " + ", ".join(f"{k} {v}" for k, v in base[0].items()) + f" = **{sum(base[0].values())}**, all_passed {base[1]}, pipeline {base[2].get('test_sh5_pipeline.py')} s.")
    if mine:
        lines.append(f"- MINE (`work/tests_summary_amend2.json`): " + ", ".join(f"{k} {v}" for k, v in mine[0].items()) + f" = **{sum(mine[0].values())}**, all_passed {mine[1]}, pipeline {mine[2].get('test_sh5_pipeline.py')} s.")
    else:
        lines.append("- MINE: **PENDING** (the full run was still going when this file was written; re-run this script after `work/tests_summary_amend2.json` exists).")
    lines += ["", "## 6. Two things for the Master Mind (see the amendment, sections 5 and 6)", "",
              "- **VOID keeps its sealed numeric rule (lower CI > +0.02)**, per the ruling, so it is now less sensitive than the new CTX-REACHABLE rule. Following the new rule would cost a false VOID of 0.5-1.3 % (FULL) / 2.5-7.3 % (LOCAL) on a clean machinery. Your call; a change is an Amendment 3 before any number exists.",
              "- The between-seed correlation of the both-seed power is stated, not measured (bracket rho 0..1).", ""]
    open(f"{a.copy}/SWAP.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("SWAP.md written;", len(swap), "files to copy;", len(o_new), "new /", len(o_chg), "changed out/ entries; tests mine:", bool(mine))


if __name__ == "__main__":
    main()
