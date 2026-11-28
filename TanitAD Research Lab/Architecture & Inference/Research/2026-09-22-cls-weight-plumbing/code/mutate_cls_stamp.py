"""MUTATION PROOF for `test_cls_weight_stamp.py`. Anchors carry NO line terminator (CRLF files).

Arms — each one reintroduces a defect this programme has ACTUALLY PAID FOR:
  S1  `built` re-reads the artifact instead of the model  -> the self-agreeing stamp (305debd)
  S2  the digest ignores class names                      -> a permuted vector reads identical
  S3  `load_cls_class_weight` stops verifying its digest  -> the attestation goes decorative
  S4  the agent_losses call site drops the weight         -> declared-not-plumbed, 2-D seam
  S5  the box3d_loss_row call site drops the weight       -> declared-not-plumbed, SCORED head
  S6  the "record says off" branch is disabled            -> one direction of a bidirectional guard
  S7  the digest comparison branch is disabled            -> right flag, WRONG vector, unseen
  S8  the "requested but not built" branch is disabled    -> the 305debd defect itself

⛔ Carried over, each paid for: decode the child as utf-8 (never `text=True`, which uses the
PARENT's cp1252 and returns EMPTY streams on one stray byte); an arm counts only if a NAMED test
failed; an arm whose anchor is not UNIQUE aborts the whole run as INVALID rather than being
silently skipped — 4 of 5 arms once skipped while the prover printed a ratio.
⛔ And the verdict is read off the ARTIFACT (the JSON written at the end), never off an exit code
routed through a shell.
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
SLOTS = REPO / "stack" / "tanitad" / "models" / "agent_slots.py"
TESTS = REPO / "stack" / "tests" / "test_cls_weight_stamp.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

# ⛔ Built with chr() ON PURPOSE. Writing this constant as a backslash escape inside a generated
# script is the trap that has now bitten three times in this session: a shell heredoc consumes one
# level of backslash, so the escape arrives as a REAL newline and the file dies at PARSE time with
# "unterminated string literal". The lesson in memory is "use the Edit tool, not a heredoc"; this
# is the belt to that suspenders — the value cannot be mangled because there is no escape to eat.
EOL_CHARS = chr(13) + chr(10)

# ⛔ ANCHORS ARE WHOLE LINES, MATCHED BY EQUALITY OVER LINES — not substrings.
# TWO traps are closed by that choice, and I hit BOTH on the way here:
#   (a) CRLF: an anchor carrying a line terminator matches nothing in these files, and 4 of 5
#       arms once skipped silently while the prover still printed a ratio;
#   (b) INDENTATION AS A SUBSTRING: the two loss call sites differ only by indent, so the
#       12-space anchor is a proper substring of the 20-space line and matched BOTH.
# Full-line equality is immune to both, and the uniqueness check still aborts the run.
MUTATIONS = [
    ("S1_built_reads_the_artifact_not_the_model", TRAIN,
     '        None if _cwv is None else _agent_slots.cls_weight_digest(_cwv))',
     '        None if _cwv is None else _agent_slots.load_cls_class_weight()[1]["digest"])'),
    ("S2_digest_ignores_class_names", SLOTS,
     '    payload = "|".join(f"{c}={v:.6f}" for c, v in zip(classes, vals))',
     '    payload = "|".join(f"{v:.6f}" for v in sorted(vals))'),
    ("S3_load_stops_verifying_its_own_digest", SLOTS,
     '    if _stated != _dig:',
     '    if False:'),
    ("S4_agent_losses_site_drops_the_weight", TRAIN,
     '            cls_class_weight=getattr(model, "_cls_class_weight", None))',
     '            cls_class_weight=None)'),
    ("S5_box3d_site_drops_the_weight", TRAIN,
     '                    cls_class_weight=getattr(model, "_cls_class_weight", None))',
     '                    cls_class_weight=None)'),
    # ⛔ S6 IS THE ARM THAT ESCAPED ONCE, AND WHY IT ESCAPED IS THE POINT. Its first form
    # deleted half of a refusal MESSAGE, and the test then pinned to message text went green
    # against a guard it had not exercised. The arm now removes the BRANCH, and the test that
    # catches it CALLS the function. A prose mutation is not a defect; a disabled branch is.
    ("S6_seam_check_loses_the_reverse_direction", TRAIN,
     '        if cw_built and not wants:',
     '        if False:'),
    ("S7_seam_check_stops_comparing_digests", TRAIN,
     '        if cw_built and cw.get("built") is not None:',
     '        if False:'),
    ("S8_seam_check_loses_the_forward_direction", TRAIN,
     '        if wants and not cw_built:',
     '        if False:'),
    # ⛔ S9 reintroduces the defect the CORRECTED PROSE now denies: an out-of-vocabulary
    # label silently becoming class 0 (`automobile`) instead of being masked. 10,077 TRAIN
    # boxes ride on this line, and the old docstring's claim that the label "does not exist
    # in the corpus" is exactly what would have stopped anyone testing it.
    ("S9_out_of_vocabulary_class_relabelled_to_zero", SLOTS,
     '                [idx.get(str(c), -1) for c in list(classes)[:n]],',
     '                [idx.get(str(c), 0) for c in list(classes)[:n]],'),
    # ⛔ C-arms: the CORPUS-LINE guard, added after a vector counted on the parity join (4.09 %
    # overlap with refcv6's corpus) was reported as refcv6 readiness. Each arm re-opens the exact
    # door that defect walked through.
    ("C1_loader_stops_checking_the_corpus_line", SLOTS,
     '    if expect_corpus_line is not None and str(_line) != str(expect_corpus_line):',
     '    if False:'),
    ("C2_loader_accepts_an_artifact_with_no_line", SLOTS,
     '    if not _line:',
     '    if False:'),
    ("C3_the_b1_choice_points_at_the_parity_artifact", TRAIN,
     '    "b1": (_agent_slots.CLS_WEIGHTS_B1, _agent_slots.CORPUS_LINE_B1),',
     '    "b1": (_agent_slots.CLS_WEIGHTS_TRAIN2400, _agent_slots.CORPUS_LINE_B1),'),
]


def run_tests():
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        env={"PYTHONPATH": str(REPO / "stack"), "PATH": os.environ["PATH"],
             "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2"})
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


def main() -> int:
    files = {TRAIN, SLOTS}
    orig = {p: p.read_bytes() for p in files}
    md5 = {p: hashlib.md5(b).hexdigest() for p, b in orig.items()}
    res = {"_what": "mutation proof that test_cls_weight_stamp.py can actually FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "targets": {str(p.relative_to(REPO)): m for p, m in md5.items()}, "arms": []}
    try:
        rc, failed, dead, tail = run_tests()
        res["baseline"] = {"rc": rc, "failed": failed}
        print(f"BASELINE rc={rc} failed={failed}")
        if dead:
            print("ZZABORT baseline produced NO OUTPUT — INCONCLUSIVE, not green")
            return 6
        if rc != 0:
            print("ZZABORT baseline not green\n" + tail)
            return 3
        for name, target, old, new in MUTATIONS:
            txt = orig[target].decode("utf-8")
            lines = txt.splitlines(keepends=True)
            hits = [i for i, ln in enumerate(lines) if ln.rstrip(EOL_CHARS) == old]
            if len(hits) != 1:
                print(f"ZZABORT {name}: anchor matches {len(hits)} LINES in "
                      f"{target.name} — the proof is INVALID, not weak")
                res["_VERDICT"] = "⛔ INVALID — an arm could not be applied."
                res["arms"].append({"arm": name, "error": "anchor not unique",
                                    "occurrences": len(hits)})
                pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_cls_stamp.json"
                             ).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
                print(json.dumps(res, indent=1, ensure_ascii=False))
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
        f"⭐ MUTATION-PROVEN — all {n}/{len(MUTATIONS)} arms RED, including the "
        "self-agreeing `built` stamp and BOTH loss-site forwards; both files restored "
        "byte-identical and the final clean run is green."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"⛔ ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_cls_stamp.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.exit(main())
