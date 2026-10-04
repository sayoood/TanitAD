"""Copy the v7f_r3 deliverable into the repo folder, with clip-id hygiene.

⛔ Repo artifacts carry NO bare UUID and NO 8-char prefix of a real clip id (tools/clipid_scan.py's
rule: a bare UUID is an unstable handle; sha12 = sha256(id)[:12] is the stable one). Every text
file is scanned against the UUID pattern (clipid_scan.UUID_RE, copied) and against the 8-char
prefixes of the 4,719 v7.2 clip ids (train 4,572 + eval 147, read from the local blobs); each hit
is replaced by ``sha12:<12 hex>`` and COUNTED. Source files that must stay verbatim (code) are
REFUSED if they carry a hit instead of being edited.

usage: python copy_deliverable.py <dest root>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

UUID_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     r"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-fA-F])")
P8_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
SRC = Path("C:/Users/Admin/v7f_r3")
BLOBS = ("C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_train.jsonl.gz",
         "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz")


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def known_ids() -> set[str]:
    out: set[str] = set()
    for b in BLOBS:
        with gzip.open(b, "rt", encoding="utf-8") as fh:
            out |= {json.loads(l)["clip_id"] for l in fh if l.strip()}
    return out


def redact(text: str, prefixes: dict[str, str]) -> tuple[str, int, int]:
    n_u = n_p = 0

    def _u(m):
        nonlocal n_u
        n_u += 1
        return f"sha12:{sha12(m.group(0))}"

    def _p(m):
        nonlocal n_p
        if m.group(0) in prefixes:
            n_p += 1
            return f"sha12:{prefixes[m.group(0)]}"
        return m.group(0)

    text = UUID_RE.sub(_u, text)
    text = P8_RE.sub(_p, text)
    return text, n_u, n_p


#: (source relative to SRC, destination relative to the deliverable root, verbatim?)
PLAN = [
    ("stack/scripts/train_v6_staged.py", "code/fix/stack/scripts/train_v6_staged.py", True),
    ("stack/tests/test_v6_effective_weights.py", "code/fix/stack/tests/test_v6_effective_weights.py", True),
    ("stack/tanitad/train/declared_vs_built_v6.py", "code/fix/stack/tanitad/train/declared_vs_built_v6.py", True),
    ("stack/tests/test_tactical_label_reach_v6.py", "code/fix/stack/tests/test_tactical_label_reach_v6.py", True),
    *[(f"deliverable/diffs/{n}", f"code/diffs_vs_c36b6ddd/{n}", True) for n in (
        "stack__scripts__train_v6_staged.py.diff", "stack__tests__test_v6_effective_weights.py.diff",
        "stack__tanitad__train__declared_vs_built_v6.py.diff",
        "stack__tests__test_tactical_label_reach_v6.py.diff", ".gitattributes")],
    *[(f"proof/{n}", f"code/proof/{n}", True) for n in (
        "bit_identity_off.py", "make_batches.py", "seeded_main.py", "smoke_real_eval_labels.py",
        "copy_deliverable.py", "rerun_all.sh", "summarize_all.py")],
    ("deliverable/RESULT.md", "RESULT.md", False),
    *[(f"raw/{n}", f"raw/{n}", False) for n in (
        "audit_census_train_blob.txt", "baseline_c36b6ddd_runbook_tests.log",
        "tests_baseline_c36b6ddd_full.log", "junit_baseline_c36b6ddd.xml",
        "tests_v7f_r3_final.log", "junit_v7f_r3_final.xml", "tests_outcome_compare.txt",
        "test_files_touching.txt", "cli_off_dryrun_compare.txt", "cli_dry_on_ST.log",
        "cli_dry_on_SJ.log", "cli_dry_off_ST.log", "cli_refusal_on_SW.log",
        "cli_refusal_no_multilabel.log", "cli_dry_runs_summary.txt", "smoke_real_eval_labels.log",
        "smoke_real_eval_labels_trajectory.txt", "train_smoke_SW.log", "train_smoke_ST_on.log",
        "train_smoke_ST_off.log", "train_smoke_ST_summary.txt",
        "tree_identity_base_vs_snapshot.txt", "tree_diff_mine_vs_snapshot.txt",
        "diff_apply_check.txt", "preexisting_failure_reasons.txt", "rerun_all.log")],
    ("proof/out/off_compare.txt", "raw/off_compare.txt", False),
    ("proof/out/off_c36b6ddd.json", "raw/off_c36b6ddd.json", False),
    ("proof/out/off_v7f_r3.json", "raw/off_v7f_r3.json", False),
    ("proof/out/smoke_real_eval_labels.json", "raw/smoke_real_eval_labels.json", False),
    ("proof/out/train_smoke/ST_on/config.json", "raw/train_smoke_ST_on_config.json", False),
    ("proof/out/train_smoke/ST_on/train_log.jsonl", "raw/train_smoke_ST_on_train_log.jsonl", False),
    ("proof/out/train_smoke/ST_off/train_log.jsonl", "raw/train_smoke_ST_off_train_log.jsonl", False),
]


def main() -> None:
    dest = Path(sys.argv[1])
    ids = known_ids()
    prefixes = {i[:8]: sha12(i) for i in ids}
    report = {"n_known_clip_ids": len(ids), "files": {}}
    for src_rel, dst_rel, verbatim in PLAN:
        src = SRC / src_rel
        raw = src.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        red, n_u, n_p = redact(text, prefixes)
        dst = dest / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if verbatim:
            if n_u or n_p:
                raise SystemExit(f"REFUSED: {src_rel} is copied verbatim but carries {n_u} UUID(s) "
                                 f"and {n_p} clip prefix(es)")
            shutil.copyfile(src, dst)
        else:
            dst.write_bytes(red.encode("utf-8"))
        report["files"][dst_rel] = {"uuids_redacted": n_u, "clip_prefixes_redacted": n_p,
                                    "verbatim": verbatim}
    (dest / "raw" / "clipid_redaction_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    tot_u = sum(v["uuids_redacted"] for v in report["files"].values())
    tot_p = sum(v["clip_prefixes_redacted"] for v in report["files"].values())
    print(f"copied {len(PLAN)} files; redacted {tot_u} UUIDs and {tot_p} clip-id prefixes "
          f"(against {len(ids)} known v7.2 clip ids)")
    for k, v in report["files"].items():
        if v["uuids_redacted"] or v["clip_prefixes_redacted"]:
            print("  ", k, v)


if __name__ == "__main__":
    main()
