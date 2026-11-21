"""The read of PREREG_T2_DOMINANCE (frozen, sha256 38aa9df3...), section 6: written by AGENT L on 2026-11-06 BEFORE any C0C or P10 row of stages 2, 3 or 4 was computed or read.

    python -I read_t2_dominance.py --root2 STAGE2_PULL/rollouts --root3 STAGE3_PULL/rollouts --root4 STAGE4_PULL/rollouts --sources2 stage2/SCENE_SOURCES.json
           --sources3 stage3/SCENE_SOURCES3.json --g1 C0C_ID_RESULT.json --out-dir OUT          (writes OUT/DOMINANCE_T2.json and OUT/DOMINANCE_T2.md)
           [--exclusions2 EXCLUSIONS2.json --exclusions3 EXCLUSIONS3.json --exclusions4 EXCLUSIONS4.json]       (rule R-SCN: all three or none)
           [--arm NAME] [--arm-stage N --arm-root R --arm-result STAGEN_RESULT.json --arm-exclusions EXCLUSIONSN.json --arm-overlay TAR::pkg/MANIFESTN.json --arm-gate-log LOG]   (AGENT Q: another arm in C0C's place)

ARM (AGENT Q, re-pinned 2026-11-07 before any LoRA or weight-edited row was read). `--arm NAME` (default C0C) names the arm whose cells `<tc>|<NAME>_s0` and `<tc>|<NAME>_s1` are read against the floor P10 in
C0C's place; `--arm-stage N` (default 4) says which stage holds them. With the defaults the read is the previous pin's, field for field (`logs/q_identity.json` compares a fresh read of the real pulls
with the read of record). Stage 4 holds C0C: any other arm lives in a stage beyond 4, read with `--arm-root` (its rollouts dir), `--arm-result` (its STAGE<N>_RESULT.json, default beside the root) and
`--arm-exclusions` (its EXCLUSIONS*.json), plus `--arm-overlay` (G2 a: its overlay MANIFEST<N>.json) and `--arm-gate-log` (G2 b: its green pytest log). That stage is laid out as stage 4 is (T2 x {NAME_s0,
NAME_s1} deciding, RR x {NAME_s0, NAME_s1} beside, over all 62 scenes); a stage that ran otherwise is REFUSED by the R-SCN conditions, never read loosely. Stages 2, 3 and 4 are still read for the
R-SCN conditions, and the UNION of the excluded scenes over every stage read (2, 3, 4 and N) leaves BOTH the arm and P10 (R-SCN rule 4). Every output (DOMINANCE_T2.json keys `arm`, `arm_stage`, `argv`;
the .md header) names the arm and records the argv. A wrong arm is REFUSED with a named error (ARM_NAME_FORM, ARM_NAME_SEED_SUFFIX, ARM_NAME_RESERVED, ARM_STAGE_BELOW_4, ARM_STAGE4_IS_C0C,
ARM_STAGE4_EXTRA_INPUTS, ARM_ROOT_REQUIRED before anything is read; ARM_ABSENT_IN_PULL as gate `ARM` when the pull holds no such cell).

RULE R-SCN (AGENT P, re-pinned 2026-11-06 before any stage-2, 3 or 4 row was computed; AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb, and the card's 19:31 log entry). Without the three flags the read is the
previous pin's, line for line. With them, G3 reads, per stage 2, 3 and 4 (`rscn_check`):
  * the EXCLUSIONS*.json (from scene_exclusions.py) is ATTRIBUTED with no problems, is of THAT stage (its `stage` label) and was built from THAT stage's result (`manifest_sha256` equal, and its
    `inputs.result_sha256` equal to the sha256 of the result file on hand; the manifest sha alone cannot tell the stages apart, all three ran the stage-1 package);
  * the stage's failed cells are exactly all the cells of its excluded scenes, deciding and beside (stages 2 and 3: 10 + 4 per scene; stage 4: 2 + 2), and on disk every cell of an excluded scene is FAIL
    with 0 records and none is DONE;
  * every cell of every included scene is DONE (the result's counts AND the cell directories), no pass and no scene was skipped, and `overall` is COMPLETE, or INCOMPLETE only with those conditions met;
  * the UNION of the excluded scenes over stages 2, 3 and 4 is removed from BOTH C0C and P10 before `pair_check` and before any row is built, and the expected start counts fall by the starts of those scenes
    (read from the cells' own `records=a/N` markers).
G2(e) tabulates an excluded scene as `EXCLUDED (R-SCN)` with its reason (it has no actor_attach.json); the included scenes' deciding cells still need the file. DOMINANCE_T2.json (`r_scn`) and .md list the
excluded scenes per stage, with the reason, the cell counts and the starts lost.

What it does, in this order (each gate is a REFUSAL: if one fails, no row is built and the outcome is GATE-FAIL, card section 4):
  G4 the pinned reader `recover_next/read/code/recover_read.py` (sha256 de5cfbf5...) is loaded BY FILE PATH, sha checked (as `cl_read.init()` does); it reproduces the banked T1-kin Read A block of the
     C0C composite vs P10 (recover_next/read/raw/arrays/C0C_s{0,1}.npz against RECOVER_READ_R.json composite.read_a_C0C_vs_P10 and seeds.s*.C0C_minus_P10), every row and every field, bit for bit
     (JSON round trip of the same floats). If a control cannot be reproduced no row is read.
  G1 the C0C-ID result file (`--g1`, written by the gate run): pass true, the three RED arms (smooth off, feasible off, prior swapped for zeros) all RED, and off/off reproduces C0.
  G2 the card's 17:42 reading, all of (a) to (e): (a) stage 1 ALL_GATES_PASS and the same package manifest in every stage (no stage-1 file replaced by an overlay), (b) the T0 / T4 dev-box tests exist and each
     stage cites a green pytest log not older than the gate test files, (c) each stage's STAGE*_RESULT.json gates.preflight and gates.gate PASS, (d) ZERO(T2) == P10(T2) exactly on every window of stages 2 and 3,
     (e) every deciding cell's actor_attach.json present, its verdict tabulated per scene.
  G3 every C0C_s0 / C0C_s1 (stage 4) and P10 (stages 2 and 3) cell under T2 / objects is DONE (`records=N/N`, no FAIL, line count = N), the 62 scenes and 121 starts (81 launch + 40 guard) are all there,
     every stage's `overall` is COMPLETE, and the (scene, f0) keys of C0C and P10 pair exactly (`t2v8f_score.pair_check`).
Then, per seed s: the 19 per-window rows of C0C_s{s}(T2, objects) and P10(T2, objects) over the 121 starts from the T2 scorer's own row builder (`t2v8f_lead.rows_for_records`, as
`t2v8f_run.score_main` uses it), then `rr.rows_of(ua, va, eid = SCENE index, cls = GD.classify(v0, g), n_boot = 2000)` and `rr.read_a_block(SR, GD, rows)`. `v0` and `g` are the records' own fields
(the executed speed at t0, equal to the logged one, and the logged future in the ego frame at t0); no class rule is invented here. A row that cannot be built (missing from `rows_for_records`) is a refusal.
Outcome names are the card's section 5: DOMINANT-T2 / NOT-DOMINANT / UNDERPOWERED / GATE-FAIL.
  READING (the card lists the outcomes "in this order" but outcome 3 is a special case of outcome 2, so it is unreachable unless it is tested FIRST): UNDERPOWERED is tested before NOT-DOMINANT.
  It means: at BOTH seeds no row of the 19 is separated. Reported as `reading_underpowered_first` in the JSON; the Master Mind may overrule by a dated amendment.
The interval answers "another draw of SCENES" only (paired scene-cluster bootstrap, B 2000, seed 0, taniteval/ci.py through the reader). Both seeds must hold; there is no replicate arm (H-ESTIM-SEED-1).
Tier: T2 re-perception, agents replayed at log time: a DOMINANT-T2 is not a reactive-traffic result. STRATEGIC is read as Read A defines it; the three-family form is reported beside (PI ITEM 25).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

DL = "E:/Projects/TanitAD-artifacts/dagland"
CARD = f"{DL}/motion_c/t2_prereg/PREREG_T2_DOMINANCE.md"
CARD_SHA = "38aa9df37b6a2ca85a6873b2f09ccf6657fb56796d728b91ef4ae42c4fe70b41"
CARD_FROZEN_BYTES = 8497      # the card as frozen at 17:30:46 (111 lines, sha256 CARD_SHA); section 8 is APPEND-ONLY, so later dated entries lengthen the file but never touch these bytes
RR = f"{DL}/recover_next/read/code/recover_read.py"
RR_SHA = "de5cfbf572ba2184e5f85a942ac79a6a6fa4df422f83367bbd9fa0e6966e95d1"
BANK_DIR = f"{DL}/recover_next/read/raw"
BANK_JSON = f"{BANK_DIR}/RECOVER_READ_R.json"
BANK_PINS = {f"{BANK_DIR}/arrays/C0C_s0.npz": "265c804427eb66d6b541236697e941c184e487c8a31c03ee894a377d6141c8dc", f"{BANK_DIR}/arrays/C0C_s1.npz": "33e78abcf17519b425f81be7a8091b23595f5fae9d825e32e6977b1ab972b791",
             BANK_JSON: "6b26ede6c3535ca24294d09db2bd470d4caeda031d8440c76d576749fe068734"}
SEEDS = (0, 1)
N_BOOT = 2000
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
EXPECT = {"scenes_launch": 42, "scenes_guard": 20, "starts_launch": 81, "starts_guard": 40}
RED_ARMS = ("smooth_off", "feasible_off", "prior_zeros")
OUTCOMES = ("DOMINANT-T2", "NOT-DOMINANT", "UNDERPOWERED", "GATE-FAIL")
STATE = {}
ARM_DEFAULT, ARM_STAGE_DEFAULT = "C0C", 4
STAGE4_ARM = "C0C"                                                    # stage 4 ran the C0C composite; an arm of another name lives in a stage beyond 4
ARM_RESERVED = ("P10", "ZERO", "BLIND", "C0", "DAGOACC")             # the floor and the stage-2/3 arms: not an arm to read against the floor
ARM_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
ARM_SEED_SUFFIX_RE = re.compile(r".*_s[0-9]+")                        # the read appends `_s0` / `_s1` itself


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def card_frozen_ok(path=CARD):
    """The frozen part of the card is byte-identical: sha256 of its first CARD_FROZEN_BYTES == CARD_SHA. Appended dated log entries (section 8) are allowed and do not move the check; an edit above them fails it."""
    if not os.path.isfile(path):
        return False
    b = open(path, "rb").read()
    return len(b) >= CARD_FROZEN_BYTES and hashlib.sha256(b[:CARD_FROZEN_BYTES]).hexdigest() == CARD_SHA


def init(rr_path=None, rr_sha=RR_SHA):
    """Load the pinned reader BY FILE PATH (sha256 checked FIRST), its instruments (`heavy_init`), then the T2 scorer's row builder. Refuses on a sha mismatch."""
    if "rr" in STATE and rr_path is None:
        return STATE["rr"]
    path = rr_path or RR
    got = sha256_file(path)
    if got != rr_sha:
        raise SystemExit(f"REFUSED: the reader {path} is {got}, not the pinned {rr_sha}")
    spec = importlib.util.spec_from_file_location("recover_read_dom", path)
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    rr.heavy_init()
    import t2v8f_lead as L
    import t2v8f_score as S
    assert tuple(S.LABELS) == tuple(rr.GD.LABELS), "the T2 scorer's 19 labels differ from the reader's"
    assert tuple(rr.SEEDS) == SEEDS and tuple(rr.FAMILIES) == FAMILIES
    STATE.update(rr=rr, L=L, S=S)
    return rr


# ------------------------------------------------------------------------------------------------ the outcome (pure)
def n_separated(rows):
    return int(sum(bool(r["separated"]) for r in rows.values()))


def decide(gates_ok, blocks):
    """Card section 5. `blocks` = {seed: {"rows": {label: {delta, lo, hi, separated}}, "read_a": read_a_block}} for both seeds (None when a gate failed: no row was read).
    -> {outcome, reading_underpowered_first, per_seed, failing_terms}."""
    note = "UNDERPOWERED is tested before NOT-DOMINANT: with the card's literal order it would be unreachable (no separated row => D1 fails => NOT-DOMINANT)"
    if not gates_ok or blocks is None:
        return {"outcome": "GATE-FAIL", "reading_underpowered_first": note, "per_seed": None, "failing_terms": None}
    if sorted(blocks) != list(SEEDS):
        raise ValueError(f"decide(): need both seeds {SEEDS}, got {sorted(blocks)}")
    per = {f"s{s}": {"DOMINANT": bool(b["read_a"]["DOMINANT"]), "D1": bool(b["read_a"]["D1"]), "D2": bool(b["read_a"]["D2"]), "D3": bool(b["read_a"]["D3"]),
                     "DOMINANT_three_families_strategic_paused": bool(b["read_a"]["DOMINANT_three_families_strategic_paused"]), "n_rows_separated": n_separated(b["rows"])} for s, b in blocks.items()}
    ft = {f"s{s}": list(b["read_a"]["failing_terms"]) for s, b in blocks.items()}
    if all(p["DOMINANT"] for p in per.values()):
        name = "DOMINANT-T2"
    elif all(p["n_rows_separated"] == 0 for p in per.values()):
        name = "UNDERPOWERED"
    else:
        name = "NOT-DOMINANT"
    return {"outcome": name, "reading_underpowered_first": note, "per_seed": per, "failing_terms": ft}


# ------------------------------------------------------------------------------------------------ gates
def g1_check(path):
    """The C0C-ID result file: {"gate": "G1_C0C_ID", "pass": true, "red_arms": {"smooth_off": "RED", "feasible_off": "RED", "prior_zeros": "RED"}, "off_off_reproduces_C0": true, ...}."""
    out = {"file": path, "sha256": None, "PASS": False, "why": []}
    if not path or not os.path.isfile(path):
        out["why"].append("G1_FILE_MISSING")
        return out
    out["sha256"] = sha256_file(path)
    d = json.load(open(path, encoding="utf-8"))
    if d.get("gate") != "G1_C0C_ID":
        out["why"].append(f"G1_WRONG_FILE gate={d.get('gate')!r}")
    if d.get("pass") is not True:
        out["why"].append(f"G1_NOT_PASS pass={d.get('pass')!r}")
    red = d.get("red_arms") or {}
    for a in RED_ARMS:
        if red.get(a) != "RED":
            out["why"].append(f"G1_RED_ARM_NOT_RED {a}={red.get(a)!r}")
    if d.get("off_off_reproduces_C0") is not True:
        out["why"].append("G1_OFF_OFF_NOT_C0")
    out["PASS"] = not out["why"]
    return out


def g2_check(results):
    """results {stage: path to STAGE<n>_RESULT.json}. Recorded PASS of preflight and gate in each."""
    out = {"stages": {}, "PASS": True, "why": [], "note": "STAGE*_RESULT.json records `gates.preflight` and `gates.gate` (the CPU gate); the T0..T5 gates of the crawl card are not itemised in it"}
    for st, p in sorted(results.items()):
        if not p or not os.path.isfile(p):
            out["why"].append(f"G2_RESULT_MISSING stage {st}: {p}")
            continue
        g = json.load(open(p, encoding="utf-8")).get("gates", {})
        out["stages"][st] = {"preflight": g.get("preflight"), "gate": g.get("gate")}
        for k in ("preflight", "gate"):
            if g.get(k) != "PASS":
                out["why"].append(f"G2_NOT_PASS stage {st} {k}={g.get(k)!r}")
    out["PASS"] = not out["why"]
    return out


# ------------------------------------------------------------------------------------------------ G2 (a), (b), (d), (e): the card's 17:42 reading (section 8, reading 3). (c) is `g2_check` above.
T2V8F = os.path.dirname(HERE).replace("\\", "/")
MANIFEST_KEY = {2: "manifest2_sha256", 3: "manifest3_sha256", 4: "manifest4_sha256"}


def manifest_key(st):
    """the result key that records the overlay manifest of stage `st` (stages beyond 4 follow the same spelling)."""
    return MANIFEST_KEY.get(st) or f"manifest{st}_sha256"


DEFAULT_RESULT1 = f"{T2V8F}/stage1/pull_box2/pull/STAGE1_RESULT.json"
DEFAULT_MANIFEST1 = f"{T2V8F}/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json"
DEFAULT_OVERLAYS = {2: f"{T2V8F}/stage2/overlay2/t2v8f_stage2_overlay2.tar::pkg/MANIFEST2.json", 3: f"{T2V8F}/stage3/t2v8f_stage3_overlay.tar::pkg/MANIFEST3.json",
                    4: f"{T2V8F}/stage4/t2v8f_stage4_overlay.tar::pkg/MANIFEST4.json"}
# T0 (policy reproduction: banked tokens give the T1-kin tool's tick-0 plan) and T4 (metric pin: the scorer reproduces the T1 reader's rows on a banked dump) are DEV-BOX tests of the crawl card, not box-2 gates.
GATE_TESTS = {"T0": (("tests/test_t2v8f_gates.py", "test_g1_tick0"), ("tests/test_t2v8f_gates.py", "test_g1_ticks")), "T4": (("tests/test_t2v8f_e5.py", "test_t4"),)}
DEFAULT_GATE_LOGS = {2: f"{T2V8F}/logs/pytest_stage2_full.log", 3: f"{T2V8F}/logs/pytest_stage3_full.log", 4: f"{T2V8F}/logs/pytest_full_M.log"}
ZERO_P10_KEYS = ("f0", "v0", "g", "cl", "plans", "track", "frame_md5")


def load_blob(spec):
    """'<file>' or '<tar>::<member>' -> bytes (None when absent)."""
    path, _, member = spec.partition("::")
    if not os.path.isfile(path):
        return None
    if not member:
        return open(path, "rb").read()
    import tarfile
    with tarfile.open(path) as t:
        try:
            return t.extractfile(member).read()
        except KeyError:
            return None


def g2a_check(result1, results, manifest1=DEFAULT_MANIFEST1, overlays=None):
    """(a) stage 1's `overall` is ALL_GATES_PASS, and the CELL CODE is the stage-1 package's: every stage's recorded `manifest_sha256` (the sha256 of the package MANIFEST.json it ran, whose per-file sha256 box 2's preflight
    verified: `pkg_files_ok == pkg_files`) equals stage 1's; the overlay manifests carry that same stage-1 manifest sha256, match what each stage recorded, and replace NO stage-1 file (a file is named)."""
    out = {"PASS": True, "why": [], "rollout_code": {}, "stages": {}}
    why = out["why"]
    r1 = json.load(open(result1, encoding="utf-8")) if result1 and os.path.isfile(result1) else None
    if r1 is None:
        why.append(f"G2A_STAGE1_RESULT_MISSING {result1}")
    elif r1.get("overall") != "ALL_GATES_PASS":
        why.append(f"G2A_STAGE1_NOT_PASS overall={r1.get('overall')!r}")
    blob = load_blob(manifest1)
    m1 = json.loads(blob) if blob else None
    if m1 is None:
        why.append(f"G2A_MANIFEST1_MISSING {manifest1}")
    sha1 = r1.get("manifest_sha256") if r1 else None
    if m1 is not None and sha1 and hashlib.sha256(blob).hexdigest() != sha1:
        why.append(f"G2A_MANIFEST1_DIFFERS the stage-1 manifest on hand is {hashlib.sha256(blob).hexdigest()[:12]}, STAGE1_RESULT.json ran {sha1[:12]}")
    files1 = {e["path"]: e["sha256"] for e in (m1 or {}).get("files", [])}
    out["rollout_code"] = {p: h for p, h in files1.items() if p.startswith("code/")}
    for st, p in sorted(results.items()):
        r = json.load(open(p, encoding="utf-8")) if p and os.path.isfile(p) else None
        if r is None:
            why.append(f"G2A_STAGE_RESULT_MISSING stage {st}: {p}")
            continue
        s = {"manifest_sha256": r.get("manifest_sha256")}
        if sha1 and r.get("manifest_sha256") != sha1:
            why.append(f"G2A_PACKAGE_DIFFERS stage {st} ran package manifest {str(r.get('manifest_sha256'))[:12]}, stage 1 ran {sha1[:12]}")
        pf = (r.get("steps") or {}).get("preflight") or {}
        info = pf.get("info") or {}
        s["preflight"], s["pkg_files_ok"], s["pkg_files"] = pf.get("verdict"), info.get("pkg_files_ok"), info.get("pkg_files")
        if pf.get("verdict") != "PASS" or info.get("pkg_files_ok") is None or info.get("pkg_files_ok") != info.get("pkg_files"):
            why.append(f"G2A_PKG_FILES stage {st}: preflight {pf.get('verdict')!r}, files ok {info.get('pkg_files_ok')}/{info.get('pkg_files')}; " + "; ".join(str(x) for x in (pf.get("failures") or [])[:3]))
        spec = (overlays or DEFAULT_OVERLAYS).get(st)
        ob = load_blob(spec) if spec else None
        if ob is None:
            why.append(f"G2A_OVERLAY_MANIFEST_MISSING stage {st}: {spec}")
        else:
            om = json.loads(ob)
            s["overlay_manifest_sha256"] = hashlib.sha256(ob).hexdigest()
            if r.get(manifest_key(st)) != s["overlay_manifest_sha256"]:
                why.append(f"G2A_OVERLAY_DIFFERS stage {st} ran overlay manifest {str(r.get(manifest_key(st)))[:12]}, on hand {s['overlay_manifest_sha256'][:12]}")
            if sha1 and om.get("stage1_manifest_sha256") != sha1:
                why.append(f"G2A_OVERLAY_BASE stage {st}: built against stage-1 manifest {str(om.get('stage1_manifest_sha256'))[:12]}")
            for e in om.get("files", []):
                if e["path"] in files1:
                    why.append(f"G2A_CODE_REPLACED stage {st}: the overlay replaces the stage-1 file {e['path']}")
        out["stages"][st] = s
    out["PASS"] = not why
    return out


def _tests_in(path):
    import ast
    return {n.name for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())) if isinstance(n, ast.FunctionDef)}


def g2b_check(t2v8f_root=T2V8F, logs=None):
    """(b) T0 and T4 are dev-box tests: each named test exists in its file, and each stage's package build cites its last green pytest log (a summary line `N passed` with no `failed` / `error`, rc 0 when it records one). A gate test file
    edited AFTER the cited log is a stale citation and fails (named)."""
    logs = logs or DEFAULT_GATE_LOGS
    out = {"PASS": True, "why": [], "tests": {}, "logs": {}}
    why = out["why"]
    files = set()
    for g, tests in GATE_TESTS.items():
        for rel, name in tests:
            p = f"{t2v8f_root}/{rel}"
            ok = os.path.isfile(p) and name in _tests_in(p)
            out["tests"][f"{g}: {rel}::{name}"] = ok
            files.add(p)
            if not ok:
                why.append(f"G2B_TEST_MISSING {g}: {rel}::{name}")
    for st, lp in sorted(logs.items()):
        if not lp or not os.path.isfile(lp):
            why.append(f"G2B_LOG_MISSING stage {st}: {lp}")
            continue
        t = open(lp, encoding="utf-8", errors="replace").read()
        import re
        m = re.findall(r"(\d+) passed", t)
        bad = re.search(r"\d+ (failed|error)", t) or re.search(r"\brc=[1-9]", t) or re.search(r"\bEXIT [1-9]", t)
        out["logs"][st] = {"path": lp, "sha256": sha256_file(lp), "n_passed": int(m[-1]) if m else None, "mtime": os.path.getmtime(lp)}
        if not m or bad:
            why.append(f"G2B_LOG_NOT_GREEN stage {st}: {lp} (passed {m[-1] if m else None}, failure marker {bad.group(0) if bad else None})")
        for p in sorted(files):
            if os.path.isfile(p) and os.path.getmtime(p) > os.path.getmtime(lp) + 1:
                why.append(f"G2B_LOG_STALE stage {st}: {os.path.basename(p)} was edited after {os.path.basename(lp)}")
    out["PASS"] = not why
    return out


def g2d_check(roots_arms):
    """(d) T3 recomputed on each stage's own data: ZERO(T2, objects) equals P10(T2, objects) EXACTLY on every window (the executed track, every plan, the frame md5s, `cl`, `v0`, `g`, compared as the JSON values they were written as).
    roots_arms = {stage: rollouts root} (stages 2 and 3). Every scene dir that holds a P10 cell must hold a ZERO cell, and the other way round."""
    out = {"PASS": True, "why": [], "windows": {}}
    why = out["why"]
    for st, root in sorted(roots_arms.items()):
        z, p = dict(load_cells(root, "ZERO")), dict(load_cells(root, "P10"))
        key = lambda d: os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d))))            # noqa: E731
        zz, pp = {key(d): r for d, r in z.items()}, {key(d): r for d, r in p.items()}
        for sc in sorted(set(zz) ^ set(pp)):
            why.append(f"G2D_CELL_MISSING stage {st} scene {sc}: {'ZERO' if sc not in zz else 'P10'} cell absent")
        n = 0
        for sc in sorted(set(zz) & set(pp)):
            if len(zz[sc]) != len(pp[sc]):
                why.append(f"G2D_WINDOWS_DIFFER stage {st} scene {sc}: ZERO {len(zz[sc])} records, P10 {len(pp[sc])}")
                continue
            for a, b in zip(sorted(zz[sc], key=lambda r: int(r["f0"])), sorted(pp[sc], key=lambda r: int(r["f0"]))):
                n += 1
                bad = [k for k in ZERO_P10_KEYS if a.get(k) != b.get(k)]
                if bad:
                    why.append(f"G2D_ZERO_NE_P10 stage {st} scene {sc} f0 {a.get('f0')}: {bad}")
        out["windows"][st] = n
        if n == 0:
            why.append(f"G2D_NO_WINDOWS stage {st}: no ZERO/P10 window pair under {root}")
    out["PASS"] = not why
    return out


def g2e_check(cells_by_arm_root, excluded=None):
    """(e) every deciding cell's `actor_attach.json` is present and parses, with a `verdict`; the verdict is tabulated PER SCENE per arm. cells_by_arm_root = [(label, root, arm)]. A non-ACCEPTED verdict is tabulated and
    listed in `non_accepted` (it is the Master Mind's to rule on; this gate fails only a missing or unreadable file).
    `excluded` {scene8: reason text} (rule R-SCN, the union over the stages): such a scene is out of the read and has no actor_attach.json; it is tabulated `EXCLUDED (R-SCN): <reason>` and is never a gate failure.
    Every other scene is an included scene and still needs the file."""
    out = {"PASS": True, "why": [], "per_scene": {}, "non_accepted": []}
    why = out["why"]
    for label, root, arm in cells_by_arm_root:
        for d, _ in load_cells(root, arm):
            sc = scene_of_cell(d)
            f = f"{d}/actor_attach.json"
            if excluded and sc in excluded:
                try:
                    seen = json.load(open(f, encoding="utf-8")).get("verdict")
                except Exception:                                        # noqa: BLE001
                    seen = None
                out["per_scene"].setdefault(sc, {})[f"{arm}@{label}"] = f"EXCLUDED (R-SCN): {excluded[sc]}" + (f"; verdict on file {seen}" if seen else "")
                continue
            try:
                v = json.load(open(f, encoding="utf-8")).get("verdict")
            except Exception as e:                                       # noqa: BLE001
                why.append(f"G2E_ACTOR_ATTACH_MISSING {label} {sc} {arm}: {type(e).__name__}")
                continue
            if v is None:
                why.append(f"G2E_NO_VERDICT {label} {sc} {arm}")
            out["per_scene"].setdefault(sc, {})[f"{arm}@{label}"] = v
            if v is not None and v != "ACCEPTED":
                out["non_accepted"].append(f"{sc} {arm}@{label}: {v}")
    out["PASS"] = not why
    return out


def g2_full(a, results, excluded=None):
    """G2 as the card's 17:42 reading: all of (a) to (e). `a` carries the paths; any part failing fails G2. `excluded` {scene8: reason text} = the R-SCN union, for part (e).
    The arm under read (`arm_spec(a)`; C0C at stage 4 by default): for an arm in a stage N beyond 4, N joins (a) with `--arm-overlay`, (b) with `--arm-gate-log`, (c) through `results`, and (e) with its
    own cells; a missing overlay or log is the named failure the checks already give. (d) stays on stages 2 and 3."""
    sp = arm_spec(a)
    st_a, arm = sp["stage"], sp["arm"]
    root_a = a.root4 if st_a == 4 else sp["root"]
    overlays = logs = None
    if st_a > 4:
        overlays, logs = {**DEFAULT_OVERLAYS, st_a: sp["overlay"]}, {**DEFAULT_GATE_LOGS, st_a: sp["gate_log"]}
    parts = {"a": g2a_check(getattr(a, "result1", None) or DEFAULT_RESULT1, results, getattr(a, "manifest1", None) or DEFAULT_MANIFEST1, overlays),
             "b": g2b_check(logs=logs), "c": g2_check(results),
             "d": g2d_check({2: a.root2, 3: a.root3}),
             "e": g2e_check([("stage2", a.root2, "P10"), ("stage3", a.root3, "P10"), (STAGE_LABEL[st_a], root_a, f"{arm}_s0"), (STAGE_LABEL[st_a], root_a, f"{arm}_s1")], excluded=excluded)}
    why = [f"({k}) {w}" for k, p in parts.items() for w in p.get("why", [])]
    return {"PASS": not why, "why": why, "parts": parts}


def load_cells(root, arm, tc="T2", agents="objects"):
    """-> [(cell_dir, [records])] for every cell dir `<scene8>/<agents>/<tc>/<arm>` under root that exists."""
    out = []
    for d in sorted(glob.glob(f"{root}/*/{agents}/{tc}/{arm}")):
        f = f"{d}/records.jsonl"
        recs = [json.loads(line) for line in open(f, encoding="utf-8")] if os.path.isfile(f) else []
        out.append((d.replace("\\", "/"), recs))
    return out


def cell_state(d, recs):
    """DONE with `records=N/N`, no FAIL, and as many record lines as N. -> reason or None."""
    if os.path.isfile(f"{d}/FAIL"):
        return "FAIL marker"
    if not os.path.isfile(f"{d}/DONE"):
        return "no DONE marker"
    t = open(f"{d}/DONE", encoding="utf-8").read()
    try:
        a, b = t.split("records=")[1].split()[0].split("/")
    except Exception:                                                   # noqa: BLE001
        return f"unreadable DONE text {t!r}"
    if not (int(a) == int(b) == len(recs)):
        return f"DONE says records={a}/{b}, file has {len(recs)}"
    return None


# ------------------------------------------------------------------------------------------------ rule R-SCN (AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb; the card's 19:31 log entry)
RSCN_RULE = "R-SCN, AMENDMENT_T2_SCENE_REFUSAL.md (frozen 2167a7eb)"
class _StageLabel(dict):
    """{2: 'stage2', 3: 'stage3', 4: 'stage4'}; a stage beyond 4 is spelled the same way (`stage5`)."""

    def __missing__(self, st):
        return f"stage{st}"


STAGE_LABEL = _StageLabel({2: "stage2", 3: "stage3", 4: "stage4"})
RSCN_REASONS = ("ACTOR_REFUSED", "NO_DYNAMIC_LAYER")
# The cells of ONE scene, by pass, exactly as the drivers plan them (tests pin these to t2v8f_stage2.PASSES and t2v8f_stage4.PASSES4): stages 2 and 3 run {T2, RR} x {C0_s0, C0_s1, P10, ZERO, BLIND} (deciding) and
# {T2, RR} x {DAGOACC_s0, DAGOACC_s1} (beside); stage 4 runs T2 x {C0C_s0, C0C_s1} (deciding) and RR x {C0C_s0, C0C_s1} (beside).
STAGE_CELLS = {2: {"deciding": frozenset(f"{tc}|{a}" for tc in ("T2", "RR") for a in ("C0_s0", "C0_s1", "P10", "ZERO", "BLIND")),
                   "beside": frozenset(f"{tc}|{a}" for tc in ("T2", "RR") for a in ("DAGOACC_s0", "DAGOACC_s1"))},
               4: {"deciding": frozenset(f"T2|{a}" for a in ("C0C_s0", "C0C_s1")), "beside": frozenset(f"RR|{a}" for a in ("C0C_s0", "C0C_s1"))}}
STAGE_CELLS[3] = STAGE_CELLS[2]


def arm_cells(arm):
    """The cells of the stage that holds `arm`, laid out as stage 4's: T2 x {arm_s0, arm_s1} deciding, RR x {arm_s0, arm_s1} beside. arm_cells('C0C') == STAGE_CELLS[4]."""
    return {"deciding": frozenset(f"T2|{arm}_s{s}" for s in SEEDS), "beside": frozenset(f"RR|{arm}_s{s}" for s in SEEDS)}


def stage_cells(st, arm_name=ARM_DEFAULT, arm_stage=ARM_STAGE_DEFAULT):
    """The planned cells of one scene in stage `st`: the arm's layout in the arm's stage, the drivers' layout in every other stage (stage 4 stays the C0C stage when the arm is read from a later one)."""
    if st == arm_stage:
        return arm_cells(arm_name)
    return STAGE_CELLS[st]


def arm_spec(a):
    """The arm under read and where it lives, from the parsed args (any attribute may be absent: the previous pin's tests hand over a bare namespace) -> {arm, stage, root, result, exclusions, overlay, gate_log}.
    REFUSED (SystemExit, a named error each, before anything is read): the name is not a plain name (ARM_NAME_FORM), carries a seed suffix (ARM_NAME_SEED_SUFFIX: the read appends `_s0` / `_s1`), is the floor or
    a stage-2/3 arm (ARM_NAME_RESERVED); the stage is below 4 (ARM_STAGE_BELOW_4); stage 4 with an arm other than C0C (ARM_STAGE4_IS_C0C) or with the extra `--arm-*` inputs, which would name a second
    source for stage 4 (ARM_STAGE4_EXTRA_INPUTS); a stage beyond 4 without `--arm-root` (ARM_ROOT_REQUIRED)."""
    arm = getattr(a, "arm", None)
    arm = ARM_DEFAULT if arm is None else arm
    st = getattr(a, "arm_stage", None)
    st = ARM_STAGE_DEFAULT if st is None else st
    paths = {k: getattr(a, "arm_" + k, None) for k in ("root", "result", "exclusions", "overlay", "gate_log")}
    why = []
    if not isinstance(arm, str) or not ARM_NAME_RE.fullmatch(arm):
        why.append(f"ARM_NAME_FORM {arm!r}: an arm name is letters, digits, '-' and '_' and starts with a letter")
    else:
        if ARM_SEED_SUFFIX_RE.fullmatch(arm):
            why.append(f"ARM_NAME_SEED_SUFFIX {arm!r}: give the name without its seed (the read appends _s0 and _s1)")
        if arm.upper() in ARM_RESERVED:
            why.append(f"ARM_NAME_RESERVED {arm!r}: the floor and the stage-2/3 arms {ARM_RESERVED} are not arms to read against the floor")
    if isinstance(st, bool) or not isinstance(st, int) or st < 4:
        why.append(f"ARM_STAGE_BELOW_4 {st!r}: stages 2 and 3 hold the floor P10; the arm lives in stage 4 or a later stage")
    elif st == 4:
        if isinstance(arm, str) and arm != STAGE4_ARM:
            why.append(f"ARM_STAGE4_IS_C0C {arm!r}: stage 4 holds the {STAGE4_ARM} cells; an arm of another name lives in a stage beyond 4 (--arm-stage N with --arm-root)")
        extra = sorted("--arm-" + k.replace("_", "-") for k, v in paths.items() if v)
        if extra:
            why.append(f"ARM_STAGE4_EXTRA_INPUTS {extra}: stage 4 is read from --root4 / --result4 / --exclusions4; the --arm-* inputs are for a stage beyond 4")
    elif not paths["root"]:
        why.append(f"ARM_ROOT_REQUIRED stage {st}: an arm in a stage beyond 4 needs --arm-root (and --arm-result, --arm-exclusions, --arm-overlay, --arm-gate-log)")
    if why:
        raise SystemExit("REFUSED: " + "; ".join(why))
    return {"arm": arm, "stage": st, **paths}


def arm_absent(root, arm):
    """-> [problems]: the pull under `root` holds no cell directory `<scene8>/objects/T2/<arm>_s<seed>` for a seed (a mistyped arm); the T2 cells that ARE there are named."""
    why = []
    for s in SEEDS:
        if not glob.glob(f"{root}/*/objects/T2/{arm}_s{s}"):
            present = sorted({os.path.basename(d) for d in glob.glob(f"{root}/*/objects/T2/*")})
            why.append(f"ARM_ABSENT_IN_PULL {arm}_s{s}: no cell directory {root}/<scene8>/objects/T2/{arm}_s{s}; T2 cells present under that root: {present[:12]}")
    return why


def scene_of_cell(d):
    """the 8-character scene id of a cell dir `<root>/<scene8>/<agents>/<tc>/<arm>`."""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d))))


def cell_marker(d):
    """-> (state, a, b): state DONE / FAIL / BOTH / NONE by the marker files; (a, b) = the `records=a/b` of the marker text (b = the cell's number of starts); None, None when absent or unreadable."""
    has_d, has_f = os.path.exists(f"{d}/DONE"), os.path.exists(f"{d}/FAIL")
    state = "BOTH" if has_d and has_f else "DONE" if has_d else "FAIL" if has_f else "NONE"
    a = b = None
    if state in ("DONE", "FAIL"):
        try:
            a, b = (int(x) for x in open(f"{d}/{state}", encoding="utf-8").read().split("records=")[1].split()[0].split("/"))
        except Exception:                                                  # noqa: BLE001
            a = b = None
    return state, a, b


def scene_cell_dirs(root, scene8):
    """{"<tc>|<arm>": dir} for every cell directory of the scene under `root` (the glob scene_exclusions.scene_cells uses)."""
    out = {}
    for d in sorted(glob.glob(f"{root}/{scene8}/objects/*/*")):
        if os.path.isdir(d):
            d = d.replace("\\", "/")
            out["|".join(d.split("/")[-2:])] = d
    return out


def _n_lines(path):
    return sum(1 for _ in open(path, encoding="utf-8")) if os.path.isfile(path) else 0


def _jload(x):
    return x if isinstance(x, dict) else json.load(open(x, encoding="utf-8"))


def rscn_stage_check(st, exc, res, root, scene_ids, rng, exc_path=None, res_path=None, cells=None):
    """Rule R-SCN for ONE stage (st in 2, 3, 4, or the arm's stage beyond 4): -> (problems, info). `exc` / `res` = the parsed EXCLUSIONS*.json / STAGE*_RESULT.json, `root` the stage's rollouts dir, `rng` the
    stage's scene indices, `cells` the planned cells of one scene ({"deciding", "beside"}; default the drivers' layout of that stage)."""
    why = []
    cells = STAGE_CELLS[st] if cells is None else cells
    all_cells = cells["deciding"] | cells["beside"]
    s8 = {i: scene_ids[i][:8] for i in rng if i in scene_ids}
    idx_of = {v: k for k, v in s8.items()}
    ex = exc.get("excluded") or {}
    ro = (res.get("steps") or {}).get("rollouts") or {}
    state = ro.get("state") or {}
    info = {"exclusions_file": exc_path, "exclusions_sha256": sha256_file(exc_path) if exc_path and os.path.isfile(exc_path) else None, "verdict": exc.get("verdict"), "overall": res.get("overall"),
            "n_scenes": len(rng), "n_excluded": len(ex), "cells_per_scene": {p: len(c) for p, c in cells.items()}, "excluded": {}, "extra_cell_dirs": {}}
    # (1) the file is ATTRIBUTED, of THIS stage, and built from THIS stage's result
    if exc.get("verdict") != "ATTRIBUTED" or exc.get("problems"):
        why.append(f"RSCN_NOT_ATTRIBUTED stage {st}: exclusions verdict {exc.get('verdict')!r}, problems {(exc.get('problems') or [])[:3]}")
    m = res.get("manifest_sha256")
    if not m or exc.get("manifest_sha256") != m:
        why.append(f"RSCN_OTHER_RESULT stage {st}: the exclusions were built from another stage result (manifest_sha256 {str(exc.get('manifest_sha256'))[:12]} != {str(m)[:12]})")
    if exc.get("stage") != STAGE_LABEL[st] or res.get("stage") not in (None, STAGE_LABEL[st]):
        why.append(f"RSCN_OTHER_STAGE stage {st}: the exclusions file is for {exc.get('stage')!r}, the result is {res.get('stage')!r}, this is {STAGE_LABEL[st]!r}")
    if res_path:
        bound = (exc.get("inputs") or {}).get("result_sha256")
        if not bound:
            why.append(f"RSCN_NO_RESULT_BINDING stage {st}: the exclusions file records no inputs.result_sha256 (it was not written by scene_exclusions.py)")
        elif bound != sha256_file(res_path):
            why.append(f"RSCN_OTHER_RESULT stage {st}: the exclusions were built from another stage result file (inputs.result_sha256 {bound[:12]} != {sha256_file(res_path)[:12]})")
    # (2) each excluded scene is a scene of this stage, with an index and one of the two reasons
    for sc, rec in sorted(ex.items()):
        i = idx_of.get(sc)
        if i is None:
            why.append(f"RSCN_SCENE_NOT_IN_STAGE stage {st}: excluded scene {sc} is not one of the stage's {len(rng)} scenes")
            continue
        if rec.get("idx") is None or int(rec["idx"]) != i:
            why.append(f"RSCN_IDX_MISMATCH stage {st} scene {sc}: the exclusions say index {rec.get('idx')!r}, the scene sources say {i}")
        if rec.get("reason") not in RSCN_REASONS:
            why.append(f"RSCN_REASON stage {st} scene {sc}: reason {rec.get('reason')!r} is not one of {RSCN_REASONS}")
        if rec.get("problems"):
            why.append(f"RSCN_SCENE_UNATTRIBUTED stage {st} scene {sc}: {rec['problems'][:2]}")
    # (3) the stage's failed cells are exactly all the cells of its excluded scenes (deciding and beside); every included cell is counted DONE
    failed_by = {}
    for p in ("deciding", "beside"):
        by = {}
        for f in (state.get(p) or {}).get("failed") or []:
            k = f.split("|")
            by.setdefault(k[0], []).append("|".join(k[1:]))
        failed_by[p] = by
        outside = sorted(set(by) - set(ex))
        if outside:
            why.append(f"RSCN_FAILED_OUTSIDE stage {st} {p}: failed cells outside the excluded scenes: {outside[:6]}")
        for sc in sorted(ex):
            got = by.get(sc, [])
            if len(got) != len(set(got)) or set(got) != cells[p]:
                why.append(f"RSCN_PARTIAL_SCENE stage {st} {p}: excluded scene {sc} has {len(set(got))} failed cells, not all {len(cells[p])}")
        done = int((state.get(p) or {}).get("done", 0)) + int((state.get(p) or {}).get("skipped_done", 0))
        need = len(cells[p]) * (len(rng) - len(ex))
        info[f"{p}_done"], info[f"{p}_required"] = done, need
        if done != need:
            why.append(f"RSCN_INCLUDED_NOT_DONE stage {st} {p}: {done} cells DONE, {need} required for the {len(rng) - len(ex)} included scenes")
    if ro.get("passes_skipped_disk"):
        why.append(f"RSCN_PASS_SKIPPED stage {st}: a pass was skipped for disk: {ro.get('passes_skipped_disk')}")
    if ro.get("scenes_skipped"):
        why.append(f"RSCN_SCENES_SKIPPED stage {st}: scenes skipped at preflight: {sorted(ro.get('scenes_skipped'))[:6]}")
    if ro.get("n_cells_planned") != len(all_cells) * len(rng):
        why.append(f"RSCN_CELLS_PLANNED stage {st}: {ro.get('n_cells_planned')} cells planned, {len(all_cells) * len(rng)} expected")
    if res.get("n_scenes_manifest") not in (None, len(rng)):
        why.append(f"RSCN_SCENE_COUNT stage {st}: the result ran {res.get('n_scenes_manifest')} scenes, {len(rng)} expected")
    o = res.get("overall")
    if o == "COMPLETE":
        if ex or any(by for by in failed_by.values()):
            why.append(f"RSCN_OVERALL stage {st}: overall COMPLETE but excluded scenes {sorted(ex)[:4]} / failed cells are recorded")
    elif o == "INCOMPLETE":
        if not ex:
            why.append(f"RSCN_OVERALL stage {st}: overall INCOMPLETE with no excluded scene: a cell failed or a pass was skipped for another reason")
    else:
        why.append(f"G3_OVERALL_NOT_COMPLETE stage {st}: {o!r} (rule R-SCN accepts COMPLETE, or INCOMPLETE only with the conditions above)")
    # (4) on disk: an excluded scene has every cell FAIL with 0 records and none DONE; an included scene has every planned cell DONE
    for i in rng:
        sc = s8.get(i)
        if sc is None:
            why.append(f"RSCN_SCENE_IDS stage {st}: no scene id for index {i}")
            continue
        got = scene_cell_dirs(root, sc)
        if sc in ex:
            miss = sorted(all_cells - set(got))
            if miss:
                why.append(f"RSCN_EXCLUDED_CELLS_ABSENT stage {st} scene {sc}: cell directories absent: {miss[:4]}")
            starts = set()
            for k, d in sorted(got.items()):
                stt, a, b = cell_marker(d)
                if stt != "FAIL" or a != 0:
                    why.append(f"RSCN_EXCLUDED_SCENE_HAS_DONE_CELL stage {st} scene {sc} {k}: {stt}, records={a}/{b}; every cell of an excluded scene is FAIL with 0 records")
                if b is not None:
                    starts.add(b)
            info["excluded"][sc] = {"scene": ex[sc].get("scene"), "idx": i, "reason": ex[sc].get("reason"), "cells_failed": {p: len(failed_by[p].get(sc, [])) for p in failed_by}, "cells_total": len(all_cells),
                                    "starts_lost": sorted(starts)[0] if len(starts) == 1 else None}
            if len(starts) != 1:
                why.append(f"RSCN_STARTS_UNKNOWN stage {st} scene {sc}: the cells carry start counts {sorted(starts)}")
        else:
            for k in sorted(all_cells):
                d = got.get(k)
                if d is None:
                    why.append(f"RSCN_INCLUDED_CELL_ABSENT stage {st} scene {sc} {k}: no cell directory")
                    continue
                r = cell_state(d, range(_n_lines(f"{d}/records.jsonl")))
                if r:
                    why.append(f"RSCN_INCLUDED_CELL_NOT_DONE stage {st} scene {sc} {k}: {r}")
            extra = sorted(set(got) - all_cells)
            if extra:
                info["extra_cell_dirs"][sc] = extra
    return why, info


def rscn_check(roots, results, scene_ids, exclusions, expect=EXPECT, arm_name=ARM_DEFAULT, arm_stage=ARM_STAGE_DEFAULT):
    """Rule R-SCN as G3 reads it (AMENDMENT_T2_SCENE_REFUSAL.md rules 1 to 4): the per-stage conditions of `rscn_stage_check` for stages 2, 3 and 4 (and the arm's stage `arm_stage` when it is beyond 4, laid
    out as `arm_cells(arm_name)`), then the UNION of the excluded scenes over every stage read.
    `exclusions` {stage: EXCLUSIONS*.json path or parsed dict}: all of them or none (a partial set is refused). `results` {stage: STAGE*_RESULT.json path or dict}, `roots` {stage: rollouts dir}.
    -> {"applied", "rule", "problems", "per_stage", "union": {scene8: {scene, idx, stages {stage: reason}, starts_lost}}, "starts": {...}, "scenes": {...}}. Problems here end G3 (the exclusions are its precondition)."""
    n_l, n_g = expect["scenes_launch"], expect["scenes_guard"]
    rng = {2: range(0, n_l), 3: range(n_l, n_l + n_g), 4: range(0, n_l + n_g)}
    stages = (2, 3, 4)
    if arm_stage > 4:
        stages, rng[arm_stage] = (2, 3, 4, arm_stage), range(0, n_l + n_g)
    idx_of = {v[:8]: k for k, v in scene_ids.items()}
    probs = []
    out = {"applied": True, "rule": RSCN_RULE, "problems": probs, "per_stage": {}, "union": {}, "starts": {}, "scenes": {}}
    missing = [st for st in stages if not (exclusions or {}).get(st)]
    if missing:
        need = "--exclusions2, --exclusions3 and --exclusions4" if arm_stage <= 4 else "--exclusions2, --exclusions3, --exclusions4 and --arm-exclusions"
        probs.append(f"RSCN_FLAGS_PARTIAL: no exclusions file for stage {missing}; rule R-SCN needs {need} together (a scene excluded in any stage leaves every pairing)")
        return out
    ex_by = {}
    for st in stages:
        try:
            exc, res = _jload(exclusions[st]), _jload(results[st])
        except Exception as e:                                              # noqa: BLE001
            probs.append(f"RSCN_FILE_UNREADABLE stage {st}: {type(e).__name__}: {str(e)[:120]}")
            continue
        why, info = rscn_stage_check(st, exc, res, roots[st], scene_ids, rng[st], exclusions[st] if isinstance(exclusions[st], str) else None, results[st] if isinstance(results[st], str) else None,
                                     cells=stage_cells(st, arm_name, arm_stage))
        probs += why
        out["per_stage"][str(st)] = info
        ex_by[st] = exc.get("excluded") or {}
    union = out["union"]
    for st in sorted(ex_by):
        for sc, rec in ex_by[st].items():
            u = union.setdefault(sc, {"scene": rec.get("scene") or next((v for v in scene_ids.values() if v[:8] == sc), None), "idx": idx_of.get(sc), "stages": {}})
            u["stages"][str(st)] = rec.get("reason")
    for sc, u in sorted(union.items()):
        ns = set()
        for st in stages:
            if u["idx"] is not None and u["idx"] in rng[st]:
                for d in scene_cell_dirs(roots[st], sc).values():
                    b = cell_marker(d)[2]
                    if b is not None:
                        ns.add(b)
        u["starts_lost"] = sorted(ns)[0] if len(ns) == 1 else None
        if len(ns) != 1:
            probs.append(f"RSCN_STARTS_UNKNOWN scene {sc}: the cells of the scene carry start counts {sorted(ns)}; the starts lost cannot be stated")
    lost_l = sum(u["starts_lost"] or 0 for u in union.values() if u["idx"] is not None and u["idx"] < n_l)
    lost_g = sum(u["starts_lost"] or 0 for u in union.values() if u["idx"] is not None and u["idx"] >= n_l)
    out["starts"] = {"expected_before": {"launch": expect["starts_launch"], "guard": expect["starts_guard"], "total": expect["starts_launch"] + expect["starts_guard"]},
                     "lost_launch": lost_l, "lost_guard": lost_g, "lost_total": lost_l + lost_g,
                     "read_on": {"launch": expect["starts_launch"] - lost_l, "guard": expect["starts_guard"] - lost_g, "total": expect["starts_launch"] + expect["starts_guard"] - lost_l - lost_g}}
    out["scenes"] = {"of": n_l + n_g, "excluded": len(union), "read_on": n_l + n_g - len(union)}
    return out


def rscn_labels(rs):
    """{scene8: 'REASON (stage n), ...'} of the union, for G2(e)."""
    if not rs:
        return None
    return {sc: ", ".join(f"{r} (stage {st})" for st, r in sorted(u["stages"].items())) for sc, u in rs["union"].items()}


def rscn_pairs_on_excluded(recs_by_arm, union_idx):
    """The post-condition of rule 4: no record of any arm that is paired may sit on an excluded scene. -> [problems]."""
    out = []
    for arm, recs in sorted(recs_by_arm.items()):
        on = sorted({int(r["scene"]) for r in recs} & set(union_idx))
        if on:
            out.append(f"RSCN_PAIRS_ON_EXCLUDED {arm}: records on excluded scenes {on}")
    return out


def g3_check(roots, results, scene_ids, expect=EXPECT, exclusions=None, rscn=None, arm_name=ARM_DEFAULT, arm_stage=ARM_STAGE_DEFAULT):
    """roots {2, 3, 4}: rollouts dirs; results {2, 3, 4}: STAGE*_RESULT.json; scene_ids {idx: scene id} (all 62). P10 comes from stages 2 and 3 (idx 0..41 / 42..61), C0C_s0/_s1 from stage 4.
    The arm: `<arm_name>_s0` / `_s1` from stage `arm_stage` (roots and results then also carry that stage when it is beyond 4); the defaults are C0C at stage 4. The returned records are keyed by cell name.
    Rule R-SCN: with `exclusions` {2, 3, 4} (or a ready `rscn` from `rscn_check`) the stage conditions of `rscn_check` replace the plain `overall` COMPLETE test, and the UNION of the excluded scenes is removed from
    BOTH C0C and P10 before pairing; the expected start counts fall by the starts of those scenes. Without them this is the previous pin, line for line.
    -> (gate dict, {"C0C_s0": recs, "C0C_s1": recs, "P10": recs} sorted by (scene, f0), or None on a failure)."""
    init()
    L, S = STATE["L"], STATE["S"]
    n_l, n_g = expect["scenes_launch"], expect["scenes_guard"]
    exp_l, exp_g = expect["starts_launch"], expect["starts_guard"]
    why, per_arm = [], {}
    union8, rs = frozenset(), None
    if exclusions is not None or rscn is not None:
        rs = rscn if rscn is not None else rscn_check(roots, results, scene_ids, exclusions, expect, arm_name=arm_name, arm_stage=arm_stage)
        if rs["problems"]:
            return {"PASS": False, "why": list(rs["problems"]), "pairing": None, "n_starts": None, "r_scn": rs}, None
        union8 = frozenset(rs["union"])
        exp_l, exp_g = exp_l - rs["starts"]["lost_launch"], exp_g - rs["starts"]["lost_guard"]
    else:
        for st, p in sorted(results.items()):
            o = json.load(open(p, encoding="utf-8")).get("overall") if p and os.path.isfile(p) else None
            if o != "COMPLETE":
                why.append(f"G3_OVERALL_NOT_COMPLETE stage {st}: {o!r}")
    plan = (("P10", 2, range(0, n_l)), ("P10", 3, range(n_l, n_l + n_g)), (f"{arm_name}_s0", arm_stage, range(n_l + n_g)), (f"{arm_name}_s1", arm_stage, range(n_l + n_g)))
    for arm, st, idxs in plan:
        want = {scene_ids[i][:8] for i in idxs} - union8
        cells = [(d, recs) for d, recs in load_cells(roots[st], arm) if scene_of_cell(d) not in union8]
        got = {scene_of_cell(d) for d, _ in cells}
        if got != want:
            why.append(f"G3_CELLS stage {st} {arm}: missing {sorted(want - got)[:4]} extra {sorted(got - want)[:4]}")
        for d, recs in cells:
            r = cell_state(d, recs)
            if r:
                why.append(f"G3_CELL_NOT_DONE {d}: {r}")
        per_arm.setdefault(arm, []).extend(r for _, recs in cells for r in recs)
    out_recs = {}
    for arm, recs in per_arm.items():
        recs.sort(key=lambda r: (int(r["scene"]), int(r["f0"])))
        keys = L.pair_keys(recs)
        if len(set(keys)) != len(keys):
            why.append(f"G3_DUPLICATE_STARTS {arm}")
        if len(recs) != exp_l + exp_g:
            why.append(f"G3_START_COUNT {arm}: {len(recs)} != {exp_l + exp_g}")
        if sum(1 for k in keys if k[0] < n_l) != exp_l:
            why.append(f"G3_LAUNCH_STARTS {arm}: {sum(1 for k in keys if k[0] < n_l)} != {exp_l}")
        out_recs[arm] = recs
    if rs is not None:
        why += rscn_pairs_on_excluded(out_recs, {u["idx"] for u in rs["union"].values()})
    pairing = {}
    for s in SEEDS:
        try:
            S.pair_check(L.pair_keys(out_recs.get(f"{arm_name}_s{s}", [])), L.pair_keys(out_recs.get("P10", [])))
            pairing[f"s{s}"] = "PAIRED"
        except ValueError as e:
            pairing[f"s{s}"] = f"UNPAIRED: {e}"
            why.append(f"G3_PAIRING seed {s}: {e}")
    gate = {"PASS": not why, "why": why, "pairing": pairing, "n_starts": {a: len(r) for a, r in out_recs.items()}}
    if rs is not None:
        gate["r_scn"] = rs
    return gate, (out_recs if not why else None)


def g4_check(arrays=None, bank_json=BANK_JSON, pins=None, mutate=None):
    """The reader reproduces the banked T1-kin Read A block of C0C vs P10 from its banked arrays, bit for bit. `pins` {path: sha256} (default BANK_PINS); `mutate(seed, arrays)` = RED-arm hook."""
    rr = init()
    GD, SR = rr.GD, rr.SR
    out = {"PASS": False, "why": [], "arrays": [], "banked_json": bank_json}
    pins = BANK_PINS if pins is None else pins
    for p, want in pins.items():
        got = sha256_file(p)
        if got != want:
            out["why"].append(f"G4_PIN_MISMATCH {p}: {got}")
    if out["why"]:
        return out
    bank = json.load(open(bank_json, encoding="utf-8"))
    for s in SEEDS:
        path = (arrays or {}).get(s) or f"{BANK_DIR}/arrays/C0C_s{s}.npz"
        out["arrays"].append(path)
        ar = rr.read_arrays(path)
        if mutate:
            mutate(s, ar)
        cls, _ = GD.classify(ar["v0"], ar["g"])
        rows, _ = rr.rows_of(rr.Aof(ar), rr.Bof(ar), ar["eid"], cls, N_BOOT)
        ra = rr.read_a_block(SR, GD, rows)
        rt = lambda o: json.loads(json.dumps(o, default=rr._jd))             # noqa: E731  (the banked file is a JSON round trip of the same floats)
        if rt(rows) != bank["seeds"][f"s{s}"]["C0C_minus_P10"]:
            out["why"].append(f"G4_ROWS_DIFFER seed {s}")
        if rt(ra) != bank["composite"]["read_a_C0C_vs_P10"][f"s{s}"]:
            out["why"].append(f"G4_READ_A_DIFFERS seed {s}")
    out["PASS"] = not out["why"]
    return out


# ------------------------------------------------------------------------------------------------ rows and the read
def scenes_from(recs_all, sources, root_by_stage, skip=()):
    """{scene idx: SceneLog} from the scene_<idx>.npz in each cell dir and the sequence_tracks.json of the SCENE_SOURCES files. `skip` = scene indices left out (R-SCN: the excluded scenes)."""
    import t2v8f_stage2_score as SC
    L = STATE["L"]
    tr = SC.tracks_map(None, [f"{k}={v}" for k, v in sources.items()])
    out = {}
    for root in root_by_stage.values():
        for z in glob.glob(f"{root}/**/scene_*.npz", recursive=True):
            i = int(os.path.basename(z)[6:-4])
            if i not in out and i not in skip:
                f = np.load(z)
                out[i] = L.SceneLog(f["gt_T"], f["gt_ts_us"], L.load_tracks(tr[i]) if i in tr else None)
    return out


def arrays_for_seed(recs_c, recs_p, scenes, s, arm_name=ARM_DEFAULT):
    """-> the reader's array format {labels, eid, v0, g, a_i (the ARM's rows; C0C by default), b_i (P10 rows)} for one seed; refuses a missing row or windows that differ between the arms."""
    L = STATE["L"]
    GD = STATE["rr"].GD
    rc, mc, ic = L.rows_for_records(recs_c, scenes, f"{arm_name}_s{s}|T2|objects")
    rp, mp, ip = L.rows_for_records(recs_p, scenes, "P10|T2|objects")
    if mc or mp or any(lab not in rc or lab not in rp for lab in GD.LABELS):
        raise RuntimeError(f"ROWS_MISSING: {arm_name} {mc}, P10 {mp}; the 19 rows must all be built")
    if not (np.array_equal(ic["eid"], ip["eid"]) and ic["keys"] == ip["keys"]):
        raise RuntimeError(f"WINDOWS_DIFFER between {arm_name} and P10")
    v0 = np.array([float(r["v0"]) for r in recs_c])
    g = np.stack([np.asarray(r["g"], np.float64) for r in recs_c])
    if not (np.array_equal(v0, np.array([float(r["v0"]) for r in recs_p])) and np.array_equal(g, np.stack([np.asarray(r["g"], np.float64) for r in recs_p]))):
        raise RuntimeError(f"WINDOWS_DIFFER: v0 / g of {arm_name} and P10 are not identical on the same starts")
    z = {"labels": np.array(GD.LABELS), "eid": ic["eid"], "v0": v0, "g": g}
    for i, lab in enumerate(GD.LABELS):
        z[f"a_{i}"], z[f"b_{i}"] = np.asarray(rc[lab], np.float64), np.asarray(rp[lab], np.float64)
    return z


def read_blocks(arrays_by_seed, n_boot=N_BOOT):
    """{seed: array dict} -> {seed: {"rows", "read_a", "classes"}} through the pinned reader's own `rows_of`, `GD.classify` and `read_a_block`."""
    rr = init()
    GD, SR = rr.GD, rr.SR
    out = {}
    for s, ar in arrays_by_seed.items():
        cls, _ = GD.classify(ar["v0"], ar["g"])
        rows, _ = rr.rows_of(rr.Aof(ar), rr.Bof(ar), ar["eid"], cls, n_boot)
        out[s] = {"rows": rows, "read_a": rr.read_a_block(SR, GD, rows), "classes": {c: int((cls == i).sum()) for i, c in enumerate(GD.CLASSES)}}
    return out


def family_table(rows):
    rr = STATE["rr"]
    GD, SR = rr.GD, rr.SR
    t = {}
    for lab, r in rows.items():
        fam = SR.FAMILY[lab]
        v = "separated-better" if SR.better(GD, lab, r) else "separated-worse" if SR.worse(GD, lab, r) else "unseparated"
        t.setdefault(fam, []).append({"row": lab, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "verdict": v})
    return t


def seed_gap_table(blocks):
    """Section 7: for each row, both seeds' points and the gap |X_s1 - X_s0|; a row separated-better at both seeds also carries `margin_lt_gap` (the nearer CI bound's distance from 0 is below
    the gap). 'margin' is this script's spelling of a word the card leaves open; it decides nothing."""
    rr = STATE["rr"]
    GD, SR = rr.GD, rr.SR
    t = {}
    for lab in GD.LABELS:
        r0, r1 = blocks[0]["rows"][lab], blocks[1]["rows"][lab]
        gap = abs(r1["delta"] - r0["delta"])
        both = bool(SR.better(GD, lab, r0) and SR.better(GD, lab, r1))
        t[lab] = {"delta_s0": r0["delta"], "delta_s1": r1["delta"], "gap": gap, "separated_better_both_seeds": both}
        if both:
            m = min(min(abs(r["lo"]), abs(r["hi"])) for r in (r0, r1))
            t[lab].update(clear_margin=m, margin_lt_gap=bool(m < gap))
    return t


def run(a):
    """The whole read. Never raises past a gate: a failed gate gives GATE-FAIL with no row read. (A malformed arm or stage argument is refused before anything is read: `arm_spec`.)"""
    spec = arm_spec(a)
    arm, arm_st = spec["arm"], spec["stage"]
    card_ok = card_frozen_ok()
    res = {"card": "PREREG_T2_DOMINANCE", "card_sha256_expected": CARD_SHA, "card_sha_ok": card_ok, "card_sha256_now": sha256_file(CARD) if os.path.isfile(CARD) else None, "card_frozen_bytes": CARD_FROZEN_BYTES, "script_sha256": sha256_file(os.path.abspath(__file__)), "reader_sha256": RR_SHA,
           "tier": "T2 re-perception (agents replayed at log time; not a reactive-traffic result)", "estimator": "paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only)", "gates": {}}
    res.update(arm=arm, arm_stage=arm_st, floor="P10 (stages 2 and 3)", argv=getattr(a, "argv", None))
    rr = init()
    G = res["gates"]
    G["G4"] = g4_check()
    G["G1"] = g1_check(a.g1)
    results = {2: a.result2 or f"{os.path.dirname(a.root2)}/STAGE2_RESULT.json", 3: a.result3 or f"{os.path.dirname(a.root3)}/STAGE3_RESULT.json", 4: a.result4 or f"{os.path.dirname(a.root4)}/STAGE4_RESULT.json"}
    srcs, sids = {}, {}
    for f in (a.sources2, a.sources3):
        for e in json.load(open(f, encoding="utf-8")) if f else []:
            if e["rel"] == "extracted/sequence_tracks.json":
                srcs[int(e["idx"])] = e["source"]
            sids[int(e["idx"])] = e["scene"]
    roots = {2: a.root2, 3: a.root3, 4: a.root4}
    flags = {2: getattr(a, "exclusions2", None), 3: getattr(a, "exclusions3", None), 4: getattr(a, "exclusions4", None)}
    if arm_st > 4:                                                   # the arm's own stage: its pull, its result (default beside the root) and its exclusions
        roots[arm_st], flags[arm_st] = spec["root"], spec["exclusions"]
        results[arm_st] = spec["result"] or f"{os.path.dirname(spec['root'])}/STAGE{arm_st}_RESULT.json"
    rscn_on = any(flags.values())
    complete_sids = len(sids) == sum(EXPECT[k] for k in ("scenes_launch", "scenes_guard"))
    rs = rscn_check(roots, results, sids, flags, EXPECT, arm_name=arm, arm_stage=arm_st) if rscn_on and complete_sids else None
    G["G2"] = g2_full(a, results, excluded=rscn_labels(rs))
    G["G3"], recs = g3_check(roots, results, sids, EXPECT, arm_name=arm, arm_stage=arm_st, exclusions=flags if rscn_on else None, rscn=rs) if complete_sids else (
        {"PASS": False, "why": [f"G3_SCENE_SOURCES {len(sids)} scenes, need 62"]}, None)
    if rs is not None:
        res["r_scn"] = rs
    absent = arm_absent(roots[arm_st], arm)
    if absent:                                                       # a mistyped arm: named here, so it is not buried under the cell-by-cell refusals
        G["ARM"] = {"PASS": False, "why": absent}
    ok = card_ok and all(g["PASS"] for g in G.values())
    if not card_ok:
        G["CARD"] = {"PASS": False, "why": ["the frozen part of the card (first 8497 bytes) differs from FROZEN_SHA_DOMINANCE.txt"]}
    blocks = None
    if ok:
        scenes = scenes_from(recs, srcs, roots, skip={u["idx"] for u in rs["union"].values()} if rs is not None else ())
        kw = {} if arm == ARM_DEFAULT else {"arm_name": arm}          # the default arm: the previous pin's call, unchanged
        arrays = {s: arrays_for_seed(recs[f"{arm}_s{s}"], recs["P10"], scenes, s, **kw) for s in SEEDS}
        blocks = read_blocks(arrays)
        res["by_seed"] = {f"s{s}": {"classes": b["classes"], "rows": b["rows"], "read_a": b["read_a"], "family_table": family_table(b["rows"])} for s, b in blocks.items()}
        res["seed_gap_beside"] = seed_gap_table(blocks)
        res["three_family_form_beside"] = {f"s{s}": {"D3": b["read_a"]["D3_three_families_strategic_paused"], "DOMINANT": b["read_a"]["DOMINANT_three_families_strategic_paused"]} for s, b in blocks.items()}
    res["decision"] = decide(ok, blocks)
    res["outcome"] = res["decision"]["outcome"]
    return res


def _and_list(xs):
    """['2', '3', '4'] -> '2, 3 and 4'."""
    xs = [str(x) for x in xs]
    return ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)


def write_md(res, path):
    L = [f"# DOMINANCE_T2: {res['outcome']}", "", f"Card PREREG_T2_DOMINANCE (sha256 {res['card_sha256_expected'][:8]}..., card sha ok: {res['card_sha_ok']}); script sha256 {res['script_sha256']}.",
         f"Arm read: **{res.get('arm')}** (stage {res.get('arm_stage')}, cells {res.get('arm')}_s0 and {res.get('arm')}_s1) against the floor {res.get('floor')}. argv: `{' '.join(res.get('argv') or [])}`",
         f"Tier: {res['tier']}. Estimator: {res['estimator']}.", "", "## Gates", ""]
    for k, g in sorted(res["gates"].items()):
        L.append(f"* {k}: {'PASS' if g['PASS'] else 'FAIL'} {'; '.join(g.get('why', []))[:600]}")
    rsd = res.get("r_scn")
    if rsd:
        L += ["", "## Excluded scenes (rule R-SCN)", "", f"Rule {rsd['rule']}. A scene the renderer refuses before any planner runs leaves the read for BOTH arms; a scene excluded in ANY stage is out of every pairing.",
              "The excluded scenes' starts are never re-drawn and never replaced. The result is a result on the refusal-surviving scenes, never on the 62.", ""]
        sc_, st_ = rsd.get("scenes") or {}, rsd.get("starts") or {}
        if sc_ and st_:
            L += [f"* Scenes read on: {sc_['read_on']} of {sc_['of']} ({sc_['excluded']} excluded in the union of stages {_and_list(sorted(rsd.get('per_stage') or ['2', '3', '4']))}).",
                  f"* Starts read on: {st_['read_on']['total']} of {st_['expected_before']['total']} (launch {st_['read_on']['launch']} of {st_['expected_before']['launch']}, guard {st_['read_on']['guard']} of {st_['expected_before']['guard']}); "
                  f"starts lost {st_['lost_total']} (launch {st_['lost_launch']}, guard {st_['lost_guard']}).", ""]
        L += ["| stage | scene | idx | reason | failed cells (deciding + beside) of total | starts lost |", "|---|---|---|---|---|---|"]
        for st, info in sorted((rsd.get("per_stage") or {}).items()):
            for sc, e in sorted(info.get("excluded", {}).items()):
                cf = e["cells_failed"]
                L.append(f"| {st} | {sc} | {e['idx']} | {e['reason']} | {cf.get('deciding')} + {cf.get('beside')} of {e['cells_total']} | {e['starts_lost']} |")
            if not info.get("excluded"):
                L.append(f"| {st} | none excluded in this stage | | | | |")
        L += ["", "| scene | idx | excluded in (stage: reason) | starts lost |", "|---|---|---|---|"]
        for sc, u in sorted(rsd.get("union", {}).items(), key=lambda kv: (kv[1]["idx"] is None, kv[1]["idx"])):
            L.append(f"| {sc} | {u['idx']} | " + ", ".join(f"{st}: {r}" for st, r in sorted(u["stages"].items())) + f" | {u['starts_lost']} |")
        if rsd.get("problems"):
            L += ["", "R-SCN problems: " + "; ".join(rsd["problems"])[:900]]
    g2 = res["gates"].get("G2", {}).get("parts", {}).get("e", {})
    if g2.get("per_scene"):
        L += ["", "## Actor verdict per scene (G2 e)", ""] + [f"* {sc}: {v}" for sc, v in sorted(g2["per_scene"].items())]
        if g2.get("non_accepted"):
            L += ["", "Not ACCEPTED: " + "; ".join(g2["non_accepted"])]
    d = res["decision"]
    L += ["", "## Decision", "", f"Outcome: **{d['outcome']}**. {d['reading_underpowered_first']}", ""]
    if d["per_seed"]:
        for s, p in d["per_seed"].items():
            L.append(f"* {s}: DOMINANT {p['DOMINANT']} (D1 {p['D1']}, D2 {p['D2']}, D3 {p['D3']}), three-family form {p['DOMINANT_three_families_strategic_paused']}, rows separated {p['n_rows_separated']} of 19; failing_terms: {d['failing_terms'][s]}")
        for s, b in res["by_seed"].items():
            L += ["", f"## Per-family table, {s}", "", "| family | row | delta | lo | hi | verdict |", "|---|---|---|---|---|---|"]
            for fam, rs in sorted(b["family_table"].items()):
                for r in rs:
                    L.append(f"| {fam} | {r['row']} | {r['delta']:+.4f} | {r['lo']:+.4f} | {r['hi']:+.4f} | {r['verdict']} |")
        L += ["", "## Seed gap beside (section 7)", ""] + [f"* {k}: s0 {v['delta_s0']:+.4f}, s1 {v['delta_s1']:+.4f}, gap {v['gap']:.4f}" + (f", margin {v['clear_margin']:.4f}, margin < gap: {v['margin_lt_gap']}" if "clear_margin" in v else "")
                                                   for k, v in res["seed_gap_beside"].items() if v["separated_better_both_seeds"]]
    if rsd and rsd.get("scenes") and rsd.get("starts"):
        L += ["", f"Rule R-SCN applied: this read is on the {rsd['scenes']['read_on']} refusal-surviving scenes of the 62 ({rsd['starts']['read_on']['total']} of {rsd['starts']['expected_before']['total']} starts)."]
    L += ["", "Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2, never on PhysicalAI-AV at large.",
          "The render gap (gate T1 of the crawl card) is reported beside by its own record and is never deciding. A T2 DOMINANT is not a reactive-traffic result."]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("root2", "root3", "root4", "sources2", "sources3", "g1", "out_dir"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    for k in ("result1", "result2", "result3", "result4", "manifest1"):
        ap.add_argument("--" + k, default=None)
    for k in ("exclusions2", "exclusions3", "exclusions4"):
        ap.add_argument("--" + k, default=None, help="EXCLUSIONS*.json of scene_exclusions.py for that stage (rule R-SCN: all three or none)")
    ap.add_argument("--arm", default=ARM_DEFAULT, help="the arm read against P10 in C0C's place: its cells are <NAME>_s0 and <NAME>_s1 (default C0C; give the name without its seed)")
    ap.add_argument("--arm-stage", type=int, default=ARM_STAGE_DEFAULT, help="the stage that holds the arm's cells (default 4, the C0C stage; another arm lives in a stage beyond 4)")
    for k in ("arm-root", "arm-result", "arm-exclusions", "arm-overlay", "arm-gate-log"):
        ap.add_argument("--" + k, default=None, help="for an arm in a stage beyond 4: its rollouts dir / STAGE<N>_RESULT.json / EXCLUSIONS*.json (rule R-SCN) / overlay '<tar>::pkg/MANIFEST<N>.json' (G2 a) / green pytest log (G2 b)")
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    res = run(a)
    os.makedirs(a.out_dir, exist_ok=True)
    rr = STATE["rr"]
    with open(f"{a.out_dir}/DOMINANCE_T2.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=rr._jd)
    write_md(res, f"{a.out_dir}/DOMINANCE_T2.md")
    print(json.dumps({"outcome": res["outcome"], "gates": {k: g["PASS"] for k, g in res["gates"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
