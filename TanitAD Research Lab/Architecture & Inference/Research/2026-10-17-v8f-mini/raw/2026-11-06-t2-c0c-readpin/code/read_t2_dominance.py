"""The read of PREREG_T2_DOMINANCE (frozen, sha256 38aa9df3...), section 6: written by AGENT L on 2026-11-06 BEFORE any C0C or P10 row of stages 2, 3 or 4 was computed or read.

    python -I read_t2_dominance.py --root2 STAGE2_PULL/rollouts --root3 STAGE3_PULL/rollouts --root4 STAGE4_PULL/rollouts --sources2 stage2/SCENE_SOURCES.json
           --sources3 stage3/SCENE_SOURCES3.json --g1 C0C_ID_RESULT.json --out-dir OUT          (writes OUT/DOMINANCE_T2.json and OUT/DOMINANCE_T2.md)

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
            if r.get(MANIFEST_KEY[st]) != s["overlay_manifest_sha256"]:
                why.append(f"G2A_OVERLAY_DIFFERS stage {st} ran overlay manifest {str(r.get(MANIFEST_KEY[st]))[:12]}, on hand {s['overlay_manifest_sha256'][:12]}")
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


def g2e_check(cells_by_arm_root):
    """(e) every deciding cell's `actor_attach.json` is present and parses, with a `verdict`; the verdict is tabulated PER SCENE per arm. cells_by_arm_root = [(label, root, arm)]. A non-ACCEPTED verdict is tabulated and
    listed in `non_accepted` (it is the Master Mind's to rule on; this gate fails only a missing or unreadable file)."""
    out = {"PASS": True, "why": [], "per_scene": {}, "non_accepted": []}
    why = out["why"]
    for label, root, arm in cells_by_arm_root:
        for d, _ in load_cells(root, arm):
            sc = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d))))
            f = f"{d}/actor_attach.json"
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


def g2_full(a, results):
    """G2 as the card's 17:42 reading: all of (a) to (e). `a` carries the paths; any part failing fails G2."""
    parts = {"a": g2a_check(getattr(a, "result1", None) or DEFAULT_RESULT1, results, getattr(a, "manifest1", None) or DEFAULT_MANIFEST1),
             "b": g2b_check(logs=None), "c": g2_check(results),
             "d": g2d_check({2: a.root2, 3: a.root3}),
             "e": g2e_check([("stage2", a.root2, "P10"), ("stage3", a.root3, "P10"), ("stage4", a.root4, "C0C_s0"), ("stage4", a.root4, "C0C_s1")])}
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


def g3_check(roots, results, scene_ids, expect=EXPECT):
    """roots {2, 3, 4}: rollouts dirs; results {2, 3, 4}: STAGE*_RESULT.json; scene_ids {idx: scene id} (all 62). P10 comes from stages 2 and 3 (idx 0..41 / 42..61), C0C_s0/_s1 from stage 4.
    -> (gate dict, {"C0C_s0": recs, "C0C_s1": recs, "P10": recs} sorted by (scene, f0), or None on a failure)."""
    init()
    L, S = STATE["L"], STATE["S"]
    n_l, n_g = expect["scenes_launch"], expect["scenes_guard"]
    why, per_arm = [], {}
    for st, p in sorted(results.items()):
        o = json.load(open(p, encoding="utf-8")).get("overall") if p and os.path.isfile(p) else None
        if o != "COMPLETE":
            why.append(f"G3_OVERALL_NOT_COMPLETE stage {st}: {o!r}")
    plan = (("P10", 2, range(0, n_l)), ("P10", 3, range(n_l, n_l + n_g)), ("C0C_s0", 4, range(n_l + n_g)), ("C0C_s1", 4, range(n_l + n_g)))
    for arm, st, idxs in plan:
        want = {scene_ids[i][:8] for i in idxs}
        cells = load_cells(roots[st], arm)
        got = {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d)))) for d, _ in cells}
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
        if len(recs) != expect["starts_launch"] + expect["starts_guard"]:
            why.append(f"G3_START_COUNT {arm}: {len(recs)} != {expect['starts_launch'] + expect['starts_guard']}")
        if sum(1 for k in keys if k[0] < n_l) != expect["starts_launch"]:
            why.append(f"G3_LAUNCH_STARTS {arm}: {sum(1 for k in keys if k[0] < n_l)} != {expect['starts_launch']}")
        out_recs[arm] = recs
    pairing = {}
    for s in SEEDS:
        try:
            S.pair_check(L.pair_keys(out_recs.get(f"C0C_s{s}", [])), L.pair_keys(out_recs.get("P10", [])))
            pairing[f"s{s}"] = "PAIRED"
        except ValueError as e:
            pairing[f"s{s}"] = f"UNPAIRED: {e}"
            why.append(f"G3_PAIRING seed {s}: {e}")
    gate = {"PASS": not why, "why": why, "pairing": pairing, "n_starts": {a: len(r) for a, r in out_recs.items()}}
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
def scenes_from(recs_all, sources, root_by_stage):
    """{scene idx: SceneLog} from the scene_<idx>.npz in each cell dir and the sequence_tracks.json of the SCENE_SOURCES files."""
    import t2v8f_stage2_score as SC
    L = STATE["L"]
    tr = SC.tracks_map(None, [f"{k}={v}" for k, v in sources.items()])
    out = {}
    for root in root_by_stage.values():
        for z in glob.glob(f"{root}/**/scene_*.npz", recursive=True):
            i = int(os.path.basename(z)[6:-4])
            if i not in out:
                f = np.load(z)
                out[i] = L.SceneLog(f["gt_T"], f["gt_ts_us"], L.load_tracks(tr[i]) if i in tr else None)
    return out


def arrays_for_seed(recs_c, recs_p, scenes, s):
    """-> the reader's array format {labels, eid, v0, g, a_i (C0C rows), b_i (P10 rows)} for one seed; refuses a missing row or windows that differ between the arms."""
    L = STATE["L"]
    GD = STATE["rr"].GD
    rc, mc, ic = L.rows_for_records(recs_c, scenes, f"C0C_s{s}|T2|objects")
    rp, mp, ip = L.rows_for_records(recs_p, scenes, "P10|T2|objects")
    if mc or mp or any(lab not in rc or lab not in rp for lab in GD.LABELS):
        raise RuntimeError(f"ROWS_MISSING: C0C {mc}, P10 {mp}; the 19 rows must all be built")
    if not (np.array_equal(ic["eid"], ip["eid"]) and ic["keys"] == ip["keys"]):
        raise RuntimeError("WINDOWS_DIFFER between C0C and P10")
    v0 = np.array([float(r["v0"]) for r in recs_c])
    g = np.stack([np.asarray(r["g"], np.float64) for r in recs_c])
    if not (np.array_equal(v0, np.array([float(r["v0"]) for r in recs_p])) and np.array_equal(g, np.stack([np.asarray(r["g"], np.float64) for r in recs_p]))):
        raise RuntimeError("WINDOWS_DIFFER: v0 / g of C0C and P10 are not identical on the same starts")
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
    """The whole read. Never raises past a gate: a failed gate gives GATE-FAIL with no row read."""
    card_ok = card_frozen_ok()
    res = {"card": "PREREG_T2_DOMINANCE", "card_sha256_expected": CARD_SHA, "card_sha_ok": card_ok, "card_sha256_now": sha256_file(CARD) if os.path.isfile(CARD) else None, "card_frozen_bytes": CARD_FROZEN_BYTES, "script_sha256": sha256_file(os.path.abspath(__file__)), "reader_sha256": RR_SHA,
           "tier": "T2 re-perception (agents replayed at log time; not a reactive-traffic result)", "estimator": "paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only)", "gates": {}}
    rr = init()
    G = res["gates"]
    G["G4"] = g4_check()
    G["G1"] = g1_check(a.g1)
    results = {2: a.result2 or f"{os.path.dirname(a.root2)}/STAGE2_RESULT.json", 3: a.result3 or f"{os.path.dirname(a.root3)}/STAGE3_RESULT.json", 4: a.result4 or f"{os.path.dirname(a.root4)}/STAGE4_RESULT.json"}
    G["G2"] = g2_full(a, results)
    srcs, sids = {}, {}
    for f in (a.sources2, a.sources3):
        for e in json.load(open(f, encoding="utf-8")) if f else []:
            if e["rel"] == "extracted/sequence_tracks.json":
                srcs[int(e["idx"])] = e["source"]
            sids[int(e["idx"])] = e["scene"]
    G["G3"], recs = g3_check({2: a.root2, 3: a.root3, 4: a.root4}, results, sids) if len(sids) == sum(EXPECT[k] for k in ("scenes_launch", "scenes_guard")) else (
        {"PASS": False, "why": [f"G3_SCENE_SOURCES {len(sids)} scenes, need 62"]}, None)
    ok = card_ok and all(g["PASS"] for g in G.values())
    if not card_ok:
        G["CARD"] = {"PASS": False, "why": ["the frozen part of the card (first 8497 bytes) differs from FROZEN_SHA_DOMINANCE.txt"]}
    blocks = None
    if ok:
        scenes = scenes_from(recs, srcs, {2: a.root2, 3: a.root3, 4: a.root4})
        arrays = {s: arrays_for_seed(recs[f"C0C_s{s}"], recs["P10"], scenes, s) for s in SEEDS}
        blocks = read_blocks(arrays)
        res["by_seed"] = {f"s{s}": {"classes": b["classes"], "rows": b["rows"], "read_a": b["read_a"], "family_table": family_table(b["rows"])} for s, b in blocks.items()}
        res["seed_gap_beside"] = seed_gap_table(blocks)
        res["three_family_form_beside"] = {f"s{s}": {"D3": b["read_a"]["D3_three_families_strategic_paused"], "DOMINANT": b["read_a"]["DOMINANT_three_families_strategic_paused"]} for s, b in blocks.items()}
    res["decision"] = decide(ok, blocks)
    res["outcome"] = res["decision"]["outcome"]
    return res


def write_md(res, path):
    L = [f"# DOMINANCE_T2: {res['outcome']}", "", f"Card PREREG_T2_DOMINANCE (sha256 {res['card_sha256_expected'][:8]}..., card sha ok: {res['card_sha_ok']}); script sha256 {res['script_sha256']}.",
         f"Tier: {res['tier']}. Estimator: {res['estimator']}.", "", "## Gates", ""]
    for k, g in sorted(res["gates"].items()):
        L.append(f"* {k}: {'PASS' if g['PASS'] else 'FAIL'} {'; '.join(g.get('why', []))[:600]}")
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
    L += ["", "Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2, never on PhysicalAI-AV at large.",
          "The render gap (gate T1 of the crawl card) is reported beside by its own record and is never deciding. A T2 DOMINANT is not a reactive-traffic result."]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("root2", "root3", "root4", "sources2", "sources3", "g1", "out_dir"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    for k in ("result1", "result2", "result3", "result4", "manifest1"):
        ap.add_argument("--" + k, default=None)
    a = ap.parse_args(argv)
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
