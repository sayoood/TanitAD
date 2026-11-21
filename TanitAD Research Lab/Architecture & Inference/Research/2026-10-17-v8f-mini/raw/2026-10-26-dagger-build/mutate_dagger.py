"""Mutation proof for the DAgger data side (PREREG_V8F_DAGGER sections 1, 2, 6): ONE mutant per rule, and every mutant must turn the named tests of stack/tests/test_v8f_dagger.py RED; the unmutated tree
must be GREEN before and after, and every mutated file is restored BYTE-IDENTICAL (sha256 verified).

    python mutate_dagger.py <repo root> <python exe> [--list] [--only <substring of a mutant's name>]

Each mutant is one or two exact substrings replaced once in ONE file; an anchor that is not found exactly once aborts the run before anything is touched, and every mutated text must COMPILE (a mutant that is
red because of a syntax error proves nothing). Each mutant carries the pytest -k selector of the tests that must go RED. A mutant is KILLED when the selected tests FAIL (rc != 0 and a failed test, not a
collection error); it SURVIVES when they pass -- and any survivor makes this script exit 1.

R77 (never copy a deliverable out of a tree while a mutation run is editing it): this script edits files IN PLACE under <repo root>, so snapshot the deliverables before running it and compare after; it prints
the sha256 of every file it can touch before and after.  At most ONE pytest process at a time: before each launch it waits while ``pytest_runs.py`` reports >= 2 running pytest processes.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
LIST = "--list" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
LIB = ROOT / "stack" / "tanitad" / "data" / "v8f_dagger.py"
BANK = ROOT / "stack" / "scripts" / "v8f_dagger_bank.py"
TEST = "stack/tests/test_v8f_dagger.py"
PYTEST_RUNS = "C:/Users/Admin/orcread/pytest_runs.py"

KB4 = "kb4"
SWAP = "swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap"
KB5 = "kb5"
BANKFMT = "shard_roundtrip or bank_holds or bank_rows_are"
DRAW = "draws_are or p_zero or p_is_a_parameter or draw_among"
LOSS = "reaches_the_loss"
CTRL = "kb3 or kb2 or dry_run or off_route"
# the K-B4 / K-B5 RECORDS (PROVENANCE.json controls). Three disjoint selectors: the fast unit tests of the library functions, and the in-body generator tests (the generator is run INSIDE the test body, so a generator that
# breaks FAILS the test instead of erroring a module fixture -- the runner counts only FAILED lines as a kill)
K4REC = "kb4"
K5REC = "kb5_measure or kb5_pass"
KCTL = "kbctl_gen or kbctl_merge"

MUT = [
    # ---- K-B4: the matcher (section 2)
    (LIB, "lateral cut dropped", [("    if e_lat > OFF_ROUTE_LAT_M:\n", "    if False:\n")], KB4),
    (LIB, "lateral cut inclusive (>=)", [("    if e_lat > OFF_ROUTE_LAT_M:\n", "    if e_lat >= OFF_ROUTE_LAT_M:\n")], KB4),
    (LIB, "heading cut dropped", [("    if dpsi > OFF_ROUTE_HEAD_DEG:\n", "    if False:\n")], KB4),
    (LIB, "heading cut inclusive (>=)", [("    if dpsi > OFF_ROUTE_HEAD_DEG:\n", "    if dpsi >= OFF_ROUTE_HEAD_DEG:\n")], KB4),
    (LIB, "end edge off by one", [("    if r > T - END_EDGE_ROWS:\n", "    if r >= T - END_EDGE_ROWS:\n")], KB4),
    (LIB, "end edge dropped", [("    if r > T - END_EDGE_ROWS:\n", "    if False:\n")], KB4),
    (LIB, "cut order: the end edge before the off-route cuts", [("    if e_lat > OFF_ROUTE_LAT_M:\n        return Match(r, e_lat, dpsi, OFF_LAT)\n", "    if r > T - END_EDGE_ROWS:\n        return Match(r, e_lat, dpsi, END_EDGE)\n    if e_lat > OFF_ROUTE_LAT_M:\n        return Match(r, e_lat, dpsi, OFF_LAT)\n")], KB4),
    (LIB, "stationary tie: the first row (the reader's first-minimum rule), not the row nearest in time", [("    r = int(rows[np.lexsort((rows, np.abs(rows - int(j))))[0]])", "    r = int(rows[0])")], KB4),
    (LIB, "stationary tie: the LATER row on equal time distance", [("np.lexsort((rows, np.abs(rows - int(j))))", "np.lexsort((-rows, np.abs(rows - int(j))))")], KB4),
    (LIB, "tie set = the probe's band around the nearest distance (not 'within 0.05 m of that nearest arc')", [("    tied = np.nonzero(np.abs(arc - arc[i0]) <= ARC_TIE_M)[0]", "    tied = np.nonzero(np.abs(arc - s_star) <= np.abs(arc - s_star).min() + ARC_TIE_M)[0]")], KB4),
    (LIB, "tie set = the nearest row only (no 0.05 m tolerance)", [("    tied = np.nonzero(np.abs(arc - arc[i0]) <= ARC_TIE_M)[0]", "    tied = np.array([i0])")], KB4),
    (LIB, "projection onto the infinite line, not the segment", [("        u = np.clip(u, 0.0, 1.0)\n", "")], KB4),
    (LIB, "candidate rows start at 0, not at t0", [("    P = pg[int(t0):, :3]", "    P = pg[:, :3]")], KB4),
    (LIB, "the replan step is not 10", [("REPLAN_STEPS = 10 ", "REPLAN_STEPS = 5 ")], "constants"),
    # ---- re-anchoring (K-B5)
    (LIB, "no re-anchoring: the own poses stay in the world frame", [("    return np.where(ident[:, None, None], P, out)", "    return P")], KB5 + " or " + SWAP),
    (LIB, "the identity short-circuit is dropped (own NOW == logged NOW is not bit-exact)", [("    ident = (own_now.astype(np.float32) == log_now.astype(np.float32)).all(1)", "    ident = np.zeros(len(own_now), bool)")], KB5 + " or tick0_swap"),
    (LIB, "re-anchoring drops the heading", [("P[..., 2] - own_now[:, None, 2] + log_now[:, None, 2]], -1)", "P[..., 2]], -1)")], KB5),
    # ---- the swap (section 2, the table)
    (LIB, "THE swap line is deleted: `out` is `batch`", [("            out.update(upd)                                                           # the ONE line that applies the swap: delete it and `out` is `batch`\n", "            pass\n")], SWAP),
    (LIB, "kin3 is not replaced", [('    new["kin3"][rows] = torch.from_numpy(rec["kin3"]).to(dev, batch["kin3"].dtype)\n', "")], SWAP),
    (LIB, "base is not rebuilt from the own kin3", [('    new["base"][rows] = torch.from_numpy(D.kdx_base(rec["kin3"], D.KIN_UNITS["yaw"])).to(dev, batch["base"].dtype)\n', "")], SWAP),
    (LIB, "p10_t is the frame before NOW", [('P10[:, -1].to(batch["p10_t"].dtype)', 'P10[:, -2].to(batch["p10_t"].dtype)')], SWAP),
    (LIB, "p10f keeps the logged value on every frame", [('    new["p10f"][rows] = torch.where(own_ok[..., None], torch.nan_to_num(P10, 0.0), batch["p10f"][rows])\n', "")], SWAP),
    (LIB, "pose_t ignores the age", [('new["pose_t"][rows] = pose2[k, F - 1 - batch["a_t"][rows]]', 'new["pose_t"][rows] = pose2[k, F - 1]')], SWAP),
    (LIB, "pose_s ignores the age", [('new["pose_s"][rows] = pose2[k, F - 1 - batch["a_s"][rows]]', 'new["pose_s"][rows] = pose2[k, F - 1]')], SWAP),
    (LIB, "nav progress is not rebuilt", [('    new["nav"][rows, :, 3] = torch.where(nav_ok, nav3.to(nav_r.dtype), nav_r[..., 3])\n', "")], SWAP),
    (LIB, "nav: the token / validity test is dropped (rebuilt across a manoeuvre boundary)", [("    nav_ok = nav_same & own_ok & (", "    nav_ok = own_ok & (")], SWAP),
    (LIB, "nav: progress is rebuilt on frames WITHOUT a route too (the PREREG's letter; see NAV_REBUILD_INVALID)", [("NAV_REBUILD_INVALID = False", "NAV_REBUILD_INVALID = True")], SWAP),
    (LIB, "nav: the distance behind NOW is dropped", [("    nav3 = (nav_r[:, -1:, 3].to(f64) + back.to(f64) / 100.0).clamp(0.0, 3.0)", "    nav3 = (nav_r[:, -1:, 3].to(f64) + 0.0 * back.to(f64) / 100.0).clamp(0.0, 3.0)")], SWAP),
    (LIB, "lab_s.rc keeps the logged frame (no frame change)", [('    rc[rows] = rc_frame_change(ls["rc"][rows], ls["rc_ok"][rows], batch["pose_s"][rows], new["pose_s"][rows])\n', "")], SWAP),
    (LIB, "rc: the identity at a_s = 0 is not exact (no coincident-frame short-circuit)", [("    return torch.where(same | ~rc_ok[..., None], rc, moved)", "    return torch.where(~rc_ok[..., None], rc, moved)")], SWAP),
    (LIB, "rc: placeholders (rc_ok False) are moved too", [("    return torch.where(same | ~rc_ok[..., None], rc, moved)", "    return torch.where(same, rc, moved)")], SWAP),
    (LIB, "the natural-frame logic uses the OWN pose_s on both sides (rc moved by the wrong source frame)", [('batch["pose_s"][rows], new["pose_s"][rows])\n    new["lab_s"]', 'new["pose_s"][rows], new["pose_s"][rows])\n    new["lab_s"]')], SWAP),
    # ---- the draws and the eligibility
    (LIB, "the draws come from the global RNG", [("g = torch.Generator().manual_seed(self._seed_of(step))", "g = torch.default_generator")], DRAW),
    (LIB, "eligibility is ignored (every row may be selected)", [("sel = (u[0] < p).to(dev) & elig", "sel = (u[0] < p).to(dev)")], DRAW + " or swap_reachability"),
    (LIB, "p = 0 returns a copy, not the same batch", [('            return batch\n        bad = self.law_problems()', '            return dict(batch)\n        bad = self.law_problems()')], DRAW),
    (LIB, "the draw among a window's states always takes the first", [("        k = np.minimum(np.floor(np.asarray(u, np.float64) * np.maximum(cnt, 1)).astype(np.int64), np.maximum(cnt - 1, 0))", "        k = np.zeros(len(cnt), np.int64)")], "draw_among"),
    (LIB, "a cut record is eligible (only reason == ok may be swapped in)", [("        ok = np.nonzero(self.reason == OK)[0]", "        ok = np.nonzero(self.reason >= 0)[0]")], "off_route or kb2"),
    # ---- the bank, the controls
    (LIB, "K-B3: the eval-clip refusal is removed (both checks)", [("    if b & ev:\n", "    if False:\n"), ("        D.assert_disjoint(b, ev)\n", "        pass\n")], "kb3"),
    (LIB, "K-B3: the TRAIN-subset check is removed", [("    if not b <= tr:\n", "    if False:\n")], "kb3"),
    (LIB, "K-B3: the window-list sha256 is not compared", [("    if complete and bank_window_list_sha256 != train_window_list_sha256:\n", "    if False:\n")], "kb3"),
    (LIB, "a partial bank may train an arm", [('        if self.header.get("partial", False) and not allow_partial:\n', "        if False:\n")], "dry_run"),
    (LIB, "K-B2: the on-policy check is removed", [('        if not g or str(g).lower() != str(init_ckpt_sha256).lower():\n', "        if False:\n")], "kb2"),
    (LIB, "the shard's sha256 and content hash are not verified at load", [("            if verify and sha256_file(f) != s[\"file_sha256\"]:\n", "            if False:\n"), ("            if verify and content_sha256(arrays) != s[\"content_sha256\"]:\n", "            if False:\n")], "shard_roundtrip"),
    (LIB, "the declared bank sha256 is not compared", [("        if expect_sha256 and sha.lower() != str(expect_sha256).lower():\n", "        if False:\n")], "shard_roundtrip"),
    (LIB, "COVERAGE counts states, not windows", [("        n_el, n = int((cnt > 0).sum()), int(len(w))", "        n_el, n = int(cnt.sum()), int(len(w))")], "kb2"),
    (LIB, "the rollout configuration constant says nav = logged", [('"nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map"', '"nav": "logged", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map"')], "constants"),
    (LIB, "the off-route lateral constant is moved", [("OFF_ROUTE_LAT_M = 3.0 ", "OFF_ROUTE_LAT_M = 4.0 ")], "constants or kb4"),
    (LIB, "p is given a non-zero default", [('DAGGER_DEFAULTS = {"p": 0.0, "seed": 0}', 'DAGGER_DEFAULTS = {"p": 0.3, "seed": 0}')], "constants"),
    # ---- the generator
    (BANK, "tick 0 is stored as an own state", [("        if n in G.TICKS:\n", "        if n in (0,) + G.TICKS:\n")], BANKFMT),
    (BANK, "the history window is shifted by one row", [("    idx = np.arange(j - hmax + 1, j + 1)\n", "    idx = np.arange(j - hmax, j)\n")], BANKFMT),
    (BANK, "K-B0 is not checked", [("        if n == 0 and self.tick0_check:\n", "        if False:\n")], "tick0_identity"),
    (BANK, "K-B1 never compares the replay", [('            replay["mismatch_chunks"] += int(not arrays_equal(arr, records_to_arrays(again)))\n', "            pass\n")], "kb1"),
    (BANK, "the eval-clip refusal is removed from the generator", [("        G.check_train_only(bank_clips=clips, train_clips=clips, eval_clips=eval_clips, complete=False)\n    except G.BankRefused as e:\n        raise SystemExit(f\"REFUSED: {e}\")\n    if a.expect_clips", "        pass\n    except G.BankRefused as e:\n        raise SystemExit(f\"REFUSED: {e}\")\n    if a.expect_clips")], "kb3_the_generator"),
    (BANK, "--limit-clips does not mark the bank partial", [('if k != "ckpt"}, "limit_clips": a.limit_clips, "partial": bool(a.limit_clips),', 'if k != "ckpt"}, "limit_clips": a.limit_clips, "partial": False,')], "dry_run"),
    (BANK, "every shard takes every clip", [("    positions = [i for i in range(len(sel_clips)) if i % a.n_shards == a.shard]", "    positions = [i for i in range(len(sel_clips))]")], "dry_run"),
    (BANK, "the replay set is empty", [("    replay_chunks = set(rng.choice(", "    replay_chunks = set() and set(rng.choice(")], "kb1 or generator_is_deterministic"),
    (BANK, "the rollout is not the decisive configuration (--limit logged instead of causal)", [("limit=ROLL[\"limit\"]", "limit=\"logged\"")], "bank_rows_are"),
    # ---- the K-B4 / K-B5 records the generator WRITES (PREREG section 6; the reader refuses a bank without them). Library: the functions the unit tests and the generator share
    (LIB, "K-B4 record: PASS is written True whatever the cases say", [('    return {"PASS": all(cases.get(c) is True for c in KB4_CASES), "cases": cases}', '    return {"PASS": True, "cases": cases}')], K4REC),
    (LIB, "K-B4 cases: the matcher is not run (every case True)", [("            out[name] = bool(fns[name]())", "            out[name] = True")], K4REC),
    (LIB, "K-B4 cases: a case that raises counts as passed", [("            out[name] = False\n    return out", "            out[name] = True\n    return out")], K4REC),
    (LIB, "K-B4 merge: the shards' recorded cases are ignored", [('all(isinstance(r, dict) and isinstance(r.get("cases"), dict) and r["cases"].get(c) is True for r in recs)', "True")], K4REC),
    (LIB, "K-B4 merge: an empty list of shard records passes", [("now.get(c) is True and recs and all(", "now.get(c) is True and all(")], K4REC),
    (LIB, "K-B4 merge: the cases are not re-run at the merge", [("bool(now.get(c) is True and recs and all(", "bool(recs and all(")], K4REC),
    (LIB, "K-B5 pass: True whatever the numbers say", [("    return n > 0 and all(math.isfinite(v) and v <= KB5_TOL_M for v in vals)", "    return True")], K5REC),
    (LIB, "K-B5 pass: a record that measured nothing passes", [("    return n > 0 and all(math.isfinite(v)", "    return all(math.isfinite(v)")], K5REC),
    (LIB, "K-B5 pass: the tolerance is 1e-2", [("KB5_TOL_M = 1e-4 ", "KB5_TOL_M = 1e-2 ")], K5REC),
    (LIB, "K-B5 pass: the heading of NOW is not gated", [('("max_pose_now_err_m", "max_rel_transform_err_m", "max_pose_now_yaw_err_rad")]', '("max_pose_now_err_m", "max_rel_transform_err_m")]')], K5REC),
    (LIB, "K-B5 measure: the relative transforms are not compared (reads 0)", [('("max_rel_transform_err_m", e_rel),', '("max_rel_transform_err_m", e_rel * 0.0),')], K5REC),
    (LIB, "K-B5 measure: pose_now is compared with itself (reads 0)", [("        e_now = np.hypot(*(R32[:, -1, :2] - lg[:, :2]).T)", "        e_now = np.hypot(*(R32[:, -1, :2] - R32[:, -1, :2]).T)")], K5REC),
    (LIB, "K-B5 measure: the heading of NOW is not compared (reads 0)", [("        e_nyaw = np.abs(wrap(R32[:, -1, 2] - lg[:, 2]))", "        e_nyaw = np.zeros(len(R32))")], K5REC),
    (LIB, "K-B5 measure: frames before the clip (NaN) are counted as held frames", [("        ok = np.isfinite(rb).all(-1)", "        ok = np.ones(rb.shape[:-1], bool)")], K5REC),
    (LIB, "K-B5 accumulator: a NaN is swallowed by a plain max()", [("            self.max[k] = float(np.maximum(self.max[k], m[k]))", "            self.max[k] = max(self.max[k], m[k])")], K5REC),
    (LIB, "K-B5 accumulator: a non-finite record is measured too", [('if int(r["reason"]) != NONFINITE and np.isfinite(', "if np.isfinite(")], K5REC),
    (LIB, "K-B5 combine: the MIN over shards, not the max", [("float(np.max([float(r[k]) for r in recs]))", "float(np.min([float(r[k]) for r in recs]))")], K5REC),
    (LIB, "K-B5 combine: the record counts are not summed", [('"n_records": sum(int(r.get("n_records") or 0) for r in recs),', '"n_records": max([int(r.get("n_records") or 0) for r in recs] or [0]),')], K5REC),
    # ---- the generator: where the records are fed, written, and acted on
    (BANK, "K-B5 is never fed the records (the accumulator stays empty)", [("        if kb5 is not None:\n            kb5.update(recs)                                      # K-B5, on this chunk's own records\n", "")], KCTL),
    (BANK, "K-B5 reads the logged pose at the wrong row (r' + 1)", [("np.asarray(pose_grid)[m.r][:3]", "np.asarray(pose_grid)[m.r + 1][:3]")], KCTL),
    (LIB, "K-B5 accumulator: the digest of the logged poses compared is not updated", [('            self._h.update(np.ascontiguousarray(np.asarray(r["log_now"], np.float64)).tobytes())', "            pass")], KCTL),
    (BANK, "K-B5 never sees the logged pose (log_now is NaN)", [("    log_now = np.array(np.asarray(pose_grid)[m.r][:3], np.float64) if m.r >= 0 else np.full(3, np.nan)", "    log_now = np.full(3, np.nan)")], KCTL),
    (BANK, "the shard's provenance carries no controls", [('"overlap": 0}, "controls": controls, "causal_bucket_stats"', '"overlap": 0}, "causal_bucket_stats"')], KCTL),
    (BANK, "PROVENANCE.json carries no controls", [('"partial": man["partial"], "controls": controls,', '"partial": man["partial"],')], KCTL),
    (BANK, "a failed control still writes bank.json (the failure branch is skipped)", [("    if failed_controls:", "    if False:")], KCTL),
    (BANK, "a failed control at the merge exits 0", [('NO bank.json")\n        return 6\n', 'NO bank.json")\n        return 0\n')], KCTL),
    (BANK, "a failed control at a shard (several shards) exits 0", [("    return 6 if failed else 0  ", "    return 0  ")], KCTL),
    (BANK, "the merge records K-B4 True without running the cases", [('    controls = {"K-B4": G.kb4_combine([p["controls"]["K-B4"] for p in provs]),', '    controls = {"K-B4": {"PASS": True, "cases": {c: True for c in G.KB4_CASES}},')], KCTL),
    (BANK, "the merge takes K-B5 from the first shard only", [('"K-B5": G.kb5_combine([p["controls"]["K-B5"] for p in provs])}', '"K-B5": G.kb5_combine([provs[0]["controls"]["K-B5"]])}')], KCTL),
]

if LIST:
    for i, (p, n, e, sel) in enumerate(MUT):
        print(i, p.name, "|", n, "|", sel)
    sys.exit(0)

FILES = sorted({m[0] for m in MUT})
orig = {p: p.read_bytes() for p in FILES}
sha = lambda b: hashlib.sha256(b).hexdigest()
before = {str(p.relative_to(ROOT)): sha(orig[p]) for p in FILES}
for k, v in before.items():
    print("BEFORE", v, k, flush=True)
bad_anchor = 0
for path, name, edits, sel in MUT:
    txt = orig[path].decode("utf-8")
    for a, b in edits:
        n = txt.count(a)
        if n != 1:
            bad_anchor += 1
            print(f"ANCHOR FOUND {n}x (need exactly 1): [{path.name}] {name}\n    {a!r}", flush=True)
    new = txt
    for a, b in edits:
        new = new.replace(a, b, 1)
    try:
        compile(new, str(path), "exec")
    except SyntaxError as e:
        bad_anchor += 1
        print(f"MUTANT DOES NOT COMPILE: [{path.name}] {name}: {e}", flush=True)
if bad_anchor:
    sys.exit(3)
print(f"{len(MUT)} mutants over {len(orig)} files, each anchor found exactly once, each mutant compiles", flush=True)
if "--check" in sys.argv:
    sys.exit(0)

BASE = Path(os.environ.get("DAGGER_MUT_BASETEMP") or tempfile.gettempdir()) / "pt_dagger_mut"
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}


def wait_for_slot():
    for _ in range(360):
        r = subprocess.run([PY, PYTEST_RUNS], capture_output=True, text=True)
        try:
            if int(r.stdout.strip().splitlines()[-1]) < 2:
                return
        except Exception:
            return
        time.sleep(10)
    raise SystemExit("no free pytest slot after 60 min")


def run(sel):
    wait_for_slot()
    cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(BASE), TEST, "-k", sel]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV, timeout=3600)
    out = r.stdout.strip().splitlines()
    failed = [x for x in out if x.startswith("FAILED")]
    tail = " | ".join(x[:170] for x in (failed[:1] or ([x for x in out if " passed" in x or " failed" in x or " error" in x][-1:] or [(r.stderr.strip() or "no output")[-170:]])))
    collection_error = any("ERROR collecting" in x for x in out) or ("error" in (out[-1] if out else "") and not failed and r.returncode != 0 and "failed" not in (out[-1] if out else ""))
    return r.returncode, bool(failed), collection_error, tail


def restore():
    for p, b in orig.items():
        p.write_bytes(b)


sels = sorted({m[3] for m in MUT})
for s in sels:
    rc, failed, cerr, tail = run(s)
    print(f"baseline [{s}]: rc {rc}  {tail}", flush=True)
    assert rc == 0, "the unmutated tree must be green"
killed, survived = 0, []
try:
    for path, name, edits, sel in MUT:
        if ONLY and ONLY not in name and ONLY not in path.name:
            continue
        txt = orig[path].decode("utf-8")
        for a, b in edits:
            txt = txt.replace(a, b, 1)
        path.write_text(txt, encoding="utf-8", newline="")
        try:
            rc, failed, cerr, tail = run(sel)
        finally:
            path.write_bytes(orig[path])
        red = rc != 0 and failed and not cerr
        killed += int(red)
        if not red:
            survived.append(name)
        print(f"{'RED ' if red else 'SURVIVED'}  [{path.name}] {name}   -k '{sel}'  ->  {tail}", flush=True)
finally:
    restore()
after = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in FILES}
for k, v in after.items():
    print("AFTER ", v, k, flush=True)
print("RESTORED byte-identical:", after == before, flush=True)
for s in sels:
    rc, failed, cerr, tail = run(s)
    print(f"final baseline [{s}]: rc {rc}  {tail}", flush=True)
    assert rc == 0, "the restored tree must be green"
print(f"{killed} of {killed + len(survived)} mutants killed" + (f"; SURVIVORS: {survived}" if survived else ""), flush=True)
sys.exit(0 if (not survived and after == before) else 1)
