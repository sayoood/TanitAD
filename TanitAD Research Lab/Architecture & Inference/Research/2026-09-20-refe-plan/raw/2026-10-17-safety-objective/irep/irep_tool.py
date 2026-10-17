#!/usr/bin/env python3
"""irep_tool: the three glue steps around `fp_forward_irep.py` that turn two forward reads of ONE scorer state into the `irep.json` that `navtest_E.py build --irep-json` consumes.

    python irep_tool.py drive --work W --read-dir R --python PY --refe-code D --backbone-root B --forward fp_forward_irep.py --inputs F --calib F --ckpt P --frames-dir D
                              --state NAME=path:KEY --batch 8 --seed 1 [--device cuda] [--chunks 0,1]       -> R/out/chunk_NNN.npz  (one forward process per chunk, resumable)
    python irep_tool.py expected --state R0d_B=<scorer_states_u1000.pt>:B --out EXPECTED_IREP.json      -> the independent tensor-set sha256 (dev box, from the HF copy)
    python irep_tool.py seams --chunks R/out --work R --state R0d_B                                          -> R/seam_R0d_B.npz (+ seam_BASE.npz, points_E.npz, picks_report_E.json)
    python irep_tool.py make  --state R0d_B --a-chunks A/out --b-chunks B/out --a-work A --b-work B --a-csv X.csv --b-csv Y.csv --out irep.json
                              [--expected-states EXPECTED_STATES_E.json] [--n-tokens 12146] [--batch-a 8 --batch-b 32]
                                                                                                             -> irep.json (+ irep.check.json) or exit 2 and NO irep.json

WHY `make` EXISTS.  `navtest_E.py irep` subtracts the mean PDMS of ANY two harness tables.  Handed the tables of two DIFFERENT scorer states (R0d-B and R1d-B, say) it returns a perfectly
formed `{"irep_abs_delta": ..., "unit": "PDMS x100"}` that is the TRAINING-seed contrast under the inference floor's name: the number would pass every check downstream (bar E clause 3 only
compares it).  `make` therefore refuses unless the two reads are provably ONE state, ONE checkpoint, ONE machine and ONE token set, differing in exactly what the I-rep registers
(eval batch, token order), and unless each CSV was scored from the seam built from that read's own chunks.  The value itself is then computed by the consumer, `navtest_E.py irep`,
unchanged and in its own process, and recomputed here from the two CSVs; the two must agree to 1e-9.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NAVTEST_E = HERE.parent / "navtestE" / "navtest_E.py"
FORWARD = HERE / "fp_forward_irep.py"
UNIT = "PDMS x100"
EXPECT_UPDATE = 1000
TOOL = "irep_tool/1"


class Refused(Exception):
    """a hard refusal: nothing is written"""


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================ drive: one forward process per chunk of the free-point chunk plan
def cmd_drive(a) -> int:
    plan = json.load(open(f"{a.work}/ship_plan.json", encoding="utf-8"))["chunks"]
    only = {int(x) for x in a.chunks.split(",")} if a.chunks else None
    out = Path(a.read_dir) / "out"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"batch": a.batch, "seed": a.seed, "states": sorted(a.state), "ckpt": os.path.basename(a.ckpt), "forward": os.path.basename(a.forward), "device": a.device}
    cfgp = Path(a.read_dir) / "driver_config.json"
    if cfgp.exists():
        old = json.load(open(cfgp, encoding="utf-8"))
        if old != cfg:
            raise Refused(f"{cfgp} says {old}, this call says {cfg}: a resumed read must use the settings it started with (mixed settings would make one 'read' of two)")
    else:
        json.dump(cfg, open(cfgp, "w", encoding="utf-8"), indent=1)
    shim = Path(a.read_dir) / "shim.py"                                        # `-I` drops PYTHONPATH: the model code is put on sys.path by a tiny shim (the free-point smoke's way)
    shim.write_text("import sys, runpy\nsys.dont_write_bytecode = True\nsys.path.insert(0, %r)\nfwd = sys.argv[1]\nsys.argv = [fwd] + sys.argv[2:]\nrunpy.run_path(fwd, run_name='__main__')\n" % a.refe_code,
                    encoding="utf-8")
    logs_dir = Path(a.read_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    done = 0
    for i, ch in enumerate(plan):
        if only is not None and i not in only:
            continue
        final = out / f"{ch['name']}.npz"
        if final.exists():
            print(f"skip {ch['name']} (exists)", flush=True)
            done += 1
            continue
        lf = logs_dir / f"{ch['name']}.logs"
        lf.write_text("\n".join(ch["logs"]) + "\n", encoding="utf-8")
        tmp = out / f"{ch['name']}.tmp.npz"
        cmd = [a.python, "-I", str(shim), os.path.abspath(a.forward), "run", "--inputs", a.inputs, "--calib", a.calib, "--ckpt", a.ckpt, "--frames-dir", a.frames_dir, "--device", a.device,
               "--logs-file", str(lf), "--out", str(tmp), "--eval-batch", str(a.batch), "--token-order-seed", str(a.seed), "--ckpt-md5-prefix", a.ckpt_md5_prefix]
        for s in a.state:
            cmd += ["--state", s]
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), REFE_BACKBONE_ROOT=a.backbone_root, PYTHONIOENCODING="utf-8")
        with open(logs_dir / f"{ch['name']}.log", "w", encoding="utf-8") as lg:
            rc = subprocess.call(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)
        if rc != 0:
            if tmp.exists():
                os.replace(tmp, out / f"{ch['name']}.rc{rc}.npz")
            raise Refused(f"{ch['name']}: the forward exited {rc} (rc 2 = IC2: the replicated scorer path differs from the model's own); see {logs_dir / (ch['name'] + '.log')}")
        os.replace(tmp, final)                                                 # a chunk file exists only if its forward finished
        print(f"done {ch['name']}", flush=True)
        done += 1
    print(f"ZZIREP_DRIVE_DONE {done} chunk(s) -> {out}")
    return 0


# ============================================================ expected: the independent hash of the scorer state (ONE state; navtest_E.py expected-states needs all four)
def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_expected(a) -> int:
    """the tensor-set sha256 of a scorer state, computed on the DEV BOX from the downloaded copy (never from the pod): the same record `navtest_E.py expected-states` writes, for the states given.
    `make --expected-states` reads either file.  At the R0d-R1d slot only R0d-B exists, which is why this subcommand exists."""
    import torch
    sys.path.insert(0, str(NAVTEST_E.parent / "fp"))
    import fp_forward as FF                                                  # the byte-identical copy of the worker: the very function the worker hashes with
    out = {}
    for spec in a.state:
        name, _, rest = spec.partition("=")
        path, _, key = rest.rpartition(":")
        if key not in ("A", "B") or (name[-1:] in ("A", "B") and key != name[-1]):
            raise Refused(f"{spec!r}: the key must be A or B and equal the state name's suffix")
        ss = torch.load(path, map_location="cpu", weights_only=False)
        if int(ss.get("update", -1)) != EXPECT_UPDATE:
            raise Refused(f"{path} is update {ss.get('update')}, expected {EXPECT_UPDATE}")
        out[name] = {"file": os.path.basename(path), "key": key, "update": int(ss["update"]), "sha256_tensors": FF.sha256_tensors(ss[key]), "file_md5": md5_file(path)}
    if not out:
        raise Refused("no --state given")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v["sha256_tensors"][:16] for k, v in out.items()}))
    print("ZZIREP_EXPECTED_OK")
    return 0


# ============================================================ seams: the free-point picker for ONE state (navtest_E.build_seams, imported read-only)
def import_navtest_e():
    sys.path.insert(0, str(NAVTEST_E.parent))
    sys.path.insert(0, str(NAVTEST_E.parent / "fp"))
    import navtest_E as NE
    return NE


def cmd_seams(a) -> int:
    fs = sorted(glob.glob(f"{a.chunks}/chunk_*.npz"))
    if not fs:
        raise Refused(f"no chunk_*.npz in {a.chunks}")
    if not a.allow_plain_chunks:
        for f in fs:
            m = json.loads(str(np.load(f, allow_pickle=False)["meta_json"]))
            if "irep" not in m:
                raise Refused(f"{f} has no `irep` record: it was not written by fp_forward_irep.py with a non-default option, so it is not an I-rep read")
    NE = import_navtest_e()
    try:
        rep = NE.build_seams(a.chunks, a.work, names=(a.state,), limit=a.limit, present_only=a.present_only)
    except NE.Refused as e:
        raise Refused(str(e))
    print(json.dumps({k: v for k, v in rep.items() if k != "state_info"}, indent=1)[:1500])
    print(f"ZZIREP_SEAMS_OK {a.state} {rep['n_tokens']} tokens -> {a.work}/seam_{a.state}.npz")
    return 0


# ============================================================ make: the guard and the value
def read_chunks(d: str) -> list:
    fs = sorted(glob.glob(f"{d}/chunk_*.npz"))
    out = []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        out.append({"name": os.path.basename(f), "tokens": [str(t) for t in z["token"]], "meta": json.loads(str(z["meta_json"])), "keys": list(z.files), "file": f})
    return out


def read_csv_scores(path: str) -> dict:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("token") in (None, "", "average"):
            continue
        out[r["token"]] = 100.0 * float(r["score"])
    return out


def seam_label_of(csv_path: str) -> str:
    return Path(csv_path).stem


def harness_side(work, csvp, seam, toks: set, arm_expected: str) -> tuple[bool, dict]:
    """one read's harness side: the seam is the one built from this read's chunks (arm string, token set), the harness's own counts.json says it scored THAT seam file, its own status.json
    says PASS on exactly these tokens, and the CSV carries exactly these tokens.  A missing or unreadable file is a failed gate with its reason."""
    ok, det = True, {}
    try:
        sz = np.load(seam, allow_pickle=False)
        det["seam_arm"] = str(sz["arm"])
        ok &= str(sz["arm"]) == arm_expected and {str(t) for t in sz["token"]} == toks
        label = seam_label_of(csvp)
        counts = json.load(open(Path(csvp).parent / f"{label}.counts.json", encoding="utf-8"))
        arm = str(counts.get("arm", ""))
        det["counts_arm"] = arm
        ok &= arm.startswith("SEAM:") and os.path.realpath(arm[5:]) == os.path.realpath(seam)
        st = json.load(open(Path(work) / "score_logs" / f"{label}.status.json", encoding="utf-8"))
        det["harness_status"] = st.get("status")
        ok &= st.get("status") == "PASS" and int(st.get("csv_valid_rows", -1)) == len(toks)
        sc = read_csv_scores(csvp)
        det["csv_rows"] = len(sc)
        ok &= set(sc) == toks
    except Exception as e:
        ok, det["error"] = False, f"{type(e).__name__}: {str(e)[:160]}"
    return bool(ok), det


def gates(a) -> tuple[dict, dict]:
    """-> (gate name -> {"ok": bool, "detail": ...}, facts).  Every gate is evaluated even when an earlier one fails, so the sidecar says everything that is wrong at once."""
    G, facts = {}, {}

    def gate(name, ok, detail=None):
        G[name] = {"ok": bool(ok), "detail": detail}

    A, B = read_chunks(a.a_chunks), read_chunks(a.b_chunks)
    gate("chunks_present_and_paired", bool(A) and bool(B) and [c["name"] for c in A] == [c["name"] for c in B] and os.path.realpath(a.a_chunks) != os.path.realpath(a.b_chunks),
         {"a": [c["name"] for c in A], "b": [c["name"] for c in B]})
    gate("every_chunk_is_an_irep_run", all("irep" in c["meta"] for c in A + B), [c["name"] for c in A + B if "irep" not in c["meta"]])
    if not (A and B):
        return G, facts

    def sts(cs):
        return sorted({c["meta"].get("states", {}).get(a.state, {}).get("sha256_tensors") for c in cs}, key=str)
    sa, sb = sts(A), sts(B)
    # THE gate the consumer cannot have: ONE scorer state.  Two different states would make this the TRAINING-seed contrast under the inference floor's name.
    same_state = (len(sa) == 1 and sa == sb and sa[0] is not None)
    gate("both_reads_are_one_scorer_state", same_state, {"a": [str(x)[:16] for x in sa], "b": [str(x)[:16] for x in sb], "state": a.state})
    ups = {c["meta"].get("states", {}).get(a.state, {}).get("update") for c in A + B}
    keys = {c["meta"].get("states", {}).get(a.state, {}).get("key") for c in A + B}
    gate("state_is_update_1000_and_its_key_is_its_name_suffix", ups == {EXPECT_UPDATE} and (keys == {a.state[-1]} if a.state[-1:] in ("A", "B") else len(keys) == 1), {"update": sorted(map(str, ups)), "key": sorted(map(str, keys))})
    if a.expected_states:
        ex = json.load(open(a.expected_states, encoding="utf-8")).get(a.state, {})
        gate("state_equals_the_independently_computed_hash", bool(sa) and sa == [ex.get("sha256_tensors")], {"expected": str(ex.get("sha256_tensors"))[:16], "got": [str(x)[:16] for x in sa]})
    else:
        gate("state_equals_the_independently_computed_hash", False, "no --expected-states given: the read is not bound to a hash computed outside the forward (pass the main read's EXPECTED_STATES_E.json)")
    ident = {k: sorted({str(c["meta"].get(k)) for c in A + B}) for k in ("ckpt_md5", "device", "gpu", "torch", "cuda")}
    gate("one_checkpoint_and_one_machine", all(len(v) == 1 for v in ident.values()), ident)
    tokA = [sorted(c["tokens"]) for c in A]
    tokB = [sorted(c["tokens"]) for c in B]
    allA = [t for c in A for t in c["tokens"]]
    gate("same_tokens_per_chunk_and_the_expected_count", tokA == tokB and len(set(allA)) == len(allA) == a.n_tokens, {"n_a": len(allA), "n_b": sum(len(c["tokens"]) for c in B), "expected": a.n_tokens})
    ra, rb = [c["meta"].get("irep", {}) for c in A], [c["meta"].get("irep", {}) for c in B]
    seeds_a, seeds_b = {r.get("token_order_seed") for r in ra}, {r.get("token_order_seed") for r in rb}
    batch_a, batch_b = {r.get("eval_batch") for r in ra}, {r.get("eval_batch") for r in rb}
    # NOTE (defence in depth, on purpose): two reads with the same token-order seed are caught TWICE, by `seeds_a != seeds_b` and by the permutation hashes differing (same seed and same chunk
    # size give the same permutation).  Removing one layer is not detectable end to end; the test's mutation removes both.
    settings_ok = (batch_a == {a.batch_a} and batch_b == {a.batch_b} and a.batch_a != a.batch_b and len(seeds_a) == 1 and len(seeds_b) == 1 and None not in seeds_a | seeds_b
                   and seeds_a != seeds_b and all((x.get("order_sha256") != y.get("order_sha256")) or x.get("n", 0) < 2 for x, y in zip(ra, rb)))
    gate("settings_are_the_registered_ones_and_differ", settings_ok, {"batch_a": sorted(map(str, batch_a)), "batch_b": sorted(map(str, batch_b)), "seed_a": sorted(map(str, seeds_a)), "seed_b": sorted(map(str, seeds_b)),
                                                                         "registered": [a.batch_a, a.batch_b]})
    rep_ok = True
    for c in A + B:
        r = c["meta"].get("irep", {})
        sizes = {int(k): int(v) for k, v in r.get("batch_size_counts", {}).items()}
        rep_ok &= bool(sizes) and sum(k * v for k, v in sizes.items()) == len(c["tokens"]) == r.get("n") and max(sizes) == min(int(r.get("eval_batch", 0)), len(c["tokens"]))
    gate("self_reported_batch_sizes_are_consistent", rep_ok, None)
    # the harness side: each CSV was scored from the seam built from THIS read's own chunks
    for side, work, csvp, cs in (("a", a.a_work, a.a_csv, A), ("b", a.b_work, a.b_csv, B)):
        ok, det = harness_side(work, csvp, Path(work) / f"seam_{a.state}.npz", {t for c in cs for t in c["tokens"]}, f"SOBJE_{a.state}")
        gate(f"read_{side}_csv_was_scored_from_its_own_seam", ok, det)
    gate("the_two_reads_are_two_files", os.path.realpath(a.a_csv) != os.path.realpath(a.b_csv) and os.path.realpath(a.a_work) != os.path.realpath(a.b_work), None)
    facts["n_tokens"] = len(allA)
    facts["settings"] = {"a": {"eval_batch": sorted(map(str, batch_a)), "token_order_seed": sorted(map(str, seeds_a))}, "b": {"eval_batch": sorted(map(str, batch_b)), "token_order_seed": sorted(map(str, seeds_b))}}
    # diagnostics (not gates): how far apart are the two reads in logits and in picks, so that a 0.0 can be read
    try:
        la = {}
        for side, cs in (("a", A), ("b", B)):
            la[side] = {}
            for c in cs:
                z = np.load(c["file"], allow_pickle=False)
                for i, t in enumerate(c["tokens"]):
                    la[side][t] = z[f"logits_{a.state}"][i]
        d = max(float(np.abs(la["a"][t] - la["b"][t]).max()) for t in la["a"])
        facts["logit_maxabs_a_vs_b"] = d
        ta, tb = np.load(Path(a.a_work) / "points_E.npz", allow_pickle=False), np.load(Path(a.b_work) / "points_E.npz", allow_pickle=False)
        ia, ib = {str(t): i for i, t in enumerate(ta["token"])}, {str(t): i for i, t in enumerate(tb["token"])}
        facts["pick_changes_a_vs_b"] = int(sum(int(ta[f"pick_{a.state}"][ia[t]] != tb[f"pick_{a.state}"][ib[t]]) for t in ia if t in ib))
    except Exception as e:
        facts["diagnostics_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return G, facts


def run_consumer(csv_a: str, csv_b: str) -> dict:
    """the consumer, unchanged, in its own process: `navtest_E.py irep --a --b --out`"""
    with tempfile.TemporaryDirectory(prefix="irep_consumer_", dir=str(HERE)) as td:
        out = os.path.join(td, "irep_consumer.json")
        r = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(NAVTEST_E), "irep", "--a", csv_a, "--b", csv_b, "--out", out], capture_output=True, text=True, cwd=str(NAVTEST_E.parent))
        if r.returncode != 0 or not os.path.exists(out):
            raise Refused(f"navtest_E.py irep exited {r.returncode}: {(r.stdout + r.stderr).strip()[-400:]}")
        return json.load(open(out, encoding="utf-8"))


def cmd_make(a) -> int:
    G, facts = gates(a)
    failed = [k for k, v in G.items() if not v["ok"]]
    sidecar = Path(a.out).with_suffix(".check.json")
    side = {"tool": TOOL, "state": a.state, "gates": G, "failed": failed, "facts": facts}
    json.dump(side, open(sidecar, "w", encoding="utf-8"), indent=1)
    if failed:
        print(f"REFUSED: {len(failed)} gate(s) failed: {failed} (details in {sidecar}); no {a.out} written")
        return 2
    cons = run_consumer(a.a_csv, a.b_csv)
    sa, sb = read_csv_scores(a.a_csv), read_csv_scores(a.b_csv)
    toks = sorted(set(sa) & set(sb))
    mine = abs(float(np.mean([sa[t] for t in toks])) - float(np.mean([sb[t] for t in toks])))
    if not (math.isfinite(cons["irep_abs_delta"]) and abs(cons["irep_abs_delta"] - mine) <= 1e-9 and cons["unit"] == UNIT and cons["n_tokens"] == a.n_tokens == len(toks)):
        print(f"REFUSED: the consumer says {cons['irep_abs_delta']!r} ({cons['unit']!r}, n {cons['n_tokens']}) but the two CSVs give {mine!r} on {len(toks)} tokens; no {a.out} written")
        return 2
    cons["irep_guard"] = {"tool": TOOL, "state": a.state, "gates_all_pass": True, "check_sidecar": os.path.abspath(sidecar), "check_sidecar_sha256": sha256_file(sidecar), "settings": facts["settings"],
                          "independent_recompute_abs_delta": mine}
    if cons["irep_abs_delta"] == 0.0:
        cons["note"] = ("irep_abs_delta is EXACTLY 0.0: read facts.pick_changes_a_vs_b and facts.logit_maxabs_a_vs_b in the sidecar to tell 'no pick changed' from 'the two reads were bit-identical'; "
                        "E clause 3 then rests on the other floors")
    tmp = str(a.out) + ".tmp"
    json.dump(cons, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"irep_abs_delta": cons["irep_abs_delta"], "unit": cons["unit"], "n_tokens": cons["n_tokens"], "state": a.state, "facts": facts}))
    print("ZZIREP_MAKE_OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("drive")
    p.add_argument("--work", required=True)
    p.add_argument("--read-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--refe-code", required=True)
    p.add_argument("--backbone-root", default="E:/Projects/TanitAD/data/backbones")
    p.add_argument("--forward", default=str(FORWARD))
    p.add_argument("--inputs", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-md5-prefix", default="8e382eeb")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--state", action="append", default=[])
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--chunks", default=None, help="DRY RUNS ONLY: comma-separated chunk indices")
    p = sp.add_parser("expected")
    p.add_argument("--state", action="append", default=[], help="NAME=path:KEY")
    p.add_argument("--out", required=True)
    p = sp.add_parser("seams")
    p.add_argument("--chunks", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--state", default="R0d_B")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--present-only", action="store_true", help="DRY RUNS ONLY")
    p.add_argument("--allow-plain-chunks", action="store_true", help="TEST ONLY: accept chunks that carry no `irep` record (the real free-point dry chunk)")
    p = sp.add_parser("make")
    p.add_argument("--state", default="R0d_B")
    p.add_argument("--a-chunks", required=True)
    p.add_argument("--b-chunks", required=True)
    p.add_argument("--a-work", required=True)
    p.add_argument("--b-work", required=True)
    p.add_argument("--a-csv", required=True)
    p.add_argument("--b-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expected-states", default=None)
    p.add_argument("--n-tokens", type=int, default=12146)
    p.add_argument("--batch-a", type=int, default=8)
    p.add_argument("--batch-b", type=int, default=32)
    a = ap.parse_args(argv)
    try:
        return {"drive": cmd_drive, "expected": cmd_expected, "seams": cmd_seams, "make": cmd_make}[a.cmd](a)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
