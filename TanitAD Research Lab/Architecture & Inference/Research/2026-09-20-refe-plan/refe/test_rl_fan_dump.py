#!/usr/bin/env python3
"""Tests for refe/rl_fan_dump.py (RL-S1 priority 1).  Run: python refe/test_rl_fan_dump.py  ->  ZZTEST_PASS <n>   (prints the mutation table).

CPU only.  A deterministic ROW-INDEPENDENT stub model stands in for REFe and a stub dataset for train.TargetBank, so what is tested is everything the
dumper itself does: key lists, the partition-manifest check, sharding, the record format, the checkpoint it loads, resume, and the bit-identity gate.
The record is also read by the REAL consumers (train.OnPolicyBank, onpolicy_relabel_pdm.served_keys, select_bank_subset.head_fields).
Expected values are HAND-DERIVED (the arithmetic is next to each) or recomputed with independent code, never read back from the module under test.
Each mutation edits the module's SOURCE (the snippet must occur exactly once, else the run errors: a vacuous mutation proves nothing) and the suite
must then report >= 1 failing check.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_fan_dump.py")
sys.path.insert(0, HERE)
import torch  # noqa: E402

M_, T_ = 4, 3                                                        # stub fan: 4 proposals x 3 steps


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ------------------------------------------------------------------------------------------------------------------ stubs
class StubModel(torch.nn.Module):
    """Row-independent: traj[b, m, t, c] = img[b].sum() * w[m, t, c] + ego[b].sum() * v[m, t, c].  w, v ARE the checkpoint."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(M_, T_, 3))
        self.v = torch.nn.Parameter(torch.zeros(M_, T_, 3))

    def forward(self, img, ego, goal, calib=None):
        a = img.flatten(1).sum(1).view(-1, 1, 1, 1)
        e = ego.flatten(1).sum(1).view(-1, 1, 1, 1)
        return a * self.w + e * self.v, torch.zeros(img.shape[0], M_, 6)


def write_ckpt(path, seed, step=10075):
    g = torch.Generator().manual_seed(seed)
    torch.save({"model": {"w": torch.randn(M_, T_, 3, generator=g), "v": torch.randn(M_, T_, 3, generator=g)},
                "meta": {"step": step, "per_sample_calib": True}}, path)


def stub_loader(model, path, meta_out):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd["model"])
    meta_out.update(sd["meta"])
    stub_loader.loaded = path


class StubDS:
    def __init__(self, rows):
        self.rows = rows
        self.n_dropped_missing = 0
        self.n_dropped_calib = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        z = torch.zeros(0, dtype=torch.float64)
        return (torch.full((2,), float(r["_x"])), torch.full((2,), float(r["_e"])), torch.zeros(2), torch.zeros(1), None, None, None, None, z)


LOGS = [f"L{k:02d}" for k in range(9)]
ROWS = [{"log_name": LOGS[k], "token": f"tok{k}_{j}", "step": 0, "rank": 0, "aug": None, "_x": 0.01 * (k * 7 + j + 1), "_e": 0.5 * j + 1.0}
        for k in range(9) for j in range(3 + k % 3)]                  # 9 logs, 3/4/5 scenes each: 3+4+5 = 12 per 3 logs, x3 = 36 scenes
ROWS += [dict(r, rank=1, aug={"route": "x"}) for r in ROWS[:5]]       # 5 augmented twins the key list never names


def ds_factory(a):
    return StubDS([dict(r) for r in ROWS]), M_, T_


def write_tsv(path, scenes, header="token\tlog"):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n" + "".join(f"{t}\t{l}\n" for t, l in scenes))


SCENES = [(r["token"], r["log_name"]) for r in ROWS if r["rank"] == 0]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sets_of(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def expected_props(model_ckpt, row):
    """The stub's output for one row, recomputed with plain tensors from the checkpoint file."""
    sd = torch.load(model_ckpt)["model"]
    a, e = 2 * row["_x"], 2 * row["_e"]                                # the stub sums a 2-vector of one value
    return a * sd["w"] + e * sd["v"]


def run_dump(rl, d, *, ckpt, args=(), scenes=None, out="out"):
    tsv = os.path.join(d, "scenes.tsv")
    write_tsv(tsv, SCENES if scenes is None else scenes)
    ap = rl.build_parser()
    a = ap.parse_args(["--ckpt", ckpt, "--keys", tsv, "--targets", "unused", "--images", "unused", "--calib", "unused", "--out", os.path.join(d, out),
                       "--device", "cpu", "--batch", "3", "--chunk", "5", *args])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rl.run(a, build=StubModel, loader=stub_loader, ds_factory=ds_factory)
    return rc, buf.getvalue(), os.path.join(d, out)


# ------------------------------------------------------------------------------------------------------------------ the checks
def suite(rl, tag="rl_fan_dump"):
    """-> list of (check name, ok, detail).  `rl` is the module under test (possibly a mutant)."""
    out = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    def guarded(name, fn):
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            check(name, False, f"raised {type(e).__name__}: {str(e)[:120]}")

    with tempfile.TemporaryDirectory() as d:
        ckA, ckB = os.path.join(d, "model_final.pt"), os.path.join(d, "ckpt_last.pt")
        write_ckpt(ckA, 1)
        write_ckpt(ckB, 2)

        # ---- key lists
        def t_scenes():
            p = os.path.join(d, "ok.tsv")
            write_tsv(p, [("a", "L1"), ("b", "L2")])
            check("read_scenes: header + rows", rl.read_scenes(p) == [("a", "L1"), ("b", "L2")])
            pg = os.path.join(d, "ok.tsv.gz")
            with gzip.open(pg, "wt", encoding="utf-8", newline="\n") as f:
                f.write("token\tlog\na\tL1\n")
            check("read_scenes: .gz", rl.read_scenes(pg) == [("a", "L1")])
            for nm, body in (("no header", "a\tL1\n"), ("dup token", "token\tlog\na\tL1\na\tL2\n"), ("short row", "token\tlog\na\n"), ("empty log", "token\tlog\na\t\n")):
                q = os.path.join(d, "bad.tsv")
                open(q, "w", newline="\n").write(body)
                try:
                    rl.read_scenes(q)
                    check(f"read_scenes refuses {nm}", False)
                except ValueError:
                    check(f"read_scenes refuses {nm}", True)
        guarded("read_scenes", t_scenes)

        def t_manifest():
            tr, va = os.path.join(d, "rl_train_scenes.tsv"), os.path.join(d, "rl_val_scenes.tsv")
            write_tsv(tr, [("a", "L1")])
            write_tsv(va, [("b", "L2")])
            man = {"verify": "clean", "partition_id": "pid", "train": {"file": "rl_train_scenes.tsv", "sha256": sha(tr)}, "val": {"file": "rl_val_scenes.tsv", "sha256": sha(va)}}
            mp = os.path.join(d, "man.json")
            json.dump(man, open(mp, "w"))
            r = rl.verify_scene_files([tr, va], mp)
            check("manifest: matching files accepted", r["partition_id"] == "pid" and set(r["files"]) == {"rl_train_scenes.tsv", "rl_val_scenes.tsv"})
            write_tsv(tr, [("a", "L1"), ("c", "L3")])                 # one extra scene: a different list under the same name
            try:
                rl.verify_scene_files([tr], mp)
                check("manifest: a changed list is REFUSED", False)
            except ValueError:
                check("manifest: a changed list is REFUSED", True)
            write_tsv(tr, [("a", "L1")])
            other = os.path.join(d, "other.tsv")
            write_tsv(other, [("a", "L1")])
            try:
                rl.verify_scene_files([other], mp)
                check("manifest: a file the manifest does not name is REFUSED", False)
            except ValueError:
                check("manifest: a file the manifest does not name is REFUSED", True)
            man["verify"] = "violations"
            json.dump(man, open(mp, "w"))
            try:
                rl.verify_scene_files([tr], mp)
                check("manifest: verify != clean is REFUSED", False)
            except ValueError:
                check("manifest: verify != clean is REFUSED", True)
        guarded("manifest", t_manifest)

        # ---- sharding
        def t_shard():
            import rl_terms_label as RT
            ok = all(rl.shard_of(f"2021.06.{k:04d}_veh-35_{k}", n) == RT.shard_of(f"2021.06.{k:04d}_veh-35_{k}", n) for k in range(300) for n in (1, 2, 3, 5, 8))
            check("shard_of == rl_terms_label.shard_of (dump shard i and label shard i see the same logs)", ok)
            # disjoint and exhaustive over the stub bank, for several N
            rows = [dict(r) for r in ROWS]
            allk = sorted((r["log_name"], r["token"]) for r in rows if r["rank"] == 0)
            for n in (1, 2, 3, 4):
                got, per = [], []
                for i in range(n):
                    idx, miss, mism = rl.plan(rows, SCENES, [0], (i, n))
                    keys = [(rows[j]["log_name"], rows[j]["token"]) for j in idx]
                    got += keys
                    per.append(set(keys))
                check(f"shards {n}: exhaustive (union = all {len(allk)} scenes)", sorted(got) == allk, f"{len(got)} vs {len(allk)}")
                check(f"shards {n}: disjoint (no scene in two shards)", len(got) == len(set(got)), f"{len(got) - len(set(got))} duplicated")
                check(f"shards {n}: a LOG is never split", all(len({k[0] for k in s} & {k[0] for t in per if t is not s for k in t}) == 0 for s in per))
            try:
                rl.parse_shard("3/3")
                check("parse_shard refuses I >= N", False)
            except ValueError:
                check("parse_shard refuses I >= N", True)
        guarded("shard", t_shard)

        # ---- plan
        def t_plan():
            rows = [dict(r) for r in ROWS]
            shuffled = list(reversed(SCENES))
            i1, m1, x1 = rl.plan(rows, SCENES, [0], (0, 1))
            i2, m2, x2 = rl.plan(rows, shuffled, [0], (0, 1))
            check("plan: independent of the key list's order", i1 == i2 and len(i1) == len(SCENES), f"{len(i1)}")
            ks = [(rows[j]["log_name"], rows[j]["token"]) for j in i1]
            check("plan: sorted (log, token) order", ks == sorted(ks))
            ii, mm, xx = rl.plan(rows, SCENES + [("nope", "L00")], [0], (0, 1))
            check("plan: a scene with no bank row is MISSING, not skipped silently", mm == [("nope", "L00", 0)] and len(ii) == len(SCENES), str(mm))
            ii, mm, xx = rl.plan(rows, [("tok0_0", "L05")], [0], (0, 1))
            check("plan: a token under another log is a LOG MISMATCH (a defect), not a coverage gap", xx == [("tok0_0", "L05", 0)] and mm == [], f"{mm} {xx}")
            i01, m01, _ = rl.plan(rows, SCENES, [0, 1], (0, 1))
            check("plan: ranks 0,1 -> 36 plain + the 5 twins (the others have no rank-1 row: missing)", len(i01) == len(SCENES) + 5 and len(m01) == len(SCENES) - 5, f"{len(i01)} {len(m01)}")
            ir, _, _ = rl.plan(rows, SCENES, [1], (0, 1))
            check("plan: --ranks 1 selects only the augmented twins", all(rows[j]["rank"] == 1 for j in ir) and len(ir) == 5)
            try:
                rl.plan(rows + [dict(rows[0])], SCENES, [0], (0, 1))
                check("plan: a duplicated bank row is REFUSED", False)
            except ValueError:
                check("plan: a duplicated bank row is REFUSED", True)
        guarded("plan", t_plan)

        # ---- the record
        def t_record():
            row = {"log_name": "LOG", "token": "tk", "step": 0, "rank": 0, "aug": None}
            props = [[[0.123456789, -1.000004, 3.14159265], [10.0, 0.5, -0.000001]] for _ in range(2)]
            r = rl.set_record(row, 0, 10075, props, "abcd" * 4, n_prop=2, horizon=2)
            # 0.123456789 -> 0.12346 ; -1.000004 -> -1.0 ; 3.14159265 -> 3.14159 (yaw, 6 dp of a 5-dp value) ; -0.000001 -> -0.0 (5 dp) -> yaw 6 dp of -0.0
            check("record: x, y rounded to 5 dp (hand-derived)", r["traj"][0][0] == [0.12346, -1.0] and r["traj"][1][1] == [10.0, 0.5], str(r["traj"][0]))
            check("record: yaw is the dump's 5 dp value (hand-derived)", r["yaw"][0][0] == 3.14159 and r["yaw"][1][1] == 0.0, str(r["yaw"][0]))
            head = list(r)[:9]
            check("record: key fields come FIRST (the PDM labeller's --served-only reads head + tail only)", head == ["kind", "label_version", "comfort_source", "log_name", "token", "step", "rank", "ckpt_step", "aug"], str(head))
            check("record: kind / ckpt_step / label_version 0 (never supersedes a teacher-labelled set)", r["kind"] == "onpolicy_set" and r["ckpt_step"] == 10075 and r["label_version"] == 0)
            check("record: M empty-dict target placeholders + marked lean", r["targets"] == [{}, {}] and r["lean"] is True and r["teacher_targets"] == {})
            try:
                rl.set_record(row, 0, 1, props[:1], n_prop=2, horizon=2)
                check("record: a fan with a dropped proposal is REFUSED", False)
            except ValueError:
                check("record: a fan with a dropped proposal is REFUSED", True)
            try:
                rl.set_record(row, 0, 1, [p[:1] for p in props], n_prop=2, horizon=2)
                check("record: a fan with a dropped horizon step is REFUSED", False)
            except ValueError:
                check("record: a fan with a dropped horizon step is REFUSED", True)
        guarded("record", t_record)

        # ---- end to end with the stub model
        def t_e2e():
            rc, so, od = run_dump(rl, d, ckpt=ckA)
            f = os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl")
            recs = sets_of(f)
            check("dump: exit 0, every scene written once", rc == 0 and len(recs) == len(SCENES) and len({rl.key_of(r) for r in recs}) == len(SCENES), f"rc {rc}, {len(recs)}")
            byk = {(r["log_name"], r["token"]): r for r in recs}
            row = next(r for r in ROWS if r["rank"] == 0 and r["token"] == "tok4_2")
            exp = expected_props(ckA, row)
            got = byk[(row["log_name"], row["token"])]
            ok = all(abs(got["traj"][m][t][c] - float(exp[m, t, c])) < 6e-6 for m in range(M_) for t in range(T_) for c in (0, 1)) \
                and all(abs(got["yaw"][m][t] - float(exp[m, t, 2])) < 6e-6 for m in range(M_) for t in range(T_))
            check("dump: proposal m, step t of the record IS the model's output m, t (order kept, none dropped)", ok and len(got["traj"]) == M_ and len(got["yaw"]) == M_)
            check("dump: every record has all M proposals x T steps", all(len(r["traj"]) == M_ and all(len(p) == T_ for p in r["traj"]) for r in recs))
            man = json.load(open(os.path.join(od, "rl_fan_dump_fan_s0of1.manifest.json")))
            check("manifest: records the sha256 of the FILE --ckpt named, and its step", man["ckpt_sha256"] == sha(ckA) and man["ckpt_step"] == 10075 and man["finished"] is True)
            check("manifest: the file stub_loader actually loaded is the one hashed", stub_loader.loaded == ckA)
            check("dump: a different --ckpt file gives a different fan (the checkpoint is what is dumped)",
                  run_dump(rl, d, ckpt=ckB, out="out_ckB")[0] == 0 and sets_of(os.path.join(d, "out_ckB", "onpolicy_rlfan_fan_s0of1.jsonl"))[0]["traj"] != recs[0]["traj"])
            # shards: each shard run separately -> the union of lines equals the single run, byte for byte
            lines_all = open(f, encoding="utf-8").read().splitlines()
            union = []
            for i in range(3):
                rc3, _, od3 = run_dump(rl, d, ckpt=ckA, args=("--shard", f"{i}/3"), out=f"sh{i}")
                union += open(os.path.join(od3, f"onpolicy_rlfan_fan_s{i}of3.jsonl"), encoding="utf-8").read().splitlines()
            check("shards: 3 shard runs reproduce the single run exactly (same lines, none twice)", sorted(union) == sorted(lines_all), f"{len(union)} vs {len(lines_all)}")
            # batch composition: a different batch size / prefetch must not change one byte
            _, _, odb = run_dump(rl, d, ckpt=ckA, args=("--batch", "1", "--workers", "2"), out="out_batch1")
            check("batch size 1 + prefetch threads: identical file", open(os.path.join(odb, "onpolicy_rlfan_fan_s0of1.jsonl"), encoding="utf-8").read() == "\n".join(lines_all) + "\n")
        guarded("e2e", t_e2e)

        # ---- resume
        def t_resume():
            rc, _, od = run_dump(rl, d, ckpt=ckA, args=("--limit", "10"), out="res")
            f = os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl")
            n1 = len(sets_of(f))
            with open(f, "a", encoding="utf-8", newline="\n") as fh:                 # a kill mid-write: one torn line
                fh.write('{"kind": "onpolicy_set", "log_name": "L0')
            rc2, _, od = run_dump(rl, d, ckpt=ckA, out="res")
            full = run_dump(rl, d, ckpt=ckA, out="resfull")[2]
            good = [l for l in open(f, encoding="utf-8").read().splitlines()]
            parsed, torn = [], 0
            for l in good:
                try:
                    parsed.append(json.loads(l))
                except json.JSONDecodeError:
                    torn += 1
            ref = sets_of(os.path.join(full, "onpolicy_rlfan_fan_s0of1.jsonl"))
            check("resume: 10 sets, a torn line, then the rest -> exactly one record per scene", n1 == 10 and len({rl.key_of(r) for r in parsed}) == len(SCENES) == len(parsed), f"{n1} {len(parsed)} {len(SCENES)}")
            check("resume: the torn line stays ONE bad line (terminated, not fused to the next record)", torn == 1, f"torn {torn}")
            check("resume: the resumed sets equal an uninterrupted run", sorted(json.dumps(r) for r in parsed) == sorted(json.dumps(r) for r in ref))
            # a file of another checkpoint step in the same --out is refused
            write_ckpt(ckB, 2, step=999)
            try:
                run_dump(rl, d, ckpt=ckB, out="res")
                check("resume: a file holding another ckpt_step is REFUSED", False)
            except ValueError:
                check("resume: a file holding another ckpt_step is REFUSED", True)
            write_ckpt(ckB, 2)
        guarded("resume", t_resume)

        # ---- refusals
        def t_refusals():
            rc, so, _ = run_dump(rl, d, ckpt=ckA, args=("--ckpt-step", "10074"), out="r1")
            check("--ckpt-step that disagrees with the file is REFUSED (exit 4)", rc == 4, f"rc {rc}")
            rc, so, _ = run_dump(rl, d, ckpt=ckA, args=("--expect-ckpt-sha256", "0" * 64), out="r2")
            check("--expect-ckpt-sha256 mismatch is REFUSED (exit 4)", rc == 4, f"rc {rc}")
            rc, so, od = run_dump(rl, d, ckpt=ckA, args=("--expect-ckpt-sha256", sha(ckA)), out="r3")
            check("--expect-ckpt-sha256 match is accepted", rc == 0)
            rc, so, od = run_dump(rl, d, ckpt=ckA, scenes=SCENES + [("ghost", "L00")], out="r4")
            check("a scene with no bank row -> exit 3, listed in unavailable_scenes_*.tsv, the rest still dumped",
                  rc == 3 and os.path.exists(os.path.join(od, "unavailable_scenes_fan_s0of1.tsv")) and len(sets_of(os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl"))) == len(SCENES), f"rc {rc}")
            rc, so, od = run_dump(rl, d, ckpt=ckA, scenes=SCENES + [("ghost", "L00")], args=("--allow-missing",), out="r5")
            check("--allow-missing -> exit 0 (still listed)", rc == 0 and os.path.exists(os.path.join(od, "unavailable_scenes_fan_s0of1.tsv")))
            rc, so, od = run_dump(rl, d, ckpt=ckA, scenes=[("tok0_0", "L05")], out="r6")
            check("a key list that belongs to another bank (token under a different log) -> exit 4", rc == 4, f"rc {rc}")
            rc, so, od = run_dump(rl, d, ckpt=ckA, scenes=SCENES + [("ghost", "L00")], args=("--plan-only",), out="r8")
            check("--plan-only with a scene that has no bank row -> exit 3 (the pod preflight must FAIL, not pass)", rc == 3, f"rc {rc}")
            tsv_a, tsv_b = os.path.join(d, "ka.tsv"), os.path.join(d, "kb.tsv")
            write_tsv(tsv_a, SCENES[:3])
            write_tsv(tsv_b, SCENES[2:5])
            a2 = rl.build_parser().parse_args(["--ckpt", ckA, "--keys", tsv_a, tsv_b, "--targets", "u", "--images", "u", "--calib", "u", "--out", os.path.join(d, "r9"), "--device", "cpu"])
            with contextlib.redirect_stdout(io.StringIO()):
                rc9 = rl.run(a2, build=StubModel, loader=stub_loader, ds_factory=ds_factory)
            check("a token listed in two key files (train and val overlap) -> exit 4", rc9 == 4, f"rc {rc9}")
            rc, so, od = run_dump(rl, d, ckpt=ckA, args=("--plan-only",), out="r7")
            check("--plan-only: no model, no sets, a finished manifest", rc == 0 and not os.path.exists(os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl")) and
                  json.load(open(os.path.join(od, "rl_fan_dump_fan_s0of1.manifest.json")))["plan_only"] is True)
        guarded("refusals", t_refusals)

        # ---- the bit-identity gate (--compare-sets)
        def t_compare():
            rc, _, od = run_dump(rl, d, ckpt=ckA, out="cmp0")
            f0 = os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl")
            ref_dir = os.path.join(d, "refsets")
            os.makedirs(ref_dir, exist_ok=True)
            recs = sets_of(f0)
            with open(os.path.join(ref_dir, "onpolicy_r0_w0.jsonl"), "w", encoding="utf-8", newline="\n") as fh:      # an EXISTING bank: same sets, teacher-labelled (extra fields)
                for r in recs[:20]:
                    fh.write(json.dumps(dict(r, label_version=3, targets=[{"x": 1}] * M_, lean=False)) + "\n")
                stale = dict(recs[0], ckpt_step=3668, prev={"ckpt_step": 10075})    # an older checkpoint's set: carries the prefilter marker text, must still be ignored
                stale["yaw"] = [[7.0] * T_] * M_
                fh.write(json.dumps(stale) + "\n")
            rc, so, od1 = run_dump(rl, d, ckpt=ckA, args=("--compare-sets", ref_dir), out="cmp1")
            m1 = json.load(open(os.path.join(od1, "rl_fan_dump_fan_s0of1.manifest.json")))["compare_sets"]
            check("compare-sets: identical sets -> exit 0, 20 matched, 20 bit-identical", rc == 0 and m1["reference_keys_matched"] == 20 and m1["bit_identical"] == 20 and m1["not_identical"] == 0, str(m1))
            # perturb ONE yaw value by 1e-5 in the reference: the gate must trip
            bad = dict(recs[3])
            bad["yaw"] = [list(y) for y in bad["yaw"]]
            bad["yaw"][1][2] = round(bad["yaw"][1][2] + 1e-5, 6)
            with open(os.path.join(ref_dir, "onpolicy_r0_w0.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
                for r in recs[:20]:
                    fh.write(json.dumps(bad if r is recs[3] else r) + "\n")
            rc, so, od2 = run_dump(rl, d, ckpt=ckA, args=("--compare-sets", ref_dir), out="cmp2")
            m2 = json.load(open(os.path.join(od2, "rl_fan_dump_fan_s0of1.manifest.json")))["compare_sets"]
            check("compare-sets: ONE yaw value off by 1e-5 -> exit 5, 1 not identical, 19 identical", rc == 5 and m2["not_identical"] == 1 and m2["bit_identical"] == 19, f"rc {rc} {m2}")
            check("compare-sets: reports the size of the disagreement (>= 1e-5)", abs(m2["max_abs_delta_over_unequal"] - 1e-5) < 2e-6, str(m2["max_abs_delta_over_unequal"]))
            # no overlap at all -> refused (a gate comparing nothing proves nothing)
            with open(os.path.join(ref_dir, "onpolicy_r0_w0.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(dict(recs[0], ckpt_step=3668)) + "\n")
            rc, so, od3 = run_dump(rl, d, ckpt=ckA, args=("--compare-sets", ref_dir), out="cmp3")
            check("compare-sets: matching NOTHING is refused (exit 4), not passed", rc == 4, f"rc {rc}")
            # two reference lines of one key that disagree with each other
            with open(os.path.join(ref_dir, "onpolicy_r0_w0.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(recs[0]) + "\n" + json.dumps(bad if bad["token"] == recs[0]["token"] else dict(recs[0], yaw=[[9.0] * T_] * M_)) + "\n")
            rc, so, od4 = run_dump(rl, d, ckpt=ckA, args=("--compare-sets", ref_dir), out="cmp4")
            check("compare-sets: reference lines of one key that disagree with each other trip the gate (exit 5)", rc == 5, f"rc {rc}")
        guarded("compare", t_compare)

        # ---- the REAL consumers read the record
        def t_consumers():
            import train as T
            import onpolicy_relabel_pdm as PDM
            import select_bank_subset as SB
            rc, _, od = run_dump(rl, d, ckpt=ckA, out="cons")
            f = os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl")
            recs = sets_of(f)
            bank = T.OnPolicyBank(od, M_, T_)
            check("OnPolicyBank loads every lean set: 0 incomplete, 0 bad lines (targets=[{}]*M accepted)", bank.n_rows == len(recs) and bank.n_incomplete == 0 and bank.n_bad_lines == 0 and len(bank.by) == len(recs),
                  f"rows {bank.n_rows} incomplete {bank.n_incomplete} bad {bank.n_bad_lines} by {len(bank.by)}")
            k = (recs[0]["log_name"], recs[0]["token"], 0, 0)
            e = bank.by[k]
            check("OnPolicyBank: ckpt_step, [M, T, 3] proposals, label_version 0", e[0] == 10075 and e[1].shape == (M_, T_, 3) and e[3] == (10075, 0), f"{e[1].shape} {e[3]}")
            check("OnPolicyBank: proposals == the record's traj, yaw", abs(float(e[1][1, 2, 0]) - recs[0]["traj"][1][2][0]) < 1e-6 and abs(float(e[1][1, 2, 2]) - recs[0]["yaw"][1][2]) < 1e-6)
            # PDM side file for 2 of the sets -> pdm_only serves exactly those 2, with the labels, and not the placeholder ones
            pd = os.path.join(d, "pdmside")
            os.makedirs(pd, exist_ok=True)
            with open(os.path.join(pd, "pdm_onpolicy_rlfan_fan_s0of1.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
                for r in recs[:2]:
                    fh.write(json.dumps({"kind": "pdm_targets", "key": [r["log_name"], r["token"], 0, 0], "ckpt_step": 10075, "label_version": 0,
                                         **{c: [0.25] * M_ for c in ("nc", "dac", "ep", "ttc", "c", "ddc")}, "pdms": [0.1] * M_}) + "\n")
            b2 = T.OnPolicyBank(od, M_, T_, pdm_labels=pd, pdm_only=True)
            check("OnPolicyBank(pdm_only): serves ONLY the PDM-labelled sets (the placeholder targets can never train)", len(b2.by) == 2 and b2.n_pdm == 2 and b2.n_dropped_non_pdm == len(recs) - 2, f"by {len(b2.by)} pdm {b2.n_pdm} dropped {b2.n_dropped_non_pdm}")
            check("OnPolicyBank(pdm_only): the six components ARE the PDM labels", float(b2.by[k][2][0, 0]) == 0.25 and float(b2.by[k][2][3, 5]) == 0.25)
            b3 = T.OnPolicyBank(od, M_, T_)
            check("without pdm_only the placeholder is a PERFECT all-ones target (why --require-pdm must be in the argv)", bool((b3.by[k][2] == 1.0).all()))
            best = PDM.served_keys([f])
            check("onpolicy_relabel_pdm.served_keys reads every lean set (head fields), rank (ckpt_step, label_version) = (10075, 0)",
                  len(best) == len(recs) and best[(recs[0]["log_name"], recs[0]["token"], 0, 0)] == (10075, 0), f"{len(best)}")
            line = open(f, encoding="utf-8").readline()
            hf = SB.head_fields(line[:2000] + line[-400:])
            check("select_bank_subset.head_fields finds log_name / step / ckpt_step / token / rank on the lean line", {"log_name", "step", "ckpt_step"} <= set(hf) and hf["ckpt_step"] == 10075, str(hf))
            # the labellers' own inline reader: kind, key fields, traj [M][T][2], yaw [M][T]
            r0 = recs[0]
            check("labeller inputs: kind onpolicy_set; traj [M][T][2]; yaw [M][T]; ckpt_step; label_version",
                  r0["kind"] == "onpolicy_set" and all(len(p[0]) == 2 for p in r0["traj"]) and len(r0["yaw"]) == M_ and "label_version" in r0 and "ckpt_step" in r0)
        guarded("consumers", t_consumers)
    return out


# ------------------------------------------------------------------------------------------------------------------ mutations
MUTATIONS = [
    ("wrong_checkpoint_loaded", "ckpt", 'model, meta = load_model(a.ckpt, a.backbone, a.device, build=build, loader=loader)',
     'model, meta = load_model(os.path.join(os.path.dirname(a.ckpt), "ckpt_last.pt"), a.backbone, a.device, build=build, loader=loader)'),
    ("manifest_hashes_other_file", "ckpt", "ck_sha = sha256_bytes(a.ckpt)", 'ck_sha = sha256_bytes(os.path.join(os.path.dirname(a.ckpt), "ckpt_last.pt"))'),
    ("shard_overlap", "shard", "if shard_of(log, sn) != si:", "if shard_of(log, sn) > si:"),
    ("shard_not_exhaustive", "shard", "return zlib.crc32(log.encode(\"utf-8\")) % n", "return zlib.crc32(log.encode(\"utf-8\")) % (n + 1)"),
    ("shard_splits_a_log", "shard", "return zlib.crc32(log.encode(\"utf-8\")) % n", "return zlib.crc32(log.encode(\"utf-8\")[:-1]) % n"),
    ("dropped_proposal_rank", "fan", "for prop in props]\n    yaw", "for prop in props[:-1]]\n    yaw"),
    ("proposal_order_reversed", "fan", "for prop in props]\n    yaw", "for prop in props[::-1]]\n    yaw"),
    ("fan_size_guard_removed", "fan", "if len(props) != n_prop or any(len(p) != horizon for p in props):", "if False:"),
    ("yaw_rounded_to_3dp", "fan", "round(_r5(p[2]), 6)", "round(_r5(p[2]), 3)"),
    ("targets_placeholder_missing", "format", '"targets": [{} for _ in range(n_prop)]', '"targets": []'),
    ("label_version_3", "format", "LABEL_VERSION = 0", "LABEL_VERSION = 3"),
    ("ckpt_step_zero", "format", '"ckpt_step": int(ckpt_step)', '"ckpt_step": 0'),
    ("manifest_sha_not_checked", "keys", "if sha != want[name]:", "if False:"),
    ("manifest_verify_not_checked", "keys", 'if m.get("verify") != "clean":', "if False:"),
    ("resume_rewrites_done", "resume", "int(ds.rows[i].get(\"rank\", 0))) not in done]", "int(ds.rows[i].get(\"rank\", 0))) not in set()]"),
    ("torn_line_not_terminated", "resume", "            if f.read(1) != b\"\\n\":", "            if False:"),
    ("mixed_ckpt_step_resume_allowed", "resume", 'if int(r["ckpt_step"]) != int(ckpt_step):\n                raise ValueError(f"{path} already', 'if False:\n                raise ValueError(f"{path} already'),
    ("ckpt_step_disagreement_ignored", "ckpt", "if a.ckpt_step is not None and a.ckpt_step != step:", "if False:"),
    ("expected_sha_ignored", "ckpt", "if a.expect_ckpt_sha256 and ck_sha != a.expect_ckpt_sha256.lower():", "if False:"),
    ("compare_nothing_passes", "gate", 'if cmp_["reference_keys_matched"] == 0:', "if False:"),
    ("compare_ignores_yaw", "gate", "if new[k] == v:", "if new[k][0] == v[0]:"),
    ("compare_other_ckpt_step", "gate", 'or int(r["ckpt_step"]) != int(ckpt_step) or r["token"] not in tokens:', 'or r["token"] not in tokens:'),
    ("log_mismatch_not_refused", "keys", "    if mism:\n        print(f\"REFUSED:", "    if False:\n        print(f\"REFUSED:"),
    ("missing_scenes_exit_0", "keys", "    return EXIT_MISSING if (missing and not a.allow_missing) else EXIT_OK\n\n\ndef _write_json", "    return EXIT_OK\n\n\ndef _write_json"),
    ("plan_only_missing_exit_0", "keys", "        return EXIT_MISSING if (missing and not a.allow_missing) else EXIT_OK\n\n    # ---- the model", "        return EXIT_OK\n\n    # ---- the model"),
    ("dup_token_across_files_allowed", "keys", "if len({t for t, _ in scenes}) != len(scenes):", "if False:"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    base = suite(load(src, "rl_fan_dump_under_test"))
    real = [(n, d) for n, ok, d in base if not ok]
    for n, ok, d in base:
        if not ok:
            print(f"  FAIL {n}  {d}")
    assert not real, f"the unmutated module FAILS its own checks: {[n for n, _ in real]}"
    print(f"unmutated: {len(base)} checks green")
    print(f"{'mutation':36s} {'class':8s} caught-by (failing checks)")
    for name, cls, old, new in MUTATIONS:
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"rl_fan_dump_mut_{name}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        failing = [n for n, ok, _ in suite(mod) if not ok]
        print(f"{name:36s} {cls:8s} {len(failing)}  {failing[0][:70] if failing else ''}")
        assert failing, f"mutation {name} ({cls}) stayed GREEN: the suite cannot see it"
    need = {"ckpt", "shard", "fan", "format", "keys", "resume", "gate"}
    assert {c for _, c, _, _ in MUTATIONS} >= need, "a required mutation class is missing"
    n = 1 + len(MUTATIONS) + 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {len(base)} checks + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
