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

        # ---- the ABSOLUTE-TOLERANCE gate (--compare-atol): bit-identity holds only at matched batching on one GPU (eval/rl_fan_identity.py: 0.25 floor)
        def t_atol():
            import math
            # -- the pure helpers, hand-derived
            check("quantile_nearest_rank [1..10]: p50 = 5, p90 = 9, p99 = 10, p100 = 10 (nearest rank = ceil(q n)-th smallest)",
                  [rl.quantile_nearest_rank(list(range(1, 11)), q) for q in (0.5, 0.9, 0.99, 1.0)] == [5, 9, 10, 10])
            check("quantile_nearest_rank: empty -> 0.0", rl.quantile_nearest_rank([], 0.5) == 0.0)
            t0 = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
            y0 = [[0.1, 0.2], [0.3, 0.4]]
            same = rl.set_delta_strict((t0, y0), (t0, y0))
            check("set_delta_strict: identical -> (0, 0)", same == (0.0, 0.0), str(same))
            t1 = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.75]]]            # one y value 0.75 away
            y1 = [[0.1, 0.2], [0.3, 0.4 - 0.125]]                                 # one yaw value 0.125 away
            d_xy, d_yaw = rl.set_delta_strict((t1, y1), (t0, y0))
            check("set_delta_strict: x/y and yaw deltas are kept apart (hand-derived 0.75 and 0.125)", abs(d_xy - 0.75) < 1e-12 and abs(d_yaw - 0.125) < 1e-12, f"{d_xy} {d_yaw}")
            nan_t = [[[1.0, 2.0], [3.0, float("nan")]], [[5.0, 6.0], [7.0, 8.0]]]
            nan_y = [[0.1, 0.2], [0.3, float("nan")]]
            check("set_delta_strict: a NaN in x/y is inf (max(0.0, nan) would read 0.0 = 'within')", rl.set_delta_strict((nan_t, y0), (t0, y0))[0] == math.inf)
            check("set_delta_strict: a NaN in yaw is inf", rl.set_delta_strict((t0, nan_y), (t0, y0))[1] == math.inf)
            check("set_delta_strict: an inf on one side is inf", rl.set_delta_strict(([[[1.0, 2.0], [3.0, float("inf")]], t0[1]], y0), (t0, y0))[0] == math.inf)
            check("set_delta_strict: a dropped proposal is inf (zip would truncate it to 0.0)", rl.set_delta_strict((t0[:1], y0), (t0, y0))[0] == math.inf)
            check("set_delta_strict: a dropped yaw proposal is inf", rl.set_delta_strict((t0, y0[:1]), (t0, y0))[1] == math.inf)
            check("set_delta_strict: a dropped horizon step is inf", rl.set_delta_strict(([p[:1] for p in t0], y0), (t0, y0))[0] == math.inf)
            check("set_delta_strict: a dropped yaw step is inf", rl.set_delta_strict((t0, [y[:1] for y in y0]), (t0, y0))[1] == math.inf)

            # -- end to end through the parser and run(): a dump, then references built from it with KNOWN perturbations
            rc, _, od = run_dump(rl, d, ckpt=ckA, out="atol0")
            recs = sets_of(os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl"))
            ref_dir = os.path.join(d, "atol_refs")
            os.makedirs(ref_dir, exist_ok=True)
            ref_file = os.path.join(ref_dir, "onpolicy_r0_w0.jsonl")

            def perturbed(i, kind, amount):
                r = json.loads(json.dumps(recs[i]))
                if kind == "x":
                    r["traj"][1][2][0] = r["traj"][1][2][0] + amount
                elif kind == "y":
                    r["traj"][2][1][1] = r["traj"][2][1][1] + amount
                elif kind == "yaw":
                    r["yaw"][0][2] = r["yaw"][0][2] + amount
                return r

            def write_ref(rows):
                with open(ref_file, "w", encoding="utf-8", newline="\n") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")

            def gate(args, out):
                rc_, so_, od_ = run_dump(rl, d, ckpt=ckA, args=("--compare-sets", ref_dir, *args), out=out)
                mf = os.path.join(od_, "rl_fan_dump_fan_s0of1.manifest.json")
                return rc_, so_, (json.load(open(mf))["compare_sets"] if os.path.exists(mf) else None)

            A, B, C = 3, 7, 11                                                       # indices of the perturbed sets among the first 20 references
            rowsA = perturbed(A, "x", 0.25)
            rowsB = perturbed(B, "y", 3.0)
            rowsC = perturbed(C, "yaw", 0.001)
            dA = abs(recs[A]["traj"][1][2][0] - rowsA["traj"][1][2][0])               # the delta AS THE MODULE WILL SEE IT (binary rounding of new - ref), recomputed here
            write_ref([recs[i] if i not in (A, B, C) else {A: rowsA, B: rowsB, C: rowsC}[i] for i in range(20)])
            rc1, so1, m1 = gate(("--compare-atol", "0.5"), "atol1")
            check("atol 0.5: sets off by 0.25 (x) and 0.001 (yaw) are WITHIN, the one off by 3.0 (y) is beyond -> 19 within, 1 beyond, exit 5",
                  rc1 == 5 and m1["within_atol"] == 19 and m1["beyond_atol"] == 1 and m1["reference_keys_matched"] == 20, f"rc {rc1} {m1}")
            check("atol: the three perturbed sets are NOT bit-identical (17 are) -- the exact report keeps its meaning", m1["not_identical"] == 3 and m1["bit_identical"] == 17)
            check("atol: max |delta| is the 3.0 y offset; the x/y and yaw maxima are kept apart (3.0 and 0.001)",
                  abs(m1["max_abs_delta"] - 3.0) < 1e-9 and abs(m1["max_abs_delta_xy"] - 3.0) < 1e-9 and abs(m1["max_abs_delta_yaw"] - 0.001) < 1e-9, str(m1))
            q = m1["delta_quantiles"]
            check("atol: per-set quantiles over ALL 20 matched sets (17 zeros, 0.001, 0.25, 3.0): p50 0, p90 0.001, p99 3.0, max 3.0 (hand-derived, nearest rank)",
                  q["p50"] == 0.0 and abs(q["p90"] - 0.001) < 1e-9 and abs(q["p99"] - 3.0) < 1e-9 and abs(q["max"] - 3.0) < 1e-9, str(q))
            check("atol: the offender is named (first_beyond_atol) and the FAIL marker is printed, not PASS",
                  m1["first_beyond_atol"] and m1["first_beyond_atol"][0][0][1] == recs[B]["token"] and "ZZRLFAN_ATOL_FAIL 20 1 " in so1 and "ZZRLFAN_ATOL_PASS" not in so1, so1[-300:])
            rc2, so2, m2 = gate(("--compare-atol", "0.2"), "atol2")
            check("atol 0.2: monotone -- now the 0.25 offset is beyond too (2 beyond, exit 5)", rc2 == 5 and m2["beyond_atol"] == 2, f"rc {rc2} {m2['beyond_atol']}")
            rc3, so3, m3 = gate(("--compare-atol", "5.0"), "atol3")
            check("atol 5.0: all 20 within -> exit 0, PASS marker with the matched count",
                  rc3 == 0 and m3["beyond_atol"] == 0 and m3["within_atol"] == 20 and "ZZRLFAN_ATOL_PASS 20 " in so3 and "ZZRLFAN_ATOL_FAIL" not in so3, f"rc {rc3}")
            # -- the boundary: a set whose delta EQUALS atol is within (<=); one ulp below atol it is beyond
            write_ref([recs[i] if i != A else rowsA for i in range(20)])
            rc4, so4, m4 = gate(("--compare-atol", repr(dA)), "atol4")
            check("atol == the set's delta exactly -> WITHIN (the test is <=, not <)", rc4 == 0 and m4["beyond_atol"] == 0 and m4["within_atol"] == 20, f"rc {rc4} dA {dA!r} {m4['max_abs_delta']!r}")
            rc5, so5, m5 = gate(("--compare-atol", repr(math.nextafter(dA, 0.0))), "atol5")
            check("atol one ulp below the set's delta -> beyond (exit 5)", rc5 == 5 and m5["beyond_atol"] == 1, f"rc {rc5}")
            # -- the default is still bit-identity, and the SAME tiny disagreement flips only with the flag
            tiny = perturbed(5, "yaw", 1e-5)
            write_ref([recs[i] if i != 5 else tiny for i in range(20)])
            rc6, so6, m6 = gate((), "atol6")
            check("no --compare-atol: a 1e-5 yaw offset is still exit 5 and the report carries NONE of the atol keys (the default path is unchanged)",
                  rc6 == 5 and m6["not_identical"] == 1 and not ({"atol", "within_atol", "beyond_atol", "delta_quantiles", "max_abs_delta"} & set(m6)) and "ZZRLFAN_ATOL" not in so6,
                  f"rc {rc6} {sorted(m6)}")
            rc7, so7, m7 = gate(("--compare-atol", "0.001"), "atol7")
            check("--compare-atol 0.001: the same 1e-5 offset is WITHIN -> exit 0 (the flag, and only the flag, changes the verdict)",
                  rc7 == 0 and m7["within_atol"] == 20 and m7["not_identical"] == 1, f"rc {rc7}")
            rc8, so8, m8 = gate(("--compare-atol", "0.0"), "atol8")
            check("--compare-atol 0.0 behaves as bit-identity (the 1e-5 offset is beyond)", rc8 == 5 and m8["beyond_atol"] == 1, f"rc {rc8}")
            # -- a fan the reference does not have the shape of, and a NaN, are never within tolerance
            cut = json.loads(json.dumps(recs[4]))
            cut["traj"] = cut["traj"][:-1]
            cut["yaw"] = cut["yaw"][:-1]
            write_ref([recs[i] if i != 4 else cut for i in range(20)])
            rc9, so9, m9 = gate(("--compare-atol", "1e9"), "atol9")
            check("a reference fan with a dropped proposal is beyond even atol 1e9 (shape mismatch = inf)", rc9 == 5 and m9["beyond_atol"] == 1, f"rc {rc9} {m9 and m9.get('beyond_atol')}")
            nanr = json.loads(json.dumps(recs[6]))
            nanr["traj"][0][0][0] = float("nan")
            write_ref([recs[i] if i != 6 else nanr for i in range(20)])
            rc10, so10, m10 = gate(("--compare-atol", "1e9"), "atol10")
            check("a reference with a NaN x value is beyond even atol 1e9", rc10 == 5 and m10["beyond_atol"] == 1, f"rc {rc10}")
            nany = json.loads(json.dumps(recs[6]))
            nany["yaw"][0][0] = float("nan")
            write_ref([recs[i] if i != 6 else nany for i in range(20)])
            rc10b, _, m10b = gate(("--compare-atol", "1e9"), "atol10b")
            check("a reference with a NaN yaw value is beyond even atol 1e9", rc10b == 5 and m10b["beyond_atol"] == 1, f"rc {rc10b}")
            # -- the reference-disagreement condition survives atol mode (two reference lines of one key that differ)
            write_ref([recs[0], dict(recs[0], yaw=[[9.0] * T_] * M_)])
            rc11, _, _ = gate(("--compare-atol", "1e9"), "atol11")
            check("atol mode: reference lines of one key that disagree with each other still trip the gate (exit 5)", rc11 == 5, f"rc {rc11}")
            # -- refusals: a tolerance that would be silently ignored, a nonsense tolerance, too few matches
            write_ref(recs[:20])
            rc12, so12, _ = run_dump(rl, d, ckpt=ckA, args=("--compare-atol", "0.5"), out="atol12")
            check("--compare-atol without --compare-sets is REFUSED (exit 4), not silently ignored", rc12 == 4 and "need --compare-sets" in so12, f"rc {rc12}")
            for nm, val in (("negative", "-0.1"), ("NaN", "nan"), ("infinite", "inf")):
                rcn, son, _ = gate(("--compare-atol", val), "atol_bad_" + nm)
                check(f"--compare-atol {val} ({nm}) is REFUSED (exit 4)", rcn == 4, f"rc {rcn}")
            rcm, som, _ = gate(("--compare-atol", "5", "--compare-min-matched", "21"), "atol13")
            check("--compare-min-matched 21 with 20 matched keys is REFUSED (exit 4)", rcm == 4 and "fewer than --compare-min-matched 21" in som, f"rc {rcm}")
            rcm2, _, _ = gate(("--compare-atol", "5", "--compare-min-matched", "20"), "atol14")
            check("--compare-min-matched 20 with 20 matched keys passes (exit 0)", rcm2 == 0, f"rc {rcm2}")
            rcm3, som3, _ = gate(("--compare-min-matched", "0"), "atol15")
            check("--compare-min-matched 0 is REFUSED (a floor of zero is no floor)", rcm3 == 4, f"rc {rcm3}")
            ap = rl.build_parser().parse_args(["--ckpt", "x", "--keys", "y", "--targets", "t", "--images", "i", "--calib", "c", "--out", "o"])
            check("the parser default is None for both flags (no tolerance unless named in the argv)", ap.compare_atol is None and ap.compare_min_matched is None)
        guarded("atol", t_atol)

        # ---- the MEAN bound (--compare-mean-atol, 2026-10-12): round 2 (raw/rl_fan/atol_calibration/round2, PASS, CHOSEN_MEAN_ATOL 0.025) pre-registered m = the per-set mean |delta|
        #      over the x / y values (yaw excluded); the gate must compute exactly that number, judge it strictly (m > M fails), and keep the catastrophe bound beside it.
        def t_mean():
            import math
            import random
            mf = rl.set_mean_abs_xy_strict
            # -- the statistic, hand-derived (2 proposals x 2 steps x (x, y) = 8 values)
            t0 = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
            y0 = [[0.1, 0.2], [0.3, 0.4]]
            check("set_mean_abs_xy_strict: identical -> 0.0", mf((t0, y0), (t0, y0)) == 0.0)
            t1 = [[[1.5, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 7.75]]]            # one x value 0.5 away, one y value 0.25 away
            check("set_mean_abs_xy_strict: the MEAN, not the max -- (0.5 + 0.25) / 8 = 0.09375 exactly (the max delta is 0.5)", mf((t1, y0), (t0, y0)) == 0.09375, str(mf((t1, y0), (t0, y0))))
            check("set_mean_abs_xy_strict: symmetric in its arguments", mf((t0, y0), (t1, y0)) == 0.09375)
            tx = [[[1.75, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]            # one x value 0.75 away
            ty = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.75]]]            # one y value 0.75 away
            check("set_mean_abs_xy_strict: x AND y both count (0.75 / 8 = 0.09375 for an x-only and for a y-only difference)", mf((tx, y0), (t0, y0)) == 0.09375 and mf((ty, y0), (t0, y0)) == 0.09375)
            y1 = [[0.1, 0.2], [0.3, 0.4 - 0.125]]                                 # one yaw value 0.125 away
            check("set_mean_abs_xy_strict: YAW IS EXCLUDED -- a yaw-only difference is m = 0.0", mf((t0, y1), (t0, y0)) == 0.0, str(mf((t0, y1), (t0, y0))))
            check("set_mean_abs_xy_strict: ... and a yaw difference adds nothing to an x / y one", mf((t1, y1), (t0, y0)) == 0.09375)
            nan_t = [[[1.0, 2.0], [3.0, float("nan")]], t0[1]]
            check("set_mean_abs_xy_strict: a NaN in x / y is inf (a NaN sum makes `m > M` False, i.e. 'within')", mf((nan_t, y0), (t0, y0)) == math.inf and mf((t0, y0), (nan_t, y0)) == math.inf)
            check("set_mean_abs_xy_strict: an inf in x / y is inf", mf(([[[1.0, 2.0], [3.0, float("inf")]], t0[1]], y0), (t0, y0)) == math.inf)
            check("set_mean_abs_xy_strict: a dropped proposal is inf (zip would truncate it to a 'perfect' 0.0)", mf((t0[:1], y0), (t0, y0)) == math.inf)
            check("set_mean_abs_xy_strict: a dropped horizon step is inf", mf(([p[:1] for p in t0], y0), (t0, y0)) == math.inf)
            check("set_mean_abs_xy_strict: a dropped coordinate is inf", mf(([[[1.0], [3.0, 4.0]], t0[1]], y0), (t0, y0)) == math.inf)
            check("set_mean_abs_xy_strict: yaw is out of the statistic altogether (a NaN yaw and a dropped yaw proposal read m = 0.0: the max bound is the one that reads yaw)",
                  mf((t0, [[0.1, float("nan")], [0.3, 0.4]]), (t0, y0)) == 0.0 and mf((t0, y0[:1]), (t0, y0)) == 0.0)
            check("set_mean_abs_xy_strict: nothing to average -> 0.0, not a ZeroDivisionError", mf(([], []), ([], [])) == 0.0)

            # -- ONE dump; the references below are built from it with KNOWN perturbations of x / y / yaw values
            rc, _, od = run_dump(rl, d, ckpt=ckA, out="mean0")
            new_path = os.path.join(od, "onpolicy_rlfan_fan_s0of1.jsonl")
            recs = sets_of(new_path)
            ref_dir = os.path.join(d, "mean_refs")
            os.makedirs(ref_dir, exist_ok=True)
            ref_file = os.path.join(ref_dir, "onpolicy_r0_w0.jsonl")
            NV = M_ * T_ * 2                                                       # x / y values per set: 24

            def shifted(i, xy=(), yaw=()):
                r = json.loads(json.dumps(recs[i]))
                for (mi, ti, ci), amt in xy:
                    r["traj"][mi][ti][ci] = r["traj"][mi][ti][ci] + amt
                for (mi, ti), amt in yaw:
                    r["yaw"][mi][ti] = r["yaw"][mi][ti] + amt
                return r

            def write_ref(rows):
                with open(ref_file, "w", encoding="utf-8", newline="\n") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")

            def gate(args, out):
                rc_, so_, od_ = run_dump(rl, d, ckpt=ckA, args=("--compare-sets", ref_dir, *args), out=out)
                mf_ = os.path.join(od_, "rl_fan_dump_fan_s0of1.manifest.json")
                return rc_, so_, (json.load(open(mf_))["compare_sets"] if os.path.exists(mf_) else None)

            def marker(so, name):
                return [ln for ln in so.splitlines() if ln.startswith(name + " ")]

            MEAN_KEYS = {"mean_atol", "within_mean_atol", "beyond_mean_atol", "max_mean_abs_delta", "mean_abs_delta_quantiles", "first_beyond_mean_atol"}

            # -- the SAME statistic as the analysis code that chose M: recompute round 2's per-arm numbers (sets, sets above M, worst m, quantiles) from a small fixture with
            #    run_calibration's own code (xy_moments / arm_deltas element 3), and demand the gate's numbers EQUAL them -- exactly, not approximately (same summation order)
            cal = os.path.normpath(os.path.join(HERE, os.pardir, "raw", "rl_fan", "atol_calibration"))
            if cal not in sys.path:
                sys.path.insert(0, cal)
            import run_calibration as RC
            rng = random.Random(20261012)
            fix_rows = []
            for r in recs[:16]:
                q = json.loads(json.dumps(r))
                for _ in range(rng.randrange(0, 6)):                               # 0 to 5 values moved by up to 0.6, rounded to 5 decimals like a dump
                    mi, ti, ci = rng.randrange(M_), rng.randrange(T_), rng.randrange(2)
                    q["traj"][mi][ti][ci] = round(q["traj"][mi][ti][ci] + rng.uniform(-0.6, 0.6), 5)
                fix_rows.append(q)
            write_ref(fix_rows)
            STEP = 10075
            ref_sets = RC.load_sets(ref_file, STEP)
            new_sets = RC.load_sets(new_path, STEP)
            dd = RC.arm_deltas(new_path, STEP, ref_sets)
            ms_an = sorted(x[3] for x in dd.values())
            above_an = sum(1 for v in ms_an if v > 0.025)                          # run_calibration2.count_above
            check("fixture sanity: 16 matched sets, some above M = 0.025 and some not, many distinct m values (the comparison below is not vacuous)",
                  len(dd) == 16 and 0 < above_an < 16 and len(set(ms_an)) > 8, f"{len(dd)} {above_an} {len(set(ms_an))}")
            check("per set: the gate's m EQUALS run_calibration.arm_deltas' element 3, bit for bit, on all 16 sets",
                  all(rl.set_mean_abs_xy_strict(new_sets[k], ref_sets[k]) == dd[k][3] for k in dd), "")
            rep = rl.compare_sets(new_path, [ref_file], STEP, mean_atol=0.025)
            qa = RC.RF.quantile_nearest_rank
            check("per arm: worst m, p50 / p90 / p99, sets above M and sets within M all EQUAL round 2's analysis numbers on the fixture",
                  rep["reference_keys_matched"] == 16 and rep["max_mean_abs_delta"] == ms_an[-1] and rep["beyond_mean_atol"] == above_an and rep["within_mean_atol"] == 16 - above_an
                  and [rep["mean_abs_delta_quantiles"][k] for k in ("p50", "p90", "p99", "max")] == [qa(ms_an, 0.5), qa(ms_an, 0.9), qa(ms_an, 0.99), ms_an[-1]], str(rep))
            check("compare_sets without mean_atol adds NONE of the mean keys", not (MEAN_KEYS & set(rl.compare_sets(new_path, [ref_file], STEP, atol=0.5))))

            # -- end to end through the parser and run().  Five known sets among the first 20 references (24 x / y values each):
            A, F, G, D, Y = 3, 5, 8, 12, 15
            rowsA = shifted(A, xy=[((0, 0, 0), 0.25)])                             # ONE value off by 0.25 (the bf16 floor):               d = 0.25, m = 0.25 / 24 = 0.0104
            rowsF = shifted(F, xy=[((0, 0, 0), 0.25), ((1, 1, 1), 0.25)])          # two values off by 0.25:                              d = 0.25, m = 0.5 / 24  = 0.0208
            rowsG = shifted(G, xy=[((0, 0, 0), 0.3), ((1, 1, 1), 0.3), ((3, 2, 0), 0.3)])   # three values off by 0.3:                   d = 0.3,  m = 0.9 / 24  = 0.0375
            rowsD = shifted(D, xy=[((mi, ti, ci), 0.05) for mi in range(M_) for ti in range(T_) for ci in range(2)])   # ALL values by 0.05: d = 0.05, m = 0.05
            rowsY = shifted(Y, yaw=[((0, 0), 2.0)])                                # ONE yaw value off by 2.0:                            d = 2.0,  m = 0 (yaw excluded)

            def refs_with(use):
                rows = {A: rowsA, F: rowsF, G: rowsG, D: rowsD, Y: rowsY}
                write_ref([rows[i] if i in use else recs[i] for i in range(20)])

            refs_with((A, F))
            rc1, so1, m1 = gate(("--compare-atol", "0.5", "--compare-mean-atol", "0.025"), "mean1")
            check("0.5 + 0.025, only the floor-sized offsets (m 0.0104 and 0.0208): both bounds hold -> exit 0, 20 within each, nothing beyond",
                  rc1 == 0 and m1["within_mean_atol"] == 20 and m1["beyond_mean_atol"] == 0 and m1["within_atol"] == 20 and m1["beyond_atol"] == 0 and m1["reference_keys_matched"] == 20, f"rc {rc1} {m1}")
            check("mean: the report echoes the bound and the worst m (0.5 / 24 = 0.0208333)", m1["mean_atol"] == 0.025 and abs(m1["max_mean_abs_delta"] - 0.5 / NV) < 1e-9, str(m1.get("max_mean_abs_delta")))
            pm = marker(so1, "ZZRLFAN_MEAN_ATOL_PASS")
            check("mean: PASS marker `ZZRLFAN_MEAN_ATOL_PASS <matched> <max_m> <bound>` (one line, the bound last), next to the unchanged ATOL one; no FAIL marker",
                  len(pm) == 1 and pm[0].startswith("ZZRLFAN_MEAN_ATOL_PASS 20 0.02083") and pm[0].endswith(" 0.025") and len(marker(so1, "ZZRLFAN_ATOL_PASS")) == 1
                  and marker(so1, "ZZRLFAN_ATOL_PASS")[0].endswith(" 0.5") and "FAIL" not in so1, so1[-400:])
            check("mean: the printed compare-sets line carries the mean keys too", "within_mean_atol" in so1 and "mean_abs_delta_quantiles" in so1)

            refs_with((A, F, G, D))
            rc2, so2, m2 = gate(("--compare-atol", "0.5", "--compare-mean-atol", "0.025"), "mean2")
            check("m 0.0375 (three values 0.3 off) and m 0.05 (every value 0.05 off) are beyond 0.025 while their max delta (0.3, 0.05) is far inside 0.5 -> exit 5, 2 beyond the MEAN bound, 0 beyond the max bound",
                  rc2 == 5 and m2["beyond_mean_atol"] == 2 and m2["within_mean_atol"] == 18 and m2["beyond_atol"] == 0 and m2["within_atol"] == 20, f"rc {rc2} {m2}")
            check("mean: FAIL marker `ZZRLFAN_MEAN_ATOL_FAIL <matched> <beyond> <max_m> <bound>` while the max bound still prints PASS (they are judged separately)",
                  len(marker(so2, "ZZRLFAN_MEAN_ATOL_FAIL")) == 1 and marker(so2, "ZZRLFAN_MEAN_ATOL_FAIL")[0].startswith("ZZRLFAN_MEAN_ATOL_FAIL 20 2 0.05")
                  and marker(so2, "ZZRLFAN_MEAN_ATOL_FAIL")[0].endswith(" 0.025") and not marker(so2, "ZZRLFAN_MEAN_ATOL_PASS") and len(marker(so2, "ZZRLFAN_ATOL_PASS")) == 1 and not marker(so2, "ZZRLFAN_ATOL_FAIL"), so2[-500:])
            fb = m2["first_beyond_mean_atol"]
            check("mean: the offenders are named, worst first (D at m 0.05, then G at 0.0375)",
                  len(fb) == 2 and fb[0][0][1] == recs[D]["token"] and fb[1][0][1] == recs[G]["token"] and abs(fb[0][1] - 0.05) < 1e-9 and abs(fb[1][1] - 0.0375) < 1e-9, str(fb))
            q = m2["mean_abs_delta_quantiles"]
            check("mean: quantiles over ALL 20 matched sets (16 zeros, 0.0104, 0.0208, 0.0375, 0.05): p50 0, p90 0.0208 (18th smallest), p99 0.05, max 0.05 (hand-derived, nearest rank)",
                  q["p50"] == 0.0 and abs(q["p90"] - 0.5 / NV) < 1e-9 and abs(q["p99"] - 0.05) < 1e-9 and abs(q["max"] - 0.05) < 1e-9 and m2["max_mean_abs_delta"] == q["max"], str(q))
            for bound, want in (("0.01", 4), ("0.04", 1), ("1.0", 0)):
                rcb, _, mb = gate(("--compare-atol", "0.5", "--compare-mean-atol", bound), "mean_b" + bound)
                check(f"mean bound {bound}: monotone -- {want} set(s) beyond, exit {5 if want else 0} (the max bound, untouched, passes all 20)",
                      mb["beyond_mean_atol"] == want and rcb == (5 if want else 0) and mb["beyond_atol"] == 0, f"rc {rcb} {mb['beyond_mean_atol']}")

            # -- the boundary: m == M is within (the test is `m > M`); one ulp below M it is beyond.  Only set A differs, by ONE value: the sum is that value alone, m = d / 24 exactly
            refs_with((A,))
            dA = abs(recs[A]["traj"][0][0][0] - rowsA["traj"][0][0][0])
            mA = dA / NV
            rc4, so4, m4 = gate(("--compare-atol", "0.5", "--compare-mean-atol", repr(mA)), "mean4")
            check("mean == the set's m exactly -> WITHIN (the test is m > M, not >=)", rc4 == 0 and m4["beyond_mean_atol"] == 0 and m4["within_mean_atol"] == 20 and m4["max_mean_abs_delta"] == mA, f"rc {rc4} mA {mA!r} {m4['max_mean_abs_delta']!r}")
            rc5, so5, m5 = gate(("--compare-atol", "0.5", "--compare-mean-atol", repr(math.nextafter(mA, 0.0))), "mean5")
            check("mean one ulp below the set's m -> beyond (exit 5)", rc5 == 5 and m5["beyond_mean_atol"] == 1 and "ZZRLFAN_MEAN_ATOL_FAIL 20 1 " in so5, f"rc {rc5}")
            rc5b, _, m5b = gate(("--compare-atol", "0.5", "--compare-mean-atol", "0.0"), "mean5b")
            check("mean bound 0.0: any x / y difference at all is beyond (exit 5)",
                  rc5b == 5 and m5b["beyond_mean_atol"] == 1, f"rc {rc5b}")

            # -- the two bounds are INDEPENDENT: a yaw fault trips only the max bound; a big x offset trips both
            refs_with((Y,))
            rc6, so6, m6 = gate(("--compare-atol", "0.5", "--compare-mean-atol", "0.025"), "mean6")
            check("a yaw-only fault (2.0): exit 5 from the MAX bound alone -- ATOL_FAIL, MEAN_ATOL_PASS (m = 0)",
                  rc6 == 5 and m6["beyond_atol"] == 1 and m6["beyond_mean_atol"] == 0 and len(marker(so6, "ZZRLFAN_ATOL_FAIL")) == 1 and len(marker(so6, "ZZRLFAN_MEAN_ATOL_PASS")) == 1
                  and not marker(so6, "ZZRLFAN_MEAN_ATOL_FAIL"), f"rc {rc6} {m6['beyond_atol']} {m6['beyond_mean_atol']}")
            rc6b, so6b, m6b = gate(("--compare-atol", "0.5", "--compare-mean-atol", "0.0"), "mean6b")
            check("... and with mean bound 0.0 the yaw-only fault still does not move m (exit 5 is the max bound's)", m6b["beyond_mean_atol"] == 0 and m6b["beyond_atol"] == 1 and rc6b == 5)
            big = shifted(7, xy=[((2, 1, 1), 3.0)])
            write_ref([big if i == 7 else recs[i] for i in range(20)])
            rc7, so7, m7 = gate(("--compare-atol", "0.5", "--compare-mean-atol", "0.025"), "mean7")
            check("one x / y value 3.0 off (d 3.0, m 0.125): beyond BOTH bounds -> exit 5, both FAIL markers",
                  rc7 == 5 and m7["beyond_atol"] == 1 and m7["beyond_mean_atol"] == 1 and len(marker(so7, "ZZRLFAN_ATOL_FAIL")) == 1 and len(marker(so7, "ZZRLFAN_MEAN_ATOL_FAIL")) == 1, f"rc {rc7}")
            rc7b, _, m7b = gate(("--compare-atol", "5.0", "--compare-mean-atol", "0.025"), "mean7b")
            check("max bound loosened to 5.0: the same set is still beyond the MEAN bound (exit 5) -- the mean bound is not subsumed by the max one",
                  rc7b == 5 and m7b["beyond_atol"] == 0 and m7b["beyond_mean_atol"] == 1)

            # -- a fan the reference does not have the shape of, and a NaN, are never within the mean bound either
            cut = json.loads(json.dumps(recs[4]))
            cut["traj"] = cut["traj"][:-1]
            cut["yaw"] = cut["yaw"][:-1]
            write_ref([recs[i] if i != 4 else cut for i in range(20)])
            rc8, _, m8 = gate(("--compare-atol", "1e9", "--compare-mean-atol", "1e9"), "mean8")
            check("a reference fan with a dropped proposal is beyond even a mean bound of 1e9 (m = inf)", rc8 == 5 and m8["beyond_mean_atol"] == 1 and m8["max_mean_abs_delta"] == math.inf, f"rc {rc8} {m8['beyond_mean_atol']}")
            nanr = json.loads(json.dumps(recs[6]))
            nanr["traj"][0][0][0] = float("nan")
            write_ref([recs[i] if i != 6 else nanr for i in range(20)])
            rc9, _, m9 = gate(("--compare-atol", "1e9", "--compare-mean-atol", "1e9"), "mean9")
            check("a reference with a NaN x value is beyond even a mean bound of 1e9", rc9 == 5 and m9["beyond_mean_atol"] == 1, f"rc {rc9}")
            nany = json.loads(json.dumps(recs[6]))
            nany["yaw"][0][0] = float("nan")
            write_ref([recs[i] if i != 6 else nany for i in range(20)])
            rc9b, _, m9b = gate(("--compare-atol", "1e9", "--compare-mean-atol", "1e9"), "mean9b")
            check("a reference with a NaN yaw value is NOT a mean-bound failure (yaw excluded) -- the max bound is the one that trips (exit 5)", rc9b == 5 and m9b["beyond_mean_atol"] == 0 and m9b["beyond_atol"] == 1, f"rc {rc9b}")
            # -- the reference-disagreement condition reaches the mean marker too
            write_ref([recs[0], dict(recs[0], yaw=[[9.0] * T_] * M_)])
            rc10, so10, _ = gate(("--compare-atol", "1e9", "--compare-mean-atol", "1e9"), "mean10")
            check("reference lines of one key that disagree with each other: exit 5 and BOTH FAIL markers", rc10 == 5 and len(marker(so10, "ZZRLFAN_MEAN_ATOL_FAIL")) == 1 and len(marker(so10, "ZZRLFAN_ATOL_FAIL")) == 1, f"rc {rc10}")

            # -- the default path is unchanged: without the flag NO mean key, NO mean marker, the same exit
            refs_with((A, G))
            rc11, so11, m11 = gate(("--compare-atol", "0.5"), "mean11")
            check("no --compare-mean-atol: the report carries NONE of the mean keys, no MEAN marker, and the verdict is the max bound's alone (m 0.0375 is not looked at -> exit 0)",
                  rc11 == 0 and not (MEAN_KEYS & set(m11)) and "ZZRLFAN_MEAN_ATOL" not in so11 and "mean_atol" not in so11 and m11["within_atol"] == 20, f"rc {rc11} {sorted(m11)}")
            rc12, so12, m12 = gate((), "mean12")
            check("no flags at all: still the bit-identity gate (exit 5, 2 not identical), no atol and no mean keys, no markers",
                  rc12 == 5 and m12["not_identical"] == 2 and not (MEAN_KEYS & set(m12)) and "atol" not in m12 and "ZZRLFAN_ATOL" not in so12 and "ZZRLFAN_MEAN" not in so12, f"rc {rc12}")

            # -- refusals: the mean bound without the catastrophe bound, without compare-sets, a nonsense value, too few matches
            write_ref(recs[:20])
            rcr, sor, _ = gate(("--compare-mean-atol", "0.025"), "mean_r1")
            check("--compare-mean-atol WITHOUT --compare-atol is REFUSED (exit 4): the max bound is the one that reads yaw", rcr == 4 and "needs --compare-atol" in sor, f"rc {rcr}")
            rcr2, sor2, _ = run_dump(rl, d, ckpt=ckA, args=("--compare-mean-atol", "0.025"), out="mean_r2")
            check("--compare-mean-atol with no --compare-sets and no --compare-atol is REFUSED (exit 4)", rcr2 == 4, f"rc {rcr2}")
            rcr3, sor3, _ = run_dump(rl, d, ckpt=ckA, args=("--compare-atol", "0.5", "--compare-mean-atol", "0.025"), out="mean_r3")
            check("--compare-atol + --compare-mean-atol without --compare-sets is REFUSED (exit 4)", rcr3 == 4 and "need --compare-sets" in sor3, f"rc {rcr3}")
            for nm, val in (("negative", "-0.1"), ("NaN", "nan"), ("infinite", "inf")):
                rcn, son, _ = gate(("--compare-atol", "0.5", "--compare-mean-atol", val), "mean_bad_" + nm)
                check(f"--compare-mean-atol {val} ({nm}) is REFUSED (exit 4)", rcn == 4 and "--compare-mean-atol" in son, f"rc {rcn}")
            rcm, som, _ = gate(("--compare-atol", "5", "--compare-mean-atol", "5", "--compare-min-matched", "21"), "mean_r4")
            check("--compare-min-matched 21 with 20 matched keys is still REFUSED (exit 4) with both bounds given, and neither marker is printed", rcm == 4 and "fewer than --compare-min-matched 21" in som and "ZZRLFAN_MEAN_ATOL" not in som and "ZZRLFAN_ATOL" not in som, f"rc {rcm}")
            ap = rl.build_parser().parse_args(["--ckpt", "x", "--keys", "y", "--targets", "t", "--images", "i", "--calib", "c", "--out", "o"])
            check("the parser default is None (no mean bound unless named in the argv)", ap.compare_mean_atol is None)
        guarded("mean", t_mean)

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
    # ---- --compare-atol (2026-10-12): each edit removes one property of the absolute-tolerance gate
    ("atol_flag_ignored_by_the_gate", "atol", "        elif a.compare_atol is not None:\n            if cmp_", "        elif False:\n            if cmp_"),
    ("atol_never_fails", "atol", 'if cmp_["beyond_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n                rc = EXIT_MISMATCH', 'if False:\n                rc = EXIT_MISMATCH'),
    ("atol_beyond_not_exit_5", "atol", 'if cmp_["beyond_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n                rc = EXIT_MISMATCH',
     'if cmp_["beyond_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n                rc = EXIT_OK'),
    ("atol_compared_with_ge", "atol", "if d > atol), key=lambda kd: -kd[1])", "if d >= atol), key=lambda kd: -kd[1])"),
    ("atol_doubled", "atol", "if d > atol), key=lambda kd: -kd[1])", "if d > 2 * atol), key=lambda kd: -kd[1])"),
    ("atol_default_path_tolerant", "atol", 'elif cmp_["not_identical"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n            rc = EXIT_MISMATCH',
     'elif cmp_["reference_lines_disagreeing_with_each_other"]:\n            rc = EXIT_MISMATCH'),
    ("atol_default_not_none", "atol", 'ap.add_argument("--compare-atol", type=float, default=None,', 'ap.add_argument("--compare-atol", type=float, default=0.5,'),
    ("atol_shape_mismatch_reads_zero", "atol", "if len(ta) != len(tb) or len(ya) != len(yb):\n        return math.inf, math.inf", "if len(ta) != len(tb) or len(ya) != len(yb):\n        return 0.0, 0.0"),
    ("atol_nan_xy_passes", "atol", "dxy = max(dxy, d if math.isfinite(d) else math.inf)", "dxy = max(dxy, d)"),
    ("atol_nan_yaw_passes", "atol", "dyaw = max(dyaw, d if math.isfinite(d) else math.inf)", "dyaw = max(dyaw, d)"),
    ("atol_without_compare_sets_ignored", "atol", "if (a.compare_atol is not None or a.compare_min_matched is not None) and a.compare_sets is None:", "if False:"),
    ("atol_negative_allowed", "atol", "if a.compare_atol is not None and not (math.isfinite(a.compare_atol) and a.compare_atol >= 0):", "if False:"),
    ("atol_min_matched_ignored", "atol", 'elif a.compare_min_matched is not None and cmp_["reference_keys_matched"] < a.compare_min_matched:', "elif False:"),
    ("atol_min_matched_zero_allowed", "atol", "if a.compare_min_matched is not None and a.compare_min_matched < 1:", "if False:"),
    ("atol_quantile_floor_rank", "atol", "return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]", "return sorted_vals[max(0, math.floor(q * len(sorted_vals)) - 1)]"),
    ("atol_identical_sets_not_counted_within", "atol", '"within_atol": len(per_set) - len(beyond)', '"within_atol": len(per_set) - len(beyond) - n_equal'),
    # ---- --compare-mean-atol (2026-10-12, round-2 PASS): each edit removes one property of the second bound
    ("mean_bound_ignored", "mean", "if m > mean_atol), key=lambda km: -km[1])", "if False), key=lambda km: -km[1])"),
    ("mean_compared_with_ge", "mean", "if m > mean_atol), key=lambda km: -km[1])", "if m >= mean_atol), key=lambda km: -km[1])"),
    ("mean_doubled", "mean", "if m > mean_atol), key=lambda km: -km[1])", "if m > 2 * mean_atol), key=lambda km: -km[1])"),
    ("mean_includes_yaw", "mean", "    (ta, _), (tb, _) = a, b\n", "    (ta, ya), (tb, yb) = a, b\n    ta, tb = ta + [[[u, u] for u in y] for y in ya], tb + [[[u, u] for u in y] for y in yb]\n"),
    ("mean_x_only", "mean", "for u, w in zip(qa, qb):\n                d = u - w", "for u, w in zip(qa[:1], qb[:1]):\n                d = u - w"),
    ("mean_is_max", "mean", "sa += abs(d)\n                n += 1", "sa = max(sa, abs(d))\n                n = 1"),
    ("mean_denominator_off_by_one", "mean", "return sa / n if n else 0.0", "return sa / (n + 1) if n else 0.0"),
    ("mean_proposal_mismatch_reads_zero", "mean", "    if len(ta) != len(tb):\n        return math.inf\n    sa, n = 0.0, 0", "    if False:\n        return math.inf\n    sa, n = 0.0, 0"),
    ("mean_step_mismatch_reads_zero", "mean", "        if len(pa) != len(pb):\n            return math.inf\n", "        if False:\n            return math.inf\n"),
    ("mean_coordinate_mismatch_reads_zero", "mean", "            if len(qa) != len(qb):\n                return math.inf\n", "            if False:\n                return math.inf\n"),
    ("mean_nan_passes", "mean", "                if not math.isfinite(d):\n                    return math.inf\n", "                if False:\n                    return math.inf\n"),
    ("mean_within_miscounted", "mean", '"within_mean_atol": len(mean_ms) - len(mbeyond)', '"within_mean_atol": len(mean_ms) - len(mbeyond) - 1'),
    ("mean_report_reads_the_max_delta", "mean", "        ms = sorted(m for _, m in mean_ms)\n", "        ms = sorted(d for _, d, _, _ in per_set)\n"),
    ("mean_keys_without_the_flag", "mean", "    if mean_atol is not None:\n        mbeyond", "    if True:\n        mbeyond"),
    ("mean_not_given_to_compare_sets", "mean", "atol=a.compare_atol, mean_atol=a.compare_mean_atol)", "atol=a.compare_atol)"),
    ("mean_flag_ignored_by_the_gate", "mean", '            if a.compare_mean_atol is not None:\n                if cmp_["beyond_mean_atol"]', '            if False:\n                if cmp_["beyond_mean_atol"]'),
    ("mean_never_fails", "mean", 'if cmp_["beyond_mean_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n                    rc = EXIT_MISMATCH',
     'if False:\n                    rc = EXIT_MISMATCH'),
    ("mean_beyond_not_exit_5", "mean", 'if cmp_["beyond_mean_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n                    rc = EXIT_MISMATCH',
     'if cmp_["beyond_mean_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:\n                    rc = EXIT_OK'),
    ("mean_disagreement_not_tripping", "mean", 'if cmp_["beyond_mean_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:', 'if cmp_["beyond_mean_atol"]:'),
    ("mean_marker_shows_the_catastrophe_bound", "mean", "ZZRLFAN_MEAN_ATOL_PASS {cmp_['reference_keys_matched']} {cmp_['max_mean_abs_delta']:.6g} {a.compare_mean_atol:g}",
     "ZZRLFAN_MEAN_ATOL_PASS {cmp_['reference_keys_matched']} {cmp_['max_mean_abs_delta']:.6g} {a.compare_atol:g}"),
    ("mean_default_not_none", "mean", 'ap.add_argument("--compare-mean-atol", type=float, default=None,', 'ap.add_argument("--compare-mean-atol", type=float, default=0.025,'),
    ("mean_without_atol_allowed", "mean", "if a.compare_mean_atol is not None and a.compare_atol is None:", "if False:"),
    ("mean_negative_allowed", "mean", "if a.compare_mean_atol is not None and not (math.isfinite(a.compare_mean_atol) and a.compare_mean_atol >= 0):", "if False:"),
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
    need = {"ckpt", "shard", "fan", "format", "keys", "resume", "gate", "atol", "mean"}
    assert {c for _, c, _, _ in MUTATIONS} >= need, "a required mutation class is missing"
    n = 1 + len(MUTATIONS) + 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {len(base)} checks + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
