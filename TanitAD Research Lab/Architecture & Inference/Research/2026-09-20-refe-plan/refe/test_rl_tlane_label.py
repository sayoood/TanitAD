#!/usr/bin/env python3
"""Tests for refe/rl_tlane_label.py (RL-S1 TL-1: paper-pure teacher-lane labels from the frozen-fan dump).  Run: python refe/test_rl_tlane_label.py -> ZZTEST_PASS <n>.

CPU only, no simulator.  What is REAL here: `rl_fan_dump.set_record` (the dump's record), `onpolicy_relabel_teacher_lane.label_log` (the production labelling function, with
only its simulator `_S` replaced by a deterministic stub whose answer is a pure function of the candidate it is handed), `planner.repair_last_heading`,
`train.OnPolicyBank._apply_teacher_lane` (the trainer's own reader of the output) and `rls1_checks.tlane_check`.  What is stubbed: the DriveRL Scorer.  The stub's
centre-line distance depends on the candidate's x/y AND on its LAST yaw, so a missing repair, a dropped yaw column, a reversed proposal order or a wrong teacher is a
different number.  Expected values are recomputed here with independent numpy code, never read back from the module.
Each mutation edits the module's SOURCE (the snippet must occur exactly once) and the suite must then report >= 1 failing check.
The real simulator is exercised by eval/rl_tlane_identity.py + raw/rl_tlane/ (dev-box run against SFT-4's banked labels), not here.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_tlane_label.py")
sys.path.insert(0, HERE)
import torch  # noqa: E402

import onpolicy_relabel_teacher_lane as TL  # noqa: E402  (the REAL label_log)
import planner as PL  # noqa: E402
import rl_fan_dump as RFD  # noqa: E402
import rls1_checks as RC  # noqa: E402
import train as T  # noqa: E402

M_, T_ = 64, 20
CKPT = 10075
LOGS = ["logA", "logB", "logC"]


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ------------------------------------------------------------------------------------------------------------------ the stub simulator
def _t(a):
    return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float32)


def sim_leaves(xy, yaw):
    """Pure function of the candidate.  info = max |y| + 0.01 * |last yaw| (so the repaired last heading is visible); reward = 1 - mean|y| / 10."""
    xy, yaw = np.asarray(xy, float), np.asarray(yaw, float)
    info = float(np.abs(xy[:, 1]).max() + 0.01 * abs(yaw[-1]))
    return {"center_line.CenterLine.reward": 1.0 - float(np.abs(xy[:, 1]).mean()) / 10.0, "center_line.CenterLine.info": info,
            "off_road.CrossLane.reward": 1.0, "off_road.CrossLane.info": 0.0, "off_road.CrossLane.lane_change_info": float(xy[-1, 1] > 1.0),
            "off_road.CrossLane.solid_lane_change_info": 0.0, "off_road.OffRoad.info": 0.0, "ddc.violation": 0.0, "dac.violation": 0.0}


class StubScorer:
    def __init__(self, fail_tokens=()):
        self.calls, self.fail = [], set(fail_tokens)

    def scenarios(self, log, tokens):
        return {t: ("scenario", log, t) for t in tokens}

    def score(self, sc, row, cands):
        if sc[2] in self.fail:
            raise RuntimeError("simulated failure")
        self.calls.append((sc[2], [(n, np.asarray(x), np.asarray(y)) for n, x, y in cands]))
        return {n: sim_leaves(x.numpy(), y.numpy()) for n, x, y in cands}, 5


class Harness:
    """Installs the stub as the labeller's simulator (module-level _S, as TL._init would) and exposes the call log."""

    def __init__(self, fail_tokens=()):
        self.stub = StubScorer(fail_tokens)
        self.inits = []

    def init_fn(self, rank, stride, repair):
        self.inits.append((rank, stride, repair))
        TL._S["OL"] = types.SimpleNamespace(_t=_t)
        TL._S["S"] = self.stub
        TL._S["repair"] = repair


# ------------------------------------------------------------------------------------------------------------------ fixtures
def rng_fan(seed):
    r = np.random.RandomState(seed)
    xy = np.cumsum(r.rand(M_, T_, 2) * [1.0, 0.2] - [0.0, 0.1], axis=1)
    yaw = (r.rand(M_, T_) - 0.5) * 3.0
    return np.concatenate([xy, yaw[..., None]], -1)


def scenes_of(n_per_log):
    out = []
    for li, lg in enumerate(LOGS):
        for i in range(n_per_log):
            out.append((f"{lg.lower()}tok{i:02d}", lg, 3 * i, 100 * li + i))
    return out


def build(tmp, scenes, ckpt=CKPT, rank=0, extra=None):
    """-> (sets_dir, bank_dir, {key: (props [M][T][3], teacher [T][3])}).  Sets are written by the REAL dumper record function."""
    sd, bd = os.path.join(tmp, "sets"), os.path.join(tmp, "bank")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(bd, exist_ok=True)
    truth, sl, bl = {}, [], []
    for tok, lg, step, seed in scenes:
        P = rng_fan(seed)
        teacher = (rng_fan(seed + 5000)[0]).tolist()
        row = {"log_name": lg, "token": tok, "step": step, "aug": None}
        rec = RFD.set_record(row, rank, ckpt, P.tolist())
        sl.append(json.dumps(rec) + "\n")
        bl.append(json.dumps({"log_name": lg, "token": tok, "step": step, "rank": 0, "traj": teacher, "image": ["x"]}) + "\n")
        truth[(lg, tok, step, rank)] = (P, np.asarray(teacher))
    with open(os.path.join(sd, "onpolicy_rlfan_fan_s0of1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(sl)
    with open(os.path.join(bd, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(bl)
    return sd, bd, truth


def expected_label(P, teacher, repair=True):
    """Independent recomputation of one set's label: teacher first, then the 64 proposals, the last heading held when `repair` (planner semantics, rewritten here)."""
    P = np.array(P, copy=True)
    if repair:
        P[:, -1, 2] = P[:, -2, 2]
    t = sim_leaves(np.asarray(teacher)[:, :2], np.asarray(teacher)[:, 2])
    return t, [sim_leaves(P[k, :, :2], P[k, :, 2]) for k in range(P.shape[0])]


def mk_args(sets, bank, out, **kw):
    d = dict(sets=sets, bank=bank, out=out, keys=None, shard="0/1", workers=1, stride=2, expect_ckpt_step=CKPT, repair_last_heading=True, limit=0)
    d.update(kw)
    return argparse.Namespace(**d)


def run_tool(mod, args, hz=None):
    hz = hz or Harness()
    buf = io.StringIO()
    refused = None
    with contextlib.redirect_stdout(buf):
        try:
            rc = mod.run(args, label_fn=TL.label_log, init_fn=hz.init_fn)
        except mod.Refused as e:
            rc, refused = 4, str(e)
    return rc, refused, buf.getvalue(), hz


def read_out(out):
    rows = []
    for fp in sorted(glob.glob(os.path.join(out, "teacher_lane_r*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue                                                              # a torn fragment, as the trainer's reader skips it
    return rows


def refuses(mod, args, what):
    try:
        rc, refused, _, _ = run_tool(mod, args)
    except Exception as e:                                                              # noqa: BLE001
        return False, f"{what}: raised {type(e).__name__}: {e}"
    return rc == 4 and refused is not None, f"{what}: rc {rc} refused={refused!r}"


# ------------------------------------------------------------------------------------------------------------------ the suite
def suite(mod):
    R = []

    def chk(name, ok, detail=""):
        R.append((name, bool(ok), detail))

    scenes = scenes_of(5)                                             # 15 sets over 3 logs
    with tempfile.TemporaryDirectory() as tmp:
        sd, bd, truth = build(tmp, scenes)
        out = os.path.join(tmp, "out")
        rc, refused, text, hz = run_tool(mod, mk_args(sd, bd, out))
        rows = read_out(out)
        chk("run_exit_0_and_marker", rc == 0 and "ZZTLANE_DONE 15 0 0" in text, f"rc {rc} {text.strip().splitlines()[-1:]}")
        chk("one_row_per_set", len(rows) == 15 and len({tuple(r["key"]) for r in rows}) == 15, str(len(rows)))
        chk("labeller_initialised_rank0_stride2_repair", hz.inits == [(0, 2, True)], str(hz.inits))
        chk("row_schema", all(set(r) == {"kind", "key", "ckpt_step", "repair", "teacher", "props"} and r["kind"] == "teacher_lane_labels" and r["ckpt_step"] == CKPT
                              and r["repair"] == "last_heading_hold" and len(r["props"]) == 64 for r in rows), "")
        by = {tuple(r["key"]): r for r in rows}
        keys_ok = all(k in by for k in truth)
        chk("keys_are_log_token_step_rank", keys_ok, str(sorted(by)[:2]))
        # the numbers: every leaf of the teacher and of each of the 64 proposals equals the independent recomputation
        worst_t = worst = 0.0
        for k, (P, tr) in truth.items():
            t_exp, p_exp = expected_label(np.round(P, 5), tr, repair=True)          # the dump rounds proposals to 5 decimals; the teacher is verbatim
            got = by[k]
            worst_t = max(worst_t, max(abs(float(got["teacher"][lf]) - v) for lf, v in t_exp.items()))
            for g, e in zip(got["props"], p_exp):
                worst = max(worst, max(abs(float(g[lf]) - v) for lf, v in e.items()))
        chk("teacher_leaves_equal_independent_recompute", worst_t < 1e-5, f"max |delta| {worst_t:.2e} (the simulator is handed float32 tensors)")
        chk("proposal_leaves_equal_independent_recompute_in_order", worst < 1e-5, f"max |delta| {worst:.2e}")
        # what the simulator was HANDED: teacher first (the bank trajectory, verbatim), then the 64 proposals in order, last heading held
        calls = {c[0]: c[1] for c in hz.stub.calls}
        hand_ok = True
        for k, (P, tr) in truth.items():
            cands = calls.get(k[1])
            if cands is None or cands[0][0] != "teacher" or [c[0] for c in cands[1:]] != list(range(64)):
                hand_ok = False
                continue
            if not (np.array_equal(cands[0][1], tr[:, :2].astype(np.float32)) and np.array_equal(cands[0][2], tr[:, 2].astype(np.float32))):
                hand_ok = False
            R5 = np.round(P, 5)
            R5[:, -1, 2] = R5[:, -2, 2]
            if not (np.allclose(np.stack([c[1] for c in cands[1:]]), R5[..., :2], atol=1e-6) and np.allclose(np.stack([c[2] for c in cands[1:]]), R5[..., 2], atol=1e-6)):
                hand_ok = False
        chk("simulator_handed_teacher_first_then_64_props_in_order_with_held_heading", hand_ok, "")
        # the TRAINER's own reader accepts every row
        ob = object.__new__(T.OnPolicyBank)
        ob.by = {k: (CKPT,) for k in truth}
        ob.teacher_lane, ob.n_tlane, ob.n_tlane_stale = {}, 0, 0
        ob._apply_teacher_lane(out, 64)
        chk("trainer_OnPolicyBank_accepts_every_row", ob.n_tlane == 15 and ob.n_tlane_stale == 0, f"n_tlane {ob.n_tlane} stale {ob.n_tlane_stale}")
        k0 = sorted(truth)[0]
        d_exp = np.array([r_["center_line.CenterLine.info"] for r_ in expected_label(*truth[k0])[1]], np.float32)
        soft = np.clip(1.0 - (d_exp - T.OnPolicyBank.LANE_D0) / (T.OnPolicyBank.LANE_D1 - T.OnPolicyBank.LANE_D0), 0.0, 1.0)
        chk("trainer_soft_target_equals_recompute", np.allclose(ob.teacher_lane[k0], soft, atol=1e-4), "")
        tsv = os.path.join(tmp, "scenes.tsv")
        with open(tsv, "w", encoding="utf-8", newline="\n") as f:
            f.write("token\tlog\n" + "".join(f"{t}\t{lg}\n" for t, lg, _, _ in scenes))
        ok, why, n_good = RC.tlane_check(out, tsv, CKPT, 15)
        chk("rls1_checks_tlane_check_counts_all_15", ok and n_good == 15, why)

        # RESUME: a complete output relabels nothing; a truncated one relabels exactly the missing sets, keeps the rest
        hz2 = Harness()
        rc2, _, text2, _ = run_tool(mod, mk_args(sd, bd, out), hz2)
        chk("rerun_is_a_noop", rc2 == 0 and "ZZTLANE_DONE 0 15 0" in text2 and not hz2.stub.calls, text2.strip().splitlines()[-1:].__str__())
        fp = glob.glob(os.path.join(out, "teacher_lane_r*.jsonl"))[0]
        lines = open(fp, encoding="utf-8").read().splitlines(True)
        dropped = [tuple(json.loads(x)["key"]) for x in lines[-4:]]
        with open(fp, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines[:-4])
            f.write(lines[-4][:40])                                                       # a torn last line
        hz3 = Harness()
        rc3, _, text3, _ = run_tool(mod, mk_args(sd, bd, out), hz3)
        redone = sorted(tuple(r["key"]) for r in read_out(out))
        chk("resume_relabels_only_the_missing", rc3 == 0 and sorted(c[0] for c in hz3.stub.calls) == sorted(k[1] for k in dropped), str(sorted(c[0] for c in hz3.stub.calls)))
        chk("resume_keeps_earlier_rows_and_completes", len(read_out(out)) == 15 + 0 and sorted(set(redone)) == sorted(truth), f"{len(redone)} rows")

        # a stale / foreign row never counts as labelled: wrong ckpt_step, or no repair marker
        out_b = os.path.join(tmp, "out_b")
        os.makedirs(out_b)
        k1 = sorted(truth)[1]
        with open(os.path.join(out_b, "teacher_lane_r0_s0of1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"kind": "teacher_lane_labels", "key": list(k1), "ckpt_step": 9999, "repair": "last_heading_hold", "teacher": {},
                                "props": [{"center_line.CenterLine.info": 0.1}] * 64}) + "\n")                       # valid in every respect but the checkpoint
            f.write(json.dumps({"kind": "teacher_lane_labels", "key": list(sorted(truth)[2]), "ckpt_step": CKPT, "repair": None, "teacher": {},
                                "props": [{"center_line.CenterLine.info": 0.1}] * 64}) + "\n")                       # valid in every respect but the recipe marker
            f.write(json.dumps({"kind": "teacher_lane_labels", "key": list(sorted(truth)[4]), "ckpt_step": CKPT, "repair": "last_heading_hold", "teacher": {},
                                "props": [{"center_line.CenterLine.info": None}] * 64}) + "\n")                       # right key, right ckpt, right repair, null distances
        hz4 = Harness()
        rc4, _, text4, _ = run_tool(mod, mk_args(sd, bd, out_b), hz4)
        chk("foreign_rows_are_not_credited_and_the_sets_are_labelled", rc4 == 0 and len(hz4.stub.calls) == 15, f"rc {rc4} calls {len(hz4.stub.calls)}")

        # a foreign-checkpoint row for a set that then FAILS must not make the set look labelled (the final coverage check, not the resume logic, is what stops it)
        out_d = os.path.join(tmp, "out_d")
        os.makedirs(out_d)
        with open(os.path.join(out_d, "teacher_lane_r0_s0of1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"kind": "teacher_lane_labels", "key": list(k1), "ckpt_step": 9999, "repair": "last_heading_hold", "teacher": {},
                                "props": [{"center_line.CenterLine.info": 0.1}] * 64}) + "\n")
        rcd, _, textd, _ = run_tool(mod, mk_args(sd, bd, out_d), Harness(fail_tokens=[k1[1]]))
        chk("foreign_ckpt_row_does_not_cover_a_failed_set", rcd == 3 and "ZZTLANE_DONE 14 0 1" in textd, f"rc {rcd} {textd.strip().splitlines()[-2:]}")

        # a failing set: the others are written, the exit is 3, the marker names the count, a rerun finishes
        out_c = os.path.join(tmp, "out_c")
        bad_tok = sorted(truth)[3][1]
        rc5, _, text5, _ = run_tool(mod, mk_args(sd, bd, out_c), Harness(fail_tokens=[bad_tok]))
        chk("one_failing_set_exit_3_others_written", rc5 == 3 and "ZZTLANE_DONE 14 0 1" in text5 and len(read_out(out_c)) == 14, f"rc {rc5} {text5.strip().splitlines()[-2:]}")
        rc6, _, text6, hz6 = run_tool(mod, mk_args(sd, bd, out_c))
        chk("rerun_after_failure_labels_only_that_set", rc6 == 0 and [c[0] for c in hz6.stub.calls] == [bad_tok] and len(read_out(out_c)) == 15, text6.strip().splitlines()[-1:].__str__())

        # SHARDS: disjoint, exhaustive, by log, the same function the dump and the terms labeller use
        got = []
        for i in range(3):
            o = os.path.join(tmp, f"sh{i}")
            rcs, _, _, _ = run_tool(mod, mk_args(sd, bd, o, shard=f"{i}/3"))
            rs = read_out(o)
            got.append({tuple(r["key"]) for r in rs})
            chk(f"shard_{i}_of_3_rows_are_its_logs", rcs in (0, 4) and all(RFD.shard_of(r["key"][0], 3) == i for r in rs), f"rc {rcs}")
        chk("shards_disjoint_and_exhaustive", sum(len(g) for g in got) == 15 and set().union(*got) == set(truth) and not (got[0] & got[1]) and not (got[1] & got[2]), str([len(g) for g in got]))
        import rl_terms_label as RTL_
        chk("shard_function_equals_dump_and_terms_labeller", all(mod.shard_of(lg, n) == RFD.shard_of(lg, n) == RTL_.shard_of(lg, n) for lg in LOGS + ["x", "2021.06.14.18.13.35_veh-26_01537_01717"] for n in (1, 2, 3, 4, 7)), "")

        # PIPELINING: the chain labels shard 0 while the dump is still writing the other shards into the SAME directory.  A half-written last line in a
        # neighbour's file must not stop shard 0's labeller, and shard 0 must not label a neighbour's sets.  (3 shards: logC -> 0, logA and logB -> 2.)
        sdp = os.path.join(tmp, "sets_piped")
        os.makedirs(sdp)
        recs_all = [json.loads(x) for x in open(os.path.join(sd, "onpolicy_rlfan_fan_s0of1.jsonl"), encoding="utf-8")]
        part = {0: [], 1: [], 2: []}
        for r_ in recs_all:
            part[RFD.shard_of(r_["log_name"], 3)].append(json.dumps(r_) + "\n")
        for i_ in (0, 1, 2):
            with open(os.path.join(sdp, f"onpolicy_rlfan_fan_s{i_}of3.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(part[i_])
                if i_ == 2:
                    f.write('{"kind": "onpolicy_set", "log_name": "logB", "tok')                  # the dump is mid-write here
        o = os.path.join(tmp, "piped0")
        rcp, refp, _, _ = run_tool(mod, mk_args(sdp, bd, o, shard="0/3"))
        got0 = {tuple(r["key"]) for r in read_out(o)}
        want0 = {k for k in truth if RFD.shard_of(k[0], 3) == 0}
        chk("shard_reads_only_its_own_dump_file_not_a_half_written_neighbour", rcp == 0 and refp is None and got0 == want0 and want0 and len(want0) < len(truth),
            f"rc {rcp} {refp} {len(got0)}/{len(want0)}/{len(truth)}")

        # LIMIT (the smoke flag): the first N sets in (log, token) order
        o = os.path.join(tmp, "lim")
        rcl, _, _, hzl = run_tool(mod, mk_args(sd, bd, o, limit=3))
        chk("limit_labels_the_first_n_in_log_token_order", rcl == 0 and [c[0] for c in hzl.stub.calls] == sorted(k[1] for k in truth)[:3], str([c[0] for c in hzl.stub.calls]))

        # REFUSALS (nothing may be written)
        for what, kw in (("no_repair_flag", dict(repair_last_heading=False)), ("wrong_expected_ckpt", dict(expect_ckpt_step=1234)), ("bad_shard", dict(shard="0/0")), ("bad_shard_index", dict(shard="3/3")),
                         ("keys_outside_partition", dict(keys=[os.path.join(tmp, "other.tsv")]))):
            if what == "keys_outside_partition":
                with open(kw["keys"][0], "w", encoding="utf-8", newline="\n") as f:
                    f.write("token\tlog\n" + "".join(f"{t}\t{lg}\n" for t, lg, _, _ in scenes[:-1]))
            o = os.path.join(tmp, "ref_" + what)
            ok, why = refuses(mod, mk_args(sd, bd, o, **kw), what)
            chk(f"refuses_{what}", ok and not read_out(o), why)

        # a fan from another checkpoint, a rank-1 set, a doubled key, a torn dump line
        for what, tweak in (("foreign_ckpt_set", lambda recs: recs[0].__setitem__("ckpt_step", 10074)),
                            ("rank1_set", lambda recs: recs[0].__setitem__("rank", 1)),
                            ("duplicate_set", lambda recs: recs.append(dict(recs[0]))),
                            ("not_an_onpolicy_set", lambda recs: recs[0].__setitem__("kind", "onpolicy_props")),
                            ("short_fan", lambda recs: recs[0].__setitem__("traj", recs[0]["traj"][:63]))):
            sd2 = os.path.join(tmp, "sets_" + what)
            os.makedirs(sd2)
            recs = [json.loads(x) for x in open(os.path.join(sd, "onpolicy_rlfan_fan_s0of1.jsonl"), encoding="utf-8")]
            tweak(recs)
            with open(os.path.join(sd2, "onpolicy_rlfan_fan_s0of1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            o = os.path.join(tmp, "ref_" + what)
            bank_for = bd
            if what == "rank1_set":                                                       # the bank HAS the rank-1 row, so only the rank rule can refuse
                bank_for = os.path.join(tmp, "bank_with_rank1")
                os.makedirs(bank_for)
                blines = open(os.path.join(bd, "targets_rank0.jsonl"), encoding="utf-8").read().splitlines(True)
                r1 = json.loads(blines[0])
                r1["rank"] = 1
                with open(os.path.join(bank_for, "targets_rank1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(r1) + "\n")
                with open(os.path.join(bank_for, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(blines[1:])
            ok, why = refuses(mod, mk_args(sd2, bank_for, o), what)
            chk(f"refuses_{what}", ok and not read_out(o), why)
        sd3 = os.path.join(tmp, "sets_torn")
        os.makedirs(sd3)
        txt = open(os.path.join(sd, "onpolicy_rlfan_fan_s0of1.jsonl"), encoding="utf-8").read()
        with open(os.path.join(sd3, "onpolicy_rlfan_fan_s0of1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(txt[:-50])
        ok, why = refuses(mod, mk_args(sd3, bd, os.path.join(tmp, "ref_torn")), "torn_dump_line")
        chk("refuses_torn_dump_line", ok, why)
        ok, why = refuses(mod, mk_args(os.path.join(tmp, "nonexistent"), bd, os.path.join(tmp, "ref_empty")), "no_dump_files")
        chk("refuses_missing_dump", ok, why)

        # the bank: a dumped set with no bank row; a bank that holds a wanted key twice
        bd2 = os.path.join(tmp, "bank_missing")
        os.makedirs(bd2)
        bl = open(os.path.join(bd, "targets_rank0.jsonl"), encoding="utf-8").read().splitlines(True)
        with open(os.path.join(bd2, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(bl[1:])
        ok, why = refuses(mod, mk_args(sd, bd2, os.path.join(tmp, "ref_nobank")), "set_without_bank_row")
        chk("refuses_set_without_bank_row", ok, why)
        bd3 = os.path.join(tmp, "bank_dup")
        os.makedirs(bd3)
        with open(os.path.join(bd3, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(bl + [bl[0]])
        ok, why = refuses(mod, mk_args(sd, bd3, os.path.join(tmp, "ref_dupbank")), "bank_holds_key_twice")
        chk("refuses_bank_row_twice", ok, why)
        bd4 = os.path.join(tmp, "bank_short")
        os.makedirs(bd4)
        r0 = json.loads(bl[0])
        r0["traj"] = r0["traj"][:19]
        with open(os.path.join(bd4, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines([json.dumps(r0) + "\n"] + bl[1:])
        ok, why = refuses(mod, mk_args(sd, bd4, os.path.join(tmp, "ref_shortteacher")), "short_teacher")
        chk("refuses_teacher_of_wrong_length", ok, why)

        # props_from_set / queue_row pinned on a hand-made record (4 numbers derived by hand)
        rec = RFD.set_record({"log_name": "L", "token": "T", "step": 7, "aug": None}, 0, CKPT, np.zeros((64, 20, 3)).tolist())
        rec["traj"][2][5] = [1.25, -0.5]
        rec["yaw"][2][5] = 0.125
        pr = mod.props_from_set(rec)
        chk("props_from_set_layout_x_y_yaw", pr[2][5] == [1.25, -0.5, 0.125] and len(pr) == 64 and len(pr[0]) == 20 and pr[3][5] == [0.0, 0.0, 0.0], str(pr[2][5]))
        qr = mod.queue_row(rec, [[1.0, 2.0, 3.0]] * 20)
        chk("queue_row_has_the_dumpers_fields", qr["kind"] == "onpolicy_props" and qr["step"] == 7 and qr["rank"] == 0 and qr["ckpt_step"] == CKPT
            and qr["teacher"][0] == [1.0, 2.0, 3.0] and qr["aug"] is None and qr["log_name"] == "L" and qr["token"] == "T", "")
    return R


# ------------------------------------------------------------------------------------------------------------------ mutations (source edits; each must turn >= 1 check red)
MUTATIONS = [
    ("teacher_is_a_proposal", "teacher", "queue_row(recs[k], teachers[k])", "queue_row(recs[k], props_from_set(recs[k])[0])"),
    ("yaw_column_zeroed", "join", "float(yaw[k][t])] for t in range(HORIZON)])", "0.0] for t in range(HORIZON)])"),
    ("proposals_reversed", "join", "        out.append([[float(traj[k][t][0])", "        out.insert(0, [[float(traj[k][t][0])"),
    ("xy_swapped", "join", "[[float(traj[k][t][0]), float(traj[k][t][1]), float(yaw[k][t])]", "[[float(traj[k][t][1]), float(traj[k][t][0]), float(yaw[k][t])]"),
    ("repair_flag_not_required", "repair", 'if not args.repair_last_heading:\n        raise Refused("--repair-last-heading is required', 'if False:\n        raise Refused("--repair-last-heading is required'),
    ("repair_off_in_the_labeller", "repair", "bool(args.repair_last_heading))", "False)"),
    ("stride_one", "repair", "init_args = (0, args.stride,", "init_args = (0, 1,"),
    ("rank_one_init", "repair", "init_args = (0, args.stride,", "init_args = (1, args.stride,"),
    ("ckpt_step_unchecked", "ckpt", 'if int(r.get("ckpt_step", -1)) != expect_ckpt_step:', "if False:"),
    ("rank_unchecked", "ckpt", "if k[3] != 0:", "if False:"),
    ("duplicate_key_unchecked", "keys", "                if k in recs:\n", "                if False:\n"),
    ("keys_not_enforced", "keys", "if keys is not None and (k[1], k[0]) not in keys:", "if False:"),
    ("kind_unchecked", "format", 'if r.get("kind") != "onpolicy_set":', "if False:"),
    ("fan_size_unchecked", "format", "if len(traj) != N_PROP or len(yaw) != N_PROP:", "if False:"),
    ("torn_line_skipped", "format", 'raise Refused(f"{fp}:{n}: not JSON (a torn dump line: finish the dump first)")', "continue"),
    ("shard_unfiltered", "shard", "if shard_of(k[0], shard[1]) != shard[0]:", "if False:"),
    ("shard_reads_every_dump_file", "shard", '    files = (sorted(glob.glob(os.path.join(sets_dir, f"onpolicy_rlfan_*_s{shard[0]}of{shard[1]}.jsonl")))\n             or sorted(',
     '    files = (sorted(glob.glob(os.path.join(sets_dir, "onpolicy_rlfan_*.jsonl")))\n             or sorted('),
    ("shard_by_token", "shard", "if shard_of(k[0], shard[1]) != shard[0]:", "if shard_of(k[1], shard[1]) != shard[0]:"),
    ("shard_bounds_unchecked", "shard", "if not (n >= 1 and 0 <= i < n):", "if False:"),
    ("missing_bank_row_ignored", "bank", "    if missing:\n        raise Refused", "    if False:\n        raise Refused"),
    ("bank_duplicates_ignored", "bank", "    if dup:\n        raise Refused", "    if False:\n        raise Refused"),
    ("teacher_shape_unchecked", "bank", "if len(tr) != HORIZON or any(len(p) != 3 for p in tr):", "if False:"),
    ("resume_ignores_done", "resume", "    done = read_done(args.out)\n", "    done = set()\n"),
    ("resume_ignores_ckpt_step", "resume", 'done.add(((k[0], k[1], int(k[2]), int(k[3])), int(d["ckpt_step"])))', 'done.add(((k[0], k[1], int(k[2]), int(k[3])), CKPT_STEP))'),
    ("resume_credits_invalid_rows", "resume", "                    if valid_label_row(d):\n                        k = tuple(d[\"key\"])", "                    if True:\n                        k = tuple(d[\"key\"])"),
    ("valid_row_accepts_null_distance", "resume", 'all(p.get("center_line.CenterLine.info") is not None and math.isfinite(float(p["center_line.CenterLine.info"])) for p in props)', "True"),
    ("valid_row_ignores_repair_marker", "resume", ' and d.get("repair") == "last_heading_hold"', ""),
    ("output_truncated_each_run", "resume", 'with open(dst, "a", encoding="utf-8", newline="\\n") as fh:', 'with open(dst, "w", encoding="utf-8", newline="\\n") as fh:'),
    ("torn_line_not_terminated", "resume", "    terminate_torn_line(dst)\n    with open(dst", "    with open(dst"),
    ("coverage_accepts_anything", "gate", "ok_keys = {k: False for k in recs}", "ok_keys = {k: True for k in recs}"),
    ("coverage_ignores_ckpt_step", "gate", 'int(d.get("ckpt_step", -1)) == int(recs[k]["ckpt_step"]) and ', ""),
    ("failure_exits_zero", "gate", "        return EXIT_FAILED\n    return EXIT_OK", "        return EXIT_OK\n    return EXIT_OK"),
    ("limit_ignored", "keys", "if args.limit:\n        recs = ", "if False:\n        recs = "),
    ("step_dropped_from_key", "join", 'return (rec["log_name"], rec["token"], int(rec["step"]), int(rec.get("rank", 0)))', 'return (rec["log_name"], rec["token"], 0, int(rec.get("rank", 0)))'),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    base = suite(load(src, "rl_tlane_label_under_test"))
    real = [(n, d) for n, ok, d in base if not ok]
    for n, ok, d in base:
        if not ok:
            print(f"  FAIL {n}  {d}")
    assert not real, f"the unmutated module FAILS its own checks: {[n for n, _ in real]}"
    print(f"unmutated: {len(base)} checks green")
    print(f"{'mutation':32s} {'class':8s} caught-by (failing checks)")
    for name, cls, old, new in MUTATIONS:
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"rl_tlane_label_mut_{name}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            failing = [n for n, ok, _ in suite(mod) if not ok]
        except Exception as e:                                                          # noqa: BLE001  (a crash IS the suite seeing the defect)
            failing = [f"suite raised {type(e).__name__}"]
        print(f"{name:32s} {cls:8s} {len(failing)}  {failing[0][:70] if failing else ''}")
        assert failing, f"mutation {name} ({cls}) stayed GREEN: the suite cannot see it"
    need = {"join", "repair", "ckpt", "keys", "format", "shard", "bank", "resume", "gate"}
    assert {c for _, c, _, _ in MUTATIONS} >= need, "a required mutation class is missing"
    n = 1 + len(MUTATIONS) + 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {len(base)} checks + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
