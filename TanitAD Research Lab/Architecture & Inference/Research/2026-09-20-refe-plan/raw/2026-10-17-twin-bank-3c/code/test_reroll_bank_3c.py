#!/usr/bin/env python3
"""Offline tests for reroll_bank_3c.py (plan_shards + merge + progress). No nuplan, no pod:  python3 test_reroll_bank_3c.py
Every guard has a MUTATION arm (a deliberately broken input that must be REFUSED), so a green run cannot come from a merge that checks nothing."""
from __future__ import annotations

import io
import json
import random
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reroll_bank_3c as B  # noqa: E402


def lane_row(log, tok, k, extra=0):
    return {"image": ["a"], "ego": [1.0, 0, 0, 0, 0, 0, 5.0], "goal": [1.0, 2.0, 3.0, 4.0], "traj": [[0.1 * i, 0.0, 0.0] for i in range(20)],
            "traj_world": [[float(i), 0.0, 0.0] for i in range(20)], "origin_world": [0.0, 0.0, 0.0], "rank": 1, "step": 3, "token": tok,
            "log_name": log, "sim_hz": 10, "aug": {"kind": "lane_rank", "value": k, "divergence_m": 0.7 + extra, "lanes": 2}}


def goal_row(log, tok):
    return {"token": tok, "log_name": log, "rank": 1, "traj": [[0.0, 0.0, 0.0]], "aug": {"kind": "goal_horizon_s", "value": 8.0}}


def kept_version(r, f10=0.9):
    o = json.loads(json.dumps(r))
    o["traj"] = [[0.5 * i, 0.1, 0.0] for i in range(20)]
    o["aug"].update(divergence_m=1.5, route_hold="step0", goal_horizon_s=6.0, commit_f10=f10, reroll="3c")
    return o


def build(tmp: Path, n_k1=40, keep_every=2, n_shards=3, arm="held", sabotage=None):
    """A synthetic bank (lane rows interleaved with odd-byte non-lane rows) and fully written shard dirs. Returns (bank dir, shards glob, expected lines)."""
    rng = random.Random(7)
    lines, keys = [], []
    for i in range(n_k1):
        lg = f"log{i % 7}"
        r = lane_row(lg, f"{i:016x}", 1)
        lines.append(json.dumps(r) + "\n"); keys.append(B.key_of(r))
        if i % 3 == 0:                                   # a non-lane row, deliberately odd bytes (double spaces, unicode, key order) -- must pass through verbatim
            lines.append('{"token":  "%016x", "log_name": "%s",  "aug": {"kind": "goal_horizon_s", "value": 8.0}, "note": "ü ß"}\n' % (10 ** 9 + i, lg))
    for i in range(6):
        r = lane_row("logK2", f"{i + 500:016x}", 2)
        lines.append(json.dumps(r) + "\n"); keys.append(B.key_of(r))
    lines.append(json.dumps(goal_row("logZ", "ffffffffffffffff")))          # last line WITHOUT a trailing newline
    bank = tmp / "bank"; bank.mkdir()
    (bank / "targets_rank1.jsonl").write_bytes("".join(lines).encode("utf-8"))
    (bank / "targets_rank0.jsonl").write_bytes(b'{"rank": 0}\n')
    in_sha = B.sha256_file(bank / "targets_rank1.jsonl")
    units, assign = B.plan_shards(keys, n_shards, 8)
    by_key = {B.key_of(json.loads(l)): l for l in lines if '"lane_rank"' in l}
    out = tmp / "out" / "shards"
    kept_n = 0
    for i in range(n_shards):
        sd = out / f"s{i:04d}"; sd.mkdir(parents=True)
        meta = {"i": i, "N": n_shards, "chunk": 8, "arm": arm, "input_sha256": in_sha, "goal_horizon_s": 6.0, "span_s": 10.0, "keep_f10": 0.75,
                "code": {"x": "y"}}
        (sd / "shard_meta.json").write_text(json.dumps(meta))
        fa, fr = open(sd / "audit.jsonl", "w"), open(sd / "rows.jsonl", "w")
        for u in assign[i]:
            for k in units[u][2]:
                j = int(k[1], 16)
                keep = (k[2] == 1 and j % keep_every == 0) or (k[2] == 2 and j % 2 == 0)
                rec = {"log_name": k[0], "token": k[1], "k": k[2], "step": 3, "arm": arm, "v0": 5.0, "f_at_s": [0.9] * 10 if keep else [0.1] * 10, "f10_raw": 0.9 if keep else 0.1,
                       "d_twin_at_s": [1.0] * 10, "commit": bool(keep), "drop_reason": None if keep else "not_committed", "rollout_s": 12.0}
                if keep:
                    fr.write(json.dumps(kept_version(json.loads(by_key[k]), 0.9)) + "\n"); kept_n += 1
                fa.write(json.dumps(rec) + "\n")
        fa.close(); fr.close()
        (sd / "DONE").write_text("{}")
    if sabotage:
        sabotage(out, bank)
    return bank, str(out / "s*"), lines, keys, kept_n


def run_merge(bank, shards, dst):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = B.main(["merge", "--bank", str(bank), "--shards", shards, "--out-dir", str(dst)])
    return rc, buf.getvalue()


def test_plan_shards():
    rng = random.Random(1)
    keys = [(f"log{rng.randint(0, 30)}", f"{i:016x}", rng.randint(1, 3)) for i in range(500)]
    keys = list(dict.fromkeys(keys))
    for n in (1, 2, 7, 90):
        for chunk in (1, 8, 32):
            units, assign = B.plan_shards(keys, n, chunk)
            flat = [k for i in range(n) for u in assign[i] for k in units[u][2]]
            assert sorted(flat) == sorted(keys) and len(flat) == len(keys), (n, chunk)            # every twin in exactly one shard
            assert all(len({k[0] for k in ks}) == 1 and len(ks) <= chunk for _, _, ks in units)  # a unit never spans logs
            assert (units, assign) == B.plan_shards(keys, n, chunk)                              # deterministic
    # MUTATION: a plan that drops a unit must be detectable by the same identity
    units, assign = B.plan_shards(keys, 7, 8)
    assign[3] = assign[3][1:]
    flat = [k for i in range(7) for u in assign[i] for k in units[u][2]]
    assert len(flat) != len(keys)
    print("test_plan_shards OK")


def test_merge_ok():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bank, shards, lines, keys, kept_n = build(tmp)
        rc, outp = run_merge(bank, shards, tmp / "merged")
        assert rc == 0, outp
        got = (tmp / "merged" / "targets_rank1.jsonl").read_bytes().split(b"\n")
        src = "".join(lines).encode("utf-8").split(b"\n")
        # non-lane rows: byte-identical AND in the same relative order; lane rows: kept replaced in place, dropped absent
        src_other = [l for l in src if b'"lane_rank"' not in l]
        got_other = [l for l in got if b'"lane_rank"' not in l]
        assert src_other == got_other
        rep = json.loads((tmp / "merged" / "merge_report.json").read_text())
        assert rep["output"]["kept_twins"] == kept_n and rep["output"]["lines"] == rep["input"]["lines"] - rep["output"]["dropped_twins"]
        assert rep["by_rank"]["1"]["in"] == 40 and rep["by_rank"]["2"]["in"] == 6
        # order of kept lane rows == input order
        kept_in_order = [B.key_of(json.loads(l)) for l in got if b'"lane_rank"' in l]
        assert kept_in_order == [k for k in keys if k in set(kept_in_order)]
        assert all(json.loads(l)["aug"]["reroll"] == "3c" for l in got if b'"lane_rank"' in l)
        assert (tmp / "merged" / "targets_rank0.jsonl").read_bytes() == b'{"rank": 0}\n'
        print("test_merge_ok OK  (kept %d, lines %d -> %d)" % (kept_n, rep["input"]["lines"], rep["output"]["lines"]))


def test_merge_mutations():
    def drop_audit(out, bank):                              # one audit record missing => unfinished work must be REFUSED
        p = out / "s0001" / "audit.jsonl"
        ls = p.read_text().splitlines(); p.write_text("\n".join(ls[:-1]) + "\n")

    def wrong_input(out, bank):                             # shards were run on another input
        with open(bank / "targets_rank1.jsonl", "ab") as fh:
            fh.write(b'{"token": "00000000000000aa", "log_name": "x", "aug": {"kind": "goal_horizon_s", "value": 1.0}}\n')

    def tamper_row(out, bank):                              # a kept row whose commit_f10 disagrees with its audit
        p = out / "s0000" / "rows.jsonl"
        ls = [json.loads(l) for l in p.read_text().splitlines()]
        if not ls:
            p = out / "s0001" / "rows.jsonl"; ls = [json.loads(l) for l in p.read_text().splitlines()]
        ls[0]["aug"]["commit_f10"] = 0.5
        p.write_text("".join(json.dumps(l) + "\n" for l in ls))

    def missing_row(out, bank):                             # audited kept but its row is gone
        for i in range(3):
            p = out / f"s{i:04d}" / "rows.jsonl"
            if p.read_text().strip():
                p.write_text(""); return

    def legacy(out, bank):                                  # a control-arm shard (REFE_ROUTE_HOLD=0) must never become a bank
        for i in range(3):
            p = out / f"s{i:04d}" / "shard_meta.json"
            m = json.loads(p.read_text()); m["arm"] = "legacy"; p.write_text(json.dumps(m))

    def lost_shard(out, bank):
        import shutil; shutil.rmtree(out / "s0002")

    for name, fn in (("drop_audit", drop_audit), ("wrong_input", wrong_input), ("tamper_row", tamper_row), ("missing_row", missing_row),
                     ("legacy_arm", legacy), ("lost_shard", lost_shard)):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bank, shards, *_ = build(tmp, sabotage=fn)
            rc, outp = run_merge(bank, shards, tmp / "merged")
            assert rc == 3 and "REFUSED" in outp, (name, rc, outp)
            assert not (tmp / "merged" / "targets_rank1.jsonl").exists(), name
    print("test_merge_mutations OK (6 refusals)")


def test_yield_guard():
    with tempfile.TemporaryDirectory() as td:                # 40 K=1 twins, 1 in 20 kept => 5 %: TRIPS
        tmp = Path(td)
        bank, shards, *_ = build(tmp, keep_every=20)
        rc, outp = run_merge(bank, shards, tmp / "merged")
        assert rc == 4 and "YIELD_GUARD_TRIPPED" in outp, (rc, outp)
        assert not (tmp / "merged" / "targets_rank1.jsonl").exists() and (tmp / "merged" / "targets_rank1.jsonl.REJECTED_YIELD_GUARD").exists()
    with tempfile.TemporaryDirectory() as td:                # 1 in 2 kept => 50 %: holds
        tmp = Path(td)
        bank, shards, *_ = build(tmp, keep_every=2)
        rc, outp = run_merge(bank, shards, tmp / "merged")
        assert rc == 0
        assert json.loads((tmp / "merged" / "merge_report.json").read_text())["yield_guard"]["status"] == "HOLDS"
    with tempfile.TemporaryDirectory() as td:                # n_k1 = 10 < 30: guard not evaluated (never trips on a toy bank)
        tmp = Path(td)
        bank, shards, *_ = build(tmp, n_k1=10, keep_every=20)
        rc, outp = run_merge(bank, shards, tmp / "merged")
        assert rc == 0 and json.loads((tmp / "merged" / "merge_report.json").read_text())["yield_guard"]["status"].startswith("NOT_EVALUATED")
    print("test_yield_guard OK")


def test_repair_tail_and_tolerant_read():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.jsonl"
        p.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": 3')                 # a killed writer's partial last line
        assert [r["a"] for r in B._read_jsonl_tolerant(p)] == [1, 2]
        B._repair_tail(p)
        assert p.read_bytes() == b'{"a": 1}\n{"a": 2}\n'
        p.write_bytes(b'{"a": 1}\nGARBAGE\n{"a": 3}\n')               # corruption in the MIDDLE is an error, not tolerated
        try:
            B._read_jsonl_tolerant(p); raise AssertionError("middle corruption was tolerated")
        except json.JSONDecodeError:
            pass
    print("test_repair_tail_and_tolerant_read OK")


if __name__ == "__main__":
    test_plan_shards(); test_merge_ok(); test_merge_mutations(); test_yield_guard(); test_repair_tail_and_tolerant_read()
    print("ALL OK")
