#!/usr/bin/env python3
"""P5-C tests (scorer_gym_20261107/p5c/P5C_READINESS.md section 3 and G-LIVE (a)-(e)): the row -> trajectory label table and the trainer flag `--traj-label-table`.

  python test_traj_label_table.py            everything, incl. the source-mutation harness (minutes, CPU)
  python test_traj_label_table.py --fast     no mutation harness
  python test_traj_label_table.py --check N  one named check (what the mutation harness runs on a mutated copy of this tree)

CPU only (CUDA_VISIBLE_DEVICES=-1), synthetic banks, torch + numpy only (NO scipy: the Thor venv has none).  Prints ZZTRAJLABEL_PASS <n>.
The mechanism is a DATA SUBSTITUTION, so the probe is not a parameter gradient: (a) through the REAL TargetBank class item[3] is the table's label bit for bit and flag-off is raw bit for bit,
(b) with fixed weights the loss differs with the table and differs by EXACTLY 0.0 without it, and the target reaches the gradient (a zero control: a prediction equal to the label has an exactly
zero gradient under the table and a non-zero one under the raw traj), (c) an identity table reproduces flag-off bit for bit, also END TO END through train.py (identical parameters), (d) the
deliberate-regression arms must FAIL: label_table_not_passed (G-DVB, and a source mutant) and label_lookup_removed (a source mutant of the lookup), (e) `verify` = the check-bank.
Every positive assertion has a negative twin; the MUTATIONS table DELETES / neutralises one emitting line of a COPY of this tree and requires the check to go RED.  Any skip would print ZZTEST_SKIP.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import traj_label_table as TLT  # noqa: E402

N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    return None


# ------------------------------------------------------------------------------------------------ fixtures
def make_bank(root: Path, n_scenes: int = 5) -> str:
    """tuples in `n_scenes` scenes; scenes 0 and 1 also carry a rank-1 goal TWIN that shares log, token AND the rank-0 traj bytes (the real banks' property: why the key has the rank)."""
    rng = np.random.default_rng(7)
    d = root / "bank"
    d.mkdir(parents=True, exist_ok=True)
    rows0, rows1 = [], []
    for s in range(n_scenes):
        gy = [0.0, 9.0, -9.0, 1.0, 3.0][s % 5]
        e = [float(x) for x in rng.normal(0, 0.3, 7)]
        e[4], e[5] = 0.0, 0.0
        base = {"image": [f"log{s}/CAM_F0/a{s}.jpg"], "cameras": ["CAM_F0"], "ego": e, "goal": [30.0, gy, 60.0, 2 * gy],
                "traj": [[float(0.7 * (t + 1) + 0.013 * s), float(gy * (t + 1) / 80.0), float(0.02 * s + 0.001 * t)] for t in range(20)],
                "step": 0, "token": f"tok{s:04x}", "log_name": f"log{s}", "scenario_type": "t", "source": "fixture"}
        rows0.append({**base, "rank": 0})
        if s < 2:
            rows1.append({**base, "rank": 1, "goal": [30.0, -gy, 60.0, -2 * gy], "aug": {"kind": "lane_rank", "value": 1, "divergence_m": 2.0}})
    for nm, rows in (("targets_rank0.jsonl", rows0), ("targets_rank1.jsonl", rows1)):
        (d / nm).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return str(d)


def label_of(r: dict) -> list:
    """a label that differs from the row's traj in every pose and is DISTINCT per (token, rank): the Tc20s stand-in.  Yaw stays on the raw branch."""
    k = (int(r["token"][3:], 16) + 1) * 0.1 + 0.05 * int(r.get("rank", 0))
    return [[float(p[0] - k * 0.5 + 0.001 * i), float(p[1] + k * 0.25), float(p[2] + 0.01 * k)] for i, p in enumerate(r["traj"])]


def make_table(root: Path, rows: list, name="t.json.gz", skip_idx=(), require=0.0, label_name="Tc20s-fixture", fn=label_of) -> str:
    labs = {TLT.row_triple(r): (fn(r), None) for i, r in enumerate(rows) if i not in skip_idx}
    tab, meta, _ = TLT.build_table(rows, labs, {"label_name": label_name}, require_coverage=require, allow_partial=bool(skip_idx))
    p = root / name
    TLT.save(p, tab, meta)
    return str(p)


def cfg_for_bank():
    from model import REFeConfig
    c = REFeConfig.for_backbone("vits16")
    c.n_cameras = 1
    return c


def same_item(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if not (isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


def f32(label) -> torch.Tensor:
    return torch.tensor(label, dtype=torch.float32)


# ------------------------------------------------------------------------------------------------ S1 the table tool
def chk_table():
    td = Path(tempfile.mkdtemp(prefix="tlt_tab_"))
    try:
        bank = make_bank(td)
        rows = TLT.read_bank_rows(bank)
        ok(len(rows) == 7)
        # the key: stable, 64-hex, distinct per row incl. the rank-1 twin that shares log, token and traj bytes
        keys = [TLT.row_key(r) for r in rows]
        ok(len(set(keys)) == 7 and all(len(k.rsplit("|", 1)[1]) == 64 for k in keys))
        tw = [r for r in rows if r["rank"] == 1][0]
        r0 = [r for r in rows if r["rank"] == 0 and r["token"] == tw["token"]][0]
        ok(tw["traj"] == r0["traj"] and TLT.traj_sha256(tw["traj"]) == TLT.traj_sha256(r0["traj"]) and TLT.row_key(tw) != TLT.row_key(r0), "twins share the traj bytes, not the key")
        ok(TLT.traj_sha256([[1.0, 2.0, 3.0]]) != TLT.traj_sha256([[1.0, 2.0, 3.0000000000000004]]), "the sha is over the exact float64 bytes")
        # save is deterministic, load round-trips floats exactly
        tp = make_table(td, rows)
        tp2 = make_table(td, rows, name="t2.json.gz")
        ok(TLT.sha256_file(tp) == TLT.sha256_file(tp2), "same content, same file sha256")
        t = TLT.load(tp)
        ok(t.sha256 == TLT.sha256_file(tp) and len(t) == 7 and t.meta["schema"] == TLT.SCHEMA and t.meta["label_name"] == "Tc20s-fixture")
        for r in rows:
            ok(t.lookup(r) == label_of(r), "lookup returns the stored label exactly (JSON float round trip)")
        ok(raises(TLT.TableRefused, TLT.load, tp, "0" * 64) is not None, "a pinned sha that is not the file's is refused")
        # lookup: absent triple -> None; same triple, other traj -> REFUSED
        ghost = dict(rows[0], token="nope")
        ok(t.lookup(ghost) is None)
        moved = dict(rows[0], traj=[[p[0] + 1e-9, p[1], p[2]] for p in rows[0]["traj"]])
        e = raises(TLT.TableRefused, t.lookup, moved)
        ok(e is not None and "REFUSED" in str(e), "a row whose own traj differs from the one the label came from is refused, never substituted")
        # build refusals: claimed sha of another traj; bad shape; non-finite; yaw branch; partial without permission
        labs = {TLT.row_triple(r): (label_of(r), None) for r in rows}
        ok(raises(TLT.TableRefused, TLT.build_table, rows, dict(labs, **{TLT.row_triple(rows[0]): (label_of(rows[0]), "0" * 64)})) is not None)
        ok(raises(TLT.TableRefused, TLT.build_table, rows, dict(labs, **{TLT.row_triple(rows[0]): (label_of(rows[0])[:19], None)})) is not None, "19 poses vs 20")
        bad = label_of(rows[0])
        bad[3][0] = float("nan")
        ok(raises(TLT.TableRefused, TLT.build_table, rows, dict(labs, **{TLT.row_triple(rows[0]): (bad, None)})) is not None, "NaN")
        wrap = label_of(rows[0])
        wrap[5][2] += 2 * 3.14159265358979
        ok(raises(TLT.TableRefused, TLT.build_table, rows, dict(labs, **{TLT.row_triple(rows[0]): (wrap, None)})) is not None, "a 2 pi yaw branch mismatch")
        part = {k: v for i, (k, v) in enumerate(labs.items()) if i != 2}
        ok(raises(TLT.TableRefused, TLT.build_table, rows, part) is not None, "partial coverage needs allow_partial")
        tab, meta, rep = TLT.build_table(rows, part, allow_partial=True)
        ok(rep["missing"] == 1 and meta["partial"] is True and len(tab) == 6)
        # a file that is not this schema
        junk = td / "junk.json.gz"
        with gzip.open(junk, "wt", encoding="utf-8") as f:
            json.dump({"meta": {"schema": "other"}, "table": {}}, f)
        ok(raises(TLT.TableRefused, TLT.load, junk) is not None)
        # CLI: identity -> verify rc 0, partial table with require_coverage 1.0 -> verify rc != 0 (e)
        cmd = [sys.executable, str(HERE / "traj_label_table.py")]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")

        def run(*args):
            return subprocess.run(cmd + list(args), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        idp = str(td / "id.json.gz")
        r1 = run("identity", "--bank-dir", bank, "--out", idp)
        ok(r1.returncode == 0, r1.stderr)
        r2 = run("verify", "--table", idp, "--bank-dir", bank)
        ok(r2.returncode == 0 and json.loads(r2.stdout)["coverage"] == 1.0 and json.loads(r2.stdout)["differs_from_raw"] == 0, r2.stdout + r2.stderr)
        lj = td / "labels.jsonl"
        lj.write_text("".join(json.dumps({"log_name": r["log_name"], "token": r["token"], "rank": r["rank"], "label": label_of(r)}) + "\n" for r in rows[:-1]), encoding="utf-8", newline="\n")
        r3 = run("build", "--bank-dir", bank, "--labels", str(lj), "--out", str(td / "b.json.gz"))
        ok(r3.returncode == 2 and "no label" in r3.stderr, "build over a bank the labels do not cover is refused")
        r4 = run("build", "--bank-dir", bank, "--labels", str(lj), "--out", str(td / "b.json.gz"), "--allow-partial", "--require-coverage", "1.0")
        ok(r4.returncode == 0, r4.stderr)
        r5 = run("verify", "--table", str(td / "b.json.gz"), "--bank-dir", bank)
        ok(r5.returncode == 1 and json.loads(r5.stdout)["refused"] and "required" in json.loads(r5.stdout)["refused"], "check-bank: below the declared coverage = rc 1 (e)")
        r6 = run("info", "--table", tp)
        ok(r6.returncode == 0 and json.loads(r6.stdout)["keys"] == 7)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_mismatch_refuses():
    """The table refuses to substitute a label computed from a different traj than the row's own bytes (the trainer's construction exits)."""
    import train as T
    td = Path(tempfile.mkdtemp(prefix="tlt_mm_"))
    try:
        bank = make_bank(td)
        rows = TLT.read_bank_rows(bank)
        # a table built from a bank whose row 3 had a slightly different traj, then handed THIS bank
        other = [dict(r) for r in rows]
        other[3] = dict(other[3], traj=[[p[0] + 0.5, p[1], p[2]] for p in other[3]["traj"]])
        tp = make_table(td, other)
        try:
            T.TargetBank(bank, None, cfg_for_bank(), synthetic=True, traj_label_table=tp)
        except SystemExit as e:
            ok("refused" in str(e).lower() and "REFUSED" in str(e) and "never substituted" in str(e), str(e))
        else:
            ok(False, "a table built from another traj must refuse the bank")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ S2 G-LIVE (a): the real loader
def chk_loader():
    import train as T
    td = Path(tempfile.mkdtemp(prefix="tlt_ld_"))
    try:
        bank = make_bank(td)
        cfg = cfg_for_bank()
        tp = make_table(td, TLT.read_bank_rows(bank))
        d0 = T.TargetBank(bank, None, cfg, synthetic=True)
        d1 = T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=None)
        ok(d0.traj_label is None and d0.traj_label_sha256 is None and len(d0) == len(d1) == 7)
        pa = T.build_parser().parse_args(["--targets", bank])
        ok(pa.traj_label_table is None, "the flag defaults OFF")
        dt = T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=tp)
        ok(dt.traj_label_sha256 == TLT.sha256_file(tp) and dt.traj_label_cov["covered"] == 7 and dt.traj_label_cov["kept_raw"] == 0 and dt.traj_label_cov["differs_from_raw"] == 7)
        n_diff = 0
        for i in range(len(dt)):
            r = dt.rows[i]
            raw, off, on = f32(r["traj"]), d0[i][3], dt[i][3]
            ok(torch.equal(off, raw) and off.dtype == torch.float32, "flag off: item[3] is the raw traj, bit for bit")
            ok(same_item(d0[i], d1[i]), "flag-off item == explicit None, every element")
            ok(torch.equal(on, f32(label_of(r))) and on.dtype == torch.float32 and on.shape == raw.shape, "flag on: item[3] is the table's label, bit for bit")
            n_diff += int(not torch.equal(on, raw))
            ok(all(torch.equal(x, y) for j, (x, y) in enumerate(zip(d0[i], dt[i])) if j != 3 and isinstance(x, torch.Tensor)), "every other slot (image, ego, goal, candidates...) is untouched")
        ok(n_diff == 7 == dt.traj_label_cov["differs_from_raw"], "the label differs from raw on every row here (the fraction measured at table build)")
        # the rank-1 twin and its rank-0 sibling share log, token AND traj bytes: each gets its OWN label
        i0 = [i for i, r in enumerate(dt.rows) if r["token"] == "tok0000" and r["rank"] == 0][0]
        i1 = [i for i, r in enumerate(dt.rows) if r["token"] == "tok0000" and r["rank"] == 1][0]
        ok(not torch.equal(dt[i0][3], dt[i1][3]) and torch.equal(dt[i0][3], f32(label_of(dt.rows[i0]))) and torch.equal(dt[i1][3], f32(label_of(dt.rows[i1]))))
        # a partial table: the covered rows are replaced, the others KEEP raw, and the coverage is reported (not refused: the table declares no requirement)
        part = make_table(td, TLT.read_bank_rows(bank), name="part.json.gz", skip_idx=(1, 4))
        dp = T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=part)
        ok(dp.traj_label_cov["covered"] == 5 and dp.traj_label_cov["kept_raw"] == 2 and abs(dp.traj_label_cov["coverage"] - 5 / 7) < 1e-12)
        for i, r in enumerate(dp.rows):
            want = f32(r["traj"]) if i in (1, 4) else f32(label_of(r))
            ok(torch.equal(dp[i][3], want), f"row {i}")
        # a partial table that DECLARES full coverage refuses the bank at construction
        req = make_table(td, TLT.read_bank_rows(bank), name="req.json.gz", skip_idx=(1,), require=1.0)
        try:
            T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=req)
        except SystemExit as e:
            ok("required" in str(e) and "refused" in str(e).lower(), str(e))
        else:
            ok(False, "required coverage not met must refuse")
        # a label with another shape than the row's traj refuses at construction
        t = TLT.load(make_table(td, TLT.read_bank_rows(bank), name="x.json.gz"))
        k0 = sorted(t.labels)[0]
        short = dict(t.labels, **{k0: t.labels[k0][:19]})
        TLT.save(td / "short.json.gz", short, {k: v for k, v in t.meta.items() if k not in ("schema", "n_keys")})
        try:
            T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=str(td / "short.json.gz"))
        except SystemExit as e:
            ok("19 poses" in str(e), str(e))
        else:
            ok(False, "a 19-pose label for a 20-pose row must refuse")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_callsites():
    """EVERY TargetBank construction in train.py hands the declared table to the loader (the --grow rebuild is the second call site; a dropped kwarg there is the 'built, tested, unreachable' class)."""
    import re
    src = (HERE / "train.py").read_text(encoding="utf-8")
    calls = [m.start() for m in re.finditer(r"\bTargetBank\(", src) if not src[max(0, m.start() - 6):m.start()].endswith("class ")]
    ok(len(calls) == 2, f"expected the 2 construction sites (train bank, --grow rebuild), found {len(calls)}")
    for c in calls:
        depth, j = 0, src.index("(", c)
        for k in range(j, len(src)):
            depth += src[k] == "("
            depth -= src[k] == ")"
            if depth == 0:
                body = src[j:k + 1]
                break
        ok("traj_label_table=a.traj_label_table or None" in body, f"a TargetBank( call at char {c} does not pass --traj-label-table")


# ------------------------------------------------------------------------------------------------ S3 G-LIVE (b), (c): the loss and the gradient
def _loss_and_grad(ds, idxs, pred_from=None):
    """fixed weights: a deterministic proposal tensor [B, M, 20, 3]; the REAL wta_loss on the loader's own item[3]."""
    from model import wta_loss
    tgt = torch.stack([ds[i][3] for i in idxs])
    g = torch.Generator().manual_seed(3)
    pred = (tgt.unsqueeze(1) + 0.5 * torch.randn(tgt.shape[0], 4, 20, 3, generator=g)) if pred_from is None else pred_from.clone()
    pred = pred.detach().requires_grad_(True)
    loss, idx, lp, ly = wta_loss(pred, tgt, yaw_wrap=True, parts=True)
    loss.backward()
    return loss.detach(), pred.grad.clone(), pred.detach()


def chk_loss():
    import train as T
    td = Path(tempfile.mkdtemp(prefix="tlt_ls_"))
    try:
        bank = make_bank(td)
        cfg = cfg_for_bank()
        rows = TLT.read_bank_rows(bank)
        tp = make_table(td, rows)
        idp = str(td / "id.json.gz")
        tab, meta, _ = TLT.build_table(rows, {TLT.row_triple(r): ([[float(x) for x in p] for p in r["traj"]], None) for r in rows}, {"label_name": "identity"}, 1.0)
        TLT.save(idp, tab, meta)
        d0 = T.TargetBank(bank, None, cfg, synthetic=True)
        d0b = T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=None)
        dt = T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=tp)
        di = T.TargetBank(bank, None, cfg, synthetic=True, traj_label_table=idp)
        idxs = list(range(7))
        # predictions are built from the RAW target in every arm, so only the TARGET differs between arms
        raw_t = torch.stack([d0[i][3] for i in idxs])
        g = torch.Generator().manual_seed(3)
        pred = raw_t.unsqueeze(1) + 0.5 * torch.randn(7, 4, 20, 3, generator=g)
        l0, g0, _ = _loss_and_grad(d0, idxs, pred)
        lb, gb, _ = _loss_and_grad(d0b, idxs, pred)
        lt, gt, _ = _loss_and_grad(dt, idxs, pred)
        li, gi, _ = _loss_and_grad(di, idxs, pred)
        # (b) with the table the loss moves; without it the difference is EXACTLY 0.0
        ok(float(lt - l0) != 0.0 and float((gt - g0).abs().max()) > 0.0, "table: loss and gradient differ from raw")
        ok(float(lb - l0) == 0.0 and torch.equal(gb, g0), "flag off: loss difference is exactly 0.0, gradient bit-identical (the control)")
        # (c) the identity table reproduces flag-off bit for bit
        ok(float(li - l0) == 0.0 and torch.equal(gi, g0), "identity table == flag off, loss and gradient bit for bit")
        # the replaced target REACHES the gradient: predictions equal to the label have an exactly zero gradient under the table, a non-zero one under the raw traj
        lab_t = torch.stack([dt[i][3] for i in idxs])
        pl = lab_t.unsqueeze(1).repeat(1, 4, 1, 1)
        lz, gz, _ = _loss_and_grad(dt, idxs, pl)
        lr_, gr, _ = _loss_and_grad(d0, idxs, pl)
        ok(float(lz) == 0.0 and float(gz.abs().max()) == 0.0, "zero control: prediction == table label -> loss 0.0, gradient exactly 0.0 under the table")
        ok(float(lr_) > 0.0 and float(gr.abs().max()) > 0.0, "and the same prediction is NOT optimal for the raw target")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ S4 the trainer end to end
def _train(td: Path, tag: str, extra=()):
    bank = str(td / "bank")
    out = td / f"out_{tag}"
    argv = [sys.executable, str(HERE / "train.py"), "--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", bank,
            "--scorer-targets", bank + "_none", "--batch", "1", "--accum", "2", "--steps", "3", "--lr", "5e-5", "--log-every", "1", "--ckpt-every-min", "999", "--out", str(out)] + list(extra)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(HERE))
    return r, out


def _params(out: Path):
    ck = torch.load(out / "model_final.pt", map_location="cpu", weights_only=False)
    return ck, {k: v for k, v in ck["model"].items()} if "model" in ck else {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}


def chk_train_e2e():
    """The REAL train.py: the startup line prints the table's sha256, the meta and run identity carry it, an IDENTITY table gives parameters bit-identical to flag-off, a real table moves them."""
    td = Path(tempfile.mkdtemp(prefix="tlt_e2e_"))
    try:
        bank = make_bank(td)
        rows = TLT.read_bank_rows(bank)
        tp = make_table(td, rows)
        tab, meta, _ = TLT.build_table(rows, {TLT.row_triple(r): ([[float(x) for x in p] for p in r["traj"]], None) for r in rows}, {"label_name": "identity"}, 1.0)
        idp = str(td / "id.json.gz")
        TLT.save(idp, tab, meta)
        r0, o0 = _train(td, "off")
        ok(r0.returncode == 0, r0.stdout[-1500:] + r0.stderr[-1500:])
        ok("TRAJ LABEL" not in r0.stdout, "flag off prints nothing about labels")
        rt, ot = _train(td, "tab", ("--traj-label-table", tp))
        ok(rt.returncode == 0, rt.stdout[-1500:] + rt.stderr[-1500:])
        sha = TLT.sha256_file(tp)
        ok("TRAJ LABEL (P5-C)" in rt.stdout and sha in rt.stdout and "7/7 rows (100.000 %)" in rt.stdout, "the startup line prints the whole table's sha256 and the coverage")
        ri, oi = _train(td, "id", ("--traj-label-table", idp))
        ok(ri.returncode == 0, ri.stdout[-1500:] + ri.stderr[-1500:])
        c0, p0 = _params(o0)
        ct, pt = _params(ot)
        ci, pi = _params(oi)
        ok(len(p0) > 10 and set(p0) == set(pt) == set(pi))
        ok(all(torch.equal(p0[k], pi[k]) for k in p0), "identity table: every parameter bit-identical to flag-off after training (c, end to end)")
        ok(any(not torch.equal(p0[k], pt[k]) for k in p0), "a real table moves the trained parameters (the label reaches the loss and the optimiser)")
        # meta / identity
        mt = ct["meta"]
        ok(mt.get("traj_label_table_sha256") == sha and mt["traj_label_coverage"]["covered"] == 7, mt)
        ok(not any(k.startswith("traj_label") for k in c0["meta"]), "flag off: no label keys in the meta")
        ck = torch.load(ot / "ckpt_last.pt", map_location="cpu", weights_only=False)
        ok(ck["identity"].get("traj_label_sha256") == sha, "the resume file carries the table's content hash")
        ok("traj_label_sha256" not in torch.load(o0 / "ckpt_last.pt", map_location="cpu", weights_only=False)["identity"], "flag off: identity unchanged")
        # a table that does not match the bank exits non-zero BEFORE any training
        other = [dict(r) for r in rows]
        other[2] = dict(other[2], traj=[[p[0] + 0.5, p[1], p[2]] for p in other[2]["traj"]])
        bp = make_table(td, other, name="bad.json.gz")
        rb, _ = _train(td, "bad", ("--traj-label-table", bp))
        ok(rb.returncode != 0 and "never substituted" in (rb.stdout + rb.stderr), "train.py refuses a table built from other trajectories")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ S5 the launch gate (G-DVB)
def _prof():
    import launch_gate as L
    return {**L.PROFILES["refe_p6"], "forbidden_flags": (), "required_flags": {}, "provisional_flags": ()}


def _argv_p6(bank: str, out: str, extra=()):
    return ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", bank, "--scorer-targets", bank + "_none",
            "--yaw-wrap", "off", "--nav-mode", "command", "--labeller", "online", "--label-backend", "synthetic", "--label-workers", "2",
            "--batch", "1", "--accum", "1", "--steps", "2", "--lr", "2e-4", "--log-every", "1", "--ckpt-every-min", "999", "--out", out,
            "--scorer-sees-trunk"] + list(extra)


def _build(td, extra, arm_b=None):
    import gate_checks as C
    import launch_gate as L
    bank = str(td / "bank")
    argv = _argv_p6(bank, str(td / "o"), extra)
    b_argv = C._set(C._drop(argv, "--steps", 1), "--epochs", "1")
    b = C.instrumented_run(b_argv, mode="build", work=td / "wb", arm=arm_b)
    L._free(b)
    return b_argv, b


def chk_gate_registry():
    import gate_checks as C
    import train as T
    from types import SimpleNamespace
    ok(C.DVB_KINDS.get("traj_label_table") == "built")
    dests = {x.dest for x in T.build_parser()._actions if x.dest != "help"}
    ok(set(C.DVB_KINDS) == dests and "traj_label_table" in dests, sorted(dests ^ set(C.DVB_KINDS)))
    dummy = SimpleNamespace(parser_unknown=[], rc="X", error=None, log="-")
    saved = C.DVB_KINDS.pop("traj_label_table")
    try:
        ev = C.check_dvb(["--targets", "x"], dummy, None, _prof())
        ok(ev["status"] == "FAIL" and "traj_label_table" in " ".join(ev["reasons"]) and "NO declared-vs-built entry" in " ".join(ev["reasons"]), ev["reasons"])
    finally:
        C.DVB_KINDS["traj_label_table"] = saved
    ok(C.check_hyg()["status"] == "PASS")


def chk_dvb_flag_set():
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="tlt_dvb_"))
    try:
        bank = make_bank(td)
        tp = make_table(td, TLT.read_bank_rows(bank))
        b_argv, b = _build(td, ("--traj-label-table", tp))
        ok(b.rc == "STOPPED_AT_BUILD", (b.rc, b.error))
        prof = {**_prof(), "expect_schedule": None}
        ev = C.check_dvb(b_argv, b, None, prof)
        rows = [r for r in ev["details"]["rows"] if r["dest"] == "traj_label_table"]
        ok(ev["status"] == "PASS" and rows and all(r["ok"] for r in rows) and rows[0]["declared"] == tp == rows[0]["built"], (ev["reasons"], rows))
        pin = TLT.sha256_file(tp)
        ev_ok = C.check_dvb(b_argv, b, None, {**prof, "traj_label_sha256": pin})
        ok(ev_ok["status"] == "PASS" and any(r["how"].startswith("the table file's sha256") and r["ok"] for r in ev_ok["details"]["rows"]), ev_ok["reasons"])
        ev_bad = C.check_dvb(b_argv, b, None, {**prof, "traj_label_sha256": "0" * 64})
        ok(ev_bad["status"] == "FAIL" and "--traj-label-table" in " ".join(ev_bad["reasons"]), ev_bad["reasons"])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_dvb_red_not_passed():
    """RED arm `traj_label_not_passed` (d): the flag is declared but the loader is built WITHOUT the table -> declared != built."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="tlt_dvbr_"))
    try:
        bank = make_bank(td)
        tp = make_table(td, TLT.read_bank_rows(bank))
        b_argv, b = _build(td, ("--traj-label-table", tp), arm_b="traj_label_not_passed")
        ev = C.check_dvb(b_argv, b, None, {**_prof(), "expect_schedule": None}, arm="traj_label_not_passed")
        ok(ev["status"] == "FAIL" and "--traj-label-table: declared" in " ".join(ev["reasons"]) and "built None" in " ".join(ev["reasons"]), ev["reasons"])
        # and the control: the same build without the arm passes (so the RED is the arm's doing)
        b2_argv, b2 = _build(td, ("--traj-label-table", tp))
        ok(C.check_dvb(b2_argv, b2, None, {**_prof(), "expect_schedule": None})["status"] == "PASS")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_flag_off_dvb_silent():
    """No flag -> the G-DVB row reads declared None == built None for every construction (the default is the unchanged trainer)."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="tlt_dvb0_"))
    try:
        make_bank(td)
        b_argv, b = _build(td, ())
        ev = C.check_dvb(b_argv, b, None, {**_prof(), "expect_schedule": None})
        rows = [r for r in ev["details"]["rows"] if r["dest"] == "traj_label_table"]
        ok(ev["status"] == "PASS" and rows and all(r["declared"] is None and r["built"] is None and r["ok"] for r in rows), (ev["reasons"], rows))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_no_scipy():
    """THOR VENV: no scipy.  This test and the table tool import none; every subprocess.run with text=True carries an explicit encoding= (the subprocess-encoding guard)."""
    import re
    for fn in ("traj_label_table.py", "test_traj_label_table.py"):
        src = (HERE / fn).read_text(encoding="utf-8")
        ok(not re.search(r"^\s*(import|from)\s+scipy", src, re.M), f"{fn} imports scipy")
        for m in re.finditer(r"subprocess\.run\(", src):
            depth, j = 0, m.end() - 1
            for k in range(j, len(src)):
                depth += src[k] == "("
                depth -= src[k] == ")"
                if depth == 0:
                    break
            call = src[j:k + 1]
            ok("text=True" not in call or "encoding=" in call, f"{fn}: a subprocess.run with text=True and no encoding= at char {m.start()}")


CHECKS = {"table": chk_table, "mismatch_refuses": chk_mismatch_refuses, "loader": chk_loader, "callsites": chk_callsites, "loss": chk_loss, "train_e2e": chk_train_e2e,
          "gate_registry": chk_gate_registry, "dvb_flag_set": chk_dvb_flag_set, "dvb_red_not_passed": chk_dvb_red_not_passed, "flag_off_dvb_silent": chk_flag_off_dvb_silent,
          "no_scipy": chk_no_scipy}

# (file, old, new, the check that must go RED).  Each DELETES / neutralises ONE emitting line of a COPY of this tree.
LOOKUP_LINE = '        tgt_t = torch.tensor(self.traj_label[i] if (self.traj_label is not None and i in self.traj_label) else r["traj"], dtype=torch.float32)\n'
MUTATIONS = [
    ("train.py", LOOKUP_LINE, '        tgt_t = torch.tensor(r["traj"], dtype=torch.float32)\n', "loader"),                                  # label_lookup_removed: __getitem__ reads the raw traj again
    ("train.py", LOOKUP_LINE, '        tgt_t = torch.tensor(r["traj"], dtype=torch.float32)\n', "loss"),
    ("train.py", "                    traj_label_table=a.traj_label_table or None,              # P5-C (None = the unchanged loader)\n", "", "callsites"),   # label_table_not_passed at the train bank
    ("train.py", "                       traj_label_table=a.traj_label_table or None,\n", "", "callsites"),                                          # ... at the --grow rebuild
    ("train.py", "                    traj_label_table=a.traj_label_table or None,              # P5-C (None = the unchanged loader)\n",
     "                    traj_label_table=None,\n", "dvb_flag_set"),                                                                                # the call site passes None: G-DVB reads declared != built
    ("train.py", '            meta["traj_label_table_sha256"] = ds.traj_label_sha256\n', "", "train_e2e"),                                         # the checkpoint no longer says so
    ("train.py", '        ident["traj_label_sha256"] = getattr(ds, "traj_label_sha256", None)\n', "", "train_e2e"),                                 # the run identity no longer carries it
    ("train.py", "                res = TLT.apply_to_rows(tlt, self.rows)\n", "                res = {'labels': {}, 'coverage': {'covered': 0, 'rows': len(self.rows), 'coverage': 0.0, 'kept_raw': len(self.rows), 'differs_from_raw': 0, 'table_keys': len(tlt)}}\n", "loader"),   # the audit + lookup never run
    ("traj_label_table.py", "        if sha not in shas:\n", "        if sha not in shas:\n            return self.labels[f\"{triple}|{sorted(shas)[0]}\"]\n", "table"),     # silent substitution on a traj-sha mismatch
    ("traj_label_table.py", "    if worst >= MAX_DYAW:\n", "    if False:\n", "table"),                                                                                 # the yaw-branch check
    ("gate_checks.py", '    "traj_label_table": "built",\n', "", "gate_registry"),                                                                      # the G-DVB registry entry
    ("gate_checks.py", "    rows += traj_label_dvb_rows(a, b, profile)\n", "", "dvb_flag_set"),                                                        # the G-DVB row is not wired in
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="tltmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:60]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_traj_label_table.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
            print(f"  mutation RED as required: {fn} [{old.strip()[:56]}] -> {check}", flush=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        run_mutations()
    print(f"ZZTRAJLABEL_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
