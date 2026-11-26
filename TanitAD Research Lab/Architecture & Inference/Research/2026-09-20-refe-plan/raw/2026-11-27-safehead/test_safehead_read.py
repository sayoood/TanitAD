#!/usr/bin/env python3
"""Tests of safehead_read.py (PREREG_SAFEHEAD.md 4-6) on a SYNTHETIC anchor (table shards, banked dump, seam) and a synthetic sealed navtest s-cache.  CPU only.

    python test_safehead_read.py --refe <PK/refe> --eval-dir <PK/eval>       -> ZZSHREAD_PASS <n>
"""
from __future__ import annotations

import argparse
import atexit
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--refe", default=str(HERE.parent.parent / "refe"))
ap.add_argument("--eval-dir", default=str(HERE.parent.parent / "eval"))
A = ap.parse_args()
sys.path.insert(0, str(Path(A.refe)))
sys.path.insert(0, str(HERE))
import safehead_data as D  # noqa: E402
import safehead_read as SR  # noqa: E402

N = 0
TMP = Path(tempfile.mkdtemp(prefix="shread_"))
atexit.register(lambda: shutil.rmtree(TMP, ignore_errors=True))


def ok(msg):
    global N
    N += 1
    print(f"ok {N:02d} {msg}", flush=True)


def world(root: Path, n_logs=6, per_log=12, seed=0, leak=False):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    U = rng.normal(0, 1, (6, 256)).astype(np.float32) / 16.0
    W0 = (0.6 * U + rng.normal(0, 0.04, (6, 256))).astype(np.float32)
    b0 = np.array([2.0, 2.0, 0.0, 1.5, 1.0, 2.0], np.float32)
    toks, logs, tabs, S, LG, PR = [], [], [], [], [], []
    for li in range(n_logs):
        for _ in range(per_log):
            s = rng.normal(0, 1, (64, 256)).astype(np.float32)
            z = s @ U.T * 3
            lab = {nm: (rng.random(64) < 1 / (1 + np.exp(-(z[:, r] + 1.0)))).astype(float) for r, nm in ((0, "nc"), (1, "dac"), (3, "ttc"), (5, "ddc"))}
            lab["ep"] = np.clip(1 / (1 + np.exp(-z[:, 2])), 0, 1)
            lab["c"] = (rng.random(64) < 0.9).astype(float)
            lab["pdms"] = lab["nc"] * lab["dac"] * (5 * lab["ep"] + 5 * lab["ttc"] + 2 * lab["c"]) / 12
            toks.append(f"{len(toks):016x}")
            logs.append(f"2021.05.0{li}.00.00.00_veh-1_nt{li}")
            tabs.append(lab)
            S.append(s)
            LG.append((s @ W0.T + b0).astype(np.float32))
            PR.append(rng.normal(0, 3, (64, 8, 3)).astype(np.float32))
    LG = np.stack(LG)
    pick = D.select(LG)
    n = len(toks)
    half = n // 2
    for sh, sl in ((0, slice(0, half)), (1, slice(half, n))):
        with open(root / f"full_C1_s{sh}of2.jsonl", "w", encoding="utf-8") as fh:
            for i in range(n)[sl]:
                fh.write(json.dumps({"token": toks[i], "log": logs[i], "pick": int(pick[i]), "tr": {k: [float(x) for x in v] for k, v in tabs[i].items()}}) + "\n")
    PR = np.stack(PR)
    fp = np.array([f"fp{i}" for i in range(n)])
    np.savez(root / "props.npz", token=np.array(toks), fingerprint=fp, proposals=PR, logits=LG, pick=pick.astype(np.int64))
    np.savez(root / "seam.npz", token=np.array(toks), fingerprint=fp, poses=PR[np.arange(n), pick], sampling=np.array([0.5, 8.0]))
    nc = root / "ntcache"
    nc.mkdir()
    with open(nc / "index.csv", "w", encoding="utf-8") as fh:
        fh.write("token,log,split,file\n")
        for i in range(n):
            np.savez(nc / f"{toks[i]}.npz", s=S[i], logits=LG[i], pick=np.int64(pick[i]), props=np.zeros((64, 20, 3), np.float32))
            fh.write(f"{toks[i]},{logs[i]},navtest,{toks[i]}.npz\n")
    D.seal_index(nc / "index.csv", root / "run" / "nt_sealed.csv")
    tr_logs = ["2021.07.07.00.00.00_veh-7_tr0"] + ([logs[0]] if leak else [])
    with open(root / "run" / "train_sealed.csv", "w", encoding="utf-8") as fh:
        fh.write("token,log,split,file,sha256\n")
        for j, lg in enumerate(tr_logs + ["2021.06.09.14.50.36_veh-26_00832_00905"]):
            fh.write(f"{'f' * 15}{j},{lg},{'heldout' if lg.startswith('2021.06.09') else 'train'},x.npz,{'0' * 64}\n")
    (root / "sft.txt").write_text("2021.06.09.14.50.36_veh-26_00832_00905\n", encoding="utf-8")
    return {"root": root, "W0": W0, "b0": b0, "U": U, "tok": toks, "logs": logs, "tabs": tabs, "pick": pick, "S": np.stack(S), "LG": LG}


def fake_run(w, d: Path, arm: str, W, b, step: int, *, smoke=False, red_degrades=True):
    d.mkdir(parents=True)
    seed, perm = {"s0": (0, -1), "s1": (1, -1), "red": (0, 20261127)}[arm]
    idx_sha = D.sha256_file(w["root"] / "run" / "train_sealed.csv")
    shas = {}
    for st, (Ws, bs) in ((0, (w["W0"], w["b0"])), (step, (W, b))):
        shas[st] = D.save_head(d / f"head_step{st:05d}.npz", W=Ws, b=bs, W0=w["W0"], b0=w["b0"],
                               meta={"rows": [0, 1, 3, 5], "seed": seed, "init_sha256": SR.C1_SHA256, "index_sha256": idx_sha, "smoke": smoke, "permute_labels_seed": perm, "step": st})
    (d / "heads.sha256").write_text("".join(f"{v}  head_step{k:05d}.npz\n" for k, v in shas.items()), encoding="utf-8")
    base = {"safety_miss_rate": 0.05, "auc_nc": 0.9, "auc_dac": 0.9, "auc_ttc": 0.9}
    fin = ({"safety_miss_rate": 0.12, "auc_nc": 0.7, "auc_dac": 0.7, "auc_ttc": 0.7} if red_degrades else dict(base)) if arm == "red" else         {"safety_miss_rate": 0.04, "auc_nc": .92, "auc_dac": .92, "auc_ttc": .92}
    (d / "heldout_select.json").write_text(json.dumps({"chosen_step": step, "per_step": {"0": base, str(max(step, 2000) if arm == "red" else step): fin}}), encoding="utf-8")
    (d / "run.json").write_text(json.dumps({"token": {"profile": f"refe_safehead_{arm}", "token_sha256": "ab" * 32}}), encoding="utf-8")
    (d / "metrics.jsonl").write_text("".join(json.dumps({"step": i, "grad_abs": {"row0": 1.0, "row1": 1.0, "row3": 1.0, "row5": 1.0},
                                                          "frozen_grad_abs": {"row2": 0.0, "row4": 0.0}}) + "\n" for i in range(1, 2001)), encoding="utf-8")
    return d


def trained(w, alpha=0.6):
    W, b = w["W0"].copy(), w["b0"].copy()
    W[[0, 1, 3]] += alpha * w["U"][[0, 1, 3]]
    return W, b


def read(w, runs, out, extra=()):
    r = w["root"]
    a = ["read", "--table", str(r / "full_C1_s*of2.jsonl"), "--no-sha", "--proposals", str(r / "props.npz"), "--seam", str(r / "seam.npz"),
         "--navtest-index", str(r / "run" / "nt_sealed.csv"), "--train-index", str(r / "run" / "train_sealed.csv"), "--heldout-logs", str(r / "sft.txt"),
         "--run-s0", str(runs["s0"]), "--run-s1", str(runs["s1"]), "--run-red", str(runs["red"]), "--eval-dir", A.eval_dir, "--refe", A.refe, "--out", str(out),
         "--n-tokens", str(len(w["tok"])), "--n-logs", str(len(set(w["logs"]))),
         "--c1-pdms", repr(float(np.mean([w["tabs"][i]["pdms"][w["pick"][i]] for i in range(len(w["tok"]))]) * 100)), *extra]
    rc = SR.main(a)
    return rc, json.loads((Path(out) / "READ_SAFEHEAD.json").read_text())


def main() -> int:
    w = world(TMP / "w")
    n, ar = len(w["tok"]), np.arange(len(w["tok"]))
    W, b = trained(w)
    # 1. CTRL-ID + an untrained "chosen" head: every control PASSES and the arm reads Delta == 0 exactly -> NULL
    runs = {k: fake_run(w, TMP / f"id_{k}", k, w["W0"], w["b0"], 250) for k in ("s0", "s1", "red")}
    rc, rep = read(w, runs, TMP / "o_id")
    c = rep["controls"]
    assert all(v["pass"] for v in c.values()), {k: v for k, v in c.items() if not v["pass"]}
    assert c["CTRL-ID"]["logits_bitexact"] and c["CTRL-ID"]["pick_agree"] == n and c["CTRL-ID"]["per_token_pdms_exact"]
    assert rep["M1_M3"]["s0"]["pdms"] == {"point": 0.0, "lo": 0.0, "hi": 0.0} and rep["outcome_provisional"]["outcome"] == "NULL"
    ok("CTRL-ID through the read: logits bit-exact, picks n/n, per-token PDMS exact; an unchanged head reads Delta 0 [0, 0] -> NULL")
    # 2. a trained head: the point and CI equal an independent computation on the same draws
    runs = {"s0": fake_run(w, TMP / "t_s0", "s0", W, b, 500), "s1": fake_run(w, TMP / "t_s1", "s1", *trained(w, 0.5), 500),
            "red": fake_run(w, TMP / "t_red", "red", w["W0"], w["b0"], 0)}
    rc, rep = read(w, runs, TMP / "o_t")
    S32 = w["S"]
    k = D.select(D.delta_logits(w["LG"], S32, W, b, w["W0"], w["b0"]))
    pdms = np.array([w["tabs"][i]["pdms"][k[i]] for i in range(n)]) * 100
    pc1 = np.array([w["tabs"][i]["pdms"][w["pick"][i]] for i in range(n)]) * 100
    sys.path.insert(0, str(Path(A.eval_dir) / "p5c"))
    import p5c_read
    bt = p5c_read.Boot(np.array(w["logs"]))
    ref = bt.ci(pdms - pc1)
    got = rep["M1_M3"]["s0"]["pdms"]
    assert abs(got["point"] - ref["point"]) < 1e-12 and abs(got["lo"] - ref["lo"]) < 1e-12 and abs(got["hi"] - ref["hi"]) < 1e-12, (got, ref)
    assert rep["M1_M3"]["s0"]["switch_rate"] == float((k != w["pick"]).mean()) and rep["M1_M3"]["s0"]["direct_form_pick_agree"] >= n - 2
    sm = np.load(TMP / "o_t" / "seams" / "seam_SH0.npz")
    sc = np.load(TMP / "o_t" / "seams" / "seam_C1pick.npz")
    props = np.load(w["root"] / "props.npz")["proposals"]
    assert np.array_equal(sm["poses"], props[ar, k]) and np.array_equal(sc["poses"], np.load(w["root"] / "seam.npz")["poses"])
    ok(f"a trained head: Delta {got['point']:+.3f} [{got['lo']:+.3f}, {got['hi']:+.3f}] == an independent p5c_read.Boot computation; seams = proposals[pick]")
    # 3. CTRL-RED reads the RED run's held-out degradation; a RED run that does not degrade FAILS it
    assert rep["controls"]["CTRL-RED"]["pass"]
    runs_bad = dict(runs, red=fake_run(w, TMP / "t_red_flat", "red", w["W0"], w["b0"], 0, red_degrades=False))
    rc, rep2 = read(w, runs_bad, TMP / "o_red")
    assert not rep2["controls"]["CTRL-RED"]["pass"] and rep2["outcome_provisional"]["outcome"] == "INVALID"
    ok("CTRL-RED: a degrading RED run PASSES, a RED run that does not degrade held-out selection FAILS -> INVALID")
    # 4. a smoke head, a navtest cache that is not the banked forward, and a leak each make the read INVALID
    runs_smoke = dict(runs, s0=fake_run(w, TMP / "sm_s0", "s0", W, b, 500, smoke=True))
    rc, rep3 = read(w, runs_smoke, TMP / "o_sm")
    assert not rep3["controls"]["CTRL-SHA"]["pass"] and rep3["outcome_provisional"]["outcome"] == "INVALID"
    w2 = world(TMP / "w_leak", leak=True)
    runs2 = {"s0": fake_run(w2, TMP / "l_s0", "s0", *trained(w2), 500), "s1": fake_run(w2, TMP / "l_s1", "s1", *trained(w2), 500),
             "red": fake_run(w2, TMP / "l_red", "red", w2["W0"], w2["b0"], 0)}
    rc, rep4 = read(w2, runs2, TMP / "o_leak")
    assert not rep4["controls"]["CTRL-LEAK"]["pass"] and rep4["outcome_provisional"]["outcome"] == "INVALID"
    f0 = sorted((w["root"] / "ntcache").glob("*.npz"))[0]
    zz = dict(np.load(f0))
    np.savez(f0, **{**zz, "logits": zz["logits"] + 1e-3})
    D.seal_index(w["root"] / "ntcache" / "index.csv", w["root"] / "run" / "nt_sealed.csv")
    rc, rep5 = read(w, runs, TMP / "o_det")
    assert not rep5["controls"]["CTRL-DET-NT"]["pass"] and rep5["outcome_provisional"]["outcome"] == "INVALID"
    np.savez(f0, **zz)
    D.seal_index(w["root"] / "ntcache" / "index.csv", w["root"] / "run" / "nt_sealed.csv")
    ok("INVALID on a smoke head (CTRL-SHA), a navtrain log that is a navtest log (CTRL-LEAK), cached navtest logits 1e-3 off the banked dump (CTRL-DET-NT)")
    # 5. the committed reading, mechanically
    def M(d0, d1, sub0=None, sub1=None, abs0=93.0, abs1=93.0):
        s0 = {k: {"point": 0.1, "lo": -0.1, "hi": 0.3} for k in SR.SUBS}
        s0.update(sub0 or {})
        s1 = {k: {"point": 0.1, "lo": -0.1, "hi": 0.3} for k in SR.SUBS}
        s1.update(sub1 or {})
        return {"s0": {"pdms": d0, "sub": s0, "pdms_point_abs": abs0}, "s1": {"pdms": d1, "sub": s1, "pdms_point_abs": abs1}}
    good = {"point": 0.5, "lo": 0.2, "hi": 0.8}
    assert SR.outcome(M(good, good), True, 500)["outcome"] == "ADOPT"
    assert SR.outcome(M({"point": 0.2, "lo": 0.05, "hi": 0.35}, good), True, 500)["outcome"] == "SEPARATED-SMALL"          # P2: 0.2 < 2 x 0.1158
    assert SR.outcome(M(good, {"point": -0.1, "lo": -0.4, "hi": 0.2}), True, 500)["outcome"] == "SEPARATED-SMALL"          # R fails
    assert SR.outcome(M(good, good, abs0=93.5, abs1=93.0), True, 500)["outcome"] == "SEPARATED-SMALL"                     # F = 0.5 -> P2 bar 1.0 > 0.5
    assert SR.outcome(M({"point": 0.5, "lo": 0.2, "hi": 0.8}, good, abs0=93.5, abs1=93.0), True, 500)["bars"]["P2"] is False
    assert SR.outcome(M({"point": 0.1, "lo": -0.2, "hi": 0.4}, good), True, 500)["outcome"] == "NULL"
    assert SR.outcome(M({"point": -0.5, "lo": -0.8, "hi": -0.2}, good), True, 500)["outcome"] == "ADVERSE"
    o = SR.outcome(M(good, good, sub0={"ttc": {"point": -0.4, "lo": -0.7, "hi": -0.1}}), True, 500)
    assert o["outcome"] == "ADVERSE" and o["bars"]["adverse_s0"] == ["ttc"]
    assert SR.outcome(M(good, good, sub0={"ddc": {"point": -0.4, "lo": -0.7, "hi": -0.1}}), True, 500)["outcome"] == "ADOPT"   # DDC is reported only
    assert SR.outcome(M(good, good), False, 500)["outcome"] == "INVALID" and SR.outcome(M(good, good), True, 0)["outcome"] == "NULL-HELDOUT"
    ok("the committed reading: ADOPT / SEPARATED-SMALL (P2, R) / NULL / ADVERSE (PDMS, a judged sub-score; DDC not judged) / INVALID / NULL-HELDOUT")
    # 6. confirm: CTRL-T2 against the table at the new picks; missing families -> INCOMPLETE
    pk = np.load(TMP / "o_t" / "seams" / "picks.npz")
    R, _, _ = SR.load_table(str(w["root"] / "full_C1_s*of2.jsonl"), check_sha=False)
    rows = TMP / "pass_rows.jsonl"
    with open(rows, "w", encoding="utf-8") as fh:
        for i, t in enumerate(pk["token"].astype(str)):
            fh.write(json.dumps({"token": t, "arms": {a_: {"tracked": {s: R[t]["tr"][s][int(pk[key][i])] for s in SR.SUBS + ("pdms",)}}
                                                    for a_, key in (("SH0", "SH0"), ("SH1", "SH1"), ("C1pick", "C1"))}}) + "\n")
    fam = TMP / "fam.json"
    fam.write_text("{}", encoding="utf-8")
    base = ["confirm", "--table", str(w["root"] / "full_C1_s*of2.jsonl"), "--read", str(TMP / "o_t" / "READ_SAFEHEAD.json"), "--pass-rows", str(rows), "--out", str(TMP / "o_t")]
    SR.main(base + ["--families", str(fam), "--tactical", str(fam), "--strategic", str(fam)])
    v = json.loads((TMP / "o_t" / "VERDICT_SAFEHEAD.json").read_text())
    assert v["controls"]["CTRL-T2"]["pass"] and v["outcome"]["outcome"] == rep["outcome_provisional"]["outcome"]
    SR.main(base + ["--families", str(fam)])
    assert json.loads((TMP / "o_t" / "VERDICT_SAFEHEAD.json").read_text())["outcome"]["outcome"] == "INCOMPLETE"
    lines = rows.read_text().splitlines()
    r0 = json.loads(lines[0])
    r0["arms"]["SH0"]["tracked"]["pdms"] += 0.01
    rows.write_text("\n".join([json.dumps(r0)] + lines[1:]) + "\n")
    SR.main(base + ["--families", str(fam), "--tactical", str(fam), "--strategic", str(fam)])
    v = json.loads((TMP / "o_t" / "VERDICT_SAFEHEAD.json").read_text())
    assert not v["controls"]["CTRL-T2"]["pass"] and v["outcome"]["outcome"] == "INVALID"
    ok("confirm: CTRL-T2 PASSES on harness rows equal to the table, FAILS (INVALID) on a 0.01 PDMS difference; a missing family -> INCOMPLETE")
    print(f"ZZSHREAD_PASS {N}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                                                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ZZSHREAD_FAIL {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
