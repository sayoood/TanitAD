"""REHEARSAL of code/recover_read.py's PRODUCTION path on STAND-INS (not a result; nothing here is a RECOVER-FT number).

The six box-2 checkpoints do not exist yet, so the file controls (KP1 KR1 KR2 KR3 KMARK KC1 KC3 KA1 KFZ1-4 KK1 KK2 KB1 KF), the array handling and the whole rule set are exercised on:
  * the 15 rollouts' directories = EXISTING rollouts relabelled as the 15 tags (R / C0 / C0REP := KAPPA-T1's SREP_DAGOACC_s, KA1R := FEAS2's SREP2 rerun, FZR / FZC0 := FRAMEFREEZE's FZY_DAGOACC_s,
    RC / C0C := FEAS2's KBOX_DAGOACC_s), headers patched only in `tag` and `ckpt`;
  * a FAKE pulled-run directory: random-byte checkpoints, DAG-OACC's summary.json plus the recover fields, the box-2 argv files, the R_0 gate selfcheck (bank sha patched for s1), box-2 style markers.
Expected: every file control PASSES (the stand-ins are built to satisfy them), R - C0 is exactly 0, the label is NOT MOVED, the composite NOT DOMINANT.  A control that fails here is a bug in the reader, found before 04:00.
Usage: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/rehearse_production_path.py        (scratch: F:/TanitAD-scratch/recover_read/rehearsal; about 5 minutes)
"""
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
DL = f"{ART}/dagland"
BASE = "F:/TanitAD-scratch/recover_read/rehearsal"
OUT, PULLED, RAW = f"{BASE}/out", f"{BASE}/v8f-recover", f"{BASE}/raw"


def load():
    spec = importlib.util.spec_from_file_location("recover_read", f"{PK}/code/recover_read.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def standin(row):
    s, role = row["seed"], row["role"]
    k, f, z = "F:/TanitAD-scratch/kappa_t1/out", "F:/TanitAD-scratch/feas2_t1/out", "F:/TanitAD-scratch/framefreeze/out"
    if role in ("R", "C0", "C0REP"):
        return f"{k}/SREP_DAGOACC_s{s}", f"SREP_DAGOACC_s{s}", f"{DL}/kappa_t1/raw/arrays/SREP_DAGOACC_s{s}.npz"
    if role == "KA1R":
        return f"{f}/SREP2_DAGOACC_s0", "SREP2_DAGOACC_s0", None
    if role in ("FZR", "FZC0"):
        return f"{z}/FZY_DAGOACC_s{s}", f"FZY_DAGOACC_s{s}", f"{DL}/framefreeze/raw/arrays/FZY_DAGOACC_s{s}.npz"
    return f"{f}/KBOX_DAGOACC_s{s}", f"KBOX_DAGOACC_s{s}", f"{DL}/feas2_t1/raw/arrays/KBOX_DAGOACC_s{s}.npz"


def main():
    R = load()
    os.makedirs(BASE, exist_ok=True)
    for d in (OUT, PULLED, RAW):
        if os.path.exists(d):
            shutil.move(d, f"{d}_aside_{int(os.path.getmtime(d))}")      # never delete: move aside
        os.makedirs(d)
    os.makedirs(f"{RAW}/arrays")
    table = R.run_table()
    for row in table:
        src, orig, arr = standin(row)
        tag = row["tag"]
        dst = f"{OUT}/{tag}"
        shutil.copytree(src, dst)
        os.rename(f"{dst}/dump_{orig}", f"{dst}/dump_{tag}")
        os.rename(f"{dst}/ticks/ticks_{orig}.npz", f"{dst}/ticks/ticks_{tag}.npz")
        h = json.load(open(f"{dst}/t1v8f_{orig}.json", encoding="utf-8"))
        os.remove(f"{dst}/t1v8f_{orig}.json")
        h["t1v8f"]["tag"], h["t1v8f"]["ckpt"] = tag, f"{PULLED}/{row['ckpt_run']}/ckpt_last.pt"
        json.dump(h, open(f"{dst}/t1v8f_{tag}.json", "w", encoding="utf-8", newline="\n"))
        json.dump({"tag": tag, "stand_in_for": orig}, open(f"{OUT}/done_{tag}.txt", "w", encoding="utf-8", newline="\n"))
        if arr and os.path.exists(arr):
            shutil.copy(arr, f"{RAW}/arrays/{tag}.npz")
    rnd = random.Random(7)
    sc = json.load(open(f"{DL}/recover_ft_launch/evidence/R_0/selfcheck.json", encoding="utf-8"))
    os.makedirs(f"{PULLED}/recft_markers")
    for s in (0, 1):
        ref = json.load(open(f"{ART}/v8f-dagoacc/M2b_s{s}_dagoacc/summary.json", encoding="utf-8"))
        for base, tg in (("R", f"R_{s}"), ("C0", f"C0_{s}"), ("C0rep", f"C0rep_{s}")):
            d = f"{PULLED}/M2b_s{s}_recft_{base}"
            os.makedirs(d)
            blob = bytes(rnd.getrandbits(8) for _ in range(4096))
            open(f"{d}/ckpt_last.pt", "wb").write(blob)
            summ = dict(ref, recover_p=0.3 if base == "R" else 0.0, recover_n=90 if base == "R" else 0, recover_eligible=300 if base == "R" else 300)
            json.dump(summ, open(f"{d}/summary.json", "w", encoding="utf-8", newline="\n"))
            toks = open(f"{DL}/recover_ft_launch/box2_files/argv/{tg}.train.txt", encoding="utf-8").read().split()
            json.dump({"argv": toks, "sha": "stand-in"}, open(f"{d}/argv.json", "w", encoding="utf-8", newline="\n"))
            sc2 = json.loads(json.dumps(sc))
            sc2["info"]["recover"]["bank_sha256"] = R.BANK_SHA[s]
            json.dump(sc2, open(f"{d}/gate_selfcheck.json", "w", encoding="utf-8", newline="\n"))
            sha = hashlib.sha256(blob).hexdigest()
            open(f"{d}/SHA256SUMS.txt", "w", encoding="utf-8", newline="\n").write(f"{sha} *ckpt_last.pt\n")
            json.dump({"evidence": {"ckpt_sha256": sha}}, open(f"{PULLED}/recft_markers/RUN_{tg}.done", "w", encoding="utf-8", newline="\n"))
    open(f"{PULLED}/recft_markers/CHAIN_RECOVER.done", "w").write("stand-in\n")
    rc = R.main(["--mode", "production", "--out-dir", OUT, "--raw", RAW, "--recover-dir", PULLED])
    res = json.load(open(f"{RAW}/RECOVER_READ.json", encoding="utf-8"))
    print("rc", rc, "refused", res.get("refused"), "controls_failed", res.get("controls_failed"))
    print({q: v.get("PASS") for q, v in res["controls"].items()})
    ok = bool(rc == 0 and res.get("refused") is None and res["label"] == "NOT MOVED" and res["composite"]["status"] == "NOT DOMINANT" and all(v["PASS"] for v in res["controls"].values()))
    print("REHEARSAL", "OK" if ok else "FAILED")
    return 0 if ok else 1


def negatives():
    """After a rehearsal built the stand-ins: tamper ONE thing at a time (reverted after each) and require the read to REFUSE (rc 3, 'refused' in the JSON, the named control failed)."""
    R = load()
    cases = []
    def tamper_file(path, fn):
        def run():
            old = open(path, "rb").read()
            try:
                fn(path)
                return R.main(["--mode", "production", "--out-dir", OUT, "--raw", f"{BASE}/raw_neg", "--recover-dir", PULLED])
            finally:
                open(path, "wb").write(old)
        return run
    def edit_json(key_path, value):
        def fn(path):
            d = json.load(open(path, encoding="utf-8"))
            cur = d
            for k in key_path[:-1]:
                cur = cur[k]
            cur[key_path[-1]] = value
            json.dump(d, open(path, "w", encoding="utf-8", newline="\n"))
        return fn
    def copy_over(src):
        return lambda path: open(path, "wb").write(open(src, "rb").read())
    os.makedirs(f"{BASE}/raw_neg", exist_ok=True)
    cases.append(("KR1: C0 selected one sample", f"{PULLED}/M2b_s0_recft_C0/summary.json", edit_json(["recover_n"], 1), "KR1_selected_over_eligible_in_band_for_R_zero_for_C0"))
    cases.append(("KP1: window list differs", f"{PULLED}/M2b_s1_recft_R/summary.json", edit_json(["train_window_list_sha256"], "0" * 64), "KP1_parity_train_and_window_list_sha_equal_DAGOACC"))
    cases.append(("KR2: C0 ckpt identical to R", f"{PULLED}/M2b_s0_recft_C0/ckpt_last.pt", copy_over(f"{PULLED}/M2b_s0_recft_R/ckpt_last.pt"), "KR2_R_ckpt_differs_from_C0_and_C0_recipe_is_R_minus_the_lever"))
    cases.append(("KR3: overlap with eval", f"{PULLED}/M2b_s1_recft_R/gate_selfcheck.json", edit_json(["info", "recover", "kb3", "overlap_with_eval"], 3), "KR3_table_build_summary_exists_all_records_TRAIN"))
    cases.append(("KC1: RC header lost kbox", f"{OUT}/RC_s1/t1v8f_RC_s1.json", lambda p: (lambda d: (d["t1v8f"].pop("feasible"), json.dump(d, open(p, "w", encoding="utf-8", newline="\n"))))(json.load(open(p, encoding="utf-8"))), "KC1_flags_reached_the_runs"))
    cases.append(("KMARK: box-2 marker sha differs", f"{PULLED}/recft_markers/RUN_R_1.done", lambda p: open(p, "w").write(json.dumps({"evidence": {"ckpt_sha256": "1" * 64}})), "KMARK_box2_marker_sha_equals_pulled_ckpt"))
    bad = []
    for name, path, fn, ctl in cases:
        rc = tamper_file(path, fn)()
        res = json.load(open(f"{BASE}/raw_neg/RECOVER_READ.json", encoding="utf-8"))
        got = ctl in (res.get("controls_failed") or [])
        print("NEGATIVE", name, "-> rc", rc, "refused", bool(res.get("refused")), "named control failed", got)
        if not (rc == 3 and res.get("refused") and got):
            bad.append(name)
    # a failed KFZ control refuses the frame-freeze contrast ONLY: the read still finishes (rc 0) and says so
    path = f"{OUT}/FZR_s0/t1v8f_FZR_s0.json"
    old = open(path, "rb").read()
    try:
        d = json.loads(old.decode("utf-8"))
        d["t1v8f"]["frame_freeze"]["max_row_dev"] = 0.5
        json.dump(d, open(path, "w", encoding="utf-8", newline="\n"))
        rc = R.main(["--mode", "production", "--out-dir", OUT, "--raw", f"{BASE}/raw_neg", "--recover-dir", PULLED])
    finally:
        open(path, "wb").write(old)
    res = json.load(open(f"{BASE}/raw_neg/RECOVER_READ.json", encoding="utf-8"))
    kfz = rc == 0 and res.get("refused") is None and "REFUSED" in str(res["frame_freeze"].get("refused")) and "KFZ1_production_receipt_every_item_FZR_FZC0" in res["controls_failed"] and res["composite"]["status"] == "NOT DOMINANT"
    print("NEGATIVE KFZ1: FZR receipt max_row_dev 0.5 -> rc", rc, "freeze contrast refused, read finished", kfz)
    if not kfz:
        bad.append("KFZ1 scoped refusal")
    print("NEGATIVES", "OK" if not bad else f"FAILED {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(negatives() if "--negative" in sys.argv else main())
