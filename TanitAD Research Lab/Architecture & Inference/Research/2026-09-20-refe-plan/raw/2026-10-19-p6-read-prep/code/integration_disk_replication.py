"""DISK-PATH REPLICATION (CPU, no GPU, no pod): the EP-RELIAB composite is written to disk as a CANDIDATE BANK in S0's unit format (rows reversed inside every unit, top8 recomputed with the frozen rule,
a new MANIFEST), then pushed through the exact day-of chain:  build_candidate_bank `manifest` -> `verify` -> ep_admission `run`.  Every number must equal `selftest-real` (which reads the same composite from
memory): the loader, the key alignment, the manifest, the verify controls and the CLI are exercised end to end on S0's real geometry (15,808 sets, 117 logs).  A SECOND bank, a bit-for-bit COPY of P2's logits,
is the planted wrong-checkpoint arm: `verify` must read it as a FAIL.  Everything is written under --work (on D:) and deleted at the end unless --keep.

    python integration_disk_replication.py --work D:/Projects/TanitAD-artifacts/_p6prep_it --arrays-npz <pickrule arrays.npz> --out integration_disk_replication.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_candidate_bank as BB                                                          # noqa: E402
import ep_admission as A                                                                   # noqa: E402

S0 = BB.S0_ROOT
EP = 2


def top8_frozen(z6):
    p = 1.0 / (1.0 + np.exp(-z6.astype(np.float64)))
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
    return np.argsort(-agg, axis=-1, kind="stable")[..., :8].astype(np.int16)


def write_candidate_bank(root: str, ep_rows: np.ndarray | None, ckpt: str, reverse: bool) -> dict:
    """S0's units -> a candidate bank: EP column replaced by `ep_rows` (in the loader's row order = sorted units, unit order; None = leave P2's logits untouched), top8 recomputed, rows optionally reversed."""
    bank = f"{root}/work/bank"
    os.makedirs(bank, exist_ok=True)
    units = sorted(n for n in os.listdir(f"{S0}/work/bank") if n.endswith(".sets.npz"))
    off, files = 0, {}
    for n in units:
        z = dict(np.load(f"{S0}/work/bank/{n}", allow_pickle=False))
        S = len(z["token"])
        if ep_rows is not None:
            lo = z["logits"].copy()
            lo[..., EP] = ep_rows[off:off + S]
            z["logits"] = lo
            z["top8"] = top8_frozen(lo)
        off += S
        z["ckpt"] = np.array(ckpt)
        if reverse:
            for k, v in list(z.items()):
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == S and k not in ("label_order",):
                    z[k] = v[::-1].copy()
        p = f"{bank}/{n}"
        np.savez(p, **z)
        files[f"work/bank/{n}"] = {"path": p.replace("\\", "/"), "sha256": BB.sha256_file(p), "bytes": os.path.getsize(p)}
        # the .fan.npz / .report.json of the unit are copied (the manifest lists them; `verify` reads only the sets files)
        for ext in (".fan.npz", ".report.json"):
            src = f"{S0}/work/bank/{n.replace('.sets.npz', ext)}"
            if os.path.exists(src):
                shutil.copyfile(src, f"{bank}/{n.replace('.sets.npz', ext)}")
    return {"units": len(units), "rows": off}


def prepare_root(root: str, ckpt: str, ckpt_sha: str, cid: str) -> None:
    ns = argparse.Namespace(root=root, ckpt=ckpt, ckpt_sha256=ckpt_sha, candidate_id=cid, s0_root=S0, s0_code=BB.S0_CODE)
    assert BB.cmd_prepare(ns) == 0
    for m in ("S0_DONE", "S0_SUMMARY.json"):                                              # the markers `manifest` demands (copied from S0 for THIS test only)
        shutil.copyfile(f"{S0}/run/{m}", f"{root}/run/{m}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--arrays-npz", required=True)
    ap.add_argument("--landed-result", default=None)
    ap.add_argument("--ranked-logs", default=f"{S0}/run/plan/ranked_fit_logs.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    work = os.path.abspath(a.work).replace("\\", "/")
    if os.path.splitdrive(work)[0].upper() == "C:":
        raise SystemExit("--work must be on D: (C: is nearly full)")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    ckpt, ckpt_sha = BB.P2_CKPT, BB.P2_CKPT_SHA256                                       # the stand-in candidate checkpoint: P2's file (the bank is planted, not produced by it)
    z = np.load(a.arrays_npz, allow_pickle=False)
    res = {"what": "disk-path replication of the day-of chain on S0's real geometry; candidate = the EP-RELIAB EP-led composite (EP-R1 c*, OOF seed 0) written as a bank"}
    # --- the planted GOOD-FORMAT candidate: the EP-R1 composite
    r1 = f"{work}/cand_ep_r1"
    prepare_root(r1, ckpt, ckpt_sha, "ep_r1_planted")
    res["written"] = write_candidate_bank(r1, z["ep_OOF_s0"], ckpt, reverse=True)
    assert BB.cmd_manifest(argparse.Namespace(root=r1)) == 0
    vr = BB.cmd_verify(argparse.Namespace(root=r1, p2_bank=BB.P2_BANK))
    res["verify_ep_r1_bank"] = json.load(open(f"{r1}/run/verify_report.json", encoding="utf-8"))
    res["verify_ep_r1_bank_rc"] = vr
    # --- the planted WRONG-CHECKPOINT arm: a bit-for-bit copy of P2's logits must be refused by verify
    cp = f"{work}/cand_p2_copy"
    prepare_root(cp, ckpt, ckpt_sha, "p2_copy_planted")
    write_candidate_bank(cp, None, ckpt, reverse=False)
    assert BB.cmd_manifest(argparse.Namespace(root=cp)) == 0
    vc = BB.cmd_verify(argparse.Namespace(root=cp, p2_bank=BB.P2_BANK))
    res["verify_p2_copy_bank"] = {k: v for k, v in json.load(open(f"{cp}/run/verify_report.json", encoding="utf-8")).items() if k in ("PASS", "same_logits_as_p2_on_any_set", "note")}
    res["verify_p2_copy_bank_rc"] = vc
    # --- the admission CLI on the disk bank
    out = f"{work}/admission_ep_r1.json"
    rc = A.main(["run", "--p2-bank", BB.P2_BANK, "--cand-bank", f"{r1}/work/bank", "--cand-id", "ep_r1_planted", "--ranked-logs", a.ranked_logs, "--allow-unsealed", "--enull-reps", "200", "--out", out])
    adm = json.load(open(out, encoding="utf-8"))
    res["admission_rc"] = rc
    res["admission_label"] = adm["verdict"]["label"]
    res["admission_A1"], res["admission_A2"] = adm["verdict"]["A1"], adm["verdict"]["A2"]
    res["admission_controls"] = {k: {"pass": v.get("pass"), "status": v.get("status")} for k, v in adm["controls"].items()}
    res["admission_warnings"] = adm["warnings"]
    pc = adm["per_candidate"]["ep_r1_planted"]
    res["admission_D_M"] = {k: pc["G"]["D_M"][k] for k in ("point", "lo", "hi")}
    # equality with the in-memory replication (selftest-real) and, if given, the landed EP-RELIAB result
    ref = json.load(open(os.path.join(HERE, "..", "raw", "results", "selftest_real.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "..", "raw", "results", "selftest_real.json")) else None
    if ref is not None:
        mem = ref["composites"]["EP-R1_c*_OOF_s0"]["A1"]["D_M"]
        res["equals_selftest_real_D_M"] = bool(all(abs(float(mem[k]) - float(res["admission_D_M"][k])) <= 1e-12 for k in ("point", "lo", "hi")))
    if a.landed_result:
        landed = json.load(open(a.landed_result, encoding="utf-8"))["per_seed"]["0"]["G"]["D_M"]
        res["equals_landed_ep_reliab_D_M"] = bool(all(abs(float(landed[k]) - float(res["admission_D_M"][k])) <= 1e-9 for k in ("point", "lo", "hi")))
    res["PASS"] = bool(res["verify_ep_r1_bank"]["PASS"] and res["verify_ep_r1_bank_rc"] == 0 and res["verify_p2_copy_bank"]["PASS"] is False and res["verify_p2_copy_bank_rc"] != 0
                       and res["admission_label"] == "REFUSE-A1" and res["admission_A2"] == {"ep_r1_planted": "ADMIT-A2"} and all(v["pass"] for v in res["admission_controls"].values())
                       and res.get("equals_landed_ep_reliab_D_M", True) and res.get("equals_selftest_real_D_M", True))
    res["seconds"] = round(time.time() - t0, 1)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(A.jsonable(res), indent=1, sort_keys=True))
    print(f"DISK_REPLICATION pass={res['PASS']} label={res['admission_label']} {res['seconds']} s -> {a.out}")
    return 0 if res["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
