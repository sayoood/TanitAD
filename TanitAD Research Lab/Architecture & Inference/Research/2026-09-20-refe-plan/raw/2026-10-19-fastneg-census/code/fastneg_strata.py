"""FASTNEG-CENSUS step 2: the LABEL-FREE strata script.  Computes the speed ratio r, the standstill set S and the quartile cut points of the f10 fit set from SPEEDS ONLY and writes
`strata.json` (+ its sha256) and `speeds.npz`.  It NEVER opens a label: it reads the members `log, token, step, rank, ckpt_step, logits` of the bank units and `token, step, rank, ckpt_step,
traj3` of the per-log stores through `fastneg_lib.read_members`, which refuses any member whose name contains "label".  `--labels-path` is accepted only so that the proof can point it at a
NON-EXISTENT file: the script records it and never opens it (the only access is an existence check, `os.path.exists`, to record that it was absent).

REGISTERED SOURCE: PREREG_FASTNEG_CENSUS.md s3 + PREREG_FASTNEG_CENSUS_AMENDMENT1.md A1.1 / A1.2.
    python fastneg_strata.py --out-dir RUN/strata [--labels-path C:/does/not/exist/labels6.npz]
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastneg_lib as L                                                               # noqa: E402

DEFAULT_BANK = "C:/Users/Admin/srttc_s0/work/bank"
DEFAULT_MANIFEST = DEFAULT_BANK + "/MANIFEST.sha256.json"
DEFAULT_RANKED = "C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv"
DEFAULT_STORES = "C:/Users/Admin/srttc_s0/work/sets"
EXPECT_SETS, EXPECT_LOGS, EXPECT_MANIFEST_SHA = 15808, 117, "6296dfc0b03905038fbf132f82009f6b6d7e8d61fa9abfb65c9164f487a5f2a1"


def build(a) -> dict:
    E = L.load_epr1(a.epr1)
    guard = E.PathGuard(read_dirs=[a.bank_dir, a.stores_dir], read_files=[a.manifest, a.ranked], write_dirs=[a.out_dir])
    manifest, man_sha = E.load_manifest(guard, a.manifest)
    if a.expect_manifest_sha and man_sha != a.expect_manifest_sha:
        raise L.CensusError(f"bank manifest sha256 {man_sha} != the pinned {a.expect_manifest_sha}")
    rows, ranked_sha = E.load_ranked_logs(guard, a.ranked, (a.ranked_sha or None), (a.ranked_n or None))
    selected, _ = E.select_logs(rows, a.fraction, None)
    if a.expect_logs and len(selected) != a.expect_logs:
        raise L.CensusError(f"{len(selected)} f{a.fraction} logs != the pinned {a.expect_logs}")
    sel = {r["log"]: i for i, r in enumerate(selected)}
    units = [n for n in E._io_listdir(guard, a.bank_dir) if n.endswith(".sets.npz")]
    if not units:
        raise L.CensusError(f"no *.sets.npz in {a.bank_dir}")
    # ---- bank: keys and the frozen-rule pick, in the ORDER epr1.load_fit_data uses (sorted unit names, in-unit order)
    keys = {k: [] for k in ("log", "token", "step", "rank", "ckpt_step")}
    picks, unit_of = [], []
    for n in units:
        if n not in manifest:
            raise L.CensusError(f"unit {n} is not in the manifest")
        guard.check_read(os.path.join(a.bank_dir, n))
        m = L.read_members(os.path.join(a.bank_dir, n), L.BANK_MEMBERS)
        mask = np.array([str(x) in sel for x in m["log"]], dtype=bool)
        if not mask.any():
            continue
        for k in keys:
            keys[k].append(m[k][mask])
        picks.append(E.picks_np(m["logits"][mask]))
        unit_of += [n] * int(mask.sum())
    key = {k: np.concatenate(v) for k, v in keys.items()}
    pick = np.concatenate(picks).astype(np.int64)
    N = len(pick)
    if a.expect_sets and N != a.expect_sets:
        raise L.CensusError(f"{N} sets != the pinned {a.expect_sets}")
    if {str(x) for x in key["log"]} != set(sel):
        raise L.CensusError("the bank does not cover exactly the selected logs")
    # ---- stores: trajectories, joined by the exact key
    v = np.empty((N, L.N_PROP), np.float64)
    seg_first, seg_rest = [], []
    log_arr = np.array([str(x) for x in key["log"]])
    for lg in selected:
        name = lg["log"]
        p = os.path.join(a.stores_dir, name + ".npz")
        guard.check_read(p)
        s = L.read_members(p, L.STORE_MEMBERS)
        tr = s["traj3"]
        if tr.ndim != 4 or tr.shape[1:] != (L.N_PROP, L.N_STEPS, 3) or not np.isfinite(tr).all():
            raise L.CensusError(f"{name}: traj3 {tr.shape} is not [S,64,20,3] finite")
        where = {(str(t), int(st), int(r)): i for i, (t, st, r) in enumerate(zip(s["token"], s["step"], s["rank"]))}
        if len(where) != len(s["token"]):
            raise L.CensusError(f"{name}: duplicate (token, step, rank) in the store")
        idx = np.nonzero(log_arr == name)[0]
        used = set()
        for i in idx:
            kk = (str(key["token"][i]), int(key["step"][i]), int(key["rank"][i]))
            j = where.get(kk)
            if j is None:
                raise L.CensusError(f"{name}: bank set {kk} is not in the store")
            if int(s["ckpt_step"][j]) != int(key["ckpt_step"][i]):
                raise L.CensusError(f"{name}: ckpt_step differs for {kk}")
            used.add(j)
        if len(used) != len(s["token"]) or len(idx) != len(used):
            raise L.CensusError(f"{name}: the store has {len(s['token'])} sets, the bank {len(idx)}, joined {len(used)} (must be equal)")
        for i in idx:
            j = where[(str(key["token"][i]), int(key["step"][i]), int(key["rank"][i]))]
            seg = L.segment_lengths(tr[j])                                       # [64,20]
            v[i] = seg.sum(-1) / L.HORIZON_S
            fast = seg.sum(-1) > 8.0                                              # > 2 m/s mean: a stationary pose 1 would read ~0 on these
            if fast.any():
                seg_first.append(float(np.median(seg[fast, 0])))
                seg_rest.append(float(np.median(seg[fast, 1:])))
    v_b = v[np.arange(N), pick]
    r, cuts, code = L.build_strata(v, v_b)
    s_set = v_b < L.STAND_V
    return {"E": E, "key": key, "pick": pick, "unit_of": unit_of, "v": v, "v_b": v_b, "r": r, "cuts": cuts, "code": code, "s_set": s_set, "selected": selected, "man_sha": man_sha,
            "ranked_sha": ranked_sha, "manifest": manifest, "units": units, "seg_first": seg_first, "seg_rest": seg_rest}


def payload(R: dict) -> dict:
    code, key = R["code"], R["key"]
    counts = {n: int((code == i).sum()) for i, n in enumerate(L.STRATA_NAMES)}
    ratio = np.array(R["seg_first"]) / np.array(R["seg_rest"])
    return {
        "cut_points": {"c25": float(R["cuts"][0]), "c50": float(R["cuts"][1]), "c75": float(R["cuts"][2]),
                       "c25_hex": float(R["cuts"][0]).hex(), "c50_hex": float(R["cuts"][1]).hex(), "c75_hex": float(R["cuts"][2]).hex()},
        "rule": "S: v_b < 0.5 m/s (strict); Q1: r <= c25; Q2: c25 < r <= c50; Q3: c50 < r <= c75; Q4 = F: r > c75; r = v_p / v_b; numpy.quantile linear float64 over every proposal of every non-S set",
        "n_sets": int(len(R["pick"])), "n_logs": int(len(R["selected"])), "n_tokens": int(len({str(t) for t in key["token"]})),
        "n_S_sets": int(R["s_set"].sum()), "n_nonS_sets": int((~R["s_set"]).sum()), "rows_per_stratum": counts, "rows_total": int(code.size),
        "stand_v_mps": L.STAND_V, "horizon_s": L.HORIZON_S, "dt_s": L.DT_S,
        "speed": {"v_p_p50": float(np.median(R["v"])), "v_p_max": float(R["v"].max()), "v_b_p50": float(np.median(R["v_b"])), "v_b_max": float(R["v_b"].max()),
                  "r_p50": float(np.nanmedian(R["r"])), "r_min": float(np.nanmin(R["r"])), "r_max": float(np.nanmax(R["r"]))},
        "control_first_segment_over_later": {"median_ratio": float(np.median(ratio)) if len(ratio) else float("nan"), "n_proposals_fast": int(len(ratio)),
                                             "reading": "a pose 1 at t0 would read ~0; a pose 1 at t = 0.2 s reads ~1 (A1.2)"},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default=DEFAULT_BANK)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--ranked", default=DEFAULT_RANKED)
    ap.add_argument("--stores-dir", default=DEFAULT_STORES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fraction", type=int, default=10)
    ap.add_argument("--labels-path", default="", help="never opened; the proof points it at a non-existent file")
    ap.add_argument("--epr1", default=None)
    ap.add_argument("--ranked-sha", default="a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1", help="pinned sha256 of ranked_fit_logs.tsv (tests pass an empty string)")
    ap.add_argument("--ranked-n", type=int, default=1167, help="pinned number of ranked fit logs (tests pass 0)")
    ap.add_argument("--expect-sets", type=int, default=EXPECT_SETS)
    ap.add_argument("--expect-logs", type=int, default=EXPECT_LOGS)
    ap.add_argument("--expect-manifest-sha", default=EXPECT_MANIFEST_SHA)
    ap.add_argument("--no-write", action="store_true", help="compute and return the payload only (the stripped-copy proof)")
    a = ap.parse_args(argv)
    L.OPENED.clear()
    R = build(a)
    P = payload(R)
    arrays = {"key_log": R["key"]["log"].astype("U"), "key_token": R["key"]["token"].astype("U"), "key_step": R["key"]["step"].astype(np.int64), "key_rank": R["key"]["rank"].astype(np.int64),
              "key_ckpt_step": R["key"]["ckpt_step"].astype(np.int64), "pick": R["pick"], "v": R["v"], "v_b": R["v_b"], "r": R["r"], "code": R["code"]}
    content = L.arrays_digest(arrays)
    members = sorted({m for _, m in L.OPENED})
    labels_opened = [m for m in members if "label" in m.lower()]
    if labels_opened:
        raise L.LabelAccessError(f"label members were opened: {labels_opened}")
    res = {"payload": P, "content_digest": content, "members_read": members}
    if a.no_write:
        print("STRATA_CONTENT_DIGEST", content)
        return res
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    sp = os.path.join(out, "speeds.npz")
    sj = os.path.join(out, "strata.json")
    if os.path.exists(sj) or os.path.exists(sp):
        raise L.CensusError(f"{sj} / {sp} already exist: strata are computed ONCE")
    np.savez(sp, **arrays)
    doc = {"schema": L.SCHEMA, "kind": "strata", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "prereg_sha256": L.PREREG_SHA256, "amendment1_sha256": L.AMENDMENT1_SHA256, "epr1_sha256": L.EPR1_SHA256, "ranked_logs_sha256": R["ranked_sha"], "bank_manifest_sha256": R["man_sha"],
           "bank_units": {n: R["manifest"][n] for n in R["units"]}, "fraction": a.fraction,
           "ordering": "sorted bank unit basenames, in-unit row order, restricted to the f10 logs (= epr1.load_fit_data order)",
           "speeds_npz": {"file": "speeds.npz", "sha256": L.sha256_file(sp), "content_digest": content, "arrays": sorted(arrays)},
           "members_read": members, "bank_members_allowed": list(L.BANK_MEMBERS), "store_members_allowed": list(L.STORE_MEMBERS), "n_member_reads": len(L.OPENED),
           "labels_path_arg": a.labels_path, "labels_path_exists_at_run": bool(a.labels_path) and os.path.exists(a.labels_path), "labels_opened": labels_opened,
           "code_sha256": {"fastneg_strata.py": L.sha256_file(os.path.abspath(__file__)), "fastneg_lib.py": L.sha256_file(os.path.join(HERE, "fastneg_lib.py"))},
           **P}
    data = L.canon_json(doc)
    with open(sj, "xb") as f:
        f.write(data)
    sha = L.sha256_bytes(data)
    with open(sj + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha} *strata.json\n")
    print("STRATA_JSON_SHA256", sha)
    print("STRATA_CUTS", P["cut_points"], "rows", P["rows_per_stratum"], "S_sets", P["n_S_sets"], "of", P["n_sets"])
    print("STRATA_MEMBERS_READ", members)
    return 0


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
