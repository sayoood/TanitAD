"""PREREG_V8F_DAG_HACC section 3: the controls table. C-H1 / C-H2 (recipe identity, train-only), C-H3 / C-H4 (bake identity, bake scope) come from their own banked JSONs
(ch1/, fit/, ckpt/); C-H5 = Read A's C1-C4 re-run on the new arm, computed here:
  C1  the readers reproduce their banked numbers on their banked arms, +-1e-4: t1k_read comps ADE6 HACC-P10 (causal) -0.7680 / -0.7336; t1klc lat_rows L2 +0.0210 / +0.0082;
      M1 / M2 (4 dp) 0.0536 / 0.0277, +0.0202 / +0.0072, M1(P10) 0.0137; t1kf_read_a1 (unmodified) on HACC's banked dumps == banked T1KF_READ_A1.json.   (S-POS path: spos_repro.json)
  C2  (arm, arm) reads +0 [0, 0] on every row of every reader (comps rows; t1kf rows; the t1kf reader's own internal identity control)
  C3  P10's tracks paired with DAG-HACC equal the banked floor, max |d| = 0.0 m at both seeds
  C4  every pair is scored on the same 2,399 windows / 141 episodes
  + reachability (added): the post-map moves the executed plan in the rollouts (max |cl - cl_DAG| > 1e-3 m)"""
import io
import json
import os
import sys
import tarfile

os.environ.setdefault("RR_READERS", "E:/Projects/TanitAD-artifacts/dagreread/reader_src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
import numpy as np  # noqa: E402
import readA as RA  # noqa: E402
import rr_lib as R  # noqa: E402

PK = "E:/Projects/TanitAD-artifacts/daghacc"
G = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/gitdumps"
HT = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc"
TIP = f"{PK}/tip/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731


def tar_dir(tar_path, prefix):
    out = {}
    with tarfile.open(tar_path) as tf:
        names = sorted(m.name for m in tf.getmembers() if m.name.startswith(prefix + "/ep") and m.name.endswith(".npz"))
        assert len(names) == 141, (prefix, len(names))
        zs = [np.load(io.BytesIO(tf.extractfile(n).read())) for n in names]
    for k in R.KEYS:
        if k in zs[0].files:
            out[k] = np.concatenate([z[k] for z in zs])
    return out


def numdiff(a, b, path="", acc=None):
    acc = [] if acc is None else acc
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b and k not in ("runtime_s", "provenance", "inputs"):
                numdiff(a[k], b[k], f"{path}/{k}", acc)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            numdiff(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        acc.append(abs(float(a) - float(b)))
    return acc


C = {}
# ---- C1: comps / L2 / M1 on HACC's banked dumps (git-packed bytes the banked reads used, and the float64 originals)
WANT = {"ade6": {0: -0.7680, 1: -0.7336}, "L2": {0: 0.0210, 1: 0.0082}}
WM = {"M1": (0.0536, 0.0277), "M2": (0.0202, 0.0072), "M1_P10": (0.0137, 0.0137)}
c1 = {}
for src in ("git_packed", "f64_tar"):
    o = {}
    for s in R.SEEDS:
        d = R.load_npz(f"{G}/dumps_HACC_s{s}_m10_causal.npz") if src == "git_packed" else tar_dir(f"{HT}/t1kin_causalbucket_dumps_f64_2026-10-26.tar", f"dump_HACC_s{s}_m10_causal")
        a6 = R.paired(R.comps(d["cl"], d["g"])["ade6_m"], R.comps(d["fl"], d["g"])["ade6_m"], d["eid"])
        l2 = R.paired(R.lat_rows(d["cl"], d["g"])["L2_curvature_mae_fast_1pm"], R.lat_rows(d["fl"], d["g"])["L2_curvature_mae_fast_1pm"], d["eid"])
        mr = RA.m_rows(d["cl"], d["fl"], d["g"])
        o[f"s{s}"] = {"ADE6_diff": abs(a6["delta"] - WANT["ade6"][s]), "L2_diff": abs(l2["delta"] - WANT["L2"][s]), "M_matches_4dp": {k: bool(abs(mr[k] - WM[k][s]) <= 5e-5) for k in WM}}
        o[f"s{s}"]["PASS"] = bool(o[f"s{s}"]["ADE6_diff"] <= 1e-4 and o[f"s{s}"]["L2_diff"] <= 1e-4 and all(o[f"s{s}"]["M_matches_4dp"].values()))
    c1[src] = o
C["C1_comps_L2_M1"] = {**c1, "PASS": bool(all(v["PASS"] for src in c1.values() for v in src.values()))}
p = f"{PK}/raw/c1_t1kf/T1KF_READ_A1_HACC_reproduction.json"
if os.path.exists(p):
    rep, ban = J(p), J(f"{TIP}/2026-10-26-t1-families/T1KF_READ_A1.json")
    diffs = numdiff(rep["configs"], ban["configs"])
    C["C1_t1kf_unmodified_on_HACC_banked_dumps"] = {"n_numbers_compared": len(diffs), "max_abs_diff_vs_banked_T1KF_READ_A1": max(diffs) if diffs else None, "PASS": bool(diffs and max(diffs) <= 1e-4)}
else:
    C["C1_t1kf_unmodified_on_HACC_banked_dumps"] = {"PASS": False, "reason": "NOT RUN"}
p = f"{PK}/raw/spos_repro.json"
C["C1_spos_path_reproduces_DAG_READ"] = {"PASS": bool(J(p).get("PASS")) if os.path.exists(p) else False, **({} if os.path.exists(p) else {"reason": "NOT RUN"})}
# ---- C2 / C3 / C4 / reachability on the new arm
rA = J(f"{PK}/raw/readA_DAGHACC.json")
c2, c3, c4, reach = {}, {}, {}, {}
BANKED_FL = {"causal": ("t1kin_causalbucket_dumps_f64_2026-10-26.tar", "dump_HACC_s{s}_m10_causal"), "logged": ("t1kin_hacc_dumps_f64_2026-10-26.tar", "dump_HACC_s{s}_m10"),
             "frozen_causal": ("t1kin_strict_dumps_f64_2026-10-26.tar", "dump_HACC_s{s}_m10_strict")}
for cfg, (tarn, pre) in BANKED_FL.items():
    for s in R.SEEDS:
        nm = "DAGHACCSTRICT" if cfg == "frozen_causal" else "DAGHACC"
        d = R.load_npz(f"{PK}/packed/dumps_{nm}_s{s}_m10_{cfg}.npz")
        hb = tar_dir(f"{HT}/{tarn}", pre.format(s=s))
        k = f"{cfg}_s{s}"
        c3[k] = {"max_abs_fl_vs_banked_HACC_floor_m": float(np.abs(d["fl"] - hb["fl"]).max()), "decisive": cfg == "causal"}
        c4[k] = {"n_windows": int(len(d["eid"])), "n_episodes": int(len(np.unique(d["eid"]))), "g_bit_equal_banked_HACC": bool(np.array_equal(d["g"], hb["g"])),
                 "clip_index_equal_banked_HACC": bool(np.array_equal(d["clip_index"], hb["clip_index"])), "ws_equal_banked_HACC": bool(np.array_equal(d["ws"], hb["ws"])),
                 "eid_equal_banked_HACC": bool(np.array_equal(d["eid"], hb["eid"])), **{f"{x}_equal_DAG": v for x, v in rA["cfgs"][cfg][f"s{s}"]["C4_same_windows_as_DAG"].items() if x not in ("n_windows", "n_episodes")}}
        c2[k] = rA["cfgs"][cfg][f"s{s}"]["C2_identity_comps_rows"]["all_exactly_zero"]
        reach[k] = rA["cfgs"][cfg][f"s{s}"]["plan_moved_vs_DAG"]
tk = f"{PK}/raw/tacdk_daghacc.json"
c2["tacdk_identity"] = {s: v["identity_DAGHACC_DAGHACC_exactly_zero"] for s, v in J(tk)["seeds"].items()} if os.path.exists(tk) else "NOT RUN"
t1kf = f"{PK}/raw/t1kf_DAGHACC/T1KF_READ_A1_DAGHACC.json"
if os.path.exists(t1kf):
    tj = J(t1kf)
    c2["t1kf_reader_own_controls"] = {k: (v if isinstance(v, (bool, int, float, str)) else (v.get("PASS") if isinstance(v, dict) else str(v)[:80])) for k, v in tj["controls"].items()} if isinstance(tj["controls"], dict) else str(tj["controls"])[:300]
C["C2_identity"] = {"detail": c2, "PASS": bool(all(v is True for k, v in c2.items() if k.startswith(("causal", "logged", "frozen"))) and c2.get("tacdk_identity") not in ("NOT RUN",)
                                                 and all(v is True for v in (c2["tacdk_identity"].values() if isinstance(c2["tacdk_identity"], dict) else [False])))}
C["C3_floor"] = {"detail": c3, "PASS_decisive": bool(all(v["max_abs_fl_vs_banked_HACC_floor_m"] == 0.0 for v in c3.values() if v["decisive"])),
                 "PASS_all_configs": bool(all(v["max_abs_fl_vs_banked_HACC_floor_m"] == 0.0 for v in c3.values()))}
ok4 = all(v["n_windows"] == 2399 and v["n_episodes"] == 141 and all(x for kk, x in v.items() if kk.endswith(("_banked_HACC", "_DAG")) and isinstance(x, bool)) for v in c4.values())
C["C4_same_windows"] = {"detail": c4, "PASS": bool(ok4)}
C["reachability_post_map_moves_the_executed_plan"] = {"detail": reach, "PASS": bool(all(v["max_abs_cl_m"] > 1e-3 for v in reach.values()))}
C["DAG_minus_P10_reproduces_banked_reread_rows"] = {s: rA["cfgs"]["causal"][s].get("DAG_minus_P10_reproduces_banked_reread_rows") for s in ("s0", "s1")}
# ---- C-H1..C-H4 from their banked JSONs
C["C_H1_recipe_identity"] = {s: J(f"{PK}/ch1/ch1_{s}.json") for s in ("M2b_s0", "M2b_s1")}
C["C_H1_recipe_identity"]["PASS"] = bool(all(v["C_H1_pass"] for v in C["C_H1_recipe_identity"].values()))
C["C_H2_train_only"] = {"pytest": "test_the_lambda_selection_refuses_anything_that_is_not_train + pinned + 2 real-window ties: 4 passed (logs/ch2_pytest.log)",
                        "refusals_on_the_DAG_fits": {s: J(f"{PK}/fit/CH2_refusal_DAG_{s}.json") for s in ("s0", "s1")}}
C["C_H2_train_only"]["PASS"] = bool(all(v["all_refused"] for v in C["C_H2_train_only"]["refusals_on_the_DAG_fits"].values()) and "4 passed" in open(f"{PK}/logs/ch2_pytest.log").read())
bk = {s: J(f"{PK}/ckpt/BAKE_REPORT_seed{s}.json") for s in (0, 1)}
C["C_H3_bake_identity"] = {f"s{s}": bk[s]["C_H3"] for s in (0, 1)}
C["C_H3_bake_identity"]["PASS"] = bool(all(bk[s]["C_H3"]["pass"] for s in (0, 1)))
C["C_H4_bake_scope"] = {f"s{s}": bk[s]["C_H4"] for s in (0, 1)}
C["C_H4_bake_scope"]["PASS"] = bool(all(bk[s]["C_H4"]["pass"] for s in (0, 1)))
C["C_X1_added_plan_route_reproduces_banked_M2b_fit_windows"] = J(f"{PK}/plans/CX1_route_reproduces_banked_M2b.json")
rollouts = {}
for f in sorted(os.listdir(f"{PK}/t1")):
    if f.startswith("done_") and f.endswith(".json"):
        j = J(f"{PK}/t1/{f}")
        rollouts[j["tag"]] = {"rc": j["rc"], "n_dumps": j["n_dumps"], "ckpt_sha_unchanged": j["ckpt_sha256_before"] == j["ckpt_sha256_after"], "wall_s": j["wall_s"]}
C["rollouts_clean"] = {"detail": rollouts, "PASS": bool(len(rollouts) == 6 and all(v["rc"] == 0 and v["n_dumps"] == 141 and v["ckpt_sha_unchanged"] for v in rollouts.values()))}
passes = {k: v.get("PASS", v.get("PASS_decisive")) for k, v in C.items() if isinstance(v, dict) and ("PASS" in v or "PASS_decisive" in v)}
C["SUMMARY"] = {"each": passes, "all_registered_controls_pass": bool(all(passes[k] for k in ("C_H1_recipe_identity", "C_H2_train_only", "C_H3_bake_identity", "C_H4_bake_scope", "C1_comps_L2_M1", "C1_t1kf_unmodified_on_HACC_banked_dumps",
                                                                                                          "C1_spos_path_reproduces_DAG_READ", "C2_identity", "C3_floor", "C4_same_windows", "rollouts_clean", "reachability_post_map_moves_the_executed_plan"))),
                "failed": [k for k, v in passes.items() if not v]}
json.dump(C, open(f"{PK}/raw/CONTROLS.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps(C["SUMMARY"], indent=1))
print("ZZCONTROLS-DONE")
