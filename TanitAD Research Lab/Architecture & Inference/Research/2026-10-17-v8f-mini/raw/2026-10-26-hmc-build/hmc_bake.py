#!/usr/bin/env python3
"""hmc_bake.py -- PREREG_V8F_HACC_MC section 2 / 4 (C-4) : the MC-H, MC-F and STEEP tables, their TRAIN-side qualification, and their bakes.  CPU only, ZERO training.

  tables              fit / derive the 3 tables x 2 seeds (TRAIN data only), C-4, the TRAIN qualification bar  -> tables/*.json, bake/tables_report.json
  bake --seed S       copy M2b_sS_hacc.pt, replace the existing ``hacc_w`` [30, 2] buffer with each table, verify, write to E:  -> bake/ckpt_report_sS.json
  finalize            merge both reports + sha256 of everything -> BAKE_REPORT_MC.json

NOTHING IS RE-DERIVED DIFFERENTLY: the fit lines are probe12_mc_trainfit.py's own (``ref/lcdiag``, byte-snapshot of the repo file); ``d2`` / ``apply`` / ``ade`` are AST-extracted
from that file; MC-H's c is the banked HACC table's value at knot 10 (probe12b); STEEP is probe12's ``wst``.  The values are CONTROLS: they must reproduce probe12.json /
probe12b.json (c to 1e-12) and the PREREG section-2 table (to 4 dp).
"""
import argparse
import ast
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402

FR = "C:/Users/Admin/hlat/frozen"
os.environ["HMC_READ_ROOT"] = FR                                               # taniteval and tanitad must come from ONE tree (taniteval's stack guard); the modules are file-identical to the tip's
sys.path.insert(0, "C:/Users/Admin/hmc")
import hmc_lib as L  # noqa: E402
HMC = L.HMC
TAB = f"{HMC}/tables"
BK = f"{HMC}/bake"
K_FLAT = 12                                                                    # knots 1..12  (w rows 0..11)
KNOT_MCH = 10                                                                  # "HACC's own banked w_along at knot 10, the replan knot"  -> row 9
GRID_N = 201
SHA = {"hacc_ckpt": {0: "71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0", 1: "32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a"},
       "hacc_w_json": {0: "5b8e368fb06b6fea4ab710d071d2d94df58f1e28d4f0648ef6fab79ba664b1ed", 1: "56f78bfefe6fed234bc44e1edb2a97cd47ac47f1a8c4b3ced8a69d1aded18469"},
       "hacc_w_canon": {0: "5c330a9ba98d66ccdaf3770b518f2a23ffee2612987684776a76a89eb33e9b7f", 1: "eee64e066792404710eff44009e4eb6e961466060a72ce378a8706d271d4e0a1"}}
MD5 = {"fit_clips": "e419b10e3b1234986eb6b0d0a3372b04", "eval_join": "532fcd644c0e0835e847b2f1cee011ad"}
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
FIT_CLIPS, EVAL_JOIN = f"{SC}/hblend/fit_clips.json", "C:/Users/Admin/hlat/inputs/join_eval141.npz"
TRAIN_TICKS = "C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz"
PROBE12 = json.load(open(f"{L.REF}/lcdiag/probe12.json"))
PROBE12B = json.load(open(f"{L.REF}/lcdiag/probe12b.json"))
# the PREREG section-2 qualification table (printed values; compared at 4 dp)
PREREG_TRAIN = {"ade6": {"HACC": (2.0373, 1.9865), "MC-H": (2.0397, 1.9887), "MC-F": (2.0429, 1.9925), "P10": (2.5502, 2.5502), "STEEP": (2.0382, 1.9874)},
                "ade2": {"HACC": (0.2109, 0.2119), "MC-H": (0.2179, 0.2186), "MC-F": (0.2276, 0.2295), "P10": (0.2186, 0.2186)}}
TRAIN_BAR_M = 0.0604                                                           # "HACC's + F/2 (0.0604 m)"

# probe12's own pure helpers, AST-extracted (reuse, never re-implement)
_ns = {"np": np}
_src = f"{L.REF}/lcdiag/probe12_mc_trainfit.py"
for _n in ast.parse(open(_src, encoding="utf-8").read()).body:
    if isinstance(_n, ast.FunctionDef) and _n.name in ("d2", "apply", "inc", "ade"):
        exec(compile(ast.Module([_n], []), _src, "exec"), _ns)
d2, apply, inc, ade = _ns["d2"], _ns["apply"], _ns["inc"], _ns["ade"]


def jdump(o, path):
    def enc(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(o, fh, indent=1, default=enc)


def md5_file(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_hacc_table(s):
    p = f"{HMC}/inputs/weights_HACC_M2b_s{s}.json"
    if L.sha256_file(p) != SHA["hacc_w_json"][s]:
        raise SystemExit(f"REFUSED: {p} is not the banked HACC weights file")
    wj = json.load(open(p))
    wh = np.stack([np.asarray(wj["w_along_by_step"], np.float64), np.asarray(wj["w_cross_by_step"], np.float64)], 1)
    if L.canon(wh) != SHA["hacc_w_canon"][s]:
        raise SystemExit("REFUSED: HACC's table is not the scored one (canonical sha)")
    return wh, float(wj["lam_star"]), wj


def train_ticks(s):
    """The TRAIN tick dump of the HACC model's own T1 loop (HACC-LAT's), refused unless it is TRAIN-only and disjoint from eval-141 (HACC-LAT's TickBundle guard, same three checks)."""
    p = TRAIN_TICKS.format(s=s)
    if md5_file(FIT_CLIPS) != MD5["fit_clips"] or md5_file(EVAL_JOIN) != MD5["eval_join"]:
        raise SystemExit("REFUSED: fit_clips / eval join is not the pinned file")
    z = np.load(p, allow_pickle=False)
    if str(z["split"]) != "train":
        raise SystemExit(f"REFUSED: {p}: split {str(z['split'])!r} is not 'train'")
    clips = {str(c) for c in z["clip"]}
    fit = set(json.load(open(FIT_CLIPS, encoding="utf-8"))["clips"])
    ev = {str(c) for c in np.load(EVAL_JOIN, allow_pickle=False)["clip"]}
    if clips & ev or fit & ev or not clips <= fit:
        raise SystemExit("REFUSED: the tick dump is not TRAIN-only / shares a clip with eval-141")
    return z, {"path": p, "sha256": L.sha256_file(p), "split": "train", "n_clips": len(clips), "n_windows": int(z["n_windows"]), "checked": "split == train; clips subset of fit_clips.json; disjoint from the eval-141 join"}


def fit_c(P, T, Y, wh, lam, k_flat=K_FLAT):
    """probe12's inline lines, verbatim, as a function: the closed-form minimiser of HACC's own objective (positions + lambda* x D2, origin prepended) over c in [0, 1] with the
    along weights of knots 1..k_flat replaced by the constant c and the rest held at HACC's.  P, T, Y [n, 30] (the along axis)."""
    D = T - P
    flat = np.arange(30) < k_flat
    u = np.where(flat[None], D, 0.0)
    e0 = (P - Y) + np.where(flat[None], 0.0, wh[None, :, 0] * D)
    num = (e0 * u).sum() + lam * (d2(e0) * d2(u)).sum()
    den = (u * u).sum() + lam * (d2(u) ** 2).sum()
    c = float(np.clip(-num / den, 0.0, 1.0))

    def J(cc):
        e = e0 + cc * u
        return float((e ** 2).sum() + lam * (d2(e) ** 2).sum())
    grid = np.linspace(0, 1, GRID_N)
    cg = float(grid[np.argmin([J(g) for g in grid])])
    return c, cg, float(-num / den)


def make_tables(wh, c_f):
    w_h = wh.copy(); w_h[:K_FLAT, 0] = wh[KNOT_MCH - 1, 0]
    w_f = wh.copy(); w_f[:K_FLAT, 0] = c_f
    w_s = wh.copy(); w_s[7:12, 0] = np.linspace(0.0, 0.6, 5)                   # knots 8..12: a ramp 0 -> 0.6 (probe12's ``wst``)
    return {"MC-H": w_h, "MC-F": w_f, "STEEP": w_s}


def tag_of(arm):
    return {"MC-H": "MCH", "MC-F": "MCF", "STEEP": "STEEP"}[arm]


def cmd_tables(a):
    os.makedirs(TAB, exist_ok=True); os.makedirs(BK, exist_ok=True)
    rep = {"prereg": "Project Steering/PREREG_V8F_HACC_MC.md @ 8d4b936d", "tier": "TRAIN-side fit and qualification only; no eval array is opened", "seeds": {}}
    all_pass = True
    for s in L.SEEDS:
        t0 = time.time()
        wh, lam, wj = read_hacc_table(s)
        z, zinfo = train_ticks(s)
        m0 = (z["tick"] == 0) & z["gt_ok"]
        n0 = int(m0.sum())
        P, T, Y = z["P"][m0][..., 0], z["tau"][m0][..., 0], z["gt"][m0][..., 0]
        c_f, c_grid, c_raw = fit_c(P, T, Y, wh, lam)
        # ---- C-4 clause 1: closed form vs grid argmin
        c4a = abs(c_f - c_grid)
        # ---- C-4 clause 2 (LITERAL): the regression GT := P gives c = 0
        c_reg, c_reg_grid, c_reg_raw = fit_c(P, T, P, wh, lam)
        # beside, NOT a control (documents why clause 2 cannot hold as written): data-only c (lambda = 0); the regression with the held tail also zeroed (analytic identity)
        D = T - P
        c_data_only = float(np.clip(-((P - Y)[:, :K_FLAT] * D[:, :K_FLAT]).sum() / (D[:, :K_FLAT] ** 2).sum(), 0.0, 1.0))
        wh0 = wh.copy(); wh0[:, 0] = 0.0
        c_reg_tail0 = fit_c(P, T, P, wh0, lam)[0]
        tabs = make_tables(wh, c_f)
        # ---- reproduction controls against probe12 / probe12b
        rep_ok = {"c_mc_f_vs_probe12": abs(c_f - PROBE12[f"s{s}"]["c_mc"]), "c_mc_h_vs_probe12b": abs(tabs["MC-H"][0, 0] - PROBE12B[f"s{s}"]["c"]),
                  "grid_vs_probe12": abs(c_grid - PROBE12[f"s{s}"]["c_grid_check"]),
                  "w_along_mc_f_vs_probe12_4dp": float(np.abs(tabs["MC-F"][:, 0].round(4) - np.asarray(PROBE12[f"s{s}"]["w_along_mc"])).max()),
                  "w_along_steep_vs_probe12_4dp": float(np.abs(tabs["STEEP"][:, 0].round(4) - np.asarray(PROBE12[f"s{s}"]["w_along_steep"])).max()),
                  "n_tick0_windows": n0, "n_tick0_expected": 9582}
        rep_ok["PASS"] = bool(rep_ok["c_mc_f_vs_probe12"] <= 1e-12 and rep_ok["c_mc_h_vs_probe12b"] <= 1e-12 and rep_ok["grid_vs_probe12"] <= 1e-12 and
                              rep_ok["w_along_mc_f_vs_probe12_4dp"] <= 1e-12 and rep_ok["w_along_steep_vs_probe12_4dp"] <= 1e-12 and n0 == 9582)
        # ---- TRAIN qualification: open-loop ADE2 / ADE6 on the TRAIN tick-0 samples (n = 9,582)
        PP, TT, YY = z["P"][m0], z["tau"][m0], z["gt"][m0]
        ev = {"P10": ade(PP, YY), "HACC": ade(apply(PP, TT, wh), YY)}
        for arm, w in tabs.items():
            ev[arm] = ade(apply(PP, TT, w), YY)
        q = {}
        for arm, (a2, a6) in ev.items():
            want6, want2 = PREREG_TRAIN["ade6"][arm][s], PREREG_TRAIN["ade2"].get(arm, (None, None))[s]
            q[arm] = {"ade2": a2, "ade6": a6, "prereg_ade6": want6, "prereg_ade2": want2,
                      "matches_prereg_4dp": bool(abs(a6 - want6) <= 5e-5 and (want2 is None or abs(a2 - want2) <= 5e-5))}
            if arm in tabs:
                q[arm]["ade6_minus_HACC_m"] = a6 - ev["HACC"][1]
                q[arm]["BAR_ade6_le_HACC_plus_F_half"] = bool(a6 <= ev["HACC"][1] + TRAIN_BAR_M)
        train_bar = all(q[arm]["BAR_ade6_le_HACC_plus_F_half"] for arm in tabs)
        c4 = {"clause1_closed_form_vs_grid_argmin": {"c_closed_form": c_f, "c_grid_argmin_201pt": c_grid, "abs_diff": c4a, "bar": 0.005, "PASS": bool(c4a <= 0.005)},
              "clause2_regression_GT_is_P_gives_c_zero": {"c": c_reg, "c_raw_unclipped": c_reg_raw, "bar": "c = 0 (the PREREG says 'as stated'; no tolerance is printed, so exact 0 and <= 1e-9 are both reported)",
                                                          "PASS": bool(abs(c_reg) <= 1e-9)},
              "beside_NOT_A_CONTROL": {"c_data_only_lambda0": c_data_only, "c_regression_with_held_tail_also_zeroed": c_reg_tail0,
                                       "HACC_w_along_knot13": float(wh[12, 0]), "HACC_w_along_knot10": float(wh[9, 0]),
                                       "reading": "with lambda* > 0 the fit is pinned by D2-continuity with the HELD tail (HACC's w at knots 13+), not by the data: GT := P moves c by "
                                                  f"{c_reg - c_f:+.4f} only; the data-only c is {c_data_only:.4f}"}}
        c4["PASS"] = bool(c4["clause1_closed_form_vs_grid_argmin"]["PASS"] and c4["clause2_regression_GT_is_P_gives_c_zero"]["PASS"])
        all_pass &= bool(rep_ok["PASS"] and train_bar and c4["clause1_closed_form_vs_grid_argmin"]["PASS"])
        # ---- write the tables
        paths = {}
        for arm, w in tabs.items():
            p = f"{TAB}/table_{tag_of(arm)}_M2b_s{s}.json"
            c_arm = None if arm == "STEEP" else float(w[0, 0])
            jdump({"w_along_by_step": w[:, 0], "w_cross_by_step": w[:, 1], "arm": arm, "seed": f"M2b_s{s}", "prereg": "PREREG_V8F_HACC_MC", "c_knots_1_12": c_arm,
                   "tail_and_cross": f"HACC's banked table (canonical sha {SHA['hacc_w_canon'][s][:12]})", "tier": "TRAIN-side derivation, zero training"}, p)
            paths[arm] = {"path": p, "file_sha256": L.sha256_file(p), "canonical_sha256": L.canon(w), "c": c_arm, "w_along_knots_1_14": w[:14, 0].round(6).tolist()}
        rep["seeds"][f"s{s}"] = {"lam_star": lam, "train_ticks": zinfo, "n_tick0_windows": n0, "reproduces_probe12": rep_ok, "C-4": c4, "train_qualification": q,
                                 "train_bar_ade6_le_HACC_plus": TRAIN_BAR_M, "TRAIN_BAR_PASS": bool(train_bar), "tables": paths, "seconds": time.time() - t0}
        print(f"[tables s{s}] c_MC-H {tabs['MC-H'][0, 0]:.4f}  c_MC-F {c_f:.4f} (grid {c_grid:.3f}, diff {c4a:.4f})  regression c {c_reg:.4f}  reproduces probe12 {rep_ok['PASS']}  "
              f"TRAIN bar {train_bar}  ADE6 " + " ".join(f"{k} {v[1]:.4f}" for k, v in ev.items()) + f"  {time.time() - t0:.0f}s", flush=True)
    rep["C4_PASS_as_written"] = all(rep["seeds"][f"s{s}"]["C-4"]["PASS"] for s in L.SEEDS)
    rep["TRAIN_BAR_PASS"] = all(rep["seeds"][f"s{s}"]["TRAIN_BAR_PASS"] for s in L.SEEDS)
    rep["reproduces_probe12_and_prereg_values"] = all(rep["seeds"][f"s{s}"]["reproduces_probe12"]["PASS"] and
                                                       all(v["matches_prereg_4dp"] for v in rep["seeds"][f"s{s}"]["train_qualification"].values()) for s in L.SEEDS)
    jdump(rep, f"{BK}/tables_report.json")
    print("[tables] C-4 as written:", rep["C4_PASS_as_written"], "| TRAIN bar:", rep["TRAIN_BAR_PASS"], "| reproduces probe12 + PREREG table:", rep["reproduces_probe12_and_prereg_values"])
    return 0


# ================================================================================================ the bake
def cmd_bake(a):
    import torch
    for p in (f"{FR}/stack", f"{FR}/taniteval", f"{FR}/stack/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import v8f_hlat as HL                                                      # HACC-LAT's bake machinery (_build, verify_batch_from_train): the existing bake path
    s = a.seed
    os.makedirs(L.ART, exist_ok=True); os.makedirs(BK, exist_ok=True)
    src = f"{L.HACC_ART}/M2b_s{s}_hacc.pt"
    if L.sha256_file(src) != SHA["hacc_ckpt"][s]:
        raise SystemExit("REFUSED: the HACC checkpoint is not the pinned file")
    wh, _, _ = read_hacc_table(s)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    assert torch.equal(ck["model"]["hacc_w"], torch.as_tensor(wh, dtype=torch.float32)), "the shipped checkpoint's hacc_w is not HACC's banked table"
    assert "hacc_gate_v" not in ck["model"]
    vb = HL.verify_batch_from_train(ck["cfg"])
    m_src = HL._build(ck)
    infer = lambda m: {k: t for k, t in m.forward_infer(vb).items() if torch.is_tensor(t)}
    with torch.no_grad():
        y_src = infer(m_src)
    rep = {"seed": s, "source_ckpt": src, "source_sha256": SHA["hacc_ckpt"][s], "n_verify_windows": int(y_src["traj30"].shape[0]), "arms": {}}
    # control: the bake path fed HACC's OWN table reproduces the shipped checkpoint tensor for tensor (in memory)
    m_ctl = HL._build(ck); m_ctl.set_hacc_weights(wh)
    sd_ctl = m_ctl.state_dict()
    rep["control_bake_path_with_HACC_table_equals_shipped"] = bool(set(sd_ctl) == set(ck["model"]) and all(torch.equal(sd_ctl[k], ck["model"][k]) for k in ck["model"]))
    assert rep["control_bake_path_with_HACC_table_equals_shipped"]
    for arm in ("MC-H", "MC-F", "STEEP"):
        t0 = time.time()
        tp = f"{TAB}/table_{tag_of(arm)}_M2b_s{s}.json"
        tj = json.load(open(tp))
        w = np.stack([np.asarray(tj["w_along_by_step"], np.float64), np.asarray(tj["w_cross_by_step"], np.float64)], 1)
        out = f"{L.ART}/M2b_s{s}_hmc_{tag_of(arm)}.pt"
        if os.path.exists(out):
            raise SystemExit(f"REFUSED: {out} exists: a baked checkpoint is never overwritten")
        m_rt = HL._build(ck); m_rt.set_hacc_weights(w)
        sd = m_rt.state_dict()
        prov = {"prereg": "PREREG_V8F_HACC_MC", "arm": arm, "seed": f"M2b_s{s}", "source_ckpt_sha256": SHA["hacc_ckpt"][s], "table_file": tp, "table_file_sha256": L.sha256_file(tp),
                "weights_canonical_sha256": L.canon(w), "what": "forward_infer returns P + w (O - P); only the existing hacc_w [30, 2] buffer differs from HACC's shipped checkpoint; training untouched"}
        torch.save({**ck, "model": sd, "hmc": prov}, out)
        ck2 = torch.load(out, map_location="cpu", weights_only=False)
        assert set(ck2["model"]) == set(ck["model"]), set(ck2["model"]) ^ set(ck["model"])
        others = [k for k in ck["model"] if k != "hacc_w"]
        assert all(torch.equal(ck2["model"][k], ck["model"][k]) for k in others), "a source tensor changed"
        assert torch.equal(ck2["model"]["hacc_w"], torch.as_tensor(w, dtype=torch.float32)), "hacc_w is not the table"
        m_file = HL._build(ck2)
        assert m_file.hacc_active and not m_file.hacc_gate_active
        with torch.no_grad():
            y_file, y_rt = infer(m_file), infer(m_rt)
        assert set(y_file) == set(y_rt) and all(torch.equal(y_file[k], y_rt[k]) for k in y_file), "the baked file's forward_infer differs from the runtime-set model"
        dif = (y_file["traj30"] - y_src["traj30"]).abs()
        rep["arms"][arm] = {"out": out, "sha256": L.sha256_file(out), "bytes": os.path.getsize(out), "table_file": tp, "table_file_sha256": prov["table_file_sha256"],
                            "canonical_sha256": prov["weights_canonical_sha256"], "other_tensors_bit_identical_to_shipped_HACC": len(others),
                            "hacc_w_equals_float32_table": True, "baked_vs_runtime_set": "bit-identical on every output tensor",
                            "max_abs_vs_HACC_traj30_m": float(dif.max()), "max_abs_vs_HACC_traj30_along_m": float(dif[..., 0].max()), "max_abs_vs_HACC_traj30_cross_m": float(dif[..., 1].max()),
                            "seconds": time.time() - t0}
        print(f"[bake s{s} {arm}] {out}  sha {rep['arms'][arm]['sha256'][:16]}  max|d vs HACC| along {rep['arms'][arm]['max_abs_vs_HACC_traj30_along_m']:.3f} m cross "
              f"{rep['arms'][arm]['max_abs_vs_HACC_traj30_cross_m']:.2e} m  {time.time() - t0:.0f}s", flush=True)
    jdump(rep, f"{BK}/ckpt_report_s{s}.json")
    return 0


def cmd_finalize(a):
    t = json.load(open(f"{BK}/tables_report.json"))
    out = {"prereg": t["prereg"], "built": time.strftime("%Y-%m-%d %H:%M:%S"), "tier": "zero training; CPU", "tables": t, "ckpts": {}}
    for s in L.SEEDS:
        out["ckpts"][f"s{s}"] = json.load(open(f"{BK}/ckpt_report_s{s}.json"))
    out["C-4_as_written_PASS"] = t["C4_PASS_as_written"]
    out["TRAIN_BAR_PASS"] = t["TRAIN_BAR_PASS"]
    out["input_pins"] = {"hmc_lib_sha256": L.sha256_file(f"{HMC}/hmc_lib.py"), "hmc_bake_sha256": L.sha256_file(f"{HMC}/hmc_bake.py"), "ref_snapshots": open(f"{HMC}/ref_SHA256.txt").read().split("\n")}
    jdump(out, f"{HMC}/BAKE_REPORT_MC.json")
    print("wrote", f"{HMC}/BAKE_REPORT_MC.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tables")
    p = sub.add_parser("bake"); p.add_argument("--seed", type=int, required=True, choices=(0, 1))
    sub.add_parser("finalize")
    a = ap.parse_args()
    return {"tables": cmd_tables, "bake": cmd_bake, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
