"""pf_lane.py -- the CPU lane of the per-frame final read (SHIP-R8 AMENDMENT 3), run with the battery package's OWN instruments, UNCHANGED, re-pointed.

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  Every number is computed by the live battery package's modules (READ-ONLY imports): clamp_posthoc_r8 (CP),
compliant_reread_r8 (CR), a0k0_headroom_retime_r8 (RT), retime_trainfit_r8 (TFM), build_deploy_gains_r8 (BD), deploy_forward_r8 (DF), deploy8_forward_r8 (D8),
tac_retime_r8 (TAC), prior_parity_r8 (PP).  This module only RE-POINTS module attributes, never a function body except ONE: ``BD.check_deploy``, whose last two
checks are literals of the CLIP-FED table (the register's 3-dp knots 0.967 ... and S6's 0.639 / 0.597) and would refuse any other table by construction;
``check_deploy_pf`` keeps every STRUCTURAL check of Amendment 1 and replaces the two clip-fed literals by the knot values read from the frozen file whose
sha256 is the REGISTERED one (the registration is the independent authority for a re-fitted table).

Layout: every PKG-relative output lands under ``<pf_pkg>/raw/...`` (the battery package layout, so the instruments' cross-reads -- the compliant tree json,
FORWARD_TRAINFIT, FORWARD_DEPLOY8, the DEPLOY8 npz -- resolve to the per-frame files):
    raw/clamp_posthoc/trees/<T>.json      CP.read_tree            raw/compliant_reread/trees/<T>.json  CR.read_tree
    raw/retime_trainfit/                  the pf table (byte copy as FROZEN_GAINS_RETIME_train.json + .sha256) and FORWARD_TRAINFIT_<T>.json
    raw/retime_deploy/                    FROZEN_GAINS_RETIME_deploy.json (+ .sha256, Amendment 1 form), refcv8_retime_deploy_gains_pf.json (+ .sha256, the
                                          runtime stage's table, read by D8 through RS.DEPLOY_TABLE_PATH), FORWARD_DEPLOY8_<T>.json + DEPLOY8_POS_<T>_s*.npz,
                                          FORWARD_DEPLOY_<T>.json, TAC_<T>.json, PRIOR_PARITY.json
The battery TREE itself is read from the archive root (CP.ARCHIVE, unchanged): C:/Users/Admin/refcv8_eval_archive/battery/<T>.

    python pf_lane.py install --pkg <pf_pkg> --table <FROZEN_GAINS_RETIME_train_pf.json> --registered <REGISTERED_PF_TABLE.sha256>
    python pf_lane.py clamp|compliant|tf_forward|d8|df|tac|parity --pkg <pf_pkg> --tree <T>
Every command re-checks the installed table against the registered sha256 first (an empty / short string is INCONCLUSIVE, never a match).
CPU only (CUDA_VISIBLE_DEVICES=-1 is forced; RT.forward refuses otherwise).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
R8_CODE = Path(os.environ.get("PF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
RS_TABLE_NAME = "refcv8_retime_deploy_gains_pf.json"
KNOT_T = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]           # LITERAL (Amendment 1)
S2_T, S6_T = [0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_registered(path) -> str:
    try:
        s = "".join(c for c in Path(path).read_text(encoding="utf-8-sig").split()[0] if c in "0123456789abcdefABCDEF").lower()   # BOM / CRLF / "sha  file" tolerant
    except Exception:      # noqa: BLE001
        return ""
    return s if (len(s) == 64 and all(c in "0123456789abcdef" for c in s)) else ""


def mods():
    if str(R8_CODE) not in sys.path:
        sys.path.insert(0, str(R8_CODE))
    import clamp_posthoc_r8 as CP
    import compliant_reread_r8 as CR
    import a0k0_headroom_retime_r8 as RT
    import retime_trainfit_r8 as TFM
    import build_deploy_gains_r8 as BD
    import deploy_forward_r8 as DF
    import deploy8_forward_r8 as D8
    import tac_retime_r8 as TAC
    import prior_parity_r8 as PP
    return {"CP": CP, "CR": CR, "RT": RT, "TFM": TFM, "BD": BD, "DF": DF, "D8": D8, "TAC": TAC, "PP": PP}


def knots_of(fz: dict) -> list:
    """The 8 knot gains of Amendment 1 from a frozen train dict: S2 RETIME at 0.5/1/1.5/2 s, S6 RETIME at 3/4/5/6 s (exact floats)."""
    s2, s6 = fz["surfaces"]["S2"], fz["surfaces"]["S6"]
    if [float(x) for x in s2["instants_s"]] != S2_T or [float(x) for x in s6["instants_s"]] != S6_T:
        raise SystemExit(f"[pf_lane] surface instants {s2['instants_s']} / {s6['instants_s']} are not Amendment 1's")
    return [float(x) for x in s2["all"]["RETIME"]] + [float(x) for x in s6["all"]["RETIME"][2:]]


def check_deploy_pf(d: dict, fz: dict) -> list:
    """Amendment 1's STRUCTURAL checks (literal knot times and instants, S2 identity, the S6 substitution) + the knot gains equal to the REGISTERED frozen file."""
    bad = []
    kt, kg = d["knots"]["t_s"], d["knots"]["gain"]
    if kt != KNOT_T:
        bad.append(f"knot times {kt}")
    if kg != knots_of(fz):
        bad.append(f"knot gains {kg} != the registered frozen file's {knots_of(fz)}")
    s2, s6 = d["surfaces"]["S2"], d["surfaces"]["S6"]
    if s2["instants_s"] != S2_T or s6["instants_s"] != S6_T:
        bad.append("surface instants")
    if s2["RETIME_DEPLOY"] != s2["RETIME_TRAIN"] or not s2["identical_to_train"] or s2["changed_vs_train"]:
        bad.append("S2 deploy table is not identical to the S2 train table")
    if s6["RETIME_DEPLOY"] != [s2["RETIME_TRAIN"][1], s2["RETIME_TRAIN"][3]] + s6["RETIME_TRAIN"][2:]:
        bad.append("S6 deploy table != [S2@1.0, S2@2.0, S6@3, S6@4, S6@5, S6@6]")
    if [c["t_s"] for c in s6["changed_vs_train"]] != [1.0, 2.0]:
        bad.append(f"S6 changed instants {[c['t_s'] for c in s6['changed_vs_train']]} != [1.0, 2.0]")
    return bad


def configure(M: dict, pkg: Path, registered: str) -> dict:
    raw = Path(pkg) / "raw"
    for sub in ("clamp_posthoc/trees", "compliant_reread/trees", "a0k0_retime", "retime_trainfit", "retime_deploy"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    CP, CR, RT, TFM, BD, DF, D8, TAC, PP = (M[k] for k in ("CP", "CR", "RT", "TFM", "BD", "DF", "D8", "TAC", "PP"))
    tf, dep = raw / "retime_trainfit", raw / "retime_deploy"
    CP.OUT = raw / "clamp_posthoc"
    CR.OUT = raw / "compliant_reread"
    RT.OUT, RT.BANKED = raw / "a0k0_retime", raw / "a0k0_headroom"            # RT.forward reads BANKED.parent / compliant_reread / trees
    TFM.TF, TFM.FROZEN_TRAIN = tf, tf / "FROZEN_GAINS_RETIME_train.json"
    BD.TF, BD.OUT = tf, dep
    BD.SOURCE, BD.SOURCE_SHA256_FILE, BD.SOURCE_SHA256 = tf / "FROZEN_GAINS_RETIME_train.json", tf / "FROZEN_GAINS_RETIME_train.sha256", registered
    BD.DEPLOY_JSON, BD.DEPLOY_SHA256_FILE = dep / "FROZEN_GAINS_RETIME_deploy.json", dep / "FROZEN_GAINS_RETIME_deploy.sha256"
    DF.DEP, DF.TF = dep, tf
    D8.DEP, D8.TF, D8.FROZEN_SHA = dep, tf, registered
    D8.RS.DEPLOY_TABLE_PATH = dep / RS_TABLE_NAME
    TAC.DEP, TAC.PKG = dep, Path(pkg)
    PP.DEP = dep
    if (tf / "FROZEN_GAINS_RETIME_train.json").exists():
        fz = json.load(open(tf / "FROZEN_GAINS_RETIME_train.json", encoding="utf-8"))
        BD.REGISTER_KNOTS_3DP = tuple(round(g, 3) for g in knots_of(fz))
        BD.check_deploy = lambda d, _fz=fz: check_deploy_pf(d, _fz)
    return {"pkg": str(pkg), "registered_sha256": registered, "train_dir": str(tf), "deploy_dir": str(dep), "rs_table": str(dep / RS_TABLE_NAME)}


def verify_installed(pkg: Path, registered: str) -> None:
    tf = Path(pkg) / "raw" / "retime_trainfit"
    a = sha256_file(tf / "FROZEN_GAINS_RETIME_train.json") if (tf / "FROZEN_GAINS_RETIME_train.json").exists() else ""
    b = read_registered(tf / "FROZEN_GAINS_RETIME_train.sha256")
    if len(registered) != 64 or len(a) != 64 or len(b) != 64:
        raise SystemExit(f"[pf_lane] INCONCLUSIVE: registered '{registered}' installed '{a}' record '{b}' (an empty or short string is never a match)")
    if not (a == b == registered):
        raise SystemExit(f"[pf_lane] the installed table {a} / record {b} is not the REGISTERED {registered}")


def cmd_install(pkg: Path, table: Path, registered_file: Path) -> int:
    reg = read_registered(registered_file)
    src = sha256_file(table)
    if len(reg) != 64 or src != reg:
        raise SystemExit(f"[pf_lane] REFUSING: the table {table} hashes to {src}, the registration file says '{reg}'")
    M = mods()
    tf = Path(pkg) / "raw" / "retime_trainfit"
    tf.mkdir(parents=True, exist_ok=True)
    dst = tf / "FROZEN_GAINS_RETIME_train.json"
    if dst.exists() and sha256_file(dst) != reg:
        raise SystemExit(f"[pf_lane] REFUSING: {dst} exists with another table ({sha256_file(dst)}) -- a pf package carries ONE table")
    shutil.copyfile(table, dst)
    (tf / "FROZEN_GAINS_RETIME_train.sha256").write_text(reg + "\n")
    verify_installed(pkg, reg)
    cfg = configure(M, pkg, reg)
    BD, TFM, RS = M["BD"], M["TFM"], M["D8"].RS
    fz = json.load(open(dst, encoding="utf-8"))
    TFM.assert_frozen_is_train(fz)
    d = BD.build(fz)
    bad = BD.check_deploy(d)
    if bad:
        raise SystemExit("[pf_lane] the deploy table fails Amendment 1's structural checks: " + "; ".join(bad))
    BD.DEPLOY_JSON.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    BD.DEPLOY_SHA256_FILE.write_text(sha256_file(BD.DEPLOY_JSON) + "\n")
    kt, kg = BD.knots_from_frozen(fz)
    rs = {"knots_s": kt, "gains": kg, "interpolation": BD.INTERPOLATION, "source_file": str(dst), "source_sha256": reg,
          "derived_by": "pf_lane.py install: build_deploy_gains_r8.knots_from_frozen on the REGISTERED per-frame table (no fit)",
          "spec": "SHIP-R8 AMENDMENT 3 (ii) + AMENDMENT 1 table form", "generated": time.strftime("%FT%T%z")}
    rsp = Path(RS.DEPLOY_TABLE_PATH)
    rsp.write_bytes((json.dumps(rs, indent=1) + "\n").encode("utf-8"))      # LF only: the stage's CRLF-folded hash == the plain hash
    Path(str(rsp) + ".sha256").write_text(sha256_file(rsp) + "\n")
    t = RS.load_deploy_table()                                                 # the stage's own sha-verified loader must accept it
    ok = bool(t["knots_s"].tolist() == KNOT_T and t["gains"].tolist() == knots_of(fz))
    rec = {**cfg, "installed_sha256": sha256_file(dst), "deploy_json_sha256": sha256_file(BD.DEPLOY_JSON), "rs_table_sha256": t["sha256"],
           "knots": dict(zip([str(x) for x in kt], kg)), "rs_table_loads_and_matches": ok, "lane_sha256": sha256_file(__file__),
           "battery_code": {f: sha256_file(R8_CODE / f) for f in ("clamp_posthoc_r8.py", "compliant_reread_r8.py", "a0k0_headroom_retime_r8.py", "retime_trainfit_r8.py",
                                                                  "build_deploy_gains_r8.py", "deploy_forward_r8.py", "deploy8_forward_r8.py", "tac_retime_r8.py",
                                                                  "prior_parity_r8.py")},
           "stage_file": str(M["D8"].RS_FILE), "stage_sha256": sha256_file(M["D8"].RS_FILE)}
    (Path(pkg) / "PF_LANE_INSTALL.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({"installed": rec["installed_sha256"], "rs_ok": ok}), "ZZPFLANEINSTALLOKZZ" if ok else "ZZPFLANEINSTALLBADZZ", flush=True)
    return 0 if ok else 4


def run_stage(stage: str, pkg: Path, tree: str, registered_file: Path) -> int:
    reg = read_registered(registered_file)
    verify_installed(pkg, reg)
    M = mods()
    configure(M, pkg, reg)
    if stage == "clamp":
        return M["CP"].read_tree(tree, True)
    if stage == "compliant":
        return M["CR"].read_tree(tree, True)
    if stage == "tf_forward":
        return M["TFM"].cmd_forward(tree)
    if stage == "d8":
        return M["D8"].forward(tree)
    if stage == "df":
        return M["DF"].forward(tree)
    if stage == "tac":
        return M["TAC"].run(tree)
    if stage == "parity":
        sys.argv = [sys.argv[0], "--trees", tree]
        return M["PP"].main()
    raise SystemExit(f"unknown stage {stage}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("install", "clamp", "compliant", "tf_forward", "d8", "df", "tac", "parity"))
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--tree")
    ap.add_argument("--table")
    ap.add_argument("--registered", required=True)
    a = ap.parse_args()
    if a.stage == "install":
        return cmd_install(Path(a.pkg), Path(a.table), Path(a.registered))
    if not a.tree:
        ap.error("--tree is required")
    return run_stage(a.stage, Path(a.pkg), a.tree, Path(a.registered))


if __name__ == "__main__":
    sys.exit(main())
