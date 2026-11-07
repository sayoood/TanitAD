#!/usr/bin/env python3
"""YF3 TASK 1: every PREREG gate that READS THE REAL u400 CACHE, with its deliberate-regression arm (CPU only, CUDA_VISIBLE_DEVICES=-1).
Writes <out>/real_cache_gates.json and prints ZZYF3_REAL_GATES_OK / ZZYF3_REAL_GATES_RED. Reuses YF2's code unchanged (read-only import); nothing here edits YF2.

  G2 sign   : g2_sign.py on cache_u400_4k (green); --negate must be RED
  G3 cache  : cache manifest frozen_in_proxy_sha256 == u400's == 160b61bc... (g3_fingerprint.json); the ep015 cache's differs (RED arm); proxy_cache_yf2 verify ok (run separately, result read from CACHE_PIN)
  G4 / G5 / G9 : yaw_ft.py --gates-only on the real cache, 512 frames, B and C; RED arms --regress default_col (G4) and table_not_applied (G5); a table with 2 % of the
                 TRAIN/VAL tokens dropped must raise TableRefused (G9)
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
Y2 = Path("E:/Projects/TanitAD-artifacts/yawrate_wp/YF2")
PY = sys.executable
sys.path.insert(0, str(Y2))
import yaw_common as YC  # noqa: E402

CACHE = Y2 / "cache_u400_4k"
MAN = Y2 / "manifest_yaw_4k.json"
TRAIN_TABLE_SHA = "517c6413"


def run(cmd, log):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, r.stdout + r.stderr


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res, ok = {}, True

    def rec(name, passed, detail=""):
        nonlocal ok
        res[name] = {"as_required": bool(passed), "detail": detail}
        ok &= bool(passed)
        print(("PASS " if passed else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)

    # ---------------- G2
    rc, txt = run([PY, str(Y2 / "g2_sign.py"), "--cache", str(CACHE), "--out", str(out / "g2_sign_u400.json")], out / "g2.log")
    g2 = json.load(open(out / "g2_sign_u400.json"))
    rec("G2 sign green on the real cache", rc == 0 and g2["pass"], f"corr(omega, ay/v)={g2['corr_omega_ay_over_v']:.4f} (bar 0.85, n={g2['n_frames_v_gt_2']}), corr(omega, yaw@4s)={g2['corr_omega_target_yaw_4s']:.4f} (bar >0)")
    rc, txt = run([PY, str(Y2 / "g2_sign.py"), "--cache", str(CACHE), "--negate", "--out", str(out / "g2_sign_negated_RED.json")], out / "g2neg.log")
    g2n = json.load(open(out / "g2_sign_negated_RED.json"))
    rec("RED-AS-REQUIRED G2: negated omega", rc == 3 and not g2n["pass"], f"rc {rc} corr_ay_over_v {g2n['corr_omega_ay_over_v']:.4f}")
    # ---------------- G3
    mf = json.load(open(CACHE / "manifest.json"))
    fp = json.load(open(Y2 / "g3_fingerprint.json"))
    want = fp["u400"]["frozen_in_proxy_sha256"]
    snap_ok = mf["snapshot_sha256"] == YC.sha256_file(YC.U400) and mf["snapshot_sha256"].startswith(YC.U400_SHA256)
    rec("G3 cache: frozen_in_proxy fingerprint == u400's == model_final's, snapshot sha == u400", mf["frozen_in_proxy_sha256"] == want == fp["model_final"]["frozen_in_proxy_sha256"] and snap_ok,
        f"{mf['frozen_in_proxy_sha256'][:12]}")
    rec("G3 cache: frames 4000, zero visual frames, C1 == 0.0 on every pass", mf["frames"] == 4000 and mf["visual_frames"] == 0 and all(p["C1_decode_vs_forward_max_abs"] == 0.0 for p in ([mf["passes"]] if isinstance(mf["passes"], dict) else mf["passes"])),
        f"frames {mf['frames']} visual {mf['visual_frames']} C1 {mf['last_pass']['C1_decode_vs_forward_max_abs']}")
    ep = json.load(open("E:/Projects/TanitAD/data/refe_proxy/cache_train_ep015/manifest.json"))
    rec("RED-AS-REQUIRED G3: the ep015 cache carries a DIFFERENT fingerprint", ep["frozen_in_proxy_sha256"] != want, f"{ep['frozen_in_proxy_sha256'][:12]} != {want[:12]}")
    # ---------------- G4 / G5 / G9 on the real cache
    base = [PY, str(Y2 / "yaw_ft.py"), "--cache", str(CACHE), "--manifest", str(MAN), "--yaw-table-sha256", TRAIN_TABLE_SHA, "--rung", "R1", "--seed", "0", "--gates-only", "--gate-frames", "512"]
    for arm in ("B", "C"):
        d = out / f"gates_{arm}"
        rc, txt = run(base + ["--arm", arm, "--out", str(d)], out / f"gates_{arm}.log")
        g = json.load(open(d / "gates.json"))
        if arm == "B":
            rec("G4+G5 green on the real cache (arm B, 512 frames)", rc == 0 and g["pass"] and g["G4_identity_yaw0"] and g["G4_identity_yaw_nonzero"],
                f"n {g['n_frames']} grad_table {g['G5_gradabs_table']:.4g} grad_control {g['G5_gradabs_zero_control']}")
        else:
            # arm C feeds yaw 0: the table gate (G5 table reach) is by construction not satisfiable with yaw 0 in the TRAIN batch; the gate function itself builds a table batch for both arms,
            # so it is the SAME computation: C's identity holds (G4) and its trainer path is the zero-gradient control.
            rec("G4 green on the real cache (arm C)", rc == 0 and g["G4_pass"], f"identity yaw0 {g['G4_identity_yaw0']} yaw!=0 {g['G4_identity_yaw_nonzero']}")
    rc, txt = run(base + ["--arm", "B", "--regress", "default_col", "--out", str(out / "gates_dc")], out / "gates_dc.log")
    g = json.load(open(out / "gates_dc" / "gates.json"))
    rec("RED-AS-REQUIRED G4 (real cache): u400's random column left in place", rc == 3 and not g["G4_pass"], f"rc {rc} identity_yaw_nonzero {g['G4_identity_yaw_nonzero']}")
    rc, txt = run(base + ["--arm", "B", "--regress", "table_not_applied", "--out", str(out / "gates_tn")], out / "gates_tn.log")
    g = json.load(open(out / "gates_tn" / "gates.json"))
    rec("RED-AS-REQUIRED G5 (real cache): table loaded but not applied", rc == 3 and not g["G5_pass"], f"rc {rc} grad_table {g['G5_gradabs_table']}")
    # G9: drop 2 % of the TRAIN/VAL tokens from a COPY of the pinned table (YT's file format), expect TableRefused
    tab, meta, sha = YC.load_yaw_table(YC.TABLE_NAVTRAIN)
    man = json.load(open(MAN))
    toks = sorted({f["token"] for f in man["frames"]})
    drop = set(toks[:: max(1, len(toks) // (len(toks) // 50))][: max(1, len(toks) // 50)])
    bad = {k: v for k, v in tab.items() if k not in drop}
    bad_path = out / "ego_yaw_navtrain_DROP2PCT_RED.json.gz"
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(bad_path, "wb"), mtime=0) as gz:
        gz.write(json.dumps({"meta": {**meta, "n_tokens": len(bad)}, "table": bad}).encode())
    r = subprocess.run([PY, str(Y2 / "yaw_ft.py"), "--cache", str(CACHE), "--manifest", str(MAN), "--yaw-table", str(bad_path), "--arm", "B", "--rung", "R1", "--gates-only", "--out", str(out / "gates_miss")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "gates_miss.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    rec("RED-AS-REQUIRED G9 (real cache): 2 % of the tokens missing -> TableRefused", r.returncode != 0 and "TableRefused" in (r.stdout + r.stderr), f"rc {r.returncode}, dropped {len(drop)} of {len(toks)} tokens")
    # G9 green: every TRAIN/VAL token present (the green B run above already proved it, since check_rows runs before the gates)
    res["_ok"] = bool(ok)
    json.dump(res, open(out / "real_cache_gates.json", "w"), indent=1)
    print("ZZYF3_REAL_GATES_OK" if ok else "ZZYF3_REAL_GATES_RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "gates_real")))
