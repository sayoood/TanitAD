"""MM 2026-11-16: derive lattok_ctrl.py from lattok_analysis.py (the frozen LAT-TOK run's own code) for the admissibility control.

kstep's two readouts (RA = ridge on D, RY = MLP pair) on the BANKED z1 (z1b, box-2 bytes) and on z1 PROJECTED from the pod-cache token mean (z1p = wm.z1_from_mean(set a)).
Checks, fixed before the run (MM ruling in LOOP_STATE, ~14:25Z), each per chassis, tolerance +/- 0.02 R2:
  lin.z1b.acc.r2 - kstep r2_readout_acc ; mlp.z1b.yaw.r2 - kstep r2_readout_yaw ; lin.z1p-z1b.acc.dR2 ; mlp.z1p-z1b.yaw.dR2.
All within -> the LAT-TOK verdict on the pod cache is ADMISSIBLE; else NOT-ADMISSIBLE (box-2 clip bytes next)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


sub('''    sets = {"z1": (torch.stack(ZB1).numpy().astype(np.float64), torch.stack(ZB2).numpy().astype(np.float64))}
    for s, sl in SL.items():
        sets[s] = (np.asarray(feat[IT][:, sl], dtype=np.float32), np.asarray(feat[IK][:, sl], dtype=np.float32))
''', '''    assert wm.enc1 is None, "z1_from_mean per frame needs enc1 None (wm_k 1)"
    with torch.no_grad():
        zp1 = wm.z1_from_mean(torch.from_numpy(np.asarray(feat[IT][:, SL["a"]], dtype=np.float32))[:, None, :])[:, 0].double().numpy()
        zp2 = wm.z1_from_mean(torch.from_numpy(np.asarray(feat[IK][:, SL["a"]], dtype=np.float32))[:, None, :])[:, 0].double().numpy()
    zb1, zb2 = torch.stack(ZB1).numpy().astype(np.float64), torch.stack(ZB2).numpy().astype(np.float64)
    ctrl["z1_pod_vs_banked_rel_l2"] = float(np.linalg.norm(zp1 - zb1) / np.linalg.norm(zb1))
    ctrl["kstep_RA"], ctrl["kstep_RY"] = kst["heldout"]["r2_readout_acc"]["point"], kst["heldout"]["r2_readout_yaw"]["point"]
    sets = {"z1b": (zb1, zb2), "z1p": (zp1, zp2)}
''')
sub('            if s != "z1":\n', '            if True:\n')
sub('                if s == "z1" and r == "mlp":\n                    continue\n', '')
sub('''        for r in ("lin", "mlp"):
            for s in ("b", "c"):
                for nm in ("yaw", "acc"):
                    out[f"{r}.{s}-a.{nm}.dR2"] = out[f"{r}.{s}.{nm}.r2"] - out[f"{r}.a.{nm}.r2"]
''', '''        for r in ("lin", "mlp"):
            for nm in ("yaw", "acc"):
                out[f"{r}.z1p-z1b.{nm}.dR2"] = out[f"{r}.z1p.{nm}.r2"] - out[f"{r}.z1b.{nm}.r2"]
''')
i = s.index("\nR = {}\n")
s = s[:i] + '''
R = {}
for k in (0, 1):
    R[f"R35_s{k}"] = run_chassis(k)
chk = {}
for tag, res in R.items():
    c = res["_ctrl"]
    chk[tag] = {"lin_z1b_acc_minus_kstep_RA": res["lin.z1b.acc.r2"]["point"] - c["kstep_RA"],
                "mlp_z1b_yaw_minus_kstep_RY": res["mlp.z1b.yaw.r2"]["point"] - c["kstep_RY"],
                "lin_acc_pod_minus_banked": res["lin.z1p-z1b.acc.dR2"]["point"],
                "mlp_yaw_pod_minus_banked": res["mlp.z1p-z1b.yaw.dR2"]["point"],
                "z1_pod_vs_banked_rel_l2": c["z1_pod_vs_banked_rel_l2"]}
ok = all(abs(v) <= 0.02 for d in chk.values() for kk, v in d.items() if kk != "z1_pod_vs_banked_rel_l2")
verdict = "ADMISSIBLE" if ok else "NOT-ADMISSIBLE"
json.dump({"what": "LAT-TOK admissibility control (MM 2026-11-16): kstep readouts on banked z1 vs z1 projected from the pod cache", "tolerance_r2": 0.02,
           "checks": chk, "verdict": verdict, "results": R, "wall_s": time.time() - T0}, open(OUT, "w"), indent=1)
log("CTRL", verdict, json.dumps(chk))
'''
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", dst)
