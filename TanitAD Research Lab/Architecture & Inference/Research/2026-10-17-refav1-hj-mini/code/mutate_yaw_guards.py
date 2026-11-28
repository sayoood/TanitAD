"""MUTATE, do not inspect (CLAUDE.md, "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER", rule 3).  PI ruling D-PI-YAWRATE-1.

Each mutation re-introduces ONE way the steer (or a future pose) could reach the model's inputs in the compliant ``yaw`` mode, by editing
the SOURCE of the compliant path in a checkout; the guard tests named for it MUST then fail.  A mutation whose guard stays green means
that guard does not guard.  The file is restored byte-for-byte after every mutation (and verified).

    python mutate_yaw_guards.py <tree> <out.json>
"""
import json
import os
import subprocess
import sys

TREE, OUT = sys.argv[1], sys.argv[2]
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=os.pathsep.join([os.path.join(TREE, "stack"), os.path.join(TREE, "taniteval")]))
T = "stack/tests/test_yaw_rate_ruling.py"

MUTATIONS = [
    dict(id="M1-ra1-loader-reads-steer", file="stack/tanitad/data/ra1_pooled.py",
         old="ch1 = torch.from_numpy(np.nan_to_num(omega, nan=0.0).astype(np.float32))", new='ch1 = o["steer"].float()',
         guard="test_ra1_yaw_clip_is_bit_identical_under_garbage_steer_and_steer_mode_is_not or test_ra1_model_loss_is_bit_identical"),
    dict(id="M2-v8f-windows-read-steer-in-act", file="stack/tanitad/data/v8f_mini_data.py",
         old='"ch1": np.nan_to_num(omega, nan=0.0).astype(np.float32), "v": o["v"].numpy()}', new='"ch1": o["steer"].numpy(), "v": o["v"].numpy()}',
         guard="test_v8f_mini_yaw_windows_ignore_the_steer_and_the_steer_mode_does_not"),
    dict(id="M3-v8f-base-integrates-the-cache-kin3", file="stack/tanitad/data/v8f_mini_data.py",
         old='d["base"] = kdx_base(d["kin3"], KIN_UNITS["yaw"])', new='d["base"] = kdx_base(o["kin3"].numpy().astype(np.float64))',
         guard="test_v8f_mini_yaw_windows_ignore_the_steer_and_the_steer_mode_does_not"),
    dict(id="M4-p10-features-from-the-cache", file="stack/tanitad/data/ra1_pooled.py",
         old='        p10f[1:] = torch.from_numpy(D._fk().kin_features(kin3[1:]).astype(np.float32))',
         new='        p10f = o["p10"].float()',
         guard="test_ra1_yaw_clip_is_bit_identical_under_garbage_steer_and_steer_mode_is_not"),
    dict(id="M5-yaw-read-one-frame-late", file="stack/tanitad/data/v8f_mini_data.py",
         old="    ch = ego_channels_native(track)[0::2]", new="    ch = np.roll(ego_channels_native(track), -1, axis=0)[0::2]",
         guard="test_yaw_spelling_equals_ego_channels_dyaw_on_real_windows_and_a_one_frame_shift_fails or test_yaw_mode_is_causal_and_the_forward_difference_is_not or test_kappa0_kin3_equals_pose_kin3_the_p10_pose_triple"),
    dict(id="M6-window-kin-yaw-reads-actions", file="stack/tanitad/data/v8f_mini_data.py",
         old='    if kin_mode == "yaw":\n        kin = yaw_kin_from_track(poses)[0][ts]', new='    if False:\n        kin = yaw_kin_from_track(poses)[0][ts]',
         guard="test_window_kin_yaw_never_reads_actions_and_steer_mode_does"),
    dict(id="M7-sens-reads-the-steer-channel-name", file="stack/tanitad/models/ra1_hjepa.py",
         old='chans = {"all": [0, 1, 2], "a": [0], model.act_channels[1]: [1], "v": [2]}', new='chans = {"all": [0, 1, 2], "a": [0], "steer": [1], "v": [2]}',
         guard="test_eval_and_verdict_end_to_end_in_the_compliant_mode_and_mixing_is_refused"),
    dict(id="M8-kappa0-lost-the-curvature-information", file="stack/tanitad/data/v8f_mini_data.py",
         old="    kin = np.stack([ch[:, 0], ch[:, 1], kap], 1)\n    kin[0] = np.nan\n    return kin, ch[:, 2].copy()",
         new="    kin = np.stack([ch[:, 0], ch[:, 1], 0.0 * kap], 1)\n    kin[0] = np.nan\n    return kin, ch[:, 2].copy()",
         guard="test_kappa0_retains_the_recorded_curvature_information or test_kappa0_kin3_equals_pose_kin3_the_p10_pose_triple"),
    dict(id="M10-ra1-action-channel-keeps-the-NaN-of-an-undefined-step", file="stack/tanitad/data/ra1_pooled.py",
         old="ch1 = torch.from_numpy(np.nan_to_num(omega, nan=0.0).astype(np.float32))", new="ch1 = torch.from_numpy(omega.astype(np.float32))",
         guard="test_train_style_tracks_with_nan_lead_rows_give_a_finite_action_vector_and_loss"),
    dict(id="M11-v8f-action-channel-keeps-the-NaN-of-an-undefined-step", file="stack/tanitad/data/v8f_mini_data.py",
         old='"ch1": np.nan_to_num(omega, nan=0.0).astype(np.float32), "v": o["v"].numpy()}', new='"ch1": omega.astype(np.float32), "v": o["v"].numpy()}',
         guard="test_train_style_tracks_with_nan_lead_rows_give_a_finite_action_vector_and_loss"),
    dict(id="M9-banked-artifact-vs-loader-wrong-unit", file="stack/tanitad/data/ra1_pooled.py",
         old='base = D.kdx_base(kin3, D.KIN_UNITS["yaw"])', new="base = D.kdx_base(kin3)",
         guard="test_banked_p10_yaw_artifact_equals_what_the_ra1_loader_computes"),
]


def run(expr):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", T, "-k", expr], cwd=TREE, env=ENV, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1] if r.stdout.strip() else ""
    return r.returncode, tail


res = []
base_rc, base_tail = run(" or ".join(sorted({m["guard"] for m in MUTATIONS}, key=len)))
res.append(dict(id="BASELINE (no mutation, every guard) must be GREEN", rc=base_rc, tail=base_tail, ok=base_rc == 0))
for m in MUTATIONS:
    path = os.path.join(TREE, m["file"])
    orig = open(path, "rb").read()
    txt = orig.decode("utf-8")
    ok_apply = txt.count(m["old"]) == 1
    rec = dict(id=m["id"], file=m["file"], applied=ok_apply, guard=m["guard"])
    if ok_apply:
        try:
            open(path, "wb").write(txt.replace(m["old"], m["new"]).encode("utf-8"))
            rc, tail = run(m["guard"])
            rec.update(rc=rc, tail=tail, went_red=rc != 0)
        finally:
            open(path, "wb").write(orig)
        assert open(path, "rb").read() == orig, "restore failed"
    res.append(rec)
bad = [r["id"] for r in res[1:] if not r.get("went_red")] + ([] if res[0]["ok"] else ["BASELINE"])
json.dump(dict(results=res, all_guards_bite=not bad, failures=bad), open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("ALL GUARDS BITE" if not bad else f"GUARDS THAT DID NOT BITE / BASELINE RED: {bad}")
sys.exit(1 if bad else 0)
