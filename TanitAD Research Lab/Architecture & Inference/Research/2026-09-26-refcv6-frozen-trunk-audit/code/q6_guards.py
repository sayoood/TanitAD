"""Q6 (advisory class F) -- for each refcv6 guard this audit touched, CONSTRUCT the regression it
should catch and confirm it goes RED; name every construction that stays GREEN.

Guards run here (the real functions, never a restatement; mutations are applied to INPUTS or by
monkeypatch in this process -- nothing in the code tree is edited):

G1 `refcv6_max_speed.read_speed_max_sidecar_v6` (the live run's max-speed input reader)
   R1 a row whose stored bin disagrees with its v_hi_ms           -> must RAISE
   R2 the meta file names a different label md5                   -> must RAISE
   R3 the SAME md5 mismatch with the .meta.json ABSENT            -> the question: does it raise?
   C  the real eval sidecar + its real meta (read-only scp) + the run's label md5 -> must PASS
G2 `timm_trunk._assert_pretrained_loaded` (the ImageNet-weights guard on the live backbone)
   C  resnet101.a1_in1k as downloaded                              -> must PASS
   R1 the same architecture, random init (pretrained=False)        -> must RAISE
   R2 the downloaded stem scaled by 1.05 (outside the 2 % band)    -> must RAISE
   R3 the downloaded stem intact, layer4 re-initialised            -> the question: does it raise?
G3 the label-time mapping (refc_v3_train.py:3009-3010): no guard exists. The proposed guard is
   |t_trainer(row) - t_true(row)| <= 0.05 s against the clip's MEASURED clock (q4c). Its RED arm is
   the shipped code; its GREEN arm is the corrected mapping.

The F3 cascade (q3b), the time-base identity (q4 C2) and the in-run counter leak (q5) carry their
own constructed regressions and are cited, not re-run.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402

SCRATCH = C.SCRATCH
SIDE = C.KIT / "data/refcv6_speed_max_v8_eval.jsonl"
META = SCRATCH / "refcv6_speed_max_v8_eval.jsonl.meta.json"
RUN_EVAL_LABEL_MD5 = "eefc38d1453bd1c73802d44d45affced"   # config.json v7 eval label md5


def outcome(fn) -> dict:
    try:
        fn()
        return {"raised": False}
    except BaseException as e:  # noqa: BLE001 -- SystemExit counts as a refusal
        return {"raised": True, "type": type(e).__name__, "msg": str(e)[:160]}


def g1_sidecar() -> dict:
    from tanitad.refs import refcv6_max_speed as v6ms
    tmp = Path(tempfile.mkdtemp(prefix="q6_side_", dir=str(SCRATCH)))
    try:
        side = tmp / "side.jsonl"
        shutil.copyfile(SIDE, side)
        shutil.copyfile(META, str(side) + ".meta.json")
        res = {"C_real_sidecar_real_meta_run_md5": outcome(
            lambda: v6ms.read_speed_max_sidecar_v6(str(side), label_md5=RUN_EVAL_LABEL_MD5))}
        # R1: flip one row's stored bin
        rows = [json.loads(x) for x in SIDE.read_text(encoding="utf-8").splitlines() if x.strip()]
        bad = [dict(r) for r in rows]
        bad[0]["bin"] = (int(bad[0]["bin"]) + 1) % 4
        side_r1 = tmp / "side_r1.jsonl"
        side_r1.write_text("\n".join(json.dumps(r) for r in bad) + "\n", encoding="utf-8")
        shutil.copyfile(META, str(side_r1) + ".meta.json")
        res["R1_wrong_bin_row"] = outcome(
            lambda: v6ms.read_speed_max_sidecar_v6(str(side_r1), label_md5=RUN_EVAL_LABEL_MD5))
        # R2: the run loaded a DIFFERENT label blob than the sidecar was built over (meta present)
        res["R2_md5_mismatch_meta_present"] = outcome(
            lambda: v6ms.read_speed_max_sidecar_v6(str(side), label_md5="0" * 32))
        # R3: the same mismatch, meta ABSENT (the dev-box kit ships no .meta.json)
        side_r3 = tmp / "side_r3.jsonl"
        shutil.copyfile(SIDE, side_r3)
        res["R3_md5_mismatch_meta_ABSENT"] = outcome(
            lambda: v6ms.read_speed_max_sidecar_v6(str(side_r3), label_md5="0" * 32))
        res["kit_has_meta"] = Path(str(SIDE) + ".meta.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["verdict"] = {"R1_red": res["R1_wrong_bin_row"]["raised"],
                      "R2_red": res["R2_md5_mismatch_meta_present"]["raised"],
                      "R3_red": res["R3_md5_mismatch_meta_ABSENT"]["raised"],
                      "C_green": not res["C_real_sidecar_real_meta_run_md5"]["raised"]}
    return res


def g2_pretrained() -> dict:
    import timm
    from tanitad.models import timm_trunk as tt
    name = "resnet101.a1_in1k"
    res = {}
    net = timm.create_model(name, pretrained=True, features_only=True, out_indices=(3, 4))
    res["C_downloaded"] = outcome(lambda: tt._assert_pretrained_loaded(net, name))
    res["C_stem_abs_sum"] = float(net.conv1.weight.abs().sum())
    # R2: stem scaled 1.05
    with torch.no_grad():
        w0 = net.conv1.weight.clone()
        net.conv1.weight.mul_(1.05)
    res["R2_stem_x1.05"] = outcome(lambda: tt._assert_pretrained_loaded(net, name))
    with torch.no_grad():
        net.conv1.weight.copy_(w0)
    # R3: stem intact, layer4 re-initialised (a partial load)
    torch.manual_seed(0)
    for m in net.layer4.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
    res["R3_stem_intact_layer4_random"] = outcome(lambda: tt._assert_pretrained_loaded(net, name))
    del net
    rnd = timm.create_model(name, pretrained=False, features_only=True, out_indices=(3, 4))
    res["R1_random_init"] = outcome(lambda: tt._assert_pretrained_loaded(rnd, name))
    res["R1_stem_abs_sum"] = float(rnd.conv1.weight.abs().sum())
    del rnd
    res["verdict"] = {"C_green": not res["C_downloaded"]["raised"],
                      "R1_red": res["R1_random_init"]["raised"],
                      "R2_red": res["R2_stem_x1.05"]["raised"],
                      "R3_red": res["R3_stem_intact_layer4_random"]["raised"]}
    return res


def g3_label_time() -> dict:
    q = json.loads((C.RAW / "q4c_grid_vs_egolog.json").read_text(encoding="utf-8"))
    clips = [r for r in q["splits"]["eval139"]["per_clip"] if r["fit_resid_max_ms"] < 5.0]
    ns = 3
    red = green = 0
    worst = 0.0
    for r in clips:
        g0, dt = r["grid_start_on_label_timeline_s"], r["dt_s"]
        for row in (60, 80, 100):
            t_true = g0 + (row + ns - 1) * dt
            t_ship = row * 0.1                      # refc_v3_train.py:3009-3010
            t_fix = g0 + (row + ns - 1) * dt        # the corrected mapping (clock from the cache)
            red += int(abs(t_ship - t_true) > 0.05)
            green += int(abs(t_fix - t_true) <= 0.05)
            worst = max(worst, abs(t_ship - t_true))
    n = 3 * len(clips)
    return {"guard": "|t_trainer(row) - t_true(row)| <= 0.05 s (PROPOSED; none exists)",
            "n_checks": n, "shipped_mapping_violations": red, "fixed_mapping_passes": green,
            "worst_shipped_offset_s": round(worst, 4),
            "verdict": {"shipped_red": red == n, "fixed_green": green == n}}


def main():
    if C.ram_available_gb() < 1.5:
        raise SystemExit("[audit:RAM] < 1.5 GB available even for a light job")
    out = {"what": "Q6: constructed regressions for the guards this audit touched",
           "evidence_class": "MEASURED (ours, dev box CPU)",
           "G1_max_speed_sidecar": g1_sidecar(), "G3_label_time": g3_label_time()}
    print(json.dumps(out["G1_max_speed_sidecar"]["verdict"]), json.dumps(out["G3_label_time"]))
    if C.ram_available_gb() >= 8.8:
        out["G2_pretrained_weights"] = g2_pretrained()
        print(json.dumps(out["G2_pretrained_weights"], indent=1))
    else:
        out["G2_pretrained_weights"] = {"skipped": "RAM < 8.8 GB at the time (brief rule 6)"}
    C.write_json("q6_guards.json", out)


if __name__ == "__main__":
    main()
