"""Does STEP 1 -> STEP 2 -> STEP 3 actually work? EXECUTED, not read.

Runs the REAL producer (`scripts/v6_dump_sw_latents.collect_latents`) over a REAL
`V6Stack` and a REAL `FlagshipWindowDataset` whose 2 s endpoints carry a PLANTED
sigma, then hands the producer's own output to the REAL estimator
(`e_wc2_sigma_star.run`) and the REAL chain reader (`v6_chain.read_sw_admission`
-> `assert_selector_admissible`).

⛔ Nothing here hand-writes a dump. C94 — the defect this whole stream sits on —
was a fixture that modelled the CONSUMER'S EXPECTATION instead of the PRODUCER'S
OUTPUT, so the join was never exercised. The construction is imported from
`stack/tests/test_v6_dump_sw_latents.py` so the probe and the pin cannot drift.

Run (dev box, CPU, ~25 s):
    PYTHONUTF8=1 python code/sw_dumper_roundtrip_probe.py > raw/sw_dumper_roundtrip.json
"""
import contextlib
import json
import pathlib
import sys
import time

import pyarrow  # noqa: F401  (must precede torch on this box)

REPO = pathlib.Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "tests"))

import torch  # noqa: E402

import e_wc2_sigma_star as E  # noqa: E402
import v6_chain as C  # noqa: E402
import v6_dump_sw_latents as D  # noqa: E402
import test_v6_dump_sw_latents as T  # noqa: E402

PLANTED = (0.30, 1.10, 2.00)


def main() -> int:
    # ⚠️ the producer logs progress to stdout; this file's stdout IS the
    # artifact, so the whole compute runs with stdout redirected to stderr and
    # only the JSON is printed at the end.
    t0 = time.time()
    torch.manual_seed(0)
    stack = T.V6Stack(T.tiny_cfg()).eval()
    eps = [T._episode(e) for e in range(T.N_EP)]
    eps[0] = T._episode(0, T=T.T_FRAMES + 8)
    ds = T._dataset(eps)
    grid = D.select_grid(ds, episodes=T.N_EP)

    out = {
        "_what": "STEP 1 (v6_dump_sw_latents) -> STEP 2 (e_wc2_sigma_star) -> "
                 "STEP 3 (v6_chain.read_sw_admission / "
                 "assert_selector_admissible), executed end-to-end on a "
                 "PLANTED sigma. No dict is hand-written anywhere.",
        "_evidence_class": "MEASURED (ours) — CPU only, no GPU, no checkpoint, "
                           "no corpus. Thor was never touched by this probe.",
        "surface": {"episodes": T.N_EP, "windows": len(grid),
                    "grid": {"window": D.WINDOW, "stride": D.STRIDE,
                             "k_max": D.K_MAX_GRID},
                    "d_op": int(stack.cfg.d_op),
                    "d_str": int(stack.cfg.d_str),
                    "model_window": int(stack.cfg.predictor.window),
                    "features": ["pooled", "ctx"]},
        "thresholds": {k: C.SW_LATENT_ADMISSION[k] for k in
                       ("field", "funded_at_or_below_m", "refused_above_m",
                        "pre_registered")},
        "rows": [],
    }

    tmp = pathlib.Path(__import__("tempfile").mkdtemp(prefix="swdump"))
    for sigma in PLANTED:
        with contextlib.redirect_stdout(sys.stderr):
            d = T._plant_sigma(stack, ds, grid, sigma, seed=int(sigma * 100))
        res = E.run(d, features=["pooled", "ctx"], n_boot=0)

        root = tmp / f"s{sigma}" / "experiments"
        cfg = C.ChainConfig(root=str(root).replace("\\", "/"))
        sw = pathlib.Path(cfg.path(cfg.sw_dir))
        sw.mkdir(parents=True, exist_ok=True)
        (sw / C.SW_LATENT_ADMISSION["artifact"]).write_text(
            json.dumps(res, indent=1, default=float), encoding="utf-8")
        adm = C.read_sw_admission(cfg)

        step = C.step_by_key(C.build_plan(C.ChainConfig(root=cfg.root,
                                                        st_arms=("goal",))),
                             "S-T:goal")
        try:
            C.assert_selector_admissible(step, cfg)
            refusal = None
        except C.ChainRefusal as exc:
            refusal = str(exc).splitlines()[0]

        got = res["references_and_ratios"]["sigma_perax_2s_m"]
        out["rows"].append({
            "planted_sigma_perax_m": sigma,
            "recovered_sigma_perax_m": got,
            "recovery_error_pct": round(100.0 * (got - sigma) / sigma, 2),
            "producer_instrument_fail": d["instrument_fail"],
            "producer_has_sel": "sel" in d,
            "estimator_own_verdict": res["decision"]["verdict"],
            "estimator_refusal_reasons": res["decision"]["refusal_reasons"],
            "read_at": adm.get("read_at"),
            "chain_verdict": adm.get("verdict"),
            "admits_a_selector_arm": adm.get("admits_a_selector_arm"),
            "selector_launch_refused": refusal is not None,
            "refusal_first_line": refusal,
            "sigma_long_m": res["sigma"]["2s"]["sigma_long_m"],
            "sigma_lat_m": res["sigma"]["2s"]["sigma_lat_m"],
            "sigma_radial_rms_m": res["sigma"]["2s"]["sigma_radial_rms_m"],
            "r2_oof": {k: res["sigma"]["2s"]["axes"][k]["r2_oof"]
                       for k in ("long", "lat")},
            "n_windows_2s": res["sigma"]["2s"]["n_windows"],
            "n_windows_6s": res["sigma"]["6s"]["n_windows"],
        })

    # the producer's own controls, from the LAST dump (identical across sigmas)
    out["producer_controls"] = D._jsonable(d["controls"])
    # the emitted step-1 command, as an operator would receive it
    rec = C.sw_admission_recipe(C.ChainConfig())
    out["emitted_recipe"] = {str(s["n"]): {"status": s["status"],
                                           "cmd": s.get("cmd")}
                             for s in rec["steps"]}
    out["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
