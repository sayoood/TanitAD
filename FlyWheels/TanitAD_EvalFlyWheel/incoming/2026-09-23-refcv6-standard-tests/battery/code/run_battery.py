"""run_battery.py -- the refcv6 four-family held-out battery, ONE command (SPEC.md §4).

    python run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step5000.pt \
        [--config D:/refcv6_eval_kit/ckpt/config.json] [--infer-seeds 0,1] [--tag step5000]

Steps (each banks its artifact before the next starts):
  0. SPEC sha256 + dev-box GPU gate (wait, re-check every 60 s).
  1. G0 (`reproduce_inrun_eval.py`) when `--metrics` has an eval row at the checkpoint's step and
     `--no-g0` is not given. A G0 that does not PASS stops the battery (SPEC §2).
  2. Per inference seed: the T1 roll (`refcv6_roll`, i.e. refcv3_arm.run_dump fed like the
     trainer) on S2 = the banked baseline windows restricted to the kit clips.
  3. Per seed: the panel dump (+ STOP + the banked baselines, pairing verified), the analysis
     (refcv3_arm.analyze_refcv3), cross-model paired families, the refcv6 tactical decoder
     metrics and the frozen acceptance instruments.
  4. The inference-seed replicate, the S6 (6 s) read and the pre-registered bars.
Everything lands in `<out-root>/<tag>/`. Assert on the JSON artifacts, never on an exit code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import refcv6_roll as RR  # noqa: E402
import refcv6_panel as P  # noqa: E402
import gpu_gate  # noqa: E402

KIT_EVAL = str(L.KIT / "data/refcv6-b1-416x1024-eval139")
KIT_LABELS = str(L.KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
BARS = [
    {"id": "BAR-R6-1", "surface": "S2", "a": "ha0_ext", "b": "os", "metric": "ade_m",
     "statement": "refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0"},
    {"id": "BAR-R6-2", "surface": "S2", "a": "ha", "b": "os", "metric": "ade_m",
     "statement": "refcv6 beats HOLD-ACTION: os - ha < 0, separated"},
    {"id": "BAR-R6-3", "surface": "S2", "a": "b_refcv4b", "b": "os", "metric": "ade_m",
     "statement": "refcv6 beats refcv4b: os - refcv4b.os < 0, separated"},
    {"id": "BAR-R6-4", "surface": "S2", "a": "b_refcv5v2_s0", "b": "os", "metric": "ade_m",
     "statement": "refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated"},
    {"id": "BAR-R6-5", "surface": "S6", "a": "ha0_ext", "b": "os", "metric": "ade_m",
     "statement": "at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated"},
]
PAIRS = [("ha0_ext", "os", "os_minus_ha0ext"), ("ha", "os", "os_minus_ha"),
         ("ha0", "os", "os_minus_ha0__CV"), ("stop", "os", "os_minus_stop"),
         ("b_refcv4b", "os", "os_minus_refcv4b"), ("b_refcv5v2_s0", "os", "os_minus_refcv5v2_s0"),
         ("b_refcv5v2_s1", "os", "os_minus_refcv5v2_s1"),
         ("os", "os_navzero", "navzero_minus_os"), ("os", "os_navshuf", "navshuf_minus_os"),
         ("os", "os_vmaxzero", "vmaxzero_minus_os"), ("ha0_ext", "os_navzero", "navzero_minus_ha0ext"),
         ("ha0_ext", "b_refcv4b", "refcv4b_minus_ha0ext"),
         ("ha0_ext", "b_refcv5v2_s0", "refcv5v2_s0_minus_ha0ext"),
         # controls that MUST read exactly 0 are checked in the pairing record (bit identity)
         ]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gate_wait(out_json):
    """⛔ The parent NEVER holds a CUDA context (every GPU stage is a child process), and it still
    passes its own pid, so it can never wait on itself (the 2026-09-24 self-deadlock)."""
    t0 = time.time()
    while True:
        g = gpu_gate.gate(self_pid=os.getpid())
        if g["ok"]:
            json.dump(g, open(out_json, "w"), indent=1)
            return g
        print(f"[battery] GPU gate WAIT: {json.dumps(g)}", flush=True)
        if time.time() - t0 > 12 * 3600:
            json.dump(g, open(out_json, "w"), indent=1)
            raise SystemExit("[battery] GPU gate did not pass within 12 h")
        time.sleep(60)


def roll_one(ckpt, config, seed, dump_dir, windows, log, device="cuda", frame_memo=True):
    st = RR.install_patches(config_path=config, windows=windows, frame_memo=frame_memo)
    R = RR.ra3()
    os.makedirs(dump_dir, exist_ok=True)
    argv = ["--ckpt", ckpt, "--config", config, "--episodes", KIT_EVAL, "--labels", KIT_LABELS,
            "--nav-source", "v72", "--grid", "2s", "--action-units", "steer",
            "--window-stride", "1", "--with-oracle-sel", "--with-navflip",
            "--dump-dir", dump_dir, "--dump-only", "--out", os.path.join(dump_dir, "_unused.json"),
            "--device", device, "--infer-seed", str(seed), "--seed", "0", "--arm", "refcv6",
            "--lru", "8"]
    t0 = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    R.main(argv)
    man = json.load(open(os.path.join(dump_dir, "manifest.json"), encoding="utf-8"))
    ex_path = os.path.join(dump_dir, "refcv6_extras.npz")
    meta = RR.save_extras(st, ex_path, [e for e in _clip_ids_by_index(KIT_EVAL)])
    rec = {"seed": seed, "dump_dir": dump_dir, "wall_s": round(time.time() - t0, 1),
           "n_windows": man["grid"]["n_windows"], "n_episodes": man["grid"]["n_episodes"],
           "skipped": man["grid"]["n_windows_skipped"], "extras": meta,
           "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                             if torch.cuda.is_initialized() else None),
           "model_record": {k: st["model_record"][k] for k in ("state_dict", "param_breakdown",
                                                                "anchor_file_vs_ckpt_buffers",
                                                                "trunk_memory_levers_built",
                                                                "departures")},
           "window_restriction": st.get("window_restriction"),
           "frame_memo": (None if st.get("memo") is None else
                          {"hits": st["memo"].hits, "misses": st["memo"].misses})}
    del st
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return rec


def _clip_ids_by_index(kit_eval):
    from tanitad.data.v2_dataset import load_or_build_manifest
    return [str(c) for c in load_or_build_manifest(kit_eval, verbose=False)["clip_id"]]


def bar_verdicts(cells_by_seed: dict, s6_by_seed: dict, is_milestone: bool) -> list:
    out = []
    for bar in BARS:
        v = dict(bar)
        per = {}
        for s, cells in (cells_by_seed if bar["surface"] == "S2" else s6_by_seed).items():
            nm = next((n for a, b, n in PAIRS if a == bar["a"] and b == bar["b"]), None)
            c = (((cells or {}).get("pairs") or {}).get(nm) or {}).get("families", {})
            c = (c.get("ADE") or {}).get(bar["metric"])
            if c is None:
                per[s] = {"status": "ABSENT"}
                continue
            ok = (c["delta"] < 0) and (c["hi"] < 0)
            per[s] = {"delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "separated": c["separated"],
                      "n_windows": c.get("n_windows"), "n_episodes": c.get("n_episodes"),
                      "pass": bool(ok)}
        v["per_inference_seed"] = per
        if not is_milestone:
            v["verdict"] = "NOT EVALUATED (pipeline validation checkpoint; SPEC §3.6)"
        elif not per or any(x.get("status") == "ABSENT" for x in per.values()):
            v["verdict"] = "NOT EVALUABLE (a cell is absent)"
        else:
            v["verdict"] = "PASS" if all(x["pass"] for x in per.values()) else "FAIL"
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(L.KIT / "ckpt/config.json"))
    ap.add_argument("--infer-seeds", default="0,1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default=str(HERE.parent / "raw"))
    ap.add_argument("--metrics", default=None, help="metrics.jsonl for G0 (read-only copy)")
    ap.add_argument("--no-g0", action="store_true")
    ap.add_argument("--g0-json", default=None,
                    help="reuse an EXISTING G0 artifact for this checkpoint (its md5 must match)")
    ap.add_argument("--g0-a1-json", default=None,
                    help="reuse an EXISTING G0-A1 artifact (SPEC AMENDMENT A1) for this checkpoint")
    ap.add_argument("--g0-a2-json", default=None,
                    help="reuse an EXISTING G0-A2 wrapper-probe artifact (SPEC AMENDMENT A2)")
    ap.add_argument("--g0-seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--min-milestone-step", type=int, default=5000)
    ap.add_argument("--skip-roll", action="store_true", help="re-analyse existing dumps")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-clips", type=int, default=0, help="SMOKE ONLY: restrict S2 to N clips")
    ap.add_argument("--max-windows-per-clip", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true",
                    help="SMOKE ONLY: trunk bf16/NHWC off (this CPU has no native bf16)")
    ap.add_argument("--skip-gate", action="store_true", help="SMOKE ONLY (CPU runs)")
    a = ap.parse_args()
    if a.smoke_cpu_fp32_trunk:
        _orig = L.build_model

        def _fp32(*aa, **kk):
            m, c, ar, r = _orig(*aa, **kk)
            for mod in m.modules():
                lv = getattr(mod, "memory_levers", None)
                if isinstance(lv, dict) and "bf16" in lv:
                    lv["bf16"] = False
                    lv["channels_last"] = False
            r["smoke_note"] = "trunk bf16/NHWC OFF (--smoke-cpu-fp32-trunk)"
            return m, c, ar, r
        L.build_model = _fp32
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    step = int(ck.get("step"))
    del ck
    tag = a.tag or f"step{step}"
    root = Path(a.out_root) / tag
    root.mkdir(parents=True, exist_ok=True)
    spec = HERE.parent / "SPEC.md"
    summary = {"tool": "run_battery.py", "tag": tag, "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
               "step": step, "config": a.config, "spec_sha256": sha256(spec) if spec.exists() else None,
               "is_milestone": step >= a.min_milestone_step,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": {}}

    def bank():
        json.dump(summary, open(root / "battery_summary.json", "w", encoding="utf-8"),
                  indent=1, default=str)

    bank()
    # ---- 0. gate ----------------------------------------------------------------------- #
    summary["stages"]["gate"] = ({"skipped": "--skip-gate (smoke)"} if a.skip_gate
                                 else gate_wait(str(root / "gate.json")))
    bank()
    # ---- 1. G0 ------------------------------------------------------------------------- #
    def gate_on(g0_json: str, a1_json: str | None, a2_json: str | None = None):
        """SPEC §2 + AMENDMENTS A1/A2: the battery proceeds iff G0-A2 PASSES (G0-A1 when no A2
        artifact exists). G0 as registered and G0-A1 are recorded beside it, in every report."""
        g0full = json.load(open(g0_json, encoding="utf-8"))
        if g0full.get("ckpt_md5") != summary["ckpt_md5"]:
            raise SystemExit(f"[battery] G0 artifact is for md5 {g0full.get('ckpt_md5')}, "
                             f"not this checkpoint {summary['ckpt_md5']}")
        g0 = g0full["verdict"]
        st = {"G0_as_registered": g0["G0"], "reasons": g0["reasons"],
              "by_class_counts": g0["by_class_counts"], "medians": g0.get("medians"),
              "json": g0_json}
        if a1_json:
            a1 = json.load(open(a1_json, encoding="utf-8"))
            st.update(G0_A1=a1["G0_A1"], G0_A1_reasons=a1["G0_A1_reasons"],
                      detection={m: d["n_terms_out"] for m, d in a1["detection"].items()},
                      a1_json=a1_json)
        if a2_json:
            a2 = json.load(open(a2_json, encoding="utf-8"))
            if a2.get("ckpt_md5") != summary["ckpt_md5"]:
                raise SystemExit("[battery] the A2 probe artifact is for another checkpoint")
            st.update(G0_A2=a2.get("G0_A2"), G0_A2_reasons=a2.get("G0_A2_reasons"),
                      A2_wrapper_clause=a2.get("A2_wrapper_clause"),
                      A2_max_wrapper_rel={c: v["analysis"]["max_wrapper_rel"]
                                          for c, v in a2["conditions"].items()},
                      a2_json=a2_json)
        summary["stages"]["g0"] = st
        bank()
        verdict = st.get("G0_A2", st.get("G0_A1", g0["G0"]))
        if verdict != "PASS":
            raise SystemExit(f"[battery] G0-A2/G0-A1 = {verdict} -- STOP (SPEC §2 + A1 + A2)")

    if a.g0_json:
        gate_on(a.g0_json, a.g0_a1_json, a.g0_a2_json)
    elif not a.no_g0:
        if not a.metrics:
            raise SystemExit("[battery] G0 needs --metrics (or pass --no-g0 and say why)")
        g0_json = root / "g0.json"
        cmd = [sys.executable, str(HERE / "reproduce_inrun_eval.py"), "--ckpt", a.ckpt,
               "--config", a.config, "--metrics", a.metrics, "--seeds", a.g0_seeds,
               "--mutation", "m1_no_equalize", "--out", str(g0_json)]
        with open(root / "g0.log", "w", encoding="utf-8") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if not g0_json.exists():
            summary["stages"]["g0"] = {"G0": "NO ARTIFACT", "log": str(root / "g0.log")}
            bank()
            raise SystemExit("[battery] G0 wrote no JSON -- STOP (SPEC §2)")
        a1_json = root / "g0_A1.json"
        gate_wait(str(root / "gate_g0_A1.json"))          # every GPU child waits on the gate
        with open(root / "g0_A1.log", "w", encoding="utf-8") as lf:
            subprocess.run([sys.executable, str(HERE / "g0_mutations.py"), "--g0-json",
                            str(g0_json), "--out", str(a1_json)],
                           stdout=lf, stderr=subprocess.STDOUT)
        a2_json = root / "g0_A2_wrapper_probe.json"
        gate_wait(str(root / "gate_g0_A2.json"))
        with open(root / "g0_A2_wrapper_probe.log", "w", encoding="utf-8") as lf:
            subprocess.run([sys.executable, str(HERE / "wrapper_probe.py"), "--ckpt", a.ckpt,
                            "--config", a.config, "--out", str(a2_json), "--g0-a1-json",
                            str(a1_json)], stdout=lf, stderr=subprocess.STDOUT)
        gate_on(str(g0_json), str(a1_json) if a1_json.exists() else None,
                str(a2_json) if a2_json.exists() else None)
    else:
        summary["stages"]["g0"] = {"G0": "SKIPPED (--no-g0)"}
    # ---- 2. rolls ---------------------------------------------------------------------- #
    kit_ids = set(_clip_ids_by_index(KIT_EVAL))
    windows = RR.s2_window_list(P.BASELINES["b_refcv4b"], kit_ids)
    if a.max_clips:
        windows = {c: windows[c] for c in sorted(windows)[:a.max_clips]}
    if a.max_windows_per_clip:
        windows = {c: w[:a.max_windows_per_clip] for c, w in windows.items()}
    if a.max_clips or a.max_windows_per_clip:
        summary["SMOKE_RESTRICTED_S2"] = True
    summary["S2"] = {"n_clips": len(windows), "n_windows": sum(len(v) for v in windows.values()),
                     "source": P.BASELINES["b_refcv4b"]}
    seeds = [int(s) for s in a.infer_seeds.split(",") if s.strip()]
    rolls = {}
    for s in seeds:
        d = str(root / f"dump_s{s}")
        if a.skip_roll and os.path.exists(os.path.join(d, "manifest.json")):
            rolls[s] = {"dump_dir": d, "reused": True}
            continue
        if not a.skip_gate:
            gate_wait(str(root / f"gate_roll_s{s}.json"))
        # ⛔ THE ROLL RUNS IN A CHILD PROCESS: its CUDA context dies with it, so the parent never
        # holds the card while it waits for the next gate (and the NavSim sibling is not blocked).
        wj = root / f"windows_s{s}.json"
        json.dump(windows, open(wj, "w", encoding="utf-8"))
        rj = root / f"roll_s{s}.json"
        cmd = [sys.executable, str(HERE / "roll_seed.py"), "--ckpt", a.ckpt, "--config", a.config,
               "--seed", str(s), "--dump-dir", d, "--windows-json", str(wj), "--device", a.device,
               "--out-json", str(rj)]
        if a.smoke_cpu_fp32_trunk:
            cmd.append("--smoke-cpu-fp32-trunk")
        with open(root / f"roll_s{s}.log", "w", encoding="utf-8") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if not rj.exists():
            summary["stages"][f"roll_s{s}"] = {"status": "NO ARTIFACT", "log": str(root / f"roll_s{s}.log")}
            bank()
            raise SystemExit(f"[battery] the seed-{s} roll wrote no record -- see roll_s{s}.log")
        rolls[s] = json.load(open(rj, encoding="utf-8"))
        summary["stages"][f"roll_s{s}"] = rolls[s]
        bank()
    # ---- 3. panels --------------------------------------------------------------------- #
    cells, s6cells, clip_eid = {}, {}, None
    for s in seeds:
        d = rolls[s]["dump_dir"]
        pd = str(root / f"panel_s{s}")
        prec = P.build_panel_dump(d, pd, os.path.join(d, "refcv6_extras.npz"))
        summary["stages"][f"panel_s{s}"] = {"pairing": prec["pairing"], "void_gates_3_4":
                                            prec["void_gates_3_4"], "arms": prec["arms"],
                                            "n_windows": prec["n_windows"]}
        bank()
        an = P.analyze(pd, str(root / f"analysis_s{s}.json"), KIT_LABELS, n_boot=a.n_boot)
        vg = P.void_gate_checks(pd, an)
        json.dump(vg, open(root / f"void_gates_s{s}.json", "w", encoding="utf-8"), indent=1)
        summary["stages"][f"void_gates_s{s}"] = vg
        cells[s] = P.cross_paired(pd, PAIRS, n_boot=a.n_boot)
        json.dump(cells[s], open(root / f"cross_paired_s{s}.json", "w", encoding="utf-8"), indent=1)
        man = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
        clip_eid = {int(e["episode_index"]): int(e["file_index"]) for e in man["episodes"]}
        tac = P.tactical_v6(os.path.join(d, "refcv6_extras.npz"), clip_eid, n_boot=a.n_boot)
        json.dump(tac, open(root / f"tactical_v6_s{s}.json", "w", encoding="utf-8"), indent=1)
        accp = P.acceptance(an, os.path.join(d, "refcv6_extras.npz"), clip_eid, pd, n_boot=a.n_boot)
        accp["tzero"] = P.tzero_from_tactical(tac)
        json.dump(accp, open(root / f"acceptance_s{s}.json", "w", encoding="utf-8"), indent=1,
                  default=str)
        s6d = str(root / f"s6_s{s}")
        s6rec = P.build_s6_dump(d, s6d)
        P.analyze(s6d, str(root / f"analysis_s6_s{s}.json"), KIT_LABELS, n_boot=a.n_boot)
        s6cells[s] = P.cross_paired(s6d, PAIRS, n_boot=a.n_boot)
        json.dump(s6cells[s], open(root / f"cross_paired_s6_s{s}.json", "w", encoding="utf-8"),
                  indent=1)
        summary["stages"][f"analysis_s{s}"] = {"s6": s6rec,
                                               "tflip": accp.get("tflip", {}).get("verdict"),
                                               "obedience": accp.get("obedience", {}).get("verdict")}
        bank()
    if len(seeds) >= 2:
        rep = P.seed_replicate(str(root / f"panel_s{seeds[0]}"), str(root / f"panel_s{seeds[1]}"),
                               n_boot=a.n_boot)
        json.dump(rep, open(root / "inference_seed_replicate.json", "w", encoding="utf-8"), indent=1)
        summary["inference_seed_replicate"] = {"max_abs_path_diff_m": rep["max_abs_path_diff_m"],
                                               "ade": rep["families"]["ADE"]["ade_m"]}
        # the replicate travels INTO each analysis record (criteria registry `hyg.inference_seed`)
        floor = abs(float(rep["families"]["ADE"]["ade_m"]["delta"]))
        for s in seeds:
            ap_ = root / f"analysis_s{s}.json"
            an_ = json.load(open(ap_, encoding="utf-8"))
            an_["inference_seeds"] = seeds
            an_["seed_floor_m"] = floor
            an_["inference_seed_replicate"] = {"file": "inference_seed_replicate.json",
                                               "ade_m": rep["families"]["ADE"]["ade_m"],
                                               "max_abs_path_diff_m": rep["max_abs_path_diff_m"]}
            json.dump(an_, open(ap_, "w", encoding="utf-8"), indent=1, default=str)
    # ---- the programme's existing gates, re-used (criteria completeness + echo gate 1) ------ #
    cmp = None
    try:
        import importlib.util as _iu
        _p = (L.REPO / "TanitAD Research Lab" / "Benchmarks & Evals" / "Research" /
              "2026-09-07-refcv5-v2-comparison" / "scripts" / "refcv5_compare.py")
        _spec = _iu.spec_from_file_location("refcv5_compare", str(_p))
        cmp = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(cmp)
    except Exception as exc:                                    # noqa: BLE001 -- recorded
        summary["echo_gate_import_error"] = f"{type(exc).__name__}: {exc}"
    for s in seeds:
        if cmp is not None:
            eg = cmp.run_echo_gate(cmp.load_dump(str(root / f"panel_s{s}")), a.n_boot, 0)
            json.dump(eg, open(root / f"echo_gate_s{s}.json", "w", encoding="utf-8"), indent=1,
                      default=str)
            summary["stages"][f"echo_gate_s{s}"] = {
                "status": eg.get("status"), "per_reference": eg.get("per_reference"),
                "margins": "refcv5_compare BAR_PRIMARY relative_margin_required (INHERITED, not a "
                           "refcv6 bar)"}
        cj = root / f"criteria_s{s}.json"
        with open(root / f"criteria_s{s}.log", "w", encoding="utf-8") as lf:
            subprocess.run([sys.executable, str(L.REPO / "tools" / "criteria_check.py"),
                            str(root / f"analysis_s{s}.json"), "--json", str(cj)],
                           stdout=lf, stderr=subprocess.STDOUT, cwd=str(L.REPO))
        if cj.exists():
            cr = json.load(open(cj, encoding="utf-8"))
            arts = cr.get("artifacts") or {}
            summary["stages"][f"criteria_s{s}"] = {
                "registry_version": cr.get("registry_version"),
                "per_artifact": {os.path.basename(str(k)): {x: v.get(x) for x in
                                                           ("scope", "tier", "n_violations",
                                                            "n_work_items")}
                                 for k, v in arts.items()}}
        bank()
    summary["bars"] = bar_verdicts(cells, s6cells, summary["is_milestone"])
    # ⛔ the orchestrator must never have held the card (the 2026-09-24 self-deadlock)
    summary["parent_cuda_initialized"] = bool(torch.cuda.is_initialized())
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    bank()
    print(json.dumps({"tag": tag, "bars": [(b["id"], b["verdict"]) for b in summary["bars"]]}, indent=1))


if __name__ == "__main__":
    main()
