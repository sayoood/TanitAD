"""``python -m taniteval.bench navsim_v2 …`` — NAVSIM v2 (EPDMS, two-stage splits) end to end.

    preflight (runtime, yaml counts, metric cache by content, logs, venv import probe)
    -> export (devkit AgentInputs, only if a seam arm needs it)
    -> seams: STOP (+ ECHO when a ckpt is scored) ; model arms via the bridge on the GPU-gap device
    -> official two-stage scoring per arm, sequential, through the promoted E1 wrapper (+ E2 guards)
    -> scores/<arm>.csv (UNMODIFIED) · artifacts/<arm>.json · criteria/<arm>.txt
    -> summary.json (schema-validated) · reference checks against banked E1/E2 CSVs

⛔ STOP and CV are MANDATORY floors (BUILD_PLAN.md §1): added by rule when not requested, never
removable. ECHO is added whenever a checkpoint is scored. The human agent is REFUSED on two-stage
splits (E1 finding 3: the official runner crashes — 204/204 stage-2 IndexError, no CSV).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from ..contract import REPO, NAVSIM_FLOORS
from . import artifacts as ART
from . import plans as PL
from . import model_arms as MA
from . import profiles as P
from . import scoring as SC
from . import summarize as S

FLOOR_SPECS = {
    "CV": {"kind": "floor", "official_agent": "constant_velocity_agent", "declared_inputs": ["ego_velocity[t0]"],
           "spec": {"agent": "constant_velocity_agent",
                    "ii": dict(cameras=False, lidar=False, ego_velocity=True, ego_acceleration=False,
                               ego_pose_history=False, driving_command=False),
                    "sensor_set": "none — ego-status only (ego_statuses[-1].ego_velocity)",
                    "setting": "kinematic floor: constant-velocity straight-line extrapolation; perception-free, map-free, route-free",
                    "goal": "none — constant_velocity_agent.py:31-46 reads only ego_statuses[-1].ego_velocity",
                    "route_input": False}},
    "STOP": {"kind": "floor", "seam": "stop", "declared_inputs": [],
             "spec": {"agent": "all-zero plan through the seam agent",
                      "ii": dict(cameras=False, lidar=False, ego_velocity=False, ego_acceleration=False,
                                 ego_pose_history=False, driving_command=False),
                      "sensor_set": "none — reads no input at all",
                      "setting": ("trivial floor: x = y = heading = 0 at every 0.5 s pose (EP == 1 when the best "
                                  "compliant progress <= 5 m, pdm_scorer.py:231-236)"),
                      "goal": "none", "route_input": False}},
    "ECHO": {"kind": "floor", "seam": "echo", "declared_inputs": ["ego_velocity[t0]", "ego_acceleration[t0]"],
             "spec": {"agent": "ECHO ha0_ext: constant measured a0 and curvature k0 (refc_v3.kinematic_goal_extrapolation)",
                      "ii": dict(cameras=False, lidar=False, ego_velocity=True, ego_acceleration=True,
                                 ego_pose_history=False, driving_command=False),
                      "sensor_set": "none — the model arm's own declared t0 ego inputs, echoed",
                      "setting": "echo floor: what the ego inputs alone produce, no perception, no route",
                      "goal": "none", "route_input": False}},
    "CVSEAM": {"kind": "reference", "seam": "cv", "declared_inputs": ["ego_velocity[t0]"],
               "spec": {"agent": "K4 seam-transparency control: the devkit CV poses through the seam agent",
                        "ii": dict(cameras=False, lidar=False, ego_velocity=True, ego_acceleration=False,
                                   ego_pose_history=False, driving_command=False),
                        "sensor_set": "none — ego-status only", "setting": "control: must reproduce CV bit-for-bit",
                        "goal": "none", "route_input": False}},
}
REFUSED_ARMS = {"HUMAN": ("the human agent is UNDEFINED on two-stage splits: all synthetic scenes carry "
                          "num_future_frames = 0 and HumanAgent indexes frames[3..11] -> 204/204 stage-2 IndexError and "
                          "no CSV (E1 RESULT finding 3); it is a STAGE-1-ONLY reference (one-stage runner)")}


def resolve_arms(requested: list, ckpt) -> tuple:
    """-> (ordered [(name, info)], notes). Floors first; STOP + CV always; ECHO iff a ckpt."""
    MODEL_ARMS = MA.MODEL_ARMS
    req = [a.strip().upper() if a.strip().upper() in FLOOR_SPECS or a.strip().upper() in REFUSED_ARMS else a.strip()
           for a in requested if a.strip()]
    notes = []
    for a in req:
        if a in REFUSED_ARMS:
            raise P.Refusal(f"arm {a} refused: {REFUSED_ARMS[a]}")
        if a not in FLOOR_SPECS and a not in MODEL_ARMS:
            raise P.Refusal(f"unknown arm {a!r}; floors {sorted(FLOOR_SPECS)}, model arms {sorted(MODEL_ARMS)}")
        if a in MODEL_ARMS and ckpt is None:
            raise P.Refusal(f"model arm {a} needs --ckpt <path> (got --ckpt none)")
    order = []
    for f in ("CV", "STOP") + (("ECHO",) if ckpt is not None else ()):
        added = f not in req
        order.append((f, {**FLOOR_SPECS[f], "added_by_rule": added}))
        if added:
            notes.append(f"{f} added by rule (" + ("MANDATORY NavSim floor" if f in NAVSIM_FLOORS else
                                                   "ECHO floor: a checkpoint is scored") + ")")
    for a in req:
        if a in ("CV", "STOP", "ECHO"):
            continue
        if a in FLOOR_SPECS:
            order.append((a, {**FLOOR_SPECS[a], "added_by_rule": False}))
        else:
            order.append((a, {"kind": "model", "e2_arm": MODEL_ARMS[a], "added_by_rule": False}))
    return order, notes


def _model_spec(e2_arm: str) -> dict:
    from . import bridge as B
    sp = B.ARMS[e2_arm]
    cam = sp["frames"] != "BLIND"
    ego = bool(sp["declared"])
    dc = "driving_command[t0]" in sp["declared"]
    spec = {"agent": f"TanitAD checkpoint via the declared-input seam ({e2_arm})",
            "ii": dict(cameras=cam, lidar=False, ego_velocity=ego, ego_acceleration=ego, ego_pose_history=False,
                       driving_command=dc, claimed_abstention_from_ego=not ego),
            "sensor_set": ("3-camera stitch cam_l0+cam_f0+cam_r0 -> 256x640 cylindrical (PHYSICALAI_WIDE120_256x640), "
                           f"frames construction {sp['frames']}" if cam else "NONE (frames-blind: one constant grey)"),
            "setting": "perception-free one-shot anchor planner, ZERO-SHOT on NavSim (no NavSim training)",
            "goal": ("NavSim driving_command[t0] mapped onto refb.NAV_COMMANDS" if dc else "none — nav withheld"),
            "route_input": dc, "vision_only_claimed": cam and not ego}
    if spec["vision_only_claimed"]:
        spec["ego_enforcement"] = {
            "vision_only_claimed": True,
            "mechanism": ("declared-input seam: bench/navsim/bridge.py::declare COPIES only the arm's declared t0 fields "
                          "(none for this arm) out of the devkit AgentInput export; the model call receives that dict + "
                          "frames; the NavSim-side agent is a lookup keyed by the scorer token"),
            "evidence": {"file": "raw/model/<arm>.manifest.json (declared + withheld fields per token)",
                         "mutation": ("E2 K5c (FlyWheels/…/2026-09-19-navsim-refcv4b-bridge/raw/K5_K6_ego_mutation.json): "
                                      "every ego field randomised -> 20/20 byte-identical; K6 declared-field mutation "
                                      "changed 20/20 (the probe can fail)")}}
    return spec



def _combined_row_submetrics(raw_rows: dict) -> dict:
    """Every sub-metric of the official combined row, per stage (keys <TERM>_s1 / <TERM>_s2)."""
    r = raw_rows[S.HEADLINE_ROW]
    out = {}
    for k, c in S.SUB.items():
        for suf, tag in (("_stage_one", "_s1"), ("_stage_two", "_s2")):
            v = r.get(c + suf)
            out[k + tag] = float(v) if v not in (None, "") else None
    return {"combined_row": out,
            "_note": "per-stage sub-metrics are also in per_stage.<stage>.submetrics (official stage summary rows)"}

def _route_oracle_caveat() -> dict:
    """⭐ W2 (2026-09-20, four probes): NavSim's ``driving_command`` is a ROUTE-LEVEL ORACLE — it is
    computed from (ego pose NOW, nuPlan's route, the map) and reproduced 1,902/1,902 by OpenScene's
    own function, but the ROUTE is the expert's driven path at roadblock granularity (79.8 % of
    forward blocks are the driven ones vs a 34.6 % null control; a future-derived route reproduces
    96.1 % of turning commands vs 66.4 % without), and on stage 2 the command is COPIED from the
    expert's own frame (5,462/5,462 navhard, 204/204 warmup). Not a trajectory leak. ⇒ a
    command-conditioned NavSim number is *driving with an ORACLE ROUTE*: comparable within NavSim,
    never evidence of route or strategic skill."""
    return {"what": "NavSim driving_command is a ROUTE-LEVEL ORACLE", "source": "W2 / E3 (2026-09-20), four probes",
            "verdict": ART.ad.ROUTE_LEAK_VERDICT, "applies_to": "every arm that consumes driving_command"}


def _caveats(spec: dict) -> list:
    out = []
    if spec.get("route_input"):
        v = ART.ad.ROUTE_LEAK_VERDICT
        out.append({"id": "ROUTE_LEVEL_ORACLE", "one_line": v.get("one_line"), "consequence": v.get("consequence"),
                    "measured": v.get("measured")})
    return out


def _seam_standin_rows(seam_path) -> int:
    """How many rows of a seam are the devkit CV STAND-IN (source == 'cv_standin')."""
    import numpy as np
    try:
        z = np.load(seam_path, allow_pickle=False)
    except Exception:                                                # noqa: BLE001
        return 0
    return int(sum(1 for x in z["source"] if str(x) == "cv_standin"))


def _hooks(raw_arm_dir: Path, arm: str) -> list:
    p = raw_arm_dir / f"{arm}_hooks.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("pdm_score_calls", [])


def _interval(prof, frame_dump: Path, mapping: list, log_of: dict, official_value) -> dict:
    """The arm's interval, through W2's SETTLED estimator (log-cluster bootstrap, B = 2000, RG-14
    floor 8 clusters).

    ⛔ STRUCTURAL (W2, 2026-09-20): the PUBLISHED CSV drops `weight` and `log_name`
    (``run_pdm_score.py:422``), so an interval can only come from the PRE-CSV frame the wrapper
    banks (``<arm>_final_scores_frame.csv``). A run that did not capture it says exactly that —
    never a silent omission, and never an interval computed on a different aggregate.
    """
    cap = {"captured": Path(frame_dump).exists(), "path": str(frame_dump).replace(os.sep, "/"),
           "why_needed": "the published CSV drops `weight`/`log_name` (run_pdm_score.py:422)"}
    if not cap["captured"]:
        return {"status": "UNAVAILABLE", "n": prof.n_logs, "pre_csv_frame": cap,
                "reason": ("pre-CSV frame not captured — no interval is possible from the published CSV "
                           "(it drops `weight` and `log_name`, run_pdm_score.py:422)")}
    if prof.no_interval_reason:
        return {"status": "UNAVAILABLE", "reason": prof.no_interval_reason, "n": prof.n_logs, "pre_csv_frame": cap}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("taniteval_bench_navsim_ci",
                                                      REPO / "taniteval" / "adapters" / "navsim_ci.py")
        ci = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ci)
    except Exception as e:                                               # noqa: BLE001
        return {"status": "UNAVAILABLE", "n": prof.n_logs, "pre_csv_frame": cap,
                "reason": f"W2's NavSim estimator (taniteval/adapters/navsim_ci.py) is not importable: "
                          f"{type(e).__name__}: {e}"[:400]}
    try:
        rows, tok2log = ci.rows_from_score_frame(frame_dump)
        tok2log = {**(log_of or {}), **{k: v for k, v in tok2log.items() if v}}
        keys = [str(m[0]) for m in mapping]
        missing = [k for k in keys if k not in tok2log]
        if missing:
            return {"status": "UNAVAILABLE", "n": prof.n_logs, "pre_csv_frame": cap,
                    "reason": (f"{len(missing)} mapping key(s) have no log_name (first {missing[0]!r}) — the cluster "
                               "unit cannot be assigned and is never guessed")}
        blk = ci.interval_from_run(protocol=prof.protocol, clusters_by_unit={k: tok2log[k] for k in keys},
                                   rows=rows, mapping=mapping, official_value=official_value)
        verdict, why = ci.is_admissible_interval(blk, protocol=prof.protocol, max_clusters=prof.n_logs)
        if verdict == "PASS" and blk.get("lo") is not None:
            return {"status": "OK", "estimator": blk.get("estimator"), "cluster_unit": blk.get("cluster_unit", "log_name"),
                    "lo": blk["lo"], "hi": blk["hi"], "n_clusters": int(blk.get("n_clusters")), "detail": blk,
                    "pre_csv_frame": cap, "question_answered": getattr(ci, "QUESTION_ANSWERED", None)}
        return {"status": "UNAVAILABLE", "n": int(blk.get("n_clusters") or prof.n_logs), "pre_csv_frame": cap,
                "reason": f"W2 estimator verdict {verdict}: {why}"[:600], "detail": blk}
    except Exception as e:                                               # noqa: BLE001
        return {"status": "UNAVAILABLE", "n": prof.n_logs, "pre_csv_frame": cap,
                "reason": f"W2 estimator raised {type(e).__name__}: {e}"[:400]}


def run_benchmark(ctx) -> None:
    a = ctx.args
    if a.split not in P.SPLITS:
        raise P.Refusal(f"split {a.split!r} not supported by navsim_v2 here; supported {sorted(P.SPLITS)} "
                        "(PDMS_v1 on navtest is W3's navsim_v1 --split navtest)")
    prof = P.SPLITS[a.split]
    if prof.stages == 1:
        # ⭐ W8 2026-09-26: the ONE-STAGE runner path (navtest_single_stage). The two-stage path below is
        # unchanged; the dispatch is on the PROFILE's stage count, never on the split's name.
        from .single_stage import run_single_stage
        return run_single_stage(ctx, prof)
    if getattr(a, "tokens_file", None) or getattr(a, "metric_cache", None) or getattr(a, "reuse_scored_arms", None):
        raise P.Refusal(f"--tokens-file / --metric-cache / --reuse-scored-arms are single-stage options; {prof.name} is "
                        "a two-stage split whose stage 2 is tied to its stage 1 by the mapping (use --reuse-floors)")
    arms, notes = resolve_arms(a.arms, a.ckpt)
    for n in notes:
        ctx.log(f"[navsim] {n}")
    y = P.read_split_yaml(prof.name)
    S1, S2 = set(y["stage_one"]), set(y["stage_two"])
    stage_of = {t: 1 for t in S1} | {t: 2 for t in S2}
    dk = P.devkit_sha_measured()
    ctx.set_protocol(prof.protocol)
    ctx.set_split(prof.name, prof.n_scenes, prof.n_logs, n_stage_one=prof.n_stage1, n_stage_two=prof.n_stage2,
                  notes=list(prof.notes))
    ctx.set_devkit(P.DEVKIT_REPO, dk["sha"], P.devkit_patches(), pin=P.DEVKIT_PIN_NOTE, sha_probe=dk,
                   runtime=str(P.CR).replace(os.sep, "/"), wrapper=str(P.WRAPPER.relative_to(REPO)).replace(os.sep, "/"))
    loop = {"stage_one": "OPEN (PI ruling 2026-09-02)", "stage_two": "UNRULED (2026-09-02 vocabulary)",
            "background_traffic": "IDM-reactive vehicles in BOTH stages; ego non-reactive"}
    ctx.set_stamps("T1-family", loop, "MEASURED", closed_loop=False)
    ctx.set_claim_bearing(True)
    for name, info in arms:
        di = info["declared_inputs"] if info["kind"] != "model" else MA.declared_inputs(info["e2_arm"])
        ctx.add_arm(name, info["kind"], di, added_by_rule=info.get("added_by_rule", False), status="FAILED")
    run = ctx.run
    pre = P.preflight(prof)
    run.write_json("raw/preflight.json", pre)
    if getattr(a, "dry_run", False):
        run.write_json("raw/plan.json", {"arms": [n for n, _ in arms], "notes": notes, "preflight": pre})
        for n, _ in arms:
            ctx.arm_rec(n)["status"] = "SKIPPED"
        ctx.rec["status"] = "DRY_RUN_PASSED"      # a PASS, not a refusal (E1 2026-09-20); exits 0
        ctx.rec["dry_run"] = True
        ctx.rec["dry_run_note"] = "DRY RUN: preflight passed; nothing scored"
        return
    exp_dir = P.EXP_ROOT / "runs" / ctx.run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    seams, seam_notes, n_standin = {}, {}, {}
    doc = None
    exp_rec = {}
    need_export = any(info.get("seam") or info["kind"] == "model" for _, info in arms)
    if need_export:
        from .export import ensure_export, compare_exports
        doc, exp_rec = ensure_export(prof, log=ctx.log)
        run.write_json("raw/export_record.json", exp_rec)
        refs = json.loads((Path(__file__).parent / "references.json").read_text(encoding="utf-8")).get(prof.name, {})
        if "_export" in refs:
            run.write_json("raw/export_vs_banked.json", compare_exports(doc, REPO / refs["_export"]["path"]))
    from . import seams as SM
    seam_dir = run.p("raw/seams")
    seam_dir.mkdir(parents=True, exist_ok=True)
    model_req = {n: info["e2_arm"] for n, info in arms if info["kind"] == "model"}
    for name, info in arms:
        kind = info.get("seam")
        if kind == "stop":
            seams[name] = SM.make_stop_seam(doc, seam_dir / f"{name}.npz", arm=name)
        elif kind == "cv":
            seams[name] = SM.make_cv_seam(doc, seam_dir / f"{name}.npz", arm=name)
        elif kind == "echo":
            seams[name], seam_notes[name] = SM.make_echo_seam(doc, seam_dir / f"{name}.npz", arm=name)
    device_used = "none"
    if model_req:
        from ..gpu_gap import GpuGapLauncher
        bank = getattr(a, "frame_bank", None) or MA.DEFAULT_BANKS.get(prof.name)
        if not bank or not Path(bank).exists():
            raise P.Refusal(f"no frame bank for {prof.name} (pass --frame-bank; E2's build_frames.py builds one)")
        ctx.gpu = GpuGapLauncher(requested=a.device, log=ctx.log,
                                 limit_mib=getattr(a, "gpu_mem_limit_mib", None))
        mres = MA.run_model_arms(arms=model_req, doc=doc, ckpt=a.ckpt, bank=bank, out_dir=run.p("raw/model"),
                              gpu=ctx.gpu, threads=int(getattr(a, "cpu_threads", 6) or 6),
                              ckpt_md5=getattr(a, "ckpt_md5", None),
                              reuse=getattr(a, "reuse_seams", None), log=ctx.log)
        run.write_json("raw/model/model_run.json", mres)
        for n in model_req:
            seams[n] = run.p(f"raw/model/{n}.npz")
            n_standin[n] = mres["arms"][n]["n_cv_standin_rows"]
        # ⛔ when EVERY model arm was adopted from a banked seam, no inference ran and the gap
        # launcher was never asked for a device. Its `record()` would then report `used: "cpu"`,
        # which is a claim about a thing that did not happen — the same shape as a status code
        # standing in for an artifact. Say what is true instead.
        if all(v.get("reused") for v in mres["arms"].values()):
            ctx.gpu = None
            ctx.rec["device_used_override"] = "none"
            ctx.rec["device_detail"] = {
                "status": "NOT_INVOKED", "reason": ("no inference ran: every model arm was adopted "
                                                    "from a banked seam (--reuse-seams)"),
                "reused_from": getattr(a, "reuse_seams", None)}
    # ⛔ W-25: the stand-in count is read from the SEAM ARTIFACT, not from a producer's report — any
    # arm whose stage 1 was answered by the devkit CV stand-in gets NO official two-stage headline.
    for name, sp in seams.items():
        n_standin[name] = max(n_standin.get(name, 0), _seam_standin_rows(sp))
    scored, counts = {}, {}
    reuse_floors = getattr(a, "reuse_floors", None)
    reuse_echo = getattr(a, "reuse_echo", None)
    reuse_model_scores = getattr(a, "reuse_model_scores", None)
    floors_reused = {}
    model_scores_adopted = {}
    for name, info in arms:
        rdir = run.p(f"raw/{name}")
        is_model = info["kind"] == "model"
        src_for = (reuse_floors if (reuse_floors and name in ("CV", "STOP")) else
                   (reuse_echo if (reuse_echo and name == "ECHO") else
                    (reuse_model_scores if (reuse_model_scores and is_model) else None)))
        if src_for and is_model:
            # ⭐ W7 2026-09-21 — RE-DERIVATION of a completed run after an artifact-builder fix: the
            # model arm's SCORED rows are adopted (plan byte-equal; see floor_reuse kind="model"), so the
            # artifacts/summary/criteria are rebuilt WITHOUT re-scoring. Recorded as a re-derivation,
            # never as a floor.
            from . import floor_reuse as FR
            rep = FR.check_and_adopt(arm=name, src_run=Path(src_for), run=run, prof=prof, split_yaml=y,
                                     devkit_sha=dk["sha"], patches=P.devkit_patches(), preflight=pre,
                                     export_sha256=exp_rec.get("sha256"), new_seam=seams.get(name),
                                     log=ctx.log, kind="model")
            model_scores_adopted[name] = rep["reused_from"]
            counts[name] = rep
            ctx.write_scores(name, rep["csv"])
            scored[name] = run.p(f"scores/{name}.csv")
            ctx.arm_rec(name)["status"] = "OK"
            ctx.arm_rec(name)["scores_adopted_from"] = rep["reused_from"]["source_run"]
            continue
        if src_for:
            # ⭐ W7 2026-09-21 — the floor is ADOPTED from a run that scored the identical token set,
            # after an identity check that RAISES on any mismatch (floor_reuse.py). The mandatory-
            # floors rule is satisfied by a floor ON THE SAME TOKENS; the adoption is recorded below
            # and in raw/<arm>/<arm>.reused.json, never a silent skip.
            from . import floor_reuse as FR
            rep = FR.check_and_adopt(arm=name, src_run=Path(src_for), run=run, prof=prof, split_yaml=y,
                                     devkit_sha=dk["sha"], patches=P.devkit_patches(), preflight=pre,
                                     export_sha256=exp_rec.get("sha256"), new_seam=seams.get(name),
                                     log=ctx.log)
            floors_reused[name] = rep["reused_from"]
            counts[name] = rep
            ctx.write_scores(name, rep["csv"])
            scored[name] = run.p(f"scores/{name}.csv")
            ctx.arm_rec(name)["status"] = "OK"
            ctx.arm_rec(name)["reused_from"] = rep["reused_from"]["source_run"]
            continue
        rep = SC.score_arm(
            arm=name, prof=prof, raw_dir=rdir, exp_dir=exp_dir, seam=seams.get(name),
            official_agent=info.get("official_agent"), ram_floor_mb=getattr(a, "ram_floor_mb", 3000.0),
            log=ctx.log,
            # the RUN-START snapshot, so each arm can say whether the devkit-side code MOVED
            # under it (orchestrator ruling 2026-09-20: a snapshot blind to a mid-run change is
            # provenance that reads true and is not).
            baseline_blobs=(ctx.rec.get("git") or {}).get("suite_code_blobs") or {})
        counts[name] = rep
        if rep["status"] == "PASS":
            ctx.write_scores(name, rep["csv"])
            scored[name] = run.p(f"scores/{name}.csv")
            ctx.arm_rec(name)["status"] = "OK"
    # ---- W5's FAILURE GALLERY inputs: plans/<arm>.npz + scenes.json (the PI asked for the gallery)
    gallery_inputs = {}
    for name in scored:
        gallery_inputs[name] = PL.write_plans(run.p(f"plans/{name}.npz"), arm=name,
                                              hooks=_hooks(run.p(f"raw/{name}"), name), seam=seams.get(name))
    if doc is not None:
        bank = getattr(a, "frame_bank", None) or MA.DEFAULT_BANKS.get(prof.name)
        gallery_inputs["scenes"] = PL.write_scenes(run.p("scenes.json"), doc, prof=prof, frame_bank=bank)
    else:
        gallery_inputs["scenes"] = {"status": "UNAVAILABLE", "n": 0,
                                    "reason": ("no devkit export was needed by this run (no seam arm), so the "
                                               "token -> scene/log/v0/command map was never built; the gallery "
                                               "needs it — run with at least one seam arm (STOP is mandatory)")}
    run.write_json("raw/gallery_inputs.json", gallery_inputs)

    # ------------------------------------------------------------------ summary
    refs = json.loads((Path(__file__).parent / "references.json").read_text(encoding="utf-8")).get(prof.name, {})
    tok, raw, heads, us, log_of = {}, {}, {}, {}, {}
    for name in scored:
        tok[name] = S.load_scores(scored[name], stage_of)
        hdr, raw[name] = S.read_raw_rows(scored[name])
        for c in _hooks(run.p(f"raw/{name}"), name):
            if c.get("token") and c.get("log_name"):
                log_of[c["token"]] = c["log_name"]
        heads[name] = None if n_standin.get(name) else S.headline_value(raw[name], hdr)
        u = S.s2_group_uniform(tok[name], y["mapping"])
        us[name] = u.get("value")
    floors = [f for f in ("STOP", "CV") if f in scored]
    summ_arms = {}
    for name, info in arms:
        rdir = run.p(f"raw/{name}")
        if name not in scored:
            summ_arms[name] = {"kind": info["kind"], "status": "FAILED", "declared_inputs": ctx.arm_rec(name)["declared_inputs"],
                               "headline": {"status": "UNAVAILABLE", "reason": f"scoring FAILED: {counts[name]['failures']}", "n": 0},
                               "per_stage": {}, "submetrics": {}, "per_log": {},
                               "paired": {f: ({"status": "SELF"} if f == name else
                                              {"status": "UNAVAILABLE", "reason": "this arm was not scored", "n": 0})
                                          for f in ("STOP", "CV")},
                               "interval": {"status": "UNAVAILABLE", "reason": "not scored", "n": 0},
                               "families": S.families_refused("not scored"), "files": {"counts": f"raw/{name}/{name}.counts.json"}}
            continue
        spec = info["spec"] if info["kind"] != "model" else _model_spec(info["e2_arm"])
        hooks = _hooks(rdir, name)
        full_df = pd.read_csv(scored[name], index_col=0)
        frame_dump = rdir / f"{name}_final_scores_frame.csv"
        ctl = {"C4_formula": S.c4(full_df, True)}
        if frame_dump.exists():
            try:
                ctl["C5_aggregate"] = S.c5_aggregate(full_df, pd.read_csv(frame_dump), y["mapping"])
            except Exception as e:                                       # noqa: BLE001
                ctl["C5_aggregate"] = {"status": "RAISED", "reason": f"{type(e).__name__}: {e}"[:300]}
        ref = refs.get(name)
        if ref and ref.get("csv") and (REPO / ref["csv"]).exists() and not n_standin.get(name):
            rc = S.compare_to_reference(scored[name], REPO / ref["csv"])
            if "combined_score" in ref:
                rc["headline_equals_banked"] = heads[name] == ref["combined_score"]
                rc["banked_combined_score"] = ref["combined_score"]
            if "external" in ref:
                ext = ref["external"]
                rc["external"] = {**ext, "ours_x100_rounded": round(100 * heads[name], ext["round_dp"]),
                                  "reproduced": round(100 * heads[name], ext["round_dp"]) == ext["value_x100"]}
            rc["source"] = ref.get("source")
            ctl["reference_check"] = rc
        # STAGE-1 sub-metric reference (navhard; E1 relay 2026-09-20). Independent of the CSV check
        # above and guarded separately, so a split with one and not the other still emits what it has.
        if ref and ref.get("stage_one") and not n_standin.get(name):
            try:
                s1 = S.stage1_submetrics(tok[name])
                ctl["stage1_reference_check"] = S.stage1_reference_check(
                    s1["values_x100"], ref["stage_one"], n=s1["n"])
            except Exception as e:                                       # noqa: BLE001
                ctl["stage1_reference_check"] = {"status": "RAISED", "reason": f"{type(e).__name__}: {e}"[:300]}
        interval = _interval(prof, frame_dump, y["mapping"], log_of, heads[name])
        if n_standin.get(name) and interval.get("status") == "OK":
            # ⛔⛔ W7 2026-09-21, MEASURED on navhard A1: the interval came back `status: OK` with point
            # 0.1113 == the HYBRID combined row 0.111329 — i.e. an interval on (devkit CV stage 1 x the
            # model's stage 2). Published under the arm's name it would state CV's stage-1 uncertainty as
            # the model's. The headline was already refused for this reason; the interval must be too.
            hybrid_interval = interval
            interval = {"status": "UNAVAILABLE", "n": int(interval.get("n_clusters") or prof.n_logs),
                        "reason": (f"the only two-stage aggregate for {name} is the CV-stand-in HYBRID "
                                   f"({n_standin[name]} stage-1 rows are the devkit CV agent, which also sets the "
                                   "stage-2 kernel weights); an interval on it is not an interval on this arm. "
                                   "The hybrid's own interval is kept under "
                                   "statistics.official_combined_row_HYBRID.interval_NOT_this_arm")}
        else:
            hybrid_interval = None
        art = ART.build_arm_artifact(
            arm=name, spec=spec, split=prof.name, protocol=prof.protocol, raw_rows=raw[name], hooks=hooks,
            S1=S1, S2=S2, n_logs=prof.n_logs, interval=interval, n_standin=n_standin.get(name, 0),
            controls={k: v for k, v in ctl.items() if k != "reference_check"})
        ctx.write_artifact(name, art)
        crit = ctx.run_criteria(name)
        ctl["criteria_check"] = crit
        ctl["navsim_gates"] = art["navsim_gate_selfcheck"]["gates"]
        ctl["navsim_gate_mutations_all_red"] = art["navsim_gate_selfcheck"]["mutations_all_red"]
        if n_standin.get(name):
            headline = {"status": "UNAVAILABLE", "n": int(n_standin[name]),
                        "reason": (f"two-stage EPDMS UNDEFINED for {name}: {n_standin[name]} stage-1 rows are the devkit CV "
                                   "stand-in (no stage-1 frames), which set both the stage-1 factor and the stage-2 kernel "
                                   "weights — see statistics.S2_EPDMS_u (stage 2 only) and statistics.official_combined_row_HYBRID")}
        else:
            headline = {"value": heads[name], "x100": 100.0 * heads[name], "column": S.HEADLINE_COLUMN,
                        "row": S.HEADLINE_ROW, "n": int(sum(1 for t in raw[name] if t in stage_of)),
                        "statistic": ("official two-stage EPDMS: run_pdm_score.py::compute_final_scores (EC injected, /16) "
                                      "aggregated over mapping keys ((s1*wavg(now) + s1'*wavg(prev))/2)")}
        u = S.s2_group_uniform(tok[name], y["mapping"])
        statistics = {"S2_EPDMS_u": {**u, "what": ("E2's stage-2-only statistic: mean over groups of the uniform "
                                                   "within-group mean of the official stage-2 `score` — NOT a two-stage EPDMS")},
                      "stage_one_scene_mean": float(tok[name][tok[name].stage == 1].score.mean()),
                      "stage_two_scene_mean": float(tok[name][tok[name].stage == 2].score.mean())}
        if n_standin.get(name):
            statistics["official_combined_row_HYBRID"] = {
                "value": float(raw[name][S.HEADLINE_ROW][S.HEADLINE_COLUMN]),
                "why_not_the_headline": "stage 1 = devkit CV stand-in; the combined row mixes CV and the model",
                "interval_NOT_this_arm": hybrid_interval}
        paired = {}
        for f in ("STOP", "CV"):
            if f == name:
                paired[f] = {"status": "SELF"}
            elif f in scored:
                paired[f] = S.paired_block(tok[name], tok[f], heads[name], heads[f], raw[name], raw[f], us.get(name), us.get(f))
            else:
                paired[f] = {"status": "UNAVAILABLE", "reason": f"floor {f} was not scored in this run", "n": 0}
        fams = (S.families_from_artifact(art, art["four_families"].get("_w1_scope", "")) if not n_standin.get(name)
                # ⛔ W7 2026-09-21: the reason is DERIVED from THIS split, never the warmup constant
                # (MEASURED: navhard A1 was told "the 204 stage-2 scenes ... the 16 stage-1 scenes").
                # n = the stage-1 scenes the families WOULD cover once their frames exist.
                else S.families_refused(ART.no_gt_reason(prof.name, len(S1), len(S2), int(n_standin[name])),
                                        len(S1)))
        summ_arms[name] = {
            "kind": info["kind"], "status": "OK", "declared_inputs": ctx.arm_rec(name)["declared_inputs"],
            "added_by_rule": info.get("added_by_rule", False),
            "headline": headline,
            "per_stage": {st: S._stage_block(raw[name], S.STAGE_ROWS[st], "_" + st,
                                             int((tok[name].stage == (1 if st == "stage_one" else 2)).sum()))
                          for st in ("stage_one", "stage_two")},
            "submetrics": _combined_row_submetrics(raw[name]),
            "per_log": S.per_log_block(tok[name], log_of),
            "paired": paired, "interval": interval, "families": fams, "statistics": statistics,
            "caveats": _caveats(spec) + ([{
                "id": "FLOOR_REUSED_FROM_SOURCE_RUN",
                "one_line": (f"{name} was NOT re-scored in this run: its per-token rows are adopted, byte-"
                             f"identical, from {floors_reused[name]['source_run']}, which scored the identical "
                             "token set under the identical devkit, patches, metric cache and agent inputs"),
                "consequence": ("its numbers equal the source run's by construction; the pairing with this "
                                "run's model arms is on the same tokens"),
                "record": f"raw/{name}/{name}.reused.json"}] if name in floors_reused else []) + ([{
                "id": "SCORES_ADOPTED_FOR_RE_DERIVATION",
                "one_line": (f"{name} was NOT re-scored: its per-token rows are adopted, byte-identical, from "
                             f"{model_scores_adopted[name]['source_run']} (the run that scored them; plan "
                             "byte-equal) to rebuild artifacts/summary/criteria after a builder fix"),
                "consequence": "no number here is a new measurement; every score equals the source run's",
                "record": f"raw/{name}/{name}.reused.json"}] if name in model_scores_adopted else []),
            **({"reused_from": floors_reused[name]["source_run"]} if name in floors_reused else {}),
            **({"scores_adopted_from": model_scores_adopted[name]["source_run"]}
               if name in model_scores_adopted else {}),
            "modality": {"sensor_set": spec["sensor_set"], "setting": spec["setting"],
                         "ego_status_used": bool(spec["ii"].get("ego_velocity") or spec["ii"].get("ego_acceleration")),
                         "vision_only_claimed": bool(spec.get("vision_only_claimed", False))},
            "controls": ctl,
            "files": {"scores": f"scores/{name}.csv", "artifact": f"artifacts/{name}.json",
                      **({"plans": f"plans/{name}.npz"} if (gallery_inputs.get(name) or {}).get("status") == "OK" else {}),
                      "criteria": f"criteria/{name}.txt", "counts": f"raw/{name}/{name}.counts.json",
                      "wrapper_manifest": f"raw/{name}/{name}_manifest.json", "hooks": f"raw/{name}/{name}_hooks.json",
                      "final_scores_frame": f"raw/{name}/{name}_final_scores_frame.csv"},
        }
    summary = {
        "schema": "taniteval.bench.summary/1", "run_id": ctx.run_id, "benchmark": "navsim_v2",
        "protocol": prof.protocol, "split": prof.name, "claim_bearing": True, "evidence_class": "MEASURED",
        "stamps": {"tier": "T1-family", "loop": loop, "closed_loop": False},
        "headline_metric": {"name": "EPDMS", "column": S.HEADLINE_COLUMN, "higher_is_better": True,
                            "statistic": "official two-stage EPDMS (row extended_pdm_score_combined)",
                            "forbidden_column": S.FORBIDDEN_COLUMN, "range": [0.0, 1.0]},
        "floors": list(NAVSIM_FLOORS), "arms": summ_arms,
        "controls": {"arm_notes": notes, "route_command_is_a_route_level_oracle": _route_oracle_caveat(),
                     # ⛔ W7 2026-09-21: when a floor was ADOPTED rather than re-scored, the artifact says
                     # so, names the source run, and states how the mandatory-floors rule is satisfied.
                     "floors_provenance": ({"status": "REUSED", "arms": floors_reused,
                                            "mandatory_floors_rule": (
                                                "SATISFIED by floors scored ON THE SAME TOKENS: the adopted "
                                                "per-token rows come from a run that scored the identical "
                                                "token set (by value, with stage labels) under the identical "
                                                "devkit sha, patch set, metric cache and agent inputs; see "
                                                "raw/<arm>/<arm>.reused.json")}
                                           if floors_reused else
                                           {"status": "SCORED_IN_THIS_RUN", "arms": sorted(scored)}),
                     # ⛔ a model arm whose SCORED rows were adopted is a RE-DERIVATION, and says so here
                     "model_scores_provenance": ({"status": "ADOPTED_FOR_RE_DERIVATION", "arms": model_scores_adopted,
                                                  "what": ("the model arm was NOT re-scored: its per-token rows are "
                                                           "adopted from the run that scored them (plan byte-equal), "
                                                           "to rebuild artifacts/summary/criteria after a builder fix. "
                                                           "No number in this run is a new measurement of the model.")}
                                                 if model_scores_adopted else
                                                 {"status": "SCORED_IN_THIS_RUN"}),
                     "gallery_inputs": gallery_inputs, "scoring_counts": {k: {kk: v[kk] for kk in ("status", "failures", "wall_s",
                                                                                    "log_successful", "log_failed",
                                                                                    "csv_valid_rows", "agent_calls")}
                                                             for k, v in counts.items()}},
        # ⭐ ckpt is the SAME object bench_run.json carries (ctx.rec["ckpt"]), never a copy built here:
        # two independently assembled triples are two things that can disagree, and the leaderboard
        # reads THIS one. W4, 2026-09-20: without it a MODEL row cannot say which checkpoint made it.
        "provenance": {"devkit": ctx.rec.get("devkit"), "split_yaml_counts": pre.get("yaml_counts"),
                       "ckpt": ctx.rec.get("ckpt"),
                       "promoted_from": ["E1 2026-09-19-navsim-warmup-reference-epdms", "E2 2026-09-19-navsim-refcv4b-bridge"]},
    }
    ctx.write_summary(summary)
    if any(v.get("retryable") for v in counts.values()):
        ctx.rec["retryable"] = True
        ctx.rec["retryable_reason"] = ("the promoted wrapper's RAM guard aborted an arm (available memory below the "
                                       "floor while other jobs held the box) — nothing about the harness failed")
    ok_floors = all(f in scored for f in NAVSIM_FLOORS)
    ctx.rec["status"] = ("COMPLETE" if len(scored) == len(arms) else ("PARTIAL" if ok_floors else "FAILED"))
    if ctx.gpu is not None:
        ctx.rec["device_detail"] = ctx.gpu.record()
        device_used = ctx.gpu.record()["used"]
    ctx.rec["device_used_override"] = device_used
