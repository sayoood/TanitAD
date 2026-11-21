"""SR-TTC S0 dump: P2's scorer on the FIT SETS' OWN 64 proposals -- features (the score head's INPUT), the six logits, the frozen top-8 contender membership, the PDM labels P2 trained on.

Derived from `p2ho_dump.py` (the working held-out dump; its diff to the landed navtest seam is in the p2-heldout-table package).  What is the SAME, unchanged: the planner, the
scenario builder call, the input construction (`planner._image_for`, `planner.infer`), the checkpoint, the rule (navsim_v1), the A7 repair, the PER-LOG frame resolver.
What is NEW (and only this):
  * `--sets-dir`: per-log stores `<log>.npz` written by srttc_s0_podsets.py with `train.OnPolicyBank` (the class P2's trainer used): for every served fit set of the log
    traj3 [S,64,20,3] (x, y, yaw: the trainer's `cxy`), labels6 [S,64,6] (NC, DAC, EP, TTC, C, DDC, clipped: the trainer's `cg`), ckpt_step.
  * for each token the planner runs ONE image forward (`infer`), a forward PRE-hook records the scoring decoder's visual context `sctx`, and for every fit set of that token
    `model.score_trajectories(traj3, sctx)` -- the very call the trainer made (`REFe.forward(score_extra=cxy)` -> `score_trajectories`) -- gives the set's logits [64,6];
    a hook on `score_head` records its INPUT [64,256] (fp16).  Cost: one image forward per TOKEN plus one scoring-decoder pass per set.
  * the frozen contender membership: top-8 of the planner's own selection score NC*DAC*(5EP+5TTC+2C)/12 on the sigmoid of the set's logits (float64, stable order).  Computed once.
  * IDENTITY CONTROLS (integrity, not results).  POSITIVE: the model's OWN fan fed through `score_trajectories` with the captured `sctx` must reproduce the logits `forward` returned
    (max abs ~0: same modules, same inputs; per set `ctrl_fanpath_logit_maxabs`).  NEGATIVE: a stored set is a DIFFERENT fan, so |set logits - fan logits| (`ctrl_logit_maxabs`) must read clearly non-zero.
    Diagnostic only: `ctrl_traj_maxabs` = max|stored set trajectory - this forward's fan|.  MEASURED in the smoke: it is 0.3-8 m for every stored checkpoint (smaller for later ones), also for
    ckpt_step 10075 -- P2 was trained with the visual context un-detached (`--scorer-sees-trunk`), so P2's own fan is NOT any stored fan; the stored sets are scored by P2's scorer as P2's trainer did.
  * `--lean`: do not store the token-level fan arrays (proposals / proposals20 / fan features): S0 does not need them.  Without `--sets-dir` the script is the held-out dump (DG-DUMP).
Prints ZZS0DUMP_OK <tokens> <sets> only when every requested token has a finite row and every fit set of those tokens a finite one.

P6-READ-GAPS G4 (2026-10-22): COMMAND-MODE checkpoints (P6, P6-N) are accepted when a COMMAND SOURCE is given: `--commands <commands.json>` (or env SRTTC_COMMANDS_JSON, because the landed
driver builds this script's argv and cannot forward a flag; `--commands-sha256` / SRTTC_COMMANDS_SHA256 pins the file). The source is P6's own training command table (HF logs/commands.json,
103,037 scenes): {"<log>/<token>": one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN] = NAVSIM's driving_command}. Each token's command is handed to the planner by the seam's OWN
`set_driving_command` (the function the navtest seam and the P6 dry run use). A token with no command, or a non-one-hot vector, RAISES (never a default). A command-mode (or sanitising)
checkpoint with NO source is still REFUSED, exactly as before; a goal-mode checkpoint (P2) ignores a supplied source and the run is the landed one.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe srttc_s0_dump.py --ckpt <model_final.pt> --plan <plan.json> --out <dump.npz> --sets-dir <dir> --sets-out <sets.npz> --device cuda --lean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, os.path.join(TREE, "refe"))
sys.path.insert(0, os.path.join(TREE, "eval"))
NAVSIM_DT, NAVSIM_N = 0.5, 8
SRC_DT = 0.2


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_command_table(path, sha256=None) -> dict:
    """P6's command table: {"<log>/<token>": [4 ints]}; refuses a file whose sha256 differs from the pin, an empty table, and any non-one-hot row"""
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if sha256 and got != sha256.lower():
        raise SystemExit(f"REFUSED: command table {path} sha256 {got} != the pinned {sha256}")
    tab = json.loads(raw.decode("utf-8"))
    if not isinstance(tab, dict) or not tab:
        raise SystemExit(f"REFUSED: command table {path} is empty or not a dict")
    for k, v in tab.items():
        if "/" not in k or len(v) != 4 or sorted(int(x) for x in v) != [0, 0, 0, 1]:
            raise SystemExit(f"REFUSED: command table {path}: row {k!r} = {v!r} is not '<log>/<token>' -> one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN]")
    return {"path": os.path.abspath(path), "sha256": got, "rows": tab}


def command_vec(table: dict, log: str, tok: str) -> list:
    """the token's one-hot command; RAISES when the table has none (a command-mode model fed a missing command would answer an unknown question)"""
    key = f"{log}/{tok}"
    if key not in table["rows"]:
        raise RuntimeError(f"token {tok} (log {log}): no command in {table['path']}: a command-mode checkpoint is never run on a default command")
    return [int(x) for x in table["rows"][key]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True, help="{'tokens', 'token_log', 'token_root', 'db'} from p2ho_audit.py (or a smoke plan of the same shape)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat-out", default=None)
    ap.add_argument("--arm", default="REFe_p2heldout")
    ap.add_argument("--select", default="best")
    ap.add_argument("--rule", default="navsim_v1", choices=("v2_shape", "navsim_v1", "navsim_v1_lane"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--sets-dir", default=None, help="dir of per-log fit-set stores <log>.npz (srttc_s0_podsets.py); omitted -> the held-out fan dump (DG-DUMP)")
    ap.add_argument("--sets-out", default=None, help="npz for the per-set features / logits / top-8 / labels / identity controls")
    ap.add_argument("--lean", action="store_true", help="do not store the token-level fan arrays (proposals, proposals20, fan features)")
    ap.add_argument("--commands", default=os.environ.get("SRTTC_COMMANDS_JSON") or None, help="G4: P6's command table (see the docstring); required for a command-mode checkpoint")
    ap.add_argument("--commands-sha256", default=os.environ.get("SRTTC_COMMANDS_SHA256") or None)
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8"))
    toks = sorted(plan["tokens"])
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(plan["token_log"][t], []).append(t)
    print(f"  tokens: {len(toks):,} over {len(by_log)} logs; ckpt {a.ckpt}", flush=True)

    import torch
    import navtrain_scenarios as NS
    import planner as PLM
    from refe_navtest_seam import to_navsim            # landed conversion, imported unchanged (its main() is guarded)
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    first_db = next(iter(plan["db"].values()))
    planner = REFePlanner(checkpoint=a.ckpt, images_root="", db_dir=os.path.dirname(first_db), backbone=a.backbone, device=a.device, select=a.select, rule=a.rule)
    nav_mode = getattr(planner.model.cfg, "nav_mode", "goal")
    cmd_table = None
    if nav_mode == "command" or planner.sanitize_goal:
        if not a.commands:
            print("  REFUSED: a command-mode / sanitising planner needs NAVSIM's driving_command, which navtrain tokens do not carry here "
                  "(give --commands <P6 commands.json> or set SRTTC_COMMANDS_JSON)", flush=True)
            return 2
        cmd_table = load_command_table(a.commands, a.commands_sha256)
        print(f"  command source: {cmd_table['path']} sha256 {cmd_table['sha256'][:16]} rows {len(cmd_table['rows']):,}", flush=True)
    n_cmd_set = 0
    print(f"  planner: trained={planner.trained} per_sample_calib={planner.per_sample_calib} device={planner.device} select={planner.select} rule={planner.rule} "
          f"goal_fix={planner.goal_fix} repair_last_heading={planner.repair_last_heading} nav_mode={nav_mode}", flush=True)
    # one FrameResolver per image root; switched per token (db_dir is per log).  The resolver class is the planner's own, unmodified.
    frs = {r: PLM.FrameResolver(db_dir="", images_root=r) for r in sorted(set(plan["token_root"].values()))}
    feat_box: list = []
    sctx_box: list = []
    store: dict = {}                       # log -> {"z": npz dict, "by_tok": {token: [row idx]}}
    out_sets: dict = {k: [] for k in ("log", "token", "step", "rank", "ckpt", "feat", "logits", "top8", "labels6", "ctrl_traj", "ctrl_logit", "ctrl_fanpath")}
    fanpath_tok: list = []                 # per token: max|score_trajectories(the fan this forward produced) - the logits this forward produced|  (must be ~0: same modules, same inputs)
    if a.sets_dir:
        for lg in by_log:
            p = os.path.join(a.sets_dir, f"{lg}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"no fit-set store for {lg}: {p}")
            z = dict(np.load(p, allow_pickle=False))
            bt: dict = {}
            for i, t in enumerate(z["token"]):
                bt.setdefault(str(t), []).append(i)
            store[lg] = {"z": z, "by_tok": bt}
        print(f"  fit-set stores: {len(store)} logs, {sum(len(v['z']['token']) for v in store.values()):,} sets", flush=True)
    hook = planner.model.score_head.register_forward_hook(lambda m, inp, out: feat_box.append((inp[0].detach().float().cpu().numpy().astype(np.float16), out.detach().float().cpu().numpy())))
    # the scoring decoder's visual context: the 2nd positional argument of the FIRST scoring block (model.py: `s = blk(s, sctx)`); the same tensor `score_trajectories` is given
    hook2 = planner.model.score_dec[0].register_forward_pre_hook(lambda m, args: sctx_box.append(args[1].detach()))
    rows_tok, rows_fp, rows_pose, misses = [], [], [], []
    dump_props, dump_props20, dump_logits, dump_pick, dump_feat = [], [], [], [], []
    t0 = time.time()
    for li, (log, lt) in enumerate(sorted(by_log.items())):
        db = plan["db"][log]
        got = set()
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            got.add(tok)
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            fr = frs[plan["token_root"][tok]]
            fr.db_dir = os.path.dirname(db)
            planner.frames = fr
            img = planner._image_for(ego)
            if img is None:
                misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            feat_box.clear()
            sctx_box.clear()
            if cmd_table is not None:                                       # G4: NAVSIM's driving_command for this token, through the seam's own hop
                from refe_navtest_seam import set_driving_command
                n_cmd_set += int(set_driving_command(planner, tok, {"ego_statuses": [{"driving_command": command_vec(cmd_table, log, tok)}]}))
            traj, _score, k = planner.infer(ego, img)
            if len(feat_box) != 1:
                raise RuntimeError(f"token {tok}: the score_head hook fired {len(feat_box)} times in one infer (want exactly 1: a second forward would have run)")
            feat = feat_box[0][0][0]                                        # [64, dec_width] fp16
            if not np.array_equal(feat_box[0][1][0], _score.float().cpu().numpy()):
                raise RuntimeError(f"token {tok}: the hooked score_head output is not the logits infer() returned (the hook is on the wrong call)")
            ev, gl = planner.last_inputs
            paths = planner.frames.resolve(planner._log_hint, int(ego.time_point.time_us)) or []
            fp = hashlib.sha256("|".join([log, tok, ev.detach().cpu().numpy().astype(np.float64).tobytes().hex(),
                                          gl.detach().cpu().numpy().astype(np.float64).tobytes().hex(), *[os.path.basename(p.split("::")[-1]) for p in paths]]).encode()).hexdigest()[:16]
            exe20 = [planner.executed(traj, j).float().cpu().numpy() for j in range(traj.shape[0])]
            poses = to_navsim(planner.executed(traj, k).float().cpu().numpy())
            props8 = np.stack([to_navsim(e) for e in exe20]).astype(np.float32)
            if not (np.isfinite(poses).all() and np.isfinite(props8).all() and np.isfinite(_score.float().cpu().numpy()).all()):
                misses.append((tok, "non-finite output"))
                continue
            assert np.array_equal(props8[k].astype(np.float32), poses.astype(np.float32)), "the picked proposal differs from the seam row (conversion path)"
            rows_tok.append(tok)
            rows_fp.append(fp)
            rows_pose.append(poses.astype(np.float32))
            dump_props.append(props8)
            dump_props20.append(np.stack(exe20).astype(np.float32))
            dump_logits.append(_score.float().cpu().numpy().astype(np.float32))
            dump_pick.append(int(k))
            dump_feat.append(feat)
            if a.sets_dir:
                if len(sctx_box) != 1:
                    raise RuntimeError(f"token {tok}: the scoring-decoder pre-hook fired {len(sctx_box)} times in one infer (want 1)")
                sctx = sctx_box[0]
                fan3 = traj.detach().float().cpu().numpy()                    # [64, 20, 3] the raw proposals of THIS forward
                fan_logits = _score.float().cpu().numpy()
                # POSITIVE CONTROL of the set-scoring path: feeding the model's OWN fan through score_trajectories must reproduce the logits `forward` returned (same modules, same sctx)
                feat_box.clear()
                with torch.no_grad():
                    lg_chk = planner.model.score_trajectories(traj.detach().unsqueeze(0), sctx)
                chk = float(np.abs(lg_chk[0].float().cpu().numpy() - fan_logits).max())
                fanpath_tok.append(chk)
                zs = store[log]["z"]
                for si in store[log]["by_tok"].get(tok, []):
                    feat_box.clear()
                    cxy = torch.from_numpy(zs["traj3"][si]).float().unsqueeze(0).to(planner.device)          # [1, 64, 20, 3]
                    with torch.no_grad():
                        lg_set = planner.model.score_trajectories(cxy, sctx)                                     # [1, 64, 6]
                    if len(feat_box) != 1:
                        raise RuntimeError(f"token {tok} set {si}: the score_head hook fired {len(feat_box)} times in one score_trajectories (want 1)")
                    f_set = feat_box[0][0][0]                                                                  # [64, dec_width] fp16
                    lo = lg_set[0].float().cpu().numpy()
                    if not np.array_equal(feat_box[0][1][0], lo):
                        raise RuntimeError(f"token {tok} set {si}: the hooked score_head output is not the logits score_trajectories returned")
                    p_ = 1.0 / (1.0 + np.exp(-lo.astype(np.float64)))
                    agg = p_[:, 0] * p_[:, 1] * (5 * p_[:, 2] + 5 * p_[:, 3] + 2 * p_[:, 4]) / 12.0
                    top8 = np.argsort(-agg, kind="stable")[:8].astype(np.int16)
                    out_sets["log"].append(log)
                    out_sets["token"].append(tok)
                    out_sets["step"].append(int(zs["step"][si]))
                    out_sets["rank"].append(int(zs["rank"][si]))
                    out_sets["ckpt"].append(int(zs["ckpt_step"][si]))
                    out_sets["feat"].append(f_set)
                    out_sets["logits"].append(lo.astype(np.float32))
                    out_sets["top8"].append(top8)
                    out_sets["labels6"].append(zs["labels6"][si])
                    out_sets["ctrl_traj"].append(float(np.abs(fan3 - zs["traj3"][si]).max()))
                    out_sets["ctrl_logit"].append(float(np.abs(fan_logits - lo).max()))
                    out_sets["ctrl_fanpath"].append(chk)
        for tok in set(lt) - got:
            misses.append((tok, "scenario builder skipped the token (margin guard)"))
        print(f"    [{li + 1}/{len(by_log)}] {log} rows {len(rows_tok):,} misses {len(misses)} {time.time() - t0:.0f} s", flush=True)
    hook.remove()
    hook2.remove()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.lean:
        dump_props, dump_props20, dump_feat = [], [], []
    np.savez(a.out, token=np.array(rows_tok), fingerprint=np.array(rows_fp), proposals=np.stack(dump_props) if dump_props else np.zeros((0, 64, NAVSIM_N, 3), np.float32),
             proposals20=np.stack(dump_props20) if dump_props20 else np.zeros((0, 64, 20, 3), np.float32),
             logits=np.stack(dump_logits) if dump_logits else np.zeros((0, 64, 6), np.float32), pick=np.array(dump_pick, dtype=np.int64),
             log=np.array([plan["token_log"][t] for t in rows_tok]), select=np.array(a.select), rule=np.array(planner.rule),
             repair_last_heading=np.array(planner.repair_last_heading), ckpt=np.array(str(a.ckpt)), sampling=np.array([NAVSIM_N, NAVSIM_DT]))
    if a.feat_out and dump_feat:
        os.makedirs(os.path.dirname(os.path.abspath(a.feat_out)), exist_ok=True)
        np.savez(a.feat_out, token=np.array(rows_tok), log=np.array([plan["token_log"][t] for t in rows_tok]), prehead_features=np.stack(dump_feat),
                 what=np.array("input of model.score_head, [N, 64 proposals, dec_width], fp16, captured by a forward hook in the same forward as the logits"))
    margin_only = all("margin guard" in m for _t, m in misses)             # a token the scenario builder cannot build (too close to the log's end) is DISCLOSED, any other miss fails
    n_sets = len(out_sets["token"])
    if a.sets_out and a.sets_dir:
        os.makedirs(os.path.dirname(os.path.abspath(a.sets_out)), exist_ok=True)
        S = out_sets
        np.savez(a.sets_out, log=np.array(S["log"]), token=np.array(S["token"]), step=np.array(S["step"], np.int32), rank=np.array(S["rank"], np.int32),
                 ckpt_step=np.array(S["ckpt"], np.int64), feat=np.stack(S["feat"]) if n_sets else np.zeros((0, 64, 256), np.float16),
                 logits=np.stack(S["logits"]) if n_sets else np.zeros((0, 64, 6), np.float32), top8=np.stack(S["top8"]) if n_sets else np.zeros((0, 8), np.int16),
                 labels6=np.stack(S["labels6"]) if n_sets else np.zeros((0, 64, 6), np.float32), label_order=np.array(["nc", "dac", "ep", "ttc", "c", "ddc"]),
                 ctrl_traj_maxabs=np.array(S["ctrl_traj"], np.float64), ctrl_logit_maxabs=np.array(S["ctrl_logit"], np.float64), ctrl_fanpath_logit_maxabs=np.array(S["ctrl_fanpath"], np.float64), rule=np.array(planner.rule), ckpt=np.array(str(a.ckpt)),
                 what=np.array("P2 scorer on the fit sets' own 64 proposals: feat = input of score_head [S,64,256] fp16; logits [S,64,6]; top8 = frozen contender membership; labels6 = P2's PDM training labels"))
    got_set = set(rows_tok)
    sets_expected = sum(len(store[lg]["by_tok"].get(t, [])) for lg, lt in by_log.items() for t in lt if t in got_set) if a.sets_dir else 0
    rep = {"tokens_asked": len(toks), "sets_asked": sets_expected, "sets_rows": n_sets, "rows": len(rows_tok), "ctrl_fanpath_max": (max(fanpath_tok) if fanpath_tok else None), "ctrl_fanpath_n_tokens": len(fanpath_tok), "misses": len(misses), "miss_examples": misses[:10], "miss_tokens": [t for t, _m in misses], "margin_only": bool(margin_only), "ckpt": a.ckpt, "rule": planner.rule,
           "repair_last_heading": planner.repair_last_heading, "goal_fix": planner.goal_fix, "nav_mode": nav_mode, "sanitize_goal": planner.sanitize_goal,
           "commands": ({"path": cmd_table["path"], "sha256": cmd_table["sha256"], "rows": len(cmd_table["rows"]), "set": n_cmd_set} if cmd_table is not None else None),
           "prehead_feature_shape": list(np.stack(dump_feat).shape) if dump_feat else None, "lean": bool(a.lean), "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    ok = (len(rows_tok) == len(toks) or (a.sets_dir and margin_only)) and (not a.sets_dir or n_sets == sets_expected)
    print(f"ZZP2HODUMP_{'OK' if ok else 'FAIL'} {len(rows_tok)}")
    print(f"ZZS0DUMP_{'OK' if ok else 'FAIL'} {len(rows_tok)} {n_sets}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
