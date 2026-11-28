"""Emit the v7f S-W and S-T launch lines (Thor paths mirroring the gate's ST_ARGV) and adjudicate them:
  (1) the launch gate's own v7f argv rules, whole-profile AND per rule row;
  (2) a flag-by-flag diff against the gate's rehearsed ST_ARGV / SW_ARGV;
  (3) the TRAINER's own preflight on the REAL (non-dry) lines, with the Thor data paths swapped for
      local copies of the same files (canonical v7.2 blob + a sidecar built over it), the corpus flags
      pointed at an EMPTY dir (the gate's rehearsal trick) and --init-from/--prev-gate at dummies.
usage: python emit_and_gate.py <out_dir> <local_labels_blob> <local_sidecar>
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

out_dir, local_blob, local_side = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out_dir.mkdir(parents=True, exist_ok=True)
STACK = Path("<V7F_CHAIN>/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))
import v6_chain as C                                            # noqa: E402
import launch_gate as LG                                        # noqa: E402

THOR = dict(root="/home/nvidia/experiments", workdir="/home/nvidia/TanitAD/stack",
            train_cache="/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
            val_cache="/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
            nav_labels="/home/nvidia/data/v72/nav.jsonl.gz",
            s2_labels="/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz",
            speed_max_sidecar="/home/nvidia/data/refcv6_speed_max_v8_train.jsonl")


def trainer_parser():
    import train_v6_staged as T
    ap = T.build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    return T, ap


def sw_record(sw_argv) -> dict:
    """What the trainer records as config.json['args'] for exactly this S-W line (the _run_config rule:
    vars(a) minus the `_ew_*` carriers, tuples as lists)."""
    _T, ap = trainer_parser()
    a = ap.parse_args(sw_argv)
    return {"args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items()
                     if not k.startswith("_ew_")},
            "_read": "DERIVED: the trainer's REAL parser applied to the chain's emitted S-W argv -- what "
                     "S-T carries IF S-W runs with exactly that line. On Thor the real record wins."}


cfg = C.v7f_config(**THOR)
plan = C.build_plan(cfg)
sw, st = C.step_by_key(plan, "S-W"), C.step_by_key(plan, "S-T")
sw_argv = C.trainer_argv(sw, cfg, plan)
rec_p = out_dir / "sw_record" / "config.json"
rec_p.parent.mkdir(parents=True, exist_ok=True)
rec_p.write_text(json.dumps(sw_record(sw_argv), indent=1), encoding="utf-8")
cfg.geometry_from = str(rec_p)
st_argv = C.trainer_argv(st, cfg, plan)
lines = {"S-W": C.launch_line(sw, cfg, plan), "S-T": C.launch_line(st, cfg, plan)}
argvs = {"S-W": sw_argv, "S-T": st_argv}

# ---------------------------------------------------------------- (1) gate rules
prof = LG.PROFILES["v7f"]


def per_rule(argv):
    stage = (LG.flag_values(argv, "--stage") or [None])[0]
    rows = []
    rows.append({"rule": "stage_allowed", "req": "R6", "flag": "--stage",
                 "want": list(prof["stage_allowed"]), "have": stage,
                 "verdict": "PASS" if stage in prof["stage_allowed"] else "REFUSED"})
    for f, why in prof["required_on_v6"].items():
        rows.append({"rule": "required_on (every stage)", "req": why.split(" ")[0], "flag": f,
                     "have": LG.has_flag(argv, f),
                     "verdict": "PASS" if LG.has_flag(argv, f) else "REFUSED"})
    for f, why in (prof["required_on_v6_stage"].get(stage) or {}).items():
        rows.append({"rule": f"required_on ({stage})", "req": why.split(" ")[0], "flag": f,
                     "have": LG.flag_values(argv, f),
                     "verdict": "PASS" if LG.has_flag(argv, f) else "REFUSED"})
    for f, why in (prof["required_positive_v6_stage"].get(stage) or {}).items():
        v = LG.flag_values(argv, f)
        ok = bool(v) and float(v[0]) > 0
        rows.append({"rule": f"required_positive ({stage})", "req": why.split(" ")[0], "flag": f,
                     "have": v, "verdict": "PASS" if ok else "REFUSED"})
    for f, why in prof["forbidden_flags_v6"].items():
        rows.append({"rule": "forbidden_flag", "req": why.split(":")[0], "flag": f,
                     "have": LG.has_flag(argv, f),
                     "verdict": "REFUSED" if LG.has_flag(argv, f) else "PASS"})
    for f, (bad, why) in prof["forbidden_values_v6"].items():
        v = LG.flag_values(argv, f)
        rows.append({"rule": "forbidden_values", "req": why.split(" ")[0], "flag": f, "bad": list(bad),
                     "have": v, "verdict": "REFUSED" if (v and v[0] in bad) else "PASS"})
    for f, why in prof["forbidden_positive_v6"].items():
        v = LG.flag_values(argv, f)
        rows.append({"rule": "forbidden_positive", "req": "R6", "flag": f, "have": v,
                     "verdict": "REFUSED" if (v and float(v[0]) > 0) else "PASS"})
    for f, why in prof["pi_pending_values"].items():
        v = LG.flag_values(argv, f)
        rows.append({"rule": "pi_pending_values", "req": "R4 / D1", "flag": f, "have": v,
                     "verdict": "PI-DECISION" if v else "n/a (flag absent)"})
    return rows


gate = {}
for k, av in argvs.items():
    refusals, pending = LG.profile_argv_rules(prof, av)
    gate[k] = {"profile_argv_rules": {"refusals": refusals, "pi_pending": pending},
               "per_rule": per_rule(av),
               "chain_rules": C.v7f_argv_rules(C.step_by_key(plan, k), av)}

# ---------------------------------------------------------------- (2) diff vs the gate's ST_ARGV / SW_ARGV
spec = importlib.util.spec_from_file_location("_tlg7f", str(STACK / "tests" / "test_launch_gate_v7f.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["_tlg7f"] = mod
spec.loader.exec_module(mod)
ref = {"S-T": mod.ST_ARGV, "S-W": mod.SW_ARGV}


def as_map(av):
    m = {}
    for f, v in LG.flag_pairs(av):
        m[f] = v
    return m


diff = {}
for k in ("S-W", "S-T"):
    a, b = as_map(argvs[k]), as_map(ref[k])
    diff[k] = {"only_in_chain": {f: a[f] for f in a if f not in b},
               "only_in_gate_argv": {f: b[f] for f in b if f not in a},
               "value_differs": {f: {"chain": a[f], "gate": b[f]} for f in a if f in b and a[f] != b[f]},
               "same": sorted(f for f in a if f in b and a[f] == b[f])}

# ---------------------------------------------------------------- (3) the trainer's own preflight, real lines
T, ap = trainer_parser()
empty = out_dir / "EMPTY_v2cache"
empty.mkdir(exist_ok=True)
sub = {THOR["nav_labels"]: local_blob, THOR["s2_labels"]: local_blob,
       THOR["speed_max_sidecar"]: local_side, THOR["train_cache"]: str(empty),
       THOR["val_cache"]: str(empty)}
pre = {}
for k, av in argvs.items():
    loc = [sub.get(x, x) for x in av]
    loc = LG.set_flag(loc, "--out", [str(out_dir / f"pf_{k}")])
    a = ap.parse_args(loc)
    a._ew_parser, a._ew_argv = ap, list(loc)
    probs = T.preflight(a)
    pre[k] = {"n_refusals": len(probs), "refusals": probs,
              "substituted": {kk: vv for kk, vv in sub.items() if kk in av},
              "_read": "trainer.preflight on the NON-dry emitted line, Thor data paths -> local copies of "
                       "the same files, corpus -> an EMPTY dir; --init-from/--prev-gate/--dump-seam-plan "
                       "kept as emitted (preflight checks their presence / importability, not the files)"}

res = {"cfg": {k: getattr(cfg, k) for k in ("root", "workdir", "train_cache", "val_cache", "nav_labels",
                                            "s2_labels", "speed_max_sidecar", "tac_op_cond",
                                            "n_candidates", "tac_goal_cond", "sw_dir")},
       "geometry_from": str(rec_p), "argv": argvs, "launch_line": lines, "gate": gate,
       "diff_vs_gate_argv": diff, "trainer_preflight": pre}
(out_dir / "emitted_and_gate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
for k in ("S-W", "S-T"):
    print(f"== {k}\n{lines[k]}\n")
    g = gate[k]
    print(f"gate refusals={g['profile_argv_rules']['refusals']} pi_pending="
          f"{[p.split(':')[0] for p in g['profile_argv_rules']['pi_pending']]} "
          f"chain_refusals={g['chain_rules']['chain_refusals']}")
    for r in g["per_rule"]:
        print(f"   {r['verdict']:12s} {r['req']:8s} {r['rule']:30s} {r['flag']}")
    d = diff[k]
    print(f"   only_in_chain: {sorted(d['only_in_chain'])}")
    print(f"   only_in_gate_argv: {d['only_in_gate_argv']}")
    print(f"   value_differs: {d['value_differs']}")
    print(f"   trainer preflight refusals: {pre[k]['n_refusals']}")
    for p in pre[k]["refusals"]:
        print("     !", p[:300])
