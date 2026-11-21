#!/usr/bin/env python3
"""Self-checks for the UNSEEN-POOL-2 probe (stdlib only, CPU, reads the package's raw/ and the same local inputs). Run from the package root:
    python -I code/test_pool.py
Every check has a control that must read a known value; the last block is a MUTATION arm that must make the headline check FAIL."""
import sys, os, json, importlib.util, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print("PASS", name, detail)
    else:
        fail += 1; print("FAIL", name, detail)
spec = importlib.util.spec_from_file_location("draw_reserve_ref", os.path.join(os.path.dirname(os.path.abspath(__file__)), "draw_reserve_REF.py"))
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
B = "D:/TanitAstraResearch/data/e024_corpus_20261010/"
NT = [l.strip() for l in open("raw/src/navtrain_logs.txt") if l.strip()]; NS = json.load(open("raw/src/navtest_logs.json"))["logs"]
TVm = json.load(open("raw/members_trainval_tgz.json"))["members"]; TV = sorted(m["name"].rsplit("/", 1)[-1][:-4] for m in TVm if m["isfile"] and m["name"].endswith(".pkl"))
TEST = json.load(open("raw/test_release_logs.json"))["test_logs"]; MINI = json.load(open("raw/mini_logs.json"))["mini_logs"]
# T1 definitions identical to the reference draw rule, on every name we touch
allnames = set(NT) | set(NS) | set(TV) | set(TEST) | set(MINI)
check("T1 session_of == draw_reserve.session_of on %d names" % len(allnames), all(session_of(l) == ref.session_of(l) for l in allnames))
check("T1 vehicle-day == draw_reserve.unit_of(vehicle-day)", all(vehicle_day_of(session_of(l)) == ref.unit_of(ref.session_of(l), "vehicle-day") for l in allnames))
# T2 control values stated in the brief
cn, cs = counts(NT), counts(NS)
check("T2 navtrain 1192/162/46", (cn["logs"], cn["sessions"], cn["vehicle_days"]) == (1192, 162, 46), str(cn))
check("T2 navtest 136/44/20", (cs["logs"], cs["sessions"], cs["vehicle_days"]) == (136, 44, 20), str(cs))
nts = {session_of(l) for l in NT}; nss = {session_of(l) for l in NS}
check("T2 navtrain and navtest share 0 sessions and 0 vehicle-days", not (nts & nss) and not ({vehicle_day_of(s) for s in nts} & {vehicle_day_of(s) for s in nss}))
# T3 release structure
check("T3 trainval tar listing has 1310 distinct pkl members", len(TV) == 1310 == len(set(TV)))
check("T3 navtrain subset of trainval, navtest subset of test", set(NT) <= set(TV) and set(NS) <= set(TEST))
check("T3 trainval and test disjoint; trainval and navtest disjoint", not (set(TV) & set(TEST)) and not (set(TV) & set(NS)))
check("T3 mini = 64 logs, all inside trainval or test", len(MINI) == 64 and set(MINI) <= (set(TV) | set(TEST)))
# T4 byte-exact prediction of the three nuPlan public_set txt sizes (S3 listing) from the hypothesised log sets
def pred(lens, groups): return sum(L + 1 for L in lens) + sum(len("File group: %d" % k) + 1 for k in range(groups))
val = {l.strip() for l in open("raw/src/public_set_val_sensor.txt") if l.strip() and not l.startswith("File group")}
size = {k["key"].rsplit("/", 1)[-1]: k["size"] for s in ("mini_set", "test_set", "train_set", "val_set") for k in json.load(open("raw/s3_list_%s.json" % s))["keys"] if k["key"].endswith(".txt")}
check("T4 control: val txt predicted bytes == actual (8945)", pred([len(l) for l in val], 12) == 8945 == size["public_set_val_sensor.txt"])
check("T4 train txt: (trainval minus val) predicts 42950 B", pred([len(l) for l in TV if l not in val], 43) == size["public_set_train_sensor.txt"] == 42950)
check("T4 test txt: 147 test logs predicts 5903 B", pred([len(l) for l in TEST], 12) == size["public_set_test_sensor.txt"] == 5903)
check("T4 mini txt: 64 mini logs predicts 2622 B", pred([len(l) for l in MINI], 9) == size["nuplan_mini_sensor.txt"] == 2622)
# T5 the independent third-party mirror listing agrees by NAME
mir = {sp: {i["path"].rsplit("/", 1)[-1] for i in json.load(open("raw/hf_other/kesai_logs_nuplan_%s.json" % sp))} for sp in ("train", "val", "test")}
check("T5 mirror train|val == OpenScene trainval, mirror test == OpenScene test", (mir["train"] | mir["val"]) == set(TV) and mir["test"] == set(TEST), str({k: len(v) for k, v in mir.items()}))
# T6 headline numbers
L = json.load(open("raw/pool_lists.json")); u = L["universe_public_sensor_logs"]
check("T6 universe 1457 logs; outside navtrain U navtest = 129 logs, 11 sessions, 0 vehicle-days", (u["logs"], u["outside_navtrain_union_navtest_logs"], u["outside_sessions"], u["outside_vehicle_days"]) == (1457, 129, 11, 0), str(u))
tv_out = L["trainval_not_in_navtrain"]; te_out = L["test_not_in_navtest"]
un = [r for r in tv_out if not r["session_in_navtrain"]]
check("T6 trainval: 118 outside navtrain, 58 logs in 9 sessions absent from navtrain", len(tv_out) == 118 and len(un) == 58 and len({r["session"] for r in un}) == 9)
check("T6 all 9 unseen sessions have >=1 navtrain sibling session on the same vehicle-day (0 vehicle-day-unseen)", all(r["vehicle_day_in_navtrain"] for r in un))
check("T6 scorable (has_route) among the 58: exactly 2 logs, 1 session", sum(1 for r in un if r["scorable_has_route"]) == 2 and len({r["session"] for r in un if r["scorable_has_route"]}) == 1)
check("T6 test: 11 outside navtest, 9 route-less", len(te_out) == 11 and sum(1 for r in te_out if not r["scorable_has_route"]) == 9)
# T7 profile controls: the probe must read known values
P = json.load(open("raw/profile_logs.json"))["logs"]; W = json.load(open("raw/wanted_profile.json")); D = json.load(open("raw/navtrain_density.json"))
check("T7 control: 40 navtrain logs profiled from the tgz reproduce the independent compact_metadata parse (frames, route frames, anchors)", all(P[l]["n_frames"] == D[l]["n_frames"] and P[l]["n_route_frames"] == D[l]["n_route_frames"] and P[l]["anchors_route"] == D[l]["anchors_route"] for l in W["controls"]))
check("T7 control: navtrain has 0 route-less logs (1192 checked)", sum(1 for v in D.values() if v["n_route_frames"] == 0) == 0 and len(D) == 1192)
check("T7 probe is able to read zero: outside logs include route-less ones", sum(1 for r in tv_out + te_out if not r["scorable_has_route"]) == 69)
# T8 MUTATION arm: a wrong session rule must break the headline control (so T2 is not vacuous)
mut = lambda l: l.rsplit("_", 1)[0]
n_mut = len({mut(l) for l in NT})
check("T8 mutation (rsplit('_',1)) changes navtrain session count away from 162 -> T2 would FAIL", n_mut != 162, "mutated count %d" % n_mut)
print("\nSUMMARY pass=%d fail=%d" % (ok, fail))
sys.exit(1 if fail else 0)
