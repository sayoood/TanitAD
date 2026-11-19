"""make_actlevel2.py : writes actlevel/ops/pod_actlevel2_auto.sh (note PREREG_V8F_PROOF_ADD37_N7: arms RLVc and RLVd at bases 0 and 1)
from actlevel/ops/pod_actlevel_auto.sh (note N6) by asserted text edits. Every old text must occur exactly as often as stated; the source is not changed.
Unchanged from the source: the pins check, mk(), slot(), the selfcheck body apart from the weight table and its output folder, dry(), start()."""
import hashlib
import io
import os

H = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(H, "pod_actlevel_auto.sh"), encoding="utf-8", newline="").read()
assert "\r" not in src
t = src


def rep(old, new, n=1):
    global t
    assert t.count(old) == n, (old[:70], t.count(old), n)
    t = t.replace(old, new)


def cut(start, end, new):
    """replace everything from the start marker to the end of the line that holds the end marker"""
    global t
    assert t.count(start) == 1 and t.count(end) == 1, (start[:50], t.count(start), end[:50], t.count(end))
    a = t.index(start)
    b = t.index(end, a)
    b = t.index("\n", b) + 1
    t = t[:a] + new + t[b:]


# 1. header
a = t.index("set -u\n")
t = ("# pod_actlevel2_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37, note N7 (two interior weights of the level term).\n"
     "# Made by actlevel/ops/make_actlevel2.py from pod_actlevel_auto.sh (note N6). Two chains, one per base k in 0 1:\n"
     "# actlevel2_<k> = RLVc_s<k> (W 0.001, V 0.002), RLVd_s<k> (W 0.001, V 0.004), each the line of RLAm_s<k> of arms_actcons_c<k>.txt with the tree, the name\n"
     "# and the named flags changed. Tree: /workspace/v8proof/tree_actlevel (pins unchanged).\n"
     "# Before a chain starts: a free slot (fewer than three GPU jobs on two probes 45 s apart) with 28 GB of the memory limit free, the pins of the tree, the dry\n"
     "# gates of its arms (selfcheck only, they train nothing) and a check of the selfchecks (both weights, start checkpoint). Chain actlevel2_1 starts only after\n"
     "# chain actlevel2_0 has started and a slot is free again. Fail closed: a refusal writes ACTLEVEL2_NOT_STARTED.txt and ends the script. No process test\n"
     "# decides anything. It deletes nothing.\n") + t[a:]

# 2. folders, markers, chain names
rep("Z=$W/runs/actlevel_z; LOG=$Z/actlevel_auto.log;", "Z=$W/runs/actlevel2_z; LOG=$Z/actlevel2_auto.log;")
rep("> $Z/ACTLEVEL_NOT_STARTED.txt", "> $Z/ACTLEVEL2_NOT_STARTED.txt")
rep("for c in actlevel_0 actlevel_1; do [ -e $W/chains/$c ]", "for c in actlevel2_0 actlevel2_1; do [ -e $W/chains/$c ]")

# 3. arms files
cut("for k in 0 1; do\n  A=arms_actlevel_$k.txt", 'stopf "arms file $A is not the three lines this script expects"\ndone',
    "for k in 0 1; do\n"
    "  A=arms_actlevel2_$k.txt\n"
    "  [ -e $A ] || { { echo \"# Two interior weights of the level term ($CARD, note N7), base $k: RLVc (W 0.001, V 0.002), RLVd (W 0.001, V 0.004); each the line of RLAm_s$k with the tree, the name and the named flags changed.\"\n"
    "      grep '^PIN|' $PINS\n"
    "      mk arms_actcons_c$k.txt RLAm_s$k RLVc_s$k 0.001 0.002\n"
    "      mk arms_actcons_c$k.txt RLAm_s$k RLVd_s$k 0.001 0.004; } > $A\n"
    "    say \"arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)\"; }\n"
    "  [ \"$(grep -c '^ARM|' $A)\" = 2 ] && [ \"$(grep -c \"^ARM|[A-Za-z0-9]*_s$k|$TREE|\" $A)\" = 2 ] && [ \"$(grep -c \"runs/imag_a/RL_s$k/ckpt_last.pt\" $A)\" = 2 ] \\\n"
    "    && [ \"$(grep \"^ARM|RLVc_s$k|\" $A | grep -c -- '--w-act-cons 0.001 --w-act-level 0.002|.*levers.w_act_cons=0.001 levers.act_cons_max=1.5 levers.w_act_level=0.002 ')\" = 1 ] \\\n"
    "    && [ \"$(grep \"^ARM|RLVd_s$k|\" $A | grep -c -- '--w-act-cons 0.001 --w-act-level 0.004|.*levers.w_act_cons=0.001 levers.act_cons_max=1.5 levers.w_act_level=0.004 ')\" = 1 ] || stopf \"arms file $A is not the two lines this script expects\"\n")

# 4. the waits for the earlier chains of note N6 are not part of this script (the slot wait decides)
cut("# order of the pod (second version", '  say "queue of card ADD42: ',
    "# order of the pod: first the rollouts of the LS dose arms (note ADD42_N3) must have ended (their DONE or STOP marker; at most 4 h, then it goes on);\n"
    "# after that the slot wait decides (fewer than three GPU jobs and 28 GB of memory free).\n"
    "j=0; while [ -d $W/qhead_dose_ls/run_roll ] && [ ! -e $W/qhead_dose_ls/run_roll/ROLLOUTS_DONE.txt ] && [ ! -e $W/qhead_dose_ls/run_roll/ROLLOUTS_STOP.txt ]; do j=$((j+1)); [ $j -gt 240 ] && { say 'the rollouts of the LS dose arms have not ended in 4 h; going on'; break; }; sleep 60; done\n"
    "say \"rollouts of the LS dose arms: $(ls $W/qhead_dose_ls/run_roll/ 2>/dev/null | grep -c ROLLOUTS_) end marker(s)\"\n")

# 5. selfcheck: weight table and output folder
rep("WT = {'RLVa': (0.001, 0.001), 'RLVb': (0.001, 0.01), 'RLAh': (0.0003, 0.0)}", "WT = {'RLVc': (0.001, 0.002), 'RLVd': (0.001, 0.004)}")
rep("'/workspace/v8proof/runs/actlevel_z/selfcheck_%s_pass.txt'", "'/workspace/v8proof/runs/actlevel2_z/selfcheck_%s_pass.txt'")

# 6. dry gate outputs
rep("$W/chains/actlevel_dry_$a.out", "$W/chains/actlevel2_dry_$a.out", 3)

# 7. main
a = t.index("say 'auto start (card ADD37, note N6)'")
t = t[:a] + ("say 'auto start (card ADD37, note N7)'\n"
             "slot || stopf 'no free GPU slot with 28 GB of memory in 8 h'\n"
             "ARMSF=arms_actlevel2_0.txt; dry RLVc_s0 RLVd_s0\n"
             "selfcheck k0 RLVc_s0 RLVd_s0 || stopf 'a selfcheck at base 0 does not show the weights or the start checkpoint its arm names'\n"
             "start actlevel2_0 arms_actlevel2_0.txt\n"
             "slot || stopf 'actlevel2_1: no free GPU slot with 28 GB of memory in 8 h'\n"
             "ARMSF=arms_actlevel2_1.txt; dry RLVc_s1 RLVd_s1\n"
             "selfcheck k1 RLVc_s1 RLVd_s1 || stopf 'a selfcheck at base 1 does not show the weights or the start checkpoint its arm names'\n"
             "start actlevel2_1 arms_actlevel2_1.txt\n"
             "say 'auto end'\n")

for bad in ("actlevel_z", "actlevel_0", "actlevel_1", "RLVa", "RLVb", "RLAh", "ACTLEVEL_NOT", "waitq", "ended()", "note N6)", "actlevel_dry"):
    assert bad not in t.replace("pod_actlevel_auto.sh (note N6)", ""), bad
out = os.path.join(H, "pod_actlevel2_auto.sh")
io.open(out, "w", encoding="utf-8", newline="\n").write(t)
print("written", out, "lines", t.count("\n"), "sha256", hashlib.sha256(t.encode("utf-8")).hexdigest()[:16])
