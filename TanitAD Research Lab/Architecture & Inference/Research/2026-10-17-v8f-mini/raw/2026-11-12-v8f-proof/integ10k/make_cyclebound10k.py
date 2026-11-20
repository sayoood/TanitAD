"""make_cyclebound10k.py -- writes the generated run tools of card PREREG_V8F_PROOF_ADD69 (CYCLEBOUND10K: the 1.0 s planning cycle on a kinematic executor with a heading
bound; no training) by asserted text edits (each old text the counted number of times):
  pod_boundtier10k_queue.sh -> pod_cyclebound10k_queue.sh
  chain_boundtier10k.sh     -> chain_cyclebound10k.sh
The originals are not changed. The rollout wrapper is that of card ADD67, unchanged (v8f_t1_eval_reader_rule_rzgfh.py, sha256 373ffe72f2655c35, asserted here).
Pure text generator (standard library only). Usage: python make_cyclebound10k.py (in this folder)."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_cyclebound10k.py"
WRAPPER_SHA = "373ffe72f2655c35"


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert ME in head, "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes) sha256 %s" % (name, len(text.encode("utf-8")), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))


def code_rep(lines, old, new, n):
    c = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            continue
        k = ln.count(old)
        if k:
            lines[i] = ln.replace(old, new)
            c += k
    assert c == n, "anchor %r occurs %d times in code lines, expected %d" % (old[:60], c, n)


def count(lines, s):
    return sum(ln.count(s) for ln in lines if not ln.startswith("#"))


def index_of(lines, start):
    hit = [i for i, ln in enumerate(lines) if ln.startswith(start)]
    assert len(hit) == 1, "line start %r found %d times" % (start[:60], len(hit))
    return hit[0]


def queue():
    L = load("pod_boundtier10k_queue.sh").split("\n")
    assert L[0].startswith("#!") and L[1].startswith("# pod_boundtier10k_queue.sh -- written by make_boundtier10k.py")
    hdr = [
        "# pod_cyclebound10k_queue.sh -- written by make_cyclebound10k.py from pod_boundtier10k_queue.sh. Card PREREG_V8F_PROOF_ADD69 (CYCLEBOUND10K): the three tool checks of card ADD67",
        "# again (control y3; marker cb6check), then the control I10R0M5HbID (no rule, --readout pinned, --kappa-max 1e9, 5 steps, model seed 0, compared file for file with the banked",
        "# dump of I10R0M5_s0: marker cb1cmp), then at two model seeds six rollouts at a cycle of 5 cache steps (1.0 s) with the bound 0.2: I10R0M5Hb20 (no rule, pinned), I10R0KM5Hb20",
        "# (no rule, readout K), I10RLcM5Hb20 and I10RQ50M5Hb20 (the rule at every plan, pinned), I10RLcKM5Hb20 and I10RQ50KM5Hb20 (the rule at every plan, readout K).",
        "# Wrapper v8f_t1_eval_reader_rule_rzgfh.py of card ADD67, unchanged. No training. The queue waits for the end marker of the queue of card ADD68 (WAIT_DONE / WAIT_STOP, defaults",
        "# BOUNDTIER10K_DONE.txt / BOUNDTIER10K_STOP.txt in the folder $W). Markers START_CYCLEBOUND10K.txt / CYCLEBOUND10K_STOP.txt / CYCLEBOUND10K_DONE.txt; step markers cb6check,",
        "# cb1cmp. The folder and tag names keep the ending m10_causal_heads (the readers find the dumps by it); the cycle of a run is in its run record.",
        "# The header lines below are those of the earlier queues.",
    ]
    L = [L[0]] + hdr + L[1:]
    # the wait defaults first (they hold the names of the queue of card ADD67), then this card's own markers
    code_rep(L, "WAIT_DONE=${WAIT_DONE:-HEADBOUND10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-HEADBOUND10K_STOP.txt}", "WAIT_DONE=${WAIT_DONE:-@@WD@@}; WAIT_STOP=${WAIT_STOP:-@@WS@@}", 1)
    assert count(L, "HEADBOUND10K") == 0
    code_rep(L, "BOUNDTIER10K_STOP.txt", "CYCLEBOUND10K_STOP.txt", 1)
    code_rep(L, "START_BOUNDTIER10K.txt", "START_CYCLEBOUND10K.txt", 2)
    code_rep(L, "BOUNDTIER10K_DONE", "CYCLEBOUND10K_DONE", 2)
    assert count(L, "BOUNDTIER10K") == 0
    code_rep(L, "@@WD@@", "BOUNDTIER10K_DONE.txt", 1)
    code_rep(L, "@@WS@@", "BOUNDTIER10K_STOP.txt", 1)
    code_rep(L, "card ADD68: three tool checks, then two controls and sixteen rollouts on the bounded executor", "card ADD69: three tool checks, then one control and twelve rollouts at a 1.0 s cycle on the bounded executor", 1)
    code_rep(L, "the rollouts of card ADD68 start", "the rollouts of card ADD69 start", 1)
    code_rep(L, "for nm in I10R0 I10R0C10 I10RLc I10RQ50 I10R0Hb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do",
             "for nm in I10R0 I10RLc I10RQ50 I10R0M5 I10RLcM5 I10RQ50M5 I10R0Hb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do", 1)
    code_rep(L, "&& for nm in I10R0Hb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do", "&& for nm in I10RLcHb20 I10RQ50Hb20; do", 1)
    code_rep(L, "tick files of the four reused runs of card ADD67 (both model seeds)", "tick files of the two reused runs of card ADD68 (both model seeds)", 1)
    code_rep(L, "hashes of the reused runs of card ADD67 taken", "hashes of the reused runs of card ADD68 taken", 1)
    n_small, n_test, n_chk = count(L, "small_bt"), count(L, "bt_test"), count(L, "bt6check")
    assert n_small >= 6 and n_test >= 10 and n_chk == 2, (n_small, n_test, n_chk)
    code_rep(L, "small_bt", "small_cb", n_small)
    code_rep(L, "bt_test", "cb_test", n_test)
    code_rep(L, "bt6check", "cb6check", n_chk)
    a = index_of(L, "READOUT=causal KAPPA=1e9 rollout I10R0C10HbID R0 0 0")
    b = index_of(L, 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
    old = L[a:b]
    assert len(old) == 22 and old[-1] == "done" and sum(1 for ln in old if " rollout " in ln) == 10, (len(old), old[-1])
    new = [
        'REPLAN=5 READOUT=pinned KAPPA=1e9 rollout I10R0M5HbID R0 0 0       # control y1: the bound with a huge K at a cycle of 5 steps, pinned readout, no rule, must give the banked dump of I10R0M5 at model seed 0',
        'if [ "$DRY" != 1 ] && ! done_step cb1cmp; then',
        '  mkdir -p "$W/small_cb"',
        '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0M5_s0/dump_I10R0M5_s0_m10_causal_heads" "$W/t1/I10R0M5HbID_s0/dump_I10R0M5HbID_s0_m10_causal_heads" > "$W/small_cb/CMP_R0M5HbID.txt" 2>&1 || stop "Y1-RERUN-DIFFERS"',
        '  mark cb1cmp',
        'fi',
        'for k in "${ARMS[@]}"; do',
        '  REPLAN=5 READOUT=pinned KAPPA=0.2 rollout I10R0M5Hb20 R0 "$k" "$k"',
        '  REPLAN=5 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10R0KM5Hb20 R0 "$k" "$k"',
        '  REPLAN=5 READOUT=pinned KAPPA=0.2 rollout I10RLcM5Hb20 RLc "$k" "$k"',
        '  REPLAN=5 READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50M5Hb20 RR "$k" "$k"',
        '  REPLAN=5 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10RLcKM5Hb20 RLc "$k" "$k"',
        '  REPLAN=5 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50KM5Hb20 RR "$k" "$k"',
        'done',
    ]
    L = L[:a] + new + L[b:]
    for bad in ("bt1cmp", "bt2cmp", "C10HbID"):
        assert all(bad not in ln for ln in L if not ln.startswith("#")), bad
    save("pod_cyclebound10k_queue.sh", "\n".join(L))


def chain():
    L = load("chain_boundtier10k.sh").split("\n")
    assert L[0].startswith("#!")
    hdr = ["# chain_cyclebound10k.sh -- written by make_cyclebound10k.py from chain_boundtier10k.sh. Card PREREG_V8F_PROOF_ADD69: pushes the queue of the card, starts it (it waits on the",
           "# pod for the end marker of the queue of card ADD68), waits for its end marker and pulls the small folder. The run folders stay on the pod.",
           "# usage: bash chain_cyclebound10k.sh [--no-start]      last line: ZZ CHAIN-CYCLEBOUND10K DONE | STOP"]
    L = [L[0]] + hdr + L[1:]
    n1, n2, n3 = count(L, "BOUNDTIER10K"), count(L, "boundtier"), count(L, "small_bt")
    assert n1 >= 8 and n2 >= 4 and n3 >= 3, (n1, n2, n3)
    code_rep(L, "BOUNDTIER10K", "CYCLEBOUND10K", n1)
    code_rep(L, "boundtier", "cyclebound", n2)
    code_rep(L, "small_bt", "small_cb", n3)
    save("chain_cyclebound10k.sh", "\n".join(L))


if __name__ == "__main__":
    w = hashlib.sha256(open(os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfh.py"), "rb").read()).hexdigest()[:16]
    assert w == WRAPPER_SHA, "wrapper sha256 %s is not that of card ADD67" % w
    queue()
    chain()
