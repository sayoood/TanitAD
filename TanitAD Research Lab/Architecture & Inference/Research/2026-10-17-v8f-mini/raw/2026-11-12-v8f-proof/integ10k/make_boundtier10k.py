"""make_boundtier10k.py -- writes the generated run tools of card PREREG_V8F_PROOF_ADD68 (BOUNDTIER10K: the readout, the lead rule and the candidate on a kinematic executor
with a heading bound; doses of the bound in the convention; no training) by asserted text edits (each old text the counted number of times):
  pod_headbound10k_queue.sh -> pod_boundtier10k_queue.sh
  chain_headbound10k.sh     -> chain_boundtier10k.sh      (the run folders are not pulled to the dev box; the small folder only)
The originals are not changed. The rollout wrapper is that of card ADD67, unchanged (v8f_t1_eval_reader_rule_rzgfh.py, sha256 373ffe72f2655c35, asserted here).
Pure text generator (standard library only). Usage: python make_boundtier10k.py (in this folder)."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_boundtier10k.py"
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
    """replace old by new in the lines that are not comment lines; the number of occurrences is asserted"""
    c = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            continue
        k = ln.count(old)
        if k:
            lines[i] = ln.replace(old, new)
            c += k
    assert c == n, "anchor %r occurs %d times in code lines, expected %d" % (old[:60], c, n)


def index_of(lines, start):
    hit = [i for i, ln in enumerate(lines) if ln.startswith(start)]
    assert len(hit) == 1, "line start %r found %d times" % (start[:60], len(hit))
    return hit[0]


def queue():
    L = load("pod_headbound10k_queue.sh").split("\n")
    assert L[0].startswith("#!") and L[1].startswith("# pod_headbound10k_queue.sh -- written by make_headbound10k.py")
    hdr = [
        "# pod_boundtier10k_queue.sh -- written by make_boundtier10k.py from pod_headbound10k_queue.sh. Card PREREG_V8F_PROOF_ADD68 (BOUNDTIER10K): the three tool checks of card ADD67",
        "# again (control g4; marker bt6check), then the control I10R0C10HbID (no rule, --readout causal, --kappa-max 1e9, model seed 0, compared file for file with the banked dump of",
        "# I10R0C10_s0: marker bt1cmp), then the control I10RLcHbID (label-fed rule, --readout pinned, --kappa-max 1e9, model seed 0, compared with the banked dump of I10RLc_s0: marker",
        "# bt2cmp), then at two model seeds the eight rollouts of the card table: I10R0C10Hb20 (causal, 0.2), I10R0C10X (causal_g gain 1 on the executed path, no bound), I10R0C10XHb20",
        "# (the same, 0.2), I10RLcHb20 (label-fed, pinned, 0.2), I10RQ50Hb20 (reader-fed at level 0.50, pinned, 0.2), I10R0Hb05, I10R0Hb10, I10R0Hb40 (no rule, pinned, 0.05 / 0.1 / 0.4).",
        "# Wrapper v8f_t1_eval_reader_rule_rzgfh.py of card ADD67, unchanged. No training. The queue waits for the end marker of the queue of card ADD67 (WAIT_DONE / WAIT_STOP, defaults",
        "# HEADBOUND10K_DONE.txt / HEADBOUND10K_STOP.txt in the folder $W). Markers START_BOUNDTIER10K.txt / BOUNDTIER10K_STOP.txt / BOUNDTIER10K_DONE.txt; step markers bt6check, bt1cmp,",
        "# bt2cmp. The header lines below are those of the earlier queues.",
    ]
    L = [L[0]] + hdr + L[1:]
    code_rep(L, "HEADBOUND10K_STOP.txt", "BOUNDTIER10K_STOP.txt", 1)
    code_rep(L, "START_HEADBOUND10K.txt", "START_BOUNDTIER10K.txt", 2)
    code_rep(L, "card ADD67: three tool checks, then two controls and ten rollouts with the heading bound", "card ADD68: three tool checks, then two controls and sixteen rollouts on the bounded executor", 1)
    code_rep(L, "HEADBOUND10K_DONE", "BOUNDTIER10K_DONE", 2)
    code_rep(L, "CAND10K_DONE.txt", "HEADBOUND10K_DONE.txt", 1)
    code_rep(L, "CAND10K_STOP.txt", "HEADBOUND10K_STOP.txt", 1)
    code_rep(L, "the rollouts of card ADD67 start", "the rollouts of card ADD68 start", 1)
    code_rep(L, "for nm in I10R0 I10R0G0X I10RLc I10RQ50; do", "for nm in I10R0 I10R0C10 I10RLc I10RQ50 I10R0Hb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do", 1)
    code_rep(L, "for nm in I10RLcKF5 I10RQ50KF5; do", "for nm in I10R0Hb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do", 1)
    code_rep(L, "tick files of I10RLcKF5 and I10RQ50KF5 (both model seeds)", "tick files of the four reused runs of card ADD67 (both model seeds)", 1)
    code_rep(L, "hashes of the reused runs of card ADD65 taken", "hashes of the reused runs of card ADD67 taken", 1)
    n_small = sum(ln.count("small_hb") for ln in L if not ln.startswith("#"))
    n_test = sum(ln.count("hb_test") for ln in L if not ln.startswith("#"))
    n_chk = sum(ln.count("hb6check") for ln in L if not ln.startswith("#"))
    assert n_small >= 6 and n_test >= 10 and n_chk == 2, (n_small, n_test, n_chk)
    code_rep(L, "small_hb", "small_bt", n_small)
    code_rep(L, "hb_test", "bt_test", n_test)
    code_rep(L, "hb6check", "bt6check", n_chk)
    a = index_of(L, "READOUT=pinned KAPPA=1e9 rollout I10R0HbID R0 0 0")
    b = index_of(L, 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
    old = L[a:b]
    assert len(old) == 19 and old[-1] == "done" and sum(1 for ln in old if " rollout " in ln) == 7, (len(old), old[-1])
    new = [
        'READOUT=causal KAPPA=1e9 rollout I10R0C10HbID R0 0 0       # control g1: the bound with a huge K under the causal readout, no rule, must give the banked dump of I10R0C10 at model seed 0',
        'if [ "$DRY" != 1 ] && ! done_step bt1cmp; then',
        '  mkdir -p "$W/small_bt"',
        '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0C10_s0/dump_I10R0C10_s0_m10_causal_heads" "$W/t1/I10R0C10HbID_s0/dump_I10R0C10HbID_s0_m10_causal_heads" > "$W/small_bt/CMP_R0C10HbID.txt" 2>&1 || stop "G1-RERUN-DIFFERS"',
        '  mark bt1cmp',
        'fi',
        'READOUT=pinned KAPPA=1e9 rollout I10RLcHbID RLc 0 0       # control g2: the bound with a huge K, pinned readout, label-fed rule, must give the banked dump of I10RLc at model seed 0',
        'if [ "$DRY" != 1 ] && ! done_step bt2cmp; then',
        '  mkdir -p "$W/small_bt"',
        '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLc_s0/dump_I10RLc_s0_m10_causal_heads" "$W/t1/I10RLcHbID_s0/dump_I10RLcHbID_s0_m10_causal_heads" > "$W/small_bt/CMP_RLcHbID.txt" 2>&1 || stop "G2-RERUN-DIFFERS"',
        '  mark bt2cmp',
        'fi',
        'for k in "${ARMS[@]}"; do',
        '  READOUT=causal KAPPA=0.2 rollout I10R0C10Hb20 R0 "$k" "$k"',
        '  READOUT=causal_g ACCGAIN=1 PATHEXEC=1 rollout I10R0C10X R0 "$k" "$k"                 # reference without a bound',
        '  READOUT=causal_g ACCGAIN=1 PATHEXEC=1 KAPPA=0.2 rollout I10R0C10XHb20 R0 "$k" "$k"',
        '  READOUT=pinned KAPPA=0.2 rollout I10RLcHb20 RLc "$k" "$k"',
        '  READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50Hb20 RR "$k" "$k"',
        '  READOUT=pinned KAPPA=0.05 rollout I10R0Hb05 R0 "$k" "$k"       # dose, names nothing',
        '  READOUT=pinned KAPPA=0.1 rollout I10R0Hb10 R0 "$k" "$k"        # dose, names nothing',
        '  READOUT=pinned KAPPA=0.4 rollout I10R0Hb40 R0 "$k" "$k"        # dose, names nothing',
        'done',
    ]
    L = L[:a] + new + L[b:]
    text = "\n".join(L)
    for bad in ("HbID_s0/dump_I10R0HbID", "hb1cmp", "hb2cmp"):
        assert all(bad not in ln for ln in L if not ln.startswith("#")), bad
    save("pod_boundtier10k_queue.sh", text)


def chain():
    L = load("chain_headbound10k.sh").split("\n")
    assert L[0].startswith("#!")
    hdr = ["# chain_boundtier10k.sh -- written by make_boundtier10k.py from chain_headbound10k.sh. Card PREREG_V8F_PROOF_ADD68: pushes the queue of the card (the other files are on the pod",
           "# already and are only compared), starts it, waits for its end marker and pulls the small folder. The run folders stay on the pod (the reads run there).",
           "# usage: bash chain_boundtier10k.sh [--no-start]      last line: ZZ CHAIN-BOUNDTIER10K DONE | STOP"]
    L = [L[0]] + hdr + L[1:]
    n1 = sum(ln.count("HEADBOUND10K") for ln in L if not ln.startswith("#"))
    n2 = sum(ln.count("headbound") for ln in L if not ln.startswith("#"))
    n3 = sum(ln.count("small_hb") for ln in L if not ln.startswith("#"))
    assert n1 >= 8 and n2 >= 4 and n3 >= 3, (n1, n2, n3)
    code_rep(L, "HEADBOUND10K", "BOUNDTIER10K", n1)
    code_rep(L, "headbound", "boundtier", n2)
    code_rep(L, "small_hb", "small_bt", n3)
    a = index_of(L, "for m in 0 1; do")
    assert L[a + 1].startswith("  for nm in I10R0HbID I10R0KHbID") and "pull_dir" in L[a + 2] and L[a + 3] == "  done" and L[a + 4] == "done", L[a:a + 5]
    L = L[:a] + ["# the run folders are not pulled (they stay on the pod; the reads run there)"] + L[a + 5:]
    save("chain_boundtier10k.sh", "\n".join(L))


if __name__ == "__main__":
    w = hashlib.sha256(open(os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfh.py"), "rb").read()).hexdigest()[:16]
    assert w == WRAPPER_SHA, "wrapper sha256 %s is not that of card ADD67" % w
    queue()
    chain()
