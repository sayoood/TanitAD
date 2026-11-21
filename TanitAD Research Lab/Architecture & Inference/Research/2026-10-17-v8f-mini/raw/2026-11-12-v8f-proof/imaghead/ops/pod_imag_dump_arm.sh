# pod_imag_dump_arm.sh : runs ON THE POD (bash -s -- <chain> <tree> <ARM>...). Cards PREREG_V8F_PROOF_ADD36 / ADD37: for each arm of a chain, on the pod CPU
# (no GPU job is added): eval state file (extract_states.py), head outputs on it (pod_lead_apply.py, the banked reference of control c2 (b)), the imagined-lead dump
# (imag_dump.py), then c2 (b). Idempotent: a step whose record exists is not run again.
set -u
Q=${1:?chain}; TREE=${2:?tree}; shift 2
W=/workspace/v8proof; P=$W/probe; I=$P/imag_in; O=$W/runs/imag_z/dump; PY=/workspace/venv/bin/python; mkdir -p $O $P/states
a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
[ $((m - a)) -gt 24000000000 ] || { echo "DUMP-NOT-RUN: less than 24 GB of the memory limit free"; exit 1; }
echo "DUMP-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ) memory free $(( (m - a) / 1000000000 )) GB; GPU jobs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
for A in "$@"; do
  CK=$W/runs/$Q/$A/ckpt_last.pt
  [ -f $CK ] || { echo "$A: no checkpoint at runs/$Q/$A"; continue; }
  cd $P || exit 1
  [ -f states/${A}_eval.npz.json ] || CUDA_VISIBLE_DEVICES= nice -n 19 $PY extract_states.py --tree $W/$TREE --ckpt $CK --cache $W/data/cache_pod --split eval --lo 0 --hi 141 --out states/${A}_eval.npz > states/${A}_eval.log 2>&1
  [ -f states/${A}_eval.npz.json ] || { echo "$A: extraction failed: $(tail -n 2 states/${A}_eval.log | cut -c1-300)"; continue; }
  ( cd states && { [ -f ${A}_eval_lead.npz.json ] || CUDA_VISIBLE_DEVICES=-1 nice -n 19 $PY $P/pod_lead_apply.py --tree $W/$TREE --ckpt $CK --states ${A}_eval.npz --out ${A}_eval_lead.npz > ${A}_eval_lead.log 2>&1; } )
  [ -f states/${A}_eval_lead.npz.json ] || { echo "$A: head outputs failed: $(tail -n 2 states/${A}_eval_lead.log | cut -c1-300)"; continue; }
  [ -f $O/DUMP_$A.npz.json ] || CUDA_VISIBLE_DEVICES= nice -n 5 $PY $P/imag_dump.py --ckpt $CK --tree $W/$TREE --windows $I/LATENTS_CTX_G_s0_R1.npz --pack $I/eval_pack.npz \
    --states $P/states/${A}_eval.npz --out $O/DUMP_$A.npz --device cpu --threads 4 > $O/dump_$A.log 2>&1
  echo "$A dump: $(tail -n 1 $O/dump_$A.log | cut -c1-330)"
  [ -f $O/DUMP_$A.npz.json ] || continue
  A=$A $PY - <<'PYEOF'
import json, os, numpy as np, hashlib
A = os.environ["A"]; P = "/workspace/v8proof/probe/states/"; O = "/workspace/v8proof/runs/imag_z/dump/"
D = np.load(O + "DUMP_%s.npz" % A, allow_pickle=True); meta = json.load(open(O + "DUMP_%s.npz.json" % A))
S = np.load(P + "%s_eval.npz" % A, allow_pickle=True); B = np.load(P + "%s_eval_lead.npz" % A, allow_pickle=True)
bj = json.load(open(P + "%s_eval_lead.npz.json" % A)); sj = json.load(open(P + "%s_eval.npz.json" % A))
names = np.asarray(S["clips"]).astype(str)
key = {(names[int(c)], int(t)): i for i, (c, t) in enumerate(zip(S["clip_idx"], S["t"]))}
rows = np.array([key[(str(e), int(g))] for e, g in zip(D["eid"], D["g0"])])
pres = np.asarray(D["present"], np.float64); ok = np.asarray(B["has_history"])[rows]
d = (np.abs(pres[ok, 0] - np.asarray(B["presence_logit"], np.float64)[rows][ok]).max(), np.abs(pres[ok, 1] - np.asarray(B["gap_m"], np.float64)[rows][ok]).max() / 50.0,
     np.abs(pres[ok, 2] - np.asarray(B["closing_ms"], np.float64)[rows][ok]).max() / 10.0)
c = meta["controls_at_dump"]
print("%s n %d with history %d | c2b trainer units: logit %.3e gap %.3e closing %.3e -> %s | same ckpt %s same states %s" % (
    A, len(rows), int(ok.sum()), d[0], d[1], d[2], "PASS" if max(d) <= 1e-5 else "FAIL", meta["ckpt_sha256"] == bj["ckpt_sha256"], meta["inputs"]["states_sha256"] == sj["out_sha256"] == bj["states_sha256"]))
print("   c1 %r | c2a %s roll %.3e head %.3e | c4 zero model %r real model %.4f | facts ok %s | cfg %s" % (
    c["c1_offset0_max_abs_response"], c["c2"]["pass"], c["c2"]["rollout_vs_roll_pairs_max_abs"], c["c2"]["head_vs_sequence_max_abs"],
    c["c4"]["max_abs_response_zero_model"], c["c4"]["max_abs_response_real_model_nonzero_offsets"], {k: v["ok"] for k, v in meta["facts"].items()}, meta["cfg_wm"]))
print("   SHA %s %s %s" % (A, hashlib.sha256(open(O + "DUMP_%s.npz" % A, "rb").read()).hexdigest(), hashlib.sha256(open(O + "DUMP_%s.npz.json" % A, "rb").read()).hexdigest()))
PYEOF
done
