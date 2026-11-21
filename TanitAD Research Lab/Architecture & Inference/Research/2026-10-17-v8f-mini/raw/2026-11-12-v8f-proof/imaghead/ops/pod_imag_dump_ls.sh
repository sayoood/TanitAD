# pod_imag_dump_ls.sh : runs ON THE POD (bash -s). Card PREREG_V8F_PROOF_ADD36 with note N1: the imagined-lead dump of LS_s0 and LS_s1 on the pod CPU
# (three GPU jobs train; no GPU job is added), then control c2 (b): the head on the true states at frame t against the banked head outputs of the LEADSUP read.
set -u
W=/workspace/v8proof; P=$W/probe; I=$P/imag_in; O=$W/runs/imag_z/dump; PY=/workspace/venv/bin/python; mkdir -p $O
a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
[ $((m - a)) -gt 20000000000 ] || { echo "DUMP-NOT-RUN: less than 20 GB of the memory limit free"; exit 1; }
echo "DUMP-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ) memory free $(( (m - a) / 1000000000 )) GB; GPU jobs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
for k in 0 1; do
  CUDA_VISIBLE_DEVICES= nice -n 5 $PY $P/imag_dump.py --ckpt $W/runs/leadsup_a/LS_s$k/ckpt_last.pt --tree $W/tree_leadsup --windows $I/LATENTS_CTX_G_s0_R1.npz --pack $I/eval_pack.npz \
    --states $P/states/LS_s${k}_eval.npz --out $O/DUMP_LS_s$k.npz --device cpu --threads 4 > $O/dump_LS_s$k.log 2>&1
  echo "LS_s$k rc $? : $(tail -n 1 $O/dump_LS_s$k.log | cut -c1-400)"
done
$PY - <<'PYEOF'
import json, numpy as np, hashlib
P = "/workspace/v8proof/probe/states/"; O = "/workspace/v8proof/runs/imag_z/dump/"
for k in (0, 1):
    D = np.load(O + "DUMP_LS_s%d.npz" % k, allow_pickle=True); meta = json.load(open(O + "DUMP_LS_s%d.npz.json" % k))
    S = np.load(P + "LS_s%d_eval.npz" % k, allow_pickle=True); B = np.load(P + "LS_s%d_eval_lead.npz" % k, allow_pickle=True)
    bj = json.load(open(P + "LS_s%d_eval_lead.npz.json" % k)); sj = json.load(open(P + "LS_s%d_eval.npz.json" % k))
    names = np.asarray(S["clips"]).astype(str)
    key = {(names[int(c)], int(t)): i for i, (c, t) in enumerate(zip(S["clip_idx"], S["t"]))}
    rows = np.array([key[(str(e), int(g))] for e, g in zip(D["eid"], D["g0"])])
    pres = np.asarray(D["present"], np.float64)                      # [n, 3] logit, gap m, closing m/s
    ok = np.asarray(B["has_history"])[rows]
    d_logit = np.abs(pres[ok, 0] - np.asarray(B["presence_logit"], np.float64)[rows][ok]).max()
    d_gap = np.abs(pres[ok, 1] - np.asarray(B["gap_m"], np.float64)[rows][ok]).max() / 50.0
    d_clos = np.abs(pres[ok, 2] - np.asarray(B["closing_ms"], np.float64)[rows][ok]).max() / 10.0
    c = meta["controls_at_dump"]
    print("LS_s%d n %d with history %d | c2b trainer units: logit %.3e gap %.3e closing %.3e -> %s | same ckpt %s same states %s" % (
        k, len(rows), int(ok.sum()), d_logit, d_gap, d_clos, "PASS" if max(d_logit, d_gap, d_clos) <= 1e-5 else "FAIL",
        meta["ckpt_sha256"] == bj["ckpt_sha256"], meta["inputs"]["states_sha256"] == sj["out_sha256"] == bj["states_sha256"]))
    print("   c1 %r | c2a %s roll %.3e head %.3e | c4 zero model %r real model %.4f | windows %d | device %s | %s s" % (
        c["c1_offset0_max_abs_response"], c["c2"]["pass"], c["c2"]["rollout_vs_roll_pairs_max_abs"], c["c2"]["head_vs_sequence_max_abs"],
        c["c4"]["max_abs_response_zero_model"], c["c4"]["max_abs_response_real_model_nonzero_offsets"], meta["n_windows_dumped"], meta["device"], meta["seconds"]))
    print("   sha256 dump %s json %s" % (hashlib.sha256(open(O + "DUMP_LS_s%d.npz" % k, "rb").read()).hexdigest(), hashlib.sha256(open(O + "DUMP_LS_s%d.npz.json" % k, "rb").read()).hexdigest()))
PYEOF
