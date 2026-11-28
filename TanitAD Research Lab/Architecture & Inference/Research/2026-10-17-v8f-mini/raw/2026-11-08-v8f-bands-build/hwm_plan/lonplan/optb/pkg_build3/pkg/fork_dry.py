"""Stage 2 of the test fork: adapt tests/dry_helpers.py, tests/plant_trainer.py, tests/test_optb_dryrun.sh (names already forked by fork_pkg.py) to Option B. Run once from optb/pkg."""
import re

BS = chr(92)


def load(p):
    return open(p, encoding="utf-8", newline="").read()


def save(p, s):
    open(p, "w", encoding="utf-8", newline="").write(s)


def sub(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:90])
    return s.replace(old, new)


# ------------------------------------------------------------------ plant_trainer
p = "tests/plant_trainer.py"
s = load(p)
s = sub(s, 'split("M2b_s")[1][0])', 'split("ladder_R35_s")[1][0])', 2)
s = sub(s, "    n = 2011970\n", "    n = 2011970\n    nt = n - 4000      # trainable = WM minus wm.proj (exact count UNVERIFIED until the pilot selfcheck)\n")
s = sub(s, '"WM": n}, "n_params_trainable": n,', '"WM": nt}, "n_params_trainable": nt,')
s = sub(s, '"n_wm": n, "n_trainable": n},', '"n_wm": n, "n_trainable": nt},')
s = sub(s, '"mode": "wm_only", "source_arm": "M2b", "source_seed": init, "wm_reinit_seed": seed}', '"mode": "wm_only_init_wm", "init_wm": True}')
s = sub(s, '''    if rsp is not None:
        terms["wm/rsp/L1"] = 0.4''', '''    lv["wm_rsp_readback"] = {"declared": 0.0, "built_config": 0.0, "hmax": 91, "declared_readout": "idm", "built_readout": "idm", "built": {"w_in_loss_rsp": 0.0},
                             "freeze_proj_declared": "on", "wm_proj_requires_grad": [False], "init_from_ckpt_wm_declared": "on"}
    if rsp is not None:
        terms["wm/rsp/L1"] = 0.4''')
s = sub(s, 'lv["wm_rsp_readback"] = {"declared": rsp, "built_config": rsp, "hmax": 91, "built": {"calls": 1, "contrasts": 4, "anchors": 9, "k": 10, "ctx_passed": 1, "w_in_loss_rsp": rsp, "rsp_term": 0.4}}',
        'lv["wm_rsp_readback"].update({"declared": rsp, "built_config": rsp, "declared_readout": "fixed_k", "built_readout": "fixed_k", "built": {"calls": 1, "contrasts": 2, "anchors": 9, "k": 10, "ctx_passed": 1, "w_in_loss_rsp": rsp, "rsp_term": 0.4, "readout": "fixed_k", "s_cal": (0.20076, 0.20323)[init]}})')
s = sub(s, '''    elif mode == "bad_steps":''', '''    elif mode == "bad_null_readout" and rsp is None:
        lv["wm_rsp_readback"]["built_readout"] = "fixed_k"
    elif mode == "bad_steps":''')
s = sub(s, '"wm.stack.pred.w", "wm.proj.w"]', '"wm.stack.pred.w", "wm.proj.0.w", "wm.proj.0.b", "wm.proj.1.w", "wm.proj.1.b"]')
s = sub(s, '''        if k.startswith("wm."):
            st[k] = st[k] + 1.0
''', '''        if k.startswith("wm.") and (not k.startswith("wm.proj.") or tm == "move_proj"):
            st[k] = st[k] + 1.0
''')
s = sub(s, "    n_opt = 12\n", "    n_opt = 141          # the 145 wm.* tensors minus the 4 frozen wm.proj tensors (AMEND_OPTB C5)\n")
save(p, s)

# ------------------------------------------------------------------ dry_helpers
p = "tests/dry_helpers.py"
s = load(p)
s = sub(s, 'CK.MODEL_REL: "# wm_rsp_w  def built_rsp  RSP_DELTA\\n"', 'CK.MODEL_REL: "# wm_rsp_w  def built_rsp  RSP_DELTA  def load_rsp_probe\\n"')
s = sub(s, 'CK.HJEPA_REL: "# def response_loss  def built_rsp  RSP_K, RSP_STRIDE, RSP_CTX, RSP_S_MIN  rsp_w: float = 0.0\\n"',
        'CK.HJEPA_REL: "# def response_loss  def _response_loss_fixed_k  def built_rsp  RSP_K, RSP_STRIDE, RSP_CTX, RSP_S_MIN  rsp_w: float = 0.0  rsp_readout\\n"')
s = sub(s, '''    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
''', '''    for s in (0, 1):     # the R35 start checkpoints live on the DEV box (box/pulls_src/ladder_R35_s{k}/ckpt_last.pt); the inputs stage puts them on the box
        p = os.path.join(box, "pulls_src", f"ladder_R35_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
''')
save(p, s)

# the stand-in trainer must carry the new needles too
src = load("tests/dry_helpers.py")
assert "def bad_wm_freeze" not in src
s = load("tests/plant_trainer.py")
s = sub(s, "Needle strings the ship step requires", "Needle strings the ship step requires (NEEDLES also: --wm-rsp-readout --wm-freeze-proj def bad_wm_freeze)")
save("tests/plant_trainer.py", s)

# ------------------------------------------------------------------ hf stub
save("tests/hf_stub.py", '''"""Dry-run stand-in for optb_hfbank.py: copies the bank dir to $OPTB_HF_STUB_DIR/<repo>/<tag> and compares sha256 file by file. No network."""
import argparse, hashlib, os, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument("--tag"); ap.add_argument("--bank-dir"); ap.add_argument("--repo"); ap.add_argument("--keys", default="")
a = ap.parse_args()
dst = os.path.join(os.environ["OPTB_HF_STUB_DIR"], a.repo.replace("/", "_"), a.tag)
shutil.copytree(a.bank_dir, dst, dirs_exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [f for f in os.listdir(a.bank_dir) if sha(os.path.join(a.bank_dir, f)) != sha(os.path.join(dst, f))]
if bad:
    print("VERIFY-FAIL", bad, file=sys.stderr); sys.exit(5)
print("ZZ OPTB-HF-OK " + a.tag)
''')
s = load("stages/optb_hfbank.sh")
s = sub(s, '"$PY" -I "$PKG/optb_hfbank.py"', '"$PY" -I "${OPTB_HFBANK_PY:-$PKG/optb_hfbank.py}"')
save("stages/optb_hfbank.sh", s)

# ------------------------------------------------------------------ the dry-run script
p = "tests/test_optb_dryrun.sh"
s = load(p).replace("\r\n", "\n")
s = re.sub(r"(?<![A-Za-z])(RSP|NULL)_s([0-5])", r"FT\1_s\2", s)
s = sub(s, "for t in FTRSP_s0 FTRSP_s2 FTNULL_s4 FTRSP_s1 FTRSP_s3 FTNULL_s5; do", "for t in FTRSP_s0 FTNULL_s0 FTRSP_s2 FTNULL_s2 FTNULL_s4 FTRSP_s1 FTNULL_s1 FTRSP_s3 FTNULL_s3 FTNULL_s5; do")
s = sub(s, "== 1 normal flow: wait check, ship, pilot, 72 h permit, six runs, banks, cleanups", "== 1 normal flow: wait check, ship, inputs, pilot, 72 h permit, ten runs, banks, HF banks, cleanups")
s = sub(s, "projection covers 6 runs x 19,164 steps", "projection covers 10 runs x 4,000 steps")
s = sub(s, "d[\\\"steps_total\\\"]==114984 and d[\\\"n_runs_planned\\\"]==6", "d[\\\"steps_total\\\"]==40000 and d[\\\"n_runs_planned\\\"]==10")
s = sub(s, "the six tokens bind six distinct argvs", "the ten tokens bind ten distinct argvs")
s = sub(s, '| wc -l | tr -d " ")" -eq 6 ]\'\nok "run order', '| wc -l | tr -d " ")" -eq 10 ]\'\nok "run order')
s = sub(s, 'run order in chain.log: FTRSP_s0, FTRSP_s2, FTNULL_s4, FTRSP_s1, FTRSP_s3, FTNULL_s5', 'run order in chain.log: chassis 0 (FTRSP_s0, FTNULL_s0, FTRSP_s2, FTNULL_s2, FTNULL_s4) then chassis 1')
s = sub(s, 'done FTRSP_s0 done FTRSP_s2 done FTNULL_s4 done FTRSP_s1 done FTRSP_s3 done FTNULL_s5 "', 'done FTRSP_s0 done FTNULL_s0 done FTRSP_s2 done FTNULL_s2 done FTNULL_s4 done FTRSP_s1 done FTNULL_s1 done FTRSP_s3 done FTNULL_s3 done FTNULL_s5 "')
s = sub(s, "only the six runs exist on the box", "only the ten runs exist on the box")
s = sub(s, '| wc -l | tr -d " ")" -eq 6 ] && [ ! -e "$B/D/tanitad/optb/runs/OB-BAND_s0" ]', '| wc -l | tr -d " ")" -eq 10 ] && [ ! -e "$B/D/tanitad/optb/runs/OB-BAND_s0" ]')
s = sub(s, "(>= 8 reads: ship, pilot, six runs)", "(>= 13 reads: ship, inputs, pilot, ten runs)")
s = sub(s, '-ge 8 ]', '-ge 13 ]')
s = sub(s, "optb/pilot_out/FTRSP_s0", "optb/pilot_out/FTRSP_s0", 0) if False else s
s = sub(s, "pilot used the exact FTRSP_s0 argv", "pilot used the exact FTRSP_s0 argv")
s = sub(s, "lets the six runs go", "lets the ten runs go")
s = sub(s, "a full-length steps_total that is not 19,164", "a full-length steps_total that is not 4,000")
# newbox: the dev-side R35 checkpoints, the fake HF
s = sub(s, '  unset OPTB_RESUME OPTB_AMENDMENT\n}', '''  mkdir -p "$OPTB_PULLS"; cp -r "$B/pulls_src/." "$OPTB_PULLS/"; export OPTB_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")"
  export OPTB_HF_REPO="dry/private" OPTB_HFBANK_PY="$(cygpath -m "$TESTS/hf_stub.py")" OPTB_HF_STUB_DIR="$T/hf_$1"
  unset OPTB_RESUME OPTB_AMENDMENT
}''')
# normal flow extras
s = sub(s, 'ok "busy marker handed on"', '''ok "the inputs are on the box with the pinned sha256 (two checkpoints, two probes)" '[ "$(sha256sum "$B/D/tanitad/optb/inputs/ladder_R35_s0.pt" | cut -d" " -f1)" = "$(sha256sum "$OPTB_PULLS/ladder_R35_s0/ckpt_last.pt" | cut -d" " -f1)" ] && [ "$(sha256sum "$B/D/tanitad/optb/inputs/readout_K10_R35_s1.pt" | cut -d" " -f1)" = "$(sha256sum "$PKG/frozen/inputs/readout_K10_R35_s1.pt" | cut -d" " -f1)" ] && [ "$(ls "$B/D/tanitad/optb/inputs" | wc -l | tr -d " ")" -eq 4 ]'
ok "every run was banked to the (stub) private HF repo and verified" '[ "$(ls "$OPTB_HF_STUB_DIR/dry_private" | wc -l | tr -d " ")" -eq 10 ] && [ -e "$OPTB_STATE/done/hf_FTNULL_s5.done" ]'
ok "busy marker handed on"''')
# section 6 / 7 expectation edits for the new order
s = sub(s, 'ok "the independent arms after it were banked, the refused one was not trained" \'[ -e "$OPTB_PULLS/optb_FTRSP_s1/BANK.done" ] && [ -e "$OPTB_PULLS/optb_FTNULL_s5/BANK.done" ]',
        'ok "the independent arms after it were banked, the refused one was not trained" \'[ -e "$OPTB_PULLS/optb_FTRSP_s1/BANK.done" ] && [ -e "$OPTB_PULLS/optb_FTNULL_s3/BANK.done" ] && [ -e "$OPTB_PULLS/optb_FTNULL_s5/BANK.done" ]')
s = sub(s, "the four earlier arms are banked", "the earlier arms are banked")
s = sub(s, '[ ! -e "$B/D/tanitad/optb/runs/FTNULL_s5/argv.txt" ] && [ -e "$OPTB_PULLS/optb_FTRSP_s0/BANK.done" ] && [ -e "$OPTB_PULLS/optb_FTRSP_s1/BANK.done" ]', '[ ! -e "$B/D/tanitad/optb/runs/FTNULL_s3/argv.txt" ] && [ -e "$OPTB_PULLS/optb_FTRSP_s0/BANK.done" ] && [ -e "$OPTB_PULLS/optb_FTRSP_s1/BANK.done" ]')
s = sub(s, '[ ! -e "$B/D/tanitad/optb/runs/FTRSP_s2/RUN.done" ] && [ ! -e "$OPTB_STATE/done/FTRSP_s2.done" ]', '[ ! -e "$B/D/tanitad/optb/runs/FTNULL_s0/RUN.done" ] && [ ! -e "$OPTB_STATE/done/FTNULL_s0.done" ]')
# new sections: chassis-0 stop, wm.proj moved, HF repo unset
s = sub(s, 'echo "== 8 commit, tag and package refusals"', '''echo "== 7b chassis 0 failing stops before chassis 1; wm.proj that moved; HF repo unset"
newbox ch0 0; control '{"sps": 1.0, "modes": {"FTNULL_s4": "bad_null_readout"}}'
bash "$PKG/optb_chain.sh" > "$T/c0.out" 2>&1; rc=$?
ok "a chassis-0 arm refused at its gate: chain ends rc 4 with the reason chassis0-failed" '[ $rc -eq 4 ] && [ -e "$OPTB_STATE/ARM_REFUSED_FTNULL_s4.txt" ] && stoptxt | grep -q "chassis0-failed"'
ok "chassis 1 never started: no argv, no run folder on the box" '[ ! -e "$B/D/tanitad/optb/runs/FTRSP_s1/argv.txt" ] && [ ! -e "$B/D/tanitad/runs/v8f_optb/FTRSP_s1" ] && [ -e "$OPTB_PULLS/optb_FTRSP_s2/BANK.done" ]'
newbox pproj 0; control '{"sps": 1.0, "train_modes": {"FTRSP_s0": "move_proj"}}'
bash "$PKG/optb_chain.sh" > "$T/pj.out" 2>&1; rc=$?
ok "a wm.proj that moved is a post-check failure (rc 5), named" '[ $rc -eq 5 ] && stoptxt | grep -q "wm.proj moved although frozen"'
newbox nohf 0; control '{"sps": 1.0}'; unset OPTB_HF_REPO
bash "$PKG/optb_chain.sh" > "$T/nh.out" 2>&1; rc=$?
ok "HF repo unset: every run still banks on the dev box, the chain ends in STOP hf-bank-pending (rc 5), never DONE" '[ $rc -eq 5 ] && stoptxt | grep -q "hf-bank-pending" && [ ! -e "$OPTB_STATE/DONE.txt" ] && [ -e "$OPTB_PULLS/optb_FTNULL_s5/BANK.done" ] && [ -e "$OPTB_STATE/HF_BANK_PENDING_FTRSP_s0.txt" ]'

echo "== 8 commit, tag and package refusals"''')
save(p, s)
print("dry fork ok")
