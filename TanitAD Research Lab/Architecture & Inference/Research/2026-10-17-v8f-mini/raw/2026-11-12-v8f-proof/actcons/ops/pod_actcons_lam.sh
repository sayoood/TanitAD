#!/bin/bash
# pod_actcons_lam.sh : card PREREG_V8F_PROOF_ADD37, note N2. Inserts the dose arm LAm (weight 0.0001) after LA_s1 into the arms file of
# chain actcons_a, only while that chain does not exist. Keeps the first arms file beside it. Starts nothing and uses no GPU.
set -u
W=/workspace/v8proof
cd $W/chain2 || { echo ZZ-NO-DIR; exit 1; }
[ -e $W/chains/actcons_a ] && { echo ZZ-CHAIN-EXISTS; exit 3; }
[ -e arms_actcons_a.v1.txt ] && { echo ZZ-V1-EXISTS; exit 4; }
OLD=$(sha256sum arms_actcons_a.txt | cut -c1-64)
[ "$OLD" = 6128edb45704d6bfd9ad1745141632e692bc0f0ca75858d748af454e3f403030 ] || { echo "ZZ-SHA $OLD"; exit 5; }
python3 - <<'EOF' || { echo ZZ-BUILD-FAILED; exit 6; }
import io, sys
sys.path.insert(0, ".")
import gate_extra
assert gate_extra.norm(float("0.0001")) == "0.0001", gate_extra.norm(float("0.0001"))
assert gate_extra.norm(float("1.0")) == "1" and gate_extra.norm(float("0.1")) == "0.1"
src = io.open("arms_actcons_a.txt", encoding="utf-8").read()
tail_nl = src.endswith("\n")
lines = src.split("\n")
if tail_nl:
    lines = lines[:-1]
la = {ln.split("|")[1]: ln for ln in lines if ln.startswith("ARM|LA_s")}
assert sorted(la) == ["LA_s0", "LA_s1"], sorted(la)
out, done = [], False
for ln in lines:
    if ln.startswith("ARM|LA01_s0|"):
        out.append("# note PREREG_V8F_PROOF_ADD37_N2: dose arm LAm_s0, LAm_s1 at weight 0.0001 (the argv of LA apart from the weight).")
        for s in ("LA_s0", "LA_s1"):
            b = la[s]
            for a, r in (("ARM|LA_s", "ARM|LAm_s"), ("--w-act-cons 1.0|", "--w-act-cons 0.0001|"), ("levers.w_act_cons=1 ", "levers.w_act_cons=0.0001 ")):
                assert b.count(a) == 1, (s, a, b.count(a))
                b = b.replace(a, r)
            out.append(b)
        done = True
    out.append(ln)
assert done
arms = [ln.split("|")[1] for ln in out if ln.startswith("ARM|")]
assert arms == ["LA_s0", "LA_s1", "LAm_s0", "LAm_s1", "LA01_s0", "LA01_s1"], arms
assert [ln for ln in out if ln.startswith("PIN")] == [ln for ln in lines if ln.startswith("PIN")]
io.open("arms_actcons_a.new.txt", "w", encoding="utf-8", newline="\n").write("\n".join(out) + ("\n" if tail_nl else ""))
print("built", arms)
EOF
# the training argv of LAm must equal that of LA apart from the weight (dry run of the chain script: prints, starts nothing)
dr() { C2_CARD=PREREG_V8F_PROOF_ADD37 bash chain_arms_ft6.sh dryx arms_actcons_a.new.txt --dry-run 2>&1 | grep "^TRAIN $1" | sed -e "s|/runs/[a-z_0-9]*/$1|/runs/X/A|g" -e "s|^TRAIN $1|TRAIN A|"; }
for s in 0 1; do
  A=$(dr LA_s$s); B=$(dr LAm_s$s)
  [ -n "$A" ] && [ -n "$B" ] || { echo "ZZ-DRY-EMPTY s$s"; exit 7; }
  [ "$(echo "$A" | sed 's|--w-act-cons 1.0|--w-act-cons 0.0001|')" = "$B" ] || { echo "ZZ-ARGV-DIFFERS s$s"; echo "$A" | cut -c1-600; echo "$B" | cut -c1-600; exit 8; }
  [ "$A" != "$B" ] || { echo "ZZ-ARGV-SAME s$s"; exit 9; }
  echo "argv LAm_s$s = argv LA_s$s apart from the weight: $(echo "$B" | grep -o -- '--w-act-cons [0-9.e-]*')"
done
[ -e $W/chains/actcons_a ] && { echo ZZ-CHAIN-EXISTS-LATE; exit 3; }
mv -n arms_actcons_a.txt arms_actcons_a.v1.txt && [ ! -e arms_actcons_a.txt ] && mv -n arms_actcons_a.new.txt arms_actcons_a.txt || { echo ZZ-MOVE-FAILED; exit 10; }
echo "first file: $(sha256sum arms_actcons_a.v1.txt)"
echo "new file:   $(sha256sum arms_actcons_a.txt)"
grep "^ARM" arms_actcons_a.txt | cut -d'|' -f2,4,5 | tr '\n' ' '; echo
ls -d $W/chains/actcons_a 2>/dev/null || echo "chain actcons_a absent at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -d $W/chains/dryx 2>/dev/null
echo ZZ-LAM-INSERTED
