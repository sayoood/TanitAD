"""Builds decode_check_arm.py from the code of record decode_check.py (sha256 027da9ee...) by replacing ONLY its R35 path lines with environment-driven paths. Every replaced line must exist exactly once
(fail loud otherwise); the rest of the file is byte-identical, so the diagnostic is the same code. Run once; the result is hash-recorded in the arm JSON."""
import hashlib, sys
SRC = "F:/TanitAD-scratch/hwm_plan/decode_check/decode_check.py"
DST = "F:/TanitAD-scratch/actsens/read/decode_check_arm.py"
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("027da9ee"), hashlib.sha256(raw).hexdigest()
t = raw.decode("utf-8")
REP = [
    ('S = sys.argv[1]; NB = int(sys.argv[2]); T = f"R35_{S}"', 'S = sys.argv[1]; NB = int(sys.argv[2]); T = os.environ["DC_TAG"]'),
    ('B = "F:/TanitAD-scratch/hwm_plan"; OUT = f"{B}/decode_check"', 'B = "F:/TanitAD-scratch/hwm_plan"; OUT = os.environ["DC_OUT_DIR"]'),
    ('O = f"{B}/phase1_R35/{S}"', 'O = os.environ["DC_WIN_DIR"]'),
    ('m, cfg, ck = D.load_model(f"C:/Users/Admin/t2v8f_pulls/ladder_{T}/ckpt_last.pt")', 'm, cfg, ck = D.load_model(os.environ["DC_CKPT"])'),
    ('Z, A = D.trainset_from_latents_pt(f"{B}/latents_train_{T}.pt", T)', 'Z, A = D.trainset_from_latents_pt(os.environ["DC_LATENTS"], T)'),
    ('Lt = P.load_latents(f"{B}/latents_train_{T}.pt", T)', 'Lt = P.load_latents(os.environ["DC_LATENTS"], T)'),
    ('banked = np.load(f"{O}/plans_hwm_oracle_s0.npz")', 'banked = np.load(os.environ["DC_PLANS"])'),
    ('json.dump(R, open(f"{OUT}/result_{S}.json", "w"), indent=1)', 'json.dump(R, open(f"{OUT}/result_{S}.json", "w"), indent=1)'),
]
for a, b in REP:
    assert t.count(a) == 1, ("not exactly once", a)
    t = t.replace(a, b)
open(DST, "w", encoding="utf-8", newline="\n").write(t)
print("wrote", DST, hashlib.sha256(t.encode()).hexdigest()[:8])
