"""Is `--accum` equivalent to one large batch? Test the TWO loss terms separately, against
LITERAL expectations, using train.py's own arithmetic transcribed from source.

train.py:493   l_score = (per*cand_m).sum() / cand_m.sum().clamp(min=1) * score_w
train.py:516   l_score = score.sum()*0.0                      (uncovered micro-batch)
train.py:525   loss    = (l_traj + l_score) / accum
"""
import torch, sys
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import wta_loss
torch.manual_seed(0)
B, M, H, D = 8, 64, 20, 3
ACC = 4

# ---------- 1. TRAJECTORY term: does accumulation reproduce the big batch exactly? ----------
w = torch.randn(M, H, D, requires_grad=True)
tgt = torch.randn(B, H, D)
def traj_from(w, n):  return w.unsqueeze(0).expand(n, -1, -1, -1)
# big batch
l, _ = wta_loss(traj_from(w, B), tgt); l.backward(); g_big = w.grad.clone(); w.grad = None
# accumulated: B/ACC micro-batches of size ACC... use micro=B//ACC=2 per window -> 4 micros of 2
micro = B // ACC
for i in range(ACC):
    li, _ = wta_loss(traj_from(w, micro), tgt[i*micro:(i+1)*micro]); (li / ACC).backward()
g_acc = w.grad.clone()
print("== 1. TRAJECTORY term ==")
print(f"  batch {B} one step  vs  {ACC} micro-batches of {micro} with /accum")
print(f"  max |g_big - g_acc| = {(g_big-g_acc).abs().max().item():.3e}   rel "
      f"{((g_big-g_acc).abs().max()/g_big.abs().max()).item():.3e}")
print(f"  -> {'EQUIVALENT (as claimed)' if (g_big-g_acc).abs().max().item() < 1e-5 else 'NOT equivalent'}")

# ---------- 2. SCORE term: the per-batch normaliser ----------
# 8 samples, K=4 candidate slots each; coverage is sparse (the MEASURED ~10 % regime)
cand_m = torch.zeros(B, 4); cand_m[0, :3] = 1.0; cand_m[5, 0] = 1.0     # 3 covered + 1 covered
per    = torch.zeros(B, 4); per[0, :3] = torch.tensor([1.0, 2.0, 3.0]); per[5, 0] = 10.0
print("\n== 2. SCORE term: per-sample BCE means, mask, and the two normalisations ==")
print(f"  covered candidates: sample 0 -> 3 (losses 1,2,3), sample 5 -> 1 (loss 10); total mass 4")
big = (per*cand_m).sum() / cand_m.sum().clamp(min=1.0)
print(f"  ONE BATCH of {B}: (sum {float((per*cand_m).sum()):.1f}) / (mask {float(cand_m.sum()):.1f}) = {float(big):.4f}")
tot = 0.0
for i in range(ACC):
    sl = slice(i*micro, (i+1)*micro)
    m_i, p_i = cand_m[sl], per[sl]
    li = (p_i*m_i).sum() / m_i.sum().clamp(min=1.0) if float(m_i.sum()) > 0 else torch.tensor(0.0)
    tot += float(li) / ACC
    print(f"    micro {i} (samples {sl.start}-{sl.stop-1}): mask {float(m_i.sum()):.0f}  "
          f"l_score {float(li):.4f}  contributes {float(li)/ACC:.4f}")
print(f"  ACCUMULATED total = {tot:.4f}   vs one-batch {float(big):.4f}")
print(f"  RATIO accumulated/true = {tot/float(big):.4f}   LITERAL EXPECTATION for equivalence: 1.0000")
print(f"  -> {'EQUIVALENT' if abs(tot/float(big)-1) < 1e-6 else '** NOT EQUIVALENT ** the score loss is re-weighted'}")
# what a big batch would weight each sample at, vs what accumulation weights it at
print("\n  per-sample effective weight (what the gradient actually sees):")
print(f"    one batch : sample 0 -> 3/4 = 0.750 of the loss mass, sample 5 -> 1/4 = 0.250")
print(f"    accumulated: sample 0 -> its micro has mask 3, weight 1/{ACC} = {1/ACC:.3f} of a mean over 3")
print(f"                 sample 5 -> its micro has mask 1, weight 1/{ACC} = {1/ACC:.3f} of a mean over 1")
print(f"    => the ONE covered candidate in micro 2 is weighted {3.0:.0f}x more heavily than each of")
print(f"       the three in micro 0. A true batch of 256 weights all four equally.")
# 3. the uncovered-micro dilution
print("\n== 3. uncovered micro-batches dilute the score loss ==")
cov = sum(1 for i in range(ACC) if float(cand_m[i*micro:(i+1)*micro].sum()) > 0)
print(f"  {cov} of {ACC} micro-batches carry any supervision; the /accum divisor is {ACC} regardless")
print(f"  => the score term's effective weight is scaled by {cov}/{ACC} = {cov/ACC:.2f} against a true batch")
print(f"  at the MEASURED ~10 % frame coverage and batch 4, P(a micro is uncovered) = 0.9^4 = {0.9**4:.3f}")
