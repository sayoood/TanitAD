"""Micro-benchmark for the AMEND_7 cost estimate: the RY head's shape (256-256 GELU) on 256 inputs, AdamW, batch 256, 2 threads, synthetic data. Times steps only."""
import time
import numpy as np
import torch

torch.set_num_threads(2)
torch.manual_seed(0)
n = 47000
X = torch.randn(n, 256)
y = torch.randn(n, 1)
net = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
idx = np.arange(n)
np.random.RandomState(0).shuffle(idx)
steps = 0
t0 = time.time()
for i in range(0, n, 256):
    b = idx[i:i + 256]
    loss = ((net(X[b]) - y[b]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    steps += 1
dt = time.time() - t0
print(f"steps {steps} wall {dt:.2f} s  per step {1000 * dt / steps:.2f} ms  per epoch {dt:.2f} s")
t0 = time.time()
with torch.no_grad():
    p = net(X)
print(f"predict {n} rows {time.time() - t0:.3f} s")
