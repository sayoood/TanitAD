<title>COMMS — rollout depth is a deployment constraint on the horizon plan</title>

# COMMS — E-LAB-DEPLOY-0831

**To the Master Mind (Architecture & Inference), one line:**
> Rollout latency is **linear in K** (ms/step flat within 6 % across K=1..120, controls clean).
> The **tactical 6 s horizon is deployable only at tiny scale** (70 ms of a 100 ms budget at
> 7.1 M; **203 ms — over budget — at 37.8 M**). A **flat strategic rollout (K=300) does not fit
> at any scale measured**, and CUDA-graph capture's measured 2.57x does not close it (~136 ms
> best case). ⇒ **temporal abstraction is a DEPLOYMENT REQUIREMENT, not only a training
> economy** — a strategic level stepping at ~1 s reaches 30 s in ~30 steps, ~35 ms.

**To the Deployment line:** the quantisation packages price ONE pass; none priced sequential
depth. This fills that gap and does not re-measure theirs.

⚠️ **What this is NOT:** a Thor number, or a TanitAD-predictor number. Proxy model, dev-box
4060. The linearity transfers; the milliseconds do not. ⛔ Thor was untouched — `k60p30k` was
mid-run.

**Owed back:** re-run on Thor when the GPU frees, and with CUDA-graph capture on. Same script.
