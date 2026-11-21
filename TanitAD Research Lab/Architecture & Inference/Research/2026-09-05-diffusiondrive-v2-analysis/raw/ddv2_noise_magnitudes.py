import math, json
# diffusers DDIMScheduler(beta_schedule="scaled_linear", beta_start=1e-4, beta_end=0.02, num_train_timesteps=1000)
T = 1000
b0, b1 = math.sqrt(1e-4), math.sqrt(0.02)
betas = [(b0 + (b1 - b0) * i / (T - 1)) ** 2 for i in range(T)]
ac = []
p = 1.0
for b in betas:
    p *= (1 - b); ac.append(p)
final_ac = 1.0   # set_alpha_to_one=True (diffusers default)
def ddim_var(t, prev):
    a_t = ac[t]; a_p = ac[prev] if prev >= 0 else final_ac
    return (1 - a_p) / (1 - a_t) * (1 - a_t / a_p)
rows = []
print("t prev  abar_t    sqrt(1-abar) sigma_t(eta=1) sigma_mul(clip.04) sigma_logp(clip.10)")
for t in [18, 16, 14, 12, 10, 8, 6, 4, 2, 0]:
    prev = t - 1          # prev_timestep = t - num_train_timesteps // num_inference_steps = t - 1 (set_timesteps(1000))
    s = math.sqrt(ddim_var(t, prev))
    r = dict(t=t, prev=prev, alpha_bar_t=ac[t], sqrt_1m_abar=math.sqrt(1 - ac[t]), ddim_sigma_eta1=s,
             sigma_mul_used=max(s, 0.04), sigma_logprob_used=max(s, 0.1))
    rows.append(r)
    print("%2d %3d  %.6f  %.4f       %.5f        %.4f            %.4f" % (t, prev, ac[t], math.sqrt(1 - ac[t]), s, max(s, .04), max(s, .1)))
s8 = math.sqrt(1 - ac[8])
print("start noise at t=8: sqrt(1-abar_8) = %.4f -> V2 norm_odo (x/50, y/20): sigma_x = %.3f m, sigma_y = %.3f m" % (s8, 50 * s8, 20 * s8))
print("(v1 audit used half-ranges x 28.45 / y 23 m -> 0.899 / 0.727 m; V2's norm_odo is /50 and /20, verbatim in both V2 model files)")
for x in [5, 10, 20, 40]: print("multiplicative floor 0.04 at x = %2d m -> sigma_along = %.2f m" % (x, 0.04 * x))
for y in [0.5, 2.0, 5.0]: print("multiplicative floor 0.04 at y = %.1f m -> sigma_lat = %.3f m" % (y, 0.04 * y))
print("log-prob std floor 0.10 in normalised units = %.1f m (x) / %.1f m (y) per coordinate" % (0.1 * 50, 0.1 * 20))
print("selector augmentation std U(0.1,0.2) train / U(0.1,0.3) test: at x = 20 m -> sigma_along 2-4 m / 2-6 m")
json.dump(dict(schedule="diffusers DDIM scaled_linear 1e-4..0.02 T=1000, prev=t-1", rows=rows,
               start_noise_t8=dict(sqrt_1m_abar=s8, sigma_x_m=50 * s8, sigma_y_m=20 * s8)),
          open("ddv2_noise_magnitudes.json", "w"), indent=1)
