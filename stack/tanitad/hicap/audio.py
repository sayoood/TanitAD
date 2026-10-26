"""HiCAP audio: a frozen audio encoder bridged into the frozen backbone's embedding space (spec HICAP_V10_SPEC.md s3.2,
paper sec_elastic.tex 'Audio in the shared embedding space', research note RG_audio_bridge.md).

    mic (mono, or an array) -> AudioFrontend: log-mel (2 s window, 0.5 s hop) + inter-channel level/phase/GCC-PHAT (bearing)
       -> FROZEN audio encoder (protocol below; default recommendation: Qwen3-Omni AuT, contender BEATs -- note RG s4)
       -> AudioBridge (trainable): 2-layer Perceiver, 8 latent queries, -> MLP -> 8 tokens in R^{d_v}
            (a) HiCAP fusion: tokens -> model.GatedAudio with HiCAPConfig.d_audio = d_v   (default; audio-null bit-exact)
            (b) frozen-VLM prefill: the same 8 tokens spliced at placeholder positions (`splice_soft_tokens`; interface only)
    Stage A1: BridgeAlignLoss -- symmetric InfoNCE + cosine between the pooled bridge output and the FROZEN text path's embedding
              of an event caption; captions come from event METADATA only (`caption_for_event`), never from driving labels.
    Stage A2: EventHeads (class, bearing sector, distance bin, approaching) on the bridge tokens, from event metadata.
    Synthetic events: `render_synthetic_event` places a procedural emitter on an agent track (delay, 1/r gain, Doppler through
              retarded time, air absorption, per-microphone ITD + directivity ILD, speed-dependent ego-noise floor).

This is the omni-model pattern (encoder -> small projector -> soft tokens in the LM's input space; Qwen2.5/3-Omni, Gemma 3n/4,
note RG s1) with two documented differences: a Perceiver fixes the token count at 8 per window, and A1 is contrastive to the frozen
text path instead of the LM's next-token loss (the generative arm A1-gen is registered in note RG s1.2).
numpy/torch only (torchaudio is not installed in the test environment; STFT is torch.stft). Nothing here has been trained.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

C_SOUND = 343.0                                             # m/s at ~20 C
L_REF_DB = 120.0                                            # digital convention: RMS 1.0 <-> 120 dB SPL (re 20 uPa)
EVENT_CLASSES = ("none", "siren_wail", "siren_yelp", "horn", "reversing_beeper", "tyre_squeal", "train_horn")
CLASS_PHRASE = {"none": "no salient sound event", "siren_wail": "a wailing siren", "siren_yelp": "a yelping siren",
                "horn": "a car horn", "reversing_beeper": "a reversing beeper", "tyre_squeal": "squealing tyres",
                "train_horn": "a train horn"}
SOURCE_LEVEL_DB = {"siren_wail": 118.0, "siren_yelp": 118.0, "horn": 110.0, "reversing_beeper": 100.0,
                   "tyre_squeal": 105.0, "train_horn": 125.0}  # SPL at 1 m -- ESTIMATED order-of-magnitude defaults, not measured
DIST_EDGES_M = (20.0, 40.0, 80.0)                           # -> 4 distance bins
DIST_PHRASE = ("under 20 m away", "about 20 to 40 m away", "about 40 to 80 m away", "more than 80 m away")
N_BEARING = 8                                               # 45-degree sectors, bearing CCW from +x (rig frame: +x forward, +y left)
SECTOR_NAMES = ("front", "front-left", "left", "rear-left", "rear", "rear-right", "right", "front-right")
APPROACH_MPS = 0.5                                          # radial speed above which an emitter is "approaching"
# Air absorption, dB/m at 20 C / 70 % RH, octave bands 63 Hz..8 kHz. UNVERIFIED: values recalled from ISO 9613-2 Table 2, not re-seen.
_ABS_F = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], np.float64)
_ABS_DB_PER_M = np.array([0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6], np.float64) / 1000.0
MONO = np.zeros((1, 2))
STEREO_1M = np.array([[0.0, 0.5], [0.0, -0.5]])            # left, right (y = +-0.5 m)
ROOF4 = np.array([[0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])  # FL, FR, RL, RR roof corners


# =============================================================================================== front end
@dataclass
class AudioConfig:
    sr: int = 16000
    win_s: float = 2.0                  # analysis window per refresh
    hop_s: float = 0.5                  # refresh hop (>= 2 Hz)
    n_fft: int = 400                    # 25 ms frame (Whisper / Qwen-Omni front end)
    stft_hop: int = 160                 # 10 ms
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    log_floor: float = 1e-10
    n_spatial_bands: int = 16           # coarse mel bands for inter-channel level / phase
    spatial_pool: int = 4               # STFT frames per spatial token (40 ms)
    max_lag: int = 64                   # GCC-PHAT lags +-64 samples = +-4 ms at 16 kHz (covers baselines up to 1.37 m)

    @property
    def frame_hz(self) -> float: return self.sr / self.stft_hop

    def d_spatial(self, n_channels: int) -> int:
        return 0 if n_channels < 2 else (n_channels - 1) * (3 * self.n_spatial_bands + 2 * self.max_lag + 1)


def _hz_to_mel(f): return 2595.0 * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)
def _mel_to_hz(m): return 700.0 * (10.0 ** (np.asarray(m, np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
    """HTK-mel triangular filters, area-normalised (Slaney), [n_fft//2+1, n_mels]."""
    f_max = f_max or sr / 2
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2))
    fb = np.zeros((len(freqs), n_mels))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(c - lo, 1e-9); dn = (hi - freqs) / max(hi - c, 1e-9)
        fb[:, i] = np.maximum(0.0, np.minimum(up, dn)) * (2.0 / max(hi - lo, 1e-9))
    return fb.astype(np.float32)


def window_stream(wave: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """[B, C, T_total] -> [B, N, C, T_win]: 2 s windows at 0.5 s hop (the 2 Hz refresh clock); the last window ends at T_total."""
    w, h = int(round(cfg.win_s * cfg.sr)), int(round(cfg.hop_s * cfg.sr))
    if wave.shape[-1] < w: raise ValueError(f"stream shorter than one window ({wave.shape[-1]} < {w})")
    return wave.unfold(-1, w, h).permute(0, 2, 1, 3).contiguous()


class AudioFrontend(nn.Module):
    """Level-preserving log-mel per channel (no per-clip normalisation, so the SNR dose-response control stays meaningful, note RF
    s2.3), an incoherent mono log-mel for the (mono) frozen tower, and -- for arrays -- inter-channel features against channel 0:
    ILD (dB) and IPD (cos, sin) on coarse mel bands plus GCC-PHAT over +-max_lag samples, pooled to 40 ms tokens."""

    def __init__(self, cfg: AudioConfig | None = None, n_channels: int = 1):
        super().__init__()
        self.cfg = cfg = cfg or AudioConfig(); self.n_channels = n_channels
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        self.register_buffer("mel_fb", torch.from_numpy(mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.f_min, cfg.f_max)), persistent=False)
        self.register_buffer("band_fb", torch.from_numpy(mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_spatial_bands, cfg.f_min, cfg.f_max)), persistent=False)

    @property
    def d_spatial(self) -> int: return self.cfg.d_spatial(self.n_channels)

    def stft(self, wave: torch.Tensor) -> torch.Tensor:          # [B, C, T] -> complex [B, C, n_freq, F]
        B, C, T = wave.shape
        X = torch.stft(wave.reshape(B * C, T).float(), self.cfg.n_fft, self.cfg.stft_hop, window=self.window,
                       center=True, pad_mode="constant", return_complex=True)
        return X.reshape(B, C, X.shape[-2], X.shape[-1])

    def gcc_phat(self, X: torch.Tensor) -> torch.Tensor:
        """Cross-correlation of channel m against channel 0, per spatial token [B, P, F//pool, 2L+1]: PHAT-whitened cross-spectra with a
        per-bin Wiener-like gain (1 - median-over-window / current, clamped to [0, 1]) that suppresses stationary, partly common-mode
        ego noise (which would otherwise pin the peak at lag 0), summed coherently over the `spatial_pool` frames of the token (40 ms,
        short enough that a moving emitter's delay does not drift within it). A peak at lag +D: channel m lags channel 0 by D samples.
        Mirroring the array time-reverses it exactly. Tonal sources are ambiguous at baselines > c / (2 f) -- physics, not a bug."""
        L, n, k = self.cfg.max_lag, self.cfg.n_fft, self.cfg.spatial_pool
        G = X[:, 1:] * X[:, :1].conj()                                             # [B,P,f,F]
        A_ = G.abs()
        gain = (1.0 - A_.median(-1, keepdim=True).values / (A_ + 1e-12)).clamp(0.0, 1.0)
        Gw = G / (A_ + 1e-12) * gain
        Fp = Gw.shape[-1] // k
        Gc = Gw[..., : Fp * k].reshape(*Gw.shape[:-1], Fp, k).sum(-1)               # [B,P,f,F']
        r = torch.fft.irfft(Gc, n=n, dim=2)
        r = torch.cat([r[:, :, n - L:], r[:, :, : L + 1]], 2)                       # lags -L..L  -> [B,P,2L+1,F']
        return r.permute(0, 1, 3, 2) / k

    def forward(self, wave: torch.Tensor) -> dict:
        """wave [B, C, T] (C must equal n_channels). Returns logmel [B,C,n_mels,F], mono_logmel [B,n_mels,F],
        spatial [B,F//pool,d_spatial] (None for mono) and gcc [B,C-1,2L+1] (None for mono)."""
        if wave.dim() == 2: wave = wave.unsqueeze(1)
        if wave.shape[1] != self.n_channels: raise ValueError(f"expected {self.n_channels} channels, got {wave.shape[1]}")
        cfg = self.cfg; X = self.stft(wave)
        P = X.real ** 2 + X.imag ** 2                                               # [B,C,f,F]
        mel = torch.einsum("bcft,fm->bcmt", P, self.mel_fb)
        out = {"logmel": torch.log10(mel.clamp_min(cfg.log_floor)),
               "mono_logmel": torch.log10(mel.mean(1).clamp_min(cfg.log_floor)), "spatial": None, "gcc": None}
        if self.n_channels < 2: return out
        Pb = torch.einsum("bcft,fk->bctk", P, self.band_fb)                         # [B,C,F,K]
        ild = 10.0 * torch.log10((Pb[:, 1:] + 1e-12) / (Pb[:, :1] + 1e-12))         # [B,P,F,K]
        G = X[:, 1:] * X[:, :1].conj()
        gr = torch.einsum("bpft,fk->bptk", G.real, self.band_fb); gi = torch.einsum("bpft,fk->bptk", G.imag, self.band_fb)
        mag = torch.sqrt(gr ** 2 + gi ** 2) + 1e-12
        lvl = torch.cat([ild / 10.0, gr / mag, gi / mag], -1)                       # ILD in units of 10 dB; IPD cos, sin
        B, Pn, Fr, D = lvl.shape
        k = cfg.spatial_pool; Fp = Fr // k
        lvl = lvl[:, :, : Fp * k].reshape(B, Pn, Fp, k, D).mean(3)                   # [B,P,F',3K]
        gcc = self.gcc_phat(X)                                                       # [B,P,F',2L+1]
        out["spatial"] = torch.cat([lvl, gcc], -1).permute(0, 2, 1, 3).reshape(B, Fp, -1)
        out["gcc"] = gcc.mean(2)                                                     # window-level [B,P,2L+1]
        return out


def whisper_normalise(logmel: torch.Tensor) -> torch.Tensor:
    """Whisper / Qwen-Omni feature scaling for wrapping a REAL tower: peak-relative clamp (max - 8) then (x + 4) / 4."""
    m = logmel.amax(dim=(-2, -1), keepdim=True)
    return (torch.maximum(logmel, m - 8.0) + 4.0) / 4.0


# =============================================================================================== frozen encoder
@runtime_checkable
class FrozenAudioEncoder(Protocol):
    """What the bridge needs from a frozen tower. A real adapter wraps e.g. `Qwen3OmniMoe...thinker.audio_tower` (d_enc 1280 before its
    own projector, 12.5 Hz), BEATs (768, patch tokens) or the Gemma-4 tower (1536, 25 Hz): it maps a mono log-mel [B, n_mels, F] to
    frame features [B, T_enc, d_enc], is in eval mode and has requires_grad False on every parameter (Qwen3-Omni's report warns that
    co-training the encoder against a frozen LM degrades perception -- note RG s1.2)."""
    d_enc: int
    frame_hz: float
    def __call__(self, mono_logmel: torch.Tensor) -> torch.Tensor: ...


def freeze_(m: nn.Module) -> nn.Module:
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval()


class StandInEncoder(nn.Module):
    """TESTS / SMOKE ONLY: a tiny deterministic frozen tower with the Qwen2.5-Omni geometry (conv k3 -> conv k3 stride 2 -> avg-pool 2,
    i.e. 40 ms frames) and seeded random weights. It carries spectral information but no learned audio semantics."""

    def __init__(self, n_mels: int = 128, d_enc: int = 128, seed: int = 0, frame_hz_in: float = 100.0):
        super().__init__()
        self.d_enc, self.frame_hz = d_enc, frame_hz_in / 4.0
        self.c1 = nn.Conv1d(n_mels, d_enc, 3, padding=1); self.c2 = nn.Conv1d(d_enc, d_enc, 3, stride=2, padding=1)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters(): p.copy_(torch.randn(p.shape, generator=g) * (1.0 / math.sqrt(max(p.shape[-1] * (p.shape[1] if p.dim() > 1 else 1), 1))))
        freeze_(self)

    def train(self, mode: bool = True):                            # always frozen
        return super().train(False)

    def forward(self, mono_logmel: torch.Tensor) -> torch.Tensor:
        x = (mono_logmel + 4.0) / 4.0                               # fixed affine; level preserved (no peak-relative clamp)
        x = F.gelu(self.c2(F.gelu(self.c1(x))))
        x = F.avg_pool1d(x, 2, 2)
        return F.layer_norm(x.transpose(1, 2), (self.d_enc,))      # [B, T_enc, d_enc]


# =============================================================================================== bridge
@dataclass
class BridgeConfig:
    d_enc: int = 1280                   # AuT / Whisper-large width before the vendor projector
    d_v: int = 2048                     # frozen backbone hidden size (Edge reasoner 2048; 8 B tier 4096)
    d_spatial: int = 0                  # AudioFrontend.d_spatial (0 = mono)
    width: int = 512
    heads: int = 8
    layers: int = 2
    n_latents: int = 8
    mlp_hidden: int = 0                 # 0 -> 2 * width (Qwen3-Omni's projector is Linear-GELU-Linear at encoder width)
    enc_hz: float = 12.5                # encoder frame rate (AuT 12.5, Whisper/Qwen2.5 25, Gemma-4 25)
    sp_hz: float = 25.0                 # spatial token rate (100 Hz STFT / spatial_pool 4)


def _time_embed(t_s: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal embedding of time in seconds (periods 0.04 s .. ~40 s); parameter-free."""
    half = width // 2
    per = 0.04 * (1000.0 ** (torch.arange(half, device=t_s.device, dtype=torch.float32) / max(half - 1, 1)))
    a = 2 * math.pi * t_s.float().unsqueeze(-1) / per
    e = torch.cat([a.sin(), a.cos()], -1)
    return F.pad(e, (0, width - e.shape[-1]))


class _PerceiverLayer(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.lq, self.lkv = nn.LayerNorm(w), nn.LayerNorm(w)
        self.att = nn.MultiheadAttention(w, heads, batch_first=True)
        self.lf = nn.LayerNorm(w); self.ff = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))

    def forward(self, lat, x, pad=None):                            # Flamingo resampler: latents attend to [inputs ; latents]
        q = self.lq(lat); kv = torch.cat([self.lkv(x), q], 1)
        if pad is not None: pad = torch.cat([pad, torch.zeros(pad.shape[0], lat.shape[1], dtype=torch.bool, device=pad.device)], 1)
        lat = lat + self.att(q, kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return lat + self.ff(self.lf(lat))


class AudioBridge(nn.Module):
    """Frozen-tower frames (+ optional spatial tokens) -> 8 tokens in R^{d_v}.
    in: LayerNorm + Linear to `width`, type + time embeddings; 2 Perceiver layers over 8 learned latents; out: LayerNorm -> Linear -> GELU
    -> Linear(d_v) -> RMSNorm without scale x `out_rms` (Gemma's post-projection norm idea): set `out_rms` to the backbone's measured
    input-embedding RMS before using mode (b), so the soft tokens arrive at the scale the LM expects. Mode (a) is scale-free."""

    def __init__(self, c: BridgeConfig | None = None):
        super().__init__()
        self.c = c = c or BridgeConfig(); w = c.width
        self.enc_in = nn.Sequential(nn.LayerNorm(c.d_enc), nn.Linear(c.d_enc, w))
        self.sp_in = nn.Sequential(nn.LayerNorm(c.d_spatial), nn.Linear(c.d_spatial, w)) if c.d_spatial else None
        self.type_emb = nn.Parameter(torch.zeros(2, w))
        self.latents = nn.Parameter(torch.randn(c.n_latents, w) * 0.02)
        self.layers = nn.ModuleList([_PerceiverLayer(w, c.heads) for _ in range(c.layers)])
        h = c.mlp_hidden or 2 * w
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, h), nn.GELU(), nn.Linear(h, c.d_v))
        self.register_buffer("out_rms", torch.tensor(1.0))

    def set_output_rms(self, rms: float):
        self.out_rms.fill_(float(rms)); return self

    def forward(self, enc: torch.Tensor, spatial: torch.Tensor | None = None, enc_pad: torch.Tensor | None = None) -> torch.Tensor:
        """enc [B, T, d_enc] (frozen tower output), spatial [B, T2, d_spatial] or None, enc_pad [B, T] True = padding.
        Returns tokens [B, n_latents, d_v]."""
        B, T, _ = enc.shape; c = self.c
        x = self.enc_in(enc) + self.type_emb[0] + _time_embed(torch.arange(T, device=enc.device) / c.enc_hz, c.width)
        pad = enc_pad
        if self.sp_in is not None and spatial is not None:
            s = self.sp_in(spatial) + self.type_emb[1] + _time_embed(torch.arange(spatial.shape[1], device=enc.device) / c.sp_hz, c.width)
            x = torch.cat([x, s], 1)
            if pad is not None: pad = torch.cat([pad, torch.zeros(B, s.shape[1], dtype=torch.bool, device=enc.device)], 1)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        for L in self.layers: lat = L(lat, x, pad)
        y = self.out(lat)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
        return y * self.out_rms

    @staticmethod
    def pooled(tokens: torch.Tensor) -> torch.Tensor: return tokens.mean(1)


def hicap_audio_input(tokens: torch.Tensor | None, present: torch.Tensor | None = None):
    """Mode (a): what goes into batch['audio'] for model.HiCAP (HiCAPConfig.d_audio must equal d_v).
    Returns None when no sample has audio -> GatedAudio skips its branch (bit-exact audio-null, test T1). With a MIXED batch the current
    GatedAudio has no per-sample mask, so absent samples must not receive zero tokens (cross-attention over zeros is not null):
    the orchestrator hook requested in the stream-A report is a `present` mask multiplying the gated residual."""
    if tokens is None: return None
    if present is not None and not bool(present.any()): return None
    return tokens


# ---- mode (b): soft tokens in the frozen VLM prefill (interface only; RF note s2.2 recommends the L3 'audio suffix' layout)
@dataclass
class SoftTokenSpec:
    n_tokens: int = 8
    open_text: str = "<audio>"
    close_text: str = "</audio>"
    layout: str = "suffix"              # after cameras + query, so no cached visual state depends on audio (RF s2.2, L3)


def splice_soft_tokens(inputs_embeds: torch.Tensor, placeholder: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """The omni pattern (Qwen `masked_scatter` at audio-token positions): inputs_embeds [B, L, d_v], placeholder [B, L] bool with exactly
    n_tokens True per row, tokens [B, n_tokens, d_v] -> new embeddings. Rows without audio must carry no placeholder (and no suffix)."""
    n = placeholder.sum(1)
    if not bool((n == tokens.shape[1]).all()): raise ValueError(f"placeholder counts {n.tolist()} != n_tokens {tokens.shape[1]}")
    return inputs_embeds.masked_scatter(placeholder.unsqueeze(-1).expand_as(inputs_embeds), tokens.to(inputs_embeds.dtype))


# =============================================================================================== losses and heads
class BridgeAlignLoss(nn.Module):
    """Stage A1 (CLAP-style): symmetric InfoNCE between pooled bridge outputs a [B, d_v] and the frozen text path's caption embeddings
    t [B, d_v] (the SAME space -- no text projection), + lam_cos * (1 - cos(a, t)). Identical captions in a batch are all positives
    (multi-positive soft targets), so duplicate class captions are not pushed apart as false negatives."""

    def __init__(self, lam_cos: float = 1.0, logit_init: float = math.log(1 / 0.07), logit_max: float = math.log(100.0)):
        super().__init__()
        self.lam_cos, self.logit_max = lam_cos, logit_max
        self.theta = nn.Parameter(torch.tensor(float(logit_init)))

    def forward(self, a: torch.Tensor, t: torch.Tensor, caption_ids: torch.Tensor | None = None) -> dict:
        an, tn = F.normalize(a, dim=-1), F.normalize(t.to(a.dtype), dim=-1)
        logits = self.theta.clamp(max=self.logit_max).exp() * an @ tn.T
        pos = (caption_ids[:, None] == caption_ids[None, :]) if caption_ids is not None else (tn @ tn.T > 1 - 1e-6)
        pos = pos.to(a.dtype)
        tgt_r, tgt_c = pos / pos.sum(1, keepdim=True), pos.T / pos.T.sum(1, keepdim=True)
        nce = 0.5 * (-(tgt_r * logits.log_softmax(1)).sum(1).mean() - (tgt_c * logits.T.log_softmax(1)).sum(1).mean())
        cos = (1 - (an * tn).sum(-1)).mean()
        with torch.no_grad(): top1 = pos.gather(1, logits.argmax(1, keepdim=True)).mean()
        return {"loss": nce + self.lam_cos * cos, "nce": nce, "cos": cos, "top1": top1}


def retrieve(a: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Index of the nearest caption embedding (cosine) for each pooled audio embedding."""
    return (F.normalize(a, dim=-1) @ F.normalize(bank.to(a.dtype), dim=-1).T).argmax(-1)


class EventHeads(nn.Module):
    """Stage A2 heads on the bridge tokens: class (incl. 'none'), bearing sector (8 x 45 deg; arrays only), distance bin (4),
    approaching (1 logit). Targets come from event metadata only (the label-echo argument, sec_elastic.tex)."""

    def __init__(self, d_v: int, n_classes: int = len(EVENT_CLASSES), n_bearing: int = N_BEARING, n_dist: int = len(DIST_EDGES_M) + 1,
                 hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_v); self.trunk = nn.Sequential(nn.Linear(d_v, hidden), nn.GELU())
        self.cls, self.bearing, self.dist, self.approach = (nn.Linear(hidden, n_classes), nn.Linear(hidden, n_bearing),
                                                            nn.Linear(hidden, n_dist), nn.Linear(hidden, 1))

    def forward(self, tokens: torch.Tensor) -> dict:
        h = self.trunk(self.norm(tokens).mean(1))
        return {"cls": self.cls(h), "bearing": self.bearing(h), "dist": self.dist(h), "approach": self.approach(h).squeeze(-1)}

    @staticmethod
    def loss(out: dict, tgt: dict) -> dict:
        """tgt: cls [B] long, bearing [B] long, dist [B] long, approach [B] float, bearing_valid [B] bool. Bearing / distance / approach
        are masked to samples with an event (cls != 0); bearing additionally to multi-channel samples."""
        ev = tgt["cls"] != 0
        z = out["cls"].sum() * 0.0
        l_cls = F.cross_entropy(out["cls"], tgt["cls"])
        mb = ev & tgt.get("bearing_valid", torch.ones_like(ev))
        l_b = F.cross_entropy(out["bearing"][mb], tgt["bearing"][mb]) if bool(mb.any()) else z
        l_d = F.cross_entropy(out["dist"][ev], tgt["dist"][ev]) if bool(ev.any()) else z
        l_a = F.binary_cross_entropy_with_logits(out["approach"][ev], tgt["approach"][ev].float()) if bool(ev.any()) else z
        return {"loss": l_cls + l_b + l_d + l_a, "cls": l_cls, "bearing": l_b, "dist": l_d, "approach": l_a}


# =============================================================================================== synthetic events
@dataclass
class EmitterTrack:
    """Emitter positions in the EGO RIG frame (+x forward, +y left, metres) at clip times t_s [K]; linear extrapolation outside."""
    t_s: np.ndarray
    xy: np.ndarray

    def at(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, np.float64); ts, xy = np.asarray(self.t_s, np.float64), np.asarray(self.xy, np.float64)
        if len(ts) == 1: return np.repeat(xy[:1], len(t), 0)
        out = np.stack([np.interp(t, ts, xy[:, k]) for k in range(2)], 1)
        lo, hi = t < ts[0], t > ts[-1]
        if lo.any(): out[lo] = xy[0] + (t[lo, None] - ts[0]) * (xy[1] - xy[0]) / (ts[1] - ts[0])
        if hi.any(): out[hi] = xy[-1] + (t[hi, None] - ts[-1]) * (xy[-1] - xy[-2]) / (ts[-1] - ts[-2])
        return out


def track_from_agent(agent_t, agent_xy_world, ego_t, ego_xy_world, ego_yaw) -> EmitterTrack:
    """An obstacle.offline track (clip-local world frame) -> rig frame, using the ego pose interpolated at the agent's timestamps."""
    agent_t = np.asarray(agent_t, np.float64)
    ex = np.interp(agent_t, ego_t, np.asarray(ego_xy_world)[:, 0]); ey = np.interp(agent_t, ego_t, np.asarray(ego_xy_world)[:, 1])
    yaw = np.interp(agent_t, ego_t, np.unwrap(np.asarray(ego_yaw, np.float64)))
    d = np.asarray(agent_xy_world, np.float64) - np.stack([ex, ey], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    return EmitterTrack(agent_t, np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1))


def linear_track(p0, v, t0: float = 0.0, duration: float = 2.0) -> EmitterTrack:
    p0, v = np.asarray(p0, np.float64), np.asarray(v, np.float64)
    return EmitterTrack(np.array([t0, t0 + duration]), np.stack([p0, p0 + v * duration]))


def source_components(cls: str, rng: np.random.Generator) -> list[dict]:
    """Procedural (licence-free) sources as sums of FM sinusoids: dict(fc, fd, T, amp, gate=(period, duty) | None, phase0).
    Instantaneous frequency fc + fd sin(2 pi t / T). Parameters are plausible, not calibrated against recordings (HYPOTHESIS)."""
    comps = []
    def add(fc, fd=0.0, T=1.0, amp=1.0, gate=None, harm=1, decay=1.0):
        for k in range(1, harm + 1): comps.append(dict(fc=fc * k, fd=fd * k, T=T, amp=amp * decay ** (k - 1), gate=gate, phase0=float(rng.uniform(0, 2 * np.pi))))
    j = lambda x, r=0.05: x * (1 + rng.uniform(-r, r))
    if cls == "siren_wail": add(j(950), j(350), j(4.0, 0.2), harm=4, decay=0.5)
    elif cls == "siren_yelp": add(j(950), j(350), j(0.3, 0.2), harm=4, decay=0.5)
    elif cls == "horn": [add(j(f, 0.03), harm=6, decay=0.7) for f in (420.0, 500.0)]
    elif cls == "reversing_beeper": add(j(1100, 0.03), gate=(1.0, 0.5), harm=2, decay=0.3)
    elif cls == "tyre_squeal":
        for _ in range(8): add(rng.uniform(1400, 2200), rng.uniform(20, 80), rng.uniform(0.1, 0.3), amp=rng.uniform(0.3, 1.0))
    elif cls == "train_horn": [add(f, harm=3, decay=0.6) for f in (311.0, 370.0, 440.0)]
    elif cls != "none": raise ValueError(f"unknown event class {cls}")
    return comps


def air_absorption_db_per_m(f: np.ndarray) -> np.ndarray:
    lf = np.log(np.clip(np.asarray(f, np.float64), _ABS_F[0], _ABS_F[-1] * 2))
    return np.exp(np.interp(lf, np.log(_ABS_F), np.log(_ABS_DB_PER_M), right=None))


def _phase(c: dict, t: np.ndarray) -> np.ndarray:
    if c["fd"] == 0: return 2 * np.pi * c["fc"] * t + c["phase0"]
    return 2 * np.pi * (c["fc"] * t - c["fd"] * c["T"] / (2 * np.pi) * np.cos(2 * np.pi * t / c["T"])) + c["phase0"]


def _colored_noise(n: int, rng: np.random.Generator, exponent: float, sr: int, f_hi: float | None = None) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / sr); f[0] = f[1]
    X = X / f ** (exponent / 2)
    if f_hi: X = X / np.sqrt(1 + (f / f_hi) ** 4)
    x = np.fft.irfft(X, n); return x / (x.std() + 1e-12)


def ego_noise_db(ego_speed: float) -> float:
    """Speed-dependent exterior-microphone noise floor, dB SPL. HYPOTHESIS (shape only): 50 dB at standstill, +25 log10(1 + v/5)."""
    return 50.0 + 25.0 * math.log10(1.0 + max(ego_speed, 0.0) / 5.0)


def distance_bin(r: float) -> int: return int(np.searchsorted(DIST_EDGES_M, r, side="right"))
def bearing_sector(bearing_deg: float) -> int: return int(np.round(bearing_deg / (360.0 / N_BEARING))) % N_BEARING


def render_synthetic_event(cls: str, track: EmitterTrack | None, t_start: float = 0.0, duration: float = 2.0, sr: int = 16000,
                           mic_xy: np.ndarray = MONO, ego_speed: float = 0.0, level_db: float | None = None, noise: bool = True,
                           seed: int = 0, components: list[dict] | None = None, directivity: float = 0.3,
                           noise_corr: float = 0.7) -> tuple[np.ndarray, dict]:
    """Render one window [C, N] float32 (RMS 1.0 <-> 120 dB SPL) with an emitter on `track` and its metadata.
    Physics: each microphone hears s(t_e) with retarded time t_e = t - r(t_e)/c (two fixed-point iterations), which yields the
    propagation delay, the per-microphone ITD and the Doppler shift f' = f c / (c - v_r) of a source moving at radial speed v_r
    toward the microphone; gain 1/max(r, 1 m); per-component air absorption; a cardioid-like microphone directivity
    (1 + d cos psi)/(1 + d) about each mic's outward normal (ILD); a speed-dependent ego-noise floor, `noise_corr` of its power
    correlated across channels (road/engine) and the rest independent (wind). Motion is relative to the rig frame (the receiver's
    own motion through air is folded into the relative velocity, first-order exact). Metadata are taken at the window END."""
    rng = np.random.default_rng(seed); mic_xy = np.asarray(mic_xy, np.float64); C = len(mic_xy)
    n = int(round(duration * sr)); t = t_start + np.arange(n) / sr
    wave = np.zeros((C, n))
    meta = dict(cls=cls, cls_id=EVENT_CLASSES.index(cls), t_end=t_start + duration, n_channels=C, bearing_valid=C >= 2,
                bearing_deg=float("nan"), sector=-1, distance_m=float("nan"), dist_bin=-1, v_radial=float("nan"),
                approaching=False, doppler_ratio=float("nan"), level_db=float("nan"), snr_db=float("nan"))
    if cls != "none":
        if track is None: raise ValueError("an event needs an emitter track")
        comps = components if components is not None else source_components(cls, rng)
        L = SOURCE_LEVEL_DB[cls] if level_db is None else level_db
        amp_src = 10 ** ((L - L_REF_DB) / 20) / math.sqrt(max(sum(c["amp"] ** 2 for c in comps) / 2, 1e-12))
        centre = mic_xy.mean(0)
        for m in range(C):
            p = track.at(t); r = np.linalg.norm(p - mic_xy[m], axis=1); te = t - r / C_SOUND
            for _ in range(2):
                p = track.at(te); r = np.linalg.norm(p - mic_xy[m], axis=1); te = t - r / C_SOUND
            nrm = mic_xy[m] - centre; nn_ = np.linalg.norm(nrm)
            if nn_ > 1e-9:
                u = (p - mic_xy[m]) / np.maximum(r, 1e-9)[:, None]
                dirg = (1 + directivity * (u @ (nrm / nn_))) / (1 + directivity)
            else: dirg = 1.0
            s = np.zeros(n)
            for c in comps:
                f_inst = c["fc"] + c["fd"] * np.sin(2 * np.pi * te / c["T"]) if c["fd"] else np.full(n, c["fc"])
                g = 10 ** (-air_absorption_db_per_m(f_inst) * r / 20)
                if c["gate"] is not None: g = g * ((np.mod(te, c["gate"][0]) < c["gate"][0] * c["gate"][1]).astype(np.float64))
                s += c["amp"] * g * np.sin(_phase(c, te))
            wave[m] = amp_src * s * dirg / np.maximum(r, 1.0)
        te_end = t[-1]; pe = track.at(np.array([te_end - 0.05, te_end]))
        rel = pe - centre; dist = np.linalg.norm(rel, axis=1)
        v_r = float(-(dist[1] - dist[0]) / 0.05)
        b = float(np.degrees(np.arctan2(rel[1, 1], rel[1, 0])))
        meta.update(bearing_deg=b, sector=bearing_sector(b), distance_m=float(dist[1]), dist_bin=distance_bin(float(dist[1])),
                    v_radial=v_r, approaching=bool(v_r > APPROACH_MPS), doppler_ratio=C_SOUND / (C_SOUND - v_r), level_db=float(L))
    ev_rms = float(np.sqrt((wave ** 2).mean())) if cls != "none" else 0.0
    if noise:
        a_n = 10 ** ((ego_noise_db(ego_speed) - L_REF_DB) / 20)
        common = _colored_noise(n, rng, 2.0, sr, f_hi=200.0)
        nz = np.stack([math.sqrt(noise_corr) * common + math.sqrt(1 - noise_corr) * _colored_noise(n, rng, 1.0, sr) for _ in range(C)])
        wave = wave + a_n * nz
        if cls != "none": meta["snr_db"] = float(20 * np.log10(max(ev_rms, 1e-12) / a_n))
    return wave.astype(np.float32), meta


def caption_for_event(meta: dict, detail: str = "full") -> str:
    """Caption from event METADATA only (class, motion, bearing sector if an array, distance bin); never from any driving label.
    detail='class' gives the class phrase alone (used for class-level retrieval)."""
    cls = meta["cls"]
    if cls == "none": return "no salient sound event, only road and wind noise"
    s = CLASS_PHRASE[cls]
    if detail == "class": return f"the sound of {s}"
    v = meta.get("v_radial", 0.0)
    motion = "approaching" if v > APPROACH_MPS else ("moving away" if v < -APPROACH_MPS else "holding its distance")
    where = f" from the {SECTOR_NAMES[meta['sector']]}" if meta.get("bearing_valid") and meta.get("sector", -1) >= 0 else ""
    return f"{s} {motion}{where}, {DIST_PHRASE[meta['dist_bin']]}"


# =============================================================================================== rule-based controls
def gcc_phat_tdoa(x_ref: np.ndarray, x: np.ndarray, sr: int, max_tau: float) -> float:
    """Delay of x relative to x_ref in seconds (positive: x arrives LATER), whole-signal GCC-PHAT."""
    nfft = 1 << int(np.ceil(np.log2(len(x) + len(x_ref))))
    G = np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(x_ref, nfft))
    r = np.fft.irfft(G / (np.abs(G) + 1e-12), nfft)
    L = max(1, int(round(max_tau * sr)))
    r = np.concatenate([r[-L:], r[: L + 1]])
    return (int(np.argmax(r)) - L) / sr


def rule_bearing_side(wave: np.ndarray, mic_xy: np.ndarray, sr: int = 16000) -> tuple[str, float]:
    """A deliberately simple, non-learned left/right estimate from the mic pair with the largest lateral baseline: if the right mic
    hears the sound later than the left mic, the source is on the left. Returns (side, tdoa_s). Used by the bearing-flip control."""
    mic_xy = np.asarray(mic_xy, np.float64)
    i_l, i_r = int(np.argmax(mic_xy[:, 1])), int(np.argmin(mic_xy[:, 1]))
    if mic_xy[i_l, 1] - mic_xy[i_r, 1] < 1e-6: raise ValueError("array has no lateral baseline")
    max_tau = np.linalg.norm(mic_xy[i_l] - mic_xy[i_r]) / C_SOUND * 1.2
    tau = gcc_phat_tdoa(np.asarray(wave[i_l], np.float64), np.asarray(wave[i_r], np.float64), sr, max_tau)
    return ("left" if tau > 0 else "right"), float(tau)


def mirror_permutation(mic_xy: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Channel permutation that mirrors the array across the vehicle's x axis (y -> -y). Raises if the array is not mirror-symmetric."""
    mic_xy = np.asarray(mic_xy, np.float64); mir = mic_xy * np.array([1.0, -1.0])
    perm = np.array([int(np.argmin(np.linalg.norm(mic_xy - mir[i], axis=1))) for i in range(len(mic_xy))])
    if np.abs(mic_xy[perm] - mir).max() > tol: raise ValueError("array is not mirror-symmetric")
    return perm


def mirror_channels(wave, mic_xy: np.ndarray):
    """The bearing-flip control: the signal the mirrored array would have recorded (left and right channels swapped)."""
    return wave[..., mirror_permutation(mic_xy), :] if isinstance(wave, np.ndarray) else wave[..., torch.as_tensor(mirror_permutation(mic_xy)), :]


# =============================================================================================== end-to-end convenience
class AudioPath(nn.Module):
    """front end -> frozen encoder -> bridge; returns tokens [B, 8, d_v] for a batch of windows wave [B, C, T]."""

    def __init__(self, frontend: AudioFrontend, encoder: FrozenAudioEncoder, bridge: AudioBridge):
        super().__init__()
        self.frontend, self.encoder, self.bridge = frontend, encoder, bridge

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        f = self.frontend(wave)
        with torch.no_grad(): enc = self.encoder(f["mono_logmel"])
        return self.bridge(enc, f["spatial"])


def n_params(m: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
