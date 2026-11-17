"""Transformer shapes and an operator-level roofline performance/energy model
for one *instance* = k chips running tensor parallelism (TP=k)."""
from dataclasses import dataclass
import numpy as np

ETA_ATTN_PREFILL = 0.55   # FlashAttention-class efficiency vs peak (softmax/exp bound)
ETA_ATTN_DECODE = 0.50
M_HALF = 24.0             # GEMM efficiency halves at M=24 rows (small-batch penalty)
T_FIXED = 1.5e-3          # per-iteration launch/scheduling overhead (s)
LINK_PJ_BIT = 5.0


@dataclass(frozen=True)
class LLM:
    name: str
    L: int
    d: int
    nh: int
    nkv: int
    hd: int
    f: int
    V: int
    wbytes: float = 1.0   # FP8 weights
    kvbytes: float = 1.0  # FP8 KV cache

    @property
    def p_layer(self):
        attn = self.d * self.nh * self.hd + 2 * self.d * self.nkv * self.hd + self.nh * self.hd * self.d
        mlp = 3 * self.d * self.f
        return attn + mlp

    @property
    def params(self):
        return self.L * self.p_layer + 2 * self.d * self.V

    @property
    def w_bytes(self):
        return self.params * self.wbytes

    @property
    def kv_tok(self):  # bytes of KV per token (all layers)
        return 2 * self.nkv * self.hd * self.L * self.kvbytes

    @property
    def act_tok_layer(self):  # activation bytes moved per token per layer (bf16)
        return (6 * self.d + 3 * self.f) * 2


LLAMA3_70B = LLM("Llama-3.1-70B", 80, 8192, 64, 8, 128, 28672, 128256)
LLAMA3_8B = LLM("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256)
LLAMA2_70B = LLM("Llama-2-70B", 80, 8192, 64, 8, 128, 28672, 32000)


def fits(model, chip, k, headroom=0.10):
    """Weights must leave at least `headroom` of instance memory for KV/activations."""
    return model.w_bytes / k <= chip.cap * (1 - headroom) - 2e9


def kv_capacity_tokens(model, chip, k):
    free = k * (chip.cap * 0.95 - 2e9) - model.w_bytes
    return max(0.0, free / model.kv_tok)


def _comm(model, chip, k, T_new):
    if k == 1:
        return 0.0, 0.0
    size = T_new * model.d * 2.0
    per_ar = 2 * (k - 1) / k * size / chip.link_bw + 2 * (k - 1) * chip.link_alpha
    t = model.L * 2 * per_ar
    bytes_ = model.L * 2 * 2 * (k - 1) / k * size * k
    return t, bytes_


def prefill_batch(model, chip, k, lens, cached=None):
    """Time (s) and dynamic energy (J) for one prefill batch of prompt lengths
    `lens`.  `cached` (optional, same shape) is the number of prompt tokens per
    request whose KV is already cached (prefix-cache hit): only the remaining
    tokens are computed, but they still attend to the cached prefix, whose KV is
    read from memory."""
    lens = np.asarray(lens, dtype=float)
    cached = np.zeros_like(lens) if cached is None else np.asarray(cached, dtype=float)
    new = lens - cached
    T = new.sum()
    n = len(lens)
    peak = chip.tflops * 1e12
    bw = chip.bw * chip.mem_eff
    # Linear layers on the new tokens (+ LM head for last token of each sequence)
    fl_lin = (2 * T * model.p_layer * model.L + 2 * n * model.d * model.V) / k
    by_lin = (model.w_bytes + T * model.L * model.act_tok_layer) / k
    eta = chip.eta_c * T / (T + M_HALF)
    t_lin = max(fl_lin / (peak * eta), by_lin / bw)
    # Causal attention: new tokens attend to the cached prefix and causally to each other
    # (with cached = 0 this equals 2 * s^2 * nh * hd * L)
    pairs = (new * cached + new ** 2 / 2).sum()
    fl_att = (4 * pairs * model.nh * model.hd * model.L) / k
    by_att = (T * model.kv_tok + cached.sum() * model.kv_tok
              + T * model.L * model.nh * model.hd * 2 * 2) / k
    t_att = max(fl_att / (peak * ETA_ATTN_PREFILL), by_att / bw)
    t_com, by_com = _comm(model, chip, k, T)
    t = t_lin + t_att + t_com + T_FIXED
    e = k * ((fl_lin + fl_att) * chip.e_flop + (by_lin + by_att) * chip.e_byte) \
        + by_com * LINK_PJ_BIT * 8e-12
    return t, e


def decode_iter(model, chip, k, B, S):
    """Vectorized decode-iteration time/energy.  B = batch size, S = sum of
    context lengths (array-like).  Returns (t, e) arrays."""
    S = np.asarray(S, dtype=float)
    peak = chip.tflops * 1e12
    bw = chip.bw * chip.mem_eff
    fl_lin = (2 * B * model.p_layer * model.L + 2 * B * model.d * model.V) / k
    by_lin = (model.w_bytes + B * model.L * model.act_tok_layer) / k
    eta = chip.eta_c * B / (B + M_HALF)
    t_lin = max(fl_lin / (peak * eta), by_lin / bw)
    fl_att = 4 * S * model.nh * model.hd * model.L / k
    by_att = S * model.kv_tok / k
    t_att = np.maximum(fl_att / (peak * ETA_ATTN_DECODE), by_att / bw)
    t_com, by_com = _comm(model, chip, k, B)
    t = t_lin + t_att + t_com + T_FIXED
    e = k * ((fl_lin + fl_att) * chip.e_flop + (by_lin + by_att) * chip.e_byte) \
        + by_com * LINK_PJ_BIT * 8e-12
    return t, e
