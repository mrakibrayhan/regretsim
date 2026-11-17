"""Phase-disaggregated serving: discrete-event simulators for one prefill
instance and one decode instance, capacity search under SLOs, a fast
analytical decode-capacity model (validated against the simulator), and the
cost-per-request metric used throughout the study."""
from collections import deque
from functools import lru_cache
import numpy as np

from .hardware import H100, SCALEOUT_BW, USD_PER_J
from .workloads import load_trace
from .model import (LLAMA3_70B, prefill_batch, decode_iter, fits,
                    kv_capacity_tokens)

PREFILL_TOKEN_BUDGET = 16384
DECODE_BMAX = 512
TTFT_SLO_NORM = 3.0       # P90 TTFT <= 3x unloaded TTFT on reference H100 (TP=2)
TPOT_SLO = 0.050          # P90 time-per-output-token <= 50 ms
P = 90
WARMUP = 0.10
TP_OPTIONS = (1, 2, 4, 8)
SAMPLE_OFFSET = 0         # start index in the (cyclic) trace stream; varied in experiments/08

# analytical decode model calibration constants (fitted in experiments/02_validate_sim.py)
GAMMA = 1.10              # P90 TPOT / mean iteration time at capacity
KAPPA = 0.90              # usable fraction of KV capacity (fluctuation headroom)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
def kv_xfer(model, k, lens):
    return lens * model.kv_tok / (k * SCALEOUT_BW)


def ref_ttft(model, lens):
    """Unloaded single-request TTFT on the reference instance (H100-like, TP=2)."""
    out = np.empty(len(lens))
    for i, s in enumerate(lens):
        out[i] = prefill_batch(model, H100, 2, [s])[0]
    return out + kv_xfer(model, 2, lens)


@lru_cache(None)
def _ref_table(model, base, ctx_scale):
    from .workloads import Workload
    c, _ = Workload(base, ctx_scale, 1).lengths()
    u = np.unique(c)
    return dict(zip(u, ref_ttft(model, u)))


def sim_prefill(model, chip, k, arr, lens):
    n = len(arr)
    ttft = np.empty(n)
    E = 0.0
    busy = 0.0
    t = 0.0
    i = 0
    while i < n:
        if t - arr[i] > 120.0:          # backlog > 2 min: unstable, abort
            ttft[i:] = np.inf
            break
        t = max(t, arr[i])
        j = i
        tok = 0.0
        while j < n and arr[j] <= t and (tok + lens[j] <= PREFILL_TOKEN_BUDGET or j == i):
            tok += lens[j]
            j += 1
        dt, e = prefill_batch(model, chip, k, lens[i:j])
        t += dt
        busy += dt
        E += e
        ttft[i:j] = t - arr[i:j]
        i = j
    ttft += kv_xfer(model, k, lens)
    return ttft, E


def prefill_metric(model, chip, k, wl, rate, n=1500):
    arr, c, _ = wl.sample(n, rate, offset=SAMPLE_OFFSET)
    ttft, E = sim_prefill(model, chip, k, arr, c)
    tab = _ref_table(model, wl.base, wl.ctx_scale)
    ref = np.array([tab[x] for x in c])
    w = int(WARMUP * n)
    x = np.nan_to_num((ttft / ref)[w:], nan=1e9, posinf=1e9)
    return np.percentile(x, P), E / n


def _bisect_rate(ok, lo, hi, iters=7):
    """Largest rate with ok(rate) True, assuming monotonicity."""
    if not ok(lo):
        return 0.0
    while ok(hi):
        lo, hi = hi, hi * 2
        if hi > 1e5:
            return lo
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def prefill_capacity(model, chip, k, wl, n=1500):
    c, _ = wl.lengths()
    t1 = prefill_batch(model, chip, k, [np.mean(c)])[0]
    guess = 1.0 / t1
    cache = {}

    def ok(r):
        m, e = prefill_metric(model, chip, k, wl, r, n)
        cache[r] = e
        return m <= TTFT_SLO_NORM
    cap = _bisect_rate(ok, guess * 0.05, guess)
    if cap == 0:
        return 0.0, np.inf
    return cap, cache[cap]


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def sim_decode(model, chip, k, arr, ctx, out, bmax=DECODE_BMAX):
    """Continuous batching with KV reservation (ctx+out).  Iteration-level
    simulation with exact fast-forwarding between scheduling events."""
    n = len(arr)
    cap_tok = kv_capacity_tokens(model, chip, k)
    steps = np.maximum(out - 1, 0)
    need = ctx + out
    finish = np.full(n, np.nan)
    waiting = deque()
    run_idx = np.empty(0, dtype=int)
    cur = np.empty(0)
    rem = np.empty(0)
    kv = 0.0
    t = 0.0
    qi = 0
    E = 0.0
    done = 0
    for i in range(n):
        if need[i] > cap_tok:
            return None  # request can never be served on this instance
    while done < n:
        if len(waiting) > max(64, n // 4):
            return np.full(int((steps > 0).sum()), np.inf), E   # unstable: abort
        while qi < n and arr[qi] <= t:
            if steps[qi] == 0:
                finish[qi] = arr[qi]
                done += 1
            else:
                waiting.append(qi)
            qi += 1
        add = []
        while waiting and len(run_idx) + len(add) < bmax and kv + need[waiting[0]] <= cap_tok:
            w = waiting.popleft()
            kv += need[w]
            add.append(w)
        if add:
            add = np.array(add)
            run_idx = np.concatenate([run_idx, add])
            cur = np.concatenate([cur, ctx[add]])
            rem = np.concatenate([rem, steps[add]])
        if len(run_idx) == 0:
            if qi < n:
                t = arr[qi]
                continue
            break
        B = len(run_idx)
        ns = int(rem.min())
        js = np.arange(ns)
        dts, es = decode_iter(model, chip, k, B, cur.sum() + js * B)
        cum = np.cumsum(dts)
        if qi < n and not waiting and B < bmax and arr[qi] < t + cum[-1]:
            cut = int(np.searchsorted(cum, arr[qi] - t)) + 1
            ns = min(ns, cut)
        t += cum[ns - 1]
        E += es[:ns].sum()
        cur += ns
        rem -= ns
        fin = rem <= 0
        if fin.any():
            fi = run_idx[fin]
            finish[fi] = t
            kv -= need[fi].sum()
            done += len(fi)
            keep = ~fin
            run_idx, cur, rem = run_idx[keep], cur[keep], rem[keep]
    tpot = (finish - arr) / np.maximum(steps, 1)
    return tpot[steps > 0], E


def decode_metric_sim(model, chip, k, wl, rate, n=1500):
    arr, c, o = wl.sample(n, rate, offset=SAMPLE_OFFSET)
    r = sim_decode(model, chip, k, arr, c, o)
    if r is None:
        return np.inf, np.inf
    tpot, E = r
    w = int(WARMUP * len(tpot))
    x = np.nan_to_num(tpot[w:], nan=1e9, posinf=1e9)
    return np.percentile(x, P), E / n


def decode_capacity_sim(model, chip, k, wl, n=1500):
    cache = {}

    def ok(r):
        m, e = decode_metric_sim(model, chip, k, wl, r, n)
        cache[r] = e
        return m <= TPOT_SLO
    a, _ = decode_capacity_analytic(model, chip, k, wl)
    g = a if a > 0 else 1.0
    cap = _bisect_rate(ok, g * 0.05, g)
    if cap == 0:
        return 0.0, np.inf
    return cap, cache[cap]


@lru_cache(None)
def _decode_stats(wl):
    c, o = wl.lengths()
    s = np.maximum(o - 1, 0)
    Es = s.mean()
    cbar = (s * (c + s / 2)).sum() / s.sum()
    rbar = (s * (c + o)).sum() / s.sum()
    # service-step distribution (for the P90 TPOT tail) and burstiness
    sv = s[s > 0]
    qs = np.quantile(sv, np.linspace(0.0025, 0.9975, 200))
    gaps, _, _ = load_trace(wl.base)
    ca2 = gaps.var() / gaps.mean() ** 2
    cs2 = sv.var() / sv.mean() ** 2
    return Es, cbar, rbar, (c + o).max(), qs, 0.5 * (ca2 + cs2)


def _erlang_c(c, a):
    """Erlang-C probability of waiting for c servers, offered load a (a < c)."""
    b = 1.0
    for n in range(1, int(c) + 1):
        b = a * b / (n + a * b)
    rho = a / c
    return b / (1 - rho * (1 - b))


def decode_capacity_analytic(model, chip, k, wl, gamma=None, kappa=None):
    """Little's-law fixed point N = lambda*E[steps]*t_iter(N), with an
    Erlang-C / Allen-Cunneen queueing correction for the admission wait that is
    included in TPOT when KV slots (or the batch cap) are the binding limit."""
    gamma = GAMMA if gamma is None else gamma
    kappa = KAPPA if kappa is None else kappa
    Es, cbar, rbar, mx, qs, f_ac = _decode_stats(wl)
    cap_tok = kv_capacity_tokens(model, chip, k)
    if mx > cap_tok or cap_tok <= 0:
        return 0.0, np.inf
    slots = max(1.0, min(float(DECODE_BMAX), kappa * cap_tok / rbar))

    def t_of(N):
        return decode_iter(model, chip, k, N, [N * cbar])[0][0]

    if gamma * t_of(1.0) > TPOT_SLO:
        return 0.0, np.inf

    def ok(N):  # N = mean number of running requests
        if N >= slots:
            return False
        t = t_of(N)
        if gamma * t > TPOT_SLO:
            return False
        lam = N / (Es * t)
        mu = 1.0 / (Es * t)
        c = int(np.floor(slots))
        if N >= c:
            return False
        pc = _erlang_c(c, N)
        slack = np.maximum(qs * (TPOT_SLO / gamma - t), 0.0)
        p_viol = (pc * np.exp(-(c * mu - lam) * slack / f_ac)).mean()
        return p_viol <= (100 - P) / 100.0

    lo, hi = 0.0, slots
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mid > 0 and ok(mid):
            lo = mid
        else:
            hi = mid
    N = lo
    if N <= 1e-6:
        return 0.0, np.inf
    t, e = decode_iter(model, chip, k, N, [N * cbar])
    lam = N / (Es * t[0])
    e_req = e[0] / N * Es
    return lam, e_req


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def cost_per_req(chip, k, cap, e_req):
    """USD per request for one phase on this instance type at its SLO capacity."""
    if cap <= 0:
        return np.inf
    return k * chip.fixed_usd_per_h() / (3600.0 * cap) + e_req * USD_PER_J


def best_phase_cost(chip, wl, phase, model=LLAMA3_70B, decode_mode="sim"):
    """Best cost/request over TP options for `phase` on `chip`."""
    best = (np.inf, None, 0.0)
    ks = [k for k in TP_OPTIONS if fits(model, chip, k)]
    ks = ks[:2]                      # smallest feasible TP degree and 2x that
    for k in ks:
        if phase == "prefill":
            cap, e = prefill_capacity(model, chip, k, wl)
        elif decode_mode == "analytic":
            cap, e = decode_capacity_analytic(model, chip, k, wl)
        else:
            cap, e = decode_capacity_sim(model, chip, k, wl)
        c = cost_per_req(chip, k, cap, e)
        if c < best[0]:
            best = (c, k, cap)
    return best
