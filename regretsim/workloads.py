"""Workloads built from the Azure LLM Inference Trace 2023 (code, conversation),
plus drifted variants produced by scaling context and/or output lengths."""
from dataclasses import dataclass
from functools import lru_cache
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_SEQ = 131072  # Llama-3.1 context window
FLEET_RATE = 1000.0  # fleet-level arrival rate (req/s) that a load balancer spreads round-robin


@lru_cache(None)
def load_trace(base):
    f = os.path.join(DATA, f"AzureLLMInferenceTrace_{base}.csv")
    d = pd.read_csv(f, parse_dates=["TIMESTAMP"]).sort_values("TIMESTAMP")
    t = (d.TIMESTAMP - d.TIMESTAMP.iloc[0]).dt.total_seconds().to_numpy()
    gaps = np.diff(t, prepend=t[0])
    gaps[0] = gaps[1:].mean()
    gaps = gaps / gaps.mean()                      # unit-mean inter-arrival gaps (keeps burstiness)
    ctx = d.ContextTokens.to_numpy().astype(float)
    out = d.GeneratedTokens.to_numpy().astype(float)
    return gaps, ctx, out


@dataclass(frozen=True)
class Workload:
    base: str            # 'code' | 'conv'
    ctx_scale: float = 1.0
    out_scale: float = 1.0
    label: str = ""

    @property
    def name(self):
        return self.label or f"{self.base}-c{self.ctx_scale:g}-o{self.out_scale:g}"

    def lengths(self):
        _, ctx, out = load_trace(self.base)
        o = np.maximum(1, np.round(out * self.out_scale))
        c = np.maximum(1, np.round(ctx * self.ctx_scale))
        c = np.minimum(c, MAX_SEQ - o)
        return c, o

    def sample(self, n, rate, offset=0, fleet_rate=None):
        """Arrivals seen by ONE instance serving `rate` req/s inside a fleet that
        receives the production trace (cyclically repeated) at `fleet_rate`
        req/s and dispatches round-robin.  This keeps the trace's burst
        structure while spreading each burst across instances, as a real load
        balancer does.  Returns n requests (arrival time, ctx, out)."""
        F = FLEET_RATE if fleet_rate is None else fleet_rate
        gaps, _, _ = load_trace(self.base)
        c, o = self.lengths()
        L = len(gaps)
        cg = np.cumsum(gaps)
        tot = cg[-1]
        if rate >= F:
            idx = np.arange(n) + offset
            scale = rate
        else:
            m = F / rate
            idx = np.floor(np.arange(n) * m).astype(np.int64) + offset
            scale = F
        arr = ((idx // L) * tot + cg[idx % L]) / scale
        arr = arr - arr[0]
        return arr, c[idx % L], o[idx % L]


# Named scenarios (the "futures" a chip bought today may face)
TODAY = [Workload("code", label="Code-2023"), Workload("conv", label="Conv-2023")]
FUTURES = [
    Workload("conv", 1, 8, "Reasoning (out x8)"),
    Workload("conv", 16, 1, "Agentic/RAG (ctx x16)"),
    Workload("code", 16, 1, "Repo-scale code (ctx x16)"),
    Workload("conv", 8, 8, "Long-ctx reasoning (ctx x8, out x8)"),
]
SCENARIOS = TODAY + FUTURES


def drift_grid(base="conv", scales=(1, 2, 4, 8, 16, 32)):
    return [Workload(base, c, o) for c in scales for o in scales]
