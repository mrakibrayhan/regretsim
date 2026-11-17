"""Chip design, memory-technology catalog, and first-order cost / power models.

Every constant here is an explicit, documented assumption.  The study's main
results are *relative* (regret ratios), and experiments/05_sensitivity.py
re-runs the analysis with these constants perturbed.
"""
from dataclasses import dataclass, replace
import math

# ---------------------------------------------------------------------------
# Memory technologies (per "unit": HBM stack, GDDR7 32-bit device, LPDDR5X
# 64-bit package).  Bandwidths are peak; `eff` is achievable fraction.
# ---------------------------------------------------------------------------
MEMTECH = {
    # HBM3e 8-high, ~8 Gb/s/pin x 1024 -> ~1.0 TB/s, 24 GB per stack
    "HBM3e":   dict(bw=1.00e12, cap=24e9, usd_gb=15.0, pj_bit=3.5,
                    phy_mm2=12.0, adv_pkg=True,  eff=0.85),
    # HBM3 (H100 generation) for the validation baselines
    "HBM3":    dict(bw=0.67e12, cap=16e9, usd_gb=12.0, pj_bit=3.9,
                    phy_mm2=12.0, adv_pkg=True,  eff=0.85),
    # GDDR7 32 Gb/s x 32-bit device = 128 GB/s, 32 Gb (4 GB) device
    "GDDR7":   dict(bw=128e9,  cap=4e9,  usd_gb=6.5,  pj_bit=5.5,
                    phy_mm2=1.9,  adv_pkg=False, eff=0.80),
    # LPDDR5X 9600 MT/s x 64-bit package = 76.8 GB/s, 32 GB package
    "LPDDR5X": dict(bw=76.8e9, cap=32e9, usd_gb=4.0,  pj_bit=4.5,
                    phy_mm2=2.5,  adv_pkg=False, eff=0.75),
}

LINKS = {  # per-chip scale-up link (used for tensor parallelism)
    "fast": dict(bw=900e9, alpha=1.0e-6, usd=250.0, watt=30.0),   # NVLink-class
    "slow": dict(bw=128e9, alpha=3.0e-6, usd=40.0,  watt=8.0),    # PCIe6-class
}
SCALEOUT_BW = 50e9      # per-chip NIC (400 Gb/s) used for prefill->decode KV transfer

# ---------------------------------------------------------------------------
# Silicon / package / TCO constants
# ---------------------------------------------------------------------------
WAFER_USD = 18000.0         # N4-class 300 mm wafer
D0_PER_MM2 = 0.0007         # defect density (0.07 /cm^2), Poisson yield
RETICLE_MM2 = 800.0
BASE_MM2 = 120.0            # IO, NoC, control, L2 baseline
MM2_PER_TF = 0.22           # dense FP8 tensor throughput density
D2D_MM2 = 25.0              # die-to-die interface per die in multi-die parts
PKG_ORGANIC_USD = 100.0
PKG_ADV_BASE_USD = 600.0    # CoWoS-S class interposer
PKG_ADV_PER_HBM = 100.0
PKG_ADV_MULTIDIE = 400.0    # CoWoS-L / bridge premium
BOARD_USD = 300.0
BOARD_USD_PER_W = 0.3
MARKUP = 1.5                # BOM -> acquisition cost (custom-ASIC margin, test, NRE share)
HOST_USD_PER_CHIP = 2500.0  # share of host CPU, NIC, chassis, rack

PJ_PER_FLOP = 0.20          # FP8 incl. on-chip data movement (W per TF/s at 100% util)
STATIC_W = 40.0
STATIC_W_PER_TF = 0.03
MEM_STATIC_W_PER_GB = 0.06
MISC_W = 30.0

LIFETIME_H = 4 * 8760.0
FACILITY_USD_PER_W = 12.0   # datacenter build cost per provisioned W
FACILITY_LIFE_H = 12 * 8760.0
USD_PER_KWH = 0.08
PUE = 1.25


@dataclass(frozen=True)
class Chip:
    name: str
    tflops: float            # dense FP8 TFLOP/s
    mem: str                 # key into MEMTECH
    units: int               # stacks / devices / packages
    clamshell: bool = False  # GDDR7 only: 2 devices per channel (capacity x2, bw x1)
    link: str = "fast"
    bw_override: float = None
    cap_override: float = None
    eta_c: float = 0.45      # realized GEMM efficiency; calibrated on MLPerf H100 (experiments/01)

    # ---- memory -----------------------------------------------------------
    @property
    def m(self):
        return MEMTECH[self.mem]

    @property
    def bw(self):
        return self.bw_override or self.units * self.m["bw"]

    @property
    def cap(self):
        if self.cap_override:
            return self.cap_override
        return self.units * self.m["cap"] * (2 if self.clamshell else 1)

    @property
    def mem_eff(self):
        return self.m["eff"]

    # ---- link -------------------------------------------------------------
    @property
    def link_bw(self):
        return LINKS[self.link]["bw"]

    @property
    def link_alpha(self):
        return LINKS[self.link]["alpha"]

    # ---- silicon ----------------------------------------------------------
    @property
    def logic_mm2(self):
        return BASE_MM2 + MM2_PER_TF * self.tflops + self.units * self.m["phy_mm2"]

    @property
    def n_dies(self):
        return max(1, math.ceil(self.logic_mm2 / (RETICLE_MM2 - D2D_MM2)))

    @property
    def die_mm2(self):
        n = self.n_dies
        return self.logic_mm2 / n + (D2D_MM2 if n > 1 else 0.0)

    def die_cost(self):
        a = self.die_mm2
        r = 150.0
        dpw = math.pi * r * r / a - math.pi * 2 * r / math.sqrt(2 * a)
        y = math.exp(-a * D0_PER_MM2)
        return WAFER_USD / (dpw * y)

    # ---- power ------------------------------------------------------------
    @property
    def e_flop(self):          # J / FLOP
        return PJ_PER_FLOP * 1e-12

    @property
    def e_byte(self):          # J / byte of DRAM traffic
        return self.m["pj_bit"] * 8e-12

    @property
    def static_w(self):
        return (STATIC_W + STATIC_W_PER_TF * self.tflops + MISC_W
                + MEM_STATIC_W_PER_GB * self.cap / 1e9 + LINKS[self.link]["watt"])

    @property
    def tdp(self):
        dyn_c = self.tflops * 1e12 * self.e_flop
        dyn_m = self.bw * self.e_byte
        return self.static_w + dyn_c + dyn_m

    # ---- cost ---------------------------------------------------------------
    def bom(self):
        dies = self.n_dies * self.die_cost()
        mem = self.cap / 1e9 * self.m["usd_gb"]
        if self.m["adv_pkg"] or self.n_dies > 1:
            pkg = PKG_ADV_BASE_USD + (PKG_ADV_PER_HBM * self.units if self.m["adv_pkg"] else 0)
            if self.n_dies > 1:
                pkg += PKG_ADV_MULTIDIE
        else:
            pkg = PKG_ORGANIC_USD
        board = BOARD_USD + BOARD_USD_PER_W * self.tdp
        return dies + mem + pkg + board + LINKS[self.link]["usd"]

    def capex(self):
        return MARKUP * self.bom() + HOST_USD_PER_CHIP

    def fixed_usd_per_h(self):
        """Hourly cost independent of load: amortized capex, facility, static energy."""
        cap = self.capex() / LIFETIME_H
        fac = FACILITY_USD_PER_W * self.tdp / FACILITY_LIFE_H
        static_e = self.static_w * PUE / 1000.0 * USD_PER_KWH
        return cap + fac + static_e

    def describe(self):
        mem = f"{self.units}x{self.mem}" + ("(cs)" if self.clamshell else "")
        return (f"{self.name}: {self.tflops:.0f}TF {mem} {self.bw/1e12:.2f}TB/s "
                f"{self.cap/1e9:.0f}GB link={self.link} dies={self.n_dies} "
                f"BOM=${self.bom():,.0f} TDP={self.tdp:.0f}W")


USD_PER_J = USD_PER_KWH / 3.6e6 * PUE
TDP_CAP_W = 1800.0   # package power limit (liquid-cooled, Rubin-class)

# ---------------------------------------------------------------------------
# Reference (real-world-like) chips used for validation and as baselines
# ---------------------------------------------------------------------------
H100 = Chip("H100-like", 1979, "HBM3", 5, link="fast", bw_override=3.35e12, cap_override=80e9)
H200 = Chip("H200-like", 1979, "HBM3e", 6, link="fast", bw_override=4.8e12, cap_override=141e9)
B200 = Chip("B200-like", 4500, "HBM3e", 8, link="fast", bw_override=8.0e12, cap_override=192e9)


def design_space():
    """The chip design space explored in this study."""
    tfs = [500, 1000, 2000, 3000, 4500, 6000]
    mems = []
    for s in (2, 4, 6, 8):
        mems.append(("HBM3e", s, False))
    for dev in (8, 12, 16):                  # 256 / 384 / 512-bit GDDR7
        for cs in (False, True):
            mems.append(("GDDR7", dev, cs))
    for pk in (8, 16):                       # 512 / 1024-bit LPDDR5X
        mems.append(("LPDDR5X", pk, False))
    out = []
    for tf in tfs:
        for (m, u, cs) in mems:
            for ln in ("fast", "slow"):
                nm = f"{tf}TF-{u}{m}{'cs' if cs else ''}-{ln}"
                c = Chip(nm, tf, m, u, cs, ln)
                if c.tdp <= TDP_CAP_W:
                    out.append(c)
    return out
