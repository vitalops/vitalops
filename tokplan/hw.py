"""The hardware catalogue: public datasheet figures for each GPU.

Every number here comes from the vendor's datasheet, except the two marked as assumptions
(`latency`, the DRAM latency used nowhere in the planner but kept for parity with the kernel
tools, and `price`, a default dollars-per-GPU-hour that the caller can override).

Fields
  mem        GB, usable by one process (the vendor's figure)
  mem_type   HBM3, HBM3e, GDDR6, ...
  stacks     HBM stacks or GDDR channels, and `stack_gb` the capacity of each
  bw         TB/s, peak memory bandwidth
  flops      dense TFLOPS by dtype; `fp32` is the vector rate, the others tensor-core rates
  sms        streaming multiprocessors (NVIDIA) or compute units (AMD)
  arch       compute capability or ISA family
  link       interconnect usable per direction in GB/s, and its name
  tdp        W
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class GPU:
    key: str
    name: str
    vendor: str
    arch: str
    sms: int
    mem: float
    mem_type: str
    stacks: int
    stack_gb: float
    bw: float
    flops: Dict[str, float]
    link_name: str
    link_gbps: float
    tdp: int
    price: float
    latency_ns: int = 650
    note: str = ""

    @property
    def pcie(self) -> bool:
        return "PCIe" in self.link_name

    def rate(self, weights: str) -> tuple:
        """The dense tensor-core rate the matmuls run at for a weight format: (TFLOPS, dtype used)."""
        f = self.flops
        if weights == "bf16":
            return f["bf16"], "bf16"
        if weights == "int4" and "fp4" in f:
            return f["fp4"], "fp4"
        if "fp8" in f:
            return f["fp8"], "fp8"
        return f["bf16"], "bf16"


def _g(**kw) -> GPU:
    return GPU(**kw)


GPUS: Dict[str, GPU] = {
    "h100": _g(key="h100", name="H100 SXM", vendor="nvidia", arch="sm_90", sms=132, mem=80, mem_type="HBM3", stacks=5, stack_gb=16, bw=3.35,
               flops={"fp32": 67, "bf16": 989, "fp8": 1979, "int8": 1979}, link_name="NVLink 4", link_gbps=450, tdp=700, price=2.50),
    "h200": _g(key="h200", name="H200 SXM", vendor="nvidia", arch="sm_90", sms=132, mem=141, mem_type="HBM3e", stacks=6, stack_gb=24, bw=4.8,
               flops={"fp32": 67, "bf16": 989, "fp8": 1979, "int8": 1979}, link_name="NVLink 4", link_gbps=450, tdp=700, price=3.50),
    "b200": _g(key="b200", name="B200", vendor="nvidia", arch="sm_100", sms=148, mem=180, mem_type="HBM3e", stacks=8, stack_gb=24, bw=8.0,
               flops={"fp32": 80, "bf16": 2250, "fp8": 4500, "int8": 4500, "fp4": 9000}, link_name="NVLink 5", link_gbps=900, tdp=1000, price=5.50, latency_ns=700),
    "a100": _g(key="a100", name="A100 SXM 80 GB", vendor="nvidia", arch="sm_80", sms=108, mem=80, mem_type="HBM2e", stacks=5, stack_gb=16, bw=2.0,
               flops={"fp32": 19.5, "bf16": 312, "int8": 624}, link_name="NVLink 3", link_gbps=300, tdp=400, price=1.60, latency_ns=550,
               note="no fp8 tensor cores: fp8 weights are dequantised and the matmuls run at the bf16 rate"),
    "l40s": _g(key="l40s", name="L40S", vendor="nvidia", arch="sm_89", sms=142, mem=48, mem_type="GDDR6", stacks=12, stack_gb=4, bw=0.864,
               flops={"fp32": 91.6, "bf16": 362, "fp8": 733, "int8": 733}, link_name="PCIe 4.0 x16", link_gbps=25, tdp=350, price=1.00, latency_ns=550),
    "rtx5090": _g(key="rtx5090", name="RTX 5090", vendor="nvidia", arch="sm_120", sms=170, mem=32, mem_type="GDDR7", stacks=16, stack_gb=2, bw=1.79,
                  flops={"fp32": 104.8, "bf16": 209, "fp8": 419, "int8": 419, "fp4": 838}, link_name="PCIe 5.0 x16", link_gbps=50, tdp=575, price=0.70, latency_ns=550),
    "mi300x": _g(key="mi300x", name="MI300X", vendor="amd", arch="cdna3", sms=304, mem=192, mem_type="HBM3", stacks=8, stack_gb=24, bw=5.3,
                 flops={"fp32": 163.4, "bf16": 1307, "fp8": 2615, "int8": 2615}, link_name="Infinity Fabric", link_gbps=448, tdp=750, price=2.50, latency_ns=850),
    "agx_orin": _g(key="agx_orin", name="Jetson AGX Orin 64 GB", vendor="nvidia", arch="sm_87", sms=16, mem=64, mem_type="LPDDR5", stacks=8, stack_gb=8, bw=0.2048,
                   flops={"fp32": 5.3, "bf16": 42, "int8": 85}, link_name="PCIe 4.0 x8", link_gbps=12, tdp=60, price=0.0, latency_ns=450,
                   note="memory is shared with the CPU; price 0 because it is a module you own"),
}

ALIASES = {
    "h100-sxm": "h100", "h100sxm": "h100", "h200-sxm": "h200", "5090": "rtx5090", "rtx-5090": "rtx5090",
    "mi300": "mi300x", "a100-80": "a100", "a100-80gb": "a100", "orin": "agx_orin", "agx-orin": "agx_orin", "jetson": "agx_orin",
}


def gpu(key: str) -> GPU:
    """Look a GPU up by key or alias; case-insensitive."""
    k = key.lower().replace(" ", "")
    k = ALIASES.get(k, k)
    if k not in GPUS:
        raise KeyError("unknown GPU %r; known: %s" % (key, ", ".join(GPUS)))
    return GPUS[k]


GiB = 1073741824
