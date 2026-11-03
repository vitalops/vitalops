# GPUs

`vitalops gpus` lists the hardware catalogue. Every figure is from the vendor's datasheet except the price, which is a default you can override with `--price`, and the DRAM latency, which is an assumption the planner does not use (it is kept for parity with the kernel tools on vitalops.ai).

| Key | Name | Memory | Bandwidth | Dense TFLOPS | Interconnect, per direction | Power | Default $/h |
|---|---|---|---|---|---|---|---|
| `h100` | H100 SXM | 80 GB HBM3, 5 × 16 GB | 3.35 TB/s | bf16 989 · fp8 1,979 | NVLink 4, 450 GB/s | 700 W | 2.50 |
| `h200` | H200 SXM | 141 GB HBM3e, 6 × 24 GB | 4.8 TB/s | bf16 989 · fp8 1,979 | NVLink 4, 450 GB/s | 700 W | 3.50 |
| `b200` | B200 | 180 GB HBM3e, 8 × 24 GB | 8.0 TB/s | bf16 2,250 · fp8 4,500 · fp4 9,000 | NVLink 5, 900 GB/s | 1,000 W | 5.50 |
| `a100` | A100 SXM 80 GB | 80 GB HBM2e, 5 × 16 GB | 2.0 TB/s | bf16 312 · int8 624 | NVLink 3, 300 GB/s | 400 W | 1.60 |
| `l40s` | L40S | 48 GB GDDR6, 12 channels | 0.864 TB/s | bf16 362 · fp8 733 | PCIe 4.0 x16, 25 GB/s | 350 W | 1.00 |
| `rtx5090` | RTX 5090 | 32 GB GDDR7, 16 channels | 1.79 TB/s | bf16 209 · fp8 419 · fp4 838 | PCIe 5.0 x16, 50 GB/s | 575 W | 0.70 |
| `mi300x` | MI300X | 192 GB HBM3, 8 × 24 GB | 5.3 TB/s | bf16 1,307 · fp8 2,615 | Infinity Fabric, 448 GB/s | 750 W | 2.50 |
| `agx_orin` | Jetson AGX Orin 64 GB | 64 GB LPDDR5, shared with the CPU | 0.205 TB/s | bf16 42 · int8 85 | PCIe 4.0 x8, 12 GB/s | 60 W | 0 |

Aliases: `h100-sxm`, `h200-sxm`, `5090`, `rtx-5090`, `mi300`, `a100-80gb`, `orin`, `jetson`.

## How each figure is used

- **Memory** is the budget: 8% of it plus 1.5 GB is reserved for the runtime, the weights take their share, and what is left is KV cache. The stacks and channels are drawn by the interactive planner; memory is interleaved across them, so each shows the same fill.
- **Bandwidth** sets the decode step: the weights plus the batch's KV cache are read at 75% of this figure.
- **Dense TFLOPS** by format sets the compute floor and the prefill rate. The matmuls run at the rate for the weight format the GPU has: bf16 everywhere; fp8 where there are fp8 tensor cores (not on the A100, so fp8 weights are dequantised and run at the bf16 rate there); int4 as NVFP4 on Blackwell (B200, RTX 5090) and otherwise at the fp8 or bf16 rate. The report's `weights` line says which rate was used.
- **Interconnect** sets the tensor-parallel cost: two all-reduces per layer, each moving 2(tp−1)/tp of the activation tensor through this link, with 8 µs of latency on NVLink and Infinity Fabric and 25 µs on PCIe. Over PCIe this is a large share of the step, which is why the L40S and RTX 5090 plans prefer replicas to tensor parallelism where the memory allows.
- **Price** multiplies the GPU count for the cost lines.

## Comparing

```
$ vitalops plan llama-70b --all-gpus -q
```

prints one row per GPU for the same scenario, with the layout, per-stream and aggregate rates, TTFT and cost per million output tokens; the Jetson is left out. A GPU that cannot hold the model with one request's KV cache in a group of up to eight says `does not fit up to TP8`.

## Hardware not in the catalogue

Construct a `GPU` with the datasheet figures and pass it to `vitalops.plan`; see [Python API](#python-api). To add it to the catalogue for everyone, see [Contributing](#contributing): an entry in `vitalops/hw.py` with a link to the datasheet.
