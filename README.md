# README.md

[![Build Status](https://github.com/mashu/FlashMaxSim.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mashu/FlashMaxSim.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# FlashMaxSim.jl

Julia implementation of **Flash-MaxSim** — IO-aware fused MaxSim for
ColBERT / ColPali late-interaction retrieval
([Pony et al., 2026](https://arxiv.org/abs/2605.29517);
reference code [roipony/flash-maxsim](https://github.com/roipony/flash-maxsim)).

Computes

```text
MaxSim(q, d) = Σ_t  max_u  ⟨q_t, d_u⟩
```

Pair, paired-batch, and candidate scoring fuse the per-token argmax
(paper Algorithm 1) without storing a dense query-token × document-token
matrix. In-batch contrastive scoring (the ColBERT training layout) computes
tiled GEMM chunks of `D'Q` then a light argmax accumulate. Training keeps
only argmax indices and aggregates gradients with a fused atomic-unified
scatter (one launch; Q hoisted), or inverse-grid CSR (paper §4.2 /
Algorithm 3). Exact up to floating-point evaluation order (paper Prop. 1).

Forward and backward run on the **same KernelAbstractions backend** as the
features (CPU `Array` uses BLAS-tiled GEMM; CUDA / ROCm / Metal use SRAM-tiled
kernels). `CuArray{Float16}` pair / paired scans use WMMA tensor cores when
the device is sm ≥ 7.0. Scores, argmax tape, and cotangents stay on-device —
no host round-trips of `Q` / `D`.

**Contract:** features and masks must be colocated (same KA backend). Host
`BitArray` masks are allowed with `Array` features; GPU features need
device masks (`true_mask` / `similar`). Candidate index matrices must already
be on the feature backend.

`neg` fills invalid candidate index slots only. It is not a clamp on token
similarities; empty documents contribute `0`.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/mashu/FlashMaxSim.jl")
```

## Usage

Feature layout is `(dim, T)` / `(dim, T, B)` (tokens as columns).

```julia
using FlashMaxSim

q = randn(Float32, 64, 32)   # query tokens
d = randn(Float32, 64, 48)   # document tokens
s = maxsim(q, d)             # scalar

# masks must be colocated with features (same device / KA backend)
s = maxsim(q, d, trues(32), trues(48); normalize = true)

# callable config — match feature eltype with MaxSim(T; ...)
cfg = MaxSim(Float32; normalize = true, backward = InvGrid())
s = cfg(q, d, trues(32), trues(48))

# paired batch → (B,)
Q, D = randn(Float32, 64, 32, 8), randn(Float32, 64, 48, 8)
scores = cfg(Q, D)

# in-batch contrastive → (Bd, Bq): S[j,i] = MaxSim(Q[:,:,i], D[:,:,j])
S = cfg(Q, D, InBatch())

# candidate set → (C, B); invalid idxs keep `neg`
gallery = randn(Float32, 64, 48, 100)
idxs = rand(1:100, 16, 8)
Sc = cfg(Q, gallery, idxs)

# packed documents (no padding): 1-based cu_seqlens
docs = [randn(Float32, 64, 40), randn(Float32, 64, 12), randn(Float32, 64, 55)]
packed, cu = pack_docs(docs)
s_pack = maxsim(q, packed, cu)          # (B,)

# INT8 index (quantize D once; Q quantized on the fly; forward-only)
d8 = quantize_int8_symmetric(d)
s8 = maxsim(q, d8)
```

Gradients (Zygote / Flux) use a ChainRules `rrule` — sparse argmax tape, no
dense `S`. Dense reference: `maxsim_dense` (tests / oracles).

```julia
using CUDA   # optional — default masks allocate on-device
qg, dg = CuArray(q), CuArray(d)
s = maxsim(qg, dg)
```

## Alignment with the paper

| Paper | This package |
|:------|:-------------|
| Alg. 1 materialized `S` then max | `maxsim_dense` (oracle only) |
| Alg. 2 fused forward (no `S` in HBM) | SRAM-tiled KA kernels; WMMA `tl.dot` equivalent for `CuArray{Float16}` |
| In-batch `D'Q` tiles | `inbatch_forward` materializes GEMM chunks (`INBATCH_TILE_BYTES`), not Alg. 2 |
| Prop. 1 exactness | tests vs `maxsim_dense` |
| §4.2 Eq. 2–3 sparse grads | ChainRules `rrule` on the feature backend |
| Atomic-unified `∇D` | `backward = AtomicUnified()` — fused ∇Q+∇D, one D load, Q hoisted |
| Inverse-grid `∇D` (Alg. 3) | `backward = InvGrid()` — CSR on every KA backend |
| Packed `cu_seqlens` | `pack_docs` / `pack_pairs`; `maxsim(q, packed, cu)` |
| INT8 deferred dequant | `quantize_int8_symmetric` / `Int8Index` (forward-only) |
| In-batch / candidate scoring | `InBatch`, index-matrix layouts |

Not ported: split-`d` for `d>512`, Chamfer, INT8 tensor-core WMMA. Exact FP
MaxSim scores and sparse AD structure match the paper operator used for
ColBERT training / reranking.

## Design

| Piece | Role |
|:------|:-----|
| `maxsim` / `MaxSim` | One verb + callable config; layout via methods |
| `BackwardStrategy` | `AtomicUnified` / `InvGrid` via dispatch |
| KernelAbstractions | Forward + backward from the array backend |
| `maxsim_dense` | Materializing host reference for correctness tests |

Loss functions (InfoNCE, hard-negative mining) belong in the caller.

## Citation

```bibtex
@article{pony2026flashmaxsim,
  title   = {Flash-MaxSim: IO-Aware Fused Kernels for Late-Interaction Retrieval},
  author  = {Pony, Roi and Ezer, Daniel and Goldfarb, Adi Raz and Friedman, Idan
             and Naparstek, Oshri and Barzelay, Udi},
  journal = {arXiv preprint arXiv:2605.29517},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.29517}
}
```
