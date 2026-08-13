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

**without materializing** the query-token × document-token similarity matrix
(paper Algorithm 1). Training keeps only argmax indices and aggregates
gradients with atomic-unified or inverse-grid CSR scatter (paper §4.2 /
Algorithm 2). Exact up to floating-point evaluation order (paper Prop. 1).

Pair forward uses [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
(CPU / CUDA / ROCm / Metal from the array backend). Batch layouts compose
pair scoring; sparse pullbacks currently aggregate on the host.

**Contract:** features and masks must be colocated (same KA backend). Host
`BitArray` masks are allowed with `Array` features; GPU features need
device masks (`true_mask` / `similar`). Candidate index matrices are host
`AbstractMatrix{<:Integer}`.

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
| Alg. 1 fused forward (no `S` in HBM) | `pair_forward` (host tiled / KA token kernel) |
| Prop. 1 exactness | tests vs `maxsim_dense` |
| §4.2 Eq. 2–3 sparse grads | ChainRules `rrule` |
| Atomic-unified `∇D` | `backward = AtomicUnified()` (host scatter today) |
| Inverse-grid CSR `∇D` | `backward = InvGrid()` |
| In-batch / candidate scoring | `InBatch`, index-matrix layouts |

Not ported: INT8 tensor-core path, split-`d`, Chamfer, device atomics / device
CSR pullback. Exact FP MaxSim scores and sparse AD structure match the paper
operator used for ColBERT training / reranking.

## Design

| Piece | Role |
|:------|:-----|
| `maxsim` / `MaxSim` | One verb + callable config; layout via methods |
| `BackwardStrategy` | `AtomicUnified` / `InvGrid` via dispatch |
| KernelAbstractions | Pair kernel backend from array type |
| `maxsim_dense` | Materializing reference for correctness tests |

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
