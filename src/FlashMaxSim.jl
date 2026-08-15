# FlashMaxSim.jl — Julia port of Flash-MaxSim (Pony et al., 2026).
#
# Pair / paired / candidate MaxSim fuse the per-token argmax without storing
# a dense query×doc similarity matrix (paper Alg. 1). GPU scans tile Q/D in
# SRAM; `CuArray{Float16}` uses WMMA tensor cores. In-batch contrastive
# scoring tiles GEMM chunks of `D'Q`. Packed `cu_seqlens` skips padding.
# INT8 indices use deferred dequant (forward-only). Training backward saves
# only argmax indices and aggregates with fused atomic-unified scatter, or
# inverse-grid CSR (paper Alg. 3 / §4.2) built on-device for every KA backend.
#
# Backend-agnostic via KernelAbstractions. Features, masks, scores, argmax,
# and cotangents stay on the same backend — no host round-trips of Q/D.

module FlashMaxSim

using Adapt
using ChainRulesCore
using KernelAbstractions
using LinearAlgebra
import KernelAbstractions: @atomic

export MaxSim, InBatch, BackwardStrategy, AtomicUnified, InvGrid
export maxsim, maxsim_dense
export pack_docs, pack_pairs, PackedSeq, nseq, Int8Index, quantize_int8_symmetric

include("types.jl")
include("colocation.jl")
include("ka.jl")
include("normalize.jl")
include("dense.jl")
include("invgrid.jl")
include("accumulate.jl")
include("kernels_forward.jl")
include("kernels_backward.jl")
include("kernels_packed.jl")
include("kernels_int8.jl")
include("csr_ka.jl")
include("forward_pair.jl")
include("forward_layouts.jl")
include("packed.jl")
include("quant.jl")
include("backward_pair.jl")
include("backward_layouts.jl")
include("dispatch.jl")
include("rrule.jl")

end # module
