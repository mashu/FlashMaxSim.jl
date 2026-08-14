# FlashMaxSim.jl — Julia port of Flash-MaxSim (Pony et al., 2026).
#
# Pair / paired / candidate MaxSim fuse the per-token argmax without storing
# a dense query×doc similarity matrix (paper Alg. 1). In-batch contrastive
# scoring tiles GEMM chunks of `D'Q`. Training backward saves only argmax
# indices and aggregates with atomic-unified scatter, or inverse-grid CSR
# (paper Alg. 3 / §4.2) built on-device for every KA backend.
#
# Backend-agnostic via KernelAbstractions. Features, masks, scores, argmax,
# and cotangents stay on the same backend — no host round-trips of Q/D.

module FlashMaxSim

using ChainRulesCore
using KernelAbstractions
using LinearAlgebra
import KernelAbstractions: @atomic

export MaxSim, InBatch, BackwardStrategy, AtomicUnified, InvGrid
export maxsim, maxsim_dense

include("types.jl")
include("colocation.jl")
include("ka.jl")
include("normalize.jl")
include("dense.jl")
include("invgrid.jl")
include("accumulate.jl")
include("kernels_forward.jl")
include("kernels_backward.jl")
include("csr_ka.jl")
include("forward_pair.jl")
include("forward_layouts.jl")
include("backward_pair.jl")
include("backward_layouts.jl")
include("dispatch.jl")
include("rrule.jl")

end # module
