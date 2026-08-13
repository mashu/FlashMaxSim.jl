# FlashMaxSim.jl — Julia port of Flash-MaxSim (Pony et al., 2026).
#
# Exact ColBERT MaxSim without materializing the query×doc similarity tensor
# (paper Alg. 1). Training backward saves only argmax indices and aggregates
# with atomic-unified or inverse-grid CSR scatter (paper Alg. 2 / §4.2).
#
# Backend-agnostic via KernelAbstractions (CPU default; CUDA/ROCm/Metal when
# the corresponding array type is loaded). Batch layouts compose pair forward;
# sparse pullbacks currently aggregate on the host.

module FlashMaxSim

using ChainRulesCore
using KernelAbstractions
using LinearAlgebra

export MaxSim, InBatch, BackwardStrategy, AtomicUnified, InvGrid
export maxsim, maxsim_dense

include("types.jl")
include("colocation.jl")
include("storage.jl")
include("normalize.jl")
include("dense.jl")
include("invgrid.jl")
include("accumulate.jl")
include("forward_pair.jl")
include("forward_layouts.jl")
include("backward.jl")
include("dispatch.jl")
include("rrule.jl")

end # module
