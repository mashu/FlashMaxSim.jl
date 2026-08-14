# colocation.jl — Same-backend contract for features and masks.

"""All-true mask on the same storage as `prototype`."""
function true_mask(prototype::AbstractArray, dims::Integer...)
    fill!(similar(prototype, Bool, map(Int, dims)...), true)
end

"""
KA backend used for colocation checks.

`BitArray` is host-only *and* is its own `parent`, so it must terminate the
wrapper walk: KernelAbstractions' fallback is `get_backend(A) =
get_backend(parent(A))`, which recurses forever on any array whose innermost
parent is a `BitArray`. Walking `parent` here with an explicit fixpoint check
covers `SubArray`, `Adjoint`, `PermutedDimsArray`, `ReshapedArray` and any
other wrapper uniformly.
"""
array_backend(::BitArray) = KernelAbstractions.CPU()

function array_backend(x::AbstractArray)
    p = parent(x)
    p === x && return get_backend(x)
    array_backend(p)
end

"""Require features and masks to share one KernelAbstractions backend."""
function require_colocated(xs::AbstractArray...)
    isempty(xs) && return nothing
    b0 = array_backend(first(xs))
    for x in xs
        array_backend(x) === b0 || throw(ArgumentError(
            "FlashMaxSim requires colocated arrays (same KernelAbstractions backend) for features and masks"))
    end
    nothing
end

ChainRulesCore.@non_differentiable true_mask(::AbstractArray, ::Integer...)
ChainRulesCore.@non_differentiable require_colocated(::AbstractArray...)
