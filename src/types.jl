# types.jl — Configuration, layout markers, ∇D strategies.

"""
Abstract ∇D aggregation strategy (paper §4.2).

Concrete singletons: [`AtomicUnified`](@ref), [`InvGrid`](@ref).
"""
abstract type BackwardStrategy end

"""Source-parallel ``∇D`` scatter (atomics on GPU; sequential on CPU)."""
struct AtomicUnified <: BackwardStrategy end

"""Destination-owned ``∇D`` via inverse-grid CSR (paper Algorithm 3) on every KA backend."""
struct InvGrid <: BackwardStrategy end

"""
    MaxSim{T}(; neg, normalize, backward)
    MaxSim(::Type{T}; ...)

Callable MaxSim scorer. `T` is the feature / score eltype.

`neg` fills **invalid candidate indices** only. It is not a clamp on token
similarities — MaxSim always uses the true max over valid document tokens.
Empty / fully-masked documents contribute `0`, not `neg`.
"""
struct MaxSim{T<:AbstractFloat, B<:BackwardStrategy}
    neg::T
    normalize::Bool
    backward::B
end

coerce_backward(b::BackwardStrategy) = b
coerce_backward(::Type{B}) where {B<:BackwardStrategy} = B()

function MaxSim{T}(neg::Real, normalize::Bool, backward) where {T<:AbstractFloat}
    b = coerce_backward(backward)
    MaxSim{T, typeof(b)}(T(neg), normalize, b)
end

function MaxSim{T}(; neg::Real = T(-1.0f4), normalize::Bool = false,
                   backward = AtomicUnified()) where {T<:AbstractFloat}
    MaxSim{T}(neg, normalize, backward)
end

function MaxSim(::Type{T}; kwargs...) where {T<:AbstractFloat}
    MaxSim{T}(; kwargs...)
end

# Default `MaxSim(...)` keeps Float32 for ergonomics; prefer `MaxSim(T; ...)` with features.
MaxSim(; kwargs...) = MaxSim{Float32}(; kwargs...)

"""Marker: ``S[j,i] = MaxSim(Q[:,:,i], D[:,:,j])`` → `(Bd, Bq)`."""
struct InBatch end
