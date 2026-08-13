# types.jl — Configuration, layout markers, ∇D strategies.

"""
Abstract ∇D aggregation strategy (paper §4.2).

Concrete singletons: [`AtomicUnified`](@ref), [`InvGrid`](@ref).
"""
abstract type BackwardStrategy end

"""Destination scatter for ``∇D`` (host sequential; paper atomic-unified)."""
struct AtomicUnified <: BackwardStrategy end

"""Inverse-grid CSR, destination-owned, atomic-free (paper §4.2.2)."""
struct InvGrid <: BackwardStrategy end

"""
    MaxSim{T}(; neg, normalize, backward)
    MaxSim(::Type{T}; ...)

Callable MaxSim scorer. `T` is the feature / score eltype.
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
