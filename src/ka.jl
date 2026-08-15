# ka.jl — KernelAbstractions launch and same-backend allocation.
#
# Kernels enqueue in order on a backend. Do not synchronize after every launch.
# `finish!` syncs only on CPU (KA CPU is async); GPU relies on stream ordering
# so device→device work stays off the host.

range_count(n::Integer) = Int(n)
range_count(n::Tuple) = prod(n)

"""Enqueue a kernel. No device synchronize — see [`finish!`](@ref)."""
function launch!(kernel, backend, ndrange, args...)
    iszero(range_count(ndrange)) && return nothing
    kernel(backend)(args...; ndrange)
    nothing
end

"""Enqueue with a static workgroup size (SRAM-tiled GPU scans)."""
function launch_grouped!(kernel, backend, group, ndrange, args...)
    iszero(range_count(ndrange)) && return nothing
    kernel(backend, group)(args...; ndrange)
    nothing
end

"""Block until queued work finishes (CPU KA only; no-op on GPU backends)."""
finish!(backend::CPU) = KernelAbstractions.synchronize(backend)
finish!(::Backend) = nothing

zeros_like(x::AbstractArray{T}) where {T} = fill!(similar(x), zero(T))
zeros_like(x::AbstractArray, ::Type{T}, dims::Integer...) where {T} =
    fill!(similar(x, T, dims...), zero(T))

ones_like(x::AbstractArray{T}) where {T} = fill!(similar(x), one(T))
ones_like(x::AbstractArray, ::Type{T}, dims::Integer...) where {T} =
    fill!(similar(x, T, dims...), one(T))

"""
Require `idxs` already on `prototype`'s backend.

Refuses host→device copies in the hot path — pass `CuArray(idxs)` (or equivalent).
"""
function indices_on(prototype::AbstractArray, idxs::AbstractMatrix{<:Integer})
    array_backend(idxs) === array_backend(prototype) && return idxs
    throw(ArgumentError(
        "FlashMaxSim requires idxs on the same KernelAbstractions backend as features " *
        "(e.g. CuArray(idxs)); refusing a host→device copy in the hot path"))
end

"""On-device `sum(x)` into a length-1 buffer (no host materialization)."""
function sum_length1(backend, x::AbstractVector{T}) where {T}
    out = zeros_like(x, T, 1)
    launch!(reduce_sum1_kernel!, backend, 1, out, x)
    out
end

"""On-device `max(count(mask), 1)` as length-1 `T` buffer."""
function count_true_length1(backend, mask::AbstractVector{Bool}, ::Type{T}) where {T}
    out = zeros_like(mask, T, 1)
    launch!(count_true1_kernel!, backend, 1, out, mask)
    out
end

query_count(qmask::AbstractVector{Bool}) = max(Int(sum(qmask)), 1)

"""Convert a score buffer to the feature eltype without a host round-trip."""
convert_scores(::Type{T}, s::AbstractArray{T}) where {T} = s
convert_scores(::Type{T}, s::AbstractArray) where {T} = T.(s)
convert_scores(::Type{T}, s::T) where {T} = s
convert_scores(::Type{T}, s::Number) where {T} = T(s)

"""Whether this backend launches WMMA tensor-core scans for `T` features."""
tensor_cores_active(::Backend, ::Type) = false
