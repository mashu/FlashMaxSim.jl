# ka.jl — KernelAbstractions launch and same-backend allocation.
#
# Kernels enqueue in order on a backend; do **not** synchronize after every
# launch. Sync only before reading a host-visible value (`sum`, scalar index,
# `Array(...)`) or before returning buffers that a host test will inspect.

"""Enqueue a kernel. No device synchronize — see [`sync!`](@ref)."""
function launch!(kernel, backend, ndrange, args...)
    prod(ndrange) == 0 && return nothing
    kernel(backend)(args...; ndrange)
    nothing
end

"""Block until all work queued on `backend` has finished."""
sync!(backend) = KernelAbstractions.synchronize(backend)

zeros_like(x::AbstractArray{T}) where {T} = fill!(similar(x), zero(T))
zeros_like(x::AbstractArray, ::Type{T}, dims::Integer...) where {T} =
    fill!(similar(x, T, dims...), zero(T))

"""Place `idxs` on `prototype`'s backend (no-op when already colocated)."""
function indices_on(prototype::AbstractArray, idxs::AbstractMatrix{<:Integer})
    array_backend(idxs) === array_backend(prototype) && return idxs
    out = similar(prototype, eltype(idxs), size(idxs))
    copyto!(out, idxs)
    out
end

query_count(qmask::AbstractVector{Bool}) = max(Int(sum(qmask)), 1)
