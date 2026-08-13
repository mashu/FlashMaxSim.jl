# colocation.jl — Same-backend contract for features and masks.

"""All-true mask on the same storage as `prototype`."""
function true_mask(prototype::AbstractArray, dims::Integer...)
    fill!(similar(prototype, Bool, map(Int, dims)...), true)
end

"""KA backend used for colocation checks (BitArrays are host-only)."""
array_backend(x::BitArray) = KernelAbstractions.CPU()
array_backend(x::SubArray) = array_backend(parent(x))
array_backend(x::AbstractArray) = get_backend(x)

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
