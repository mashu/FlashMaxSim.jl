# storage.jl — Host↔device copies for scores / cotangents (no Adapt).

"""Copy `x` onto storage matching `prototype` (identity when both are `Array`)."""
function match_storage(prototype::AbstractArray, x::AbstractArray)
    out = similar(prototype, eltype(x), size(x))
    copyto!(out, x)
    out
end
match_storage(::Array, x::Array) = x

"""Place a host buffer onto `prototype`'s storage."""
upload_like(::Array, x::Array) = x
function upload_like(prototype::AbstractArray, x::Array)
    out = similar(prototype, eltype(x), size(x))
    copyto!(out, x)
    out
end

"""Dense host `Bool` mask for length-norm / CSR bookkeeping."""
host_bool(m::AbstractArray{Bool}) = Array{Bool}(m)
