# backward_pair.jl — Single-pair sparse MaxSim pullback (paper §4.2, Eqs. 2–3).
#
# CPU: sequential gather + [`accumulate_doc!`](@ref).
# Other KA backends: gather/scatter kernels. InvGrid uses device CSR from
# [`build_pair_csr`](@ref). Named `pair_pullback_ka` is reachable in CI on CPU().

function pair_pullback(δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    pair_pullback(array_backend(q), δ, q, d, qmask, argmax_u, mode)
end

function pair_pullback(::CPU, δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    dq = zeros_like(q)
    dd = zeros_like(d)
    δ == zero(T) && return dq, dd
    dim, Tq = size(q)
    Td = size(d, 2)
    length(qmask) == Tq || throw(DimensionMismatch("qmask vs query tokens"))
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    δ_src = zeros(T, Tq)
    @inbounds for t in 1:Tq
        qmask[t] || continue
        u = Int(argmax_u[t])
        (1 <= u <= Td) || continue
        δ_src[t] = δ
        @simd for k in 1:dim
            dq[k, t] += δ * d[k, u]
        end
    end
    accumulate_doc!(mode, dd, q, δ_src, argmax_u)
    dq, dd
end

function pair_pullback(backend::Backend, δ::T, q::AbstractMatrix{T},
                       d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    pair_pullback_ka(backend, δ, q, d, qmask, argmax_u, mode)
end

function pair_pullback_ka(backend, δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                          qmask::AbstractVector{Bool},
                          argmax_u::AbstractVector{<:Integer},
                          mode::BackwardStrategy) where {T<:AbstractFloat}
    dq = zeros_like(q)
    dd = zeros_like(d)
    δ == zero(T) && return dq, dd
    dim, Tq = size(q)
    Td = size(d, 2)
    length(qmask) == Tq || throw(DimensionMismatch("qmask vs query tokens"))
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    launch!(gather_pair_kernel!, backend, (dim, Tq), dq, d, qmask, argmax_u, δ, Td)
    scatter_pair!(mode, backend, dd, q, qmask, argmax_u, δ)
    sync!(backend)
    dq, dd
end

scatter_pair!(::AtomicUnified, backend, dd, q, qmask, argmax_u, δ) =
    launch!(scatter_pair_atomic_kernel!, backend, (size(q, 1), size(q, 2)),
            dd, q, qmask, argmax_u, δ, size(dd, 2))

function scatter_pair!(::InvGrid, backend, dd, q, qmask, argmax_u, δ)
    Td = size(dd, 2)
    Tq = size(q, 2)
    row_ptr, col_idx = build_pair_csr(backend, q, argmax_u, qmask, Td, Tq)
    launch!(scatter_pair_csr_kernel!, backend, (size(dd, 1), Td),
            dd, q, row_ptr, col_idx, δ)
    nothing
end
