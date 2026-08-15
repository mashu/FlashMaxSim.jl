# forward_pair.jl — Fused single-pair MaxSim (paper Alg. 1).
#
# Host backend uses BLAS GEMM over document tiles (`DOC_TILE × Tq` scratch).
# GPU backends stream Q/D tiles through SRAM (paper Alg. 1); CPU KA uses a
# scalar token scan. Scores stay on-device (length-1 buffer).
# `neg` is unused in the pair scan — it is a candidate-index sentinel only.
#
# Dispatch is on the *backend*, not on the concrete array type: a `view`,
# `Adjoint` or `PermutedDimsArray` over host memory must take the tiled BLAS
# path, not the scalar KA loop.

const DOC_TILE = 64

function require_pair_shapes(q, d, qmask, dmask)
    dim, Tq = size(q)
    Td = size(d, 2)
    size(d, 1) == dim || throw(DimensionMismatch("feature dim"))
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    length(dmask) == Td || throw(DimensionMismatch("dmask"))
    dim, Tq, Td
end

"""Allocate host scratch for tiled pair GEMM (`DOC_TILE × Tq`, clipped to `Td`).

Scratch uses [`dot_accum`](@ref)`(T)` so Float16 host GEMM runs in FP32 BLAS."""
function pair_host_scratch(::Type{T}, Tq::Int, Td::Int) where {T}
    A = dot_accum(T)
    tile = min(DOC_TILE, max(Td, 1))
    Matrix{A}(undef, tile, Tq), zeros(A, Tq), zeros(Int32, Tq)
end

"""Keep views when features already match the GEMM eltype."""
host_gemm_query(q::AbstractMatrix{A}, ::Type{A}) where {A} = q
host_gemm_query(q::AbstractMatrix, ::Type{A}) where {A} = convert(Matrix{A}, q)

host_gemm_doc_tile(d::AbstractMatrix{A}, u::Int, u_end::Int, ::Type{A}) where {A} =
    view(d, :, u:u_end)
host_gemm_doc_tile(d::AbstractMatrix, u::Int, u_end::Int, ::Type{A}) where {A} =
    convert(Matrix{A}, view(d, :, u:u_end))

host_gemm_batch(Q::AbstractArray{A,3}, ::Type{A}) where {A} = Q
host_gemm_batch(Q::AbstractArray, ::Type{A}) where {A} =
    convert(typeof(similar(Q, A)), Q)

function pair_forward_host(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                           qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                           neg::T) where {T<:AbstractFloat}
    _, Tq, Td = require_pair_shapes(q, d, qmask, dmask)
    Stile, mx, argmax_u = pair_host_scratch(T, Tq, Td)
    pair_forward_host!(argmax_u, mx, Stile, q, d, qmask, dmask, neg)
end

"""In-place host pair forward using caller-owned `Stile`, `mx`, `argmax_u` scratch.

The last `::T` argument is the ignored `neg` candidate-index sentinel.
`Stile` / `mx` are in [`dot_accum`](@ref)`(T)` (FP32 for Float16 features).
The inner GEMM and the `Σ_t` reduction run in that accumulate type; the
returned score stays in [`score_eltype`](@ref)`(T)` (Float32 for Float16 features)."""
function pair_forward_host!(argmax_u::AbstractVector{Int32}, mx::AbstractVector{A},
                            Stile::AbstractMatrix{A},
                            q::AbstractMatrix{T}, d::AbstractMatrix{T},
                            qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                            ::T) where {T<:AbstractFloat, A<:AbstractFloat}
    _, Tq, Td = require_pair_shapes(q, d, qmask, dmask)
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u"))
    length(mx) == Tq || throw(DimensionMismatch("mx"))
    size(Stile, 2) == Tq || throw(DimensionMismatch("Stile"))
    fill!(argmax_u, Int32(0))
    fill!(mx, zero(A))
    qA = host_gemm_query(q, A)
    u = 1
    while u <= Td
        u_end = min(u + DOC_TILE - 1, Td)
        w = u_end - u + 1
        dA = host_gemm_doc_tile(d, u, u_end, A)
        mul!(view(Stile, 1:w, :), transpose(dA), qA)
        @inbounds for t in 1:Tq
            qmask[t] || continue
            for uu in 1:w
                dmask[u + uu - 1] || continue
                s = Stile[uu, t]
                if argmax_u[t] == Int32(0) || s > mx[t]
                    mx[t] = s
                    argmax_u[t] = Int32(u + uu - 1)
                end
            end
        end
        u = u_end + 1
    end
    score = zero(A)
    @inbounds for t in 1:Tq
        qmask[t] && argmax_u[t] > 0 && (score += mx[t])
    end
    score, argmax_u
end

function pair_forward_ka(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                         qmask::AbstractVector{Bool},
                         dmask::AbstractVector{Bool},
                         neg::T) where {T<:AbstractFloat}
    pair_forward_ka(array_backend(q), q, d, qmask, dmask, neg)
end

function pair_forward_ka(backend::Backend, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                         qmask::AbstractVector{Bool},
                         dmask::AbstractVector{Bool},
                         ::T) where {T<:AbstractFloat}
    _, Tq, _ = require_pair_shapes(q, d, qmask, dmask)
    A = dot_accum(T)
    argmax_u = zeros_like(q, Int32, Tq)
    partial = zeros_like(q, A, Tq)
    launch_pair_scan!(backend, argmax_u, partial, q, d, qmask, dmask)
    score = sum_length1(backend, partial)   # length-1, same backend — no host sum
    finish!(backend)
    score, argmax_u
end

function launch_pair_scan!(backend::Backend, argmax_u, partial, q, d, qmask, dmask;
                           force_tiles::Bool = false)
    if force_tiles
        launch_grouped!(pair_tile_kernel!, backend, query_tile_group(size(q, 2)),
                        size(q, 2), argmax_u, partial, q, d, qmask, dmask)
    else
        launch!(pair_token_kernel!, backend, size(q, 2), argmax_u, partial, q, d, qmask, dmask)
    end
    nothing
end

launch_pair_scan!(backend::GPU, argmax_u, partial, q, d, qmask, dmask;
                  force_tiles::Bool = true) =
    launch_grouped!(pair_tile_kernel!, backend, query_tile_group(size(q, 2)),
                    size(q, 2), argmax_u, partial, q, d, qmask, dmask)

function pair_forward(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                      qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                      neg::T) where {T<:AbstractFloat}
    require_colocated(q, d, qmask, dmask)
    pair_forward(array_backend(q), q, d, qmask, dmask, neg)
end

pair_forward(::CPU, q::AbstractMatrix{T}, d::AbstractMatrix{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_host(q, d, qmask, dmask, neg)

pair_forward(::Backend, q::AbstractMatrix{T}, d::AbstractMatrix{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_ka(q, d, qmask, dmask, neg)

# ---- pair score normalize (host scalar vs device length-1) -------------------

pair_finalize(score::T, qmask, ::Val{false}) where {T<:AbstractFloat} = score, one(T)

function pair_finalize(score::T, qmask, ::Val{true}) where {T<:AbstractFloat}
    inv_n = one(T) / T(query_count(qmask))
    score * inv_n, inv_n
end

function pair_finalize(score::AbstractVector{T}, qmask, ::Val{false}) where {T<:AbstractFloat}
    score, ones_like(score)
end

function pair_finalize(score::AbstractVector{T}, qmask, ::Val{true}) where {T<:AbstractFloat}
    backend = array_backend(score)
    n = count_true_length1(backend, qmask, T)
    inv_n = one(T) ./ n
    finish!(backend)
    score .* inv_n, inv_n
end

pair_finalize(score, qmask, normalize::Bool) =
    pair_finalize(score, qmask, Val(normalize))
