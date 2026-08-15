# quant.jl — Per-token symmetric INT8 index (paper §4.3 deferred dequant).
#
# ⟨q_t, d_u⟩ = qscale[t] * dscale[u] * ⟨qcode_t, dcode_u⟩_Int32
# Scales are applied once per output cell, not per embedding element.
# Forward-only: there is no INT8 ChainRules pullback.
#
# Int32 accumulation of 127 × 127 × dim overflows at dim ≈ 133_000.
# Safe for any realistic embedding; the bound is not checked at runtime.

"""INT8 token embeddings plus per-token absmax scales (`codes ./ 127`)."""
struct Int8Index{C, S}
    codes::C
    scales::S
end

function Base.show(io::IO, idx::Int8Index)
    print(io, "Int8Index(codes=", size(idx.codes), ", scales=", size(idx.scales),
          ", eltype=", eltype(idx.scales), ")")
end

int8_scale_eps(::Type{T}) where {T<:AbstractFloat} = T(1.0f-8)
int8_scale_floor(::Type{T}) where {T<:AbstractFloat} = floatmin(T)
const INT8_ABSMAX = 127

function token_absmax(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    size(x, 1) == 0 && return zeros_like(x, T, size(x, 2))
    vec(maximum(abs, x; dims = 1))
end

function token_absmax(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    size(x, 1) == 0 && return zeros_like(x, T, size(x, 2), size(x, 3))
    dropdims(maximum(abs, x; dims = 1); dims = 1)
end

"""
    quantize_int8_symmetric(x) -> Int8Index

Per-token symmetric INT8 quantization. `x` is `(dim, T)` or `(dim, T, B)`.
Scales are floored at `floatmin(T)` so Float16 never produces a zero scale
(which would `InexactError` in `round(Int8, …)`).
"""
function quantize_int8_symmetric(x::AbstractArray{T}) where {T<:AbstractFloat}
    am = token_absmax(x)
    raw = max.(am, int8_scale_eps(T)) ./ T(INT8_ABSMAX)
    scales = max.(raw, int8_scale_floor(T))
    codes = round.(Int8, clamp.(x ./ reshape_scales(scales, x),
                                -T(INT8_ABSMAX), T(INT8_ABSMAX)))
    Int8Index(codes, scales)
end

reshape_scales(scales::AbstractVector, x::AbstractMatrix) = reshape(scales, 1, :)
reshape_scales(scales::AbstractMatrix, x::AbstractArray{<:Any,3}) =
    reshape(scales, 1, size(x, 2), size(x, 3))

function dequant_int8(idx::Int8Index{<:AbstractArray, <:AbstractArray{T}}) where {T}
    T.(idx.codes) .* reshape_scales(idx.scales, idx.codes)
end

Adapt.adapt_structure(to, idx::Int8Index) =
    Int8Index(adapt(to, idx.codes), adapt(to, idx.scales))

function require_int8_index(d::Int8Index{<:AbstractMatrix})
    size(d.codes, 2) == length(d.scales) || throw(DimensionMismatch("Int8Index token count"))
    nothing
end

function require_int8_index(d::Int8Index{<:AbstractArray{<:Any,3}})
    size(d.codes, 2) == size(d.scales, 1) || throw(DimensionMismatch("Int8Index token count"))
    size(d.codes, 3) == size(d.scales, 2) || throw(DimensionMismatch("Int8Index batch"))
    nothing
end

function require_int8_pair_shapes(qc, qs, dc, ds, qmask, dmask)
    dim, Tq = size(qc)
    Td = size(dc, 2)
    size(dc, 1) == dim || throw(DimensionMismatch("feature dim"))
    length(qs) == Tq || throw(DimensionMismatch("query scales"))
    length(ds) == Td || throw(DimensionMismatch("document scales"))
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    length(dmask) == Td || throw(DimensionMismatch("dmask"))
    dim, Tq, Td
end

function require_int8_paired_shapes(Qc, Qs, Dc, Ds, qmask, dmask)
    dim, Tq, B = size(Qc)
    Td = size(Dc, 2)
    size(Dc, 1) == dim || throw(DimensionMismatch("feature dim"))
    size(Dc, 3) == B || throw(DimensionMismatch("batch"))
    size(Qs) == (Tq, B) || throw(DimensionMismatch("query scales"))
    size(Ds) == (Td, B) || throw(DimensionMismatch("document scales"))
    size(qmask) == (Tq, B) || throw(DimensionMismatch("qmask"))
    size(dmask) == (Td, B) || throw(DimensionMismatch("dmask"))
    dim, Tq, Td, B
end

function int8_pair_forward_host(qc::AbstractMatrix{Int8}, qs::AbstractVector{T},
                                dc::AbstractMatrix{Int8}, ds::AbstractVector{T},
                                qmask::AbstractVector{Bool},
                                dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    _, Tq, Td = require_int8_pair_shapes(qc, qs, dc, ds, qmask, dmask)
    A = score_eltype(T)
    args = zeros(Int32, Tq)
    mx = zeros(A, Tq)
    @inbounds for t in 1:Tq
        qmask[t] || continue
        qt = A(qs[t])
        for u in 1:Td
            dmask[u] || continue
            acc = Int32(0)
            for k in 1:size(qc, 1)
                acc += Int32(qc[k, t]) * Int32(dc[k, u])
            end
            s = qt * A(ds[u]) * A(acc)
            if args[t] == Int32(0) || s > mx[t]
                mx[t] = s
                args[t] = Int32(u)
            end
        end
    end
    score = zero(A)
    @inbounds for t in 1:Tq
        qmask[t] && args[t] > 0 && (score += mx[t])
    end
    score, args
end

function int8_pair_forward_ka(backend::Backend, qc, qs, dc, ds, qmask, dmask)
    _, Tq, _ = require_int8_pair_shapes(qc, qs, dc, ds, qmask, dmask)
    T = eltype(qs)
    A = dot_accum(T)
    args = zeros_like(qs, Int32, Tq)
    partial = zeros_like(qs, A, Tq)
    launch_int8_pair_scan!(backend, args, partial, qc, qs, dc, ds, qmask, dmask)
    score = sum_length1(backend, partial)
    finish!(backend)
    score, args
end

function launch_int8_pair_scan!(backend::Backend, args, partial, qc, qs, dc, ds, qmask, dmask;
                                force_tiles::Bool = false)
    if force_tiles
        launch_grouped!(int8_pair_tile_kernel!, backend, query_tile_group(size(qc, 2)),
                        size(qc, 2), args, partial, qc, qs, dc, ds, qmask, dmask)
    else
        launch!(int8_pair_token_kernel!, backend, size(qc, 2),
                args, partial, qc, qs, dc, ds, qmask, dmask)
    end
    nothing
end

launch_int8_pair_scan!(backend::GPU, args, partial, qc, qs, dc, ds, qmask, dmask;
                       force_tiles::Bool = true) =
    launch_grouped!(int8_pair_tile_kernel!, backend, query_tile_group(size(qc, 2)),
                    size(qc, 2), args, partial, qc, qs, dc, ds, qmask, dmask)

function int8_pair_forward(q::AbstractMatrix{T}, d::Int8Index,
                           qmask::AbstractVector{Bool},
                           dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    qq = quantize_int8_symmetric(q)
    require_int8_index(d)
    require_colocated(q, qq.codes, qq.scales, d.codes, d.scales, qmask, dmask)
    int8_pair_forward(array_backend(q), qq, d, qmask, dmask)
end

int8_pair_forward(::CPU, qq::Int8Index, d::Int8Index, qmask, dmask) =
    int8_pair_forward_host(qq.codes, qq.scales, d.codes, d.scales, qmask, dmask)

int8_pair_forward(::Backend, qq::Int8Index, d::Int8Index, qmask, dmask) =
    int8_pair_forward_ka(array_backend(qq.codes), qq.codes, qq.scales, d.codes, d.scales,
                         qmask, dmask)

function int8_paired_forward_host(Qc, Qs::AbstractMatrix{T}, Dc, Ds, qmask, dmask) where {T}
    _, Tq, _, B = require_int8_paired_shapes(Qc, Qs, Dc, Ds, qmask, dmask)
    A = score_eltype(T)
    scores = zeros(A, B)
    args = zeros(Int32, Tq, B)
    @inbounds for b in 1:B
        s, a = int8_pair_forward_host(view(Qc, :, :, b), view(Qs, :, b),
                                      view(Dc, :, :, b), view(Ds, :, b),
                                      view(qmask, :, b), view(dmask, :, b))
        scores[b] = s
        args[:, b] .= a
    end
    scores, args
end

function int8_paired_forward_ka(backend::Backend, Qc, Qs::AbstractMatrix{T},
                                Dc, Ds, qmask, dmask) where {T}
    _, Tq, _, B = require_int8_paired_shapes(Qc, Qs, Dc, Ds, qmask, dmask)
    A = dot_accum(T)
    args = zeros_like(Qs, Int32, Tq, B)
    partial = zeros_like(Qs, A, Tq, B)
    launch_int8_paired_scan!(backend, args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function launch_int8_paired_scan!(backend::Backend, args, partial, Qc, Qs, Dc, Ds, qmask, dmask;
                                  force_tiles::Bool = false)
    nd = (size(Qc, 2), size(Qc, 3))
    if force_tiles
        launch_grouped!(int8_paired_tile_kernel!, backend, query_tile_group(nd), nd,
                        args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    else
        launch!(int8_paired_token_kernel!, backend, nd,
                args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    end
    nothing
end

function launch_int8_paired_scan!(backend::GPU, args, partial, Qc, Qs, Dc, Ds, qmask, dmask;
                                  force_tiles::Bool = true)
    nd = (size(Qc, 2), size(Qc, 3))
    launch_grouped!(int8_paired_tile_kernel!, backend, query_tile_group(nd), nd,
                    args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
end

function int8_paired_forward(Q::AbstractArray{T,3}, D::Int8Index,
                             qmask::AbstractMatrix{Bool},
                             dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    QQ = quantize_int8_symmetric(Q)
    require_int8_index(D)
    require_colocated(Q, QQ.codes, QQ.scales, D.codes, D.scales, qmask, dmask)
    int8_paired_forward(array_backend(Q), QQ, D, qmask, dmask)
end

int8_paired_forward(::CPU, QQ::Int8Index, D::Int8Index, qmask, dmask) =
    int8_paired_forward_host(QQ.codes, QQ.scales, D.codes, D.scales, qmask, dmask)

int8_paired_forward(::Backend, QQ::Int8Index, D::Int8Index, qmask, dmask) =
    int8_paired_forward_ka(array_backend(QQ.codes), QQ.codes, QQ.scales,
                           D.codes, D.scales, qmask, dmask)

ChainRulesCore.@non_differentiable quantize_int8_symmetric(::AbstractArray)
