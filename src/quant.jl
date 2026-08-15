# quant.jl — Per-token symmetric INT8 index (paper §4.3 deferred dequant).
#
# ⟨q_t, d_u⟩ = qscale[t] * dscale[u] * ⟨qcode_t, dcode_u⟩_Int32
# Scales are applied once per output cell, not per embedding element.
# Forward-only: there is no INT8 ChainRules pullback.

"""INT8 token embeddings plus per-token absmax scales (`codes ./ 127`)."""
struct Int8Index{C, S}
    codes::C
    scales::S
end

int8_scale_eps(::Type{T}) where {T<:AbstractFloat} = T(1.0f-8)
const INT8_ABSMAX = 127

function token_absmax(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    vec(maximum(abs, x; dims = 1))
end

function token_absmax(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    dropdims(maximum(abs, x; dims = 1); dims = 1)
end

"""
    quantize_int8_symmetric(x) -> Int8Index

Per-token symmetric INT8 quantization. `x` is `(dim, T)` or `(dim, T, B)`.
"""
function quantize_int8_symmetric(x::AbstractArray{T}) where {T<:AbstractFloat}
    am = token_absmax(x)
    scales = max.(am, int8_scale_eps(T)) ./ T(INT8_ABSMAX)
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

function int8_pair_forward_host(qc::AbstractMatrix{Int8}, qs::AbstractVector{T},
                                dc::AbstractMatrix{Int8}, ds::AbstractVector{T},
                                qmask::AbstractVector{Bool},
                                dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    dim, Tq = size(qc)
    Td = size(dc, 2)
    size(dc, 1) == dim || throw(DimensionMismatch("feature dim"))
    args = zeros(Int32, Tq)
    mx = zeros(T, Tq)
    @inbounds for t in 1:Tq
        qmask[t] || continue
        qt = T(qs[t])
        for u in 1:Td
            dmask[u] || continue
            acc = Int32(0)
            for k in 1:dim
                acc += Int32(qc[k, t]) * Int32(dc[k, u])
            end
            s = qt * T(ds[u]) * T(acc)
            if args[t] == Int32(0) || s > mx[t]
                mx[t] = s
                args[t] = Int32(u)
            end
        end
    end
    score = zero(T)
    @inbounds for t in 1:Tq
        qmask[t] && args[t] > 0 && (score += mx[t])
    end
    score, args
end

function int8_pair_forward_ka(backend::Backend, qc, qs, dc, ds, qmask, dmask)
    Tq = size(qc, 2)
    args = zeros_like(qs, Int32, Tq)
    partial = zeros_like(qs)
    launch!(int8_pair_token_kernel!, backend, Tq, args, partial, qc, qs, dc, ds, qmask, dmask)
    score = sum_length1(backend, partial)
    finish!(backend)
    score, args
end

function int8_pair_forward_ka(backend::GPU, qc, qs, dc, ds, qmask, dmask)
    Tq = size(qc, 2)
    args = zeros_like(qs, Int32, Tq)
    partial = zeros_like(qs)
    launch_int8_pair_scan!(backend, args, partial, qc, qs, dc, ds, qmask, dmask)
    score = sum_length1(backend, partial)
    finish!(backend)
    score, args
end

launch_int8_pair_scan!(backend::GPU, args, partial, qc, qs, dc, ds, qmask, dmask) =
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
    Tq, B = size(Qc, 2), size(Qc, 3)
    scores = zeros(T, B)
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
    Tq, B = size(Qc, 2), size(Qc, 3)
    args = zeros_like(Qs, Int32, Tq, B)
    partial = zeros_like(Qs, T, Tq, B)
    launch!(int8_paired_token_kernel!, backend, (Tq, B), args, partial,
            Qc, Qs, Dc, Ds, qmask, dmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function int8_paired_forward_ka(backend::GPU, Qc, Qs::AbstractMatrix{T},
                                Dc, Ds, qmask, dmask) where {T}
    Tq, B = size(Qc, 2), size(Qc, 3)
    args = zeros_like(Qs, Int32, Tq, B)
    partial = zeros_like(Qs, T, Tq, B)
    launch_int8_paired_scan!(backend, args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

launch_int8_paired_scan!(backend::GPU, args, partial, Qc, Qs, Dc, Ds, qmask, dmask) =
    launch_grouped!(int8_paired_tile_kernel!, backend,
                    query_tile_group((size(Qc, 2), size(Qc, 3))),
                    (size(Qc, 2), size(Qc, 3)), args, partial, Qc, Qs, Dc, Ds, qmask, dmask)

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
