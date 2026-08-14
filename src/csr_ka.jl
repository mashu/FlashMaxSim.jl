# csr_ka.jl — On-device inverse-grid CSR builders (paper Alg. 3 counting sort).
#
# Host CSR lives in `invgrid.jl`. Kernels live in `kernels_backward.jl`.
# `row_ptr` uses 0-based exclusive ends; `@atomic x += v` returns the new
# value, used as the 1-based `col_idx` write position.

"""Device CSR for one pair: `argmax_u[t] → u`."""
function build_pair_csr(backend, prototype, argmax_u, Td, Tq)
    row_ptr = zeros_like(prototype, Int32, Td + 1)
    launch!(csr_count_pair_kernel!, backend, Tq, row_ptr, argmax_u, Td)
    launch!(csr_prefix_pair_kernel!, backend, 1, row_ptr, Td)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq)
    launch!(csr_fill_pair_kernel!, backend, Tq, col_idx, cursor, argmax_u, Td)
    row_ptr, col_idx
end

function build_paired_csr(backend, prototype, args, Td, Tq, B)
    row_ptr = zeros_like(prototype, Int32, Td + 1, B)
    launch!(csr_count_paired_kernel!, backend, (Tq, B), row_ptr, args, Td)
    launch!(csr_prefix_paired_kernel!, backend, B, row_ptr, Td)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq * B)
    launch!(csr_fill_paired_kernel!, backend, (Tq, B), col_idx, cursor, args, Td, Tq)
    row_ptr, col_idx
end

function build_inbatch_csr(backend, prototype, args, Td, Tq, Bq, Bd)
    row_ptr = zeros_like(prototype, Int32, Td + 1, Bd)
    launch!(csr_count_inbatch_kernel!, backend, (Tq, Bd, Bq), row_ptr, args, Td)
    launch!(csr_prefix_inbatch_kernel!, backend, Bd, row_ptr, Td)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq * Bq * Bd)
    launch!(csr_fill_inbatch_kernel!, backend, (Tq, Bd, Bq),
            col_idx, cursor, args, Td, Tq, Bq)
    row_ptr, col_idx
end

function build_candidates_csr(backend, prototype, idxs, args, Td, N, Tq, C, B)
    n_dest = Td * N
    row_ptr = zeros_like(prototype, Int32, n_dest + 1)
    launch!(csr_count_candidates_kernel!, backend, (Tq, C, B),
            row_ptr, idxs, args, Td, N)
    launch!(csr_prefix_candidates_kernel!, backend, 1, row_ptr, n_dest)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq * C * B)
    launch!(csr_fill_candidates_kernel!, backend, (Tq, C, B),
            col_idx, cursor, idxs, args, Td, N, Tq, C)
    row_ptr, col_idx
end
