from algorithm import parallelize
from layout import Layout, LayoutTensor
from .amx import tile_zero, tile_load, tile_store, tile_dp, load_amx_tilecfg, release_amx_tilecfg

fn matmul_amx_uint8_int8_blocked[
    M: Int,
    N: Int,
    K: Int,
    PACK_N_BLOCK: Int,
    PACK_K_BLOCK: Int,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
](
    A: LayoutTensor[DType.uint8, A_layout, _],
    B: LayoutTensor[DType.int8, B_layout, _],
    C: LayoutTensor[mut=True, DType.int32, C_layout, _],
    n_offset: Int = 0,
    n_size: Int = N,
):
    alias TILE_M = 16
    alias TILE_N = 16
    alias TILE_K = 64
    alias M_STEP = 32
    alias N_STEP = 32
    alias K_STEP = 64

    constrained[M % M_STEP == 0, "M must be divisible by M_STEP"]()
    constrained[N % N_STEP == 0, "N must be divisible by N_STEP"]()
    constrained[K % K_STEP == 0, "K must be divisible by K_STEP"]()
    constrained[M_STEP == TILE_M * 2, "M_STEP must be 2*TILE_M"]()
    constrained[N_STEP == TILE_N * 2, "N_STEP must be 2*TILE_N"]()
    constrained[K_STEP == TILE_K, "K_STEP must equal TILE_K"]()


    for k_block_begin in range(0, K, PACK_K_BLOCK):
        var k_block_size = min(PACK_K_BLOCK, K - k_block_begin)

        for m_begin in range(0, M, M_STEP):
            for n_begin in range(n_offset, n_offset + n_size, N_STEP):
                if k_block_begin == 0:
                    tile_zero[4]()
                    tile_zero[5]()
                    tile_zero[6]()
                    tile_zero[7]()
                else:
                    var c00_ptr = C.ptr.offset(m_begin * N + n_begin)
                    var c01_ptr = C.ptr.offset(m_begin * N + n_begin + TILE_N)
                    var c10_ptr = C.ptr.offset((m_begin + TILE_M) * N + n_begin)
                    var c11_ptr = C.ptr.offset((m_begin + TILE_M) * N + n_begin + TILE_N)
                    tile_load[4, N * 4](c00_ptr)
                    tile_load[5, N * 4](c01_ptr)
                    tile_load[6, N * 4](c10_ptr)
                    tile_load[7, N * 4](c11_ptr)

                for k_begin in range(0, k_block_size, K_STEP):
                    var k = k_block_begin + k_begin

                    var a0_ptr = A.ptr.offset(m_begin * K + k)
                    var a1_ptr = A.ptr.offset((m_begin + TILE_M) * K + k)

                    var n_block_begin = (n_begin // PACK_N_BLOCK) * PACK_N_BLOCK
                    var n_block_size = min(PACK_N_BLOCK, N - n_block_begin)
                    var n_within_block = n_begin - n_block_begin

                    var k_block_base = (k // PACK_K_BLOCK) * PACK_K_BLOCK
                    var k_block_sz = min(PACK_K_BLOCK, K - k_block_base)
                    var k_within_block = k - k_block_base

                    var b_offset = (n_block_begin * K +
                                   k_block_base * n_block_size +
                                   n_within_block * k_block_sz +
                                   k_within_block * N_STEP)

                    var b0_ptr = B.ptr.offset(b_offset)
                    var b1_ptr = B.ptr.offset(b_offset + TILE_N * K_STEP)

                    tile_load[0, K](a0_ptr)
                    tile_load[1, K](a1_ptr)
                    tile_load[2, K_STEP](b0_ptr)
                    tile_load[3, K_STEP](b1_ptr)

                    tile_dp[4, 0, 2](a0_ptr, b0_ptr)
                    tile_dp[5, 0, 3](a0_ptr, b1_ptr)
                    tile_dp[6, 1, 2](a1_ptr, b0_ptr)
                    tile_dp[7, 1, 3](a1_ptr, b1_ptr)

                var c00_ptr = C.ptr.offset(m_begin * N + n_begin)
                var c01_ptr = C.ptr.offset(m_begin * N + n_begin + TILE_N)
                var c10_ptr = C.ptr.offset((m_begin + TILE_M) * N + n_begin)
                var c11_ptr = C.ptr.offset((m_begin + TILE_M) * N + n_begin + TILE_N)
                tile_store[4, N * 4](c00_ptr)
                tile_store[5, N * 4](c01_ptr)
                tile_store[6, N * 4](c10_ptr)
                tile_store[7, N * 4](c11_ptr)

fn matmul_amx_uint8_int8_blocked_parallel[
    M: Int,
    N: Int,
    K: Int,
    PACK_N_BLOCK: Int,
    PACK_K_BLOCK: Int,
    PARALLEL_N_CHUNK: Int,
    num_workers: Int,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
](
    A: LayoutTensor[DType.uint8, A_layout, _],
    B: LayoutTensor[DType.int8, B_layout, _],
    C: LayoutTensor[mut=True, DType.int32, C_layout, _],
):
    constrained[N % PARALLEL_N_CHUNK == 0, "N must be divisible by PARALLEL_N_CHUNK"]()

    @parameter
    fn worker(thread_id: Int):
        var n_start = thread_id * PARALLEL_N_CHUNK
        var n_end = min(n_start + PARALLEL_N_CHUNK, N)
        var n_slice_size = n_end - n_start

        matmul_amx_uint8_int8_blocked[M, N, K, PACK_N_BLOCK, PACK_K_BLOCK, A_layout, B_layout, C_layout](
            A, B, C, n_start, n_slice_size
        )

    var num_blocks = N // PARALLEL_N_CHUNK
    parallelize[worker](num_blocks, num_workers)
