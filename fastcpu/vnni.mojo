from memory import ImmutUnsafePointer, MutUnsafePointer
from collections import InlineArray
from .transpose import transpose_16x16

fn pack_b_matrix_vnni_int8_small[K: Int, N: Int](
    src: ImmutUnsafePointer[Int8],
    dst: MutUnsafePointer[Int8],
):
    @parameter
    if K >= 16 and N >= 16 and K % 16 == 0 and N % 16 == 0:
        var buf = InlineArray[SIMD[DType.int8, 16], 16](uninitialized=True)
        for n_tile in range(N // 16):
            for k_tile in range(K // 16):
                var src_tile = src.offset(k_tile * 16 * N + n_tile * 16)
                var dst_tile = dst.offset(n_tile * 16 * K + k_tile * 16)
                transpose_16x16(src_tile, N, dst_tile, K, buf)
    else:
        for k in range(K):
            for n in range(N):
                dst[n * K + k] = src[k * N + n]

fn pack_b_matrix_vnni_int8_large[
    K: Int,
    N: Int,
    N_BLOCK: Int,
    K_BLOCK: Int,
    TILE_K: Int = 64,
    TILE_N: Int = 16,
    N_STEP: Int = 32,
    K_STEP: Int = 64,
](
    src: ImmutUnsafePointer[Int8],
    dst: MutUnsafePointer[Int8],
):
    constrained[N % N_STEP == 0, "N must be divisible by N_STEP"]()
    constrained[K % K_STEP == 0, "K must be divisible by K_STEP"]()
    constrained[N_STEP == TILE_N * 2, "N_STEP must be 2*TILE_N"]()
    constrained[K_STEP == TILE_K, "K_STEP must equal TILE_K"]()

    var transpose_buf = InlineArray[SIMD[DType.int8, 16], 16](uninitialized=True)

    for n_block_begin in range(0, N, N_BLOCK):
        var n_block_end = min(n_block_begin + N_BLOCK, N)
        var n_block_size = n_block_end - n_block_begin

        for k_block_begin in range(0, K, K_BLOCK):
            var k_block_end = min(k_block_begin + K_BLOCK, K)
            var k_block_size = k_block_end - k_block_begin

            for n_begin in range(0, n_block_size, N_STEP):
                for k_begin in range(0, k_block_size, K_STEP):
                    var tile_base = (n_block_begin * K +
                                    k_block_begin * n_block_size +
                                    n_begin * k_block_size +
                                    k_begin * N_STEP)

                    for i in range(N_STEP):
                        var src_ptr = src.offset((n_block_begin + n_begin + i) * K + k_block_begin + k_begin)
                        var dst_ptr = dst.offset(tile_base + i * K_STEP)
                        dst_ptr.store(src_ptr.load[width=K_STEP]())

                    var tile0 = dst.offset(tile_base)
                    var tile1 = dst.offset(tile_base + TILE_N * K_STEP)
                    transpose_16x16(tile0, K_STEP, tile0, TILE_N, transpose_buf)
                    transpose_16x16(tile1, K_STEP, tile1, TILE_N, transpose_buf)
