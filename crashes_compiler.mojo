from algorithm import vectorize
from layout import Layout, LayoutTensor
from memory import alloc, stack_allocation, UnsafePointer
from fastcpu.vnni import pack_b_matrix_vnni_int8_large
from fastcpu.quantization import QuantizedTensor, quantize_symmetric_channelwise_row, quantize_symmetric_channelwise_col
from fastcpu.amx import init_intel_amx, load_amx_tilecfg, release_amx_tilecfg, tile_zero, tile_load, tile_store, tile_dp

alias TILE_M = 16
alias TILE_N = 16
alias TILE_K = 64
alias M_STEP = 32
alias N_STEP = 32
alias K_STEP = 64

fn matmul_amx_slice[
    M: Int, N: Int, K: Int, PACK_N_BLOCK: Int, PACK_K_BLOCK: Int,
    A_layout: Layout, B_layout: Layout, C_layout: Layout,
    A_scales_layout: Layout, B_scales_layout: Layout,
    A_origin: Origin[_], B_origin: Origin[_], C_origin: MutOrigin,
](
    A: QuantizedTensor[DType.uint8, A_layout, A_scales_layout, A_origin],
    B: QuantizedTensor[DType.int8, B_layout, B_scales_layout, B_origin],
    C: LayoutTensor[mut=True, DType.int32, C_layout, C_origin],
    n_offset: Int,
    n_size: Int,
):
    var acc_buf = stack_allocation[M_STEP * N_STEP, DType.int32, alignment=64]()

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

                    var a0_ptr = A.tensor.ptr.offset(m_begin * K + k)
                    var a1_ptr = A.tensor.ptr.offset((m_begin + TILE_M) * K + k)

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

                    var b0_ptr = B.tensor.ptr.offset(b_offset)
                    var b1_ptr = B.tensor.ptr.offset(b_offset + TILE_N * K_STEP)

                    tile_load[0, K](a0_ptr)
                    tile_load[1, K](a1_ptr)
                    tile_load[2, K_STEP](b0_ptr)
                    tile_load[3, K_STEP](b1_ptr)

                    tile_dp[4, 0, 2](a0_ptr, b0_ptr)
                    tile_dp[5, 0, 3](a0_ptr, b1_ptr)
                    tile_dp[6, 1, 2](a1_ptr, b0_ptr)
                    tile_dp[7, 1, 3](a1_ptr, b1_ptr)

                var is_last_k_block = (k_block_begin + k_block_size) >= K

                if is_last_k_block:
                    tile_store[4, N_STEP * 4](acc_buf)
                    tile_store[5, N_STEP * 4](acc_buf.offset(TILE_N))
                    tile_store[6, N_STEP * 4](acc_buf.offset(TILE_M * N_STEP))
                    tile_store[7, N_STEP * 4](acc_buf.offset(TILE_M * N_STEP + TILE_N))

                    for m_local in range(M_STEP):
                        var m = m_begin + m_local
                        var scale_a = A.scales[m, 0][0]

                        fn apply_scale[width: Int](n_local: Int) unified {read n_begin, read B, read acc_buf, read m_local, read scale_a, read C, read m}:
                            var n = n_begin + n_local
                            var scale_b = B.scales[0, n][0]
                            var acc_vals = acc_buf.offset(m_local * N_STEP + n_local).load[width=width]()
                            var scaled = (acc_vals.cast[DType.float32]() * scale_a * scale_b).cast[DType.int32]()
                            C.ptr.offset(m * N + n).store(scaled)

                        vectorize[16, size=N_STEP](apply_scale)
                else:
                    var c00_ptr = C.ptr.offset(m_begin * N + n_begin)
                    var c01_ptr = C.ptr.offset(m_begin * N + n_begin + TILE_N)
                    var c10_ptr = C.ptr.offset((m_begin + TILE_M) * N + n_begin)
                    var c11_ptr = C.ptr.offset((m_begin + TILE_M) * N + n_begin + TILE_N)
                    tile_store[4, N * 4](c00_ptr)
                    tile_store[5, N * 4](c01_ptr)
                    tile_store[6, N * 4](c10_ptr)
                    tile_store[7, N * 4](c11_ptr)

def main():
    alias M = 512
    alias N = 512
    alias K = 512
    alias PACK_N_BLOCK = 128
    alias PACK_K_BLOCK = 512

    if not init_intel_amx():
        print("ERROR: Failed to initialize Intel AMX")
        return

    load_amx_tilecfg()

    var a_fp32 = alloc[Float32](M * K)
    var b_fp32 = alloc[Float32](K * N)

    for i in range(M * K):
        a_fp32[i] = Float32((i % 7) - 3) * 10.0 + 50.0

    for i in range(K * N):
        b_fp32[i] = Float32((i % 11) - 5) * 10.0 + 50.0

    var a_u8 = alloc[UInt8](M * K)
    var b_i8_unpacked = alloc[Int8](K * N)
    var b_i8_vnni = alloc[Int8](N * K)
    var a_scales_ptr = alloc[Float32](M)
    var b_scales_ptr = alloc[Float32](N)

    alias simd_width = 16
    quantize_symmetric_channelwise_row[DType.uint8, M, K, simd_width](a_fp32, a_u8, a_scales_ptr)
    quantize_symmetric_channelwise_col[DType.int8, K, N, simd_width](b_fp32, b_i8_unpacked, b_scales_ptr)
    pack_b_matrix_vnni_int8_large[K, N, PACK_N_BLOCK, PACK_K_BLOCK](b_i8_unpacked, b_i8_vnni)

    var c_ptr = alloc[Int32](M * N)

    alias A_layout = Layout.row_major(M, K)
    alias B_layout = Layout.row_major(N, K)
    alias C_layout = Layout.row_major(M, N)
    alias A_scales_layout = Layout.row_major(M, 1)
    alias B_scales_layout = Layout.row_major(1, N)

    var A_tensor = LayoutTensor[DType.uint8, A_layout](a_u8)
    var A_scales = LayoutTensor[DType.float32, A_scales_layout](a_scales_ptr)
    var A = QuantizedTensor(A_tensor, A_scales)

    var B_tensor = LayoutTensor[DType.int8, B_layout](b_i8_vnni)
    var B_scales = LayoutTensor[DType.float32, B_scales_layout](b_scales_ptr)
    var B = QuantizedTensor(B_tensor, B_scales)

    var C = LayoutTensor[DType.int32, C_layout](c_ptr)

    matmul_amx_slice[M, N, K, PACK_N_BLOCK, PACK_K_BLOCK](A, B, C, 0, N)

    print("C[0,0] =", C[0, 0][0])
    print("C[0,1] =", C[0, 1][0])
    print("C[M-1,N-1] =", C[M-1, N-1][0])

    release_amx_tilecfg()

    a_fp32.free()
    b_fp32.free()
    a_u8.free()
    b_i8_unpacked.free()
    b_i8_vnni.free()
    a_scales_ptr.free()
    b_scales_ptr.free()
    c_ptr.free()
