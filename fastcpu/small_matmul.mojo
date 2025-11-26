from algorithm import vectorize, parallelize
from layout import Layout, LayoutTensor
from sys.intrinsics import llvm_intrinsic
from .quantization import QuantizedTensor

fn vpdpbusd[width: Int](
    src: SIMD[DType.int32, 4 * width],
    a: SIMD[DType.uint8, 16 * width],
    b: SIMD[DType.int8, 16 * width]
) -> SIMD[DType.int32, 4 * width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(128 * width),
        SIMD[DType.int32, 4 * width]
    ](src, a, b)

fn vpdpbusd_512(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.uint8, 64],
    b: SIMD[DType.int8, 64]
) -> SIMD[DType.int32, 16]:
    return vpdpbusd[4](src, a, b)

fn small_matmul_uint8_vnni_channelwise_parallel[
    M: Int,
    N: Int,
    K: Int,
    num_workers: Int,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    A_scales_layout: Layout,
    B_scales_layout: Layout,
    A_origin: Origin[_],
    B_origin: Origin[_],
    C_origin: MutOrigin,
](
    A: QuantizedTensor[DType.uint8, A_layout, A_scales_layout, A_origin],
    B: QuantizedTensor[DType.int8, B_layout, B_scales_layout, B_origin],
    C: LayoutTensor[mut=True, DType.int32, C_layout, C_origin],
):
    constrained[K % 64 == 0, "K must be divisible by 64 for VNNI"]()

    @parameter
    fn compute_row(m: Int):
        for n in range(N):
            var acc = SIMD[DType.int32, 16](0)

            fn dot_k[width: Int](k: Int) unified {mut acc, read A, read B, read m, read n}:
                var a_u8 = A.tensor.load[width=64](m, k)
                var b_u8 = B.tensor.load[width=64](n, k)
                acc = vpdpbusd_512(acc, a_u8, b_u8)

            vectorize[64, size=K](dot_k)

            var scale_a = A.scales[m, 0][0]
            var scale_b = B.scales[0, n][0]
            var result = Int32(Float32(acc.reduce_add()) * scale_a * scale_b)
            C.store(m, n, SIMD[DType.int32, 1](result))

    parallelize[compute_row](M, num_workers)
