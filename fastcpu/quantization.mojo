from layout import Layout, LayoutTensor
from memory import ImmutUnsafePointer, MutUnsafePointer

struct QuantizedTensor[
    dtype: DType,
    layout: Layout,
    scales_layout: Layout,
    origin: Origin[_],
]:
    var tensor: LayoutTensor[Self.dtype, Self.layout, Self.origin]
    var scales: LayoutTensor[DType.float32, Self.scales_layout, Self.origin]

    fn __init__(
        out self,
        tensor: LayoutTensor[Self.dtype, Self.layout, Self.origin],
        scales: LayoutTensor[DType.float32, Self.scales_layout, Self.origin],
    ):
        self.tensor = tensor
        self.scales = scales

fn quant_range[dtype: DType]() -> Tuple[Float32, Float32, Float32]:
    @parameter
    if dtype == DType.uint8:
        return Tuple(Float32(255.0), Float32(0.0), Float32(255.0))
    elif dtype == DType.int8:
        return Tuple(Float32(127.0), Float32(-128.0), Float32(127.0))
    else:
        constrained[False, "Only uint8 and int8 supported"]()
        return Tuple(Float32(0.0), Float32(0.0), Float32(0.0))

fn quantize_symmetric_channelwise_row[
    dtype: DType,
    M: Int,
    K: Int,
    simd_width: Int,
](
    data_fp32: ImmutUnsafePointer[Float32],
    data_out: MutUnsafePointer[Scalar[dtype]],
    scales: MutUnsafePointer[Float32],
):
    constrained[K % simd_width == 0, "K must be divisible by simd_width"]()
    alias max_quant, min_clamp, max_clamp = quant_range[dtype]()

    for m in range(M):
        var row_ptr = data_fp32.offset(m * K)
        var max_vec = SIMD[DType.float32, simd_width](0)

        @parameter
        for k in range(0, K, simd_width):
            var vals = row_ptr.load[width=simd_width](k)
            var abs_vals = abs(vals)
            max_vec = max(max_vec, abs_vals)

        var max_val = max_vec.reduce_max()
        var scale = max_val / max_quant
        if scale == 0:
            scale = 1.0
        scales[m] = scale

        var inv_scale = 1.0 / scale
        var out_ptr = data_out.offset(m * K)

        @parameter
        for k in range(0, K, simd_width):
            var vals = row_ptr.load[width=simd_width](k)
            var scaled = vals * inv_scale
            var clamped = scaled.clamp(min_clamp, max_clamp)
            var quantized = clamped.cast[dtype]()
            out_ptr.store(k, quantized)

fn quantize_symmetric_channelwise_col[
    dtype: DType,
    K: Int,
    N: Int,
    simd_width: Int,
](
    data_fp32: ImmutUnsafePointer[Float32],
    data_out: MutUnsafePointer[Scalar[dtype]],
    scales: MutUnsafePointer[Float32],
):
    constrained[N % simd_width == 0, "N must be divisible by simd_width"]()
    alias max_quant, min_clamp, max_clamp = quant_range[dtype]()
    alias eps = Float32(1e-10)

    @parameter
    for n_base in range(0, N, simd_width):
        var max_vec = SIMD[DType.float32, simd_width](0)

        for k in range(K):
            var row_ptr = data_fp32.offset(k * N + n_base)
            var vals = row_ptr.load[width=simd_width]()
            var abs_vals = abs(vals)
            max_vec = max(max_vec, abs_vals)

        var scale_vec = max_vec / max_quant
        scale_vec = max(scale_vec, SIMD[DType.float32, simd_width](eps))
        scales.offset(n_base).store(scale_vec)

        var inv_scale_vec = 1.0 / scale_vec

        for k in range(K):
            var row_ptr = data_fp32.offset(k * N + n_base)
            var vals = row_ptr.load[width=simd_width]()
            var scaled = vals * inv_scale_vec
            var clamped = scaled.clamp(min_clamp, max_clamp)
            var quantized = clamped.cast[dtype]()
            data_out.offset(k * N + n_base).store(quantized)
