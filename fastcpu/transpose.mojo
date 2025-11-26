from collections import InlineArray

fn log2[N: Int]() -> Int:
    @parameter
    if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()

fn bit_reverse[bits: Int, x: Int]() -> Int:
    @parameter
    if bits == 0:
        return 0
    else:
        alias lsb = x & 1
        alias rest = x >> 1
        return (lsb << (bits - 1)) | bit_reverse[bits - 1, rest]()

alias Row16 = SIMD[DType.int8, 16]

fn interleave_idx[N: Int, i: Int, stride: Int, high: Bool]() -> Int:
    alias half = N // 2
    alias base = half if high else 0
    alias group = i // (2 * stride)
    alias within = i % (2 * stride)
    @parameter
    if within < stride:
        return base + group * stride + within
    else:
        return N + base + group * stride + (within - stride)

@always_inline
fn interleave_16[stride: Int, high: Bool](a: Row16, b: Row16) -> Row16:
    alias i0 = interleave_idx[16, 0, stride, high]()
    alias i1 = interleave_idx[16, 1, stride, high]()
    alias i2 = interleave_idx[16, 2, stride, high]()
    alias i3 = interleave_idx[16, 3, stride, high]()
    alias i4 = interleave_idx[16, 4, stride, high]()
    alias i5 = interleave_idx[16, 5, stride, high]()
    alias i6 = interleave_idx[16, 6, stride, high]()
    alias i7 = interleave_idx[16, 7, stride, high]()
    alias i8 = interleave_idx[16, 8, stride, high]()
    alias i9 = interleave_idx[16, 9, stride, high]()
    alias i10 = interleave_idx[16, 10, stride, high]()
    alias i11 = interleave_idx[16, 11, stride, high]()
    alias i12 = interleave_idx[16, 12, stride, high]()
    alias i13 = interleave_idx[16, 13, stride, high]()
    alias i14 = interleave_idx[16, 14, stride, high]()
    alias i15 = interleave_idx[16, 15, stride, high]()
    return a.shuffle[i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15](b)

@always_inline
fn transpose[
    N: Int,
    interleave_fn: fn[stride: Int, high: Bool](SIMD[DType.int8, N], SIMD[DType.int8, N]) -> SIMD[DType.int8, N],
    src_origin: Origin[_],
    dst_origin: MutOrigin,
](
    src: UnsafePointer[Int8, origin=src_origin], src_stride: Int,
    dst: UnsafePointer[Int8, origin=dst_origin], dst_stride: Int,
    mut r: InlineArray[SIMD[DType.int8, N], N],
):
    @parameter
    for i in range(N):
        r[i] = src.offset(i * src_stride).load[width=N]()
    alias num_stages = log2[N]()
    @parameter
    for stage in range(num_stages):
        alias stride = 1 << stage
        alias groups = N // (2 * stride)
        @parameter
        for g in range(groups):
            @parameter
            for j in range(stride):
                var idx0 = g * 2 * stride + j
                var idx1 = idx0 + stride
                var lo = interleave_fn[stride, False](r[idx0], r[idx1])
                var hi = interleave_fn[stride, True](r[idx0], r[idx1])
                r[idx0] = lo
                r[idx1] = hi
    @parameter
    for i in range(N):
        alias dest = bit_reverse[num_stages, i]()
        dst.offset(dest * dst_stride).store(r[i])

@always_inline
fn transpose_16x16[src_origin: Origin[_], dst_origin: MutOrigin](
    src: UnsafePointer[Int8, origin=src_origin], src_stride: Int,
    dst: UnsafePointer[Int8, origin=dst_origin], dst_stride: Int,
    mut r: InlineArray[Row16, 16],
):
    transpose[16, interleave_16](src, src_stride, dst, dst_stride, r)