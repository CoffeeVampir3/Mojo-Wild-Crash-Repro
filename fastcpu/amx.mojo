from sys.intrinsics import llvm_intrinsic
from memory import UnsafePointer
from collections import InlineArray

struct TileConfig(Movable):
    var palette_id: UInt8
    var start_row: UInt8
    var reserved: InlineArray[UInt8, 14]
    var colsb: InlineArray[UInt16, 16]
    var rows: InlineArray[UInt8, 16]

    fn __init__(out self):
        self.palette_id = 0
        self.start_row = 0
        self.reserved = InlineArray[UInt8, 14](fill=0)
        self.colsb = InlineArray[UInt16, 16](fill=0)
        self.rows = InlineArray[UInt8, 16](fill=0)

fn make_i8_gemm_config[TILE_M: Int, TILE_K: Int, TILE_N: Int]() -> TileConfig:
    var cfg = TileConfig()
    cfg.palette_id = 1
    cfg.rows[0] = TILE_M
    cfg.rows[1] = TILE_M
    cfg.colsb[0] = TILE_K
    cfg.colsb[1] = TILE_K
    cfg.rows[2] = TILE_K // 4
    cfg.rows[3] = TILE_K // 4
    cfg.colsb[2] = TILE_N * 4
    cfg.colsb[3] = TILE_N * 4
    for i in range(4, 8):
        cfg.rows[i] = TILE_M
        cfg.colsb[i] = TILE_N * 4
    return cfg^

fn init_intel_amx() -> Bool:
    alias SYS_arch_prctl = 158
    alias ARCH_REQ_XCOMP_PERM = 0x1023
    alias XFEATURE_XTILEDATA = 18

    var result = __mlir_op.`pop.external_call`[
        func = "syscall".value,
        _type=Int64,
    ](Int64(SYS_arch_prctl), Int64(ARCH_REQ_XCOMP_PERM), Int64(XFEATURE_XTILEDATA))

    return result == 0

fn load_amx_tilecfg():
    var cfg = make_i8_gemm_config[16, 64, 16]()
    var cfg_ptr = UnsafePointer(to=cfg)
    llvm_intrinsic["llvm.x86.ldtilecfg", NoneType](cfg_ptr.bitcast[NoneType]())

fn release_amx_tilecfg():
    llvm_intrinsic["llvm.x86.tilerelease", NoneType]()

fn tile_zero[tile_id: Int]():
    constrained[tile_id >= 0 and tile_id < 8, "tile_id must be 0-7"]()
    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(tile_id))

fn tile_load[tile_id: Int, byte_stride: Int, dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    constrained[tile_id >= 0 and tile_id < 8, "tile_id must be 0-7"]()
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(tile_id),
        ptr,
        byte_stride
    )

fn tile_store[tile_id: Int, byte_stride: Int, dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    constrained[tile_id >= 0 and tile_id < 8, "tile_id must be 0-7"]()
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](
        Int8(tile_id),
        ptr,
        byte_stride
    )


fn tile_dp[tmm_c: Int, tmm_a: Int, tmm_b: Int, a_dtype: DType, b_dtype: DType](
    a_ptr: UnsafePointer[Scalar[a_dtype]],
    b_ptr: UnsafePointer[Scalar[b_dtype]],
):
    constrained[tmm_c >= 0 and tmm_c < 8 and tmm_a >= 0 and tmm_a < 8 and tmm_b >= 0 and tmm_b < 8, "tile register IDs must be 0-7"]()

    @parameter
    if a_dtype == DType.uint8 and b_dtype == DType.int8:
        llvm_intrinsic["llvm.x86.tdpbusd", NoneType](Int8(tmm_c), Int8(tmm_a), Int8(tmm_b))
    elif a_dtype == DType.int8 and b_dtype == DType.uint8:
        llvm_intrinsic["llvm.x86.tdpbsud", NoneType](Int8(tmm_c), Int8(tmm_a), Int8(tmm_b))
    elif a_dtype == DType.int8 and b_dtype == DType.int8:
        llvm_intrinsic["llvm.x86.tdpbssd", NoneType](Int8(tmm_c), Int8(tmm_a), Int8(tmm_b))
    elif a_dtype == DType.uint8 and b_dtype == DType.uint8:
        llvm_intrinsic["llvm.x86.tdpbuud", NoneType](Int8(tmm_c), Int8(tmm_a), Int8(tmm_b))
    elif a_dtype == DType.bfloat16 and b_dtype == DType.bfloat16:
        llvm_intrinsic["llvm.x86.tdpbf16ps", NoneType](Int8(tmm_c), Int8(tmm_a), Int8(tmm_b))
    elif a_dtype == DType.float16 and b_dtype == DType.float16:
        llvm_intrinsic["llvm.x86.tdpfp16ps", NoneType](Int8(tmm_c), Int8(tmm_a), Int8(tmm_b))
    else:
        constrained[False, "Unsupported dtype combination"]()
