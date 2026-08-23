"""Numerical kernels for blocked arrays and dataframe aggregations."""

from std.math import abs, exp, log, pow, sqrt
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("md_binary")
def md_binary(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, op: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var i = 0
    if op < 4:
        while i + W <= n:
            var av = a.unsafe_load[width=W](i)
            var bv = b.unsafe_load[width=W](i)
            if op == 0:
                dst.unsafe_store(i, av + bv)
            elif op == 1:
                dst.unsafe_store(i, av - bv)
            elif op == 2:
                dst.unsafe_store(i, av * bv)
            else:
                dst.unsafe_store(i, av / bv)
            i += W
    while i < n:
        if op == 0:
            dst[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
        elif op == 1:
            dst[unsafe_offset=i] = a[unsafe_offset=i] - b[unsafe_offset=i]
        elif op == 2:
            dst[unsafe_offset=i] = a[unsafe_offset=i] * b[unsafe_offset=i]
        elif op == 3:
            dst[unsafe_offset=i] = a[unsafe_offset=i] / b[unsafe_offset=i]
        else:
            dst[unsafe_offset=i] = pow(a[unsafe_offset=i], b[unsafe_offset=i])
        i += 1


@export("md_scalar")
def md_scalar(a_addr: Int, value: Float64, dst_addr: Int, n: Int, op: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var v = SIMD[DType.float64, W](value)
    var i = 0
    if op < 4:
        while i + W <= n:
            var av = a.unsafe_load[width=W](i)
            if op == 0:
                dst.unsafe_store(i, av + v)
            elif op == 1:
                dst.unsafe_store(i, av - v)
            elif op == 2:
                dst.unsafe_store(i, av * v)
            else:
                dst.unsafe_store(i, av / v)
            i += W
    while i < n:
        if op == 0:
            dst[unsafe_offset=i] = a[unsafe_offset=i] + value
        elif op == 1:
            dst[unsafe_offset=i] = a[unsafe_offset=i] - value
        elif op == 2:
            dst[unsafe_offset=i] = a[unsafe_offset=i] * value
        elif op == 3:
            dst[unsafe_offset=i] = a[unsafe_offset=i] / value
        elif op == 4:
            dst[unsafe_offset=i] = pow(a[unsafe_offset=i], value)
        elif op == 5:
            dst[unsafe_offset=i] = value - a[unsafe_offset=i]
        else:
            dst[unsafe_offset=i] = value / a[unsafe_offset=i]
        i += 1


@export("md_scalar_binary")
def md_scalar_binary(
    a_addr: Int,
    b_addr: Int,
    value: Float64,
    dst_addr: Int,
    n: Int,
    scalar_op: Int,
    binary_op: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var broadcast = SIMD[DType.float64, W](value)
    var i = 0
    while i + W <= n:
        var av = a.unsafe_load[width=W](i)
        var bv = b.unsafe_load[width=W](i)
        var intermediate: SIMD[DType.float64, W]
        if scalar_op == 0:
            intermediate = av + broadcast
        elif scalar_op == 1:
            intermediate = av - broadcast
        elif scalar_op == 2:
            intermediate = av * broadcast
        else:
            intermediate = av / broadcast
        if binary_op == 0:
            dst.unsafe_store(i, intermediate + bv)
        elif binary_op == 1:
            dst.unsafe_store(i, intermediate - bv)
        elif binary_op == 2:
            dst.unsafe_store(i, intermediate * bv)
        else:
            dst.unsafe_store(i, intermediate / bv)
        i += W
    while i < n:
        var intermediate: Float64
        if scalar_op == 0:
            intermediate = a[unsafe_offset=i] + value
        elif scalar_op == 1:
            intermediate = a[unsafe_offset=i] - value
        elif scalar_op == 2:
            intermediate = a[unsafe_offset=i] * value
        else:
            intermediate = a[unsafe_offset=i] / value
        if binary_op == 0:
            dst[unsafe_offset=i] = intermediate + b[unsafe_offset=i]
        elif binary_op == 1:
            dst[unsafe_offset=i] = intermediate - b[unsafe_offset=i]
        elif binary_op == 2:
            dst[unsafe_offset=i] = intermediate * b[unsafe_offset=i]
        else:
            dst[unsafe_offset=i] = intermediate / b[unsafe_offset=i]
        i += 1


@export("md_unary")
def md_unary(a_addr: Int, dst_addr: Int, n: Int, op: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        if op == 0:
            dst[unsafe_offset=i] = -a[unsafe_offset=i]
        elif op == 1:
            dst[unsafe_offset=i] = abs(a[unsafe_offset=i])
        elif op == 2:
            dst[unsafe_offset=i] = exp(a[unsafe_offset=i])
        elif op == 3:
            dst[unsafe_offset=i] = log(a[unsafe_offset=i])
        else:
            dst[unsafe_offset=i] = sqrt(a[unsafe_offset=i])


@export("md_stats")
def md_stats(a_addr: Int, n: Int, dst_addr: Int) abi("C"):
    """Write count, sum, mean, M2, minimum, maximum."""
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    if n <= 0:
        dst[unsafe_offset=0] = 0.0
        dst[unsafe_offset=1] = 0.0
        dst[unsafe_offset=2] = 0.0
        dst[unsafe_offset=3] = 0.0
        dst[unsafe_offset=4] = 1.7976931348623157e308
        dst[unsafe_offset=5] = -1.7976931348623157e308
        return
    var lane_total = SIMD[DType.float64, W](0.0)
    var lane_mean = SIMD[DType.float64, W](0.0)
    var lane_m2 = SIMD[DType.float64, W](0.0)
    var lane_lo = SIMD[DType.float64, W](1.7976931348623157e308)
    var lane_hi = SIMD[DType.float64, W](-1.7976931348623157e308)
    var lane_count = 0
    var i = 0
    while i + W <= n:
        var values = a.unsafe_load[width=W](i)
        lane_count += 1
        lane_total += values
        var delta = values - lane_mean
        lane_mean += delta / Float64(lane_count)
        lane_m2 += delta * (values - lane_mean)
        lane_lo = min(lane_lo, values)
        lane_hi = max(lane_hi, values)
        i += W

    var total = lane_total.reduce_add()
    var mean = 0.0
    var m2 = 0.0
    var seen = 0.0
    var lo = 1.7976931348623157e308
    var hi = -1.7976931348623157e308
    for lane in range(W):
        if lane_count > 0:
            var count = Float64(lane_count)
            var combine_delta = lane_mean[lane] - mean
            var combined = seen + count
            m2 += (
                lane_m2[lane]
                + combine_delta * combine_delta * seen * count / combined
            )
            mean += combine_delta * count / combined
            seen = combined
            if lane_lo[lane] < lo:
                lo = lane_lo[lane]
            if lane_hi[lane] > hi:
                hi = lane_hi[lane]

    while i < n:
        var x = a[unsafe_offset=i]
        total += x
        var delta = x - mean
        seen += 1.0
        mean += delta / seen
        m2 += delta * (x - mean)
        if x < lo:
            lo = x
        if x > hi:
            hi = x
        i += 1
    dst[unsafe_offset=0] = Float64(n)
    dst[unsafe_offset=1] = total
    dst[unsafe_offset=2] = mean
    dst[unsafe_offset=3] = m2
    dst[unsafe_offset=4] = lo
    dst[unsafe_offset=5] = hi


@export("md_sum")
def md_sum(a_addr: Int, n: Int) abi("C") -> Float64:
    var a = fp(a_addr)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[unsafe_offset=i]
        i += 1
    return total


@export("md_dot")
def md_dot(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Float64:
    var a = fp(a_addr)
    var b = fp(b_addr)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return total


@export("md_matmul")
def md_matmul(
    a_addr: Int,
    b_addr: Int,
    dst_addr: Int,
    rows: Int,
    inner: Int,
    cols: Int,
    accumulate: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    if accumulate == 0:
        for i in range(rows * cols):
            dst[unsafe_offset=i] = 0.0
    for i in range(rows):
        for k in range(inner):
            var av = a[unsafe_offset=i * inner + k]
            var j = 0
            var vv = SIMD[DType.float64, W](av)
            while j + W <= cols:
                dst.unsafe_store(
                    i * cols + j,
                    dst.unsafe_load[width=W](i * cols + j)
                    + vv * b.unsafe_load[width=W](k * cols + j),
                )
                j += W
            while j < cols:
                dst[unsafe_offset=i * cols + j] += av * b[unsafe_offset=k * cols + j]
                j += 1


def gpu_matmul_kernel(
    a: Pointer[Float64, AnyOrigin[mut=True]],
    b: Pointer[Float64, AnyOrigin[mut=True]],
    dst: Pointer[Float64, AnyOrigin[mut=True]],
    rows_arg: Int64,
    inner_arg: Int64,
    cols_arg: Int64,
):
    var rows = Int(rows_arg)
    var inner = Int(inner_arg)
    var cols = Int(cols_arg)
    var index = block_idx.x * block_dim.x + thread_idx.x
    if index < rows * cols:
        var row = index // cols
        var col = index - row * cols
        var total = 0.0
        for k in range(inner):
            total += a[unsafe_offset=row * inner + k] * b[unsafe_offset=k * cols + col]
        dst[unsafe_offset=index] = total


@export("md_matmul_gpu")
def md_matmul_gpu(
    a_addr: Int, b_addr: Int, dst_addr: Int, rows: Int, inner: Int, cols: Int
) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var a_device = ctx.enqueue_create_buffer[DType.float64](rows * inner)
        var b_device = ctx.enqueue_create_buffer[DType.float64](inner * cols)
        var dst_device = ctx.enqueue_create_buffer[DType.float64](rows * cols)
        ctx.enqueue_copy(a_device, fp(a_addr))
        ctx.enqueue_copy(b_device, fp(b_addr))
        var threads = 256
        var blocks = (rows * cols + threads - 1) // threads
        ctx.enqueue_function[gpu_matmul_kernel](
            a_device,
            b_device,
            dst_device,
            Int64(rows),
            Int64(inner),
            Int64(cols),
            grid_dim=blocks,
            block_dim=threads,
        )
        ctx.enqueue_copy(fp(dst_addr), dst_device)
        ctx.synchronize()
        return 1
    except:
        return 0


@export("md_group_stats")
def md_group_stats(
    codes_addr: Int,
    values_addr: Int,
    n: Int,
    ngroups: Int,
    counts_addr: Int,
    sums_addr: Int,
    means_addr: Int,
    m2_addr: Int,
    mins_addr: Int,
    maxs_addr: Int,
) abi("C"):
    """Per-code statistics. NaNs are excluded, matching pandas groupby."""
    var codes = ip(codes_addr)
    var values = fp(values_addr)
    var counts = ip(counts_addr)
    var sums = fp(sums_addr)
    var means = fp(means_addr)
    var m2 = fp(m2_addr)
    var mins = fp(mins_addr)
    var maxs = fp(maxs_addr)
    for g in range(ngroups):
        counts[unsafe_offset=g] = 0
        sums[unsafe_offset=g] = 0.0
        means[unsafe_offset=g] = 0.0
        m2[unsafe_offset=g] = 0.0
        mins[unsafe_offset=g] = 1.7976931348623157e308
        maxs[unsafe_offset=g] = -1.7976931348623157e308
    for i in range(n):
        var g = Int(codes[unsafe_offset=i])
        var x = values[unsafe_offset=i]
        if g < 0 or g >= ngroups or x != x:
            continue
        counts[unsafe_offset=g] += 1
        sums[unsafe_offset=g] += x
        var delta = x - means[unsafe_offset=g]
        means[unsafe_offset=g] += delta / Float64(counts[unsafe_offset=g])
        m2[unsafe_offset=g] += delta * (x - means[unsafe_offset=g])
        if x < mins[unsafe_offset=g]:
            mins[unsafe_offset=g] = x
        if x > maxs[unsafe_offset=g]:
            maxs[unsafe_offset=g] = x
