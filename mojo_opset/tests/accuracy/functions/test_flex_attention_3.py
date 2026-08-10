import csv
import gc
import os

import pytest
import torch
import torch.nn.functional as F

from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.utils.platform import get_platform
from mojo_opset.backends.ttx.kernels.npu.utils import is_910
from mojo_opset.backends.ttx.kernels.npu.flex_attention import _build_packed_block_mask_streaming
from mojo_opset.backends.ttx.kernels.npu.flex_attention import create_block_mask_patched
from mojo_opset.tests.accuracy.functions.test_flex_attention import _sync
from mojo_opset.tests.accuracy.functions.test_flex_attention import _device
from mojo_opset.tests.accuracy.functions.test_flex_attention import _MASK_FUNC_TO_TYPE,GEN_MASK_TRITON
from mojo_opset.tests.accuracy.functions.test_flex_attention import Q_BLOCK_SIZE, KV_BLOCK_SIZE
from mojo_opset.tests.accuracy.functions.test_flex_attention import _build_dense_mask,_sdpa_with_dense_mask
from mojo_opset.tests.accuracy.functions.test_flex_attention import _flex_attention_mojo,_build_block_mask
from mojo_opset.tests.accuracy.functions.test_flex_attention import _sparse_mask_mod ,_full_mask_mod
from mojo_opset.tests.accuracy.functions.test_flex_attention import _cross_sample_causal_video_bidir_mask_mod
from mojo_opset.tests.accuracy.functions.test_flex_attention import _video_stair_mask_mod ,_stair_mask_mod ,build_problem


# NPU device validation monkey-patch (same as original test)
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass

_MB = 1024 ** 2
# ============================================================================
# Performance benchmark (torch_npu.profiler based)
# ============================================================================
def _perf_benchmark(label, build_mask_fn, fwd_fn, q, k, v, prof_dir_root, mask_func,n_element):
    import torch_npu

    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    return_grid = torch.tensor(520000, dtype=q.dtype, device=torch.device(_device()))

    # mask build measurement: peak + stable
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    _sync()
    mask = build_mask_fn()
    mask_peak = torch.npu.max_memory_allocated() / _MB
    torch.npu.empty_cache()
    gc.collect()
    _sync()
    mask_mem = torch.npu.memory_allocated() / _MB

    # fwd measurement (includes grad graph)
    torch.npu.reset_peak_memory_stats()
    _sync()
    out = fwd_fn(q, k, v, mask)
    _sync()
    fwd_mem = torch.npu.max_memory_allocated() / _MB

    # bwd measurement
    torch.npu.reset_peak_memory_stats()
    _sync()
    out.float().mean().backward(return_grid)
    _sync()
    bwd_mem = torch.npu.max_memory_allocated() / _MB

    q.grad = k.grad = v.grad = None
    peak_mem = max(fwd_mem, bwd_mem)
    print(f"[{label}] mask: {mask_mem:.1f}MB(peak:{mask_peak:.1f}MB), fwd_mem: {fwd_mem:.1f}MB, bwd_mem: {bwd_mem:.1f}MB, peak: {peak_mem:.1f}MB")

    # ===== torch_npu.profiler based timing =====
    prof_dir = os.path.join(prof_dir_root, label)
    os.makedirs(prof_dir, exist_ok=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    print(f"\n======================== prof begin ({label}) ====================")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1, warmup=1, active=10, repeat=1, skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
    ) as prof:
        for i in range(12):
            # 重新构造数据，避免L2 Cache影响
            # problem = build_problem(mask_func)
            # q = problem["q"].detach().clone().requires_grad_(True)
            # k = problem["k"].detach().clone().requires_grad_(True)
            # v = problem["v"].detach().clone().requires_grad_(True)

            out = fwd_fn(q, k, v, mask)
            _sync()

            # 插入其他算子，重置L2 Cache, 112M，总访存224MB覆盖112MB L2, 模拟整网调用场景
            for j in range(5):
                a = torch.randn(19573419, dtype=torch.float32, device="cpu").to(q.device)
                b = torch.randn(19573419, dtype=torch.float32, device="cpu").to(q.device)
                c = a + b       # 冲刷全部L2
            _sync()

            out.float().mean().backward(return_grid)
            _sync()
            prof.step()
    print(f"======================== prof end ({label}) ====================")
    if n_element is not None and os.path.exists(prof_dir):
        kernel_profiling_path = max(
            [
                os.path.join(prof_dir, d)
                for d in os.listdir(prof_dir)
                if os.path.isdir(os.path.join(prof_dir, d))
            ],
            key=os.path.getmtime,
        )
        csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

        if os.path.exists(csv_file_path):
            kernel_times = {}
            with open(csv_file_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kernel_name = row["OP Type"]
                    for target in [
                        "flex_attention_backward_dkdv_kernel_tasklist",
                        "flex_attention_backward_dkdv_kernel",
                        "flex_attention_backward_dq_kernel",
                        "flex_attention_kernel",
                    ]:
                        if target in kernel_name:
                            kernel_times[target] = float(row["Avg Time(us)"])
                            break

            active_steps = 5
            peak_tflops = 378.0

            num_n_elements = {
                "flex_attention_backward_dkdv_kernel_tasklist": 8,
                "flex_attention_backward_dkdv_kernel": 8,
                "flex_attention_backward_dq_kernel": 6,
                "flex_attention_kernel":4,
            }
            _, q_head, _,head_dim,= q.shape
            _, kv_head, _, _,= k.shape
            effective_qk_flops = q_head * n_element * head_dim

            print(f"\n{'='*70}")
            print(f"[Tasklist Perf] Hq={q_head}, Hkv={kv_head}, SEQ_LEN={q.shape}")
            print(f"[Tasklist Perf] D={head_dim}, Peak={peak_tflops} TFLOPs")
            print(f"{'='*70}")

            for kernel_name, num_n_element in num_n_elements.items():
                if kernel_name not in kernel_times:
                    print(f"[Tasklist Perf] {kernel_name}: not found in op_statistic.csv")
                    continue
                avg_time_us = kernel_times[kernel_name]
                duration_s = avg_time_us / 1e6
                effective_flops = effective_qk_flops * num_n_element
                total_flops_t = effective_flops / 1e12
                mfu = total_flops_t / duration_s / peak_tflops
                print(
                    f"[Tasklist Perf] {kernel_name}: "
                    f"Avg Time={avg_time_us:.2f} us, "
                    f"num_n_element={num_n_element}, "
                    f"FLOPs={total_flops_t:.4f} T, "
                    f"MFU={mfu:.4f} ({mfu*100:.2f}%)"
                )
            print(f"{'='*70}\n")
        else:
            print(f"[Tasklist Perf] op_statistic.csv not found at: {csv_file_path}")
    else:
        print(f"[Tasklist Perf] Profiling directory not found: {prof_dir}")
    del out, mask
    torch.npu.empty_cache()
    return {"mask_mem_mb": mask_mem, "mask_peak_mb": mask_peak,
            "fwd_mem_mb": fwd_mem, "bwd_mem_mb": bwd_mem,
            "peak_mem_mb": peak_mem}


def _perf_flex_attention(mask_func, problem=None):
    SEQ_LEN = problem["total_s"]
    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]

    prof_dir_root = os.path.join("./npu_profilling", mask_type_str)
    os.makedirs(prof_dir_root, exist_ok=True)

    results = {}

    # mojo_packed: streaming stripe build (no full dense_mask materialized)
    gc.collect()
    torch.npu.empty_cache()
    if SEQ_LEN <= MAX_DENSE_SEQ:
        dense_mask = _build_dense_mask(mask_func, problem)
        _sync()
        n_element = dense_mask.to("cpu").sum().item()
    else:
        # 大序列：分块统计激活元素，避免物化 [S,S] 稠密 mask 导致 OOM
        n_element = _count_n_element(mask_func, problem)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>dense_mask.sum().item() in perf", n_element)
    results["mojo_packed"] = _perf_benchmark(
        "mojo_packed",
        lambda: _build_block_mask(mask_func,problem),
        lambda q, k, v, bm: _flex_attention_mojo(q, k, v, None, bm, 0.0, None),
        problem["q"], problem["k"], problem["v"],
        prof_dir_root,
        mask_func,
        n_element,
    )

    # ascendc: torch SDPA + dense_mask（仅小序列，大序列避免物化稠密 mask）
    gc.collect()
    torch.npu.empty_cache()

    if SEQ_LEN <= MAX_DENSE_SEQ:
        results["ascendc"] = _perf_benchmark(
            "ascendc",
            lambda: _build_dense_mask(mask_func, problem),
            lambda q, k, v, m: _sdpa_with_dense_mask(q, k, v, m, 0.0, None),
            problem["q"], problem["k"], problem["v"],
            prof_dir_root,
            mask_func,
            None
        )
    return results

# ============================================================================
# 分块统计 mask 激活元素（大序列避免物化全量稠密 mask）
# ============================================================================
MAX_DENSE_SEQ = 20000


def _count_n_element(mask_func, problem, q_chunk=1024):
    mask_mod = mask_func(problem)
    S = problem["total_s"]
    device = problem["q"].device
    total = 0
    for qs in range(0, S, q_chunk):
        qe = min(qs + q_chunk, S)
        cb = qe - qs
        q_idx = torch.arange(qs, qe, device=device)[:, None].expand(cb, S)
        kv_idx = torch.arange(0, S, device=device)[None, :].expand(cb, S)
        m = mask_mod(0, 0, q_idx, kv_idx)
        total += int(m.sum().item())
    return total


# ============================================================================
# 随机用例生成（固定种子保证可复现，同时保证随机性）
# ============================================================================
import random as _random

_RNG = _random.Random(2026)
_RAND_BATCH = [1, 2]
_RAND_QHEAD = [16, 32]
_RAND_HDIM = [64, 128]
_RAND_DTYPES = [torch.bfloat16, torch.float16]
_RAND_MAG = [5000, 60000, 300000, 1000000]
_RAND_SLIDE = [512, 1024, 4096, 65536]
_RAND_GLOBAL = [4, 8, 16]
_RAND_TEXT_MASKS = [_sparse_mask_mod, _full_mask_mod, _cross_sample_causal_video_bidir_mask_mod]


def _random_positive_split(rng, total, k):
    """把 total 拆成 k 个正整数（任意数值，非 10 整数倍）。"""
    if k == 1:
        return [total]
    points = sorted(rng.sample(range(1, total), k - 1))
    parts = [points[0]]
    for i in range(1, k - 1):
        parts.append(points[i] - points[i - 1])
    parts.append(total - points[-1])
    return parts


def _make_random_case(rng):
    batch = rng.choice(_RAND_BATCH)
    mag = rng.choice(_RAND_MAG)
    # 大序列（300k/1M）用较小的 head/dim，避免 q/k/v 与参考比对在 60GiB 卡上显存溢出
    if mag >= 300000:
        q_head = rng.choice([4, 8])
        hdim = 64
    else:
        q_head = rng.choice(_RAND_QHEAD)
        hdim = rng.choice(_RAND_HDIM)
    kv_head = rng.choice([h for h in [2, 4, 8, 16] if h <= q_head])
    n_samples = rng.randint(1, 3)
    weights = [rng.random() for _ in range(n_samples)]
    wsum = sum(weights) or 1.0
    sample_totals = [max(int(mag * w / wsum), 2) for w in weights]
    data_lens, data_types = [], []
    for st in sample_totals:
        n_seg = rng.randint(1, 4)
        segs = _random_positive_split(rng, st, n_seg)
        types = [rng.choice(["text", "image_gen"]) for _ in range(n_seg)]
        data_lens.append(segs)
        data_types.append(types)
    sliding = rng.choice(_RAND_SLIDE)
    global_w = rng.choice(_RAND_GLOBAL)
    dtype = rng.choice(_RAND_DTYPES)
    mask = rng.choice(_RAND_TEXT_MASKS)
    return batch, q_head, kv_head, hdim, data_lens, data_types, sliding, global_w, dtype, mask


_RANDOM_CASES = [
    pytest.param(*_make_random_case(_RNG), id=f"rand_{i:02d}")
    for i in range(10)
]

@pytest.mark.parametrize(
    "batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func,",
    [
        # ===== sparse（文本/图像混合，seq 1k ~ 1M，非 10 整数倍） =====
        pytest.param(1, 16, 8, 128, [[123, 4567, 89]], [["text", "image_gen", "text"]],
                     512, 4, torch.bfloat16, _sparse_mask_mod, id="sparse_b1_s5k"),
        pytest.param(2, 16, 8, 128, [[1233, 4567], [891, 2345]], [["text", "image_gen"], ["text", "image_gen"]],
                     1024, 4, torch.bfloat16, _sparse_mask_mod, id="sparse_b2_s9k"),
        pytest.param(1, 32, 16, 128, [[12345, 23456, 34567]], [["text", "image_gen", "text"]],
                     4096, 8, torch.bfloat16, _sparse_mask_mod, id="sparse_b1_s70k"),
        pytest.param(1, 4, 2, 64, [[333333, 333333, 333334]], [["text", "image_gen", "text"]],
                     65536, 16, torch.bfloat16, _sparse_mask_mod, id="sparse_b1_s1M"),

        # ===== full =====
        pytest.param(2, 16, 8, 128, [[1233, 4567], [891, 2345]], [["text", "image_gen"], ["text", "image_gen"]],
                     1024, 4, torch.bfloat16, _full_mask_mod, id="full_b2_s9k"),
        pytest.param(1, 32, 16, 128, [[45678, 98765, 56789]], [["text", "image_gen", "text"]],
                     65536, 16, torch.bfloat16, _full_mask_mod, id="full_b1_s201k"),

        # ===== cross_sample_causal_video_bidir =====
        pytest.param(1, 16, 8, 128, [[10007, 20003]], [["text", "image_gen"]],
                     1024, 4, torch.bfloat16, _cross_sample_causal_video_bidir_mask_mod, id="cross_b1_s30k"),
        pytest.param(2, 16, 8, 64, [[2345, 6789], [1111, 2222]], [["text", "image_gen"], ["text", "image_gen"]],
                     2048, 8, torch.float16, _cross_sample_causal_video_bidir_mask_mod, id="cross_b2_d64_s12k"),

        # ===== video_stair（帧长度拆分为任意数值，非 10 整数倍） =====
        pytest.param(1, 16, 8, 128, [[1234, 2345]], [[[600, 634], [1234, 1111]]],
                     1024, 4, torch.bfloat16, _video_stair_mask_mod, id="video_stair_s3579"),
        pytest.param(1, 32, 16, 128, [[12345, 23456, 34567]], [
                        [[1234, 2266, 8845], [3456, 7890, 12110], [5678, 9123, 19766]],
                    ], 4096, 8, torch.bfloat16, _video_stair_mask_mod, id="video_stair_s70368"),

        # ===== stair =====
        pytest.param(1, 16, 8, 128, [[3500, 4100]], [[[1234, 2266], [987, 3113]]],
                     1024, 4, torch.bfloat16, _stair_mask_mod, id="stair_s7600"),
        pytest.param(2, 16, 8, 64, [[12345, 23456], [34567, 45678]], [
                        [[6000, 6345], [11111, 12345]],
                        [[11111, 23456], [22222, 23456]],
                    ], 2048, 8, torch.float16, _stair_mask_mod, id="stair_b2_s116046"),
    ] + _RANDOM_CASES
)
@pytest.mark.skipif(get_platform() != "npu", reason="FlexAttention TTX backend requires NPU")
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_flex_attention_perf(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func,):
    problem = build_problem(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func,)
    results = _perf_flex_attention(mask_func, problem)
    print(f"\n{'=' * 60}")
    print(f"Performance results for {_MASK_FUNC_TO_TYPE[id(mask_func)]}:")
    for label, r in results.items():
        print(f"  [{label}] mask: {r['mask_mem_mb']:.1f}MB(peak:{r['mask_peak_mb']:.1f}MB), "
              f"fwd_mem: {r['fwd_mem_mb']:.1f}MB, bwd_mem: {r['bwd_mem_mb']:.1f}MB, "
              f"peak: {r['peak_mem_mb']:.1f}MB")
    print(f"{'=' * 60}")
