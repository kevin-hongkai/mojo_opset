import csv
import gc
import os
import time

import pytest
import torch
import torch.nn.functional as F

from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.utils.platform import get_platform
from mojo_opset.backends.ttx.kernels.npu.utils import is_910
from mojo_opset.backends.ttx.kernels.npu.flex_attention import _build_packed_block_mask_streaming
from mojo_opset.backends.ttx.kernels.npu.flex_attention import create_block_mask_patched
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import USE_MOJO_FLEX_ATTENTION,_sync
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _device
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _MASK_FUNC_TO_TYPE,GEN_MASK_TRITON
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import Q_BLOCK_SIZE, KV_BLOCK_SIZE
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _build_dense_mask,_sdpa_with_dense_mask
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _flex_attention_mojo,_build_block_mask
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _sparse_mask_mod ,_full_mask_mod
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _cross_sample_causal_video_bidir_mask_mod
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import _video_stair_mask_mod ,_stair_mask_mod ,build_problem
# 复用 test_flex_attention_3 中的随机/多样本/混合段数/共用固定用例，避免重复定义
from mojo_opset.tests.accuracy.functions.test_flex_attention_3 import (
    _RANDOM_CASES, _MULTI_SAMPLE_CASES, _MIXED_SEG_CASES, _COMMON_FIXED_CASES,
)


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
            time.sleep(0.5)
    print(f"======================== prof end ({label}) ====================")
    if n_element is not None and os.path.exists(prof_dir):
        if USE_MOJO_FLEX_ATTENTION:
            num_n_elements = {
                                "flex_attention_backward_dkdv_kernel_tasklist": 8,
                                "flex_attention_backward_dkdv_kernel": 8,
                                "flex_attention_backward_dq_kernel": 6,
                                "flex_attention_kernel":4,
                        }
        else:
            num_n_elements = {
                                "triton_flex_attention_bwd_dkdv_tasklist": 8,
                                "triton_flex_attention_bwd_dkdv_mask_out": 8,
                                "triton_flex_attention_bwd_dq_mask_out": 6,
                                "triton_flex_attention_fwd_mask_out":4,
                                    }
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
                    for target,_ in num_n_elements.items():
                        if target in kernel_name:
                            kernel_times[target] = float(row["Avg Time(us)"])
                            break

            active_steps = 5
            peak_tflops = 378.0

            
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

    prof_dir_root = os.path.join("./npu_profiling", mask_type_str)
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


def _count_n_element(mask_func, problem, q_chunk=512):
    """分块统计 mask 激活元素个数，避免物化全量稠密 mask。

    使用广播优化：q_idx [cb,1] + kv_idx [1,S]，mask_mod 内部自动广播到 [cb,S]，
    gather 次数从 cb×S 降为 cb+S。在 NPU 上计算（元素级 bool 运算快），
    1D 索引张量转 int32 减半内存。
    """
    npu_problem = {}
    for key, val in problem.items():
        if isinstance(val, torch.Tensor):
            t = val.detach()
            if t.dtype in (torch.int64, torch.long) and t.dim() == 1:
                t = t.to(torch.int32)
            npu_problem[key] = t
        else:
            npu_problem[key] = val
    mask_mod = mask_func(npu_problem)
    S = problem["total_s"]
    device = problem["q"].device
    total = 0
    for qs in range(0, S, q_chunk):
        qe = min(qs + q_chunk, S)
        q_idx = torch.arange(qs, qe, device=device, dtype=torch.int32)[:, None]
        kv_idx = torch.arange(0, S, device=device, dtype=torch.int32)[None, :]
        with torch.no_grad():
            m = mask_mod(0, 0, q_idx, kv_idx)
        total += int(m.sum().item())
        del m, q_idx, kv_idx
    return total


import random as _random  


@pytest.mark.parametrize(
    "batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func,",
    # 共用固定用例前 3 个（sparse 5k/9k/70k）
     _COMMON_FIXED_CASES + _RANDOM_CASES
     + _MIXED_SEG_CASES + _MULTI_SAMPLE_CASES
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
