import gc
import torch
import torch.nn.functional as F
import pytest

from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.backends.ttx.kernels.npu.flex_attention import create_block_mask_patched
from mojo_opset.backends.ttx.kernels.npu.flex_attention import MASK_BLOCK_SIZE

from mojo_opset.tests.accuracy.functions.test_flex_attention import _build_dense_mask, _device, _sdpa_with_dense_mask
from mojo_opset.tests.accuracy.functions.test_flex_attention import _sync
from mojo_opset.tests.accuracy.functions.test_flex_attention import build_problem
from mojo_opset.tests.accuracy.functions.test_flex_attention import _flex_attention_mojo
from mojo_opset.tests.accuracy.functions.test_flex_attention import Q_BLOCK_SIZE, KV_BLOCK_SIZE
from mojo_opset.tests.accuracy.functions.test_flex_attention import USE_MOJO_FLEX_ATTENTION
from mojo_opset.tests.accuracy.functions.test_flex_attention import _build_block_mask

# NPU device validation monkey-patch (same as original test)
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass

def create_causal_mask_mod(cu_seqlens, device, causal=True):
    seg_ids = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1, device=device),
                                      cu_seqlens[1:] - cu_seqlens[:-1])
    pos = torch.arange(cu_seqlens[-1], device=device)

    def mask_mod(b, h, q_idx, kv_idx, causal=True):
        q_seg = seg_ids[q_idx]
        k_seg = seg_ids[kv_idx]
        q_pos = pos[q_idx]
        k_pos = pos[kv_idx]
        same_seg_mask = (q_seg == k_seg)
        if causal:
            return same_seg_mask & (q_pos >= k_pos)
        return same_seg_mask

    from functools import partial
    return partial(mask_mod, causal=causal)


def create_block_diffusion_mask_mod(cu_seqlens, block_size, device, input_len):
    seqlen = cu_seqlens[-1]
    seg_ids = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1, device=device),
                                      cu_seqlens[1:] - cu_seqlens[:-1])
    pos = torch.arange(seqlen, device=device)
    offset = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1])
    semi_lens = torch.repeat_interleave((cu_seqlens[1:] - cu_seqlens[:-1]) // 2,
                                        cu_seqlens[1:] - cu_seqlens[:-1])
    pos_in_seg = pos - offset
    pos_in_semi_seg = pos_in_seg % semi_lens
    block_id_in_semi_seg = pos_in_semi_seg // block_size

    pad_len = input_len - seqlen
    if pad_len > 0:
        pad_value = -1
        seg_ids = torch.nn.functional.pad(seg_ids, (0, pad_len), value=pad_value)
        pos_in_seg = torch.nn.functional.pad(pos_in_seg, (0, pad_len), value=pad_value)
        block_id_in_semi_seg = torch.nn.functional.pad(block_id_in_semi_seg, (0, pad_len), value=pad_value)
        semi_lens = torch.nn.functional.pad(semi_lens, (0, pad_len), value=1)
        offset = torch.nn.functional.pad(offset, (0, pad_len), value=0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = seg_ids[q_idx]
        k_seg = seg_ids[kv_idx]
        q_pos = pos_in_seg[q_idx]
        k_pos = pos_in_seg[kv_idx]
        q_block_id = block_id_in_semi_seg[q_idx]
        k_block_id = block_id_in_semi_seg[kv_idx]
        semi_len = semi_lens[q_idx]

        same_seg_mask = (q_seg == k_seg)
        cond_block_diag = (q_pos < semi_len) & (k_pos < semi_len) & (q_block_id == k_block_id)
        cond_block_causal = (q_pos >= semi_len) & (k_pos >= semi_len) & (q_block_id >= k_block_id)
        cond_cross_attn = (q_pos < semi_len) & (k_pos >= semi_len) & (q_block_id > k_block_id)
        is_pad = (q_idx >= pos.shape[-1]) | (kv_idx >= pos.shape[-1])
        return same_seg_mask & (cond_block_diag | cond_block_causal | cond_cross_attn) & ~is_pad
    return mask_mod


def create_causal_block_mask(cu_seqlens, block_size, device, input_len, n_inter_head=4):
    seqlen = cu_seqlens[-1]
    seg_ids = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1, device=device),
                                      cu_seqlens[1:] - cu_seqlens[:-1])
    pos = torch.arange(seqlen, device=device)
    offset = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1])
    pos_in_seg = pos - offset
    block_id_in_seg = pos_in_seg // block_size

    pad_len = input_len - seqlen
    if pad_len > 0:
        pad_value = -1
        seg_ids = torch.nn.functional.pad(seg_ids, (0, pad_len), value=pad_value)
        pos_in_seg = torch.nn.functional.pad(pos_in_seg, (0, pad_len), value=pad_value)
        block_id_in_seg = torch.nn.functional.pad(block_id_in_seg, (0, pad_len), value=pad_value)
        offset = torch.nn.functional.pad(offset, (0, pad_len), value=0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = seg_ids[q_idx]
        k_seg = seg_ids[kv_idx]
        q_pos = pos_in_seg[q_idx]
        k_pos = pos_in_seg[kv_idx]
        q_block_id = block_id_in_seg[q_idx]
        k_block_id = block_id_in_seg[kv_idx]
        same_seg_mask = (q_seg == k_seg)
        cond_block_causal = (q_block_id >= k_block_id)
        is_pad = (q_idx >= pos.shape[-1]) | (kv_idx >= pos.shape[-1])
        return same_seg_mask & cond_block_causal & ~is_pad
    return mask_mod


def create_in_block_mask(cu_seqlens, block_size, device, input_len):
    seqlen = cu_seqlens[-1]
    seg_ids = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1, device=device),
                                      cu_seqlens[1:] - cu_seqlens[:-1])
    pos = torch.arange(seqlen, device=device)
    offset = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1])
    pos_in_seg = pos - offset
    block_id_in_seg = pos_in_seg // block_size

    pad_len = input_len - seqlen
    if pad_len > 0:
        pad_value = -1
        seg_ids = torch.nn.functional.pad(seg_ids, (0, pad_len), value=pad_value)
        pos_in_seg = torch.nn.functional.pad(pos_in_seg, (0, pad_len), value=pad_value)
        block_id_in_seg = torch.nn.functional.pad(block_id_in_seg, (0, pad_len), value=pad_value)
        offset = torch.nn.functional.pad(offset, (0, pad_len), value=0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = seg_ids[q_idx]
        k_seg = seg_ids[kv_idx]
        q_pos = pos_in_seg[q_idx]
        k_pos = pos_in_seg[kv_idx]
        q_block_id = block_id_in_seg[q_idx]
        k_block_id = block_id_in_seg[kv_idx]
        same_seg_mask = (q_seg == k_seg)
        cond_block_diag = (q_block_id == k_block_id)
        is_pad = (q_idx >= pos.shape[-1]) | (kv_idx >= pos.shape[-1])
        return same_seg_mask & cond_block_diag & ~is_pad
    return mask_mod


def create_between_block_mask(cu_seqlens, block_size, device, input_len):
    seqlen = cu_seqlens[-1]
    seg_ids = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1, device=device),
                                      cu_seqlens[1:] - cu_seqlens[:-1])
    pos = torch.arange(seqlen, device=device)
    offset = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1])
    pos_in_seg = pos - offset
    block_id_in_seg = pos_in_seg // block_size

    pad_len = input_len - seqlen
    if pad_len > 0:
        pad_value = -1
        seg_ids = torch.nn.functional.pad(seg_ids, (0, pad_len), value=pad_value)
        pos_in_seg = torch.nn.functional.pad(pos_in_seg, (0, pad_len), value=pad_value)
        block_id_in_seg = torch.nn.functional.pad(block_id_in_seg, (0, pad_len), value=pad_value)
        offset = torch.nn.functional.pad(offset, (0, pad_len), value=0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = seg_ids[q_idx]
        k_seg = seg_ids[kv_idx]
        q_pos = pos_in_seg[q_idx]
        k_pos = pos_in_seg[kv_idx]
        q_block_id = block_id_in_seg[q_idx]
        k_block_id = block_id_in_seg[kv_idx]
        same_seg_mask = (q_seg == k_seg)
        cond_block_causal = (q_block_id > k_block_id)
        is_pad = (q_idx >= pos.shape[-1]) | (kv_idx >= pos.shape[-1])
        return same_seg_mask & cond_block_causal & ~is_pad
    return mask_mod


def create_block_diffusion_mask_mod_concat(block_size, n, reverse=False, model_block_size=None):
    def mask_mod(b, h, q_idx, kv_idx, block_size=None, n=None, model_block_size=None):
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)
        block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
        if model_block_size is not None:
            block_q = torch.where(x0_flag_q == 1, (q_idx - n) // model_block_size, q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // model_block_size, kv_idx // block_size)
        block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
        return block_diagonal | offset_block_causal | block_causal

    def mask_mod_reverse(b, h, q_idx, kv_idx, block_size=None, n=None, model_block_size=None):
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)
        block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (block_q < block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
        if model_block_size is not None:
            block_q = torch.where(x0_flag_q == 1, (q_idx - n) // model_block_size, q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // model_block_size, kv_idx // block_size)
        block_causal = (block_q <= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
        return block_diagonal | offset_block_causal | block_causal

    from functools import partial
    if reverse:
        return partial(mask_mod_reverse, block_size=block_size, n=n, model_block_size=model_block_size)
    return partial(mask_mod, block_size=block_size, n=n, model_block_size=model_block_size)


def create_block_diffusion_more_causal_mask_mod_concat(block_size, n, reverse=False):
    def mask_mod(b, h, q_idx, kv_idx, block_size=None, n=None):
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)
        block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
        block_diagonal = (block_q == block_kv) & (x0_flag_q == 0) & (x0_flag_kv == 0)
        offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
        block_causal = (q_idx >= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)
        return block_diagonal | offset_block_causal | block_causal

    def mask_mod_reverse(b, h, q_idx, kv_idx, block_size=None, n=None):
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)
        block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
        block_diagonal = (block_q == block_kv) & (x0_flag_q == 0) & (x0_flag_kv == 0)
        offset_block_causal = (block_q < block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
        block_causal = (q_idx <= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)
        return block_diagonal | offset_block_causal | block_causal

    from functools import partial
    if reverse:
        return partial(mask_mod_reverse, block_size=block_size, n=n)
    return partial(mask_mod, block_size=block_size, n=n)


# ============================================================================
# 泛化 mask：统一接口 mask_func(problem) -> mask_mod(b, h, q_idx, kv_idx)
# ============================================================================
def _causal_mask_mod(problem):
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return mask_mod


def _seg_causal_mask_mod(problem):
    return create_causal_mask_mod(problem["cu_seqlens"], problem["q"].device, causal=True)


def _block_diffusion_mask_mod(problem):
    return create_block_diffusion_mask_mod(
        problem["cu_seqlens"], MASK_BLOCK_SIZE, problem["q"].device, problem["total_s"])


def _causal_block_mask_mod(problem):
    return create_causal_block_mask(
        problem["cu_seqlens"], MASK_BLOCK_SIZE, problem["q"].device, problem["total_s"])


def _in_block_mask_mod(problem):
    return create_in_block_mask(
        problem["cu_seqlens"], MASK_BLOCK_SIZE, problem["q"].device, problem["total_s"])


def _between_block_mask_mod(problem):
    return create_between_block_mask(
        problem["cu_seqlens"], MASK_BLOCK_SIZE, problem["q"].device, problem["total_s"])


def _bd_concat_mask_mod(problem):
    n = problem["total_s"] // 2
    return create_block_diffusion_mask_mod_concat(MASK_BLOCK_SIZE, n, reverse=False, model_block_size=None)


def _bd_more_causal_concat_mask_mod(problem):
    n = problem["total_s"] // 2
    return create_block_diffusion_more_causal_mask_mod_concat(MASK_BLOCK_SIZE, n, reverse=False)


_MASK_FUNCS_2 = [
    ("causal", _causal_mask_mod),
    ("seg_causal", _seg_causal_mask_mod),
    ("block_diffusion", _block_diffusion_mask_mod),
    ("causal_block", _causal_block_mask_mod),
    ("in_block", _in_block_mask_mod),
    ("between_block", _between_block_mask_mod),
    ("bd_concat", _bd_concat_mask_mod),
    ("bd_more_causal_concat", _bd_more_causal_concat_mask_mod),
]
_MASK_FUNC_TO_TYPE_2 = {id(fn): name for name, fn in _MASK_FUNCS_2}

MAX_DENSE_SEQ = 20000
# ============================================================================
# 分块参考实现：不物化全量稠密 mask，支持最大 1M 序列长度
# ============================================================================
def _sdpa_chunked_reference(q, k, v, mask_func, problem, dropout_rate=0.0, q_chunk=None):
    """q/k/v 为 [B, H, S, D] (head-first), 返回 [B, S, H, D]。

    分块计算参考注意力，不物化 [S, S] 全量稠密 mask，因此大 seq（1M）也不会 OOM。

    关键优化：q_idx 用 [cb,1]、kv_idx 用 [1,S]，mask_mod 内部的 gather 和比较
    自动广播到 [cb,S]，但 gather 次数从 cb×S 降为 cb+S，显存占用大幅降低。
    q_chunk 按当前空闲显存自适应，整个循环在 no_grad 下执行避免计算图累积中间张量。
    """
    mask_mod = mask_func(problem)

    B, H, S, D = q.shape
    device = q.device
    if q_chunk is None:
        try:
            free_bytes = torch.npu.mem_get_info()[0]
        except Exception:
            free_bytes = int(8 * (1 << 30))
        q_chunk = max(64, min(512, int(0.15 * free_bytes / (12 * S))))
        print(f"  [chunked_ref] S={S}, free={free_bytes/1e9:.1f}GB, q_chunk={q_chunk}", flush=True)
    chunks = []
    n_chunks = (S + q_chunk - 1) // q_chunk
    with torch.no_grad():
        for ci, qs in enumerate(range(0, S, q_chunk)):
            if ci % 200 == 0:
                print(f"  [chunked_ref] chunk {ci}/{n_chunks}, qs={qs}", flush=True)
            qe = min(qs + q_chunk, S)
            cb = qe - qs
            q_idx = torch.arange(qs, qe, device=device, dtype=torch.int32)[:, None]
            kv_idx = torch.arange(0, S, device=device, dtype=torch.int32)[None, :]
            m = mask_mod(0, 0, q_idx, kv_idx)  # [cb,S] bool on NPU (broadcast)
            col_any = m.any(dim=0)
            nz = col_any.nonzero(as_tuple=False)
            if nz.numel() == 0:
                chunks.append(q.new_zeros((B, H, cb, D)))
                del m, q_idx, kv_idx, col_any, nz
                continue
            kmin = int(nz[0].item())
            kmax = int(nz[-1].item()) + 1
            attn = F.scaled_dot_product_attention(
                q[:, :, qs:qe], k[:, :, kmin:kmax], v[:, :, kmin:kmax],
                attn_mask=m[:, kmin:kmax][None, None, :, :],
                dropout_p=dropout_rate, enable_gqa=False,
            )
            chunks.append(attn)
            del m, q_idx, kv_idx, col_any, nz
            if ci % 50 == 49:
                torch.npu.empty_cache()
    out = torch.cat(chunks, dim=2)
    return out.transpose(1, 2).contiguous()


def _sdpa_chunked_reference_backward(q, k, v, mask_func, problem, grad_output, q_chunk=512):
    """分块重计算前向并立即反向，逐块累积 q/k/v 梯度。

    避免同时保留所有 chunk 的 attention 权重（前向 no_grad 已丢弃），每个 chunk
    只重算一次前向（带计算图）并立即 backward，中间张量约 100MB/块。

    q/k/v: [B, H, S, D] (head-first)
    grad_output: [B, S, H, D] 流回 ref_output 的梯度
    返回: gq, gk, gv [B, H, S, D]
    """
    mask_mod = mask_func(problem)
    B, H, S, D = q.shape
    device = q.device

    gq = torch.zeros_like(q)
    gk = torch.zeros_like(k)
    gv = torch.zeros_like(v)

    grad_attn_full = grad_output.transpose(1, 2).contiguous()  # [B, H, S, D]

    n_chunks = (S + q_chunk - 1) // q_chunk
    for ci, qs in enumerate(range(0, S, q_chunk)):
        if ci % 200 == 0:
            print(f"  [chunked_ref_bwd] chunk {ci}/{n_chunks}, qs={qs}", flush=True)
        qe = min(qs + q_chunk, S)
        cb = qe - qs
        q_idx = torch.arange(qs, qe, device=device, dtype=torch.int32)[:, None]
        kv_idx = torch.arange(0, S, device=device, dtype=torch.int32)[None, :]
        with torch.no_grad():
            m = mask_mod(0, 0, q_idx, kv_idx)
            col_any = m.any(dim=0)
            nz = col_any.nonzero(as_tuple=False)
        if nz.numel() == 0:
            del m, q_idx, kv_idx, col_any, nz
            continue
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1

        qc = q[:, :, qs:qe].detach().requires_grad_(True)
        kc = k[:, :, kmin:kmax].detach().requires_grad_(True)
        vc = v[:, :, kmin:kmax].detach().requires_grad_(True)
        m_slice = m[:, kmin:kmax][None, None, :, :]
        attn = F.scaled_dot_product_attention(
            qc, kc, vc,
            attn_mask=m_slice,
            dropout_p=0.0, enable_gqa=False,
        )
        grad_chunk = grad_attn_full[:, :, qs:qe, :]
        attn.backward(grad_chunk)
        gq[:, :, qs:qe] += qc.grad
        gk[:, :, kmin:kmax] += kc.grad
        gv[:, :, kmin:kmax] += vc.grad
        del m, q_idx, kv_idx, col_any, nz, m_slice, qc, kc, vc, attn, grad_chunk
        if ci % 50 == 49:
            torch.npu.empty_cache()
    return gq, gk, gv


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


# ============================================================================
# shape 组合扩充（data_lens 最大支持 1M）
# ============================================================================
# ============================================================================
# 随机用例生成（固定种子保证可复现，同时保证随机性），追加到 _SHAPE_CASES
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
    return batch, q_head, kv_head, hdim, data_lens, data_types, sliding, global_w, dtype


_RANDOM_CASES_2 = [
    pytest.param(*_make_random_case(_RNG), id=f"rand_{i:02d}")
    for i in range(10)
]

_SHAPE_CASES = [
    # ===== batch / head / dim 各种组合，seq 1k ~ 1M，长度取任意数值（非 10 整数倍） =====
    pytest.param(1, 16, 8, 128,
                 [[123, 4567, 89]], [["text", "image_gen", "text"]],
                 512, 4, torch.bfloat16, id="b1_h16kv8_d128_s4779"),
    pytest.param(2, 16, 8, 128,
                 [[1233, 4567], [891, 2345]], [["text", "image_gen"], ["text", "image_gen"]],
                 1024, 4, torch.bfloat16, id="b2_h16kv8_d128_s9036"),
    pytest.param(1, 32, 16, 128,
                 [[12345, 23456, 34567]], [["text", "image_gen", "text"]],
                 4096, 8, torch.bfloat16, id="b1_h32kv16_d128_s70368"),
    pytest.param(2, 16, 8, 64,
                 [[2345, 6789], [1111, 2222]], [["text", "image_gen"], ["text", "image_gen"]],
                 2048, 8, torch.float16, id="b2_h16kv8_d64_s12467"),
    pytest.param(1, 16, 8, 64,
                 [[10007, 20003, 30011]], [["text", "image_gen", "text"]],
                 65536, 16, torch.float16, id="b1_h16kv8_d64_s60021"),
    pytest.param(1, 32, 16, 128,
                 [[45678, 98765, 56789]], [["text", "image_gen", "text"]],
                 65536, 16, torch.bfloat16, id="b1_h32kv16_s201232"),
    pytest.param(2, 16, 8, 128,
                 [[12345, 23456], [34567, 45678]], [["text", "text"], ["text", "text"]],
                 8192, 8, torch.bfloat16, id="b2_h16kv8_s116046"),
    pytest.param(1, 16, 8, 128,
                 [[1000, 3000, 1000]], [["text", "image_gen", "text"]],
                 1024, 4, torch.bfloat16, id="b1_h16kv8_s5k"),
    pytest.param(2, 16, 8, 128,
                 [[2000, 22000, 2000], [2000, 22000, 2000]],
                 [["text", "image_gen", "text"], ["text", "image_gen", "text"]],
                 1024, 4, torch.bfloat16, id="b2_h16kv8_s52k"),
    pytest.param(1, 32, 16, 128,
                 [[50000], [50000]], [["text"], ["text"]],
                 65536, 16, torch.bfloat16, id="b1_h32kv16_s100k"),
    pytest.param(2, 16, 8, 64,
                 [[40000, 60000], [40000, 60000]], [["text", "text"], ["text", "text"]],
                 2048, 8, torch.float16, id="b2_h16kv8_d64_s200k"),
    pytest.param(1, 4, 2, 64,
                 [[200000], [200000]], [["text"], ["text"]],
                 65536, 16, torch.bfloat16, id="b1_h16kv8_s400k"),
    pytest.param(2, 4, 2, 64,
                 [[100000, 150000], [200000, 250000]], [["text", "text"], ["text", "text"]],
                 65536, 16, torch.bfloat16, id="b2_h32kv16_s700000"),
    pytest.param(1, 4, 2, 64,
                 [[333333, 333333, 333334]], [["text", "image_gen", "text"]],
                 65536, 16, torch.bfloat16, id="b1_h16kv8_s1M"),
    pytest.param(1, 4, 2, 64,
                 [[1000000]], [["text"]], 65536, 16, torch.bfloat16, id="b1_h16kv8_s1M_single"),
] + _RANDOM_CASES_2

_mask_func_param_2 = pytest.mark.parametrize(
    "mask_func",
    [fn for _, fn in _MASK_FUNCS_2],
    ids=[name for name, _ in _MASK_FUNCS_2],
)


@_mask_func_param_2
@pytest.mark.parametrize(
    "batch_size,q_head,kv_head,head_dim,data_lens,data_types,sliding_windows,global_windows,dtype",
    _SHAPE_CASES,
)
@pytest.mark.skipif(get_platform() != "npu", reason="FlexAttention TTX backend requires NPU")
@bypass_not_implemented
def test_flex_attention_2(batch_size, q_head, kv_head, head_dim, data_lens, data_types,
                          sliding_windows, global_windows, dtype, mask_func):
    problem = build_problem(batch_size, q_head, kv_head, head_dim, data_lens, data_types,
                            sliding_windows, global_windows, dtype, mask_func)

    q_base = problem["q"]
    k_base = problem["k"]
    v_base = problem["v"]

    # 参考输入复用 base（不 clone），mojo 输入在参考之后再创建，省显存
    q_ref = q_base.requires_grad_(True)
    k_ref = k_base.requires_grad_(True)
    v_ref = v_base.requires_grad_(True)

    SEQ_LEN = problem["total_s"]

    return_grid = torch.tensor(SEQ_LEN, dtype=dtype, device=torch.device(_device()))

    # 释放缓存碎片，给参考计算留出最大可用显存
    gc.collect()
    torch.npu.empty_cache()

    # 参考计算放在 mojo 前向之前：此时尚未创建 mojo 输入和 block_mask，显存占用最低
    if SEQ_LEN <= MAX_DENSE_SEQ:
        # 小序列：走全量稠密 mask 参考路径
        dense_mask = _build_dense_mask(mask_func, problem)
        _sync()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>dense_mask.sum().item()", dense_mask.to("cpu").sum().item())
        ref_output = _sdpa_with_dense_mask(q_ref, k_ref, v_ref, dense_mask, 0.0, None)
    else:
        # 大序列：用分块参考，避免物化 [S,S] 稠密 mask 导致 OOM
        n_element = _count_n_element(mask_func, problem)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>dense_mask.sum().item() (chunked)", n_element)
        ref_output = _sdpa_chunked_reference(q_ref, k_ref, v_ref, mask_func, problem, 0.0)
    _sync()

    # 参考完成后创建 mojo 输入
    q_mojo = q_base.detach().clone().requires_grad_(True)
    k_mojo = k_base.detach().clone().requires_grad_(True)
    v_mojo = v_base.detach().clone().requires_grad_(True)

    # 构建 block mask，供 mojo 前向使用
    packed_block_mask = _build_block_mask(mask_func, problem)
    _sync()

    mojo_output = _flex_attention_mojo(q_mojo, k_mojo, v_mojo, None, packed_block_mask, 0.0, None)
    _sync()

    # mojo 前向完成后释放 block mask 并清空缓存，为反向腾出显存
    del packed_block_mask
    gc.collect()
    torch.npu.empty_cache()

    assert mojo_output.shape == ref_output.shape
    torch.testing.assert_close(mojo_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    _sync()

    mojo_output.float().mean().backward(return_grid)
    _sync()

    if SEQ_LEN <= MAX_DENSE_SEQ:
        # 小序列：参考有完整计算图，直接反向
        ref_output.float().mean().backward(return_grid)
        _sync()
    else:
        # 大序列：参考前向在 no_grad 下计算，通过分块重算前向+反向获取梯度
        ref_numel = ref_output.numel()
        grad_out = torch.full_like(ref_output, return_grid.item() / ref_numel)
        _sync()
        gq, gk, gv = _sdpa_chunked_reference_backward(
            q_ref, k_ref, v_ref, mask_func, problem, grad_out)
        q_ref.grad = gq
        k_ref.grad = gk
        v_ref.grad = gv
        _sync()

    torch.testing.assert_close(q_mojo.grad.cpu(), q_ref.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k_mojo.grad.cpu(), k_ref.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v_mojo.grad.cpu(), v_ref.grad.cpu(), atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    import sys

    mask_map = dict(_MASK_FUNCS_2)
    mask_name = sys.argv[1] if len(sys.argv) > 1 else "causal"
    shape_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    _default_shape = _SHAPE_CASES[shape_idx].values[0]
    for name, fn in _MASK_FUNCS_2:
        if mask_name == "all" or mask_name == name:
            print(f"\n{'=' * 60}\nTesting mask: {name}, shape={_SHAPE_CASES[shape_idx].id}\n{'=' * 60}")
            test_flex_attention_2(
                batch_size=_default_shape[0], q_head=_default_shape[1], kv_head=_default_shape[2],
                head_dim=_default_shape[3], data_lens=_default_shape[4], data_types=_default_shape[5],
                sliding_windows=_default_shape[6], global_windows=_default_shape[7],
                dtype=_default_shape[8], mask_func=fn,
            )
            if mask_name != "all":
                break
