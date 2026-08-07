import torch
import torch.nn.functional as F
import pytest

from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.backends.ttx.kernels.npu.flex_attention import create_block_mask_patched
from mojo_opset.backends.ttx.kernels.npu.flex_attention import MASK_BLOCK_SIZE

from mojo_opset.tests.accuracy.functions.test_flex_attention import _device
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


# ============================================================================
# 分块参考实现：不物化全量稠密 mask，支持最大 1M 序列长度
# ============================================================================
def _sdpa_chunked_reference(q, k, v, mask_func, problem, dropout_rate=0.0, q_chunk=1024):
    """q/k/v 为 [B, H, S, D] (head-first), 返回 [B, S, H, D]。"""
    mask_mod = mask_func(problem)
    B, H, S, D = q.shape
    device = q.device
    chunks = []
    for qs in range(0, S, q_chunk):
        qe = min(qs + q_chunk, S)
        cb = qe - qs
        q_idx = torch.arange(qs, qe, device=device)[:, None].expand(cb, S)
        kv_idx = torch.arange(0, S, device=device)[None, :].expand(cb, S)
        m = mask_mod(0, 0, q_idx, kv_idx)  # [cb, S] bool
        col_any = m.any(dim=0)
        nz = col_any.nonzero(as_tuple=False)
        if nz.numel() == 0:
            chunks.append(q.new_zeros((B, H, cb, D)))
            continue
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1
        attn = F.scaled_dot_product_attention(
            q[:, :, qs:qe], k[:, :, kmin:kmax], v[:, :, kmin:kmax],
            attn_mask=m[:, kmin:kmax][None, None, :, :],
            dropout_p=dropout_rate, enable_gqa=False,
        )
        chunks.append(attn)
    out = torch.cat(chunks, dim=2)
    return out.transpose(1, 2).contiguous()


def _count_n_element(mask_func, problem, q_chunk=1024):
    """统计 mask 激活元素个数（分块，避免物化全量稠密 mask）。"""
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
    q_head = rng.choice(_RAND_QHEAD)
    kv_head = rng.choice([h for h in [8, 16] if h <= q_head])
    hdim = rng.choice(_RAND_HDIM)
    n_samples = rng.randint(1, 3)
    mag = rng.choice(_RAND_MAG)
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
    pytest.param(1, 16, 8, 128,
                 [[200000], [200000]], [["text"], ["text"]],
                 65536, 16, torch.bfloat16, id="b1_h16kv8_s400k"),
    pytest.param(2, 32, 16, 128,
                 [[100000, 150000], [200000, 250000]], [["text", "text"], ["text", "text"]],
                 65536, 16, torch.bfloat16, id="b2_h32kv16_s700000"),
    pytest.param(1, 16, 8, 128,
                 [[333333, 333333, 333334]], [["text", "image_gen", "text"]],
                 65536, 16, torch.bfloat16, id="b1_h16kv8_s1M"),
    pytest.param(1, 16, 8, 128,
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

    q_mojo = q_base.detach().clone().requires_grad_(True)
    k_mojo = k_base.detach().clone().requires_grad_(True)
    v_mojo = v_base.detach().clone().requires_grad_(True)

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)

    SEQ_LEN = problem["total_s"]

    packed_block_mask = _build_block_mask(mask_func, problem)
    _sync()

    return_grid = torch.tensor(SEQ_LEN, dtype=dtype, device=torch.device(_device()))

    mojo_output = _flex_attention_mojo(q_mojo, k_mojo, v_mojo, None, packed_block_mask, 0.0, None)
    _sync()

    ref_output = _sdpa_chunked_reference(q_ref, k_ref, v_ref, mask_func, problem, 0.0)
    _sync()

    assert mojo_output.shape == ref_output.shape
    torch.testing.assert_close(mojo_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    _sync()

    mojo_output.float().mean().backward(return_grid)
    _sync()
    ref_output.float().mean().backward(return_grid)
    _sync()
    print(f"q_mojo.grad {q_mojo.grad}")
    print(f"q_ref.grad {q_ref.grad}")
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
