import gc
import pytest
import torch
import torch.nn.functional as F

from mojo_opset.experimental import mojo_flex_attention
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.backends.ttx.kernels.npu.utils import is_910
import torch_npu._inductor
from mojo_opset.backends.ttx.kernels.npu.flex_attention import _build_packed_block_mask_streaming
from mojo_opset.backends.ttx.kernels.npu.flex_attention import create_block_mask_patched
from mojo_opset.backends.ttx.kernels.npu.flex_attention import triton_create_mask
from mojo_opset.backends.ttx.kernels.npu.flex_attention import MASK_BLOCK_SIZE
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


# NPU device validation monkey-patch (same as original test)
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass

GEN_MASK_TRITON = False
USE_MOJO_FLEX_ATTENTION = False
FULL_MASK_MODALITIES = ("image_gen", "image_vae")

SEED = 0
APPLY_Q_CHUNK = 2048
Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128

def _device():
    return get_torch_device()
def _sync():
    if _device() == "npu":
        torch.npu.synchronize()
    elif _device() == "cuda":
        torch.cuda.synchronize()

# ============================================================================
# Mask function definitions
# ============================================================================
def _sparse_mask_mod(problem):
    segment_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= W)
        glob = causal & (kv_idx >= doc_start[q_idx]) & (kv_idx < doc_start[q_idx] + G)
        sparse = same_doc & (window | glob)
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return sparse | same_img
    return mask_mod


def _stair_mask_mod(problem):
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = video_ids[q_idx] == video_ids[kv_idx]
        frame_causal = frame_ids[q_idx] >= frame_ids[kv_idx]
        return same_doc & frame_causal
    return mask_mod

def _video_stair_mask_mod(problem):
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_video = video_ids[q_idx] == video_ids[kv_idx]
        same_frame = frame_ids[q_idx] == frame_ids[kv_idx]
        prev_frame = frame_ids[q_idx] > frame_ids[kv_idx]
        return same_video & (same_frame | prev_frame)
    return mask_mod

def _cross_sample_causal_video_bidir_mask_mod(problem):
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_video = modality[q_idx] > 0
        same_video = is_video & (modality[q_idx] == modality[kv_idx])
        return causal | same_video
    return mask_mod

def _full_mask_mod(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        samedoc_causal = same_doc & causal
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return samedoc_causal | same_img
    return mask_mod

_MASK_FUNCS = [
    ("sparse", _sparse_mask_mod),
    ("full", _full_mask_mod),
    ("stair", _stair_mask_mod),
    ("video_stair", _video_stair_mask_mod),
    ("cross_sample_causal_video_bidir", _cross_sample_causal_video_bidir_mask_mod),
]

_MASK_FUNC_TO_TYPE = {id(fn): name for name, fn in _MASK_FUNCS}

def _build_dense_mask(mask_func, problem):
    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]
    return triton_create_mask(problem, mask_type_str, tile_size=MASK_BLOCK_SIZE)

# ============================================================================
# Attention wrappers
# ============================================================================
def _flex_attention_mojo(q, k, v, mask, block_mask, dropout_rate=0.0, input_format=None):
    if input_format == "head-first":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    if mask is not None:
        block_mask.dense_mask = mask
    if USE_MOJO_FLEX_ATTENTION:
        output = mojo_flex_attention(q, k, v, block_mask=block_mask)
    else:
        flex_compiled = torch.compile(flex_attention, backend="inductor")
        output = flex_compiled(q, k, v, block_mask=block_mask,
                               enable_gqa=True, return_lse=False)                            
    return output.transpose(1, 2)

def _build_block_mask(mask_func, problem):
    SEQ_LEN = problem["total_s"]
    device=problem["q"].device
    if GEN_MASK_TRITON:
        classify_strategy= "fused" if not is_910() else "decoupled"
        mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]
        packed_block_mask = _build_packed_block_mask_streaming(mask_type_str, problem, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, classify_strategy=classify_strategy)
    else:
        if USE_MOJO_FLEX_ATTENTION:
            packed_block_mask = create_block_mask_patched(
                mask_func(problem), B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN,
                device=device, BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
            )
        else:
            packed_block_mask = create_block_mask(mask_func(problem),B=1, H=1, Q_LEN=SEQ_LEN, 
                KV_LEN=SEQ_LEN,device=device, BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE))
    return packed_block_mask

def _sdpa_with_dense_mask(query_states, key_states, value_states, attention_mask, dropout_rate, input_format):
    if input_format == "head-first":
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    mask_2d = attention_mask
    while mask_2d.dim() > 2:
        mask_2d = mask_2d[0]

    q_len_total = query_states.size(2)
    block_q = APPLY_Q_CHUNK if APPLY_Q_CHUNK is not None else q_len_total
    chunks = []
    for qs in range(0, q_len_total, block_q):
        qe = min(qs + block_q, q_len_total)
        row = mask_2d[qs:qe]
        col_any = row.any(dim=0)
        nz = col_any.nonzero(as_tuple=False)
        if nz.numel() == 0:
            chunks.append(query_states.new_zeros((query_states.size(0), query_states.size(1), qe - qs, query_states.size(3))))
            continue
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1
        chunks.append(
            F.scaled_dot_product_attention(
                query_states[:, :, qs:qe], key_states[:, :, kmin:kmax], value_states[:, :, kmin:kmax],
                attn_mask=row[None, None, :, kmin:kmax], dropout_p=dropout_rate, enable_gqa=False,
            )
        )
    return torch.cat(chunks, dim=2).transpose(1, 2).contiguous()

# 超过该长度时改用分块参考实现，避免物化全量稠密 mask 导致 OOM
MAX_DENSE_SEQ = 20000


def _sdpa_chunked_reference(q, k, v, mask_func, problem, dropout_rate=0.0, q_chunk=None):
    """q/k/v 为 [B, H, S, D] (head-first), 返回 [B, S, H, D]。

    分块计算参考注意力，不物化 [S, S] 全量稠密 mask，因此大 seq（1M）也不会 OOM。

    关键优化：q_idx 用 [cb,1]、kv_idx 用 [1,S]，mask_mod 内部的 gather 和比较
    自动广播到 [cb,S]，但 gather 次数从 cb×S 降为 cb+S，且 gather 结果仅 [cb,1] 和 [1,S]
    （不再是 [cb,S] int64），显存占用大幅降低。
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
        # 每块峰值约 cb×S×12 bytes（int32 差 4B + ~8 bool 中间量 8B）
        q_chunk = max(64, min(512, int(0.15 * free_bytes / (12 * S))))
        print(f"  [chunked_ref] S={S}, free={free_bytes/1e9:.1f}GB, q_chunk={q_chunk}", flush=True)
    chunks = []
    n_chunks = (S + q_chunk - 1) // q_chunk
    # 整个循环在 no_grad 下执行，防止 requires_grad=True 的 q/k/v 构建计算图
    # 导致中间 mask 和 attn 张量无法释放（这是大序列 OOM 的关键原因）
    with torch.no_grad():
        for ci, qs in enumerate(range(0, S, q_chunk)):
            if ci % 200 == 0:
                print(f"  [chunked_ref] chunk {ci}/{n_chunks}, qs={qs}", flush=True)
            qe = min(qs + q_chunk, S)
            cb = qe - qs
            # q_idx [cb,1] + kv_idx [1,S] → 广播到 [cb,S]，gather 只需 cb+S 次（非 cb×S）
            q_idx = torch.arange(qs, qe, device=device, dtype=torch.int32)[:, None]  # [cb,1]
            kv_idx = torch.arange(0, S, device=device, dtype=torch.int32)[None, :]  # [1,S]
            m = mask_mod(0, 0, q_idx, kv_idx)  # [cb,S] bool on NPU (broadcast)
            col_any = m.any(dim=0)
            nz = col_any.nonzero(as_tuple=False)
            if nz.numel() == 0:
                chunks.append(q.new_zeros((B, H, cb, D)))
                del m, q_idx, kv_idx, col_any, nz
                continue
            kmin = int(nz[0].item())
            kmax = int(nz[-1].item()) + 1
            # 只切需要的 mask 列做 SDPA
            attn = F.scaled_dot_product_attention(
                q[:, :, qs:qe], k[:, :, kmin:kmax], v[:, :, kmin:kmax],
                attn_mask=m[:, kmin:kmax][None, None, :, :],
                dropout_p=dropout_rate, enable_gqa=False,
            )
            chunks.append(attn)
            del m, q_idx, kv_idx, col_any, nz
            # 定期释放缓存碎片，防止 NPU caching allocator 碎片化
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

    # grad_output 是 [B, S, H, D]，转成 [B, H, S, D] 匹配 attn 输出
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

        # 前向（带计算图）：detach 切片后 requires_grad
        qc = q[:, :, qs:qe].detach().requires_grad_(True)
        kc = k[:, :, kmin:kmax].detach().requires_grad_(True)
        vc = v[:, :, kmin:kmax].detach().requires_grad_(True)
        m_slice = m[:, kmin:kmax][None, None, :, :]
        attn = F.scaled_dot_product_attention(
            qc, kc, vc,
            attn_mask=m_slice,
            dropout_p=0.0, enable_gqa=False,
        )  # [B, H, cb, D_k]
        grad_chunk = grad_attn_full[:, :, qs:qe, :]  # [B, H, cb, D]
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
    # 1D 索引张量转 int32，减少 gather 内存
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
        # q_idx [cb,1] + kv_idx [1,S] → 广播到 [cb,S]，gather 只需 cb+S 次（非 cb×S）
        q_idx = torch.arange(qs, qe, device=device, dtype=torch.int32)[:, None]  # [cb,1]
        kv_idx = torch.arange(0, S, device=device, dtype=torch.int32)[None, :]  # [1,S]
        with torch.no_grad():
            m = mask_mod(0, 0, q_idx, kv_idx)  # [cb,S] bool on NPU (broadcast)
        total += int(m.sum().item())
        del m, q_idx, kv_idx
    return total

# ============================================================================
# Data building
# ============================================================================
def _build_video_indicators(device,frame_lens):
    segment_ids, doc_start, video_ids, frame_ids, modality = [], [], [], [], []
    sample_start = 0
    next_video_id = 0

    for sample_id, sample_videos in enumerate(frame_lens):
        for frame_lens in sample_videos:
            cur_video_id = next_video_id
            next_video_id += 1
            for frame_id, frame_len in enumerate(frame_lens):
                segment_ids.append(torch.full((frame_len,), sample_id, dtype=torch.long))
                doc_start.append(torch.full((frame_len,), sample_start, dtype=torch.long))
                video_ids.append(torch.full((frame_len,), cur_video_id, dtype=torch.long))
                frame_ids.append(torch.full((frame_len,), frame_id, dtype=torch.long))
                modality.append(torch.full((frame_len,), cur_video_id + 1, dtype=torch.long))
        sample_start += sum(sum(fl) for fl in sample_videos)

    return {
        "segment_ids": torch.cat(segment_ids).to(device),
        "doc_start": torch.cat(doc_start).to(device),
        "video_ids": torch.cat(video_ids).to(device),
        "frame_ids": torch.cat(frame_ids).to(device),
        "modality": torch.cat(modality).to(device),
    }

def _build_modality_indicators(device, data_length=None, data_input_type=None, image_modalities=None):
    indicator = []
    iidx = 1
    for sample_types, sample_lens in zip(data_input_type, data_length):
        for sample_type, sample_len in zip(sample_types, sample_lens):
            if sample_type in image_modalities:
                indicator.append(torch.full((sample_len,), iidx, dtype=torch.long))
                iidx += 1
            else:
                indicator.append(torch.full((sample_len,), -1, dtype=torch.long))
    return torch.cat(indicator).to(device)

def build_problem(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func):
    device = _device()
    torch.manual_seed(SEED)

    num_q_heads = q_head
    num_kv_heads = kv_head

    sample_lens = [sum(s) for s in data_lens]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_s = int(cu_seqlens[-1].item())
    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=device, dtype=torch.int32),
        torch.tensor(sample_lens, device=device),
    )
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).to(torch.long)

    q = torch.rand(batch_size, num_q_heads, total_s, head_dim, device=device, dtype=dtype)
    k = torch.rand(batch_size, num_kv_heads, total_s, head_dim, device=device, dtype=dtype)
    v = torch.rand(batch_size, num_kv_heads, total_s, head_dim, device=device, dtype=dtype)

    if mask_func in [_video_stair_mask_mod, _stair_mask_mod]:
        meta = _build_video_indicators(device,data_types)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": meta["segment_ids"], "doc_start": meta["doc_start"],
            "video_ids": meta["video_ids"], "frame_ids": meta["frame_ids"], "modality": meta["modality"],
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": sliding_windows, "global_window": global_windows,
            "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        }
    else:
        modality = _build_modality_indicators(device=device,data_length=data_lens, 
                                              data_input_type=data_types, image_modalities=FULL_MASK_MODALITIES,)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": segment_ids.long(), "modality": modality, "doc_start": doc_start,
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": sliding_windows, "global_window": global_windows,
            "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        }


# ============================================================================
# Shared parametrize decorator for mask functions
# ============================================================================
_mask_func_param = pytest.mark.parametrize(
    "mask_func",
    [fn for _, fn in _MASK_FUNCS],
    ids=[name for name, _ in _MASK_FUNCS],
)

# ============================================================================
# 随机用例生成（固定种子保证可复现，同时保证随机性）
# ============================================================================
import random as _random

_RNG = _random.Random(2026)
_RAND_BATCH = [1]
_RAND_QHEAD = [16, 32]
_RAND_HDIM = [64, 128,256,512]
_RAND_DTYPES = [torch.bfloat16]
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


# ============================================================================
# 多样本随机用例（5-100 个样本，每个样本内段数和段长度均随机）
# ============================================================================
_RNG_MS = _random.Random(4242)
_MS_NSEG = [10, 15, 20, 100, 433, 1000]  # 每个样本内段数
_MS_BATCH = [1]
_MS_QHEAD = [16, 32]
_MS_HDIM = [64, 128, 256, 512]
_MS_SLIDE = [512, 1024, 4096]
_MS_GLOBAL = [4, 8, 16]
_MS_DTYPES = [torch.bfloat16]
_MS_MASKS = [_sparse_mask_mod, _full_mask_mod, _cross_sample_causal_video_bidir_mask_mod]
_MS_MIN_SEG = 1            # 单段最小长度
_MS_MAX_SEG = 1000000      # 单段最大长度（1 ~ 1M 随机）
_MS_TOTAL_BUDGET = 180000  # 总序列长度预算，防止 OOM


def _random_segs(rng, n_seg, budget):
    """随机生成 n_seg 个段长度，每段独立 randint(1, 1M)，总长度不超过 budget。

    若随机生成的总长超过 budget，则按比例缩放至 budget（保留各段比例差异）。
    """
    segs = [rng.randint(_MS_MIN_SEG, _MS_MAX_SEG) for _ in range(n_seg)]
    s = sum(segs)
    if s > budget:
        # 按比例缩放，保留各段相对差异
        segs = [max(_MS_MIN_SEG, int(x * budget / s)) for x in segs]
    rng.shuffle(segs)
    return segs


def _sample_budget(rng, n_seg, total_s):
    """计算单样本预算：budget = n_seg * avg，avg 在 [10, 2000] 随机。

    平衡段长度差异（1~1M 随机）和样本数量，受总预算约束。
    返回 (budget, ok)，ok=False 表示预算不足应停止。
    """
    avg = rng.randint(10, 2000)
    wanted = n_seg * avg
    remaining = _MS_TOTAL_BUDGET - total_s
    # 至少保证每段有 _MS_MIN_SEG，否则停止
    if remaining < n_seg * _MS_MIN_SEG:
        return 0, False
    budget = min(wanted, remaining)
    return budget, True


def _make_multi_sample_case(rng):
    """生成 5-100 个样本的随机用例，每个样本内段数从 _MS_NSEG 随机选取，
    段长度完全随机生成（非固定池），每样本独立预算保证段长度有差异。
    """
    batch = rng.choice(_MS_BATCH)
    q_head = rng.choice(_MS_QHEAD)
    hdim = rng.choice(_MS_HDIM)
    kv_head = rng.choice([h for h in [2, 4, 8, 16] if h <= q_head])
    n_samples_wanted = rng.randint(5, 100)
    data_lens, data_types = [], []
    total_s = 0
    for _ in range(n_samples_wanted):
        n_seg = rng.choice(_MS_NSEG)
        budget, ok = _sample_budget(rng, n_seg, total_s)
        if not ok:
            break
        segs = _random_segs(rng, n_seg, budget)
        data_lens.append(segs)
        types = [rng.choice(["text", "image_gen"]) for _ in range(n_seg)]
        data_types.append(types)
        total_s += sum(segs)
    sliding = rng.choice(_MS_SLIDE)
    global_w = rng.choice(_MS_GLOBAL)
    dtype = rng.choice(_MS_DTYPES)
    mask = rng.choice(_MS_MASKS)
    print(f"data_lens {data_lens}")
    return batch, q_head, kv_head, hdim, data_lens, data_types, sliding, global_w, dtype, mask


_MULTI_SAMPLE_CASES = [
    pytest.param(*_make_multi_sample_case(_RNG_MS), id=f"msample_{i:02d}")
    for i in range(10)
]


# ============================================================================
# 混合段数多样本用例（每个样本内段数差异大：10/15/20 段循环混合，段长度随机）
# ============================================================================
_RNG_MIX = _random.Random(8888)
_MIX_BATCH = [1]
_MIX_QHEAD = [16, 32]
_MIX_HDIM = [64, 128, 256, 512]
_MIX_SLIDE = [512, 1024, 4096]
_MIX_GLOBAL = [4, 8, 16]
_MIX_DTYPES = [torch.bfloat16]
_MIX_MASKS = [_sparse_mask_mod, _full_mask_mod, _cross_sample_causal_video_bidir_mask_mod]


def _make_mixed_seg_case(rng):
    """生成多样本用例，强制每个样本内段数各不相同（10/15/20 段循环混合），
    段长度完全随机生成（非固定池），每样本独立预算保证段长度有差异。

    例如一个用例内可能有：
      [537, 2048, 89, ...], [12000, 47, 678, ...], [22, 9999, 345, ...], ...
    """
    batch = rng.choice(_MIX_BATCH)
    q_head = rng.choice(_MIX_QHEAD)
    hdim = rng.choice(_MIX_HDIM)
    kv_head = rng.choice([h for h in [2, 4, 8, 16] if h <= q_head])
    n_samples_wanted = rng.randint(5, 100)
    # 段数模式循环：10,15,20,10,15,20,... 保证段数多样且至少 10
    seg_pattern = [10, 15, 20]
    data_lens, data_types = [], []
    total_s = 0
    for i in range(n_samples_wanted):
        n_seg = seg_pattern[i % len(seg_pattern)]
        budget, ok = _sample_budget(rng, n_seg, total_s)
        if not ok:
            break
        segs = _random_segs(rng, n_seg, budget)
        data_lens.append(segs)
        types = [rng.choice(["text", "image_gen"]) for _ in range(n_seg)]
        data_types.append(types)
        total_s += sum(segs)
    sliding = rng.choice(_MIX_SLIDE)
    global_w = rng.choice(_MIX_GLOBAL)
    dtype = rng.choice(_MIX_DTYPES)
    mask = rng.choice(_MIX_MASKS)
    print(f"data_lens {data_lens}")
    return batch, q_head, kv_head, hdim, data_lens, data_types, sliding, global_w, dtype, mask


_MIXED_SEG_CASES = [
    pytest.param(*_make_mixed_seg_case(_RNG_MIX), id=f"mixseg_{i:02d}")
    for i in range(10)
]


# ============================================================================
# 共用固定用例（test_flex_attention_3 与 test_flex_attention_mfu_3 共享）
# 注意：sparse_b1_s1M 因精度/性能需求不同，各自单独定义
# ============================================================================
_COMMON_FIXED_CASES = [
    # ===== sparse（前 3 个，sparse_b1_s1M 由各文件单独定义） =====
    pytest.param(1, 16, 8, 128, [[123, 4567, 89]], [["text", "image_gen", "text"]],
                 512, 4, torch.bfloat16, _sparse_mask_mod, id="sparse_b1_s5k"),
    pytest.param(1, 16, 8, 128, [[1233, 4567], [891, 2345]], [["text", "image_gen"], ["text", "image_gen"]],
                 1024, 4, torch.bfloat16, _sparse_mask_mod, id="sparse_b2_s9k"),
    pytest.param(1, 32, 16, 128, [[12345, 23456, 34567]], [["text", "image_gen", "text"]],
                 4096, 8, torch.bfloat16, _sparse_mask_mod, id="sparse_b1_s70k"),
    # ===== full =====
    pytest.param(1, 16, 8, 128, [[1233, 4567], [891, 2345]], [["text", "image_gen"], ["text", "image_gen"]],
                 1024, 4, torch.bfloat16, _full_mask_mod, id="full_b2_s9k"),
    pytest.param(1, 32, 16, 128, [[45678, 98765, 56789]], [["text", "image_gen", "text"]],
                 65536, 16, torch.bfloat16, _full_mask_mod, id="full_b1_s201k"),
    # ===== cross_sample_causal_video_bidir =====
    pytest.param(1, 16, 8, 128, [[10007, 20003]], [["text", "image_gen"]],
                 1024, 4, torch.bfloat16, _cross_sample_causal_video_bidir_mask_mod, id="cross_b1_s30k"),
    pytest.param(2, 16, 8, 64, [[2345, 6789], [1111, 2222]], [["text", "image_gen"], ["text", "image_gen"]],
                 2048, 8, torch.bfloat16, _cross_sample_causal_video_bidir_mask_mod, id="cross_b2_d64_s12k"),
    # ===== video_stair（帧长度拆分为任意数值，非 10 整数倍） =====
    pytest.param(1, 16, 8, 128, [[1234, 2345]], [[[600, 634], [1234, 1111]]],
                 1024, 4, torch.bfloat16, _video_stair_mask_mod, id="video_stair_s3579"),
    pytest.param(1, 32, 16, 128, [[12345, 23456, 34567]], [
                    [[1234, 2266, 8845], [3456, 7890, 12110], [5678, 9123, 19766]],
                ], 4096, 8, torch.bfloat16, _video_stair_mask_mod, id="video_stair_s70368"),
    # ===== stair =====
    pytest.param(1, 16, 8, 128, [[3500, 4100]], [[[1234, 2266], [987, 3113]]],
                 1024, 4, torch.bfloat16, _stair_mask_mod, id="stair_s7600"),
    pytest.param(1, 16, 8, 64, [[12345, 23456], [34567, 45678]], [
                    [[6000, 6345], [11111, 12345]],
                    [[11111, 23456], [22222, 23456]],
                ], 2048, 8, torch.bfloat16, _stair_mask_mod, id="stair_b2_s116046"),
    pytest.param(1, 16, 8, 128, [[333333, 333333, 333334]], [["text", "image_gen", "text"]],
                         65536, 16, torch.bfloat16, _sparse_mask_mod, id="sparse_b1_s1M"),
]



@pytest.mark.parametrize(
    "batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func,",
    [
        # ===== 精度测试专属用例（mfu 不使用） =====
        pytest.param(1, 16, 8, 128, [[2000, 22000, 2000], [2000, 22000, 2000]],[["text", "image_gen", "text"],
            ["text", "image_gen", "text"]], 1024, 4, torch.bfloat16, _sparse_mask_mod,id="sparse_2000_22000"),
        pytest.param(1, 16, 8, 128, [[2000, 22000, 2000], [2000, 22000, 2000]],[["text", "image_gen", "text"],
                    ["text", "image_gen", "text"]], 1024, 4, torch.bfloat16, _full_mask_mod,id="full_2000_22000"),
        pytest.param(1, 16, 8, 128, [[2000, 22000, 2000], [2000, 22000, 2000]],[["text", "image_gen", "text"],
                            ["text", "image_gen", "text"]], 1024, 4, torch.bfloat16, _cross_sample_causal_video_bidir_mask_mod,id="cross_2000_22000"),
        pytest.param(1, 16, 8, 128, [[6500, 6500, 6500, 6500], [6500, 6500, 6500, 6500]],[
                        [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
                        [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],]
                    , 1024, 4, torch.bfloat16, _video_stair_mask_mod,id="video_stair_6500"),
        pytest.param(1, 16, 8, 128, [[6500, 6500, 6500, 6500], [6500, 6500, 6500, 6500]],[
                        [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
                        [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],], 1024, 4, torch.bfloat16, _stair_mask_mod,id="stair_6500"),
    ]
    +_COMMON_FIXED_CASES
    + _RANDOM_CASES + _MIXED_SEG_CASES + _MULTI_SAMPLE_CASES
)
@pytest.mark.skipif(get_platform() != "npu", reason="FlexAttention TTX backend requires NPU")
@bypass_not_implemented
def test_flex_attention(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func):
    problem = build_problem(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func)

    # 复用 base 张量作为参考输入，避免每个 q/k/v 各保留 3 份拷贝。
    # 1M/16 头/128 维下每份约 4GiB，3 份共 36GiB，是 60GiB 卡 OOM 的主因。
    q_base = problem["q"]
    k_base = problem["k"]
    v_base = problem["v"]

    # 参考输入复用 base（不 clone），mojo 输入在参考之后再创建，省 ~1GB 显存
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
        # grad_output = d(mean(ref_output.float())) * return_grid = return_grid / numel
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

    mask_map = dict(_MASK_FUNCS)
    name = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    if name == "all":
        for n, fn in _MASK_FUNCS:
            print(f"\n{'=' * 60}")
            print(f"Testing: {n}")
            print(f"{'=' * 60}")
            test_flex_attention(fn)
    else:
        test_flex_attention(mask_map[name])

