# Copyright 2027 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import types

import torch
import triton
import triton.language as tl

from flag_gems.ops.gru import (
    _bias_stride,
    _block_size,
    _ceil_power_of_2,
    _copy_hx_slice,
    _empty,
    _gru_forward_impl as _generic_gru_forward_impl,
    _gru_gemv_kernel,
    _gru_input_gemm_kernel as _generic_input_gemm,
    _max_persistent_programs,
    _param_group,
    _store_hx_slice,
    _transpose_weight,
    _validate_weight,
    gru as _generic_gru,
    gru_data as _generic_gru_data,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# MetaX GRU override of flag_gems/ops/gru.py: the flagtree triton fork
# miscompiles tl.dot (fp64 garbage, flaky pipelined fp32) and its AABS
# autotuner can shrink BLOCK_K below the dot's K >= 16 minimum. Workarounds:
# NO_DOT FMA kernels for fp64, num_stages=1 + BLOCK_H=16 (32 if hidden >=
# 1024) for the fp32 recurrence, and a dot-free input GEMM for fp64 and
# input_size <= 8. Everything else is imported from the generic file; the
# remaining behavioral divergences (validated on C550) are the autotuned
# input GEMM (8-22% faster than the generic's fixed tile) and the pow2(batch)
# persistent-batch shrink (the generic's fixed tile miscompiles on small
# batch).

_BLOCK_B = 16
_BLOCK_H_MAX = 32
_BLOCK_K_MAX = 64
_GEMV_BLOCK_N_MAX = 16
_GEMV_BLOCK_K_MAX = 128
_GEMV_HOIST_BLOCK_N = 2
_GEMV_HOIST_NUM_WARPS = 1
_GRU_INPUT_GEMM_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
]


def _fma_block_k(size: int, cap: int) -> int:
    # K chunk for the FMA fallbacks, capped to stay register-resident.
    return min(cap, max(8, _ceil_power_of_2(size)))


# The generic input GEMM kernel body is identical; re-wrap it with the metax
# autotune configs (kept over the generic's fixed tile, see the header).
_gru_input_gemm_kernel = libentry()(
    triton.autotune(
        _GRU_INPUT_GEMM_CONFIGS,
        key=["input_size", "hidden_size", "batch_size"],
    )(_generic_input_gemm.jit_function)
)


@libentry()
@triton.jit
def _gru_input_gemm_fma_kernel(
    x_ptr,
    w_ih_ptr,
    b_ih_ptr,
    u_ptr,
    batch_sizes_ptr,
    input_size,
    hidden_size,
    batch_size,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    w_ih_stride_r,
    w_ih_stride_c,
    b_ih_stride,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    PACKED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Dot-free input GEMM: fp64 tl.dot is broken, and for input_size <= 8 the
    # autotuner can shrink BLOCK_K below the dot's K >= 16 minimum.
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size

    if PACKED:
        bs_t = tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        if pid_b * BLOCK_B >= bs_t:
            out_offsets = (
                seq_idx * u_stride_s
                + offs_b[:, None] * u_stride_b
                + offs_n[None, :] * u_stride_f
            )
            tl.store(
                u_ptr + out_offsets,
                tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE),
                mask=b_mask[:, None] & n_mask[None, :],
            )
            return

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for k_block in range(0, tl.cdiv(input_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr
            + seq_idx * x_stride_s
            + offs_b[:, None] * x_stride_b
            + offs_k[None, :] * x_stride_f,
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < input_size),
            other=0.0,
        )
        w = tl.load(
            w_ih_ptr
            + offs_k[:, None] * w_ih_stride_r
            + offs_n[None, :] * w_ih_stride_c,
            mask=(offs_k[:, None] < input_size) & (offs_n[None, :] < 3 * hidden_size),
            other=0.0,
        )
        acc += tl.sum(x[:, :, None] * w[None, :, :], axis=1)

    if HAS_BIAS:
        b = tl.load(b_ih_ptr + offs_n * b_ih_stride, mask=n_mask, other=0.0)
        acc += b[None, :]

    out_offsets = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_n[None, :] * u_stride_f
    )
    tl.store(u_ptr + out_offsets, acc, mask=b_mask[:, None] & n_mask[None, :])


@libentry()
@triton.jit
def _gru_step_kernel(
    u_ptr,
    h_prev_ptr,
    w_hh_ptr,
    b_hh_ptr,
    h_next_ptr,
    out_ptr,
    batch_sizes_ptr,
    seq_idx,
    out_feature_offset,
    hidden_size,
    batch_size,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
    NO_DOT: tl.constexpr = False,
):
    # One recurrence step; NO_DOT switches the hidden GEMM to element-wise
    # FMA (fp64 tl.dot is broken on the toolchain).
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)

    if PACKED:
        bs_t = tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        if pid_b * BLOCK_B >= bs_t:
            state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
            out_offsets = (
                seq_idx * out_stride_s
                + offs_b[:, None] * out_stride_b
                + (out_feature_offset + offs_h[None, :]) * out_stride_f
            )
            h_prev_tile = tl.load(
                h_prev_ptr + state_offsets, mask=bh_mask, other=0.0
            )
            tl.store(h_next_ptr + state_offsets, h_prev_tile, mask=bh_mask)
            tl.store(out_ptr + out_offsets, h_prev_tile, mask=bh_mask)
            return

    u_base = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_h[None, :] * u_stride_f
    )
    r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    z_acc = tl.load(
        u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_in = tl.load(
        u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

    for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        h = tl.load(
            h_prev_ptr + offs_b[:, None] * hidden_size + offs_k[None, :],
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
            other=0.0,
        )
        w_r = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_h[None, :] * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        w_z = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        w_n = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        if NO_DOT:
            r_acc += tl.sum(h[:, :, None] * w_r[None, :, :], axis=1)
            z_acc += tl.sum(h[:, :, None] * w_z[None, :, :], axis=1)
            n_h_acc += tl.sum(h[:, :, None] * w_n[None, :, :], axis=1)
        else:
            r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b_hr = tl.load(b_hh_ptr + offs_h * b_hh_stride, mask=offs_h < hidden_size, other=0.0)
        b_hz = tl.load(
            b_hh_ptr + (hidden_size + offs_h) * b_hh_stride,
            mask=offs_h < hidden_size,
            other=0.0,
        )
        b_hn = tl.load(
            b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
            mask=offs_h < hidden_size,
            other=0.0,
        )
        r_acc += b_hr[None, :]
        z_acc += b_hz[None, :]
        n_h_acc += b_hn[None, :]

    r_gate = tl.sigmoid(r_acc)
    z_gate = tl.sigmoid(z_acc)
    n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

    h_prev = tl.load(
        h_prev_ptr + offs_b[:, None] * hidden_size + offs_h[None, :],
        mask=bh_mask,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

    if PACKED:
        active = offs_b < tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        h_next = tl.where(active[:, None], h_next, h_prev)

    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    out_offsets = (
        seq_idx * out_stride_s
        + offs_b[:, None] * out_stride_b
        + (out_feature_offset + offs_h[None, :]) * out_stride_f
    )
    tl.store(h_next_ptr + state_offsets, h_next, mask=bh_mask)
    tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)


@libentry()
@triton.jit
def _gru_persistent_kernel(
    u_ptr,
    h_ptr,
    w_hh_ptr,
    b_hh_ptr,
    out_ptr,
    barrier_ptr,
    batch_sizes_ptr,
    out_feature_offset,
    hidden_size,
    batch_size,
    seq_len,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    HAS_BIAS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
    NO_DOT: tl.constexpr = False,
):
    # Persistent recurrence with a grid-wide barrier; NO_DOT switches the
    # hidden GEMM to element-wise FMA.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)
    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    buf_stride = batch_size * hidden_size

    h_cur = tl.load(h_ptr + state_offsets, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)

    for step in range(seq_len):
        t = seq_len - 1 - step if REVERSE else step
        read_base = (step % 2) * buf_stride
        write_base = ((step + 1) % 2) * buf_stride

        u_base = (
            t * u_stride_s
            + offs_b[:, None] * u_stride_b
            + offs_h[None, :] * u_stride_f
        )
        r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
        z_acc = tl.load(
            u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
        ).to(COMPUTE_DTYPE)
        n_in = tl.load(
            u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
        ).to(COMPUTE_DTYPE)
        n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

        for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            h = tl.load(
                h_ptr + read_base + offs_b[:, None] * hidden_size + offs_k[None, :],
                mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
                other=0.0,
            )
            w_r = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + offs_h[None, :] * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            w_z = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            w_n = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            if NO_DOT:
                r_acc += tl.sum(h[:, :, None] * w_r[None, :, :], axis=1)
                z_acc += tl.sum(h[:, :, None] * w_z[None, :, :], axis=1)
                n_h_acc += tl.sum(h[:, :, None] * w_n[None, :, :], axis=1)
            else:
                r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
                z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
                n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

        if HAS_BIAS:
            b_hr = tl.load(b_hh_ptr + offs_h * b_hh_stride, mask=offs_h < hidden_size, other=0.0)
            b_hz = tl.load(
                b_hh_ptr + (hidden_size + offs_h) * b_hh_stride,
                mask=offs_h < hidden_size,
                other=0.0,
            )
            b_hn = tl.load(
                b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
                mask=offs_h < hidden_size,
                other=0.0,
            )
            r_acc += b_hr[None, :]
            z_acc += b_hz[None, :]
            n_h_acc += b_hn[None, :]

        r_gate = tl.sigmoid(r_acc)
        z_gate = tl.sigmoid(z_acc)
        n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)
        h_next = (1.0 - z_gate) * n_gate + z_gate * h_cur

        if PACKED:
            active = offs_b < tl.load(batch_sizes_ptr + t).to(tl.int32)
            h_next = tl.where(active[:, None], h_next, h_cur)

        tl.store(h_ptr + write_base + state_offsets, h_next, mask=bh_mask)
        out_offsets = (
            t * out_stride_s
            + offs_b[:, None] * out_stride_b
            + (out_feature_offset + offs_h[None, :]) * out_stride_f
        )
        tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)

        h_cur = h_next

        # Grid-wide barrier: release this program's h write, then acquire
        # everyone else's before the next step reads h.
        tl.atomic_add(barrier_ptr + step, 1, sem="acq_rel")
        while tl.atomic_add(barrier_ptr + step, 0, sem="acquire") < NUM_PROGRAMS:
            pass


def _run_direction(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx: int,
    param_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
    batch_sizes=None,
):
    w_ih, w_hh, b_ih, b_hh = _param_group(params, param_idx, has_biases)
    _validate_weight(w_ih, 3 * hidden_size, input_size)
    _validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    # Transposed (K, 3H) so the B operand's gate dim is contiguous.
    w_ih = _transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    if batch_size == 0:
        return

    block_k_h = _block_size(hidden_size, _BLOCK_K_MAX)
    # pow2(batch) shrink over the generic's fixed _BLOCK_B: the fixed tile
    # miscompiles on small batch.
    block_b_persist = min(_BLOCK_B, _ceil_power_of_2(batch_size))
    block_h_persist = _block_size(hidden_size, _BLOCK_H_MAX // 2)
    grid_persist = (
        triton.cdiv(batch_size, block_b_persist),
        triton.cdiv(hidden_size, block_h_persist),
    )
    num_programs_persist = grid_persist[0] * grid_persist[1]

    if layer_input.dtype == torch.float64:
        compute_dtype = tl.float64
        gate_dtype = torch.float64
    else:
        compute_dtype = tl.float32
        gate_dtype = torch.float32

    max_persistent = _max_persistent_programs(layer_input.device)

    # Input-side pre-activations for every timestep in one batched GEMM.
    input_gates = _empty(
        (seq_len, batch_size, 3 * hidden_size), gate_dtype, layer_input.device
    )
    input_gemm_grid = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_B"]),
        seq_len,
        triton.cdiv(3 * hidden_size, META["BLOCK_N"]),
    )
    use_fma_input = (
        compute_dtype == tl.float64 or _ceil_power_of_2(input_size) < 16
    )

    with torch_device_fn.device(layer_input.device):
        if use_fma_input:
            fma_block_b, fma_block_n = 4, 32
            fma_block_k = _fma_block_k(input_size, 128)
            fma_grid = (
                triton.cdiv(batch_size, fma_block_b),
                seq_len,
                triton.cdiv(3 * hidden_size, fma_block_n),
            )
            _gru_input_gemm_fma_kernel[fma_grid](
                layer_input,
                w_ih,
                b_ih,
                input_gates,
                batch_sizes if batch_sizes is not None else input_gates,
                input_size,
                hidden_size,
                batch_size,
                layer_input.stride(0),
                layer_input.stride(1),
                layer_input.stride(2),
                w_ih_stride_r,
                w_ih_stride_c,
                b_ih_stride,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                PACKED=batch_sizes is not None,
                HAS_BIAS=has_biases,
                BLOCK_B=fma_block_b,
                BLOCK_N=fma_block_n,
                BLOCK_K=fma_block_k,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=4,
                num_stages=1,
            )
        else:
            _gru_input_gemm_kernel[input_gemm_grid](
                layer_input,
                w_ih,
                b_ih,
                input_gates,
                batch_sizes if batch_sizes is not None else input_gates,
                input_size,
                hidden_size,
                batch_size,
                layer_input.stride(0),
                layer_input.stride(1),
                layer_input.stride(2),
                w_ih_stride_r,
                w_ih_stride_c,
                b_ih_stride,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                PACKED=batch_sizes is not None,
                HAS_BIAS=has_biases,
                COMPUTE_DTYPE=compute_dtype,
            )

        block_k_g = _block_size(hidden_size, _GEMV_BLOCK_K_MAX)
        gemv_hoist = hidden_size <= block_k_g
        gemv_block_n = (
            _GEMV_HOIST_BLOCK_N if gemv_hoist else _block_size(hidden_size, _GEMV_BLOCK_N_MAX)
        )
        gemv_num_programs = triton.cdiv(hidden_size, gemv_block_n)
        # Grow BLOCK_N until the barrier grid is co-resident (<= 1 CTA/SM).
        while gemv_num_programs > max_persistent and gemv_block_n < hidden_size:
            gemv_block_n *= 2
            gemv_num_programs = triton.cdiv(hidden_size, gemv_block_n)

        use_gemv = batch_size == 1 and gemv_num_programs <= max_persistent
        use_persistent = batch_size != 1 and num_programs_persist <= max_persistent
        no_dot_recur = compute_dtype == tl.float64

        if use_gemv:
            h_buf = _empty((2, batch_size, hidden_size), hx.dtype, hx.device)
            _copy_hx_slice(hx, h_buf[0], state_idx, batch_size, hidden_size)
            barrier = torch.zeros(
                (seq_len,), device=layer_input.device, dtype=torch.int32
            )
            _gru_gemv_kernel[(gemv_num_programs,)](
                input_gates,
                h_buf,
                w_hh,
                b_hh,
                layer_output,
                barrier,
                out_feature_offset,
                hidden_size,
                seq_len,
                input_gates.stride(0),
                input_gates.stride(2),
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                layer_output.stride(0),
                layer_output.stride(2),
                HAS_BIAS=has_biases,
                REVERSE=reverse,
                HOIST=gemv_hoist,
                BLOCK_N=gemv_block_n,
                BLOCK_K=block_k_g,
                NUM_PROGRAMS=gemv_num_programs,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=_GEMV_HOIST_NUM_WARPS if gemv_hoist else 4,
            )
            final_h_state = h_buf[seq_len % 2]
        elif use_persistent:
            h_buf = _empty((2, batch_size, hidden_size), hx.dtype, hx.device)
            _copy_hx_slice(hx, h_buf[0], state_idx, batch_size, hidden_size)
            barrier = torch.zeros(
                (seq_len,), device=layer_input.device, dtype=torch.int32
            )
            _gru_persistent_kernel[grid_persist](
                input_gates,
                h_buf,
                w_hh,
                b_hh,
                layer_output,
                barrier,
                batch_sizes if batch_sizes is not None else h_buf,
                out_feature_offset,
                hidden_size,
                batch_size,
                seq_len,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                layer_output.stride(0),
                layer_output.stride(1),
                layer_output.stride(2),
                HAS_BIAS=has_biases,
                REVERSE=reverse,
                BLOCK_B=block_b_persist,
                BLOCK_H=block_h_persist,
                BLOCK_K=_fma_block_k(hidden_size, 64) if no_dot_recur else block_k_h,
                NUM_PROGRAMS=num_programs_persist,
                COMPUTE_DTYPE=compute_dtype,
                PACKED=batch_sizes is not None,
                NO_DOT=no_dot_recur,
                **({"num_stages": 1, "num_warps": 8} if no_dot_recur else {}),
            )
            final_h_state = h_buf[seq_len % 2]
        else:
            # num_stages=1 + BLOCK_H=16 avoid the pipeliner miscompile; fp64
            # uses a narrow batch tile with a wide K chunk.
            if no_dot_recur:
                step_block_b, step_block_h = 4, 16
                step_block_k = _fma_block_k(hidden_size, 256)
                step_launch_kwargs = {"num_stages": 1, "num_warps": 8}
            else:
                step_block_b = _BLOCK_B
                step_block_h = 32 if hidden_size >= 1024 else 16
                step_block_k = block_k_h
                step_launch_kwargs = {"num_stages": 1}
            step_grid = (
                triton.cdiv(batch_size, step_block_b),
                triton.cdiv(hidden_size, step_block_h),
            )
            h_work = _empty((batch_size, hidden_size), hx.dtype, hx.device)
            _copy_hx_slice(hx, h_work, state_idx, batch_size, hidden_size)
            h_next = _empty((batch_size, hidden_size), hx.dtype, hx.device)
            for step in range(seq_len):
                seq_idx = seq_len - 1 - step if reverse else step
                _gru_step_kernel[step_grid](
                    input_gates,
                    h_work,
                    w_hh,
                    b_hh,
                    h_next,
                    layer_output,
                    batch_sizes if batch_sizes is not None else h_work,
                    seq_idx,
                    out_feature_offset,
                    hidden_size,
                    batch_size,
                    input_gates.stride(0),
                    input_gates.stride(1),
                    input_gates.stride(2),
                    w_hh_stride_r,
                    w_hh_stride_c,
                    b_hh_stride,
                    layer_output.stride(0),
                    layer_output.stride(1),
                    layer_output.stride(2),
                    HAS_BIAS=has_biases,
                    BLOCK_B=step_block_b,
                    BLOCK_H=step_block_h,
                    BLOCK_K=step_block_k,
                    COMPUTE_DTYPE=compute_dtype,
                    PACKED=batch_sizes is not None,
                    NO_DOT=no_dot_recur,
                    **step_launch_kwargs,
                )
                h_work, h_next = h_next, h_work
            final_h_state = h_work

    _store_hx_slice(final_h_state, final_h, state_idx, batch_size, hidden_size)


def _rehome(fn, **overrides):
    # Copy a generic-module function with selected globals bound to this
    # module, so the entry chain calls the metax _run_direction without
    # mutating the generic module.
    g = dict(fn.__globals__)
    g.update(overrides)
    f = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
    f.__doc__ = fn.__doc__
    return f


_gru_forward_impl = _rehome(_generic_gru_forward_impl, _run_direction=_run_direction)
gru = _rehome(_generic_gru, _gru_forward_impl=_gru_forward_impl, logger=logger)
gru_data = _rehome(_generic_gru_data, _gru_forward_impl=_gru_forward_impl, logger=logger)

del _generic_gru, _generic_gru_data, _generic_gru_forward_impl, _generic_input_gemm, _rehome
