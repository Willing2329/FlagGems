import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)

# gru.input shapes: (batch, seq, input, hidden, layers). has_biases, bidirectional
# and batch_first are each swept over {True, False} by the parametrize decorators
# below, so only the shape dimension is listed here. Together the shapes exercise
# the three recurrence dispatch branches and their tile boundaries: batch == 1
# routes through the GEMV kernel; small grids take the persistent (grid-wide
# barrier) kernel; grids past the co-residency limit fall back to a per-step
# launch. Tile boundaries hit hidden == 32 (one full H tile), 64 (two H tiles),
# 128 (two K tiles) and input_size > 64 (input-GEMM K tiling), plus grids past
# the persistent co-residency cap (per-step path), even/odd sequence parity, and
# 2-3 stacked layers.
_GRU_INPUT_SHAPES = [
    # torch-alignment shape (persistent path).
    (6, 7, 10, 6, 2),
    # batch == 1 -> GEMV, narrow and wide hidden.
    (1, 7, 10, 6, 2),
    (1, 7, 10, 128, 2),
    # grid > 256 programs -> per-step launch.
    (256, 5, 64, 512, 1),
    # hidden == _BLOCK_H_MAX: exactly one full H tile (mask is all-true).
    (4, 7, 10, 32, 1),
    # persistent path with hidden spanning two H tiles.
    (4, 7, 10, 64, 1),
    # persistent path with hidden spanning four H tiles and two K tiles.
    (4, 7, 10, 128, 1),
    # grid == 256: past the persistent co-residency cap -> per-step.
    (128, 3, 32, 512, 1),
    # grid == 272 programs: also per-step.
    (136, 3, 32, 512, 1),
    # three stacked layers.
    (6, 7, 10, 6, 3),
    # even sequence length: final hidden state read from the opposite double buffer.
    (6, 2, 10, 6, 2),
    # input_size > _GEMM_BLOCK_K: input GEMM loops over multiple K tiles.
    (4, 7, 128, 32, 1),
]


@pytest.mark.gru
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_biases", [True, False])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize(
    ("batch_size", "seq_len", "input_size", "hidden_size", "num_layers"),
    _GRU_INPUT_SHAPES,
)
def test_gru(
    dtype,
    has_biases,
    bidirectional,
    batch_first,
    batch_size,
    seq_len,
    input_size,
    hidden_size,
    num_layers,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    gru = torch.nn.GRU(
        input_size,
        hidden_size,
        num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=flag_gems.device, dtype=dtype)

    input_shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    input = torch.randn(input_shape, device=flag_gems.device, dtype=dtype)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, device=flag_gems.device, dtype=dtype)
    params = tuple(gru._flat_weights)

    # The cuDNN reference uses TF32 for its fp32 GRU GEMMs by default, which lands
    # ~2e-4 from true fp32; FlagGems accumulates in full fp32, so disable TF32 to
    # keep the fp32 comparison meaningful at 1e-4.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    ref_input = utils.to_reference(input)
    ref_h0 = utils.to_reference(h0)
    ref_params = tuple(utils.to_reference(param) for param in params)
    ref_out, ref_hn = torch.gru(
        ref_input,
        ref_h0,
        ref_params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )

    with flag_gems.use_gems():
        res_out, res_hn = torch.gru(
            input,
            h0,
            params,
            has_biases,
            num_layers,
            0.0,
            False,
            bidirectional,
            batch_first,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_hn, ref_hn, dtype)


# Packed variable-length (gru.data) cases: (input, hidden, layers, lengths).
# has_biases is swept over {True, False} by the parametrize decorator below, so it
# is not listed here. batch_sizes is non-increasing; rows whose sequence has
# already ended are frozen at their last hidden value. Covers strictly-decreasing
# lengths, ties (plateaus), batch == 1 (GEMV), a grid >256 programs (per-step),
# and a hidden spanning two H tiles under the packed freeze mask. Packed input is
# unidirectional only.
_GRU_DATA_CASES = [
    (10, 6, 2, [7, 6, 5, 4, 3, 2]),
    (10, 6, 2, [7, 7, 5, 5, 3, 2]),
    (10, 6, 1, [7, 6, 5, 4, 3, 2]),
    (10, 6, 2, [7]),
    (10, 128, 2, [7]),
    (64, 512, 1, [5] * 128 + [3] * 128),
    # hidden == 64 -> persistent path with two H tiles under the packed freeze mask.
    (10, 64, 1, [7, 6, 5, 4, 3, 2]),
]

def _packed_fixture(dtype, input_size, hidden_size, num_layers, has_biases, lengths):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch = len(lengths)
    gru = torch.nn.GRU(
        input_size,
        hidden_size,
        num_layers,
        bias=has_biases,
        bidirectional=False,
    ).to(device=flag_gems.device, dtype=dtype)
    gru.flatten_parameters()
    padded = torch.randn(
        (batch, max(lengths), input_size), device=flag_gems.device, dtype=dtype
    )
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded,
        torch.tensor(lengths, dtype=torch.long),
        batch_first=True,
        enforce_sorted=True,
    )
    h0 = torch.randn(
        (num_layers, batch, hidden_size), device=flag_gems.device, dtype=dtype
    )
    return gru, packed, h0


@pytest.mark.gru
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_biases", [True, False])
@pytest.mark.parametrize(
    ("input_size", "hidden_size", "num_layers", "lengths"),
    _GRU_DATA_CASES,
)
def test_gru_data(dtype, input_size, hidden_size, num_layers, has_biases, lengths):
    gru, packed, h0 = _packed_fixture(
        dtype, input_size, hidden_size, num_layers, has_biases, lengths
    )
    params = tuple(gru._flat_weights)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    data = packed.data
    batch_sizes = packed.batch_sizes

    ref_data = utils.to_reference(data)
    ref_batch_sizes = utils.to_reference(batch_sizes)
    ref_h0 = utils.to_reference(h0)
    ref_params = tuple(utils.to_reference(param) for param in params)
    ref_out, ref_hn = torch.ops.aten.gru.data(
        ref_data,
        ref_batch_sizes,
        ref_h0,
        ref_params,
        has_biases,
        num_layers,
        0.0,
        False,
        False,
    )
    with flag_gems.use_gems():
        res_out, res_hn = torch.ops.aten.gru.data(
            data, batch_sizes, h0, params, has_biases, num_layers, 0.0, False, False
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_hn, ref_hn, dtype)
