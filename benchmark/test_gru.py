import pytest
import torch

import flag_gems

from . import base, consts

DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)

# The cuDNN reference uses TF32 for its fp32 GRU GEMMs by default, which is
# ~2x faster than true fp32 but lands ~2e-4 off; FlagGems accumulates in full
# fp32. Disable it so the fp32 latency comparison is precision-equivalent.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# gru.input shapes: (batch, seq, input, hidden, layers, bidirectional, batch_first,
# has_bias). A representative subset spanning the three recurrence dispatch paths
# (batch == 1 -> GEMV, small grid -> persistent, large grid -> per-step) plus one
# representative flip of each boolean axis.
_GRU_INPUT_SHAPES = [
    # batch == 1 -> GEMV kernel.
    (1, 512, 128, 128, 1, False, False, True),    # long seq
    (1, 10, 512, 512, 1, False, False, True),     # wide hidden, streaming
    # small batch -> persistent kernel.
    (16, 100, 512, 1024, 1, False, False, True),  # large hidden
    (64, 100, 512, 512, 1, False, False, True),   # canonical production shape
    (64, 100, 512, 512, 2, False, False, True),   # multi-layer
    (16, 128, 256, 256, 2, False, False, True),   # long seq + 2-layer
    # large batch -> per-step kernel.
    (256, 100, 512, 512, 1, False, False, True),
    # one representative flip of each boolean axis.
    (64, 100, 512, 512, 1, True, False, True),    # bidirectional
    (64, 100, 512, 512, 1, False, True, True),    # batch_first
    (64, 100, 512, 512, 1, False, False, False),  # no bias
    # large model capacity.
    (64, 100, 512, 2048, 1, False, False, True),  # large hidden
    # ASR acoustic model (small input, long sequence).
    (8, 1000, 80, 512, 1, False, False, True),
]

# gru.data shape tuple: (batch, seq_len, input, hidden, layers, has_biases).
# Packed GRU is unidirectional only; lengths are generated non-increasing inside
# gru_data_input_fn.
_GRU_DATA_SHAPES = [
    # Core sanity, mirroring the correctness-test packed shapes.
    (6, 7, 10, 6, 2, True),
    (6, 7, 10, 6, 2, False),
    # Production-scale packed inference (mirrors gru.input).
    (64, 100, 512, 512, 1, False),
    (64, 100, 512, 512, 2, False),
    (16, 128, 256, 256, 2, True),
    (1, 512, 128, 128, 1, False),    # batch 1 -> GEMV
    (16, 100, 512, 1024, 1, False),
    (256, 100, 512, 512, 1, False),  # large batch -> per-step
    # Streaming / low-latency.
    (1, 10, 512, 512, 1, False),
    # ASR (small input, long sequence).
    (8, 1000, 80, 512, 1, False),
]


class GRUBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _GRU_INPUT_SHAPES


class GRUDataBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _GRU_DATA_SHAPES


def gru_input_fn(shape, dtype, device):
    (
        batch_size,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        batch_first,
        has_biases,
    ) = shape

    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=device, dtype=dtype)
    gru.flatten_parameters()

    input_shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    input = torch.randn(input_shape, dtype=dtype, device=device)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, dtype=dtype, device=device)
    params = tuple(gru._flat_weights)

    yield (
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


def gru_data_input_fn(shape, dtype, device):
    (batch_size, seq_len, input_size, hidden_size, num_layers, has_biases) = shape

    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=False,
    ).to(device=device, dtype=dtype)
    gru.flatten_parameters()

    # Non-increasing lengths (sorted descending) so batch_sizes is a valid packed
    # prefix-sum; clamping to 1 creates ties at the tail that exercise the freeze
    # path in the packed recurrence.
    lengths = [max(1, seq_len - i) for i in range(batch_size)]
    padded = torch.randn(
        (batch_size, seq_len, input_size), dtype=dtype, device=device
    )
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded,
        torch.tensor(lengths, dtype=torch.long),
        batch_first=True,
        enforce_sorted=True,
    )
    h0 = torch.randn(
        (num_layers, batch_size, hidden_size), dtype=dtype, device=device
    )
    params = tuple(gru._flat_weights)

    yield (
        packed.data,
        packed.batch_sizes,
        h0,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        False,
    )


@pytest.mark.gru
def test_gru():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bench = GRUBenchmark(
        input_fn=gru_input_fn,
        op_name="gru",
        torch_op=torch.gru,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.gru
def test_gru_data():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bench = GRUDataBenchmark(
        input_fn=gru_data_input_fn,
        op_name="gru.data",
        torch_op=torch.ops.aten.gru.data,
        dtypes=DTYPES,
    )
    bench.run()
