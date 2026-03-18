"""
s0: SpikeLog baseline model (clean rewrite using spikingjelly).

Original: SpikeLog (Qi et al., IEEE TKDE 2024) — DualInputNet_Spike
Reference: logadempirical/logdeep/models/spiknet_bgl.py

Architecture:
    SpikeNet: 3-layer feedforward LIF stack
        input (seq_len, B, 300)
        → Linear(300 → hidden) → LIF
        → Linear(hidden → 64) → LIF
        → Linear(64 → num_out) → LIF (no reset, leaky integrator output)
        → output (seq_len, B, num_out)

    DualSpikeNet: shared SpikeNet + LSTM + concat + Linear
        x1, x2: (B, seq_len, 300)
        → SpikeNet(x1), SpikeNet(x2) → (seq_len, B, num_out)
        → LSTM(x1), LSTM(x2) → take last hidden
        → concat([h1, h2]) → Linear(2*num_out → 1)
        → anomaly score

Key differences from SpikeLog reference:
- Uses spikingjelly LIFNode instead of snntorch.RLeaky (consistent with SorLog)
- No recurrent feedback (feedforward LIF stack, simpler and more neuromorphic)
- Learnable tau + threshold via spikingjelly's tau parameter (clamped > 0)
"""

import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import neuron, functional
    SPIKINGJELLY = True
except ImportError:
    SPIKINGJELLY = False


class LIFBlock(nn.Module):
    """Linear + LIF neuron block (single timestep)."""

    def __init__(self, in_features: int, out_features: int,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 detach_reset: bool = True, no_reset: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

        if SPIKINGJELLY:
            reset_mechanism = "none" if no_reset else "subtract"
            self.lif = neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                surrogate_function=_get_surrogate("atan"),
                step_mode="s",
            )
            if no_reset:
                self.lif.v_reset = None
        else:
            raise ImportError("spikingjelly not found. Install: pip install spikingjelly")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))

    def reset(self):
        functional.reset_net(self.lif)


class SpikeNet(nn.Module):
    """3-layer feedforward LIF stack.

    Input:  x (B, emb_dim) — single timestep
    Output: spk (B, num_out)
    """

    def __init__(
        self,
        num_inputs: int = 300,
        num_hidden: int = 128,
        num_out: int = 32,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.block1 = LIFBlock(num_inputs, num_hidden, tau, v_threshold, detach_reset)
        self.block2 = LIFBlock(num_hidden, 64, tau, v_threshold, detach_reset)
        self.block3 = LIFBlock(64, num_out, tau, v_threshold, detach_reset, no_reset=True)

    def reset(self):
        self.block1.reset()
        self.block2.reset()
        self.block3.reset()

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Process full sequence.

        Args:
            x: (B, seq_len, emb_dim)
        Returns:
            out: (seq_len, B, num_out)  — membrane potential trace for LSTM
        """
        B, T, D = x.shape
        self.reset()
        outputs = []
        for t in range(T):
            spk = self.forward_step(x[:, t, :])  # (B, num_out)
            outputs.append(spk)
        return torch.stack(outputs, dim=0)  # (T, B, num_out)

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single timestep forward."""
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        return h


class DualSpikeNet(nn.Module):
    """s0: Dual pairwise SNN baseline.

    Shared SpikeNet processes both sequences, LSTM captures temporal dynamics,
    final Linear scores their relative anomalousness.

    Input:
        x1, x2: (B, seq_len, emb_dim)
    Output:
        score: (B, 1)  — higher = more anomalous pair
    """

    def __init__(
        self,
        num_inputs: int = 300,
        num_hidden: int = 128,
        num_out: int = 32,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.spike_rnn = SpikeNet(num_inputs, num_hidden, num_out, tau, v_threshold, detach_reset)
        self.rnn = nn.LSTM(input_size=num_out, hidden_size=num_out, batch_first=False)
        self.output_layer = nn.Linear(num_out * 2, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence to a fixed-length vector via SpikeNet + LSTM.

        Args:
            x: (B, T, D)
        Returns:
            h: (B, num_out) — last LSTM hidden state
        """
        spk_seq = self.spike_rnn.forward_seq(x)  # (T, B, num_out)
        _, (h, _) = self.rnn(spk_seq)             # h: (1, B, num_out)
        return h.squeeze(0)                        # (B, num_out)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self._encode(x1)
        h2 = self._encode(x2)
        out = self.output_layer(torch.cat([h1, h2], dim=1))  # (B, 1)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Single-sequence encoding for anomaly scoring."""
        return self._encode(x)


# ─── Surrogate helper (same as SorLog) ───────────────────────────────────────

def _get_surrogate(name: str):
    from spikingjelly.activation_based import surrogate
    mapping = {
        "atan": surrogate.ATan(),
        "sigmoid": surrogate.Sigmoid(),
        "fast_sigmoid": surrogate.PiecewiseLeakyReLU(),
    }
    return mapping.get(name, surrogate.ATan())
