"""
s0: SpikeLog baseline model (clean rewrite using spikingjelly).

Original: SpikeLog (Qi et al., IEEE TKDE 2024) — spiknet_bgl.py
Reference: logadempirical/logdeep/models/spiknet_bgl.py (BGL-specific version)

Architecture:
    SpikeNet: 3-layer recurrent LIF stack
        input (seq_len, B, 300)
        → Linear(300 → hidden=128) → RecurrentLIF(128)
        → Linear(128 → 64) → RecurrentLIF(64)
        → Linear(64 → num_out=32) → RecurrentLIF(32, threshold=0.1, no_reset)
        → spike output (seq_len, B, 32)

    DualSpikeNet: shared SpikeNet + 1-layer LSTM + last timestep + Linear
        x1, x2: (B, seq_len, 300)
        → SpikeNet(x1), SpikeNet(x2) → spike trains (seq_len, B, 32)
        → LSTM(spikes) → take last timestep output
        → concat([x1_last, x2_last]) → Linear(64 → 1)
        → anomaly score

Key notes:
- RLeaky in snntorch has recurrent connections: V[t] = beta*V[t-1] + fc(input) + recurrent(spk[t-1])
  We replicate this with an explicit recurrent Linear layer alongside spikingjelly LIFNode.
- BGL version feeds spike output (not membrane potential) to LSTM.
- BGL version uses simple last-timestep + Linear(64→1), not attention + MLP.
- li_out threshold is 0.1 (not 1.0) in BGL version.
"""

import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import neuron, functional
    SPIKINGJELLY = True
except ImportError:
    SPIKINGJELLY = False


class RecurrentLIFBlock(nn.Module):
    """Linear + Recurrent LIF neuron block (single timestep).

    Matches snntorch.RLeaky behavior:
        I[t] = fc(x[t]) + recurrent(spk[t-1])
        V[t] = beta * V[t-1] + I[t]
        spk[t] = Heaviside(V[t] - threshold)
    """

    def __init__(self, in_features: int, out_features: int,
                 tau: float = 2.0, v_threshold: float = 0.3,
                 detach_reset: bool = False, no_reset: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.recurrent = nn.Linear(out_features, out_features, bias=False)

        if not SPIKINGJELLY:
            raise ImportError("spikingjelly required")

        self.lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_function=_get_surrogate("sigmoid"),  # match original
            step_mode="s",
        )
        if no_reset:
            self.lif.v_reset = None

        self.prev_spike = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single timestep forward.

        Returns:
            spike: (B, out_features)
        """
        current = self.fc(x)
        if self.prev_spike is not None:
            current = current + self.recurrent(self.prev_spike)
        spike = self.lif(current)
        self.prev_spike = spike
        return spike

    def reset(self):
        functional.reset_net(self.lif)
        self.prev_spike = None


class SpikeNet(nn.Module):
    """3-layer recurrent LIF stack matching spiknet_bgl.py.

    Layer dims: input(300) → hidden(128) → 64 → num_out(32)
    """

    def __init__(
        self,
        num_inputs: int = 300,
        num_hidden: int = 128,
        num_out: int = 32,
        tau: float = 2.0,
        v_threshold: float = 0.3,
        out_threshold: float = 0.1,
        detach_reset: bool = False,
    ):
        super().__init__()
        self.block1 = RecurrentLIFBlock(num_inputs, num_hidden, tau, v_threshold, detach_reset)
        self.block2 = RecurrentLIFBlock(num_hidden, 64, tau, v_threshold, detach_reset)
        self.block3 = RecurrentLIFBlock(64, num_out, tau, out_threshold, detach_reset,
                                        no_reset=True)

    def reset(self):
        self.block1.reset()
        self.block2.reset()
        self.block3.reset()

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Process full sequence.

        Args:
            x: (B, seq_len, emb_dim) — batch_first
        Returns:
            spikes: (seq_len, B, num_out) — spike train for LSTM (time-first)
        """
        B, T, D = x.shape
        self.reset()
        spk_list = []
        for t in range(T):
            spk1 = self.block1(x[:, t, :])
            spk2 = self.block2(spk1)
            spk3 = self.block3(spk2)
            spk_list.append(spk3)
        return torch.stack(spk_list, dim=0)  # (T, B, num_out)


class DualSpikeNet(nn.Module):
    """s0: Dual pairwise SNN baseline — faithful to spiknet_bgl.py.

    Shared SpikeNet → spike train → 1-layer LSTM → last output → concat → Linear

    Input:
        x1, x2: (B, seq_len, emb_dim)
    Output:
        score: (B, 1)
    """

    def __init__(
        self,
        num_inputs: int = 300,
        num_hidden: int = 128,
        num_out: int = 32,
        tau: float = 2.0,
        v_threshold: float = 0.3,
        out_threshold: float = 0.1,
        detach_reset: bool = False,
    ):
        super().__init__()
        self.spike_rnn = SpikeNet(
            num_inputs, num_hidden, num_out, tau, v_threshold, out_threshold, detach_reset
        )
        self.rnn = nn.LSTM(input_size=num_out, hidden_size=num_out)
        self.output_layer = nn.Linear(num_out * 2, 1)

    def _encode_last(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence → last LSTM output.

        Args:
            x: (B, T, D)
        Returns:
            h: (B, num_out) — last timestep of LSTM output
        """
        spk_seq = self.spike_rnn.forward_seq(x)  # (T, B, num_out)
        lstm_out, _ = self.rnn(spk_seq)            # (T, B, num_out)
        return lstm_out[-1]                         # (B, num_out) — last timestep

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self._encode_last(x1)
        h2 = self._encode_last(x2)
        out = self.output_layer(torch.cat([h1, h2], dim=1))  # (B, 1)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Single-sequence encoding for anomaly scoring."""
        return self._encode_last(x)


# ─── Surrogate helper ────────────────────────────────────────────────────────

def _get_surrogate(name: str):
    from spikingjelly.activation_based import surrogate
    mapping = {
        "atan": surrogate.ATan(),
        "sigmoid": surrogate.Sigmoid(),
        "fast_sigmoid": surrogate.PiecewiseLeakyReLU(),
    }
    return mapping.get(name, surrogate.ATan())
