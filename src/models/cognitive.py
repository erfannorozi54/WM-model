import torch
import torch.nn as nn
from typing import Tuple, Optional


class CognitiveModule(nn.Module):
    """
    Base interface for cognitive (RNN-family) modules.

    Contract:
    - Forward expects inputs of shape (B, T, D_in), where typically D_in = hidden_size + 3
      (visual embedding + task one-hot).
    - Returns a tuple: (outputs, final_state, hidden_seq)
        outputs: (B, T, H)  - RNN outputs per timestep
        final_state: RNN final state (type depends on cell)
        hidden_seq: (B, T, H) - hidden state per timestep (identical to outputs for RNN/GRU/LSTM)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Preprocessor to map (hidden_size + 3) -> hidden_size, with LN and nonlinearity
        self.pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def get_init_hidden(self, batch_size: int, device: torch.device):
        raise NotImplementedError


class VanillaRNN(CognitiveModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0, nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, num_layers, dropout, bidirectional=False)
        self.rnn = nn.RNN(
            input_size=hidden_size,  # after preprocessor
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        # x: (B, T, input_size)
        x = self.pre(x)  # (B, T, H)
        out, h_n = self.rnn(x, hidden)  # out: (B, T, H)
        # For RNN, out already equals hidden seq
        return out, h_n, out

    def get_init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class GRUCog(CognitiveModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, num_layers, dropout, bidirectional=False)
        self.rnn = nn.GRU(
            input_size=hidden_size,  # after preprocessor
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        x = self.pre(x)
        out, h_n = self.rnn(x, hidden)
        return out, h_n, out

    def get_init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class LSTMCog(CognitiveModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, num_layers, dropout, bidirectional=False)
        self.rnn = nn.LSTM(
            input_size=hidden_size,  # after preprocessor
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x = self.pre(x)
        out, (h_n, c_n) = self.rnn(x, hidden)
        # For LSTM, hidden sequence equals out; return full state tuple
        return out, (h_n, c_n), out

    def get_init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
