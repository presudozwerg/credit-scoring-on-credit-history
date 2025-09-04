import numpy as np
import torch
import torch.nn as nn

from constants import DROPOUT_P, HIDDEN_DIM, IS_BIDIR, N_LAYERS


class CreditRNNModel(nn.Module):
    """RNN-based model for credit scoring"""

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        rnn_type: str = "rnn",
        aggregation_type: str = "last",
        is_bidir: bool = IS_BIDIR,
        dropout_prob: float = DROPOUT_P,
    ):
        super().__init__()
        rnn_input_size = seq_len

        model_params_dict = {
            "input_size": rnn_input_size,
            "hidden_size": hidden_dim,
            "num_layers": n_layers,
            "bidirectional": is_bidir,
            "batch_first": True,
        }

        if rnn_type == "rnn":
            self.rnn = nn.RNN(**model_params_dict)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(**model_params_dict)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(**model_params_dict)
        else:
            raise ValueError("Wrong type of RNN block!")

        linear_dim = (is_bidir + 1) * hidden_dim
        self.linear = nn.Linear(linear_dim, linear_dim)
        self.projection = nn.Linear(linear_dim, 1)
        self.aggregation_type = aggregation_type

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the `CreditRNNModel`

        Args:
            input_batch (torch.Tensor): Input batch to pass
                through the model.

        Raises:
            ValueError: Raises if type of RNN layer is incorrect

        Returns:
            torch.Tensor: Output of the model (predicted probs)
        """
        input_batch = input_batch.to(torch.float32)
        output, _ = self.rnn(input_batch)  # [batch_size, n_seq, hidden_dim]

        if self.aggregation_type == "max":
            output = output.max(dim=1)[0]  # [batch_size, hidden_dim]
        elif self.aggregation_type == "mean":
            output = output.mean(dim=1)  # [batch_size, hidden_dim]
        elif self.aggregation_type == "last":
            output = output[:, -1, :]
        else:
            raise ValueError("Invalid aggregation_type")

        output = self.dropout(
            self.linear(self.non_lin(output))
        )  # [batch_size, hidden_dim]
        projection = self.projection(self.non_lin(output))  # [batch_size, 1]

        return torch.sigmoid(projection)


if __name__ == "__main__":
    inp = torch.tensor(np.zeros((128, 10, 59))).to(torch.float32)
    m = CreditRNNModel(59)
    print(m(inp).shape)
