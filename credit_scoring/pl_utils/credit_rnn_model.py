from omegaconf import DictConfig

import torch
import torch.nn as nn

class CreditRNNModel(nn.Module):
    """RNN-based model for credit scoring"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.conf = config

        if config.rnn_type == "rnn":
            self.rnn = nn.RNN(**config.model_params)
        elif config.rnn_type == "gru":
            self.rnn = nn.GRU(**config.model_params)
        elif config.rnn_type == "lstm":
            self.rnn = nn.LSTM(**config.model_params)
        else:
            raise ValueError("Wrong type of RNN block!")

        bidir_fact = config.model_params.bidirectional + 1
        linear_dim = bidir_fact * config.model_params.hidden_size

        self.linear = nn.Linear(linear_dim, linear_dim)
        self.projection = nn.Linear(linear_dim, config.n_classes)
        self.aggregation_type = config.agg_type

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=config.dropout_prob)

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
