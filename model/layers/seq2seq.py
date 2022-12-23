"""Attention Seq2Seq Model Layers"""
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Attention(nn.Module):
    def __init__(self, hidden_size: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src: torch.Tensor, h_t_target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.linear(h_t_target)
        weight = torch.bmm(query, h_src.transpose(1, 2))

        if mask is not None:
            weight.masked_fill_(mask.unsqueeze(1), -float("inf"))
        weight = self.softmax(weight)
        return self.bmm(weight, h_src)


class Encoder(nn.Module):
    def __init__(self, word_vec_size: int, hidden_size: int,
                 n_layers: int = 4, dropout_p: float = .2) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be an even number.")

        self.lstm = nn.LSTM(
            word_vec_size,
            int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, embedding: Union[torch.Tensor, tuple]) -> tuple:
        if isinstance(embedding, tuple):
            x, lengths = embedding
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = embedding

        y, h = self.lstm(x)

        if isinstance(embedding, tuple):
            y, _ = unpack(y, batch_first=True)
        return y, h


class Decoder(nn.Module):
    def __init__(self, word_vec_size: int, hidden_size: int,
                 n_layers: int = 4, dropout_p: float = .2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, embedding_t: torch.Tensor, hidden_t_1_tilde: Optional[torch.Tensor],
                hidden_t_1: torch.Tensor) -> tuple:
        batch_size = embedding_t.size(0)
        hidden_size = hidden_t_1[0].size(-1)

        if hidden_t_1_tilde is None:
            # 첫 timestep에서는 0
            hidden_t_1_tilde = embedding_t.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([embedding_t, hidden_t_1_tilde], dim=-1)
        y, h = self.lstm(x, hidden_t_1)
        return y, h


class Seq2Seq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
