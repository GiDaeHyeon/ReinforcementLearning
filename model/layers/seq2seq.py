"""Attention Seq2Seq Model Layers"""
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from data.dataset import SPECIAL_TOKEN


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


class Generator(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.output(x))  # Log prob


class Seq2Seq(nn.Module):
    def __init__(self, input_size: int, word_vec_size: int, hidden_size: int,
                 output_size: int, n_layers: int = 4, dropout_p: float = .2,
                 **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)  # Encoder is bi-directional LSTM
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x: torch.Tensor, length: int) -> torch.Tensor:
        mask = []
        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [
                    torch.cat(
                        [x.new_ones(1, l).zero_(), x.new_ones(1, (max_length - l))], dim=-1
                    )
                ]
            else:
                mask += [x.new_ones(1, l).zero_()]
        mask = torch.cat(mask, dim=0).bool()
        return mask

    def forward(self, src: Union[torch.Tensor, tuple], tgt: torch.Tensor) -> torch.Tensor:
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt)

        h_tilde = []
        h_t_tilde = None
        decoder_hidden = h_0_tgt
        for t in range(tgt.size(1)):
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden
                                                          )
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                         ], dim=-1)))
            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
        y_hat = self.generator(h_tilde)
        return y_hat

    def merge_encoder_hiddens(self, encoder_hiddens: torch.Tensor) -> tuple:
        new_hiddens, new_cells = [], []

        hiddens, cells = encoder_hiddens
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)]
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)

    def fast_merge_encoder_hiddens(self, encoder_hiddens: torch.Tensor) -> tuple:
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(
            batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(
            batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        return h_0_tgt, c_0_tgt

    def search(self, src: Union[torch.Tensor, tuple],
               is_greedy: bool = True, max_length: int = 255) -> tuple:
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x, x_length = src, None
            mask = None
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        decoder_hidden = self.fast_merge_encoder_hiddens(h_0_tgt)

        y = x.new(batch_size, 1).zero_() + SPECIAL_TOKEN.get("BOS")

        is_decoding = x.new_ones(batch_size, 1).bool()
        h_t_tilde, y_hats, indice = None, [], []

        while is_decoding.sum() > 0 and len(indice) < max_length:
            emb_t = self.emb_dec(y)
            decoder_output, decoder_hidden = self.decoder(
                emb_t, h_t_tilde, decoder_hidden
                )
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(
                torch.cat([decoder_output, context_vector], dim=-1)))
            y_hat = self.generator(h_t_tilde)
            y_hats += [y_hat]

            if is_greedy:
                y = y_hat.argmax(dim=-1)
            else:
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)

            # Put PAD if the sample is done.
            y = y.masked_fill_(~is_decoding, SPECIAL_TOKEN.get("PAD"))
            # Update is_decoding if there is EOS token.
            is_decoding = is_decoding * torch.ne(y, SPECIAL_TOKEN.get("EOS"))
            indice += [y]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)
        return y_hats, indice
