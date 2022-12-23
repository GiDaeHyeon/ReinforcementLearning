from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src: torch.Tensor, h_t_tgt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.linear(h_t_tgt)
        weight = torch.bmm(query, h_src.transpose(1, 2))

        if mask is not None:
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))
        weight = self.softmax(weight)
        context_vector = torch.bmm(weight, h_src)
        return context_vector


class Encoder(nn.Module):
    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super().__init__()
        self.rnn = nn.LSTM(
            word_vec_size,
            int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        return self.rnn(x.squeeze())


class Decoder(nn.Module):

    def __init__(self, word_vec_size: int, hidden_size: int,
                 n_layers: int = 4, dropout_p: float=.2) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, emb_t: torch.Tensor, hidden_t_1_tilde: torch.Tensor,
                hidden_t_1: torch.Tensor) -> tuple:
        batch_size = emb_t.size(0)
        hidden_size = hidden_t_1[0].size(-1)

        if hidden_t_1_tilde is None:
            hidden_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([emb_t, hidden_t_1_tilde], dim=-1)
        y, h = self.rnn(x, hidden_t_1)
        return y, h


class Generator(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.output(x))


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

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x: torch.Tensor, length: list) -> torch.Tensor:
        mask = []
        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [
                    torch.cat(
                        [
                            x.new_ones(1, l).zero_(), x.new_ones(1, (max_length - l))
                        ], dim=-1
                    )
                ]
            else:
                mask += [x.new_ones(1, l).zero_()]
        mask = torch.cat(mask, dim=0).bool()
        return mask

    def fast_merge_encoder_hiddens(self, encoder_hiddens: torch.Tensor) -> tuple:
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(
            batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(
            batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        return h_0_tgt, c_0_tgt

    def forward(self, src: Union[tuple, torch.Tensor],
                tgt: Union[tuple, torch.Tensor]) -> torch.Tensor:
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
        h_src, h_0_tgt = self.encoder(emb_src)
        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt)

        h_tilde = []
        h_t_tilde = None
        decoder_hidden = h_0_tgt

        for t in range(tgt.size(1)):
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(
                emb_t, h_t_tilde, decoder_hidden
                )
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat(
                [decoder_output, context_vector], dim=-1)))
            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
        y_hat = self.generator(h_tilde)
        return y_hat

    def search(self, src: Union[tuple, torch.Tensor],
               is_greedy: bool = True, max_length: int = 255) -> tuple:
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x, x_length = src, None
            mask = None
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder(emb_src)
        decoder_hidden = self.fast_merge_encoder_hiddens(h_0_tgt)

        y = x.new(batch_size, 1).zero_() + 101

        is_decoding = x.new_ones(batch_size, 1).bool()
        h_t_tilde, y_hats, indice = None, [], []
        
        while is_decoding.sum() > 0 and len(indice) < max_length:
            emb_t = self.emb_dec(y)
            decoder_output, decoder_hidden = self.decoder(
                emb_t, h_t_tilde, decoder_hidden)
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(
                torch.cat([decoder_output, context_vector], dim=-1)))
            y_hat = self.generator(h_t_tilde)

            y_hats += [y_hat]
            if is_greedy:
                y = y_hat.argmax(dim=-1)
            else:
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)

            y = y.masked_fill_(~is_decoding, 0)
            is_decoding = is_decoding * torch.ne(y, 102)
            indice += [y]
        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)
        return y_hats, indice
