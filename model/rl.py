import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
from transformers import AutoTokenizer
from nltk.translate.gleu_score import sentence_gleu as glue


class Reward(nn.Module):
    def __init__(self, weight: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(weight)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        scores = []
        y = y.squeeze()
        for idx in range(y.size(0)):
            ref = self.tokenizer.decode(y[idx]).replace(self.tokenizer.eos_token, "")
            hyp = self.tokenizer.decode(y_hat[idx]).replace(self.tokenizer.eos_token, "")
            scores += [glue(ref, hyp) * 100.]
        scores = torch.Tensor(scores)
        return scores


class RLLossFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.squeeze()
        print(y_hat.shape, y.shape)
        log_prob = -nll_loss(torch.exp(y_hat.reshape(2, -1, 64)), y.squeeze(), ignore_index=50256, reduction="none")
        return (log_prob * -reward.cuda()).sum()
