"""Transformers layers"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2LMModel(nn.Module):
    def __init__(self, weight: str) -> None:
        super().__init__()
        self.LM = GPT2LMHeadModel.from_pretrained(weight)

    def forward(self, tokens: dict) -> torch.Tensor:
        return self.LM(**tokens)
