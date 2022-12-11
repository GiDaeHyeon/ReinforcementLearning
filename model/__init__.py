"""RL Text Generator Module"""
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from .layers import GPT2LMModel
from .rl import Reward, RLLossFunction


class NMTTrainModule(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.lm = GPT2LMModel(kwargs.get("weight"))
        self.reward = Reward(weight=kwargs.get("weight"))
        self.loss_fn = RLLossFunction()
        self.lr = kwargs.get("learning_rate")

    def forward(self, x) -> torch.Tensor:
        return self.lm(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            params=self.lm.parameters(), lr=self.lr
        )

    def training_step(self, batch, batch_idx, **kwargs) -> dict:
        kor, eng = batch
        y_hat = self(kor)
        reward = self.reward(eng, y_hat)
        loss = self.loss_fn(eng, y_hat, reward)
        self.log("reward", torch.mean(reward), prog_bar=True)
        return {"loss": loss, "train_reward": reward}

    def validation_step(self, batch, batch_idx, **kwargs) -> dict:
        kor, eng = batch
        y_hat = self(kor)
        reward = self.reward(eng, y_hat)
        self.log("val_reward", torch.mean(reward))

    def test_step(self, batch, batch_idx, **kwargs) -> dict:
        kor, eng = batch
        y_hat = self(kor)
        reward = self.reward(eng, y_hat)
        self.log("test_reward", torch.mean(reward))
