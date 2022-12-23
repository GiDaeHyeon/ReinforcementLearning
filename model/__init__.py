"""RL Text Generator Module"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from nltk.translate.gleu_score import sentence_gleu

from .layers.seq2seq import Seq2Seq
from data import NMTDataModule
from data.dataset import SPECIAL_TOKEN


class NMTTrainModule(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = Seq2Seq(
            **kwargs.get("seq2seq_hp")
        )
        self.hp = kwargs.get("seq2seq_hp")

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            params=self.model.parameters(), lr=self.hp.get("lr")
        )

    def get_reward(self, y: torch.Tensor, y_hat: torch.Tensor, n_gram=6):
        score_func = lambda ref, hyp: sentence_gleu([ref], hyp, max_len=n_gram)
        with torch.no_grad():
            scores = []
            for b in range(y.size(0)):
                ref, hyp = [], []
                for t in range(y.size(-1)):
                    ref += [str(int(y[b, t]))]
                    if y[b, t] == SPECIAL_TOKEN.get("EOS"):
                        break

                for t in range(y_hat.size(-1)):
                    hyp += [str(int(y_hat[b, t]))]
                    if y_hat[b, t] == SPECIAL_TOKEN.get("EOS"):
                        break
                scores += [score_func(ref, hyp) * 100.]
            scores = torch.FloatTensor(scores).cuda()
        return scores

    def get_loss(self, y_hat: torch.Tensor, indice: torch.Tensor, reward: int = 1) -> torch.Tensor:
        batch_size = indice.size(0)
        output_size = y_hat.size(-1)
        log_prob = -F.nll_loss(
            y_hat.view(-1, output_size),
            ignore_index=SPECIAL_TOKEN.get("PAD"),
            reduction="none"
        ).view(batch_size, -1).sum(dim=-1)
        return (log_prob * -reward).sum()

    def forward(self, x: torch.Tensor, is_greedy: bool = False,
                max_length: int = 32) -> torch.Tensor:
        return self.model.search(x, is_greedy=is_greedy, max_length=max_length)

    def training_step(self, batch, batch_idx, **kwargs) -> dict:
        kor, eng = batch
        x, _ = kor
        y, _ = eng
        y_hat, indice = self(x, max_length=self.hp.get("max_length"))

        with torch.no_grad():
            actor_reward = self.get_reward(y=y, y_hat=y_hat)
            
            baseline = []
            for _ in range(self.hp.get("n_samples")):
                _, sampled_indice = self(x, max_length=self.hp.get("max_length"))
                baseline += [self.get_reward(y=y, y_hat=sampled_indice)]
            baseline = torch.stack(baseline).mean(dim=0)
            reward = actor_reward - baseline
        loss = self.get_loss(y_hat=y_hat, indice=indice, reward=reward)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_reward", reward, prog_bar=True)
        return {"loss": loss, "train_reward": reward}

    def validation_step(self, batch, batch_idx, **kwargs) -> dict:
        kor, eng = batch
        x, _ = kor
        y, _ = eng
        _, indice = self(x, max_length=self.hp.get("max_length"))
        reward = self.get_reward(y=y, y_hat=indice)
        self.log("val_reward", reward)
        return {"val_reward": reward}
