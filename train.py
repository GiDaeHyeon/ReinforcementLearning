import os

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from model import NMTTrainModule
from data import NMTDataModule

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.init()


logger = TensorBoardLogger(
    save_dir="logs",
    name="seq2seq",
    default_hp_metric=False
)

ckpt_callback = ModelCheckpoint(
    monitor="val_reward",
    dirpath="ckpt",
    filename="nmt_with_rl",
    mode="max"
)

earlystop_callback = EarlyStopping(
    monitor="val_reward",
    min_delta=1e-4,
    patience=100,
    verbose=True,
    mode='max'
)

trainer = Trainer(
    logger=logger, callbacks=[ckpt_callback, earlystop_callback],
    accelerator="gpu", log_every_n_steps=5, devices=[0]
)

if __name__ == "__main__":
    import yaml

    with open("./config.yaml", "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    model = NMTTrainModule(**config)
    data_module = NMTDataModule(**config)
    trainer.fit(model=model, datamodule=data_module)