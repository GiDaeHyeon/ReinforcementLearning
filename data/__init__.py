"""
Neural Machine Translation Data Module
"""
from torch.cuda import device_count
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import NMTDataset


class NMTDataModule(LightningDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.train_dataset = NMTDataset(
            weight=kwargs.get("weight"),
            phase="train",
            max_length=kwargs.get("max_length")
            )
        self.val_dataset = NMTDataset(
            weight=kwargs.get("weight"),
            phase="validation",
            max_length=kwargs.get("max_length")
            )
        self.test_dataset = NMTDataset(
            weight=kwargs.get("weight"),
            phase="test",
            max_length=kwargs.get("max_length")
            )
        self.batch_size = kwargs.get("batch_size")
        self.num_workers = 4

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False
        )
