import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from bc_network.datasets.isaac_dataset import TrajectoriesDataset

class BCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, config, device):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.device = device
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # load dataset
        self.training_dataset = TrajectoriesDataset(self.data_dir + "/train", self.config["sequence_len"])
        self.val_dataset = TrajectoriesDataset(self.data_dir + "/validation", self.config["sequence_len"])
        self.test_dataset = TrajectoriesDataset(self.data_dir + "/test", self.config["sequence_len"])
        return 

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
