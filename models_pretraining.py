# coding=utf-8

import json
import transformers

from dataclasses import dataclass
from typing import Optional

from transformers import BertConfig, BertForMaskedLM

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule



class BertForPreTraining(LightningModule):
    """
    PyTorch Lightning module for pre-training of BERTs.
    Input:
    -----
    config: `str` - An absolute path to a JSON config file. Default = None
    train_dataset: Initialized pytorch dataset class
    val_dataset: Initialized pytorch dataset class
    batch_size: `int`, Default = 8
    acc_grad_steps: `int`, Default = 4
    learning_rate: `float`, Default = 2e-5
    pytorch_start_dump: `str`; An absolute path to the folder with
        `pytorch_model.bin`, `config.json` and `vocab.txt`. Default = None
    """
    def __init__(
        self, config: str = None, train_dataset=None, val_dataset=None,
        batch_size: int = 8, acc_grad_steps: int = 4,
        learning_rate: float = 2e-5, pytorch_start_dump = None
    ):
        super(BertForPreTraining, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        
        if pytorch_start_dump:
            self.model = BertForMaskedLM.from_pretrained(
                pytorch_start_dump
            )
        elif config and (not pytorch_start_dump):
            with open(config) as f:
                config_dict = json.load(f)
                self._config = BertConfig.from_dict(config_dict)
            self.model = BertForMaskedLM(self._config)

        if train_dataset:
            self.train_dataset = train_dataset
        else:
            raise ValueError("Argument train_dataset must be provided.")
        if val_dataset:
            self.val_dataset = val_dataset

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch_, batch_idx):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=5
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=5
        )

    def configure_optimizers(self):
        pass

    def save_model(self):
        pass


@dataclass
class BertOptimizerArguments():
    """
    Default AdamW optimizer used for BERT pre-training.
    """
    optimizer: Optional = optim.AdamW
    learning_rate: Optional[float] = 5e-4
    weight_decay: Optional[float] = 0.01
    adam_epsilon: Optional[float] = 1e-6
    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.999


@dataclass
class BertSchedulerArguments():
    """
    Default lr_scheduler used for BERT pre-training.
    """
    scheduler: Optional = transformers.get_linear_schedule_with_warmup
    num_warmup_steps: Optional[int] = 10000  # yet to be configured