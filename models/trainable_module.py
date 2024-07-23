import sys
from pathlib import Path
from typing import Union, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss_dict import LossDict


class TrainableModule(nn.Module):
    """
    Abstract class providing functionality for the training of modules.
    Inheriting trainer classes must at least implement the following methods:
    - training_step: Step where the optimization is performed. Individual losses are returned within a LossDict object.
        The method is used in the `train_epoch` method.
    - validation_step: Method used in `validation_epoch` method. Here the validation losses are calculated and returned
        within a LossDict object.

    This class abstracts the functionality of iterating over DataLoaders, copying data to the respective devices, and
    saving the best and last model checkpoints into a specified directory.
    Thereby, the last checkpoint is saved after each training epoch and the best checkpoint is saved on the basis of the
    validation results.
    """

    BEST_CHECKPOINT_PATH = 'best.pt'
    LAST_CHECKPOINT_PATH = 'last.pt'
    HPARAMS_PATH = 'hparams.yaml'

    def __init__(self, path: str):
        super(TrainableModule, self).__init__()
        self.path = Path(path)
        self.register_buffer('epoch', torch.tensor(0))
        self.register_buffer('best_val_loss', None)

    def training_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor]]) -> LossDict:
        pass

    def validation_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor]]) -> LossDict:
        pass

    def log(self, loss: LossDict):
        pass

    def train_epoch(self, dataloader: DataLoader) -> LossDict:
        device = next(iter(self.parameters()))
        self.train()
        total_loss = LossDict()
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {self.epoch + 1}', file=sys.stdout) as pbar:
            for step, batch in pbar:
                batch = (p.to(device) for p in batch)
                loss = self.training_step(batch)
                self.log(loss)
                total_loss += loss
                mean_loss = total_loss / (step + 1)
                pbar.set_postfix(**mean_loss)
        self.epoch += 1
        self.save_last_checkpoint()
        return mean_loss


    @torch.no_grad()
    def validation_epoch(self, dataloader: DataLoader) -> LossDict:
        device = next(iter(self.parameters())).device
        self.eval()
        total_loss = LossDict()
        with tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation', file=sys.stdout) as pbar:
            for step, batch in pbar:
                batch = (p.to(device) for p in batch)
                loss = self.validation_step(batch)
                total_loss += loss
                mean_loss = total_loss / (step + 1)
                pbar.set_postfix(**mean_loss)
        self.log(mean_loss)
        val_loss = torch.tensor(mean_loss.item())  # convert to tensor to store value as buffer
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_best_checkpoint()
        return mean_loss

    @torch.no_grad()
    def on_validation_epoch_end(self, batch):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, strict=True):
        state_dict = torch.load(path)
        if 'best_val_loss' in state_dict:
            self.best_val_loss = torch.tensor(0, device=next(self.parameters()).device)
        self.load_state_dict(state_dict, strict=strict)


    @property
    def best_checkpoint_path(self):
        return self.path / TrainableModule.BEST_CHECKPOINT_PATH

    @property
    def last_checkpoint_path(self):
        return self.path / TrainableModule.LAST_CHECKPOINT_PATH

    @property
    def hparams_path(self):
        return self.path / TrainableModule.HPARAMS_PATH

    def save_best_checkpoint(self):
        self.save(self.best_checkpoint_path)

    def save_last_checkpoint(self):
        self.save(self.last_checkpoint_path)

    def load_best_checkpoint(self, strict=True):
        self.load(self.best_checkpoint_path, strict=strict)

    def load_last_checkpoint(self, strict=True):
        self.load(self.last_checkpoint_path, strict=strict)