from itertools import combinations
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from diffusers import AutoencoderKL
from torch import nn, Tensor, optim
from torch.nn import functional as F

from .loss_dict import LossDict
from .timestamp_shuffle import TimestampShuffle
from .trainable_module import TrainableModule


class LatentGrowthModel(nn.Module):
    """
    Abstract class for Latent Growth Models, that defines the API for all inheriting subclasses.
    """
    def __init__(self):
        super(LatentGrowthModel, self).__init__()

    def forward(self, z_in: Tensor, t_in: Tensor, t_out: Tensor, z_out: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Args:
            z_in: Tensor of shape [B N C H W] if batch_first else [N B C H W]
            t_in: Tensor of shape [B N] if batch_first else [N B]
            t_out: Tensor of shape [B M] if batch_first else [M B]
            z_out: Optional Tensor of shape [B M C H W] if batch_first else [M B C H W]

        Returns:
            Tensor of shape [B M C H W] if batch_first else [M B C H W].
            If z_out is not None the loss can also be returned as a second return value.
        """
        raise NotImplementedError()

class LGMTrainer(TrainableModule):
    """
    Trainer module to train LGM models.

    Args:
        path (str | Path): directory to save model checkpoints during training
        model (torch.nn.Module): model to train
        lr (float): learning rate (default: 1e-4)
        gradient_accumulation_steps (int): number of steps to accumulate the gradients before the optimization step is
            performed (default: 1)

    Example:
        >>> from dlbpgm.models import LGMTrainer, LGT
        >>> lgt = LGT(resolution=256)
        >>> trainer = LGMTrainer(path="path/to/directory", model=lgt.model, lr=1e-4)
        >>> trainer.train_epoch(train_dataloader)
        >>> trainer.valdation_epoch(val_dataloader)
    """
    def __init__(self,
                 path: str | Path,
                 model: LatentGrowthModel,
                 lr: float = 1e-4,
                 gradient_accumulation_steps: int = 1,
                 timestamp_shuffle: TimestampShuffle = None
                 ):
        super(LGMTrainer, self).__init__(path=path)
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.training_stage = 'training'
        self.vae = LatentGrowthVAE()
        self.timestamp_shuffle = timestamp_shuffle or nn.Identity()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step = 0

    def gradient_accumulation_callback(self) -> bool:
        self.step = (self.step + 1) % self.gradient_accumulation_steps
        return self.step == 0

    def set_learning_rate(self, lr: float):
        self.optimizer.param_groups[0]['lr'] = lr

    def preprocess_batch(self, batch) -> tuple:
        if self.training_stage == 'training':
            return batch
        elif self.training_stage == 'finetune':
            x_in, t_in, t_out, x_out = batch
            z_in = self.vae.encode(x_in)
            z_out = self.vae.encode(x_out)
            return z_in, t_in, t_out, z_out
        else:
            raise ValueError(f'Unknown training stage: {self.training_stage}')

    def training_step(self, batch: Union[Tensor, Tuple[Tensor]]) -> LossDict:
        z_in, t_in, t_out, z_out = self.preprocess_batch(batch)
        t_in = self.timestamp_shuffle(t_in.int())
        t_out = self.timestamp_shuffle(t_out.int())
        _, loss = self.model(z_in=z_in, t_in=t_in, t_out=t_out, z_out=z_out)
        loss.backward()
        if self.gradient_accumulation_callback():
            self.optimizer.step()
            self.optimizer.zero_grad()
        return LossDict(loss=loss.item())

    @torch.no_grad()
    def validation_step(self, batch) -> LossDict:
        z_in, t_in, t_out, z_out = batch
        _, loss = self.model(z_in=z_in, t_in=t_in, t_out=t_out, z_out=z_out)
        return LossDict(val_l2_loss=loss.item())

def make_input_target_variations(n_inputs: int, n_targets: int) -> list[tuple[list[int], list[int]]]:
    """
    Helper function to generate combinations of input and target variations
    Args:
        n_inputs (int):
        n_targets (int):

    Returns:

    """
    s = set(range(n_inputs + n_targets))
    return [(list(c), list(s - c)) for c in map(set, combinations(s, n_inputs))]

class LatentGrowthTrainer(TrainableModule):
    def __init__(self, path: Union[str, Path], lgm: LatentGrowthModel):
        super(LatentGrowthTrainer, self).__init__(path)
        self.lgm = lgm
        self.vae = LatentGrowthVAE()
        self.optimizer = optim.AdamW(self.lgm.parameters(), lr=1e-4)
        self.training_stage = 'training'

    def preprocess_batch(self, batch) -> tuple:
        if self.training_stage == 'training':
            return batch
        elif self.training_stage == 'finetuning':
            x_in, x_out, t_in, t_out = batch
            z_in = self.vae.encode(x_in)
            z_out = self.vae.encode(x_out)
            return z_in, z_out, t_in, t_out
        else:
            raise ValueError(f'Unknown training stage: {self.training_stage}')


    def input_target_variation_loss(self, z_in: Tensor, t_in: Tensor, t_out: Tensor, z_out: Tensor) -> Tensor:
        z = torch.cat([z_in, z_out], dim=1)
        t = torch.cat([t_in, t_out], dim=1)
        losses = []
        for idx_in, idx_out in make_input_target_variations(z_in.shape[1], z_out.shape[1]):
            _, loss = self.lgm(z_in=z[:, idx_in], t_in=t[:, idx_in], t_out=t[:, idx_out], z_out=z[:, idx_out])
            losses.append(loss)

        loss = torch.stack(losses).mean()
        return loss

    def training_step(self, batch: Union[Tensor, Tuple[Tensor]]) -> LossDict:
        z_in, z_out, t_in, t_out = self.preprocess_batch(batch)
        loss = self.input_target_variation_loss(z_in, t_in, t_out, z_out)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return LossDict(loss=loss.item())

    @torch.no_grad()
    def validation_step(self, batch) -> LossDict:
        z_in, z_out, t_in, t_out = batch
        loss = self.input_target_variation_loss(z_in, t_in, t_out, z_out)
        return LossDict(val_l2_loss=loss.item())


class LatentGrowthVAE(nn.Module):
    """
    Pre-trained VAE like it was used for Latent Diffusion Models in https://arxiv.org/abs/2112.10752
    This class mimics its behaviour by using its functions but provides additional functionality to
    work with sequences of images.

    It is also possible to pass a LatentGrowthModel for temporal modeling on latent variables.
    """
    scale = 0.18215

    def __init__(self, latent_growth_model: LatentGrowthModel = None):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float32)
        self.vae.requires_grad = False
        if latent_growth_model is None:
            def identity(x, *args, **kwargs):
                return x
            latent_growth_model = identity
        self.latent_growth_model = latent_growth_model


    def encode(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): Tensor of shape [..., 3, H, W]

        Returns: Tensor with normalized latents of shape [..., 4, H', W'] with H' = H // 8 and W' = W // 8

        """
        NB, img_shape = x.shape[:-3], x.shape[-3:]
        x = x.view(-1, *img_shape) * 2 - 1      # [N*B, 3, 256, 256]
        x = self.vae.encode(x).latent_dist.sample()     # [N*B, 4, 32, 32]
        x = x.reshape(*NB, *x.shape[-3:])               # [N, B, 4, 32, 32]
        return x * LatentGrowthVAE.scale

    def decode(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): Tensor with normalized latents from the encoder with shape [..., 4, H', W']

        Returns: Tensor of shape [..., 3, H, W] containing image data

        """
        NB, img_shape = x.shape[:-3], x.shape[-3:]
        x = x / LatentGrowthVAE.scale
        x = x.view(-1, *img_shape)                      # [(...), 4, 32, 32]
        x = self.vae.decode(x).sample / 2 + 0.5         # [(...), 3, 256, 256]
        x = x.reshape(*NB, *x.shape[-3:])               # [ ... , 3, 256, 256]
        return x.clip(0, 1)

    def forward(self, x_in: Tensor, t_in: Tensor, t_out: Tensor, x_out: Tensor = None, **kwargs):
        """

        Args:
            x_in: [N, B, C, H, W]
            t_in: [N, B]
            t_out: [M, B] or [B]
            y: Optional [M, B, C, H, W] or [B, C, H, W]

        Returns: [M, B, C, H, W] or [B, C, H, W]

        """
        z_in = self.encode(x_in)
        z_pred = self.latent_growth_model(z_in, t_in, t_out)
        x_pred = self.decode(z_pred)
        if x_out is None:
            return x_pred
        z_out = self.encode(x_out)
        loss = dict(embedding_loss=F.mse_loss(z_pred, z_out),
                    reconstruction_loss=F.l1_loss(x_pred, x_out))
        return x_pred, loss


    def embedding_loss(self, x: Tensor, t_in: Tensor, t_out: Tensor, y: Tensor) -> Tensor:
        z_y = self.encode(y)
        z_x = self.encode(x)
        z_x = self.latent_growth_model(z_x, t_in, t_out)
        return F.l1_loss(z_x, z_y)
