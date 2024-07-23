import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm

class Metrics(nn.Module):
    """
    This class combines the metrics used to measure the model's performances during the experiments conducted.
    Should only be used internally within the Evaluator class.

    Args:
        fid_feature (int):
            Either an integer or ``nn.Module``:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        msssim_betas: Exponent parameters for individual similarities and contrastive sensitivities returned by different image
            resolutions.
    """
    def __init__(self,
                 fid_feature: int = 2048,
                 msssim_betas: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
        super(Metrics, self).__init__()
        self.is_metric = InceptionScore(normalize=True)
        self.fid_metric = FrechetInceptionDistance(feature=fid_feature, normalize=True, reset_real_features=False)
        self.msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(betas=msssim_betas)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    def reset(self):
        self.is_metric.reset()
        self.fid_metric.reset()
        self.msssim_metric.reset()
        self.psnr_metric.reset()

    def update(self, preds, targets):
        self.is_metric.update(preds)
        self.fid_metric.update(imgs=targets, real=True)
        self.fid_metric.update(imgs=preds, real=False)
        self.msssim_metric.update(preds, targets)
        self.psnr_metric.update(preds, targets)

    def compute(self) -> dict:
        """
        Computes the metrics and returns them in a dictionary.
        """
        return {
            "inception_score": self.is_metric.compute(),
            "fid_score": self.fid_metric.compute(),
            "msssim": self.msssim_metric.compute(),
            "psnr": self.psnr_metric.compute()
        }

class Evaluator(nn.Module):
    """
    Evaluator for Growth models that a sequence of images x_in, their corresponding timestamps t_in and requested
    timestamps t_out.
    The Evaluator computs the FID, IS, MS-SSIM and PSNR from the reconstructed images and the ground truth.

    Args:
        model (torch.nn.Module): Model to be evaluated
        dataset (Dataset): Dataset used to provide the samples during evaluation
        batch_size (int): Batch size used during evaluation (default: 8)
        num_workers (int): Number of workers to enable multi-process data loading (default: 0)
        fid_feature (int):
            Either an integer or ``nn.Module`` (default: 2048):

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        msssim_betas: Exponent parameters for individual similarities and contrastive sensitivities returned by different image
            resolutions.
        post_processing (Callable | torch.nn.Module): Functions that perform specified operations on the inputs before
            they are processed by the models.
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 fid_feature: int = 2048,
                 msssim_betas: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
                 post_processing = None):
        super(Evaluator, self).__init__()
        self.model = model
        self.dataset = dataset
        self.post_processing = post_processing
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
        self.metrics_general = Metrics(fid_feature, msssim_betas)
        self.metrics_interpolate = Metrics(fid_feature, msssim_betas)
        self.metrics_extrapolate = Metrics(fid_feature, msssim_betas)

    def evaluate(self):
        """
        Starts the evaluation of the given model and the provided dataset and updates the metrics with every mini-batch.
        """
        self.metrics_general.reset()
        self.metrics_interpolate.reset()
        self.metrics_extrapolate.reset()

        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating", file=sys.stdout):
                batch = [t.to(device) for t in batch]
                x_in, t_in, t_out, x_out = batch
                x_pred = self.model(x_in=x_in, t_in=t_in, t_out=t_out).clip(0, 1)
                x_pred_interpolated = x_pred[:, 1:-1].flatten(0, 1)
                x_out_interpolated = x_out[:, 1:-1].flatten(0, 1)
                x_pred_extrapolated = x_pred[:, [0, -1]].flatten(0, 1)
                x_out_extrapolated = x_out[:, [0, -1]].flatten(0, 1)
                x_pred = x_pred.flatten(0, 1)
                x_out = x_out.flatten(0, 1)
                if self.post_processing is not None:
                    x_pred = self.post_processing(x_pred)
                    x_out = self.post_processing(x_out)
                self.metrics_general.update(x_pred, x_out)
                self.metrics_interpolate.update(x_pred_interpolated, x_out_interpolated)
                self.metrics_extrapolate.update(x_pred_extrapolated, x_out_extrapolated)



    def compute_metrics(self) -> tuple[dict, ...]:
        """
        Computes the metrics and returns them in a dictionary.
        """
        return (
            self.metrics_general.compute(),
            self.metrics_interpolate.compute(),
            self.metrics_extrapolate.compute()
        )



