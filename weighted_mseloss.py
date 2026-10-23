"""The weighted loss function for GenCast training."""

from typing import Optional

import numpy as np
import torch


class WeightedMSELoss(torch.nn.Module):
    """Module WeightedMSELoss.

    This module implement the loss described in GenCast's paper.
    """

    def __init__(
        self,
        grid_lat: Optional[torch.Tensor] = None,
        pressure_levels: Optional[torch.Tensor] = None,
        num_atmospheric_features: Optional[int] = None,
        single_features_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize the WeightedMSELoss Module.

        More details about the features weights are reported in GraphCast's paper. In short, if the
        single features are "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure" and "total_precipitation_12hr", then it's suggested to set
        corresponding weights as 1, 0.1, 0.1, 0.1 and 0.1.

        Args:
            grid_lat (torch.Tensor, optional): 1D tensor containing all the latitudes.
            pressure_levels (torch.Tensor, optional): 1D tensor containing all the pressure levels
                per variable.
            num_atmospheric_features (int, optional): number of atmospheric features.
            single_features_weights (torch.Tensor, optional): 1D tensor containing single features
                weights.
        """
        super().__init__()

        area_weights = None
        features_weights = None

        if grid_lat is not None:
            area_weights = torch.abs(torch.cos(grid_lat * np.pi / 180.0))
            area_weights = area_weights / torch.mean(area_weights)
        if (
            pressure_levels is not None
            and num_atmospheric_features is not None
            and single_features_weights is not None
        ):
            # pressure_weights = pressure_levels / torch.sum(pressure_levels)
            pressure_weights = torch.cat((torch.ones(9*4), torch.full((7*4,), 1.5), torch.full((7*4,), 1)))
            # features_weights = torch.cat(
            #     (single_features_weights, pressure_weights.repeat(num_atmospheric_features)), dim=-1
            # )
            # features_weights = torch.cat(
            #     (single_features_weights, torch.repeat_interleave(pressure_weights, repeats=num_atmospheric_features)), dim=-1
            # )
            features_weights = torch.cat(
                (single_features_weights, torch.tensor([1, 1.1, 1.05, 1.05]).repeat(23) * pressure_weights), dim=-1
            )
            # features_weights = torch.tensor([1.2, 1.5,1.2,1,1, 1.5,1.2,1,1, 1.5,1.2,1,1, 1.2,1,1,1, 1.2,1,1,1, 1.2,1,1,1, 1.2,1,1,1,
            #                     1.5,1,1,1, 1.5,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1,
            #                     2,1,1,1, 1.5,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
        elif (
            pressure_levels is not None
            or num_atmospheric_features is not None
            or single_features_weights is not None
        ):
            raise ValueError(
                "Please to use features weights provide all three: pressure_levels,"
                "num_atmospheric_features and single_features_weights."
            )


        self.register_buffer("area_weights", area_weights, persistent=False)
        self.register_buffer("features_weights", features_weights, persistent=False)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        prev_inputs: Optional[torch.Tensor] = None,
        cli: Optional[np.array] = None,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred (torch.Tensor): prediction of the model [batch, lon, lat, var].
            noise_level (torch.Tensor): noise levels fed to the model for the corresponding
                predictions [batch, 1].
            target (torch.Tensor): target tensor [batch, lon, lat, var].

        Returns:
            torch.Tensor: weighted MSE loss.
        """
        # check shapes
        # if not (pred.shape == target.shape):
        #     raise ValueError(
        #         "Predictions and targets must have same shape. The actual shapes "
        #         f"are {pred.shape} and {target.shape}."
        #     )
        # if not (len(pred.shape) == 4):
        #     raise ValueError(
        #         "The expected shape for predictions and targets is "
        #         f"[batch, lon, lat, var], but got {pred.shape}."
        #     )
        if cli is not None:
            cli = cli.clone().detach()

        # compute loss
        pred = pred.squeeze(1)
        loss = ((pred - target) ** 2) * mask

        if torch.isnan(loss).any():
            raise ValueError("NaN values encountered in loss calculation.")

        # apply area and features weights to residuals
        if self.area_weights is not None:
            if not (len(self.area_weights) == loss.shape[2]):
                raise ValueError(
                    f"The size of grid_lat at initialization ({len(self.area_weights)}) "
                    f"and the number of latitudes in predictions ({pred.shape[2]}) "
                    "don't match."
                )
            loss *= self.area_weights[None, None, :, None]
        # compute acc loss
        if cli is not None:
            acc_loss = 1 - (torch.sum(((pred.mean(dim=1) + prev_inputs - cli) * (target + prev_inputs - cli))*mask*self.area_weights[None, None, :, None], dim=(1, 2)) /
                        torch.sqrt(
                            torch.sum((pred.mean(dim=1) + prev_inputs - cli)**2*self.area_weights[None, None, :, None]*mask,dim=(1, 2)) * torch.sum((target + prev_inputs - cli)**2*self.area_weights[None, None, :, None]*mask,dim=(1,2))))
            acc_loss = torch.mean(acc_loss, dim=3)

        if self.features_weights is not None:
            if not (len(self.features_weights) == pred.shape[-1]):
                raise ValueError(
                    f"The size of features weights at initialization ({len(self.features_weights)})"
                    f" and the number of features in predictions ({pred.shape[-1]}) "
                    "don't match."
                )
            loss *= self.features_weights[None, None, None, :]

        # compute means across lon, lat, var for each sample in the batch
        loss = loss.flatten(1).sum(-1)  # [batch]
        loss = loss / (mask.sum())

        # if cli is not None:
        #     # acc_loss *= self._lambda_sigma(noise_level).flatten()
        #     if epoch is not None and epoch > 35:
        #         return loss.mean() + 2 * acc_loss.mean()
        #     else:
        #         return loss.mean() + acc_loss.mean()
        return loss.mean()
