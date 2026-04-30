import torch.nn as nn
import torch.nn.functional as F

class MatryoshkaTripletLoss(nn.Module):
    def __init__(self, nest_dims=[64, 128, 256, 384], margin=0.2, weights=None):
        super().__init__()
        self.nest_dims = sorted(nest_dims)
        self.margin = margin
        self.weights = weights if weights is not None else [1.0] * len(nest_dims)

        self.triplet_criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        total_loss = 0.0
        losses_dict = {}

        for dim, weight in zip(self.nest_dims, self.weights):
            a_slice = anchor[:, :dim]
            p_slice = positive[:, :dim]
            n_slice = negative[:, :dim]

            a_slice = F.normalize(a_slice, p=2, dim=1)
            p_slice = F.normalize(p_slice, p=2, dim=1)
            n_slice = F.normalize(n_slice, p=2, dim=1)

            step_loss = self.triplet_criterion(a_slice, p_slice, n_slice)

            total_loss += weight * step_loss
            losses_dict[f"loss_dim_{dim}"] = step_loss.item()

        return total_loss, losses_dict
