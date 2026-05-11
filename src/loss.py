
import torch.nn as nn
import torch.nn.functional as F

class MatryoshkaTripletLoss(nn.Module):
    def __init__(self, nest_dims=[64, 128, 256, 384], margin=0.3, weights=None):
        super().__init__()
        self.nest_dims = sorted(nest_dims)
        self.margin = margin
        self.weights = weights if weights is not None else [1.0] * len(nest_dims)
        
        distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.triplet_criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, 
            margin=self.margin
        )

    def forward(self, anchor, positive, negative):
        total_loss = 0.0
        losses_dict = {}

        for dim, weight in zip(self.nest_dims, self.weights):
            a_s = F.normalize(anchor[:, :dim], p=2, dim=1)
            p_s = F.normalize(positive[:, :dim], p=2, dim=1)
            n_s = F.normalize(negative[:, :dim], p=2, dim=1)

            step_loss = self.triplet_criterion(a_s, p_s, n_s)
            total_loss += weight * step_loss
            losses_dict[f"loss_dim_{dim}"] = step_loss.item()

        return total_loss, losses_dict