import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AutoRec(nn.Module):
    """
    input -> hidden -> output(output.shape == input.shape)
    encoder: input -> hidden
    decoder: hidden -> output
    """
    def __init__(self, n_users, n_items, n_factors=200):
        super(AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items

        self.encoder = nn.Sequential(
            nn.Linear(self.n_users,self.n_factors),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_users),
            nn.Identity(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).to(device)

class MRMSELoss(nn.Module):
    """
    MaskedRootMSELoss() uses only observed ratings.
    According to docs(https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), 
    'mean' is set by default for 'reduction' and can be avoided by 'reduction="sum"'
    """
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction).to(device)
    def forward(self, pred, rating):
        mask = rating != 0
        masked_pred = pred * mask.float()
        num_observed = torch.sum(mask).to(device) if self.reduction == 'mean' else torch.Tensor([1.]).to(device)
        loss = torch.sqrt(self.mse(masked_pred, rating) / num_observed)
        return loss