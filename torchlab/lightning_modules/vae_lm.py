import torch
import lightning as L
from collections.abc import Sequence
from torchlab.models.vanilla_vae import VAE

class VAELM(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = VAE(input_dim=28*28,
                         latent_dim=64,
                         hidden_dims=[512, 256, 128])
        self.lr = lr

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int):
        x, _ = batch
        x_recon, mu, log_var = self.model(x)
        recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = (recon_loss + kl_loss) / x.size(0)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
