import torch
import torch.nn as nn

from collections.abc import Sequence

class EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        assert input_dim > 0, f"input_dim must be greater than 0, got {input_dim}"
        assert latent_dim > 0, f"latent_dim must be greater than 0, got {latent_dim}"
        
        layers = [torch.flatten(start_dim=1, end_dim=-1)]
        previous_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            assert isinstance(dim, int) and dim > 0, f"hidden_dim at index {idx} must be int and non zero but got type {type(dim)} and hidden_dim {dim}."
            layers += [nn.Linear(previous_dim, dim), nn.ReLU()]
            previous_dim = dim
        layers.append(nn.Linear(previous_dim, 2 * latent_dim))

        self.model = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.latent_dim = latent_dim        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        mu, log_var = out[:, :self.latent_dim], out[:, self.latent_dim:]
        return mu, log_var.clamp(-10, 10)
    
class DecoderMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        assert input_dim > 0, f"input_dim must be greater than 0, got {input_dim}."
        assert output_dim > 0, f"output_dim must be greater than 0, got {output_dim}."
        
        layers = []
        previous_dim = input_dim

        for idx, dim in enumerate(hidden_dims):
            assert isinstance(dim, int) and dim > 0, f"hidden_dim at index {idx} must be int and non zero but got type {type(dim)} and hidden_dim {dim}."
            layers += [nn.Linear(previous_dim, dim), nn.ReLU()]
            previous_dim = dim

        layers.append(nn.Linear(previous_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.encoder = EncoderMLP(input_dim, latent_dim, hidden_dims)
        self.decoder = DecoderMLP(latent_dim, input_dim, [x for x in reversed(hidden_dims)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)

        z = mu + std * torch.randn_like(std)

        x_recon = self.decoder(z)

        return x_recon, mu, log_var
    