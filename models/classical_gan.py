import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (batch, past_window, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        return self.fc(out)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ClassicalGAN:
    def __init__(self, past: int = 3, future: int = 1, hidden_dim: int = 64, lr: float = 0.00016, batch_size: int = 128, epochs: int = 50, device: str = "cpu"):
        self.past, self.future, self.epochs, self.batch_size = past, future, epochs, batch_size
        self.device = torch.device(device)
        self.G = Generator(past, hidden_dim, future).to(self.device)
        self.D = Discriminator(future).to(self.device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            g_loss_ep, d_loss_ep = 0.0, 0.0
            for x_batch, y_real in loader:
                x_batch, y_real = x_batch.to(self.device), y_real.to(self.device)
                bs = x_batch.size(0)
                real_labels, fake_labels = torch.ones(bs, 1, device=self.device), torch.zeros(bs, 1, device=self.device)

                # Train Discriminator
                self.opt_D.zero_grad()
                d_real = self.D(y_real)
                y_fake = self.G(x_batch).detach()
                d_fake = self.D(y_fake)
                d_loss = (self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels)) / 2
                d_loss.backward(); self.opt_D.step()

                # Train Generator
                self.opt_G.zero_grad()
                y_fake = self.G(x_batch)
                g_loss = self.criterion(self.D(y_fake), real_labels)
                g_loss.backward(); self.opt_G.step()
                g_loss_ep += g_loss.item(); d_loss_ep += d_loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}]  G_loss={g_loss_ep/len(loader):.4f}  D_loss={d_loss_ep/len(loader):.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.G.eval(); preds = self.G(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy(); self.G.train()
        return preds

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ck["G"]); self.D.load_state_dict(ck["D"])
