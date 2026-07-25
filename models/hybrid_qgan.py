import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits: int = 3, n_layers: int = 2, output_dim: int = 1):
        super().__init__()
        self.n_qubits, self.n_layers, self.output_dim = n_qubits, n_layers, output_dim
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.01)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")
        self.fc_out = nn.Linear(n_qubits, output_dim)

    def _circuit(self, inputs, weights):
        for i in range(self.n_qubits): qml.RY(inputs[i] * np.pi, wires=i)
        for layer in range(self.n_layers):
            for i in range(self.n_qubits): qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            for i in range(self.n_qubits - 1): qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 2: qml.CNOT(wires=[self.n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x.to(torch.float32), 0.0, 1.0)
        batch_results = []
        for i in range(x.shape[0]):
            res = self.qnode(x[i], self.weights)
            batch_results.append(torch.stack(res).to(torch.float32))
        out = self.fc_out(torch.stack(batch_results))
        return (out + 1) / 2

class HybridQGAN:
    def __init__(self, past: int = 3, future: int = 1, n_layers: int = 2, lr: float = 0.00016, batch_size: int = 128, epochs: int = 50, device: str = "cpu"):
        self.past, self.future, self.epochs, self.batch_size = past, future, epochs, batch_size
        self.device = torch.device(device)
        self.G = QuantumGenerator(n_qubits=past, n_layers=n_layers, output_dim=future)
        self.D = nn.Sequential(nn.Linear(future, 64), nn.LeakyReLU(0.2), nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1), nn.Sigmoid()).to(self.device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            g_loss_ep, d_loss_ep = 0.0, 0.0
            for x_batch, y_real in loader:
                y_real = y_real.to(self.device); bs = x_batch.size(0)
                real_labels, fake_labels = torch.ones(bs, 1, device=self.device), torch.zeros(bs, 1, device=self.device)
                y_fake = self.G(x_batch).to(self.device)
                # Train Discriminator
                self.opt_D.zero_grad()
                d_loss = (self.criterion(self.D(y_real), real_labels) + self.criterion(self.D(y_fake.detach()), fake_labels)) / 2
                d_loss.backward(); self.opt_D.step()
                # Train Generator
                self.opt_G.zero_grad()
                g_loss = self.criterion(self.D(self.G(x_batch).to(self.device)), real_labels)
                g_loss.backward(); self.opt_G.step()
                g_loss_ep += g_loss.item(); d_loss_ep += d_loss.item()
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.epochs - 1):
                print(f"  Epoch [{epoch+1}/{self.epochs}]  G_loss={g_loss_ep/len(loader):.4f}  D_loss={d_loss_ep/len(loader):.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.G.eval(); preds = self.G(torch.tensor(X, dtype=torch.float32)).detach().numpy(); self.G.train()
        return preds

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ck["G"]); self.D.load_state_dict(ck["D"])
