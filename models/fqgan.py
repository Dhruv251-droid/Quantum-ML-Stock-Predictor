import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

class FQGAN(nn.Module):
    def __init__(self, past: int = 3, future: int = 1, n_layers: int = 3):
        super().__init__()
        self.past, self.future, self.n_layers = past, future, n_layers
        # Invertible strategy: predict a window of size 'past' to create overlap
        self.n_f = math.ceil(math.log2(max(past, 2))) 
        self.gen_wires = list(range(1, self.n_f + 1))
        self.real_wires = list(range(self.n_f + 1, 2 * self.n_f + 1))
        self.ancilla, self.n_total = 0, 2 * self.n_f + 1
        self.weights = nn.Parameter(torch.randn(n_layers, len(self.gen_wires), 3) * 0.01)
        self.dev = qml.device("default.qubit", wires=self.n_total)
        self.qnode = qml.QNode(self._full_circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _pad(self, vec, size):
        if len(vec) < size: 
            padding = torch.zeros(size - len(vec), dtype=vec.dtype, device=vec.device)
            vec = torch.cat([vec, padding])
        else: 
            vec = vec[:size]
        norm = torch.norm(vec)
        if norm > 1e-9: return vec / norm
        res = torch.ones(size, dtype=vec.dtype, device=vec.device)
        return res / torch.norm(res)

    def _full_circuit(self, real_state, gen_input, weights):
        qml.Hadamard(wires=self.ancilla)
        qml.AmplitudeEmbedding(real_state, wires=self.real_wires, normalize=True, pad_with=0.0)
        qml.AmplitudeEmbedding(gen_input, wires=self.gen_wires, normalize=True, pad_with=0.0)
        for layer in range(self.n_layers):
            for i, w in enumerate(self.gen_wires): qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=w)
            for i in range(len(self.gen_wires) - 1): qml.CNOT(wires=[self.gen_wires[i], self.gen_wires[i + 1]])
        for rw, gw in zip(self.real_wires, self.gen_wires): qml.CSWAP(wires=[self.ancilla, rw, gw])
        qml.Hadamard(wires=self.ancilla)
        return qml.expval(qml.PauliZ(self.ancilla))

    def forward(self, x_real, x_past):
        scores = []
        for i in range(x_real.shape[0]):
            s = self.qnode(self._pad(x_real[i], 2**self.n_f).numpy(), self._pad(x_past[i], 2**self.n_f).numpy(), self.weights)
            scores.append(s.unsqueeze(0))
        return torch.stack(scores)

    def predict_with_norm(self, x_past):
        dev_gen = qml.device("default.qubit", wires=self.gen_wires)
        @qml.qnode(dev_gen, interface="torch")
        def gen_circ(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=self.gen_wires, normalize=True, pad_with=0.0)
            for l in range(self.n_layers):
                for i, w in enumerate(self.gen_wires): qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=w)
                for i in range(len(self.gen_wires) - 1): qml.CNOT(wires=[self.gen_wires[i], self.gen_wires[i + 1]])
            return qml.probs(wires=self.gen_wires)
        
        preds = []
        for i in range(x_past.shape[0]):
            p = gen_circ(self._pad(x_past[i], 2**self.n_f).numpy(), self.weights)
            p_vals = torch.sqrt(p[:self.past]) # amplitudes
            overlap_size = self.past - self.future
            if overlap_size > 0:
                known_overlap = x_past[i, -overlap_size:]
                pred_overlap = p_vals[:overlap_size]
                f = torch.mean(known_overlap / (pred_overlap + 1e-9))
                preds.append(p_vals[-self.future:] * f)
            else:
                preds.append(p_vals[-self.future:])
        return torch.stack(preds)

class FQGANTrainer:
    def __init__(self, past=3, future=1, n_layers=3, lr=0.016, batch_size=16, epochs=5):
        self.past, self.future = past, future
        self.model = FQGAN(past, future, n_layers); self.epochs, self.batch_size = epochs, batch_size
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, verbose=True):
        # Target for Invertible: window shifted by 'future'
        # Pad X_train to match shifted sequence lengths
        x_in = torch.tensor(X_train[:-self.future], dtype=torch.float32)
        x_target = torch.tensor(X_train[self.future:], dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_in, x_target), batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            loss_ep = 0.0
            for x, y in loader:
                self.opt.zero_grad(); loss = -self.model(y, x).mean(); loss.backward(); self.opt.step()
                if torch.isnan(self.model.weights).any():
                    torch.nan_to_num(self.model.weights.data, nan=0.01, out=self.model.weights.data)
                loss_ep += loss.item()
            if verbose: print(f"  Epoch [{epoch+1}/{self.epochs}]  Loss={loss_ep/len(loader):.6f}")

    def predict(self, X):
        return self.model.predict_with_norm(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    def save(self, path): os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(self.model.state_dict(), path)

    def load(self, path): self.model.load_state_dict(torch.load(path))
