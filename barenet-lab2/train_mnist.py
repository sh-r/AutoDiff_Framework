#!/usr/bin/env python3
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mygrad.engine import AGTensor, no_grad   

def make_loaders(batch_size=128):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1).numpy().astype(np.float32))])
    class NumpyMNIST(torch.utils.data.Dataset):
        def __init__(self, train): self.ds = datasets.MNIST("./data", train=train, download=True)
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            img, y = self.ds[idx]
            x = tfm(img)                 # (784,) float32 numpy
            return x, np.int8(y)
    train_ds = NumpyMNIST(True); test_ds = NumpyMNIST(False)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=1000)

def param(shape_hw, is_cuda=True):
    h, w = shape_hw
    rng = np.random.default_rng()
    arr = (rng.standard_normal((h, w)).astype(np.float32) * 0.01)
    return AGTensor(arr, is_cuda=is_cuda)

def sgd_step(params, lr):
    for p in params:
        if p.grad is None: continue        
        p.data -= p.grad * lr
        p.grad.fill(0.0)

def accuracy_from_logits(logits: AGTensor, y_np: np.ndarray) -> float:
    # Predicted indices (AGTensor, shape (N,1))
    pred = logits.argmax()

    y_mt = AGTensor(y_np.astype(np.uint32).reshape(-1, 1), requires_grad=False, is_cuda=logits.data.is_cuda)  # (N,1) AGTensor
    sum_t = (pred == y_mt).sum()  # AGTensor, shape (1,1)
    out = sum_t.data.to_numpy()  # Get NumPy array 
    return out[0, 0]/float(pred.data.shape[0])

def train(epochs=20, batch_size=128, lr=0.1, hidden=128, is_cuda=True):
    train_loader, test_loader = make_loaders(batch_size)

    # 784 -> hidden -> 10
    W1 = param((784, hidden), is_cuda=is_cuda) 
    b1 = param((1, hidden), is_cuda=is_cuda)
    W2 = param((hidden, 10), is_cuda=is_cuda)  
    b2 = param((1, 10), is_cuda=is_cuda)
    params = [W1, b1, W2, b2]

    for epoch in range(1, epochs+1):
        epoch_start = time.perf_counter()
        total_loss = 0.0; n = 0
        for xb, yb in train_loader:
            xb = torch.stack(xb).numpy() if isinstance(xb, list) else np.asarray(xb)
            yb = torch.as_tensor(yb).numpy() if isinstance(yb, list) else np.asarray(yb)

            x = AGTensor(xb, is_cuda=is_cuda)               # (N,784)
            tt = x@W1
            tt1 = tt + b1
            tt2 = tt1.relu()
            h = (x @ W1 + b1).relu()
            logits = h @ W2 + b2

            loss = logits.cross_entropy_loss(yb)  # fused CE (CPU)
            total_loss += float(loss.numpy()[0,0]) * xb.shape[0]
            n += xb.shape[0]

            loss.backward()
            sgd_step(params, lr)

        train_latency = time.perf_counter() - epoch_start
        print(f"Epoch {epoch}: train loss = {total_loss / n:.4f} (latency {train_latency:.2f}s)")

        # quick test accuracy
        correct = 0; count = 0
        for xb, yb in test_loader:
            xb = torch.stack(xb).numpy() if isinstance(xb, list) else np.asarray(xb)
            yb = torch.as_tensor(yb).numpy() if isinstance(yb, list) else np.asarray(yb)
            with no_grad():
                x = AGTensor(xb, is_cuda=is_cuda)
                h = (x @ W1 + b1).relu()
                logits = h @ W2 + b2
                correct += int(accuracy_from_logits(logits, yb) * xb.shape[0])
                count += xb.shape[0]
        print(f"  Test acc: {correct / count:.4f}")

if __name__ == "__main__":
    train(epochs=20, batch_size=128, lr=0.1, hidden=128, is_cuda=True)
