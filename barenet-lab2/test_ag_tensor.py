import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mygrad.engine import AGTensor

ATOL = 1e-4
RTOL = 1e-4

is_cuda = True

def _assert_close(actual: np.ndarray, expected: np.ndarray):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

def test_sum():
    a_np = np.arange(1, 4, dtype=np.float32).reshape(1, 3)

    a = AGTensor(a_np, is_cuda=is_cuda)
    out = a.sum()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_out = torch_a.sum()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

    b_np = np.arange(1.0, 7.0, dtype=np.float32).reshape(2, 3)
    for axis in (0, 1):
        print(f"Testing axis={axis}")
        b = AGTensor(b_np, is_cuda=is_cuda)
        out = b.sum(axis=axis)
        out.backward()

        torch_b = torch.tensor(b_np, requires_grad=True)
        torch_out = torch_b.sum(axis=axis).sum()
        torch_out.backward()

        _assert_close(out.sum().numpy(), torch_out.detach().cpu().numpy())
        _assert_close(b.grad.to_numpy(), torch_b.grad.detach().cpu().numpy())

def test_transpose():
    a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3) 

    a = AGTensor(a_np, is_cuda=is_cuda)
    out = a.T.mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_out = torch_a.T.mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

def test_mul():
    a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b_np = np.arange(7, 13, dtype=np.float32).reshape(2, 3)
    a = AGTensor(a_np, is_cuda=is_cuda)
    b = AGTensor(b_np, is_cuda=is_cuda)
    out = (a * b).sum()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_out = (torch_a * torch_b).sum()
    torch_out.backward()    
    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())
    _assert_close(b.grad.to_numpy(), torch_b.grad.detach().cpu().numpy())   

    c = 2.0
    a.grad.fill(0.0)
    b.grad.fill(0.0)
    out = (a * c).sum()
    out.backward()
    torch_a.grad.zero_()
    torch_out = (torch_a * c).sum()
    torch_out.backward()
    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())


def test_accum():
    a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    a = AGTensor(a_np, is_cuda=is_cuda)
    out = (a * a).sum()
    out.backward()
    out = (a * a).sum()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_out = (torch_a * torch_a).sum()
    torch_out.backward()    
    torch_out = (torch_a * torch_a).sum()
    torch_out.backward()
    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

def test_mean():    
    a_np = np.arange(1,10, dtype=np.float32).reshape(3,3)
    a = AGTensor(a_np, is_cuda=is_cuda)
    out = a.mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_out = torch_a.mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

def test_add():
    a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b_np = np.arange(6, 0, -1, dtype=np.float32).reshape(2, 3)

    a = AGTensor(a_np, is_cuda=is_cuda)
    b = AGTensor(b_np, is_cuda=is_cuda)
    out = (a + b).mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_out = (torch_a + torch_b).mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())
    _assert_close(b.grad.to_numpy(), torch_b.grad.detach().cpu().numpy())

    c = 2.0
    a.grad.fill(0.0)
    out = (a + c).mean()
    out.backward()
    torch_a.grad.zero_()
    torch_out = (torch_a + c).mean()
    torch_out.backward()
    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

def test_sub():
    a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b_np = np.arange(7, 13, dtype=np.float32).reshape(2, 3)

    a = AGTensor(a_np, is_cuda=is_cuda)
    b = AGTensor(b_np, is_cuda=is_cuda)
    out = (a - b).mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_out = (torch_a - torch_b).mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())
    _assert_close(b.grad.to_numpy(), torch_b.grad.detach().cpu().numpy())


def test_matmul():
    a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b_np = np.arange(1, 13, dtype=np.float32).reshape(3, 4)

    a = AGTensor(a_np, is_cuda=is_cuda)
    b = AGTensor(b_np, is_cuda=is_cuda)
    out = (a @ b).mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_out = (torch_a @ torch_b).mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())
    _assert_close(b.grad.to_numpy(), torch_b.grad.detach().cpu().numpy())


def test_relu():
    a_np = np.arange(-5, 7, dtype=np.float32).reshape(3, 4)

    a = AGTensor(a_np, is_cuda=is_cuda)
    out = a.relu().mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_out = torch.relu(torch_a).mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

def test_relu_cpu():
    a_np = np.arange(-5, 7, dtype=np.float32).reshape(3, 4)
    a = AGTensor(a_np, is_cuda=False)  # force CPU
    out = a.relu().mean()
    out.backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_out = torch.relu(torch_a).mean()
    torch_out.backward()

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())
    _assert_close(a.grad.to_numpy(), torch_a.grad.detach().cpu().numpy())

def test_cross_entropy():
    logits_np = np.arange(-5, 7, dtype=np.float32).reshape(4, 3)
    targets_np = np.array([0, 2, 1, 0], dtype=np.int32)

    logits = AGTensor(logits_np, is_cuda=is_cuda)
    loss = logits.cross_entropy_loss(targets_np)
    loss.backward()

    torch_logits = torch.tensor(logits_np, requires_grad=True)
    torch_targets = torch.tensor(targets_np, dtype=torch.long)
    torch_loss = F.cross_entropy(torch_logits, torch_targets, reduction="mean")
    torch_loss.backward()

    _assert_close(loss.numpy(), torch_loss.detach().cpu().numpy())
    _assert_close(logits.grad.to_numpy(), torch_logits.grad.detach().cpu().numpy())

def test_argmax():
    a_np = np.array([[1, 3, 2], [4, 0, 5]], dtype=np.float32)

    a = AGTensor(a_np, is_cuda=is_cuda)
    out = a.argmax()
    print(f'out: {out.numpy()}')

    torch_a = torch.tensor(a_np, requires_grad=False)
    torch_out = torch_a.argmax(axis=1, keepdims=True)
    print(f'torch_out: {torch_out.detach().cpu().numpy()}')

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())

def test_eq():
    a_np = np.array([[1, 3, 2], [4, 0, 5]], dtype=np.float32)
    b_np = np.array([[1, 0, 2], [4, 5, 5]], dtype=np.float32)

    a = AGTensor(a_np, is_cuda=is_cuda)
    b = AGTensor(b_np, is_cuda=is_cuda)
    out = (a == b)

    torch_a = torch.tensor(a_np, requires_grad=False)
    torch_b = torch.tensor(b_np, requires_grad=False)
    torch_out = (torch_a == torch_b)

    _assert_close(out.numpy(), torch_out.detach().cpu().numpy())

def test_2layer_mlp():
    seed=123
    Bsz, Din, Hdim, Dout = 8, 5, 7, 3
    rng = np.random.default_rng(seed)

    X_np = rng.standard_normal((Bsz, Din)).astype(np.float32)
    Y_np  = rng.standard_normal((Bsz, Dout)).astype(np.float32)

    W1_np = rng.standard_normal((Din, Hdim)).astype(np.float32)
    b1_np = rng.standard_normal((1, Hdim)).astype(np.float32)
    W2_np = rng.standard_normal((Hdim, Dout)).astype(np.float32)
    b2_np = rng.standard_normal((1, Dout)).astype(np.float32)

    x  = AGTensor(X_np, requires_grad=False, is_cuda=is_cuda)
    y  = AGTensor(Y_np, requires_grad=False, is_cuda=is_cuda)
    W1 = AGTensor(W1_np, is_cuda=is_cuda) ; b1 = AGTensor(b1_np, is_cuda=is_cuda)
    W2 = AGTensor(W2_np, is_cuda=is_cuda) ; b2 = AGTensor(b2_np, is_cuda=is_cuda)

    x2 = (x @ W1 + b1).relu()
    yhat = x2 @ W2 + b2
    loss_ours = ((yhat - y) * (yhat - y)).mean()  # MSE

    loss_ours.backward()


    torch.manual_seed(seed)  # only affects torch-created tensors; we copy from numpy anyway

    X_t  = torch.tensor(X_np, dtype=torch.float32, requires_grad=False)
    Y_t  = torch.tensor(Y_np, dtype=torch.float32, requires_grad=False)

    W1_t = torch.tensor(W1_np, dtype=torch.float32, requires_grad=True)
    b1_t = torch.tensor(b1_np, dtype=torch.float32, requires_grad=True)
    W2_t = torch.tensor(W2_np, dtype=torch.float32, requires_grad=True)
    b2_t = torch.tensor(b2_np, dtype=torch.float32, requires_grad=True)

    h = torch.relu(X_t @ W1_t + b1_t)
    yhat = h @ W2_t + b2_t
    loss_t = torch.mean((yhat - Y_t) ** 2)

    loss_t.backward()

    _assert_close(loss_ours.numpy(), loss_t.detach().cpu().numpy())
    _assert_close(W1.grad.to_numpy(), W1_t.grad.detach().cpu().numpy())
    _assert_close(b1.grad.to_numpy(), b1_t.grad.detach().cpu().numpy())
    _assert_close(W2.grad.to_numpy(), W2_t.grad.detach().cpu().numpy())
    _assert_close(b2.grad.to_numpy(), b2_t.grad.detach().cpu().numpy())


