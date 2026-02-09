import numpy as np
from typing import Union, Optional, Tuple
import sys; sys.path.append("build")
import bten
from contextlib import contextmanager

_grad_enabled = True
def is_grad_enabled():
    return _grad_enabled
    
@contextmanager
def no_grad():
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old
from numbers import Number

# -------- autograd wrapper --------

class AGTensor:
    """
    Minimal autograd wrapper over bten.TensorF (and/or) bten.TensorU32.
    Forward ops: +, *, -, @, relu, mean, cross_entropy_loss
    non-differentiable ops: ==, argmax
    Backward: autodiff via backward() call.

    data is tensor that AGTensor wraps. It can be either bten.TensorF or bten.TensorU32 or a numpy array.
    If data is a numpy array, it will be converted to bten.TensorF or bten.TensorU32 based on its dtype.

    children is the set of AGTensors that this tensor's gradients should propagate to. Alternatively, you can 
    think of these children as the parents used to compute this AGTensor.

    op is the operation that produced this tensor, for debugging purposes only.

    requires_grad is a boolean flag indicating whether to track gradients for this tensor.

    is_cuda is an optional boolean flag indicating whether to store the data on GPU or CPU if data is a numpy array.
    """
    def __init__(self, data, children=(), op='', requires_grad=True, is_cuda=True):
        if isinstance(data, bten.TensorF) or isinstance(data, bten.TensorU32):
            self.data = data

        elif isinstance(data, np.ndarray):
            h, w = data.shape
            if data.dtype == np.float32:
                t = bten.TensorF(h, w, is_cuda)
            elif data.dtype == np.uint32:
                t = bten.TensorU32(h, w, is_cuda)
            t.copy_from_numpy(data)
            self.data = t

        elif isinstance(data, (float, int)):
            t = bten.TensorF(1, 1, is_cuda)
            t.copy_from_numpy(np.array([[data]], dtype=np.float32))
            self.data = t

        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    @property
    def shape(self):
         return self.data.shape

    @property
    def is_cuda(self): 
        return self.data.is_cuda

    # a = input
    # b = a + 3
    # c = b * 2
    # loss = sum(c)

    # Backward:
    # dloss/dc = 1
    # dc/db = 2 # c depends on b
    # db/da = 1 # b depends on a
    # dloss/da = dloss/dc * dc/db * db/da = 1 * 2 * 1 = 2
    # dloss/db = dloss/dc * dc/db = 1 * 2 = 2

    # But variable wise if we store grads:
    # - a.grad = dloss/da = 2
    # - b.grad = dloss/b = 2
    # - c.grad = dloss/dc = 1

    def __hash__(self):
        return id(self) # Hash by object identity — needed only because we use sets for _prev
    

    #     # a = x + y
    #     # b = x + 2
    #     # out = a + b
    #     # dout/dx = dout/da * da/dx  +  dout/db * db/dx

    #     # here x is self and y is other
    #     # when we call out.backward, ultimately a._backward also called
    #     # so out.grad would be dloss/da or dout/da

    #     # ex1: only 1 var depends on x or y
    #     # a = x + y
    #     # da/dx = 1, da/dy = 1
    #     # x.grad = dloss/dx = dloss/da * da/dx = a.grad * 1 = out.grad
    #     # y.grad = dloss/dy = dloss/da * da/dy = a.grad * 1 = other.grad

    #     # But what if 2 vars depend on x or y?
    #     # ex2: 2 vars depend on x
    #     # a = x + y; b = x + 2; loss = a + b
    #     # da/dx=1, db/dx=1
    #     # x.grad = dloss/dx = dloss/da * da/dx + dloss/db * db/dx = a.grad * 1 + b.grad * 1 so that's why += is used

    # def __add__(self, other: 'AGTensor | float | int'):  # Python 3.10+
    #     """
    #     Elementwise addition with broadcasting: self + other
    #     other can be an AGTensor, float, or int.
    #     """
    #     # Lab-2: add your code here

    #     # first wrapping other as AGTensor this is what we add to data and children is none
    #     # the requires_grad is false because constant does not need grad
    #     if isinstance(other, AGTensor):
    #         out_data = self.data + other.data
    #         requires_grad = self.requires_grad or other.requires_grad
    #         prev = {self, other}
    #     elif isinstance(other, (float, int)):
    #         out_data = self.data + float(other)
    #         requires_grad = self.requires_grad
    #         prev = {self}

    #     out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
    #     out._prev = prev

    #     def _backward():
    #         if self.requires_grad:
    #             if self.grad is None:
    #                 self.grad = out.grad
    #             else:
    #                 self.grad += out.grad

    #         if isinstance(other, AGTensor) and other.requires_grad:
    #             if other.grad is None:
    #                 other.grad = out.grad
    #             else:
    #                 other.grad += out.grad

    #     out._backward = _backward
    #     return out


    # def __sub__(self, other: 'AGTensor | float | int'):
    #     if isinstance(other, AGTensor):
    #         out_data = self.data - other.data
    #         requires_grad = self.requires_grad or other.requires_grad
    #         prev = {self, other}
    #     elif isinstance(other, (float, int)):
    #         out_data = self.data - float(other)
    #         requires_grad = self.requires_grad
    #         prev = {self}

    #     out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
    #     out._prev = prev

    #     out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
    #     out._prev = prev

    #     def _backward():
    #         if self.requires_grad:
    #             if self.grad is None:
    #                 self.grad = out.grad
    #             else:
    #                 self.grad += out.grad

    #         if isinstance(other, AGTensor) and other.requires_grad:
    #             if other.grad is None:
    #                 other.grad = out.grad * (-1)
    #             else:
    #                 other.grad += out.grad * (-1)

    #     out._backward = _backward
    #     return out

    # def __mul__(self, other):
    #     """
    #     Elementwise multiplication with broadcasting: self * other
    #     other can be an AGTensor, float, or int.
    #     """
    #     if isinstance(other, AGTensor):
    #         out_data = self.data * other.data
    #         requires_grad = self.requires_grad or other.requires_grad
    #         prev = {self, other}
    #     elif isinstance(other, (float, int)):
    #         out_data = self.data * float(other)  # scalar overload
    #         requires_grad = self.requires_grad
    #         prev = {self}

    #     out = AGTensor(out_data,
    #                 requires_grad=requires_grad,
    #                 is_cuda=self.is_cuda)
    #     out._prev = prev

    #     def _backward():
    #         # d(a*b)/da = b, d(a*b)/db = a
    #         if self.requires_grad:
    #             grad_term = other.data if isinstance(other, AGTensor) else float(other)
    #             if self.grad is None:
    #                 self.grad = out.grad * grad_term
    #             else:
    #                 self.grad += out.grad * grad_term

    #         if isinstance(other, AGTensor) and other.requires_grad:
    #             if other.grad is None:
    #                 other.grad = out.grad * self.data
    #             else:
    #                 other.grad += out.grad * self.data

    #     out._backward = _backward
    #     return out

    # @property
    # def T(self):
    #     """
    #     Transpose of a 2D tensor. This function is only partially filled. 
    #     You need to add the backward() function.
    #     """
    #     out_data = self.data.T
    #     out = AGTensor(out_data, requires_grad=self.requires_grad, is_cuda=self.is_cuda)
    #     out._prev = {self}

    #     def _backward():
    #         if self.requires_grad:
    #             grad_in = out.grad.T
    #             if self.grad is None:
    #                 self.grad = grad_in
    #             else:
    #                 self.grad += grad_in
    #     out._backward = _backward
    #     return out
    
    # def __add__(self, other: 'AGTensor | float | int'):
    #     """
    #     Elementwise addition with broadcasting: self + other
    #     other can be an AGTensor, float, or int.
    #     """
    #     if isinstance(other, AGTensor):
    #         out_data = self.data + other.data
    #         requires_grad = self.requires_grad or other.requires_grad
    #         prev = {self, other}
    #     elif isinstance(other, (float, int)):
    #         out_data = self.data + float(other)
    #         requires_grad = self.requires_grad
    #         prev = {self}

    #     out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
    #     out._prev = prev

    #     def _backward():
    #         if self.requires_grad:
    #             grad_self = out.grad
                
    #             # Handle broadcasting: if self was broadcasted, sum gradients back
    #             if self.data.shape != out.grad.shape:
    #                 # Need to sum gradients to match self's shape
    #                 h_self, w_self = self.data.shape
    #                 h_grad, w_grad = out.grad.shape
                    
    #                 grad_self = out.grad
    #                 if h_self == 1 and h_grad > 1:
    #                     grad_self = grad_self.sum(0)  # sum along axis 0
    #                 if w_self == 1 and w_grad > 1:
    #                     grad_self = grad_self.sum(1)  # sum along axis 1
                
    #             if self.grad is None:
    #                 self.grad = grad_self
    #             else:
    #                 self.grad += grad_self

    #         if isinstance(other, AGTensor) and other.requires_grad:
    #             grad_other = out.grad
                
    #             # Handle broadcasting: if other was broadcasted, sum gradients back
    #             if other.data.shape != out.grad.shape:
    #                 h_other, w_other = other.data.shape
    #                 h_grad, w_grad = out.grad.shape
                    
    #                 grad_other = out.grad
    #                 if h_other == 1 and h_grad > 1:
    #                     grad_other = grad_other.sum(0)  # sum along axis 0
    #                 if w_other == 1 and w_grad > 1:
    #                     grad_other = grad_other.sum(1)  # sum along axis 1
                
    #             if other.grad is None:
    #                 other.grad = grad_other
    #             else:
    #                 other.grad += grad_other

    #     out._backward = _backward
    #     return out

    # def __sub__(self, other: 'AGTensor | float | int'):
    #     if isinstance(other, AGTensor):
    #         out_data = self.data - other.data
    #         requires_grad = self.requires_grad or other.requires_grad
    #         prev = {self, other}
    #     elif isinstance(other, (float, int)):
    #         out_data = self.data - float(other)
    #         requires_grad = self.requires_grad
    #         prev = {self}

    #     out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
    #     out._prev = prev

    #     def _backward():
    #         if self.requires_grad:
    #             grad_self = out.grad
                
    #             # Handle broadcasting for self
    #             if self.data.shape != out.grad.shape:
    #                 h_self, w_self = self.data.shape
    #                 h_grad, w_grad = out.grad.shape
                    
    #                 grad_self = out.grad
    #                 if h_self == 1 and h_grad > 1:
    #                     grad_self = grad_self.sum(0)
    #                 if w_self == 1 and w_grad > 1:
    #                     grad_self = grad_self.sum(1)
                
    #             if self.grad is None:
    #                 self.grad = grad_self
    #             else:
    #                 self.grad += grad_self

    #         if isinstance(other, AGTensor) and other.requires_grad:
    #             grad_other = out.grad * (-1)
                
    #             # Handle broadcasting for other
    #             if other.data.shape != out.grad.shape:
    #                 h_other, w_other = other.data.shape
    #                 h_grad, w_grad = out.grad.shape
                    
    #                 grad_other = out.grad * (-1)
    #                 if h_other == 1 and h_grad > 1:
    #                     grad_other = grad_other.sum(0)
    #                 if w_other == 1 and w_grad > 1:
    #                     grad_other = grad_other.sum(1)
                
    #             if other.grad is None:
    #                 other.grad = grad_other
    #             else:
    #                 other.grad += grad_other

    #     out._backward = _backward
    #     return out


    # def __mul__(self, other):
    #     """
    #     Elementwise multiplication with broadcasting: self * other
    #     other can be an AGTensor, float, or int.
    #     """
    #     if isinstance(other, AGTensor):
    #         out_data = self.data * other.data
    #         requires_grad = self.requires_grad or other.requires_grad
    #         prev = {self, other}
    #     elif isinstance(other, (float, int)):
    #         out_data = self.data * float(other)
    #         requires_grad = self.requires_grad
    #         prev = {self}

    #     out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
    #     out._prev = prev

    #     def _backward():
    #         # d(a*b)/da = b, d(a*b)/db = a
    #         if self.requires_grad:
    #             grad_term = other.data if isinstance(other, AGTensor) else float(other)
    #             grad_self = out.grad * grad_term
                
    #             # Handle broadcasting for self
    #             if self.data.shape != out.grad.shape:
    #                 h_self, w_self = self.data.shape
    #                 h_grad, w_grad = out.grad.shape
                    
    #                 if h_self == 1 and h_grad > 1:
    #                     grad_self = grad_self.sum(0)
    #                 if w_self == 1 and w_grad > 1:
    #                     grad_self = grad_self.sum(1)
                
    #             if self.grad is None:
    #                 self.grad = grad_self
    #             else:
    #                 self.grad += grad_self

    #         if isinstance(other, AGTensor) and other.requires_grad:
    #             grad_other = out.grad * self.data
                
    #             # Handle broadcasting for other
    #             if other.data.shape != out.grad.shape:
    #                 h_other, w_other = other.data.shape
    #                 h_grad, w_grad = out.grad.shape
                    
    #                 if h_other == 1 and h_grad > 1:
    #                     grad_other = grad_other.sum(0)
    #                 if w_other == 1 and w_grad > 1:
    #                     grad_other = grad_other.sum(1)
                
    #             if other.grad is None:
    #                 other.grad = grad_other
    #             else:
    #                 other.grad += grad_other

    #     out._backward = _backward
    #     return out

    # def __matmul__(self, other: 'AGTensor'):
    #     out_data = self.data @ other.data
    #     out = AGTensor(out_data, requires_grad=self.requires_grad or other.requires_grad, is_cuda=self.is_cuda)
    #     out._prev = {self, other}

    #     def _backward():
    #         # d(A @ B)/dA = grad @ B.T
    #         # d(A @ B)/dB = A.T @ grad
    #         if self.requires_grad:
    #             grad_A = out.grad @ other.data.T
    #             if self.grad is None:
    #                 self.grad = grad_A
    #             else:
    #                 self.grad += grad_A
    #         if other.requires_grad:
    #             grad_B = self.data.T @ out.grad
    #             if other.grad is None:
    #                 other.grad = grad_B
    #             else:
    #                 other.grad += grad_B
    #     out._backward = _backward
    #     return out


    # def relu(self):
    #     """
    #     Elementwise ReLU. 
    #     Forward: y = max(0, x)
    #     Backward: uses self.data.relu_back for d/dx.
    #     """
    #     # Call the bound Tensor method, not a top-level op
    #     out = AGTensor(self.data.relu(), requires_grad=self.requires_grad)
    #     out._prev = {self}

    #     def _backward():
    #         if self.requires_grad:
    #             # Again, use the bound method on Tensor
    #             grad_in = self.data.relu_back(out.grad)
    #             if self.grad is None:
    #                 self.grad = grad_in
    #             else:
    #                 self.grad += grad_in

    #     out._backward = _backward
    #     return out


    # def sum(self, axis: Optional[int] = None):
    #     """Sum reduction. axis=None → scalar, axis=0 or 1 → reduce along that axis."""
    #     if axis is None:
    #         out_data = self.data.sum(1).sum(0)
    #     elif axis == 0:
    #         out_data = self.data.sum(1)
    #     elif axis == 1:
    #         out_data = self.data.sum(0)

    #     out = AGTensor(out_data, requires_grad=self.requires_grad, is_cuda=self.is_cuda)
    #     out._prev = {self}

    #     # ---- Backward ----
    #     def _backward():
    #         if not self.requires_grad:
    #             return

    #         h, w = self.data.shape
    #         ones = bten.TensorF(h, w, self.is_cuda)
    #         ones.fill(1.0)

    #         if axis is None:
    #             # Every element gets same scalar grad
    #             grad_scalar = out.grad.to_numpy()[0, 0]
    #             grad_in = ones * grad_scalar

    #         elif axis == 0:
    #             grad_in = ones * out.grad

    #         elif axis == 1:
    #             grad_in = ones * out.grad

    #         if self.grad is None:
    #             self.grad = grad_in
    #         else:
    #             self.grad += grad_in

    #     out._backward = _backward
    #     return out

    # def cross_entropy_loss(self, targets: np.ndarray):
    #     h, w = self.shape
    #     d_logits = bten.TensorF(h, w, self.is_cuda)
    #     t_targets = bten.TensorU32(targets.shape[0], 1, self.is_cuda)
    #     t_targets.copy_from_numpy(targets.reshape(-1, 1).astype(np.uint32))

    #     loss_val = self.data.cross_entropy_loss(t_targets, d_logits)

    #     out_data = bten.TensorF(1, 1, self.is_cuda)
    #     out_data.copy_from_numpy(np.array([[loss_val]], dtype=np.float32))

    #     out = AGTensor(out_data, requires_grad=self.requires_grad, is_cuda=self.is_cuda)
    #     out._prev = {self}

    #     def _backward():
    #         if not self.requires_grad:
    #             return
    #         if self.grad is None:
    #             self.grad = d_logits
    #         else:
    #             self.grad += d_logits

    #     out._backward = _backward
    #     return out


    def __add__(self, other: 'AGTensor | float | int'):
        """
        Elementwise addition with broadcasting: self + other
        other can be an AGTensor, float, or int.
        """
        if isinstance(other, AGTensor):
            out_data = self.data + other.data
            requires_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
            prev = {self, other} if requires_grad else set()
        elif isinstance(other, (float, int)):
            out_data = self.data + float(other)
            requires_grad = self.requires_grad and is_grad_enabled()
            prev = {self} if requires_grad else set()

        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = prev

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                
                # Handle broadcasting: if self was broadcasted, sum gradients back
                if self.data.shape != out.grad.shape:
                    h_self, w_self = self.data.shape
                    h_grad, w_grad = out.grad.shape
                    
                    grad_self = out.grad
                    if h_self == 1 and h_grad > 1:
                        grad_self = grad_self.sum(0)  # sum along axis 0
                    if w_self == 1 and w_grad > 1:
                        grad_self = grad_self.sum(1)  # sum along axis 1
                
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

            if isinstance(other, AGTensor) and other.requires_grad:
                grad_other = out.grad
                
                # Handle broadcasting: if other was broadcasted, sum gradients back
                if other.data.shape != out.grad.shape:
                    h_other, w_other = other.data.shape
                    h_grad, w_grad = out.grad.shape
                    
                    grad_other = out.grad
                    if h_other == 1 and h_grad > 1:
                        grad_other = grad_other.sum(0)  # sum along axis 0
                    if w_other == 1 and w_grad > 1:
                        grad_other = grad_other.sum(1)  # sum along axis 1
                
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward if requires_grad else lambda: None
        return out


    def __sub__(self, other: 'AGTensor | float | int'):
        if isinstance(other, AGTensor):
            out_data = self.data - other.data
            requires_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
            prev = {self, other} if requires_grad else set()
        elif isinstance(other, (float, int)):
            out_data = self.data - float(other)
            requires_grad = self.requires_grad and is_grad_enabled()
            prev = {self} if requires_grad else set()

        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = prev

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                
                # Handle broadcasting for self
                if self.data.shape != out.grad.shape:
                    h_self, w_self = self.data.shape
                    h_grad, w_grad = out.grad.shape
                    
                    grad_self = out.grad
                    if h_self == 1 and h_grad > 1:
                        grad_self = grad_self.sum(0)
                    if w_self == 1 and w_grad > 1:
                        grad_self = grad_self.sum(1)
                
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

            if isinstance(other, AGTensor) and other.requires_grad:
                grad_other = out.grad * (-1)
                
                # Handle broadcasting for other
                if other.data.shape != out.grad.shape:
                    h_other, w_other = other.data.shape
                    h_grad, w_grad = out.grad.shape
                    
                    grad_other = out.grad * (-1)
                    if h_other == 1 and h_grad > 1:
                        grad_other = grad_other.sum(0)
                    if w_other == 1 and w_grad > 1:
                        grad_other = grad_other.sum(1)
                
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward if requires_grad else lambda: None
        return out


    def __mul__(self, other):
        """
        Elementwise multiplication with broadcasting: self * other
        other can be an AGTensor, float, or int.
        """
        if isinstance(other, AGTensor):
            out_data = self.data * other.data
            requires_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
            prev = {self, other} if requires_grad else set()
        elif isinstance(other, (float, int)):
            out_data = self.data * float(other)
            requires_grad = self.requires_grad and is_grad_enabled()
            prev = {self} if requires_grad else set()

        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = prev

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            if self.requires_grad:
                grad_term = other.data if isinstance(other, AGTensor) else float(other)
                grad_self = out.grad * grad_term
                
                # Handle broadcasting for self
                if self.data.shape != out.grad.shape:
                    h_self, w_self = self.data.shape
                    h_grad, w_grad = out.grad.shape
                    
                    if h_self == 1 and h_grad > 1:
                        grad_self = grad_self.sum(0)
                    if w_self == 1 and w_grad > 1:
                        grad_self = grad_self.sum(1)
                
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

            if isinstance(other, AGTensor) and other.requires_grad:
                grad_other = out.grad * self.data
                
                # Handle broadcasting for other
                if other.data.shape != out.grad.shape:
                    h_other, w_other = other.data.shape
                    h_grad, w_grad = out.grad.shape
                    
                    if h_other == 1 and h_grad > 1:
                        grad_other = grad_other.sum(0)
                    if w_other == 1 and w_grad > 1:
                        grad_other = grad_other.sum(1)
                
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward if requires_grad else lambda: None
        return out


    def __matmul__(self, other: 'AGTensor'):
        out_data = self.data @ other.data
        requires_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = {self, other} if requires_grad else set()

        def _backward():
            # d(A @ B)/dA = grad @ B.T
            # d(A @ B)/dB = A.T @ grad
            if self.requires_grad:
                grad_A = out.grad @ other.data.T
                if self.grad is None:
                    self.grad = grad_A
                else:
                    self.grad += grad_A
            if other.requires_grad:
                grad_B = self.data.T @ out.grad
                if other.grad is None:
                    other.grad = grad_B
                else:
                    other.grad += grad_B
        
        out._backward = _backward if requires_grad else lambda: None
        return out


    def relu(self):
        """
        Elementwise ReLU. 
        Forward: y = max(0, x)
        Backward: uses self.data.relu_back for d/dx.
        """
        requires_grad = self.requires_grad and is_grad_enabled()
        out = AGTensor(self.data.relu(), requires_grad=requires_grad)
        out._prev = {self} if requires_grad else set()

        def _backward():
            if self.requires_grad:
                grad_in = self.data.relu_back(out.grad)
                if self.grad is None:
                    self.grad = grad_in
                else:
                    self.grad += grad_in

        out._backward = _backward if requires_grad else lambda: None
        return out


    def sum(self, axis: Optional[int] = None):
        """Sum reduction. axis=None → scalar, axis=0 or 1 → reduce along that axis."""
        if axis is None:
            out_data = self.data.sum(1).sum(0)
        elif axis == 0:
            out_data = self.data.sum(1)
        elif axis == 1:
            out_data = self.data.sum(0)

        requires_grad = self.requires_grad and is_grad_enabled()
        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = {self} if requires_grad else set()

        def _backward():
            if not self.requires_grad:
                return

            h, w = self.data.shape
            ones = bten.TensorF(h, w, self.is_cuda)
            ones.fill(1.0)

            if axis is None:
                # Every element gets same scalar grad
                grad_scalar = out.grad.to_numpy()[0, 0]
                grad_in = ones * grad_scalar

            elif axis == 0:
                grad_in = ones * out.grad

            elif axis == 1:
                grad_in = ones * out.grad

            if self.grad is None:
                self.grad = grad_in
            else:
                self.grad += grad_in

        out._backward = _backward if requires_grad else lambda: None
        return out


    def cross_entropy_loss(self, targets: np.ndarray):
        h, w = self.shape
        d_logits = bten.TensorF(h, w, self.is_cuda)
        t_targets = bten.TensorU32(targets.shape[0], 1, self.is_cuda)
        t_targets.copy_from_numpy(targets.reshape(-1, 1).astype(np.uint32))

        loss_val = self.data.cross_entropy_loss(t_targets, d_logits)

        out_data = bten.TensorF(1, 1, self.is_cuda)
        out_data.copy_from_numpy(np.array([[loss_val]], dtype=np.float32))

        requires_grad = self.requires_grad and is_grad_enabled()
        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = {self} if requires_grad else set()

        def _backward():
            if not self.requires_grad:
                return
            if self.grad is None:
                self.grad = d_logits
            else:
                self.grad += d_logits

        out._backward = _backward if requires_grad else lambda: None
        return out


    @property
    def T(self):
        """
        Transpose of a 2D tensor. This function is only partially filled. 
        You need to add the backward() function.
        """
        out_data = self.data.T
        requires_grad = self.requires_grad and is_grad_enabled()
        out = AGTensor(out_data, requires_grad=requires_grad, is_cuda=self.is_cuda)
        out._prev = {self} if requires_grad else set()

        def _backward():
            if self.requires_grad:
                grad_in = out.grad.T
                if self.grad is None:
                    self.grad = grad_in
                else:
                    self.grad += grad_in
        
        out._backward = _backward if requires_grad else lambda: None
        return out


    def mean(self):
        """Mean of all elements, returning a scalar AGTensor(1x1 tensor)."""
        # Lab-2: add your code here
        # Hint: It can be implemented using AGTensor's sum and * operation and therefore no need for separate backward logic.

        total = self.sum()  # AGTensor
        h, w = self.shape
        return total * (1.0 / (h * w)) # divide by number of elements


    def argmax(self):
        """
        Per-row argmax, returns an AGTensor of shape (N,1).
        This function is non-differentiable.
        """
        out_data = self.data.argmax()
        out = AGTensor(out_data, requires_grad=False, is_cuda=self.is_cuda)
        return out

    def __eq__(self, other: 'Union[AGTensor, np.ndarray]'):
        """
        Elementwise equality (with broadcasting).
        Returns a AGTensor with the broadcasted output shape whose elements are 1 (True) or 0 (False) 
        at the location where the correspond elements in self and other are equal.
        other can be an AGTensor or a NumPy array.
        This function is non-differentiable.
        """
        if isinstance(other, AGTensor):
            out_data = self.data == other.data
        else:
            other_t = AGTensor.from_numpy(other, is_cuda=self.is_cuda)
            out_data = self.data == other_t.data
        out = AGTensor(out_data, requires_grad=False)
        return out

    # ----- backprop driver -----
    def backward(self, grad=None):
        """
        Backpropagate the gradient of the loss through the saved computation graph.
        Call this function on a scalar AGTensor (i.e., shape (1,1)).
        The optional grad argument is the initial gradient to be backpropagated.
        If grad is None, it defaults to a tensor of ones with the same shape as self.
        """
        # 1. Initialize gradient for the root tensor (usually scalar loss)
        if grad is None:
            h, w = self.data.shape  # unpack shape from TensorF
            grad = bten.TensorF(h, w, self.is_cuda)
            grad.fill(1.0)
        self.grad = grad

        # 2. Topological sort of all nodes in the computation graph
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        # 3. Traverse in reverse (from loss to leaves)
        for node in reversed(topo):
            node._backward()


    # Convenience: numpy view for debugging
    def numpy(self):
        return self.data.to_numpy()

    def __repr__(self):
        dev = "cuda" if self.is_cuda else "cpu"
        return f"Tensor(shape={self.data.shape}, device={dev}, op='{self._op}')"


# Chatgpt's explanation of what i have to do:
# Lab-1 provided CUDA kernels and a Tensor class (float, uint32 etc).
# Lab-2's bindings expose that as a Python-visible type — call it bten.TensorF
# AGTensor will call those bound functions (matmul, add, relu, sum, cross_entropy, etc.) for forward computation. Those functions return low-level tensors.
# Your job: add bookkeeping around these calls so you can compute gradients in Python later.
# You must wrap those low-level C++ tensor ops.
# You must record the computational graph (who depends on whom).
# You must propagate gradients backward once .backward() is called.

# What AGTensor actually tracks:
# The data (a real tensor)
# The gradient (another tensor of same shape)
# The operation that created it (_op)
# The inputs/children (_prev)
# The backward function that computes the gradient for parents.



# Ex:
# a = AGTensor(np.arange(3, dtype=np.float32))  # leaf tensor
# b = a + 3.0                                   # b depends on a
# b.backward()

# fORWARD: b = a + 3.0  
# Python calls your AGTensor.__add__ method.
# Inside it, you’ll do three things:
# - out_data = a.data.add(3.0) # does the computation on CPU/GPU
# - out = AGTensor(out_data, children=(a,), op='+') # reates a new tensor to wrap the resul but also keep track of a and the op done
# - Then caclulate the backward function for out

# Backward: b.backward()
# Initialize b.grad
# We need to visit all tensors that b depends on
# call backwards on each tensor in reverse topological order
# Example:
# For b = a + 3, a.grad = b.grad
# For c = a * b, a.grad += b.data * c.grad, etc.

# Why we store a new AGTensor every time
# When you do x = a + b, you get a new tensor x because:
# - You need to record how it was created (which parents and which op)
# - You need to be able to call _backward() later
# - If you didn’t make a new object, you’d lose that information and couldn’t build the computation graph.