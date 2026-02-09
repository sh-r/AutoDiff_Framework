# AutoDiff_Framework: Auto-differentiation and training with MNIST dataset.

- Operators: `op_mm.cuh`, `op_elemwise.cuh`, `op_reduction.cuh` and `op_cross_entropy_loss`

- Uses [pybind11](https://github.com/pybind/pybind11) to create Python bindings for C++ and CUDA code. Configuration file (`CMakeLists.txt`) automatically downloads the pybind11 library from github.  

- C++ files (`src/bindings.cu` and `src/py_tensor_shim.hh`) use pybind11 to expose operators.
  
- Keep track of the dataflow graph of operations invoked by the user and then call each operator's backward operation in reverse to back propagate the gradient.
Similar to Pytorch:
```python
a_np = np.arange(1, 4, dtype=np.float32).reshape(1, 3)
a = AGTensor(a_np, is_cuda=True) #create AGTensor by passing it a numpy array
b = a + 3.0
b.backward() # Perform backward propagation to calculate gradients
print(a.grad.to_numpy()) # print out the gradients of a.
```

- Trained a simple 2 layer MLP on MNIST with auto-diff framework

