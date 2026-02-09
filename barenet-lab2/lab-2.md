# Lab-2: Implementing Multilayer Perceptron (MLP)

The goal of this Lab-2 is build a shiny Python frontend with auto-differentiation to the bare minimum ML framework of Lab 1.  
And you'll also write a simple MLP model and train it with MNIST dataset using your framework. 

## Preliminary: Obtaining the lab code
Firstly, click on the lab-2 assignment link given in the Campuswire. Then clone your lab2 repository:
```
$ git clone git@github.com:nyu-mlsys-fa25/barenet-lab2-<YourGithubUsername>.git barenet-lab2
```

You need to copy all four files (`op_mm.cuh`, `op_elemwise.cuh`, `op_reduction.cuh` and `op_cross_entropy_loss`) in the `ops/` subdirectory to your new Lab2 respository.  Suppose your Lab-1's repository is `barenet-lab1` and your Lab-2's repository is `barenet-lab2`. Then do the following:
```
$ cd barenet-lab2
$ cp ../barenet-lab1/src/ops/* src/ops/
```

## Lab-2's existing C++ to Python binding code

Lab-2 uses [pybind11](https://github.com/pybind/pybind11) to create Python bindings of Lab-1 existing C++ code. In Lab-2's new configuration file (`CMakeLists.txt`), we've instructed `cmake` to automatically download the pybind11 library from github.  

We have provided two C++ files (`src/bindings.cu` and `src/py_tensor_shim.hh`) that use pybind11 to expose Lab-1's Tensor and the various operators that operate on it.  

In the `py_tensor_shim.hh` file, we have wrapped Lab-1's Tensor into a new class called PyTensor (which contains the corresponding Tensor as a member field).  PyTensor wraps various Tensor operators as its member functions, including `fill`, `transpose`, `add`, `sub`, `mul`, `relu`, `relu_back`, `matmul`, `sum`, `argmax`, `eq` etc.
 We expose PyTensor and its member functions through the `bind_tensor_type` function.   

In the `src/bindings.cu` file, we call the `src/bind_tensor_type` function to bind PyTensor (and its member functions) to Python. Note that although Lab-1 uses completely generic Tensor element types, we have only bound the float and uint32_t element types in `bindings.cu` for simplicity.  

## Implement AGTensor to track dataflow graph for auto-differentation

You job in Lab-2 is to implement auto-differentation.  You'll do this by keeping track of the dataflow graph of operations invoked by the user and then call each operator's backward operation in reverse to back propagate the gradient. Your implementation will be done in Python by completing the skeleton file `mygrad/engine.py`.  

In file `mygrad/engine.py`, we define the AGTensor class which is the autograd wrapper for bten.TensorF (or bten.TensorUI32). 
Below is a very simple example showing how users use our AGTensor library.  Our user interface is very similar to that of Pytorch's.


```python
a_np = np.arange(1, 4, dtype=np.float32).reshape(1, 3)
a = AGTensor(a_np, is_cuda=True) #create AGTensor by passing it a numpy array
b = a + 3.0
b.backward() # Perform backward propagation to calculate gradients
print(a.grad.to_numpy()) # print out the gradients of a.
```

More concretely, your job is to complete each member function defined for the AGTensor class. These AGTensor member funtions all have a similar structure.  It computes the result and saves it in a new AGTensor `out` to be returned. (Question: Why does it need to save the result?) It also defines and saves a closure function `backward` which captures the backward logic to calculate gradient for (each of) this member function's inputs.  You also need to complete AGTensor's member function `backward` in which you traverse the saved dataflow directed acyclic graph (DAG) and invoke each node's `backward` function in the reverse order of that node's creation.

## Testing your code

First, you need to build the C++ binding code. To do so, follow the following steps:

```
cd build
cmake ..
make
```

Afterwards, you can test your Python library by doing the following:
```
cd ~/barelab-lab2/
pytest -v test_ag_tensor.py
```

If all goes right, you'll see output like this:
```   
test_ag_tensor.py::test_sum PASSED           [  7%]
test_ag_tensor.py::test_transpose PASSED     [ 15%]
test_ag_tensor.py::test_mul PASSED           [ 23%]
test_ag_tensor.py::test_accum PASSED         [ 30%]
test_ag_tensor.py::test_mean PASSED          [ 38%]
test_ag_tensor.py::test_add PASSED           [ 46%]
test_ag_tensor.py::test_sub PASSED           [ 53%]
test_ag_tensor.py::test_matmul PASSED        [ 61%]
test_ag_tensor.py::test_relu PASSED          [ 69%]
test_ag_tensor.py::test_cross_entropy PASSED [ 76%]
test_ag_tensor.py::test_argmax PASSED        [ 84%]
test_ag_tensor.py::test_eq PASSED            [ 92%]
test_ag_tensor.py::test_2layer_mlp PASSED    [100%]

============= 13 passed in 1.83s ==============================
```

When you are debugging, you might want to run a simple test case only. Let's say you want to run `test_mul` only. You can do this by typing:
```
pytest -v test_ag_tensor.py::test_mul
```

## Training a simple 2 layer MLP on MNIST

You can also test our AGTensor library by training a small MLP on the [MNIST hardwritten digit dataset](https://huggingface.co/datasets/ylecun/mnist). We have already written the `train_mnist.py` file for you. You just run it like follows:
```
$ python train_mnist.py
```

An example output from a working lab is shown below:
```
Epoch 1: train loss = 0.7655 (latency 16.58s)
  Test acc: 0.9022
Epoch 2: train loss = 0.3108 (latency 16.29s)
  Test acc: 0.9245
Epoch 3: train loss = 0.2559 (latency 16.34s)
  Test acc: 0.9348

...

Epoch 18: train loss = 0.0677 (latency 15.98s)
  Test acc: 0.9732
Epoch 19: train loss = 0.0642 (latency 16.50s)
  Test acc: 0.9746
Epoch 20: train loss = 0.0610 (latency 16.00s)
  Test acc: 0.9748
```

For your reference, a C++ implementation of the same 2-layer MLP model using the same operator library runs 5X faster. So our AGTenor library is incurring a lot of overhead. This is likely due to the Python overhead as well as the framework doing lots of memory allocation/deallocation.

## Hand-in procedure

As in Lab-1, you should save your progress frequently by doing `git commit` followed by `git push origin master`.

To hand in your lab, first commit all of your modifications by following the instructions in the section on [Saving your progress](#Saving-your-progress). Second, make a tag to mark the latest commit point as your submittion for Lab1. Do so by typing the following:
```
$ git tag -a lab2 -m "submit lab2"
```

Finally, push your commit and your tag to Github by typing the following
```
$ git push --tags origin master
```


