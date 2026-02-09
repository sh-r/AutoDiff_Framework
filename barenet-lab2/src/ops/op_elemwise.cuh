#pragma once

#include "utils/tensor.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <sstream>
#include <stdexcept>

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads

extern unsigned long long randgen_seed;

class RandGenGPU
{
public:
    RandGenGPU(unsigned long long s)
    {
        CURAND_OK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_OK(curandSetPseudoRandomGeneratorSeed(gen, s));
    }
    curandGenerator_t gen;
};

////////////////////////////////////////////////////
// Group 1: Functors
////////////////////////////////////////////////////

// Note: Typename indicates that it can be int/float/tensor any datatype
//This functor adds two input elements "a" and "b" together
template <typename T>
class AddFunc
{
public:
    __host__ __device__ T operator()(T a, T b)
    { 
      return a + b;
    }
};

//This functor adds constant "b" to the input element
template <typename T>
class AddConstFunc
{
public:
    __host__ __device__ T operator()(T a)
    {
      return a + b;
    }
    const T b;
};

//This functor substracts b from a, aka a - b
template <typename T>
class SubFunc
{
public:
    __host__ __device__ T operator()(T a, T b)
    {
        return a - b;
    }
};


//This functor multiplies two input elements x and a 
template <typename T>
class MultiplyFunc
{
public:
    __host__ __device__ T operator()(T x, T a)
    {
        return x * a;
    }
};

//This functor multiplies constant "b" to the input element
template <typename T>
class MultiplyConstFunc
{
public:
    __host__ __device__ T operator()(T x)
    {

        return x * b;
    }
    const T b;
};

// Note for below function:
//AT → type of the first input tensor
//BT → type of the second input tensor
//OutT → type of the output tensor

//This functor returns 1 if inputs "a" and "b" are equal
//and returns 0 otherwise. 
template <typename AT, typename BT, typename OutT>
class EqualityFunc
{
public:
    __host__ __device__ OutT operator()(AT a, BT b)
    {
        return (a == b) ? (OutT)1 : (OutT)0;
    }
};

//This functor returns the ReLu value of x
template <typename T>
class ReluFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        return (x >= 0) ? x : (T)0; // return x if greateer than 0 else return 0
    }
};

//This functor implements the backwards of the 
//ReLu operation for a single element.
template <typename T>
class ReluBackFunc
{
public:
    __host__ __device__ T operator()(T x, T dy)
    {
        return (x > 0) ? dy : (T)0; // if x >0, return dy else return 0
    }
};


////////////////////////////////////////////////////
// Group 2: GPU Kernels
////////////////////////////////////////////////////


//This is the GPU kernel function for performing element wise operation 
//It invokes the given functor f on all elements of "t"
//and stores the result in "out" tensor.

// op_elemwise_unary_kernel<ConstFillFunc, T><<<grid, block>>>(f, t, t); called this way basically in launcher
// so f is the functor, t is input tensor, out is output tensor
// kernel always does f(t, row, col) 
template <typename OpFunc, typename T>
__global__ void op_elemwise_unary_kernel(OpFunc f, Tensor<T> in, Tensor<T> out)
{
    //blockDim, threadIdx are CUDA buikt in vars
    // threadIdx is thread pos inside block
    // blockIdx is block pos inside grid
    // blockDim size of block in threads
    // gridDim size of grid in blocks

    // Suppose:
    // Each block = 2 rows × 3 cols (so blockDim.y = 2, blockDim.x = 3).
    // Grid = 2 blocks tall × 2 blocks wide.
    // So total threads = 4 blocks × (2×3) = 24 threads.
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // getting the absolute row in tensor
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Badically Index(t, row, col) = t.rawp[offset + row * stride_h + col * stride_w];
    // this means we can akways calculate correct mem location
    // t.rawp -> actual pointer to data
    // offset, stride_h, stride_w -> what they discussed in class, that matrix representation converted to linear for im memory
    // Refer example at end of this file
    if (row < in.h && col < in.w) {
        // Instead of always doing f(Index(in, row, col)),
        // pass a dummy value if in == out (like const fill).
        // Otherwise, pass the real input.
        T val;
        if (in.rawp == out.rawp) {
            val = T{};   // dummy (e.g. 0.0), ConstFillFunc ignores it and just takes x anyways ignores T
        } else {
            val = Index(in, row, col);
        }
        // Index macro handles offset/stride, works for slices/transposes too
        Index(out, row, col) = f(val);
    }
}

//This is the GPU kernel function for performing element wise operation with 
//two input arguments "in1" and "in2" with potential broadcasting. 
// Input tensor "in2" is always the one to be 
// broadcasted when broadcasting is necessary.  Broadcasting is needed if 
// "in2" only have one dimension (instead of both dimensions) in common with "in1" 
// and its other dimension has size 1. In this case, to perform elemwise operation,
// we essentially broadcast the values of "in2" along the dimension with size 1 
// to match the dimension size of "in1". 
// Example1: a = [[1, 2, 3], [4, 5, 6]] and b = [[10],[20]], 
// then a+b = [[11, 12, 13], [24, 25, 26]]
// Example2: a = [[1, 2, 3], [4, 5, 6]] and b = [[10,20,30]]
// then a+b = [[11,22,33], [14, 25, 36]] 
template <typename OpFunc, typename AT, typename BT, typename OutT>
__global__ void op_elemwise_binary_w_bcast_kernel(OpFunc f, Tensor<AT> in1, Tensor<BT> in2, Tensor<OutT> out)
{
    //Lab-1: add your code here
    // this is basically for output thread by the way
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // same as above func
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < in1.h && col < in1.w) { // again same check as above bu can't directly apply the function like above as 2 inputs
        AT a = Index(in1, row, col); // load from a and b
        BT b;
        if (in2.h == 1 && in2.w == 1) {  
            // scalar broadcast
            b = Index(in2, 0, 0);
        }
        else if (in2.h == 1) {  
            // broadcast row-wise
            b = Index(in2, 0, col);
        }
        else if (in2.w == 1) {  
            // broadcast col-wise
            b = Index(in2, row, 0);
        }
        else {  
            // same shape
            b = Index(in2, row, col);
        }
        Index(out, row, col) = f(a, b);
    }
}

////////////////////////////////////////////////////
// Group 3: Launchers
////////////////////////////////////////////////////

//This helper function launches the GPU kernel to 
//perform unary element wise operation configurable by the functor f. 
template <typename OpFunc, typename T>
void op_elemwise_unary_gpu(OpFunc f, const Tensor<T> &t, Tensor<T> &out)
{
  //Lab-1:add your code here. Somewhere in this function,
  //you need to call op_elemwise_unary_kernel<<<???, ???>>>(f, t, out);
    const dim3 block(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM); // dim3 means int type CUDA uses to represent blocks/grids its 2d only in this case
    // refer below eof for exanple on grid calculation
    const dim3 grid((t.w + block.x - 1) / block.x, (t.h + block.y - 1) / block.y); // basically calculates int grid size something like how we did in gpu assignment
    op_elemwise_unary_kernel<OpFunc, T><<<grid, block>>>(f, t, out);
    CUDA_OK(cudaGetLastError()); // basically for alerts
    CUDA_OK(cudaDeviceSynchronize()); // would be async without this and  and host would immediately move onto next task when GPU still running
}

//This helper function launches the GPU kernel that performs elementwise operation 
//(with potential broadcast) with two input tensor arguments "in1" and "in2",
// and stores the result in "out".  
template <typename OpFunc, typename AT, typename BT, typename OutT>
void op_elemwise_binary_w_bcast_gpu(OpFunc f, const Tensor<AT> &in1, const Tensor<BT> &in2, Tensor<OutT> &out)
{
  //Lab-1: add your code here. Somewhere in this function
  //you need to call op_elemwise_binary_w_bcast_kernel<<<???, ???>>>(f, in1, in2, out);
    const dim3 block(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM); // same as above
    const dim3 grid((in1.w + block.x - 1) / block.x, (in1.h + block.y - 1) / block.y); // same as above
    op_elemwise_binary_w_bcast_kernel<<<grid, block>>>(f, in1, in2, out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
}


////////////////////////////////////////////////////
// Group 4: HELPERS
// From here onwards all are helper functions I didn't make any chnage other than rearranging functions
// It's useful to look at them to see how different its done in GPU
///////////////////////////////////////////////////


//This helper functions performs unary element wise operation on CPU
//It naively double loops through all elements of "t"
template <typename OpFunc, typename T>
void op_elemwise_unary_cpu(OpFunc f, const Tensor<T> &t, Tensor<T> &out)
{
    for (int i = 0; i < t.h; i++)
    {
        for (int j = 0; j < t.w; j++)
        {
            Index(out, i, j) = f(Index(t, i, j));
        }
    }
}

//This helper functions performs element wise operation on CPU with broadcasting.
template <typename OpFunc, typename AT, typename BT, typename OutT>
void op_elemwise_binary_w_bcast_cpu(OpFunc f, const Tensor<AT> &in1, const Tensor<BT> &in2, Tensor<OutT> &out)
{
    for (int i = 0; i < in1.h; i++)
    {
        for (int j = 0; j < in1.w; j++)
        {
            AT a = Index(in1, i, j);
            BT b;
            if (in2.h == 1)
            {
                b = Index(in2, 0, j);
            }
            else if (in2.w == 1)
            {
                b = Index(in2, i, 0);
            }
            else
            {
                b = Index(in2, i, j);
            }
            Index(out, i, j) = f(a, b);
        }
    }
}

//This functor returns a constant value
// it is used to fill a tensor with constant values
template <typename T>
class ConstFillFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        return val;
    }
    const T val;
};

//This functor returns a random value that is uniformly distributed
//between min and max
template <typename T>
class UniformFillFuncCPU
{
public:
    UniformFillFuncCPU(T min, T max) : dist(min, max) {}
    T operator()(T x)
    {
        static std::default_random_engine gen(randgen_seed);
        return dist(gen);
    }
    std::uniform_real_distribution<T> dist;
};

//Helper function to test for broadcasting compatibility
template <typename AT, typename BT, typename OT>
static void ensure_bcast_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{

    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch");

    if (out.h != a.h || out.w != a.w) 
        throw std::runtime_error("out, a tensor shape mismatch: " + out.repr() + " vs " + a.repr());

    if (a.h == b.h && a.w == b.w) {
    } else if (a.h == b.h && b.w == 1) {
    }else if (a.w == b.w && b.h == 1) {
    } else {
        throw std::runtime_error("a,b shape mismatch: " + a.repr() + " vs " + b.repr());
    }

}

//Helper function to test for same shape
template <typename AT, typename BT, typename OT>
static void ensure_same_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (out.h != a.h || out.w != a.w)
        throw std::runtime_error("out, a tensor shape mismatch: " + out.repr() + " vs " + a.repr());

    if (a.h != b.h || a.w != b.w)
        throw std::runtime_error("a,b input tensor shape mismatch: " + a.repr() + " vs " + b.repr());

    if (a.on_device != b.on_device) {
        throw std::runtime_error("a,b tensor device mismatch");
    }
    if (a.on_device != out.on_device) {
        throw std::runtime_error("a,out tensor device mismatch");
    }
}


//This operator implements ReLu and stores the result in "out".
//Suppose y = Relu(x) Then y = x if x >=0.  y= 0 if x < 0.
template <typename T>
static void op_relu(const Tensor<T> &t, Tensor<T> &out)
{
    ensure_same_shape_device(t, t, out);
    ReluFunc<T> f;
    if (t.on_device)
    {
        op_elemwise_unary_gpu(f, t, out);
    }
    else 
    {
        op_elemwise_unary_cpu(f, t, out);
    }
}

//This operator is the "backward" function of ReLu. Let out = ReLu(in).
//Let "d_out" represents the gradient of "out". Calculate the gradient 
//of "in" using the chain rule and store the result in "d_in".
template <typename T>
void op_relu_back(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
{
    ensure_same_shape_device(in, d_out, d_in);
    ReluBackFunc<T> f;
    if (d_in.on_device)
    {
        op_elemwise_binary_w_bcast_gpu(f, in, d_out, d_in);
    }
    else 
    {
        op_elemwise_binary_w_bcast_cpu(f, in, d_out, d_in);
    }
}


//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    ensure_bcast_shape_device(a, b, out);

    AddFunc<T> f;
    if (a.on_device)
    {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    }
    else 
    {
        op_elemwise_binary_w_bcast_cpu(f, a, b, out);
    }
  
}



//This operator performs element-wise addition of "a" and constant b
//stores the result in tensor "out"
template <typename T>   
void op_add(const Tensor<T> &a, T c, Tensor<T> &out)
{
    ensure_same_shape_device(a, a, out);
    
    AddConstFunc<T> f{c};
    if (a.on_device)
    {
        op_elemwise_unary_gpu(f, a, out);
    }
    else 
    {
        op_elemwise_unary_cpu(f, a, out);
    }

}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_sub(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    ensure_bcast_shape_device(a, b, out);

    SubFunc<T> f;
    if (a.on_device)
    {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    }
    else 
    {
        op_elemwise_binary_w_bcast_cpu(f, a, b, out);
    }
  
}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_multiply(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    ensure_bcast_shape_device(a, b, out);
    MultiplyFunc<T> f;
    if (a.on_device)
    {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    }
    else 
    {
        op_elemwise_binary_w_bcast_cpu(f, a, b, out);
    }
}

//This operator performs element-wise multiplication of "a" and constant b
//stores the result in tensor "out"
template <typename T>
void op_multiply(const Tensor<T> &a, T c, Tensor<T> &out)
{
    ensure_same_shape_device(a, a, out);
    MultiplyConstFunc<T> f{c};
    if (a.on_device)
    {
        op_elemwise_unary_gpu(f, a, out);
    }
    else 
    {
        op_elemwise_unary_cpu(f, a, out);
    }
}

//This operator checks if tensor "a" and "b" are the same
//and stores in the "out" tensor value 0 at places where "a" and "b" are not equal 
//and 1 at places where "a" and "b" are equal.
template <typename AT, typename BT, typename OutT>
void op_equal(const Tensor<AT> &a, const Tensor<BT> &b, Tensor<OutT> &out)
{
    ensure_same_shape_device(a, b, out);
    EqualityFunc<AT, BT, OutT> f;
    if (a.on_device)
    {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    }
    else 
    {
        op_elemwise_binary_w_bcast_cpu(f, a, b, out);
    }
}

//This operator initializes tensor with constant values
template <typename T>
void op_const_fill(Tensor<T> &t, T value)
{
    ConstFillFunc<T> f{value};
    if (t.on_device)
    {
        op_elemwise_unary_gpu(f, t, t);
    }
    else
    {
        op_elemwise_unary_cpu(f, t, t);
    }
}

//This operator initializes tensor with random values
//that are uniformly distributed between min and max
template <typename T>
void op_uniform_fill(Tensor<T> &t, T min = 0, T max = 1)
{
    if (t.on_device)
    {
        static RandGenGPU g(randgen_seed);
        //curandGenerateUniform generates elements in the range [0,1)
        CURAND_OK(curandGenerateUniform(g.gen, t.rawp, t.h * t.w));
        //scale the shift the elements to be in the range [min, max)
        //op_add(t, (T)0.1, t);
        op_add<T>(t, min/(max-min), t);
        op_multiply(t, max-min, t);
    }
    else
    {
        UniformFillFuncCPU<T> f{min, max};
        op_elemwise_unary_cpu(f, t, t);
       // std::cout << "op_uniform_init t=" << t.str() << std::endl;
    }
}

// 1. Example to understand offset/stride

// Say we have a 3×4 tensor:
// Matrix (h=3, w=4):
// [ a  b  c  d ]
// [ e  f  g  h ]
// [ i  j  k  l ]

// Stored in memory (linear ribbon):
// rawp → [ a  b  c  d  e  f  g  h  i  j  k  l ]

// Here:
// offset = 0
// stride_h = 4 (because 1 row = 4 steps in memory)
// stride_w = 1 (because 1 column = 1 step in memory)

// Formula:
// Index(t,row,col) = rawp[offset + row*stride_h + col*stride_w]

// Example:
// (2,1) → offset + 24 + 11 = 9 → j

// 2. Example to get grid size
// Suppose t.w = 200 columns, t.h = 100 rows, and block = (32,32).
// grid.x = (200 + 32 - 1) / 32 = 231 / 32 = 7 (integer division) → 7 blocks across columns THE +32-1 IS THE TRICK TO GET 200/32 =6.25 TO 7 instead of doing +1 every time
// grid.y = (100 + 32 - 1) / 32 = 131 / 32 = 4 → 4 blocks down rows
// So grid = (7, 4) → 28 blocks total.
// Those 28 blocks (each 32×32 threads) cover
// 7x32 = 224 columns and 4x32 = 128 rows — that’s bigger than 200×100
// extra threads will just do nothing (we check bounds inside kernel)