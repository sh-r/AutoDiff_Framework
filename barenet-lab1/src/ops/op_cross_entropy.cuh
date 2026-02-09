#pragma once
#include "utils/tensor.cuh"

// Kernel: one thread per sample (row)
template <typename T, typename S>
__global__ void cross_entropy_kernel(const Tensor<T> logits, const Tensor<S> targets,
                                     Tensor<T> d_logits, T* loss_out, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // sample index
    if (i >= logits.h) return;

    int num_classes = logits.w;

    // --- 1) Numerically stable log-sum-exp ---
    T max_logit = Index(logits, i, 0);
    for (int j = 1; j < num_classes; ++j) {
        T v = Index(logits, i, j);
        if (v > max_logit) max_logit = v;
    }

    T sum_exp = (T)0;
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += exp(Index(logits, i, j) - max_logit);
    }
    T log_denom = max_logit + log(sum_exp); // log(sum_j exp(logit_j))

    // --- 2) loss for this sample (scalar) ---
    int target = (int)Index(targets, i, 0); // targets is N x 1
    T loss_i = -( Index(logits, i, target) - log_denom );

    // accumulate total loss (sum over batch) safely
    atomicAdd(loss_out, loss_i);

    // --- 3) fill gradient d_logits (averaged over batch) ---
    // d_logits[i,j] = softmax_j - 1_{j==target}
    // we divide by batch_size so tests that expect average grad pass
    for (int j = 0; j < num_classes; ++j) {
        T p = exp(Index(logits, i, j) - max_logit) / sum_exp; // softmax probability
        T g = p - (j == target ? (T)1 : (T)0);
        Index(d_logits, i, j) = g / (T)batch_size; // **important: divide by batch_size**
    }
}

//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.

//Lab-1: please add your code here
//You need to define separate GPU kernel function(s) and launch them here
//In order to calculate d_logits, you should derive what its values should be 
//symbolically.

template <typename T, typename S>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<S> &targets,
                               Tensor<T> &d_logits)
{
    if (logits.h != d_logits.h || logits.w != d_logits.w)
        throw std::runtime_error("op_cross_entropy_loss: d_logits shape mismatch");

    if (targets.h != logits.h || targets.w != 1)
        throw std::runtime_error("op_cross_entropy_loss: targets shape mismatch");

    if (logits.on_device != d_logits.on_device || logits.on_device != targets.on_device)
        throw std::runtime_error("op_cross_entropy_loss: device mismatch");

    int batch_size = logits.h;
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    // temporary device scalar to accumulate loss (sum over samples)
    T *d_loss = nullptr;
    CUDA_OK(cudaMalloc(&d_loss, sizeof(T)));
    CUDA_OK(cudaMemset(d_loss, 0, sizeof(T)));

    // Launch kernel
    cross_entropy_kernel<<<blocks, threads>>>(logits, targets, d_logits, d_loss, batch_size);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // Copy summed loss back to host and free temporary buffer
    T h_loss = (T)0;
    CUDA_OK(cudaMemcpy(&h_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_loss));

    // return average loss
    return h_loss / (T)batch_size;
}

// notes:
// input = logits (raw scores from your network, size = batch × num_classes), target (one int per row telling which is correct)
// output = The average loss (a single number), The gradient wrt logits (d_logits)
// steps:
// 1. get softmax e^j/sigma (e^/k)
// 2. get cross entropy=  -log(softmax of correct class)
// 3. gradient = softmax - 1 for correct class and -0 for rest do the grad calculation yourself it makes sense

// Suppose 100 threads all try to do loss_sum += loss_i into one global variable.
// If they all just write at the same time, their results will collide, and you’ll lose some contributions (race condition).
// so use atomic add