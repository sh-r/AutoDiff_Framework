#pragma once

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"

#define TILE_DIM 16

template <typename AT, typename BT, typename OT>
static void ensure_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h)
        throw std::runtime_error("a,b,out tensor shape mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());

    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch a:" + 
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
}

// why naive and tiled kernels separate? because different logic
// in both each thread computes one element of C
// but in naive, each A elem or B elem is read multiple times by different threads
// since we are accessing global memory which is slow, we want to reduce that
template <typename T>
__global__ void naive_matmul_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.h && col < C.w) {
        T sum = 0;
        for (int k = 0; k < A.w; ++k) { 
            sum += Index(A, row, k) * Index(B, k, col); // gets elem in row,k of A and k,col of B and A.w is same as B.h
        }
        Index(C, row, col) = sum;
    }
}

// in naive, each A elem or B elem is read multiple times by different threads
// in tiled, we load a tile of A and B into shared memory which is faster to access
// so each elem of A and B is read from global memory only once per tile,
// then multiple threads can access it from shared memory
// shared memory is limited, so we use small tiles

// works only TILE_DIM and BLOCK_DIM are same NOTE BIG NOTE
template <typename T>
__global__ void tiled_matmul_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
    __shared__ T tileA[TILE_DIM][TILE_DIM];
    __shared__ T tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // same as elem wise file  refer eof example
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    T value = 0; // final C[row, col]
    int numTiles = (A.w + TILE_DIM - 1) / TILE_DIM; // same as elemwise file grid calc

    for (int t = 0; t < numTiles; t++) {

        int tiledColA = t * TILE_DIM + threadIdx.x; // goes tile by tile
        int tiledRowB = t * TILE_DIM + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < A.h && tiledColA < A.w) ? Index(A, row, tiledColA) : 0; // fill excess with 0
        tileB[threadIdx.y][threadIdx.x] = (tiledRowB < B.h && col < B.w) ? Index(B, tiledRowB, col) : 0;
        __syncthreads(); // make sure all threads finished loading

        for (int k = 0; k < TILE_DIM; k++) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads(); 
    }

    if (row < C.h && col < C.w) { // bounds check for safety
        Index(C, row, col) = value;
    }
}

//compute C = A@B
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    ensure_mm_shape_device(A, B, C);
    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid((C.w + block.x - 1)/block.x, (C.h + block.y - 1)/block.y); // same as elemwise file grid calc
    // naive_matmul_kernel<T><<<grid, block>>>(A, B, C);
    tiled_matmul_kernel<T><<<grid, block>>>(A, B, C);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
}

// Example:
// Suppose TILE_DIM = 16 → block has 16×16 threads.
// blockIdx.x = 1, threadIdx.x = 5 → column = 1*16 + 5 = 21
// blockIdx.y = 0, threadIdx.y = 3 → row = 0*16 + 3 = 3
// This thread computes C[3, 21].

// Example:
// this is what int row, col line is doing
// C = 4x4, TILE_DIM = 2
// gridDim.x = gridDim.y = 2 (2x2 blocks)
// block(0,0): threads handle C[0,0], C[0,1], C[1,0], C[1,1]
// block(0,1): threads handle C[0,2], C[0,3], C[1,2], C[1,3]
// block(1,0): threads handle C[2,0], C[2,1], C[3,0], C[3,1]
// block(1,1): threads handle C[2,2], C[2,3], C[3,2], C[3,3]

// Example of matrix read for line 61 62
// TileA = |1 2|
//         |5 6|

// TileB = |1 2|
//         |5 6|
// saved in TileA Tileb like that

