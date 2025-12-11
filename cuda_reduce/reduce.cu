#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <cassert>
#include <cuda_runtime.h>

#define CHECK_ERR do { \
    auto err = cudaGetLastError(); \
    if(err != cudaSuccess){ \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

namespace device{
    template <typename T>
    struct Sum {
        static constexpr T defaultValue = static_cast<T>(0);
        __device__ T operator()(T a, T b) const { return a+b;}
    };

    template <typename T>
    struct Max {
        static constexpr T defaultValue = std::numeric_limits<T>::lowest();
        __device__ T operator()(T a, T b) const {return (a>b)? a: b;}
    };

    template <typename T>
    struct Min{
        static constexpr T defaultValue = std::numeric_limits<T>::max();
        __device__ T operator()(T a, T b) const {return (a<b)? a: b;}
    };
}

enum class ReductionType { Add, Max, Min};

template<typename T, typename Op>
__device__ __forceinline__ T warpReduce(T val, Op operation, int warpSize){
    // loop based on the warpSize
    // stardard NVIDIA is 32, AMD is 64, but should support other sizes
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        val = operation(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    return val;
}

template<typename T, int BLOCK_SIZE, typename Op>
__device__ __forceinline__ T blockReduce(T val, T* sharedMem, Op operation, int warpSize){
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // reduce within warp
    val = warp_reduce(val, operation, warpSize);

    // warp leaders write to shared mem
    if(lane == 0){
        sharedMem[wid] = val;
    }
    __syncthreads();

    int numWarps = BLOCK_SIZE / warpSize;

    val = (threadIdx.x < numWarps) ? sharedMem[lane] : operation.defaultValue;

    if(wid == 0){
        val = warpReduce(val, operation, warpSize);
    }
    return val;
}

//AtomicAdd is available, others need CAS
template <typename T, typename Op>
__device__ void atomicUpdate(T* address, T val, Op operation){
    // copied from : https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    unsigned long long* address_as_ull = (unsigned long long*) address;
    unsigned long long old = *address_as_ull, assumed;
}