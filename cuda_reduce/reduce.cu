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
     if(err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Allow tuning parameters to be injected by the compiler script
#ifndef TUNING_WG_SIZE
#define TUNING_WG_SIZE 256
#endif

#ifndef TUNING_ITEMS
#define TUNING_ITEMS 8
#endif

// =================================================================================
// 1. FORWARD DECLARATIONS / TYPES (Must be at the top)
// =================================================================================
namespace device {
    template <typename T>
    struct Sum {
        static constexpr T defaultValue = static_cast<T>(0);
        __device__ T operator()(T a, T b) const {return a+b;}
    };

    template <typename T>
    struct Max {
        static constexpr T defaultValue = std::numeric_limits<T>::lowest();
        __device__ T operator()(T a, T b) const {return (a>b)?a:b;}
    };

    template <typename T>
    struct Min {
        static constexpr T defaultValue = std::numeric_limits<T>::max();
        __device__ T operator()(T a, T b) const {return (a<b)?a:b;}
    };
}

enum class ReductionType{Add, Max, Min};
namespace internals{using DeviceStreamT = cudaStream_t;}


// =================================================================================
// 2. DEVICE HELPERS
// =================================================================================
template<typename T>
__device__ __forceinline__ T ntload(const T* ptr){
    return __ldg(ptr);
}

template<typename T>
__device__ __forceinline__ void ntstore(T* ptr, T val){
    *ptr = val;
}

template<typename T>
__device__ __forceinline__ T shuffledown(T val, int offset){
    return __shfl_down_sync(0xFFFFFFFF, val, offset);
}

// Warp reduce operation
template<typename T, typename Op>
__device__ __forceinline__ T warpReduce(T val, Op operation, int warpSize = 32){
    for (int offset = warpSize / 2; offset > 0; offset /= 2){
        val = operation(val, shuffledown(val, offset));
    }
    return val;
}

// Generic Atomic Update (with CAS fallback)
template<typename T, typename Op>
__device__ void atomicUpdate(T* address, T val, Op operation){
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        T calculatedRes = operation(*(T*)&assumed, val);
        old = atomicCAS(address_as_ull, assumed, *(unsigned long long*)&calculatedRes);
    } while (assumed != old);
}

// Native Atomics
template <> __device__ void atomicUpdate<int, device::Sum<int>>(int* address, int val, device::Sum<int> op) { atomicAdd(address, val); }
template <> __device__ void atomicUpdate<float, device::Sum<float>>(float* address, float val, device::Sum<float> op) { atomicAdd(address, val); }
#if __CUDA_ARCH__ >= 600
template <> __device__ void atomicUpdate<double, device::Sum<double>>(double* address, double val, device::Sum<double> op) { atomicAdd(address, val); }
#endif

// Block Reduce
template<typename T, int BLOCK_SIZE, typename Op>
__device__ __forceinline__ T blockReduce(T val, T* shmem, Op operation, int warpSize = 32){
    const int laneId = threadIdx.x % warpSize;
    const int warpId = threadIdx.x / warpSize;

    val = warpReduce(val, operation, warpSize);

    if (laneId == 0) shmem[warpId] = val;
    __syncthreads();

    const int numWarps = BLOCK_SIZE / warpSize;
    val = (threadIdx.x < numWarps) ? shmem[laneId] : operation.defaultValue;

    if (warpId == 0) val = warpReduce(val, operation, warpSize);

    return val; 
}


// =================================================================================
// 3. KERNELS
// =================================================================================

// Init Kernel to handle overrideResult safely across multi-block
// FIXED TYPO: changed 'typname' to 'typename'
template<typename T, typename Op>
__global__ void initKernel(T* result, Op operation){
    if(threadIdx.x == 0){
        *result = operation.defaultValue;
    }
}

// The Optimized Kernel (Exact Signature Retained)
template<int WORK_GROUP_SIZE, int ITEMS_PER_WORK_ITEM, typename AccT, typename VecT, typename OperationT>
__launch_bounds__(WORK_GROUP_SIZE)
void __global__ kernel_reduce(AccT* result, const VecT* vector, size_t size, bool overrideResult, OperationT operation){
    
    __shared__ AccT shmem[32]; 
    const int warpSize = 32;

    AccT threadAcc = operation.defaultValue;
    size_t blockBaseIdx = blockIdx.x * (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM);
    size_t threadBaseIdx = blockBaseIdx + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < ITEMS_PER_WORK_ITEM; i++){
        size_t idx = threadBaseIdx + i * WORK_GROUP_SIZE;
        if(idx < size){
            threadAcc = operation(threadAcc, static_cast<AccT>(ntload(&vector[idx])));
        }
    }

    AccT blockAcc = blockReduce<AccT, WORK_GROUP_SIZE, OperationT>(threadAcc, shmem, operation, warpSize);

    if(threadIdx.x == 0){
        (void)overrideResult; // Handled by host now
        atomicUpdate(result, blockAcc, operation);
    }
}


// =================================================================================
// 4. HOST CODE / BENCHMARK
// =================================================================================

// Host Wrapper (Exact Signature Retained)
struct Algorithms{
    template<typename AccT, typename VecT>
    static void reduceVector(AccT* result, const VecT* buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr){
        auto* stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);

        // Uses macros injected by runner.sh
        constexpr int WORK_GROUP_SIZE = TUNING_WG_SIZE;
        constexpr int ITEMS_PER_WORK_ITEM = TUNING_ITEMS;

        size_t totalItems = WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM;
        size_t numBlocks = (size + totalItems - 1) / totalItems;

        // Safely Initialize for Multi-Block execution
        if (overrideResult) {
            switch(type) {
                case ReductionType::Add: initKernel<<<1, 1, 0, stream>>>(result, device::Sum<AccT>()); break;
                case ReductionType::Max: initKernel<<<1, 1, 0, stream>>>(result, device::Max<AccT>()); break;
                case ReductionType::Min: initKernel<<<1, 1, 0, stream>>>(result, device::Min<AccT>()); break;
                default: assert(false && "reduction type not implemented");
            }
        }

        switch(type){
            case ReductionType::Add: {
                kernel_reduce<WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM><<<numBlocks, WORK_GROUP_SIZE, 0, stream>>>(result, buffer, size, overrideResult, device::Sum<AccT>());
                break;
            }
            case ReductionType::Max:{
                kernel_reduce<WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM><<<numBlocks, WORK_GROUP_SIZE, 0, stream>>>(result, buffer, size, overrideResult, device::Max<AccT>());
                break;
            }
            case ReductionType::Min:{
                kernel_reduce<WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM><<<numBlocks, WORK_GROUP_SIZE, 0, stream>>>(result, buffer, size, overrideResult, device::Min<AccT>());
                break;
            }
            default:{
                assert(false && "reduction type is not yet implemented");
            }
        }
        CHECK_ERR;
    }
};

template <typename T>
void run_benchmark(size_t vector_size, std::string type_name){
    std::cout << "------------------------------------------------\n";
    std::cout << "Testing Type: " << type_name << " | Size: " << vector_size << " elements\n";
    std::cout << "Config: WG_SIZE=" << TUNING_WG_SIZE << " | ITEMS=" << TUNING_ITEMS << "\n";

    size_t bytes = vector_size*sizeof(T);
    T* h_data = (T*)malloc(bytes);
    T h_result_gpu = 0;

    for(size_t i=0; i < vector_size; i++) h_data[i] = static_cast<T>(1);
    
    double expected_sum = static_cast<double>(vector_size);
    T* d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(T));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t_warm_start = std::chrono::high_resolution_clock::now();
    Algorithms::reduceVector(d_result, d_data, true, vector_size, ReductionType::Add, stream);
    cudaStreamSynchronize(stream);
    auto t_warm_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> warm_ms = t_warm_end - t_warm_start;
    double warmup_bandwidth_gbps = (static_cast<double>(bytes)/1e9)/(warm_ms.count()/1000.0);
    
    std::cout << "Warmup Time:   " << warm_ms.count() << " ms\n";
    std::cout << "Warump Bandwidth: " << warmup_bandwidth_gbps << " GB/s\n";

    const int ITERATIONS = 10;
    cudaDeviceSynchronize();

    auto t_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERATIONS; i++) {
        Algorithms::reduceVector(d_result, d_data, true, vector_size, ReductionType::Add, stream);
    }
    cudaStreamSynchronize(stream); 
    auto t_end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&h_result_gpu, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    
    std::chrono::duration<double, std::milli> total_ms = t_end - t_start;
    double avg_ms = total_ms.count()/ITERATIONS;

    double total_bytes_processed = static_cast<double>(bytes)*ITERATIONS;
    double total_seconds = total_ms.count() / 1000.0;
    double bandwidth_gbps = (total_bytes_processed/1e9)/total_seconds;
    std::cout << "Average Time: " << avg_ms << " ms\n";
    std::cout << "Bandwidth:     " << bandwidth_gbps << " GB/s\n";

    double gpu_val = static_cast<double>(h_result_gpu);
    double diff = std::abs(gpu_val - expected_sum);
    bool passed = false;

    if constexpr(std::is_integral<T>::value) passed = (diff==0);
    else passed = (diff < 1e-2*expected_sum);

    if(passed) std::cout << "Result Status: [PASS] \n";
    else std::cout << "Result Status: [FAIL] (Got " << gpu_val<< ", Expected " << expected_sum << ")\n";

    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
}

int main(int argc, char* argv[]){

    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << " <vector_size> <type>\n";
        std::cerr << "Types: int, float, double, long\n";
        return 1;
    }

    size_t VECTOR_SIZE = std::atol(argv[1]);
    std::string type = argv[2];

    try {        
        if(type == "int") {
            run_benchmark<int>(VECTOR_SIZE, type);
        } else if(type == "float") {
            run_benchmark<float>(VECTOR_SIZE, type);
        } else if(type == "double") {
            run_benchmark<double>(VECTOR_SIZE, type);
        } else {
            std::cerr << "Unknown type: " << type << "\n";
            return 1;
        }
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}