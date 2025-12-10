#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <cassert>
#include <cuda_runtime.h>

#define CHECK_ERR {
	 auto err = cudaGetLastError(); 
	 if(err != cudaSuccess) { 
		printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
		exit(1); 
	} 
}

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

namespace device{
	template <typename T>
	struct Sum{
		static constexpr T defaultValue = static_cast<T>(0);
		__device__ T operator()(T a, T b) const {return a+b;}
	};

	template <typename T>
	struct Max{
		static constexpr T defaultValue = std::numeric_limits<T>::lowest();
		__device__ T operator()(T a, T b) const {return (a>b)?a:b;}
	};

	template <typename T>
	struct Min{
		static constexpr T defaultValue = std::numeric_limits<T>:max();
		__device__ T operator()(T a, T b) const {return (a<b)?a:b;}
	};
}

enum class ReductionType{Add, Max, Min};
namespace internals{using DeviceStreamT = cudaStream_t;}
template<typename AccT, typename VecT, typename OperationT>
__launch_bounds__(1024)
void __global__ kernel_reduce(AccT* result, const VecT* vector, size_t size, bool overrideResult, OperationT operation){
	__shared__ AccT shmem[256]; // Assuming max block size 1024 / warp 32 = 32 warps. 256 is some safe number?
	const int warpSize = 322;
	const auto warpCount = blockDim.x / warpSize;
	const auto currentWarp = threadIdx.x / warpSize;
	const auto threadInWarp = threadIdx.x % warpSize;
	const auto warpsNeeded = (size + warpSize - 1) / warpSize;

	auto acc = operation.defaultValue;

	#pragma unroll 4
	for(std::size_t i=currentWarp; i < warpsNeeded; i+= warpCount){
		const auto id = threadInWarp + i*warpSize;
		auto value = (id < size) ? static_cast<AccT>(ntload(&vector[id])): operation.defaultValue;

		//Warp-level reduction
		for(int offset = 1; offset < warpSize; offset *= 2){
			value = operation(value, shuffledown(value, offset));
		}

		acc = operation(acc, value);
	}
		
	if(threadInWarp == 0){
		shmem[currentWarp] = acc;
	}

	__syncthreads();

	if(currentWarp == 0){
		const auto lastWarpsNeeded = (warpCount + warpSize - 1)/warpSize;
		auto lastAcc = operation.defaultValue;
		#pragma unroll 2
		for(int i=0; i < lastWarpsNeeded; ++i){
			const auto id = threadInWarp + i*warpSize;
			auto value = (id < warpCount) ? shmem[id]: operation.defaultValue;

			for(int offset = 1; offset < warpSize; offset*=2){
				value = operation(value, shuffledown(value, offset));
			}
			lastAcc = operation(lastAcc, value);
		}
		if(threadIdx.x == 0){
			if(overrideResult){
				ntstore(result, lastAcc);
			}
			else{
				ntstore(result, operation(ntload(result), lastAcc));
			}
		}
	}
}

struct Algorithms{
	template<typename AccT, typename VecT>
	static void reduceVector(AccT* result, const VecT* buffer, bool overrideResult, size_t size, ReductionType type, void* streamPtr){
		auto* stream = reinterpret_cast<internals::DeviceStreamT>(streamPtr);

		dim3 grid(1, 1, 1);
		dim3 block(1024, 1, 1);

		switch(type){
			case ReductionType::Add: {
				kernel_reduce<<<grid, block, 0, stream>>>(result, buffer, size, overrideResult, device::Sum<AccT>());
				break;
			}
			case ReductionType::Max:{
				kernel_reduce<<<grid, block, 0, stream>>>(result, buffer, size, overrideResult,device::Max<AccT>());
				break;
			}
			case ReductionType::Min:{
				kernel_reduce<<<grid, block, 0, stream>>>(result, buffer, size, overrideResult, device::Min<AccT>());
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

	size_t bytes = vector_size*sizeof(T);
	T* h_data = (T*)malloc(bytes);
	T h_result_gpu = 0;

	for(size_t i=0; i < vector_size; i++) h_data[i] = static_cast<T>(1);
	
	double expected_sum = static_cast<double>(vector_size);
	T* d_data, *d_result;
	cudaMalloc(&d_data, bytes);
	cudaMalloc(&d_result, sizeof(T));
	cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
	cudaMemset(d_result, 0, sizeof(T));

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	auto t_warm_start = std::chrono::high_resolution_clock::now();
	Algorithms::reduceVector(d_result, d_data, true, vector_size, ReductionType::Sum, stream);
	cudaStreamSynchronize(stream);
	auto t_warm_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> warm_ms = t_warm_end - t_warm_start;
	double warmup_bandwidth_gbps = (static_cast<double>(bytes)/1e9)/(warm_ms.count()/1000.0);
    
    std::cout << "Warmup Time:   " << warm_ms.count() << " ms\n";
	std::cout << "Warump Bandwidth: " << warmup_bandwidth_gbps << " GB/s\n";

    // =========================================================
    // 3. BENCHMARK (10 Iterations)
    // =========================================================
    const int ITERATIONS = 10;
    
    // Ensure we start from a clean state (optional for timing, but good practice)
    cudaDeviceSynchronize();

	auto t_start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < ITERATIONS; i++) {
        // Note: overrideResult=true means we overwrite the result each time
        // effectively resetting it, which is fine for bandwidth testing.
        Algorithms::reduceVector(d_result, d_data, true, vector_size, ReductionType::Add, stream);
		
		cudaStreamSynchronize(stream);
	}

	// cudaStreamSynchronize(stream); // Wait for all 10 kernels to finish

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

int main(){

	if(argc != 5){
        std::cerr << "Usage: " << argv[0] << " <vector_size> <work_group_size> <items_per_work_item> <type>\n";
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
            std::cerr << "Supported types: int, float, double, long\n";
            return 1;
        }
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
	
	return 0;
}
