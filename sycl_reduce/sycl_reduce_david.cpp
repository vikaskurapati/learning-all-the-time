#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>

using namespace sycl;

enum class ReductionType {
    SUM,
    PRODUCT,
    MIN,
    MAX
};

template<ReductionType Type, typename T>
constexpr T neutral() {
    if constexpr (Type == ReductionType::SUM) return T(0);
    else if constexpr (Type == ReductionType::PRODUCT) return T(1);
    else if constexpr (Type == ReductionType::MIN) return std::numeric_limits<T>::max();
    else if constexpr (Type == ReductionType::MAX) return std::numeric_limits<T>::lowest();
}

template<typename T>
inline T ntload(const T* ptr) {
    return *ptr;
}

template<typename T>
inline void ntstore(T* ptr, T value) {
    *ptr = value;
}

template <ReductionType Type, typename AccT, typename VecT, typename OpT> 
void launchReduction(AccT* result, const VecT *buffer, size_t size, OpT operation, bool overrideResult, void* streamPtr) {
    constexpr auto DefaultValue = neutral<Type, AccT>();
    ((sycl::queue *) streamPtr)->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<AccT, 1> shmem(256, cgh);
      cgh.parallel_for(sycl::nd_range<1> { 1024, 1024 },
        [=](sycl::nd_item<1> idx) {
          const auto subgroup = idx.get_sub_group();
          const auto sgSize = subgroup.get_local_range().size();
          const auto warpCount = subgroup.get_group_range().size();
          const int currentWarp = subgroup.get_group_id();
          const int threadInWarp = subgroup.get_local_id();
          const auto warpsNeeded = (size + sgSize - 1) / sgSize;
          auto acc = DefaultValue;
          
          #pragma unroll 4
          for (std::size_t i = currentWarp; i < warpsNeeded; i += warpCount) {
            const auto id = threadInWarp + i * sgSize;
            auto value = (id < size) ? static_cast<AccT>(ntload(&buffer[id])) : DefaultValue;
            value = sycl::reduce_over_group(subgroup, value, operation);
            acc = operation(acc, value);
          }
          if (threadInWarp == 0) {
            shmem[currentWarp] = acc;
          }
          idx.barrier();
          if (currentWarp == 0) {
            const auto lastWarpsNeeded = (warpCount + sgSize - 1) / sgSize;
            auto lastAcc = DefaultValue;
            #pragma unroll 2
            for (int i = 0; i < lastWarpsNeeded; ++i) {
              const auto id = threadInWarp + i * sgSize;
              auto value = (id < warpCount) ? shmem[id] : DefaultValue;
              value = sycl::reduce_over_group(subgroup, value, operation);
              lastAcc = operation(lastAcc, value);
            }
            if (threadInWarp == 0) {
              if (overrideResult) {
                ntstore(result, lastAcc);
              }
              else {
                ntstore(result, operation(ntload(result), lastAcc));
              }
            }
          }
        });
    });
}

template<typename T>
void run_benchmark(size_t VECTOR_SIZE, queue& q) {
    
    T *buffer = malloc_shared<T>(VECTOR_SIZE, q);
    T *result = malloc_shared<T>(1, q);
    
    // Initialize array
    q.parallel_for(VECTOR_SIZE, [=](id<1> i){ buffer[i] = 1; }).wait();

    std::vector<double> run_times;
    const int NUM_ITERATIONS = 10;
    
    for(int iter = 0; iter < NUM_ITERATIONS; iter++){
        
        using std::chrono::high_resolution_clock;
        using std::chrono::duration;
        
        auto t1 = high_resolution_clock::now();
        
        launchReduction<ReductionType::SUM, T, T>(
            result, 
            buffer, 
            VECTOR_SIZE, 
            sycl::plus<T>(), 
            true,  // overrideResult
            (void*)&q
        );
        
        q.wait();
        
        auto t2 = high_resolution_clock::now();
        
        duration<double, std::milli> ms_double = t2 - t1;
        run_times.push_back(ms_double.count());
    }
    
    double total_time = std::accumulate(run_times.begin()+1, run_times.end(), 0.0) / (NUM_ITERATIONS - 1);
    std::cout << run_times[0] << std::endl;
    std::cout << 1e-6*VECTOR_SIZE*sizeof(T)/run_times[0] << std::endl;
    std::cout << total_time << std::endl;
    std::cout << 1e-6*VECTOR_SIZE*sizeof(T)/total_time << std::endl;

    // Validation
    T expected = static_cast<T>(VECTOR_SIZE);
    if(std::fabs(static_cast<double>(*result - expected)) > 1e-3){
        std::cerr << "\nError: Reduction incorrect. Expected " << expected 
                  << ", got " << *result << "\n";
        throw std::runtime_error("Reduction incorrect");
    }

    // Cleanup
    sycl::free(buffer, q);
    sycl::free(result, q);
}

int main(int argc, char* argv[]){

    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << " <vector_size> <type>\n";
        std::cerr << "Types: int, float, double, long\n";
        return 1;
    }

    size_t VECTOR_SIZE = std::atol(argv[1]);
    std::string type = argv[2];

    queue q{property::queue::in_order()};

    try {
        if(type == "int") {
            run_benchmark<int>(VECTOR_SIZE, q);
        } else if(type == "float") {
            run_benchmark<float>(VECTOR_SIZE, q);
        } else if(type == "double") {
            run_benchmark<double>(VECTOR_SIZE, q);
        } else if(type == "long") {
            run_benchmark<long>(VECTOR_SIZE, q);
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