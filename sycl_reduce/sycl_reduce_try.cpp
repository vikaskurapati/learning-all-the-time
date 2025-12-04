#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>

using namespace sycl;

template<typename T>
void run_benchmark(size_t VECTOR_SIZE, size_t WORK_GROUP_SIZE, size_t ITEMS_PER_WORK_ITEM, queue& q) {
    
    T *A = malloc_shared<T>(VECTOR_SIZE, q);
    
    // Initialize array
    q.parallel_for(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; }).wait();

    size_t num_work_groups = (VECTOR_SIZE + WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM - 1) 
                             / (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM);
    
    T* partial_sums = malloc_device<T>(num_work_groups, q);

    std::vector<double> run_times;
    const int NUM_ITERATIONS = 10;
    T sum = 0;
    
    for(int iter = 0; iter < NUM_ITERATIONS; iter++){
        
        using std::chrono::high_resolution_clock;
        using std::chrono::duration;
        
        auto t1 = high_resolution_clock::now();
        
        T* result = malloc_device<T>(1, q);
        q.memset(result, 0, sizeof(T));
        
        q.submit([&](handler &h){
            auto local_mem = local_accessor<T, 1>(WORK_GROUP_SIZE, h);

            h.parallel_for(nd_range<1>(num_work_groups * WORK_GROUP_SIZE, WORK_GROUP_SIZE), 
            [=](nd_item<1> it){
                size_t local_id = it.get_local_id(0);
                size_t group_id = it.get_group(0);

                size_t base_idx = group_id * (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM) + local_id * ITEMS_PER_WORK_ITEM;

                T thread_sum = 0;
                for(size_t i = 0; i < ITEMS_PER_WORK_ITEM; i++){
                    size_t idx = base_idx + i;
                    if(idx < VECTOR_SIZE){
                        thread_sum += A[idx];
                    }
                }

                local_mem[local_id] = thread_sum;
                it.barrier(access::fence_space::local_space);

                // Tree reduction in shared memory
                for(size_t stride = WORK_GROUP_SIZE / 2; stride > 0; stride >>= 1){
                    if(local_id < stride){
                        local_mem[local_id] += local_mem[local_id + stride];
                    }
                    it.barrier(access::fence_space::local_space);
                }

                if(local_id == 0){
                    partial_sums[group_id] = local_mem[0];
                }
            });
        });

        // Final reduction of partial sums (atomic)
        q.submit([&](handler& h){
            h.parallel_for(num_work_groups, [=](id<1> i){
                auto atomic = atomic_ref<T, memory_order::relaxed, 
                                         memory_scope::device, 
                                         access::address_space::global_space>(result[0]);
                atomic.fetch_add(partial_sums[i]);
            });
        });

        q.wait();
        q.memcpy(&sum, result, sizeof(T)).wait();
        
        auto t2 = high_resolution_clock::now();
        
        duration<double, std::milli> ms_double = t2 - t1;
        run_times.push_back(ms_double.count());
        
        sycl::free(result, q);
    }
    
    double total_time = std::accumulate(run_times.begin(), run_times.end(), 0.0) / NUM_ITERATIONS;
    std::cout << total_time;

    // Cleanup
    sycl::free(A, q);
    sycl::free(partial_sums, q);

    // Validation
    T expected = static_cast<T>(VECTOR_SIZE);
    if(std::fabs(static_cast<double>(sum - expected)) > 1e-3){
        throw std::runtime_error("Reduction incorrect");
    }
}

int main(int argc, char* argv[]){

    if(argc != 5){
        std::cerr << "Usage: " << argv[0] << " <vector_size> <work_group_size> <items_per_work_item> <type>\n";
        std::cerr << "Types: int, float, double, long\n";
        return 1;
    }

    size_t VECTOR_SIZE = std::atol(argv[1]);
    size_t WORK_GROUP_SIZE = std::atoi(argv[2]);
    size_t ITEMS_PER_WORK_ITEM = std::atoi(argv[3]);
    std::string type = argv[4];

    queue q{property::queue::in_order()};

    try {
        if(type == "int") {
            run_benchmark<int>(VECTOR_SIZE, WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM, q);
        } else if(type == "float") {
            run_benchmark<float>(VECTOR_SIZE, WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM, q);
        } else if(type == "double") {
            run_benchmark<double>(VECTOR_SIZE, WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM, q);
        } else if(type == "long") {
            run_benchmark<long>(VECTOR_SIZE, WORK_GROUP_SIZE, ITEMS_PER_WORK_ITEM, q);
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