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
    
    T *A = malloc_device<T>(VECTOR_SIZE, q);
    T sum = 0;

    // Initialize array
    q.parallel_for(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; }).wait();

    size_t num_work_groups = (VECTOR_SIZE + (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM) - 1)
                             / (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM);


    std::vector<double> run_times;
    const int NUM_ITERATIONS = 10;
    
    for(int i = 0; i < NUM_ITERATIONS; i++){

        using std::chrono::high_resolution_clock;
        using std::chrono::duration;
        
        T* result = malloc_device<T>(1, q);
        
        auto t1 = high_resolution_clock::now();

        sum = 0;

        q.memset(result, 0, sizeof(T));
        q.submit([&](handler &h){
            auto local_mem = local_accessor<T, 1>(WORK_GROUP_SIZE, h);

            h.parallel_for(nd_range<1>(num_work_groups * WORK_GROUP_SIZE, WORK_GROUP_SIZE),
            [=](nd_item<1> item){
                size_t local_id = item.get_local_id(0);
                size_t group_id = item.get_group(0);

                // Thread-Local Reduction
                T thread_sum = 0;
                size_t base_idx = group_id * (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM) + local_id;

                #pragma unroll
                for(size_t i = 0; i < ITEMS_PER_WORK_ITEM*WORK_GROUP_SIZE; i+=WORK_GROUP_SIZE){
                    size_t idx = base_idx + i;
                    if(idx < VECTOR_SIZE){
                        thread_sum += A[idx];
                    }
                }

                local_mem[local_id] = thread_sum;
                item.barrier(access::fence_space::local_space);

                auto reduced_value = reduce_over_group(item.get_group(), thread_sum, plus<>());

                if(local_id == 0){
                    auto atomic = atomic_ref<T, memory_order::relaxed,
                                            memory_scope::device,
                                            access::address_space::global_space>(result[0]);
                    atomic.fetch_add(reduced_value);
                }
            });
        });

        q.wait();

        auto t2 = high_resolution_clock::now();

        q.memcpy(&sum, result, sizeof(T)).wait();


        duration<double, std::milli> ms_double = t2 - t1;
        run_times.push_back(ms_double.count());
        sycl::free(result, q);
    }
    
    double total_time = std::accumulate(run_times.begin()+1, run_times.end(), 0.0)/((NUM_ITERATIONS - 1));
    std::cout << run_times[0] << std::endl;
    std::cout << 1e-6*VECTOR_SIZE*sizeof(T)/run_times[0] << std::endl;
    std::cout << total_time << std::endl;
    std::cout << 1e-6*VECTOR_SIZE*sizeof(T)/total_time << std::endl;


    // Cleanup resources
    sycl::free(A, q);

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