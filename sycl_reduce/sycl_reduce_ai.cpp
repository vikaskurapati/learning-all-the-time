#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cmath>

using namespace sycl;

#define VECTOR_SIZE 1000000000

int main(){

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    constexpr size_t WORK_GROUP_SIZE = 256;
    constexpr size_t ITEMS_PER_WORK_ITEM = 8; 

    // Use an in-order queue to simplify dependency management
    queue q{property::queue::in_order()};

    std::cout << "Running on: " 
              << q.get_device().get_info<info::device::name>() << "\n";

    int *A = malloc_shared<int>(VECTOR_SIZE, q);
    int sum = 0;

    // Initialize array
    q.parallel_for(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; }).wait();

    auto t1 = high_resolution_clock::now();

    size_t num_work_groups = (VECTOR_SIZE + (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM) - 1) 
                             / (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM);

    int* partial_sums = malloc_device<int>(num_work_groups, q);

    q.submit([&](handler &h){
        auto local_mem = local_accessor<int, 1>(WORK_GROUP_SIZE, h);

        h.parallel_for(nd_range<1>(num_work_groups * WORK_GROUP_SIZE, WORK_GROUP_SIZE), 
        [=](nd_item<1> item){
            size_t global_id = item.get_global_id(0);
            size_t local_id = item.get_local_id(0);
            size_t group_id = item.get_group(0);

            // 1. Thread-Local Reduction
            int thread_sum = 0;
            size_t base_idx = group_id * (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM) 
                              + (local_id * ITEMS_PER_WORK_ITEM);

            #pragma unroll
            for(size_t i = 0; i < ITEMS_PER_WORK_ITEM; i++){
                size_t idx = base_idx + i;
                // Boundary check handles the last work-group if it's not full
                if(idx < VECTOR_SIZE){
                    thread_sum += A[idx];
                }
            }

            local_mem[local_id] = thread_sum;
            item.barrier(access::fence_space::local_space);

            for(size_t stride = WORK_GROUP_SIZE / 2; stride > 0; stride >>= 1){
                if(local_id < stride){
                    local_mem[local_id] += local_mem[local_id + stride];
                }
                item.barrier(access::fence_space::local_space);
            }

            // 3. Write Partial Sum
            if(local_id == 0){
                partial_sums[group_id] = local_mem[0];
            }
        });
    });

    int* result = malloc_device<int>(1, q);
    q.memset(result, 0, sizeof(int));

    // Final reduction of partial sums (atomic)
    q.submit([&](handler& h){
        h.parallel_for(num_work_groups, [=](id<1> i){
            auto atomic = atomic_ref<int, memory_order::relaxed, 
                                     memory_scope::device, 
                                     access::address_space::global_space>(result[0]);
            atomic.fetch_add(partial_sums[i]);
        });
    });

    q.wait();

    q.memcpy(&sum, result, sizeof(int)).wait();

    auto t2 = high_resolution_clock::now();

    // Cleanup resources (Fixing the memory leak)
    sycl::free(A, q);
    sycl::free(partial_sums, q);
    sycl::free(result, q);

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "GPU code took " << ms_double.count() << "ms\n";

    if(std::abs(sum - VECTOR_SIZE) > 1e-3){ // Tolerance not strictly needed for integers
        throw std::runtime_error("Reduction incorrect");
    }

    return 0;
}