#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>

using namespace sycl;

#define VECTOR_SIZE 1000000000

int main(){

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

    constexpr size_t WORK_GROUP_SIZE = 256;
    constexpr size_t ITEMS_PER_WORK_ITEM = 8;

  queue q{property::queue::in_order()};

  int *A = malloc_shared<int>(VECTOR_SIZE, q);
  int sum = 0;

  q.parallel_for(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; });

  auto t1 = high_resolution_clock::now();

  auto num_work_groups = (VECTOR_SIZE + WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM - 1) / (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM);

  int* partial_sums = malloc_device<int>(num_work_groups, q);

  q.submit([&](handler &h){

    auto local_mem = local_accessor<int, 1>(WORK_GROUP_SIZE, h);

    h.parallel_for(nd_range<1>(nd_range<1>{num_work_groups*WORK_GROUP_SIZE, WORK_GROUP_SIZE}), [=](nd_item<1> it){
        size_t global_id = it.get_global_id(0);
        size_t local_id = it.get_local_id(0);
        size_t group_id = it.get_group(0);

        auto base_idx = group_id*(WORK_GROUP_SIZE*ITEMS_PER_WORK_ITEM) + local_id*ITEMS_PER_WORK_ITEM;

        // local_mem[local_id] = 0;
        int thread_sum = 0;
        for(size_t i = 0; i < ITEMS_PER_WORK_ITEM; i++){
            size_t idx = base_idx + i;
            if(idx < VECTOR_SIZE){
                thread_sum += A[idx];
            }
        }

        local_mem[local_id] = thread_sum;
        it.barrier(access::fence_space::local_space);

        if(local_id == 0){
            for(size_t i = 1; i < WORK_GROUP_SIZE; i++){
                local_mem[0] += local_mem[i];
            }
        }
        it.barrier(access::fence_space::local_space);

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


  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "gpu code took " << ms_double.count()<<"ms\n";

  if(std::abs(*sum - VECTOR_SIZE) > 1e-3){

  throw std::runtime_error ("Reduction incorrect");
  }
  return 0;
}
