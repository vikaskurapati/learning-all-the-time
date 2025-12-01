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

    constexpr size_t SUBGROUP_SIZE = 16; // PVC native output by other functions
    constexpr size_t WORK_GROUP_SIZE = 256;
    constexpr size_t ITEMS_PER_WORK_ITEM = 8; // SIMD lane width

  queue q{property::queue::in_order()};

  int *A = malloc_shared<int>(VECTOR_SIZE, q);
  int sum = 0;

  q.parallel_for(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; });

  auto t1 = high_resolution_clock::now();

  size_t num_work_groups = (VECTOR_SIZE + (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM) - 1) / (WORK_GROUP_SIZE * ITEMS_PER_WORK_ITEM);

  int* partial_sums = malloc_device<int>(num_work_groups, q);
  q.submit([&](handler &h){
    auto local_mem = local_accessor<int, 1>(WORK_GROUP_SIZE, h);

    h.parallel_for(nd_range<1>(num_work_groups*WORK_GROUP_SIZE, WORK_GROUP_SIZE),[=](nd_item<1> item){
        size_t global_id = item.get_global_id(0);
        size_t local_id = item.get_local_id(0);
        size_t group_id = item.get_group(0);

        int thread_sum = 0;
        size_t base_idx = global_id*ITEMS_PER_WORK_ITEM;

        #pragma unroll
        for(size_t i = 0; i < ITEMS_PER_WORK_ITEM; i++){
            size_t idx = base_idx + i;
            if(idx < VECTOR_SIZE){
                thread_sum += A[idx];
            }
        }

        local_mem[local_id] = thread_sum;
        item.barrier(access::fence_space::local_space);

        for(size_t stride = WORK_GROUP_SIZE/2 ; stride > SUBGROUP_SIZE; stride >>=1){
            if(local_id < stride){
                local_mem[local_id] += local_mem[local_id + stride];
            }
            item.barrier(access::fence_space::local_space);
        }

        auto sg = item.get_sub_group();
        int subgroup_sum = local_mem[local_id];

        if(local_id < SUBGROUP_SIZE){
            subgroup_sum = reduce_over_group(sg, subgroup_sum, plus<int>());
            if(local_id == 0){
                partial_sums[group_id] = subgroup_sum;
            }
        }
    });
  });

  int* result = malloc_device<int>(1, q);
  q.memset(result, 0, sizeof(int));

  q.submit([&](handler& h){
    h.parallel_for(num_work_groups, [=](id<1> i){
        auto atomic = atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(result[0]);
        
        atomic.fetch_add(partial_sums[i]);
    });
  });

  q.wait();

  q.memcpy(&sum, result, sizeof(int)).wait();

  auto t2 = high_resolution_clock::now();

    auto kid = get_kernel_id<class Reduce>();
  auto kb = get_kernel_bundle<bundle_state::executable>(
q.get_context(), {q.get_device()}, {kid});
auto kernel = kb.get_kernel(kid);
std::cout
<< "The maximum work-group size for the kernel and "
"this device is: "
<< kernel.get_info<info::kernel_device_specific::
work_group_size>(
q.get_device())
<< "\n";
std::cout
<< "The preferred work-group size multiple for the "
"kernel and this device is: "
<< kernel.get_info<
info::kernel_device_specific::
preferred_work_group_size_multiple>(
q.get_device())
<< "\n";

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "gpu code took " << ms_double.count()<<"ms\n";

  if(std::abs(sum - VECTOR_SIZE) > 1e-3){

  throw std::runtime_error ("Reduction incorrect");
  }
  return 0;
}
