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


  queue q{property::queue::in_order()};

  int *A = malloc_shared<int>(VECTOR_SIZE, q);
  int *sum = malloc_shared<int>(1, q);
  *sum = 0;

  q.parallel_for(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; });

  auto t1 = high_resolution_clock::now();

  q.parallel_for<class Reduce>(nd_range<1>{VECTOR_SIZE, 16}, [=](nd_item<1> it){
    int i = it.get_global_id(0);
    auto grp = it.get_group();
    int group_sum = reduce_over_group(grp, A[i], plus<>());
    if(grp.leader()){
        atomic_ref<int, memory_order::relaxed, memory_scope::system, access::address_space::global_space>(*sum)+= group_sum;
    }
  }).wait();

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

  if(std::abs(*sum - VECTOR_SIZE) > 1e-3){

  throw std::runtime_error ("Reduction incorrect");
  }
  return 0;
}
