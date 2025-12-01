#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cmath>

using namespace sycl;

#define VECTOR_SIZE 100000000

int main(){

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

    queue q{};

    // C-style
    int *A = static_cast<int*>(malloc_shared(VECTOR_SIZE * sizeof(int), q.get_device(), q.get_context()));

    // C++-style
    int *B = malloc_shared<int>(VECTOR_SIZE, q);

    //C++-allocator style
    usm_allocator<int, usm::alloc::shared> alloc(q);
    int *C = alloc.allocate(VECTOR_SIZE);

  for(std::size_t i=0; i < VECTOR_SIZE; i++){
    A[i] = 1;
    B[i] = 2;
  }

  
  auto t1 = high_resolution_clock::now();
  // time measurement in this scope
  {
  
  std::cout << "Selected device: "<< q.get_device().get_info<info::device::name>()<< "\n";

  q.submit([&](handler &h){
    size_t work_per_group = 2048;
    size_t num_groups = (VECTOR_SIZE + work_per_group - 1) / work_per_group;
		  h.parallel_for<class Add>(range{num_groups, work_per_group}, [=](id<2> i){
        size_t idx = i[0] * work_per_group + i[1];
        if(idx < VECTOR_SIZE)
        {C[idx] = A[idx] + B[idx];}});
		  });
  q.wait();
  // write kernel code here

  }
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "sycl code took " << ms_double.count()<<"ms\n";

  for(std::size_t i=0; i < VECTOR_SIZE; i++){
    if(C[i]!=3){
	std::cout << i << " : " << C[i]<<std::endl;
      throw std::runtime_error ("C value not matching in GPU code");
    }
  }

  auto kid = get_kernel_id<class Add>();
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

  free(A, q.get_context());
    free(B, q);
  alloc.deallocate(C, VECTOR_SIZE);

  //std::cout << "Selected device: "<< q.get_device().get_info<info::device::name>()<< "\n";
  
  return 0;
}
