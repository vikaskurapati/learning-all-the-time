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

  q.parallel_for<class Reduce>(VECTOR_SIZE, [=] (id<1> i){ 
    atomic_ref<int, memory_order::relaxed, memory_scope::system, access::address_space::global_space>(*sum)+= A[i];
  }).wait();

  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "gpu code took " << ms_double.count()<<"ms\n";

  if(std::abs(*sum - VECTOR_SIZE) > 1e-3){

  throw std::runtime_error ("Reduction incorrect");
  }
  return 0;
}
