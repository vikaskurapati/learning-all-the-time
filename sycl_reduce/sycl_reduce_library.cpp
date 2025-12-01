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

  float *A = malloc_shared<float>(VECTOR_SIZE, q);

  q.parallel_for<class Assign>(VECTOR_SIZE, [=](id<1> i){ A[i] = 1; });

  auto t1 = high_resolution_clock::now();

  float sum = 0;

  q.parallel_for<class Reduce>(range<1>{VECTOR_SIZE}, reduction(sum, plus<>()),[=](id<1> i, auto& sum_reduction){
    sum_reduction += A[i];
  });

  q.wait();

  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "gpu code took " << ms_double.count()<<"ms\n";

  if(std::abs(sum - VECTOR_SIZE) > 1e-3){

  throw std::runtime_error ("Reduction incorrect");
  }
  return 0;
}
