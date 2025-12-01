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

  std::array<int, VECTOR_SIZE> A;

  for(std::size_t i=0; i < VECTOR_SIZE; i++){
  	A[i] = 1;
  }

  auto t1 = high_resolution_clock::now();

  for(std::size_t i=1; i < VECTOR_SIZE; i++){
  	A[0] += A[i];
  }
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "cpu code took " << ms_double.count()<<"ms\n";

  if(std::abs(A[0] - VECTOR_SIZE) > 1e-3){

  throw std::runtime_error ("Reduction incorrect");
  }
  return 0;
}
