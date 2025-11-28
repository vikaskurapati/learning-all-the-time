#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>

using namespace sycl;

#define VECTOR_SIZE 100000000

int main(){

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

  std::array<int, VECTOR_SIZE> A, B, C_cpu, C_sycl;
  for(std::size_t i=0; i < VECTOR_SIZE; i++){
    A[i] = 1;
    B[i] = 2;
  }

  auto t1 = high_resolution_clock::now();
  for(std::size_t i=0; i < VECTOR_SIZE; i++){
  	C_cpu[i] = A[i] + B[i];
  }
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "cpu code took " << ms_double.count()<<"ms\n";


  for(std::size_t i=0; i < VECTOR_SIZE; i++){
    if(C_cpu[i]!=3){
      throw std::runtime_error ("C value not matching in CPU code");
    }
  }
  
  return 0;
}
