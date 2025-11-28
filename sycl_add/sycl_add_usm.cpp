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

  std::array<int, VECTOR_SIZE> A, B, C;
  for(std::size_t i=0; i < VECTOR_SIZE; i++){
    A[i] = 1;
    B[i] = 2;
  }

  
  auto t1 = high_resolution_clock::now();
  // time measurement in this scope
  {
  queue q{};
  
  std::cout << "Selected device: "<< q.get_device().get_info<info::device::name>()<< "\n";
 
  int *aDevice = malloc_device<int>(VECTOR_SIZE, q);
  int *bDevice = malloc_device<int>(VECTOR_SIZE, q);
  int *cDevice = malloc_device<int>(VECTOR_SIZE, q);

  q.submit([&](handler &h){
		  h.memcpy(aDevice, &A, VECTOR_SIZE*sizeof(int));
		 });
  q.submit([&](handler &h){
		  h.memcpy(bDevice, &B, VECTOR_SIZE*sizeof(int));
		  });

  q.wait();

  q.submit([&](handler &h){
		  h.parallel_for(range{VECTOR_SIZE}, [=](id<1> i){cDevice[i] = aDevice[i] + bDevice[i];});
		  });
  q.wait();
  // write kernel code here
  
  q.submit([&](handler &h){
		  //copy back to host
		  h.memcpy(&C[0], cDevice, VECTOR_SIZE*sizeof(int));
		  });
  q.wait();

  }
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "sycl code took " << ms_double.count()<<"ms\n";

  for(std::size_t i=0; i < VECTOR_SIZE; i++){
    if(C[i]!=3){
	std::cout << i << " : " << C[i]<<std::endl;
      throw std::runtime_error ("C value not matching in CPU code");
    }
  } 

  //std::cout << "Selected device: "<< q.get_device().get_info<info::device::name>()<< "\n";
  
  return 0;
}
