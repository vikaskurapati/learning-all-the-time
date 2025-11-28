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
  
  range<1> num_items{VECTOR_SIZE};

  buffer<int> ABuf(&A[0], VECTOR_SIZE);
  buffer<int> BBuf(&B[0], VECTOR_SIZE);
  buffer<int> CBuf(&C[0], VECTOR_SIZE);

  std::cout << "Selected device: "<< q.get_device().get_info<info::device::name>()<< "\n";
 
  // write kernel code here
  q.submit([&](auto &h){
		  //create device accessors
		  accessor aA(ABuf, h, read_only);
		  accessor aB(BBuf, h, read_only);
		  accessor aC(CBuf, h, write_only, no_init);
		  
		  h.parallel_for(num_items, [=](auto i){aC[i] = aB[i] + aA[i]; });
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
