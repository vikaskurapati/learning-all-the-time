#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cmath>

using namespace sycl;

#define VECTOR_SIZE 100000000

auto handle_async_error = [](exception_list elist){
    for(auto &e: elist){
        try{
            std::rethrow_exception(e);
        } catch(sycl::exception& e){
            std::cout << "Caught SYCL ASYNC exception!!\n";
        }catch(...){
            std::cout << "Caught non SYCL ASYNC exception!!\n";
        }
    }
    std::terminate();
};

void say_device(const queue& Q){
    std::cout << "Device: " << Q.get_device().get_info<info::device::name>() << "\n";
}

class something_went_wrong {}; //Example exception type

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
  queue q{gpu_selector_v, handle_async_error};
  
    say_device(q);
 
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
    size_t N = VECTOR_SIZE;
    size_t W = 16;
    h.parallel_for(range{W}, [=](item<1> it){
        for(int i = it.get_id()[0]; i < N; i+= it.get_range()[0]){
            cDevice[i] = aDevice[i] + bDevice[i];
        }
    });
});

    q.wait();
  
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
      throw std::runtime_error ("C value not matching in code");
    }
  }
  
  return 0;
}
