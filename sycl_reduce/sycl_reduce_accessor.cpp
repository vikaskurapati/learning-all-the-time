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
  buffer<int> A{range{VECTOR_SIZE}};
  
  q.submit([&](handler &h){
          accessor a{A, h};
          h.parallel_for(VECTOR_SIZE, [=](id<1> i){ a[i] = 1;});
  });
  
  auto t1 = high_resolution_clock::now();
  q.submit([&](handler &h){
          accessor a{A, h};  // Changed from data1 to A
          h.single_task([=](){
                  for (int i=1; i<VECTOR_SIZE; i++){
                          a[0] += a[i];
                  }
          });
  });
  q.wait();
  auto t2 = high_resolution_clock::now();
  
  duration<double, std::milli> ms_double = t2-t1;
  std::cout << "gpu code took " << ms_double.count()<<"ms\n";
  
  // Need host accessor to read back result
  host_accessor h_a{A, read_only};
  if(std::abs(h_a[0] - VECTOR_SIZE) > 1e-3){
          throw std::runtime_error("Reduction incorrect");
  }
  return 0;
}
