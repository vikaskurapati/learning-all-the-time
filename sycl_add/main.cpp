#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

const std::string secret{
"Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf/!"
"J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"};

const auto sz = secret.size();

#define VECTOR_SIZE 1000000

int main(){
  std::array<int, VECTOR_SIZE> A, B, C;
  for(std::size_t i; i < VECTOR_SIZE; i++){
    A[i] = 1;
    B[i] = 2;
    C[i] = 3;
  }

  std::cout << result << "\n";
  queue q;
  char* result = malloc_shared<char> (sz, q);
  std::memcpy(result, secret.data(), sz);

  q.parallel_for(sz, [=](auto& i){
      result[i] -= 1;
  }).wait();

  std::cout << result << "\n";
  free(result, q);
  return 0;
}
