#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Compile flag: -arch=sm_" << props.major << props.minor << "\n";
    
    return 0;
}
