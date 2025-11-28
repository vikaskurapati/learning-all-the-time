#include <array>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(){
constexpr int size = 16;
std::array<int, size> data;
buffer B{data};

queue q{};

std::cout << "Selected device is: "
<< q.get_device().get_info<info::device::name>()
<< "\n";

q.submit([&](handler& h) {
accessor acc{B, h};
h.parallel_for(size,
[=](auto& idx) { acc[idx] = idx; });
});

for(int i = 0; i < 20; i++){
	std::cout << data[i] << std::endl;
}

for(int i = 0; i < 20; i++){
        std::cout << data[size- i- 1] << std::endl;
}

return 0;
}

