#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

void output_dev_info(const device& dev, const std::string& selector_name){
	std::cout << selector_name << " : Selected device: " << dev.getinfo<info::device::name() << "\n";
}

int main(){
	//Create queue on whatever default device that the implementation chooses. Implicit use of default_selector_v
	queue q_gpu{gpu_selector_v};
	std::cout << "Selected device: " << q_gpu.get_device().get_info<info::device::name>();
	std::cout
<< " -> Device vendor: "
<< q_gpu.get_device().get_info<info::device::vendor>()
<< "\n";

	queue q{cpu_selector_v};

	std::cout << "Selected device: "
<< q.get_device().get_info<info::device::name>();
std::cout
<< " -> Device vendor: "
<< q.get_device().get_info<info::device::vendor>()
<< "\n";

	return 0;
}
