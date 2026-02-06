#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    int provided;
    // DLB works best with MPI_THREAD_MULTIPLE or MPI_THREAD_SERIALIZED
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Each rank performs a workload
    // We can simulate imbalance by giving higher ranks more work
    long iterations = (world_rank + 1) * 10000000; 

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        #pragma omp for
        for (long i = 0; i < iterations; i++) {
            // Dummy computation
            double val = std::sqrt(i);
        }

        #pragma omp critical
        {
            std::cout << "Rank " << world_rank << " / " << world_size 
                      << " | Thread " << thread_id << " / " << num_threads 
                      << " finished work." << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) std::cout << "All ranks reached the barrier." << std::endl;

    MPI_Finalize();
    return 0;
}
