#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <limits>
#include <sycl/sycl.hpp>

using namespace std;
using namespace sycl;

template <typename T>
void intialize_matrix(vector<T>& mat, int rows, int cols){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for(int i=0; i < rows*cols; ++i){
        mat[i] = dis(gen);
    }
}

template <typename T>
void gemm_naive(const vector<T>& A, const vector<T>& B, vector<T>& C, int M, int N, int K){
    for(int i=0; i < M; ++i){
        for(int j=0; j < N; ++j){
            T sum=0;
            for(int k=0; k < K; ++k){
                sum += A[i*K + k]*B[k*N+j];
            }
            C[i*N + j] = sum;
        }
    }
}

template<typename T>
bool verify(const vector<T>& ref, const vector<T>& test, int M, int N){
    T epsilon = std::numeric_limits<T>::epsilon()*1000;
    for(int i=0; i < M*N; ++i){
        if(abs(ref[i]-test[i]) > epsilon){
            cout << "Verification failed at index " << i << ": Expected " << ref[i] << ", got " << test[i] << "\n";
            return false;
        }
    }

    return true;
}

int main(){
    //Matrix dimensions
    const int M = 56;
    const int N = 9;
    const int K = 35;

    cout << "Matrix Dimensions: " << M << "x" << K << " * " << K << "x" << N << "\n";

    vector<float> A(M*K), B(K*N), C_ref(M*N, 0.0), C_test(M*N, 0.0);

    intialize_matrix(A, M, K);
    intialize_matrix(B, K, N);

    gemm_naive(A, B, C_ref, M, N, K);

    const int NUM_ITERATIONS = 10;

    queue q(default_selector_v);
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

    float* d_A = malloc_device<float>(M*K, q);
    float* d_B = malloc_device<float>(K*N, q);
    float* d_C = malloc_device<float>(M*N, q);
    q.memcpy(d_A, A.data(), M*K*sizeof(float));
    q.memcpy(d_B, B.data(), K*N*sizeof(float));
    q.wait();

    double elapsedTime = 0.0;

    for(int i = 0; i < NUM_ITERATIONS; i++){
        auto start = chrono::high_resolution_clock::now();

        q.submit([&](handler& h){

size_t num_vec_groups = N / 4;

h.parallel_for(range<2>(M, 1), [=](id<2> index) {
    int row = index[0];

    // 1. Process blocks of 4
    for (int g = 0; g < num_vec_groups; g++) {
        int col = g * 4;
        float4 acc = float4(0.0f);
        for (int k = 0; k < K; k++) {
            float a_val = d_A[row * K + k];
            float4 b_vec = *reinterpret_cast<const float4*>(&d_B[k * N + col]);
            acc += a_val * b_vec;
        }
        *reinterpret_cast<float4*>(&d_C[row * N + col]) = acc;
    }

    // 2. EDGE CASE: Cleanup loop for remaining columns
    for (int col = num_vec_groups * 4; col < N; col++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += d_A[row * K + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = acc;
    }
});
        }).wait();

        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;

        elapsedTime += elapsed.count();
    }

    elapsedTime = elapsedTime / NUM_ITERATIONS;

    double gFlops = (2*M*N*K) / (elapsedTime * 1e9);

    cout << "Execution Time: " << std::fixed << std::setprecision(4) << elapsedTime*1000 << " milli seconds" << "\n";
    cout << "Performance: " << gFlops << " GFLOPS" << "\n";


    gemm_naive(A, B, C_ref, M, N, K);
    q.memcpy(C_test.data(), d_C, M*N*sizeof(float)).wait();

    if (verify(C_ref, C_test, M, N)) {
        cout << "Status: SUCCESS (Results Match)" << "\n";
    } else {
        cout << "Status: FAILURE" << "\n";
    }

    return 0;

}