#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <limits>

using namespace std;

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
            cout << "Verification failed at index " << i << ": Expected " << ref[i] << ", got " << test[i] << endl;
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

    cout << "Matrix Dimensions: " << M << "x" << K << " * " << K << "x" << N << endl;

    vector<double> A(M*K), B(K*N), C_ref(M*N, 0.0), C_test(M*N, 0.0);

    intialize_matrix(A, M, K);
    intialize_matrix(B, K, N);

    gemm_naive(A, B, C_ref, M, N, K);

    const int NUM_ITERATIONS = 10;

    double elapsedTime = 0.0;

    for(int i = 0; i < NUM_ITERATIONS; i++){
        auto start = chrono::high_resolution_clock::now();
        gemm_naive(A, B, C_test, M, N, K);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;

        elapsedTime += elapsed.count();
    }

    elapsedTime = elapsedTime / NUM_ITERATIONS;

    double gFlops = (2*M*N*K) / (elapsedTime * 1e9);

    cout << "Execution Time: " << fixed << setprecision(4) << elapsedTime*1000 << " milli seconds" << endl;
    cout << "Performance: " << gFlops << " GFLOPS" << endl;


    gemm_naive(A, B, C_ref, M, N, K); 
    if (verify(C_ref, C_test, M, N)) {
        cout << "Status: SUCCESS (Results Match)" << endl;
    } else {
        cout << "Status: FAILURE" << endl;
    }

    return 0;

}