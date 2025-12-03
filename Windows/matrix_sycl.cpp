#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

class MatrixMultiplyKernel;

void matrixMultiplySYCL(sycl::queue& q,
                        const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int N) {
    
    // Create SYCL buffers
    sycl::buffer<float, 1> bufA(A.data(), sycl::range<1>(A.size()));
    sycl::buffer<float, 1> bufB(B.data(), sycl::range<1>(B.size()));
    sycl::buffer<float, 1> bufC(C.data(), sycl::range<1>(C.size()));
    
    // Submit kernel to queue
    auto event = q.submit([&](sycl::handler& h) {
        // Create accessors
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);
        
        // Define kernel
        h.parallel_for<MatrixMultiplyKernel>(
            sycl::range<2>(N, N), 
            [=](sycl::id<2> idx) {
                int i = idx[0];
                int j = idx[1];
                float sum = 0.0f;
                
                for (int k = 0; k < N; ++k) {
                    sum += accA[i * N + k] * accB[k * N + j];
                }
                
                accC[i * N + j] = sum;
            }
        );
    });
    
    event.wait(); // Wait for kernel completion
}

void initializeMatrix(std::vector<float>& matrix, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dist(gen);
    }
}

int main() {
    const int N = 512;
    const int iterations = 5;
    const int warmup = 2;  // Warmup runs
    
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);
    
    // Initialize matrices
    initializeMatrix(A, N);
    initializeMatrix(B, N);
    
    try {
        // Create SYCL queue (default device selector)
        sycl::queue q;
        
        std::cout << "SYCL Matrix Multiplication Benchmark\n";
        std::cout << "Running on: " 
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Matrix size: " << N << "x" << N << "\n";
        std::cout << "Iterations: " << iterations << " (plus " << warmup << " warmup)\n\n";
        
        // Warmup runs
        for (int i = 0; i < warmup; ++i) {
            matrixMultiplySYCL(q, A, B, C, N);
        }
        
        double total_time = 0.0;
        
        // Benchmark runs
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(C.begin(), C.end(), 0.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplySYCL(q, A, B, C, N);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double seconds = duration.count() / 1e6;
            total_time += seconds;
            
            std::cout << "Iteration " << (iter + 1) << ": " 
                      << duration.count() / 1000.0 << " ms\n";
        }
        
        double avg_time = total_time / iterations;
        double gflops = (2.0 * N * N * N) / (avg_time * 1e9);
        
        std::cout << "\n=== SYCL RESULTS ===\n";
        std::cout << "Average time: " << avg_time * 1000 << " ms\n";
        std::cout << "Performance: " << gflops << " GFLOP/s\n";
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}