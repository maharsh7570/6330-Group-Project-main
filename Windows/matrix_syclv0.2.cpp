#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>

class MatrixMultiplyKernel;
class TiledMatrixMultiplyKernel;

// Naive kernel (baseline - your existing implementation)
void matrixMultiplySYCL(sycl::queue& q,
                        const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int N) {
    
    sycl::buffer<float, 1> bufA(A.data(), sycl::range<1>(A.size()));
    sycl::buffer<float, 1> bufB(B.data(), sycl::range<1>(B.size()));
    sycl::buffer<float, 1> bufC(C.data(), sycl::range<1>(C.size()));
    
    auto event = q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);
        
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
    
    event.wait();
}

// Tiled kernel with shared memory optimization
void matrixMultiplyTiledSYCL(sycl::queue& q,
                            const std::vector<float>& A,
                            const std::vector<float>& B,
                            std::vector<float>& C,
                            int N,
                            int tile_size = 16) {
    
    sycl::buffer<float, 1> bufA(A.data(), sycl::range<1>(A.size()));
    sycl::buffer<float, 1> bufB(B.data(), sycl::range<1>(B.size()));
    sycl::buffer<float, 1> bufC(C.data(), sycl::range<1>(C.size()));
    
    auto event = q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);
        
        // Use local_accessor for shared memory
        sycl::local_accessor<float, 1> localA(sycl::range<1>(tile_size * tile_size), h);
        sycl::local_accessor<float, 1> localB(sycl::range<1>(tile_size * tile_size), h);
        
        h.parallel_for<TiledMatrixMultiplyKernel>(
            sycl::nd_range<2>(sycl::range<2>(N, N), sycl::range<2>(tile_size, tile_size)),
            [=](sycl::nd_item<2> item) {
                int local_i = item.get_local_id(0);
                int local_j = item.get_local_id(1);
                int global_i = item.get_global_id(0);
                int global_j = item.get_global_id(1);
                
                float sum = 0.0f;
                
                // Number of tiles needed to cover the matrix
                int num_tiles = (N + tile_size - 1) / tile_size;
                
                for (int tile = 0; tile < num_tiles; ++tile) {
                    // Calculate tile indices
                    int tile_row = tile * tile_size + local_j;
                    int tile_col = tile * tile_size + local_i;
                    
                    // Load tile from matrix A into local memory
                    if (global_i < N && tile_row < N) {
                        localA[local_i * tile_size + local_j] = accA[global_i * N + tile_row];
                    } else {
                        localA[local_i * tile_size + local_j] = 0.0f;
                    }
                    
                    // Load tile from matrix B into local memory
                    if (tile_col < N && global_j < N) {
                        localB[local_i * tile_size + local_j] = accB[tile_col * N + global_j];
                    } else {
                        localB[local_i * tile_size + local_j] = 0.0f;
                    }
                    
                    // Synchronize to ensure all tiles are loaded
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Compute partial sum for this tile
                    for (int k = 0; k < tile_size; ++k) {
                        sum += localA[local_i * tile_size + k] * localB[k * tile_size + local_j];
                    }
                    
                    // Synchronize before loading next tile
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                // Write final result to global memory
                if (global_i < N && global_j < N) {
                    accC[global_i * N + global_j] = sum;
                }
            }
        );
    });
    
    event.wait();
}

// Advanced tiled kernel
void matrixMultiplyAdvancedTiledSYCL(sycl::queue& q,
                                    const std::vector<float>& A,
                                    const std::vector<float>& B,
                                    std::vector<float>& C,
                                    int N,
                                    int tile_size = 32) {
    
    sycl::buffer<float, 1> bufA(A.data(), sycl::range<1>(A.size()));
    sycl::buffer<float, 1> bufB(B.data(), sycl::range<1>(B.size()));
    sycl::buffer<float, 1> bufC(C.data(), sycl::range<1>(C.size()));
    
    auto event = q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);
        
        // Use local_accessor for shared memory
        sycl::local_accessor<float, 1> localA(sycl::range<1>(tile_size * tile_size), h);
        sycl::local_accessor<float, 1> localB(sycl::range<1>(tile_size * tile_size), h);
        
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(N, N), sycl::range<2>(tile_size, tile_size)),
            [=](sycl::nd_item<2> item) {
                int local_i = item.get_local_id(0);
                int local_j = item.get_local_id(1);
                int global_i = item.get_global_id(0);
                int global_j = item.get_global_id(1);
                
                float sum = 0.0f;
                
                int num_tiles = (N + tile_size - 1) / tile_size;
                
                for (int tile = 0; tile < num_tiles; ++tile) {
                    // Load current tile from A
                    if (global_i < N && (tile * tile_size + local_j) < N) {
                        localA[local_i * tile_size + local_j] = 
                            accA[global_i * N + (tile * tile_size + local_j)];
                    } else {
                        localA[local_i * tile_size + local_j] = 0.0f;
                    }
                    
                    // Load current tile from B  
                    if ((tile * tile_size + local_i) < N && global_j < N) {
                        localB[local_i * tile_size + local_j] = 
                            accB[(tile * tile_size + local_i) * N + global_j];
                    } else {
                        localB[local_i * tile_size + local_j] = 0.0f;
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Compute with the tile
                    for (int k = 0; k < tile_size; ++k) {
                        sum += localA[local_i * tile_size + k] * localB[k * tile_size + local_j];
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                // Write results
                if (global_i < N && global_j < N) {
                    accC[global_i * N + global_j] = sum;
                }
            }
        );
    });
    
    event.wait();
}

void initializeMatrix(std::vector<float>& matrix, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dist(gen);
    }
}

bool validateResults(const std::vector<float>& C1, const std::vector<float>& C2, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(C1[i] - C2[i]) > tolerance) {
            std::cout << "Validation failed at index " << i 
                      << ": " << C1[i] << " vs " << C2[i] 
                      << " (diff: " << std::fabs(C1[i] - C2[i]) << ")\n";
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 512;
    const int iterations = 5;
    const int warmup = 2;
    
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_naive(N * N, 0.0f);
    std::vector<float> C_tiled16(N * N, 0.0f);
    std::vector<float> C_tiled32(N * N, 0.0f);
    std::vector<float> C_advanced(N * N, 0.0f);
    
    // Initialize matrices
    initializeMatrix(A, N);
    initializeMatrix(B, N);
    
    try {
        sycl::queue q;
        
        std::cout << "SYCL Matrix Multiplication Benchmark\n";
        std::cout << "Running on: " 
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Matrix size: " << N << "x" << N << "\n";
        std::cout << "Iterations: " << iterations << " (plus " << warmup << " warmup)\n\n";
        
        // Store results for each tile size
        double tiled16_time = 0.0;
        double tiled32_time = 0.0;
        double advanced_time = 0.0;
        
        // Test 16x16 tiled kernel
        std::cout << "=== Testing Tiled Kernel (Tile Size: 16) ===\n";
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            matrixMultiplyTiledSYCL(q, A, B, C_tiled16, N, 16);
        }
        
        double min_time_16 = std::numeric_limits<double>::max();
        double max_time_16 = 0.0;
        
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(C_tiled16.begin(), C_tiled16.end(), 0.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplyTiledSYCL(q, A, B, C_tiled16, N, 16);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double seconds = duration.count() / 1e6;
            tiled16_time += seconds;
            min_time_16 = std::min(min_time_16, seconds);
            max_time_16 = std::max(max_time_16, seconds);
            
            std::cout << "Iteration " << (iter + 1) << ": " 
                      << duration.count() / 1000.0 << " ms\n";
        }
        
        double avg_time_16 = tiled16_time / iterations;
        double gflops_16 = (2.0 * N * N * N) / (avg_time_16 * 1e9);
        
        std::cout << "Tile Size 16 - Average: " << avg_time_16 * 1000 
                  << " ms, Performance: " << gflops_16 << " GFLOP/s\n";
        std::cout << "Min: " << min_time_16 * 1000 << " ms, Max: " << max_time_16 * 1000 << " ms\n";
        
        // Test 32x32 tiled kernel
        std::cout << "\n=== Testing Tiled Kernel (Tile Size: 32) ===\n";
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            matrixMultiplyTiledSYCL(q, A, B, C_tiled32, N, 32);
        }
        
        double min_time_32 = std::numeric_limits<double>::max();
        double max_time_32 = 0.0;
        
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(C_tiled32.begin(), C_tiled32.end(), 0.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplyTiledSYCL(q, A, B, C_tiled32, N, 32);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double seconds = duration.count() / 1e6;
            tiled32_time += seconds;
            min_time_32 = std::min(min_time_32, seconds);
            max_time_32 = std::max(max_time_32, seconds);
            
            std::cout << "Iteration " << (iter + 1) << ": " 
                      << duration.count() / 1000.0 << " ms\n";
        }
        
        double avg_time_32 = tiled32_time / iterations;
        double gflops_32 = (2.0 * N * N * N) / (avg_time_32 * 1e9);
        
        std::cout << "Tile Size 32 - Average: " << avg_time_32 * 1000 
                  << " ms, Performance: " << gflops_32 << " GFLOP/s\n";
        std::cout << "Min: " << min_time_32 * 1000 << " ms, Max: " << max_time_32 * 1000 << " ms\n";
        
        // Test advanced tiled kernel
        std::cout << "\n=== Testing Advanced Tiled Kernel ===\n";
        
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(C_advanced.begin(), C_advanced.end(), 0.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplyAdvancedTiledSYCL(q, A, B, C_advanced, N, 32);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double seconds = duration.count() / 1e6;
            advanced_time += seconds;
            
            std::cout << "Iteration " << (iter + 1) << ": " 
                      << duration.count() / 1000.0 << " ms\n";
        }
        
        double avg_advanced = advanced_time / iterations;
        double gflops_advanced = (2.0 * N * N * N) / (avg_advanced * 1e9);
        
        std::cout << "Advanced Tiled - Average: " << avg_advanced * 1000 
                  << " ms, Performance: " << gflops_advanced << " GFLOP/s\n";
        
        // Validation
        std::cout << "\n=== Validation ===\n";
        matrixMultiplySYCL(q, A, B, C_naive, N);
        matrixMultiplyTiledSYCL(q, A, B, C_tiled16, N, 16);
        
        if (validateResults(C_naive, C_tiled16, N)) {
            std::cout << "✓ Tiled kernel results match naive kernel\n";
        } else {
            std::cout << "✗ Tiled kernel results DO NOT match naive kernel!\n";
        }
        
        // Performance comparison summary
        std::cout << "\n=== PERFORMANCE SUMMARY ===\n";
        std::cout << "Naive (from your results): ~5.54 ms, 48.44 GFLOP/s\n";
        std::cout << "Tiled 16x16: " << avg_time_16 * 1000 << " ms, " << gflops_16 << " GFLOP/s\n";
        std::cout << "Tiled 32x32: " << avg_time_32 * 1000 << " ms, " << gflops_32 << " GFLOP/s\n";
        std::cout << "Advanced Tiled: " << avg_advanced * 1000 << " ms, " << gflops_advanced << " GFLOP/s\n";
        
        // Calculate speedup
        double naive_time = 5.54; // From your original results
        std::cout << "\n=== SPEEDUP OVER NAIVE ===\n";
        std::cout << "Tiled 16x16: " << naive_time / (avg_time_16 * 1000) << "x faster\n";
        std::cout << "Tiled 32x32: " << naive_time / (avg_time_32 * 1000) << "x faster\n";
        std::cout << "Advanced Tiled: " << naive_time / (avg_advanced * 1000) << "x faster\n";
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}