Installed and configured AdaptiveCpp (SYCL 2020) on macOS
Successfully compiled and ran a real SYCL parallel kernel
Verified correct output and timing
Your environment can now compile any SYCL project


Test Run...
(base) ahmadmohammad@MacBook-Pro-101 macOS % ./matrix_sycl 
SYCL Matrix Multiplication Benchmark
Running on: AdaptiveCpp OpenMP host device
Matrix size: 512x512
Iterations: 5 (plus 2 warmup)

[AdaptiveCpp Warning] This application uses SYCL buffers; the SYCL buffer-accessor model is well-known to introduce unnecessary overheads. Please consider migrating to the SYCL2020 USM model, in particular device USM (sycl::malloc_device) combined with in-order queues for more performance. See the AdaptiveCpp performance guide for more information: 
https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md
Iteration 1: 84.48 ms
Iteration 2: 115.125 ms
Iteration 3: 72.713 ms
Iteration 4: 66.991 ms
Iteration 5: 64.229 ms

=== SYCL RESULTS ===
Average time: 80.7076 ms
Performance: 3.32602 GFLOP/s



Run Commands...

(base) ahmadmohammad@MacBook-Pro-101 macOS % brew tap adaptivecpp/homebrew-tap                             
brew install adaptivecpp
acppcc -O3 -std=c++17 matrix_sycl.cpp -o matrix_sycl
./matrix_sycl 

... if you get error ...

- brew info adaptivecpp  
- export PATH="/usr/local/Cellar/adaptivecpp/25.02.0_3/bin:$PATH"
- which acppcc           
- ls /usr/local/Cellar/adaptivecpp/25.02.0_3/bin 
- export PATH="/usr/local/Cellar/adaptivecpp/25.02.0_3/bin:$PATH"
- which syclcc    
- syclcc -O3 -std=c++17 matrix_sycl.cpp -o matrix_sycl  
- ./matrix_sycl    

# working blah blah
