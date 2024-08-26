# Parallel Matrix Multiplication with NVIDIA CUDA

This repository contains CUDA-based implementations of parallel matrix multiplication, optimized using different tiling strategies to leverage GPU architecture. The project explores how varying the tile size affects performance, memory utilization, and computation speedup.

## Features

- **Parallel Matrix Multiplication**: Efficient multiplication of large matrices using CUDA.
- **Tile Size Optimization**: Experiments with different tile sizes to find the optimal configuration for maximizing GPU performance.
- **Memory Coalescing**: Optimization of memory access patterns to improve bandwidth utilization.
- **Occupancy and Cache Utilization**: Insights into how tile size impacts GPU occupancy and cache usage.

## Files

- **`matrix_mult_ex01.cu`**: CUDA implementation of matrix multiplication with initial tiling strategy.
- **`matrix_mult_ex02.cu`**: Modified version with swapped matrix dimensions to explore alternative memory access patterns.

## Results

The experiments conducted revealed the following key findings:

1. **Optimal Tile Size**: Tile sizes that fit well within shared memory and promote memory coalescing offer significant performance gains.
2. **Impact of Swapping Rows and Columns**: Swapping dimensions in matrix multiplication can affect performance, but the effect varies with tile size and GPU architecture.

Detailed performance measurements, including speedup and execution time for different tile sizes, are documented in the project report pdf.

## How to Run

Note that you need suitable hardware with a CUDA-enabled GPU to run the code.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cuda-parallel-matrix-multiplication.git
   ```
2. Compile the CUDA code:
   ```bash
   nvcc -o matrix_mult matrix_mult_ex01.cu
   ```
3. Run the executable:
   ```bash
   ./matrix_mult
   ```

## Authors

- **Patrick Wissiak**
- **Thomas Rüegg**
