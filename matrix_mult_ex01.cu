#include "cuda_helper.cuh"

// C = A * B

const int C_ROWS = 1000;
const int C_COLS = 2000;
const int A_COLS = 3000;

const int A_ROWS = C_ROWS;
const int B_ROWS = A_COLS;
const int B_COLS = C_COLS;

const int TILE_SIZE = 32;

__global__ void matrixMultKernel(float *A, float *B, float *C)
{
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Bsub[TILE_SIZE][TILE_SIZE];
	int tx = threadIdx.x, ty = threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + tx;
	int row = blockIdx.y * TILE_SIZE + ty;
	int nofTiles = (A_COLS + TILE_SIZE - 1) / TILE_SIZE;
	float sum = 0.0;

	for (int t = 0; t < nofTiles; t++)
	{
		if (row < A_ROWS && t * TILE_SIZE + tx < A_COLS)
		{
			int Aidx = row * A_COLS + t * TILE_SIZE + tx;
			Asub[ty][tx] = A[Aidx];
		}

		if (col < B_COLS && t * TILE_SIZE + ty < B_ROWS)
		{
			int Bidx = (t * TILE_SIZE + ty) * B_COLS + col;
			Bsub[ty][tx] = B[Bidx];
		}
		// ensure all threads will benefit from data in shared memory (which is being prepared)?
		__syncthreads();

		if (row < C_ROWS && col < C_COLS)
		{
			for (int k = 0; k < TILE_SIZE; k++)
			{
				if (t * TILE_SIZE + k < A_COLS)
				{
					sum += Asub[ty][k] * Bsub[k][tx];
				}
			}
		}

		// ensure all calculations are done before writing into C
		__syncthreads();

		if (row < C_ROWS && col < C_COLS)
		{
			C[row * C_COLS + col] = sum;
		}
	}
}

void cudaMatrixMult(float *A, float *B, float *C, int repetitions, bool warmup)
{
	clock_t start = clock();

	for (int i = 0; i < repetitions; i++)
	{
		// TODO: Implement parallel tiled matrix multiplication on CUDA

		// Allocate memory on GPU
		float *d_A, *d_B, *d_C;
		int sizeA = A_ROWS * A_COLS * sizeof(float);
		int sizeB = B_ROWS * B_COLS * sizeof(float);
		int sizeC = C_ROWS * C_COLS * sizeof(float);

		handleCudaError(cudaMalloc(&d_A, sizeA));
		handleCudaError(cudaMalloc(&d_B, sizeB));
		handleCudaError(cudaMalloc(&d_C, sizeC));

		// Copy data to GPU
		handleCudaError(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
		handleCudaError(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));

		// Set up grid and block sizes
		dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
		dim3 blocksPerGrid((C_COLS + TILE_SIZE - 1) / TILE_SIZE, (C_ROWS + TILE_SIZE - 1) / TILE_SIZE);

		printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

		// Do dem calculations
		matrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
		handleCudaError(cudaGetLastError());

		// Copy result from GPU to host
		handleCudaError(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

		// Deallocate memory on GPU
		handleCudaError(cudaFree(d_A));
		handleCudaError(cudaFree(d_B));
		handleCudaError(cudaFree(d_C));
	}
	if (!warmup)
	{
		float diff = float(clock() - start) / (CLOCKS_PER_SEC * repetitions);
		printf("Tile Size: %d\n", TILE_SIZE);
		printf("CUDA: %.3lf seconds\n", diff);
	}
}

void fillRandomArray(float *A, int numElements)
{
	for (int i = 0; i < numElements; i++)
	{
		A[i] = rand() / (float)RAND_MAX;
	}
}

void verifyResults(float *A, float *B, float *C)
{
	printf("Verifying ...");
	for (int row = 0; row < C_ROWS; row++)
	{
		for (int col = 0; col < C_COLS; col++)
		{
			float sum = 0.0;
			for (int k = 0; k < A_COLS; k++)
			{
				sum += A[row * A_COLS + k] * B[k * B_COLS + col];
			}
			if (fabsf(C[row * C_COLS + col] - sum) > 1e-3f)
			{
				fprintf(stderr, "Result verification failed at element %d: %f vs. %f!\n", row, C[row * C_COLS + col], sum);
				exit(EXIT_FAILURE);
			}
		}
	}
	printf(" done\n");
}

void sequentialMatrixMult(float *A, float *B, float *C)
{
	clock_t start = clock();

	for (int row = 0; row < C_ROWS; row++)
	{
		for (int col = 0; col < C_COLS; col++)
		{
			float sum = 0.0;
			for (int k = 0; k < A_COLS; k++)
			{
				sum += A[row * A_COLS + k] * B[k * B_COLS + col];
			}
			C[row * C_COLS + col] = sum;
		}
	}

	float diff = float(clock() - start) / CLOCKS_PER_SEC;
	printf("Sequential: %.3lf seconds\n", diff);
}

int main()
{
	int nofElemA = A_ROWS * A_COLS;
	float *h_A = (float *)malloc(nofElemA * sizeof(float));
	handleAllocationError(h_A);
	fillRandomArray(h_A, nofElemA);

	int nofElemB = B_ROWS * B_COLS;
	float *h_B = (float *)malloc(nofElemB * sizeof(float));
	handleAllocationError(h_B);
	fillRandomArray(h_B, nofElemB);

	int nofElemC = C_ROWS * C_COLS;
	float *h_C = (float *)malloc(nofElemC * sizeof(float));
	handleAllocationError(h_C);

	cudaMatrixMult(h_A, h_B, h_C, 2, true);
	cudaMatrixMult(h_A, h_B, h_C, 4, false);
	verifyResults(h_A, h_B, h_C);

	sequentialMatrixMult(h_A, h_B, h_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
