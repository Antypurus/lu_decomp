#include <stdio.h>

__global__ 
void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width / TILE_WIDTH; ++m) {
		// Coolaborative loading of Md and Nd tiles into shared memory
		Mds[tx][ty] = Md[(m*TILE_WIDTH + tx)*Width + Row];
		Nds[tx][ty] = Nd[Col*Width + (m*TILE_WIDTH + ty)];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[tx][k] * Nds[k][ty];
		__synchthreads();

	}
	Pd[Row*Width + Col] = Pvalue;
}

__global__ 
void SequentialLUDecomposition(float *Md, float *Nd, float *Pd, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width / TILE_WIDTH; ++m)
	{
		// Coolaborative loading of Md and Nd tiles into shared memory
		Mds[tx][ty] = Md[(m * TILE_WIDTH + tx) * Width + Row];
		Nds[tx][ty] = Nd[Col * Width + (m * TILE_WIDTH + ty)];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[tx][k] * Nds[k][ty];
		__synchthreads();
	}
	Pd[Row * Width + Col] = Pvalue;
}

int main(void)
{
	int N = 1 << 20;
	float *x, *y, *d_x, *d_y;
	x = (float *)malloc(N * sizeof(float));
	y = (float *)malloc(N * sizeof(float));

	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));
	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}