#include <iostream>
#include <cstdlib>
#include "Matrix/Matrix.cu"
#include "RNG.h"
#include <time.h>

#define TILE_WIDTH 30

__global__ void LUDecomposition(float* matrix, unsigned int width, unsigned int block_size)
{
	for (size_t k = 0; k < block_size; ++k)
	{
		unsigned int x = threadIdx.x;
		unsigned int y = threadIdx.y;
		

		if(x>=k && x<width && y==k)
		{	
			double sum = 0.;
			for (size_t p = 0; p < k; ++p)
			{
				sum += matrix[p+k*width] * matrix[x+ p*width];
			}
			matrix[x+ k*width] = matrix[x+ k*width] - sum;	
		}
		__syncthreads();

		if(y>=k+1 && y<width && x==k)
		{
			double sum = 0.;
			for (size_t p = 0; p < k; ++p)
			{
				sum += matrix[p+y*width] * matrix[k+p*width];
			}
			matrix[k+ y*width] = (matrix[k + y*width] - sum) / matrix[k+ k*width];
		}
		__syncthreads();
	}
}


__global__ void MatrixMulKernel(CPAR::Alias_Matrix Md,CPAR::Alias_Matrix Nd,CPAR::Alias_Matrix Pd, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	unsigned int bx = blockIdx.x; int by = blockIdx.y;
	unsigned int tx = threadIdx.x; int ty = threadIdx.y;
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (unsigned int m = 0; m < Width / TILE_WIDTH; ++m) {
		// Coolaborative loading of Md and Nd tiles into shared memory
		Mds[tx][ty] = Md.get(Row,(m*TILE_WIDTH + tx));
		Nds[tx][ty] = Nd.get((m*TILE_WIDTH + ty),Col);
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[tx][k] * Nds[k][ty];
			__syncthreads();
	}
	Pd.get_element(Col, Row) = Pvalue;
}

float* generateMatrix(int size, size_t max_val) {
	RandomNumberGenerator gen((unsigned long long)1, (unsigned long long)max_val);
	float* m;
	m = (float*)malloc(size*size * sizeof(float));

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {
			m[j+i*size] = gen.operator()((int)max_val);
		}
	}

	return m;
}

void run_lu(unsigned int matrix_lenght,unsigned int block_size)
{
	using namespace CPAR;

	unsigned int MATRIX_LENGTH = matrix_lenght;
	unsigned int size = MATRIX_LENGTH * MATRIX_LENGTH;

	float* m2 = generateMatrix(matrix_lenght,50);
	float* c = (float*)malloc(size*sizeof(float*));
	float* device_matrix;// device memory matrix pointer
	int matrix_size = sizeof(float)*size;

	clock_t Time1 = clock();
	unsigned int lu_block_num = 1000;
	dim3 number_of_block_threads(MATRIX_LENGTH/lu_block_num,MATRIX_LENGTH/lu_block_num);
	
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(MATRIX_LENGTH / TILE_WIDTH,MATRIX_LENGTH / TILE_WIDTH);

	cudaMalloc((void **)&device_matrix, matrix_size);
	clock_t Time2 = clock();
	printf("CUDA Memory Allocation And Dim Setup Time:%3.3f\n",(double)(Time2 - Time1) / CLOCKS_PER_SEC);

	Time1 = clock();
	cudaMemcpy(device_matrix, m2, matrix_size, cudaMemcpyHostToDevice);
	Time2 = clock();
	printf("CUDA Data Copy From Host To Device Time:%3.3f\n",(double)(Time2 - Time1) / CLOCKS_PER_SEC);

	Alias_Matrix upper_matrix(device_matrix,MATRIX_LENGTH,MATRIX_LENGTH,block_size,MATRIX_LENGTH-1,0,block_size-1);
	upper_matrix.set_data_prt(device_matrix);

	Alias_Matrix lower_matrix(device_matrix,MATRIX_LENGTH,MATRIX_LENGTH,0,block_size-1,block_size,MATRIX_LENGTH-1);
	lower_matrix.set_data_prt(device_matrix);

	Alias_Matrix res(device_matrix,MATRIX_LENGTH,MATRIX_LENGTH,block_size,MATRIX_LENGTH-1,block_size,MATRIX_LENGTH-1);
	res.set_data_prt(device_matrix);

	Alias_Matrix all(device_matrix,MATRIX_LENGTH,MATRIX_LENGTH,0,MATRIX_LENGTH-1,0,MATRIX_LENGTH-1);
	all.set_data_prt(device_matrix);

	Time1 = clock();
	LUDecomposition<<<lu_block_num, number_of_block_threads>>>(device_matrix, MATRIX_LENGTH,block_size);
	
	cudaError_t err = cudaGetLastError();
	printf("err:%s\n",cudaGetErrorString(err));

	//MatrixMulKernel<<<dimGrid,dimBlock>>>(lower_matrix,upper_matrix,res,MATRIX_LENGTH);
	MatrixMulKernel<<<dimGrid,dimBlock>>>(all,all,all,MATRIX_LENGTH);
	cudaThreadSynchronize();

	cudaError_t err = cudaGetLastError();
	printf("err:%s\n",cudaGetErrorString(err));

	Time2 = clock();
	printf("CUDA LU Decomposition Time:%3.9f\n",(double)(Time2 - Time1) / CLOCKS_PER_SEC);

	Time1 = clock();
	cudaMemcpy(c, device_matrix, matrix_size, cudaMemcpyDeviceToHost);
	Time2 = clock();
	printf("CUDA Data Copy From Device To Host Time:%3.3f\n",(double)(Time2 - Time1) / CLOCKS_PER_SEC);

	cudaFree(device_matrix);
	free(c);
	free(m2);
}

int main(void) {
	printf("128 Blocks:\n");
	for(unsigned int i=1000;i<=6000;i+=1000)
	{
		printf("Size:%d\n",i);
		run_lu(i,128);
		printf("\n");
	}
	printf("End Of 128 Blocks\n\n");
	printf("256 Blocks:\n");
	for(unsigned int i=1000;i<=6000;i+=1000)
	{
		printf("Size:%d\n",i);
		run_lu(i,256);
		printf("\n");
	}
	printf("End Of 256 Blocks\n\n");
	printf("512 Blocks:\n");
	for(unsigned int i=1000;i<=6000;i+=1000)
	{
		printf("Size:%d\n",i);
		run_lu(i,512);
		printf("\n");
	}
	printf("End Of 512 Blocks\n\n");
	return 0;
}
