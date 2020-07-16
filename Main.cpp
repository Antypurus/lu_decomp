#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <cstdlib>

#include "RNG.h"
#include "Matrix/Matrix.hpp"

using namespace std;

double** generateMatrix(int size, size_t max_val) {
	RandomNumberGenerator gen((unsigned long long)1, (unsigned long long)max_val);
	double** m;
	m = (double**)malloc(size * sizeof(*m));

	for (int i = 0; i < size; i++) {
		m[i] = (double*)malloc(size * sizeof(*m[i]));

		for (int j = 0; j < size; j++) {
			m[i][j] = gen.operator()((int)max_val);
		}
	}

	return m;
}

void runSequentialForSize(size_t size)
{
	clock_t Time1 = clock();
	double** m2 = generateMatrix(size, size);

	CPAR::Matrix m(m2, size, size);
	auto ret = CPAR::Matrix::sequential_LU_decomposition(m);
	clock_t Time2 = clock();

	cout << "Matrix with size " << size << endl;
	printf("Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
}

void testLUSequential()
{
	cout << endl << "Starting LU Sequential" << endl;

	for (size_t i = 1000; i <= 6000; i += 1000)
	{
		runSequentialForSize(i);
		cout << endl << endl;
	}

	cout << "End LU Sequential" << endl;
}

void runParallelForSize(size_t size)
{
	clock_t Time1 = clock();
	double** m2 = generateMatrix(size, size);

	CPAR::Matrix m(m2, size, size);
	auto ret = CPAR::Matrix::parallel_LU_decomposition(m);
	clock_t Time2 = clock();

	cout << "Matrix with size " << size << endl;
	printf("Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
}

void testLUParallel()
{
	cout << endl << "Starting LU Parallel" << endl;

	for (size_t i = 1000; i <= 6000; i += 1000)
	{
		runParallelForSize(i);
		cout << endl << endl;
	}

	cout << "End LU Parallel" << endl;
}

void runBlockForSize(size_t size, size_t block_size)
{
	clock_t Time1 = clock();
	double** m2 = generateMatrix(size, size);

	CPAR::Matrix m(m2, size, size);
	auto ret = CPAR::Matrix::block_LU_decomposition(m, block_size);
	clock_t Time2 = clock();

	cout << "Matrix with size " << size << endl;
	printf("Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
}

void testLUWithBlocks()
{
	cout << endl << "Starting LU Blocks" << endl;

	for (size_t i = 1000; i <= 6000; i += 1000)
	{
		runBlockForSize(i, 128);
		cout << endl << endl;
	}
	cout << "End of block 128" << endl;

	for (size_t i = 1000; i <= 6000; i += 1000)
	{
		runBlockForSize(i, 256);
		cout << endl << endl;
	}
	cout << "End of block 256" << endl;

	for (size_t i = 1000; i <= 6000; i += 1000)
	{
		runBlockForSize(i, 512);
		cout << endl << endl;
	}
	cout << "End of block 512" << endl;

	cout << "End LU Blocks" << endl;
}

int main()
{
	testLUSequential();
	testLUWithBlocks();
	testLUParallel();
	getchar();
}
